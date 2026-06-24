"""
KNN regression model: head selection, training, and inference orchestration.

Provides :class:`KnnRegModel` (the trained model dataclass) and
:func:`fit_knn_reg_with_heads` (the main entry point for head selection +
model fitting).  Only two selection modes are supported: ``"topk"`` and
``"best_add"``.

H5 layout assumed by :func:`fit_knn_reg_with_heads`
----------------------------------------------------
Each episode group contains frame sub-groups; each frame group must have::

    first_action_token_attn  : (18, 8, 256)   — raw attention features
    action_label (preferred) : (A,)            — action vector

See :mod:`openpi.head_tuning.knn.data` for the full fallback chain.

Unit modes
----------
``unit_mode="head"``   — 144 attention heads, each 256-D.
``unit_mode="layer"``  — 18 layers; each layer's 8 heads are concatenated
                          to give one 2048-D unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .data import HeadPreprocessor, build_dataset
from .eval import evaluate_leave_one_episode_out
from .inference import build_feat_subset
from .selection import greedy_forward_select, rank_single_heads, simple_topk_select


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_LAYERS = 18
_HEADS_PER_LAYER_HEAD_MODE = 8
_HEADS_PER_LAYER_LAYER_MODE = 1   # one "unit" per layer


def _unit_mode_params(unit_mode: str) -> Tuple[int, int, int, int]:
    """Return (num_layers, heads_per_layer, H, d_per_head) for the given unit_mode."""
    if unit_mode == "head":
        num_layers = _NUM_LAYERS
        heads_per_layer = _HEADS_PER_LAYER_HEAD_MODE
        H = num_layers * heads_per_layer          # 144
        d_per_head = 256
    elif unit_mode == "layer":
        num_layers = _NUM_LAYERS
        heads_per_layer = _HEADS_PER_LAYER_LAYER_MODE
        H = num_layers * heads_per_layer          # 18
        d_per_head = _HEADS_PER_LAYER_HEAD_MODE * 256   # 2048
    else:
        raise ValueError(f"unit_mode must be 'head' or 'layer', got {unit_mode!r}")
    return num_layers, heads_per_layer, H, d_per_head


# ---------------------------------------------------------------------------
# Excluded-head normalisation helpers
# ---------------------------------------------------------------------------

def _to_flat_head_index(item, H: int, num_layers: int, heads_per_layer: int) -> Optional[int]:
    """
    Convert a head specification to a flat head index in ``[0, H)``.

    Accepted formats: ``int`` (flat index), ``(layer, head)`` tuple/list,
    ``{"layer": l, "head": h}`` dict, ``"layer,head"`` or ``"flat_int"`` string.
    Returns ``None`` if the specification is invalid.
    """
    try:
        if isinstance(item, int):
            return int(item) if 0 <= int(item) < H else None
        if isinstance(item, (tuple, list)) and len(item) == 2:
            layer, head = int(item[0]), int(item[1])
        elif isinstance(item, dict):
            if "layer" in item and "head" in item:
                layer, head = int(item["layer"]), int(item["head"])
            else:
                layer = int(item.get("l", item.get("layer", -1)))
                head = int(item.get("h", item.get("head", -1)))
        elif isinstance(item, str):
            s = item.strip().replace("(", "").replace(")", "")
            if "," in s:
                parts = s.split(",")
                layer, head = int(parts[0].strip()), int(parts[1].strip())
            else:
                val = int(s)
                return val if 0 <= val < H else None
        else:
            return None
        if 0 <= layer < num_layers and 0 <= head < heads_per_layer:
            return layer * heads_per_layer + head
        return None
    except Exception:
        return None


def _normalize_excluded_heads(
    excluded_heads_input,
    H: int,
    num_layers: int,
    heads_per_layer: int,
) -> Set[int]:
    """Convert an arbitrary excluded-head specification list to a set of flat indices."""
    if excluded_heads_input is None:
        return set()
    result: Set[int] = set()
    for item in excluded_heads_input:
        flat = _to_flat_head_index(item, H, num_layers, heads_per_layer)
        if flat is not None:
            result.add(flat)
    return result


# ---------------------------------------------------------------------------
# Model dataclass
# ---------------------------------------------------------------------------

@dataclass
class KnnRegModel:
    """Trained KNN regression model.

    Attributes
    ----------
    heads:
        Flat indices of the selected attention-head units.
    k:
        Number of nearest neighbours used for prediction.
    metric:
        Distance metric for nearest-neighbour lookup (``"cosine"`` or
        ``"euclidean"``).
    d_per_head:
        Feature dimensionality contributed by each head unit (256 for head
        mode, 2048 for layer mode).
    preproc:
        Per-head preprocessors (identity pass-through, kept for API
        compatibility with inference helpers).
    X_bank:
        Feature bank used for retrieval, shape ``(N, |heads| * d_per_head)``.
    Y_bank:
        Action bank used for retrieval, shape ``(N, A)``.
    """

    heads: List[int]
    k: int
    metric: str
    d_per_head: int
    preproc: List[HeadPreprocessor]
    X_bank: np.ndarray   # (N, |heads| * d_per_head)
    Y_bank: np.ndarray   # (N, A)


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_knn_reg_with_heads(
    attn_h5: str,
    episodes: List[str],
    *,
    selection_mode: str = "topk",
    target_heads: int,
    unit_mode: str,
    k_grid: List[int],
    metric: str = "cosine",
    temp_excl_w: int,
    excluded_heads: Optional[List] = None,
) -> Tuple[KnnRegModel, Dict]:
    """Select attention heads and fit a KNN regression model.

    Parameters
    ----------
    attn_h5:
        Path to the attention H5 file.
    episodes:
        List of episode keys to use for training (format ``"task/episode"``).
    selection_mode:
        Head selection strategy: ``"topk"`` or ``"best_add"``.
    target_heads:
        Number of head units to select.
    unit_mode:
        Feature granularity: ``"head"`` (144 units × 256-D) or
        ``"layer"`` (18 units × 2048-D).
    k_grid:
        Candidate values of *k* evaluated during cross-validation.
    metric:
        Distance metric for LOEO cross-validation and inference
        (``"cosine"`` or ``"euclidean"``).
    temp_excl_w:
        Temporal-exclusion window (±W frames) for single-episode leave-one-
        frame-out evaluation.
    excluded_heads:
        Optional list of head units to exclude from the final selection.
        Accepts flat indices, ``(layer, head)`` pairs, dicts, or strings.

    Returns
    -------
    model:
        Trained :class:`KnnRegModel` ready for inference.
    info:
        Dictionary with keys:

        - ``"selected_heads"`` — list of selected flat head indices
        - ``"best_k"`` — best *k* found by cross-validation
        - ``"cv_mse"`` — LOEO MSE for the selected heads at ``best_k``
        - ``"per_k_mse"`` — ``{k: mse}`` over the full k-grid

    Raises
    ------
    ValueError
        If *selection_mode* is not ``"topk"`` or ``"best_add"``.
    """
    if selection_mode not in {"topk", "best_add"}:
        raise ValueError(
            f"selection_mode must be 'topk' or 'best_add', got {selection_mode!r}"
        )

    # ---- Resolve unit-mode parameters ----
    num_layers, heads_per_layer, H, d_per_head_expected = _unit_mode_params(unit_mode)

    # ---- Normalise excluded heads ----
    excluded_set = _normalize_excluded_heads(
        excluded_heads, H, num_layers, heads_per_layer
    )

    # ---- Build dataset ----
    pre = build_dataset(
        attn_h5,
        episodes,
        num_layers=num_layers,
        heads_per_layer=heads_per_layer,
        d_head=d_per_head_expected,
    )
    d_per_head = pre.Xh_red.shape[-1]   # resolved from actual data

    # ---- Step 1: rank single heads per k ----
    rankings_per_k = rank_single_heads(
        pre.features,
        pre.actions,
        pre.episode_ids,
        pre.frame_ids,
        k_grid,
        metric,
        temp_excl_w,
        H=H,
        print_rankings=True,
        unit_mode=unit_mode,
    )

    # ---- Step 2: select heads by mode ----
    if selection_mode == "topk":
        selected_heads, best_k, cv_mse, per_k_mse = simple_topk_select(
            pre.features,
            pre.actions,
            pre.episode_ids,
            pre.frame_ids,
            rankings_per_k,
            k_grid,
            metric,
            temp_excl_w,
            target_heads,
        )
    else:  # best_add
        selected_heads, best_k, cv_mse, per_k_mse = greedy_forward_select(
            pre.features,
            pre.actions,
            pre.episode_ids,
            pre.frame_ids,
            rankings_per_k,
            k_grid,
            metric,
            temp_excl_w,
            target_heads,
        )

    # ---- Step 3: enforce excluded heads and refill ----
    effective_target = min(target_heads, max(0, H - len(excluded_set)))
    selected_heads = _enforce_excluded_and_fill(
        selected_heads, effective_target, best_k, rankings_per_k, excluded_set
    )

    # ---- Step 4: recompute per-k MSE with the final head set ----
    per_k_mse = {}
    for k in k_grid:
        mse_k = evaluate_leave_one_episode_out(
            pre.features,
            pre.actions,
            pre.episode_ids,
            pre.frame_ids,
            selected_heads,
            k,
            metric,
            temp_excl_w,
        )
        per_k_mse[int(k)] = float(mse_k)
    cv_mse = float(per_k_mse[int(best_k)])

    # ---- Step 5: build feature bank ----
    X_bank = build_feat_subset(pre.Xh_red, selected_heads).astype(np.float32)

    # ---- Assemble model and info ----
    model = KnnRegModel(
        heads=selected_heads,
        k=best_k,
        metric=metric,
        d_per_head=d_per_head,
        preproc=pre.preproc,
        X_bank=X_bank,
        Y_bank=pre.Y.astype(np.float32),
    )

    # ---- Console summary ----
    unit_word = "layers" if unit_mode == "layer" else "heads"
    per_k_str = ", ".join(f"k={k}: {v:.6f}" for k, v in sorted(per_k_mse.items()))
    print(f"Per-k MSE -> {per_k_str}")
    print("================ Train Config ================")
    print(f"Unit mode            : {unit_mode} (H={H}, d_per_unit={d_per_head})")
    print(f"Selection mode       : {selection_mode}")
    print(f"Target {unit_word:14s}: {target_heads}")
    print(f"K grid               : {k_grid}")
    print(f"Best k               : {best_k}")
    print(f"Metric               : {metric}")
    print(f"Temporal excl window : {temp_excl_w}")
    if excluded_set:
        print(f"Excluded {unit_word:11s}: {sorted(excluded_set)}")
    print(f"Selected {unit_word} ({len(selected_heads)}): {selected_heads}")
    if unit_mode == "layer":
        equiv = [l * 8 + h for l in selected_heads for h in range(8)]
        print(f"  -> equivalent head IDs ({len(equiv)}): {equiv}")
    print(f"LOEO-CV MSE          : {cv_mse:.6f}")
    print("==============================================")

    info: Dict = {
        "selected_heads": selected_heads,
        "best_k": best_k,
        "cv_mse": cv_mse,
        "per_k_mse": per_k_mse,
        "selection_mode": selection_mode,
        "rankings_per_k": rankings_per_k,
        "episodes": pre.episodes,
        "d_per_head": d_per_head,
        "metric": metric,
        "k_grid": list(k_grid),
        "temp_excl_w": temp_excl_w,
        "target_heads": target_heads,
        "excluded_heads": sorted(excluded_set),
        "unit_mode": unit_mode,
    }
    return model, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enforce_excluded_and_fill(
    cur_sel: List[int],
    target_k: int,
    best_k: int,
    rankings_per_k: Dict,
    excluded_set: Set[int],
) -> List[int]:
    """Remove excluded heads from *cur_sel* and top-up from the ranking pool."""
    filtered: List[int] = []
    seen: Set[int] = set()
    for h in cur_sel:
        if h in excluded_set or h in seen:
            continue
        filtered.append(h)
        seen.add(h)
    # Top-up from the best-k ranking
    for _mse, h in rankings_per_k.get(int(best_k), []):
        if len(filtered) >= target_k:
            break
        if h in excluded_set or h in seen:
            continue
        filtered.append(h)
        seen.add(h)
    return filtered
