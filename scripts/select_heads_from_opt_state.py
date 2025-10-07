import argparse
import os
import sys
from typing import Dict, List, Tuple
import dataclasses as _dc

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import flax.nnx as nnx  # noqa: E402
import optax  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402
from openpi.training import sharding as _sharding  # noqa: E402
from openpi.training import utils as training_utils  # noqa: E402


NUM_LAYERS = 18
HEADS_PER_LAYER = 8


DEFAULT_WEIGHT_PATHS: Dict[str, str] = {
    "query": "PaliGemma/llm/layers/attn/q_einsum",
    "key_value": "PaliGemma/llm/layers/attn/kv_einsum",
    "output": "PaliGemma/llm/layers/attn/attn_vec_einsum",
}


def _path_to_str(path) -> str:
    parts = []
    for p in path:
        s = str(p)
        if s.startswith("['") and s.endswith("']"):
            parts.append(s[2:-2])
        else:
            parts.append(s)
    return "/".join(parts)


def _create_masked_tx_for_head_tuning(cfg: _config.TrainConfig, params_structure: nnx.State) -> optax.GradientTransformation:
    base = _optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)
    mask_arrays = _optimizer._create_head_tuning_mask(
        params_structure.filter(cfg.trainable_filter),
        cfg.optimizer.trainable_head_indices,
        freeze_kv=getattr(cfg.optimizer, "freeze_kv", False),
        only_attention=getattr(cfg.optimizer, "only_attention", False),
        freeze_mlp=getattr(cfg.optimizer, "freeze_mlp", False),
    )
    mask_dict = nnx.State(mask_arrays).to_pure_dict()

    def masked_update(updates, state, params=None):
        updates_dict = updates.to_pure_dict() if hasattr(updates, "to_pure_dict") else updates
        masked = jax.tree_util.tree_map(lambda u, m: u * m, updates_dict, mask_dict)
        return base.update(masked, state, params)

    return optax.GradientTransformation(base.init, masked_update)


def _build_state_shape(cfg: _config.TrainConfig) -> training_utils.TrainState:
    rng = jax.random.key(cfg.seed)
    rng, model_rng = jax.random.split(rng)
    model = cfg.model.create(model_rng)
    params = nnx.state(model)

    if isinstance(cfg.optimizer, _optimizer.AdamWForHeadTuning):
        tx = _create_masked_tx_for_head_tuning(cfg, jax.eval_shape(lambda: params))
        opt_init_arg = params.filter(cfg.trainable_filter).to_pure_dict()
    else:
        tx = _optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)
        opt_init_arg = params.filter(cfg.trainable_filter)

    opt_state = tx.init(opt_init_arg)

    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=opt_state,
        ema_decay=cfg.ema_decay,
        ema_params=None if cfg.ema_decay is None else params,
    )


def _collect_adam_nu(opt_state) -> List:
    """Collect all Adam second moment (nu) pytrees from an optax chain state.

    Robust to:
    - tuples/lists from optax.chain
    - dataclass states (ScaleByAdamState has mu, nu, count)
    - dict/FrozenDict-wrapped states after checkpoint restore
    """
    nu_trees: List = []

    # Lazy import to avoid hard dependency if flax not present in some envs
    try:
        from flax.core.frozen_dict import FrozenDict  # type: ignore
    except Exception:  # pragma: no cover
        FrozenDict = tuple()  # sentinel non-matching type

    def _walk(node):
        if node is None:
            return
        # Direct hit: any object that exposes mu/nu attributes
        try:
            if hasattr(node, "mu") and hasattr(node, "nu") and getattr(node, "nu") is not None:
                nu_trees.append(getattr(node, "nu"))
                return
            # Fallback by class name and tuple-like layout (count, mu, nu)
            cls_name = type(node).__name__
            if "ScaleByAdamState" in cls_name:
                # try attribute first
                nu_attr = getattr(node, "nu", None)
                if nu_attr is not None:
                    nu_trees.append(nu_attr)
                    return
                # try tuple indexing
                try:
                    if hasattr(node, "__len__") and len(node) >= 3:
                        nu_trees.append(node[2])
                        return
                except Exception:
                    pass
        except Exception:
            pass
        # Handle list/tuple (optax.chain packs states here)
        if isinstance(node, (list, tuple)):
            for x in node:
                _walk(x)
            return
        # Handle FrozenDict/dict
        if isinstance(node, dict) or (hasattr(FrozenDict, "__class__") and isinstance(node, FrozenDict)):
            # Direct hit if dict contains mu/nu
            if isinstance(node, dict) and ("nu" in node and "mu" in node):
                nu_trees.append(node["nu"])  # type: ignore[index]
            # Recurse values
            try:
                values = node.values() if isinstance(node, dict) else node.unfreeze().values()  # type: ignore[attr-defined]
                for v in values:
                    _walk(v)
            except Exception:
                pass
            return
        # Handle dataclass-like states
        fields = getattr(node, "__dataclass_fields__", None)
        if fields or _dc.is_dataclass(node):
            # Direct hit on ScaleByAdamState
            try:
                if hasattr(node, "nu") and hasattr(node, "mu") and getattr(node, "nu") is not None:
                    nu_trees.append(getattr(node, "nu"))
            except Exception:
                pass
            # Try via asdict to fetch 'nu' even if attributes are not accessible
            try:
                d = _dc.asdict(node)
                for key in ("nu", "v"):  # support alt naming
                    if key in d and d[key] is not None:
                        nu_trees.append(d[key])
                for v in d.values():
                    _walk(v)
                return
            except Exception:
                # Fallback: walk attribute fields by name
                for name in fields or {}:
                    try:
                        _walk(getattr(node, name))
                    except Exception:
                        pass
                return
        # Fallback: no-op for scalars/arrays/unknown objects
        return

    _walk(opt_state)
    return nu_trees


def _debug_print_opt_state(opt_state, *, max_depth: int = 5) -> None:
    """Pretty-print the optimizer state structure with types and array shapes.

    This helps diagnose cases where Adam's (mu, nu) are nested/packaged in unusual ways.
    """
    try:
        from flax.core.frozen_dict import FrozenDict  # type: ignore
    except Exception:  # pragma: no cover
        class FrozenDict(dict):  # type: ignore
            pass

    def _is_array(x) -> bool:
        return hasattr(x, "shape") and hasattr(x, "dtype")

    def _arr_str(x) -> str:
        try:
            return f"array{tuple(x.shape)}@{getattr(x, 'dtype', 'unk')}"
        except Exception:
            return "array"

    def _indent(n: int) -> str:
        return "  " * n

    def _walk(node, depth: int, name: str | None = None):
        prefix = _indent(depth)
        label = f"{name}: " if name else ""

        if depth > max_depth:
            print(f"{prefix}{label}... (max_depth reached)")
            return

        if node is None:
            print(f"{prefix}{label}None")
            return

        # list/tuple
        if isinstance(node, (list, tuple)):
            print(f"{prefix}{label}{type(node).__name__}(len={len(node)})")
            for i, v in enumerate(node):
                _walk(v, depth + 1, name=f"[{i}]")
            return

        # dict or FrozenDict
        if isinstance(node, (dict, FrozenDict)):
            keys = list(node.keys()) if isinstance(node, dict) else list(node.unfreeze().keys())
            print(f"{prefix}{label}{type(node).__name__}(keys={keys})")
            values = node.values() if isinstance(node, dict) else node.unfreeze().values()
            for k, v in zip(keys, values):
                _walk(v, depth + 1, name=str(k))
            return

        # dataclass-like
        fields = getattr(node, "__dataclass_fields__", None)
        if fields:
            cls = type(node).__name__
            has_mu = hasattr(node, "mu")
            has_nu = hasattr(node, "nu")
            mu_str = _arr_str(getattr(node, "mu")) if has_mu and _is_array(getattr(node, "mu")) else str(has_mu)
            nu_str = _arr_str(getattr(node, "nu")) if has_nu and _is_array(getattr(node, "nu")) else str(has_nu)
            print(f"{prefix}{label}{cls}(mu={mu_str}, nu={nu_str})")
            for fname in fields:
                try:
                    _walk(getattr(node, fname), depth + 1, name=fname)
                except Exception:
                    print(f"{prefix}  {fname}: <unavailable>")
            return

        # arrays
        if _is_array(node):
            print(f"{prefix}{label}{_arr_str(node)}")
            return

        # fallback scalar/object
        print(f"{prefix}{label}{type(node).__name__}")

    print("=== Optimizer opt_state structure (preview) ===")
    _walk(opt_state, depth=0, name="opt_state")


def _aggregate_head_scores_from_nu(
    params_tree,
    nu_tree,
    weight_paths: Dict[str, str],
    *,
    include_kv: bool = True,
    score_norm: str = "none",
) -> np.ndarray:
    # Normalize to pure dicts if wrapped (e.g., nnx.State)
    if hasattr(params_tree, "to_pure_dict"):
        try:
            params_tree = params_tree.to_pure_dict()
        except Exception:
            pass
    if hasattr(nu_tree, "to_pure_dict"):
        try:
            nu_tree = nu_tree.to_pure_dict()
        except Exception:
            pass

    # Build mapping by flattening param tree and nu tree in lockstep
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params_tree)
    paths = [p for (p, _) in path_leaves]
    flat_p = [leaf for (_, leaf) in path_leaves]
    flat_nu, _ = jax.tree_util.tree_flatten(nu_tree)
    if len(flat_p) != len(flat_nu):
        raise RuntimeError(
            f"Param tree and nu tree size mismatch: params={len(flat_p)} vs nu={len(flat_nu)}"
        )

    # Scores per (layer, head)
    scores = {
        "query": np.zeros((NUM_LAYERS, HEADS_PER_LAYER), dtype=np.float64),
        "key_value": np.zeros((NUM_LAYERS, HEADS_PER_LAYER), dtype=np.float64),
        "output": np.zeros((NUM_LAYERS, HEADS_PER_LAYER), dtype=np.float64),
    }

    for path, p_leaf, nu_leaf in zip(paths, flat_p, flat_nu):
        if not hasattr(p_leaf, "shape") or p_leaf is None:
            continue
        path_str = _path_to_str(path)
        # We only care about attention weights and LoRA weights
        is_attn = "PaliGemma/llm/layers/attn" in path_str
        ends_with = ["/w", "/lora_a", "/lora_b"]
        if not is_attn or not any(path_str.endswith(suf) for suf in ends_with):
            continue

        # Identify which weight type
        which = None
        if weight_paths["query"] in path_str:
            which = "query"
        elif weight_paths["key_value"] in path_str:
            which = "key_value"
        elif weight_paths["output"] in path_str:
            which = "output"
        else:
            continue

        # Accept either raw arrays or objects that wrap arrays with `.value`
        try:
            nu_arr = np.asarray(getattr(nu_leaf, "value", nu_leaf), dtype=np.float32)
        except Exception:
            # Skip non-array leaves
            continue
        if which in ("query", "output"):
            # expected shape: (layers, heads, ...)
            if nu_arr.ndim < 2:
                continue
            L = min(NUM_LAYERS, nu_arr.shape[0])
            H = min(HEADS_PER_LAYER, nu_arr.shape[1])
            # per-head score = sqrt(nu) aggregated over remaining dims
            for l in range(L):
                for h in range(H):
                    # Fisher-like magnitude
                    val = float(np.sqrt(np.mean(nu_arr[l, h].astype(np.float32) + 1e-12)))
                    scores[which][l, h] += val
        else:  # key_value
            if not include_kv:
                continue
            # kv shape often: (layers, 2, num_kv_heads, ...). Most cases num_kv_heads==1
            if nu_arr.ndim < 2:
                continue
            L = min(NUM_LAYERS, nu_arr.shape[0])
            # We treat up to 2 heads for KV to align with selection API
            kv_heads_dim = 2 if nu_arr.shape[1] >= 2 else 1
            for l in range(L):
                for h in range(kv_heads_dim):
                    # If no explicit per-head dim beyond the 2 (q/k), just aggregate that slice
                    slicer = (l, h)
                    val = float(np.sqrt(np.mean(nu_arr[slicer].astype(np.float32) + 1e-12)))
                    scores[which][l, h] += val

    # Aggregate Q + O (+ KV for head<2)
    agg = np.zeros((NUM_LAYERS, HEADS_PER_LAYER), dtype=np.float64)
    for l in range(NUM_LAYERS):
        for h in range(HEADS_PER_LAYER):
            s = scores["query"][l, h] + scores["output"][l, h]
            if include_kv and h < 2:
                s += scores["key_value"][l, h]
            agg[l, h] = s

    # Optional per-layer normalization
    if score_norm == "layer_zscore":
        means = agg.mean(axis=1, keepdims=True)
        stds = agg.std(axis=1, keepdims=True)
        agg = (agg - means) / (stds + 1e-12)
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Select heads using optimizer moments (nu) from checkpoints")
    p.add_argument("config_name", type=str, help="Train config name, e.g., All_heads_LoRA")
    p.add_argument("--exp-name", type=str, required=True, help="Experiment name used in training")
    p.add_argument("--step", type=int, default=None, help="Checkpoint step to load (default: latest)")
    p.add_argument("--num-heads", type=int, default=20, help="Top-K heads to select")
    p.add_argument("--freeze-kv", action="store_true", help="Exclude KV contributions from scoring (freeze KV)")
    p.add_argument(
        "--score-norm",
        type=str,
        choices=["none", "layer_zscore"],
        default="layer_zscore",
        help="Score normalization: none or layer_zscore (default)",
    )
    p.add_argument(
        "--print-depth",
        type=int,
        default=10,
        help="Max depth when printing optimizer state structure (for debugging)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _config.get_config(args.config_name)
    cfg = _config.TrainConfig(**{**cfg.__dict__, "exp_name": args.exp_name, "resume": True, "wandb_enabled": False})

    # Build state shape compatible with saved opt_state
    state_shape = _build_state_shape(cfg)

    # Create checkpoint manager and restore specified step (or latest)
    ckpt_dir = cfg.checkpoint_dir
    mngr, _ = _checkpoints.initialize_checkpoint_dir(ckpt_dir, keep_period=cfg.keep_period, overwrite=False, resume=True)
    state = _checkpoints.restore_state(mngr, state_shape, data_loader=None, step=args.step)

    # Always print a structural preview to help debugging
    _debug_print_opt_state(state.opt_state, max_depth=args.print_depth)

    # Collect Adam second moment trees (there may be multiple in a chained transform)
    nu_trees = _collect_adam_nu(state.opt_state)
    if not nu_trees:
        print("[WARN] Failed to locate Adam 'nu' (second moment) in optimizer state. See structure above.")
        raise RuntimeError("Failed to locate Adam 'nu' (second moment) in optimizer state.")

    # Use the first found nu tree; if multiple, sum their contributions
    params_trainable = state.params.filter(cfg.trainable_filter)
    agg_scores = np.zeros((NUM_LAYERS, HEADS_PER_LAYER), dtype=np.float64)
    for nu_tree in nu_trees:
        # For head-tuning masked flow, opt_state was built from pure dict of trainable params
        # Align structure using the trainable param view
        try:
            agg_scores += _aggregate_head_scores_from_nu(
                params_trainable.to_pure_dict(),
                nu_tree,
                DEFAULT_WEIGHT_PATHS,
                include_kv=not args.freeze_kv,
                score_norm=args.score_norm,
            )
        except Exception:
            agg_scores += _aggregate_head_scores_from_nu(
                params_trainable,
                nu_tree,
                DEFAULT_WEIGHT_PATHS,
                include_kv=not args.freeze_kv,
                score_norm=args.score_norm,
            )

    # Rank heads
    rankings: List[Tuple[float, int, int]] = []
    for l in range(NUM_LAYERS):
        for h in range(HEADS_PER_LAYER):
            rankings.append((agg_scores[l, h], l, h))
    rankings.sort(reverse=True)

    selected: List[Tuple[int, int]] = []
    for i in range(min(args.num_heads, len(rankings))):
        _, l, h = rankings[i]
        selected.append((l, h))

    literal = ", ".join(f"({l}, {h})" for l, h in selected)
    print("Top heads by Adam second-moment (nu) proxy:")
    for i, (score, l, h) in enumerate(rankings[:args.num_heads], 1):
        print(f"  Rank {i:2d}  L{l}.H{h}  score={score:.6e}")
    print("\nSelected heads (copy-paste):")
    print(f"trainable_head_indices=[{literal}]")


if __name__ == "__main__":
    main()


