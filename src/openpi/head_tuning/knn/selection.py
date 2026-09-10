"""
KNN head selection algorithms.

Provides single-head ranking, greedy forward selection, and simple top-k selection.
All methods evaluate candidate head subsets via leave-one-episode-out (LOEO) cross-validation
with cosine (default) or euclidean distance — no metric-learning transforms.

The deprecated functions `reinforce_select_heads`, `learn_head_weights`,
`compute_loeo_mse_torch`, and `compute_loeo_mse_torch_with_metric` were
removed as part of the metric-machinery cleanup (Tasks 4–5).
"""

import math
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from .eval import evaluate_leave_one_episode_out


def rank_single_heads_per_k(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    k: int,
    metric: str,
    temp_excl_w: int,
    H: int,
    print_all_scores: bool = True,
    unit_mode: str = "head",
) -> List[Tuple[float, int]]:
    """Rank single heads for a specific k value. Returns [(mse, head_id), ...].

    unit_mode: "head" prints per-head labels (LxHy); "layer" prints per-layer labels.
    """
    scores: List[Tuple[float, int]] = []
    is_layer = unit_mode == "layer"
    unit_word = "Layer" if is_layer else "Head"

    def _label(idx: int) -> str:
        if is_layer:
            return f"Layer {idx:2d}"
        return f"Head {idx:3d} (L{idx//8:2d}H{idx%8})"

    print(f"\n=== Single {unit_word} Ranking for k={k} ===")
    for h in range(H):
        mse = evaluate_leave_one_episode_out(
            per_head_features, actions, episode_ids, frame_ids, [h], k, metric, temp_excl_w,
        )
        scores.append((float(mse), h))

        if print_all_scores:
            print(f"{_label(h)}: MSE = {mse:.6f}")

    scores.sort(key=lambda x: x[0])

    if print_all_scores:
        top_n = min(20, len(scores))
        print(f"\n=== Top {top_n} {unit_word}s for k={k} ===")
        for i, (mse, h) in enumerate(scores[:top_n]):
            print(f"Rank {i+1:2d}: {_label(h)}, MSE = {mse:.6f}")

        bot_n = min(20, len(scores))
        print(f"\n=== Bottom {bot_n} {unit_word}s for k={k} ===")
        for i, (mse, h) in enumerate(scores[-bot_n:]):
            rank = len(scores) - bot_n + i + 1
            print(f"Rank {rank:2d}: {_label(h)}, MSE = {mse:.6f}")

        # Print MSE statistics
        mse_values = [mse for mse, _ in scores]
        best_mse = min(mse_values)
        worst_mse = max(mse_values)
        median_mse = np.median(mse_values)
        mean_mse = np.mean(mse_values)

        print(f"\n=== MSE Statistics for k={k} ===")
        print(f"Best MSE:   {best_mse:.6f}")
        print(f"Worst MSE:  {worst_mse:.6f}")
        print(f"Median MSE: {median_mse:.6f}")
        print(f"Mean MSE:   {mean_mse:.6f}")
        print(f"MSE Range:  {worst_mse - best_mse:.6f} ({((worst_mse - best_mse) / best_mse * 100):.1f}% of best)")
        print("=" * 50)

    return scores


def rank_single_heads(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    k_grid: Iterable[int],
    metric: str,
    temp_excl_w: int,
    H: int,
    print_rankings: bool = True,
    unit_mode: str = "head",
) -> dict:
    """Return rankings per k: {k: [(mse, head_id), ...]}"""
    rankings_per_k = {}
    desc = "Single-layer ranking per k" if unit_mode == "layer" else "Single-head ranking per k"
    for k in tqdm(k_grid, desc=desc):
        rankings_per_k[k] = rank_single_heads_per_k(
            per_head_features, actions, episode_ids, frame_ids, k, metric, temp_excl_w, H,
            print_all_scores=print_rankings, unit_mode=unit_mode,
        )
    return rankings_per_k


def simple_topk_select(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    rankings_per_k: dict,  # {k: [(mse, head_id), ...]}
    k_grid: Iterable[int],
    metric: str,
    temp_excl_w: int,
    target_heads: int,
):
    """For each k, select top heads and evaluate. Return the best (k, heads, mse) combination."""
    best_overall_mse, best_overall_k, best_overall_heads = math.inf, None, None
    per_k_results = {}

    for k in k_grid:
        # Get top heads for this k
        head_ranking = rankings_per_k[k]  # [(mse, head_id), ...]
        sel_heads = [h for _, h in head_ranking[:target_heads]]

        mse = evaluate_leave_one_episode_out(
            per_head_features, actions, episode_ids, frame_ids, sel_heads, k, metric, temp_excl_w,
        )

        per_k_results[int(k)] = float(mse)
        if mse < best_overall_mse:
            best_overall_mse, best_overall_k, best_overall_heads = mse, k, sel_heads

    return best_overall_heads, int(best_overall_k), float(best_overall_mse), per_k_results


def greedy_forward_select(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    rankings_per_k: dict,  # {k: [(mse, head_id), ...]}
    k_grid: Iterable[int],
    metric: str,
    temp_excl_w: int,
    target_heads: int,
):
    """For each k, do greedy forward selection and return the best (k, heads, mse) combination."""
    best_overall_mse, best_overall_k, best_overall_heads = math.inf, None, None
    per_k_results = {}

    for k in k_grid:
        # Get candidate heads for this k (use ranking as guidance)
        head_ranking = rankings_per_k[k]  # [(mse, head_id), ...]
        candidates = [h for _, h in head_ranking]

        # Greedy forward selection for this k
        selected: List[int] = []
        while len(selected) < target_heads:
            best_h, best_mse = None, float("inf")
            for h in candidates:
                if h in selected:
                    continue
                trial = selected + [h]

                mse = evaluate_leave_one_episode_out(
                    per_head_features, actions, episode_ids, frame_ids, trial, k, metric, temp_excl_w,
                )
                if mse < best_mse:
                    best_mse, best_h = mse, h

            selected.append(best_h)

        # Final MSE for this k's selected heads
        final_mse = best_mse  # MSE from the last greedy step
        per_k_results[int(k)] = float(final_mse)

        if final_mse < best_overall_mse:
            best_overall_mse, best_overall_k, best_overall_heads = final_mse, k, selected

    return best_overall_heads, int(best_overall_k), float(best_overall_mse), per_k_results


__all__ = [
    "rank_single_heads_per_k",
    "rank_single_heads",
    "simple_topk_select",
    "greedy_forward_select",
]
