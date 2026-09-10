"""
KNN inference utilities: distance-weighted prediction and feature subset construction.

These functions are used at inference time to compute predictions from nearest-neighbor
lookups and to build concatenated feature vectors from a subset of attention heads.
"""

from typing import List

import numpy as np


def weighted_avg(neighbor_actions: np.ndarray, neighbor_dists: np.ndarray) -> np.ndarray:
    """
    Distance-weighted average of neighbor actions.
    - If any neighbor distance is near zero, return the nearest neighbor's action directly
      to avoid dilution by suboptimal neighbors.
    - Otherwise use inverse-distance weights with normalization.
    """
    eps = 1e-8
    if np.any(neighbor_dists < 1e-8):
        return neighbor_actions[np.argmin(neighbor_dists)]
    weights = 1.0 / (neighbor_dists + eps)
    weights /= (weights.sum() + eps)
    return (weights[:, None] * neighbor_actions).sum(axis=0)


def weighted_avg_batched(neighbor_actions: np.ndarray, neighbor_dists: np.ndarray) -> np.ndarray:
    """
    Batched distance-weighted average.
    Inputs:
      - neighbor_actions: (B, K, A)
      - neighbor_dists:   (B, K)
    Returns:
      - (B, A)
    If any distance in a row is near zero, use the single nearest neighbor's action for that row.
    """
    eps = 1e-8
    B, K = neighbor_dists.shape
    near_mask = neighbor_dists < 1e-8  # (B, K)
    has_near = near_mask.any(axis=1)   # (B,)

    # Default: inverse-distance weighted average
    weights = 1.0 / (neighbor_dists + eps)            # (B, K)
    weights /= (weights.sum(axis=1, keepdims=True) + eps)
    weighted = (weights[:, :, None] * neighbor_actions).sum(axis=1)  # (B, A)

    if has_near.any():
        # For rows with near neighbor, override with nearest neighbor action
        nearest_idx = np.argmin(neighbor_dists, axis=1)  # (B,)
        overrides = neighbor_actions[np.arange(B), nearest_idx, :]    # (B, A)
        weighted[has_near] = overrides[has_near]
    return weighted


def build_feat_subset(per_head_features: np.ndarray, selected_heads: List[int]) -> np.ndarray:
    """
    Concatenate reduced features of selected heads into a single feature vector.
    Input: per_head_features shape (N, H, d)
    Output: (N, |selected_heads| * d)
    """
    subset = per_head_features[:, selected_heads, :]
    num_samples, num_heads, dim_per_head = subset.shape
    return subset.reshape(num_samples, num_heads * dim_per_head)


__all__ = ["weighted_avg", "weighted_avg_batched", "build_feat_subset"]
