import math
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm

from .inference import build_feat_subset, weighted_avg, weighted_avg_batched
from .data import transform_episode
from .metrics import pairwise_dist, pairwise_dist_matrix


def evaluate_leave_one_episode_out_batched(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    heads: List[int],
    k: int,
    metric: str = "cosine",
    temp_excl_w: int = 30,
    batch_size: int = 1000,
) -> float:
    X = build_feat_subset(per_head_features, heads)
    uniq_eps = np.unique(episode_ids)
    if len(uniq_eps) == 1:
        return evaluate_leave_one_frame_out_single_episode(
            X, actions, frame_ids, k, metric, temp_excl_w,
        )

    sum_squared_error, num_evaluated = 0.0, 0

    for ep in uniq_eps:
        te_mask = (episode_ids == ep)
        tr_mask = ~te_mask
        Xtr, Ytr = X[tr_mask], actions[tr_mask]
        Xte, Yte = X[te_mask], actions[te_mask]

        n_test = Xte.shape[0]
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            Xq = Xte[start:end]                   # (B, P)
            D = pairwise_dist_matrix(Xq, Xtr, metric)  # (B, Ntr)
            B, Ntr = D.shape
            effective_k = min(k, Ntr)
            # Top-k indices per row (partial sort)
            part = np.argpartition(D, kth=effective_k-1, axis=1)[:, :effective_k]  # (B, K)
            # Gather distances of top-k and sort them
            row_idx = np.arange(B)[:, None]
            topk_d = D[row_idx, part]
            order = np.argsort(topk_d, axis=1)
            sorted_idx = part[row_idx, order]            # (B, K)
            sorted_d   = D[row_idx, sorted_idx]          # (B, K)
            topk_actions = Ytr[sorted_idx]               # (B, K, A)
            pred = weighted_avg_batched(topk_actions, sorted_d)  # (B, A)
            se = ((pred - Yte[start:end]) ** 2).mean(axis=1)     # (B,)
            sum_squared_error += float(se.sum())
            num_evaluated += int(B)

    return sum_squared_error / max(num_evaluated, 1)


def evaluate_leave_one_episode_out(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    heads: List[int],
    k: int,
    metric: str = "cosine",
    temp_excl_w: int = 30,
) -> float:
    X = build_feat_subset(per_head_features, heads)
    uniq_eps = np.unique(episode_ids)
    if len(uniq_eps) == 1:
        return evaluate_leave_one_frame_out_single_episode(
            X, actions, frame_ids, k, metric, temp_excl_w,
        )

    sum_squared_error, num_evaluated = 0.0, 0
    for ep in uniq_eps:
        te_mask = (episode_ids == ep)
        tr_mask = ~te_mask
        Xtr, Ytr = X[tr_mask], actions[tr_mask]
        Xte, Yte = X[te_mask], actions[te_mask]

        # Vectorized over test rows
        D = pairwise_dist_matrix(Xte, Xtr, metric)   # (B, Ntr)
        B, Ntr = D.shape
        effective_k = min(k, Ntr)
        part = np.argpartition(D, kth=effective_k-1, axis=1)[:, :effective_k]
        row_idx = np.arange(B)[:, None]
        topk_d = D[row_idx, part]
        order = np.argsort(topk_d, axis=1)
        sorted_idx = part[row_idx, order]
        sorted_d   = D[row_idx, sorted_idx]
        topk_actions = Ytr[sorted_idx]
        pred = weighted_avg_batched(topk_actions, sorted_d)
        se = ((pred - Yte) ** 2).mean(axis=1)
        sum_squared_error += float(se.sum())
        num_evaluated += int(B)
    return sum_squared_error / max(num_evaluated, 1)


def evaluate_leave_one_frame_out_single_episode(
    features: np.ndarray,
    actions: np.ndarray,
    frame_ids: np.ndarray,
    k: int,
    metric: str = "cosine",
    temp_excl_w: int = 30,
) -> float:
    N = features.shape[0]
    sum_squared_error, num_evaluated = 0.0, 0

    # Precompute pairwise temporal mask (N, N) to enforce LOFO exclusion
    t_diff = np.abs(frame_ids[:, None] - frame_ids[None, :])
    valid = (t_diff > temp_excl_w)
    # Exclude self
    np.fill_diagonal(valid, False)

    for i in range(N):
        mask = valid[i]   # (N,)
        Xtr, Ytr = features[mask], actions[mask]
        if Xtr.shape[0] == 0:
            continue

        distances = pairwise_dist(features[i], Xtr, metric)
        effective_k = min(k, len(distances))
        neighbor_indices = np.argpartition(distances, effective_k)[:effective_k]
        neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
        predicted_action = weighted_avg(Ytr[neighbor_indices], distances[neighbor_indices])
        sum_squared_error += ((predicted_action - actions[i]) ** 2).mean()
        num_evaluated += 1
    return sum_squared_error / max(num_evaluated, 1)


def evaluate_head_subset_cross_validation(
    attn_h5: str,
    episodes: List[str],
    heads: List[int],
    k_grid: Iterable[int],
    temp_excl_w: int,
    build_dataset_fn,
    metric: str = "cosine",
) -> Tuple[float, int]:
    pre = build_dataset_fn(attn_h5, episodes)
    best_mse, best_k = math.inf, None
    for k in k_grid:
        mse = evaluate_leave_one_episode_out(
            pre.features, pre.actions, pre.episode_ids, pre.frame_ids, heads, k, metric, temp_excl_w,
        )
        if mse < best_mse:
            best_mse, best_k = mse, k
    return float(best_mse), int(best_k)


def evaluate_model_on_h5(
    attn_h5_eval: str,
    episodes_eval: List[str],
    model,  # KnnRegModel
    ks: Optional[Iterable[int]] = None,
    max_frames: Optional[int] = None
) -> Dict:
    """
    Evaluate a trained model (feature/action bank from train H5) on a separate eval H5.
    - No LOEO exclusion (eval and train are different H5); nearest neighbors are retrieved from the train bank.
    - Supports multiple k in ks; if ks=None, evaluate only model.k.
    - Returns overall MSE, per-k MSE, and per-episode MSE.
    """
    from .data import load_episode_frames
    from .metrics import pairwise_dist_matrix

    ks_list = sorted(set(list(ks))) if ks is not None else [model.k]
    per_k_squared_error = {int(k): 0.0 for k in ks_list}
    per_k_count         = {int(k): 0   for k in ks_list}
    per_episode_squared_error = {ep: {int(k): 0.0 for k in ks_list} for ep in episodes_eval}
    per_episode_count         = {ep: {int(k): 0   for k in ks_list} for ep in episodes_eval}

    import h5py
    with h5py.File(attn_h5_eval, "r") as f:
        for ep in tqdm(episodes_eval, desc="Eval episodes"):
            attn, act = load_episode_frames(f, ep)
            F = attn.shape[0] if max_frames is None else min(max_frames, attn.shape[0])
            Q = transform_episode(attn[:F], model)               # (F, |S|*d)
            D = pairwise_dist_matrix(Q, model.X_bank, model.metric)  # (F, N)
            # For each k, take top-k and compute batch-weighted average
            for k in ks_list:
                k_eff = min(int(k), D.shape[1])
                part = np.argpartition(D, kth=k_eff-1, axis=1)[:, :k_eff]
                row_idx = np.arange(F)[:, None]
                topk_d = D[row_idx, part]
                order = np.argsort(topk_d, axis=1)
                sorted_idx = part[row_idx, order]
                sorted_d   = D[row_idx, sorted_idx]
                topk_actions = model.Y_bank[sorted_idx]  # (F, K, A)
                pred = weighted_avg_batched(topk_actions, sorted_d)  # (F, A)
                se = ((pred - act[:F]) ** 2).mean(axis=1)            # (F,)
                per_k_squared_error[int(k)] += float(se.sum())
                per_k_count[int(k)]         += int(F)
                per_episode_squared_error[ep][int(k)] += float(se.sum())
                per_episode_count[ep][int(k)]         += int(F)

    per_k_mse = {k: (per_k_squared_error[k] / max(per_k_count[k], 1)) for k in ks_list}
    overall_mse = per_k_mse[ks_list[0]] if len(ks_list) == 1 else min(per_k_mse.values())
    per_ep_mse = {
        ep: {k: (per_episode_squared_error[ep][k] / max(per_episode_count[ep][k], 1)) for k in ks_list}
        for ep in episodes_eval
    }
    print(f"[evaluate_model_on_h5] per-k MSE: {per_k_mse}")
    return {
        "overall_mse": float(overall_mse),
        "per_k_mse": {int(k): float(v) for k, v in per_k_mse.items()},
        "per_episode_mse": per_ep_mse,
        "ks": ks_list,
        "episodes": episodes_eval,
    }


def predict_episode(attn_h5: str, episode_key: str, model) -> np.ndarray:
    """Predict actions frame-by-frame for the specified episode (vectorized)."""
    from .data import load_episode_frames, transform_episode
    from .metrics import pairwise_dist_matrix

    import h5py
    with h5py.File(attn_h5, "r") as f:
        attn, _ = load_episode_frames(f, episode_key)  # (F, 18, 8, 256)
    F = attn.shape[0]

    Q = transform_episode(attn, model)  # (F, |S|*d)
    D = pairwise_dist_matrix(Q, model.X_bank, model.metric)  # (F, N)

    k_eff = min(model.k, D.shape[1])
    part = np.argpartition(D, kth=k_eff-1, axis=1)[:, :k_eff]
    row_idx = np.arange(F)[:, None]
    topk_d = D[row_idx, part]
    order = np.argsort(topk_d, axis=1)
    sorted_idx = part[row_idx, order]
    sorted_d   = D[row_idx, sorted_idx]
    topk_actions = model.Y_bank[sorted_idx]
    pred = weighted_avg_batched(topk_actions, sorted_d)
    return pred  # (F, A)


__all__ = [
    "evaluate_leave_one_episode_out_batched",
    "evaluate_leave_one_episode_out",
    "evaluate_leave_one_frame_out_single_episode",
    "evaluate_head_subset_cross_validation",
    "evaluate_model_on_h5",
    "predict_episode",
]


# --------------------------
# Predict from raw activations (np array)
# --------------------------
def predict_episode_from_activation(attn_or_frames: np.ndarray, model) -> np.ndarray:
    """
    Predict actions given activations without reading H5.
    Inputs:
      - attn_or_frames: (18, 8, 256) for single frame, or (F, 18, 8, 256) for multiple frames
    Returns:
      - (A,) for single frame input, or (F, A) for multiple frames
    """
    from .data import transform_episode
    from .inference import weighted_avg_batched
    from .metrics import pairwise_dist_matrix

    if attn_or_frames.ndim == 3 and attn_or_frames.shape == (18, 8, 256):
        attn = attn_or_frames[None, ...]
        single = True
    elif attn_or_frames.ndim == 4 and attn_or_frames.shape[1:] == (18, 8, 256):
        attn = attn_or_frames
        single = False
    else:
        raise ValueError("attn_or_frames must be (18, 8, 256) or (F, 18, 8, 256)")

    F = attn.shape[0]
    Q = transform_episode(attn, model)  # (F, |S|*d)
    D = pairwise_dist_matrix(Q, model.X_bank, model.metric)

    k_eff = min(model.k, D.shape[1])
    part = np.argpartition(D, kth=k_eff-1, axis=1)[:, :k_eff]
    row_idx = np.arange(F)[:, None]
    topk_d = D[row_idx, part]
    order = np.argsort(topk_d, axis=1)
    sorted_idx = part[row_idx, order]
    sorted_d   = D[row_idx, sorted_idx]
    topk_actions = model.Y_bank[sorted_idx]
    pred = weighted_avg_batched(topk_actions, sorted_d)  # (F, A)

    if single:
        return pred[0]
    return pred
