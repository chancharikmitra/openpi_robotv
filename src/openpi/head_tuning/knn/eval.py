import math
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm

from .inference import build_feat_subset, weighted_avg, weighted_avg_batched
from .data import transform_episode
from .metrics import pairwise_dist, pairwise_dist_matrix, build_metric_ctx_from_train


def evaluate_leave_one_episode_out_batched(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    heads: List[int],
    k: int,
    metric: str,
    temp_excl_w: int,
    pca_dim: int,
    proj_alpha: float,
    metric_scope: str,
    metric_ctx_global: Optional[dict] = None,
    batch_size: int = 1000,
) -> float:
    X = build_feat_subset(per_head_features, heads)
    uniq_eps = np.unique(episode_ids)
    if len(uniq_eps) == 1:
        return evaluate_leave_one_frame_out_single_episode(
            X, actions, frame_ids, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=metric_ctx_global,
        )

    sum_squared_error, num_evaluated = 0.0, 0

    # Optional global metric context
    if metric_scope == "global":
        global_ctx = metric_ctx_global
    else:
        global_ctx = None

    for ep in uniq_eps:
        te_mask = (episode_ids == ep)
        tr_mask = ~te_mask
        Xtr, Ytr = X[tr_mask], actions[tr_mask]
        Xte, Yte = X[te_mask], actions[te_mask]

        metric_ctx = (
            global_ctx
            if global_ctx is not None
            else build_metric_ctx_from_train(Xtr, Ytr if metric == "proj" else None, metric, pca_dim, proj_alpha)
        )

        n_test = Xte.shape[0]
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            Xq = Xte[start:end]                   # (B,P)
            D = pairwise_dist_matrix(Xq, Xtr, metric, metric_ctx)  # (B,Ntr)
            B, Ntr = D.shape
            effective_k = min(k, Ntr)
            # Top-k indices per row (partial sort)
            part = np.argpartition(D, kth=effective_k-1, axis=1)[:, :effective_k]  # (B,K)
            # Gather distances of top-k and sort them
            row_idx = np.arange(B)[:, None]
            topk_d = D[row_idx, part]
            order = np.argsort(topk_d, axis=1)
            sorted_idx = part[row_idx, order]            # (B,K)
            sorted_d   = D[row_idx, sorted_idx]          # (B,K)
            topk_actions = Ytr[sorted_idx]               # (B,K,A)
            pred = weighted_avg_batched(topk_actions, sorted_d)  # (B,A)
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
    metric: str,
    temp_excl_w: int,
    pca_dim: int,
    proj_alpha: float,
    metric_scope: str,
    metric_ctx_global: Optional[dict] = None,
) -> float:
    X = build_feat_subset(per_head_features, heads)
    uniq_eps = np.unique(episode_ids)
    if len(uniq_eps) == 1:
        return evaluate_leave_one_frame_out_single_episode(
            X, actions, frame_ids, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=metric_ctx_global,
        )

    sum_squared_error, num_evaluated = 0.0, 0
    for ep in uniq_eps:
        te_mask = (episode_ids == ep)
        tr_mask = ~te_mask
        Xtr, Ytr = X[tr_mask], actions[tr_mask]
        Xte, Yte = X[te_mask], actions[te_mask]

        metric_ctx = (
            metric_ctx_global
            if metric_scope == "global"
            else build_metric_ctx_from_train(Xtr, Ytr if metric == "proj" else None, metric, pca_dim, proj_alpha)
        )

        # Vectorized over test rows
        D = pairwise_dist_matrix(Xte, Xtr, metric, metric_ctx)   # (B,Ntr)
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
    metric: str,
    temp_excl_w: int,
    pca_dim: int,
    proj_alpha: float,
    metric_scope: str,
    metric_ctx_global: Optional[dict] = None,
) -> float:
    N = features.shape[0]
    sum_squared_error, num_evaluated = 0.0, 0

    # Precompute pairwise temporal mask (N,N) to enforce LOFO exclusion
    t_diff = np.abs(frame_ids[:, None] - frame_ids[None, :])
    valid = (t_diff > temp_excl_w)
    # Exclude self
    np.fill_diagonal(valid, False)

    for i in range(N):
        mask = valid[i]   # (N,)
        Xtr, Ytr = features[mask], actions[mask]
        if Xtr.shape[0] == 0:
            continue

        metric_ctx = (
            metric_ctx_global
            if metric_scope == "global"
            else build_metric_ctx_from_train(Xtr, Ytr if metric == "proj" else None, metric, pca_dim, proj_alpha)
        )

        distances = pairwise_dist(features[i], Xtr, metric, metric_ctx)
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
    metric: str,
    temp_excl_w: int,
    build_dataset_fn,
) -> Tuple[float, int]:
    pre = build_dataset_fn(attn_h5, episodes)
    X = build_feat_subset(pre.features, heads)
    best_mse, best_k = math.inf, None
    for k in k_grid:
        mse = evaluate_leave_one_episode_out(
            pre.features, pre.actions, pre.episode_ids, pre.frame_ids, heads, k, metric, temp_excl_w,
            pca_dim=256, proj_alpha=1e-2, metric_scope="per_episode",
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
            D = pairwise_dist_matrix(Q, model.X_bank, model.metric, model.metric_ctx)  # (F, N)
            # 对每个 k，一次性取 top-k 并计算批量加权平均
            for k in ks_list:
                k_eff = min(int(k), D.shape[1])
                part = np.argpartition(D, kth=k_eff-1, axis=1)[:, :k_eff]
                row_idx = np.arange(F)[:, None]
                topk_d = D[row_idx, part]
                order = np.argsort(topk_d, axis=1)
                sorted_idx = part[row_idx, order]
                sorted_d   = D[row_idx, sorted_idx]
                topk_actions = model.Y_bank[sorted_idx]  # (F,K,A)
                pred = weighted_avg_batched(topk_actions, sorted_d)  # (F,A)
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


def evaluate_leave_one_episode_out_weighted(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    head_weights: np.ndarray,  # (H,) weights for each head
    k: int,
    metric: str,
    temp_excl_w: int,
    pca_dim: int,
    proj_alpha: float,
    pre_obj,
    metric_scope: str,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict,
) -> float:
    """
    LOEO evaluation using weighted combination of all heads instead of selecting subset.
    Each head gets a learnable scalar weight.
    """
    # Build weighted feature representation: sum over heads with weights
    N, H_total, d = per_head_features.shape
    assert H_total == H == len(head_weights)
    
    # Weighted sum: X_weighted[i] = sum_h (weight_h * per_head_features[i, h, :])
    X_weighted = np.sum(head_weights[None, :, None] * per_head_features, axis=1)  # (N, d)
    
    uniq_eps = np.unique(episode_ids)
    if len(uniq_eps) == 1:
        return evaluate_leave_one_frame_out_single_episode_weighted(
            X_weighted, actions, frame_ids, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=None,
        )

    sum_squared_error, num_evaluated = 0.0, 0
    for ep in uniq_eps:
        te_mask = (episode_ids == ep)
        tr_mask = ~te_mask
        Xtr, Ytr = X_weighted[tr_mask], actions[tr_mask]
        Xte, Yte = X_weighted[te_mask], actions[te_mask]

        metric_ctx = (
            None  # For weighted features, we don't use global metric context
            if metric_scope == "global"
            else build_metric_ctx_from_train(Xtr, Ytr if metric in {"proj", "pls"} else None, metric, pca_dim, proj_alpha)
        )

        # Vectorized over test rows
        D = pairwise_dist_matrix(Xte, Xtr, metric, metric_ctx)   # (B,Ntr)
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


def evaluate_leave_one_frame_out_single_episode_weighted(
    features: np.ndarray,
    actions: np.ndarray,
    frame_ids: np.ndarray,
    k: int,
    metric: str,
    temp_excl_w: int,
    pca_dim: int,
    proj_alpha: float,
    metric_scope: str,
    metric_ctx_global: Optional[dict] = None,
) -> float:
    """LOFO evaluation for weighted features (single episode)."""
    N = features.shape[0]
    sum_squared_error, num_evaluated = 0.0, 0

    # Precompute pairwise temporal mask (N,N) to enforce LOFO exclusion
    t_diff = np.abs(frame_ids[:, None] - frame_ids[None, :])
    valid = (t_diff > temp_excl_w)
    # Exclude self
    np.fill_diagonal(valid, False)

    for i in range(N):
        mask = valid[i]   # (N,)
        Xtr, Ytr = features[mask], actions[mask]
        if Xtr.shape[0] == 0:
            continue

        metric_ctx = (
            metric_ctx_global
            if metric_scope == "global"
            else build_metric_ctx_from_train(Xtr, Ytr if metric in {"proj", "pls"} else None, metric, pca_dim, proj_alpha)
        )

        distances = pairwise_dist(features[i], Xtr, metric, metric_ctx)
        effective_k = min(k, len(distances))
        neighbor_indices = np.argpartition(distances, effective_k)[:effective_k]
        neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
        predicted_action = weighted_avg(Ytr[neighbor_indices], distances[neighbor_indices])
        sum_squared_error += ((predicted_action - actions[i]) ** 2).mean()
        num_evaluated += 1
    return sum_squared_error / max(num_evaluated, 1)


def predict_episode(attn_h5: str, episode_key: str, model) -> np.ndarray:
    """Predict actions frame-by-frame for the specified episode (vectorized)."""
    from .data import load_episode_frames, transform_episode, frame_to_vec
    from .metrics import pairwise_dist_matrix

    import h5py
    with h5py.File(attn_h5, "r") as f:
        attn, _ = load_episode_frames(f, episode_key)  # (F,18,8,256), (F,A) actions not needed here
    F = attn.shape[0]

    if model.head_weights is not None:
        # Weighted mode: transform all heads and apply weights
        Q_per_head = np.empty((F, len(model.preproc), model.d_per_head), dtype=np.float32)
        for t in range(F):
            attention_vector = frame_to_vec(attn[t])  # (144,256)
            for h in range(len(model.preproc)):
                transformed = model.preproc[h].scaler.transform(attention_vector[h][None, :])
                if model.preproc[h].pca is not None:
                    transformed = model.preproc[h].pca.transform(transformed)
                Q_per_head[t, h, :] = transformed.astype(np.float32).flatten()
        
        # Apply learned weights: Q[t] = sum_h (weight_h * Q_per_head[t, h, :])
        Q = np.sum(model.head_weights[None, :, None] * Q_per_head, axis=1)  # (F, d)
        
        # Reconstruct X_bank with weights for distance computation
        X_bank_per_head = model.X_bank.reshape(model.X_bank.shape[0], len(model.preproc), model.d_per_head)
        X_bank_weighted = np.sum(model.head_weights[None, :, None] * X_bank_per_head, axis=1)  # (N, d)
        
        D = pairwise_dist_matrix(Q, X_bank_weighted, model.metric, model.metric_ctx)  # (F, N)
    else:
        # Standard mode: use selected heads
        Q = transform_episode(attn, model)  # (F, |S|*d)
        D = pairwise_dist_matrix(Q, model.X_bank, model.metric, model.metric_ctx)  # (F, N)
    
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
    "evaluate_leave_one_episode_out_batched",  # batched version (no real perf gain)
    "evaluate_leave_one_episode_out",         # standard version
    "evaluate_leave_one_frame_out_single_episode",
    "evaluate_head_subset_cross_validation",
    "evaluate_model_on_h5",
    "predict_episode",
    "evaluate_leave_one_episode_out_weighted",
    "evaluate_leave_one_frame_out_single_episode_weighted",
]


# --------------------------
# Predict from raw activations (np array)
# --------------------------
def predict_episode_from_activation(attn_or_frames: np.ndarray, model) -> np.ndarray:
    """
    Predict actions given activations without reading H5.
    Inputs:
      - attn_or_frames: (18,8,256) for single frame, or (F,18,8,256) for multiple frames
    Returns:
      - (A,) for single frame input, or (F, A) for multiple frames
    """
    from .data import transform_episode, frame_to_vec
    from .inference import weighted_avg_batched
    from .metrics import pairwise_dist_matrix

    if attn_or_frames.ndim == 3 and attn_or_frames.shape == (18, 8, 256):
        attn = attn_or_frames[None, ...]
        single = True
    elif attn_or_frames.ndim == 4 and attn_or_frames.shape[1:] == (18, 8, 256):
        attn = attn_or_frames
        single = False
    else:
        raise ValueError("attn_or_frames must be (18,8,256) or (F,18,8,256)")

    F = attn.shape[0]

    if getattr(model, "head_weights", None) is not None:
        # Weighted mode: transform all heads, then apply learned weights
        Q_per_head = np.empty((F, len(model.preproc), model.d_per_head), dtype=np.float32)
        for t in range(F):
            attention_vector = frame_to_vec(attn[t])  # (144,256)
            for h in range(len(model.preproc)):
                transformed = model.preproc[h].scaler.transform(attention_vector[h][None, :])
                if model.preproc[h].pca is not None:
                    transformed = model.preproc[h].pca.transform(transformed)
                Q_per_head[t, h, :] = transformed.astype(np.float32).reshape(-1)
        Q = np.sum(model.head_weights[None, :, None] * Q_per_head, axis=1)  # (F, d)

        X_bank_per_head = model.X_bank.reshape(model.X_bank.shape[0], len(model.preproc), model.d_per_head)
        X_bank_weighted = np.sum(model.head_weights[None, :, None] * X_bank_per_head, axis=1)  # (N, d)
        D = pairwise_dist_matrix(Q, X_bank_weighted, model.metric, model.metric_ctx)
    else:
        # Standard selected-heads mode
        Q = transform_episode(attn, model)  # (F, |S|*d)
        D = pairwise_dist_matrix(Q, model.X_bank, model.metric, model.metric_ctx)

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
