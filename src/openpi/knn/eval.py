import math
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm

from .utils import build_feat_subset, weighted_avg
from .metrics import pairwise_dist, build_metric_ctx_from_train


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
            Xte_batch = Xte[start:end]
            for i in range(Xte_batch.shape[0]):
                distances = pairwise_dist(Xte_batch[i], Xtr, metric, metric_ctx)
                effective_k = min(k, len(distances))
                neighbor_indices = np.argpartition(distances, effective_k)[:effective_k]
                neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
                predicted_action = weighted_avg(Ytr[neighbor_indices], distances[neighbor_indices])
                sum_squared_error += ((predicted_action - Yte[start + i]) ** 2).mean()
                num_evaluated += 1

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

        for i in range(Xte.shape[0]):
            distances = pairwise_dist(Xte[i], Xtr, metric, metric_ctx)
            effective_k = min(k, len(distances))
            neighbor_indices = np.argpartition(distances, effective_k)[:effective_k]
            neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
            predicted_action = weighted_avg(Ytr[neighbor_indices], distances[neighbor_indices])
            sum_squared_error += ((predicted_action - Yte[i]) ** 2).mean()
            num_evaluated += 1
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
    sum_squared_error, num_evaluated = 0.0, 0
    for i in range(features.shape[0]):
        mask = np.ones(features.shape[0], dtype=bool)
        mask[i] = False
        too_close = np.abs(frame_ids - frame_ids[i]) <= temp_excl_w
        mask &= ~too_close
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
    from .utils import load_episode_frames, transform_frame
    from .metrics import pairwise_dist
    
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
            for t in range(F):
                q = transform_frame(attn[t], model)              # (|S|*d,)
                distances = pairwise_dist(q, model.X_bank, model.metric, model.metric_ctx)
                sorted_indices = np.argsort(distances)
                for k in ks_list:
                    effective_k = min(int(k), len(sorted_indices))
                    top_indices = sorted_indices[:effective_k]
                    predicted_action = weighted_avg(model.Y_bank[top_indices], distances[top_indices])
                    squared_error = ((predicted_action - act[t]) ** 2).mean()
                    per_k_squared_error[int(k)] += float(squared_error)
                    per_k_count[int(k)]         += 1
                    per_episode_squared_error[ep][int(k)] += float(squared_error)
                    per_episode_count[ep][int(k)]         += 1

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
    """Predict actions frame-by-frame for the specified episode"""
    from .utils import load_episode_frames, transform_frame
    from .metrics import pairwise_dist
    
    import h5py
    with h5py.File(attn_h5, "r") as f:
        attn, _ = load_episode_frames(f, episode_key)  # (F,18,8,256), (F,A) actions not needed here
    F = attn.shape[0]
    predictions = []
    for t in range(F):
        query_features = transform_frame(attn[t], model)
        distances = pairwise_dist(query_features, model.X_bank, model.metric, model.metric_ctx)
        effective_k = min(model.k, len(distances))
        neighbor_indices = np.argpartition(distances, effective_k)[:effective_k]
        neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
        predicted_action = weighted_avg(model.Y_bank[neighbor_indices], distances[neighbor_indices])
        predictions.append(predicted_action)
    return np.stack(predictions, axis=0)  # (F, A)


__all__ = [
    "evaluate_leave_one_episode_out_batched",  # batched version (no real perf gain)
    "evaluate_leave_one_episode_out",         # standard version
    "evaluate_leave_one_frame_out_single_episode",
    "evaluate_head_subset_cross_validation",
    "evaluate_model_on_h5",
    "predict_episode",
]


