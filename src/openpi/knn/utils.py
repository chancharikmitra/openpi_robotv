import math
import h5py
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Note: avoid importing from metrics at module top-level to prevent circular imports with metrics -> utils


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


def build_feat_subset(per_head_features: np.ndarray, selected_heads: List[int]) -> np.ndarray:
    """
    Concatenate reduced features of selected heads into a single feature vector.
    Input: per_head_features shape (N, H, d)
    Output: (N, |selected_heads| * d)
    """
    subset = per_head_features[:, selected_heads, :]
    num_samples, num_heads, dim_per_head = subset.shape
    return subset.reshape(num_samples, num_heads * dim_per_head)


__all__ = ["weighted_avg", "build_feat_subset"]


# --------------------------
# Head selection utilities (decoupled from the main script)
# --------------------------
def rank_single_heads(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    k_grid: Iterable[int],
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
) -> List[Tuple[float, int, int]]:
    scores: List[Tuple[float, int, int]] = []

    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads
    from .eval import evaluate_leave_one_episode_out

    ctx_global_full = None
    if metric_scope == "global" and metric in {"whiten", "proj", "pls"} and pre_obj is not None and use_fullspace:
        ctx_global_full = build_global_metric_ctx_for_heads(
            pre_obj, list(range(H)), metric, pca_dim, proj_alpha,
            pls_components=pca_dim,  # upstream may pass specific PLS dim; otherwise capped by pca_dim
            use_fullspace=use_fullspace, H=H,
            metric_ctx_cache=metric_ctx_cache,
            metric_ctx_fullspace=metric_ctx_fullspace,
        )

    for h in tqdm(range(H), desc="Single-head CV MSE"):
        if metric_scope == "global" and metric in {"whiten", "proj", "pls"}:
            assert pre_obj is not None
            if ctx_global_full is not None and use_fullspace:
                dim_per_head = pre_obj.Xh_red.shape[-1]
                row_indices = list(range(h*dim_per_head, (h+1)*dim_per_head))
                if metric == "whiten":
                    mean_subset  = ctx_global_full["mu"][row_indices]
                    whitening_matrix_subset  = ctx_global_full["Wg"][row_indices, :]
                    ctx_global = {"Wg": whitening_matrix_subset, "mu": mean_subset}
                else:
                    projection_matrix_subset = ctx_global_full["W"][row_indices, :]
                    ctx_global = {"W": projection_matrix_subset}
            else:
                ctx_global = build_global_metric_ctx_for_heads(
                    pre_obj, [h], metric, pca_dim, proj_alpha,
                    pls_components=pca_dim,
                    use_fullspace=use_fullspace, H=H,
                    metric_ctx_cache=metric_ctx_cache,
                    metric_ctx_fullspace=metric_ctx_fullspace,
                )
        else:
            ctx_global = None

        best_mse, best_k = math.inf, None
        for k in k_grid:
            mse = evaluate_leave_one_episode_out(
                per_head_features, actions, episode_ids, frame_ids, [h], k, metric, temp_excl_w,
                pca_dim=pca_dim, proj_alpha=proj_alpha,
                metric_scope=metric_scope, metric_ctx_global=ctx_global,
            )
            if mse < best_mse:
                best_mse, best_k = mse, k
        scores.append((float(best_mse), int(best_k), h))
    scores.sort(key=lambda x: x[0])
    return scores


def simple_topk_select(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    head_rank: List[Tuple[float, int, int]],
    k_grid: Iterable[int],
    metric: str,
    temp_excl_w: int,
    target_heads: int,
    pca_dim: int,
    proj_alpha: float,
    pre_obj,
    metric_scope: str,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict,
):
    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads
    from .eval import evaluate_leave_one_episode_out

    sel_heads = [h for _, _, h in head_rank[:target_heads]]
    if metric_scope == "global" and metric in {"whiten", "proj", "pls"}:
        assert pre_obj is not None
        if use_fullspace:
            ctx_full = build_global_metric_ctx_for_heads(
                pre_obj, list(range(H)), metric, pca_dim, proj_alpha,
                pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
                metric_ctx_cache=metric_ctx_cache,
                metric_ctx_fullspace=metric_ctx_fullspace,
            )
            dim_per_head = pre_obj.Xh_red.shape[-1]
            row_indices = []
            for h in sorted(sel_heads):
                start = h*dim_per_head; row_indices.extend(range(start, start+dim_per_head))
            if metric == "whiten":
                ctx_global = {"Wg": ctx_full["Wg"][row_indices, :], "mu": ctx_full["mu"][row_indices]}
            else:
                ctx_global = {"W": ctx_full["W"][row_indices, :]}
        else:
            ctx_global = build_global_metric_ctx_for_heads(
                pre_obj, sel_heads, metric, pca_dim, proj_alpha,
                pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
                metric_ctx_cache=metric_ctx_cache,
                metric_ctx_fullspace=metric_ctx_fullspace,
            )
    else:
        ctx_global = None

    best_mse, best_k = math.inf, None
    per_k_mse = {}
    for k in k_grid:
        mse = evaluate_leave_one_episode_out(
            per_head_features, actions, episode_ids, frame_ids, sel_heads, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=ctx_global,
        )
        per_k_mse[int(k)] = float(mse)
        if mse < best_mse:
            best_mse, best_k = mse, k
    return sel_heads, int(best_k), float(best_mse), per_k_mse


def greedy_forward_select(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    head_rank: List[Tuple[float, int, int]],
    k_grid: Iterable[int],
    metric: str,
    temp_excl_w: int,
    target_heads: int,
    pca_dim: int,
    proj_alpha: float,
    pre_obj,
    metric_scope: str,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict,
):
    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads
    from .eval import evaluate_leave_one_episode_out

    candidates = [h for _, _, h in head_rank]
    selected: List[int] = []
    best_overall_mse, best_overall_k = float("inf"), None

    while len(selected) < target_heads:
        best_h, best_mse, best_k = None, float("inf"), None
        for h in candidates:
            if h in selected:
                continue
            trial = selected + [h]
            if metric_scope == "global" and metric in {"whiten", "proj", "pls"}:
                assert pre_obj is not None
                if use_fullspace:
                    ctx_full = build_global_metric_ctx_for_heads(
                        pre_obj, list(range(H)), metric, pca_dim, proj_alpha,
                        pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
                        metric_ctx_cache=metric_ctx_cache,
                        metric_ctx_fullspace=metric_ctx_fullspace,
                    )
                    dim_per_head = pre_obj.Xh_red.shape[-1]
                    row_indices = []
                    for hh in sorted(trial):
                        start = hh*dim_per_head; row_indices.extend(range(start, start+dim_per_head))
                    if metric == "whiten":
                        ctx_global = {"Wg": ctx_full["Wg"][row_indices, :], "mu": ctx_full["mu"][row_indices]}
                    else:
                        ctx_global = {"W": ctx_full["W"][row_indices, :]}
                else:
                    ctx_global = build_global_metric_ctx_for_heads(
                        pre_obj, trial, metric, pca_dim, proj_alpha,
                        pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
                        metric_ctx_cache=metric_ctx_cache,
                        metric_ctx_fullspace=metric_ctx_fullspace,
                    )
            else:
                ctx_global = None

            for k in k_grid:
                mse = evaluate_leave_one_episode_out(
                    per_head_features, actions, episode_ids, frame_ids, trial, k, metric, temp_excl_w,
                    pca_dim=pca_dim, proj_alpha=proj_alpha,
                    metric_scope=metric_scope, metric_ctx_global=ctx_global,
                )
                if mse < best_mse:
                    best_mse, best_k, best_h = mse, k, h
        selected.append(best_h)
        best_overall_mse, best_overall_k = best_mse, best_k

    # 统计最终 per-k MSE
    if metric_scope == "global" and metric in {"whiten", "proj", "pls"}:
        assert pre_obj is not None
        if use_fullspace:
            ctx_full = build_global_metric_ctx_for_heads(
                pre_obj, list(range(H)), metric, pca_dim, proj_alpha,
                pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
                metric_ctx_cache=metric_ctx_cache,
                metric_ctx_fullspace=metric_ctx_fullspace,
            )
            dim_per_head = pre_obj.Xh_red.shape[-1]
            row_indices = []
            for hh in sorted(selected):
                start = hh*dim_per_head; row_indices.extend(range(start, start+dim_per_head))
            if metric == "whiten":
                ctx_global = {"Wg": ctx_full["Wg"][row_indices, :], "mu": ctx_full["mu"][row_indices]}
            else:
                ctx_global = {"W": ctx_full["W"][row_indices, :]}
        else:
            ctx_global = build_global_metric_ctx_for_heads(
                pre_obj, selected, metric, pca_dim, proj_alpha,
                pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
                metric_ctx_cache=metric_ctx_cache,
                metric_ctx_fullspace=metric_ctx_fullspace,
            )
    else:
        ctx_global = None

    per_k_mse = {}
    for k in k_grid:
        mse = evaluate_leave_one_episode_out(
            per_head_features, actions, episode_ids, frame_ids, selected, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=ctx_global,
        )
        per_k_mse[int(k)] = float(mse)
    return selected, int(best_overall_k), float(best_overall_mse), per_k_mse


def reinforce_select_heads(
    pre,
    K_target: int,
    iters: int,
    batch: int,
    lr: float,
    k_grid: Iterable[int],
    metric: str,
    temp_excl_w: int,
    entropy_bonus: float,
    episode_subsample: Optional[float],
    pca_dim: int,
    proj_alpha: float,
):
    def _sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

    Hh = pre.Xh_red.shape[1]
    theta = np.zeros(Hh, dtype=np.float32)
    baseline = 0.0
    eps = 1e-8

    sub_pre = pre
    if episode_subsample is not None and 0 < episode_subsample < 1:
        rng = np.random.RandomState(0)
        uniq_eps = np.unique(pre.ep_ids)
        keep_eps = rng.choice(uniq_eps, size=max(1,int(len(uniq_eps)*episode_subsample)), replace=False)
        mask_ep = np.isin(pre.ep_ids, keep_eps)
        sub_pre = type(pre)(
            Xh_red=pre.Xh_red[mask_ep], Y=pre.Y[mask_ep],
            ep_ids=pre.ep_ids[mask_ep], t_ids=pre.t_ids[mask_ep],
            preproc=pre.preproc, episodes=pre.episodes,
        )

    for _ in range(iters):
        p = _sigmoid(theta)
        saved_log_probs = []
        rewards = []
        for _ in range(batch):
            m = (np.random.rand(Hh) < p).astype(np.float32)
            heads = np.where(m == 1)[0].tolist()
            if len(heads) == 0:
                best_mse = 1e9
            else:
                best_mse = np.inf
                for k in k_grid:
                    mse = evaluate_leave_one_episode_out(
                        sub_pre.Xh_red, sub_pre.Y, sub_pre.ep_ids, sub_pre.t_ids,
                        heads, k, metric, temp_excl_w,
                        pca_dim=pca_dim, proj_alpha=proj_alpha,
                        metric_scope="per_episode", metric_ctx_global=None,
                    )
                    if mse < best_mse:
                        best_mse = mse
            r = -best_mse
            rewards.append(r)
            log_prob = (m*np.log(p+eps) + (1-m)*np.log(1-p+eps)).sum()
            saved_log_probs.append(log_prob)

        rewards = np.array(rewards, dtype=np.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        entropy = -np.sum(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))
        grad = np.zeros_like(theta)
        for logp, r in zip(saved_log_probs, rewards):
            grad += ((np.random.rand(Hh) < p).astype(np.float32) - p) * (r - baseline)
        entropy_grad = p*(1-p) * (-np.log(p+eps) + np.log(1-p+eps))
        grad += (entropy_bonus * entropy_grad)
        theta += lr * grad / float(batch)
        baseline = 0.9*baseline + 0.1*float(np.mean(rewards))

    p = _sigmoid(theta)
    sel = np.argsort(-p)[:K_target].tolist()
    best_mse, best_k = np.inf, None
    for k in k_grid:
        mse = evaluate_leave_one_episode_out(
            pre.Xh_red, pre.Y, pre.ep_ids, pre.t_ids,
            sel, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope="per_episode", metric_ctx_global=None,
        )
        if mse < best_mse:
            best_mse, best_k = mse, k
    return sel, int(best_k), float(best_mse), p


# --------------------------
# Data processing utilities
# --------------------------

def _coerce_float32(arr: np.ndarray) -> np.ndarray:
    """Force convert array to float32"""
    if arr.dtype.kind == 'V' and arr.itemsize == 2:
        return arr.view('<f2').astype(np.float32)
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def _iter_frame_keys(group: h5py.Group):
    """Sort frame keys by numeric order"""
    keys = list(group.keys())
    try:
        keys_sorted = sorted(keys, key=lambda k: int(k.split("_")[-1]) if "_" in k else int(k))
    except Exception:
        keys_sorted = sorted(keys)
    return keys_sorted


def load_episode_frames(h5_file: h5py.File, episode_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all frames for a given episode.
    Returns:
      attention_data: (F, 18, 8, 256)
      action_data: (F, A) or (F,1) if scalar
    """
    group = h5_file[episode_key]
    attention_list, action_list = [], []
    for frame_key in _iter_frame_keys(group):
        frame_group = group[frame_key]
        if "last_token_attn" not in frame_group or "pi_droid_action" not in frame_group:
            raise KeyError(f"{episode_key}/{frame_key} missing 'last_token_attn' or 'pi_droid_action'")
        attention = _coerce_float32(np.asarray(frame_group["last_token_attn"]))   # (18,8,256)
        action = _coerce_float32(np.asarray(frame_group["pi_droid_action"]))   # (8,)
        attention_list.append(attention)
        action_list.append(action)
    
    attention_data = np.stack(attention_list, axis=0)  # (F,18,8,256)
    action_data = np.stack(action_list, axis=0)   # (F, A?) or (F,)
    if action_data.ndim == 1: 
        action_data = action_data[:, None]
    return attention_data, action_data


def frame_to_vec(frame_attention: np.ndarray, num_layers: int = 18, heads_per_layer: int = 8, d_head: int = 256) -> np.ndarray:
    """Flatten a single frame attention tensor: (18,8,256) -> (144,256)"""
    assert frame_attention.shape == (num_layers, heads_per_layer, d_head)
    return frame_attention.reshape(num_layers * heads_per_layer, d_head)


@dataclass
class HeadPreprocessor:
    """Per-head preprocessor"""
    scaler: StandardScaler
    pca: Optional[PCA]


class IdentityScaler:
    """Identity transform aligned to StandardScaler API; implements transform only."""
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


@dataclass
class PreprocessedData:
    """Preprocessed dataset"""
    Xh_red: np.ndarray          # (N, 144, d)  d = PCA_D or 256
    Y: np.ndarray               # (N, A)
    ep_ids: np.ndarray          # (N,)
    t_ids: np.ndarray           # (N,)
    preproc: List[HeadPreprocessor]
    episodes: List[str]

    # ---- Human-friendly aliases (without changing original fields) ----
    @property
    def features(self) -> np.ndarray:
        return self.Xh_red

    @property
    def actions(self) -> np.ndarray:
        return self.Y

    @property
    def episode_ids(self) -> np.ndarray:
        return self.ep_ids

    @property
    def frame_ids(self) -> np.ndarray:
        return self.t_ids


def build_dataset(
    attn_h5: str, 
    episodes: List[str],
    use_pca: bool = False,
    use_zscore: bool = False,
    pca_components: int = 32,
    num_layers: int = 18,
    heads_per_layer: int = 8,
    d_head: int = 256
) -> PreprocessedData:
    """
    Build the full dataset and apply per-head preprocessing (z-score + optional PCA).
    """
    total_heads = num_layers * heads_per_layer
    
    with h5py.File(attn_h5, "r") as f:
        total_frames = sum(len(f[ep].keys()) for ep in episodes)
    
    X_all = np.empty((total_frames, total_heads, d_head), dtype=np.float32)
    Y_all = []

    episodes_sorted = sorted(episodes)
    episode_to_id = {ep: i for i, ep in enumerate(episodes_sorted)}
    episode_ids = np.empty((total_frames,), dtype=np.int32)
    frame_ids = np.empty((total_frames,), dtype=np.int32)

    ptr = 0
    with h5py.File(attn_h5, "r") as f:
        for ep in tqdm(episodes, desc="Load episodes"):
            attention, actions = load_episode_frames(f, ep)     # (F,18,8,256), (F,A)
            F = attention.shape[0]
            X_all[ptr:ptr+F] = attention.reshape(F, total_heads, d_head)
            Y_all.append(actions.astype(np.float32))
            episode_ids[ptr:ptr+F] = episode_to_id[ep]
            frame_ids[ptr:ptr+F] = np.arange(F, dtype=np.int32)
            ptr += F

    Y = np.concatenate(Y_all, axis=0)  # (N, A)
    d_output = pca_components if use_pca else d_head
    Xh_reduced = np.empty((X_all.shape[0], total_heads, d_output), dtype=np.float32)
    preprocessors: List[HeadPreprocessor] = []
    
    for head in tqdm(range(total_heads), desc="Per-head zscore+PCA"):
        head_features = X_all[:, head, :]                          # (N,256)
        if use_zscore:
            scaler = StandardScaler().fit(head_features)
            standardized_features = scaler.transform(head_features)
        else:
            scaler = IdentityScaler()
            standardized_features = head_features
            
        if use_pca:
            pca = PCA(n_components=pca_components, svd_solver="auto", random_state=0).fit(standardized_features)
            reduced_features = pca.transform(standardized_features).astype(np.float32)  # (N,d_output)
        else:
            pca = None
            reduced_features = standardized_features.astype(np.float32)
            
        Xh_reduced[:, head, :] = reduced_features
        preprocessors.append(HeadPreprocessor(scaler, pca))

    return PreprocessedData(
        Xh_red=Xh_reduced, Y=Y, ep_ids=episode_ids, t_ids=frame_ids, 
        preproc=preprocessors, episodes=episodes_sorted
    )


def transform_frame(frame_attention: np.ndarray, model, num_layers: int = 18, heads_per_layer: int = 8, d_head: int = 256) -> np.ndarray:
    """Transform a single frame using the trained model's preprocessing"""
    attention_vector = frame_to_vec(frame_attention, num_layers, heads_per_layer, d_head)  # (144,256)
    feature_parts = []
    for head in model.heads:
        transformed = model.preproc[head].scaler.transform(attention_vector[head][None, :])
        if model.preproc[head].pca is not None:
            transformed = model.preproc[head].pca.transform(transformed)
        feature_parts.append(transformed.astype(np.float32))
    return np.concatenate(feature_parts, axis=1).reshape(-1)  # (|selected_heads|*d,)


__all__ += [
    "rank_single_heads",
    "simple_topk_select", 
    "greedy_forward_select",
    "reinforce_select_heads",
    "load_episode_frames",
    "frame_to_vec",
    "build_dataset", 
    "transform_frame",
    "HeadPreprocessor",
    "PreprocessedData",
]


# --------------------------
# Action label export utilities
# --------------------------
def get_action_labels(
    attn_h5: str,
    episodes: List[str],
    return_metadata: bool = True
):
    """
    Read action labels for the given episodes.
    Returns:
      - actions: np.ndarray (N, A)
      - meta (optional): { 'episode_ids', 'frame_ids', 'episodes' }
    """
    with h5py.File(attn_h5, "r") as f:
        total_frames = sum(len(f[ep].keys()) for ep in episodes)

    # First pass to collect counts
    with h5py.File(attn_h5, "r") as f:
        actions_list = []
        episodes_sorted = sorted(episodes)
        ep2id = {ep: i for i, ep in enumerate(episodes_sorted)}
        episode_ids = np.empty((total_frames,), dtype=np.int32)
        frame_ids = np.empty((total_frames,), dtype=np.int32)

        ptr = 0
        for ep in episodes:
            attn, act = load_episode_frames(f, ep)  # act: (F, A)
            F = act.shape[0]
            actions_list.append(act.astype(np.float32))
            episode_ids[ptr:ptr+F] = ep2id[ep]
            frame_ids[ptr:ptr+F] = np.arange(F, dtype=np.int32)
            ptr += F

    actions = np.concatenate(actions_list, axis=0)
    if not return_metadata:
        return actions
    return actions, {
        "episode_ids": episode_ids,
        "frame_ids": frame_ids,
        "episodes": episodes_sorted,
    }


def save_action_labels(
    attn_h5: str,
    episodes: List[str],
    out_path: str,
    fmt: str = "npy"
):
    """
    Export action labels for the given episodes.
    - fmt="npy": save a .npz with (actions, episode_ids, frame_ids, episodes)
    - fmt="csv": save a .csv with columns: episode, frame, action_0 ... action_{A-1}
    """
    actions, meta = get_action_labels(attn_h5, episodes, return_metadata=True)
    if fmt.lower() == "npy":
        np.savez(out_path, actions=actions, **meta)
        return out_path

    if fmt.lower() == "csv":
        import csv
        episodes_sorted = meta["episodes"]
        episode_ids = meta["episode_ids"]
        frame_ids = meta["frame_ids"]
        A = actions.shape[1]
        header = ["episode", "frame"] + [f"action_{i}" for i in range(A)]
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(actions.shape[0]):
                ep_name = episodes_sorted[int(episode_ids[i])]
                row = [ep_name, int(frame_ids[i])] + list(map(float, actions[i]))
                writer.writerow(row)
        return out_path

    raise ValueError("fmt must be 'npy' or 'csv'")


__all__ += [
    "get_action_labels",
    "save_action_labels",
]

