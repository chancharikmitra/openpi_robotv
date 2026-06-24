"""
KNN head selection algorithms.

Provides single-head ranking, greedy forward selection, simple top-k selection,
REINFORCE-based selection, and differentiable head weight learning. All methods
evaluate candidate head subsets via leave-one-episode-out (LOEO) cross-validation.

Note: `reinforce_select_heads`, `learn_head_weights`, `compute_loeo_mse_torch`, and
`compute_loeo_mse_torch_with_metric` are kept here temporarily and will be removed in
a later task.
"""

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def rank_single_heads_per_k(
    per_head_features: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
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

    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads
    from .eval import evaluate_leave_one_episode_out

    ctx_global_full = None
    if metric_scope == "global" and metric in {"whiten", "proj", "pls"} and pre_obj is not None and use_fullspace:
        ctx_global_full = build_global_metric_ctx_for_heads(
            pre_obj, list(range(H)), metric, pca_dim, proj_alpha,
            pls_components=pca_dim,
            use_fullspace=use_fullspace, H=H,
            metric_ctx_cache=metric_ctx_cache,
            metric_ctx_fullspace=metric_ctx_fullspace,
        )

    print(f"\n=== Single {unit_word} Ranking for k={k} ===")
    for h in range(H):
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

        mse = evaluate_leave_one_episode_out(
            per_head_features, actions, episode_ids, frame_ids, [h], k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=ctx_global,
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
    pca_dim: int,
    proj_alpha: float,
    pre_obj,
    metric_scope: str,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict,
    print_rankings: bool = True,
    unit_mode: str = "head",
) -> dict:
    """Return rankings per k: {k: [(mse, head_id), ...]}"""
    rankings_per_k = {}
    desc = "Single-layer ranking per k" if unit_mode == "layer" else "Single-head ranking per k"
    for k in tqdm(k_grid, desc=desc):
        rankings_per_k[k] = rank_single_heads_per_k(
            per_head_features, actions, episode_ids, frame_ids, k, metric, temp_excl_w,
            pca_dim, proj_alpha, pre_obj, metric_scope, use_fullspace, H,
            metric_ctx_cache, metric_ctx_fullspace,
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
    pca_dim: int,
    proj_alpha: float,
    pre_obj,
    metric_scope: str,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict,
):
    """For each k, select top heads and evaluate. Return the best (k, heads, mse) combination."""
    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads
    from .eval import evaluate_leave_one_episode_out

    best_overall_mse, best_overall_k, best_overall_heads = math.inf, None, None
    per_k_results = {}

    for k in k_grid:
        # Get top heads for this k
        head_ranking = rankings_per_k[k]  # [(mse, head_id), ...]
        sel_heads = [h for _, h in head_ranking[:target_heads]]

        # Evaluate this head subset at this k
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

        mse = evaluate_leave_one_episode_out(
            per_head_features, actions, episode_ids, frame_ids, sel_heads, k, metric, temp_excl_w,
            pca_dim=pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=ctx_global,
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
    pca_dim: int,
    proj_alpha: float,
    pre_obj,
    metric_scope: str,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict,
):
    """For each k, do greedy forward selection and return the best (k, heads, mse) combination."""
    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads
    from .eval import evaluate_leave_one_episode_out

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

                # Evaluate this trial at this specific k
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

                mse = evaluate_leave_one_episode_out(
                    per_head_features, actions, episode_ids, frame_ids, trial, k, metric, temp_excl_w,
                    pca_dim=pca_dim, proj_alpha=proj_alpha,
                    metric_scope=metric_scope, metric_ctx_global=ctx_global,
                )
                if mse < best_mse:
                    best_mse, best_h = mse, h

            selected.append(best_h)

        # Final MSE for this k's selected heads
        final_mse = best_mse  # This is the MSE from the last iteration
        per_k_results[int(k)] = float(final_mse)

        if final_mse < best_overall_mse:
            best_overall_mse, best_overall_k, best_overall_heads = final_mse, k, selected

    return best_overall_heads, int(best_overall_k), float(best_overall_mse), per_k_results


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

    # Lazy import to avoid circular dependency
    from .eval import evaluate_leave_one_episode_out

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


def learn_head_weights(
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
    lr: float = 0.01,
    iters: int = 100,
    weight_decay: float = 1e-4,
):
    """
    Learn lightweight scalar weights for each head using PyTorch autograd.
    Uses the same global metric transform as other head selection methods.
    Returns (head_weights, best_k, best_mse, per_k_mse)
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required for learn_head_weights. Install with: pip install torch")

    # Lazy import to avoid circular dependency
    from .metrics import build_global_metric_ctx_for_heads

    # Convert to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert numpy arrays to torch tensors
    features_torch = torch.from_numpy(per_head_features).float().to(device)  # (N, H, d)
    actions_torch = torch.from_numpy(actions).float().to(device)  # (N, A)
    episode_ids_torch = torch.from_numpy(episode_ids).long().to(device)
    frame_ids_torch = torch.from_numpy(frame_ids).long().to(device)

    # Build global metric context (same as other methods)
    global_metric_ctx = None
    if metric_scope == "global" and metric in {"whiten", "proj", "pls"} and use_fullspace:
        global_metric_ctx = build_global_metric_ctx_for_heads(
            pre_obj, list(range(H)), metric, pca_dim, proj_alpha,
            pls_components=pca_dim, use_fullspace=use_fullspace, H=H,
            metric_ctx_cache=metric_ctx_cache,
            metric_ctx_fullspace=metric_ctx_fullspace,
        )
        print(f"Using global {metric} metric context for weight learning")

    best_overall_mse, best_overall_k = math.inf, None
    results_per_k = {}

    for k in k_grid:
        print(f"Learning head weights for k={k}...")

        # Initialize learnable weights (log scale for positivity)
        log_weights = torch.zeros(H, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([log_weights], lr=lr, weight_decay=weight_decay)

        for iter_idx in range(iters):
            optimizer.zero_grad()

            # Convert log weights to normalized weights
            weights = F.softmax(log_weights, dim=0)  # (H,)

            # Compute weighted features: (N, H, d) * (H,) -> (N, d)
            weighted_features = torch.sum(weights.unsqueeze(0).unsqueeze(-1) * features_torch, dim=1)

            # Compute LOEO MSE using PyTorch operations with global metric context
            mse_loss = compute_loeo_mse_torch_with_metric(
                weighted_features, actions_torch, episode_ids_torch, frame_ids_torch,
                k, metric, temp_excl_w, device, global_metric_ctx, weights
            )

            # Backward pass
            mse_loss.backward()
            optimizer.step()

            if (iter_idx + 1) % 20 == 0:
                print(f"  Iter {iter_idx+1}/{iters}, MSE: {mse_loss.item():.6f}")

        # Final evaluation
        with torch.no_grad():
            final_weights = F.softmax(log_weights, dim=0)
            weighted_features = torch.sum(final_weights.unsqueeze(0).unsqueeze(-1) * features_torch, dim=1)
            final_mse = compute_loeo_mse_torch_with_metric(
                weighted_features, actions_torch, episode_ids_torch, frame_ids_torch,
                k, metric, temp_excl_w, device, global_metric_ctx, final_weights
            )

            final_weights_np = final_weights.cpu().numpy()
            final_mse_float = float(final_mse.item())

            results_per_k[int(k)] = {
                'mse': final_mse_float,
                'weights': final_weights_np.copy()
            }

            if final_mse_float < best_overall_mse:
                best_overall_mse, best_overall_k = final_mse_float, k
                best_weights = final_weights_np.copy()

    per_k_mse = {k: results_per_k[k]['mse'] for k in results_per_k}

    return best_weights, int(best_overall_k), float(best_overall_mse), per_k_mse


def compute_loeo_mse_torch(
    features: torch.Tensor,  # (N, d)
    actions: torch.Tensor,   # (N, A)
    episode_ids: torch.Tensor,  # (N,)
    frame_ids: torch.Tensor,    # (N,)
    k: int,
    metric: str,
    temp_excl_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute LOEO MSE using PyTorch operations for differentiability.
    Simplified version that supports euclidean/cosine distance.
    """
    unique_episodes = torch.unique(episode_ids)
    total_mse = torch.tensor(0.0, device=device)
    total_count = 0

    for ep in unique_episodes:
        # Split into train/test
        test_mask = (episode_ids == ep)
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = features[train_mask]  # (N_tr, d)
        Y_train = actions[train_mask]   # (N_tr, A)
        X_test = features[test_mask]    # (N_te, d)
        Y_test = actions[test_mask]     # (N_te, A)

        # Compute pairwise distances: (N_te, N_tr)
        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            X_test_norm = F.normalize(X_test, p=2, dim=1)
            X_train_norm = F.normalize(X_train, p=2, dim=1)
            cosine_sim = torch.mm(X_test_norm, X_train_norm.T)
            dist_matrix = 1.0 - cosine_sim
        else:
            # For euclidean, whiten, proj, pls: use euclidean distance
            test_norm = torch.sum(X_test**2, dim=1, keepdim=True)  # (N_te, 1)
            train_norm = torch.sum(X_train**2, dim=1, keepdim=True).T  # (1, N_tr)
            cross_term = torch.mm(X_test, X_train.T)  # (N_te, N_tr)
            dist_matrix = torch.sqrt(torch.clamp(test_norm + train_norm - 2 * cross_term, min=1e-8))

        # Get k nearest neighbors for each test point
        k_eff = min(k, X_train.shape[0])
        topk_distances, topk_indices = torch.topk(dist_matrix, k_eff, dim=1, largest=False)

        # Weighted average of k nearest neighbors using inverse distance weighting
        eps = 1e-8
        weights = 1.0 / (topk_distances + eps)  # (N_te, k)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)  # normalize

        # Get neighbor actions and compute weighted average
        neighbor_actions = Y_train[topk_indices]  # (N_te, k, A)
        predicted_actions = torch.sum(weights.unsqueeze(-1) * neighbor_actions, dim=1)  # (N_te, A)

        # Compute MSE for this episode
        mse_ep = torch.mean((predicted_actions - Y_test) ** 2)
        total_mse += mse_ep * test_mask.sum().float()
        total_count += test_mask.sum().item()

    return total_mse / max(total_count, 1)


def compute_loeo_mse_torch_with_metric(
    features: torch.Tensor,  # (N, d) - weighted features
    actions: torch.Tensor,   # (N, A)
    episode_ids: torch.Tensor,  # (N,)
    frame_ids: torch.Tensor,    # (N,)
    k: int,
    metric: str,
    temp_excl_w: int,
    device: torch.device,
    global_metric_ctx: dict = None,  # Global metric context from fullspace
    head_weights: torch.Tensor = None,  # (H,) head weights for slicing metric
) -> torch.Tensor:
    """
    Compute LOEO MSE using PyTorch operations with global metric context.
    This ensures consistency with other head selection methods.
    """
    unique_episodes = torch.unique(episode_ids)
    total_mse = torch.tensor(0.0, device=device)
    total_count = 0

    for ep in unique_episodes:
        # Split into train/test
        test_mask = (episode_ids == ep)
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = features[train_mask]  # (N_tr, d)
        Y_train = actions[train_mask]   # (N_tr, A)
        X_test = features[test_mask]    # (N_te, d)
        Y_test = actions[test_mask]     # (N_te, A)

        # Apply global metric transform if available
        if global_metric_ctx is not None:
            if metric == "whiten" and "Wg" in global_metric_ctx and "mu" in global_metric_ctx:
                # Apply whitening transform
                Wg = torch.from_numpy(global_metric_ctx["Wg"]).float().to(device)  # (d, r)
                mu = torch.from_numpy(global_metric_ctx["mu"]).float().to(device)  # (d,)

                # For weighted features, we need to slice the transform to match the feature dimension
                d_feat = X_train.shape[1]  # actual feature dimension
                if d_feat <= Wg.shape[0]:
                    Wg_slice = Wg[:d_feat, :]  # (d_feat, r)
                    mu_slice = mu[:d_feat]     # (d_feat,)

                    X_train_centered = X_train - mu_slice.unsqueeze(0)
                    X_test_centered = X_test - mu_slice.unsqueeze(0)
                    X_train = torch.mm(X_train_centered, Wg_slice)
                    X_test = torch.mm(X_test_centered, Wg_slice)

            elif metric in {"proj", "pls"} and "W" in global_metric_ctx:
                # Apply projection transform
                W = torch.from_numpy(global_metric_ctx["W"]).float().to(device)  # (d, A)

                # For weighted features, we need to slice the transform
                d_feat = X_train.shape[1]  # actual feature dimension
                if d_feat <= W.shape[0]:
                    W_slice = W[:d_feat, :]  # (d_feat, A)
                    X_train = torch.mm(X_train, W_slice)
                    X_test = torch.mm(X_test, W_slice)

        # Compute pairwise distances: (N_te, N_tr)
        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            X_test_norm = F.normalize(X_test, p=2, dim=1)
            X_train_norm = F.normalize(X_train, p=2, dim=1)
            cosine_sim = torch.mm(X_test_norm, X_train_norm.T)
            dist_matrix = 1.0 - cosine_sim
        else:
            # For euclidean, whiten, proj, pls: use euclidean distance after transform
            test_norm = torch.sum(X_test**2, dim=1, keepdim=True)  # (N_te, 1)
            train_norm = torch.sum(X_train**2, dim=1, keepdim=True).T  # (1, N_tr)
            cross_term = torch.mm(X_test, X_train.T)  # (N_te, N_tr)
            dist_matrix = torch.sqrt(torch.clamp(test_norm + train_norm - 2 * cross_term, min=1e-8))

        # Get k nearest neighbors for each test point
        k_eff = min(k, X_train.shape[0])
        topk_distances, topk_indices = torch.topk(dist_matrix, k_eff, dim=1, largest=False)

        # Weighted average of k nearest neighbors using inverse distance weighting
        eps = 1e-8
        weights = 1.0 / (topk_distances + eps)  # (N_te, k)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)  # normalize

        # Get neighbor actions and compute weighted average
        neighbor_actions = Y_train[topk_indices]  # (N_te, k, A)
        predicted_actions = torch.sum(weights.unsqueeze(-1) * neighbor_actions, dim=1)  # (N_te, A)

        # Compute MSE for this episode
        mse_ep = torch.mean((predicted_actions - Y_test) ** 2)
        total_mse += mse_ep * test_mask.sum().float()
        total_count += test_mask.sum().item()

    return total_mse / max(total_count, 1)


__all__ = [
    "rank_single_heads_per_k",
    "rank_single_heads",
    "simple_topk_select",
    "greedy_forward_select",
    "reinforce_select_heads",
    "learn_head_weights",
    "compute_loeo_mse_torch",
    "compute_loeo_mse_torch_with_metric",
]
