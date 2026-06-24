"""
KNN visualization and analysis utilities
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import List, Optional, Iterable

from .data import build_dataset
from .inference import build_feat_subset
from .metrics import pairwise_dist, build_global_metric_ctx_for_heads


def _loeo_train_mask(episode_ids: np.ndarray, frame_ids: np.ndarray, query_idx: int, temp_excl_window: int):
    """Return mask of indices allowed as training candidates for query_idx (LOEO or LOFO+window)"""
    same_episode = (episode_ids == episode_ids[query_idx])
    unique_episodes = np.unique(episode_ids)
    
    if len(unique_episodes) == 1:
        # Single episode: exclude self and temporal neighbors(LOFO)
        too_close = np.abs(frame_ids - frame_ids[query_idx]) <= temp_excl_window
        mask = ~(too_close | (np.arange(len(episode_ids)) == query_idx))
    else:
        # Multiple episodes: exclude same episode(LOO)
        mask = ~same_episode
    return mask


def _knn_indices(query_vec: np.ndarray, base_matrix: np.ndarray, k: int, metric: str, metric_ctx: Optional[dict] = None):
    """KNN in the chosen feature subspace (or action space in caller)"""
    distances = pairwise_dist(query_vec, base_matrix, metric, metric_ctx)
    k_effective = min(k, len(distances))
    indices = np.argpartition(distances, k_effective)[:k_effective]
    indices = indices[np.argsort(distances[indices])]
    return indices, distances


def _action_distance(query_action: np.ndarray, action_bank: np.ndarray, metric: str):
    """Compute action distance"""
    if metric == "mse":          # same as head selection metric: mean squared error
        diff = action_bank - query_action[None, :]
        return ((diff * diff).mean(axis=1))
    elif metric == "euclidean":
        diff = action_bank - query_action[None, :]
        return np.sqrt((diff * diff).sum(axis=1))
    elif metric == "cosine":
        query_norm = query_action / (np.linalg.norm(query_action) + 1e-8)
        action_norm = action_bank / (np.linalg.norm(action_bank, axis=1, keepdims=True) + 1e-8)
        return 1.0 - (action_norm @ query_norm)
    else:
        raise ValueError("action metric must be 'mse', 'euclidean' or 'cosine'")


def _build_feature_bank_for_heads(attn_h5: str, episodes: List[str], heads: List[int]):
    """Build feature bank with given heads; return feature_bank, action_bank, episode_ids, frame_ids."""
    preprocessed_data = build_dataset(attn_h5, episodes)
    feature_bank = build_feat_subset(preprocessed_data.Xh_red, heads)
    return dict(
        feature_bank=feature_bank.astype(np.float32),
        action_bank=preprocessed_data.Y.astype(np.float32),
        episode_ids=preprocessed_data.ep_ids,
        frame_ids=preprocessed_data.t_ids,
        episodes=preprocessed_data.episodes
    )


def knn_overlap_and_plots(
    attn_h5: str,
    episodes: List[str],
    model=None,  # KnnRegModel
    heads_override: Optional[List[int]] = None,   # if None use model.heads；use my own heads if provided(SAV best)
    ks=(5, 10, 20),
    feature_metric: str = None,
    action_metric: str = "mse",                   # heads are ranked by LOEO MSE
    output_dir: str = "knn_viz",
    max_points_for_scatter: int = 400,
    use_tsne: bool = True,
    temp_excl_window: int = 30,
    global_metric_fullspace: bool = True,
    whiten_pca_dim: int = 256,
    proj_alpha: float = 1e-2,
    pls_components: int = 32,
    total_heads: int = 144
):
    """Compute overlap@k and save visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select heads & metric
    if heads_override is not None:
        heads = heads_override
        feature_metric = feature_metric or (model.metric if model is not None else "euclidean")
    else:
        assert model is not None, "Either provide model or heads_override."
        heads = model.heads
        feature_metric = feature_metric or model.metric

    bank_data = _build_feature_bank_for_heads(attn_h5, episodes, heads)
    feature_bank, action_bank = bank_data["feature_bank"], bank_data["action_bank"]
    episode_ids, frame_ids = bank_data["episode_ids"], bank_data["frame_ids"]

    # If no model, fit global metric context for supervised/whiten metrics
    metric_context = None
    if model is None and feature_metric in {"whiten", "proj", "pls"}:
        # Reuse fullspace fit + slicing strategy with local caches
        preprocessed_data = build_dataset(attn_h5, episodes)
        _metric_cache = {}
        _metric_fullspace_cache = {}
        if global_metric_fullspace:
            full_context = build_global_metric_ctx_for_heads(
                preprocessed_data, list(range(total_heads)), feature_metric,
                whiten_pca_dim, proj_alpha, pls_components,
                global_metric_fullspace, total_heads, _metric_cache, _metric_fullspace_cache,
            )
            dim_per_head = preprocessed_data.Xh_red.shape[-1]
            row_indices = []
            for h in sorted(heads):
                start = h*dim_per_head; row_indices.extend(range(start, start+dim_per_head))
            if feature_metric == "whiten":
                metric_context = {"Wg": full_context["Wg"][row_indices, :], "mu": full_context["mu"][row_indices]}
            else:
                metric_context = {"W": full_context["W"][row_indices, :]}
        else:
            metric_context = build_global_metric_ctx_for_heads(
                preprocessed_data, heads, feature_metric,
                whiten_pca_dim, proj_alpha, pls_components,
                global_metric_fullspace, total_heads, _metric_cache, _metric_fullspace_cache,
            )

    N = feature_bank.shape[0]
    ks = tuple(sorted(set([k for k in ks if k > 0])))

    # Accumulators for statistics
    overlaps = {k: [] for k in ks}

    # Visualize a subset of query frames (random 50) for clarity; stats can be computed on all
    rng = np.random.RandomState(0)
    viz_indices = rng.choice(N, size=min(50, N), replace=False)

    # Full overlap@k statistics
    for i in range(N):
        mask = _loeo_train_mask(episode_ids, frame_ids, i, temp_excl_window)
        train_features, train_actions = feature_bank[mask], action_bank[mask]

        # feature-KNN
        feature_indices, feature_distances = _knn_indices(
            feature_bank[i], train_features, max(ks), feature_metric,
            model.metric_ctx if model is not None else metric_context
        )

        # action-KNN
        action_distances = _action_distance(action_bank[i], train_actions, action_metric)
        action_indices = np.argpartition(action_distances, max(ks))[:max(ks)]
        action_indices = action_indices[np.argsort(action_distances[action_indices])]

        # accumulate overlap@k
        feature_sets = {k: set(feature_indices[:k].tolist()) for k in ks}
        action_sets = {k: set(action_indices[:k].tolist()) for k in ks}
        for k in ks:
            intersection = len(feature_sets[k].intersection(action_sets[k]))
            overlaps[k].append(intersection / float(k))

        # visualize selected queries
        if i in viz_indices:
            # For clarity, only keep top max_points_for_scatter by feature distance
            current_overlaps = []
            for k in ks:
                feature_set_k = set(feature_indices[:k].tolist())
                action_set_k = set(action_indices[:k].tolist())
                current_overlaps.append(len(feature_set_k & action_set_k) / float(k))

            # title: current query + global mean
            title = (
                f"query #{i} | overlap_this@{ks}=" + ",".join([f"{v:.2f}" for v in current_overlaps])
            )            
            candidates = np.argpartition(feature_distances, min(len(feature_distances)-1, max_points_for_scatter-1))[:min(len(feature_distances), max_points_for_scatter)]
            candidates = candidates[np.argsort(feature_distances[candidates])]
            distances_feature = feature_distances[candidates]
            distances_action = action_distances[candidates]

            # 1) dist-dist scatter: x=feature distance, y=action distance
            plt.figure(figsize=(5,4))
            plt.scatter(distances_feature, distances_action, s=12, alpha=0.7, label="candidates")
            # highlight top-k
            for k in ks:
                top_candidates = candidates[:k]
                plt.scatter(feature_distances[top_candidates], action_distances[top_candidates], s=22, alpha=0.9, marker='x', label=f"feat-top{k}")
            plt.xlabel("feature distance"); plt.ylabel("action distance")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"q{i:05d}_dist_scatter.png"))
            plt.close()

            # 2) optional: t-SNE (query + union of two top-K sets)
            if use_tsne:
                union = set()
                for k in ks:
                    union.update(feature_indices[:k].tolist())
                    union.update(action_indices[:k].tolist())
                union = np.array(sorted(list(union)))
                X_tsne = np.vstack([feature_bank[i][None, :], train_features[union]])  # (1+M, D)
                labels = np.zeros(1+len(union), dtype=int)
                labels[0] = 2  # query point
                # annotate feat-topK vs act-topK
                label_map = np.zeros(len(union), dtype=int)  # 0=others,1=feat-top,3=act-top,4=both
                feature_set = set(feature_indices[:max(ks)].tolist())
                action_set = set(action_indices[:max(ks)].tolist())
                for j, u in enumerate(union):
                    if u in feature_set and u in action_set: 
                        label_map[j] = 4  # both
                    elif u in feature_set: 
                        label_map[j] = 1
                    elif u in action_set:  
                        label_map[j] = 3
                    else:               
                        label_map[j] = 0
                # t-SNE
                embedding = TSNE(n_components=2, perplexity=min(30, len(union)//3 + 5), init="pca", random_state=0).fit_transform(X_tsne)
                plt.figure(figsize=(5,4))
                # others
                mask0 = (label_map == 0)
                plt.scatter(embedding[1:][mask0,0], embedding[1:][mask0,1], s=10, alpha=0.4, label="others")
                # feat-only
                mask1 = (label_map == 1)
                plt.scatter(embedding[1:][mask1,0], embedding[1:][mask1,1], s=24, alpha=0.8, marker='x', label="feat-KNN")
                # act-only
                mask3 = (label_map == 3)
                plt.scatter(embedding[1:][mask3,0], embedding[1:][mask3,1], s=24, alpha=0.8, marker='^', label="act-KNN")
                # both
                mask4 = (label_map == 4)
                plt.scatter(embedding[1:][mask4,0], embedding[1:][mask4,1], s=28, alpha=0.9, marker='s', label="both")
                # query
                plt.scatter(embedding[0,0], embedding[0,1], s=40, alpha=1.0, marker='*', label="query")
                plt.axis('off'); plt.legend(loc='best', fontsize=8)
                plt.title(f"t-SNE around query #{i}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"q{i:05d}_tsne.png"))
                plt.close()

    # Summary statistics
    summary = {f"overlap@{k}": (float(np.mean(overlaps[k])), float(np.std(overlaps[k]))) for k in ks}
    print("KNN overlap summary:", summary)
    return {
        "summary": summary,          # every k (mean, std)
        "per_query": overlaps,       # every k (per-query overlap list)
        "output_dir": output_dir
    }
