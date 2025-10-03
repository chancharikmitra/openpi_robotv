# knn_reg_with_heads.py
# KNN regression on per-frame activations (18,8,256) vs per-frame pi_droid_action
# Leave-One-Episode-Out MSE for head selection. Modes: topk / best_add / reinforce

import h5py, random, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Set
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# --------------------------
# Basic configuration
# --------------------------
NUM_LAYERS   = 18
HEADS_PER_L  = 8
H            = NUM_LAYERS * HEADS_PER_L   # 144
D_HEAD       = 256

# Metric scope:
#   - "per_episode": fit metric on train subset for each fold (strict, slower)
#   - "global"     : fit once on all samples for fixed heads and reuse (faster)
METRIC_SCOPE = "global"


# ---- Metric learning hyperparameters ----
WHITEN_PCA_DIM = 256     # truncated PCA dim for whiten
PROJ_ALPHA     = 1e-2    # Ridge regularization for proj
PLS_COMPONENTS = 32      # upper bound for PLS components


USE_PCA      = False
USE_ZSCORE   = False             # per-head z-score normalization
PCA_D        = 32                # per-head PCA dim; disable by setting USE_PCA=False
DIST_METRIC  = "euclidean"         # "cosine" | "euclidean" | "whiten" | "proj" | "pls"
K_GRID       = [10,20,30,40]        # candidate k for KNN
TEMP_EXCL_W  = 30                # LOFO temporal exclusion window (±W frames)

# ---- Head selection mode and target ----
HEAD_SELECTION_MODE = "topk"      # "topk" | "best_add" | "reinforce" | "learn_weights"
TARGET_HEADS        = 20          # target number of heads (not used for learn_weights)
# ---- Optional: exclude specific heads from final selection ----
EXCLUDED_HEADS: List[int] = []    # any head indices to exclude from final selection

# Utilities to normalize excluded head formats
def _to_flat_head_index(item) -> Optional[int]:
    """
    Accepts formats: int (flat index), (layer, head), {layer, head}, "layer,head", "(layer, head)".
    Returns flat index in [0, H) or None if invalid.
    """
    try:
        # Flat int index
        if isinstance(item, int):
            return int(item) if 0 <= int(item) < H else None
        # Pair formats: tuple/list
        if isinstance(item, (tuple, list)) and len(item) == 2:
            layer = int(item[0])
            head = int(item[1])
        # Dict format
        elif isinstance(item, dict):
            if "layer" in item and "head" in item:
                layer = int(item["layer"]) ; head = int(item["head"])
            else:
                layer = int(item.get("l", item.get("layer", -1)))
                head  = int(item.get("h", item.get("head", -1)))
        # String format
        elif isinstance(item, str):
            s = item.strip().replace("(", "").replace(")", "")
            if "," in s:
                parts = s.split(",")
                layer = int(parts[0].strip())
                head  = int(parts[1].strip())
            else:
                # as flat integer string
                val = int(s)
                return val if 0 <= val < H else None
        else:
            return None
        if 0 <= layer < NUM_LAYERS and 0 <= head < HEADS_PER_L:
            return layer * HEADS_PER_L + head
        return None
    except Exception:
        return None

def _normalize_excluded_heads(excluded_heads_input) -> Set[int]:
    if excluded_heads_input is None:
        excluded_heads_input = EXCLUDED_HEADS
    result: Set[int] = set()
    for item in excluded_heads_input:
        flat = _to_flat_head_index(item)
        if flat is not None:
            result.add(flat)
    return result

# ---- best_add stopping thresholds ----
MAX_HEADS    = TARGET_HEADS
IMPROVE_EPS  = 0.01               # stop if <1% relative improvement

# ---- REINFORCE hyperparameters ----
RF_ITERS             = 120
RF_BATCH             = 6
RF_LR                = 0.08
RF_ENTROPY_BONUS     = 0.01
RF_EPISODE_SUBSAMPLE = 0.5        # subsample episodes for approximate LOEO; None for full

# ---- Debug and printing options ----
PRINT_HEAD_RANKINGS  = True      # print detailed head rankings and MSE statistics


from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
# Import commonly used utilities directly for readability
from openpi.knn.utils import (
    build_dataset, build_feat_subset,
    rank_single_heads, simple_topk_select, greedy_forward_select, reinforce_select_heads, learn_head_weights,
    get_action_labels, HeadPreprocessor, load_episode_frames,
)
from openpi.knn.metrics import build_global_metric_ctx_for_heads
from openpi.knn.eval import evaluate_model_on_h5, evaluate_leave_one_episode_out, predict_episode, predict_episode_from_activation
from openpi.knn.viz import knn_overlap_and_plots

# Global metric caches for METRIC_SCOPE=="global"
_METRIC_CTX_CACHE = {}
GLOBAL_METRIC_FULLSPACE = True  # fit once in full feature space and slice for subsets
_METRIC_CTX_FULLSPACE: Dict[str, dict] = {}





# --------------------------
# H5 layout assumptions
# Each frame group contains:
#   - "last_token_attn": (18,8,256)
#   - "pi_droid_action": (A,) or scalar
# --------------------------


# --------------------------
# Dataset flattening and per-head preprocessing
# --------------------------


# --------------------------
# Distances and weighted averages
# --------------------------


# --------------------------
# Evaluation: LOEO (or LOFO + temporal exclusion when single episode)
# --------------------------



# --------------------------
# Single-head scoring (used by all selection modes)
# --------------------------


# --------------------------
# Model and inference
# --------------------------
@dataclass
class KnnRegModel:
    heads: List[int]
    k: int
    metric: str
    d_per_head: int
    preproc: List[HeadPreprocessor]
    X_bank: np.ndarray   # (N, |S|*d)
    Y_bank: np.ndarray   # (N, A)
    metric_ctx: Optional[dict] = None
    head_weights: Optional[np.ndarray] = None  # (H,) for learn_weights mode

def fit_knn_reg_with_heads(attn_h5: str, episodes: List[str], selection_mode: str = HEAD_SELECTION_MODE, excluded_heads: Optional[List[int]] = None) -> Tuple[KnnRegModel, Dict]:
    """
    selection_mode:
      - "topk"      : take top-K single-heads by LOEO MSE ranking
      - "best_add"  : greedy forward add until K heads
      - "reinforce" : REINFORCE to optimize head subset
    excluded_heads:
      - heads that must NOT appear in the final selected set
    """
    # Normalize excluded heads (supports (layer, head) pairs)
    excluded_set = _normalize_excluded_heads(excluded_heads)
    # Build preprocessed dataset
    preprocessed_data = build_dataset(
        attn_h5, episodes, use_pca=USE_PCA, use_zscore=USE_ZSCORE, pca_components=PCA_D
    )
    d_per_head = preprocessed_data.Xh_red.shape[-1]

    # 1) Single-head ranking per k
    rankings_per_k = rank_single_heads(
        preprocessed_data.features, preprocessed_data.actions, preprocessed_data.episode_ids, preprocessed_data.frame_ids,
                              K_GRID, DIST_METRIC, TEMP_EXCL_W,
        pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
        pre_obj=preprocessed_data, metric_scope=METRIC_SCOPE,
        use_fullspace=GLOBAL_METRIC_FULLSPACE, H=H,
        metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE,
        print_rankings=PRINT_HEAD_RANKINGS
    )

    # Helper: enforce excluded heads and top-up from rankings
    def _enforce_excluded_and_fill(cur_sel: List[int], target_k: int, best_k: int) -> List[int]:
        # Deduplicate while preserving order, and drop excluded
        filtered = []
        seen = set()
        for h in cur_sel:
            if h in excluded_set:
                continue
            if h in seen:
                continue
            filtered.append(h)
            seen.add(h)
        # Available pool from ranking of best_k
        ranking_list = rankings_per_k.get(int(best_k), [])
        for mse, h in ranking_list:
            if h in excluded_set or h in seen:
                continue
            filtered.append(h)
            seen.add(h)
            if len(filtered) >= target_k:
                break
        return filtered

    # 2) Select heads by mode
    if selection_mode == "topk":
        selected_heads, best_k_value, cross_val_mse, per_k_mse_scores = simple_topk_select(
            preprocessed_data.features, preprocessed_data.actions, preprocessed_data.episode_ids, preprocessed_data.frame_ids,
            rankings_per_k, K_GRID, DIST_METRIC, TEMP_EXCL_W, TARGET_HEADS, 
            pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA, pre_obj=preprocessed_data,
            metric_scope=METRIC_SCOPE, use_fullspace=GLOBAL_METRIC_FULLSPACE,
            H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
        )
        # Enforce exclusion and refill
        target_k = min(TARGET_HEADS, max(0, H - len(excluded_set)))
        selected_heads = _enforce_excluded_and_fill(selected_heads, target_k, best_k_value)
        head_probabilities = None
        # Recompute per-k MSE and CV MSE with final selected heads
        ctx_global = None
        if METRIC_SCOPE == "global" and DIST_METRIC in {"whiten","proj","pls"}:
            ctx_global = build_global_metric_ctx_for_heads(
                preprocessed_data, selected_heads, DIST_METRIC,
                pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
                pls_components=PLS_COMPONENTS, use_fullspace=GLOBAL_METRIC_FULLSPACE,
                H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
            )
        per_k_mse_scores = {}
        for k in K_GRID:
            mse_k = evaluate_leave_one_episode_out(
                preprocessed_data.features, preprocessed_data.actions,
                preprocessed_data.episode_ids, preprocessed_data.frame_ids,
                selected_heads, k, DIST_METRIC, TEMP_EXCL_W,
                pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
                metric_scope=METRIC_SCOPE, metric_ctx_global=ctx_global,
            )
            per_k_mse_scores[int(k)] = float(mse_k)
        cross_val_mse = float(per_k_mse_scores[int(best_k_value)])
    elif selection_mode == "best_add":
        selected_heads, best_k_value, cross_val_mse, per_k_mse_scores = greedy_forward_select(
            preprocessed_data.features, preprocessed_data.actions, preprocessed_data.episode_ids, preprocessed_data.frame_ids,
            rankings_per_k, K_GRID, DIST_METRIC, TEMP_EXCL_W, TARGET_HEADS, 
            pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA, pre_obj=preprocessed_data,
            metric_scope=METRIC_SCOPE, use_fullspace=GLOBAL_METRIC_FULLSPACE,
            H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
        )
        # Enforce exclusion and refill
        target_k = min(TARGET_HEADS, max(0, H - len(excluded_set)))
        selected_heads = _enforce_excluded_and_fill(selected_heads, target_k, best_k_value)
        head_probabilities = None
        # Recompute per-k MSE and CV MSE with final selected heads
        ctx_global = None
        if METRIC_SCOPE == "global" and DIST_METRIC in {"whiten","proj","pls"}:
            ctx_global = build_global_metric_ctx_for_heads(
                preprocessed_data, selected_heads, DIST_METRIC,
                pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
                pls_components=PLS_COMPONENTS, use_fullspace=GLOBAL_METRIC_FULLSPACE,
                H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
            )
        per_k_mse_scores = {}
        for k in K_GRID:
            mse_k = evaluate_leave_one_episode_out(
                preprocessed_data.features, preprocessed_data.actions,
                preprocessed_data.episode_ids, preprocessed_data.frame_ids,
                selected_heads, k, DIST_METRIC, TEMP_EXCL_W,
                pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
                metric_scope=METRIC_SCOPE, metric_ctx_global=ctx_global,
            )
            per_k_mse_scores[int(k)] = float(mse_k)
        cross_val_mse = float(per_k_mse_scores[int(best_k_value)])
    elif selection_mode == "reinforce":
        selected_heads, best_k_value, cross_val_mse, head_probabilities = reinforce_select_heads(
            preprocessed_data, K_target=TARGET_HEADS, iters=RF_ITERS, batch=RF_BATCH,
            lr=RF_LR, k_grid=K_GRID, metric=DIST_METRIC, temp_excl_w=TEMP_EXCL_W,
            entropy_bonus=RF_ENTROPY_BONUS, episode_subsample=RF_EPISODE_SUBSAMPLE
        )
        # Enforce exclusion and refill
        target_k = min(TARGET_HEADS, max(0, H - len(excluded_set)))
        selected_heads = _enforce_excluded_and_fill(selected_heads, target_k, best_k_value)
        # Additionally compute per-k MSE for printing with final selected heads
        per_k_mse_scores = {}
        # Reuse global metric if enabled
        if METRIC_SCOPE == "global" and DIST_METRIC in {"whiten","proj","pls"}:
            ctx_global = build_global_metric_ctx_for_heads(
                preprocessed_data, selected_heads, DIST_METRIC,
                pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
                pls_components=PLS_COMPONENTS, use_fullspace=GLOBAL_METRIC_FULLSPACE,
                H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
            )
        else:
            ctx_global = None
        for k in K_GRID:
            mse_k = evaluate_leave_one_episode_out(
                preprocessed_data.features, preprocessed_data.actions,
                preprocessed_data.episode_ids, preprocessed_data.frame_ids,
                selected_heads, k, DIST_METRIC, TEMP_EXCL_W,
                pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
                metric_scope=METRIC_SCOPE, metric_ctx_global=ctx_global,
            )
            per_k_mse_scores[int(k)] = float(mse_k)
    elif selection_mode == "learn_weights":
        head_weights, best_k_value, cross_val_mse, per_k_mse_scores = learn_head_weights(
            preprocessed_data.features, preprocessed_data.actions, preprocessed_data.episode_ids, preprocessed_data.frame_ids,
            K_GRID, DIST_METRIC, TEMP_EXCL_W,
            pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
            pre_obj=preprocessed_data, metric_scope=METRIC_SCOPE,
            use_fullspace=GLOBAL_METRIC_FULLSPACE, H=H,
            metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE,
            lr=0.01, iters=100, weight_decay=1e-4
        )
        # For weighted approach, we use all heads
        selected_heads = list(range(H))
        # Apply exclusion by zeroing weights and renormalizing
        if isinstance(head_weights, np.ndarray) and head_weights.shape[0] == H and len(excluded_set) > 0:
            head_weights = head_weights.copy()
            for h in excluded_set:
                if 0 <= h < H:
                    head_weights[h] = 0.0
            s = float(head_weights.sum())
            if s > 0:
                head_weights = head_weights / s
        head_probabilities = head_weights  # Store learned weights
    else:
        raise ValueError("selection_mode must be one of {'topk','best_add','reinforce','learn_weights'}")

    # 3) Build the final feature bank (all samples)
    if selection_mode == "learn_weights":
        # For weighted mode, we need to store per-head features and apply weights during inference
        feature_bank = preprocessed_data.Xh_red.reshape(preprocessed_data.Xh_red.shape[0], -1)  # (N, H*d)
    else:
        feature_bank = build_feat_subset(preprocessed_data.Xh_red, selected_heads)
    
    # 4) Learn metric context for inference-time distance computation
    learned_metric_context = None
    # Use preprocessed_data (contains Xh_red/Y) to build final metric context
    if DIST_METRIC == "whiten":
        learned_metric_context = build_global_metric_ctx_for_heads(
            preprocessed_data, selected_heads, "whiten",
            pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
            pls_components=PLS_COMPONENTS, use_fullspace=GLOBAL_METRIC_FULLSPACE,
            H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
        )
    elif DIST_METRIC == "proj":
        learned_metric_context = build_global_metric_ctx_for_heads(
            preprocessed_data, selected_heads, "proj",
            pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
            pls_components=PLS_COMPONENTS, use_fullspace=GLOBAL_METRIC_FULLSPACE,
            H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
        )
    elif DIST_METRIC == "pls":
        learned_metric_context = build_global_metric_ctx_for_heads(
            preprocessed_data, selected_heads, "pls",
            pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
            pls_components=PLS_COMPONENTS, use_fullspace=GLOBAL_METRIC_FULLSPACE,
            H=H, metric_ctx_cache=_METRIC_CTX_CACHE, metric_ctx_fullspace=_METRIC_CTX_FULLSPACE
        )

    trained_model = KnnRegModel(
        heads=selected_heads,
        k=best_k_value,
        metric=DIST_METRIC,
        d_per_head=d_per_head,
        preproc=preprocessed_data.preproc,
        X_bank=feature_bank.astype(np.float32),
        Y_bank=preprocessed_data.Y.astype(np.float32),
        head_weights=head_weights if selection_mode == "learn_weights" else None,
    )
    trained_model.metric_ctx = learned_metric_context
    training_info = {
        "selection_mode": selection_mode,
        "cv_mse": cross_val_mse,
        "per_k_mse": per_k_mse_scores,
        "rankings_per_k": rankings_per_k,        # {k: [(mse, head_id), ...]}
        "head_probabilities": head_probabilities,      # for reinforce
        "episodes": preprocessed_data.episodes,
        "d_per_head": d_per_head,
        "use_pca": USE_PCA,
        "pca_d": PCA_D,
        "metric": DIST_METRIC,
        "k_grid": list(K_GRID),
        "temp_excl_w": TEMP_EXCL_W,
        "selected_heads": selected_heads,
        "best_k": best_k_value,
        "target_heads": TARGET_HEADS,
        "excluded_heads": list(sorted(excluded_set)),
    }
    # Print per-k MSE
    if isinstance(per_k_mse_scores, dict) and len(per_k_mse_scores) > 0:
        per_k_str = ", ".join([f"k={k}: {mse:.6f}" for k, mse in sorted(per_k_mse_scores.items())])
        print(f"Per-k MSE -> {per_k_str}")

    # Print training configuration summary
    print("================ Train Config ================")
    print(f"Selection mode       : {selection_mode}")
    print(f"Target heads         : {TARGET_HEADS}")
    print(f"K grid               : {K_GRID}")
    print(f"Best k               : {best_k_value}")
    print(f"Metric               : {DIST_METRIC} (scope={METRIC_SCOPE}, fullspace={GLOBAL_METRIC_FULLSPACE})")
    print(f"Use PCA              : {USE_PCA} (PCA_D={PCA_D})")
    print(f"Use Z-score          : {USE_ZSCORE}")
    if DIST_METRIC == "whiten":
        print(f"Whiten PCA dim       : {WHITEN_PCA_DIM}")
    if DIST_METRIC == "proj":
        print(f"Ridge alpha          : {PROJ_ALPHA}")
    if DIST_METRIC == "pls":
        print(f"PLS components       : {PLS_COMPONENTS}")
    print(f"Temporal excl window : {TEMP_EXCL_W}")
    if len(excluded_set) > 0:
        print(f"Excluded heads       : {sorted(list(excluded_set))}")
    print(f"Selected heads ({len(selected_heads)}): {selected_heads}")
    print(f"LOEO-CV MSE          : {cross_val_mse:.6f}")
    print("==============================================")
    return trained_model, training_info













def evaluate_custom_heads_on_k_grid(
    attn_h5: str, 
    episodes: List[str], 
    custom_heads: List[int], 
    k_grid: List[int] = None,
    metric: str = DIST_METRIC,
    temp_excl_w: int = TEMP_EXCL_W,
    use_pca: bool = USE_PCA,
    use_zscore: bool = USE_ZSCORE,
    pca_d: int = PCA_D,
    whiten_pca_dim: int = WHITEN_PCA_DIM,
    proj_alpha: float = PROJ_ALPHA,
    pls_components: int = PLS_COMPONENTS,
    metric_scope: str = METRIC_SCOPE,
    use_fullspace: bool = GLOBAL_METRIC_FULLSPACE,
    print_results: bool = True
) -> Dict:
    """
    评估指定头列表在不同k值上的MSE性能
    
    Args:
        attn_h5: 注意力数据文件路径
        episodes: 要评估的episode列表
        custom_heads: 要评估的头索引列表
        k_grid: k值网格，默认使用全局K_GRID
        其他参数: 使用全局配置的默认值
        
    Returns:
        包含每个k值MSE结果的字典
    """
    if k_grid is None:
        k_grid = K_GRID
    
    print(f"正在评估自定义头列表: {custom_heads}")
    print(f"头数量: {len(custom_heads)}")
    print(f"K值网格: {k_grid}")
    print(f"距离度量: {metric}")
    print("=" * 50)
    
    # 1) 构建预处理数据集
    preprocessed_data = build_dataset(
        attn_h5, episodes, 
        use_pca=use_pca, 
        use_zscore=use_zscore, 
        pca_components=pca_d
    )
    
    # 2) 构建全局度量上下文（如果需要）
    ctx_global = None
    if metric_scope == "global" and metric in {"whiten", "proj", "pls"}:
        ctx_global = build_global_metric_ctx_for_heads(
            preprocessed_data, custom_heads, metric,
            pca_dim=whiten_pca_dim, proj_alpha=proj_alpha,
            pls_components=pls_components, use_fullspace=use_fullspace,
            H=H, metric_ctx_cache={}, metric_ctx_fullspace={}
        )
    
    # 3) 对每个k值评估MSE
    per_k_mse = {}
    best_mse = float('inf')
    best_k = None
    
    for k in k_grid:
        mse = evaluate_leave_one_episode_out(
            preprocessed_data.features, preprocessed_data.actions,
            preprocessed_data.episode_ids, preprocessed_data.frame_ids,
            heads=custom_heads, k=k, metric=metric, temp_excl_w=temp_excl_w,
            pca_dim=whiten_pca_dim, proj_alpha=proj_alpha,
            metric_scope=metric_scope, metric_ctx_global=ctx_global,
        )
        per_k_mse[int(k)] = float(mse)
        
        if mse < best_mse:
            best_mse = mse
            best_k = k
        
        if print_results:
            print(f"k={k:2d}: MSE = {mse:.6f}")
    
    if print_results:
        print("=" * 50)
        print(f"最佳k值: {best_k}, 最佳MSE: {best_mse:.6f}")
        print("=" * 50)
    
    return {
        "per_k_mse": per_k_mse,
        "best_k": best_k,
        "best_mse": best_mse,
        "custom_heads": custom_heads,
        "k_grid": list(k_grid),
        "metric": metric,
        "episodes": episodes,
        "num_heads": len(custom_heads)
    }


if __name__ == "__main__":
    ATTN_H5 = "/home/yusenluo/openpi_robotv/attention_dataset/place_marker_in_mug_first_20_heads.h5" #attention_dataset/PI0DROID_place_marker_in_mug_200_state_first_action.h5
    # ATTN_H5_EVAL = "/scr2/yusenluo/openpi_robotv/src/openpi/pick_eval_attention_last_token_keyframe_positive_with_action.h5"
    with h5py.File(ATTN_H5, "r") as f:
        all_eps = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]


    # with h5py.File(ATTN_H5_EVAL, "r") as f:
    #     eval_eps = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()] #

    # 示例1: 训练完整模型
    # model, info = fit_knn_reg_with_heads(ATTN_H5, all_eps, selection_mode=HEAD_SELECTION_MODE)
    
    # 示例2: 评估自定义头列表
    # 你可以在这里指定你想要评估的头列表
    custom_heads_example = [92, 139, 142, 91, 23, 105, 31, 95, 113, 119,120, 88, 114, 32] #[10, 19, 92, 139, 142, 91, 23, 105, 31, 95,12, 13, 5, 113, 119, 9, 120, 88, 114, 32]  # 示例头列表
    
    print("\n" + "="*60)
    print("评估自定义头列表性能")
    print("="*60)
    
    custom_results = evaluate_custom_heads_on_k_grid(
        attn_h5=ATTN_H5,
        episodes=all_eps,
        custom_heads=custom_heads_example,
        k_grid=K_GRID,  # 或者你可以指定自定义的k值列表，如 [10, 20, 30]
        print_results=True
    )
    
    # 你也可以这样使用：
    # 
    # # 评估最好的头
    # best_heads = [53, 75, 23, 20, 52]
    # results_best = evaluate_custom_heads_on_k_grid(ATTN_H5, all_eps, best_heads)
    # 
    # # 评估最差的头  
    # worst_heads = [3, 141, 143, 138, 142]
    # results_worst = evaluate_custom_heads_on_k_grid(ATTN_H5, all_eps, worst_heads)
    # 
    # # 比较结果
    # print(f"最好头的最佳MSE: {results_best['best_mse']:.6f}")
    # print(f"最差头的最佳MSE: {results_worst['best_mse']:.6f}")
    #

    # print("Learned weights:", model.head_weights)
    # print("Head probabilities:", info["head_probabilities"])
    # Evaluate on a separate test set (optional)
    # evaluation_results = evaluate_model_on_h5(ATTN_H5_EVAL, eval_eps, model, ks=K_GRID)
    # print("evaluation results:", evaluation_results["per_k_mse"], evaluation_results["overall_mse"])

    # Inference: predict actions (via KNN) frame-by-frame for an episode
    # test_ep = all_eps[0]
    # pred_actions = predict_episode(ATTN_H5, test_ep, model)
    # print("Pred shape:", pred_actions.shape)

    # # Inference: predict actions (via KNN) given activations
    # attn_frame = np.random.rand(18, 8, 256)
    # print("Attn frame shape:", attn_frame.shape)
    # action = predict_episode_from_activation(attn_frame, model)
    # print("Pred shape:", action.shape)

    # Example: Visualize neighbors for the trained model (optional)
    # diag = knn_overlap_and_plots(
    #     attn_h5=ATTN_H5,
    #     episodes=all_eps,
    #     model=model,
    #     ks=K_GRID,
    #     feature_metric=model.metric,
    #     action_metric="mse",
    #     output_dir="knn_viz_debug_cosine_on_robot",
    #     max_points_for_scatter=400,
    #     use_tsne=False,
    #     temp_excl_window=TEMP_EXCL_W,
    # )


    # ------------------------------------------------------------------
    # Use-case A: ALL HEADS → compute LOEO MSE and predict via a temp KNN model
    # ------------------------------------------------------------------
    # pre_all = build_dataset(ATTN_H5, all_eps, use_pca=USE_PCA, use_zscore=USE_ZSCORE, pca_components=PCA_D)
    # all_heads = list(range(H))
    # # Build global metric context if using whiten/proj/pls
    # ctx_global_all = None
    # if DIST_METRIC in {"whiten", "proj", "pls"}:
    #     ctx_global_all = build_global_metric_ctx_for_heads(
    #         pre_all, all_heads, DIST_METRIC,
    #         pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA, pls_components=PLS_COMPONENTS,
    #         use_fullspace=GLOBAL_METRIC_FULLSPACE, H=H,
    #         metric_ctx_cache={}, metric_ctx_fullspace={}
    #     )
    # # 1) LOEO MSE on train (with correct context)
    # mse_all = evaluate_leave_one_episode_out(
    #     pre_all.features, pre_all.actions, pre_all.episode_ids, pre_all.frame_ids,
    #     heads=all_heads, k=max(K_GRID), metric=DIST_METRIC, temp_excl_w=TEMP_EXCL_W,
    #     pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
    #     metric_scope=("global" if ctx_global_all is not None else "per_episode"),
    #     metric_ctx_global=ctx_global_all,
    # )
    # print(f"[ALL HEADS] LOEO-CV MSE={mse_all:.6f} (k={max(K_GRID)})")
    # # 2) Predict on a chosen episode using a temp model
    # X_bank_all = build_feat_subset(pre_all.Xh_red, all_heads).astype(np.float32)
    # Y_bank_all = pre_all.Y.astype(np.float32)
    # temp_model_all = KnnRegModel(
    #     heads=all_heads,
    #     k=max(K_GRID),
    #     metric=DIST_METRIC,
    #     d_per_head=pre_all.Xh_red.shape[-1],
    #     preproc=pre_all.preproc,
    #     X_bank=X_bank_all,
    #     Y_bank=Y_bank_all,
    #     metric_ctx=ctx_global_all,
    # )
    # pred_actions_all = predict_episode(ATTN_H5, all_eps[0], temp_model_all)
    # print("Pred (ALL HEADS) shape:", pred_actions_all.shape)

    # ------------------------------------------------------------------
    # Use-case B: CUSTOM HEAD SET → compute LOEO MSE and predict via a temp KNN model
    # ------------------------------------------------------------------
    # custom_heads = [0, 1, 2, 3, 10, 20]  # example
    #sav_worst_heads = [3, 141, 143, 138, 142, 6, 140, 137, 136, 135, 139, 0, 132, 134, 130, 131, 128, 4, 133, 129]
    #sav_best_heads = [53, 75, 23, 20, 52, 71, 47, 57, 55, 49, 58, 51, 43, 8, 86, 92, 72, 65, 24, 97]
    # pre_custom = pre_all  # reuse the same preprocessed dataset if applicable
    # ctx_global_custom = None
    # if DIST_METRIC in {"whiten", "proj", "pls"}:
    #     ctx_global_custom = build_global_metric_ctx_for_heads(
    #         pre_custom, custom_heads, DIST_METRIC,
    #         pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA, pls_components=PLS_COMPONENTS,
    #         use_fullspace=GLOBAL_METRIC_FULLSPACE, H=H,
    #         metric_ctx_cache={}, metric_ctx_fullspace={}
    #     )
    # # 1) LOEO MSE on train (with correct context)
    # mse_custom = evaluate_leave_one_episode_out(
    #     pre_custom.features, pre_custom.actions, pre_custom.episode_ids, pre_custom.frame_ids,
    #     heads=custom_heads, k=max(K_GRID), metric=DIST_METRIC, temp_excl_w=TEMP_EXCL_W,
    #     pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
    #     metric_scope=("global" if ctx_global_custom is not None else "per_episode"),
    #     metric_ctx_global=ctx_global_custom,
    # )
    # print(f"[CUSTOM HEADS] LOEO-CV MSE={mse_custom:.6f} (k={max(K_GRID)})")
    # # 2) Predict on a chosen episode using a temp model
    # X_bank_custom = build_feat_subset(pre_custom.Xh_red, custom_heads).astype(np.float32)
    # Y_bank_custom = pre_custom.Y.astype(np.float32)
    # temp_model_custom = KnnRegModel(
    #     heads=custom_heads,
    #     k=max(K_GRID),
    #     metric=DIST_METRIC,
    #     d_per_head=pre_custom.Xh_red.shape[-1],
    #     preproc=pre_custom.preproc,
    #     X_bank=X_bank_custom,
    #     Y_bank=Y_bank_custom,
    #     metric_ctx=ctx_global_custom,
    # )
    # pred_actions_custom = predict_episode(ATTN_H5, all_eps[0], temp_model_custom)
    # print("Pred (CUSTOM HEADS) shape:", pred_actions_custom.shape)
    # print(f"[ALL HEADS] LOEO-CV MSE={mse_all:.6f} (k={max(K_GRID)})")
    # # Visualization for all heads (metric context is built internally if needed)
    # diag_all = knn_overlap_and_plots(
#     attn_h5=ATTN_H5, episodes=all_eps,
#     model=None, heads_override=all_heads,
    #     ks=tuple(K_GRID), feature_metric=DIST_METRIC, action_metric="mse",
    #     output_dir="knn_viz_all_heads", use_tsne=False, temp_excl_window=TEMP_EXCL_W,
    #     global_metric_fullspace=GLOBAL_METRIC_FULLSPACE,
    #     whiten_pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA, pls_components=PLS_COMPONENTS,
    #     total_heads=H,
    # )

#     # your own heads set
#     sav_worst_heads = [3, 141, 143, 138, 142, 6, 140, 137, 136, 135, 139, 0, 132, 134, 130, 131, 128, 4, 133, 129]
#     sav_best_heads = [53, 75, 23, 20, 52, 71, 47, 57, 55, 49, 58, 51, 43, 8, 86, 92, 72, 65, 24, 97]
    # mse_custom = evaluate_leave_one_episode_out(
    #     pre.features, pre.actions, pre.episode_ids, pre.frame_ids,
    #     heads=custom_heads, k=max(K_GRID), metric=DIST_METRIC, temp_excl_w=TEMP_EXCL_W,
    #     pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA,
    #     metric_scope=METRIC_SCOPE, metric_ctx_global=None,
    # )
    # print(f"[CUSTOM HEADS] LOEO-CV MSE={mse_custom:.6f} (k={max(K_GRID)})")

    # # Visualization for custom heads (metric context is built internally if needed)
    # diag_custom = knn_overlap_and_plots(
#     attn_h5=ATTN_H5, episodes=all_eps,
    #     model=None, heads_override=custom_heads,
    #     ks=tuple(K_GRID), feature_metric=DIST_METRIC, action_metric="mse",
    #     output_dir="knn_viz_custom_heads", use_tsne=True, temp_excl_window=TEMP_EXCL_W,
    #     global_metric_fullspace=GLOBAL_METRIC_FULLSPACE,
    #     whiten_pca_dim=WHITEN_PCA_DIM, proj_alpha=PROJ_ALPHA, pls_components=PLS_COMPONENTS,
    #     total_heads=H,
    # )


