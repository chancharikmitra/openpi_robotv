import h5py
import json, random, re, pickle
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

ATTN_H5_PATH = "/scr2/yusenluo/openpi_robotv/src/openpi/pick_train_attention_last_token_keyframe_new.h5"
TASK_JSON    = "/scr2/yusenluo/openpi_debug/openpi/src/openpi/tasks.json"



import re, h5py
from typing import List, Tuple, Set, Iterable


def plot_tsne_from_sav(
    h5_path: str,
    episode_keys: List[str],          # list of episode keys to visualize
    labels: Dict[str, int],           # mapping {ep_key: 1/0}
    sav_model: Dict,
    head: str | int = "best",         # "best" | "all" | specific head idx
    perp: int = 30,
    dim: int = 2,                     # 2 → 2D; 3 → 3D
):
    """
    head:
        "best" → use the head with highest head_acc (default in paper)
        "all"  → concatenate all selected heads before feeding to t-SNE
        int    → specify a concrete head index
    dim:
        2 → produce 2D embedding and plot a scatter in 2D (default)
        3 → produce 3D embedding and plot a 3-D scatter
    """
    sel_heads = sav_model["sel_heads"]
    head_acc  = sav_model.get("head_acc")
    if head == "best":
        head_idx = sel_heads[0] if head_acc is None else head_acc.argmax()
        head_idxs = [head_idx]
    elif head == "all":
        head_idxs = sel_heads
    elif isinstance(head, int):
        head_idxs = [head]
    else:
        raise ValueError("head arg must be 'best', 'all', or an int index")

    # ---- ① Extract feature matrix X ----
    X, y = [], []
    with h5py.File(h5_path, "r") as f:
        for ep in episode_keys:
            vec = episode_to_vec(load_episode(f, ep), sav_model["agg"])  # (144,256)
            feat = vec[head_idxs].reshape(-1)   # single head =256-dim; n heads = p*256
            X.append(feat)
            y.append(labels[ep])
    X = np.stack(X); y = np.array(y)

    # ---- ② Run t-SNE ----
    tsne = TSNE(n_components=dim, perplexity=perp, init="pca", random_state=0)
    X_emb = tsne.fit_transform(X)

    # ---- ③ Plot ----
    if dim == 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(X_emb[y == 1, 0], X_emb[y == 1, 1],
                    c="red",  marker="o", label="Pick (pos)", alpha=.7, s=40)
        plt.scatter(X_emb[y == 0, 0], X_emb[y == 0, 1],
                    c="blue", marker="x", label="Non-Pick (neg)", alpha=.7, s=40)
        plt.axis("off")
        legend_handle = plt.legend()
    elif dim == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_emb[y == 1, 0], X_emb[y == 1, 1], X_emb[y == 1, 2],
                   c="red",  marker="o", label="Pick (pos)", alpha=.7, s=40)
        ax.scatter(X_emb[y == 0, 0], X_emb[y == 0, 1], X_emb[y == 0, 2],
                   c="blue", marker="x", label="Non-Pick (neg)", alpha=.7, s=40)
        legend_handle = ax.legend()
    else:
        raise ValueError("dim must be 2 or 3")

    title = (f"t-SNE – head {head_idxs}" if head != 'all' else
             f"t-SNE – {len(sel_heads)} heads concat")
    legend_handle.set_title(title, prop={'size': 13})  # add title near legend
    plt.tight_layout()
    plt.savefig(f"pick_tsne_{head}_{dim}d.png")



def split_episode_keys(
    h5_path: str,
    pos_keywords:   Set[str],       # {"\\bpick\\b"}   positive-class keywords (regex)
    exclude_kw:     Set[str],       # {"\\bwipe\\b"}   exclusion keywords (regex)
    train_task_set: Set[str],       # tasks used in training, for deduplication
    eval_task_set: Set[str] = None,       # tasks reserved for evaluation, for deduplication
) -> Tuple[List[str], List[str]]:
    """
    Return (pos_keys, neg_keys), each key formatted as "task_name/episode_xxx".

    Decision rules:
      1. Skip the task if its name exists in train_task_set or eval_task_set.
      2. If the task matches both pos_keywords and exclude_kw → regarded as conflicting, skip.
      3. Else, if task matches pos_keywords → positive class (pos); otherwise and not containing exclude_kw → negative class (neg).
    """
    # --- compile all regex ---
    pos_pats     = [re.compile(p, re.I) for p in pos_keywords]
    exclude_pats = [re.compile(p, re.I) for p in exclude_kw]

    pos_keys, neg_keys = [], []
    with h5py.File(h5_path, "r") as h5:
        for task in h5.keys():                          # top-level group name
            task_norm = task.strip()

            # ① deduplicate: skip if task name already in training or eval set
            if task_norm in train_task_set:
                continue
            if eval_task_set is not None and task_norm in eval_task_set:
                continue

            # ② check keyword hits
            hit_pos  = any(p.search(task_norm) for p in pos_pats)
            hit_excl = any(p.search(task_norm) for p in exclude_pats)

            # → conflict: hits positive and exclusion keyword simultaneously
            if hit_pos and hit_excl:
                continue       # skip this task

            # determine class
            is_pos = hit_pos

            # ③ collect episode_xxx
            for ep in h5[task].keys():
                key = f"{task}/{ep}"
                (pos_keys if is_pos else neg_keys).append(key)

    return pos_keys, neg_keys


# ---------- load task splits ----------
with open(TASK_JSON, "r") as f:
    task_dict = json.load(f)
PICK_TASKS = set(task_dict["Pick_training_tasks"])

# WIPE_TASKS = set(task_dict["Wipe_training_tasks"])

EVAL_TASKS = set(task_dict["Pick_test_tasks"])

# ------------------------------------------------------------
# 0. split episode keys by task
# ------------------------------------------------------------
def collect_episode_keys(
    h5_path: str,
    pos_tasks: set,
    neg_tasks: set = None,
) -> Tuple[List[str], List[str]]:
    """Return (pos_keys, neg_keys); key format: task/episode_xxx"""
    pos_keys, neg_keys = [], []
    with h5py.File(h5_path, "r") as h5:
        for task in h5.keys():
            # ① determine whether the task is positive or negative
            is_pos = task in pos_tasks
            is_neg = task in neg_tasks if neg_tasks is not None else False
            # ② collect all episode groups
            for ep in h5[task].keys():            # ep = 'episode_000', ...
                key = f"{task}/{ep}"
                if neg_tasks is None:
                    (pos_keys if is_pos else neg_keys).append(key)
                else:
                    if is_pos:
                        pos_keys.append(key)
                    elif is_neg:
                        neg_keys.append(key)
    return pos_keys, neg_keys


# ------------------------------------------------------------
# 0b.  sample episodes from multiple H5 files (new)
# ------------------------------------------------------------
def _list_all_episode_keys(h5_path: str) -> List[str]:
    eps: List[str] = []
    with h5py.File(h5_path, "r") as h5:
        for task in h5.keys():
            grp = h5[task]
            if not isinstance(grp, h5py.Group):
                continue
            for ep in grp.keys():
                eps.append(f"{task}/{ep}")  # task/episode_xxx
    return eps


def sample_episode_keys_from_h5s(
    pos_h5_paths: List[str],
    neg_h5_paths: List[str],
    k_pos: int = 20,
    k_neg: int = 20,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """随机从多个 H5 中抽取正/负 episode。

    - 在每个集合内（正/负）先合并所有 H5 的 episode，再整体随机抽取 k 个。
    - 返回的键格式为 "task/episode_xxx"，与本文件其它函数兼容。
    """
    random.seed(seed)

    pos_all: List[Tuple[str, str]] = []  # (h5_path, "task/episode_xxx")
    for p in pos_h5_paths:
        for ep in _list_all_episode_keys(p):
            pos_all.append((p, ep))

    neg_all: List[Tuple[str, str]] = []
    for n in neg_h5_paths:
        for ep in _list_all_episode_keys(n):
            neg_all.append((n, ep))

    if len(pos_all) < k_pos:
        raise ValueError(f"正类可用 episodes 数量不足：{len(pos_all)} < {k_pos}")
    if len(neg_all) < k_neg:
        raise ValueError(f"负类可用 episodes 数量不足：{len(neg_all)} < {k_neg}")

    pos_sample = random.sample(pos_all, k_pos)
    neg_sample = random.sample(neg_all, k_neg)
    return pos_sample, neg_sample


# ------------------------------------------------------------
# 2b.  run_sav on multi-H5 sampled episodes (new)
# ------------------------------------------------------------
def run_sav_multi(
    pos_eps: List[Tuple[str, str]],
    neg_eps: List[Tuple[str, str]],
    k: int = 20,
    agg: str = "mean",
    sel_metric: str = "accuracy",
):
    """与 run_sav 等价，但支持 (h5_path, ep_key) 形式的 episode 列表。"""
    def _group(eps: List[Tuple[str, str]]):
        by_file: Dict[str, List[str]] = {}
        for p, e in eps:
            by_file.setdefault(p, []).append(e)
        return by_file

    pos_by_file = _group(pos_eps)
    neg_by_file = _group(neg_eps)

    feats_pos: List[np.ndarray] = []
    for h5_path, eps in pos_by_file.items():
        with h5py.File(h5_path, "r") as f:
            for e in tqdm(eps):
                feats_pos.append(episode_to_vec(load_episode(f, e), agg))

    feats_neg: List[np.ndarray] = []
    for h5_path, eps in neg_by_file.items():
        with h5py.File(h5_path, "r") as f:
            for e in tqdm(eps):
                feats_neg.append(episode_to_vec(load_episode(f, e), agg))

    feats   = np.concatenate([np.stack(feats_pos), np.stack(feats_neg)], axis=0)
    labels  = np.array([1]*len(feats_pos) + [0]*len(feats_neg))
    pos_c, neg_c = build_centroids(feats, labels)

    if sel_metric == "accuracy":
        score = head_accuracy(feats, labels, pos_c, neg_c)
    elif sel_metric in {"diff", "margin", "cos_diff"}:
        score = head_similarity_margin(feats, labels, pos_c, neg_c)
    else:
        raise ValueError("sel_metric must be 'accuracy' or 'diff'")

    heads = select_top_heads(score, k)
    n_full = n_90 = 0
    if sel_metric == "accuracy":
        n_full = np.sum(score == 1.0)
        n_90   = np.sum(score >= 0.9)
        total_heads = score.size
        print(f"✅ Heads @100% accuracy  : {n_full}/{total_heads}")
        print(f"⭐ Heads ≥90% accuracy   : {n_90}/{total_heads}")
        print("accuracy values:", [f"{v:.6f}" for v in score])
    elif sel_metric != "accuracy":
        print(heads)
        margin_vals = [float(score[h]) for h in heads]
        print("ΔMargin values (pos_vs_neg):", [f"{v:.6f}" for v in margin_vals])

    return dict(
        pos_cent=pos_c,
        neg_cent=neg_c,
        sel_heads=heads,
        head_acc=score,
        agg=agg,
        n_full=int(n_full),
    )


def run_sav_frame_level(
    pos_eps: List[Tuple[str, str]],
    neg_eps: List[Tuple[str, str]],
    k: int = 20,
    sel_metric: str = "accuracy",
    batch_frames: int = 64,
):
    """Frame-level SAV: 以每个frame作为分类的最小单位，不需要aggregation。

    两遍流式计算，避免内存爆炸：
      1) 逐帧累计每个 head 的正/负类中心（pos/neg centroids）
      2) 用小批量帧计算每个 head 的 margin 或 accuracy 并累加

    参数:
      - pos_eps/neg_eps: [(h5_path, ep_key), ...]
      - k: 选择top-k个heads
      - sel_metric: "accuracy" | "margin"
      - batch_frames: 第二遍统计时的每批帧数
    """

    def _group(eps: List[Tuple[str, str]]):
        by_file: Dict[str, List[str]] = {}
        for p, e in eps:
            by_file.setdefault(p, []).append(e)
        return by_file

    pos_by_file = _group(pos_eps)
    neg_by_file = _group(neg_eps)

    num_layers, num_heads = 18, 8
    H = num_layers * num_heads  # 144
    D = 256

    # ---------- 第一遍：累计每个 head 的正负类中心 ----------
    pos_sum = np.zeros((H, D), dtype=np.float64)
    neg_sum = np.zeros((H, D), dtype=np.float64)
    pos_cnt = np.zeros((H,), dtype=np.int64)
    neg_cnt = np.zeros((H,), dtype=np.int64)

    n_frames_pos = 0
    n_frames_neg = 0

    print("🎬 第一遍：累计正类centroid")
    for h5_path, eps in pos_by_file.items():
        with h5py.File(h5_path, "r") as f:
            for e in tqdm(eps, desc="正类episodes"):
                frames = load_episode(f, e)                 # (F,18,8,256)
                F = frames.shape[0]
                feats_fhd = frames.reshape(F, H, D)         # (F,144,256)
                pos_sum += feats_fhd.sum(axis=0)            # (144,256)
                pos_cnt += F
                n_frames_pos += F

    print("🎬 第一遍：累计负类centroid")
    for h5_path, eps in neg_by_file.items():
        with h5py.File(h5_path, "r") as f:
            for e in tqdm(eps, desc="负类episodes"):
                frames = load_episode(f, e)                 # (F,18,8,256)
                F = frames.shape[0]
                feats_fhd = frames.reshape(F, H, D)         # (F,144,256)
                neg_sum += feats_fhd.sum(axis=0)
                neg_cnt += F
                n_frames_neg += F

    # 避免除零
    pos_cnt_safe = np.maximum(pos_cnt, 1)[:, None]
    neg_cnt_safe = np.maximum(neg_cnt, 1)[:, None]
    pos_c = (pos_sum / pos_cnt_safe).astype(np.float32)     # (144,256)
    neg_c = (neg_sum / neg_cnt_safe).astype(np.float32)     # (144,256)

    print("📊 Frame-level数据统计:")
    print(f"   正类: {len(pos_eps)} episodes → {n_frames_pos} frames")
    print(f"   负类: {len(neg_eps)} episodes → {n_frames_neg} frames")

    # ---------- 第二遍：计算每个 head 的分数（margin 或 accuracy） ----------
    if sel_metric == "accuracy":
        correct = np.zeros((H,), dtype=np.int64)
        total = 0
    else:  # margin
        margin_sum = np.zeros((H,), dtype=np.float64)
        total = 0

    def _accumulate(frames: np.ndarray, label: int):
        nonlocal correct, margin_sum, total
        F = frames.shape[0]
        feats_fhd = frames.reshape(F, H, D)                  # (F,144,256)
        # 分批处理，按帧维度切分，避免一次性占用太多内存
        for i in range(0, F, batch_frames):
            sub = feats_fhd[i:i+batch_frames]               # (b,144,256)
            sim_pos = _cos(sub, pos_c)                      # (b,144)
            sim_neg = _cos(sub, neg_c)                      # (b,144)
            if sel_metric == "accuracy":
                pred = (sim_pos > sim_neg)                  # (b,144)
                if label == 1:
                    correct += pred.sum(axis=0)
                else:
                    correct += (~pred).sum(axis=0)
                total += sub.shape[0]
            else:
                sign = 1 if label == 1 else -1
                margin_sum += ((sim_pos - sim_neg) * sign).sum(axis=0)
                total += sub.shape[0]

    print("🧮 第二遍：统计分数（流式）——正类")
    for h5_path, eps in pos_by_file.items():
        with h5py.File(h5_path, "r") as f:
            for e in tqdm(eps, desc="正类frames"):
                frames = load_episode(f, e)
                _accumulate(frames, label=1)

    print("🧮 第二遍：统计分数（流式）——负类")
    for h5_path, eps in neg_by_file.items():
        with h5py.File(h5_path, "r") as f:
            for e in tqdm(eps, desc="负类frames"):
                frames = load_episode(f, e)
                _accumulate(frames, label=0)

    if sel_metric == "accuracy":
        score = (correct / max(total, 1)).astype(np.float32)
    else:
        score = (margin_sum / max(total, 1)).astype(np.float32)

    heads = select_top_heads(score, k)

    # 统计信息
    n_full = n_90 = 0
    if sel_metric == "accuracy":
        n_full = int(np.sum(score == 1.0))
        n_90   = int(np.sum(score >= 0.9))
        total_heads = score.size
        print(f"✅ Heads @100% accuracy  : {n_full}/{total_heads}")
        print(f"⭐ Heads ≥90% accuracy   : {n_90}/{total_heads}")
        print(f"🎯 Top-{k} heads (frame-level): {heads}")
    else:
        margin_vals = [float(score[h]) for h in heads]
        print("ΔMargin values (pos_vs_neg):", [f"{v:.6f}" for v in margin_vals])
        print(f"🎯 Top-{k} heads (frame-level): {heads}")

    return dict(
        pos_cent=pos_c,
        neg_cent=neg_c,
        sel_heads=heads,
        head_acc=score,
        agg="frame_level_stream",
        n_full=int(n_full),
        n_frames_pos=n_frames_pos,
        n_frames_neg=n_frames_neg,
    )

# ------------------------------------------------------------
# 2c.  variance-based head selection across one/multi H5s (new)
# ------------------------------------------------------------
def find_top_variance_heads_from_h5s(
    h5_paths: List[str],
    k: int = 20,
    agg: str = "mean",
    normalize: str = "l2",
):
    """在给定多个 H5 文件中，找出 activation 方差最大的 K 个 heads。

    多文件处理逻辑：
      1. 遍历每个 H5 文件，提取所有 episodes 的 head activations
      2. 将所有文件的 episodes 合并到一个大的特征矩阵中
      3. 在整个合并数据集上计算每个 head 的方差
      4. 返回方差最大的 top-k heads

    计算方式：
      - 对每个 episode 计算 (144,256) 的 head 表示（见 episode_to_vec）
      - 在样本维度上计算每个 head 的特征方差（逐维方差后再对 256 维取均值），得到 (144,) 的分数
      - 返回方差最大的前 K 个 head 的索引

    参数：
      h5_paths: 多个 H5 文件路径列表
      k: 返回 top-k 个 heads
      agg: 聚合方式 ("mean" | "max" | "last")
      normalize: 归一化方式 ("l2" | "none"), 默认 "l2" 对每个样本/每个head的256维向量做单位范数

    返回：
      dict(sel_heads=list[int], var_scores=np.ndarray(shape (144,)), agg=str, 
           n_episodes=int, n_files=int)
    """
    feats: List[np.ndarray] = []  # 每个元素形状为 (144,256)
    total_episodes = 0

    print(f"📁 处理 {len(h5_paths)} 个 H5 文件...")
    
    # 按文件批量读取，避免重复打开
    for i, h5_path in enumerate(h5_paths):
        print(f"🔄 [{i+1}/{len(h5_paths)}] 处理文件: {h5_path}")
        
        with h5py.File(h5_path, "r") as f:
            episode_keys = _list_all_episode_keys(h5_path)
            print(f"   📊 找到 {len(episode_keys)} 个 episodes")
            
            for ep in tqdm(episode_keys, desc=f"   提取 activations", leave=False):
                feats.append(episode_to_vec(load_episode(f, ep), agg))
                
        total_episodes += len(episode_keys)

    if len(feats) == 0:
        raise ValueError("未在提供的 H5 文件中找到任何 episode")

    feat_stack = np.stack(feats, axis=0)  # (N,144,256)
    print(f"🔢 合并特征矩阵形状: {feat_stack.shape}")
    
    # 可选归一化（减少层间/样本幅度差异的影响）
    if normalize == "l2":
        norms = np.linalg.norm(feat_stack, axis=-1, keepdims=True)
        feat_stack = feat_stack / np.maximum(norms, 1e-8)
        print("⚖️ 已进行 L2 归一化后再计算方差")
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'l2' or 'none'")
    print(f"📈 总计处理: {total_episodes} episodes from {len(h5_paths)} files")
    
    # 样本维度上计算方差 → (144,256)
    var_per_dim = feat_stack.var(axis=0)
    # 聚合到每个 head 的标量分数 → (144,)
    var_scores = var_per_dim.mean(axis=1)
    heads = var_scores.argsort()[::-1][:k].tolist()

    print(f"🎯 Top-{k} heads by variance: {heads}")
    print(f"📈 方差分数范围: [{var_scores.min():.6f}, {var_scores.max():.6f}]")
    
    # 打印 top-k heads 的方差值
    top_k_scores = [float(var_scores[h]) for h in heads]
    print("🔝 Top-k heads 方差值:", [f"{v:.6f}" for v in top_k_scores])
    
    return dict(
        sel_heads=heads, 
        var_scores=var_scores, 
        agg=agg,
        normalize=normalize,
        n_episodes=total_episodes,
        n_files=len(h5_paths)
    )


def find_top_variance_heads_single_h5(
    h5_path: str,
    k: int = 20,
    agg: str = "mean",
    normalize: str = "l2",
):
    """在单个 H5 文件中，找出 activation 方差最大的 K 个 heads。

    计算方式：
      - 对每个 episode 计算 (144,256) 的 head 表示（见 episode_to_vec）。
      - 在样本维度上计算每个 head 的特征方差（逐维方差后再对 256 维取均值），得到 (144,) 的分数。
      - 返回方差最大的前 K 个 head 的索引。

    参数：
      h5_path: 单个 H5 文件路径
      k: 返回 top-k 个 heads
      agg: 聚合方式 ("mean" | "max" | "last")
      normalize: 归一化方式 ("l2" | "none"), 默认 "l2" 对每个样本/每个head的256维向量做单位范数

    返回：
      dict(sel_heads=list[int], var_scores=np.ndarray(shape (144,)), agg=str, n_episodes=int)
    """
    feats: List[np.ndarray] = []  # 每个元素形状为 (144,256)

    with h5py.File(h5_path, "r") as f:
        episode_keys = _list_all_episode_keys(h5_path)
        print(f"📁 处理文件: {h5_path}")
        print(f"📊 找到 {len(episode_keys)} 个 episodes")
        
        for ep in tqdm(episode_keys, desc="提取 head activations"):
            feats.append(episode_to_vec(load_episode(f, ep), agg))

    if len(feats) == 0:
        raise ValueError(f"在 H5 文件 {h5_path} 中未找到任何 episode")

    feat_stack = np.stack(feats, axis=0)  # (N,144,256)
    print(f"🔢 特征矩阵形状: {feat_stack.shape}")
    
    # 可选归一化（减少层间/样本幅度差异的影响）
    if normalize == "l2":
        norms = np.linalg.norm(feat_stack, axis=-1, keepdims=True)
        feat_stack = feat_stack / np.maximum(norms, 1e-8)
        print("⚖️ 已进行 L2 归一化后再计算方差")
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'l2' or 'none'")
    
    # 样本维度上计算方差 → (144,256)
    var_per_dim = feat_stack.var(axis=0)
    # 聚合到每个 head 的标量分数 → (144,)
    var_scores = var_per_dim.mean(axis=1)
    heads = var_scores.argsort()[::-1][:k].tolist()

    print(f"🎯 Top-{k} heads by variance: {heads}")
    print(f"📈 方差分数范围: [{var_scores.min():.6f}, {var_scores.max():.6f}]")
    
    # 打印 top-k heads 的方差值
    top_k_scores = [float(var_scores[h]) for h in heads]
    print("🔝 Top-k heads 方差值:", [f"{v:.6f}" for v in top_k_scores])
    
    return dict(
        sel_heads=heads, 
        var_scores=var_scores, 
        agg=agg,
        normalize=normalize,
        n_episodes=len(feats)
    )

# ------------------------------------------------------------
# 1.  load a single episode → (F, 18, 8, 256)                ★
# ------------------------------------------------------------
def load_episode(h5_file, ep_key):
    grp = h5_file[ep_key]
    frames = []
    for fk in sorted(grp.keys()):
        # raw = grp[fk]["last_token_attn"]
        raw = grp[fk]["last_token_attn"]
        # raw = grp[fk]["first_action_token_attn"]
        # —— convert dtype to float32 if necessary ——            ★ new
        arr = np.asarray(raw)
        if arr.dtype.kind == 'V' and arr.itemsize == 2: 
            arr = arr.view('<f2').astype(np.float32)
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        frames.append(arr)   # (18,8,256)
    return np.stack(frames, axis=0).astype(np.float32)


# ------------------------------------------------------------
# 2.  episode → heads×dim representation                           ★
# ------------------------------------------------------------
def episode_to_vec(attn_frames: np.ndarray, agg="mean") -> np.ndarray:
    """Convert attention frames (F,18,8,256) to flattened (144,256) head representations."""
    if agg == "last":
        use_frames = attn_frames[-1:]             # (1,18,8,256)
    elif agg == "max":
        use_frames = attn_frames.max(axis=0, keepdims=True)
    else:                                         # mean
        use_frames = attn_frames.mean(axis=0, keepdims=True)
    vec = use_frames[0]                           # (18,8,256)
    L, H, D = vec.shape
    return vec.reshape(L * H, D)                  # (144,256)


def episode_to_frame_vecs(attn_frames: np.ndarray) -> np.ndarray:
    """Convert attention frames (F,18,8,256) to frame-level head representations (F*144,256).
    
    不需要aggregation，每个frame的每个head都作为独立样本。
    
    返回:
        np.ndarray: shape (F*144, 256), 每一行代表一个frame中一个head的activation
    """
    F, L, H, D = attn_frames.shape  # (frames, layers, heads, dim)
    # 重塑为 (F*144, 256)
    return attn_frames.reshape(F * L * H, D)

# ------------------------------------------------------------
# 3. Single-head scoring functions (new)
# ------------------------------------------------------------
def build_centroids(feats: np.ndarray, labels: np.ndarray):
    """Compute mean vectors per head for positive / negative classes."""
    pos_cent = feats[labels == 1].mean(axis=0)
    neg_cent = feats[labels == 0].mean(axis=0)
    return pos_cent, neg_cent


def _cos(a: np.ndarray, b: np.ndarray):
    """Batch cosine similarity with broadcasting support."""
    a_n = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a_n * b_n).sum(-1)


def _cos_memory_efficient(feats: np.ndarray, centroids: np.ndarray, batch_size: int = 10000):
    """Memory-efficient cosine similarity for large datasets."""
    n_samples, n_heads, dim = feats.shape[0], centroids.shape[0], centroids.shape[1]
    result = np.zeros((n_samples, n_heads), dtype=np.float32)
    
    # 预先归一化centroids
    cent_norm = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
    
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        batch_feats = feats[i:end_i]  # (batch, n_heads, dim)
        
        # 归一化batch features
        batch_norm = batch_feats / np.linalg.norm(batch_feats, axis=-1, keepdims=True)
        
        # 计算cosine similarity: (batch, n_heads, dim) × (n_heads, dim) → (batch, n_heads)
        result[i:end_i] = (batch_norm * cent_norm[None, :, :]).sum(axis=-1)
        
    return result


def head_accuracy(feats: np.ndarray, labels: np.ndarray, pos_cent: np.ndarray, neg_cent: np.ndarray):
    """Compute training accuracy per head."""
    sim_pos = _cos(feats, pos_cent)
    sim_neg = _cos(feats, neg_cent)
    pred = (sim_pos > sim_neg).astype(np.int32)
    return (pred == labels[:, None]).mean(axis=0)


def head_similarity_margin(feats: np.ndarray, labels: np.ndarray, pos_cent: np.ndarray, neg_cent: np.ndarray):
    """Head score based on cosine-similarity margin (confidence)."""
    # 对于大数据集，使用内存友好版本
    if feats.shape[0] > 50000:  # 超过5万样本时使用batch处理
        return head_similarity_margin_memory_efficient(feats, labels, pos_cent, neg_cent)
    
    sim_pos = _cos(feats, pos_cent)
    sim_neg = _cos(feats, neg_cent)
    sign = np.where(labels == 1, 1, -1)[:, None]  # (N,1); +1 for positive, -1 for negative
    margin = (sim_pos - sim_neg) * sign          # (N,144)
    return margin.mean(axis=0)                   # (144,)


def head_similarity_margin_memory_efficient(feats: np.ndarray, labels: np.ndarray, pos_cent: np.ndarray, neg_cent: np.ndarray, batch_size: int = 10000):
    """Memory-efficient version of head_similarity_margin for large datasets."""
    n_samples = feats.shape[0]
    n_heads = pos_cent.shape[0]
    margins = np.zeros(n_heads, dtype=np.float64)
    
    print(f"🧠 使用内存友好版本处理 {n_samples:,} 样本，batch_size={batch_size:,}")
    
    for i in tqdm(range(0, n_samples, batch_size), desc="计算margins"):
        end_i = min(i + batch_size, n_samples)
        batch_feats = feats[i:end_i]  # (batch, 256)
        batch_labels = labels[i:end_i]
        
        # 简化版cosine similarity计算，避免复杂的broadcasting
        sim_pos = np.zeros((batch_feats.shape[0], n_heads))
        sim_neg = np.zeros((batch_feats.shape[0], n_heads))
        
        # 归一化batch features
        batch_norm = batch_feats / np.linalg.norm(batch_feats, axis=-1, keepdims=True)
        pos_cent_norm = pos_cent / np.linalg.norm(pos_cent, axis=-1, keepdims=True)
        neg_cent_norm = neg_cent / np.linalg.norm(neg_cent, axis=-1, keepdims=True)
        
        # 计算cosine similarity: (batch, 256) @ (144, 256).T = (batch, 144)
        sim_pos = batch_norm @ pos_cent_norm.T
        sim_neg = batch_norm @ neg_cent_norm.T
        
        # 计算margin
        sign = np.where(batch_labels == 1, 1, -1)[:, None]  # (batch,1)
        batch_margin = (sim_pos - sim_neg) * sign           # (batch,144)
        
        # 累加到总margin
        margins += batch_margin.sum(axis=0)
    
    # 返回平均margin
    return margins / n_samples


# Top-k selection
def select_top_heads(scores: np.ndarray, k: int = 20):
    return scores.argsort()[::-1][:k].tolist()

# Select the worst k heads
def select_bottom_heads(scores: np.ndarray, k: int = 20):
    """Return indices of the k lowest-scoring heads (ascending order)."""
    return scores.argsort()[:k].tolist()

# ------------------------------------------------------------
# 4.  Majority vote prediction with sparse heads (same as before)
# ------------------------------------------------------------
def predict_with_sparse_heads(feat, pos_c, neg_c, heads):
    def cos1(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    votes = 0
    for h in heads:
        votes += 1 if cos1(feat[h], pos_c[h]) > cos1(feat[h], neg_c[h]) else -1
    return 1 if votes > 0 else 0

# ------------------------------------------------------------
# 5.  Main pipeline: train / select heads
# ------------------------------------------------------------
def run_sav(h5_path, pos_eps, neg_eps, k=20, agg="mean", sel_metric: str = "accuracy"):
    with h5py.File(h5_path, "r") as f:
        feats_pos = [episode_to_vec(load_episode(f, e), agg) for e in tqdm(pos_eps)]
        feats_neg = [episode_to_vec(load_episode(f, e), agg) for e in tqdm(neg_eps)]

    feats   = np.concatenate([feats_pos, feats_neg], axis=0)           # (N,144,256)
    labels  = np.array([1]*len(feats_pos) + [0]*len(feats_neg))
    pos_c, neg_c = build_centroids(feats, labels)

    # choose scoring metric
    if sel_metric == "accuracy":
        score = head_accuracy(feats, labels, pos_c, neg_c)
    elif sel_metric in {"diff", "margin", "cos_diff"}:
        score = head_similarity_margin(feats, labels, pos_c, neg_c)
    else:
        raise ValueError("sel_metric must be 'accuracy' or 'diff'")

    # heads = select_top_heads(score, k)
    heads = select_bottom_heads(score, k)
    # ---- statistics ----------------------------------------------------
    print(f"🎯 Top-{k} heads ({sel_metric}) : {heads}")

    # If margin scoring is used, print per-head margin values for reference
    if sel_metric != "accuracy":
        margin_vals = [float(score[h]) for h in heads]
        print("ΔMargin values (pos_vs_neg):", [f"{v:.6f}" for v in margin_vals])

    # Only compute 100% / 90% stats when scoring metric is accuracy
    n_full = n_90 = 0
    if sel_metric == "accuracy":
        n_full = np.sum(score == 1.0)
        n_90   = np.sum(score >= 0.9)
        total_heads = score.size                     # 144
        print(f"✅ Heads @100% accuracy  : {n_full}/{total_heads}")
        print(f"⭐ Heads ≥90% accuracy   : {n_90}/{total_heads}")
    # -----------------------------------------------------------------

    return dict(
        pos_cent=pos_c,
        neg_cent=neg_c,
        sel_heads=heads,
        head_acc=score,      # legacy field name for compatibility
        agg=agg,
        n_full=int(n_full),
    )


# ------------------------------------------------------------
# 6.  Evaluation
# ------------------------------------------------------------
def evaluate(h5_path: str, episodes: list[str], labels_dict: dict[str,int], model):
    # ----------1. Compute vote matrix ----------
    sample_k = 20  # keep consistent with previous behaviour
    votes_mat, used_heads = compute_votes_per_sample(
        h5_path, episodes, model, sample_k=sample_k, verbose=False)

    labels = np.array([labels_dict[ep] for ep in episodes])           # (N,)

    # ----------2. Majority vote prediction accuracy ----------
    preds = (votes_mat.sum(axis=1) > 0).astype(np.int32)              # (N,)
    acc   = (preds == labels).mean()
    print(f"✅ Test accuracy = {acc:.3%}")

    # ----------3. Calculate confidence per sample ----------
    sign = np.where(labels == 1, 1, -1)[:, None]                      # (N,1)
    conf_per_sample = (votes_mat * sign).sum(axis=1)                  # (N,)
    avg_conf = conf_per_sample.mean()
    print(f"📊 Avg confidence (correct−wrong votes) = {avg_conf:.2f} / {votes_mat.shape[1]}")


# ------------------------------------------------------------
# 6b.  Statistics on votes per sample in selected heads
# ------------------------------------------------------------
def compute_votes_per_sample(
    h5_path: str,
    episodes: list[str],
    model,
    sample_k: int | None = 20,
    verbose: bool = False,
):
    """Return (votes_mat, heads).

    votes_mat: ndarray of shape (N, K) with values +1 / -1 indicating each head's vote.
    heads:     list[int] of head indices corresponding to the columns in votes_mat.

    If sample_k is None, all heads in model['sel_heads'] are used; otherwise randomly sample
    sample_k heads (same policy as evaluate)."""

    pc, nc, sel_heads, agg = (
        model["pos_cent"],
        model["neg_cent"],
        model["sel_heads"],
        model["agg"],
    )

    # head sampling
    heads = np.asarray(sel_heads)
    #heads = np.array([64, 49, 116, 13, 9, 39, 113, 87, 58, 82, 54, 4, 10, 59, 109, 37, 77, 38, 28, 45]) 
    # if sample_k is not None and heads.shape[0] > sample_k:
    #     np.random.seed(1)
    #     heads = heads[np.random.choice(heads.shape[0], size=sample_k, replace=False)]

    votes_all: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        for ep in tqdm(episodes, desc="compute votes"):
            feat = episode_to_vec(load_episode(f, ep), agg)        # (144,256)
            sel_feat = feat[heads]                                # (K,256)

            pos_sim = _cos(sel_feat, pc[heads])
            neg_sim = _cos(sel_feat, nc[heads])
            votes = np.where(pos_sim > neg_sim, 1, -1)           # (K,)
            votes_all.append(votes)

            if verbose:
                pos_cnt = int((votes == 1).sum())
                neg_cnt = votes.size - pos_cnt
                print(f"{ep}: +{pos_cnt} / -{neg_cnt},  majority = {'+1' if pos_cnt>neg_cnt else '-1'}")

    votes_mat = np.stack(votes_all, axis=0)   # (N,K)
    return votes_mat, heads.tolist()



def save_selected_head_activations(
    h5_path: str,
    episode_keys: List[str],              # list of episodes to export
    sel_heads: List[int],                # selected head indices (flat 0~143)
    agg: str = "mean",                 # "mean" | "max" | "last"
    out_h5: str = "selected_heads_feat.h5",
    zero_fill: bool = True,              # True → fill unselected heads with 0; False → save only selected heads
):
    """Save activations from selected heads into a new H5 file.

    For each episode, a dataset named "task__episode" (slash replaced by double underscore)
    is stored with shape (144,256) or (k,256) when ``zero_fill=False``.
    """
    assert agg in {"mean", "max", "last"}, "agg must be 'mean', 'max', or 'last'"

    with h5py.File(h5_path, "r") as fin, h5py.File(out_h5, "w") as fout:
        # store selected heads & aggregation mode in file attributes for later use
        fout.attrs["sel_heads"] = json.dumps(sel_heads)
        fout.attrs["agg"] = agg

        for ep in tqdm(episode_keys, desc="Saving activations"):
            feat = episode_to_vec(load_episode(fin, ep), agg)  # (144,256)

            if zero_fill:
                # restore original 3D shape (layer, head, dim) = (18,8,256)
                num_layers, num_heads = 18, 8
                data = np.zeros((num_layers, num_heads, feat.shape[-1]), dtype=np.float32)
                # copy activations for sel_heads into flattened view, then reshape back
                data.reshape(-1, feat.shape[-1])[sel_heads] = feat[sel_heads]
            else:
                data = feat[sel_heads].astype(np.float32)      # (k,256)

            # keep original hierarchy: task -> episode -> dataset
            task_name, ep_name = ep.split("/", 1)  # task, episode_xxx
            task_grp = fout.require_group(task_name)
            ep_grp = task_grp.require_group(ep_name)

            # write/overwrite dataset "selected_heads"
            if "selected_heads" in ep_grp:
                del ep_grp["selected_heads"]
            ep_grp.create_dataset("selected_heads", data=data, compression="gzip")

    print(f"✅ Saved {len(episode_keys)} episodes → {out_h5}")

# --------------------------- Example usage ---------------------------
# The following demonstrates how to export Top-k heads (mean aggregation) obtained from run_sav.
# To export max/last aggregation instead, simply set agg="max"/"last" and call again.
# ----------------------------------------------------------------
# example_out_h5 = "pick_topk_heads_mean.h5"
# save_selected_head_activations(
#     h5_path=ATTN_H5_PATH,
#     episode_keys=support_pos + support_neg,  # or any other episode list
#     sel_heads=sav_model["sel_heads"],
#     agg=sav_model["agg"],
#     out_h5=example_out_h5,
#     zero_fill=True,
# )



# ------------------------------------------------------------
# 7.  Example run: Pick vs Non-Pick
# ------------------------------------------------------------
# 7.1  Sample support set
all_pos, all_neg = collect_episode_keys(ATTN_H5_PATH, pos_tasks=PICK_TASKS)
# all_pos, all_neg = sample_episode_keys_from_h5s(
#     pos_h5_paths=[
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_remove_marker_from_mug_20_state_first_action.h5",
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_place_marker_in_mug_20_state_first_action.h5",
#     ],
#     neg_h5_paths=[
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_wipe_table_with_cloth_20_state_first_action.h5",
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_wipe_table_with_yellow_cloth_20_state_first_action.h5",
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_pick_up_green_cube_20_state_first_action.h5",
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_pick_up_red_cube_20_state_first_action.h5",
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_place_green_cube_in_red_bowl_20_state_first_action.h5",
#         "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_pick_up_red_mug_20_state_first_action.h5",
#     ], 
#     k_pos=20, k_neg=20, seed=42,
# )
print(len(all_pos), len(all_neg))
random.seed(42)
support_pos = random.sample(all_pos, 20)
support_neg = random.sample(all_neg, 20)    

sav_model = run_sav(ATTN_H5_PATH, support_pos, support_neg, k=20, agg="mean", sel_metric="margin")
# sav_model = run_sav_multi(all_pos, all_neg, k=20, agg="mean", sel_metric="margin")
# run_sav_frame_level(all_pos, all_neg, k=20, sel_metric="margin")
# ATTN_H5_PATH = "/scr2/yusenluo/openpi_debug/openpi/attention_dataset/PI0DROID_remove_marker_from_mug_20_state_first_action.h5"
# find_top_variance_heads_single_h5(ATTN_H5_PATH, k=20, agg="mean", normalize="l2")

# pickle.dump(sav_model, open("sav_pick.pkl", "wb"))

# example_out_h5 = "steer_train_wipe_top_20_heads_mean.h5"
# save_selected_head_activations(
#     h5_path=ATTN_H5_PATH,
#     episode_keys=support_pos,  # or any other episode list
#     sel_heads=sav_model["sel_heads"],
#     agg=sav_model["agg"],
#     out_h5=example_out_h5,
#     zero_fill=True,
# )

EVAL_H5 = "/scr2/yusenluo/openpi_robotv/src/openpi/pick_eval_attention_last_token_single_action_keyframe.h5"    
all_pos_eval, all_neg_eval = split_episode_keys(EVAL_H5, pos_keywords={"\\bpick\\b"}, exclude_kw={"\\bwipe\\b"}, train_task_set=PICK_TASKS)
print(len(all_pos_eval), len(all_neg_eval)) 
# print(all_pos_eval)
# print(all_neg_eval)
pos_eval = random.sample(all_pos_eval, 200)
neg_eval = random.sample(all_neg_eval, 200)
eval_eps    = pos_eval + neg_eval          # or a balanced sample
eval_labels = {ep: (1 if ep in pos_eval else 0) for ep in eval_eps}
# # plot_tsne_from_sav(EVAL_H5, eval_eps, eval_labels, sav_model, head="all", dim=3)
evaluate(EVAL_H5, eval_eps, eval_labels, sav_model)
# ks, accs = evaluate_curve(EVAL_H5, eval_eps, eval_labels, sav_model,
#                           max_k=32, step=4, plot_file="topk_curve.png")



# ---------- Heatmap ----------


def plot_head_heatmap(values: np.ndarray, sel_heads: List[int],
                      num_layers: int = 18, num_heads: int = 8,
                      title: str = "", cmap: str = "RdYlBu_r",
                      vmin: float | None = None, vmax: float | None = None):
    """通用 head heatmap，可视化任意 per-head 数值（accuracy、margin、contribution 等）。

    vmin/vmax 默认为数据范围；accuracy 可手动设为 0~1 以对齐颜色条。
    """

    mat = values.reshape(num_layers, num_heads)
    if vmin is None:
        vmin = float(mat.min())
    if vmax is None:
        vmax = float(mat.max())

    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xticks(range(num_heads)); ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
    ax.set_yticks(range(num_layers)); ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title, pad=12, weight="bold")

    # draw red rectangle highlight
    for idx in sel_heads:
        l, h = divmod(idx, num_heads)
        rect = patches.Rectangle((h-0.5, l-0.5), 1, 1, linewidth=1.8,
                                 edgecolor="black", facecolor="none")
        ax.add_patch(rect)
    # 注释显示使用的 active heads
    ax.text(-0.4, -1.1,
            f"Active Heads: {len(sel_heads)}/{num_layers*num_heads}"
            f" ({len(sel_heads)/(num_layers*num_heads):.1%})",
            fontsize=8, bbox=dict(facecolor="lightgray", alpha=.7))

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig("SAV_head_heatmap/pick_head_heatmap_generic.png")
    plt.show()


# ------------------------------------------------------------
# Per-head contribution heatmap (Δ between pos/neg centroids)
# ------------------------------------------------------------


# ------------------------------------------------------------
# Simple wrapper: visualize SAV-trained metric (accuracy / margin)
# ------------------------------------------------------------


def plot_sav_head_heatmap(sav_model: dict, title: str | None = None):
    """使用训练时的度量 (sav_model['head_acc']) 绘制 heatmap，并高亮所选 top-k heads。

    sav_model: 字典，run_sav 的返回值。
    title: 自定义标题；默认根据 sel_metric 自动生成。
    """

    values = sav_model["head_acc"]          # same metric used for选头 (accuracy / margin)
    sel_heads = sav_model["sel_heads"]
    print(sel_heads)
    # 自动判断是否是 accuracy（范围 0~1）
    if values.min() >= 0 and values.max() <= 1:
        vmin, vmax = 0., 1.
        cmap = "RdYlBu_r"
    else:                       # margin 等其他指标
        vmin = float(values.min())
        vmax = float(values.max())
        cmap = "YlOrRd"

    if title is None:
        title = "Head Metric Heatmap (training metric)"

    plot_head_heatmap(values, sel_heads, title=title,
                      cmap=cmap, vmin=vmin, vmax=vmax)
# plot_sav_head_heatmap(sav_model)

