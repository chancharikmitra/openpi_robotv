import hdf5plugin
import h5py
import json, random, re, pickle
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore

ATTN_H5_PATH = "pick_train_attention_last_token_keyframe.h5"
TASK_JSON    = "tasks.json"



import re, h5py
from typing import List, Tuple, Set, Iterable


def plot_tsne_from_sav(
    h5_path: str,
    episode_keys: List[str],          # 要可视化的 episode 列表
    labels: Dict[str, int],           # {ep_key: 1/0}
    sav_model: Dict,
    head: str | int = "best",         # "best" | "all" | 具体 head idx
    perp: int = 30,
):
    """
    head:
        "best" → 用 head_acc 最高的一个头          (论文示例)
        "all"  → concat sel_heads 全部头向量再 t-SNE
        int    → 指定某一头索引
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
        raise ValueError("head 参数必须 'best' / 'all' / int")

    # ---- ① 提取特征矩阵 X ----
    X, y = [], []
    with h5py.File(h5_path, "r") as f:
        for ep in episode_keys:
            vec = episode_to_vec(load_episode(f, ep), sav_model["agg"])  # (144,256)
            feat = vec[head_idxs].reshape(-1)   # 1 头=256dim; n头=p*256
            X.append(feat)
            y.append(labels[ep])
    X = np.stack(X); y = np.array(y)

    # ---- ② t-SNE ----
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0)
    X_2d = tsne.fit_transform(X)

    # ---- ③ 画图 ----
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                c="red",  marker="o", label="Pick (pos)", alpha=.7, s=40)
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                c="blue", marker="x", label="Non-Pick (neg)", alpha=.7, s=40)

    title = (f"t-SNE – head {head_idxs}" if head != 'all' else
             f"t-SNE – {len(sel_heads)} heads concat")
    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pick_tsne_{head}.png")



def split_episode_keys(
    h5_path: str,
    pos_keywords:   Set[str],       # {"\\bpick\\b"}   正类关键字 (正则字符串)
    exclude_kw:     Set[str],       # {"\\bwipe\\b"}   任一命中则排除
    train_task_set: Set[str],       # JSON 训练任务名 → 用于去重
) -> Tuple[List[str], List[str]]:
    """
    返回 (pos_keys, neg_keys)，键 = "task_name/episode_xxx"
    逻辑：
      1. 先检查 task 是否在 train_task_set → 若是直接跳过
      2. 若 task 同时命中 pos_keywords & exclude_kw → 视为冲突，跳过
      3. 若 task 命中 pos_keywords              → 正类
         否则且不含 exclude_kw                 → 负类
    """
    # --- 编译全部正则 ---
    pos_pats     = [re.compile(p, re.I) for p in pos_keywords]
    exclude_pats = [re.compile(p, re.I) for p in exclude_kw]

    pos_keys, neg_keys = [], []
    with h5py.File(h5_path, "r") as h5:
        for task in h5.keys():                          # 顶层 group 名
            task_norm = task.strip()

            # ① 去重：若与训练集任务同名 → 跳过
            if task_norm in train_task_set:
                continue

            # ② 检查关键字命中
            hit_pos  = any(p.search(task_norm) for p in pos_pats)
            hit_excl = any(p.search(task_norm) for p in exclude_pats)

            # → 冲突：同时命中正关键词 & 排除关键词
            if hit_pos and hit_excl:
                continue       # 直接 skip 该 task

            # 分类
            is_pos = hit_pos

            # ③ 收集 episode_xxx
            for ep in h5[task].keys():
                key = f"{task}/{ep}"
                (pos_keys if is_pos else neg_keys).append(key)

    return pos_keys, neg_keys


# ---------- 读取任务分组 ----------
with open(TASK_JSON, "r") as f:
    task_dict = json.load(f)
PICK_TASKS = set(task_dict["Pick_training_tasks"])
WIPE_TASKS = set(task_dict["Wipe_training_tasks"])

# ------------------------------------------------------------
# 0. 按 task 划分正/负 episode key
# ------------------------------------------------------------
def collect_episode_keys(
    h5_path: str,
    pos_tasks: set,
) -> Tuple[List[str], List[str]]:
    """返回 (pos_keys, neg_keys)，键格式: task/episode_xxx"""
    pos_keys, neg_keys = [], []
    with h5py.File(h5_path, "r") as h5:
        for task in h5.keys():
            # ① 判断该 task 是正类还是负类
            is_pos = task in pos_tasks
            # ② 收集所有 episode group
            for ep in h5[task].keys():            # ep = 'episode_000', ...
                key = f"{task}/{ep}"
                (pos_keys if is_pos else neg_keys).append(key)
    return pos_keys, neg_keys

# ------------------------------------------------------------
# 1.  读取单个 episode → (F, 18, 8, 256)                ★
# ------------------------------------------------------------
def load_episode(h5_file, ep_key):
    grp = h5_file[ep_key]
    frames = []
    for fk in sorted(grp.keys()):
        raw = grp[fk]["last_token_attn"]
        # —— 若 dtype 不是 float32 就转 ——            ★ 新增
        arr = np.asarray(raw)
        if arr.dtype.kind == 'V' and arr.itemsize == 2:
            arr = arr.view('<f2').astype(np.float32)
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        frames.append(arr)   # (18,8,256)
    return np.stack(frames, axis=0).astype(np.float32)


# ------------------------------------------------------------
# 2.  episode → heads×dim 表示                           ★
# ------------------------------------------------------------
def episode_to_vec(attn_frames: np.ndarray, agg="mean") -> np.ndarray:
    """
    attn_frames: (F,18,8,256)  →  return (144,256)
    """
    if agg == "last":
        use_frames = attn_frames[-1:]             # (1,18,8,256)
    elif agg == "max":
        use_frames = attn_frames.max(axis=0, keepdims=True)
    else:                                         # mean
        use_frames = attn_frames.mean(axis=0, keepdims=True)
    vec = use_frames[0]                           # (18,8,256)
    L, H, D = vec.shape
    return vec.reshape(L * H, D)                  # (144,256)

# ------------------------------------------------------------
# 3.  计算单头准确率 → 选 Top-k        (与之前相同)
# ------------------------------------------------------------
def build_centroids(feats, labels):
    pos_cent = feats[labels == 1].mean(axis=0)
    neg_cent = feats[labels == 0].mean(axis=0)
    return pos_cent, neg_cent

def _cos(a, b):
    a_n = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a_n * b_n).sum(-1)

def head_accuracy(feats, labels, pos_cent, neg_cent):
    sim_pos = _cos(feats, pos_cent)
    sim_neg = _cos(feats, neg_cent)
    pred     = (sim_pos > sim_neg).astype(np.int32)
    return (pred == labels[:, None]).mean(axis=0)

def select_top_heads(acc, k=20): return acc.argsort()[::-1][:k].tolist()

# ------------------------------------------------------------
# 4.  用稀疏头多数投票预测          (与之前相同)
# ------------------------------------------------------------
def predict_with_sparse_heads(feat, pos_c, neg_c, heads):
    def cos1(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    votes = 0
    for h in heads:
        votes += 1 if cos1(feat[h], pos_c[h]) > cos1(feat[h], neg_c[h]) else -1
    return 1 if votes > 0 else 0

# ------------------------------------------------------------
# 5.  主流程: 训练 / 选头
# ------------------------------------------------------------
def run_sav(h5_path, pos_eps, neg_eps, k=20, agg="mean"):
    with h5py.File(h5_path, "r") as f:
        feats_pos = [episode_to_vec(load_episode(f, e), agg) for e in tqdm(pos_eps)]
        feats_neg = [episode_to_vec(load_episode(f, e), agg) for e in tqdm(neg_eps)]

    feats   = np.concatenate([feats_pos, feats_neg], axis=0)           # (N,144,256)
    labels  = np.array([1]*len(feats_pos) + [0]*len(feats_neg))
    pos_c, neg_c = build_centroids(feats, labels)
    acc     = head_accuracy(feats, labels, pos_c, neg_c)               # (144,)
    heads   = select_top_heads(acc, k)

    # ---- 新增统计 ----------------------------------------------------
    n_full = np.sum(acc == 1.0)                # 100 % 准确率
    n_90   = np.sum(acc >= 0.9)                # ≥90 % 准确率
    total_heads = acc.size                     # 144

    print(f"🎯 Top-{k} heads          : {heads}")
    print(f"✅ Heads @100% accuracy  : {n_full}/{total_heads}")
    print(f"⭐ Heads ≥90% accuracy   : {n_90}/{total_heads}")
    # -----------------------------------------------------------------

    return dict(
        pos_cent=pos_c,
        neg_cent=neg_c,
        sel_heads=heads,
        head_acc=acc,
        agg=agg,
        n_full=int(n_full),
    )


# ------------------------------------------------------------
# 6.  评估
# ------------------------------------------------------------
def evaluate(h5_path: str, episodes: list[str], labels_dict: dict[str,int], model):
    pc, nc, heads, agg = model["pos_cent"], model["neg_cent"], model["sel_heads"], model["agg"]
    correct = 0
    with h5py.File(h5_path, "r") as f:
        for ep in tqdm(episodes):
            feat  = episode_to_vec(load_episode(f, ep), agg)
            pred  = predict_with_sparse_heads(feat, pc, nc, heads)
            correct += (pred == labels_dict[ep])
    acc = correct / len(episodes)
    print(f"✅ Test accuracy = {acc:.3%}")

# ------------------------------------------------------------
# 4b. 评估 Top-k 曲线（一次性）
# ------------------------------------------------------------
def evaluate_curve(
    h5_path: str,
    episodes: list[str],
    labels_dict: dict[str, int],
    model,
    max_k: int = 32,
    step: int = 4,
    plot_file: str | None = "topk_curve.png",
):
    """一次性评估 k=1..max_k 的准确率并绘制曲线。

    通过向量化多数投票，避免重复遍历数据集 max_k 次。
    """

    pc, nc, head_order, agg = model["pos_cent"], model["neg_cent"], model["sel_heads"], model["agg"]

    # 若 sel_heads 没按准确率排序，则根据 head_acc 重新排序 ↓
    if "head_acc" in model:
        head_order = np.argsort(model["head_acc"])[::-1]  # 降序

    head_order = head_order[:max_k]

    # ----------1. 预计算每 episode × head 的投票 (+1/-1) ----------
    votes_all: list[np.ndarray] = []   # (N, max_k)
    with h5py.File(h5_path, "r") as f:
        for ep in tqdm(episodes, desc="precompute votes"):
            feat = episode_to_vec(load_episode(f, ep), agg)          # (144,256)
            # 取排序后所需头向量
            sel_feat = feat[head_order]                              # (K,256)

            # 余弦相似度正负 → +1 / -1 投票
            pos_sim = _cos(sel_feat, pc[head_order])
            neg_sim = _cos(sel_feat, nc[head_order])
            votes   = np.where(pos_sim > neg_sim, 1, -1)            # (K,)
            votes_all.append(votes)

    votes_mat = np.stack(votes_all, axis=0)      # (N, K)
    labels    = np.array([labels_dict[e] for e in episodes])  # (N,)

    # ----------2. 累积多数投票 & 计算准确率 ----------
    cum_votes = np.cumsum(votes_mat, axis=1)  # (N,K)

    # 仅保留每 step 个头的点：例如 4,8,...
    ks = np.arange(step, max_k + 1, step)
    idxs = ks - 1  # 索引从 0 开始

    preds_step = np.where(cum_votes[:, idxs] > 0, 1, 0)
    acc_k = (preds_step == labels[:, None]).mean(axis=0)  # (len(ks),)

    # ----------3. 绘图 ----------
    if plot_file is not None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5,3.5))
        plt.plot(ks, acc_k * 100, marker="o", linewidth=1.2, label="Accuracy")

        # 垂直黑线：训练阶段 100% 准确头数
        if "n_full" in model and model["n_full"] > 0:
            plt.axvline(model["n_full"], color="black", linestyle="--", label=f"Train 100% heads = {model['n_full']}")

        plt.xlabel("Top-k heads")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(alpha=.3)
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_file, dpi=120)
        plt.close()

        print(f"📈 Top-k 曲线已保存 → {plot_file}")

    return ks, acc_k

def save_selected_head_activations(
    h5_path: str,
    episode_keys: List[str],              # 要导出的 episode 列表
    sel_heads: List[int],                # 已选中的头索引 (一维 0~143)
    agg: str = "mean",                 # "mean" | "max" | "last"
    out_h5: str = "selected_heads_feat.h5",
    zero_fill: bool = True,              # True → 未选头用 0 填充；False → 仅保存选头
):
    """把指定 episode 的 Top-k 头激活保存到新的 H5 文件中。

    每个 episode 将保存为一个 dataset，名称用 "task_episode" 形式（把 "/" 替换为
    "__"），shape=(144,256)（或 (k,256) 若 ``zero_fill=False``）。
    """
    assert agg in {"mean", "max", "last"}, "agg 必须是 mean/max/last"

    with h5py.File(h5_path, "r") as fin, h5py.File(out_h5, "w") as fout:
        # 在文件属性里记录所选头及聚合方式，方便后续读取
        fout.attrs["sel_heads"] = json.dumps(sel_heads)
        fout.attrs["agg"] = agg

        for ep in tqdm(episode_keys, desc="Saving activations"):
            feat = episode_to_vec(load_episode(fin, ep), agg)  # (144,256)

            if zero_fill:
                # 还原三维形状 (layer, head, dim) = (18,8,256)
                num_layers, num_heads = 18, 8
                data = np.zeros((num_layers, num_heads, feat.shape[-1]), dtype=np.float32)
                # 展平后按 sel_heads 复制，再 reshape 回去
                data.reshape(-1, feat.shape[-1])[sel_heads] = feat[sel_heads]
            else:
                data = feat[sel_heads].astype(np.float32)      # (k,256)

            # 保持原 "task/episode" 层级结构：task -> episode -> dataset
            task_name, ep_name = ep.split("/", 1)  # task, episode_xxx
            task_grp = fout.require_group(task_name)
            ep_grp = task_grp.require_group(ep_name)

            # 写入/覆盖 dataset "selected_heads"
            if "selected_heads" in ep_grp:
                del ep_grp["selected_heads"]
            ep_grp.create_dataset("selected_heads", data=data, compression="gzip")

    print(f"✅ 已保存 {len(episode_keys)} 个 episode → {out_h5}")

# --------------------------- 示例调用 ---------------------------
# 下面示例展示如何把 run_sav 得到的 Top-k 头 (mean 聚合) 保存出来。
# 若你想导出 max/last，只需设置 agg="max"/"last" 并重新调用即可。
# ----------------------------------------------------------------
# example_out_h5 = "pick_topk_heads_mean.h5"
# save_selected_head_activations(
#     h5_path=ATTN_H5_PATH,
#     episode_keys=support_pos + support_neg,  # 或其它 episode 列表
#     sel_heads=sav_model["sel_heads"],
#     agg=sav_model["agg"],
#     out_h5=example_out_h5,
#     zero_fill=True,
# )



# ------------------------------------------------------------
# 7.  运行示例：Pick vs Non-Pick
# ------------------------------------------------------------
# 7.1  采样支持集
all_pos, all_neg = collect_episode_keys(ATTN_H5_PATH, PICK_TASKS)

random.seed(42)
support_pos = random.sample(all_pos, 20)
support_neg = random.sample(all_neg, 20)

sav_model = run_sav(ATTN_H5_PATH, support_pos, support_neg, k=4, agg="mean")

pickle.dump(sav_model, open("sav_pick.pkl", "wb"))

# example_out_h5 = "steer_train_wipe_top_20_heads_mean.h5"
# save_selected_head_activations(
#     h5_path=ATTN_H5_PATH,
#     episode_keys=support_pos,  # 或其它 episode 列表
#     sel_heads=sav_model["sel_heads"],
#     agg=sav_model["agg"],
#     out_h5=example_out_h5,
#     zero_fill=True,
# )

# 7.2  构造（示例）测试集标签
#     — 真实实验里你应有独立测试集；这里仅示范
EVAL_H5 = "pick_eval_attention_last_token_single_action_keyframe.h5"    
all_pos_eval, all_neg_eval = split_episode_keys(EVAL_H5, pos_keywords={"\\bpick\\b"}, exclude_kw={"\\bwipe\\b"}, train_task_set=PICK_TASKS)
print(len(all_pos_eval), len(all_neg_eval)) 
# print(all_pos_eval)
# print(all_neg_eval)
pos_eval = random.sample(all_pos_eval, 200)
neg_eval = random.sample(all_neg_eval, 200)
eval_eps    = pos_eval + neg_eval          # 或 balanced 抽样
eval_labels = {ep: (1 if ep in pos_eval else 0) for ep in eval_eps}
# plot_tsne_from_sav(EVAL_H5, eval_eps, eval_labels, sav_model, head="best")
evaluate(EVAL_H5, eval_eps, eval_labels, sav_model)
# ks, accs = evaluate_curve(EVAL_H5, eval_eps, eval_labels, sav_model,
#                           max_k=32, step=4, plot_file="topk_curve.png")



# ---------- Heatmap ----------


def plot_head_heatmap(acc: np.ndarray, sel_heads: List[int],
                      num_layers=18, num_heads=8, title=""):
    acc_mat = acc.reshape(num_layers, num_heads)
    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(acc_mat, cmap="RdYlBu_r", vmin=0., vmax=1.)
    ax.invert_yaxis()
    ax.set_xticks(range(num_heads)); ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
    ax.set_yticks(range(num_layers)); ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title, pad=12, weight="bold")

    # 红框高亮
    for idx in sel_heads:
        l, h = divmod(idx, num_heads)
        rect = patches.Rectangle((h-0.5, l-0.5), 1, 1, linewidth=1.8,
                                 edgecolor="black", facecolor="none")
        ax.add_patch(rect)

    ax.text(-0.4, -1.1,
            f"Active Heads: {len(sel_heads)}/{num_layers*num_heads}"
            f" ({len(sel_heads)/(num_layers*num_heads):.1%})",
            fontsize=8, bbox=dict(facecolor="lightgray", alpha=.7))
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig("SAV_head_heatmap/pick_head_heatmap_mean.png")
    plt.show()

plot_head_heatmap(sav_model["head_acc"], sav_model["sel_heads"],
                  title="Pick vs Non-Pick – Head Accuracy")

