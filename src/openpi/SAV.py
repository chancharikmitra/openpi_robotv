import hdf5plugin
import h5py
import json, random, re, pickle
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore

ATTN_H5_PATH = "pick_train_attention_last_token_keyframe_new.h5"
TASK_JSON    = "tasks.json"



import re, h5py
from typing import List, Tuple, Set, Iterable


def plot_tsne_from_sav(
    h5_path: str,
    episode_keys: List[str],          # list of episode keys to visualize
    labels: Dict[str, int],           # mapping {ep_key: 1/0}
    sav_model: Dict,
    head: str | int = "best",         # "best" | "all" | specific head idx
    perp: int = 30,
):
    """
    head:
        "best" → use the head with highest head_acc (default in paper)
        "all"  → concatenate all selected heads before feeding to t-SNE
        int    → specify a concrete head index
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
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0)
    X_2d = tsne.fit_transform(X)

    # ---- ③ Plot ----
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
    pos_keywords:   Set[str],       # {"\\bpick\\b"}   positive-class keywords (regex)
    exclude_kw:     Set[str],       # {"\\bwipe\\b"}   exclusion keywords (regex)
    train_task_set: Set[str],       # tasks used in training, for deduplication
    eval_task_set: Set[str],       # tasks reserved for evaluation, for deduplication
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
            if task_norm in eval_task_set:
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

WIPE_TASKS = set(task_dict["Wipe_training_tasks"])

EVAL_TASKS = set(task_dict["Pick_test_tasks"])

# ------------------------------------------------------------
# 0. split episode keys by task
# ------------------------------------------------------------
def collect_episode_keys(
    h5_path: str,
    pos_tasks: set,
) -> Tuple[List[str], List[str]]:
    """Return (pos_keys, neg_keys); key format: task/episode_xxx"""
    pos_keys, neg_keys = [], []
    with h5py.File(h5_path, "r") as h5:
        for task in h5.keys():
            # ① determine whether the task is positive or negative
            is_pos = task in pos_tasks
            # ② collect all episode groups
            for ep in h5[task].keys():            # ep = 'episode_000', ...
                key = f"{task}/{ep}"
                (pos_keys if is_pos else neg_keys).append(key)
    return pos_keys, neg_keys

# ------------------------------------------------------------
# 1.  load a single episode → (F, 18, 8, 256)                ★
# ------------------------------------------------------------
def load_episode(h5_file, ep_key):
    grp = h5_file[ep_key]
    frames = []
    for fk in sorted(grp.keys()):
        raw = grp[fk]["last_token_attn"]
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


def head_accuracy(feats: np.ndarray, labels: np.ndarray, pos_cent: np.ndarray, neg_cent: np.ndarray):
    """Compute training accuracy per head."""
    sim_pos = _cos(feats, pos_cent)
    sim_neg = _cos(feats, neg_cent)
    pred = (sim_pos > sim_neg).astype(np.int32)
    return (pred == labels[:, None]).mean(axis=0)


def head_similarity_margin(feats: np.ndarray, labels: np.ndarray, pos_cent: np.ndarray, neg_cent: np.ndarray):
    """Head score based on cosine-similarity margin (confidence)."""
    sim_pos = _cos(feats, pos_cent)
    sim_neg = _cos(feats, neg_cent)
    sign = np.where(labels == 1, 1, -1)[:, None]  # (N,1); +1 for positive, -1 for negative
    margin = (sim_pos - sim_neg) * sign          # (N,144)
    return margin.mean(axis=0)                   # (144,)


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

    heads = select_top_heads(score, k)
    #heads = select_bottom_heads(score, k)
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
    if sample_k is not None and heads.shape[0] > sample_k:
        np.random.seed(1)
        heads = heads[np.random.choice(heads.shape[0], size=sample_k, replace=False)]

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
all_pos, all_neg = collect_episode_keys(ATTN_H5_PATH, PICK_TASKS)

random.seed(42)
support_pos = random.sample(all_pos, 20)
support_neg = random.sample(all_neg, 20)    

sav_model = run_sav(ATTN_H5_PATH, support_pos, support_neg, k=20, agg="mean", sel_metric="margin")

pickle.dump(sav_model, open("sav_pick.pkl", "wb"))

# example_out_h5 = "steer_train_wipe_top_20_heads_mean.h5"
# save_selected_head_activations(
#     h5_path=ATTN_H5_PATH,
#     episode_keys=support_pos,  # or any other episode list
#     sel_heads=sav_model["sel_heads"],
#     agg=sav_model["agg"],
#     out_h5=example_out_h5,
#     zero_fill=True,
# )

# 7.2  Build (example) test set labels
#     — In real experiments you should provide an independent test set; this is just a demo
EVAL_H5 = "pick_eval_attention_last_token_single_action_keyframe.h5"    
all_pos_eval, all_neg_eval = split_episode_keys(EVAL_H5, pos_keywords={"\\bpick\\b"}, exclude_kw={"\\bwipe\\b"}, train_task_set=PICK_TASKS, eval_task_set=EVAL_TASKS)
print(len(all_pos_eval), len(all_neg_eval)) 
# print(all_pos_eval)
# print(all_neg_eval)
pos_eval = random.sample(all_pos_eval, 200)
neg_eval = random.sample(all_neg_eval, 200)
eval_eps    = pos_eval + neg_eval          # or a balanced sample
eval_labels = {ep: (1 if ep in pos_eval else 0) for ep in eval_eps}
# plot_tsne_from_sav(EVAL_H5, eval_eps, eval_labels, sav_model, head="all")
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

    # draw red rectangle highlight
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

