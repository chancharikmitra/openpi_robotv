# head_tuning 方法层 Release 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 openpi fork 之上的 "KNN 选头 + head-tuning 微调" 方法层重构为一个干净、可复现、可发布的 `openpi.head_tuning` 子包，并显式管理对 openpi 核心的 fork delta。

**Architecture:** 新建 `src/openpi/head_tuning/` 子包，4 阶段各有 CLI 入口（extract / select / configs / finetune-via-train.py）。KNN 代码按职责拆分并删除未用的降维/度量学习机制。openpi 核心改动原地保留 + 统一标记 + `UPSTREAM_CHANGES.md`；`delta_heads` steering 与 `serve_policy.py` 私有内容回退。

**Tech Stack:** Python, JAX/Flax (openpi), tyro (CLI), h5py, numpy, scikit-learn (仅 KNN/PCA-free 部分), pytest, conda env `openpi`。

## Global Constraints

- 运行任何 `python src/openpi/...` 用 `conda run -n openpi python ...`（不是 `uv run`）。
- fork 基准点：上游 `Physical-Intelligence/openpi`，merge-base `5bff19b`。所有 `git diff` 以此为基准。
- 不改 openpi 训练循环语义；`models/gemma.py` 的 `return_attention_heads` activation 提取路径必须逐值保持不变。
- 方法实际配置：`DIST_METRIC="cosine"`、`USE_PCA=False`、`USE_ZSCORE=False`、选头 `topk`/`best_add`。
- benchmark 命名用 `droid`（不是 `robot`）。
- Paper-release 标准：英文注释/docstring、完整类型标注、零死代码、零 `breakpoint()`、tyro+help、`__init__.py` 精选导出。
- 已知回归基准：`swap_green_red_cube_20` 选头结果 `best_k=30, cv_mse=0.021498`（cosine/无 PCA）。
- 规范文档：`docs/superpowers/specs/2026-06-24-head-tuning-release-refactor-design.md`。

---

### Task 1: 捕获回归基准（golden artifacts，必须在删除/移动旧代码前完成）

**Files:**
- Create: `tests/golden/knn_swap_green_red_cube.json`（select 基准）
- Create: `tests/golden/extract_droid_smoke.h5`（extract 基准，小样本）
- Create: `tests/golden/README.md`（如何重新生成 golden）

**Interfaces:**
- Produces: golden 文件供 Task 6（select 复现）与 Task 7（extract 逐值比对）做回归对照。

- [ ] **Step 1: 确认现有 attn H5 与一个小输入 H5 可用**

Run:
```bash
ls -la /scr2/yusenluo/openpi_robotv/temp_data/*swap_green_red_cube*pi05_action.h5 2>/dev/null
ls -la /scr2/yusenluo/openpi_robotv/temp_data/*_20.h5 2>/dev/null | head
```
Expected: 至少存在一个 `*_action.h5`（select 基准用）与一个 `*_20.h5`（extract 输入用）。
若缺失：在本任务停下，向用户索取这两份数据路径，作为人工 gate。

- [ ] **Step 2: 跑现有 KNN_regression.py 记录 golden 选头结果**

Run:
```bash
cd /scr2/yusenluo/openpi_robotv
KNN_UNIT_MODE=head KNN_TARGET=20 \
KNN_ATTN_H5=$(ls temp_data/*swap_green_red_cube*pi05_action.h5 | head -1) \
KNN_OUT_TXT=tests/golden/knn_swap_green_red_cube.txt \
conda run -n openpi python src/openpi/KNN_regression.py 2>&1 | tee tests/golden/knn_swap_run.log
```
Expected: 日志末尾 `cv_mse≈0.021498`、`best_k=30`、一组 20 个 `(layer,head)`。

- [ ] **Step 3: 把 golden 选头结果固化为 JSON**

将 Step 2 输出的 `trainable_head_indices`、`best_k`、`cv_mse` 抄入 `tests/golden/knn_swap_green_red_cube.json`：
```json
{
  "attn_h5_basename": "swap_green_red_cube_20_pi05_action.h5",
  "unit_mode": "head",
  "target": 20,
  "best_k": 30,
  "cv_mse": 0.021498,
  "cv_mse_tol": 1e-4,
  "trainable_head_indices": [[9,7],[9,4],[8,4],[9,2],[5,7],[7,6],[8,5],[9,1],[7,1],[1,5],[3,2],[9,0],[3,1],[9,5],[1,6],[4,7],[7,0],[8,6],[5,3],[1,3]]
}
```

- [ ] **Step 4: 生成 extract golden（小样本 attn H5）**

Run:
```bash
cd /scr2/yusenluo/openpi_robotv
TASK=swap_green_red_cube \
INPUT_H5=$(ls temp_data/*_20.h5 | head -1) \
ATTN_H5=tests/golden/extract_droid_smoke.h5 \
MAX_EPISODES=2 \
conda run -n openpi python src/openpi/generate_activation_dataset_on_robot.py 2>&1 | tail -20
```
Expected: 生成 `tests/golden/extract_droid_smoke.h5`，含 `last_token_attn (18,8,256)` 等数据集。
若无 GPU：标记为人工 gate，记录在 `tests/golden/README.md`，后续 Task 7 比对延后。

- [ ] **Step 5: 写 golden README**

`tests/golden/README.md` 写明每个 golden 文件的来源命令、生成环境（GPU/checkpoint）、用途（哪个 Task 比对）。

- [ ] **Step 6: Commit**

```bash
git add tests/golden/
git commit -m "test: capture KNN+extract golden baselines before head_tuning refactor"
```

---

### Task 2: 搭建 head_tuning 子包骨架 + 迁入 knn/

**Files:**
- Create: `src/openpi/head_tuning/__init__.py`
- Move: `src/openpi/knn/` → `src/openpi/head_tuning/knn/`（`git mv`）
- Modify: 所有 `from openpi.knn` / `import openpi.knn` 引用 → `openpi.head_tuning.knn`

**Interfaces:**
- Produces: `openpi.head_tuning.knn` 包，导出与原 `openpi.knn` 暂时一致（本任务只搬不改逻辑）。

- [ ] **Step 1: 建子包目录与空 __init__**

```bash
cd /scr2/yusenluo/openpi_robotv
mkdir -p src/openpi/head_tuning
printf '"""head_tuning: KNN-based attention head selection and head-tuning finetuning for pi0/pi05."""\n' > src/openpi/head_tuning/__init__.py
```

- [ ] **Step 2: git mv knn 包进子包**

```bash
git mv src/openpi/knn src/openpi/head_tuning/knn
```

- [ ] **Step 3: 找出所有 openpi.knn 引用**

Run:
```bash
grep -rn "openpi\.knn\|from openpi import knn\|import openpi\.knn" src/ scripts/ | grep -v "head_tuning/knn"
```
Expected: 列出 `KNN_regression.py`、`variance_selection.py`（将删，可忽略）等引用点。

- [ ] **Step 4: 批量改引用为 head_tuning.knn**

对 Step 3 列出的**保留**文件，把 `openpi.knn` → `openpi.head_tuning.knn`。`KNN_regression.py` 本任务先改引用使其可导入（下游 Task 5/6 再迁移它）。

- [ ] **Step 5: 验证可导入**

Run:
```bash
conda run -n openpi python -c "import openpi.head_tuning.knn as k; print(sorted(k.__all__)[:3])"
```
Expected: 打印前 3 个导出名，无 ImportError。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move knn package into openpi.head_tuning subpackage"
```

---

### Task 3: 拆分 knn/utils.py → data.py / inference.py / selection.py

**Files:**
- Create: `src/openpi/head_tuning/knn/data.py`
- Create: `src/openpi/head_tuning/knn/inference.py`
- Create: `src/openpi/head_tuning/knn/selection.py`
- Delete: `src/openpi/head_tuning/knn/utils.py`（拆完后删）
- Modify: `src/openpi/head_tuning/knn/__init__.py`

**Interfaces:**
- Produces:
  - `data.py`: `build_dataset(...) -> PreprocessedData`、`PreprocessedData`、`HeadPreprocessor`、`load_episode_frames`、`frame_to_vec`、`transform_frame`、`get_action_labels`、`save_action_labels`
  - `inference.py`: `weighted_avg`、`weighted_avg_batched`、`build_feat_subset`
  - `selection.py`: `rank_single_heads_per_k`、`rank_single_heads`、`simple_topk_select`、`greedy_forward_select`，以及（**暂时**搬入、Task 5 再删）`reinforce_select_heads`/`learn_head_weights`/`compute_loeo_mse_torch*`。本任务是纯机械搬运，不删函数。

- [ ] **Step 1: 按职责把 utils.py 的函数移入三个新文件**

参照规范 §2 的归属表。`utils.py` 函数清单（行号见原文件）：
- → `inference.py`: `weighted_avg`、`weighted_avg_batched`、`build_feat_subset`
- → `data.py`: `_coerce_float32`、`_iter_frame_keys`、`load_episode_frames`、`frame_to_vec`、`HeadPreprocessor`、`IdentityScaler`、`PreprocessedData`、`build_dataset`、`transform_frame`、`transform_episode`、`get_action_labels`、`save_action_labels`
- → `selection.py`: `rank_single_heads_per_k`、`rank_single_heads`、`simple_topk_select`、`greedy_forward_select`、`reinforce_select_heads`、`learn_head_weights`、`compute_loeo_mse_torch`、`compute_loeo_mse_torch_with_metric`

每个新文件顶部加模块 docstring + 必要 import。保持函数体此刻**不变**（纯搬运，便于回归）。

- [ ] **Step 2: 更新 __init__.py 导出指向新模块**

`__init__.py` 把 `from .utils import ...` 拆成 `from .data import ...` / `.inference` / `.selection`，`__all__` 不变。

- [ ] **Step 3: 删除空的 utils.py**

```bash
git rm src/openpi/head_tuning/knn/utils.py
```

- [ ] **Step 4: 验证导入与冒烟**

Run:
```bash
conda run -n openpi python -c "
from openpi.head_tuning.knn import build_dataset, weighted_avg, rank_single_heads, simple_topk_select
print('ok')"
```
Expected: `ok`，无 ImportError。

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: split knn/utils.py into data/inference/selection by responsibility"
```

---

### Task 4: 移除未用的降维/度量学习机制（PCA / zscore / whiten / proj / pls）

**Files:**
- Modify: `src/openpi/head_tuning/knn/metrics.py`（收缩为 cosine+euclidean 两个无状态距离函数）
- Modify: `src/openpi/head_tuning/knn/eval.py`、`selection.py`、`data.py`（删 metric_ctx/pca/zscore 参数）
- Modify: `src/openpi/head_tuning/knn/__init__.py`

**Interfaces:**
- Produces:
  - `metrics.py`: `pairwise_dist(q, base, metric="cosine") -> np.ndarray`、`pairwise_dist_matrix(Q, B, metric="cosine") -> np.ndarray`（`metric ∈ {"cosine","euclidean"}`）。**删除** `build_metric_ctx_from_train`、`build_global_metric_ctx_for_heads`。
  - `eval.py`: `evaluate_leave_one_episode_out(features, actions, episode_ids, frame_ids, heads, k, metric="cosine", temp_excl_w=...) -> float`（删 `pca_dim`/`proj_alpha`/`metric_scope`/`metric_ctx_global`）。
  - `data.py`: `build_dataset(attn_h5, episodes, *, num_layers, heads_per_layer, d_head) -> PreprocessedData`（删 `use_pca`/`use_zscore`/`pca_components`）。

- [ ] **Step 1: 收缩 metrics.py**

`pairwise_dist` / `pairwise_dist_matrix` 只留 `cosine`（默认）与 `euclidean` 两分支，删 `whiten`/`proj`/`pls` 分支、`metric_ctx` 参数、`build_metric_ctx_from_train`、`build_global_metric_ctx_for_heads`，以及 `PLSRegression`/`Ridge`/`PCA` import。错误信息改为 `metric must be 'cosine' or 'euclidean'`。

- [ ] **Step 2: 清 eval.py 的 metric/pca 参数**

从 `evaluate_leave_one_episode_out(_batched)`、`evaluate_head_subset_cross_validation`、`evaluate_model_on_h5`、`predict_episode` 删掉 `pca_dim`/`proj_alpha`/`metric_scope`/`metric_ctx*`/`use_fullspace` 形参与内部分支；删 `_weighted` 变体（属 learn_weights，Task 5 一并清）。距离调用改为 `pairwise_dist_matrix(Q, B, metric)`。

- [ ] **Step 3: 清 data.py 的 PCA/zscore**

`build_dataset` 删 `use_pca`/`use_zscore`/`pca_components`；`HeadPreprocessor`/`IdentityScaler` 退化为直通（或删除后让 `Xh_red == 原始特征`）。确保 `PreprocessedData.Xh_red` 仍是 `(N, H, d)`。

- [ ] **Step 4: 清 selection.py 的 metric/pca 参数**

`rank_single_heads(_per_k)`、`simple_topk_select`、`greedy_forward_select` 删掉 `pca_dim`/`proj_alpha`/`metric_scope`/`metric_ctx*`/`use_fullspace`/`H=...` 中与 metric_ctx 相关的透传，调用 `evaluate_leave_one_episode_out` 时不再传这些。

- [ ] **Step 5: 更新 __init__.py，移除已删符号**

从导出与 `__all__` 删 `build_metric_ctx_from_train`、`build_global_metric_ctx_for_heads`。

- [ ] **Step 6: 冒烟验证**

Run:
```bash
conda run -n openpi python -c "
from openpi.head_tuning.knn.metrics import pairwise_dist
import numpy as np
d = pairwise_dist(np.ones((1,4),'f'), np.eye(4,dtype='f'), 'cosine')
print('cosine ok', d.shape)"
```
Expected: `cosine ok (1, 4)`。

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: drop unused PCA/zscore/whiten/proj/pls metric machinery (cosine-KNN only)"
```

---

### Task 5: 上提编排逻辑到 knn/model.py + 只留 topk/best_add

**Files:**
- Create: `src/openpi/head_tuning/knn/model.py`
- Modify: `src/openpi/head_tuning/knn/selection.py`（删 reinforce/learn_weights/torch 变体）
- Modify: `src/openpi/head_tuning/knn/__init__.py`

**Interfaces:**
- Produces:
  - `model.py`: `@dataclass KnnRegModel`（字段：`heads:list[int]`、`k:int`、`metric:str`、`d_per_head:int`、`preproc`、`X_bank:np.ndarray`、`Y_bank:np.ndarray`）；`fit_knn_reg_with_heads(attn_h5, episodes, *, selection_mode="topk", target_heads, unit_mode, k_grid, metric="cosine", temp_excl_w, excluded_heads=None) -> tuple[KnnRegModel, dict]`。`dict` 含 `selected_heads`、`best_k`、`cv_mse`、`per_k_mse`。
- Consumes: `data.build_dataset`、`selection.{rank_single_heads,simple_topk_select,greedy_forward_select}`、`eval.evaluate_leave_one_episode_out`、`inference.build_feat_subset`。

- [ ] **Step 1: 把 KNN_regression.py 的 KnnRegModel + fit_knn_reg_with_heads 迁入 model.py**

只保留 `selection_mode ∈ {"topk","best_add"}` 两分支；删除 `reinforce`/`learn_weights` 分支、`head_weights`/`head_probabilities` 相关字段与逻辑、所有 metric_ctx/pca 透传（已在 Task 4 删）。全局常量（`UNIT_MODE`/`DIST_METRIC`/`K_GRID`/…）改为函数形参。英文化注释。

- [ ] **Step 2: selection.py 删除 reinforce/learn_weights/torch 变体**

`git`-删除 `reinforce_select_heads`、`learn_head_weights`、`compute_loeo_mse_torch`、`compute_loeo_mse_torch_with_metric` 函数体。

- [ ] **Step 3: 更新 __init__.py**

导出新增 `fit_knn_reg_with_heads`、`KnnRegModel`；移除 `reinforce_select_heads`。

- [ ] **Step 4: 冒烟（用 golden attn h5 跑一次 fit）**

Run:
```bash
cd /scr2/yusenluo/openpi_robotv
conda run -n openpi python -c "
import h5py, json
from openpi.head_tuning.knn.model import fit_knn_reg_with_heads
h5='$(ls temp_data/*swap_green_red_cube*pi05_action.h5 | head -1)'
with h5py.File(h5) as f:
    eps=[f'{t}/{e}' for t in f.keys() for e in f[t].keys()]
m,info=fit_knn_reg_with_heads(h5, eps, selection_mode='topk', target_heads=20, unit_mode='head', k_grid=[10,20,30,40], temp_excl_w=30)
print('cv_mse', round(info['cv_mse'],6), 'best_k', info['best_k'])"
```
Expected: `cv_mse 0.021498 best_k 30`（与 golden 一致，容差 1e-4）。

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: lift KnnRegModel/fit into knn/model.py, keep only topk+best_add"
```

---

### Task 6: select.py tyro CLI + heads.json（回归门：复现 cv_mse）

**Files:**
- Create: `src/openpi/head_tuning/select.py`
- Create: `tests/head_tuning/test_select_regression.py`
- Delete: `src/openpi/KNN_regression.py`

**Interfaces:**
- Consumes: `knn.model.fit_knn_reg_with_heads`。
- Produces: CLI `python -m openpi.head_tuning.select`；输出 `heads.json`：`{"trainable_head_indices": [[l,h],...], "best_k": int, "cv_mse": float, "unit_mode": str, "target": int, "attn_h5": str}`。

- [ ] **Step 1: 写回归测试（先失败）**

`tests/head_tuning/test_select_regression.py`：
```python
import json, subprocess, sys, glob, pathlib

def test_select_reproduces_golden(tmp_path):
    golden = json.loads(pathlib.Path("tests/golden/knn_swap_green_red_cube.json").read_text())
    attn = glob.glob("temp_data/*swap_green_red_cube*pi05_action.h5")[0]
    out = tmp_path / "heads.json"
    subprocess.run([sys.executable, "-m", "openpi.head_tuning.select",
                    "--attn-h5", attn, "--unit-mode", "head",
                    "--target", "20", "--out", str(out)], check=True)
    got = json.loads(out.read_text())
    assert got["best_k"] == golden["best_k"]
    assert abs(got["cv_mse"] - golden["cv_mse"]) < golden["cv_mse_tol"]
    assert [list(p) for p in got["trainable_head_indices"]] == golden["trainable_head_indices"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n openpi python -m pytest tests/head_tuning/test_select_regression.py -v`
Expected: FAIL（`No module named openpi.head_tuning.select`）。

- [ ] **Step 3: 写 select.py CLI**

```python
"""Stage 2 CLI: KNN-based attention head selection.

Reads an activation H5 (produced by `openpi.head_tuning.extract`) and writes a
heads.json describing the selected (layer, head) indices for head-tuning.
"""
import dataclasses, json
import h5py, tyro
from openpi.head_tuning.knn.model import fit_knn_reg_with_heads


@dataclasses.dataclass
class Args:
    attn_h5: str
    """Path to the activation H5 produced by the extract stage."""
    out: str = "heads.json"
    """Output path for the selected-heads JSON."""
    unit_mode: str = "head"
    """'head' (144 heads) or 'layer' (18 layers)."""
    target: int = 20
    """Number of heads (or layers) to select."""
    metric: str = "cosine"
    """Distance metric: 'cosine' or 'euclidean'."""
    selection_mode: str = "topk"
    """'topk' or 'best_add'."""
    k_grid: tuple[int, ...] = (10, 20, 30, 40)
    """Candidate k values for KNN."""
    temp_excl_w: int = 30
    """Leave-one-frame-out temporal exclusion window (+/- frames)."""


def main(args: Args) -> None:
    with h5py.File(args.attn_h5, "r") as f:
        episodes = [f"{t}/{e}" for t in f.keys() for e in f[t].keys()]
    _, info = fit_knn_reg_with_heads(
        args.attn_h5, episodes, selection_mode=args.selection_mode,
        target_heads=args.target, unit_mode=args.unit_mode,
        k_grid=list(args.k_grid), metric=args.metric, temp_excl_w=args.temp_excl_w,
    )
    sel = info["selected_heads"]
    heads = ([[l, h] for l in sel for h in range(8)] if args.unit_mode == "layer"
             else [[idx // 8, idx % 8] for idx in sel])
    out = {"trainable_head_indices": heads, "best_k": info["best_k"],
           "cv_mse": info["cv_mse"], "unit_mode": args.unit_mode,
           "target": args.target, "attn_h5": args.attn_h5}
    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=2)
    print(f"[select] wrote {args.out}: best_k={info['best_k']} cv_mse={info['cv_mse']:.6f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 4: 删除旧 KNN_regression.py**

```bash
git rm src/openpi/KNN_regression.py
```

- [ ] **Step 5: 跑测试确认通过**

Run: `conda run -n openpi python -m pytest tests/head_tuning/test_select_regression.py -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add head_tuning.select CLI; remove KNN_regression.py (regression-gated)"
```

---

### Task 7: extract 适配器 (base/droid/libero) + extract.py CLI（回归门：逐值比对）

**Files:**
- Create: `src/openpi/head_tuning/adapters/__init__.py`
- Create: `src/openpi/head_tuning/adapters/base.py`
- Create: `src/openpi/head_tuning/adapters/droid.py`
- Create: `src/openpi/head_tuning/adapters/libero.py`
- Create: `src/openpi/head_tuning/extract.py`
- Create: `tests/head_tuning/test_extract_regression.py`
- Delete: `src/openpi/generate_activation_dataset_on_robot.py`、`src/openpi/generate_activation_dataset_on_libero.py`

**Interfaces:**
- Produces:
  - `base.py`: `extract_key_idcs(...)` 的共享骨架 + `run_inference_and_save(policy, episodes, out_h5)`（封装 `policy.infer(return_attention_heads=True, return_state_and_first_action_heads=True)` 与 H5 写入 schema：`last_token_attn`/`state_token_attn`/`first_action_token_attn`/`action_label`/`predicted_action_label`）。
  - `droid.py`: `load_droid_episodes(input_h5, task_prompt, max_episodes) -> dict`、`droid_key_idcs(...)`。
  - `libero.py`: `load_libero_episodes(src, task_slug, ...) -> dict`、`libero_key_idcs(...)`。
  - `extract.py`: CLI `python -m openpi.head_tuning.extract --benchmark {droid,libero} ...`。
- Consumes: `openpi.training.config.get_config("pi05_droid")`、`openpi.policies.policy_config.create_trained_policy`。

- [ ] **Step 1: 写逐值比对回归测试（先失败）**

`tests/head_tuning/test_extract_regression.py`：
```python
import subprocess, sys, glob, pathlib
import h5py, numpy as np, pytest

GOLDEN = pathlib.Path("tests/golden/extract_droid_smoke.h5")

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden extract H5 not generated (needs GPU)")
def test_extract_droid_matches_golden(tmp_path):
    inp = sorted(glob.glob("temp_data/*_20.h5"))[0]
    out = tmp_path / "out.h5"
    subprocess.run([sys.executable, "-m", "openpi.head_tuning.extract",
                    "--benchmark", "droid", "--input-h5", inp,
                    "--out-h5", str(out), "--task-prompt", "swap green red cube",
                    "--max-episodes", "2"], check=True)
    with h5py.File(GOLDEN) as g, h5py.File(out) as o:
        gkeys = sorted(k for k in _leaf_groups(g))
        okeys = sorted(k for k in _leaf_groups(o))
        assert gkeys == okeys
        for k in gkeys:
            np.testing.assert_allclose(g[k]["last_token_attn"][:], o[k]["last_token_attn"][:],
                                       rtol=1e-5, atol=1e-5)

def _leaf_groups(h):
    out=[]
    h.visititems(lambda n,obj: out.append(n) if isinstance(obj,h5py.Group) and "last_token_attn" in obj else None)
    return out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n openpi python -m pytest tests/head_tuning/test_extract_regression.py -v`
Expected: FAIL（`No module named openpi.head_tuning.extract`）或 skip（无 golden）。

- [ ] **Step 3: 实现 base.py 共享逻辑**

从 `generate_activation_dataset_on_robot.py` 抽出与 benchmark 无关的部分：`extract_key_idcs` 的通用部分、`policy.infer(...)` 调用、把 `attention_outputs` 切片为 last/state/first-action 并写入 H5 的逻辑（原文件 330-465 行）。删掉所有 `print(...shape...)`、注释掉的直方图/decode_probs 代码、`breakpoint()`。英文化。

- [ ] **Step 4: 实现 droid.py**

从 `generate_activation_dataset_on_robot.py` 移入 `extract_observations`（重命名 `load_droid_episodes`）与 droid 专属 keyframe 规则（gripper flip + 手臂静止）。环境变量 `TASK/INPUT_H5/ATTN_H5/TASK_PROMPT` → 函数参数。

- [ ] **Step 5: 实现 libero.py**

从 `generate_activation_dataset_on_libero.py` 移入 parquet 读取（`load_libero_episodes`）与 EE-pose keyframe 规则。环境变量 → 参数。

- [ ] **Step 6: 实现 extract.py CLI（tyro，`--benchmark` 分派）**

```python
"""Stage 1 CLI: extract per-head last-token activations from a pi05 policy."""
import dataclasses, tyro
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.head_tuning.adapters import droid as _droid, libero as _libero, base as _base


@dataclasses.dataclass
class Args:
    benchmark: str            # "droid" | "libero"
    input_h5: str = ""        # droid: source H5
    src: str = ""             # libero: parquet root
    out_h5: str = "activations.h5"
    task_prompt: str = ""
    max_episodes: int = 200
    use_keyframe: bool = True


def main(args: Args) -> None:
    cfg = _config.get_config("pi05_droid")
    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid/assets")
    policy = policy_config.create_trained_policy(cfg, ckpt)
    if args.benchmark == "droid":
        episodes = _droid.load_droid_episodes(args.input_h5, args.task_prompt, args.max_episodes)
        key_fn = _droid.droid_key_idcs
    elif args.benchmark == "libero":
        episodes = _libero.load_libero_episodes(args.src, args.task_prompt, args.max_episodes)
        key_fn = _libero.libero_key_idcs
    else:
        raise ValueError("benchmark must be 'droid' or 'libero'")
    _base.run_inference_and_save(policy, episodes, args.out_h5,
                                 key_fn=key_fn if args.use_keyframe else None)


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 7: 删除两个旧生成器**

```bash
git rm src/openpi/generate_activation_dataset_on_robot.py src/openpi/generate_activation_dataset_on_libero.py
```

- [ ] **Step 8: 跑回归测试**

Run: `conda run -n openpi python -m pytest tests/head_tuning/test_extract_regression.py -v`
Expected: PASS（有 golden+GPU 时）或 skip（无 GPU，记入人工 gate）。

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: add head_tuning.extract CLI with droid/libero adapters (byte-compare gated)"
```

---

### Task 8: configs.py 工厂 + 示例 config + 清理 config.py

**Files:**
- Create: `src/openpi/head_tuning/configs.py`
- Modify: `src/openpi/training/config.py`（删实验 config + 死注释，留 droid/libero 各 1 示例）

**Interfaces:**
- Produces: `configs.py`: `make_head_tuning_config(name, repo_id, heads_json_or_indices, *, pi05=True, num_train_steps=3000, ...) -> TrainConfig`（封装 §1391-1429 那条 config 的样板：`Pi0Config(gemma_2b_lora/gemma_300m_lora)`、`AdamWForHeadTuning(freeze_kv=True, freeze_mlp=False, trainable_head_indices=...)`、`get_freeze_filter_always_freeze_expert_and_siglip()`、cosine LR）。
- Consumes: `head_tuning.select` 输出的 `heads.json`（可传路径或直接传 indices 列表）。

- [ ] **Step 1: 实现 configs.py 工厂**

把选中的 config 模板参数化；支持 `heads` 传 `list[tuple[int,int]]` 或 `heads.json` 路径（内部读 `trainable_head_indices`）。英文 docstring + 类型标注。

- [ ] **Step 2: 在 config.py 注册两条示例并删冗余**

`_CONFIGS` 里只保留：1 条 `pi05_head_tuning_droid_example`（调 `make_head_tuning_config`）+ 1 条 `pi05_head_tuning_libero_example`。删除其余几十条实验性 head-tuning TrainConfig 与所有注释掉的 `# trainable_head_indices=[...]` 块。

- [ ] **Step 3: 验证 config 能 build**

Run:
```bash
conda run -n openpi python -c "
from openpi.training import config as c
cfg = c.get_config('pi05_head_tuning_droid_example')
print(type(cfg.optimizer).__name__, len(cfg.optimizer.trainable_head_indices))"
```
Expected: `AdamWForHeadTuning 20`（或示例设定的头数）。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add head_tuning.configs factory; prune experimental configs from config.py"
```

---

### Task 9: 移除 delta_heads steering（gemma/policy）+ finetune smoke

**Files:**
- Modify: `src/openpi/models/gemma.py`（摘 delta_heads 注入分支与参数）
- Modify: `src/openpi/policies/policy.py`（摘 infer 的 delta_heads 参数）

**Interfaces:**
- 不变（保留）：`Attention.__call__(..., return_attention_heads, return_attention_probs)`、`policy.infer(..., return_attention_heads, return_attention_probs, return_state_and_first_action_heads)`。
- 移除：`delta_heads`、`delta_token_index` 形参及其 scan/vmap 透传、`infer` 的 `delta_heads`。

- [ ] **Step 1: 从 gemma.py 摘 delta_heads**

删除 `Attention.__call__` 的 `delta_heads`/`delta_token_index` 形参、注入分支（原 274-284 行）、`Block.__call__` 与 scan/vmap `in_axes` 中对 `delta_heads` 的透传（原 359-465 区段）。**保留** `return_attention_heads`/`attention_heads = encoded` 路径不动。

- [ ] **Step 2: 从 policy.py 摘 delta_heads**

删除 `infer` 的 `delta_heads` 形参与 `if delta_heads is not None:` 透传块；保留 `return_attention_heads`/`return_attention_probs`。

- [ ] **Step 3: extract 回归仍通过（证明只摘了 steering）**

Run: `conda run -n openpi python -m pytest tests/head_tuning/test_extract_regression.py -v`
Expected: PASS（activation 输出逐值不变）或 skip。

- [ ] **Step 4: finetune smoke（masked optimizer 能 build + 跑几步）**

Run:
```bash
cd /scr2/yusenluo/openpi_robotv
conda run -n openpi python scripts/train.py pi05_head_tuning_droid_example \
  --num-train-steps 2 --batch-size 2 --save-interval 1000 --exp-name smoke_$$ 2>&1 | tail -30
```
Expected: 日志出现 masked optimizer 初始化、跑完 2 步无异常（若数据/GPU 不可用，记人工 gate）。

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove delta_heads steering from gemma/policy (activation path unchanged)"
```

---

### Task 10: openpi 核心改动标记 + 回退 serve_policy/compute_norm_stats + UPSTREAM_CHANGES.md

**Files:**
- Modify: `models/gemma.py`、`models/pi0.py`、`models/pi0_config.py`、`policies/policy.py`、`training/optimizer.py`、`scripts/train.py`（统一标记）
- Revert: `scripts/serve_policy.py`、`scripts/compute_norm_stats.py`（回退上游）
- Create: `docs/UPSTREAM_CHANGES.md`

**Interfaces:**
- Produces: `docs/UPSTREAM_CHANGES.md` —— 逐文件 fork delta 权威清单。

- [ ] **Step 1: 给每个核心改动 hunk 加统一标记**

在每段我们改的代码块首尾加 `# [head_tuning] BEGIN ... # [head_tuning] END`（或单行 `# [head_tuning]`），重点补齐当前无标记的 `optimizer.py`（`AdamWForHeadTuning`/`_create_head_tuning_mask`）、`train.py`（`_create_masked_optimizer_for_head_tuning` + masked update）、`pi0_config.py`（`get_freeze_filter_always_freeze_expert_and_siglip`）。

- [ ] **Step 2: 回退 serve_policy.py 到上游**

```bash
cd /scr2/yusenluo/openpi_robotv
git checkout 5bff19b -- scripts/serve_policy.py
```
验证：`git diff 5bff19b HEAD -- scripts/serve_policy.py` 为空。

- [ ] **Step 3: 核查 compute_norm_stats.py**

Run: `git diff 5bff19b HEAD -- scripts/compute_norm_stats.py`
若改动非方法必需：`git checkout 5bff19b -- scripts/compute_norm_stats.py`。否则加 `# [head_tuning]` 标记并在 UPSTREAM_CHANGES 记录。

- [ ] **Step 4: 生成 UPSTREAM_CHANGES.md**

逐文件 `git diff 5bff19b HEAD -- <file>` 派生，写每个核心文件："改了什么 / 为什么 / 对应方法阶段"。覆盖 gemma/pi0/pi0_config/policy/optimizer/train。

- [ ] **Step 5: 验收 fork delta**

Run:
```bash
git diff --stat 5bff19b HEAD -- src/openpi/models src/openpi/policies src/openpi/training/optimizer.py scripts/train.py scripts/serve_policy.py
```
Expected: serve_policy 不出现；其余文件与 UPSTREAM_CHANGES.md 列表一致。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: mark openpi core fork delta, revert serve_policy, add UPSTREAM_CHANGES.md"
```

---

### Task 11: 执行清理清单 + 更新 .gitignore

**Files:**
- Delete: 规范 §6.1 全部条目
- Modify: `.gitignore`

**Interfaces:**
- Produces: 干净工作区；`.gitignore` 覆盖临时产物。

- [ ] **Step 1: 删除 src/openpi 下的替代方法与工具文件**

```bash
cd /scr2/yusenluo/openpi_robotv
git rm -f src/openpi/SAV.py src/openpi/CMA_selection.py src/openpi/variance_selection.py \
  src/openpi/inspect_h5.py src/openpi/llm_instruction_verb_filter.py src/openpi/convert2lerobot.py \
  "src/openpi/models/gemma_fast copy.py"
rm -f src/openpi/knn.sh; rm -rf src/openpi/keyframe_gifs src/openpi/slurm_output
```

- [ ] **Step 2: 删除 scripts 下 debug/verify/CALVIN 脚本**

```bash
git rm -f scripts/debug_mask.py scripts/debug_print_params.py scripts/compare_mask_methods.py \
  scripts/test_mask_application.py scripts/compare_checkpoints.py scripts/train_test.py \
  scripts/verify_head_tuning.py scripts/select_heads_from_opt_state.py \
  scripts/extract_calvin_push5_subset.py scripts/split_calvin_by_task.py \
  scripts/eval_calvin_push5.sbatch scripts/eval_calvin_rlinf_sft.sbatch \
  scripts/sft_calvin_push5_knnlora.sbatch scripts/auto_submit_eval.sh \
  scripts/setup_libero_eval_env.sh scripts/2506.09937v1.pdf
```

- [ ] **Step 3: 删除根目录临时产物**

```bash
rm -f all_heads_rank_ablation.sh eccv*.sh eval_libero10.sh layer_ablation.sh libero_finetune.sh \
  run_swap_green_red_cube.sh head_selection_*.txt layer_selection_*.txt dataset_structure*.txt \
  gripper_diff_hist.png extract_single_episode.py organize_data_into_train_format_optimized.py \
  environment-openpi-history.yml knn.sh
rm -rf temp_data logs third_party
git add -A
```

- [ ] **Step 4: 更新 .gitignore**

追加：
```gitignore
# head_tuning release: local artifacts
temp_data/
logs/
wandb/
slurm_output/
*.h5
head_selection_*.txt
layer_selection_*.txt
*_structure*.txt
checkpoints/
.cache/
```
（注意：`tests/golden/*.h5` 若需纳入版本控制，用 `!tests/golden/` 例外或改放 Git LFS——在本步确认后决定。）

- [ ] **Step 5: 验证导入无残留断链**

Run:
```bash
conda run -n openpi python -c "import openpi.head_tuning.select, openpi.head_tuning.extract; print('ok')"
grep -rn "openpi\.knn\b\|generate_activation_dataset\|KNN_regression\|import SAV\|CMA_selection\|variance_selection" src/ scripts/ || echo "no dangling refs"
```
Expected: `ok` 且 `no dangling refs`。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove research detritus and CALVIN/alt-method files; update .gitignore"
```

---

### Task 12: 文档（head_tuning README + 顶层 README 指引）

**Files:**
- Create: `src/openpi/head_tuning/README.md`
- Modify: `README.md`（顶层加一节）

**Interfaces:**
- Produces: 4 阶段贯穿教程 + fork delta 指引。

- [ ] **Step 1: 写 head_tuning/README.md**

四节：① extract（droid+libero 命令）② select（heads.json）③ configs（注入 + 注册）④ finetune（`scripts/train.py <config>`）。附"openpi 核心改动"一节指向 `docs/UPSTREAM_CHANGES.md`，并说明 `gemma.py`/`pi0.py` 的 activation 提取与 `AdamWForHeadTuning` masked optimizer 原理。

- [ ] **Step 2: 顶层 README 加指引节**

在 `README.md` 末尾加 "## Attention Head-Tuning (this fork)" 一节，1 段话 + 指向 `src/openpi/head_tuning/README.md` 与 `docs/UPSTREAM_CHANGES.md`。openpi 原文不动。

- [ ] **Step 3: 端到端命令走查（doc 自检）**

按 README 命令在脑中/实跑串一遍 extract→select→configs→train，确认参数名与 CLI `--help` 一致：
```bash
conda run -n openpi python -m openpi.head_tuning.extract --help | head
conda run -n openpi python -m openpi.head_tuning.select --help | head
```
Expected: help 文本与 README 命令一致。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: add head_tuning README and top-level pointer"
```

---

## 验收清单（全部任务后）

- [ ] `conda run -n openpi python -m pytest tests/head_tuning -v` 全绿（extract 在无 GPU 时 skip）。
- [ ] `conda run -n openpi python -m pytest src/openpi/transforms_test.py -v` 仍绿。
- [ ] `git diff 5bff19b HEAD -- scripts/serve_policy.py` 为空。
- [ ] `git status` 工作区干净（无 untracked 临时产物）。
- [ ] `grep -rn "breakpoint()\|build_metric_ctx\|reinforce_select\|delta_heads" src/openpi/head_tuning src/openpi/models/gemma.py` 无残留（gemma 中 delta_heads 已摘）。
- [ ] `docs/UPSTREAM_CHANGES.md` 与核心文件 diff 一一对应。
