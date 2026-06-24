# head_tuning 方法层 Release 重构设计

- **日期**: 2026-06-24
- **状态**: 已批准，待实现
- **范围**: 把 openpi fork 之上的"KNN 选头 + head-tuning 微调"方法层重构为一个干净、可复现、可发布的子包。

---

## 1. 目标与边界

### 1.1 要交付的
openpi 之上一个干净、可复现的 `head_tuning` 方法层，4 阶段 pipeline 各有明确 CLI 入口和文档：

1. **提取 activation** — 对预训练 pi05 policy 逐帧 inference，抽取每层每头的 last-token activation。
2. **KNN 选头** — 用 Leave-One-Episode-Out KNN 回归评估每个头，选出 task/action 相关的头。
3. **写 config** — 把选中的 `trainable_head_indices` 注入 TrainConfig。
4. **head-tuning 微调** — 用 masked optimizer 只训练选中的头。

### 1.2 不动的（方法根基，原样保留，只补文档）
- `scripts/train.py` 的训练循环与 `_create_masked_optimizer_for_head_tuning`。
- `training/optimizer.py` 的 `AdamWForHeadTuning` 与 `_create_head_tuning_mask`。
- `training/config.py` 的 head-tuning 相关字段。
- `models/gemma.py` 与 `models/pi0.py` 中所有 `@yusen` 标记的 activation 提取 / `delta_heads` 注入改动。

### 1.3 发布范围
- **benchmark**: droid 为主、libero 保留；CALVIN 全部清除。
- **选头方法**: 只发布 KNN 主方法。SAV / CMA / variance 三个替代方法清除。
- **文档**: 只讲方法层贡献；openpi 原生 README 不动，仅加一节指向方法层文档。

---

## 2. 子包结构

```
src/openpi/head_tuning/
  __init__.py
  extract.py              # 阶段① CLI (tyro)
  adapters/
    __init__.py
    base.py               # 共享: keyframe 抽取 + policy.infer + H5 写入 schema
    droid.py              # DROID H5 读取  (源自 generate_activation_dataset_on_robot.py)
    libero.py             # LeRobot parquet 读取 (源自 generate_activation_dataset_on_libero.py)
  select.py               # 阶段② 瘦 CLI (tyro)，取代 KNN_regression.py 的 __main__
  knn/
    __init__.py           # 精选 public API
    data.py               # build_dataset / PreprocessedData / HeadPreprocessor /
                          #   load_episode_frames / frame_to_vec / action_labels  ← IO + 预处理
    inference.py          # weighted_avg(_batched) / build_feat_subset / KNN 预测核  ← 推理原语
    selection.py          # rank_single_heads / topk / greedy(best_add)  ← 选头算法
    metrics.py            # 距离与度量学习（保留，清理）
    eval.py               # LOEO 评估（保留，裁掉没用到的 torch 变体）
    model.py              # KnnRegModel + fit_knn_reg_with_heads（从 KNN_regression.py 上提）
    viz.py                # 可视化（保留，可选依赖）
  configs.py              # 阶段③ 参数化的 steering TrainConfig 工厂
  README.md               # 4 阶段贯穿文档
```

---

## 3. 四个阶段入口

| 阶段 | 入口 | 输入 → 输出 |
|---|---|---|
| ① 提取 activation | `python -m openpi.head_tuning.extract --benchmark droid --input-h5 ... --out-h5 ...` | 机器人数据 → activation H5 `(18,8,256)` |
| ② KNN 选头 | `python -m openpi.head_tuning.select --attn-h5 ... --out heads.json` | activation H5 → `trainable_head_indices` + best_k + cv_mse |
| ③ 写 config | `head_tuning/configs.py` 工厂把 `heads.json` 注入 TrainConfig | heads.json → 注册的 TrainConfig |
| ④ head-tuning 微调 | `python scripts/train.py <config_name>`（沿用 openpi 现有训练） | TrainConfig → checkpoint |

### 3.1 阶段①：extract（adapter 模式）
- `base.py` 收两个生成器 ~45% 重叠的逻辑：keyframe 抽取、`policy.infer(return_attention_heads=True, ...)`、H5 写入 schema（`last_token_attn` / `state_token_attn` / `first_action_token_attn` / `action_label` / `predicted_action_label`）。
- `droid.py` / `libero.py` 各自实现 benchmark 特有的数据读取：
  - droid: H5 nested group 结构，joint-space keyframe 规则（gripper flip + 手臂静止）。
  - libero: LeRobot parquet，EE-pose keyframe 规则（gripper sign flip + EE-pose delta）。
- 环境变量（`TASK` / `INPUT_H5` / `ATTN_H5` / `LIBERO_SRC` 等）与硬编码路径 → tyro 参数。
- benchmark 名为 `droid`（不再叫 `robot`）。

### 3.2 阶段②：select（封装 knn/）
- 把 `KNN_regression.py` 的 `__main__` 逻辑变成正式 tyro CLI；删除其中所有注释掉的死代码（约占文件大半）。
- 模块级全局配置（`UNIT_MODE` / `DIST_METRIC` / `K_GRID` / `TEMP_EXCL_W` 等）→ CLI 参数或一个 dataclass config。
- 输出结构化 `heads.json`：`{trainable_head_indices, best_k, cv_mse, unit_mode, attn_h5, ...}`。

### 3.3 阶段③：configs.py
- 不做一键化（保持"看 KNN 结果再写 config"的手动串联，YAGNI）。
- 提供 TrainConfig 工厂 + 1 个 droid 示例 config + 1 个 libero 示例 config。
- 替换掉 `config.py` 里几十条注释掉的实验 config。

### 3.4 阶段④：finetune
- 零改动，沿用 `scripts/train.py` + `config.cli()`。
- README 写清楚怎么从 `heads.json` → 注入 config → 跑 train.py。

---

## 4. KNN 重构细节

现状问题：
- `knn/utils.py`（1072 行）是杂货铺，混了 KNN 推理原语、4 种选头算法、torch LOEO、数据加载/预处理、IO。
- `KNN_regression.py`（778 行）的 `__main__` 大半是注释掉的死代码。

重构动作：
1. `utils.py` 按职责拆成 **data.py / inference.py / selection.py**。
2. `KNN_regression.py` 的编排逻辑（`fit_knn_reg_with_heads` + `KnnRegModel`）上提到 `knn/model.py`。
3. `select.py` 只剩一个 tyro CLI。
4. **选头算法只保留 topk + best_add**（config 实际用 topk，best_add 为 greedy 备用）；删除 `reinforce_select_heads` 与 `learn_head_weights`。
5. `eval.py` 裁掉没用到的 torch LOEO 变体（`compute_loeo_mse_torch*`，若确认未被引用）。
6. `metrics.py` / `viz.py` 保留并清理。

---

## 5. 清理清单

### 5.1 删除
- **根目录**: `all_heads_rank_ablation.sh`、`eccv*.sh`、`eval_libero10.sh`、`layer_ablation.sh`、`libero_finetune.sh`、`run_swap_green_red_cube.sh`、所有 `head_selection_*.txt`、`layer_selection_*.txt`、`dataset_structure*.txt`、`gripper_diff_hist.png`、`temp_data/`、`logs/`、`extract_single_episode.py`、`organize_data_into_train_format_optimized.py`、`environment-openpi-history.yml`、`third_party/`（calvin + RLinf）。
- **src/openpi/**: `SAV.py`、`CMA_selection.py`、`variance_selection.py`、`inspect_h5.py`、`llm_instruction_verb_filter.py`、`convert2lerobot.py`、`generate_activation_dataset_on_robot.py`、`generate_activation_dataset_on_libero.py`（内容迁入 adapters 后删）、`KNN_regression.py`（迁入 model.py + select.py 后删）、`"gemma_fast copy.py"`、`knn.sh`、`keyframe_gifs/`、`slurm_output/`。
- **scripts/**: `debug_mask.py`、`debug_print_params.py`、`compare_mask_methods.py`、`test_mask_application.py`、`compare_checkpoints.py`、`train_test.py`、`verify_head_tuning.py`、`select_heads_from_opt_state.py`、CALVIN 相关（`extract_calvin_push5_subset.py`、`split_calvin_by_task.py`、`eval_calvin_*.sbatch`、`sft_calvin_push5_knnlora.sbatch`、`auto_submit_eval.sh`、`setup_libero_eval_env.sh`）、`2506.09937v1.pdf`。

### 5.2 保留
- `environment-openpi.yml`（作为 release 依赖凭证；删 `-history` 版本）。
- openpi 原生数据/训练管线、`transforms.py`、`scripts/compute_norm_stats.py`、`scripts/serve_policy.py`、`scripts/extract_libero_task_subset.py`（libero 数据子集抽取）。

### 5.3 .gitignore 补充
```
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

---

## 6. 文档

- `src/openpi/head_tuning/README.md`：4 阶段贯穿教程，给 droid + libero 两套具体命令；说明 `gemma.py` / `pi0.py` 的 activation 改动和 head-tuning masked optimizer 原理。
- 顶层 `README.md`：只加一节指向 `head_tuning/README.md`，openpi 原文不动。

---

## 7. 验证策略（重构必须证明行为不变）

- **extract**: 重构后 `extract --benchmark droid` 在小样本上跑，输出 H5 的 `last_token_attn` 等数据集与原 `generate_activation_dataset_on_robot.py` 输出**逐值比对**（相同 keyframe、相同 activation）；libero 同理。
- **select**: 在现有 attn H5 上跑重构后的 `select`，复现已知结果——`swap_green_red_cube_20` 的 `best_k=30, cv_mse=0.021498` 及那组 20 个 head。
- **finetune**: steering TrainConfig 能正常 build，`scripts/train.py` 能构出 masked optimizer（几步 smoke run，不跑满训练）。
- 现有测试（`transforms_test.py` 等）保持绿。

---

## 8. 实现顺序建议

1. 建子包骨架 + 把 `knn/` 迁入并按职责拆分（data/inference/selection/model），裁掉 reinforce/learn_weights 与未用 torch 变体。
2. `select.py` 瘦 CLI + `heads.json` 输出；用 `swap_green_red_cube` 复现 cv_mse 验证。
3. `extract.py` + adapters（base/droid/libero）；逐值比对验证。
4. `configs.py` 工厂 + droid/libero 示例 config；清理 `config.py` 死注释；finetune smoke。
5. 执行清理清单 + 更新 `.gitignore`。
6. 写 `head_tuning/README.md` + 顶层 README 指引。
