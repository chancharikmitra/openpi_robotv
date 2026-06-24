# Attention Head-Tuning Pipeline

This directory implements a four-stage pipeline for **task-specific attention head selection and fine-tuning** of pi0/pi0.5 policies.  The key idea is that only a small subset of the 144 attention heads (18 layers × 8 heads) carry strong task-relevant information; training exclusively those heads achieves competitive adaptation with far fewer trainable parameters than full LoRA fine-tuning.

```
Stage 1 – Extract   →   Stage 2 – Select   →   Stage 3 – Config   →   Stage 4 – Finetune
activations.h5           heads.json              TrainConfig             checkpoint/
```

> **Environment:** all commands must run inside the `openpi` conda environment.
> ```bash
> conda activate openpi
> # or, non-interactively:
> conda run -n openpi python ...
> ```

---

## Stage 1 — Extract Activations

**Module:** `openpi.head_tuning.extract`

Loads the frozen pi0.5 policy for the chosen benchmark, runs keyframe-selected inference over a demonstration dataset, and writes per-frame, per-head last-token activations to an HDF5 file.

The output HDF5 stores tensors of shape **`(18 layers, 8 heads, 256 dim)`** per selected frame, together with the corresponding action labels.  These activations are consumed by Stage 2.

### DROID benchmark

Input: a DROID HDF5 file (converted to LeRobot format via `src/openpi/convert2lerobot.py`).
Policy checkpoint loaded automatically: `gs://openpi-assets/checkpoints/pi05_droid/params`.

```bash
conda run -n openpi python -m openpi.head_tuning.extract \
    --benchmark droid \
    --input-h5 temp_data/swap_green_red_cube_20.h5 \
    --out-h5   activations_droid.h5 \
    --task-prompt "swap green and red cube" \
    --max-episodes 200
```

| Flag | Description |
|------|-------------|
| `--benchmark droid` | Select the DROID adapter. |
| `--input-h5 <path>` | **(Required)** Path to the input DROID HDF5 file. |
| `--out-h5 <path>` | Output HDF5 file (default: `activations.h5`). |
| `--task-prompt "<text>"` | **(Required for droid)** Language instruction; also used as the task-name key in the output HDF5. |
| `--max-episodes N` | Process at most N episodes (default: 200; 0 = all). |
| `--use-keyframe` / `--no-use-keyframe` | Apply the benchmark-specific keyframe selection rule (default: enabled). |

### LIBERO benchmark

Input: a LeRobot v2.0 dataset directory (produced by `scripts/extract_libero_task_subset.py` or similar).
Policy checkpoint loaded automatically: `gs://openpi-assets/checkpoints/pi05_libero/params`.

```bash
conda run -n openpi python -m openpi.head_tuning.extract \
    --benchmark libero \
    --src      temp_data/libero10_task0_20 \
    --out-h5   activations_libero.h5 \
    --max-episodes 200
```

| Flag | Description |
|------|-------------|
| `--benchmark libero` | Select the LIBERO adapter. |
| `--src <dir>` | **(Required)** Root directory of the LeRobot v2.0 dataset. |
| `--out-h5 <path>` | Output HDF5 file (default: `activations.h5`). |
| `--task-prompt "<text>"` | Optional override — when non-empty, every episode uses this string as its prompt.  When omitted (default), each episode's prompt is read from `meta/episodes.jsonl`. |
| `--max-episodes N` | Process at most N episodes (default: 200; 0 = all). |
| `--use-keyframe` / `--no-use-keyframe` | Apply the benchmark-specific keyframe selection rule (default: enabled). |

---

## Stage 2 — Select Heads

**Module:** `openpi.head_tuning.select`

Runs **Leave-One-Episode-Out KNN regression** (cosine distance by default) over the activations produced in Stage 1.  Each of the 144 candidate units (heads or layers) is evaluated by how well its activation feature predicts the action label for held-out episodes; the top-scoring units are written to a JSON file.

```bash
conda run -n openpi python -m openpi.head_tuning.select \
    --attn-h5 activations_droid.h5 \
    --out      heads.json \
    --unit-mode head \
    --target    20 \
    --metric    cosine \
    --selection-mode topk
```

| Flag | Description |
|------|-------------|
| `--attn-h5 <path>` | **(Required)** Path to the activation HDF5 from Stage 1. |
| `--out <path>` | Output JSON file (default: `heads.json`). |
| `--unit-mode head\|layer` | Evaluate individual heads (144 units) or whole layers (18 units). Default: `head`. |
| `--target N` | Number of heads (or layers) to select. Default: 20. |
| `--metric cosine\|euclidean` | KNN distance metric. Default: `cosine`. |
| `--selection-mode topk\|best_add` | `topk`: rank all units by CV-MSE and take the best N; `best_add`: greedy forward selection. Default: `topk`. |
| `--k-grid N [N ...]` | Candidate k values for KNN cross-validation sweep. Default: `10 20 30 40`. |
| `--temp-excl-w W` | Temporal exclusion window (±W frames) for leave-one-frame-out. Default: 30. |

### Output format — `heads.json`

```json
{
    "trainable_head_indices": [[9, 7], [9, 4], [8, 4], ...],
    "best_k": 20,
    "cv_mse": 0.031,
    ...
}
```

The `trainable_head_indices` list contains `[layer, head]` pairs (0-indexed) for the selected units and is consumed directly by Stage 3.

---

## Stage 3 — Build a TrainConfig

**Module / file:** `src/openpi/head_tuning/configs.py`

The factory function `make_head_tuning_config` constructs a fully-specified `TrainConfig` that trains **only the selected attention heads** while keeping all other parameters frozen.

```python
from openpi.head_tuning.configs import make_head_tuning_config

cfg = make_head_tuning_config(
    name="my_task_head_tuning",          # unique experiment identifier
    repo_id="hf_user/my_dataset_20",     # HuggingFace LeRobot dataset
    heads="heads.json",                  # path to Stage 2 output  OR  list of (layer, head) pairs
    benchmark="droid",                   # "droid" (default) or "libero"
    num_train_steps=3000,                # gradient steps (default: 3000)
    peak_lr=2.5e-5,                      # peak learning rate (default: 2.5e-5)
    pi05=True,                           # True → pi0.5 checkpoint; False → pi0 checkpoint
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | — | Unique config name / experiment identifier. |
| `repo_id` | — | HuggingFace `username/dataset` for the LeRobot fine-tuning dataset. |
| `heads` | — | Either a list of `(layer, head)` integer pairs **or** a path (`str`/`Path`) to a `heads.json` file from Stage 2. |
| `benchmark` | `"droid"` | `"droid"`: uses `LeRobotDROIDDataConfig` + pi05_droid checkpoint.  `"libero"`: uses `LeRobotLiberoDataConfig` (with `extra_delta_transform=True`) + pi05_libero checkpoint. |
| `num_train_steps` | `3000` | Total gradient steps. |
| `peak_lr` | `2.5e-5` | Peak learning rate for cosine-decay schedule (decays to `peak_lr / 10` over training). |
| `pi05` | `True` | If `True`, build a pi0.5 model (`action_dim=32`, `action_horizon=16`) and load the pi0.5 checkpoint for the selected benchmark.  If `False`, use the pi0 checkpoint instead. |

### Optimizer and freeze strategy

Internally, `make_head_tuning_config` wires up:

- **`AdamWForHeadTuning`** — a masked AdamW optimizer that applies gradient updates only to the query and output projections of the selected `(layer, head)` pairs.  KV projections remain frozen (`freeze_kv=True`).
- **`get_freeze_filter_always_freeze_expert_and_siglip()`** — a freeze filter that hard-freezes the action-expert sub-model and all SigLIP vision-encoder weights, regardless of which heads are selected.

As a result, a 20-head run trains roughly **~20 M parameters** out of the full ~3 B parameter model.

### Pre-registered example configs

Two ready-to-use configs are registered in `src/openpi/training/config.py` for quick experimentation:

| Config name | Benchmark | Dataset |
|-------------|-----------|---------|
| `pi05_head_tuning_droid_example` | DROID | `yusenluo9z/swap_green_red_cube_20` |
| `pi05_head_tuning_libero_example` | LIBERO | `yusenluo9z/libero10_task0_20` |

To register your own config, call `make_head_tuning_config(...)` inside the `_CONFIGS` list in `src/openpi/training/config.py`.

---

## Stage 4 — Fine-Tune

**Script:** `scripts/train.py` (upstream openpi training script, unchanged except for the marked regions described below)

```bash
conda run -n openpi python scripts/train.py pi05_head_tuning_droid_example \
    --exp-name  my_experiment \
    --num-train-steps 3000 \
    --batch-size 32 \
    --overwrite
```

Replace `pi05_head_tuning_droid_example` with any config name registered in Step 3.

### Common flags

| Flag | Description |
|------|-------------|
| `--exp-name <name>` | Sub-directory name under `checkpoints/<config>/`. |
| `--num-train-steps N` | Override the number of gradient steps from the config. |
| `--batch-size B` | Per-step batch size. |
| `--overwrite` | Allow overwriting an existing experiment directory. |
| `--resume` | Resume from the latest saved checkpoint. |

### Notes

- **Batch size and FSDP:** `--batch-size` must be evenly divisible by the number of visible GPUs (FSDP shards the batch).  Set `CUDA_VISIBLE_DEVICES` to control which GPUs are used, e.g. `CUDA_VISIBLE_DEVICES=0,1`.
- **Checkpoint cache:** the base-model weights are downloaded from GCS to `~/.cache/openpi` by default.  Set `OPENPI_DATA_HOME=/path/to/cache` to redirect the cache.
- **Norm stats:** run `conda run -n openpi python scripts/compute_norm_stats.py <config_name>` before training if norm stats are not yet cached for your dataset.
- **Logging:** training metrics are logged every `log_interval=100` steps; checkpoints are saved every `save_interval=1000` steps.

---

## openpi Core Modifications

The head-tuning pipeline requires a small number of **in-place changes** to the upstream openpi core.  Every modified region is annotated with `[head_tuning]` markers (block form `# [head_tuning] BEGIN ... # [head_tuning] END` or single-line `# [head_tuning]`) so they can be located with a single grep:

```bash
grep -rn "\[head_tuning\]" src/ scripts/train.py
```

The key modifications are:

| File | What was changed |
|------|-----------------|
| `src/openpi/models/gemma.py` | Added `return_attention_heads` / `return_attention_probs` kwargs to `Attention.__call__` and `Block.__call__`; updated `nn.remat` static argnums and `nn.scan` broadcast axes in `Module.setup` to propagate the new flags across all layers; `Module.__call__` returns an optional `out_dict` containing stacked per-layer activations. |
| `src/openpi/models/pi0.py` | Added a pass-through path in `_sample_actions` to collect and return the attention activation dict when head extraction is requested. |
| `src/openpi/training/optimizer.py` | Added the `AdamWForHeadTuning` class: a masked AdamW that computes a per-parameter binary mask from the `trainable_head_indices` list and zeros out gradients for all other parameters before the Adam update. |
| `scripts/train.py` | Added an import and wiring call so that `AdamWForHeadTuning` is passed the model's parameter tree at initialization (needed to build the mask). |
| `src/openpi/models/pi0_config.py` | Added `get_freeze_filter_always_freeze_expert_and_siglip()` helper that returns a freeze predicate matching SigLIP and action-expert parameter paths. |

For the complete, authoritative list of every changed line and its rationale see **[`docs/UPSTREAM_CHANGES.md`](../../../docs/UPSTREAM_CHANGES.md)**.

> **What was intentionally NOT included in this release:**
> Inference-time `delta_heads` activation-steering (the mechanism that patched attention outputs at inference time) and the associated private serving entries have been removed and reverted to upstream.  This release covers only the training-side pipeline (extract → select → finetune).

---

## Quick-start Checklist

```bash
# 1. Extract activations (DROID example)
conda run -n openpi python -m openpi.head_tuning.extract \
    --benchmark droid \
    --input-h5  temp_data/my_task_20.h5 \
    --out-h5    activations.h5 \
    --task-prompt "pick up red cube"

# 2. Select the top-20 heads
conda run -n openpi python -m openpi.head_tuning.select \
    --attn-h5 activations.h5 \
    --out      heads.json \
    --target   20

# 3. Register a config (edit src/openpi/training/config.py)
#    make_head_tuning_config(
#        name="my_task_head_tuning",
#        repo_id="hf_user/my_task_20",
#        heads="heads.json",
#    )

# 4. Fine-tune
conda run -n openpi python scripts/train.py my_task_head_tuning \
    --exp-name run_001 \
    --overwrite
```
