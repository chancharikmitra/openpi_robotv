# Attention Head-Tuning Pipeline

This directory implements a four-stage pipeline for **task-specific attention head selection and fine-tuning** of pi0/pi0.5 policies.  The key idea is that only a small subset of the 144 attention heads (18 layers × 8 heads) carry strong task-relevant information.  The pipeline selects those heads and fine-tunes their attention LoRA adapters — together with the main LLM's FFN LoRA adapters — while keeping the action expert, the vision encoder, and all non-LoRA base weights frozen, for far fewer trainable parameters than full-model fine-tuning.

```
Stage 1 – Extract   →   Stage 2 – Select   →   Stage 3 – Config   →   Stage 4 – Finetune
activations.h5           heads.json              TrainConfig             checkpoint/
```

> **Environment:** all commands assume the [uv](https://docs.astral.sh/uv/) environment is set up (see the [top-level README](../../../README.md#environment-setup)):
> ```bash
> git submodule update --init --recursive          # if not cloned with --recurse-submodules
> GIT_LFS_SKIP_SMUDGE=1 uv sync                     # GIT_LFS_SKIP_SMUDGE=1 pulls LeRobot
> GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
> ```
> `uv run python ...` (used below) runs inside this environment automatically.

---

## Stage 1 — Extract Activations

**Module:** `openpi.head_tuning.extract`

Loads the frozen pi0.5 policy for the chosen setup, runs keyframe-selected inference over a demonstration dataset, and writes per-frame, per-head last-token activations to an HDF5 file.

The output HDF5 stores tensors of shape **`(18 layers, 8 heads, 256 dim)`** per selected frame, together with the corresponding action labels.  These activations are consumed by Stage 2.

### DROID setup

Input: an HDF5 file of demonstrations **collected on our own custom DROID-style robot setup** — it follows the DROID data layout (RGB views + joint/gripper state + actions) but is **not** the public DROID release; it is our in-house teleop collection. The example `swap_green_red_cube_20.h5` is 20 such episodes.
Policy checkpoint loaded automatically: `gs://openpi-assets/checkpoints/pi05_droid/params`.

```bash
uv run python -m openpi.head_tuning.extract \
    --setup droid \
    --input-h5 path/to/swap_green_red_cube_20.h5 \
    --out-h5   activations_droid.h5 \
    --task-prompt "swap green and red cube" \
    --max-episodes 200
```

| Flag | Description |
|------|-------------|
| `--setup droid` | Select the DROID adapter. |
| `--input-h5 <path>` | **(Required)** Path to the input DROID HDF5 file. |
| `--out-h5 <path>` | Output HDF5 file (default: `activations.h5`). |
| `--task-prompt "<text>"` | **(Required for droid)** Language instruction; also used as the task-name key in the output HDF5. |
| `--max-episodes N` | Process at most N episodes (default: 200; 0 = all). |
| `--use-keyframe` / `--no-use-keyframe` | Apply the setup-specific keyframe selection rule (default: enabled). |

### LIBERO setup

Input: a LeRobot v2.0 dataset directory (produced by `scripts/extract_libero_task_subset.py` or similar).
Policy checkpoint loaded automatically: `gs://openpi-assets/checkpoints/pi05_libero/params`.

```bash
uv run python -m openpi.head_tuning.extract \
    --setup libero \
    --src      path/to/libero10_task0_20 \
    --out-h5   activations_libero.h5 \
    --max-episodes 200
```

| Flag | Description |
|------|-------------|
| `--setup libero` | Select the LIBERO adapter. |
| `--src <dir>` | **(Required)** Root directory of the LeRobot v2.0 dataset. |
| `--out-h5 <path>` | Output HDF5 file (default: `activations.h5`). |
| `--task-prompt "<text>"` | Optional override — when non-empty, every episode uses this string as its prompt.  When omitted (default), each episode's prompt is read from `meta/episodes.jsonl`. |
| `--max-episodes N` | Process at most N episodes (default: 200; 0 = all). |
| `--use-keyframe` / `--no-use-keyframe` | Apply the setup-specific keyframe selection rule (default: enabled). |

---

## Stage 2 — Select Heads

**Module:** `openpi.head_tuning.select`

Runs **Leave-One-Episode-Out KNN regression** (cosine distance by default) over the activations produced in Stage 1.  Each of the 144 candidate units (heads or layers) is evaluated by how well its activation feature predicts the action label for held-out episodes; the top-scoring units are written to a JSON file.

```bash
uv run python -m openpi.head_tuning.select \
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

The factory function `make_head_tuning_config` constructs a fully-specified `TrainConfig` that trains the selected attention heads' LoRA adapters **plus the main LLM's FFN LoRA adapters and the small flow-matching action head** (timestep MLP + action projections), while keeping the action expert, the vision encoder, and all non-LoRA base weights frozen.  See the *Optimizer and freeze strategy* section below for the full per-group breakdown.

```python
from openpi.head_tuning.configs import make_head_tuning_config

cfg = make_head_tuning_config(
    name="my_task_head_tuning",          # unique experiment identifier
    repo_id="hf_user/my_dataset_20",     # HuggingFace LeRobot dataset
    heads="heads.json",                  # path to Stage 2 output  OR  list of (layer, head) pairs
    setup="droid",                   # "droid" (default) or "libero"
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
| `setup` | `"droid"` | `"droid"`: uses `LeRobotDROIDDataConfig` + pi05_droid checkpoint.  `"libero"`: uses `LeRobotLiberoDataConfig` (with `extra_delta_transform=True`) + pi05_libero checkpoint. |
| `num_train_steps` | `3000` | Total gradient steps. |
| `peak_lr` | `2.5e-5` | Peak learning rate for cosine-decay schedule (decays to `peak_lr / 10` over training). |
| `pi05` | `True` | If `True`, build a pi0.5 model (`action_dim=32`, `action_horizon=16`) and load the pi0.5 checkpoint for the selected setup.  If `False`, use the pi0 checkpoint instead. |
| `freeze_kv` | `True` | Forwarded to `AdamWForHeadTuning`. Freeze the attention KV LoRA so only the selected heads' Q + O LoRA train. |
| `only_attention` | `False` | Forwarded to `AdamWForHeadTuning`. If `True`, restrict updates to attention LoRA only (FFN LoRA and all other non-attention weights masked to zero). |
| `freeze_mlp` | `False` | Forwarded to `AdamWForHeadTuning` (only effective when `only_attention=False`). If `True`, additionally freeze the main-LLM MLP/FFN LoRA. |

### Optimizer and freeze strategy

Internally, `make_head_tuning_config` wires up `AdamWForHeadTuning` with `freeze_kv=True, only_attention=False, freeze_mlp=False`, together with `get_freeze_filter_always_freeze_expert_and_siglip()`.  The net set of **trainable** parameters (≈ 19.6 M, enumerated over the real parameter tree) is:

| Group | Params | Share |
|-------|--------|-------|
| Selected `(layer, head)` attention **Q + O LoRA** (`q_einsum`, `attn_vec_einsum`), masked head-wise | ~1.5 M | 8 % |
| All main-LLM **FFN/MLP LoRA** (`mlp` gating + linear, every layer) | ~15.9 M | 81 % |
| Flow-matching **timestep MLP** (`time_mlp_in`, `time_mlp_out`) | ~2.1 M | 11 % |
| **Action projections** (`action_in_proj`, `action_out_proj`) | ~0.07 M | <1 % |

The selected heads are only ~8 % of the trainable count — the **FFN LoRA dominates**.  The last two groups (the flow-matching timestep MLP and the action projections) are top-level Pi0 modules outside the `llm`/`img` paths, so the freeze filter does not cover them and they train too (this is the standard flow-matching action head).

Everything else is **frozen**: the attention **KV LoRA** (`freeze_kv=True`), the attention LoRA of non-selected heads, all main-LLM **non-LoRA base weights**, the entire **action expert** (`.*llm.*_1.*`), and the entire **SigLIP** vision encoder (`.*img.*`).

The mask is built per-parameter by `_create_head_tuning_mask` (0 = frozen, 1 = trainable) and applied in `scripts/train.py` by multiplying each gradient with the mask before the AdamW step.  A standalone enumeration confirms the attention masking is **exact**: the trained `(layer, head)` set equals the selected set with zero leakage to non-selected heads.

### Pre-registered example configs

Two ready-to-use configs are registered in `src/openpi/training/config.py` for quick experimentation:

| Config name | Setup | Dataset |
|-------------|-----------|---------|
| `pi05_head_tuning_droid_example` | DROID | `<your-hf-username>/swap_green_red_cube_20` |
| `pi05_head_tuning_libero_example` | LIBERO | `<your-hf-username>/libero10_task0_20` |

To register your own config, call `make_head_tuning_config(...)` inside the `_CONFIGS` list in `src/openpi/training/config.py`.

---

## Stage 4 — Fine-Tune

**Script:** `scripts/train.py` (upstream openpi training script, unchanged except for the marked regions described below).

This follows the standard openpi finetuning flow — **two steps: compute norm stats, then train.**

### 4a. Compute normalization statistics (required, once per dataset)

`scripts/train.py` errors out if norm stats are missing.  Compute them for your config first:

```bash
uv run python scripts/compute_norm_stats.py pi05_head_tuning_droid_example
```

This reads the `repo_id` baked into the config (e.g. `<your-hf-username>/swap_green_red_cube_20`), streams the dataset, and writes `norm_stats.json` into the config's assets directory keyed by `repo_id`.  Re-run it whenever you change `repo_id`.

### 4b. Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/train.py pi05_head_tuning_droid_example \
    --exp-name my_experiment \
    --overwrite
```

Replace `pi05_head_tuning_droid_example` with any config registered in Stage 3.  `num_train_steps` and `batch_size` come from the config; override them on the CLI only if needed (see flags).  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use up to 90 % of GPU memory (default 75 %).

### Common flags

| Flag | Description |
|------|-------------|
| `--exp-name <name>` | Sub-directory name under `checkpoints/<config>/`. |
| `--num-train-steps N` | Override the number of gradient steps from the config. |
| `--batch-size B` | Override the per-step batch size from the config. |
| `--fsdp-devices N` | Shard the model across N GPUs (FSDP) to reduce per-GPU memory. |
| `--overwrite` | Allow overwriting an existing experiment directory. |
| `--resume` | Resume from the latest saved checkpoint. |

### What is set in the config (Stage 3), not on the CLI

- **`repo_id`** — the HuggingFace LeRobot dataset to fine-tune on; also what norm stats are computed for.
- **`heads`** — the selected `(layer, head)` pairs from Stage 2.
- **`freeze_kv` / `only_attention` / `freeze_mlp`** — the masking knobs forwarded to `AdamWForHeadTuning`.  Defaults `True / False / False` train the selected heads' attention LoRA + the main-LLM FFN LoRA + the flow-matching action head.  Set `only_attention=True` for attention-only, or `freeze_mlp=True` to keep the heads but drop the FFN LoRA.

### Notes

- **Batch size and FSDP:** `--batch-size` must be evenly divisible by the number of visible GPUs (FSDP shards the batch).  Set `CUDA_VISIBLE_DEVICES` to control which GPUs are used, e.g. `CUDA_VISIBLE_DEVICES=0,1`.
- **Checkpoint cache:** the base-model weights are downloaded from GCS to `~/.cache/openpi` by default.  Set `OPENPI_DATA_HOME=/path/to/cache` to redirect the cache.
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
| `src/openpi/models/pi0.py` | Added a pass-through path in `sample_actions` to collect and return the attention activation dict when head extraction is requested. |
| `src/openpi/training/optimizer.py` | Added the `AdamWForHeadTuning` class: a masked AdamW that computes a per-parameter binary mask from the `trainable_head_indices` list. The mask keeps the selected `(layer, head)` attention-LoRA slices trainable and, when `only_attention=False` (the example-config default), leaves the FFN LoRA trainable as well; gradients for everything else are zeroed before the Adam update. |
| `scripts/train.py` | Added an import and wiring call so that `AdamWForHeadTuning` is passed the model's parameter tree at initialization (needed to build the mask). |
| `src/openpi/models/pi0_config.py` | Added `get_freeze_filter_always_freeze_expert_and_siglip()` helper that returns a freeze predicate matching SigLIP and action-expert parameter paths. |

For the complete, authoritative list of every changed line and its rationale see **[`docs/UPSTREAM_CHANGES.md`](../../../docs/UPSTREAM_CHANGES.md)**.

> **What was intentionally NOT included in this release:**
> Inference-time `delta_heads` activation-steering (the mechanism that patched attention outputs at inference time) and the associated private serving entries have been removed and reverted to upstream.  This release covers only the training-side pipeline (extract → select → finetune).

---

## Quick-start Checklist

```bash
# 1. Extract activations (DROID example)
uv run python -m openpi.head_tuning.extract \
    --setup droid \
    --input-h5  path/to/my_task_20.h5 \
    --out-h5    activations.h5 \
    --task-prompt "pick up red cube"

# 2. Select the top-20 heads
uv run python -m openpi.head_tuning.select \
    --attn-h5 activations.h5 \
    --out      heads.json \
    --target   20

# 3. Register a config (edit src/openpi/training/config.py)
#    make_head_tuning_config(
#        name="my_task_head_tuning",
#        repo_id="hf_user/my_task_20",
#        heads="heads.json",
#        # freeze_kv / only_attention / freeze_mlp default to True / False / False
#    )

# 4a. Compute norm stats for the config's dataset (required)
uv run python scripts/compute_norm_stats.py my_task_head_tuning

# 4b. Fine-tune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/train.py my_task_head_tuning \
    --exp-name run_001 \
    --overwrite
```
