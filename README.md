# Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations @ ECCV 2026

Official code for **Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations** (ECCV 2026).

We adapt π₀ / π₀.₅ vision-language-action policies to a new task from only a handful of demonstrations by **mechanistically targeting the attention heads that matter**: out of the 144 attention heads (18 layers × 8 heads), only a small subset carries strong task-relevant signal. We (1) identify that subset, then (2) fine-tune those heads' attention LoRA together with the model's FFN LoRA and the flow-matching action head, keeping everything else frozen.

Head relevance is scored with **Leave-One-Episode-Out KNN regression over frozen-model activations** — selection needs no gradient training, so it works directly in the few-shot regime.

## Pipeline

```
1. Extract activations  →  2. Select heads  →  3. Build TrainConfig  →  4. Fine-tune
   (activations.h5)         (heads.json)        (head-tuning config)     (checkpoint/)
```

1. **Extract** — run the frozen policy over the few-shot demos and collect per-head attention activations.
2. **Select** — LOEO KNN regression ranks the 144 heads by task relevance; keep the top-N.
3. **Config** — build a head-tuning `TrainConfig` that trains the selected heads' attention LoRA (+ FFN LoRA + flow-matching action head); KV, the action expert, the vision encoder, and all non-LoRA base weights stay frozen.
4. **Fine-tune** — a masked AdamW optimizer applies updates only to the selected slices.

**→ Full walkthrough, CLI reference, and per-stage flags: [`src/openpi/head_tuning/README.md`](src/openpi/head_tuning/README.md)**

## Quickstart

### Environment setup

Requires an NVIDIA GPU and Python 3.11 (tested on Ubuntu 22.04). We use [uv](https://docs.astral.sh/uv/) to manage dependencies:

```bash
# if you did not clone with --recurse-submodules:
git submodule update --init --recursive

# GIT_LFS_SKIP_SMUDGE=1 is required to pull LeRobot as a dependency
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`uv run` (used below) automatically runs inside this environment — no manual activation needed. For GPU memory requirements, PyTorch support, Docker, and troubleshooting, see [Built on openpi](#built-on-openpi).

### Pipeline

```bash
# 1. extract per-head activations from your few-shot demos
uv run python -m openpi.head_tuning.extract \
    --setup droid --input-h5 path/to/demos.h5 \
    --out-h5 activations.h5 --task-prompt "<your task>"

# 2. select the top-20 task-relevant heads
uv run python -m openpi.head_tuning.select \
    --attn-h5 activations.h5 --out heads.json --target 20

# 3. register a config in src/openpi/training/config.py via make_head_tuning_config(...)

# 4. compute norm stats, then fine-tune
uv run python scripts/compute_norm_stats.py <config_name>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/train.py <config_name> --exp-name run_001 --overwrite
```

## Built on openpi

This repository is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence). The base π₀ / π₀.₅ models, the training/inference framework, installation, and tooling are unchanged from upstream — our method is added as the `openpi.head_tuning` package plus a small, clearly marked set of edits to the openpi core.

- **Installation, model checkpoints, inference, PyTorch support, troubleshooting** → [`docs/openpi_upstream.md`](docs/openpi_upstream.md) (the upstream openpi README, preserved verbatim).
- **Every change this fork makes to the openpi core** (each annotated with a `[head_tuning]` marker) → [`docs/UPSTREAM_CHANGES.md`](docs/UPSTREAM_CHANGES.md).

To locate the fork edits in the source:

```bash
grep -rn "\[head_tuning\]" src/ scripts/train.py
```

## Citation

```bibtex
@inproceedings{
    mitra2026mechanistic,
    title={Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations},
    author={Chancharik Mitra and Yusen Luo and Raj Saravanan and Dantong Niu and Anirudh Pai and Jesse Thomason and Trevor Darrell and Abrar Anwar and Deva Ramanan and Roei Herzig},
    booktitle={19th European Conference on Computer Vision (ECCV)},
    year={2026}
}
```
