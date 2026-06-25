# Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations

> **ECCV 2026**

Official code for **Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations**.

We adapt π₀ / π₀.₅ vision-language-action policies to a new task from only a handful of demonstrations by **mechanistically targeting the attention heads that matter**: out of the 144 attention heads (18 layers × 8 heads), only a small subset carries strong task-relevant signal. We (1) identify that subset, then (2) fine-tune those heads' attention LoRA together with the model's FFN LoRA and the flow-matching action head, keeping everything else frozen.

Head relevance is scored with **Leave-One-Episode-Out KNN regression over frozen-model activations** — selection needs no gradient training, so it works directly in the few-shot regime.

## Pipeline

```
① Extract activations  →  ② Select heads  →  ③ Build TrainConfig  →  ④ Fine-tune
   activations.h5            heads.json          (head-tuning)          checkpoint/
```

- **① Extract** — run the frozen policy over the few-shot demos and collect per-head attention activations.
- **② Select** — LOEO KNN regression ranks the 144 heads by task relevance; keep the top-N.
- **③ Config** — build a head-tuning `TrainConfig` that trains the selected heads' attention LoRA (+ FFN LoRA + flow-matching action head); KV, the action expert, the vision encoder, and all non-LoRA base weights stay frozen.
- **④ Fine-tune** — a masked AdamW optimizer applies updates only to the selected slices.

**→ Full walkthrough, CLI reference, and per-stage flags: [`src/openpi/head_tuning/README.md`](src/openpi/head_tuning/README.md)**

## Quickstart

Install the `openpi` environment first (see [Built on openpi](#built-on-openpi)), then:

```bash
# ① extract per-head activations from your few-shot demos
conda run -n openpi python -m openpi.head_tuning.extract \
    --setup droid --input-h5 path/to/demos.h5 \
    --out-h5 activations.h5 --task-prompt "<your task>"

# ② select the top-20 task-relevant heads
conda run -n openpi python -m openpi.head_tuning.select \
    --attn-h5 activations.h5 --out heads.json --target 20

# ③ register a config in src/openpi/training/config.py via make_head_tuning_config(...)

# ④ compute norm stats, then fine-tune
conda run -n openpi python scripts/compute_norm_stats.py <config_name>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n openpi python scripts/train.py <config_name> --exp-name run_001 --overwrite
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
@inproceedings{mechanistic_finetuning_vla_2026,
  title     = {Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations},
  author    = {<authors>},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
