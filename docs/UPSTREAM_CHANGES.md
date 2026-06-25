# Fork Delta vs Upstream openpi

**Fork base:** `Physical-Intelligence/openpi` @ commit `5bff19b`

All additions made in this fork are marked with `# [head_tuning] BEGIN ... # [head_tuning] END` block markers (or inline `# [head_tuning]` for single-line annotations). To locate every fork-delta region:

```bash
grep -rn "\[head_tuning\]" src/ scripts/train.py
```

The changes implement a three-stage **attention-head selection and finetuning** pipeline on top of the pi0/pi0.5 base models:

1. **Extract** — run inference and collect per-head attention activations from specific layers.
2. **Select** — score and rank heads using KNN-based probing; identify the most task-relevant heads.
3. **Finetune** — train the selected heads' attention LoRA (query + output projections) together with the main LLM's FFN LoRA adapters and the flow-matching action head (timestep MLP + action projections), using a masked AdamW optimizer. The trainable set is ~19.6 M params, dominated (~81%) by the FFN LoRA. KV projections, non-selected heads, the action expert, and the vision encoder stay frozen.

---

## Modified Files

### `src/openpi/models/gemma.py`

**What changed:**

- **`Variant` type literal** (line ~55): Extended with ten new rank-ablation variants (`gemma_2b_lora_r4/r8/r16/r32/r64`, `gemma_300m_lora_r4/r8/r16/r32/r64`) to support LoRA rank sweep experiments.
- **`get_config()`** (line ~114, ~137): Added two new `if`-blocks to construct `Config` objects for the rank-ablation variants, using `rank = int(variant.split("_r")[-1])` to parse rank from the variant name.
- **`Attention.__call__()`** (line ~210): Added keyword arguments `return_attention_heads: bool` and `return_attention_probs: bool`. When enabled, the method additionally returns `attention_heads` (per-head encoded vectors, shape `[B, T, K*G, H]`) and/or `attention_probs` (last-query softmax probabilities, shape `[B, K*G, S]`). The return type becomes a 4-tuple instead of 2-tuple when either flag is set.
- **`Block.__call__()`** (line ~354): Added the same two flags; propagates them to `Attention` and unpacks the 4-tuple result; returns `(xs, (kv_cache, attention_heads, attention_probs))` when active.
- **`Module.setup()`** (line ~439): Changed `static_argnums=(5,)` to `static_argnums=(6, 7, 8)` in `nn.remat` to include `deterministic` and the two new flags; added two extra `nn.broadcast` entries in `nn.scan`'s `in_axes` to broadcast the flags across all layers.
- **`Module.__call__()`** (line ~471): Added `return_attention_heads` and `return_attention_probs` keyword arguments; when either is set, unpacks `(kv_cache, attention_heads, attention_probs)` from layers output and returns a 3-tuple `(embedded_norm, kv_cache, out_dict)` where `out_dict` contains the collected activations.

**Why / Stage:** Extract — these changes enable collecting per-layer attention-head activations during a standard inference forward pass without any computational overhead when flags are `False`.

**Markers:** `# [head_tuning] BEGIN` around `Variant` literal, `get_config` rank-ablation blocks, `Attention.__call__`, `Block.__call__`, `Module.setup`, and `Module.__call__`.

---

### `src/openpi/models/pi0.py`

**What changed:**

- **`embed_suffix()`** (line ~140): Added `include_action_tokens: bool = True` keyword argument. When `False`, the method skips embedding action tokens and returns only the state token (pi0) or an empty suffix (pi0.5). This allows a single extra forward pass to extract state-only activations without running the full diffusion suffix.
- **`sample_actions()`** (line ~228): Added four keyword arguments: `return_attention_heads`, `return_attention_probs`, `return_state_heads`, `return_state_and_first_action_heads`. When any flag is set, the method:
  - Passes `return_attention_heads`/`return_attention_probs` to the prefix prefill call and stores results in `attention_outputs["llm_activations"]` / `"llm_attn_probs_prefill"`.
  - If `return_state_heads` or `return_state_and_first_action_heads` is set, runs an additional suffix forward pass (with or without action tokens respectively) and stores `"llm_state_activations"`, `"llm_state_first_action_activations"`, or `"llm_first_action_activations"`.
  - Returns `(x_0, attention_outputs)` tuple instead of just `x_0`.

**Why / Stage:** Extract — makes `sample_actions` the single entry point for collecting all activation types needed by the KNN head-selection algorithm.

**Markers:** `# [head_tuning] BEGIN` around `embed_suffix` and `sample_actions`.

---

### `src/openpi/models/pi0_config.py`

**What changed:**

- **`get_freeze_filter_always_freeze_expert_and_siglip()`** (line ~110): New method on `Pi0Config`. Returns an `nnx.filterlib.Filter` that always freezes the Action Expert branch (`.*llm.*_1.*`) and SigLIP branch (`.*img.*`), while optionally also freezing the non-LoRA parts of the main LLM branch when `"lora"` is in `paligemma_variant`. This is distinct from the upstream `get_freeze_filter()` which only freezes the main LLM non-LoRA weights.

**Why / Stage:** Finetune — required freeze configuration for the head-tuning stage where we want the Action Expert and vision encoder fully frozen while allowing the main LLM's selected-head LoRA weights to be updated.

**Markers:** `# [head_tuning] BEGIN` around the entire new method.

---

### `src/openpi/policies/policy.py`

**What changed:**

- **`Policy.__init__()` — JIT setup** (line ~64): Changed `nnx_utils.module_jit(model.sample_actions)` to pass `static_argnames=("return_attention_heads", "return_attention_probs", "return_state_heads", "return_state_and_first_action_heads")`. This is required to prevent `TracerBoolConversionError` when these flags are used under JIT compilation.
- **`Policy.infer()` signature** (line ~78): Extended the method signature with four `return_*` keyword arguments (all `bool`, default `False`). These are forwarded as `sample_kwargs` to `sample_actions`.
- **`Policy.infer()` body**: Added logic to unpack the `(actions, attention_outputs)` or `(actions, attention_outputs, decode_step)` tuple returned by `sample_actions` when any flag is active; attaches `attention_outputs` and `decode_step` to the output dict.
- **Removed**: Orphan commented-out debug line `#result = self._sample_actions(sample_rng_or_pytorch_device, observation)` (was around line 117 before this commit).

**Why / Stage:** Extract — `Policy.infer()` is the external API surface; extending it allows callers to request activations without modifying the inference loop.

**Markers:** `# [head_tuning] BEGIN` around the JIT static-argnames block, the `infer()` signature flags, the kwarg forwarding block, the result-unpacking block, and the attention-output attachment block.

---

### `src/openpi/training/optimizer.py`

**What changed:**

- **New imports** (line ~6): Added `import numpy as np` and `from jax import tree_util` — needed by `_create_head_tuning_mask`.
- **`AdamWForHeadTuning` class** (line ~108): New `@dataclasses.dataclass(frozen=True)` implementing `OptimizerConfig`. Fields: standard AdamW hyperparameters (`b1`, `b2`, `eps`, `weight_decay`, `clip_gradient_norm`) plus `trainable_head_indices: list[tuple[int, int]]`, `freeze_kv: bool`, `only_attention: bool`, `freeze_mlp: bool`. The `create()` method is identical to `AdamW.create()`; the actual masking is applied externally by `_create_masked_optimizer_for_head_tuning` in `scripts/train.py`.
- **`_create_head_tuning_mask()` function** (line ~143): New function that takes `params` (an `nnx.State` pytree), `trainable_heads: list[tuple[int, int]]`, and optional `freeze_kv`/`only_attention`/`freeze_mlp` flags. Returns a pytree of `int8` arrays (0=frozen, 1=trainable) with the same structure as `params.to_pure_dict()`. Uses `jax.tree_util.tree_map_with_path` to inspect each parameter's path string and determine: (a) whether it is an attention weight at all, (b) which axis corresponds to layer/head, (c) whether to mask by individual head index or by layer-level KV sharing (MQA case).

**Why / Stage:** Finetune — `AdamWForHeadTuning` is the optimizer config for the head-finetuning stage; `_create_head_tuning_mask` constructs the binary mask that keeps the selected heads' attention-LoRA slices trainable and — because the example configs pass `only_attention=False` — leaves the FFN LoRA trainable too, zeroing gradients for everything else (KV, non-selected heads, base weights).

**Markers:** `# [head_tuning] BEGIN` around the import block, the `AdamWForHeadTuning` class, and `_create_head_tuning_mask`.

---

### `scripts/train.py`

**What changed:**

- **`_create_masked_optimizer_for_head_tuning()` function** (line ~50): New function that (1) creates a base `AdamW` optimizer, (2) calls `_create_head_tuning_mask` to build a binary mask dict, (3) defines a `masked_update_fn` closure that multiplies gradients element-wise by the mask before passing them to the base optimizer, and (4) returns `optax.GradientTransformation(base_optimizer.init, masked_update_fn)`. This wraps the standard optax API so the rest of the training loop needs no awareness of masking.
- **`init_train_state()` — optimizer selection** (line ~129): Added `isinstance(config.optimizer, _optimizer.AdamWForHeadTuning)` branch; when true, calls `jax.eval_shape` to get the parameter structure without materializing weights, then calls `_create_masked_optimizer_for_head_tuning`. Optimizer state is initialized from `params.filter(...).to_pure_dict()` (pure dict, not `nnx.State`) when head-tuning is active.
- **`train_step()` — masked update** (line ~214): Added an `isinstance(config.optimizer, _optimizer.AdamWForHeadTuning)` branch that converts `grads` and `params_trainable` to pure dicts before calling `state.tx.update`, applies updates with `optax.apply_updates`, then calls `nnx.update(model, nnx.State(new_params_trainable_dict))` to write results back into the NNX model. The standard branch is unchanged.

**Why / Stage:** Finetune — wires `AdamWForHeadTuning` into the standard openpi training loop with minimal diff to the non-head-tuning paths.

**Markers:** `# [head_tuning] BEGIN` around `_create_masked_optimizer_for_head_tuning`, the conditional optimizer creation in `init_train_state`, the `opt_state` init expression, and the masked-update branch in `train_step`.

---

## Reverted Files (no delta vs upstream)

- **`scripts/serve_policy.py`**: Was extended with ~350 lines of lab-private `EnvMode`/`Checkpoint` path entries (`/darrell_robotics/...`). Reverted to upstream `5bff19b` to remove private filesystem paths. Not method-essential.
- **`scripts/compute_norm_stats.py`**: Contained a 17-line monkey-patch for a `datasets` library compatibility issue (`List` → `Sequence` feature type). Reverted to upstream `5bff19b`; the fix is a dataset-version workaround, not part of the head-tuning method, and should be applied at the environment level if needed.
