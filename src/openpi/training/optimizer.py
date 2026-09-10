import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax
# [head_tuning] BEGIN: additional imports for head-tuning mask construction
import numpy as np
from jax import tree_util
# [head_tuning] END

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


# [head_tuning] BEGIN: AdamWForHeadTuning optimizer config (head-selective finetuning stage)
@dataclasses.dataclass(frozen=True)
class AdamWForHeadTuning(OptimizerConfig):
    """AdamW optimizer that only trains specific attention heads."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
    # List of (layer_index, head_index) tuples to train
    trainable_head_indices: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    # If True, completely freeze all KV (kv_einsum) parameters even in LoRA
    freeze_kv: bool = False
    # If True, strictly restrict updates to attention weights only (non-attention weights fully masked to 0)
    only_attention: bool = False
    # If True, additionally freeze only the MLP (FFN) submodules under main LLM
    # while allowing other non-attention modules (proj, norm, etc.) to train.
    # Effective only when only_attention is False.
    freeze_mlp: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        # This create function is identical to AdamW, as the masking is handled separately.
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)
# [head_tuning] END


# [head_tuning] BEGIN: _create_head_tuning_mask — builds per-param binary mask for head-selective update
def _create_head_tuning_mask(
    params: at.Params,
    trainable_heads: list[tuple[int, int]],
    *,
    freeze_kv: bool = False,
    only_attention: bool = True,
    freeze_mlp: bool = False,
) -> at.Params:
    """Creates a mask to freeze all but specific attention heads based on observed parameter paths."""
    trainable_heads_map = {}
    for layer, head in trainable_heads:
        if layer not in trainable_heads_map:
            trainable_heads_map[layer] = []
        trainable_heads_map[layer].append(head)

    def _get_mask(path: tuple[str, ...], leaf: at.Array) -> at.Array:
        # Fix: correctly handle path format
        # path is a tuple; each element may be a string or a list-form string
        actual_path_parts = []
        for part in path:
            part_str = str(part)
            # If in the format "['xxx']", extract xxx
            if part_str.startswith("['") and part_str.endswith("']"):
                actual_path_parts.append(part_str[2:-2])
            else:
                actual_path_parts.append(part_str)
        
        actual_path = "/".join(actual_path_parts)
        
        # Check for attention weights (both regular and LoRA)
        is_attn_weight = "PaliGemma/llm/layers/attn" in actual_path
        is_regular_weight = actual_path.endswith("/w")
        is_lora_weight = actual_path.endswith("/lora_a") or actual_path.endswith("/lora_b")
        
        # Only process attention weights (regular or LoRA)
        if not (is_attn_weight and (is_regular_weight or is_lora_weight)):
            # Non-attention weights:
            if only_attention:
                return jnp.zeros_like(leaf, dtype=jnp.int8) if hasattr(leaf, 'shape') else 0
            if freeze_mlp:
                is_mlp_node = (
                    "PaliGemma/llm/layers/mlp" in actual_path or
                    "PaliGemma/llm/layers/mlp_1" in actual_path
                )
                return (jnp.zeros_like(leaf, dtype=jnp.int8) if is_mlp_node else
                        (jnp.ones_like(leaf, dtype=jnp.int8) if hasattr(leaf, 'shape') else 1))
            return jnp.ones_like(leaf, dtype=jnp.int8) if hasattr(leaf, 'shape') else 1

        layer_dim = leaf.shape[0]
        
        # Determine the head axis based on parameter type and name
        head_axis = -1
        param_name = actual_path_parts[-2] if len(actual_path_parts) >= 2 else ""  # e.g., 'q_einsum', 'kv_einsum', etc.
        
        if is_lora_weight:
            # For LoRA weights, head dimension is usually axis 1, but KV LoRA needs special handling
            if param_name in ["kv_einsum"]:
                if freeze_kv:
                    # Completely freeze KV LoRA
                    return jnp.zeros_like(leaf, dtype=jnp.int8)
                # KV LoRA shapes (Gemma): (layers, 2, num_kv_heads, in_dim, rank) or (layers, 2, num_kv_heads, rank, out_dim)
                if leaf.ndim >= 3 and leaf.shape[2] > 1:
                    head_axis = 2  # multi-KV heads
                else:
                    # MQA: share a single KV across Q heads; train KV if any Q head in layer is selected
                    head_axis = -2  # sentinel for special layer-wise handling
            else:
                # For Q and output attn LoRA: (layers, heads, ...)
                head_axis = 1
        else:
            # For regular weights, determine based on parameter name and actual shape
            if param_name in ["q_einsum", "attn_vec_einsum"]:
                # Query and output projections: (layers, heads, input_dim, output_dim)
                # The head dimension is at axis 1
                head_axis = 1
            elif param_name in ["kv_einsum"]:
                if freeze_kv:
                    # Completely freeze KV regular weights
                    return jnp.zeros_like(leaf, dtype=jnp.int8)
                # For Gemma, KV projection: (layers, 2, num_kv_heads, input_dim, output_dim)
                # Gemma uses multi-query attention (num_kv_heads=1), so the practical shape is (layers, 2, 1, ...)
                # In this case, we need special handling
                if leaf.ndim >= 3 and leaf.shape[2] > 1:
                    head_axis = 2  # If the number of KV heads > 1
                else:
                    # For multi-query attention (num_kv_heads=1), all Q heads share the same KV
                    # If any head in the layer is trained, the layer's KV should also be trained
                    head_axis = -2  # Use a sentinel value to indicate special handling
            elif param_name in ["qkv_einsum"]:
                # Merged QKV projection: (layers, 3, heads, ...)
                head_axis = 2

        # Safety check
        if head_axis == -1:
            # If the head axis cannot be determined, return an all-zero mask (fully frozen)
            return jnp.zeros_like(leaf, dtype=jnp.int8)

        final_mask = jnp.zeros_like(leaf, dtype=jnp.int8)

        # Special handling for MQA KV weights (regular or LoRA)
        if param_name == "kv_einsum" and head_axis == -2:
            # For multi-query attention, if any head in a layer is trained, train the layer's KV
            for layer_idx, heads_to_train in trainable_heads_map.items():
                if layer_idx >= layer_dim:
                    continue
                if heads_to_train:  # This layer has at least one head to train
                    # Train KV weights for the entire layer
                    slicer = [slice(None)] * leaf.ndim
                    slicer[0] = layer_idx  # Layer dimension
                    final_mask = final_mask.at[tuple(slicer)].set(1)
        else:
            # Regular head-wise masking
            if head_axis >= leaf.ndim:
                return jnp.zeros_like(leaf, dtype=jnp.int8)
                
            head_dim_size = leaf.shape[head_axis]
            
            for layer_idx, heads_to_train in trainable_heads_map.items():
                if layer_idx >= layer_dim:
                    continue
                
                for head_idx in heads_to_train:
                    if head_idx < head_dim_size:
                        # Create a multidimensional slice to set the mask value
                        slicer = [slice(None)] * leaf.ndim
                        slicer[0] = layer_idx      # Layer dimension is always axis 0
                        slicer[head_axis] = head_idx  # Head dimension varies by parameter type
                        final_mask = final_mask.at[tuple(slicer)].set(1)
        
        return final_mask

    return tree_util.tree_map_with_path(_get_mask, params.to_pure_dict())
# [head_tuning] END


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    """Creates an optimizer from the config."""
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
