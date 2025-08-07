import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax
import numpy as np
from jax import tree_util

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


def _create_head_tuning_mask(params: at.Params, trainable_heads: list[tuple[int, int]]) -> at.Params:
    """Creates a mask to freeze all but specific attention heads based on observed parameter paths."""
    trainable_heads_map = {}
    for layer, head in trainable_heads:
        if layer not in trainable_heads_map:
            trainable_heads_map[layer] = []
        trainable_heads_map[layer].append(head)

    def _get_mask(path: tuple[str, ...], leaf: at.Array) -> at.Array:
        path_str = "/".join(map(str, path))
        
        # Check for regular attention weights OR LoRA attention weights
        is_attn_weight = "PaliGemma/llm/layers/attn" in path_str
        is_lora = "lora_" in path[-1]
        
        # We only care about attention weights (vanilla or LoRA)
        if not (is_attn_weight and (path_str.endswith("/w") or is_lora)):
            return jnp.ones_like(leaf, dtype=jnp.int8) if hasattr(leaf, 'shape') else 1

        layer_dim = leaf.shape[0]
        mask = jnp.zeros_like(leaf, dtype=jnp.int8)

        # --- Determine the axis of the head dimension ---
        # This logic is based on inspecting the parameter shapes from debug_print_params.py
        head_axis = -1
        # For LoRA, the head dimension is consistently at axis 1
        if is_lora:
            head_axis = 1
        else: # For original weights
            param_name = path[-2]
            if param_name in ["q_einsum", "attn_vec_einsum"]:
                head_axis = 1
            elif param_name in ["kv_einsum", "qkv_einsum"]:
                # For kv_einsum, shape is (2, num_kv_heads, ...), head is at axis 1
                # For qkv_einsum, shape is (3, num_heads, ...), head is at axis 1
                 head_axis = 1

        # If we couldn't determine the head axis, something is wrong. Freeze it to be safe.
        if head_axis == -1 or leaf.ndim <= head_axis:
             logging.warning(f"Could not determine head axis for {path_str} with shape {leaf.shape}. Freezing.")
             return mask

        head_dim_size = leaf.shape[head_axis]
        final_mask = jnp.zeros_like(leaf, dtype=jnp.int8)

        for layer_idx, heads_to_train in trainable_heads_map.items():
            if layer_idx >= layer_dim:
                continue
            for head_idx in heads_to_train:
                if head_idx >= head_dim_size:
                    continue
                
                # Create a multidimensional slice to set the mask value
                slicer = [slice(None)] * leaf.ndim
                slicer[0] = layer_idx      # Layer dimension is always axis 0
                slicer[head_axis] = head_idx # Head dimension varies
                final_mask = final_mask.at[tuple(slicer)].set(1)
                        
        return final_mask

    return tree_util.tree_map_with_path(_get_mask, params.to_pure_dict())


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    """Creates an optimizer from the config."""
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
