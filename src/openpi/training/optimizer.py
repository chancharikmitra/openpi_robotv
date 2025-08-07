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
        # 修复：正确处理路径格式
        # path是一个tuple，每个元素可能是字符串或列表形式的字符串
        actual_path_parts = []
        for part in path:
            part_str = str(part)
            # 如果是 "['xxx']" 格式，提取xxx
            if part_str.startswith("['") and part_str.endswith("']"):
                actual_path_parts.append(part_str[2:-2])
            else:
                actual_path_parts.append(part_str)
        
        actual_path = "/".join(actual_path_parts)
        
        # Check for attention weights (both regular and LoRA)
        is_attn_weight = "PaliGemma/llm/layers/attn" in actual_path
        is_regular_weight = actual_path.endswith("/w")
        is_lora_weight = actual_path.endswith("/lora_a") or actual_path.endswith("/lora_b")
        
        # We only process attention weights (regular or LoRA)
        if not (is_attn_weight and (is_regular_weight or is_lora_weight)):
            return jnp.ones_like(leaf, dtype=jnp.int8) if hasattr(leaf, 'shape') else 1

        layer_dim = leaf.shape[0]
        
        # Determine the head axis based on parameter type and name
        head_axis = -1
        param_name = actual_path_parts[-2] if len(actual_path_parts) >= 2 else ""  # e.g., 'q_einsum', 'kv_einsum', etc.
        
        if is_lora_weight:
            # For LoRA weights, head dimension is consistently at axis 1
            head_axis = 1
        else:
            # For regular weights, determine based on parameter name and actual shape
            if param_name in ["q_einsum", "attn_vec_einsum"]:
                # Query和output投影：(layers, heads, input_dim, output_dim)
                # 头部维度在axis 1
                head_axis = 1
            elif param_name in ["kv_einsum"]:
                # KV投影对于Gemma：(layers, 2, num_kv_heads, input_dim, output_dim)
                # 但Gemma使用多查询注意力，KV头数=1，所以实际形状是(layers, 2, 1, ...)
                # 在这种情况下，我们需要特殊处理
                if leaf.ndim >= 3 and leaf.shape[2] > 1:
                    head_axis = 2  # 如果KV头数>1
                else:
                    # 对于多查询注意力（KV头数=1），我们需要不同的策略
                    # 在这种情况下，所有query头共享同一个KV，所以如果任何头被训练，KV就应该被训练
                    head_axis = -2  # 使用特殊值表示需要特殊处理
            elif param_name in ["qkv_einsum"]:
                # QKV合并投影：(layers, 3, heads, ...)
                head_axis = 2

        # Safety check
        if head_axis == -1:
            # 如果无法确定head axis，默认返回全零掩码（完全冻结）
            return jnp.zeros_like(leaf, dtype=jnp.int8)

        final_mask = jnp.zeros_like(leaf, dtype=jnp.int8)

        # 特殊处理多查询注意力的KV权重
        if param_name == "kv_einsum" and head_axis == -2:
            # 对于多查询注意力，如果任何层有训练的头，那么该层的KV都应该被训练
            for layer_idx, heads_to_train in trainable_heads_map.items():
                if layer_idx >= layer_dim:
                    continue
                if heads_to_train:  # 如果这一层有任何头需要训练
                    # 训练整个层的KV权重
                    slicer = [slice(None)] * leaf.ndim
                    slicer[0] = layer_idx  # Layer dimension
                    final_mask = final_mask.at[tuple(slicer)].set(1)
        else:
            # 普通的头部掩码处理
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


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    """Creates an optimizer from the config."""
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
