from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict, return_attention_heads: bool=False, delta_heads=None) -> dict | tuple[dict, dict]:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        if return_attention_heads:
            actions, attention_outputs, last_token_idx = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), return_attention_heads=return_attention_heads, delta_heads=delta_heads, **self._sample_kwargs)
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), return_attention_heads=return_attention_heads, **self._sample_kwargs),
            }
        
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        
        if return_attention_heads:
            # Process attention outputs - unbatch them - maybe not necessary
            # processed_attention = {}
            # if attention_outputs and "llm_activations" in attention_outputs and attention_outputs["llm_activations"] is not None:
            #     processed_attention["llm_activations"] = np.asarray(attention_outputs["llm_activations"][0])  # Remove batch dimension
            # else:
            #     processed_attention["prefill"] = None
            attention_outputs['last_token_idx'] = last_token_idx
            return outputs, attention_outputs#processed_attention
        else:
            return outputs
    # def infer(self, obs: dict, return_attention_heads: bool=False) -> dict:  # type: ignore[misc]
    #     # Make a copy since transformations may modify the inputs in place.
    #     inputs = jax.tree.map(lambda x: x, obs)
    #     inputs = self._input_transform(inputs)
    #     # Make a batch and convert to jax.Array.
    #     inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    #     start_time = time.monotonic()
    #     self._rng, sample_rng = jax.random.split(self._rng)
    #     if return_attention_heads:
    #         actions, attention_outputs = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), return_attention_heads=return_attention_heads, **self._sample_kwargs)
    #         outputs = {
    #             "state": inputs["state"],
    #             "actions": actions,
    #         }
    #     else:
    #         outputs = {
    #             "state": inputs["state"],
    #             "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), return_attention_heads=return_attention_heads, **self._sample_kwargs),
    #         }
    #     # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
    #     outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
    #     model_time = time.monotonic() - start_time

    #     outputs = self._output_transform(outputs)
    #     outputs["policy_timing"] = {
    #         "infer_ms": model_time * 1000,
    #     }
    #     if return_attention_heads:
    #         return outputs, attention_outputs
    #     else:
    #         return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
