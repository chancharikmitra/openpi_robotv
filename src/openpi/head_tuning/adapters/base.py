"""Shared inference and H5 serialization logic for Stage-1 activation extraction.

This module contains setup-agnostic code:
- ``save_attn_blocks``: write the standard attention datasets into an open HDF5 group.
- ``run_inference_and_save``: iterate episodes, select key frames, call ``policy.infer``,
  and persist activations under the standard HDF5 path scheme::

      {task_name}/episode_{ep_idx:03d}/frame_{frame_idx:04d}/
          last_token_attn           (18, 8, 256)  float32
          state_token_attn          (18, 8, 256)  float32  (when available)
          first_action_token_attn   (18, 8, 256)  float32  (when available)
          action_label              (8,)           float32
          predicted_action_label    (8,)           float32
"""
from __future__ import annotations

from typing import Callable

import h5py
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x


# ---------------------------------------------------------------------------
# H5 serialization helpers
# ---------------------------------------------------------------------------

def _overwrite_dataset(grp: h5py.Group, name: str, data: np.ndarray) -> None:
    """Delete-then-create a dataset so repeated runs stay idempotent."""
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data, compression="gzip")


def save_attn_blocks(grp: h5py.Group, attention_outputs: dict) -> None:
    """Write the standard activation datasets into *grp*.

    Handles both 5-D (FAST-style ``[L, B, T, H, D]``) and 4-D
    (``[B, T, H, D]``) tensors by stripping the leading batch/scan singletons
    first, then selecting the relevant token index.

    Args:
        grp: Open, writable HDF5 group for the current frame.
        attention_outputs: Mapping returned by ``policy.infer`` under the
            key ``"attention_outputs"``.
    """
    # ---- last-token activations (full sequence) ----
    full_attn = attention_outputs["llm_activations"]
    while full_attn.ndim > 5 and full_attn.shape[0] == 1:
        full_attn = full_attn[0]  # strip leading scan dim → (18, 1, T, 8, 256)
    if full_attn.ndim == 5:
        last_token_attn = full_attn[:, 0, -1, :, :]  # (18, 8, 256)
    elif full_attn.ndim == 4:
        last_token_attn = full_attn[0, -1, :, :]
    else:
        raise ValueError(f"Unexpected llm_activations shape: {full_attn.shape}")
    last_token_attn = np.asarray(last_token_attn, dtype=np.float32)
    _overwrite_dataset(grp, "last_token_attn", last_token_attn)
    grp.attrs["full_llm_shape"] = full_attn.shape

    # ---- state + first-action heads (preferred path) ----
    sa_attn = attention_outputs.get("llm_state_first_action_activations")
    if sa_attn is not None:
        while sa_attn.ndim > 5 and sa_attn.shape[0] == 1:
            sa_attn = sa_attn[0]
        if sa_attn.ndim == 5:
            state_token_attn = sa_attn[:, 0, 0, :, :]
            first_action_token_attn = sa_attn[:, 0, 1, :, :]
        elif sa_attn.ndim == 4:
            state_token_attn = sa_attn[0, 0, :, :]
            first_action_token_attn = sa_attn[0, 1, :, :]
        else:
            raise ValueError(f"Unexpected llm_state_first_action_activations shape: {sa_attn.shape}")
        _overwrite_dataset(grp, "state_token_attn", np.asarray(state_token_attn, dtype=np.float32))
        _overwrite_dataset(grp, "first_action_token_attn", np.asarray(first_action_token_attn, dtype=np.float32))
    else:
        # Fallback: state-only activations
        state_attn = attention_outputs.get("llm_state_activations")
        if state_attn is not None:
            while state_attn.ndim > 5 and state_attn.shape[0] == 1:
                state_attn = state_attn[0]
            if state_attn.ndim == 5:
                state_token_attn = state_attn[:, 0, 0, :, :]
            elif state_attn.ndim == 4:
                state_token_attn = state_attn[0, 0, :, :]
            else:
                raise ValueError(f"Unexpected llm_state_activations shape: {state_attn.shape}")
            _overwrite_dataset(grp, "state_token_attn", np.asarray(state_token_attn, dtype=np.float32))

    # ---- first-action token activations (separate forward, if available) ----
    fa_attn = attention_outputs.get("llm_first_action_activations")
    if fa_attn is not None:
        fa = np.asarray(fa_attn, dtype=np.float32)
        while fa.ndim > 4 and fa.shape[0] == 1:
            fa = fa[0]
        if fa.ndim == 4:
            fa = fa[:, 0, :, :]
        elif fa.ndim != 3:
            raise ValueError(f"Unexpected llm_first_action_activations shape: {fa_attn.shape}")
        _overwrite_dataset(grp, "first_action_token_attn", fa)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_inference_and_save(
    policy,
    episodes: dict,
    out_h5: str,
    *,
    key_fn: Callable | None = None,
) -> None:
    """Run inference over *episodes* and write activations to *out_h5*.

    Args:
        policy: A trained ``openpi`` policy object with an ``.infer()`` method.
        episodes: Nested dict ``{task_name: {ep_idx: {"observations": [...],
            "actions": [...], "action_dict_list": [...]}}}`` as returned by a
            setup adapter's ``load_*_episodes`` function.
        out_h5: Path of the output HDF5 file to create (mode ``"w"``).
        key_fn: Optional callable ``(joint_pos, gripper_pos, actions) ->
            np.ndarray`` that returns the sorted integer indices of key frames
            to process.  When *None*, all frames are processed.
    """
    ep_counter = 0
    with h5py.File(out_h5, "w") as h5_out:
        for task_name, eps in episodes.items():
            for ep_idx, ep_data in eps.items():
                observations = ep_data["observations"]
                actions = ep_data["actions"]
                total_frames = len(observations)

                if key_fn is not None:
                    jp = np.stack([obs["observation/joint_position"] for obs in observations])
                    gp = np.stack([obs["observation/gripper_position"] for obs in observations])
                    acts_arr = np.stack(actions)
                    key_idcs = key_fn(jp, gp, acts_arr)
                else:
                    key_idcs = np.arange(total_frames)

                print(
                    f"Task {task_name!r} Episode {ep_idx}: "
                    f"keep {len(key_idcs)}/{total_frames} key-frames"
                )

                for frame_idx in tqdm(key_idcs, desc=f"{task_name} ep{ep_idx:03d}", total=len(key_idcs)):
                    obs = observations[frame_idx]
                    act = actions[frame_idx]
                    grp_path = f"{task_name}/episode_{ep_idx:03d}/frame_{frame_idx:04d}"

                    results = policy.infer(
                        obs,
                        return_attention_heads=True,
                        return_attention_probs=True,
                        return_state_heads=False,
                        return_state_and_first_action_heads=True,
                    )
                    attention_outputs = results.get("attention_outputs")
                    predicted_actions = results.get("actions")[0]

                    grp = h5_out.require_group(grp_path)
                    save_attn_blocks(grp, attention_outputs)

                    action_arr = np.asarray(act, dtype=np.float32)
                    _overwrite_dataset(grp, "action_label", action_arr)
                    _overwrite_dataset(
                        grp,
                        "predicted_action_label",
                        np.asarray(predicted_actions, dtype=np.float32),
                    )

                ep_counter += 1
                if ep_counter % 50 == 0:
                    h5_out.flush()

            print(f"Finished task {task_name!r} – {len(eps)} episodes processed.")
            h5_out.flush()
