"""LIBERO parquet benchmark adapter for Stage-1 activation extraction.

Loads episodes from a LeRobot v2.0 parquet dataset (LIBERO subset) and defines
the LIBERO-specific keyframe selection rule (EE-pose static + gripper-action
sign flip).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


# ---------------------------------------------------------------------------
# Image decode helper
# ---------------------------------------------------------------------------

def _decode_img(cell) -> np.ndarray:
    """Decode a parquet image cell to ``(H, W, 3)`` uint8.

    The cell is expected to be a dict ``{'bytes': bytes, 'path': str}`` as
    produced by the LeRobot parquet writer, or raw bytes.
    """
    if isinstance(cell, dict):
        b = cell["bytes"]
    else:
        b = cell
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------

def libero_key_idcs(
    joint_pos: np.ndarray,
    gripper_pos: np.ndarray,
    actions: np.ndarray,
    th_vel: float = 0.05,
) -> np.ndarray:
    """Select key frame indices for a LIBERO episode.

    LIBERO state is EE-pose (not joint angles) and gripper action is binary
    ``{-1, 1}``.

    Rules applied:
    - R1: frames where the EE-pose change norm is below *th_vel* (static).
    - R2: frames where the gripper action sign flips.
    - Always include first and last frame.

    Note: *joint_pos* here holds EE-pose state (6-DoF + gripper = 7-D);
    *gripper_pos* is the 7th element of the action column, used for the flip
    rule.  The signature matches the shared ``key_fn`` interface expected by
    ``base.run_inference_and_save``.

    Args:
        joint_pos:   ``(F, 7)`` float array — in LIBERO this is the EE-pose
                     state from the dataset (columns 0:6 used for velocity norm).
        gripper_pos: ``(F, 1)`` or ``(F,)`` float array — not used directly;
                     the gripper action column from *actions* is used instead
                     (``actions[:, 6]``).
        actions:     ``(F, 7)`` float array of actions.
        th_vel:      EE-pose change norm threshold.

    Returns:
        Sorted 1-D integer array of selected frame indices.
    """
    F = joint_pos.shape[0]
    key = np.zeros(F, dtype=bool)
    gripper_action = actions[:, 6]
    flip = np.where(np.sign(gripper_action[1:]) != np.sign(gripper_action[:-1]))[0] + 1
    key[flip] = True
    dnorm = np.linalg.norm(np.diff(joint_pos[:, :6], axis=0), axis=1)
    static = np.where(dnorm < th_vel)[0] + 1
    key[static] = True
    key[0] = key[-1] = True
    return np.where(key)[0]


# ---------------------------------------------------------------------------
# Episode loader
# ---------------------------------------------------------------------------

def _load_single_episode(parquet_path: Path, prompt: str):
    """Return ``(obs_list, actions)`` for one LIBERO parquet episode file."""
    table = pq.read_table(parquet_path)
    images = table["image"].to_pylist()
    wrist_images = table["wrist_image"].to_pylist()
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)

    obs_list = []
    for i in range(table.num_rows):
        obs_list.append({
            "observation/image":          _decode_img(images[i]),
            "observation/wrist_image":    _decode_img(wrist_images[i]),
            "observation/state":          states[i],
            # NOTE: these alias keys are only consumed by the keyframe function
            # (libero_key_idcs via base.run_inference_and_save) to satisfy the
            # shared (joint_pos, gripper_pos) signature; they are never forwarded
            # to policy.infer.
            "observation/joint_position": states[i],
            "observation/gripper_position": states[i, 6:7],  # gripper col of state
            "prompt":                     prompt,
        })
    return obs_list, actions


def load_libero_episodes(
    src: str,
    task_prompt: str = "",
    max_episodes: int | None = None,
) -> dict:
    """Load LIBERO episodes from a LeRobot v2.0 parquet dataset.

    Reads ``meta/info.json`` and ``meta/episodes.jsonl`` to enumerate
    episodes, then loads each parquet shard under ``data/``.

    Each episode's language prompt is determined as follows:

    - If *task_prompt* is non-empty, it is used as the prompt for **all**
      episodes (useful when you want to override the dataset's own labels).
    - Otherwise (default), the prompt is read from the per-episode ``tasks``
      field in ``meta/episodes.jsonl`` (a LeRobot v2.0 list of task strings),
      specifically ``ep["tasks"][0]``.

    The top-level key in the returned dict is the prompt string for each
    episode; episodes that share the same prompt are grouped under the same
    key.

    Args:
        src:          Root directory of the LeRobot dataset (contains
                      ``data/`` and ``meta/``).
        task_prompt:  Optional override prompt.  When non-empty, every episode
                      uses this string as its prompt and H5 group key.  When
                      empty (default), each episode's prompt is derived from
                      the per-episode ``tasks`` field in ``meta/episodes.jsonl``.
        max_episodes: Maximum number of episodes to load.  ``None`` means all.

    Returns:
        Nested dict ``{prompt: {ep_idx: {"observations": [...],
        "actions": [...], "action_dict_list": [...]}}}`` where ``ep_idx``
        matches the original ``episode_index`` from the dataset metadata.
    """
    src_path = Path(src)
    info = json.loads((src_path / "meta" / "info.json").read_text())
    episodes_meta = [
        json.loads(line)
        for line in (src_path / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    data: dict[str, dict[int, dict[str, list]]] = {}
    ep_counter = 0

    for ep in episodes_meta:
        if max_episodes is not None and ep_counter >= max_episodes:
            break
        ep_idx = ep["episode_index"]

        if task_prompt:
            # Caller-supplied override: use the same prompt for every episode.
            prompt = task_prompt
        else:
            # Default: derive prompt from the per-episode tasks field (LeRobot v2.0).
            assert len(ep["tasks"]) == 1, (
                f"ep {ep_idx} has {len(ep['tasks'])} tasks; LIBERO subset assumed single-task"
            )
            prompt = ep["tasks"][0]

        parquet_rel = info["data_path"].format(
            episode_chunk=ep_idx // info["chunks_size"],
            episode_index=ep_idx,
        )
        parquet_path = src_path / parquet_rel
        obs_list, actions = _load_single_episode(parquet_path, prompt)

        n = len(obs_list)
        action_dict_list = [
            {"act_full": np.asarray(actions[i], dtype=np.float32)}
            for i in range(n)
        ]

        episodes: dict[int, dict[str, list]] = data.get(prompt, {})
        episodes[ep_idx] = {
            "observations":    obs_list,
            "actions":         [actions[i] for i in range(n)],
            "action_dict_list": action_dict_list,
        }
        data[prompt] = episodes

        ep_counter += 1

    return data
