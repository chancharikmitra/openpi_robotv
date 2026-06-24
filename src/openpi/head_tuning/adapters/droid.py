"""DROID H5 benchmark adapter for Stage-1 activation extraction.

Loads episodes from a DROID-format HDF5 file (top-level group per episode,
numeric step subgroups) and defines the DROID-specific keyframe selection rule
(gripper state flip + arm-static heuristic).
"""
from __future__ import annotations

import re

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------

def droid_key_idcs(
    joint_pos: np.ndarray,
    gripper_pos: np.ndarray,
    actions: np.ndarray,
    dt: float = 1 / 15,
    th_vel: float = 0.03,
    th_act: float = 0.3,
    chunk_size: int = 10,
) -> np.ndarray:
    """Select key frame indices for a DROID episode.

    Rules applied (in order):
    - R2: frames where the gripper binary state flips.
    - R1: frames where the arm joint velocity norm is below *th_vel* (arm static).
    - Always include the first and last frame.

    Args:
        joint_pos:   ``(F, 7)`` float array of joint positions.
        gripper_pos: ``(F,)`` or ``(F, 1)`` float array of gripper positions
                     (binary-quantized values expected for the flip rule).
        actions:     ``(F, 8)`` float array of actions (used for shape reference
                     but not in the active rules).
        dt:          Time step in seconds (default 1/15 for DROID's 15 Hz).
        th_vel:      Joint velocity norm threshold below which a frame is
                     considered "arm static" (rad/s).
        th_act:      Action norm threshold for the (currently inactive) action
                     peak rule.
        chunk_size:  Window size for the (currently inactive) action peak rule.

    Returns:
        Sorted 1-D integer array of selected frame indices.
    """
    F = len(joint_pos)
    key = np.zeros(F, dtype=bool)

    # R2: gripper state flip
    flip = np.where(gripper_pos[1:] != gripper_pos[:-1])[0] + 1
    key[flip] = True

    # R1: arm static (joint velocity norm below threshold)
    dq_norm = np.linalg.norm(np.diff(joint_pos, axis=0) / dt, axis=1)
    static = np.where(dq_norm < th_vel)[0] + 1
    key[static] = True

    # Always include first and last frame
    key[0] = key[-1] = True
    return np.where(key)[0]  # ascending indices


# ---------------------------------------------------------------------------
# Episode loader
# ---------------------------------------------------------------------------

def _extract_instruction_from_group(name: str, group: h5py.Group) -> str:
    """Attempt to read a language instruction from HDF5 attributes or datasets.

    Falls back to inferring a readable name from the group name itself.
    """
    # 1) Group attributes
    try:
        for key in ("instruction", "prompt", "language_instruction", "task"):
            if key in group.attrs:
                val = group.attrs[key]
                try:
                    arr = np.array(val)
                    if arr.ndim == 0:
                        val = arr.item()
                except Exception:
                    pass
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="ignore")
                return str(val)
    except Exception:
        pass

    # 2) String datasets inside the group
    for key in ("instruction", "prompt", "language_instruction", "task"):
        try:
            if key in group and isinstance(group[key], h5py.Dataset):
                ds = group[key]
                try:
                    val = ds[()] if ds.shape == () else ds[0]
                    if isinstance(val, bytes):
                        return val.decode("utf-8", errors="ignore")
                    return str(val)
                except Exception:
                    continue
        except Exception:
            continue

    # 3) Fallback: derive from group name
    lower = name.lower()
    m = re.search(r"push[-_ ]red[-_ ]cup[-_ ]to[-_ ]red[-_ ]bowl", lower)
    if m:
        return "pick up red cube"
    base = name.replace("-", " ").replace("_", " ")
    base = re.sub(r"\s+", " ", base).strip()
    return base


def load_droid_episodes(
    h5_path: str,
    task_prompt: str,
    max_episodes: int | None = None,
) -> dict:
    """Load DROID episodes from an HDF5 file.

    The file is expected to have one top-level group per episode, each
    containing numeric step subgroups (``"0"``, ``"1"``, …) with datasets:

    - ``obs``:       ``(8,)`` float64 — first 7 are joint_position, last 1 is
                     gripper_position.
    - ``act_vel``:   ``(8,)`` float64 — ``[joint_velocity(7), gripper_position(1)]``.
    - ``act_pos``:   ``(8,)`` float64 — ``[joint_position(7), gripper_position(1)]``.
    - ``rgb_right``: ``(256, 256, 3)`` uint8 — exterior camera.
    - ``rgb_wrist``: ``(256, 256, 3)`` uint8 — wrist camera.

    Args:
        h5_path:      Path to the input DROID HDF5 file.
        task_prompt:  Language instruction string used as the task-name key in
                      the returned dict (overrides any embedded instruction).
        max_episodes: Maximum number of episodes to load.  ``None`` means all.

    Returns:
        Nested dict ``{task_name: {ep_idx: {"observations": [...],
        "actions": [...], "action_dict_list": [...]}}}`` where ``ep_idx``
        starts at 1 and increments in input order.
    """
    data: dict[str, dict[int, dict[str, list]]] = {}
    processed = 0

    with h5py.File(h5_path, "r") as f:
        for episode_name in f.keys():
            grp = f[episode_name]
            if not isinstance(grp, h5py.Group):
                continue

            steps = sorted([s for s in grp.keys() if s.isdigit()], key=lambda x: int(x))
            if not steps:
                continue

            task_name = task_prompt
            episodes: dict[int, dict[str, list]] = data.get(task_name, {})

            obs_list: list[dict] = []
            act_list: list[np.ndarray] = []
            action_dict_list: list[dict] = []

            for s in steps:
                sg = grp[s]
                req = all(k in sg for k in ("obs", "act_vel", "act_pos", "rgb_right", "rgb_wrist"))
                if not req:
                    continue
                obs = np.asarray(sg["obs"]).astype(np.float32)           # (8,)
                act_vel = np.asarray(sg["act_vel"]).astype(np.float32)   # (8,)
                act_pos = np.asarray(sg["act_pos"]).astype(np.float32)   # (8,)
                img_left = np.asarray(sg["rgb_right"])                    # (256, 256, 3)
                img_wrist = np.asarray(sg["rgb_wrist"])                   # (256, 256, 3)

                jp = np.asarray(obs[:7], dtype=np.float32)               # (7,)
                gp = np.asarray(obs[7:8], dtype=np.float32)              # (1,)
                jpos_act = np.asarray(act_pos[:7], dtype=np.float32)     # (7,)
                jv = np.asarray(act_vel[:7], dtype=np.float32)           # (7,)
                ga = np.asarray(act_vel[7], dtype=np.float32)            # scalar

                obs_list.append({
                    "observation/exterior_image_1_left": img_left,
                    "observation/wrist_image_left":      img_wrist,
                    "observation/joint_position":        jp,
                    "observation/gripper_position":      gp,
                    "prompt":                            task_prompt,
                })
                act_list.append(np.concatenate([jv, [ga]], dtype=np.float32))
                action_dict_list.append({
                    "act_joint_pos": jpos_act,
                    "act_joint_vel": jv,
                    "act_gripper_pos": np.array([ga], dtype=np.float32),
                })

            if not obs_list:
                continue

            next_ep_idx = (max(episodes.keys()) + 1) if episodes else 1
            episodes[next_ep_idx] = {
                "observations":    obs_list,
                "actions":         act_list,
                "action_dict_list": action_dict_list,
            }
            data[task_name] = episodes

            processed += 1
            if max_episodes is not None and processed >= max_episodes:
                return data

    return data
