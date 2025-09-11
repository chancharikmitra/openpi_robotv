import jax, jax.numpy as jnp, numpy as np, flax.linen as nn
import os
from flax.linen.attention import MultiHeadDotProductAttention as MHA
from openpi.training import config as pi_cfg
from openpi.models import model as pi_model
from openpi.shared import download
# ------------------------- Dependencies -------------------------
import h5py  # type: ignore
import numpy as np
import re
import random
import h5py  # type: ignore
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
# ---------------------------------- Output configuration ----------------------------------
# Change this path if you want to write to a different location
ATTN_H5_PATH = "/scr2/yusenluo/openpi_robotv/attention_dataset/pick-up-red-mug-20_keyframe.h5" # "wipe_eval_attention_last_token_single_action_negative.h5"
# Max number of episodes to process (across all tasks)
MAX_EPISODES = 20
USE_KEYFRAME = True
# from tasks import Pick_training_tasks
from openpi.llm_instruction_verb_filter import instruction_matches_prompt
LLM_PROMPT_KEY = "pick_place"
# If True, only append action labels to an existing attention file without recomputing activations
APPEND_ACTION_LABELS_ONLY = False
# 1) Load model config & weights ----------------------------------------------------------
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

if not APPEND_ACTION_LABELS_ONLY:
    config = config.get_config("pi0_fast_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
    # Ensure normalization assets are present (includes droid/norm_stats.json)
    download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid/assets")

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, checkpoint_dir)      # Pi0FAST Module instance (no weights)

# 2) Build Observation (with batch dimension) -------------------------------------------
from openpi.models.model import Observation


def extract_key_idcs(
    joint_pos: np.ndarray,           # (F,7)
    gripper_pos: np.ndarray,         # (F,)  (or (F,1))
    actions: np.ndarray,             # (F,7)
    dt: float = 1/15,                # Droid default 15 Hz
    th_vel: float = 0.03,            # rad/s 'static' threshold
    th_act: float = 0.3,             # action peak threshold
    chunk_size: int = 10,
) -> np.ndarray:
    F = len(joint_pos)
    key = np.zeros(F, dtype=bool)
    # diff = np.diff(gripper_pos.squeeze()).astype(float)   # drop single-dim for diff

    # # ------ Visualize gripper velocity distribution ------
    # try:
    #     import matplotlib.pyplot as plt  # for debugging only; remove if not needed

    #     plt.figure(figsize=(4, 3))
    #     plt.hist(diff.flatten(), bins=40, color="steelblue", edgecolor="black")
    #     plt.xlabel("|Δ gripper_pos|")
    #     plt.ylabel("Count")
    #     plt.title("Histogram of gripper position differences")
    #     plt.tight_layout()
    #     plt.savefig("gripper_diff_hist.png")
    #     plt.close()
    #     breakpoint()
    # except Exception as e:  # noqa: BLE001
    #     # tolerate environments without display/matplotlib
    #     print(f"[WARN] cannot plot histogram: {e}")
    # ---- R2: gripper state flip ----
    flip = np.where(gripper_pos[1:] != gripper_pos[:-1])[0] + 1
    key[flip] = True

    # ---- R1: arm static ----
    dq_norm = np.linalg.norm(np.diff(joint_pos, axis=0)/dt, axis=1)
    static = np.where(dq_norm < th_vel)[0] + 1
    key[static] = True

    # ---- R3: action peak per 10 frames ----
    # for c in range(0, F, chunk_size):
    #     seg = actions[c:c+chunk_size]
    #     if seg.size == 0: continue
    #     idx = np.argmax(np.linalg.norm(seg, axis=1))
    #     if np.linalg.norm(seg[idx]) > th_act:
    #         key[c+idx] = True

    key[0] = key[-1] = True          # always keep first/last frame
    return np.where(key)[0]          # ascending indices



def extract_observations(h5_path, max_episodes: int | None = None):
    """Adapted to the new H5 structure: top-level group per episode, with numeric step subgroups.

    Per-frame fields:
      - obs: (8,) float64 → first 7 are joint_position, last 1 is gripper_position
      - action: (15,) float64 → [7 joint_position, 7 joint_velocity, 1 gripper_position]
      - rgb_left / rgb_right / rgb_wrist: (256,256,3) uint8

    Returns {task_name: {ep_idx: {observations, actions, action_dict_list}}}.
    task_name is inferred from the prompt (prefer attributes/datasets; fallback to group name).
    """

    def extract_instruction_from_group(name: str, group: h5py.Group) -> str:
        # 1) group attributes
        try:
            for key in ["instruction", "prompt", "language_instruction", "task"]:
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

        # 2) string datasets inside the group
        for key in ["instruction", "prompt", "language_instruction", "task"]:
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

        # 3) fallback: infer from group name
        lower = name.lower()
        m = re.search(r"pick[-_ ]red[-_ ]cube", lower)
        if m:
            return "pick red cube"
        base = name.replace("-", " ").replace("_", " ")
        base = re.sub(r"\s+", " ", base).strip()
        return base

    data: dict[str, dict[int, dict[str, list]]] = {}
    processed = 0

    with h5py.File(h5_path, "r") as f:
        for episode_name in f.keys():
            grp = f[episode_name]
            if not isinstance(grp, h5py.Group):
                continue

            # Step subgroups (numeric strings)
            steps = sorted([s for s in grp.keys() if s.isdigit()], key=lambda x: int(x))
            if not steps:
                continue

            # prompt as task_name key; if FORCED_PROMPT is set, override
            # prompt_text = extract_instruction_from_group(episode_name, grp)
            prompt_text = "pick up red mug"
            # forced_prompt = os.environ.get("FORCED_PROMPT", "").strip()
            # if forced_prompt:
            #     prompt_text = forced_prompt
            task_name = prompt_text

            episodes: dict[int, dict[str, list]] = data.get(task_name, {})  # support multiple episodes per task

            obs_list: list[dict] = []
            act_list: list[np.ndarray] = []
            action_dict_list: list[dict] = []

            for s in steps:
                sg = grp[s]
                req = all(k in sg for k in ("obs", "action", "rgb_left", "rgb_wrist"))
                if not req:
                    continue
                obs = np.asarray(sg["obs"]).astype(np.float32)          # (8,)
                act = np.asarray(sg["action"]).astype(np.float32)       # (15,)
                img_left  = np.asarray(sg["rgb_left"])                  # (256,256,3)
                img_wrist = np.asarray(sg["rgb_wrist"])                 # (256,256,3)

                # Normalize dtypes/shapes for consistent stacking
                jp = np.asarray(obs[:7], dtype=np.float32)      # (7,)
                gp = np.asarray(obs[7:8], dtype=np.float32)     # (1,)
                # New action layout: [jpos(7), jvel(7), gpos(1)]
                jpos_act = np.asarray(act[:7], dtype=np.float32)      # (7,)
                jv       = np.asarray(act[7:14], dtype=np.float32)    # (7,)
                ga       = np.asarray(act[14], dtype=np.float32)      # scalar

                obs_list.append({
                    "observation/exterior_image_1_left": img_left,
                    "observation/wrist_image_left":     img_wrist,
                    "observation/joint_position":       jp,
                    "observation/gripper_position":     gp,
                    "prompt": prompt_text,
                })
                # For training with joint velocity action space: [joint_velocity(7), gripper_position(1)]
                act_list.append(np.concatenate([jv, [ga]], dtype=np.float32))
                action_dict_list.append({
                    # Use target joint pos/vel/gripper from the action
                    "act_joint_pos": jpos_act,
                    "act_joint_vel": jv,
                    "act_gripper_pos": np.array([ga], dtype=np.float32),
                })

            if not obs_list:
                continue

            # Create a new sequential episode index under this task_name
            next_ep_idx = (max(episodes.keys()) + 1) if episodes else 1
            episodes[next_ep_idx] = {
                "observations": obs_list,
                "actions": act_list,
                "action_dict_list": action_dict_list,
            }
            data[task_name] = episodes

            processed += 1
            if max_episodes is not None and processed >= max_episodes:
                return data

    return data

# Debug-only: set EXTRACT_OBS_ONLY=1 to parse only and print stats, then exit
if __name__ == "__main__":
    import os, sys
    if os.environ.get("EXTRACT_OBS_ONLY") == "1":
        h5_path = "/home/yusenluo/pick-red-cube_250827.h5"
        dataset = extract_observations(h5_path, max_episodes=3)
        num_tasks = len(dataset)
        num_eps = sum(len(eps) for eps in dataset.values())
        print(f"tasks={num_tasks} episodes={num_eps}")
        total_frames = 0
        for task_name, eps in dataset.items():
            print(f"\n== Task: {task_name}")
            for ep_idx, ep in eps.items():
                obs = ep["observations"]
                acts = ep["actions"]
                adict = ep["action_dict_list"]
                F = len(obs)
                total_frames += F
                print(f"  Episode {ep_idx}: frames={F}")
                if F:
                    s0 = obs[0]
                    print("    keys:", list(s0.keys()))
                    img = s0["observation/exterior_image_1_left"]
                    wrist = s0["observation/wrist_image_left"]
                    jp = np.asarray(s0["observation/joint_position"])  # (7,)
                    gp = np.asarray(s0["observation/gripper_position"]) # () or (1,)
                    print(f"    image shape={img.shape} dtype={img.dtype}")
                    print(f"    wrist shape={wrist.shape} dtype={wrist.dtype}")
                    print(f"    joint_position shape={jp.shape} dtype={jp.dtype} min={jp.min():.3f} max={jp.max():.3f}")
                    gpv = float(gp.reshape(-1)[0])
                    print(f"    gripper_position example={gpv:.3f}")
                    a0 = np.asarray(acts[0])
                    print(f"    action[0] shape={a0.shape} dtype={a0.dtype}")
        print(f"\nTOTAL_FRAMES={total_frames}")
        sys.exit(0)

# Usage
h5_path = "/scr2/yusenluo/openpi_robotv/robotv_dataset/pick-up-red-mug-20.h5"
dataset = extract_observations(h5_path, max_episodes=MAX_EPISODES)

# Iterate and run inference, printing progress
file_mode = "a" if APPEND_ACTION_LABELS_ONLY else "w"
with h5py.File(ATTN_H5_PATH, file_mode) as h5_out:  # auto flush & close on exit
    ep_counter = 0  # number of episodes written
    for task_name, eps in dataset.items():
        for ep_idx, ep_data in eps.items():
            total_frames = len(ep_data["observations"])

            if USE_KEYFRAME:
                jp  = np.stack([obs["observation/joint_position"]      for obs in ep_data["observations"]])
                gp  = np.stack([obs["observation/gripper_position"]    for obs in ep_data["observations"]])
                acts= np.stack(ep_data["actions"])
                key_idcs = extract_key_idcs(jp, gp, acts)          # ndarray
            else:
                key_idcs = np.arange(total_frames) # all frames

            print(f"Task {task_name} Episode {ep_idx}: keep {len(key_idcs)}/{total_frames} key-frames")
            # ---------- Only iterate key_idcs (with progress bar) ----------
            for frame_idx in tqdm(key_idcs, desc=f"{task_name} ep{ep_idx:03d}", total=len(key_idcs)):
                # frame_idx = original index (0‑based)
                obs = ep_data["observations"][frame_idx]
                act = ep_data["actions"][frame_idx]
                true_frame_idx = frame_idx + 1                # keep 1‑based naming
                # HDF5 path: /task/episode_xxx/frame_xxxx/
                grp_path = f"{task_name}/episode_{ep_idx:03d}/frame_{frame_idx:04d}"

                # Append action labels only on existing groups; do not recompute attention
                if APPEND_ACTION_LABELS_ONLY:
                    if grp_path not in h5_out:
                        continue  # only fill existing samples
                    grp = h5_out[grp_path]
                    action_arr = np.asarray(act, dtype=np.float32)
                    if "action_label" in grp:
                        del grp["action_label"]
                    grp.create_dataset(
                        "action_label",
                        data=action_arr,
                        compression="gzip",
                    )

                outputs, attention_outputs = policy.infer(obs, return_attention_heads=True, return_attention_probs=True)
                grp = h5_out.require_group(grp_path)  # create hierarchy if needed

                # Keep only last-token attention: (layer, head, dim) = (18, 8, 256)
                full_attn = attention_outputs["llm_activations"][0]  # (18, 1, 1018, 8, 256)
                last_token_attn = full_attn[:, 0, -1, :, :]  # (18, 8, 256)
                # Force float32 (h5py does not support bfloat16; otherwise becomes raw bytes)
                last_token_attn = np.asarray(last_token_attn, dtype=np.float32)
                assert last_token_attn.shape == (18, 8, 256)
                if "last_token_attn" in grp:
                    del grp["last_token_attn"]
                grp.create_dataset(
                    "last_token_attn",
                    data=last_token_attn,
                    compression="gzip",
                )

                # Optionally save prefill attention probabilities
                # prefill_probs = attention_outputs.get("llm_attn_probs_prefill") #(18, 1, 8, 1018)
                # if prefill_probs is not None:
                #     prefill_probs = np.asarray(prefill_probs, dtype=np.float32)
                #     if "attn_probs_prefill" in grp:
                #         del grp["attn_probs_prefill"]
                #     grp.create_dataset("attn_probs_prefill", data=prefill_probs, compression="gzip")

                # Optionally save decode attention probabilities
                # decode_probs = attention_outputs.get("llm_attn_probs_decode") #(1, 256, 18, 8, 1274)
                # if decode_probs is not None:
                #     decode_probs = np.asarray(decode_probs, dtype=np.float32)
                #     if "attn_probs_decode" in grp:
                #         del grp["attn_probs_decode"]
                #     grp.create_dataset("attn_probs_decode", data=decode_probs, compression="gzip")

                # Record metadata for traceability
                grp.attrs["full_llm_shape"] = full_attn.shape
                grp.attrs["last_token_idx"] = int(attention_outputs["last_token_idx"])  # type: ignore[arg-type]

                # Write action label
                action_arr = np.asarray(act, dtype=np.float32)
                if "action_label" in grp:
                    del grp["action_label"]
                grp.create_dataset("action_label", data=action_arr, compression="gzip")

            # —— one episode done ——
            ep_counter += 1
            if ep_counter % 50 == 0:
                print("has infereced :", ep_counter)
                h5_out.flush()  # periodic flush to reduce data loss

        # —— one task done ——
        print(f"Finished task {task_name} – {len(eps)} episodes processed.")
        h5_out.flush()
print("done")