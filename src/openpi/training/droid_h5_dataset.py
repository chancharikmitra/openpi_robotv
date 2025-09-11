import h5py, tensorflow as tf, numpy as np
import re
from enum import Enum, auto

class DroidActionSpace(Enum):
    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()

# ---------------- episode generator ----------------
def _episode_generator(h5_path, fixed_instruction=None):
    """
    Dataset structure assumptions:
    - Top-level: a group per trajectory/episode
    - Within each episode: numeric step subgroups ("0", "1", ...)
      Each step contains:
        - "obs": (8,) float64  -> first 7 are joint_position, last 1 is gripper_position
        - "action": (15,) float64 -> [7 joint_position, 7 joint_velocity, 1 gripper_position]
        - "rgb_left", "rgb_right", "rgb_wrist": (256,256,3) uint8
    - Prompt is extracted from attributes/datasets, fallback to group name
    """
    with h5py.File(h5_path, "r") as f:
        for episode_name in f.keys():
            episode_group = f[episode_name]
            if not isinstance(episode_group, h5py.Group):
                continue

            # Collect per-step data
            step_names = [name for name in episode_group.keys() if name.isdigit()]
            if len(step_names) == 0:
                continue
            step_names = sorted(step_names, key=lambda x: int(x))

            obs_list = []
            act_list = []
            left_imgs = []
            right_imgs = []
            wrist_imgs = []

            for s in step_names:
                step_group = episode_group[s]
                if "obs" not in step_group or "action" not in step_group:
                    continue
                obs = np.asarray(step_group["obs"]).astype(np.float32)
                act = np.asarray(step_group["action"]).astype(np.float32)
                obs_list.append(obs)
                act_list.append(act)
                # If any image is missing for that step, drop the step for alignment
                try:
                    left_imgs.append(np.asarray(step_group["rgb_left"]))
                    right_imgs.append(np.asarray(step_group["rgb_right"]))
                    wrist_imgs.append(np.asarray(step_group["rgb_wrist"]))
                except Exception:
                    obs_list.pop()
                    act_list.pop()
                    continue

            if len(obs_list) == 0:
                continue

            obs_arr = np.stack(obs_list, axis=0)                 # (T, 8)
            act_arr = np.stack(act_list, axis=0)                 # (T, 15)
            left_arr = np.stack(left_imgs, axis=0)               # (T,256,256,3)
            right_arr = np.stack(right_imgs, axis=0)             # (T,256,256,3)
            wrist_arr = np.stack(wrist_imgs, axis=0)             # (T,256,256,3)

            T = obs_arr.shape[0]
            joint_pos_obs = obs_arr[:, :7].astype(np.float32)    # (T,7)
            gripper_pos_obs = obs_arr[:, 7:8].astype(np.float32) # (T,1)

            # New action layout: [jpos(7), jvel(7), gpos(1)]
            joint_pos_act   = act_arr[:, :7].astype(np.float32)      # (T,7)
            joint_vel_act   = act_arr[:, 7:14].astype(np.float32)    # (T,7)
            gripper_pos_act = act_arr[:, 14:15].astype(np.float32)   # (T,1)

            # Extract instruction
            def extract_instruction_from_group(name: str, group: h5py.Group) -> str:
                # 1) attributes
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

                # 2) datasets
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

                # 3) fallback: parse from name
                lower = name.lower()
                m = re.search(r"pick[-_ ]red[-_ ]cube", lower)
                if m:
                    return "pick red cube"
                base = name.replace("-", " ").replace("_", " ")
                base = re.sub(r"\s+", " ", base).strip()
                return base

            # fixed instruction override
            if fixed_instruction is not None:
                instruction_text = fixed_instruction
            else:
                instruction_text = extract_instruction_from_group(episode_name, episode_group)
            instruction_bytes = np.array([instruction_text.encode("utf-8")] * T)
            file_path_bytes = np.array([episode_name.encode()] * T)

            yield {
                "observation":{
                    "joint_position"       : joint_pos_obs,
                    "gripper_position"     : gripper_pos_obs,
                    "exterior_image_1_left": left_arr,
                    "exterior_image_2_left": right_arr,
                    "wrist_image_left"     : wrist_arr,
                },
                "action_dict":{
                    # Use target values from action
                    "joint_position"   : joint_pos_act,
                    "joint_velocity"   : joint_vel_act,
                    "gripper_position" : gripper_pos_act,
                },
                "language_instruction"  : instruction_bytes,
                "language_instruction_2": np.array([b""] * T),
                "language_instruction_3": np.array([b""] * T),
                "traj_metadata":{
                    "episode_metadata":{"file_path": file_path_bytes}
                },
            }

# ---------------- Data-loader ----------------
class DroidH5Dataset:
    def __init__(
        self,
        h5_path      : str,
        batch_size   : int,
        *,
        shuffle      : bool=True,
        action_chunk_size : int=16,
        action_space : DroidActionSpace=DroidActionSpace.JOINT_VELOCITY,
        shuffle_buffer_size : int = 256,
        num_parallel_calls  : int = tf.data.AUTOTUNE,
        fixed_instruction : str = None,  # fixed instruction string
    ):
        tf.config.set_visible_devices([], "GPU")

        # 1. Build tf.data.Dataset (trajectories)
        output_sig = {
            "observation":{
                "joint_position"       : tf.TensorSpec((None,7), tf.float32),
                "gripper_position"     : tf.TensorSpec((None,1), tf.float32),
                "exterior_image_1_left": tf.TensorSpec((None,256,256,3), tf.uint8),
                "exterior_image_2_left": tf.TensorSpec((None,256,256,3), tf.uint8),
                "wrist_image_left"     : tf.TensorSpec((None,256,256,3), tf.uint8),
            },
            "action_dict":{
                "joint_position"   : tf.TensorSpec((None,7), tf.float32),
                "joint_velocity"   : tf.TensorSpec((None,7), tf.float32),
                "gripper_position" : tf.TensorSpec((None,1), tf.float32),
            },
            "language_instruction"  : tf.TensorSpec((None,), tf.string),
            "language_instruction_2": tf.TensorSpec((None,), tf.string),
            "language_instruction_3": tf.TensorSpec((None,), tf.string),
            "traj_metadata":{
                "episode_metadata":{"file_path": tf.TensorSpec((None,), tf.string)}
            },
        }
        dataset = tf.data.Dataset.from_generator(
            lambda: _episode_generator(h5_path, fixed_instruction),
            output_signature=output_sig,
        )

        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(buffer_size=20)

        # Do not filter on success flag for custom datasets
        # dataset = dataset.filter(
        #     lambda traj: tf.strings.regex_full_match(
        #         traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*")
        # )

        dataset = dataset.repeat()                         # Repeat indefinitely

        # --------- restructure ----------
        def restructure(traj):
            actions = tf.concat(
                ( traj["action_dict"]["joint_position"]
                  if action_space==DroidActionSpace.JOINT_POSITION
                  else traj["action_dict"]["joint_velocity"],
                  traj["action_dict"]["gripper_position"]), axis=-1)
            exterior_img = tf.cond(
                tf.random.uniform([]) > .5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # instruction = tf.random.shuffle([
            #     traj["language_instruction"],
            #     traj["language_instruction_2"],
            #     traj["language_instruction_3"]])[0]
            instruction = traj["language_instruction"]
            return {
                "actions": actions,
                "observation":{
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
            }
        dataset = dataset.map(restructure, num_parallel_calls)

        # --------- chunk_actions ----------
        def chunk_actions(traj):
            T = tf.shape(traj["actions"])[0]
            H = action_chunk_size
            pre_idx = tf.range(H)[None] + tf.range(T)[:, None]          # (T,H)
            idx = tf.minimum(pre_idx, T-1)

            chunk = tf.gather(traj["actions"], idx)                    # (T,H,8)

            # For JOINT_VELOCITY: zero-pad velocity beyond tail, keep last gripper pos
            if action_space == DroidActionSpace.JOINT_VELOCITY:
                over = pre_idx >= T                                     # (T,H)
                over_exp = over[..., None]                              # (T,H,1)

                jv = tf.where(over_exp, tf.zeros_like(chunk[..., :7]), chunk[..., :7])
                last_gp = traj["actions"][T-1:T, 7:]
                gp_target = tf.broadcast_to(last_gp, tf.shape(chunk[..., 7:]))
                gp = tf.where(over_exp, gp_target, chunk[..., 7:])

                chunk = tf.concat([jv, gp], axis=-1)

            traj["actions"] = chunk
            return traj
        dataset = dataset.map(chunk_actions, num_parallel_calls)

        # --------- filter_idle (disabled) ----------
        def filter_idle(traj):
            first_half = traj["actions"][:action_chunk_size//2]
            if action_space==DroidActionSpace.JOINT_POSITION:
                return tf.reduce_any(tf.abs(first_half - first_half[:1]) > 1e-3)
            return tf.reduce_any(tf.abs(first_half) > 1e-3)
        # dataset = dataset.filter(filter_idle)

        # --------- flatten ----------
        dataset = dataset.flat_map(lambda traj:
            tf.data.Dataset.from_tensor_slices(traj))

        # shuffle / batch / prefetch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self._ds = dataset

    # iterable
    def __iter__(self):
        yield from self._ds.as_numpy_iterator()

    def __len__(self):               # Estimated value
        return 10000

if __name__ == "__main__":
    # Example 1: use fixed instruction
    loader = DroidH5Dataset(
        h5_path="/scr2/yusenluo/openpi_robotv/robotv_dataset/pick-up-red-mug-20.h5",
        batch_size=32,
        action_space=DroidActionSpace.JOINT_VELOCITY,
        shuffle=True,
        fixed_instruction="pick up red mug",
    )
    for batch in loader:
        print(batch["actions"].shape)   # (32,16,8)
        print(batch["observation"]["image"].shape)   # (32,256,256,3)
        print(batch["observation"]["wrist_image"].shape)   # (32,256,256,3)
        print(batch["observation"]["joint_position"].shape)   # (32,7)
        print(batch["observation"]["gripper_position"].shape)   # (32,1)
        print(batch["prompt"])
        break
  
