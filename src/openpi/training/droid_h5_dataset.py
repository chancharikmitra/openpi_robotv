import h5py, tensorflow as tf, numpy as np
import re
from enum import Enum, auto

class DroidActionSpace(Enum):
    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()

# ---------------- episode generator ----------------
def _episode_generator(h5_path):
    """
    适配 pick-red-cube_250827.h5 的结构：
    - 顶层：每个轨迹为一个组，名称中包含 success/…
    - 轨迹组内：按时间步的子组（如 "0", "1", ...）
      每个子组包含：
        - "obs": (8,) float64  -> 前7位为 joint_position，最后1位为 gripper_position
        - "action": (8,) float64 -> 同理，前7位 + 最后1位
        - "rgb_left", "rgb_right", "rgb_wrist": (256,256,3) uint8
    - prompt 从文件中读取（优先顺序：组属性 -> 组内字符串数据集 -> 组名推断）
    """
    with h5py.File(h5_path, "r") as f:
        for episode_name in f.keys():
            episode_group = f[episode_name]
            if not isinstance(episode_group, h5py.Group):
                continue

            # 收集按时间步的数据
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
                # 图像（若缺失则跳过该帧）
                try:
                    left_imgs.append(np.asarray(step_group["rgb_left"]))
                    right_imgs.append(np.asarray(step_group["rgb_right"]))
                    wrist_imgs.append(np.asarray(step_group["rgb_wrist"]))
                except Exception:
                    # 若任一视角缺失，对齐去掉该步
                    obs_list.pop()
                    act_list.pop()
                    continue

            if len(obs_list) == 0:
                continue

            obs_arr = np.stack(obs_list, axis=0)                 # (T, 8)
            act_arr = np.stack(act_list, axis=0)                 # (T, 8)
            left_arr = np.stack(left_imgs, axis=0)               # (T,256,256,3)
            right_arr = np.stack(right_imgs, axis=0)             # (T,256,256,3)
            wrist_arr = np.stack(wrist_imgs, axis=0)             # (T,256,256,3)

            T = obs_arr.shape[0]
            joint_pos_obs = obs_arr[:, :7].astype(np.float32)    # (T,7)
            gripper_pos_obs = obs_arr[:, 7:8].astype(np.float32) # (T,1)

            # action 前7位为关节速度，最后一位为夹爪
            joint_vel_act = act_arr[:, :7].astype(np.float32)    # (T,7)
            gripper_pos_act = act_arr[:, 7:8].astype(np.float32) # (T,1)

            # 语言与元数据：从文件中读取 prompt
            def extract_instruction_from_group(name: str, group: h5py.Group) -> str:
                # 1) 组属性
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

                # 2) 组内字符串数据集
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

                # 3) 退化：从组名解析
                lower = name.lower()
                m = re.search(r"pick[-_ ]red[-_ ]cube", lower)
                if m:
                    return "pick red cube"
                # 一般化清洗
                base = name.replace("-", " ").replace("_", " ")
                base = re.sub(r"\s+", " ", base).strip()
                return base

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
                    # 未提供动作的关节位置，放置为全零占位，形状一致
                    "joint_position"   : np.zeros_like(joint_pos_obs),
                    # 关节速度由 action 提供
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
            lambda: _episode_generator(h5_path),
            output_signature=output_sig,
        )

        # ====== Original DroidRldsDataset pipeline, adapted line-by-line; replace dlimp calls with native tf.data ======

        if shuffle:
            dataset = dataset.shuffle(buffer_size=20)

        # Filter successful trajectories
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*")
        )

        dataset = dataset.repeat()                         # Repeat dataset indefinitely

        # --------- traj_map (restructure) ----------
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
            # Optionally sample one of the language instruction fields at random:
            # instruction = tf.random.shuffle(
            #     [traj["language_instruction"],
            #      traj["language_instruction_2"],
            #      traj["language_instruction_3"]])[0]
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

        # --------- traj_map (chunk_actions) ----------
        def chunk_actions(traj):
            T = tf.shape(traj["actions"])[0]
            idx = tf.range(action_chunk_size)[None] + tf.range(T)[:,None]
            idx = tf.minimum(idx, T-1)
            traj["actions"] = tf.gather(traj["actions"], idx)
            return traj
        dataset = dataset.map(chunk_actions, num_parallel_calls)

        # --------- filter_idle ----------
        def filter_idle(traj):
            first_half = traj["actions"][:action_chunk_size//2]
            if action_space==DroidActionSpace.JOINT_POSITION:
                return tf.reduce_any(tf.abs(first_half - first_half[:1]) > 1e-3)
            return tf.reduce_any(tf.abs(first_half) > 1e-3)
        dataset = dataset.filter(filter_idle)

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
    loader = DroidH5Dataset(
        h5_path="/home/yusenluo/pick_red_cube_20.h5",
        batch_size=32,
        action_space=DroidActionSpace.JOINT_VELOCITY,
        shuffle=True,
    )
    for batch in loader:
        print(batch["actions"].shape)   # (32,16,8)
        print(batch["observation"]["image"].shape)   # (32,256,256,3)
        print(batch["observation"]["wrist_image"].shape)   # (32,256,256,3)
        print(batch["observation"]["joint_position"].shape)   # (32,7)
        print(batch["observation"]["gripper_position"].shape)   # (32,1)
        print(batch["prompt"])
        # 保存前1-2帧的两路图像到 /home/yusenluo/sample_frames
        import os
        from pathlib import Path
        out_dir = Path("/home/yusenluo/sample_frames")
        out_dir.mkdir(parents=True, exist_ok=True)

        def save_image_fallback(image_array, output_path):
            try:
                import imageio.v3 as iio
                iio.imwrite(str(output_path), image_array)
                return str(output_path)
            except Exception:
                try:
                    from PIL import Image
                    Image.fromarray(image_array).save(str(output_path))
                    return str(output_path)
                except Exception:
                    np.save(str(Path(output_path).with_suffix(".npy")), image_array)
                    return str(Path(output_path).with_suffix(".npy"))

        num_to_save = min(2, batch["observation"]["image"].shape[0])
        for frame_index in range(num_to_save):
            ext_img = batch["observation"]["image"][frame_index]
            wrist_img = batch["observation"]["wrist_image"][frame_index]
            ext_path = out_dir / f"frame{frame_index}_exterior.png"
            wrist_path = out_dir / f"frame{frame_index}_wrist.png"
            saved_ext = save_image_fallback(ext_img, ext_path)
            saved_wrist = save_image_fallback(wrist_img, wrist_path)
            print(f"saved: {saved_ext}")
            print(f"saved: {saved_wrist}")
        break
  
