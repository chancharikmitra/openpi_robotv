import h5py, tensorflow as tf, numpy as np
from enum import Enum, auto

class DroidActionSpace(Enum):
    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()

# ---------------- episode generator 保持不变 ----------------
def _episode_generator(h5_path):
    name_map = {"view_0":"exterior_image_1_left",
                "view_1":"exterior_image_2_left",
                "view_2":"wrist_image_left"}
    with h5py.File(h5_path, "r") as f:
        for instr in f:
            grp = f[instr]
            ep_ids = sorted({int(x.split("_")[1]) for x in grp if x.startswith("ep_")})
            for k in ep_ids:
                pre = f"ep_{k}_"
                T   = grp[pre+"joint_positions"].shape[0]
                yield {
                    "observation":{
                        "joint_position" : grp[pre+"joint_positions"][:].astype(np.float32),
                        "gripper_position": grp[pre+"gripper_positions"][:].astype(np.float32),
                        name_map["view_0"]: grp[pre+"view_0"][:],
                        name_map["view_1"]: grp[pre+"view_1"][:],
                        name_map["view_2"]: grp[pre+"view_2"][:],
                    },
                    "action_dict":{
                        "joint_position" : grp[pre+"act_joint_pos"][:].astype(np.float32),
                        "joint_velocity" : grp[pre+"act_joint_vel"][:].astype(np.float32),
                        "gripper_position": grp[pre+"act_gripper_pos"][:].astype(np.float32),
                    },
                    "language_instruction"  : np.array([instr.encode()]*T),
                    "language_instruction_2": np.array([b""]*T),
                    "language_instruction_3": np.array([b""]*T),
                    "traj_metadata":{
                        "episode_metadata":{
                            "file_path": np.array([f"dummy_success_{k}".encode()]*T)
                        }
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
        action_space : DroidActionSpace=DroidActionSpace.JOINT_POSITION,
        shuffle_buffer_size : int = 256,
        num_parallel_calls  : int = tf.data.AUTOTUNE,
    ):
        tf.config.set_visible_devices([], "GPU")

        # 1. 生成 tf.data.Dataset（trajs）
        output_sig = {
            "observation":{
                "joint_position"       : tf.TensorSpec((None,7), tf.float32),
                "gripper_position"     : tf.TensorSpec((None,1), tf.float32),
                "exterior_image_1_left": tf.TensorSpec((None,180,320,3), tf.uint8),
                "exterior_image_2_left": tf.TensorSpec((None,180,320,3), tf.uint8),
                "wrist_image_left"     : tf.TensorSpec((None,180,320,3), tf.uint8),
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

        # ====== ↓ 原 DroidRldsDataset 流程，逐行沿用，但把 dlimp 调用改成 tf.data 原生 ======

        if shuffle:
            dataset = dataset.shuffle(buffer_size=20)

        # 过滤成功轨迹
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*")
        )

        dataset = dataset.repeat()                         # 无限循环数据

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
        h5_path="/scr2/yusenluo/openpi/droid_pick_train_positive_new.h5",
        batch_size=32,
        action_space=DroidActionSpace.JOINT_POSITION,
        shuffle=True,
    )
    for batch in loader:
        print(batch["actions"].shape)   # (32,16,8)
        print(batch["observation"]["image"].shape)   # (32,180,320,3)
        print(batch["observation"]["wrist_image"].shape)   # (32,180,320,3)
        print(batch["observation"]["joint_position"].shape)   # (32,7)
        print(batch["observation"]["gripper_position"].shape)   # (32,1)
        print(batch["prompt"])
        break
  
