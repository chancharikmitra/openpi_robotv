"""
将单个 DROID 风格的 H5 数据集转换为 LeRobot 数据集的最小脚本。

参考 examples/libero/convert_libero_data_to_lerobot.py 的用法与 CLI。

用法:
uv run examples/droid/convert_droid_h5_to_lerobot.py --h5_path /path/to/data.h5

可选推送到 Hugging Face Hub:
uv run examples/droid/convert_droid_h5_to_lerobot.py --h5_path /path/to/data.h5 --push_to_hub

输出数据集将保存在 $HF_LEROBOT_HOME 下。
"""

import shutil
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tyro


def _safe_get_step_dataset(step_group: h5py.Group, key_primary: str, key_fallback: Optional[str] = None):
    if key_primary in step_group:
        return step_group[key_primary]
    if key_fallback is not None and key_fallback in step_group:
        return step_group[key_fallback]
    raise KeyError(f"step group missing datasets: '{key_primary}' and '{key_fallback}'")


def _infer_prompt_from_episode_name(ep_name: str, fixed_instruction: Optional[str]) -> str:
    if fixed_instruction is not None:
        return fixed_instruction
    base = ep_name.replace("-", " ").replace("_", " ")
    return " ".join(base.split()).lower()


def main(
    h5_path: str,
    *,
    repo_id: str = "your_hf_username/droid_h5",
    fps: int = 15,
    robot_type: str = "franka",
    fixed_instruction: Optional[str] = None,
    push_to_hub: bool = False,
):
    # 清理输出目录
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # 定义 LeRobot 数据集的特征
    # 本脚本假定图像大小为 (256,256,3)，与 H5 中存储一致
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=int(fps),
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # 7D 关节速度 + 1D 夹爪位置
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # 遍历 H5 文件: /<episode_name>/<step_id>
    h5_path_obj = Path(h5_path)
    with h5py.File(h5_path_obj, "r") as f:
        episode_names = [ep for ep in f.keys() if isinstance(f[ep], h5py.Group)]
        episode_names.sort()
        for ep in tqdm(episode_names, desc="Converting episodes"):
            grp = f[ep]
            step_keys = [s for s in grp.keys() if s.isdigit()]
            step_keys.sort(key=lambda x: int(x))
            if len(step_keys) == 0:
                continue

            language_instruction = _infer_prompt_from_episode_name(ep, fixed_instruction)
            #language_instruction = "pick up green cube"
            for step_key in step_keys:
                sg = grp[step_key]

                # 图像: 与内部 Torch 数据集保持一致，使用 rgb_right 作为外部左相机，rgb_wrist 作为腕部相机
                # 允许不同命名: rgb_left/rgb_right/rgb_wrist 或 image_left/image_right/image_wrist
                if "rgb_right" in sg:
                    ext_left = np.asarray(sg["rgb_right"])  # (256,256,3) uint8
                elif "image_right" in sg:
                    ext_left = np.asarray(sg["image_right"])  # 兼容别名
                elif "rgb_left" in sg:
                    # 如果只有 left，仍按外部左相机存储
                    ext_left = np.asarray(sg["rgb_left"])  # (256,256,3) uint8
                else:
                    raise KeyError(f"step {ep}/{step_key} missing exterior image ('rgb_right' or 'rgb_left')")

                # exterior_image_2_left: 使用 rgb_left（若无则 image_left，再无则沿用 ext_left 防止下游 KeyError）
                if "rgb_left" in sg:
                    ext2_left = np.asarray(sg["rgb_left"])  # (256,256,3) uint8
                elif "image_left" in sg:
                    ext2_left = np.asarray(sg["image_left"])  # 兼容别名
                else:
                    ext2_left = ext_left

                if "rgb_wrist" in sg:
                    wrist = np.asarray(sg["rgb_wrist"])  # (256,256,3) uint8
                elif "image_wrist" in sg:
                    wrist = np.asarray(sg["image_wrist"])  # 兼容别名
                else:
                    raise KeyError(f"step {ep}/{step_key} missing wrist image ('rgb_wrist')")

                # 观测 state: obs[0:7] = 关节位置, obs[7] = 夹爪位置
                obs_ds = _safe_get_step_dataset(sg, "obs")
                obs = np.asarray(obs_ds, dtype=np.float32)
                if obs.shape[-1] < 8:
                    raise ValueError(f"obs shape should be (8,), got {obs.shape} at {ep}/{step_key}")
                joint_position = obs[:7].astype(np.float32)
                gripper_position = np.asarray([obs[7]], dtype=np.float32)

                # 动作: 优先 joint velocity + gripper position
                act_vel_ds = _safe_get_step_dataset(sg, "act_vel", "action_vel")
                act_vel = np.asarray(act_vel_ds, dtype=np.float32)
                if act_vel.shape[-1] < 8:
                    raise ValueError(f"act_vel shape should be (8,), got {act_vel.shape} at {ep}/{step_key}")
                joint_velocity = act_vel[:7].astype(np.float32)
                gripper_position_action = np.asarray([act_vel[7]], dtype=np.float32)

                actions = np.concatenate([joint_velocity, gripper_position_action], axis=0).astype(np.float32)

                dataset.add_frame(
                    {
                        "exterior_image_1_left": ext_left,
                        "exterior_image_2_left": ext2_left,
                        "wrist_image_left": wrist,
                        "joint_position": joint_position,
                        "gripper_position": gripper_position,
                        "actions": actions,
                        "task": language_instruction,
                    }
                )

            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)


