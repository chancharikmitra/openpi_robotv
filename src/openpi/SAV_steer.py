import jax, jax.numpy as jnp, numpy as np, flax.linen as nn
from flax.linen.attention import MultiHeadDotProductAttention as MHA
from openpi.training import config as pi_cfg
from openpi.models import model as pi_model
from openpi.shared import download
import pickle
from openpi.models import model as _model
# ------------------------- 依赖 -------------------------
import h5py  # type: ignore
import numpy as np
import re
import random
import h5py  # type: ignore
# ---------------------------------- 保存配置 ----------------------------------
# 如果希望输出到不同路径，可修改此处
ATTN_H5_PATH = "steer_debug.h5" #"wipe_eval_attention_last_token_single_action_negative.h5"
# 最多处理多少个 episode（跨所有 task 总计）
MAX_EPISODES = 120
# from tasks import Pick_training_tasks
from openpi.llm_instruction_verb_filter import instruction_matches_prompt
LLM_PROMPT_KEY = "wipe"
# 1) 取模型定义 & 权重 ----------------------------------------------------------
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)      # Pi0FAST Module 实例（没有权重）

# 2) 构造 Observation（带 batch 维）-------------------------------------------
from openpi.models.model import Observation

def extract_observations(h5_path, max_episodes: int | None = None):
    """按 task->episode->frame 三层结构组织数据，并为每个 episode 按出现顺序重新编号。

    参数 max_episodes 用于限制总共提取的 episode 数量（跨任务累计）。
    """

    data: dict[str, dict[int, dict[str, list]]] = {}

    processed = 0

    with h5py.File(h5_path, "r") as f:
        for task_name in f:  # 第一层：任务
            # if not instruction_matches_prompt(task_name, prompt_key=LLM_PROMPT_KEY):
            #     continue
            print("task_name:",task_name)
            grp = f[task_name]  # type: ignore[index]

            # 收集所有 ep_ 前缀
            prefixes: set[str] = set()
            for ds_name in grp.keys():  # type: ignore[attr-defined]
                m = re.match(r"(ep_\d+)_", ds_name)
                if m:
                    prefixes.add(m.group(1))

            if not prefixes:
                continue  # 该任务没有 episode，跳过

            episodes: dict[int, dict[str, list]] = {}

            for idx, ep_prefix in enumerate(sorted(prefixes), start=1):  # 重新编号 1,2,3...
                required = [
                    f"{ep_prefix}_actions",
                    f"{ep_prefix}_gripper_positions",
                    f"{ep_prefix}_joint_positions",
                    f"{ep_prefix}_view_0",
                    f"{ep_prefix}_view_2",
                ]

                if not all(k in grp for k in required):  # type: ignore[operator]
                    continue  # 缺字段跳过

                actions_arr   = grp[f"{ep_prefix}_actions"][:]  # type: ignore[index]
                gripper_arr   = grp[f"{ep_prefix}_gripper_positions"][:]  # type: ignore[index]
                joint_arr     = grp[f"{ep_prefix}_joint_positions"][:]  # type: ignore[index]
                img_primary   = grp[f"{ep_prefix}_view_0"][:]  # type: ignore[index]
                img_wrist     = grp[f"{ep_prefix}_view_2"][:]  # type: ignore[index]
                obs_list: list[dict] = []
                act_list: list[np.ndarray] = []

                for frame_idx in range(actions_arr.shape[0]):  # type: ignore[attr-defined]
                    obs_list.append({
                        "observation/exterior_image_1_left": img_primary[frame_idx],  # type: ignore[index]
                        "observation/wrist_image_left":     img_wrist[frame_idx],    # type: ignore[index]
                        "observation/joint_position":       joint_arr[frame_idx],     # type: ignore[index]
                        "observation/gripper_position":     gripper_arr[frame_idx],   # type: ignore[index]
                        "prompt": task_name,
                    })
                    act_list.append(actions_arr[frame_idx])  # type: ignore[arg-type]

                episodes[idx] = {  # type: ignore[index]
                    "observations": obs_list,
                    "actions": act_list,
                }

                processed += 1

                # 如果达到上限，直接结束并返回
                if max_episodes is not None and processed >= max_episodes:
                    data[task_name] = episodes  # type: ignore[index]  # 保留已收集的本任务数据
                    return data

            if episodes:
                data[task_name] = episodes  # type: ignore[index]

    return data


target_shape = (18, 8, 256)

def calc_mean(h5_path):
    sum_arr = np.zeros(target_shape, dtype=np.float64)
    n_samples = 0

    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            nonlocal sum_arr, n_samples
            if isinstance(obj, h5py.Dataset) and obj.shape == target_shape \
            and np.issubdtype(obj.dtype, np.floating):
                data = obj[()]            # 读到内存；如需省内存可分块读取
                sum_arr += data
                n_samples += 1

        f.visititems(visit)

    if n_samples == 0:
        raise ValueError("未找到任何形状为 (18, 8, 256) 的浮点型数据集")

    mean_arr = sum_arr / n_samples   # 结果仍为 (18, 8, 256)

    print("样本数:", n_samples)
    return mean_arr
# ① 重新加载模型本体 --------------------------------------------------------
model = config.model.load(
    pi_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
)

# ---------- 读 steer 权重 ----------
mean_arr = calc_mean("/scr2/yusenluo/openpi_robotv/src/openpi/steer_train_wipe_top_20_heads_mean.h5")          # 见上文函数

# ---------- 遍历数据集 ----------
key = jax.random.key(0)
loss_list = []

dataset = extract_observations("/scr2/yusenluo/openpi/droid_LLM_wipe_eval_positive.h5", max_episodes=MAX_EPISODES)

for task_name, eps in dataset.items():
    for ep_idx, ep_data in eps.items():
        for obs, act in zip(ep_data["observations"], ep_data["actions"]):
            # ------------------ 转 Observation ------------------
            inp = policy._input_transform(obs)
            inp = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inp)
            observation = pi_model.Observation.from_dict(inp)

            # ------------------ 计算 loss -----------------------
            key, subkey = jax.random.split(key)
            loss = model.compute_loss(
                subkey, observation, act,
                return_attention_heads=False,
                delta_heads=mean_arr,
            )
            print("loss:", loss)
            loss_list.append(loss)
        print(f"finished ep {ep_idx}")

print("mean loss:", np.mean(loss_list))


pickle.dump(loss_list, open("loss_list.pkl", "wb"))
print("loss_list:", loss_list)
print("mean_loss:", np.mean(loss_list))
print("std_loss:", np.std(loss_list))
print("min_loss:", np.min(loss_list))
print("max_loss:", np.max(loss_list))

