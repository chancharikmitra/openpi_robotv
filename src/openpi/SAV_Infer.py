import jax, jax.numpy as jnp, numpy as np, flax.linen as nn
from flax.linen.attention import MultiHeadDotProductAttention as MHA
from openpi.training import config as pi_cfg
from openpi.models import model as pi_model
from openpi.shared import download
# ------------------------- 依赖 -------------------------
import h5py  # type: ignore
import numpy as np
import re
import random
import h5py  # type: ignore
# ---------------------------------- 保存配置 ----------------------------------
# 如果希望输出到不同路径，可修改此处
ATTN_H5_PATH = "pick_train_attention_last_token_keyframe_new.h5" #"wipe_eval_attention_last_token_single_action_negative.h5"
# 最多处理多少个 episode（跨所有 task 总计）
MAX_EPISODES = 250
USE_KEYFRAME = True
# from tasks import Pick_training_tasks
from openpi.llm_instruction_verb_filter import instruction_matches_prompt
LLM_PROMPT_KEY = "pick_place"
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


def extract_key_idcs(
    joint_pos: np.ndarray,           # (F,7)
    gripper_pos: np.ndarray,         # (F,)  (或 (F,1))
    actions: np.ndarray,             # (F,7)
    dt: float = 1/15,                # Droid default 15 Hz
    th_vel: float = 0.03,            # rad/s  “静止”阈
    th_act: float = 0.3,             # 动作峰值阈
    chunk_size: int = 10,
) -> np.ndarray:
    F = len(joint_pos)
    key = np.zeros(F, dtype=bool)
    # diff = np.diff(gripper_pos.squeeze()).astype(float)   # 先去掉多余维度

    # # ------ 可视化 gripper 速度分布 ------
    # try:
    #     import matplotlib.pyplot as plt  # 本段仅用于调试，可根据需要移除

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
    #     # 在无图形后端或 matplotlib 缺失的环境下容错
    #     print(f"[WARN] 无法绘制直方图: {e}")
    # ---- R2: 抓手翻转 ----
    flip = np.where(gripper_pos[1:] != gripper_pos[:-1])[0] + 1
    key[flip] = True

    # ---- R1: 手臂静止 ----
    dq_norm = np.linalg.norm(np.diff(joint_pos, axis=0)/dt, axis=1)
    static = np.where(dq_norm < th_vel)[0] + 1
    key[static] = True

    # ---- R3: 每 10 帧动作峰值 ----
    # for c in range(0, F, chunk_size):
    #     seg = actions[c:c+chunk_size]
    #     if seg.size == 0: continue
    #     idx = np.argmax(np.linalg.norm(seg, axis=1))
    #     if np.linalg.norm(seg[idx]) > th_act:
    #         key[c+idx] = True

    key[0] = key[-1] = True          # 始末帧必保留
    return np.where(key)[0]          # 升序索引



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

# 用法
h5_path = "/scr2/yusenluo/openpi/SAV_training/pick_train_new.h5" #"/scr2/yusenluo/openpi/droid_LLM_pick_eval_negative.h5" #"/scr2/yusenluo/openpi/SAV_training/pick_train.h5" 
dataset = extract_observations(h5_path, max_episodes=MAX_EPISODES)

# 遍历并推理，同时打印当前进度：
with h5py.File(ATTN_H5_PATH, "w") as h5_out:  # 在退出时自动 flush & close
    ep_counter = 0  # 统计已写入的 episode 数
    for task_name, eps in dataset.items():
        for ep_idx, ep_data in eps.items():
            total_frames = len(ep_data["observations"])

            # for frame_idx, (obs, act) in enumerate(
            #     zip(ep_data["observations"], ep_data["actions"]), start=1
            # ):
            if USE_KEYFRAME:
                jp  = np.stack([obs["observation/joint_position"]      for obs in ep_data["observations"]])
                gp  = np.stack([obs["observation/gripper_position"]    for obs in ep_data["observations"]])
                acts= np.stack(ep_data["actions"])
                key_idcs = extract_key_idcs(jp, gp, acts)          # ndarray
            else:
                key_idcs = np.arange(total_frames) # 全帧

            print(f"Task {task_name} Episode {ep_idx}: keep {len(key_idcs)}/{total_frames} key‑frames")
            # ----------② 只枚举 key_idcs ----------
            for frame_idx in key_idcs:                         # frame_idx = 原始索引 (0‑based)
                obs = ep_data["observations"][frame_idx]
                act = ep_data["actions"][frame_idx]
                true_frame_idx = frame_idx + 1                # 仍保持 1‑based 路径编号
                outputs, attention_outputs = policy.infer(obs, return_attention_heads=True)
                # HDF5 路径： /task/episode_xxx/frame_xxxx/
                grp_path = f"{task_name}/episode_{ep_idx:03d}/frame_{frame_idx:04d}"
                grp = h5_out.require_group(grp_path)  # 创建层级

                # 只保留最后一个 token 的注意力：(layer, head, dim) = (18, 8, 256)
                full_attn = attention_outputs["llm_activations"][0]  # (18, 1, 1018, 8, 256)
                last_token_attn = full_attn[:, 0, -1, :, :]  # (18, 8, 256)
                # JAX 默认可能是 bfloat16，h5py 不支持，保存会变成 "|V2" 字节串类型 → 读取报错
                # 因此强制转换为 float32 后再保存
                last_token_attn = np.asarray(last_token_attn, dtype=np.float32)
                assert last_token_attn.shape == (18, 8, 256)
                if "last_token_attn" in grp:
                    del grp["last_token_attn"]
                grp.create_dataset(
                    "last_token_attn",
                    data=last_token_attn,
                    compression="gzip",
                )

                # 记录元信息方便追溯
                grp.attrs["full_llm_shape"] = full_attn.shape
                grp.attrs["last_token_idx"] = int(attention_outputs["last_token_idx"])  # type: ignore[arg-type]

            # —— 一个 episode 写完 ——
            ep_counter += 1
            if ep_counter % 50 == 0:
                print("has infereced :", ep_counter)
                h5_out.flush()  # 定期 flush，减少因作业中断造成的数据丢失

        # —— 一个 task 写完 ——
        print(f"Finished task {task_name} – {len(eps)} episodes processed.")
        h5_out.flush()

# 一个 episode 完成

#print(policy.infer)
# outputs, attention_outputs = policy.infer(obs_list[0], return_attention_heads=True)
# print(obs_list[0]["prompt"])
# action_chunk = outputs["actions"]
# print("action_chunk.shape:", action_chunk.shape)
#print(action_chunk)
# print(attention_outputs)
# print("attention_outputs['llm_activations'].shape:", attention_outputs['llm_activations'].shape)
# print("attention_outputs['last_token_idx']: ", attention_outputs['last_token_idx'])

# outputs, attention_outputs = policy.infer(obs_list[1], return_attention_heads=True)
# print(obs_list[1]["prompt"])
# action_chunk = outputs["actions"]
# print("action_chunk.shape:", action_chunk.shape)
#print(action_chunk)
# print(attention_outputs)

# print("attention_outputs['llm_activations'].shape:", attention_outputs['llm_activations'].shape)
# print("attention_outputs['last_token_idx']: ", attention_outputs['last_token_idx'])

# outputs, attention_outputs = policy.infer(obs_list[2], return_attention_heads=True)
# print(obs_list[2]["prompt"])
# action_chunk = outputs["actions"]
# print("action_chunk.shape:", action_chunk.shape)
# print(attention_outputs)

# print("attention_outputs['llm_activations'].shape:", attention_outputs['llm_activations'].shape)
# print("attention_outputs['last_token_idx']: ", attention_outputs['last_token_idx'])