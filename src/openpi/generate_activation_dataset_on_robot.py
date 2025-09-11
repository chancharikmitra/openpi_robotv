import jax, jax.numpy as jnp, numpy as np, flax.linen as nn
import os
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
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
# ---------------------------------- 保存配置 ----------------------------------
# 如果希望输出到不同路径，可修改此处
ATTN_H5_PATH = "/home/yusenluo/on_robot_activation/pick_red_cube.h5" #"wipe_eval_attention_last_token_single_action_negative.h5"
# 最多处理多少个 episode（跨所有 task 总计）
MAX_EPISODES = 20
USE_KEYFRAME = False
# from tasks import Pick_training_tasks
from openpi.llm_instruction_verb_filter import instruction_matches_prompt
LLM_PROMPT_KEY = "pick_place"
# 若只想在已生成的注意力文件上追加动作标签而不重算激活，请置 True
APPEND_ACTION_LABELS_ONLY = False
# 1) 取模型定义 & 权重 ----------------------------------------------------------
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

if not APPEND_ACTION_LABELS_ONLY:
    config = config.get_config("pi0_fast_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
    # 确保归一化等资产已下载（包含 droid/norm_stats.json）
    download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid/assets")

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
    """适配新H5结构：顶层为 episode 组，每个 episode 下有数字帧子组。

    每帧包含：
      - obs: (8,) float64 → 前7为 joint_position, 最后1为 gripper_position
      - action: (8,) float64 → 前7为 joint_velocity, 最后1为 gripper
      - rgb_left / rgb_right / rgb_wrist: (256,256,3) uint8

    返回结构仍为 {task_name: {ep_idx: {observations, actions, action_dict_list}}}
    其中 task_name 由 prompt 推断（优先组属性/数据集，其次组名）。
    """

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

            # 帧子组（数字字符串）
            steps = sorted([s for s in grp.keys() if s.isdigit()], key=lambda x: int(x))
            if not steps:
                continue

            # prompt 作为 task_name 分组键；若设置了环境变量 FORCED_PROMPT，则统一覆盖
            # prompt_text = extract_instruction_from_group(episode_name, grp)
            prompt_text = "pick red cube"
            # forced_prompt = os.environ.get("FORCED_PROMPT", "").strip()
            # if forced_prompt:
            #     prompt_text = forced_prompt
            task_name = prompt_text

            episodes: dict[int, dict[str, list]] = data.get(task_name, {})  # 兼容多 episode 同一 task

            obs_list: list[dict] = []
            act_list: list[np.ndarray] = []
            action_dict_list: list[dict] = []

            for s in steps:
                sg = grp[s]
                req = all(k in sg for k in ("obs", "action", "rgb_left", "rgb_wrist"))
                if not req:
                    continue
                obs = np.asarray(sg["obs"]).astype(np.float32)          # (8,)
                act = np.asarray(sg["action"]).astype(np.float32)       # (8,)
                img_left  = np.asarray(sg["rgb_left"])                  # (256,256,3)
                img_wrist = np.asarray(sg["rgb_wrist"])                 # (256,256,3)

                # 统一dtype与形状，保证后续拼接一致：
                jp = np.asarray(obs[:7], dtype=np.float32)      # (7,)
                gp = np.asarray(obs[7:8], dtype=np.float32)     # (1,)
                jv = np.asarray(act[:7], dtype=np.float32)      # (7,)
                ga = np.asarray(act[7], dtype=np.float32)       # 标量

                obs_list.append({
                    "observation/exterior_image_1_left": img_left,
                    "observation/wrist_image_left":     img_wrist,
                    "observation/joint_position":       jp,
                    "observation/gripper_position":     gp,
                    "prompt": prompt_text,
                })
                act_list.append(np.concatenate([jv, [ga]], dtype=np.float32))
                action_dict_list.append({
                    # 该数据集未提供动作空间下的关节位置，退化为当前观测位置
                    "act_joint_pos": jp,
                    "act_joint_vel": jv,
                    "act_gripper_pos": np.array([ga], dtype=np.float32),
                })

            if not obs_list:
                continue

            # 该 task_name 下新建一个顺序编号的 episode
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

# 仅验证解析：设置环境变量 EXTRACT_OBS_ONLY=1 将只运行解析并打印统计后退出
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

# 用法
h5_path = "/home/yusenluo/robotv_dataset/pick_red_cube_20.h5"
dataset = extract_observations(h5_path, max_episodes=MAX_EPISODES)

# 遍历并推理，同时打印当前进度：
file_mode = "a" if APPEND_ACTION_LABELS_ONLY else "w"
with h5py.File(ATTN_H5_PATH, file_mode) as h5_out:  # 在退出时自动 flush & close
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
            # ----------② 只枚举 key_idcs（带进度条） ----------
            for frame_idx in tqdm(key_idcs, desc=f"{task_name} ep{ep_idx:03d}", total=len(key_idcs)):
                # frame_idx = 原始索引 (0‑based)
                obs = ep_data["observations"][frame_idx]
                act = ep_data["actions"][frame_idx]
                true_frame_idx = frame_idx + 1                # 仍保持 1‑based 路径编号
                # HDF5 路径： /task/episode_xxx/frame_xxxx/
                grp_path = f"{task_name}/episode_{ep_idx:03d}/frame_{frame_idx:04d}"

                # 仅在已存在的分组上追加动作标签；不重算注意力
                if APPEND_ACTION_LABELS_ONLY:
                    if grp_path not in h5_out:
                        continue  # 仅补齐已有样本，避免创建新帧组
                    grp = h5_out[grp_path]
                    action_arr = np.asarray(act, dtype=np.float32)
                    if "action_label" in grp:
                        del grp["action_label"]
                    grp.create_dataset(
                        "action_label",
                        data=action_arr,
                        compression="gzip",
                    )

                    # jp = np.asarray(ep_data["action_dict_list"][frame_idx]["act_joint_pos"], dtype=np.float32).reshape(-1)
                    # jv = np.asarray(ep_data["action_dict_list"][frame_idx]["act_joint_vel"], dtype=np.float32).reshape(-1)
                    # gp = np.asarray(ep_data["action_dict_list"][frame_idx]["act_gripper_pos"], dtype=np.float32).reshape(-1)
                    # for key, arr in (("joint_position", jp), ("joint_velocity", jv), ("gripper_position", gp)):
                    #     if key in grp:
                    #         del grp[key]
                    #     grp.create_dataset(key, data=arr, compression="gzip")
                    # continue

                outputs, attention_outputs = policy.infer(obs, return_attention_heads=True, return_attention_probs=True)
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

                # 可选：保存前缀阶段的注意力概率（每层每头对序列位置的分布）
                # prefill_probs = attention_outputs.get("llm_attn_probs_prefill") #(18, 1, 8, 1018)
                # print("prefill_probs.shape:", prefill_probs.shape)
                # if prefill_probs is not None:
                #     prefill_probs = np.asarray(prefill_probs, dtype=np.float32)
                #     if "attn_probs_prefill" in grp:
                #         del grp["attn_probs_prefill"]
                #     grp.create_dataset(
                #         "attn_probs_prefill",
                #         data=prefill_probs,
                #         compression="gzip",
                #     )

                # 可选：保存解码阶段（动作 token 序列）的注意力概率轨迹
                # decode_probs = attention_outputs.get("llm_attn_probs_decode") #(1, 256, 18, 8, 1274)
                # print("decode_probs.shape:", decode_probs.shape)
                # if decode_probs is not None:
                #     decode_probs = np.asarray(decode_probs, dtype=np.float32)
                #     if "attn_probs_decode" in grp:
                #         del grp["attn_probs_decode"]
                #     grp.create_dataset(
                #         "attn_probs_decode",
                #         data=decode_probs,
                #         compression="gzip",
                #     )

                # 记录元信息方便追溯
                grp.attrs["full_llm_shape"] = full_attn.shape
                grp.attrs["last_token_idx"] = int(attention_outputs["last_token_idx"])  # type: ignore[arg-type]

                # 写入动作标签，并改为独立的关节/夹爪字段（不再保存 pi_droid_action）
                action_arr = np.asarray(act, dtype=np.float32)
                if "action_label" in grp:
                    del grp["action_label"]
                grp.create_dataset("action_label", data=action_arr, compression="gzip")


                # jp = np.asarray(ep_data["action_dict_list"][frame_idx]["act_joint_pos"], dtype=np.float32).reshape(-1)
                # jv = np.asarray(ep_data["action_dict_list"][frame_idx]["act_joint_vel"], dtype=np.float32).reshape(-1)
                # gp = np.asarray(ep_data["action_dict_list"][frame_idx]["act_gripper_pos"], dtype=np.float32).reshape(-1)
                # for key, arr in (("joint_position", jp), ("joint_velocity", jv), ("gripper_position", gp)):
                #     if key in grp:
                #         del grp[key]
                #     grp.create_dataset(key, data=arr, compression="gzip")

            # —— 一个 episode 写完 ——
            ep_counter += 1
            if ep_counter % 50 == 0:
                print("has infereced :", ep_counter)
                h5_out.flush()  # 定期 flush，减少因作业中断造成的数据丢失

        # —— 一个 task 写完 ——
        print(f"Finished task {task_name} – {len(eps)} episodes processed.")
        h5_out.flush()
print("done")
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