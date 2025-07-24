import h5py, numpy as np, matplotlib.pyplot as plt, re
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
from tqdm import tqdm
from matplotlib import patches

# ---------- 路径配置 ----------
OBS_H5   = "/scr2/yusenluo/openpi/eval/droid_pick_eval.h5"  # 原图像文件
OUT_DIR  = Path("keyframe_gifs")
OUT_DIR.mkdir(exist_ok=True)

MAX_EPISODES = 10      # 导出的 episode 数
FPS           = 6
INT_MS        = 1000 // FPS

# ---------- 判断关键帧 ----------

def keyframe_flags(joint_pos: np.ndarray, gripper_pos: np.ndarray, dt: float = 1/15,
                   th_vel: float = 0.03) -> np.ndarray:
    """返回 flags 数组：0=非关键, 1=gripper flip, 2=arm static"""
    F = len(joint_pos)
    flags = np.zeros(F, dtype=np.int8)

    # ① gripper flip
    flip = np.where(gripper_pos[1:] != gripper_pos[:-1])[0] + 1
    flags[flip] = 1

    # ② arm static (速度阈值)
    dq_norm = np.linalg.norm(np.diff(joint_pos, axis=0)/dt, axis=1)
    static  = np.where(dq_norm < th_vel)[0] + 1
    flags[static] = 2  # 若与 flip 重叠，后者覆盖

    flags[0] = flags[-1] = 3   # 首尾帧标记为 3
    return flags  # (F,)

# ---------- 辅助读取 ----------

def get_rgb(obs_file: h5py.File, task: str, ep: str) -> np.ndarray:
    m = re.match(r"episode_(\d+)", ep)
    ep_short = f"ep_{int(m.group(1))}" if m else ep.replace("episode", "ep")
    ds = obs_file[f"{task}/{ep_short}_view_0"]
    return ds[()]

def load_signals(obs_file: h5py.File, task: str, ep: str):
    m = re.match(r"episode_(\d+)", ep)
    ep_short = f"ep_{int(m.group(1))}" if m else ep.replace("episode", "ep")
    jp = obs_file[f"{task}/{ep_short}_joint_positions"][:]     # (F,7)
    gp = obs_file[f"{task}/{ep_short}_gripper_positions"][:]
    return jp, gp

# ---------- 动画 ----------

def make_animation(rgb: np.ndarray, flags: np.ndarray, save_path: Path, fps: int = 6, interval: int = 166):
    F = len(rgb)
    colors = {0:None, 1:"red", 2:"blue", 3:"green"}
    labels = {1:"Flip", 2:"Static", 3:"Start/End"}

    fig, ax = plt.subplots(figsize=(5,5))
    ax.axis("off")
    im = ax.imshow(rgb[0])
    # 预创建一个无填充的矩形框用于高亮
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             linewidth=4, edgecolor="none", facecolor="none")
    ax.add_patch(rect)
    txt = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", va="top",
                  color="white", fontsize=12, weight="bold", bbox=dict(facecolor="black",alpha=.6,pad=3))

    def update(i):
        im.set_data(rgb[i])
        flag = int(flags[i])
        if flag:
            txt.set_text(labels.get(flag, "Key"))
            txt.set_visible(True)
            if colors[flag]:
                rect.set_edgecolor(colors[flag])
                rect.set_visible(True)
        else:
            txt.set_visible(False)
            rect.set_visible(False)
        ax.set_title(f"Frame {i+1}/{F}")
        return [im, txt, rect]

    ani = FuncAnimation(fig, update, frames=F, interval=interval, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=-1)
    ani.save(save_path.with_suffix(".mp4"), writer=writer, dpi=120)
    plt.close(fig)

# ---------- 主流程 ----------

# ---------- 重新遍历：按 ep_ 前缀分组 ----------

done = 0
with h5py.File(OBS_H5, "r") as f_obs:
    for task in sorted(f_obs.keys()):
        task_grp = f_obs[task]

        # 先收集该 task 下所有 ep_ 前缀
        prefixes: set[str] = set()
        for ds_name in task_grp.keys():
            m = re.match(r"(ep_\d+)_", ds_name)
            if m:
                prefixes.add(m.group(1))

        for ep_prefix in sorted(prefixes):
            if done >= MAX_EPISODES:
                break

            ep = f"episode_{int(ep_prefix.split('_')[1]):03d}"  # 统一命名

            try:
                rgb = get_rgb(f_obs, task, ep)            # (F,H,W,3)
                jp, gp = load_signals(f_obs, task, ep)
            except KeyError:
                continue

            flags = keyframe_flags(jp, gp)                # (F,)
            out_file = OUT_DIR / f"{task.replace(' ','_')}_{ep}"
            make_animation(rgb, flags, out_file, fps=FPS, interval=INT_MS)
            print(f"🎞  Saved {out_file}.mp4")
            done += 1

        if done >= MAX_EPISODES:
            break
print(f"✅  Done. Generated {done} episode animations in {OUT_DIR}") 