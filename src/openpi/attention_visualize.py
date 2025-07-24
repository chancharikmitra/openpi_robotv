import h5py, numpy as np, matplotlib.pyplot as plt, re
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
from tqdm import tqdm

OBS_H5   = "/scr2/yusenluo/openpi/eval/droid_wipe_eval.h5"
ATTN_H5  = "wipe_eval_attention_last_token.h5"
OUT_DIR  = Path("attention_gifs/wipe")
OUT_DIR.mkdir(exist_ok=True)

MAX_EPISODES = 3          # ← 要导出的前 N 个 episode
FPS           = 10          # ← 导出帧率
INT_MS        = 1000 // FPS

# ---------- 帮助函数 ----------
def get_rgb(obs_file: h5py.File, task: str, ep: str) -> np.ndarray:
    """
    返回 (F,H,W,3) uint8 的 primary image
    假设 obs 文件里 still 是 ep_X_view_0 这种一整段数组
    """
    # 将 episode_0010 → ep_10
    m = re.match(r"episode_(\d+)", ep)
    ep_short = f"ep_{int(m.group(1))}" if m else ep.replace("episode", "ep")
    ds = obs_file[f"{task}/{ep_short}_view_0"]
    return ds[()]

def get_attn(attn_grp: h5py.Group) -> np.ndarray:
    """把 episode_xxx group → (F,18,8,256) float32"""
    frames = [np.asarray(attn_grp[k]["last_token_attn"], dtype=np.float32)
              for k in sorted(attn_grp.keys())]
    return np.stack(frames, axis=0)

def make_animation(rgb: np.ndarray, heat: np.ndarray,
                   save_path: Path, fps: int = 6, interval: int = 166):
    """heat 已是 (F,18,8)"""
    F, L, H = heat.shape
    vmin, vmax = heat.min(), heat.max()

    fig, (ax_img, ax_heat) = plt.subplots(
        1, 2, figsize=(7, 9), gridspec_kw={'width_ratios':[2,1]}
    )
    ax_img.axis("off")
    im_left  = ax_img.imshow(rgb[0])
    im_right = ax_heat.imshow(heat[0], cmap='RdYlBu_r',
                              vmin=vmin, vmax=vmax, origin='lower')
    ax_heat.set_xticks(range(H));  ax_heat.set_yticks(range(L))
    fig.colorbar(im_right, ax=ax_heat, fraction=0.046, pad=0.04)
    ax_heat.set_xlabel("Head"); ax_heat.set_ylabel("Layer")

    def update(i):
        im_left.set_data(rgb[i]); im_right.set_data(heat[i])
        ax_img.set_title(f"Frame {i+1}/{F}")
        return [im_left, im_right]

    ani = FuncAnimation(fig, update, frames=F, interval=interval, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=-1)
    ani.save(save_path.with_suffix(".mp4"), writer=writer, dpi=120)
    plt.close(fig)

# ---------- 主循环 ----------
done = 0
with h5py.File(OBS_H5, "r") as f_obs, h5py.File(ATTN_H5, "r") as f_attn:

    # 以字典序遍历 /task/episode_xxx
    for task in sorted(f_attn.keys()):
        task_grp = f_attn[task]
        for ep in sorted(task_grp.keys()):          # ep = episode_000
            if done >= MAX_EPISODES:
                break

            attn_grp = task_grp[ep]
            try:
                rgb   = get_rgb(f_obs, task, ep)                # (F,H,W,3)
            except KeyError:
                print(f"⚠ 找不到 {task}/{ep} 的图像，跳过")
                continue

            attn  = get_attn(attn_grp)                          # (F,18,8,256)
            # 压缩 256-D → L2 范数
            heat  = np.linalg.norm(attn, axis=-1)               # (F,18,8)

            out_file = OUT_DIR / f"{task.replace(' ','_')}_{ep}"
            make_animation(rgb, heat, out_file, fps=FPS, interval=INT_MS)
            print(f"🎞  Saved {out_file}.mp4")

            done += 1
        if done >= MAX_EPISODES:
            break

print(f"✅  Done. Generated {done} episode animations in {OUT_DIR}")
