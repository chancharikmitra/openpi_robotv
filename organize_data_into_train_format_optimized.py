import os
import random
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2


CAM_ID_MAP = {
    "34520144": "rgb_right",
    "15571257": "rgb_wrist",
    "37849686": "rgb_left",
}


def center_crop_and_resize(img_np, size=256):
    h, w, _ = img_np.shape
    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    img_cropped = img_np[top:top + crop_size, left:left + crop_size]
    img_resized = cv2.resize(img_cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return img_resized


def extract_mp4_frames(mp4_path):
    # Raw mp4 is a side-by-side stereo composite (e.g. 2560x720). Take the
    # left half only before center-cropping/resizing, otherwise the center
    # crop straddles the two views and the resulting image is unusable.
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {mp4_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        if w >= 2 * h:
            frame = frame[:, : w // 2]
        img_rgb = frame[:, :, ::-1]  # BGR to RGB
        frames.append(center_crop_and_resize(img_rgb))
    cap.release()
    return frames


def load_trajectory(trajectory_path):
    with h5py.File(trajectory_path, "r") as a:
        joint_pos = a['action']['joint_position'][()]
        joint_vel = a['action']['joint_velocity'][()]
        gripper_pos = a['action']['gripper_position'][()]

        act_pos = np.concatenate([joint_pos, gripper_pos[:, None]], axis=1)
        act_vel = np.concatenate([joint_vel, gripper_pos[:, None]], axis=1)

        joint_obs = a['observation']['robot_state']['joint_positions'][()]
        gripper_obs = a['observation']['robot_state']['gripper_position'][()]
        obs = np.concatenate([joint_obs, gripper_obs[:, None]], axis=1)

    return act_pos, act_vel, obs


def convert_demo_to_dict(demo_folder, traj_name):
    traj_path = os.path.join(demo_folder, "trajectory.h5")
    mp4_dir = os.path.join(demo_folder, "recordings", "MP4")

    act_pos, act_vel, obs = load_trajectory(traj_path)

    cam_frames = {}
    for mp4_path in glob(os.path.join(mp4_dir, "*.mp4")):
        cam_id = os.path.splitext(os.path.basename(mp4_path))[0]
        cam_name = CAM_ID_MAP.get(cam_id)
        if cam_name:
            cam_frames[cam_name] = extract_mp4_frames(mp4_path)

    if len(cam_frames) != len(CAM_ID_MAP):
        raise ValueError(f"Missing camera views in {demo_folder}")

    T = min(len(act_pos), *(len(v) for v in cam_frames.values()))
    result = {}

    for t in range(T):
        step = {
            "obs": obs[t],
            "act_pos": act_pos[t],
            "act_vel": act_vel[t],
        }
        for cam_key, frames in cam_frames.items():
            step[cam_key] = frames[t]
        result[str(t)] = step

    return traj_name, result


def find_success_demo_folders(root_input):
    demo_folders = []
    for dirpath, dirnames, filenames in os.walk(root_input):
        if "trajectory.h5" in filenames and "success" in dirpath:
            mp4_dir = os.path.join(dirpath, "recordings", "MP4")
            if os.path.isdir(mp4_dir):
                demo_folders.append(dirpath)
    return demo_folders


def make_traj_name(demo_path, root_input):
    rel_path = os.path.relpath(demo_path, root_input)
    return rel_path.replace(os.sep, "_")


def batch_convert_all_to_one_h5(root_input, output_file_path, max_workers=4, num_samples=None, seed=0):
    demo_folders = find_success_demo_folders(root_input)
    print(f"🔍 Found {len(demo_folders)} demos")

    if num_samples is not None and num_samples < len(demo_folders):
        rng = random.Random(seed)
        demo_folders = sorted(demo_folders)
        demo_folders = rng.sample(demo_folders, num_samples)
        print(f"🎲 Sampled {len(demo_folders)} demos (seed={seed})")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for demo_path in demo_folders:
            traj_name = make_traj_name(demo_path, root_input)
            futures.append(executor.submit(convert_demo_to_dict, demo_path, traj_name))

        for future in tqdm(futures, desc="Processing demos", unit="demo"):
            try:
                traj_name, demo_data = future.result()
                results.append((traj_name, demo_data))
            except Exception as e:
                print(f"❌ Error: {e}")

    print(f"💾 Writing to {output_file_path}")
    with h5py.File(output_file_path, "w") as h5file:
        for traj_name, traj_data in results:
            traj_group = h5file.create_group(traj_name)
            for t_str, step_data in traj_data.items():
                step = traj_group.create_group(t_str)
                for key, val in step_data.items():
                    step.create_dataset(key, data=val, dtype='uint8' if isinstance(val, np.ndarray) and val.dtype == np.uint8 else None)

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--root_input", required=True, help="Root folder of success demos")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel threads")
    parser.add_argument("--num_samples", type=int, default=None, help="Randomly sample N success episodes (default: use all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling")
    parser.add_argument("--output", type=str, default=None, help="Output H5 path (default: <root_input>.h5)")
    args = parser.parse_args()

    output_path = args.output if args.output is not None else args.root_input.rstrip("/") + ".h5"
    batch_convert_all_to_one_h5(
        args.root_input,
        output_path,
        max_workers=args.workers,
        num_samples=args.num_samples,
        seed=args.seed,
    )
