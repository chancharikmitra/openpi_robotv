import h5py
import numpy as np
from typing import SupportsIndex, List, Tuple, Dict


class DroidActionSpace:
    JOINT_POSITION = "JOINT_POSITION"
    JOINT_VELOCITY = "JOINT_VELOCITY"


class TorchDroidH5Dataset:
    """
    Map-style Dataset for DROID-like H5 (single file) used by TorchDataLoader.

    Expects H5 layout:
      /<episode_name>/<step_id>
        - obs: (8,) float32 -> [7 joint_position, 1 gripper_position]
        - action: (15,) float32 -> [7 joint_position, 7 joint_velocity, 1 gripper_position]
        - rgb_left, rgb_right, rgb_wrist: (256,256,3) uint8

    Emits per-step samples with keys aligned to openpi pipeline:
      - observation/exterior_image_1_left, observation/wrist_image_left
      - observation/joint_position, observation/gripper_position
      - actions: (action_horizon, A) where A=8 for JOINT_VELOCITY, A=8 for JOINT_POSITION
      - prompt: fixed instruction if provided, else inferred from episode name
    """

    def __init__(
        self,
        h5_path: str,
        action_horizon: int,
        *,
        action_space: str = DroidActionSpace.JOINT_VELOCITY,
        fixed_instruction: str | None = None,
    ) -> None:
        self._h5_path = h5_path
        self._H = int(action_horizon)
        self._action_space = action_space
        self._fixed_instruction = fixed_instruction

        # Build flat indices: list of (episode_name, step_index)
        self._indices: List[Tuple[str, int]] = []
        with h5py.File(self._h5_path, "r") as f:
            for ep in f.keys():
                grp = f[ep]
                if not isinstance(grp, h5py.Group):
                    continue
                steps = sorted([s for s in grp.keys() if s.isdigit()], key=lambda x: int(x))
                self._indices.extend([(ep, int(s)) for s in steps])

    def __len__(self) -> int:
        return len(self._indices)

    def _get_episode_arrays(self, f: h5py.File, ep: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grp = f[ep]
        steps = sorted([s for s in grp.keys() if s.isdigit()], key=lambda x: int(x))
        # stack obs and action into arrays for chunking
        obs = [np.asarray(grp[s]["obs"], dtype=np.float32) for s in steps]
        act = [np.asarray(grp[s]["action"], dtype=np.float32) for s in steps]
        obs_arr = np.stack(obs, axis=0)  # (T,8)
        act_arr = np.stack(act, axis=0)  # (T,15)
        # split
        joint_pos_obs = obs_arr[:, :7].astype(np.float32)
        gripper_pos_obs = obs_arr[:, 7:8].astype(np.float32)
        return joint_pos_obs, gripper_pos_obs, act_arr.astype(np.float32)

    def _infer_prompt(self, ep: str) -> str:
        if self._fixed_instruction is not None:
            return self._fixed_instruction
        base = ep.replace("-", " ").replace("_", " ")
        return " ".join(base.split()).lower()

    def __getitem__(self, index: SupportsIndex) -> Dict:
        ep, step_idx = self._indices[int(index)]
        with h5py.File(self._h5_path, "r") as f:
            grp = f[ep]
            step_key = str(step_idx)
            sg = grp[step_key]
            # images
            img_left = np.asarray(sg["rgb_right"])         # (256,256,3)
            wrist_img = np.asarray(sg["rgb_wrist"])       # (256,256,3)

            # obs arrays for this episode
            jp_obs, gp_obs, act_arr = self._get_episode_arrays(f, ep)  # (T,7), (T,1), (T,15)
            T = act_arr.shape[0]

            # action split for the whole episode
            jpos = act_arr[:, :7]
            jvel = act_arr[:, 7:14]
            gpos = act_arr[:, 14:15]

            # build chunk starting at step_idx
            horizon = self._H
            idx = np.arange(step_idx, step_idx + horizon)
            idx_clamped = np.minimum(idx, T - 1)

            if self._action_space == DroidActionSpace.JOINT_POSITION:
                # actions = [joint_position, gripper_position]
                chunk_pos = jpos[idx_clamped]  # (H,7)
                chunk_gp = gpos[idx_clamped]   # (H,1)
                actions = np.concatenate([chunk_pos, chunk_gp], axis=-1)
            else:
                # actions = [joint_velocity, gripper_position], tail pad velocity with zeros, keep last gpos
                chunk_vel = jvel[idx_clamped].copy()  # (H,7)
                over_mask = (idx >= T)
                if over_mask.any():
                    # zero-pad velocities for indices beyond T-1
                    chunk_vel[over_mask] = 0.0
                chunk_gp = gpos[idx_clamped]
                if (idx >= T).any():
                    # fill gpos beyond tail with last value
                    chunk_gp[over_mask] = gpos[-1]
                actions = np.concatenate([chunk_vel, chunk_gp], axis=-1)  # (H,8)
                assert actions.shape == (horizon, 8)

            sample = {
                "observation/exterior_image_1_left": img_left,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": jp_obs[step_idx],
                "observation/gripper_position": gp_obs[step_idx],
                "actions": actions.astype(np.float32),
                "prompt": self._infer_prompt(ep),
            }
            return sample



if __name__ == "__main__":
    h5_path = "/scr2/yusenluo/openpi_robotv/robotv_dataset/pick_red_cube_20.h5"
    action_horizon = 16
    action_space = DroidActionSpace.JOINT_VELOCITY
    fixed_instruction = "pick up red mug"
    num_samples = 300
    start_index = 200
    ds = TorchDroidH5Dataset(
        h5_path=h5_path,
        action_horizon=action_horizon,
        action_space=action_space,
        fixed_instruction=fixed_instruction,
    )

    print(f"Dataset size: {len(ds)} samples")
    n = min(num_samples, len(ds) - start_index)
    for i in range(n):
        idx = start_index + i
        sample = ds[idx]
        actions = sample["actions"]
        print(f"\n[idx={idx}] prompt='{sample['prompt']}'")
        print(f" image: ext={sample['observation/exterior_image_1_left'].shape} wrist={sample['observation/wrist_image_left'].shape}")
        print(f" state: jp={sample['observation/joint_position'].shape} gp={sample['observation/gripper_position'].shape}")
        print(f" actions: {actions.shape}  min={actions.min():.5f} max={actions.max():.5f}")
        if actions.ndim == 2 and actions.shape[0] > 0:
            print(f"  actions[0]: {np.array2string(actions[0], precision=4, floatmode='fixed')}")
            print(f"  actions[-1]: {np.array2string(actions[-1], precision=4, floatmode='fixed')}")
