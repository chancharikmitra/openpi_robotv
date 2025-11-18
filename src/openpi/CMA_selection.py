"""
Causal Mediation Analysis (CMA) for Attention Head Selection

Goal: Identify causally important attention heads by replacing a target head's
state-position activation with its baseline mean and measuring the MSE degradation
of predicted actions against ground-truth actions.
"""

import h5py
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import h5py
import re
from tqdm import tqdm

# Configuration constants
NUM_LAYERS = 18
HEADS_PER_LAYER = 8
TOTAL_HEADS = NUM_LAYERS * HEADS_PER_LAYER  # 144
D_HEAD = 256


def load_state_heads_and_actions(
    h5_path: str,
    episodes: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Load state-position head activations and actions from a keyframe-only H5.

    Returns:
      state_heads: (N, 18, 8, 256)
      gt_actions: (N, action_dim)
      pred_actions: (N, action_dim)  # from generation time, baseline predictions
      frame_ids: list of (episode_path, frame_key)
    """
    all_activations = []
    all_actions = []
    predicted_actions = []
    frame_ids: List[Tuple[str, str]] = []  # (episode_path, frame_key)
    
    with h5py.File(h5_path, "r") as f:
        for episode_path in episodes:
            if episode_path not in f:
                continue
                
            episode_group = f[episode_path]
            
            # Iterate through frames in this episode
            for frame_key in episode_group.keys():
                if not frame_key.startswith("frame_"):
                    continue
                    
                frame_group = episode_group[frame_key]
                
                # Load activations - expecting shape (18, 8, 256)
                if "first_action_token_attn" in frame_group:
                    attn = frame_group["first_action_token_attn"][:]
                    all_activations.append(attn)
                
                # Load actions
                if "pi_droid_action" in frame_group:
                    action = frame_group["pi_droid_action"][:]
                    all_actions.append(action)
                elif "action_label" in frame_group:
                    action = frame_group["action_label"][:]
                    all_actions.append(action)
                if "predicted_action_label" in frame_group:
                    predicted_action = frame_group["predicted_action_label"][:]
                    predicted_actions.append(predicted_action)
                # Track frame id if we appended an activation and have GT
                if (len(all_activations) > len(frame_ids)) and (len(all_actions) == len(all_activations)):
                    frame_ids.append((episode_path, frame_key))
    
    state_heads = np.array(all_activations, dtype=np.float32)
    gt_actions = np.array(all_actions, dtype=np.float32)
    pred_actions = np.array(predicted_actions, dtype=np.float32)
    return state_heads, gt_actions, pred_actions, frame_ids


def compute_baseline_state_mean(
    baseline_attn_h5: str,
    baseline_episodes: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute per-head baseline mean at first action position across keyframes.

    Returns: (18, 8, 256)
    """
    print("Computing baseline mean activations...")
    # Support two formats:
    # 1) Aggregated baseline H5 with root dataset 'first_action_token_attn' (mean already)
    # 2) Per-episode H5 (task/episode/frame_*/first_action_token_attn), compute mean over frames
    with h5py.File(baseline_attn_h5, "r") as f:
        if "first_action_token_attn" in f and isinstance(f["first_action_token_attn"], h5py.Dataset):
            mean_activations = f["first_action_token_attn"][:].astype(np.float32)
            num_frames = int(f.attrs.get("num_frames", -1))
            print(f"Baseline dataset (aggregated): frames={num_frames} mean_shape={mean_activations.shape}")
            return mean_activations

    # If not aggregated, assume per-episode structure
    with h5py.File(baseline_attn_h5, "r") as f:
        if baseline_episodes is None:
            baseline_episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys() if isinstance(f[task], h5py.Group)]
    first_action_tokens, _, _, _ = load_state_heads_and_actions(baseline_attn_h5, baseline_episodes)
    mean_activations = np.mean(first_action_tokens, axis=0)
    print(f"Baseline dataset: {first_action_tokens.shape[0]} frames")
    print(f"Mean activations shape: {mean_activations.shape}")
    return mean_activations


def build_delta_heads_for_frame(
    baseline_mean_state: np.ndarray,  # (18, 8, 256)
    current_state_heads: np.ndarray,  # (18, 8, 256) for a single frame
    layer: int,
    head: int,
) -> np.ndarray:
    """Construct per-layer delta_heads tensor for a single frame at target (layer, head).

    delta_heads[L,H,D] = baseline_mean[layer, head, :] - current[layer, head, :]
    other heads = 0
    """
    delta_heads = np.zeros_like(baseline_mean_state, dtype=np.float32)
    delta = baseline_mean_state[layer, head, :] - current_state_heads[layer, head, :]
    delta_heads[layer, head, :] = delta
    return delta_heads


def causal_mediation_head_selection(
    baseline_attn_h5: str,
    task_attn_h5: str,
    *,
    infer_fn: Callable[[int, np.ndarray, Optional[int]], np.ndarray] | None = None,
    policy=None,
    raw_h5_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    delta_token_index: int = 0,
    num_heads: int = 20,
    max_episodes: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Select attention heads using Causal Mediation Analysis.
    
    This function:
    1. Computes mean activations from baseline dataset
    2. For each head, replaces its activation with the baseline mean
    3. Measures MSE increase (degradation) on task dataset
    4. Selects heads with largest degradation (most important)
    
    Args:
        baseline_attn_h5: Path to baseline activations HDF5
        task_attn_h5: Path to task-specific activations HDF5
        num_heads: Number of heads to select
        max_episodes: Optional limit on episodes to use
        
    Returns:
        trainable_head_indices: List of (layer, head) tuples
    """
    print("="*60)
    print("CAUSAL MEDIATION ANALYSIS HEAD SELECTION")
    print("="*60)
    
    # Get task episodes from file (baseline may be aggregated without episodes)
    with h5py.File(task_attn_h5, "r") as f:
        task_episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]
        if max_episodes:
            task_episodes = task_episodes[:max_episodes]

    print(f"Task dataset: {len(task_episodes)} episodes")
    
    # Step 1: Compute baseline means
    baseline_means = compute_baseline_state_mean(baseline_attn_h5, None)
    
    # Step 2: Load task data
    print(f"\nLoading task-specific dataset...")
    task_state_heads, task_actions, task_predicted_actions, frame_ids = load_state_heads_and_actions(task_attn_h5, task_episodes)
    if max_frames is not None and task_state_heads.shape[0] > max_frames:
        idx = np.random.permutation(task_state_heads.shape[0])[:max_frames]
        task_state_heads = task_state_heads[idx]
        task_actions = task_actions[idx]
        task_predicted_actions = task_predicted_actions[idx]
        frame_ids = [frame_ids[i] for i in idx]
    print(f"Task dataset: {task_state_heads.shape[0]} frames")
    
    # Step 3: Compute baseline MSE (no ablation) using predicted actions from H5
    print("\nComputing baseline MSE (no ablation) from H5 predicted_action_label...")
    if task_predicted_actions.size == 0:
        raise ValueError("H5 missing 'predicted_action_label'; cannot compute baseline MSE without running model.")
    baseline_predictions = task_predicted_actions
    baseline_mse = np.mean((baseline_predictions - task_actions) ** 2)
    
    # For demonstration, we'll simulate degradation scores
    # In real usage, you would compute actual MSE with and without ablation
    
    # Step 4: Compute degradation for each head
    print(f"\nEvaluating ablation impact for {TOTAL_HEADS} heads...")
    
    degradation_scores = []
    
    if infer_fn is None:
        if policy is None or raw_h5_path is None:
            raise ValueError("Either provide infer_fn, or provide (policy, raw_h5_path) to reconstruct obs for infer.")

        # Build observation index from raw dataset, mirroring generate_activation_dataset_on_robot
        obs_index = _build_obs_index_from_raw(raw_h5_path, prompt_text or "remove marker from mug")

        # Map attention-H5 frame_ids (task/episode_xxx, frame_yyyy) to (ep_idx, frame_idx)
        parsed_ids: List[Tuple[int, int]] = []
        for ep_path, frame_key in frame_ids:
            m = re.search(r"episode_(\d+)", ep_path)
            if not m:
                raise ValueError(f"Cannot parse episode index from: {ep_path}")
            ep_idx = int(m.group(1))  # 1-based in writer
            m2 = re.search(r"frame_(\d+)", frame_key)
            if not m2:
                raise ValueError(f"Cannot parse frame index from: {frame_key}")
            fr_idx = int(m2.group(1))  # 0-based in writer name
            parsed_ids.append((ep_idx - 1, fr_idx))  # convert to 0-based episode

        def _infer_fn(i: int, delta_heads_full: np.ndarray, dti: Optional[int]) -> np.ndarray:
            ep_i, fr_i = parsed_ids[i]
            try:
                obs = obs_index[ep_i][fr_i]
            except Exception as e:
                raise IndexError(f"Obs not found for (episode={ep_i}, frame={fr_i}) from {raw_h5_path}") from e
            out = policy.infer(obs, delta_heads=delta_heads_full, delta_token_index=dti)
            return np.asarray(out["actions"])  # (1, ah, ad)

        infer_fn = _infer_fn

    for layer in tqdm(range(NUM_LAYERS), desc="Layers"):
        for head in range(HEADS_PER_LAYER):
            # Compute ablated predictions by injecting delta at state position
            ablated_preds = []
            for i, (ep, fr) in enumerate(frame_ids):
                delta_heads_full = build_delta_heads_for_frame(
                    baseline_means,
                    task_state_heads[i],
                    layer,
                    head,
                )
                pred = infer_fn(i, delta_heads_full, delta_token_index)
                # Expect policy returns shape (1, ah, ad); reduce to per-frame action vector to match GT
                ablated_preds.append(pred[0])
            ablated_preds = np.asarray(ablated_preds, dtype=np.float32)
            ablated_mse = np.mean((ablated_preds - task_actions) ** 2)
            degradation = ablated_mse - baseline_mse
            
            # Placeholder degradation score (replace with actual computation)
            # This simulates degradation based on variance of the ablation value
            # degradation = np.var(ablation_value) * np.random.uniform(0.5, 2.0)
            
            degradation_scores.append((degradation, layer, head))
    
    # Step 5: Sort by degradation and select top heads
    degradation_scores.sort(reverse=True)  # Highest degradation first
    
    # Extract top k heads
    trainable_head_indices = []
    print(f"\nTop {num_heads} causally important heads:")
    
    for i in range(num_heads):
        degradation, layer, head = degradation_scores[i]
        trainable_head_indices.append((layer, head))
        head_idx = layer * HEADS_PER_LAYER + head
        print(f"  Rank {i+1}: Head {head_idx} (L{layer}.H{head}) - Degradation: {degradation:.6f}")
    
    # Print final output in desired format
    print(f"\n" + "="*60)
    print("FINAL OUTPUT - Copy this for your training pipeline:")
    print("="*60)
    print("trainable_head_indices=[")
    for i, (layer, head) in enumerate(trainable_head_indices):
        if i == len(trainable_head_indices) - 1:
            print(f"    ({layer}, {head})  # Rank {i+1}")
        else:
            print(f"    ({layer}, {head}),  # Rank {i+1}")
    print("]")
    print("="*60)
    # Diagnostics for full distribution
    all_deg = np.array([d for d, _, _ in degradation_scores], dtype=np.float32)
    best_deg, best_layer, best_head = degradation_scores[0]
    worst_deg, worst_layer, worst_head = degradation_scores[-1]
    print(f"Total heads evaluated: {len(degradation_scores)}")
    print(f"Best head:  L{best_layer}.H{best_head}  degradation={best_deg:.6f}")
    print(f"Worst head: L{worst_layer}.H{worst_head} degradation={worst_deg:.6f}")
    print(f"Gap (best - worst): {best_deg - worst_deg:.6f}")
    print(f"Mean={all_deg.mean():.6f}  Std={all_deg.std():.6f}  Min={all_deg.min():.6f}  Max={all_deg.max():.6f}")
    print(f"\nBottom {num_heads} heads:")
    for i in range(1, min(num_heads, len(degradation_scores)) + 1):
        d, l, h = degradation_scores[-i]
        head_idx = l * HEADS_PER_LAYER + h
        print(f"  Rank -{i}: Head {head_idx} (L{l}.H{h}) - Degradation: {d:.6f}")
    
    return trainable_head_indices


# Note: Actual implementation with model
def causal_mediation_with_model(
    policy,  # The actual policy object with infer method
    baseline_attn_h5: str,
    task_observations: List[dict],  # List of observation dicts for task
    num_heads: int = 20
) -> List[Tuple[int, int]]:
    """
    CMA head selection using actual model inference.
    
    This version would use the policy's infer method with delta_heads
    to perform actual ablation during model forward passes.
    
    Args:
        policy: Policy object with infer method that accepts delta_heads
        baseline_attn_h5: Path to baseline activations
        task_observations: List of task observations
        num_heads: Number of heads to select
        
    Returns:
        trainable_head_indices: List of (layer, head) tuples
    """
    # Get baseline means
    with h5py.File(baseline_attn_h5, "r") as f:
        episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]
    
    baseline_means = compute_baseline_means(baseline_attn_h5, episodes)
    
    # Reshape to (144, 256) for delta_heads format
    baseline_means_flat = baseline_means.reshape(TOTAL_HEADS, D_HEAD)
    
    degradation_scores = []
    
    # Compute baseline MSE
    baseline_mse = 0
    for obs in task_observations:
        output = policy.infer(obs, delta_heads=None)
        # Compare output['actions'] with ground truth
        baseline_mse += compute_mse(output['actions'], ground_truth)
    
    # Test each head
    for head_idx in range(TOTAL_HEADS):
        # Create delta_heads that replaces this head with baseline mean
        # Note: delta_heads is typically additive, so you'd need to
        # subtract current and add baseline mean
        delta_heads = np.zeros((TOTAL_HEADS, D_HEAD))
        delta_heads[head_idx] = baseline_means_flat[head_idx]
        
        # Compute MSE with ablation
        ablated_mse = 0
        for obs in task_observations:
            output = policy.infer(obs, delta_heads=delta_heads)
            ablated_mse += compute_mse(output['actions'], ground_truth)
        
        degradation = ablated_mse - baseline_mse
        
        layer = head_idx // HEADS_PER_LAYER
        head = head_idx % HEADS_PER_LAYER
        degradation_scores.append((degradation, layer, head))
    
    # Sort and select top heads
    degradation_scores.sort(reverse=True)
    trainable_head_indices = [(layer, head) for _, layer, head in degradation_scores[:num_heads]]
    
    return trainable_head_indices


def _build_obs_index_from_raw(h5_path: str, prompt_text: str) -> List[List[dict]]:
    """Builds a 2D list obs_index[episode_idx][frame_idx] -> obs dict, mirroring generate_activation_dataset_on_robot.

    Only fields needed by policy.infer are constructed.
    """
    obs_index: List[List[dict]] = []
    with h5py.File(h5_path, "r") as f:
        for episode_name in f.keys():
            grp = f[episode_name]
            if not isinstance(grp, h5py.Group):
                continue
            # numeric step subgroups
            steps = sorted([s for s in grp.keys() if s.isdigit()], key=lambda x: int(x))
            if not steps:
                continue
            episode_obs: List[dict] = []
            for s in steps:
                sg = grp[s]
                req = all(k in sg for k in ("obs", "act_vel", "act_pos", "rgb_left", "rgb_wrist"))
                if not req:
                    continue
                obs = np.asarray(sg["obs"]).astype(np.float32)
                img_left = np.asarray(sg["rgb_right"])          # (256,256,3)
                img_wrist = np.asarray(sg["rgb_wrist"])        # (256,256,3)
                jp = np.asarray(obs[:7], dtype=np.float32)
                gp = np.asarray(obs[7:8], dtype=np.float32)
                episode_obs.append({
                    "observation/exterior_image_1_left": img_left,
                    "observation/wrist_image_left":     img_wrist,
                    "observation/joint_position":       jp,
                    "observation/gripper_position":     gp,
                    "prompt": prompt_text,
                })
            if episode_obs:
                obs_index.append(episode_obs)
    return obs_index


def build_baseline_h5_from_multiple(
    input_h5_paths: List[str],
    output_h5_path: str,
    *,
    max_episodes: Optional[int] = None,
) -> None:
    """Aggregate multiple keyframe attention H5s into a single baseline H5.

    Produces an H5 with datasets:
      - baseline_state_mean: (18, 8, 256)
      - baseline_state_std:  (18, 8, 256)
    And attributes:
      - num_frames: int
      - sources: list of source paths
    """
    sum_heads = np.zeros((NUM_LAYERS, HEADS_PER_LAYER, D_HEAD), dtype=np.float64)
    sumsq_heads = np.zeros_like(sum_heads)
    total_frames = 0

    for h5_path in input_h5_paths:
        with h5py.File(h5_path, "r") as f:
            episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]
            if max_episodes:
                episodes = episodes[:max_episodes]
            for ep in episodes:
                if ep not in f:
                    continue
                grp = f[ep]
                for frame_key in grp.keys():
                    if not frame_key.startswith("frame_"):
                        continue
                    fr = grp[frame_key]
                    if "first_action_token_attn" not in fr:
                        continue
                    attn = fr["first_action_token_attn"][:].astype(np.float64)  # (18,8,256)
                    sum_heads += attn
                    sumsq_heads += attn * attn
                    total_frames += 1

    if total_frames == 0:
        raise ValueError("No first_action_token_attn found across provided H5s.")

    mean = (sum_heads / total_frames).astype(np.float32)
    var = np.maximum(0.0, (sumsq_heads / total_frames) - (sum_heads / total_frames) ** 2)
    std = np.sqrt(var).astype(np.float32)

    with h5py.File(output_h5_path, "w") as out:
        out.create_dataset("first_action_token_attn", data=mean, compression="gzip")
        out.attrs["num_frames"] = int(total_frames)
        out.attrs["sources"] = np.array(input_h5_paths, dtype=h5py.string_dtype())


# Example usage
if __name__ == "__main__":
    import openpi.training.config as config
    from openpi.shared import download
    from openpi.policies import policy_config
    # # Paths to your data
    
    TASK_ATTN_H5 = "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_place_marker_in_mug_20.h5"
    config = config.get_config("pi05_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    # Ensure normalization assets are present (includes droid/norm_stats.json)
    download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid/assets")

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    # Run CMA head selection
    # selected_heads = causal_mediation_head_selection(
    #     baseline_attn_h5=BASELINE_ATTN_H5,
    #     task_attn_h5=TASK_ATTN_H5,
    #     num_heads=20
    # )
    build_baseline_h5_from_multiple(
    [
        "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_pick_up_red_mug_20.h5",
        "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_place_green_cube_in_red_bowl_20.h5",
        "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_press_the_button_hard_20.h5",
        "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_push_red_bowl_to_red_cup_20.h5",
        "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_pick_up_red_cube_20.h5",
    ],
    "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_CMA_baseline_agg_first_action.h5",
    )
    BASELINE_ATTN_H5 = "/home/yusenluo/openpi_robotv/attention_dataset/pi0.5_CMA_baseline_agg_first_action.h5"
    selected_heads = causal_mediation_head_selection(
    baseline_attn_h5=BASELINE_ATTN_H5,
    task_attn_h5=TASK_ATTN_H5,
    policy=policy,                            # 已创建好的 policy (pi05)
    raw_h5_path="/home/yusenluo/openpi_robotv/9_21/place_marker_in_mug_20.h5",    # 就是 generate_activation 使用的原始 h5
    prompt_text="place marker in mug",     # 与生成时一致
    delta_token_index=0,                      # pi05: 注入到第一个 action token (pi05 无 state token)
    num_heads=20,                          # 可选限量
    max_frames=300,
    )
    
    # print(f"\nSelected heads: {selected_heads}")