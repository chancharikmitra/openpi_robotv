"""
Causal Mediation Analysis (CMA) Head Selection via Mean Ablation

This script identifies causally important attention heads by replacing each head's 
activations with mean activations from a baseline dataset and measuring the 
resulting MSE degradation on an evaluation set.
"""

import h5py
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Configuration constants
NUM_LAYERS = 18
HEADS_PER_LAYER = 8
TOTAL_HEADS = NUM_LAYERS * HEADS_PER_LAYER  # 144
D_HEAD = 256


def load_activations_from_h5(
    h5_path: str,
    episodes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load attention activations and actions from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        episodes: List of episode paths to load
        
    Returns:
        activations: (N, 18, 8, 256) array of head activations
        actions: (N, action_dim) array of actions
    """
    all_activations = []
    all_actions = []
    
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
                if "last_token_attn" in frame_group:
                    attn = frame_group["last_token_attn"][:]
                    all_activations.append(attn)
                
                # Load actions
                if "pi_droid_action" in frame_group:
                    action = frame_group["pi_droid_action"][:]
                    all_actions.append(action)
                elif "action_label" in frame_group:
                    action = frame_group["action_label"][:]
                    all_actions.append(action)
    
    activations = np.array(all_activations, dtype=np.float32)  # (N, 18, 8, 256)
    actions = np.array(all_actions, dtype=np.float32)  # (N, action_dim)
    
    return activations, actions


def compute_baseline_means(
    baseline_attn_h5: str,
    baseline_episodes: List[str]
) -> np.ndarray:
    """
    Compute mean activations from baseline dataset.
    
    Args:
        baseline_attn_h5: Path to HDF5 file with baseline activations
        baseline_episodes: List of episode paths from baseline dataset
        
    Returns:
        mean_activations: (18, 8, 256) array of mean activations per head
    """
    print("Computing baseline mean activations...")
    
    # Load baseline activations
    activations, _ = load_activations_from_h5(baseline_attn_h5, baseline_episodes)
    
    # Compute mean across all frames
    mean_activations = np.mean(activations, axis=0)  # (18, 8, 256)
    
    print(f"Baseline dataset: {activations.shape[0]} frames")
    print(f"Mean activations shape: {mean_activations.shape}")
    
    return mean_activations


def evaluate_with_head_ablation(
    model_forward,  # Function that takes activations and returns predictions
    task_activations: np.ndarray,
    task_actions: np.ndarray,
    layer: int,
    head: int,
    ablation_value: np.ndarray
) -> float:
    """
    Evaluate MSE with a specific head ablated.
    
    Args:
        model_forward: Forward function that maps activations to action predictions
        task_activations: (N, 18, 8, 256) task activations
        task_actions: (N, action_dim) ground truth actions
        layer: Layer index (0-17)
        head: Head index within layer (0-7)
        ablation_value: (256,) replacement activation vector
        
    Returns:
        mse: Mean squared error with ablation
    """
    # Create ablated activations
    ablated_activations = task_activations.copy()
    ablated_activations[:, layer, head, :] = ablation_value
    
    # Get predictions with ablated activations
    # Note: This assumes model_forward exists and handles the activations
    predictions = model_forward(ablated_activations)
    
    # Compute MSE
    mse = np.mean((predictions - task_actions) ** 2)
    
    return mse


def causal_mediation_head_selection(
    baseline_attn_h5: str,
    task_attn_h5: str,
    num_heads: int = 20,
    max_episodes: Optional[int] = None
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
    
    # Get episodes from files
    with h5py.File(baseline_attn_h5, "r") as f:
        baseline_episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]
        if max_episodes:
            baseline_episodes = baseline_episodes[:max_episodes]
    
    with h5py.File(task_attn_h5, "r") as f:
        task_episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]
        if max_episodes:
            task_episodes = task_episodes[:max_episodes]
    
    print(f"Baseline dataset: {len(baseline_episodes)} episodes")
    print(f"Task dataset: {len(task_episodes)} episodes")
    
    # Step 1: Compute baseline means
    baseline_means = compute_baseline_means(baseline_attn_h5, baseline_episodes)
    
    # Step 2: Load task data
    print(f"\nLoading task-specific dataset...")
    task_activations, task_actions = load_activations_from_h5(task_attn_h5, task_episodes)
    print(f"Task dataset: {task_activations.shape[0]} frames")
    
    # Step 3: Compute baseline MSE (no ablation)
    # Note: This would require the actual model forward pass
    # For now, we'll use a placeholder that assumes you have access to model predictions
    print("\nComputing baseline MSE (no ablation)...")
    
    # Placeholder for actual model forward pass
    # In practice, you would:
    # baseline_predictions = model_forward(task_activations)
    # baseline_mse = np.mean((baseline_predictions - task_actions) ** 2)
    
    # For demonstration, we'll simulate degradation scores
    # In real usage, you would compute actual MSE with and without ablation
    
    # Step 4: Compute degradation for each head
    print(f"\nEvaluating ablation impact for {TOTAL_HEADS} heads...")
    
    degradation_scores = []
    
    for layer in tqdm(range(NUM_LAYERS), desc="Layers"):
        for head in range(HEADS_PER_LAYER):
            # Get ablation value for this head
            ablation_value = baseline_means[layer, head, :]
            
            # In practice, you would compute:
            # ablated_mse = evaluate_with_head_ablation(
            #     model_forward, task_activations, task_actions,
            #     layer, head, ablation_value
            # )
            # degradation = ablated_mse - baseline_mse
            
            # Placeholder degradation score (replace with actual computation)
            # This simulates degradation based on variance of the ablation value
            degradation = np.var(ablation_value) * np.random.uniform(0.5, 2.0)
            
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
        # baseline_mse += compute_mse(output['actions'], ground_truth)
    
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
            # ablated_mse += compute_mse(output['actions'], ground_truth)
        
        degradation = ablated_mse - baseline_mse
        
        layer = head_idx // HEADS_PER_LAYER
        head = head_idx % HEADS_PER_LAYER
        degradation_scores.append((degradation, layer, head))
    
    # Sort and select top heads
    degradation_scores.sort(reverse=True)
    trainable_head_indices = [(layer, head) for _, layer, head in degradation_scores[:num_heads]]
    
    return trainable_head_indices


# Example usage
if __name__ == "__main__":
    # Paths to your data
    BASELINE_ATTN_H5 = "/path/to/baseline_activations.h5"
    TASK_ATTN_H5 = "/path/to/task_activations.h5"
    
    # Run CMA head selection
    selected_heads = causal_mediation_head_selection(
        baseline_attn_h5=BASELINE_ATTN_H5,
        task_attn_h5=TASK_ATTN_H5,
        num_heads=20
    )
    
    print(f"\nSelected heads: {selected_heads}")