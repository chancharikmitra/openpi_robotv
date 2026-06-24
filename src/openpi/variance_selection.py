"""
Head selection based on activation variance metrics over green cube support set.

This module implements mechanistic interpretability approaches that select attention heads
showing maximum variance in their activations across the support set using different metrics:
1. L2 norm variance - variance in activation magnitudes
2. Total variance (trace of covariance) - statistically principled multivariate variance
"""

import h5py
import numpy as np
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from tqdm import tqdm

# Import existing utilities
from openpi.head_tuning.knn.data import build_dataset, PreprocessedData


@dataclass 
class ActivationVarianceSelector:
    """
    Head selector based on activation variance metrics.
    
    Supports multiple approaches:
    - "l2_variance": Variance in L2 norms of activations (magnitude-focused)
    - "total_variance": Trace of covariance matrix (statistically principled)
    """
    
    def __init__(
        self,
        use_pca: bool = False,
        use_zscore: bool = False, 
        pca_components: int = 32,
        num_layers: int = 18,
        heads_per_layer: int = 8,
        d_head: int = 256
    ):
        self.use_pca = use_pca
        self.use_zscore = use_zscore
        self.pca_components = pca_components
        self.num_layers = num_layers
        self.heads_per_layer = heads_per_layer
        self.d_head = d_head
        self.total_heads = num_layers * heads_per_layer
        
    def compute_l2_norms_per_head(
        self, 
        preprocessed_data: PreprocessedData
    ) -> np.ndarray:
        """
        Compute L2 norms for each head across all frames.
        
        Args:
            preprocessed_data: PreprocessedData object with activations
            
        Returns:
            l2_norms: (N, H) array where N=frames, H=heads - L2 norm per frame per head
        """
        # Xh_red shape: (N, total_heads, d) where d = pca_components or d_head
        N, H, d = preprocessed_data.Xh_red.shape
        assert H == self.total_heads, f"Expected {self.total_heads} heads, got {H}"
        
        print(f"Computing L2 norms for {N} frames across {H} heads...")
        
        # Vectorized L2 norm computation: ||activation||_2 for each (frame, head)
        l2_norms = np.linalg.norm(preprocessed_data.Xh_red, axis=2)  # (N, H)
        
        return l2_norms.astype(np.float32)
    
    def compute_l2_variance_scores(self, l2_norms: np.ndarray) -> np.ndarray:
        """
        Compute variance of L2 norms for each head.
        
        Args:
            l2_norms: (N, H) array of L2 norms per frame per head
            
        Returns:
            variance_scores: (H,) array - variance of L2 magnitudes across frames per head
        """
        print("Computing L2 norm variance scores...")
        
        # Compute variance across frames for each head
        variance_scores = np.var(l2_norms, axis=0)  # (H,)
        
        return variance_scores
    
    def compute_total_variance_scores(self, preprocessed_data: PreprocessedData) -> np.ndarray:
        """
        Compute total variance (trace of covariance matrix) for each head.
        
        This is the statistically principled approach: for each head, treat its activations
        as samples from a multivariate distribution and compute the total variance.
        
        Args:
            preprocessed_data: PreprocessedData object with activations
            
        Returns:
            total_variance_scores: (H,) array - trace of covariance matrix per head
        """
        N, H, d = preprocessed_data.Xh_red.shape
        print(f"Computing total variance (covariance trace) for {H} heads...")
        
        total_variance_scores = np.zeros(H, dtype=np.float32)
        
        for head_idx in tqdm(range(H), desc="Computing total variance"):
            # Get activations for this head: (N, d)
            head_activations = preprocessed_data.Xh_red[:, head_idx, :]
            
            # Compute total variance = trace of covariance matrix = sum of variances across dimensions
            total_variance = np.sum(np.var(head_activations, axis=0))  # Sum across d dimensions
            total_variance_scores[head_idx] = total_variance
            
        return total_variance_scores
    
    def select_heads_by_metric(
        self,
        attn_h5: str,
        episodes: List[str], 
        num_heads: int,
        metric: Literal["l2_variance", "total_variance"] = "total_variance",
        return_scores: bool = False
    ) -> Tuple[List[int], Optional[dict]]:
        """
        Select heads with highest variance using the specified metric.
        
        Args:
            attn_h5: Path to HDF5 file
            episodes: List of episode paths
            num_heads: Number of top heads to select
            metric: "l2_variance" or "total_variance"
            return_scores: If True, return detailed scoring information
            
        Returns:
            selected_heads: List[int] of length num_heads - head indices ranked by variance (highest first)
            scores: Optional dict containing computed scores and metadata
        """
        print("Building dataset with preprocessing...")
        preprocessed_data = build_dataset(
            attn_h5=attn_h5,
            episodes=episodes,
            use_pca=self.use_pca,
            use_zscore=self.use_zscore,
            pca_components=self.pca_components,
            num_layers=self.num_layers,
            heads_per_layer=self.heads_per_layer,
            d_head=self.d_head
        )
        
        # Compute variance scores based on selected metric
        if metric == "l2_variance":
            l2_norms = self.compute_l2_norms_per_head(preprocessed_data)
            variance_scores = self.compute_l2_variance_scores(l2_norms)
            score_description = "L2 norm variance"
        elif metric == "total_variance":
            variance_scores = self.compute_total_variance_scores(preprocessed_data)
            l2_norms = None  # Not computed for this metric
            score_description = "total variance (covariance trace)"
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'l2_variance' or 'total_variance'")
        
        # Select top heads by variance
        top_head_indices = np.argsort(variance_scores)[::-1][:num_heads]
        selected_heads = top_head_indices.tolist()
        
        print(f"\nSelected {num_heads} heads with highest {score_description}:")
        print(f"Head indices: {selected_heads}")
        print(f"Variance scores: {variance_scores[top_head_indices]}")
        
        # Convert to layer.head notation for interpretability and output formatting
        trainable_head_indices = []
        print(f"\nHead locations (layer.head):")
        for i, head_idx in enumerate(selected_heads):
            layer = head_idx // self.heads_per_layer
            head_in_layer = head_idx % self.heads_per_layer
            trainable_head_indices.append((layer, head_in_layer))
            variance = variance_scores[head_idx]
            print(f"  Rank {i+1}: Head {head_idx} (L{layer}.H{head_in_layer}) - {score_description}: {variance:.6f}")
        
        # Print final output in desired format
        print(f"\n" + "="*60)
        print("FINAL OUTPUT - Copy this for your training pipeline:")
        print("="*60)
        print("trainable_head_indices=[")
        for i, (layer, head) in enumerate(trainable_head_indices):
            if i == len(trainable_head_indices) - 1:  # Last element
                print(f"    ({layer}, {head})  # Rank {i+1}")
            else:
                print(f"    ({layer}, {head}),  # Rank {i+1}")
        print("]")
        print("="*60)
        
        scores = None
        if return_scores:
            scores = {
                "variance_scores": variance_scores,
                "metric_used": metric,
                "selected_heads": selected_heads,
                "trainable_head_indices": trainable_head_indices,  # Add tuple format
                "selected_head_variances": variance_scores[top_head_indices],
                "preprocessed_data": preprocessed_data
            }
            if l2_norms is not None:
                scores["l2_norms"] = l2_norms
        
        return selected_heads, scores


def select_heads_by_activation_variance(
    attn_h5: str,
    episodes: List[str],
    num_heads: int = 20,
    metric: Literal["l2_variance", "total_variance"] = "total_variance",
    use_pca: bool = False,
    use_zscore: bool = False,
    pca_components: int = 32,
    return_analysis: bool = False
) -> Tuple[List[int], Optional[dict]]:
    """
    Convenience function for activation variance head selection.
    
    Args:
        attn_h5: Path to HDF5 file with attention data
        episodes: List of episode paths to use as support set
        num_heads: Number of heads to select
        metric: "l2_variance" (magnitude variance) or "total_variance" (statistically principled)
        use_pca: Whether to apply PCA to head activations
        use_zscore: Whether to z-score normalize head activations
        pca_components: Number of PCA components if use_pca=True
        return_analysis: Whether to return detailed analysis
        
    Returns:
        selected_heads: List[int] of length num_heads - head indices ranked by variance (highest first)
        analysis: Optional dict containing:
            - "variance_scores": np.ndarray (144,) float32 - variance metric for each head (PLOTTING ARRAY)
            - "metric_used": str - which metric was used for selection
            - "selected_head_variances": np.ndarray (num_heads,) float32 - variances of selected heads only
            - "mean_variance": float - mean variance across all 144 heads
            - "std_variance": float - standard deviation of variance across all heads  
            - "variance_percentiles": np.ndarray (6,) float32 - [25th, 50th, 75th, 90th, 95th, 99th] percentiles
            - "l2_norms": np.ndarray (N_frames, 144) float32 - raw L2 norms (only for l2_variance metric)
    """
    selector = ActivationVarianceSelector(
        use_pca=use_pca,
        use_zscore=use_zscore,
        pca_components=pca_components
    )
    
    selected_heads, scores = selector.select_heads_by_metric(
        attn_h5=attn_h5,
        episodes=episodes,
        num_heads=num_heads,
        metric=metric,
        return_scores=True  # Always return scores to ensure variance_scores is available
    )
    
    variance_scores = scores["variance_scores"]
    analysis = {
        "variance_scores": variance_scores,  # (144,) array - THIS IS YOUR PLOTTING ARRAY
        "metric_used": metric,
        "trainable_head_indices": scores["trainable_head_indices"],  # (layer, head) tuples
        "selected_head_variances": scores["selected_head_variances"],
        "mean_variance": np.mean(variance_scores),
        "std_variance": np.std(variance_scores),
        "variance_percentiles": np.percentile(variance_scores, [25, 50, 75, 90, 95, 99])
    }
    
    # Add l2_norms if available (only for l2_variance metric)
    if "l2_norms" in scores:
        analysis["l2_norms"] = scores["l2_norms"]
    
    if return_analysis:
        print(f"\nVariance Analysis ({metric}):")
        print(f"  Mean variance across all heads: {analysis['mean_variance']:.6f}")
        print(f"  Std variance across all heads:  {analysis['std_variance']:.6f}")
        print(f"  Selected heads mean variance:   {np.mean(analysis['selected_head_variances']):.6f}")
        print(f"  Variance percentiles (25,50,75,90,95,99): {analysis['variance_percentiles']}")
        
        return selected_heads, analysis
    else:
        # Still return variance_scores for plotting even if detailed analysis not requested
        return selected_heads, {
            "variance_scores": variance_scores, 
            "metric_used": metric,
            "trainable_head_indices": scores["trainable_head_indices"]
        }


def compare_head_selection_metrics(
    attn_h5: str,
    episodes: List[str],
    num_heads: int = 20,
    use_pca: bool = False,
    use_zscore: bool = False,
    pca_components: int = 32
) -> dict:
    """
    Compare head selection using both L2 variance and total variance metrics.
    
    Returns:
        comparison: dict with results from both metrics and overlap analysis
    """
    print("=" * 60)
    print("COMPARING HEAD SELECTION METRICS")
    print("=" * 60)
    
    # Select using L2 variance
    print("\n1. L2 NORM VARIANCE SELECTION:")
    print("-" * 40)
    heads_l2, analysis_l2 = select_heads_by_activation_variance(
        attn_h5=attn_h5, episodes=episodes, num_heads=num_heads,
        metric="l2_variance", use_pca=use_pca, use_zscore=use_zscore,
        pca_components=pca_components, return_analysis=True
    )
    
    # Select using total variance
    print("\n2. TOTAL VARIANCE (COVARIANCE TRACE) SELECTION:")
    print("-" * 40)
    heads_total, analysis_total = select_heads_by_activation_variance(
        attn_h5=attn_h5, episodes=episodes, num_heads=num_heads,
        metric="total_variance", use_pca=use_pca, use_zscore=use_zscore,
        pca_components=pca_components, return_analysis=True
    )
    
    # Analyze overlap
    overlap = len(set(heads_l2) & set(heads_total))
    overlap_percent = (overlap / num_heads) * 100
    
    print("\n3. COMPARISON ANALYSIS:")
    print("-" * 40)
    print(f"L2 variance heads:     {heads_l2}")
    print(f"Total variance heads:  {heads_total}")
    print(f"Overlap:               {overlap}/{num_heads} heads ({overlap_percent:.1f}%)")
    print(f"L2-only heads:         {set(heads_l2) - set(heads_total)}")
    print(f"Total-only heads:      {set(heads_total) - set(heads_l2)}")
    
    return {
        "l2_variance": {
            "selected_heads": heads_l2,
            "analysis": analysis_l2
        },
        "total_variance": {
            "selected_heads": heads_total,
            "analysis": analysis_total
        },
        "overlap": {
            "count": overlap,
            "percentage": overlap_percent,
            "common_heads": list(set(heads_l2) & set(heads_total)),
            "l2_only": list(set(heads_l2) - set(heads_total)),
            "total_only": list(set(heads_total) - set(heads_l2))
        }
    }


# Example usage function
def example_green_cube_head_selection():
    """Example of how to use activation variance head selection for green cube task."""
    
    # Example paths - replace with your actual data
    ATTN_H5 = "/home/cmitra/pick_green_cube_20.h5"
    
    # Get episodes from H5 file
    with h5py.File(ATTN_H5, "r") as f:
        episodes = [f"{task}/{ep}" for task in f.keys() for ep in f[task].keys()]
    
    print(f"Found {len(episodes)} episodes for head selection")
    
    # Option 1: Use total variance (recommended - statistically principled)
    print("\n" + "="*50)
    print("TOTAL VARIANCE HEAD SELECTION (RECOMMENDED)")
    print("="*50)
    selected_heads_total, analysis_total = select_heads_by_activation_variance(
        attn_h5=ATTN_H5,
        episodes=episodes,
        num_heads=20,
        metric="total_variance",  # Statistically principled approach
        use_pca=False,  # Start without PCA to see raw activation patterns
        use_zscore=False,  # Start without z-score to preserve magnitude info
        return_analysis=True
    )
    
    # Option 2: Compare both metrics
    print("\n" + "="*50)
    print("COMPARING BOTH METRICS")
    print("="*50)
    comparison = compare_head_selection_metrics(
        attn_h5=ATTN_H5,
        episodes=episodes,
        num_heads=20,
        use_pca=False,
        use_zscore=False
    )
    
    # For plotting variance scores
    variance_scores_total = analysis_total["variance_scores"]  # (144,) array
    variance_scores_l2 = comparison["l2_variance"]["analysis"]["variance_scores"]  # (144,) array
    
    print(f"\nFinal recommended heads (total variance): {selected_heads_total}")
    print(f"Variance scores available for plotting: total_variance.shape={variance_scores_total.shape}, l2_variance.shape={variance_scores_l2.shape}")
    
    return {
        "selected_heads": selected_heads_total,
        "analysis": analysis_total,
        "comparison": comparison
    }


if __name__ == "__main__":
    # Run example
    results = example_green_cube_head_selection()