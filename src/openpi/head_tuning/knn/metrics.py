import numpy as np
from .inference import build_feat_subset


def pairwise_dist(q: np.ndarray, base: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Compute distances from query q to each row of base.

    q may be 1-D (P,) or 2-D (Bq, P).  When 2-D, delegates to
    pairwise_dist_matrix and returns (Bq, Nb).
    """
    if q.ndim == 2:
        return pairwise_dist_matrix(q, base, metric)
    # 1-D query path
    if metric == "cosine":
        qn = q / (np.linalg.norm(q) + 1e-8)
        bn = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-8)
        return 1.0 - (bn @ qn)
    elif metric == "euclidean":
        diff = base - q[None, :]
        return np.sqrt((diff * diff).sum(axis=1))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")


def pairwise_dist_matrix(Q: np.ndarray, B: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise distances between rows of Q (Bq, P) and B (Nb, P): returns (Bq, Nb).
    """
    if metric == "cosine":
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        # cosine distance = 1 - cos_sim = 1 - Qn @ Bn^T
        return 1.0 - (Qn @ Bn.T)
    elif metric == "euclidean":
        # ||Q-B|| = sqrt(||Q||^2 + ||B||^2 - 2 Q B^T)
        Q2 = np.sum(Q * Q, axis=1, keepdims=True)         # (Bq, 1)
        B2 = np.sum(B * B, axis=1, keepdims=True).T       # (1, Nb)
        cross = Q @ B.T                                    # (Bq, Nb)
        D2 = np.maximum(Q2 + B2 - 2.0 * cross, 0.0)
        return np.sqrt(D2)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")


__all__ = ["pairwise_dist", "pairwise_dist_matrix"]
