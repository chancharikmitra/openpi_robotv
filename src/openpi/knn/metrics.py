import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from .utils import build_feat_subset


def pairwise_dist(q: np.ndarray, base: np.ndarray, metric: str, metric_ctx: dict | None = None) -> np.ndarray:
    if metric == "cosine":
        qn = q / (np.linalg.norm(q) + 1e-8)
        bn = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-8)
        return 1.0 - (bn @ qn)
    elif metric == "euclidean":
        diff = base - q[None, :]
        return np.sqrt((diff * diff).sum(axis=1))

    elif metric == "whiten":
        assert metric_ctx is not None and "Wg" in metric_ctx and "mu" in metric_ctx
        Wg = metric_ctx["Wg"]
        mu = metric_ctx["mu"]
        q2 = (q - mu) @ Wg
        b2 = (base - mu[None, :]) @ Wg
        diff = b2 - q2[None, :]
        return np.sqrt((diff * diff).sum(axis=1))

    elif metric == "proj" or metric == "pls":
        assert metric_ctx is not None and "W" in metric_ctx
        W  = metric_ctx["W"]
        if W.ndim == 2 and W.shape[0] != q.shape[0] and W.shape[1] == q.shape[0]:
            W = W.T
        q2 = q @ W
        b2 = base @ W
        diff = b2 - q2[None, :]
        return np.sqrt((diff * diff).sum(axis=1))

    else:
        raise ValueError("metric must be 'cosine' or 'euclidean' or 'whiten' or 'proj' or 'pls'")


__all__ = ["pairwise_dist"]


def build_metric_ctx_from_train(
    Xtr: np.ndarray,
    Ytr: Optional[np.ndarray],
    metric: str,
    pca_dim: int,
    proj_alpha: float
) -> Optional[dict]:
    if metric == "whiten":
        mean_train  = Xtr.mean(axis=0, keepdims=False)
        centered_X  = Xtr - mean_train
        num_components = min(pca_dim, Xtr.shape[1])
        pca_model   = PCA(n_components=num_components, svd_solver="auto", random_state=0).fit(centered_X)
        whitening_matrix = pca_model.components_.T / (np.sqrt(pca_model.explained_variance_ + 1e-8))
        return {"Wg": whitening_matrix.astype(np.float32), "mu": mean_train.astype(np.float32)}
    elif metric == "proj":
        assert Ytr is not None, "proj metric requires Ytr"
        ridge_model = Ridge(alpha=proj_alpha, fit_intercept=False, random_state=0).fit(Xtr, Ytr)
        projection_matrix = ridge_model.coef_.T
        return {"W": projection_matrix.astype(np.float32)}
    elif metric == "pls":
        assert Ytr is not None, "pls metric requires Ytr"
        num_components = min(Xtr.shape[1], Ytr.shape[1])
        pls_model = PLSRegression(n_components=num_components, scale=False)
        pls_model.fit(Xtr, Ytr)
        projection_matrix = pls_model.coef_
        if projection_matrix.shape[0] != Xtr.shape[1] and projection_matrix.shape[1] == Xtr.shape[1]:
            projection_matrix = projection_matrix.T
        return {"W": projection_matrix.astype(np.float32)}
    else:
        return None


def build_global_metric_ctx_for_heads(
    pre,
    heads,
    metric: str,
    pca_dim: int,
    proj_alpha: float,
    pls_components: int,
    use_fullspace: bool,
    H: int,
    metric_ctx_cache: dict,
    metric_ctx_fullspace: dict
) -> Optional[dict]:
    if metric not in {"whiten", "proj", "pls"}:
        return None
    key = tuple(sorted(heads))
    if key in metric_ctx_cache:
        return metric_ctx_cache[key]

    if use_fullspace and metric in {"whiten", "proj", "pls"}:
        full_key = f"full::{metric}"
        if full_key not in metric_ctx_fullspace:
            X_full = build_feat_subset(pre.Xh_red, list(range(H))).astype(np.float32)
            if metric == "whiten":
                mean_full  = X_full.mean(axis=0, keepdims=False)
                Xc         = X_full - mean_full
                r          = min(pca_dim, X_full.shape[1])
                pca        = PCA(n_components=r, svd_solver="auto", random_state=0).fit(Xc)
                whitening_matrix_full = pca.components_.T / (np.sqrt(pca.explained_variance_ + 1e-8))
                metric_ctx_fullspace[full_key] = {"Wg": whitening_matrix_full.astype(np.float32), "mu": mean_full.astype(np.float32)}
            else:
                Y_full = pre.Y.astype(np.float32)
                if metric == "proj":
                    reg = Ridge(alpha=proj_alpha, fit_intercept=False, random_state=0).fit(X_full, Y_full)
                    projection_matrix_full  = reg.coef_.T
                    if projection_matrix_full.shape[0] != X_full.shape[1] and projection_matrix_full.shape[1] == X_full.shape[1]:
                        projection_matrix_full = projection_matrix_full.T
                    metric_ctx_fullspace[full_key] = {"W": projection_matrix_full.astype(np.float32)}
                else:
                    n_comp = min(pls_components, X_full.shape[1], Y_full.shape[1])
                    pls    = PLSRegression(n_components=n_comp, scale=False)
                    pls.fit(X_full, Y_full)
                    projection_matrix_full    = pls.coef_
                    if projection_matrix_full.shape[0] != X_full.shape[1] and projection_matrix_full.shape[1] == X_full.shape[1]:
                        projection_matrix_full = projection_matrix_full.T
                    metric_ctx_fullspace[full_key] = {"W": projection_matrix_full.astype(np.float32)}

        dim_per_head = pre.Xh_red.shape[-1]
        row_indices = []
        for h in sorted(heads):
            start = h * dim_per_head
            row_indices.extend(range(start, start + dim_per_head))
        if metric == "whiten":
            full_context = metric_ctx_fullspace[full_key]
            mean_subset  = full_context["mu"][row_indices]
            whitening_matrix_subset  = full_context["Wg"][row_indices, :]
            ctx = {"Wg": whitening_matrix_subset.astype(np.float32), "mu": mean_subset.astype(np.float32)}
        else:
            full_context = metric_ctx_fullspace[full_key]
            projection_matrix_subset = full_context["W"][row_indices, :]
            ctx = {"W": projection_matrix_subset.astype(np.float32)}
    else:
        X = build_feat_subset(pre.Xh_red, heads).astype(np.float32)
        if metric == "whiten":
            mean_selected  = X.mean(axis=0, keepdims=False)
            centered_X     = X - mean_selected
            num_components = min(pca_dim, X.shape[1])
            pca_model      = PCA(n_components=num_components, svd_solver="auto", random_state=0).fit(centered_X)
            whitening_matrix = pca_model.components_.T / (np.sqrt(pca_model.explained_variance_ + 1e-8))
            ctx = {"Wg": whitening_matrix.astype(np.float32), "mu": mean_selected.astype(np.float32)}
        else:
            targets = pre.Y.astype(np.float32)
            if metric == "proj":
                ridge_model = Ridge(alpha=proj_alpha, fit_intercept=False, random_state=0).fit(X, targets)
                projection_matrix = ridge_model.coef_.T
                if projection_matrix.shape[0] != X.shape[1] and projection_matrix.shape[1] == X.shape[1]:
                    projection_matrix = projection_matrix.T
                ctx = {"W": projection_matrix.astype(np.float32)}
            else:
                num_components = min(pls_components, X.shape[1], targets.shape[1])
                pls_model = PLSRegression(n_components=num_components, scale=False)
                pls_model.fit(X, targets)
                projection_matrix = pls_model.coef_
                if projection_matrix.shape[0] != X.shape[1] and projection_matrix.shape[1] == X.shape[1]:
                    projection_matrix = projection_matrix.T
                ctx = {"W": projection_matrix.astype(np.float32)}

    metric_ctx_cache[key] = ctx
    return ctx


__all__ += [
    "build_metric_ctx_from_train",
    "build_global_metric_ctx_for_heads",
]


