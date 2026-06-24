"""
KNN data loading and preprocessing utilities.

Handles loading attention/action data from HDF5 files, per-head z-score and PCA
preprocessing, dataset construction, and action label export.
"""

import h5py
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _coerce_float32(arr: np.ndarray) -> np.ndarray:
    """Force convert array to float32."""
    if arr.dtype.kind == 'V' and arr.itemsize == 2:
        return arr.view('<f2').astype(np.float32)
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def _iter_frame_keys(group: h5py.Group):
    """Sort frame keys by numeric order."""
    keys = list(group.keys())
    try:
        keys_sorted = sorted(keys, key=lambda k: int(k.split("_")[-1]) if "_" in k else int(k))
    except Exception:
        keys_sorted = sorted(keys)
    return keys_sorted


def load_episode_frames(h5_file: h5py.File, episode_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all frames for a given episode.
    Returns:
      attention_data: (F, 18, 8, 256)
      action_data: (F, A) or (F,1) if scalar
    """
    group = h5_file[episode_key]
    attention_list, action_list = [], []
    for frame_key in _iter_frame_keys(group):
        frame_group = group[frame_key]
        if "first_action_token_attn" not in frame_group:
            raise KeyError(f"{episode_key}/{frame_key} missing 'first_action_token_attn'")
        attention = _coerce_float32(np.asarray(frame_group["first_action_token_attn"]))   # (18,8,256)

        # Prefer new-format action_label (8 dims for droid, 7 for libero, ...)
        if "action_label" in frame_group:
            action = _coerce_float32(np.asarray(frame_group["action_label"]))
            action = action.reshape(-1)
            if action.size == 0:
                raise ValueError(f"{episode_key}/{frame_key} 'action_label' empty")

        # Fall back to legacy joint_velocity(7) + gripper_position(1)
        elif "joint_velocity" in frame_group and "gripper_position" in frame_group:
            jv = _coerce_float32(np.asarray(frame_group["joint_velocity"]))  # (7,)
            gp = _coerce_float32(np.asarray(frame_group["gripper_position"]))  # (1,) or scalar
            jv = jv.reshape(-1)
            gp = gp.reshape(-1)
            if jv.size != 7:
                raise ValueError(f"{episode_key}/{frame_key} 'joint_velocity' must have 7 elements, got {jv.shape}")
            if gp.size == 0:
                raise ValueError(f"{episode_key}/{frame_key} 'gripper_position' empty")
            gp = gp[:1]
            action = np.concatenate([jv, gp], axis=0)

        # Further fall back to pi_droid_action (should be 8 dims)
        elif "pi_droid_action" in frame_group:
            action = _coerce_float32(np.asarray(frame_group["pi_droid_action"]))
            action = action.reshape(-1)
            if action.size != 8:
                raise ValueError(f"{episode_key}/{frame_key} 'pi_droid_action' must have 8 elements, got {action.shape}")

        else:
            raise KeyError(
                f"{episode_key}/{frame_key} missing action fields (expected 'action_label' or 'joint_velocity'+'gripper_position' or 'pi_droid_action')"
            )

        attention_list.append(attention)
        action_list.append(action)

    attention_data = np.stack(attention_list, axis=0)  # (F,18,8,256)
    action_data = np.stack(action_list, axis=0)   # (F, A?) or (F,)
    if action_data.ndim == 1:
        action_data = action_data[:, None]
    return attention_data, action_data


def frame_to_vec(frame_attention: np.ndarray, num_layers: int = 18, heads_per_layer: int = 8, d_head: int = 256) -> np.ndarray:
    """Flatten a single frame attention tensor: (18,8,256) -> (144,256)"""
    assert frame_attention.shape == (num_layers, heads_per_layer, d_head)
    return frame_attention.reshape(num_layers * heads_per_layer, d_head)


@dataclass
class HeadPreprocessor:
    """Per-head preprocessor."""
    scaler: StandardScaler
    pca: Optional[PCA]


class IdentityScaler:
    """Identity transform aligned to StandardScaler API; implements transform only."""
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


@dataclass
class PreprocessedData:
    """Preprocessed dataset."""
    Xh_red: np.ndarray          # (N, 144, d)  d = PCA_D or 256
    Y: np.ndarray               # (N, A)
    ep_ids: np.ndarray          # (N,)
    t_ids: np.ndarray           # (N,)
    preproc: List[HeadPreprocessor]
    episodes: List[str]

    # ---- Human-friendly aliases (without changing original fields) ----
    @property
    def features(self) -> np.ndarray:
        return self.Xh_red

    @property
    def actions(self) -> np.ndarray:
        return self.Y

    @property
    def episode_ids(self) -> np.ndarray:
        return self.ep_ids

    @property
    def frame_ids(self) -> np.ndarray:
        return self.t_ids


def build_dataset(
    attn_h5: str,
    episodes: List[str],
    use_pca: bool = False,
    use_zscore: bool = False,
    pca_components: int = 32,
    num_layers: int = 18,
    heads_per_layer: int = 8,
    d_head: int = 256
) -> PreprocessedData:
    """
    Build the full dataset and apply per-head preprocessing (z-score + optional PCA).
    """
    total_heads = num_layers * heads_per_layer

    with h5py.File(attn_h5, "r") as f:
        total_frames = sum(len(f[ep].keys()) for ep in episodes)

    X_all = np.empty((total_frames, total_heads, d_head), dtype=np.float32)
    Y_all = []

    episodes_sorted = sorted(episodes)
    episode_to_id = {ep: i for i, ep in enumerate(episodes_sorted)}
    episode_ids = np.empty((total_frames,), dtype=np.int32)
    frame_ids = np.empty((total_frames,), dtype=np.int32)

    ptr = 0
    with h5py.File(attn_h5, "r") as f:
        for ep in tqdm(episodes, desc="Load episodes"):
            attention, actions = load_episode_frames(f, ep)     # (F,18,8,256), (F,A)
            F = attention.shape[0]
            X_all[ptr:ptr+F] = attention.reshape(F, total_heads, d_head)
            Y_all.append(actions.astype(np.float32))
            episode_ids[ptr:ptr+F] = episode_to_id[ep]
            frame_ids[ptr:ptr+F] = np.arange(F, dtype=np.int32)
            ptr += F

    Y = np.concatenate(Y_all, axis=0)  # (N, A)
    d_output = pca_components if use_pca else d_head
    Xh_reduced = np.empty((X_all.shape[0], total_heads, d_output), dtype=np.float32)
    preprocessors: List[HeadPreprocessor] = []

    for head in tqdm(range(total_heads), desc="Per-head zscore+PCA"):
        head_features = X_all[:, head, :]                          # (N,256)
        if use_zscore:
            scaler = StandardScaler().fit(head_features)
            standardized_features = scaler.transform(head_features)
        else:
            scaler = IdentityScaler()
            standardized_features = head_features

        if use_pca:
            pca = PCA(n_components=pca_components, svd_solver="auto", random_state=0).fit(standardized_features)
            reduced_features = pca.transform(standardized_features).astype(np.float32)  # (N,d_output)
        else:
            pca = None
            reduced_features = standardized_features.astype(np.float32)

        Xh_reduced[:, head, :] = reduced_features
        preprocessors.append(HeadPreprocessor(scaler, pca))

    return PreprocessedData(
        Xh_red=Xh_reduced, Y=Y, ep_ids=episode_ids, t_ids=frame_ids,
        preproc=preprocessors, episodes=episodes_sorted
    )


def transform_frame(frame_attention: np.ndarray, model, num_layers: int = 18, heads_per_layer: int = 8, d_head: int = 256) -> np.ndarray:
    """Transform a single frame using the trained model's preprocessing."""
    attention_vector = frame_to_vec(frame_attention, num_layers, heads_per_layer, d_head)  # (144,256)
    feature_parts = []
    for head in model.heads:
        transformed = model.preproc[head].scaler.transform(attention_vector[head][None, :])
        if model.preproc[head].pca is not None:
            transformed = model.preproc[head].pca.transform(transformed)
        feature_parts.append(transformed.astype(np.float32))
    return np.concatenate(feature_parts, axis=1).reshape(-1)  # (|selected_heads|*d,)


def transform_episode(attn_frames: np.ndarray, model, num_layers: int = 18, heads_per_layer: int = 8, d_head: int = 256) -> np.ndarray:
    """
    Transform all frames in an episode using the trained model's preprocessing.
    Inputs:
      - attn_frames: (F, 18, 8, 256)
    Returns:
      - (F, |selected_heads| * d)
    """
    F = attn_frames.shape[0]
    flat = attn_frames.reshape(F, num_layers * heads_per_layer, d_head)  # (F, 144, 256)
    parts = []
    for head in model.heads:
        feat = flat[:, head, :]                                         # (F,256)
        feat = model.preproc[head].scaler.transform(feat)               # (F,256)
        if model.preproc[head].pca is not None:
            feat = model.preproc[head].pca.transform(feat)              # (F,d)
        parts.append(feat.astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)             # (F, |S|*d)


def get_action_labels(
    attn_h5: str,
    episodes: List[str],
    return_metadata: bool = True
):
    """
    Read action labels for the given episodes.
    Returns:
      - actions: np.ndarray (N, A)
      - meta (optional): { 'episode_ids', 'frame_ids', 'episodes' }
    """
    with h5py.File(attn_h5, "r") as f:
        total_frames = sum(len(f[ep].keys()) for ep in episodes)

    with h5py.File(attn_h5, "r") as f:
        actions_list = []
        episodes_sorted = sorted(episodes)
        ep2id = {ep: i for i, ep in enumerate(episodes_sorted)}
        episode_ids = np.empty((total_frames,), dtype=np.int32)
        frame_ids = np.empty((total_frames,), dtype=np.int32)

        ptr = 0
        for ep in episodes:
            attn, act = load_episode_frames(f, ep)  # act: (F, A)
            F = act.shape[0]
            actions_list.append(act.astype(np.float32))
            episode_ids[ptr:ptr+F] = ep2id[ep]
            frame_ids[ptr:ptr+F] = np.arange(F, dtype=np.int32)
            ptr += F

    actions = np.concatenate(actions_list, axis=0)
    if not return_metadata:
        return actions
    return actions, {
        "episode_ids": episode_ids,
        "frame_ids": frame_ids,
        "episodes": episodes_sorted,
    }


def save_action_labels(
    attn_h5: str,
    episodes: List[str],
    out_path: str,
    fmt: str = "npy"
):
    """
    Export action labels for the given episodes.
    - fmt="npy": save a .npz with (actions, episode_ids, frame_ids, episodes)
    - fmt="csv": save a .csv with columns: episode, frame, action_0 ... action_{A-1}
    """
    actions, meta = get_action_labels(attn_h5, episodes, return_metadata=True)
    if fmt.lower() == "npy":
        np.savez(out_path, actions=actions, **meta)
        return out_path

    if fmt.lower() == "csv":
        import csv
        episodes_sorted = meta["episodes"]
        episode_ids = meta["episode_ids"]
        frame_ids = meta["frame_ids"]
        A = actions.shape[1]
        header = ["episode", "frame"] + [f"action_{i}" for i in range(A)]
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(actions.shape[0]):
                ep_name = episodes_sorted[int(episode_ids[i])]
                row = [ep_name, int(frame_ids[i])] + list(map(float, actions[i]))
                writer.writerow(row)
        return out_path

    raise ValueError("fmt must be 'npy' or 'csv'")


__all__ = [
    "_coerce_float32",
    "_iter_frame_keys",
    "load_episode_frames",
    "frame_to_vec",
    "HeadPreprocessor",
    "IdentityScaler",
    "PreprocessedData",
    "build_dataset",
    "transform_frame",
    "transform_episode",
    "get_action_labels",
    "save_action_labels",
]
