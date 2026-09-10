"""Stage 2 CLI: KNN-based attention head selection.

Reads an activation H5 (produced by `openpi.head_tuning.extract`) and writes a
heads.json describing the selected (layer, head) indices for head-tuning.
"""
import dataclasses, json
import h5py, tyro
from openpi.head_tuning.knn.model import fit_knn_reg_with_heads


@dataclasses.dataclass
class Args:
    attn_h5: str
    """Path to the activation H5 produced by the extract stage."""
    out: str = "heads.json"
    """Output path for the selected-heads JSON."""
    unit_mode: str = "head"
    """'head' (144 heads) or 'layer' (18 layers)."""
    target: int = 20
    """Number of heads (or layers) to select."""
    metric: str = "cosine"
    """Distance metric: 'cosine' or 'euclidean'."""
    selection_mode: str = "topk"
    """'topk' or 'best_add'."""
    k_grid: tuple[int, ...] = (10, 20, 30, 40)
    """Candidate k values for KNN."""
    temp_excl_w: int = 30
    """Leave-one-frame-out temporal exclusion window (+/- frames)."""


def main(args: Args) -> None:
    with h5py.File(args.attn_h5, "r") as f:
        episodes = [f"{t}/{e}" for t in f.keys() for e in f[t].keys()]
    _, info = fit_knn_reg_with_heads(
        args.attn_h5, episodes, selection_mode=args.selection_mode,
        target_heads=args.target, unit_mode=args.unit_mode,
        k_grid=list(args.k_grid), metric=args.metric, temp_excl_w=args.temp_excl_w,
    )
    sel = info["selected_heads"]
    heads = ([[l, h] for l in sel for h in range(8)] if args.unit_mode == "layer"
             else [[idx // 8, idx % 8] for idx in sel])
    out = {"trainable_head_indices": heads, "best_k": info["best_k"],
           "cv_mse": info["cv_mse"], "unit_mode": args.unit_mode,
           "target": args.target, "attn_h5": args.attn_h5}
    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=2)
    print(f"[select] wrote {args.out}: best_k={info['best_k']} cv_mse={info['cv_mse']:.6f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
