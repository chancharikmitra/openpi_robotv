"""Stage-1 CLI: extract per-head last-token activations from a pi05 policy.

Usage examples::

    # DROID benchmark (H5 input)
    python -m openpi.head_tuning.extract \\
        --benchmark droid \\
        --input-h5 temp_data/swap_green_red_cube_20.h5 \\
        --out-h5 activations.h5 \\
        --task-prompt "swap green red cube" \\
        --max-episodes 2

    # LIBERO benchmark (parquet input — task-prompt is optional)
    python -m openpi.head_tuning.extract \\
        --benchmark libero \\
        --src temp_data/libero10_task0_20 \\
        --out-h5 libero_activations.h5 \\
        --max-episodes 5
"""
from __future__ import annotations

import dataclasses
import os

import tyro

from openpi.head_tuning.adapters import base as _base
from openpi.head_tuning.adapters import droid as _droid
from openpi.head_tuning.adapters import libero as _libero
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Command-line arguments for the activation-extraction CLI."""

    benchmark: str
    """Benchmark to extract activations for: ``"droid"`` or ``"libero"``."""

    input_h5: str = ""
    """(droid only) Path to the input DROID HDF5 file."""

    src: str = ""
    """(libero only) Root directory of the LeRobot v2.0 dataset."""

    out_h5: str = "activations.h5"
    """Path of the output HDF5 file to write."""

    task_prompt: str = ""
    """Language instruction string.  For droid: required; used as the task-name
    key in the output HDF5 (overrides any embedded instruction).  For libero:
    optional override — when non-empty, every episode uses this string as its
    prompt; when empty (default), each episode's prompt is read from the
    per-episode ``tasks`` field in ``meta/episodes.jsonl``."""

    max_episodes: int = 200
    """Maximum number of episodes to process (0 = all)."""

    use_keyframe: bool = True
    """If True, apply the benchmark-specific keyframe selection rule.
    If False, process all frames."""


_BENCHMARK_CONFIG: dict[str, tuple[str, str]] = {
    "droid":  ("pi05_droid",  "gs://openpi-assets/checkpoints/pi05_droid"),
    "libero": ("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
}


def main(args: Args) -> None:
    """Load policy, load episodes, run inference, and write activations to H5."""
    os.environ.setdefault("OPENPI_DATA_HOME", "/scr2/yusenluo/openpi_robotv/.cache/openpi")

    if args.benchmark not in _BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark {args.benchmark!r}; must be 'droid' or 'libero'.")

    config_name, gcs_dir = _BENCHMARK_CONFIG[args.benchmark]
    cfg = _config.get_config(config_name)
    ckpt = download.maybe_download(gcs_dir)
    download.maybe_download(gcs_dir + "/assets")
    policy = policy_config.create_trained_policy(cfg, ckpt)

    max_ep = args.max_episodes if args.max_episodes > 0 else None

    if args.benchmark == "droid":
        episodes = _droid.load_droid_episodes(args.input_h5, args.task_prompt, max_ep)
        key_fn = _droid.droid_key_idcs
    elif args.benchmark == "libero":
        episodes = _libero.load_libero_episodes(args.src, args.task_prompt, max_ep)
        key_fn = _libero.libero_key_idcs
    else:
        raise ValueError(f"Unknown benchmark {args.benchmark!r}; must be 'droid' or 'libero'.")

    _base.run_inference_and_save(
        policy,
        episodes,
        args.out_h5,
        key_fn=key_fn if args.use_keyframe else None,
    )
    print(f"[extract] Done → {args.out_h5}")


if __name__ == "__main__":
    main(tyro.cli(Args))
