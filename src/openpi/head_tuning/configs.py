"""Factory for creating head-tuning TrainConfigs.

This module provides :func:`make_head_tuning_config`, a convenience factory that
builds a :class:`~openpi.training.config.TrainConfig` for the **KNN attention-head
fine-tuning** recipe used in the openpi head-steering experiments.

The recipe fixes:
- Model: ``Pi0Config(action_horizon=16, pi05=True, action_dim=32,
  paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")``
- Optimizer: :class:`~openpi.training.optimizer.AdamWForHeadTuning` with
  ``freeze_kv=True, only_attention=False, freeze_mlp=False`` and a user-supplied
  list of ``(layer, head)`` pairs to train.
- Freeze filter: ``get_freeze_filter_always_freeze_expert_and_siglip()``
- Checkpoint: ``gs://openpi-assets/checkpoints/pi05_droid/params``
- LR schedule: cosine decay, ``peak_lr=2.5e-5``, 200 warm-up steps, decaying
  to ``2.5e-6`` over ``num_train_steps`` steps.

All imports of training-package symbols are done **lazily inside the function
body** to avoid a circular-import at module load time (``config.py`` imports
this module at the top level, so this module must not import ``config.py`` at
the top level).

Example usage::

    from openpi.head_tuning.configs import make_head_tuning_config

    cfg = make_head_tuning_config(
        name="my_task_head_tuning",
        repo_id="hf_user/my_droid_dataset_20",
        heads=[(9, 7), (9, 4), (8, 4), ...],  # 20 (layer, head) pairs
    )
"""

from __future__ import annotations

import json
import pathlib
from typing import Union


HeadsInput = Union[
    list[tuple[int, int]],
    list[list[int]],
    str,
    pathlib.Path,
]


def make_head_tuning_config(
    name: str,
    repo_id: str,
    heads: HeadsInput,
    *,
    num_train_steps: int = 3000,
    peak_lr: float = 2.5e-5,
    pi05: bool = True,
) -> "TrainConfig":  # noqa: F821  (resolved lazily)
    """Create a head-tuning :class:`~openpi.training.config.TrainConfig`.

    Parameters
    ----------
    name:
        Unique config name (used as the experiment identifier).
    repo_id:
        HuggingFace ``username/dataset`` for the LeRobot DROID dataset to
        fine-tune on.
    heads:
        Specifies which attention heads to train.  Accepts:

        * A ``list`` of ``(layer, head)`` integer pairs (or two-element lists).
        * A path (``str`` or :class:`pathlib.Path`) to a JSON file produced by
          :mod:`openpi.head_tuning.select` containing a
          ``"trainable_head_indices"`` key whose value is a list of
          ``[layer, head]`` pairs.
    num_train_steps:
        Total number of gradient steps.  Defaults to ``3000``.
    peak_lr:
        Peak learning rate for the cosine-decay schedule.  Defaults to
        ``2.5e-5``.
    pi05:
        If ``True`` (default), build a **pi05** model
        (``action_dim=32, action_horizon=16``).  Set to ``False`` to build a
        vanilla pi0 model (``action_horizon=16``); note that the DROID
        checkpoint URL is not changed automatically in that case.

    Returns
    -------
    TrainConfig
        A fully constructed training config ready to be registered in
        ``_CONFIGS``.
    """
    # ------------------------------------------------------------------ #
    # Lazy imports — avoid circular import with config.py                  #
    # ------------------------------------------------------------------ #
    from openpi.models import pi0_config as _pi0_config  # noqa: PLC0415
    from openpi.training import config as _cfg  # noqa: PLC0415
    from openpi.training import optimizer as _opt  # noqa: PLC0415
    from openpi.training import weight_loaders as _wl  # noqa: PLC0415

    # ------------------------------------------------------------------ #
    # Resolve heads                                                         #
    # ------------------------------------------------------------------ #
    if isinstance(heads, (str, pathlib.Path)):
        path = pathlib.Path(heads)
        with path.open() as fh:
            data = json.load(fh)
        raw_pairs = data["trainable_head_indices"]
    else:
        raw_pairs = heads

    trainable_head_indices: list[tuple[int, int]] = [
        (int(p[0]), int(p[1])) for p in raw_pairs
    ]

    # ------------------------------------------------------------------ #
    # Model config                                                          #
    # ------------------------------------------------------------------ #
    if pi05:
        model_cfg = _pi0_config.Pi0Config(
            action_horizon=16,
            pi05=True,
            action_dim=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        checkpoint_url = "gs://openpi-assets/checkpoints/pi05_droid/params"
    else:
        model_cfg = _pi0_config.Pi0Config(
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        checkpoint_url = "gs://openpi-assets/checkpoints/pi0_droid/params"

    # ------------------------------------------------------------------ #
    # Assemble TrainConfig                                                  #
    # ------------------------------------------------------------------ #
    decay_lr = peak_lr / 10.0

    return _cfg.TrainConfig(
        name=name,
        model=model_cfg,
        data=_cfg.LeRobotDROIDDataConfig(
            repo_id=repo_id,
            base_config=_cfg.DataConfig(prompt_from_task=True),
        ),
        optimizer=_opt.AdamWForHeadTuning(
            freeze_kv=True,
            only_attention=False,
            freeze_mlp=False,
            trainable_head_indices=trainable_head_indices,
        ),
        freeze_filter=model_cfg.get_freeze_filter_always_freeze_expert_and_siglip(),
        weight_loader=_wl.CheckpointWeightLoader(checkpoint_url),
        lr_schedule=_opt.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
            decay_lr=decay_lr,
        ),
        num_train_steps=num_train_steps,
        batch_size=32,
        num_workers=8,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        ema_decay=None,
    )
