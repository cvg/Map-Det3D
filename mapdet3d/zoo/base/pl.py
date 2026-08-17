"""PyTorch Lightning config."""

from __future__ import annotations

from ml_collections import ConfigDict

from mapdet3d.config.typing import ExperimentConfig, ExperimentParameters
from mapdet3d.zoo.base import get_default_pl_trainer_cfg


def get_pl_cfg(
    config: ExperimentConfig,
    params: ExperimentParameters,
    epoch_based: bool = True,
    bf16_mixed: bool = False,
    gradient_clip_val: float = 1.0,
) -> ConfigDict:
    """Returns the PyTorch Lightning configuration."""
    pl_trainer = get_default_pl_trainer_cfg(config)

    if bf16_mixed:
        pl_trainer.precision = "bf16-mixed"

    if epoch_based:
        pl_trainer.max_epochs = params.num_epochs
        pl_trainer.check_val_every_n_epoch = params.check_val_every_n_epoch
    else:
        pl_trainer.epoch_based = False
        pl_trainer.max_steps = params.num_iters
        pl_trainer.checkpoint_period = params.checkpoint_period
        pl_trainer.val_check_interval = params.val_freq
        pl_trainer.check_val_every_n_epoch = None

    pl_trainer.gradient_clip_val = gradient_clip_val
    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches

    return pl_trainer
