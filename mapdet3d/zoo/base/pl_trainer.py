"""Default runtime configuration for PyTorch Lightning."""

import inspect

from lightning import Trainer

from mapdet3d.config import FieldConfigDict
from mapdet3d.config.typing import ExperimentConfig


def get_default_pl_trainer_cfg(config: ExperimentConfig) -> ExperimentConfig:
    """Get PyTorch Lightning Trainer config."""
    pl_trainer = FieldConfigDict()

    # PL Trainer arguments
    for k, v in inspect.signature(Trainer).parameters.items():
        if not k in {"callbacks", "devices", "logger", "strategy"}:
            pl_trainer[k] = v.default

    # PL Trainer settings
    pl_trainer.benchmark = config.benchmark
    pl_trainer.use_distributed_sampler = False
    pl_trainer.num_sanity_val_steps = 0

    # logger
    pl_trainer.enable_progress_bar = False
    pl_trainer.log_every_n_steps = config.log_every_n_steps

    # Default Trainer arguments
    pl_trainer.work_dir = config.work_dir
    pl_trainer.exp_name = config.experiment_name
    pl_trainer.version = config.version
    pl_trainer.find_unused_parameters = False
    pl_trainer.checkpoint_period = 1
    pl_trainer.save_top_k = 1
    pl_trainer.wandb = False

    return pl_trainer
