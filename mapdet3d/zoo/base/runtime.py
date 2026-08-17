"""Default runtime configuration for the project."""

from __future__ import annotations

import platform
from datetime import datetime

from ml_collections import ConfigDict

from mapdet3d.config import class_config
from mapdet3d.config.typing import ExperimentConfig
from mapdet3d.engine.callbacks import LoggingCallback


def get_default_cfg(
    exp_name: str, work_dir: str = "work_dir"
) -> ExperimentConfig:
    """Set default config for the project.

    Args:
        exp_name (str): Experiment name.
        work_dir (str, optional): Working directory. Defaults to "work_dir".

    Returns:
        ExperimentConfig: Config for the project.
    """
    config = ExperimentConfig()

    config.work_dir = work_dir
    config.experiment_name = exp_name

    timestamp = (
        str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace(":", "-")
    )
    config.timestamp = timestamp
    config.version = timestamp

    if platform.system() == "Windows":
        path_component = "\\"
    else:
        path_component = "/"

    config.output_dir = (
        config.work_dir
        + path_component
        + config.experiment_name
        + path_component
        + config.version
    )

    # Set default value for the following fields
    config.seed = -1
    config.log_every_n_steps = 50
    config.use_tf32 = False
    config.tf32_matmul_precision = "highest"
    config.benchmark = False
    config.check_unused_parameters = False

    return config


def get_default_callbacks_cfg(
    epoch_based: bool = True,
    refresh_rate: int = 50,
) -> list[ConfigDict]:
    """Get default callbacks config.

    It will return a list of callbacks config including:
        - LoggingCallback

    Args:
        epoch_based (bool, optional): Whether to use epoch based logging.
        refresh_rate (int, optional): Refresh rate for the logging. Defaults to
            50.

    Returns:
        list[ConfigDict]: List of callbacks config.
    """
    callbacks = []

    # Logger
    callbacks.append(
        class_config(
            LoggingCallback, epoch_based=epoch_based, refresh_rate=refresh_rate
        )
    )

    return callbacks
