"""This module contains logging utility functions.

We provide utilities for setting up a logger and logging in a distributed
setting.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

from termcolor import colored

from mapdet3d.common.distributed import rank_zero_only
from mapdet3d.common.typing import ArgsType
from mapdet3d.config.typing import ExperimentConfig


def _debug(*args: ArgsType, stacklevel: int = 2, **kwargs: ArgsType) -> None:
    """Function used to log debug-level messages."""
    log = logging.getLogger(__name__)
    kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(
    *args: ArgsType, stacklevel: int = 4, **kwargs: ArgsType
) -> None:
    """Function used to log debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: ArgsType, stacklevel: int = 2, **kwargs: ArgsType) -> None:
    """Function used to log info-level messages."""
    kwargs["stacklevel"] = stacklevel
    log = logging.getLogger(__name__)
    log.info(*args, **kwargs)


@rank_zero_only
def rank_zero_info(
    *args: ArgsType, stacklevel: int = 4, **kwargs: ArgsType
) -> None:
    """Function used to log info-level messages only on global rank 0."""
    _info(*args, stacklevel=stacklevel, **kwargs)


def _warn(
    message: str | Warning, stacklevel: int = 2, **kwargs: ArgsType
) -> None:
    """Function used to log warn-level messages."""
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(
    message: str | Warning, stacklevel: int = 4, **kwargs: ArgsType
) -> None:
    """Function used to log warn-level messages only on global rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)


class _ColorFormatter(logging.Formatter):
    """Formatter for terminal messages with colors."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Add appropriate color to log message."""
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno in [logging.ERROR, logging.CRITICAL]:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@rank_zero_only
def setup_logger(
    logger: logging.Logger,
    filepath: None | str = None,
    color: bool = True,
    std_out_level: int = logging.INFO,
) -> None:
    """Configure logging.

    Args:
        logger (logging.Logger): The logger instance to be configured.
        filepath (None | str, optional): The filepath to the log file that
            stores the console output. Defaults to None.
        color (bool, optional): Whether to use a colored console output.
            Defaults to True.
        std_out_level (int, optional): Which logging level to output to the
            console. Defaults to logging.INFO. Note that all levels will be
            logged to file.
    """
    # get logger, remove handlers to re-define behavior
    for h in logger.handlers:
        logger.removeHandler(h)

    # console logger
    plain_formatter = logging.Formatter(
        "[%(asctime)s] mapdet3d %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(std_out_level)
    if color:
        formatter = _ColorFormatter(
            colored("[%(asctime)s mapdet3d]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
    else:
        ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    # file logger
    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fh = logging.FileHandler(filepath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


@rank_zero_only
def dump_config(config: ExperimentConfig, config_file: str) -> None:
    """Dump the configuration to a file.

    Args:
        config (ExperimentConfig): The configuration to dump.
        config_file (str): The path to the file to dump the configuration to.
    """
    config.dump(config_file)
