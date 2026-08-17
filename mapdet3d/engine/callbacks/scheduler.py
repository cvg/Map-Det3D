"""Callback to configure learning rate during training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import lightning.pytorch as pl

from mapdet3d.engine.optim.scheduler import LRSchedulerWrapper

from .base import Callback


class LRSchedulerCallback(Callback):
    """Callback to configure learning rate during training."""

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.last_step = 0

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook on training batch end."""
        schedulers = pl_module.lr_schedulers()

        if not isinstance(schedulers, Iterable):
            schedulers = [schedulers]  # type: ignore

        if trainer.global_step != self.last_step:
            for scheduler in schedulers:
                if scheduler is None:
                    continue
                assert isinstance(scheduler, LRSchedulerWrapper)
                scheduler.step_on_batch(trainer.global_step)

                self.last_step = trainer.global_step
