"""This module contains utilities for callbacks."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import numpy as np

from mapdet3d.common.typing import ArgsType

from .base import Callback


class RandomSeqCallback(Callback):
    """Callback for random seq length."""

    def __init__(
        self, *args: ArgsType, seq_len: int, **kwargs: ArgsType
    ) -> None:
        """Init callback."""
        super().__init__(*args, **kwargs)
        self._seq_len = seq_len

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        num_ref_samples = np.random.randint(1, self._seq_len)
        for mv_d in trainer.train_dataloader.dataset.datasets:
            mv_d.num_ref_samples = num_ref_samples
