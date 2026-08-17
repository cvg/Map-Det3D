"""This module contains utilities for callbacks."""

from __future__ import annotations

import os
from typing import Any

import lightning.pytorch as pl

from mapdet3d.common.distributed import broadcast, synchronize
from mapdet3d.common.typing import ArgsType
from mapdet3d.vis.base import Visualizer

from .base import Callback


class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        *args: ArgsType,
        visualizer: Visualizer,
        visualize_train: bool = False,
        show: bool = False,
        save_to_disk: bool = True,
        save_prefix: str | None = None,
        output_dir: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            visualize_train (bool): If the training data should be visualized.
                Defaults to False.
            show (bool): If the visualizations should be shown. Defaults to
                False.
            save_to_disk (bool): If the visualizations should be saved to disk.
                Defaults to True.
            save_prefix (str): Output directory prefix for distinguish
                different visualizations.
            output_dir (str): Output directory for saving the visualizations.
        """
        super().__init__(*args, **kwargs)
        self.visualizer = visualizer
        self.visualize_train = visualize_train
        self.save_prefix = save_prefix
        self.show = show
        self.save_to_disk = save_to_disk

        if self.save_to_disk:
            assert (
                output_dir is not None
            ), "If save_to_disk is True, output_dir must be provided."

            output_dir = os.path.join(output_dir, "vis")

            self.output_dir = output_dir
            self.save_prefix = save_prefix

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_to_disk:
            self.output_dir = broadcast(self.output_dir)

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        cur_iter = batch_idx + 1

        if self.visualize_train:
            self.visualizer.process(
                cur_iter=cur_iter,
                **self.get_train_callback_inputs(outputs, batch),
            )

            if self.show:
                self.visualizer.show(cur_iter=cur_iter)

            if self.save_to_disk:
                self.save(cur_iter=cur_iter, stage="train")

            self.visualizer.reset()

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a validation batch."""
        cur_iter = batch_idx + 1

        self.visualizer.process(
            cur_iter=cur_iter,
            **self.get_test_callback_inputs(outputs, batch),
        )

        if self.show:
            self.visualizer.show(cur_iter=cur_iter)

        if self.save_to_disk:
            self.save(cur_iter=cur_iter, stage="val")

        self.visualizer.reset()

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch."""
        cur_iter = batch_idx + 1

        self.visualizer.process(
            cur_iter=cur_iter,
            **self.get_test_callback_inputs(outputs, batch),
        )

        if self.show:
            self.visualizer.show(cur_iter=cur_iter)

        if self.save_to_disk:
            self.save(cur_iter=cur_iter, stage="test")

        self.visualizer.reset()

    def save(self, cur_iter: int, stage: str) -> None:
        """Save the visualizer state."""
        output_folder = os.path.join(self.output_dir, stage)

        if self.save_prefix is not None:
            output_folder = os.path.join(output_folder, self.save_prefix)

        os.makedirs(output_folder, exist_ok=True)

        self.visualizer.save_to_disk(
            cur_iter=cur_iter, output_folder=output_folder
        )

        # TODO: Add support for logging images to WandB.
        # if get_rank() == 0:
        #     if isinstance(trainer.logger, WandbLogger) and image is not None:
        #         trainer.logger.log_image(
        #             key=f"{self.visualizer}/{cur_iter}",
        #             images=[image],
        #         )

        synchronize()
