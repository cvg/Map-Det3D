"""This module contains utilities for callbacks."""

from __future__ import annotations

import os
from typing import Any

import lightning.pytorch as pl

from mapdet3d.common.distributed import (
    all_gather_object_cpu,
    broadcast,
    rank_zero_only,
    synchronize,
)
from mapdet3d.common.logging import rank_zero_info
from mapdet3d.common.typing import ArgsType, MetricLogs
from mapdet3d.data.typing import DictData
from mapdet3d.eval.base import Evaluator

from .base import Callback


class EvaluatorCallback(Callback):
    """Callback for model evaluation."""

    def __init__(
        self,
        *args: ArgsType,
        evaluator: Evaluator,
        metrics_to_eval: list[str] | None = None,
        save_predictions: bool = False,
        save_prefix: None | str = None,
        output_dir: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            evaluator (Evaluator): Evaluator.
            metrics_to_eval (list[str], Optional): Metrics to evaluate. If
                None, all metrics in the evaluator will be evaluated. Defaults
                to None.
            save_predictions (bool): If the predictions should be saved.
                Defaults to False.
            save_prefix (str, Optional): Output directory for saving the
                evaluation results. Defaults to None.
            output_dir (str, Optional): Output directory for saving the
                evaluation results.
        """
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.save_predictions = save_predictions
        self.metrics_to_eval = metrics_to_eval or self.evaluator.metrics

        if self.save_predictions:
            assert (
                output_dir is not None
            ), "If save_predictions is True, save_prefix must be provided."

            output_dir = os.path.join(output_dir, "eval")

            self.output_dir = output_dir
            self.save_prefix = save_prefix

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_predictions:
            self.output_dir = broadcast(self.output_dir)

            if self.save_prefix is not None:
                self.output_dir = os.path.join(
                    self.output_dir, self.save_prefix
                )

            for metric in self.metrics_to_eval:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
        self.evaluator.reset()

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
        self.on_test_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""
        log_dict = self.run_eval()

        for k, v in log_dict.items():
            pl_module.log(f"val/{k}", v, sync_dist=True, rank_zero_only=True)

    def on_test_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch."""
        self.evaluator.process_batch(
            **self.get_test_callback_inputs(outputs, batch)
        )
        for metric in self.metrics_to_eval:
            # Save output predictions in current batch.
            if self.save_predictions:
                output_dir = os.path.join(self.output_dir, metric)
                self.evaluator.save_batch(metric, output_dir)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a testing epoch."""
        log_dict = self.run_eval()

        for k, v in log_dict.items():
            pl_module.log(f"test/{k}", v, sync_dist=True, rank_zero_only=True)

    def run_eval(self) -> MetricLogs:
        """Run evaluation for the given evaluator."""
        self.evaluator.gather(all_gather_object_cpu)

        synchronize()
        self.process()

        log_dict: MetricLogs = {}
        for metric in self.metrics_to_eval:
            metric_dict = self.evaluate(metric)
            metric_dict = broadcast(metric_dict)
            assert isinstance(metric_dict, dict)
            log_dict.update(metric_dict)

        self.evaluator.reset()

        return log_dict

    @rank_zero_only
    def process(self) -> None:
        """Process the evaluator."""
        self.evaluator.process()

    @rank_zero_only
    def evaluate(self, metric: str) -> MetricLogs:
        """Evaluate the performance after processing all input/output pairs.

        Returns:
            MetricLogs: A dictionary containing the evaluation results. The
                keys are formatted as {metric_name}/{key_name}, and the
                values are the corresponding evaluated values.
        """
        rank_zero_info(
            f"Running evaluator {str(self.evaluator)} with {metric} metric... "
        )
        log_dict = {}

        # Save output predictions. This is done here instead of
        # on_test_batch_end because the evaluator may not have processed
        # all batches yet.
        if self.save_predictions:
            output_dir = os.path.join(self.output_dir, metric)
            self.evaluator.save(metric, output_dir)

        # Evaluate metric
        metric_dict, metric_str = self.evaluator.evaluate(metric)
        for k, v in metric_dict.items():
            log_k = metric + "/" + k
            rank_zero_info("%s: %.4f", log_k, v)
            log_dict[f"{metric}/{k}"] = v

        rank_zero_info("Showing results for metric: %s", metric)
        rank_zero_info(metric_str)

        return log_dict
