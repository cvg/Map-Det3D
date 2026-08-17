"""Evaluation base class."""

from __future__ import annotations

from mapdet3d.common.typing import GenericFunc, MetricLogs, unimplemented


class Evaluator:  # pragma: no cover
    """Abstract evaluator class.

    The evaluator is responsible for evaluating the model on a given dataset.
    At each end of batches, the process_batch() is called with the model
    outputs and the batch data to accumulate the data for evaluation. An
    optional save_batch() can be implemented to save the predictions in the
    current batch.

    After all batches are processed, the gather() method is called to gather
    the data from all ranks. Then, the process() method is used to process all
    the accumulated data that are metrics-independent. Finally, the evaluate()
    method is called to evaluate the model for the specified metrics and return
    the results. Optionally, the save() method can be implemented to save the
    predictions for the specified metrics.

    The following diagram illustrates the evaluation process::

                      RANK 0                          RANK 1                  ...

        x num_batches
        ┌────────────────────────────────────────────────────────────────┐
        │  ┌──────────────────────────┐    ┌──────────────────────────┐  │
        │  │ process_batch(data, ...) │    │ process_batch(data, ...) │  │ <- Process a batch (predictions, labels, etc.)
        │  └──────────────────────────┘    └──────────────────────────┘  │    and accumulate the data for evaluation.
        │                ▼                              ▼                │
        │     ┌────────────────────┐          ┌────────────────────┐     │
        │     │ save_batch(metric) │          │ save_batch(metric) │     │ <- Dump the predictions in a batch for a specified
        │     └────────────────────┘          └────────────────────┘     │    metric (e.g., for online evaluation).
        └────────────────┬──────────────────────────────┬────────────────┘
                   ┌─────┴────┐                         │
                   │ gather() ├─────────────────────────┘
                   └──────────┘      <- Gather the data from all ranks
                         ▼
                   ┌───────────┐
                   │ process() │     <- Process the data that are
                   └───────────┘        metrics-independent (if any)
                         ▼
               ┌──────────────────┐
               │ evaluate(metric) │  <- Evaluate for a specified metric and
               └──────────────────┘    return the results.
                         ▼
                 ┌──────────────┐
                 │ save(metric) │    <- Dump the predictions for a specified
                 └──────────────┘       metric (e.g., for online evaluation).

    Note:
        The save_batch() saves the predictions every batch, which is helpful
        for reducing the memory usage, compared to saving all predictions at
        once in the save() method. However, the save_batch() is optional and
        can be omitted if the data can be saved only after all batches are
        processed.
    """  # pylint: disable=line-too-long

    @property
    def metrics(self) -> list[str]:
        """Return list of metrics to evaluate.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return []

    def gather(self, gather_func: GenericFunc) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError

    # Process a batch of data.
    process_batch: GenericFunc = unimplemented

    def process(self) -> None:
        """Process all accumulated data at the end of an epoch, if any."""

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate all predictions according to given metric.

        Args:
            metric (str): Metric to evaluate.

        Raises:
            NotImplementedError: This is an abstract class method.

        Returns:
            tuple[MetricLogs, str]: Dictionary of scores to log and a pretty
                printed string.
        """
        raise NotImplementedError

    def save_batch(self, metric: str, output_dir: str) -> None:
        """Save batch of predictions to file.

        Args:
            metric (str): Save predictions for the specified metrics.
            output_dir (str): Output directory.
        """

    def save(self, metric: str, output_dir: str) -> None:
        """Save all predictions to file at the end of an epoch.

        Args:
            metric (str): Save predictions for the specified metrics.
            output_dir (str): Output directory.
        """
