"""Definitions of loss reducers.

Loss reducers are usually used as the last step in loss computation to average
or sum the loss maps from dense predictions or object detections.
"""

from __future__ import annotations

from typing import Callable

from torch import Tensor

LossReducer = Callable[[Tensor], Tensor]


def identity_loss(loss: Tensor) -> Tensor:
    """Make no change to the loss."""
    return loss


def mean_loss(loss: Tensor) -> Tensor:
    """Average the loss tensor values to a single value.

    Args:
        loss (Tensor): Input multi-dimentional tensor.

    Returns:
        Tensor: Tensor containing a single loss value.
    """
    return loss.mean()


def sum_loss(loss: Tensor) -> Tensor:
    """Sum the loss tensor values to a single value.

    Args:
        loss (Tensor): Input multi-dimentional tensor.

    Returns:
        Tensor: Tensor containing a single loss value.
    """
    return loss.sum()


class SumWeightedLoss:
    """A loss reducer to calculated weighted sum loss."""

    def __init__(
        self, weight: float | Tensor, avg_factor: float | Tensor
    ) -> None:
        """Initialize the loss reducer.

        Args:
            weight (float | Tensor): Weights for each loss elements
            avg_factor (float | Tensor): average factor for the weighted loss
        """
        self.weight = weight
        self.avg_factor = avg_factor

    def __call__(self, loss: Tensor) -> Tensor:
        """Weight the loss elements and take the sum with the average factor.

        Args:
            loss (Tensor): input loss

        Returns:
            Tensor: output loss
        """
        return (loss * self.weight).sum() / self.avg_factor
