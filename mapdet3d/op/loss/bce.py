"""Binary cross entropy loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mapdet3d.common.typing import ArgsType
from mapdet3d.op.loss.base import Loss
from mapdet3d.op.loss.reducer import LossReducer


class BinaryCrossEntropyLoss(Loss):
    """Binary cross entropy loss class."""

    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        """Creates an instance of the class."""
        super().__init__(*args, **kwargs)

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        reducer: LossReducer | None = None,
        ignore_index: int = 255,
    ) -> Tensor:
        """Forward pass.

        Args:
            output (list[Tensor]): Model output.
            target (Tensor): Assigned segmentation target mask.
            reducer (LossReducer, optional): Reducer for the loss function.
                Defaults to None.
            ignore_index (int): Ignore class id. Default to 255.

        Returns:
            Tensor: Computed loss.
        """
        reducer = reducer or self.reducer

        return reducer(
            binary_cross_entropy(output, target, ignore_index=ignore_index)
        )


def binary_cross_entropy(
    pred: Tensor, label: Tensor, ignore_index: int = 255
) -> Tensor:
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (Tensor): The prediction with shape (N, 1).
        label (Tensor): The learning label of the prediction.
        ignore_index (int, optional): The label index to be ignored. Default:
            255.

    Returns:
        Tensor: The calculated loss.
    """
    if pred.dim() != label.dim():
        label = _expand_onehot_labels(label, pred.size(-1), ignore_index)

    return F.binary_cross_entropy_with_logits(
        pred, label.float(), reduction="none"
    )


def _expand_onehot_labels(
    labels: Tensor, label_channels: int, ignore_index: int
) -> Tensor:
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False
    )

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    return bin_labels
