"""Focal Loss."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from .base import Loss
from .reducer import LossReducer, mean_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(
        gamma
    )
    loss = (
        F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        * focal_weight
    )
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    return loss


class FocalLoss(Loss):
    """Focal loss <https://arxiv.org/abs/1708.02002>`_."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reducer: LossReducer = mean_loss,
    ) -> None:
        """Creates an instance of the class.

        Args:
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            reducer (LossReducer, optional): Reducer for the loss function.
                Defaults to mean_loss.
        """
        super().__init__(reducer)
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, pred: Tensor, target: Tensor, reducer: LossReducer | None = None
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning label of the prediction.

        Returns:
            Tensor: The calculated loss.
        """
        reducer = reducer or self.reducer

        # This means that target is in One-Hot form.
        if pred.dim() == target.dim():
            focal_loss = sigmoid_focal_loss(
                pred,
                target,
                alpha=self.alpha,
                gamma=self.gamma,
            )
        else:
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1).float()
            target = target[:, :num_classes]

            focal_loss = py_sigmoid_focal_loss(
                pred, target, alpha=self.alpha, gamma=self.gamma
            )

        return reducer(focal_loss)
