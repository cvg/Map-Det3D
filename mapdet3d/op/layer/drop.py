"""DropPath (Stochastic Depth) regularization layers.

Modified from timm (https://github.com/huggingface/pytorch-image-models).
"""

from __future__ import annotations

import torch
from torch import nn


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Drop path regularizer (Stochastic Depth) per sample.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, ...).
        drop_prob (float, optional): Probability of an element to be zeroed.
            Defaults to 0.0.
        training (bool, optional): If to apply drop path. Defaults to False.
        scale_by_keep (bool, optional): If to scale by keep probability.
            Defaults to True.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """DropPath regularizer (Stochastic Depth) per sample."""

    def __init__(
        self, drop_prob: float = 0.0, scale_by_keep: bool = True
    ) -> None:
        """Init DropPath.

        Args:
            drop_prob (float, optional): Probability of an item to be masked.
                Defaults to 0.0.
            scale_by_keep (bool, optional): If to scale by keep probability.
                Defaults to True.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies the layer.

        Args:
            data: (tensor) input shape [N, ...]
        """
        return self._call_impl(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
