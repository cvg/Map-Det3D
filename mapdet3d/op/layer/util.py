"""Utility functions for layer ops."""

from __future__ import annotations

from torch import nn


def build_activation_layer(
    activation: str, inplace: bool = False
) -> nn.Module:
    """Build activation layer.

    Args:
        activation (str): Activation layer type.
        inplace (bool, optional): If to set inplace. Defaults to False. It will
            be ignored if the activation layer is not inplace.
    """
    activation_layer = getattr(nn, activation)

    if activation_layer in {nn.Tanh, nn.PReLU, nn.Sigmoid, nn.GELU}:
        return activation_layer()

    return activation_layer(inplace=inplace)


def build_norm_layer(
    norm: str, out_channels: int, num_groups: int | None = None
) -> nn.Module:
    """Build normalization layer.

    Args:
        norm (str): Normalization layer type.
        out_channels (int): Number of output channels.
        num_groups (int | None, optional): Number of groups for GroupNorm.
            Defaults to None.
    """
    norm_layer = getattr(nn, norm)
    if norm_layer == nn.GroupNorm:
        assert (
            num_groups is not None
        ), "num_groups must be specified when using Group Norm"
        return norm_layer(num_groups, out_channels)

    return norm_layer(out_channels)
