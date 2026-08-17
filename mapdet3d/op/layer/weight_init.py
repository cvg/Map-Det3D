"""Model weight initialization."""

from typing import Literal

import numpy as np
from torch import nn

NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]
FanMode = Literal["fan_in", "fan_out"]


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    """Initialize module with constant value."""
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    """Initialize module with Xavier initialization."""
    assert distribution in {"uniform", "normal"}
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    negative_slope: float = 0.0,
    mode: FanMode = "fan_out",
    nonlinearity: NonlinearityType = "relu",
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    """Initialize module with Kaiming initialization.

    Args:
        module (nn.Module): Module to initialize.
        negative_slope (float, optional): The negative slope of the rectifier
            used after this layer (only used with ``'leaky_relu'``). Defaults
            to 0.0.
        mode (FanMode, optional): Either `"fan_in"` (default) or `"fan_out"``.
            Choosing `"fan_in"` preserves the magnitude of the variance of
            the weights in the forward pass. Choosing `"fan_out"` preserves
            magnitudes in the backwards pass. Defaults to "fan_out".
        nonlinearity (NonlinearityType, optional): The non-linear function
            (`nn.functional` name). Defaults to "relu".
        bias (float, optional): The bias to use. Defaults to 0.0.
        distribution (str, optional): Either ``'uniform'`` or ``'normal'``.
            Defaults to "normal".
    """
    assert distribution in {"uniform", "normal"}
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight,
                a=negative_slope,
                mode=mode,
                nonlinearity=nonlinearity,
            )
        else:
            nn.init.kaiming_normal_(
                module.weight,
                a=negative_slope,
                mode=mode,
                nonlinearity=nonlinearity,
            )
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def normal_init(
    module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0
) -> None:
    """Initialize module with normal distribution."""
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob: float) -> float:
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))


def uniform_init(
    module: nn.Module,
    lower: float = 0.0,
    upper: float = 1.0,
    bias: float = 0.0,
) -> None:
    """Initialize module with uniform distribution."""
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        nn.init.uniform_(module.weight, lower, upper)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)
