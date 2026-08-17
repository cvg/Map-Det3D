"""Activation functions."""

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)


class GEGLU(nn.Module):
    """GEGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
