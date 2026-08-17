"""Multi-layer perceptron (MLP)."""

import torch.nn.functional as F
from torch import Tensor, nn

from .activation import SwiGLU


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = output_dim if output_dim is not None else input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        """Type definition for call implementation."""
        return self._call_impl(x)


class SimpleMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerBlockMLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU(),
        bias: bool = True,
        drop: float = 0.0,
    ):
        """Init MLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features.
                Defaults to None.
            out_features (int, optional): Number of output features.
                Defaults to None.
            act_layer (nn.Module, optional): Activation layer.
                Defaults to nn.GELU.
            bias (bool, optional): If bias should be used. Defaults to True.
            drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def __call__(self, data: Tensor) -> Tensor:
        """Applies the layer.

        Args:
            data: (tensor) input shape [N, C]
        """
        return self._call_impl(data)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (tensor) input shape [N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
