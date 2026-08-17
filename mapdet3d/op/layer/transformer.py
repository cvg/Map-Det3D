"""Transformer layer.

Modified from timm (https://github.com/huggingface/pytorch-image-models) and
mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .attention import Attention
from .drop import DropPath
from .mlp import TransformerBlockMLP
from .util import build_activation_layer


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.

    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_clones(module: nn.Module, num: int) -> nn.ModuleList:
    """Create N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class LayerScale(nn.Module):
    """Layer scaler."""

    def __init__(
        self,
        dim: int,
        inplace: bool = False,
        data_format: str = "channels_last",
        init_values: float = 1e-5,
    ):
        """Init layer scaler.

        Args:
            dim (int): Input tensor's dimension.
            inplace (bool): Whether performs operation in-place. Default:
                False.
            data_format (str): The input data format, could be 'channels_last'
                or 'channels_first', representing (B, C, H, W) and (B, N, C)
                format data respectively. Default: channels_last.
            init_values (float, optional): Initial values for layer scale.
                Defaults to 1e-5.
        """
        super().__init__()
        assert data_format in {
            "channels_last",
            "channels_first",
        }, "data_format could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.data_format == "channels_first":
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))

        if self.inplace:
            return x.mul_(self.gamma.view(*shape))

        return x * self.gamma.view(*shape)


class TransformerBlock(nn.Module):
    """Transformer block for Vision Transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU(),
        norm_layer: nn.Module | None = None,
    ):
        """Init transformer block.

        Args:
            dim (int): Input tensor's dimension.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding
                dim. Defaults to 4.0.
            qkv_bias (bool, optional): If to add bias to qkv. Defaults to
                False.
            drop (float, optional): Dropout rate for attention and projection.
                Defaults to 0.0.
            attn_drop (float, optional): Dropout rate for attention. Defaults
                to 0.0.
            init_values (tuple[float, float] | None, optional): Initial values
                for layer scale. Defaults to None.
            drop_path (float, optional): Dropout rate for drop path. Defaults
                to 0.0.
            act_layer (nn.Module, optional): Activation layer. Defaults to
                nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. If None, use
                nn.LayerNorm.
        """
        super().__init__()
        self.norm1 = (
            norm_layer(dim) if norm_layer else nn.LayerNorm(dim, eps=1e-6)
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = (
            norm_layer(dim) if norm_layer else nn.LayerNorm(dim, eps=1e-6)
        )
        self.mlp = TransformerBlockMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            data (torch.Tensor): Input tensor of shape (B, N, dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim).
        """
        return self._call_impl(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection."""

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        dropout: float = 0.0,
        activation: str = "ReLU",
        inplace: bool = True,
        dropout_layer: nn.Module | None = None,
        add_identity: bool = True,
        layer_scale_init_value: float = 0.0,
    ) -> None:
        """Init FFN.

        Args:
            embed_dims (int): The feature dimension. Defaults: 256.
            feedforward_channels (int): The hidden dimension of FFNs.
                Defaults: 1024.
            num_fcs (int): The number of fully-connected layers in FFNs.
                Defaults: 2.
            dropout (float): The dropout rate of FFNs.
            activation (str): The activation function of FFNs.
            inplace (bool): Whether to set inplace for activation.
            dropout_layer (nn.Module | None, optional): The dropout_layer used
                when adding the shortcut. Defaults to None. If None, Identity
                is used.
            add_identity (bool, optional): Whether to add the identity
                connection. Default: True.
            layer_scale_init_value (float): Initial value of scale factor in
                LayerScale. Default: 0.0
        """
        super().__init__()
        self.embed_dims = embed_dims

        layers: list[nn.Module] = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    build_activation_layer(activation, inplace),
                    nn.Dropout(dropout),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

        self.dropout_layer = dropout_layer or nn.Identity()
        self.add_identity = add_identity
        self.layer_scale_init_value = layer_scale_init_value

        if self.layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                embed_dims, init_values=self.layer_scale_init_value
            )

    def forward(self, x: Tensor, identity: Tensor | None = None) -> None:
        """Forward function for FFN.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)

        if self.layer_scale_init_value > 0:
            out = self.gamma2(out)

        if self.add_identity:
            identity = x if identity is None else identity
            return identity + self.dropout_layer(out)

        return self.dropout_layer(out)
