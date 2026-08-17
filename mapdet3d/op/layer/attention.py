"""Attention layer."""

from __future__ import annotations

from torch import Tensor, nn

from mapdet3d.common.logging import rank_zero_warn
from mapdet3d.common.typing import ArgsType


class Attention(nn.Module):
    """ViT Attention Layer.

    Modified from timm (https://github.com/huggingface/pytorch-image-models).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Init attention layer.

        Args:
            dim (int): Input tensor's dimension.
            num_heads (int, optional): Number of attention heads. Defaults to
                8.
            qkv_bias (bool, optional): If to add bias to qkv. Defaults to
                False.
            attn_drop (float, optional): Dropout rate for attention. Defaults
                to 0.0.
            proj_drop (float, optional): Dropout rate for projection. Defaults
                to 0.0.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, data: Tensor) -> Tensor:
        """Applies the layer.

        Args:
            data (Tensor): Input tensor of shape (B, N, dim).

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        return self._call_impl(data)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        batch_size, num_samples, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                num_samples,
                3,
                self.num_heads,
                dim // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(
            0
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_samples, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding is also passed as input.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: nn.Module | None = None,
        batch_first: bool = False,
        need_weights: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Init MultiheadAttention.

        Args:
            embed_dims (int): The embedding dimension.
            num_heads (int): Parallel attention heads.
            attn_drop (float): A Dropout layer on attn_output_weights.
                Default: 0.0.
            proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
                Default: 0.0.
            dropout_layer (nn.Module | None, optional): The dropout_layer used
                when adding the shortcut. Defaults to None.
            batch_first (bool): When it is True,  Key, Query and Value are
                shape of (batch, n, embed_dim), otherwise (n, batch,
                embed_dim). Default to False.
            need_weights (bool): Whether to return the attention weights.
                If True, the output will be a tuple of (attn_output,
                attn_output_weights) and not using FlashAttention. If False,
                only the attn_output will be returned. Default to False.
        """
        super().__init__()
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.need_weights = need_weights

        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, **kwargs
        )

        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_layer = dropout_layer or nn.Identity()

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        identity: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as query,
                will be used for the identity link.
                If None, `query` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`. If not None, it will
                be added to `query` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims]
                if self.batch_first is False, else [bs, num_queries,
                embed_dims].
        """
        if key is None:
            key = query

        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None and query_pos is not None:
            # use query_pos if key_pos is not available
            if query_pos.shape == key.shape:
                key_pos = query_pos
            else:
                rank_zero_warn(
                    f"Position encoding of key in {self.__class__.__name__}"
                    + "is missing, and positional encodeing of query has "
                    + "has different shape and cannot be usde for key. "
                    + "It it is not desired, please provide key_pos."
                )

        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query, batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.need_weights,
        )

        if isinstance(out, tuple):
            out = out[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
