"""Various positional encodings for the transformer."""

import math

import torch
from torch import Tensor, nn


class PositionEmbeddingSineHW(nn.Module):
    """A more standard version of the position embedding.

    It is very similar to the one used by the Attention is all you need paper,
    generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperatureH: int = 10000,
        temperatureW: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        """Constructor method for PositionEmbeddingSineHW class."""
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor, mask: Tensor | None = None):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_tx = self.temperatureW ** (
            2
            * (torch.div(dim_tx, 2, rounding_mode="floor"))
            / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_ty = self.temperatureH ** (
            2
            * (torch.div(dim_ty, 2, rounding_mode="floor"))
            / self.num_pos_feats
        )
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """Generate sine position embedding from a position tensor.

    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.

    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack(
            (sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        return sin_x

    pos_res = [
        sine_func(x)
        for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    ]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(
            "Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1))
        )
    pos = pos.to(pos_tensor.dtype)
    return pos


def coordinate_to_encoding(
    coord_tensor: Tensor,
    num_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> Tensor:
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device
    )
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack(
            (pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1
        ).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack(
            (pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(
            "Unknown pos_tensor shape(-1):{}".format(coord_tensor.size(-1))
        )
    return pos


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(
        self,
        num_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        """Initialization for `SinePositionalEncoding`.

        Args:
            num_feats (int): The feature dimension for each position
                along x-axis or y-axis. Note the final returned dimension
                for each position is 2 times of this value.
            temperature (int, optional): The temperature used for scaling
                the position embedding. Defaults to 10000.
            normalize (bool, optional): Whether to normalize the position
                embedding. Defaults to False.
            scale (float, optional): A scale factor that scales the position
                embedding. The scale will be used only when normalize is True.
                Defaults to 2*pi.
            eps (float, optional): A value added to the denominator for
                numerical stability. Defaults to 1e-6.
            offset (float, optional): offset add to embed when do the
                normalization. Defaults to 0.
        """
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(
        self, mask: Tensor | None, inputs: Tensor | None = None
    ) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor | None): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w]. If None, it means single
                image or batch image with no padding.
            inputs (Tensor | None): The input tensor. It mask is None, this
                input tensor is required to get the shape of the input image.

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        if mask is not None:
            # For convenience of exporting to ONNX, it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            b, h, w = mask.size()
            device = mask.device
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            assert isinstance(inputs, Tensor)
            b, _, h, w = inputs.shape
            device = inputs.device
            x_embed = torch.arange(
                1, w + 1, dtype=torch.float32, device=device
            )
            x_embed = x_embed.view(1, 1, -1).repeat(b, h, 1)
            y_embed = torch.arange(
                1, h + 1, dtype=torch.float32, device=device
            )
            y_embed = y_embed.view(1, -1, 1).repeat(b, 1, w)
        if self.normalize:
            y_embed = (
                (y_embed + self.offset)
                / (y_embed[:, -1:, :] + self.eps)
                * self.scale
            )
            x_embed = (
                (x_embed + self.offset)
                / (x_embed[:, :, -1:] + self.eps)
                * self.scale
            )
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(b, h, w, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(b, h, w, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
