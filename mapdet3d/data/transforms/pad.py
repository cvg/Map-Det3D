"""Pad transformation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mapdet3d.common.typing import NDArrayF32, NDArrayUI8
from mapdet3d.data.const import CommonKeys as K

from .base import Transform


@Transform(
    [K.images, K.input_hw],
    [K.images, K.input_hw],
)
class PadImages:
    """Pad batch of images at the bottom right."""

    def __init__(
        self,
        stride: int = 32,
        mode: str = "constant",
        value: float = 0.0,
        update_input_hw: bool = False,
        shape: tuple[int, int] | None = None,
        pad2square: bool = False,
    ) -> None:
        """Creates an instance of PadImage.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
            shape (tuple[int, int], optional): Shape of the padded image
                (H, W). Defaults to None.
            pad2square (bool, optional): Pad to square. Defaults to False.
        """
        if pad2square:
            assert (
                shape is None
            ), "Cannot specify shape when pad2square is True."
        self.stride = stride
        self.mode = mode
        self.value = value
        self.update_input_hw = update_input_hw
        self.shape = shape
        self.pad2square = pad2square

    def __call__(
        self, images: list[NDArrayF32], input_hw: list[tuple[int, int]]
    ) -> tuple[list[NDArrayF32], list[tuple[int, int]]]:
        """Pad images to consistent size."""
        heights = [im.shape[1] for im in images]
        widths = [im.shape[2] for im in images]
        max_hw = _get_max_shape(
            heights, widths, self.stride, self.shape, self.pad2square
        )

        # generate params for torch pad
        for i, (image, h, w) in enumerate(zip(images, heights, widths)):
            pad_param = (0, max_hw[1] - w, 0, max_hw[0] - h)
            image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
            image_ = F.pad(  # pylint: disable=not-callable
                image_, pad_param, self.mode, self.value
            )
            images[i] = image_.permute(0, 2, 3, 1).numpy()

            if self.update_input_hw:
                input_hw[i] = image_.shape[2], image_.shape[3]

        return images, input_hw


@Transform(K.depth_maps, K.depth_maps)
class PadDepthMaps:
    """Pad batch of depth maps at the bottom right."""

    def __init__(
        self,
        stride: int = 32,
        mode: str = "constant",
        value: int = 0,
        shape: tuple[int, int] | None = None,
        pad2square: bool = False,
    ) -> None:
        """Creates an instance.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
            shape (tuple[int, int], optional): Shape of the padded image
                (H, W). Defaults to None.
            pad2square (bool, optional): Pad to square. Defaults to False.
        """
        if pad2square:
            assert (
                shape is None
            ), "Cannot specify shape when pad2square is True."
        self.stride = stride
        self.mode = mode
        self.value = value
        self.shape = shape
        self.pad2square = pad2square

    def __call__(self, depth_maps: list[NDArrayF32]) -> list[NDArrayF32]:
        """Pad images to consistent size."""
        heights = [depth.shape[0] for depth in depth_maps]
        widths = [depth.shape[1] for depth in depth_maps]
        max_hw = _get_max_shape(
            heights, widths, self.stride, self.shape, self.pad2square
        )

        # generate params for torch pad
        for i, (depth, h, w) in enumerate(zip(depth_maps, heights, widths)):
            pad_param = ((0, max_hw[0] - h), (0, max_hw[1] - w))
            depth_maps[i] = np.pad(  # type: ignore
                depth, pad_param, mode=self.mode, constant_values=self.value
            )
        return depth_maps


@Transform(K.seg_masks, K.seg_masks)
class PadSegMasks:
    """Pad batch of segmentation masks at the bottom right."""

    def __init__(
        self,
        stride: int = 32,
        mode: str = "constant",
        value: int = 255,
        shape: tuple[int, int] | None = None,
        pad2square: bool = False,
    ) -> None:
        """Creates an instance of PadSegMasks.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
            shape (tuple[int, int], optional): Shape of the padded image
                (H, W). Defaults to None.
            pad2square (bool, optional): Pad to square. Defaults to False.
        """
        if pad2square:
            assert (
                shape is None
            ), "Cannot specify shape when pad2square is True."
        self.stride = stride
        self.mode = mode
        self.value = value
        self.shape = shape
        self.pad2square = pad2square

    def __call__(self, masks: list[NDArrayUI8]) -> list[NDArrayUI8]:
        """Pad images to consistent size."""
        heights = [mask.shape[0] for mask in masks]
        widths = [mask.shape[1] for mask in masks]
        max_hw = _get_max_shape(
            heights, widths, self.stride, self.shape, self.pad2square
        )

        # generate params for torch pad
        for i, (mask, h, w) in enumerate(zip(masks, heights, widths)):
            pad_param = ((0, max_hw[0] - h), (0, max_hw[1] - w))
            masks[i] = np.pad(  # type: ignore
                mask, pad_param, mode=self.mode, constant_values=self.value
            )
        return masks


def _get_max_shape(
    heights: list[int],
    widths: list[int],
    stride: int,
    shape: tuple[int, int] | None,
    pad2square: bool,
) -> tuple[int, int]:
    """Get max shape for padding.

    Args:
        stride (int): Chooses padding size so that the input will be divisible
            by stride.
        shape (tuple[int, int], optional): Shape of the padded image (H, W).
            Defaults to None.
        heights (list[int]): List of heights of input.
        widths (list[int]): List of widths of input.
        pad2square (bool): Pad to square.

    Returns:
        tuple[int, int]: Max shape for padding.
    """
    if pad2square:
        max_size = max(heights + widths)
        max_hw = (max_size, max_size)
    elif shape is not None:
        max_hw = shape
    else:
        max_hw = max(heights), max(widths)

    return tuple(_make_divisible(x, stride) for x in max_hw)


def _make_divisible(x: int, stride: int) -> int:
    """Ensure divisibility by stride."""
    return (x + (stride - 1)) // stride * stride
