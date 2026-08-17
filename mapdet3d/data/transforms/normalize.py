"""Normalize Transform."""

from __future__ import annotations

import torch

from mapdet3d.common.typing import NDArrayF32

from ..const import CommonKeys as K
from .base import Transform


@Transform(K.images, K.images)
class NormalizeImages:
    """Normalize a list of image tensor with given mean and std.

    Image tensor is of shape [N, H, W, C] and range (0, 255).
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] = (58.395, 57.12, 57.375),
        epsilon: float = 1e-08,
    ) -> None:
        """Creates an instance of NormalizeImage.

        Args:
            mean (Tuple[float, float, float], optional): Mean value. Defaults
                to (123.675, 116.28, 103.53).
            std (Tuple[float, float, float], optional): Standard deviation
                value. Defaults to (58.395, 57.12, 57.375).
            epsilon (float, optional): Epsilon for numerical stability of
                division. Defaults to 1e-08.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Normalize image tensor."""
        for i, image in enumerate(images):
            img = torch.from_numpy(image).permute(0, 3, 1, 2)
            pixel_mean = torch.tensor(self.mean).view(-1, 1, 1)
            pixel_std = torch.tensor(self.std).view(-1, 1, 1)
            img = (img - pixel_mean) / (pixel_std + self.epsilon)

            images[i] = img.permute(0, 2, 3, 1).numpy()

        return images


@Transform(K.depth_maps, [K.depth_maps, "depth_shift_scale"])
class NormalizeDepthMaps:
    """Normalize a list of depth map tensor with given mean and std."""

    def __init__(self, trunc_value: float = 0.1, eps: float = 1e-2) -> None:
        """Init."""
        self.trunc_value = trunc_value
        self.eps = eps

    def __call__(self, depth_maps: list[NDArrayF32]):
        """Standardize depth maps."""
        normalized_depth_maps = []
        depth_shift_scale = []
        for depth in depth_maps:
            depth_img = torch.from_numpy(depth)

            # Set invalid depth to nan
            depth_img[depth_img <= 0.0] = torch.nan
            sorted_img = torch.sort(torch.flatten(depth_img))[0]

            # Remove nan, nan at the end of sort
            num_nan = sorted_img.isnan().sum()
            if num_nan > 0:
                sorted_img = sorted_img[:-num_nan]

            # Remove outliers
            trunc_img = sorted_img[
                int(self.trunc_value * len(sorted_img)) : int(
                    (1 - self.trunc_value) * len(sorted_img)
                )
            ]

            if len(trunc_img) <= 1:
                # guard against no valid Jasper.
                trunc_mean = 0.0
                trunc_std = 1.0
            else:
                trunc_mean = trunc_img.mean().item()
                trunc_std = torch.sqrt(trunc_img.var() + self.eps).item()

            # Replace nan by mean
            depth_img = torch.nan_to_num(depth_img, nan=trunc_mean)

            # Standardize
            depth_img = (depth_img - trunc_mean) / trunc_std

            normalized_depth_maps.append(depth_img.numpy())
            depth_shift_scale.append([trunc_mean, trunc_std])

        return normalized_depth_maps, depth_shift_scale
