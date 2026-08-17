"""Resize transformation."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mapdet3d.common.imports import OPENCV_AVAILABLE
from mapdet3d.common.typing import NDArrayF32
from mapdet3d.data.const import CommonKeys as K
from mapdet3d.op.box2d import transform_bbox

from .base import Transform

if OPENCV_AVAILABLE:
    import cv2
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        INTER_AREA,
        INTER_CUBIC,
        INTER_LANCZOS4,
        INTER_LINEAR,
        INTER_NEAREST,
    )
else:
    raise ImportError("Please install opencv-python to use this module.")


class ResizeParam(TypedDict):
    """Parameters for Resize."""

    target_shape: tuple[int, int]
    scale_factor: tuple[float, float]


@Transform(K.images, ["transforms.resize", K.input_hw])
class GenResizeParameters:
    """Generate the parameters for a resize operation.

    Width will be set to target_size while keeping the original aspect ratio.
    Height will be rounded to be divisible to patch_size.
    """

    def __init__(self, target_size: int = 518, patch_size: int = 14) -> None:
        """Init."""
        self.target_size = target_size
        self.patch_size = patch_size

    def __call__(
        self, images: list[NDArrayF32]
    ) -> tuple[list[ResizeParam], list[tuple[int, int]]]:
        """Compute the parameters and put them in the data dict."""
        image = images[0]  # Assume all images have the same shape
        height, width = image.shape[1], image.shape[2]

        # Original behavior: set width to 518px
        new_width = self.target_size

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = (
            round(height * (new_width / width) / self.patch_size)
            * self.patch_size
        )

        target_shape = (new_height, new_width)

        scale_factor = (
            target_shape[1] / width,
            target_shape[0] / height,
        )

        resize_params = [
            ResizeParam(target_shape=target_shape, scale_factor=scale_factor)
        ] * len(images)
        target_shapes = [target_shape] * len(images)

        return resize_params, target_shapes


@Transform([K.images, "transforms.resize.target_shape"], K.images)
class ResizeImages:
    """Resize Images."""

    def __init__(
        self,
        interpolation: str = "bicubic",
        antialias: bool = False,
        imresize_backend: str = "torch",
    ) -> None:
        """Creates an instance of the class.

        Args:
            interpolation (str, optional): Interpolation method. One of
                ["nearest", "bilinear", "bicubic"]. Defaults to "bilinear".
            antialias (bool): Whether to use antialiasing. Defaults to False.
            imresize_backend (str): One of torch, cv2. Defaults to torch.
        """
        self.interpolation = interpolation
        self.antialias = antialias
        self.imresize_backend = imresize_backend
        assert imresize_backend in {
            "torch",
            "cv2",
        }, f"Invalid imresize backend: {imresize_backend}"

    def __call__(
        self, images: list[NDArrayF32], target_shapes: list[tuple[int, int]]
    ) -> list[NDArrayF32]:
        """Resize an image of dimensions [N, H, W, C].

        Args:
            image (Tensor): The image.
            target_shape (tuple[int, int]): The target shape after resizing.

        Returns:
            list[NDArrayF32]: Resized images according to parameters in resize.
        """
        for i, (image, target_shape) in enumerate(zip(images, target_shapes)):
            images[i] = resize_image(
                image,
                target_shape,
                interpolation=self.interpolation,
                antialias=self.antialias,
                backend=self.imresize_backend,
            )
        return images


def resize_image(
    inputs: NDArrayF32,
    shape: tuple[int, int],
    interpolation: str = "bilinear",
    antialias: bool = False,
    backend: str = "torch",
) -> NDArrayF32:
    """Resize image."""
    if backend == "torch":
        image = torch.from_numpy(inputs).permute(0, 3, 1, 2)
        image = resize_tensor(image, shape, interpolation, antialias)
        return image.permute(0, 2, 3, 1).numpy()

    if backend == "cv2":
        cv2_interp_codes = {
            "nearest": INTER_NEAREST,
            "bilinear": INTER_LINEAR,
            "bicubic": INTER_CUBIC,
            "area": INTER_AREA,
            "lanczos": INTER_LANCZOS4,
        }
        return cv2.resize(  # pylint: disable=no-member, unsubscriptable-object
            inputs[0].astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2_interp_codes[interpolation],
        )[None, ...].astype(np.float32)

    raise ValueError(f"Invalid imresize backend: {backend}")


@Transform([K.boxes2d, "transforms.resize.scale_factor"], K.boxes2d)
class ResizeBoxes2D:
    """Resize list of 2D bounding boxes."""

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize 2D bounding boxes.

        Args:
            boxes_list: (list[NDArrayF32]): The bounding boxes to be resized.
            scale_factors (list[tuple[float, float]]): scaling factors.

        Returns:
            list[NDArrayF32]: Resized bounding boxes according to parameters in
                resize.
        """
        for i, (boxes, scale_factor) in enumerate(
            zip(boxes_list, scale_factors)
        ):
            boxes_ = torch.from_numpy(boxes)
            scale_matrix = torch.eye(3)
            scale_matrix[0, 0] = scale_factor[0]
            scale_matrix[1, 1] = scale_factor[1]
            boxes_list[i] = transform_bbox(scale_matrix, boxes_).numpy()
        return boxes_list


@Transform(
    [
        K.depth_maps,
        "transforms.resize.target_shape",
        "transforms.resize.scale_factor",
    ],
    K.depth_maps,
)
class ResizeDepthMaps:
    """Resize depth maps."""

    def __init__(
        self,
        interpolation: str = "nearest",
        rescale_depth_values: bool = False,
        check_scale_factors: bool = False,
    ):
        """Initialize the transform.

        Args:
            interpolation (str, optional): Interpolation method. One of
                ["nearest", "bilinear", "bicubic"]. Defaults to "nearest".
            rescale_depth_values (bool, optional): If the depth values should
                be rescaled according to the new scale factor. Defaults to
                False. This is useful if we want to keep the intrinsic
                parameters of the camera the same.
            check_scale_factors (bool, optional): If the scale factors should
                be checked to ensure they are the same. Defaults to False.
                If False, the scale factor is assumed to be the same for both
                dimensions and will just use the first scale factor.
        """
        self.interpolation = interpolation
        self.rescale_depth_values = rescale_depth_values
        self.check_scale_factors = check_scale_factors

    def __call__(
        self,
        depth_maps: list[NDArrayF32],
        target_shapes: list[tuple[int, int]],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize depth maps."""
        for i, (depth_map, target_shape, scale_factor) in enumerate(
            zip(depth_maps, target_shapes, scale_factors)
        ):
            depth_map_ = torch.from_numpy(depth_map)
            depth_map_ = (
                resize_tensor(
                    depth_map_.float().unsqueeze(0).unsqueeze(0),
                    target_shape,
                    interpolation=self.interpolation,
                )
                .type(depth_map_.dtype)
                .squeeze(0)
                .squeeze(0)
            )
            if self.rescale_depth_values:
                if self.check_scale_factors:
                    assert np.isclose(
                        scale_factor[0], scale_factor[1], atol=1e-4
                    ), "Depth map scale factors must be the same"
                depth_map_ /= scale_factor[0]
            depth_maps[i] = depth_map_.numpy()
        return depth_maps


@Transform(
    [
        K.optical_flows,
        "transforms.resize.target_shape",
        "transforms.resize.scale_factor",
    ],
    K.optical_flows,
)
class ResizeOpticalFlows:
    """Resize optical flows."""

    def __init__(self, normalized_flow: bool = True):
        """Create a ResizeOpticalFlows instance.

        Args:
            normalized_flow (bool): Whether the optical flow is normalized.
                Defaults to True. If false, the optical flow will be scaled
                according to the scale factor.
        """
        self.normalized_flow = normalized_flow

    def __call__(
        self,
        optical_flows: list[NDArrayF32],
        target_shapes: list[tuple[int, int]],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize optical flows."""
        for i, (optical_flow, target_shape, scale_factor) in enumerate(
            zip(optical_flows, target_shapes, scale_factors)
        ):
            optical_flow_ = torch.from_numpy(optical_flow).permute(2, 0, 1)
            optical_flow_ = (
                resize_tensor(
                    optical_flow_.float().unsqueeze(0),
                    target_shape,
                    interpolation="bilinear",
                )
                .type(optical_flow_.dtype)
                .squeeze(0)
                .permute(1, 2, 0)
            )
            # scale optical flows
            if not self.normalized_flow:
                optical_flow_[:, :, 0] *= scale_factor[0]
                optical_flow_[:, :, 1] *= scale_factor[1]
            optical_flows[i] = optical_flow_.numpy()
        return optical_flows


@Transform(
    [K.instance_masks, "transforms.resize.target_shape"], K.instance_masks
)
class ResizeInstanceMasks:
    """Resize instance segmentation masks."""

    def __call__(
        self,
        masks_list: list[NDArrayF32],
        target_shapes: list[tuple[int, int]],
    ) -> list[NDArrayF32]:
        """Resize masks."""
        for i, (masks, target_shape) in enumerate(
            zip(masks_list, target_shapes)
        ):
            if len(masks) == 0:  # handle empty masks
                continue
            masks_ = torch.from_numpy(masks)
            masks_ = (
                resize_tensor(
                    masks_.float().unsqueeze(1),
                    target_shape,
                    interpolation="nearest",
                )
                .type(masks_.dtype)
                .squeeze(1)
            )
            masks_list[i] = masks_.numpy()
        return masks_list


@Transform([K.seg_masks, "transforms.resize.target_shape"], K.seg_masks)
class ResizeSegMasks:
    """Resize segmentation masks."""

    def __call__(
        self,
        masks_list: list[NDArrayF32],
        target_shape_list: list[tuple[int, int]],
    ) -> list[NDArrayF32]:
        """Resize masks."""
        for i, (masks, target_shape) in enumerate(
            zip(masks_list, target_shape_list)
        ):
            masks_ = torch.from_numpy(masks)
            masks_ = (
                resize_tensor(
                    masks_.float().unsqueeze(0).unsqueeze(0),
                    target_shape,
                    interpolation="nearest",
                )
                .type(masks_.dtype)
                .squeeze(0)
                .squeeze(0)
            )
            masks_list[i] = masks_.numpy()
        return masks_list


@Transform([K.intrinsics, "transforms.resize.scale_factor"], K.intrinsics)
class ResizeIntrinsics:
    """Resize Intrinsics."""

    def __call__(
        self,
        intrinsics: list[NDArrayF32],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Scale camera intrinsics when resizing."""
        for i, scale_factor in enumerate(scale_factors):
            scale_matrix = np.eye(3, dtype=np.float32)
            scale_matrix[0, 0] *= scale_factor[0]
            scale_matrix[1, 1] *= scale_factor[1]
            intrinsics[i] = scale_matrix @ intrinsics[i]
        return intrinsics


def resize_tensor(
    inputs: Tensor,
    shape: tuple[int, int],
    interpolation: str = "bilinear",
    antialias: bool = False,
) -> Tensor:
    """Resize Tensor."""
    assert interpolation in {"nearest", "bilinear", "bicubic"}
    align_corners = None if interpolation == "nearest" else False
    output = F.interpolate(
        inputs,
        shape,
        mode=interpolation,
        align_corners=align_corners,
        antialias=antialias,
    )
    return output
