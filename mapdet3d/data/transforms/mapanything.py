"""MapAnything Transform."""

from __future__ import annotations

import cv2
import numpy as np
import PIL
import torch
from torch import Tensor

from mapdet3d.common.typing import NDArrayF32
from mapdet3d.data.const import CommonKeys as K
from mapdet3d.op.box2d import transform_bbox

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

from .base import Transform

# Fixed resolution mappings with precomputed aspect ratios as keys
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),  # 1:1
        1.321: (518, 392),  # 4:3
        1.542: (518, 336),  # 3:2
        1.762: (518, 294),  # 16:9
        2.056: (518, 252),  # 2:1
        3.083: (518, 168),  # 3.2:1
        0.757: (392, 518),  # 3:4
        0.649: (336, 518),  # 2:3
        0.567: (294, 518),  # 9:16
        0.486: (252, 518),  # 1:2
    },
    512: {
        1.000: (512, 512),  # 1:1
        1.333: (512, 384),  # 4:3
        1.524: (512, 336),  # 3:2
        1.778: (512, 288),  # 16:9
        2.000: (512, 256),  # 2:1
        3.200: (512, 160),  # 3.2:1
        0.750: (384, 512),  # 3:4
        0.656: (336, 512),  # 2:3
        0.562: (288, 512),  # 9:16
        0.500: (256, 512),  # 1:2
    },
}

# Precomputed sorted aspect ratio keys for efficient lookup
ASPECT_RATIO_KEYS = {
    518: sorted(RESOLUTION_MAPPINGS[518].keys()),
    512: sorted(RESOLUTION_MAPPINGS[512].keys()),
}


def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]


@Transform(in_keys=[K.images, "resolution"], out_keys=["resolution"])
class PickResolution:
    """Pick resolution."""

    def __init__(
        self,
        resolution_set: list[tuple[int, int]] = [
            (518, 518),
            (518, 392),
            (518, 336),
            (518, 294),
            (518, 252),
            (518, 168),
            (392, 518),
            (336, 518),
            (294, 518),
            (252, 518),
        ],
    ) -> None:
        """Init."""
        self.resolution_set = resolution_set

    def __call__(
        self, images, resolution: list[tuple[int, int]] | None
    ) -> list[tuple[int, int]]:
        """Forward."""
        aspect_ratios = []
        for img in images:
            H, W = img.shape[1], img.shape[2]
            aspect_ratios.append(W / H)

        average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)

        if resolution is None:
            # resolution = [random.choice(self.resolution_set)] * len(images)
            target_width, target_height = find_closest_aspect_ratio(
                average_aspect_ratio, resolution_set=518
            )
            resolution = [(target_width, target_height)] * len(images)

        return resolution


@Transform(
    in_keys=[K.images, K.intrinsics, K.boxes2d, "resolution"],
    out_keys=[K.images, K.intrinsics, K.boxes2d, K.input_hw],
)
class ResizeAndCrop:
    """Resize and Crop transform integrating _crop_resize_if_necessary logic.

    This transform processes images, depth maps, and camera intrinsics in a batch-wise
    manner, applying high-quality resizing and cropping operations while maintaining
    geometric consistency across all modalities.
    """

    def __init__(
        self,
        principal_point_centered: bool = False,
        aug_crop: int = 0,
        seed: int = 777,
    ) -> None:
        """Initialize the ResizeAndCrop transform.

        Args:
            principal_point_centered: If True, crop centered on the principal point
                before resizing. This ensures the optical center is preserved.
            aug_crop: Augmentation crop pixels. If > 0, adds random pixels to target
                resolution for data augmentation during training.
            seed: Random seed for reproducibility of augmentation.
        """
        self.principal_point_centered = principal_point_centered
        self.aug_crop = aug_crop
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        images: list[NDArrayF32],
        intrinsics: list[NDArrayF32],
        boxes2d: list[NDArrayF32],
        resolutions: list[tuple[int, int]],
        depth_maps: list[NDArrayF32] | None = None,
    ) -> tuple[list[NDArrayF32], list[NDArrayF32], list[NDArrayF32] | None]:
        """Process images, intrinsics, and depth maps for a batch.

        The processing pipeline:
        1. Convert images to PIL format
        2. (Optional) Apply principal point centered crop
        3. Calculate target resolution with optional augmentation
        4. Apply high-quality Lanczos downscaling
        5. Apply final precision crop to match exact target resolution
        6. Transform boxes2d according to all crops and resizes

        Args:
            images: List of images as numpy arrays (H, W, 3) or (B, H, W, 3)
            intrinsics: List of camera intrinsic matrices (3, 3)
            boxes2d: List of 2D bounding boxes (N, 4) in xyxy format
            resolutions: List of target resolutions as (width, height) tuples
            depth_maps: List of depth maps (H, W) or (B, H, W) or None

        Returns:
            Tuple of (processed_images, updated_intrinsics, processed_depth_maps)
        """

        # Use the first resolution for all images in batch (batch-level consistency)
        resolution = resolutions[0]

        processed_images = []
        processed_intrinsics = []
        processed_boxes2d = []
        processed_input_hw = []
        processed_depth_maps = [] if depth_maps is not None else None

        # Prepare depth maps list or None placeholders
        depth_maps_iter = (
            depth_maps if depth_maps is not None else [None] * len(images)
        )

        for idx, (image, intrinsic, boxes, depth_map) in enumerate(
            zip(images, intrinsics, boxes2d, depth_maps_iter)
        ):
            # Handle batched input (B, H, W, C) or single input (H, W, C)
            if image.ndim == 4:
                # Process first item in batch dimension
                image_data = image[0]
                intrinsic_data = intrinsic
                depth_data = depth_map[0] if depth_map is not None else None
            else:
                image_data = image
                intrinsic_data = intrinsic
                depth_data = depth_map

            # Get original image size
            if isinstance(image_data, PIL.Image.Image):
                orig_width, orig_height = image_data.size
            else:
                orig_height, orig_width = image_data.shape[:2]

            # Process single view through the pipeline
            (
                processed_img,
                processed_depth,
                processed_intrinsic,
                transform_matrix,
            ) = self._crop_resize_if_necessary(
                image=image_data,
                resolution=resolution,
                depthmap=depth_data,
                intrinsics=intrinsic_data.copy(),
                additional_quantities=None,
                return_transform=True,
                orig_size=(orig_width, orig_height),
            )
            processed_input_hw.append(processed_img.size[::-1])  # (H, W)

            # Transform boxes2d using the accumulated transformation matrix
            if boxes.shape[0] > 0:  # Only transform if there are boxes
                boxes_tensor = torch.from_numpy(boxes).float()
                transformed_boxes = transform_bbox(
                    transform_matrix, boxes_tensor
                )
                processed_boxes2d.append(transformed_boxes.numpy())
            else:
                processed_boxes2d.append(boxes)

            # Convert back to numpy array
            processed_images.append(np.array(processed_img)[None])
            processed_intrinsics.append(processed_intrinsic)
            if processed_depth_maps is not None:
                processed_depth_maps.append(processed_depth)

        return (
            processed_images,
            processed_intrinsics,
            processed_boxes2d,
            processed_input_hw,
        )

    def _convert_to_pil_image(self, image: NDArrayF32) -> PIL.Image.Image:
        """Step 1: Convert image to PIL.Image if necessary.

        Args:
            image: Input image as numpy array or PIL Image

        Returns:
            PIL.Image.Image: Image in PIL format
        """
        if not isinstance(image, PIL.Image.Image):
            # Handle different data types
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
        return image

    def _apply_principal_point_centered_crop(
        self,
        image: PIL.Image.Image,
        resolution: tuple[int, int],
        depthmap: NDArrayF32 | None,
        intrinsics: NDArrayF32,
        additional_quantities: list[NDArrayF32] | None,
    ) -> tuple[
        PIL.Image.Image, NDArrayF32 | None, NDArrayF32, list[NDArrayF32] | None
    ]:
        """Step 2: Apply centered crop around the principal point if necessary.

        This crops the image to be centered on the camera's principal point,
        ensuring the crop is larger than the target resolution.

        Args:
            image: Input PIL image
            resolution: Target resolution as (width, height)
            depthmap: Depth map corresponding to the image
            intrinsics: Camera intrinsics matrix (3x3)
            additional_quantities: Additional image-related data

        Returns:
            Tuple of (cropped_image, cropped_depthmap, updated_intrinsics, cropped_additional)
        """
        if not self.principal_point_centered:
            return image, depthmap, intrinsics, additional_quantities

        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)

        # Skip if principal point is outside image bounds
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            return image, depthmap, intrinsics, additional_quantities

        # Calculate crop centered on principal point
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        left, top = cx - min_margin_x, cy - min_margin_y
        right, bottom = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (left, top, right, bottom)

        # Only perform the centered crop if the crop_bbox is larger than target resolution
        crop_width = right - left
        crop_height = bottom - top
        if crop_width > resolution[0] and crop_height > resolution[1]:
            image, depthmap, intrinsics, additional_quantities = (
                crop_image_and_other_optional_info(
                    image=image,
                    crop_bbox=crop_bbox,
                    depthmap=depthmap,
                    camera_intrinsics=intrinsics,
                    additional_quantities=additional_quantities,
                )
            )

        return image, depthmap, intrinsics, additional_quantities

    def _calculate_target_rescale_resolution(
        self, resolution: tuple[int, int]
    ) -> np.ndarray:
        """Step 3: Calculate the target resolution for rescaling.

        Optionally adds augmentation crop if aug_crop > 1.

        Args:
            resolution: Base target resolution as (width, height)

        Returns:
            numpy.ndarray: Target rescale resolution
        """
        target_rescale_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_rescale_resolution += self._rng.integers(0, self.aug_crop)
        return target_rescale_resolution

    def _apply_high_quality_downscaling(
        self,
        image: PIL.Image.Image,
        target_rescale_resolution: np.ndarray,
        depthmap: NDArrayF32 | None,
        intrinsics: NDArrayF32,
        additional_quantities: list[NDArrayF32] | None,
    ) -> tuple[
        PIL.Image.Image, NDArrayF32 | None, NDArrayF32, list[NDArrayF32] | None
    ]:
        """Step 4: Apply high-quality Lanczos down-scaling if necessary.

        Args:
            image: Input PIL image
            target_rescale_resolution: Target resolution for rescaling
            depthmap: Depth map corresponding to the image
            intrinsics: Camera intrinsics matrix (3x3)
            additional_quantities: Additional image-related data

        Returns:
            Tuple of (rescaled_image, rescaled_depthmap, updated_intrinsics, rescaled_additional)
        """
        image, depthmap, intrinsics, additional_quantities = (
            rescale_image_and_other_optional_info(
                image=image,
                output_resolution=target_rescale_resolution,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities_to_be_resized_with_nearest=additional_quantities,
            )
        )
        return image, depthmap, intrinsics, additional_quantities

    def _apply_final_crop(
        self,
        image: PIL.Image.Image,
        resolution: tuple[int, int],
        depthmap: NDArrayF32 | None,
        intrinsics: NDArrayF32,
        additional_quantities: list[NDArrayF32] | None,
    ) -> tuple[
        PIL.Image.Image, NDArrayF32 | None, NDArrayF32, list[NDArrayF32] | None
    ]:
        """Step 5: Apply final cropping to match the exact target resolution.

        Args:
            image: Input PIL image
            resolution: Target resolution as (width, height)
            depthmap: Depth map corresponding to the image
            intrinsics: Camera intrinsics matrix (3x3)
            additional_quantities: Additional image-related data

        Returns:
            Tuple of (cropped_image, cropped_depthmap, updated_intrinsics, cropped_additional)
        """
        new_intrinsics = camera_matrix_of_crop(
            input_camera_matrix=intrinsics,
            input_resolution=image.size,
            output_resolution=resolution,
            offset_factor=0.5,
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            input_camera_matrix=intrinsics,
            output_camera_matrix=new_intrinsics,
            output_resolution=resolution,
        )
        image, depthmap, new_intrinsics, additional_quantities = (
            crop_image_and_other_optional_info(
                image=image,
                crop_bbox=crop_bbox,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities=additional_quantities,
            )
        )
        return image, depthmap, new_intrinsics, additional_quantities

    def _crop_resize_if_necessary(
        self,
        image: NDArrayF32,
        resolution: tuple[int, int],
        depthmap: NDArrayF32 | None,
        intrinsics: NDArrayF32,
        additional_quantities: list[NDArrayF32] | None = None,
        return_transform: bool = False,
        orig_size: tuple[int, int] | None = None,
    ) -> (
        tuple[PIL.Image.Image, NDArrayF32 | None, NDArrayF32]
        | tuple[PIL.Image.Image, NDArrayF32 | None, NDArrayF32, "Tensor"]
    ):
        """Process an image through the full crop-resize pipeline.

        This orchestrates a step-by-step process:
        1. Converts the image to PIL.Image if necessary
        2. Crops the image centered on the principal point if requested
        3. Calculates target resolution with optional augmentation
        4. Downsamples the image using high-quality Lanczos filtering
        5. Performs final cropping to match the target resolution

        Args:
            image: Input image to be processed (H, W, 3)
            resolution: Target resolution as (width, height)
            depthmap: Depth map corresponding to the image (H, W)
            intrinsics: Camera intrinsics matrix (3x3)
            additional_quantities: Additional image-related data to be processed
                alongside the main image with nearest interpolation
            return_transform: If True, return transformation matrix for boxes
            orig_size: Original image size (width, height) before any processing

        Returns:
            Tuple of (processed_image, processed_depthmap, updated_intrinsics)
            or (processed_image, processed_depthmap, updated_intrinsics, transform_matrix)
        """
        import torch

        # Step 1: Convert to PIL Image
        image = self._convert_to_pil_image(image)

        # Initialize transformation matrix as identity
        if return_transform:
            if orig_size is None:
                orig_size = image.size
            orig_width, orig_height = orig_size
            transform_matrix = torch.eye(3, dtype=torch.float32)

        # Track size before each step for transformation computation
        size_before_pp_crop = image.size

        # Step 2: Apply principal point centered crop
        image, depthmap, intrinsics, additional_quantities = (
            self._apply_principal_point_centered_crop(
                image, resolution, depthmap, intrinsics, additional_quantities
            )
        )

        if return_transform and image.size != size_before_pp_crop:
            # Compute crop transformation
            W_before, H_before = size_before_pp_crop
            W_after, H_after = image.size
            cx, cy = intrinsics[:2, 2].round().astype(int)
            min_margin_x = min(cx, W_before - cx)
            min_margin_y = min(cy, H_before - cy)
            left = cx - min_margin_x
            top = cy - min_margin_y

            crop_transform = torch.eye(3, dtype=torch.float32)
            crop_transform[0, 2] = -left
            crop_transform[1, 2] = -top
            transform_matrix = crop_transform @ transform_matrix

        size_before_resize = image.size

        # Step 3: Calculate target rescale resolution
        target_rescale_resolution = self._calculate_target_rescale_resolution(
            resolution
        )

        # Step 4: Apply high-quality downscaling
        image, depthmap, intrinsics, additional_quantities = (
            self._apply_high_quality_downscaling(
                image,
                target_rescale_resolution,
                depthmap,
                intrinsics,
                additional_quantities,
            )
        )

        if return_transform and image.size != size_before_resize:
            # Compute resize transformation
            W_before, H_before = size_before_resize
            W_after, H_after = image.size
            scale_x = W_after / W_before
            scale_y = H_after / H_before

            scale_transform = torch.eye(3, dtype=torch.float32)
            scale_transform[0, 0] = scale_x
            scale_transform[1, 1] = scale_y
            transform_matrix = scale_transform @ transform_matrix

        size_before_final_crop = image.size

        # Step 5: Apply final crop
        image, depthmap, new_intrinsics, additional_quantities = (
            self._apply_final_crop(
                image, resolution, depthmap, intrinsics, additional_quantities
            )
        )

        if return_transform and image.size != size_before_final_crop:
            # Compute final crop transformation
            new_crop_intrinsics = camera_matrix_of_crop(
                input_camera_matrix=intrinsics,
                input_resolution=size_before_final_crop,
                output_resolution=resolution,
                offset_factor=0.5,
            )
            crop_bbox = bbox_from_intrinsics_in_out(
                input_camera_matrix=intrinsics,
                output_camera_matrix=new_crop_intrinsics,
                output_resolution=resolution,
            )
            left, top, _, _ = crop_bbox

            final_crop_transform = torch.eye(3, dtype=torch.float32)
            final_crop_transform[0, 2] = -left
            final_crop_transform[1, 2] = -top
            transform_matrix = final_crop_transform @ transform_matrix

        if return_transform:
            return image, depthmap, new_intrinsics, transform_matrix
        else:
            return image, depthmap, new_intrinsics


def crop_image_and_other_optional_info(
    image,
    crop_bbox,
    depthmap=None,
    camera_intrinsics=None,
    additional_quantities=None,
):
    """
    Return a crop of the input view and associated data.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image to be cropped
        crop_bbox (tuple): Crop bounding box as (left, top, right, bottom)
        depthmap (numpy.ndarray, optional): Depth map associated with the image
        camera_intrinsics (numpy.ndarray, optional): Camera intrinsics matrix
        additional_quantities (list of numpy.ndarray, optional): Additional data arrays to crop

    Returns:
        tuple: A tuple containing:
            - The cropped image
            - The cropped depth map (if provided or None)
            - Updated camera intrinsics (if provided or None)
            - List of cropped additional quantities (if provided or None)
    """
    image = ImageList(image)
    left, top, right, bottom = crop_bbox

    image = image.crop((left, top, right, bottom))
    if depthmap is not None:
        depthmap = depthmap[top:bottom, left:right]
    if additional_quantities is not None:
        additional_quantities = [
            quantity[top:bottom, left:right]
            for quantity in additional_quantities
        ]

    if camera_intrinsics is not None:
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] -= left
        camera_intrinsics[1, 2] -= top

    return (image.to_pil(), depthmap, camera_intrinsics, additional_quantities)


def rescale_image_and_other_optional_info(
    image,
    output_resolution,
    depthmap=None,
    camera_intrinsics=None,
    force=True,
    additional_quantities_to_be_resized_with_nearest=None,
):
    """
    Rescale the image and depthmap to the output resolution.
    If the image is larger than the output resolution, it is rescaled with lanczos interpolation.
    If force is false and the image is smaller than the output resolution, it is not rescaled.
    If force is true and the image is smaller than the output resolution, it is rescaled with bicubic interpolation.
    Depth and other quantities are rescaled with nearest interpolation.

    Args:
        image (PIL.Image.Image or np.ndarray): The input image to be rescaled.
        output_resolution (tuple): The desired output resolution as a tuple (width, height).
        depthmap (np.ndarray, optional): The depth map associated with the image. Defaults to None.
        camera_intrinsics (np.ndarray, optional): The camera intrinsics matrix. Defaults to None.
        force (bool, optional): If True, force rescaling even if the image is smaller than the output resolution. Defaults to True.
        additional_quantities_to_be_resized_with_nearest (list of np.ndarray, optional): Additional quantities to be rescaled using nearest interpolation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - The rescaled image (PIL.Image.Image)
            - The rescaled depthmap (numpy.ndarray or None)
            - The updated camera intrinsics (numpy.ndarray or None)
            - The list of rescaled additional quantities (list of numpy.ndarray or None)
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W, H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    if additional_quantities_to_be_resized_with_nearest is not None:
        assert all(
            tuple(additional_quantity.shape[:2]) == image.size[::-1]
            for additional_quantity in additional_quantities_to_be_resized_with_nearest
        )

    # Define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if (
        scale_final >= 1 and not force
    ):  # image is already smaller than what is asked
        output = (
            image.to_pil(),
            depthmap,
            camera_intrinsics,
            additional_quantities_to_be_resized_with_nearest,
        )
        return output
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # First rescale the image so that it contains the crop
    image = image.resize(
        tuple(output_resolution),
        resample=lanczos if scale_final < 1 else bicubic,
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    if additional_quantities_to_be_resized_with_nearest is not None:
        resized_additional_quantities = []
        for quantity in additional_quantities_to_be_resized_with_nearest:
            resized_additional_quantities.append(
                cv2.resize(
                    quantity,
                    output_resolution,
                    fx=scale_final,
                    fy=scale_final,
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        additional_quantities_to_be_resized_with_nearest = (
            resized_additional_quantities
        )

    # No offset here; simple rescaling
    if camera_intrinsics is not None:
        camera_intrinsics = camera_matrix_of_crop(
            camera_intrinsics,
            input_resolution,
            output_resolution,
            scaling=scale_final,
        )

    # Return
    return (
        image.to_pil(),
        depthmap,
        camera_intrinsics,
        additional_quantities_to_be_resized_with_nearest,
    )


class ImageList:
    """
    Convenience class to apply the same operation to a whole set of images.

    This class wraps a list of PIL.Image objects and provides methods to perform
    operations on all images simultaneously.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        """Return the number of images in the list."""
        return len(self.images)

    def to_pil(self):
        """
        Convert ImageList back to PIL Image(s).

        Returns:
            PIL.Image.Image or tuple: Single PIL Image if list contains one image,
                                      or tuple of PIL Images if multiple images
        """
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        """
        Get the size of images in the list.

        Returns:
            tuple: (width, height) of the images

        Raises:
            AssertionError: If images have different sizes
        """
        sizes = [im.size for im in self.images]
        assert all(
            sizes[0] == s for s in sizes
        ), "All images must have the same size"
        return sizes[0]

    def resize(self, *args, **kwargs):
        """
        Resize all images with the same parameters.

        Args:
            *args, **kwargs: Arguments passed to PIL.Image.resize()

        Returns:
            ImageList: New ImageList containing resized images
        """
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        """
        Crop all images with the same parameters.

        Args:
            *args, **kwargs: Arguments passed to PIL.Image.crop()

        Returns:
            ImageList: New ImageList containing cropped images
        """
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        """
        Apply a PIL.Image method to all images in the list.

        Args:
            func (str): Name of the PIL.Image method to call
            *args, **kwargs: Arguments to pass to the method

        Returns:
            list: List of results from applying the method to each image
        """
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def bbox_from_intrinsics_in_out(
    input_camera_matrix, output_camera_matrix, output_resolution
):
    """
    Calculate the bounding box for cropping based on input and output camera intrinsics.

    Args:
        input_camera_matrix (numpy.ndarray): Original camera intrinsics matrix
        output_camera_matrix (numpy.ndarray): Target camera intrinsics matrix
        output_resolution (tuple): Target resolution as (width, height)

    Returns:
        tuple: Crop bounding box as (left, top, right, bottom)
    """
    out_width, out_height = output_resolution
    left, top = np.int32(
        np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2])
    )
    crop_bbox = (left, top, left + out_width, top + out_height)
    return crop_bbox


def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    """
    Calculate the camera matrix for a cropped image.

    Args:
        input_camera_matrix (numpy.ndarray): Original camera intrinsics matrix
        input_resolution (tuple or numpy.ndarray): Original image resolution as (width, height)
        output_resolution (tuple or numpy.ndarray): Target image resolution as (width, height)
        scaling (float, optional): Scaling factor for the image. Defaults to 1.
        offset_factor (float, optional): Factor to determine crop offset. Defaults to 0.5 (centered).
        offset (tuple or numpy.ndarray, optional): Explicit offset to use. If None, calculated from offset_factor.

    Returns:
        numpy.ndarray: Updated camera matrix for the cropped image
    """
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(
        input_camera_matrix
    )
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(
        output_camera_matrix_colmap
    )

    return output_camera_matrix


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5

    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5

    return K
