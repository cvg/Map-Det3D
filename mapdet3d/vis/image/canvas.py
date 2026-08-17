"""Pillow backend implementation to draw on images."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageFont import ImageFont, load_default

from mapdet3d.common.typing import (
    NDArrayBool,
    NDArrayF32,
    NDArrayF64,
    NDArrayUI8,
)

from .util import get_intersection_point, project_point


class PillowCanvasBackend:
    """Canvas backend using Pillow."""

    def __init__(
        self, font: ImageFont | None = None, font_size: int | None = None
    ) -> None:
        """Creates a new canvas backend.

        Args:
            font (ImageFont): Pillow font to use for the label.
            font_size (int): Font size to use for the label.
        """
        self._image_draw: ImageDraw.ImageDraw | None = None
        self._font = font if font is not None else load_default(font_size)
        self._image: Image.Image | None = None

    def create_canvas(
        self,
        image: NDArrayUI8 | None = None,
        image_hw: tuple[int, int] | None = None,
    ) -> None:
        """Creates a new canvas with a given image or shape internally.

        Either provide a background image or the desired height, width
        of the canvas.

        Args:
            image (np.array[uint8] | None): Numpy array with a background image
            image_hw (tuple[int, int] | None): height, width of the canvas

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if image_hw is not None:
            white_image = np.ones([*image_hw, 3]) * 255
            image = white_image.astype(np.uint8)
        else:
            assert (
                image is not None
            ), "Image or Image Shapes required to create canvas"

        self._image = Image.fromarray(image)
        self._image_draw = ImageDraw.Draw(self._image)

    def draw_bitmap(
        self,
        bitmap: NDArrayBool,
        color: tuple[int, int, int],
        top_left_corner: tuple[float, float] = (0, 0),
        alpha: float = 0.5,
    ) -> None:
        """Draws a binary mask onto the given canvas.

        Args:
            bitmap (ndarray): The binary mask to draw.
            color (tuple[int, int, int]): Color of the box [0,255].
            top_left_corner (tuple(float, float)): Coordinates of top left
                corner of the bitmap.
            alpha (float): Alpha value for transparency of this mask.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        mask = np.squeeze(bitmap)
        assert len(mask.shape) == 2, "Bitmap expected to have shape [h,w]"

        bitmap_with_alpha: NDArrayF64 = np.repeat(
            mask[:, :, None], 4, axis=2
        ).astype(np.float64)
        bitmap_with_alpha[..., -1] = bitmap_with_alpha[..., -1] * alpha * 255
        bitmap_pil = Image.fromarray(
            bitmap_with_alpha.astype(np.uint8), mode="RGBA"
        )
        self._image_draw.bitmap(
            top_left_corner, bitmap_pil, fill=color  # type: ignore
        )

    def draw_text(
        self,
        position: tuple[float, float],
        text: str,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Draw text onto canvas at given position.

        Args:
            position (tuple[float, float]): x,y position where the text will
                start.
            text (str): Text to be placed at the given location.
            color (tuple[int, int, int], optional): Text color. Defaults to
                (255, 255, 255).

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        left, top, right, bottom = self._image_draw.textbbox(
            position, text, font=self._font
        )
        self._image_draw.rectangle(
            (left - 2, top - 2, right + 2, bottom + 2), fill=color
        )
        self._image_draw.text(position, text, (255, 255, 255), font=self._font)

    def draw_box(
        self,
        corners: tuple[float, float, float, float],
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draws a box onto the given canvas.

        Args:
            corners (list[float]): Containing [x1,y2,x2,y2] the corners of
                the box.
            color (tuple[int, int, int]): Color of the box [0,255].
            width (int, optional): Line width. Defaults to 1.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )

        self._image_draw.rectangle(corners, outline=color, width=width)

    def draw_rotated_box(
        self,
        corners: list[tuple[float, float]],
        color: tuple[int, int, int],
        width: int = 0,
    ) -> None:
        """Draws a box onto the given canvas.

        Corner ordering:

        (2) +---------+ (3)
            |         |
            |         |
            |         |
        (0) +---------+ (1)

        Args:
            corners (list[tuple[float, float]]): Containing the four corners of
                the box.
            color (tuple[int, int, int]): Color of the box [0,255].
            width (int, optional): Line width. Defaults to 0.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        assert len(corners) == 4, "2D box must consist of 4 corner points."
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )

        self.draw_line(corners[0], corners[1], color, 2 * width)
        self.draw_line(corners[0], corners[2], color, width)
        self.draw_line(corners[1], corners[3], color, width)
        self.draw_line(corners[2], corners[3], color, width)

        center_forward = np.mean(corners[:2], axis=0, dtype=np.float32)
        center = np.mean(corners, axis=0, dtype=np.float32)
        self.draw_line(
            tuple(center.tolist()),
            tuple(center_forward.tolist()),
            color,
            width,
        )

    def draw_line(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        color: tuple[int, int, int],
        width: int = 0,
    ) -> None:
        """Draw a line onto canvas from point 1 to 2.

        Args:
            point1 (tuple[float, float]): Start point (2D pixel coordinates).
            point2 (tuple[float, float]): End point (2D pixel coordinates).
            color (tuple[int, int, int]): Color of the line.
            width (int, optional): Line width. Defaults to 0.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.line((point1, point2), width=width, fill=color)

    def draw_circle(
        self,
        center: tuple[float, float],
        color: tuple[int, int, int],
        radius: int = 2,
    ) -> None:
        """Draw a circle onto canvas.

        Args:
            center (tuple[float, float]): Center of the circle.
            color (tuple[int, int, int]): Color of the circle.
            radius (int, optional): Radius of the circle. Defaults to 2.
        """
        x1 = center[0] - radius
        y1 = center[1] - radius
        x2 = center[0] + radius
        y2 = center[1] + radius
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)

    def _draw_box_3d_line(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        color: tuple[int, int, int],
        intrinsics: NDArrayF32,
        width: int = 0,
        camera_near_clip: float = 0.15,
    ) -> None:
        """Draws a line between two points.

        Args:
            point1 (tuple[float, float, float]): The first point. The third
                coordinate is the depth.
            point2 (tuple[float, float, float]): The first point. The third
                coordinate is the depth.
            color (tuple[int, int, int]): Color of the line.
            intrinsics (NDArrayF32): Camera intrinsics matrix.
            width (int, optional): The width of the line. Defaults to 0.
            camera_near_clip (float, optional): The near clipping plane of the
                camera. Defaults to 0.15.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if point1[2] < camera_near_clip and point2[2] < camera_near_clip:
            return

        if point1[2] < camera_near_clip:
            point1 = get_intersection_point(point1, point2, camera_near_clip)
        elif point2[2] < camera_near_clip:
            point2 = get_intersection_point(point1, point2, camera_near_clip)

        pt1 = project_point(point1, intrinsics)
        pt2 = project_point(point2, intrinsics)

        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.line((pt1, pt2), width=width, fill=color)

    def draw_box_3d(
        self,
        corners: list[tuple[float, float, float]],
        color: tuple[int, int, int],
        intrinsics: NDArrayF32,
        width: int = 0,
        camera_near_clip: float = 0.15,
        plot_heading: bool = True,
    ) -> None:
        """Draws a 3D box onto the given canvas."""
        # Draw Front
        self._draw_box_3d_line(
            corners[1], corners[5], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[5], corners[6], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[6], corners[2], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[2], corners[1], color, intrinsics, width, camera_near_clip
        )

        # Draw Sides
        self._draw_box_3d_line(
            corners[1], corners[0], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[5], corners[4], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[6], corners[7], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[2], corners[3], color, intrinsics, width, camera_near_clip
        )

        # Draw Back
        self._draw_box_3d_line(
            corners[0], corners[4], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[4], corners[7], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[7], corners[3], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[3], corners[0], color, intrinsics, width, camera_near_clip
        )

        # Draw line indicating the front
        if plot_heading:
            v2 = np.array(corners[2])
            v6 = np.array(corners[6])
            v3 = np.array(corners[3])
            v7 = np.array(corners[7])
            center_bottom_forward = np.mean([v2, v6], axis=0, dtype=np.float32)
            center_bottom = np.mean([v2, v3, v6, v7], axis=0, dtype=np.float32)
            self._draw_box_3d_line(
                tuple(center_bottom.tolist()),
                tuple(center_bottom_forward.tolist()),
                color,
                intrinsics,
                width,
                camera_near_clip,
            )

    def as_numpy_image(self) -> NDArrayUI8:
        """Returns the current canvas as numpy image.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image is None:
            raise ValueError(
                "No Image initialized! Did you call 'create_canvas'?"
            )
        return np.asarray(self._image)

    def save_to_disk(self, image_path: str) -> None:
        """Writes the current canvas to disk.

        Args:
            image_path (str): Full image path (with file name and ending).

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image is None:
            raise ValueError(
                "No Image initialized! Did you call 'create_canvas'?"
            )
        self._image.save(image_path)
