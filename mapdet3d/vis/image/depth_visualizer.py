"""Depth visualizer."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from mapdet3d.common.array import array_to_numpy
from mapdet3d.common.typing import (
    ArgsType,
    ArrayLikeFloat,
    NDArrayF32,
    NDArrayUI8,
)
from mapdet3d.vis.base import Visualizer
from mapdet3d.vis.image.util import preprocess_image
from mapdet3d.vis.util import generate_color_map

from .util import (
    colorize,
    get_pointcloud_from_rgbd,
    save_depth_map,
    save_file_ply,
)


@dataclass
class DataSample:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    image_name: str
    depth: NDArrayF32
    depth_gt: NDArrayF32 | None = None
    depth_error: NDArrayF32 | None = None
    points_rgb: NDArrayF32 | None = None


class DepthVisualizer(Visualizer):
    """Depth visualizer class."""

    def __init__(
        self,
        *args: ArgsType,
        max_depth: None | float = None,
        plot_error: bool = False,
        lift: bool = False,
        color_palette: list[tuple[int, int, int]] | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new Visualizer for Depth.

        Args:
            max_depth (None | float): Maximum depth to visualize.
        """
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self._samples: list[DataSample] = []
        self._gt_samples = []
        self.plot_error = plot_error
        self.lift = lift
        self.color_palette = (
            generate_color_map(50) if color_palette is None else color_palette
        )

    def __repr__(self):
        """String representation."""
        return f"DepthVisualizer(max_depth={self.max_depth}, plot_error={self.plot_error}, lift={self.lift})"

    def reset(self) -> None:
        """Reset the visualizer."""
        self._samples.clear()
        self._gt_samples.clear()

    def process(
        self,
        cur_iter: int,
        images: list[ArrayLikeFloat],
        image_names: list[str],
        depths: list[ArrayLikeFloat],
        depth_gts: ArrayLikeFloat | None = None,
        intrinsics: ArrayLikeFloat | None = None,
    ) -> None:
        """Process data of a batch of data."""
        if self._run_on_batch(cur_iter):
            for i, image in enumerate(images):
                image = preprocess_image(image)

                self._samples.append(
                    self.process_single_image(
                        image,
                        image_names[i],
                        array_to_numpy(depths[i]),
                        (
                            array_to_numpy(depth_gts[i])
                            if depth_gts is not None
                            else None
                        ),
                        (
                            array_to_numpy(intrinsics[i])
                            if intrinsics is not None
                            else None
                        ),
                    )
                )

    def process_single_image(
        self,
        image: NDArrayUI8,
        image_name: str,
        depth: NDArrayF32,
        depth_gt: NDArrayF32 | None = None,
        intrinsic: NDArrayF32 | None = None,
    ) -> DataSample:
        """Process data of a batch of data."""
        if self.max_depth is not None:
            mask = depth <= self.max_depth
        else:
            mask = np.full(depth.shape, True)

        if self.plot_error:
            assert (
                depth_gt is not None
            ), "Ground truth depth is required for plotting error."
            error = np.zeros_like(depth_gt)
            error[depth_gt > 0] = (
                np.abs(depth_gt - depth)[depth_gt > 0] / depth_gt[depth_gt > 0]
            )
        else:
            error = None

        if self.lift:
            assert (
                intrinsic is not None
            ), "Intrinsic matrix is required for lifting."
            points_rgb = get_pointcloud_from_rgbd(
                image, depth, intrinsic, mask
            )
        else:
            points_rgb = None

        return DataSample(
            image=image,
            image_name=image_name,
            depth=depth,
            depth_gt=depth_gt,
            depth_error=error,
            points_rgb=points_rgb,
        )

    def save_to_disk(self, cur_iter: int, output_folder: str) -> None:
        """Saves the visualization to disk.

        Args:
            cur_iter (int): Current iteration.
            output_folder (str): Folder where the output should be written.
        """
        if self._run_on_batch(cur_iter):
            for sample in self._samples:
                Image.fromarray(sample.image).save(
                    f"{output_folder}/{sample.image_name}.png",
                )

                if self.plot_error:
                    error = sample.depth_error

                    error_image = Image.fromarray(
                        colorize(
                            error.clip(0.0, 0.3),
                            vmin=0.001,
                            vmax=0.3,
                            cmap="coolwarm",
                        )
                    )

                    error_image.save(
                        f"{output_folder}/{sample.image_name}_error.png"
                    )

                save_depth_map(
                    sample.depth,
                    f"{output_folder}/{sample.image_name}_pred.png",
                )

                if sample.depth_gt is not None:
                    save_depth_map(
                        sample.depth_gt,
                        f"{output_folder}/{sample.image_name}_gt.png",
                    )

                if self.lift:
                    if sample.points_rgb is not None:
                        save_file_ply(
                            sample.points_rgb[:, :3],
                            sample.points_rgb[:, 3:],
                            os.path.join(
                                output_folder, f"{sample.image_name}.ply"
                            ),
                        )
