"""Bounding box 3D visualizer."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from mapdet3d.common.array import array_to_numpy
from mapdet3d.common.typing import (
    ArgsType,
    ArrayLike,
    ArrayLikeFloat,
    ArrayLikeInt,
    NDArrayF32,
    NDArrayUI8,
)
from mapdet3d.data.const import AxisMode
from mapdet3d.op.geometry.transform import inverse_rigid_transform
from mapdet3d.vis.base import Visualizer
from mapdet3d.vis.util import generate_color_map

from .canvas import PillowCanvasBackend
from .util import preprocess_boxes3d, preprocess_image, project_point
from .viewer import MatplotlibImageViewer


@dataclass
class DetectionBox3D:
    """Dataclass storing box informations."""

    corners: list[tuple[float, float, float]]
    label: str
    color: tuple[int, int, int]
    track_id: int | None


@dataclass
class DataSample:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    image_name: str
    intrinsics: NDArrayF32
    extrinsics: NDArrayF32 | None
    sequence_name: str | None
    camera_name: str | None
    boxes: list[DetectionBox3D]


class BoundingBox3DVisualizer(Visualizer):
    """Bounding box 3D visualizer class."""

    def __init__(
        self,
        *args: ArgsType,
        n_colors: int = 50,
        cat_mapping: dict[str, int] | None = None,
        file_type: str = "png",
        image_mode: str = "RGB",
        width: int = 2,
        camera_near_clip: float = 0.15,
        plot_heading: bool = True,
        axis_mode: AxisMode = AxisMode.ROS,
        trajectory_length: int = 10,
        plot_trajectory: bool = True,
        save_boxes3d: bool = False,
        canvas: PillowCanvasBackend | None = None,
        viewer: MatplotlibImageViewer | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new Visualizer for Image and 3D Bounding Boxes."""
        super().__init__(*args, **kwargs)
        self._samples: list[DataSample] = []
        self.axis_mode = axis_mode
        self.trajectories: dict[int, list[tuple[float, float, float]]] = (
            defaultdict(list)
        )
        self.trajectory_length = trajectory_length
        self.plot_trajectory = plot_trajectory

        self.color_palette = generate_color_map(n_colors)

        self.class_id_mapping = (
            {v: k for k, v in cat_mapping.items()}
            if cat_mapping is not None
            else {}
        )

        self.file_type = file_type
        self.image_mode = image_mode
        self.width = width

        self.camera_near_clip = camera_near_clip
        self.plot_heading = plot_heading
        self.save_boxes3d = save_boxes3d

        self.canvas = canvas if canvas is not None else PillowCanvasBackend()
        self.viewer = viewer if viewer is not None else MatplotlibImageViewer()

    def reset(self) -> None:
        """Reset visualizer."""
        self._samples.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return "BoundingBox3DVisualizer"

    def process(
        self,
        cur_iter: int,
        views: list[list[dict[str, ArrayLike]]] | None = None,
        images: list[ArrayLike] | None = None,
        image_names: list[str] | None = None,
        boxes3d: list[ArrayLikeFloat] | None = None,
        intrinsics: ArrayLikeFloat | None = None,
        extrinsics: None | ArrayLikeFloat = None,
        scores: None | list[ArrayLikeFloat] = None,
        class_ids: None | list[ArrayLikeInt] = None,
        track_ids: None | list[ArrayLikeInt] = None,
        sequence_names: None | list[str] = None,
        categories: None | list[list[str]] = None,
    ) -> None:
        """Processes a batch of data."""
        if self._run_on_batch(cur_iter):

            if views is not None:
                images = [v[0]["img"] for v in views]

                image_names = [n[0] for n in image_names]
                boxes3d = [b[0] for b in boxes3d]

                intrinsics = [K[0] for K in intrinsics]

                if categories is not None:
                    categories = [c[0] for c in categories]

                if track_ids is not None:
                    track_ids = [t[0] for t in track_ids]

            for batch, image in enumerate(images):
                self.process_single_image(
                    image,
                    image_names[batch],
                    boxes3d[batch],
                    intrinsics[batch],
                    (None if extrinsics is None else extrinsics[batch]),
                    None if scores is None else scores[batch],
                    None if class_ids is None else class_ids[batch],
                    None if track_ids is None else track_ids[batch],
                    None if sequence_names is None else sequence_names[batch],
                    None if categories is None else categories[batch],
                )

            for tid in self.trajectories:
                if len(self.trajectories[tid]) > self.trajectory_length:
                    self.trajectories[tid].pop(0)

    def process_single_image(
        self,
        image: ArrayLike,
        image_name: str,
        boxes3d: ArrayLikeFloat,
        intrinsics: ArrayLikeFloat,
        extrinsics: None | ArrayLikeFloat = None,
        scores: None | ArrayLikeFloat = None,
        class_ids: None | ArrayLikeInt = None,
        track_ids: None | ArrayLikeInt = None,
        sequence_name: None | str = None,
        categories: None | list[str] = None,
        camera_name: None | str = None,
    ) -> None:
        """Processes a single image entry.

        Args:
            image (ArrayLike): Image to show.
            image_name (str): Image name.
            boxes3d (ArrayLikeFloat): Predicted bounding boxes with shape
                [N, 10], where  N is the number of boxes.
            intrinsics (ArrayLikeFloat): Camera intrinsics with shape [3, 3].
            extrinsics (None | ArrayLikeFloat, optional): Camera extrinsics
                with shape [4, 4]. Defaults to None.
            scores (None | ArrayLikeFloat, optional): Predicted box scores of
                shape [N]. Defaults to None.
            class_ids (None | ArrayLikeInt, optional): Predicted class ids of
                shape [N]. Defaults to None.
            track_ids (None | ArrayLikeInt, optional): Predicted track ids of
                shape [N]. Defaults to None.
            sequence_name (None | str, optional): Sequence name. Defaults to
                None.
            categories (None | list[str], optional): List of categories for
                each box. Instead of class ids, the categories will be used to
                label the boxes. Defaults to None.
            camera_name (None | str, optional): Camera name. Defaults to None.
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        image_hw = (img_normalized.shape[0], img_normalized.shape[1])

        intrinsics_np = array_to_numpy(intrinsics, n_dims=2, dtype=np.float32)
        extrinsics_np = (
            array_to_numpy(extrinsics, n_dims=2, dtype=np.float32)
            if extrinsics is not None
            else None
        )
        data_sample = DataSample(
            img_normalized,
            image_name,
            intrinsics_np,
            extrinsics_np,
            sequence_name,
            camera_name,
            [],
        )

        if len(boxes3d) != 0:  # type: ignore
            for center, corners, label, color, track_id in zip(
                *preprocess_boxes3d(
                    image_hw,
                    boxes3d,
                    intrinsics,
                    extrinsics,
                    scores,
                    class_ids,
                    track_ids,
                    self.color_palette,
                    self.class_id_mapping,
                    axis_mode=self.axis_mode,
                    categories=categories,
                )
            ):
                data_sample.boxes.append(
                    DetectionBox3D(
                        corners=corners,
                        label=label,
                        color=color,
                        track_id=track_id,
                    )
                )
                if track_id is not None:
                    self.trajectories[track_id].append(center)

        self._samples.append(data_sample)

    def show(self, cur_iter: int, blocking: bool = True) -> None:
        """Shows the processed images in a interactive window.

        Args:
            cur_iter (int): Current iteration.
            blocking (bool): If the visualizer should be blocking i.e. wait for
                human input for each image. Defaults to True.
        """
        if self._run_on_batch(cur_iter):
            image_data = [self._draw_image(d) for d in self._samples]
            self.viewer.show_images(image_data, blocking=blocking)

    def _draw_image(self, sample: DataSample) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize.

        Returns:
            NDArrayUI8: A image with the visualized data sample.
        """
        self.canvas.create_canvas(sample.image)

        if self.plot_trajectory:
            assert (
                sample.extrinsics is not None
            ), "Extrinsics is needed to plot trajectory."
            global_to_cam = inverse_rigid_transform(
                torch.from_numpy(sample.extrinsics)
            ).numpy()

        for box in sample.boxes:
            self.canvas.draw_box_3d(
                box.corners,
                box.color,
                sample.intrinsics,
                self.width,
                self.camera_near_clip,
                self.plot_heading,
            )

            selected_corner = project_point(box.corners[2], sample.intrinsics)
            self.canvas.draw_text(
                (selected_corner[0], selected_corner[1]), box.label, box.color
            )

            if self.plot_trajectory:
                assert (
                    box.track_id is not None
                ), "track id must be set to plot trajectory."

                trajectory = self.trajectories[box.track_id]
                for center in trajectory:
                    # Move global center to current camera frame
                    center_cam = np.dot(global_to_cam, [*center, 1])[:3]

                    if center_cam[2] > 0:
                        projected_center = project_point(
                            center_cam, sample.intrinsics
                        )
                        self.canvas.draw_circle(
                            projected_center, box.color, self.width * 2
                        )

        return self.canvas.as_numpy_image()

    def save_to_disk(self, cur_iter: int, output_folder: str) -> None:
        """Saves the visualization to disk.

        Writes all processes samples to the output folder naming each image
        <sample.image_name>.<filetype>.

        Args:
            cur_iter (int): Current iteration.
            output_folder (str): Folder where the output should be written.
        """
        if self._run_on_batch(cur_iter):
            for sample in self._samples:
                output_dir = output_folder
                image_name = f"{sample.image_name}.{self.file_type}"

                self._draw_image(sample)

                if sample.sequence_name is not None:
                    output_dir = os.path.join(output_dir, sample.sequence_name)

                if sample.camera_name is not None:
                    output_dir = os.path.join(output_dir, sample.camera_name)

                os.makedirs(output_dir, exist_ok=True)
                self.canvas.save_to_disk(os.path.join(output_dir, image_name))

                if self.save_boxes3d:
                    corners = np.array([box.corners for box in sample.boxes])

                    np.save(
                        os.path.join(output_dir, f"{sample.image_name}.npy"),
                        corners,
                    )
