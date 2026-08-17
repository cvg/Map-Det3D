"""Rerun visualization."""

from __future__ import annotations

import os
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import trimesh
from scipy.spatial.transform import Rotation
from torch import Tensor

from mapdet3d.common.typing import ArgsType
from mapdet3d.data.const import AxisMode
from mapdet3d.op.box3d import transform_boxes3d
from mapdet3d.op.geometry.rotation import quaternion_to_matrix

from .base import Visualizer


class RerunVisualizer(Visualizer):
    """Rerun visualizer."""

    def __init__(
        self,
        *args: ArgsType,
        start_iter: int = 1,
        stop_iter: int = -1,
        image_plane_distance: float = 0.1,
        line_width: float = 0.02,
        plot_traj: bool = True,
        convert_to_world: bool = False,
        show_box_labels: bool = True,
        log_mesh: bool = False,
        save_to_disk: bool = False,
        output_dir: str = "work_dir/outputs",
        bg_color: tuple[int, int, int] = (255, 255, 255),
        mesh_z_clip_offset: float = 1.5,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.start_iter = start_iter
        self.stop_iter = stop_iter
        self.image_plane_distance = image_plane_distance
        self.line_width = line_width
        self.convert_to_world = convert_to_world

        self.bg_color = bg_color
        self.plot_traj = plot_traj
        self.show_box_labels = show_box_labels
        self.log_mesh = log_mesh

        self.save_to_disk = save_to_disk
        self.output_dir = os.path.join(output_dir, "rerun_vis")
        self.mesh_z_clip_offset = mesh_z_clip_offset

        self.traj_xyz: list[np.ndarray] = []
        self.gt_traj_xyz: list[np.ndarray] = []
        self.previous_tid: list[str] = []
        self.gt_previous_tid: list[str] = []

        rgb_contents = ["$origin/image", "world/predictions/**"]

        self.blueprint = rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D",
                background=self.bg_color,
                line_grid=False,
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                ),
            ),
            rrb.Spatial2DView(
                name="RGB", origin="world/camera", contents=rgb_contents
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return "RerunVisualizer for 3D Object Detection"

    def reset(self) -> None:
        """Reset visualizer."""
        pass

    def process(
        self,
        cur_iter: int,
        images: list[Tensor | np.ndarray],
        sequence_names: list[str],
        original_hw: list[tuple[int, int]],
        intrinsics: list[Tensor | np.ndarray],
        extrinsics: list[Tensor | np.ndarray],
        boxes3d: list[Tensor | np.ndarray] | None = None,
        scores: list[Tensor | np.ndarray] | None = None,
        track_ids: list[Tensor | np.ndarray] | None = None,
        mesh_paths: list[str] | None = None,
    ) -> None:
        """Processes a batch of data."""
        assert len(images) == 1, "Batch size must be 1 for RerunVisualizer."

        if cur_iter < self.start_iter:
            return

        seq_name = sequence_names[0]

        if cur_iter == self.start_iter:
            rr.init("rerun_vis", default_blueprint=self.blueprint)

            if self.save_to_disk:
                # Save directly to disk without starting a gRPC server.
                os.makedirs(self.output_dir, exist_ok=True)

                rr.set_sinks(
                    rr.FileSink(
                        os.path.join(self.output_dir, f"{seq_name}.rrd")
                    ),
                    default_blueprint=self.blueprint,
                )
            else:
                # Start a gRPC server and web viewer for live viewing.
                server_uri = rr.serve_grpc(default_blueprint=self.blueprint)
                rr.serve_web_viewer(connect_to=server_uri)

            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        rr.set_time("pts", timestamp=cur_iter)

        if self.log_mesh:
            self._log_mesh(mesh_paths)

        image = self._prepare_image(images[0])
        intrinsics_np = self._to_numpy(intrinsics[0])
        extrinsics_np = self._to_numpy(extrinsics[0])
        hw = original_hw[0]

        self._log_camera(image, intrinsics_np, extrinsics_np, hw)

        self.previous_tid = self._log_boxes(
            boxes3d[0],
            extrinsics_np,
            track_ids[0],
            previous_tid=self.previous_tid,
            scores=scores[0] if scores is not None else None,
        )

        if cur_iter == self.stop_iter:
            self._keep_server_running()

    def _log_mesh(self, mesh_paths: list[str] | None) -> None:
        """Load and log a static world mesh."""

        assert (
            mesh_paths is not None and len(mesh_paths) == 1
        ), "Batch size must be 1 for RerunVisualizer."

        mesh = trimesh.load(mesh_paths[0])

        # Remove everything above (z_max - offset) to strip ceilings.
        z_top = mesh.vertices[mesh.faces, 2].max(axis=1).max()
        z_max = z_top - self.mesh_z_clip_offset
        mask = mesh.vertices[mesh.faces, 2].max(axis=1) < z_max
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()

        rr.log(
            "world/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=mesh.visual.vertex_colors,
            ),
        )

    def _keep_server_running(self) -> None:
        """Block so the web viewer has time to load before exit."""
        print("\nKeep server running...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
            exit(0)

    # ------------------------------------------------------------------
    # Camera logging
    # ------------------------------------------------------------------
    def _log_camera(
        self,
        image: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        original_hw: tuple[int, int],
    ) -> None:
        """Log the ground-truth camera, image, and trajectory."""
        rr.log("world/camera", self._make_pinhole(intrinsics, original_hw))
        rr.log("world/camera", self._pose_transform(extrinsics))
        rr.log("world/camera/image", rr.Image(image, opacity=0.5))

        if self.plot_traj:
            self.gt_traj_xyz.append(extrinsics[:3, 3])
            rr.log(
                "world/trajectory",
                rr.LineStrips3D(
                    [np.array(self.gt_traj_xyz)],
                    colors=[84, 255, 159],
                    radii=self.line_width,
                ),
            )

    # ------------------------------------------------------------------
    # Box logging
    # ------------------------------------------------------------------
    def _log_boxes(
        self,
        boxes: Tensor | np.ndarray,
        extrinsics: np.ndarray,
        track_ids,
        previous_tid: list[str] | None = None,
        scores: list[float] | None = None,
    ) -> list[str]:
        """Clear prior entities, then dispatch on box representation."""
        self._clear_entities(previous_tid or [])

        boxes = self._to_numpy(boxes)

        return self._log_param_boxes(
            boxes,
            extrinsics,
            track_ids,
            scores=scores,
        )

    def _log_param_boxes(
        self,
        boxes: np.ndarray,
        extrinsics: np.ndarray,
        track_ids: list[int] | np.ndarray,
        scores: list[float] | None,
    ) -> list[str]:
        """Log (xyz, wlh, quat) boxes transformed to world frame."""
        if self.convert_to_world:
            boxes_world = torch.from_numpy(boxes)
        else:
            boxes_world = transform_boxes3d(
                torch.from_numpy(boxes),
                torch.from_numpy(extrinsics),
                AxisMode.OPENCV,
                AxisMode.ROS,
            )

        current_tid: list[str] = []
        for i, box in enumerate(boxes_world.numpy()):
            entity_path = f"world/predictions/box-{track_ids[i]}"
            # w, l, h -> size_y, size_x, size_z
            half_size = 0.5 * box[3:6][[1, 0, 2]]
            centroid = box[:3]
            mat3x3 = quaternion_to_matrix(
                torch.from_numpy(box[6:10].reshape(1, 4))
            ).numpy()[0]

            rr.log(
                entity_path,
                rr.Boxes3D(
                    half_sizes=half_size,
                    labels=f"{scores[i]:.2f}" if scores is not None else None,
                    radii=self.line_width,
                ),
                rr.InstancePoses3D(translations=centroid, mat3x3=mat3x3),
            )
            current_tid.append(entity_path)
        return current_tid

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
        """Normalize a tensor or array to a numpy array on CPU."""
        return x.cpu().numpy() if isinstance(x, Tensor) else x

    @staticmethod
    def _prepare_image(image: Tensor | np.ndarray) -> np.ndarray:
        """Unwrap batch dim, flip CHW->HWC, and cast to uint8."""
        arr = RerunVisualizer._to_numpy(image)
        arr = arr[0]
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr.astype(np.uint8)

    def _make_pinhole(
        self,
        intrinsics: np.ndarray,
        original_hw: tuple[int, int],
    ) -> rr.Pinhole:
        """Build a Rerun Pinhole with the configured plane distance."""
        return rr.Pinhole(
            image_from_camera=intrinsics,
            resolution=(original_hw[1], original_hw[0]),
            image_plane_distance=self.image_plane_distance,
        )

    @staticmethod
    def _pose_transform(rt: np.ndarray) -> rr.Transform3D:
        """Convert a 4x4 extrinsics matrix to a Rerun Transform3D."""
        return rr.Transform3D(
            translation=rt[:3, 3],
            rotation=rr.Quaternion(
                xyzw=Rotation.from_matrix(rt[:3, :3]).as_quat()
            ),
        )

    @staticmethod
    def _clear_entities(entity_paths: list[str]) -> None:
        """Clear prior per-box entities without recursion."""
        for path in entity_paths:
            rr.log(path, rr.Clear(recursive=False))
