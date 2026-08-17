"""Utility functions for 3D bounding boxes."""

from __future__ import annotations

import torch
from torch import Tensor
from vis4d_cuda_ops import iou_box3d

from mapdet3d.data.const import AxisMode
from mapdet3d.op.geometry.projection import project_points
from mapdet3d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mapdet3d.op.geometry.transform import transform_points

from .iou3d import check_coplanar, check_nonzero


def boxes3d_to_corners(
    boxes3d: Tensor, axis_mode: AxisMode = AxisMode.OPENCV
) -> Tensor:
    """Convert a Tensor of 3D boxes to its respective corner points.

    Args:
        boxes3d (Tensor): Box parameters. Tensor of shape [N, 10].
        axis_mode (AxisMode): Coordinate system convention.

    Returns:
        Tensor: [N, 8, 3] 3D bounding box corner coordinates, in this order:

                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
    """
    w = boxes3d[:, 3:4]
    l = boxes3d[:, 4:5]
    h = boxes3d[:, 5:6]

    if axis_mode == AxisMode.OPENCV:
        sizes = torch.cat([l, h, w], dim=1)
        unit_corners = (
            torch.tensor(
                [
                    [-1, -1, -1],
                    [+1, -1, -1],
                    [+1, +1, -1],
                    [-1, +1, -1],
                    [-1, -1, +1],
                    [+1, -1, +1],
                    [+1, +1, +1],
                    [-1, +1, +1],
                ],
                dtype=torch.float32,
            )
            * 0.5
        )

    elif axis_mode == AxisMode.ROS:
        sizes = torch.cat([l, w, h], dim=1)
        unit_corners = (
            torch.tensor(
                [
                    [-1, -1, +1],
                    [+1, -1, +1],
                    [+1, -1, -1],
                    [-1, -1, -1],
                    [-1, +1, +1],
                    [+1, +1, +1],
                    [+1, +1, -1],
                    [-1, +1, -1],
                ],
                dtype=torch.float32,
            )
            * 0.5
        )
    else:
        raise NotImplementedError

    unit_corners = unit_corners.to(device=boxes3d.device, dtype=boxes3d.dtype)

    # [N, 8, 3] local corners scaled to each box's extents.
    verts = unit_corners.unsqueeze(0) * sizes.unsqueeze(1)

    # Rotate (R @ v) per corner, batched as verts @ Rᵀ.
    rotation_matrix = quaternion_to_matrix(boxes3d[:, 6:])
    verts = verts @ rotation_matrix.transpose(-1, -2)

    # Translate by center.
    verts = verts + boxes3d[:, :3].unsqueeze(1)

    return verts.contiguous()


def boxes3d_in_image(
    box_corners: Tensor, cam_intrinsics: Tensor, image_hw: tuple[int, int]
) -> Tensor:
    """Check if a 3D bounding box is (partially) in an image.

    Args:
        box_corners (Tensor): [N, 8, 3] Tensor of 3D boxes corners. In OpenCV
            coordinate frame.
        cam_intrinsics (Tensor): [3, 3] Camera matrix.
        image_hw (tuple[int, int]): image height / width.

    Returns:
        Tensor: [N,] boolean values.
    """
    points = project_points(box_corners.view(-1, 3), cam_intrinsics).view(
        -1, 8, 2
    )
    mask = (points[..., 0] >= 0) * (points[..., 0] < image_hw[1]) * (
        points[..., 1] >= 0
    ) * (points[..., 1] < image_hw[0]) * box_corners[..., 2] > 0.0
    mask = mask.any(dim=-1)
    return mask


def boxes3d_to_boxes2d(
    boxes3d: Tensor,
    cam_intrinsics: Tensor,
    axis_mode: AxisMode = AxisMode.OPENCV,
    camera_near_clip: float = 0.0,
    image_hw: tuple[int, int] | None = None,
) -> Tensor:
    """Convert 3D bounding boxes to 2D bounding boxes via projection.

    This function converts 3D boxes to their 8 corner points, projects them
    to 2D image coordinates, and computes the tight 2D bounding box enclosing
    all projected corners.

    Args:
        boxes3d (Tensor): [N, 10] Tensor of 3D boxes. Format: [x, y, z, w, l, h, qw, qx, qy, qz]
            where (x,y,z) is center, (w,l,h) are dimensions, and (qw,qx,qy,qz) is quaternion.
        cam_intrinsics (Tensor): [3, 3] Camera intrinsic matrix.
        axis_mode (AxisMode): Coordinate system convention. Default: AxisMode.OPENCV.
        image_hw (tuple[int, int] | None): Optional image (height, width) to clip
            bounding boxes to image boundaries.

    Returns:
        Tensor: [N, 4] 2D bounding boxes in xyxy format (x1, y1, x2, y2).
            Boxes with all corners behind the camera (z <= 0) will have all zeros.
    """
    # Convert 3D boxes to 8 corner points
    box_corners = boxes3d_to_corners(boxes3d, axis_mode)  # [N, 8, 3]

    # Project 3D corners to 2D image coordinates
    corners_2d = project_points(
        box_corners.view(-1, 3), cam_intrinsics
    )  # [N*8, 2]
    corners_2d = corners_2d.view(-1, 8, 2)  # [N, 8, 2]

    # Filter out corners behind the camera (z <= 0)
    valid_depth = box_corners[..., 2] > camera_near_clip  # [N, 8]

    # Initialize 2D boxes
    boxes2d = torch.zeros(
        (boxes3d.shape[0], 4), dtype=boxes3d.dtype, device=boxes3d.device
    )

    for i in range(boxes3d.shape[0]):
        if valid_depth[i].any():
            # Get valid corners (in front of camera)
            valid_corners = corners_2d[i][
                valid_depth[i]
            ]  # [K, 2] where K <= 8

            # Compute bounding box as min/max of projected corners
            x_min = valid_corners[:, 0].min()
            y_min = valid_corners[:, 1].min()
            x_max = valid_corners[:, 0].max()
            y_max = valid_corners[:, 1].max()

            # Clip to image boundaries if provided
            if image_hw is not None:
                x_min = torch.clamp(x_min, min=0, max=image_hw[1])
                y_min = torch.clamp(y_min, min=0, max=image_hw[0])
                x_max = torch.clamp(x_max, min=0, max=image_hw[1])
                y_max = torch.clamp(y_max, min=0, max=image_hw[0])

            boxes2d[i] = torch.tensor(
                [x_min, y_min, x_max, y_max],
                dtype=boxes3d.dtype,
                device=boxes3d.device,
            )

    return boxes2d


def transform_boxes3d(
    boxes3d: Tensor,
    transform_matrix: Tensor,
    source_axis_mode: AxisMode,
    target_axis_mode: AxisMode,
) -> Tensor:
    """Transform 3D boxes using given transform matrix.

    This function transforms 3D bounding boxes from one coordinate system to
    another, handling center transformation, dimension reordering, and
    orientation correction.

    Args:
        boxes3d (Tensor): [N, 10] Tensor of 3D boxes. Format:
            [x, y, z, w, l, h, qw, qx, qy, qz] where (x,y,z) is center,
            (w,l,h) are dimensions, and (qw,qx,qy,qz) is quaternion.
        transform_matrix (Tensor): [4, 4] Transform matrix.
        source_axis_mode (AxisMode): Source coordinate system convention.
        target_axis_mode (AxisMode): Target coordinate system convention.

    Returns:
        Tensor: [N, 10] Transformed 3D boxes in target coordinate system.

    Raises:
        NotImplementedError: If the axis mode conversion is not supported.
    """
    n_boxes = boxes3d.shape[0]
    boxes3d_transformed = boxes3d.new_zeros(boxes3d.shape)

    # Transform center
    boxes3d_transformed[:, :3] = transform_points(
        boxes3d[:, :3], transform_matrix
    )

    # NOTE: Dimension is the same but the axis will be different
    # ROS: (w, l, h) -> (dy, dx, dz) where Z is height (up)
    # OPENCV: (w, l, h) -> (dz, dx, dy) where Y is height (down)
    boxes3d_transformed[:, 3:6] = boxes3d[:, 3:6]

    # Transform orientation
    if (
        source_axis_mode == AxisMode.OPENCV
        and target_axis_mode == AxisMode.ROS
    ):
        # Transform orientation:
        # 1. Apply rigid rotation from transform matrix
        rot_quat = matrix_to_quaternion(
            transform_matrix[:3, :3].unsqueeze(0)
        )  # [1, 4]
        rigid_quat = quaternion_multiply(
            rot_quat.expand(n_boxes, -1), boxes3d[:, 6:]
        )  # [N, 4]

        # 2. Apply axis correction: 90° rotation around local X-axis
        correction_euler = boxes3d.new_tensor([[torch.pi / 2, 0.0, 0.0]])
        correction_mat = euler_angles_to_matrix(
            correction_euler, convention="XYZ"
        )  # [1, 3, 3]
        correction_quat = matrix_to_quaternion(correction_mat)  # [1, 4]

        # Final quaternion: Q_world = Q_rigid * Q_correction
        boxes3d_transformed[:, 6:] = quaternion_multiply(
            rigid_quat, correction_quat.expand(n_boxes, -1)
        )
    elif (
        source_axis_mode == AxisMode.ROS
        and target_axis_mode == AxisMode.OPENCV
    ):
        # Transform orientation:
        # 1. Apply rigid rotation from transform matrix
        rot_quat = matrix_to_quaternion(
            transform_matrix[:3, :3].unsqueeze(0)
        )  # [1, 4]
        rigid_quat = quaternion_multiply(
            rot_quat.expand(n_boxes, -1), boxes3d[:, 6:]
        )  # [N, 4]

        # 2. Apply inverse axis correction: -90° rotation around local X-axis
        correction_euler = boxes3d.new_tensor([[-torch.pi / 2, 0.0, 0.0]])
        correction_mat = euler_angles_to_matrix(
            correction_euler, convention="XYZ"
        )  # [1, 3, 3]
        correction_quat = matrix_to_quaternion(correction_mat)  # [1, 4]

        # Final quaternion: Q_cam = Q_rigid * Q_correction
        boxes3d_transformed[:, 6:] = quaternion_multiply(
            rigid_quat, correction_quat.expand(n_boxes, -1)
        )
    else:
        raise NotImplementedError(
            f"Axis mode conversion from {source_axis_mode} to "
            f"{target_axis_mode} is not implemented."
        )

    return boxes3d_transformed


def box3d_overlap(
    boxes_dt: Tensor,
    boxes_gt: Tensor,
    eps_coplanar: float = 1e-4,
    eps_nonzero: float = 1e-8,
) -> Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero
    invalid_coplanar = ~check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero = ~check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print(
            "Warning: skipping {:d} non-coplanar boxes at eval.".format(
                int(invalid_coplanar.float().sum())
            )
        )

    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print(
            "Warning: skipping {:d} zero volume boxes at eval.".format(
                int(invalid_nonzero.float().sum())
            )
        )

    return ious
