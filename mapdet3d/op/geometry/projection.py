"""Projection utilities."""

from __future__ import annotations

import torch

from .transform import inverse_pinhole


def project_points(
    points: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Project points to pixel coordinates with given intrinsics.

    Args:
        points: (N, 3) or (B, N, 3) 3D coordinates.
        intrinsics: (3, 3) or (B, 3, 3) intrinsic camera matrices.

    Returns:
        torch.Tensor: (N, 2) or (B, N, 2) 2D pixel coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    assert points.shape[-1] == 3, "Input coordinates must be 3 dimensional!"
    hom_coords = points / points[..., 2:3]
    if len(hom_coords.shape) == 2:
        assert (
            len(intrinsics.shape) == 2
        ), "Got multiple intrinsics for single point set!"
        intrinsics = intrinsics.T
    elif len(hom_coords.shape) == 3:
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        intrinsics = intrinsics.permute(0, 2, 1)
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    pts_2d = hom_coords @ intrinsics
    return pts_2d[..., :2].to(points.dtype)


def unproject_points(
    points: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Un-projects pixel coordinates to 3D coordinates with given intrinsics.

    Args:
        points: (N, 2) or (B, N, 2) 2D pixel coordinates.
        depths: (N,) / (N, 1) or (B, N,) / (B, N, 1) depth values.
        intrinsics: (3, 3) or (B, 3, 3) intrinsic camera matrices.

    Returns:
        torch.Tensor: (N, 3) or (B, N, 3) 3D coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    if len(points.shape) == 2:
        assert (
            len(intrinsics.shape) == 2 or intrinsics.shape[0] == 1
        ), "Got multiple intrinsics for single point set!"
        if len(intrinsics.shape) == 3:
            intrinsics = intrinsics.squeeze(0)
        inv_intrinsics = inverse_pinhole(intrinsics).transpose(0, 1)
        if len(depths.shape) == 1:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 2, "depths must have same dims as points"
    elif len(points.shape) == 3:
        inv_intrinsics = inverse_pinhole(intrinsics).transpose(-2, -1)
        if len(depths.shape) == 2:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 3, "depths must have same dims as points"
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    hom_coords = torch.cat([points, torch.ones_like(points)[..., 0:1]], -1)
    pts_3d = hom_coords @ inv_intrinsics
    pts_3d *= depths
    return pts_3d


def points_inside_image(
    points_coord: torch.Tensor,
    depths: torch.Tensor,
    images_hw: torch.Tensor | tuple[int, int],
) -> torch.Tensor:
    """Generate binary mask.

    Creates a mask that is true for all point coordiantes that lie inside the
    image,

    Args:
        points_coord (torch.Tensor): 2D pixel coordinates of shape [..., 2].
        depths (torch.Tensor): Associated depth of each 2D pixel coordinate.
        images_hw:  (torch.Tensor| tuple[int, int]]) Associated tensor of image
                    dimensions, shape [..., 2] or single height, width pair.

    Returns:
        torch.Tensor: Binary mask of points inside an image.
    """
    mask = torch.ones_like(depths)
    h: int | torch.Tensor
    w: int | torch.Tensor

    if isinstance(images_hw, tuple):
        h, w = images_hw
    else:
        h, w = images_hw[..., 0], images_hw[..., 1]
    mask = torch.logical_and(mask, torch.greater(depths, 0))
    mask = torch.logical_and(mask, points_coord[..., 0] > 0)
    mask = torch.logical_and(mask, points_coord[..., 0] < w - 1)
    mask = torch.logical_and(mask, points_coord[..., 1] > 0)
    mask = torch.logical_and(mask, points_coord[..., 1] < h - 1)
    return mask


def generate_depth_map(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Generate depth map for given pointcloud.

    Args:
        points: (N, 3) coordinates.
        intrinsics: (3, 3) intrinsic camera matrices.
        image_hw: (tuple[int,int]) height, width of the image

    Returns:
        torch.Tensor: Projected depth map of the given pointcloud.
                      Invalid depth has 0 values
    """
    pts_2d = project_points(points, intrinsics).round()
    depths = points[:, 2]
    depth_map = points.new_zeros(image_hw)
    mask = points_inside_image(pts_2d, depths, image_hw)
    pts_2d = pts_2d[mask].long()
    depth_map[pts_2d[:, 1], pts_2d[:, 0]] = depths[mask]
    return depth_map
