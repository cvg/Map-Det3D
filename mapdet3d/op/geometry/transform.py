"""Geometric transformation functions."""

import torch
from torch import Tensor


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Applies transform to points.

    Args:
        points (Tensor): points of shape (N, D) or (B, N, D).
        transform (Tensor): transforms of shape (D+1, D+1) or (B, D+1, D+1).

    Returns:
        Tensor: (N, D) / (B, N, D) transformed points.

    Raises:
        ValueError: Either points or transform have incorrect shape
    """
    hom_coords = torch.cat([points, torch.ones_like(points[..., 0:1])], -1)
    if len(points.shape) == 2:
        if len(transform.shape) == 3:
            assert (
                transform.shape[0] == 1
            ), "Got multiple transforms for single point set!"
            transform = transform.squeeze(0)
        transform = transform.T
    elif len(points.shape) == 3:
        if len(transform.shape) == 2:
            transform = transform.T.unsqueeze(0)
        elif len(transform.shape) == 3:
            transform = transform.permute(0, 2, 1)
        else:
            raise ValueError(f"Shape of transform invalid: {transform.shape}")
    else:
        raise ValueError(f"Shape of input points invalid: {points.shape}")
    points_transformed = hom_coords @ transform
    return points_transformed[..., : points.shape[-1]]


def inverse_pinhole(intrinsic_matrix: Tensor) -> Tensor:
    """Calculate inverse of pinhole projection matrix.

    Args:
        intrinsic_matrix (Tensor): [..., 3, 3] intrinsics or single [3, 3]
            intrinsics.

    Returns:
        Tensor:  Inverse of input intrinisics.
    """
    squeeze = False
    inv = intrinsic_matrix.clone()
    if len(intrinsic_matrix.shape) == 2:
        inv = inv.unsqueeze(0)
        squeeze = True

    inv[..., 0, 0] = 1.0 / inv[..., 0, 0]
    inv[..., 1, 1] = 1.0 / inv[..., 1, 1]
    inv[..., 0, 2] = -inv[..., 0, 2] * inv[..., 0, 0]
    inv[..., 1, 2] = -inv[..., 1, 2] * inv[..., 1, 1]

    if squeeze:
        inv = inv.squeeze(0)
    return inv


def inverse_rigid_transform(transformation: Tensor) -> Tensor:
    """Calculate inverse of rigid body transformation(s).

    Args:
        transformation (Tensor): [N, 4, 4] transformations or single [4, 4]
            transformation.

    Returns:
        Tensor: Inverse of input transformation(s).
    """
    squeeze = False
    if len(transformation.shape) == 2:
        transformation = transformation.unsqueeze(0)
        squeeze = True
    rotation, translation = transformation[:, :3, :3], transformation[:, :3, 3]
    rot = rotation.permute(0, 2, 1)
    t = -rot @ translation[:, :, None]
    inv = torch.cat([torch.cat([rot, t], -1), transformation[:, 3:4]], 1)
    if squeeze:
        inv = inv.squeeze(0)
    return inv


def get_transform_matrix(rotation: Tensor, translation: Tensor) -> Tensor:
    """Assembles 4x4 transformation from rotation / translation pair(s).

    Args:
        rotation (Tensor): [N, 3, 3] or [3, 3] rotation(s).
        translation (Tensor): [N, 3] or [3,] translation(s).

    Returns:
        Tensor: [N, 4, 4] or [4, 4] transformation.
    """
    squeeze = False
    if len(rotation.shape) == 2:
        assert len(translation.shape) == 1
        rotation = rotation.unsqueeze(0)
        translation = translation.unsqueeze(0)
        squeeze = True
        batch_size = 1
    else:
        assert len(rotation.shape) == 3 and len(translation.shape) == 2
        assert rotation.shape[0] == translation.shape[0]
        batch_size = rotation.shape[0]
    assert (
        rotation.shape[-2] == rotation.shape[-1] == translation.shape[-1] == 3
    )
    transforms = rotation.new_zeros((batch_size, 4, 4))
    transforms[:, :3, :3] = rotation
    transforms[:, :3, 3] = translation
    transforms[:, 3, 3] = 1.0
    if squeeze:
        transforms = transforms.squeeze(0)
    return transforms
