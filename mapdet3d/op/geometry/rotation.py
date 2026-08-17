"""Rotation utilities."""

import functools
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from mapdet3d.data.const import AxisMode

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def normalize_angle(input_angles: Tensor) -> Tensor:
    """Normalize content of input_angles to range [-pi, pi].

    Args:
        input_angles: (Tensor) tensor of any shape containing
                       unnormalized angles.

    Returns:
        Tensor with angles normalized to +/- pi
    """
    return torch.sub((input_angles + torch.pi) % (2 * torch.pi), torch.pi)


def acute_angle(theta_1: Tensor, theta_2: Tensor) -> Tensor:
    """Update theta_1 to mkae the agnle between two thetas is acute."""
    # Make sure the angle between two thetas is acute
    if torch.pi / 2.0 < abs(theta_2 - theta_1) < torch.pi * 3 / 2.0:
        theta_1 += torch.pi
        if theta_1 > torch.pi:
            theta_1 -= torch.pi * 2
        if theta_1 < -torch.pi:
            theta_1 += torch.pi * 2

    # Convert the case of > 270 to < 90
    if abs(theta_2 - theta_1) >= torch.pi * 3 / 2.0:
        if theta_2 > 0:
            theta_1 += torch.pi * 2
        else:
            theta_1 -= torch.pi * 2
    return theta_1


def yaw2alpha(rot_y: Tensor, center: Tensor) -> Tensor:
    """Get alpha by vertical rotation - theta.

    Args:
        rot_y: Rotation around Y-axis in camera coordinates [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        alpha: Observation angle of object, ranging [-pi..pi]
    """
    alpha = rot_y - torch.atan2(center[..., 0], center[..., 2])
    return normalize_angle(alpha)


def alpha2yaw(alpha: Tensor, center: Tensor) -> Tensor:
    """Get vertical rotation by alpha + theta.

    Args:
        alpha: Observation angle of object, ranging [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        rot_y: Vertical rotation in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(center[..., 0], center[..., 2])
    return normalize_angle(rot_y)


def rotation_output_to_alpha(output: Tensor, num_bins: int = 2) -> Tensor:
    """Get alpha from bin-based regression output.

    Uses method described in (with two bins):
    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
    Mousavian et al., CVPR'17

    Args:
        output: (Tensor) bin based regressed output.
        num_bins: (int) number of bins to use

    Returns:
        Tensor containing the angle from the bin-based regression output
    """
    out_range = torch.tensor(list(range(len(output))), device=output.device)
    bin_idx = output[:, :num_bins].argmax(dim=-1)
    res_idx = num_bins + 2 * bin_idx
    bin_centers = torch.arange(
        -torch.pi, torch.pi, 2 * torch.pi / num_bins, device=output.device
    )
    bin_centers += torch.pi / num_bins
    alpha = (
        torch.atan(output[out_range, res_idx] / output[out_range, res_idx + 1])
        + bin_centers[bin_idx]
    )
    return alpha


def generate_rotation_output(pred: Tensor, num_bins: int = 2) -> Tensor:
    """Convert output to bin confidence and cos / sin of residual.

    The viewpoint (alpha) prediction (N, num_bins + 2 * num_bins) consists of:
    bin confidences (N, num_bins): softmax logits for bin probability.
    1st entry is probability for orientation being in bin 1,
    2nd entry is probability for orientation being in bin 2,
    and so on.
    bin residual (N, num_bins * 2): angle residual w.r.t. bin N orientation,
    represented as sin and cos values.

    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
    Mousavian et al., CVPR'17
    """
    pred = pred.view(pred.size(0), -1, 3 * num_bins)
    bin_logits = pred[..., :num_bins]

    bin_residuals = []
    for i in range(num_bins):
        res_idx = num_bins + 2 * i
        norm = pred[..., res_idx : res_idx + 2].norm(dim=-1, keepdim=True)
        bsin = pred[..., res_idx : res_idx + 1] / norm
        bcos = pred[..., res_idx + 1 : res_idx + 2] / norm
        bin_residuals.append(bsin)
        bin_residuals.append(bcos)

    rot = torch.cat([bin_logits, *bin_residuals], -1)
    return rot


# Rotation conversion functions adapted from:
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """Get rotation matrix for an angle around an axis.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    assert axis in {"X", "Y", "Z"}, f"Invalid axis {axis}."
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rot_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        rot_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    else:
        rot_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(rot_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(
    euler_angles: Tensor, convention: str = "XYZ"
) -> Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
        "X", "Y", and "Z".

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Raises:
        ValueError: if convention string is not a combination of XYZ
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, a)
        for c, a in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return functools.reduce(torch.matmul, matrices)


def _index_from_letter(letter: str) -> int:  # pragma: no cover
    """Return index from letter.

    Args:
        letter: (str) letter in [X,Y,Z]

    Returns:
        int mapping of the corresponding letter [0,1,2]

    Raises:
        ValueError: if the given letter is not valid
    """
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter not valid!")


def _angle_from_tan(
    axis: str,
    other_axis: str,
    data: Tensor,
    horizontal: bool,
    tait_bryan: bool,
) -> Tensor:
    """Helper function for matrix_to_euler_angles.

    Extracts the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = axis + other_axis in {"XY", "YZ", "ZX"}
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: Tensor, convention: str = "XYZ") -> Tensor:
    """Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).

    Raises:
        ValueError: if convention string is not a combination of XYZ
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        rads = matrix[..., i0, i2]
        # safety for nan
        rads[torch.where(rads > 1.0)] = rads.new_tensor([1.0]).to(rads.device)
        rads[torch.where(rads < -1.0)] = rads.new_tensor([-1.0]).to(
            rads.device
        )
        central_angle = torch.asin(
            rads * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """Returns sqrt(max(0, x)) with a zero subgradient at x <= 0.

    Implemented with a double-where to avoid:
    - a CUDA sync from boolean-masked assignment, and
    - NaN gradients from sqrt'(0) = inf leaking through torch.where.
    """
    positive = x > 0
    # Replace non-positive entries with 1 so sqrt's backward is finite (0.5)
    # on the dead branch; the outer where zeroes those gradients anyway.
    safe = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(x))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Raises:
        ValueError: If shape of input matrix is not correct.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))

    # Pick the best-conditioned candidate (largest denominator) per batch
    # element. Uses gather to keep a static output shape — boolean-mask
    # indexing here would force a CUDA sync to discover the mask cardinality.
    best_idx = q_abs.argmax(dim=-1)
    return quat_candidates.gather(
        -2, best_idx[..., None, None].expand(*best_idx.shape, 1, 4)
    ).squeeze(-2)


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """Convert a unit quaternion to a standard form.

    Standard form: One in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(quat1: Tensor, quat2: Tensor) -> Tensor:
    """Multiply two quaternions.

    Usual torch rules for broadcasting apply.

    Args:
        quat1: Quaternions as tensor of shape (..., 4), real part first.
        quat2: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of quat1 and quat2, tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(quat1, -1)
    bw, bx, by, bz = torch.unbind(quat2, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(quat1: Tensor, quat2: Tensor) -> Tensor:
    """Multiply two quaternions representing rotations.

    Returns the quaternion representing their composition, i.e. the version
    with nonnegative real part. Usual torch rules for broadcasting apply.

    Args:
        quat1: Quaternions as tensor of shape (..., 4), real part first.
        quat2: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of quat1 and quat2, tensor of quaternions shape (..., 4).
    """
    return standardize_quaternion(quaternion_raw_multiply(quat1, quat2))


def quaternion_invert(quaternion: Tensor) -> Tensor:
    """Return quaternion that represents inverse rotation.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion: Tensor, points: Tensor) -> Tensor:
    """Apply the rotation given by a quaternion to a 3D point.

    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        points: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).

    Raises:
        ValueError: If points is not a valid 3D point set.
    """
    if points.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {points.shape}.")
    real_parts = points.new_zeros(points.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, points), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def rotation_matrix_yaw(
    rotation_matrix: Tensor, axis_mode: AxisMode
) -> Tensor:
    """Get yaw of 3D boxes in euler angle under given axis mode.

    Args:
        rotation_matrix (Tensor): [N, 3, 3] Rotation matrix of the object.
        axis_mode (AxisMode): Coordinate system convention.

    Returns:
        orientation (Tensor): [N, 3] Yaw in euler angle.
    """
    orientation = rotation_matrix.new_zeros(rotation_matrix.shape[0], 3)

    if axis_mode == AxisMode.OPENCV:
        orientation[:, 1] = matrix_to_euler_angles(rotation_matrix, "YZX")[
            :, 0
        ]
    else:
        orientation[:, 2] = matrix_to_euler_angles(rotation_matrix, "ZYX")[
            :, 0
        ]
    return orientation


def rotate_orientation(
    orientation: Tensor, extrinsics: Tensor, axis_mode: AxisMode = AxisMode.ROS
) -> Tensor:
    """Rotate the orientation of the object in different coordinate.

    Args:
        orientation (Tensor): [N, 3] Orientation of the object in euler angles.
        extrinsics (Tensor): [4, 4] Extrinsic matrix of the object.
        axis_mode (AxisMode): Coordinate system convention. Default:
            AxisMode.ROS
    """
    rot = extrinsics[:3, :3] @ euler_angles_to_matrix(orientation)
    return rotation_matrix_yaw(rot, axis_mode)


def rotate_velocities(velocities: Tensor, extrinsics: Tensor) -> Tensor:
    """Rotate the velocities of the object in different coordinate."""
    return (extrinsics[:3, :3] @ velocities.unsqueeze(-1)).squeeze(-1)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _acos_linear_approximation(x: Tensor, x0: float) -> Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def acos_linear_extrapolation(
    x: Tensor,
    bounds: tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError(
            "lower bound has to be smaller or equal to upper bound."
        )

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError(
            "Both lower bound and upper bound have to be within (-1, 1)."
        )

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def so3_rotation_angle(
    R: Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    _, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError(
            "A matrix has trace outside valid range [-1-eps,3+eps]."
        )

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def so3_relative_angle(
    R1: Tensor,
    R2: Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1), out_dtype=R1.dtype)
    return so3_rotation_angle(
        R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps
    )


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Converts 6D rotation representation to rotation matrix.

    It uses Gram--Schmidt orthogonalization per Section B of Zhou et al. [1].

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Converts rotation matrices to 6D rotation representation by

    It drops the last row as Zhou et al. [1]. Note that 6D representation is
    not unique.

    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def R_from_allocentric(K: Tensor, R_view, u=None, v=None):
    """Convert a rotation matrix or series of rotation matrices to egocentric
    representation given a 2D location (u, v) in pixels.

    When u or v are not available, we fall back on the principal point of K.
    """
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    sx = K[:, 0, 2]
    sy = K[:, 1, 2]

    if u is None:
        u = sx

    if v is None:
        v = sy

    oray = torch.stack(((u - sx) / fx, (v - sy) / fy, torch.ones_like(u))).T
    oray = oray / torch.linalg.norm(oray, dim=1).unsqueeze(1)
    angle = torch.acos(oray[:, -1])

    axis = torch.zeros_like(oray)
    axis[:, 0] = axis[:, 0] - oray[:, 1]
    axis[:, 1] = axis[:, 1] + oray[:, 0]
    norms = torch.linalg.norm(axis, dim=1)

    valid_angle = angle > 0

    M = axis_angle_to_matrix(angle.unsqueeze(1) * axis / norms.unsqueeze(1))

    R = R_view.clone()
    R[valid_angle] = torch.bmm(M[valid_angle], R_view[valid_angle]).to(
        dtype=R.dtype
    )

    return R


def R_to_allocentric(K: Tensor, R, u=None, v=None):
    """Convert a rotation matrix or series of rotation matrices to allocentric
    representation given a 2D location (u, v) in pixels.

    When u or v are not available, we fall back on the principal point of K.
    """
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    sx = K[:, 0, 2]
    sy = K[:, 1, 2]

    if u is None:
        u = sx

    if v is None:
        v = sy

    oray = torch.stack(((u - sx) / fx, (v - sy) / fy, torch.ones_like(u))).T
    oray = oray / torch.linalg.norm(oray, dim=1).unsqueeze(1)
    angle = torch.acos(oray[:, -1])

    axis = torch.zeros_like(oray)
    axis[:, 0] = axis[:, 0] - oray[:, 1]
    axis[:, 1] = axis[:, 1] + oray[:, 0]
    norms = torch.linalg.norm(axis, dim=1)

    valid_angle = angle > 0

    M = axis_angle_to_matrix(angle.unsqueeze(1) * axis / norms.unsqueeze(1))

    R_view = R.clone()
    R_view[valid_angle] = torch.bmm(
        M[valid_angle].transpose(2, 1), R[valid_angle]
    )

    return R_view
