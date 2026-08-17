"""Utility functions for evaluation."""

import numpy as np

from mapdet3d.common.array import array_to_numpy
from mapdet3d.common.typing import ArrayLike, NDArrayNumber


def dense_inputs_to_numpy(
    prediction: ArrayLike, target: ArrayLike
) -> tuple[NDArrayNumber, NDArrayNumber]:
    """Convert dense prediction and target to numpy arrays."""
    prediction = array_to_numpy(prediction, n_dims=None, dtype=np.float32)
    target = array_to_numpy(target, n_dims=None, dtype=np.float32)
    return prediction, target


def check_shape_match(
    prediction: NDArrayNumber, target: NDArrayNumber
) -> None:
    """Check if the shape of prediction and target matches."""
    assert prediction.shape == target.shape, (
        f"Shape mismatch between prediction {prediction.shape} and target"
        f"{target.shape}."
    )
