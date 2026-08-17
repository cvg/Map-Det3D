"""This module contains array utility functions."""

from __future__ import annotations

from typing import overload

import numpy as np
import torch

from mapdet3d.common.typing import (
    ArrayLike,
    NDArrayBool,
    NDArrayF32,
    NDArrayF64,
    NDArrayI32,
    NDArrayI64,
    NDArrayNumber,
    NDArrayUI8,
    NDArrayUI16,
    NDArrayUI32,
)


# Bool dtypes
@overload
def array_to_numpy(
    data: ArrayLike, n_dims: int | None, dtype: type[np.bool_]
) -> NDArrayBool: ...


# Float dtypes
@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.float32]
) -> NDArrayF32: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.float64]
) -> NDArrayF64: ...


# Int dtypes
@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.int32]
) -> NDArrayI32: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.int64]
) -> NDArrayI64: ...


# UInt dtypes
@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.uint8]
) -> NDArrayUI8: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.uint16]
) -> NDArrayUI16: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[np.uint32]
) -> NDArrayUI32: ...


# Union of all dtypes
@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None
) -> NDArrayNumber: ...


@overload
def array_to_numpy(data: None) -> None: ...


def array_to_numpy(
    data: ArrayLike | None,
    n_dims: int | None = None,
    dtype: (
        type[np.bool_]
        | type[np.float32]
        | type[np.float64]
        | type[np.int32]
        | type[np.int64]
        | type[np.uint8]
        | type[np.uint16]
        | type[np.uint32]
    ) = np.float32,
) -> NDArrayNumber | None:
    """Converts a given array like object to a numpy array.

    Helper function to convert an array like object to a numpy array.
    This functions converts torch.Tensors or Sequences to numpy arrays.

    If the argument is None, None will be returned.

    Examples:
        >>> convert_to_array([1,2,3])
        >>> # -> array([1,2,3])
        >>> convert_to_array(None)
        >>> # -> None
        >>> convert_to_array(torch.tensor([1,2,3]).cuda())
        >>> # -> array([1,2,3])
        >>> convert_to_array([1,2,3], n_dims = 2).shape
        >>> # -> [1, 3]

    Args:
        data (ArrayLike | None): ArrayLike object that should be converted
            to numpy.

        n_dims (int | None, optional): Target number of dimension of the array.
            If the provided array does not have this shape, it will be
            squeezed or exanded (from the left). If it still does not match,
            an error is raised.

        dtype (SUPPORTED_DTYPES, optional): Target dtype of the array. Defaults
            to np.float32.

    Raises:
        ValueError: If the provied array like objects can not be converted
            with the target dimensions.

    Returns:
        NDArrayNumber | None: The converted numpy array or None if None was
            provided.
    """
    if data is None:
        return data

    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, torch.Tensor):
        array = np.asarray(data.detach().cpu().numpy())
    else:
        array = np.asarray(data)

    if n_dims is not None:
        # Squeeze if needed
        for _ in range(len(array.shape) - n_dims):
            if array.shape[0] == 1:
                array = array.squeeze(0)
            elif array.shape[-1] == 1:
                array = array.squeeze(-1)

        # expand if needed
        for _ in range(n_dims - len(array.shape)):
            array = np.expand_dims(array, 0)

        if len(array.shape) != n_dims:
            raise ValueError(
                f"Failed to convert target array of shape {array.shape} to"
                f"have {n_dims} dimensions."
            )

    return array.astype(dtype)  # type: ignore
