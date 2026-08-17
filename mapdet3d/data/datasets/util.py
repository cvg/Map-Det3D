"""Utility functions for datasets."""

from __future__ import annotations

import copy
import itertools
import os
import pickle
from collections.abc import Callable, Sequence
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import plyfile
from PIL import Image, ImageOps
from tabulate import tabulate
from termcolor import colored
from torch.utils.data import Dataset

from mapdet3d.common.distributed import broadcast, rank_zero_only
from mapdet3d.common.imports import OPENCV_AVAILABLE
from mapdet3d.common.logging import rank_zero_info
from mapdet3d.common.time import Timer
from mapdet3d.common.typing import (
    DictStrAny,
    ListAny,
    NDArrayFloat,
    NDArrayI64,
    NDArrayUI8,
)

from ..typing import DictData

if OPENCV_AVAILABLE:
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        COLOR_BGR2RGB,
        IMREAD_COLOR,
        IMREAD_GRAYSCALE,
        cvtColor,
        imdecode,
    )
else:
    raise ImportError("cv2 is not installed.")


def im_decode(
    im_bytes: bytes, mode: str = "RGB", backend: str = "PIL"
) -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    assert mode in {
        "BGR",
        "RGB",
        "L",
    }, f"{mode} not supported for image decoding!"
    if backend == "PIL":
        pil_img_file = Image.open(BytesIO(bytearray(im_bytes)))
        pil_img = ImageOps.exif_transpose(pil_img_file)
        assert pil_img is not None, "Image could not be loaded!"
        if pil_img.mode == "L":  # pragma: no cover
            if mode == "L":
                img: NDArrayUI8 = np.array(pil_img)[..., None]
            else:
                # convert grayscale image to RGB
                pil_img = pil_img.convert("RGB")
        elif mode == "L":  # pragma: no cover
            raise ValueError("Cannot convert colorful image to grayscale!")
        if mode == "BGR":  # pragma: no cover
            img = np.array(pil_img)[..., [2, 1, 0]]
        elif mode == "RGB":
            img = np.array(pil_img)
    elif backend == "cv2":  # pragma: no cover
        if not OPENCV_AVAILABLE:
            raise ImportError(
                "Please install opencv-python to use cv2 backend!"
            )
        img_np: NDArrayUI8 = np.frombuffer(im_bytes, np.uint8)
        img = imdecode(  # type: ignore
            img_np, IMREAD_GRAYSCALE if mode == "L" else IMREAD_COLOR
        )
        if mode == "RGB":
            cvtColor(img, COLOR_BGR2RGB, img)
    else:
        raise NotImplementedError(f"Image backend {backend} not known!")
    return img


def ply_decode(ply_bytes: bytes, mode: str = "XYZI") -> NDArrayFloat:
    """Decode to point clouds (numpy array) from bytes.

    Args:
        ply_bytes (bytes): The bytes of the ply file.
        mode (str, optional): The point format of the ply file. If "XYZI", the
            intensity channel will be included, otherwise only the XYZ
            coordinates. Defaults to "XYZI".
    """
    assert mode in {
        "XYZ",
        "XYZI",
    }, f"{mode} not supported for points decoding!"

    plydata = plyfile.PlyData.read(BytesIO(bytearray(ply_bytes)))
    num_points = plydata["vertex"].count
    num_channels = 3 if mode == "XYZ" else 4
    points = np.zeros((num_points, num_channels), dtype=np.float32)

    points[:, 0] = plydata["vertex"].data["x"]
    points[:, 1] = plydata["vertex"].data["y"]
    points[:, 2] = plydata["vertex"].data["z"]
    if mode == "XYZI":
        points[:, 3] = plydata["vertex"].data["intensity"]
    return points


def npy_decode(npy_bytes: bytes, key: str | None = None) -> NDArrayFloat:
    """Decode to numpy array from npy/npz file bytes."""
    data = np.load(BytesIO(bytearray(npy_bytes)))
    if key is not None:
        data = data[key]
    return data


def filter_by_keys(
    data_dict: DictData, keys_to_keep: Sequence[str]
) -> DictData:
    """Filter a dictionary by keys.

    Args:
        data_dict (DictData): The dictionary to filter.
        keys_to_keep (list[str]): The keys to keep.

    Returns:
        DictData: The filtered dictionary.
    """
    return {key: data_dict[key] for key in keys_to_keep if key in data_dict}


def get_used_data_groups(
    data_groups: dict[str, list[str]], keys: list[str]
) -> list[str]:
    """Get the data groups that are used by the given keys.

    Args:
        data_groups (dict[str, list[str]]): The data groups.
        keys (list[str]): The keys to check.

    Returns:
        list[str]: The used data groups.
    """
    used_groups = []
    for group_name, group_keys in data_groups.items():
        if not group_keys:
            continue
        if any(key in keys for key in group_keys):
            used_groups.append(group_name)
    return used_groups


def to_onehot(categories: NDArrayI64, num_classes: int) -> NDArrayFloat:
    """Transform integer categorical labels to onehot vectors.

    Args:
        categories (NDArrayI64): Integer categorical labels of shape (N, ).
        num_classes (int): Number of classes.

    Returns:
        NDArrayFloat: Onehot vector of shape (N, num_classes).
    """
    _eye = np.eye(num_classes, dtype=np.float32)
    return _eye[categories]


class CacheMappingMixin:
    """Caches a mapping for fast I/O and multi-processing.

    This class provides functionality for caching a mapping from dataset index
    requested by a call on __getitem__ to a dictionary that holds relevant
    information for loading the sample in question from the disk.
    Caching the mapping reduces startup time by loading the mapping instead of
    re-computing it at every startup.

    NOTE: Make sure your annotations file is up-to-date. Otherwise, the mapping
    will be wrong and you will get wrong samples.
    """

    @rank_zero_only
    def _load_mapping_data(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        cache_as_binary: bool,
        cached_file_path: str | None,
    ) -> ListAny:
        """Load possibly cached mapping via generate_map_func.

        Args:
            generate_map_func (Callable[[], list[DictStrAny]]): The function
                that generates the mapping.
            cache_as_binary (bool): Whether to cache the mapping as binary.
            cached_file_path (str | None): The path to the cached mapping file.
        """
        if cache_as_binary:
            assert (
                cached_file_path is not None
            ), "cached_file_path must be set if cache_as_binary is True!"
            if not os.path.exists(cached_file_path):
                rank_zero_info(
                    f"Did not find {cached_file_path}, generating it..."
                )
                data = generate_map_func()
                os.makedirs(os.path.dirname(cached_file_path), exist_ok=True)
                with open(cached_file_path, "wb") as file:
                    file.write(pickle.dumps(data))
            else:
                dt = datetime.fromtimestamp(os.stat(cached_file_path).st_mtime)
                rank_zero_info(
                    f"Found {cached_file_path} generated at {dt.isoformat()} "
                    + "and loading it..."
                )
                with open(cached_file_path, "rb") as file:
                    data = pickle.loads(file.read())
        else:
            rank_zero_info(f"Generating {self} data mapping...")
            data = generate_map_func()
        return data

    def _load_mapping(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        filter_func: Callable[[ListAny], ListAny] = lambda x: x,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
    ) -> tuple[DatasetFromList, int]:
        """Load cached mapping or generate if not exists.

        Args:
            generate_map_func (Callable[[], list[DictStrAny]]): The function
                that generates the mapping.
            filter_func (Callable[[ListAny], ListAny], optional): The function
                that filters the mapping. Defaults to lambda x: x.
            cache_as_binary (bool, optional): Whether to cache the mapping as
                binary. Defaults to True.
            cached_file_path (str | None, optional): The path to the cached
                mapping file. Defaults to None.
        """
        timer = Timer()
        dataset = self._load_mapping_data(
            generate_map_func, cache_as_binary, cached_file_path
        )
        original_len = 0
        if dataset is not None:
            original_len = len(dataset)
            dataset = filter_func(dataset)
            dataset = DatasetFromList(dataset)
        dataset = broadcast(dataset)
        original_len = broadcast(original_len)
        rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")
        return dataset, original_len


# reference:
# https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/data/common.py#L109
class DatasetFromList(Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(
        self, lst: ListAny, deepcopy: bool = False, serialize: bool = True
    ):
        """Creates an instance of the class.

        Args:
            lst: a list which contains elements to produce.
            deepcopy: whether to deepcopy the element when producing it, s.t.
            the result can be modified in place without affecting the source
            in the list.
            serialize: whether to hold memory using serialized objects. When
            enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
        """
        self._copy = deepcopy
        self._serialize = serialize

        def _serialize(data: Any) -> NDArrayUI8:  # type: ignore
            """Serialize python object to numpy array."""
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            self._lst = [_serialize(x) for x in lst]
            self._addr: NDArrayI64 = np.asarray(
                [len(x) for x in self._lst], dtype=np.int64
            )
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)  # type: ignore
        else:
            self._lst = lst  # pragma: no cover

    def __len__(self) -> int:
        """Return len of list."""
        if self._serialize:
            return len(self._addr)
        return len(self._lst)  # pragma: no cover

    def __getitem__(self, idx: int) -> Any:  # type: ignore
        """Return item of list at idx."""
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes_ = memoryview(self._lst[start_addr:end_addr])  # type: ignore
            return pickle.loads(bytes_)
        if self._copy:  # pragma: no cover
            return copy.deepcopy(self._lst[idx])

        return self._lst[idx]  # pragma: no cover


def print_class_histogram(class_frequencies: dict[str, int]) -> None:
    """Prints out given class frequencies."""
    if len(class_frequencies) == 0:  # pragma: no cover
        return

    class_names = list(class_frequencies.keys())
    frequencies = list(class_frequencies.values())
    num_classes = len(class_names)

    n_cols = min(6, len(class_names) * 2)

    def short_name(name: str) -> str:
        """Make long class names shorter."""
        if len(name) > 13:
            return name[:11] + ".."  # pragma: no cover
        return name

    data = list(
        itertools.chain(
            *[
                [short_name(class_names[i]), int(v)]
                for i, v in enumerate(frequencies)
            ]
        )
    )
    total_num_instances = sum(data[1::2])  # type: ignore
    data.extend([None] * (n_cols - (len(data) % n_cols)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])

    table = tabulate(
        itertools.zip_longest(*[data[i::n_cols] for i in range(n_cols)]),
        headers=["category", "#instances"] * (n_cols // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )

    rank_zero_info(
        f"Distribution of instances among all {num_classes} categories:\n"
        + colored(table, "cyan")
    )


def get_category_names(det_mapping: dict[str, int]) -> list[str]:
    """Get category names from a mapping of category names to ids."""
    return sorted(det_mapping, key=det_mapping.get)  # type: ignore
