"""Base dataset classes.

We implement a typed version of the PyTorch dataset class here. In addition, we
provide a number of Mixin classes which a dataset can inherit from to implement
additional functionality.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from torch.utils.data import Dataset as TorchDataset

from mapdet3d.common.typing import ArgsType
from mapdet3d.data.io.base import DataBackend
from mapdet3d.data.io.file import FileBackend
from mapdet3d.data.typing import DictData


class Dataset(TorchDataset[DictData]):
    """Basic pytorch dataset with defined return type."""

    # Dataset metadata.
    DESCRIPTION = ""
    HOMEPAGE = ""
    PAPER = ""
    LICENSE = ""

    # List of all keys supported by this dataset.
    KEYS: Sequence[str] = []

    def __init__(
        self,
        image_channel_mode: str = "RGB",
        data_backend: None | DataBackend = None,
    ) -> None:
        """Initialize dataset.

        Args:
            image_channel_mode (str): Image channel mode to use. Default: RGB.
            data_backend (None | DataBackend): Data backend to use.
                Default: None.
        """
        self.image_channel_mode = image_channel_mode
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )

    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DictData:
        """Get a data sample by index."""
        raise NotImplementedError

    def validate_keys(self, keys_to_load: Sequence[str]) -> None:
        """Validate that all keys to load are supported.

        Args:
            keys_to_load (list[str]): List of keys to load.

        Raises:
            ValueError: Raise if any key is not defined in AVAILABLE_KEYS.
        """
        for k in keys_to_load:
            if k not in self.KEYS:
                raise ValueError(f"Key '{k}' is not supported!")


class VideoMapping(TypedDict):
    """Grouped dataset sample indices and frame indices."""

    video_to_indices: dict[str, list[int]]
    video_to_frame_ids: dict[str, list[int]]


class VideoDataset(Dataset):
    """Video datasets.

    Provides video_mapping attribute for video based interface and reference
    view samplers.
    """

    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        """Initialize dataset."""
        super().__init__(*args, **kwargs)
        self.video_mapping: VideoMapping = {
            "video_to_indices": {},
            "video_to_frame_ids": {},
        }

    def _sort_video_mapping(self, video_mapping: VideoMapping) -> VideoMapping:
        """Sort video mapping by frame ids."""
        video_to_indices = video_mapping["video_to_indices"]
        video_to_frame_ids = video_mapping["video_to_frame_ids"]

        for seq in video_to_indices:
            sorted_zipped = sorted(
                list(zip(video_to_indices[seq], video_to_frame_ids[seq])),
                key=lambda x: x[1],
            )
            sorted_indices, sorted_frame_ids = zip(*sorted_zipped)
            video_mapping["video_to_indices"][seq] = list(sorted_indices)
            video_mapping["video_to_frame_ids"][seq] = list(sorted_frame_ids)

        return video_mapping

    def _generate_video_mapping(self) -> VideoMapping:
        """Group dataset sample by their associated video ID.

        The sample index is an integer while video IDs are string.

        Returns:
            VideoMapping: Mapping of video IDs to sample indices and frame IDs.
        """
        raise NotImplementedError
