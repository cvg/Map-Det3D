"""Reference View Sampling.

These Classes sample reference views from a dataset that contains videos.
This is usually used when a model needs multiple samples of a video during
training.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, List

import numpy as np
from torch.utils.data import Dataset

from .const import CommonKeys as K
from .datasets.base import VideoDataset
from .typing import DictData

SortingFunc = Callable[[DictData, list[DictData]], List[DictData]]


def sort_key_first(
    cur_sample: DictData, ref_data: list[DictData]
) -> list[DictData]:
    """Sort views as key first."""
    return [cur_sample, *ref_data]


def sort_temporal(
    cur_sample: DictData, ref_data: list[DictData]
) -> list[DictData]:
    """Sort views temporally."""
    return sorted([cur_sample, *ref_data], key=lambda x: x[K.frame_ids])


class ReferenceViewSampler:
    """Base reference view sampler."""

    def __init__(self, num_ref_samples: int) -> None:
        """Creates an instance of the class.

        Args:
            num_ref_samples (int): Number of reference views to sample.
        """
        self.num_ref_samples = num_ref_samples

    @abstractmethod
    def __call__(
        self,
        key_dataset_index: int,
        indices_in_video: list[int],
        frame_ids: list[int],
    ) -> list[int]:
        """Sample num_ref_samples reference view indices.

        Args:
            key_index (int): Index of key view in the video.
            indices_in_video (list[int]): All dataset indices in the video.
            frame_ids (list[int]): Frame ids of all views in the video.

        Returns:
            list[int]: dataset indices of reference views.
        """
        raise NotImplementedError


class SequentialViewSampler(ReferenceViewSampler):
    """Sequential View Sampler."""

    def __call__(
        self,
        key_dataset_index: int,
        indices_in_video: list[int],
        frame_ids: list[int],
    ) -> list[int]:
        """Sample sequential reference views."""
        # assert len(frame_ids) >= self.num_ref_samples + 1

        key_index = indices_in_video.index(key_dataset_index)

        right = key_index + 1 + self.num_ref_samples
        if right <= len(indices_in_video):
            ref_dataset_indices = indices_in_video[key_index + 1 : right]
        else:
            # left = key_index - (right - len(indices_in_video))
            valid_len = len(indices_in_video[key_index + 1 :])
            ref_dataset_indices = [key_index] * (
                self.num_ref_samples - valid_len
            ) + indices_in_video[key_index + 1 :]
        return ref_dataset_indices


class UniformViewSampler(ReferenceViewSampler):
    """View Sampler that chooses reference views uniform at random."""

    def __init__(self, scope: int, num_ref_samples: int) -> None:
        """Creates an instance of the class.

        Args:
            scope (int): Define scope of neighborhood to key view to sample
                from.
            num_ref_samples (int): Number of reference views to sample.
        """
        super().__init__(num_ref_samples)
        if scope != 0 and scope < num_ref_samples // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        self.scope = scope

    def _get_valid_indices(
        self, key_index: int, indices_in_video: list[int], frame_ids: list[int]
    ) -> list[int]:
        """Get valid indices in video."""
        key_fid = frame_ids[key_index]
        min_fid = max(0, key_fid - self.scope)
        max_fid = min(key_fid + self.scope, frame_ids[-1])

        return [
            ind
            for i, ind in enumerate(indices_in_video)
            if min_fid <= frame_ids[i] <= max_fid and i != key_index
        ]

    def __call__(
        self,
        key_dataset_index: int,
        indices_in_video: list[int],
        frame_ids: list[int],
    ) -> list[int]:
        """Uniformly sample reference views."""
        if self.scope > 0:
            key_index = indices_in_video.index(key_dataset_index)

            valid_indices = self._get_valid_indices(
                key_index, indices_in_video, frame_ids
            )

            if len(valid_indices) > 0:
                assert len(valid_indices) >= self.num_ref_samples
                return np.random.choice(
                    valid_indices, self.num_ref_samples, replace=False
                ).tolist()

        return [key_dataset_index] * self.num_ref_samples


class MultiViewDataset(Dataset[list[DictData]]):
    """Dataset that samples reference views from a video dataset."""

    def __init__(
        self,
        dataset: VideoDataset,
        sampler: ReferenceViewSampler,
        sort_fn: SortingFunc = sort_temporal,
        num_retry: int = 3,
        match_key: str = K.boxes2d_track_ids,
        skip_nomatch_samples: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            dataset (Dataset): Video dataset to sample from.
            sampler (ReferenceViewSampler): Sampler that samples reference
                views.
            sort_fn (SortingFunc, optional): Function that sorts key and
                reference views. Defaults to sort_key_first.
            num_retry (int, optional): Number of retries if no match is found.
                Defaults to 3.
            match_key (str, optional): Key to match reference views with key
                view. Defaults to K.boxes2d_track_ids.
            skip_nomatch_samples (bool, optional): Whether to skip samples
                where no match is found. Defaults to False.
        """
        self.dataset = dataset
        self.sampler = sampler
        self.sort_fn = sort_fn
        self.num_retry = num_retry
        self.match_key = match_key
        self.skip_nomatch_samples = skip_nomatch_samples

    def has_matches(
        self, key_data: DictData, ref_data: list[DictData]
    ) -> bool:
        """Check if key / ref data have matches."""
        key_target = key_data[self.match_key]
        for ref_view in ref_data:
            ref_target = ref_view[self.match_key]
            match = np.equal(
                np.expand_dims(key_target, axis=1), ref_target[None]
            )
            if match.any():
                return True
        return False  # pragma: no cover

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.dataset)

    def get_ref_data(self, ref_indices: list[int]) -> list[DictData]:
        """Get reference data from dataset."""
        ref_data = []
        for ref_index in ref_indices:
            ref_sample = self.dataset[ref_index]
            ref_sample["keyframes"] = False
            ref_data.append(ref_sample)

        assert self.sampler.num_ref_samples == len(ref_data)
        return ref_data

    def __getitem__(self, index: int) -> list[DictData]:
        """Get item from dataset."""
        cur_sample = self.dataset[index]
        cur_sample["keyframes"] = True

        indices_in_video = self.dataset.video_mapping["video_to_indices"][
            cur_sample[K.sequence_names]
        ]
        frame_ids = self.dataset.video_mapping["video_to_frame_ids"][
            cur_sample[K.sequence_names]
        ]

        if self.sampler.num_ref_samples > 0:
            for _ in range(self.num_retry):
                ref_indices = self.sampler(index, indices_in_video, frame_ids)

                ref_data = self.get_ref_data(ref_indices)

                if self.skip_nomatch_samples and not (
                    self.has_matches(cur_sample, ref_data)
                ):
                    continue

                return self.sort_fn(cur_sample, ref_data)

            ref_indices = [index] * self.sampler.num_ref_samples
            ref_data = self.get_ref_data(ref_indices)
            return [cur_sample, *ref_data]

        return [cur_sample]
