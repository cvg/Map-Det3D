"""Video Dataset Wrapper."""

from __future__ import annotations

from torch.utils.data import Dataset

from .const import CommonKeys as K
from .datasets.base import VideoDataset
from .typing import DictData


class SequentialVideoDataset(Dataset[list[DictData]]):
    """VideoDataset wrapper to have sequential inputs."""

    def __init__(self, dataset: VideoDataset, num_frames: int = 2):
        """Init."""
        self.dataset = dataset

        # Video settings
        self.video_mapping = dataset.video_mapping
        self.num_frames = num_frames
        self.has_reference = True

    def __len__(self) -> int:
        """Get length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list[DictData]:
        """Get item."""
        cur_sample = self.dataset[idx]

        indices_in_video = self.video_mapping["video_to_indices"][
            cur_sample[K.sequence_names]
        ]
        frame_ids = self.video_mapping["video_to_frame_ids"][
            cur_sample[K.sequence_names]
        ]

        cur_frame_id = frame_ids[indices_in_video.index(idx)]

        samples = []
        for i in range(self.num_frames - 1, 0, -1):
            past_frame_id = cur_frame_id - i

            if past_frame_id >= 0:
                samples.append(self.dataset[indices_in_video[past_frame_id]])
            # else:
            #     samples.append(self.dataset[idx])

        # Append current sample as the last frame
        samples.append(cur_sample)

        return samples
