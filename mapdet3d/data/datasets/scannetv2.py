"""ScanNetv2."""

from __future__ import annotations

import os

from mapdet3d.common.typing import ArgsType, DictStrAny
from mapdet3d.data.const import CommonKeys as K

from .ca1m import CA1M


class ScanNetv2(CA1M):
    """ScanNetv2 dataset."""

    def __init__(
        self,
        data_root: str = "data/scannet",
        split: str = "val",
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(data_root=data_root, split=split, **kwargs)

        if split in {"val", "val_200"}:
            with open(
                os.path.join(data_root, "meta_data", "val.txt"), "r"
            ) as f:
                self.seqs = [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Invalid split: {split}.")

        self.coco_ids = {seq_name: i for i, seq_name in enumerate(self.seqs)}
        self.coco_offset = 10000

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"ScanNetv2 {self.split}"

    def _get_sample_data(
        self,
        sample: DictStrAny,
        sample_data: list[DictStrAny],
    ) -> DictStrAny:
        """Get single sample from raw data."""
        data_dict = super()._get_sample_data(sample, sample_data)

        seq_name = sample["sequence_name"]

        # Mesh path for visualization (optional)
        data_dict["mesh_path"] = os.path.join(
            self.data_root,
            "data",
            seq_name,
            f"{seq_name}_vh_clean_2.ply",
        )

        data_dict[K.sample_names] = f"{seq_name}_{sample['frame_id']:06d}"

        data_dict[K.timestamp] = (
            self.coco_ids[seq_name] * self.coco_offset + sample["frame_id"]
        )
        data_dict["image_ids"] = data_dict[K.timestamp]

        return data_dict
