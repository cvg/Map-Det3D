"""CA1M dataset."""

from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from mapdet3d.common.typing import ArgsType, DictStrAny
from mapdet3d.data.const import AxisMode
from mapdet3d.data.const import CommonKeys as K
from mapdet3d.op.mapanything.image import preprocess_inputs

from .base import VideoDataset, VideoMapping
from .util import CacheMappingMixin, im_decode


def select_training_frame_ids(
    key_fid: int,
    max_fid: int,
    num_views: int,
    max_sampling_rate: int = 1,
) -> list[int]:
    """Select future training frame ids with a random or fixed stride."""
    if num_views < 1:
        raise ValueError("num_views must be greater than 0.")

    if max_sampling_rate < 1:
        raise ValueError("max_sampling_rate must be greater than 0.")

    sampling_rate = int(np.random.randint(1, max_sampling_rate + 1))

    selected_fids = list(range(key_fid, max_fid + 1, sampling_rate))[
        :num_views
    ]
    if len(selected_fids) < num_views:
        selected_fids = [key_fid] * (
            num_views - len(selected_fids)
        ) + selected_fids

    return selected_fids


class CA1M(CacheMappingMixin, VideoDataset):
    """CA1M dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_depth: float = 10.0,
        depth_scale: float = 1000.0,
        remove_empty: bool = False,
        valid_scenes: list[str] | None = None,
        cache_as_binary: bool = False,
        cached_dir: str = "cache",
        use_arkit_depth: bool = True,
        max_sampling_rate: int = 5,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(**kwargs)
        self.data_root = data_root
        self.split = split
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.use_arkit_depth = use_arkit_depth
        self.max_sampling_rate = max_sampling_rate

        self.remove_empty = remove_empty
        self.valid_scenes = valid_scenes

        self.cache_as_binary = cache_as_binary
        self.cached_dir = cached_dir
        self.cached_file_path = os.path.join(
            self.data_root, self.cached_dir, f"{self.split}.pkl"
        )

        # Load annotations
        self.samples, _ = self._load_mapping(
            self._generate_data_mapping,
            self._filter_data,
            cache_as_binary=cache_as_binary,
            cached_file_path=self.cached_file_path,
        )

        # Generate video mapping
        self.video_mapping = self._generate_video_mapping()

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"CA1M {self.split}"

    def _filter_data(self, data: list[DictStrAny]) -> list[DictStrAny]:
        """Remove empty samples."""
        if not self.remove_empty:
            return data

        samples = []
        for sample in data:
            if sample["sequence_name"] in self.valid_scenes:
                samples.append(sample)

        return samples

    def _generate_video_mapping(self) -> VideoMapping:
        """Group dataset sample indices by their associated video ID.

        The sample index is an integer while video IDs are string.

        Returns:
            VideoMapping: Mapping of video IDs to sample indices and frame IDs.
        """
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        video_to_frame_ids: dict[str, list[int]] = defaultdict(list)

        for i, sample in enumerate(self.samples):
            seq = sample["sequence_name"]
            fid = sample["frame_id"]
            video_to_indices[seq].append(i)
            video_to_frame_ids[seq].append(fid)

        return self._sort_video_mapping(
            {
                "video_to_indices": video_to_indices,
                "video_to_frame_ids": video_to_frame_ids,
            }
        )

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generates the data mapping."""
        with open(self.cached_file_path, "rb") as file:
            data = pickle.loads(file.read())
        return data

    def __len__(self):
        return len(self.samples)

    def _get_sample_data(
        self, sample: DictStrAny, sample_data: list[DictStrAny]
    ) -> DictStrAny:
        """Get single sample from raw data."""
        data_dict = {}

        data_dict[K.sample_names] = sample["timestamp"]
        data_dict["image_ids"] = int(sample["timestamp"])
        data_dict[K.timestamp] = int(sample["timestamp"]) / 1e9
        data_dict[K.sequence_names] = sample["sequence_name"]
        data_dict[K.frame_ids] = sample["frame_id"]

        # Load image
        im_bytes = self.data_backend.get(sample_data["image_file_path"])

        image = np.ascontiguousarray(
            im_decode(im_bytes, mode=self.image_channel_mode),
            dtype=np.float32,
        )[None]

        intrinsics = sample_data["intrinsics"]

        data_dict[K.images] = image
        data_dict[K.input_hw] = (image.shape[1], image.shape[2])

        data_dict[K.original_images] = image
        data_dict[K.original_hw] = (image.shape[1], image.shape[2])

        data_dict[K.intrinsics] = intrinsics
        data_dict["original_intrinsics"] = intrinsics

        data_dict[K.extrinsics] = sample_data["extrinsics"]
        data_dict["T_gravity"] = sample_data["T_gravity"]

        # Load annotations
        data_dict[K.boxes2d] = sample_data["boxes2d"]
        data_dict[K.boxes2d_names] = sample_data["categories"]
        data_dict[K.boxes2d_classes] = np.zeros(
            len(sample_data["categories"]), dtype=np.int64
        )
        data_dict[K.boxes2d_track_ids] = sample_data["track_ids"]
        data_dict[K.boxes3d] = sample_data["boxes3d"]

        data_dict["boxes3d_cam_from_world"] = sample_data[
            "boxes3d_cam_from_world"
        ]
        data_dict[K.boxes3d_classes] = np.zeros(
            len(sample_data["categories"]), dtype=np.int64
        )
        data_dict[K.boxes3d_names] = sample_data["categories"]
        data_dict[K.boxes3d_track_ids] = sample_data["track_ids"]
        data_dict[K.axis_mode] = AxisMode.OPENCV

        # Load depth
        depth_bytes = self.data_backend.get(
            sample_data["arkit_depth_file_path"]
            if self.use_arkit_depth
            else sample_data["depth_file_path"]
        )

        depth_array = im_decode(depth_bytes)

        depth = np.ascontiguousarray(depth_array, dtype=np.float32)

        depth = depth / self.depth_scale

        depth[depth > self.max_depth] = 0

        depth = F.interpolate(
            torch.from_numpy(depth)[None, None, ...],
            size=(image.shape[1], image.shape[2]),
            mode="nearest",
        )[0, 0].numpy()

        data_dict[K.depth_maps] = depth

        # Load global annotations
        if "boxes3d_world" in sample:
            data_dict["boxes3d_world"] = sample["boxes3d_world"]

        # Mesh path for visualization (optional)
        data_dict["mesh_path"] = os.path.join(
            self.data_root,
            "mesh",
            sample["sequence_name"],
            "mesh.ply",
        )

        return data_dict

    def __getitem__(self, idx: int | tuple[int, int, float]) -> DictStrAny:
        """Get single sample."""

        # For training
        if isinstance(idx, tuple):
            key_sample = self.samples[idx[0]]
            num_views = idx[1]
            aspect_ratio = idx[2]

            seq_name = key_sample["sequence_name"]
            frame_ids = self.video_mapping["video_to_frame_ids"][seq_name]
            video_indices = self.video_mapping["video_to_indices"][seq_name]

            with open(
                os.path.join(
                    self.data_root,
                    self.cached_dir,
                    self.split,
                    f"{seq_name}.pkl",
                ),
                "rb",
            ) as file:
                data = pickle.loads(file.read())

            key_fid = key_sample["frame_id"]
            max_fid = frame_ids[-1]

            selected_fids = select_training_frame_ids(
                key_fid, max_fid, num_views, self.max_sampling_rate
            )

            # Get data for all selected frames
            seq = []
            for fid in selected_fids:
                frame_data = self._get_sample_data(
                    self.samples[video_indices[fid]], data["seq_data"][fid]
                )
                seq.append(frame_data)

            self.data_backend.close()

            views = [
                {
                    "img": s["pil_image"],
                    "intrinsics": s[K.intrinsics],
                    "camera_poses": s[K.extrinsics],
                    "depth_z": s[K.depth_maps],
                    "is_metric_scale": torch.tensor([True]),
                    "boxes2d": s[K.boxes2d],
                }
                for s in seq
            ]

            processed_views = preprocess_inputs(
                views,
                padding_mode=True,
                aspect_ratio=aspect_ratio,
            )

            pad_info_list = [v.pop("pad_info", None) for v in processed_views]

            input_hw_list = [
                [v["img"].shape[-2], v["img"].shape[-1]]
                for v in processed_views
            ]
            intrinsics_list = [
                v["intrinsics"][0].numpy().copy() for v in processed_views
            ]

            boxes2d_list = [
                v.pop("boxes2d")[0].numpy() for v in processed_views
            ]
            track_ids_list = [s[K.boxes3d_track_ids] for s in seq]
            class_ids_list = [s[K.boxes3d_classes] for s in seq]
            boxes3d_list = [s[K.boxes3d] for s in seq]

            sample_names_list = [s[K.sample_names] for s in seq]
            categories_list = [s[K.boxes3d_names] for s in seq]

            return {
                "processed_views": processed_views,
                "sample_names": sample_names_list,
                "input_hw": input_hw_list,
                "intrinsics": intrinsics_list,
                "boxes2d": boxes2d_list,
                "boxes3d": boxes3d_list,
                "track_ids": track_ids_list,
                "class_ids": class_ids_list,
                "categories": categories_list,
                "pad_info": pad_info_list,
            }

        sample = self.samples[idx]
        seq_name = sample["sequence_name"]

        with open(
            os.path.join(
                self.data_root, self.cached_dir, self.split, f"{seq_name}.pkl"
            ),
            "rb",
        ) as file:
            data = pickle.loads(file.read())

        self.data_backend.close()

        return self._get_sample_data(
            sample, data["seq_data"][sample["frame_id"]]
        )
