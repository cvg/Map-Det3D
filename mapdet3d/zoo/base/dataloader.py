"""Dataloader configuration."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from mapdet3d.config import class_config
from mapdet3d.data.data_pipe import DataPipe
from mapdet3d.data.loader import (
    build_inference_dataloaders,
    build_train_dataloader,
)
from mapdet3d.data.transforms.to_tensor import ToTensor


def get_train_dataloader_cfg(
    datasets_cfg: ConfigDict | list[ConfigDict],
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    batchprocess_cfg: ConfigDict | None = None,
    pin_memory: bool | FieldReference = True,
    dynamic_sampler: bool = False,
    shuffle: bool | FieldReference = True,
    aspect_ratio_grouping: bool | FieldReference = False,
    image_num_range: tuple[int, int] = (1, 1),
) -> ConfigDict:
    """Creates dataloader configuration given dataset and preprocessing."""
    if batchprocess_cfg is None:
        batchprocess_cfg = class_config(ToTensor)

    if isinstance(datasets_cfg, list):
        dataset = class_config(DataPipe, datasets=datasets_cfg)
    else:
        dataset = datasets_cfg

    return class_config(
        build_train_dataloader,
        dataset=dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_fn=batchprocess_cfg,
        pin_memory=pin_memory,
        dynamic_sampler=dynamic_sampler,
        shuffle=shuffle,
        aspect_ratio_grouping=aspect_ratio_grouping,
        image_num_range=image_num_range,
    )


def get_inference_dataloaders_cfg(
    datasets_cfg: ConfigDict | list[ConfigDict],
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    video_based_inference: bool | FieldReference = False,
    batchprocess_cfg: ConfigDict | None = None,
) -> ConfigDict:
    """Creates dataloader configuration given dataset for inference."""
    if batchprocess_cfg is None:
        batchprocess_cfg = class_config(ToTensor)

    return class_config(
        build_inference_dataloaders,
        datasets=datasets_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=video_based_inference,
        batchprocess_fn=batchprocess_cfg,
    )
