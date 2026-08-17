"""Data config."""

from __future__ import annotations

from mapdet3d.config.config_dict import FieldConfigDict, class_config
from mapdet3d.config.typing import DataConfig
from mapdet3d.data.data_pipe import DataPipe, TupleDataPipe
from mapdet3d.data.datasets.ca1m import CA1M
from mapdet3d.data.datasets.scannetv2 import ScanNetv2
from mapdet3d.data.transforms.base import compose
from mapdet3d.data.transforms.to_tensor import ToTensor
from mapdet3d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)


def get_ca1m_data_cfg(
    data_root: str = "data/ca1m",
    cached_dir: str = "cache",
    window_size: int = 5,
    data_backend: None | FieldConfigDict = None,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get CA1M data config."""
    data = DataConfig()

    # Train
    train_dataset = class_config(
        CA1M,
        data_root=data_root,
        split="train",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_dir=cached_dir,
    )

    train_data_pipe_cfg = class_config(TupleDataPipe, datasets=[train_dataset])

    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_data_pipe_cfg,
        samples_per_gpu=samples_per_gpu,
        dynamic_sampler=True,
        workers_per_gpu=workers_per_gpu,
        image_num_range=(1, window_size),
    )

    # Test
    test_dataset_cfg = class_config(
        CA1M,
        data_root=data_root,
        split="val",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_dir=cached_dir,
    )

    test_datasets_cfg = class_config(DataPipe, datasets=test_dataset_cfg)

    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets_cfg,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
    )

    return data


def get_scannet_data_cfg(
    data_root: str = "data/scannet",
    cached_dir: str = "cache",
    scannet200: bool = False,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get ScanNet data config."""
    data = DataConfig()

    # Test
    test_dataset_cfg = class_config(
        ScanNetv2,
        data_root=data_root,
        split="val_200" if scannet200 else "val",
        cache_as_binary=True,
        cached_dir=cached_dir,
    )

    test_datasets_cfg = class_config(DataPipe, datasets=test_dataset_cfg)

    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets_cfg,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
    )

    return data
