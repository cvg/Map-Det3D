"""This module contains convenience functions for checkpoint loading.

The code is based on https://github.com/open-mmlab/mmcv/
"""

from __future__ import annotations

import os.path as osp
import re
from collections import OrderedDict
from typing import Callable, Union

import torch
import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url as load_url

from mapdet3d.common.distributed import (
    get_rank,
    get_world_size,
    is_module_wrapper,
    synchronize,
)
from mapdet3d.common.logging import rank_zero_info, rank_zero_warn
from mapdet3d.common.typing import TorchCheckpoint

CheckpointLoadFunc = Callable[
    [str, Union[str, torch.device, None]], TorchCheckpoint
]

# Define mapping for specific model checkpoints
BDD100K_MODEL_PREFIX = "https://dl.cv.ethz.ch/bdd100k/"
MM_MODEL_MAP = {
    "mmdet://": "https://download.openmmlab.com/mmdetection/v2.0/",
    "mmseg://": "https://download.openmmlab.com/mmsegmentation/v0.5/",
}
MM_CFG_MAP = {
    "mmdet://": "syscv/mmdetection/master/configs/",
    "mmseg://": "open-mmlab/mmsegmentation/master/configs/",
}
MM_ZIP_MAP = {
    "mmdet://": "mmdetection-master/configs/",
    "mmseg://": "mmsegmentation-master/configs/",
}


def load_model_checkpoint(
    model: nn.Module,
    weights: str,
    strict: bool = False,
    rev_keys: None | list[tuple[str, str]] = None,
    map_location: str | torch.device | None = "cpu",
) -> None:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        weights (str): Accept local filepath, URL, or e.g.``torchvision://xxx``
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        rev_keys (tuple[tuple[str, str]]): A tuple of customized keywords to
            modify the state_dict in checkpoint. Each item is a
            (pattern, replacement) pair of the regular expression operations.
            Default: strip the prefix 'module.' by [(r'^module.', '')].
        map_location (str | torch.device | None): Same as :func:`torch.load`.
            Default: 'cpu'.
    """
    if rev_keys is None:  # pragma: no cover
        rev_keys = [(r"^module\.", "")]
    if re.compile(r"^mm(det|seg)://").search(weights):
        pre = weights[:8]
        weights = MM_MODEL_MAP[pre] + weights.split(pre)[-1]
        _load_checkpoint(
            model, weights, map_location, strict=strict, revise_keys=rev_keys
        )
    elif weights.startswith("bdd100k://"):
        weights = BDD100K_MODEL_PREFIX + weights.split("bdd100k://")[-1]
        _load_checkpoint(
            model, weights, map_location, strict=strict, revise_keys=rev_keys
        )
    else:  # pragma: no cover
        _load_checkpoint(
            model, weights, map_location, strict=strict, revise_keys=rev_keys
        )


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: dict[str, CheckpointLoadFunc] = {}

    @classmethod
    def _register_scheme(
        cls,
        prefixes: str | tuple[str, ...],
        loader: CheckpointLoadFunc,
        force: bool = False,
    ) -> None:
        """Register a scheme."""
        if isinstance(prefixes, str):
            prefixes = (prefixes,)

        assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f"{prefix} is already registered as a loader backend, "
                    'add "force=True" if you want to override it'
                )
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True)
        )

    @classmethod
    def register_scheme(
        cls,
        prefixes: str | tuple[str, ...],
        force: bool = False,
    ) -> Callable[[CheckpointLoadFunc], CheckpointLoadFunc]:
        """Register a loader to CheckpointLoader.

        This method should be used as a decorator.

        Args:
            prefixes (str or Sequence[str]): The register prefix of the loader.
            force (bool, optional): Whether to override the loader if the
                prefix has already been registered. Defaults to False.
        """

        def _register(
            loader_cls: CheckpointLoadFunc,
        ) -> CheckpointLoadFunc:
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path: str) -> CheckpointLoadFunc:
        """Finds a loader that supports the given path.

        Falls back to the local loader if no other loader is found, since it is
        registered with an empty prefix.

        Args:
            path (str): checkpoint path.

        Raises:
            ValueError: If the path cannot be matched to any prefix, raise an
                error. This should usually not happen, since the local loader
                is registered with an empty prefix.

        Returns:
            CheckpointLoadFunc: checkpoint loader.
        """
        for prefix, func in cls._schemes.items():
            if re.match(prefix, path) is not None:
                return func
        raise ValueError("Invalid path! No prefix matched.")

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        map_location: str | torch.device | None = None,
    ) -> TorchCheckpoint:
        """Load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        rank_zero_info(
            f"Load checkpoint from {class_name[10:]} path: {filename}"
        )
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes="")
def load_from_local(
    filename: str,
    map_location: str | torch.device | None = None,
) -> TorchCheckpoint:
    """Load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Raises:
        FileNotFoundError: If file not found.

    Returns:
        TorchCheckpoint: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    checkpoint = torch.load(
        filename, weights_only=True, map_location=map_location
    )
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=("http://", "https://"))
def load_from_http(
    filename: str, map_location: str | torch.device | None = None
) -> TorchCheckpoint:
    """Load checkpoint through HTTP or HTTPS scheme path.

    In distributed setting, this function only download checkpoint at local
    rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        TorchCheckpoint: The loaded checkpoint.
    """
    rank, world_size = get_rank(), get_world_size()
    if rank == 0:
        checkpoint = load_url(filename, map_location=map_location)
    if world_size > 1:
        synchronize()
        if rank > 0:
            checkpoint = load_url(filename, map_location=map_location)
    return checkpoint  # pylint: disable=used-before-assignment


def get_torchvision_models() -> dict[str, str]:
    """Get full URLs of all torchvision paths.

    Requires torchvision >= 0.14.0a0.
    """
    model_urls: dict[str, str] = {}
    weights_list = [
        torchvision.models.get_model_weights(model)
        for model in torchvision.models.list_models(torchvision.models)
    ]
    for model_cls in weights_list:
        # The name of torchvision model weights classes ends with
        # `_Weights` such as `ResNet18_Weights`. However, some model weight
        # classes, such as `MNASNet0_75_Weights` does not have any urls in
        # torchvision 0.13.0 and cannot be iterated. Here we simply check
        # `DEFAULT` attribute to ensure the class is not empty.
        if not hasattr(model_cls, "DEFAULT"):
            continue
        # Since `cls.DEFAULT` can not be accessed by iterating cls, we set
        # default urls explicitly.
        cls_name = model_cls.__name__
        cls_key = cls_name.replace("_Weights", "").lower()
        model_urls[f"{cls_key}.default"] = model_cls.DEFAULT.url
        for weight_enum in model_cls:
            cls_key = cls_name.replace("_Weights", "").lower()
            cls_key = f"{cls_key}.{weight_enum.name.lower()}"
            model_urls[cls_key] = weight_enum.url

    return model_urls


@CheckpointLoader.register_scheme(prefixes="torchvision://")
def load_from_torchvision(
    filename: str, map_location: str | torch.device | None = None
) -> TorchCheckpoint:
    """Load checkpoint through the file path prefixed with torchvision.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.'

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    model_name = filename[14:]

    # Support getting model urls in the same way as torchvision
    # `ResNet50_Weights.IMAGENET1K_V1` will be mapped to
    # resnet50.imagenet1k_v1.
    model_name = model_name.lower().replace("_weights", "")
    return load_from_http(model_urls[model_name], map_location)


def load_state_dict(
    module: nn.Module, state_dict: TorchCheckpoint, strict: bool = False
) -> None:
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Raises:
        RuntimeError: If strict, it will raise a runtime error if module and
            state_dict do not match completely.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (dict or OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    """
    unexpected_keys: list[str] = []
    all_missing_keys: list[str] = []
    err_msg: list[str] = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        # pylint: disable=protected-access
        state_dict._metadata = metadata  # type: ignore

    # use _load_from_state_dict to enable checkpoint version control
    def load(module: nn.Module, prefix: str = "") -> None:
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module  # type: ignore
        local_metadata = (
            {} if metadata is None else metadata.get(prefix[:-1], {})
        )
        module._load_from_state_dict(  # pylint: disable=protected-access
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        # pylint: disable=protected-access
        for name, child in module._modules.items():
            if child is not None:
                # pylint: disable=not-callable
                load(child, prefix + name + ".")

    load(module)
    # break load->load reference cycle
    load = None  # type: ignore

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if "num_batches_tracked" not in key
    ]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source "
            f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    rank = get_rank()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, "The model and loaded state dict do not match exactly\n"
        )
        err_msg = "\n".join(err_msg)  # type: ignore
        if strict:
            raise RuntimeError(err_msg)
        rank_zero_warn(err_msg)


def _load_checkpoint(
    model: torch.nn.Module,
    filename: str,
    map_location: str | torch.device | None = None,
    strict: bool = False,
    revise_keys: tuple[tuple[str, str]] | list[tuple[str, str]] = (
        (r"^module\.", ""),
    ),
) -> TorchCheckpoint:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        revise_keys (tuple[tuple[str, str]]): A tuple of customized keywords to
            modify the state_dict in checkpoint. Each item is a
            (pattern, replacement) pair of the regular expression operations.
            Default: strip the prefix 'module.' by [(r'^module.', '')].

    Raises:
        RuntimeError: If no state_dict is found in the checkpoint file.

    Returns:
        TorchCheckpoint: The loaded checkpoint.
    """
    checkpoint = CheckpointLoader.load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"No state_dict found in checkpoint file {filename}"
        )
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v for k, v in state_dict.items()}
        )
    # Keep metadata in state_dict
    state_dict._metadata = metadata  # pylint: disable=protected-access

    # load state_dict
    load_state_dict(model, state_dict, strict)
    return checkpoint
