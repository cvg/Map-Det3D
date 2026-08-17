# mypy: disable-error-code=misc
"""This module contains utilities for multiprocess parallelism."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import tempfile
from collections import OrderedDict
from functools import wraps
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import broadcast_object_list
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mapdet3d.common.typing import ArgsType, DictStrAny, GenericFunc


# no coverage for these functions, since we don't unittest distributed setting
def get_world_size() -> int:  # pragma: no cover
    """Get the world size (number of processes) of torch.distributed.

    Returns:
        int: The world size.
    """
    if os.environ.get("WORLD_SIZE", None):
        return int(os.environ["WORLD_SIZE"])

    # In interactive job not using slurm ntasks
    if os.environ.get("SLURM_JOB_NAME", None) != "bash":
        if os.environ.get("SLURM_NTASKS", None):
            return int(os.environ["SLURM_NTASKS"])

    return 1


def get_rank() -> int:  # pragma: no cover
    """Get the global rank of the current process in torch.distributed.

    Returns:
        int: The global rank.
    """
    # For torchrun
    if os.environ.get("RANK", None):
        return int(os.environ["RANK"])

    # Because pl don't set global rank, use local rank for interactive job and
    # slurm process id for submitted job
    if os.environ.get("SLURM_JOB_NAME", None) == "bash":
        return get_local_rank()
    if os.environ.get("SLURM_PROCID", None):
        return int(os.environ["SLURM_PROCID"])

    # Return local rank
    return get_local_rank()


def get_local_rank() -> int:  # pragma: no cover
    """Get the local rank of the current process in torch.distributed.

    Returns:
        int: The local rank.
    """
    if os.environ.get("LOCAL_RANK", None):
        return int(os.environ["LOCAL_RANK"])
    if os.environ.get("SLURM_LOCALID", None):
        return int(os.environ["SLURM_LOCALID"])

    return 0


def distributed_available() -> bool:  # pragma: no cover
    """Check if torch.distributed is available.

    Returns:
        bool: Whether torch.distributed is available.
    """
    return dist.is_available() and dist.is_initialized()


def synchronize() -> None:  # pragma: no cover
    """Sync (barrier) among all processes when using distributed training."""
    if not distributed_available():
        return
    if get_world_size() == 1:
        return
    dist.barrier(group=dist.group.WORLD, device_ids=[get_local_rank()])


def broadcast(obj: Any, src: int = 0) -> Any:  # type: ignore
    """Broadcast an object from a source to all processes."""
    if not distributed_available():
        return obj
    obj = [obj]
    rank = get_rank()
    if rank != src:
        obj = [None]
    broadcast_object_list(obj, src, group=dist.group.WORLD)
    return obj[0]


def serialize_to_tensor(data: Any) -> Tensor:  # type: ignore
    """Serialize arbitrary picklable data to a Tensor.

    Args:
        data (Any): The data to serialize.

    Returns:
        Tensor: The serialized data as a Tensor.

    Raises:
        AssertionError: If the backend of torch.distributed is not gloo or
            nccl.
    """
    backend = dist.get_backend()
    assert backend in {
        "gloo",
        "nccl",
    }, "_serialize_to_tensor only supports gloo and nccl backends."
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank %s tries all-gather %.2f GB of data on device %s",
            get_rank(),
            len(buffer) / (1024**3),
            device,
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def rank_zero_only(func: GenericFunc) -> GenericFunc:
    """Allows the decorated function to be called only on global rank 0.

    Args:
        func(GenericFunc): The function to decorate.

    Returns:
        GenericFunc: The decorated function.

    """

    @wraps(func)
    def wrapped_fn(*args: ArgsType, **kwargs: ArgsType) -> Any:  # type: ignore
        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapped_fn


def pad_to_largest_tensor(
    tensor: Tensor,
) -> tuple[list[int], Tensor]:  # pragma: no cover
    """Pad tensor to largest size among the tensors in each process.

    Args:
        tensor: tensor to be padded.

    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = get_world_size()
    assert (
        world_size >= 1
    ), "_pad_to_largest_tensor requires distributed setting!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    local_size_list = [local_size.clone() for _ in range(world_size)]
    dist.all_gather_object(local_size_list, local_size)
    size_list = [int(size.item()) for size in local_size_list]
    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_object_gpu(  # type: ignore
    data: Any, rank_zero_return_only: bool = True
) -> list[Any] | None:  # pragma: no cover
    """Run pl_module.all_gather on arbitrary picklable data.

    Args:
        data: any picklable object
        rank_zero_return_only: if results should only be returned on rank 0

    Returns:
        list[Any]: list of data gathered from each process
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # encode
    tensor = serialize_to_tensor(data)
    size_list, tensor = pad_to_largest_tensor(tensor)
    tensor_list = [tensor.clone() for _ in range(world_size)]
    dist.all_gather_object(tensor_list, tensor)  # (world_size, N)

    if rank_zero_return_only and rank != 0:
        return None

    # decode
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def create_tmpdir(
    rank: int, tmpdir: None | str = None, use_system_tmp: bool = True
) -> str:  # pragma: no cover
    """Create and distribute a temporary directory across all processes."""
    if tmpdir is not None:
        os.makedirs(tmpdir, exist_ok=True)
        return tmpdir
    if rank == 0:
        # create a temporary directory
        default_tmpdir = tempfile.gettempdir()
        if default_tmpdir is not None and use_system_tmp:
            dist_tmpdir = os.path.join(default_tmpdir, ".dist_tmp")
        else:
            dist_tmpdir = os.path.join("work_dir", ".dist_tmp")
        os.makedirs(dist_tmpdir, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=dist_tmpdir)
    else:
        tmpdir = None
    return broadcast(tmpdir)


def all_gather_object_cpu(  # type: ignore
    data: Any,
    tmpdir: None | str = None,
    rank_zero_return_only: bool = True,
    use_system_tmp: bool = False,
) -> list[Any] | None:  # pragma: no cover
    """Share arbitrary picklable data via file system caching.

    Args:
        data: any picklable object.
        tmpdir: Save path for temporary files. If None, safely create tmpdir.
        rank_zero_return_only: if results should only be returned on rank 0.
        use_system_tmp: if use system tmpdir or not.

    Returns:
        list[Any]: list of data gathered from each process.
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # make tmp dir
    tmpdir = create_tmpdir(rank, tmpdir, use_system_tmp)

    # encode & save
    with open(os.path.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(data, f)
    synchronize()

    if rank_zero_return_only and rank != 0:
        return None

    # load & decode
    data_list = []
    for i in range(world_size):
        with open(os.path.join(tmpdir, f"part_{i}.pkl"), "rb") as f:
            data_list.append(pickle.load(f))

    # remove dir
    if not rank_zero_return_only:
        # wait for all processes to finish loading before removing tmpdir
        synchronize()
    if rank == 0:
        shutil.rmtree(tmpdir)

    return data_list


def reduce_mean(tensor: Tensor) -> Tensor:
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def obj2tensor(  # type: ignore
    pyobj: Any, device: torch.device = torch.device("cuda")
) -> Tensor:
    """Serialize picklable python object to tensor.

    Args:
        pyobj (Any): Any picklable python object.
        device (torch.device): Device to put on. Defaults to "cuda".
    """
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2obj(tensor: Tensor) -> Any:  # type: ignore
    """Deserialize tensor to picklable python object.

    Args:
        tensor (Tensor): Tensor to be deserialized.
    """
    return pickle.loads(tensor.cpu().numpy().tobytes())


def all_reduce_dict(
    py_dict: DictStrAny, reduce_op: str = "sum", to_float: bool = True
) -> DictStrAny:  # pragma: no cover
    """Apply all reduce function for python dict object.

    The code is modified from
    https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/allreduce_norm.py.

    NOTE: make sure that py_dict in different ranks has the same keys and
    the values should be in the same shape. Currently only supports
    NCCL backend.

    Args:
        py_dict (DictStrAny): Dict to be applied all reduce op.
        reduce_op (str): Operator, could be 'sum' or 'mean'. Default: 'sum'.
        to_float (bool): Whether to convert all values of dict to float.
            Default: True.

    Returns:
        DictStrAny: reduced python dict object.
    """
    world_size = get_world_size()
    if world_size == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    if not isinstance(py_dict, OrderedDict):
        py_key_tensor = obj2tensor(py_key)
        dist.broadcast(py_key_tensor, src=0)
        py_key = tensor2obj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    if to_float:
        flatten_tensor = torch.cat(
            [py_dict[k].flatten().float() for k in py_key]
        )
    else:
        flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])

    dist.all_reduce(flatten_tensor, op=dist.ReduceOp.SUM)
    if reduce_op == "mean":
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape)
        for x, shape in zip(
            torch.split(flatten_tensor, tensor_numels), tensor_shapes
        )
    ]
    out_dict: DictStrAny = dict(zip(py_key, split_tensors))
    if isinstance(py_dict, OrderedDict):
        out_dict = OrderedDict(out_dict)
    return out_dict


def is_module_wrapper(module: nn.Module) -> bool:
    """Checks recursively if a module is wrapped.

    Two modules are regarded as wrapper: DataParallel, DistributedDataParallel.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return True
    if any(is_module_wrapper(child) for child in module.children()):
        return True
    return False
