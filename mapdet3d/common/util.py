"""Utility functions for common usage."""

import random
from difflib import get_close_matches

import numpy as np
import torch
from packaging import version

from .imports import is_torch_tf32_available
from .logging import rank_zero_info, rank_zero_warn


def create_did_you_mean_msg(keys: list[str], query: str) -> str:
    """Create a did you mean message.

    Args:
        keys (list[str]): List of available keys.
        query (str): Query.

    Returns:
        str: Did you mean message.

    Examples:
        >>> keys = ["foo", "bar", "baz"]
        >>> query = "fo"
        >>> print(create_did_you_mean_msg(keys, query))
        Did you mean:
            foo
    """
    msg = ""
    if len(keys) > 0:
        msg = "Did you mean:\n\t"
        msg += "\n\t".join(get_close_matches(query, keys, cutoff=0.75))
    return msg


def set_tf32(use_tf32: bool, precision: str) -> None:  # pragma: no cover
    """Set torch TF32.

    Args:
        use_tf32: Whether to use torch TF32. Details:
            https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        precision: Internal precision of float32 matrix multiplications.
             Details: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision # pylint: disable=line-too-long
    """
    if use_tf32:  # pragma: no cover
        rank_zero_info(
            "Using Torch TF32. "
            + "It might harm the performance due to the precision. "
            + "You can turn it off by setting config.use_tf32=False."
        )
        if not is_torch_tf32_available():
            rank_zero_warn("Torch TF32 is not available.")
        elif (
            version.parse("1.11")
            >= version.parse(torch.__version__)
            >= version.parse("1.7")
        ):
            rank_zero_info("Torch TF32 is turned on by default!")
        else:
            rank_zero_info("Turn on Torch TF32 on matmul.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Control the precision of matmul operations.
    # Equivalent to setting torch.backends.cuda.matmul.allow_tf32.
    torch.set_float32_matmul_precision(precision)


def init_random_seed() -> int:
    """Initialize random seed for the experiment."""
    return int(np.random.randint(2**31))


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
