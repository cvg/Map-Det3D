"""This module contains dictionary utility functions."""

from __future__ import annotations

from typing import Any, NamedTuple


def get_all_keys(entry: NamedTuple) -> list[str]:
    """Get all keys in a NamedTuple."""
    keys = []
    for key in entry._fields:
        if is_namedtuple(getattr(entry, key)):
            keys.extend(
                [f"{key}.{k}" for k in get_all_keys(getattr(entry, key))]
            )
        else:
            keys.append(key)
    return keys


def get_from_namedtuple(entry: NamedTuple, key: str) -> Any:  # type: ignore
    """Get a value from a nested Named tuple.

    Example passing key = "test.my.data" will resolve the value of the
    named tuple at 'test' 'my' 'data'.

    Raises:
        ValueError: If the key is not present in the named tuple.
    """
    keys = key.split(".")
    first_key = keys[0]
    if not hasattr(entry, first_key):
        raise ValueError(
            f"Key {first_key} not in named tuple! Current keys: "
            f"{get_all_keys(entry)}"
        )
    if len(keys) == 1:
        return getattr(entry, first_key)

    return get_from_namedtuple(getattr(entry, first_key), ".".join(keys[1:]))


def is_namedtuple(obj: object) -> bool:
    """Check if obj is namedtuple.

    https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    """
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and hasattr(obj, "_fields")
    )
