"""This module contains dictionary utility functions."""

from __future__ import annotations

from typing import Any

from mapdet3d.common.typing import DictStrAny


def flatten_dict(dictionary: DictStrAny, seperator: str) -> list[str]:
    """Flatten a nested dictionary.

    Args:
        dictionary (DictStrAny): The dictionary to flatten.
        seperator (str): The seperator to use between keys.

    Returns:
        List[str]: A list of flattened keys.

    Examples:
        >>> d = {'a': {'b': {'c': 10}}}
        >>> flatten_dict(d, '.')
        ['a.b.c']
    """
    flattened = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flattened.extend(
                [
                    f"{key}{seperator}{subkey}"
                    for subkey in flatten_dict(value, seperator)
                ]
            )
        else:
            flattened.append(key)
    return flattened


def get_dict_nested(  # type: ignore
    dictionary: DictStrAny, keys: list[str], allow_missing: bool = False
) -> Any:
    """Get a value from a nested dictionary.

    Args:
        dictionary (DictStrAny): The dictionary to get the value from.
        keys (list[str]): A list of keys specifying the location in the nested
            dictionary where the value is located.
        allow_missing (bool, optional): Whether to allow missing keys. Defaults
            to False. If False, a ValueError is raised if a key is not present,
            otherwise None is returned.

    Returns:
        list[str]: The value from the dictionary.

    Raises:
        ValueError: If the key is not present in the dictionary.

    Examples:
        >>> d = {'a': {'b': {'c': 10}}}
        >>> get_dict_nested(d, ['a', 'b', 'c'])
        10

        >>> get_dict_nested(d, ['a', 'b', 'd'])
        ValueError: Key d not in dictionary! Current keys: dict_keys(['c'])
    """
    for key in keys:
        if key not in dictionary:
            if allow_missing:
                return None
            raise ValueError(
                f"Key {key} not in dictionary! Current keys: "
                f"{dictionary.keys()}"
            )
        dictionary = dictionary[key]
    return dictionary


def set_dict_nested(  # type: ignore
    dictionary: DictStrAny, keys: list[str], value: Any
) -> None:
    """Set a value in a nested dictionary.

    Args:
        dictionary (dict[str, Any]): The dictionary to set the value in.
        keys (list[str]): A list of keys specifying the location in the nested
            dictionary where the value should be set.
        value (Any): The value to set in the dictionary.

    Examples:
        >>> d = {}
        >>> set_dict_nested(d, ['a', 'b', 'c'], 10)
        >>> d
        {'a': {'b': {'c': 10}}}
    """
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value
