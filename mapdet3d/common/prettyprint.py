"""This module contains utilities for pretty printing."""

from typing import Any

import numpy as np
import torch


class PrettyRepMixin:
    """Creates a pretty string representation of a class with parameters.

    Examples:
        >>> class TestClass(PrettyRepMixin):
        ...     def __init__(self, a: int, b: str):
        ...         self.a = a
        ...         self.b = b
        >>> obj = TestClass(1, 'hello')
        >>> str(obj)
        'TestClass(a=1, b=hello)'
    """

    def __repr__(self) -> str:
        """Return a string representation of the class and its parameters.

        Returns:
            The string representation of the class and its parameters.

        Examples:
            >>> class TestClass(PrettyRepMixin):
            ...     def __init__(self, a: int, b: str):
            ...         self.a = a
            ...         self.b = b
            >>> obj = TestClass(1, 'hello')
            >>> obj.__repr__()
            'TestClass(a=1, b=hello)'
        """
        attr_str = ""
        for k, v in vars(self).items():
            if k != "type" and not k.startswith("_"):
                attr_str += f"{k}={str(v)}, "
        attr_str = attr_str.rstrip(", ")
        return f"{self.__class__.__name__}({attr_str})"


def describe_shape(obj: Any) -> str:  # type: ignore
    """Recursively output the shape of tensors in an object's structure.

    Args:
        obj (Any): The object to describe the shape of. Can be a dictionary,
        list, torch.Tensor, numpy.ndarray, float, or any other type.

    Returns:
        str: A string representing the shapes of all tensors in the object's
            structure.

    Examples:
        >>> describe_shape({'a': torch.rand(2, 3)})
        "{a: shape[2, 3]}"
        >>> describe_shape({'a': [torch.rand(2, 3), torch.rand(4, 5)]})
        "{a: [shape[2, 3], shape[4, 5]]}"
        >>> describe_shape([torch.rand(2, 3), {'a': torch.rand(4, 5)}])
        "[shape[2, 3], {a: shape[4, 5]}]"
    """
    log_str = ""
    if isinstance(obj, dict):
        log_str += "{"
        log_str += ", ".join(
            [f"{k}: {describe_shape(obj[k])}" for k in obj.keys()]
        )
        log_str += "}"
    elif isinstance(obj, list):
        log_str += "["
        log_str += ", ".join([describe_shape(v) for v in obj])
        log_str += "]"
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        log_str += f"shape[{', '.join([str(s) for s in obj.shape])}]"
    elif isinstance(obj, float):
        log_str += f"{obj:.4f}"
    else:
        log_str += str(obj)
    return log_str
