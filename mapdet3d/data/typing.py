"""Type definitions related to the data pipeline.

This file defines the data format `DictData` as an arbitrary dictionary that
can, in principle, hold arbitrary data. However, we provide `CommonKeys` in
`mapdet3d.data.const` to define the input format for commonly used input types,
so that the data pre-processing pipeline can take advantage of pre-defined
data formats that are necessary to properly pre-process a given data sample.
"""

from __future__ import annotations

from typing import Union

from mapdet3d.common.typing import DictStrAny

DictData = DictStrAny
DictDataOrList = Union[DictData, list[DictData]]
