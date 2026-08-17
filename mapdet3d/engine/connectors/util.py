"""Utility functions for the connectors module."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import NamedTuple, TypedDict

from torch import Tensor
from typing_extensions import NotRequired

from mapdet3d.common.dict import get_dict_nested
from mapdet3d.common.named_tuple import get_from_namedtuple, is_namedtuple
from mapdet3d.common.typing import DictStrArrNested
from mapdet3d.data.typing import DictData


class SourceKeyDescription(TypedDict):
    """Defines a data entry by providing the key and source of the data.

    Attributes:
        key (str): Key that is used to index data from the specified source
        source (str): Which datasource to choose from.
            Options are ['data', 'prediction'] where data referes to the
            output of the dataloader and prediction refers to the model
            output
        sensors (Sequence[str]): Which sensors to use for the data.
    """

    key: str
    source: str
    sensors: NotRequired[Sequence[str]]


def remap_pred_keys(
    info: dict[str, SourceKeyDescription], parent_key: str
) -> dict[str, SourceKeyDescription]:
    """Remaps the key of a connection mapping to a new parent key.

    Args:
        info (SourceKeyDescription): Description to remap.
        parent_key (str): New parent_key to use.

    Returns:
        SourceKeyDescription: Description with new key.

    """
    info = deepcopy(info)

    for value in info.values():
        if value["source"] == "prediction":
            value["key"] = parent_key + "." + value["key"]
    return info


def data_key(
    key: str, sensors: Sequence[str] | None = None
) -> SourceKeyDescription:
    """Returns a SourceKeyDescription with data as source.

    Args:
        key (str): Key to use for the data entry.
        sensors (Sequence[str] | None, optional): Which sensors to use for the
            data. Defaults to None.

    Returns:
        SourceKeyDescription: A SourceKeyDescription with data as source.
    """
    if sensors is None:
        return SourceKeyDescription(key=key, source="data")

    return SourceKeyDescription(key=key, source="data", sensors=sensors)


def pred_key(key: str) -> SourceKeyDescription:
    """Returns a SourceKeyDescription with prediction as source.

    Args:
        key (str): Key to use for the data entry.

    Returns:
        SourceKeyDescription: A SourceKeyDescription with prediction as source.
    """
    return SourceKeyDescription(key=key, source="prediction")


def get_field_from_prediction(
    prediction: DictData | NamedTuple,
    old_key_name: SourceKeyDescription,
) -> Tensor | DictStrArrNested:
    """Extracts a field from the prediction dict.

    Args:
        prediction (DictData): Dict containing the model prediction output.
        old_key_name (SourceKeyDescription): Description of the data to
            extract.

    Returns:
        Tensor | DictStrArrNested: Data extracted from the prediction dict.
    """
    if "predictions" in prediction:
        prediction = prediction["predictions"]

    if is_namedtuple(prediction):
        return get_from_namedtuple(
            prediction, old_key_name["key"]  # type: ignore
        )

    old_key = old_key_name["key"]
    return get_dict_nested(prediction, old_key.split("."))  # type: ignore


def get_inputs_for_pred_and_data(
    connection_dict: dict[str, SourceKeyDescription],
    prediction: DictData | NamedTuple,
    data: DictData,
) -> dict[str, Tensor | DictStrArrNested]:
    """Extracts input data from the provided SourceKeyDescription.

    Args:
        connection_dict (dict[str, SourceKeyDescription]): Input Key
            description which is used to gather and remap data from the
            two data dicts.
        prediction (DictData): Dict containing the model prediction output.
        data (DictData):  Dict containing the dataloader output.

    Raises:
        ValueError: If the datasource is invalid.

    Returns:
        out (dict[str, Tensor | DictStrArrNested]): Dict containing new kwargs
            consisting of new key name and data extracted from the data dicts.
    """
    out = {}
    for new_key_name, old_key_name in connection_dict.items():
        # Assign field from data
        if old_key_name["source"] == "data":
            if isinstance(data, list):
                out[new_key_name] = [d[old_key_name["key"]] for d in data]
            else:
                if old_key_name["key"] not in data:
                    raise ValueError(
                        f"Key {old_key_name['key']} not found in data dict."
                        f" Available keys: {data.keys()}"
                    )
                out[new_key_name] = data[old_key_name["key"]]

        # Assign field from model prediction
        elif old_key_name["source"] == "prediction":
            out[new_key_name] = get_field_from_prediction(
                prediction, old_key_name
            )
        else:
            raise ValueError(
                f"Unknown data source {old_key_name['source']}."
                f" Available: [prediction, data]"
            )
    return out
