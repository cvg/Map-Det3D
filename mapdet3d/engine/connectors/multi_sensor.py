"""Data connector for multi-sensor dataset."""

from __future__ import annotations

from typing import NamedTuple

from mapdet3d.data.typing import DictData, DictDataOrList

from .base import CallbackConnector, DataConnector, LossConnector
from .util import SourceKeyDescription, get_field_from_prediction


class MultiSensorDataConnector(DataConnector):
    """Data connector for multi-sensor data dict."""

    def __init__(self, key_mapping: dict[str, str | SourceKeyDescription]):
        """Initializes the data connector with static remapping of the keys.

        Args:
            key_mapping (dict[str, | SourceKeyDescription]): Defines which
                kwargs to pass onto the module.

        TODO: Add Simple Example Configuration:
        """
        _key_mapping = {}
        multi_sensor_key_mapping = {}

        for k, v in key_mapping.items():
            if isinstance(v, dict):
                sensors = v.get("sensors")
                if sensors is not None:
                    multi_sensor_key_mapping[k] = v
                else:
                    _key_mapping[k] = v["key"]
            else:
                _key_mapping[k] = v

        super().__init__(_key_mapping)
        self.multi_sensor_key_mapping = multi_sensor_key_mapping

    def __call__(self, data: DictDataOrList) -> DictData:
        """Returns the train input for the model."""
        input_dict = super().__call__(data)

        for target_key, source_key in self.multi_sensor_key_mapping.items():
            key = source_key["key"]
            sensors = source_key["sensors"]

            if isinstance(data, list):
                input_dict[target_key] = [
                    [d[sensor][key] for sensor in sensors] for d in data
                ]
            else:
                input_dict[target_key] = [
                    data[sensor][key] for sensor in sensors
                ]
        return input_dict


class MultiSensorLossConnector(LossConnector):
    """Multi-sensor Data connector for loss module of the training pipeline."""

    def __call__(
        self, prediction: DictData | NamedTuple, data: DictData
    ) -> DictData:
        """Returns the kwargs that are passed to the loss module.

        Args:
            prediction (DictData | NamedTuple): The output from model.
            data (DictData): The data dictionary from the dataloader which
                contains all data that was loaded.

        Returns:
            DictData: kwargs that are passed onto the loss.
        """
        return get_multi_sensor_inputs(self.key_mapping, prediction, data)


class MultiSensorCallbackConnector(CallbackConnector):
    """Multi-sensor data connector for the callback."""

    def __call__(
        self, prediction: DictData | NamedTuple, data: DictData
    ) -> DictData:
        """Returns the kwargs that are passed to the callback.

        Args:
            prediction (DictData | NamedTuple): The output from model.
            data (DictData): The data dictionary from the dataloader which
                contains all data that was loaded.

        Returns:
            DictData: kwargs that are passed onto the callback.
        """
        return get_multi_sensor_inputs(self.key_mapping, prediction, data)


def get_multi_sensor_inputs(
    connection_dict: dict[str, SourceKeyDescription],
    prediction: DictData | NamedTuple,
    data: DictData,
) -> DictData:
    """Extracts multi-sensor input data from the provided SourceKeyDescription.

    Args:
        connection_dict (dict[str, SourceKeyDescription]): Input Key
            description which is used to gather and remap data from the
            two data dicts.
        prediction (DictData): Dict containing the model prediction output.
        data (DictData):  Dict containing the dataloader output.

    Raises:
        ValueError: If the datasource is invalid.

    Returns:
        out (DictData): Dict containing new kwargs consisting of new key name
            and data extracted from the data dicts.
    """
    out: DictData = {}
    for new_key_name, old_key_name in connection_dict.items():
        # Assign field from data
        if old_key_name["source"] == "data":
            sensors = old_key_name.get("sensors")

            if sensors is None:
                if old_key_name["key"] not in data:
                    raise ValueError(
                        f"Key {old_key_name['key']} not found in data dict."
                        f" Available keys: {data.keys()}"
                    )
                out[new_key_name] = data[old_key_name["key"]]
            else:
                out[new_key_name] = [
                    data[sensor][old_key_name["key"]] for sensor in sensors
                ]

        # Assign field from prediction
        elif old_key_name["source"] == "prediction":
            out[new_key_name] = get_field_from_prediction(
                prediction, old_key_name
            )
        else:
            raise ValueError(
                f"Unknown data source {old_key_name['source']}."
                f"Available: [prediction, data]"
            )
    return out
