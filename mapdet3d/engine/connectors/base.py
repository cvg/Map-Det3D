"""Base data connector to define data structures for data connection."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor

from mapdet3d.common.typing import DictStrArrNested
from mapdet3d.data.typing import DictData, DictDataOrList

from .util import SourceKeyDescription, get_inputs_for_pred_and_data


class DataConnector:
    """Defines which data to pass to which component.

    It extracts the required data from a 'DictData' objects and passes it to
    the next component with the provided new key.
    """

    def __init__(self, key_mapping: dict[str, str]):
        """Initializes the data connector with static remapping of the keys.

        Args:
            key_mapping (dict[str, str]): Defines which kwargs to pass onto the
                module.

        Simple Example Configuration:

        >>> train = dict(images = "images", gt = "gt_images)
        >>> train_data_connector = DataConnector(train)
        >>> test = dict(images = "images")
        >>> test_data_connector = DataConnector(test)
        """
        self.key_mapping = key_mapping

    def __call__(self, data: DictDataOrList) -> DictData:
        """Returns the kwargs that are passed to the module.

        Args:
            data (DictDataorList): The data (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            DictData: kwargs that are passed onto the model.
        """
        if isinstance(data, list):
            return {
                k: [d[v] for d in data] for k, v in self.key_mapping.items()
            }
        return {k: data[v] for k, v in self.key_mapping.items()}


class LossConnector:
    """Defines which data to pass to loss module of the training pipeline.

    It extracts the required data from prediction and data and passes it to
    the next component with the provided new key.
    """

    def __init__(self, key_mapping: dict[str, SourceKeyDescription]) -> None:
        """Initializes the data connector with static remapping of the keys."""
        self.key_mapping = key_mapping

    def __call__(
        self, prediction: DictData | NamedTuple, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the loss module.

        Args:
            prediction (DictData | NamedTuple): The output from model.
            data (DictData): The data dictionary from the dataloader which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the loss.
        """
        return get_inputs_for_pred_and_data(self.key_mapping, prediction, data)


class CallbackConnector:
    """Data connector for the callback.

    It extracts the required data from prediction and datas and passes it to
    the next component with the provided new key.
    """

    def __init__(self, key_mapping: dict[str, SourceKeyDescription]) -> None:
        """Initializes the data connector with static remapping of the keys."""
        self.key_mapping = key_mapping

    def __call__(
        self, prediction: DictData | NamedTuple, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the callback.

        Args:
            prediction (DictData | NamedTuple): The output from model.
            data (DictData): The data dictionary from the dataloader which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the callback.
        """
        return get_inputs_for_pred_and_data(self.key_mapping, prediction, data)
