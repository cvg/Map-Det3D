"""Base module for callbacks."""

from __future__ import annotations

import lightning.pytorch as pl
from torch import Tensor

from mapdet3d.common.typing import DictStrArrNested
from mapdet3d.data.typing import DictData
from mapdet3d.engine.connectors import CallbackConnector


class Callback(pl.Callback):
    """Base class for Callbacks."""

    def __init__(
        self,
        epoch_based: bool = True,
        train_connector: None | CallbackConnector = None,
        test_connector: None | CallbackConnector = None,
    ) -> None:
        """Init callback.

        Args:
            epoch_based (bool, optional): Whether the callback is epoch based.
                Defaults to False.
            train_connector (None | CallbackConnector, optional): Defines which
                kwargs to use during training for different callbacks. Defaults
                to None.
            test_connector (None | CallbackConnector, optional): Defines which
                kwargs to use during testing for different callbacks. Defaults
                to None.
        """
        self.epoch_based = epoch_based
        self.train_connector = train_connector
        self.test_connector = test_connector

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Setup callback."""

    def get_train_callback_inputs(
        self, outputs: DictData, batch: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the data connector results for training.

        It extracts the required data from prediction and datas and passes it
        to the next component with the provided new key.

        Args:
            outputs (DictData): Outputs of the model.
            batch (DictData): Batch data.

        Returns:
            dict[str, Tensor | DictStrArrNested]: Data connector results.

        Raises:
            AssertionError: If train connector is None.
        """
        assert self.train_connector is not None, "Train connector is None."

        return self.train_connector(outputs, batch)

    def get_test_callback_inputs(
        self, outputs: DictData, batch: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the data connector results for inference.

        It extracts the required data from prediction and datas and passes it
        to the next component with the provided new key.

        Args:
            outputs (DictData): Outputs of the model.
            batch (DictData): Batch data.

        Returns:
            dict[str, Tensor | DictStrArrNested]: Data connector results.

        Raises:
            AssertionError: If test connector is None.
        """
        assert self.test_connector is not None, "Test connector is None."

        return self.test_connector(outputs, batch)
