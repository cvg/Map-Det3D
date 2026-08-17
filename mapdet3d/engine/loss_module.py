"""Loss module maps loss function input keys and controls loss weight."""

from __future__ import annotations

from typing import TypedDict, Union

import torch
from torch import Tensor, nn
from typing_extensions import NotRequired

from mapdet3d.common.named_tuple import is_namedtuple
from mapdet3d.common.typing import LossesType
from mapdet3d.data.typing import DictData
from mapdet3d.engine.connectors import LossConnector
from mapdet3d.op.loss.base import Loss

NestedLossesType = Union[dict[str, "NestedLossesType"], LossesType]


class LossDefinition(TypedDict):
    """Loss definition.

    Attributes:
        loss (Loss | nn.Module): Loss function to use.
        connector (LossConnector): Connector to use for the loss.
        weight (float | dict[str, float], optional): Weight to use for the
            loss.
        name (str, optional): Name to use for the loss.
    """

    loss: Loss | nn.Module
    connector: LossConnector
    weight: NotRequired[float | dict[str, float]]
    name: NotRequired[str]


def _get_tensors_nested(
    loss_dict: NestedLossesType, prefix: str = ""
) -> list[tuple[str, Tensor]]:
    """Get tensors from loss dict.

    Args:
        loss_dict (LossesType): Loss dict.
        prefix (str, optional): Prefix to add to keys. Defaults to "".

    Returns:
        list[tuple[str, Tensor]]: List of tensors.

    Raises:
        ValueError: If loss dict contains non-tensor or dict values.
    """
    named_tensors: list[tuple[str, Tensor]] = []
    for key in loss_dict:
        value = loss_dict[key]

        if isinstance(value, Tensor):
            named_tensors.append((prefix + key, value))
        elif isinstance(value, dict):
            named_tensors.extend(
                _get_tensors_nested(value, prefix + key + ".")
            )
        else:
            raise ValueError(
                f"Loss dict must only contain tensors or dicts. "
                f"Found {type(loss_dict[key])} at {prefix + key}."
            )
    return named_tensors


class LossModule(nn.Module):
    """Loss module maps input keys and combines losses with weights.

    This loss combines multiple losses with weights. The loss values are
    weighted by the corresponding weight and returned as a dictionary.
    """

    def __init__(
        self,
        losses: list[LossDefinition] | LossDefinition,
        exclude_attributes: list[str] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Each loss will be called with arguments matching the kwargs of the loss
        function through its connector. By default, the weight is set to 1.0.

        Args:
            losses (list[LossDefinition]): List of loss definitions.
            exclude_attributes (list[str] | None): List of attributes returned
                by the losses that should be excluded from the total loss
                computation. Use it to log metrics that should not be
                optimised. Defaults to None.

        Example:
            >>> loss = LossModule(
            >>>     [
            >>>         {
            >>>             "loss": nn.MSELoss(),
            >>>             "weight": 0.7,
            >>>             "connector": LossConnector(
            >>>                 {
            >>>                     "input": pred_key("input"),
            >>>                     "target": data_key("target"),
            >>>                 }
            >>>             ),
            >>>         },
            >>>         {
            >>>             "loss": nn.L1Loss(),
            >>>             "weight": 0.3
            >>>             "connector": LossConnector(
            >>>                 {
            >>>                     "input": pred_key("input"),
            >>>                     "target": data_key("target"),
            >>>                 }
            >>>             ),
            >>>         },
            >>>     ]
            >>> )
        """
        super().__init__()
        self.losses: list[LossDefinition] = []

        if not isinstance(losses, list):
            losses = [losses]

        for loss in losses:
            assert "loss" in loss, "Loss definition must contain a loss."
            assert (
                "connector" in loss
            ), "Loss definition must contain a connector."

            if "name" not in loss:
                loss["name"] = loss["loss"].__class__.__name__

            if "weight" not in loss:
                loss["weight"] = 1.0

            self.losses.append(loss)

        self.exclude_attributes = exclude_attributes

    def forward(
        self, output: DictData, batch: DictData
    ) -> tuple[Tensor, dict[str, float]]:
        """Forward of loss module.

        This function will call all loss functions and return a dictionary
        containing the loss values. The loss values are weighted by the
        corresponding weight.

        If two losses have the same name, the name will be appended with
        two underscores.

        Args:
            output (DictData): Output of the model.
            batch (DictData): Batch data.

        Returns:
            total_loss: The total loss value.
            metrics: The metrics disctionary.
        """
        loss_dict: LossesType = {}

        for loss in self.losses:
            loss_values_as_dict: LossesType = {}
            name = loss["name"]

            loss_value = loss["loss"](**loss["connector"](output, batch))

            # Convert loss value to one level dict.
            if isinstance(loss_value, Tensor):
                # Loss returned a simple tensor
                loss_values_as_dict[name] = loss_value
            elif isinstance(loss_value, dict):
                # Loss returned a dictionary.
                for loss_name, loss_value in _get_tensors_nested(
                    loss_value, name + "."
                ):
                    loss_values_as_dict[loss_name] = loss_value
            elif is_namedtuple(loss_value):
                # Loss returned a named tuple.
                for loss_name, loss_value in zip(
                    loss_value._fields, loss_value
                ):
                    loss_values_as_dict[name + "." + loss_name] = loss_value

            # Assign values
            for key, value in loss_values_as_dict.items():
                if value is None:
                    continue

                if isinstance(loss["weight"], dict):
                    loss_weight = loss["weight"].get(key, 1.0)
                else:
                    loss_weight = loss["weight"]

                while key in loss_dict:
                    key = "__" + key

                loss_dict[key] = torch.mul(loss_weight, value)

        # Convert loss_dict to total loss and metrics dictionary
        metrics: dict[str, float] = {}
        keep_loss_dict: LossesType = {}
        for k, v in loss_dict.items():
            metrics[k] = v.detach().cpu().item()
            if (
                self.exclude_attributes is None
                or k not in self.exclude_attributes
            ):
                keep_loss_dict[k] = v
        total_loss: Tensor = sum(keep_loss_dict.values())  # type: ignore
        metrics["loss"] = total_loss.detach().cpu().item()

        return total_loss, metrics
