"""Loss config."""

from __future__ import annotations

from ml_collections import ConfigDict

from mapdet3d.config import class_config
from mapdet3d.engine.connectors import LossConnector
from mapdet3d.engine.loss_module import LossModule
from mapdet3d.op.mapdet3d.loss import MapDet3DLoss
from mapdet3d.zoo.mapdet3d.base.connector import CONN_BOXES3D_LOSS


def get_loss_cfg(
    params: ConfigDict,
) -> ConfigDict:
    """Returns the loss configuration."""
    losses = []
    boxes3d_loss = {
        "loss": class_config(MapDet3DLoss),
        "weight": 1.0,
        "connector": class_config(
            LossConnector, key_mapping=CONN_BOXES3D_LOSS
        ),
    }

    losses.append(boxes3d_loss)

    return class_config(LossModule, losses=losses)
