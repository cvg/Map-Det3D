"""PyTorch Lightning callbacks utilities."""

from __future__ import annotations

import lightning.pytorch as pl
from torch import nn

from mapdet3d.engine.loss_module import LossModule
from mapdet3d.engine.training_module import TrainingModule


def get_model(model: pl.LightningModule) -> nn.Module:
    """Get model from pl module."""
    if isinstance(model, TrainingModule):
        return model.model
    return model


def get_loss_module(loss_module: pl.LightningModule) -> LossModule:
    """Get loss_module from pl module."""
    assert hasattr(loss_module, "loss_module") and isinstance(
        loss_module.loss_module, LossModule
    ), "Loss module is not set in the training module."
    return loss_module.loss_module
