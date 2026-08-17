"""Data module composing the data loading pipeline."""

from __future__ import annotations

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from mapdet3d.config import instantiate_classes
from mapdet3d.config.typing import DataConfig
from mapdet3d.data.typing import DictData


class DataModule(pl.LightningDataModule):
    """DataModule for PyTorch Lightning.

    This is a wrapper allows to use PyTorch-Lightning for training and testing.
    """

    def __init__(self, data_cfg: DataConfig) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.data_cfg = data_cfg

    def train_dataloader(self) -> DataLoader[DictData]:
        """Return dataloader for training."""
        if self.trainer is not None and hasattr(self.trainer, "seed"):
            seed = self.trainer.seed
        else:
            seed = None
        return instantiate_classes(self.data_cfg.train_dataloader, seed=seed)

    def test_dataloader(self) -> list[DataLoader[DictData]]:
        """Return dataloaders for testing."""
        return instantiate_classes(self.data_cfg.test_dataloader)

    def val_dataloader(self) -> list[DataLoader[DictData]]:
        """Return dataloaders for validation."""
        return self.test_dataloader()
