"""Map-Det3D with CA-1M."""

from __future__ import annotations

from mapdet3d.config.config_dict import class_config
from mapdet3d.config.typing import ExperimentConfig
from mapdet3d.data.io.hdf5 import HDF5Backend
from mapdet3d.engine.connectors import DataConnector
from mapdet3d.zoo.base import get_default_cfg
from mapdet3d.zoo.base.pl import get_pl_cfg
from mapdet3d.zoo.mapdet3d.base.callback import get_callback_cfg
from mapdet3d.zoo.mapdet3d.base.connector import (
    CONN_BBOX_3D_TEST,
    CONN_BBOX_3D_TRAIN,
)
from mapdet3d.zoo.mapdet3d.base.data import get_ca1m_data_cfg
from mapdet3d.zoo.mapdet3d.base.loss import get_loss_cfg
from mapdet3d.zoo.mapdet3d.base.model import get_hyperparams_cfg, get_model_cfg
from mapdet3d.zoo.mapdet3d.base.optim import get_optim_cfg


def get_config() -> ExperimentConfig:
    """Returns mapdet3d config."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="mapdet3d_ca1m")

    config.use_tf32 = True
    config.use_checkpoint = True

    # Hyper Parameters
    params = get_hyperparams_cfg()

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    config.data = get_ca1m_data_cfg(
        data_backend=data_backend,
        window_size=params.window_size,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model = get_model_cfg(params, use_checkpoint=config.use_checkpoint)

    config.loss = get_loss_cfg(params)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = get_optim_cfg(params)

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_3D_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_3D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    callbacks = get_callback_cfg(output_dir=config.output_dir)

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(
        config, params, bf16_mixed=True, epoch_based=False
    )

    return config.value_mode()
