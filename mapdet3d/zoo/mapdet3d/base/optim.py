"""3D Grounding DINO Optimizer config."""

from __future__ import annotations

from ml_collections import ConfigDict
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mapdet3d.config import class_config
from mapdet3d.config.typing import ExperimentParameters
from mapdet3d.zoo.base import get_lr_scheduler_cfg, get_optimizer_cfg


def get_optim_cfg(params: ExperimentParameters) -> list[ConfigDict]:
    """Returns the optimizer configuration."""

    lr_schedulers = []

    lr_schedulers.append(
        get_lr_scheduler_cfg(
            class_config(CosineAnnealingLR, T_max=params.lr_iters),
            epoch_based=False,
        ),
    )

    param_groups = [
        {
            "custom_keys": [
                # Multi-modal fusion
                "mapa.ray_dirs_encoder",
                "mapa.depth_encoder",
                "mapa.depth_scale_encoder",
                "mapa.cam_rot_encoder",
                "mapa.cam_trans_encoder",
                "mapa.cam_trans_scale_encoder",
                "mapa.fusion_norm_layer",
                # Multi-view
                "mapa.info_sharing",
                # Metric Scale
                "mapa.scale_head",
                "mapa.scale_adaptor",
            ],
            "lr_mult": 0.1,
        },
    ]

    optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                AdamW, lr=params.lr, weight_decay=params.weight_decay
            ),
            lr_schedulers=lr_schedulers,
            param_groups=param_groups,
        )
    ]
    return optimizers
