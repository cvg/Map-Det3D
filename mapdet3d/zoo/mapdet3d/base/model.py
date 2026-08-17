"""Model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from mapdet3d.config.config_dict import class_config
from mapdet3d.config.typing import ExperimentParameters
from mapdet3d.model.mapdet3d import MapDet3D
from mapdet3d.op.mapdet3d.head import Box3DHead, RoI2Det


def get_hyperparams_cfg() -> ExperimentParameters:
    """Get the hyperparameters."""
    params = ExperimentParameters()

    # Training
    params.samples_per_gpu = 5
    params.workers_per_gpu = 10
    params.accumulate_grad_batches = 1
    params.lr = 0.0001
    params.weight_decay = 0.0001

    # Learning rate
    params.num_iters = 50000
    params.lr_iters = 50000

    # Validation and checkpointing
    params.val_freq = 5000
    params.checkpoint_period = 5000

    # Model
    params.window_size = 5
    params.use_intrinsics = True
    params.use_extrinsics = True

    # RoI2Det
    params.nms = False
    params.score_threshold = 0.0
    params.iou_threshold = 0.5

    # Tracking
    params.track_whole_scene = False

    return params


def get_model_cfg(
    params: ExperimentParameters, use_checkpoint: bool | FieldReference = False
) -> ConfigDict:
    """Get the model config."""
    return class_config(
        MapDet3D,
        use_intrinsics=params.use_intrinsics,
        use_extrinsics=params.use_extrinsics,
        use_checkpoint=use_checkpoint,
        window_size=params.window_size,
        track_whole_scene=params.track_whole_scene,
        box3d_head=class_config(Box3DHead, use_checkpoint=use_checkpoint),
        roi2det=class_config(
            RoI2Det,
            nms=params.nms,
            score_threshold=params.score_threshold,
            iou_threshold=params.iou_threshold,
        ),
    )
