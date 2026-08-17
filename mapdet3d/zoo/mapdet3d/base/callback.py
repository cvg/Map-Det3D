"""Callbacks config."""

from __future__ import annotations

from mapdet3d.config.config_dict import FieldConfigDict, class_config
from mapdet3d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from mapdet3d.engine.connectors import CallbackConnector
from mapdet3d.eval.detect3d_frame import Detect3DFrameEvaluator
from mapdet3d.eval.detect3d_scene import Detect3DSceneEvaluator
from mapdet3d.vis.rerun import RerunVisualizer
from mapdet3d.zoo.base import get_default_callbacks_cfg

from .connector import (
    CONN_COCO_DET3D_EVAL,
    CONN_RERUN_VIS,
    CONN_SCENE_DET3D_EVAL,
)


def get_callback_cfg(
    output_dir: FieldConfigDict | str,
    per_scene: bool = False,
) -> list[FieldConfigDict]:
    """Get callbacks config."""
    callbacks = get_default_callbacks_cfg(epoch_based=False)

    # Rerun Visualizer
    if per_scene:
        callbacks.append(
            class_config(
                VisualizerCallback,
                visualizer=class_config(
                    RerunVisualizer,
                    plot_traj=True,
                    convert_to_world=True,
                    log_mesh=True,
                    save_to_disk=True,
                    output_dir=output_dir,
                ),
                save_to_disk=False,
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_RERUN_VIS
                ),
            )
        )
    else:
        callbacks.append(
            class_config(
                VisualizerCallback,
                visualizer=class_config(
                    RerunVisualizer,
                    plot_traj=True,
                    log_mesh=True,
                    save_to_disk=True,
                    output_dir=output_dir,
                ),
                save_to_disk=False,
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_RERUN_VIS
                ),
            )
        )

    # Evaluation
    if per_scene:
        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=class_config(Detect3DSceneEvaluator),
                output_dir=output_dir,
                save_predictions=True,
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_SCENE_DET3D_EVAL
                ),
                metrics_to_eval=["3D"],
            )
        )
    else:
        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=class_config(Detect3DFrameEvaluator),
                output_dir=output_dir,
                save_predictions=True,
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_COCO_DET3D_EVAL
                ),
                metrics_to_eval=["2D", "3D"],
            )
        )

    return callbacks
