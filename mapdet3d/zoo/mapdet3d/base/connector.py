"""Data connector."""

from mapdet3d.data.const import CommonKeys as K
from mapdet3d.engine.connectors import data_key, pred_key

CONN_BBOX_3D_TRAIN = {
    "views": "processed_views",
    "boxes2d": "boxes2d",
    "class_ids": "class_ids",
    "input_hw": K.input_hw,
}

CONN_BBOX_3D_TEST = {
    "images": K.images,
    "intrinsics": K.intrinsics,
    "extrinsics": K.extrinsics,
    "frame_ids": K.frame_ids,
}

CONN_BOXES3D_LOSS = {
    "all_layers_cls_scores": pred_key("all_layers_cls_scores"),
    "all_layers_bbox_preds": pred_key("all_layers_bbox_preds"),
    "all_layers_outputs_3d": pred_key("all_layers_outputs_3d"),
    "scales": pred_key("scales"),
    "enc_cls_scores": pred_key("enc_outputs_class"),
    "enc_bbox_preds": pred_key("enc_outputs_coord"),
    "enc_boxes3d_preds": pred_key("enc_outputs_3d"),
    "input_hw": data_key(K.input_hw),
    "intrinsics": data_key(K.intrinsics),
    "batch_gt_boxes": data_key(K.boxes2d),
    "batch_gt_boxes_classes": data_key("class_ids"),
    "batch_gt_boxes3d": data_key(K.boxes3d),
    "dn_meta": pred_key("dn_meta"),
}


CONN_RERUN_VIS = {
    "images": data_key(K.original_images),
    "sequence_names": data_key(K.sequence_names),
    "original_hw": data_key(K.original_hw),
    "intrinsics": data_key("original_intrinsics"),
    "extrinsics": data_key(K.extrinsics),
    "boxes3d": pred_key("boxes3d"),
    "scores": pred_key("scores"),
    "track_ids": pred_key("track_ids"),
    "mesh_paths": data_key("mesh_path"),
}

CONN_COCO_DET3D_EVAL = {
    "coco_image_id": data_key("image_ids"),
    "pred_boxes": pred_key("boxes2d"),
    "pred_boxes3d": pred_key("boxes3d"),
    "pred_classes": pred_key("class_ids"),
    "pred_scores": pred_key("scores"),
    "gt_boxes": data_key(K.boxes2d),
    "gt_boxes3d": data_key(K.boxes3d),
    "gt_classes": data_key(K.boxes3d_classes),
}

CONN_SCENE_DET3D_EVAL = {
    "seq_names": data_key(K.sequence_names),
    "pred_boxes3d": pred_key("boxes3d"),
    "pred_scores": pred_key("scores"),
    "gt_boxes3d": data_key("boxes3d_world"),
}
