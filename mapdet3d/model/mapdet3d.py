"""Map-Det3D."""

from __future__ import annotations

import time
from typing import NamedTuple

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from uniception.models.encoders import ViTEncoderInput
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadTokenInput,
)

from mapdet3d.op.mapanything.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
)
from mapdet3d.op.mapanything.image import preprocess_inputs
from mapdet3d.op.mapanything.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from mapdet3d.op.mapdet3d.head import Box3DHead, RoI2Det
from mapdet3d.state.track3d import MapDet3DTrackGraph

from .mapanything import MapAnything


class MapDet3DOut(NamedTuple):
    """Output of the 4D detection model."""

    boxes2d: list[Tensor]  # (B, num_queries, 4)
    boxes3d: list[Tensor]  # (B, num_queries, 10)
    class_ids: list[Tensor]  # (B, num_queries)
    track_ids: list[Tensor]  # (B, num_queries)
    scores: list[Tensor]  # (B, num_queries)


class MapDet3DTrainOut(NamedTuple):
    """Output of the Map-Det3D for training."""

    all_layers_cls_scores: list[Tensor]
    all_layers_bbox_preds: list[Tensor]
    all_layers_outputs_3d: list[Tensor] | None
    enc_outputs_class: Tensor
    enc_outputs_coord: Tensor
    enc_outputs_3d: Tensor | None
    scales: Tensor  # (B, 1)
    dn_meta: dict | None  # Denoising outputs


class MapDet3D(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="mapdet3d",
    repo_url="https://github.com/cvg/Map-Det3D",
    paper_url="https://arxiv.org/abs/2608.12179",
    license="apache-2.0",
    pipeline_tag="object-detection",
    tags=["object-detection", "arxiv:2608.12179"],
):
    """Map-Det3D model."""

    def __init__(
        self,
        window_size: int = 5,
        use_intrinsics: bool = True,
        use_extrinsics: bool = True,
        roi2det: RoI2Det | None = None,
        box3d_head: Box3DHead | None = None,
        use_checkpoint: bool = True,
        track_whole_scene: bool = False,
        track_graph: MapDet3DTrackGraph | None = None,
        compute_fps: bool = False,
    ) -> None:
        """Init."""
        super().__init__()
        self.mapa = MapAnything.from_pretrained("facebook/map-anything")

        # Multi-view settings
        self.views = []
        self.cam_poses = []

        # Compute FPS
        self.compute_fps = compute_fps
        self.dinov2_features = []
        self.additonal_tokens = []

        self.window_size = window_size
        self.use_intrinsics = use_intrinsics
        self.use_extrinsics = use_extrinsics

        # Gradient checkpointing for memory efficiency
        self.use_checkpoint = use_checkpoint

        # 3D Detection
        self.box_head = Box3DHead() if box3d_head is None else box3d_head

        self.roi2det = RoI2Det() if roi2det is None else roi2det

        self._freeze_mapa()

        # Tracking
        self.track_whole_scene = track_whole_scene

        self.track_graph = (
            MapDet3DTrackGraph() if track_graph is None else track_graph
        )

    def _freeze_mapa(self) -> None:
        """Freeze the MapAnything model."""
        mapa = [
            # DINOv2
            self.mapa.encoder,
            # DPT
            self.mapa.dpt_feature_head,
            self.mapa.dpt_regressor_head,
            # Depth
            self.mapa.dense_head,
            self.mapa.dense_adaptor,
            # Pose
            self.mapa.pose_head,
            self.mapa.pose_adaptor,
        ]

        # Freeze multi-modal fusion
        mapa += [
            self.mapa.ray_dirs_encoder,
            self.mapa.depth_encoder,
            self.mapa.depth_scale_encoder,
            self.mapa.cam_rot_encoder,
            self.mapa.cam_trans_encoder,
            self.mapa.cam_trans_scale_encoder,
            self.mapa.fusion_norm_layer,
        ]

        # Multi-view information sharing
        if self.use_checkpoint:
            for i, block in enumerate(
                self.mapa.info_sharing.self_attention_blocks
            ):
                self.mapa.info_sharing.self_attention_blocks[i] = (
                    self.mapa.info_sharing.wrap_module_with_gradient_checkpointing(
                        block
                    )
                )

        for model in mapa:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def _forward_train(
        self,
        input_views,
        boxes2d: list[list[Tensor]] | None = None,
        class_ids: list[list[Tensor]] | None = None,
        input_hw: list[list[tuple[int, int]]] | None = None,
    ) -> MapDet3DTrainOut:
        """MapAnthing forward."""
        batch_size = len(input_views)
        seq_len = len(input_views[0])

        processed_views = []
        for i in range(seq_len):
            single_view = {
                "img": torch.cat([b[i]["img"] for b in input_views], 0),
                "data_norm_type": ["dinov2"] * batch_size,
            }

            if self.use_intrinsics:
                single_view["intrinsics"] = torch.cat(
                    [b[i]["intrinsics"] for b in input_views], 0
                )

            if self.use_extrinsics:
                single_view["camera_poses"] = torch.cat(
                    [b[i]["camera_poses"] for b in input_views], 0
                )

            if self.use_extrinsics:
                single_view["is_metric_scale"] = torch.cat(
                    [b[i]["is_metric_scale"] for b in input_views], 0
                )

            processed_views.append(single_view)

        validated_views = validate_input_views_for_inference(processed_views)

        # Pre-process the input views
        views = preprocess_input_views_for_inference(validated_views)

        num_views = len(views)

        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))

        # Run the image encoder on all the input views
        with torch.no_grad():
            (
                all_encoder_features_across_views,
                all_encoder_registers_across_views,
            ) = self.mapa._encode_n_views(views)

        # Encode the optional geometric inputs and fuse with the encoded
        # features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense
        # representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                all_encoder_features_across_views = (
                    self.mapa._encode_and_fuse_optional_geometric_inputs(
                        views, all_encoder_features_across_views
                    )
                )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.mapa.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views
        # after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )

        # With intermediate_features
        (
            final_info_sharing_multi_view_feat,
            intermediate_info_sharing_multi_view_feat,
        ) = self.mapa.info_sharing(info_sharing_input)

        # VGGT pred head: dpt+pose + use_encoder_features_for_dpt
        dense_head_inputs_list = []

        # Stack all the image encoder features for all views
        stacked_encoder_features = torch.cat(
            all_encoder_features_across_views, dim=0
        )
        dense_head_inputs_list.append(stacked_encoder_features)
        # Stack the first intermediate features for all views
        stacked_intermediate_features_1 = torch.cat(
            intermediate_info_sharing_multi_view_feat[0].features,
            dim=0,
        )
        dense_head_inputs_list.append(stacked_intermediate_features_1)
        # Stack the second intermediate features for all views
        stacked_intermediate_features_2 = torch.cat(
            intermediate_info_sharing_multi_view_feat[1].features,
            dim=0,
        )
        dense_head_inputs_list.append(stacked_intermediate_features_2)
        # Stack the last layer features for all views
        stacked_final_features = torch.cat(
            final_info_sharing_multi_view_feat.features, dim=0
        )
        dense_head_inputs_list.append(stacked_final_features)

        with torch.autocast("cuda", enabled=False):
            # Prepare inputs for the downstream heads
            dense_head_inputs = dense_head_inputs_list

            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Scale prediction
            scale_head_output = self.mapa.scale_head(
                PredictionHeadTokenInput(last_feature=scale_head_inputs)
            )
            scale_final_output = self.mapa.scale_adaptor(
                AdaptorInput(
                    adaptor_feature=scale_head_output.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )

            scales = scale_final_output.value.squeeze(
                -1
            )  # (B, 1, 1) -> (B, 1)

            (
                enc_outputs_class,
                enc_outputs_coord,
                enc_ouptputs_3d,
                all_layers_cls_scores,
                all_layers_bbox_preds,
                all_layers_outputs_3d,
                dn_meta,
            ) = self.box_head(
                dense_head_inputs,
                batch_size_per_view,
                num_views=num_views,
                boxes=boxes2d,
                class_ids=class_ids,
                input_hw=input_hw,
            )

        return MapDet3DTrainOut(
            all_layers_cls_scores=all_layers_cls_scores,
            all_layers_bbox_preds=all_layers_bbox_preds,
            all_layers_outputs_3d=all_layers_outputs_3d,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            enc_outputs_3d=enc_ouptputs_3d,
            scales=torch.tile(scales.squeeze(-1), (num_views, 1)),
            dn_meta=dn_meta,
        )

    def _encode_n_views(self, views) -> list[Tensor]:
        """Encode N views with cached DINOv2 features."""
        num_views = len(views)
        data_norm_type = views[0]["data_norm_type"][0]
        imgs_list = [view["img"] for view in views]
        all_imgs_across_views = torch.cat(imgs_list, dim=0)
        encoder_input = ViTEncoderInput(
            image=all_imgs_across_views[-1].unsqueeze(0),
            data_norm_type=data_norm_type,
        )
        encoder_output = self.mapa.encoder(encoder_input)

        all_encoder_features_across_views = encoder_output.features.chunk(
            num_views, dim=0
        )

        self.dinov2_features.append(all_encoder_features_across_views[0])

        if len(self.dinov2_features) > len(imgs_list):
            self.dinov2_features.pop(0)

        all_encoder_features_across_views = self.dinov2_features

        all_encoder_registers_across_views = None
        if (
            self.mapa.use_register_tokens_from_encoder
            and encoder_output.registers is not None
        ):
            all_encoder_registers_across_views = (
                encoder_output.registers.chunk(num_views, dim=0)
            )
            self.additonal_tokens.append(all_encoder_registers_across_views[0])

            if len(self.additonal_tokens) > len(imgs_list):
                self.additonal_tokens.pop(0)

            all_encoder_registers_across_views = self.additonal_tokens

        return (
            all_encoder_features_across_views,
            all_encoder_registers_across_views,
        )

    def _forward_test(
        self,
        images: Tensor,
        frame_ids: list[int],
        intrinsics: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> MapDet3DOut:
        """Forward for testing."""
        assert len(images) == 1, "Only support batch size 1 now."
        images = images[0]

        if frame_ids[0] == 0:
            self.views.clear()
            self.cam_poses.clear()
            self.dinov2_features.clear()
            self.additonal_tokens.clear()

        input_view = {
            "img": images[0].permute(1, 2, 0),
            "intrinsics": intrinsics[0],
            "data_norm_type": ["dinov2"] * images.shape[0],
        }

        if self.use_extrinsics:
            input_view["camera_poses"] = extrinsics[0]

        if self.use_extrinsics:
            input_view["is_metric_scale"] = torch.tensor([True])

        self.views.append(input_view)
        self.cam_poses.append(extrinsics[0])

        if len(self.views) > self.window_size:
            self.views.pop(0)
            self.cam_poses.pop(0)

        processed_views = preprocess_inputs(self.views, padding_mode=True)

        if self.use_intrinsics:
            intrinsics_list = [
                v["intrinsics"].to(images.device, non_blocking=True)
                for v in processed_views
            ]
        else:
            intrinsics_list = [
                v.pop("intrinsics").to(images.device, non_blocking=True)
                for v in processed_views
            ]

        pad_info_list = [v.pop("pad_info", None) for v in processed_views]
        pad_info = pad_info_list[-1]

        # Validate the input views
        validated_views = validate_input_views_for_inference(processed_views)

        # Transfer the views to the same device as the model
        ignore_keys = set(
            [
                "instance",
                "idx",
                "true_shape",
                "data_norm_type",
            ]
        )
        for view in validated_views:
            for name in view.keys():
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(images.device, non_blocking=True)

        # Pre-process the input views
        views = preprocess_input_views_for_inference(validated_views)
        num_views = len(views)

        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))

        # Run the image encoder on all the input views
        if self.compute_fps:
            start_time = time.time()
            (
                all_encoder_features_across_views,
                all_encoder_registers_across_views,
            ) = self._encode_n_views(views)
        else:
            (
                all_encoder_features_across_views,
                all_encoder_registers_across_views,
            ) = self.mapa._encode_n_views(views)

        # Encode the optional geometric inputs and fuse with the encoded
        # features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense
        # representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self.mapa._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.mapa.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views
        # after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )

        # With intermediate_features
        (
            final_info_sharing_multi_view_feat,
            intermediate_info_sharing_multi_view_feat,
        ) = self.mapa.info_sharing(info_sharing_input)

        # VGGT pred head: dpt+pose + use_encoder_features_for_dpt
        dense_head_inputs_list = []

        # Stack all the image encoder features for all views
        stacked_encoder_features = torch.cat(
            all_encoder_features_across_views, dim=0
        )
        dense_head_inputs_list.append(stacked_encoder_features)
        # Stack the first intermediate features for all views
        stacked_intermediate_features_1 = torch.cat(
            intermediate_info_sharing_multi_view_feat[0].features,
            dim=0,
        )
        dense_head_inputs_list.append(stacked_intermediate_features_1)
        # Stack the second intermediate features for all views
        stacked_intermediate_features_2 = torch.cat(
            intermediate_info_sharing_multi_view_feat[1].features,
            dim=0,
        )
        dense_head_inputs_list.append(stacked_intermediate_features_2)
        # Stack the last layer features for all views
        stacked_final_features = torch.cat(
            final_info_sharing_multi_view_feat.features, dim=0
        )
        dense_head_inputs_list.append(stacked_final_features)

        with torch.autocast("cuda", enabled=False):
            # Prepare inputs for the downstream heads
            dense_head_inputs = dense_head_inputs_list

            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Scale prediction is lightweight, so we can run it in one go
            scale_head_output = self.mapa.scale_head(
                PredictionHeadTokenInput(last_feature=scale_head_inputs)
            )
            scale_final_output = self.mapa.scale_adaptor(
                AdaptorInput(
                    adaptor_feature=scale_head_output.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
            scale_final_output = scale_final_output.value.squeeze(
                -1
            )  # (B, 1, 1) -> (B, 1)

            # 3D Head
            (
                _,
                _,
                _,
                all_layers_cls_scores,
                all_layers_bbox_preds,
                all_layers_outputs_3d,
                _,
            ) = self.box_head(
                dense_head_inputs, batch_size_per_view, num_views=num_views
            )

            # [B, 1] -> [1]
            scale = scale_final_output[0]

            # Get the last frame's detections
            cls_scores = all_layers_cls_scores[-1][-1]
            bbox_preds = all_layers_bbox_preds[-1][-1]
            outputs_3d = all_layers_outputs_3d[-1][-1]

            det_bboxes, det_bboxes3d, scores, det_labels = self.roi2det(
                cls_scores,
                bbox_preds,
                outputs_3d,
                scale=scale,
                intrinsics=intrinsics_list[-1][0],
                img_shape=img_shape,
                ori_shape=(images.shape[2], images.shape[3]),
                pad_info=pad_info,
            )

            if self.track_whole_scene:
                tracks = self.track_graph(
                    det_bboxes3d, scores, extrinsics[0], frame_ids[0]
                )

                det_bboxes3d = tracks.boxes_3d_world
                scores = tracks.scores
                track_ids = tracks.track_ids
                det_labels = torch.zeros_like(track_ids)

                # A scene-level track has no single 2D box.
                det_bboxes = det_bboxes3d.new_empty((0, 4))
            else:
                track_ids = torch.arange(len(det_labels))

        if self.compute_fps:
            print(f"Inference time: {time.time() - start_time:.2f} seconds")

            print(
                f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            )

        return MapDet3DOut(
            boxes2d=[det_bboxes],
            boxes3d=[det_bboxes3d],
            class_ids=[det_labels],
            track_ids=[track_ids],
            scores=[scores],
        )

    def forward(
        self,
        images: list[list[Tensor]] | None = None,
        views=None,
        boxes2d: list[list[Tensor]] | None = None,
        class_ids: list[list[Tensor]] | None = None,
        input_hw: list[list[tuple[int, int]]] | None = None,
        intrinsics: list[list[Tensor]] | None = None,
        extrinsics: list[list[Tensor]] | None = None,
        frame_ids: list[list[int]] | None = None,
    ) -> MapDet3DOut | MapDet3DTrainOut:
        """Forward."""
        if self.training:
            return self._forward_train(views, boxes2d, class_ids, input_hw)
        else:
            return self._forward_test(
                images, frame_ids, intrinsics, extrinsics
            )
