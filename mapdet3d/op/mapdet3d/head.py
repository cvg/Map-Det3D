"""3D Bbox head for mapdet3d."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.ops import nms

from mapdet3d.op.box2d import bbox_cxcywh_to_xyxy
from mapdet3d.op.geometry.projection import project_points
from mapdet3d.op.geometry.rotation import (
    R_from_allocentric,
    matrix_to_quaternion,
    rotation_6d_to_matrix,
)
from mapdet3d.op.layer.mlp import MLP
from mapdet3d.op.layer.transformer import get_clones, inverse_sigmoid

from .dino import CdnQueryGenerator, DinoTransformerDecoder


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    """Initialize module with Xavier initialization."""
    assert distribution in {"uniform", "normal"}
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


class Box3DHead(nn.Module):
    """3D Bounding Box head."""

    def __init__(
        self,
        input_embed_dims: int = 1536,
        embed_dims: int = 256,
        num_classes: int = 1,
        num_reg_fcs: int = 2,
        reg_dims: int = 12,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        use_checkpoint: bool = False,
    ) -> None:
        """Initialize the 3D Grounding DINO head."""
        super().__init__()
        self.embed_dims = embed_dims
        self.reg_dims = reg_dims
        self.num_reg_fcs = num_reg_fcs
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels

        self.num_queries = num_queries

        # Denoising
        self.dn_query_generator = CdnQueryGenerator(
            num_classes=num_classes,
            embed_dims=embed_dims,
            num_matching_queries=num_queries,
            label_noise_scale=0.5,
            box_noise_scale=1.0,  # 0.4 for DN-DETR
            dynamic=True,
            num_groups=None,
            num_dn_queries=100,
        )

        self.box_token = nn.Parameter(
            torch.zeros(self.num_queries, self.embed_dims)
        )
        torch.nn.init.trunc_normal_(self.box_token, std=0.02)

        self.act = nn.ModuleList()
        self.act.append(MLP(input_embed_dims, output_dim=self.embed_dims))
        self.act.append(MLP(input_embed_dims, output_dim=self.embed_dims))
        self.act.append(MLP(input_embed_dims, output_dim=self.embed_dims))
        self.act.append(MLP(input_embed_dims, output_dim=self.embed_dims))

        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        self.decoder = DinoTransformerDecoder(
            num_layers=6, use_checkpoint=use_checkpoint
        )

        fc_cls = nn.Linear(self.embed_dims, self.num_classes)
        fc_reg = self._get_fc_reg(reg_dims=4)

        self.cls_branches = get_clones(fc_cls, 7)
        self.reg_branches = get_clones(fc_reg, 7)

        # 3D Head
        fc_reg_cen = self._get_fc_reg(reg_dims=2)
        self.reg_cen_branches = get_clones(fc_reg_cen, 7)

        fc_reg_depth = self._get_fc_reg(reg_dims=1)
        self.reg_depth_branches = get_clones(fc_reg_depth, 7)

        fc_reg_dim = self._get_fc_reg(reg_dims=3)
        self.reg_dim_branches = get_clones(fc_reg_dim, 7)

        fc_reg_rot = self._get_fc_reg(reg_dims=6)
        self.reg_rot_branches = get_clones(fc_reg_rot, 7)

        self.init_weights()

    def _get_fc_reg(self, reg_dims: int) -> nn.Sequential:
        """Get the fc regression."""
        fc_reg = []
        for _ in range(self.num_reg_fcs):
            fc_reg.append(nn.Linear(self.embed_dims, self.embed_dims))
            fc_reg.append(nn.ReLU())
        fc_reg.append(nn.Linear(self.embed_dims, reg_dims))
        return nn.Sequential(*fc_reg)

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, bias_init)

        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        nn.init.xavier_uniform_(self.memory_trans_fc.weight)

        for m in self.reg_cen_branches:
            xavier_init(m, distribution="uniform")
        for m in self.reg_depth_branches:
            xavier_init(m, distribution="uniform")
        for m in self.reg_dim_branches:
            xavier_init(m, distribution="uniform")
        for m in self.reg_rot_branches:
            xavier_init(m, distribution="uniform")

    def gen_encoder_output_proposals(
        self, memory: Tensor, spatial_shapes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory."""
        bs = memory.size(0)

        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW
            scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)

        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).sum(-1, keepdim=True) == output_proposals.shape[-1]

        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))

        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        # [bs, sum(hw), 2]
        output_memory = memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)

        return output_memory, output_proposals

    def single_3d(self, hidden_state: Tensor, layer_id: int) -> Tensor:
        """Single layer forward pass of the 3D Grounding DINO head."""
        reg_cen_output = self.reg_cen_branches[layer_id](hidden_state)
        reg_depth_output = self.reg_depth_branches[layer_id](hidden_state)
        reg_dim_output = self.reg_dim_branches[layer_id](hidden_state)
        reg_rot_output = self.reg_rot_branches[layer_id](hidden_state)

        reg_output = torch.cat(
            [reg_cen_output, reg_depth_output, reg_dim_output, reg_rot_output],
            dim=-1,
        )

        return reg_output

    def enc_fwd(
        self,
        memory: Tensor,
        spatial_shapes: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encoder Forward."""
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, spatial_shapes
        )

        enc_outputs_coord_unact = (
            self.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )
        enc_outputs_class = self.cls_branches[self.decoder.num_layers](
            output_memory
        )

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]
        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, self.num_classes),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        topk_output_memory = torch.gather(
            output_memory,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, self.embed_dims),
        )

        topk_output_3d = self.single_3d(topk_output_memory, layer_id=-1)

        return (
            topk_score,
            topk_coords,
            topk_output_3d,
            topk_coords_unact,
        )

    def forward(
        self,
        dense_head_inputs: list[Tensor],
        batch_size_per_view: int,
        num_views: int,
        boxes: list[Tensor] | None = None,
        class_ids: list[Tensor] | None = None,
        input_hw: list[tuple[int, int]] | None = None,
    ):
        """Forward pass of the 3D DINO head."""
        query = (
            self.box_token[:, None, :]
            .repeat(1, batch_size_per_view * num_views, 1)
            .transpose(0, 1)
        )

        feats_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(dense_head_inputs):
            b, c, _, _ = feat.shape

            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            spatial_shapes.append(spatial_shape)

            feat_flatten = feat.reshape(b, c, -1).permute(0, 2, 1)
            feat_flatten = self.act[lvl](feat_flatten)
            feats_flatten.append(feat_flatten)

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        memory = torch.cat(feats_flatten, 1)

        valid_ratios = memory.new_ones(b, len(dense_head_inputs), 2)

        # Encoder
        topk_score, topk_coords, topk_output_3d, topk_coords_unact = (
            self.enc_fwd(memory, spatial_shapes)
        )

        # Denoising part
        dn_meta = None
        if self.training:
            assert (
                boxes is not None
            ), "GT boxes are required for training with denoising."
            assert (
                class_ids is not None
            ), "GT class IDs are required for training with denoising."
            assert (
                input_hw is not None
            ), "Input image sizes are required for training with denoising."

            batch_gt_boxes = [
                boxes[b][v]
                for v in range(num_views)
                for b in range(batch_size_per_view)
            ]

            batch_class_ids = [
                class_ids[b][v]
                for v in range(num_views)
                for b in range(batch_size_per_view)
            ]

            batch_input_hw = [
                input_hw[b][v]
                for v in range(num_views)
                for b in range(batch_size_per_view)
            ]

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = (
                self.dn_query_generator(
                    batch_gt_boxes, batch_class_ids, batch_input_hw
                )
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1
            )
        else:
            reference_points = topk_coords_unact
            dn_mask = None

        reference_points = reference_points.sigmoid()

        hidden_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=None,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.reg_branches,
        )

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            hidden_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_3d = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])

            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)

            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()

            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

            if self.training:
                hidden_state_3d = hidden_state[
                    :, dn_meta["num_denoising_queries"] :, :
                ]
            else:
                hidden_state_3d = hidden_state

            reg_output = self.single_3d(hidden_state_3d, layer_id)
            all_layers_outputs_3d.append(reg_output)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_3d = torch.stack(all_layers_outputs_3d)

        return (
            topk_score,
            topk_coords,
            topk_output_3d,
            all_layers_outputs_classes,
            all_layers_outputs_coords,
            all_layers_outputs_3d,
            dn_meta,
        )


class RoI2Det:
    """Convert RoI to Detection."""

    def __init__(
        self,
        max_per_img: int = 100,
        num_classes: int = 1,
        nms: bool = False,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
    ) -> None:
        """Create an instance of RoI2Det."""
        self.nms = nms
        self.num_classes = num_classes
        self.max_per_img = max_per_img
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        output_3d: Tensor,
        scale: Tensor,
        intrinsics: Tensor,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        pad_info: dict | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score: Classification scores
            bbox_pred: Bounding box predictions (normalized cxcywh format)
            output_3d: 3D output predictions
            scale: Scale factor for depth
            intrinsics: Camera intrinsics
            img_shape: Processed image shape (H, W)
            ori_shape: Original image shape (H, W)
            pad_info: Optional padding info dict from pad_resize_if_necessary.
                If provided, uses padding-based recovery instead of crop-based.
                Contains 'pad_left', 'pad_top', 'scale' keys.
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        cls_score = cls_score.sigmoid()
        scores, indexes = cls_score.view(-1).topk(self.max_per_img)
        det_labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes
        det_bboxes = det_bboxes[bbox_index]
        output_3d = output_3d[bbox_index]

        # Remove low scoring boxes
        if self.score_threshold > 0.0:
            mask = scores > self.score_threshold
            det_bboxes = det_bboxes[mask]
            det_labels = det_labels[mask]
            scores = scores[mask]
            output_3d = output_3d[mask]

        if self.nms:
            keep = nms(det_bboxes, scores, self.iou_threshold)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]
            scores = scores[keep]
            output_3d = output_3d[keep]

        # Decode 3D boxes
        pred_loc, pred_dims, pred_rot = decode_3d_boxes(
            output_3d, scale=scale, intrinsics=intrinsics
        )

        # rescale to original shape
        if pad_info is not None:
            # Padding mode: subtract pad offset and divide by scale
            pad_left = pad_info["pad_left"]
            pad_top = pad_info["pad_top"]
            scale_final = pad_info["scale"]
            det_bboxes = det_bboxes - det_bboxes.new_tensor(
                [pad_left, pad_top, pad_left, pad_top]
            )
            det_bboxes = det_bboxes / scale_final
        else:
            # Crop mode: add back crop offset and divide by scale
            ori_h, ori_w = ori_shape[:2]
            img_h, img_w = img_shape[:2]
            scale_final = max(img_w / ori_w, img_h / ori_h) + 1e-8
            if scale_final >= 1:
                scale_final = 1.0
            scaled_w = int(ori_w * scale_final)
            scaled_h = int(ori_h * scale_final)
            left = (scaled_w - img_w) // 2
            top = (scaled_h - img_h) // 2
            det_bboxes = det_bboxes + det_bboxes.new_tensor(
                [left, top, left, top]
            )
            det_bboxes = det_bboxes / scale_final

        det_bboxes3d = torch.cat([pred_loc, pred_dims, pred_rot], -1)

        return det_bboxes, det_bboxes3d, scores, det_labels


def decode_3d_boxes(
    output_3d: Tensor,
    scale: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    """Decode 3D boxes from the model output."""
    pred_xy_3d = output_3d[:, :2] * scale

    pred_z = torch.exp(output_3d[:, 2:3]) * scale

    pred_loc = torch.cat([pred_xy_3d, pred_z], -1)

    pred_xy = project_points(pred_loc, intrinsics)

    pred_dims = torch.exp(output_3d[:, 3:6]) * scale

    pose = rotation_6d_to_matrix(output_3d[:, 6:])

    # Allocentric to egocentric
    K_per_box = intrinsics.repeat(len(output_3d), 1, 1)

    pred_rot = matrix_to_quaternion(
        R_from_allocentric(
            K_per_box,
            pose,
            u=pred_xy[:, 0].detach(),
            v=pred_xy[:, 1].detach(),
        )
    )

    return pred_loc, pred_dims, pred_rot
