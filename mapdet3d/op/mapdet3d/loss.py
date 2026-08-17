"""3D Head Loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mapdet3d.common.distributed import reduce_mean
from mapdet3d.data.const import AxisMode
from mapdet3d.op.box2d import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mapdet3d.op.box3d import boxes3d_to_corners
from mapdet3d.op.geometry.projection import unproject_points
from mapdet3d.op.loss.common import l1_loss
from mapdet3d.op.loss.focal_loss import FocalLoss
from mapdet3d.op.loss.iou_loss import GIoULoss
from mapdet3d.op.loss.reducer import SumWeightedLoss
from mapdet3d.op.matcher.cost import BBoxL1Cost, FocalLossCost, IoUCost
from mapdet3d.op.matcher.hungarian import HungarianMatcher
from mapdet3d.op.util import multi_apply

from .head import decode_3d_boxes


class MapDet3DLoss(nn.Module):
    """Map-Det3D loss module."""

    def __init__(self, num_classes: int = 1, sync_cls_avg_factor: bool = True):
        """Init."""
        super().__init__()
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_classes = num_classes

        # Matcher
        self.cls_cost = FocalLossCost(weight=2.0)
        self.reg_cost = BBoxL1Cost(weight=5.0, box_format="xywh")
        self.iou_cost = IoUCost(weight=2.0, iou_mode="giou")

        self.assigner = HungarianMatcher()

        # Losses
        self.loss_cls = FocalLoss(alpha=0.25, gamma=2.0)
        self.bg_cls_weight = 0.0
        self.cls_loss_weight = 1.0

        self.loss_bbox = l1_loss
        self.bbox_loss_weight = 5.0

        self.loss_iou = GIoULoss()
        self.iou_loss_weight = 2.0

    def get_targets(
        self,
        cls_scores_list: list[Tensor],
        bbox_preds_list: list[Tensor],
        bbox3d_preds_list: list[Tensor] | None,
        scales: list[Tensor] | None,
        intrinsics: Tensor,
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        batch_gt_boxes3d: list[Tensor],
    ) -> tuple:
        """Compute regression and classification targets for a batch images."""
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            gt_corners_list,
            boxes3d_weights_list,
            xy_corners_list,
            z_corners_list,
            dim_corners_list,
            rot_corners_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_targets_single,
            cls_scores_list,
            bbox_preds_list,
            bbox3d_preds_list,
            scales,
            intrinsics,
            input_hw,
            batch_gt_boxes,
            batch_gt_boxes_classes,
            batch_gt_boxes3d,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            gt_corners_list,
            boxes3d_weights_list,
            xy_corners_list,
            z_corners_list,
            dim_corners_list,
            rot_corners_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_cost(self, cls_score, bbox_pred, gt_boxes, gt_classes, input_hw):
        """Compute regression and classification cost for one image."""
        if self.cls_cost.weight != 0:
            cls_cost = self.cls_cost(cls_score, gt_classes)
        else:
            cls_cost = 0

        if self.reg_cost.weight != 0:
            reg_cost = self.reg_cost(
                bbox_pred, gt_boxes, input_hw[0], input_hw[1]
            )
        else:
            reg_cost = 0

        if self.iou_cost.weight != 0:
            iou_cost = self.iou_cost(bbox_pred, gt_boxes)
        else:
            iou_cost = 0

        return cls_cost + reg_cost + iou_cost

    def _get_targets_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        bbox3d_pred: Tensor | None,
        scales: Tensor | None,
        input_hw: tuple[int, int],
        intrinsics: Tensor,
        gt_boxes: Tensor,
        gt_classes: Tensor,
        gt_boxes3d: Tensor | None,
    ):
        """Compute regression and classification targets for one image."""
        img_h, img_w = input_hw
        num_bboxes = bbox_pred.size(0)
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(
            0
        )

        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        # assigner and sampler
        with torch.no_grad():
            cost = self._get_cost(
                cls_score, bbox_pred, gt_boxes, gt_classes, input_hw
            )

        assigned_gt_indices = self.assigner(cost, bbox_pred, gt_boxes)

        pos_inds = (
            torch.nonzero(assigned_gt_indices > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assigned_gt_indices == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assigned_gt_indices[pos_inds] - 1
        pos_gt_bboxes = gt_boxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = gt_boxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_classes[pos_assigned_gt_inds]
        label_weights = gt_boxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_boxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_boxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # Decode Positive boxes for 2D-3D box association
        pos_pred_bboxes = bbox_pred[pos_inds]

        # 3D Targets
        pos_gt_boxes3d = gt_boxes3d[pos_assigned_gt_inds.long(), :]
        pos_pred_boxes3d = bbox3d_pred[pos_inds]
        num_total = bbox3d_pred.size(0)
        num_pos = pos_pred_boxes3d.size(0)

        pos_pred_loc, pos_pred_dims, pos_pred_rot = decode_3d_boxes(
            pos_pred_boxes3d, scales, intrinsics
        )

        pos_gt_xy_3d = pos_gt_boxes3d[:, :2]
        pos_gt_z = pos_gt_boxes3d[:, 2:3]
        pos_gt_loc = pos_gt_boxes3d[:, :3]
        pos_gt_dims = pos_gt_boxes3d[:, 3:6]
        pos_gt_rot = pos_gt_boxes3d[:, 6:]

        xy_center = torch.cat([pos_pred_loc[:, :2], pos_gt_z], -1)
        z_center = torch.cat([pos_gt_xy_3d, pos_pred_loc[:, 2:3]], -1)

        # Fuse the 5 disentangled boxes (gt | xy | z | dim | rot) into one
        # [5*num_pos, 10] tensor and run a single boxes3d_to_corners call.
        disentangled = torch.cat(
            [
                torch.cat([pos_gt_loc, pos_gt_dims, pos_gt_rot], -1),
                torch.cat([xy_center, pos_gt_dims, pos_gt_rot], -1),
                torch.cat([z_center, pos_gt_dims, pos_gt_rot], -1),
                torch.cat([pos_gt_loc, pos_pred_dims, pos_gt_rot], -1),
                torch.cat([pos_gt_loc, pos_gt_dims, pos_pred_rot], -1),
            ],
            dim=0,
        )
        all_corners = boxes3d_to_corners(disentangled, AxisMode.OPENCV)
        gt_pos, xy_pos, z_pos, dim_pos, rot_pos = all_corners.split(
            [num_pos] * 5, dim=0
        )

        gt_corners = bbox3d_pred.new_zeros((num_total, 8, 3))
        gt_corners[pos_inds] = gt_pos

        xy_corners = bbox3d_pred.new_zeros((num_total, 8, 3))
        xy_corners[pos_inds] = xy_pos

        z_corners = bbox3d_pred.new_zeros((num_total, 8, 3))
        z_corners[pos_inds] = z_pos

        dim_corners = bbox3d_pred.new_zeros((num_total, 8, 3))
        dim_corners[pos_inds] = dim_pos

        rot_corners = bbox3d_pred.new_zeros((num_total, 8, 3))
        rot_corners[pos_inds] = rot_pos

        # 3D weights
        boxes3d_weights = bbox3d_pred.new_zeros((num_total, 8, 3))
        boxes3d_weights[pos_inds] = 1.0

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            gt_corners,
            boxes3d_weights,
            xy_corners,
            z_corners,
            dim_corners,
            rot_corners,
            pos_inds,
            neg_inds,
        )

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        bbox3d_preds: Tensor | None,
        scales: Tensor | None,
        input_hw: list[tuple[int, int]],
        intrinsics: Tensor,
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        batch_gt_boxes3d: list[Tensor],
    ) -> tuple[Tensor]:
        """Loss function for outputs from a single decoder layer."""
        num_imgs = cls_scores.size(0)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        bbox3d_preds_list = [
            bbox3d_preds[i] if bbox3d_preds is not None else None
            for i in range(num_imgs)
        ]

        scales_list = [
            scales[i] if scales is not None else None for i in range(num_imgs)
        ]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            gt_corners_list,
            boxes3d_weights_list,
            xy_corners_list,
            z_corners_list,
            dim_corners_list,
            rot_corners_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            bbox3d_preds_list,
            scales_list,
            input_hw,
            intrinsics,
            batch_gt_boxes,
            batch_gt_boxes_classes,
            batch_gt_boxes3d,
        )

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.cls_loss_weight * self.loss_cls(
            cls_scores,
            labels,
            reducer=SumWeightedLoss(
                weight=label_weights.view(-1, 1), avg_factor=cls_avg_factor
            ),
        )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_hw, bbox_pred in zip(input_hw, bbox_preds):
            img_h, img_w = img_hw
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression L1 loss
        loss_bbox = self.bbox_loss_weight * self.loss_bbox(
            bbox_preds,
            bbox_targets,
            reducer=SumWeightedLoss(
                weight=bbox_weights, avg_factor=num_total_pos
            ),
        )

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.iou_loss_weight * self.loss_iou(
            bboxes,
            bboxes_gt,
            reducer=SumWeightedLoss(
                weight=bbox_weights.mean(-1), avg_factor=num_total_pos
            ),
        )

        # 3D Loss
        xy_corners = torch.cat(xy_corners_list)
        z_corners = torch.cat(z_corners_list)
        dim_corners = torch.cat(dim_corners_list)
        rot_corners = torch.cat(rot_corners_list)

        num_boxes = xy_corners.shape[0]

        boxes3d_weights = torch.cat(boxes3d_weights_list)

        gt_corners = torch.cat(gt_corners_list)

        # localization Loss
        loss_delta_2d = (
            l1_loss(xy_corners, gt_corners) * boxes3d_weights
        ).view(num_boxes, -1).mean(1).sum() / num_total_pos

        loss_depth = (l1_loss(z_corners, gt_corners) * boxes3d_weights).view(
            num_boxes, -1
        ).mean(1).sum() / num_total_pos

        # Dimension Loss
        loss_dim = (l1_loss(dim_corners, gt_corners) * boxes3d_weights).view(
            num_boxes, -1
        ).mean(1).sum() / num_total_pos

        # Rotation Loss
        loss_rot = (
            chamfer_loss(rot_corners, gt_corners)
            * boxes3d_weights.view(num_boxes, -1).mean(1)
        ).sum() / num_total_pos

        return (
            loss_cls,
            loss_bbox,
            loss_iou,
            loss_delta_2d,
            loss_depth,
            loss_dim,
            loss_rot,
        )

    def forward(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_outputs_3d: list[Tensor],
        scales: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_boxes3d_preds: Tensor | None,
        input_hw: list[tuple[int, int]],
        intrinsics: list[Tensor],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        batch_gt_boxes3d: list[Tensor],
        dn_meta: dict[str, int] | None = None,
    ) -> dict[str, Tensor]:
        loss_dict = dict()

        batch_size = len(input_hw)
        num_views = len(input_hw[0])

        if dn_meta is not None:
            # extract denoising and matching part of outputs
            (
                all_layers_matching_cls_scores,
                all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
            ) = split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta
            )
        else:
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds

        # Transpose and flatten from [batch_size][num_views] to [num_views][batch_size]
        input_hw = [
            input_hw[b][v] for v in range(num_views) for b in range(batch_size)
        ]
        intrinsics = [
            intrinsics[b][v]
            for v in range(num_views)
            for b in range(batch_size)
        ]
        batch_gt_boxes = [
            batch_gt_boxes[b][v]
            for v in range(num_views)
            for b in range(batch_size)
        ]
        batch_gt_boxes_classes = [
            batch_gt_boxes_classes[b][v]
            for v in range(num_views)
            for b in range(batch_size)
        ]
        batch_gt_boxes3d = [
            batch_gt_boxes3d[b][v]
            for v in range(num_views)
            for b in range(batch_size)
        ]

        scales = scales.flatten()

        # DETRHead loss_by_feat
        (
            losses_cls,
            losses_bbox,
            losses_iou,
            losses_delta_2d,
            losses_depth,
            losses_dim,
            losses_rot,
        ) = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_outputs_3d,
            scales=scales,
            input_hw=input_hw,
            intrinsics=intrinsics,
            batch_gt_boxes=batch_gt_boxes,
            batch_gt_boxes_classes=batch_gt_boxes_classes,
            batch_gt_boxes3d=batch_gt_boxes3d,
        )

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_delta_2d"] = losses_delta_2d[-1]
        loss_dict["loss_depth"] = losses_depth[-1]
        loss_dict["loss_dim"] = losses_dim[-1]
        loss_dict["loss_rot"] = losses_rot[-1]

        # loss from other decoder layers
        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1])
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_delta_2d"] = losses_delta_2d[
                num_dec_layer
            ]
            loss_dict[f"d{num_dec_layer}.loss_depth"] = losses_depth[
                num_dec_layer
            ]
            loss_dict[f"d{num_dec_layer}.loss_dim"] = losses_dim[num_dec_layer]
            loss_dict[f"d{num_dec_layer}.loss_rot"] = losses_rot[num_dec_layer]

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            (
                enc_loss_cls,
                enc_losses_bbox,
                enc_losses_iou,
                enc_losses_delta_2d,
                enc_losses_depth,
                enc_losses_dim,
                enc_losses_rot,
            ) = self.loss_by_feat_single(
                enc_cls_scores,
                enc_bbox_preds,
                enc_boxes3d_preds,
                scales=scales,
                input_hw=input_hw,
                intrinsics=intrinsics,
                batch_gt_boxes=batch_gt_boxes,
                batch_gt_boxes_classes=batch_gt_boxes_classes,
                batch_gt_boxes3d=batch_gt_boxes3d,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou
            loss_dict["enc_loss_delta_2d"] = enc_losses_delta_2d
            loss_dict["enc_loss_depth"] = enc_losses_depth
            loss_dict["enc_loss_dim"] = enc_losses_dim
            loss_dict["enc_loss_rot"] = enc_losses_rot

        if dn_meta is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                boxes2d=batch_gt_boxes,
                boxes2d_classes=batch_gt_boxes_classes,
                input_hw=input_hw,
                dn_meta=dn_meta,
            )

            # collate denoising loss
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            loss_dict["dn_loss_iou"] = dn_losses_iou[-1]

            for num_dec_layer, (
                loss_cls_i,
                loss_bbox_i,
                loss_iou_i,
            ) in enumerate(
                zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1]
                )
            ):
                loss_dict[f"d{num_dec_layer}.dn_loss_cls"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.dn_loss_bbox"] = loss_bbox_i
                loss_dict[f"d{num_dec_layer}.dn_loss_iou"] = loss_iou_i

        return loss_dict

    def _get_dn_targets_single(
        self,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
        img_shape: tuple[int, int],
        num_groups: int,
        num_denoising_queries: int,
    ) -> tuple:
        """Get targets in denoising part for one image."""
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device
            )
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = gt_bboxes.new_tensor(
                [], dtype=torch.long
            )

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full(
            (num_denoising_queries,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0

        img_h, img_w = img_shape

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(
            0
        )
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_dn_targets(
        self,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        input_hw: list[tuple[int, int]],
        dn_meta: dict[str, int],
    ) -> tuple:
        """Get targets in denoising part for a batch of images."""
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_dn_targets_single,
            boxes2d,
            boxes2d_classes,
            input_hw,
            num_groups=dn_meta["num_denoising_groups"],
            num_denoising_queries=dn_meta["num_denoising_queries"],
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _loss_dn_single(
        self,
        dn_cls_scores: Tensor,
        dn_bbox_preds: Tensor,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        input_hw: list[tuple[int, int]],
        dn_meta,
    ):
        """Denoising loss for outputs from a single decoder layer."""
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_dn_targets(boxes2d, boxes2d_classes, input_hw, dn_meta)

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.cls_loss_weight * self.loss_cls(
                cls_scores,
                labels,
                reducer=SumWeightedLoss(
                    weight=label_weights.view(-1, 1), avg_factor=cls_avg_factor
                ),
            )
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device
            )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_hw, bbox_pred in zip(input_hw, dn_bbox_preds):
            img_h, img_w = img_hw
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        if bbox_targets.shape[0] == 0:
            loss_bbox = bbox_preds.sum()
            loss_iou = bbox_preds.sum()
            return loss_cls, loss_bbox, loss_iou

        # regression L1 loss
        loss_bbox = self.bbox_loss_weight * self.loss_bbox(
            bbox_preds,
            bbox_targets,
            reducer=SumWeightedLoss(
                weight=bbox_weights, avg_factor=num_total_pos
            ),
        )

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.iou_loss_weight * self.loss_iou(
            bboxes,
            bboxes_gt,
            reducer=SumWeightedLoss(
                weight=bbox_weights.mean(-1), avg_factor=num_total_pos
            ),
        )

        return loss_cls, loss_bbox, loss_iou

    def loss_dn(
        self,
        all_layers_denoising_cls_scores: Tensor,
        all_layers_denoising_bbox_preds: Tensor,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        input_hw: list[tuple[int, int]],
        dn_meta: dict[str, int],
    ):
        """Calculate denoising loss."""
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            boxes2d=boxes2d,
            boxes2d_classes=boxes2d_classes,
            input_hw=input_hw,
            dn_meta=dn_meta,
        )


def chamfer_loss(vals, target):
    """Chamfer loss between two point clouds."""
    B = vals.shape[0]
    xx = vals.view(B, 8, 1, 3)
    yy = target.view(B, 1, 8, 3)
    l1_dist = (xx - yy).abs().sum(-1)
    l1 = l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1)
    return l1


def split_outputs(
    all_layers_cls_scores: Tensor,
    all_layers_bbox_preds: Tensor,
    dn_meta: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split outputs of the denoising part and the matching part.

    For the total outputs of `num_queries_total` length, the former
    `num_denoising_queries` outputs are from denoising queries, and
    the rest `num_matching_queries` ones are from matching queries,
    where `num_queries_total` is the sum of `num_denoising_queries` and
    `num_matching_queries`.

    Args:
        all_layers_cls_scores (Tensor): Classification scores of all
            decoder layers, has shape (num_decoder_layers, bs,
            num_queries_total, cls_out_channels).
        all_layers_bbox_preds (Tensor): Regression outputs of all decoder
            layers. Each is a 4D-tensor with normalized coordinate format
            (cx, cy, w, h) and has shape (num_decoder_layers, bs,
            num_queries_total, 4).
        dn_meta (Dict[str, int]): The dictionary saves information about
            group collation, including 'num_denoising_queries' and
            'num_denoising_groups'.

    Returns:
        Tuple[Tensor]: a tuple containing the following outputs.

        - all_layers_matching_cls_scores (Tensor): Classification scores
            of all decoder layers in matching part, has shape
            (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
        - all_layers_matching_bbox_preds (Tensor): Regression outputs of
            all decoder layers in matching part. Each is a 4D-tensor with
            normalized coordinate format (cx, cy, w, h) and has shape
            (num_decoder_layers, bs, num_matching_queries, 4).
        - all_layers_denoising_cls_scores (Tensor): Classification scores
            of all decoder layers in denoising part, has shape
            (num_decoder_layers, bs, num_denoising_queries,
            cls_out_channels).
        - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
            all decoder layers in denoising part. Each is a 4D-tensor with
            normalized coordinate format (cx, cy, w, h) and has shape
            (num_decoder_layers, bs, num_denoising_queries, 4).
    """
    num_denoising_queries = dn_meta["num_denoising_queries"]

    if dn_meta is not None:
        all_layers_denoising_cls_scores = all_layers_cls_scores[
            :, :, :num_denoising_queries, :
        ]
        all_layers_denoising_bbox_preds = all_layers_bbox_preds[
            :, :, :num_denoising_queries, :
        ]
        all_layers_matching_cls_scores = all_layers_cls_scores[
            :, :, num_denoising_queries:, :
        ]
        all_layers_matching_bbox_preds = all_layers_bbox_preds[
            :, :, num_denoising_queries:, :
        ]
    else:
        all_layers_denoising_cls_scores = None
        all_layers_denoising_bbox_preds = None
        all_layers_matching_cls_scores = all_layers_cls_scores
        all_layers_matching_bbox_preds = all_layers_bbox_preds

    return (
        all_layers_matching_cls_scores,
        all_layers_matching_bbox_preds,
        all_layers_denoising_cls_scores,
        all_layers_denoising_bbox_preds,
    )
