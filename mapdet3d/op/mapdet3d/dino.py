"""DINO op."""

from __future__ import annotations

import torch
from fairscale.nn.checkpoint import checkpoint_wrapper
from torch import Tensor, nn

from mapdet3d.op.box2d import bbox_xyxy_to_cxcywh
from mapdet3d.op.layer.mlp import SimpleMLP as MLP
from mapdet3d.op.layer.positional_encoding import coordinate_to_encoding
from mapdet3d.op.layer.transformer import inverse_sigmoid

from .deformable_detr import (
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerDecoderLayer,
)


class DinoTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer decoder of DINO."""

    def __init__(
        self,
        num_layers: int,
        layer: DeformableDetrTransformerDecoderLayer | None = None,
        with_post_norm: bool = False,
        return_intermediate: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        """Create an instance of DinoTransformerDecoder."""
        super().__init__(
            num_layers=num_layers,
            layer=layer,
            with_post_norm=with_post_norm,
            return_intermediate=return_intermediate,
        )

        if use_checkpoint:
            for i in range(self.num_layers):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.ref_point_head = MLP(
            self.embed_dims * 2, self.embed_dims, self.embed_dims, 2
        )
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
    ) -> tuple[Tensor, Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, _ in enumerate(self.layers):
            layer: DeformableDetrTransformerDecoderLayer = self.layers[lid]

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2,
            )
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points_input,
            )

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))

                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
            )

        return query, reference_points

    def __call__(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query,
            value,
            key_padding_mask,
            self_attn_mask,
            reference_points,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            reg_branches,
        )


class CdnQueryGenerator(nn.Module):
    """Query generator for Contrastive Denoising (CDN).

    Adapted from 3D-MOOD: https://github.com/IDEA-Research/DINO
    """

    def __init__(
        self,
        num_classes: int,
        embed_dims: int,
        num_matching_queries: int,
        label_noise_scale: float = 0.5,
        box_noise_scale: float = 1.0,
        dynamic: bool = True,
        num_groups: int | None = None,
        num_dn_queries: int = 100,
    ) -> None:
        """Create an instance of CDN query generator.

        Args:
            num_classes: Number of object classes.
            embed_dims: The embedding dimensions of the generated queries.
            num_matching_queries: The queries number of the matching part.
            label_noise_scale: The scale of label noise, defaults to 0.5.
            box_noise_scale: The scale of box noise, defaults to 1.0.
            dynamic: Use dynamic dn groups if True, defaults to True.
            num_groups: Number of denoising query groups (static mode).
            num_dn_queries: Max number of denoising queries (dynamic mode).
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        self.dynamic_dn_groups = dynamic

        if self.dynamic_dn_groups:
            assert isinstance(num_dn_queries, int)
            self.num_dn_queries = num_dn_queries
        else:
            assert num_groups is not None
            assert isinstance(num_groups, int)
            self.num_groups = num_groups

        self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)

    def forward(
        self,
        boxes: list[Tensor],
        class_ids: list[Tensor],
        input_hw: list[tuple[int, int]],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, int]]:
        """Generate contrastive denoising queries with ground truth.

        Args:
            boxes: List of GT boxes per sample in xyxy format.
            class_ids: List of GT class IDs per sample.
            input_hw: List of (height, width) tuples for each sample.

        Returns:
            dn_label_query: Denoising label queries (bs, num_dn_queries, dim).
            dn_bbox_query: Denoising bbox queries (bs, num_dn_queries, 4).
            attn_mask: Attention mask (num_queries_total, num_queries_total).
            dn_meta: Dict with 'num_denoising_queries' and 'num_denoising_groups'.
        """
        batch_size = len(input_hw)

        # Normalize bbox and collate ground truth
        gt_labels_list = []
        gt_bboxes_list = []
        for i, bboxes in enumerate(boxes):
            img_h, img_w = input_hw[i]
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(
                0
            )
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(class_ids[i])

        gt_labels = torch.cat(gt_labels_list)
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        batch_idx = torch.cat(
            [
                torch.full_like(t.long(), i)
                for i, t in enumerate(gt_labels_list)
            ]
        )
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, batch_size, num_groups
        )

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device
        )

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups,
        )

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def get_num_groups(self, max_num_target: int = None) -> int:
        """Calculate denoising query groups number."""
        if self.dynamic_dn_groups:
            assert max_num_target is not None
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target
        else:
            num_groups = self.num_groups
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_dn_label_query(
        self, gt_labels: Tensor, num_groups: int
    ) -> Tensor:
        """Generate noisy labels and their query embeddings."""
        assert self.label_noise_scale > 0
        gt_labels_expand = gt_labels.repeat(2 * num_groups, 1).view(-1)
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(
            -1
        )
        new_labels = torch.randint_like(chosen_indice, 0, self.num_classes)
        noisy_labels_expand = gt_labels_expand.scatter(
            0, chosen_indice, new_labels
        )
        dn_label_query = self.label_embedding(noisy_labels_expand)
        return dn_label_query

    def generate_dn_bbox_query(
        self, gt_bboxes: Tensor, num_groups: int
    ) -> Tensor:
        """Generate noisy bboxes and their query embeddings."""
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)

        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device
        )
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += (
            2
            * len(gt_bboxes)
            * torch.arange(num_groups, dtype=torch.long, device=device)[
                :, None
            ]
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        rand_sign = (
            torch.randint_like(
                gt_bboxes_expand, low=0, high=2, dtype=torch.float32
            )
            * 2.0
            - 1.0
        )

        rand_part = torch.rand_like(gt_bboxes_expand)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign

        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = (
            gt_bboxes_expand
            + torch.mul(rand_part, bboxes_whwh) * self.box_noise_scale / 2
        )
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query

    def collate_dn_queries(
        self,
        input_label_query: Tensor,
        input_bbox_query: Tensor,
        batch_idx: Tensor,
        batch_size: int,
        num_groups: int,
    ) -> tuple[Tensor, Tensor]:
        """Collate generated queries to obtain batched dn queries."""
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat(
            [
                torch.arange(num_target, device=device)
                for num_target in num_target_list
            ]
        )
        map_query_index = torch.cat(
            [
                map_query_index + max_num_target * i
                for i in range(2 * num_groups)
            ]
        ).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device
        )
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device
        )

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(
        self,
        max_num_target: int,
        num_groups: int,
        device: torch.device | str,
    ) -> Tensor:
        """Generate attention mask to prevent information leakage."""
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool,
        )
        # Matching part cannot see denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Denoising groups cannot see each other
        for i in range(num_groups):
            row_scope = slice(
                max_num_target * 2 * i, max_num_target * 2 * (i + 1)
            )
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(
                max_num_target * 2 * (i + 1), num_denoising_queries
            )
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask
