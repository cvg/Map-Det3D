"""Per-frame 3D Object Detection Evaluation."""

from __future__ import annotations

import itertools
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from mapdet3d.common.distributed import all_gather_object_cpu
from mapdet3d.common.typing import (
    ArrayLike,
    GenericFunc,
    MetricLogs,
)
from mapdet3d.eval.base import Evaluator
from mapdet3d.op.box3d import box3d_overlap, boxes3d_to_corners


@dataclass
class DetectionResult:
    """Single detection result with matching info per depth range.

    Matching depends on the depth range being evaluated, because a range makes
    every GT outside it "ignored" and a detection prefers a GT that counts.
    Both flag arrays are therefore (num_depth_ranges, num_thresholds), with
    row 0 the catch-all range.
    """

    # Identifies the frame this detection came from. Must be unique across the
    # whole evaluation run, sequences included: AR groups detections by
    # (image_id, class_id) and keeps only the top max_dets of each group, so a
    # reused ID silently merges frames and caps recall.
    image_id: int
    score: float
    class_id: int
    depth: float  # Depth (z coordinate) of the detection
    # True where the detection matched a GT that counts for that range
    tp_flags: np.ndarray
    # True where the detection is neither a TP nor a FP for that range
    ignore_flags: np.ndarray


@dataclass
class DetectionResult2D:
    """Single 2D detection result with matching info per area range.

    Flag arrays are (num_area_ranges, num_thresholds); see
    :class:`DetectionResult`.
    """

    # Unique per frame across the run; see :class:`DetectionResult`.
    image_id: int
    score: float
    class_id: int
    area: float  # Box area in pixels² (w * h)
    # True where the detection matched a GT that counts for that range
    tp_flags: np.ndarray
    # True where the detection is neither a TP nor a FP for that range
    ignore_flags: np.ndarray


class Detect3DFrameEvaluator(Evaluator):
    """Per-frame 3D object detection evaluator."""

    def __init__(
        self, depth_ranges: list[tuple[float, float]] | None = None
    ) -> None:
        """Create an instance of the class.

        Args:
            depth_ranges: The (min, max) depths in meters defining the near,
                medium and far breakdown. The default is tuned for indoor
                data, where GT depth spans roughly 0.3-10 m; pass
                driving-scale bounds such as [(0, 10), (10, 35), (35, 1e5)]
                for outdoor datasets.
        """
        self.max_dets = [1, 10, 100]  # Max detections for AR computation
        self.rec_thresholds = np.linspace(0.0, 1.0, 101)

        # 3D evaluation parameters
        self.iou_thresholds_3d = np.linspace(0.05, 0.5, 10).tolist()
        self.num_thresholds_3d = len(self.iou_thresholds_3d)

        # Depth ranges for 3D evaluation. The leading entry is the catch-all
        # range reported as "all"; the rest are near, medium and far.
        near_medium_far = depth_ranges or [(0.0, 2.0), (2.0, 4.0), (4.0, 1e5)]
        assert (
            len(near_medium_far) == 3
        ), "depth_ranges must hold the near, medium and far ranges"
        self.depth_ranges = [[0.0, 1e5]] + [list(r) for r in near_medium_far]
        self.depth_range_labels = ["all", "near", "medium", "far"]

        # 3D detection storage
        self.detections: list[DetectionResult] = []
        self.gt_counts: dict[int, int] = {}  # class_id -> count
        self.gt_counts_per_depth: dict[tuple[int, int], int] = (
            {}
        )  # (class_id, depth_range_idx) -> count
        self.gt_depths: list[tuple[int, float]] = (
            []
        )  # List of (class_id, depth) for all GTs

        # 2D evaluation parameters (matching COCO)
        self.iou_thresholds_2d = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        ).tolist()
        self.num_thresholds_2d = len(self.iou_thresholds_2d)

        # Area ranges for 2D evaluation, matching COCO. The leading entry is
        # the catch-all range reported as "all".
        self.area_ranges = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_range_labels = ["all", "small", "medium", "large"]

        # 2D detection storage
        self.detections_2d: list[DetectionResult2D] = []
        self.gt_counts_2d: dict[int, int] = {}  # class_id -> count
        self.gt_counts_per_area: dict[tuple[int, int], int] = (
            {}
        )  # (class_id, area_range_idx) -> count
        self.gt_areas: list[tuple[int, float]] = (
            []
        )  # List of (class_id, area) for all GTs

        # Guards the uniqueness that AR grouping depends on. Per rank, which is
        # enough: the inference sampler gives each rank a disjoint set of
        # sequences rather than padding them to equal length.
        self._seen_image_ids: set[int] = set()

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return "3D Object Detection Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return ["2D", "3D"]

    def gather(self, gather_func: GenericFunc = all_gather_object_cpu) -> None:
        """Accumulate predictions across processes."""
        # Gather 3D detections
        all_detections = gather_func(self.detections, use_system_tmp=False)
        if all_detections is not None:
            self.detections = list(itertools.chain(*all_detections))

        all_gt_counts = gather_func(self.gt_counts, use_system_tmp=False)
        if all_gt_counts is not None:
            merged_gt_counts: dict[int, int] = {}
            for gt_counts in all_gt_counts:
                for cls_id, count in gt_counts.items():
                    merged_gt_counts[cls_id] = (
                        merged_gt_counts.get(cls_id, 0) + count
                    )
            self.gt_counts = merged_gt_counts

        all_gt_counts_per_depth = gather_func(
            self.gt_counts_per_depth, use_system_tmp=False
        )
        if all_gt_counts_per_depth is not None:
            merged_gt_counts_per_depth: dict[tuple[int, int], int] = {}
            for gt_counts_per_depth in all_gt_counts_per_depth:
                for key, count in gt_counts_per_depth.items():
                    merged_gt_counts_per_depth[key] = (
                        merged_gt_counts_per_depth.get(key, 0) + count
                    )
            self.gt_counts_per_depth = merged_gt_counts_per_depth

        all_gt_depths = gather_func(self.gt_depths, use_system_tmp=False)
        if all_gt_depths is not None:
            self.gt_depths = list(itertools.chain(*all_gt_depths))

        # Gather 2D detections
        all_detections_2d = gather_func(
            self.detections_2d, use_system_tmp=False
        )
        if all_detections_2d is not None:
            self.detections_2d = list(itertools.chain(*all_detections_2d))

        all_gt_counts_2d = gather_func(self.gt_counts_2d, use_system_tmp=False)
        if all_gt_counts_2d is not None:
            merged_gt_counts_2d: dict[int, int] = {}
            for gt_counts_2d in all_gt_counts_2d:
                for cls_id, count in gt_counts_2d.items():
                    merged_gt_counts_2d[cls_id] = (
                        merged_gt_counts_2d.get(cls_id, 0) + count
                    )
            self.gt_counts_2d = merged_gt_counts_2d

        all_gt_counts_per_area = gather_func(
            self.gt_counts_per_area, use_system_tmp=False
        )
        if all_gt_counts_per_area is not None:
            merged_gt_counts_per_area: dict[tuple[int, int], int] = {}
            for gt_counts_per_area in all_gt_counts_per_area:
                for key, count in gt_counts_per_area.items():
                    merged_gt_counts_per_area[key] = (
                        merged_gt_counts_per_area.get(key, 0) + count
                    )
            self.gt_counts_per_area = merged_gt_counts_per_area

        all_gt_areas = gather_func(self.gt_areas, use_system_tmp=False)
        if all_gt_areas is not None:
            self.gt_areas = list(itertools.chain(*all_gt_areas))

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        # Reset 3D data
        self.detections.clear()
        self.gt_counts.clear()
        self.gt_counts_per_depth.clear()
        self.gt_depths.clear()
        # Reset 2D data
        self.detections_2d.clear()
        self.gt_counts_2d.clear()
        self.gt_counts_per_area.clear()
        self.gt_areas.clear()
        self._seen_image_ids.clear()

    @staticmethod
    def _check_lengths(name: str, **arrays: ArrayLike) -> None:
        """Assert that every given array holds the same number of entries.

        Mismatched inputs otherwise surface as an IndexError from inside the
        matching loops, far from the caller that got it wrong.
        """
        lengths = {key: len(value) for key, value in arrays.items()}
        assert (
            len(set(lengths.values())) == 1
        ), f"{name}: mismatched lengths {lengths}"

    def process_batch(
        self,
        coco_image_id: list[int],
        pred_scores: list[ArrayLike],
        pred_classes: list[ArrayLike],
        pred_boxes: list[ArrayLike] | None = None,
        gt_boxes: list[ArrayLike] | None = None,
        pred_boxes3d: list[ArrayLike] | None = None,
        gt_boxes3d: list[ArrayLike] | None = None,
        gt_classes: list[ArrayLike] | None = None,
    ) -> None:
        """Process sample and convert detections to coco format.

        ``gt_classes`` is optional in the signature only because the 2D and 3D
        boxes are: it is required as soon as either is given. It is shared by
        both paths, so it must be as long as whichever GT boxes are passed.
        """
        for i, image_id in enumerate(coco_image_id):
            assert image_id not in self._seen_image_ids, (
                f"duplicate image_id {image_id!r}: IDs must be unique across "
                "sequences, not only within one, because AR groups detections "
                "by (image_id, class_id) and keeps only the top max_dets of "
                "each group"
            )
            self._seen_image_ids.add(image_id)

            # Process 3D detections
            if gt_boxes3d is not None and pred_boxes3d is not None:
                assert (
                    gt_classes is not None
                ), "gt_classes is required to evaluate 3D detections"
                self._check_lengths(
                    "3D predictions",
                    boxes3d=pred_boxes3d[i],
                    scores=pred_scores[i],
                    classes=pred_classes[i],
                )
                self._check_lengths(
                    "3D ground truth",
                    boxes3d=gt_boxes3d[i],
                    classes=gt_classes[i],
                )

                detections, _, gt_depths = self._match_frame_3d(
                    image_id,
                    pred_boxes3d[i],
                    pred_scores[i],
                    pred_classes[i],
                    gt_boxes3d[i],
                    gt_classes[i],
                )

                self.detections.extend(detections)
                self.gt_depths.extend(gt_depths)

                # Count GTs per class and per depth range. Range 0 is skipped:
                # the catch-all denominator is gt_counts, which also covers
                # any GT falling outside every range.
                for cls_id, gt_depth in gt_depths:
                    self.gt_counts[cls_id] = self.gt_counts.get(cls_id, 0) + 1
                    # Count per depth range
                    for range_idx, (d_min, d_max) in enumerate(
                        self.depth_ranges
                    ):
                        if range_idx > 0 and d_min <= gt_depth < d_max:
                            key = (cls_id, range_idx)
                            self.gt_counts_per_depth[key] = (
                                self.gt_counts_per_depth.get(key, 0) + 1
                            )

            # Process 2D detections
            if gt_boxes is not None and pred_boxes is not None:
                assert (
                    gt_classes is not None
                ), "gt_classes is required to evaluate 2D detections"
                self._check_lengths(
                    "2D predictions",
                    boxes=pred_boxes[i],
                    scores=pred_scores[i],
                    classes=pred_classes[i],
                )
                self._check_lengths(
                    "2D ground truth",
                    boxes=gt_boxes[i],
                    classes=gt_classes[i],
                )

                detections_2d, _, gt_areas = self._match_frame_2d(
                    image_id,
                    pred_boxes[i],
                    pred_scores[i],
                    pred_classes[i],
                    gt_boxes[i],
                    gt_classes[i],
                )

                self.detections_2d.extend(detections_2d)
                self.gt_areas.extend(gt_areas)

                # Count GTs per class and per area range. Range 0 is skipped;
                # see the depth-range counting above.
                for cls_id, gt_area in gt_areas:
                    self.gt_counts_2d[cls_id] = (
                        self.gt_counts_2d.get(cls_id, 0) + 1
                    )
                    # Count per area range
                    for range_idx, (a_min, a_max) in enumerate(
                        self.area_ranges
                    ):
                        if range_idx > 0 and a_min <= gt_area < a_max:
                            key = (cls_id, range_idx)
                            self.gt_counts_per_area[key] = (
                                self.gt_counts_per_area.get(key, 0) + 1
                            )

    def _match_frame_3d(
        self,
        image_id: int,
        pred_boxes3d: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_classes: torch.Tensor,
        gt_boxes3d: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> tuple[list[DetectionResult], int, list[tuple[int, float]]]:
        """Match predictions to GTs for a single frame (3D evaluation).

        Returns:
            detections: List of detection results
            M: Number of ground truths
            gt_depths: List of (class_id, depth) for each GT
        """
        N = len(pred_boxes3d)
        M = len(gt_boxes3d)

        # Read the GT scalars once. Reading them element by element inside the
        # matching loops costs one host-device sync each, which dominates the
        # runtime when the tensors live on GPU.
        gt_cls_list = gt_classes.tolist() if M > 0 else []
        gt_depth_list = gt_boxes3d[:, 2].tolist() if M > 0 else []

        # GT depths for depth-range evaluation
        gt_depths = list(zip(gt_cls_list, gt_depth_list))

        # Handle empty predictions
        if N == 0:
            return [], M, gt_depths

        # Sort predictions by score (descending) and keep the best max_dets of
        # the frame. This cap is deliberately class-agnostic: COCO applies
        # maxDets per (image, category), which is equivalent here because every
        # dataset gives its boxes a single shared class. Restore the per-class
        # cap before evaluating genuinely multi-class predictions, or a frame
        # that fills the budget with one class will starve the others.
        max_det = self.max_dets[-1]  # Use largest max_dets for matching
        score_order = torch.argsort(pred_scores, descending=True)
        if N > max_det:
            score_order = score_order[:max_det]
            N = max_det

        pred_boxes3d = pred_boxes3d[score_order]
        pred_scores = pred_scores[score_order]
        pred_classes = pred_classes[score_order]

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix_3d(pred_boxes3d, gt_boxes3d)

        pred_score_list = pred_scores.tolist()
        pred_class_list = pred_classes.tolist()
        pred_depth_list = pred_boxes3d[:, 2].tolist()

        tp_flags, ignore_flags = self._match_ranges(
            iou_matrix,
            self.iou_thresholds_3d,
            self.depth_ranges,
            pred_class_list,
            gt_cls_list,
            pred_depth_list,
            gt_depth_list,
        )

        detections = [
            DetectionResult(
                image_id=image_id,
                score=pred_score_list[pred_idx],
                class_id=pred_class_list[pred_idx],
                depth=pred_depth_list[pred_idx],
                tp_flags=tp_flags[pred_idx].copy(),
                ignore_flags=ignore_flags[pred_idx].copy(),
            )
            for pred_idx in range(N)
        ]

        return detections, M, gt_depths

    def _match_ranges(
        self,
        iou_matrix: np.ndarray,
        iou_thresholds: list[float],
        ranges: list[list[float]],
        pred_class_list: list[int],
        gt_cls_list: list[int],
        pred_values: list[float],
        gt_values: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Greedily match detections to GTs, once per evaluation range.

        Matching cannot be done once and sliced by range afterwards: a range
        makes every GT outside it "ignored", and following COCO an ignored GT
        is only offered to a detection after every GT that counts has been
        tried. A detection that still lands on an ignored GT counts as neither
        TP nor FP, and so does an unmatched detection whose own value falls
        outside the range. Range 0 is the catch-all: it ignores nothing.

        Detections are expected in descending score order.

        Args:
            iou_matrix: (N, M) IoU between detections and GTs.
            iou_thresholds: IoU thresholds to match at.
            ranges: (min, max) per range, the first being the catch-all.
            pred_class_list: Class per detection; a detection only matches a
                GT of the same class.
            gt_cls_list: Class per GT.
            pred_values: Value placing each detection in a range.
            gt_values: Value placing each GT in a range.

        Returns:
            tp_flags: (N, num_ranges, num_thresholds) bool, matched a GT that
                counts for that range.
            ignore_flags: (N, num_ranges, num_thresholds) bool, excluded from
                that range's precision/recall entirely.
        """
        num_preds, num_gts = iou_matrix.shape
        num_ranges = len(ranges)
        num_thresholds = len(iou_thresholds)

        tp_flags = np.zeros(
            (num_preds, num_ranges, num_thresholds), dtype=bool
        )
        ignore_flags = np.zeros(
            (num_preds, num_ranges, num_thresholds), dtype=bool
        )

        for r_idx in range(num_ranges):
            if r_idx == 0:
                # Catch-all range: every GT counts, nothing is ignored.
                gt_ignored = [False] * num_gts
                pred_ignored = [False] * num_preds
                gt_order = list(range(num_gts))
            else:
                v_min, v_max = ranges[r_idx]
                gt_ignored = [not v_min <= v < v_max for v in gt_values]
                pred_ignored = [not v_min <= v < v_max for v in pred_values]
                # Offer the GTs that count first, keeping their relative order.
                gt_order = sorted(range(num_gts), key=lambda g: gt_ignored[g])

            # For each IoU threshold, track which GTs are matched
            gt_matched = np.zeros((num_thresholds, num_gts), dtype=bool)

            for pred_idx in range(num_preds):
                class_id = pred_class_list[pred_idx]

                # For each threshold, find best matching GT
                for t_idx, threshold in enumerate(iou_thresholds):
                    best_gt_idx = -1
                    best_iou = threshold  # Must exceed threshold

                    for gt_idx in gt_order:
                        # Skip already matched GTs at this threshold
                        if gt_matched[t_idx, gt_idx]:
                            continue

                        # Class must match. Scoring buckets detections by exact
                        # class, so there is no class-agnostic shortcut here:
                        # evaluate class-agnostically by giving predictions and
                        # GTs a single shared class instead.
                        if gt_cls_list[gt_idx] != class_id:
                            continue

                        # A GT that counts always beats an ignored one, and
                        # the ignored ones come last, so stop here.
                        if (
                            best_gt_idx >= 0
                            and not gt_ignored[best_gt_idx]
                            and gt_ignored[gt_idx]
                        ):
                            break

                        # IoU-based: higher is better
                        if iou_matrix[pred_idx, gt_idx] >= best_iou:
                            best_iou = iou_matrix[pred_idx, gt_idx]
                            best_gt_idx = gt_idx

                    if best_gt_idx >= 0:
                        gt_matched[t_idx, best_gt_idx] = True
                        if gt_ignored[best_gt_idx]:
                            ignore_flags[pred_idx, r_idx, t_idx] = True
                        else:
                            tp_flags[pred_idx, r_idx, t_idx] = True
                    elif pred_ignored[pred_idx]:
                        # Unmatched, and outside the range being scored: this
                        # detection is not evidence either way.
                        ignore_flags[pred_idx, r_idx, t_idx] = True

        return tp_flags, ignore_flags

    def _match_frame_2d(
        self,
        image_id: int,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> tuple[list[DetectionResult2D], int, list[tuple[int, float]]]:
        """Match predictions to GTs for a single frame (2D evaluation).

        Args:
            pred_boxes: (N, 4) predictions in xyxy format
            gt_boxes: (M, 4) ground truths in xyxy format

        Returns:
            detections: List of 2D detection results
            M: Number of ground truths
            gt_areas: List of (class_id, area) for each GT
        """
        N = len(pred_boxes)
        M = len(gt_boxes)

        # Compute GT areas. The scalars are read once rather than per element
        # inside the matching loops, which would sync the device each time.
        gt_cls_list = gt_classes.tolist() if M > 0 else []
        gt_area_list = self._box_areas(gt_boxes).tolist() if M > 0 else []
        gt_areas_list = list(zip(gt_cls_list, gt_area_list))

        # Handle empty predictions
        if N == 0:
            return [], M, gt_areas_list

        # Sort predictions by score (descending) and keep the best max_dets of
        # the frame. This cap is deliberately class-agnostic: COCO applies
        # maxDets per (image, category), which is equivalent here because every
        # dataset gives its boxes a single shared class. Restore the per-class
        # cap before evaluating genuinely multi-class predictions, or a frame
        # that fills the budget with one class will starve the others.
        max_det = self.max_dets[-1]  # Use largest max_dets for matching
        score_order = torch.argsort(pred_scores, descending=True)
        if N > max_det:
            score_order = score_order[:max_det]
            N = max_det

        pred_boxes = pred_boxes[score_order]
        pred_scores = pred_scores[score_order]
        pred_classes = pred_classes[score_order]

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix_2d(pred_boxes, gt_boxes)

        pred_score_list = pred_scores.tolist()
        pred_class_list = pred_classes.tolist()
        pred_area_list = self._box_areas(pred_boxes).tolist()

        tp_flags, ignore_flags = self._match_ranges(
            iou_matrix,
            self.iou_thresholds_2d,
            self.area_ranges,
            pred_class_list,
            gt_cls_list,
            pred_area_list,
            gt_area_list,
        )

        detections = [
            DetectionResult2D(
                image_id=image_id,
                score=pred_score_list[pred_idx],
                class_id=pred_class_list[pred_idx],
                area=pred_area_list[pred_idx],
                tp_flags=tp_flags[pred_idx].copy(),
                ignore_flags=ignore_flags[pred_idx].copy(),
            )
            for pred_idx in range(N)
        ]

        return detections, M, gt_areas_list

    def _compute_iou_matrix_3d(
        self,
        pred_boxes3d: torch.Tensor,
        gt_boxes3d: torch.Tensor,
    ) -> np.ndarray:
        """Compute 3D IoU matrix between predictions and GTs.

        Args:
            pred_boxes3d: (N, 10) predictions [cx, cy, cz, w, l, h, qw, qx, qy, qz]
            gt_boxes3d: (M, 10) ground truths

        Returns:
            iou_matrix: (N, M) IoU matrix
        """
        if len(pred_boxes3d) == 0 or len(gt_boxes3d) == 0:
            return np.zeros((len(pred_boxes3d), len(gt_boxes3d)))

        # Compute 3D box IoU using corners
        pred_corners = boxes3d_to_corners(pred_boxes3d)
        gt_corners = boxes3d_to_corners(gt_boxes3d)
        iou_matrix = box3d_overlap(pred_corners, gt_corners).cpu().numpy()
        return iou_matrix

    def _compute_iou_matrix_2d(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> np.ndarray:
        """Compute 2D IoU matrix between predictions and GTs.

        Args:
            pred_boxes: (N, 4) predictions in xyxy format
            gt_boxes: (M, 4) ground truths in xyxy format

        Returns:
            iou_matrix: (N, M) IoU matrix
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return np.zeros((len(pred_boxes), len(gt_boxes)))

        pred = pred_boxes.detach().double()
        gt = gt_boxes.detach().double()

        # Pairwise intersection: (N, 1, 2) against (1, M, 2)
        top_left = torch.maximum(pred[:, None, :2], gt[None, :, :2])
        bottom_right = torch.minimum(pred[:, None, 2:], gt[None, :, 2:])
        inter_wh = (bottom_right - top_left).clamp_min(0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        pred_area = self._box_areas(pred)
        gt_area = self._box_areas(gt)
        union_area = pred_area[:, None] + gt_area[None, :] - inter_area

        # Degenerate boxes can give a non-positive union; report 0 IoU.
        iou_matrix = torch.where(
            union_area > 0,
            inter_area / union_area,
            torch.zeros_like(inter_area),
        )

        return iou_matrix.cpu().numpy()

    @staticmethod
    def _box_areas(boxes: torch.Tensor) -> torch.Tensor:
        """Compute areas of (N, 4) boxes in xyxy format."""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def _compute_ap_single_class(
        self,
        detections: list[DetectionResult] | list[DetectionResult2D],
        gt_counts: dict[int, int],
        class_id: int,
        gt_counts_per_range: dict[tuple[int, int], int] | None = None,
        range_idx: int = 0,
        mode: str = "3D",
    ) -> tuple[np.ndarray, float]:
        """Compute AP for a single class (COCO-style).

        Args:
            detections: List of detection results
            gt_counts: Dict of class_id -> GT count, used for range 0
            class_id: The class to compute AP for
            gt_counts_per_range: Dict of (class_id, range_idx) -> GT count
            range_idx: Range to score; 0 is the catch-all range
            mode: "3D" for depth-based evaluation, "2D" for area-based evaluation

        Returns:
            ap_per_threshold: AP at each IoU threshold (-1 if category absent)
            mAP: Mean AP across thresholds (-1 if category absent)
        """
        num_thresholds = (
            self.num_thresholds_3d if mode == "3D" else self.num_thresholds_2d
        )

        # Filter detections by class. Range membership is already baked into
        # the per-range flags by _match_ranges.
        detections = [d for d in detections if d.class_id == class_id]

        # Get GT count. Range 0 is the catch-all, which counts every GT
        # including any that fall outside all of the narrower ranges.
        if range_idx == 0:
            total_gt = gt_counts.get(class_id, 0)
        else:
            total_gt = (gt_counts_per_range or {}).get(
                (class_id, range_idx), 0
            )

        # COCO convention: return -1 for absent categories
        if total_gt == 0:
            return np.full(num_thresholds, -1.0), -1.0

        if len(detections) == 0:
            return np.zeros(num_thresholds), 0.0

        # Sort by score descending
        detections = sorted(detections, key=lambda d: d.score, reverse=True)

        # (num_detections, num_thresholds) for the range being scored
        range_tp = np.array([d.tp_flags[range_idx] for d in detections])
        range_ignore = np.array(
            [d.ignore_flags[range_idx] for d in detections]
        )

        # Compute AP at each threshold
        ap_per_threshold = np.zeros(num_thresholds)

        for t_idx in range(num_thresholds):
            # Ignored detections stay in the ranking but contribute to neither
            # sum, exactly as COCO's dtIgnore does.
            keep = ~range_ignore[:, t_idx]
            tp = (range_tp[:, t_idx] & keep).astype(np.float64)
            fp = (~range_tp[:, t_idx] & keep).astype(np.float64)

            nd = len(tp)
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recall = tp_cumsum / total_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))

            # COCO-style: make precision monotonically decreasing (right to left)
            for i in range(nd - 1, 0, -1):
                if precision[i] > precision[i - 1]:
                    precision[i - 1] = precision[i]

            # 101-point interpolation using searchsorted
            inds = np.searchsorted(recall, self.rec_thresholds, side="left")
            q = np.zeros(len(self.rec_thresholds))
            for ri, pi in enumerate(inds):
                if pi < nd:
                    q[ri] = precision[pi]
            ap_per_threshold[t_idx] = np.mean(q)

        mAP = np.mean(ap_per_threshold)

        return ap_per_threshold, mAP

    def _compute_ap(
        self,
        detections: list[DetectionResult] | list[DetectionResult2D],
        gt_counts: dict[int, int],
        gt_counts_per_range: dict[tuple[int, int], int] | None = None,
        range_idx: int = 0,
        mode: str = "3D",
    ) -> tuple[np.ndarray, float, dict[int, float]]:
        """Compute mAP using COCO convention (per-category then average).

        Args:
            detections: List of detection results
            gt_counts: Dict of class_id -> GT count, used for range 0
            gt_counts_per_range: Dict of (class_id, range_idx) -> GT count
            range_idx: Range to score; 0 is the catch-all range
            mode: "3D" for depth-based evaluation, "2D" for area-based evaluation

        Returns:
            ap_per_threshold: Mean AP at each IoU threshold (across valid categories)
            mAP: Mean AP across thresholds and categories
            per_class_aps: Dict of class_id -> mAP for each category
        """
        num_thresholds = (
            self.num_thresholds_3d if mode == "3D" else self.num_thresholds_2d
        )

        # Get all class IDs that have GTs
        class_ids = sorted(gt_counts.keys())

        if len(class_ids) == 0:
            return np.full(num_thresholds, float("nan")), float("nan"), {}

        # Compute AP per class
        per_class_ap_arrays = {}  # class_id -> ap_per_threshold array
        per_class_aps = {}  # class_id -> mAP float

        for class_id in class_ids:
            ap_arr, mAP = self._compute_ap_single_class(
                detections,
                gt_counts,
                class_id,
                gt_counts_per_range=gt_counts_per_range,
                range_idx=range_idx,
                mode=mode,
            )
            per_class_ap_arrays[class_id] = ap_arr
            per_class_aps[class_id] = mAP

        # Aggregate across categories (exclude absent categories with -1)
        valid_ap_arrays = [
            arr for arr in per_class_ap_arrays.values() if arr[0] > -1
        ]
        valid_mAPs = [m for m in per_class_aps.values() if m > -1]

        if len(valid_mAPs) == 0:
            return (
                np.full(num_thresholds, float("nan")),
                float("nan"),
                per_class_aps,
            )

        # Mean across valid categories at each threshold
        ap_per_threshold = np.mean(np.stack(valid_ap_arrays, axis=0), axis=0)
        mAP = np.mean(valid_mAPs)

        return ap_per_threshold, mAP, per_class_aps

    def _compute_ar(
        self,
        detections: list[DetectionResult] | list[DetectionResult2D],
        gt_counts: dict[int, int],
        max_dets: int,
        gt_counts_per_range: dict[tuple[int, int], int] | None = None,
        range_idx: int = 0,
        mode: str = "3D",
        iou_threshold: float | None = None,
    ) -> float:
        """Compute Average Recall (AR) at a specific max_dets.

        COCO-style AR: max_dets is applied per-category per-image.
        For each category and image, take top max_dets detections,
        then compute recall.

        Args:
            detections: List of detection results
            gt_counts: Dict of class_id -> GT count, used for range 0
            max_dets: Maximum number of detections per category per image
            gt_counts_per_range: Dict of (class_id, range_idx) -> GT count
            range_idx: Range to score; 0 is the catch-all range
            mode: "3D" for depth-based evaluation, "2D" for area-based evaluation
            iou_threshold: Optional single IoU threshold. If given, AR is
                computed at this threshold only instead of averaging across
                all thresholds.

        Returns:
            AR: Average Recall across IoU thresholds and categories
        """
        iou_thresholds = (
            self.iou_thresholds_3d if mode == "3D" else self.iou_thresholds_2d
        )
        if iou_threshold is not None:
            t_indices = [
                int(
                    np.argmin(np.abs(np.array(iou_thresholds) - iou_threshold))
                )
            ]
        else:
            t_indices = list(range(len(iou_thresholds)))

        class_ids = sorted(gt_counts.keys())
        if len(class_ids) == 0:
            return float("nan")

        # Group detections by (image_id, class_id)
        dets_by_img_cls: dict[
            tuple[int, int], list[DetectionResult] | list[DetectionResult2D]
        ] = {}
        for d in detections:
            key = (d.image_id, d.class_id)
            if key not in dets_by_img_cls:
                dets_by_img_cls[key] = []
            dets_by_img_cls[key].append(d)

        recalls_per_class = []
        for class_id in class_ids:
            # Get GT count. Range 0 is the catch-all; see
            # _compute_ap_single_class.
            if range_idx == 0:
                total_gt = gt_counts.get(class_id, 0)
            else:
                total_gt = (gt_counts_per_range or {}).get(
                    (class_id, range_idx), 0
                )

            if total_gt == 0:
                continue  # Skip absent categories

            # Collect top max_dets detections per image for this class
            class_dets = []
            for (img_id, cls_id), img_cls_dets in dets_by_img_cls.items():
                if cls_id != class_id:
                    continue
                # Sort by score and take top max_dets per image per category
                img_cls_dets = sorted(
                    img_cls_dets, key=lambda d: d.score, reverse=True
                )
                class_dets.extend(img_cls_dets[:max_dets])

            if len(class_dets) == 0:
                recalls_per_class.append(0.0)
                continue

            # Compute recall at each IoU threshold. tp_flags already holds only
            # the matches that count for this range.
            recalls_per_thresh = []
            for t_idx in t_indices:
                tp_count = sum(
                    1 for d in class_dets if d.tp_flags[range_idx, t_idx]
                )
                recall = tp_count / total_gt
                recalls_per_thresh.append(recall)

            # Average recall across thresholds for this class
            recalls_per_class.append(np.mean(recalls_per_thresh))

        if len(recalls_per_class) == 0:
            return float("nan")

        return float(np.mean(recalls_per_class))

    @staticmethod
    def _format_per_class_aps(
        per_class_aps: dict[int, float], mode: str
    ) -> str:
        """Render the per-category AP breakdown for the log.

        Only useful once there is more than one category; with a single class
        it just repeats the headline AP, so it is omitted. It stays out of the
        returned metrics so that a many-class run does not flood the logger.
        """
        if len(per_class_aps) < 2:
            return ""

        lines = f"mode={mode}  Average Precision (AP) per category\n"
        for class_id, class_ap in sorted(per_class_aps.items()):
            # COCO convention: -1 marks a category with no GT
            value = "     n/a" if class_ap < 0 else f"{class_ap:8.3f}"
            lines += f"mode={mode}    class {class_id:>4d} = {value}\n"
        return lines

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions."""
        if metric == "2D":
            return self._evaluate_2d()
        else:  # 3D
            return self._evaluate_3d()

    def _evaluate_3d(self) -> tuple[MetricLogs, str]:
        """Evaluate 3D predictions."""
        # Compute overall mAP (depth_range="all")
        ap_arr, mAP, per_class_aps = self._compute_ap(
            self.detections, self.gt_counts, mode="3D"
        )

        score_dict: MetricLogs = {"AP": mAP}

        # Add AP at specific thresholds (0.15, 0.25, 0.5)
        for thresh in [0.15, 0.25, 0.5]:
            idx = np.argmin(np.abs(np.array(self.iou_thresholds_3d) - thresh))
            score_dict[f"AP{int(thresh*100)}"] = float(ap_arr[idx])

        # Compute mAP for each depth range (near, medium, far)
        depth_range_aps = {}
        for range_idx, label in enumerate(self.depth_range_labels):
            if label == "all":
                depth_range_aps[label] = mAP
                continue

            _, range_mAP, _ = self._compute_ap(
                self.detections,
                self.gt_counts,
                gt_counts_per_range=self.gt_counts_per_depth,
                range_idx=range_idx,
                mode="3D",
            )
            depth_range_aps[label] = range_mAP

        # Add depth range metrics to score_dict (APn=near, APm=medium, APf=far)
        score_dict["APn"] = float(depth_range_aps.get("near", float("nan")))
        score_dict["APm"] = float(depth_range_aps.get("medium", float("nan")))
        score_dict["APf"] = float(depth_range_aps.get("far", float("nan")))

        # Overall AR, averaged across IoU thresholds like the headline AP.
        # Only max_dets=100 is reported, so only that one is computed.
        score_dict["AR"] = float(
            self._compute_ar(
                self.detections, self.gt_counts, self.max_dets[-1], mode="3D"
            )
        )

        # AR at specific IoU thresholds (0.15, 0.25, 0.5)
        for thresh in [0.15, 0.25, 0.5]:
            ar_t = self._compute_ar(
                self.detections,
                self.gt_counts,
                max_dets=100,
                mode="3D",
                iou_threshold=thresh,
            )
            score_dict[f"AR{int(thresh * 100)}"] = float(ar_t)

        # Compute AR at different depth ranges (using max_dets=100)
        ar_depth_results = {}
        for range_idx, label in enumerate(self.depth_range_labels):
            if label == "all":
                ar_depth_results[label] = score_dict["AR"]
                continue

            ar_range = self._compute_ar(
                self.detections,
                self.gt_counts,
                max_dets=100,
                gt_counts_per_range=self.gt_counts_per_depth,
                range_idx=range_idx,
                mode="3D",
            )
            ar_depth_results[label] = ar_range

        score_dict["ARn"] = float(ar_depth_results.get("near", float("nan")))
        score_dict["ARm"] = float(ar_depth_results.get("medium", float("nan")))
        score_dict["ARf"] = float(ar_depth_results.get("far", float("nan")))

        # Build log string in COCO-style format
        log_str = "\n"

        def _format_line(
            ap: bool,
            iou_str: str,
            range_label: str,
            max_dets: int,
            value: float,
        ) -> str:
            title = "Average Precision" if ap else "Average Recall"
            type_str = "(AP)" if ap else "(AR)"
            return (
                f"mode=3D  {title:18s} {type_str} @[ IoU={iou_str:9s} | "
                f"depth={range_label:>6s} | maxDets={max_dets:>3d} ] = {value:.3f}\n"
            )

        iou_range_str = (
            f"{self.iou_thresholds_3d[0]:.2f}:{self.iou_thresholds_3d[-1]:.2f}"
        )

        # AP metrics
        log_str += _format_line(True, iou_range_str, "all", 100, mAP)
        log_str += _format_line(True, "0.15", "all", 100, score_dict["AP15"])
        log_str += _format_line(True, "0.25", "all", 100, score_dict["AP25"])
        log_str += _format_line(True, "0.50", "all", 100, score_dict["AP50"])
        log_str += _format_line(
            True, iou_range_str, "near", 100, score_dict["APn"]
        )
        log_str += _format_line(
            True, iou_range_str, "medium", 100, score_dict["APm"]
        )
        log_str += _format_line(
            True, iou_range_str, "far", 100, score_dict["APf"]
        )

        # AR metrics
        log_str += _format_line(
            False, iou_range_str, "all", 100, score_dict["AR"]
        )
        log_str += _format_line(False, "0.15", "all", 100, score_dict["AR15"])
        log_str += _format_line(False, "0.25", "all", 100, score_dict["AR25"])
        log_str += _format_line(False, "0.50", "all", 100, score_dict["AR50"])
        log_str += _format_line(
            False, iou_range_str, "near", 100, score_dict["ARn"]
        )
        log_str += _format_line(
            False, iou_range_str, "medium", 100, score_dict["ARm"]
        )
        log_str += _format_line(
            False, iou_range_str, "far", 100, score_dict["ARf"]
        )

        log_str += self._format_per_class_aps(per_class_aps, "3D")

        return score_dict, log_str

    def _evaluate_2d(self) -> tuple[MetricLogs, str]:
        """Evaluate 2D predictions."""
        # Compute overall mAP (area_range="all")
        ap_arr, mAP, per_class_aps = self._compute_ap(
            self.detections_2d, self.gt_counts_2d, mode="2D"
        )

        score_dict: MetricLogs = {"AP": mAP}

        # Add AP at specific thresholds (0.5, 0.75, 0.95)
        for thresh in [0.5, 0.75, 0.95]:
            idx = np.argmin(np.abs(np.array(self.iou_thresholds_2d) - thresh))
            score_dict[f"AP{int(thresh*100)}"] = float(ap_arr[idx])

        # Compute mAP for each area range (small, medium, large)
        area_range_aps = {}
        for range_idx, label in enumerate(self.area_range_labels):
            if label == "all":
                area_range_aps[label] = mAP
                continue

            _, range_mAP, _ = self._compute_ap(
                self.detections_2d,
                self.gt_counts_2d,
                gt_counts_per_range=self.gt_counts_per_area,
                range_idx=range_idx,
                mode="2D",
            )
            area_range_aps[label] = range_mAP

        # Add area range metrics to score_dict (APs=small, APm=medium, APl=large)
        score_dict["APs"] = float(area_range_aps.get("small", float("nan")))
        score_dict["APm"] = float(area_range_aps.get("medium", float("nan")))
        score_dict["APl"] = float(area_range_aps.get("large", float("nan")))

        # Overall AR, averaged across IoU thresholds like the headline AP.
        # Only max_dets=100 is reported, so only that one is computed.
        score_dict["AR"] = float(
            self._compute_ar(
                self.detections_2d,
                self.gt_counts_2d,
                self.max_dets[-1],
                mode="2D",
            )
        )

        # AR at specific IoU thresholds, mirroring the AP thresholds above
        for thresh in [0.5, 0.75, 0.95]:
            ar_t = self._compute_ar(
                self.detections_2d,
                self.gt_counts_2d,
                max_dets=100,
                mode="2D",
                iou_threshold=thresh,
            )
            score_dict[f"AR{int(thresh * 100)}"] = float(ar_t)

        # Compute AR at different area ranges (using max_dets=100)
        ar_area_results = {}
        for range_idx, label in enumerate(self.area_range_labels):
            if label == "all":
                ar_area_results[label] = score_dict["AR"]
                continue

            ar_range = self._compute_ar(
                self.detections_2d,
                self.gt_counts_2d,
                max_dets=100,
                gt_counts_per_range=self.gt_counts_per_area,
                range_idx=range_idx,
                mode="2D",
            )
            ar_area_results[label] = ar_range

        score_dict["ARs"] = float(ar_area_results.get("small", float("nan")))
        score_dict["ARm"] = float(ar_area_results.get("medium", float("nan")))
        score_dict["ARl"] = float(ar_area_results.get("large", float("nan")))

        # Build log string in COCO-style format
        log_str = "\n"

        def _format_line(
            ap: bool,
            iou_str: str,
            range_label: str,
            max_dets: int,
            value: float,
        ) -> str:
            title = "Average Precision" if ap else "Average Recall"
            type_str = "(AP)" if ap else "(AR)"
            return (
                f"mode=2D  {title:18s} {type_str} @[ IoU={iou_str:9s} | "
                f"area={range_label:>6s} | maxDets={max_dets:>3d} ] = {value:.3f}\n"
            )

        iou_range_str = (
            f"{self.iou_thresholds_2d[0]:.2f}:{self.iou_thresholds_2d[-1]:.2f}"
        )

        # AP metrics
        log_str += _format_line(True, iou_range_str, "all", 100, mAP)
        log_str += _format_line(True, "0.50", "all", 100, score_dict["AP50"])
        log_str += _format_line(True, "0.75", "all", 100, score_dict["AP75"])
        log_str += _format_line(True, "0.95", "all", 100, score_dict["AP95"])
        log_str += _format_line(
            True, iou_range_str, "small", 100, score_dict["APs"]
        )
        log_str += _format_line(
            True, iou_range_str, "medium", 100, score_dict["APm"]
        )
        log_str += _format_line(
            True, iou_range_str, "large", 100, score_dict["APl"]
        )

        # AR metrics
        log_str += _format_line(
            False, iou_range_str, "all", 100, score_dict["AR"]
        )
        log_str += _format_line(False, "0.50", "all", 100, score_dict["AR50"])
        log_str += _format_line(False, "0.75", "all", 100, score_dict["AR75"])
        log_str += _format_line(False, "0.95", "all", 100, score_dict["AR95"])
        log_str += _format_line(
            False, iou_range_str, "small", 100, score_dict["ARs"]
        )
        log_str += _format_line(
            False, iou_range_str, "medium", 100, score_dict["ARm"]
        )
        log_str += _format_line(
            False, iou_range_str, "large", 100, score_dict["ARl"]
        )

        log_str += self._format_per_class_aps(per_class_aps, "2D")

        return score_dict, log_str

    def save(
        self, metric: str, output_dir: str, prefix: str | None = None
    ) -> None:
        """Save the results to json files."""
        assert metric in self.metrics

        if prefix is not None:
            result_folder = os.path.join(output_dir, prefix)
            os.makedirs(result_folder, exist_ok=True)
        else:
            result_folder = output_dir

        if metric == "3D":
            with open(
                os.path.join(result_folder, "detections.pkl"), mode="wb"
            ) as f:
                pickle.dump(self.detections, f)
            with open(
                os.path.join(result_folder, "gt_counts.pkl"), mode="wb"
            ) as f:
                pickle.dump(self.gt_counts, f)
            with open(
                os.path.join(result_folder, "gt_counts_per_depth.pkl"),
                mode="wb",
            ) as f:
                pickle.dump(self.gt_counts_per_depth, f)
            with open(
                os.path.join(result_folder, "gt_depths.pkl"), mode="wb"
            ) as f:
                pickle.dump(self.gt_depths, f)
        else:
            with open(
                os.path.join(result_folder, "detections_2d.pkl"), mode="wb"
            ) as f:
                pickle.dump(self.detections_2d, f)
            with open(
                os.path.join(result_folder, "gt_counts_2d.pkl"), mode="wb"
            ) as f:
                pickle.dump(self.gt_counts_2d, f)
            with open(
                os.path.join(result_folder, "gt_counts_per_area.pkl"),
                mode="wb",
            ) as f:
                pickle.dump(self.gt_counts_per_area, f)
            with open(
                os.path.join(result_folder, "gt_areas.pkl"), mode="wb"
            ) as f:
                pickle.dump(self.gt_areas, f)
