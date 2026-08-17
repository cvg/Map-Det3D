"""Per-Scene 3D Object Detection Evaluation."""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from shapely.geometry import MultiPoint
from torch import Tensor

from mapdet3d.common.distributed import all_gather_object_cpu
from mapdet3d.common.typing import ArrayLike, GenericFunc, MetricLogs
from mapdet3d.data.const import AxisMode
from mapdet3d.eval.base import Evaluator
from mapdet3d.op.box3d import boxes3d_to_corners


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Detach a tensor or pass an array-like through to numpy."""
    if isinstance(data, Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def obb_to_aabb_corners(obb_data):
    """Convert OBB data to AABB corner coordinates.

    Args:
        obb_data (np.ndarray): Array of OBB corners, shape [N,8,3].

    Returns:
        np.ndarray: Array of AABB corners, shape [N,8,3].
    """
    # 1. Compute min and max along each axis (X, Y, Z) for every OBB [N, 3]
    min_vals = np.min(obb_data, axis=1)  # Shape: [N, 3]
    max_vals = np.max(obb_data, axis=1)  # Shape: [N, 3]

    # 2. Allocate output array for AABB corners [N, 8, 3]
    corners = np.zeros_like(obb_data)

    for i in range(len(obb_data)):
        # Extract min and max coordinates for current box
        x_min, y_min, z_min = min_vals[i]
        x_max, y_max, z_max = max_vals[i]

        # Generate 8 corners for the AABB in fixed order
        corners[i] = np.array(
            [
                [x_min, y_min, z_min],  # 0: front-left-bottom
                [x_max, y_min, z_min],  # 1: front-right-bottom
                [x_max, y_max, z_min],  # 2: back-right-bottom
                [x_min, y_max, z_min],  # 3: back-left-bottom
                [x_min, y_min, z_max],  # 4: front-left-top
                [x_max, y_min, z_max],  # 5: front-right-top
                [x_max, y_max, z_max],  # 6: back-right-top
                [x_min, y_max, z_max],  # 7: back-left-top
            ]
        )

    return corners


def convex_hull_intersection_area(points1, points2):
    """Calculate the intersection area of two convex hulls.

    The convex hull of each point set is taken explicitly rather than treating
    the points as an already-ordered ring. That makes the result independent of
    vertex ordering: an unordered set read as a ring traces a self-intersecting
    polygon, which shapely either measures as near-zero area or rejects with a
    TopologyException.

    Args:
        points1 (list or np.ndarray): (N, 2) points of the first hull.
        points2 (list or np.ndarray): (M, 2) points of the second hull.

    Returns:
        tuple:
            - intersection_area (float): The area of the intersection between the two convex hulls.
            - area1 (float): Area of the first convex hull.
            - area2 (float): Area of the second convex hull.
    """
    # Create convex hull polygons from the input point sets
    poly1 = MultiPoint(np.asarray(points1)).convex_hull
    poly2 = MultiPoint(np.asarray(points2)).convex_hull

    # Calculate the intersection polygon between the two convex polygons
    intersection = poly1.intersection(poly2)

    # Return the area of the intersection and the areas of the original two convex hulls
    return intersection.area, poly1.area, poly2.area


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU for Z-up boxes.

    Up axis is +Z, so the volume factorises into a footprint area in the X-Y
    plane times the overlap of the two height ranges. That holds for any
    upright box -- axis aligned or yaw rotated -- but not for boxes with roll
    or pitch, which need a real convex-polyhedron intersection.

    Corner order does not matter: the footprint is the convex hull of the X-Y
    projection of all eight corners, and the height range is a min/max over all
    of them. (Reading a fixed subset such as ``corners[:4]`` as a ring ties the
    result to one vertex convention -- with ROS-ordered corners those four are
    the y-min side face, giving zero area, and with some orderings shapely
    raises outright.)

    Input:
        corners1: numpy array (8,3)
        corners2: numpy array (8,3)
    Output:
        iou:    3D bounding box IoU
        iou_2d: bird's eye view (X-Y) 2D IoU
    """
    # Footprint = convex hull of every corner projected onto the X-Y plane.
    rect1 = corners1[:, :2]
    rect2 = corners2[:, :2]
    inter_area, area1, area2 = convex_hull_intersection_area(rect1, rect2)

    union_area = area1 + area2 - inter_area
    iou_2d = inter_area / union_area if union_area > 0 else 0.0

    # Vertical (Z) overlap between the two height ranges.
    zmax = min(np.max(corners1[:, 2]), np.max(corners2[:, 2]))
    zmin = max(np.min(corners1[:, 2]), np.min(corners2[:, 2]))
    inter_h = max(0.0, zmax - zmin)

    inter_vol = inter_area * inter_h
    vol1 = area1 * (np.max(corners1[:, 2]) - np.min(corners1[:, 2]))
    vol2 = area2 * (np.max(corners2[:, 2]) - np.min(corners2[:, 2]))
    union_vol = vol1 + vol2 - inter_vol
    iou = inter_vol / union_vol if union_vol > 0 else 0.0

    return iou, iou_2d


def eval_det_cls(
    pred,
    gt,
    scores,
    ovthresh: float = 0.25,
    use_07_metric: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute precision/recall for object detection for a single class.

    Input:
        pred: map of {img_id: [bbox]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        scores: map of {img_id: [score]} used to rank the detections.
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """
    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "det": det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        img_scores = scores[img_id]
        for j, box in enumerate(pred[img_id]):
            image_ids.append(img_id)
            confidence.append(float(img_scores[j]))
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou, _ = box3d_iou(bb, BBGT[j, ...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos + 1e-6)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_ap(
    rec: np.ndarray, prec: np.ndarray, use_07_metric: bool = False
) -> float:
    """Compute VOC AP given precision and recall.

    If use_07_metric is true, uses the VOC 07 11 point method.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


class Detect3DSceneEvaluator(Evaluator):
    """Per-scene 3D object detection evaluation.

    Predictions and ground truth are world-frame boxes, given either as
    [N, 10] parameters or [N, 8, 3] corners (see :func:`to_world_corners`).
    Both are scored in ScanNet's axis-aligned frame; see the module docstring
    for why.
    """

    def __init__(
        self, data_root: str = "data/scannet", threshold: float = 0.3
    ) -> None:
        """Create an instance of the class.

        Args:
            data_root: Path to the data root directory.
            threshold: Minimum size threshold for filtering small boxes.
        """
        self.data_root = data_root
        self.threshold = threshold

        self.detections: dict[str, ArrayLike] = {}
        self.detections_gt: dict[str, ArrayLike] = {}
        self.scores: dict[str, ArrayLike] = {}

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return "Per-scene 3D Object Detection Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return ["3D"]

    def gather(self, gather_func: GenericFunc = all_gather_object_cpu) -> None:
        """Accumulate predictions across processes."""
        for name in ("detections", "detections_gt", "scores"):
            gathered = gather_func(getattr(self, name))

            if gathered is not None:
                merged: dict[str, ArrayLike] = {}
                for part in gathered:
                    merged.update(part)

                setattr(self, name, merged)

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self.detections.clear()
        self.detections_gt.clear()
        self.scores.clear()

    def process_batch(
        self,
        seq_names: list[str],
        pred_boxes3d: list[ArrayLike],
        pred_scores: list[ArrayLike],
        gt_boxes3d: list[ArrayLike],
    ) -> None:
        """Accumulate one batch of world-frame boxes, keyed by sequence.

        Each entry may be [N, 10] parameters or [N, 8, 3] corners; the two may
        be mixed freely between predictions and ground truth.
        """
        for i, seq_name in enumerate(seq_names):
            self.detections[seq_name] = pred_boxes3d[i]
            self.scores[seq_name] = pred_scores[i]
            self.detections_gt[seq_name] = gt_boxes3d[i]

    def _get_align_transform(self, seq_name: str) -> np.ndarray:
        """Get the axis-alignment transform for a ScanNet sequence."""
        return np.load(
            os.path.join(
                self.data_root,
                "scannet_instance_data",
                f"{seq_name}_axis_align_matrix.npy",
            )
        ).astype(np.float32)

    def _to_aligned_aabb(
        self, boxes3d: ArrayLike, transform: np.ndarray
    ) -> np.ndarray:
        """Convert world boxes or corners to AABB format in aligned frame."""
        boxes3d_np = _to_numpy(boxes3d)

        # Empty
        if boxes3d_np.size == 0:
            corners = np.zeros((0, 8, 3), dtype=np.float32)

        # Already corners
        elif boxes3d_np.ndim == 3 and boxes3d_np.shape[1:] == (8, 3):
            corners = boxes3d_np.astype(np.float32)

        # Convert from [N, 10] box parameters to corners
        elif boxes3d_np.ndim == 2 and boxes3d_np.shape[1] == 10:
            corners = (
                boxes3d_to_corners(
                    torch.from_numpy(boxes3d_np).float(), AxisMode.ROS
                )
                .numpy()
                .astype(np.float32)
            )
        else:
            raise ValueError(
                "expected [N, 10] world boxes or [N, 8, 3] world corners, "
                f"got shape {boxes3d_np.shape}"
            )

        # Move to axis-aligned frame
        algined_corners = corners @ transform[:3, :3].T + transform[:3, 3]

        return obb_to_aabb_corners(algined_corners)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions."""
        assert metric in self.metrics, f"Unsupported metric: {metric}"

        detects: dict[int, np.ndarray] = {}
        detects_gt: dict[int, np.ndarray] = {}
        detect_scores: dict[int, np.ndarray] = {}

        for i, seq_name in enumerate(self.detections):
            transform = self._get_align_transform(seq_name)
            detects[i] = self._to_aligned_aabb(
                self.detections[seq_name], transform
            )
            detect_scores[i] = _to_numpy(self.scores[seq_name])
            detects_gt[i] = self._to_aligned_aabb(
                self.detections_gt[seq_name], transform
            )

        score_dict: MetricLogs = {}
        rows: list[tuple[str, float, float, float]] = []
        for name, thresh in (("AP15", 0.15), ("AP25", 0.25), ("AP50", 0.50)):
            rec, prec, ap = eval_det_cls(
                detects, detects_gt, detect_scores, ovthresh=thresh
            )
            score_dict[name] = float(ap)

            # rec/prec are cumulative over the detection list, so the last
            # entry is the value over the full set. Both are empty when
            # nothing was predicted. Reported in the log only: unlike AP they
            # are single operating points, not curve summaries.
            rows.append(
                (
                    f"{thresh:.2f}",
                    float(ap),
                    float(rec[-1]) if rec.size else 0.0,
                    float(prec[-1]) if prec.size else 0.0,
                )
            )

        header = f"{'IoU':>6s} {'AP':>8s} {'Recall':>8s} {'Precision':>10s}"
        log_str = "\n" + header + "\n" + "-" * len(header) + "\n"
        for iou_str, ap, recall, precision in rows:
            log_str += (
                f"{iou_str:>6s} {ap:>8.3f} {recall:>8.3f} {precision:>10.3f}\n"
            )

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

        result_file = os.path.join(result_folder, "detections.pkl")

        with open(result_file, mode="wb") as f:
            pickle.dump(self.detections, f)

        score_file = os.path.join(result_folder, "scores.pkl")

        with open(score_file, mode="wb") as f:
            pickle.dump(self.scores, f)
