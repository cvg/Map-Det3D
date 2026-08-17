"""Map-Det3D tracking graph."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from mapdet3d.data.const import AxisMode
from mapdet3d.op.box3d import (
    box3d_overlap,
    boxes3d_to_corners,
    transform_boxes3d,
)

NEW_TRACK = -1
SUPPRESSED = -2


def greedy_assign(
    detection_scores: Tensor,
    tracklet_ids: Tensor,
    affinity_scores: Tensor,
    match_score_thr: float = 0.5,
    obj_score_thr: float = 0.3,
    nms_conf_thr: None | float = None,
) -> Tensor:
    """Greedy assignment of detections to tracks given affinities."""
    ids = torch.full(
        (len(detection_scores),),
        NEW_TRACK,
        dtype=torch.long,
        device=detection_scores.device,
    )

    for i, score in enumerate(detection_scores):
        conf, memo_ind = torch.max(affinity_scores[i, :], dim=0)
        cur_id = tracklet_ids[memo_ind]
        if conf > match_score_thr:
            if cur_id > NEW_TRACK:
                if score > obj_score_thr:
                    ids[i] = cur_id
                    affinity_scores[:i, memo_ind] = 0
                    affinity_scores[(i + 1) :, memo_ind] = 0
                elif nms_conf_thr is not None and conf > nms_conf_thr:
                    ids[i] = SUPPRESSED

    return ids


class Track3DOut(NamedTuple):
    """Tracklets accumulated over a scene.

    Attributes:
        boxes_3d (Tensor): Camera-frame boxes of each tracklet's
            representative observation (N, 10).
        boxes_3d_world (Tensor): World-frame boxes (N, 10).
        track_ids (Tensor): Track ids (N,).
        scores (Tensor): Per-tracklet score, the maximum over its observations
            (N,).
    """

    boxes_3d: Tensor
    boxes_3d_world: Tensor
    track_ids: Tensor
    scores: Tensor


class MapDet3DTrackGraph:
    """Map-Det3D tracking graph.

    Detections are associated to tracklets frame by frame using the 3D IoU of
    their world-coordinate boxes, and every tracklet ever created is
    kept: the output is the whole scene's set of objects, not the objects
    visible in the current frame.
    """

    def __init__(
        self,
        min_size: float = 0.3,
        match_score_thr: float = 0.25,
        obj_score_thr: float = 0.0,
        nms_conf_thr: None | float = None,
    ) -> None:
        """Initialize the tracking graph."""
        self.min_size = min_size
        self.match_score_thr = match_score_thr
        self.obj_score_thr = obj_score_thr
        self.nms_conf_thr = nms_conf_thr

        self.tracklets = {}
        self.count: int = 0

    def get_ids(
        self, num_ids: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Generate a num_ids number of new unique tracking ids.

        Args:
            num_ids (int): number of ids
            device (torch.device, optional): Device to create ids on. Defaults
                to torch.device("cpu").

        Returns:
            Tensor: Tensor of new contiguous track ids.
        """
        new_ids = torch.arange(self.count, self.count + num_ids, device=device)
        self.count = self.count + num_ids
        return new_ids

    def reset(self) -> None:
        """Reset the tracking graph."""
        self.tracklets.clear()
        self.count = 0

    def is_empty(self) -> bool:
        """Check if the tracking graph is empty."""
        return len(self.tracklets) == 0

    def get_tracks(self, device: torch.device) -> Track3DOut:
        """Get the current tracks.

        Args:
            device: Device for the tensors returned when there is no track.
        """
        track_ids = [track_id for track_id, tracklet in self.tracklets.items()]

        if len(track_ids) == 0:
            return Track3DOut(
                boxes_3d=torch.empty((0, 10), device=device),
                boxes_3d_world=torch.empty((0, 10), device=device),
                track_ids=torch.empty((0,), dtype=torch.long, device=device),
                scores=torch.empty((0,), device=device),
            )

        return Track3DOut(
            boxes_3d=torch.stack(
                [self.tracklets[track_id]["box3d"] for track_id in track_ids]
            ),
            boxes_3d_world=torch.stack(
                [
                    self.tracklets[track_id]["box3d_world"]
                    for track_id in track_ids
                ]
            ),
            track_ids=torch.tensor(track_ids, device=device),
            scores=torch.stack(
                [self.tracklets[track_id]["score"] for track_id in track_ids]
            ),
        )

    def post_process(self, boxes3d: Tensor) -> Tensor:
        """Filter out small boxes by their own extents.

        The frame does not matter: :func:`transform_boxes3d` carries w/l/h across
        unchanged, so camera- and world-frame boxes give the same mask.

        Args:
            boxes3d: Tensor of shape [N, 10], boxes as
                [x, y, z, w, l, h, qw, qx, qy, qz].
            threshold: Minimum box dimension.

        Returns:
            Boolean mask of shape [N] indicating valid boxes.
        """
        return boxes3d[:, 3:6].min(dim=-1).values >= self.min_size

    def __call__(
        self,
        boxes3d: Tensor,
        scores: Tensor,
        extrinsics: Tensor,
        frame_id: int,
    ) -> Track3DOut:
        """Track 3D Objects."""
        if frame_id == 0:
            self.reset()

        device = boxes3d.device

        if len(boxes3d) == 0:
            return self.get_tracks(device=device)

        boxes3d_world = transform_boxes3d(
            boxes3d,
            extrinsics,
            source_axis_mode=AxisMode.OPENCV,
            target_axis_mode=AxisMode.ROS,
        )

        # Remove small objects
        valid_mask = self.post_process(boxes3d_world)
        boxes3d = boxes3d[valid_mask]
        scores = scores[valid_mask]
        boxes3d_world = boxes3d_world[valid_mask]

        if len(boxes3d) == 0:
            return self.get_tracks(device=device)

        if self.is_empty():
            ids = self.get_ids(len(boxes3d), device=device)
        else:
            tracks = self.get_tracks(device=device)

            # 3D IoU similarity
            similarity_scores = box3d_overlap(
                boxes3d_to_corners(boxes3d_world, AxisMode.ROS),
                boxes3d_to_corners(tracks.boxes_3d_world, AxisMode.ROS),
            )

            # Greedy assignment
            ids = greedy_assign(
                scores,
                tracks.track_ids,
                similarity_scores,
                match_score_thr=self.match_score_thr,
                obj_score_thr=self.obj_score_thr,
                nms_conf_thr=self.nms_conf_thr,
            )

            new_inds = ids == NEW_TRACK
            ids[new_inds] = self.get_ids(
                int(new_inds.sum()), device=ids.device
            )

        for i, track_id in enumerate(ids.tolist()):
            if track_id == SUPPRESSED:
                continue

            if track_id in self.tracklets:
                if (
                    boxes3d[i][3:6].prod()
                    > self.tracklets[track_id]["box3d"][3:6].prod()
                ):
                    self.tracklets[track_id] = {
                        "box3d": boxes3d[i],
                        "box3d_world": boxes3d_world[i],
                        "score": scores[i],
                    }
            else:
                self.tracklets[track_id] = {
                    "box3d": boxes3d[i],
                    "box3d_world": boxes3d_world[i],
                    "score": scores[i],
                }

        return self.get_tracks(device=device)
