"""Defines data related constants.

While the datasets can hold arbitrary data types and formats, this file
provides some constants that are used to define a common data format which is
helpful to use for better data transformation.
"""

from dataclasses import dataclass
from enum import Enum

# A custom value to distinguish instance ID and category ID; need to be greater
# than the number of categories. For a pixel in the panoptic result map:
# panaptic_id = instance_id * INSTANCE_OFFSET + category_id
INSTANCE_OFFSET = 1000


class AxisMode(Enum):
    """Enum for choosing among different coordinate frame conventions.

    ROS: The coordinate frame aligns with the right hand rule:
        - x axis points forward.
        - y axis points left.
        - z axis points up.
    See also: https://www.ros.org/reps/rep-0103.html#axis-orientation

    OpenCV: The coordinate frame aligns with a camera coordinate system:
        - x axis points right.
        - y axis points down.
        - z axis points forward.
    See also: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html

    LiDAR: The coordinate frame aligns with a LiDAR coordinate system:
        - x axis points right.
        - y axis points forward.
        - z axis points up.
    See also: https://www.nuscenes.org/nuscenes#data-collection
    """

    ROS = 0
    OPENCV = 1
    LIDAR = 2


@dataclass
class CommonKeys:
    """Common supported keys for DictData.

    While DictData can hold arbitrary keys of data, we define a common set of
    keys where we expect a pre-defined format to enable the usage of common
    data pre-processing operations among different datasets.

    General Info:
        - sample_names (str): Name of the sample.

    If the dataset contains videos:
        - sequence_names (str): The name of the sequence.
        - frame_ids (int): The temporal frame index of the sample.

    Image Based Inputs:
        - images (NDArrayF32): Image of shape [1, H, W, C].
        - input_hw (Tuple[int, int]): Shape of image in (height, width) after
          transformations.
        - original_images (NDArrayF32): Original image of shape [1, H, W, C].
        - original_hw (Tuple[int, int]): Shape of original image in
          (height, width).

    Image Classification:
        - categories (NDArrayI64): Class labels of shape [1, ].

    2D Object Detection:
        - boxes2d (NDArrayF32): 2D bounding boxes of shape [N, 4] in xyxy
          format.
        - boxes2d_classes (NDArrayI64): Classes of 2D bounding boxes of shape
          [N,].
        - boxes2d_names (List[str]): Names of 2D bounding box classes, same
          order as `boxes2d_classes`.

    2D Object Tracking:
        - boxes2d_track_ids (NDArrayI64): Tracking IDs of 2D bounding boxes of
          shape [N,].

    Segmentation:
        - masks (NDArrayUI8): Segmentation masks of shape [N, H, W].
        - seg_masks (NDArrayUI8): Semantic segmentation masks [H, W].
        - instance_masks (NDArrayUI8): Instance segmentation masks of shape
          [N, H, W].
        - panoptic_masks (NDArrayI64): Panoptic segmentation masks [H, W].

    Depth Estimation:
        - depth_maps (NDArrayF32): Depth maps of shape [H, W].

    Optical Flow:
        - optical_flows (NDArrayF32): Optical flow maps of shape [H, W, 2].

    Sensor Calibration:
        - intrinsics (NDArrayF32): Intrinsic sensor calibration. Shape [3, 3].
        - extrinsics (NDArrayF32): Extrinsic sensor calibration, transformation
          of sensor to world coordinate frame. Shape [4, 4].
        - axis_mode (AxisMode): Coordinate convention of the current sensor.
        - timestamp (int): Sensor timestamp in Unix format.

    3D Point Cloud Data:
        - points3d (NDArrayF32): 3D pointcloud data, assumed to be [N, 3] and
          in sensor frame.
        - colors3d (NDArrayF32): Associated color values for each point [N, 3].

    3D Point Cloud Annotations:
        - semantics3d (NDArrayI64): Semantic classes of 3D points [N, 1].
        - instances3d (NDArrayI64): Instance IDs of 3D points [N, 1].

    3D Object Detection:
        - boxes3d (NDArrayF32): 3D bounding boxes of shape [N, 10], each
          consists of center (XYZ), dimensions (WLH), and orientation
          quaternion (WXYZ).
        - boxes3d_classes (NDArrayI64): Associated semantic classes of 3D
          bounding boxes of shape [N,].
        - boxes3d_names (List[str]): Names of 3D bounding box classes, same
          order as `boxes3d_classes`.
        - boxes3d_track_ids (NDArrayI64): Associated tracking IDs of 3D
          bounding boxes of shape [N,].
        - boxes3d_velocities (NDArrayF32): Associated velocities of 3D bounding
          boxes of shape [N, 3], where each velocity is in the form of
          (vx, vy, vz).
    """

    # General Info
    sample_names = "sample_names"
    sequence_names = "sequence_names"
    frame_ids = "frame_ids"

    # image based inputs
    images = "images"
    input_hw = "input_hw"
    original_images = "original_images"
    original_hw = "original_hw"

    # Image Classification
    categories = "categories"

    # 2D Object Detection
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    boxes2d_names = "boxes2d_names"

    # 2D Object Tracking
    boxes2d_track_ids = "boxes2d_track_ids"

    # Segmentation
    masks = "masks"
    seg_masks = "seg_masks"
    instance_masks = "instance_masks"
    panoptic_masks = "panoptic_masks"

    # Depth Estimation
    depth_maps = "depth_maps"

    # Optical Flow
    optical_flows = "optical_flows"

    # Sensor Calibration
    intrinsics = "intrinsics"
    extrinsics = "extrinsics"
    axis_mode = "axis_mode"
    timestamp = "timestamp"

    # 3D Point Cloud Data
    points3d = "points3d"
    colors3d = "colors3d"

    # 3D Point Cloud Annotations
    semantics3d = "semantics3d"
    instances3d = "instances3d"

    # 3D Object Detection
    boxes3d = "boxes3d"
    boxes3d_classes = "boxes3d_classes"
    boxes3d_names = "boxes3d_names"
    boxes3d_track_ids = "boxes3d_track_ids"
    boxes3d_velocities = "boxes3d_velocities"
