"""Convert ScanNet."""

from __future__ import annotations

import argparse
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from mapdet3d.data.const import AxisMode
from mapdet3d.op.box3d import (
    boxes3d_in_image,
    boxes3d_to_boxes2d,
    boxes3d_to_corners,
    transform_boxes3d,
)
from mapdet3d.op.geometry.rotation import (
    matrix_to_quaternion,
    quaternion_multiply,
)
from mapdet3d.op.geometry.transform import (
    inverse_rigid_transform,
    transform_points,
)

from batch_load_scannet_data import SCANNET200_OBJ_CLASS_IDS

SAMPLE_RATE = 25

# ScanNet 18-class detection categories (NYU40 ID -> name)
NYU40_ID_TO_CLASS = {
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "refrigerator",
    28: "shower curtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
    39: "other furniture",
}

SCANNET200_CLASS_MAP = {
    "chair": 2,
    "book": 22,
    "door": 5,
    "object": 1163,
    "window": 16,
    "table": 4,
    "trash can": 56,
    "pillow": 13,
    "picture": 15,
    "box": 26,
    "doorframe": 161,
    "monitor": 19,
    "cabinet": 7,
    "desk": 9,
    "shelf": 8,
    "office chair": 10,
    "towel": 31,
    "couch": 6,
    "sink": 14,
    "backpack": 48,
    "lamp": 28,
    "bed": 11,
    "bookshelf": 18,
    "mirror": 71,
    "curtain": 21,
    "plant": 40,
    "whiteboard": 52,
    "radiator": 96,
    "kitchen cabinet": 29,
    "toilet paper": 49,
    "armchair": 23,
    "shoe": 63,
    "coffee table": 24,
    "toilet": 17,
    "bag": 47,
    "clothes": 32,
    "keyboard": 46,
    "bottle": 65,
    "recycling bin": 97,
    "nightstand": 34,
    "stool": 38,
    "tv": 33,
    "file cabinet": 75,
    "dresser": 36,
    "computer tower": 64,
    "telephone": 101,
    "cup": 130,
    "refrigerator": 27,
    "end table": 44,
    "jacket": 131,
    "shower curtain": 55,
    "bathtub": 42,
    "microwave": 59,
    "kitchen counter": 159,
    "sofa chair": 74,
    "paper towel dispenser": 82,
    "bathroom vanity": 1164,
    "suitcase": 93,
    "laptop": 77,
    "ottoman": 67,
    "shower wall": 128,
    "printer": 50,
    "counter": 35,
    "board": 69,
    "soap dispenser": 100,
    "stove": 62,
    "light": 105,
    "closet wall": 1165,
    "mini fridge": 165,
    "fan": 76,
    "tissue box": 230,
    "blanket": 54,
    "bathroom stall": 125,
    "copier": 72,
    "bench": 68,
    "bar": 145,
    "soap dish": 157,
    "laundry hamper": 1166,
    "storage bin": 132,
    "bathroom stall door": 1167,
    "light switch": 232,
    "coffee maker": 134,
    "tv stand": 51,
    "decoration": 250,
    "ceiling": 41,
    "ceiling light": 1168,
    "range hood": 342,
    "blackboard": 89,
    "clock": 103,
    "wardrobe": 99,
    "rail": 95,
    "bulletin board": 154,
    "mat": 140,
    "trash bin": 1169,
    "ledge": 193,
    "seat": 116,
    "mouse": 202,
    "basket": 73,
    "shower": 78,
    "dumbbell": 1170,
    "paper": 79,
    "person": 80,
    "windowsill": 141,
    "closet": 57,
    "bucket": 102,
    "sign": 261,
    "speaker": 118,
    "dishwasher": 136,
    "container": 98,
    "stair rail": 1171,
    "shower curtain rod": 170,
    "tube": 1172,
    "bathroom cabinet": 1173,
    "storage container": 221,
    "paper bag": 570,
    "paper towel roll": 138,
    "ball": 168,
    "closet door": 276,
    "laundry basket": 106,
    "cart": 214,
    "dish rack": 323,
    "stairs": 58,
    "blinds": 86,
    "purse": 399,
    "bicycle": 121,
    "tray": 185,
    "plunger": 300,
    "paper cutter": 180,
    "toilet paper dispenser": 163,
    "bin": 66,
    "toilet seat cover dispenser": 208,
    "guitar": 112,
    "mailbox": 540,
    "handicap bar": 395,
    "fire extinguisher": 166,
    "ladder": 122,
    "column": 120,
    "pipe": 107,
    "vacuum cleaner": 283,
    "plate": 88,
    "piano": 90,
    "water cooler": 177,
    "cd case": 1174,
    "bowl": 562,
    "closet rod": 1175,
    "bathroom counter": 1156,
    "oven": 84,
    "stand": 104,
    "scale": 229,
    "washing machine": 70,
    "broom": 325,
    "hat": 169,
    "guitar case": 331,
    "rack": 87,
    "water pitcher": 488,
    "laundry detergent": 776,
    "hair dryer": 370,
    "pillar": 191,
    "divider": 748,
    "power outlet": 242,
    "dining table": 45,
    "shower floor": 417,
    "shower door": 188,
    "coffee kettle": 1176,
    "structure": 1178,
    "clothes dryer": 110,
    "toaster": 148,
    "ironing board": 155,
    "alarm clock": 572,
    "shower head": 1179,
    "water bottle": 392,
    "keyboard piano": 1180,
    "projector screen": 609,
    "case of water bottles": 1181,
    "toaster oven": 195,
    "music stand": 581,
    "coat rack": 1182,
    "storage organizer": 1183,
    "machine": 139,
    "folded chair": 1184,
    "fire alarm": 1185,
    "fireplace": 156,
    "vent": 408,
    "furniture": 213,
    "power strip": 1186,
    "calendar": 1187,
    "poster": 1188,
    "toilet paper holder": 115,
    "potted plant": 1189,
    "stuffed animal": 304,
    "luggage": 1190,
    "headphones": 312,
    "crate": 233,
    "candle": 286,
    "projector": 264,
    "mattress": 1191,
    "dustpan": 356,
    "cushion": 39,
    "stick": 1163,
}


CAMERA_NEAR_CLIP = 0.1


def invert_class_map(class_map: dict[str, int]) -> dict[int, str]:
    """Invert a name-to-id class map while preserving first-name wins."""
    id_to_class = {}
    for class_name, class_id in class_map.items():
        id_to_class.setdefault(int(class_id), class_name)
    return id_to_class


SCANNET200_ID_TO_CLASS = invert_class_map(SCANNET200_CLASS_MAP)


def load_scannet200_obj_class_ids() -> set[int]:
    """Load ScanNet200 care-class ids from the batch exporter."""

    return {int(class_id) for class_id in SCANNET200_OBJ_CLASS_IDS.tolist()}


def validate_scannet200_class_map() -> None:
    """Ensure every exported ScanNet200 class has a conversion name."""
    scannet200_obj_class_ids = load_scannet200_obj_class_ids()
    missing_ids = sorted(
        scannet200_obj_class_ids - set(SCANNET200_ID_TO_CLASS)
    )
    if missing_ids:
        raise ValueError(
            "SCANNET200_CLASS_MAP is missing names for class ids: "
            f"{missing_ids}"
        )


def get_class_id_to_name(scannet200: bool) -> dict[int, str]:
    """Return the active ScanNet class id to category-name mapping."""
    if scannet200:
        validate_scannet200_class_map()
        return SCANNET200_ID_TO_CLASS
    return NYU40_ID_TO_CLASS


def default_instance_data_dir(data_root: str, scannet200: bool) -> str:
    """Return the default instance-data directory for the label set."""
    if scannet200:
        return os.path.join(data_root, "scannet200_instance_data")
    return os.path.join(data_root, "scannet_instance_data")


def format_missing_inputs(
    missing: list[tuple[str, str]], limit: int = 20
) -> str:
    """Format missing input paths without dumping hundreds of lines."""
    shown = missing[:limit]
    lines = [f"{scene_id}: {path}" for scene_id, path in shown]
    if len(missing) > limit:
        lines.append(f"... and {len(missing) - limit} more missing inputs")
    return "\n  - ".join(lines)


def collect_missing_inputs(
    scenes: list[str], data_root: str, instance_data_dir: str
) -> list[tuple[str, str]]:
    """Collect required files/directories missing for conversion."""
    missing = []
    for scene_id in scenes:
        scene_root = os.path.join(data_root, "data", scene_id)
        required_dirs = [
            os.path.join(scene_root, "frames", "color"),
            os.path.join(scene_root, "frames", "pose"),
        ]
        required_files = [
            os.path.join(
                scene_root,
                "frames",
                "intrinsic",
                "intrinsic_color.txt",
            ),
            os.path.join(instance_data_dir, f"{scene_id}_aligned_bbox.npy"),
            os.path.join(
                instance_data_dir, f"{scene_id}_axis_align_matrix.npy"
            ),
        ]

        for directory in required_dirs:
            if not os.path.isdir(directory):
                missing.append((scene_id, directory))
        for filename in required_files:
            if not os.path.isfile(filename):
                missing.append((scene_id, filename))

    return missing


def validate_convert_inputs(
    scenes: list[str], data_root: str, instance_data_dir: str
) -> None:
    """Fail early when required ScanNet conversion inputs are absent."""
    if not scenes:
        raise ValueError("No scenes found in the requested split file.")

    missing = collect_missing_inputs(scenes, data_root, instance_data_dir)
    if missing:
        raise FileNotFoundError(
            "Missing required ScanNet conversion inputs:\n  - "
            f"{format_missing_inputs(missing)}\n"
            "Run scripts/scannet/batch_load_scannet_data.py first with "
            "labeled ScanNet annotations, or pass --instance_data_dir to "
            "an existing instance-data directory."
        )


def frame_id_from_filename(filename: str) -> int:
    """Parse a ScanNet numeric frame id from a frame filename."""
    stem = os.path.splitext(filename)[0]
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(
            f"Expected numeric ScanNet frame filename, got {filename}."
        ) from exc


def list_frame_paths(frame_dir: str, suffix: str) -> dict[int, str]:
    """List ScanNet frame files keyed by their numeric frame id."""
    frame_paths = {}
    for filename in os.listdir(frame_dir):
        if not filename.endswith(suffix):
            continue
        frame_id = frame_id_from_filename(filename)
        frame_paths[frame_id] = os.path.join(frame_dir, filename)
    return frame_paths


def load_valid_frames(
    pose_image_dir: str, pose_dir: str
) -> list[tuple[int, str, np.ndarray]]:
    """Load finite-pose ScanNet frames as raw id, image path, and pose."""
    image_paths = list_frame_paths(pose_image_dir, ".jpg")
    pose_paths = list_frame_paths(pose_dir, ".txt")

    valid_frames = []
    for raw_frame_id in sorted(set(image_paths) & set(pose_paths)):
        cam_to_world = np.loadtxt(pose_paths[raw_frame_id]).astype(np.float32)
        if np.all(np.isfinite(cam_to_world)):
            valid_frames.append(
                (raw_frame_id, image_paths[raw_frame_id], cam_to_world)
            )
    return valid_frames


def process_scene(args):
    """Process a single ScanNet scene."""
    scene_id, data_root, cache_dir, instance_data_dir, class_id_to_name = args

    pose_image_dir = os.path.join(
        data_root, "data", scene_id, "frames", "color"
    )
    pose_dir = os.path.join(data_root, "data", scene_id, "frames", "pose")

    # ------------------------------------------------------------------
    # Load valid frame paths and extrinsics
    # ------------------------------------------------------------------
    valid_frames = load_valid_frames(pose_image_dir, pose_dir)
    if len(valid_frames) == 0:
        return scene_id

    # ------------------------------------------------------------------
    # Camera intrinsics (shared across all frames)
    # ------------------------------------------------------------------
    intrinsics_np = np.loadtxt(
        os.path.join(
            data_root,
            "data",
            scene_id,
            "frames",
            "intrinsic",
            "intrinsic_color.txt",
        )
    ).astype(np.float32)[:3, :3]
    intrinsics = torch.from_numpy(intrinsics_np)

    # ------------------------------------------------------------------
    # 3D bounding boxes (axis-aligned in aligned coordinate frame)
    # ------------------------------------------------------------------
    aligned_bbox_path = os.path.join(
        instance_data_dir, f"{scene_id}_aligned_bbox.npy"
    )
    if not os.path.exists(aligned_bbox_path):
        return scene_id

    aligned_box_label = np.load(aligned_bbox_path)
    if aligned_box_label.shape[0] == 0:
        return scene_id

    # Filter by valid ScanNet classes
    class_ids = aligned_box_label[:, -1].astype(np.int64)
    valid_mask = np.array([int(cid) in class_id_to_name for cid in class_ids])
    if not valid_mask.any():
        return scene_id

    # Original row indices → stable track IDs across frames
    valid_indices = np.where(valid_mask)[0].astype(np.int64)

    aligned_box_label = aligned_box_label[valid_mask]
    class_ids = class_ids[valid_mask]
    categories_all = [class_id_to_name[int(c)] for c in class_ids]

    # Build boxes3d [N, 10]: xyz, w, l, h, qw, qx, qy, qz
    aligned_box = aligned_box_label[:, :6].astype(np.float32)
    n_boxes = aligned_box.shape[0]

    # Identity quaternion (axis-aligned boxes)
    x, y, z, w = R.from_euler("XYZ", [0, 0, 0]).as_quat()
    identity_quat = np.array([w, x, y, z], dtype=np.float32)

    boxes3d_aligned = np.zeros((n_boxes, 10), dtype=np.float32)
    boxes3d_aligned[:, :3] = aligned_box[:, :3]  # center
    boxes3d_aligned[:, 3] = aligned_box[:, 4]  # w (y_size)
    boxes3d_aligned[:, 4] = aligned_box[:, 3]  # l (x_size)
    boxes3d_aligned[:, 5] = aligned_box[:, 5]  # h (z_size)
    boxes3d_aligned[:, 6:10] = identity_quat

    # Axis alignment matrix & its inverse
    axis_align_matrix = torch.from_numpy(
        np.load(
            os.path.join(
                instance_data_dir,
                f"{scene_id}_axis_align_matrix.npy",
            )
        ).astype(np.float32)
    )
    inv_align_matrix = inverse_rigid_transform(axis_align_matrix)

    # Transform from aligned frame to world frame
    boxes3d_aligned = torch.from_numpy(boxes3d_aligned)

    boxes3d_world = boxes3d_aligned.new_zeros(boxes3d_aligned.shape)

    # Transform center
    boxes3d_world[:, :3] = transform_points(
        boxes3d_aligned[:, :3], inv_align_matrix
    )

    boxes3d_world[:, 3:6] = boxes3d_aligned[:, 3:6]

    rot_quat = matrix_to_quaternion(
        inv_align_matrix[:3, :3].unsqueeze(0)
    )  # [1, 4]
    rigid_quat = quaternion_multiply(
        rot_quat.expand(n_boxes, -1), boxes3d_aligned[:, 6:]
    )  # [N, 4]

    # Final quaternion: Q_world = Q_rigid * Q_correction
    boxes3d_world[:, 6:] = rigid_quat

    # ------------------------------------------------------------------
    # Subsample frames
    # ------------------------------------------------------------------
    seq_data = []
    samples = []
    frame_id = 0
    landscape_count = 0
    portrait_count = 0
    total_annotations = 0

    for raw_frame_id, image_path, cam_to_world in valid_frames[::SAMPLE_RATE]:
        timestamp = raw_frame_id

        # ---- Image ----
        pil_img = ImageOps.exif_transpose(Image.open(image_path))
        img_arr = np.array(pil_img)
        img_h, img_w = img_arr.shape[:2]

        if img_w >= img_h:
            landscape_count += 1
        else:
            portrait_count += 1

        # ---- Camera transforms ----
        world_to_cam = inverse_rigid_transform(
            torch.from_numpy(cam_to_world.astype(np.float32))
        )

        # ---- Transform boxes to camera frame ----
        boxes3d_cam = transform_boxes3d(
            boxes3d_world,
            world_to_cam,
            AxisMode.ROS,
            AxisMode.OPENCV,
        )

        # ---- Filter visible boxes ----
        corners = boxes3d_to_corners(boxes3d_cam, AxisMode.OPENCV)
        vis_mask = boxes3d_in_image(corners, intrinsics, (img_h, img_w))

        boxes3d_cam_vis = boxes3d_cam[vis_mask]
        cats_vis = [c for c, m in zip(categories_all, vis_mask.tolist()) if m]

        # ---- Project to 2D ----
        if boxes3d_cam_vis.shape[0] > 0:
            boxes2d = boxes3d_to_boxes2d(
                boxes3d_cam_vis,
                intrinsics,
                AxisMode.OPENCV,
                CAMERA_NEAR_CLIP,
                (img_h, img_w),
            ).numpy()
        else:
            boxes2d = np.empty((0, 4), dtype=np.float32)

        boxes3d_np = (
            boxes3d_cam_vis.numpy()
            if boxes3d_cam_vis.shape[0] > 0
            else np.empty((0, 10), dtype=np.float32)
        )

        # Track IDs (static scene – each box is a unique instance)
        track_ids = (
            valid_indices[vis_mask.numpy()]
            if vis_mask.any()
            else np.empty((0,), dtype=np.int64)
        )

        # ---- Depth map ----
        depth_file_path = image_path.replace(
            "frames/color", "frames/depth"
        ).replace(".jpg", ".png")

        # ---- Assemble frame dict (CA1M-compatible) ----
        seq_data.append(
            {
                "image_file_path": image_path,
                "depth_file_path": depth_file_path,
                "arkit_depth_file_path": depth_file_path,
                "intrinsics": intrinsics_np,
                "extrinsics": cam_to_world,
                "T_gravity": np.eye(4, dtype=np.float32),
                "boxes2d": boxes2d,
                "boxes3d": boxes3d_np,
                "boxes3d_cam_from_world": boxes3d_np,
                "categories": cats_vis,
                "track_ids": track_ids,
            }
        )

        samples.append(
            {
                "sequence_name": scene_id,
                "timestamp": timestamp,
                "frame_id": frame_id,
                "num_instances": len(boxes2d),
                "boxes3d_world": boxes3d_world.numpy(),
            }
        )
        frame_id += 1
        total_annotations += len(boxes2d)

    if total_annotations == 0:
        return scene_id

    # Save per-scene cache
    cached_file = os.path.join(cache_dir, f"{scene_id}.pkl")
    with open(cached_file, "wb") as f:
        f.write(
            pickle.dumps(
                {
                    "seq_data": seq_data,
                    "format": (
                        "landscape"
                        if landscape_count >= portrait_count
                        else "portrait"
                    ),
                }
            )
        )

    return samples, set(categories_all)


if __name__ == "__main__":
    """Convert ScanNet V2 dataset."""
    parser = argparse.ArgumentParser(description="Convert ScanNet.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scannet",
        help="Root path of the raw ScanNet dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val"],
        default="val",
        help="Which split to convert.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=(
            "Directory containing per-scene *_aligned_bbox.npy and "
            "*_axis_align_matrix.npy files. Defaults to "
            "<data_root>/scannet_instance_data, or "
            "<data_root>/scannet200_instance_data with --scannet200."
        ),
    )
    parser.add_argument(
        "--scannet200",
        action="store_true",
        help=(
            "Use ScanNet200 class ids/names and default to separate "
            "scannet200 instance-data and cache paths."
        ),
    )
    args = parser.parse_args()

    data_root = args.data_root
    instance_data_dir = args.instance_data_dir or default_instance_data_dir(
        data_root, args.scannet200
    )
    class_id_to_name = get_class_id_to_name(args.scannet200)

    if args.scannet200:
        assert (
            "200" in args.split
        ), "ScanNet200 requires a split with '200' in the name."

    split_file = os.path.join(data_root, "meta_data", f"{args.split}.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        scenes = f.read().splitlines()

    validate_convert_inputs(scenes, data_root, instance_data_dir)

    cache_dir = os.path.join(data_root, "cache", args.split)
    cache_file = os.path.join(data_root, "cache", f"{args.split}.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    task_args = [
        (scene, data_root, cache_dir, instance_data_dir, class_id_to_name)
        for scene in scenes
    ]

    n_procs = min(cpu_count(), len(scenes))
    print(f"Using {n_procs} processes to convert" f" {len(scenes)} scenes...")

    results = []
    invalid_seqs = []
    valid_seq = 0
    total_cats = set()

    with Pool(n_procs) as pool:
        for result in tqdm(
            pool.imap_unordered(process_scene, task_args),
            total=len(task_args),
        ):
            if isinstance(result, str):
                invalid_seqs.append(result)
            else:
                samples_list, cats = result
                valid_seq += 1
                results.append(samples_list)
                total_cats.update(cats)

    data = sum(results, [])

    with open(cache_file, "wb") as f:
        f.write(pickle.dumps(data))

    print(
        f"Processed {valid_seq} scenes with {len(data)} images."
        f" Categories: {sorted(total_cats)}"
    )
    if invalid_seqs:
        print(f"Invalid/empty scenes: {len(invalid_seqs)}")
