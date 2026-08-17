"""Convert CA-1M dataset.

For each video:
    # image: wide/image, _wide/image/size, wide/image/K
    # instances: wide/instances
    # arkit depth: wide/depth, _wide/depth/size, wide/depth/K
    # GT depth: gt/depth, _gt/depth/size, gt/depth/K
    # GT pose: gt/RT
    # Global GT: world.gt/instances
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from mapdet3d.data.const import AxisMode
from mapdet3d.data.io.to_hdf5 import convert_with_map
from mapdet3d.op.box3d import transform_boxes3d
from mapdet3d.op.geometry.rotation import standardize_quaternion
from mapdet3d.op.geometry.transform import inverse_rigid_transform


def process_sequence(args):
    """Process a single sequence."""
    seq, split_dir, target_split_dir, cache_dir = args

    seq_dir = os.path.join(split_dir, seq)
    target_seq_dir = os.path.join(target_split_dir, seq)

    target_image_dir = os.path.join("", "images")
    target_depth_dir = os.path.join("", "depths")
    target_arkit_depth_dir = os.path.join("", "arkit_depths")
    pathmap = {}

    gt_json = os.path.join(seq_dir, "world.gt", "instances.json")
    instances_data = json.load(
        open(gt_json, encoding="utf-8"),
    )
    global_anns = {box["id"]: box for box in instances_data}

    boxes3d_world = torch.tensor(
        [
            [
                *ann["position"],
                ann["scale"][1],  # size_y
                ann["scale"][0],  # size_x
                ann["scale"][2],  # size_z
                R.from_matrix(np.array(ann["R"])).as_quat()[3].item(),
                R.from_matrix(np.array(ann["R"])).as_quat()[0].item(),
                R.from_matrix(np.array(ann["R"])).as_quat()[1].item(),
                R.from_matrix(np.array(ann["R"])).as_quat()[2].item(),
            ]
            for _, ann in global_anns.items()
        ],
        dtype=torch.float32,
    )

    boxes3d_world[:, 6:10] = standardize_quaternion(boxes3d_world[:, 6:10])

    samples = []
    seq_data = []
    instance_ids = []
    frame_id = 0
    seen_timestamps = []

    landscape_count = 0
    portrait_count = 0
    record_id = -1
    total_annotations = 0
    records = sorted(os.listdir(seq_dir))
    categories = set()
    for record in records:
        if record in {"world.gt", "mesh.ply"}:
            continue

        timestamp = record.split(".")[0]

        if timestamp in seen_timestamps:
            continue

        record_id += 1
        seen_timestamps.append(timestamp)

        # Image
        img_file_path = os.path.join(seq_dir, f"{timestamp}.wide", "image.png")
        target_image_path = os.path.join(
            target_seq_dir, "images", f"{timestamp}.png"
        )
        pathmap[img_file_path] = os.path.join(
            target_image_dir, f"{timestamp}.png"
        )

        with open(
            os.path.join(seq_dir, f"{timestamp}._wide", "image", "size"),
            encoding="utf-8",
        ) as f:
            img_width, img_height = f.read().strip().strip("[]").split(", ")
            img_height = int(img_height)
            img_width = int(img_width)

            if img_width >= img_height:
                landscape_count += 1
            else:
                portrait_count += 1

        # Camera info
        intrinsics = np.array(
            json.load(
                open(
                    os.path.join(
                        seq_dir, f"{timestamp}.wide", "image", "K.json"
                    ),
                    encoding="utf-8",
                )
            ),
            dtype=np.float32,
        )

        T_gravity = np.array(
            json.load(
                open(
                    os.path.join(
                        seq_dir, f"{timestamp}.wide", "T_gravity.json"
                    ),
                    encoding="utf-8",
                )
            ),
            dtype=np.float32,
        )

        extrinsics = np.array(
            json.load(
                open(
                    os.path.join(seq_dir, f"{timestamp}.gt", "RT.json"),
                    encoding="utf-8",
                )
            ),
            dtype=np.float32,
        )

        # GT Depth
        depth_file_path = os.path.join(seq_dir, f"{timestamp}.gt", "depth.png")
        target_depth_path = os.path.join(
            target_seq_dir, "depths", f"{timestamp}.png"
        )
        pathmap[depth_file_path] = os.path.join(
            target_depth_dir, f"{timestamp}.png"
        )

        # ARkit Depth
        arkit_depth_file_path = os.path.join(
            seq_dir, f"{timestamp}.wide", "depth.png"
        )
        target_arkit_depth_path = os.path.join(
            target_seq_dir, "arkit_depths", f"{timestamp}.png"
        )
        pathmap[arkit_depth_file_path] = os.path.join(
            target_arkit_depth_dir, f"{timestamp}.png"
        )

        # Annotations (wide/instances)
        instances_data = json.load(
            open(
                os.path.join(seq_dir, f"{timestamp}.wide", "instances.json"),
                encoding="utf-8",
            ),
        )

        # Convert 3D boxes from world coordinates to camera coordinates
        boxes3d_cam = transform_boxes3d(
            torch.asarray(boxes3d_world),
            inverse_rigid_transform(torch.from_numpy(extrinsics)),
            AxisMode.ROS,
            AxisMode.OPENCV,
        )

        boxes3d_cam_dict = {
            uid: boxes3d_cam[i] for i, uid in enumerate(global_anns)
        }

        boxes2d_list = []
        boxes3d_list = []
        boxes3d_cam_from_world_list = []
        categories_list = []
        track_ids_list = []
        for ann in instances_data:
            # Remove annotations not in global anns
            if ann["id"] not in global_anns:
                continue

            # Category
            category = global_anns[ann["id"]]["category"]

            box3d_cam_global = boxes3d_cam_dict[ann["id"]]

            # xyz
            center = ann["position"]

            # size_x (l), size_y (h), size_z (w)
            length, height, width = ann["scale"]

            # Rotation matrix
            try:
                x, y, z, w = R.from_matrix(np.array(ann["R"])).as_quat()
                if w < 0:
                    w, x, y, z = -w, -x, -y, -z
            except Exception as e:
                print(
                    f"Error processing rotation matrix for annotation {ann['id']}: {e}"
                )
                continue

            orientation = Quaternion([w, x, y, z])

            boxes3d_cam_from_world_list.append(
                box3d_cam_global.numpy().tolist()
            )

            boxes3d_list.append(
                [*center, width, length, height, *orientation.elements]
            )

            categories_list.append(category)
            categories.add(category)

            # 2D box
            if "box_2d_rend" in ann:
                box2d = ann["box_2d_rend"]
            elif "box_2d_proj" in ann:
                box2d = ann["box_2d_proj"]
            else:
                print(f"Unknown 2D box format for annotation: {ann}")
                continue

            boxes2d_list.append(box2d)

            # Track ID
            if ann["id"] not in instance_ids:
                instance_ids.append(ann["id"])

            track_ids_list.append(instance_ids.index(ann["id"]))

        boxes2d = (
            np.empty((0, 4), dtype=np.float32)
            if not boxes2d_list
            else np.array(boxes2d_list, dtype=np.float32)
        )

        boxes3d = (
            np.empty((0, 10), dtype=np.float32)
            if not boxes3d_list
            else np.array(boxes3d_list, dtype=np.float32)
        )

        boxes3d_cam_from_world = (
            np.empty((0, 10), dtype=np.float32)
            if not boxes3d_cam_from_world_list
            else np.array(boxes3d_cam_from_world_list, dtype=np.float32)
        )

        track_ids = (
            np.empty((0,), dtype=np.int64)
            if not track_ids_list
            else np.array(track_ids_list, dtype=np.int64)
        )

        seq_data.append(
            {
                "image_file_path": target_image_path,
                "depth_file_path": target_depth_path,
                "arkit_depth_file_path": target_arkit_depth_path,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "T_gravity": T_gravity,
                "boxes2d": boxes2d,
                "boxes3d": boxes3d,
                "boxes3d_cam_from_world": boxes3d_cam_from_world,
                "categories": categories_list,
                "track_ids": track_ids,
            }
        )

        samples.append(
            {
                "sequence_name": seq,
                "timestamp": timestamp,
                "frame_id": frame_id,
                "num_instances": len(boxes2d),
            }
        )
        frame_id += 1
        total_annotations += len(boxes2d)

    if total_annotations == 0:
        return seq

    # Cache sample data
    cached_file_path = os.path.join(cache_dir, f"{seq}.pkl")

    with open(cached_file_path, "wb") as file:
        file.write(
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

    # Generate HDF5 file with pathmap
    convert_with_map(
        pathmap,
        out_hdf5=os.path.join(target_split_dir, f"{seq}.hdf5"),
        with_progress=False,
    )

    return samples, categories


if __name__ == "__main__":
    """Convert CA-1M dataset."""
    parser = argparse.ArgumentParser(description="Convert CA-1M dataset.")
    parser.add_argument(
        "--data_root",
        type=str,
        help="The path of the original CA-1M dataset.",
        default="./data/CA1M",
    )
    parser.add_argument(
        "--target_data_root",
        type=str,
        help="The path to save the converted CA-1M dataset.",
        default="./data/ca1m",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Which split to convert (train or val).",
    )
    parser.add_argument(
        "--cached_dir",
        type=str,
        default="cache",
        help="Directory to store cached data.",
    )
    args = parser.parse_args()
    data_root = args.data_root
    target_data_root = args.target_data_root

    sequences = [
        l.split("/")[-1].split(".")[0].split("-")[-1]
        for l in open(
            os.path.join(data_root, f"{args.split}.txt"), "r"
        ).readlines()
    ]

    split_dir = os.path.join(data_root, args.split)

    target_split_dir = os.path.join(target_data_root, args.split)

    # Cache directory
    cache_dir = os.path.join(target_data_root, args.cached_dir, args.split)
    cache_file = os.path.join(
        target_data_root, args.cached_dir, f"{args.split}.pkl"
    )
    os.makedirs(cache_dir, exist_ok=True)

    task_args = []
    for seq in sequences:
        task_args.append((seq, split_dir, target_split_dir, cache_dir))

    # Process sequences in parallel
    n_procs = min(cpu_count(), len(sequences))
    print(
        f"Using {n_procs} processes to convert {len(sequences)} sequences..."
    )

    with Pool(n_procs) as pool:
        results = []
        invalid_seqs = []
        valid_seq = 0
        total_cats = set()
        for result in tqdm(
            pool.imap_unordered(process_sequence, task_args),
            total=len(task_args),
        ):
            if isinstance(result, str):
                invalid_seqs.append(result)
            else:
                samples, cats = result
                valid_seq += 1
                results.append(samples)
                total_cats.update(cats)

    data = sum(results, [])

    with open(cache_file, "wb") as file:
        file.write(pickle.dumps(data))

    # Dump invalid sequences to txt file
    print(f"{len(invalid_seqs)} invalid sequences: {', '.join(invalid_seqs)}")

    print(f"Processed {valid_seq} sequences with {len(data)} images.")
