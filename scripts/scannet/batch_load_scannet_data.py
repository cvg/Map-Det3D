"""Batch mode in loading Scannet scenes with vertices and ground truth.

# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""

import argparse
import datetime
import os
from os import path as osp

import numpy as np
from load_scannet_data import export

DONOTCARE_CLASS_IDS = np.array([])

SCANNET_OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)

SCANNET200_OBJ_CLASS_IDS = np.array(
    [
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        35,
        36,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        54,
        55,
        56,
        57,
        58,
        59,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        82,
        84,
        86,
        87,
        88,
        89,
        90,
        93,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        110,
        112,
        115,
        116,
        118,
        120,
        121,
        122,
        125,
        128,
        130,
        131,
        132,
        134,
        136,
        138,
        139,
        140,
        141,
        145,
        148,
        154,
        155,
        156,
        157,
        159,
        161,
        163,
        165,
        166,
        168,
        169,
        170,
        177,
        180,
        185,
        188,
        191,
        193,
        195,
        202,
        208,
        213,
        214,
        221,
        229,
        230,
        232,
        233,
        242,
        250,
        261,
        264,
        276,
        283,
        286,
        300,
        304,
        312,
        323,
        325,
        331,
        342,
        356,
        370,
        392,
        395,
        399,
        408,
        417,
        488,
        540,
        562,
        570,
        572,
        581,
        609,
        748,
        776,
        1156,
        1163,
        1164,
        1165,
        1166,
        1167,
        1168,
        1169,
        1170,
        1171,
        1172,
        1173,
        1174,
        1175,
        1176,
        1178,
        1179,
        1180,
        1181,
        1182,
        1183,
        1184,
        1185,
        1186,
        1187,
        1188,
        1189,
        1190,
        1191,
    ]
)


def default_output_folder(scannet200: bool) -> str:
    """Return the default instance-data folder for the selected label set."""
    if scannet200:
        return "data/scannet/scannet200_instance_data"
    return "data/scannet/scannet_instance_data"


def resolve_scene_dir(scannet_dir: str, scan_name: str) -> str:
    """Resolve old and current ScanNet scene layouts.

    Supported inputs:
        data/scannet/data -> data/scannet/data/<scene>
        data/scannet      -> data/scannet/data/<scene>
        data/scannet/scans -> data/scannet/scans/<scene>
    """
    candidates = [
        osp.join(scannet_dir, scan_name),
        osp.join(scannet_dir, "data", scan_name),
    ]

    seen = set()
    for candidate in candidates:
        normalized = osp.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if osp.isdir(candidate):
            return candidate

    return candidates[0]


def get_scene_files(scan_name: str, scene_dir: str) -> dict[str, str]:
    """Return canonical ScanNet input paths for a scene."""
    return {
        "mesh": osp.join(scene_dir, scan_name + "_vh_clean_2.ply"),
        "aggregation": osp.join(scene_dir, scan_name + ".aggregation.json"),
        "segmentation": osp.join(
            scene_dir, scan_name + "_vh_clean_2.0.010000.segs.json"
        ),
        "meta": osp.join(scene_dir, f"{scan_name}.txt"),
    }


def validate_scene_inputs(
    scan_name: str,
    scene_dir: str,
    scene_files: dict[str, str],
    label_map_file: str,
    test_mode: bool,
) -> None:
    """Validate scene inputs before calling the lower-level exporter."""
    required = ["mesh", "meta"]
    if not test_mode:
        required.extend(["aggregation", "segmentation"])

    missing = [
        scene_files[key]
        for key in required
        if not osp.isfile(scene_files[key])
    ]
    if not test_mode and not (label_map_file and osp.isfile(label_map_file)):
        missing.append(label_map_file or "<label_map_file not set>")

    if missing:
        missing_lines = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing ScanNet inputs for "
            f"{scan_name} under {scene_dir}:\n  - {missing_lines}\n"
            "Labeled export requires the ScanNet aggregation and "
            "segmentation JSON files. Use --test_mode/--mesh_only only "
            "when labels and boxes are not needed."
        )


def export_one_scan(
    scan_name,
    output_filename_prefix,
    max_num_point,
    label_map_file,
    scannet_dir,
    test_mode=False,
    scannet200=False,
):
    scene_dir = resolve_scene_dir(scannet_dir, scan_name)
    scene_files = get_scene_files(scan_name, scene_dir)
    validate_scene_inputs(
        scan_name,
        scene_dir,
        scene_files,
        label_map_file,
        test_mode,
    )

    (
        mesh_vertices,
        semantic_labels,
        instance_labels,
        unaligned_bboxes,
        aligned_bboxes,
        instance2semantic,
        axis_align_matrix,
    ) = export(
        scene_files["mesh"],
        scene_files["aggregation"],
        scene_files["segmentation"],
        scene_files["meta"],
        label_map_file,
        None,
        test_mode=test_mode,
        scannet200=scannet200,
    )

    if not test_mode:
        mask = np.logical_not(np.isin(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f"Num of instances: {num_instances}")
        if scannet200:
            OBJ_CLASS_IDS = SCANNET200_OBJ_CLASS_IDS
        else:
            OBJ_CLASS_IDS = SCANNET_OBJ_CLASS_IDS

        bbox_mask = np.isin(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.isin(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f"Num of care instances: {unaligned_bboxes.shape[0]}")

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

    np.save(f"{output_filename_prefix}_vert.npy", mesh_vertices)
    np.save(
        f"{output_filename_prefix}_axis_align_matrix.npy",
        axis_align_matrix,
    )
    if not test_mode:
        np.save(f"{output_filename_prefix}_sem_label.npy", semantic_labels)
        np.save(f"{output_filename_prefix}_ins_label.npy", instance_labels)
        np.save(
            f"{output_filename_prefix}_unaligned_bbox.npy", unaligned_bboxes
        )
        np.save(f"{output_filename_prefix}_aligned_bbox.npy", aligned_bboxes)


def batch_export(
    max_num_point: int | None,
    output_folder: str,
    scan_names_file: str,
    label_map_file: str,
    scannet_dir: str,
    scannet200: bool = False,
    test_mode: bool = False,
    skip_missing: bool = False,
):
    """Batch export scans."""
    print(f"Creating new data folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    with open(scan_names_file, encoding="utf-8") as f:
        scan_names = [line.strip() for line in f if line.strip()]

    processed = 0
    skipped_existing = 0
    skipped_missing = 0
    for scan_name in scan_names:
        print("-" * 20 + "begin")
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f"{output_filename_prefix}_vert.npy"):
            print("File already exists. skipping.")
            print("-" * 20 + "done")
            skipped_existing += 1
            continue

        try:
            export_one_scan(
                scan_name,
                output_filename_prefix,
                max_num_point,
                label_map_file,
                scannet_dir,
                test_mode=test_mode,
                scannet200=scannet200,
            )
        except FileNotFoundError as exc:
            if not skip_missing:
                raise
            print(exc)
            skipped_missing += 1
            print("-" * 20 + "done")
            continue

        processed += 1
        print("-" * 20 + "done")

    print(
        "Batch export complete. "
        f"processed={processed}, "
        f"skipped_existing={skipped_existing}, "
        f"skipped_missing={skipped_missing}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_num_point",
        default=None,
        help="The maximum number of the points.",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help=(
            "output folder of the result. Defaults to "
            "data/scannet/scannet_instance_data or "
            "data/scannet/scannet200_instance_data."
        ),
    )
    parser.add_argument(
        "--scannet_dir",
        default="data/scannet/data",
        help=(
            "ScanNet scene directory. Accepts data/scannet/data, "
            "data/scannet, or the legacy data/scannet/scans layout."
        ),
    )
    parser.add_argument(
        "--label_map_file",
        default="data/scannet/meta_data/scannetv2-labels.combined.tsv",
        help="The path of label map file.",
    )
    parser.add_argument(
        "--scan_names_file",
        default="data/scannet/meta_data/val.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument(
        "--scannet200",
        action="store_true",
        help="Use it for scannet200 mapping",
    )
    parser.add_argument(
        "--test_mode",
        "--mesh_only",
        dest="test_mode",
        action="store_true",
        help=(
            "Export mesh vertices and axis alignment without semantic labels, "
            "instance labels, or bounding boxes."
        ),
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip scenes with missing files instead of failing immediately.",
    )
    args = parser.parse_args()

    output_folder = args.output_folder or default_output_folder(
        args.scannet200
    )

    batch_export(
        args.max_num_point,
        output_folder,
        args.scan_names_file,
        args.label_map_file,
        args.scannet_dir,
        scannet200=args.scannet200,
        test_mode=args.test_mode,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
