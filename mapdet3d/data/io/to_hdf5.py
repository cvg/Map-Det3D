"""Script to convert a dataset to hdf5 format."""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
from tqdm import tqdm

from mapdet3d.common.logging import rank_zero_warn


def _ensure_group(h5: "h5py.File", group_path: str):
    """
    Create (and return) nested groups like 'a/b/c'. Empty or '/' returns root.
    """
    if not group_path or group_path == "/":
        return h5
    cur = h5
    # remove leading/trailing slashes and split
    for part in group_path.strip("/").split("/"):
        if part not in cur:
            cur = cur.create_group(part)
        else:
            cur = cur[part]
    return cur


def convert_dataset(source_dir: str) -> None:
    """Original walk-based conversion (kept for backward compatibility)."""
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"No such file or directory: {source_dir}")

    source_dir = os.path.join(source_dir, "")  # must end with trailing slash
    hdf5_path = source_dir.rstrip("/") + ".hdf5"
    if os.path.exists(hdf5_path):
        print(f"File {hdf5_path} already exists! Skipping {source_dir}")
        return

    print(f"Converting dataset at: {source_dir}")
    hdf5_file = h5py.File(hdf5_path, mode="w")
    sub_dirs = list(os.walk(source_dir))
    file_count = sum(len(files) for (_, _, files) in sub_dirs)

    with tqdm(total=file_count) as pbar:
        for root, _, files in sub_dirs:
            g_name = root.replace(source_dir, "")
            g = hdf5_file.create_group(g_name) if g_name else hdf5_file
            for f in files:
                filepath = os.path.join(root, f)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as fp:
                        file_content = fp.read()
                    g.create_dataset(
                        f, data=np.frombuffer(file_content, dtype="uint8")
                    )
                pbar.update()

    hdf5_file.close()
    print("done.")


def convert_with_map(
    mapping,
    out_hdf5: str,
    base_dir: str | None = None,
    with_progress: bool = True,
) -> None:
    """Convert using a JSON mapping { src_path: hdf5_path }.

    - src_path may be absolute or relative to base_dir (if provided).
    - hdf5_path MUST be 'group1/group2/.../dataset_name' (no leading slash).
    """
    if os.path.exists(out_hdf5):
        rank_zero_warn(
            f"Output HDF5 already exists: {out_hdf5}. Will overwrite."
        )

    os.makedirs(os.path.dirname(out_hdf5) or ".", exist_ok=True)

    if with_progress:
        pbar = tqdm(total=len(mapping))

    with h5py.File(out_hdf5, "w") as h5:
        for src, dst in mapping.items():
            src_path = (
                os.path.join(base_dir, src)
                if (base_dir and not os.path.isabs(src))
                else src
            )
            if not os.path.isfile(src_path):
                if with_progress:
                    pbar.update()
                continue

            # split dst into group + dataset name
            dst = dst.strip("/")
            group_path = os.path.dirname(dst)
            dset_name = os.path.basename(dst)
            if not dset_name:
                raise ValueError(
                    f"Bad mapping (no dataset name) for: {src} -> {dst}"
                )

            g = _ensure_group(h5, group_path)
            with open(src_path, "rb") as fp:
                payload = fp.read()
            g.create_dataset(
                dset_name, data=np.frombuffer(payload, dtype="uint8")
            )
            if with_progress:
                pbar.update()


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Converts a dataset at the specified path to hdf5. The "
        "local directory structure is preserved in the hdf5 file."
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="path to the root folder of a specific dataset to convert",
    )
    parser.add_argument(
        "--map", help="JSON mapping {src: hdf5_path}", default=None
    )
    parser.add_argument(
        "--out", help="output HDF5 path (required with --map)", default=None
    )
    parser.add_argument(
        "--base",
        help="base dir to resolve relative src paths in --map",
        default=None,
    )
    args = parser.parse_args()

    if args.map:
        with open(args.map, "r") as f:
            mapping: dict[str, str] = json.load(f)
        convert_with_map(mapping, args.out, base_dir=args.base)
    else:
        convert_dataset(args.path)
