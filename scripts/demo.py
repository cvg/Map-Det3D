"""Demo Map-Det3D."""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from mapdet3d.model.mapdet3d import MapDet3D, MapDet3DOut
from mapdet3d.op.mapdet3d.head import RoI2Det
from mapdet3d.vis.rerun import RerunVisualizer


def list_frame_paths(
    image_dir: str,
    pose_dir: str,
    image_suffix: str = ".jpg",
    pose_suffix: str = ".txt",
) -> tuple[list[str], list[str]]:
    """List paired image and pose paths ordered by numeric frame id.

    Only frame ids that have both an image and a pose file are kept.
    """
    images = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(image_suffix)
    }
    poses = {
        int(os.path.splitext(f)[0]): os.path.join(pose_dir, f)
        for f in os.listdir(pose_dir)
        if f.endswith(pose_suffix)
    }

    frame_ids = sorted(set(images) & set(poses))

    return (
        [images[fid] for fid in frame_ids],
        [poses[fid] for fid in frame_ids],
    )


def run(data_root: str):
    """Run demo on the sequence stored in data_root."""
    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("highest")

    # Init visualizer
    visualizer = RerunVisualizer(convert_to_world=True, log_mesh=True)

    # Init model
    model = MapDet3D.from_pretrained("RoyYang0714/Map-Det3D").to(device)

    # Enable tracking
    model.track_whole_scene = True
    model.roi2det = RoI2Det(nms=True, score_threshold=0.25, iou_threshold=0.5)

    model.eval()

    # Views
    mesh_path = os.path.join(data_root, "mesh.ply")

    image_paths, pose_paths = list_frame_paths(
        os.path.join(data_root, "color"), os.path.join(data_root, "pose")
    )

    # Camera intrinsics
    intrinsics_np = np.loadtxt(
        os.path.join(data_root, "intrinsic", "intrinsic_color.txt")
    ).astype(np.float32)[:3, :3]

    intrinsics = torch.from_numpy(intrinsics_np).to(device)

    with torch.no_grad():
        for fid, (image_path, pose_path) in enumerate(
            zip(image_paths, pose_paths)
        ):
            frame_id = fid + 1  # Frame ID starts from 1

            # Load image
            pil_img = ImageOps.exif_transpose(Image.open(image_path))
            image_np = np.array(pil_img).astype(np.float32)[None]

            image = torch.from_numpy(
                np.ascontiguousarray(image_np.transpose(0, 3, 1, 2))
            ).to(device)

            # Load pose
            extrinsics_np = np.loadtxt(pose_path).astype(np.float32)
            extrinsics = torch.from_numpy(extrinsics_np).to(device)

            # Run inference
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                predictions: MapDet3DOut = model(
                    images=[image],
                    intrinsics=[intrinsics],
                    extrinsics=[extrinsics],
                    frame_ids=[frame_id],
                )

            visualizer.process(
                cur_iter=frame_id,
                images=[image],
                sequence_names=["demo"],
                original_hw=[(image.shape[2], image.shape[3])],
                intrinsics=[intrinsics],
                extrinsics=[extrinsics],
                boxes3d=predictions.boxes3d,
                scores=predictions.scores,
                track_ids=predictions.track_ids,
                mesh_paths=[mesh_path],
            )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Map-Det3D demo.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/demo",
        help=(
            "Sequence root holding color/, pose/, intrinsic/ and mesh.ply "
            "(default: ./data/demo)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Demo."""
    args = parse_args()
    run(args.data_root)
