"""Check if optional packages required by some modules are available."""

from functools import lru_cache
from importlib.util import find_spec

import torch
from packaging import version


@lru_cache()
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment."""
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:  # pragma: no cover
        return False


# io
H5PY_AVAILABLE = package_available("h5py")

# vision
MMCV_AVAILABLE = package_available("mmcv") or package_available("mmcv-full")
MMDET_AVAILABLE = package_available("mmdet")
MMSEG_AVAILABLE = package_available("mmseg")
DETECTRON2_AVAILABLE = package_available("detectron2")
TIMM_AVAILABLE = package_available("timm")
FVCORE_AVAILABLE = package_available("fvcore")

# datasets
WAYMO_AVAILABLE = package_available("waymo")
NUSCENES_AVAILABLE = package_available("nuscenes")
SCALABEL_AVAILABLE = package_available("scalabel")
BDD100K_AVAILABLE = package_available("bdd100k")

# visualization
OPENCV_AVAILABLE = package_available("cv2")
DASH_AVAILABLE = package_available("dash")
OPEN3D_AVAILABLE = package_available("open3d")
PLOTLY_AVAILABLE = package_available("plotly")

# logging
TENSORBOARD_AVAILABLE = package_available("tensorboardX") or package_available(
    "tensorboard"
)


def is_torch_tf32_available() -> bool:  # pragma: no cover
    """Check if torch TF32 is available.

    Returns:
        bool: True if torch TF32 is available.
    """
    return not (
        not torch.cuda.is_available()
        or torch.version.cuda is None
        or int(torch.version.cuda.split(".", maxsplit=1)[0]) < 11
        or version.parse(torch.__version__) < version.parse("1.7")
    )
