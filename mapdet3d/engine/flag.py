"""Engine Flags."""

from absl import flags

from .parser import DEFINE_config_file

_CONFIG = DEFINE_config_file("config", method_name="get_config")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs per node")
_NODES = flags.DEFINE_integer("nodes", default=1, help="Number of nodes")
_WANDB = flags.DEFINE_bool(
    "wandb", default=False, help="If set, use Weights & Biases for logging."
)
_CKPT = flags.DEFINE_string("ckpt", default=None, help="Checkpoint path")
_RESUME = flags.DEFINE_bool("resume", default=False, help="Resume training")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)
_VIS = flags.DEFINE_bool(
    "vis",
    default=False,
    help="If set, running visualization using visualizer callback.",
)


__all__ = [
    "_CONFIG",
    "_GPUS",
    "_NODES",
    "_CKPT",
    "_RESUME",
    "_SHOW_CONFIG",
    "_WANDB",
    "_VIS",
]
