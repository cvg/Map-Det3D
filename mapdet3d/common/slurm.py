"""Slurm job submission.

Code adapted from:
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
"""

import os
import socket
import subprocess

import torch


def _find_free_port() -> str:
    """Find a free port on the current machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port: int) -> bool:
    """Check if a port is free on the current machine."""
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append("localhost")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def init_dist_slurm() -> None:
    """Initialize slurm distributed training environment."""
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    # WORLD_SIZE
    os.environ["WORLD_SIZE"] = str(ntasks)

    # use MASTER_ADDR in the environment variable if it already exists
    if "MASTER_ADDR" not in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ["MASTER_ADDR"] = addr

    # use MASTER_PORT in the environment variable if it already exists
    if "MASTER_PORT" not in os.environ:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ["MASTER_PORT"] = "29500"
        else:
            os.environ["MASTER_PORT"] = str(_find_free_port())

    # LOCAL RANK
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)

    # GLOBAL RANK
    os.environ["RANK"] = str(proc_id)
