"""Convert model weights for release."""

from __future__ import annotations

import argparse
import hashlib
import os

import torch


def save_weights_with_hash(
    state_dict: dict[str, torch.Tensor],
    path: str,
    filename: str,
    digits: int = 6,
) -> None:
    """Saves the model weights and append a 6-digit hash to the filename.

    Args:
        state_dict (dict[str, torch.Tensor]): The model weights to save.
        path (str): The directory path to save the model.
        filename (str): The filename to save the model.
        digits (int, optional): The number of digits to use for the hash.
            Defaults to 6.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "wb") as f:
        torch.save(state_dict, f)

    # Create a hash of the file
    sha256_hash = hashlib.sha256()
    with open(os.path.join(path, filename), "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    # Get the hexadecimal representation of the hash
    short_hash = sha256_hash.hexdigest()[:digits]
    os.rename(
        os.path.join(path, filename),
        os.path.join(path, f"{filename}_{short_hash}.pt"),
    )


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Save trained model checkpoint with a filename hash."
    )
    parser.add_argument("path", type=str, help="The path to the checkpoint.")
    parser.add_argument(
        "--outdir",
        type=str,
        help="The path to output the model.",
        default="./work_dir/release",
    )
    parser.add_argument(
        "--name", type=str, help="The base name of the released file."
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.path, weights_only=False, map_location=torch.device("cpu")
    )
    state_dict = {"state_dict": checkpoint["state_dict"]}

    save_weights_with_hash(state_dict, args.outdir, args.name)


if __name__ == "__main__":
    main()
