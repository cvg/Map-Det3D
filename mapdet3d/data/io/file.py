"""Standard backend for local files on a hard drive.

This backends loads data from and saves data to the local hard drive.
"""

import os
from typing import Literal

from .base import DataBackend


class FileBackend(DataBackend):
    """Raw file from hard disk data backend."""

    def isfile(self, filepath: str) -> bool:
        """Check if filepath is a file.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        return os.path.isfile(filepath)

    def listdir(self, filepath: str) -> list[str]:
        """List all files in the directory.

        Args:
            filepath (str): Path to file.

        Returns:
            list[str]: List of all files in the directory.
        """
        return sorted(os.listdir(filepath))

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        return os.path.exists(filepath)

    def set(
        self, filepath: str, content: bytes, mode: Literal["w", "a"] = "w"
    ) -> None:
        """Write the file content to disk.

        Args:
            filepath (str): Path to file.
            content (bytes): Content to write in bytes.
            mode (Literal["w", "a"], optional): Overwrite or append mode.
                Defaults to "w".
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mode_binary: Literal["wb", "ab"] = "wb" if mode == "w" else "ab"
        with open(filepath, mode_binary) as f:
            f.write(content)

    def get(self, filepath: str) -> bytes:
        """Get file content as bytes.

        Args:
            filepath (str): Path to file.

        Raises:
            FileNotFoundError: If filepath does not exist.

        Returns:
            bytes: File content as bytes.
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"File not found:" f" {filepath}")
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf

    def close(self) -> None:
        """No need to close manually."""
