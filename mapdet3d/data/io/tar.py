"""Tar data backend.

This backend works with filepaths pointing to files inside tar archives.
Similar to HDF5Backend but for tar files.
"""

from __future__ import annotations

import os
import tarfile
from typing import Literal

from .base import DataBackend


class TarBackend(DataBackend):
    """Backend for loading data from tar files.

    This backend works with filepaths pointing to files inside tar archives.
    The filepath format should be: 'path/to/file.tar/internal/path/to/file'

    Examples:
        >>> backend = TarBackend()
        >>> data = backend.get("dataset.tar/images/img001.png")
        >>> files = backend.listdir("dataset.tar/images")
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.tar_cache: dict[str, tarfile.TarFile] = {}

    @staticmethod
    def _get_tar_path(
        filepath: str, allow_omitted_ext: bool = True
    ) -> tuple[str, str]:
        """Get .tar path and internal path from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have format: 'path/to/file.tar/internal/path'
            allow_omitted_ext (bool, optional): Whether to allow omitted
                extension, in which case the backend will try to append
                '.tar' to the filepath. Defaults to True.

        Returns:
            tuple[str, str]: The .tar path and the internal path.

        Examples:
            >>> TarBackend._get_tar_path("path/to/file.tar/key1/key2")
            ("path/to/file.tar", "key1/key2")
            >>> TarBackend._get_tar_path("path/to/file/key1/key2", True)
            ("path/to/file.tar", "key1/key2")  # if file.tar exists
        """
        filepath_as_list = filepath.split("/")
        internal_parts = []

        tar_path = filepath
        while True:
            if tar_path.endswith(".tar") or tar_path == "":
                break
            if allow_omitted_ext and os.path.exists(tar_path + ".tar"):
                tar_path = tar_path + ".tar"
                break
            if internal_parts or not os.path.exists(tar_path):
                internal_parts.insert(0, filepath_as_list.pop())
                tar_path = "/".join(filepath_as_list)
            else:
                break

        internal_path = "/".join(internal_parts)
        return tar_path, internal_path

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file inside tar.

        Returns:
            bool: True if file exists, False otherwise.
        """
        tar_path, internal_path = self._get_tar_path(filepath)
        if not os.path.exists(tar_path):
            return False

        try:
            tar_file = self._get_client(tar_path)
            try:
                tar_file.getmember(internal_path)
                return True
            except KeyError:
                return False
        except (tarfile.TarError, FileNotFoundError):
            return False

    def _get_client(self, tar_path: str) -> tarfile.TarFile:
        """Get TarFile client from path.

        Args:
            tar_path (str): Path to tar file.

        Returns:
            tarfile.TarFile: The opened tar file.
        """
        if tar_path not in self.tar_cache:
            client = tarfile.open(tar_path, mode="r:*")
            self.tar_cache[tar_path] = client
        return self.tar_cache[tar_path]

    def get(self, filepath: str) -> bytes:
        """Get file content as bytes.

        Args:
            filepath (str): The path to the file. It consists of a tar path
                together with the internal path, e.g.: "/path/to/file.tar/
                internal/path/data.json".

        Raises:
            FileNotFoundError: If tar file doesn't exist.
            ValueError: If internal path not found inside tar.

        Returns:
            bytes: The file content in bytes
        """
        tar_path, internal_path = self._get_tar_path(filepath)

        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")

        tar_file = self._get_client(tar_path)

        try:
            member = tar_file.getmember(internal_path)
            if member.isdir():
                raise ValueError(f"{internal_path} is a directory, not a file")

            file_obj = tar_file.extractfile(member)
            if file_obj is None:
                raise ValueError(f"Cannot extract file: {internal_path}")

            return file_obj.read()

        except KeyError:
            raise ValueError(f"Path {internal_path} not found in {tar_path}")

    def isfile(self, filepath: str) -> bool:
        """Check if filepath is a file (not a directory).

        Args:
            filepath (str): Path to file inside tar.

        Raises:
            FileNotFoundError: If tar file doesn't exist.
            ValueError: If path not found inside tar.

        Returns:
            bool: True if it's a file, False if it's a directory.
        """
        tar_path, internal_path = self._get_tar_path(filepath)

        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")

        tar_file = self._get_client(tar_path)

        try:
            member = tar_file.getmember(internal_path)
            return member.isfile()
        except KeyError:
            raise ValueError(f"Path {internal_path} not found in {tar_path}")

    def listdir(self, filepath: str) -> list[str]:
        """List all files and directories in the given path.

        Args:
            filepath (str): Path to directory inside tar.

        Raises:
            FileNotFoundError: If tar file doesn't exist.
            ValueError: If path not found or is not a directory.

        Returns:
            list[str]: List of file/directory names (not full paths).
        """
        tar_path, internal_path = self._get_tar_path(filepath)

        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")

        tar_file = self._get_client(tar_path)

        # Normalize path
        internal_path = internal_path.rstrip("/")
        if internal_path and not internal_path.endswith("/"):
            prefix = internal_path + "/"
        else:
            prefix = internal_path

        # Collect immediate children
        children = set()
        for member in tar_file.getmembers():
            if member.name.startswith(prefix):
                # Get relative path
                relative = member.name[len(prefix) :]

                # Only get immediate children
                if relative and "/" not in relative.rstrip("/"):
                    children.add(relative.rstrip("/"))
                elif "/" in relative:
                    # Add the directory name only
                    dir_name = relative.split("/")[0]
                    children.add(dir_name)

        return sorted(children)

    def set(
        self, filepath: str, content: bytes, mode: Literal["w", "a"] = "a"
    ) -> None:
        """Set file content (not implemented for tar files).

        Tar files are typically read-only. Use tar command line tools
        to create/modify tar files.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "Writing to tar files is not supported. "
            "Use tar command line tools to create tar archives."
        )

    def close(self) -> None:
        """Close all opened tar files."""
        for tar_file in self.tar_cache.values():
            tar_file.close()
        self.tar_cache.clear()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
