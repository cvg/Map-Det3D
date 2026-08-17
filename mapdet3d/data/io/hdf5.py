"""Hdf5 data backend.

This backend works with filepaths pointing to valid HDF5 files. We assume that
the given HDF5 file contains the whole dataset associated to this backend.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from mapdet3d.common.imports import H5PY_AVAILABLE

from .base import DataBackend

if H5PY_AVAILABLE:
    import h5py
    from h5py import File
else:
    raise ImportError("Please install h5py to enable HDF5Backend.")


class HDF5Backend(DataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with filepaths pointing to valid HDF5 files. We assume
    that the given HDF5 file contains the whole dataset associated to this
    backend.

    You can use the provided script at to_hdf5.py to convert your dataset to
    the expected hdf5 format before using this backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        if not H5PY_AVAILABLE:
            raise ImportError("Please install h5py to enable HDF5Backend.")
        self.db_cache: dict[str, File] = {}

    @staticmethod
    def _get_hdf5_path(
        filepath: str, allow_omitted_ext: bool = True
    ) -> tuple[str, list[str]]:
        """Get .hdf5 path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.hdf5/key1/key2'
            allow_omitted_ext (bool, optional): Whether to allow omitted
                extension, in which case the backend will try to append
                '.hdf5' to the filepath. Defaults to True.

        Returns:
            tuple[str, list[str]]: The .hdf5 path and the keys to retrieve.

        Examples:
            >>> HDF5Backend._get_hdf5_path("path/to/file.hdf5/key1/key2")
            ("path/to/file.hdf5", ["key2", "key1"])
            >>> HDF5Backend._get_hdf5_path("path/to/file/key1/key2", True)
            ("path/to/file.hdf5", ["key2", "key1"]) # if file.hdf5 exists and
                                                    # is a valid hdf5 file
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while True:
            if filepath.endswith(".hdf5") or filepath == "":
                break
            if allow_omitted_ext and h5py.is_hdf5(filepath + ".hdf5"):
                filepath = filepath + ".hdf5"
                break
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)
        if not os.path.exists(hdf5_path):
            return False
        value_buf = self._get_client(hdf5_path, "r")

        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                return False
        return True

    def set(
        self, filepath: str, content: bytes, mode: Literal["w", "a"] = "a"
    ) -> None:
        """Set the file content.

        Args:
            filepath: path/to/file.hdf5/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
                within another group key1, for example.
            mode: "w" to overwrite the file, "a" to append to it.

        Raises:
            ValueError: If filepath is not a valid .hdf5 file
        """
        if ".hdf5" not in filepath:
            raise ValueError(f"{filepath} not a valid .hdf5 filepath!")
        hdf5_path, keys_str = filepath.split(".hdf5")
        key_list = keys_str.split("/")
        file = self._get_client(hdf5_path + ".hdf5", mode)
        if len(key_list) > 1:
            group_str = "/".join(key_list[:-1])
            if group_str == "":
                group_str = "/"

            group = file[group_str]
            key = key_list[-1]
            group.create_dataset(
                key, data=np.frombuffer(content, dtype="uint8")
            )

    def _get_client(self, hdf5_path: str, mode: str) -> File:
        """Get HDF5 client from path.

        Args:
            hdf5_path (str): Path to HDF5 file.
            mode (str): Mode to open the file in.

        Returns:
            File: the hdf5 file.
        """
        if hdf5_path not in self.db_cache:
            client = File(hdf5_path, mode, swmr=True, libver="latest")
            self.db_cache[hdf5_path] = [client, mode]
        else:
            client, current_mode = self.db_cache[hdf5_path]
            if current_mode != mode:
                client.close()
                client = File(hdf5_path, mode, swmr=True, libver="latest")
                self.db_cache[hdf5_path] = [client, mode]
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an HDF5 path
                together with the relative path inside it, e.g.: "/path/to/
                file.hdf5/key/subkey/data". If no .hdf5 given inside filepath,
                the function will search for the first .hdf5 file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.hdf5.

        Raises:
            FileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            bytes: The file content in bytes
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )
        value_buf = self._get_client(hdf5_path, "r")
        url = "/".join(reversed(keys))
        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                raise ValueError(f"Value {url} not found in {hdf5_path}!")

        return bytes(value_buf[()])

    def isfile(self, filepath: str) -> bool:
        """Check if filepath is a file.

        Args:
            filepath (str): Path to file.

        Raises:
            FileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )
        value_buf = self._get_client(hdf5_path, "r")
        url = "/".join(reversed(keys))
        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                raise ValueError(f"Value {url} not found in {hdf5_path}!")
        return not isinstance(value_buf, h5py.Group)

    def listdir(self, filepath: str) -> list[str]:
        """List all files in the given directory.

        Args:
            filepath (str): Path to directory.

        Raises:
            FileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            list[str]: List of files in the given directory.
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )
        value_buf = self._get_client(hdf5_path, "r")
        url = "/".join(reversed(keys))
        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                raise ValueError(f"Value {url} not found in {hdf5_path}!")
        if not isinstance(value_buf, h5py.Group):
            raise ValueError(f"Value {url} is not a group in {hdf5_path}!")

        return sorted(list(value_buf.keys()))

    def close(self) -> None:
        """Close all opened HDF5 files."""
        for client, _ in self.db_cache.values():
            client.close()
        self.db_cache.clear()
