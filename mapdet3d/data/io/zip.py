"""Zip data backend.

This backend works with filepaths pointing to valid Zip files. We assume that
the given Zip file contains the whole dataset associated to this backend.
"""

from __future__ import annotations

import os
import zipfile
from typing import Literal
from zipfile import ZipFile

from .base import DataBackend


class ZipBackend(DataBackend):
    """Backend for loading data from Zip files.

    This backend works with filepaths pointing to valid Zip files. We assume
    that the given Zip file contains the whole dataset associated to this
    backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.db_cache: dict[str, tuple[ZipFile, str]] = {}

    @staticmethod
    def _get_zip_path(
        filepath: str, allow_omitted_ext: bool = True
    ) -> tuple[str, list[str]]:
        """Get .zip path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.zip/key1/key2'
            allow_omitted_ext (bool, optional): Whether to allow omitted
                extension, in which case the backend will try to append
                '.zip' to the filepath. Defaults to True.

        Returns:
            tuple[str, list[str]]: The .hdf5 path and the keys to retrieve.

        Examples:
            >>> _get_zip_path("path/to/file.zip/key1/key2")
            ("path/to/file.zip", ["key2", "key1"])
            >>> _get_zip_path("path/to/file/key1/key2", True)
            ("path/to/file.zip", ["key2", "key1"]) # if file.hdf5 exists and
                                                    # is a valid hdf5 file
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while True:
            if filepath.endswith(".zip") or filepath == "":
                break
            if allow_omitted_ext and zipfile.is_zipfile(filepath + ".zip"):
                filepath = filepath + ".zip"
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
        zip_path, keys = self._get_zip_path(filepath)
        if not os.path.exists(zip_path):
            return False
        file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        return url in file.namelist()

    def set(
        self, filepath: str, content: bytes, mode: Literal["w", "a"] = "w"
    ) -> None:
        """Write the file content to the zip file.

        Args:
            filepath: path/to/file.zip/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
                within another group key1, for example.
            mode: Mode to open the file in. "w" for writing a file, "a" for
                appending to existing file.

        Raises:
            ValueError: If filepath is not a valid .zip file
            NotImplementedError: If the method is not implemented.
        """
        if ".zip" not in filepath:
            raise ValueError(f"{filepath} not a valid .zip filepath!")

        zip_path, keys = self._get_zip_path(filepath)
        zip_file = self._get_client(zip_path, mode)
        url = "/".join(reversed(keys))
        zip_file.writestr(url, content)

    def _get_client(
        self, zip_path: str, mode: Literal["r", "w", "a", "x"]
    ) -> ZipFile:
        """Get Zip client from path.

        Args:
            zip_path (str): Path to Zip file.
            mode (str): Mode to open the file in.

        Returns:
            ZipFile: the hdf5 file.
        """
        assert len(mode) == 1, "Mode must be a single character for zip file."
        if zip_path not in self.db_cache:
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            client = ZipFile(zip_path, mode)
            self.db_cache[zip_path] = (client, mode)
        else:
            client, current_mode = self.db_cache[zip_path]
            if current_mode != mode:
                client.close()
                client = ZipFile(  # pylint:disable=consider-using-with
                    zip_path, mode
                )
                self.db_cache[zip_path] = (client, mode)
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an Zip path
                together with the relative path inside it, e.g.: "/path/to/
                file.zip/key/subkey/data". If no .zip given inside filepath,
                the function will search for the first .zip file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.zip.

        Raises:
            ZipFileNotFoundError: If no suitable file exists.
            OSError: If the file cannot be opened.
            ValueError: If key not found inside zip file.

        Returns:
            bytes: The file content in bytes
        """
        zip_path, keys = self._get_zip_path(filepath)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"Corresponding zip file not found:" f" {filepath}"
            )
        zip_file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        try:
            with zip_file.open(url) as zf:
                content = zf.read()
        except KeyError as e:
            raise ValueError(f"Value '{url}' not found in {zip_path}!") from e
        return bytes(content)

    def listdir(self, filepath: str) -> list[str]:
        """List all files in the given directory.

        Args:
            filepath (str): The path to the directory.

        Returns:
            list[str]: List of all files in the given directory.
        """
        zip_path, keys = self._get_zip_path(filepath)
        zip_file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        files = [
            os.path.basename(key)
            for key in zip_file.namelist()
            if key.startswith(url) and os.path.basename(key) != ""
        ]
        return sorted(files)

    def isfile(self, filepath: str) -> bool:
        """Check if filepath is a file.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        zip_path, keys = self._get_zip_path(filepath)
        if not os.path.exists(zip_path):
            return False
        zip_file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        return url in zip_file.namelist()

    def close(self) -> None:
        """Close all opened Zip files."""
        for client, _ in self.db_cache.values():
            client.close()
        self.db_cache = {}
