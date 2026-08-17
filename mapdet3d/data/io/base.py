"""Backends for the data types a dataset of interest is saved in.

Those can be used to load data from diverse storage backends, e.g. from HDF5
files which are more suitable for data centers. The naive backend is the
FileBackend, which loads from / saves to file naively.
"""

from abc import abstractmethod
from typing import Literal


class DataBackend:
    """Abstract class of storage backends.

    All backends need to implement three functions: get(), set() and exists().
    get() reads the file as a byte stream and set() writes a byte stream to a
    file. exists() checks if a certain filepath exists.
    """

    @abstractmethod
    def set(
        self, filepath: str, content: bytes, mode: Literal["w", "a"]
    ) -> None:
        """Set the file content at the given filepath.

        Args:
            filepath (str): The filepath to store the data at.
            content (bytes): The content to store as bytes.
            mode (str): The mode to open the file in.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        """Get the file content at the given filepath as bytes.

        Args:
            filepath (str): The filepath to retrieve the data from."

        Returns:
            bytes: The content of the file as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): The filepath to check.

        Returns:
            bool: True if the filepath exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def isfile(self, filepath: str) -> bool:
        """Check if filepath is a file.

        Args:
            filepath (str): The filepath to check.

        Returns:
            bool: True if the filepath is a file, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def listdir(self, filepath: str) -> list[str]:
        """List all files in a directory.

        Args:
            filepath (str): The directory to list.

        Returns:
            list[str]: A list of all files in the directory.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close all opened files in the backend."""
        raise NotImplementedError
