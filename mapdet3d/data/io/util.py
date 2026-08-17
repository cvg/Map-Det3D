"""Data I/O Utilities."""

from __future__ import annotations

import sys


def str_decode(str_bytes: bytes, encoding: None | str = None) -> str:
    """Decode to string from bytes.

    Args:
        str_bytes (bytes): Bytes to decode.
        encoding (None | str): Encoding to use. Defaults to None which is
            equivalent to sys.getdefaultencoding().

    Returns:
        str: Decoded string.
    """
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)
