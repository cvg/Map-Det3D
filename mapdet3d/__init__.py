"""Detect Anything in 4D."""

import logging

__version__ = "0.0.0"

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# if root logger has handlers, propagate messages up and let root logger
# process them
if not _root_logger.hasHandlers():  # pragma: no cover
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False
