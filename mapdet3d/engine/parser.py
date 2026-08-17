"""Parser for config files that can be used with absl flags."""

from __future__ import annotations

import logging
import re
import sys
import traceback
from typing import Any

from absl import flags
from ml_collections import ConfigDict, FieldReference
from ml_collections.config_flags.config_flags import (
    _ConfigFlag,
    _ErrorConfig,
    _LockConfig,
)

from mapdet3d.config import copy_and_resolve_references
from mapdet3d.config.registry import get_config_by_name


class ConfigFileParser(flags.ArgumentParser):  # type: ignore
    """Parser for config files."""

    def __init__(
        self,
        name: str,
        lock_config: bool = True,
        method_name: str = "get_config",
    ) -> None:
        """Initializes the parser.

        Args:
            name (str): The name of the flag (e.g. config for --config flag)
            lock_config (bool, optional): Whether or not to lock the config.
                Defaults to True.
            method_name (str, optional): Name of the method to call in the
                config. Defaults to "get_config".
        """
        self.name = name
        self._lock_config = lock_config
        self.method_name = method_name

    def parse(  # pylint: disable=arguments-renamed
        self, path: str
    ) -> ConfigDict | _ErrorConfig:
        """Loads a config module from `path` and returns the `method_name()`.

        This implementation is based on the original ml_collections and
        modified to allow for a custom method name.

        If a colon is present in `path`, everything to the right of the first
        colon is passed to `method_name` as an argument. This allows the
        structure of what
        is returned to be modified, which is useful when performing complex
        hyperparameter sweeps.

        Args:
          path: string, path pointing to the config file to execute. May also
              contain a config_string argument, e.g. be of the form
              "config.py:some_configuration".

        Returns:
          Result of calling `method_name` in the specified module.
        """
        # This will be a 2 element list iff extra configuration args are
        # present.
        split_path = path.split(":", 1)

        try:
            config = get_config_by_name(
                split_path[0],
                *split_path[1:],
                method_name=self.method_name,
            )
            if config is None:
                logging.warning(
                    "%s:%s() returned None, did you forget a return "
                    "statement?",
                    path,
                    self.method_name,
                )
        except IOError as e:
            # Don't raise the error unless/until the config is
            # actually accessed.
            return _ErrorConfig(e)
        # Third party flags library catches TypeError and ValueError
        # and rethrows,
        # removing useful information unless it is added here (b/63877430):
        except (TypeError, ValueError) as e:
            error_trace = traceback.format_exc()
            raise type(e)(
                "Error whilst parsing config file:\n\n" + error_trace
            )

        if self._lock_config:
            _LockConfig(config)

        return config

    def flag_type(self) -> str:
        """Returns the type of the flag."""
        return "config object"


def DEFINE_config_file(  # pylint: disable=invalid-name
    name: str,
    default: str | None = None,
    help_string: str = "path to config file [.py |.yaml].",
    lock_config: bool = False,
    method_name: str = "get_config",
) -> flags.FlagHolder:  # type: ignore
    """Registers a new flag for a config file.

    Args:
        name (str): The name of the flag (e.g. config for --config flag)
        default (str | None, optional): Default Value. Defaults to None.
        help_string (str, optional): Help String.
            Defaults to "path to config file.".
        lock_config (bool, optional): Whether or note to lock the returned
            config. Defaults to False.
        method_name (str, optional): Name of the method to call in the config.

    Returns:
        flags.FlagHolder: Flag holder instance.
    """
    parser = ConfigFileParser(
        name=name, lock_config=lock_config, method_name=method_name
    )
    flag = _ConfigFlag(
        parser=parser,
        serializer=flags.ArgumentSerializer(),
        name=name,
        default=default,
        help_string=help_string,
        flag_values=flags.FLAGS,
    )

    # Get the module name for the frame at depth 1 in the call stack.
    module_name = sys._getframe(  # pylint: disable=protected-access
        1
    ).f_globals.get("__name__", None)
    module_name = sys.argv[0] if module_name == "__main__" else module_name
    return flags.DEFINE_flag(flag, flags.FLAGS, module_name=module_name)


def pprints_config(data: ConfigDict) -> str:
    """Converts a Config Dict into a string with a .yaml like structure.

    This function differs from __repr__ of ConfigDict in that it will not
    encode python classes using binary formats but just prints the __repr__
    of these classes.

    Args:
        data (ConfigDict): Configuration dict to convert to string

    Returns:
        str: A string representation of the ConfigDict
    """
    return _pprints_config(copy_and_resolve_references(data))


def _pprints_config(  # type: ignore
    data: Any, prefix: str = "", n_indents: int = 1
) -> str:
    """Converts a ConfigDict into a string with a YAML like structure.

    This is the recursive implementation of 'pprints_config' and will be called
    recursively for every element in the dict.

    This function differs from __repr__ of ConfigDict in that it will not
    encode python classes using binary formats but just prints the __repr__
    of these classes.

    Args:
        data (Any): Configuration dict or object to convert to
            string
        prefix (str): Prefix to print on each new line
        n_indents (int): Number of spaces to append for each nester property.

    Returns:
        str: A string representation of the ConfigDict
    """
    string_repr = ""
    if isinstance(data, FieldReference):
        data = data.get()

    if not isinstance(data, (dict, ConfigDict, list, tuple, dict)):
        return str(data)

    string_repr += "\n"

    if isinstance(data, (ConfigDict, dict)):
        for key in data:
            value = data[key]
            string_repr += (
                prefix
                + key
                + ": "
                + _pprints_config(value, prefix=prefix + " " * n_indents)
            ) + "\n"

    elif isinstance(data, (list, tuple)):
        for value in data:
            string_repr += prefix + "- "
            if isinstance(value, (ConfigDict, dict)):
                string_repr += "\n"

            string_repr += (
                _pprints_config(value, prefix=prefix + " " + " " * n_indents)
                + "\n"
            )
        string_repr += " \n"  # Add newline after list for better readability.

    # Clean up some formatting issues using regex. Could be done better
    string_repr = re.sub("\n\n+", "\n", string_repr)
    return re.sub("- +\n +", "- ", string_repr)
