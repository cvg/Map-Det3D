"""Config dict module."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import yaml
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict

from mapdet3d.common.named_tuple import get_all_keys, is_namedtuple
from mapdet3d.common.typing import ArgsType


# NOTE: Most of these functions need to deal with unknown parameters and are
# therefore not strictly typed
class FieldConfigDict(ConfigDict):  # type: ignore # pylint: disable=too-many-instance-attributes, line-too-long
    """A configuration dict which allows to access fields via dot notation.

    This class is a subclass of ConfigDict and overwrites the dot notation to
    return a FieldReference instead of a dict.

    For more information on the ConfigDict class, see:
        ml_collections.ConfigDict.

    Examples of using the ref and value mode:
        >>> config = FieldConfigDict({"a": 1, "b": 2})
        >>> type(config.a)
        <class 'ml_collections.field_reference.FieldReference'>
        >>> config.value_mode() # Set the config to return values
        >>> type(config.a)
        <class 'int'>
    """

    def __init__(  # type: ignore
        self,
        initial_dictionary: Mapping[str, Any] | None = None,
        type_safe: bool = True,
        convert_dict: bool = True,
    ):
        """Creates an instance of FieldConfigDict.

        Args:
          initial_dictionary: May be one of the following:

            1) dict. In this case, all values of initial_dictionary that are
            dictionaries are also be converted to ConfigDict. However,
            dictionaries within values of non-dict type are untouched.

            2) ConfigDict. In this case, all attributes are uncopied, and only
            the top-level object (self) is re-addressed. This is the same
            behavior as Python dict, list, and tuple.

            3) FrozenConfigDict. In this case, initial_dictionary is converted
            to a ConfigDict version of the initial dictionary for the
            FrozenConfigDict (reversing any mutability changes FrozenConfigDict
            made).

          type_safe: If set to True, once an attribute value is assigned, its
            type cannot be overridden without .ignore_type() context manager.

          convert_dict: If set to True, all dict used as value in the
            ConfigDict will automatically be converted to ConfigDict.
        """
        super().__init__(initial_dictionary, type_safe, convert_dict)
        object.__setattr__(self, "_return_refs", True)

    @classmethod
    def from_yaml(cls, path: str) -> FieldConfigDict:
        """Creates a config from a .yaml file.

        Args:
            path: The path to the .yaml file that should be loaded.
        """
        return cls(
            yaml.load(
                open(path, "r", encoding="utf-8"), Loader=yaml.UnsafeLoader
            )
        )

    def to_yaml(self, **kwargs: ArgsType) -> str:
        """Returns a YAML representation of the object.

        ConfigDict serializes types of fields as well as the values of fields
        themselves. Deserializing the YAML representation hence requires using
        YAML's UnsafeLoader:

        ```
        yaml.load(cfg.to_yaml(), Loader=yaml.UnsafeLoader)
        ```

        or equivalently:

        ```
        yaml.unsafe_load(cfg.to_yaml())
        ```

        Please see the PyYAML documentation and https://msg.pyyaml.org/load
        for more details on the consequences of this.

        Args:
          **kwargs: Keyword arguments for yaml.dump.

        Returns:
          YAML representation of the object.
        """
        return copy_and_resolve_references(self.value_mode()).to_yaml(**kwargs)

    def dump(self, output_path: str) -> None:
        """Writes the config to a .yaml file.

        Args:
            output_path: The path to the output file.
        """
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(self.to_yaml())

    def set_ref_mode(self, ref_mode: bool) -> None:
        """Sets the config to return references instead of values."""

        def _rec_resolve_iterable(  # type: ignore
            iterable: Iterable[Any], cfgs: list[FieldConfigDict]
        ) -> None:
            """Recursively adds all FieldConfigDicts to a list."""
            for item in iterable:
                if isinstance(item, FieldConfigDict):
                    cfgs.append(item)
                elif isinstance(item, (list, tuple)):
                    _rec_resolve_iterable(item, cfgs)
                elif isinstance(item, (dict, ConfigDict)):
                    _rec_resolve_iterable(item.values(), cfgs)

        # Update value of this dict
        object.__setattr__(self, "_return_refs", ref_mode)

        # propagate to sub configs
        for value in self.values():
            if isinstance(value, FieldConfigDict):
                value = value.value_mode()
            elif isinstance(value, (list, tuple, ConfigDict, dict)):
                cfgs: list[FieldConfigDict] = []
                _rec_resolve_iterable(value, cfgs)
                for cfg in cfgs:
                    cfg.set_ref_mode(ref_mode)

    def ref_mode(self) -> FieldConfigDict:
        """Sets the config to return references instead of values."""
        self.set_ref_mode(True)
        return self

    def value_mode(self) -> FieldConfigDict:
        """Sets the config to return values instead of references."""
        self.set_ref_mode(False)
        return self

    def __getitem__(self, key: str) -> FieldReference:
        """Returns the reference for the given key."""
        # private properties are always returned as values

        if self._return_refs:
            try:
                return super().get_ref(key)
            except ValueError:
                pass
        return super().__getitem__(key)


def resolve_class_name(clazz: type | Callable[Any, Any] | str) -> str:  # type: ignore # pylint: disable=line-too-long
    """Resolves the full class name of the given class object, callable or str.

    This function takes a class object and returns the class name as a string.
    Args:
        clazz (type | Callable[[Any], Any] | str): The object to resolve
            the full path of.

    Returns:
        str: The full path of the given object.

    Raises:
        ValueError: If the given object is a lambda function.

    Examples:
        >>> class MyClass: pass
        >>> resolve_class_name(MyClass)
        '__main__.MyClass'
        >>> resolve_class_name("path.to.MyClass")
        'path.to.MyClass'
        >>> def my_function(): pass
        >>> resolve_class_name(my_function)
        '__main__.my_function'
        >>> class MyClass:
        >>>     @classmethod
        >>>     def build(cls): pass
        >>> resolve_class_name(MyClass.build)
        '__main__.MyClass.build'
    """
    if isinstance(clazz, str):
        return clazz

    if clazz.__name__ == "lambda":
        raise ValueError(
            "Resolving the full class path of lambda functions"
            "is not supported. Please define a inline function instead."
        )

    # Classmethods resolve to the class they are bound to, since __module__
    # points to the class that defines them, which may be a base class.
    owner = getattr(clazz, "__self__", None)
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__name__}.{clazz.__name__}"

    module = clazz.__module__
    if module is None or module == str.__class__.__module__:
        return clazz.__name__
    return module + "." + clazz.__name__


def resolve_class_path(class_path: str) -> Any:
    """Resolves the object a full class path points to.

    Attributes are resolved on top of the longest importable module prefix,
    which allows paths to nested objects such as classmethods, e.g.
    'my.module.MyClass.from_pretrained'.

    Args:
        class_path (str): The full path of the object to resolve.

    Returns:
        Any: The object the given path points to.

    Raises:
        ImportError: If no module prefix of the given path is importable.
    """
    parts = class_path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # Only keep looking if this prefix is not a module. Errors raised
            # by missing dependencies of an existing module are propagated.
            if e.name is None or not module_name.startswith(e.name):
                raise
            continue

        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj

    raise ImportError(f"Could not resolve class path: {class_path}")


def class_config(
    clazz: type | Callable[Any, Any] | str,  # type: ignore
    **kwargs: ArgsType,
) -> ConfigDict:
    """Creates a configuration which can be instantiated as a class.

    This function creates a configuration dict which can be passed to
    'instantiate_classes' to create a instance of the given class or functor.

    Example:
    >>> class_cfg_obj = class_config("your.module.Module", arg1="arg1", arg2=2)
    >>> print(class_cfg_obj)
    >>> # Prints :
    >>> class_path: your.module.Module
    >>> init_args:
    >>>   arg1: arg1
    >>>   arg2: 2

    >>> # instantiate object
    >>> inst_obj = instantiate_classes(class_cfg_obj)
    >>> print(type(inst_obj)) # -> Will print <class 'your.module.Module'>

    >>> # Example by directly passing objects:
    >>> class MyClass:
    >>>     def __init__(self, name: str, age: int):
    >>>         self.name = name
    >>>         self.age = age
    >>> class_cfg_obj = class_config(MyClass, name="John", age= 25)

    >>> print(class_cfg_obj)
    >>> # Prints :
    >>> class_path: __main__.MyClass
    >>> init_args:
    >>>   name: John
    >>>   age: 25

    >>> # instantiate object
    >>> inst_obj = instantiate_classes(class_cfg_obj)
    >>> print(type(inst_obj)) # -> Will print <class '__main__.MyClass'>
    >>> print(inst_obj.name) # -> Will print John

    Args:
        clazz (type | Callable[[Any], Any] | str): class type or functor or
            class string path.
        **kwargs (ArgsType): Kwargs to pass to the class constructor.

    Returns:
        ConfigDict: _description_
    """
    class_path = resolve_class_name(clazz)
    if class_path is None or len(kwargs) == 0:
        return ConfigDict({"class_path": class_path})
    return ConfigDict(
        {"class_path": class_path, "init_args": ConfigDict(kwargs)}
    )


def delay_instantiation(instantiable: ConfigDict) -> ConfigDict:
    """Delays the instantiation of the given configuration object.

    This is a somewhat hacky way to delay the initialization of the optimizer
    configuration object. It works by replacing the class_path with _class_path
    which basically tells the instantiate_classes function to not instantiate
    the class. Instead, it returns a function that can be called to instantiate
    the class

    Args:
        instantiable (ConfigDict): The configuration object to delay the
            instantiation of.
    """
    instantiable["_class_path"] = instantiable["class_path"]
    del instantiable["class_path"]

    return class_config(DelayedInstantiator, instantiable=instantiable)


class DelayedInstantiator:
    """Class that delays the instantiation of the given configuration object.

    This is a somewhat hacky way to delay the initialization of the optimizer
    configuration object. It works by replacing the class_path with _class_path
    which basically tells the instantiate_classes function to not instantiate
    the class. Instead, it returns a function that can be called to instantiate
    the class.

    Args:
        instantiable (ConfigDict): The configuration object to delay the
            instantiation of.
    """

    def __init__(self, instantiable: ConfigDict) -> None:
        """Instantiates the DelayedInstantiator."""
        self.instantiable = instantiable

    def __call__(self, **kwargs: ArgsType) -> Any:  # type: ignore
        """Instantiates the configuration object."""
        instantiable = class_config(
            self.instantiable["_class_path"],
            **self.instantiable.get("init_args", {}),
        )

        return instantiate_classes(instantiable, **kwargs)


def instantiate_classes(data: ConfigDict | FieldReference, **kwargs: ArgsType) -> ConfigDict | Any:  # type: ignore # pylint: disable=line-too-long
    """Instantiates all classes in a given ConfigDict.

    This function iterates over the configuration data and instantiates
    all classes. Class defintions are provided by a config dict that has
    the following structure:

    {
        'data_path': 'path.to.my.class.Class',
        'init_args': ConfigDict(
            {
                'arg1': 'value1',
                'arg2': 'value2',
            }
        )
    }

    Args:
        data (ConfigDict | FieldReference): The general configuration object.
        **kwargs (ArgsType): Additional arguments to pass to the class
            constructor.

    Returns:
        ConfigDict | Any: The instantiated objects.
    """
    if isinstance(data, FieldReference):  # De-Reference the field reference
        data = data.get()

    assert isinstance(data, ConfigDict), "Data must be a ConfigDict."

    if isinstance(data, FieldConfigDict):
        data.value_mode()  # make sure data is in value mode

    if len(kwargs) > 0:
        if "init_args" not in data:
            data["init_args"] = ConfigDict(kwargs)
        else:
            for k, v in kwargs.items():
                data["init_args"][k] = v

    resolved_data = copy_and_resolve_references(data)
    instantiated_objects = _instantiate_classes(resolved_data)
    return instantiated_objects


def copy_and_resolve_references(  # type: ignore
    data: Any, visit_map: dict[int, Any] | None = None
) -> Any:
    """Returns a ConfigDict copy with FieldReferences replaced by values.

    If the object is a FrozenConfigDict, the copy returned is also a
    FrozenConfigDict. However, note that FrozenConfigDict should already have
    FieldReferences resolved to values, so this method effectively produces
    a deep copy.

    Note: This method is overwritten from the ConfigDict class and allows to
    also resolve FieldReferences in list, tuple and dict.

    Args:
        data (Any): object to copy.
        visit_map (dict[int, Any]): A mapping from ConfigDict object ids to
            their copy. Method is recursive in nature, and it will call
            "copy_and_resolve_references(visit_map)" on each encountered
            object, unless it is already in visit_map.

    Returns:
        Any: ConfigDict copy with previous FieldReferences replaced by values.
    """
    if isinstance(data, FieldReference):
        data = data.get()

    if is_namedtuple(data):
        return type(data)(
            **{
                key: copy_and_resolve_references(getattr(data, key))
                for key in get_all_keys(data)
            }
        )

    if isinstance(data, (list, tuple)):
        return type(data)(
            copy_and_resolve_references(value, visit_map) for value in data
        )

    if isinstance(data, dict):
        return {
            k: copy_and_resolve_references(v, visit_map)
            for k, v in data.items()
        }

    if not isinstance(data, ConfigDict):
        return data

    visit_map = visit_map or {}
    config_dict = ConfigDict()

    # copy attributes
    super(ConfigDict, config_dict).__setattr__(
        "_convert_dict", config_dict.convert_dict
    )
    visit_map[id(config_dict)] = config_dict

    for key, value in data._fields.items():
        if isinstance(value, FieldReference):
            value = value.get()

        if id(value) in visit_map:
            value = visit_map[id(value)]

        elif isinstance(value, ConfigDict):
            value = copy_and_resolve_references(value, visit_map)

        elif is_namedtuple(value):
            value = type(value)(
                **{
                    key: copy_and_resolve_references(getattr(value, key))
                    for key in get_all_keys(value)
                }
            )

        elif isinstance(value, (list, tuple)):
            value = type(value)(
                copy_and_resolve_references(v, visit_map) for v in value
            )

        elif isinstance(value, dict):
            value = {
                k: copy_and_resolve_references(v, visit_map)
                for k, v in value.items()
            }

        if isinstance(data, FrozenConfigDict):
            config_dict._frozen_setattr(  # pylint:disable=protected-access
                key, value
            )
        else:
            config_dict[key] = value

    # copy attributes
    super(ConfigDict, config_dict).__setattr__("_locked", data.is_locked)
    super(ConfigDict, config_dict).__setattr__("_type_safe", data.is_type_safe)
    return config_dict


def _get_index(data: Any) -> Any:  # type: ignore
    """Internal function to generate a Sequence of indexes for a given object.

    Example:
    >>> [data[idx] for idx in _get_index(data)]

    Args:
        data (Any): The data entry to get an index for.

    Returns:
        Any: Iterable that can be used to index the data entry using e.g.
            [data[idx] for idx in _get_index(data)]
    """
    if isinstance(data, (list, tuple)):
        return range(len(data))
    return data


def _instantiate_classes(data: Any) -> Any:  # type: ignore
    """Instantiates all classes in a given data.

    Data could be ConfigDict, FieldReference, tuple, list or dict.

    This is the recursive implementation of the 'instantiate_classes'.

    This function iterates over the configuration data and instantiates
    all classes. Class defintions are provided by a config dict that has
    the following structure:

    {
        'data_path': 'path.to.my.class.Class',
        'init_args': ConfigDict(
            {
                'arg1': 'value1',
                'arg2': 'value2',
            }
        )
    }

    Args:
        data (Any): The general configuration object.

    Returns:
        Any: The ConfigDict with all classes intialized. Or, if the top level
        element is a class config, the returned element will be the
        instantiated class.
    """
    if isinstance(data, FieldReference):
        data = data.get()

    if not isinstance(data, (ConfigDict, dict, list, tuple)):
        return data

    for key in _get_index(data):
        value = data[key]

        if isinstance(value, FieldReference):
            value = value.get()

        if isinstance(value, (ConfigDict, dict)):
            if isinstance(data, ConfigDict):
                with data.ignore_type():
                    data[key] = _instantiate_classes(value)
            else:
                data[key] = _instantiate_classes(value)

        elif is_namedtuple(value):
            if isinstance(data, ConfigDict):
                with data.ignore_type():
                    data[key] = type(value)(
                        **{
                            key: _instantiate_classes(getattr(value, key))
                            for key in get_all_keys(value)
                        }
                    )
            else:
                data[key] = type(value)(
                    **{
                        key: _instantiate_classes(getattr(value, key))
                        for key in get_all_keys(value)
                    }
                )

        elif isinstance(value, (list, tuple)):
            if isinstance(data, ConfigDict):
                with data.ignore_type():
                    data[key] = type(value)(
                        _instantiate_classes(value[idx])
                        for idx in range(len(value))
                    )
            else:
                data[key] = type(value)(
                    _instantiate_classes(value[idx])
                    for idx in range(len(value))
                )

    # Instantiate classs
    if "class_path" in data and not isinstance(data["class_path"], ConfigDict):
        init_args = data.get("init_args", {})

        # Convert ConfigDict to normal dictionary
        if isinstance(init_args, ConfigDict):
            init_args = init_args.to_dict()

        # Instantiate class
        clazz = resolve_class_path(data["class_path"])(**init_args)
        return clazz

    return data
