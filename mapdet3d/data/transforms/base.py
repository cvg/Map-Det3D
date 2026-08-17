"""Basic data augmentation class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar, no_type_check

import torch

from mapdet3d.common.dict import get_dict_nested, set_dict_nested
from mapdet3d.data.typing import DictData

TFunctor = TypeVar("TFunctor", bound=object)  # pylint: disable=invalid-name
TransformFunction = Callable[[list[DictData]], list[DictData]]


class Transform:
    """Transforms Decorator.

    This class stores which `in_keys` are input to a transformation function
    and which `out_keys` are overwritten in the data dictionary by the output
    of this transformation.
    Nested keys in the data dictionary can be accessed via key.subkey1.subkey2
    If any of `in_keys` is 'data', the full data dictionary will be forwarded
    to the transformation.
    If the only entry in `out_keys` is 'data', the full data dictionary will
    be updated with the return value of the transformation.
    For the case of multi-sensor data, the sensors that the transform should be
    applied can be set via the 'sensors' attribute. By default, we assume
    a transformation is applied to all sensors.
    This class will add a 'apply_to_data' method to a given Functor which is
    used to call it on a DictData object. NOTE: This is an issue for static
    checking and is not recognized by pylint. It will usually be called in the
    compose() function and will not be called directly.

    Example:
        >>> @Transform(in_keys="images", out_keys="images")
        >>> class MyTransform:
        >>>     def __call__(images: list[np.array]) -> list[np.array]:
        >>>         images = do_something(images)
        >>>         return images
        >>> my_transform = MyTransform()
        >>> data = my_transform.apply_to_data(data)
    """

    def __init__(
        self,
        in_keys: Sequence[str] | str,
        out_keys: Sequence[str] | str,
        sensors: Sequence[str] | str | None = None,
        same_on_batch: bool = True,
    ) -> None:
        """Creates an instance of Transform.

        Args:
            in_keys (Sequence[str] | str): Specifies one or multiple (if any)
                input keys of the data dictionary which should be remapeed to
                another key. Defaults to None.
            out_keys (Sequence[str] | str): Specifies one or multiple (if any)
                output keys of the data dictionary which should be remaped to
                another key. Defaults to None.
            sensors (Sequence[str] | str | None, optional): Specifies the
                sensors this transformation should be applied to. If None, it
                will be applied to all available sensors. Defaults to None.
            same_on_batch (bool, optional): Whether to use the same
                transformation parameters to all sensors / view. Defaults to
                True.
        """
        if isinstance(in_keys, str):
            in_keys = [in_keys]
        self.in_keys = in_keys

        if isinstance(out_keys, str):
            out_keys = [out_keys]
        self.out_keys = out_keys

        if isinstance(sensors, str):
            sensors = [sensors]
        self.sensors = sensors

        self.same_on_batch = same_on_batch

    @no_type_check
    def __call__(self, transform: TFunctor) -> TFunctor:
        """Add in_keys / out_keys / sensors / apply_to_data attributes.

        Args:
            transform (TFunctor): A given Functor.

        Returns:
            TFunctor: The decorated Functor.
        """
        original_init = transform.__init__

        def apply_to_data(
            self_, input_batch: list[DictData]
        ) -> list[DictData]:
            """Wrap function with a handler for input / output keys.

            We use the specified in_keys in order to extract the positional
            input arguments of a function from the data dictionary, and the
            out_keys to replace the corresponding values in the output
            dictionary.
            """

            def _transform_fn(batch: list[DictData]) -> list[DictData]:
                in_batch = []
                for key in self_.in_keys:
                    key_data = []
                    for data in batch:
                        # Optionally allow the function to get the full data
                        # dict as aux input and set default value to None if
                        # key is not found
                        key_data += [
                            (
                                get_dict_nested(
                                    data, key.split("."), allow_missing=True
                                )
                                if key != "data"
                                else data
                            )
                        ]
                    if any(d is None for d in key_data):
                        # If any of the data in the batch is None, replace
                        # the input of the key with None.
                        in_batch.append(None)
                    else:
                        in_batch.append(key_data)

                result = self_(*in_batch)

                if len(self_.out_keys) == 1:
                    if self_.out_keys[0] == "data":
                        return result
                    result = [result]

                for key, values in zip(self_.out_keys, result):
                    if values is None:
                        continue
                    for data, value in zip(batch, values):
                        if value is not None:
                            set_dict_nested(data, key.split("."), value)
                return batch

            if self_.sensors is not None:
                if self_.same_on_batch:
                    for sensor in self_.sensors:
                        batch_sensor = _transform_fn(
                            [d[sensor] for d in input_batch]
                        )
                        for i, d in enumerate(batch_sensor):
                            input_batch[i][sensor] = d
                else:
                    for i, data in enumerate(input_batch):
                        for sensor in self_.sensors:
                            input_batch[i][sensor] = _transform_fn(
                                [data[sensor]]
                            )
            elif self_.same_on_batch:
                input_batch = _transform_fn(input_batch)
            else:
                for i, data in enumerate(input_batch):
                    input_batch[i] = _transform_fn([data])[0]

            return input_batch

        def init(
            *args,
            in_keys: Sequence[str] = self.in_keys,
            out_keys: Sequence[str] = self.out_keys,
            sensors: Sequence[str] | None = self.sensors,
            same_on_batch: bool = self.same_on_batch,
            **kwargs,
        ):
            self_ = args[0]
            original_init(*args, **kwargs)
            self_.in_keys = in_keys
            self_.out_keys = out_keys
            self_.sensors = sensors
            self_.same_on_batch = same_on_batch
            self_.apply_to_data = lambda *args, **kwargs: apply_to_data(
                self_, *args, **kwargs
            )

        transform.__init__ = init
        return transform


def compose(transforms: list[TFunctor]) -> TransformFunction:
    """Compose transformations.

    This function composes a given set of transformation functions, i.e. any
    functor decorated with Transform, into a single transform.
    """

    def _preprocess_func(batch: list[DictData]) -> list[DictData]:
        for op in transforms:
            batch = op.apply_to_data(batch)  # type: ignore
        return batch

    return _preprocess_func


@Transform("data", "data")
class RandomApply:
    """Randomize the application of a given set of transformations."""

    def __init__(
        self, transforms: list[TFunctor], probability: float = 0.5
    ) -> None:
        """Creates an instance of RandomApply.

        Args:
            transforms (list[TFunctor]): Transformations that are applied with
                a given probability.
            probability (float, optional): Probability to apply
                transformations. Defaults to 0.5.
        """
        self.transforms = transforms
        self.probability = probability

    def __call__(self, batch: list[DictData]) -> list[DictData]:
        """Apply transforms with a given probability."""
        if torch.rand(1) < self.probability:
            for op in self.transforms:
                batch = op.apply_to_data(batch)  # type: ignore
        return batch
