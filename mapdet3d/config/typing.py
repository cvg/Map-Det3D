"""Type definitions for configuration files."""

from __future__ import annotations

from typing import Any, TypedDict

from ml_collections import ConfigDict, FieldReference
from typing_extensions import NotRequired

from .config_dict import FieldConfigDict


class ParamGroupCfg(TypedDict):
    """Parameter group config.

    Attributes:
        custom_keys (list[str]): List of custom keys.
        lr_mult (NotRequired[float]): Learning rate multiplier.
        decay_mult (NotRequired[float]): Weight Decay multiplier.
    """

    custom_keys: list[str]
    lr_mult: NotRequired[float]
    decay_mult: NotRequired[float]
    norm_decay_mult: NotRequired[float]
    bias_decay_mult: NotRequired[float]


class DataConfig(ConfigDict):  # type: ignore
    """Configuration for a data set.

    This data object is used to configure the training and test data of an
    experiment. In particular, the train_dataloader and test_dataloader
    need to be config dicts that can be instantiated as a dataloader.

    Attributes:
        train_dataloader (ConfigDict): Configuration for the training
            dataloader.
        test_dataloader (ConfigDict): Configuration for the test dataloader.


    Example:
        >>> from mapdet3d.config.types import DataConfig
        >>> from mapdet3d.zoo.base import class_config
        >>> from my_package.data import MyDataLoader
        >>> cfg = DataConfig()
        >>> cfg.train_dataloader = class_config(MyDataLoader, ...)
    """

    train_dataloader: ConfigDict
    test_dataloader: ConfigDict


class LrSchedulerConfig(ConfigDict):  # type: ignore
    """Configuration for a learning rate scheduler.

    Attributes:
        scheduler (ConfigDict): Configuration for the learning rate scheduler.
        begin (int): Begin epoch.
        end (int): End epoch.
        epoch_based (bool): Whether the learning rate scheduler is epoch based
            or step based.
        convert_epochs_to_steps (bool): Whether to convert the begin and end
            for a step based scheduler to steps automatically based on length
            of train dataloader. Enables users to set the iteration breakpoints
            as epochs. Defaults to False.
        convert_attributes (list[str] | None): List of attributes in the
            scheduler that should be converted to steps. Defaults to None.
    """

    scheduler: ConfigDict
    begin: int
    end: int
    epoch_based: bool
    convert_epochs_to_steps: bool = False
    convert_attributes: list[str] | None = None


class OptimizerConfig(ConfigDict):  # type: ignore
    """Configuration for an optimizer.

    Attributes:
        optimizer (ConfigDict): Configuration for the optimizer.
        lr_scheduler (list[LrSchedulerConfig] | None): Configuration for the
            learning rate scheduler.
        param_groups (list[ParamGroupCfg] | None): Configuration for the
            parameter groups.
    """

    optimizer: ConfigDict
    lr_scheduler: list[LrSchedulerConfig] | None
    param_groups: list[ParamGroupCfg] | None


class ExperimentParameters(FieldConfigDict):
    """Parameters for an experiment.

    Attributes:
        samples_per_gpu (int): Number of samples per GPU.
        workers_per_gpu (int): Number of workers per GPU.
    """

    samples_per_gpu: int
    workers_per_gpu: int


class ExperimentConfig(FieldConfigDict):
    """Configuration for an experiment.

    This data object is used to configure an experiment. It contains the
    minimal required configuration to run an experiment. In particular, the
    data, model, optimizers, and loss need to be config dicts that can be
    instantiated as a data set, model, optimizer, and loss function,
    respectively.

    Attributes:
        work_dir (str | FieldReference): The working directory for the
            experiment.
        experiment_name (str | FieldReference): The name of the experiment.
        timestamp (str | FieldReference): The timestamp of the experiment.
        version (str | FieldReference): The version of the experiment.
        output_dir (str | FieldReference): The output directory for the
            experiment.
        seed (int | FieldReference): The random seed for the experiment.
        log_every_n_steps (int | FieldReference): The number of steps after
            which the logs should be written.
        use_tf32 (bool | FieldReference): Whether to use tf32.
        benchmark (bool | FieldReference): Whether to enable benchmarking.
        params (ExperimentParameters): Configuration for the experiment
            parameters.
        data (DataConfig): Configuration for the dataset.
        model (FieldConfigDictOrRef): Configuration for the model.
        loss (FieldConfigDictOrRef): Configuration for the loss function.
        optimizers (list[OptimizerConfig]): Configuration for the optimizers.
        data_connector (FieldConfigDictOrRef): Configuration for the data
            connector.
        callbacks (list[FieldConfigDictOrRef]): Configuration for the
            callbacks which are used in the engine.
    """

    # General
    work_dir: str | FieldReference
    experiment_name: str | FieldReference
    timestamp: str | FieldReference
    version: str | FieldReference
    output_dir: str | FieldReference
    seed: int | FieldReference
    log_every_n_steps: int | FieldReference
    use_tf32: bool | FieldReference
    benchmark: bool | FieldReference
    tf32_matmul_precision: str | FieldReference

    params: ExperimentParameters

    # Data
    data: DataConfig

    # Model
    model: ConfigDict

    # Loss
    loss: ConfigDict

    # Optimizer
    optimizers: list[OptimizerConfig]

    # Data connector
    train_data_connector: ConfigDict
    test_data_connector: ConfigDict

    # Callbacks
    callbacks: list[ConfigDict]


class ParameterSweepConfig(FieldConfigDict):
    """Configuration for a parameter sweep.

    Confguration object for a parameter sweep. It contains the minimal required
    configuration to run a parameter sweep.

    Attributes:
        method (str): Sweep method that should be used (e.g. grid)
        sampling_args (list[tuple[str, Any]]): Arguments that should be passed
            to the sweep method. E.g. for grid, this would be a list of tuples
            of the form (parameter_name, parameter_values).
        suffix (str): Suffix that should be appended to the output directory.
            This will be interpreted as a string template and can contain
            references to the sampling_args.
            E.g. "lr_{lr:.2e}_bs_{batch_size}".
    """

    method: str | FieldReference
    sampling_args: list[tuple[str, Any]] | FieldReference  # type: ignore
    suffix: str | FieldReference = ""
