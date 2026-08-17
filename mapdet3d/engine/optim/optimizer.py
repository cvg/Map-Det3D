"""Optimizer."""

from __future__ import annotations

from typing import TypedDict

from torch import nn
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.optim.optimizer import Optimizer
from typing_extensions import NotRequired

from mapdet3d.common.logging import rank_zero_info
from mapdet3d.config import instantiate_classes
from mapdet3d.config.typing import OptimizerConfig, ParamGroupCfg

from .scheduler import LRSchedulerWrapper


class ParamGroup(TypedDict):
    """Parameter dictionary.

    Attributes:
        params (list[nn.Parameter]): List of parameters.
        lr (NotRequired[float]): Learning rate.
        weight_decay (NotRequired[float]): Weight decay.
    """

    params: list[nn.Parameter]
    lr: NotRequired[float]
    weight_decay: NotRequired[float]


# TODO: Add true support for multiple optimizers. This will need to
# modify config to specify which optimizer to use for which module.
def set_up_optimizers(
    optimizers_cfg: list[OptimizerConfig],
    models: list[nn.Module],
    steps_per_epoch: int = -1,
) -> tuple[list[Optimizer], list[LRSchedulerWrapper]]:
    """Set up optimizers."""
    optimizers = []
    lr_schedulers = []
    for optim_cfg, model in zip(optimizers_cfg, models):
        optimizer = configure_optimizer(optim_cfg, model)
        optimizers.append(optimizer)

        if optim_cfg.lr_schedulers is not None:
            lr_schedulers.append(
                LRSchedulerWrapper(
                    optim_cfg.lr_schedulers, optimizer, steps_per_epoch
                )
            )

    return optimizers, lr_schedulers


def configure_optimizer(
    optim_cfg: OptimizerConfig, model: nn.Module
) -> Optimizer:
    """Configure optimizer with parameter groups."""
    param_groups_cfg = optim_cfg.get("param_groups", None)

    if param_groups_cfg is None:
        return instantiate_classes(
            optim_cfg.optimizer, params=model.parameters()
        )

    params = []
    base_lr = optim_cfg.optimizer["init_args"].lr
    weight_decay = optim_cfg.optimizer["init_args"].get("weight_decay", None)
    for group in param_groups_cfg:
        lr_mult = group.get("lr_mult", 1.0)
        decay_mult = group.get("decay_mult", 1.0)
        norm_decay_mult = group.get("norm_decay_mult", None)
        bias_decay_mult = group.get("bias_decay_mult", None)

        param_group: ParamGroup = {"params": [], "lr": base_lr * lr_mult}

        if weight_decay is not None:
            if norm_decay_mult is not None:
                param_group["weight_decay"] = weight_decay * norm_decay_mult
            elif bias_decay_mult is not None:
                param_group["weight_decay"] = weight_decay * bias_decay_mult
            else:
                param_group["weight_decay"] = weight_decay * decay_mult

        params.append(param_group)

    # Create a param group for the rest of the parameters
    param_group = {"params": [], "lr": base_lr}
    if weight_decay is not None:
        param_group["weight_decay"] = weight_decay
    params.append(param_group)

    # Add the parameters to the param groups
    add_params(params, model, param_groups_cfg)

    return instantiate_classes(optim_cfg.optimizer, params=params)


def add_params(
    params: list[ParamGroup],
    module: nn.Module,
    param_groups_cfg: list[ParamGroupCfg],
    prefix: str = "",
) -> None:
    """Add all parameters of module to the params list.

    The parameters of the given module will be added to the list of param
    groups, with specific rules defined by paramwise_cfg.

    Args:
        params (list[DictStrAny]): A list of param groups, it will be modified
            in place.
        module (nn.Module): The module to be added.
        param_groups_cfg (dict[str, list[str] | float]): The configuration
            of the param groups.
        prefix (str): The prefix of the module. Default: ''.
    """
    for name, param in module.named_parameters(recurse=False):
        if not param.requires_grad:
            params[-1]["params"].append(param)
            continue

        is_norm = isinstance(
            module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm)
        )

        # if the parameter match one of the custom keys, ignore other rules
        is_custom = False
        msg = f"{prefix}.{name}"
        for i, group in enumerate(param_groups_cfg):
            for key in group["custom_keys"]:
                if key not in f"{prefix}.{name}":
                    continue
                norm_decay_mult = group.get("norm_decay_mult", None)
                bias_decay_mult = group.get("bias_decay_mult", None)
                if group.get("lr_mult", None) is not None:
                    msg += f" with lr_mult: {group['lr_mult']}"
                if norm_decay_mult is not None:
                    if not is_norm:
                        continue
                    msg += f" with norm_decay_mult: {norm_decay_mult}"
                if bias_decay_mult is not None:
                    if name != "bias":
                        continue
                    msg += f" with bias_decay_mult: {bias_decay_mult}"
                if group.get("decay_mult", None) is not None:
                    msg += f" with decay_mult: {group['decay_mult']}"
                params[i]["params"].append(param)
                is_custom = True
                break
            if is_custom:
                break

        if is_custom:
            rank_zero_info(msg)
        else:
            # add parameter to the last param group
            params[-1]["params"].append(param)

    for child_name, child_mod in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        add_params(params, child_mod, param_groups_cfg, prefix=child_prefix)
