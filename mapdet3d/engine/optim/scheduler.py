# pylint: disable=no-member
"""LR schedulers."""

from __future__ import annotations

from typing import TypedDict

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from mapdet3d.common.typing import DictStrAny
from mapdet3d.config import copy_and_resolve_references, instantiate_classes
from mapdet3d.config.typing import LrSchedulerConfig


class LRSchedulerDict(TypedDict):
    """LR scheduler."""

    scheduler: LRScheduler
    begin: int
    end: int
    epoch_based: bool


class LRSchedulerWrapper(LRScheduler):
    """LR scheduler wrapper."""

    def __init__(
        self,
        lr_schedulers_cfg: list[LrSchedulerConfig],
        optimizer: Optimizer,
        steps_per_epoch: int = -1,
    ) -> None:
        """Initialize LRSchedulerWrapper."""
        self.lr_schedulers_cfg: list[LrSchedulerConfig] = (
            copy_and_resolve_references(lr_schedulers_cfg)
        )
        self.lr_schedulers: dict[int, LRSchedulerDict] = {}
        super().__init__(optimizer)
        self.steps_per_epoch = steps_per_epoch
        self._convert_epochs_to_steps()

        for i, lr_scheduler_cfg in enumerate(self.lr_schedulers_cfg):
            if lr_scheduler_cfg["begin"] == 0:
                self._instantiate_lr_scheduler(i, lr_scheduler_cfg)

    def _convert_epochs_to_steps(self) -> None:
        """Convert epochs to steps."""
        for lr_scheduler_cfg in self.lr_schedulers_cfg:
            if (
                lr_scheduler_cfg["convert_epochs_to_steps"]
                and not lr_scheduler_cfg["epoch_based"]
            ):
                lr_scheduler_cfg["begin"] *= self.steps_per_epoch
                lr_scheduler_cfg["end"] *= self.steps_per_epoch
                if lr_scheduler_cfg["convert_attributes"] is not None:
                    for attr in lr_scheduler_cfg["convert_attributes"]:
                        lr_scheduler_cfg["scheduler"]["init_args"][
                            attr
                        ] *= self.steps_per_epoch

    def _instantiate_lr_scheduler(
        self, scheduler_idx: int, lr_scheduler_cfg: LrSchedulerConfig
    ) -> None:
        """Instantiate LR schedulers."""
        # OneCycleLR needs max_lr to be set
        if "max_lr" in lr_scheduler_cfg["scheduler"]["init_args"]:
            lr_scheduler_cfg["scheduler"]["init_args"]["max_lr"] = [
                pg["lr"] for pg in self.optimizer.param_groups
            ]

        self.lr_schedulers[scheduler_idx] = {
            "scheduler": instantiate_classes(
                lr_scheduler_cfg["scheduler"], optimizer=self.optimizer
            ),
            "begin": lr_scheduler_cfg["begin"],
            "end": lr_scheduler_cfg["end"],
            "epoch_based": lr_scheduler_cfg["epoch_based"],
        }

    def get_lr(self) -> list[float]:
        """Get current learning rate."""
        lr = []
        for lr_scheduler in self.lr_schedulers.values():
            lr.extend(lr_scheduler["scheduler"].get_lr())
        return lr

    def state_dict(self) -> dict[int, DictStrAny]:  # type: ignore
        """Get state dict."""
        state_dict = {}
        for scheduler_idx, lr_scheduler in self.lr_schedulers.items():
            state_dict[scheduler_idx] = lr_scheduler["scheduler"].state_dict()
        return state_dict

    def load_state_dict(
        self, state_dict: dict[int, DictStrAny]  # type: ignore
    ) -> None:
        """Load state dict."""
        for scheduler_idx, _state_dict in state_dict.items():
            # Instantiate the lr scheduler if it is not instantiated yet
            if not scheduler_idx in self.lr_schedulers:
                self._instantiate_lr_scheduler(
                    scheduler_idx, self.lr_schedulers_cfg[scheduler_idx]
                )
            self.lr_schedulers[scheduler_idx]["scheduler"].load_state_dict(
                _state_dict
            )

    def _step_lr(self, lr_scheduler: LRSchedulerDict, step: int) -> None:
        """Step the learning rate."""
        if lr_scheduler["begin"] <= step and (
            lr_scheduler["end"] == -1 or lr_scheduler["end"] >= step
        ):
            lr_scheduler["scheduler"].step()

    def step(self, epoch: int | None = None) -> None:
        """Step on training epoch end."""
        if epoch is not None:
            for lr_scheduler in self.lr_schedulers.values():
                if lr_scheduler["epoch_based"]:
                    self._step_lr(lr_scheduler, epoch)

            for i, lr_scheduler_cfg in enumerate(self.lr_schedulers_cfg):
                if lr_scheduler_cfg["epoch_based"] and (
                    lr_scheduler_cfg["begin"] == epoch + 1
                ):
                    self._instantiate_lr_scheduler(i, lr_scheduler_cfg)

    def step_on_batch(self, step: int) -> None:
        """Step on training batch end."""
        for lr_scheduler in self.lr_schedulers.values():
            if not lr_scheduler["epoch_based"]:
                self._step_lr(lr_scheduler, step)

        for i, lr_scheduler_cfg in enumerate(self.lr_schedulers_cfg):
            if not lr_scheduler_cfg["epoch_based"] and (
                lr_scheduler_cfg["begin"] == step
            ):
                self._instantiate_lr_scheduler(i, lr_scheduler_cfg)


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of steps.
        factor (float): Scale factor. Default: 1.0 / 3.0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        factor: float = 1.0 / 3.0,
        last_epoch: int = -1,
    ):
        """Initialize ConstantLR."""
        self.max_steps = max_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate."""
        step_count = self._step_count - 1
        if step_count == 0:
            return [
                group["lr"] * self.factor
                for group in self.optimizer.param_groups
            ]
        if step_count == self.max_steps:
            return [
                group["lr"] * (1.0 / self.factor)
                for group in self.optimizer.param_groups
            ]
        return [group["lr"] for group in self.optimizer.param_groups]


class PolyLR(LRScheduler):
    """Polynomial learning rate decay.

    Example:
        Assuming lr = 0.001, max_steps = 4, min_lr = 0.0, and power = 1.0, the
        learning rate will be:
        lr = 0.001     if step == 0
        lr = 0.00075   if step == 1
        lr = 0.00050   if step == 2
        lr = 0.00025   if step == 3
        lr = 0.0       if step >= 4

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of steps.
        power (float, optional): Power factor. Default: 1.0.
        min_lr (float): Minimum learning rate. Default: 0.0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize PolyLRScheduler."""
        self.max_steps = max_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate."""
        step_count = self._step_count - 1
        if step_count == 0 or step_count > self.max_steps:
            return [group["lr"] for group in self.optimizer.param_groups]
        decay_factor = (
            (1.0 - step_count / self.max_steps)
            / (1.0 - (step_count - 1) / self.max_steps)
        ) ** self.power
        return [
            (group["lr"] - self.min_lr) * decay_factor + self.min_lr
            for group in self.optimizer.param_groups
        ]


class QuadraticLRWarmup(LRScheduler):
    """Quadratic learning rate warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of steps.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        last_epoch: int = -1,
    ):
        """Initialize QuadraticLRWarmup."""
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate."""
        step_count = self._step_count - 1
        if step_count >= self.max_steps:
            return self.base_lrs
        factors = [
            base_lr * (2 * step_count + 1) / self.max_steps**2
            for base_lr in self.base_lrs  # pylint: disable=not-an-iterable
        ]
        if step_count == 0:
            return factors
        return [
            group["lr"] + factor
            for factor, group in zip(factors, self.optimizer.param_groups)
        ]
