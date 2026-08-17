"""Base class for meta architectures."""

import abc

from torch import nn

from .reducer import LossReducer, identity_loss


class Loss(nn.Module, abc.ABC):
    """Base loss class."""

    def __init__(self, reducer: LossReducer = identity_loss) -> None:
        """Initialize a loss functor.

        Args:
            reducer (LossReducer): A function to aggregate the loss values into
            a single tensor value. It is commonly used for dense prediction
            tasks to merge pixel-wise loss to a final loss.

            Example::
                def mean_loss(loss: torch.Tensor) -> torch.Tensor:
                    return loss.mean()
        """
        super().__init__()
        self.reducer = reducer
