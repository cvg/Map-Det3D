"""Visualizer base class."""

from mapdet3d.common.typing import ArgsType


class Visualizer:
    """Base visualizer class."""

    def __init__(self, vis_freq: int = 50, image_mode: str = "RGB") -> None:
        """Initialize the visualizer.

        Args:
            vis_freq (int): Visualization frequency. Defaults to 0.
            image_mode (str): Image channel mode (RGB or BGR).
        """
        self.vis_freq = vis_freq
        self.image_mode = image_mode
        assert image_mode in {"RGB", "BGR"}

    def _run_on_batch(self, cur_iter: int) -> bool:
        """Return whether to run on current iteration.

        Args:
            cur_iter (int): Current iteration.
        """
        return cur_iter % self.vis_freq == 0

    def reset(self) -> None:
        """Reset visualizer for new round of evaluation."""
        raise NotImplementedError()

    def process(self, cur_iter: int, *args: ArgsType) -> None:
        """Process data of single sample."""
        raise NotImplementedError()

    def show(self, cur_iter: int, blocking: bool = True) -> None:
        """Shows the visualization.

        Args:
            cur_iter (int): Current iteration.
            blocking (bool): If the visualization should be blocking and wait
                for human input. Defaults to True.
        """
        raise NotImplementedError()

    def save_to_disk(self, cur_iter: int, output_folder: str) -> None:
        """Saves the visualization to disk.

        Args:
            cur_iter (int): Current iteration.
            output_folder (str): Folder where the output should be written.
        """
        raise NotImplementedError()
