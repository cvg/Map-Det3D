"""Matplotlib based image viewer."""

from __future__ import annotations

import matplotlib.pyplot as plt

from mapdet3d.common.typing import NDArrayUI8


class MatplotlibImageViewer:
    """A image viewer using matplotlib.pyplot."""

    def show_images(
        self, images: list[NDArrayUI8], blocking: bool = True
    ) -> None:
        """Shows a list of images.

        Args:
            images (list[NDArrayUI8]): Images to display.
            blocking (bool): If the viewer should be blocking and wait
                for human input after each image.
        """
        for image in images:
            plt.imshow(image)
            plt.axis("off")
            plt.show(block=blocking)

    def save_images(
        self, images: list[NDArrayUI8], file_paths: list[str]
    ) -> None:
        """Saves a list of images.

        Args:
            images (list[NDArrayUI8]): Images to save.
            file_paths (list[str]): File paths to save the images to.
        """
        for i, image in enumerate(images):
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(f"{file_paths[i]}", bbox_inches="tight")
