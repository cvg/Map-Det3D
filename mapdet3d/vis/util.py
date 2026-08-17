"""Utilities for visualization."""

from __future__ import annotations

import colorsys

import numpy as np


def generate_color_map(length: int) -> list[tuple[int, int, int]]:
    """Generate a color palette of [length] colors.

    Args:
        length (int): Number of colors to generate.

    Returns:
        list[tuple[int, int, int]]: List with different colors ranging
            from [0,255].
    """
    brightness = 0.7
    hsv = [(i / length, 1, brightness) for i in range(length)]
    colors_float = [colorsys.hsv_to_rgb(*c) for c in hsv]
    colors: list[int] = (
        (np.array(colors_float) * 255).astype(np.uint8).tolist()
    )
    s = np.random.get_state()
    np.random.seed(0)
    result = [tuple(colors[i]) for i in np.random.permutation(len(colors))]
    np.random.set_state(s)
    return result


DEFAULT_COLOR_MAPPING = generate_color_map(50)
