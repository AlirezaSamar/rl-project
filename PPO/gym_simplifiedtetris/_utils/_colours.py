from enum import Enum
from typing import Tuple

from matplotlib import colors
import numpy as np


def _get_bgr_code(colour_name: str, /) -> Tuple[float, float, float]:
    """
    Get the inverted RGB code corresponding to the arg provided.

    :param colour_name: a string of the colour name,
    :return: an inverted RGB code of the inputted colour name.
    """
    return tuple(np.array([255, 255, 255]) * colors.to_rgb(colour_name))[::-1]


class _Colours(Enum):
    """
    Enumerate inverted RGB code.
    """

    WHITE = _get_bgr_code("white")
    BLACK = _get_bgr_code("black")
    CYAN = _get_bgr_code("cyan")
    ORANGE = _get_bgr_code("orange")
    YELLOW = _get_bgr_code("yellow")
    PURPLE = _get_bgr_code("purple")
    BLUE = _get_bgr_code("blue")
    GREEN = _get_bgr_code("green")
    RED = _get_bgr_code("red")
