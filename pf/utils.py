r"""Helpers for Pixel-Flipping algorithm."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

from typing import Tuple

import torch


def loop(generator) -> None:
    r"""Loop over a generator without retrieving any values.

    :param generator: Generator to loop over.
    """
    for _ in generator:
        pass


def get_batch_size(input_nchw: torch.Tensor) -> int:
    r"""Get the batch size of the input tensor.

    :param input_nchw: Input tensor in NCHW format.

    :return: Batch size.
    """
    return input_nchw.size(dim=0)


def get_height_width(input_nchw: torch.Tensor) -> Tuple[int, int]:
    r"""Get the height and width of the input tensor.

    :param input_nchw: Input tensor in NCHW format.

    :return: Tuple containing the height and width of the input tensor.
    """
    height: int = input_nchw.size(dim=2)
    width: int = input_nchw.size(dim=3)

    return height, width
