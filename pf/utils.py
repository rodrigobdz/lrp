r'''Helpers for pixel-flipping algorithm.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'

from typing import Generator
from .objectives import PixelFlippingObjectives
import torch


def _loop(generator) -> None:
    r'''Loop over a generator without retrieving any values.

    :param generator: Generator to loop over.
    '''
    for _ in generator:
        pass


def _argsort(relevance_scores: torch.Tensor, objective: str = PixelFlippingObjectives.MORF) -> Generator[torch.Tensor, None, None]:
    r'''Generator function that sorts relevance scores in order defined by objective.

    :param relevance_scores: Relevance scores.
    :param objective: Sorting order for relevance scores.

    :yields: Mask to flip pixels/patches input in order specified by objective based on relevance scores.
    '''

    # Controls the sorting order (ascending or descending).
    descending: bool = False

    # Objective 'Most Relevant First' (MORF) refers to descending order.
    if objective == PixelFlippingObjectives.MORF:
        descending = True

    # Sort relevance scores according to objective
    sorted_values, _ = relevance_scores[0].flatten().sort(
        descending=descending, stable=True)

    for threshold_value in sorted_values:
        # Create mask to flip pixels/patches in input located at the index of the
        # threshold value in the sorted relevance scores.
        mask: torch.Tensor = relevance_scores[0] == threshold_value

        yield mask