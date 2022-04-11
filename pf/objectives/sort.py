r'''Objectives to sort relevance scores in Pixel-Flipping Algorithm.
Defines the order in which the relevance scores are flipped.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Generator
import torch


class PixelFlippingObjectives:
    r'''Objectives for Pixel-Flipping Algorithm.'''
    MORF: str = 'Most Relevant First'


def _argsort(relevance_scores: torch.Tensor, objective: str = PixelFlippingObjectives.MORF) -> Generator[torch.Tensor, None, None]:
    r'''Generator function that sorts relevance scores in order defined by objective.

    :param relevance_scores: Relevance scores in NCHW format.
    :param objective: Sorting order for relevance scores.

    :yields: Mask in CHW format to flip pixels/patches input in order specified by objective based on relevance scores.
    '''

    if objective != PixelFlippingObjectives.MORF:
        raise NotImplementedError(f'Objective {objective} not supported.')

    # Controls the sorting order (ascending or descending).
    # Set default value to descending—i.e., most relevant first.
    descending: bool = True

    # FIXME: Add switch case to implement the user's selected objective.

    # Sort relevance scores according to objective
    sorted_values, _ = relevance_scores[0].flatten().sort(
        descending=descending, stable=True)

    for threshold_value in sorted_values:
        # Create mask to flip pixels/patches in input located at the index of the
        # threshold value in the sorted relevance scores.
        mask: torch.Tensor = relevance_scores[0] == threshold_value

        yield mask
