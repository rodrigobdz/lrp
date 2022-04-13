r'''Objectives to sort relevance scores in Pixel-Flipping Algorithm.
Defines the order in which the relevance scores are flipped.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from torchvision import transforms
from typing import Generator
import torch


class PixelFlippingObjectives:
    r'''Objectives for Pixel-Flipping Algorithm.'''
    MORF: str = 'Most Relevant First'


def _argsort(relevance_scores: torch.Tensor, objective: str = PixelFlippingObjectives.MORF) -> torch.Tensor:
    r'''Generator function that sorts relevance scores in order defined by objective.

    :param relevance_scores: Relevance scores in NCHW format.
    :param objective: Sorting order for relevance scores.

    :returns: Sorted relevance scores as one-dimensional list.
    '''

    if objective != PixelFlippingObjectives.MORF:
        raise NotImplementedError(f'Objective {objective} not supported.')

    # Controls the sorting order (ascending or descending).
    # Set default value to descending—i.e., most relevant first.
    descending: bool = True

    # TODO: Add switch case to implement the user's selected objective.

    # Sort relevance scores according to objective
    sorted_values, _ = relevance_scores[0].flatten().sort(
        descending=descending, stable=True)

    return sorted_values


def _mask_generator(relevance_scores: torch.Tensor,
                    sorted_values: torch.Tensor,
                    perturbation_size: int
                    ) -> Generator[torch.Tensor, None, None]:
    r'''Generator function that creates masks with one value selected at a time from the order in which they are sorted.

    :param relevance_scores: Relevance scores in NCHW format.
    :param sorted_values: Sorted relevance scores as one-dimensional list.
    :param perturbation_size: Size of the region to flip.
    A size of 1 corresponds to single pixels, whereas a higher number to patches of size nxn.

    :yields: Mask in CHW (1-channel) format to flip pixels/patches input in order specified by sorted_values.
    '''
    # Calculate maximum height and width for generating patches around the selected pixel.
    max_height: int = relevance_scores[0].shape[1]
    max_width: int = relevance_scores[0].shape[2]

    for threshold_value in sorted_values:
        # Create mask to flip pixels/patches in input located at the index of the
        # threshold value in the sorted relevance scores.
        mask: torch.Tensor = relevance_scores[0] == threshold_value

        # Reduce dimensionality of mask from 3-channel to 1-channel.
        # any(dim=0) returns True if any element along dimension 0 is True. Returns HW format, no C
        # unsqueeze(0) creates artificial channel dimension using to make mask in CHW format.
        mask = mask.any(dim=0).unsqueeze(0)

        # Region Perturbation in action.
        # FIXME: Extract into function.
        i, j = mask.squeeze().nonzero().flatten().tolist()
        # TODO: Integrate heuristics on edges
        # max(j+perturbation_size, max_height)
        # max(, max_width)

        i_centered: int = i - (perturbation_size//2)
        j_centered: int = j - (perturbation_size//2)
        # print('i j', i, j)
        # print('centered', i_centered, j_centered)
        # Create patches around selected pixel.
        patch_mask: torch.Tensor = transforms.functional.erase(img=mask,
                                                               i=i_centered,
                                                               j=j_centered,
                                                               h=perturbation_size,
                                                               w=perturbation_size,
                                                               v=True)

        yield patch_mask
