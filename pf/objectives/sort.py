r"""Objectives to sort relevance scores in Pixel-Flipping Algorithm.
Defines the order in which the relevance scores are flipped."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from typing import Generator

import torch
from torchvision import transforms


class PixelFlippingObjectives:
    r"""Objectives for Pixel-Flipping Algorithm."""
    MORF: str = 'Most Relevant First'


def _argsort(relevance_scores_nchw: torch.Tensor, objective: str = PixelFlippingObjectives.MORF) -> torch.Tensor:
    r"""Generator function that sorts relevance scores in order defined by objective.

    :param relevance_scores_nchw: Relevance scores in NCHW format.
    :param objective: Sorting order for relevance scores.

    :returns: Sorted relevance scores as a tensor with N one-dimensional lists,
              one for each image in the batch of size and each list with m elements,
              m is # relevance scores for each image. Shape is (N, m).
    """

    if objective != PixelFlippingObjectives.MORF:
        raise NotImplementedError(f'Objective {objective} not supported.')

    # Controls the sorting order (ascending or descending).
    # Set default value to descending—i.e., most relevant first.
    descending: bool = True

    # TODO: Add switch case to implement the user's selected objective.

    # Sort relevance scores according to objective

    sorted_values_nm, _ = relevance_scores_nchw.flatten(start_dim=1).sort(
        descending=descending, stable=True)

    return sorted_values_nm


def _mask_generator(relevance_scores_nchw: torch.Tensor,
                    sorted_values_nm: torch.Tensor,
                    perturbation_size: int
                    ) -> Generator[torch.Tensor, None, None]:
    r"""Generator function that creates masks with one or multiple pixels selected for flipping
    at a time from the order in which they are sorted.

    :param relevance_scores_nchw: Relevance scores in NCHW format.
    :param sorted_values_nm: Sorted relevance scores as a tensor with N one-dimensional lists,
                             one for each image in the batch of size and each list with m elements,
                             m is # relevance scores for each image. Shape is (N, m).
    :param perturbation_size: Size of the region to flip.
    A size of 1 corresponds to single pixels, whereas a higher number to patches of size nxn.

    The patches for region perturbation (perturbation_size > 1) are overlapping.

    :yields: Mask in N1HW (1-channel) format to flip pixels/patches input in order specified by sorted_values_nm.
    """

    # Loop over number of elements in each individual list of sorted values.
    for m in range(sorted_values_nm.shape[1]):
        # Create empty boolean tensor.
        mask_nhw: torch.Tensor = torch.zeros(0, dtype=torch.bool)

        batch_size: int = sorted_values_nm.shape[0]

        # Loop over number of sorted value lists (number of images in batch).
        for n in range(batch_size):
            # Extract sorted value at index m for current image at batch index n.
            threshold_value: float = sorted_values_nm[n, m]

            # Create mask to flip pixels/patches in input located at the index of the
            # threshold value in the sorted relevance scores.
            mask_chw: torch.Tensor = relevance_scores_nchw[n] == threshold_value

            # Reduce dimensionality of mask from 3-channel to 1-channel.
            # any(dim=0) returns True if any element along dimension 0 is True. Returns HW format, no C
            # unsqueeze(0) creates artificial channel dimension using to make mask in CHW format.
            mask_1hw = mask_chw.any(dim=0).unsqueeze(0)

            # Region Perturbation in action.
            # i, j are the coordinates of the pixel to flip.
            i, j = mask_1hw.squeeze().nonzero().flatten().tolist()

            # Create patch around selected pixels.
            i_centered: int = i - (perturbation_size//2)
            j_centered: int = j - (perturbation_size//2)

            # TODO: Integrate heuristics around edges for non-overlapping patches.
            # Currently the patches are overlapping.

            # Create patches around selected pixel.
            flipped_mask_1hw: torch.Tensor = transforms.functional.erase(img=mask_1hw,
                                                                         i=i_centered,
                                                                         j=j_centered,
                                                                         h=perturbation_size,
                                                                         w=perturbation_size,
                                                                         v=True)

            # Concatenate the mask for the current image to the list of masks.
            # Initially mask_nhw is empty and masks for each image are added incrementally.
            # Shape of mask_nhw is (N, H, W).
            mask_nhw = torch.cat((mask_nhw, flipped_mask_1hw))

        # unsqueze(1) creates artificial channel dimension using to make mask in N1HW format from NHW.
        # Shape of mask_n1hw is (N, 1, H, W).
        mask_n1hw: torch.Tensor = mask_nhw.unsqueeze(1)

        yield mask_n1hw
