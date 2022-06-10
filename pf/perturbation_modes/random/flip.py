r"""Flipping logic for Pixel-Flipping Algorithm with Perturbation Mode Random."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import logging
from typing import Optional, Tuple, Union

import torch

from pf import utils

from .random_number_generators import RandomNumberGenerator

DEVICE: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def flip_random(input_nchw: torch.Tensor,
                mask_n1hw: torch.Tensor,
                ran_num_gen: RandomNumberGenerator,
                low: float,
                high: float,
                perturbation_size: Union[int, Tuple[int]],
                logger: Optional[logging.Logger] = None) -> torch.Tensor:
    r"""Flip pixels of image in-place according to the relevance scores with
    perturbation technique random.

    Pixels to be flipped will be replaced by random samples drawn from the interval
    between the values of the low and high parameters.

    :param input_nchw: Image to be flipped in NCHW format.
    :param mask_n1hw: Mask to select which pixels to flip in CHW format.
    A size of 1 corresponds to single pixels, whereas a tuple to patches.

    :param ran_num_gen: Random number generator to use.
    :param low: Lower bound of the range of values to be flipped.
    :param high: Upper bound of the range of values to be flipped.

    Unused parameter:
    :param perturbation_size: Size of the region to flip. Used to determine the number of random values to generate.

    :param logger: Logger instance to be used to print to console.

    :returns: Flipped image.
    """
    # Error handling for missing or wrong parameters
    # Initialize logger, if not provided.
    if not logger:
        logger = logging.getLogger(__name__)

    batch_size: int = utils.get_batch_size(input_nchw=input_nchw)

    # Loop over all images and masks in batch
    for batch_index in range(batch_size):
        mask_1hw: torch.Tensor = mask_n1hw[batch_index].to(device=DEVICE)
        input_chw: torch.Tensor = input_nchw[batch_index].to(device=DEVICE)

        # Convert mask from (1, H, W) to (C, H, W) where C is the number of channels in the image.
        # Expanding a tensor does not allocate new memory.
        # Expanding a tensor basically duplicates tensor C times.
        expanded_mask_chw: torch.Tensor = mask_1hw.expand(
            input_chw.shape).to(device=DEVICE)

        logger.debug("Expanded mask has shape %s.", expanded_mask_chw.shape)

        # Draw a random number.
        # Size of perturbation/patch is NxN, where N is perturbation_size.
        flip_value: float = ran_num_gen.draw(low=low,
                                             high=high)

        # TODO: Add parameter size for generating n multiple random numbers.
        # The following line is part of the changes required to support multiple random numbers.
        #  size=perturbation_size**2)

        # Compute indices selected for flipping in mask.
        flip_indices = mask_n1hw.nonzero().flatten().tolist()
        # Count how many elements are set to True—i.e., would be flipped.
        flip_count: int = input_chw[expanded_mask_chw].count_nonzero().item()
        logger.debug("Flipping input_chw %s to %s: %s element(s).",
                     flip_indices,
                     flip_value,
                     flip_count)

        # Flip pixels/patch in-place. Changes are reflected in input_nchw.
        # Disable gradient computation for the pixel-flipping operations.
        # Avoid error "A leaf Variable that requires grad is being used in an in-place operation."
        with torch.no_grad():
            input_chw[expanded_mask_chw] = flip_value

    return input_nchw.to(device=DEVICE)
