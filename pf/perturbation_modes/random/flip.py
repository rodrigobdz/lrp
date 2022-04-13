r'''Flipping logic for Pixel-Flipping Algorithm with Perturbation Mode Random.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Tuple, Union, Optional
from .random_number_generators import RandomNumberGenerator
import logging
import torch


def flip_random(image: torch.Tensor,
                mask: torch.Tensor,
                perturbation_size: Union[int, Tuple[int]],
                ran_num_gen: RandomNumberGenerator,
                low: float,
                high: float,
                logger: Optional[logging.Logger] = None) -> torch.Tensor:
    r'''Flip pixels of image in-place according to the relevance scores with
    perturbation technique random.

    Pixels to be flipped will be replaced by random samples drawn from the interval
    between the values of the low and high parameters.

    :param image: Image to be flipped in NCHW format.
    :param mask: Mask to select which pixels to flip in CHW format.
    :param perturbation_size: Size of the region to flip. Used to determine the number of random values to generate.
    A size of 1 corresponds to single pixels, whereas a tuple to patches.

    :param ran_num_gen: Random number generator to use.
    :param low: Lower bound of the range of values to be flipped.
    :param high: Upper bound of the range of values to be flipped.

    :param logger: Logger instance to be used to print to console.

    :returns: Flipped image.
    '''
    # Error handling for missing or wrong parameters
    # Initialize logger, if not provided.
    if not logger:
        logger = logging.getLogger(__name__)

    # Convert mask from (1, H, W) to (C, H, W) where C is the number of channels in the image.
    # Expanding a tensor does not allocate new memory.
    expanded_mask: torch.Tensor = mask.expand(image[0].shape)

    logger.debug(f'Expanded mask has shape {expanded_mask.shape}.')

    # Draw a random number.
    # Size of perturbation/patch is NxN, where N is perturbation_size.
    flip_value: float = ran_num_gen.draw(
        low=low, high=high, size=perturbation_size**2)

    # Compute indices selected for flipping in mask.
    flip_indices = mask.nonzero().flatten().tolist()
    # Count how many elements are set to True—i.e., would be flipped.
    flip_count: int = image[0][expanded_mask].count_nonzero().item()
    logger.debug(
        f'Flipping X[0]{flip_indices} to {flip_value}: {flip_count} element(s).')

    # Error handling during flipping.
    # FIXME: Remove this check to vectorize operation
    # DEBUG: Check what happens when flip_count is greater than one.
    # It seems like the mask_generator returns the mask repeatedly for #simultaneous flips times.
    if flip_count != 3:
        logger.debug(
            f'''Flip count {flip_count} is not one. The mask is flipping more than one element.''')

    # Flip pixels/patch
    # Disable gradient computation for the pixel-flipping operations.
    # Avoid error "A leaf Variable that requires grad is being used in an in-place operation."
    with torch.no_grad():
        image[0][expanded_mask] = flip_value

    return image
