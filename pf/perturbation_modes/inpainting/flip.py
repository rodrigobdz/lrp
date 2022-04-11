r'''Flipping logic for Pixel-Flipping Algorithm with Perturbation Mode Inpainting.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Tuple, Union, Optional
import logging
import torch
import torchvision
import numpy
import cv2


def flip_inpainting(input: torch.Tensor,
                    mask: torch.Tensor,
                    perturbation_size: Union[int, Tuple[int]],
                    logger: Optional[logging.Logger] = None) -> None:
    r'''Flip pixels of input in -place according to the relevance scores with
    perturbation technique random.

    Pixels to be flipped will be replaced by random samples drawn from the interval
    between the values of the low and high parameters.

    : param input: Input to be flipped.
    : param mask: Mask to select which pixels to flip.
    : param perturbation_size: Size of the region to flip.
    A size of 1 corresponds to single pixels, whereas a tuple to patches.

    : param logger: Logger instance to be used to print to console.
    '''

    # Reduce number of channels in mask from 3 to 1.
    mask_grayscale: torch.Tensor = torchvision.transforms.functional.rgb_to_grayscale(
        img=mask, num_output_channels=1)

    # OpenCV's inpainting method has the following prerequisites for the mask:
    # 1. Data type: numpy array of 8-bit integers with 1-channel
    # Non-zero pixels indicate the area that needs to be inpainted.
    # 2. The format it expects is HWC instead of CHW.
    mask_arr: numpy.array = mask_grayscale.int().cpu().numpy().transpose((1, 2, 0))
    # Type-cast to 8-bit integers.
    mask_arr = mask_arr.astype(numpy.uint8)
    img: numpy.array = input[0].cpu().numpy().transpose(
        (1, 2, 0)).astype(numpy.uint8)
    # Type-cast to 8-bit integers.
    img = img.astype(numpy.uint8)

    inpainted_img: numpy.array = cv2.inpaint(
        img, mask_arr, 3, cv2.INPAINT_TELEA)

    # FIXME: convert image back to tensor and save as input

    print('inpainted_img', inpainted_img.shape)
    input: torch.Tensor = torch.from_numpy(inpainted_img)
    print('input', input.shape)

    # FIXME: Input needs to be set in-place or returned. Currently not saved.

    # Error handling for missing or wrong parameters
    # Initialize logger, if not provided.
    if not logger:
        logger = logging.getLogger(__name__)
