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
from pf.convert_img import opencv_to_tensor, tensor_to_opencv_inpainting


def flip_inpainting(input: torch.Tensor,
                    mask: torch.Tensor,
                    logger: Optional[logging.Logger] = None) -> torch.Tensor:
    r'''Flip pixels of input (not in-place) according to the relevance scores with
    perturbation technique random.

    Pixels to be flipped will be replaced by random samples drawn from the interval
    between the values of the low and high parameters.

    :param input: Input to be flipped in NCHW format with int values.
    :param mask: Mask to select which pixels to flip in CHW format.

    :param logger: Logger instance to be used to print to console.

    :returns: Flipped input.
    '''
    # Error handling for missing or wrong parameters
    # Initialize logger, if not provided.
    if not logger:
        logger = logging.getLogger(__name__)

    if input.is_floating_point():
        raise TypeError('Tensor must be of integer data type.')

    # Reduce number of channels in mask from 3 to 1.
    mask_arr: numpy.array = tensor_to_opencv_inpainting(mask, grayscale=True)
    img_bgr_hwc: numpy.array = tensor_to_opencv_inpainting(input[0])

    inpainted_img_bgr_hwc: numpy.array = cv2.inpaint(
        img_bgr_hwc, mask_arr, 3, cv2.INPAINT_TELEA)

    # Convert back inpainted image to tensor
    inpainted_img_rgb_chw: numpy.array = opencv_to_tensor(
        img_bgr_hwc=inpainted_img_bgr_hwc)

    # Simulate batch by adding a new dimension
    inpainted_img_rgb_chw = torch.unsqueeze(inpainted_img_rgb_chw, 0)

    return inpainted_img_rgb_chw
