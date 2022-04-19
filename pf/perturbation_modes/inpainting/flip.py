r'''Flipping logic for Pixel-Flipping Algorithm with Perturbation Mode Inpainting.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Optional
import logging
import torch
import numpy
import cv2
from pf.convert_img import opencv_to_tensor, tensor_to_opencv_inpainting


def flip_inpainting(input_nchw: torch.Tensor,
                    mask_n1hw: torch.Tensor,
                    logger: Optional[logging.Logger] = None) -> torch.Tensor:
    r'''Flip pixels of image (not in-place) according to the relevance scores with
    perturbation technique random.

    Pixels to be flipped will be replaced by random samples drawn from the interval
    between the values of the low and high parameters.

    :param input_nchw: Image to be flipped in NCHW format with int values.
    :param mask_n1hw: Mask to select which pixels to flip in CHW format.
    :param logger: Logger instance to be used to print to console.

    :raises TypeError: If input_nchw does not have data type integer.
    :rasises ValueError: If input_nchw and mask_n1hw do not have the same batch size.

    :returns: Flipped image.
    '''
    # Error handling for missing or wrong parameters
    # Initialize logger, if not provided.
    if not logger:
        logger = logging.getLogger(__name__)

    if input_nchw.is_floating_point():
        raise TypeError('Tensor must be of integer data type.')

    # Verify that input and mask have the same batch size.
    if input_nchw.shape[0] != mask_n1hw.shape[0]:
        raise ValueError(
            f'Number of images in input ({input_nchw.shape[0]}) must equal number of masks ({mask_n1hw.shape[0]})')

    inpainted_img_rgb_nchw: torch.Tensor = torch.zeros(0, dtype=torch.float)
    batch_size: int = input_nchw.shape[0]

    # Loop over all images and masks in batch
    for n in range(batch_size):
        mask_1hw: torch.Tensor = mask_n1hw[n]
        input_chw: torch.Tensor = input_nchw[n]

        logger.debug(
            f'Mask {n} will flip a total of {mask_1hw.count_nonzero().item()} elements in image.')

        # Reduce number of channels in mask from 3 to 1.
        mask_arr_hw1: numpy.array = tensor_to_opencv_inpainting(
            img_rgb_chw=mask_1hw, grayscale=True)
        img_bgr_hwc: numpy.array = tensor_to_opencv_inpainting(
            img_rgb_chw=input_chw)

        inpainted_img_bgr_hwc: numpy.array = cv2.inpaint(
            img_bgr_hwc, mask_arr_hw1, 3, cv2.INPAINT_TELEA)

        # Convert back inpainted image to tensor
        inpainted_img_rgb_chw: torch.Tensor = opencv_to_tensor(
            img_bgr_hwc=inpainted_img_bgr_hwc)

        # Simulate batch by adding a new dimension using unsqueeze(0)
        # Concatenate inpainted image with the rest of the batch of inpainted images.
        # Shape of inpainted_img_rgb_nchw: (batch_size, 3, height, width)
        inpainted_img_rgb_nchw = torch.cat(
            (inpainted_img_rgb_nchw, inpainted_img_rgb_chw.unsqueeze(0)))

    return inpainted_img_rgb_nchw
