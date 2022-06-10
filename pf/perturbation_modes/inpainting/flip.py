r"""Flipping logic for Pixel-Flipping Algorithm with Perturbation Mode Inpainting."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import logging
from typing import Optional

import cv2
import numpy
import torch

from pf import sanity_checks, utils
from pf.convert_img import opencv_to_tensor, tensor_to_opencv_inpainting

DEVICE: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def flip_inpainting(input_nchw: torch.Tensor,
                    mask_n1hw: torch.Tensor,
                    logger: Optional[logging.Logger] = None) -> torch.Tensor:
    r"""Flip pixels of image (not in-place) according to relevance scores with inpainting.

    Pixels to be flipped will be replaced by random samples drawn from the interval
    between the values of the low and high parameters.

    :param input_nchw: Image to be flipped in NCHW format with int values.
    :param mask_n1hw: Mask to select which pixels to flip in CHW format.
    :param logger: Logger instance to be used to print to console.

    :raises TypeError: If input_nchw does not have data type integer.
    :rasises ValueError: If input_nchw and mask_n1hw do not have the same batch size.

    :returns: Flipped image.
    """
    # Error handling for missing or wrong parameters
    # Initialize logger, if not provided.
    if not logger:
        logger = logging.getLogger(__name__)

    if input_nchw.is_floating_point():
        raise TypeError('Tensor must be of integer data type.')

    sanity_checks.verify_batch_size(input_nchw, mask_n1hw)

    inpainted_img_rgb_nchw: torch.Tensor = torch.zeros(0,
                                                       dtype=torch.float,
                                                       device=DEVICE)
    batch_size: int = utils.get_batch_size(input_nchw=input_nchw)

    # Loop over all images and masks in batch
    for batch_index in range(batch_size):
        mask_1hw: torch.Tensor = mask_n1hw[batch_index].to(device=DEVICE)
        input_chw: torch.Tensor = input_nchw[batch_index].to(device=DEVICE)

        logger.debug("Mask %s will flip a total of %s elements in image.",
                     batch_index,
                     mask_1hw.count_nonzero().item())

        # Reduce number of channels in mask from 3 to 1.
        mask_arr_hw1: numpy.ndarray = tensor_to_opencv_inpainting(
            img_rgb_chw=mask_1hw, grayscale=True)
        img_bgr_hwc: numpy.ndarray = tensor_to_opencv_inpainting(
            img_rgb_chw=input_chw)

        inpainted_img_bgr_hwc: numpy.ndarray = cv2.inpaint(img_bgr_hwc,
                                                           mask_arr_hw1,
                                                           3,
                                                           cv2.INPAINT_TELEA)

        # Convert back inpainted image to tensor
        inpainted_img_rgb_chw: torch.Tensor = opencv_to_tensor(
            img_bgr_hwc=inpainted_img_bgr_hwc).to(device=DEVICE)

        # Simulate batch by adding a new dimension using unsqueeze(0)
        # Concatenate inpainted image with the rest of the batch of inpainted images.
        # Shape of inpainted_img_rgb_nchw: (batch_size, 3, height, width)
        inpainted_img_rgb_nchw = torch.cat(
            (inpainted_img_rgb_nchw, inpainted_img_rgb_chw.unsqueeze(0))).to(device=DEVICE)

    return inpainted_img_rgb_nchw
