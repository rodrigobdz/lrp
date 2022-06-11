r"""Transforms for images between libraries.

Considers RGB/BGR and channel order and NCHW/NHWC formats.
"""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import cv2
import numpy
import torch

from pf.sanity_checks import ensure_chw_format


def arr_chw_to_hwc(arr_chw: numpy.ndarray) -> numpy.ndarray:
    r"""Convert numpy array from CHW to HWC format.

    :param arr_chw: numpy array to be converted.

    :returns: numpy array in HWC format.
    """
    return arr_chw.transpose((1, 2, 0))


def arr_hwc_to_chw(arr_hwc: numpy.ndarray) -> numpy.ndarray:
    r"""Convert numpy array from HWC to CHW format.

    :param arr_hwc: numpy array to be converted.

    :returns: numpy array in CHW format.
    """
    return arr_hwc.transpose((2, 0, 1))


def opencv_to_tensor(img_bgr_hwc: numpy.ndarray) -> torch.Tensor:
    r"""Convert image as numpy array to torch tensor.

    Operations performed:
        Convert color format from BGR to RGB.
        Convert from HWC to CHW format.

    :param img_bgr_hwc: Numpy array with image with HWC format and BGR color format to be converted.

    :returns: A torch tensor with CHW format and RGB color format.
    """
    if not isinstance(img_bgr_hwc, numpy.ndarray):
        raise TypeError('Input must be a numpy array.')

    # Convert from BGR to RGB color space
    img_rgb_hwc: numpy.ndarray = cv2.cvtColor(img_bgr_hwc, cv2.COLOR_BGR2RGB)

    # Convert from HWC to CHW format.
    img_rgb_chw: numpy.ndarray = arr_hwc_to_chw(img_rgb_hwc)
    return torch.from_numpy(img_rgb_chw)


def tensor_to_opencv(img_rgb_chw: torch.Tensor, grayscale=False) -> numpy.ndarray:
    r"""Convert image as torch tensor to numpy array.

    Operations performed:
        Convert from CHW to HWC format.
        Convert color format from RGB to BGR.

    :param img_rgb_chw: Torch tensor in CHW format and RGB color format to be converted.
    :param grayscale: Whether to convert skip color conversions because image has only one channel.

    :returns: A numpy array with HWC format and BGR color format.
    """
    if not isinstance(img_rgb_chw, torch.Tensor):
        raise TypeError('Input must be a torch tensor.')

    ensure_chw_format(input_chw=img_rgb_chw)

    # Convert to numpy array and from CHW to HWC format
    # Use Tensor.cpu() to copy the tensor to host memory before converting to numpy().
    img_rgb_hwc: numpy.ndarray = arr_chw_to_hwc(
        img_rgb_chw.cpu().detach().numpy())

    if grayscale:
        # Skip color format conversion
        return img_rgb_hwc

    # Convert from RGB to BGR color space and to 8-bit integer data type
    img_bgr_hwc: numpy.ndarray = cv2.cvtColor(
        img_rgb_hwc.astype(numpy.uint8), cv2.COLOR_RGB2BGR)

    return img_bgr_hwc


def tensor_to_opencv_inpainting(img_rgb_chw: torch.Tensor, grayscale=False) -> numpy.ndarray:
    r"""Convert tensor to numpy array with requirements for inpainting with OpenCV.

    Operations performed:
        Ensure image is in CPU memory before any operation.
        Type-cast to 8-bit integers required by OpenCV's inpainting function.

    :param img_rgb_chw: Image to be converted. Assumes values are of type integer.
    :param grayscale: Whether to convert skip color conversions because image has only one channel.

    :returns: Numpy array with image in HWC format and BGR color format.
    """
    if not isinstance(img_rgb_chw, torch.Tensor):
        raise TypeError('Input must be a torch tensor.')

    if img_rgb_chw.is_floating_point():
        raise TypeError('Tensor must be of integer data type.')

    return tensor_to_opencv(
        img_rgb_chw=img_rgb_chw.int().cpu(), grayscale=grayscale).astype(numpy.uint8)
