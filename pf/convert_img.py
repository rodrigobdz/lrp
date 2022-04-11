r'''Transforms for images between libraries.

Considers RGB/BGR and channel order and NCHW/NHWC formats.
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


import numpy
import torch
import cv2


def opencv_to_tensor(img_bgr_hwc: numpy.array) -> torch.Tensor:
    r'''Convert image as numpy array to torch tensor.
    Operations performed:
        Convert color format from BGR to RGB.
        Convert from HWC to CHW format.

    :param img_bgr_hwc: Numpy array with image with HWC format and BGR color format to be converted.

    :returns: A torch tensor with CHW format and RGB color format.
    '''
    if not isinstance(img_bgr_hwc, numpy.ndarray):
        raise TypeError('Input must be a numpy array.')
    # TODO: Ensure data type is correct and return tensor is in format CHW.

    # Convert from BGR to RGB color space
    img_rgb_hwc: numpy.array = cv2.cvtColor(img_bgr_hwc, cv2.COLOR_BGR2RGB)

    # Convert from HWC to CHW format.
    img_rgb_chw: numpy.array = img_rgb_hwc.transpose((2, 0, 1))
    return torch.from_numpy(img_rgb_chw)


def tensor_to_opencv(img_rgb_chw: torch.Tensor, grayscale=False) -> numpy.array:
    r'''Convert image as torch tensor to numpy array.
    Operations performed:
        Convert from CHW to HWC format.
        Convert color format from RGB to BGR.

    :param img_rgb_chw: Torch tensor in CHW format and RGB color format to be converted.
    :param grayscale: Whether to convert skip color conversions because image has only one channel.

    :returns: A numpy array with HWC format and BGR color format.
    '''
    if not isinstance(img_rgb_chw, torch.Tensor):
        raise TypeError('Input must be a torch tensor.')

    # Convert to numpy array and from CHW to HWC format
    img_rgb_hwc: numpy.array = img_rgb_chw.numpy().transpose((1, 2, 0))

    if grayscale:
        # Skip color format conversion
        return img_rgb_hwc

    # Convert from RGB to BGR color space
    img_bgr_hwc: numpy.array = cv2.cvtColor(img_rgb_hwc, cv2.COLOR_RGB2BGR)

    return img_bgr_hwc


def tensor_to_opencv_inpainting(img_rgb_chw: torch.Tensor, grayscale=False) -> numpy.array:
    r'''Convert tensor to numpy array with requirements for inpainting with OpenCV.
    Operations performed:
        Ensure image is in CPU memory before any operation.
        Type-cast to 8-bit integers required by OpenCV's inpainting function.

    :param img_rgb_chw: Image to be converted.
    :param grayscale: Whether to convert skip color conversions because image has only one channel.

    :returns: Numpy array with image in HWC format and BGR color format.
    '''
    if not isinstance(img_rgb_chw, torch.Tensor):
        raise TypeError('Input must be a torch tensor.')

    return tensor_to_opencv(
        img_rgb_chw=img_rgb_chw.int().cpu(), grayscale=grayscale).astype(numpy.uint8)
