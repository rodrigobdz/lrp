r"""Image pre-processing and handling
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

from . import norm


def load_img_norm_zero_one(path: str) -> numpy.ndarray:
    r"""Load image with pixel values [0, 255] and normalize to [0, 1].

    Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb

    :param path: Path to the image
    :returns: Normalized image
    """
    # Returns image as numpy array with HWC format in BGR color space (not RGB) and with pixel values in [0, 255]
    img_bgr_hwc = cv2.imread(path)

    # Convert from BGR to RGB color space
    img_rgb_hwc = cv2.cvtColor(img_bgr_hwc, cv2.COLOR_BGR2RGB)

    # Alternative way to convert from BGR to RGB color space
    # img_rgb_hwc = img_bgr_hwc[..., ::-1]

    # img_rgb_hwc.shape is (224, 224, 3), where 3 corresponds to RGB channels

    return norm.norm_img_pxls(img_rgb_hwc, min_pixel_val=0., max_pixel_val=255.)


def img_to_tensor(img_nhwc: numpy.ndarray) -> torch.Tensor:
    r"""Convert image as numpy.ndarray to tensor

    Inspired by https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb and zennit
    Add new axis to simulate batch dimension of size 1.
    Convert from NHWC to NCHW format using transpose.
    Set the data type to float.

    Meaning of NCHW format:
        N: number of images in the batch
        C: number of channels of the image (3 for RGB, 1 for grayscale)
        H: height of the image
        W: width of the image

    :param img_nhwc: Image to be converted

    :returns: Tensor with image data
    """
    return torch.FloatTensor(img_nhwc[numpy.newaxis].transpose([0, 3, 1, 2]) * 1)
