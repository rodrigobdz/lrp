r"""Normalization functions and pre-processing of input data.
"""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from typing import Callable, Union

import numpy
import torch
from torchvision import transforms


class ImageNetNorm:
    r"""Normalize batch of images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012-2017 image
    classification and localization dataset.

    Further reading: https://image-net.org/challenges/LSVRC/2012/index.php

    Mean and std are calculated from the dataset ImageNet
    https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92-L93
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # Convert from range [0,1] to custom one.
    normalize: Callable[[torch.Tensor],
                        torch.Tensor] = transforms.Normalize(MEAN, STD)
    # Convert from custom range to [0,1]
    inverse_normalize: Callable[[torch.Tensor],
                                torch.Tensor] = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float),
        normalize
    ])


def _verify_range(img: Union[numpy.array, torch.Tensor], low: float, high: float) -> None:
    r"""Verify that image has values in the range [low, high].

    :param img: Image to be verified
    :param low: Minimum possible value of the pixels in image
    :param high: Maximum possible value of the pixels in image

    :raises ValueError: If image has values outside of the range [low, high]
    """
    f: Callable[[torch.Tensor], torch.Tensor]
    if isinstance(img, torch.Tensor):
        f = torch.all
    elif isinstance(img, numpy.ndarray):
        f = numpy.all
    else:
        raise ValueError(
            f'Expected torch.Tensor or numpy.ndarray, got {type(img)}.')

    if not f((img >= low) & (img <= high)):
        raise ValueError(
            f'Image has values outside of expected range [{low}, {high}]. Actual range [{img.min()}, {img.max()}]')


def denorm_img_pxls(img: Union[numpy.array, torch.Tensor],
                    max: float = 255.0) -> Union[numpy.array, torch.Tensor]:
    r"""Denormalize pixel values in image from [0, 1] to [0, max].

    :param img: Image with values to be denormalized/re-scaled.

    :raises ValueError: If image has values outside of the range [0, 1]

    :returns: Image with values in range [0, max]
    """
    _verify_range(img, 0, 1)

    return img * max


def norm_img_pxls(src_img: numpy.array,
                  min: float = 0.,
                  max: float = 255.0) -> numpy.array:
    r"""Normalize pixel values in image from [min, max] to [0, 1].

    Divide pixels in image by 'max' to normalize pixel values to [0, 1].

    :param src_img: Image with values to be normalized

    :param min: Minimum possible value of the pixels in image
    :param max: Maximum possible value of the pixels in image

    :raises ValueError: If image has values outside of the range [min, max]
    :raises ValueError: If normalized image has values outside of the range [0, 1]

    :returns: Image with values normalized to [0, 1]
    """
    _verify_range(src_img, min, max)

    # Normalize pixel values to [0, 1]
    target_img: numpy.array = src_img / max

    _verify_range(target_img, 0, 1)

    return target_img
