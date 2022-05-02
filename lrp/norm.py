r"""Normalization functions and pre-processing of input data."""

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


class ImageNetNorm:  # pylint: disable=too-few-public-methods
    r"""Normalize image batch from ILSVRC 2012-2017 image classification and localization dataset.

    ILSVRC: ImageNet Large Scale Visual Recognition Challenge
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


def _verify_range(img: Union[numpy.ndarray, torch.Tensor], low: float, high: float) -> None:
    r"""Verify that image has values in the range [low, high].

    :param img: Image to be verified
    :param low: Minimum possible value of the pixels in image
    :param high: Maximum possible value of the pixels in image

    :raises ValueError: If image has values outside of the range [low, high]
    """
    verification_func: Callable[[torch.Tensor], torch.Tensor]
    if isinstance(img, torch.Tensor):
        verification_func = torch.all
    elif isinstance(img, numpy.ndarray):
        verification_func = numpy.all
    else:
        raise ValueError(
            f'Expected torch.Tensor or numpy.ndarray, got {type(img)}.')

    if not verification_func((img >= low) & (img <= high)):
        raise ValueError(f"""Image has values outside of expected range [{low}, {high}].
Actual range [{img.min()}, {img.max()}]""")


def denorm_img_pxls(img: Union[numpy.ndarray, torch.Tensor],
                    max_pixel_val: float = 255.0) -> Union[numpy.ndarray, torch.Tensor]:
    r"""Denormalize pixel values in image from [0, 1] to [0, max_pixel_val].

    :param img: Image with values to be denormalized/re-scaled.
    :param max_pixel_val: Maximum possible value of the pixels in image.

    :raises ValueError: If image has values outside of the range [0, 1]

    :returns: Image with values in range [0, max_pixel_val]
    """
    _verify_range(img, 0, 1)

    return img * max_pixel_val


def norm_img_pxls(src_img: Union[numpy.ndarray, torch.Tensor],
                  min_pixel_val: float = 0.,
                  max_pixel_val: float = 255.0) -> Union[numpy.ndarray, torch.Tensor]:
    r"""Normalize pixel values in image from [min_pixel_val, max_pixel_val] to [0, 1].

    Divide pixels in image by 'max_pixel_val' to normalize pixel values to [0, 1].

    :param src_img: Image with values to be normalized

    :param min_pixel_val: Minimum possible value of the pixels in image
    :param max_pixel_val: Maximum possible value of the pixels in image

    :raises ValueError: If image has values outside of the range [min_pixel_val, max_pixel_val]
    :raises ValueError: If normalized image has values outside of the range [0, 1]

    :returns: Image with values normalized to [0, 1]
    """
    _verify_range(src_img, min_pixel_val, max_pixel_val)

    # Normalize pixel values to [0, 1]
    target_img: numpy.ndarray = src_img / max_pixel_val

    _verify_range(target_img, 0, 1)

    return target_img
