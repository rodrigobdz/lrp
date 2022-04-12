r'''Normalization functions and pre-processing of input data.
'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Callable, Union
from torchvision import transforms
import numpy
import torch


class ImageNetNorm:
    r'''Normalize batch of images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012-2017 image
    classification and localization dataset.

    Further reading: https://image-net.org/challenges/LSVRC/2012/index.php

    Mean and std are calculated from the dataset ImageNet
    https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92-L93
    '''
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    normalize: Callable[[torch.Tensor],
                        torch.Tensor] = transforms.Normalize(MEAN, STD)
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


def denorm_img_pxls(img: Union[numpy.array, torch.Tensor],
                    max: float = 255.0) -> Union[numpy.array, torch.Tensor]:
    r'''Denormalize pixel values in image from [0, 1] to [0, max].

    :param img: Image with values to be denormalized/re-scaled.

    :returns: Image with values in range [0, max]
    '''
    # Verify that image has correct range of values
    f: Callable[[torch.Tensor], torch.Tensor]
    if isinstance(img, torch.Tensor):
        f = torch.all
    elif isinstance(img, numpy.ndarray):
        f = numpy.all
    else:
        raise ValueError(
            f'Expected torch.Tensor or numpy.ndarray, got {type(img)}')

    if not f((img >= 0) & (img <= 1)):
        raise ValueError(f'Image has values outside of expected range [0, 1].')

    return img * max


def norm_img_pxls(src_img: numpy.array,
                  min: float = 0.,
                  max: float = 255.0) -> numpy.array:
    r'''Normalize pixel values in image from [min, max] to [0, 1].

    Divide pixels in image by 'max' to normalize pixel values to [0, 1].

    :param src_img: Image with values to be normalized

    :param min: Minimum possible value of the pixels in image
    :param max: Maximum possible value of the pixels in image

    :returns: Image with values normalized to [0, 1]
    '''
    # Verify that image has correct range of values
    if not numpy.all((src_img >= min) & (src_img <= max)):
        raise ValueError(
            f'Image contains values outside of the source range [{min}, {max}]. Verify the passed arguments \'min\' and \'max\'.')

    # Normalize pixel values to [0, 1]
    target_img: numpy.array = src_img / max

    # Verify that the resulting image has the correct range
    if not numpy.all((target_img >= 0) & (target_img <= 1)):
        raise ValueError(
            f'Normalized image contains values outside of the target range [0, 1]. Verify the passed arguments \'min\' and \'max\'.')

    return target_img
