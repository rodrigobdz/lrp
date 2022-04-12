r'''Normalization functions and pre-processing of input data.
'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


import numpy


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
