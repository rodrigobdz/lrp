r'''Normalization functions and pre-processing of input data.
'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'









    '''



def normalize_rgb_img(img: numpy.array, max_value: float = 255.0) -> numpy.array:
    r'''Normalize RGB image

    Divide by 255 (max. RGB value) to normalize pixel values to [0,1]

    :param img: RGB image
    :param max_value: Maximum value of the RGB image

    :returns: Normalized RGB image
    '''
    return img / max_value
