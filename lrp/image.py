r'''Image pre-processing and handling
'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from matplotlib.figure import Figure
import cv2
import torch
import numpy
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from . import norm


def load_normalized_img(path: str) -> numpy.array:
    r'''Load image and normalize pixel values to [0, 1]

    Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb

    :param path: Path to the image
    :returns: Normalized image
    '''
    # Returns a numpy array in BGR color space, not RGB
    img = cv2.imread(path)

    # Convert from BGR to RGB color space
    img = img[..., ::-1]
    # img.shape is (224, 224, 3), where 3 corresponds to RGB channels

    return norm.normalize_rgb_img(img)


def img_to_tensor(img: numpy.array) -> torch.Tensor:
    r'''Convert image as numpy.array to tensor

    Inspired by https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb and zennit
    Add new axis to simulate batch dimension of size 1.
    Convert from NHWC to NCHW format using transpose.
    Set the data type to float.

    Meaning of NCHW format:
        N: number of images in the batch
        C: number of channels of the image (3 for RGB, 1 for grayscale)
        H: height of the image
        W: width of the image

    :param img: Image to be converted
    :returns: Tensor with image data
    '''
    return torch.FloatTensor(img[numpy.newaxis].transpose([0, 3, 1, 2]) * 1)
