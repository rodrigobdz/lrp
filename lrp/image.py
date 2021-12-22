r'''Visualization functions and pre-processing of input data
'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


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

    :param img: Image to be converted
    :returns: Tensor with image data
    '''
    normalized_img = norm.ILSVRC2012_BatchNorm()(
        torch.FloatTensor(img[numpy.newaxis].transpose([0, 3, 1, 2]) * 1)
    )
    return normalized_img


def heatmap(relevance_scores: numpy.array, width: float, height: float):
    r'''Plot heatmap of relevance scores

    :param relevance_scores: Relevance scores in pixel layer only
    :param width: Size of the image in x-direction
    :param height: Size of the image in y-direction
    '''
    abs_bound = 10 * ((numpy.abs(relevance_scores) ** 3.0).mean() ** (1.0 / 3))

    cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    cmap[:, 0:3] *= 0.85
    cmap = ListedColormap(cmap)

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(relevance_scores, cmap=cmap, vmin=-abs_bound,
               vmax=abs_bound, interpolation='nearest')

    plt.show()
