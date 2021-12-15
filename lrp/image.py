r'''Visualization functions and pre-processing of input data
'''

__author__ = 'Rodrigo Bermudez Schettino'
__credits__ = ['Rodrigo Bermudez Schettino']
__maintainer__ = 'Rodrigo Bermudez Schettino'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


import cv2
import torch
import numpy
from typing import List
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class StandardScoreNormalization:
    r'''Normalize matrix by calculating standard score, i.e., subtracting mean and dividing by standard deviation of the dataset.

    Source: https://github.com/chr5tphr/zennit/blob/cc9ac0f3016e1b842f2c60e8986c794b2ae7096e/share/example/feed_forward.py#L32-L38
    '''

    def __init__(self, mean: List[float], std: List[float]) -> None:
        r'''Convert mean and standard deviation to tensors

        :param mean: Mean of the dataset
        :param std: Standard deviation of the dataset
        '''
        self.mean = torch.tensor(mean)[None, :, None, None]
        self.std = torch.tensor(std)[None, :, None, None]

    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        r'''Caclculate standard score

        :param matrix: Matrix to be normalized
        :returns: Normalized matrix
        '''
        return (matrix - self.mean) / self.std


class ILSVRC2012_BatchNormalize(StandardScoreNormalization):
    r'''Normalize batch of images from ILSVRC2012 dataset

    Mean and std are calculated from the dataset ImageNet
    https://github.com/Cadene/pretrained-models.pytorch/blob/8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0/pretrainedmodels/models/torchvision_models.py#L64-L65
    '''

    def __init__(self):
        super().__init__(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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
    # Divide by 255 (max. RGB value) to normalize pixel values to [0,1]
    return img / 255.0


def img_to_tensor(img: numpy.array) -> torch.Tensor:
    r'''Convert image as numpy.array to tensor

    Inspired by https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb and zennit

    :param img: Image to be converted
    :returns: Tensor with image data
    '''
    normalized_img = ILSVRC2012_BatchNormalize()(
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
