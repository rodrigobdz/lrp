r'''Evaluation metrics for Pixel-Flipping-related results.'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import List, Union

import numpy
import sklearn.metrics


def area_under_the_curve(class_prediction_scores: Union[List[float], numpy.ndarray]) -> float:
    r'''Calculate the area under the curve (AUC).

    The AUC score is not bounded between 0 and 1.

    :param y: List of Y values.

    :return: Area under the curve.
    '''
    x: numpy.array = numpy.arange(0, len(class_prediction_scores))
    return numpy.round_(sklearn.metrics.auc(x=x, y=class_prediction_scores), decimals=2)


def area_over_the_pertubation_curve(class_prediction_scores: Union[List[float], numpy.ndarray]) -> float:
    r'''Calculate the area over the perturbation curve (AOPC) using the formula 12 from paper:

        Samek, Wojciech, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, and Klaus-Robert Müller.
        "Evaluating the visualization of what a deep neural network has learned."
        IEEE transactions on neural networks and learning systems 28, no. 11 (2016): 2660-2673.

        (Chicago-style citation)

    :param class_prediction_scores: List of Y values.

    :return: Area over the perturbation curve.
    '''
    y: numpy.array = numpy.array(class_prediction_scores)

    # L stands for the number of perturbation steps
    L: int = len(y)

    # y[0] - y represents the deviation of the class prediction scores (y) from the ground truth (y[0]).
    # The first element is the ground truth because it is the class prediction score before the perturbation.
    return numpy.round_((y[0] - y).sum() / (L + 1), decimals=2)
