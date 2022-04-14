r'''Evaluation metrics for Pixel-Flipping-related results.'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import List
import numpy


def area_over_the_curve(class_prediction_scores: List[float]) -> float:
    r'''Calculate the area over the curve (AOC) using the formula 12 from paper:

        Samek, Wojciech, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, and Klaus-Robert Müller.
        "Evaluating the visualization of what a deep neural network has learned."
        IEEE transactions on neural networks and learning systems 28, no. 11 (2016): 2660-2673.

        (Chicago-style citation)

    :param y: List of Y values.

    :return: Area over the curve.
    '''
    y: numpy.array = numpy.array(class_prediction_scores)

    # L stands for the number of perturbation steps
    L: int = len(y)

    # y[0] - y represents the deviation of the class prediction scores (y) from the ground truth (y[0]).
    return (y[0] - y).sum() / (L + 1)
