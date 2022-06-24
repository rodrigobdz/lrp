r"""Evaluation metrics for Pixel-Flipping-related results."""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from typing import List, Union

import numpy
import sklearn.metrics


def area_under_the_curve(class_prediction_scores: Union[List[float], numpy.ndarray]) -> float:
    r"""Calculate the area under the curve (AUC).

    The AUC score is not bounded between 0 and 1.

    :param class_prediction_scores: List of class prediction scores.

    :return: Area under the curve.
    """
    x_values: numpy.ndarray = numpy.arange(0, len(class_prediction_scores))
    return sklearn.metrics.auc(x=x_values, y=class_prediction_scores)
