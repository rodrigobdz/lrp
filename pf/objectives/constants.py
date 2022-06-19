r"""Sorting objectives constants for Pixel-Flipping and Region Perturbation algorithms."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


class PixelFlippingObjectives:  # pylint: disable=too-few-public-methods
    r"""Objectives for Pixel-Flipping Algorithm."""

    MoRF: str = 'Most Relevant First'
    LRF: str = 'Least Relevant First'
    RANDOM: str = 'Random'
