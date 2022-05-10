r"""Constants for perturbation techniques available for Pixel-Flipping and Region Perturbation."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


class PerturbModes:  # pylint: disable=too-few-public-methods
    r"""Constants for perturbation techniques."""

    RANDOM: str = 'Random sampling'
    INPAINTING: str = 'Inpainting'
