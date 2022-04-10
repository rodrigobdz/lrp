r'''Helpers for Pixel-Flipping algorithm.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


def _loop(generator) -> None:
    r'''Loop over a generator without retrieving any values.

    :param generator: Generator to loop over.
    '''
    for _ in generator:
        pass
