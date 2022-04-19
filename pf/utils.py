r'''Helpers for Pixel-Flipping algorithm.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'

import torch


def _loop(generator) -> None:
    r'''Loop over a generator without retrieving any values.

    :param generator: Generator to loop over.
    '''
    for _ in generator:
        pass


def _ensure_nchw_format(input_nchw: torch.Tensor) -> None:
    r'''Ensure that the input tensor is in NCHW format.

    :param input_nchw: Input tensor in NCHW format.

    :raise ValueError: If the input tensor is not in NCHW format.
    '''
    if input_nchw.dim() == 4:
        return
    raise ValueError(
        f'Input tensor must be in NCHW format. Got {input_nchw.dim()} dimensions and shape {input_nchw.shape}.')


def _verify_batch_size(*tensors) -> None:
    r'''Verify that all tensors have the same batch size.

    :param tensors: Tensors to verify.

    :raises ValueError: If the batch sizes are different.
    '''
    batch_size: int = tensors[0].shape[0]

    for n, tensor in enumerate(tensors):
        _ensure_nchw_format(tensor)

        # Verify that tensor has the same batch size.
        if tensor.shape[0] == batch_size:
            continue

        raise ValueError(
            f'All tensors must have the same batch size. Batch size is {batch_size} but tensor {n} has shape {tensor.shape[0]}.')
