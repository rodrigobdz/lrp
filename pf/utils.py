r"""Helpers for Pixel-Flipping algorithm."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

import torch


def _loop(generator) -> None:
    r"""Loop over a generator without retrieving any values.

    :param generator: Generator to loop over.
    """
    for _ in generator:
        pass


def ensure_nchw_format(input_nchw: torch.Tensor) -> None:
    r"""Ensure that the input tensor is in NCHW format.

    :param input_nchw: Input tensor in NCHW format.

    :raise ValueError: If the input tensor is not in NCHW format.
    """
    if input_nchw.dim() == 4:
        return
    raise ValueError(
        f'Input tensor must be in NCHW format. Got {input_nchw.dim()} dimensions and shape {input_nchw.shape}.')


def _ensure_non_overlapping_patches_possible(input_nchw: torch.Tensor,
                                             perturbation_size: int) -> None:
    r"""Ensure input can be divided into non-overlapping patches of said size.

    :param input_nchw: Input tensor in NCHW format.
    :param perturbation_size: Size of the patches.

    :raise ValueError: If the input cannot be divided into non-overlapping patches.
    """
    # Ensure input image has square shape (equal height and width).
    input_height: int = input_nchw.size(dim=2)
    input_width: int = input_nchw.size(dim=3)
    if input_height != input_width:
        raise ValueError(
            f'Input image has shape {input_nchw.size()} which is not square.')

    # Ensure perturbation size is evenly divisible by the dimensions of the input image.
    # This requirement ensures that regions are non-overlapping.
    if input_height % perturbation_size == 0:
        return
    raise ValueError(f'Perturbation size ({perturbation_size}) is not evenly \
        divisible by input image height ({input_height}).\
        Dividing the input image into non-overlapping patches is not possible.')


def _verify_batch_size(*tensors) -> None:
    r"""Verify that all tensors have the same batch size.

    :param tensors: Tensors to verify.

    :raises ValueError: If the batch sizes are different.
    """
    batch_size: int = tensors[0].shape[0]

    for n, tensor in enumerate(tensors):
        ensure_nchw_format(tensor)

        # Verify that tensor has the same batch size.
        if tensor.shape[0] == batch_size:
            continue

        raise ValueError(
            f'All tensors must have the same batch size. Batch size is {batch_size} but tensor {n} has shape {tensor.shape[0]}.')
