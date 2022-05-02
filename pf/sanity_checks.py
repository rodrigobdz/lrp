r"""Sanity Checks for Pixel-Flipping algorithm."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

import torch

from . import utils


def ensure_nchw_format(input_nchw: torch.Tensor) -> None:
    r"""Ensure that the input tensor is in NCHW format.

    :param input_nchw: Input tensor in NCHW format.

    :raise ValueError: If the input tensor is not in NCHW format.
    """
    if input_nchw.dim() == 4:
        return

    raise ValueError(f"""Input tensor must be in NCHW format.
Got {input_nchw.dim()} dimensions and shape {input_nchw.shape}.""")


def ensure_chw_format(input_chw: torch.Tensor) -> None:
    r"""Ensure that the input tensor is in CHW format.

    :param input_chw: Input tensor in CHW format.

    :raise ValueError: If the input tensor is not in CHW format.
    """
    if input_chw.dim() == 3:
        return

    raise ValueError(f"""Input tensor must be in CHW format.
Got {input_chw.dim()} dimensions and shape {input_chw.shape}.""")


def ensure_non_overlapping_patches_possible(input_nchw: torch.Tensor,
                                            perturbation_size: int) -> None:
    r"""Ensure input can be divided into non-overlapping patches of said size.

    :param input_nchw: Input tensor in NCHW format.
    :param perturbation_size: Size of the patches.

    :raise ValueError: If the input cannot be divided into non-overlapping patches.
    """
    verify_square_input(input_nchw)
    height, _ = utils.get_height_width(input_nchw)

    # Ensure perturbation size is evenly divisible by the dimensions of the input image.
    # This requirement ensures that regions are non-overlapping.
    if height % perturbation_size == 0:
        return
    raise ValueError(f"""Perturbation size ({perturbation_size}) is not evenly divisible by
input image height ({height}).
Dividing the input image into non-overlapping patches is not possible.""")


def verify_square_input(*tensors) -> None:
    r"""Verify that the input is square (equal height and width).

    :param tensors: Tensors to verify.

    :raises ValueError: If the input is not square.
    """
    for tensor_idx, tensor in enumerate(tensors):
        height, width = utils.get_height_width(tensor)

        if width == height:
            return

        raise ValueError(f"""Input tensor must be square.
Tensor {tensor_idx} has width {width} and height {height}.""")


def verify_batch_size(*tensors) -> None:
    r"""Verify that all tensors have the same batch size.

    :param tensors: Tensors to verify.

    :raises ValueError: If the batch sizes are different.
    """
    batch_size: int = utils.get_batch_size(input_nchw=tensors[0])

    for batch_index, tensor in enumerate(tensors):
        ensure_nchw_format(tensor)

        # Verify that tensor has the same batch size.
        if utils.get_batch_size(tensor) == batch_size:
            continue

        raise ValueError(f"""All tensors must have the same batch size. Batch size
is {batch_size} but tensor {batch_index} has batch size {utils.get_batch_size(tensor)}.""")


def verify_perturbation_args(input_nchw: torch.Tensor,
                             perturbation_size: int,
                             perturbation_steps: int) -> None:
    r"""Verify that the perturbation arguments are valid.

    :param input_nchw: Input tensor in NCHW format.
    :param perturbation_size: Size of the patches.
    :param perturbation_steps: Number of steps to take in the perturbation.

    :raises ValueError: If the perturbation arguments are invalid.
    """
    verify_square_input(input_nchw)

    _, width = utils.get_height_width(input_nchw)

    # FIXME: Calculation is only valid if a patch is removed each step.
    # Get number of patches per image.
    num_patches_per_img: int = width // perturbation_size
    max_num_patches: int = num_patches_per_img * num_patches_per_img
    max_perturbation_steps: int = max_num_patches

    if perturbation_steps > max_perturbation_steps:
        raise ValueError(f"""Perturbation steps ({perturbation_steps}) cannot be greater than
the maximum number of steps ({max_perturbation_steps}).""")
