r"""Objectives to sort relevance scores in Pixel-Flipping Algorithm.

Defines the order in which the relevance scores are flipped.
"""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from typing import Generator

import numpy
import torch
from torchvision import transforms

from pf import sanity_checks, utils


class PixelFlippingObjectives:  # pylint: disable=too-few-public-methods
    r"""Objectives for Pixel-Flipping Algorithm."""

    MORF: str = 'Most Relevant First'


def _argsort(patches_nmmpp: torch.Tensor,
             perturbation_size: int,
             objective: str = PixelFlippingObjectives.MORF) -> torch.Tensor:
    r"""Sort relevance scores in order defined by objective.

    :param patches_nmmpp: Patches in NMMPP format.
    :param objective: Sorting order for relevance scores.

    :returns: Sorted relevance scores as a tensor with N one-dimensional lists,
              one for each image in the batch of size and each list with m elements,
              m is # relevance scores for each image. Shape is (N, m).
    """
    if objective != PixelFlippingObjectives.MORF:
        raise NotImplementedError(f'Objective {objective} not supported.')

    # Controls the sorting order (ascending or descending).
    # Set default value to descending—i.e., most relevant first.
    descending: bool = True

    # TODO: Add switch case to implement the user's selected objective.

    # Sort relevance scores according to objective
    batch_size: int = utils.get_batch_size(input_nchw=patches_nmmpp)

    # Flatten each patch
    # Notation:
    #   n: batch size
    #   m2: number of patches per image squared. (m2 = m * m)
    #   p2: perturbation/patch size squared. (p2 = p * p)
    patches_nm2p2: torch.Tensor = patches_nmmpp.reshape(batch_size,
                                                        -1,
                                                        perturbation_size * perturbation_size)

    # Sum each patch
    sum_patches_nm2: torch.Tensor = patches_nm2p2.sum(dim=2)

    # Sort patches by their sum
    order_nm2: torch.Tensor = sum_patches_nm2.argsort(descending=descending)

    return order_nm2


def _patchify(input_n1hw: torch.Tensor,
              perturbation_size: int) -> torch.Tensor:
    r"""Divide tensor into patches of desired size.

    :param input_n1hw: Input tensor in NCHW format.
    :param perturbation_size: Size of the patch to flip.

    :returns: Patched input tensor in NCHW format.
    """
    sanity_checks.ensure_nchw_format(input_nchw=input_n1hw)

    # Patch size.
    kernel_size: int = perturbation_size
    # Stride of kernel size ensures non-overlapping patches.
    stride: int = perturbation_size

    return input_n1hw.data.unfold(dimension=2,
                                  size=kernel_size,
                                  step=stride).unfold(dimension=3,
                                                      size=kernel_size,
                                                      step=stride)


def _create_flip_mask(order_nm_flat: torch.Tensor,
                      num_patches_per_img: int,
                      width: int,
                      height: int,
                      perturbation_size: int,
                      batch_index: int,
                      patch_index: int) -> torch.Tensor:
    r"""Create a mask to flip a patch.

    :param order_nm_flat: Sorted relevance scores as a tensor with N one-dimensional lists
    :param num_patches_per_img: Number of patches per image.
    :param width: Width of the image.
    :param height: Height of the image.
    :param perturbation_size: Size of the patch to flip.
    :param batch_index: Index of the image in the batch.
    :param patch_index: Index of the patch in the image.

    :returns: Mask to flip a patch.
    """
    # Get 1D index of patch to flip
    # item() is used to convert a one-element tensor to a scalar.
    # Cast to int because order_nm_flat is a tensor of type float.
    flat_index: int = int(order_nm_flat[batch_index, patch_index].item())

    # Convert index of patch to flip from 1D to 2D
    # pylint: disable=unbalanced-tuple-unpacking
    i, j = numpy.unravel_index(indices=flat_index,
                               shape=(num_patches_per_img, num_patches_per_img))

    # Create mask to flip pixels/patches in input located at the index of the
    # threshold value in the sorted relevance scores.
    mask_1hw: torch.Tensor = torch.zeros(1,
                                         width,
                                         height,
                                         dtype=torch.bool)

    # Create mask with patch at coordinates (i, j) marked as True.
    #
    # l and k are the origin coordinates in the patchified grid, thus the conversion
    # to the original image coordinates by multiplying by perturbation size.
    #
    # i and j are the origin coordinates of the selected patch.
    return transforms.functional.erase(img=mask_1hw,
                                       i=i*perturbation_size,
                                       j=j*perturbation_size,
                                       h=perturbation_size,
                                       w=perturbation_size,
                                       v=True)


def flip_mask_generator(relevance_scores_nchw: torch.Tensor,
                        perturbation_size: int) -> Generator[torch.Tensor, None, None]:
    r"""Create masks with pixel(s) selected for flipping.

    :param relevance_scores_nchw: Relevance scores in NCHW format.
    :param perturbation_size: Size of the region to flip.
    A size of 1 corresponds to single pixels, whereas a higher number to patches of size nxn.

    :raises ValueError: If relevance scores are not in square format.

    :yields: Mask in N1HW (1-channel) format to flip pixels/patches input in order specified
    by sorted_values_nm.
    """
    batch_size: int = utils.get_batch_size(input_nchw=relevance_scores_nchw)
    height, width = utils.get_height_width(relevance_scores_nchw)

    # Get number of patches per image.
    num_patches_per_img: int = width // perturbation_size
    max_num_patches: int = num_patches_per_img * num_patches_per_img

    # Reduce number of channels to one.
    relevance_scores_n1hw: torch.Tensor = transforms.Grayscale()(relevance_scores_nchw)

    # 1. Divide tensor into patches
    # Notation:
    #   p: perturbation_size
    #   m: modified width = original width/p.
    #      Eg. m=224/8=28, p=8, original width=224
    #      28 is the number of patches of size 8 in the image.
    #
    # Shape of patches_nmmpp: (batch_size, m, m, p, p) = (batch_size, 28, 28, 8, 8)
    patches_nmmpp: torch.Tensor = _patchify(input_n1hw=relevance_scores_n1hw,
                                            perturbation_size=perturbation_size)

    # 2. Sort patches by their sum
    order_nm_flat: torch.Tensor = _argsort(patches_nmmpp,
                                           perturbation_size,
                                           objective=PixelFlippingObjectives.MORF)

    # Loop over patches to remove.
    for patch_index in range(max_num_patches):
        # Create empty boolean tensor of size 0.
        mask_nhw: torch.Tensor = torch.zeros(0, dtype=torch.bool)

        # Loop over images in batch.
        for batch_index in range(batch_size):
            # 3. Create mask from individual patch
            flip_mask_1hw: torch.Tensor = _create_flip_mask(order_nm_flat=order_nm_flat,
                                                            num_patches_per_img=num_patches_per_img,
                                                            width=width,
                                                            height=height,
                                                            perturbation_size=perturbation_size,
                                                            batch_index=batch_index,
                                                            patch_index=patch_index)

            # Concatenate the mask for the current image to the list of masks.
            # Initially mask_nhw is empty and masks for each image are added incrementally.
            # Shape of mask_nhw is (N, H, W).
            mask_nhw = torch.cat((mask_nhw, flip_mask_1hw))

        # unsqueze(1) creates artificial channel dimension to make mask in N1HW format from NHW.
        # Shape of mask_n1hw is (N, 1, H, W).
        mask_n1hw: torch.Tensor = mask_nhw.unsqueeze(1)

        yield mask_n1hw
