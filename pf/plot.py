r"""Plotting library."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

from typing import Dict
import matplotlib.ticker as plticker
import torch
from matplotlib import pyplot as plt

import lrp.plot


def plot_number_flips_per_step(number_flips_per_step: Dict[int, int]) -> None:
    r"""Plot number of flips per step.

    :param number_flips_per_step: Number of pixels that were flipped per step.
    """
    # Plot number of flips per step.
    plt.plot(number_flips_per_step.values(), marker='.', linestyle='')

    # Add offset to avoid overlapping markers with function values.
    offset: float = 0.2
    # Annotate plot with number of steps as text next to each marker.
    for idx, val in enumerate(number_flips_per_step.values()):
        if val == 1:
            break
        plt.annotate(val, xy=(idx+offset, val))

    plt.title('Number of simultaneous flip per step')
    plt.margins(0.1, tight=True)
    # Add padding for better alignment of (sup)title
    # Source: https://stackoverflow.com/a/45161551
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()


def plot_image_comparison(batch_size: int,
                          original_input_nchw: torch.Tensor,
                          flipped_input_nchw: torch.Tensor,
                          relevance_scores_nchw: torch.Tensor,
                          acc_flip_mask_nhw: torch.Tensor,
                          perturbation_size: int,
                          show_plot: bool = True) -> None:
    r"""Plot original and flipped input images alongside the relevances scores that were flipped.

    :param batch_size: Batch size of the input images.
    :param original_input_nchw: Original input images.
    :param flipped_input_nchw: Flipped input images.
    :param relevance_scores_nchw: Relevance scores of the pixels that were flipped.
    :param acc_flip_mask_nhw: Mask of pixels that were flipped.
    :param perturbation_size: Size of the perturbation.
    :param show_plot: If True, show the plot.
    """
    # Show plot must be False here to display image grid.
    plot_kwargs: dict = {'width': 5, 'height': 5, 'show_plot': False}
    for batch_index in range(batch_size):

        # tensor[n] returns an image tensor of shape (C, W, H)
        # unsqueeze adds a new dimension to the tensor to make it of shape (1, C, W, H)
        original_input_1chw: torch.Tensor = original_input_nchw[batch_index].unsqueeze(
            0)
        flipped_input_1chw: torch.Tensor = flipped_input_nchw[batch_index].unsqueeze(
            0)
        # sum along the channel dimension to convert from (C, W, H) to (W, H)
        relevance_scores_hw: torch.Tensor = relevance_scores_nchw[batch_index].sum(
            dim=0)
        acc_flip_mask_hw: torch.Tensor = acc_flip_mask_nhw[batch_index]

        # Create grid of original and perturbed images.
        _, ax = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

        # Plot images.
        lrp.plot.plot_imagenet(
            original_input_1chw,
            ax=ax[0][0],
            show_plot=plot_kwargs['show_plot'])

        lrp.plot.plot_imagenet(
            flipped_input_1chw,
            ax=ax[0][1],
            show_plot=plot_kwargs['show_plot'])

        # Rotate ticks on x axis to make them readable.
        # FIXME: Only works for last image in grid.
        plt.xticks(rotation=80)
        # Turn on grid for selected heatmaps.
        for heatmap_idx in [1]:
            # Show grid to distinguish patches from each other in perturbed regions plot.
            # Set the gridding interval: here we use the major tick interval
            locator = plticker.MultipleLocator(base=perturbation_size)
            ax[1][heatmap_idx].xaxis.set_major_locator(locator)
            ax[1][heatmap_idx].yaxis.set_major_locator(locator)
            ax[1][heatmap_idx].grid(visible=True,
                                    which='major',
                                    axis='both',
                                    linestyle='-',
                                    color='w',
                                    linewidth=1)

        # Plot heatmaps.
        lrp.plot.heatmap(relevance_scores=relevance_scores_hw.detach().numpy(),
                         fig=ax[1][0],
                         show_axis=True,
                         **plot_kwargs)

        # Plot heatmap of perturbed regions.
        lrp.plot.heatmap(relevance_scores=acc_flip_mask_hw,
                         fig=ax[1][1],
                         show_axis=True,
                         **plot_kwargs)

        x_position: int = 75
        y_position: int = -10
        size: int = 12

        # Add captions.
        ax[0][0].text(x=x_position,
                      y=y_position,
                      s='Original Input',
                      size=size)

        ax[0][1].text(x=x_position,
                      y=y_position,
                      s='Flipped Input',
                      size=size)

        ax[1][0].text(x=x_position,
                      y=y_position,
                      s='Relevance scores',
                      size=size)

        ax[1][1].text(x=x_position,
                      y=y_position,
                      s='Perturbed Regions',
                      size=size)

        # tight_layout automatically adjusts subplot params so that subplot(s) fits into the figure area.
        plt.rcParams["savefig.bbox"] = 'tight'
        plt.tight_layout()

        if show_plot:
            # Show plots.
            plt.show()
