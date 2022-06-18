r"""Plotting library."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

from typing import List

import matplotlib.ticker as plticker
import numpy
import torch
from matplotlib import pyplot as plt

import lrp.plot


def plot_number_of_flips_per_step(number_of_flips_per_step_arr: List[int],
                                  max_perturbation_steps: int) -> None:
    r"""Plot number of flips per step.

    :param number_of_flips_per_step_dict: Number of pixels that were flipped per step.
    :param max_perturbation_steps: Maximum number of perturbation steps.
    """
    current_perturbation_step: int = len(number_of_flips_per_step_arr) - 1
    xticks = numpy.arange(0, len(number_of_flips_per_step_arr))

    # Plot number of flips per step.
    plt.plot(xticks, number_of_flips_per_step_arr, marker='.', linestyle='')

    # FIXME: Calculate offset for text annotations dynamically.
    # This is a hack to show annotations on the plot, optimized for 28 perturbation
    # steps (hard-coded).
    offsets_by_perturbation_step_x: numpy.ndarray = numpy.logspace(start=-3,
                                                                   stop=0.02,
                                                                   num=max_perturbation_steps,
                                                                   endpoint=True)
    offsets_by_perturbation_step_x = numpy.append(arr=offsets_by_perturbation_step_x,
                                                  values=[0.3])
    offsets_by_perturbation_step_y: numpy.ndarray = numpy.logspace(start=-1.4,
                                                                   stop=0.3,
                                                                   num=max_perturbation_steps,
                                                                   endpoint=True)
    offsets_by_perturbation_step_y = numpy.append(arr=offsets_by_perturbation_step_y,
                                                  values=[10.])

    # Add offset to avoid overlapping markers with function values.
    offset_x: float = offsets_by_perturbation_step_x[current_perturbation_step]
    offset_y: float = offsets_by_perturbation_step_y[current_perturbation_step]
    # Annotate plot with number of steps as text next to each marker.
    for idx, val in enumerate(number_of_flips_per_step_arr):
        plt.annotate(text=val,
                     xy=(idx - offset_x,
                         val + offset_y)
                     )

    plt.xticks(xticks)
    plt.title(f'Total number of flips: {sum(number_of_flips_per_step_arr)}')
    plt.ylabel('Number of simultaneous flip per step')
    plt.xlabel('Perturbation step')
    plt.margins(0.1, tight=True)
    # Add padding for better alignment of (sup)title
    # Source: https://stackoverflow.com/a/45161551
    plt.tight_layout(rect=[0, 0, 1.3, 1])

    plt.show()


def plot_image_comparison(batch_size: int,
                          original_input_nchw: torch.Tensor,
                          flipped_input_nchw: torch.Tensor,
                          before_last_step: bool,
                          relevance_scores_nchw: torch.Tensor,
                          acc_flip_mask_nhw: torch.Tensor,
                          perturbation_size: int,
                          show_plot: bool = True) -> None:
    r"""Plot original and flipped input images alongside the relevances scores that were flipped.

    :param batch_size: Batch size of the input images.
    :param original_input_nchw: Original input images.
    :param flipped_input_nchw: Flipped input images.
    :param before_last_step: Whether the flipped input is plotted before the last step
                             of the perturbation.
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
            0).cpu()
        flipped_input_1chw: torch.Tensor = flipped_input_nchw[batch_index].unsqueeze(
            0).cpu()
        # sum along the channel dimension to convert from (C, W, H) to (W, H)
        relevance_scores_hw: torch.Tensor = relevance_scores_nchw[batch_index].sum(
            dim=0).cpu()
        acc_flip_mask_hw: torch.Tensor = acc_flip_mask_nhw[batch_index].cpu()

        # Create grid of original and perturbed images.
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

        # Plot images.
        lrp.plot.plot_imagenet(
            original_input_1chw,
            ax=axes[0][0],
            show_plot=plot_kwargs['show_plot'])

        lrp.plot.plot_imagenet(
            flipped_input_1chw,
            ax=axes[0][1],
            show_plot=plot_kwargs['show_plot'])

        # Rotate ticks on x axis to make them readable.
        # FIXME: Only works for last image in grid.
        plt.xticks(rotation=80, fontsize=9)
        plt.yticks(fontsize=9)
        # Turn on grid for selected heatmaps.
        for heatmap_idx in [1]:
            # Show grid to distinguish patches from each other in perturbed regions plot.
            # Set the gridding interval: here we use the major tick interval
            locator = plticker.MultipleLocator(base=perturbation_size)
            axes[1][heatmap_idx].xaxis.set_major_locator(locator)
            axes[1][heatmap_idx].yaxis.set_major_locator(locator)
            axes[1][heatmap_idx].grid(visible=True,
                                      which='major',
                                      axis='both',
                                      linestyle='-',
                                      color='w',
                                      linewidth=1)

        # Plot heatmaps.
        # Use Tensor.cpu() to copy the tensor to host memory before converting to numpy().
        lrp.plot.heatmap(relevance_scores=relevance_scores_hw.cpu().detach().numpy(),
                         fig=axes[1][0],
                         show_axis=True,
                         **plot_kwargs)

        # Plot heatmap of perturbed regions.
        lrp.plot.heatmap(relevance_scores=acc_flip_mask_hw,
                         fig=axes[1][1],
                         show_axis=True,
                         **plot_kwargs)

        x_position: int = 75
        y_position: int = -10
        size: int = 12

        # Add extra leading whitespace to center multiline text.
        flipped_input_title: str = '    Flipped input'
        if before_last_step:
            flipped_input_title += '\n(penultimate step)'

        # Add captions.
        axes[0][0].text(x=x_position,
                        y=y_position,
                        s='Original Input',
                        size=size)

        axes[0][1].text(x=x_position,
                        y=y_position,
                        s=flipped_input_title,
                        size=size)

        axes[1][0].text(x=x_position,
                        y=y_position,
                        s='Relevance scores',
                        size=size)

        axes[1][1].text(x=x_position,
                        y=y_position,
                        s='Perturbed Regions',
                        size=size)

        # tight_layout automatically adjusts subplot params so that
        # subplot(s) fits into the figure area.
        plt.rcParams["savefig.bbox"] = 'tight'
        plt.tight_layout()

        if show_plot:
            # Show plots.
            plt.show()
