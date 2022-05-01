r"""Plotting library
"""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

import torch
from matplotlib import pyplot as plt

import lrp.plot


def plot_image_comparison(batch_size: int,
                          original_input_nchw: torch.Tensor,
                          flipped_input_nchw: torch.Tensor,
                          relevance_scores_nchw: torch.Tensor,
                          acc_flip_mask_nhw: torch.Tensor,
                          show_plot: bool = True) -> None:
    r"""Plot the original and flipped input images alongside the relevance scores
    of the pixels that were flipped.

    :param batch_size: Batch size of the input images.
    :param original_input_nchw: Original input images.
    :param flipped_input_nchw: Flipped input images.
    :param relevance_scores_nchw: Relevance scores of the pixels that were flipped.
    :param acc_flip_mask_nhw: Mask of pixels that were flipped.
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

        # Plot heatmaps.
        lrp.plot.heatmap(relevance_scores_hw.detach().numpy(),
                         fig=ax[1][0], **plot_kwargs)
        # Plot heatmap of perturbed regions.
        lrp.plot.heatmap(acc_flip_mask_hw, fig=ax[1][1], **plot_kwargs)

        x: int = 75
        y: int = -10
        size: int = 12

        # Add captions.
        ax[0][0].text(x, y, 'Original Input', size=size)
        ax[0][1].text(x, y, 'Flipped Input', size=size)
        ax[1][0].text(x, y, 'Relevance scores', size=size)
        ax[1][1].text(x, y, 'Perturbed Regions', size=size)

        # tight_layout automatically adjusts subplot params so that subplot(s) fits into the figure area.
        plt.rcParams["savefig.bbox"] = 'tight'
        plt.tight_layout()

        if show_plot:
            # Show plots.
            plt.show()
