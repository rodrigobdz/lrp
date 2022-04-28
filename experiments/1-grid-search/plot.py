r'''Grid plotting function for grid search experiments.'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'

from typing import Any, Callable, List, Optional, Tuple

import torch
from matplotlib import pyplot as plt

import lrp.plot
from pf.convert_img import arr_chw_to_hwc


def grid(results: List[Tuple[float, torch.Tensor]],
         image_chw_rgb: torch.Tensor,
         title: str,
         filename: str,
         gridsize: Tuple[int, int],
         param_name: str,
         transform: Optional[Callable] = None,
         param_print: Callable[[Any], Any] = lambda p: p,
         figsize: List[int] = [10, 10],
         alpha: float = 0.2
         ) -> None:
    r'''Plot the results of the hyperparameter grid search.

    Source: https://stackoverflow.com/a/46616645

    :param results: List of tuples of the form (rule_param, R)
    :param image_chw_rgb: Reference image in CHW format with RGB color format used to calculate the relevance scores
    :param title: Title of the plot
    :param filename: Name of the file to save the plot to
    :param gridsize: Number of rows and columns in the grid
    :param param_name: Name of the parameter to plot
    :param param_print: Function to print the parameter value
    :param figsize: Figure size
    :param alpha: Alpha value for the reference image
    '''

    # Settings
    grid_rows, grid_cols = gridsize
    fontsize = figsize[0]*2
    dpi = 150

    # Create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=figsize)
    fig.suptitle(title, fontsize=fontsize)

    # Plot simple raster image on each sub-plot
    # axi is equivalent with ax[row][col]
    for i, axi in enumerate(ax.flat):
        rule_param, R = results[i]
        relevance_scores = R[0].sum(dim=0).detach().numpy()

        axi.set_title(f'{param_name}={param_print(rule_param)}',
                      fontsize=fontsize*0.5)

        # Plot heatmap
        lrp.plot.heatmap(relevance_scores=relevance_scores,
                         width=1, height=1, fig=axi,
                         show_plot=False, dpi=dpi)

        # Plot reference image
        if transform:
            img_hwc = arr_chw_to_hwc(transform(image_chw_rgb).detach().numpy())
            axi.imshow(img_hwc, alpha=alpha)
        else:
            axi.imshow(arr_chw_to_hwc(image_chw_rgb), alpha=alpha)

        # Hide axis
        axi.axis('off')

    # Add padding for better alignment of suptitle
    # Source: https://stackoverflow.com/a/45161551
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Facecolor sets the background color of the figure
    plt.savefig(filename, dpi=dpi, facecolor='w')
