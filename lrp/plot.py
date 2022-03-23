r'''Plotting library
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Any, List, Tuple, Callable, Optional
import numpy
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import lrp.plot


def heatmap(relevance_scores: numpy.array, width: float, height: float,
            fig: Figure = plt, show_plot: bool = True, dpi: float = 100.0) -> None:
    r'''Plot heatmap of relevance scores

    :param relevance_scores: Relevance scores in pixel layer only
    :param width: Size of the image in x-direction
    :param height: Size of the image in y-direction

    :param fig: Figure to plot on
    :param show_plot: Show plot or not
    '''
    CMAP: ListedColormap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    CMAP[:, 0:3] *= 0.85
    CMAP = ListedColormap(CMAP)

    abs_bound = 10 * ((numpy.abs(relevance_scores) ** 3.0).mean() ** (1.0 / 3))

    if fig is plt:
        fig.figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.axis('off')
    fig.imshow(relevance_scores, cmap=CMAP, vmin=-abs_bound,
               vmax=abs_bound, interpolation='nearest')

    if show_plot:
        plt.show()


def grid(results: List[Tuple[float, torch.Tensor]],
         image: torch.Tensor,
         title: str,
         filename: str,
         gridsize: Tuple[int, int],
         param_name: str,
         transform: Optional[Callable] = None,
         param_print: Callable[[Any], Any] = lambda p: p,
         figsize: List[int] = [10, 10],
         alpha: float = 0.2) -> None:
    r'''Plot the results of the hyperparameter grid search.

    Source: https://stackoverflow.com/a/46616645

    :param results: List of tuples of the form (rule_param, R)
    :param image: Reference image used to calculate the relevance scores
    :param title: Title of the plot
    :param filename: Name of the file to save the plot to
    :param gridsize: Number of rows and columnds in the grid
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
            # Convert from NCHW to NHWC format.
            img_nhwc = transform(image).numpy().transpose(1, 2, 0)
            axi.imshow(img_nhwc, alpha=alpha)
        else:
            axi.imshow(image, alpha=alpha)

        # Hide axis
        axi.axis('off')

    # Add padding for better alignment of suptitle
    # Source: https://stackoverflow.com/a/45161551
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Facecolor sets the background color of the figure
    plt.savefig(filename, dpi=dpi, facecolor='w')
