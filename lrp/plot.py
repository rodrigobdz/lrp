r"""Plotting library
"""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from typing import Sequence

import numpy
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

import lrp.norm
import pf.utils
from pf.convert_img import arr_chw_to_hwc


def _show(imgs: Sequence) -> None:
    r"""Show a batch of images.

    Function imported directly from:
        https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py

    :param imgs: Batch of images
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    plt.rcParams["savefig.bbox"] = 'tight'

    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(numpy.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def grid_plot_imagenet(img_nchw_rgb: torch.Tensor) -> None:
    r"""Plot a grid of images in NCHW format and RGB color format with ImageNet mean and
    standard deviance.

    :param img_nchw_rgb: Images to plot.
    """
    # Create grid with batch of images.
    grid: torch.Tensor = torchvision.utils.make_grid(
        lrp.norm.ImageNetNorm.inverse_normalize(img_nchw_rgb)
    )
    # Plot image grid
    _show(grid)


def plot_imagenet(img_nchw_rgb: torch.Tensor, ax: Figure = plt, show_plot: bool = True) -> None:
    r"""Plot an image in NCHW format and RGB color format with ImageNet mean and standard deviance.

    Image is converted to [0,1] range first, then plotted.

    :param img_nchw_rgb: Images to plot.
    :param ax: Axis to plot on
    :param show_plot: Show plot or not
    """
    return plot_tensor_img_nchw_rgb(
        img_nchw_rgb=lrp.norm.ImageNetNorm.inverse_normalize(img_nchw_rgb),
        ax=ax,
        show_plot=show_plot
    )


def plot_tensor_img_nchw_rgb(img_nchw_rgb: torch.Tensor,
                             ax: Figure = plt,
                             show_plot: bool = True) -> None:
    r"""Plot an image as a tensor in NCHW format with RGB color format using matplotlib.

    "valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)."

    :param img_nchw_rgb: Images to plot
    :param ax: Axis to plot on (default: plt)
    :param show_plot: Show plot or not
    """
    pf.sanity_checks.ensure_nchw_format(img_nchw_rgb)

    for img_chw_rgb in img_nchw_rgb:
        # Convert from NCHW to HWC format and from tensor to numpy array.
        # Use Tensor.cpu() to copy the tensor to host memory before converting to numpy().
        img_hwc_rgb: numpy.ndarray = arr_chw_to_hwc(
            img_chw_rgb.cpu().detach().numpy())

        # Plot image
        ax.imshow(img_hwc_rgb)
        ax.axis('off')
        if show_plot:
            plt.show()


def heatmap(relevance_scores: numpy.ndarray,
            width: float = 2,
            height: float = 2,
            fig: Figure = plt,
            show_plot: bool = True,
            dpi: float = 100.0,
            show_axis: bool = False) -> None:
    r"""Plot heatmap of relevance scores.

    :param relevance_scores: Relevance scores in pixel layer only
    :param width: Size of the image in x-direction
    :param height: Size of the image in y-direction

    :param fig: Figure to plot on
    :param show_plot: Show plot or not
    :param show_axis: Show axis or not
    """
    CMAP: ListedColormap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    CMAP[:, 0:3] *= 0.85
    CMAP = ListedColormap(CMAP)

    # TODO: Document this heuristic to control the color intensity in the heatmap.
    abs_bound = 10 * ((numpy.abs(relevance_scores) ** 3.0).mean() ** (1.0 / 3))

    # zennit's calculation
    # # get the absolute maximum, to center the heat map around 0
    # # sum over the color channels
    # heatmap = relevance.sum(1)
    # amax = heatmap.abs().numpy().max((1, 2))

    if fig is plt:
        fig.figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if not show_axis:
        fig.axis('off')

    fig.imshow(relevance_scores, cmap=CMAP, vmin=-abs_bound,
               vmax=abs_bound, interpolation='nearest')

    if show_plot:
        plt.show()
