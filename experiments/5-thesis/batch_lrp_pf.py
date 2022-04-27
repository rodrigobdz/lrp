r'''Run LRP and Pixel-Flipping experiments for image batches and save results to file.'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from pf.pixel_flipping import PixelFlipping
from pf.perturbation_modes.constants import PerturbModes
from lrp.core import LRP
from lrp.zennit.types import AvgPool, Linear
from lrp.filter import LayerFilter
from lrp.rules import LrpZBoxRule, LrpGammaRule, LrpEpsilonRule, LrpZeroRule
from data_loader.core import imagenet_data_loader
from typing import List, Dict, Union, Tuple, Callable
from matplotlib import pyplot as plt
from pathlib import Path
import lrp.rules as rules
import lrp.plot
import torchvision
import torch
import multiprocessing


# LRP hyperparameters
# GAMMA = 0.0001
# EPSILON = 1

# Experiment parameters
NUMBER_OF_BATCHES: int = 1
BATCH_SIZE: int = 25  # multiprocessing.cpu_count()
PERTURBATION_STEPS: int = 100
CLASSES: List[str] = ['axolotl']
PERTURBATION_SIZE: int = 9
# Plotting parameters
WORKSPACE_ROOT: str = '/Users/rodrigobermudezschettino/Documents/personal/unterlagen/bildung/uni/master/masterarbeit'
EXPERIMENT_DIR: str = f'{WORKSPACE_ROOT}/experiment-results/20-04-22/lrp-pf-auc/batch-size-{BATCH_SIZE}/composite-gamma-decreasing'
DPI: float = 150
# Toggle for plt.show() for each figure
SHOW_PLOT: bool = False

# Constants
SEED: int = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_ROOT: str = f'{WORKSPACE_ROOT}/code/lrp/data'
# Model parameters
VGG16_IMAGE_DIM: int = 224
CHANNELS: int = 3
HEIGHT: int = VGG16_IMAGE_DIM
WIDTH: int = VGG16_IMAGE_DIM
INPUT_SHAPE: Tuple[int, int, int, int] = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


# Enable reproducibility
torch.manual_seed(SEED)


def _save_image_batch(image_batch: torch.Tensor,
                      batch_index: int,
                      suffix: str = '') -> None:
    r'''Plot the image batch and save results to file.

    :param image_batch: Image batch
    :param batch_index: Index of the batch
    :param suffix: Prefix for the filename
    '''
    for image_index, image_chw in enumerate(image_batch):
        image_1chw: torch.Tensor = image_chw.unsqueeze(dim=0)
        lrp.plot.plot_imagenet(image_1chw, show_plot=SHOW_PLOT)

        filename: str = f'{EXPERIMENT_DIR}/batch-{batch_index}-image-{image_index}-{suffix}input-1chw.png'
        # Facecolor sets the background color of the figure
        plt.savefig(filename, dpi=DPI, facecolor='w')
        plt.close()


def _plot_lrp_results(relevance_scores_nchw: torch.Tensor,
                      batch_index: int) -> None:
    r'''Plot the results of the LRP experiment and save results to file.

    :param relevance_scores_nchw: Relevance scores of the LRP experiment
    :param batch_index: Index of the batch
    '''

    # Convert each heatmap from 3-channel to 1-channel.
    # Channel dimension is now omitted.
    r_nhw = relevance_scores_nchw.sum(dim=1)

    # Loop over relevance scores for each image in batch
    for image_index, r_hw in enumerate(r_nhw):
        lrp.plot.heatmap(relevance_scores=r_hw.detach().numpy(),
                         width=1,
                         height=1,
                         show_plot=SHOW_PLOT,
                         dpi=DPI)

        filename: str = f'{EXPERIMENT_DIR}/batch-{batch_index}-image-{image_index}-layerwise-relevance-propagation-heatmap.png'
        # Facecolor sets the background color of the figure
        plt.savefig(filename, dpi=DPI, facecolor='w')
        plt.close()


def _plot_pixel_flipping_results(pf_instance: PixelFlipping,
                                 batch_index: int) -> None:
    r'''Plot the results of the pixel flipping experiment and save results to file.

    :param pf_instance: Pixel flipping instance with experiment results
    :param batch_index: Index of the batch
    '''

    # Plot classification scores throughout perturbation steps
    title: str = f'''Region Perturbation
        Perturbation steps: {pf_instance.perturbation_steps}
        Perturbation size: {pf_instance.perturbation_size}x{pf_instance.perturbation_size}
        Perturbation mode: {pf_instance.perturb_mode}
        Batch size: {pf_instance._batch_size}
        Batch index/Total number of batches: {batch_index+1}/{NUMBER_OF_BATCHES}'''
    xlabel: str = 'Perturbation step'
    ylabel: str = 'Classification score'

    # Save to file
    pf_instance.plot_class_prediction_scores(title=title,
                                             xlabel=xlabel,
                                             ylabel=ylabel,
                                             show_plot=SHOW_PLOT)
    filename: str = f'{EXPERIMENT_DIR}/batch-{batch_index}-pixel-flipping-class-prediction-scores-{BATCH_SIZE}.png'
    # Facecolor sets the background color of the figure
    plt.savefig(filename, dpi=DPI, facecolor='w')
    plt.close()

    for image_index in range(BATCH_SIZE):
        # Plot image comparison for each image in batch to be able to save to file.
        original_input_1chw: torch.Tensor = pf_instance.original_input_nchw[image_index].unsqueeze(
            dim=0)
        flipped_input_1chw: torch.Tensor = pf_instance.flipped_input_nchw[image_index].unsqueeze(
            dim=0)
        relevance_scores_1chw: torch.Tensor = pf_instance.relevance_scores_nchw[image_index].unsqueeze(
            dim=0)
        acc_flip_mask_1hw: torch.Tensor = pf_instance.acc_flip_mask_nhw[image_index].unsqueeze(
            dim=0)
        PixelFlipping._plot_image_comparison(batch_size=1,
                                             original_input_nchw=original_input_1chw,
                                             flipped_input_nchw=flipped_input_1chw,
                                             relevance_scores_nchw=relevance_scores_1chw,
                                             acc_flip_mask_nhw=acc_flip_mask_1hw,
                                             show_plot=SHOW_PLOT)

        filename: str = f'{EXPERIMENT_DIR}/batch-{batch_index}-image-{image_index}-pixel-flipping-image-comparison.png'
        # Facecolor sets the background color of the figure, in this case to color white
        plt.savefig(filename, dpi=DPI, facecolor='w')
        plt.close()


def run_lrp_experiment(image_batch: torch.Tensor,
                       batch_index: int,
                       label_idx_n: torch.Tensor) -> Tuple[LRP, torch.Tensor, torch.Tensor]:
    r'''Run LRP experiment on a batch of images.

    :param image_batch: Batch of images
    :param batch_index: Index of the batch
    :param label_idx_n: Label indices of classes to explain

    :return: LRP instance, batch of images, relevance scores
    '''
    input_nchw: torch.Tensor = image_batch.to(DEVICE)

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    model.to(DEVICE)

    # Low and high parameters for zB-rule
    low: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.zeros(*INPUT_SHAPE))
    high: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.ones(*INPUT_SHAPE))

    # Init layer filter
    vgg16_target_types: Tuple[type, type] = (Linear, AvgPool)
    filter_by_layer_index_type: LayerFilter = LayerFilter(model)
    filter_by_layer_index_type.set_target_types(vgg16_target_types)

    # TODO: Export to configure as parameter and run script with multiple values.
    # TODO: Save values of name map to file to reconstruct parameters used.
    name_map: List[Tuple[List[str], rules.LrpRule,
                         Dict[str, Union[torch.Tensor, float]]]]
    name_map = [(filter_by_layer_index_type(lambda n: n == 0), LrpZBoxRule,
                 {'low': low, 'high': high}),
                (filter_by_layer_index_type(lambda n: 1 <= n <= 10), LrpGammaRule,
                 {'gamma': 0.5}),
                (filter_by_layer_index_type(lambda n: 11 <= n <= 17), LrpGammaRule,
                 {'gamma': 0.25}),
                (filter_by_layer_index_type(lambda n: 18 <= n <= 24), LrpGammaRule,
                 {'gamma': 0.1}),
                (filter_by_layer_index_type(lambda n: n >= 25), LrpGammaRule,
                 {'gamma': 0}), ]

    lrp_instance: LRP = LRP(model)
    lrp_instance.convert_layers(name_map)
    relevance_scores_nchw: torch.Tensor = lrp_instance.relevance(input_nchw=input_nchw,
                                                                 label_idx_n=label_idx_n)

    _plot_lrp_results(relevance_scores_nchw=relevance_scores_nchw,
                      batch_index=batch_index)

    return (lrp_instance, input_nchw, relevance_scores_nchw)


def run_pixel_flipping_experiment(lrp_instance: LRP,
                                  input_nchw: torch.Tensor,
                                  relevance_scores_nchw: torch.Tensor,
                                  batch_index: int) -> None:
    r'''Run the pixel flipping experiment.

    :param lrp_instance: LRP instance
    :param input_nchw: Input image
    :param relevance_scores_nchw: Relevance scores of the LRP experiment
    :param batch_index: Index of the batch
    '''
    pf_instance: PixelFlipping = PixelFlipping(perturbation_steps=PERTURBATION_STEPS,
                                               perturbation_size=PERTURBATION_SIZE,
                                               perturb_mode=PerturbModes.INPAINTING)
    pf_input_nchw: torch.Tensor = input_nchw.clone().detach()

    pf_relevance_scores_nchw: torch.Tensor = relevance_scores_nchw.clone().detach()

    # Function should return the (single-class) classification score for the given input to measure
    # difference between flips.
    # Access the score of predicted classes in every image in batch.
    forward_pass: Callable[[torch.Tensor], float] = lambda input_nchw: lrp_instance.model(input_nchw)[
        lrp_instance.explained_class_indices[:, 0],
        lrp_instance.explained_class_indices[:, 1]
    ]

    # Run Pixel-Flipping algorithm
    pf_instance(pf_input_nchw,
                pf_relevance_scores_nchw,
                forward_pass,
                should_loop=True)

    _save_image_batch(image_batch=pf_instance.flipped_input_nchw,
                      batch_index=batch_index,
                      suffix='flipped-')

    _plot_pixel_flipping_results(pf_instance=pf_instance,
                                 batch_index=batch_index)


if __name__ == "__main__":
    print(f'Batch size = {BATCH_SIZE}')

    dataloader: torch.utils.data.DataLoader = imagenet_data_loader(root=DATASET_ROOT,
                                                                   batch_size=BATCH_SIZE,
                                                                   classes=CLASSES,
                                                                   seed=SEED)

    # Create root directory (with intermediate directories, if these don't already exist)
    # to save artifacts from experiments.
    Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

    for batch_index, data in enumerate(dataloader):
        # Unpack data from dataloader
        image_batch, ground_truth_labels = data

        # Save data to file
        torch.save(image_batch,
                   f'{EXPERIMENT_DIR}/batch-{batch_index}-input-nchw.pt')
        torch.save(ground_truth_labels,
                   f'{EXPERIMENT_DIR}/batch-{batch_index}-ground-truth-labels.pt')

        # Save images as png to file
        _save_image_batch(image_batch=image_batch,
                          batch_index=batch_index,
                          suffix='original-')

        # Run LRP experiment
        lrp_instance, input_nchw, relevance_scores_nchw = run_lrp_experiment(image_batch=image_batch,
                                                                             batch_index=batch_index,
                                                                             label_idx_n=ground_truth_labels)

        # Save relevance scores to file
        torch.save(relevance_scores_nchw,
                   f'{EXPERIMENT_DIR}/batch-{batch_index}-relevance-scores-nchw.pt')

        # Run Pixel-Flipping/Region Perturbation experiment
        run_pixel_flipping_experiment(lrp_instance=lrp_instance,
                                      input_nchw=input_nchw,
                                      relevance_scores_nchw=relevance_scores_nchw,
                                      batch_index=batch_index)

        print(f'Finished batch {batch_index}')
        if batch_index + 1 == NUMBER_OF_BATCHES:
            print(f'Done. {batch_index + 1} batches processed.')
            break
