r"""Run LRP and Pixel-Flipping experiments for image batches and save results to file."""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import argparse
import ast
import logging
import shutil
import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy
import torch
import torchvision
from matplotlib import pyplot as plt

import lrp.plot
import pf.plot
from data_loader.core import imagenet_data_loader
from lrp import rules
from lrp.core import LRP
from lrp.filter import LayerFilter
from lrp.rules import LrpEpsilonRule, LrpGammaRule, LrpZBoxRule, LrpZeroRule
from lrp.zennit.types import AvgPool, Linear
from pf.core import PixelFlipping
from pf.decorators import timer
from pf.perturbation_modes.constants import PerturbModes

# These functions at the top are the functions that need to be
# modified to support different composites:
#   _get_rule_layer_map_by_experiment_id
#   aggregate_results_for_plot


def _get_rule_layer_map_by_experiment_id(model: torch.nn.Module) -> List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]]:
    r"""Get rule layer map by experiment id.

    :param model: Model to get rule layer map for.
    :param experiment_id: Experiment id

    :return: Rule layer map
    """
    LOGGER.info('Hyperparameters for model %s:', str(MODEL))

    # Init layer filter
    target_types: Tuple[type] = (Linear, AvgPool)
    filter_by_layer_index_type: LayerFilter = LayerFilter(model=model,
                                                          target_types=target_types)

    if EXP_NAME_SHORT == ExperimentShortNames.DECREASING_GAMMA:
        return _get_rule_layer_map_of_decreasing_gamma(layer_filter=filter_by_layer_index_type)

    if EXP_NAME_SHORT == ExperimentShortNames.LRP_TUTORIAL:
        return _get_rule_layer_map_of_lrp_tutorial(layer_filter=filter_by_layer_index_type)

    raise ValueError(f'''Unknown experiment name: {EXP_NAME_SHORT}.
Check available experiment names in definition of class ExperimentShortNames.''')


def aggregate_results_for_plot():
    r"""Aggregate results for plotting and save to file."""
    experiment_parent_path: Path = Path(EXPERIMENT_PARENT_ROOT)

    # Files containing x and y values
    rule_layer_map_list: List[str] = list(
        experiment_parent_path.glob('**/batch-*-lrp-rule-layer-map.npy')
    )
    # Files containing z values
    auc_list: List[str] = list(
        experiment_parent_path.glob('**/batch-*-area-under-the-curve.npy')
    )

    # Sanity check that the number of AUCs and rule-layer maps match
    if len(auc_list) != len(rule_layer_map_list):
        raise ValueError(f'Number of AUC files ({len(auc_list)}) does not match '
                         f'number of rule layer maps ({len(rule_layer_map_list)})')

    x_values: List[float] = []
    y_values: List[float] = []

    for rule_layer_map_path in rule_layer_map_list:
        rule_layer_map: List[
            Tuple[
                List[str], rules.LrpRule,
                Dict[str, Union[torch.Tensor, float]]
            ]
        ]
        rule_layer_map = numpy.load(file=rule_layer_map_path,
                                    allow_pickle=True)

        x_val: float
        y_val: float

        if EXP_NAME_SHORT == ExperimentShortNames.DECREASING_GAMMA:
            # Order of rule in rule_layer_map:
            # lrp.rules.LrpZBoxRule
            # lrp.rules.LrpGammaRule
            # lrp.rules.LrpGammaRule
            # lrp.rules.LrpGammaRule
            x_val = rule_layer_map[1][2].get('gamma')
            y_val = rule_layer_map[2][2].get('gamma')

        elif EXP_NAME_SHORT == ExperimentShortNames.LRP_TUTORIAL:
            # Order of rule in rule_layer_map:
            # lrp.rules.LrpZBoxRule
            # lrp.rules.LrpGammaRule
            # lrp.rules.LrpEpsiloinRule
            # lrp.rules.LrpZeroRule
            x_val = rule_layer_map[1][2].get('gamma')
            y_val = rule_layer_map[2][2].get('epsilon')
        else:
            raise ValueError(f'''Unknown experiment name: {EXP_NAME_SHORT}.
Check available experiment names in definition of class ExperimentShortNames.''')

        # Save x and y values for plotting
        x_values.append(x_val)
        y_values.append(y_val)

    z_values: List[float] = []

    for auc_file in auc_list:
        auc_score: float = numpy.load(file=auc_file,
                                      allow_pickle=True).item()
        z_values.append(auc_score)

    LOGGER.debug('Saving aggregated results (x, y and z values) to file')
    numpy.save(file=PLOT_X_VALUES_PATH,
               arr=x_values)
    numpy.save(file=PLOT_Y_VALUES_PATH,
               arr=y_values)
    numpy.save(file=PLOT_Z_VALUES_PATH,
               arr=z_values)


def _get_rule_layer_map_of_lrp_tutorial(layer_filter: LayerFilter) -> List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]]:
    r"""Get rule layer map of experiment with id 'lrp-tutorial'.

    :param layer_filter: Layer filter

    :return: Rule layer map
    """
    # Decreasing gamma permutations

    # Low and high parameters for zB-rule
    low: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.zeros(*INPUT_SHAPE)
    )
    high: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.ones(*INPUT_SHAPE)
    )

    # Hyperparameter values for each experiment
    # Manually add zero because log(0) = -inf
    gammas: numpy.ndarray = numpy.logspace(start=SAMPLING_RANGE_START,
                                           stop=SAMPLING_RANGE_END,
                                           num=NUMBER_OF_HYPERPARAMETER_VALUES - 1)
    gammas = numpy.concatenate((numpy.array([0.0]), gammas))

    epsilons: numpy.ndarray = numpy.logspace(start=SAMPLING_RANGE_START,
                                             stop=SAMPLING_RANGE_END,
                                             num=NUMBER_OF_HYPERPARAMETER_VALUES - 1)
    epsilons = numpy.concatenate((numpy.array([0.0]), epsilons))

    # Compute all permutations between gammas and epsilons
    hyperparam_permutations: List[Tuple[float, float]] = [
        (gam, eps) for gam in gammas for eps in epsilons
    ]
    if TOTAL_NUMBER_OF_EXPERIMENTS != len(hyperparam_permutations):
        raise ValueError(f'Total number of experiments is {TOTAL_NUMBER_OF_EXPERIMENTS} but '
                         f'{len(hyperparam_permutations)} hyperparameter permutations were found.')

    gamma, epsilon = hyperparam_permutations[EXPERIMENT_ID]

    LOGGER.info("\tgamma (Layers: 1-16): %s",
                str(gamma))
    LOGGER.info("\tepsilon (Layers: 17-30): %s",
                str(epsilon))

    rule_layer_map: List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]
    ]

    rule_layer_map = [
        (layer_filter(lambda n: n == 0),
         LrpZBoxRule, {'low': low, 'high': high}),
        (layer_filter(lambda n: 1 <= n <= 16), LrpGammaRule, {'gamma': gamma}),
        (layer_filter(lambda n: 17 <= n <= 30),
         LrpEpsilonRule, {'epsilon': epsilon}),
        (layer_filter(lambda n: 31 <= n), LrpZeroRule, {}),
    ]

    return rule_layer_map


def _get_rule_layer_map_of_decreasing_gamma(layer_filter: LayerFilter) -> List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]]:
    r"""Get rule layer map of experiment with id 'decr-gamma'.

    :param layer_filter: Layer filter

    :return: Rule layer map
    """
    # Decreasing gamma permutations

    # Low and high parameters for zB-rule
    low: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.zeros(*INPUT_SHAPE)
    )
    high: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.ones(*INPUT_SHAPE)
    )

    # Hyperparameter values for each experiment
    # Manually add zero because log(0) = -inf
    gammas: numpy.ndarray = numpy.logspace(start=SAMPLING_RANGE_START,
                                           stop=SAMPLING_RANGE_END,
                                           num=NUMBER_OF_HYPERPARAMETER_VALUES - 1)
    gammas = numpy.concatenate((numpy.array([0.0]), gammas))

    # Compute all permutations between gammas and epsilons
    hyperparam_permutations: List[Tuple[float, float]] = [
        (gamma_one, gamma_two) for gamma_one in gammas for gamma_two in gammas
    ]
    if TOTAL_NUMBER_OF_EXPERIMENTS != len(hyperparam_permutations):
        raise ValueError(f'Total number of experiments is {TOTAL_NUMBER_OF_EXPERIMENTS} but '
                         f'{len(hyperparam_permutations)} hyperparameter permutations were found.')

    gamma_one, gamma_two = hyperparam_permutations[EXPERIMENT_ID]

    LOGGER.info("\tgamma (Layers: 11-17): %s",
                str(gamma_one))
    LOGGER.info("\tgamma (Layers: 18-24): %s",
                str(gamma_two))

    rule_layer_map: List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]
    ]

    rule_layer_map = [
        (layer_filter(lambda n: n == 0), LrpZBoxRule,
         {'low': low, 'high': high}),
        (layer_filter(lambda n: 1 <= n <= 17), LrpGammaRule,
         {'gamma': gamma_one}),
        (layer_filter(lambda n: 18 <= n <= 24), LrpGammaRule,
         {'gamma': gamma_two}),
        (layer_filter(lambda n: 25 <= n <= 31), LrpGammaRule,
         {'gamma': 0})
    ]

    return rule_layer_map


class ExperimentShortNames:  # pylint: disable=too-few-public-methods
    r"""Constants for experiment names (identifiers)."""

    DECREASING_GAMMA: str = 'decr-gamma'
    LRP_TUTORIAL: str = 'lrp-tutorial'


class Helpers:
    r"""Encapsulate all helper functions."""

    @staticmethod
    def create_directories_if_not_exists(*directories) -> None:
        r"""Create directories (if these don't already exist).

        :param directories: Directories to create
        """
        for directory in directories:
            Path(directory).mkdir(parents=True,
                                  exist_ok=True)

    @staticmethod
    def save_torch_object(torch_object: torch.Tensor,
                          filename: str) -> None:
        r"""Save the torch objects to file.

        :param torch_object: Torch object to save
        :param filename: Filename to save the object to
        """
        torch.save(torch_object.cpu(),
                   f'{TORCH_OBJECTS_DIR}/{filename}')

    @staticmethod
    def save_image_batch_plot(image_batch: torch.Tensor,
                              batch_index: int,
                              suffix: str = '') -> None:
        r"""Plot the image batch and save results to file.

        :param image_batch: Image batch
        :param batch_index: Index of the batch
        :param suffix: Prefix for the filename
        """
        # Save images into nested directory with suffix name for better file organization
        root_dir: str = f'{INDIVIDUAL_RESULTS_DIR}/{suffix}'
        Helpers.create_directories_if_not_exists(root_dir)

        for image_index, image_chw in enumerate(image_batch):
            image_1chw: torch.Tensor = image_chw.unsqueeze(dim=0)
            lrp.plot.plot_imagenet(image_1chw, show_plot=SHOW_PLOT)

            filename: str = f'{root_dir}/batch-{batch_index}-image-{image_index}-' \
                f'{suffix}-input-1chw.png'
            # Facecolor sets the background color of the figure
            plt.savefig(fname=filename, dpi=DPI, facecolor='w')
            plt.close()

    @staticmethod
    def save_plot_lrp_results(relevance_scores_nchw: torch.Tensor,
                              batch_index: int) -> None:
        r"""Plot the results of the LRP experiment and save results to file.

        :param relevance_scores_nchw: Relevance scores of the LRP experiment
        :param batch_index: Index of the batch
        """
        # Convert each heatmap from 3-channel to 1-channel.
        # Channel dimension is now omitted.
        r_nhw: torch.Tensor = relevance_scores_nchw.sum(dim=1)

        # Save images into nested directory with suffix name for better file organization
        root_dir: str = f'{INDIVIDUAL_RESULTS_DIR}/lrp-heatmap'
        Helpers.create_directories_if_not_exists(root_dir)

        # Loop over relevance scores for each image in batch
        for image_index, r_hw in enumerate(r_nhw):
            # Use Tensor.cpu() to copy the tensor to host memory before converting to numpy().
            lrp.plot.heatmap(relevance_scores=r_hw.cpu().detach().numpy(),
                             width=1,
                             height=1,
                             show_plot=SHOW_PLOT,
                             dpi=DPI)

            filename: str = f'{root_dir}/batch-{batch_index}-image-{image_index}-' \
                'layerwise-relevance-propagation-heatmap.png'
            # Facecolor sets the background color of the figure
            plt.savefig(fname=filename, dpi=DPI, facecolor='w')
            plt.close()

    @staticmethod
    def save_plot_pf_results(pf_instance: PixelFlipping,
                             batch_index: int) -> None:
        r"""Plot the results of the pixel flipping experiment and save results to file.

        :param pf_instance: Pixel flipping instance with experiment results
        :param batch_index: Index of the batch
        """
        pf_instance.plot_class_prediction_scores(show_plot=SHOW_PLOT)

        class_scores_filename: str = f'{EXPERIMENT_ROOT}/batch-{batch_index}' \
            '-pixel-flipping-class-prediction-scores' \
            f'-experiment-id-{EXPERIMENT_ID}.png'
        # Facecolor sets the background color of the figure
        plt.savefig(fname=class_scores_filename, dpi=DPI, facecolor='w')
        plt.close()

        for image_index in range(BATCH_SIZE):
            # Plot either the currently flipped input or the penultimate flipped input
            # to avoid plotting a gray image, which corresponds to the last flipping step.
            plot_flipped_input_before_last_step: bool = False
            flipped_input_nchw: torch.Tensor = pf_instance.flipped_input_nchw

            if torch.is_tensor(pf_instance.flipped_input_nchw_before_last_step):
                flipped_input_nchw = pf_instance.flipped_input_nchw_before_last_step
                plot_flipped_input_before_last_step = True
                # Plot image comparison for each image in batch to be able to save to file.
            original_input_1chw: torch.Tensor = pf_instance.original_input_nchw[
                image_index].unsqueeze(dim=0)
            flipped_input_1chw: torch.Tensor = flipped_input_nchw[
                image_index].unsqueeze(dim=0)
            relevance_scores_1chw: torch.Tensor = pf_instance.relevance_scores_nchw[
                image_index].unsqueeze(dim=0)
            acc_flip_mask_1hw: torch.Tensor = pf_instance.acc_flip_mask_nhw[image_index].unsqueeze(
                dim=0)

            pf.plot.plot_image_comparison(batch_size=1,
                                          original_input_nchw=original_input_1chw.cpu(),
                                          flipped_input_nchw=flipped_input_1chw.cpu(),
                                          before_last_step=plot_flipped_input_before_last_step,
                                          relevance_scores_nchw=relevance_scores_1chw.cpu(),
                                          acc_flip_mask_nhw=acc_flip_mask_1hw.cpu(),
                                          perturbation_size=pf_instance.perturbation_size,
                                          show_plot=SHOW_PLOT)

            pf_comparison_filename: str = f'{INDIVIDUAL_RESULTS_DIR}/batch-{batch_index}-' \
                f'image-{image_index}-pixel-flipping-image-comparison.png'
            # Facecolor sets the background color of the figure, in this case to color white
            plt.savefig(fname=pf_comparison_filename, dpi=DPI, facecolor='w')
            plt.close()

    @staticmethod
    def save_settings():
        r"""Save settings to file for reproducibility."""
        LOGGER.debug(
            "Save experiment parameters to file for reproducibility and proper archival.")

        # Get filename of this file with extension
        filename_with_ext: str = Path(__file__).name

        # Get filename of this file (without absolute path and without extension)
        filename_no_ext: str = Path(__file__).stem

        # Path to save local variables used for this experiment
        absolute_path_no_ext: str = f'{NUMPY_OBJECTS_DIR}/' \
            f'{filename_no_ext}-locals-filtered-by-type'

        # Create a dictionary from locals() with entries filtered by type to avoid common pitfalls
        # of trying to save modules or classes which are not accepted by numpy.save.
        local_vars_dict: Dict[str, Any]
        local_vars_dict = {dict_key: dict_val for dict_key, dict_val in locals().items()
                           if isinstance(dict_val,
                                         (str, int, list, tuple, dict))}

        LOGGER.debug('Save local variables to file for archival purposes.')

        LOGGER.debug('Save local variables as dictionary')
        numpy.save(file=absolute_path_no_ext + '.npy',
                   arr=local_vars_dict)

        LOGGER.debug('Save local variables as text (human-readable)')
        with open(file=absolute_path_no_ext + '.txt',
                  mode='w',
                  encoding='utf8') as file:
            file.write(str(local_vars_dict))

        # Copying config file to experiment directory for reproducibility of results.
        LOGGER.debug('Copying config file %s to experiment directory:\n%s',
                     str(config_file_path.name),
                     str(EXPERIMENT_ROOT))
        # Source: https://stackoverflow.com/a/33626207
        shutil.copy(config_file_path, EXPERIMENT_ROOT)

        LOGGER.debug('Copying %s to experiment directory:\n%s',
                     str(filename_with_ext),
                     str(EXPERIMENT_ROOT))
        shutil.copy(__file__, Path(EXPERIMENT_ROOT) / filename_with_ext)

        # Script is used to generate plots from experiment results.
        visualize_py_script: Path = Path(
            __file__).parent / Path('visualize.py')

        if not Path.exists(visualize_py_script):
            raise ValueError(
                f'Could not find visualize.py script under {visualize_py_script}.')

        LOGGER.debug('Copying %s to experiment directory:\n%s',
                     str(visualize_py_script.name),
                     str(EXPERIMENT_ROOT))
        shutil.copy(visualize_py_script, Path(
            EXPERIMENT_ROOT) / visualize_py_script.name)

        LOGGER.info('Done saving experiment parameters to file.')

    @staticmethod
    def save_artifacts(lrp_instance: LRP,
                       pf_instance: PixelFlipping,
                       batch_index: int) -> None:
        r"""Save artifacts of the LRP and pixel flipping experiments to file.

        :param pf_instance: Pixel flipping instance with experiment results
        :param batch_index: Index of the batch
        """
        original_input_nchw: torch.Tensor = lrp_instance.input_nchw

        # Save original input to file
        Helpers.save_torch_object(torch_object=original_input_nchw,
                                  filename=f'batch-{batch_index}-input-nchw.pt')

        # Save relevance scores to file
        Helpers.save_torch_object(torch_object=lrp_instance.relevance_scores_nchw,
                                  filename=f'batch-{batch_index}-relevance-scores-nchw.pt')

        # Save  ground truth labels to file
        Helpers.save_torch_object(torch_object=lrp_instance.label_idx_n,
                                  filename=f'batch-{batch_index}-ground-truth-labels.pt')

        # Save images as png to file
        Helpers.save_image_batch_plot(image_batch=original_input_nchw,
                                      batch_index=batch_index,
                                      suffix='original')

        Helpers.save_image_batch_plot(image_batch=pf_instance.flipped_input_nchw,
                                      batch_index=batch_index,
                                      suffix='pf-perturbed')

        # Save AUC score to file
        auc_score_arr: numpy.ndarray = numpy.array(pf_instance.calculate_auc_score(),
                                                   dtype=object)
        numpy.save(file=f'{NUMPY_OBJECTS_DIR}/batch-{batch_index}-area-under-the-curve.npy',
                   arr=auc_score_arr)

        # Save LRP rule-layer map to file
        numpy.save(file=f'{NUMPY_OBJECTS_DIR}/batch-{batch_index}-lrp-rule-layer-map.npy',
                   arr=numpy.array(lrp_instance.rule_layer_map, dtype=object))


@timer
def run_lrp_experiment(image_batch: torch.Tensor,
                       batch_index: int,
                       label_idx_n: torch.Tensor) -> LRP:
    r"""Run LRP experiment on a batch of images.

    :param image_batch: Batch of images
    :param batch_index: Index of the batch
    :param label_idx_n: Label indices of classes to explain

    :return: LRP instance, batch of images, relevance scores
    """
    input_nchw: torch.Tensor = image_batch.to(device=DEVICE)
    label_idx_n.to(device=DEVICE)

    model: torch.nn.Module = torchvision.models.vgg16(pretrained=True)
    model.eval().to(device=DEVICE)

    rule_layer_map: List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]
    ] = _get_rule_layer_map_by_experiment_id(model=model)

    lrp_instance: LRP = LRP(model)
    lrp_instance.convert_layers(rule_layer_map)
    relevance_scores_nchw: torch.Tensor = lrp_instance.relevance(input_nchw=input_nchw,
                                                                 label_idx_n=label_idx_n).to(
                                                                     device=DEVICE)

    Helpers.save_plot_lrp_results(relevance_scores_nchw=relevance_scores_nchw,
                                  batch_index=batch_index)

    return lrp_instance


@timer
def run_pixel_flipping_experiment(lrp_instance: LRP,
                                  batch_index: int) -> PixelFlipping:
    r"""Run the pixel flipping experiment.

    :param lrp_instance: LRP instance
    :param batch_index: Index of the batch

    :return: Pixel flipping instance
    """
    pf_instance: PixelFlipping = PixelFlipping(perturbation_steps=PERTURBATION_STEPS,
                                               perturbation_size=PERTURBATION_SIZE,
                                               perturb_mode=PerturbModes.INPAINTING,
                                               sort_objective=SORT_OBJECTIVE)
    pf_input_nchw: torch.Tensor = lrp_instance.input_nchw.clone().detach()
    pf_relevance_scores_nchw: torch.Tensor = lrp_instance.relevance_scores_nchw.clone().detach()

    # Transfer tensors to selected device
    pf_input_nchw.to(device=DEVICE)
    pf_relevance_scores_nchw.to(device=DEVICE)

    # Function should return the (single-class) classification score for
    # the given input to measure difference between flips.
    # Access the score of predicted classes in every image in batch.
    forward_pass: Callable[
        [torch.Tensor],
        torch.Tensor
    ] = lambda input_nchw: lrp_instance.model(input_nchw).to(device=DEVICE)[
        lrp_instance.explained_class_indices[:, 0],
        lrp_instance.explained_class_indices[:, 1]
    ]

    # Run Pixel-Flipping algorithm
    pf_instance(pf_input_nchw,
                pf_relevance_scores_nchw,
                forward_pass,
                should_loop=True)

    Helpers.save_plot_pf_results(pf_instance=pf_instance,
                                 batch_index=batch_index)

    return pf_instance


class CommandLine():  # pylint: disable=too-few-public-methods
    r"""Encapsulate CLI-related functions."""

    def __init__(self) -> None:
        r"""Initialize argument parser."""
        self.parser = argparse.ArgumentParser(description='Specify the experiment parameters.',
                                              epilog='For more information, review the function'
                                              "called '_get_rule_layer_map_by_experiment_id'.")

        self.parser.add_argument('-i', '--experiment-id',
                                 type=int,
                                 help='ID of experiment (LRP rule-layer map) to use',
                                 required=True)

        self.parser.add_argument('-c', '--config-file',
                                 type=Path,
                                 help='Absolute path to configuration file'
                                 ' with parameters for experiments',
                                 required=True)

    def parse_arguments(self) -> argparse.Namespace:
        r"""Parse CLI arguments.

        :return: Parsed arguments
        """
        return self.parser.parse_args()


@timer
def run_experiments() -> None:
    r"""Run Layer-wise Relevance Propagation and Pixel-Flipping experiments."""
    Helpers.create_directories_if_not_exists(EXPERIMENT_ROOT,
                                             INDIVIDUAL_RESULTS_DIR,
                                             TORCH_OBJECTS_DIR,
                                             NUMPY_OBJECTS_DIR,
                                             PLOT_ROOT)

    Helpers.save_settings()

    LOGGER.info("Progress: %s/%s",
                str(EXPERIMENT_ID + 1),
                str(TOTAL_NUMBER_OF_EXPERIMENTS))

    LOGGER.info('Experiment: %s', str(EXP_NAME_SHORT))
    LOGGER.info("Experiment ID: %s",
                str(EXPERIMENT_ID))
    LOGGER.info('Batch size = %s', str(BATCH_SIZE))

    LOGGER.debug('Setting manual seed in PyTorch to %s', str(SEED))
    torch.manual_seed(SEED)

    my_dataloader: torch.utils.data.DataLoader = imagenet_data_loader(root=DATASET_ROOT,
                                                                      batch_size=BATCH_SIZE,
                                                                      classes=IMAGE_CLASSES,
                                                                      seed=SEED)

    for my_batch_index, (my_image_batch, my_ground_truth_labels) in enumerate(my_dataloader):
        my_image_batch.to(device=DEVICE, non_blocking=True)
        my_ground_truth_labels.to(device=DEVICE, non_blocking=True)

        LOGGER.info('Running LRP experiment')
        my_lrp_instance = run_lrp_experiment(image_batch=my_image_batch,
                                             batch_index=my_batch_index,
                                             label_idx_n=my_ground_truth_labels)

        LOGGER.info('Running Pixel-Flipping/Region Perturbation experiment')
        my_pf_instance = run_pixel_flipping_experiment(lrp_instance=my_lrp_instance,
                                                       batch_index=my_batch_index)

        Helpers.save_artifacts(lrp_instance=my_lrp_instance,
                               pf_instance=my_pf_instance,
                               batch_index=my_batch_index)

        LOGGER.info('Finished batch %s', str(my_batch_index))

        if my_batch_index + 1 == NUMBER_OF_BATCHES:
            LOGGER.info('Done. %s batches processed.',
                        str(my_batch_index + 1))
            break


if __name__ == "__main__":
    args_parser: CommandLine = CommandLine()
    parsed_args: argparse.Namespace = args_parser.parse_arguments()

    EXPERIMENT_ID: int = parsed_args.experiment_id

    config: ConfigParser = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_path: Path = parsed_args.config_file

    # Ensure that the configuration file exists.
    if not config_file_path.absolute().exists():
        raise ValueError(
            f'Configuration file {config_file_path.absolute()} does not exist.')

    # pylint: disable=pointless-statement
    config.read(config_file_path)
    # pylint: enable=pointless-statement

    verbose: bool = config.getboolean('DEFAULT', 'VERBOSE', fallback=True)

    # Init logger instance
    logging.basicConfig(
        stream=sys.stderr,
        format='%(levelname)-8s  %(message)s'
    )

    # Assign logger instance a name
    LOGGER: logging.Logger = logging.getLogger(__name__)

    LOGGER.setLevel(logging.INFO)
    if verbose:
        LOGGER.setLevel(logging.DEBUG)

    experiments_section_name: str = 'EXPERIMENTS'
    lrp_section_name: str = 'LRP'
    pf_section_name: str = 'PIXEL_FLIPPING'
    data_section_name: str = 'DATA'
    paths_section_name: str = 'PATHS'

    BATCH_SIZE: int = config.getint(data_section_name,
                                    'BATCH_SIZE')
    PERTURBATION_STEPS: int = config.getint(pf_section_name,
                                            'PERTURBATION_STEPS')
    PERTURBATION_SIZE: int = config.getint(pf_section_name,
                                           'PERTURBATION_SIZE')
    SORT_OBJECTIVE: str = config[pf_section_name]['SORT_OBJECTIVE']

    SAMPLING_RANGE_START: float = config.getfloat(lrp_section_name,
                                                  'SAMPLING_RANGE_START')
    SAMPLING_RANGE_END: float = config.getfloat(lrp_section_name,
                                                'SAMPLING_RANGE_END')

    # Experiment parameters
    MODEL: str = config[experiments_section_name]['MODEL']
    EXP_NAME_SHORT: str = config[experiments_section_name]['EXP_NAME_SHORT']

    if MODEL != 'vgg16':
        raise ValueError(
            f'Model {MODEL} is not supported. Only vgg16 is supported.')

    # Workspace constants
    DATASET_ROOT: str = config[paths_section_name]['DATASET_ROOT']

    # Directories to be created (if they don't already exist)
    EXPERIMENT_PARENT_ROOT: str = config[paths_section_name]['EXPERIMENT_PARENT_ROOT']
    PLOT_ROOT: str = config[paths_section_name]['PLOT_ROOT']
    EXPERIMENT_ROOT: str = f'{EXPERIMENT_PARENT_ROOT}/experiment-id-{EXPERIMENT_ID}'

    # Derivated workspace constants
    INDIVIDUAL_RESULTS_DIR: str = f'{EXPERIMENT_ROOT}/individual-results'
    TORCH_OBJECTS_DIR: str = f'{EXPERIMENT_ROOT}/pytorch-objects'
    NUMPY_OBJECTS_DIR: str = f'{EXPERIMENT_ROOT}/numpy-objects'

    # Experiment parameters
    NUMBER_OF_BATCHES: int = config.getint(data_section_name,
                                           'NUMBER_OF_BATCHES')

    # Value from config file is loaded as string.
    # Convert string representation of list to list
    # Source: https://stackoverflow.com/a/1894296
    #
    # I am aware of the potential arbitrary code execution vulnerability this implies due to eval().
    # Nevertheless, the code is intended for research purposes. Proceeding after acknowledging
    # and assessing the risks.
    IMAGE_CLASSES: List[str] = ast.literal_eval(config[data_section_name]
                                                ['IMAGE_CLASSES'])

    NUMBER_OF_HYPERPARAMETER_VALUES: int = config.getint(lrp_section_name,
                                                         'NUMBER_OF_HYPERPARAMETER_VALUES')

    # Total number of experiments should be this number squared.
    # TOTAL_NUMBER_OF_EXPERIMENTS: int = NUMBER_OF_HYPERPARAMETER_VALUES ** 2
    TOTAL_NUMBER_OF_EXPERIMENTS: int = config.getint(lrp_section_name,
                                                     'TOTAL_NUMBER_OF_EXPERIMENTS')

    if EXPERIMENT_ID < 0 or EXPERIMENT_ID >= TOTAL_NUMBER_OF_EXPERIMENTS:
        raise ValueError(
            f'Experiment ID {EXPERIMENT_ID} is out of range [0-{TOTAL_NUMBER_OF_EXPERIMENTS - 1}].')

    # PyTorch constants
    SEED: int = 0
    DEVICE: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Model parameters
    IMAGE_DIMENSION: int = 224
    CHANNELS: int = 3
    HEIGHT: int = IMAGE_DIMENSION
    WIDTH: int = IMAGE_DIMENSION
    INPUT_SHAPE: Tuple[int, int, int, int] = (BATCH_SIZE,
                                              CHANNELS,
                                              HEIGHT,
                                              WIDTH)

    # Plotting parameters
    DPI: float = 150

    # Toggle for plt.show() for each figure
    SHOW_PLOT: bool = False

    # Paths from where to load the values to plot.
    PLOT_X_VALUES_PATH: str = config[paths_section_name]['PLOT_X_VALUES_PATH']
    PLOT_Y_VALUES_PATH: str = config[paths_section_name]['PLOT_Y_VALUES_PATH']
    PLOT_Z_VALUES_PATH: str = config[paths_section_name]['PLOT_Z_VALUES_PATH']

    run_experiments()

    # Check if this experiment is the last one.
    if EXPERIMENT_ID == TOTAL_NUMBER_OF_EXPERIMENTS - 1:
        # Aggregate results for plot and save to file.
        aggregate_results_for_plot()

    LOGGER.info("Done running experiment %s",
                str(EXPERIMENT_ID))
