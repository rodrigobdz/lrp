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
import shutil
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
from pf.perturbation_modes.constants import PerturbModes

DEVICE: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def _get_rule_layer_map_by_experiment_id(filter_by_layer_index_type: LayerFilter) -> List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]]:
    r"""Get rule layer map by experiment id.

    :param filter_by_layer_index_type: Layer filter
    :param experiment_id: Experiment id

    :return: Rule layer map
    """
    # Low and high parameters for zB-rule
    low: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.zeros(*INPUT_SHAPE, device=DEVICE)
    )
    high: torch.Tensor = lrp.norm.ImageNetNorm.normalize(
        torch.ones(*INPUT_SHAPE, device=DEVICE)
    )

    # Hyperparameter values for each experiment
    # Manually add zero because log(0) = -inf
    gammas: numpy.ndarray = numpy.logspace(start=0.00001,
                                           stop=0.25,
                                           num=NUMBER_OF_HYPERPARAMETER_VALUES - 1)
    gammas = numpy.concatenate((numpy.array([0.0]), gammas))

    epsilons: numpy.ndarray = numpy.logspace(start=0.00001,
                                             stop=0.5,
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
    print(f'Experiment ID: {EXPERIMENT_ID}. '
          f'Progress: {EXPERIMENT_ID + 1}/{TOTAL_NUMBER_OF_EXPERIMENTS}'
          f', gamma: {gamma}'
          f', epsilon: {epsilon}')

    rule_layer_map: List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]
    ]

    rule_layer_map = [
        (filter_by_layer_index_type(lambda n: n == 0), LrpZBoxRule,
         {'low': low, 'high': high}),
        (filter_by_layer_index_type(lambda n: 1 <= n <= 16), LrpGammaRule,
         {'gamma': gamma}),
        (filter_by_layer_index_type(lambda n: 17 <= n <= 30), LrpEpsilonRule,
         {'epsilon': epsilon}),
        (filter_by_layer_index_type(lambda n: 31 <= n), LrpZeroRule, {}),
    ]

    return rule_layer_map


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
        torch.save(torch_object,
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
            # Plot image comparison for each image in batch to be able to save to file.
            original_input_1chw: torch.Tensor = pf_instance.original_input_nchw[
                image_index].unsqueeze(dim=0)
            flipped_input_1chw: torch.Tensor = pf_instance.flipped_input_nchw[
                image_index].unsqueeze(dim=0)
            relevance_scores_1chw: torch.Tensor = pf_instance.relevance_scores_nchw[
                image_index].unsqueeze(dim=0)
            acc_flip_mask_1hw: torch.Tensor = pf_instance.acc_flip_mask_nhw[image_index].unsqueeze(
                dim=0)

            pf.plot.plot_image_comparison(batch_size=1,
                                          original_input_nchw=original_input_1chw.cpu(),
                                          flipped_input_nchw=flipped_input_1chw.cpu(),
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

    # Init layer filter
    vgg16_target_types: Tuple[type, type] = (Linear, AvgPool)
    filter_by_layer_index_type: LayerFilter = LayerFilter(model)
    filter_by_layer_index_type.set_target_types(vgg16_target_types)

    rule_layer_map: List[
        Tuple[
            List[str], rules.LrpRule,
            Dict[str, Union[torch.Tensor, float]]
        ]
    ] = _get_rule_layer_map_by_experiment_id(filter_by_layer_index_type=filter_by_layer_index_type)

    lrp_instance: LRP = LRP(model)
    lrp_instance.convert_layers(rule_layer_map)
    relevance_scores_nchw: torch.Tensor = lrp_instance.relevance(input_nchw=input_nchw,
                                                                 label_idx_n=label_idx_n).to(
                                                                     device=DEVICE)

    Helpers.save_plot_lrp_results(relevance_scores_nchw=relevance_scores_nchw,
                                  batch_index=batch_index)

    return lrp_instance


def run_pixel_flipping_experiment(lrp_instance: LRP,
                                  batch_index: int) -> PixelFlipping:
    r"""Run the pixel flipping experiment.

    :param lrp_instance: LRP instance
    :param batch_index: Index of the batch

    :return: Pixel flipping instance
    """
    pf_instance: PixelFlipping = PixelFlipping(perturbation_steps=PERTURBATION_STEPS,
                                               perturbation_size=PERTURBATION_SIZE,
                                               perturb_mode=PerturbModes.INPAINTING)
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


def run_experiments() -> None:
    r"""Run Layer-wise Relevance Propagation and Pixel-Flipping experiments."""
    # Enable reproducibility
    torch.manual_seed(SEED)

    print(f'Batch size = {BATCH_SIZE}')

    my_dataloader: torch.utils.data.DataLoader = imagenet_data_loader(root=DATASET_ROOT,
                                                                      batch_size=BATCH_SIZE,
                                                                      classes=IMAGE_CLASSES,
                                                                      seed=SEED)

    for my_batch_index, (my_image_batch, my_ground_truth_labels) in enumerate(my_dataloader):
        my_image_batch.to(device=DEVICE, non_blocking=True)
        my_ground_truth_labels.to(device=DEVICE, non_blocking=True)

        # Run LRP experiment
        my_lrp_instance = run_lrp_experiment(image_batch=my_image_batch,
                                             batch_index=my_batch_index,
                                             label_idx_n=my_ground_truth_labels)

        # Run Pixel-Flipping/Region Perturbation experiment
        my_pf_instance = run_pixel_flipping_experiment(lrp_instance=my_lrp_instance,
                                                       batch_index=my_batch_index)

        Helpers.save_artifacts(lrp_instance=my_lrp_instance,
                               pf_instance=my_pf_instance,
                               batch_index=my_batch_index)

        print(f'Finished batch {my_batch_index}')

        if my_batch_index + 1 == NUMBER_OF_BATCHES:
            print(f'Done. {my_batch_index + 1} batches processed.')
            break


if __name__ == "__main__":
    args_parser: CommandLine = CommandLine()
    parsed_args: argparse.Namespace = args_parser.parse_arguments()

    EXPERIMENT_ID: int = parsed_args.experiment_id

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_path: Path = parsed_args.config_file

    # Ensure that the configuration file exists.
    if not config_file_path.absolute().exists():
        raise ValueError(
            f'Configuration file {config_file_path.absolute()} does not exist.')

    # pylint: disable=pointless-statement
    config.read(config_file_path)
    # pylint: enable=pointless-statement

    config_section_name: str = 'PARAMETERS'

    BATCH_SIZE: int = config.getint(config_section_name,
                                    'BATCH_SIZE')
    PERTURBATION_STEPS: int = config.getint(config_section_name,
                                            'PERTURBATION_STEPS')
    PERTURBATION_SIZE: int = config.getint(config_section_name,
                                           'PERTURBATION_SIZE')

    # Workspace constants
    DATASET_ROOT: str = config['PATHS']['DATASET_ROOT']
    # Directories to be created (if they don't already exist)
    EXPERIMENT_PARENT_ROOT: str = config['PATHS']['EXPERIMENT_PARENT_ROOT']
    EXPERIMENT_ROOT: str = f'{EXPERIMENT_PARENT_ROOT}/experiment-id-{EXPERIMENT_ID}'

    # Experiment parameters
    NUMBER_OF_BATCHES: int = config.getint(config_section_name,
                                           'NUMBER_OF_BATCHES')

    # Value from config file is loaded as string.
    # Convert string representation of list to list
    # Source: https://stackoverflow.com/a/1894296
    #
    # I am aware of the potential arbitrary code execution vulnerability this implies due to eval().
    # Nevertheless, the code is intended for research purposes. Proceeding after acknowledging
    # and assessing the risks.
    IMAGE_CLASSES: List[str] = ast.literal_eval(config[config_section_name]
                                                ['IMAGE_CLASSES'])

    NUMBER_OF_HYPERPARAMETER_VALUES: int = config.getint(config_section_name,
                                                         'NUMBER_OF_HYPERPARAMETER_VALUES')

    # Total number of experiments will be this number squared.
    # TOTAL_NUMBER_OF_EXPERIMENTS: int = NUMBER_OF_HYPERPARAMETER_VALUES ** 2
    TOTAL_NUMBER_OF_EXPERIMENTS: int = config.getint(config_section_name,
                                                     'TOTAL_NUMBER_OF_EXPERIMENTS')

    if EXPERIMENT_ID < 0 or EXPERIMENT_ID >= TOTAL_NUMBER_OF_EXPERIMENTS:
        raise ValueError(
            f'Experiment ID {EXPERIMENT_ID} is out of range [0-{TOTAL_NUMBER_OF_EXPERIMENTS}].')

    # Derivated workspace constants
    INDIVIDUAL_RESULTS_DIR: str = f'{EXPERIMENT_ROOT}/individual-results'
    TORCH_OBJECTS_DIR: str = f'{EXPERIMENT_ROOT}/pytorch-objects'
    NUMPY_OBJECTS_DIR: str = f'{EXPERIMENT_ROOT}/numpy-objects'

    # PyTorch constants
    SEED: int = 0
    DEVICE: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Model parameters
    VGG16_IMAGE_DIM: int = 224
    CHANNELS: int = 3
    HEIGHT: int = VGG16_IMAGE_DIM
    WIDTH: int = VGG16_IMAGE_DIM
    INPUT_SHAPE: Tuple[int, int, int, int] = (BATCH_SIZE,
                                              CHANNELS,
                                              HEIGHT,
                                              WIDTH)

    # Plotting parameters
    DPI: float = 150
    # Toggle for plt.show() for each figure
    SHOW_PLOT: bool = False

    Helpers.create_directories_if_not_exists(EXPERIMENT_ROOT,
                                             INDIVIDUAL_RESULTS_DIR,
                                             TORCH_OBJECTS_DIR,
                                             NUMPY_OBJECTS_DIR)

    # Copying config file to experiment directory for reproducibility of results.
    print('Copying config file to experiment directory:'
          f'{EXPERIMENT_PARENT_ROOT}.')
    # Source: https://stackoverflow.com/a/33626207
    shutil.copy(config_file_path, EXPERIMENT_PARENT_ROOT)

    # Get filename of this file (without absolute path and without extension)
    filename_no_ext: str = Path(__file__).stem
    absolute_path_no_ext: str = f'{EXPERIMENT_PARENT_ROOT}/' \
        f'{filename_no_ext}-locals-filtered-by-type'

    # Create a dictionary from locals() with entries filtered by type to avoid common pitfalls
    # of trying to save modules or classes which are not accepted by numpy.save.
    local_vars_dict: Dict[str, Any]
    local_vars_dict = {dict_key: dict_val for dict_key, dict_val in locals().items()
                       if isinstance(dict_val,
                                     (str, int, list, tuple, dict))}

    # Save local variables to file for archival purposes.

    # Save local variables as dictionary
    numpy.save(file=absolute_path_no_ext + '.npy',
               arr=local_vars_dict)

    # Save local variables as text (human-readable)
    with open(file=absolute_path_no_ext + '.txt',
              mode='w',
              encoding='utf8') as file:
        file.write(str(local_vars_dict))

    run_experiments()
