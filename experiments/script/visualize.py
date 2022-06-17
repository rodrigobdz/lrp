r"""Visualize results from LRP and Pixel-Flipping experiments."""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

import argparse
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy
import torch
from matplotlib import pyplot as plt

from lrp import rules

if __name__ == "__main__":

    # Read configuration file passed as argument.
    config = ConfigParser(interpolation=ExtendedInterpolation())
    parser = argparse.ArgumentParser(description='Specify the path to the configuration file.',
                                     epilog='For more information,'
                                     ' review the batch_lrp_pf.py script')
    parser.add_argument('-c', '--config-file',
                        type=Path,
                        help='Absolute path to configuration file'
                        ' with parameters for experiments',
                        required=True)
    parsed_args: argparse.Namespace = parser.parse_args()
    config_file_path: Path = parsed_args.config_file

    # Ensure that the configuration file exists.
    if not config_file_path.absolute().exists():
        raise ValueError(
            f'Configuration file {config_file_path.absolute()} does not exist.')

    # pylint: disable=pointless-statement
    config.read(config_file_path)
    # pylint: enable=pointless-statement

    param_section_name: str = 'PARAMETERS'
    plots_section_name: str = 'PLOTS'

    BATCH_SIZE: int = config.getint(param_section_name,
                                    'BATCH_SIZE')
    PERTURBATION_STEPS: int = config.getint(param_section_name,
                                            'PERTURBATION_STEPS')
    PERTURBATION_SIZE: int = config.getint(param_section_name,
                                           'PERTURBATION_SIZE')

    # Path to dir where the results should be stored.
    # Directories will be created, if they don't already exist.
    EXPERIMENT_PARENT_ROOT: str = config['PATHS']['EXPERIMENT_PARENT_ROOT']

    TITLE: str = config[plots_section_name]['TITLE']
    X_LABEL: str = config[plots_section_name]['X_LABEL']
    Y_LABEL: str = config[plots_section_name]['Y_LABEL']
    PLOT_PATH: str = config[plots_section_name]['PLOT_PATH']

    experiment_parent_path: Path = Path(EXPERIMENT_PARENT_ROOT)
    auc_list: List[str] = list(
        experiment_parent_path.glob('**/batch-*-area-under-the-curve.npy')
    )
    rule_layer_map_list: List[str] = list(
        experiment_parent_path.glob('**/batch-*-lrp-rule-layer-map.npy')
    )

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
        # TODO: Add support for different LRP composite rules.
        # Current order of rule in rule_layer_map:
        # lrp.rules.LrpZBoxRule
        # lrp.rules.LrpGammaRule
        # lrp.rules.LrpGammaRule
        # lrp.rules.LrpGammaRule
        gamma_one: float = rule_layer_map[1][2].get('gamma')
        gamma_two: float = rule_layer_map[2][2].get('gamma')

        x_values.append(gamma_one)
        y_values.append(gamma_two)

    z_values: List[float] = []

    for auc_file in auc_list:
        z_values.append(numpy.load(file=auc_file,
                                   allow_pickle=True).item())

    plt.tricontourf(x_values, y_values, z_values)

    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.colorbar()

    plt.savefig(fname=PLOT_PATH, facecolor='w')
