r"""Visualize results from LRP and Pixel-Flipping experiments."""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy
import torch
from matplotlib import pyplot as plt

from lrp import rules

if __name__ == "__main__":
    # TODO: Load constants from pickle/npy file.
    BATCH_SIZE: int = 4
    # Experiment constants
    PERTURBATION_STEPS: int = 28
    PERTURBATION_SIZE: int = 8

    # Workspace constants
    WORKSPACE_ROOT: str = '/Users/rodrigobermudezschettino/Documents/personal' \
        '/unterlagen/bildung/uni/master/masterarbeit'
    # Directories to be created (if they don't already exist)
    EXPERIMENT_PARENT_ROOT: str = f'{WORKSPACE_ROOT}/experiment-results/2022-05-30/' \
        f'lrp-pf-auc/batch-size-{BATCH_SIZE}'

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
        # lrp.rules.LrpEpsilonRule
        # lrp.rules.LrpZeroRule
        gamma: float = rule_layer_map[1][2].get('gamma')
        epsilon: float = rule_layer_map[2][2].get('epsilon')

        x_values.append(gamma)
        y_values.append(epsilon)

    z_values: List[float] = []

    for auc_file in auc_list:
        z_values.append(numpy.load(file=auc_file,
                                   allow_pickle=True).item())

    plt.tricontourf(x_values, y_values, z_values)

    plt.title('Contourf plot of AUC values across experiments')
    plt.xlabel('Gammas for LRP-Gamma')
    plt.ylabel('Epsilons for LRP-Epsilon')
    plt.colorbar()
    plt.show()
