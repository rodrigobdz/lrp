r'''Experiments-related functions for convenience.
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
import os.path
import pickle
import torch
import numpy
from lrp.core import LRP
from lrp import rules


class Experiments:
    r'''Boilerplatte code for preparing the experiments and avoid repetition
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 X: torch.Tensor,
                 name_map: List[Tuple[List[str], rules.LrpRule, Dict[str, Union[torch.Tensor, float]]]],
                 output_dir_path: str) -> None:
        r'''Store base variables

        :param model: Model to be explained
        :param X: Input data
        :param name_map: List of tuples containing layer names and LRP rule
        :param output_dir_path: Path to directory where artifacts will be stored
        '''
        self.model = model
        self.X = X
        self.name_map = name_map
        self.output_dir_path = output_dir_path

    def generate_multiple_lrp_relevances(self,
                                         hyperparams: numpy.ndarray,
                                         name_map_hyperparam_setter: Callable[[
                                             float], None]
                                         ) -> List[Tuple[float, torch.Tensor]]:
        r'''Compute multiple LRP relevances for a given set of hyperparameters.
        Cache the results in a file or load cached results, if possible.
        Stores the passed hyperparameters in a file for reference.

        :param hyperparams: Hyperparameters to be used
        :param name_map_hyperparam_setter: Function to set missing hyperparameters

        :return: List of tuples containing hyperparameter and relevance
        '''
        cache_filename = f'{self.output_dir_path}/results.pickle'

        # Check if cached results already exist and return these
        if os.path.isfile(cache_filename):
            print('Using cached results')
            # Load cached results
            with open(cache_filename, 'rb') as f:
                return pickle.load(f)

        results: List[Tuple[float, torch.Tensor]] = [None]*len(hyperparams)

        for i, hyperparam_val in enumerate(hyperparams):
            # Set hyperparameter value for each LRP iteration
            name_map_hyperparam_setter(hyperparam_val)

            # Init LRP object
            lrp = LRP(self.model)

            # Convert layers according to name map with given hyperparameter value
            lrp.convert_layers(self.name_map)

            # Compute relevance scores
            R: torch.Tensor = lrp.relevance(self.X)

            # Store results
            results[i] = (hyperparam_val, R)

        # Cache results
        #
        # Create intermediate directories if necessary
        Path(self.output_dir_path).mkdir(parents=True, exist_ok=True)
        with open(cache_filename, 'wb') as f:
            pickle.dump(results, f)

        # Save hyperparameter values to file for reference
        numpy.savetxt(
            fname=f'{self.output_dir_path}/hyperparams.csv', X=hyperparams, newline=',')

        return results
