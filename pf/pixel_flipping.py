r'''Pixel-Flipping Algorithm for Evaluation of Heatmaps.
Also called Region Perturbation.

Source in Chicago citation format:
Samek, Wojciech, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, and Klaus-Robert Müller. "Evaluating the visualization of what a deep neural network has learned." IEEE transactions on neural networks and learning systems 28, no. 11 (2016): 2660-2673.
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


from typing import Optional, Generator, Callable, List
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

import torch
import numpy
import logging
import sys


# logging.warning('warning')
# logging.debug('debug')
# logging.info('info')
# logging.critical('critical')
# logging.exception('exception')


class RandomNumberGenerator(ABC):
    r'''
    Base random number generator class. Acts as interface for number generator classes.
    '''

    def __init__(self,
                 seed: int = 42):
        '''
        Constructor

        :param seed: Seed for the random number generator.
        '''
        self.generator: Generator = numpy.random.default_rng(seed=seed)

    @abstractmethod
    def draw(self) -> float:
        r'''
        Draws a random number from the distribution.
        '''


class UniformRNG(RandomNumberGenerator):
    r'''
    Uniform random number generator class.
    '''

    def draw(self, lower: float = -1.0, upper: float = 1.0, size: int = 1) -> float:
        r'''
        Draws a random number from the distribution.

        :param lower: Lower bound of the distribution.
        :param upper: Upper bound of the distribution.
        :param size: Number of random numbers to draw.

        :returns: A random number from the uniform distribution.
        '''
        return self.generator.uniform(lower, upper)


class PixelFlipping:
    r'''Pixel-Flipping Algorithm.'''

    def __init__(self,
                 perturbation_steps: int = 100,
                 verbose: bool = False,
                 ran_num_gen: Optional[RandomNumberGenerator] = None
                 ) -> None:
        r'''Constructor'''

        # # Code for patches
        # if len(size) >= 2:
        #     raise ValueError(
        #         f'Size must be a tuple of length 1 or 2, got {len(size)}.')

        # self.size: Tuple[int, int] = size
        logging.basicConfig(
            stream=sys.stderr,
            format='%(levelname)-8s  %(message)s'
        )
        if verbose:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

        # Number of times to flip pixels/patches
        self.perturbation_steps: int = perturbation_steps

        self.class_prediction_scores: List[float] = []

        self.ran_num_gen: RandomNumberGenerator = ran_num_gen or UniformRNG()

    def __call__(self,
                 X: torch.Tensor,
                 relevance_scores: torch.Tensor,
                 f: Callable[[torch.Tensor], float],
                 g: Callable[[torch.Tensor], torch.Tensor]
                 ) -> torch.Tensor:
        r'''Flip pixels of input according to the relevance scores.

        :param X: Input to be explained.
        :param relevance_scores: Relevance scores.
        :param f: Classifier function to measure accuracy change in pixel-flipping iterations.

        :returns: Flipped input.
        '''

        # Deep copy input to avoid in-place modifications.
        # Detach X from computational graph to avoid computing gradient for the
        # pixel-flipping operations.
        flipped_input: torch.Tensor = X.detach().clone()
        flipped_input.requires_grad = False

        # FIXME: Delete!
        relevance_scores: torch.Tensor = relevance_scores.detach().clone()
        relevance_scores.requires_grad = False

        # Get initial classification score.
        self.class_prediction_scores.append(f(flipped_input))

        self.logger.debug(
            f'Initial classification score {self.class_prediction_scores[-1]}')

        for i in range(self.perturbation_steps):
            self.logger.debug(f'Step {i}')

            # Flip pixels
            self.flip(flipped_input, relevance_scores)

            # Measure classification accuracy change
            self.class_prediction_scores.append(f(flipped_input))

            self.logger.debug(
                f'Classification score: {self.class_prediction_scores[-1]}')

            # FIXME: Delete. Relevance scores should not be updated.
            relevance_scores = g(flipped_input)
            # yield flipped_input, self.class_prediction_scores[-1]
        return flipped_input

    def flip(self,
             X: torch.Tensor,
             relevance_scores: torch.Tensor
             ) -> None:
        r'''Flip pixels of input in-place according to the relevance scores.

        :param X: Input to be flipped.
        :param relevance_scores: Relevance scores.
        '''

        # Draw a random number.
        flip_value: float = self.ran_num_gen.draw()

        # Get value of maximum relevance score
        flip_threshold = torch.max(relevance_scores[0]).item()

        # Mask with elements that equal to the maximum relevance score set to True, otherwise False.
        # In practice, this is a representation of the indexes to flip.
        max_rel_score_mask: torch.Tensor = relevance_scores[0] == flip_threshold

        flip_indices = max_rel_score_mask.nonzero().flatten().tolist()

        # Count how many elements are set to True—i.e., would be flipped.
        flip_count: torch.Tensor = X[0][max_rel_score_mask].count_nonzero()

        self.logger.debug(
            f'Flipping X[0]{flip_indices} to {flip_value}: {flip_count} elements.')

        with torch.no_grad():
            X[0][relevance_scores[0] == flip_threshold] = flip_value

    def plot(self) -> None:
        r'''Plot the updated prediction scores throughout the perturbation steps of the pixel-flipping algorithm to
        visualize the accuracy of the explanation.

        :raises ValueError: If class prediction scores are empty.
        '''

        if not self.class_prediction_scores:
            raise ValueError('No class prediction scores to plot.')

        plt.plot(self.class_prediction_scores)
        plt.show()

# TODO: Corroborar dim val.
# sorted_values, sorted_indices = relevance_scores.sort(dim=-1,
#                                                       descending=True,
#                                                       stable=True)
# flip_val = self.generator.uniform(
#     low=RANDOM_LOW, high=RANDOM_HIGH)

# r_mask[r_mask > s_values[K]] = flip_val

# # def pixel_flipping() -> None:

# # Number of relevance scores in R
# N = 10
# # Number of flips
# K = 5
# # Parameters for random number generation
# RANDOM_LOW = 0
# RANDOM_HIGH = 100

# # Random Number Generator
# torch_rng = torch.Generator()
# torch_rng.manual_seed(SEED)

# numpy_rng = default_rng(seed=SEED)
# numpy_rng_flip_val = numpy_rng.uniform(low=RANDOM_LOW, high=RANDOM_HIGH)

# R = torch.randint(low=RANDOM_LOW, high=RANDOM_HIGH,
#                   size=(N,), generator=torch_rng)
# r_mask = R.detach().clone()
# r_seq = R.detach().clone()

# s_values, s_indices = R.sort(dim=-1, descending=True, stable=True)

# r_mask[r_mask > s_values[K]] = numpy_rng_flip_val
