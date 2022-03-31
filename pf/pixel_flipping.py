r'''Pixel-Flipping Algorithm for Evaluation of Heatmaps.
Related to Region Perturbation.

Source in Chicago citation format:
  Samek, Wojciech, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, and Klaus-Robert Müller.
  "Evaluating the visualization of what a deep neural network has learned."
  IEEE transactions on neural networks and learning systems 28, no. 11 (2016): 2660-2673.
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'


import torch
import logging
import sys

from typing import Optional, Generator, Callable, List, Tuple
from matplotlib import pyplot as plt
from .objectives import PixelFlippingObjectives
from .random_number_generators import RandomNumberGenerator, UniformRNG


class PixelFlipping:
    r'''Pixel-Flipping Algorithm.'''

    def __init__(self,
                 perturbation_steps: int = 100,
                 verbose: bool = False,
                 ran_num_gen: Optional[RandomNumberGenerator] = None
                 ) -> None:
        r'''Constructor

        :param perturbation_steps: Number of perturbation steps.
        :param verbose: Whether to print debug messages.
        :param ran_num_gen: Random number generator to use.
        '''

        logging.basicConfig(
            stream=sys.stderr,
            format='%(levelname)-8s  %(message)s'
        )

        # Init logger instance
        self.logger = logging.getLogger(__name__)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        # Number of times to flip pixels/patches
        self.perturbation_steps: int = perturbation_steps

        # List to store updated classification scores after each perturbation step.
        self.class_prediction_scores: List[float] = []

        self.ran_num_gen: RandomNumberGenerator = ran_num_gen or UniformRNG()

    def __call__(self,
                 input: torch.Tensor,
                 relevance_scores: torch.Tensor,
                 forward_pass: Callable[[torch.Tensor], float],
                 should_loop: bool = True
                 ) -> None:
        r'''Run pixel-flipping algorithm.

        :param input: Input to be explained.
        :param relevance_scores: Relevance scores.
        :param forward_pass: Classifier function to measure accuracy change in pixel-flipping iterations.
        :param should_loop: Whether to loop over the generator or not.

        :yields: Tuple of flipped input and updated classification score
        after one perturbation step.
        '''

        pixel_flipping_generator: Generator[Tuple[torch.Tensor, float], None, None] = self._generator(
            input, relevance_scores, forward_pass)

        # Toggle to return generator or loop automatically over it.
        if not should_loop:
            return

        utils._loop(pixel_flipping_generator)

    def _generator(self,
                   input: torch.Tensor,
                   relevance_scores: torch.Tensor,
                   forward_pass: Callable[[torch.Tensor], float]
                   ) -> Generator[Tuple[torch.Tensor, float], None, None]:
        r'''Generator to flip pixels of input according to the relevance scores.

        :param input: Input to be explained.
        :param relevance_scores: Relevance scores.
        :param forward_pass: Classifier function to measure accuracy change in pixel-flipping iterations.

        :yields: Tuple of flipped input and updated classification score
        after one perturbation step.

        Generators are annotated in the format: Generator[YieldType, SendType, ReturnType],
        therefore, SendType and ReturnType are set to None above.
        Source: https://docs.python.org/3/library/typing.html
        '''

        # Deep copy input to avoid in-place modifications.
        # Detach input from computational graph to avoid computing gradient for the
        # pixel-flipping operations.
        flipped_input: torch.Tensor = input.detach().clone()
        flipped_input.requires_grad = False

        # Get initial classification score.
        self.class_prediction_scores.append(forward_pass(flipped_input))

        self.logger.debug(
            f'Initial classification score {self.class_prediction_scores[-1]}')

        mask_generator: Generator[torch.Tensor, None,
                                  None] = utils._argsort(relevance_scores)

        for i in range(self.perturbation_steps):
            self.logger.debug(f'Step {i}')

            # Mask to select which pixels to flip.
            mask: torch.Tensor = next(mask_generator)

            # Flip pixels
            self._flip(flipped_input, mask)

            # Measure classification accuracy change
            self.class_prediction_scores.append(forward_pass(flipped_input))

            self.logger.debug(
                f'Classification score: {self.class_prediction_scores[-1]}')

            yield flipped_input, self.class_prediction_scores[-1]

    def _flip(self,
              input: torch.Tensor,
              mask: torch.Tensor
              ) -> None:
        r'''Flip pixels of input in-place according to the relevance scores.

        :param input: Input to be flipped.
        :param relevance_scores: Relevance scores.
        '''

        # Draw a random number.
        flip_value: float = self.ran_num_gen.draw()

        # Debug: Compute indices selected for flipping in mask.
        flip_indices = mask.nonzero().flatten().tolist()
        # Debug: Count how many elements are set to True—i.e., would be flipped.
        flip_count: torch.Tensor = input[0][mask].count_nonzero()
        self.logger.debug(
            f'Flipping X[0]{flip_indices} to {flip_value}: {flip_count} elements.')

        # Error handling
        if flip_count != 1:
            self.logger.exception(
                f'Flip count {flip_count} is not one. This means that the mask is flipping more than one element.')

        # Flip pixels/patch
        # Disable gradient computation for the pixel-flipping operations.
        # Avoid error "A leaf Variable that requires grad is being used in an in-place operation."
        with torch.no_grad():
            # TODO: Add support for patches
            input[0][mask] = flip_value

    def plot(self) -> None:
        r'''Plot the updated prediction scores throughout the perturbation steps of
        the pixel-flipping algorithm to visualize the accuracy of the explanation.

        :raises ValueError: If class prediction scores are empty.
        '''

        # Error handling
        # FIXME: logger.exception and raise ValueError seems redundant.
        if not self.class_prediction_scores:
            self.logger.exception('Executed plot() before calling __call__()')
            raise ValueError(
                'No class prediction scores to plot. Please run pixel-flipping first.')

        plt.plot(self.class_prediction_scores)
        plt.show()
