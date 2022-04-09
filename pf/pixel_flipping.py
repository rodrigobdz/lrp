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

from typing import Optional, Generator, Callable, List, Tuple, Union
from matplotlib import pyplot as plt
from .random_number_generators import RandomNumberGenerator, UniformRNG
from . import utils


class PixelFlipping:
    r'''Pixel-Flipping Algorithm.'''

    DEFAULT_PERTURBATION_SIZE: int = 1

    def __init__(self,
                 perturbation_steps: int = 100,
                 perturbation_size: Union[int, Tuple[int]
                                          ] = DEFAULT_PERTURBATION_SIZE,
                 verbose: bool = False,
                 ran_num_gen: Optional[RandomNumberGenerator] = None,
                 ) -> None:
        r'''Constructor

        :param perturbation_steps: Number of perturbation steps.
        :param perturbation_size: Size of the region to flip.
        A size of 1 corresponds to single pixels, whereas a tuple to patches.
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

        # Error handling when perturbation size does not conform to standard format of two elements: width and height.
        # FIXME: logger.exception and raise ValueError seems redundant.
        if isinstance(perturbation_size, tuple) and len(perturbation_size) >= 2:
            self.logger.exception(
                f'Perturbation size parameter expected a max. of two elements, got {len(perturbation_size)}')
            raise ValueError(
                f'Perturbation size must be a tuple of length 1 or 2, got {len(perturbation_size)}.')

        # Number of times to flip pixels/patches
        self.perturbation_steps: int = perturbation_steps

        # Size of the region to flip
        self.perturbation_size: Union[int, Tuple[int]] = perturbation_size

        # List to store updated classification scores after each perturbation step.
        self.class_prediction_scores: List[float] = []

        self.ran_num_gen: RandomNumberGenerator = ran_num_gen or UniformRNG()

    def __call__(self,
                 input: torch.Tensor,
                 relevance_scores: torch.Tensor,
                 forward_pass: Callable[[torch.Tensor], float],
                 should_loop: bool = True,
                 simultaneous_pixel_flips: int = 1,
                 ) -> None:
        r'''Run pixel-flipping algorithm.

        :param input: Input to be explained.
        :param relevance_scores: Relevance scores.
        :param forward_pass: Classifier function to measure accuracy change in pixel-flipping iterations.
        :param should_loop: Whether to loop over the generator or not.
        :param simultaneous_pixel_flips: Number of pixels to flip simultaneously.

        :yields: Tuple of flipped input and updated classification score
        after one perturbation step.
        '''

        # Number of pixels/patches to flip simultaneously
        self.simultaneous_pixel_flips: int = simultaneous_pixel_flips

        # Verify that number of flips does not exceed the number of elements in the input.
        # FIXME: Include perturbation_size in calculation
        if (simultaneous_pixel_flips * self.perturbation_steps) > torch.numel(input):
            raise ValueError(
                f'simultaneous_pixel_flips * perturbation_steps ({simultaneous_pixel_flips * self.perturbation_steps}) exceeds the number of elements in the input ({torch.numel(input)}).')

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

        # TODO: Add support for custom low and high bounds (random number generation).
        # Infer (min. and max.) bounds of input for random number generation
        low: float = float(input.min())
        high: float = float(input.max())

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

            # FIXME: Vectorize simultaneous flip operation
            for k in range(self.simultaneous_pixel_flips):
                self.logger.debug(
                    f'Step {i} - simultaneous flip {k}/{self.simultaneous_pixel_flips}')

                # FIXME: Verify what happens with mask_generator when mask_generator selects multiple pixels at once.
                # Reproduce error:
                # steps 100
                # simultaneous 10

                # Mask to select which pixels to flip.
                mask: torch.Tensor = next(mask_generator)

                # Flip pixels
                self._flip(input=flipped_input,
                           mask=mask,
                           low=low,
                           high=high,
                           perturbation_size=self.perturbation_size)

            # Measure classification accuracy change
            self.class_prediction_scores.append(forward_pass(flipped_input))

            self.logger.debug(
                f'Classification score: {self.class_prediction_scores[-1]}')

            yield flipped_input, self.class_prediction_scores[-1]

    def _flip(self,
              input: torch.Tensor,
              mask: torch.Tensor,
              low: float = 0.0,
              high: float = 1.0,
              perturbation_size: Union[int, Tuple[int]
                                       ] = DEFAULT_PERTURBATION_SIZE
              ) -> None:
        r'''Flip pixels of input in-place according to the relevance scores.

        Pixels to be flipped will be replaced by samples drawn from the interval between the
        values of the low and high parameters.

        :param input: Input to be flipped.
        :param relevance_scores: Relevance scores.
        :param low: Lower bound of the range of values to be flipped.
        :param high: Upper bound of the range of values to be flipped.
        :param perturbation_size: Size of the region to flip.
        A size of 1 corresponds to single pixels, whereas a tuple to patches.
        '''

        if isinstance(perturbation_size, tuple):
            raise ValueError(
                'Region Perturbation algorithm not supported yet. Size can only be a single integer value.')

        # Draw a random number.
        flip_value: float = self.ran_num_gen.draw(
            low=low, high=high, size=perturbation_size)

        # Debug: Compute indices selected for flipping in mask.
        flip_indices = mask.nonzero().flatten().tolist()
        # Debug: Count how many elements are set to True—i.e., would be flipped.
        flip_count: int = input[0][mask].count_nonzero().item()
        self.logger.debug(
            f'Flipping X[0]{flip_indices} to {flip_value}: {flip_count} element(s).')

        # Error handling
        # FIXME: Remove this check to vectorize operation
        # FIXME: Check what happens when flip_count is greater than one.
        # It seems like the mask_generator returns the mask repeatedly for #simultaneous flips times.
        if flip_count != 1:
            self.logger.debug(
                f'''Flip count {flip_count} is not one. The mask is flipping more than one element.''')

        # Flip pixels/patch
        # Disable gradient computation for the pixel-flipping operations.
        # Avoid error "A leaf Variable that requires grad is being used in an in-place operation."
        with torch.no_grad():
            # TODO: Add support for patches
            input[0][mask] = flip_value

    def plot(self,
             title: str = '',
             xlabel: str = '',
             ylabel: str = '') -> None:
        r'''Plot the updated prediction scores throughout the perturbation steps of
        the pixel-flipping algorithm to visualize the accuracy of the explanation.

        :param title: Title of the plot.

        :raises ValueError: If class prediction scores are empty.
        '''

        # Error handling
        # FIXME: logger.exception and raise ValueError seems redundant.
        if not self.class_prediction_scores:
            self.logger.exception('Executed plot() before calling __call__()')
            raise ValueError(
                'No class prediction scores to plot. Please run pixel-flipping first.')

        plt.plot(self.class_prediction_scores)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
