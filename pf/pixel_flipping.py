r'''Pixel-Flipping Algorithm for Evaluation of Heatmaps.
Also called Region Perturbation when perturbation size is greater than one pixel at once.

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

from typing import Generator, Callable, List, Tuple, Union, Optional
from matplotlib import pyplot as plt

from .perturbation_modes.random.flip import flip_random
from .perturbation_modes.random.random_number_generators import RandomNumberGenerator, UniformRNG
from .perturbation_modes.constants import PerturbModes
from .objectives import sort
from . import utils


class PixelFlipping:
    r'''Pixel-Flipping Algorithm.'''

    DEFAULT_PERTURBATION_SIZE: int = 1

    def __init__(self,
                 perturbation_steps: int = 100,
                 perturbation_size: Union[int, Tuple[int]
                                          ] = DEFAULT_PERTURBATION_SIZE,
                 verbose: bool = False,
                 perturb_mode: str = PerturbModes.INPAINTING,
                 ran_num_gen: Optional[RandomNumberGenerator] = None,
                 ) -> None:
        r'''Constructor

        :param perturbation_steps: Number of perturbation steps.
        :param perturbation_size: Size of the region to flip.
        A size of 1 corresponds to single pixels, whereas a tuple to patches.
        :param verbose: Whether to print debug messages.
        :param perturb_mode: Perturbation technique to decide how to replace flipped values.

        :param ran_num_gen: Random number generator to use. Only available with PerturbModes.RANDOM.
        '''

        # Ensure perturbation size conforms to standard format of two elements: width and height.
        if isinstance(perturbation_size, tuple) and len(perturbation_size) >= 2:
            raise ValueError(
                f'Perturbation size must be a tuple of length 1 or 2, got {len(perturbation_size)}.')

        # Ensure ran_num_gen is only specified when the perturbation technique is random.
        if perturb_mode != PerturbModes.RANDOM and ran_num_gen:
            raise ValueError(
                'Argument ran_num_gen is only available with PerturbModes.RANDOM and should not be passed otherwise.')

        # Limit perturbation modes to the ones available in the library.
        if perturb_mode != PerturbModes.INPAINTING and perturb_mode != PerturbModes.RANDOM:
            raise NotImplementedError(
                f'Perturbation mode \'{perturb_mode}\' not implemented yet.')

        logging.basicConfig(
            stream=sys.stderr,
            format='%(levelname)-8s  %(message)s'
        )

        # Init logger instance
        self.logger: logging.Logger = logging.getLogger(__name__)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        # Number of times to flip pixels/patches
        self.perturbation_steps: int = perturbation_steps

        # Size of the region to flip
        self.perturbation_size: Union[int, Tuple[int]] = perturbation_size

        # Name of the perturbation technique
        self.perturb_mode: str = perturb_mode

        if self.perturb_mode == PerturbModes.RANDOM:
            self.ran_num_gen: RandomNumberGenerator = ran_num_gen or UniformRNG()

        # List to store updated classification scores after each perturbation step.
        self.class_prediction_scores: List[float] = []

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

        if self.perturb_mode == PerturbModes.RANDOM:
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
                                  None] = sort._argsort(relevance_scores)

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

                # Flip pixels with respective perturbation technique

                if self.perturb_mode == PerturbModes.RANDOM:
                    flip_random(input=flipped_input,
                                mask=mask,
                                perturbation_size=self.perturbation_size,
                                ran_num_gen=self.ran_num_gen,
                                low=low,
                                high=high,
                                logger=self.logger)

                if self.perturb_mode == PerturbModes.INPAINTING:
                    self._flip(input=flipped_input,
                               mask=mask,
                               perturbation_size=self.perturbation_size)

            # Measure classification accuracy change
            self.class_prediction_scores.append(forward_pass(flipped_input))

            self.logger.debug(
                f'Classification score: {self.class_prediction_scores[-1]}')

            yield flipped_input, self.class_prediction_scores[-1]

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
        if not self.class_prediction_scores:
            raise ValueError(
                'No class prediction scores to plot. Please run pixel-flipping first.')

        plt.plot(self.class_prediction_scores)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
