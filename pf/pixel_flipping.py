r'''Pixel-Flipping Algorithm for Evaluation of Explanations.
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

from .perturbation_modes.random.random_number_generators import RandomNumberGenerator, UniformRNG
from .perturbation_modes.random.flip import flip_random
from .perturbation_modes.inpainting.flip import flip_inpainting
from .perturbation_modes.constants import PerturbModes
from .objectives import sort
from . import utils
from .decorators import timer
from lrp import norm
import lrp.plot


class PixelFlipping:
    r'''Pixel-Flipping Algorithm.'''

    def __init__(self,
                 perturbation_steps: int = 100,
                 perturbation_size: int = 1,
                 verbose: bool = False,
                 perturb_mode: str = PerturbModes.INPAINTING,
                 ran_num_gen: Optional[RandomNumberGenerator] = None,
                 ) -> None:
        r'''Constructor

        :param perturbation_steps: Number of perturbation steps.
        :param perturbation_size: Size of the region to flip.
        A size of 1 corresponds to single pixels, whereas a higher number to patches of size nxn.
        :param verbose: Whether to print debug messages.
        :param perturb_mode: Perturbation technique to decide how to replace flipped values.

        :param ran_num_gen: Random number generator to use. Only available with PerturbModes.RANDOM.
        '''
        # Ensure perturbation size is a valid number.
        if perturbation_size < 1:
            raise ValueError(
                'Perturbation size must be greater than or equal to 1.')

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

        # Store flipped input after perturbation.
        self.flipped_input: torch.Tensor

        # Store (accumulative) masks applied to flip the input together in a single mask.
        self.acc_flip_mask: torch.Tensor

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

    @timer
    def __call__(self,
                 input: torch.Tensor,
                 relevance_scores: torch.Tensor,
                 forward_pass: Callable[[torch.Tensor], float],
                 should_loop: bool = True
                 ) -> Optional[Generator[Tuple[torch.Tensor, float], None, None]]:
        r'''Run pixel-flipping algorithm.

        :param input: Input to be explained.
        :param relevance_scores: Relevance scores.
        :param forward_pass: Classifier function to measure accuracy change in pixel-flipping iterations.
        :param should_loop: Whether to loop over the generator or not.

        :yields: Tuple of flipped input and updated classification score
        after one perturbation step.

        :returns: None if should_loop is True, otherwise a generator.
        '''

        # Store input for comparison at the end.
        self.original_input: torch.Tensor = input.detach().clone()

        self.relevance_scores: torch.Tensor = relevance_scores.detach().clone()

        # Initialize accumulative mask to False.
        # Each mask used to flip the input will be stored in this tensor with logical OR.
        #
        # Input has dimensions (batch_size, channels, height, width).
        # Accumulative mask has dimensions (height, width).
        #   .sum(dim=0) is used to reduce the number of channels to 1.
        #   I.e., to convert from (channels, height, width) to (height, width).
        self.acc_flip_mask: torch.Tensor = torch.zeros(
            *input[0].shape, dtype=torch.bool).sum(dim=0)

        # Count number of pixels affected by the perturbation.
        #
        # If perturbation size is one, then we only need to flip one pixel.
        # Otherwise, we need to flip a patch of size nxn = # affected pixels.
        perturbation_size_numel: int = self.perturbation_size**2

        # Verify that number of flips does not exceed the number of elements in the input.
        if (perturbation_size_numel * self.perturbation_steps) > torch.numel(input):
            raise ValueError(
                f'''perturbation_size_numel * perturbation_steps =
{perturbation_size_numel} * {self.perturbation_steps} = {perturbation_size_numel * self.perturbation_steps}
exceeds the number of elements in the input ({torch.numel(input)}).''')

        pixel_flipping_generator: Generator[Tuple[torch.Tensor, float], None, None] = self._generator(
            input, relevance_scores, forward_pass)

        # Toggle to return generator or loop automatically over it.
        if not should_loop:
            return pixel_flipping_generator

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

        sorted_values: torch.Tensor = sort._argsort(relevance_scores)
        mask_iter: Generator[torch.Tensor, None,
                             None] = sort._mask_generator(relevance_scores,
                                                          sorted_values,
                                                          self.perturbation_size)

        for i in range(self.perturbation_steps):
            self.logger.debug(f'Step {i}')

            # DEBUG: Verify what happens with mask_iter when mask_iter selects multiple pixels at once.
            # Reproduce error:
            # steps 100
            # simultaneous 10

            # Mask to select which pixels to flip.
            mask: torch.Tensor = next(mask_iter)

            # Flip pixels with respective perturbation technique

            if self.perturb_mode == PerturbModes.RANDOM:
                flip_random(image=flipped_input,
                            mask=mask,
                            perturbation_size=self.perturbation_size,
                            ran_num_gen=self.ran_num_gen,
                            low=low,
                            high=high,
                            logger=self.logger)

            elif self.perturb_mode == PerturbModes.INPAINTING:
                flipped_input = norm.denorm_img_pxls(
                    norm.ImageNetNorm.inverse_normalize(flipped_input))

                flipped_input = flip_inpainting(image=flipped_input.int(),
                                                mask=mask,
                                                logger=self.logger).float()

                flipped_input = norm.ImageNetNorm.normalize(
                    norm.norm_img_pxls(flipped_input))

            else:
                raise NotImplementedError(
                    f'Perturbation mode \'{self.perturb_mode}\' not implemented yet.')

            # Store flipped input for comparison at the end with the original input.
            self.flipped_input: torch.Tensor = flipped_input

            # Store number of flipped pixels before this perturbation step.
            flipped_pixel_count: int = self.acc_flip_mask.count_nonzero().item()

            # Squeeze mask to empty channel dimension.
            self.acc_flip_mask: torch.Tensor = torch.logical_or(
                self.acc_flip_mask, mask.squeeze())

            # Calculate delta of flipped pixels:
            #   I.e., total number of flipped pixels in this perturbation step
            #   minus the count of already flipped pixels.
            flipped_pixel_count = self.acc_flip_mask.count_nonzero().item() - \
                flipped_pixel_count

            self.logger.info(f'Flipped {flipped_pixel_count} pixels.')

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

    def plot_image_comparison(self) -> None:
        r'''Plot the original input and the perturbed input to visualize the changes.
        '''

        # Create grid of original and perturbed images.
        _, ax = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

        # Plot images.
        lrp.plot.plot_imagenet_tensor(self.original_input, ax=ax[0][0])
        lrp.plot.plot_imagenet_tensor(self.flipped_input, ax=ax[0][1])

        # Plot heatmaps.
        kwargs: dict = {'width': 5, 'height': 5, 'show_plot': False}
        lrp.plot.heatmap(self.relevance_scores[0].sum(dim=0).detach().numpy(),
                         fig=ax[1][0], **kwargs)
        lrp.plot.heatmap(self.acc_flip_mask, fig=ax[1][1], **kwargs)

        # Set titles.
        ax[0][0].text(75, -10, 'Original Input', size=12)
        ax[0][1].text(75, -10, 'Flipped Input', size=12)
        ax[1][0].text(75, -10, 'Relevance scores', size=12)
        ax[1][1].text(75, -10, 'Perturbed Regions', size=12)

        # Show plots.
        plt.show()

    # TODO: Add function to create heatmap of flipped values only with mask
