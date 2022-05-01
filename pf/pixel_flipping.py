r"""Pixel-Flipping Algorithm for Evaluation of Explanations.
Also called Region Perturbation when perturbation size is greater than one pixel at once.

Source in Chicago citation format:
  Samek, Wojciech, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin,
  and Klaus-Robert Müller.
  "Evaluating the visualization of what a deep neural network has learned."
  IEEE transactions on neural networks and learning systems 28, no. 11 (2016): 2660-2673.
"""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import logging
import sys
from typing import Callable, Generator, List, Optional, Tuple

import torch
from matplotlib import pyplot as plt

from lrp import norm

from . import plot, sanity_checks, utils
from .decorators import timer
from .metrics import area_over_the_pertubation_curve, area_under_the_curve
from .objectives import sort
from .perturbation_modes.constants import PerturbModes
from .perturbation_modes.inpainting.flip import flip_inpainting
from .perturbation_modes.random.flip import flip_random
from .perturbation_modes.random.random_number_generators import (
    RandomNumberGenerator, UniformRNG)


class PixelFlipping:
    r"""Pixel-Flipping Algorithm."""

    def __init__(self,
                 perturbation_steps: int = 100,
                 perturbation_size: int = 9,
                 verbose: bool = False,
                 perturb_mode: str = PerturbModes.INPAINTING,
                 ran_num_gen: Optional[RandomNumberGenerator] = None,
                 ) -> None:
        r"""Constructor

        :param perturbation_steps: Number of perturbation steps.
        :param perturbation_size: Size of the region to flip.
        A size of 1 corresponds to single pixels, whereas a higher number to patches of size nxn.
        :param verbose: Whether to print debug messages.
        :param perturb_mode: Perturbation technique to decide how to replace flipped values.

        :param ran_num_gen: Random number generator to use. Only available with PerturbModes.RANDOM.
        """
        # Ensure perturbation size is a valid number.
        if perturbation_size < 1:
            raise ValueError(
                f'Perturbation size ({perturbation_size}) must be greater than or equal to 1.')

            # Ensure ran_num_gen is only specified when the perturbation technique is random.
        if perturb_mode != PerturbModes.RANDOM and ran_num_gen:
            raise ValueError(
                f"""Argument ran_num_gen is only available with PerturbModes.RANDOM and \
                    should not be passed otherwise.
Selected perturbation mode: {perturb_mode}""")

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
        self.flipped_input_nchw: torch.Tensor

        # Store min. and max. values of tensor in case of perturbation mode random.
        # Otherwise these remain uninitialized.
        self._low: Optional[float]
        self._high: Optional[float]

        # Iterator over flip masks in each perturbation step.
        self._flip_mask_generator: Generator[torch.Tensor, None, None]

        # Store (accumulative) masks applied to flip the input together in a single mask.
        self.acc_flip_mask_nhw: torch.Tensor

        # Number of times to flip pixels/patches
        self.perturbation_steps: int = perturbation_steps

        # Size of the region to flip
        self.perturbation_size: int = perturbation_size

        # Name of the perturbation technique
        self.perturb_mode: str = perturb_mode

        if self.perturb_mode == PerturbModes.RANDOM:
            self.ran_num_gen: RandomNumberGenerator = ran_num_gen or UniformRNG()

        # List to store updated classification scores after each perturbation step.
        self.class_prediction_scores_n: torch.Tensor

    @timer
    def __call__(self,
                 input_nchw: torch.Tensor,
                 relevance_scores_nchw: torch.Tensor,
                 forward_pass: Callable[[torch.Tensor], torch.Tensor],
                 should_loop: bool = True
                 ) -> Optional[Generator[Tuple[torch.Tensor, float], None, None]]:
        r"""Run pixel-flipping algorithm.

        :param input_nchw: Input to be explained.
        :param relevance_scores_nchw: Relevance scores.
        :param forward_pass: Classifier function to measure accuracy change
            in pixel-flipping iterations.
            It should take an input tensor in NCHW format and return the class prediction
            score of a single class.
            The index in the output layer of the class to be explained should be hardcoded.
            It shouldn't make use of .max() to retrieve the class index because—as a result
            of perturbation—the class index with maximum score can change.
            Example:
                # In this example 483 is the class index of the class to be explained,
                # which corresponds to the castle class in ImageNet.
                # The function assumes that all input images in the batch have the same
                # class index 483—i.e., are castle images.
                forward_pass: Callable[
                    [torch.Tensor],
                    torch.Tensor] = lambda input: lrp_instance.model(input)[:][483].item()
        :param should_loop: Whether to loop over the generator or not.

        :yields: Tuple of flipped input and updated classification score
        after one perturbation step.

        :returns: None if should_loop is True, otherwise a generator.
        """
        sanity_checks.ensure_nchw_format(input_nchw)
        sanity_checks.verify_square_input(input_nchw, relevance_scores_nchw)
        sanity_checks.verify_batch_size(input_nchw, relevance_scores_nchw)
        sanity_checks.ensure_non_overlapping_patches_possible(input_nchw,
                                                              self.perturbation_size)
        sanity_checks.verify_perturbation_args(input_nchw=input_nchw,
                                               perturbation_size=self.perturbation_size,
                                               perturbation_steps=self.perturbation_steps)

        self._batch_size: int = utils.get_batch_size(input_nchw=input_nchw)

        # Store input for comparison at the end.
        self.original_input_nchw: torch.Tensor = input_nchw.detach().clone()

        self.relevance_scores_nchw: torch.Tensor = relevance_scores_nchw.detach().clone()

        # Initialize accumulative mask to False.
        # Each mask used to flip the input will be stored in this tensor with logical OR.
        #
        # Input has dimensions (batch_size, channels, height, width).
        # Accumulative mask has dimensions (batch_size, height, width).
        #   .sum(dim=1) is used to reduce the number of channels to 1.
        #   I.e., to convert from (batch_size, channels, height, width) to
        #   (batch_size, height, width).
        self.acc_flip_mask_nhw: torch.Tensor = torch.zeros(
            *input_nchw.shape,
            dtype=torch.bool).sum(dim=1)

        pixel_flipping_generator: Generator[
            Tuple[torch.Tensor, float], None, None] = self._generator(input_nchw,
                                                                      relevance_scores_nchw,
                                                                      forward_pass)

        # Toggle to return generator or loop automatically over it.
        if not should_loop:
            return pixel_flipping_generator

        return utils.loop(pixel_flipping_generator)

    def _generator(self,
                   input_nchw: torch.Tensor,
                   relevance_scores_nchw: torch.Tensor,
                   forward_pass: Callable[[torch.Tensor], torch.Tensor]
                   ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        r"""Generator to flip pixels of input according to the relevance scores.

        :param input_nchw: Input to be explained.
        :param relevance_scores_nchw: Relevance scores.

        :param forward_pass: Classifier function to measure accuracy change
                             in pixel-flipping iterations.

        :yields: Tuple of flipped input and updated classification score
        after one perturbation step.

        Generators are annotated in the format: Generator[YieldType, SendType, ReturnType],
        therefore, SendType and ReturnType are set to None above.
        Source: https://docs.python.org/3/library/typing.html
        """

        if self.perturb_mode == PerturbModes.RANDOM:
            # TODO: Add support for custom low and high bounds (random number generation).
            # Infer (min. and max.) bounds of input for random number generation
            self._low: Optional[float] = float(input_nchw.min())
            self._high: Optional[float] = float(input_nchw.max())

        # Deep copy input to avoid in-place modifications.
        # Detach input from computational graph to avoid computing gradient for the
        # pixel-flipping operations.
        flipped_input_nchw: torch.Tensor = input_nchw.detach().clone()
        flipped_input_nchw.requires_grad = False

        # Tensor with class prediction scores has shape (batch_size, perturbation_steps+1).
        # First perturbation step is the original class prediction score without perturbation.
        self.class_prediction_scores_n: torch.Tensor = torch.zeros(
            (self._batch_size, self.perturbation_steps+1), dtype=torch.float)

        # Get initial classification score.
        self._measure_class_prediction_score(
            forward_pass, flipped_input_nchw, perturbation_step=0)

        self.logger.debug(
            f'Initial classification score {self.class_prediction_scores_n}')

        # Contains N one-dimensional lists of relevance scores with m elements. Shape (N, m).
        self._flip_mask_generator: Generator[
            torch.Tensor, None, None
        ] = sort.flip_mask_generator(relevance_scores_nchw,
                                     self.perturbation_size)

        for perturbation_step in range(self.perturbation_steps):
            # Perturbation step 0 is the original input.
            # Shift perturbation step by one to start from 1.
            shifted_perturbation_step: int = perturbation_step + 1

            self.logger.debug(f'Step {shifted_perturbation_step}')

            # Run a perturbation step.
            flipped_input_nchw, last_class_prediction_score = self._flip(
                forward_pass, flipped_input_nchw, shifted_perturbation_step)

            # Store flipped input for comparison at the end with the original input.
            self.flipped_input_nchw: torch.Tensor = flipped_input_nchw

            yield flipped_input_nchw, last_class_prediction_score

    def _measure_class_prediction_score(self,
                                        forward_pass: Callable[[torch.Tensor], torch.Tensor],
                                        flipped_input_nchw: torch.Tensor,
                                        perturbation_step: int) -> None:
        r"""Measure class prediction score of input using forward pass.

        :param forward_pass: Classifier function to measure accuracy change
                             in pixel-flipping iterations.
        :param flipped_input_nchw: Input to be explained.
        :param perturbation_step: Current perturbation step.
        """
        score = forward_pass(flipped_input_nchw).detach()
        self.class_prediction_scores_n[:, perturbation_step] = score

        self.logger.debug(
            f'Classification score: {self.class_prediction_scores_n[:, perturbation_step]}')

    @timer
    def _flip(self,
              forward_pass: Callable[[torch.Tensor], torch.Tensor],
              flipped_input_nchw: torch.Tensor,
              perturbation_step: int
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Execute a single iteration of the Region Perturbation algorithm.

        :param forward_pass: Classifier function to measure accuracy change in
                             pixel-flipping iterations.
        :param flipped_input_nchw: Input to be explained.
        :param perturbation_step: Current perturbation step.

        :returns: Tuple of flipped input and updated classification score
        """
        # Mask with region selected for flipping.
        mask_n1hw: torch.Tensor = next(self._flip_mask_generator)

        # Flip pixels with respective perturbation technique
        if self.perturb_mode == PerturbModes.RANDOM:
            flip_random(input_nchw=flipped_input_nchw,
                        mask_n1hw=mask_n1hw,
                        perturbation_size=self.perturbation_size,
                        ran_num_gen=self.ran_num_gen,
                        low=self._low,
                        high=self._high,
                        logger=self.logger)

        elif self.perturb_mode == PerturbModes.INPAINTING:
            flipped_input_nchw = norm.denorm_img_pxls(
                norm.ImageNetNorm.inverse_normalize(flipped_input_nchw))

            flipped_input_nchw = flip_inpainting(input_nchw=flipped_input_nchw.int(),
                                                 mask_n1hw=mask_n1hw,
                                                 logger=self.logger).float()

            flipped_input_nchw = norm.ImageNetNorm.normalize(
                norm.norm_img_pxls(flipped_input_nchw))

        else:
            raise NotImplementedError(
                f'Perturbation mode \'{self.perturb_mode}\' not implemented yet.')

        # Loop for debugging purposes only.
        for n in range(self._batch_size):
            # Mask with all pixels previously flipped.
            old_acc_flip_mask_hw: torch.Tensor = self.acc_flip_mask_nhw[n]
            # Mask with pixels flipped only in this perturbation step.
            mask_hw: torch.Tensor = mask_n1hw.squeeze()[n]
            # Mask with all pixels flipped as of this perturbation step (including previous flips).
            new_acc_flip_mask_hw: torch.Tensor = torch.logical_or(
                old_acc_flip_mask_hw, mask_hw)

            # Store number of flipped pixels before this perturbation step.
            flipped_pixel_count: int = old_acc_flip_mask_hw.count_nonzero().item()

            # Calculate delta of flipped pixels:
            #   I.e., total number of flipped pixels in this perturbation step
            #   minus the count of already flipped pixels.
            flipped_pixel_count = new_acc_flip_mask_hw.count_nonzero().item() - \
                flipped_pixel_count

            self.logger.info(
                f'Batch image {n}: Flipped {flipped_pixel_count} pixels.')

        # Squeeze mask to empty channel dimension and convert from (N, 1, H, W) to (N, H, W).
        mask_nhw: torch.Tensor = mask_n1hw.squeeze()
        self.acc_flip_mask_nhw: torch.Tensor = torch.logical_or(
            self.acc_flip_mask_nhw, mask_nhw)

        # Measure classification accuracy change
        self._measure_class_prediction_score(forward_pass,
                                             flipped_input_nchw,
                                             perturbation_step)

        # return flipped_input_nchw, self.class_prediction_scores_n[..., -1:]
        return flipped_input_nchw, self.class_prediction_scores_n

    def plot_class_prediction_scores(self,
                                     title: str = '',
                                     xlabel: str = '',
                                     ylabel: str = '',
                                     show_plot: bool = True) -> None:
        r"""Plot the updated prediction scores throughout the perturbation steps of
        the pixel-flipping algorithm to visualize the accuracy of the explanation.

        :param title: Title of the plot.
        :param xlabel: Label of the x-axis.
        :param ylabel: Label of the y-axis.
        :param show_plot: If True, show the plot.

        :raises ValueError: If class prediction scores are empty.
        """

        # Error handling
        # Check if class prediction scores are empty—i.e., initialized to zeros.
        if (self.class_prediction_scores_n == 0).all():
            raise ValueError(
                'No class prediction scores to plot. Please run pixel-flipping first.')

        mean_class_prediction_scores_n: torch.Tensor = torch.mean(
            self.class_prediction_scores_n, dim=0)

        for _, class_prediction_scores in enumerate(self.class_prediction_scores_n):
            plt.plot(class_prediction_scores,
                     color='lightgrey')

        auc: float = area_under_the_curve(
            mean_class_prediction_scores_n.detach().numpy()
        )
        aopc: float = area_over_the_pertubation_curve(
            mean_class_prediction_scores_n.detach().numpy()
        )
        plt.plot(mean_class_prediction_scores_n,
                 label='Mean',
                 linewidth=5,
                 alpha=0.9,
                 color='black')

        x: List[int] = range(len(mean_class_prediction_scores_n))
        plt.fill_between(x=x,
                         y1=mean_class_prediction_scores_n,
                         y2=plt.ylim()[1],
                         facecolor='thistle',
                         label=f'AOPC={aopc}',
                         alpha=0.2)
        plt.fill_between(x=x,
                         y1=mean_class_prediction_scores_n,
                         facecolor='wheat',
                         label=f'AUC={auc}',
                         alpha=0.2)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(0, tight=True)
        # Add padding for better alignment of (sup)title
        # Source: https://stackoverflow.com/a/45161551
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.legend(loc='upper right')
        if show_plot:
            plt.show()

    def plot_image_comparison(self, show_plot: bool = True) -> None:
        r"""Plot the original input and the perturbed input to visualize the changes.

        :param show_plot: If True, show the plot.
        """
        plot.plot_image_comparison(batch_size=self._batch_size,
                                   original_input_nchw=self.original_input_nchw,
                                   flipped_input_nchw=self.flipped_input_nchw,
                                   relevance_scores_nchw=self.relevance_scores_nchw,
                                   acc_flip_mask_nhw=self.acc_flip_mask_nhw,
                                   show_plot=show_plot)
