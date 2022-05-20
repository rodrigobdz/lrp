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
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy
import torch
from matplotlib import pyplot as plt

from lrp import norm

from . import plot, sanity_checks, utils
from .decorators import timer
from .metrics import area_under_the_curve
from .objectives import sort
from .perturbation_modes.constants import PerturbModes
from .perturbation_modes.inpainting.flip import flip_inpainting
from .perturbation_modes.random.flip import flip_random
from .perturbation_modes.random.random_number_generators import (
    RandomNumberGenerator, UniformRNG)


class PixelFlipping:
    r"""Pixel-Flipping Algorithm."""

    def __init__(self,
                 perturbation_steps: int = 28,
                 perturbation_size: int = 8,
                 verbose: bool = False,
                 perturb_mode: str = PerturbModes.INPAINTING,
                 ran_num_gen: Optional[RandomNumberGenerator] = None,
                 ) -> None:
        r"""Initialize PixelFlipping class.

        :param perturbation_steps: Number of perturbation steps.
        :param perturbation_size: Size of the region to flip.
                                    A size of 1 corresponds to single pixels,
                                    whereas a higher number to patches of size nxn.
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
        if perturb_mode not in (PerturbModes.INPAINTING, PerturbModes.RANDOM):
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

        self.batch_size: int = utils.get_batch_size(input_nchw=input_nchw)
        # Store original input for comparison at the end.
        self.original_input_nchw: torch.Tensor = input_nchw.detach().clone()
        self.relevance_scores_nchw: torch.Tensor = relevance_scores_nchw.detach().clone()

        self.number_of_flips_per_step_dict: Dict[int,
                                                 int] = self._define_number_of_flips_per_step_dict()
        self.max_perturbation_steps: int = len(
            self.number_of_flips_per_step_dict) - 1

        sanity_checks.verify_perturbation_args(perturbation_steps=self.perturbation_steps,
                                               max_perturbation_steps=self.max_perturbation_steps)

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
        r"""Create generator to flip pixels of input according to the relevance scores.

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
            (self.batch_size, self.perturbation_steps+1), dtype=torch.float)

        # Get initial classification score.
        self._measure_class_prediction_score(
            forward_pass, flipped_input_nchw, perturbation_step=0)

        self.logger.debug("Initial classification score %s",
                          self.class_prediction_scores_n)

        # Contains N one-dimensional lists of relevance scores with m elements. Shape (N, m).
        self._flip_mask_generator: Generator[
            torch.Tensor, None, None
        ] = sort.flip_mask_generator(relevance_scores_nchw,
                                     self.perturbation_size)

        # Perturbation step 0 is the original input.
        # Shift perturbation step by one to start from 1.
        for perturbation_step in range(1, self.perturbation_steps+1):
            self.current_perturbation_step: int = perturbation_step + 1
            self.logger.debug("Step %s",
                              perturbation_step)

            # Run a perturbation step.
            flipped_input_nchw, last_class_prediction_score = self._flip(forward_pass,
                                                                         flipped_input_nchw,
                                                                         perturbation_step)

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
        scores: torch.Tensor = forward_pass(flipped_input_nchw).detach()
        self.class_prediction_scores_n[:, perturbation_step] = scores

        self.logger.debug("Classification score: %s",
                          self.class_prediction_scores_n[:, perturbation_step])

    def _define_number_of_flips_per_step_dict(self) -> Dict[int, int]:
        r"""Define number of flips per step.

        At the beginning few regions are flipped, then the number of regions flipped in each step
        progressively increases using a formula which involves the squares of perturbation steps.

        :param input_nchw: Input to be explained.
        :param perturbation_size: Number of pixels to perturb per step.

        :returns: Number of flips per step as a dictionary with
                    keys: perturbation step, value: number of flips.

        Example:
            Setup:
                perturbation_size = 8
                input_nchw.shape = (N, C, 224, 224)

            Intermediate calculations:
                width = 224
                num_patches_per_img_1d = 224//8 = 28

                total_num_patches = 28 * 28 = 784
                max_power_of_two_possible = int(log2(784)) = 9

                power_of_two_exponents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                power_of_two_vals = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                number_of_flips_per_step_arr =  [1, 1, 2, 4, 8, 16, 32, 64, 128, 256]

                num_patches_to_flip_with_power_of_two = 512
                rest_patches_count = 272

                number_of_flips_per_step_arr = [1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 272]
                number_of_flips_per_step_arr.sum() = 784

            Result:
                {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 272}
        """
        _, width = utils.get_height_width(self.original_input_nchw)

        # Get number of patches per image in a dimension (width or height).
        # I.e., how many patches of size perturbation_size can fit horizontally or vertically
        # in the new grid of patches.
        num_patches_per_img_1d: int = width // self.perturbation_size

        # Get total number of patches of size perturbation_size in image.
        total_num_patches: int = num_patches_per_img_1d * num_patches_per_img_1d

        # Calculate squared values to increasingly flip patches of pixels at each perturbation step.
        #
        # The maximum number which squared is equal to the total
        # number of patches is num_patches_per_img_1d.
        #
        # Add one because the range starts at 0.
        base_numbers_to_square: numpy.ndarray = numpy.arange(
            num_patches_per_img_1d + 1)
        squared_vals: numpy.ndarray = base_numbers_to_square**2

        # Calculate the differences between consecutive elements of an array and prepend
        # an element with value 1 at the beginning. The goal is to "slow start" and then
        # progressively flip more elements as step count increases.
        number_of_flips_per_step_arr: numpy.ndarray = numpy.ediff1d(
            squared_vals)

        # Sanity check. The total number of patches flipped in all steps altogether should
        # be equal to the total number of patches in image.
        if total_num_patches != number_of_flips_per_step_arr.sum():
            raise ValueError(f"""Total number of patches {total_num_patches} is not equal to the sum
of number of patches flipped in all steps {number_of_flips_per_step_arr.sum()}.""")

        # Prepend an element with value 0 at the beginning.
        # This element represents the original input with no perturbations.
        number_of_flips_per_step_arr = numpy.insert(arr=number_of_flips_per_step_arr,
                                                    obj=0,
                                                    values=0,
                                                    axis=0)

        # Convert array of number of patches to flip per step to dictionary with keys:
        # perturbation step, value: number of flips.
        number_of_flips_per_step_dict: Dict[int, int] = dict(
            enumerate(number_of_flips_per_step_arr)
        )

        return number_of_flips_per_step_dict

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
        number_of_flips_in_current_step: int = self.number_of_flips_per_step_dict[
            perturbation_step]

        # Mask to store regions to flip in current perturbation step.
        mask_n1hw: torch.Tensor
        for multi_flip_index in range(number_of_flips_in_current_step):
            # Mask with region selected for flipping.
            next_mask_n1hw: torch.Tensor = next(self._flip_mask_generator)

            # Set value of base mask for first iteration.
            if multi_flip_index == 0:
                mask_n1hw = next_mask_n1hw

            # Merge multiple masks into one for a single perturbation step.
            mask_n1hw = torch.logical_or(mask_n1hw, next_mask_n1hw)

        # Loop for debugging purposes only.
        for batch_index in range(self.batch_size):
            # Mask with all pixels previously flipped.
            old_acc_flip_mask_hw: torch.Tensor = self.acc_flip_mask_nhw[batch_index]
            # Mask with pixels flipped only in this perturbation step.
            mask_hw: torch.Tensor = mask_n1hw.squeeze()[batch_index]
            # Mask with all pixels flipped as of this perturbation step (including previous flips).
            new_acc_flip_mask_hw: torch.Tensor = torch.logical_or(
                old_acc_flip_mask_hw, mask_hw)

            # Store number of flipped pixels before this perturbation step.
            flipped_pixel_count: int = int(
                old_acc_flip_mask_hw.count_nonzero().item()
            )

            # Calculate delta of flipped pixels:
            #   I.e., total number of flipped pixels in this perturbation step
            #   minus the count of already flipped pixels.
            flipped_pixel_count = int(
                new_acc_flip_mask_hw.count_nonzero().item() - flipped_pixel_count
            )

            self.logger.info("Batch image %s: Flipped %s pixels.",
                             batch_index,
                             flipped_pixel_count)

        # Squeeze mask to empty channel dimension and convert from (N, 1, H, W) to (N, H, W).
        mask_nhw: torch.Tensor = mask_n1hw.squeeze()
        self.acc_flip_mask_nhw: torch.Tensor = torch.logical_or(
            self.acc_flip_mask_nhw, mask_nhw)

        # Paint all previously inpainted regions and the currrent one altogether.
        # To paint each region only once, pass mask_n1hw as argument to the flip_* functions.
        acc_mask_n1hw: torch.Tensor = self.acc_flip_mask_nhw.unsqueeze(1)

        # Flip pixels with respective perturbation technique
        if self.perturb_mode == PerturbModes.RANDOM:
            flip_random(input_nchw=flipped_input_nchw,
                        mask_n1hw=acc_mask_n1hw,
                        perturbation_size=self.perturbation_size,
                        ran_num_gen=self.ran_num_gen,
                        low=self._low,
                        high=self._high,
                        logger=self.logger)

        elif self.perturb_mode == PerturbModes.INPAINTING:
            flipped_input_nchw = norm.denorm_img_pxls(
                norm.ImageNetNorm.inverse_normalize(flipped_input_nchw))

            flipped_input_nchw = flip_inpainting(input_nchw=flipped_input_nchw.int(),
                                                 mask_n1hw=acc_mask_n1hw,
                                                 logger=self.logger).float()

            flipped_input_nchw = norm.ImageNetNorm.normalize(
                norm.norm_img_pxls(flipped_input_nchw))

        else:
            raise NotImplementedError(
                f'Perturbation mode \'{self.perturb_mode}\' not implemented yet.')

        # Measure classification accuracy change
        self._measure_class_prediction_score(forward_pass,
                                             flipped_input_nchw,
                                             perturbation_step)

        # return flipped_input_nchw, self.class_prediction_scores_n[..., -1:]
        return flipped_input_nchw, self.class_prediction_scores_n

    def _calculate_percentage_flipped(self) -> float:
        r"""Calculate percentage of pixels flipped in all perturbation steps.

        :returns: Percentage of pixels flipped in all perturbation steps.
        """
        # Calculate for first image because all images have the same number of perturbation steps.
        # Select one arbitrary channel to calculate max. number of elements.
        original_input_hw: torch.Tensor = self.original_input_nchw[0][0].squeeze(
        )
        acc_flip_mask_hw: torch.Tensor = self.acc_flip_mask_nhw[0]

        # Calculate max. number of pixels and number of pixels flipped.
        max_num_elem_to_flip: int = original_input_hw.numel()
        num_elem_flipped: int = acc_flip_mask_hw.count_nonzero().item()

        # Calculate percentage with rule of three and round to two decimal places.
        return round((num_elem_flipped*100)/max_num_elem_to_flip, 2)

    def _get_class_prediction_scores_for_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Get class prediction scores for current perturbation step.

        :returns: Class prediction scores for current perturbation step and
                    its mean along the batch dimension.
        """
        indices: torch.Tensor = torch.arange(self.current_perturbation_step)
        class_prediction_scores_sliced: torch.Tensor = self.class_prediction_scores_n.index_select(
            dim=1,
            index=indices)
        mean_class_prediction_scores_n: torch.Tensor = torch.mean(class_prediction_scores_sliced,
                                                                  dim=0)

        return class_prediction_scores_sliced, mean_class_prediction_scores_n

    def calculate_auc_score(self) -> float:
        r"""Calculate AUC score.

        :raises ValueError: If the number of perturbation steps is less or equal to 1.
        :returns: AUC score.
        """
        # AUC can only be computed for at least two points.
        if self.current_perturbation_step <= 1:
            raise ValueError('AUC can only be computed for at least two points.'
                             'Perturbation step should therefore be higher than 1; current:'
                             f'{self.current_perturbation_step}.')

        (_, mean_class_prediction_scores_n) = self._get_class_prediction_scores_for_step()

        auc: float = area_under_the_curve(
            mean_class_prediction_scores_n.detach().numpy())

        return auc

    def plot_class_prediction_scores(self,
                                     show_plot: bool = True) -> None:
        r"""Plot prediction scores throughout perturbation in pixel-flipping algorithm.

        Visualize the accuracy of the explanation.

        :param show_plot: If True, show the plot.

        :raises ValueError: If class prediction scores are empty.
        """
        # Error handling
        # Check if class prediction scores are empty—i.e., initialized to zeros.
        if (self.class_prediction_scores_n == 0).all():
            raise ValueError(
                'No class prediction scores to plot. Please run pixel-flipping first.')

        (class_prediction_scores_sliced,
         mean_class_prediction_scores_n) = self._get_class_prediction_scores_for_step()

        for _, class_prediction_scores in enumerate(class_prediction_scores_sliced):
            plt.plot(class_prediction_scores,
                     color='lightgrey')

        # AUC can only be computed for at least two points.
        if self.current_perturbation_step > 1:
            auc: float = self.calculate_auc_score()

            plt.plot(mean_class_prediction_scores_n,
                     label='Mean',
                     linewidth=5,
                     alpha=0.9,
                     color='black')

            x_values: range = range(len(mean_class_prediction_scores_n))
            plt.fill_between(x=x_values,
                             y1=mean_class_prediction_scores_n,
                             facecolor='wheat',
                             label=f'AUC={auc}',
                             alpha=0.2)

        title: str = f"""Pixel-Flipping
        Perturbation steps: {self.current_perturbation_step-1}
        Perturbation size: {self.perturbation_size}x{self.perturbation_size}
        Percentage flipped: {self._calculate_percentage_flipped()}%
        Perturbation mode: {self.perturb_mode}
        Batch size: {self.batch_size}"""
        plt.title(title)
        plt.xlabel('Perturbation step')
        plt.ylabel('Classification score')
        # Set x ticks to perturbation step indices.
        # Step 0 corresponds to the unperturbed input.
        xticks: range = range(self.current_perturbation_step+1)
        labels: List[Union[str, int]] = ['0\nUnperturbed'] + \
            list(range(1, self.current_perturbation_step+1))
        plt.xticks(ticks=xticks,
                   labels=labels)

        horizontal_margin: float = 0.03
        vertical_margin: float = 0.1
        plt.margins(horizontal_margin, vertical_margin, tight=True)

        # Add padding for better alignment of (sup)title
        # Source: https://stackoverflow.com/a/45161551
        plt.tight_layout(rect=[0, 0, 1, 1])

        plt.legend(loc='upper right')
        if show_plot:
            plt.show()

    def plot_image_comparison(self, show_plot: bool = True) -> None:
        r"""Plot the original input and the perturbed input to visualize the changes.

        :param show_plot: If True, show the plot.
        """
        plot.plot_image_comparison(batch_size=self.batch_size,
                                   original_input_nchw=self.original_input_nchw,
                                   flipped_input_nchw=self.flipped_input_nchw,
                                   relevance_scores_nchw=self.relevance_scores_nchw,
                                   acc_flip_mask_nhw=self.acc_flip_mask_nhw,
                                   perturbation_size=self.perturbation_size,
                                   show_plot=show_plot)

    def plot_number_of_flips_per_step(self) -> None:
        r"""Plot the number of flipped pixels per perturbation step."""
        # Extract values from dictionary
        number_of_flips_per_step_arr: List[int] = list(
            self.number_of_flips_per_step_dict.values())
        plot.plot_number_of_flips_per_step(
            number_of_flips_per_step_arr=number_of_flips_per_step_arr[
                :self.current_perturbation_step],
            max_perturbation_steps=self.max_perturbation_steps)
