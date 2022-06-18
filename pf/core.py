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
from .objectives.constants import PixelFlippingObjectives
from .perturbation_modes.constants import PerturbModes
from .perturbation_modes.inpainting.flip import flip_inpainting
from .perturbation_modes.random.flip import flip_random
from .perturbation_modes.random.random_number_generators import (
    RandomNumberGenerator, UniformRNG)

DEVICE: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


class PixelFlipping:
    r"""Pixel-Flipping Algorithm."""

    def __init__(self,
                 perturbation_steps: int = 29,
                 perturbation_size: int = 8,
                 verbose: bool = False,
                 perturb_mode: str = PerturbModes.INPAINTING,
                 sort_objective: str = PixelFlippingObjectives.MoRF,
                 ran_num_gen: Optional[RandomNumberGenerator] = None,
                 ) -> None:
        r"""Initialize PixelFlipping class.

        :param perturbation_steps: Number of perturbation steps.
        :param perturbation_size: Size of the region to flip.
                                    A size of 1 corresponds to single pixels,
                                    whereas a higher number to patches of size nxn.
        :param verbose: Whether to print debug messages.
        :param perturb_mode: Perturbation technique to decide how to replace flipped values.
        :param sort_objective: Objective used to sort the order of the relevances for perturbation.
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

        # Store perturbation objective
        self.sort_objective: str = sort_objective

        # Store flipped input after perturbation.
        self.flipped_input_nchw: torch.Tensor

        # Store flipped input before last perturbation step (for visualization only).
        self.flipped_input_nchw_before_last_step: Optional[torch.Tensor] = None

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
        self.original_input_nchw: torch.Tensor = input_nchw.detach().clone().to(device=DEVICE)
        self.relevance_scores_nchw: torch.Tensor = relevance_scores_nchw.detach().clone()
        self.relevance_scores_nchw.to(device=DEVICE)

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
        self.acc_flip_mask_nhw: torch.Tensor = torch.zeros(*input_nchw.shape,
                                                           dtype=torch.bool,
                                                           device=DEVICE).sum(dim=1)

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
        self.class_prediction_scores_n: torch.Tensor = torch.zeros((self.batch_size,
                                                                    self.perturbation_steps+1),
                                                                   dtype=torch.float,
                                                                   device=DEVICE)

        # Get initial classification score.
        self._measure_class_prediction_score(
            forward_pass, flipped_input_nchw, perturbation_step=0)

        self.logger.debug("Initial classification score %s",
                          self.class_prediction_scores_n)

        # Contains N one-dimensional lists of relevance scores with m elements. Shape (N, m).
        self._flip_mask_generator: Generator[
            torch.Tensor, None, None
        ] = sort.flip_mask_generator(relevance_scores_nchw=relevance_scores_nchw,
                                     perturbation_size=self.perturbation_size,
                                     sort_objective=self.sort_objective)

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

        The last step flips all elements to a fixed color (gray).

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
                max_num_squared_possible = int(sqrt(784)) = 28

                base_numbers_to_square = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,
                                           9, 10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24,
                                          25, 26, 27, 28]
                squared_vals = [  0,   1,   4,   9,  16,  25,  36,  49,  64,  81,
                                100, 121, 144, 169, 196, 225, 256, 289, 324, 361,
                                400, 441, 484, 529, 576, 625, 676, 729, 784]
                number_of_flips_per_step_arr = [ 0,  1,  3,  5,  7,  9, 11, 13,
                                                15, 17, 19, 21, 23, 25, 27, 29,
                                                31, 33, 35, 37, 39, 41, 43, 45,
                                                47, 49, 51, 53, 55]

                # Flip all patches in the last step to a fixed color.
                number_of_flips_per_step_arr[-1] = 784

                number_of_flips_per_step_arr[:-1].sum() = 784

            Result:
                {0: 0,
                 1: 1,
                 2: 3,
                 3: 5,
                 4: 7,
                 5: 9,
                 6: 11,
                 7: 13,
                 8: 15,
                 9: 17,
                 10: 19,
                 11: 21,
                 12: 23,
                 13: 25,
                 14: 27,
                 15: 29,
                 16: 31,
                 17: 33,
                 18: 35,
                 19: 37,
                 20: 39,
                 21: 41,
                 22: 43,
                 23: 45,
                 24: 47,
                 25: 49,
                 26: 51,
                 27: 53,
                 28: 55,
                 29: 784}
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

        # Append an artificial step where all patches are flipped to color gray
        number_of_flips_per_step_arr = numpy.append(arr=number_of_flips_per_step_arr,
                                                    values=total_num_patches)

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

        if perturbation_step != self.max_perturbation_steps:
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
        else:
            mask_n1hw: torch.Tensor = next(self._flip_mask_generator)

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

        # Flip pixels according to perturbation mode selected if current perturbation step is not
        # the last one.
        if perturbation_step != self.max_perturbation_steps:
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
                # Denormalize image values for inpaiting
                flipped_input_nchw = norm.denorm_img_pxls(
                    norm.ImageNetNorm.inverse_normalize(flipped_input_nchw))

                # Paint all previously inpainted regions and the currrent one altogether.
                # To paint each region only once, pass mask_n1hw as argument to the flip_* functions.
                acc_mask_n1hw: torch.Tensor = self.acc_flip_mask_nhw.unsqueeze(
                    1)
                flipped_input_nchw = flip_inpainting(input_nchw=flipped_input_nchw.int(),
                                                     mask_n1hw=acc_mask_n1hw,
                                                     logger=self.logger).float()

                # Re-normalize flipped input after perturbation.
                flipped_input_nchw = norm.ImageNetNorm.normalize(
                    norm.norm_img_pxls(flipped_input_nchw))

            else:
                raise NotImplementedError(
                    f'Perturbation mode \'{self.perturb_mode}\' not implemented yet.')

        # Last perturbation step.
        # Flip all pixels to constant value equivalent to gray color.
        else:
            # Last perturbation step.
            # Store flipped input before it becomes a gray image (def. last perturbation step).
            self.flipped_input_nchw_before_last_step = flipped_input_nchw.detach().clone()

            mask_nchw = mask_n1hw.expand(flipped_input_nchw.shape)
            midpoint: float = (self._low + self._high)/2
            flipped_input_nchw[mask_nchw] = midpoint

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
        indices: torch.Tensor = torch.arange(end=self.current_perturbation_step,
                                             device=DEVICE)
        class_prediction_scores_sliced: torch.Tensor = self.class_prediction_scores_n.index_select(
            dim=1,
            index=indices).to(device=DEVICE)
        mean_class_prediction_scores_n: torch.Tensor = torch.mean(class_prediction_scores_sliced,
                                                                  dim=0).to(device=DEVICE)

        return class_prediction_scores_sliced, mean_class_prediction_scores_n

    @timer
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

        # Use Tensor.cpu() to copy the tensor to host memory before converting to numpy().
        auc: float = area_under_the_curve(
            mean_class_prediction_scores_n.cpu().detach().numpy())

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
            plt.plot(class_prediction_scores.cpu(),
                     color='lightgrey')

        # AUC can only be computed for at least two points.
        if self.current_perturbation_step > 1:
            auc: float = self.calculate_auc_score()

            plt.plot(mean_class_prediction_scores_n.cpu(),
                     label='Mean',
                     linewidth=5,
                     alpha=0.9,
                     color='black')

            x_values: range = range(len(mean_class_prediction_scores_n))
            plt.fill_between(x=x_values,
                             y1=mean_class_prediction_scores_n.cpu(),
                             facecolor='wheat',
                             label=f'AUC={auc}',
                             alpha=0.2)

        title: str = f"""Pixel-Flipping
        Perturbation steps: {self.current_perturbation_step-1}
        Perturbation size: {self.perturbation_size}x{self.perturbation_size}
        Percentage flipped: {self._calculate_percentage_flipped()}%
        Perturbation mode: {self.perturb_mode}
        Sorting objective: {self.sort_objective}
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
        # Default value for flipped input to plot.
        flipped_input_nchw: torch.Tensor = self.flipped_input_nchw
        plot_flipped_input_before_last_step: bool = False

        # If flipped input before last perturbation step is available, plot it.
        if torch.is_tensor(self.flipped_input_nchw_before_last_step):
            plot_flipped_input_before_last_step = True
            flipped_input_nchw = self.flipped_input_nchw_before_last_step

        plot.plot_image_comparison(batch_size=self.batch_size,
                                   original_input_nchw=self.original_input_nchw.cpu(),
                                   flipped_input_nchw=flipped_input_nchw.cpu(),
                                   before_last_step=plot_flipped_input_before_last_step,
                                   relevance_scores_nchw=self.relevance_scores_nchw.cpu(),
                                   acc_flip_mask_nhw=self.acc_flip_mask_nhw.cpu(),
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
