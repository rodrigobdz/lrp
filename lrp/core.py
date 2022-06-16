r"""Composite Layer-wise Relevance Propagation using rules defined by layer."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import copy
from typing import Dict, List, Optional, Tuple, Union

import torch

import pf.sanity_checks
import pf.utils
from pf.decorators import timer

from . import builtin, plot, rules

DEVICE: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


class LRP:
    r"""Compute relevance propagation using Layer-wise Relevance Propagation algorithm."""

    def __init__(self, model: torch.nn.Module) -> None:
        r"""Prepare model for LRP computation.

        :param model: Model to be explained
        """
        self.model = copy.deepcopy(model)
        self.model.eval().to(device=DEVICE)
        self.rule_layer_map: List[Tuple[List[str], rules.LrpRule,
                                  Dict[str, Union[torch.Tensor, float]]]] = []
        self.input_nchw: Optional[torch.Tensor] = None
        self.label_idx_n: Optional[torch.Tensor] = None
        self.relevance_scores_nchw: Optional[torch.Tensor] = None
        self.explained_class_indices: Optional[torch.Tensor] = None

    def convert_layers(self, rule_layer_map:
                       List[
                           Tuple[
                               # Layer names
                               List[str],

                               # LRP rule to apply
                               rules.LrpRule,

                               # Parameters for rule
                               Dict[str, Union[torch.Tensor, float]]
                           ]
                       ]) -> None:
        r"""Add LRP support to layers according to given mapping.

        :param rule_layer_map: List of tuples containing layer names, LRP rule and parameters
        """
        self.rule_layer_map = rule_layer_map

        for name, layer in self.model.named_modules():
            # Check which rule to apply
            rule = self.mapping(name)

            # No rule to apply
            if rule is None:
                continue

            # Unwrap tuple containing rule and its parameters
            rule_class, rule_kwargs = rule

            # Initialize rule class
            lrp_layer = rule_class(layer, **rule_kwargs)

            # Apply rule to named layer
            builtin.rsetattr(self.model, name, lrp_layer)

    def mapping(self, name: str) -> Optional[Tuple[rules.LrpRule,
                                                   Dict[str, Union[torch.Tensor, float]]]]:
        r"""Get LRP rule and parameters for layer with given name.

        :param name: Layer name
        :return: LRP rule and parameters or None if no rule is found
        """
        for layer_names, rule, rule_kwargs in self.rule_layer_map:
            # Apply rule only to layers included in mapping
            if name in layer_names:
                return rule, rule_kwargs

        return None

    @timer
    def relevance(self,
                  input_nchw: torch.Tensor,
                  label_idx_n: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Compute relevance for input_nchw by applying Gradient*Input.

        Source: "Algorithm 8 LRP implementation based on forward hooks" in
        "Toward Interpretable Machine Learning: Transparent Deep Neural Networks and Beyond"

        :param input_nchw: Input to be explained
        :param label_idx_n: Labels to be explained corresponding to input_nchw

            The labels are used in case a specific class should be explained instead of
            the class corresponding to the neuron with the highest activation score.
            The tensor should have one dimension with n labels as int, where n is batch size.
            The label is the index of the class to be explained.
            Example for a single-element batch:

            # Explain first class
            label_idx_n = torch.tensor([0])

        :returns: Relevance for input_nchw
        """
        pf.sanity_checks.ensure_nchw_format(input_nchw)
        pf.sanity_checks.verify_square_input(input_nchw)

        self.input_nchw = input_nchw.to(device=DEVICE)

        # Prepare to compute input gradient
        # Reset gradient
        self.model.zero_grad()
        input_nchw.requires_grad = True

        # ZBoxrule-specific parameters
        low: Union[torch.Tensor, int]
        high: Union[torch.Tensor, int]
        c2_low_grad: Union[torch.Tensor, int]
        c3_high_grad: Union[torch.Tensor, int]

        # Set default values
        low = high = c2_low_grad = c3_high_grad = 0

        # Vars to retrieve gradients from first layer
        first_layer: torch.nn.Module = self.model.features[0]

        if isinstance(first_layer, rules.LrpZBoxRule):
            # Access high and low copy layers in first layer.
            low = first_layer.low.to(device=DEVICE)
            high = first_layer.high.to(device=DEVICE)

            # Reset stored gradients.
            low.grad = None
            high.grad = None

        # Reset stored gradients
        input_nchw.grad = None

        # Compute explanation by storing value of gradient in input_nchw.grad.
        # Only the predicted class is propagated backwards.
        #
        # 1. Compute forward pass
        forward_pass: torch.Tensor = self.model(input_nchw).to(device=DEVICE)

        # 2. Get index of classes to be explained
        idx: torch.Tensor
        if label_idx_n is not None:
            # Compute classes passed as argument explicitly
            idx = label_idx_n.to(device=DEVICE)

            # Save index of classes to be explained as instance variable
            self.label_idx_n = label_idx_n
        else:
            # Get index maximum activation in the output layer (index of the predicted class)
            idx = forward_pass.max(dim=1).indices.to(device=DEVICE)

        # 3. Create new tensor where elements are tuples (i, idx[i]) with i: counter.
        # Tensor i looks like this: [0, 1, ..., len(idx)]
        i: torch.Tensor = torch.arange(end=len(idx),
                                       device=DEVICE)

        # Stacked tensor looks like this: [(i, idx[i]), (i+1, idx[i+1]), ...],
        # where i is the counter and idx[i] is the index of
        # the maximum activation in the output layer.

        # Indices of selected classes are particularly useful for Pixel-Flipping algorithm.
        self.explained_class_indices = torch.stack((i, idx), dim=1)

        # 4. One-hot encoding for the predicted class in each sample.
        # This is a mask where the predicted class is True and the rest is False.
        batch_size: int = pf.utils.get_batch_size(input_nchw=input_nchw)
        number_of_classes: int = forward_pass.shape[1]
        # Init zeros tensor for one-hot encoding
        gradient: torch.Tensor = torch.zeros(batch_size,
                                             number_of_classes,
                                             dtype=torch.bool,
                                             device=DEVICE)
        # Set the predicted class to True
        #
        # The following statement should be equivalent to:
        #   gradient[*self.explained_class_indices.T] = True
        gradient[self.explained_class_indices[:, 0],
                 self.explained_class_indices[:, 1]] = True

        # 5. Compute gradient of output layer for the predicted class of each sample.
        forward_pass.backward(gradient=gradient)

        if isinstance(first_layer, rules.LrpZBoxRule):
            # Compute gradients
            c2_low_grad = low.grad.to(device=DEVICE)
            c3_high_grad = high.grad.to(device=DEVICE)

        # Compute input gradient
        c1_input_grad = input_nchw.grad.to(device=DEVICE)

        # Compute relevance
        self.relevance_scores_nchw = (input_nchw * c1_input_grad +
                                      low * c2_low_grad +
                                      high * c3_high_grad)
        return self.relevance_scores_nchw.detach().to(device=DEVICE)

    @staticmethod
    def heatmap(relevance_scores_nchw: torch.Tensor,
                width: int = 4,
                height: int = 4) -> None:
        r"""Create heatmap of relevance.

        :param relevance_scores_nchw: Relevance tensor with N3HW format
        :param width: Width of heatmap
        :param height: Height of heatmap
        """
        pf.sanity_checks.ensure_nchw_format(relevance_scores_nchw)
        # Convert each heatmap from 3-channel to 1-channel.
        # Channel dimension is now omitted.
        r_nhw = relevance_scores_nchw.sum(dim=1)

        # Loop over relevance scores for each image in batch
        for r_hw in r_nhw:
            # Use Tensor.cpu() to copy the tensor to host memory before converting to numpy().
            plot.heatmap(r_hw.cpu().detach().numpy(), width, height)
