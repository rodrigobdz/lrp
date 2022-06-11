r"""LRP rules
"""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import copy
from typing import Callable, List, Tuple

import torch

from .zennit import core as zennit_core

DEVICE: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


class LrpRule(torch.nn.Module):
    r"""Base class for LRP rules."""

    def __init__(
        self, layer: torch.nn.Module, param_modifiers: List[Tuple[str, Callable]]
    ) -> None:
        r"""Initialize an LRP rule.

        :param layer: Layer to be modified
        :param param_modifiers: Tuples of (layer_name, modifier)
                A new layer will be created for each layer_name with modifier applied
        """
        super().__init__()
        self.layer: torch.nn.Module = layer

        for layer_name, param_mod in param_modifiers:
            # Create new layer
            setattr(self, layer_name, copy.deepcopy(layer))

            # Convenience variable to store new layer
            copy_layer = getattr(self, layer_name)

            # Modify the parameters of the new layer
            zennit_core.mod_params(copy_layer, param_mod)

    def forward_mod_gradient(
        self, z: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        r"""Modify the gradient of the layer while keeping the output unchanged.

        The resulting gradient returns the relevance scores of LRP when invoking automatic differentiation.

        :param z: Input to the layer
        :return: result of original forward function
        """
        # Heuristic used in lrp-tutorial stabilizes both z occurrences (multiplier and denominator):
        # return zennit_core.stabilize(z) * (output / zennit_core.stabilize(z)).detach()

        # Without any heuristic to enforce 0/0 = 0, NaN may occur:
        # return z * (output / z).detach()

        return z * (output / zennit_core.stabilize(z)).detach()


class _LrpGenericRule(LrpRule):
    r"""Define generic LRP rule.

    Source: 10.2 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    """

    def __init__(self, layer: torch.nn.Module, epsilon: float, gamma: float) -> None:
        r"""Define parameter modifiers for the layer.

        :param layer: Layer to be modified
        :param epsilon: Epsilon value for LRP rule with same name
        :param gamma: Gamma value for LRP rule with same name
        """
        param_modifiers: List[Tuple[str, Callable]] = [
            ('copy_layer', lambda _, param: param + gamma * param.clamp(min=0))
        ]
        self.epsilon: float = epsilon
        super().__init__(layer, param_modifiers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Forward passes on layer and copy layer.

        :param X: Input to the layer
        :returns: Output of the layer with modified gradient
        """
        output: torch.Tensor = self.layer.forward(X)
        z: torch.Tensor = self.epsilon + self.copy_layer.forward(X)

        return super().forward_mod_gradient(z, output)


class LrpZeroRule(_LrpGenericRule):
    r"""LRP-0 rule.

    Source: 10.1 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        r"""
        :param layer: Layer to be modified
        """
        super().__init__(layer, epsilon=0, gamma=0)


class LrpEpsilonRule(_LrpGenericRule):
    r"""LRP-ε rule.

    Source: 10.1 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    """

    def __init__(self, layer: torch.nn.Module, epsilon: float) -> None:
        r"""
        :param layer: Layer to be modified
        :param epsilon: Epsilon value for LRP rule with same name
        """
        super().__init__(layer, epsilon, gamma=0)


class LrpGammaRule(_LrpGenericRule):
    r"""LRP-γ rule.

    Source: 10.1 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    """

    def __init__(self, layer: torch.nn.Module, gamma: float) -> None:
        r"""
        :param layer: Layer to be modified
        :param gamma: Gamma value for LRP rule with same name
        """
        super().__init__(layer, epsilon=0, gamma=gamma)


class LrpZBoxRule(LrpRule):
    r"""LRP-Z-Box rule.

    Source: Algorithm 7 in Appendix B section A. in https://arxiv.org/abs/2003.07631v1
    """

    def __init__(
        self, layer: torch.nn.Module, low: torch.Tensor, high: torch.Tensor
    ) -> None:
        r"""Define parameter modifiers for the layer.

        Excerpt from the paper:
        "The functions f1+ and f1− are forward passes on copies of the first layer whose parameters
        have been processed by the functions max(0, ·) and min(0, ·) respectively."

        :param layer: Layer to be modified
        :param low: Tensor with lowest pixel values in the image
        :param high: Tensors with highest pixel values in the image
        """
        param_modifiers = [
            ('low_layer', lambda _, param: param.clamp(min=0)),
            ('high_layer', lambda _, param: param.clamp(max=0)),
        ]

        super().__init__(layer, param_modifiers)

        self.low: torch.Tensor = low.to(device=DEVICE)
        self.high: torch.Tensor = high.to(device=DEVICE)

        # Enable gradient computation
        self.low.requires_grad = True
        self.high.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Forward passes on layer and low and high layers.

        :param X: Input to the layer
        """
        output: torch.Tensor = self.layer.forward(X)
        z_low: torch.Tensor = self.low_layer.forward(self.low)
        z_high: torch.Tensor = self.high_layer.forward(self.high)

        # Compute modified forward pass
        z: torch.Tensor = output - z_low - z_high

        return super().forward_mod_gradient(z, output)
