r'''LRP rules
'''


__author__ = 'Rodrigo Bermudez Schettino'
__credits__ = ['Rodrigo Bermudez Schettino']
__maintainer__ = 'Rodrigo Bermudez Schettino'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


import copy
import torch
from typing import Callable, List, Tuple
from .custom_zennit import mod_params, stabilize


class _LrpRule(torch.nn.Module):
    r'''Base class for LRP rules'''

    def __init__(
        self, layer: torch.nn.Module, param_modifiers: List[Tuple[str, Callable]]
    ) -> None:
        r'''
        :param layer: Layer to be modified
        :param param_modifiers: Tuples of (layer_name, modifier)
                A new layer will be created for each layer_name with modifier applied
        '''
        super().__init__()
        self.layer = layer

        for layer_name, param_mod in param_modifiers:
            # Create new layer
            setattr(self, layer_name, copy.deepcopy(layer))

            # Convenience variable to store new layer
            copy_layer = getattr(self, layer_name)

            # Modify the parameters of the new layer
            mod_params(copy_layer, param_mod)

    def forward_mod_gradient(
        self, z: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        r'''Modifies the gradient of the layer while keeping the output unchanged.
        The resulting gradient returns the relevance scores of LRP when invoking automatic differentiation.

        :param z: Input to the layer
        :return: result of original forward function
        '''
        return z * (output / stabilize(z)).detach()


class _LrpGenericRule(_LrpRule):
    r'''Generic LRP rule

    Source: 10.2 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    '''

    def __init__(self, layer: torch.nn.Module, epsilon: float, gamma: float) -> None:
        r'''Define parameter modifiers for the layer

        :param layer: Layer to be modified
        :param epsilon: Epsilon value for LRP rule with same name
        :param gamma: Gamma value for LRP rule with same name
        '''
        param_modifiers = [
            ('copy_layer', lambda _, param: param + gamma * param.clamp(min=0))
        ]
        self.epsilon = epsilon
        super().__init__(layer, param_modifiers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r'''Forward passes on layer and copy layer

        :param X: Input to the layer
        :returns: Output of the layer with modified gradient
        '''
        output: torch.Tensor = self.layer.forward(X)
        z: torch.Tensor = self.epsilon + self.copy_layer.forward(X)

        return super().forward_mod_gradient(z, output)


class LrpZeroRule(_LrpGenericRule):
    r'''LRP-0 rule
    Source: 10.1 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    '''

    def __init__(self, layer: torch.nn.Module) -> None:
        r'''
        :param layer: Layer to be modified
        '''
        super().__init__(layer, epsilon=0, gamma=0)


class LrpEpsilonRule(_LrpGenericRule):
    r'''LRP-ε rule
    Source: 10.1 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    '''

    def __init__(self, layer: torch.nn.Module, epsilon: float) -> None:
        r'''
        :param layer: Layer to be modified
        '''
        super().__init__(layer, epsilon, gamma=0)


class LrpGammaRule(_LrpGenericRule):
    r'''LRP-γ rule
    Source: 10.1 in https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
    '''

    def __init__(self, layer: torch.nn.Module, gamma: float) -> None:
        r'''
        :param layer: Layer to be modified
        '''
        super().__init__(layer, epsilon=0, gamma=gamma)


class LrpZBoxRule(_LrpRule):
    r'''LRP-Z-Box rule
    Source: Algorithm 7 in Appendix B section A. in https://arxiv.org/abs/2003.07631v1
    '''

    def __init__(
        self, layer: torch.nn.Module, low: torch.Tensor, high: torch.Tensor
    ) -> None:
        r'''Define parameter modifiers for the layer

        Excerpt from the paper:
        "The functions f1+ and f1− are forward passes on copies of the first layer whose parameters
        have been processed by the functions max(0, ·) and min(0, ·) respectively."
        '''
        param_modifiers = [
            ('low_layer', lambda _, param: param.clamp(min=0)),
            ('high_layer', lambda _, param: param.clamp(max=0)),
        ]

        super().__init__(layer, param_modifiers)

        self.low = low
        self.high = high

        # Enable gradient computation
        self.low.requires_grad = True
        self.high.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r'''Forward passes on layer and low and high layers
        '''
        # Forward passes on all layers
        output: torch.Tensor = self.layer.forward(X)
        z_low: torch.Tensor = self.low_layer.forward(self.low)
        z_high: torch.Tensor = self.high_layer.forward(self.high)

        # Compute modified forward pass
        z: torch.Tensor = output - z_low - z_high

        return super().forward_mod_gradient(z, output)
