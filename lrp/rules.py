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
from .custom_zennit.core import mod_params, stabilize


class _LrpRule(torch.nn.Module):
    def __init__(
        self, layer: torch.nn.Module, param_modifiers: List[Tuple[str, Callable]]
    ) -> None:
        super().__init__()
        self.layer = layer

        for layer_name, param_mod in param_modifiers:
            # Create a new layer that applies the parameter modifier
            setattr(self, layer_name, copy.deepcopy(layer))
            copy_layer = getattr(self, layer_name)
            mod_params(copy_layer, param_mod)

    def forward_mod_gradient(
        self, z: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        # output is result of original forward function
        return z * (output / stabilize(z)).detach()


class _LrpGenericRule(_LrpRule):
    def __init__(self, layer: torch.nn.Module, epsilon: float, gamma: float) -> None:
        param_modifiers = [
            ('copy_layer', lambda _, param: param + gamma * param.clamp(min=0))
        ]
        self.epsilon = epsilon
        super().__init__(layer, param_modifiers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.layer.forward(X)
        z: torch.Tensor = self.epsilon + self.copy_layer.forward(X)

        return super().forward_mod_gradient(z, output)


class LrpZeroRule(_LrpGenericRule):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__(layer, epsilon=0, gamma=0)


class LrpEpsilonRule(_LrpGenericRule):
    def __init__(self, layer: torch.nn.Module, epsilon: float) -> None:
        super().__init__(layer, epsilon, gamma=0)


class LrpGammaRule(_LrpGenericRule):
    def __init__(self, layer: torch.nn.Module, gamma: float) -> None:
        super().__init__(layer, 0, gamma)


class LrpZBoxRule(_LrpRule):
    def __init__(
        self, layer: torch.nn.Module, low: torch.Tensor, high: torch.Tensor
    ) -> None:
        param_modifiers = [
            ('low_layer', lambda _, param: param.clamp(min=0)),
            ('high_layer', lambda _, param: param.clamp(max=0)),
        ]
        super().__init__(layer, param_modifiers)

        self.low = low
        self.high = high

        self.low.requires_grad = True
        self.high.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.layer.forward(X)

        z_low = self.low_layer.forward(self.low)
        z_high = self.high_layer.forward(self.high)
        z = output - z_low - z_high

        return super().forward_mod_gradient(z, output)
