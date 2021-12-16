r'''Composite Layer-wise Relevance propagation using rules defined by layer
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


import torch
from .rules import LrpEpsilonRule, LrpGammaRule, LrpZeroRule, LrpZBoxRule
from .image import heatmap


class LRP:
    r'''Compute relevance propagation using Layer-wise Relevance Propagation algorithm'''

    def __init__(self, model: torch.nn.Module) -> None:
        r'''Prepare model for LRP computation

        :param model: Model to be explained
        '''
        self.model = model
        self.model.eval()
        self.R = None

    @staticmethod
    def convert_single_layer(
        i: int,
        layer: torch.nn.Module,
        low: torch.Tensor = None,
        high: torch.Tensor = None,
    ):
        r'''Convert a single layer to LRP layer

        :param i: Layer index
        :param layer: Layer to be converted
        :param low: Lowest pixel values for input
        :param high: Highest pixel values for input

        :returns: Layer whose autograd computation corresponds to LRP
        '''
        if isinstance(layer, torch.nn.Linear):
            i += 31

        if not (
            isinstance(layer, torch.nn.Conv2d) or isinstance(
                layer, torch.nn.Linear)
        ):
            return layer

        # Case: pixel layer, apply ZBox rule
        if i == 0:
            print(f'Layer {i}: LRP ZBox')
            return LrpZBoxRule(layer, low, high)

        # Case: bottom layers, apply LRP-Gamma with gamma = 0.25
        if i <= 16:
            print(f'Layer {i}: LRP-Gamma with gamma = 0.25')
            return LrpGammaRule(layer, gamma=0.25)

        # Case: middle layers, apply LRP-Epsilon with epsilon = 0.25. Alt. 1e-6
        if 17 <= i <= 30:
            print(f'Layer {i}: LRP-Epsilon with epsilon = 0.25')
            return LrpEpsilonRule(layer, epsilon=0.25)

        # Case: top layers, apply LRP-0
        if i >= 31:
            print(f'Layer {i}: LRP-Zero')
            return LrpZeroRule(layer)

    def convert_layers(self, low: torch.Tensor, high: torch.Tensor):
        r'''Convert layers in model to LRP layers

        :param low: Lowest pixel values for input
        :param high: Highest pixel values for input
        '''
        self.model.features[0] = LRP.convert_single_layer(
            0, self.model.features[0], low=low, high=high
        )

        for i, layer in enumerate(self.model.features):
            if i == 0:
                continue
            self.model.features[i] = LRP.convert_single_layer(i, layer)

        self.model.avgpool = LrpZeroRule(self.model.avgpool)

        for i, layer in enumerate(self.model.classifier):
            self.model.classifier[i] = self.convert_single_layer(i, layer)

    def relevance(self, X: torch.Tensor) -> torch.Tensor:
        r'''Compute relevance for input X by applying Gradient*Input

        :param X: Input to be explained
        :returns: Relevance for input X
        '''
        # Prepare to compute input gradient
        # Reset gradient
        self.model.zero_grad()
        X.requires_grad = True

        # Compute explanation
        # Stores value of gradient in X.grad
        # [0].max() retrieves the maximum activation/relevance in the first layer
        # = 483 for castle
        self.model.forward(X)[0].max().backward()

        # Retrieve gradients from first layer
        first_layer = self.model.features[0]
        low = first_layer.low
        high = first_layer.high

        print(f'Absolute sum of attribution {X.grad.abs().sum().item()}')

        # Calculate gradients
        c1, c2, c3 = X.grad, low.grad, high.grad

        # Calculate relevance
        self.R = X * c1 + low * c2 + high * c3
        return self.R

    def visualize(self, width: int = 4, height: int = 4) -> torch.Tensor:
        r'''Create heatmap of relevance

        :param width: Width of heatmap
        :param height: Height of heatmap
        '''
        heatmap(self.R[0].sum(dim=0).detach().numpy(), width, height)
