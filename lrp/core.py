r'''Composite Layer-wise Relevance Propagation using rules defined by layer
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


from typing import Union, Dict, List, Tuple, Optional
import torch
import copy
from . import rules, builtin, plot


class LRP:
    r'''Compute relevance propagation using Layer-wise Relevance Propagation algorithm'''

    def __init__(self, model: torch.nn.Module) -> None:
        r'''Prepare model for LRP computation

        :param model: Model to be explained
        '''
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.name_map: List[Tuple[List[str], rules.LrpRule,
                                  Dict[str, Union[torch.Tensor, float]]]] = []
        self.R: Optional[torch.Tensor] = None

    def convert_layers(self, name_map:
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
        r'''Add LRP support to layers according to given mapping

        :param name_map: List of tuples containing layer names, LRP rule and parameters
        '''

        self.name_map = name_map

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

    def mapping(self, name: str) -> Optional[Tuple[rules.LrpRule, Dict[str, Union[torch.Tensor, float]]]]:
        r'''Get LRP rule and parameters for layer with given name

        :param name: Layer name
        :return: LRP rule and parameters or None if no rule is found
        '''
        for layer_names, rule, rule_kwargs in self.name_map:
            # Apply rule only to layers included in mapping
            if name in layer_names:
                return rule, rule_kwargs

        return None

    def relevance(self, X: torch.Tensor) -> torch.Tensor:
        r'''Compute relevance for input X by applying Gradient*Input

        Source: "Algorithm 8 LRP implementation based on forward hooks" in
        "Toward Interpretable Machine Learning: Transparent Deep Neural Networks and Beyond"

        :param X: Input to be explained
        :returns: Relevance for input X
        '''
        # Prepare to compute input gradient
        # Reset gradient
        self.model.zero_grad()
        X.requires_grad = True

        # Vars to retrieve gradients from first layer
        first_layer: torch.nn.Module = self.model.features[0]
        low: torch.Tensor = first_layer.low
        high: torch.Tensor = first_layer.high

        # Reset stored gradients
        X.grad = None
        low.grad = None
        high.grad = None

        # Compute explanation
        # Stores value of gradient in X.grad
        # [0].max() retrieves the maximum activation/relevance in the first layer
        self.model.forward(X)[0].max().backward()

        # Calculate gradients
        c1, c2, c3 = X.grad, low.grad, high.grad

        # Calculate relevance
        self.R = X * c1 + low * c2 + high * c3
        return self.R

    @staticmethod
    def heatmap(R: torch.Tensor, width: int = 4, height: int = 4) -> None:
        r'''Create heatmap of relevance

        :param R: Relevance tensor
        :param width: Width of heatmap
        :param height: Height of heatmap
        '''
        return plot.heatmap(R[0].sum(dim=0).detach().numpy(), width, height)
