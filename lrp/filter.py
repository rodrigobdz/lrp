r'''Filter layers before rules are mapped.
'''

__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


import torch
from lrp.zennit.core import collect_leaves
from typing import List, Dict, Tuple, Callable, Optional


class LayerFilter:
    r'''Filter the layers by layer index and type.
    '''

    def __init__(self, model: torch.nn.Module) -> None:
        r'''Build a filter for the given model.

        :param model: Model to filter layers from
        '''
        # Collect all layers by name
        self.name_lookup: Dict[str, torch.nn.Module] = {module: name for name, module in model.named_modules()}
        # Model layers in sequential order
        self.layers: List[Tuple[int, torch.nn.Module]] = list(enumerate(collect_leaves(model)))

    def set_target_types(self, target_types: Tuple[type]) -> None:
        r'''Set the types of layers to filter.

        :param target_types: Types of layers to filter
        '''
        self.target_types = target_types

    def __call__(self, cond: Callable[[int], bool], target_types: Optional[Tuple[type]] = None) -> List[str]:
        r'''Filter layers by layer index and type.

        :param cond: Function to filter layers by layer index
        :param target_types: Types of layers to filter

        :return: List of layer names
        '''
        # Use default target types if not specified
        allowed_types = target_types or self.target_types

        return [self.name_lookup[module] for n, module in self.layers if cond(n) and isinstance(module, allowed_types)]

