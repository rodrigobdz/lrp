r'''Imported methods from zennit framework with customizations.
'''


__author__ = 'Rodrigo Bermudez Schettino'
__credits__ = ['Rodrigo Bermudez Schettino']
__maintainer__ = 'Rodrigo Bermudez Schettino'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'


from typing import Callable, List
import torch


def stabilize(dividend: torch.Tensor, epsilon: float = 1e-6):
    r'''Ensure dividend is not zero, thus, guarantee safe division.

    :param dividend:
    :param epsilon: Epsilon value to add to dividend to avoid division by zero

    :returns: Non-zero value
    '''
    return dividend + epsilon


def mod_params(module: torch.nn.Module, modifier: Callable, param_keys: List[str] = None):
    r'''Modify parameter attributes (all by default) of a module.

    Source: https://github.com/chr5tphr/zennit/blob/6251a9e17aa31c3381799de92f92b1d259b392b2/zennit/core.py#L45-L90
    It omits require_params and context_manager.

    :param module: Module whose parmeters should be modified
        If requires_params is True, it must have all elements given in param_keys as attributes (attributes are allowed to be None, in which case they are ignored).
    :param modifier: Function that modifies the parameter
        A function used to modify parameter attributes. If param_keys is empty, this is not used.
    :param param_keys: List of parameter keys to modify
        If None (default), all parameters are modified (which may be none). If [], no parameters are modified and modifier is ignored.
    '''
    if param_keys is None:
        param_keys = [name for name,
                      _ in module.named_parameters(recurse=False)]

    for key in param_keys:
        param = getattr(module, key)
        if param is not None:
            # Changed order of param and name in lambda func modifier call
            setattr(module, key, torch.nn.Parameter(modifier(key, param.data)))
