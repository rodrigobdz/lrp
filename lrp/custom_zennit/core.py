import torch


def stabilize(input, epsilon=1e-6):
    '''Stabilize input for safe division.'''
    return input + epsilon


# Source: https://github.com/chr5tphr/zennit/blob/6251a9e17aa31c3381799de92f92b1d259b392b2/zennit/core.py#L45-L90
# Without require_params and context manager


def mod_params(module, modifier, param_keys=None):
    '''Modify parameter attributes (all by default) of a module.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Module of which to modify parameters. If `requires_params` is `True`, it must have all elements given in
        `param_keys` as attributes (attributes are allowed to be `None`, in which case they are ignored).
    modifier: function
        A function used to modify parameter attributes. If `param_keys` is empty, this is not used.
    param_keys: list[str], optional
        A list of parameters that shall be modified. If `None` (default), all parameters are modified (which may be
        none). If `[]`, no parameters are modified and `modifier` is ignored.
    '''
    if param_keys is None:
        param_keys = [name for name,
                      _ in module.named_parameters(recurse=False)]

    for key in param_keys:
        param = getattr(module, key)
        if param is not None:
            # Changed order of param and name in lambda func modifier call
            setattr(module, key, torch.nn.Parameter(modifier(key, param.data)))
