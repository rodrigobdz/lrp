r"""Built-in extensions
"""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from typing import Any


def rsetattr(obj: object, name: str, value: Any, sep: str = '.') -> None:
    r"""(Recursively) set attribute of an object with (nested) name dynamically.
    Compared to setattr, it supports nested attributes, which is needed to solve this issue:
    https://discuss.pytorch.org/t/access-replace-layer-using-its-name-string/105925

    Example 1: Replace the first layer of the model with a new one
        rsetattr(model, 'features.0', torch.nn.Conv2d(*args))

        # Results in:
        getattr(model, features)[0] = torch.nn.Conv2d(*args)

    Example 2: For non-nested attributes, setattr is used
        rsetattr(obj, 'foo', 'bar') is equal to:
        setattr(obj, 'foo', 'bar')

    :param obj: Object to be modified
    :param name: Name of the attribute to be modified
    :param value: Value to be assigned to the attribute

    :raises NotImplementedError: If attributes have a nested level deeper than one.
    """
    # Count recursion level in name
    n: int = name.count(sep)

    # If the attribute is not nested, use setattr
    if n == 0:
        return setattr(obj, name, value)

    # Only one level of nesting is supported
    # We could use https://stackoverflow.com/a/31174427
    # TODO: If the attribute is nested, use recursion
    if n > 1:
        raise NotImplementedError(
            'Nested attributes with more than one level of recursion are not supported yet')

    key, index = name.split(".")
    getattr(obj, key)[int(index)] = value
