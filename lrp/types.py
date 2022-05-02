r"""Type definitions for type checking."""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


from .rules import LrpEpsilonRule, LrpGammaRule, LrpZBoxRule, LrpZeroRule
from .zennit.types import SubclassMeta


class LrpRuleType(metaclass=SubclassMeta):
    """Abstract base class that describes available LRP rules."""
    __subclass__ = (
        LrpEpsilonRule,
        LrpGammaRule,
        LrpZeroRule,
        LrpZBoxRule,
    )
