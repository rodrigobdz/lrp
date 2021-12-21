r'''Type definitions for type checking
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'

from .rules import LrpEpsilonRule, LrpGammaRule, LrpZeroRule, LrpZBoxRule
from .zennit.types import SubclassMeta


class LrpRuleType(metaclass=SubclassMeta):
    '''Abstract base class that describes available LRP rules.'''
    __subclass__ = (
        LrpEpsilonRule,
        LrpGammaRule,
        LrpZeroRule,
        LrpZBoxRule,
    )
