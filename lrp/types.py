r'''Type definitions for type checking
'''


__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'rodrigobdz@tu-berlin.de'
__status__ = 'Development'

from .rules import LrpEpsilonRule, LrpGammaRule, LrpZeroRule, LrpZBoxRule


class SubclassMeta(type):
    '''Meta class to bundle multiple subclasses.'''
    def __instancecheck__(cls, inst):
        """Implement isinstance(inst, cls) as subclasscheck."""
        return cls.__subclasscheck__(type(inst))

    def __subclasscheck__(cls, sub):
        """Implement issubclass(sub, cls) with by considering additional __subclass__ members."""
        candidates = cls.__dict__.get("__subclass__", tuple())
        return type.__subclasscheck__(cls, sub) or issubclass(sub, candidates)


class LrpRuleType(metaclass=SubclassMeta):
    '''Abstract base class that describes available LRP rules.'''
    __subclass__ = (
        LrpEpsilonRule,
        LrpGammaRule,
        LrpZeroRule,
        LrpZBoxRule,
    )
