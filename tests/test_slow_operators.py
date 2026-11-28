"""THE SLOW-OPERATOR POLICY (the specification's 4.12): an operator's spec may declare its
`family` and that it is `slow`; a rule brings a slow operator into its right side only where its
left side holds an operator of the same family. The licence names the policy (`slow_operator`), as
the mint's scan and licence apply it."""
import pytest

from simplipy import SimpliPyEngine
from simplipy.verify import POLICIES, policy_refusals


def _spec(name: str, arity: int, **extra) -> dict:
    return {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
            'precedence': 2, 'commutative': False, **extra}


def _engine(declared: bool = True) -> SimpliPyEngine:
    slow = {'family': 'bessel', 'slow': True} if declared else {}
    operators = {
        '+': {'realization': 'simplipy.operators.add', 'alias': [], 'arity': 2, 'precedence': 1,
              'commutative': True},
        'log': _spec('log', 1),
        'exp': _spec('exp', 1),
        'besselj': _spec('besselj', 2, **slow),
        'besselk': _spec('besselk', 2, **slow),
    }
    return SimpliPyEngine(operators=operators, rules=[])


def test_the_policy_is_named() -> None:
    assert 'slow_operator' in POLICIES


@pytest.mark.parametrize('lhs, rhs, refused', [
    ('log _0', 'besselj 0 _0', True),
    ('log _0', '+ _0 besselk 1 _0', True),
    ('besselk 0 _0', 'besselj 0 _0', False),
    ('exp besselj 0 _0', 'besselj 0 _0', False),
    ('besselj 0 _0', 'exp _0', False),
])
def test_a_slow_operator_comes_only_with_its_family(lhs, rhs, refused) -> None:
    got = policy_refusals(lhs.split(), rhs.split(), 'strict', _engine())
    assert ('slow_operator' in got) is refused, got


def test_an_undeclared_operator_is_not_slow() -> None:
    assert 'slow_operator' not in policy_refusals(['log', '_0'], ['besselj', '0', '_0'], 'strict',
                                                  _engine(declared=False))


@pytest.mark.parametrize('extra', [{'family': 3}, {'slow': 'yes'}])
def test_the_declaration_is_checked(extra) -> None:
    with pytest.raises(ValueError, match='family|slow'):
        SimpliPyEngine(operators={'besselj': _spec('besselj', 2, **extra)}, rules=[])
