"""Permissive's own steps without a definedness check (like terms cancel, a zero factor absorbs the
product) are the engine's: its constructors perform them on every instance, with no rule loaded,
each named once in the core's list, and a rule stating that a zero factor absorbs the product is
engine-owned, which the mine does not mint and the load drops. Strict keeps the product where
float64 gives it no value.
"""
import os

import pytest
import yaml

from simplipy import SimpliPyEngine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')) as f:
        ops = yaml.safe_load(f)['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


@pytest.mark.parametrize('expression, strict', [
    ('* 0 log x2', '* 0 log x2'),
    ('* 0 rootn x2 2', '* 0 rootn x2 2'),
    ('* 0 pow x2 x2', '* 0 pow x2 x2'),
    ('* 0 asin x2', '* 0 asin x2'),
    ('/ 0 pow x2 0.5', '/ 0 pow x2 / 1 2'),
    ('* 0 acosh x2', '* 0 acosh x2'),
    ('* 0 inv x2', '0'),
    ('* - x1 x1 log x2', '* 0 log x2'),
    ('- tan inv + x1 x2 tan inv + x1 x2', '- tan inv + x1 x2 tan inv + x1 x2'),
    ('/ * x1 + x2 inv + x3 1 + x2 inv + x3 1', '/ * x1 + x2 inv + x3 1 + x2 inv + x3 1'),
])
def test_permissive_takes_its_own_steps_without_rules(engine: SimpliPyEngine, expression: str,
                                                      strict: str) -> None:
    want = {'/ * x1 + x2 inv + x3 1 + x2 inv + x3 1': 'x1'}.get(expression, '0')
    assert ' '.join(engine.simplify(expression.split(), mode='permissive', effort=0)) == want
    assert ' '.join(engine.simplify(expression.split(), mode='strict', effort=0)) == strict


#: a rule stating that a zero factor absorbs the product is engine-owned in permissive; strict keeps
#: the product float64 gives no value, and a rule stating a like-term cancellation stays a rule (the
#: walk takes it where the constructors' merge does not)
@pytest.mark.parametrize('lhs, rhs, permissive_owned', [
    ('* 0 !0', '0', True), ('/ 0 $0', '0', True),
    ('- !0 !0', '0', False), ('/ $0 $0', '1', False),
])
def test_a_rule_stating_the_zero_factor_is_engine_owned(engine: SimpliPyEngine, lhs: str, rhs: str,
                                                        permissive_owned: bool) -> None:
    law = engine._core.law_instance
    owned = law(lhs.split(), rhs.split(), rule_mode='permissive') == 'engine-owned-law'
    assert owned is permissive_owned
    assert law(lhs.split(), rhs.split(), rule_mode='strict') != 'engine-owned-law'


def test_the_zero_factor_is_a_named_step_beside_like_terms() -> None:
    from simplipy import _core
    names = _core.permissive_simplifications()
    assert names.index('a zero factor absorbs the product') == names.index('like terms cancel') + 1
    schemas = dict(_core.permissive_schemas())
    assert schemas['a zero factor absorbs the product'] == [(['*', '0', '_0'], ['0'])]
    assert (['*', '0', '_0'], ['0']) not in schemas['like terms cancel']
