"""Permissive's reciprocal join takes a factor into a reciprocal sum, `f * A^-1 -> (A/f)^-1`, only
where `f` has no zero: at a zero of `f` where `A` is finite and nonzero the left side is 0 and
`(A/0)^-1` has no value, a value lost, which permissive's criterion refuses at a point. So
`2*tan(x0)/(1 - tan(x0)^2)` does not become `2/(1/tan(x0) - tan(x0))`, and every answer below is
a rewrite permissive holds."""
import os

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.verify import verify_rule

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')


@pytest.fixture(scope='module')
def bare() -> SimpliPyEngine:
    return SimpliPyEngine(operators=yaml.safe_load(open(ACJ2_4))['operators'], rules=[],
                          rules_permissive=[])


@pytest.mark.parametrize('source, lost', [
    ('/ * 2 tan x0 - 1 pow tan x0 2', '/ 2 - inv tan x0 tan x0'),
    ('/ x0 - 1 pow x0 2', 'inv - inv x0 x0'),
    ('/ x0 - x0 x1', 'inv - 1 / x1 x0'),
])
def test_a_factor_with_a_zero_stays_outside_its_reciprocal_sum(bare, source, lost) -> None:
    answer = bare.simplify(source.split(), mode='permissive')
    assert answer != lost.split(), answer
    if answer != source.split():
        assert verify_rule(source.split(), answer)['permissive'] is True, answer


def test_a_factor_without_a_zero_still_joins(bare) -> None:
    source = '/ exp x0 - x0 x1'.split()
    answer = bare.simplify(source, mode='permissive')
    assert verify_rule(source, answer)['permissive'] is True, answer
