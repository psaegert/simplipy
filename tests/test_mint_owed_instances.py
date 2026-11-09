"""The mint fits the instances of a source's constants that the judge owes a witness at.

At a special constant where the source is finite on a set of positive measure, the judge owes the
rule a witness; the mint fits the target there as at a drawn instance. In f64, whose values the
fit reads, a target with no witness is refused by the mint, not left to the judge loop. A mode
that reads the reals is not read by a fit in float64: there an unfitted instance binds nothing,
and the judge decides it.
"""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.verify import _contract, verify_rule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


def _scan(engine: SimpliPyEngine, source: str, target: str, mode: str):
    data = _contract.mine_matrix(42, 512, 2, False)
    lib = engine._core.build_candidate_library([target.split()], ['x0', 'x1'],
                                               np.asarray(data).flatten().tolist(),
                                               data.shape[0], rule_mode=mode)
    pick, steps = engine._core.find_rule_lib_traced(source.split(), lib, None, rtol=1e-11,
                                                    atol=1e-12, challenges=8, retries=8,
                                                    rule_mode=mode)
    return pick, [verdict for _, _, verdict in steps]


@pytest.mark.parametrize('source, target', [
    # at C = 0 float64's left side is 0 wherever x < 0 (1/inf), and no finite C' gives C'^x = 0
    ('inv pow <constant> x0', 'pow <constant> x0'),
])
def test_f64_refuses_where_an_owed_special_constant_has_no_witness(engine, source, target):
    """The f64 mint refuses the pair at the special constant, as the judge refuses it in f64;
    it used to skip the unfitted instance and leave the refusal to the judge loop."""
    pick, verdicts = _scan(engine, source, target, 'default')
    assert pick is None and verdicts == ['special_constants'], verdicts
    assert verify_rule(source.split(), target.split())['basis']['f64'] == 'refused'


def test_f64s_fit_at_a_special_constant_reads_float64(engine) -> None:
    """At C = 0 float64's `exp(x + log(C))` is 0 everywhere, and `0*exp(x)` is NaN where exp
    overflows, where the reals' value is 0 (the range standard). The f64 mint's fit at the
    special constant reads float64's NaN and refuses the pair, within the mine's tolerances;
    strict holds it by its second reading (true in the reals, float64's two sides of one kind)."""
    source, target = 'exp + x0 log <constant>', '* <constant> exp x0'
    pick, verdicts = _scan(engine, source, target, 'default')
    assert pick is None and verdicts == ['special_constants'], verdicts
    d = verify_rule(source.split(), target.split())
    assert d['basis']['f64'] == 'evidence' and d['realised_kinds'] is True, d


@pytest.mark.parametrize('source, target', [
    ('inv pow <constant> x0', 'pow <constant> x0'),
    ('exp + x0 log <constant>', '* <constant> exp x0'),
])
def test_permissive_leaves_an_unfitted_instance_to_the_judge(engine, source, target):
    """In the reals `1/0^x` has no value (nothing is owed at C = 0), and `0*exp(838)` is 0 (the
    witness 0 holds): float64's fit decides neither, and the judge holds both in permissive."""
    pick, verdicts = _scan(engine, source, target, 'permissive')
    assert 'special_constants' not in verdicts and 'certified' in verdicts, verdicts
    assert verify_rule(source.split(), target.split())['basis']['permissive'] in (
        'proved', 'evidence')


def test_a_witness_at_every_owed_special_constant_certifies(engine) -> None:
    """`x/C -> C'*x` has the witness 1/C at every special constant the source is finite at."""
    pick, verdicts = _scan(engine, '/ x0 <constant>', '* <constant> x0', 'default')
    assert 'special_constants' not in verdicts and 'certified' in verdicts, verdicts
