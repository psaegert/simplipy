"""A ground rule in f64 holds only where its two sides are the same double, the sign of a zero
included, with no range standard: every fully numeric rewrite is bit-exact in f64 (F8). The judge
(its ground point route and its battery's reading) and the mint decide it alike. Permissive keeps
such rules, its ground values being the table's real values."""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine, _core
from simplipy.verify import _contract, _exact, verify_rule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize('lhs, rhs, double', [
    ('exp sin np.pi', '1', 1.0000000000000002),
    ('tan atan 1', '1', 0.9999999999999999),
    ('tan atan2 1 -1', '(-1)', -1.0000000000000002),
])
def test_a_ground_rule_a_unit_off_the_double_is_refused_in_f64(lhs, rhs, double) -> None:
    """The left side is one unit in the last place from the right side's double: F6's 8 units
    admitted it, F8 admits none. In the reals the two are equal, so permissive keeps it."""
    assert float(_contract.d_eval(_contract.parse(lhs.split()), {})) == double
    d = verify_rule(lhs.split(), rhs.split())
    assert d['realised'] is False and d['tier'] == 'permissive', d
    assert d['basis']['f64'] == 'refused' and d['basis']['permissive'] != 'refused', d
    assert d['exact']['f64'] == 'refuted', d


def test_a_ground_rule_on_the_double_holds_in_f64() -> None:
    d = verify_rule('cos np.pi'.split(), ['(-1)'])
    assert d['realised'] is True and d['basis']['f64'] == 'proved', d


def test_the_sign_of_a_zero_is_part_of_the_double() -> None:
    """`sin(-0.0)` is -0.0 in float64: `-> 0` changes the double."""
    d = verify_rule('sin -0.0'.split(), ['0'])
    assert d['realised'] is False, d
    assert verify_rule('sin -0.0'.split(), ['-0.0'])['realised'] is True


def test_a_ground_rule_has_no_range_standard() -> None:
    """`exp(800)/exp(800)` is inf/inf, NaN, in float64; its true value 1 is no excuse at one
    point, as a fully numeric expression takes the realizations' value."""
    d = verify_rule('/ exp 800 exp 800'.split(), ['1'])
    assert d['realised'] is False, d


def test_the_ground_route_reads_the_same_double() -> None:
    tl, tr = (_contract.parse(s.split()) for s in ('exp sin np.pi', '1'))
    out = _exact.ground(tl, tr, _contract._real_spec, lambda t, rng: _contract.d_eval(t, {}, rng),
                        _core.table_f64_agree, _core.table_f64_same)
    assert out['f64'] == 'refuted' and out['reals'] == 'proved', out
    assert _core.table_f64_same(float('nan'), float('nan'))
    assert not _core.table_f64_same(0.0, -0.0)
    assert not _core.table_f64_same(1.0, 1.0000000000000002)


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


def test_the_f64_mint_refuses_a_ground_candidate_off_the_double(engine) -> None:
    """The mint decides as the judge does: `exp(sin(pi))` against the candidate `1` fails on its
    rows in f64 (it was certified at 1 unit in the last place)."""
    data = _contract.mine_matrix(42, 512, 2, False)
    lib = engine._core.build_candidate_library([['1']], ['x0', 'x1'],
                                               np.asarray(data).flatten().tolist(),
                                               data.shape[0], rule_mode='default')
    pick, steps = engine._core.find_rule_lib_traced('exp sin np.pi'.split(), lib, None,
                                                    rtol=1e-11, atol=1e-12, challenges=8,
                                                    retries=8, rule_mode='default')
    assert pick is None and [v for _, _, v in steps] == ['rows'], steps
