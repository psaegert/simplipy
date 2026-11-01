"""The interval layer's value sets hold every value their cell reaches, the exceptional ones
included, and decide the measure of an exceptional value's preimage by a certificate in the input
space: `floor` turns the zero of a divisor into an interval of the input."""
import os

import pytest
import yaml

from simplipy import SimpliPyEngine, _core
from simplipy.verify._contract import ARITY, judge_rule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F29 = 'tanh / floor x0 floor x0'.split()


@pytest.fixture(scope='module')
def eng() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


@pytest.mark.parametrize('real', [True, False])
def test_a_zero_divisor_inside_a_hull_keeps_its_value(eng, real) -> None:
    """`floor(x)/floor(x)` over [-2, 2] is 0/0 on all of [0, 1): the enclosure holds NaN beside
    its finite part, in the reals and in float64, and `tanh` of it too."""
    for tokens in (['/', 'floor', 'x0', 'floor', 'x0'], F29):
        has_fin, _, _, nan, _, _ = eng._core.interval_value_set_box(tokens, [-2.0], [2.0],
                                                                    real=real)
        assert has_fin and nan, tokens


def test_a_pole_is_kept_where_its_preimage_may_have_positive_measure(eng) -> None:
    """1/floor(x) over [-1, 1] has its pole's value on all of [0, 1): no value in the reals,
    +inf in float64 (the one zero). 1/x takes it at 0 alone, a null set the divisor's zero-set
    certificate shows, and the value set leaves it out."""
    box = eng._core.interval_value_set_box
    assert box(['inv', 'floor', 'x0'], [-1.0], [1.0], real=True)[3]
    assert box(['inv', 'floor', 'x0'], [-1.0], [1.0], real=False)[1]
    for real in (True, False):
        has_fin, pinf, _, nan, _, _ = box(['inv', 'x0'], [-1.0], [1.0], real=real)
        assert has_fin and not pinf and not nan, real


def test_the_measure_reads_a_null_pole_in_the_input_space() -> None:
    """The judge's measure subdivides the input, and a pole on a null set the divisor's
    certificate shows ends no cell's search: x0*x1/(x1*x2) -> x0/x2 is decided, in three
    variables, and holds in every mode."""
    r = judge_rule('/ * ?0 ?1 * ?1 ?2'.split(), '/ ?0 ?2'.split())
    assert r['verdict'] == 'TOLERATED' and r['tier'] == 'core', r


def test_floor_turns_a_pole_into_an_interval(eng) -> None:
    """The finite-almost-everywhere certificate answers for preimages in the input space:
    `tanh(floor(x)/floor(x))` has no value on [0, 1), so it is not finite almost everywhere,
    and `0*e = 0` does not fire on it in f64 (float64 gives NaN there too)."""
    assert eng._core.interval_finite_ae(F29) is False
    src = ['*', '0'] + F29
    assert eng.simplify(list(src), mode='f64') == src


def test_a_null_pole_is_still_certified(eng) -> None:
    """A pole on a null set keeps its certificate (`1/x` at 0, the structural certificate), and
    float64's 1/+0 = +inf is a value: `0*tanh(1/floor(x))` is 0 at every float64 input."""
    assert eng._core.interval_finite_ae(['tanh', 'inv', 'x0']) is True
    assert eng.simplify(['*', '0', 'tanh', 'inv', 'floor', 'x0'], mode='f64') == ['0']


def test_a_pole_edge_attained_on_positive_measure_holds_the_infinity(eng) -> None:
    """float64's tanh is 1.0 on the whole ray past 19.06, where `atanh(tanh(x))` is +inf: the
    enclosure of a cell that reaches the ray holds +inf beside its finite part, a cell below it
    is finite, and float64's evidence finds `atanh(tanh(x))` finite on a set of positive
    measure (it read it as finite nowhere)."""
    tokens = ['atanh', 'tanh', 'x0']
    has_fin, pinf, _, nan, _, _ = eng._core.interval_value_set_box(tokens, [0.0], [64.0],
                                                                  real=False)
    assert has_fin and pinf and not nan
    has_fin, pinf, _, nan, _, hi = eng._core.interval_value_set_box(tokens, [0.0], [10.0],
                                                                   real=False)
    assert has_fin and not pinf and not nan and hi < 11
    assert _core.table_evidence(tokens, dict(ARITY), True) is True
