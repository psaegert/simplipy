"""sin/cos/tan of an astronomically large argument must be refused, not computed.

Reducing a periodic argument mod 2*pi costs pi to as many digits as the argument's binary
exponent. mpmath does not overflow the way f64 does -- exp(1e300) returns an exact mpf with
a ~1.4e20-bit exponent instead of inf -- so a rule like `cos exp <constant>` asks for pi to
~4.3e19 decimal digits. That does not finish.

Measured consequence before the guard: ~1.25% of the rules reaching the sort ladder were
unbounded, and a real mine sat in one of them for over 26 hours with no output.
"""
import time

import numpy as np
import pytest
from mpmath import mp, mpf

from simplipy import SimpliPyEngine
from simplipy.promotion import _f64_eval, _hp_equiv, _ladder
from simplipy.promotion._hp_equiv import PERIODIC_ARG_EXP_LIMIT, PeriodicRangeRefusal

#: Real offenders, taken verbatim from an acj-5-4 mine.
OFFENDERS = [
    (('atanh', 'neg', 'cos', 'exp', '<constant>'), ('<constant>',)),
    (('asin', 'sin', 'abs', 'sinh', '<constant>'), ('<constant>',)),
    (('acosh', 'sin', 'pow', '(-8)', '<constant>'), ('float("nan")',)),
    (('log', 'exp', 'cos', 'sinh', '<constant>'), ('<constant>',)),
]

#: The same functions applied directly to the constant: these MUST still be judged.
CONTROLS = [('cos',), ('sin',), ('tan',), ('exp',)]


@pytest.fixture(scope="module")
def engine():
    eng = SimpliPyEngine.load("base", install=True)
    _f64_eval.configure(eng)
    return eng


def test_the_evaluator_refuses_a_periodic_call_beyond_the_bound() -> None:
    huge = mp.exp(mpf(10) ** 300)          # exact in mpmath; inf in float64
    assert mp.mag(huge) > PERIODIC_ARG_EXP_LIMIT
    with pytest.raises(PeriodicRangeRefusal):
        _hp_equiv.evaluate(['cos', '<constant>'], {}, [huge])
    # ...and still computes the ordinary case at the same call site.
    assert abs(_hp_equiv.evaluate(['cos', '<constant>'], {}, [mpf(0)]) - 1) < 1e-40


@pytest.mark.parametrize("lhs,rhs", OFFENDERS)
def test_offenders_resolve_promptly_and_fail_closed(engine, lhs, rhs) -> None:
    start = time.monotonic()
    verdict, _ = _ladder.judge_ground(list(lhs), list(rhs),
                                      np.random.default_rng(_ladder.SEED))
    elapsed = time.monotonic() - start
    # FAIL-CLOSED: refused, never certified. EVAL-ERR drops the rule.
    assert verdict == 'EVAL-ERR', f"{' '.join(lhs)} returned {verdict}"
    assert elapsed < 5.0, f"{' '.join(lhs)} took {elapsed:.1f}s -- the bound is not binding"


@pytest.mark.parametrize("fn", CONTROLS)
def test_the_bound_does_not_touch_ordinary_rules(engine, fn) -> None:
    """The probe atom lattice reaches 1e300 (binary exponent ~997). Those are legitimate
    and must still be judged, so the bound has to sit above them."""
    verdict, _ = _ladder.judge_ground([fn[0], '<constant>'], ['<constant>'],
                                      np.random.default_rng(_ladder.SEED))
    assert verdict == 'PASS', f"{fn[0]} <constant> regressed to {verdict}"


def test_the_bound_sits_between_the_two_modes() -> None:
    """The cost distribution is bimodal with nothing in between, so the bound is not near a
    boundary: legitimate probes top out around 2**997, the pathology starts near 2**1.4e20."""
    assert mp.mag(mpf(10) ** 300) < PERIODIC_ARG_EXP_LIMIT < mp.mag(mp.exp(mpf(10) ** 300))
