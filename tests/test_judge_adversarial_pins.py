"""Adversarial pins of the judge (`simplipy.verify.verify_rule`): the candidates of the pre-audit
judge review (2026-11-27, at 3a6ed3a), each with the verdict the judge gives and why, so that a
change of any of them is seen.

The candidates hunt false admissions in the families most likely to slip: thresholds between and
beyond the sampled magnitudes, underflow and overflow that fake a constant, cancellation, several
variables, constant slots, and domain extensions. Each list says what its rules have in common.

Three lists pin rules the judge HOLDS THOUGH THEY ARE FALSE. They are strict xfails that assert
the verdict the semantics asks for (`SIMPLIPY_015_SEMANTICS.md` 3.1: wherever the left side has a
value, the right side agrees; 3.4: a mode's file holds the rules that hold in the mode), so a fix
turns them into failures to be moved to the refused lists, and a regression elsewhere is not
hidden by them:

- `FALSE_IN_BOTH_READINGS_HELD`: false in float64 AND in the reals at a point the judge reads, or
  on a band no point of the battery reaches. Not known limits; defects of the judge.
- `FALSE_IN_THE_REALS_HELD_IN_PERMISSIVE`: false in the reals beyond float resolution where
  float64 is false too, held by permissive's own criterion, which reads the reals nowhere below
  the mine's rows (1e-4) except at the sorts' atoms.
- `EVIDENCE_LIMITS`: false only between the points the battery reads, which `evidence` does not
  claim to exclude (`SIMPLIPY_015_ALGORITHM.md` 5.1: "no counterexample in the stated battery").

Every value quoted is SimpliPy's own evaluator (`simplipy.operators`) and mpmath at 60 digits.
"""
import math

import pytest

from simplipy import operators as ops
from simplipy.verify import _contract, verify_rule


def _judge(lhs: str, rhs: str) -> dict:
    return verify_rule(lhs.split(), rhs.split())


# ------------------------------------------------------------------ held though false (defects)

#: False in float64 and in the reals on a set of positive measure, held in tier `core`.
FALSE_IN_BOTH_READINGS_HELD = [
    # In the reduced reference mine (a0e8269, strict) and in the strict length-4 file of
    # acj-4-llm_0.15: acos(cos(y)) = |y| only for |y| <= pi, and atanh(x) passes pi on
    # tanh(pi) = 0.99627 < |x| < 1. At x = 0.999: 2.482984139929387 against 3.8002011672501994
    # (float64 and mpmath alike). The judge's own row 2611 (x0 = -0.9997006899413192) refutes it,
    # but the rows are read in the reals only at the first ROWS_JUDGED = 8 where float64 differs
    # (454 rows differ, nearly all by acos's rounding near 1), and no atom, grid point or rung of
    # the ladder lies in the band.
    ('acos cos atanh ?0', 'abs atanh ?0'),
    # The same with the band at tanh(2 pi) = 0.99999303 < |x| < 1 (7e-6 wide).
    ('acos cos * 0.5 atanh ?0', 'abs * 0.5 atanh ?0'),
    # False on 0 < |x| < 1e-5: at x = 1e-8, tanh(1) - tanh(1e-3) = 0.7606 in float64 and in the
    # reals. The judge lists x = +-1e-16 and 1e-8 among its float64 disagreements (`points`) and
    # holds the rule by strict's second reading (`strict_reading`, `kind_holds`): at a point the
    # reals do not read (the rungs at and below 1e-8 are read in float64 only) a finite float64
    # difference counts as rounding wherever the reals give both sides a value, whether or not
    # the reals agree there.
    ('- tanh min * abs ?0 10000000000000000 1 tanh min * abs ?0 100000 1', '0'),
    ('tanh min * abs ?0 100000 1', 'tanh min * abs ?0 1e300 1'),
]

#: False in the reals (and in float64) on 0 < |x| < b with b below the mine's rows' 1e-4, held in
#: tier `permissive` by permissive's own criterion, which does not read the reals at the rungs at
#: and below 1e-8. Spelled in the 0.15 alphabet (`pow 10 10`, `exp exp 3`), longer than the mines.
FALSE_IN_THE_REALS_HELD_IN_PERMISSIVE = [
    # b = 1.9e-9: at x = 1e-10, tanh(1) = 0.7616 against 1
    ('tanh * pow 10 10 abs ?0', 'ceil tanh abs ?0'),
    # b = 3.6e-8: at x = 1e-8, tanh(5.28) = 0.99995 against 1
    ('tanh * exp exp 3 abs ?0', 'ceil tanh abs ?0'),
]

#: False only between the points the battery reads: positive-measure counterexamples that no
#: atom, grid point, row, preimage or rung reaches. `evidence` claims no counterexample at the
#: stated points, and these have none there.
EVIDENCE_LIMITS = [
    # (2e4, 5e4): between the grid's widest point 1e4 and the ladder's 1e5
    (('tanh max - abs ?0 20000 0', 'tanh max - abs ?0 50000 0'), 'reject'),
    (('- tanh max - abs ?0 20000 0 tanh max - abs ?0 50000 0', '0'), 'reject'),
    # (2e7, 5e9): between the rungs 1e7 and 1e10
    (('tanh max - abs ?0 2e7 0', 'tanh max - abs ?0 5e9 0'), 'reject'),
    # x0 > 2e4 and x1 < -2e4: the ladder reads one slot at a time beside a generic value and both
    # slots on the diagonal, never two large magnitudes of opposite signs
    (('tanh max 0 - min ?0 neg ?1 20000', '0'), 'reject'),
    # an extension on (0.29999, 0.30001), narrower than the interval layer's cells (5.5's first
    # known limit): strict owes clause (b) and should refuse it; permissive may extend
    (('+ ?0 * 0 log - abs - ?0 0.3 0.00001', '?0'), 'permissive'),
]


@pytest.mark.xfail(strict=True, reason='judge defect (pre-audit review J-1, J-2): held though false '
                                       'in float64 and in the reals')
@pytest.mark.parametrize('lhs, rhs', FALSE_IN_BOTH_READINGS_HELD)
def test_false_in_both_readings_is_refused(lhs, rhs) -> None:
    r = _judge(lhs, rhs)
    assert r['tier'] == 'reject', r


@pytest.mark.parametrize('lhs, rhs', FALSE_IN_BOTH_READINGS_HELD)
def test_false_in_both_readings_is_held_today(lhs, rhs) -> None:
    """The defect as it stands, so a change of the verdict is seen either way."""
    r = _judge(lhs, rhs)
    assert r['tier'] == 'core' and r['realised'] is True, r
    assert r['realised_f6'] is False and r['realised_kinds'] is True, r


@pytest.mark.xfail(strict=True, reason='judge defect (pre-audit review J-3): permissive reads the '
                                       'reals nowhere below 1e-4')
@pytest.mark.parametrize('lhs, rhs', FALSE_IN_THE_REALS_HELD_IN_PERMISSIVE)
def test_false_in_the_reals_at_tiny_magnitudes_is_refused(lhs, rhs) -> None:
    r = _judge(lhs, rhs)
    assert r['tier'] == 'reject', r


@pytest.mark.parametrize('lhs, rhs', FALSE_IN_THE_REALS_HELD_IN_PERMISSIVE)
def test_false_in_the_reals_at_tiny_magnitudes_is_held_today(lhs, rhs) -> None:
    r = _judge(lhs, rhs)
    assert r['tier'] == 'permissive' and r['permissive_own'] is True, r


@pytest.mark.xfail(strict=True, reason='evidence limit: no point of the battery in the false region')
@pytest.mark.parametrize('rule, tier', EVIDENCE_LIMITS)
def test_false_between_the_points_read_is_refused(rule, tier) -> None:
    r = _judge(*rule)
    assert r['tier'] == tier, r


def test_the_counterexamples_are_real() -> None:
    """The values quoted above, from SimpliPy's own evaluator and mpmath."""
    mpmath = pytest.importorskip('mpmath')
    x = 0.999
    assert ops.acos(ops.cos(ops.atanh(x))) == pytest.approx(2.482984139929387, rel=1e-14)
    assert ops.abs(ops.atanh(x)) == pytest.approx(3.8002011672501994, rel=1e-14)
    with mpmath.workdps(60):
        y = mpmath.atanh(mpmath.mpf(x))
        assert abs(mpmath.acos(mpmath.cos(y)) - (2 * mpmath.pi - y)) < mpmath.mpf(10) ** -50
    assert math.tanh(math.pi) == pytest.approx(0.99627207622075)
    # the judge's own row in the band
    rows = _contract.judge_rows()
    band = [float(v) for v in rows[:, 0] if math.tanh(math.pi) < abs(v) < 1]
    assert band == [-0.9997006899413192]
    # the tiny band: x = 1e-8, in float64 and in the reals
    v64 = ops.tanh(ops.min(1e-8 * 1e16, 1.0)) - ops.tanh(ops.min(1e-8 * 1e5, 1.0))
    assert v64 == pytest.approx(0.7605941562890981, rel=1e-14)
    with mpmath.workdps(60):
        vr = mpmath.tanh(1) - mpmath.tanh(mpmath.mpf('1e-3'))
        assert abs(vr - mpmath.mpf(v64)) < 1e-15
    assert ops.tanh(1e10 * 1e-10) == pytest.approx(0.7615941559557649)


# ------------------------------------------------------------------ refused, as they should be

#: False rules the judge refuses, by family. `reject` unless the tier is given: permissive holds an
#: extension (3.1) and a named step of its own (4.9).
REFUSED = [
    # thresholds past the grid, read by the ladder in the reals (FX-74)
    ('min 1 max 0 - log abs ?0 10', '0', 'reject'),
    ('min 1 max 0 - 11 log abs ?0', '0', 'reject'),
    # a band the mine's rows reach (|x| < 8.6e-4): read in the reals at a row
    ('tanh * exp 10 abs ?0', 'ceil tanh abs ?0', 'reject'),
    # underflow that fakes a zero: Bessel functions at huge orders (FX-74's class), refuted past the
    # magnitude where the function leaves the doubles' zero, or left without an answer there
    ('- besselk 1 ?0 besselk (-1) ?0', 'besselj exp 10 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'besselj exp 12 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'besselj exp 15 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'besselj exp 20 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'besselj exp 40 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'besseli exp 10 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'inv besselk exp 10 ?0', 'reject'),
    ('- besselk 1 ?0 besselk (-1) ?0', 'inv bessely exp 10 ?0', 'reject'),
    ('* 0 ?0', 'erfc exp exp ?0', 'reject'),
    # acos(cos(y)) = |y| past pi, where the battery reads the slot's band
    ('acos cos * 2 atanh ?0', 'abs * 2 atanh ?0', 'reject'),
    ('acos cos log ?0', 'abs log ?0', 'reject'),
    ('acos cos inv ?0', 'abs inv ?0', 'reject'),
    ('acos cos atanh tanh ?0', 'abs atanh tanh ?0', 'reject'),
    ('acos cos harmonic ?0', 'abs harmonic ?0', 'reject'),
    ('acos cos tan ?0', 'abs tan ?0', 'reject'),
    ('acos cos atanh erf ?0', 'abs atanh erf ?0', 'reject'),
    ('acos cos atanh sin ?0', 'abs atanh sin ?0', 'reject'),
    ('acos cos atanh cos ?0', 'abs atanh cos ?0', 'reject'),
    # inverse pairs: false in the reals, held in permissive by its named step (4.9)
    ('asin sin atanh ?0', 'atanh ?0', 'permissive'),
    ('atan tan atanh ?0', 'atanh ?0', 'permissive'),
    ('acos cos acosh ?0', 'acosh ?0', 'permissive'),
    # several variables: on a line (found by the f64 bar's preimage of the kink under ceil) and on
    # the anti-diagonal (the atoms' product)
    ('ceil tanh abs - - ?0 ?1 10', '1', 'reject'),
    ('ceil tanh abs + ?0 ?1', '1', 'reject'),
    # constants past the span (1e4) and below it (1e-3)
    ('tanh max 0 - <constant> 100000', '0', 'reject'),
    ('tanh max 0 - abs <constant> 100000', '0', 'reject'),
    ('tanh max 0 - 1000 inv abs <constant>', '0', 'reject'),
    ('* <constant> tanh max 0 - ?0 100000', '0', 'reject'),
    # extensions of positive measure: strict refuses, permissive extends
    ('+ ?0 * 0 log - abs - ?0 0.3 0.01', '?0', 'permissive'),
    ('+ ?0 * 0 acosh + 1 - abs - ?0 1e5 1', '?0', 'permissive'),
    # cancellation: false where the precision ladder has to read the cancelled part
    ('log / cosh ?0 sinh ?0', '* 2 exp * -2 ?0', 'reject'),
    ('- erf ?0 1', '* 0 ?0', 'reject'),
    ('- erf abs ?0 1', 'neg exp neg * ?0 ?0', 'reject'),
    ('- cosh ?0 sinh ?0', '* 0 ?0', 'reject'),
    ('- cosh ?0 sinh ?0', 'inv cosh ?0', 'reject'),
    ('- cosh abs ?0 sinh abs ?0', '* 2 exp neg abs ?0', 'reject'),
]


@pytest.mark.parametrize('lhs, rhs, tier', REFUSED)
def test_false_rules_are_refused(lhs, rhs, tier) -> None:
    r = _judge(lhs, rhs)
    assert r['tier'] == tier, r
    assert r['realised'] is not True, r


#: True rules where float64 parts from the reals (cancellation, saturation): held in `core`.
HELD = [
    ('- erf ?0 1', 'neg erfc ?0'),           # 0.0 against -8.9e-89 at x = 14.1; equal in the reals
    ('- cosh ?0 sinh ?0', 'exp neg ?0'),     # 0.0 against e^-x at large x; equal in the reals
    ('- + ?0 ?1 ?1', '?0'),                  # 0 against 1 at (1, 1e20): strict's second reading
]


@pytest.mark.parametrize('lhs, rhs', HELD)
def test_true_rules_are_held(lhs, rhs) -> None:
    r = _judge(lhs, rhs)
    assert r['tier'] == 'core', r
