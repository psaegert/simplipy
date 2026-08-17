"""F74: the judge upgrade -- honest-instrument falsifiers (2026-08-10).

Every case here is a REAL row from the extreme-lane 1M fuzz stream (record
fuzz_extreme_f73_1M.json) or a distilled base-lane adjudication-table shape. The
2026-08-09 decomposition left 190 LIVE judge convictions; tracing them found the
convictions were overwhelmingly fabricated by the instrument, via five mechanisms:

  D1  c_pow swallowed the Unresolved refusal raised by _pow_mag_capped
      (`except Exception: return nan`), turning "cannot evaluate at working
      precision" into a fabricated nan -- which then read as positive-measure
      EXTENSION/SHRINK against correct engine outputs (~150 rows).
  D2  MAG_CAP at 1e50 refused magnitudes mpmath evaluates exactly in
      milliseconds (extreme-literal pows reach ~10^(3e311); measured cost 23 ms).
  D3  _snap_literal_zeros tested only MAXIMAL variable-free spans at HARDCODED
      dps 50/110: a true nonzero difference spanning 272 digits read zero on
      both rungs (the judge then rewrote the input to '0' and convicted the
      correct engine), and an inner exact-zero span (tan pi) was never snapped
      when the outer span evaluated cleanly.
  D4  working precision was fixed at dps 50, so cos(1e-320) rounded to
      exactly 1 and pow(<1, -inf) = +inf was convicted as INF-CHANGE.
  D5  tanh/cosh values whose distance from +-1/1 lies below any working
      precision (tanh(9e15) = 1 - 2e^{-1.8e16}) rounded to the boundary
      exactly, fabricating acosh(1)=0 / pow(1,-inf)=1 against correct nan/inf.

The upgrade fixes the instrument only -- the engine does not move. Verdicts that
must FLIP are fabrications dying; verdicts that must STAY are genuine items the
upgrade is forbidden to absorb (stretching the instrument to clear them would be
the no-hand-pins bar violated from the other direction).
"""
import numpy as np
import pytest

from simplipy.verify._monitor import (
    OPS,
    Unresolved,
    _fold_exact_rational_spans,
    _pair_dps,
    _snap_literal_exacts,
    evaluate,
    judge_pair,
)


def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------------------
# Extreme-lane rows whose convictions were instrument fabrications: now OK.
# (row id, input tokens, engine output tokens, defect exercised)
# ---------------------------------------------------------------------------------------

FLIP_TO_OK = [
    # D3 (adaptive rungs): 1e309 + (-1e309 - 2.55e38) needs 272 digits; the two-rung
    # zero test at 50/110 read it as exact zero and rewrote the input to '0'.
    (13391,
     ['+', '1e309', '-', '-1e309', '2.552117751907039e38'],
     ['neg', '2.552117751907039e38']),
    (490679,
     ['+', '-', '9007199254740993', '1e309', '1e309'],
     ['9007199254740993']),
    (681877,
     ['-', '-', '-3', '-1e309', '+', '1e309', '5.104235503814077e38'],
     ['neg', '+', '5.104235503814077e38', '3']),
    (435247,
     ['*', '+', '-1e309', '-', '4503599627370496', '-1e309', '-', 'np.pi',
      'pow', '-2', '-2'],
     ['*', '4503599627370496', '-', 'np.pi', '/', '1', '4']),
    # D3 via exact-rational folding: the f64-max cancellation resolves in Fraction.
    (908611,
     ['*', 'pow', '-', '1.7976931348623157e308', '+', '0.9999999999999999',
      '1.7976931348623157e308', '3', '-', '+', 'pow', 'x1',
      '3.402823669209385e38', '/', 'x1', '-2', '2'],
     ['*', 'pow', '-0.9999999999999999', '3', '-', '-', 'pow', 'x1',
      '3.402823669209385e38', '/', 'x1', '2', '2']),
    # D1+D2: pow(1.79e308, 1.79e308) is a huge FINITE number, so finite - inf = -inf
    # is exactly right; the swallowed refusal fabricated nan - inf = nan.
    (228289,
     ['-', 'pow', '1.7976931348623157e308', '1.7976931348623157e308',
      'float("inf")'],
     ['float("-inf")']),
    (64059,
     ['/', 'float("inf")', 'pow', '1.7976931348623157e308', '1e309'],
     ['float("inf")']),
    (127368,
     ['/', 'float("inf")', 'pow', '6.80564733841877e38', '1e309'],
     ['float("inf")']),
    (121456,
     ['/', '*', 'pow', '10000000000000000001', '-1e309', '1e309', '-', '*',
      '0', '-1e19', '0'],
     ['float("inf")']),
    (9070,
     ['-', '*', 'pow', '-', '2.552117751907039e38',
      '1/170141183460469231731687303715884105727', '-1e309', 'float("inf")',
      '-', 'x4', '9007199254740991'],
     ['float("inf")']),
    (296054,
     ['atanh', 'pow', 'pow', '/', '0', '*',
      '170141183460469231731687303715884105727', 'x4', '9007199254740992',
      '170141183460469231731687303715884105727'],
     ['atanh', '*', '0', 'pow', 'inv', 'pow', 'x4', '9007199254740992',
      '170141183460469231731687303715884105727']),
    # D3 (recursive snap): sin(pi) is exactly zero, so 0 * inf = nan on both sides;
    # the maximal-span-only snap left the inner residue alive as +-inf.
    (360075,
     ['*', 'sin', 'np.pi', 'float("inf")'],
     ['float("nan")']),
    # D3 (recursive snap): tan(pi) = 0 exactly, so pow(nan, 0) = 1 (ratified
    # totality) on the input side too; the outer span evaluated cleanly, so the
    # inner exact zero was never snapped and nan propagated.
    (636919,
     ['pow', '/', 'cosh', '1/170141183460469231731687303715884105727', 'acosh',
      '1e-320', 'tan', 'np.pi'],
     ['1']),
    # D4 (adaptive dps): cos(1e-320) = 1 - 5e-641 < 1 strictly, so
    # pow(<1, -inf) = +inf; at dps 50 the base rounded to exactly 1.
    (38796,
     ['pow', 'cos', '1e-320', 'float("-inf")'],
     ['float("inf")']),
]


@pytest.mark.parametrize("row,inp,out", FLIP_TO_OK, ids=[str(r[0]) for r in FLIP_TO_OK])
def test_fabricated_conviction_now_ok(row, inp, out):
    verdict, detail = judge_pair(list(inp), list(out), rng())
    assert verdict == 'OK', f'row {row}: {verdict} [{detail}]'


# ---------------------------------------------------------------------------------------
# D5 rows: the truth is UNKNOWABLE at any working precision (tanh(9e15) sits
# 10^{-8e15} below 1). The honest verdict is a refusal -- UNSCORED, recorded --
# never the old fabricated conviction of a correct output.
# ---------------------------------------------------------------------------------------

FLIP_TO_UNSCORED = [
    (658467,
     ['acosh', 'tanh', '9007199254740992'],
     ['float("nan")']),
    (608227,
     ['/', '1e-320', 'acosh', 'tanh', '10000000000000000001'],
     ['float("nan")']),
    (169452,
     ['pow', 'tanh', '1.2379400392853803e27', 'float("-inf")'],
     ['float("inf")']),
    (34296,
     ['rootn', 'pow', 'tanh', '10000000000000000001', 'float("-inf")',
      '170141183460469231731687303715884105728'],
     ['rootn', 'float("inf")', '170141183460469231731687303715884105728']),
]


@pytest.mark.parametrize("row,inp,out", FLIP_TO_UNSCORED,
                         ids=[str(r[0]) for r in FLIP_TO_UNSCORED])
def test_unknowable_saturation_is_refused_not_convicted(row, inp, out):
    verdict, detail = judge_pair(list(inp), list(out), rng())
    assert verdict == 'UNSCORED', f'row {row}: {verdict} [{detail}]'


# ---------------------------------------------------------------------------------------
# Negative regressions: genuine items the upgrade must NOT absorb.
# ---------------------------------------------------------------------------------------

STILL_VIOLATION = [
    # pow(nan, 0) = 1 is the ratified totality, so the input is a defined huge
    # value; the engine's nan output loses definedness. A genuine engine-side
    # question (rule overlap at the nan/zero-exponent corner), not instrument noise.
    (307172,
     ['pow', '*', '92233720368547758/7', 'pow', '+', 'float("nan")', '1e309',
      'pow', 'float("inf")', '-5e-324', '9007199254740992'],
     ['float("nan")']),
]

# RE-ADJUDICATED 2026-08-11 (owner: "this should be fixed properly"). Row 508487 --
# `rootn(0, -1) = +inf` vs `-3/0 = -inf`, downstream exp/pow turning that into 0 vs 1
# at the x0 = 0 atom -- was pinned here as a VIOLATION with its own comment calling it
# "a real convention-boundary item for the owner". It is the singular-input class: the
# probe sits on the LHS's own pole, where the value is a convention-completion and the
# sign is spelling-dependent under ANY +-inf-valued extension. Decisive evidence: run
# through the ENGINE's own deployed realizations (`simplipy.promotion._f64_eval` on
# acj-4-3), the two sides agree at every point including +-0.0 -- `/ x0 -3` is -0.0
# there, so `rootn(/ x0 -3, -1)` and `/ -3 x0` both give -inf. The conviction was the
# abstraction's, not the rewrite's. The falsifier is KEPT, in the new direction.
EXPECTED_OK_SINGULAR = [
    (508487,
     ['pow', 'cos', '/', '-', 'x1', '-3', 'rootn', '9007199254740991', '-3',
      'exp', 'rootn', '/', 'x0', '-3', '-1'],
     ['pow', 'cos', '*', 'rootn', '9007199254740991', '3', '+', 'x1', '3',
      'exp', '/', '-3', 'x0']),
]


@pytest.mark.parametrize("row,inp,out", STILL_VIOLATION,
                         ids=[str(r[0]) for r in STILL_VIOLATION])
def test_genuine_conviction_survives_the_upgrade(row, inp, out):
    verdict, detail = judge_pair(list(inp), list(out), rng())
    assert verdict == 'VIOLATION', f'row {row}: {verdict} [{detail}]'


@pytest.mark.parametrize("row,inp,out", EXPECTED_OK_SINGULAR,
                         ids=[str(r[0]) for r in EXPECTED_OK_SINGULAR])
def test_singular_class_is_not_convicted(row, inp, out):
    verdict, detail = judge_pair(list(inp), list(out), rng())
    assert verdict == 'OK', f'row {row}: {verdict} [{detail}]'


@pytest.mark.parametrize("row,inp,out", EXPECTED_OK_SINGULAR,
                         ids=[str(r[0]) for r in EXPECTED_OK_SINGULAR])
def test_singular_class_agrees_under_deployed_realizations(row, inp, out):
    """The evidence that moved the row, kept executable: the DEPLOYED realizations
    (what every consumer computes) agree at the point the judge used to convict on."""
    from simplipy import SimpliPyEngine
    from simplipy.promotion import _f64_eval as F

    # R6 hygiene: the suite must never WRITE the real user cache (the audit watched
    # the shared cache change mid-pytest-run); resolve the asset the same
    # local-then-cache way every other suite does.
    from conftest import acj_config_path
    engine = SimpliPyEngine.from_config(acj_config_path())
    F.configure(engine)
    for x0 in (0.0, -0.0, 1e-300, -1e-300, 2.0, -2.0):
        env = {'x0': np.array([x0]), 'x1': np.array([0.7])}
        a = F.evaluate(list(inp), env)[0]
        b = F.evaluate(list(out), env)[0]
        assert a == b or (np.isnan(a) and np.isnan(b)), \
            f'row {row}: deployed realizations DISAGREE at x0={x0!r}: {a} vs {b}'


# ---------------------------------------------------------------------------------------
# Base-lane adjudication-table shapes (the 47-entry interim table this upgrade
# is meant to dissolve): distilled EXACT / GAP class representatives.
# ---------------------------------------------------------------------------------------

class TestBasePinShapes:
    def test_derived_rootn_index_from_identity_roundtrip(self):
        # GAP class: sinh(asinh(2)) = 2 exactly; the mpf residue 2 +- 1e-49 read as a
        # non-integer index and fabricated nan. The integer snap restores the index.
        verdict, detail = judge_pair(
            ['rootn', 'x0', 'sinh', 'asinh', '2'], ['rootn', 'x0', '2'], rng())
        assert verdict == 'OK', f'{verdict} [{detail}]'

    def test_derived_negative_rootn_index_from_rational_pow(self):
        # GAP class: pow((-1/3), -3) = -27 exactly (exact-rational folding);
        # rootn(x, -27) = inv(rootn(x, 27)) is the engine's ratified spelling.
        verdict, detail = judge_pair(
            ['rootn', 'x0', 'pow', '(-1/3)', '-3'],
            ['inv', 'rootn', 'x0', '27'], rng())
        assert verdict == 'OK', f'{verdict} [{detail}]'

    def test_pow_with_sin_pi_exponent_is_one(self):
        # EXACT class: pow(x, sin pi) = x^0 = 1 for every x (ratified totality).
        verdict, detail = judge_pair(
            ['pow', 'x0', 'sin', 'np.pi'], ['1'], rng())
        assert verdict == 'OK', f'{verdict} [{detail}]'

    def test_tan_pi_zero_over_zero_is_nan(self):
        # EXACT class: tan(pi) = 0 exactly, so 0/0 = nan; the judge saw tiny/0 = inf.
        verdict, detail = judge_pair(
            ['/', 'tan', 'np.pi', '*', '0', 'x0'], ['float("nan")'], rng())
        assert verdict == 'OK', f'{verdict} [{detail}]'


# ---------------------------------------------------------------------------------------
# Component units.
# ---------------------------------------------------------------------------------------

class TestExactRationalFolding:
    def test_pure_rational_span_folds_exactly(self):
        out = _fold_exact_rational_spans(
            ['+', '1e309', '-', '-1e309', '2.552117751907039e38'])
        assert out == ['-255211775190703900000000000000000000000']

    def test_fold_speaks_the_bag_grammar(self):
        out = _fold_exact_rational_spans(
            ['<add>', '1e309', '<sub>', '1e309', '2.552117751907039e38', '</add>'])
        assert out == ['-255211775190703900000000000000000000000']

    def test_rational_pow_folds_within_budget(self):
        assert _fold_exact_rational_spans(['pow', '(-1/3)', '-3']) == ['-27']

    def test_oversized_pow_is_left_unfolded(self):
        toks = ['pow', '10000000000000000001', '-1e309']
        assert _fold_exact_rational_spans(list(toks)) == toks

    def test_transcendental_spans_are_left_to_the_ladder(self):
        toks = ['*', 'sin', 'np.pi', '2']
        assert _fold_exact_rational_spans(list(toks)) == toks

    def test_division_by_zero_is_left_to_contract_semantics(self):
        toks = ['/', '3', '0']
        assert _fold_exact_rational_spans(list(toks)) == toks

    def test_folded_zero_is_the_zero_token(self):
        assert _fold_exact_rational_spans(['-', '1e309', '1e309']) == ['0']


class TestExactSnap:
    def test_inner_span_snaps_when_outer_does_not(self):
        # the 636919 mechanism: the maximal span evaluates cleanly, the inner
        # exact zero must still snap.
        out = _snap_literal_exacts(
            ['pow', 'x0', 'tan', 'np.pi'])
        assert out == ['pow', 'x0', '0']

    def test_identity_roundtrip_snaps_to_the_integer(self):
        # direct-internal call: the caller owns the ambient precision (C35)
        from mpmath import mp
        from simplipy.verify._monitor import DPS
        with mp.workdps(DPS):
            assert _snap_literal_exacts(['sinh', 'asinh', '2']) == ['2']

    def test_true_nonzero_across_many_digits_does_not_snap(self):
        toks = ['+', '1e309', '-', '-1e309', '2.552117751907039e38']
        # (pre-folding handles this exactly; the ladder alone must also refuse
        # to snap it -- the F74 defect was snapping it to '0' at dps 50/110)
        out = _snap_literal_exacts(list(toks))
        assert '0' not in out

    def test_stable_tiny_value_is_left_alone(self):
        toks = ['exp', '-200']          # e^-200 ~ 1e-87: tiny, stable, NOT zero
        assert _snap_literal_exacts(list(toks)) == toks


class TestPowHonesty:
    def test_unresolved_propagates_out_of_c_pow(self):
        # beyond-cap magnitude: an honest refusal, never a fabricated nan
        with pytest.raises(Unresolved):
            OPS['pow'][1](evaluate(['2'], {}), evaluate(['1e2500'], {}))

    def test_extreme_magnitude_now_evaluates(self):
        from mpmath import isinf, isnan, mp
        v = OPS['pow'][1](evaluate(['1.7976931348623157e308'], {}),
                          evaluate(['1.7976931348623157e308'], {}))
        assert not isnan(v) and not isinf(v) and v > 1
        assert mp.log10(mp.log10(v)) > 300     # magnitude ~10^(5.5e310)


class TestSaturationHonesty:
    def test_exp_rounding_erasure_is_unresolved(self):
        # exp(x) = 1 iff x = 0 for reals, so mp.exp returning EXACTLY 1 on a
        # nonzero argument is always a rounding erasure. Found the hard way: the
        # snap ladder accepted exp(exp(-1e309)) as exactly 1 through every rung
        # (the true value is 1 + 10^{-4.34e308}, above ANY precision), snapped the
        # span to '1', and the judge convicted the engine's CORRECT nan for
        # asin(above-1) -- a fabrication introduced by the F74 upgrade itself
        # (extreme row 924168).
        with pytest.raises(Unresolved):
            OPS['exp'][1](evaluate(['exp', '-1e309'], {}))

    def test_row_924168_judges_unscored_not_convicted(self):
        verdict, detail = judge_pair(
            ['/', '*', 'log', '2.2250738585072014e-308', 'float("inf")', 'asin',
             'exp', 'exp', '-1e309'],
            ['float("nan")'], rng())
        assert verdict == 'UNSCORED', f'{verdict} [{detail}]'

    def test_snap_refuses_the_exp_erasure_span(self):
        toks = ['asin', 'exp', 'exp', '-1e309']
        assert _snap_literal_exacts(list(toks)) == toks

    def test_pow_rounding_erasure_is_unresolved(self):
        # same argument for pow: a^b = 1 with a != 1 iff b = 0 (handled upstream),
        # so an exact-1 result on a nonzero exponent is an erasure.
        with pytest.raises(Unresolved):
            OPS['pow'][1](evaluate(['2'], {}), evaluate(['exp', '-1e309'], {}))

    def test_tanh_rounding_erasure_is_unresolved(self):
        with pytest.raises(Unresolved):
            OPS['tanh'][1](evaluate(['9007199254740992'], {}))

    def test_cosh_rounding_erasure_is_unresolved(self):
        with pytest.raises(Unresolved):
            OPS['cosh'][1](evaluate(['1e-320'], {}))

    def test_resolvable_tanh_still_evaluates(self):
        from mpmath import mp
        from simplipy.verify._monitor import DPS
        with mp.workdps(DPS):
            v = OPS['tanh'][1](evaluate(['20'], {}))
            assert v < 1

    def test_cosh_at_zero_is_exactly_one(self):
        assert OPS['cosh'][1](evaluate(['0'], {})) == 1


class TestPairDps:
    def test_small_literals_keep_the_ambient_dps(self):
        from mpmath import mp
        with mp.workdps(50):
            assert _pair_dps(['+', 'x0', '2.0', '(-0.5)', '3.0']) == 50

    def test_extreme_span_raises_the_working_precision(self):
        d = _pair_dps(['pow', 'cos', '1e-320', 'float("-inf")'])
        assert d >= 650          # must resolve 1 - cos(1e-320) ~ 5e-641

    def test_derivation_is_capped(self):
        assert _pair_dps(['+', '1e309', 'pow', '5e-324', '3']) <= 3000


# ---------------------------------------------------------------------------------------
# The SINGULAR-INPUT gate (9.8.3(a) boundary, second rung; fbench pilot row 733,
# 2026-08-11). At a REAL probe on the input's singular manifold (an internal +-inf
# arises during evaluation) the input has no mathematical value; the reals both sides
# evaluate to are convention-completions (one-zero + limit-completion) and may
# legitimately differ between spellings: 1/(0/t) = +inf but (1/0)*t = -inf for t < 0,
# and atan renders both REAL as -+pi/2. Routed to clause (c)'s null/measure doctrine,
# exactly like the env-inf rung (inf-input-null). The gate must NOT absorb value
# changes at points where the input evaluates CLEANLY -- there mathematics has the
# answer and clause (a) keeps zero tolerance.
# ---------------------------------------------------------------------------------------

class TestSingularInputGate:
    # the pilot's minimal witness (P1 reduction of corpus row 733): the engine's sound
    # rewrite x4 - atan(E) -> x4 + atan(-E) with the division chain restructured; the
    # two spellings are a.e.-identical (independent 60-point full-measure check) and
    # diverge ONLY on the correlated diagonal where x3 - x1 = 0 zeroes a denominator.
    WITNESS_IN = ['-', 'x4', 'atan', '/', '/', '/', 'x3', 'rootn', '/', 'exp', 'x11',
                  '-', 'x3', 'x1', '-2', 'x1', '/', '-', 'x7', 'x4', 'x7']
    WITNESS_OUT = ['+', 'x4', 'atan', '/', 'neg', '*', 'x3', '*', 'x7', '*', 'rootn',
                   '/', 'exp', 'x11', '-', 'x3', 'x1', '2', 'inv', '-', 'x7', 'x4', 'x1']

    def test_convention_completion_disagreement_is_not_convicted(self):
        for seed in range(4):
            verdict, detail = judge_pair(list(self.WITNESS_IN), list(self.WITNESS_OUT),
                                         np.random.default_rng(seed))
            assert verdict == 'OK', f'seed {seed}: {verdict} [{detail}]'

    def test_the_gate_tallies_the_null_event(self):
        from simplipy.verify._monitor import TOL_COUNTS
        before = TOL_COUNTS.get('singular-real-null', 0)
        judge_pair(list(self.WITNESS_IN), list(self.WITNESS_OUT), rng())
        assert TOL_COUNTS.get('singular-real-null', 0) > before

    def test_clean_input_value_change_keeps_strict_authority(self):
        verdict, _ = judge_pair(['*', '2', 'x1'], ['*', '3', 'x1'], rng())
        assert verdict == 'VIOLATION'

    def test_output_side_singularity_keeps_strict_authority(self):
        # input x1 evaluates CLEANLY real everywhere; the output atan(inv(x1 - x1)) =
        # atan(+inf) = pi/2 crosses a singularity on ITS side only. The input's value
        # IS mathematics' answer at every probe -- the gate must not shield the output.
        verdict, _ = judge_pair(['x1'], ['atan', 'inv', '-', 'x1', 'x1'], rng())
        assert verdict == 'VIOLATION'

    def test_input_singular_predicate_boundaries(self):
        from mpmath import mpf
        from simplipy.verify._monitor import _input_singular
        env = {'x1': mpf(2)}
        assert _input_singular(['inv', '-', 'x1', 'x1'], env)          # 1/0 crossing
        assert _input_singular(['atan', 'inv', '-', 'x1', 'x1'], env)  # compressed back
        assert not _input_singular(['+', 'x1', '2'], env)              # clean real
        # a DENOTED inf literal immediately compressed is not a crossing: its
        # limit-completed value is ratified, strict authority stands.
        assert not _input_singular(['atan', 'float("inf")'], env)

    def test_no_convention_closes_the_class(self):
        """WHY the clause is scoped rather than the convention re-chosen.

        Under any extension giving poles +-inf values (the ratified one), the pole's
        SIGN is spelling-dependent: `1/(x1-x3)` and `-1/(x3-x1)` are a.e. identical yet
        give +inf and -inf, and a bounded compressor above them renders that as a
        REAL-vs-REAL disagreement -- inside clause (a)'s own jurisdiction. IEEE's signed
        zero does not close it either (`x-x` is +0 for every finite x, so the zero
        cannot record which spelling formed it), which is why option "adopt -0" was
        rejected on evidence rather than on doctrine.
        """
        from mpmath import mpf
        from simplipy.verify._monitor import evaluate
        env = {'x1': mpf(2), 'x3': mpf(2)}
        a = evaluate(['inv', '-', 'x1', 'x3'], env)
        b = evaluate(['neg', 'inv', '-', 'x3', 'x1'], env)
        assert a != b, 'the pole sign is spelling-dependent -- premise of the gate'

        # IEEE, where the signed zero is native: x-x is +0 both ways, so it cannot help
        assert np.float64(2.0) - np.float64(2.0) == 0.0
        assert not np.signbit(np.float64(2.0) - np.float64(2.0))

        # and the compressed pair is a clause-(a)-shaped disagreement the gate must clear
        for wrap in (['atan'], ['tanh']):
            verdict, detail = judge_pair(wrap + ['inv', '-', 'x1', 'x3'],
                                         wrap + ['neg', 'inv', '-', 'x3', 'x1'], rng())
            assert verdict == 'OK', f'{wrap}: {verdict} [{detail}]'

    def test_positive_measure_singular_disagreement_still_kills(self):
        """The gate tolerates NULL events only. A rewrite whose singular disagreement
        covers positive measure must still die -- routed to the INF-CHANGE measure
        clause, not silently dropped."""
        # 1/(x1-x1) is singular EVERYWHERE (the whole domain is the pole), so the
        # disagreement is not a null event.
        verdict, _ = judge_pair(['atan', 'inv', '-', 'x1', 'x1'],
                                ['atan', 'neg', 'inv', '-', 'x1', 'x1'], rng())
        assert verdict == 'VIOLATION'


class TestTwinInstrumentInLockstep:
    """The 9.8.3(a) scoping must hold in BOTH instruments. The 2026-07-19 refinement
    landed in both (`_contract` REAL-CHANGE@ext / `_monitor` inf-input-null); the
    2026-08-11 singular scoping does the same (`_contract` REAL-CHANGE@singular /
    `_monitor` singular-real-null). A divergence would mean two different contracts."""

    def test_contract_twin_detects_the_same_singularity(self):
        from mpmath import mpf

        from simplipy.verify import _contract as C
        tree = ('inv', ('-', ('slot', 'a'), ('slot', 'b')))       # 1/(a-b)
        assert C.lhs_singular(tree, {'a': mpf(2), 'b': mpf(2)})   # on the pole
        assert not C.lhs_singular(tree, {'a': mpf(2), 'b': mpf(5)})
        assert C.lhs_singular(('log', ('slot', 'a')), {'a': mpf(0)})
        assert not C.lhs_singular(('+', ('slot', 'a'), ('lit', 2.0)), {'a': mpf(3)})

    def test_contract_twin_does_not_taint_a_written_infinity(self):
        from simplipy.verify import _contract as C
        # a WRITTEN inf literal keeps its ratified limit-completed value, so the
        # strict clause retains authority over it -- same boundary as the monitor's.
        assert not C.lhs_singular(('atan', ('lit', float('inf'))), {})

    def test_contract_twin_leaves_no_global_state(self):
        from mpmath import mpf

        from simplipy.verify import _contract as C
        C.lhs_singular(('inv', ('-', ('slot', 'a'), ('slot', 'a'))), {'a': mpf(1)})
        assert C._NODE_SINK == [], 'taint sink leaked -- would poison later evaluations'

    def test_both_instruments_name_the_class(self):
        import inspect

        from simplipy.verify import _contract as C
        from simplipy.verify import _monitor as M
        assert 'REAL-CHANGE@singular' in inspect.getsource(C.judge_rule)
        assert 'singular-real-null' in M.TOL_COUNTS


class TestMagnitudeResolutionGuard:
    """F93 (2026-08-11): the snap ladder's shrink test reads a large rung-to-rung ratio
    as a dying precision residue. That inference needs each rung to have RESOLVED the
    value's magnitude. `exp(-1e309)` renders as 10^(-4.34e308) whose EXPONENT is only
    accurate to the working precision, so two rungs differ by ~10^(1e258) and the ratio
    says nothing -- the ladder snapped a provably positive value to '0'.

    Found because `test_snap_refuses_the_exp_erasure_span` passed in the full suite and
    failed alone: the fragile band is ambient dps 50-110, and whichever precision the
    previous test happened to leave behind decided the outcome."""

    ERASURE_SPAN = ['exp', '-1e309']

    @pytest.mark.parametrize('dps', [15, 50, 110, 120, 240, 500])
    def test_erasure_refused_at_every_ambient_precision(self, dps):
        from mpmath import mp
        from simplipy.verify._monitor import _span_exact_token
        old = mp.dps
        try:
            mp.dps = dps
            assert _span_exact_token(list(self.ERASURE_SPAN)) is None
            assert _snap_literal_exacts(['asin', 'exp', 'exp', '-1e309']) == \
                ['asin', 'exp', 'exp', '-1e309']
        finally:
            mp.dps = old

    @pytest.mark.parametrize('dps', [50, 120, 240])
    def test_legitimate_snaps_survive(self, dps):
        """The guard must not blunt the ladder's real job."""
        from mpmath import mp
        from simplipy.verify._monitor import _span_exact_token
        old = mp.dps
        try:
            mp.dps = dps
            assert _span_exact_token(['sin', 'np.pi']) == '0'
            assert _span_exact_token(['tan', 'np.pi']) == '0'
            assert _span_exact_token(['sinh', 'asinh', '2']) == '2'
        finally:
            mp.dps = old

    def test_erasure_no_longer_convicts_a_correct_rewrite(self):
        """The verdict-visible falsifier. `x1 + log(exp(-e^1000))` and `x1 - e^1000` are
        the SAME function; small literals keep the pair's own working precision inside
        the fragile band, so this is reachable through judge_pair. Unguarded, the span
        erased to '0', `log(0) = -inf` poisoned every draw, and the judge returned
        `positive-measure INF-CHANGE on 40/40 continuum draws` against a correct
        rewrite. The honest verdict is a REFUSAL: at this precision the value cannot be
        resolved, so UNSCORED -- never a fabricated conviction (the F74 doctrine)."""
        from mpmath import mp
        old = mp.dps
        try:
            mp.dps = 50
            verdict, detail = judge_pair(
                ['+', 'x1', 'log', 'exp', 'neg', 'exp', '1000'],
                ['+', 'x1', 'neg', 'exp', '1000'], rng())
            assert verdict != 'VIOLATION', f'fabricated conviction returned [{detail}]'
            assert verdict == 'UNSCORED', f'{verdict} [{detail}]'
        finally:
            mp.dps = old

    def test_predicate_boundaries(self):
        from mpmath import mp, mpf
        from simplipy.verify._monitor import _magnitude_resolved
        old = mp.dps
        try:
            mp.dps = 50
            assert _magnitude_resolved(mpf(0), 50)            # exact zero: nothing to place
            assert _magnitude_resolved(mpf('1e-51'), 50)      # a normal residue IS resolved
            assert not _magnitude_resolved(mp.exp(mpf('-1e309')), 50)
            assert not _magnitude_resolved(mpf('nan'), 50)    # fails closed
            assert not _magnitude_resolved(mpf('inf'), 50)
        finally:
            mp.dps = old
