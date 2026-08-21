"""Constructor-licence wall: extension-semantics soundness of the canonicalization layer.

The AC constructors rewrite spellings under algebraic identities that are theorems on the
reals but NOT under the contract's extension semantics (one zero, 1/0 = +inf, NaN
domains) once a degenerate configuration has POSITIVE measure. The 2026-07-29
constructor-licence audit (8-agent sweep + adversarial hunt, adjudicated by the
independent contract evaluator) confirmed three such gaps and this wall pins them shut:

1. pow over Mul, odd negative exponents: (a*b)^-1 = a^-1 * b^-1 flips the infinity sign
   wherever the product VANISHES with a negative co-factor -- fat for abs-style factors
   (1/(|x|-x) vs -1/(x-|x|): +inf vs -inf on the whole half-line). Licence: per-factor
   zero-set-null; refusal is confluence-normalized and serialization-stable.
2. symbolic-exponent occurrence merge: x^y * x^y -> x^(2y) extends NaN to defined values
   on {x<0} wherever y is pinned to a half-integer on positive measure (abs-pinned or
   identically-1/2 unfolded analytic identities). Licence: certainly-nonneg base or a
   nonconstant-entire exponent (disjoint interval pair, identity theorem).
3. scale_term's Const arm at r = 0: 0 * C must stay in the structural-zero class
   (masked constants are finite by doctrine); returning C laundered a kept-zero into
   the fittable constant class.

Known DOCUMENTED boundaries this wall deliberately does not convict: the f64-parity
constant fold (sin(np.pi) -> 1.2e-16, atan folds to the f64 pi/2 literal -- deployment
semantics, flagged by the exact judge at ~1e-17), and judge verdicts on rewrites that
TRANSFORM masked constants (the judge's shared-spelling binding assumes verbatim
pass-through).
"""
import os

import numpy as np
import pytest
import yaml

from simplipy import Mode
from simplipy.engine import SimpliPyEngine
from simplipy.verify._monitor import judge_pair
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


def rng():
    return np.random.default_rng(0)


def check(eng, tokens, expect_judge='OK'):
    """simplify, assert idempotence + the judge verdict; return the output.

    `simplify` answers in the DIALECT it was handed since the conversion split, so a
    site whose expectation is the canonical TAGGED spelling converts its input first
    (`check(eng, eng.to_tagged(...))`) -- the exact migration for `form='tagged'`."""
    o1 = eng.simplify(list(tokens))
    o2 = eng.simplify(list(o1))
    assert o1 == o2, f'not idempotent: {tokens} -> {o1} -> {o2}'
    v, d = judge_pair(list(tokens), list(o1), rng())
    assert v == expect_judge, f'{tokens} -> {o1}: {v} ({d})'
    return o1


class TestPowDistributionLicence:
    def test_fat_zero_inversion_keeps_infinity_sign(self, eng):
        # the 2-ppm 1M finding: refusing the -1 pull-out keeps 1/(|x|-x) at +inf
        for sk in ('/ 1 - abs x0 x0', 'inv - abs x0 x0', '/ x5 - abs x16 x16',
                   '/ x1 - x0 abs x0'):
            check(eng, sk.split())

    def test_generic_distribution_preserved(self, eng):
        # nonzero-a.e. factors (null zero sets) still distribute: canonical compression
        # of the sound generic case must not regress
        # explicit dialect in, explicit dialect out: the distribution fired (the
        # reciprocal took the -1 coefficient); `neg inv x0` is the tagged twin.
        out = check(eng, 'inv neg x0'.split())
        assert out == ['/', '-1', 'x0']

    def test_f83_odd_negative_power_magnitude_guard(self, eng):
        # F83 (owner-ruled NARROW 2026-08-11): sound mode refuses distributing an
        # odd negative power over a product with a MAGNITUDE-carrying negative
        # coefficient (|c| != 1) when a non-Num factor may vanish -- the 508487
        # conviction: at x0 = 0 the source 1/((-x0)/3) folds the product to the
        # unsigned zero first, inv(0) = +inf, while the distributed -3/x0 lands on
        # -inf; through the row's exp/pow that was a finite 0-vs-1 change. The
        # pure-sign c = -1 case is SS9.8.4's own priced residue and stays licensed
        # (test_generic_distribution_preserved pins it).
        out = check(eng, 'pow / neg x0 3 -1'.split())
        # The REFUSAL, stated against the distributed spelling F83 forbids. Comparing it
        # to `inv / neg x0 3` compares two spellings of the SOURCE, which agree whether or
        # not F83 exists -- the test stayed fully green with the guard removed.
        assert out != eng.simplify('/ -3 x0'.split()), f'F83 did not refuse: {out}'
        assert out == eng.simplify('inv / neg x0 3'.split()), f'{out}'
        # arrival spellings of the kept atom converge on one canonical fixpoint
        assert eng.simplify('pow / * -1 x0 3 -1'.split()) == out
        # provably nonvanishing factors have no pole: distribution stays licensed
        lo = check(eng, 'pow * -3 exp x1 -1'.split())
        assert lo.count('inv') + lo.count('/') + lo.count('<div>') <= 1
        # LOSSY is licensed for a.e. changes and still distributes the magnitude
        assert eng.simplify('pow / neg x0 3 -1'.split(), mode=Mode.corpus) == \
            eng.simplify('/ -3 x0'.split(), mode=Mode.corpus)

    def test_asin_class_factors_are_licensed(self, eng):
        # zero-set-null is the licence, NOT finite-nonzero-a.e.: asin has a fat NaN
        # domain but a null zero set, so it distributes (the Row-2 confluence case)
        # The DISTRIBUTION is the guard; the 1/5 coefficient then renders on the
        # `<div>` side (5 is inside the tagged vocabulary bound) instead of as the
        # atomic `0.2`. Same value, spelling only.
        out = check(eng, eng.to_tagged('/ / x8 x16 * asin x8 * 5 x16'.split()))
        assert out == '<mul> x8 <div> 5 pow x16 2 asin x8 </mul>'.split()

    def test_refused_forms_are_serialization_stable(self, eng):
        # construction-order confluence + emission round-trips of REFUSED shapes
        # (exponent composed before/after the bag; Const-bearing refused denominators)
        for sk in ('/ / / x14 + * x17 x45 <constant> x51 - / * x14 <constant> 3 asin sin tan <constant>',
                   'pow <mul> x8 <div> x16 <mul> 5 x16 asin x8 </mul> </mul> 0.5'):
            o1 = eng.simplify(sk.split())
            assert eng.simplify(list(o1)) == o1


class TestSymbolicExponentMergeLicence:
    def test_abs_pinned_exponent_refused(self, eng):
        # y = 0.5 + (|x1|-x1) is 1/2 on the whole quadrant x1 >= 0: merging would
        # define x0^(2y) = x0 on {x0<0} where x0^y * x0^y is NaN
        check(eng, '* pow x0 + 0.5 - abs x1 x1 pow x0 + 0.5 - abs x1 x1'.split())

    def test_identically_half_analytic_exponent_refused(self, eng):
        # y = 0.5*(sin^2+cos^2) == 1/2 EVERYWHERE but never folds: the disjoint-pair
        # witness cannot certify nonconstancy, so the merge is refused
        check(eng, '* pow x0 * 0.5 + pow sin x1 2 pow cos x1 2 '
                   'pow x0 * 0.5 + pow sin x1 2 pow cos x1 2'.split())

    def test_generic_symbolic_exponent_still_merges(self, eng):
        # a bare-variable exponent is certifiably nonconstant entire: levels are null
        out = check(eng, eng.to_tagged('* pow x0 x1 pow x0 x1'.split()))
        assert out == 'pow x0 <mul> 2 x1 </mul>'.split()


class TestScaleTermZeroGuard:
    def test_zero_times_const_stays_structural_zero(self, eng):
        # the kept-zero flatten path must not launder 0*C into the fittable class
        out = check(eng, '+ * 0 + <constant> inv - x0 abs x0 x0'.split())
        assert '<constant>' not in out, f'structural zero laundered into Const: {out}'


class TestAbsElimination:
    """Constructor-level `abs t -> t` on a certainly-non-negative operand: a `_`-grade
    pointwise identity (abs(nan) = nan, abs(+inf) = +inf; one-zero contract makes signed
    zero non-normative). Living in the constructor makes the fold reachable from EVERY
    entry path -- the 7-31 row-diff found 9/65,536 (64k) + 25/1M rows where the F5-era
    `inv abs t -> abs inv t` orientation respelled |x^2| into |x^-2| BEFORE the
    exponent-literal abs rules could match, sticking a dead abs in the output. Serve-order
    shadowing of a constructor identity is structurally impossible."""

    def test_even_powers_shed_abs_at_any_exponent_sign(self, eng):
        # the shadowed class itself: |x^-2| (the mined rule only knew the literal 2)
        # explicit dialect: the negative exponent renders as the reciprocal
        # (`pow x0 -2` is the tagged twin). What this pins is that `abs` is GONE.
        assert check(eng, 'abs pow x0 -2'.split()) == ['inv', 'pow', 'x0', '2']
        assert check(eng, 'abs pow x0 2'.split()) == ['pow', 'x0', '2']
        # the 64k +7 row's core: x^2 * |x^-2| -> 1 (nonzero-a.e. exponent cancel)
        assert check(eng, '* pow x0 2 abs pow x0 -2'.split()) == ['1']

    def test_nonneg_base_powers_shed_abs(self, eng):
        # |1/|x||: pow(nonneg base, ANY exponent) is nonneg, so ONE of the two abs is
        # redundant and must fold. Which of the equal-complexity spellings survives
        # (1/|x| vs |1/x| -- identical functions) belongs to the serve ordering, not
        # this wall: shipped rules mined under the OLD canon morph at translation
        # (their sides re-canonicalize) and may re-orient the survivor.
        for sk in ('abs inv abs x0', 'abs / 1 abs x0'):
            out = check(eng, sk.split())
            assert out.count('abs') == 1, f'exactly one abs must survive: {sk} -> {out}'

    def test_range_positive_compositions_shed_abs(self, eng):
        # |cos(cos x)| = cos(cos x): cos of a [-1,1]-bounded argument is >= cos(1) > 0
        assert check(eng, 'abs cos cos x0'.split()) == ['cos', 'cos', 'x0']
        assert check(eng, 'abs cos sin x0'.split()) == ['cos', 'sin', 'x0']
        assert check(eng, 'abs cos tanh x0'.split()) == ['cos', 'tanh', 'x0']
        # |1/cos(cos x)|: the exact spelling the orientation shadow left behind
        out = check(eng, 'abs inv cos cos x0'.split())
        assert 'abs' not in out, f'dead abs survived: {out}'

    def test_odd_increasing_of_nonneg_sheds_abs(self, eng):
        # |sinh |x|| = sinh |x| = |sinh x| (odd increasing through 0 maps [0,inf) into
        # [0,inf)): one abs is redundant; the surviving spelling is the ordering's.
        for sk in ('abs sinh abs x0', 'abs atan abs x0'):
            out = check(eng, sk.split())
            assert out.count('abs') == 1, f'exactly one abs must survive: {sk} -> {out}'

    def test_unconditional_ranges_shed_abs(self, eng):
        assert check(eng, 'abs acos x0'.split()) == ['acos', 'x0']
        assert check(eng, 'abs exp x0'.split()) == ['exp', 'x0']
        assert check(eng, 'abs cosh x0'.split()) == ['cosh', 'x0']

    def test_bag_closure_sheds_abs(self, eng):
        # |x^2 + |y| + 2| and |2 |x| y^2|: sums/products of nonnegs are nonneg
        assert 'abs' not in check(eng, 'abs + + pow x0 2 abs x1 2'.split())[:1]
        out = check(eng, 'abs * 2 * abs x0 pow x1 2'.split())
        assert out[0] != 'abs', f'outer abs must fold: {out}'

    def test_sign_indefinite_operands_keep_abs(self, eng):
        # NEGATIVE controls: none of these operands is certainly non-negative
        for sk in ('abs x0',            # a bare variable
                   'abs sin x0',        # sin is sign-indefinite
                   'abs cos x0',        # cos of an UNBOUNDED argument
                   'abs pow x0 3',      # odd power
                   'abs tan abs x0',    # tan is NOT globally increasing (tan 2 < 0)
                   'abs + x0 1'):       # sum with a sign-indefinite term
            out = check(eng, sk.split())
            assert 'abs' in out, f'unsound fold: {sk} -> {out}'

    def test_const_abstraction_is_a_different_mechanism(self, eng):
        # abs(<constant>) -> <constant> is the Const channel's total-function
        # absorption (|C| is again "some fitted constant"), NOT the nonneg fold --
        # certainly_nonneg(Const) stays false (a fitted value can be negative).
        assert eng.simplify(['abs', '<constant>']) == ['<constant>']


class TestJudgePoleBand:
    def test_symbolic_cancellation_residue_is_not_a_conviction(self, eng):
        # log(exp(x)) - x folds to an exact zero denominator; the judge's finite
        # precision leaves a sign-noise residue and must skip, never convict
        # `sin x0` rather than a bare `x0`: the log-exp collapse is BANDED now, and
        # this test is about the judge's residue handling, not about the band. sin's
        # range is inside every band, so the vehicle still folds and the subject is
        # unchanged.
        o1 = eng.simplify('tanh / 1 - log exp sin x0 sin x0'.split())
        assert o1 == ['1']
        v, _ = judge_pair('tanh / 1 - log exp sin x0 sin x0'.split(), list(o1), rng())
        # the MEASURED verdict under the seeded rng (audit Tier-2, 2026-08-03): the judge
        # SCORES this pair and passes it. The old `in ('OK', 'UNSCORED')` tolerated a
        # judge that silently stopped scoring the row; a regression to UNSCORED means the
        # judge lost sight of it -- investigate, never widen the assertion back.
        assert v == 'OK', v


class TestCertificateCompletenessTowers:
    """Division-tower campaign (2026-08-03, owner-approved): the odd-negative-power
    distribution's per-factor licence is exactly as sharp as the true disagreement
    sets. A reciprocal factor's zero set is {A = +-inf} (`infinite_set_null`, NaN
    harmless), and the finiteness/zero-set recursions know the binary 0.12
    vocabulary (`pow`/`rootn` arms; the retired unary hyper-op tables no longer
    carry them). Each test is a specimen shape from the 884-row 64k falsifier set."""

    def test_polynomial_sum_tower_cancels(self, eng):
        # x8/(x8/(x8^2 - x8)) -> x8^2 - x8 a.e. (row-56772 shape). Blocked before by
        # nfn's fail-closure on binary pow inside the reciprocal factor's spelling.
        out = check(eng, '/ x8 / x8 - pow x8 2 x8'.split())
        assert out == eng.simplify('- pow x8 2 x8'.split()), f'tower kept: {out}'

    def test_pow5_reciprocal_pair_kept_sound_cancels_lossy(self, eng):
        # row-56488 fragment: x5^5/(-0.0009765625*x5^5*(x3-exp x10)^-5). Until F83
        # the x5^5 pair cancelled here in SOUND mode -- but that cancel is an a.e.
        # REPAIR: the source is NaN at x5 = 0 (0/0), the cancelled form finite. F83
        # (owner-ruled NARROW 2026-08-11: an odd negative power refuses to
        # distribute over a product carrying a magnitude negative coefficient,
        # |c| != 1, when a factor may vanish; the pure-sign c = -1 residue stays
        # priced by SS9.8.4) closes the distribution route that exposed the
        # reciprocal factor, so sound mode keeps the pair, value-identical to the
        # source everywhere (judge OK via check). The repair is not lost: LOSSY is
        # licensed for a.e. changes, still distributes, and the cancel fires there.
        src = '/ pow x5 5 * -0.0009765625 * pow x5 5 pow - x3 exp x10 -5'.split()
        out = check(eng, src)
        assert 'x5' in out, f'pair unexpectedly cancelled in sound mode: {out}'
        lossy = eng.simplify(list(src), mode=Mode.corpus)
        assert 'x5' not in lossy, f'lossy cancel lost: {lossy}'
        assert lossy == eng.simplify('* -1024 pow - x3 exp x10 5'.split(),
                                     mode=Mode.corpus), f'{lossy}'

    def test_fat_nan_acosh_tower_cancels(self, eng):
        # x8/(x8/(x8 - acosh(2 x8))) (row-9624 shape): A has a FAT NaN domain but is
        # never +-inf, and NaN agrees on both spellings -- isn admits what nfn refused.
        out = check(eng, '/ x8 / x8 - x8 acosh * 2 x8'.split())
        assert out == eng.simplify('- x8 acosh * 2 x8'.split()), f'tower kept: {out}'

    def test_log_tower_cancels(self, eng):
        # x8/(x8/(x8 - log x8)) (row-40856 shape): log's infinities sit on
        # {x8 = 0} u {x8 = +inf} -- null; its fat NaN half-line is harmless.
        out = check(eng, '/ x8 / x8 - x8 log x8'.split())
        assert out == eng.simplify('- x8 log x8'.split()), f'tower kept: {out}'

    def test_bare_fat_zero_factor_still_refused(self, eng):
        # POISON: 1/(x9*(x8-|x8|)) -- the BARE sum factor has a fat zero set (the
        # audit's infinity-sign-flip class); the sharpened licence must still refuse
        # the distribution: exactly one reciprocal, never a split product of them.
        out = check(eng, 'inv * x9 - x8 abs x8'.split())
        assert 'abs' in out
        assert out.count('inv') + out.count('/') == 1, f'unsound distribution: {out}'

    def test_zero_kills_pole_bearing_factor(self, eng):
        # 0 * sinh(0.25*x5^2*(x17+cos x17)/x6) -> 0 (row-481 shape): the argument's
        # only nonfinite points sit on the null pole {x6 = 0}; the structural
        # finiteness recursion needed the binary-pow arm to see it.
        out = check(eng, '* 0 sinh / * 0.25 * pow x5 2 + x17 cos x17 x6'.split())
        assert out == ['0'], f'kept-zero survived: {out}'


class TestInfiniteTermAbsorption:
    """Class C (owner-approved): a sum term that is an infinity-carrying product is
    never finite (its values are +-inf or NaN), so every finite-a.e. co-term is
    absorbed: F + inf*u = inf*u wherever F is finite (F = +-inf/NaN is null under
    the !-licence; at a NaN of the product both sides are NaN)."""

    def test_finite_term_absorbed(self, eng):
        out = eng.simplify('+ x5 * float("inf") x5'.split())
        assert out == eng.simplify('* float("inf") x5'.split()), f'{out}'
        assert out == eng.simplify(list(out)), f'not idempotent: {out}'

    def test_const_absorbed(self, eng):
        out = eng.simplify('+ <constant> * float("inf") x5'.split())
        assert '<constant>' not in out, f'{out}'

    def test_unlicensed_term_kept(self, eng):
        # POISON: log(x5) is NaN on a fat half-line where inf + log(x5) is NaN, not
        # the infinite product -- the bag must keep both terms.
        out = eng.simplify('+ log x5 * float("inf") x5'.split())
        assert 'log' in out, f'unsound absorption: {out}'
        assert out == eng.simplify(list(out)), f'not idempotent: {out}'


class TestRootnIsCertainlyNonNegative:
    """B1(c): `certainly_nonneg` had no `rootn` arm, so an even root was never recognised
    as non-negative and eight licences quietly weakened.

    The gap was invisible on the shipped engine because the RULESET masked it for whatever
    shapes the mine happened to cover -- `abs(rootn(x,2))` folded by RULE. These tests use a
    RULE-FREE engine, where only the licence can act. The four `pow1_*` spellings that stood
    in `rootn`'s place were dead: `from_prefix` maps them to `rootn` unconditionally.
    """

    @pytest.fixture(scope='class')
    def licence_only(self):
        import yaml
        from conftest import require_or_skip
        require_or_skip(CONFIG, 'acj-4-3 config not staged')
        cfg = yaml.safe_load(open(CONFIG))
        return SimpliPyEngine(operators=cfg['operators'], rules=[])

    def test_even_root_is_non_negative(self, licence_only) -> None:
        # a range fact, exactly like `acos`: an even root is defined only on [0, inf) and
        # its principal value is non-negative there (off-domain it is NaN, which no licence
        # reads as a sign)
        for n in ('2', '4', '10'):
            assert licence_only.simplify(['abs', 'rootn', 'x0', n]) == ['rootn', 'x0', n]

    def test_odd_root_follows_its_argument(self, licence_only) -> None:
        # an odd root is an increasing odd bijection: non-negative exactly where its
        # argument is, so the `abs` must SURVIVE on a free variable...
        out = licence_only.simplify(['abs', 'rootn', 'x0', '3'])
        assert out[0] == 'abs', out
        # ...and fold once the argument is certainly non-negative.
        assert licence_only.simplify(['abs', 'rootn', 'abs', 'x0', '3']) == \
            ['rootn', 'abs', 'x0', '3']

    def test_the_folds_are_value_exact(self, eng) -> None:
        # every shape above, on the H-057 grid: negatives, zero, both poles and nan
        import simplipy
        X = np.array([-8., -2., -.5, 0., .5, 2., 8., np.inf, -np.inf, np.nan])

        def ev(tokens):
            code = eng.prefix_to_infix(list(tokens), realization=True)
            with np.errstate(all='ignore'):
                return np.asarray(eval(code, {'simplipy': simplipy, 'np': np}, {'x0': X}),
                                  dtype=float) + np.zeros(len(X))

        for tokens in (['abs', 'rootn', 'x0', '2'], ['abs', 'rootn', 'x0', '4'],
                       ['abs', 'rootn', 'abs', 'x0', '3']):
            a = ev(tokens)
            b = ev(eng.simplify(list(tokens)))
            assert bool(((np.isnan(a) & np.isnan(b))
                         | np.isclose(a, b, rtol=1e-9, atol=1e-12)).all()), tokens


class TestPowerOfExpIsExp:
    """A POWER OF `exp` IS AN `exp`: `(e^a)^b = e^(a*b)`, wherever the product FOLDS.

    B1(d) was a WILDCARD-exponent rule going blind because the constructor got there
    first: `pow np.e _0 -> exp _0` is a mined RULE, but the constructor rewrites
    `pow(e, 1/n)` into `rootn(e, n)` BEFORE it can match, so `sqrt(e)` shipped as
    `rootn np.e 2` at mu 18,000 where `exp 0.5` is 10,585.

    The first fix (2026-08-07) was an arm for the LEAF `e` alone. The rule-reachability
    census written to check it (the research harness) then found the same
    mechanism still live on the very next base: `pow exp (-1) _0 -> exp neg _0` is also
    shipped, also blinded by the same rewrite, and `pow(exp(-1), 1/2)` was shipping as
    `rootn(exp(-1), 2)` at mu 20,000 against 11,585. One base is a patch; the family is
    the principle, and `e` is simply the `a = 1` instance of it.

    The fold condition -- the composed exponent must be a single node, i.e. a literal
    against a literal, or `a = 1` against anything -- is simultaneously the mu gate (a
    symbolic `a` makes the composition a tie through `pow` and an outright ASCENT through
    `rootn`) and the soundness gate (it is what excludes `0 * inf`, the one configuration
    where the two sides disagree). See `compose_e_power` in `rust/ac/expr.rs`.
    """

    def test_e_leaf_both_spellings_and_both_signs(self, eng) -> None:
        for n, want in (('2', '1/2'), ('3', '1/3'), ('4', '1/4'),
                        ('-2', '-1/2'), ('-8', '-1/8')):
            assert eng.simplify(['rootn', 'np.e', n]) == eng.simplify(['exp', want]), n
        # route-independence: the `pow` spelling lands in the same place
        assert eng.simplify(['pow', 'np.e', '0.5']) == eng.simplify(['rootn', 'np.e', '2'])
        assert eng.simplify(['pow', 'np.e', '1/3']) == eng.simplify(['rootn', 'np.e', '3'])

    def test_the_e_leaf_composes_at_any_exponent(self, eng) -> None:
        # `a = 1` needs no fold: the product IS the exponent, so the `E` leaf is dropped
        # for a symbolic exponent too -- and by the CONSTRUCTOR, not by the rule that
        # used to do it (`pow np.e _0 -> exp _0`, which this retires).
        bare = SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])
        assert bare.simplify(['pow', 'np.e', 'x0']) == ['exp', 'x0']
        assert bare.simplify(['pow', 'exp', '1', 'x0']) == ['exp', 'x0']
        assert bare.simplify(['inv', 'np.e']) == eng.simplify(['exp', '(-1)'])

    def test_an_exp_base_composes_when_both_exponents_are_literal(self, eng) -> None:
        for base_exp, outer, want in (('2', '3', '6'), ('(-1)', '0.5', '-1/2'),
                                      ('(-1)', '0.25', '-1/4'), ('3', '(-2)', '-6')):
            got = check(eng, ['pow', 'exp', base_exp, outer])
            assert got == eng.simplify(['exp', want]), (base_exp, outer)
        for base_exp, index, want in (('2', '3', '2/3'), ('(-1)', '2', '-1/2'),
                                      ('4', '(-2)', '-2')):
            got = check(eng, ['rootn', 'exp', base_exp, index])
            assert got == eng.simplify(['exp', want]), (base_exp, index)
        # route-independence across the two spellings of one power
        assert (eng.simplify(['pow', 'exp', '(-1)', '0.5'])
                == eng.simplify(['rootn', 'exp', '(-1)', '2']))

    def test_the_canon_has_ONE_spelling_for_e(self, eng) -> None:
        # The WRITE half of what `compose_e_power` reads. Without it the composition
        # lands on `exp 1` (10,000) where `np.e` is 8,000, and the mine papers over the
        # difference one context at a time -- 42 such rules were in the artifact.
        bare = SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])
        assert bare.simplify(['exp', '1']) == ['np.e']
        for tokens in (['inv', 'inv', 'np.e'],            # composes to exponent 1
                       ['pow', 'exp', '(-1)', '(-1)'],
                       ['rootn', 'exp', '2', '2'],
                       ['pow', 'exp', '0.5', '2']):
            assert bare.simplify(list(tokens)) == ['np.e'], tokens
        # and in context, where the retired rules used to do it
        assert bare.simplify(['*', 'exp', '1', 'x0']) == bare.simplify(['*', 'np.e', 'x0'])
        assert bare.simplify(['/', 'x0', 'exp', '(-1)']) == bare.simplify(['*', 'np.e', 'x0'])
        # `exp(0) -> 1` is a literal EVALUATION. It used to stay with the rules, because
        # the constructor was forbidden to evaluate anything transcendental; under
        # mode-aware folding the f64 constructor does it directly, and exactly -- exp(0)
        # is 1 in f64 and in mathematics alike. `real` still leaves it to a rule, because
        # `real` folds only exact RATIONAL arithmetic.
        from simplipy import Mode
        bare._core.set_mode_rules('real', [])
        assert bare.simplify(['exp', '0'], mode=Mode.f64) == ['1']
        assert bare.simplify(['exp', '0'], mode=Mode.real) == ['exp', '0']
        # `exp(1) -> e` is a different thing and unchanged: it chooses between two exact
        # SYMBOLS for one value, so it is not an evaluation at all and is unconditional.
        assert bare.simplify(['exp', '1'], mode=Mode.real) == ['np.e']
        assert eng.simplify(['exp', '0']) == ['1']

    def test_it_is_a_descent(self, eng) -> None:
        # `e` leaf: 19,000 -> 11,585 under mu' (was 18,000 -> 10,585 pre-D38; the
        # rational exponent 1/2 pays the selector bit on both sides of the audit's
        # 7,415-lost comparison, so the descent and its margin are unchanged)
        assert eng.complexity(eng.simplify(['rootn', 'np.e', '2'])) < 12000
        # `exp(-1)` base: 21,000 -> 12,585, the sibling the census found
        assert eng.complexity(eng.simplify(['pow', 'exp', '(-1)', '0.5'])) < 13000
        # and the composition never lands on the dearer spelling of e: 10,000 -> 8,000
        # -> 4,000, `np.e`'s own entry in the symbol table (2026-08-21)
        assert eng.complexity(eng.simplify(['inv', 'inv', 'np.e'])) == 4000

    def test_a_symbolic_inner_exponent_is_REFUSED_by_the_fold_condition(self, eng) -> None:
        """The arm fires on the FOLD CONDITION -- `a*b` collapsing to a single node --
        and never on a mu comparison, which is why the refusal is stable while the
        prices under it are not.

        Under the flat unit the refusal also happened to be mu-justified: through `pow`
        the composition tied (26,000 = 26,000) and through `rootn` it ascended (26,000
        against 27,000 for `exp(x0/3)`). THE SYMBOL TABLE (2026-08-21) SPLIT THOSE TWO.
        `pow` still ties, but `rootn(exp x0, 3)` is now 21,000 against 19,000 -- a
        `rootn` head (6, elementary) is dearer than the `Mul` plus unit-fraction literal
        the composed spelling pays, where the flat unit priced them the other way round.
        So the arm now declines a 2,000 descent on the root spelling. That is a MISSED
        descent, not an unsound one, and the identity stays where it already lives: with
        the RULES, which is where the shipped wildcard rules are.
        """
        for tokens in (['pow', 'exp', 'x0', '3'], ['rootn', 'exp', 'x0', '3']):
            kept = eng.simplify(list(tokens))
            assert kept == tokens, tokens
        composed_pow = eng.simplify(['exp', '*', 'x0', '3'])
        assert eng.complexity(['pow', 'exp', 'x0', '3']) == eng.complexity(composed_pow)
        composed_root = eng.simplify(['exp', '/', 'x0', '3'])
        assert eng.complexity(['rootn', 'exp', 'x0', '3']) == 21000
        assert eng.complexity(composed_root) == 19000

    def test_an_INFINITE_inner_exponent_is_REFUSED_the_zero_times_inf_trap(self, eng) -> None:
        # `a = -inf, b = 0` is the one configuration where `(e^a)^b` and `e^(a*b)`
        # disagree: `0^0 = 1` against `e^nan = nan`, defined -> undefined, which §9.1's R2
        # forbids at ZERO measure tolerance.  `Ex::Num` is finite by construction, so the
        # fold condition excludes it without needing a licence -- pinned here because a
        # future widening of the gate would silently reopen it.
        bare = SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])
        for tokens in (['pow', 'exp', 'float("-inf")', 'x0'],
                       ['pow', 'exp', 'float("-inf")', '2'],
                       ['rootn', 'exp', 'float("-inf")', '2']):
            assert bare.simplify(list(tokens)) == tokens, tokens

    def test_it_does_not_fire_without_an_e_base_or_without_exp(self, eng) -> None:
        assert eng.simplify(['rootn', 'x0', '2']) == ['rootn', 'x0', '2']
        assert eng.simplify(['rootn', 'np.pi', '2']) == ['rootn', 'np.pi', '2']
        # a constructor arm has no automatic vocabulary gating, unlike the rules it
        # replaces: a config without `exp` must keep the root.
        cfg = yaml.safe_load(open(CONFIG))
        ops = {k: v for k, v in cfg['operators'].items() if k != 'exp'}
        no_exp = SimpliPyEngine(operators=ops, rules=[])
        assert no_exp.simplify(['rootn', 'np.e', '2']) == ['rootn', 'np.e', '2']
        assert no_exp.simplify(['pow', 'np.e', 'x0']) == ['pow', 'np.e', 'x0']


class TestEvenFunctionsAreSignBlind:
    """C1.19: an even function's argument is taken modulo sign, at any depth.

    The thirty mined rules this replaces (`abs asinh neg _0 -> abs asinh _0` and its 29
    siblings) are not thirty facts. They are TWO -- an odd function propagates a sign, an
    even one absorbs it -- times the fifteen {abs,cos,cosh} x {odd} pairs, enumerated
    because the engine had no name for either factor. And they only reached DEPTH 2: the
    mine caps source patterns at length 4, so `abs sin tanh neg _0` (five tokens) had no
    rule at all and did not reduce even at a variable.

    `sign_blind_rep` reads the canonical SIGN, never a `neg` NODE -- which is what makes it
    survive the constructor's own fold of `neg(3)` into `-3`, the fold that blinded every
    one of the thirty at every numeric binding.
    """

    @pytest.fixture(scope='class')
    def bare(self):
        """Rules-free: everything here must be the CONSTRUCTOR, not the artifact."""
        return SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])

    def test_a_carried_sign_is_dropped_in_every_spelling(self, bare) -> None:
        for tokens, want in ((['cos', 'neg', 'x0'], ['cos', 'x0']),
                             (['cos', '(-2)'], ['cos', '2']),
                             (['abs', 'neg', 'x0'], ['abs', 'x0']),
                             (['cosh', '*', '(-1)', 'x0'], ['cosh', 'x0'])):
            assert bare.simplify(list(tokens)) == want, tokens

    def test_it_propagates_through_odd_functions_at_any_depth(self, bare) -> None:
        # depth 2 (what the rules covered), then depths the mine cannot reach at all
        assert bare.simplify(['abs', 'asinh', '-3']) == ['asinh', '3']
        assert bare.simplify(['abs', 'sin', 'tanh', 'neg', 'x0']) == ['abs', 'sin', 'tanh', 'x0']
        assert bare.simplify(['cosh', 'abs', 'neg', 'sin', 'neg', 'x0']) == ['cosh', 'sin', 'x0']

    def test_an_abs_inside_is_a_no_op_and_is_stripped(self, bare) -> None:
        # the gap the MINER found: version one stripped `abs` only at the top, and the
        # re-mine minted `abs tanh abs ?0 -> abs tanh ?0` -- the same identity one level in
        assert bare.simplify(['cos', 'abs', 'x0']) == ['cos', 'x0']
        assert bare.simplify(['abs', 'tanh', 'abs', 'x0']) == ['abs', 'tanh', 'x0']
        assert bare.simplify(['cosh', 'tanh', 'abs', 'x0']) == ['cosh', 'tanh', 'x0']

    def test_integer_powers_pass_it_through(self, bare) -> None:
        assert bare.simplify(['abs', 'pow', 'neg', 'x0', '3']) == ['abs', 'pow', 'x0', '3']
        assert bare.simplify(['cos', 'pow', 'neg', 'x0', '2']) == ['cos', 'pow', 'x0', '2']

    def test_the_POLE_TRILEMMA_trap_is_refused(self, bare) -> None:
        # `inv` is odd on the reals minus the origin and NOT odd here: with one zero
        # (contract §9.2) `neg(0) = 0`, so `inv(neg 0)` and `neg(inv 0)` genuinely
        # disagree -- pinned as VALUES, because this is the reason `inv` has no parity.
        # §9.8.4 proves no total semantics with +inf != -inf keeps both pole symmetry and
        # absorption, and this contract keeps absorption.
        assert bare.simplify(['inv', 'neg', '0']) == ['float("inf")']
        assert bare.simplify(['neg', 'inv', '0']) == ['float("-inf")']
        # `abs(1/(-x)) -> abs(1/x)` DOES hold and does fire -- not through a parity claim
        # about `inv`, but because `pow` distributes the -1 out of the base first and the
        # bag-coefficient arm then drops it. The refusal above is about the POLE, and the
        # pole is where the two spellings differ; away from it nothing is given up.
        assert bare.simplify(['abs', 'inv', 'neg', 'x0']) == ['abs', 'inv', 'x0']

    def test_it_does_not_fire_on_a_non_even_head(self, bare) -> None:
        # `sin` and `exp` are not even, so a sign or an `abs` in their argument STAYS.
        assert bare.simplify(['sin', 'abs', 'x0']) == ['sin', 'abs', 'x0']
        assert bare.simplify(['exp', 'abs', 'x0']) == ['exp', 'abs', 'x0']
        # ...and the canon's own sign normalization is untouched: it still pulls a sign
        # INTO an odd function's literal argument, which is what made all thirty rules
        # blind there in the first place.
        assert bare.simplify(['sin', 'neg', '3']) == ['sin', '-3']
        assert bare.simplify(['exp', 'neg', '3']) == ['exp', '-3']


class TestInversePairsCollapse:
    """C1.19 family E: `f(g(t)) -> t` where the pair is an inverse pair and its licence holds.

    The census asked for ONE rule (`exp log abs _0 -> abs _0`). The artifact was carrying
    **101** identity-shaped inverse-pair rules -- the same picture family A showed, with the
    mine enumerating a fact per pair AND per literal (`asinh sinh 1`, `asinh sinh 2`, ...,
    `log exp 10`, `acosh cosh np.pi`).

    Every entry in the table, and every licence on it, is the CONTRACT JUDGE's verdict,
    obtained before the arm was written (`verify._contract.judge_rule`). The three groups
    below are that partition, and the refusals matter as much as the collapses.
    """

    @pytest.fixture(scope='class')
    def bare(self):
        return SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])

    def test_total_pairs_collapse_only_within_the_f64_BAND(self, bare) -> None:
        """Genuine bijections of the EXTENDED LINE -- and not of f64, which is why they
        now carry a magnitude band. `tanh` attains exactly 1.0 from 18.990341103219276
        and `exp` overflows at 709.782713, past which `atanh(tanh t)` is `inf` rather
        than `t`. Unbounded argument: refused. Provably bounded argument: collapses,
        because the band is a proof obligation, not a blanket ban."""
        from simplipy import Mode
        for pair in (['log', 'exp'], ['asinh', 'sinh'], ['sinh', 'asinh'],
                     ['atanh', 'tanh']):
            unbounded = pair + ['x0']
            assert bare.simplify(list(unbounded)) == unbounded, unbounded
            # LOSSY answers to no external evaluator, so it still collapses
            assert bare.simplify(list(unbounded), mode=Mode.corpus) == ['x0'], unbounded
            # sin's range is [-1, 1], inside every band
            bounded = pair + ['sin', 'x0']
            assert bare.simplify(list(bounded)) == ['sin', 'x0'], bounded

    def test_the_two_axes_are_INDEPENDENT_which_no_bool_could_express(self, bare) -> None:
        """`Cx` used to carry `lossy: bool`, which can only ask "may I skip
        certification?". That is the wrong question for a guard whose reason is f64
        saturation, and the two cases below show why one bit cannot answer both.

        The BAND axis: `atanh(tanh t) = t` for every real t, and the band exists only
        because f64 saturates (tanh attains 1.0 from 18.990341103219276, and atanh(1.0)
        is inf). So `real` and `corpus` collapse it and `f64` must not.

        The RECALL axis: `exp(log c)` needs c > 0, a certificate only the permissive mode
        skips. So `corpus` collapses it and `f64` AND `real` must not.

        `real` sits on the opposite side of each -- it relaxes the band and keeps the
        certificate -- so the two axes are genuinely independent and a single bool has no
        value that describes it."""
        from simplipy import Mode
        bare._core.set_mode_rules('real', [])   # an EMPTY real set: constructors only

        band = ['atanh', 'tanh', '30']          # 30 > 18.990341103219276
        # f64 does not merely REFUSE the collapse -- with folding it answers what f64
        # genuinely computes: tanh(30) attains exactly 1.0, so atanh of it is inf, and
        # `np.arctanh(np.tanh(30.0))` really is inf. Faithful, and still not `30`.
        assert bare.simplify(list(band), mode=Mode.f64) == ['float("inf")']
        assert bare.simplify(list(band), mode=Mode.real) == ['30']
        assert bare.simplify(list(band), mode=Mode.corpus) == ['30']

        recall = ['exp', 'log', '<constant>']   # needs <constant> > 0
        assert bare.simplify(list(recall), mode=Mode.f64) == recall
        assert bare.simplify(list(recall), mode=Mode.real) == recall
        assert bare.simplify(list(recall), mode=Mode.corpus) == ['<constant>']

        inside = ['atanh', 'tanh', '1']         # inside every band: all three agree
        for mode in (Mode.f64, Mode.real, Mode.corpus):
            assert bare.simplify(list(inside), mode=mode) == ['1']

    def test_real_is_equally_sound_not_more_permissive(self, bare) -> None:
        """Owner ruling R2, 2026-08-20: "exp(log(-3)) must be rejected under real but
        permitted under corpus. We do not want unsound simplifications in any of these
        two. That's what corpus is for."

        `f64` and `real` are two notions of SOUNDNESS under two arithmetics; only `corpus`
        trades soundness for recall. `exp(log c) = c` needs c > 0 -- at c = -3 the left
        side is nan and the right is -3, which is wrong in MATHEMATICS, not merely in
        floating point, so `real` must refuse it exactly as `f64` does."""
        from simplipy import Mode
        bare._core.set_mode_rules('real', [])

        # The RULE question: with a fitted `<constant>` the side condition cannot be
        # discharged, so `f64` and `real` both refuse and only `corpus` fires.
        unproven = ['exp', 'log', '<constant>']
        assert bare.simplify(list(unproven), mode=Mode.f64) == unproven
        assert bare.simplify(list(unproven), mode=Mode.real) == unproven
        assert bare.simplify(list(unproven), mode=Mode.corpus) == ['<constant>']

        # The owner's concrete example is GROUND, so no rule is involved at all: the
        # engine evaluates it exactly and `log(-3)` is nan. Every mode answers nan, and
        # the property that matters is the one none of them has -- none returns `-3`.
        ground = ['exp', 'log', '(-3)']
        for mode in (Mode.f64, Mode.real, Mode.corpus):
            got = bare.simplify(list(ground), mode=mode)
            assert got == ['float("nan")'], (mode, got)
            assert got != ['(-3)'], mode

    def test_acosh_of_cosh_is_ABS_not_the_identity(self, bare) -> None:
        # cosh is EVEN, so the collapse is |t|. The judge certifies this shape and KILLs
        # the identity one; getting it wrong would be a real change on t < 0.
        # BANDED: cosh overflows at 710.475860, so an unbounded argument is refused --
        # the shape assertion moves to arguments the band can prove.
        assert bare.simplify(['acosh', 'cosh', 'x0']) == ['acosh', 'cosh', 'x0']
        assert bare.simplify(['acosh', 'cosh', 'sin', 'x0']) == ['abs', 'sin', 'x0']
        assert bare.simplify(['acosh', 'cosh', '-3']) == ['3']

    def test_licensed_pairs_need_their_licence(self, bare) -> None:
        # exp o log: CERTIFIED under certainly_nonneg, KILL without it (bc-positive-measure,
        # because `exp(log t)` is NaN on t < 0 where t itself is defined).
        assert bare.simplify(['exp', 'log', 'abs', 'x0']) == ['abs', 'x0']
        assert bare.simplify(['exp', 'log', 'cosh', 'x0']) == ['cosh', 'x0']
        # ...including the shape the ARTIFACT's rules missed: an even power is non-negative
        # by the lattice's own Pow arm, but the mine never enumerated it.
        assert bare.simplify(['exp', 'log', 'pow', 'x0', '2']) == ['pow', 'x0', '2']
        assert bare.simplify(['exp', 'log', 'x0']) == ['exp', 'log', 'x0']
        # cos o acos / sin o asin / tanh o atanh: CERTIFIED under certainly_in_unit
        assert bare.simplify(['cos', 'acos', 'sin', 'x0']) == ['sin', 'x0']
        assert bare.simplify(['sin', 'asin', 'tanh', 'x0']) == ['tanh', 'x0']
        assert bare.simplify(['tanh', 'atanh', 'cos', 'x0']) == ['cos', 'x0']
        for tokens in (['cos', 'acos', 'x0'], ['sin', 'asin', 'x0'], ['tanh', 'atanh', 'x0']):
            assert bare.simplify(list(tokens)) == tokens, tokens

    def test_the_REAL_CHANGE_pairs_are_refused_outright(self, bare) -> None:
        # asin(sin 4) = pi - 4, NOT 4. The judge convicts these under clause (a) -- a real
        # change -- so they can never be arms in any licence. Pinned because they LOOK like
        # inverse pairs and the table would be wrong to include them.
        for tokens in (['asin', 'sin', 'x0'], ['acos', 'cos', 'x0'], ['atan', 'tan', 'x0']):
            assert bare.simplify(list(tokens)) == tokens, tokens

    def test_it_reaches_ground_arguments_and_through_the_e_power_arm(self, bare) -> None:
        # the census row was blind at a ground binding; and the two arms of today compose --
        # `pow(e, log 3)` becomes `exp(log 3)` by compose_e_power, then collapses here.
        assert bare.simplify(['exp', 'log', 'abs', '/', '1', '3']) == \
            bare.simplify(['/', '1', '3'])
        assert bare.simplify(['log', 'exp', '3']) == ['3']
        assert bare.simplify(['pow', 'np.e', 'log', '3']) == ['3']


class TestReciprocalBaseAndDeterminedInfinitePower:
    """C1.19 family D: a literal base absorbs the exponent's sign, and a literal base at an
    infinite exponent is determined.

    The judge SPLIT this family three ways, which is why it is not one arm: the literal-base
    flip and the infinite-exponent reciprocal are CERTIFIED, the SYMBOLIC-base versions and
    both `log(1/t)` orientations are only TOLERATED, and `cos log inv _0 -> cos log _0` is
    certified under `cos` but not under `abs` or `cosh` (at t = -inf, `log(1/t)` is -inf while
    `log t` is NaN, and only a function with no limit at infinity maps both to NaN). The
    TOLERATED ones stay rules -- same call as `tan o atan` in the inverse-pair table.
    """

    @pytest.fixture(scope='class')
    def bare(self):
        return SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])

    def test_a_literal_base_absorbs_the_sign(self, bare) -> None:
        assert bare.simplify(['pow', '2', 'neg', 'x0']) == bare.simplify(['pow', '0.5', 'x0'])
        assert bare.simplify(['inv', 'pow', '2', 'x0']) == bare.simplify(['pow', '0.5', 'x0'])
        assert bare.simplify(['pow', '3', 'neg', 'sin', 'x0']) == \
            bare.simplify(['pow', '/', '1', '3', 'sin', 'x0'])
        assert bare.simplify(['pow', '0', 'neg', 'x0']) == ['pow', 'float("inf")', 'x0']

    def test_the_flip_is_REFUSED_where_it_would_ascend(self, bare) -> None:
        # The direction genuinely turns: the reciprocal costs the base one bit (§10.10), so
        # the flip pays only if it buys back a whole node. With two or more members the Mul
        # survives the flip and nothing is bought -- 34,000 against 34,585.
        kept = bare.simplify(bare.to_tagged(['pow', '2', 'neg', '*', 'x0', 'x1']))
        assert kept == ['pow', '2', '<mul>', '-1', 'x0', 'x1', '</mul>']
        assert bare.complexity(kept) <= bare.complexity(
            bare.simplify(['pow', '0.5', '*', 'x0', 'x1']))

    def test_a_symbolic_base_is_refused(self, bare) -> None:
        # judge: TOLERATED, not CERTIFIED -- a rule-level allowance, not a constructor one
        for tokens in (['pow', 'x1', 'neg', 'x0'], ['pow', 'inv', 'x1', 'x0']):
            assert bare.simplify(list(tokens)) == tokens, tokens

    def test_reciprocal_base_at_an_infinite_exponent(self, bare) -> None:
        assert bare.simplify(['pow', 'inv', 'x0', 'float("inf")']) == \
            ['pow', 'x0', 'float("-inf")']
        assert bare.simplify(['pow', 'inv', 'x0', 'float("-inf")']) == \
            ['pow', 'x0', 'float("inf")']

    def test_a_literal_base_at_an_infinite_exponent_is_determined(self, bare) -> None:
        # the half that was MISSING (everything going to zero); the miner found it by minting
        # 18 ground rules once the reciprocal arm removed the wildcard that reached through
        assert bare.simplify(['pow', '0.5', 'float("inf")']) == ['0']
        assert bare.simplify(['pow', '2', 'float("-inf")']) == ['0']
        assert bare.simplify(['pow', '(-0.5)', 'float("inf")']) == ['0']
        # the half that already worked
        assert bare.simplify(['pow', '2', 'float("inf")']) == ['float("inf")']
        assert bare.simplify(['pow', '0.5', 'float("-inf")']) == ['float("inf")']
        # deliberately OUT: |b| = 1, b = 0, and every symbolic base (the spike-flattener's
        # subject -- a symbolic base can straddle the step)
        # (the literal re-spells `(-1)` as `-1`; the point is that it does not FOLD)
        assert bare.simplify(['pow', '(-1)', 'float("inf")']) == ['pow', '-1', 'float("inf")']
        for tokens in (['pow', '0', 'float("inf")'], ['pow', 'x0', 'float("inf")']):
            assert bare.simplify(list(tokens)) == tokens, tokens

    def test_the_two_paid_for_exclusions(self, bare) -> None:
        # `-inf` as a base: judge KILLs `pow (-inf) neg _0 -> pow 0 _0` (bc-positive-measure,
        # `(-inf)^-t` is NaN at non-integer t while `0^t` is 0). Written in from symmetry
        # WITHOUT asking, caught by the h045 wall -- pinned so it cannot come back.
        assert bare.simplify(['pow', 'float("-inf")', 'neg', 'x0']) == \
            ['pow', 'float("-inf")', 'neg', 'x0']
        # `+inf` as a base IS certified, but belongs to the determined-reciprocal arm's LOSSY
        # gate: `inv inf` must survive as a composite so the $-certificate finds its partner.
        assert bare.simplify(['pow', 'float("inf")', 'neg', 'x0']) == \
            ['pow', 'float("inf")', 'neg', 'x0']


class TestHalfPeriodShift:
    """C1.19 family C: `f(t +- pi)` drops the pi, for f in {sin, cos, tan}.

    Eleven pattern rules said this and every one was blind at a LITERAL argument, for two
    DIFFERENT reasons that no rule could fix:

      * sin/cos sit at the `_` sort and went blind at a NEGATIVE literal, because the
        constructor factors a bag's common sign: `(-2) - pi` is stored as `neg(2 + pi)`, so
        the `-pi` member the pattern needs does not exist. Family A's mechanism again.
      * tan sits at `!`, and `bind_ok` refuses GROUND subjects outright -- a documented,
        accepted recall loss on certificate-carrying slots. `tan + 2 np.pi` could not bind
        `2` at all, so tan was blind at EVERY literal, both signs.

    Measured before the arm, over the 18 `{sin,cos,tan} x {+,-} x {x0, 2, (-2)}` states: cos
    reduced 6/6, sin 5/6, tan only its 2 variable cases.
    """

    @pytest.fixture(scope='class')
    def bare(self):
        """Rules-free: every reduction here must be the CONSTRUCTOR, not the artifact."""
        return SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])

    def test_the_whole_18_state_surface_reduces(self, bare) -> None:
        # the four that were BLIND before the arm are marked; the rest were already covered
        # by the eleven rules and must not regress now that the constructor owns them.
        want = {
            ('sin', '-', 'x0'): ['neg', 'sin', 'x0'],
            ('sin', '-', '2'): ['sin', '-2'],
            ('sin', '-', '(-2)'): ['sin', '2'],          # was BLIND (sign-factored bag)
            ('sin', '+', 'x0'): ['neg', 'sin', 'x0'],
            ('sin', '+', '2'): ['sin', '-2'],
            ('sin', '+', '(-2)'): ['sin', '2'],
            ('cos', '-', 'x0'): ['neg', 'cos', 'x0'],
            ('cos', '-', '2'): ['neg', 'cos', '2'],
            ('cos', '-', '(-2)'): ['neg', 'cos', '2'],
            ('cos', '+', 'x0'): ['neg', 'cos', 'x0'],
            ('cos', '+', '2'): ['neg', 'cos', '2'],
            ('cos', '+', '(-2)'): ['neg', 'cos', '2'],
            ('tan', '-', 'x0'): ['tan', 'x0'],
            ('tan', '-', '2'): ['tan', '2'],             # was BLIND (`!` refuses ground)
            ('tan', '-', '(-2)'): ['tan', '-2'],         # was BLIND
            ('tan', '+', 'x0'): ['tan', 'x0'],
            ('tan', '+', '2'): ['tan', '2'],             # was BLIND
            ('tan', '+', '(-2)'): ['tan', '-2'],         # was BLIND
        }
        for (op, sign, arg), expect in want.items():
            got = bare.simplify([op, sign, arg, 'np.pi'])
            assert got == expect, (op, sign, arg, got, expect)

    def test_the_sign_goes_outside_at_a_variable_and_inside_at_a_literal(self, bare) -> None:
        # sin is ODD so -sin(t) = sin(-t) and the sign may sit on either side. The choice is
        # STRUCTURAL, not a mu guard (F41: mu guards in constructor arms break idempotence):
        # a literal absorbs the sign and loses a node; a variable cannot, and the WRAPPED form
        # is what an outer `neg` can cancel.
        # a literal of EITHER sign absorbs it into its own sign
        assert bare.simplify(['sin', '+', '2', 'np.pi']) == ['sin', '-2']
        assert bare.simplify(['sin', '-', '(-2)', 'np.pi']) == ['sin', '2']
        # a bare variable absorbs nothing -> wrap
        assert bare.simplify(['sin', '+', 'x0', 'np.pi']) == ['neg', 'sin', 'x0']
        # the composition that falsified build 1, which pushed the sign in ALWAYS and left
        # this at `neg sin neg x0`
        assert bare.simplify(['neg', 'sin', '+', 'x0', 'np.pi']) == ['sin', 'x0']
        # ...and the case that falsified build 2, which tested `rest is a literal` only and so
        # left `sin(pi - t)` at the uncollapsible `neg sin neg t`. An argument already CARRYING
        # a negative annihilates the sign, so this must land on `sin t` with no rule involved.
        assert bare.simplify(['sin', '-', 'np.pi', 'x0']) == ['sin', 'x0']
        assert bare.simplify(['tan', '-', 'np.pi', 'x0']) == ['tan', 'neg', 'x0']
        assert bare.simplify(['cos', '-', 'np.pi', 'x0']) == ['neg', 'cos', 'x0']

    def test_a_coefficient_other_than_plus_minus_one_does_NOT_strip(self, bare) -> None:
        # Scope guard. In a canonical bag like terms merge, so a coefficient-2 pi member is
        # 2pi-PERIODICITY -- genuine argument reduction modulo the period, which the owner
        # deferred. Keyed on +-1 the arm cannot reach it.
        for tokens in (['sin', '+', 'x0', '*', '2', 'np.pi'],
                       ['tan', '+', 'x0', '*', '3', 'np.pi'],
                       ['cos', '-', 'x0', '*', '2', 'np.pi']):
            got = bare.simplify(list(tokens))
            assert 'np.pi' in got, (tokens, got)
        # nor may a pi that is a FACTOR of another member be mistaken for a pi TERM: the
        # member is `pi*x1`, a two-factor product whose other factor is not the -1 coefficient
        # `is_unit_pi` requires, so the bag keeps it (and stays a bag -- tagged form).
        got = bare.simplify(bare.to_tagged(['sin', '+', 'x0', '*', 'np.pi', 'x1']))
        assert got == ['sin', '<add>', 'x0', '<mul>', 'np.pi', 'x1', '</mul>', '</add>'], got

    def test_only_the_three_trig_heads_shift(self, bare) -> None:
        # the arm is a half-period fact about sin/cos/tan and nothing else; their inverses and
        # the hyperbolics have no such identity and must keep the pi.
        for op in ('asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'exp', 'log'):
            got = bare.simplify([op, '+', 'x0', 'np.pi'])
            assert 'np.pi' in got, (op, got)


class TestTheBandAndTheRulesetAgree:
    """The band lives in the CONSTRUCTOR, but the same collapse also ships as loaded
    RULES, and the two are independent mechanisms: the constructor asks
    `certainly_in_band`, the ruleset asks the judge's realisation axis. Nothing makes
    them agree by construction, so today's agreement is a fact to be checked, not a
    property to be assumed -- and the owner's ruling is that mine and serve must be
    consistent in every way imaginable.

    Concretely, the audit found `simplify(inv(acosh(cosh(x0))), Mode.f64)` returning
    `abs(1/x0)` through a rule while `simplify(acosh(cosh(x0)), Mode.f64)` is refused by
    the banded constructor -- same mode, same call. At x = 800 the rule's answer is
    0.00125 and the truth is 0.0.
    """

    def test_no_acosh_cosh_collapse_is_served_in_f64(self) -> None:
        """The band lives in the constructor and refuses `acosh(cosh x)` in f64. If the
        RULESET performed the same collapse in the same mode, the two would contradict
        each other -- which is what the audit found on the pre-triple artifact, where
        `inv(acosh(cosh x0))` became `abs(1/x0)` through a rule while the constructor
        refused it. The triple resolves it by tiering them `real`; this is the tripwire
        that says so, and would fire if a future judge change tiered one `core`."""
        from conftest import require_triple_or_skip
        require_triple_or_skip()
        import json
        from conftest import acj_config_path, require_or_skip
        from simplipy.verify._contract import judge_rule
        require_or_skip(acj_config_path(), 'needs the shipped acj-4-3 ruleset')
        engine = SimpliPyEngine.from_config(acj_config_path())
        base = acj_config_path().replace('config.yaml', '')

        def collapses(rules):
            # `<constant>`-RHS rules are trivially satisfiable (the RHS constant is
            # INDEPENDENTLY fitted), so they say nothing about the band and are excluded.
            return [r for r in rules
                    if (r[0][:2] == ['acosh', 'cosh']
                        or (len(r[0]) >= 3 and r[0][1] == 'acosh' and r[0][2] == 'cosh'))
                    and r[1] != ['<constant>'] and '<constant>' not in r[0]]

        in_real = collapses(json.load(open(base + 'rules_real.json')))
        assert in_real, 'the family must exist in rules_real, or this test guards nothing'
        in_f64 = collapses(json.load(open(base + 'rules.json')))
        assert in_f64 == [], f'an acosh-cosh collapse is served in f64: {in_f64[:3]}'
        for lhs, rhs in in_real:
            tier = judge_rule(list(lhs), list(rhs), engine)['tier']
            assert tier == 'real', f'{" ".join(lhs)} -> {" ".join(rhs)} tiers {tier!r}'

    def test_the_collapse_really_is_wrong_in_f64(self) -> None:
        # NOTE: this asserts IEEE facts, not repository behaviour -- no change here can
        # turn it red. It is documentation of why the band exists, kept deliberately as
        # the numeric premise the tests above rest on.
        """Why the band exists at all: cosh overflows at 710.475860, so past it
        acosh(cosh(x)) is inf rather than |x| and any function of it follows."""
        import numpy as np
        with np.errstate(all='ignore'):
            assert np.isinf(np.cosh(800.0))
            assert 1.0 / np.arccosh(np.cosh(800.0)) == 0.0
        assert abs(1.0 / 800.0) == 0.00125


class TestCorpusTakesTheBestOfBoth:
    """`corpus` is the most capable mode, and that cannot be delivered by sequencing.
    Folding is bottom-up, so `tanh 30` becomes `1` before `atanh` ever sees it -- no
    "try real's arm first" inside a node recovers `atanh(tanh 30) -> 30`. Corpus runs
    the whole pipeline under BOTH disciplines and keeps the cheaper endpoint."""

    @pytest.fixture(scope='class')
    def bare(self):
        e = SimpliPyEngine(operators=yaml.safe_load(open(CONFIG))['operators'], rules=[])
        e._core.set_mode_rules('real', [])
        return e

    def test_corpus_keeps_the_structural_answer_when_it_is_cheaper(self, bare) -> None:
        """mu prices `30` at 4000 and `float("inf")` at 8000, so the unfolded endpoint
        wins. Folding corpus eagerly gave `inf` here -- a finite, mathematically true
        answer replaced by a non-finite token, in the mode that feeds flash-ansr
        training."""
        from simplipy import Mode
        assert bare.simplify(['atanh', 'tanh', '30'], mode=Mode.f64) == ['float("inf")']
        assert bare.simplify(['atanh', 'tanh', '30'], mode=Mode.real) == ['30']
        assert bare.simplify(['atanh', 'tanh', '30'], mode=Mode.corpus) == ['30']

    def test_corpus_still_folds_where_real_cannot(self, bare) -> None:
        """The other half: `real` leaves `asin(1e-8)` alone because no literal spells the
        true value, and the f64 endpoint is cheaper, so corpus takes it."""
        from simplipy import Mode
        assert bare.simplify(['asin', '1e-08'], mode=Mode.real) == ['asin', '0.00000001']
        assert bare.simplify(['asin', '1e-08'], mode=Mode.corpus) == ['0.00000001']

    def test_the_selection_is_STRICT_so_a_tie_goes_to_the_folded_form(self, bare) -> None:
        """POLICY, not derivation -- and it is enforced by the shape of the comparison
        rather than by a branch of its own.

        The dispatcher keeps the unfolded endpoint only when it is STRICTLY cheaper
        (`cu < cf`); every other case, tie included, takes the folded one. So the tie
        policy is not separately reachable: a search over 65 expressions whose real and
        f64 endpoints differ found NO mu tie at all. What is testable, and what actually
        carries the policy, is the strictness -- unfolded must win only on a strict
        descent, and folded must win when it is merely no worse.

        (An earlier version of this test claimed mu priced `* 2 asin 1e-8` and `2e-8`
        identically at 6170. Measured, they are 25,170 and 6,170 -- a 19,000 strict
        descent, so it exercised the ordinary arm and the stated numbers were wrong.)
        """
        from simplipy import Mode
        c = bare._core
        # folded strictly cheaper -> folded
        t = ['+', 'asin', '1e-08', 'asin', '1e-08']
        real_end = bare.simplify(list(t), mode=Mode.real)
        f64_end = bare.simplify(list(t), mode=Mode.f64)
        assert c.ac_complexity(f64_end) < c.ac_complexity(real_end)
        assert bare.simplify(list(t), mode=Mode.corpus) == f64_end

        # unfolded strictly cheaper -> unfolded, which is the only way it wins
        band = ['atanh', 'tanh', '30']
        real_band = bare.simplify(list(band), mode=Mode.real)
        f64_band = bare.simplify(list(band), mode=Mode.f64)
        assert c.ac_complexity(real_band) < c.ac_complexity(f64_band)
        assert bare.simplify(list(band), mode=Mode.corpus) == real_band

    def test_corpus_is_never_worse_than_either_parent(self, bare) -> None:
        """The property that makes 'most capable' checkable rather than aspirational."""
        from simplipy import Mode
        cases = [['atanh', 'tanh', '30'], ['asin', '1e-08'], ['tan', '1'],
                 ['+', 'tan', '1', 'tan', '1'], ['log', 'exp', '30'], ['exp', '0'],
                 ['+', 'asin', '1e-08', 'asin', '1e-08']]
        for t in cases:
            mu = {m: bare._core.ac_complexity(bare.simplify(list(t), mode=m))
                  for m in (Mode.f64, Mode.real, Mode.corpus)}
            assert mu[Mode.corpus] <= min(mu[Mode.f64], mu[Mode.real]), (t, mu)
