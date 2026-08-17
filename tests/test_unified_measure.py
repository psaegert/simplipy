"""Stage-2 battery: the unified simplicity measure (design/UNIFIED_SIMPLICITY_MEASURE.md).

RED-FIRST (written 2026-08-01 against the stage-1 build, where they fail): these tests
pin the ratified measure -- T1-T7 of the design doc, the mu-governed fold, and the
termination probe -- as behavior, before the measure lands.

The measure, as ratified (Sec. 2 of the design doc; costs in 1/8 units, exact u64):

  structural node (bag / Pow / Fun head)   8
  variable leaf, special (pi, e), inf/nan  8
  numeric literal p/q (the exact VALUE)    max(2, bitlen(|p|) + bitlen(q) - 1)
  bag coefficient slot                     the literal cost above (magnitude 1 free:
                                           a bare sign is not information -- "x - y is
                                           not more complex than x + y" stands)
  <constant>                               c_free = 1133  (DERIVED: contract 10.10(5))

The literal formula is the unique reading consistent with ALL worked examples in the
ratified doc: cost(2) = 2 (T4: mu(2x) = 18), cost(5/2) = 4 (T5: mu(2.5*pi) = 20),
52-bit dyadic ~ 105 (T1/T2), alphabet ints <= 8 cost <= 4 (T4's constraint), and
mu(0) = mu(1) = 2 (T3). The implicit denominator 1 is free; sign is free (|p|).

The ordering becomes (mu, canonical total order): the lit_size tier is ABSORBED into
mu (its bit-length content is mu's literal component), and well-foundedness carries
over: mu bounds the node count AND every literal's bit length, so each mu level holds
finitely many terms over the finite vocabulary.

The fold is governed by mu (owner 2026-07-31, option (a)): it fires exactly when the
result descends. Exact arithmetic keeps folding (2+3 -> 5, cos 0 -> 1: tiny results
are cheap); transcendental roundings refuse (exp 1 -> 2.718... is a ~105-unit literal
against a 10-unit symbolic state) -- which dissolves the stage-1 symbol/literal seam
(finding (b)): exp(1) stays symbolic. What does NOT change: the special-READ refusal
(f64 cannot certify that a special-bearing evaluation is exact, and mu cannot either
-- an f64-accidental short hit would be a value change; exact collapses stay MINED,
hiprec-certified rules), the determined-source clause (contract), and the e^C
exception (exact identity first). AMENDED 2026-08-08 (contract 10.11): the
Const-absorption licence no longer beats the gradient at bag sites -- C*pi -> C
absorbs like every ground; only non-bag positions (pi^C) still hold a special
against mu.
"""
import math
from fractions import Fraction

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

# The measure is carried in MILLI-BITS (owner ruling 2026-08-06): a literal's cost is the
# real quantity log2(1 + |n|), not a bit COUNT, so the ordering can tell 100 from 1000.
# The ratios are untouched -- one grammar symbol is still 8 bits, <constant> still 16
# symbols -- only the unit is finer, and it stays an INTEGER so no float enters a
# comparison.
MILLI = 1000
MU_SYM = 8 * MILLI
# <constant> is DERIVED, not chosen (contract 10.10(5) / H-054): the supremum of mu over
# f64 round-trip spellings (1131.931 bits, at 5.5605781537525765e-308) plus the sign bit,
# rounded up. The old 128 = 16 symbols was beaten eightfold by mu(1e308) = 1024.154.
MU_FREE = 1133 * MILLI


def L(n) -> int:
    """1000 * log2(1 + |n|), computed independently of the Rust bit-extraction.

    Splits off the integer part by bit length and takes the remaining mantissa -- which
    is exact in f64 -- through libm, so this agrees with the core without sharing its
    algorithm.
    """
    m = abs(int(n)) + 1
    if m <= 1:
        return 0
    b = m.bit_length()
    return round(MILLI * ((b - 1) + math.log2(m / (1 << (b - 1)))))


def mu_lit(value) -> int:
    """The spec's literal cost, computed independently of the Rust implementation.

    ONE RULE: a literal is written as a few integers and each costs L(n) -- an integer
    pays L(p) alone (its denominator is implicit) and a fraction pays L(p) + L(q), plus a
    bit for a negative sign, floored at two bits.

    There is NO decimal code (contract 10.10(1), H-055). mu is VALUE complexity, not
    description length, so the `m * 10^-k` spelling earns no discount and the minimum over
    codes is revoked: `mu(1000) = 9.967` and `mu(0.001) = 10.967`, so `mu(1/n) = mu(n) + 1`
    exactly for every n >= 2 -- L(1) = log2(2) is exactly one bit, and that emergent
    inversion bit is the multiplicative analogue of the additive sign bit.
    """
    f = Fraction(value)
    p, q = abs(f.numerator), f.denominator
    fraction = L(p) if q == 1 else L(p) + L(q)
    return max(fraction + (MILLI if f.numerator < 0 else 0), 2 * MILLI)


@pytest.fixture(scope='module')
def eng():
    return SimpliPyEngine.from_config(acj_config_path())


def c(eng, tokens):
    return eng._core.ac_complexity(list(tokens))


class TestMuValues:
    """The weight table, pinned against independently computed expectations."""

    def test_atoms(self, eng):
        assert c(eng, ['x0']) == MU_SYM
        assert c(eng, ['np.pi']) == MU_SYM
        assert c(eng, ['np.e']) == MU_SYM
        assert c(eng, ['<constant>']) == MU_FREE
        # The two-bit FLOOR covers 0, 1, -1 and 2 alike: L is below it there, and the
        # sign bit is invisible under it. Above the floor mu is the real quantity.
        assert c(eng, ['0']) == mu_lit(0) == 2 * MILLI
        assert c(eng, ['1']) == mu_lit(1) == 2 * MILLI
        assert c(eng, ['(-1)']) == mu_lit(-1) == 2 * MILLI
        assert c(eng, ['2']) == mu_lit(2) == 2 * MILLI
        assert c(eng, ['0.5']) == mu_lit('1/2') == 2585   # L(1) + L(2) = 1 + 1.585
        assert c(eng, ['2.5']) == mu_lit('5/2') == 4170   # L(5) + L(2) = 2.585 + 1.585

    def test_long_literal_pays_its_bits(self, eng):
        tok = '2.718281828459045'
        expected = mu_lit('2718281828459045/1000000000000000')
        assert expected > 90  # the ~105-unit class of the design doc
        assert c(eng, [tok]) == expected

    def test_t1_coefficient_preservation(self, eng):
        # mu(Mul[E, x]) = 24 strictly below mu(Mul[fl(e), x]): materializing a special
        # coefficient is a steep ascent -- the 6-rule fork is dead by arithmetic.
        sym = c(eng, ['*', 'np.e', 'x0'])
        mat = c(eng, ['*', '2.718281828459045', 'x0'])
        assert sym == 3 * MU_SYM
        assert mat == 2 * MU_SYM + mu_lit('2718281828459045/1000000000000000')
        assert sym < mat

    def test_t2_exp_pi_x_over_pow(self, eng):
        # The owner's original preference is the measure's theorem: 32 < ~121.
        assert c(eng, ['exp', '*', 'np.pi', 'x0']) == 4 * MU_SYM
        assert c(eng, ['exp', '*', 'np.pi', 'x0']) < c(
            eng, ['pow', '23.140692632779267', 'x0'])

    def test_t3_exact_collapse_is_a_descent(self, eng):
        assert c(eng, ['sin', 'np.pi']) == 2 * MU_SYM
        assert c(eng, ['sin', 'np.pi']) > c(eng, ['0'])

    def test_t4_collection_still_descends(self, eng):
        # x + x -> 2x: 24 > 18. Small integers are genuinely cheap (< variable cost).
        assert c(eng, ['*', '2', 'x0']) == 2 * MU_SYM + mu_lit(2)
        assert c(eng, ['pow', 'x0', '2']) == 2 * MU_SYM + mu_lit(2)
        for n in range(2, 9):
            assert mu_lit(n) < MU_SYM

    def test_t5_rational_special_boundary(self, eng):
        # 2.5*pi (20 units) sits far below its materialization (~105): stays symbolic.
        assert c(eng, ['*', '2.5', 'np.pi']) == 2 * MU_SYM + mu_lit('5/2')
        assert c(eng, ['*', '2.5', 'np.pi']) < c(eng, ['7.853981633974483'])

    def test_t6_const_is_the_priciest_atom(self, eng):
        assert c(eng, ['<constant>']) == MU_FREE
        assert MU_FREE > MU_SYM
        assert MU_FREE > mu_lit('2718281828459045/1000000000000000')

    def test_sign_costs_one_bit_in_atoms_and_slots(self, eng):
        # REVOKED 2026-08-06 (owner): the sign is NOT free. Without a bit for it,
        # mu(2) == mu(-2) -- the measure could not tell a number from its negation, which
        # a description length may not do. `Rat` normalises to q > 0 with the sign on the
        # numerator, so the bit is unambiguous and charged exactly once, in the atom.
        # A magnitude-1 coefficient is still a BARE SIGN carrying no literal, so `x - y`
        # keeps its old relation to `x + y`. The negation WRAPPER is separate: mu is
        # measured on the canonical TREE ("never on a serialization"), and `x - y` is
        # canonically Add[x, Mul[-1, y]] -- the wrapper Mul is a real structural node
        # at 8, so subtraction prices one grammar symbol above addition. (Delta from
        # the pre-mu table's blanket sign-free row; operationally inert -- the
        # ordering only compares EQUIVALENT states, and sign wrappers on the same
        # value canonicalize identically -- but pinned here so it is a decision, not
        # an accident.)
        assert c(eng, ['*', '-2.5', 'x0']) == c(eng, ['*', '2.5', 'x0']) + MILLI
        # mu(2) sits ON the floor while mu(-2) clears it, so the OBSERVED gap here is
        # L(2) + 1 - 2 = 0.585 bits, not the full bit -- the floor absorbs the rest.
        assert c(eng, ['pow', 'x0', '(-2)']) == c(eng, ['pow', 'x0', '2']) + mu_lit(-2) - mu_lit(2)
        assert mu_lit(-2) - mu_lit(2) == 585
        # mu(1) and mu(-1) both sit on the two-bit FLOOR, so the sign is invisible there.
        assert c(eng, ['(-1)']) == c(eng, ['1'])
        assert c(eng, ['-', 'x0', 'x1']) == c(eng, ['+', 'x0', 'x1']) + MU_SYM


class TestB22AstronomicSaturation:
    """B22 at the public surface: the ceiling is not a price, and no sum wraps.

    The exact linear schedule |scale| * log2(10) exhausts u64 at |scale| ~ 5.553e15.
    The old arms saturated there, so `1e5553000000000000` priced exactly 2^64 - 1,
    every larger scale (and every beyond-i64 exponent) collided on that one price,
    and the poisoned leaf overflowed the tree sums downstream. The knee schedule
    (rust/ac/expr.rs::MU_SCALE_KNEE) keeps every f64-reachable literal on the exact
    linear price -- the knee sits nine orders of magnitude past |scale| = 324 -- and
    orders the absurd by the exponent's own magnitude. The core-level property tests
    live in expr.rs (astronomic_literals_stay_ordered_and_sums_never_wrap); these
    pin the same facts through `ac_complexity`, the surface the mine and the gates
    actually price with.
    """

    def test_the_plan_acceptance_line(self, eng):
        # the B22 acceptance criterion, verbatim
        assert c(eng, ['1e5553000000000000']) != 2**64 - 1

    def test_astronomic_scales_stay_strictly_ordered(self, eng):
        ladder = ['1e308', '1e4294967296', '1e5553000000000000',
                  '1e9223372036854775807', '1e99999999999999999999']
        prices = [c(eng, [t]) for t in ladder]
        assert prices == sorted(set(prices)), (
            f'astronomic scales collided or inverted: {dict(zip(ladder, prices))}')

    def test_a_composite_never_prices_below_a_part(self, eng):
        part = c(eng, ['1e5553000000000000'])
        comp = c(eng, ['+', '1e5553000000000000', 'x0'])
        assert comp >= part
        assert comp >= c(eng, ['x0'])


class TestT7Termination:
    """The dense-literal chain ascends: Mul[3/2^k, x] grows in mu with k, so the
    chain cannot be a reduction sequence (the audit's O1 hang class)."""

    def test_dyadic_chain_strictly_ascends(self, eng):
        prev = None
        for k in range(1, 61):
            frac = Fraction(3, 2 ** k)
            tok = f'{frac.numerator}/{frac.denominator}'
            m = c(eng, ['*', tok, 'x0'])
            assert m == 2 * MU_SYM + mu_lit(frac)
            if prev is not None:
                assert m > prev, (k, m, prev)
            prev = m


class TestMuGovernedFold:
    """The fold fires exactly when it descends mu (owner: option (a))."""

    def test_exact_arithmetic_keeps_folding(self, eng):
        assert eng.simplify(['+', '2', '3']) == ['5']
        assert eng.simplify(['*', '2', '4']) == ['8']
        # The fold FIRES (the guard); the exact result 1/2 then takes its argmin
        # spelling, which for a 2^a denominator is the fraction, not `0.5`.
        assert eng.simplify(['/', '1', '2']) == ['<mul>', '1', '<div>', '2', '</mul>']

    def test_exact_transcendental_hits_keep_folding(self, eng):
        # cos 0 -> 1 and log 1 -> 0: tiny results are cheap, mu licenses the fold.
        assert eng.simplify(['cos', '0']) == ['1']
        assert eng.simplify(['log', '1']) == ['0']
        assert eng.simplify(['exp', '0']) == ['1']

    def test_transcendental_roundings_refuse(self, eng):
        # THE stage-2 behavior change: a ~105-unit literal never replaces a 10-unit
        # symbolic state. exp(1) stays symbolic -- the finding-(b) seam dissolves
        # (no state e - 2.718... can form, so no rule can serve on a conflated
        # literal reading).
        # REFRESH 2026-08-01: the shipped engine serves the SYMBOL rather than the
        # bare-engine refusal this pinned pre-refresh.
        # 2026-08-07: it is now the CONSTRUCTOR that serves it (`exp(1) -> E` in `fun`,
        # the write half of `compose_e_power`'s reading of `E` as `E^1`), so it is exact
        # and route-independent instead of arriving through the special-constant
        # respelling of an f64. The 2026-08-01 note claimed an artifact rule
        # `exp 1 -> np.e` was doing it; no such rule is in the artifact -- corrected here.
        assert eng.simplify(['exp', '1']) == ['np.e']

        from simplipy.io import load_config
        cfg = load_config(acj_config_path())
        assert SimpliPyEngine(operators=cfg['operators'], rules=[]).simplify(['exp', '1']) == ['np.e']
        out = eng.simplify(['sin', '(-1)'])
        assert out[0] == 'sin' and not any('.' in t for t in out), out
        assert '1.36787944117144233' not in eng.simplify(['+', '1', 'exp', '(-1)'])
        assert '0.36787944117144233' not in eng.simplify(['+', '1', 'exp', '(-1)'])

    def test_special_bearing_folds_still_refuse(self, eng):
        # NOT deleted with the licence: f64 cannot certify a special-bearing
        # evaluation exact, and neither can mu -- an accidental short hit would be
        # a silent value change. Exact collapses stay MINED rules.
        out = eng.simplify(['*', '2', 'np.pi'])
        assert 'np.pi' in out
        out = eng.simplify(['cos', 'np.e'])
        assert 'np.e' in out

    def test_policy_keeps_beating_mu_where_ratified(self, eng):
        # Contract 10.11 (2026-08-08) RETIRED the one bag-site policy that beat the
        # measure: C*pi -> C now follows mu's gradient like every ground absorption.
        assert eng.simplify(['*', '<constant>', 'np.pi']) == ['<constant>']
        # The surviving against-the-gradient refusal is positional, not special-wide:
        # pi^C keeps the special although mu(pow(pi, C)) > mu(C) -- pow is not an
        # absorption site (value set (0, inf), not "anything").
        out = eng.simplify(['pow', 'np.pi', '<constant>'])
        assert 'np.pi' in out and '<constant>' in out


class TestB10CfreeDerivation:
    """B10: c_free = 1133 is DERIVED, not asserted — and the docs now say so in four
    places (formal.md §5, the .tex, the design doc, expr.rs). This pin recomputes the
    derivation from first principles: the supremum over f64 round-trip spellings of
    the decimal model L(significand) + |decimal scale|·log2(10), plus one sign bit,
    ceiled. If the measure's literal model or the published number ever drifts, this
    recomputation and `engine.complexity(['<constant>'])` stop agreeing."""

    @staticmethod
    def _decimal_model_bits(x: float) -> float:
        s = repr(abs(x))
        if 'e' in s:
            mant, exp = s.split('e')
            exp = int(exp)
        else:
            mant, exp = s, 0
        intpart, _, frac = mant.partition('.')
        digits = (intpart + frac).lstrip('0')
        scale = exp - len(frac)
        stripped = digits.rstrip('0')
        scale += len(digits) - len(stripped)
        d = int(stripped or '0')
        return math.log2(1 + d) + abs(scale) * math.log2(10)

    def test_c_free_recomputes_from_the_f64_supremum(self, eng):
        sup, argmax = 0.0, None
        mantissas = (1.0, 1.5, 1.0 + 2**-52, 2.0 - 2**-52, 1.2345678901234567)
        for be in range(-1074, 1024):
            for m in mantissas:
                x = math.ldexp(m, be)
                if x == 0.0 or math.isinf(x):
                    continue
                b = self._decimal_model_bits(x)
                if b > sup:
                    sup, argmax = b, x
        # The recorded attained value reproduces the recorded supremum exactly.
        rec = self._decimal_model_bits(5.5605781537525765e-308)
        assert abs(rec - 1131.931) < 1e-3, rec
        # SAFETY direction -- the load-bearing assert: nothing in the sweep may price
        # ABOVE the recorded supremum, or 1133 is no longer a supremum and the
        # published constant is falsified. (The coarse 5-mantissa grid does not hit
        # the exact 17-digit argmax spelling; it must get within a bit of the family,
        # which is the anti-vacuity that the sweep reaches the extreme region at all.)
        assert sup <= rec + 1e-6, (sup, argmax)
        assert sup > rec - 1.0, (sup, argmax)
        derived_bits = math.ceil(rec + 1.0)  # + the sign bit, ceiled
        assert derived_bits == 1133
        assert c(eng, ['<constant>']) == derived_bits * MILLI
