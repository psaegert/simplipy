"""Stage-2 battery: the unified simplicity measure (design/UNIFIED_SIMPLICITY_MEASURE.md).

RED-FIRST (written 2026-08-01 against the stage-1 build, where they fail): these tests
pin the ratified measure -- T1-T7 of the design doc, the mu-governed fold, and the
termination probe -- as behavior, before the measure lands.

The measure, as ratified (Sec. 2 of the design doc; costs in 1/8 units, exact u64):

  structural node (bag / Pow / Fun head)   8   (SUPERSEDED 2026-08-21, see below)
  variable leaf, special (pi, e), inf/nan  8   (SUPERSEDED 2026-08-21, see below)
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

AMENDED 2026-08-21 (the SYMBOL TABLE): the two "8" rows above are gone. A description
length is not flat, and the flatness was minting degenerate rules -- `pi - 2` and
`asin(sin 2)` both priced 19 bits, so the mine preferred the composition. Each grammar
symbol now carries its own price, informed by the node census and rounded to whole bits
(rust/ac/expr.rs, the `MU_BITS_*` block); the module constants below mirror it.
Everything BEHAVIORAL in this file is unchanged -- the T-battery inequalities all still
hold, and T4's now holds with room it did not have at the design doc's first-draft
leaf price.

AMENDED AT D38/B2 (2026-08-17, mu'): the literal row is now the TWO-codeword price
(selector + min(rational, decimal-scientific) + sign -- see tests/test_mu_prime.py,
the falsifier suite that owns the codebook pins), and c_free re-derives to 67 bits
under the new codebook. Everything BEHAVIORAL below (fold governance, policy rows,
termination) is unchanged; the value pins in this file are re-earned under mu'.
"""
import math
from fractions import Fraction

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

# The measure is carried in MILLI-BITS (owner ruling 2026-08-06): a literal's cost is the
# real quantity log2(1 + |n|), not a bit COUNT, so the ordering can tell 100 from 1000.
# The unit stays an INTEGER so no float enters a comparison.
MILLI = 1000
# The SYMBOL TABLE (2026-08-21), mirroring rust/ac/expr.rs's `MU_BITS_*`. There is no
# single "one grammar symbol" price any more; 8 survives only as the UNIT the table is
# read against (`SIMPLIPY_MU_SYM`) and as the price of a head the table does not name.
MU_LEAF = 6 * MILLI
MU_ADD = 3 * MILLI
MU_MUL = 3 * MILLI
MU_POW = 3 * MILLI
MU_PI = 4 * MILLI
MU_E = 4 * MILLI
MU_FUN_ELEMENTARY = 6 * MILLI      # exp log abs sin cos tan rootn
MU_FUN_TRANSCENDENTAL = 8 * MILLI  # asin acos atan sinh cosh tanh asinh acosh atanh
MU_INF = 8 * MILLI                 # float("inf"), float("-inf"), float("nan")
# <constant> is DERIVED, not chosen (contract 10.10(5) / H-054), and re-derived under the
# mu' codebook at D38/B2 (2026-08-17): the supremum of mu' over f64 round-trip spellings
# (64.649 bits, at 8.9002954340287245e-308) plus the sign bit and the selector bit,
# rounded up. Under the pre-D38 fraction-only code the same construction gave 1133 --
# driven by the 10^324-scale denominators the decimal-scientific codeword now spells as
# a ~8.3-bit exponent. TestB10CfreeDerivation below re-earns the number from a sweep.
MU_FREE = 67 * MILLI


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

    mu' (D38/B2, 2026-08-17 -- this REVOKES contract 10.10(1)/H-055's fraction-only
    rule): the codebook offers TWO codewords behind a one-bit selector,

        mu'(value) = 1 + min( L(p) [+ L(q)],  max(2, L(m)) + L(|k|) ) + sign,

    the rational spelling against the decimal-scientific spelling (m, k) of the
    shortest exact decimal print -- available exactly when the denominator is
    2^a * 5^b (integers included: 1000 is the codeword (1, 3)). Each codeword total
    floors its mantissa at two bits and carries the sign bit; every priced literal
    pays the selector exactly once, so the codebook is a genuine prefix code.
    """
    f = Fraction(value)
    p, q = abs(f.numerator), f.denominator
    sign = MILLI if f.numerator < 0 else 0
    fraction = max((L(p) if q == 1 else L(p) + L(q)) + sign, 2 * MILLI)
    # the decimal-scientific codeword, when the value terminates in base ten
    a = b = 0
    d = q
    while d % 2 == 0:
        d //= 2
        a += 1
    while d % 5 == 0:
        d //= 5
        b += 1
    decimal = None
    if d == 1:
        if q == 1:
            m, k = p, 0
            while m and m % 10 == 0:
                m //= 10
                k += 1
        else:
            k = max(a, b)
            m = p * 2 ** (k - a) * 5 ** (k - b)
        if k > 0:
            decimal = max(2 * MILLI, L(m)) + L(k) + sign
    return MILLI + (fraction if decimal is None else min(fraction, decimal))


@pytest.fixture(scope='module')
def eng():
    return SimpliPyEngine.from_config(acj_config_path())


def c(eng, tokens):
    return eng._core.ac_complexity(list(tokens))


class TestMuValues:
    """The weight table, pinned against independently computed expectations."""

    def test_atoms(self, eng):
        assert c(eng, ['x0']) == MU_LEAF
        assert c(eng, ['np.pi']) == MU_PI
        assert c(eng, ['np.e']) == MU_E
        assert c(eng, ['float("inf")']) == MU_INF
        assert c(eng, ['<constant>']) == MU_FREE
        # THE LAW THAT FIXES pi AND e, which the census cannot: a named constant must
        # cost less than the cheapest expression denoting it, or writing the name is
        # never a descent. `acos(-1)` and `exp(1)` are those expressions.
        assert MU_PI < c(eng, ['acos', '(-1)']) == MU_FUN_TRANSCENDENTAL + mu_lit(-1)
        assert MU_E < MU_FUN_ELEMENTARY + mu_lit(1)
        # The two-bit FLOOR covers 0, 1, -1 and 2 alike: L is below it there, and the
        # sign bit is invisible under it. Every priced literal additionally pays the
        # mu' selector bit (D38), so the floor class sits at 3 bits total.
        assert c(eng, ['0']) == mu_lit(0) == 3 * MILLI
        assert c(eng, ['1']) == mu_lit(1) == 3 * MILLI
        assert c(eng, ['(-1)']) == mu_lit(-1) == 3 * MILLI
        assert c(eng, ['2']) == mu_lit(2) == 3 * MILLI
        assert c(eng, ['0.5']) == mu_lit('1/2') == 3585   # 1 + L(1) + L(2): rational wins
        assert c(eng, ['2.5']) == mu_lit('5/2') == 5170   # 1 + L(5) + L(2): rational wins

    def test_long_literal_pays_its_bits(self, eng):
        tok = '2.718281828459045'
        expected = mu_lit('2718281828459045/1000000000000000')
        # the design doc's ~105-unit class prices ~56 bits under mu' (the
        # decimal-scientific codeword: selector + L(2718281828459045) + L(15))
        assert 50 * MILLI < expected < 60 * MILLI
        assert c(eng, [tok]) == expected

    def test_t1_coefficient_preservation(self, eng):
        # mu(Mul[E, x]) = 24 strictly below mu(Mul[fl(e), x]): materializing a special
        # coefficient is a steep ascent -- the 6-rule fork is dead by arithmetic.
        sym = c(eng, ['*', 'np.e', 'x0'])
        mat = c(eng, ['*', '2.718281828459045', 'x0'])
        assert sym == MU_MUL + MU_E + MU_LEAF
        assert mat == MU_MUL + mu_lit('2718281828459045/1000000000000000') + MU_LEAF
        assert sym < mat

    def test_t2_exp_pi_x_over_pow(self, eng):
        # The owner's original preference is the measure's theorem: 32 < ~121.
        assert c(eng, ['exp', '*', 'np.pi', 'x0']) == (
            MU_FUN_ELEMENTARY + MU_MUL + MU_PI + MU_LEAF)
        assert c(eng, ['exp', '*', 'np.pi', 'x0']) < c(
            eng, ['pow', '23.140692632779267', 'x0'])

    def test_t3_exact_collapse_is_a_descent(self, eng):
        assert c(eng, ['sin', 'np.pi']) == MU_FUN_ELEMENTARY + MU_PI
        assert c(eng, ['sin', 'np.pi']) > c(eng, ['0'])

    def test_t4_collection_still_descends(self, eng):
        # x + x -> 2x and x*x -> x^2. Small integers are genuinely cheap (< a variable).
        assert c(eng, ['*', '2', 'x0']) == MU_MUL + mu_lit(2) + MU_LEAF
        assert c(eng, ['pow', 'x0', '2']) == MU_POW + MU_LEAF + mu_lit(2)
        for n in range(2, 9):
            assert mu_lit(n) < MU_LEAF
        # THE INEQUALITY ITSELF, which this test used to assert only the operands of.
        # The collected spellings are the ones `add`/`mul` build, so `ac_complexity`
        # cannot be handed the uncollected tree -- it canonicalizes on the way in --
        # but the uncollected price is a sum of table entries and is exactly what T4
        # compares against. The FIRST DRAFT of the symbol table failed here: a 2-bit
        # leaf priced `Add[x, x]` BELOW `2x`, because the census prices a node CLASS
        # while a leaf must also say WHICH variable.
        assert MU_MUL + mu_lit(2) + MU_LEAF < MU_ADD + 2 * MU_LEAF
        assert MU_POW + MU_LEAF + mu_lit(2) < MU_MUL + 2 * MU_LEAF

    def test_t5_rational_special_boundary(self, eng):
        # 2.5*pi (20 units) sits far below its materialization (~105): stays symbolic.
        assert c(eng, ['*', '2.5', 'np.pi']) == MU_MUL + mu_lit('5/2') + MU_PI
        assert c(eng, ['*', '2.5', 'np.pi']) < c(eng, ['7.853981633974483'])

    def test_t6_const_is_the_priciest_atom(self, eng):
        assert c(eng, ['<constant>']) == MU_FREE
        assert MU_FREE > max(MU_LEAF, MU_ADD, MU_MUL, MU_POW, MU_PI, MU_E,
                             MU_FUN_ELEMENTARY, MU_FUN_TRANSCENDENTAL, MU_INF)
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
        # at `MU_MUL`, so subtraction prices one Mul head above addition. (Delta from
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
        assert c(eng, ['-', 'x0', 'x1']) == c(eng, ['+', 'x0', 'x1']) + MU_MUL


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
            assert m == MU_MUL + mu_lit(frac) + MU_LEAF
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
        assert eng.simplify(eng.to_tagged(['/', '1', '2'])) \
            == ['<mul>', '1', '<div>', '2', '</mul>']

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
    """B10: c_free is DERIVED, not asserted — re-derived under the mu' codebook at
    D38/B2 (was 1133 under the fraction-only code). This pin recomputes the
    derivation from first principles: the supremum over f64 round-trip spellings of
    the mu' two-codeword model min(rational, max(2, L(m)) + L(|scale|)), plus one
    sign bit and one selector bit, ceiled. If the measure's literal model or the
    published number ever drifts, this recomputation and
    `engine.complexity(['<constant>'])` stop agreeing."""

    @staticmethod
    def _model_bits(x: float) -> float:
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
        if d == 0:
            return 2.0
        if scale >= 0:
            rational = math.log2(1 + d * 10 ** scale)
        else:
            f = Fraction(d, 10 ** -scale)
            rational = math.log2(1 + f.numerator) + math.log2(1 + f.denominator)
        rational = max(2.0, rational)
        if scale == 0:
            return rational
        scientific = max(2.0, math.log2(1 + d)) + math.log2(1 + abs(scale))
        return min(rational, scientific)

    def test_c_free_recomputes_from_the_f64_supremum(self, eng):
        sup, argmax = 0.0, None
        mantissas = (1.0, 1.5, 1.0 + 2**-52, 2.0 - 2**-52, 1.2345678901234567,
                     1.9999999999999998)
        for be in range(-1074, 1024):
            for m in mantissas:
                x = math.ldexp(m, be)
                if x == 0.0 or math.isinf(x):
                    continue
                b = self._model_bits(x)
                if b > sup:
                    sup, argmax = b, x
        # The recorded attained value reproduces the recorded supremum exactly: the
        # largest 17-digit mantissa any shortest repr can carry (the ulp-per-decimal-
        # grid feasibility bound at the 2^-1020 binade top) at the deepest scale a
        # 17-digit spelling reaches (-324), found by ulp-walking every binade top in
        # the boundary decades.
        rec = self._model_bits(8.9002954340287245e-308)
        assert abs(rec - 64.649) < 1e-3, rec
        # SAFETY direction -- the load-bearing assert: nothing in the sweep may price
        # ABOVE the recorded supremum, or 67 is no longer a supremum and the published
        # constant is falsified. (The coarse mantissa grid does not hit the exact
        # 17-digit argmax spelling; it must get within ~1.7 bits of the family, which
        # is the anti-vacuity that the sweep reaches the extreme region at all.)
        assert sup <= rec + 1e-6, (sup, argmax)
        assert sup > rec - 1.7, (sup, argmax)
        derived_bits = math.ceil(rec + 1.0 + 1.0)  # + the sign bit + the selector, ceiled
        assert derived_bits == 67
        assert c(eng, ['<constant>']) == derived_bits * MILLI


class TestTheSymbolTable:
    """The 2026-08-21 re-pricing: what the table was FOR, and what it must stay.

    mu' charged one flat 8 bits for every grammar symbol, and that flatness minted a
    degenerate family. `pi - 2 -> asin(sin 2)` is CORRECT (`asin(sin x) = pi - x` on
    [pi/2, pi]) and CERTIFIED, and mu' genuinely preferred the right-hand side --
    19.000 against 19.585 -- because two transcendental heads plus a literal cost
    exactly what a bag plus `pi` plus a literal cost. The whole parametrised family
    followed (`- k np.pi -> atan tan k`, `acos neg cos (-k)`).

    The table is HEURISTIC, informed by `ac_node_census` over the 400-row corpus, and
    two of its entries are fixed by laws the census cannot see. Both laws are pinned
    here, because both are load-bearing and neither is derivable from a frequency count.
    """

    @pytest.mark.parametrize('cheap,degenerate', [
        (['-', 'np.pi', '2'], ['asin', 'sin', '2']),
        (['-', 'np.pi', '1'], ['atan', 'tan', '1']),
        (['-', 'np.pi', '2'], ['acos', 'neg', 'cos', '(-2)']),
    ])
    def test_the_degenerate_family_is_dearer_than_what_it_replaces(
            self, eng, cheap, degenerate):
        assert c(eng, cheap) < c(eng, degenerate), (
            f'{" ".join(cheap)} = {c(eng, cheap)} is not below '
            f'{" ".join(degenerate)} = {c(eng, degenerate)}')

    def test_a_named_constant_costs_less_than_its_cheapest_denotation(self, eng):
        """A name that costs more than an expression for the same value is never worth
        writing, and the census cannot say so: `pi` and `e` occur ZERO times in a
        symbolic-regression skeleton corpus, so a frequency table prices them at its
        smoothing floor and makes the degenerate family STRONGER."""
        assert c(eng, ['np.pi']) < c(eng, ['acos', '(-1)'])
        # `exp 1` folds to `np.e` on the way in, so the denotation is priced from the
        # table rather than through the engine -- which is the point: the fold filter's
        # correctness rests on `exp 1 -> np.e` being a strict improvement, and under the
        # flat unit it was a TIE decided by the canonical tie-break instead.
        assert c(eng, ['np.e']) < MU_FUN_ELEMENTARY + mu_lit(1)

    def test_the_table_is_a_valid_code_over_a_real_vocabulary(self, eng):
        """Kraft, over the alphabet a tree position can hold: `sum 2^-cost <= 1`.

        Read PER SYMBOL, not per class -- a leaf must also say WHICH variable, and that
        is exactly the term the census omits (it counts the class `Leaf`, 3519 of 10,996
        nodes at 1.65 bits, while a specific variable among 18 is 5.82). The first draft
        of the table took the class number, and at a 2-bit leaf this inequality admits
        exactly ONE variable -- and T4's collection inequality inverts, which is the same
        defect showing up where it can be measured.

        The literal codebook is deliberately not in this sum: the table does not touch
        it, and `L(n) = log2(1+n)` was never summable over the integers anyway (a
        property mu inherited from `bits`, predating all of this).
        """
        # THE ALPHABET IS NODE KINDS, NOT CONFIG TOKENS. `-` is an inverted member of an
        # Add bag, `/` and `inv` are `Pow(., -1)`, `neg` is `Mul[-1, .]`; none of them is
        # a node, and counting them collides several spellings onto one code point and
        # blows the sum past 1 for no informational reason. Same trap the node census
        # documents (`ac_node_census`: serialized tokens are the wrong unit).
        STRUCTURE = {'+', '-', '*', '/', 'neg', 'pow', 'inv'}
        ELEMENTARY = {'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'rootn'}
        alphabet = [MU_ADD, MU_MUL, MU_POW, MU_PI, MU_E,
                    MU_INF, MU_INF, MU_INF, MU_FREE]
        for op in eng.operator_arity:
            if op in STRUCTURE:
                continue
            alphabet.append(MU_FUN_ELEMENTARY if op in ELEMENTARY
                            else MU_FUN_TRANSCENDENTAL)
        n_vars = 18   # x0..x17, the census corpus's own vocabulary
        mass = sum(2.0 ** (-b / MILLI) for b in alphabet)
        assert mass < 1.0, f'the fixed alphabet alone exhausts the code: {mass}'
        assert mass + n_vars * 2.0 ** (-MU_LEAF / MILLI) <= 1.0, (
            f'Kraft violated at {n_vars} variables: '
            f'{mass + n_vars * 2.0 ** (-MU_LEAF / MILLI)}')
