"""B2 falsifier suite: mu' -- the two-codeword numeric codebook as the NATIVE measure.

RED-FIRST (written 2026-08-17 against the mu build, where the pricing, fingerprint,
c_free and boundary-seam tests FAIL): these tests pin the ratified D38 mu' spec as
behavior BEFORE the measure lands.

The spec (D38, ratified -- SOOSE_INVESTIGATION_2026-08-17.md "mu' stated precisely"):
the numeric codebook gains a SECOND codeword, chosen by a one-bit selector:

    mu'(value) = 1 selector bit + min( L(p) + L(q),  L(m) + L(k) ) + sign bit

where (p, q) is the reduced rational spelling and (m, k) the decimal-scientific
spelling -- m the integer mantissa of the SHORTEST exact decimal spelling, k the
decimal exponent, L(k) = 1000*log2(1+|k|) milli-bits. A value has the decimal
codeword ONLY if its denominator is 2^a * 5^b (finite decimal expansion); otherwise
the rational codeword stands alone -- still + the selector, so every priced numeric
leaf pays exactly one selector bit and the codebook is a genuine prefix code.
Operators, structure, state semantics, the <constant> sup-construction RECIPE:
untouched. c_free' re-derives under the new codebook (~67 bits, was 1133).

THE SELECTOR-BIT DELTA, stated prominently: the D38 study rescorer
(the research harness) computed adjusted prices as min(old, dec) WITHOUT a
selector; the NATIVE mu' adds the 1-bit (1000 mB) selector to every numeric leaf
uniformly. Benchmark rescores under the study definition and native mu' therefore
differ by a constant 1000 mB per priced numeric leaf -- EXPECTED, not a bug. A second,
smaller divergence class is documented at TestIntegerScientificCodeword.

Oracle: dec_price() of the rescorer, ported exactly below (mantissa priced at the OLD
integer price max(2 bits, L(m)), + round(1000*log2(1+e)), + sign).
"""
import json
import math
import os
from fractions import Fraction

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

MILLI = 1000
SEL = 1 * MILLI          # the selector bit: one bit, once per priced numeric leaf
MU_SYM = 8 * MILLI       # the symbol UNIT; since 2026-08-21 a LEAF is priced separately
MU_LEAF = 6 * MILLI      # the symbol table's leaf entry (rust/ac/expr.rs MU_BITS_LEAF)
MU_FREE_PRIME = 67 * MILLI   # the re-derived c_free' (TestCfreePrimeDerivation earns it)

# The digest acj-4-3 was MINED under (its provenance sidecar; mu, pre-D38). The mu'
# engine must compute a DIFFERENT digest -- and still load the artifact (D25: warn,
# never refuse; the fingerprint is write-only at load).
#: The digest the PRE-0.14 artifact was mined under, kept as a historical value a
#: mismatch test can differ FROM.
ACJ_MINED_DIGEST = '84a2bc8eac4a0df1'

#: The digest the SHIPPED triple was mined under, read from its provenance sidecar. The
#: symbol table (2026-08-21) changed the measure, so the engine no longer computes this
#: and D25/R6 warns at load -- correctly, and until the re-mine under the new measure
#: replaces the artifact. `test_acj_load_warns_until_the_remine` below pins that the
#: warning is EXACTLY this one and nothing else, which is the only shape of allowance
#: that cannot swallow a genuine mismatch (the last blanket one nearly did).
ACJ_SHIPPED_DIGEST = '66ba8c4654ac9388'


def L(n) -> int:
    """1000 * log2(1 + |n|), computed independently of the Rust bit-extraction
    (same construction as test_unified_measure.L)."""
    m = abs(int(n)) + 1
    if m <= 1:
        return 0
    b = m.bit_length()
    return round(MILLI * ((b - 1) + math.log2(m / (1 << (b - 1)))))


def old_int_price(m: int) -> int:
    """What the pre-D38 engine charged for a bare non-negative integer token --
    the `c([str(mant)])` the rescorer's dec_price() calls."""
    assert m >= 0
    return max(2 * MILLI, L(m))


def old_rat_price(f: Fraction) -> int:
    """The pre-D38 mu literal price (the rescorer's `old`): fraction code + sign,
    floored at two bits."""
    p, q = abs(f.numerator), f.denominator
    raw = L(p) + (L(q) if q > 1 else 0)
    return max(2 * MILLI, raw + (MILLI if f.numerator < 0 else 0))


def oracle_dec_price(f: Fraction):
    """EXACT port of d38_muprime_rescore.dec_price on the reduced |value|, plus sign.

    None when the denominator has a prime factor other than 2 and 5 (no decimal
    codeword). For a REDUCED fraction the trailing-zero strip of the rescorer is
    provably dead (m = |p| * 2^(e-a) * 5^(e-b) is never divisible by 10), so this
    port needs no strip loop.
    """
    den = f.denominator
    a = b = 0
    d = den
    while d % 2 == 0:
        d //= 2
        a += 1
    while d % 5 == 0:
        d //= 5
        b += 1
    if d != 1:
        return None
    e = max(a, b)
    mant = abs(f.numerator) * (10 ** e // den)
    return old_int_price(mant) + round(MILLI * math.log2(1 + e)) + \
        (MILLI if f.numerator < 0 else 0)


def mu_prime_expected(f: Fraction) -> int:
    """The native mu' literal price: selector + min over the codewords the value has."""
    old = old_rat_price(f)
    dec = oracle_dec_price(f)
    return SEL + (old if dec is None else min(old, dec))


@pytest.fixture(scope='module')
def eng():
    return SimpliPyEngine.from_config(acj_config_path())


def c(eng, tokens):
    return eng._core.ac_complexity(list(tokens))


class TestSpellingInvariance:
    """Falsifier 1: mu' stays a function of the VALUE, across every spelling and
    across the Rat/numeric-string boundary."""

    def test_value_equal_spellings_price_equal(self, eng):
        assert c(eng, ['0.5']) == c(eng, ['/', '1', '2']) == c(eng, ['5e-1'])
        assert c(eng, ['0.2']) == c(eng, ['/', '1', '5']) == c(eng, ['2e-1'])
        assert c(eng, ['1.2']) == c(eng, ['/', '6', '5'])
        assert c(eng, ['(-0.2)']) == c(eng, ['/', '(-1)', '5'])

    def test_boundary_seam_closes(self, eng):
        # RED marker: under mu (H-055, min-over-codes revoked) these two spellings of
        # one value sat ONE MILLI-BIT apart (133880 vs 133879) -- a pinned, tolerated
        # residue. Under mu' both take the decimal codeword and the seam CLOSES to 0.
        e_notation = c(eng, ['1e-40'])
        as_fraction = c(eng, ['1/1' + '0' * 40])
        assert e_notation == as_fraction

    def test_states_are_spelling_free(self, eng):
        assert eng.simplify(['0.5']) == eng.simplify(['/', '1', '2'])
        assert eng.simplify(['0.2']) == eng.simplify(['/', '1', '5'])


class TestDecimalCodewordOracle:
    """Falsifier 2: the specimen table. Native mu' == the rescorer's adjusted price
    + the selector, EXACTLY, on specimens spanning every class the study rescored.

    (Integers here carry no trailing zeros: for those the two mu' readings coincide
    with the rescorer -- the 10-divisible integer class is pinned separately and
    prominently in TestIntegerScientificCodeword.)
    """

    SPECIMENS = [
        # integers (incl. the floor and the sign bit)
        (['0'], Fraction(0)),
        (['1'], Fraction(1)),
        (['2'], Fraction(2)),
        (['7'], Fraction(7)),
        (['123'], Fraction(123)),
        (['(-5)'], Fraction(-5)),
        (['2177697277405848'], Fraction(2177697277405848)),
        # short decimals
        (['0.5'], Fraction(1, 2)),
        (['0.2'], Fraction(1, 5)),
        (['1.2'], Fraction(6, 5)),
        (['2.5'], Fraction(5, 2)),
        (['0.25'], Fraction(1, 4)),
        (['0.125'], Fraction(1, 8)),
        (['(-0.2)'], Fraction(-1, 5)),
        (['3.14'], Fraction(314, 100)),
        # long decimals (the dominant SR population; the D38 headline specimen)
        (['2.177697277405848'], Fraction(2177697277405848, 10 ** 15)),
        (['1.4142135623730951'], Fraction(14142135623730951, 10 ** 16)),
        (['(-2.718281828459045)'], Fraction(-2718281828459045, 10 ** 15)),
        (['0.0293847502'], Fraction(293847502, 10 ** 10)),
        (['3.14159265358979'], Fraction(314159265358979, 10 ** 14)),
        # power-of-ten / 2^a5^b denominator fractions, both emission classes
        (['/', '7', '100'], Fraction(7, 100)),
        (['/', '355', '1000'], Fraction(355, 1000)),
        (['/', '1', '128'], Fraction(1, 128)),
        (['/', '3', '8'], Fraction(3, 8)),
        (['/', '1', '5'], Fraction(1, 5)),
        (['/', '(-3)', '20'], Fraction(-3, 20)),
        # non-decimal fractions (no decimal codeword)
        (['/', '1', '3'], Fraction(1, 3)),
        (['/', '22', '7'], Fraction(22, 7)),
        (['/', '(-1)', '3'], Fraction(-1, 3)),
        (['355/113'], Fraction(355, 113)),
    ]

    def test_specimens_price_exactly_oracle_plus_selector(self, eng):
        for tokens, value in self.SPECIMENS:
            expected = mu_prime_expected(value)
            got = c(eng, tokens)
            assert got == expected, (
                f'{tokens}: native {got} != oracle+selector {expected} '
                f'(old {old_rat_price(value)}, dec {oracle_dec_price(value)})')

    def test_the_d38_headline_specimen_moves(self, eng):
        # 2.177697277405848 priced 94,781 under mu (the rational codeword); under mu'
        # the decimal-scientific codeword (m = 2177697277405848, k = 15) takes over:
        # max(2, L(m)) + L(15) + selector = 50.952 + 4.000 + 1 bits.
        v = Fraction(2177697277405848, 10 ** 15)
        assert oracle_dec_price(v) < old_rat_price(v)
        assert c(eng, ['2.177697277405848']) == SEL + oracle_dec_price(v)
        assert c(eng, ['2.177697277405848']) < 94_781

    def test_h055_asymmetry_dissolves(self, eng):
        # H-055's named wart -- mu(1000) = 9.967 against mu(0.001) = 10.967 under the
        # fraction-only code (and 9.967 vs 3.000 under the pre-H-055 min) -- lands at
        # the symmetric point: both are (m=1, k=3) codewords, 2 + 2 + selector bits.
        assert c(eng, ['1000']) == c(eng, ['0.001']) == 5 * MILLI
        # and the reciprocal pair 10 / 0.1 likewise
        assert c(eng, ['10']) == c(eng, ['0.1']) == 4 * MILLI


class TestNonDecimalFractions:
    """Falsifier 3: a non-terminating value has ONLY the rational codeword -- its
    mu' is the old price + the selector, nothing else moves."""

    def test_rational_codeword_plus_selector_only(self, eng):
        for tokens, value in [
            (['/', '1', '3'], Fraction(1, 3)),
            (['/', '22', '7'], Fraction(22, 7)),
            (['/', '(-1)', '3'], Fraction(-1, 3)),
            (['355/113'], Fraction(355, 113)),
            (['/', '1', '7'], Fraction(1, 7)),
        ]:
            assert oracle_dec_price(value) is None
            assert c(eng, tokens) == SEL + old_rat_price(value), tokens

    def test_one_third_keeps_the_fraction_spelling(self, eng):
        # simplify is DIALECT-PRESERVING (owner-ruled): prefix in, prefix out. The mu'
        # property under test -- the fraction spelling survives -- holds in both dialects,
        # so both are pinned and `to_tagged` is the one explicit step between them.
        assert eng.simplify(['/', '1', '3']) == ['/', '1', '3']
        assert eng.to_tagged(eng.simplify(['/', '1', '3'])) \
            == ['<mul>', '1', '<div>', '3', '</mul>']


class TestIntegerScientificCodeword:
    """The spec-text corner, pinned PROMINENTLY: integers have denominator 1 = 2^0*5^0,
    so the spec ("shortest exact decimal spelling", L(k) with |k|) gives a 10-divisible
    integer the (m, k > 0) codeword -- 1000 is (1, 3) at 2 + 2 bits, not L(1000) = 9.967.

    DIVERGENCE FROM THE STUDY RESCORER, documented here because the report must carry
    it: dec_price() derives k from the DENOMINATOR only, so the study's rescored deltas
    left 10-divisible integer literals at the rational price. Native mu' discounts
    them. This is required by the ratified spec text AND forced by the c_free'
    derivation: without the positive-k codeword, integer-valued f64 spellings like
    1.7976931348623157e308 price ~1025 bits and the sup could not land at 67. Rescore
    comparisons against the study tables differ on this class beyond the selector.

    The SURFACE does not change: an integer token's digit string IS the rendering of
    both its codewords, so `1000` still prints as `1000` (never `1e3`).
    """

    def test_ten_divisible_integers_take_the_scientific_codeword(self, eng):
        # 1000 = (m=1, k=3): selector + max(2, L(1)) + L(3) = 1 + 2 + 2 bits.
        assert c(eng, ['1000']) == 5 * MILLI
        assert c(eng, ['1e3']) == 5 * MILLI          # same value, same price
        assert c(eng, ['100']) == SEL + 2 * MILLI + L(2)      # (1, 2)
        assert c(eng, ['20']) == SEL + 2 * MILLI + L(1)       # (2, 1)
        assert c(eng, ['(-30)']) == SEL + 2 * MILLI + L(1) + MILLI  # sign bit rides

    def test_non_ten_divisible_integers_are_unmoved_beyond_the_selector(self, eng):
        for tok, v in [('123', 123), ('7', 7), ('999', 999), ('2177697277405848', 2177697277405848)]:
            assert c(eng, [tok]) == SEL + old_int_price(v), tok

    def test_integer_surface_never_goes_scientific(self, eng):
        assert eng.simplify(['1000']) == ['1000']
        # Dialect-preserving, as above: the integer surface stays non-scientific in
        # either spelling.
        assert eng.simplify(['*', '100', 'x0']) == ['*', '100', 'x0']
        assert eng.to_tagged(eng.simplify(['*', '100', 'x0'])) \
            == ['<mul>', '100', 'x0', '</mul>']


class TestCfreePrimeDerivation:
    """Falsifier 4: c_free' is DERIVED under the new codebook, not pinned -- the sup
    of mu' over f64 round-trip spellings, + the sign bit, + the selector, ceiled --
    and it strictly dominates every priced f64 literal (the sup-construction's
    defining property).

    The model: an f64's round-trip spelling is its shortest repr, denoting the exact
    rational m0 * 10^s. Its mu' is selector + min(rational, scientific) + sign. The
    scientific code caps at L(m0 <= 17 digits) + L(|s| <= ~324) < 64.83 bits, so
    c_free' = 67 -- against mu's 1133, which was driven by the 10^324 DENOMINATOR of
    the same extreme spellings under the fraction-only code (16.9x).
    """

    @staticmethod
    def _token_parts(x: float):
        s = repr(abs(x))
        if 'e' in s or 'E' in s:
            mant, e = s.lower().split('e')
            e = int(e)
        else:
            mant, e = s, 0
        ip, _, fp = mant.partition('.')
        digits = (ip + fp).lstrip('0')
        scale = e - len(fp)
        stripped = digits.rstrip('0') or '0'
        scale += len(digits) - len(stripped)
        return int(stripped), scale

    @classmethod
    def _model_bits(cls, x: float) -> float:
        """min-codeword bits of |x|'s round-trip spelling (no sign, no selector)."""
        m0, s = cls._token_parts(x)
        if m0 == 0:
            return 2.0
        if s >= 0:
            frac = math.log2(1 + m0 * 10 ** s)
        else:
            f = Fraction(m0, 10 ** -s)
            frac = math.log2(1 + f.numerator) + math.log2(1 + f.denominator)
        frac = max(2.0, frac)
        if s == 0:
            return frac
        sci = max(2.0, math.log2(1 + m0)) + math.log2(1 + abs(s))
        return min(frac, sci)

    def test_c_free_prime_value(self, eng):
        assert c(eng, ['<constant>']) == MU_FREE_PRIME

    def test_c_free_prime_recomputes_from_the_f64_supremum(self, eng):
        sup, argmax = 0.0, None
        mantissas = (1.0, 1.5, 1.0 + 2 ** -52, 2.0 - 2 ** -52,
                     1.2345678901234567, 1.9999999999999998)
        for be in range(-1074, 1024):
            for m in mantissas:
                x = math.ldexp(m, be)
                if x == 0.0 or math.isinf(x):
                    continue
                b = self._model_bits(x)
                if b > sup:
                    sup, argmax = b, x
        # The sweep must reach the extreme region (anti-vacuity)...
        assert sup > 63.0, (sup, argmax)
        # ...and nothing may price above the analytic cap L(10^17) + L(343):
        cap = math.log2(1 + 10 ** 17) + math.log2(1 + 343)
        assert sup < cap < 64.92, (sup, argmax)
        # The derived constant: sup + 1 sign bit + 1 selector bit, ceiled. The band
        # that lands on 67 is sup in (64, 65] -- an earlier comment said (63, 64.92),
        # which is arithmetically wrong at its own lower end: ceil(63 + 2) is 65.
        assert math.ceil(sup + 1.0 + 1.0) == 67
        assert c(eng, ['<constant>']) == 67 * MILLI

    def test_c_free_prime_dominates_every_priced_f64_literal(self, eng):
        cf = c(eng, ['<constant>'])
        mantissas = (1.0, 1.5, 1.0 + 2 ** -52, 2.0 - 2 ** -52,
                     1.2345678901234567, 1.9999999999999998)
        worst = (0, None)
        for be in range(-1074, 1024, 7):
            for m in mantissas:
                x = math.ldexp(m, be)
                if x == 0.0 or math.isinf(x):
                    continue
                for tok in (repr(x), '-' + repr(x)):
                    got = c(eng, [tok] if not tok.startswith('-') else [f'({tok})'])
                    if got >= cf:
                        assert False, f'{tok} prices {got} >= c_free\' {cf}'
                    if got > worst[0]:
                        worst = (got, tok)
        # the extremes themselves, exactly (both boundary spellings):
        for tok in ('5e-324', '1.7976931348623157e308', '2.2250738585072014e-308',
                    '5.5605781537525765e-308'):
            assert c(eng, [tok]) < cf, tok


class TestSerializerArgmin:
    """Falsifier 5: states carry one exact rational per value; the serializer emits
    the argmin codeword's spelling; mu'(state) equals the cost of the representation
    actually emitted; the serializer never changes values.

    Dialect scope, stated: the argmin lives in the EXPLICIT and infix emitters
    (`emit_num` / `num_token`). The TAGGED dialect keeps its ratified model-facing
    exception -- a fraction whose components fit the bounded integer vocabulary
    (SIMPLIPY_TAGGED_FRACTION_MAX) spells structurally regardless of the argmin,
    because the consumer's vocabulary carries no decimal tokens. That exception
    predates mu' and is not B2's to move."""

    def test_argmin_spelling_is_emitted(self, eng):
        # decimal codeword wins -> decimal token; rational wins -> structural fraction
        assert eng.simplify(eng.to_prefix(['/', '1', '5'])) == ['0.2']
        assert eng.simplify(eng.to_prefix(['/', '6', '5'])) == ['1.2']
        assert eng.simplify(eng.to_prefix(['0.5'])) == ['/', '1', '2']
        assert eng.simplify(eng.to_prefix(['2.177697277405848'])) == ['2.177697277405848']

    def test_near_tie_boundary_both_sides(self, eng):
        # The closest calls the lattice scan found (every (a, b) with 2^a*5^b < 2^127,
        # exhaustive p <= 2000 plus spread samples to ~3.5e9: NO exact tie found, so
        # the tie->fraction clause is defensive): the argmin must follow the STRICT
        # side of each.
        q_dec = 2 ** 45 * 5 ** 14           # dec beats frac by 3 milli-bits at p=54321
        v = Fraction(54321, q_dec)
        assert oracle_dec_price(v) == old_rat_price(v) - 3
        out = eng.simplify(eng.to_prefix(['/', '54321', str(q_dec)]))
        assert len(out) == 1 and '.' in out[0], out
        assert Fraction(out[0]) == v            # never value-changing
        q_frac = 2 ** 28 * 5 ** 9           # frac beats dec by 4 milli-bits at p=19
        v2 = Fraction(19, q_frac)
        assert oracle_dec_price(v2) == old_rat_price(v2) + 4
        out2 = eng.simplify(eng.to_prefix(['/', '19', str(q_frac)]))
        assert out2 == ['/', '19', str(q_frac)], out2

    def test_mu_state_equals_emitted_cost(self, eng):
        # The invariant itself, on literal-bearing states: the emitted spelling
        # re-prices to EXACTLY the state's mu' -- the emission realizes the min the
        # measure charged (spelling-invariance makes the reparse price the state; the
        # argmin choice makes the written form the codeword that price came from).
        for tokens in (['/', '1', '5'], ['/', '3', '8'], ['2.177697277405848'],
                       ['/', '54321', str(2 ** 45 * 5 ** 14)], ['0.001'], ['1000']):
            # `form=` is gone (0.14.0): CONVERT first, which is the byte-identical
            # recipe the removed parameter documented.
            for pre in (eng.to_prefix, None):
                src = tokens if pre is None else pre(tokens)
                out = list(eng.simplify(src))
                assert c(eng, out) == c(eng, tokens), (tokens, out)
                # and the emission is a fixpoint: re-simplifying does not respell
                assert list(eng.simplify(out)) == out, (tokens, out)

    def test_round_trip_exactness_on_corpus_literals(self, eng):
        # the serializer never changes values: subtracting the emission from the
        # input must fold to EXACTLY zero in the engine's own exact arithmetic
        for num, den in [(1, 5), (6, 5), (1, 2), (3, 8), (7, 100), (355, 113),
                         (54321, 2 ** 45 * 5 ** 14), (19, 2 ** 28 * 5 ** 9),
                         (2177697277405848, 10 ** 15), (-3, 20), (1, 3)]:
            src = ['+', 'x0', '/', str(num) if num > 0 else f'({num})', str(den)]
            out = list(eng.simplify(eng.to_prefix(src)))
            assert list(eng.simplify(eng.to_prefix(['-'] + out + src))) == ['0'], (src, out)


class TestSooseStructuralSpecimens:
    """Falsifier 8: the SOOSE row-235/row-59 shapes (cos*tan). No behavior assertion
    beyond: the engine loads acj-4-3, simplifies, and mu'-never-worse holds -- the
    cos*tan -> sin rule itself is the acj-5 mine's (B3), not this artifact's."""

    SPECIMENS = [
        ['*', 'x1', '*', 'cos', 'x2', 'tan', 'x2'],           # row 235 shape
        ['*', 'cos', 'x0', 'tan', 'x0'],                      # row 59 shape (bare)
        ['*', 'x2', 'sin', '+', 'pow', 'x1', '2', '*', 'x1', 'x2'],  # poly-factor class
    ]

    def test_engine_simplifies_and_mu_prime_never_worse(self, eng):
        for expr in self.SPECIMENS:
            out = eng.simplify(eng.to_prefix(expr))
            assert eng.complexity(out) <= eng.complexity(list(expr)), (expr, out)
            assert list(eng.simplify(eng.to_prefix(out))) == list(out), expr


class TestNativeDescentNeverWorse:
    """Falsifier 6: under NATIVE mu' descent, never-worse is restored by construction
    -- the study's 3.79% made-bigger class (mu-descending outputs rescored under mu')
    must be EMPTY when the engine descends mu' itself. 400/400 on the reference
    corpus, plus idempotence."""

    def test_400_skeleton_corpus(self, eng):
        corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'benchmarks', 'corpus', 'raw_skeletons_nv.json')
        if not os.path.exists(corpus_path):
            pytest.skip('reference corpus not present')
        corpus = json.load(open(corpus_path))
        assert len(corpus) == 400
        made_bigger = []
        for expr in corpus:
            out = list(eng.simplify(eng.to_prefix(expr)))
            if eng.complexity(out) > eng.complexity(list(expr)):
                made_bigger.append(expr)
            assert list(eng.simplify(eng.to_prefix(out))) == out, expr
        assert made_bigger == [], f'{len(made_bigger)} rows ASCENDED under native mu\''


class TestFingerprintAndArtifactLoad:
    """The load contract: acj-4-3 (mined under mu) still loads and serves; the
    measure fingerprint MOVES (that is what it is for) and is write-only at load --
    D25 warns on the mismatch, never refuses."""

    def test_fingerprint_probes_pin_mu_prime(self, eng):
        fp = eng._measure_fingerprint()
        assert fp['probes'] == {
            # the literal codebook
            '1000': 5 * MILLI,        # (1, 3): selector + 2 + L(3)
            '1/2': 3585,              # rational wins: selector + L(1) + L(2)
            '355/113': 16309,         # no decimal codeword: selector + old price
            '0.2': 4 * MILLI,         # (2, 1): selector + 2 + L(1)
            '<constant>': MU_FREE_PRIME,
            # the symbol table, one probe per entry (2026-08-21). Add and Mul price the
            # same here and still need separate probes: a change to ONE of them has to
            # move the digest.
            'x0': MU_LEAF,                    # 6
            '+': 3 * MILLI + 2 * MU_LEAF,     # Add 3
            '*': 3 * MILLI + 2 * MU_LEAF,     # Mul 3
            'pow': 3 * MILLI + MU_LEAF + 3 * MILLI,   # Pow 3 + leaf + cost(2)
            'np.pi': 4 * MILLI, 'np.e': 4 * MILLI,
            'sin': 6 * MILLI + MU_LEAF,       # elementary head
            'asin': 8 * MILLI + MU_LEAF,      # transcendental head
            'float("inf")': 8 * MILLI,
        }
        assert fp['digest'] != ACJ_MINED_DIGEST

    def test_acj_load_warns_until_the_remine(self):
        """D25 warn-not-refuse, on the real asset, for exactly as long as the asset
        predates the measure.

        The shipped triple was mined under `ACJ_SHIPPED_DIGEST`; the symbol table
        (2026-08-21) changed the measure, so the engine computes a different one and
        says so. The rules stay SOUND -- the warning is about their minimality and
        ordering claims, which were certified against the old prices -- and the artifact
        still loads and serves, which is the whole point of warn-not-refuse.

        WHEN THE RE-MINE UNDER THE NEW MEASURE LANDS, this test must go back to
        asserting NO fingerprint warning. It is written to force that: it pins the
        mined digest verbatim, so a re-mined artifact fails it rather than sliding
        through. A blanket "ignore fingerprint warnings" allowance is what this shape
        deliberately avoids -- one existed before, and its own deletion note recorded
        that while it stayed it would have silently swallowed a genuine mismatch."""
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            engine = SimpliPyEngine.from_config(acj_config_path())
        mismatch = [str(x.message) for x in caught
                    if 'measure fingerprint mismatch' in str(x.message)]
        assert len(mismatch) == 1, caught
        assert ACJ_SHIPPED_DIGEST in mismatch[0], mismatch[0]
        assert engine._measure_fingerprint()['digest'] in mismatch[0], mismatch[0]
        out = engine.simplify(engine.to_prefix(['+', 'x0', 'x0']))
        assert list(out) == ['*', '2', 'x0']

    def test_load_is_not_refused_and_rules_serve(self, eng):
        # the artifact's rules still fire under the mu' ordering
        assert eng.simplify(eng.to_prefix(['abs', 'neg', 'x0'])) == ['abs', 'x0']
