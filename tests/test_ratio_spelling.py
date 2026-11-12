"""Integer-over-decimal spelling of a fraction -- in the INFIX text only.

Every literal is its exact rational value, so ``3141.592653`` is 3141592653/10^6 and
``1/(2*3141.592653)`` folds to 500000/3141592653, cheaper than its double. That value has no
finite decimal, so the printers spelled it as the division of those two integers -- exact,
but nothing a reader recognizes. The same value is also exactly ``1 / 6283.185306``:
a decimal is an exact rational, so a division by one is as exact as a division by an
integer. Taking the factors 2 and 5 out of the numerator always leaves a terminating
decimal in the denominator; the infix text uses that spelling wherever it is strictly
shorter. (``3.141592653589793``, pi's double, is read as pi, and a value whose double is
cheaper than its exact spellings is written as that double, so neither shows the spelling. A
value whose reciprocal is a terminating decimal, ``1/(2*3141.592653)`` among them, is written
``1/d`` in every dialect: that divisor is the measure's codeword for it.)

The token answers keep ``p/q`` (test_spelling_invariance.py): tokens are
what the engine and its callers read back, and a spelling chosen for a reader must not move
what they decide.

Guardrails pinned here:
* the infix text re-reads to the state it was printed from, and ``complexity`` does not
  change;
* only a strictly shorter spelling fires: ``1/2``, ``2/3``, ``5/8`` and ``1/3`` keep their
  fraction;
* the denominator is always ONE decimal, so no reader can re-associate it.
"""
from fractions import Fraction

import pytest

from conftest import acj_config_path
from conftest import legacy_from_config  # noqa: E402

CONFIG = acj_config_path()
PI = '3141.592653'


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return legacy_from_config(CONFIG)


def tokens(eng, text):
    """The token answer for an infix text: the state, in the engine's own spelling."""
    return list(eng.simplify(list(eng._core.parse(text, True, False))))


def infix(eng, text):
    out = eng.simplify(text)
    assert eng.simplify(out) == out, f'not idempotent: {text} -> {out}'
    # the reader's spelling re-reads to the state it was printed from, at the same price
    assert tokens(eng, out) == tokens(eng, text), (text, out)
    assert eng.complexity(out) == eng.complexity(tokens(eng, text)), (text, out)
    return out


def quotient(text):
    """The exact value of an `n/d` literal text."""
    n, d = text.split('/')
    return Fraction(n) / Fraction(d)


class TestStandaloneValue:
    def test_one_over_two_p(self, eng):
        out = infix(eng, f'1/(2*{PI})')
        assert out == '1/6283.185306'
        assert quotient(out) == 1 / (2 * Fraction(PI))

    def test_the_token_answer_keeps_the_exact_fraction(self, eng):
        assert tokens(eng, f'7/(2*{PI})') == ['/', '3500000', '3141592653']
        assert infix(eng, f'7/(2*{PI})') == '7/6283.185306'
        # a unit numerator: the reciprocal codeword, the divisor, in the tokens too
        assert tokens(eng, f'1/(2*{PI})') == ['/', '1', '6283.185306']

    def test_pis_double_is_pi(self, eng):
        # the recognition list reads 3.141592653589793, pi's double, as pi
        assert tokens(eng, '1/(2*3.141592653589793)') == ['inv', '*', '2', 'np.pi']
        assert infix(eng, '1/(2*3.141592653589793)') == '1/2/pi'

    def test_a_decimal_divisor_keeps_its_spelling(self, eng):
        # 1/3.142 is the fraction 500/1571; `1/3.142` is the same value, priced as its divisor
        assert infix(eng, '1/3.142') == '1/3.142'
        assert tokens(eng, '1/3.142') == ['/', '1', '3.142']

    def test_a_negative_value_keeps_its_sign_on_the_integer(self, eng):
        out = infix(eng, f'-1/(2*{PI})')
        assert out == '-1/6283.185306'
        assert quotient(out) == -1 / (2 * Fraction(PI))

    @pytest.mark.parametrize('text', ['1/2', '2/3', '5/8', '1/3'])
    def test_a_short_fraction_stays_a_fraction(self, eng, text):
        assert infix(eng, text) == text


class TestInsideExpressions:
    def test_under_a_root(self, eng):
        # (a root of a constant alone folds to its float, `1/sqrt(2 pi)` to 0.3989422804014327)
        out = infix(eng, f'rootn(x1/(2*{PI}), 2)')
        assert out == 'rootn(x1/6283.185306, 2)'

    def test_a_coefficient_with_a_numerator(self, eng):
        # Feynman 58's shape: (7/5) * q^2 / (4 p eps r) -- the coefficient 7/(20 p) has an odd
        # numerator, so no reciprocal decimal exists; integer over decimal: 7 / 62831.85306
        out = infix(eng, f'(7/5)*x1^2/(4*{PI}*x2*x3)')
        assert out == '7*x1^2/62831.85306/x2/x3'

    def test_a_token_answer_converts_to_the_readers_spelling(self, eng):
        out = tokens(eng, f'x1 + 7/(2*{PI})')
        assert out == ['+', 'x1', '/', '3500000', '3141592653']
        assert eng.simplify(eng.to_infix(out)) == 'x1 + 7/6283.185306'

    def test_the_measure_does_not_see_the_spelling(self, eng):
        a = eng.complexity(['/', '1', '*', '2', PI])
        b = eng.complexity(['/', '500000', '3141592653'])
        c = eng.complexity(['/', '1', '6283.185306'])
        assert a == b == c


class TestRandomValues:
    @pytest.mark.parametrize('seed', range(4))
    def test_every_spelling_reads_back_exactly(self, eng, seed):
        import random
        rng = random.Random(seed)
        for _ in range(60):
            num = rng.choice([1, 3, 7, 11, 13]) * 2 ** rng.randrange(0, 20) * 5 ** rng.randrange(0, 20)
            den = rng.choice([3, 7, 9, 11, 13, 3141592653589793, 271828182845905])
            value = Fraction(num, den)
            alone = infix(eng, f'{value.numerator}/{value.denominator}')
            # the exact value, or (strict) float64's double of it where that is cheaper: its
            # shortest decimal, or the double's exact value as a fraction over a power of two
            written = quotient(alone) if '/' in alone else Fraction(alone)
            assert written == value or written == Fraction(float(value)) \
                or ('/' not in alone and float(alone) == float(value)), (value, alone)
            # the integer-over-decimal spelling fires only when strictly shorter than p/q; `1/d`
            # is the reciprocal codeword, written where it is the price
            if '/' in alone and '.' in alone:
                assert written == value, (value, alone)
                if not alone.startswith('1/'):
                    assert len(alone) < len(f'{value.numerator}/{value.denominator}'), (value, alone)
            infix(eng, f'x1*{value.numerator}/{value.denominator}')
