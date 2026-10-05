"""Integer-over-decimal spelling of a fraction (owner 2026-10-02) -- in the INFIX text only.

Every literal is its exact rational value, so ``3.141592653589793`` is 3141592653589793/10^15
and ``1/(2*3.141592653589793)`` folds to 500000000000000/3141592653589793. That value has no
finite decimal, so the printers spelled it as the division of those two integers -- exact,
but nothing a reader recognizes. The same value is also exactly ``1 / 6.283185307179586``:
a decimal is an exact rational, so a division by one is as exact as a division by an
integer. Taking the factors 2 and 5 out of the numerator always leaves a terminating
decimal in the denominator; the infix text uses that spelling wherever it is strictly
shorter.

The token answers keep ``p/q`` (test_spelling_invariance.py; plan v2 phase 1): tokens are
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

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
PI = '3.141592653589793'


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


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
    def test_one_over_two_pi(self, eng):
        out = infix(eng, f'1/(2*{PI})')
        assert out == '1/6.283185307179586'
        assert quotient(out) == 1 / (2 * Fraction(PI))

    def test_the_token_answer_keeps_the_exact_fraction(self, eng):
        assert tokens(eng, f'1/(2*{PI})') == ['/', '500000000000000', '3141592653589793']

    def test_a_decimal_divisor_keeps_its_spelling(self, eng):
        # 1/3.142 is the fraction 500/1571; `1/3.142` is the same value, one character shorter
        assert infix(eng, '1/3.142') == '1/3.142'
        assert tokens(eng, '1/3.142') == ['/', '500', '1571']

    def test_a_negative_value_keeps_its_sign_on_the_integer(self, eng):
        out = infix(eng, f'-1/(2*{PI})')
        assert out == '-1/6.283185307179586'
        assert quotient(out) == -1 / (2 * Fraction(PI))

    @pytest.mark.parametrize('text', ['1/2', '2/3', '5/8', '1/3'])
    def test_a_short_fraction_stays_a_fraction(self, eng, text):
        assert infix(eng, text) == text


class TestInsideExpressions:
    def test_under_a_root(self, eng):
        # Feynman I.6.2b: exp(-((theta - theta1)/sigma)^2 / 2) / (sigma * sqrt(2 pi))
        out = infix(eng, f'exp(-((x2 - x3)/x1)^2/2)/(x1*rootn(2*{PI}, 2))')
        assert out == 'exp(-(x2 - x3)^2/2/x1^2)*rootn(1/6.283185307179586, 2)/x1'

    def test_a_coefficient_with_a_numerator(self, eng):
        # Feynman 58: (3/5) * q^2 / (4 pi eps r) -- the coefficient 3/(20 pi) has an odd numerator,
        # so no reciprocal decimal exists; integer over decimal: 3 / 62.83185307179586
        out = infix(eng, f'(3/5)*x1^2/(4*{PI}*x2*x3)')
        assert out == '3*x1^2/62.83185307179586/x2/x3'

    def test_a_token_answer_converts_to_the_readers_spelling(self, eng):
        out = tokens(eng, f'x1*rootn(1/(2*{PI}), 2)')
        assert out == ['*', 'x1', 'rootn', '/', '500000000000000', '3141592653589793', '2']
        assert eng.simplify(eng.to_infix(out)) == 'x1*rootn(1/6.283185307179586, 2)'

    def test_the_measure_does_not_see_the_spelling(self, eng):
        a = eng.complexity(['/', '1', '*', '2', PI])
        b = eng.complexity(['/', '500000000000000', '3141592653589793'])
        c = eng.complexity(['/', '1', '6.283185307179586'])
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
            if '/' in alone:
                assert quotient(alone) == value, (value, alone)
            # the integer-over-decimal spelling fires only when strictly shorter than p/q
            if '.' in alone:
                assert len(alone) < len(f'{value.numerator}/{value.denominator}'), (value, alone)
            infix(eng, f'x1*{value.numerator}/{value.denominator}')
