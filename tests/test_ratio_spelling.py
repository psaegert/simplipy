"""Integer-over-decimal spelling of a fraction (owner 2026-10-02) -- EMISSION ONLY, like the
divisor-side rule (test_divisor_side_spelling.py).

Every literal is its exact rational value, so ``3.141592653589793`` is 3141592653589793/10^15
and ``1/(2*3.141592653589793)`` folds to 500000000000000/3141592653589793. That value has no
finite decimal, so the emitters spelled it as the division of those two integers -- exact,
but nothing a reader recognizes. The same value is also exactly ``1 / 6.283185307179586``:
a decimal is an exact rational, so a division by one is as exact as a division by an
integer. Taking the factors 2 and 5 out of the numerator always leaves a terminating
decimal in the denominator; that spelling is used wherever it is strictly shorter.

Guardrails pinned here:
* spelling never enters the measure: the state is the same either way, so idempotence and
  round-trip identity hold, and ``complexity`` does not change;
* only a strictly shorter spelling fires: ``1/2``, ``2/3``, ``5/8`` and ``1/3`` keep their
  fraction;
* the denominator is always ONE decimal token, so no reader can re-associate it.
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


def simplify(eng, tokens):
    out = list(eng.simplify(eng.to_prefix(list(tokens))))
    assert list(eng.simplify(list(out))) == out, f'not idempotent: {out}'
    return out


def literal_value(tokens, x=Fraction(1)):
    """The exact value of a prefix expression of `/`, `*` and `neg` over literals and x1 (= x)."""
    def walk(i):
        t = tokens[i]
        if t in ('/', '*'):
            a, i = walk(i + 1)
            b, i = walk(i)
            return (a / b if t == '/' else a * b), i
        if t == 'neg':
            a, i = walk(i + 1)
            return -a, i
        return (x if t == 'x1' else Fraction(t)), i + 1
    value, end = walk(0)
    assert end == len(tokens)
    return value


class TestStandaloneValue:
    def test_one_over_two_pi(self, eng):
        out = simplify(eng, ['/', '1', '*', '2', PI])
        assert out == ['/', '1', '6.283185307179586']
        assert literal_value(out) == 1 / (2 * Fraction(PI))

    def test_a_decimal_divisor_keeps_its_spelling(self, eng):
        # 1/3.142 is the fraction 500/1571; `1 / 3.142` is the same value, one character shorter
        assert simplify(eng, ['/', '1', '3.142']) == ['/', '1', '3.142']

    def test_a_negative_value_keeps_its_sign_on_the_integer(self, eng):
        out = simplify(eng, ['neg', '/', '1', '*', '2', PI])
        assert '6.283185307179586' in out and '500000000000000' not in out
        assert literal_value(out) == -1 / (2 * Fraction(PI))

    @pytest.mark.parametrize('tokens, expected', [
        (['/', '1', '2'], ['/', '1', '2']),
        (['/', '2', '3'], ['/', '2', '3']),
        (['/', '5', '8'], ['/', '5', '8']),
        (['/', '1', '3'], ['/', '1', '3']),
    ])
    def test_a_short_fraction_stays_a_fraction(self, eng, tokens, expected):
        assert simplify(eng, tokens) == expected


class TestInsideExpressions:
    def test_under_a_root(self, eng):
        # Feynman I.6.2b: exp(-((theta - theta1)/sigma)^2 / 2) / (sigma * sqrt(2 pi))
        out = simplify(eng, ['/', 'exp', '/', 'neg', 'pow', '/', '-', 'x2', 'x3', 'x1', '2', '2',
                             '*', 'rootn', '*', '2', PI, '2', 'x1'])
        assert '6.283185307179586' in out
        assert not any(t.lstrip('-').isdigit() and len(t.lstrip('-')) > 6 for t in out), out

    def test_a_coefficient_with_a_numerator(self, eng):
        # Feynman 58: (3/5) * q^2 / (4 pi eps r) -- the coefficient 3/(20 pi) has an odd numerator,
        # so no reciprocal decimal exists; integer over decimal: 3 / 62.83185307179586
        out = simplify(eng, ['/', '*', '/', '3', '5', 'pow', 'x1', '2', '*', '*', '*', '4', PI, 'x2', 'x3'])
        assert '62.83185307179586' in out and '3' in out
        assert not any(t.lstrip('-').isdigit() and len(t.lstrip('-')) > 6 for t in out), out

    def test_the_measure_does_not_see_the_spelling(self, eng):
        a = eng.complexity(['/', '1', '*', '2', PI])
        b = eng.complexity(['/', '500000000000000', '3141592653589793'])
        c = eng.complexity(['/', '1', '6.283185307179586'])
        assert a == b == c


class TestInfix:
    def test_the_readable_form_reads_back(self, eng):
        out = simplify(eng, ['*', 'x1', 'rootn', '/', '1', '*', '2', PI, '2'])
        text = eng.prefix_to_infix(out)
        assert '6.283185307179586' in text and '500000000000000' not in text, text
        assert list(eng.simplify(eng.to_prefix(eng.infix_to_prefix(text)))) == out


class TestRandomValues:
    @pytest.mark.parametrize('seed', range(4))
    def test_every_spelling_round_trips_exactly(self, eng, seed):
        import random
        rng = random.Random(seed)
        for _ in range(60):
            num = rng.choice([1, 3, 7, 11, 13]) * 2 ** rng.randrange(0, 20) * 5 ** rng.randrange(0, 20)
            den = rng.choice([3, 7, 9, 11, 13, 3141592653589793, 271828182845905])
            value = Fraction(num, den)
            alone = simplify(eng, ['/', str(value.numerator), str(value.denominator)])
            assert literal_value(alone) == value, (value, alone)
            coefficient = simplify(eng, ['*', 'x1', '/', str(value.numerator), str(value.denominator)])
            assert literal_value(coefficient, Fraction(7)) == 7 * value, (value, coefficient)
            # the integer-over-decimal spelling fires only when strictly shorter than p/q
            if len(alone) == 3 and alone[0] == '/' and '.' in alone[2]:
                assert len(alone[1]) + len(alone[2]) < len(str(value.numerator)) + len(str(value.denominator))
