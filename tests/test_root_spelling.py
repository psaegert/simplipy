"""An even root is written as `rootn` wherever it stands (owner 2026-10-02) -- EMISSION ONLY, like the
divisor-side rule and the integer-over-decimal spelling.

The core stores x^(1/n) for even n as `rootn(x, n)` (owner 2026-08-06: it prices below `pow(x, 1/n)`),
but only for a unit fraction. Its inverse x^(-1/n) stays a power, so the printers, which move
negative powers below the fraction bar, wrote `1/x^(1/2)` beside `rootn(x, 2)`: one root, two
spellings. And a literal base absorbs the exponent's sign (`c^(-t) -> (1/c)^t`), so the inverse
root of a number came out as `rootn(1/c, 2)` -- the reciprocal inside the root.

The printers now follow one rule, the way odd roots already print (`1/rootn(x0, 3)`):
* R1: a power with exponent 1/n (n even) below the fraction bar prints as `rootn(b, n)`;
* R2: the even root of a literal whose reciprocal is one shorter number (the divisor-side test)
  prints as `1/rootn(reciprocal, n)`.
Same state either way: the output re-parses to it, `complexity()` is unchanged, and simplify is
idempotent on it. The tagged form is untouched.
"""
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


def infix(eng, text):
    out = eng.simplify(text)
    assert eng.simplify(out) == out, f'not idempotent: {text} -> {out}'
    return out


def prefix(eng, text):
    out = list(eng.simplify(eng.to_prefix(eng.infix_to_prefix(text))))
    assert list(eng.simplify(list(out))) == out, f'not idempotent: {out}'
    return out


R1 = [  # a symbolic base below the fraction bar
    ('1/rootn(x0, 2)', '1/rootn(x0, 2)'),
    ('rootn(1/x0, 2)', '1/rootn(x0, 2)'),
    ('x1/rootn(x0, 2)', 'x1/rootn(x0, 2)'),
    ('1/rootn(x0 + 1, 2)', '1/rootn(x0 + 1, 2)'),
    ('1/rootn(x0, 4)', '1/rootn(x0, 4)'),
    ('1/rootn(2*pi, 2)', '1/rootn(2*pi, 2)'),
]
R2 = [  # a literal whose reciprocal is one shorter number
    (f'1/rootn(2*{PI}, 2)', '1/rootn(6.283185307179586, 2)'),
    (f'x0/rootn(2*{PI}, 2)', 'x0/rootn(6.283185307179586, 2)'),
    ('1/rootn(3, 2)', '1/rootn(3, 2)'),
    ('rootn(1/2, 2)', '1/rootn(2, 2)'),
]
UNCHANGED = [
    ('rootn(x0, 2)', 'rootn(x0, 2)'),
    ('x0^(3/2)', 'x0^(3/2)'),
    ('x0^(-3/2)', '1/x0^(3/2)'),
    ('1/rootn(x0, 3)', '1/rootn(x0, 3)'),
    ('rootn(0.2, 2)', 'rootn(0.2, 2)'),      # an exact decimal never moves
    ('rootn(2/3, 2)', 'rootn(2/3, 2)'),      # its reciprocal 3/2 is a fraction, not one shorter number
]


@pytest.mark.parametrize('text, expected', R1 + R2 + UNCHANGED)
def test_infix_spelling(eng, text, expected):
    assert infix(eng, text) == expected


def test_feynman_i_6_2a_reads_the_same_with_a_decimal_or_a_symbolic_pi(eng):
    assert infix(eng, f'exp(-x0^2/2)/rootn(2*{PI}, 2)') == 'exp(-x0^2/2)/rootn(6.283185307179586, 2)'
    assert infix(eng, 'exp(-x0^2/2)/rootn(2*pi, 2)') == 'exp(-x0^2/2)/rootn(2*pi, 2)'


@pytest.mark.parametrize('text, expected', [
    ('1/rootn(x0, 2)', ['inv', 'rootn', 'x0', '2']),
    ('x1/rootn(x0, 2)', ['/', 'x1', 'rootn', 'x0', '2']),
    (f'1/rootn(2*{PI}, 2)', ['inv', 'rootn', '6.283185307179586', '2']),
    (f'x0/rootn(2*{PI}, 2)', ['/', 'x0', 'rootn', '6.283185307179586', '2']),
])
def test_prefix_spelling(eng, text, expected):
    assert prefix(eng, text) == expected


@pytest.mark.parametrize('text, _', R1 + R2 + UNCHANGED)
def test_the_state_and_its_price_do_not_move(eng, text, _):
    out = eng.simplify(text)
    assert eng.complexity(out) == eng.complexity(text)
    # the printed form re-reads to the state it was printed from
    assert eng.simplify(out) == out


class TestRandomShapes:
    @pytest.mark.parametrize('seed', range(4))
    def test_every_output_reads_back_to_itself(self, eng, seed):
        import random
        rng = random.Random(seed)
        bases = ['x0', 'x1', 'x0 + 1', '2*x1', 'pi', '3', '2', '1/3', f'2*{PI}', '0.5', '7.25', 'x0*x1']
        for _ in range(80):
            b = rng.choice(bases)
            n = rng.choice([2, 2, 4, 3])
            k = rng.choice([1, -1, 3, -3])
            root = f'rootn({b}, {n})'
            text = rng.choice([f'1/{root}', f'x0/{root}', f'{root}^{k}', f'x1*{root}', f'{root}/(x0 + 2)',
                               f'exp(1/{root})', f'x0 + 1/{root}'])
            out = eng.simplify(text)
            assert eng.simplify(out) == out, (text, out)
            assert eng.complexity(out) == eng.complexity(text), (text, out)
