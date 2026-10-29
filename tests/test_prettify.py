"""`prettify`: the display of a canonical state.

Pinned here:
* the expected-output table at D = 3, row by row, plus the token forms;
* display only: prettify changes how a number is written, never the structure -- a number
  whose short spelling would change it is written as `s + r`; a function of numbers that are
  all shown as floats is one float, computed as the evaluator computes it, rounded to the
  precision, never 0, inf or a subnormal; any other function stays;
* idempotence: `prettify(prettify(c)) == prettify(c)` in all three forms on the release
  corpus and a fold fuzz, at full precision and at a precision, and the display does not depend
  on how the state holds its reciprocals;
* the structure: the display reads back to the shape of the state after the display's folds,
  in all three forms on the corpus and the fold fuzz, at full precision and at a precision;
* the read-back: every number of the display reads back as the float64 of the value it
  stands for -- the display evaluates as the state does on the corpus, and a one-ulp change to
  a displayed float is caught;
* the grouping: with `keep_grouping=True` every sum and product is written as SimpliPy's
  explicit tokens group it, so the display, infix or explicit, evaluates in the state's order;
  idempotence and the structure hold with it, and by default sums are flat;
* the source guard: nothing inside SimpliPy calls prettify -- and the guard catches a call;
* the API: digits, form, dialects, empty input.
"""
import ast
import math
import os
import re
from decimal import Decimal
from fractions import Fraction

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
PI = '3.141592653589793'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


def show(eng, text, **kw):
    """The display of the canonical state of an infix text."""
    return eng.prettify(eng.simplify(text), **kw)


# The plan's table (NUMBER_SPELLING_PLAN.md section 1, D = 3), row by row.
TABLE = [
    (f'1/(2*{PI})', '1/(2*pi)'),
    (f'3/(2*{PI})', '3/(2*pi)'),
    (f'x0/(2*{PI})', 'x0/(2*pi)'),
    (f'exp(-x0^2/2)/rootn(2*{PI}, 2)', '0.3989422804014327*exp(-x0^2/2)'),  # Feynman I.6.2a
    # Feynman 58: the rest beside pi is written as `simplify` writes 3/20
    (f'3*x0^2/(20*{PI}*x1*x2)', '0.15*x0^2/(pi*x1*x2)'),
    ('x1/2', 'x1/2'),
    ('x1/12', 'x1/12'),
    ('x1/3.7', 'x1/3.7'),
    ('x1*5/21', 'x1/4.2'),
    ('22/7*x1', '22*x1/7'),
    ('1/16', '1/16'),
    ('x1/1024', '9.765625e-4*x1'),
    ('x1^(5/21)', 'x1^(5/21)'),
    # The plan writes this row's base `x2 + 0.5` by hand; rule 2 keeps a fraction that fits,
    # as for `x1/2`, so the base is `x2 + 1/2`. The row is about the exponent (rule 7).
    ('(x2 + 0.5)^(500000000000000/4427)', '(x2 + 1/2)^1.1294330246216399e11'),
    ('x1^2*x2^2/(6*3.1415926535897*8.854e-12*(2.99792458e8)^3)', '2.2238078067157093e-16*x1^2*x2^2'),  # Larmor
    ('x1*20847313721190114925292351/1875000000000000000000000000', '0.011118567317968061*x1'),
    ('sin(sin(9463/890))', '-0.8045036621463277'),
    ('x1*6.626e-34', '6.626e-34*x1'),
    ('x1*1e-40*3', '3e-40*x1'),
    ('x1*1e200*1e200', '1e200*1e200*x1'),
    ('x1^(-1/2)', '1/rootn(x1, 2)'),
    ('x1/0.225147640565301', '4.441529999999994*x1'),
    # rule 5: a divisor float64 holds as written stays a divisor beside another divisor (the
    # sign moves out: the zero set of `rootn(x1, 2) + sin(x2)` is null on its domain)
    ('(3.44153*x2 + 0.774852359434699*rootn(x1, 2))/(-0.225147640565301*(rootn(x1, 2) + sin(x2)))',
     '-(3.44153*x2 + 0.774852359434699*rootn(x1, 2))/(0.225147640565301*(rootn(x1, 2) + sin(x2)))'),
]


@pytest.mark.parametrize('text,expected', TABLE, ids=[t for t, _ in TABLE])
def test_the_plans_table(eng, text, expected):
    assert show(eng, text) == expected


# The digit budget decides only whether a number is written exactly or as its float; an exact
# number is written as `simplify` writes it.
@pytest.mark.parametrize('text,expected', [
    ('x1*3/10', '0.3*x1'),
    ('x2 + 0.5', 'x2 + 1/2'),
    ('x1/5', '0.2*x1'),
    ('x1/20', '0.05*x1'),
    ('22*x1/7', '22*x1/7'),
    ('x1*5/21', 'x1/4.2'),
    ('355/113*x1', '3.1415929203539825*x1'),
    ('x1/200', '0.005*x1'),
    ('x1^(3/10)', 'x1^0.3'),
    (f'0.3*{PI}*x1', '0.3*pi*x1'),
    (f'x1/(2*{PI})', 'x1/(2*pi)'),
])
def test_an_exact_number_is_written_as_simplify_writes_it(eng, text, expected):
    assert show(eng, text) == expected


def test_within_the_budget_the_display_is_simplifys_text(eng):
    """Every coefficient and every term `p/q` with `p, q < 30`: where `simplify` writes the
    number within the digit budget (its digits, and the 1 of `x1/d`), the display is
    `simplify`'s own text."""
    compared = 0
    for p in range(1, 30):
        for q in range(2, 30):
            if math.gcd(p, q) != 1:
                continue
            for text in (f'{p}/{q}*x1', f'x2 + {p}/{q}'):
                answer = eng.simplify(text)
                digits = sum(c.isdigit() for c in re.sub(r'x\d', '', answer))
                if digits + answer.startswith('x1/') <= 3:
                    assert eng.prettify(answer) == answer, (text, answer)
                    compared += 1
    assert compared == 596, compared


@pytest.mark.parametrize('text,explicit,tagged', [
    (f'1/(2*{PI})', ['/', '1', '*', '2', 'np.pi'], ['<mul>', '1', '<div>', '2', 'np.pi', '</mul>']),
    ('22/7*x1', ['/', '*', '22', 'x1', '7'], ['<mul>', '22', 'x1', '<div>', '7', '</mul>']),
    ('-x1/12', ['/', 'neg', 'x1', '12'], ['<mul>', '-1', 'x1', '<div>', '12', '</mul>']),
    ('x1/1024', ['*', '9.765625e-4', 'x1'], ['<mul>', '9.765625e-4', 'x1', '</mul>']),
    ('x1^(-1/2)', ['/', '1', 'rootn', 'x1', '2'], ['<mul>', '1', '<div>', 'rootn', 'x1', '2', '</mul>']),
])
def test_token_forms(eng, text, explicit, tagged):
    c = eng.simplify(text)
    assert eng.prettify(c, form='explicit') == explicit
    assert eng.prettify(c, form='tagged') == tagged


def test_long_literals_show_their_float(eng):
    """An exact literal longer than a float shows as the float64 nearest to it."""
    assert show(eng, '2.14583142772210003669120562399025*x1') == '2.1458314277221002*x1'
    assert show(eng, 'x1^(300000000000000000001/600000000000000000000)') == 'rootn(x1, 2)'


# A number shown as a float is read as a float. At full precision a number that would show as a
# short exact number it is not gets its rest, where the rest fits the digit budget: the display
# keeps its value. At a precision the reader asked for k digits, and a rest is written only where
# the shortened number would change the structure.
REST_FITS = [
    ('x1 + 1.000000000000000000003', 'x1 + 1 + 3e-21'),
    ('x1 - 1.000000000000000000003', 'x1 - 1 - 3e-21'),
    ('x1 + 0.10000000000000000001', 'x1 + 0.1 + 1e-20'),
    ('x1*sin(1.000000000000000000003)', 'x1*sin(1 + 3e-21)'),
    ('x1^(0.50000000000000000001)', 'x1^(1/2 + 1e-20)'),
    ('0.99999999999999999999*x1', '(1 - 1e-20)*x1'),
    # the budget counts significant digits, not the exponent's: 1 and 3.1e-21 write three
    ('x1 + 1.0000000000000000000031', 'x1 + 1 + 3.1e-21'),
]
NO_REST = [
    # a number shown as a float: a fitted constant with 17 significant digits, one unit in the
    # 17th digit from its shortest float, and a long float whose rest would fit
    ('0.0009853704666076428*x1 - 0.0088844026345499877', None,
     '9.853704666076429e-4*x1 - 0.008884402634549989'),
    ('2.1458314277221002000000000001*x1', None, '2.1458314277221002*x1'),
    ('2.14583142772210003669120562399025*x1', None, '2.1458314277221002*x1'),
    # a rest beyond the budget (1/2 and 1/6e20 write four)
    ('x1 + 1.00000000000000000000314', None, 'x1 + 1'),
    ('x1^(300000000000000000001/600000000000000000000)', None, 'rootn(x1, 2)'),
    # at a precision
    ('x1 + 1.000000000000000000003', 3, 'x1 + 1'),
]


@pytest.mark.parametrize('text,expected', REST_FITS, ids=[t for t, _ in REST_FITS])
def test_a_number_that_would_show_as_a_short_number_it_is_not_gets_its_rest(eng, text, expected):
    assert show(eng, text) == expected
    assert eng.prettify(expected) == expected
    assert structure(eng, eng.simplify(text)) == (expected, True)
    # the display reads back to the state's own numbers
    assert eng.simplify(expected) == eng.simplify(text)


# A number is written in the first form that fits the budget of significant digits: exactly, as a
# short exact sum (its first digits and the rest), or as its float. Exponents count no digits.
SHORT_SUMS = [
    ('x1*(2.5e307 + 1e170)', 'f64', '(2.5e307 + 1e170)*x1'),
    ('x1*(2.5e30 + 1e10)', 'f64', '(2.5e30 + 1e10)*x1'),
    ('x1*(2.5e30 + 1e15)', 'f64', '(2.5e30 + 1e15)*x1'),
    ('x1*(1e20 + 1)', 'f64', '(1e20 + 1)*x1'),
    ('x1 + 1.000000000000000000003', 'f64', 'x1 + 1 + 3e-21'),
    ('0.30000000000000004*x1', 'f64', '(0.3 + 4e-17)*x1'),
    # a decimal float64 writes exactly in 15 digits is written as its float, not as a sum
    ('x1*exp(1905)', 'f64', 'x1*exp(1905)'),
    ('1.0001*x1', 'f64', '1.0001*x1'),
    ('x1*1e10', 'f64', '1e10*x1'),
]


@pytest.mark.parametrize('text,mode,expected', SHORT_SUMS, ids=[t for t, _, _ in SHORT_SUMS])
def test_a_number_is_written_exactly_as_a_short_sum_or_as_its_float(eng, text, mode, expected):
    answer = eng.simplify(text, mode=mode)
    assert eng.prettify(answer, mode=mode) == expected
    assert eng.prettify(expected, mode=mode) == expected
    # every display here is exact: it reads back to the answer itself
    assert eng.simplify(expected, mode=mode) == answer


def test_a_precision_writes_a_float_of_its_digits_not_a_short_sum(eng):
    # the reader asked for k digits: `0.043` for 0.043000000000000003, not `0.043 + 3e-18`
    assert show(eng, '0.043000000000000003*x1') == '(0.043 + 3e-18)*x1'
    assert show(eng, '0.043000000000000003*x1', precision=3) == '0.043*x1'
    assert show(eng, 'x1*(1e20 + 1)', precision=3) == '1e20*x1'
    # a rest is one number: 0.07980000000000001 is no `0.08 + 1e-17 - 2e-4`
    assert show(eng, '0.07980000000000001*x1') == '0.07980000000000001*x1'


@pytest.mark.parametrize('text,precision,expected', NO_REST, ids=[f'{t}-{p}' for t, p, _ in NO_REST])
def test_a_float_a_rest_beyond_the_budget_and_a_precision_write_no_rest(eng, text, precision, expected):
    assert show(eng, text, precision=precision) == expected
    assert eng.prettify(expected, precision=precision) == expected
    assert structure(eng, eng.simplify(text), precision) == (expected, True)


@pytest.mark.parametrize('text,precision,expected', [
    ('2.14583142772210003669120562399025*x1', None, '2.1458314277221002*x1'),
    ('2.14583142772210003669120562399025*x1', 3, '2.15*x1'),
    ('2.14583142772210003669120562399025*x1', 1, '2*x1'),
    ('x1/1024', 2, '9.8e-4*x1'),
    # 1.13e11 would be an integer exponent, and so would its three-digit rest: the rest in full
    ('(x2 + 0.5)^(500000000000000/4427)', 3, '(x2 + 1/2)^(1.13e11 - 5.669753783600633e7)'),
    ('x1*sin(9463/890)', 4, '-0.9348*x1'),
    ('x1*exp(-381/2)', 3, '1.85e-83*x1'),
    ('x1*exp(-381/2)', None, '1.8488479557720672e-83*x1'),
    ('9.9999*x1', 2, '10*x1'),
    ('1.000000000000000000003*x1', 3, '(1 + 3e-21)*x1'),
    ('x1/0.225147640565301', 3, '4.44*x1'),
    # a shortened float is written as the number it reads back as: no divisor unless its
    # reciprocal holds within the precision, and an exact number as the core writes it
    ('(3.44153*x2 + x1)/(0.225147640565301*(x1 + sin(x2)))', 3,
     '4.44*(x1 + 3.44*x2)/(x1 + sin(x2))'),
    ('sin(x1 + sin(1219/6934))', 3, 'sin(x1 + 7/40)'),
    ('x1/sinh(log(90805/4514))/(x2 + 1)', 3, '0.0997*x1/(x2 + 1)'),
])
def test_precision(eng, text, precision, expected):
    """`precision=k` shows a number written as a float with k significant digits; `None` keeps
    the shortest decimal that reads back as the same float64."""
    assert show(eng, text, precision=precision) == expected


@pytest.mark.parametrize('text', ['x1*5/21', 'x1/12', '22/7*x1', 'x1^(5/21)', '3*x0^2/(20*x1*x2)'])
def test_precision_leaves_the_fractions_digits_keeps(eng, text):
    assert show(eng, text, precision=1) == show(eng, text)


# pi is factored out at every precision as at full precision, written as the symbol, and never
# shortened: a precision applies to the other numbers only.
PI_ROWS = [
    (f'1/(2*{PI})', '1/(2*pi)'),
    (f'x0/(2*{PI})', 'x0/(2*pi)'),
    (f'3*x0^2/(20*{PI}*x1*x2)', '0.15*x0^2/(pi*x1*x2)'),
    (f'0.3*{PI}*x1', '0.3*pi*x1'),
    (f'3*{PI}*x1/4', '3*pi*x1/4'),
    (PI, 'pi'),
    (f'x1 + 2*{PI}', 'x1 + 2*pi'),
    (f'x1^(2*{PI})', 'x1^(2*pi)'),
    (f'x1^(-2*{PI})', '1/x1^(2*pi)'),
    (f'x1^{PI}', 'x1^pi'),
    (f'x1*sin(x2/(2*{PI}))', 'x1*sin(x2/(2*pi))'),
    (f'sin(x1/(2*{PI})) - sin(x1/(3*{PI}))', 'sin(x1/(2*pi)) - sin(x1/(3*pi))'),
    (f'exp(-x0^2/2)/(2*{PI}*x1)', 'exp(-x0^2/2)/(2*pi*x1)'),
    # read back, a term with the symbol takes its place in the canonical order of the sum
    (f'{PI}*x2 + sin(x1)', 'sin(x1) + pi*x2'),
    (f'sin(x1 + 2*{PI}) + x2/(2*{PI})', 'sin(x1 + 2*pi) + x2/(2*pi)'),
]


@pytest.mark.parametrize('precision', [None, 3, 1])
@pytest.mark.parametrize('text,expected', PI_ROWS, ids=[t for t, _ in PI_ROWS])
def test_pi_stays_pi_at_every_precision(eng, text, expected, precision):
    assert show(eng, text, precision=precision) == expected
    assert eng.prettify(expected, precision=precision) == expected
    assert structure(eng, eng.simplify(text), precision) == (expected, True)


def test_a_precision_shortens_the_numbers_beside_pi(eng):
    text = f'2.14583142772210003669120562399025*x1 + x2/(2*{PI})'
    assert show(eng, text, precision=3) == '2.15*x1 + x2/(2*pi)'
    assert show(eng, text) == '2.1458314277221002*x1 + x2/(2*pi)'


@pytest.mark.parametrize('text,explicit,tagged', [
    (f'x1/(2*{PI})', ['/', 'x1', '*', '2', 'np.pi'], ['<mul>', 'x1', '<div>', '2', 'np.pi', '</mul>']),
    (f'0.3*{PI}*x1', ['*', '0.3', '*', 'np.pi', 'x1'], ['<mul>', '0.3', 'np.pi', 'x1', '</mul>']),
])
def test_the_token_forms_write_the_engines_pi_token(eng, text, explicit, tagged):
    c = eng._core.parse(eng.simplify(text), True, False, True)
    assert eng.prettify(c) == explicit
    assert eng.prettify(eng.to_tagged(c)) == tagged
    for shown in (explicit, tagged):
        assert eng.prettify(shown) == shown


@pytest.mark.parametrize('text', [f'x1/(2*{PI})', f'3*x0^2/(20*{PI}*x1*x2)', f'x1^(2*{PI})',
                                  f'x1*sin(x2/(2*{PI})) + x2/(3*{PI})'])
def test_pi_reads_back_as_the_literals_float64(eng, text):
    """The symbol evaluates to the float64 the literal reads as, exactly, so the display
    evaluates as the state does up to the order of its operations."""
    import numpy as np
    c = list(eng.simplify(eng._core.parse(text, strict=True)))
    p = list(eng.prettify(c))
    assert 'np.pi' in p and PI not in p
    literal = [PI if t == 'np.pi' else t for t in p]
    names = sorted({t for t in c + p if re.fullmatch(r'x\d+', t)})
    xs = np.random.default_rng(3).uniform(0.5, 2.0, size=(32, len(names)))
    assert np.array_equal(values(eng, p, names, xs), values(eng, literal, names, xs))
    assert np.allclose(values(eng, c, names, xs), values(eng, p, names, xs), rtol=P2_RTOL, atol=0)


# The literal is written where the symbol would read back as another structure: the canonical
# constructors build `sin(x1 + pi)` as `-sin(x1)`. And where the expression itself holds the symbol
# pi, a multiple of the literal keeps the literal: written alike, the two would read back as one
# (two sines that cancel, `pi*pi` that is `pi^2`).
MIXED = [
    (f'sin(x1 + {PI})', f'sin(x1 + {PI})'),
    (f'cos(x1 + {PI})*x2', f'x2*cos(x1 + {PI})'),
    (f'sin(pi*x1) - sin({PI}*x1)', f'sin(pi*x1) - sin({PI}*x1)'),
    (f'pi*x1*{PI}', f'{PI}*pi*x1'),
    (f'x1^pi + x1^{PI}', f'x1^{PI} + x1^pi'),
    (f'x1/pi + x1/(2*{PI})', f'x1/(2*{PI}) + x1/pi'),
    ('pi*x1', 'pi*x1'),
]


@pytest.mark.parametrize('precision', [None, 3])
@pytest.mark.parametrize('text,expected', MIXED, ids=[t for t, _ in MIXED])
def test_beside_the_symbol_pi_the_literal_stays(eng, text, expected, precision):
    assert show(eng, text, precision=precision) == expected
    assert eng.prettify(expected, precision=precision) == expected
    assert structure(eng, eng.simplify(text), precision) == (expected, True)


def test_digit_budget(eng):
    assert show(eng, 'x1/12', digits=2) == '0.08333333333333333*x1'
    assert show(eng, 'x1/1024', digits=5) == 'x1/1024'
    assert show(eng, '3*x1/1024', digits=5) == '3*x1/1024'
    assert show(eng, '1234*x1') == '1234*x1'  # a float that writes as an integer
    assert show(eng, '1234567*x1') == '1.234567e6*x1'


@pytest.mark.parametrize('text,expected', [
    ('x1*exp(-381/2)', '1.8488479557720672e-83*x1'),
    ('x1*sin(sin(9463/890))', '-0.8045036621463277*x1'),
    ('x1*sin(9463/890)', '-0.9348393354311442*x1'),
    ('x1 + sin(9463/890)', 'x1 - 0.9348393354311442'),
    ('x1*sin(x2 + sin(9463/890))', 'x1*sin(x2 - 0.9348393354311442)'),
    # a float whose reciprocal holds as written stays a divisor beside another divisor
    ('x1/sin(9463/890)/x2', '-x1/(0.9348393354311442*x2)'),
    # an integer power the state keeps for its size: the power of the float its base is shown as
    ('x1*0.12345670000000000001^40', '4.577059556136647e-37*x1'),
])
def test_a_function_of_floats_is_one_float(eng, text, expected):
    """A function of numbers that are all shown as floats is the float the evaluator computes."""
    assert show(eng, text) == expected


# The fold is judged on the parts as the display writes them: the inner function folds and
# combines with the exact 0.4 (or 2) into one float, so the outer function is a function of one
# float. A folded term takes its place in the sum as reading the display places it.
@pytest.mark.parametrize('text,expected', [
    ('x1*asinh(atan(7452/1615) + 0.4)', '1.3295516895428026*x1'),
    ('x1*sin(cosh(cosh(13429/3083)) + 2)', '0.9877111802133507*x1'),
    ('x2 + 1.5455017691921438*x1*sin(cos(40148/6487))', '1.2967805889953927*x1 + x2'),
    # the power of the float a long base is shown as is within the size cap: reading it combines
    ('x1*0.12345670000000000001^40', '4.577059556136647e-37*x1'),
])
def test_a_fold_is_its_own_display(eng, text, expected):
    assert show(eng, text) == expected
    c = eng._core.parse(eng.simplify(text), True, False)
    for form in ('infix', 'explicit', 'tagged'):
        p = eng.prettify(in_form(eng, c, form))
        assert eng.prettify(p) == p, (form, p)


@pytest.mark.parametrize('text,expected', [
    # a number written exactly keeps its function
    ('sin(2) + x1', 'x1 + sin(2)'),
    ('x1*sin(0.3)', 'x1*sin(0.3)'),
    ('x1*sin(2 + 9463/890*x2)', 'x1*sin(10.632584269662921*x2 + 2)'),
    # an odd function of a negative literal gives its sign to the joiner, as in `simplify`'s text
    ('x1 + sin(-2)', 'x1 - sin(2)'),
    ('3^5000*x1', '3^5000*x1'),
    ('x1*2^(1/3)', '2^(1/3)*x1'),
    # a power beyond the size cap stays a power of its base, written as a float
    ('x1*0.12345670000000000001^100', '0.1234567^100*x1'),
    # a float that would be infinite or subnormal: the function stays
    ('x1*exp(1905)', 'x1*exp(1905)'),
    ('x1*exp(-1441/2)', 'x1*exp(-720.5)'),
])
def test_any_other_function_of_numbers_stays(eng, text, expected):
    assert show(eng, text) == expected


# A fold reads back as a number, which the canonical constructors can build into a sum otherwise
# than the function stood: distributed over a sum, collected with a like term. Such a fold is not
# made and the function stays. A fold that only turns a sum around (`cos(c) - x2` is
# `-(x2 + 0.997)`) is made, and the display is of the sum reading it back builds. Numbers a
# product keeps apart for their size (`real` admits a number of at most 1,100 bits) are one
# float: the product of the floats they are written as, which reading the display combines.
LONG = '0.' + '1234567891' * 33
CONTRACT = [
    ('x1 + atan(1.2345678)*(x2 + x3)', 'f64', None, 'x1 + atan(1.2345678)*(x2 + x3)'),
    ('x1 + atan(1.2345678)*(x2 + x3)', 'f64', 3, 'x1 + atan(1.23)*(x2 + x3)'),
    ('x1 + x2 + x1/log(0.7377233226141932)', 'real', None, 'x1 + x2 + x1/log(0.7377233226141932)'),
    ('x1 + x2 + x1/log(0.7377233226141932)', 'real', 3, 'x1 + x2 + x1/log(0.738)'),
    # distributed and its numbers combined: as many terms, another structure
    ('x2*(0.35 - exp(23.145219203872042)*(0.0087/x1 + 0.113))', 'real', None,
     'x2*(0.35 - exp(23.145219203872042)*(0.0087/x1 + 0.113))'),
    ('x3/(x1*(cos(273.2392845492178) - x2))', 'real', None, 'x3/(-x1*(x2 + 0.9968592784913302))'),
    ('x3/(x1*(cos(273.2392845492178) - x2))', 'real', 3, 'x3/(-x1*(x2 + 0.997))'),
    (f'x1 + {LONG}*0.893435038202104*x2', 'real', None, 'x1 + 0.11030062109689767*x2'),
    (f'x1 + {LONG}*0.893435038202104*x2', 'real', 3, 'x1 + 0.11*x2'),
]


@pytest.mark.parametrize('text,mode,precision,expected', CONTRACT,
                         ids=[f'{t[:40]}-{m}-{p}' for t, m, p, _ in CONTRACT])
def test_the_contract_holds_where_a_fold_or_a_size_would_break_it(eng, text, mode, precision, expected):
    """The display reads back to the structure, is its own display, and at full precision
    evaluates as the expression."""
    import numpy as np
    from simplipy.engine import _RULE_MODE, _coerce_mode
    shown = eng.prettify(text, mode=mode, precision=precision)
    assert shown == expected
    assert eng.prettify(shown, mode=mode, precision=precision) == shown
    tokens = list(eng._core.parse(text, True, False, True))
    rule_mode = _RULE_MODE[_coerce_mode(mode)]
    _, kept = eng._core.prettify(tokens, 3, 'infix', precision, rule_mode=rule_mode, check_structure=True)
    assert kept
    if precision is None:
        back = list(eng._core.parse(shown, True, False, True))
        names = sorted({t for t in tokens if re.fullmatch(r'x\d+', t)})
        xs = np.random.default_rng(0).uniform(0.5, 2.0, size=(32, len(names)))
        assert np.allclose(values(eng, tokens, names, xs), values(eng, back, names, xs),
                           rtol=P2_RTOL, atol=0)


@pytest.mark.parametrize('mode', ['f64', 'real'])
@pytest.mark.parametrize('precision', [None, 3])
def test_a_fold_in_a_sum_keeps_the_structure_on_a_fuzz(eng, mode, precision):
    """Functions of numbers beside a sum, a like term and a sign: every display reads back to
    the structure and is its own display."""
    from simplipy.engine import _RULE_MODE, _coerce_mode
    shapes = ['x1 + {g}*(x2 + x3)', 'x1 + x2 + x1/{g}', 'x3/(x1*({g} - x2))', 'x1 + {g}*x1*(x2 - x3)',
              'x2*(0.35 - {g}*(0.0087/x1 + 0.113))']
    rule_mode = _RULE_MODE[_coerce_mode(mode)]
    lost = []
    for text in fold_fuzz(200, seed=5, shapes=shapes):
        tokens = list(eng._core.parse(text, True, False, True))
        shown, kept = eng._core.prettify(tokens, 3, 'infix', precision, rule_mode=rule_mode,
                                         check_structure=True)
        if not kept or eng.prettify(shown, mode=mode, precision=precision) != shown:
            lost.append((text, shown))
    assert not lost, lost[:5]


# prettify changes how a number is written, never the structure. A number whose short spelling
# would let reading the display build another structure -- a multiplier or an exponent of 1 that
# drops, an integer where a non-integer exponent stood, two parts made alike -- is written as its
# short spelling and the rest, `s + r` or `s - r`, the rest by the same rules.
SPLITS = [
    ('sin(1.00000000000000000001*x1) - sin(x1)', None, 'sin((1 + 1e-20)*x1) - sin(x1)'),
    ('x3 + (154104.1409857162*x1)^(3.00000000000000000001)', None,
     'x3 + (154104.1409857162*x1)^(3 + 1e-20)'),
    ('sin(1.0001*x1) - sin(x1)', 3, 'sin((1 + 1e-4)*x1) - sin(x1)'),
    ('sin(1.0001*x1) - sin(x1)', None, 'sin(1.0001*x1) - sin(x1)'),
    ('2.14583142772210003669120562399025*x1', None, '2.1458314277221002*x1'),
    ('2.14583142772210003669120562399025*x1', 3, '2.15*x1'),
    # two numbers with one float: the one written first keeps it, the other is it plus its rest
    ('sin(2.14583142772210003669120562399025*x1) - sin(2.1458314277221002*x1)', None,
     'sin((2.1458314277221002 - 1.6330879437600975e-16)*x1) - sin(2.1458314277221002*x1)'),
    ('0.99999999999999999999*x1', None, '(1 - 1e-20)*x1'),
    ('1.000000000000000000003*x1', None, '(1 + 3e-21)*x1'),
    ('x1^(3.00000000000000000001)', None, 'x1^(3 + 1e-20)'),
    ('x1^(1.00000000000000000001)*x2', None, 'x1^(1 + 1e-20)*x2'),
    ('x1^(600000000000000000001/300000000000000000000)', None, 'x1^(2 + 1/3e20)'),
    ('x3 + (1815963197375679774584911*x1/1.1784e19)^(9000000000000000000001/3000000000000000000000)',
     None, 'x3 + (154104.1409857162*x1)^(3 + 1/3e21)'),
    # a constant term written as `s + r` is two terms of its sum
    ('(1.12345*x1 + 111.0001)*(1.12346*x1 + 111.0002)', 3, '(1.12*x1 + 111)*(1.12*x1 + 111 + 2e-4)'),
]


@pytest.mark.parametrize('text,precision,expected', SPLITS, ids=[f'{t}-{p}' for t, p, _ in SPLITS])
def test_a_number_that_would_change_the_structure_is_written_with_its_rest(
        eng, text, precision, expected):
    assert show(eng, text, precision=precision) == expected
    assert eng.prettify(expected, precision=precision) == expected
    shown, kept = structure(eng, eng.simplify(text), precision)
    assert (shown, kept) == (expected, True)


def test_the_rest_in_the_token_forms(eng):
    c = eng._core.parse(eng.simplify('sin(1.00000000000000000001*x1) - sin(x1)'), True, False, True)
    assert eng.prettify(c) == ['-', 'sin', '*', '+', '1', '1e-20', 'x1', 'sin', 'x1']
    assert eng.prettify(eng.to_tagged(c)) == [
        '<add>', 'sin', '<mul>', '<add>', '1', '1e-20', '</add>', 'x1', '</mul>', '<sub>', 'sin', 'x1',
        '</add>']


def test_divisors(eng):
    """Rule 5: a float whose reciprocal float64 holds as written stays a divisor beside another
    divisor; a bare one is a multiplier."""
    assert show(eng, 'x1/243/x2') == 'x1/(243*x2)'
    assert show(eng, 'x1/243') == '0.00411522633744856*x1'
    assert show(eng, 'x0/(x1^2*x2)') == 'x0/(x1^2*x2)'


@pytest.mark.parametrize('text,expected', [
    ('x1/x2/log(x3)', 'x1/(x2*log(x3))'),
    ('x1/sin(x3)/(x2 + 1)', 'x1/(sin(x3)*(x2 + 1))'),
    ('x1/x2/(x3 + 1)', 'x1/(x2*(x3 + 1))'),
    ('x1/x2/x3^2', 'x1/(x2*x3^2)'),
    ('x0/(cos(x1)*x2)', 'x0/(x2*cos(x1))'),
    ('1/x1^(1/2)/sin(1/x1)', '1/(rootn(x1, 2)*sin(1/x1))'),
    ('-x1/(3*x2)/(x3 + 1)', '-x1/(3*x2*(x3 + 1))'),
])
def test_one_slash(eng, text, expected):
    """Every divisor of a product joins one denominator product, functions and sums included."""
    assert show(eng, text) == expected


def test_a_joined_reciprocal_keeps_its_sign_and_its_tagged_bag(eng):
    """A joined reciprocal's sign stays in the denominator, where the state holds it; the tagged
    section, which inverts each member on its own, keeps a joined reciprocal of a sum one bag."""
    assert eng.prettify('x3/(-(x11 + x9))') == 'x3/(-(x11 + x9))'
    assert eng.prettify(['/', 'x3', 'neg', '+', 'x11', 'x9']) == ['/', 'x3', 'neg', '+', 'x11', 'x9']
    assert eng.prettify('x1/(x2*(x3 + 1))', form='tagged') == [
        '<mul>', 'x1', '<div>', '<mul>', 'x2', '<add>', 'x3', '1', '</add>', '</mul>', '</mul>']


def test_a_number_beside_a_fitted_constant_keeps_its_place(eng):
    """A fitted constant absorbs a coefficient beside it, so the number of a joined reciprocal
    stays in the denominator: the display loses no number, and it is its own display."""
    import numpy as np
    text = '<constant>*x5/(0.2*(rootn(x2, 2) + <constant>*x7))'
    p = eng.prettify(text)
    assert p == text  # the number written as `simplify` writes it
    assert eng.prettify(p) == p
    names, xs = ['x2', 'x5', 'x7'], np.random.default_rng(1).uniform(0.5, 2.0, size=(16, 3))
    a = values(eng, eng._core.parse(text, strict=True), names, xs)
    b = values(eng, eng._core.parse(p, strict=True), names, xs)
    assert np.allclose(a, b, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- idempotence and read-back
def corpus():
    from simplipy.verify import default_corpus
    return default_corpus()


def in_form(eng, c, form):
    return eng.to_infix(c) if form == 'infix' else eng.to_tagged(c) if form == 'tagged' else c


def displays_that_move(eng, rows, form, display=None):
    """The rows whose display is not its own display: `display(p) != p`."""
    display = display or eng.prettify
    moved = []
    for row in rows:
        p = display(in_form(eng, eng.simplify(row), form))
        if display(p) != p:
            moved.append((row, p, display(p)))
    return moved


def structure(eng, expression, precision=None, keep_grouping=False):
    """`(display, kept)`: whether the display reads back, through the canonical constructors, to
    the shape of the state after the display's folds."""
    form, tokens = eng._read_input(expression)
    return eng._core.prettify(tokens, 3, form, precision, rule_mode='default', check_structure=True,
                              keep_grouping=keep_grouping)


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
def test_p1_displaying_a_display_gives_it_back_on_the_corpus(eng, form):
    moved = displays_that_move(eng, corpus(), form)
    assert not moved, moved[:5]


FUNS = ['sin', 'cos', 'exp', 'log', 'tanh', 'atan', 'sinh', 'asinh', 'cosh']


def fold_fuzz(n, seed=11, shapes=None):
    """Expressions whose canonical state keeps a function of numbers `{g}`, in the shapes of a
    product, a sum, a quotient, a reciprocal of a product, a function argument and an exponent
    (or in `shapes`)."""
    import random
    rng = random.Random(seed)

    def number():
        kind = rng.random()
        if kind < 0.4:
            return f'{rng.randint(1, 99999)}/{rng.randint(7, 9999)}'
        if kind < 0.7:
            return repr(rng.uniform(-50, 50))
        if kind < 0.85:
            return str(rng.randint(-9, 9))
        return f'{rng.randint(1, 9)}/{rng.choice([2, 3, 5, 10])}'

    def ground(depth):
        if depth == 0:
            return number()
        inner = ground(depth - 1) if rng.random() < 0.6 else number()
        if rng.random() < 0.3:
            inner = f'{inner} + {number()}'
        return f'{rng.choice(FUNS)}({inner})'

    out = []
    for _ in range(n):
        g = ground(rng.randint(1, 3))
        if shapes:
            out.append(shapes[rng.randint(0, len(shapes) - 1)].format(g=g))
            continue
        out.append([f'x1*{g}', f'x1 + {g}', f'x1/{g}/x2', f'x1/({g}*(x2 + 1))', f'sin(x1 + {g})',
                    f'{g}*{ground(1)}*x1 + x2', f'x1^({g})'][rng.randint(0, 6)])
    return out


@pytest.fixture(scope='module')
def fuzz_states(eng):
    states = []
    for text in fold_fuzz(400):
        try:
            states.append(list(eng.simplify(eng._core.parse(text, True, False))))
        except ValueError:
            continue
    assert len(states) > 300
    return states


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
@pytest.mark.parametrize('precision', [None, 3])
def test_p1_displaying_a_display_gives_it_back_on_a_fold_fuzz(eng, fuzz_states, form, precision):
    moved = []
    for c in fuzz_states:
        p = eng.prettify(in_form(eng, c, form), precision=precision)
        if eng.prettify(p, precision=precision) != p:
            moved.append((eng.to_infix(c), p))
    assert not moved, moved[:5]


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
def test_p1_displaying_a_display_gives_it_back_at_a_precision_on_the_corpus(eng, form):
    moved = []
    for row in corpus():
        p = eng.prettify(in_form(eng, eng.simplify(row), form), precision=3)
        if eng.prettify(p, precision=3) != p:
            moved.append((row, p))
    assert not moved, moved[:5]


def test_p1_check_catches_a_display_that_moves(eng):
    """The gate can fail. A display that wrote the short spelling alone would make the two sines
    alike, and reading it back collects them."""
    row = 'sin(1.00000000000000000001*x1) - sin(x1)'

    def short(expression):
        return eng.prettify(expression).replace('(1 + 1e-20)*', '')

    assert displays_that_move(eng, [row], 'infix', short) == [(row, 'sin(x1) - sin(x1)', '0')]
    assert displays_that_move(eng, [row], 'infix') == []


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
@pytest.mark.parametrize('precision', [None, 3])
def test_the_structure_is_kept_on_the_corpus(eng, form, precision):
    lost = []
    for row in corpus():
        shown, kept = structure(eng, in_form(eng, eng.simplify(row), form), precision)
        if not kept:
            lost.append((row, shown))
    assert not lost, lost[:5]


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
@pytest.mark.parametrize('precision', [None, 3])
def test_the_structure_is_kept_on_a_fold_fuzz(eng, fuzz_states, form, precision):
    lost = []
    for c in fuzz_states:
        shown, kept = structure(eng, in_form(eng, c, form), precision)
        if not kept:
            lost.append((eng.to_infix(c), shown))
    assert not lost, lost[:5]


# Joined reciprocals of numbers, variables and their powers and roots: alike in every form.
GROUPS = ['x13/(3*x4*rootn(x10, 2))', '2*x1/(4.2*x2*x3^2)', 'x0/(3125*x1^5*x2^5)/sinh(x1)^5']
# Joined reciprocals holding functions and sums: alike in the forms with one denominator.
JOINED = ['x1/(x2*(x3 + 1))', 'x1/(sin(x3)*(x2 + 1))', '2*x1/(4.2*x2*sin(x3))',
          'x1/(x2*x3*(x4 + 1))/log(x5)']


@pytest.mark.parametrize('text,form', [(t, f) for t in GROUPS for f in ('infix', 'explicit', 'tagged')]
                         + [(t, f) for t in JOINED for f in ('infix', 'explicit')])
def test_p1_the_display_does_not_depend_on_how_the_state_holds_its_reciprocals(eng, text, form):
    """The input reads as one joined reciprocal (`(x2*(x3 + 1))^-1`), the answer of `simplify`
    holds a reciprocal per factor (`x2^-1*(x3 + 1)^-1`): the two display alike."""
    joined = eng.prettify(in_form(eng, eng._core.parse(text, strict=True), form))
    split = eng.prettify(in_form(eng, eng.simplify(eng._core.parse(text, strict=True)), form))
    assert joined == split


def floats(tokens):
    """The float64 values of the numerals of a token answer, in order."""
    out = []
    for t in tokens:
        try:
            d = Decimal(t)
        except Exception:
            continue
        if d.is_finite():
            out.append(float(Fraction(d)))
    return out


def values(eng, tokens, names, xs):
    """The float64 values of `tokens` at the points `xs`, every `<constant>` set to one value."""
    import numpy as np
    params = [1.2345] * sum(t == '<constant>' for t in tokens)
    return np.asarray(eng._core.evaluate_batch(list(tokens), names, xs.ravel().tolist(), len(xs), params))


# The display writes each number exactly or as the float64 the evaluator reads it as, and writes
# the same operations; only their order can differ (a sum's association, the order of divisions).
# Measured on the corpus at these points: at most 9.1e-13 apart, where exp(sinh(...)) amplifies a
# re-associated sum. The gate catches a number written short of its float64 to a few digits; the
# one-ulp test below is the fine check that a float is written in full.
P2_RTOL = 1e-9


def test_p2_the_display_evaluates_as_the_state_does_on_the_corpus(eng):
    import numpy as np
    rng = np.random.default_rng(0)
    apart = []
    for row in corpus():
        c = list(eng.simplify(row))
        p = list(eng.prettify(c))
        names = sorted({t for t in c + p if re.fullmatch(r'x\d+', t)})
        xs = rng.uniform(0.5, 2.0, size=(64, max(len(names), 1)))
        a, b = values(eng, c, names or ['x0'], xs), values(eng, p, names or ['x0'], xs)
        if not np.allclose(a, b, rtol=P2_RTOL, atol=0, equal_nan=True):
            apart.append((eng.to_infix(c), eng.to_infix(p)))
    assert not apart, apart[:5]


def test_p2_holds_at_full_precision_only(eng):
    """With `precision=k` a number no longer reads back as its float64: the gate catches it."""
    import numpy as np
    c = list(eng.simplify(eng._core.parse('x1/3 + 7.123456789012345*x2', strict=True)))
    xs = np.random.default_rng(0).uniform(0.5, 2.0, size=(16, 2))
    a = values(eng, c, ['x1', 'x2'], xs)
    assert np.allclose(a, values(eng, eng.prettify(c), ['x1', 'x2'], xs), rtol=P2_RTOL, atol=0)
    assert not np.allclose(a, values(eng, eng.prettify(c, precision=3), ['x1', 'x2'], xs),
                           rtol=P2_RTOL, atol=0)


@pytest.mark.parametrize('text', ['x1/0.225147640565301', 'x1/3 + 7.123456789012345*x2',
                                  'x1^2*x2^2/(6*3.1415926535897*8.854e-12*(2.99792458e8)^3)'])
def test_p2_catches_a_one_ulp_change(eng, text):
    c = list(eng.simplify(eng._core.parse(text, strict=True)))
    p = eng.prettify(c)
    i = max(range(len(p)), key=lambda k: len(p[k]) if floats([p[k]]) else -1)
    assert floats(eng.simplify(p)) == floats(eng.simplify(eng.prettify(eng.simplify(p))))
    bumped = list(p)
    bumped[i] = repr(math.nextafter(float(Fraction(Decimal(p[i]))), math.inf))
    assert floats(eng.simplify(bumped)) != floats(eng.simplify(p))


# ---------------------------------------------------------------- the grouping
# With `keep_grouping=True` every sum and product is written as SimpliPy's explicit tokens group
# it: a sum's positive terms grouped from the right, then each subtracted term in turn, and a
# product's factors grouped from the right. By default a sum is flat, and reading it groups it
# from the left.
GROUPED = [
    ('x1 + x2 + exp(x3)', 'x1 + x2 + exp(x3)', 'x1 + (x2 + exp(x3))'),
    ('2*x2 + 2*x3 + exp(x2 + x3) - x1', '2*x2 + 2*x3 + exp(x2 + x3) - x1',
     '2*x2 + (2*x3 + exp(x2 + x3)) - x1'),
    ('-x1 - x2 - x3', '-x1 - x2 - x3', '-x1 - (x2 + x3)'),
    ('x1 + x2 - x3 - x4', 'x1 + x2 - x3 - x4', 'x1 + x2 - x3 - x4'),
    ('2*x1*x2*x3', '2*x1*x2*x3', '2*(x1*(x2*x3))'),
    ('x1*x2/(x3*x4*x5)', 'x1*x2/(x3*x4*x5)', 'x1*x2/(x3*(x4*x5))'),
    ('sin(x1 + x2 + x3)*x4*x5', 'x4*x5*sin(x1 + x2 + x3)', 'x4*(x5*sin(x1 + (x2 + x3)))'),
    # a number written with its rest is one term
    ('x1 + 1.000000000000000000003', 'x1 + 1 + 3e-21', 'x1 + (1 + 3e-21)'),
]


@pytest.mark.parametrize('text,flat,grouped', GROUPED, ids=[t for t, _, _ in GROUPED])
def test_keep_grouping_writes_the_grouping_of_the_explicit_tokens(eng, text, flat, grouped):
    c = eng.simplify(text)
    assert eng.prettify(c) == flat
    assert eng.prettify(c, keep_grouping=False) == flat
    assert eng.prettify(c, keep_grouping=True) == grouped
    assert eng.prettify(grouped, keep_grouping=True) == grouped
    assert structure(eng, c, keep_grouping=True) == (grouped, True)
    tokens = eng._core.parse(c, True, False, True)
    # the display, explicit tokens or the infix read, evaluates as the state's own tokens
    import numpy as np
    state = list(eng.simplify(tokens))
    names = sorted({t for t in state if re.fullmatch(r'x\d+', t)})
    xs = np.random.default_rng(0).uniform(-2.0, 2.0, size=(16, len(names)))
    a = values(eng, state, names, xs)
    assert np.array_equal(a, values(eng, eng.prettify(tokens, keep_grouping=True), names, xs))
    assert np.array_equal(a, values(eng, eng._core.parse(grouped, True, False, True), names, xs))
    # the tagged form writes bags, which have no grouping
    assert eng.prettify(eng.to_tagged(tokens), keep_grouping=True) == eng.prettify(eng.to_tagged(tokens))


# Where the terms of a sum nearly cancel, or a function amplifies its argument, the flat display
# read from the left evaluates apart from the state; the grouped display evaluates as the state,
# bit for bit, in infix and in explicit tokens. The rows are srbf's (a ground truth, and parts of
# a prediction and of a ground truth).
APART = [
    '-cos(exp(2*x2 + 2*x3 + exp(x2 + x3) - x1))/(x2 + x3 + exp(x2 + x3) - x1)',
    'cos(221742068887.4986*x1 + 0.00331/(18.464400843893329*x2 - 7.698449108356519) + 62.24454675925565)',
    '0.7088195661797923*cos(21267013.501384924*x1^5*(x3 - 0.52035366738351248*x2)^5)/x1',
    # srbf ground truths whose coefficient is a quotient, 3.1415926535897e37/3313: its float is
    # one rounding apart, which sin at ~1e34 amplifies, so the grouped display writes it exactly
    'sin(x1*x2/(6.626e-34/(2*3.1415926535897)))^2',
    'sin(x2*x3/(6.626e-34/(2*3.1415926535897)))^2',
]


@pytest.mark.parametrize('text', APART)
def test_the_grouped_display_evaluates_as_the_state(eng, text):
    import numpy as np
    state = list(eng.simplify(eng._core.parse(text, True, False, True)))
    names = sorted({t for t in state if re.fullmatch(r'x\d+', t)})
    xs = np.random.default_rng(0).uniform(0.5, 2.0, size=(32, len(names)))
    a = values(eng, state, names, xs)

    def read(display):
        return values(eng, eng._core.parse(display, True, False, True), names, xs)

    flat = eng.prettify(eng.to_infix(state))
    assert not np.allclose(a, read(flat), rtol=P2_RTOL, atol=0)
    grouped = eng.prettify(eng.to_infix(state), keep_grouping=True)
    assert np.array_equal(a, read(grouped))
    assert np.array_equal(a, values(eng, eng.prettify(state, keep_grouping=True), names, xs))


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
@pytest.mark.parametrize('precision', [None, 3])
def test_with_the_grouping_idempotence_and_the_structure_hold(eng, fuzz_states, form, precision):
    moved, lost = [], []
    for c in [list(eng.simplify(row)) for row in corpus()] + fuzz_states:
        p, kept = structure(eng, in_form(eng, c, form), precision, keep_grouping=True)
        if not kept:
            lost.append((eng.to_infix(c), p))
        if eng.prettify(p, precision=precision, keep_grouping=True) != p:
            moved.append((eng.to_infix(c), p))
    assert not moved, moved[:5]
    assert not lost, lost[:5]


def test_keep_grouping_is_a_bool(eng):
    with pytest.raises(TypeError):
        eng.prettify('x1 + x2 + x3', keep_grouping=1)
    with pytest.raises(TypeError):
        eng.prettify('x1 + x2 + x3', keep_grouping='yes')


# ---------------------------------------------------------------- the source guard
MODULES = {os.path.join('rust', 'ac', 'prettify.rs'), os.path.join('rust', 'engine', 'prettify.rs')}
# The only other lines that may name it: the module declarations and the FFI method.
ALLOWED_RUST = {
    os.path.join('rust', 'ac', 'mod.rs'): {'pub mod prettify;'},
    os.path.join('rust', 'engine', 'mod.rs'): {'mod prettify;',
                                               'pub use prettify::{Pretty, PrettyForm, PrettyOptions};'},
    os.path.join('rust', 'lib.rs'): {'fn prettify(', 'self.inner.ac_prettify(&tokens, digits, form, mode, budget)'},
}
NAME = re.compile(r'prettify|PrettyForm|Pretty\b')


def guard_violations(root):
    """Every place under `root`'s `src/simplipy` and `rust` that reaches prettify, outside its
    module, the module declarations, the FFI method and the public method itself."""
    found = []
    for base in ('rust', os.path.join('src', 'simplipy')):
        for dirpath, _, files in os.walk(os.path.join(root, base)):
            for name in files:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, root)
                if rel in MODULES:
                    continue
                if name.endswith('.rs'):
                    allowed = ALLOWED_RUST.get(rel, set())
                    for n, line in enumerate(open(path, encoding='utf-8'), 1):
                        code = line.split('//')[0].strip()
                        if NAME.search(code) and not any(code.startswith(a) for a in allowed):
                            if not (rel == os.path.join('rust', 'lib.rs') and _inside_ffi(path, n)):
                                found.append(f'{rel}:{n}: {code}')
                elif name.endswith('.py'):
                    found += [f'{rel}:{n}' for n in _python_uses(path)]
    return found


def _inside_ffi(path, lineno):
    """Is line `lineno` of lib.rs inside the body of `fn prettify(`?"""
    lines = open(path, encoding='utf-8').read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith('fn prettify(')), None)
    if start is None or lineno - 1 < start:
        return False
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == '    }')
    return lineno - 1 <= end


def _python_uses(path):
    """Lines of a Python file that name prettify outside `def prettify` itself."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'prettify':
            allowed |= {n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')}
    uses = []
    for node in ast.walk(tree):
        named = (isinstance(node, ast.Attribute) and 'prettify' in node.attr) or \
                (isinstance(node, ast.Name) and 'prettify' in node.id)
        if named and node.lineno not in allowed:
            uses.append(node.lineno)
    return uses


def test_nothing_inside_simplipy_calls_prettify():
    assert guard_violations(REPO) == []


def test_the_source_guard_catches_a_call(tmp_path):
    for rel in ('rust/ac/expr.rs', 'src/simplipy/masking.py', 'rust/lib.rs'):
        os.makedirs(tmp_path / os.path.dirname(rel), exist_ok=True)
    (tmp_path / 'rust/ac/expr.rs').write_text(
        'fn f() {\n    let s = crate::ac::prettify::to_infix(&n);\n}\n')
    (tmp_path / 'src/simplipy/masking.py').write_text('def mask(e, x):\n    return e.prettify(x)\n')
    (tmp_path / 'rust/lib.rs').write_text('fn other() {\n    self.inner.ac_prettify(&t, 3, f, m, None)\n}\n')
    found = guard_violations(tmp_path)
    assert len(found) == 3, found


# ---------------------------------------------------------------- the API
def test_api(eng):
    c = eng.simplify('x1/12')
    assert eng.prettify(c) == 'x1/12'  # a str answers in infix
    assert eng.prettify(eng.simplify(['/', 'x1', '12'])) == ['/', 'x1', '12']  # tokens in tokens
    assert eng.prettify(eng.to_tagged(['/', 'x1', '12'])) == ['<mul>', 'x1', '<div>', '12', '</mul>']
    assert eng.prettify(['/', 'x1', '12'], form='infix') == 'x1/12'
    assert eng.prettify('') == ''
    assert eng.prettify([]) == []
    with pytest.raises(ValueError):
        eng.prettify(c, digits=0)
    with pytest.raises(TypeError):
        eng.prettify(c, digits=True)
    with pytest.raises(TypeError):
        eng.prettify(c, digits=2.5)
    with pytest.raises(ValueError):
        eng.prettify(c, form='prefix')
    with pytest.raises(ValueError):
        eng.prettify(['/', 'x1'])
    with pytest.raises(ValueError):
        eng.prettify(c, precision=0)
    with pytest.raises(TypeError):
        eng.prettify(c, precision=True)
    with pytest.raises(TypeError):
        eng.prettify(c, precision=2.5)


def test_prettify_never_changes_the_state(eng):
    """Display only: the canonical answer is the same before and after a display."""
    text = '2.14583142772210003669120562399025*x1'
    before = eng.simplify(text)
    eng.prettify(before)
    assert eng.simplify(text) == before == '2.14583142772210003669120562399025*x1'
