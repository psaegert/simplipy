"""THE WRITTEN PRICE (2026-11-26): `written_complexity` applies complexity()'s code book to an
expression exactly as written -- no mode, nothing read into a canonical state -- with only the
identifications infix notation forces. Owner's ruling (via the srbf session): srbf measures ground
truths and predictions "exactly as written, without any modifications that either benefit the
benchmark or the model"; "SimpliPy should get this capability and have both (measuring in the
canonical form and in the written form)"."""

import inspect
import os
import warnings

import pytest

from simplipy import SimpliPyEngine

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SimpliPyEngine.from_config(os.path.join(_ROOT, 'assets', 'engines', 'acj2-4', 'config.yaml'))


# (written, its written price in milli-bits)
_BRIEF = [
    ('x0 - x0/1 + 2', 23585),
    ('x0 + 0', 11000),
    ('x0 + x0', 15000),
    ('x0*x0', 15000),
    ('exp(1)', 8000),
    ('-(x0 - x1)', 17000),
    ('sqrt(x0)', 12000),
    ('x0**0.5', 12585),
    ('6.674e-11*x0*x1/x2**2', 45197),
    ('x0*x1/(4*np.pi*x2*x3**2)', 45907),
]
_MORE = [
    ('x0*1', 11000), ('1*x0', 11000), ('x0 + x1 + x2', 21000), ('(x0 + x1) + x2', 21000),
    ('x2 + (x1 + x0)', 21000), ('x0 - x1', 16000), ('x0/x1', 18000), ('1/x0', 9000), ('x0/3', 13000),
    ('x0**2', 11585), ('2*x0', 11585), ('2.0*x0', 11585), ('2.50*x0', 14170), ('x1 - x0', 16000),
    ('x0 - (x1 - x2)', 26000), ('x0 - x1 + x2', 22000), ('-(-x0)', 8000), ('x0/(2*x1)', 23585),
    ('0.5*x0/x1', 21585), ('x0*x1/x1', 24000), ('rootn(x0, 2)', 14585), ('x0**(1/2)', 12585),
    ('6.674000000000001e-11*x0*x1/x2**2', 85907), ('x0*exp(-x1/x2)', 34000), ('x0*(1 + x1)**2', 25585),
    ('x0**2 + 2*x0*x1 + x1**2', 43755), ('(x0 + x1)**2', 20585), ('x0**-1', 12000), ('1.0/x0', 9000),
    ('x0*inv(x1)', 18000), ('-(1/x0)', 10000), ('x0 - (-3)', 14000), ('x0/0', 14000),
]


@pytest.mark.parametrize('expression, price', _BRIEF + _MORE)
def test_the_written_price(engine: SimpliPyEngine, expression: str, price: int) -> None:
    assert engine.written_complexity(expression) == price


@pytest.mark.parametrize('expression, price', _BRIEF + _MORE)
def test_prefix_tokens_are_priced_as_their_infix(engine: SimpliPyEngine, expression: str, price: int) -> None:
    tokens = engine.read_infix(expression)
    assert engine.written_complexity(tokens) == price
    assert engine.written_complexity(tuple(tokens)) == price
    assert engine.written_complexity(engine.to_tagged(tokens)) == price


def test_no_mode_and_no_certificate() -> None:
    assert list(inspect.signature(SimpliPyEngine.written_complexity).parameters) == ['self', 'expression']


class TestTheIdentificationsInfixForces:
    def test_runs_whatever_the_bracketing_operands_unordered(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('x0 + (x1 + x2)') == w('(x2 + x1) + x0') == w('x0 + x1 + x2')
        assert w('x0*(x1*x2)') == w('(x2*x1)*x0') == w('x0*x1*x2')
        assert w('x0 + (x1 - x2)') == w('x0 + x1 - x2')
        assert w('x0/x1/x2') == w('x0*(1/x1)*(1/x2)') == w('x0*inv(x1)*inv(x2)')

    def test_a_bracketed_subtrahend_or_divisor_stays_whole(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('x0 - (x1 + x2)') == w('x0 - x1 - x2') + 2000  # its own sum head, one sign fewer
        assert w('x0/(x1*x2)') == w('x0/x1/x2')  # its own product head, one reciprocal fewer
        assert w('1/(x1*x2)') == w('1/x1/x2') - 3000

    def test_signs(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('-x0') == w('neg(x0)') == w('x0') + 1000
        assert w('-3') == w('3') + 1000
        assert w(['neg', '3']) == w('-3')
        assert w('x0 - 3') == w('x0 + -3')

    def test_a_reciprocal_its_one_the_notations(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('1/x0') == w('inv(x0)') == w('x0') + 3000
        assert w('1/3') == w('inv(3)') == engine.complexity('1/3')
        assert w('1/x0**2') == engine.complexity('1/x0**2')  # a written power divides as y^-k

    def test_numbers_every_digit_on_the_exact_value(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('0.1234567890123456789012345*x0') == 94730
        assert w('0.12345678901234568*x0') == 67625  # its double's shortest spelling
        assert w('6.674000000000001e-11') > w('6.674e-11')
        assert w('2.5') == w('2.50')  # the literal code is the value's: no digit is free or dear
        # a fraction costs its division, a decimal its value (owner, 2026-11-28; see TestTheDecisions)
        assert w('5/2') == w('2.5') + 5000

    def test_named_constants_and_symbols(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('np.pi') == w('pi') == w('np.e') == w('e') == 4000
        assert w('<constant>') == 67000
        assert w('<constant_a>*x0') == 76000
        assert w('x0') == w('v7') == 6000
        assert w(['float("inf")']) == w(['float("-inf")']) == w(['float("nan")']) == 8000


class TestTheVocabulary:
    @pytest.mark.parametrize('alias, target', [('Abs(x0)', 'abs(x0)'), ('ln(x0)', 'log(x0)'),
                                               ('arcsin(x0)', 'asin(x0)'), ('arccos(x0)', 'acos(x0)'),
                                               ('div(x0, x1)', 'x0/x1'), ('power(x0, 3)', 'x0**3'),
                                               ('pow(x0, 3)', 'x0**3')])
    def test_an_alias_is_priced_as_its_target(self, engine: SimpliPyEngine, alias: str, target: str) -> None:
        assert engine.written_complexity(alias) == engine.written_complexity(target)

    @pytest.mark.parametrize('head, bits', [('sqrt', 6000), ('cot', 6000), ('acot', 8000), ('sec', 6000),
                                            ('csc', 6000), ('asec', 8000), ('acsc', 8000)])
    def test_a_defined_function_pays_its_own_head(self, engine: SimpliPyEngine, head: str, bits: int) -> None:
        assert engine.written_complexity(f'{head}(x0)') == bits + 6000
        assert engine.written_complexity([head, 'x0']) == bits + 6000

    @pytest.mark.parametrize('tokens', [['ln', 'x0'], ['div', 'x0', 'x1'], ['**', 'x0', '2'],
                                        ['Max', 'x0', 'x1'], ['+', 'x0', 'Abs', 'x1']])
    def test_a_prefix_alias_token_is_refused_as_complexity_refuses_it(
            self, engine: SimpliPyEngine, tokens: list) -> None:
        """W-5 (pre-audit review): written_complexity priced prefix alias tokens that every other
        prefix entry refuses; a prefix token list names the operators themselves."""
        with pytest.raises(ValueError):
            engine.complexity(tokens)
        with pytest.raises(ValueError, match='is an alias of'):
            engine.written_complexity(tokens)

    def test_any_other_undeclared_head_is_refused(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match=r"unknown function 'zeta'"):
            engine.written_complexity('zeta(x0)')
        with pytest.raises(ValueError, match='undeclared function head is refused'):
            engine.written_complexity(['zeta', 'x0'])
        with pytest.raises(ValueError, match='empty'):
            engine.written_complexity([])


class TestTheDecisions:
    """The overseer's decisions (2026-11-27): where infix notation leaves a choice, the reading is
    the one complexity()'s code book prices. Each row: the expression, its written price, and the
    price it had before the decision (milli-bits)."""

    @pytest.mark.parametrize('expression, price, before', [
        # M1: `-1/y` is the SIGN of the reciprocal, -(1/y), not (-1)/y: a numerator -1 is a sign
        ('-1/x0', 10000, 15000), ('(-1)/x0', 10000, 15000), ('-1.0/x0', 10000, 15000),
        ('x0*(-1/x1)', 19000, 21000), ('-1/x0/x1', 22000, 24000), ('-1/3', 5000, 10000),
        # S1: a written `-inf`, and `inf` as a subtrahend, is the symbol float("-inf")
        ('-inf', 8000, 9000), ('x0 - inf', 17000, 18000), ('x0 + -inf', 17000, 18000),
        ('2*(-inf)', 13585, 14585),
        # K: a divisor y^k is y^-k only for a literal k, the sign merging into it; otherwise
        # the reciprocal of the power
        ('x0/x1**x2', 27000, 25000), ('x0/exp(x1)**x2', 33000, 31000), ('x0/x1**(-2)', 20585, 22585),
        ('x0/x1**2', 21585, 21585), ('1/x0**2', 12585, 12585),
        # L: max/min written with more than two arguments is ONE call
        ('max(x0, x1, x2)', 26000, 34000), ('min(x0, x1, x2, x3)', 32000, 48000),
        ('max(x0, max(x1, x2))', 26000, 34000),  # the same tokens as max(x0, x1, x2)
        ('max(max(x0, x1), x2)', 34000, 34000),  # two calls written
    ])
    def test_m1_s1_k_l(self, engine: SimpliPyEngine, expression: str, price: int, before: int) -> None:
        assert engine.written_complexity(expression) == price
        assert engine.written_complexity(engine.read_infix(expression)) == price

    def test_the_spellings_of_minus_one_and_of_the_negative_infinity(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w(['/', 'neg', '1', 'x0']) == w(['/', '-1', 'x0']) == w('-(1/x0)') == 10000
        assert w(['neg', 'float("inf")']) == w(['float("-inf")']) == 8000
        assert w(['-', 'x0', 'float("-inf")']) == 18000  # x0 - (-inf): a sign on the symbol
        assert w('-1*x0') == 12000  # a factor -1 is a literal: M1 reads only a numerator

    # The owner's ruling (2026-11-28): "As written, fractions should cost their division, and
    # numbers their value." A quotient of numbers `p/q` is `p` and the literal `1/q`, as any number
    # divided by a number; `1/q` and `-1/q` stay the literal 1/q and -1/q (the reciprocal's 1 is
    # the notation's), and a decimal costs its value. (expression, written price in milli-bits)
    @pytest.mark.parametrize('expression, price', [
        ('3/4', 10322), ('-3/4', 11322), ('3/-4', 11322), ('5/2', 10170), ('6/4', 11129),
        ('3.0/4', 10322), ('x0**(3/2)', 18585), ('x0**(-3/2)', 19585), ('x0/x1**(3/2)', 30585),
        ('1/x0**(3/2)', 21585), ('x0/(3/4)', 22322), ('3/4*x0', 16322), ('x0*(3/4)', 16322),
        ('1/4', 4322), ('1/2', 3585), ('-1/4', 5322), ('4*x0/3', 16322), ('x0*3/4', 16322),
        ('2*x0*3', 14585), ('2.5/2', 11755), ('0.75', 5322),
    ])
    def test_a_fraction_costs_its_division(self, engine: SimpliPyEngine, expression: str, price: int) -> None:
        assert engine.written_complexity(expression) == price
        assert engine.written_complexity(engine.read_infix(expression)) == price

    def test_a_fraction_costs_what_its_product_costs_written_either_way(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        assert w('3/4*x0') == w('4*x0/3') == w('x0*3/4')
        assert w('3/4') == w('0.75') + 5000  # the division, against the value
        assert w('x0**(3/2)') == w('x0**1.5') + 5000

    # One writing, one price, in every spelling (pre-audit review W-1..W-4): (prefix tokens, the
    # price, the spelling it equals)
    @pytest.mark.parametrize('tokens, price, other', [
        # W-1: a one-token fraction costs its division (it cost its value, 5.322), also in the
        # exponent merge
        (['3/4'], 10322, ['/', '3', '4']), (['(-3/4)'], 11322, ['/', '-3', '4']),
        (['/', 'x0', 'pow', 'x1', '3/2'], 30585, 'x0/x1**(3/2)'),
        (['1/4'], 4322, '1/4'), (['-1/4'], 5322, '-1/4'),
        # W-2: neg(c) is one number as a divisor, a merged exponent and a 1/q part
        (['/', 'x0', 'neg', '3'], 14000, ['/', 'x0', '-3']),
        (['/', 'x0', 'pow', 'x1', 'neg', '2'], 20585, ['/', 'x0', 'pow', 'x1', '-2']),
        (['/', '1', 'neg', '3'], 5000, ['/', '1', '-3']),
        # W-3: -1/-3 writes two minus signs (it was priced as 1/3)
        (['/', '-1', '-3'], 6000, '-1/-3'),
        # W-4: x/y**0 writes no sign
        (['/', 'x0', 'pow', 'x1', '0'], 20000, ['*', 'x0', 'pow', 'x1', '0']),
    ])
    def test_one_writing_one_price(self, engine: SimpliPyEngine, tokens: list, price: int,
                                   other: 'list | str') -> None:
        assert engine.written_complexity(tokens) == price
        assert engine.written_complexity(other) == price


class TestTheProperty:
    """On a canonical form written as its state is held (one run per bag, one divisor per
    reciprocal, every sign where the state holds it) the written price equals complexity() in
    that mode. It is a measured agreement, not a promise: the canonical serializations write some
    states otherwise (below, and the agreement rate in the docstring)."""

    @pytest.mark.parametrize('expression', [
        'x0 + x1 + x2', 'x0 - x1', 'x0/x1', '1/x0', 'x0/3', 'x0**2', '2*x0', '6.674e-11*x0*x1/x2**2',
        'x0*exp(-x1/x2)', '(x0 + x1)**2', 'x0**(1/3)', '1/x0**(1/2)', 'x1/0.225147640565301',
        'x0**2 + x1**2 + 2*x0*x1', 'x0*(x1 + 1)**2', 'x0*x1/4/np.pi/x2/x3**2', 'x0/2/x1',
        'x0*abs(1/x1)', 'rootn(x0, 2)', 'x0 + x2 - x1', 'log(abs(x0))', 'atan(1/x0)', '1/tan(x0)',
        # the decisions M1, S1, K, L
        '-1/x0', '-1/x0/x1', '-1/3', '-x0/x1', '-inf', 'x0/x1**x2', 'x0/exp(x1)**x2', 'max(x0, x1, x2)',
        'min(x0, x1, x2, x3)',
    ])
    @pytest.mark.parametrize('mode', ['strict', 'permissive'])
    def test_written_equals_canonical(self, engine: SimpliPyEngine, expression: str, mode: str) -> None:
        canonical = engine.simplify(expression, mode=mode)
        assert engine.written_complexity(canonical) == engine.complexity(canonical, mode=mode)

    # A canonical fraction is printed `p/q` and held as the one number p/q: written, it costs its
    # division; canonical, its value (owner, 2026-11-28). (expression, written price of its
    # canonical print, complexity) in milli-bits.
    @pytest.mark.parametrize('expression, written, canonical', [
        ('3/4', 10322, 5322), ('-3/4', 11322, 6322), ('x0**(3/2)', 18585, 13585),
    ])
    @pytest.mark.parametrize('mode', ['strict', 'permissive'])
    def test_a_canonical_fraction_costs_its_division_written_and_its_value_canonical(
            self, engine: SimpliPyEngine, expression: str, written: int, canonical: int, mode: str) -> None:
        printed = engine.simplify(expression, mode=mode)
        assert engine.written_complexity(printed) == written
        assert engine.complexity(printed, mode=mode) == canonical

    # The canonical SERIALIZATIONS part from this reading where they write a state other than as
    # it is held; measured over the census and pinned here. (expression, written price of its
    # canonical print, complexity) in milli-bits.
    @pytest.mark.parametrize('expression, written, canonical', [
        ('4*x0/3', 16322, 14322),  # a coefficient p/q is printed p*x/q: numbers written apart
        ('-x0 - x1', 17000, 16000),  # N2: the printers distribute -(x0 + x1)
    ])
    def test_where_the_print_and_the_reading_part(self, engine: SimpliPyEngine, expression: str,
                                                  written: int, canonical: int) -> None:
        printed = engine.simplify(expression)
        assert engine.written_complexity(printed) == written
        assert engine.complexity(printed) == canonical

    def test_the_prefix_serializer_groups_reciprocals_and_moves_a_sign(self, engine: SimpliPyEngine) -> None:
        # G: x0^-1 * x1^-1 is serialized `inv * x0 x1`, the reciprocal of a written product
        tokens = engine.simplify(['/', '1', '*', 'x0', 'x1'])
        assert tokens == ['inv', '*', 'x0', 'x1']
        assert engine.written_complexity(tokens) == 18000
        assert engine.complexity(tokens) == 21000
        # N1: -(x0*x1)/x2 is serialized `/ neg * x0 x1 x2`, a sign on a product inside a product
        tokens = engine.simplify(['/', 'neg', '*', 'x0', 'x1', 'x2'])
        assert tokens == ['/', 'neg', '*', 'x0', 'x1', 'x2']
        assert engine.written_complexity(tokens) == 28000
        assert engine.complexity(tokens) == 25000
        assert engine.written_complexity(engine.simplify('-(x0*x1)/x2')) == 25000  # its infix print agrees
