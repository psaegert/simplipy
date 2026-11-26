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
        assert w('5/2') == w('2.5') + 5000  # F: a quotient of literals is a product of 5 and 1/2

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

    @pytest.mark.parametrize('head, bits', [('sqrt', 6000), ('cot', 6000), ('acot', 8000)])
    def test_a_defined_function_pays_its_own_head(self, engine: SimpliPyEngine, head: str, bits: int) -> None:
        assert engine.written_complexity(f'{head}(x0)') == bits + 6000
        assert engine.written_complexity([head, 'x0']) == bits + 6000

    def test_any_other_undeclared_head_is_refused(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match=r"unknown function 'sec'"):
            engine.written_complexity('sec(x0)')
        with pytest.raises(ValueError, match='undeclared function head is refused'):
            engine.written_complexity(['sec', 'x0'])
        with pytest.raises(ValueError, match='empty'):
            engine.written_complexity([])


class TestTheProperty:
    """On an expression already in canonical form the written price equals complexity() in that
    mode. These canonical forms write the canonical state as it is held (one run per bag, one
    divisor per reciprocal, every sign where the state holds it)."""

    @pytest.mark.parametrize('expression', [
        'x0 + x1 + x2', 'x0 - x1', 'x0/x1', '1/x0', 'x0/3', 'x0**2', '2*x0', '6.674e-11*x0*x1/x2**2',
        'x0*exp(-x1/x2)', '(x0 + x1)**2', 'x0**(1/3)', '1/x0**(1/2)', 'x1/0.225147640565301',
        'x0**2 + x1**2 + 2*x0*x1', 'x0*(x1 + 1)**2', 'x0*x1/4/np.pi/x2/x3**2', 'x0/2/x1',
        'x0*abs(1/x1)', 'rootn(x0, 2)', 'x0 + x2 - x1', 'log(abs(x0))', 'atan(1/x0)', '1/tan(x0)',
    ])
    @pytest.mark.parametrize('mode', ['strict', 'permissive'])
    def test_written_equals_canonical(self, engine: SimpliPyEngine, expression: str, mode: str) -> None:
        canonical = engine.simplify(expression, mode=mode)
        assert engine.written_complexity(canonical) == engine.complexity(canonical, mode=mode)

    # The canonical SERIALIZATIONS part from this reading where they write a state other than
    # as it is held, and where the brief's identifications do not cover the notation; measured
    # over the census (design notes, 2026-11-26) and pinned here, so a ruling on any of them
    # changes these tests visibly. (written, its canonical print, written, canonical) in bits.
    @pytest.mark.parametrize('expression, written, canonical', [
        ('3/4', 10322, 5322),             # F: a fraction p/q (|p| != 1) is written as a quotient
        ('4*x0/3', 16322, 14322),         # F: ... its numerator and divisor apart in a product
        ('x0**(3/2)', 18585, 13585),      # F: ... and as an exponent
        ('-1/x0', 15000, 10000),          # M1: the printers write -(1/x0) as -1/x0, read (-1)/x0
        ('-x0 - x1', 17000, 16000),       # N2: the printers distribute -(x0 + x1)
        ('max(x0, x1, x2)', 34000, 26000),  # L: the reader nests a variadic max to the right
    ])
    def test_where_the_print_and_the_reading_part(self, engine: SimpliPyEngine, expression: str,
                                                  written: int, canonical: int) -> None:
        printed = engine.simplify(expression)
        assert engine.written_complexity(printed) == written
        assert engine.complexity(printed) == canonical

    def test_the_prefix_serializer_groups_reciprocals(self, engine: SimpliPyEngine) -> None:
        # G: x0^-1 * x1^-1 is serialized `inv * x0 x1`, the reciprocal of a written product
        tokens = engine.simplify(['/', '1', '*', 'x0', 'x1'])
        assert tokens == ['inv', '*', 'x0', 'x1']
        assert engine.written_complexity(tokens) == 18000
        assert engine.complexity(tokens) == 21000
