"""THE PRICING CONTRACT, pinned where the docs state it.

`complexity()` and `written_complexity()` state worked examples and invariants in their docstrings
and in `docs/guides/simplify.md` ("The written price"). This file reads the guide's table itself and
asserts every row in both prices, and pins each stated invariant through the public API, in both
modes where the statement is about every mode, so a doc that drifts from the code fails here.

The last class records the counterexamples to the descent bound
``complexity(simplify(e, mode=m), mode=m) <= complexity(e, mode=m)`` found by the pre-audit review
(in both modes), as strict xfails: each flips to a failure once the bound holds for it, so the
record cannot go stale.
"""

import os
import re
import warnings

import pytest

from simplipy import SimpliPyEngine

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODES = ['strict', 'permissive']
MILLI = 1000


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SimpliPyEngine.from_config(os.path.join(_ROOT, 'assets', 'engines', 'acj2-4', 'config.yaml'))


def _bits(cell: str) -> int:
    """A price cell of the guide's table (bits, up to three decimals) in milli-bits."""
    return round(float(cell) * MILLI)


def _guide_table() -> list[tuple[str, int, int]]:
    """The rows `| `expression` | written | complexity |` of the guide's section "The written
    price", read from the page itself."""
    with open(os.path.join(_ROOT, 'docs', 'guides', 'simplify.md')) as f:
        text = f.read()
    section = text.split('## The written price', 1)[1].split('\n## ', 1)[0]
    rows = re.findall(r'^\| `([^`]+)` \| ([0-9.]+) \| ([0-9.]+) \|$', section, flags=re.M)
    return [(expression, _bits(written), _bits(canonical)) for expression, written, canonical in rows]


_TABLE = _guide_table()


def test_the_guide_table_is_read_whole() -> None:
    # the parser above must not silently read nothing (or half the table)
    assert len(_TABLE) == 18
    assert _TABLE[0] == ('x0 + 0', 11000, 6000)
    assert _TABLE[-1] == ('x0*x1/(4*np.pi*x2*x3**2)', 45907, 47907)


@pytest.mark.parametrize('expression, written, canonical', _TABLE)
def test_every_row_of_the_guide_table_holds(engine: SimpliPyEngine, expression: str, written: int,
                                            canonical: int) -> None:
    assert engine.written_complexity(expression) == written
    assert engine.complexity(expression) == canonical


class TestTheWrittenDocstring:
    """The worked examples of `SimpliPyEngine.written_complexity`'s docstring, as written there:
    (expression, written price, complexity or None where the docstring gives none)."""

    @pytest.mark.parametrize('expression, written, canonical', [
        ('x0 + 0', 11000, 6000), ('x0 + x0', 15000, 11585), ('exp(1)', 8000, 4000),
        ('-(x0 - x1)', 17000, 16000), ('sqrt(x0)', 12000, None), ('x0**0.5', 12585, None),
        ('x0 - x0/1 + 2', 23585, 2585), ('6.674e-11*x0*x1/x2**2', 45197, 45197),
        ('x0*x1/(4*np.pi*x2*x3**2)', 45907, 47907),
        ('-1/x0', 10000, None), ('3/4', 10322, 5322), ('0.75', 5322, 5322), ('x0**(3/2)', 18585, 13585),
        ('3/4*x0', 16322, None), ('4*x0/3', 16322, 14322), ('2*x0*3', 14585, None), ('1/2', 3585, None),
        ('-x0 - x1', 17000, 16000),
    ])
    def test_the_examples(self, engine: SimpliPyEngine, expression: str, written: int,
                          canonical: int | None) -> None:
        assert engine.written_complexity(expression) == written
        if canonical is not None:
            assert engine.complexity(expression) == canonical

    def test_the_prefix_examples_of_the_agreement_paragraph(self, engine: SimpliPyEngine) -> None:
        assert engine.written_complexity(['inv', '*', 'x0', 'x1']) == 18000
        assert engine.complexity(['inv', '*', 'x0', 'x1']) == 21000
        assert engine.written_complexity(['/', 'neg', '*', 'x0', 'x1', 'x2']) == 28000
        assert engine.complexity(['/', 'neg', '*', 'x0', 'x1', 'x2']) == 25000

    def test_the_identifications(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        # `x/y**2` holds y^-2 and `x/y**-2` holds y^2: the division's sign merges into the number
        assert w('x0/x1**2') == engine.complexity('x0/x1**2')
        assert w('x0/x1**-2') == engine.complexity('x0*x1**2')
        # `x/3` holds the literal 1/3, as complexity prices it
        assert w('x0/3') == engine.complexity('x0/3') == 13000
        # a written `-inf`, and `inf` as a subtrahend, is the symbol float("-inf"), 8 bits
        assert w('-inf') == w(['float("-inf")']) == 8000
        assert w('x0 - inf') == w('x0 + -inf') == 17000
        # max/min with more than two arguments is one call; two nested calls written are two
        assert w('max(x0, x1, x2)') == 26000
        assert w('max(max(x0, x1), x2)') == 34000

    def test_every_digit_is_charged(self, engine: SimpliPyEngine) -> None:
        w = engine.written_complexity
        # `6.674000000000001e-11` costs its 16 digits, more than `6.674e-11`
        assert w('6.674000000000001e-11') == 58322
        assert w('6.674e-11') == 17612
        # a literal of any length, beyond the 1,100-bit cap too, is priced by its spelling
        long = '1.' + '3' * 3000
        assert w(long) == engine.complexity(long) > 9_000_000

    @pytest.mark.parametrize('token, price', [
        ('np.pi', 4000), ('np.e', 4000), ('float("inf")', 8000), ('float("-inf")', 8000),
        ('float("nan")', 8000), ('<constant>', 67000), ('<constant_g>', 67000), ('x0', 6000),
    ])
    def test_the_symbols_are_priced_as_complexity_prices_them(self, engine: SimpliPyEngine, token: str,
                                                              price: int) -> None:
        assert engine.written_complexity([token]) == price
        for mode in _MODES:
            assert engine.complexity([token], mode=mode) == price

    @pytest.mark.parametrize('token', ['inf', 'nan', 'Infinity', '1e', '1/0'])
    def test_a_reserved_numeric_spelling_is_refused(self, engine: SimpliPyEngine, token: str) -> None:
        with pytest.raises(ValueError, match='reserved numeric spelling'):
            engine.written_complexity([token])

    def test_the_token_cap(self, engine: SimpliPyEngine) -> None:
        # 4,096 tokens are read (a chain 4,096 deep included); one more is refused, never recursed into
        assert engine.written_complexity(['neg'] * 4095 + ['x0']) == 4095 * MILLI + 6000
        with pytest.raises(ValueError, match='too long'):
            engine.written_complexity(['neg'] * 4096 + ['x0'])


class TestTheComplexityDocstring:
    """The invariants `SimpliPyEngine.complexity`'s docstring states, through the public API."""

    @pytest.mark.parametrize('mode', _MODES)
    @pytest.mark.parametrize('signed, unsigned', [
        ('-x1', 'x1'),                  # a negation standing alone: one bit, no product head
        ('x1 - x2', 'x1 + x2'),         # the sign of a subtrahend
        ('-x1*x2', 'x1*x2'),            # the coefficient -1 beside other factors
        ('-2', '2'),                    # a literal's sign
        ('x0*(-3)', 'x0*3'),            # a coefficient's sign
        ('x0**-2', 'x0**2'),            # an exponent's sign
        ('x0 - 2.5', 'x0 + 2.5'),       # a sum's constant term
        ('-np.pi', 'np.pi'),
        ('-sin(x0)', 'sin(x0)'),
        ('x0/(-x1)', 'x0/x1'),
        ('-1e400', '1e400'),            # a literal beyond float64's range
    ])
    def test_a_sign_costs_one_bit_wherever_it_sits(self, engine: SimpliPyEngine, mode: str, signed: str,
                                                   unsigned: str) -> None:
        assert engine.complexity(signed, mode=mode) == engine.complexity(unsigned, mode=mode) + MILLI

    @pytest.mark.parametrize('mode', _MODES)
    def test_where_the_sign_sits_is_not_a_question_of_price(self, engine: SimpliPyEngine, mode: str) -> None:
        assert engine.complexity('-x1*tanh(3)', mode=mode) == engine.complexity('x1*tanh(-3)', mode=mode)

    @pytest.mark.parametrize('mode', _MODES)
    def test_a_coefficient_one_is_free(self, engine: SimpliPyEngine, mode: str) -> None:
        for spelling in ('1*x0', '1.0*x0', 'x0*1'):
            assert engine.complexity(spelling, mode=mode) == engine.complexity('x0', mode=mode) == 6000

    @pytest.mark.parametrize('mode', _MODES)
    def test_an_exponent_minus_one_is_division(self, engine: SimpliPyEngine, mode: str) -> None:
        c = lambda e: engine.complexity(e, mode=mode)  # noqa: E731
        assert c('x0**-1') == c('1/x0') == c('inv(x0)') == c('x0') + 3000  # the Pow head, no sign
        assert c('x1*x0**-1') == c('x1/x0')

    def test_the_readings_differ_by_mode(self, engine: SimpliPyEngine) -> None:
        # strict follows float64 (tanh saturates); permissive follows the reals
        assert engine.complexity('atanh(tanh(x0))') == 22000
        assert engine.complexity('atanh(tanh(x0))', mode='permissive') == 6000
        # past an overflow the true value is accepted in strict too
        assert engine.complexity('exp(x0)/exp(x0)') == 2000

    @pytest.mark.parametrize('mode', _MODES)
    def test_a_ground_constant_is_read_as_written(self, engine: SimpliPyEngine, mode: str) -> None:
        literal = '0.1000000000000000055511151231257827'
        assert engine.complexity(f'sin({literal})', mode=mode) == 6000 + engine.written_complexity(literal)

    @pytest.mark.parametrize('mode', _MODES)
    def test_the_reading_lands_a_literal_on_the_value_its_mode_admits(
            self, engine: SimpliPyEngine, mode: str) -> None:
        # A literal at the root, a coefficient, a sum's constant or a power's operand is read at the
        # value its mode admits, the price of its double's shortest spelling here: the code book
        # charges the 34 digits written (written_complexity) only to the spelling itself.
        literal = '0.1000000000000000055511151231257827'
        assert engine.complexity(literal, mode=mode) == engine.complexity('0.1', mode=mode) == 3000
        assert engine.complexity(f'x0*{literal}', mode=mode) == engine.complexity('x0*0.1', mode=mode)
        assert engine.written_complexity(literal) == 115753
        # the measure fingerprint's long-literal probe: float64's double of it, cheaper than its 64 digits
        probe = '0.0766541268471677307861497420516056347394916180597539647375669841'
        assert engine.complexity(probe, mode=mode) == engine.complexity(repr(float(probe)), mode=mode)

    def test_permissive_lands_on_the_cheaper_of_the_two_doubles_around_a_literal(
            self, engine: SimpliPyEngine) -> None:
        # 0.3000000000000000444 lies between the doubles 0.29999999999999998890 (`0.3`) and
        # 0.30000000000000004441 (`0.30000000000000004`, the nearest): strict reads float64's double,
        # permissive the one that prints shorter, which here is not the nearest
        literal = '0.3000000000000000444'
        assert engine.complexity(literal) == engine.complexity('0.30000000000000004')
        assert engine.complexity(literal, mode='permissive') == engine.complexity('0.3') == 4000
        assert engine.simplify(literal, mode='permissive') == '0.3'

    @pytest.mark.parametrize('mode', _MODES)
    def test_the_free_constant_dominates_every_f64_literal(self, engine: SimpliPyEngine, mode: str) -> None:
        # the supremum of the codeword minimum over float64 shortest spellings, at
        # 8.9002954340287245e-308, plus the sign and selector bits, is 66.649 bits; ceiled, 67
        assert engine.complexity('-8.9002954340287245e-308', mode=mode) == 66649
        assert engine.complexity('<constant>', mode=mode) == 67000
        # a literal beyond float64's range is no fittable value and may price below it
        assert engine.complexity('1e-400', mode=mode) == 10647


# Counterexamples to `complexity(simplify(e, mode=m), mode=m) <= complexity(e, mode=m)`, found by the
# pre-audit review against 3a6ed3a (input price -> answer price, milli-bits, permissive):
_DESCENT_COUNTEREXAMPLES = [
    # P-1: the reading takes the absorbing element (0*u = 0, 1^u = 1, u^0 = 1) over a ground
    # subtree it reads as written, while the walk folds that subtree to an undefined value first
    ('0*cot(np.pi)', 'nan'),                       # 2000 -> 8000
    ('pow(1, sqrt(log(6.674e-11)))', 'nan'),       # 2000 -> 8000
    ('pow(pow(-2, np.pi), 0)', 'nan'),             # 2000 -> 8000
    # P-2: the walk joins the reciprocals of a product holding floor/ceil into one reciprocal and
    # the joined product reads distributed (x0*(x0 - 1) as x0^2 - x0): a costlier state
    ('1/(x0*(x0 - 1)*floor(x2))', '1/(floor(x2)*(x0^2 - x0))'),            # 44000 -> 44585
    ('1.5/(x0*(x0 - 1)*floor(x2))', '1/(2*floor(x2)*(x0^2 - x0)/3)'),      # 48585 -> 49170
    ('1.5/(x0*(x0 - 1)*ceil(x2))', '1/(2*ceil(x2)*(x0^2 - x0)/3)'),        # 48585 -> 49170
]
# ... and in strict, the default mode (input price -> answer price):
_STRICT_DESCENT_COUNTEREXAMPLES = [
    # S-1: the reading takes the true value past an overflow (0*u = 0, a finite addend beside an
    # infinity), while the walk folds the overflowing ground part to float64's infinity first
    ('0*pow(3, 123456789012345678901234567890)', 'nan'),                      # 2000 -> 8000
    ('exp(123456789012345678901234567890)*0', 'nan'),                         # 2000 -> 8000
    ('pow(10, 123456789012345678901234567890) + x0/-0.0', '(-inf)*x0 + inf'),  # 17000 -> 28000
    # S-2: the reading drops `abs` over a power of a ground base it takes as nonnegative (rootn,
    # asin of a positive), while the walk folds that base to NaN first and keeps the abs
    ('abs(pow(pow(-2, 0.5), x1))', 'abs(nan^(x1))'),                          # 21170 -> 23000
    ('abs(pow(asin(1.5), x2))', 'abs(nan^(x2))'),                             # 21585 -> 23000
    # S-3: the print of the state re-reads one sign dearer (inside max, beside a sibling)
    ('max(x1, -(max(x1, x2)*tanh(3) + x2))', 'max(x1, max(x1, x2)*tanh(-3) - x2)'),  # 58000 -> 59000
    # S-4: the reading groups a shared number while a named constant is used twice; the walk drops
    # one use (C^0 = 1) and distributes the number, paying it twice
    ('(pow(<constant_g>, x2) + x0)*6.674e-11 + pow(<constant_g>, 0)',
     '6.674e-11*x0 + 6.674e-11*<constant_g>^(x2) + 1'),                       # 110612 -> 128224
    # S-5: the reading lands the reciprocal of an integer coefficient on a double, the walk keeps
    # the exact integer
    ('1/(pow(log(-1e400), x1)/123456789012345678901234567890)',
     '123456789012345678901234567890/nan^(x1)'),                              # 75242 -> 118318
]


class TestTheDescentBound:
    @pytest.mark.parametrize('mode', _MODES)
    @pytest.mark.parametrize('expression', [
        'x0 + 0', 'x0*x0', 'x0 - x0/1 + 2', '-(x0 - x1)', '3/4*x0', 'x0*x1/(4*np.pi*x2*x3**2)',
        '1.5/(x0*(x0 - 1)*x1)', '1.5/(x0*(x0 - 1)*abs(x2))', '0*log(-1)', 'pow(1, log(-1))',
        'x0*0.1000000000000000055511151231257827', 'x0*0.30000000000000004*0.30000000000000004',
        'sin(asin(x1/x0))/(x0 - 1)', '(x0**2)**(1/2)', 'atanh(tanh(x0))', '0*cot(np.pi)',
        'pow(1, sqrt(log(6.674e-11)))', '1/(x0*(x0 - 1)*floor(x2))',
    ])
    def test_the_bound_holds(self, engine: SimpliPyEngine, mode: str, expression: str) -> None:
        if mode == 'permissive' and any(expression == e for e, _ in _DESCENT_COUNTEREXAMPLES):
            pytest.skip('a recorded permissive counterexample (TestTheDescentBound.test_the_counterexamples)')
        answer = engine.simplify(expression, mode=mode)
        assert engine.complexity(answer, mode=mode) <= engine.complexity(expression, mode=mode)

    @pytest.mark.xfail(strict=True, reason='pre-audit P-1/P-2: permissive answers priced above their '
                                           'input; the owner decides re-mine or after 0.15')
    @pytest.mark.parametrize('expression, answer', _DESCENT_COUNTEREXAMPLES)
    def test_the_counterexamples(self, engine: SimpliPyEngine, expression: str, answer: str) -> None:
        # `answer` is the answer recorded at 3a6ed3a, for the reader; only the bound is asserted
        got = engine.simplify(expression, mode='permissive')
        assert engine.complexity(got, mode='permissive') <= engine.complexity(expression, mode='permissive')

    @pytest.mark.xfail(strict=True, reason='pre-audit S-1..S-5: strict answers priced above their '
                                           'input; the owner decides re-mine or after 0.15')
    @pytest.mark.parametrize('expression, answer', _STRICT_DESCENT_COUNTEREXAMPLES)
    def test_the_strict_counterexamples(self, engine: SimpliPyEngine, expression: str, answer: str) -> None:
        got = engine.simplify(expression)
        assert engine.complexity(got) <= engine.complexity(expression)
