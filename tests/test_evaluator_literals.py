"""The compiled evaluator reads every literal as its nearest float64 (number plan phase 2b).

The compiled expression is Python source, and Python reads an integer literal as an exact
``int``: numpy's ufuncs reject one above int64 (``sin(100000000000000000000)`` raised
TypeError) and float arithmetic rejects one above float64's range (OverflowError). Such
integers are spelled as their nearest float64, and as ``inf`` beyond the range; integers up to
2^53 and fractions inside the range are unchanged.
"""
import numpy as np
import pytest

from simplipy import SimpliPyEngine
from simplipy.utils import evaluator_literal
from conftest import acj_config_path


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


class TestEvaluatorLiteral:
    @pytest.mark.parametrize('token, spelled', [
        ('5', '5'),
        ('-7', '-7'),
        (str(2 ** 53), str(2 ** 53)),
        (str(2 ** 53 + 1), repr(float(2 ** 53 + 1))),
        ('100000000000000000000', '1e+20'),
        ('(-100000000000000000000)', '-1e+20'),
        ('9' * 400, 'float("inf")'),
        ('-' + '9' * 400, 'float("-inf")'),
        ('1/3', '1/3'),
        ('673107593011939307760027002528/810572757194796821120128085049',
         '673107593011939307760027002528/810572757194796821120128085049'),
        ('1' + '0' * 400 + '/3', 'float("inf")'),
        ('0.5', '0.5'),
        ('1e400', '1e400'),
        ('007', '7'),
        ('+5', '5'),
        ('-0', '0'),
        ('01/3', '1/3'),
        ('9' * 5000, 'float("inf")'),              # beyond Python's integer-string limit
        ('0' * 5000 + '5', '5'),
        ('x0', 'x0'),
        ('np.pi', 'np.pi'),
        ('*', '*'),
    ])
    def test_spelling(self, token: str, spelled: str) -> None:
        assert evaluator_literal(token) == spelled


class TestCompiledLiterals:
    def test_a_huge_integer_inside_a_function_evaluates(self, engine: SimpliPyEngine) -> None:
        f = engine.as_callable(['*', 'x0', 'sin', '100000000000000000000'], ['x0'])
        assert f(np.array([2.0]))[0] == 2.0 * np.sin(1e20)

    def test_an_integer_beyond_float64_reads_as_inf(self, engine: SimpliPyEngine) -> None:
        f = engine.as_callable(['*', 'x0', '1' + '0' * 309], ['x0'])
        assert np.isinf(f(np.array([2.0]))[0])

    def test_small_literals_keep_their_values(self, engine: SimpliPyEngine) -> None:
        f = engine.as_callable(['+', 'pow', 'x0', '2', '/', '1', '3'], ['x0'])
        assert f(np.array([3.0]))[0] == 9.0 + 1 / 3


class TestOracleLiterals:
    """The mining oracles read a literal as the compiled expression does."""

    @pytest.mark.parametrize('token, value', [
        ('1/3', 1 / 3),
        ('-2.5', -2.5),
        ('1e400', float('inf')),
        ('1' + '0' * 400 + '/3', float('inf')),
        ('-' + '1' + '0' * 400 + '/3', float('-inf')),
        ('673107593011939307760027002528/810572757194796821120128085049', 0.8304098392615706),
    ])
    def test_literal_float(self, token: str, value: float) -> None:
        from simplipy.utils import literal_float
        assert literal_float(token) == value

    def test_an_integer_zero_is_positive(self) -> None:
        import math
        from simplipy.utils import literal_float
        assert math.copysign(1.0, literal_float('-0')) == 1.0      # `-0` is the int 0
        assert math.copysign(1.0, literal_float('-0.0')) == -1.0   # `-0.0` is a float

    def test_non_numerals_are_not_literals(self) -> None:
        from simplipy.utils import literal_float
        assert literal_float('x0') is None
        assert literal_float('np.pi') is None

    def test_the_f64_oracle_reads_a_fraction(self) -> None:
        from simplipy.promotion._f64_eval import _num
        assert _num('1/3') == 1 / 3
        assert _num('(-1/3)') == -1 / 3

    def test_the_high_precision_oracle_reads_a_fraction(self) -> None:
        from simplipy.promotion._hp_equiv import evaluate
        assert float(evaluate(['/', '1/3', '2'], {}, [])) == 1 / 6
        assert float(evaluate(['(-1/3)'], {}, [])) == -1 / 3   # the literal reads as its nearest double

    def test_the_contract_refuses_an_oversized_spelling_at_once(self) -> None:
        import time
        from simplipy.verify._contract import UnsupportedToken, literal_value
        t0 = time.time()
        with pytest.raises(UnsupportedToken):
            literal_value('1e999999999')
        with pytest.raises(UnsupportedToken):
            literal_value('7' * 5000)
        with pytest.raises(UnsupportedToken):
            literal_value('1e' + '9' * 5000)
        with pytest.raises(UnsupportedToken):
            literal_value('1' * 9000 + '/x')
        assert time.time() - t0 < 1.0
        assert literal_value('1e40') == 10 ** 40

    def test_the_deployed_lane_reads_a_literal_beyond_range_as_inf(self) -> None:
        from simplipy.verify._contract import judge_rule
        assert judge_rule(['*', 'x0', '1e400'], ['*', '1e400', 'x0'])['realised'] is True


Q = '1' + '0' * 400


class TestKernelLiterals:
    """Literals beyond 128 bits that the interval kernel now reads exactly: the canonical
    forms that move because of it, each checked against the exact value."""

    @pytest.mark.parametrize('prefix, out', [
        (['rootn', '-1', str(2 ** 128)], ['float("nan")']),       # an even root of -1
        (['rootn', 'rootn', 'x1', '0', str(2 ** 127)], ['float("nan")']),
        (['log', '-1/' + Q], ['float("nan")']),                   # log of a negative number
        (['pow', '-2', f'{2 ** 256}/{2 ** 128}'], None),          # (-2)^(2^128) is finite
        (['pow', '-2', f'{3 * 2 ** 200}/3'], None),               # main folded it to nan
        (['log', f'{Q}/{Q[:-1]}'], None),                         # log(10), no nan
    ])
    def test_beyond_i128_literals_fold_only_when_certain(self, engine: SimpliPyEngine,
                                                         prefix: list[str], out: list[str] | None) -> None:
        for mode in ('f64', 'real'):
            assert list(engine.simplify(prefix, mode=mode)) == (prefix if out is None else out)

    @pytest.mark.parametrize('zero', ['+0/5', '(+0/5)', '+00/7', '-0/5'])
    def test_every_spelling_of_a_zero_fraction_is_zero(self, engine: SimpliPyEngine, zero: str) -> None:
        core = engine._core
        assert core.interval_class(['log', zero]) == core.interval_class(['log', '0'])
        assert core.interval_class(['-', 'log', zero, 'log', zero]) == core.interval_class(['-', 'log', '0', 'log', '0'])
