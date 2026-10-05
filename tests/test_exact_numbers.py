"""Exact numbers beyond 128 bits (number plan phase 2c).

A literal is an exact rational whose numerator and denominator have at most 1,100 bits each;
before, a number left the exact form at 128 bits and stayed an opaque symbol. Real mode folds
every number within that cap. f64 mode folds a number only when the deployed evaluator reads
its spelling back as the same number: zero or normal, numerator at most DBL_MAX, denominator at
most 2^1022. In f64 mode an integer exponent beyond 2^53 has unknown parity, and a literal the
evaluator reads as inf is not certainly finite, as the evaluator sees them. A bag of literals
folds as far as its products (sums) stay foldable, to a fixed point that re-reads to itself.
"""
from fractions import Fraction

import numpy as np
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path

A, B = 7 ** 190, 3 ** 280      # A/B is admissible, (A/B)^2 has a numerator beyond DBL_MAX
MODES = ('f64', 'real')


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


def simplify(engine: SimpliPyEngine, prefix: str, mode: str) -> list[str]:
    return list(engine.simplify(prefix.split(), mode=mode))


def constant(engine: SimpliPyEngine, tokens: list[str]) -> float:
    """The compiled value of a variable-free expression."""
    return float(np.asarray(engine.as_callable(tokens, ['x0'])(np.array([1.0]))).ravel()[0])


def numbers(tokens: list[str]) -> list[Fraction]:
    """The exact values of the numeric tokens, sorted."""
    out = []
    for t in tokens:
        try:
            out.append(Fraction(t))
        except ValueError:
            pass
    return sorted(out)


class TestPlanConsequences:
    @pytest.mark.parametrize('mode', MODES)
    def test_a_small_literal_times_a_tiny_one_folds(self, engine: SimpliPyEngine, mode: str) -> None:
        out = simplify(engine, '* x1 * 1e-40 3', mode)
        assert len(out) == 3 and 'x1' in out
        assert numbers(out) == [Fraction(3, 10 ** 40)]

    @pytest.mark.parametrize('mode', MODES)
    def test_two_spellings_of_one_coefficient_cancel(self, engine: SimpliPyEngine, mode: str) -> None:
        assert simplify(engine, '- * 3e-40 x1 * x1 * 1e-40 3', mode) == ['0']

    @pytest.mark.parametrize('mode', MODES)
    def test_the_larmor_denominator_is_one_number(self, engine: SimpliPyEngine, mode: str) -> None:
        out = simplify(
            engine, '/ * pow x1 2 pow x2 2 * 6 * 3.1415926535897 * 8.854e-12 pow 2.99792458e8 3', mode)
        exact = 6 * Fraction('3.1415926535897') * Fraction('8.854e-12') * Fraction('2.99792458e8') ** 3
        assert numbers(out) == [Fraction(2), Fraction(2), exact]

    def test_a_sum_beyond_float64_folds_in_real_mode_only(self, engine: SimpliPyEngine) -> None:
        assert numbers(simplify(engine, '+ x1 + 1e308 1e308', 'f64')) == [10 ** 308, 10 ** 308]
        assert numbers(simplify(engine, '+ x1 + 1e308 1e308', 'real')) == [2 * 10 ** 308]

    @pytest.mark.parametrize('mode', MODES)
    def test_a_product_beyond_the_cap_keeps_its_members(self, engine: SimpliPyEngine, mode: str) -> None:
        assert numbers(simplify(engine, '* x1 * 1e200 1e200', mode)) == [10 ** 200, 10 ** 200]


class TestAdmissibleInF64Mode:
    @pytest.mark.parametrize('mode', MODES)
    def test_a_literal_beyond_the_cap_stays_as_written(self, engine: SimpliPyEngine, mode: str) -> None:
        assert simplify(engine, '* x1 1e400', mode) == ['*', '1e400', 'x1']

    def test_a_subnormal_is_a_number_in_real_mode_only(self, engine: SimpliPyEngine) -> None:
        assert simplify(engine, '* x1 5e-324', 'f64') == ['*', '5e-324', 'x1']
        assert numbers(simplify(engine, '* x1 5e-324', 'real')) == [Fraction(5, 10 ** 324)]

    def test_a_numerator_beyond_dbl_max_does_not_fold_in_f64_mode(self, engine: SimpliPyEngine) -> None:
        prefix = f'* x1 * / {A} {B} / {A} {B}'
        assert simplify(engine, prefix, 'f64') == ['*', '/', str(A), str(B), '*', '/', str(A), str(B), 'x1']
        assert simplify(engine, prefix, 'real') == ['/', '*', str(A ** 2), 'x1', str(B ** 2)]


class TestF64SeesWhatTheEvaluatorSees:
    def test_a_literal_read_as_inf_is_not_certainly_finite(self, engine: SimpliPyEngine) -> None:
        for prefix in ('* 0 1e400', '- exp 1e400 exp 1e400'):
            assert simplify(engine, prefix, 'f64') == prefix.split()
            assert simplify(engine, prefix, 'real') == ['0']
        assert np.isnan(constant(engine, ['*', '0', '1e400']))

    def test_parity_beyond_2_53_is_unknown(self, engine: SimpliPyEngine) -> None:
        odd = 'pow -1 9007199254740993'
        assert simplify(engine, odd, 'f64') == odd.split()
        assert simplify(engine, odd, 'real') == ['-1']
        # the evaluator reads the exponent as the even float 2^53
        assert constant(engine, odd.split()) == 1.0
        for mode in MODES:
            assert simplify(engine, 'pow -1 9007199254740992', mode) == ['1']


class TestCap:
    @pytest.mark.parametrize('mode', MODES)
    def test_a_power_within_the_cap_folds(self, engine: SimpliPyEngine, mode: str) -> None:
        assert numbers(simplify(engine, '* pow 2 1000 x1', mode)) == [2 ** 1000]

    @pytest.mark.parametrize('mode', MODES)
    def test_a_power_beyond_the_cap_stays_a_power(self, engine: SimpliPyEngine, mode: str) -> None:
        assert simplify(engine, '* pow 2 1100 x1', mode) == ['*', 'pow', '2', '1100', 'x1']


IDEMPOTENCE_ROWS = [
    # the phase-1 review's three rows (0.14.7 too): the first pass kept pieces the second merged
    '/ 10 / / 250000000 1571 / * 3.141592653589793 3.141592653589793 x3',
    '/ 12.5 / / 808000000 1571 / * 3.141592653589793 3.141592653589793 x2',
    '/ / 12625000 11 / 299792458 / * 3.141592653589793 3.141592653589793 x1',
    # the design review's H2 row, and a bag whose prefix folds where the whole bag does not
    f'* x1 * / {A} {B} / {A} {B}',
    '* x1 * * 1e300 1e10 1e-100',
    '* * 1e200 1e100 * 1e100 x1',
    '* 1e200 * 1e100 * 1e100 x1',
]


class TestFixedPoint:
    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('prefix', IDEMPOTENCE_ROWS)
    def test_simplify_is_idempotent(self, engine: SimpliPyEngine, prefix: str, mode: str) -> None:
        once = simplify(engine, prefix, mode)
        assert list(engine.simplify(once, mode=mode)) == once

    @pytest.mark.parametrize('prefix', IDEMPOTENCE_ROWS[:3])
    def test_the_phase1_rows_keep_their_values(self, engine: SimpliPyEngine, prefix: str) -> None:
        names = [t for t in prefix.split() if t.startswith('x')]
        x = np.linspace(0.5, 2.0, 9)
        before = engine.as_callable(prefix.split(), names)(x)
        after = engine.as_callable(simplify(engine, prefix, 'f64'), names)(x)
        np.testing.assert_allclose(after, before, rtol=4e-16)

    @pytest.mark.parametrize('mode', MODES)
    def test_a_bag_folds_as_far_as_it_can(self, engine: SimpliPyEngine, mode: str) -> None:
        # sorted ascending, 1e-100 * 1e10 * 1e300 folds in every prefix
        assert numbers(simplify(engine, '* x1 * * 1e300 1e10 1e-100', mode)) == [10 ** 210]

    @pytest.mark.parametrize('mode', MODES)
    def test_the_grouping_can_decide_a_refused_bag(self, engine: SimpliPyEngine, mode: str) -> None:
        # Known limit (design M2): at a refusal the input's grouping decides which inner
        # products fold, so these equal inputs keep different canonical forms.
        assert numbers(simplify(engine, '* * 1e200 1e100 * 1e100 x1', mode)) == [10 ** 100, 10 ** 300]
        assert numbers(simplify(engine, '* 1e200 * 1e100 * 1e100 x1', mode)) == [10 ** 200, 10 ** 200]


class TestLeftovers:
    @pytest.mark.parametrize('mode', MODES)
    def test_leading_zeros_are_normalised(self, engine: SimpliPyEngine, mode: str) -> None:
        assert simplify(engine, '* 007 x1', mode) == ['*', '7', 'x1']

    @pytest.mark.parametrize('mode', MODES)
    def test_a_signed_zero_fraction_is_zero(self, engine: SimpliPyEngine, mode: str) -> None:
        assert simplify(engine, '/ -0 5', mode) == ['0']
        assert simplify(engine, '* x1 / -0 5', mode) == ['0']

    @pytest.mark.parametrize('mode', MODES)
    def test_association_invariance(self, engine: SimpliPyEngine, mode: str) -> None:
        assert simplify(engine, '* / / x0 7 3 21', mode) == simplify(engine, '/ / * x0 21 7 3', mode) == ['x0']


class TestPythonReaders:
    """The verification instruments read literals within the engine's limits."""

    def test_the_monitor_folds_only_within_the_cap(self) -> None:
        from simplipy.verify._monitor import ENGINE_CAP_BITS, _FoldRefused, _fold_budget
        assert _fold_budget(Fraction(2) ** ENGINE_CAP_BITS - 1) == Fraction(2) ** ENGINE_CAP_BITS - 1
        with pytest.raises(_FoldRefused):
            _fold_budget(Fraction(2) ** ENGINE_CAP_BITS)

    def test_the_monitor_refuses_oversized_spellings_at_once(self) -> None:
        import time
        from simplipy.verify._monitor import _rat_leaf
        t0 = time.time()
        assert _rat_leaf('1e999999999') is None
        assert _rat_leaf('7' * 5000) is None
        assert _rat_leaf('1/' + '3' * 5000) is None
        assert time.time() - t0 < 1.0
        assert _rat_leaf('+1/3') == Fraction(1, 3)
        assert _rat_leaf('1e000040') == 10 ** 40

    def test_the_contract_reads_a_zero_padded_exponent(self) -> None:
        from simplipy.verify._contract import literal_value
        assert literal_value('1e' + '0' * 4400 + '1') == 10
