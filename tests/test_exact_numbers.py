"""Exact numbers beyond 128 bits (number plan phase 2c).

A literal is an exact rational whose numerator and denominator have at most 1,100 bits each;
before, a number left the exact form at 128 bits and stayed an opaque symbol. Real mode folds
every number within that cap. f64 mode holds a number only when its numerator and denominator
are both at most 2^1022, so the number and its reciprocal are normal float64s; other literals
stay leaves as written. In f64 mode an integer beyond 2^53 has unknown parity, and a literal
the evaluator reads as inf (or as 0) is neither certainly finite nor certainly nonzero. A bag
of literals folds until no two of its members fold, which re-reads to itself in any order.
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
        # 1e308 is beyond 2^1022: an f64 leaf as written, collected like any symbol (as in main)
        assert simplify(engine, '+ x1 + 1e308 1e308', 'f64') == ['+', '*', '2', '1e308', 'x1']
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
        out = simplify(engine, '* x1 5e-324', 'real')
        assert out[0] == '*' and out[2] == 'x1' and out[1] != '5e-324', out   # a number, re-spelled
        assert Fraction(out[1]) == Fraction(5, 10 ** 324)

    def test_a_numerator_beyond_dbl_max_does_not_fold_in_f64_mode(self, engine: SimpliPyEngine) -> None:
        prefix = f'* x1 * / {A} {B} / {A} {B}'
        assert simplify(engine, prefix, 'f64') == ['*', '/', str(A), str(B), '*', '/', str(A), str(B), 'x1']
        # beyond 128 bits a fraction prints as ONE member (a local `/ p q`), never split
        assert simplify(engine, prefix, 'real') == ['*', '/', str(A ** 2), str(B ** 2), 'x1']


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


class TestReviewFindings:
    """#65's reviews: each a reproducer that misbehaved before the fix."""

    def test_a_number_and_its_reciprocal_cancel_in_f64_mode(self, engine: SimpliPyEngine) -> None:
        assert simplify(engine, '/ 5e307 5e307', 'f64') == ['1']
        assert simplify(engine, '* x1 / 1e308 1e308', 'f64') == ['x1']
        for prefix in ('/ 0.5 * 5e307 x1', '/ 1/3 * 5e307 x1'):
            once = simplify(engine, prefix, 'f64')
            assert list(engine.simplify(once, mode='f64')) == once

    @pytest.mark.parametrize('prefix, mode', [
        ('* * pow x1 1e-306 pow x1 1000 pow x1 1001', 'f64'),
        (f'* * pow x1 1/{3 ** 693} pow x1 4 pow x1 5', 'real'),
    ])
    def test_exponent_pools_merge_in_one_pass(self, engine: SimpliPyEngine, prefix: str, mode: str) -> None:
        once = simplify(engine, prefix, mode)
        assert list(engine.simplify(once, mode=mode)) == once
        assert ('2001' if mode == 'f64' else '9') in once, once

    def test_a_constant_absorbs_kept_literal_members(self, engine: SimpliPyEngine) -> None:
        prefix = '* x1 * <constant> * 5e-324 / 1 8.0685647420949626e321'
        assert simplify(engine, prefix, 'real') == ['*', '<constant>', 'x1']

    def test_a_leaf_read_as_inf_or_zero_does_not_cancel_in_f64_mode(self, engine: SimpliPyEngine) -> None:
        for prefix in ('/ 1e400 1e400', '/ 1e-400 1e-400'):
            assert simplify(engine, prefix, 'f64') == prefix.split()
            assert np.isnan(constant(engine, prefix.split()))
            assert simplify(engine, prefix, 'real') == ['1']

    def test_exponents_merge_beyond_2_53_only_in_real_mode(self, engine: SimpliPyEngine) -> None:
        prefix = '* pow x1 9007199254740992 x1'
        assert simplify(engine, prefix, 'f64') == ['*', 'x1', 'pow', 'x1', '9007199254740992']
        assert simplify(engine, prefix, 'real') == ['pow', 'x1', '9007199254740993']

    def test_a_big_fraction_coefficient_evaluates_without_overflow(self, engine: SimpliPyEngine) -> None:
        p, q = 7 ** 360, 3 ** 630      # about 1,011 and 999 bits; p/q is about 4.5e3
        out = simplify(engine, f'* x1 / {p} {q}', 'f64')
        value = engine.as_callable(out, ['x1'])(np.array([10.0]))[0]
        assert np.isfinite(value) and value == pytest.approx(10 * p / q, rel=1e-15)

    @pytest.mark.parametrize('prefix', [
        'pow rootn x1 9007199254740993 -1/2',
        'pow rootn x1 9007199254740993 -1/4',
        '* x2 pow rootn x1 9007199254740993 -1/2',
    ])
    def test_a_root_of_unknown_parity_settles_in_one_call(self, engine: SimpliPyEngine, prefix: str) -> None:
        for mode in MODES:
            once = simplify(engine, prefix, mode)
            assert list(engine.simplify(once, mode=mode)) == once, (mode, once)

    def test_a_bag_of_refused_literals_is_quadratic(self, engine: SimpliPyEngine) -> None:
        import time
        toks = ['x1']
        for k in range(256):
            toks = ['*', f'{k % 9 + 1}e2{k % 10}0'] + toks       # pairwise products beyond the cap
        t0 = time.perf_counter()
        for mode in MODES:
            simplify(engine, ' '.join(toks), mode)
        assert time.perf_counter() - t0 < 5.0                     # 29.7 s with the cubic partition
