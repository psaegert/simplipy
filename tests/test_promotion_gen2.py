# mypy: ignore-errors
"""The promotion instruments must speak the FULL generation-2 vocabulary (audit Tier-1 #2,
2026-08-03): `rootn` was missing from the high-precision equivalence table, so every
rootn-bearing rule silently lost the mp arm -- `contract_value_at` fell back to deployed
f64 (whose atom-level unreliability is exactly what the mp arm exists to fix),
`_overturn.judge_exact` returned UNDECIDABLE, and `prefold`'s zero-snap skipped rootn
spans. These tests pin the deployed rootn convention in the mp lift and that the mp arm
actually engages on rootn-bearing rules.
"""
import pytest
from mpmath import isinf, isnan, mp, mpf

from simplipy.promotion._hp_equiv import evaluate as hp_eval


@pytest.fixture(autouse=True)
def _dps():
    old = mp.dps
    mp.dps = 60
    yield
    mp.dps = old


def ev(tokens, env=None):
    return hp_eval(list(tokens), env or {}, [])


class TestRootnConvention:
    """The mp lift of IEEE rootn must match the deployed convention (one semantics,
    all surfaces: numeric.rs rootn_f64, operators.py rootn, verify._contract.c_rootn)."""

    def test_odd_root_is_signed_and_total(self):
        assert ev(['rootn', '(-8.0)', '3']) == mpf(-2)
        assert ev(['rootn', '8.0', '3']) == mpf(2)
        assert ev(['rootn', 'float("-inf")', '3']) == mpf('-inf')

    def test_even_root_is_principal(self):
        assert ev(['rootn', '9.0', '2']) == mpf(3)
        assert isnan(ev(['rootn', '(-9.0)', '2']))
        assert isnan(ev(['rootn', 'float("-inf")', '2']))
        assert ev(['rootn', 'float("inf")', '2']) == mpf('inf')

    def test_index_must_be_finite_nonzero_integer(self):
        assert isnan(ev(['rootn', '8.0', '0']))
        assert isnan(ev(['rootn', '8.0', '2.5']))
        assert isnan(ev(['rootn', '8.0', 'float("inf")']))
        assert isnan(ev(['rootn', '8.0', 'float("nan")']))

    def test_index_one_is_identity(self):
        assert ev(['rootn', '(-7.0)', '1']) == mpf(-7)

    def test_negative_index_is_inverse_with_one_zero(self):
        assert ev(['rootn', '8.0', '(-3)']) == mpf('0.5')
        # one zero: rootn(0, -n) = +inf
        assert ev(['rootn', '0.0', '(-2)']) == mpf('inf')

    def test_nan_argument_propagates(self):
        assert isnan(ev(['rootn', 'float("nan")', '3']))


class TestRootnEngages:
    def test_equiv_certifies_a_rootn_identity(self):
        # rootn(pow(x, 3), 3) == x (signed odd root of an odd power): the checker must
        # EVALUATE it, not crash out of the mp arm.
        from simplipy.promotion._hp_equiv import equiv
        r = equiv(['rootn', 'pow', 'x0', '3', '3'], ['x0'])
        assert r['ok'] is True, r

    def test_mp_value_at_no_longer_raises_on_rootn(self):
        # Before the fix hp_eval raised on the rootn token and contract_value_at silently
        # fell back to deployed f64 -- the mp arm (symbolic atoms, zero/pole snaps) was
        # dead for every rootn-bearing rule.
        from simplipy.promotion._ladder import mp_value_at
        v = mp_value_at(['rootn', '_0', '3'], {'_0': '(-8.0)'})
        assert v == mpf(-2)

    def test_overturn_can_arbitrate_rootn_rules(self):
        # judge_exact must reach a verdict (PROMOTE for the signed odd-root identity),
        # not UNDECIDABLE-out on evaluation.
        from simplipy.promotion._overturn import judge_exact
        v, _ = judge_exact(['rootn', 'pow', '_0', '3', '3'], ['_0'],
                           [{'_0': '2.0'}, {'_0': '(-3.0)'}, {'_0': '0.5'}])
        assert v == 'PROMOTE'

    def test_prefold_zero_snap_reaches_rootn_spans(self):
        # `- rootn 8.0 3 2.0` denotes exactly zero; the zero-snap must fold it to '0'
        # instead of skipping the span because hp_eval raised.
        import simplipy as sp
        from simplipy.promotion import _f64_eval
        from simplipy.promotion._pointwise import _PREFOLD_CACHE, prefold
        engine = sp.SimpliPyEngine.from_config(sp.get_path('acj-4-3'))
        _f64_eval.configure(engine)
        _PREFOLD_CACHE.clear()
        out = prefold(['*', '-', 'rootn', '8.0', '3', '2.0', '?0'])
        assert out == ['*', '0', '?0']


class TestPowConventionStillHolds:
    """Companion pins so the pow arm cannot regress while the table changes."""

    def test_pow_one_and_zero_exponent(self):
        assert ev(['pow', '1', 'float("nan")']) == mpf(1)
        assert ev(['pow', 'float("nan")', '0']) == mpf(1)

    def test_pow_negative_base_integer_exponent(self):
        assert ev(['pow', '(-2.0)', '3']) == mpf(-8)
        assert isnan(ev(['pow', '(-2.0)', '0.5']))

    def test_pow_infinite_exponent_magnitude_step(self):
        assert ev(['pow', '0.5', 'float("inf")']) == mpf(0)
        assert isinf(ev(['pow', '2.0', 'float("inf")']))
        assert ev(['pow', '(-1.0)', 'float("inf")']) == mpf(1)

    def test_math_constants_still_bind(self):
        assert abs(ev(['sin', 'np.pi'])) < mpf('1e-50')
        assert abs(float(ev(['log', 'np.e'])) - 1.0) < 1e-15
        assert abs(float(ev(['*', '2.0', 'x0'], {'x0': mpf(3)})) - 6.0) < 1e-15
