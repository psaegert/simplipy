"""The float64 kernels of the 0.15 operators (`rust/numeric/kernels.rs`), through the
`_core` entries `kernel_unary`/`kernel_binary`.

floor, ceil, max, min, mod and harmonic are the core's own; erf and atan2 are the platform's
libm through the core's elementary-function source, as every other transcendental is. Two
gates per kernel:

* BIT IDENTITY with numpy where numpy's own operation is the deployed one: floor and ceil
  are `np.floor`/`np.ceil`, mod is `np.mod` on float64, and max/min are `np.maximum`/
  `np.minimum` everywhere except the zero tie, which is pinned to IEEE 754-2019 (numpy
  returns its second operand there). Dense grids plus every special cell.
* ACCURACY against exact arithmetic: floor, ceil, max and min are exact and mod is
  correctly rounded (rationals); harmonic is within one ulp of mpmath away from its zeros
  on the negative axis, and within a stated absolute bound near them; erf and atan2 meet the
  two-ulp bar (glibc measures 0.5 and 1.0).
"""
import math
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from simplipy import _core

F = np.float64
RNG = np.random.default_rng(20261021)
TINY = 5e-324
BIG = np.finfo(F).max


def bits_equal(a, b) -> np.ndarray:
    """Bitwise equality, every NaN equal to every NaN."""
    a = np.asarray(a, dtype=F)
    b = np.asarray(b, dtype=F)
    return (a.view(np.int64) == b.view(np.int64)) | (np.isnan(a) & np.isnan(b))


def core1(name: str, x) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=F)
    out = np.empty_like(x)
    _core.kernel_unary(name, x, out)
    return out


def core2(name: str, a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, dtype=F), np.asarray(b, dtype=F))
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    out = np.empty_like(a)
    _core.kernel_binary(name, a, b, out)
    return out


SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -3.0, TINY, -TINY,
                    2.2250738585072014e-308, BIG, -BIG, 2.0 ** 52, -(2.0 ** 52) - 1, 2.0 ** 53 + 2,
                    np.inf, -np.inf, np.nan])


def dense_unary_grid() -> np.ndarray:
    ints = RNG.integers(-1000, 1000, 20_000).astype(F)
    near = np.concatenate([np.nextafter(ints, np.inf), np.nextafter(ints, -np.inf), ints + 0.5, ints - 0.5])
    return np.concatenate([
        RNG.uniform(-1e6, 1e6, 100_000), RNG.uniform(-4, 4, 100_000), near, ints,
        np.geomspace(1e-320, 1e308, 20_000), -np.geomspace(1e-320, 1e308, 20_000),
        2.0 ** 52 + RNG.uniform(-8, 8, 2_000), SPECIAL,
    ])


def dense_binary_grid() -> tuple[np.ndarray, np.ndarray]:
    sa, sb = np.meshgrid(SPECIAL, SPECIAL)
    ties = RNG.choice([0.0, -0.0, 1.0, -2.5, 3.0], 20_000)
    return (np.concatenate([RNG.uniform(-1e3, 1e3, 100_000), RNG.uniform(-5, 5, 100_000), ties,
                            -np.geomspace(1e-300, 1e-10, 5_000), RNG.integers(-50, 50, 20_000).astype(F), sa.ravel()]),
            np.concatenate([RNG.uniform(-1e3, 1e3, 100_000), RNG.uniform(-5, 5, 100_000), ties[::-1],
                            np.full(5_000, 3.0), RNG.integers(-7, 8, 20_000).astype(F), sb.ravel()]))


def ieee_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """numpy's maximum with the zero tie set to IEEE 754-2019's +0 (unless both are -0)."""
    out = np.maximum(a, b)
    tie = (a == 0) & (b == 0)
    out[tie] = a[tie] + b[tie]
    return out


def ieee_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.minimum(a, b)
    tie = (a == 0) & (b == 0)
    out[tie] = -((-a[tie]) + (-b[tie]))
    return out


# --------------------------------------------------------------------------- bit identity

@pytest.mark.parametrize('name', ['floor', 'ceil'])
def test_floor_ceil_are_numpy_bit_for_bit(name) -> None:
    x = dense_unary_grid()
    assert bits_equal(core1(name, x), getattr(np, name)(x)).all()


def test_mod_is_numpy_mod_bit_for_bit() -> None:
    a, b = dense_binary_grid()
    with np.errstate(all='ignore'):
        same = bits_equal(core2('mod', a, b), np.mod(a, b))
    assert same.all(), list(zip(a[~same][:5], b[~same][:5]))


@pytest.mark.parametrize('name, npf, ieee', [('max', np.maximum, ieee_max), ('min', np.minimum, ieee_min)])
def test_max_min_are_numpy_off_the_zero_tie_and_ieee_on_it(name, npf, ieee) -> None:
    a, b = dense_binary_grid()
    got = core2(name, a, b)
    zero_tie = (a == 0) & (b == 0)
    assert bits_equal(got, npf(a, b))[~zero_tie].all()
    assert bits_equal(got, ieee(a, b)).all()
    assert bits_equal(got, core2(name, b, a)).all()  # commutative bit for bit


# --------------------------------------------------------------------------- the cells

def test_max_min_cells() -> None:
    for a, b, mx, mn in [(-0.0, 0.0, 0.0, -0.0), (0.0, -0.0, 0.0, -0.0), (-0.0, -0.0, -0.0, -0.0),
                         (0.0, 0.0, 0.0, 0.0), (1.0, -np.inf, 1.0, -np.inf), (np.inf, 3.0, np.inf, 3.0)]:
        assert bits_equal(core2('max', a, b), mx).all(), (a, b)
        assert bits_equal(core2('min', a, b), mn).all(), (a, b)
    for a, b in [(np.nan, 1.0), (1.0, np.nan), (np.inf, np.nan), (np.nan, np.nan)]:
        assert np.isnan(core2('max', a, b)).all() and np.isnan(core2('min', a, b)).all()


def test_mod_cells() -> None:
    inf = np.inf
    for a, b, want in [(-7.0, 3.0, 2.0), (7.0, -3.0, -2.0), (3.0, inf, 3.0), (-3.0, inf, inf),
                       (3.0, -inf, -inf), (-3.0, -inf, -3.0), (-0.0, 3.0, 0.0), (0.0, -3.0, -0.0),
                       (-1e-20, 3.0, 3.0)]:
        assert bits_equal(core2('mod', a, b), want).all(), (a, b)
    for a, b in [(5.0, 0.0), (5.0, -0.0), (inf, 2.0), (-inf, 2.0), (np.nan, 1.0), (1.0, np.nan)]:
        assert np.isnan(core2('mod', a, b)).all(), (a, b)


def test_floor_ceil_cells() -> None:
    assert bits_equal(core1('floor', [-0.0, -0.5, 2.5, -2.5]), [-0.0, -1.0, 2.0, -3.0]).all()
    assert bits_equal(core1('ceil', [-0.0, -0.5, 2.5, -2.5]), [-0.0, -0.0, 3.0, -2.0]).all()
    x = dense_unary_grid()
    assert bits_equal(core1('ceil', x), -core1('floor', -x)).all()


def test_harmonic_cells() -> None:
    assert bits_equal(core1('harmonic', [0.0, -0.0, 1.0, 2.0, np.inf]), [0.0, -0.0, 1.0, 1.5, np.inf]).all()
    poles = [-1.0, -2.0, -3.0, -50.0, -1e10, -(2.0 ** 52), -1e300, -np.inf, np.nan]
    assert np.isnan(core1('harmonic', poles)).all()


# --------------------------------------------------------------------------- accuracy

def test_exact_operations_are_exact() -> None:
    """floor and ceil are exact; mod is the correctly rounded a - b floor(a/b); max and min
    return an operand."""
    x = dense_unary_grid()
    x = x[np.isfinite(x)][::7]
    for name, g in [('floor', math.floor), ('ceil', math.ceil)]:
        want = np.array([float(g(Fraction(v))) for v in x])
        want = np.where(want == 0, np.copysign(0.0, x), want)  # the sign of a zero result
        assert bits_equal(core1(name, x), want).all()
    a, b = dense_binary_grid()
    keep = np.isfinite(a) & np.isfinite(b) & (b != 0)
    a, b = a[keep][::5], b[keep][::5]
    for u, v, m in zip(a, b, core2('mod', a, b)):
        exact = Fraction(u) - Fraction(v) * math.floor(Fraction(u) / Fraction(v))
        want = float(exact) if exact != 0 else math.copysign(0.0, v)
        assert bits_equal(m, want), (u, v)
    for name in ('max', 'min'):
        got = core2(name, a, b)
        assert ((got == a) | (got == b)).all()


def _ulp(y):
    """The float64 spacing at the exact real y (an mpf)."""
    if y == 0:
        return mp.mpf(2) ** -1074
    e = int(mp.floor(mp.log(abs(y), 2)))
    return max(mp.mpf(2) ** (e - 52), mp.mpf(2) ** -1074)


def _ulps(v: float, y) -> float:
    """|v - y| in ulps of the exact y (an mpf)."""
    if y == 0:
        return 0.0 if v == 0 else math.inf
    return float(abs(mp.mpf(v) - y) / _ulp(y))


def _digits(x: float) -> int:
    return 0 if x == 0 else abs(int(math.log10(abs(x))))


def _harmonic_exact(x: float):
    """mpmath's H(x), with the precision raised by the digits its evaluation cancels."""
    with mp.workdps(40 + _digits(x)):
        return +mp.harmonic(mp.mpf(x))


def _reflection_scale(x: float):
    """max(|H(-x-1)|, |pi cot(pi x)|): the size of the two terms whose difference H is."""
    with mp.workdps(40 + _digits(x)):
        return max(abs(mp.harmonic(-mp.mpf(x) - 1)), abs(mp.pi * mp.cot(mp.pi * mp.mpf(x))))


def test_harmonic_within_one_ulp_on_the_principal_branch() -> None:
    xs = np.concatenate([RNG.uniform(-0.999, 130, 3_000), np.geomspace(1e-300, 1e-2, 400),
                         -np.geomspace(1e-300, 1e-2, 400), np.geomspace(130, 1e300, 300),
                         RNG.uniform(-0.51, -0.49, 100), RNG.uniform(0.49, 0.51, 100),
                         RNG.uniform(8.49, 8.51, 100), [np.nextafter(-1.0, 0.0), TINY, -TINY, BIG]])
    got = core1('harmonic', xs)
    worst = max((_ulps(v, _harmonic_exact(float(x))), x) for x, v in zip(xs, got))
    assert worst[0] <= 1.0, worst


def test_harmonic_integers_are_correctly_rounded() -> None:
    n = np.arange(1, 121)
    exact = [float(sum(Fraction(1, k) for k in range(1, m + 1))) for m in n]
    assert (core1('harmonic', n) == np.array(exact)).all()


def test_harmonic_reflection_branch() -> None:
    """For x < -1, H(x) = H(-x-1) - pi cot(pi x), and H crosses zero once between two poles,
    where the two terms cancel. Everywhere the error is within one ulp of H plus 2^-58 of the
    larger term; so it is within one ulp where they cancel by at most a factor 16, and
    within two (the bar) up to a factor 100."""
    ints = RNG.integers(-40, -1, 600).astype(F)
    xs = np.concatenate([RNG.uniform(-40, -1, 2_000), ints + RNG.choice([-1, 1], 600) * np.geomspace(1e-13, 0.4, 600),
                         -np.geomspace(41, 1e15, 300) - 0.25, RNG.uniform(-1.6, -1.4, 200)])
    xs = xs[(xs < -1) & (xs != np.floor(xs))]
    within16, within100, excess = [], [], []
    for x, v in zip(xs, core1('harmonic', xs)):
        y = _harmonic_exact(float(x))
        scale = _reflection_scale(float(x))
        excess.append((float(max(0, abs(mp.mpf(v) - y) - _ulp(y)) / scale), x))
        if scale <= 100 * abs(y):
            within100.append((_ulps(v, y), x))
            if scale <= 16 * abs(y):
                within16.append((_ulps(v, y), x))
    assert max(excess)[0] <= 2.0 ** -58, max(excess)
    assert max(within16)[0] <= 1.0, max(within16)
    assert max(within100)[0] <= 2.0, max(within100)


# --------------------------------------------------------------------------- the entries

def test_core_entries_refuse_what_they_cannot_honour() -> None:
    x, out = np.ones(3), np.empty(3)
    with pytest.raises(ValueError, match='no kernel'):
        _core.kernel_unary('besselj', x, out)
    with pytest.raises(ValueError, match='no kernel'):
        _core.kernel_unary_scalar('pow', 1.0)
    with pytest.raises(ValueError, match='no kernel'):
        _core.kernel_binary('floor', x, x, out)
    with pytest.raises(ValueError, match='elements'):
        _core.kernel_unary('floor', np.ones(2), out)
    with pytest.raises(ValueError, match='overlap'):
        _core.kernel_unary('floor', x, x)
    with pytest.raises(ValueError, match='contiguous'):
        _core.kernel_unary('floor', np.ones(6)[::2], out)
    ro = np.empty(3)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        _core.kernel_unary('floor', x, ro)
    with pytest.raises((TypeError, BufferError, ValueError)):
        _core.kernel_unary('floor', np.ones(3, dtype=np.float32), out)


def test_core_entries_broadcast_one_element_operands() -> None:
    out = np.empty(4)
    _core.kernel_binary('max', np.array([1.0, -5.0, 2.0, -0.0]), np.array([0.0]), out)
    assert bits_equal(out, [1.0, 0.0, 2.0, 0.0]).all()
    _core.kernel_unary('harmonic', np.array([2.0]), out)
    assert (out == 1.5).all()
    _core.kernel_unary('floor', np.empty(0), np.empty(0))


# --------------------------------------------------------------------------- erf and atan2


def test_erf_and_atan2_cells() -> None:
    pi, inf = np.pi, np.inf
    assert bits_equal(core1('erf', [0.0, -0.0, inf, -inf, TINY, -TINY]), [0.0, -0.0, 1.0, -1.0, TINY, -TINY]).all()
    assert np.isnan(core1('erf', [np.nan])).all()
    for y, x, want in [(0.0, 0.0, 0.0), (-0.0, 0.0, -0.0), (0.0, -0.0, pi), (-0.0, -0.0, -pi),
                       (1.0, -inf, pi), (-1.0, -inf, -pi), (inf, inf, pi / 4), (-inf, -inf, -3 * pi / 4),
                       (1.0, 0.0, pi / 2), (-1.0, 0.0, -pi / 2)]:
        assert bits_equal(core2('atan2', y, x), want).all(), (y, x)
    assert np.isnan(core2('atan2', np.nan, 1.0)).all() and np.isnan(core2('atan2', 1.0, np.nan)).all()


def test_erf_meets_the_bar() -> None:
    xs = np.concatenate([RNG.uniform(-6.5, 6.5, 3_000), np.geomspace(1e-300, 6.0, 500), -np.geomspace(1e-300, 6.0, 300),
                         RNG.uniform(0.98, 1.02, 200), RNG.uniform(5.8, 6.0, 200)])
    with mp.workdps(40):
        worst = max((_ulps(v, mp.erf(mp.mpf(float(x)))), x) for x, v in zip(xs, core1('erf', xs)))
    assert worst[0] <= 2.0, worst


def test_atan2_meets_the_bar() -> None:
    y = np.concatenate([RNG.uniform(-10, 10, 3_000), np.geomspace(1e-300, 1e300, 300), -np.geomspace(1e-300, 1e300, 300)])
    x = np.concatenate([RNG.uniform(-10, 10, 3_000), RNG.uniform(-1e3, 1e3, 300), -RNG.uniform(1e-3, 1e3, 300)])
    with mp.workdps(40):
        worst = max((_ulps(v, mp.atan2(mp.mpf(float(a)), mp.mpf(float(b)))), a, b)
                    for a, b, v in zip(y, x, core2('atan2', y, x)))
    assert worst[0] <= 2.0, worst
