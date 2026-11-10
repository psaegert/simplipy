"""The float64 kernels of the 0.15 operators (`rust/numeric/kernels.rs`), through the
`_core` entries `kernel_unary`/`kernel_binary`.

floor, ceil, max, min, mod and harmonic are the core's own; erf and atan2 are the platform's
libm through the core's elementary-function source, as every other transcendental is;
besselj is scipy 1.18.1's jv, carried in the core and gated bit for bit against scipy (an
integer order's parity read on the negative axis, where scipy loses a zero's sign); erfc,
bessely, besseli and besselk are scipy 1.18.1's erfc, yv, iv and kv, carried in the core and
gated bit for bit against scipy. The AMOS kernels (jv, yv, kv) divide complex numbers, which
the core does itself and scipy does through the platform's runtime, so they meet scipy bit
for bit where the platform divides as the core does (elsewhere, as on Ubuntu 26.04, the two
differ by design; the core's elementary functions are the platform's libm on every machine, so
its values are not the same everywhere either). Two gates per kernel:

* BIT IDENTITY with numpy where numpy's own operation is the deployed one: floor and ceil
  are `np.floor`/`np.ceil`, mod is `np.mod` on float64, and max/min are `np.maximum`/
  `np.minimum` everywhere except the zero tie, which is pinned to IEEE 754-2019 (numpy
  returns its second operand there). Dense grids plus every special cell.
* ACCURACY against exact arithmetic: floor, ceil, max and min are exact and mod is
  correctly rounded (rationals); harmonic is within one ulp of mpmath away from its zeros
  on the negative axis, and within a stated absolute bound near them; erf and atan2 meet the
  two-ulp bar (glibc measures 0.5 and 1.0).
"""
import ctypes
import ctypes.util
import functools
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


class _Complex(ctypes.Structure):
    _fields_ = [('re', ctypes.c_double), ('im', ctypes.c_double)]


#: Quotients the core's own division gives (its table in `rust/numeric/complex.rs`), at a
#: Smith step and at the scaled branches, where platform runtimes are known to differ.
_QUOTIENTS = [((1.0, 2.0, 3.0, 4.0), (0.44, 0.08)),
              ((1.0, 2.0, 3.0, 5.0), (0.3823529411764706, 0.029411764705882346)),
              ((1e-310, 3.0, 1e-300, 7e-301), (1.4093959731543623e300, 2.0134228187919463e300)),
              ((1e-200, 1e-200, 1e-200, 3e-200), (0.39999999999999997, -0.20000000000000004)),
              ((1e-310, 1e-311, 3.0, 4.0), (1.36e-311, -1.48e-311)),
              ((1.5e308, -1e308, 1e308, 1.7e308), (-0.05141388174807196, -0.9125964010282777))]


@functools.cache
def platform_divides_as_the_core() -> bool:
    """Whether this platform's runtime (libgcc_s's `__divdc3`, which scipy's AMOS kernels call)
    divides complex numbers as the core does. Where it does not (Ubuntu 26.04's runtime, LLVM's
    on macOS) scipy's jv, yv and kv differ from the core by design, a named difference."""
    path = ctypes.util.find_library('gcc_s')
    if path is None:
        return False
    try:
        divide = ctypes.CDLL(path)['__divdc3']
    except (OSError, AttributeError):
        return False
    divide.restype, divide.argtypes = _Complex, [ctypes.c_double] * 4
    return all(bits_equal([z.re, z.im], want).all()
               for z, want in ((divide(*args), want) for args, want in _QUOTIENTS))


#: The kernels that divide complex numbers (AMOS).
AMOS = ('besselj', 'bessely', 'besselk')


def needs_the_cores_division(name: str) -> None:
    """Skip a comparison with scipy that this platform's complex division makes differ."""
    if name in AMOS and not platform_divides_as_the_core():
        pytest.skip(f"this platform divides complex numbers otherwise than the core, so scipy's {name} "
                    'differs from the core by design')


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


@pytest.mark.parametrize('y', [2, 2.0, -1, -1.0, 0.5, 0, 0.0, -0.0, 1, 1.0])
def test_pow_takes_numpys_fast_paths(y) -> None:
    """At the exponents numpy computes without `pow` (`x*x`, `1/x`, `sqrt(x)`, 1 and `x`), the
    core's power, the deployed realization and the evaluator give numpy's `array ** scalar` bit
    for bit: the sign of a zero under `sqrt` (`(-0.0) ** 0.5` is -0.0) included."""
    import simplipy.operators as ops
    x = np.concatenate([dense_unary_grid(), RNG.uniform(-1e154, 1e154, 20_000), RNG.uniform(-1e-160, 1e-160, 20_000)])
    with np.errstate(all='ignore'):
        want = x ** y
    assert bits_equal(core2('pow', x, float(y)), want).all()
    assert bits_equal(ops.pow(x, y), want).all()
    ev = _core.kernel_binary_scalar
    for v in SPECIAL:
        with np.errstate(all='ignore'):
            assert bits_equal(ev('pow', float(v), float(y)), np.array([v]) ** y).all(), (v, y)


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


# --------------------------------------------------------------------------- besselj


def besselj_grid() -> tuple[np.ndarray, np.ndarray]:
    """Orders and arguments over every regime scipy's jv switches on: random magnitudes of
    both signs (integer and not), the catalog ranges, orders just inside a power of two (where
    the C's left-to-right `fnu + n - 1` drops a bit), order close to argument, large
    arguments, large orders, the neighbourhood of zeros and of integer orders, and every
    special cell."""
    rng = np.random.default_rng(2026)
    parts = []
    v = 10.0 ** rng.uniform(-3, 4, 60_000) * rng.choice([-1, 1], 60_000)
    parts.append((np.where(rng.random(60_000) < 0.35, np.round(v), v),
                  10.0 ** rng.uniform(-6, 5, 60_000) * np.where(rng.random(60_000) < 0.2, -1, 1)))
    parts.append((rng.integers(-10, 11, 30_000).astype(F), rng.uniform(-50, 50, 30_000)))
    parts.append((rng.uniform(-5, 50, 30_000), rng.uniform(0, 100, 30_000)))
    k = rng.integers(1, 8, 20_000)
    parts.append((np.where(rng.random(20_000) < 0.5, -1, 1) * (2.0 ** k - rng.uniform(0, 1, 20_000)),
                  rng.uniform(0, 90, 20_000)))
    big = 10.0 ** rng.uniform(1, 5, 10_000)
    parts.append((big, big * rng.uniform(0.5, 1.5, 10_000)))
    parts.append((rng.uniform(0, 5, 5_000), 10.0 ** rng.uniform(4, 15.5, 5_000)))
    parts.append((10.0 ** rng.uniform(9, 16, 5_000), rng.uniform(0, 100, 5_000)))
    zeros = np.array([float(mp.besseljzero(0, i)) for i in range(1, 101)])
    parts.append((np.zeros(2_000), np.repeat(zeros, 20) + rng.normal(0, 1e-9, 2_000)))
    parts.append((np.round(rng.uniform(-30, 30, 5_000)) + rng.choice([0, 1e-12, -1e-12, 1e-6], 5_000),
                  rng.uniform(0, 60, 5_000)))
    cells = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 2.5, -2.5, 30.0, 1e3, -1e3, 1e300, -1e300,
             TINY, np.inf, -np.inf, np.nan]
    cv, cx = np.meshgrid(cells, cells)
    parts.append((cv.ravel(), cx.ravel()))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def scipy_besselj(nu, x) -> np.ndarray:
    """scipy 1.18.1's jv as the core reads it: at an integer order the reflection
    `(-1)^n jv(-n, x)` of a negative one and, at a negative argument (`-0.0` included), the
    parity `(-1)^n jv(n, -x)`."""
    special = pytest.importorskip('scipy.special')
    nu, x = np.broadcast_arrays(np.asarray(nu, dtype=F), np.asarray(x, dtype=F))
    integer = np.isfinite(nu) & (nu == np.floor(nu))
    flip = integer & ((x < 0) | ((x == 0) & np.signbit(x)))
    with np.errstate(all='ignore'):
        want = special.jv(np.where(integer, np.abs(nu), nu), np.where(flip, -x, x))
    odd = integer & (np.mod(nu, 2) == 1)
    return np.where(odd & (flip != (nu < 0)), -want, want)


def test_besselj_is_scipy_bit_for_bit() -> None:
    """The core carries scipy 1.18.1's jv (AMOS, the reflection through Y, the Cephes
    fallback), so its value is scipy's, bit for bit, with an integer order's parity read on
    the negative axis; a scipy release that changed jv would show here first."""
    needs_the_cores_division('besselj')
    nu, x = besselj_grid()
    got, want = core2('besselj', nu, x), scipy_besselj(nu, x)
    # where scipy gives up under an order beyond 1e17, the core's value may be a proven
    # underflow (`test_besselj_has_every_order`)
    filled = np.isnan(want) & (got == 0) & (np.abs(nu) >= 1e17)
    same = bits_equal(got, want) | filled
    assert same.all(), list(zip(nu[~same][:5], x[~same][:5]))


def test_besselj_has_every_order() -> None:
    """Every real order: integer orders of both signs to 2^53 and beyond keep their parity and
    reflection bit for bit, the sign of a zero included (scipy writes +0 at `J_1(-1e-320)` and
    NaN at `jv(2**31, -1)`); orders to 1e308 equal scipy wherever scipy has a value, and are
    +0 where scipy gives up and the value underflows."""
    orders = np.concatenate([np.arange(-60.0, 61.0), [2.0 ** 31, 2.0 ** 31 + 1, 2.0 ** 52 + 1,
                                                      2.0 ** 53, 1e17, 1e300]])
    xs = np.concatenate([[0.0, TINY, 1e-320, 1e-300, 1e-10, 3e9, 1e20, 1e300],
                         RNG.uniform(0, 100, 300), 10.0 ** RNG.uniform(-5, 6, 100)])
    nu, x = np.meshgrid(np.concatenate([orders, -orders]), xs)
    nu, x = nu.ravel(), x.ravel()
    odd = np.mod(nu, 2) == 1
    plus, minus = core2('besselj', nu, x), core2('besselj', nu, -x)
    assert bits_equal(minus, np.where(odd, -plus, plus)).all()                  # parity
    assert bits_equal(core2('besselj', -nu, x), np.where(odd, -plus, plus)).all()  # reflection
    assert bits_equal(core2('besselj', 1.0, -0.0), -0.0).all()
    assert bits_equal(core2('besselj', [3.2e17, 1e300, -1e300], [3.2, 100.0, 100.0]), 0.0).all()
    needs_the_cores_division('besselj')
    raw = scipy_besselj(nu, x)
    kept = ~np.isnan(raw)
    assert bits_equal(plus[kept], raw[kept]).all()
    filled = np.isnan(plus) != np.isnan(raw)
    assert (plus[filled] == 0).all() and (np.abs(nu[filled]) >= 1e17).all()


def test_besselj_cells() -> None:
    inf, nan = np.inf, np.nan
    for nu, x, want in [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (2.5, 0.0, 0.0), (-2.0, 0.0, 0.0), (-0.5, 0.0, inf),
                        (2.0, -1.0, 0.1149034849319005), (0.5, -1.0, nan), (0.0, inf, nan), (0.0, -inf, nan),
                        (inf, 1.0, 0.0), (nan, 1.0, nan), (1.0, nan, nan), (1000.0, 1.0, 0.0)]:
        assert bits_equal(core2('besselj', nu, x), want).all(), (nu, x)
    # J_{-n}(x) = (-1)^n J_n(x) for an integer order
    xs = RNG.uniform(0, 40, 200)
    for n in (1, 2, 5):
        assert np.allclose(core2('besselj', -n, xs), (-1) ** n * core2('besselj', n, xs), rtol=1e-13, atol=1e-15)


# ------------------------------------------------------------ erfc, bessely, besseli, besselk

#: scipy's realization of each kernel the core carries without a departure.
SCIPY = {'erfc': 'erfc', 'bessely': 'yv', 'besseli': 'iv', 'besselk': 'kv'}
ERFC_ZERO_AT, ERFC_TWO_AT = 26.64174755704633, -5.863584748755168


def scipy_value(name: str, *args):
    special = pytest.importorskip('scipy.special')
    with np.errstate(all='ignore'):
        return getattr(special, SCIPY[name])(*args)


def erfc_grid() -> np.ndarray:
    """The dense grid, and where erfc changes regime: its `1 - erf` below 1, its plateau at 2,
    its subnormal tail and its cut to 0."""
    rng = np.random.default_rng(20261103)
    return np.concatenate([dense_unary_grid(), rng.uniform(0.98, 1.02, 20_000), rng.uniform(-6.5, -5.5, 20_000),
                           rng.uniform(26.4, 26.8, 20_000), rng.uniform(7.9, 8.1, 5_000),
                           np.nextafter([ERFC_ZERO_AT, ERFC_ZERO_AT, ERFC_TWO_AT, ERFC_TWO_AT], [0, np.inf, 0, -np.inf])])


def scipy_bessel_grid() -> tuple[np.ndarray, np.ndarray]:
    """`besselj`'s grid, and where Y, I and K change regime or leave the function: subnormal and
    tiny orders and arguments, the uniform expansions from orders 50 (I) and 85.9 (AMOS), AMOS's
    cuts (below x = 2.2e-305, K past x = 697.9), I's overflow, Y's zeros from x = 7.2e8, integer
    and half-integer orders of both signs, and every infinite and NaN operand."""
    rng = np.random.default_rng(20261104)
    orders = np.array([0.0, 5e-324, 1e-310, 1e-300, 1e-100, 1e-10, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 7.3, 49.9, 50.0,
                       50.1, 85.9, 86.0, 86.1, 100.0, 1000.0, 1e17, np.inf, np.nan])
    xs = np.concatenate([[0.0, -0.0, TINY, 1e-320, 1e-310, 2.2250738585072014e-308, 1e-306, 2.2250738585072014e-305,
                          3e-305, 1e-300, 1e-200, 1e-100, 1e-10, 0.5, 2.0, 697.0, 697.87, 698.0, 712.0, 740.0,
                          7.158e8, 7.16e8, 1e9, 2.0 ** 51, 2.0 ** 51 + 1, 1e300, -1.0, -1e-300, -30.0,
                          np.inf, -np.inf, np.nan], 10.0 ** rng.uniform(-320, 3.3, 1_000)])
    a, b = np.meshgrid(np.concatenate([orders, -orders]), xs)
    nu, x = besselj_grid()
    return (np.concatenate([nu, a.ravel(), rng.uniform(-100, 100, 20_000), np.round(rng.uniform(-60, 60, 10_000))]),
            np.concatenate([x, b.ravel(), 10.0 ** rng.uniform(-8, 3.2, 20_000), -(10.0 ** rng.uniform(-8, 2.8, 10_000))]))


def test_erfc_is_scipy_bit_for_bit() -> None:
    """The core carries scipy 1.18.1's erfc (Cephes), so its value is scipy's, bit for bit."""
    x = erfc_grid()
    same = bits_equal(core1('erfc', x), scipy_value('erfc', x))
    assert same.all(), x[~same][:5]


@pytest.mark.parametrize('name', ['bessely', 'besseli', 'besselk'])
def test_scipy_bessels_are_scipy_bit_for_bit(name) -> None:
    """The core carries scipy 1.18.1's yv (AMOS, Cephes where AMOS gives NaN), iv (Cephes) and kv
    (AMOS), so each value is scipy's, bit for bit, the cells where scipy leaves the function
    included (`test_scipy_bessel_cells`); a scipy release that changed one would show here."""
    needs_the_cores_division(name)
    nu, x = scipy_bessel_grid()
    same = bits_equal(core2(name, nu, x), scipy_value(name, nu, x))
    assert same.all(), list(zip(nu[~same][:5], x[~same][:5]))


def test_scipy_bessel_cells() -> None:
    """float64's cells of erfc, Y, I and K, scipy's: the poles and their signs at x = 0, the
    plateaus and cuts, the infinite and NaN operands, and where scipy leaves the function (Y and K
    infinite below x = 2.2e-305, Y zero from x = 7.2e8 at order 86 and above, I zero where its
    Wronskian overflows and NaN at a subnormal argument, `yv(-0.5, x)` far off near 0)."""
    inf, nan = np.inf, np.nan
    cells = [
        ('erfc', (0.0,), 1.0), ('erfc', (-0.0,), 1.0), ('erfc', (inf,), 0.0), ('erfc', (-inf,), 2.0),
        ('erfc', (nan,), nan), ('erfc', (ERFC_ZERO_AT,), 0.0), ('erfc', (np.nextafter(ERFC_ZERO_AT, 0),), 1.1771759939871e-310),
        ('erfc', (ERFC_TWO_AT,), 2.0), ('erfc', (np.nextafter(ERFC_TWO_AT, 0),), 1.9999999999999998),
        ('bessely', (0.0, 0.0), -inf), ('bessely', (2.5, -0.0), -inf), ('bessely', (-1.0, 0.0), inf),
        ('bessely', (-2.0, 0.0), -inf), ('bessely', (-0.5, 0.0), -inf), ('bessely', (-1.5, 0.0), inf),
        ('bessely', (-2.5, 0.0), -inf), ('bessely', (1.0, -1.0), nan), ('bessely', (0.0, inf), nan),
        ('bessely', (inf, 1.0), nan), ('bessely', (100.0, 1e9), -0.0), ('bessely', (150.0, 1e9), 0.0),
        ('bessely', (0.0, 1e-306), -inf), ('bessely', (-1.5, 1e-300), inf),
        ('bessely', (-0.5, 1e-305), -1.544973083458289e+136),
        ('besseli', (0.0, 0.0), 1.0), ('besseli', (1.0, -0.0), 0.0), ('besseli', (-0.5, 0.0), inf),
        ('besseli', (-1.5, 0.0), inf), ('besseli', (0.5, -1.0), nan), ('besseli', (0.0, inf), nan),
        ('besseli', (2.0, -inf), nan), ('besseli', (inf, 1.0), 0.0), ('besseli', (-inf, 1.0), 0.0),
        ('besseli', (0.0, 1e-310), nan), ('besseli', (0.5, 1e-250), 0.0), ('besseli', (51.0, -1e-10), -0.0),
        ('besseli', (0.0, 1e300), inf), ('besseli', (50.5, 1e300), nan), ('besseli', (60.0, 720.0), inf),
        ('besselk', (0.0, 0.0), inf), ('besselk', (nan, 0.0), inf), ('besselk', (1.0, -1.0), nan),
        ('besselk', (0.0, inf), 0.0), ('besselk', (inf, 1.0), nan), ('besselk', (0.0, 698.0), 0.0),
        ('besselk', (0.0, 1e-306), inf), ('besselk', (1.5, 1e-203), inf), ('besselk', (2.5e305, inf), 0.0),
        ('besselk', (2.54e305, inf), nan),
    ]
    for name, args, want in cells:
        got = core1(name, [args[0]]) if len(args) == 1 else core2(name, *args)
        assert bits_equal(got, want).all(), (name, args, got)
        assert bits_equal(got, scipy_value(name, *args)).all(), (name, args)
