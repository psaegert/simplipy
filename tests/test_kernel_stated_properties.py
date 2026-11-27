"""THE KERNELS' STATED PROPERTIES, PINNED.

Every cell and property that the docstrings of `simplipy.operators` and the kernel documentation
(`rust/numeric/kernels.rs`, `harmonic.rs`, `xsf.rs`, `ac/expr/terms.rs`'s odd heads,
`interval.rs`'s readings of the kernels) state about the float64 kernels, checked through the
deployed realizations, so that a kernel change cannot silently break a documented claim.

* Each named cell is checked through both entries, the scalar fast path and the array map.
* The plateaus, cuts and bands are checked on dense grids.
* The accuracy claims are checked against mpmath at 50 or more digits.
* The NaN propagation that strict's sibling confinement relies on (FX-76) is pinned as a
  complete table: it propagates through every operand of every kernel except at `pow(NaN, +-0)`,
  `pow(1, NaN)` and `besselk(NaN, +-0)`.
* The odd and even heads that the engine's sign laws read are checked bit for bit, signed zero
  included.

The references are IEEE 754 and C99 Annex F for the arithmetic, `atan2` and `pow` cells, scipy
1.18.1's cells for erfc, J, Y, I and K, and mpmath for the values. The libm-backed heads (the
odd and even heads, the `erf` tail) are measured identical on glibc 2.39 and 2.43, where CI runs.
"""
import math
import struct
import warnings

import mpmath as mp
import numpy as np
import pytest

from simplipy import operators as o

F = np.float64
inf, nan = math.inf, math.nan
PI = math.pi
TINY = 5e-324
BIG = 1.7976931348623157e308
RNG = np.random.default_rng(20261127)


def bits_equal(a, b) -> np.ndarray:
    """Bitwise equality, every NaN equal to every NaN."""
    a, b = np.asarray(a, dtype=F), np.asarray(b, dtype=F)
    return (a.view(np.int64) == b.view(np.int64)) | (np.isnan(a) & np.isnan(b))


def both_entries(fn, *args):
    """fn at `args` through the scalar entry and through the array map (three copies, so the map
    runs rather than the one-element broadcast)."""
    with np.errstate(all='ignore'):
        return fn(*args), fn(*[np.full(3, x, dtype=F) for x in args])


def ulp_of(y) -> mp.mpf:
    """The float64 spacing at the exact real y (an mpf)."""
    if y == 0:
        return mp.mpf(2) ** -1074
    e = int(mp.floor(mp.log(abs(y), 2)))
    return max(mp.mpf(2) ** (e - 52), mp.mpf(2) ** -1074)


# ------------------------------------------------------------------------------- the cells

#: (realization, operands, value): every cell the docstrings name, and the cells beside them that
#: decide how a sentence reads (an infinite order at x = 0, the end of a band).
CELLS = [
    # pow: C's pow, NaN for a negative base (-inf included) under a finite non-integer exponent;
    # a zero base under a non-integer exponent is +0 or +inf whatever its sign; integer exponents
    # keep IEEE's signs; numpy's fast paths (sqrt at 0.5 keeps -0); C99's infinite-exponent cells
    ('pow', (-2.0, 0.5), nan), ('pow', (-inf, 0.5), nan), ('pow', (-inf, 0.25), nan), ('pow', (-inf, 1.5), nan),
    ('pow', (-inf, -0.5), nan), ('pow', (-2.0, 1 / 3), nan), ('pow', (-1e-300, 0.25), nan),
    ('pow', (-0.0, 0.25), 0.0), ('pow', (-0.0, 1.5), 0.0), ('pow', (-0.0, -0.25), inf), ('pow', (-0.0, -1.5), inf),
    ('pow', (-0.0, 3.0), -0.0), ('pow', (-0.0, -3.0), -inf), ('pow', (-0.0, 2.0), 0.0), ('pow', (-0.0, -2.0), inf),
    ('pow', (-0.0, 0.5), -0.0), ('pow', (-0.0, -1.0), -inf), ('pow', (-inf, 3.0), -inf), ('pow', (-inf, 2.0), inf),
    ('pow', (-inf, -3.0), -0.0), ('pow', (-inf, -2.0), 0.0), ('pow', (-2.0, inf), inf), ('pow', (-0.5, inf), 0.0),
    ('pow', (-1.0, inf), 1.0), ('pow', (-1.0, -inf), 1.0), ('pow', (0.5, -inf), inf), ('pow', (-inf, inf), inf),
    ('pow', (-inf, -inf), 0.0), ('pow', (1.0, nan), 1.0), ('pow', (nan, 0.0), 1.0), ('pow', (nan, -0.0), 1.0),
    ('pow', (-1.0, nan), nan), ('pow', (nan, inf), nan), ('pow', (-8.0, 3.0), -512.0),
    # rootn: odd index signed (odd at zero too), even index principal (+0 at either zero), the
    # identity at 1, the reciprocal below 0, NaN at a zero, non-integer or infinite index
    ('rootn', (-8.0, 3.0), -2.0), ('rootn', (-0.0, 3.0), -0.0), ('rootn', (-0.0, -3.0), -inf),
    ('rootn', (0.0, -3.0), inf), ('rootn', (-4.0, 2.0), nan), ('rootn', (-0.0, 2.0), 0.0), ('rootn', (-0.0, 4.0), 0.0),
    ('rootn', (-0.0, -2.0), inf), ('rootn', (-inf, 3.0), -inf), ('rootn', (-inf, 2.0), nan), ('rootn', (inf, -2.0), 0.0),
    ('rootn', (-inf, -3.0), -0.0), ('rootn', (-0.0, 1.0), -0.0), ('rootn', (-0.0, -1.0), -inf),
    ('rootn', (-inf, 1.0), -inf), ('rootn', (nan, 1.0), nan), ('rootn', (8.0, 0.0), nan), ('rootn', (8.0, -0.0), nan),
    ('rootn', (8.0, 2.5), nan), ('rootn', (8.0, inf), nan), ('rootn', (8.0, nan), nan), ('rootn', (-8.0, -3.0), -0.5),
    ('rootn', (-2.0, 2.0 ** 53), nan), ('rootn', (-1.0, 2.0 ** 53 - 1), -1.0),
    # max / min: IEEE 754-2019, NaN-propagating, commutative at the zero tie
    ('max', (-0.0, 0.0), 0.0), ('max', (0.0, -0.0), 0.0), ('max', (-0.0, -0.0), -0.0), ('max', (nan, 1.0), nan),
    ('max', (1.0, nan), nan), ('max', (inf, nan), nan), ('min', (-0.0, 0.0), -0.0), ('min', (0.0, -0.0), -0.0),
    ('min', (0.0, 0.0), 0.0), ('min', (nan, 1.0), nan), ('min', (-inf, nan), nan),
    # mod: numpy's np.mod on float64
    ('mod', (-7.0, 3.0), 2.0), ('mod', (7.0, -3.0), -2.0), ('mod', (5.0, 0.0), nan), ('mod', (5.0, -0.0), nan),
    ('mod', (inf, 3.0), nan), ('mod', (-inf, 3.0), nan), ('mod', (-0.0, 3.0), 0.0), ('mod', (0.0, -3.0), -0.0),
    ('mod', (6.0, -3.0), -0.0), ('mod', (3.0, inf), 3.0), ('mod', (-3.0, inf), inf), ('mod', (3.0, -inf), -inf),
    ('mod', (-3.0, -inf), -3.0), ('mod', (-1e-20, 3.0), 3.0), ('mod', (-0.0, inf), 0.0), ('mod', (0.0, -inf), -0.0),
    # harmonic
    ('harmonic', (0.0,), 0.0), ('harmonic', (-0.0,), -0.0), ('harmonic', (inf,), inf), ('harmonic', (-inf,), nan),
    ('harmonic', (nan,), nan), ('harmonic', (-1.0,), nan), ('harmonic', (-17.0,), nan),
    ('harmonic', (-4503599627370496.0,), nan), ('harmonic', (-BIG,), nan), ('harmonic', (4.0,), 25.0 / 12.0),
    # erf and erfc
    ('erf', (0.0,), 0.0), ('erf', (-0.0,), -0.0), ('erf', (inf,), 1.0), ('erf', (-inf,), -1.0), ('erf', (nan,), nan),
    ('erfc', (0.0,), 1.0), ('erfc', (-0.0,), 1.0), ('erfc', (inf,), 0.0), ('erfc', (-inf,), 2.0), ('erfc', (nan,), nan),
    ('erfc', (-5.863584748755168,), 2.0), ('erfc', (np.nextafter(-5.863584748755168, 0),), 1.9999999999999998),
    ('erfc', (26.64174755704633,), 0.0), ('erfc', (np.nextafter(26.64174755704633, 0),), 1.1771759939871e-310),
    # atan2: every cell of C99 F.9.1.4 (y first, as numpy's arctan2)
    ('atan2', (0.0, -0.0), PI), ('atan2', (-0.0, -0.0), -PI), ('atan2', (0.0, 0.0), 0.0), ('atan2', (-0.0, 0.0), -0.0),
    ('atan2', (0.0, -1.0), PI), ('atan2', (-0.0, -1.0), -PI), ('atan2', (0.0, 1.0), 0.0), ('atan2', (-0.0, 1.0), -0.0),
    ('atan2', (-1.0, 0.0), -PI / 2), ('atan2', (-1.0, -0.0), -PI / 2), ('atan2', (1.0, 0.0), PI / 2),
    ('atan2', (1.0, -0.0), PI / 2), ('atan2', (1.0, -inf), PI), ('atan2', (-1.0, -inf), -PI), ('atan2', (1.0, inf), 0.0),
    ('atan2', (-1.0, inf), -0.0), ('atan2', (inf, 1.0), PI / 2), ('atan2', (-inf, 1.0), -PI / 2),
    ('atan2', (inf, -inf), 3 * PI / 4), ('atan2', (-inf, -inf), -3 * PI / 4), ('atan2', (inf, inf), PI / 4),
    ('atan2', (-inf, inf), -PI / 4), ('atan2', (nan, 1.0), nan), ('atan2', (1.0, nan), nan), ('atan2', (0.0, nan), nan),
    # besselj: J_0(0) = 1, J_nu(0) = 0 (nu > 0, negative integer nu), the pole's one-sided limit
    # with the sign of 1/Gamma(nu + 1); NaN for a non-integer order at x < 0 and at x = +-inf; an
    # infinite order is 0 on 0 <= x <= 2 and NaN elsewhere (the interval layer reads this)
    ('besselj', (0.0, 0.0), 1.0), ('besselj', (1.0, 0.0), 0.0), ('besselj', (2.5, 0.0), 0.0),
    ('besselj', (-2.0, 0.0), 0.0), ('besselj', (-0.5, 0.0), inf), ('besselj', (-1.5, 0.0), -inf),
    ('besselj', (-2.5, 0.0), inf), ('besselj', (0.5, -1.0), nan), ('besselj', (0.0, inf), nan),
    ('besselj', (2.0, -inf), nan), ('besselj', (nan, 0.0), nan), ('besselj', (inf, 0.0), 0.0),
    ('besselj', (inf, -0.0), 0.0), ('besselj', (-inf, 1.0), 0.0), ('besselj', (inf, 2.0), 0.0),
    ('besselj', (inf, 2.0000000000000004), nan), ('besselj', (inf, -1e-300), nan),
    # bessely: NaN at x < 0, at x = +inf and at an infinite order with x > 0; at x = 0, -inf for
    # nu >= 0 (+inf included) and the reflection or rotation below (-inf included); AMOS's cut
    # below x = 2.2250738585072014e-305 (-inf at orders >= 0, the reflection or rotation below);
    # the zero band from x = 7.2e8 to 2^51 at orders from 86 up to a bound that grows with x
    ('bessely', (0.0, 0.0), -inf), ('bessely', (2.5, -0.0), -inf), ('bessely', (TINY, 0.0), -inf),
    ('bessely', (-1.0, 0.0), inf), ('bessely', (-0.5, 0.0), -inf), ('bessely', (inf, 0.0), -inf),
    ('bessely', (-inf, 0.0), -inf), ('bessely', (1.0, -1.0), nan), ('bessely', (0.0, inf), nan),
    ('bessely', (inf, 1.0), nan), ('bessely', (-inf, 1.0), nan), ('bessely', (nan, 0.0), nan),
    ('bessely', (0.0, 2.2250738585072014e-305), -446.65535741385196),
    ('bessely', (0.0, np.nextafter(2.2250738585072014e-305, 0)), -inf), ('bessely', (2.5, 1e-306), -inf),
    ('bessely', (-1.0, 1e-306), inf), ('bessely', (-2.0, 1e-306), -inf), ('bessely', (-1.5, 1e-306), inf),
    ('bessely', (-0.5, 1e-306), -4.885633867381603e+136),
    ('bessely', (86.0, 7.2e8), 0.0), ('bessely', (-86.0, 7.2e8), 0.0), ('bessely', (-87.0, 7.2e8), -0.0),
    ('bessely', (1e5, 7.2e8), -0.0), ('bessely', (1e5, 2.0 ** 51), -0.0), ('bessely', (1e12, 1e15), -0.0),
    ('bessely', (86.0, 7.0e8), -7.376311339772446e-06), ('bessely', (85.0, 1e9), -2.4687490708759992e-05),
    ('bessely', (3e6, 7.2e8), -2.04909107575952e-05), ('bessely', (1e9, 7.2e8), -inf),
    ('bessely', (1e5, 2.0 ** 51 + 2), 1.3045706013829762e-08),
    # besseli: I_0(0) = 1, 0 at nu > 0 and negative integer nu, +inf at a negative non-integer nu;
    # NaN for a non-integer order at x < 0 and at x = +-inf (an infinite order included); 0 at an
    # infinite order and finite x; NaN near 0 at orders to 50 (every subnormal, up to about
    # 1.74e-308 at order 0), 0 or +-inf there above order 50; +inf from about x = 713.99 at order 0;
    # the Wronskian's zero at small x
    ('besseli', (0.0, 0.0), 1.0), ('besseli', (1.0, 0.0), 0.0), ('besseli', (-2.0, 0.0), 0.0),
    ('besseli', (-0.5, 0.0), inf), ('besseli', (-1.5, 0.0), inf), ('besseli', (0.5, -1.0), nan),
    ('besseli', (0.0, inf), nan), ('besseli', (0.0, -inf), nan), ('besseli', (inf, inf), nan),
    ('besseli', (-inf, inf), nan), ('besseli', (inf, 1.0), 0.0), ('besseli', (-inf, 1.0), 0.0),
    ('besseli', (inf, -1.0), 0.0), ('besseli', (inf, 0.0), 0.0), ('besseli', (0.0, TINY), nan),
    ('besseli', (0.0, -TINY), nan), ('besseli', (50.0, TINY), nan), ('besseli', (-50.0, TINY), nan),
    ('besseli', (0.0, 1.7412578939393163e-308), nan), ('besseli', (0.0, 1.75e-308), 1.0),
    ('besseli', (50.1, TINY), 0.0), ('besseli', (60.0, 1e-310), 0.0), ('besseli', (-60.5, 1e-310), inf),
    ('besseli', (0.0, 713.98), 1.7853251347671795e+308), ('besseli', (0.0, 713.99), inf),
    ('besseli', (0.5, 1e-250), 0.0),
    # besselk: +inf at x = +-0 at every order (NaN and infinite ones included) and below AMOS's
    # cut; NaN at x < 0 and at an infinite order with x > 0; 0 from about x = 697.87 at order 0
    # and past 710 (1 + |nu|); 0 at x = +inf up to order 2.53e305, NaN past it
    ('besselk', (0.0, 0.0), inf), ('besselk', (nan, 0.0), inf), ('besselk', (inf, 0.0), inf),
    ('besselk', (-inf, 0.0), inf), ('besselk', (1.0, -0.0), inf),
    ('besselk', (0.0, 2.2250738585072014e-305), 701.6045947689403),
    ('besselk', (100.0, 1e-306), inf), ('besselk', (1.0, -1.0), nan), ('besselk', (inf, 1.0), nan),
    ('besselk', (-inf, 1.0), nan), ('besselk', (inf, inf), nan), ('besselk', (0.0, 697.87), 3.9355447445856393e-305),
    ('besselk', (0.0, 697.88), 0.0), ('besselk', (1000.0, 711.0 * 1001), 0.0), ('besselk', (0.0, inf), 0.0),
    ('besselk', (2.53e305, inf), 0.0), ('besselk', (2.532e305, inf), nan),
    # Y and K at a nonzero order below about 1.77e-309: scipy's values leave the function on
    # 0 < x < 2 (the rotation divides by sin(pi nu)), and are Y_0's and K_0's from x = 2 on
    ('bessely', (TINY, 1.0), 0.0), ('bessely', (TINY, 2.0), -1.872597015481579e+306),
    ('bessely', (TINY, 5.0), -0.3085176252490338), ('bessely', (2.2250738585072014e-308, 1.0), 0.08825696421567694),
    ('besselk', (TINY, 0.1), inf), ('besselk', (TINY, 1.0), nan), ('besselk', (-TINY, 1.0), nan),
    ('besselk', (TINY, 5.0), 0.0036910983340425942), ('besselk', (2.2250738585072014e-308, 1.0), 0.4210244382407085),
]


@pytest.mark.parametrize('name, args, want', CELLS, ids=[f'{c[0]}{c[1]}' for c in CELLS])
def test_every_stated_cell(name, args, want) -> None:
    scalar, array = both_entries(getattr(o, name), *args)
    assert bits_equal(scalar, want), (name, args, float(scalar), want)
    assert bits_equal(array, want).all(), (name, args, array, want)


# ------------------------------------------------------------------------ plateaus and tails

def test_erfc_is_two_at_and_below_its_plateau_and_zero_at_and_above_its_cut() -> None:
    """Exactly 2.0 at and below x = -5.863584748755168 and +0.0 at and above 26.64174755704633,
    positive everywhere below the cut, subnormal from about x = 26.543 (the docstring's 'about
    26.55'), where the true value is still positive."""
    two_at, zero_at = -5.863584748755168, 26.64174755704633
    low = np.concatenate([np.linspace(-40.0, two_at, 1_000_001), -np.geomspace(40.0, 1e308, 20_000)])
    assert (o.erfc(low) == 2.0).all()
    high = np.concatenate([np.linspace(zero_at, 60.0, 1_000_001), np.geomspace(60.0, 1e308, 20_000)])
    v = o.erfc(high)
    assert bits_equal(v, 0.0).all()
    tail = np.linspace(26.4, np.nextafter(zero_at, 0), 1_000_001)
    v = o.erfc(tail)
    assert (v > 0).all()
    first_subnormal = tail[np.argmax(v < 2.2250738585072014e-308)]
    assert 26.54 < first_subnormal < 26.55


def test_erf_is_exactly_one_beyond_about_5_92() -> None:
    """erf(x) rounds to exactly +-1 from about |x| = 5.92 (5.921588 on a 1e-6 grid) and stays
    there; erf is odd bit for bit."""
    g = np.linspace(5.0, 8.0, 3_000_001)
    v = o.erf(g)
    first = g[np.argmax(v == 1.0)]
    assert 5.92 <= first < 5.93
    assert (v[g >= first] == 1.0).all() and (v[g < 5.92] < 1.0).all()
    assert bits_equal(o.erf(-g), -v).all()


# --------------------------------------------------------------------------------- accuracy

def _digits(v: float) -> int:
    return 0 if v == 0 else abs(int(math.log10(abs(v))))


def _harmonic_exact(x: float) -> mp.mpf:
    """H(x) from mpmath, the precision raised by the digits its evaluation cancels (the zero at
    0, and the poles and zeros on the negative axis)."""
    extra = _digits(x)
    if x < 0:
        f = x - math.floor(x)
        extra = max(extra, _digits(f), _digits(1 - f)) + 20
    with mp.workdps(50 + extra):
        return +mp.harmonic(mp.mpf(x))


def test_harmonic_within_0_53_ulp_on_x_above_minus_one() -> None:
    """Within 0.53 ulp of the true value on x > -1, outside 1e-320 < |x| < 1e-306 (next test):
    around 0, both branch switches (+-1/2, 8.5), the asymptotic branch and its far end past 2^60,
    and toward the pole at -1."""
    xs = np.concatenate([RNG.uniform(-1, 0, 600), -1 + np.geomspace(1e-16, 0.5, 200), RNG.uniform(0, 20, 600),
                         RNG.uniform(20, 2000, 200), np.geomspace(1e-305, 0.5, 400), -np.geomspace(1e-305, 0.5, 400),
                         np.geomspace(1, 1e308, 300), RNG.uniform(-0.51, -0.49, 100), RNG.uniform(0.49, 0.51, 100),
                         RNG.uniform(8.49, 8.51, 100), 2.0 ** 60 * RNG.uniform(0.9, 1.1, 50),
                         [TINY, -TINY, 1e-323, 2.2250738585072014e-308, BIG, np.nextafter(-1.0, 0.0)]])
    got = o.harmonic(xs)
    worst = max((float(abs(mp.mpf(float(v)) - y) / ulp_of(y)), x)
                for x, v in zip(xs.tolist(), got.tolist()) for y in [_harmonic_exact(x)])
    assert worst[0] <= 0.53, worst


def test_harmonic_within_1_5_ulp_where_its_low_word_underflows() -> None:
    """On 1e-320 < |x| < 1e-306 the double-double's low word falls below the subnormals, and the
    error reaches 1.48 ulp (measured on 48,000 points): H(x) = zeta(2) x there to far beyond 50
    digits."""
    xs = 10.0 ** RNG.uniform(-320, -306, 1500)
    xs = np.concatenate([xs, -xs[:300]])
    got = o.harmonic(xs)
    z2 = mp.zeta(2)
    worst = max((float(abs(mp.mpf(float(v)) - z2 * mp.mpf(float(x))) / ulp_of(z2 * mp.mpf(float(x)))), x)
                for x, v in zip(xs, got))
    assert worst[0] <= 1.5, worst


def test_harmonic_reflection_bound() -> None:
    """Below -1: |error| <= ulp(H) + 2^-58 max(|H(-x-1)|, |pi cot(pi x)|)."""
    ints = RNG.integers(-40, -1, 150).astype(F)
    xs = np.concatenate([RNG.uniform(-60, -1, 400), ints + RNG.choice([-1, 1], 150) * np.geomspace(1e-13, 0.45, 150),
                         -np.geomspace(60, 4e15, 100) - 0.25])
    xs = xs[(xs < -1) & (xs != np.floor(xs))]
    for x, v in zip(xs.tolist(), o.harmonic(xs).tolist()):
        y = _harmonic_exact(x)
        with mp.workdps(60 + abs(int(math.log10(abs(x))))):
            scale = max(abs(mp.harmonic(-mp.mpf(x) - 1)), abs(mp.pi * mp.cot(mp.pi * mp.mpf(x))))
        assert abs(mp.mpf(v) - y) <= ulp_of(y) + scale * mp.mpf(2) ** -58, x


#: besselj's worst points found (nu, x), each about 0.1 from a zero of J_nu, where |J| is near 0.008.
_BESSELJ_WORST = [(38.13958621743932, 98.89217889943711), (50.0, 99.55889804143621),
                  (47.0, 99.51061625974495), (28.11269205613587, 98.85216381063681)]


def test_besselj_accuracy_on_the_measured_range() -> None:
    """Against 50-digit mpmath on nu in [0, 50], x in [0, 100]: within 1.4e-14 absolute
    everywhere (6.1e-15 the largest measured, on 100,000 points), which near a zero of J_nu is
    many ulp; up to about 7,000 at 0.1 from one (6,853 at the first worst point)."""
    nus = np.concatenate([RNG.uniform(0, 50, 150), np.round(RNG.uniform(0, 50, 50)), [p[0] for p in _BESSELJ_WORST]])
    xs = np.concatenate([RNG.uniform(0, 100, 150), RNG.uniform(80, 100, 50), [p[1] for p in _BESSELJ_WORST]])
    got = o.besselj(nus, xs)
    worst_abs, worst_ulp = 0.0, 0.0
    with mp.workdps(50):
        for nu, x, v in zip(nus.tolist(), xs.tolist(), got.tolist()):
            y = mp.besselj(mp.mpf(nu), mp.mpf(x))
            err = abs(mp.mpf(v) - y)
            worst_abs = max(worst_abs, float(err))
            worst_ulp = max(worst_ulp, float(err / ulp_of(y)))
    assert worst_abs <= 1.4e-14, worst_abs
    assert worst_ulp <= 7_000, worst_ulp


@pytest.mark.parametrize('n', [3, 5, 7, -3])
def test_rootn_is_pow_of_the_magnitude_and_not_correctly_rounded(n) -> None:
    """rootn(x, n) computes pow(|x|, 1/|n|) with 1/|n| rounded to a double, the sign restored at
    an odd index and the reciprocal taken at a negative one; so it is not correctly rounded: up to
    about |ln x| 2^-53 / |n| relative beyond pow's own error, 120 ulp at n = 3 near the subnormals."""
    xs = np.concatenate([10.0 ** RNG.uniform(-323.3, 308.2, 400), [TINY, 1e-320, 1e-300, 1e300, BIG]])
    xs = np.concatenate([xs, -xs[:100]])
    k = abs(n)
    want = np.array([math.copysign(math.pow(abs(x), 1.0 / k), x) for x in xs])
    if n < 0:
        want = 1.0 / want
    assert bits_equal(o.rootn(xs, float(n)), want).all()
    with mp.workdps(60):
        worst = 0.0
        for x, v in zip(xs.tolist(), o.rootn(xs, float(n)).tolist()):
            t = math.copysign(1, x) * mp.root(mp.mpf(abs(x)), k)  # the real odd root
            t = t if n > 0 else 1 / t
            worst = max(worst, float(abs(mp.mpf(v) - t) / ulp_of(t)))
    assert worst <= 125, worst


def test_rootn_at_an_even_index_is_pow_of_its_reciprocal_except_at_two() -> None:
    """At an even index other than 2, rootn(x, n) is pow(x, 1/n) bit for bit. At 2, pow takes
    numpy's sqrt: they differ at -0 (rootn +0, pow -0) and wherever the libm's pow(x, 0.5) is not
    the correctly rounded square root, by one ulp (0.09% of points)."""
    xs = np.concatenate([10.0 ** RNG.uniform(-320, 308, 200_000), RNG.uniform(0, 10, 200_000), [0.0, -0.0, inf, -1.0, -inf, nan]])
    with np.errstate(all='ignore'):
        for n in (4.0, 6.0, 8.0, 16.0):
            assert bits_equal(o.rootn(xs, n), o.pow(xs, 1.0 / n)).all(), n
        r, p = o.rootn(xs, 2.0), o.pow(xs, 0.5)
    differ = ~bits_equal(r, p)
    assert bits_equal(o.rootn(-0.0, 2.0), 0.0) and bits_equal(o.pow(-0.0, 0.5), -0.0)
    finite = differ & (xs != 0)
    assert (np.abs(r[finite] - p[finite]) <= np.spacing(p[finite])).all()
    assert finite.sum() <= 0.002 * len(xs)


# ------------------------------------------------------------------------- regions and bands

def test_bessely_zero_band() -> None:
    """Y is +-0 on about 7.16e8 <= x <= 2^51 at orders from about 85.93 (and integer orders to
    -86) up to a bound that grows with x, and nonzero outside it."""
    for x, top in [(7.2e8, 2.6e6), (1e9, 1.9e8), (1e12, 9.9e11), (2.0 ** 51, 2.2e15)]:
        orders = np.concatenate([np.geomspace(86, top, 400), -np.arange(86.0, 200.0)])
        assert (o.bessely(orders, x) == 0).all(), x
        assert o.bessely(top * 1.05, x) != 0, x
    assert (o.bessely(np.array([85.9, 50.0, 1.5]), 1e9) != 0).all()
    assert (o.bessely(np.array([86.0, 1e4]), 7.0e8) != 0).all()
    assert (o.bessely(np.array([86.0, 1e4]), 2.0 ** 51 + 2) != 0).all()


def test_bessely_and_besselk_below_amoss_cut() -> None:
    """Below x = 2.2250738585072014e-305 (AMOS's `az < ufl`): Y is -inf at every order >= 0 and K
    is +inf at every order; at the cut both are finite."""
    ufl = 2.2250738585072014e-305
    xs = np.array([TINY, 1e-310, 2.2250738585072014e-308, 1e-306, np.nextafter(ufl, 0)])
    for nu in (0.0, 0.25, 1.0, 2.5, 100.0, 999.0):
        assert (o.bessely(nu, xs) == -inf).all(), nu
        assert (o.besselk(nu, xs) == inf).all() and (o.besselk(-nu, xs) == inf).all(), nu
    assert np.isfinite(o.bessely(0.0, ufl)) and np.isfinite(o.besselk(0.0, ufl))


def test_besseli_is_nan_near_zero_only_up_to_order_50() -> None:
    """I is NaN near 0 at orders |nu| <= 50: at every argument below about 1.1e-308 (and at
    scattered ones up to 6e-307); above order 50 the uniform expansion gives 0 or +-inf there
    instead."""
    tiny = np.array([TINY, 1e-320, 1e-310, 1e-308])
    for nu in np.concatenate([np.linspace(-50, 50, 401), [0.5, -0.5, -0.95, 49.9]]):
        assert np.isnan(o.besseli(nu, tiny)).all(), nu
    for nu in (50.0001, 50.1, 60.0, 80.0, 1000.0, -50.0001, -60.0, -60.5, -79.95):
        assert not np.isnan(o.besseli(nu, tiny)).any(), nu


def test_besseli_overflow_and_besselk_underflow_thresholds() -> None:
    """I_0 overflows from about x = 713.987; K_0 is 0 from about 697.873 (the true value about
    4e-305) and every K is 0 past 710 (1 + |nu|)."""
    g = np.linspace(713.9, 714.1, 20_001)
    assert abs(g[np.argmax(np.isinf(o.besseli(0.0, g)))] - 713.987) < 1e-3
    g = np.linspace(697.8, 697.95, 15_001)
    assert abs(g[np.argmax(o.besselk(0.0, g) == 0)] - 697.873) < 1e-3
    for nu in (0.0, 10.0, 1000.0, 1e5, -3.5):
        assert o.besselk(nu, np.nextafter(710.0 * (1 + abs(nu)), inf)) == 0.0


def test_the_interval_layers_readings_of_the_bessel_kernels() -> None:
    """`interval.rs` reads Y and K as NaN-free at every finite x >= 0 up to order 1e3
    (SCIPY_BESSEL_NU), I as NaN-free on 1e-300 <= |x| <= 1e150 there (IV_TINY, IV_HUGE; x < 0 at
    an integer order), and J as NaN-free up to order 1e3 (BESSELJ_NU_MEASURED; a negative
    non-integer order to -400, BESSELJ_NEG_NU_MEASURED, at x > 0)."""
    n = 300_000
    nu = np.concatenate([RNG.uniform(-1e3, 1e3, n), np.round(RNG.uniform(-1e3, 1e3, n // 3))])
    x = np.concatenate([10.0 ** RNG.uniform(-300, 308.25, len(nu) - 2_000), RNG.uniform(0, 3000, 2_000)])
    with np.errstate(all='ignore'):
        assert not np.isnan(o.bessely(nu, x)).any()
        assert not np.isnan(o.besselk(nu, x)).any()
        inside = x <= 1e150
        assert not np.isnan(o.besseli(nu[inside], x[inside])).any()
        integer = np.round(nu[inside])
        assert not np.isnan(o.besseli(integer, -x[inside])).any()
        keep = (nu >= -400) | (nu == np.floor(nu))
        assert not np.isnan(o.besselj(nu[keep], x[keep])).any()
        assert not np.isnan(o.besselj(np.round(nu), -x)).any()


# --------------------------------------------------------------------- NaN propagation (FX-76)

UNARY = ['neg', 'inv', 'abs', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
         'atanh', 'exp', 'log', 'floor', 'ceil', 'harmonic', 'erf', 'erfc']
BINARY = ['add', 'sub', 'mul', 'div', 'pow', 'rootn', 'max', 'min', 'mod', 'atan2', 'besselj', 'bessely', 'besseli',
          'besselk']
#: Quiet NaNs of both signs, one with a payload, and a signalling one.
NANS = [math.nan, -math.nan, struct.unpack('<d', struct.pack('<Q', 0x7ff8000000000123))[0],
        struct.unpack('<d', struct.pack('<Q', 0xfff4000000000001))[0]]
#: The operand a NaN meets: every special cell and a spread of magnitudes of both signs.
OTHER = np.concatenate([[0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 2.5, TINY, -TINY,
                         2.2250738585072014e-308, BIG, -BIG, inf, -inf, 1e17, 2.0 ** 53, 710.0, 1e300, -1e300],
                        RNG.uniform(-100, 100, 300), 10.0 ** RNG.uniform(-320, 308, 300),
                        -(10.0 ** RNG.uniform(-320, 308, 300)), np.round(RNG.uniform(-100, 100, 100))])
#: Where a NaN operand does NOT give NaN: (realization, NaN's position, the other operand, value).
NOT_PROPAGATED = {('pow', 0, 0.0): 1.0, ('pow', 1, 1.0): 1.0, ('besselk', 0, 0.0): inf}


@pytest.mark.parametrize('name', UNARY)
def test_a_nan_propagates_through_every_unary_head(name) -> None:
    for n in NANS:
        scalar, array = both_entries(getattr(o, name), n)
        assert np.isnan(scalar) and np.isnan(array).all(), (name, n)


@pytest.mark.parametrize('name', BINARY)
def test_a_nan_propagates_through_both_operands_except_the_named_cells(name) -> None:
    """Strict's sibling confinement (FX-76) carries a confiner's NaN up through both operands of
    + - * / max min mod atan2 besselj bessely besseli, rootn's radicand, a power's base under a
    nonzero exponent and besselk's argument, never through pow's exponent (pow(1, NaN) = 1) or
    besselk's order (besselk(NaN, +-0) = +inf). These three cells are the only exceptions."""
    f = getattr(o, name)
    for pos in (0, 1):
        for n in NANS:
            nn = np.full_like(OTHER, n)
            args = (nn, OTHER) if pos == 0 else (OTHER, nn)
            with np.errstate(all='ignore'):
                array = f(*args)
                scalar = np.array([f(float(a), float(b)) for a, b in zip(*args)])
            for v, a, s in zip(OTHER.tolist(), array.tolist(), scalar.tolist()):
                want = NOT_PROPAGATED.get((name, pos, v), nan)  # 0.0 and -0.0 are one key
                if (name, pos, v) == ('pow', 1, 1.0) and n is NANS[3]:
                    # C99 gives pow(1, y) = 1 for a quiet NaN y; glibc's pow answers a SIGNALLING
                    # NaN exponent with NaN, which only propagates more (no arithmetic makes one)
                    assert np.isnan(a) == np.isnan(s) and (np.isnan(a) or a == 1.0), (a, s)
                    continue
                assert bits_equal(a, want) and bits_equal(s, want), (name, pos, n, v, a, s)


# ------------------------------------------------------------------- odd heads and symmetries

_SYM_X = np.concatenate([RNG.uniform(0, 50, 50_000), 10.0 ** RNG.uniform(-320, 308, 50_000), RNG.uniform(0, 1, 20_000),
                         [0.0, TINY, inf, nan, BIG]])


@pytest.mark.parametrize('name', ['sin', 'sinh', 'tan', 'tanh', 'asin', 'asinh', 'atan', 'atanh', 'erf'])
def test_the_odd_functions_are_odd_bit_for_bit(name) -> None:
    """THE ODD HEADS (`ac/expr/terms.rs::odd_fun`): odd at every value and in float64 bit for
    bit, the sign of a zero included; the engine's sign laws move a sign through them in strict."""
    f = getattr(o, name)
    with np.errstate(all='ignore'):
        assert bits_equal(f(-_SYM_X), -f(_SYM_X)).all()


@pytest.mark.parametrize('name', ['abs', 'cos', 'cosh'])
def test_the_even_functions_are_even_bit_for_bit(name) -> None:
    f = getattr(o, name)
    with np.errstate(all='ignore'):
        assert bits_equal(f(-_SYM_X), f(_SYM_X)).all()


def test_the_bessel_symmetries_hold_bit_for_bit() -> None:
    """At an integer order: J_-n = (-1)^n J_n and J_n(-x) = (-1)^n J_n(x) (the odd head of an odd
    order), Y_-n = (-1)^n Y_n, I_-n = I_n and I_n(-x) = (-1)^n I_n(x); at every order K_-nu = K_nu;
    rootn of an odd index is odd. All bit for bit."""
    n = np.round(RNG.uniform(-80, 80, 100_000))
    x = np.concatenate([RNG.uniform(0, 100, 50_000), 10.0 ** RNG.uniform(-310, 4, 50_000)])
    sign = np.where(np.mod(n, 2) == 1, -1.0, 1.0)
    with np.errstate(all='ignore'):
        jn, yn, i_n = o.besselj(n, x), o.bessely(n, x), o.besseli(n, x)
        assert bits_equal(o.besselj(-n, x), sign * jn).all()
        assert bits_equal(o.besselj(n, -x), sign * jn).all()
        assert bits_equal(o.bessely(-n, x), sign * yn).all()
        assert bits_equal(o.besseli(-n, x), i_n).all()
        assert bits_equal(o.besseli(n, -x), sign * i_n).all()
        nu = np.concatenate([RNG.uniform(-100, 100, 50_000), n[:50_000]])
        assert bits_equal(o.besselk(-nu, x), o.besselk(nu, x)).all()
        for k in (3.0, 5.0, 7.0, 9.0):
            assert bits_equal(o.rootn(-_SYM_X, k), -o.rootn(_SYM_X, k)).all()


# ------------------------------------------------------------------------- defined functions

def test_defined_functions_are_their_expansions_at_the_edges() -> None:
    """sqrt, cot, sec, csc, acot, asec and acsc evaluate bit for bit as their definitions also at
    the edges: the subnormals, the largest double, the doubles beside +-1 and pi/2, where 1/x
    overflows, and the infinities, through the callable and the realizations."""
    import os
    from simplipy import SimpliPyEngine
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        engine = SimpliPyEngine.from_config(os.path.join(root, 'tests', 'fixtures', 'vocab32', 'config.yaml'))
    x = np.array([0.0, -0.0, TINY, -TINY, 1e-310, -1e-310, 2.2250738585072014e-308, 5.562684646268003e-309,
                  5.5e-309, BIG, -BIG, PI / 2, -PI / 2, PI, 1.0, -1.0, 1.0000000000000002, 0.9999999999999999,
                  -1.0000000000000002, -0.9999999999999999, 1e22, 2.0 ** 1023, inf, -inf, nan])
    direct = {'sqrt': lambda v: o.rootn(v, 2.0), 'cot': lambda v: o.inv(o.tan(v)), 'sec': lambda v: o.inv(o.cos(v)),
              'csc': lambda v: o.inv(o.sin(v)), 'acot': lambda v: o.atan(o.inv(v)),
              'asec': lambda v: o.acos(o.inv(v)), 'acsc': lambda v: o.asin(o.inv(v))}
    with np.errstate(all='ignore'):
        for name, fn in direct.items():
            got = np.asarray(engine.as_callable(f'{name}(x0)', variables=['x0'])(x), dtype=F)
            assert bits_equal(got, fn(x)).all(), (name, x[~bits_equal(got, fn(x))])
