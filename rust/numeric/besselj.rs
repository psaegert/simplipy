//! besselj(nu, x): the Bessel function of the first kind J_nu(x) of a real order and a real
//! argument, as scipy 1.18.1's `scipy.special.jv` computes it, bit for bit at every argument
//! `x >= 0` (`-0.0` included) and at a non-integer order everywhere: a transliteration of
//! `xsf::cyl_bessel_j(double, double)`, which evaluates AMOS's complex-argument `besj` (with
//! `besy` for a negative non-integer order, `J_-v = cos(pi v) J_v - sin(pi v) Y_v`) and falls
//! back to Cephes's `jv` wherever AMOS gives NaN.
//!
//! The cells are scipy's: NaN for a non-integer order at a negative argument; J_v(+-inf) NaN;
//! J_0(0) = 1, J_v(0) = 0 for v > 0, and at a negative non-integer v the pole's one-sided limit,
//! `+-inf` with the sign of `1/Gamma(v + 1)` (`J_-0.5(0) = inf`, `J_-1.5(0) = -inf`).
//! Accuracy is scipy's: measured against mpmath, the median error is 1-2 ulp for integer orders
//! and about 50 ulp for non-integer ones, up to about 2,600 ulp at points 0.1 or more from a
//! zero of J, and an absolute error below 4.1e-10 at every point measured (|nu| <= 50,
//! x <= 1e4), most of it near the zeros.
//!
//! Two departures from scipy, both where scipy's value is not J's (`besselj` below): an integer
//! order reads its reflection and its parity, so `J_n` is even or odd and `J_{-n} = (-1)^n J_n`
//! bit for bit (scipy loses a zero's sign at a few of these cells, and is NaN at a negative
//! argument beyond order 2^31); and a NaN of
//! scipy's under an order of at least about 3e17 is `+0.0` where a bound proves the value
//! underflows. scipy's other NaN cells at finite operands stay: the oscillating region of the
//! orders beyond 3e17, and negative non-integer orders below about -590 where the value nears
//! the float range's end (`J_-593.09(133.3) = -4.8e306`, scipy NaN).
//!
//! The elementary functions, complex ones included, come from `numeric::elementary`: the
//! platform's libm, and the complex product and quotient as GCC's runtime computes them, as
//! for scipy's own build. So the bits equal scipy's on the same glibc host: measured identical
//! on 914,361 (order, argument) pairs spanning every regime above and every special cell, and
//! on 2.5 million pairs with orders to 1e308 once the two departures are read.

// The transliterations keep the C++'s literals and control flow as written.
#[allow(
    clippy::approx_constant,
    clippy::assign_op_pattern,
    clippy::excessive_precision,
    clippy::needless_late_init,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::too_many_arguments
)]
mod amos;
#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::manual_range_contains,
    clippy::neg_cmp_op_on_partial_ord
)]
mod cephes;
mod cx;
#[rustfmt::skip]
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod tables;

use cx::{c_int, C64};

/// `xsf::cyl_bessel_j(v, complex z)` for a real `z`: AMOS's J, a negative order by reflection
/// (integer) or rotation through Y (non-integer).
fn cyl_bessel_j_complex(mut v: f64, z: C64) -> C64 {
    let n = 1;
    let kode = 1;
    let mut cy_j = [C64::new(f64::NAN, f64::NAN)];
    let mut cy_y = [C64::new(f64::NAN, f64::NAN)];
    if v.is_nan() || z.re.is_nan() || z.im.is_nan() {
        return cy_j[0];
    }
    let mut sign = 1;
    if v < 0.0 {
        v = -v;
        sign = -1;
    }
    // xsf answers an AMOS overflow (`ierr == 2`) from the exponentially scaled J. J of a real
    // argument is bounded by 1, so no real argument overflows (measured over the whole coverage
    // grid); the port leaves the NaN `set_error_and_nan` writes, which `besselj` answers from
    // Cephes.
    let (nz, ierr) = amos::besj(z, v, kode, n, &mut cy_j);
    set_error_and_nan(nz, ierr, &mut cy_j[0]);
    if sign == -1 && !reflect_jy(&mut cy_j[0], v) {
        let (nz, ierr) = amos::besy(z, v, kode, n, &mut cy_y);
        set_error_and_nan(nz, ierr, &mut cy_y[0]);
        cy_j[0] = rotate_jy(cy_j[0], cy_y[0], v);
    }
    cy_j[0]
}

/// xsf's `set_error_and_nan` with `ierr_to_sferr`: an AMOS domain, overflow or no-result code
/// makes the value NaN (an underflow count takes precedence and keeps it).
fn set_error_and_nan(nz: i32, ierr: i32, value: &mut C64) {
    if nz == 0 && matches!(ierr, 1 | 2 | 4 | 5) {
        *value = C64::new(f64::NAN, f64::NAN);
    }
}

/// J_{-n} = (-1)^n J_n for an integer order; false for a non-integer one.
fn reflect_jy(jy: &mut C64, v: f64) -> bool {
    if v != v.floor() {
        return false;
    }
    let i = c_int(v - 16384.0 * (v / 16384.0).floor());
    if (i & 1) != 0 {
        *jy = -*jy;
    }
    true
}

/// `cos(pi v) J - sin(pi v) Y`.
fn rotate_jy(j: C64, y: C64, v: f64) -> C64 {
    let c = cephes::cospi(v);
    let s = cephes::sinpi(v);
    c * j - s * y
}

/// J_nu(x): `scipy.special.jv(nu, x)` (scipy 1.18.1) bit for bit at every argument `x >= 0`,
/// `-0.0` included, and at a non-integer order everywhere. See the module documentation.
///
/// A negative integer order reads the reflection `J_{-n}(x) = (-1)^n J_n(x)`, and an integer
/// order at a negative argument the parity `J_n(-x) = (-1)^n J_n(x)`, so both hold bit for bit,
/// the sign of a zero included. scipy reflects the same way where AMOS answers, but its Cephes
/// fallback loses a zero's sign (`jv(-(2**52 + 1), 1e-10)` is `+0`); it reads the parity
/// through AMOS's rotation, which agrees in value but writes `+0` where an odd order's value
/// underflows (`J_1(-1e-320)`, `J_1(-0.0)`), and tests integrality through a C `int` cast,
/// which gives NaN at every integer order beyond 2^31 (`jv(2**31, -1.0)`).
///
/// Where scipy's routines give up (NaN at a finite argument, under an order of at least about
/// 3e17), a value the bound `|J_nu(x)| <= (x/2)^nu / Gamma(nu + 1)` puts below half the
/// smallest subnormal is `+0.0`; the other such cells stay NaN.
pub fn besselj(v: f64, x: f64) -> f64 {
    let integer = v.is_finite() && v == v.floor();
    if integer && v < 0.0 {
        let j = besselj(-v, x);
        return if (v * 0.5).fract() != 0.0 { -j } else { j };
    }
    if integer && (x < 0.0 || (x == 0.0 && x.is_sign_negative())) {
        let j = besselj(v, -x);
        return if (v * 0.5).fract() != 0.0 { -j } else { j };
    }
    if !integer && x < 0.0 {
        return f64::NAN;
    }
    let res = cyl_bessel_j_complex(v, C64::real(x));
    if !res.re.is_nan() {
        return res.re;
    }
    // AMOS returned NaN, possibly on overflow
    let r = cephes::jv(v, x);
    // The bound holds for an order >= -1/2 (a negative integer order was reflected above; a
    // negative non-integer order's value is dominated by Y and may be huge).
    if r.is_nan() && v >= 0.0 && v.is_finite() && x.is_finite() && x > 0.0 && underflows(v, x) {
        return 0.0;
    }
    r
}

/// Does `(x/2)^nu / Gamma(nu + 1)` lie below `e^-800` (half the smallest subnormal is `e^-745`)?
/// By Stirling's lower bound `Gamma(nu + 1) >= sqrt(2 pi nu) (nu/e)^nu` the bound is at most
/// `exp(nu ln(e x / (2 nu)))`; the logarithm is read with a margin far above its rounding
/// error (each term below 710 in magnitude), so a `true` is a proof.
fn underflows(nu: f64, x: f64) -> bool {
    let t = super::elementary::log(x) - super::elementary::log(2.0 * nu) + 1.0;
    nu >= 1.0 && t < -1e-9 && nu * t < -800.0
}

#[cfg(test)]
mod tests {
    use super::besselj;

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    /// scipy 1.18.1's cells (measured).
    #[test]
    fn cells_follow_scipy() {
        assert_eq!(besselj(0.0, 0.0), 1.0);
        assert_eq!(bits(besselj(1.0, 0.0)), bits(0.0));
        assert_eq!(bits(besselj(-1.0, 0.0)), bits(-0.0));
        assert_eq!(besselj(-0.5, 0.0), f64::INFINITY);
        assert_eq!(besselj(0.5, 0.0), 0.0);
        assert!(besselj(0.5, -1.0).is_nan());
        for (v, x) in [
            (0.0, f64::INFINITY),
            (0.0, f64::NEG_INFINITY),
            (1.0, f64::INFINITY),
        ] {
            assert!(besselj(v, x).is_nan(), "jv({v}, {x})");
        }
        assert_eq!(besselj(f64::INFINITY, 1.0), 0.0);
        assert!(besselj(f64::NAN, 1.0).is_nan() && besselj(1.0, f64::NAN).is_nan());
        // integer orders are real on the negative axis
        assert!(besselj(3.0, -1.0) < 0.0 && besselj(2.0, -1.0) > 0.0);
    }

    /// An integer order has its parity bit for bit, at a zero and beyond 2^31 too; scipy writes
    /// `+0` at `J_1(-0.0)` and NaN at `jv(2**31, -1)`.
    #[test]
    fn integer_orders_have_their_parity() {
        assert_eq!(bits(besselj(1.0, -0.0)), bits(-0.0));
        assert_eq!(bits(besselj(-1.0, -0.0)), bits(0.0));
        assert_eq!(bits(besselj(3.0, -1e-300)), bits(-0.0));
        assert_eq!(bits(besselj(2.0, -0.0)), bits(0.0));
        assert_eq!(besselj(0.0, -0.0), 1.0);
        for n in [2147483648.0, 2147483649.0, 1e17] {
            for x in [1.0, 3e9, 1e20] {
                let s = if (n * 0.5f64).fract() != 0.0 {
                    -1.0
                } else {
                    1.0
                };
                assert_eq!(
                    bits(besselj(n, -x)),
                    bits(s * besselj(n, x)),
                    "J({n}, -{x})"
                );
            }
        }
        // a non-integer order stays NaN on the negative axis
        assert!(besselj(2.5, -1.0).is_nan() && besselj(-2.5, -1.0).is_nan());
    }

    /// Beyond the orders scipy's routines reach, a value the bound puts below the subnormals
    /// is +0; the oscillating cells stay NaN.
    #[test]
    fn huge_orders_underflow_where_the_bound_says_so() {
        assert_eq!(bits(besselj(3.2e17, 3.2)), bits(0.0));
        assert_eq!(bits(besselj(1e300, 100.0)), bits(0.0));
        assert_eq!(bits(besselj(-1e300, 100.0)), bits(0.0));
        assert!(besselj(1e300, 1.5e300).is_nan());
    }
}
