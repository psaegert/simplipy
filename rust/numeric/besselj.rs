//! besselj(nu, x): the Bessel function of the first kind J_nu(x) of a real order and a real
//! argument, as scipy 1.18.1's `scipy.special.jv` computes it, bit for bit: a transliteration of
//! `xsf::cyl_bessel_j(double, double)`, which evaluates AMOS's complex-argument `besj` (with
//! `besy` for a negative non-integer order, `J_-v = cos(pi v) J_v - sin(pi v) Y_v`) and falls
//! back to Cephes's `jv` wherever AMOS gives NaN.
//!
//! The cells are scipy's: NaN for a non-integer order at a negative argument; J_v(+-inf) NaN;
//! J_0(0) = 1, J_v(0) = 0 for v > 0, inf for a negative non-integer v. Accuracy is scipy's:
//! measured against mpmath, the median error is 1-2 ulp for integer orders and about 50 ulp
//! for non-integer ones, up to about 2,600 ulp at points 0.1 or more from a zero of J, and an
//! absolute error below 4.1e-10 at every point measured (|nu| <= 50, x <= 1e4), most of it near
//! the zeros.
//!
//! The elementary functions, complex ones included, come from `numeric::elementary`: the
//! platform's libm, and the complex product and quotient as GCC's runtime computes them, as
//! for scipy's own build. So the bits equal scipy's on the same glibc host: measured identical
//! on 914,361 (order, argument) pairs spanning every regime above and every special cell.

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

/// J_nu(x), `scipy.special.jv(nu, x)` (scipy 1.18.1) bit for bit. See the module documentation.
pub fn besselj(v: f64, x: f64) -> f64 {
    if v != c_int(v) as f64 && x < 0.0 {
        return f64::NAN;
    }
    let res = cyl_bessel_j_complex(v, C64::real(x));
    if res.re.is_nan() {
        // AMOS returned NaN, possibly on overflow
        return cephes::jv(v, x);
    }
    res.re
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
}
