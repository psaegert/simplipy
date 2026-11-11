//! THE ELEMENTARY-FUNCTION SOURCE: every transcendental the float64 evaluator, the kernels and
//! the deployed realizations compute comes from here, and nowhere else in the `numeric` module.
//! Today that is the SYSTEM libm, bound through its C symbols; replacing it (with correctly
//! rounded implementations, say) is a change to this file alone. The judge reads `log10` from
//! here too, for the decades that set its working precision. The complex product and quotient
//! are arithmetic, not the platform's: the core computes them itself (`numeric::complex`), the
//! same on every machine.
//!
//! Why the C symbols and not Rust's `f64::*` methods: Rust's std calls the platform libm for
//! most of them (so they match it at 0 ULP), but `f64::atanh` is Rust's OWN composition
//! (`0.5 * ln_1p(2x/(1-x))`), and its divergence from glibc is UNBOUNDED in ulps toward the
//! domain edges: measured 161 ulps at -0.999 and 1.5e13 ulps at
//! -(1 - 1e-15) -- the 4th significant digit -- with 39% of a dense (-1,1) grid differing.
//! (`asinh`/`acosh` measured 0-ulp identical to libm over 2.2M points on the same build.)
//! Binding the C symbols directly makes every consumer of this module compute one function.
//!
//! `pow` takes its operands through `black_box`, so the optimizer cannot specialize a constant
//! exponent into a different instruction sequence: LLVM rewrites `pow(x, 2.0)` -> `x * x` (and,
//! under fast-math, `pow(x, 0.5)` -> `sqrt`), which is ~1 ULP off the libm `pow` result.

mod c {
    extern "C" {
        pub fn sin(x: f64) -> f64;
        pub fn cos(x: f64) -> f64;
        pub fn tan(x: f64) -> f64;
        pub fn asin(x: f64) -> f64;
        pub fn acos(x: f64) -> f64;
        pub fn atan(x: f64) -> f64;
        pub fn sinh(x: f64) -> f64;
        pub fn cosh(x: f64) -> f64;
        pub fn tanh(x: f64) -> f64;
        pub fn asinh(x: f64) -> f64;
        pub fn acosh(x: f64) -> f64;
        pub fn atanh(x: f64) -> f64;
        pub fn exp(x: f64) -> f64;
        pub fn log(x: f64) -> f64;
        pub fn log10(x: f64) -> f64;
        pub fn pow(x: f64, y: f64) -> f64;
        pub fn erf(x: f64) -> f64;
        pub fn atan2(y: f64, x: f64) -> f64;
        pub fn fmod(x: f64, y: f64) -> f64;
        pub fn fmax(x: f64, y: f64) -> f64;
        pub fn fmin(x: f64, y: f64) -> f64;
        pub fn frexp(x: f64, e: *mut std::os::raw::c_int) -> f64;
        pub fn ldexp(x: f64, e: std::os::raw::c_int) -> f64;
        pub fn cabs(z: super::Complex) -> f64;
        pub fn cexp(z: super::Complex) -> super::Complex;
        pub fn clog(z: super::Complex) -> super::Complex;
        pub fn csqrt(z: super::Complex) -> super::Complex;
        pub fn csinh(z: super::Complex) -> super::Complex;
        pub fn ccosh(z: super::Complex) -> super::Complex;
    }
}

macro_rules! unary {
    ($($name:ident),*) => {$(
        #[inline]
        pub(crate) fn $name(x: f64) -> f64 {
            // SAFETY: a pure libm function of one double.
            unsafe { c::$name(x) }
        }
    )*};
}

unary!(
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, log, log10, erf
);

/// `x ** y`, the C `pow` (see the module documentation for the `black_box`).
#[inline]
pub(crate) fn pow(x: f64, y: f64) -> f64 {
    // SAFETY: a pure libm function of two doubles.
    unsafe { c::pow(std::hint::black_box(x), std::hint::black_box(y)) }
}

/// The angle of the point (x, y), the C `atan2(y, x)` with its IEEE cells.
#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    // SAFETY: a pure libm function of two doubles.
    unsafe { c::atan2(y, x) }
}

/// The exact remainder of `x / y` with the sign of `x`, the C `fmod`.
#[inline]
pub(crate) fn fmod(x: f64, y: f64) -> f64 {
    // SAFETY: a pure libm function of two doubles.
    unsafe { c::fmod(x, y) }
}

/// The C `fmax`: the larger operand, NaN only when both are.
#[inline]
pub(crate) fn fmax(x: f64, y: f64) -> f64 {
    // SAFETY: a pure libm function of two doubles.
    unsafe { c::fmax(x, y) }
}

/// The C `fmin`: the smaller operand, NaN only when both are.
#[inline]
pub(crate) fn fmin(x: f64, y: f64) -> f64 {
    // SAFETY: a pure libm function of two doubles.
    unsafe { c::fmin(x, y) }
}

/// The C `frexp`: `x = m 2^e` with `|m|` in [1/2, 1), or `(x, 0)` for zero, infinity and NaN.
#[inline]
pub(crate) fn frexp(x: f64) -> (f64, i32) {
    let mut e: std::os::raw::c_int = 0;
    // SAFETY: a pure libm function writing one int through the pointer it is given.
    let m = unsafe { c::frexp(x, &mut e) };
    (m, e)
}

/// The C `ldexp`: `x 2^e`, correctly rounded.
#[inline]
pub(crate) fn ldexp(x: f64, e: i32) -> f64 {
    // SAFETY: a pure libm function.
    unsafe { c::ldexp(x, e) }
}

/// A C99 `double _Complex`: two doubles, laid out as the C ABI passes and returns one.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct Complex {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

/// `|z|`, the C `cabs`.
#[inline]
pub(crate) fn cabs(z: Complex) -> f64 {
    // SAFETY: a pure libm function of one complex double.
    unsafe { c::cabs(z) }
}

/// `exp(z)`, the C `cexp`.
#[inline]
pub(crate) fn cexp(z: Complex) -> Complex {
    // SAFETY: a pure libm function of one complex double.
    unsafe { c::cexp(z) }
}

/// The principal `log(z)`, the C `clog`.
#[inline]
pub(crate) fn clog(z: Complex) -> Complex {
    // SAFETY: a pure libm function of one complex double.
    unsafe { c::clog(z) }
}

/// The principal `sqrt(z)`, the C `csqrt`.
#[inline]
pub(crate) fn csqrt(z: Complex) -> Complex {
    // SAFETY: a pure libm function of one complex double.
    unsafe { c::csqrt(z) }
}

/// `sinh(z)`, the C `csinh`.
#[inline]
pub(crate) fn csinh(z: Complex) -> Complex {
    // SAFETY: a pure libm function of one complex double.
    unsafe { c::csinh(z) }
}

/// `cosh(z)`, the C `ccosh`.
#[inline]
pub(crate) fn ccosh(z: Complex) -> Complex {
    // SAFETY: a pure libm function of one complex double.
    unsafe { c::ccosh(z) }
}
