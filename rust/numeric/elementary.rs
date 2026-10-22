//! THE ELEMENTARY-FUNCTION SOURCE: every transcendental the float64 evaluator, the kernels and
//! the deployed realizations compute comes from here, and nowhere else in the `numeric` module.
//! Today that is the SYSTEM libm, bound through its C symbols; replacing it (with correctly
//! rounded implementations, say) is a change to this file alone.
//!
//! Why the C symbols and not Rust's `f64::*` methods: Rust's std calls the platform libm for
//! most of them (so they match it at 0 ULP), but `f64::atanh` is Rust's OWN composition
//! (`0.5 * ln_1p(2x/(1-x))`), and its divergence from glibc is UNBOUNDED in ulps toward the
//! domain edges: measured 2026-08-15 (audit B6), 161 ulps at -0.999 and 1.5e13 ulps at
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
        pub fn pow(x: f64, y: f64) -> f64;
        pub fn erf(x: f64) -> f64;
        pub fn atan2(y: f64, x: f64) -> f64;
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

unary!(sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, log, erf);

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
