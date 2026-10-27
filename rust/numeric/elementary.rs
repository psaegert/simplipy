//! THE ELEMENTARY-FUNCTION SOURCE: every transcendental the float64 evaluator, the kernels and
//! the deployed realizations compute comes from here, and nowhere else in the `numeric` module.
//! Today that is the SYSTEM libm, bound through its C symbols; replacing it (with correctly
//! rounded implementations, say) is a change to this file alone.
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

/// THE C RUNTIME'S COMPLEX PRODUCT AND QUOTIENT. A C compiler multiplies `(a + bi)(c + di)` as
/// `(ac - bd) + (ad + bc)i` and calls the runtime's `__muldc3` only when both parts come out
/// NaN (C99 Annex G's recovery of infinities); it divides by calling the runtime's `__divdc3`.
/// Both are called here, not written out. On Linux they are libgcc's, the ones the compiled
/// scipy kernels call there, bound by their versioned names: a Rust library also carries
/// `compiler_builtins`' own `__divdc3` and `__muldc3` (LLVM's, whose quotient rounds
/// differently), and an unversioned reference resolves to those. Elsewhere the unversioned
/// names are bound: LLVM's routines, `compiler_builtins`' or the system's compiler-rt.
mod runtime {
    use super::Complex;

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[link(name = "gcc_s")]
    extern "C" {
        #[link_name = "__muldc3@GCC_4.0.0"]
        pub(super) fn muldc3(a: f64, b: f64, c: f64, d: f64) -> Complex;
        #[link_name = "__divdc3@GCC_4.0.0"]
        pub(super) fn divdc3(a: f64, b: f64, c: f64, d: f64) -> Complex;
    }

    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    extern "C" {
        #[link_name = "__muldc3"]
        pub(super) fn muldc3(a: f64, b: f64, c: f64, d: f64) -> Complex;
        #[link_name = "__divdc3"]
        pub(super) fn divdc3(a: f64, b: f64, c: f64, d: f64) -> Complex;
    }
}

/// `(a + bi)(c + di)` as a C compiler computes it (see `runtime`).
#[inline]
pub(crate) fn cmul(a: f64, b: f64, c: f64, d: f64) -> Complex {
    let x = a * c - b * d;
    let y = a * d + b * c;
    if x.is_nan() && y.is_nan() {
        // SAFETY: a pure C runtime function of four doubles.
        return unsafe { runtime::muldc3(a, b, c, d) };
    }
    Complex { re: x, im: y }
}

/// `(a + bi) / (c + di)` as a C compiler computes it (see `runtime`).
#[inline]
pub(crate) fn cdiv(a: f64, b: f64, c: f64, d: f64) -> Complex {
    // SAFETY: a pure C runtime function of four doubles.
    unsafe { runtime::divdc3(a, b, c, d) }
}

// The expected values are libgcc's, the runtime bound on Linux.
#[cfg(all(test, target_os = "linux", target_env = "gnu"))]
mod tests {
    use super::*;

    fn bits(z: Complex) -> (u64, u64) {
        let canon = |v: f64| {
            if v.is_nan() {
                f64::NAN.to_bits()
            } else {
                v.to_bits()
            }
        };
        (canon(z.re), canon(z.im))
    }

    #[test]
    fn complex_quotient_and_product_are_libgccs() {
        // (operands, libgcc_s's __divdc3 result), read off GCC 15.2's runtime: the plain
        // Smith branches, the scaling ones (subnormal and huge parts), the zero divisor and
        // the infinite parts.
        let h = |s: &str| -> f64 {
            match s {
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                "nan" => f64::NAN,
                _ => s.parse().unwrap(),
            }
        };
        let cases: [([f64; 4], (f64, f64)); 9] = [
            ([1.0, 2.0, 3.0, 4.0], (0.44, 0.08)),
            (
                [2.0, 0.0, 0.0, -27.939446353542056],
                (-0.0, 0.07158337980975946),
            ),
            ([308.3, 0.0, 0.0, -0.0], (h("inf"), h("nan"))),
            (
                [1e-310, 3.0, 1e-300, 7e-301],
                (1.4093959731543623e300, 2.0134228187919463e300),
            ),
            (
                [1.5e308, -1e308, 1e308, 1.7e308],
                (-0.05141388174807196, -0.9125964010282777),
            ),
            ([0.1, 0.2, 1e-320, 3e-321], (h("inf"), h("inf"))),
            ([h("inf"), 1.0, 2.0, 3.0], (h("inf"), h("-inf"))),
            ([1.0, 1.0, h("inf"), h("nan")], (0.0, 0.0)),
            (
                [1e-200, 1e-200, 1e-200, 3e-200],
                (0.39999999999999997, -0.20000000000000004),
            ),
        ];
        for ([a, b, c, d], (re, im)) in cases {
            assert_eq!(
                bits(cdiv(a, b, c, d)),
                bits(Complex { re, im }),
                "{a} {b} {c} {d}"
            );
        }
        // the product recovers the infinities a NaN + NaN hides (Annex G)
        assert_eq!(
            bits(cmul(f64::INFINITY, 0.0, 0.0, 1.0)),
            bits(Complex {
                re: f64::NAN,
                im: f64::INFINITY
            })
        );
        assert_eq!(
            bits(cmul(f64::INFINITY, f64::NAN, 1.0, 0.0)),
            bits(Complex {
                re: f64::INFINITY,
                im: f64::NAN
            })
        );
    }
}
