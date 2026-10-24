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
/// The two are written out here, as libgcc (GCC 12 and later) computes them, rather than bound
/// (the product: Annex G's recovery; the quotient: Smith's 1962 method with the scaling of
/// Baudin and Smith 2012, the published algorithms libgcc implements):
/// a Rust library carries `compiler_builtins`' own `__divdc3` (LLVM's, scaling by `logb`), which
/// rounds differently, so linking the symbol would not reach libgcc's. Measured bit-identical to
/// the host libgcc_s (GCC 15.2) on 200,000 random operand quadruples, extremes, zeros,
/// infinities and NaN included.
mod runtime {
    use super::Complex;

    /// C99 Annex G, G.5.1: recover the infinities a NaN + NaN product hides.
    pub(super) fn muldc3(mut a: f64, mut b: f64, mut c: f64, mut d: f64) -> Complex {
        let (ac, bd, ad, bc) = (a * c, b * d, a * d, b * c);
        let mut x = ac - bd;
        let mut y = ad + bc;
        if x.is_nan() && y.is_nan() {
            let mut recalc = false;
            let box_ = |v: f64| if v.is_infinite() { 1.0f64 } else { 0.0 }.copysign(v);
            let unnan = |v: f64| if v.is_nan() { 0.0f64.copysign(v) } else { v };
            if a.is_infinite() || b.is_infinite() {
                a = box_(a);
                b = box_(b);
                c = unnan(c);
                d = unnan(d);
                recalc = true;
            }
            if c.is_infinite() || d.is_infinite() {
                c = box_(c);
                d = box_(d);
                a = unnan(a);
                b = unnan(b);
                recalc = true;
            }
            if !recalc
                && (ac.is_infinite() || bd.is_infinite() || ad.is_infinite() || bc.is_infinite())
            {
                a = unnan(a);
                b = unnan(b);
                c = unnan(c);
                d = unnan(d);
                recalc = true;
            }
            if recalc {
                x = f64::INFINITY * (a * c - b * d);
                y = f64::INFINITY * (a * d + b * c);
            }
        }
        Complex { re: x, im: y }
    }

    /// Smith's method with libgcc's scaling: halve near overflow, scale up by `1/eps` when the
    /// larger divisor part or a subnormal numerator part would lose bits, and the `ratio`
    /// underflow branch; then Annex G's recovery of zero divisors and infinite parts.
    pub(super) fn divdc3(mut a: f64, mut b: f64, mut c: f64, mut d: f64) -> Complex {
        const RBIG: f64 = f64::MAX / 2.0;
        const RMIN: f64 = f64::MIN_POSITIVE;
        const RMIN2: f64 = f64::EPSILON;
        const RMINSCAL: f64 = 1.0 / f64::EPSILON;
        const RMAX2: f64 = RBIG * RMIN2;
        let (x, y);
        // the parts are named so the larger divisor part is `big` and the other `small`
        let swap = c.abs() < d.abs();
        let big = if swap { d } else { c };
        if big.abs() >= RBIG {
            a /= 2.0;
            b /= 2.0;
            c /= 2.0;
            d /= 2.0;
        }
        let big = if swap { d } else { c };
        if big.abs() < RMIN2
            || (a.abs() < RMIN && b.abs() < RMAX2 && big.abs() < RMAX2)
            || (b.abs() < RMIN && a.abs() < RMAX2 && big.abs() < RMAX2)
        {
            a *= RMINSCAL;
            b *= RMINSCAL;
            c *= RMINSCAL;
            d *= RMINSCAL;
        }
        if swap {
            let ratio = c / d;
            let denom = c * ratio + d;
            if ratio.abs() > RMIN {
                x = (a * ratio + b) / denom;
                y = (b * ratio - a) / denom;
            } else {
                x = (c * (a / d) + b) / denom;
                y = (c * (b / d) - a) / denom;
            }
        } else {
            let ratio = d / c;
            let denom = d * ratio + c;
            if ratio.abs() > RMIN {
                x = (b * ratio + a) / denom;
                y = (b - a * ratio) / denom;
            } else {
                x = (a + d * (b / c)) / denom;
                y = (b - d * (a / c)) / denom;
            }
        }
        if x.is_nan() && y.is_nan() {
            if c == 0.0 && d == 0.0 && (!a.is_nan() || !b.is_nan()) {
                let inf = f64::INFINITY.copysign(c);
                return Complex {
                    re: inf * a,
                    im: inf * b,
                };
            }
            if (a.is_infinite() || b.is_infinite()) && c.is_finite() && d.is_finite() {
                let a = if a.is_infinite() { 1.0f64 } else { 0.0 }.copysign(a);
                let b = if b.is_infinite() { 1.0f64 } else { 0.0 }.copysign(b);
                return Complex {
                    re: f64::INFINITY * (a * c + b * d),
                    im: f64::INFINITY * (b * c - a * d),
                };
            }
            if (c.is_infinite() || d.is_infinite()) && a.is_finite() && b.is_finite() {
                let c = if c.is_infinite() { 1.0f64 } else { 0.0 }.copysign(c);
                let d = if d.is_infinite() { 1.0f64 } else { 0.0 }.copysign(d);
                return Complex {
                    re: 0.0 * (a * c + b * d),
                    im: 0.0 * (b * c - a * d),
                };
            }
        }
        Complex { re: x, im: y }
    }
}

/// `(a + bi)(c + di)` as a C compiler computes it (see `runtime`).
#[inline]
pub(crate) fn cmul(a: f64, b: f64, c: f64, d: f64) -> Complex {
    let x = a * c - b * d;
    let y = a * d + b * c;
    if x.is_nan() && y.is_nan() {
        return runtime::muldc3(a, b, c, d);
    }
    Complex { re: x, im: y }
}

/// `(a + bi) / (c + di)` as a C compiler computes it (see `runtime`).
#[inline]
pub(crate) fn cdiv(a: f64, b: f64, c: f64, d: f64) -> Complex {
    runtime::divdc3(a, b, c, d)
}

#[cfg(test)]
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
