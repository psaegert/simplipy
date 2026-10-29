//! The float64 kernels of every operator, behind a plain Rust API: scalar functions, a lookup
//! by operator token, and element-wise maps over slices. Nothing here knows about Python; the
//! `_core` extension wraps the maps (`kernel_unary`/`kernel_binary`), the deployed realizations
//! in `simplipy.operators` call them, and another frontend can call the same functions.
//!
//! ONE FUNCTION PER OPERATOR. The evaluator's dispatch (`numeric::unary_fn` / `binary_fn`, the
//! fold, the tape and the miner) resolves the existing operators to these same functions, so
//! the engine and the deployed realization compute the same bits by construction. The
//! transcendentals come from `numeric::elementary`, the one elementary-function source.
//!
//! The operators added in 0.15 (`floor`, `ceil`, `max`, `min`, `mod`, `harmonic`, `erf`,
//! `atan2`, `besselj`) are in the evaluator's dispatch too. They have interval enclosures;
//! `besselj` also has its high-precision and contract kernel (`hiprec::besselj`), and every
//! certificate that needs one refuses the others.

pub use super::besselj::besselj;
use super::elementary;
pub use super::harmonic::harmonic;

// ------------------------------------------------------------------- the existing vocabulary

/// `-x`.
#[inline]
pub fn neg(x: f64) -> f64 {
    -x
}

/// `1/x`, IEEE: `1/+0 = +inf`, `1/-0 = -inf`.
#[inline]
pub fn inv(x: f64) -> f64 {
    1.0 / x
}

/// `|x|`, `+0` at either zero.
#[inline]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

macro_rules! from_elementary {
    ($($name:ident),*) => {$(
        #[doc = concat!("`", stringify!($name), "`, from the elementary-function source.")]
        #[inline]
        pub fn $name(x: f64) -> f64 {
            elementary::$name(x)
        }
    )*};
}

from_elementary!(
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, log, erf
);

/// `a + b`.
#[inline]
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

/// `a - b`.
#[inline]
pub fn sub(a: f64, b: f64) -> f64 {
    a - b
}

/// `a * b`.
#[inline]
pub fn mul(a: f64, b: f64) -> f64 {
    a * b
}

/// `a / b`, IEEE: the sign of a zero divisor participates (`1/-0 = -inf`), `0/0 = NaN`.
#[inline]
pub fn div(a: f64, b: f64) -> f64 {
    a / b
}

/// `x ** y`: the C `pow`, except that a negative base, `-inf` included, under a finite
/// non-integer exponent is NaN (the real-analysis reading; C99 gives `pow(-inf, 0.5) = inf`).
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    super::aligned_pow(x, y)
}

/// The real n-th root (IEEE-754 `rootn`): odd index signed, even index principal, NaN for a
/// zero or non-integer index, a negative index the reciprocal.
#[inline]
pub fn rootn(x: f64, n: f64) -> f64 {
    super::rootn_f64(x, n)
}

// ------------------------------------------------------------------- the 0.15 operators

/// The largest integer not above `x` (IEEE roundToIntegralTowardNegative): exact, the sign of a
/// zero result kept (`floor(-0) = -0`), infinities and NaN unchanged.
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// The smallest integer not below `x` (IEEE roundToIntegralTowardPositive): exact, with
/// `ceil(-0.5) = -0` and `ceil(-0) = -0`.
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// IEEE 754-2019 `maximum`: a NaN operand gives NaN (that operand), and -0 < +0, so
/// `max(-0, +0) = max(+0, -0) = +0` and the operation is commutative bit for bit. Off the
/// zero tie it equals numpy's `np.maximum`; on it, `np.maximum` returns its second operand.
/// Written as selects, so a map over it vectorizes: on a tie the operands' bits are ANDed
/// (equal values have equal bits, and +0 & -0 is +0).
#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    let r = if a > b { a } else { b };
    let r = if a == b {
        f64::from_bits(a.to_bits() & b.to_bits())
    } else {
        r
    };
    if a.is_nan() {
        a
    } else {
        r // a NaN `b` is never below `a` nor equal to it, so `r` is `b`
    }
}

/// IEEE 754-2019 `minimum`: NaN-propagating, -0 < +0, commutative bit for bit; on a tie the
/// operands' bits are ORed (-0 | +0 is -0).
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    let r = if a < b { a } else { b };
    let r = if a == b {
        f64::from_bits(a.to_bits() | b.to_bits())
    } else {
        r
    };
    if a.is_nan() {
        a
    } else {
        r
    }
}

/// The `mod` operator: numpy's `np.mod` on float64, the remainder with the sign of the
/// divisor (Python's `%`). A transliteration of numpy's `npy_remainder` (numpy 2.5.4,
/// `npymath/npy_math_internal.h.src`, the modulus branch of `npy_divmod`; Copyright (c)
/// 2005-2025, NumPy Developers, BSD-3-Clause, notice in `LICENSES_bundled.txt`): the
/// exact `fmod`, moved by one rounded addition of `b` when its sign differs from the
/// divisor's. So `mod(a, 0) = NaN`, `mod(+-inf, b) = NaN`, a zero result takes the divisor's
/// sign (`mod(-0, 3) = +0`, `mod(0, -3) = -0`), `mod(-3, inf) = inf` and
/// `mod(3, -inf) = -inf`, and a tiny negative dividend can return `b` itself
/// (`mod(-1e-20, 3) = 3`, the correctly rounded `3 - 1e-20`).
#[inline]
pub fn modulo(a: f64, b: f64) -> f64 {
    let m = a % b; // fmod: exact
    if b == 0.0 {
        return m; // NaN
    }
    if m != 0.0 {
        // NaN is nonzero here as in C; it stays NaN
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// `atan2(y, x)`, the angle of the point (x, y) in (-pi, pi], from the elementary-function
/// source with C99's cells: `atan2(+-0, +0) = +-0`, `atan2(+-0, -0) = +-pi`,
/// `atan2(+-inf, +-inf) = +-pi/4` or `+-3pi/4`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    elementary::atan2(y, x)
}

// ------------------------------------------------------------------- lookup and maps

/// Every unary kernel, by operator token.
macro_rules! unary_table {
    ($m:ident) => {
        $m!(
            "neg" => neg, "inv" => inv, "abs" => abs, "sin" => sin, "cos" => cos, "tan" => tan,
            "asin" => asin, "acos" => acos, "atan" => atan, "sinh" => sinh, "cosh" => cosh,
            "tanh" => tanh, "asinh" => asinh, "acosh" => acosh, "atanh" => atanh, "exp" => exp,
            "log" => log, "floor" => floor, "ceil" => ceil, "harmonic" => harmonic, "erf" => erf
        )
    };
}

/// Every binary kernel, by operator token.
macro_rules! binary_table {
    ($m:ident) => {
        $m!(
            "+" => add, "-" => sub, "*" => mul, "/" => div, "pow" => pow, "rootn" => rootn,
            "max" => max, "min" => min, "mod" => modulo, "atan2" => atan2,
            "besselj" => besselj
        )
    };
}

/// The unary kernel of an operator token, `None` for any other token.
pub fn unary(name: &str) -> Option<fn(f64) -> f64> {
    macro_rules! lookup {
        ($($tok:literal => $f:ident),*) => {
            match name { $($tok => Some($f as fn(f64) -> f64),)* _ => None }
        };
    }
    unary_table!(lookup)
}

/// The binary kernel of an operator token, `None` for any other token.
pub fn binary(name: &str) -> Option<fn(f64, f64) -> f64> {
    macro_rules! lookup {
        ($($tok:literal => $f:ident),*) => {
            match name { $($tok => Some($f as fn(f64, f64) -> f64),)* _ => None }
        };
    }
    binary_table!(lookup)
}

/// Why an element-wise map refused its operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// No kernel of this arity has that name.
    UnknownKernel(String),
    /// An operand's length is neither the output's nor 1.
    Length {
        operand: usize,
        len: usize,
        out: usize,
    },
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::UnknownKernel(name) => write!(f, "no kernel named {name:?} of this arity"),
            MapError::Length { operand, len, out } => write!(
                f,
                "operand {operand} has {len} elements, but the output has {out} (operands must match it or have one element)"
            ),
        }
    }
}

impl std::error::Error for MapError {}

/// Below this many elements a map runs on the calling thread.
const PARALLEL_MIN: usize = 1 << 15;

/// The threads a large map is split over: `SIMPLIPY_KERNEL_THREADS` when set to a positive
/// integer (1 keeps every map on the calling thread), else the available parallelism. The
/// threads are scoped to the call (no pool survives it), so a process that forks after a map
/// inherits no thread state.
fn kernel_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("SIMPLIPY_KERNEL_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    })
}

/// Split `out` into one contiguous piece per thread and run `body(offset, piece)` on each.
fn split_run<B: Fn(usize, &mut [f64]) + Sync>(out: &mut [f64], body: B) {
    let threads = kernel_threads()
        .min(out.len().div_ceil(PARALLEL_MIN / 2))
        .max(1);
    if threads == 1 || out.len() < PARALLEL_MIN {
        body(0, out);
        return;
    }
    let chunk = out.len().div_ceil(threads);
    std::thread::scope(|s| {
        let body = &body;
        for (i, piece) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || body(i * chunk, piece));
        }
    });
}

#[inline]
fn map1<F: Fn(f64) -> f64 + Sync>(f: F, x: &[f64], out: &mut [f64]) {
    if x.len() == 1 {
        out.fill(f(x[0]));
        return;
    }
    split_run(out, |off, piece| {
        let n = piece.len();
        for (o, &v) in piece.iter_mut().zip(&x[off..off + n]) {
            *o = f(v);
        }
    });
}

#[inline]
fn map2<F: Fn(f64, f64) -> f64 + Sync>(f: F, a: &[f64], b: &[f64], out: &mut [f64]) {
    match (a.len() == 1, b.len() == 1) {
        (true, true) => out.fill(f(a[0], b[0])),
        (true, false) => split_run(out, |off, piece| {
            let n = piece.len();
            for (o, &v) in piece.iter_mut().zip(&b[off..off + n]) {
                *o = f(a[0], v);
            }
        }),
        (false, true) => split_run(out, |off, piece| {
            let n = piece.len();
            for (o, &u) in piece.iter_mut().zip(&a[off..off + n]) {
                *o = f(u, b[0]);
            }
        }),
        (false, false) => split_run(out, |off, piece| {
            let n = piece.len();
            for ((o, &u), &v) in piece.iter_mut().zip(&a[off..off + n]).zip(&b[off..off + n]) {
                *o = f(u, v);
            }
        }),
    }
}

fn check_len(operand: usize, len: usize, out: usize) -> Result<(), MapError> {
    if len == out || len == 1 {
        Ok(())
    } else {
        Err(MapError::Length { operand, len, out })
    }
}

/// `out[i] = name(x[i])`, with a one-element `x` broadcast. Each kernel is its own
/// monomorphized loop (no per-element dispatch); large maps are split over threads.
pub fn map_unary(name: &str, x: &[f64], out: &mut [f64]) -> Result<(), MapError> {
    check_len(0, x.len(), out.len())?;
    if out.is_empty() {
        return unary(name)
            .map(|_| ())
            .ok_or_else(|| MapError::UnknownKernel(name.to_string()));
    }
    #[cfg(target_arch = "x86_64")]
    if matches!(name, "floor" | "ceil") && x.len() == out.len() && rounding::available() {
        let floor = name == "floor";
        split_run(out, |off, piece| {
            let n = piece.len();
            // SAFETY: the CPU has SSE4.1 (checked above).
            unsafe { rounding::round(floor, &x[off..off + n], piece) }
        });
        return Ok(());
    }
    macro_rules! run {
        ($($tok:literal => $f:ident),*) => {
            match name {
                $($tok => map1($f, x, out),)*
                _ => return Err(MapError::UnknownKernel(name.to_string())),
            }
        };
    }
    unary_table!(run);
    Ok(())
}

/// floor and ceil through SSE4.1's rounding instruction (`roundpd`), which a baseline x86-64
/// build cannot assume: there `f64::floor` is a libm call per element. Both are IEEE's exact
/// roundToIntegral, so the bits are the scalar kernels'.
#[cfg(target_arch = "x86_64")]
mod rounding {
    pub(super) fn available() -> bool {
        std::is_x86_feature_detected!("sse4.1")
    }

    /// SAFETY: the CPU must have SSE4.1.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn round(floor: bool, x: &[f64], out: &mut [f64]) {
        if floor {
            for (o, &v) in out.iter_mut().zip(x) {
                *o = v.floor();
            }
        } else {
            for (o, &v) in out.iter_mut().zip(x) {
                *o = v.ceil();
            }
        }
    }
}

/// `out[i] = name(a[i], b[i])`, with one-element operands broadcast.
pub fn map_binary(name: &str, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), MapError> {
    check_len(0, a.len(), out.len())?;
    check_len(1, b.len(), out.len())?;
    if out.is_empty() {
        return binary(name)
            .map(|_| ())
            .ok_or_else(|| MapError::UnknownKernel(name.to_string()));
    }
    macro_rules! run {
        ($($tok:literal => $f:ident),*) => {
            match name {
                $($tok => map2($f, a, b, out),)*
                _ => return Err(MapError::UnknownKernel(name.to_string())),
            }
        };
    }
    binary_table!(run);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    fn grid() -> Vec<f64> {
        let mut v: Vec<f64> = (-4000..=4000).map(|i| i as f64 / 512.0).collect();
        v.extend((-300..300).map(|e| 10f64.powi(e / 3) * if e % 2 == 0 { 1.0 } else { -1.0 }));
        v.extend([
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            5e-324,
            -5e-324,
            f64::MAX,
            f64::MIN,
            4503599627370495.5,
            -4503599627370495.5,
            4503599627370496.0,
            0.49999999999999994,
            -0.49999999999999994,
        ]);
        v
    }

    #[test]
    fn floor_and_ceil_cells() {
        for (x, fl, ce) in [
            (2.5, 2.0, 3.0),
            (-2.5, -3.0, -2.0),
            (-0.5, -1.0, -0.0),
            (0.5, 0.0, 1.0),
            (-0.0, -0.0, -0.0),
            (0.0, 0.0, 0.0),
            (f64::INFINITY, f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
            (4503599627370495.5, 4503599627370495.0, 4503599627370496.0),
            (
                -4503599627370495.5,
                -4503599627370496.0,
                -4503599627370495.0,
            ),
            (5e-324, 0.0, 1.0),
            (-5e-324, -1.0, -0.0),
        ] {
            assert_eq!(bits(floor(x)), bits(fl), "floor({x})");
            assert_eq!(bits(ceil(x)), bits(ce), "ceil({x})");
        }
        assert!(floor(f64::NAN).is_nan() && ceil(f64::NAN).is_nan());
    }

    /// The maps' floor and ceil (the rounding instruction where the CPU has it) are the scalar
    /// kernels bit for bit, and dual to each other, on every cell of the grid; max and min,
    /// written as selects, are IEEE 754-2019's on every pair.
    #[test]
    fn maps_equal_the_scalar_kernels() {
        let g = grid();
        let mut out = vec![0.0; g.len()];
        for (name, f) in [("floor", floor as fn(f64) -> f64), ("ceil", ceil)] {
            map_unary(name, &g, &mut out).unwrap();
            for (&x, &o) in g.iter().zip(&out) {
                assert!(
                    bits(o) == bits(f(x)) || (o.is_nan() && x.is_nan()),
                    "{name}({x})"
                );
            }
        }
        for &x in &g {
            if !x.is_nan() {
                assert_eq!(bits(ceil(x)), bits(-floor(-x)), "ceil({x}) = -floor(-{x})");
            }
        }
        for &a in g.iter().step_by(13) {
            for &b in g.iter().step_by(17) {
                let (mx, mn) = (max(a, b), min(a, b));
                if a.is_nan() || b.is_nan() {
                    assert!(mx.is_nan() && mn.is_nan());
                    continue;
                }
                let (hi, lo) = if a > b || (a == b && a.is_sign_positive()) {
                    (a, b)
                } else {
                    (b, a)
                };
                assert_eq!(bits(mx), bits(hi), "max({a}, {b})");
                assert_eq!(bits(mn), bits(lo), "min({a}, {b})");
                assert_eq!(bits(mx), bits(max(b, a)));
                assert_eq!(bits(mn), bits(min(b, a)));
            }
        }
    }

    /// IEEE 754-2019 at the zero tie, in both orders; NaN from either side.
    #[test]
    fn max_min_are_ieee_and_commutative_at_zero() {
        let (p, n) = (0.0f64, -0.0f64);
        assert_eq!(bits(max(n, p)), bits(p));
        assert_eq!(bits(max(p, n)), bits(p));
        assert_eq!(bits(max(n, n)), bits(n));
        assert_eq!(bits(max(p, p)), bits(p));
        assert_eq!(bits(min(n, p)), bits(n));
        assert_eq!(bits(min(p, n)), bits(n));
        assert_eq!(bits(min(p, p)), bits(p));
        assert_eq!(bits(min(n, n)), bits(n));
        for (a, b) in [(f64::NAN, 1.0), (1.0, f64::NAN), (f64::INFINITY, f64::NAN)] {
            assert!(max(a, b).is_nan() && min(a, b).is_nan());
        }
        assert_eq!(max(-1.0, f64::NEG_INFINITY), -1.0);
        assert_eq!(min(-1.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(max(3.0, 3.0), 3.0);
        assert_eq!(min(-2.0, -2.0), -2.0);
    }

    /// numpy's `np.mod` cells (measured on numpy 2.5.4).
    #[test]
    fn mod_cells_follow_numpy() {
        let inf = f64::INFINITY;
        for (a, b, want) in [
            (-7.0, 3.0, 2.0),
            (7.0, -3.0, -2.0),
            (7.0, 3.0, 1.0),
            (3.0, inf, 3.0),
            (-3.0, inf, inf),
            (3.0, -inf, -inf),
            (-3.0, -inf, -3.0),
            (-0.0, 3.0, 0.0),
            (0.0, -3.0, -0.0),
            (6.0, 3.0, 0.0),
            (-6.0, 3.0, 0.0),
            (6.0, -3.0, -0.0),
            (-1e-20, 3.0, 3.0),
        ] {
            assert_eq!(bits(modulo(a, b)), bits(want), "mod({a}, {b})");
        }
        for (a, b) in [
            (5.0, 0.0),
            (5.0, -0.0),
            (inf, 2.0),
            (-inf, 2.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
        ] {
            assert!(modulo(a, b).is_nan(), "mod({a}, {b})");
        }
    }

    /// atan2 keeps C99's cells.
    #[test]
    fn atan2_cells() {
        use std::f64::consts::{FRAC_PI_4, PI};
        let inf = f64::INFINITY;
        for (y, x, want) in [
            (0.0, 0.0, 0.0),
            (-0.0, 0.0, -0.0),
            (0.0, -0.0, PI),
            (-0.0, -0.0, -PI),
            (1.0, -inf, PI),
            (-1.0, -inf, -PI),
            (inf, inf, FRAC_PI_4),
            (-inf, -inf, -3.0 * FRAC_PI_4),
        ] {
            assert_eq!(bits(atan2(y, x)), bits(want), "atan2({y}, {x})");
        }
    }

    /// The evaluator resolves every existing operator to the deployed kernel: one function.
    #[test]
    fn the_evaluator_and_the_kernels_are_one_function() {
        let unary = [
            "neg", "inv", "abs", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
            "tanh", "asinh", "acosh", "atanh", "exp", "log", "floor", "ceil", "harmonic", "erf",
        ];
        let g = grid();
        for name in unary {
            let (e, k) = (
                crate::numeric::unary_fn(name).unwrap(),
                super::unary(name).unwrap(),
            );
            for &x in &g {
                assert!(bits(e(x)) == bits(k(x)), "{name}({x})");
            }
        }
        for name in [
            "+", "-", "*", "/", "pow", "rootn", "max", "min", "mod", "atan2", "besselj",
        ] {
            let (e, k) = (
                crate::numeric::binary_fn(name).unwrap(),
                super::binary(name).unwrap(),
            );
            for &a in g.iter().step_by(37) {
                for &b in g.iter().step_by(41) {
                    let (u, v) = (e(a, b), k(a, b));
                    assert!(bits(u) == bits(v) || (u.is_nan() && v.is_nan()), "{name}");
                }
            }
        }
    }

    #[test]
    fn maps_broadcast_one_element_operands_and_refuse_the_rest() {
        let mut out = [0.0; 3];
        map_binary("max", &[1.0, -5.0, 2.0], &[0.0], &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 2.0]);
        map_unary("floor", &[2.5], &mut out).unwrap();
        assert_eq!(out, [2.0, 2.0, 2.0]);
        assert_eq!(
            map_unary("besselj", &[1.0, 2.0, 3.0], &mut out),
            Err(MapError::UnknownKernel("besselj".into()))
        );
        assert_eq!(
            map_binary("mod", &[1.0, 2.0], &[1.0], &mut out),
            Err(MapError::Length {
                operand: 0,
                len: 2,
                out: 3
            })
        );
        assert!(map_unary("pow", &[], &mut []).is_err());
        assert!(map_unary("ceil", &[], &mut []).is_ok());
    }

    /// A map split over threads computes what the serial loop computes, at every offset.
    #[test]
    fn threaded_maps_equal_the_serial_loop() {
        let n = 3 * PARALLEL_MIN + 17;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 40.0).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * -0.0007 + 3.0).collect();
        let mut out = vec![0.0; n];
        map_unary("tanh", &x, &mut out).unwrap();
        assert!(out.iter().zip(&x).all(|(&o, &v)| bits(o) == bits(tanh(v))));
        map_binary("atan2", &x, &y, &mut out).unwrap();
        assert!((0..n).all(|i| bits(out[i]) == bits(atan2(x[i], y[i]))));
        map_binary("-", &[1.5], &y, &mut out).unwrap();
        assert!((0..n).all(|i| bits(out[i]) == bits(1.5 - y[i])));
    }
}
