//! The float64 kernels of the operators added in 0.15, behind a plain Rust API: scalar
//! functions, a lookup by operator token, and element-wise maps over slices. Nothing here
//! knows about Python; the `_core` extension wraps the maps (`kernel_unary`/`kernel_binary`),
//! and another frontend can call the same functions.
//!
//! Each kernel either computes exactly what the deployed realization in `simplipy.operators`
//! computes (`floor`, `ceil`, `max`, `min`, `mod`: numpy's own operations, pinned bit for bit
//! by the realization gates) or IS the deployed realization (`harmonic`, which
//! `simplipy.operators.harmonic` calls through `kernel_unary`).
//!
//! These kernels are deliberately NOT in the evaluator's dispatch (`numeric::unary_fn` /
//! `binary_fn`): an operator enters the fold, the tape and the miner together with its
//! interval, high-precision and judge arms, so until those exist every engine path refuses
//! the new heads, as it does today.

pub use super::harmonic::harmonic;

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
#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a > b {
        a
    } else if b > a {
        b
    } else if a == 0.0 {
        a + b // a zero tie: +0 unless both are -0
    } else {
        a
    }
}

/// IEEE 754-2019 `minimum`: NaN-propagating, -0 < +0, commutative bit for bit.
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a < b {
        a
    } else if b < a {
        b
    } else if a == 0.0 {
        -((-a) + (-b)) // a zero tie: -0 unless both are +0
    } else {
        a
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

/// The unary kernel of an operator token, `None` for any other token.
pub fn unary(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "floor" => floor,
        "ceil" => ceil,
        "harmonic" => harmonic,
        _ => return None,
    })
}

/// The binary kernel of an operator token, `None` for any other token.
pub fn binary(name: &str) -> Option<fn(f64, f64) -> f64> {
    Some(match name {
        "max" => max,
        "min" => min,
        "mod" => modulo,
        _ => return None,
    })
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

#[inline]
fn map1<F: Fn(f64) -> f64>(f: F, x: &[f64], out: &mut [f64]) {
    if x.len() == 1 {
        out.fill(f(x[0]));
    } else {
        for (o, &v) in out.iter_mut().zip(x) {
            *o = f(v);
        }
    }
}

#[inline]
fn map2<F: Fn(f64, f64) -> f64>(f: F, a: &[f64], b: &[f64], out: &mut [f64]) {
    match (a.len() == 1, b.len() == 1) {
        (true, true) => out.fill(f(a[0], b[0])),
        (true, false) => {
            for (o, &v) in out.iter_mut().zip(b) {
                *o = f(a[0], v);
            }
        }
        (false, true) => {
            for (o, &v) in out.iter_mut().zip(a) {
                *o = f(v, b[0]);
            }
        }
        (false, false) => {
            for ((o, &u), &v) in out.iter_mut().zip(a).zip(b) {
                *o = f(u, v);
            }
        }
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
/// monomorphized loop (no per-element dispatch).
pub fn map_unary(name: &str, x: &[f64], out: &mut [f64]) -> Result<(), MapError> {
    check_len(0, x.len(), out.len())?;
    if out.is_empty() {
        return unary(name)
            .map(|_| ())
            .ok_or_else(|| MapError::UnknownKernel(name.to_string()));
    }
    match name {
        "floor" => map1(floor, x, out),
        "ceil" => map1(ceil, x, out),
        "harmonic" => map1(harmonic, x, out),
        _ => return Err(MapError::UnknownKernel(name.to_string())),
    }
    Ok(())
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
    match name {
        "max" => map2(max, a, b, out),
        "min" => map2(min, a, b, out),
        "mod" => map2(modulo, a, b, out),
        _ => return Err(MapError::UnknownKernel(name.to_string())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f64) -> u64 {
        x.to_bits()
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
        ] {
            assert_eq!(bits(floor(x)), bits(fl), "floor({x})");
            assert_eq!(bits(ceil(x)), bits(ce), "ceil({x})");
            // the duality is exact, signed zeros included
            assert_eq!(bits(ceil(x)), bits(-floor(-x)), "ceil({x}) = -floor(-{x})");
        }
        assert!(floor(f64::NAN).is_nan() && ceil(f64::NAN).is_nan());
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

    #[test]
    fn maps_broadcast_one_element_operands_and_refuse_the_rest() {
        let mut out = [0.0; 3];
        map_binary("max", &[1.0, -5.0, 2.0], &[0.0], &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 2.0]);
        map_unary("floor", &[2.5], &mut out).unwrap();
        assert_eq!(out, [2.0, 2.0, 2.0]);
        assert_eq!(
            map_unary("erf", &[1.0, 2.0, 3.0], &mut out),
            Err(MapError::UnknownKernel("erf".into()))
        );
        assert_eq!(
            map_binary("mod", &[1.0, 2.0], &[1.0], &mut out),
            Err(MapError::Length {
                operand: 0,
                len: 2,
                out: 3
            })
        );
        assert!(map_unary("sin", &[], &mut []).is_err());
        assert!(map_unary("ceil", &[], &mut []).is_ok());
        for name in ["floor", "ceil", "harmonic"] {
            assert!(unary(name).is_some() && binary(name).is_none());
        }
        for name in ["max", "min", "mod"] {
            assert!(binary(name).is_some() && unary(name).is_none());
        }
    }

    /// The new heads stay out of the evaluator until their interval, high-precision and judge
    /// arms exist: the fold, the tape and the miner refuse them (fail closed).
    #[test]
    fn new_heads_are_not_in_the_evaluator_dispatch() {
        for name in ["floor", "ceil", "harmonic"] {
            assert!(crate::numeric::unary_fn(name).is_none(), "{name}");
        }
        for name in ["max", "min", "mod"] {
            assert!(crate::numeric::binary_fn(name).is_none(), "{name}");
        }
    }
}
