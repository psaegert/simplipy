//! THE COMPLEX PRODUCT AND QUOTIENT OF THE CORE. The Bessel kernels (`xsf`) port C++ whose
//! complex `*` and `/` a C compiler leaves to its runtime library, and runtimes differ between
//! machines in the last bits of a quotient. The core computes both here, in plain IEEE double
//! arithmetic (no fused multiply-add, no call into the platform), so they give the same bits on
//! every machine.
//!
//! * The product is `(ac - bd) + (ad + bc)i`; where both parts come out NaN, the infinities that
//!   a NaN + NaN hides are recovered by the informative algorithm of C99 Annex G (G.5.1): an
//!   infinite factor is "boxed" to its signed unit parts, NaN parts of the other factor read as
//!   signed zeros, and an overflowed partial product does the same for every NaN part.
//! * The quotient is Smith's algorithm (1962), with the scaling of Baudin and Smith, "A Robust
//!   Complex Division in Scilab" (2012): divide through by the larger divisor part, scale all
//!   four parts by one power of two where the divisor or the dividend lies near the ends of the
//!   double range, and when the ratio of the divisor parts falls below the normal range, divide
//!   before multiplying. This is the algorithm of GCC's runtime since GCC 12 (`__divdc3`). Where
//!   both parts come out NaN, Annex G's recovery of the infinities and zeros: a non-NaN dividend
//!   over a zero divisor, an infinite dividend over a finite divisor, a finite dividend over an
//!   infinite divisor.
//!
//! The papers leave the thresholds and the order of the operations open; the choices below are
//! named where they are made, and they reproduce GCC's runtime: against `__divdc3` and
//! `__muldc3` of GCC's runtime on Linux, on 29 million operand sets per function drawn to reach
//! every branch (ordinary, subnormal and huge parts, zero divisors, infinite and NaN parts, both
//! orientations, both orders of the scaled step, every recovery), no value differs. Where an
//! operand is NaN, the NaN of the result may carry another operand's sign and payload (x86
//! passes on an instruction's first NaN operand, and compilers order the operands of a sum or a
//! product differently); a NaN's sign and payload are no value of the package.

/// Halve all four parts when the larger divisor part reaches half the largest double, so that
/// `c r + d` cannot overflow (a choice: Baudin and Smith halve the dividend and the divisor
/// separately and correct the quotient by the scale; halving all four leaves it unchanged).
const HALVE_AT: f64 = f64::MAX / 2.0;
/// Scale all four parts up when the larger divisor part lies below the double's epsilon, 2^-52:
/// the products and the sum in `c r + d` then cannot underflow.
const SCALE_UP_BELOW: f64 = f64::EPSILON;
/// The upward scale, 2^52: exact, and it cannot overflow a divisor part below 2^-52.
const UP: f64 = 1.0 / f64::EPSILON;
/// A dividend part below the smallest normal double is scaled up as well (where nothing would
/// overflow: every other part below `UP_LIMIT`), so a subnormal part keeps its bits through the
/// products; and when the ratio of the divisor parts is below the smallest normal double, the
/// dividend is divided by the larger divisor part before it is multiplied by the smaller one.
const NORMAL_MIN: f64 = f64::MIN_POSITIVE;
/// `HALVE_AT / UP`: a part below it can be scaled up by `UP` without overflow.
const UP_LIMIT: f64 = HALVE_AT * f64::EPSILON;

/// A part of an infinite operand "boxed" (Annex G): its sign on 1 if it is infinite, on 0 if not.
#[inline]
fn boxed(v: f64) -> f64 {
    (if v.is_infinite() { 1.0_f64 } else { 0.0 }).copysign(v)
}

/// A NaN part read as a zero of its sign (Annex G); any other part as it is.
#[inline]
fn nan_as_zero(v: f64) -> f64 {
    if v.is_nan() {
        0.0_f64.copysign(v)
    } else {
        v
    }
}

/// `(a + bi)(c + di)`, as `(real, imaginary)`.
pub(crate) fn product(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let (ac, bd, ad, bc) = (a * c, b * d, a * d, b * c);
    let (re, im) = (ac - bd, ad + bc);
    if !(re.is_nan() && im.is_nan()) {
        return (re, im);
    }
    let (mut a, mut b, mut c, mut d) = (a, b, c, d);
    let mut recompute = false;
    if a.is_infinite() || b.is_infinite() {
        // the first factor is infinite
        (a, b) = (boxed(a), boxed(b));
        (c, d) = (nan_as_zero(c), nan_as_zero(d));
        recompute = true;
    }
    if c.is_infinite() || d.is_infinite() {
        // the second factor is infinite
        (c, d) = (boxed(c), boxed(d));
        (a, b) = (nan_as_zero(a), nan_as_zero(b));
        recompute = true;
    }
    if !recompute && (ac.is_infinite() || bd.is_infinite() || ad.is_infinite() || bc.is_infinite())
    {
        // a partial product overflowed
        (a, b, c, d) = (
            nan_as_zero(a),
            nan_as_zero(b),
            nan_as_zero(c),
            nan_as_zero(d),
        );
        recompute = true;
    }
    if recompute {
        (
            f64::INFINITY * (a * c - b * d),
            f64::INFINITY * (a * d + b * c),
        )
    } else {
        (re, im)
    }
}

/// `(a + bi) / (c + di)`, as `(real, imaginary)`.
pub(crate) fn quotient(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    // Divide through by the larger divisor part: `d` when |c| < |d|, else `c` (also where a
    // divisor part is NaN, which no comparison orders).
    let through_d = c.abs() < d.abs();
    let larger = if through_d { d.abs() } else { c.abs() };
    // One power of two scales all four parts, which leaves the quotient as it is.
    let scale = if larger >= HALVE_AT {
        0.5
    } else if larger < SCALE_UP_BELOW
        || (larger < UP_LIMIT
            && ((a.abs() < NORMAL_MIN && b.abs() < UP_LIMIT)
                || (b.abs() < NORMAL_MIN && a.abs() < UP_LIMIT)))
    {
        UP
    } else {
        1.0
    };
    let (a, b, c, d) = if scale == 1.0 {
        (a, b, c, d)
    } else {
        (a * scale, b * scale, c * scale, d * scale)
    };
    // Smith's step, each part divided by the scaled denominator `den` (a choice: Baudin and
    // Smith multiply by its reciprocal, which rounds once more). The ratio `r` of the smaller
    // divisor part to the larger is at most 1 in magnitude; below the smallest normal double it
    // has lost bits, and the dividend is divided by the larger part first.
    let (re, im) = if through_d {
        let r = c / d;
        let den = c * r + d;
        if r.abs() > NORMAL_MIN {
            ((a * r + b) / den, (b * r - a) / den)
        } else {
            ((c * (a / d) + b) / den, (c * (b / d) - a) / den)
        }
    } else {
        let r = d / c;
        let den = d * r + c;
        if r.abs() > NORMAL_MIN {
            ((b * r + a) / den, (b - a * r) / den)
        } else {
            ((d * (b / c) + a) / den, (b - d * (a / c)) / den)
        }
    };
    if !(re.is_nan() && im.is_nan()) {
        return (re, im);
    }
    // Annex G's recovery, on the scaled parts.
    if c == 0.0 && d == 0.0 && (!a.is_nan() || !b.is_nan()) {
        // a non-NaN dividend over a zero divisor: infinities, signed by the divisor's real zero
        let inf = f64::INFINITY.copysign(c);
        return (inf * a, inf * b);
    }
    if (a.is_infinite() || b.is_infinite()) && c.is_finite() && d.is_finite() {
        // an infinite dividend over a finite divisor
        let (a, b) = (boxed(a), boxed(b));
        return (
            f64::INFINITY * (a * c + b * d),
            f64::INFINITY * (b * c - a * d),
        );
    }
    if (c.is_infinite() || d.is_infinite()) && a.is_finite() && b.is_finite() {
        // a finite dividend over an infinite divisor
        let (c, d) = (boxed(c), boxed(d));
        return (0.0 * (a * c + b * d), 0.0 * (b * c - a * d));
    }
    (re, im)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits((re, im): (f64, f64)) -> (u64, u64) {
        let canon = |v: f64| {
            if v.is_nan() {
                f64::NAN.to_bits()
            } else {
                v.to_bits()
            }
        };
        (canon(re), canon(im))
    }

    fn h(s: &str) -> f64 {
        match s {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            _ => s.parse().unwrap(),
        }
    }

    /// The quotient and the product are GCC's runtime's on every machine: each case's result is
    /// the one GCC's runtime (`__divdc3`, `__muldc3`) gives on Linux, and each reaches a named
    /// branch.
    #[test]
    fn complex_quotient_and_product_are_gccs_algorithm() {
        let quotients: &[([&str; 4], [&str; 2])] = &[
            // through d (|c| < |d|), Smith's step
            (["1", "2", "3", "4"], ["0.44", "0.08"]),
            (
                ["1", "2", "3", "5"],
                ["0.3823529411764706", "0.029411764705882346"],
            ),
            // the ratio below the normal range: divide first, through d and through c
            (
                ["2", "0", "0", "-27.939446353542056"],
                ["-0", "0.07158337980975946"],
            ),
            (["1", "2", "1e-300", "1e10"], ["2e-10", "-1e-10"]),
            (["3", "-7", "1e10", "1e-300"], ["3e-10", "-7e-10"]),
            (["3", "4", "2", "0"], ["1.5", "2"]),
            // a zero divisor
            (["308.3", "0", "0", "-0"], ["inf", "nan"]),
            (["nan", "2", "0", "0"], ["nan", "inf"]),
            // scaled up: the larger divisor part below 2^-52
            (
                ["1e-310", "3", "1e-300", "7e-301"],
                ["1.4093959731543623e300", "2.0134228187919463e300"],
            ),
            (["0.1", "0.2", "1e-320", "3e-321"], ["inf", "inf"]),
            (
                ["1e-200", "1e-200", "1e-200", "3e-200"],
                ["0.39999999999999997", "-0.20000000000000004"],
            ),
            // scaled up: a subnormal dividend part
            (["1e-310", "1e-311", "3", "4"], ["1.36e-311", "-1.48e-311"]),
            // halved: the larger divisor part at half the largest double
            (
                ["1.5e308", "-1e308", "1e308", "1.7e308"],
                ["-0.05141388174807196", "-0.9125964010282777"],
            ),
            // an infinite dividend over a finite divisor
            (["inf", "1", "2", "3"], ["inf", "-inf"]),
            (["-inf", "nan", "5", "1"], ["-inf", "inf"]),
            // a finite dividend over an infinite divisor
            (["1", "1", "inf", "nan"], ["0", "0"]),
            (["1.5", "-2.5", "inf", "-inf"], ["0", "-0"]),
        ];
        for ([a, b, c, d], [re, im]) in quotients {
            let (a, b, c, d) = (h(a), h(b), h(c), h(d));
            assert_eq!(
                bits(quotient(a, b, c, d)),
                bits((h(re), h(im))),
                "{a} {b} {c} {d}"
            );
        }
        let products: &[([&str; 4], [&str; 2])] = &[
            // one part NaN: nothing to recover
            (["inf", "0", "0", "1"], ["nan", "inf"]),
            // the infinities a NaN + NaN hides: an infinite first factor, an infinite second
            // factor, an overflowed partial product
            (["inf", "nan", "1", "0"], ["inf", "nan"]),
            (["nan", "2", "-inf", "3"], ["nan", "-inf"]),
            (["1e300", "nan", "1e300", "0"], ["inf", "nan"]),
        ];
        for ([a, b, c, d], [re, im]) in products {
            let (a, b, c, d) = (h(a), h(b), h(c), h(d));
            assert_eq!(
                bits(product(a, b, c, d)),
                bits((h(re), h(im))),
                "{a} {b} {c} {d}"
            );
        }
    }
}
