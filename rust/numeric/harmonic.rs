//! The harmonic number H(x) = psi(x + 1) + gamma, on the whole real line.
//!
//! H is analytic off its poles at the negative integers, increasing between them, with
//! H(0) = 0, H(n) = 1 + 1/2 + ... + 1/n and H(x) - H(x - 1) = 1/x. The literal spelling
//! `digamma(x + 1) + euler_gamma` loses every digit near 0, where psi(1) = -gamma cancels the
//! constant (it is exactly 0 on |x| < 1e-17), so this kernel never forms that sum: every
//! branch carries the factor its zero needs, and the leading terms are carried as
//! double-doubles, so on x > -1 a result is within 0.53 ulp of the true value (measured
//! against mpmath on 210,000 points; every harmonic number H(1)..H(1000) correctly rounded).
//!
//! * x >= -1/2: on [-1/2, 1/2), H(t) = t (1/(1+t) + D_1(t)); on [1/2, K + 1/2),
//!   H(k + t) = H(k) + t D_k(t) about the nearest integer k, with |t| <= 1/2; beyond, the
//!   asymptotic series ln x + gamma + 1/(2x) - sum B_2j / (2j x^2j). D_k is a Chebyshev fit
//!   of (H(k + t) - H(k)) / t (`harmonic_tables.py` generates the tables).
//! * -1 < x < -1/2: H(x) = H(x + 1) - 1/(x + 1), x + 1 exact.
//! * x < -1: the reflection H(x) = H(-x - 1) - pi cot(pi x), -x - 1 and the reduced argument
//!   exact. Between two poles H crosses zero once, where the two terms cancel, so there the
//!   bound is absolute: |error| <= ulp(H) + 2^-58 max(|H(-x - 1)|, |pi cot(pi x)|), which is
//!   within one ulp wherever the terms cancel by at most a factor 16.
//!
//! The cells: H(+-0) = +-0 (H is 0 at 0 with a positive slope, so it keeps the sign of zero
//! as sin, atan and erf do); H(+inf) = +inf; H(-inf) = NaN; NaN at every negative integer
//! (the one-sided limits there are +inf and -inf); NaN at NaN.
//!
//! The kernel calls no libm function: its elementary ingredients (`elementary::ln` and
//! `elementary::pi_cot_pi`) are double-double series of its own, so its bits are the same on
//! every platform.

use super::dd::Dd;
use super::harmonic_tables as tab;

/// H(x) = psi(x + 1) + gamma. See the module documentation for the method and the cells.
pub fn harmonic(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return if x > 0.0 { x } else { f64::NAN };
    }
    if x == 0.0 {
        return x;
    }
    let h = if x >= -0.5 {
        principal(x)
    } else if x > -1.0 {
        let y = x + 1.0; // exact (Sterbenz)
        principal(y).add(Dd::recip_f64(y).neg())
    } else {
        // Every f64 at or below -2^52 is a negative integer; above it, x = n + f exactly with
        // n = trunc(x) (the conversion truncates toward zero) and -1 < f <= 0.
        if x <= -TWO_POW_52 {
            return f64::NAN;
        }
        let f = x - (x as i64) as f64;
        if f == 0.0 {
            return f64::NAN; // a pole
        }
        // r: x reduced to [-1/2, 1/2] by an integer (f + 1 exact by Sterbenz); -x - 1 exact
        let r = if f < -0.5 { f + 1.0 } else { f };
        principal(-x - 1.0).add(elementary::pi_cot_pi(r).neg())
    };
    h.to_f64()
}

/// A polynomial in f64 by Estrin's scheme, constant coefficient first: pairs, then pairs of
/// pairs, so the dependent chain is log2(N) steps long instead of N.
#[inline(always)]
fn estrin<const N: usize>(c: &[f64; N], t: f64) -> f64 {
    let mut a = *c;
    let mut len = N;
    let mut p = t;
    while len > 1 {
        let half = len.div_ceil(2);
        for i in 0..half {
            let hi = if 2 * i + 1 < len { a[2 * i + 1] } else { 0.0 };
            a[i] = a[2 * i] + hi * p;
        }
        len = half;
        p *= p;
    }
    a[0]
}

/// H(x) for x >= -1/2 as a double-double.
fn principal(x: f64) -> Dd {
    if x < 0.5 {
        // H(t) = t (1/(1+t) + D_1(t)), with D_1(t) = c0 + t (c1 + t P(t)): both terms carry
        // the sign of t, so the sum cannot cancel and H keeps its relative accuracy at 0.
        let w = tab::CENTRE_C1[0].add_f64(x * estrin(&tab::CENTRE_P[0], x));
        let d1 = tab::CENTRE_C0[0].add(w.mul_f64(x));
        let q = Dd::sum(1.0, x).recip();
        return q.add(d1).mul_f64(x);
    }
    if x < tab::K as f64 + 0.5 {
        // the nearest integer k (x + 1/2 is exact here, and the conversion truncates); t exact
        let k = (x + 0.5) as usize;
        let t = x - k as f64;
        let w = tab::CENTRE_C1[k - 1].add_f64(t * estrin(&tab::CENTRE_P[k - 1], t));
        let inner = tab::CENTRE_C0[k - 1].add(w.mul_f64(t));
        return tab::H_K[k].add(inner.mul_f64(t));
    }
    if x >= TWO_POW_60 {
        // Past 2^60 every term after 1/(2x) is below 2^-120 of the result.
        return elementary::ln(x).add(tab::GAMMA).add_f64(0.5 / x);
    }
    // ln x + gamma + 1/(2x) - (1/(12 x^2) + sum_{j >= 2} B_2j / (2j x^2j)); the bracket is
    // below 2^-9 of the result here, so f64 carries it to well under 2^-60.
    let inv = Dd::recip_f64(x);
    let w = inv.hi * inv.hi;
    let small = w * tab::INV12.hi + estrin(&tab::ASYM, w) * w * w;
    let s = tab::GAMMA
        .add(Dd::new(0.5 * inv.hi, 0.5 * inv.lo))
        .add_f64(-small);
    elementary::ln(x).add(s)
}

const TWO_POW_52: f64 = 4_503_599_627_370_496.0;
const TWO_POW_60: f64 = 1_152_921_504_606_846_976.0;

/// The kernel's elementary ingredients, in double-double and without libm. Kept apart so
/// their source is one function each.
mod elementary {
    use super::estrin;
    use super::tab;
    use crate::numeric::dd::Dd;

    /// ln x for a positive normal f64. x = f 2^e with f in [1, 2), and f = c_i (1 + d) at the
    /// nearest knot c_i = 1 + i/128, through the f64 r_i ~ 1/c_i: 1 + d = f r_i exactly (as a
    /// double-double), |d| <= 2^-8, and ln f = -ln r_i + ln(1 + d), the series in d past its
    /// first term below 2^-16 of the result. No division.
    pub(super) fn ln(x: f64) -> Dd {
        let bits = x.to_bits();
        let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
        let f = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
        let i = ((f - 1.0) * tab::LN_N as f64 + 0.5) as usize; // f - 1 exact; i in 0..=LN_N
        let fr = Dd::from_f64(f).mul_f64(tab::LN_RECIP[i]);
        let d = Dd::sum(fr.hi - 1.0, fr.lo); // fr.hi - 1 exact (Sterbenz)
        let tail = d.hi * d.hi * estrin(&tab::LN_TAIL, d.hi);
        tab::LN2
            .mul_f64(e as f64)
            .add(tab::LN_LOG[i])
            .add(d)
            .add_f64(tail)
    }

    /// pi cot(pi r) for 0 < |r| <= 1/2. Near the pole (|r| <= 1/4) as cos/sin of pi |r|,
    /// otherwise as sin/cos of pi (1/2 - |r|), so the series argument stays below pi/4.
    pub(super) fn pi_cot_pi(r: f64) -> Dd {
        let a = r.abs();
        let near_pole = a <= 0.25;
        let u = if near_pole { a } else { 0.5 - a }; // exact (Sterbenz)
        if u == 0.0 {
            return Dd::from_f64(0.0);
        }
        let p = tab::PI.mul_f64(u);
        let p2 = p.mul(p);
        let p3 = p2.mul(p);
        let p4 = p2.mul(p2);
        let sin_tail = estrin(&tab::SIN_TAIL, p2.hi) * p2.hi * p2.hi * p.hi;
        let sin = p.add(p3.mul(tab::INV6).neg()).add_f64(sin_tail);
        let cos_tail = estrin(&tab::COS_TAIL, p2.hi) * p2.hi * p4.hi;
        let cos = Dd::new(-0.5 * p2.hi, -0.5 * p2.lo)
            .add_f64(1.0)
            .add(p4.mul(tab::INV24))
            .add_f64(cos_tail);
        let q = tab::PI.mul(if near_pole {
            cos.div(sin)
        } else {
            sin.div(cos)
        });
        if r > 0.0 {
            q
        } else {
            q.neg()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::harmonic;

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    /// The cells: signed zeros kept, NaN at every negative integer and at -inf, +inf at +inf.
    #[test]
    fn cells() {
        assert_eq!(bits(harmonic(0.0)), bits(0.0));
        assert_eq!(bits(harmonic(-0.0)), bits(-0.0));
        assert_eq!(harmonic(f64::INFINITY), f64::INFINITY);
        assert!(harmonic(f64::NEG_INFINITY).is_nan());
        assert!(harmonic(f64::NAN).is_nan());
        for n in [-1.0, -2.0, -3.0, -17.0, -1e6, -4503599627370496.0, -1e300] {
            assert!(harmonic(n).is_nan(), "H({n}) must be NaN");
        }
    }

    /// The harmonic numbers, each the f64 nearest to its exact rational value.
    #[test]
    fn integers_are_the_harmonic_numbers() {
        assert_eq!(harmonic(1.0), 1.0);
        assert_eq!(harmonic(2.0), 1.5);
        assert_eq!(harmonic(3.0), 11.0 / 6.0);
        assert_eq!(harmonic(4.0), 25.0 / 12.0);
        assert_eq!(harmonic(10.0), 7381.0 / 2520.0);
        // H(20) = 55835135/15519504 (both exact in f64, so the quotient is correctly rounded)
        assert_eq!(harmonic(20.0), 55835135.0 / 15519504.0);
    }

    /// The zero at 0 keeps its relative accuracy: H(x) = zeta(2) x - zeta(3) x^2 + ...
    #[test]
    fn relative_accuracy_at_zero() {
        let zeta2 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        for x in [1e-300, -1e-300, 1e-17, -1e-17, 1e-10] {
            let h = harmonic(x);
            assert!(((h / x) - zeta2).abs() < 1e-9, "H({x}) = {h}");
        }
        // the smallest subnormal: zeta(2) * 2^-1074 rounds to 2 * 2^-1074
        assert_eq!(harmonic(5e-324), 1e-323);
    }

    /// The recurrence H(x) - H(x - 1) = 1/x and the reflection hold across every branch.
    #[test]
    fn recurrence_and_reflection() {
        for x in [-0.75, -0.3, 0.2, 0.7, 1.3, 7.9, 8.6, 30.25, 1e5 + 0.5] {
            let lhs = harmonic(x) - harmonic(x - 1.0);
            assert!(
                (lhs - 1.0 / x).abs() <= 4e-15 * harmonic(x).abs().max(1.0),
                "x = {x}"
            );
        }
        // H(-1/2) = -2 ln 2; H(-3/2) = H(1/2) = 2 - 2 ln 2 (pi cot(-3pi/2) = 0)
        let ln2 = std::f64::consts::LN_2;
        assert!((harmonic(-0.5) + 2.0 * ln2).abs() <= 2.3e-16);
        assert!((harmonic(0.5) - (2.0 - 2.0 * ln2)).abs() <= 1.2e-16);
        assert_eq!(bits(harmonic(-1.5)), bits(harmonic(0.5)));
    }

    /// Large arguments reach the asymptotic branch and its far end without overflow.
    #[test]
    fn large_arguments() {
        let h = harmonic(1e300);
        assert!((h - (1e300f64.ln() + 0.5772156649015329)).abs() < 1e-12);
        assert!(harmonic(f64::MAX).is_finite());
        assert!(harmonic(-4503599627370495.5).is_finite());
    }
}
