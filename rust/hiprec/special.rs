//! Arbitrary-precision kernels of the 0.15 heads, shared by both high-precision evaluators
//! (the rescue's IEEE-faithful `apply_unary`/`apply_binary` and the contract's
//! `ct_unary`/`ct_binary`, which add the contract's cells and boundary bands on top).
//!
//! * `floor`, `ceil`, `max`, `min` and `mod` are exact on the representation: astro-float's
//!   integer part and its remainder (`rem`, the truncated remainder) carry no rounding, so the
//!   floored remainder is one rounded addition of the divisor, numpy's own construction.
//! * `atan2` is `atan(y/x)`, moved by pi in the left half-plane, with C99's cells.
//! * `erf` is the everywhere-positive series
//!   `erf(a) = 2a/sqrt(pi) e^(-a^2) sum_n (2a^2)^n / (2n+1)!!`, which has no cancellation at
//!   any argument; past `a^2 >= (p + 4) ln 2` the complement is below the last bit and erf rounds
//!   to +-1.
//! * `harmonic` never forms `psi(x + 1) + gamma` (whose cancellation loses every digit near 0):
//!   on x > -1 it is `sum_{i<=m} x / (i (i + x)) + psi(x + 1 + m) - psi(1 + m)`, the digamma
//!   difference by its asymptotic series at two large arguments (no Euler gamma needed), with
//!   exact Bernoulli numbers from the tangent numbers; below -1 the reflection
//!   `H(x) = H(-x - 1) - pi cot(pi x)`. Each value carries guard bits for the cancellation it
//!   can meet (`-log2 |x|` near 0), so the result is good to the requested precision.
//!
//! Every kernel here takes FINITE operands off its poles; the callers own NaN, the infinities,
//! the signed zeros and the contract's bands. A precision past `MAX_BITS` returns `None`: the
//! series cost grows with it, and a caller treats `None` as no evidence (fail closed).

use astro_float::{BigFloat, Consts, RoundingMode, NAN};
use num_bigint::BigUint;
use std::sync::Mutex;

const RM: RoundingMode = RoundingMode::ToEven;

/// The largest working precision (bits) these kernels evaluate at: the contract ladder's
/// ceiling (2000 digits, 6,647 bits) with room for guard bits. `harmonic`'s cost grows as
/// p^2.6, so past it a value is no evidence anyone can afford.
pub(crate) const MAX_BITS: usize = 1 << 13;

fn int(n: u64, p: usize) -> BigFloat {
    BigFloat::from_u64(n, p)
}

fn gt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|o| o > 0)
}

fn lt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|o| o < 0)
}

/// `x` rounded to `p` bits.
fn round(x: &BigFloat, p: usize) -> BigFloat {
    x.add(&int(0, p), p, RM)
}

/// The binary exponent of a finite nonzero `x`: |x| in [2^(e-1), 2^e).
fn exponent(x: &BigFloat) -> i64 {
    x.exponent().map_or(0, i64::from)
}

// ------------------------------------------------------------------ the exact heads

/// The floored remainder `a - b floor(a / b)` with numpy's cells (`np.mod` on float64, at `p`
/// bits): NaN at a zero divisor or an infinite dividend; an infinite divisor returns the
/// dividend when the signs agree and the divisor otherwise (`mod(-3, inf) = inf`); a zero
/// result takes the divisor's sign. The truncated remainder is exact, so a finite result is
/// that remainder, or it plus `b` rounded once where the signs differ.
pub(crate) fn modulo(a: &BigFloat, b: &BigFloat, p: usize) -> BigFloat {
    if a.is_nan() || b.is_nan() || b.is_zero() || a.is_inf() {
        return NAN;
    }
    let zero_of_b = || {
        if b.is_negative() {
            int(0, p).neg()
        } else {
            int(0, p)
        }
    };
    if b.is_inf() {
        if a.is_zero() {
            return zero_of_b();
        }
        return if a.is_negative() == b.is_negative() {
            round(a, p)
        } else {
            b.clone()
        };
    }
    let m = a.rem(b);
    if m.is_zero() {
        return zero_of_b();
    }
    if m.is_negative() != b.is_negative() {
        m.add(b, p, RM)
    } else {
        round(&m, p)
    }
}

/// IEEE 754-2019 `maximum` / `minimum`: NaN-propagating, and -0 < +0, so both are
/// order-independent bit for bit (`kernels::max`).
pub(crate) fn lattice(a: &BigFloat, b: &BigFloat, max: bool, p: usize) -> BigFloat {
    if a.is_nan() || b.is_nan() {
        return NAN;
    }
    if a.is_zero() && b.is_zero() {
        // the tie of the two zeros: +0 for max unless both are -0, -0 for min unless both +0
        let neg = if max {
            a.is_negative() && b.is_negative()
        } else {
            a.is_negative() || b.is_negative()
        };
        return if neg { int(0, p).neg() } else { int(0, p) };
    }
    let pick_a = if max { !lt(a, b) } else { !gt(a, b) };
    round(if pick_a { a } else { b }, p)
}

/// `atan2(y, x)` at `p` bits with C99's cells: `atan2(+-0, +0) = +-0`, `atan2(+-0, -0) =
/// +-pi`, `atan2(+-inf, +inf) = +-pi/4`, `atan2(+-inf, -inf) = +-3pi/4`, `atan2(y, +-0) =
/// +-pi/2`, `atan2(y, +inf) = +-0`, `atan2(y, -inf) = +-pi`.
pub(crate) fn atan2(y: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> BigFloat {
    if y.is_nan() || x.is_nan() {
        return NAN;
    }
    let g = p + 32;
    let signed = |v: BigFloat| if y.is_negative() { v.neg() } else { v };
    let pi = cc.pi(g, RM);
    let frac = |n: u64, d: u64, cc: &mut Consts| {
        round(
            &cc.pi(g, RM).mul(&int(n, g), g, RM).div(&int(d, g), g, RM),
            p,
        )
    };
    if y.is_inf() {
        let v = if x.is_inf() {
            if x.is_negative() {
                frac(3, 4, cc)
            } else {
                frac(1, 4, cc)
            }
        } else {
            frac(1, 2, cc)
        };
        return signed(v);
    }
    if y.is_zero() || x.is_inf() {
        // y = +-0, or y finite against an infinite x: +-0 to the right, +-pi to the left
        return if x.is_negative() {
            signed(round(&pi, p))
        } else {
            signed(int(0, p))
        };
    }
    if x.is_zero() {
        return signed(frac(1, 2, cc));
    }
    let t = y.div(x, g, RM).atan(g, RM, cc);
    let v = if x.is_negative() {
        if y.is_negative() {
            t.sub(&pi, g, RM)
        } else {
            t.add(&pi, g, RM)
        }
    } else {
        t
    };
    round(&v, p)
}

// ------------------------------------------------------------------------------ erf

/// `erf(x)` for a finite nonzero `x` at `p` bits (the caller owns the zeros and the cells).
pub(crate) fn erf(x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if p > MAX_BITS {
        return None;
    }
    let neg = x.is_negative();
    let a = x.abs();
    let one = int(1, p);
    let signed = |v: BigFloat| if neg { v.neg() } else { v };
    // erfc(a) < e^(-a^2) <= 2^-(p+4) once a^2 >= (p + 4) ln 2: erf is +-1 to the last bit
    let sat = BigFloat::from_f64((p + 4) as f64 * std::f64::consts::LN_2, 64);
    let a2_est = a.mul(&a, 64, RM);
    if !lt(&a2_est, &sat) {
        return Some(signed(one));
    }
    let q = p + 64;
    let a2 = a.mul(&a, q, RM);
    let two_a2 = a2.mul(&int(2, q), q, RM);
    // sum_n (2a^2)^n / (2n+1)!!: positive terms; past the peak (n > a^2) they fall off
    let mut term = int(1, q);
    let mut sum = int(1, q);
    let peak = a2_est.floor();
    let limit = 64 * (q as u64) + 1024;
    let mut n: u64 = 0;
    loop {
        term = term.mul(&two_a2, q, RM).div(&int(2 * n + 3, q), q, RM);
        sum = sum.add(&term, q, RM);
        n += 1;
        let past_peak = gt(&int(n, 64), &peak);
        if past_peak && (term.is_zero() || exponent(&term) < exponent(&sum) - q as i64 - 2) {
            break;
        }
        if n > limit {
            return None;
        }
    }
    let sqrt_pi = cc.pi(q, RM).sqrt(q, RM);
    let v = a
        .mul(&int(2, q), q, RM)
        .div(&sqrt_pi, q, RM)
        .mul(&a2.neg().exp(q, RM, cc), q, RM)
        .mul(&sum, q, RM);
    Some(signed(round(&v, p)))
}

/// Is `erfc(a) < bound` for `a > 0` by the bound `erfc(a) < e^(-a^2) / (a sqrt(pi))`? A
/// `false` says nothing (the caller computes erf and looks).
pub(crate) fn erfc_below(a: &BigFloat, bound: &BigFloat, p: usize, cc: &mut Consts) -> bool {
    if a.is_zero() || a.is_negative() {
        return false;
    }
    let q = p + 32;
    let up =
        a.mul(a, q, RM)
            .neg()
            .exp(q, RM, cc)
            .div(&a.mul(&cc.pi(q, RM).sqrt(q, RM), q, RM), q, RM);
    lt(&up, bound)
}

// ------------------------------------------------------------------------- harmonic

/// The tangent numbers T_1, T_2, ... (1, 2, 16, 272, ...), exact, grown on demand.
static TANGENT: Mutex<Vec<BigUint>> = Mutex::new(Vec::new());

/// `|B_2k|` as BigFloats, k = 1, 2, ..., each with the precision it was computed at.
static BERNOULLI: Mutex<Vec<BigFloat>> = Mutex::new(Vec::new());

/// `|B_2k|` at (at least) `p` bits, from the tangent numbers:
/// `|B_2k| = 2k T_k / (4^k (4^k - 1))` (Brent and Harvey's in-place recurrence, exact). Both
/// tables grow on demand, so a value is converted once per precision it is first asked at.
fn bernoulli_abs(k: usize, p: usize) -> BigFloat {
    {
        let b = BERNOULLI.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = b.get(k - 1) {
            if v.precision().is_some_and(|q| q >= p) {
                return v.clone();
            }
        }
    }
    let num = {
        let mut t = TANGENT.lock().unwrap_or_else(|e| e.into_inner());
        if t.len() < k {
            let m = k.max(2 * t.len()).max(64);
            let mut v: Vec<BigUint> = vec![BigUint::from(0u32); m + 1];
            v[1] = BigUint::from(1u32);
            for i in 2..=m {
                v[i] = &v[i - 1] * BigUint::from((i - 1) as u64);
            }
            for i in 2..=m {
                for j in i..=m {
                    v[j] = &v[j - 1] * BigUint::from((j - i) as u64)
                        + &v[j] * BigUint::from((j - i + 2) as u64);
                }
            }
            *t = v.into_iter().skip(1).collect();
        }
        &t[k - 1] * BigUint::from(2 * k as u64)
    };
    let den = (BigUint::from(1u32) << (2 * k)) - BigUint::from(1u32);
    let mut w = exact(&num).div(&exact(&den), p, RM);
    // the remaining 4^-k is an exact exponent shift
    if let Some(e) = w.exponent() {
        w.set_exponent(e - 2 * k as i32);
    }
    let mut b = BERNOULLI.lock().unwrap_or_else(|e| e.into_inner());
    while b.len() < k {
        b.push(NAN); // a placeholder without precision, computed when first asked
    }
    b[k - 1] = w.clone();
    w
}

/// A natural number as an exact BigFloat.
fn exact(n: &BigUint) -> BigFloat {
    let digits = n.to_u64_digits();
    let p = 64 * (digits.len() + 1);
    let base = BigFloat::from_u64(u64::MAX, p).add(&int(1, p), p, RM); // 2^64, exact
    let mut acc = int(0, p);
    for d in digits.iter().rev() {
        acc = acc.mul(&base, p, RM).add(&int(*d, p), p, RM);
    }
    acc
}

/// `H(x)` for a finite `x` that is not a negative integer, at `p` bits (the caller owns 0, the
/// poles and the cells).
pub(crate) fn harmonic(x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if p > MAX_BITS {
        return None;
    }
    let minus_one = int(1, p).neg();
    if !lt(x, &minus_one) {
        return principal(x, p, cc);
    }
    // x < -1: H(x) = H(-x - 1) - pi cot(pi f), f the fractional part of x (period 1)
    let e = exponent(x).max(0) as usize;
    let q = p + 64 + e;
    let f = x.fract();
    if f.is_zero() {
        return Some(NAN); // a pole
    }
    let y = x.neg().sub(&int(1, q), q, RM); // exact at q
    let hy = principal(&y, q, cc)?;
    let pf = cc.pi(q, RM).mul(&f, q, RM);
    let cot = pf.cos(q, RM, cc).div(&pf.sin(q, RM, cc), q, RM);
    let v = hy.sub(&cc.pi(q, RM).mul(&cot, q, RM), q, RM);
    Some(round(&v, p))
}

/// `H(x)` on x > -1 (x != 0): the finite sum to `m` and the digamma difference at the two large
/// arguments `x + 1 + m` and `1 + m` by the asymptotic series, with `-log2 |x|` guard bits for
/// the cancellation near 0, where the result is `x (pi^2/6 - zeta(3) x + ...)`.
fn principal(x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if x.is_zero() {
        return Some(x.clone());
    }
    let ex = exponent(x);
    if ex < -(p as i64) - 16 {
        // H(x) = x pi^2/6 (1 - x zeta(3)/zeta(2) + ...): the first term is the value to p bits
        let g = p + 32;
        let pi = cc.pi(g, RM);
        let v = x.mul(&pi.mul(&pi, g, RM), g, RM).div(&int(6, g), g, RM);
        return Some(round(&v, p));
    }
    let q = p + 64 + (-ex).max(0) as usize;
    if q > 2 * MAX_BITS {
        return None;
    }
    let m = (q / 2 + 16) as u64;
    // S = sum_{i=1}^{m} x / (i (i + x)): one sign throughout, no cancellation
    let mut s = int(0, q);
    for i in 1..=m {
        let ii = int(i, q);
        let d = ii.mul(&ii.add(x, q, RM), q, RM);
        s = s.add(&x.div(&d, q, RM), q, RM);
    }
    let w0 = int(m + 1, q);
    let w1 = x.add(&w0, q, RM);
    let half = |w: &BigFloat| int(1, q).div(&w.mul(&int(2, q), q, RM), q, RM);
    let mut d = w1.ln(q, RM, cc).sub(&w0.ln(q, RM, cc), q, RM).sub(
        &half(&w1).sub(&half(&w0), q, RM),
        q,
        RM,
    );
    // - sum_k B_2k / (2k) (w1^-2k - w0^-2k), stopped once the term's bound at the smaller
    // argument is below 2^-(q+8)
    let wmin = if lt(&w1, &w0) { w1.clone() } else { w0.clone() };
    let (r1, r0, rmin) = (
        int(1, q).div(&w1.mul(&w1, q, RM), q, RM),
        int(1, q).div(&w0.mul(&w0, q, RM), q, RM),
        int(1, q).div(&wmin.mul(&wmin, q, RM), q, RM),
    );
    let kmax = q / 4 + 32;
    let (mut p1, mut p0, mut pmin) = (r1.clone(), r0.clone(), rmin.clone());
    let mut done = false;
    for i in 1..=kmax {
        let k = i as u64;
        let c = bernoulli_abs(i, q).div(&int(2 * k, q), q, RM);
        let bound = c.mul(&pmin, q, RM);
        if bound.is_zero() || exponent(&bound) < -(q as i64) - 8 {
            done = true;
            break;
        }
        let t = c.mul(&p1.sub(&p0, q, RM), q, RM);
        // B_2k alternates in sign, starting positive (B_2 = 1/6)
        d = if k % 2 == 1 {
            d.sub(&t, q, RM)
        } else {
            d.add(&t, q, RM)
        };
        p1 = p1.mul(&r1, q, RM);
        p0 = p0.mul(&r0, q, RM);
        pmin = pmin.mul(&rmin, q, RM);
    }
    if !done {
        return None;
    }
    Some(round(&s.add(&d, q, RM), p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The references, from mpmath printed to 78 digits: `mp.erf` at dps 80, and `mp.harmonic`
    /// at dps 800 (at dps 80 it loses every digit near 0: `mp.harmonic(1e-300)` is -4.1e-84
    /// there), `mp.atan2` at dps 60.
    const ERF_REF: [(f64, &str); 8] = [
        (1e-300, "1.12837916709551260217231607630436505761811009062667457958440199644244806709065e-300"),
        (0.25, "0.276326390168236932985068267764815712065353977892311254082471931262685083897302"),
        (1.0, "0.842700792949714869341220635082609259296066997966302908459937897834717254096011"),
        (-2.5, "-0.99959304798255504106043578426002508727965132259628657986087922123090299397015"),
        (5.5, "0.999999999999992642152082025601936931637601429909791776915030548123937007247072"),
        (10.0, "0.999999999999999999999999999999999999999999997911512416237455242999213705042211"),
        (-0.001, "-0.00112837879096923640343756413937148964182349907688256527384063117900939235055555"),
        (3.0e-8, "0.0000000338513750128653640362362336036404769664191905270195716196277584294344214166555"),
    ];

    const HARMONIC_REF: [(f64, &str); 10] = [
        (1e-300, "1.64493406684822647769296901065915638644211529964249261946012354800461329878812e-300"),
        (1e-20, "0.0000000000000000000164493406684822634624114235746666765431568273647118938142108066702154370979464"),
        (0.5, "0.613705638880109381165535757083646863848999731279489491758639981013212756060611"),
        (-0.5, "-1.38629436111989061883446424291635313615100026872051050824136001898678724393939"),
        (1.0, "1.0"),
        (10.25, "2.95248143119801292746895766279466380421191015791961504729120331402841547184744"),
        (-1.5, "0.613705638880109381165535757083646863848999731279489491758639981013212756060611"),
        (-2.25, "4.29135478511506069097962532726522173787208429660678714812543226767372733723402"),
        (1e17, "39.7211622458003094939123668197165939519275513092957308580390082180002688412927"),
        (-0.999, "-999.998356266907879430987539716911442894059003859371085673220569303759897997353"),
    ];

    /// An f64 as a BigFloat, the sign of a zero kept (`from_f64` drops it).
    fn big(v: f64, p: usize) -> BigFloat {
        let b = BigFloat::from_f64(v, p);
        if v == 0.0 && v.is_sign_negative() {
            b.neg()
        } else {
            b
        }
    }

    fn close(v: &BigFloat, want: &str, rel_bits: i64) {
        let mut cc = Consts::new().unwrap();
        let w = BigFloat::parse(want, astro_float::Radix::Dec, 400, RM, &mut cc);
        let d = v.sub(&w, 400, RM).abs();
        if d.is_zero() {
            return;
        }
        let scale = exponent(&w);
        assert!(
            exponent(&d) <= scale - rel_bits,
            "got {v}, want {want} (gap exponent {} vs scale {scale})",
            exponent(&d)
        );
    }

    #[test]
    fn erf_matches_mpmath() {
        let mut cc = Consts::new().unwrap();
        for p in [169usize, 402, 1024] {
            for (x, want) in ERF_REF {
                let v = erf(&BigFloat::from_f64(x, p), p, &mut cc).unwrap();
                // the reference carries ~80 digits (~265 bits): compare to the shorter of the two
                close(&v, want, (p as i64).min(250) - 4);
            }
        }
        // past the saturation the value is +-1 to the last bit
        let v = erf(&BigFloat::from_f64(40.0, 169), 169, &mut cc).unwrap();
        assert_eq!(v.cmp(&int(1, 169)), Some(0));
    }

    #[test]
    fn harmonic_matches_mpmath() {
        let mut cc = Consts::new().unwrap();
        for p in [169usize, 402, 1024] {
            for (x, want) in HARMONIC_REF {
                let v = harmonic(&BigFloat::from_f64(x, p), p, &mut cc).unwrap();
                close(&v, want, (p as i64).min(250) - 8);
            }
        }
        // a pole is NaN, and H(n) is the harmonic number
        assert!(harmonic(&BigFloat::from_f64(-3.0, 169), 169, &mut cc)
            .unwrap()
            .is_nan());
        let h3 = harmonic(&BigFloat::from_f64(3.0, 169), 169, &mut cc).unwrap();
        close(
            &h3,
            "1.8333333333333333333333333333333333333333333333333333",
            160,
        );
    }

    #[test]
    fn atan2_cells_and_values() {
        let mut cc = Consts::new().unwrap();
        let p = 169;
        let f = |v: f64| big(v, p);
        let pi = cc.pi(p, RM);
        let at = |y: f64, x: f64, cc: &mut Consts| atan2(&f(y), &f(x), p, cc);
        assert!(at(0.0, 1.0, &mut cc).is_zero() && !at(0.0, 1.0, &mut cc).is_negative());
        assert!(at(-0.0, 1.0, &mut cc).is_negative());
        assert_eq!(at(0.0, -0.0, &mut cc).cmp(&pi), Some(0));
        assert_eq!(at(-0.0, -1.0, &mut cc).cmp(&pi.neg()), Some(0));
        assert!(at(-0.0, 0.0, &mut cc).is_negative() && at(-0.0, 0.0, &mut cc).is_zero());
        let inf = f64::INFINITY;
        close(
            &at(inf, -inf, &mut cc),
            "2.35619449019234492884698253745962716314787704953132936573121",
            160,
        );
        close(&at(-1.0, inf, &mut cc), "0", 0);
        assert!(at(-1.0, inf, &mut cc).is_negative());
        // mp.atan2(-1, -2) and mp.atan2(3, 0.5)
        close(
            &at(-1.0, -2.0, &mut cc),
            "-2.67794504458898712224838715181828848216863234508898555716401",
            160,
        );
        close(
            &at(3.0, 0.5, &mut cc),
            "1.40564764938026978095219340199580798810019803922252509146944",
            160,
        );
    }

    #[test]
    fn mod_and_lattice_follow_numpy() {
        let p = 169;
        let f = |v: f64| big(v, p);
        let m = |a: f64, b: f64| modulo(&f(a), &f(b), p);
        let is = |v: BigFloat, w: f64| {
            assert_eq!(v.cmp(&f(w)), Some(0), "{v} vs {w}");
            assert_eq!(v.is_negative(), w.is_sign_negative(), "{v} vs {w}");
        };
        is(m(7.0, 3.0), 1.0);
        is(m(-7.0, 3.0), 2.0);
        is(m(7.0, -3.0), -2.0);
        is(m(-0.0, 3.0), 0.0);
        is(m(0.0, -3.0), -0.0);
        is(m(3.0, -3.0), -0.0);
        is(m(3.0, f64::INFINITY), 3.0);
        is(m(-3.0, -f64::INFINITY), -3.0);
        assert!(m(-3.0, f64::INFINITY).is_inf() && !m(-3.0, f64::INFINITY).is_negative());
        assert!(m(1.0, 0.0).is_nan() && m(f64::INFINITY, 2.0).is_nan());
        // the true remainder of a tiny negative dividend sits just below the divisor
        let r = m(-1e-20, 3.0);
        assert!(r.cmp(&f(3.0)).is_some_and(|o| o < 0));
        let l = |a: f64, b: f64, mx: bool| lattice(&f(a), &f(b), mx, p);
        is(l(-0.0, 0.0, true), 0.0);
        is(l(0.0, -0.0, false), -0.0);
        is(l(2.0, -5.0, true), 2.0);
        is(l(2.0, -5.0, false), -5.0);
        assert!(l(f64::NAN, 1.0, true).is_nan());
    }

    /// Bernoulli numbers from the tangent numbers: B_2 = 1/6, B_4 = -1/30, B_6 = 1/42,
    /// B_20 = -174611/330.
    #[test]
    fn bernoulli_from_tangent_numbers() {
        let p = 200;
        let b: Vec<BigFloat> = (1..=10).map(|k| bernoulli_abs(k, p)).collect();
        let q = |n: u64, d: u64| int(n, p).div(&int(d, p), p, RM);
        for (got, want) in [
            (&b[0], q(1, 6)),
            (&b[1], q(1, 30)),
            (&b[2], q(1, 42)),
            (&b[9], q(174611, 330)),
        ] {
            let gap = got.sub(&want, p, RM).abs();
            assert!(gap.is_zero() || exponent(&gap) < exponent(&want) - 190);
        }
    }
}
