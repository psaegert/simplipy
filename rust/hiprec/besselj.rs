//! `besselj(nu, x)` at arbitrary precision, for both high-precision evaluators (the rescue lane
//! and the contract's point machinery), with the contract's cells: the same table as
//! `verify._contract.c_besselj`, which evaluates with mpmath.
//!
//! THE CELLS (scipy's, D7): NaN at a NaN operand and at `x = +-inf`; NaN for a non-integer
//! order at a negative argument; an infinite order gives 0 at `x >= 0`; at `x = 0`,
//! `J_0(0) = 1`, `J_nu(0) = 0` for `nu > 0` and for a negative integer `nu`, and at a negative
//! non-integer order the pole's one-sided limit, the infinity of the sign of `1/Gamma(nu + 1)`.
//! An integer order reads its reflection `J_{-n} = (-1)^n J_n` and its parity
//! `J_n(-x) = (-1)^n J_n(x)`, so only `x > 0` is computed, at `nu >= 0` or a negative
//! non-integer `nu`.
//!
//! THE VALUE, within the last place of `p` bits (every error bound is held below `2^-(p + 16)`
//! of the value before the one final rounding), or `None` (Unresolved) where this cannot be had
//! within the caps (`NU_CAP`, `X_CAP`, `BITS_CAP`, `TERM_CAP`). Measured against mpmath at 400
//! and 1,110 digits on 980 points of every regime: within the last place at 169, 402, 834 and
//! 3,325 bits, the one point unresolved being `J_1000(1000)` at 169 bits. The value is:
//!
//! * the Hankel expansion `J = sqrt(2/(pi x)) (P cos w - Q sin w)`, `w = x - (nu/2 + 1/4) pi`,
//!   where `x` is large enough: for real `nu` and `x > 0` each remainder is bounded by the
//!   first neglected term once the number of terms reaches `|nu| - 1/2` (DLMF 10.17(iii)), so
//!   the error is bounded, not estimated;
//! * otherwise the power series `(x/2)^nu / Gamma(nu + 1) sum_k (-x^2/4)^k / (k! (nu + 1)_k)`,
//!   summed at a working precision that carries the cancellation (the largest term over the
//!   sum, measured, and the sum redone once at the precision it shows it needs), its tail
//!   bounded geometrically once the term ratio stays below 1/2;
//! * `Gamma` by Stirling's series after a shift (`gamma`), its remainder bounded by the
//!   first neglected term, the Bernoulli numbers exact from the tangent numbers, and Euler's
//!   reflection below 1/2.

use astro_float::{BigFloat, Consts, Radix, RoundingMode, INF_NEG, INF_POS, NAN};
use num_bigint::BigInt;
use std::sync::Mutex;

const RM: RoundingMode = RoundingMode::ToEven;

/// The largest order evaluated, the contract's (`_contract.BESSELJ_NU_CAP`): beyond it the
/// transition region `x ~ nu` costs more precision than a rung affords, and mpmath stops
/// converging (order 3e3 at twice the order, dps 50).
pub(crate) const NU_CAP: f64 = 1e3;
/// The largest argument evaluated, the contract's trig cap (`ct_unary`'s `sin`/`cos`).
pub(crate) const X_CAP: f64 = 1e12;
/// The largest working precision and the most series terms one evaluation may use.
const BITS_CAP: usize = 80_000;
const TERM_CAP: usize = 200_000;

fn small(n: u64, p: usize) -> BigFloat {
    BigFloat::from_u64(n, p)
}

/// `x` as the nearest-ish f64 (the top word of the mantissa): for planning only.
fn approx(x: &BigFloat) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_inf() {
        return if x.is_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if x.is_zero() {
        return 0.0;
    }
    let (Some(e), Some(d)) = (x.exponent(), x.mantissa_digits()) else {
        return f64::NAN;
    };
    let top = d.last().copied().unwrap_or(0) as f64 / 18_446_744_073_709_551_616.0;
    let v = if e > 1100 {
        f64::INFINITY
    } else if e < -1100 {
        0.0
    } else {
        top * 2f64.powi(e)
    };
    if x.is_negative() {
        -v
    } else {
        v
    }
}

/// `log2 |x|`, roughly (`None` for a zero or a non-finite value).
fn log2_abs(x: &BigFloat) -> Option<i64> {
    if x.is_zero() || x.is_nan() || x.is_inf() {
        return None;
    }
    x.exponent().map(i64::from)
}

/// Is the integer `n` (an integer-valued BigFloat) odd?
fn is_odd(n: &BigFloat) -> bool {
    let h = n.div(&small(2, 64), n.precision().unwrap_or(64) + 64, RM);
    !h.is_int()
}

/// `besselj(nu, x)` at the contract's cells (module documentation); `None` = Unresolved.
pub(crate) fn besselj(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if nu.is_nan() || x.is_nan() || x.is_inf() {
        return Some(NAN);
    }
    let integer = !nu.is_inf() && nu.is_int();
    if x.is_negative() && !x.is_zero() && !integer {
        return Some(NAN);
    }
    if nu.is_inf() {
        return Some(small(0, p));
    }
    if x.is_zero() {
        if nu.is_zero() {
            return Some(small(1, p));
        }
        if !nu.is_negative() || integer {
            return Some(small(0, p));
        }
        // Gamma(z), z = nu + 1, is positive for z > 0 and has the sign (-1)^ceil(-z) below.
        let z = nu.add(&small(1, p), nu.precision().unwrap_or(p) + 64, RM);
        let positive = !z.is_negative() || !is_odd(&z.neg().ceil());
        return Some(if positive { INF_POS } else { INF_NEG });
    }
    if approx(nu).abs() > NU_CAP || approx(x).abs() > X_CAP {
        return None;
    }
    if integer {
        let n = nu.abs();
        let odd = is_odd(&n);
        let flip = odd && (nu.is_negative() != x.is_negative());
        let v = besselj_positive(&n, &x.abs(), p, cc)?;
        return Some(if flip { v.neg() } else { v });
    }
    besselj_positive(nu, x, p, cc)
}

/// `J_nu(x)` for `x > 0` and `nu >= 0` or a negative non-integer `nu`, both finite and capped.
fn besselj_positive(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let (nuf, xf) = (approx(nu), approx(x));
    // The Hankel expansion reaches 2^-p only where its smallest term, about e^-2x, does; its
    // terms first grow to about e^(nu^2 / 2x), which the working precision carries.
    let growth = (nuf * nuf / (2.0 * xf) * std::f64::consts::LOG2_E).ceil() as usize;
    if xf > 0.35 * (p as f64 + 48.0) && growth <= 4 * p {
        if let Some(v) = hankel(nu, x, p, growth, cc) {
            return Some(v);
        }
    }
    series(nu, x, nuf, xf, p, cc)
}

/// The Hankel expansion (module documentation); `None` where its bound does not reach the
/// target before the terms turn.
fn hankel(
    nu: &BigFloat,
    x: &BigFloat,
    p: usize,
    growth: usize,
    cc: &mut Consts,
) -> Option<BigFloat> {
    let mag = |v: &BigFloat| log2_abs(v).unwrap_or(0).max(0) as usize;
    let wp = p + 96 + mag(x) + mag(nu) + growth;
    let mu = nu.mul(nu, wp, RM).mul(&small(4, wp), wp, RM);
    // Enough terms for the bound: at least |nu| + 1 in each of P and Q.
    let kmin = (2.0 * (approx(nu).abs() + 1.0)).ceil() as u64;
    let zero = small(0, wp);
    // t_k = a_k / x^k, added with the sign (-1)^floor(k/2) into P (even k) or Q (odd k)
    let mut term = small(1, wp);
    let (mut pp, mut qq) = (small(0, wp), small(0, wp));
    let mut least = i64::MAX;
    let mut k: u64 = 0;
    let (r_p, r_q) = loop {
        if term.is_zero() {
            // a half-integer order: the expansion terminates, exactly
            break (zero.clone(), zero.clone());
        }
        let lt = log2_abs(&term)?;
        if k >= kmin {
            let scale = log2_abs(&pp)
                .unwrap_or(0)
                .max(log2_abs(&qq).unwrap_or(0))
                .max(0);
            let target = scale - p as i64 - 48;
            if lt < target {
                let next = next_term(&term, &mu, k, x, wp);
                if next.is_zero() || log2_abs(&next)? < target {
                    // the first neglected terms of P and Q, which bound their remainders
                    break if k % 2 == 0 {
                        (term, next)
                    } else {
                        (next, term)
                    };
                }
            }
            if lt > least.saturating_add(64) {
                return None; // the expansion diverges before reaching the target
            }
        }
        least = least.min(lt);
        let t = if k % 4 < 2 { term.clone() } else { term.neg() };
        if k % 2 == 0 {
            pp = pp.add(&t, wp, RM);
        } else {
            qq = qq.add(&t, wp, RM);
        }
        term = next_term(&term, &mu, k, x, wp);
        k += 1;
        if k as usize > TERM_CAP {
            return None;
        }
    };
    let pi = cc.pi(wp, RM);
    let quarter = small(1, wp).div(&small(4, wp), wp, RM);
    let half_nu = nu.div(&small(2, wp), wp, RM);
    let w = x.sub(&half_nu.add(&quarter, wp, RM).mul(&pi, wp, RM), wp, RM);
    let (c, s) = (w.cos(wp, RM, cc), w.sin(wp, RM, cc));
    let comb = pp.mul(&c, wp, RM).sub(&qq.mul(&s, wp, RM), wp, RM);
    // The truncation error (both first neglected terms) and the rounding of w, cos and sin,
    // against the value: refuse unless the bound clears 2^-(p+16) of it.
    let lc = log2_abs(&comb)?;
    let le = log2_abs(&r_p.abs().add(&r_q.abs(), wp, RM)).unwrap_or(i64::MIN / 2);
    let lpq = log2_abs(&pp).unwrap_or(0).max(log2_abs(&qq).unwrap_or(0));
    let lround = mag(x) as i64 + mag(nu) as i64 + 8 - wp as i64 + lpq;
    if le.max(lround) > lc - p as i64 - 16 {
        return None;
    }
    let amp = small(2, wp).div(&pi.mul(x, wp, RM), wp, RM).sqrt(wp, RM);
    Some(amp.mul(&comb, p, RM))
}

/// `a_{k+1}/x^{k+1}` from `a_k/x^k`: `a_{k+1} = a_k (mu - (2k+1)^2) / (8 (k+1))`.
fn next_term(term: &BigFloat, mu: &BigFloat, k: u64, x: &BigFloat, wp: usize) -> BigFloat {
    let odd = small(2 * k + 1, wp);
    let f = mu.sub(&odd.mul(&odd, wp, RM), wp, RM);
    let d = small(8 * (k + 1), wp).mul(x, wp, RM);
    term.mul(&f, wp, RM).div(&d, wp, RM)
}

/// The power series (module documentation).
fn series(
    nu: &BigFloat,
    x: &BigFloat,
    nuf: f64,
    xf: f64,
    p: usize,
    cc: &mut Consts,
) -> Option<BigFloat> {
    // The largest term, log2, from the ratio x^2/4 / ((k+1)(nu+k+1)), in floats: the peak is
    // where the ratio crosses 1 for good (past -nu - 1, where nu + k + 1 turns positive).
    let q = xf * xf / 4.0;
    let mut lt = 0.0f64;
    let mut peak = 0.0f64;
    let mut k = 0usize;
    loop {
        let den = (k as f64 + 1.0) * (nuf + k as f64 + 1.0);
        let r = q / den.abs();
        if r < 1.0 && nuf + k as f64 + 1.0 > 0.0 {
            break;
        }
        lt += r.log2();
        peak = peak.max(lt);
        k += 1;
        if k > TERM_CAP {
            return None;
        }
    }
    let mut extra = peak.ceil() as usize + 64;
    for _attempt in 0..2 {
        let wp = p + extra;
        if wp > BITS_CAP {
            return None;
        }
        let (sum, biggest) = series_sum(nu, x, wp)?;
        let (Some(ls), Some(lb)) = (log2_abs(&sum), log2_abs(&biggest)) else {
            return None;
        };
        let lost = (lb - ls).max(0) as usize;
        if lost + 48 <= extra {
            let pre = prefactor(nu, x, p + 64, cc)?;
            return Some(pre.mul(&sum, p, RM));
        }
        extra = lost + 96;
    }
    None
}

/// `sum_k (-x^2/4)^k / (k! (nu + 1)_k)` at `wp` bits, and its largest term.
fn series_sum(nu: &BigFloat, x: &BigFloat, wp: usize) -> Option<(BigFloat, BigFloat)> {
    let q = x.mul(x, wp, RM).div(&small(4, wp), wp, RM).neg();
    let nu1 = nu.add(&small(1, wp), wp, RM);
    let mut term = small(1, wp);
    let mut sum = small(1, wp);
    let mut biggest = small(1, wp);
    let mut k: u64 = 0;
    loop {
        let kk = small(k + 1, wp);
        let den = kk.mul(&nu1.add(&small(k, wp), wp, RM), wp, RM);
        term = term.mul(&q, wp, RM).div(&den, wp, RM);
        sum = sum.add(&term, wp, RM);
        k += 1;
        if term.abs().cmp(&biggest.abs()).is_some_and(|o| o > 0) {
            biggest = term.clone();
        }
        // Past the peak, with the ratio below 1/2 and nu + k + 1 > 0 (every later ratio is
        // smaller), the tail is below twice the next term: stop once that is below the target.
        let next_den = approx(&small(k + 1, 64)) * (approx(&nu1) + k as f64);
        let ratio = approx(&q).abs() / next_den;
        if next_den > 0.0 && ratio < 0.5 {
            let lt = log2_abs(&term).unwrap_or(i64::MIN / 2);
            let ls = log2_abs(&sum).unwrap_or(0);
            if lt < ls - wp as i64 - 2 {
                return Some((sum, biggest));
            }
        }
        if k as usize > TERM_CAP {
            return None;
        }
    }
}

/// `(x/2)^nu / Gamma(nu + 1)` at `p` bits.
fn prefactor(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let half = x.div(&small(2, p), p, RM);
    let l = half.ln(p + 64, RM, cc);
    let e = nu.mul(&l, p + 64, RM);
    let guard = log2_abs(&e).unwrap_or(0).max(0) as usize;
    let wp = p + 64 + guard;
    let pw = nu.mul(&half.ln(wp, RM, cc), wp, RM).exp(wp, RM, cc);
    let z = nu.add(&small(1, wp), wp, RM);
    let g = gamma(&z, wp, cc)?;
    Some(pw.div(&g, p, RM))
}

/// `Gamma(z)` for a real `z` that is no pole, at `p` bits (cached per thread: the contract
/// evaluates one order at many points).
pub(crate) fn gamma(z: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    type Key = (i32, Vec<u64>, bool, usize);
    thread_local! {
        static CACHE: std::cell::RefCell<std::collections::HashMap<Key, BigFloat>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    if z.is_nan() || z.is_inf() || (z.is_int() && (z.is_negative() || z.is_zero())) {
        return None;
    }
    let key: Key = (
        z.exponent()?,
        z.mantissa_digits()?.to_vec(),
        z.is_negative(),
        p,
    );
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Some(hit);
    }
    let g = gamma_uncached(z, p, cc)?;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 4096 {
            c.clear();
        }
        c.insert(key, g.clone());
    });
    Some(g)
}

fn gamma_uncached(z: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let zf = approx(z);
    if z.is_int() && zf <= 1e4 {
        // (z - 1)!, exact to the rounding of each product
        let wp = p + 64;
        let mut g = small(1, wp);
        for i in 2..(zf as u64) {
            g = g.mul(&small(i, wp), wp, RM);
        }
        return Some(g.add(&small(0, p), p, RM));
    }
    let half = small(1, p + 64).div(&small(2, p + 64), p + 64, RM);
    if z.cmp(&half).is_some_and(|o| o < 0) {
        // Euler's reflection, Gamma(z) = pi / (sin(pi z) Gamma(1 - z)), with sin(pi z) read
        // off the fractional part f = z - floor(z) exactly: sin(pi z) = (-1)^floor(z) sin(pi f).
        let wp = p + 64 + log2_abs(z).unwrap_or(0).max(0) as usize;
        let pi = cc.pi(wp, RM);
        let n = z.floor();
        let f = z.sub(&n, wp + log2_abs(z).unwrap_or(0).max(0) as usize + 8, RM);
        let mut s = pi.mul(&f, wp, RM).sin(wp, RM, cc);
        if is_odd(&n) {
            s = s.neg();
        }
        let g = gamma(&small(1, wp).sub(z, wp, RM), wp, cc)?;
        return Some(pi.div(&s.mul(&g, wp, RM), p, RM));
    }
    // Shift up to w >= W: Gamma(z) = Gamma(z + m) / (z (z+1) ... (z+m-1)).
    let target = (p as f64).max(32.0);
    let m = if zf < target {
        (target - zf).ceil() as u64
    } else {
        0
    };
    let wp = p + 64 + (64 - m.leading_zeros() as usize);
    let mut prod = small(1, wp);
    let mut w = z.clone();
    for _ in 0..m {
        prod = prod.mul(&w, wp, RM);
        w = w.add(&small(1, wp), wp, RM);
    }
    let lg = ln_gamma_large(&w, wp, cc)?;
    let guard = log2_abs(&lg).unwrap_or(0).max(0) as usize;
    let g = ln_gamma_large(&w, wp + guard, cc)?.exp(wp + guard, RM, cc);
    Some(g.div(&prod, p, RM))
}

/// Stirling's series for `ln Gamma(w)`, `w` large and positive, to an absolute error below
/// `2^-p`: `(w - 1/2) ln w - w + ln(2 pi)/2 + sum_k B_2k / (2k (2k-1) w^(2k-1))`, stopped
/// where the first neglected term (which bounds the remainder for real `w > 0`) is below the
/// target.
fn ln_gamma_large(w: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let wp = p + 32;
    let lw = w.ln(wp, RM, cc);
    let half = small(1, wp).div(&small(2, wp), wp, RM);
    let two_pi = cc.pi(wp, RM).mul(&small(2, wp), wp, RM);
    let mut s = w.sub(&half, wp, RM).mul(&lw, wp, RM).sub(w, wp, RM);
    s = s.add(&two_pi.ln(wp, RM, cc).mul(&half, wp, RM), wp, RM);
    let w2 = w.mul(w, wp, RM);
    let mut wpow = w.clone(); // w^(2k-1)
    let mut prev: Option<i64> = None;
    for k in 1..=4096usize {
        let c = stirling_coefficient(k, wp, cc)?;
        let t = c.div(&wpow, wp, RM);
        let lt = log2_abs(&t)?;
        if lt < -(p as i64) - 8 {
            return Some(s);
        }
        if prev.is_some_and(|q| lt > q) {
            return None; // the series turned before reaching the target: shift further
        }
        prev = Some(lt);
        s = s.add(&t, wp, RM);
        wpow = wpow.mul(&w2, wp, RM);
    }
    None
}

/// `B_2k / (2k (2k - 1))` at `p` bits, from the tangent number `T_k`:
/// `B_2k = (-1)^(k-1) 2k T_k / (2^2k (2^2k - 1))`.
fn stirling_coefficient(k: usize, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let t = tangent_number(k)?;
    let wp = p.max(2 * k + 64) + 64;
    let tk = BigFloat::parse(&t.to_string(), Radix::Dec, wp, RM, cc);
    let pow = small(1, wp).mul(&small(2, wp).powi(2 * k, wp, RM), wp, RM);
    let den =
        pow.sub(&small(1, wp), wp, RM)
            .mul(&pow, wp, RM)
            .mul(&small(2 * k as u64 - 1, wp), wp, RM);
    let c = tk.div(&den, p, RM);
    Some(if k % 2 == 0 { c.neg() } else { c })
}

/// The tangent numbers `T_1 = 1, T_2 = 2, T_3 = 16, T_4 = 272, ...`, exact (Brent and
/// Harvey's integer recurrence), cached for every thread.
fn tangent_number(k: usize) -> Option<BigInt> {
    static CACHE: Mutex<Vec<BigInt>> = Mutex::new(Vec::new());
    let mut c = CACHE.lock().ok()?;
    if c.len() < k {
        let n = (2 * k).max(64);
        let mut t: Vec<BigInt> = vec![BigInt::from(0); n + 1];
        t[1] = BigInt::from(1);
        for j in 2..=n {
            t[j] = &t[j - 1] * BigInt::from(j - 1);
        }
        for i in 2..=n {
            for j in i..=n {
                t[j] = &t[j - 1] * BigInt::from(j - i) + &t[j] * BigInt::from(j - i + 2);
            }
        }
        t.remove(0);
        *c = t;
    }
    c.get(k - 1).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(s: &str, p: usize, cc: &mut Consts) -> BigFloat {
        BigFloat::parse(s, Radix::Dec, p, RM, cc)
    }

    #[test]
    fn tangent_numbers() {
        let want = [1u64, 2, 16, 272, 7936, 353792, 22368256];
        for (k, w) in want.iter().enumerate() {
            assert_eq!(tangent_number(k + 1).unwrap(), BigInt::from(*w));
        }
    }

    #[test]
    fn gamma_reads_known_values() {
        let mut cc = Consts::new().unwrap();
        let p = 300;
        // Gamma(1/2) = sqrt(pi), Gamma(5) = 24, Gamma(-1/2) = -2 sqrt(pi)
        let sp = cc.pi(p + 64, RM).sqrt(p + 64, RM);
        let half = bf("0.5", p, &mut cc);
        let g = gamma(&half, p, &mut cc).unwrap();
        let rel = g.sub(&sp, p, RM).div(&sp, p, RM).abs();
        assert!(log2_abs(&rel).unwrap_or(-10_000) < -(p as i64) + 8, "{g}");
        let g5 = gamma(&small(5, p), p, &mut cc).unwrap();
        assert!(log2_abs(&g5.sub(&small(24, p), p, RM)).unwrap_or(-10_000) < -(p as i64) + 16);
        let gm = gamma(&bf("-0.5", p, &mut cc), p, &mut cc).unwrap();
        let want = sp.mul(&small(2, p), p, RM).neg();
        let rel = gm.sub(&want, p, RM).div(&want, p, RM).abs();
        assert!(log2_abs(&rel).unwrap_or(-10_000) < -(p as i64) + 8, "{gm}");
    }

    /// mpmath's `besselj` at 400 digits (`besselj_mpmath.txt`: the order and argument as the
    /// exact decimals of doubles, then J), over the regimes: orders integer, half-integer,
    /// negative non-integer and to 1e4 (Unresolved beyond the cap, 1e3), arguments from 1e-300
    /// to 1e12, beside zeros of J. Within the last place at the ordinary rung and at 834 bits.
    #[test]
    fn agrees_with_mpmath() {
        let mut cc = Consts::new().unwrap();
        for p in [169usize, 834] {
            for line in include_str!("besselj_mpmath.txt").lines() {
                let f: Vec<&str> = line.split_whitespace().collect();
                let nu = bf(f[0], p, &mut cc);
                let x = bf(f[1], p, &mut cc);
                let want = bf(f[2], 1100, &mut cc);
                let got = besselj(&nu, &x, p, &mut cc);
                if approx(&nu).abs() > NU_CAP {
                    assert!(got.is_none(), "beyond the order cap: {}", &line[..40]);
                    continue;
                }
                let Some(v) = got else {
                    assert!(
                        p == 169 && f[0] == "1000",
                        "unresolved at p={p}: {}",
                        &line[..40]
                    );
                    continue;
                };
                let rel = v.sub(&want, 1100, RM).div(&want, 1100, RM);
                assert!(
                    rel.is_zero() || log2_abs(&rel).unwrap() <= -(p as i64),
                    "p={p}: {} -> {v}",
                    &line[..line.len().min(80)]
                );
            }
        }
    }

    #[test]
    fn cells() {
        let mut cc = Consts::new().unwrap();
        let p = 200;
        let j = |nu: &str, x: &str, cc: &mut Consts| {
            let (a, b) = (bf(nu, p, cc), bf(x, p, cc));
            besselj(&a, &b, p, cc)
        };
        assert!(j("0", "0", &mut cc).unwrap().cmp(&small(1, p)) == Some(0));
        assert!(j("2.5", "0", &mut cc).unwrap().is_zero());
        assert!(j("-3", "0", &mut cc).unwrap().is_zero());
        assert_eq!(j("-0.5", "0", &mut cc).unwrap().is_inf_pos(), true);
        assert_eq!(j("-1.5", "0", &mut cc).unwrap().is_inf_neg(), true);
        assert!(j("0.5", "-1", &mut cc).unwrap().is_nan());
        assert!(j("1", "inf", &mut cc).unwrap().is_nan());
        assert!(j("1e5", "3", &mut cc).is_none());
        assert!(j("1", "1e13", &mut cc).is_none());
        // parity and reflection
        let a = j("3", "2.5", &mut cc).unwrap();
        assert!(j("3", "-2.5", &mut cc).unwrap().cmp(&a.neg()) == Some(0));
        assert!(j("-3", "2.5", &mut cc).unwrap().cmp(&a.neg()) == Some(0));
        assert!(
            j("-2", "-2.5", &mut cc)
                .unwrap()
                .cmp(&j("2", "2.5", &mut cc).unwrap())
                == Some(0)
        );
    }
}
