//! `bessely(nu, x)`, `besseli(nu, x)` and `besselk(nu, x)` at arbitrary precision, for the table
//! and both high-precision evaluators: their REAL values at FINITE operands. The callers own NaN
//! and the infinities, where the reals' cells and float64's differ (`scipy_cell` gives
//! float64's).
//!
//! THE CELLS at finite operands:
//!
//! * a negative argument: Y and K have no real value (NaN); I has one at an integer order only,
//!   `I_n(-x) = (-1)^n I_n(x)`, and none otherwise;
//! * a zero argument: `K_nu(0) = +inf`; `Y_nu(0) = -inf` for `nu >= 0`, and at a negative order
//!   the limit `-sign(cos(nu pi)) inf` (`Y_-1(0) = +inf`); at a negative half-integer order,
//!   where `cos(nu pi)` changes sign, the limit taken in the order and the argument together does
//!   not exist: no value (NaN); `I_0(0) = 1`,
//!   `I_nu(0) = 0` for `nu > 0` and for a negative integer `nu`, and at a negative non-integer
//!   order the pole's one-sided limit, the infinity of the sign of `1/Gamma(nu + 1)`;
//! * an integer order reads its reflection (`Y_-n = (-1)^n Y_n`, `I_-n = I_n`), and K is even in
//!   its order.
//!
//! THE VALUE, within the last place of `p` bits (every error bound is held below `2^-(p + 16)`
//! of the value before the one final rounding), or `None` (Unresolved) where it cannot be had
//! within the caps (the order to `besselj::NU_CAP`, the argument to `X_CAP` for Y, `I_X_CAP` for
//! I and the I-difference of K, `K_X_CAP` for K):
//!
//! * Y: the Hankel expansion with `besselj`'s P and Q where x is large
//!   (`Y = sqrt(2/(pi x)) (P sin w + Q cos w)`, its remainders bounded by the first neglected
//!   terms, DLMF 10.17(iii)); at an integer order the series of DLMF 10.8.1; at a non-integer
//!   order `(J_nu cos(nu pi) - J_-nu) / sin(nu pi)` at the precision its cancellation needs.
//! * I: the power series `(x/2)^nu / Gamma(nu + 1) sum_k (x^2/4)^k / (k! (nu + 1)_k)`
//!   (`besselj::series`).
//! * K: the expansion `sqrt(pi/(2x)) e^-x sum_k a_k(nu)/x^k` (DLMF 10.40.2) where it reaches the
//!   target: for real `nu` and `x > 0` its remainder after `l >= |nu| - 1/2` terms is bounded by
//!   the first neglected term (DLMF 10.40(ii)), and it terminates at a half-integer order. Else
//!   at an integer order the series of DLMF 10.31.1, at a non-integer one
//!   `pi (I_-nu - I_nu) / (2 sin(nu pi))` at the precision its cancellation needs.
//!
//! The integer-order series need Euler's constant, read by Brent and McMillan's algorithm B1
//! (`euler_gamma`, its error below `pi e^(-4n)`).

use super::besselj::{
    approx, besselj_positive, hankel, hankel_growth, is_odd, log2_abs, series, small, BITS_CAP,
    NU_CAP, TERM_CAP, X_CAP,
};
use astro_float::{BigFloat, Consts, RoundingMode, INF_NEG, INF_POS, NAN};

const RM: RoundingMode = RoundingMode::ToEven;

/// The largest argument I is read at: its series costs about 1.4 x terms, and I grows like
/// `e^x` (`I_0(1e4)` is about `10^4339`).
pub(crate) const I_X_CAP: f64 = 1e4;
/// The largest argument K is read at: `e^-x` stays inside the float's exponent range well past
/// it.
pub(crate) const K_X_CAP: f64 = 1e8;

/// `(sin(pi nu), cos(pi nu))` at `wp` bits, each read off `nu`'s distance to the nearest of its
/// zeros, exactly (at the order's own bits, which carry it: `table::bessel_order`):
/// `sin(pi nu) = (-1)^n sin(pi (nu - n))` with `n` the nearest integer and
/// `cos(pi nu) = (-1)^(m+1) sin(pi (nu - m - 1/2))` with `m + 1/2` the nearest half-integer, so
/// each keeps the working precision however near `nu` lies to a zero of it (`sin` exactly 0 at an
/// integer, `cos` exactly 0 at a half-integer).
fn sincos_pi(nu: &BigFloat, wp: usize, cc: &mut Consts) -> (BigFloat, BigFloat) {
    let lnu = log2_abs(nu).unwrap_or(0).max(0) as usize;
    let fp = wp + lnu + nu.precision().unwrap_or(wp) + 8;
    let half = small(1, 64).div(&small(2, 64), 64, RM);
    let mut sin_pi = |r: &BigFloat| {
        if r.is_zero() {
            small(0, wp)
        } else {
            cc.pi(wp + 8, RM).mul(r, wp + 8, RM).sin(wp, RM, cc)
        }
    };
    let n = nu.add(&half, fp, RM).floor();
    let mut s = sin_pi(&nu.sub(&n, fp, RM));
    let m = nu.floor();
    let mut c = sin_pi(&nu.sub(&m, fp, RM).sub(&half, fp, RM));
    if is_odd(&n) {
        s = s.neg();
    }
    if !is_odd(&m) {
        c = c.neg();
    }
    (s, c)
}

/// `nu` a finite integer-valued order, as an integer (the caller capped it).
fn as_order(nu: &BigFloat) -> u64 {
    approx(nu).abs().round() as u64
}

/// The sign of `1/Gamma(nu + 1)` at a negative non-integer `nu`: positive above -1, and
/// `(-1)^ceil(-(nu + 1))` below.
fn reciprocal_gamma_positive(nu: &BigFloat, p: usize) -> bool {
    let z = nu.add(&small(1, p), nu.precision().unwrap_or(p) + 64, RM);
    !z.is_negative() || !is_odd(&z.neg().ceil())
}

// ------------------------------------------------------------------------------- Y

/// `Y_nu(x)` at finite operands (module documentation); `None` = Unresolved.
pub(crate) fn bessely(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if x.is_negative() && !x.is_zero() {
        return Some(NAN);
    }
    if x.is_zero() {
        if !nu.is_negative() {
            return Some(INF_NEG);
        }
        let (_, c) = sincos_pi(nu, 64, cc);
        if c.is_zero() {
            // a negative half-integer order: the orders beside it tend to infinities of both
            // signs, so the joint limit does not exist
            return Some(NAN);
        }
        return Some(if c.is_negative() { INF_POS } else { INF_NEG });
    }
    if approx(nu).abs() > NU_CAP || approx(x) > X_CAP {
        return None;
    }
    if nu.is_int() {
        let n = nu.abs();
        let v = y_positive(&n, x, p, cc)?;
        return Some(if nu.is_negative() && is_odd(&n) {
            v.neg()
        } else {
            v
        });
    }
    y_positive(nu, x, p, cc)
}

/// `Y_nu(x)` for `x > 0` and `nu >= 0` or a negative non-integer `nu`.
fn y_positive(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let (nuf, xf) = (approx(nu), approx(x));
    if let Some(growth) = hankel_growth(nuf, xf, p) {
        if let Some(v) = hankel(nu, x, p, growth, true, cc) {
            return Some(v);
        }
    }
    if nu.is_int() {
        integer_order(as_order(nu), x, p, false, cc)
    } else {
        y_rotation(nu, x, p, cc)
    }
}

/// `(J_nu cos(nu pi) - J_-nu) / sin(nu pi)` for a non-integer `nu`, the numerator read at the
/// precision its cancellation needs (measured, and read again where it fell short); at a
/// half-integer order `-J_-nu / sin(nu pi)`, exactly.
fn y_rotation(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let mut extra = 64usize;
    for _attempt in 0..3 {
        let wp = p + extra;
        if wp > BITS_CAP {
            return None;
        }
        let (s, c) = sincos_pi(nu, wp, cc);
        let jm = besselj_positive(&nu.neg(), x, wp, cc)?;
        let num = if c.is_zero() {
            jm.neg()
        } else {
            let jp = besselj_positive(nu, x, wp, cc)?;
            let a = jp.mul(&c, wp, RM);
            let num = a.sub(&jm, wp, RM);
            let big = log2_abs(&a)
                .unwrap_or(i64::MIN / 2)
                .max(log2_abs(&jm).unwrap_or(i64::MIN / 2));
            let Some(ln) = log2_abs(&num) else {
                extra += 128;
                continue;
            };
            let lost = (big - ln).max(0) as usize;
            if lost + 48 > extra {
                extra = lost + 96;
                continue;
            }
            num
        };
        return Some(num.div(&s, p, RM));
    }
    None
}

// ------------------------------------------------------------------------------- I

/// `I_nu(x)` at finite operands (module documentation); `None` = Unresolved.
pub(crate) fn besseli(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let integer = nu.is_int();
    if x.is_negative() && !x.is_zero() {
        if !integer {
            return Some(NAN);
        }
        let v = besseli(nu, &x.abs(), p, cc)?;
        return Some(if is_odd(&nu.abs()) { v.neg() } else { v });
    }
    if x.is_zero() {
        if nu.is_zero() {
            return Some(small(1, p));
        }
        if !nu.is_negative() || integer {
            return Some(small(0, p));
        }
        return Some(if reciprocal_gamma_positive(nu, p) {
            INF_POS
        } else {
            INF_NEG
        });
    }
    if approx(nu).abs() > NU_CAP || approx(x) > I_X_CAP {
        return None;
    }
    let nu = if integer { nu.abs() } else { nu.clone() };
    series(&nu, x, approx(&nu), approx(x), p, true, cc)
}

// ------------------------------------------------------------------------------- K

/// `K_nu(x)` at finite operands (module documentation); `None` = Unresolved.
pub(crate) fn besselk(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if x.is_negative() && !x.is_zero() {
        return Some(NAN);
    }
    if x.is_zero() {
        return Some(INF_POS);
    }
    let nu = nu.abs();
    if approx(&nu) > NU_CAP || approx(x) > K_X_CAP {
        return None;
    }
    if let Some(v) = k_expansion(&nu, x, p, cc) {
        return Some(v);
    }
    if nu.is_int() {
        integer_order(as_order(&nu), x, p, true, cc)
    } else {
        k_from_i(&nu, x, p, cc)
    }
}

/// K's expansion (module documentation) for `nu >= 0`, `x > 0`, where it is worth trying: a
/// half-integer order (it terminates), or x large enough for its smallest term, about `e^-2x`,
/// to reach the target while its terms first grow by no more than `4p` bits.
fn k_expansion(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let (nuf, xf) = (approx(nu), approx(x));
    let two_nu = nu.mul(&small(2, 64), nu.precision().unwrap_or(64) + 8, RM);
    let half_integer = two_nu.is_int() && !nu.is_int();
    let growth = (nuf * nuf / (2.0 * xf) * std::f64::consts::LOG2_E).ceil();
    if !half_integer && !(xf > 0.35 * (p as f64 + 48.0) && growth <= 4.0 * p as f64) {
        return None;
    }
    let mag = |v: &BigFloat| log2_abs(v).unwrap_or(0).max(0) as usize;
    if growth > BITS_CAP as f64 {
        return None;
    }
    let wp = p + 96 + mag(x) + mag(nu) + growth.max(0.0) as usize;
    if wp > BITS_CAP {
        return None;
    }
    let mu = nu.mul(nu, wp, RM).mul(&small(4, wp), wp, RM);
    let kmin = (nuf - 0.5).max(0.0).ceil() as u64;
    let mut term = small(1, wp);
    let mut sum = small(0, wp);
    let mut least = i64::MAX;
    let mut most = 0i64;
    let mut k: u64 = 0;
    let remainder = loop {
        if term.is_zero() {
            break small(0, wp); // a half-integer order: the expansion terminates, exactly
        }
        let lt_ = log2_abs(&term)?;
        if k >= kmin {
            let target = log2_abs(&sum).unwrap_or(0) - p as i64 - 48;
            if lt_ < target {
                break term; // the first neglected term bounds the remainder
            }
            if lt_ > least.saturating_add(64) {
                return None; // the terms turned before reaching the target
            }
        }
        least = least.min(lt_);
        most = most.max(lt_);
        sum = sum.add(&term, wp, RM);
        let odd = small(2 * k + 1, wp);
        let f = mu.sub(&odd.mul(&odd, wp, RM), wp, RM);
        let d = small(8 * (k + 1), wp).mul(x, wp, RM);
        term = term.mul(&f, wp, RM).div(&d, wp, RM);
        k += 1;
        if k as usize > TERM_CAP {
            return None;
        }
    };
    // The truncation and the rounding of the sum against the value.
    let ls = log2_abs(&sum)?;
    let le = log2_abs(&remainder).unwrap_or(i64::MIN / 2);
    let lround = (64 - k.leading_zeros()) as i64 + 8 - wp as i64 + most.max(ls);
    if le.max(lround) > ls - p as i64 - 16 {
        return None;
    }
    let pi = cc.pi(wp, RM);
    let pre = pi
        .div(&x.mul(&small(2, wp), wp, RM), wp, RM)
        .sqrt(wp, RM)
        .mul(&x.neg().exp(wp, RM, cc), wp, RM);
    Some(pre.mul(&sum, p, RM))
}

/// `pi (I_-nu - I_nu) / (2 sin(nu pi))` for a non-integer `nu > 0`, the difference read at the
/// precision its cancellation needs (about `2x log2(e)` bits, measured, and read again where it
/// fell short).
fn k_from_i(nu: &BigFloat, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    let xf = approx(x);
    if xf > I_X_CAP {
        return None;
    }
    let mut extra = 64 + (2.9 * xf).ceil() as usize;
    for _attempt in 0..3 {
        let wp = p + extra;
        if wp > BITS_CAP {
            return None;
        }
        let (s, _) = sincos_pi(nu, wp, cc);
        let im = besseli(&nu.neg(), x, wp, cc)?;
        let ip = besseli(nu, x, wp, cc)?;
        let d = im.sub(&ip, wp, RM);
        let big = log2_abs(&im)
            .unwrap_or(i64::MIN / 2)
            .max(log2_abs(&ip).unwrap_or(i64::MIN / 2));
        let Some(ld) = log2_abs(&d) else {
            extra += 128;
            continue;
        };
        let lost = (big - ld).max(0) as usize;
        if lost + 48 > extra {
            extra = lost + 96;
            continue;
        }
        let pi = cc.pi(wp, RM);
        return Some(pi.mul(&d, wp, RM).div(&s.mul(&small(2, wp), wp, RM), p, RM));
    }
    None
}

// ------------------------------------------------------------- integer orders, Y and K

/// `Y_n(x)` (`k_kind = false`, DLMF 10.8.1) or `K_n(x)` (DLMF 10.31.1) for an integer `n >= 0`
/// and `x > 0`, read at the precision the series' cancellation needs (measured: the largest
/// contribution against the value, and read again where it fell short):
///
/// * `Y_n = ((x/2)^n sum_k c_k t_k - (x/2)^-n F) / pi`, `t_k = (-x^2/4)^k / (k! (n+k)!)`,
///   `c_k = 2 ln(x/2) + 2 gamma - H_k - H_(n+k)`, `F = sum_(k<n) (n-k-1)!/k! (x^2/4)^k`;
/// * `K_n = (x/2)^-n G / 2 + (-1)^(n+1) (x/2)^n sum_k c_k u_k`, `u_k = (x^2/4)^k / (k! (n+k)!)`,
///   `c_k = ln(x/2) + gamma - (H_k + H_(n+k))/2`, `G = sum_(k<n) (n-k-1)!/k! (-x^2/4)^k`.
fn integer_order(
    n: u64,
    x: &BigFloat,
    p: usize,
    k_kind: bool,
    cc: &mut Consts,
) -> Option<BigFloat> {
    let xf = approx(x);
    let mut extra = 64 + ((if k_kind { 2.9 } else { 1.45 }) * xf).ceil() as usize;
    for _attempt in 0..3 {
        let wp = p + extra;
        if wp > BITS_CAP {
            return None;
        }
        let (value, biggest, terms) = integer_order_sum(n, x, wp, k_kind, cc)?;
        let Some(lv) = log2_abs(&value) else {
            extra += 128;
            continue;
        };
        let lost = (biggest - lv).max(0) as usize + (64 - terms.leading_zeros() as usize);
        if lost + 48 <= extra {
            return Some(value.add(&small(0, p), p, RM));
        }
        extra = lost + 96;
    }
    None
}

/// The sums of `integer_order` at `wp` bits: `(value, log2 of the largest contribution, terms)`.
fn integer_order_sum(
    n: u64,
    x: &BigFloat,
    wp: usize,
    k_kind: bool,
    cc: &mut Consts,
) -> Option<(BigFloat, i64, u64)> {
    let half = x.div(&small(2, wp), wp, RM);
    let lh = half.ln(wp, RM, cc);
    let g = euler_gamma(wp, cc);
    let h = half.powi(n as usize, wp, RM); // (x/2)^n
    let q = half.mul(&half, wp, RM); // x^2/4
    let mut biggest = i64::MIN / 2;
    fn note(v: &BigFloat, biggest: &mut i64) {
        if let Some(l) = log2_abs(v) {
            *biggest = (*biggest).max(l);
        }
    }
    // the finite part, F or G, over (x/2)^n
    let mut finite = small(0, wp);
    if n > 0 {
        let mut fact = small(1, wp); // (n-1)!
        for i in 2..n {
            fact = fact.mul(&small(i, wp), wp, RM);
        }
        let mut term = fact.div(&h, wp, RM);
        for k in 0..n {
            note(&term, &mut biggest);
            finite = finite.add(&term, wp, RM);
            if k + 1 < n {
                let step = q.div(&small((k + 1) * (n - k - 1), wp), wp, RM);
                term = term.mul(&step, wp, RM);
                if k_kind {
                    term = term.neg();
                }
            }
        }
    }
    // the infinite part, sum_k c_k t_k (Y) or c_k u_k (K), times (x/2)^n
    let mut nfact = small(1, wp);
    for i in 2..=n {
        nfact = nfact.mul(&small(i, wp), wp, RM);
    }
    let mut t = small(1, wp).div(&nfact, wp, RM).mul(&h, wp, RM);
    let mut hk = small(0, wp); // H_k
    let mut hnk = small(0, wp); // H_(n+k)
    for j in 1..=n {
        hnk = hnk.add(&small(1, wp).div(&small(j, wp), wp, RM), wp, RM);
    }
    let two = small(2, wp);
    let (base, halfh) = if k_kind {
        (lh.add(&g, wp, RM), true)
    } else {
        (lh.add(&g, wp, RM).mul(&two, wp, RM), false)
    };
    let mut sum = small(0, wp);
    let mut k: u64 = 0;
    loop {
        let hs = hk.add(&hnk, wp, RM);
        let c = if halfh {
            base.sub(&hs.div(&two, wp, RM), wp, RM)
        } else {
            base.sub(&hs, wp, RM)
        };
        let ct = c.mul(&t, wp, RM);
        note(&ct, &mut biggest);
        sum = sum.add(&ct, wp, RM);
        // the next term: t_(k+1) = t_k (-+ x^2/4) / ((k+1)(n+k+1))
        let den = (k + 1) as f64 * (n + k + 1) as f64;
        let ratio = approx(&q) / den;
        t = t
            .mul(&q, wp, RM)
            .div(&small((k + 1) * (n + k + 1), wp), wp, RM);
        if !k_kind {
            t = t.neg();
        }
        k += 1;
        hk = hk.add(&small(1, wp).div(&small(k, wp), wp, RM), wp, RM);
        hnk = hnk.add(&small(1, wp).div(&small(n + k, wp), wp, RM), wp, RM);
        // past the peak, with the ratio below 1/2 (and c_k growing like ln k), the tail is
        // below twice the next contribution: stop once that is below the working precision
        if ratio < 0.5 {
            let lc = log2_abs(&t).unwrap_or(i64::MIN / 2)
                + log2_abs(&base.abs().add(&hk.add(&hnk, wp, RM), wp, RM)).unwrap_or(0)
                + 1;
            if lc < biggest - wp as i64 - 4 {
                break;
            }
        }
        if k as usize > TERM_CAP {
            return None;
        }
    }
    let value = if k_kind {
        let s = if n % 2 == 1 { sum } else { sum.neg() }; // (-1)^(n+1)
        finite.div(&two, wp, RM).add(&s, wp, RM)
    } else {
        let pi = cc.pi(wp, RM);
        sum.sub(&finite, wp, RM).div(&pi, wp, RM)
    };
    Some((value, biggest, n + k))
}

/// Euler's constant at `p` bits, by Brent and McMillan's algorithm B1: with `A_0 = -ln N`,
/// `B_0 = 1`, `B_k = B_(k-1) N^2/k^2` and `A_k = (A_(k-1) N^2/k + B_k)/k`, `gamma` is
/// `sum A_k / sum B_k` within `pi e^(-4N)`; N is chosen for `2^-(p + 64)`. Every term is
/// read at `p + 64 + log2(terms)` bits; the sums are cut past `k = 2N`, where the terms fall
/// by more than four each, once a term is below the working precision. Cached per thread at
/// the largest precision asked.
pub(crate) fn euler_gamma(p: usize, cc: &mut Consts) -> BigFloat {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<(usize, BigFloat)>> =
            const { std::cell::RefCell::new(None) };
    }
    if let Some(g) = CACHE.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(q, _)| *q >= p)
            .map(|(_, g)| g.add(&small(0, p), p, RM))
    }) {
        return g;
    }
    let q = p.max(256);
    let wp = q + 80;
    let n = ((wp as f64 + 4.0) * std::f64::consts::LN_2 / 4.0).ceil() as u64 + 1;
    let nf = small(n, wp);
    let n2 = nf.mul(&nf, wp, RM);
    let mut a = nf.ln(wp, RM, cc).neg();
    let mut b = small(1, wp);
    let mut u = a.clone();
    let mut v = b.clone();
    let mut k: u64 = 1;
    loop {
        let kf = small(k, wp);
        b = b.mul(&n2, wp, RM).div(&kf.mul(&kf, wp, RM), wp, RM);
        a = a
            .mul(&n2, wp, RM)
            .div(&kf, wp, RM)
            .add(&b, wp, RM)
            .div(&kf, wp, RM);
        u = u.add(&a, wp, RM);
        v = v.add(&b, wp, RM);
        let small_b = log2_abs(&b).unwrap_or(i64::MIN / 2) < log2_abs(&v).unwrap_or(0) - wp as i64;
        let small_a = log2_abs(&a).unwrap_or(i64::MIN / 2) < log2_abs(&u).unwrap_or(0) - wp as i64;
        if k > 2 * n && small_b && small_a {
            break;
        }
        k += 1;
    }
    let g = u.div(&v, q, RM);
    CACHE.with(|c| *c.borrow_mut() = Some((q, g.clone())));
    g.add(&small(0, p), p, RM)
}

// ------------------------------------------------------------------- float64's cells

/// float64's (scipy's) value of `name` where an operand is NaN or infinite, for the
/// IEEE-faithful evaluators; `None` at finite operands (the kernels above read those).
pub(crate) fn scipy_cell(name: &str, nu: &BigFloat, x: &BigFloat, p: usize) -> Option<BigFloat> {
    let finite = |v: &BigFloat| !v.is_nan() && !v.is_inf();
    if finite(nu) && finite(x) {
        return None;
    }
    Some(match name {
        // yv: NaN off x >= 0; -inf at a zero argument; NaN at an infinite argument or order
        "bessely" => {
            if !nu.is_nan() && x.is_zero() {
                INF_NEG
            } else {
                NAN
            }
        }
        // iv: NaN at a NaN or infinite argument; 0 at an infinite order (SciPy reads -inf as an
        // integer order)
        "besseli" => {
            if nu.is_nan() || x.is_nan() || x.is_inf() {
                NAN
            } else {
                small(0, p)
            }
        }
        // kv: +inf at a zero argument whatever the order; NaN at a negative or NaN argument and
        // at an infinite order; 0 at +inf
        "besselk" => {
            if x.is_zero() {
                INF_POS
            } else if x.is_nan() || x.is_negative() || nu.is_nan() || nu.is_inf() {
                NAN
            } else {
                small(0, p)
            }
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use astro_float::Radix;

    fn bf(s: &str, p: usize, cc: &mut Consts) -> BigFloat {
        BigFloat::parse(s, Radix::Dec, p, RM, cc)
    }

    #[test]
    fn euler_gamma_reads_its_digits() {
        let mut cc = Consts::new().unwrap();
        // 250 digits (830 bits)
        let want = bf(
            "0.57721566490153286060651209008240243104215933593992359880576723488486772677766467\
             0936947063291746749514631447249807082480960504014486542836224173997644923536253500\
             3337429373377376739427925952582470949160087352039481656708532331517766115286211995\
             01508",
            900,
            &mut cc,
        );
        for p in [100usize, 300, 600] {
            let g = euler_gamma(p, &mut cc);
            let rel = g.sub(&want, 900, RM).div(&want, 900, RM).abs();
            assert!(
                log2_abs(&rel).unwrap_or(-10_000) <= -(p as i64),
                "p={p}: {g}"
            );
        }
    }
}
