//! High-precision re-certification of near-miss equivalence rows (OFFLINE miner).
//!
//! f64 is the fast PRE-FILTER for BOTH candidate arms; it has false REJECTIONS:
//! a true identity whose f64 evaluation loses precision fails the strict tolerance on
//! binding rows even though the exact values agree. The canonical casualty is
//! `atanh(tanh(x)) -> x`: for |x| in ~[10, 19.6] the f64 round-trip error reaches 2e-1
//! (tanh saturates toward 1, atanh amplifies the representation error), so a strict
//! rtol=1e-9 mine rejects an identity that dev_7-3 (mined at rtol=1e-5) had.
//! The constant-bearing arm has the same failure mode (`atanh(tanh(C*x)) -> C*x` at the
//! fitted constants), so a fit `NearMiss` verdict escalates here too: the
//! candidate is re-evaluated AT the f64-fitted constants -- a P-bit accept is still a
//! legitimate "constants EXIST" witness, so no new accept class is introduced.
//!
//! The rescue: when a candidate fails `allclose_extends` on only a small
//! fraction of its binding rows (a NEAR MISS), re-evaluate BOTH sides on exactly those
//! rows at `P` bits (astro-float, pure Rust) and re-judge with the same generic-equivalence
//! semantics (source nonfinite -> extendable; candidate nonfinite where the source is
//! finite -> reject; otherwise the same rtol/atol gate, in high precision). Every f64
//! ACCEPT is unchanged -- escalation can only rescue rows the exact math supports.
//!
//! Saturation at working precision is sound: where even `P` bits cannot separate a value
//! from its limit (e.g. tanh(x) == 1 at P bits for x > ~350), the source evaluates
//! nonfinite (atanh(1) -> inf) and the row becomes extendable -- never a wrong verdict,
//! only lost evidence. Plateau FALSITIES (e.g. `tanh(cosh(cosh(x))) -> 1`) are refuted by
//! their near-corner rows, which fail f64 by huge margins and exceed the near-miss gate,
//! so they are never escalated (and would be re-refuted at `P` bits if they were).
//!
//! Precision choice: P = 1024 bits (~308 decimal digits) decides every row the mine's
//! X mixture (|x| <= 1e3, f64 inputs lifted exactly) can produce outside doubly-
//! exponential separations, without an adaptive Ziv loop. Every precision has its own
//! saturation cliff (e.g. tanh rounds to 1 at P bits for x > ~350, and JUST BELOW that
//! cliff atanh(tanh(x)) recovers x with only the leftover mantissa bits), so a P-bit
//! REFUTATION is confirmed at `2*P` bits before it counts: only an agreeing FAIL at both
//! precisions refutes; a disagreement is UNDECIDED and never flips a verdict on its own.
//!
//! Design lesson (kept from a removed exact-arithmetic cross-check, `certify_const_free`):
//! a checker judging
//! FINER than the mine tolerance rejects true rules -- `np.e`/`np.pi` are f64 LITERALS, so
//! `pow(np.e, x) -> exp(x)` holds only to ~1e-16*|x| in exact arithmetic. The mine tolerance
//! IS the system's definition of equivalence; verify external (LLM) proposals AT that tolerance.

use crate::operators::Operators;
use astro_float::{BigFloat, Consts, RoundingMode, INF_NEG, INF_POS, NAN};

/// Working precision in bits (refutations are confirmed at `2*P`).
const P: usize = 1024;
/// BASE precision for the pole-vs-saturation escalation LADDER (`judge_row` climbs P_INF ->
/// 4*P_INF -> 16*P_INF). `tanh(x)=1` at working precision needs ~2*|x|*log10(e) digits, so the
/// mine X (|x| <= ~1e3) needs ~2900 bits, cleared at 4096; the SCALED family `atanh(tanh(k*x))`
/// (k up to 5) needs up to ~5x that, cleared only higher up the ladder (a fixed 4096 dropped
/// those true identities). A genuine pole/literal inf stays inf at every rung; only doubly-
/// exponential saturations (`atanh(tanh(cosh(x)))`) never resolve and are treated as poles.
const P_INF: usize = 4096;
const RM: RoundingMode = RoundingMode::ToEven;
/// Escalate a candidate only when it fails f64 on at most this fraction of the instance's
/// binding rows (the rows where the source is finite). This is the SINGLE near-miss gate:
/// it selects candidates that are close to the source on the finite evidence, and because
/// `failing <= FRAC * binding <= FRAC * n_rows`, it also bounds the high-precision work per
/// instance to a fraction of the X -- no separate absolute row cap is needed (a fixed cap
/// sized for the 1024-row mine X wrongly rejected true rules on the 2048-row confirm X).
/// Wide misses (plateau falsities) fail by huge margins, exceed the gate, and never escalate.
/// The escalation cost stays bounded: only near-miss instances escalate at all.
/// Experiment override: env `SIMPLIPY_HIPREC_FRAC` (calibration only, not a config surface).
const NEAR_MISS_FRAC: f64 = 0.5;

fn near_miss_frac() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SIMPLIPY_HIPREC_FRAC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(NEAR_MISS_FRAC)
    })
}

/// Verdict of one row at one precision. `Extend` = source nonfinite (domain-extendable),
/// exactly `allclose_extends`' semantics lifted to `p` bits.
#[derive(PartialEq, Eq, Clone, Copy)]
enum RowVerdict {
    Pass,
    Extend,
    Fail,
}

/// Finite `s ~= c` within (rtol, atol) at `p` bits.
fn cmp_finite(s: &BigFloat, c: &BigFloat, rtol: f64, atol: f64, p: usize) -> RowVerdict {
    let diff = s.sub(c, p, RM).abs();
    let tol =
        BigFloat::from_f64(atol, p).add(&BigFloat::from_f64(rtol, p).mul(&s.abs(), p, RM), p, RM);
    match diff.cmp(&tol) {
        Some(ord) if ord <= 0 => RowVerdict::Pass,
        _ => RowVerdict::Fail,
    }
}

/// `BigFloat` has no `is_finite`; a value is finite when it is neither inf nor NaN.
fn is_fin(x: &BigFloat) -> bool {
    !x.is_inf() && !x.is_nan()
}

/// Judge row `r` at `p` bits. `None` = unevaluable (unknown token): the caller keeps the
/// verdict it already has.
///
/// The inf verdict is by ESCALATION, not a per-source flag: a source-inf row is
/// re-evaluated at `P_INF` bits (matched to the mine X range, |x| <= ~1e3). A GENUINE infinity
/// -- literal inf, or a pole (`1/0`, `0^-k`) -- is EXACT and stays inf at any precision, so the
/// candidate must reproduce it (`pow(x, inf) -> 0` and `pow(0, cos x) -> 0` are false where the
/// source is that pole/inf). A SATURATION inf -- `atanh(tanh(x)) = inf` only because `tanh(x)`
/// rounds to 1 -- resolves to its finite true value at `P_INF` and is judged as a finite row,
/// keeping the tanh-saturation rescue. NaN is always domain-extendable (`x/x -> 1`).
#[allow(clippy::too_many_arguments)]
fn judge_row(
    source: &[String],
    src_params: &[f64],
    candidate: &[String],
    cand_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    row: usize,
    f64_src: f64,
    f64_cand: f64,
    rtol: f64,
    atol: f64,
    p: usize,
    cc: &mut Consts,
) -> Option<RowVerdict> {
    // DEPLOYMENT-FAITHFUL: the extend/pole decision keys on the f64 value (numpy = the DEPLOYED
    // evaluator), NOT the astro-float value. astro-float diverges from numpy on quirk domains --
    // `(-inf)^non-integer` is NaN in astro-float (complex-undefined) but +inf in numpy -- and
    // keying `Extend` on the astro-float NaN wrongly dropped numpy-DEFINED inf rows, which let
    // FRAC > 0.5 accept deployment-false rules (`pow(-inf, sin x) -> pow(-inf, x)`). Keying on the
    // f64 value makes soundness FRAC-independent: escalation can only REFINE a numpy-inf that is a
    // saturation artifact, never reinterpret a numpy-defined value.
    if f64_src.is_nan() {
        return Some(RowVerdict::Extend); // numpy-undefined -> domain-extendable (`x/x -> 1`)
    }
    if f64_src.is_infinite() {
        // numpy sees inf. Climb the ladder: a SATURATION artifact resolves to a finite true value
        // (`atanh(tanh(k*x)) = inf` only because `tanh` rounds to 1 -> the rule may FIX it to the
        // finite form); a GENUINE pole (`1/0`, `0^-k`, literal inf) stays inf; a numpy-quirk inf
        // (astro-float NaN where numpy defines inf) has no finite true value, so the DEPLOYMENT inf
        // is authoritative and the candidate must reproduce it. Doubly-exponential saturations
        // (`atanh(tanh(cosh x))`) never resolve and are treated as poles (rare, documented).
        for &pp in &[P_INF, 4 * P_INF, 16 * P_INF] {
            let s_hi = eval_prefix(source, ops, var_names, x_cols, src_params, row, pp, cc)?;
            if is_fin(&s_hi) {
                let c_hi =
                    eval_prefix(candidate, ops, var_names, x_cols, cand_params, row, pp, cc)?;
                return Some(if is_fin(&c_hi) {
                    cmp_finite(&s_hi, &c_hi, rtol, atol, pp)
                } else {
                    RowVerdict::Fail
                });
            }
            if s_hi.is_nan() {
                break; // astro-float undefined but numpy-inf: a numpy-quirk pole (handled below)
            }
            // s_hi still inf: a higher rung may yet resolve a deep saturation
        }
        // pole / numpy-quirk inf: the candidate must reproduce the DEPLOYMENT infinity (same sign)
        return Some(if f64_cand == f64_src {
            RowVerdict::Pass
        } else {
            RowVerdict::Fail
        });
    }
    // numpy-finite source: high-precision comparison of the near-miss row
    let s = eval_prefix(source, ops, var_names, x_cols, src_params, row, p, cc)?;
    if !is_fin(&s) {
        return Some(RowVerdict::Fail); // astro-float diverges from a finite numpy value -> keep the fail
    }
    let c = eval_prefix(candidate, ops, var_names, x_cols, cand_params, row, p, cc)?;
    if !is_fin(&c) {
        return Some(RowVerdict::Fail); // replacement non-finite where the source is finite
    }
    Some(cmp_finite(&s, &c, rtol, atol, p))
}

/// Judge row `r` with the precision-stability rule: a `P`-bit FAIL only counts once the
/// row ALSO fails at `2*P` bits (each precision has its own saturation cliff; a
/// disagreement is treated as non-refuting). `None` propagates (caller keeps its verdict).
#[allow(clippy::too_many_arguments)]
fn judge_row_confirmed(
    source: &[String],
    src_params: &[f64],
    candidate: &[String],
    cand_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    row: usize,
    f64_src: f64,
    f64_cand: f64,
    rtol: f64,
    atol: f64,
    cc: &mut Consts,
) -> Option<RowVerdict> {
    let v = judge_row(
        source,
        src_params,
        candidate,
        cand_params,
        ops,
        var_names,
        x_cols,
        row,
        f64_src,
        f64_cand,
        rtol,
        atol,
        P,
        cc,
    )?;
    if v != RowVerdict::Fail {
        return Some(v);
    }
    match judge_row(
        source,
        src_params,
        candidate,
        cand_params,
        ops,
        var_names,
        x_cols,
        row,
        f64_src,
        f64_cand,
        rtol,
        atol,
        2 * P,
        cc,
    ) {
        Some(RowVerdict::Fail) | None => Some(RowVerdict::Fail),
        Some(other) => Some(other), // P-bit cliff artifact: the higher precision decides
    }
}

/// The near-miss rescue: `y_src` / `y_cand` are the instance's f64 evaluations, `source`
/// carries `src_params` for its `<constant>` slots (left-to-right, the tape order), the
/// candidate carries `cand_params` (empty for the const-free arm; the f64-FITTED constants
/// for the fit arm's `FitVerdict::NearMiss`). Returns `true` iff every f64-failing binding
/// row passes at `P` bits. Never called on an f64 accept; a `false` leaves the f64 verdict
/// unchanged.
#[allow(clippy::too_many_arguments)]
pub fn rescue(
    y_src: &[f64],
    y_cand: &[f64],
    source: &[String],
    src_params: &[f64],
    candidate: &[String],
    cand_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    rtol: f64,
    atol: f64,
) -> bool {
    let mut binding = 0usize;
    let mut failing: Vec<usize> = Vec::new();
    for (r, &s) in y_src.iter().enumerate() {
        let c = y_cand[r];
        if s.is_nan() {
            continue; // undefined -> extendable, not evidence
        }
        if s.is_infinite() {
            // A source-inf row where the candidate does not already reproduce the same inf is a
            // FAILING row -- escalate. The P_INF re-judge inside `judge_row` then decides pole
            // (stays inf -> must-match -> Fail) vs saturation (resolves finite -> compare). Not
            // counted as finite binding evidence.
            if c != s {
                failing.push(r);
            }
            continue;
        }
        binding += 1;
        if !c.is_finite() || (s - c).abs() > atol + rtol * s.abs() {
            failing.push(r);
        }
    }
    if failing.is_empty() || (failing.len() as f64) > near_miss_frac() * (binding as f64) {
        return false; // not a near miss (or nothing to rescue): keep the f64 verdict
    }
    let mut cc = match Consts::new() {
        Ok(cc) => cc,
        Err(_) => return false,
    };
    for &r in &failing {
        match judge_row_confirmed(
            source,
            src_params,
            candidate,
            cand_params,
            ops,
            var_names,
            x_cols,
            r,
            y_src[r],
            y_cand[r],
            rtol,
            atol,
            &mut cc,
        ) {
            Some(RowVerdict::Pass) | Some(RowVerdict::Extend) => {}
            _ => return false, // confirmed fail, or unevaluable: keep the f64 verdict
        }
    }
    true
}

/// Evaluate a prefix expression at `p` bits on row `r`. Leaves: variables (index into
/// `x_cols`), `<constant>` (consumes `params` left-to-right, the tape's slot order),
/// valued literals (`numeric::leaf_value`, lifted exactly from f64). Returns `None` for
/// an unknown operator/leaf (caller falls back to the f64 verdict). Operator semantics
/// mirror `numeric.rs`'s f64 kernels, with explicit domain guards where astro-float and
/// libm could disagree.
#[allow(clippy::too_many_arguments)]
fn eval_prefix(
    tokens: &[String],
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    params: &[f64],
    row: usize,
    p: usize,
    cc: &mut Consts,
) -> Option<BigFloat> {
    let mut idx = 0usize;
    let mut next_param = 0usize;
    let v = eval_node(
        tokens,
        &mut idx,
        ops,
        var_names,
        x_cols,
        params,
        &mut next_param,
        row,
        p,
        cc,
    )?;
    if idx != tokens.len() {
        return None;
    }
    Some(v)
}

#[allow(clippy::too_many_arguments)]
fn eval_node(
    tokens: &[String],
    idx: &mut usize,
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    params: &[f64],
    next_param: &mut usize,
    row: usize,
    p: usize,
    cc: &mut Consts,
) -> Option<BigFloat> {
    let tok = tokens.get(*idx)?.clone();
    *idx += 1;
    if let Some(arity) = ops.arity_of(&tok) {
        let a = eval_node(
            tokens, idx, ops, var_names, x_cols, params, next_param, row, p, cc,
        )?;
        if arity == 1 {
            return apply_unary(&tok, &a, p, cc);
        }
        let b = eval_node(
            tokens, idx, ops, var_names, x_cols, params, next_param, row, p, cc,
        )?;
        return apply_binary(&tok, &a, &b, p, cc);
    }
    if tok == "<constant>" {
        let v = *params.get(*next_param)?;
        *next_param += 1;
        return Some(BigFloat::from_f64(v, p));
    }
    if let Some(c) = var_names.iter().position(|v| *v == tok) {
        return Some(BigFloat::from_f64(x_cols.get(c)?[row], p));
    }
    // Literals denote their TRANSCENDENTAL values, and the f64 rendering is measurement
    // error. Lifting `np.pi` from f64 injects a ~1.2e-16 denotation error into every
    // pi/e-bearing judgment at THIS, the arbiter precision -- the one place in the miner that
    // claims exactness. The f64 pre-filter may keep the dyadic (it has no verdict authority);
    // the arbiter may not.
    match tok.as_str() {
        "np.pi" => Some(cc.pi(p, RM)),
        "np.e" => Some(cc.e(p, RM)),
        _ => crate::numeric::leaf_value(&tok).map(|v| BigFloat::from_f64(v, p)),
    }
}

fn small_int(n: u64, p: usize) -> BigFloat {
    BigFloat::from_u64(n, p)
}

fn one(p: usize) -> BigFloat {
    small_int(1, p)
}

/// `x^(1/n)` for ODD n, sign-folded like `numeric::real_odd_root` (always real).
fn odd_root(x: &BigFloat, n: u64, p: usize, cc: &mut Consts) -> BigFloat {
    if n == 3 {
        return x.cbrt(p, RM);
    }
    let r = one(p).div(&small_int(n, p), p, RM);
    if x.is_negative() {
        pow_pos_base(&x.neg(), &r, p, cc).neg()
    } else {
        pow_pos_base(x, &r, p, cc)
    }
}

/// `a^b` for a > 0 finite, b finite, as `exp(b * ln(a))` at `p + 64` bits. NEVER call
/// astro-float's `BigFloat::pow` here: it escalates its INTERNAL working precision
/// (Ziv-style correct-rounding retry) on hard-to-round (base, exponent) pairs -- observed
/// grinding for tens of MINUTES inside `ln_series`/FFT mantissa multiplication on
/// mine-realistic inputs (`pow(<constant>, div5(x2))`) and single-handedly stalling
/// rayon workers. `ln`/`exp` at an explicit
/// precision stay bounded, and `p + 64` guard bits keep the relative error orders of
/// magnitude below every tolerance used in this module (>= CERT_ATOL = 1e-280), which is
/// all the tolerance-gated judging needs -- correctly-rounded pow was never required.
fn pow_pos_base(a: &BigFloat, b: &BigFloat, p: usize, cc: &mut Consts) -> BigFloat {
    let g = p + 64;
    let l = a.ln(g, RM, cc);
    b.mul(&l, g, RM).exp(g, RM, cc)
}

fn apply_unary(name: &str, x: &BigFloat, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    if x.is_nan() {
        return Some(NAN);
    }
    let v = match name {
        "neg" => x.neg(),
        "abs" => x.abs(),
        "inv" => {
            if x.is_zero() {
                if x.is_negative() {
                    INF_NEG
                } else {
                    INF_POS
                }
            } else {
                one(p).div(x, p, RM)
            }
        }
        "mult2" => x.mul(&small_int(2, p), p, RM),
        "mult3" => x.mul(&small_int(3, p), p, RM),
        "mult4" => x.mul(&small_int(4, p), p, RM),
        "mult5" => x.mul(&small_int(5, p), p, RM),
        "div2" => x.div(&small_int(2, p), p, RM),
        "div3" => x.div(&small_int(3, p), p, RM),
        "div4" => x.div(&small_int(4, p), p, RM),
        "div5" => x.div(&small_int(5, p), p, RM),
        "pow2" => x.powi(2, p, RM),
        "pow3" => x.powi(3, p, RM),
        "pow4" => x.powi(4, p, RM),
        "pow5" => x.powi(5, p, RM),
        // pow1_2 / pow1_4 = C pow(x, 0.5/0.25): negative base -> NaN.
        "pow1_2" => {
            if x.is_negative() && !x.is_zero() {
                NAN
            } else {
                x.sqrt(p, RM)
            }
        }
        "pow1_4" => {
            if x.is_negative() && !x.is_zero() {
                NAN
            } else {
                x.sqrt(p, RM).sqrt(p, RM)
            }
        }
        "pow1_3" => odd_root(x, 3, p, cc),
        "pow1_5" => odd_root(x, 5, p, cc),
        "sin" => x.sin(p, RM, cc),
        "cos" => x.cos(p, RM, cc),
        "tan" => x.tan(p, RM, cc),
        "asin" => {
            if x.abs().cmp(&one(p)).is_some_and(|o| o > 0) {
                NAN
            } else {
                x.asin(p, RM, cc)
            }
        }
        "acos" => {
            if x.abs().cmp(&one(p)).is_some_and(|o| o > 0) {
                NAN
            } else {
                x.acos(p, RM, cc)
            }
        }
        "atan" => x.atan(p, RM, cc),
        "sinh" => x.sinh(p, RM, cc),
        "cosh" => x.cosh(p, RM, cc),
        "tanh" => x.tanh(p, RM, cc),
        "asinh" => x.asinh(p, RM, cc),
        "acosh" => {
            if x.cmp(&one(p)).is_some_and(|o| o < 0) {
                NAN
            } else {
                x.acosh(p, RM, cc)
            }
        }
        "atanh" => {
            let a = x.abs().cmp(&one(p));
            match a {
                Some(o) if o > 0 => NAN,
                Some(0) => {
                    if x.is_negative() {
                        INF_NEG
                    } else {
                        INF_POS
                    }
                }
                _ => x.atanh(p, RM, cc),
            }
        }
        "exp" => x.exp(p, RM, cc),
        "log" => {
            if x.is_zero() {
                INF_NEG
            } else if x.is_negative() {
                NAN
            } else {
                x.ln(p, RM, cc)
            }
        }
        _ => return None,
    };
    Some(v)
}

fn apply_binary(
    name: &str,
    a: &BigFloat,
    b: &BigFloat,
    p: usize,
    cc: &mut Consts,
) -> Option<BigFloat> {
    if a.is_nan() || b.is_nan() {
        return Some(NAN);
    }
    let v = match name {
        "+" => a.add(b, p, RM),
        "-" => a.sub(b, p, RM),
        "*" => a.mul(b, p, RM),
        // IEEE `x / y`: zero divisor -> signed inf (the ZERO's sign participates); 0/0 -> nan.
        "/" => {
            if b.is_zero() {
                if a.is_zero() {
                    NAN
                } else {
                    let neg = a.is_negative() != b.is_negative();
                    if neg {
                        INF_NEG
                    } else {
                        INF_POS
                    }
                }
            } else {
                a.div(b, p, RM)
            }
        }
        // C pow: negative base with a non-integer exponent -> NaN; 0^negative -> +inf;
        // 0^0 -> 1 (C99). Everything else goes through `pow_pos_base` (exp(b*ln(a)) at
        // explicit precision) -- NOT astro-float's pow, whose internal correct-rounding
        // retry can stall for minutes on hard-to-round pairs (see pow_pos_base).
        "pow" => {
            if a.is_zero() {
                if b.is_zero() {
                    one(p)
                } else if b.is_negative() {
                    INF_POS
                } else {
                    small_int(0, p)
                }
            } else if a.is_negative() && !b.is_int() && !b.is_inf() {
                NAN
            } else if a.is_negative() && b.is_int() {
                // astro-float's pow is NaN for a < 0; emulate the real integer-exponent
                // result by sign folding (parity of b) over the positive-base power.
                let mag = pow_pos_base(&a.neg(), b, p, cc);
                let odd = b
                    .rem(&small_int(2, p))
                    .abs()
                    .cmp(&one(p))
                    .is_some_and(|o| o == 0);
                if odd {
                    mag.neg()
                } else {
                    mag
                }
            } else {
                pow_pos_base(a, b, p, cc) // see pow_pos_base: BigFloat::pow can stall
            }
        }
        _ => return None,
    };
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// `np.pi` denotes PI, not its f64 rendering. Lifting the
    /// literal from f64 made `sin(np.pi)` read 1.2246e-16 at EVERY precision -- the arbiter
    /// inherited the denotation error and no rung could remove it. Rendered at working precision,
    /// sin(pi at 256 bits) is ~1e-77: the arbiter can finally see that the identity is exact.
    #[test]
    fn np_pi_is_rendered_at_working_precision() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let mut cc = Consts::new().unwrap();
        let p = 256usize;
        let v = eval_prefix(&s(&["sin", "np.pi"]), ops, &[], &[], &[], 0, p, &mut cc)
            .expect("sin(np.pi) must evaluate");
        let tiny = BigFloat::from_f64(1e-70, p);
        // astro-float cmp returns Option<i128> (the sign of self - other)
        assert!(
            matches!(v.abs().cmp(&tiny), Some(sig) if sig < 0),
            "sin(np.pi) at 256 bits should be ~1e-77, got {v:?}"
        );
    }

    /// atanh(tanh(x)) -> x: the motivating rescue. Rows chosen inside the f64 dead zone
    /// (|x| in [10, 19.6]: f64 fails rtol=1e-9 by up to 2e-1) must PASS at P bits.
    #[test]
    fn rescues_atanh_tanh_identity() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // mostly well-conditioned rows (f64 passes) + a few dead-zone rows (f64 fails):
        // the shape a real mine instance has, inside the near-miss gate.
        let mut xs: Vec<f64> = (0..40).map(|i| -4.0 + 0.2 * i as f64).collect();
        xs.extend([12.0, 15.0, 18.91, -11.5]);
        let cols = vec![xs.clone()];
        let y_src: Vec<f64> = xs.iter().map(|&x| x.tanh().atanh()).collect();
        let y_cand = xs.clone();
        // sanity: the dead-zone rows really do fail in f64 at the mine tolerance
        let n_fail = y_src
            .iter()
            .zip(&y_cand)
            .filter(|(s_, c)| s_.is_finite() && (*s_ - **c).abs() > 1e-12 + 1e-9 * s_.abs())
            .count();
        assert!(
            n_fail >= 3,
            "expected dead-zone rows to fail f64, got {n_fail}"
        );
        let ok = rescue(
            &y_src,
            &y_cand,
            &s(&["atanh", "tanh", "x0"]),
            &[],
            &s(&["x0"]),
            &[],
            ops,
            &["x0".to_string()],
            &cols,
            1e-9,
            1e-12,
        );
        assert!(ok, "true identity must be rescued at P bits");
    }

    /// A genuinely wrong near-miss must stay rejected: tanh(x) vs x on small rows where
    /// f64 is exact -- the escalation gate may fire, but P-bit evaluation re-refutes it.
    #[test]
    fn does_not_rescue_falsity() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let xs = [0.4, 0.5, -0.3, 0.001, -0.002, 0.0005];
        let cols = vec![xs.to_vec()];
        let y_src: Vec<f64> = xs.iter().map(|&x: &f64| x.tanh()).collect();
        let y_cand = xs.to_vec();
        let ok = rescue(
            &y_src,
            &y_cand,
            &s(&["tanh", "x0"]),
            &[],
            &s(&["x0"]),
            &[],
            ops,
            &["x0".to_string()],
            &cols,
            1e-9,
            1e-12,
        );
        assert!(!ok, "tanh(x) != x must stay rejected");
    }

    /// Saturated-at-P rows are extendable, not wrong: atanh(tanh(400)) is inf at 1024
    /// bits (tanh(400) rounds to 1), so the row passes by extension.
    #[test]
    fn saturated_rows_extend() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // well-conditioned rows + one rescuable row + one P-bit-saturated row; f64 marks
        // 400.0 nonbinding (tanh = 1.0 -> atanh = inf) and 15.0 binding-but-failing.
        let mut xs: Vec<f64> = (0..20).map(|i| -2.0 + 0.2 * i as f64).collect();
        xs.extend([15.0, 400.0]);
        let cols = vec![xs.clone()];
        let y_src: Vec<f64> = xs.iter().map(|&x| x.tanh().atanh()).collect();
        assert!(y_src.last().unwrap().is_infinite());
        let y_cand = xs.clone();
        let ok = rescue(
            &y_src,
            &y_cand,
            &s(&["atanh", "tanh", "x0"]),
            &[],
            &s(&["x0"]),
            &[],
            ops,
            &["x0".to_string()],
            &cols,
            1e-9,
            1e-12,
        );
        assert!(ok);
    }

    /// Wide-miss candidates are never escalated (cost + correctness guard).
    #[test]
    fn wide_miss_not_escalated() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let xs: Vec<f64> = (0..64).map(|i| i as f64 / 4.0).collect();
        let cols = vec![xs.clone()];
        let y_src: Vec<f64> = xs.iter().map(|&x| x + 1.0).collect(); // x+1 vs x: all rows fail
        let y_cand = xs.clone();
        let ok = rescue(
            &y_src,
            &y_cand,
            &s(&["+", "x0", "1"]),
            &[],
            &s(&["x0"]),
            &[],
            ops,
            &["x0".to_string()],
            &cols,
            1e-9,
            1e-12,
        );
        assert!(!ok);
    }

    /// Cost anchor for the precision-policy decision (run explicitly:
    /// `cargo test --release hiprec_cost_bench -- --ignored --nocapture`):
    /// per-row eval cost of the f64 kernel vs the P=1024-bit kernel on
    /// mining-representative expressions. Not a correctness test.
    #[test]
    #[ignore]
    fn hiprec_cost_bench() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let n_rows = 1024usize;
        let xs: Vec<f64> = (0..n_rows)
            .map(|i| -5.0 + 10.0 * (i as f64) / (n_rows as f64))
            .collect();
        let cols = vec![xs.clone(), xs.iter().map(|v| v * 0.37 + 0.11).collect()];
        let var_names = ["x0".to_string(), "x1".to_string()];
        let cases: &[(&str, Vec<String>, Vec<f64>)] = &[
            (
                "arith  + * x0 x1 x0",
                s(&["+", "*", "x0", "x1", "x0"]),
                vec![],
            ),
            ("trig   atanh tanh x0", s(&["atanh", "tanh", "x0"]), vec![]),
            (
                "pow    pow abs x0 x1",
                s(&["pow", "abs", "x0", "x1"]),
                vec![],
            ),
            (
                "mixed  * <constant> cos x1",
                s(&["*", "<constant>", "cos", "x1"]),
                vec![1.7],
            ),
        ];
        let x_flat: Vec<f64> = (0..n_rows)
            .flat_map(|r| cols.iter().map(move |c| c[r]).collect::<Vec<f64>>())
            .collect();
        println!("rows={n_rows}  P={P} bits");
        for (label, tokens, params) in cases {
            // f64 kernel: repeat to get a measurable interval
            let reps = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let y =
                    crate::eval::evaluate_batch(ops, tokens, &var_names, &x_flat, n_rows, params)
                        .expect("f64 eval");
                std::hint::black_box(y);
            }
            let f64_ns = t0.elapsed().as_nanos() as f64 / (reps * n_rows) as f64;
            // P-bit kernel
            let mut cc = Consts::new().unwrap();
            let t0 = std::time::Instant::now();
            for r in 0..n_rows {
                let v = eval_prefix(tokens, ops, &var_names, &cols, params, r, P, &mut cc);
                std::hint::black_box(v);
            }
            let hp_ns = t0.elapsed().as_nanos() as f64 / n_rows as f64;
            println!(
                "{label:28} f64 {f64_ns:9.1} ns/row   P-bit {hp_ns:12.1} ns/row   ratio {:8.0}x",
                hp_ns / f64_ns
            );
        }
    }
}

#[cfg(test)]
mod magnitude_probe {
    use super::*;

    /// Diagnostic (run explicitly): does astro-float handle beyond-f64 magnitudes
    /// cheaply (early Inf) or grind? Timed per case.
    #[test]
    #[ignore]
    fn hiprec_huge_magnitude_probe() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let cases: &[(&str, Vec<&str>, f64)] = &[
            ("cosh(cosh(x)) x=27", vec!["cosh", "cosh", "x0"], 27.0),
            (
                "tanh(cosh(cosh(x))) x=4",
                vec!["tanh", "cosh", "cosh", "x0"],
                4.0,
            ),
            ("exp(exp(x)) x=30", vec!["exp", "exp", "x0"], 30.0),
            ("exp(exp(exp(x))) x=3", vec!["exp", "exp", "exp", "x0"], 3.0),
            (
                "pow(cosh(x),cosh(x)) x=27",
                vec!["pow", "cosh", "x0", "cosh", "x0"],
                27.0,
            ),
        ];
        for (label, toks, x) in cases {
            let tokens: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
            let cols = vec![vec![*x]];
            let mut cc = Consts::new().unwrap();
            let t0 = std::time::Instant::now();
            let v = eval_prefix(&tokens, ops, &["x0".to_string()], &cols, &[], 0, P, &mut cc);
            let dt = t0.elapsed().as_secs_f64();
            let desc = match &v {
                Some(b) if b.is_inf() => "inf".to_string(),
                Some(b) if b.is_nan() => "nan".to_string(),
                Some(_) => "finite".to_string(),
                None => "none".to_string(),
            };
            println!("{label:34} {dt:9.3}s -> {desc}");
        }
    }
}
