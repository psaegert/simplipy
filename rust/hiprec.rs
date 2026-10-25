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

mod besselj;
mod special;

/// Re-export for callers that rank arbiter values without naming the backing crate
/// (worker's literal resolution).
pub use astro_float::BigFloat as ArbFloat;

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
///
/// The gate counts EVALUATION POINTS: every row where the source is defined (finite or
/// infinite) is a point of the denominator, and a failing infinite row is a failing point
/// like a finite one. A pair with no variable has ONE point however many rows repeat it;
/// "at most half the points, rounded up" then escalates it, at the cost of one row.
pub const NEAR_MISS_FRAC: f64 = 0.5;

/// Verdict of one row at one precision. `Extend` = source nonfinite (domain-extendable),
/// exactly `allclose_extends`' semantics lifted to `p` bits.
#[derive(PartialEq, Eq, Clone, Copy)]
enum RowVerdict {
    Pass,
    Extend,
    Fail,
}

/// `|a - b| <= atol + rtol*|b|` at `p` bits.
fn within(a: &BigFloat, b: &BigFloat, rtol: f64, atol: f64, p: usize) -> bool {
    let diff = a.sub(b, p, RM).abs();
    let tol =
        BigFloat::from_f64(atol, p).add(&BigFloat::from_f64(rtol, p).mul(&b.abs(), p, RM), p, RM);
    matches!(diff.cmp(&tol), Some(ord) if ord <= 0)
}

/// Finite `s ~= c` within `atol + rtol*|c|` at `p` bits -- the f64 row predicate of
/// `eval::allclose_extends`, lifted (the candidate is the reference). A `Fail` carries the
/// residual `|s - c|` and both values, which `judge_row_confirmed` compares across precisions.
fn cmp_finite(s: &BigFloat, c: &BigFloat, rtol: f64, atol: f64, p: usize) -> Judged {
    if within(s, c, rtol, atol, p) {
        return (RowVerdict::Pass, None);
    }
    let residual = Residual {
        diff: s.sub(c, p, RM).abs(),
        s: s.clone(),
        c: c.clone(),
    };
    (RowVerdict::Fail, Some(residual))
}

/// A failed finite comparison: the residual `|s - c|` and the two values it came from.
struct Residual {
    diff: BigFloat,
    s: BigFloat,
    c: BigFloat,
}

/// A row verdict at one precision, with the residual of a failed finite comparison.
type Judged = (RowVerdict, Option<Residual>);

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
) -> Option<Judged> {
    // DEPLOYMENT-FAITHFUL: the extend/pole decision keys on the f64 value (numpy = the DEPLOYED
    // evaluator), NOT the astro-float value. astro-float diverges from numpy on quirk domains --
    // `(-inf)^non-integer` is NaN in astro-float (complex-undefined) but +inf in numpy -- and
    // keying `Extend` on the astro-float NaN wrongly dropped numpy-DEFINED inf rows, which let
    // FRAC > 0.5 accept deployment-false rules (`pow(-inf, sin x) -> pow(-inf, x)`). Keying on the
    // f64 value makes soundness FRAC-independent: escalation can only REFINE a numpy-inf that is a
    // saturation artifact, never reinterpret a numpy-defined value.
    if f64_src.is_nan() {
        return Some((RowVerdict::Extend, None)); // numpy-undefined -> domain-extendable (`x/x -> 1`)
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
                    (RowVerdict::Fail, None)
                });
            }
            if s_hi.is_nan() {
                break; // astro-float undefined but numpy-inf: a numpy-quirk pole (handled below)
            }
            // s_hi still inf: a higher rung may yet resolve a deep saturation
        }
        // pole / numpy-quirk inf: the candidate must reproduce the DEPLOYMENT infinity (same sign)
        return Some(if f64_cand == f64_src {
            (RowVerdict::Pass, None)
        } else {
            (RowVerdict::Fail, None)
        });
    }
    // numpy-finite source: high-precision comparison of the near-miss row
    let s = eval_prefix(source, ops, var_names, x_cols, src_params, row, p, cc)?;
    if !is_fin(&s) {
        // astro-float diverges from a finite numpy value -> keep the fail
        return Some((RowVerdict::Fail, None));
    }
    let c = eval_prefix(candidate, ops, var_names, x_cols, cand_params, row, p, cc)?;
    if !is_fin(&c) {
        return Some((RowVerdict::Fail, None)); // replacement non-finite where the source is finite
    }
    Some(cmp_finite(&s, &c, rtol, atol, p))
}

/// Judge row `r` with the precision-stability rule: a `P`-bit FAIL only counts once the
/// row ALSO fails at `2*P` bits (each precision has its own saturation cliff; a
/// disagreement is treated as non-refuting). `None` propagates (caller keeps its verdict).
///
/// WHAT ZERO MEANS at high precision. A true identity whose values are zero (`sin(pi) ->
/// 0`, pi rendered at the working precision) leaves a residual of rounding residue, about
/// `2^-P` at `P` bits and `2^-2P` at `2P` bits: no relative band admits it at either
/// precision, yet it is exactly 0. A FALSE zero (`exp(sinh(-5)) -> 0`, value 5.9e-33)
/// leaves the same residual at every precision. So a row that fails at both precisions
/// passes iff its residual COLLAPSES as the precision doubles: it shrinks by at least
/// `2^(P/2)`, the precision-stability discriminator the symbolic judge applies.
///
/// A REFUTATION NEEDS CONVERGED VALUES. A row whose source or candidate value itself moves
/// between `P` and `2P` bits by more than the tolerance has not been evaluated yet: an inner
/// saturation (`tanh(x)` is exactly -1 at `P` and at `2P` bits once |x| passes ~355 and ~710,
/// so `sin(acos(tanh(x)))` reads `sin(pi)`'s residue, ~2^-P, against `sech(x)` ~ 1e-407 at
/// x = -937) refutes a true identity at both precisions while its value halves its exponent.
/// Such a row is judged again at `P_INF` and `2*P_INF` bits, the precision matched to the mine
/// X range (|x| <= ~1e3), by the same rule; a row still unconverged there fails (fail closed).
/// Only a finite source row climbs: an infinite one is decided by `judge_row`'s own ladder.
/// A converged false row (`exp(sinh(-5)) -> 0`) is refuted at `2P` and pays nothing more.
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
    let pairs: &[usize] = if f64_src.is_finite() {
        &[P, P_INF]
    } else {
        &[P]
    };
    for &p in pairs {
        let judge = |pp: usize, cc: &mut Consts| {
            judge_row(
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
                pp,
                cc,
            )
        };
        let (v, r_p) = judge(p, cc)?;
        if v != RowVerdict::Fail {
            return Some(v);
        }
        match judge(2 * p, cc) {
            Some((RowVerdict::Fail, Some(r_2p))) => {
                let Some(r_p) = r_p else {
                    return Some(RowVerdict::Fail);
                };
                let bound = r_p.diff.mul(
                    &BigFloat::from_f64(2f64.powi(-((p / 2) as i32)), 2 * p),
                    2 * p,
                    RM,
                );
                if matches!(r_2p.diff.cmp(&bound), Some(ord) if ord <= 0) {
                    return Some(RowVerdict::Pass); // the residual collapsed: an exact zero
                }
                let converged = within(&r_p.s, &r_2p.s, rtol, atol, 2 * p)
                    && within(&r_p.c, &r_2p.c, rtol, atol, 2 * p);
                if converged {
                    return Some(RowVerdict::Fail);
                }
                // unconverged: the next pair decides
            }
            Some((RowVerdict::Fail, None)) | None => return Some(RowVerdict::Fail),
            Some((other, _)) => return Some(other), // P-bit cliff artifact: the higher precision decides
        }
    }
    Some(RowVerdict::Fail)
}

/// The near-miss rescue: `y_src` / `y_cand` are the instance's f64 evaluations, `source`
/// carries `src_params` for its `<constant>` slots (left-to-right, the tape order), the
/// candidate carries `cand_params` (empty for the const-free arm; the f64-FITTED constants
/// for the fit arm's `FitVerdict::NearMiss`). Returns `true` iff every f64-failing binding
/// row passes at `P` bits. Never called on an f64 accept; a `false` leaves the f64 verdict
/// unchanged. `atol` is the absolute band of a FITTED constant (the fit arm); a const-free
/// pair passes 0, because its agreement is relative (see `worker::candidate_matches`).
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
    // The binding POINTS: every row where the source is defined. A source-inf row the
    // candidate does not already reproduce fails and is escalated: the P_INF re-judge inside
    // `judge_row` decides pole (stays inf -> must-match -> Fail) vs saturation (resolves
    // finite -> compare). It is a point of the denominator as much as of the numerator.
    let mut binding = 0usize;
    let mut failing: Vec<usize> = Vec::new();
    for (r, &s) in y_src.iter().enumerate() {
        let c = y_cand[r];
        if s.is_nan() {
            continue; // undefined -> extendable, not evidence
        }
        binding += 1;
        let fails = if s.is_infinite() {
            c != s
        } else {
            // the row predicate of `eval::allclose_extends` (candidate as reference)
            !c.is_finite() || (s - c).abs() > atol + rtol * c.abs()
        };
        if fails {
            failing.push(r);
        }
    }
    // A pair with no variable evaluates to the same value on every row: one point.
    let one_point = !source
        .iter()
        .chain(candidate.iter())
        .any(|t| var_names.contains(t));
    if one_point {
        failing.truncate(1);
        binding = binding.min(1);
    }
    let gate = (NEAR_MISS_FRAC * binding as f64).ceil() as usize;
    if failing.is_empty() || failing.len() > gate {
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

/// THE f64 RANGE STANDARD on one instance (`numeric::RangeKind`): f64 promises the evaluator's
/// value wherever nothing overflows or underflows, and at a row where an operation of either
/// side does (`src_events`, `cand_events`; `None` marks no row), a rewrite may give the true
/// value instead. So every row that fails
/// the f64 predicate of `eval::allclose_extends` must be such a row, and must pass when judged
/// at `P` bits (`judge_row_confirmed`, the rescue's own row judgment: both sides' true values
/// agree). A failing row with no event refuses, exactly as f64 always did; a row whose true
/// values disagree refuses too, so a rewrite gives the evaluator's value or the true one there
/// and nothing else. No near-miss gate: a range event is the special case of the f64 promise,
/// not a near miss, and the high-precision work stays on the failing event rows. `y_src`,
/// `y_cand`, the parameters and `atol` are as in `rescue`.
#[allow(clippy::too_many_arguments)]
pub fn range_rescue(
    y_src: &[f64],
    y_cand: &[f64],
    src_events: Option<&[bool]>,
    cand_events: Option<&[bool]>,
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
    let mut failing: Vec<usize> = Vec::new();
    for (r, &s) in y_src.iter().enumerate() {
        let c = y_cand[r];
        if s.is_nan() {
            continue; // undefined -> extendable
        }
        let fails = if s.is_infinite() {
            c != s
        } else {
            !c.is_finite() || (s - c).abs() > atol + rtol * c.abs()
        };
        if fails {
            if !(src_events.is_some_and(|e| e[r]) || cand_events.is_some_and(|e| e[r])) {
                return false; // a failing row where nothing left the range: f64 refuses
            }
            failing.push(r);
        }
    }
    // A pair with no variable evaluates to the same value on every row: one point.
    if !source
        .iter()
        .chain(candidate.iter())
        .any(|t| var_names.contains(t))
    {
        failing.truncate(1);
    }
    if failing.is_empty() {
        return false; // nothing failed: not this function's question
    }
    let mut cc = match Consts::new() {
        Ok(cc) => cc,
        Err(_) => return false,
    };
    failing.iter().all(|&r| {
        matches!(
            judge_row_confirmed(
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
            ),
            Some(RowVerdict::Pass) | Some(RowVerdict::Extend)
        )
    })
}

/// The literal-resolution arbiter (worker's `resolve_const_slots`):
/// `target(tgt_params) − source(src_params)` on one X row at the working precision, as a
/// SIGNED difference (the resolver bisects on its sign and ranks on its magnitude).
/// Params are f64, each lifted EXACTLY (an f64 literal token denotes a dyadic rational,
/// so evaluating a candidate literal at `P` bits carries no rendering error; the
/// transcendental leaves `np.pi`/`np.e` render at `P` bits, the arbiter convention).
/// `None` = either side unevaluable or non-finite on this row: the row cannot judge
/// literal candidates and the caller must pick another probe row.
#[allow(clippy::too_many_arguments)]
pub fn point_diff(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    row: usize,
) -> Option<BigFloat> {
    let mut cc = Consts::new().ok()?;
    let s = eval_prefix(source, ops, var_names, x_cols, src_params, row, P, &mut cc)?;
    let t = eval_prefix(target, ops, var_names, x_cols, tgt_params, row, P, &mut cc)?;
    if !is_fin(&s) || !is_fin(&t) {
        return None;
    }
    Some(t.sub(&s, P, RM))
}

/// The precisions [`ground_order`] climbs: an order is read at one rung and confirmed at the
/// next, and an order no two adjacent rungs agree on is undecided (fail closed).
const ORDER_LADDER: [usize; 4] = [128, 256, 512, 1024];

/// The guard bits an order keeps below a rung's precision: a difference of two values smaller
/// than `2^-(p - ORDER_GUARD)` of their magnitude is no order at `p` bits.
const ORDER_GUARD: usize = 40;

/// THE ORDER OF TWO GROUND VALUES over the reals, at high precision: `a` below, equal to (never
/// decided) or above `b`, each a ground prefix expression whose literal tokens denote exact
/// numbers (`ac::convert::to_prefix_literal_tokens`: an integer, a decimal, `p/q`; `np.pi` and
/// `np.e` rendered at the rung's precision). The difference is read at each rung of
/// `ORDER_LADDER` and counts where it exceeds the rung's error band (`ORDER_GUARD` bits below
/// the operands' magnitude); two adjacent rungs agreeing in its sign decide the order. `None`
/// where they never agree within the ladder, where a value is not finite, or where an operator
/// or leaf has no high-precision reading. The real reading's lattice bags consult it where the
/// float64-wide enclosures overlap (`max(sin(1), 0.8414709848078965)`, 6.7e-18 apart).
pub(crate) fn ground_order(
    a: &[String],
    b: &[String],
    ops: &Operators,
) -> Option<std::cmp::Ordering> {
    let mut cc = Consts::new().ok()?;
    let mut last: Option<std::cmp::Ordering> = None;
    for &p in &ORDER_LADDER {
        let va = eval_exact(a, ops, p, &mut cc)?;
        let vb = eval_exact(b, ops, p, &mut cc)?;
        if !is_fin(&va) || !is_fin(&vb) {
            return None;
        }
        let d = va.sub(&vb, p, RM);
        let scale = [va.exponent(), vb.exponent(), Some(1)]
            .into_iter()
            .flatten()
            .max()
            .unwrap_or(1) as i64;
        let decided = !d.is_zero()
            && d.exponent()
                .is_some_and(|e| (e as i64) > scale - (p - ORDER_GUARD) as i64);
        let here = decided.then(|| {
            if d.is_negative() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        if here.is_some() && here == last {
            return here;
        }
        last = here;
    }
    None
}

/// A ground prefix expression at `p` bits, its literal tokens read exactly (`exact_leaf`).
fn eval_exact(tokens: &[String], ops: &Operators, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    fn node(
        tokens: &[String],
        idx: &mut usize,
        ops: &Operators,
        p: usize,
        cc: &mut Consts,
    ) -> Option<BigFloat> {
        let tok = tokens.get(*idx)?;
        *idx += 1;
        match ops.arity_of(tok) {
            Some(1) => {
                let a = node(tokens, idx, ops, p, cc)?;
                apply_unary(tok, &a, p, cc)
            }
            Some(2) => {
                let a = node(tokens, idx, ops, p, cc)?;
                let b = node(tokens, idx, ops, p, cc)?;
                apply_binary(tok, &a, &b, p, cc)
            }
            Some(_) => None,
            None => exact_leaf(tok, p, cc),
        }
    }
    let mut idx = 0usize;
    let v = node(tokens, &mut idx, ops, p, cc)?;
    (idx == tokens.len()).then_some(v)
}

/// The value of a literal token at `p` bits: a decimal (with an exponent) or a fraction `p/q`
/// read from its digits, so `0.1` is one tenth and not its float; `np.pi` and `np.e` at `p`
/// bits; any other leaf the evaluator reads (an infinity, `(-1)`) as its float, which is exact.
fn exact_leaf(tok: &str, p: usize, cc: &mut Consts) -> Option<BigFloat> {
    match tok {
        "np.pi" => return Some(cc.pi(p, RM)),
        "np.e" => return Some(cc.e(p, RM)),
        _ => {}
    }
    let t = tok
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .unwrap_or(tok);
    let decimal = |s: &str, cc: &mut Consts| -> Option<BigFloat> {
        let ok = !s.is_empty()
            && s.bytes()
                .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
            && s.parse::<f64>().is_ok();
        let v = ok.then(|| BigFloat::parse(s, astro_float::Radix::Dec, p, RM, cc))?;
        (!v.is_nan()).then_some(v)
    };
    if let Some((n, d)) = t.split_once('/') {
        let (n, d) = (decimal(n, cc)?, decimal(d, cc)?);
        return (!d.is_zero()).then(|| n.div(&d, p, RM));
    }
    decimal(t, cc).or_else(|| crate::numeric::leaf_value(tok).map(|v| BigFloat::from_f64(v, p)))
}

/// Strict BigFloat `|a| < |b|` (residual ranking; incomparable = false, fail closed).
/// Zeros are judged explicitly: astro-float keeps a SIGNED zero whose `cmp` orders
/// `-0 < +0`, which would rank a negative-zero residual strictly below an exact-zero
/// one and refuse a perfect pin.
pub fn residual_lt(a: &BigFloat, b: &BigFloat) -> bool {
    match (a.is_zero(), b.is_zero()) {
        (_, true) => false,
        (true, false) => true,
        (false, false) => a.abs().cmp(&b.abs()).is_some_and(|o| o < 0),
    }
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

// ---------------------------------------------------------------------------------------------
// Contract point verdict (the `battery` module's per-point referee).
//
// A faithful port of the contract's point machinery to arbitrary precision: the
// contract evaluator (one-zero conventions, limit completion, the pow spike-step,
// symbolic-cancellation snap floors, tan pole-proximity, boundary-honesty bands,
// evaluation caps -> Unresolved), the class comparison, and the two/three-rung
// confirmation at fixed precisions (50/120/250 decimal digits == 169/402/834 bits) with
// every transcendental coordinate rendered AT the rung's precision (pi/2 rebuilt per
// rung, the same way an mpmath evaluation at that dps would). Integer powers are
// SINGLE-rounded (guard bits, one rounding) to mirror correctly-rounded `mp.power`:
// `pow3(x)` and `pow2(x)*x` must round APART at a rung exactly where a correctly-rounded
// evaluator makes them -- that residue over an exactly-cancelling denominator is the
// unresolvable-seam class ((x^3 + x^2 y)/(x + y) at (pi/2, -pi/2): 0/0 = nan in f64,
// residue/0 = -inf at rung 1), which no precision can settle and certification rejects.

/// One coordinate of a special battery point: dyadic values are exact at every precision;
/// the others are rendered per rung (`mp.pi/2` etc. rebuilt at the current precision, the
/// per-precision battery builders).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ProbeAtom {
    /// an exactly-representable f64 (dyadic rational): the same point at every precision
    Val(f64),
    /// `n * pi / d` rendered at the rung's precision (pi first, then the exact/rounded ops,
    /// mirroring `mp.pi/d`)
    PiFrac(i64, u64),
    /// Euler's number at the rung's precision
    E,
    /// the non-dyadic decimal battery atom `-1.7` (rendered per rung)
    Dec17Neg,
    /// an exact rational `n/d` rendered at the rung's precision (d > 0). The registry's
    /// exceptional-point generator solves affine pole locations exactly as rationals
    /// (`1 - 3*w = 0` -> `w = 1/3`), and a non-dyadic location must be HIT exactly at
    /// every rung -- its f64 rounding sits beside the pole, not on it.
    Rat(i64, u64),
}

impl ProbeAtom {
    /// The f64 rendering (what the deployed evaluator sees).
    pub fn value(&self) -> f64 {
        match self {
            ProbeAtom::Val(v) => *v,
            ProbeAtom::PiFrac(n, d) => *n as f64 * std::f64::consts::PI / *d as f64,
            ProbeAtom::E => std::f64::consts::E,
            ProbeAtom::Dec17Neg => -1.7,
            ProbeAtom::Rat(n, d) => *n as f64 / *d as f64,
        }
    }
}

/// The ordinary working precision, in decimal digits (`_contract.BASE_DPS`).
const BASE_DPS: i32 = 50;

/// The gap ladder's rungs, in decimal digits. Rungs 2 and 3 are FLOORS, not fixtures:
/// both climb with the operands actually seen (`required_dps`). Rung 3 is consulted only
/// to resolve a rung disagreement, or when rung 1 refused outright.
const GAP_RUNGS: [i32; 3] = [BASE_DPS, 120, 250];

/// Cap on the operand-scaled rung (`_contract.MAX_DPS`).
const MAX_DPS: i32 = 2000;

/// Decimal digits -> binary precision, mpmath's own mapping (`round((dps + 1) * log2 10)`),
/// so a rung computed here lands on exactly the precision the Python judge would use:
/// 50 -> 169, 120 -> 402, 250 -> 834.
fn dps_to_prec(dps: i32) -> usize {
    (f64::from(dps + 1) * 3.321_928_094_887_362_6)
        .round()
        .max(1.0) as usize
}

/// `floor(x)` for a finite non-negative `x`, by BINARY SEARCH over integer BigFloats.
/// astro-float exposes no lossless integer conversion, and the `x` here is a decimal digit
/// count that can run past what an f64 holds exactly (`cosh(1e5)` is ~10^43429).
fn floor_nonneg(x: &BigFloat, p: usize) -> Option<u64> {
    if x.is_nan() || x.is_inf() || x.is_negative() {
        return None;
    }
    let e = x.exponent()?; // |x| in [2^(e-1), 2^e)
    if e <= 0 {
        return Some(0);
    }
    if e > 62 {
        return None;
    }
    let (mut lo, mut hi) = (0u64, 1u64 << e); // small_int(lo) <= x < small_int(hi)
    while lo + 1 < hi {
        let m = lo + (hi - lo) / 2;
        if cmp_gt(&small_int(m, p), x) {
            hi = m;
        } else {
            lo = m;
        }
    }
    Some(lo)
}

/// Precision that a cancellation between the largest intermediate `mag` and the smallest
/// nonzero one `small` can demand -- the mirror of `_contract._required_dps` and
/// `_contract._demanded_dps`.
///
/// Adding two numbers of magnitude 10^k whose true sum is 10^-k destroys about 2k
/// significant digits, so the working precision has to carry them before any correct digit
/// survives. Measured against the case that motivated it: at t = -20 the intermediates of
/// `cosh(25t) + sinh(25t)` are near e^500 ~ 10^217, and the sum only becomes representable
/// somewhere past dps 400 -- which is what `2 * 217 + 50` gives.
///
/// A SMALL INTERMEDIATE COSTS THE SAME AS A LARGE ONE, and for the same reason: it is the
/// RATIO the precision has to carry, not the size. `cos x = 1 - x^2/2 + ...` at
/// x = exp(-300) = 5.1e-131 puts the whole content of the answer 261 digits below the 1 it
/// sits next to, so the output is exactly 1.0 at dps 50, at 120 and at 250, and `acos` of
/// that is 0 at all three -- three rungs reading one manufactured zero.
///
/// Derived, not tuned: the 2 is the two-sided digit loss, `k` is MEASURED from the
/// intermediates actually seen rather than guessed from the expression, and `BASE_DPS` is
/// the ordinary working precision that has to survive on top.
/// The caller takes it twice: clamped to `MAX_DPS` for the rung it can afford, and RAW for
/// what the point asked for. The two differ exactly where the ceiling binds, and there the
/// difference is the whole finding -- a rung clamped to `MAX_DPS` is not the rung that was
/// asked for, and reading it as though it were is how the judge convicted past its own
/// evidence.
fn demanded_dps(mag: &BigFloat, small: &BigFloat, p: usize, cc: &mut Consts) -> i32 {
    let mut k: i32 = 0;
    if !mag.is_nan() && !mag.is_inf() && cmp_gt(mag, &one(p)) {
        match floor_nonneg(&mag.log10(p, RM, cc), p) {
            Some(f) => k = k.max(i32::try_from(f + 1).unwrap_or(MAX_DPS)),
            None => return BASE_DPS,
        }
    }
    if !small.is_nan() && !small.is_inf() && !small.is_zero() && cmp_lt(small, &one(p)) {
        // `-floor(log10(small))` = `ceil(log10(1/small))`, taken on the reciprocal because
        // `floor_nonneg` only reads non-negative arguments.
        let r = one(p).div(small, p, RM);
        match floor_nonneg(&r.log10(p, RM, cc), p) {
            Some(f) => {
                let f = i32::try_from(f).unwrap_or(MAX_DPS);
                let ks = if cmp_lt(&pow10(f, p), &r) { f + 1 } else { f };
                k = k.max(ks);
            }
            None => return BASE_DPS,
        }
    }
    BASE_DPS + 2 * k.max(0)
}

/// Is this side a WRITTEN literal -- one token the rule SPELLS, exact at every rung? The
/// mirror of the `tree[0] == 'lit'` test: `<constant>` is a fitted slot and a variable is a
/// binding, and neither is something the rule writes down.
fn is_written_literal(tokens: &[String], var_names: &[String]) -> bool {
    tokens.len() == 1 && tokens[0] != "<constant>" && !var_names.iter().any(|v| *v == tokens[0])
}

fn cmp_eq(a: &BigFloat, b: &BigFloat) -> bool {
    !cmp_gt(a, b) && !cmp_lt(a, b)
}

fn render_atom(a: &ProbeAtom, p: usize, cc: &mut Consts) -> BigFloat {
    match a {
        ProbeAtom::Val(v) => BigFloat::from_f64(*v, p),
        ProbeAtom::PiFrac(n, d) => {
            let mut v = cc.pi(p, RM);
            let na = n.unsigned_abs();
            if na != 1 {
                v = v.mul(&small_int(na, p), p, RM);
            }
            if *d != 1 {
                v = v.div(&small_int(*d, p), p, RM);
            }
            if *n < 0 {
                v = v.neg();
            }
            v
        }
        ProbeAtom::E => cc.e(p, RM),
        ProbeAtom::Dec17Neg => small_int(17, p).div(&small_int(10, p), p, RM).neg(),
        ProbeAtom::Rat(n, d) => {
            let v = small_int(n.unsigned_abs(), p).div(&small_int(*d, p), p, RM);
            if *n < 0 {
                v.neg()
            } else {
                v
            }
        }
    }
}

/// Round `v` to `p` bits (astro-float ops round their result to the requested precision).
fn round_to(v: BigFloat, p: usize) -> BigFloat {
    v.add(&small_int(0, p), p, RM)
}

/// SINGLE-rounded `x^k`: the product accumulates at `p + 128` guard bits and rounds ONCE to
/// `p` -- mirroring mpmath's correctly-rounded `mp.power`.
/// (The rescue path's `powi` may double-round; the probe's whole point is that `pow3(x)` and
/// `pow2(x)*x` round APART at the rung, so its own powers must be single-rounded.)
fn powi_sr(x: &BigFloat, k: u64, p: usize) -> BigFloat {
    let g = p + 128;
    let mut acc = x.clone();
    for _ in 1..k {
        acc = acc.mul(x, g, RM);
    }
    round_to(acc, p)
}

/// The point-comparison classes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PointCmp {
    /// values agree (both nan, same infinity, or finite within tolerance)
    Eq,
    /// both finite, apart beyond tolerance: clause (a) -- kills at ANY measure
    RealChange,
    /// source nan, target defined: clause (b) extension event
    Ext,
    /// target nan where the source is defined: clause (b) shrink event
    Shrink,
    /// an infinity involved and the classes differ: clause (c) event
    InfChange,
}

/// Per-rung evaluation context: precision, the dps-derived snap floors, and the snap /
/// unresolved flags.
struct CtCtx {
    p: usize,
    /// 10^(-dps+10): the symbolic-cancellation floor / boundary-honesty band
    floor: BigFloat,
    /// 10^(dps-10): the pole ceiling
    ceil: BigFloat,
    snapped: bool,
    /// Largest finite INTERMEDIATE this evaluation produced -- what sizes the next rung.
    max_mag: BigFloat,
    /// ...and the smallest nonzero finite one, which sizes it from the OTHER end: a
    /// magnitude of 10^-k costs the same 2k digits as one of 10^k (see `required_dps`).
    min_mag: BigFloat,
    cc: Consts,
}

fn pow10(k: i32, p: usize) -> BigFloat {
    let ten = small_int(10, p);
    let mut acc = one(p);
    for _ in 0..k.unsigned_abs() {
        acc = acc.mul(&ten, p, RM);
    }
    if k < 0 {
        one(p).div(&acc, p, RM)
    } else {
        acc
    }
}

fn bf(v: f64, p: usize) -> BigFloat {
    BigFloat::from_f64(v, p)
}

fn cmp_gt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|o| o > 0)
}

fn cmp_lt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|o| o < 0)
}

fn is_pos(x: &BigFloat) -> bool {
    !x.is_negative() && !x.is_zero()
}

impl CtCtx {
    fn new(p: usize, dps: i32) -> Option<CtCtx> {
        Some(CtCtx {
            p,
            floor: pow10(-dps + 10, p),
            ceil: pow10(dps - 10, p),
            snapped: false,
            max_mag: small_int(0, p),
            min_mag: small_int(0, p),
            cc: Consts::new().ok()?,
        })
    }

    /// Record one node's value for `required_dps`. Only finite values count: an infinity
    /// says nothing about how many digits a cancellation will eat.
    ///
    /// `additive` marks a `+`/`-` node, whose output is NOT evidence of a small operand --
    /// it may BE the cancellation residue, whose size is a fact about the precision and not
    /// about the expression. Sizing the next rung from it is circular: at (pi/2, -pi/2) the
    /// seam `(x^3 + x^2 y)/(x + y)` would set its own rung from its own ~1e-50 residue, and
    /// the precision roulette that rung exists to expose would confirm itself instead.
    fn note_mag(&mut self, v: &BigFloat, additive: bool) {
        if v.is_nan() || v.is_inf() {
            return;
        }
        let a = v.abs();
        if cmp_gt(&a, &self.max_mag) {
            self.max_mag = a.clone();
        }
        if !additive && !a.is_zero() && (self.min_mag.is_zero() || cmp_lt(&a, &self.min_mag)) {
            self.min_mag = a;
        }
    }

    /// Symbolic-cancellation snap: a trig output below the noise floor is exact 0; above the pole
    /// ceiling it is a pole (nan); both flag a symbolic-cancellation event.
    ///
    /// THE FLOOR IS RELATIVE TO WHAT THE ARGUMENT PREDICTS. Cancellation means the output came
    /// out far smaller than its own input can account for -- `sin(pi)` is tiny only because pi
    /// is rounded. For |x| < 1 there is nothing to cancel: sin x ~ x, so `sin(exp(-300)) =
    /// 5.1e-131` is exactly the size it should be, and an ABSOLUTE band read it as 0 at dps 50,
    /// 0 at dps 120 and 5.1e-131 at dps 250 -- a value that MOVES with the working precision,
    /// inside the evaluator. Scaled by the argument the band still fires for every cancellation
    /// it was written for, because there |x| >= 1 and the scale is 1.
    fn snap_trig(&mut self, v: BigFloat, arg: &BigFloat) -> BigFloat {
        if v.is_nan() || v.is_inf() || v.is_zero() {
            return v;
        }
        let a = v.abs();
        let mut predicted = one(self.p);
        if !arg.is_nan() && !arg.is_inf() {
            let aa = arg.abs();
            if cmp_lt(&aa, &predicted) {
                predicted = aa;
            }
        }
        let floor = predicted.mul(&self.floor, self.p, RM);
        if cmp_lt(&a, &floor) {
            self.snapped = true;
            return small_int(0, self.p);
        }
        if cmp_gt(&a, &self.ceil) {
            self.snapped = true;
            return NAN;
        }
        v
    }
}

/// Contract division (one zero; c/0 = sign(c)*inf; 0/0 undefined).
fn ct_div(a: &BigFloat, b: &BigFloat, p: usize) -> BigFloat {
    if a.is_nan() || b.is_nan() {
        return NAN;
    }
    if b.is_zero() {
        if a.is_zero() {
            return NAN;
        }
        return if is_pos(a) { INF_POS } else { INF_NEG };
    }
    if a.is_inf() && b.is_inf() {
        return NAN;
    }
    if b.is_inf() {
        return small_int(0, p);
    }
    a.div(b, p, RM)
}

/// Integral exponent test within the integer cap: `None` for
/// nan/inf/|b| > 1e15 or non-integral; else the parity of the integer.
fn int_parity(b: &BigFloat, p: usize) -> Option<bool> {
    if b.is_nan() || b.is_inf() || cmp_gt(&b.abs(), &bf(1e15, p)) {
        return None;
    }
    if !b.is_int() {
        return None;
    }
    let odd = b
        .rem(&small_int(2, p))
        .abs()
        .cmp(&one(p))
        .is_some_and(|o| o == 0);
    Some(odd)
}

/// Contract pow (the spike-step for infinite exponents, negative-base integrality,
/// x^0 = 1 including nan, one zero). `None` = Unresolved (evaluation caps). Finite powers
/// are SINGLE-rounded (`powi_sr` / `pow_pos_base` guard bits) to mirror mp.power.
fn ct_pow(a: &BigFloat, b: &BigFloat, ctx: &mut CtCtx) -> Option<BigFloat> {
    let p = ctx.p;
    if !a.is_nan() && a.cmp(&one(p)).is_some_and(|o| o == 0) {
        return Some(one(p));
    }
    if b.is_zero() {
        return Some(one(p)); // x^0 = 1 incl nan^0 (ratified v2 SS2)
    }
    if a.is_nan() || b.is_nan() {
        return Some(NAN);
    }
    if a.is_zero() {
        return Some(if is_pos(b) { small_int(0, p) } else { INF_POS });
    }
    if !a.is_inf() && a.abs().cmp(&one(p)).is_some_and(|o| o == 0) && b.is_inf() {
        return Some(one(p)); // pow(+-1, +-inf) = 1 (ratified)
    }
    if b.is_inf() {
        // the RATIFIED SS9.4 spike-step semantics: magnitude limit incl negative bases
        let aa = a.abs();
        let big = cmp_gt(&aa, &one(p));
        return Some(if is_pos(b) == big {
            INF_POS
        } else {
            small_int(0, p)
        });
    }
    if a.is_inf() {
        if is_pos(a) {
            return Some(if is_pos(b) { INF_POS } else { small_int(0, p) });
        }
        return Some(match int_parity(b, p) {
            None => NAN, // SS9.7: pow(-inf, non-integer) undefined
            Some(odd) => {
                if is_pos(b) {
                    if odd {
                        INF_NEG
                    } else {
                        INF_POS
                    }
                } else {
                    small_int(0, p) // one zero
                }
            }
        });
    }
    if a.is_negative() {
        if cmp_gt(&b.abs(), &bf(1e15, p)) {
            return None; // integrality/parity unknowable beyond the cap
        }
        let Some(odd) = int_parity(b, p) else {
            return Some(NAN); // SS9.7
        };
        let mag = pow_finite_pos(&a.abs(), b, p, &mut ctx.cc);
        return Some(if odd { mag.neg() } else { mag });
    }
    if cmp_gt(&b.abs(), &bf(1e6, p)) {
        return None;
    }
    Some(pow_finite_pos(a, b, p, &mut ctx.cc))
}

/// `a^b` for finite a > 0: single-rounded -- `powi_sr` on small integer exponents (keeps
/// dyadic exactness: pow2(1/2) == 1/4 exactly), `pow_pos_base` guard-bit exp/ln otherwise.
fn pow_finite_pos(a: &BigFloat, b: &BigFloat, p: usize, cc: &mut Consts) -> BigFloat {
    if b.is_int() && !cmp_gt(&b.abs(), &bf(64.0, p)) {
        let k = i64_of(b).unwrap_or(0);
        if k > 0 {
            return powi_sr(a, k as u64, p);
        }
        if k < 0 {
            return round_to(
                one(p + 128).div(&powi_sr_at(a, (-k) as u64, p + 128), p + 128, RM),
                p,
            );
        }
    }
    pow_pos_base(a, b, p, cc)
}

fn i64_of(b: &BigFloat) -> Option<i64> {
    // small integral BigFloat -> i64 via f64 (exact for |b| <= 64)
    let s = format!("{b}");
    s.parse::<f64>().ok().map(|v| v as i64)
}

/// `powi_sr` accumulating and REPORTING at `g` bits (no final rounding to a smaller p).
fn powi_sr_at(x: &BigFloat, k: u64, g: usize) -> BigFloat {
    let mut acc = x.clone();
    for _ in 1..k {
        acc = acc.mul(x, g, RM);
    }
    acc
}

/// The operators both high-precision evaluators (the rescue `apply_unary`/`apply_binary` and the
/// contract `ct_unary`/`ct_binary`) know. Any other head is unevaluable, at a NaN operand too,
/// so every verdict that needs one refuses it. A head joins this table together with its arms in
/// all four evaluators (`special` and `besselj` hold the 0.15 heads' kernels).
pub(crate) fn knows_operator(name: &str) -> bool {
    matches!(
        name,
        "floor" | "ceil" | "erf" | "harmonic" | "max" | "min" | "mod" | "atan2" | "besselj"
    ) || matches!(
        name,
        "neg"
            | "abs"
            | "inv"
            | "mult2"
            | "mult3"
            | "mult4"
            | "mult5"
            | "div2"
            | "div3"
            | "div4"
            | "div5"
            | "pow2"
            | "pow3"
            | "pow4"
            | "pow5"
            | "pow1_2"
            | "pow1_3"
            | "pow1_4"
            | "pow1_5"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "exp"
            | "log"
            | "+"
            | "-"
            | "*"
            | "/"
            | "pow"
            | "rootn"
    )
}

/// Can both evaluators evaluate every operator of `tokens` (read with `ops`' arities)?
pub(crate) fn evaluable(tokens: &[String], ops: &Operators) -> bool {
    tokens
        .iter()
        .all(|t| ops.arity_of(t).is_none() || knows_operator(t))
}

/// Unary/binary OPS at contract semantics. `None` = Unresolved.
fn ct_unary(name: &str, x: &BigFloat, ctx: &mut CtCtx) -> Option<Option<BigFloat>> {
    // outer None = unknown operator (caller falls back); inner None = Unresolved
    if !knows_operator(name) {
        return None;
    }
    let p = ctx.p;
    if x.is_nan() && name != "pow2" && name != "pow3" && name != "pow4" && name != "pow5" {
        // every contract op propagates nan (powk go through ct_pow, which handles nan^0 = n/a)
        return Some(Some(NAN));
    }
    let v: Option<BigFloat> = match name {
        "neg" => Some(x.neg()),
        "abs" => Some(x.abs()),
        "inv" => Some(ct_div(&one(p), x, p)),
        "mult2" | "mult3" | "mult4" | "mult5" => {
            let k = small_int(name.as_bytes()[4] as u64 - b'0' as u64, p);
            Some(if x.is_inf() {
                x.clone()
            } else {
                x.mul(&k, p, RM)
            })
        }
        "div2" | "div3" | "div4" | "div5" => {
            let k = small_int(name.as_bytes()[3] as u64 - b'0' as u64, p);
            Some(ct_div(x, &k, p))
        }
        "pow2" => ct_pow(x, &small_int(2, p), ctx),
        "pow3" => ct_pow(x, &small_int(3, p), ctx),
        "pow4" => ct_pow(x, &small_int(4, p), ctx),
        "pow5" => ct_pow(x, &small_int(5, p), ctx),
        "pow1_2" => ct_pow(x, &bf(0.5, p), ctx),
        "pow1_4" => ct_pow(x, &bf(0.25, p), ctx),
        "pow1_3" | "pow1_5" => {
            // odd real roots: sign-folded, total on R, +-inf fixed points
            let k = small_int(name.as_bytes()[5] as u64 - b'0' as u64, p);
            if x.is_inf() || x.is_zero() {
                Some(x.clone())
            } else {
                let r = one(p).div(&k, p, RM);
                let mag = pow_pos_base(&x.abs(), &r, p, &mut ctx.cc);
                Some(if x.is_negative() { mag.neg() } else { mag })
            }
        }
        "exp" => {
            if x.is_inf() {
                Some(if is_pos(x) { INF_POS } else { small_int(0, p) })
            } else if !cmp_lt(&x.abs(), &bf(1e5, p)) {
                None
            } else {
                Some(x.exp(p, RM, &mut ctx.cc))
            }
        }
        "log" => {
            if x.is_zero() {
                Some(INF_NEG)
            } else if x.is_negative() {
                Some(NAN)
            } else if x.is_inf() {
                Some(INF_POS)
            } else {
                Some(x.ln(p, RM, &mut ctx.cc))
            }
        }
        "sin" | "cos" | "tan" => {
            if x.is_inf() {
                Some(NAN)
            } else if cmp_gt(&x.abs(), &bf(1e12, p)) {
                None
            } else if name == "tan" {
                // pole PROXIMITY, not output magnitude
                let c = x.cos(p, RM, &mut ctx.cc);
                if cmp_lt(&c.abs(), &ctx.floor) {
                    ctx.snapped = true;
                    Some(NAN)
                } else {
                    let s = x.sin(p, RM, &mut ctx.cc);
                    let q = s.div(&c, p, RM);
                    Some(ctx.snap_trig(q, x))
                }
            } else if name == "sin" {
                let v = x.sin(p, RM, &mut ctx.cc);
                Some(ctx.snap_trig(v, x))
            } else {
                let v = x.cos(p, RM, &mut ctx.cc);
                Some(ctx.snap_trig(v, x))
            }
        }
        "asin" | "acos" => {
            let a = x.abs();
            if !x.is_inf() && cmp_gt(&a, &one(p)) {
                let band = a.sub(&one(p), p, RM).abs();
                if cmp_lt(&band, &ctx.floor) {
                    None // boundary honesty: indistinguishable from the boundary
                } else {
                    Some(NAN)
                }
            } else if x.is_inf() {
                Some(NAN)
            } else if name == "asin" {
                Some(x.asin(p, RM, &mut ctx.cc))
            } else {
                Some(x.acos(p, RM, &mut ctx.cc))
            }
        }
        "atan" => {
            if x.is_inf() {
                let h = ctx.cc.pi(p, RM).div(&small_int(2, p), p, RM);
                Some(if is_pos(x) { h } else { h.neg() })
            } else {
                Some(x.atan(p, RM, &mut ctx.cc))
            }
        }
        "sinh" => {
            if x.is_inf() {
                Some(x.clone())
            } else if !cmp_lt(&x.abs(), &bf(1e5, p)) {
                None
            } else {
                Some(x.sinh(p, RM, &mut ctx.cc))
            }
        }
        "cosh" => {
            if x.is_inf() {
                Some(INF_POS)
            } else if !cmp_lt(&x.abs(), &bf(1e5, p)) {
                None
            } else {
                Some(x.cosh(p, RM, &mut ctx.cc))
            }
        }
        // BOUNDARY HONESTY AT THE ASYMPTOTE, the doctrine `atanh` states below, applied
        // where the information is actually lost. Once |tanh(x)| is inside the working-
        // precision band of 1 the value is indistinguishable from 1, and everything
        // downstream inherits that: `asin(1)` comes back as pi/2 as though definite,
        // `cos(pi/2)` is a residue at the noise floor, and against a true `sech(1000)` of
        // 1e-434 that is a RELATIVE gap of 1.0 -- at EVERY rung, because tanh saturates at
        // every affordable precision. The decay test in `contract_point_verdict` reads
        // that stability as analytic, which is how the EXACT `cos asin tanh _0 ->
        // inv cosh _0` reached 6 of 167 grid points. Saturation makes a false rule look
        // true AND a true rule look false; this is the second half.
        //
        // Banding `asin`/`acos` instead was tried and reverted in the Python judge: their
        // +-1 is an ordinary point with a finite value, so a band there also refuses the
        // legitimate exact `asin(-1) = -pi/2`.
        //
        // This does NOT spare the shallow rows: at dps 50 the band is 1e-40 and
        // 1 - tanh(30) = 1.75e-26 sits far outside it. Rows inside the band at rung 1 are
        // settled by the CLIMB -- the band narrows as the precision rises.
        "tanh" => {
            if x.is_inf() {
                Some(if is_pos(x) { one(p) } else { one(p).neg() })
            } else {
                let t = x.tanh(p, RM, &mut ctx.cc);
                if cmp_lt(&t.abs().sub(&one(p), p, RM).abs(), &ctx.floor) {
                    None
                } else {
                    Some(t)
                }
            }
        }
        "asinh" => {
            if x.is_inf() {
                Some(x.clone())
            } else {
                Some(x.asinh(p, RM, &mut ctx.cc))
            }
        }
        "acosh" => {
            if x.is_inf() {
                if is_pos(x) {
                    Some(INF_POS)
                } else {
                    Some(NAN)
                }
            } else {
                let band = x.sub(&one(p), p, RM).abs();
                if cmp_lt(&band, &ctx.floor) {
                    None // boundary honesty
                } else if cmp_lt(x, &one(p)) {
                    Some(NAN)
                } else {
                    Some(x.acosh(p, RM, &mut ctx.cc))
                }
            }
        }
        "atanh" => {
            if x.is_inf() {
                Some(NAN)
            } else {
                let band = x.abs().sub(&one(p), p, RM).abs();
                if cmp_lt(&band, &ctx.floor) {
                    None // boundary honesty (incl written +-1: no literal provenance here
                         // -- battery variables never bind literals)
                } else if cmp_gt(&x.abs(), &one(p)) {
                    Some(NAN)
                } else {
                    Some(x.atanh(p, RM, &mut ctx.cc))
                }
            }
        }
        "floor" | "ceil" => ct_step(x, name == "floor", ctx),
        // erf's +-1 is an asymptote, tanh's class: inside the band of it the value is lost
        "erf" => {
            if x.is_inf() {
                Some(if is_pos(x) { one(p) } else { one(p).neg() })
            } else if x.is_zero() {
                Some(small_int(0, p))
            } else if special::erfc_below(&x.abs(), &ctx.floor, p, &mut ctx.cc) {
                None
            } else {
                special::erf(x, p, &mut ctx.cc)
                    .filter(|v| !cmp_lt(&v.abs().sub(&one(p), p, RM).abs(), &ctx.floor))
            }
        }
        // harmonic: NaN at every negative integer (the one-sided limits differ, D6) and at -inf;
        // an argument inside the band of a pole is indistinguishable from it, tan's class
        "harmonic" => {
            if x.is_inf() {
                Some(if is_pos(x) { INF_POS } else { NAN })
            } else if x.is_zero() {
                Some(small_int(0, p))
            } else if x.is_negative() && !cmp_lt(&x.abs(), &ctx.ceil) {
                None // integrality past the working resolution
            } else if x.is_negative() && x.is_int() {
                Some(NAN)
            } else if cmp_lt(x, &bf(-0.5, p)) && near_jump(x, ctx) {
                None
            } else {
                special::harmonic(x, p, &mut ctx.cc)
            }
        }
        _ => return None,
    };
    Some(v)
}

/// Is the finite `x` inside the working-precision band of an integer it is not? A step head
/// cannot tell there which side of its jump the argument lies, nor `harmonic` whether it is at
/// a pole: the boundary-honesty band of `asin`'s +-1 and `tan`'s poles, at the integers.
fn near_jump(x: &BigFloat, ctx: &CtCtx) -> bool {
    let p = ctx.p;
    let d = x.sub(&x.round(0, RM), p, RM).abs();
    if d.is_zero() {
        return false;
    }
    let a = x.abs();
    let scale = if cmp_gt(&a, &one(p)) { a } else { one(p) };
    cmp_lt(&d, &scale.mul(&ctx.floor, p, RM))
}

/// `floor` / `ceil` under the contract: exact on the representation, with the band at the jumps
/// (`near_jump`); +-inf are fixed points. Past the working resolution (|x| >= 10^(dps-10)) every
/// value reads as an integer whatever it truly is, so the point is Unresolved.
fn ct_step(x: &BigFloat, floor: bool, ctx: &CtCtx) -> Option<BigFloat> {
    if x.is_inf() {
        return Some(x.clone());
    }
    if !cmp_lt(&x.abs(), &ctx.ceil) || near_jump(x, ctx) {
        return None;
    }
    let v = if floor { x.floor() } else { x.ceil() };
    Some(if v.is_zero() { small_int(0, ctx.p) } else { v })
}

/// `mod(a, b)` under the contract: numpy's cells (D7: NaN at b = 0 and at an infinite dividend;
/// against an infinite divisor the dividend where the signs agree, the divisor otherwise), the
/// exact floored remainder, one zero, and the band at the jumps: a remainder inside the band of
/// 0 or of `b` sits at a multiple of `b`, where the residue's sign picks the side.
fn ct_mod(a: &BigFloat, b: &BigFloat, ctx: &CtCtx) -> Option<BigFloat> {
    let p = ctx.p;
    if b.is_zero() || a.is_inf() {
        return Some(NAN);
    }
    if b.is_inf() {
        if a.is_zero() {
            return Some(small_int(0, p));
        }
        return Some(if a.is_negative() == b.is_negative() {
            a.clone()
        } else {
            b.clone()
        });
    }
    let bb = b.abs();
    if !cmp_lt(&a.abs(), &bb.mul(&ctx.ceil, p, RM)) {
        return None; // the quotient's integer part is past the working resolution
    }
    let r = special::modulo(a, b, p);
    if r.is_zero() {
        return Some(small_int(0, p));
    }
    let band = bb.mul(&ctx.floor, p, RM);
    let ra = r.abs();
    if cmp_lt(&ra, &band) || cmp_lt(&bb.sub(&ra, p, RM).abs(), &band) {
        return None;
    }
    Some(r)
}

/// `atan2(y, x)` under the contract: C99's cells off the cut, one zero. THE CUT (D18: a null
/// set, no special case in the engine): on y = 0 with x < 0, and at the origin, float64 reads
/// the zero's sign and the contract has one zero, so the point is Unresolved; so is a y inside
/// the band of 0 there, a residue whose sign picks the branch.
fn ct_atan2(y: &BigFloat, x: &BigFloat, ctx: &mut CtCtx) -> Option<BigFloat> {
    let p = ctx.p;
    if y.is_zero() && (x.is_zero() || x.is_negative()) {
        return None;
    }
    if x.is_negative() && !y.is_inf() {
        let scale = if x.is_inf() { one(p) } else { x.abs() };
        if cmp_lt(&y.abs(), &scale.mul(&ctx.floor, p, RM)) {
            return None;
        }
    }
    let v = special::atan2(y, x, p, &mut ctx.cc);
    Some(if v.is_zero() { small_int(0, p) } else { v })
}

fn ct_binary(name: &str, a: &BigFloat, b: &BigFloat, ctx: &mut CtCtx) -> Option<Option<BigFloat>> {
    if !knows_operator(name) {
        return None;
    }
    let p = ctx.p;
    if (a.is_nan() || b.is_nan()) && name != "pow" {
        return Some(Some(NAN));
    }
    let v: Option<BigFloat> = match name {
        "+" => {
            if a.is_inf() && b.is_inf() && (is_pos(a) != is_pos(b)) {
                Some(NAN)
            } else {
                Some(a.add(b, p, RM))
            }
        }
        "-" => {
            if a.is_inf() && b.is_inf() && (is_pos(a) == is_pos(b)) {
                Some(NAN)
            } else {
                Some(a.sub(b, p, RM))
            }
        }
        "*" => {
            if (a.is_zero() && b.is_inf()) || (b.is_zero() && a.is_inf()) {
                Some(NAN)
            } else {
                Some(a.mul(b, p, RM))
            }
        }
        "/" => Some(ct_div(a, b, p)),
        "pow" => ct_pow(a, b, ctx),
        // `rootn(x, n)`: IEEE-754 rootn, honest for every integer index.
        "rootn" => Some(rootn_hiprec(a, b, p, &mut ctx.cc, ct_div)),
        // continuous, and exact on the representation
        "max" | "min" => {
            let v = special::lattice(a, b, name == "max", p);
            Some(if v.is_zero() { small_int(0, p) } else { v })
        }
        "mod" => ct_mod(a, b, ctx),
        "atan2" => ct_atan2(a, b, ctx),
        // `besselj(nu, x)`: the contract's cells, the value or Unresolved (`besselj`).
        "besselj" => besselj::besselj(a, b, p, &mut ctx.cc),
        _ => return None,
    };
    Some(v)
}

/// Exact integer extraction from a BigFloat index by BINARY SEARCH over integer BigFloats
/// (no lossy float conversion; magnitudes up to i64::MAX, ~63 comparisons). `None` for
/// non-integers, zero, infinities, NaN and out-of-range magnitudes -- the invalid-index
/// cases IEEE rootn maps to NaN. Replaces the former scan-to-99 shortcut, which silently
/// refused legal larger indices.
fn integer_index(b: &BigFloat, p: usize) -> Option<i64> {
    if !b.is_int() || b.is_inf() || b.is_nan() || b.is_zero() {
        return None;
    }
    let mag = b.abs();
    if cmp_gt(&mag, &small_int(i64::MAX as u64, p)) {
        return None;
    }
    let (mut lo, mut hi) = (1u64, i64::MAX as u64);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cmp_gt(&mag, &small_int(mid, p)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if !mag.sub(&small_int(lo, p), p, RM).is_zero() {
        return None; // defensive: is_int and the search should always agree here
    }
    Some(if b.is_negative() {
        -(lo as i64)
    } else {
        lo as i64
    })
}

/// IEEE-754 `rootn(x, n)`, honest for every integer index:
/// odd index = the signed root (odd at zero too: the root of `-0` is `-0`), even = the
/// principal root (NaN on negatives, `+0` at either zero), n == 1 = the identity, n < 0 =
/// the reciprocal under the calling evaluator's own division `div` (the contract's one
/// zero, or IEEE's signed one), n == 0 or non-integer = NaN. One semantics, five surfaces
/// (see `operators.py::rootn` and `numeric.rs::rootn_f64`).
fn rootn_hiprec(
    a: &BigFloat,
    b: &BigFloat,
    p: usize,
    cc: &mut Consts,
    div: fn(&BigFloat, &BigFloat, usize) -> BigFloat,
) -> BigFloat {
    let Some(n) = integer_index(b, p) else {
        return NAN;
    };
    let k = n.unsigned_abs();
    let root = if a.is_nan() {
        NAN
    } else if k == 1 || (a.is_zero() && k % 2 == 1) {
        a.clone() // index 1 is the identity, and an odd root keeps the sign of zero
    } else if a.is_zero() {
        small_int(0, p) // an even root of either zero is +0
    } else if a.is_inf() {
        if !a.is_negative() || k % 2 == 1 {
            a.clone() // +inf always, -inf under odd roots: fixed points
        } else {
            NAN // even root of -inf
        }
    } else if k % 2 == 1 {
        odd_root(a, k, p, cc)
    } else if a.is_negative() {
        NAN // even root of a negative
    } else {
        let r = one(p).div(&small_int(k, p), p, RM);
        pow_pos_base(a, &r, p, cc)
    };
    if n < 0 {
        if root.is_nan() {
            return NAN;
        }
        div(&one(p), &root, p)
    } else {
        root
    }
}

/// Evaluate `tokens` under the CONTRACT semantics at the rung context. Outer `None` =
/// unevaluable (unknown token: caller fails open); inner `None` = Unresolved at this rung
/// (the point is skipped, never convicted).
#[allow(clippy::too_many_arguments)]
fn ct_node(
    tokens: &[String],
    idx: &mut usize,
    ops: &Operators,
    var_names: &[String],
    vals: &[BigFloat],
    params: &[f64],
    next_param: &mut usize,
    ctx: &mut CtCtx,
) -> Option<Option<BigFloat>> {
    let tok = tokens.get(*idx)?.clone();
    *idx += 1;
    if let Some(arity) = ops.arity_of(&tok) {
        let Some(a) = ct_node(tokens, idx, ops, var_names, vals, params, next_param, ctx)? else {
            // Unresolved below: still consume the rest of this operator's tokens so sibling
            // parsing stays aligned -- simplest is to bail out the whole evaluation.
            return Some(None);
        };
        if arity == 1 {
            let v = ct_unary(&tok, &a, ctx)?;
            if let Some(u) = &v {
                ctx.note_mag(u, false);
            }
            return Some(v);
        }
        let Some(b) = ct_node(tokens, idx, ops, var_names, vals, params, next_param, ctx)? else {
            return Some(None);
        };
        let v = ct_binary(&tok, &a, &b, ctx)?;
        if let Some(u) = &v {
            // a remainder is a difference too: `mod`'s output can be the residue
            ctx.note_mag(u, tok == "+" || tok == "-" || tok == "mod");
        }
        return Some(v);
    }
    let p = ctx.p;
    if tok == "<constant>" {
        let v = *params.get(*next_param)?;
        *next_param += 1;
        return Some(Some(BigFloat::from_f64(v, p)));
    }
    if let Some(c) = var_names.iter().position(|v| *v == tok) {
        return Some(Some(vals.get(c)?.clone()));
    }
    match tok.as_str() {
        "np.pi" => Some(Some(ctx.cc.pi(p, RM))),
        "np.e" => Some(Some(ctx.cc.e(p, RM))),
        _ => crate::numeric::leaf_value(&tok).map(|v| Some(BigFloat::from_f64(v, p))),
    }
}

fn ct_eval(
    tokens: &[String],
    ops: &Operators,
    var_names: &[String],
    vals: &[BigFloat],
    params: &[f64],
    ctx: &mut CtCtx,
) -> Option<Option<BigFloat>> {
    let mut idx = 0usize;
    let mut np = 0usize;
    let v = ct_node(tokens, &mut idx, ops, var_names, vals, params, &mut np, ctx)?;
    if v.is_some() && idx != tokens.len() {
        return None;
    }
    Some(v)
}

/// The value class of one side, for rung-agreement: finite / +inf / -inf / nan. A point
/// whose CLASS pattern shifts across rungs cannot be judged honestly -- for mpmath the
/// slash-pair seam flips nan <-> inf between dps 50 and 120; astro-float's rounding leaves
/// a residue at every rung but flips its SIGN, the same instability in a different suit.
fn side_class(v: &BigFloat) -> i8 {
    if v.is_nan() {
        2
    } else if v.is_inf() {
        if is_pos(v) {
            1
        } else {
            -1
        }
    } else {
        0
    }
}

/// The CLASS half of a point comparison -- everything nan/inf settles by itself.
/// `None` when BOTH sides are finite, where the gap ladder decides instead.
fn ct_classes(l: &BigFloat, r: &BigFloat) -> Option<PointCmp> {
    if l.is_nan() && r.is_nan() {
        return Some(PointCmp::Eq);
    }
    if l.is_nan() {
        return Some(PointCmp::Ext);
    }
    if r.is_nan() {
        return Some(PointCmp::Shrink);
    }
    if l.is_inf() || r.is_inf() {
        let same = l.is_inf() && r.is_inf() && (is_pos(l) == is_pos(r));
        return Some(if same {
            PointCmp::Eq
        } else {
            PointCmp::InfChange
        });
    }
    None
}

/// One rung's FINITE reading -> `(|l - r|, scale)`, both ABSOLUTE.
///
/// No tolerance and no absolute floor. The floor of 1 that used to sit here read
/// `e**sinh(-5) -> 0` and `asin(1e-8) -> 1e-8` as equal, and the relative 1e-25 that
/// replaced it spared the entire `tanh(x) -> +-1` saturation family. What the reading is
/// FOR is `contract_point_verdict`'s decay test; its size decides nothing on its own.
///
/// F105 (2026-08-21): THIS USED TO RETURN THE QUOTIENT, AND AGAINST AN EXACT ZERO THE
/// QUOTIENT IS 1.0 AT EVERY PRECISION. The decay test compares this reading at two rungs,
/// so any normaliser that is the SAME at both cancels out of the ratio and cannot change a
/// verdict; `max(|l|, |r|)` is not the same at both. When one side is exactly 0 and the
/// other is pure precision residue, the scale IS the residue, the quotient is 1.0 by
/// construction, and a gap that cannot move reads as an analytic one. Measured over 85
/// true zero-side identities, that convicted 25 -- and it convicted them HERE and not in
/// mpmath, which escaped only by returning an exact 0 at the upper rung where astro-float
/// leaves a residue. Same contract, different libm, opposite verdicts.
///
/// So the separation is reported RAW and the scale beside it, leaving the ladder to fix
/// ONE scale across both rungs -- where, being rung-invariant, it can only set the
/// zero-floor and never manufacture a ratio of its own.
fn ct_gap_reading(l: &BigFloat, r: &BigFloat, p: usize) -> (BigFloat, BigFloat) {
    let mut scale = l.abs();
    if cmp_gt(&r.abs(), &scale) {
        scale = r.abs();
    }
    if !cmp_gt(l, r) && !cmp_lt(l, r) {
        return (small_int(0, p), scale); // equal, including the quotient zeros -0 == 0
    }
    (l.sub(r, p, RM).abs(), scale)
}

/// Bounded functions, and the values they NEVER attain at a FINITE argument. Every bound
/// is exactly representable, which is the point: the test below must not need a precision
/// to reach its answer.
fn saturating_bounds(op: &str) -> Option<&'static [f64]> {
    match op {
        "tanh" => Some(&[-1.0, 1.0]), // |tanh a| < 1 strictly, for every finite a
        "exp" => Some(&[0.0]),        // exp a > 0 strictly, for every finite a
        "inv" => Some(&[0.0]),        // 1/a reaches 0 only in the limit
        "erf" => Some(&[-1.0, 1.0]),  // |erf a| < 1 strictly, for every finite a
        _ => None,
    }
}

/// F106: A BOUNDED FUNCTION NEVER ATTAINS ITS LIMIT AT A FINITE ARGUMENT, so a rewrite
/// claiming it does is false -- by a fact about the FUNCTION, at no precision at all.
///
/// This is the family the gap ladder cannot reach, and depth is not the answer.
/// `tanh(cosh(10))` differs from 1 by 2e-9566, inside the boundary band at every rung
/// anyone can afford, so the honest numeric verdict is Unresolved forever;
/// `tanh(exp(10))` differs by 1e-19132. Measured, one more rung bought ten rows for ~3x
/// the cost of EVERY comparison. A symbolic fact costs nothing and settles all of them.
///
/// Consulted only where the ladder has already refused, so it can add a verdict but never
/// change one.
///
/// TWO CONDITIONS, both required, both exact. The argument must be FINITE -- at an
/// infinite one the limit IS attained (`tanh(inf) = 1`, `exp(-inf) = 0`) and the claim is
/// true. And the bound must be WRITTEN, a single token, never computed: `1 - 2*exp(-2*
/// cosh(10))` IS `tanh(cosh(10))` and rounds to exactly 1.0 at any affordable precision,
/// so accepting a computed side would convict an identity. Slicing past the unary head to
/// reach the argument keeps `<constant>` params aligned -- only `<constant>` consumes one,
/// and it keeps its prefix order.
fn ct_saturation_verdict(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    atoms: &[ProbeAtom],
) -> Option<PointCmp> {
    let dps = GAP_RUNGS[0];
    let p = dps_to_prec(dps);
    for (src, sp, tgt, tp, tgt_is_target) in [
        (source, src_params, target, tgt_params, true),
        (target, tgt_params, source, src_params, false),
    ] {
        let Some(bounds) = src.first().and_then(|o| saturating_bounds(o)) else {
            continue;
        };
        if src.len() < 2 || tgt.len() != 1 {
            continue; // computed side: provenance is not a written bound
        }
        if tgt_is_target && tgt[0] == "<constant>" {
            // a FITTED target constant is no written bound: `erf(30) -> c_t` has the real
            // witness 1 - 2.6e-393, which no float64 fit carries (`_contract.saturation_verdict`)
            continue;
        }
        let Some(mut ctx) = CtCtx::new(p, dps) else {
            continue;
        };
        let vals: Vec<BigFloat> = atoms
            .iter()
            .map(|a| render_atom(a, p, &mut ctx.cc))
            .collect();
        let (Some(Some(arg)), Some(Some(lim))) = (
            ct_eval(&src[1..], ops, var_names, &vals, sp, &mut ctx),
            ct_eval(tgt, ops, var_names, &vals, tp, &mut ctx),
        ) else {
            continue;
        };
        if ctx.snapped {
            continue; // a snapped value is a DIFFERENT point
        }
        if side_class(&arg) != 0 || side_class(&lim) != 0 {
            continue; // the limit IS attained at an infinite argument
        }
        for b in bounds {
            let bb = bf(*b, p);
            if !cmp_gt(&lim, &bb) && !cmp_lt(&lim, &bb) {
                return Some(PointCmp::RealChange);
            }
        }
    }
    None
}

/// The confirmed point verdict -- the Rust mirror of `_contract._point_verdict`, and the
/// same contract the artifact gate applies. `(None, snapped)` = the point cannot be judged
/// honestly (refused at every affordable rung, or rung disagreement); otherwise the
/// confirmed class. `snapped` is rung 1's flag. `atoms` is full-width (aligned with
/// `var_names`); coordinates are re-rendered per rung. A side holding an operator neither
/// evaluator knows (`knows_operator`) is unevaluable and gets no verdict, `(None, false)`:
/// it fails CLOSED, never reading as equal.
///
/// THE FINITE HALF HAS NO MAGNITUDE BAR. It asks how a gap RESPONDS TO PRECISION, because
/// no threshold separates the two populations -- the shipped artifact proves it by
/// ordering. The EXACT `cos asin tanh _0 -> inv cosh _0` shows a dps-50 gap of 1.8e-26
/// while the FALSE `tanh pow np.pi 3 -> 1` shows 2.3e-27: the true rule's gap is the
/// LARGER one, so every bar convicts the identity and spares the falsehood. Nor is the
/// false family clustered -- `tanh(x) -> 1` is false at every finite x by exactly
/// 2/(e^2x+1), a smooth continuum with no valley to put a threshold in.
///
/// Decay separates them. Residue is a fact about the ARITHMETIC and falls with the working
/// precision; an analytic gap is a fact about the FUNCTIONS and does not move (measured
/// bit-identical over dps 50 -> 400). The bar is half the ADDED precision, and the two
/// populations clear it by >=30 decades on either side.
///
/// Known limit, stated rather than hidden: a gap below the deepest rung's resolution is
/// invisible, so `tanh(400) -> 1` (gap 1e-348) still reads eq. That is the "saturated at P
/// is extendable, not wrong" case documented above; it ships today too.
///
/// The CLASS half keeps the two/three-rung agreement rule, because a class has no gap to
/// watch. Its finding is the verdict PLUS both sides' value classes (incl infinity signs),
/// and agreement must hold on the whole pattern: mpmath's seam flips nan <-> -inf between
/// its rungs, while astro-float's residue never cancels but flips SIGN across rungs -- the
/// same "cannot be judged honestly" instability in two suits, and only the full pattern
/// catches it under either rounding behaviour.
pub fn contract_point_verdict(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    atoms: &[ProbeAtom],
) -> (Option<PointCmp>, bool) {
    /// One rung's reading: a CLASS finding (nan/inf involved -- there the classes are the
    /// whole answer), or the finite `(separation, scale)` the decay test consumes.
    enum Read {
        Class(PointCmp, i8, i8),
        Gap(BigFloat, BigFloat),
    }
    // outer None = unevaluable; inner None = Unresolved at this rung. `want_req` asks
    // for the precision this rung's own intermediates turned out to demand -- only rung 1
    // is asked, because only rung 1 sizes the rungs above it, and the logarithm behind the
    // answer is not worth paying for at 834 bits to then discard.
    let once = |dps: i32, want_req: bool| -> Option<(Option<Read>, bool, i32, BigFloat, i32)> {
        let p = dps_to_prec(dps);
        let mut ctx = CtCtx::new(p, dps)?;
        let vals: Vec<BigFloat> = atoms
            .iter()
            .map(|a| render_atom(a, p, &mut ctx.cc))
            .collect();
        let l = ct_eval(source, ops, var_names, &vals, src_params, &mut ctx)?;
        let r = ct_eval(target, ops, var_names, &vals, tgt_params, &mut ctx)?;
        let read = match (l, r) {
            (Some(l), Some(r)) => Some(match ct_classes(&l, &r) {
                Some(c) => Read::Class(c, side_class(&l), side_class(&r)),
                None => {
                    let (d, sc) = ct_gap_reading(&l, &r, p);
                    Read::Gap(d, sc)
                }
            }),
            _ => None, // Unresolved at this rung
        };
        // rung 1 also carries out its `max_mag`: an exactly-cancelling source offers no
        // scale from its own value, and the intermediate is then the only honest one.
        let (req, mag, dem) = if want_req {
            let mag = ctx.max_mag.clone();
            let small = ctx.min_mag.clone();
            // one logarithm, read twice: the rung the point ASKED for, and the rung it gets
            let dem = demanded_dps(&mag, &small, p, &mut ctx.cc);
            (MAX_DPS.min(dem), mag, dem)
        } else {
            (BASE_DPS, small_int(0, p), BASE_DPS)
        };
        Some((read, ctx.snapped, req, mag, dem))
    };

    let Some((read1, snap1, req1, mag1, dem1)) = once(GAP_RUNGS[0], true) else {
        return (None, false); // unevaluable: no verdict (fail closed)
    };
    // RUNG 2 IS OPERAND-SCALED. A fixed 120 confirms nothing when the intermediates are
    // 10^217: both rungs are swamped alike, agree on a manufactured verdict, and the
    // comparison reads that agreement as evidence. The precision comes from the operands
    // actually seen, so rung 2 is a second opinion rather than a second copy of the first.
    let dps2 = req1.max(GAP_RUNGS[1]);
    let dps3 = GAP_RUNGS[2].max(2 * dps2);
    let rungs = [GAP_RUNGS[0], dps2, dps3];

    // CLIMB PAST AN UNRESOLVED RUNG. A rung refuses when an intermediate lands inside its
    // boundary-honesty band, and that band NARROWS as the precision rises: tanh(60) is
    // inside it at dps 50 (band 1e-40) and well outside at dps 120 (1e-110). Bailing out
    // at rung 1 would abstain on rows a higher rung can settle. An Unresolved is a
    // statement about the RUNG, not about the rule.
    let mut first = Some(read1);
    let mut climbed: Option<(Read, i32, i32)> = None;
    for i in 0..rungs.len() - 1 {
        let read = match first.take() {
            Some(rd) => rd,
            None => match once(rungs[i], false) {
                Some((rd, _, _, _, _)) => rd,
                None => return (None, false), // unevaluable: no verdict
            },
        };
        if let Some(rd) = read {
            climbed = Some((rd, rungs[i], rungs[i + 1]));
            break;
        }
    }
    let Some((r_lo, lo_dps, hi_dps)) = climbed else {
        // Refused at every rung we can afford -- so ask the one question that needs no
        // rung at all (F106). Reached only here, so it cannot move a verdict the ladder
        // was able to reach.
        let v = ct_saturation_verdict(
            source, src_params, target, tgt_params, ops, var_names, atoms,
        );
        return (v, snap1);
    };

    match r_lo {
        Read::Class(c1, cl1, cr1) => {
            // nan/inf: a CLASS is confirmed by agreement at a second rung -- there is no
            // gap to watch decay, so this half is the two/three-rung rule unchanged.
            let f1 = Some((c1, cl1, cr1));
            let Some((r2, _, _, _, _)) = once(hi_dps, false) else {
                return (None, false);
            };
            let Some(r2) = r2 else {
                return (None, snap1); // Unresolved higher up: never convict
            };
            let f2 = match r2 {
                Read::Class(a, b, c) => Some((a, b, c)),
                Read::Gap(..) => None, // the class dissolved into a finite pair
            };
            // A rung-1 DISAGREEMENT is unsettled, full stop -- and here the Rust ladder
            // stays STRICTER than the Python one, deliberately. The judge rescues a
            // disagreement by asking rung 3 and confirming on rung2 == rung3. Under
            // astro-float that rescue is unsafe: at a precision-roulette seam the residue
            // does not cancel, it flips SIGN, and two adjacent rungs land on the same sign
            // often enough to fabricate the confirmation. Measured on the seam
            // `(x^3 + x^2 y)/(x + y)` vs `x^2` at (pi/2, -pi/2), which is exactly the class
            // the miner must refuse: the finding is (InfChange, +1, 0) at dps 50 and
            // (InfChange, -1, 0) at BOTH 120 and 250, so rungs 2 and 3 agree while the
            // point stays a roulette. Only the full class pattern catches it, and only
            // against rung 1. Refusing more than the gate does costs rules; certifying what
            // the gate convicts is the failure that matters.
            if f2 != f1 {
                return (None, snap1);
            }
            if !snap1 {
                return (Some(c1), snap1);
            }
            // a snapped agreement can be fabricated: it takes one rung higher.
            let f3 = match once(GAP_RUNGS[2].max(2 * hi_dps), false) {
                Some((Some(Read::Class(a, b, c)), _, _, _, _)) => Some((a, b, c)),
                _ => None,
            };
            (if f3 == f1 { Some(c1) } else { None }, snap1)
        }
        Read::Gap(d1, s1) => {
            // BOTH FINITE -- decided by DECAY, never by size.
            let mut p = dps_to_prec(hi_dps);
            let Some((rhi, _, _, _, _)) = once(hi_dps, false) else {
                return (None, false);
            };
            let Some(Read::Gap(d2, s2)) = rhi else {
                return (None, snap1); // class flipped / Unresolved: never convict
            };
            let (mut lo_dps, mut hi_dps) = (lo_dps, hi_dps);
            let (mut d1, mut s1, mut d2, mut s2) = (d1, s1, d2, s2);
            // A RUNG THAT LOST A SIDE ENTIRELY CANNOT DATE A DECAY. `d == scale` (with a
            // nonzero scale) says exactly one side read 0 while the other did not: the
            // rung saw no agreement at all, and a rate measured from a reading pinned at
            // the instrument's rail is not a rate. `acos(cos(exp(-100)))` is 3.7e-44, but
            // at dps 50 `cos` of it rounds to exactly 1.0 and `acos` gives 0 -- rung 1
            // read a 100% gap, rung 2 read 1.1e-78, and the drop came to 34.5 against a
            // bar of 35. A true identity convicted by three tenths of a decade.
            //
            // Climb, and measure the rate between the two rungs that BOTH saw something.
            // This can only DELAY a conviction to a stricter pair, never prevent one:
            // where the vanished side is genuinely zero -- a written 0, as in
            // `exp(sinh(-10)) -> 0` -- it is still zero at the higher rung, the rail
            // reading repeats, and the kill stands.
            if !s1.is_zero() && cmp_eq(&d1, &s1) && !cmp_eq(&d2, &s2) {
                let next = GAP_RUNGS[2].max(2 * hi_dps);
                let Some((rn, _, _, _, _)) = once(next, false) else {
                    return (None, false);
                };
                let Some(Read::Gap(d3, s3)) = rn else {
                    return (None, snap1);
                };
                lo_dps = hi_dps;
                hi_dps = next;
                p = dps_to_prec(hi_dps);
                d1 = d2;
                s1 = s2;
                d2 = d3;
                s2 = s3;
            }
            // THE FINER RUNG FOUND NO SEPARATION AT ALL -- the deepest evidence of
            // agreement the ladder can produce, and not something the floor below should
            // be allowed to argue with. Flooring that exact zero by the computation's own
            // magnitude put a 9.9e-485 gap where the rung had measured none, larger than
            // the 2.2e-485 the COARSER rung reported: a negative drop, and a kill of
            // `asin(inv(cosh(1000))) -> c_t`. A false rule does not reach here -- a fixed
            // gap stays visible at the finer rung, which is why `pow np.e sinh -10 -> 0`
            // still reads 1.03e-4783 at every rung and is still convicted below. This also
            // subsumes the exactly-zero-on-both-sides case, which is why the scale below
            // can no longer be zero.
            if d2.is_zero() {
                return (Some(PointCmp::Eq), snap1);
            }
            // ONE SCALE FOR BOTH RUNGS (F105), AND IT IS THE VALUE'S, NEVER THE
            // COMPUTATION'S (S11, 2026-08-22, mirroring `_contract._point_verdict`).
            // Taking the largest INTERMEDIATE into the max scaled the floor by
            // everything the computation passed through: `acos(inv(exp(exp 5)))
            // -> div2 np.pi` has a FIXED gap of -3.5e-65, the intermediate 2.85e64
            // put the rung-1 floor at 2.85e14, and the drop read 78.9 decades
            // against a bar of 65 -- a false rule certified, in both authorities.
            // The exactly-cancelling source keeps its intermediate scale, and ONLY
            // it: when the coarse rung's whole reading vanished (`s1 == 0`, total
            // cancellation), the sides offer no value to scale by and the residue
            // at the finer rung is residue OF the intermediate -- flooring by the
            // value scale convicts `log(exp x * exp -x) -> 0`. The distinguisher is
            // s1 itself: the S11 family arrives with the coarse rung having SEEN
            // the value (s1 ~ pi/2), so the value sets the floor and the fixed
            // gap stands. Intermediates keep their say in how many digits a
            // reading NEEDS (`demanded_dps`) either way.
            let s1_vanished = s1.is_zero();
            let mut scale = s1;
            if cmp_gt(&s2, &scale) {
                scale = s2;
            }
            if s1_vanished {
                let m = mag1.abs();
                if cmp_gt(&m, &scale) {
                    scale = m;
                }
            }
            // A separation of exactly zero is still not a special VERDICT -- it enters as
            // the rung's own resolution, "closer than this rung can see", which is what
            // keeps 32 exact identities resolvable when their sides agree to full
            // precision at one rung and differ by a rounding at the next. What changed is
            // that the resolution is now taken AT THIS SCALE.
            //
            // AND NOTHING NONZERO IS FLOORED, which is the half that does the work. An
            // absolute 10^-dps floor acquits `pow np.e sinh -10 -> 0`: the true value is
            // 1.03e-4783 at every precision -- tiny, but FIXED, and false -- so a floor
            // that swamps it manufactures a decay it does not have.
            let g_lo = if d1.is_zero() {
                scale.mul(&pow10(-lo_dps, p), p, RM)
            } else {
                d1
            };
            let g_hi = d2;
            // Residue falls with the added precision; an analytic gap does not move, and a
            // gap that only BECOMES visible higher up (deep saturation) drops by less than
            // nothing. The test is `log10(g_lo / g_hi) > (hi - lo) / 2`, squared to keep it
            // integer-exponent and spare a logarithm: `g_lo^2 > g_hi^2 * 10^(hi - lo)`.
            let lhs = g_lo.mul(&g_lo, p, RM);
            let rhs = g_hi
                .mul(&g_hi, p, RM)
                .mul(&pow10(hi_dps - lo_dps, p), p, RM);
            if cmp_gt(&lhs, &rhs) {
                return (Some(PointCmp::Eq), snap1);
            }
            // PAST THE CEILING, UNRESOLVED -- NEVER CONVICTED. That is what `MAX_DPS`
            // documents, and the ladder did not do it: it read a rung clamped to 2000 as
            // though it were the rung the point asked for. At c = -3000, `exp(c) = 1e-1303`
            // needs dps 2656 before `cos` of it is anything but exactly 1.0, so every
            // affordable rung reads `acos(1) = 0` and the frozen gap convicts an identity.
            //
            // Only the CONVICTION is withheld -- a decay the clamped rungs could still see
            // keeps its Eq. AND NOT WHERE A SIDE IS WRITTEN: a side the rule SPELLS is
            // exact at every rung, so a gap against it is not one more precision could
            // close. `10^sinh(-10)` is 5.8e-11014 and no closer to the 0 the rule writes
            // than it was at dps 50.
            if dem1 > MAX_DPS
                && !is_written_literal(source, var_names)
                && !is_written_literal(target, var_names)
            {
                let v = ct_saturation_verdict(
                    source, src_params, target, tgt_params, ops, var_names, atoms,
                );
                return (v, snap1);
            }
            (Some(PointCmp::RealChange), snap1)
        }
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
    if !knows_operator(name) {
        return None;
    }
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
        // the 0.15 heads, with float64's cells and signed zeros (`numeric::kernels`)
        "floor" | "ceil" => {
            let v = if name == "floor" { x.floor() } else { x.ceil() };
            if v.is_zero() && x.is_negative() {
                small_int(0, p).neg() // floor(-0) = ceil(-0.5) = -0
            } else {
                v
            }
        }
        "erf" => {
            if x.is_zero() {
                x.clone()
            } else if x.is_inf() {
                if x.is_negative() {
                    one(p).neg()
                } else {
                    one(p)
                }
            } else {
                special::erf(x, p, cc)?
            }
        }
        "harmonic" => {
            if x.is_zero() {
                x.clone() // H(+-0) = +-0
            } else if x.is_inf() {
                if x.is_negative() {
                    NAN
                } else {
                    INF_POS
                }
            } else if x.is_negative() && x.is_int() {
                NAN // a pole (D6)
            } else {
                special::harmonic(x, p, cc)?
            }
        }
        _ => return None,
    };
    Some(v)
}

/// IEEE `x / y`: a zero divisor gives the signed infinity (the ZERO's sign participates) and
/// an infinite one the signed zero (astro-float's is unsigned); `0/0` and `inf/inf` are NaN.
fn ieee_div(a: &BigFloat, b: &BigFloat, p: usize) -> BigFloat {
    let negative = a.is_negative() != b.is_negative();
    if b.is_zero() {
        if a.is_zero() {
            NAN
        } else if negative {
            INF_NEG
        } else {
            INF_POS
        }
    } else if b.is_inf() && !a.is_inf() && !a.is_nan() {
        let zero = small_int(0, p);
        if negative {
            zero.neg()
        } else {
            zero
        }
    } else {
        a.div(b, p, RM)
    }
}

fn apply_binary(
    name: &str,
    a: &BigFloat,
    b: &BigFloat,
    p: usize,
    cc: &mut Consts,
) -> Option<BigFloat> {
    if !knows_operator(name) {
        return None;
    }
    if a.is_nan() || b.is_nan() {
        return Some(NAN);
    }
    let v = match name {
        "+" => a.add(b, p, RM),
        "-" => a.sub(b, p, RM),
        "*" => a.mul(b, p, RM),
        "/" => ieee_div(a, b, p),
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
        // `rootn(x, n)`: IEEE-754 rootn, same table as the ct_binary arm.
        "rootn" => rootn_hiprec(a, b, p, cc, ieee_div),
        "max" | "min" => special::lattice(a, b, name == "max", p),
        "mod" => special::modulo(a, b, p),
        "atan2" => special::atan2(a, b, p, cc),
        "besselj" => return besselj::besselj(a, b, p, cc),
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

    /// `ground_order` reads each literal as its exact number and decides an order two adjacent
    /// precisions agree on: sin(1) is 6.7e-18 above the decimal 0.8414709848078965 and 1.8e-18
    /// above its float; 1/10 equals the decimal 0.1 (no order), and is below 0.1's float.
    #[test]
    fn ground_order_reads_literals_exactly() {
        use std::cmp::Ordering::{Greater, Less};
        let e = crate::engine::Engine::from_strs(
            include_str!("../assets/engines/acj2-4/config.yaml"),
            "[]",
        )
        .expect("acj2-4 builds");
        let ops = e.operators_ref();
        let ord = |a: &[&str], b: &[&str]| ground_order(&s(a), &s(b), ops);
        assert_eq!(ord(&["sin", "1"], &["0.8414709848078965"]), Some(Greater));
        assert_eq!(ord(&["0.8414709848078965"], &["sin", "1"]), Some(Less));
        assert_eq!(ord(&["1/10"], &["0.1"]), None);
        assert_eq!(
            ord(&["1/10"], &["0.1000000000000000055511151231257827"]),
            Some(Less)
        );
        assert_eq!(
            ord(&["np.pi"], &["3.14159265358979323846264338327950289"]),
            Some(Less)
        );
        assert_eq!(ord(&["cosh", "acosh", "2"], &["2"]), None);
        assert_eq!(ord(&["log", "-1"], &["0"]), None);
    }

    /// THE f64 RANGE STANDARD (`range_rescue`): at a row where an operation overflowed or
    /// underflowed, the true value is accepted beside the evaluator's -- and nothing else is;
    /// a failing row with no event refuses as f64 always did.
    #[test]
    fn range_rescue_accepts_only_the_true_value_at_a_range_event() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["x0"]);
        let cols = vec![vec![1.0, 800.0, -1000.0, 20.0]];
        let n = cols[0].len();
        let eval = |t: &[&str]| {
            crate::eval::Tape::compile(&s(t), ops, &vars)
                .unwrap()
                .eval_columns_events(&cols, &[], n)
        };
        let judge = |src: &[&str], cand: &[&str], rows: std::ops::Range<usize>| {
            let ((ys, es), (yc, ec)) = (eval(src), eval(cand));
            let pick = |v: &[f64]| v[rows.clone()].to_vec();
            let (es, ec) = (es[rows.clone()].to_vec(), ec[rows.clone()].to_vec());
            let cols_r = vec![cols[0][rows.clone()].to_vec()];
            range_rescue(
                &pick(&ys),
                &pick(&yc),
                Some(&es),
                Some(&ec),
                &s(src),
                &[],
                &s(cand),
                &[],
                ops,
                &vars,
                &cols_r,
                1e-11,
                0.0,
            )
        };
        // the true value past exp's overflow (800) and underflow (-1000)
        assert!(judge(
            &["rootn", "exp", "x0", "2"],
            &["exp", "/", "x0", "2"],
            0..3
        ));
        assert!(judge(&["log", "exp", "x0"], &["x0"], 0..3));
        // a value that is neither the evaluator's nor the true one
        assert!(!judge(&["log", "exp", "x0"], &["+", "x0", "1"], 1..2));
        // saturation inside the range is no event: f64's verdict stands
        assert!(!judge(&["atanh", "tanh", "x0"], &["x0"], 3..4));
    }

    /// `np.pi` denotes PI, not its f64 rendering. Lifting the
    /// literal from f64 made `sin(np.pi)` read 1.2246e-16 at EVERY precision -- the arbiter
    /// inherited the denotation error and no rung could remove it. Rendered at working precision,
    /// sin(pi at 256 bits) is ~1e-77: the arbiter can finally see that the identity is exact.
    /// ONE SEMANTICS, FIVE SURFACES: the high-precision rootn must
    /// agree with the f64 evaluator cell-for-cell across the full index table --
    /// odd/even/unit/negative/zero indices, negative bases, zero, infinities, NaN. This
    /// is the parity gate that makes the D2 class (divergent surfaces for one operator)
    /// structurally impossible to reintroduce silently.
    #[test]
    fn rootn_parity_with_the_f64_evaluator() {
        let p = 96;
        let mut cc = Consts::new().unwrap();
        let f64_rootn = crate::numeric::binary_fn("rootn").expect("f64 arm exists");
        let xs: [f64; 11] = [
            -8.0,
            -2.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            2.0,
            8.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let ns: [f64; 12] = [
            -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 64.0,
        ];
        fn agree(want: f64, got: &BigFloat, p: usize) -> bool {
            if want.is_nan() {
                return got.is_nan();
            }
            if got.is_nan() {
                return false;
            }
            if want.is_infinite() {
                return got.is_inf() && (want > 0.0) != got.is_negative();
            }
            if got.is_inf() {
                return false;
            }
            if want == 0.0 {
                // `-0.0 == 0.0`: a zero agrees only with the zero of its sign
                return got.is_zero() && want.is_sign_negative() == got.is_negative();
            }
            let w = bf(want, p);
            let diff = got.sub(&w, p, RM).abs();
            let tol = bf(want.abs().max(1e-300) * 1e-12, p);
            !cmp_gt(&diff, &tol)
        }
        for &x in &xs {
            for &n in &ns {
                let want = f64_rootn(x, n);
                // `from_f64` reads either zero as `+0`: a `-0` input is the negated zero
                let lifted = if x == 0.0 && x.is_sign_negative() {
                    bf(0.0, p).neg()
                } else {
                    bf(x, p)
                };
                let got = rootn_hiprec(&lifted, &bf(n, p), p, &mut cc, ieee_div);
                assert!(
                    agree(want, &got, p),
                    "rootn({x}, {n}): f64 gives {want}, hiprec disagrees"
                );
            }
        }
    }

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

    /// The near-miss gate counts POINTS: a source-inf row is a point of the denominator as
    /// much as of the numerator. Three saturated rows out of seven defined ones are a near
    /// miss (counted against the four finite rows alone they were not), and the ladder
    /// resolves each to its finite value.
    #[test]
    fn inf_rows_count_in_the_denominator() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let xs: Vec<f64> = vec![0.5, 1.0, -1.5, 2.0, 25.0, 30.0, -40.0];
        let cols = vec![xs.clone()];
        let y_src: Vec<f64> = xs.iter().map(|&x| x.tanh().atanh()).collect();
        assert_eq!(y_src.iter().filter(|v| v.is_infinite()).count(), 3);
        let ok = rescue(
            &y_src,
            &xs,
            &s(&["atanh", "tanh", "x0"]),
            &[],
            &s(&["x0"]),
            &[],
            ops,
            &["x0".to_string()],
            &cols,
            1e-9,
            0.0,
        );
        assert!(ok, "3 failing points of 7 is a near miss");
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

    /// The SHARED contract, pinned against the Python judge (`_contract._point_verdict`).
    /// Every expectation below is a measured behaviour of that judge at the same point, so
    /// a drift in either implementation surfaces here rather than as a mine that certifies
    /// rows the artifact gate then convicts.
    #[test]
    fn contract_verdict_matches_the_python_judge_on_the_decay_cases() {
        fn s(v: &[&str]) -> Vec<String> {
            v.iter().map(|x| x.to_string()).collect()
        }
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars: Vec<String> = vec!["x0".to_string()];
        let one_t = s(&["1"]);
        let tanh_x = s(&["tanh", "x0"]);

        // SHALLOW SATURATION IS STILL CONVICTED. `tanh(x) -> 1` is false at every finite x
        // by exactly 2/(e^2x + 1); at x = 30 that is 1.75e-26, far outside the dps-50
        // boundary band of 1e-40, so the point is judged -- and the gap does not move with
        // the precision, which is what makes it analytic rather than residue. Under the
        // old rel-1e-25 tolerance this whole family read as equal and was mined.
        let (v, _) = contract_point_verdict(
            &tanh_x,
            &[],
            &one_t,
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(30.0)],
        );
        assert_eq!(v, Some(PointCmp::RealChange), "tanh(30) -> 1 must convict");

        // DEEP SATURATION IS CONVICTED. At x = 1000 the gap is 1e-869, inside the boundary
        // band at every rung this ladder can afford, so no amount of precision reaches it.
        // This asserted `None` until F106: abstention was the honest answer while the only
        // instrument was numeric, and the requirement was never `Eq` -- which would certify
        // a false row. A bounded function does not attain its limit at a finite argument,
        // so the row is now REFUTED outright, which is strictly the better answer and the
        // one the Python judge gives (KILL, clause a, tier f64).
        let (v, _) = contract_point_verdict(
            &tanh_x,
            &[],
            &one_t,
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(1000.0)],
        );
        assert_eq!(
            v,
            Some(PointCmp::RealChange),
            "tanh(1000) -> 1 must convict, and never read eq"
        );

        // A TRUE IDENTITY ON THE MAGNITUDE TAIL SURVIVES. `cos(asin(tanh x)) = sech(x)`
        // exactly, and this is the grid point that convicted it before the asymptote band:
        // the sides differ by a residue that falls 71.6 decades between dps 50 and 120.
        let (v, _) = contract_point_verdict(
            &s(&["cos", "asin", "tanh", "x0"]),
            &[],
            &s(&["inv", "cosh", "x0"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(-19.9862579)],
        );
        assert_eq!(
            v,
            Some(PointCmp::Eq),
            "cos asin tanh -> sech must not convict"
        );

        // DECAY BEATS SIZE. `log(cosh y + sinh y) = y` is exact on all of R, and its dps-50
        // residue of 8.5e-21 fails ANY magnitude bar -- the rel 1e-25 that used to sit in
        // `ct_compare` included. It still falls 10^69 by the next rung, so it is spared.
        let (v, _) = contract_point_verdict(
            &s(&["log", "+", "cosh", "x0", "sinh", "x0"]),
            &[],
            &s(&["x0"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(-50.0)],
        );
        assert_eq!(
            v,
            Some(PointCmp::Eq),
            "log(cosh + sinh) -> y must be spared"
        );

        // F105: AN EXACT ZERO ON ONE SIDE IS NOT AN UNDECAYABLE GAP. `log(e^a e^-a) = 0`
        // exactly, so the target is a literal zero and the source is pure residue. While
        // the reading was a QUOTIENT of `max(|l|, |r|)` the scale WAS that residue, so the
        // gap read 1.0 at every rung, could not decay by construction, and clause (a)
        // convicted. astro-float hits this where mpmath escaped it by returning an exact
        // zero at the upper rung -- the same contract, opposite verdicts, on the libm.
        let (v, _) = contract_point_verdict(
            &s(&["log", "*", "exp", "x0", "exp", "neg", "x0"]),
            &[],
            &s(&["0"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(2.0)],
        );
        assert_eq!(v, Some(PointCmp::Eq), "log(e^a e^-a) -> 0 must be spared");

        // AND THE OTHER DIRECTION, which is why the floor cannot simply be widened to
        // reach the row above: `e^sinh(-10)` is 1.03e-4783 at EVERY precision -- tiny, but
        // FIXED, and false. Flooring a nonzero separation at an absolute 10^-dps swamps it
        // and manufactures a decay it does not have, so nothing nonzero is floored.
        let (v, _) = contract_point_verdict(
            &s(&["exp", "sinh", "(-10)"]),
            &[],
            &s(&["0"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(1.0)],
        );
        assert_eq!(
            v,
            Some(PointCmp::RealChange),
            "e^sinh(-10) -> 0 must still convict"
        );

        // F106: DEEP SATURATION IS NOW CONVICTED, NOT MERELY REFUSED. tanh(cosh(10))
        // differs from 1 by 2e-9566 -- inside the boundary band at every rung anyone can
        // afford, so the ladder abstains forever. A bounded function never attains its
        // limit at a finite argument, and that fact needs no precision.
        let (v, _) = contract_point_verdict(
            &s(&["tanh", "cosh", "x0"]),
            &[],
            &s(&["1"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(10.0)],
        );
        assert_eq!(
            v,
            Some(PointCmp::RealChange),
            "tanh(cosh(10)) -> 1 must convict"
        );

        // AND THE CONDITION THAT KEEPS IT SOUND: at an INFINITE argument the limit IS
        // attained, so the same shape is TRUE there.
        let (v, _) = contract_point_verdict(
            &s(&["tanh", "x0"]),
            &[],
            &s(&["1"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(f64::INFINITY)],
        );
        assert_eq!(
            v,
            Some(PointCmp::Eq),
            "tanh(inf) -> 1 is TRUE, not a change"
        );

        // DEEP SATURATION WITHOUT A WRITTEN BOUND IS STILL REFUSED, NOT CONVICTED: the
        // ladder cannot see the gap and this test has no opinion on a computed side.
        let (v, _) = contract_point_verdict(
            &s(&["tanh", "cosh", "x0"]),
            &[],
            &s(&["tanh", "cosh", "x0"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(10.0)],
        );
        assert_eq!(
            v, None,
            "a computed side must not be read as a written bound"
        );

        // F107, and the reason the constant quantifier could not widen: at magnitudes the
        // miner explores and the gate never checked, the instrument convicted rules true by
        // construction. Every assertion below read `RealChange` on the ladder it replaces.
        //
        // Two of the F107 fixes -- the finer rung's exact zero, and the scale that stopped
        // being inflated by an intermediate that had already cancelled -- are mirrored here
        // for parity but are not reachable from this entry point, which has no witness to
        // fit; `tests/test_verify.py::TestTheQuantifierWidensWithoutConvictingTheTrue`
        // exercises them where the witness lives.

        // A SMALL INTERMEDIATE SIZES THE RUNG LIKE A LARGE ONE, and a rung that lost a side
        // entirely cannot date a decay. At c = -300 the answer sits 261 digits below the 1
        // that `cos` adds it to, so dps 50, 120 and 250 all read exactly 1.0 and `acos` of
        // that is 0 -- three rungs reading one manufactured zero, not three opinions.
        let exp_x = s(&["exp", "x0"]);
        let (v, _) = contract_point_verdict(
            &s(&["acos", "cos", "exp", "x0"]),
            &[],
            &exp_x,
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(-300.0)],
        );
        assert_eq!(v, Some(PointCmp::Eq), "acos(cos(exp(-300))) IS exp(-300)");

        // THE SNAP BAND IS RELATIVE TO THE ARGUMENT. `sin(exp(-300)) = 5.1e-131` is exactly
        // the size its own argument predicts -- nothing cancelled. An absolute band read it
        // as 0 at dps 50 and 120 and as 5.1e-131 at 250: a value that MOVES with the
        // working precision, inside the evaluator.
        for head in ["asin", "atanh"] {
            let (v, _) = contract_point_verdict(
                &s(&[head, "sin", "exp", "x0"]),
                &[],
                &exp_x,
                &[],
                ops,
                &vars,
                &[ProbeAtom::Val(-300.0)],
            );
            assert_eq!(
                v,
                Some(PointCmp::Eq),
                "a well-resolved value must not snap to 0"
            );
        }

        // PAST THE CEILING, UNRESOLVED -- NEVER CONVICTED, which is what `MAX_DPS`
        // documents. At c = -3000, `exp(c) = 1e-1303` needs dps 2656 before `cos` of it is
        // anything but exactly 1.0, and the ladder was reading its clamped rung as though it
        // were the rung the point asked for.
        let (v, _) = contract_point_verdict(
            &s(&["acos", "cos", "exp", "x0"]),
            &[],
            &exp_x,
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(-3000.0)],
        );
        assert_ne!(
            v,
            Some(PointCmp::RealChange),
            "past MAX_DPS the point is unresolved, never convicted"
        );

        // ...AND NOT WHERE A SIDE IS WRITTEN, which is what keeps that abstention from
        // swallowing the F105 family. `10^sinh(-10)` is 5.8e-11014 and no closer to the 0
        // the rule spells than it was at dps 50: no precision could close that gap, so
        // running out of precision is not the reason it is open.
        let (v, _) = contract_point_verdict(
            &s(&["pow", "10", "sinh", "x0"]),
            &[],
            &s(&["0"]),
            &[],
            ops,
            &vars,
            &[ProbeAtom::Val(-10.0)],
        );
        assert_eq!(
            v,
            Some(PointCmp::RealChange),
            "a written 0 is exact at every rung: 10^sinh(-10) is not it"
        );
    }

    /// Cost anchor for the contract ladder (run explicitly:
    /// `cargo test --release contract_verdict_cost -- --ignored --nocapture`).
    /// Not a correctness test.
    ///
    /// Measured 2026-08-21 on this mix, against the rel-1e-25 ladder it replaced:
    /// 1290 -> 4047 us/call in total, ~3.1x. The decay test pays for a SECOND rung on
    /// every point where the tolerance answered many of them at rung 1, and the climb
    /// pays for a third where rung 1 refuses (the sech tail, 227 -> 1764 us, is the
    /// worst case here at 7.8x). That is affordable because both callers pre-screen:
    /// `battery::rows_consistent` reaches this only for rows whose DEPLOYED f64 sides
    /// already diverge, and `refusals::pole_refusal` is consulted last, on candidates
    /// that would otherwise be accepted.
    ///
    /// Two of the five verdicts MOVED, which is the whole point of the change: the old
    /// ladder called `tanh(30) -> 1` equal and would have mined a row the artifact gate
    /// convicts, and it called the exact `log(cosh y + sinh y) -> y` unresolvable at
    /// y = -50 and would have thrown a true identity away.
    #[test]
    #[ignore]
    fn contract_verdict_cost() {
        fn s(v: &[&str]) -> Vec<String> {
            v.iter().map(|x| x.to_string()).collect()
        }
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars: Vec<String> = vec!["x0".to_string()];
        let cases: Vec<(&str, Vec<String>, Vec<String>, ProbeAtom)> = vec![
            (
                "equal finite",
                s(&["+", "x0", "x0"]),
                s(&["*", "2", "x0"]),
                ProbeAtom::Rat(3, 7),
            ),
            (
                "tanh saturation",
                s(&["tanh", "x0"]),
                s(&["1"]),
                ProbeAtom::Val(30.0),
            ),
            (
                "deep cancellation",
                s(&["log", "+", "cosh", "x0", "sinh", "x0"]),
                s(&["x0"]),
                ProbeAtom::Val(-50.0),
            ),
            (
                "class lane",
                s(&["/", "x0", "x0"]),
                s(&["1"]),
                ProbeAtom::Val(0.0),
            ),
            (
                "sech tail",
                s(&["cos", "asin", "tanh", "x0"]),
                s(&["inv", "cosh", "x0"]),
                ProbeAtom::Val(-19.9862579),
            ),
        ];
        let reps = 100;
        let mut total = 0.0f64;
        for (label, src, tgt, atom) in &cases {
            let t0 = std::time::Instant::now();
            let mut last = None;
            for _ in 0..reps {
                let (v, _) = contract_point_verdict(src, &[], tgt, &[], ops, &vars, &[*atom]);
                last = Some(v);
            }
            let per = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            total += per;
            println!("{label:20} {per:10.1} us/call -> {last:?}");
        }
        println!("{:20} {total:10.1} us/call (sum)", "TOTAL");
    }

    /// The contract point verdict (`contract_point_verdict`): a differently-spelled exact
    /// cancellation at an irrational point is a precision ROULETTE (the dps-50 rung
    /// resolves residue/0 = inf where f64 sees 0/0 = nan) and must come back UNRESOLVED
    /// (`None` -> the caller rejects); a once-spelled
    /// cancellation is exact at every precision and reads as a confirmed EXT event
    /// (tolerated, the ratified null-set-completion class).
    #[test]
    fn contract_verdict_separates_seam_from_stable_null_completion() {
        fn s(v: &[&str]) -> Vec<String> {
            v.iter().map(|x| x.to_string()).collect()
        }
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars: Vec<String> = vec!["x0".to_string(), "x1".to_string()];
        let pt = [ProbeAtom::PiFrac(1, 2), ProbeAtom::PiFrac(-1, 2)];
        let pow2x = s(&["pow", "x0", "2"]);
        // (x^3 + x^2 y)/(x + y) at (pi/2, -pi/2): pow(x,3) and pow(x,2)*x round APART at the
        // rungs -> residue/0 = +-inf vs nan: rung disagreement, unresolved.
        let seam = s(&[
            "/", "+", "pow", "x0", "3", "*", "pow", "x0", "2", "x1", "+", "x0", "x1",
        ]);
        let (v, _) = contract_point_verdict(&seam, &[], &pow2x, &[], ops, &vars, &pt);
        assert_eq!(v, None, "seam must be unresolved");
        // the minus variant at (pi/2, pi/2) (its own seam points)
        let seam_minus = s(&[
            "/", "-", "pow", "x0", "3", "*", "pow", "x0", "2", "x1", "-", "x0", "x1",
        ]);
        let pt_mm = [ProbeAtom::PiFrac(1, 2), ProbeAtom::PiFrac(1, 2)];
        let (v, _) = contract_point_verdict(&seam_minus, &[], &pow2x, &[], ops, &vars, &pt_mm);
        assert_eq!(v, None, "minus-variant seam must be unresolved");
        // (x+y)^3/(x+y)^2 at the same point: spelled ONCE -> exact 0 at every precision ->
        // 0/0 = nan stably: a confirmed EXT event (tolerated).
        let stable = s(&[
            "/", "pow", "+", "x0", "x1", "3", "pow", "+", "x0", "x1", "2",
        ]);
        let tgt = s(&["+", "x0", "x1"]);
        let (v, _) = contract_point_verdict(&stable, &[], &tgt, &[], ops, &vars, &pt);
        assert_eq!(v, Some(PointCmp::Ext));
        // (x^2 - y^2)/(x + y) at (pi/2, -pi/2): pow2(x) and pow2(-x) are the SAME correctly-
        // rounded square -> exact 0 at every precision -> EXT (a certified 2-1 artifact rule).
        let even = s(&[
            "/", "-", "pow", "x0", "2", "pow", "x1", "2", "+", "x0", "x1",
        ]);
        let tgt2 = s(&["-", "x0", "x1"]);
        let (v, _) = contract_point_verdict(&even, &[], &tgt2, &[], ops, &vars, &pt);
        assert_eq!(v, Some(PointCmp::Ext));
        // x/x at the exact dyadic 0: 0/0 = nan at every precision -> EXT.
        let xx = s(&["/", "x0", "x0"]);
        let one_t = s(&["1"]);
        let (v, _) = contract_point_verdict(
            &xx,
            &[],
            &one_t,
            &[],
            ops,
            &["x0".to_string()],
            &[ProbeAtom::Val(0.0)],
        );
        assert_eq!(v, Some(PointCmp::Ext));
        // pow-of-sin at the exact contract point: pow(sin(pi/2), inf) = 1 vs 0 is a confirmed
        // REAL-CHANGE (clause a) -- sin(pi/2 at the rung) rounds to exactly 1.
        let powsin = s(&["pow", "sin", "x0", "float(\"inf\")"]);
        let zero_t = s(&["0"]);
        let (v, _) = contract_point_verdict(
            &powsin,
            &[],
            &zero_t,
            &[],
            ops,
            &["x0".to_string()],
            &[ProbeAtom::PiFrac(1, 2)],
        );
        assert_eq!(v, Some(PointCmp::RealChange));
        // sec^2 - tan^2 = 1 at pi/2: the pole-proximity snap fires (the f64 algebra
        // evaluates a DIFFERENT point there) -- the row must come back snapped, whatever
        // the class, so the caller must tolerate it (a snapped point is measurement error).
        let sec = s(&["-", "inv", "pow", "cos", "x0", "2", "pow", "tan", "x0", "2"]);
        let (v, snapped) = contract_point_verdict(
            &sec,
            &[],
            &one_t,
            &[],
            ops,
            &["x0".to_string()],
            &[ProbeAtom::PiFrac(1, 2)],
        );
        assert!(snapped, "pole proximity must flag the snap (got {v:?})");
        assert_ne!(v, Some(PointCmp::RealChange));
    }
}

#[cfg(test)]
mod new_head_tests {
    use super::*;

    fn s(t: &str) -> Vec<String> {
        t.split(' ').map(String::from).collect()
    }

    fn acj2() -> crate::engine::Engine {
        crate::engine::Engine::from_strs(include_str!("../assets/engines/acj2-4/config.yaml"), "[]")
            .expect("acj2-4 builds")
    }

    /// The contract point verdict on the 0.15 heads: exact at the jumps and the poles where the
    /// reals decide, Unresolved on atan2's cut, a real change where a rule is false at a jump.
    #[test]
    fn contract_verdicts_at_the_jumps_and_the_cut() {
        let e = acj2();
        let ops = e.operators_ref();
        let v = s("x0");
        let at = |l: &str, r: &str, x: ProbeAtom| {
            contract_point_verdict(&s(l), &[], &s(r), &[], ops, &v, &[x]).0
        };
        // floor(x + 1) = floor(x) + 1 at the jump x = -1, exactly
        assert_eq!(
            at("floor + x0 1", "+ floor x0 1", ProbeAtom::Val(-1.0)),
            Some(PointCmp::Eq)
        );
        // ceil(x) = floor(x) + 1 is false at every integer
        assert_eq!(
            at("ceil x0", "+ floor x0 1", ProbeAtom::Val(2.0)),
            Some(PointCmp::RealChange)
        );
        assert_eq!(
            at("ceil x0", "+ floor x0 1", ProbeAtom::Val(0.5)),
            Some(PointCmp::Eq)
        );
        // harmonic is NaN at its poles, and H(x) - H(x - 1) = 1/x off them
        assert_eq!(
            at("harmonic x0", "float(\"nan\")", ProbeAtom::Val(-2.0)),
            Some(PointCmp::Eq)
        );
        assert_eq!(
            at(
                "- harmonic x0 harmonic - x0 1",
                "inv x0",
                ProbeAtom::PiFrac(1, 3)
            ),
            Some(PointCmp::Eq)
        );
        // mod(x, 1) = x - floor(x), at a jump too; mod(x, 1) = x is false at -1/2
        assert_eq!(
            at("mod x0 1", "- x0 floor x0", ProbeAtom::Val(-2.0)),
            Some(PointCmp::Eq)
        );
        assert_eq!(
            at("mod x0 1", "x0", ProbeAtom::Val(-0.5)),
            Some(PointCmp::RealChange)
        );
        // erf is odd; its asymptote is never attained at a finite argument
        assert_eq!(
            at("erf neg x0", "neg erf x0", ProbeAtom::E),
            Some(PointCmp::Eq)
        );
        assert_eq!(
            at("erf * x0 30", "1", ProbeAtom::Val(1.0)),
            Some(PointCmp::RealChange)
        );
        // atan2 on its cut (y = 0, x < 0) is Unresolved; off it, atan2(-y, x) = -atan2(y, x)
        let w = s("x0 x1");
        let at2 = |l: &str, r: &str, y: f64, x: f64| {
            let atoms = [ProbeAtom::Val(y), ProbeAtom::Val(x)];
            contract_point_verdict(&s(l), &[], &s(r), &[], ops, &w, &atoms).0
        };
        assert_eq!(at2("atan2 neg x0 x1", "neg atan2 x0 x1", 0.0, -1.0), None);
        assert_eq!(
            at2("atan2 neg x0 x1", "neg atan2 x0 x1", 0.5, -1.0),
            Some(PointCmp::Eq)
        );
        assert_eq!(at2("max x0 x1", "max x1 x0", -0.5, 1.5), Some(PointCmp::Eq));
    }

    /// The rescue evaluator's 0.15 arms against the float64 kernels on ordinary points: the
    /// high-precision value rounds to within an ulp of the kernel's.
    #[test]
    fn rescue_arms_round_to_the_f64_kernels() {
        let mut cc = Consts::new().unwrap();
        let p = 256;
        let read = |v: &BigFloat| -> f64 { format!("{v}").parse().unwrap() };
        for (name, x) in [
            ("floor", -2.5),
            ("ceil", -0.5),
            ("erf", 0.7),
            ("erf", -3.25),
            ("harmonic", 2.5),
            ("harmonic", -1.25),
        ] {
            let f = crate::numeric::unary_fn(name).expect("kernel");
            let v = apply_unary(name, &BigFloat::from_f64(x, p), p, &mut cc).expect("evaluable");
            let (got, want) = (read(&v), f(x));
            assert!(
                (got - want).abs() <= want.abs() * 2.5e-16,
                "{name}({x}): {got} vs {want}"
            );
        }
        for (name, a, b) in [
            ("mod", -7.0, 3.0),
            ("mod", 7.5, -2.0),
            ("max", -1.0, 2.0),
            ("min", -1.0, 2.0),
            ("atan2", -1.0, -2.0),
        ] {
            let f = crate::numeric::binary_fn(name).expect("kernel");
            let (ba, bb) = (BigFloat::from_f64(a, p), BigFloat::from_f64(b, p));
            let v = apply_binary(name, &ba, &bb, p, &mut cc).expect("evaluable");
            let (got, want) = (read(&v), f(a, b));
            assert!(
                (got - want).abs() <= want.abs() * 2.5e-16,
                "{name}({a}, {b}): {got} vs {want}"
            );
        }
        // besselj's kernel is its own module's (`besselj`): J_1(1)
        let one = BigFloat::from_f64(1.0, p);
        let j = read(&apply_binary("besselj", &one, &one, p, &mut cc).expect("evaluable"));
        assert!(
            (j - 0.44005058574493355).abs() <= 2.5e-16 * j,
            "J_1(1) = {j}"
        );
    }
}
