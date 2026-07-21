//! Special-point battery for the OFFLINE miner: certification probes at symbolic points.
//!
//! A shipped rule is a claim at EVERY point of the reals -- including the symbolic
//! coincidences (0, +-1/2, ..., +-pi/2, pi, e, -1.7) that deployed expressions actually
//! reach, the special values a pattern-bound source CONSTANT takes in deployment, and the
//! exact integers/half-integers a fitted witness constant rounds to when the rule ships.
//! The random mine X reaches none of those coincidences, so a rule can certify numerically
//! while being false AT such a point -- the three certification gaps this module closes:
//!
//! 1. WITNESS SNAPPING (`snap_candidates` + `worker::adopt_snapped_witness`): a fitted
//!    `pow(x, 2.9999999999999996)` is NaN on x < 0, so the interval domain-preservation gate
//!    sees no extension -- but the rule SHIPS as `pow ?0 <constant>` and its exists-witness
//!    set contains the snapped 3.0, at which the target is total and the extension is half
//!    the line (`exp(log(pow3 x)) -> pow(x, C)`: a positive-measure domain extension).
//!    Snapping the witness before the domain gate makes the gate test the same witness the
//!    rule ships.
//! 2. SPECIAL X POINTS (`SpecialBattery` + `rows_consistent`): both-finite disagreement at a
//!    battery point is a value-soundness violation at ANY measure
//!    (`pow(sin x, inf) -> 0` is 1 at x = pi/2 -- f64 sin(pi/2) is EXACTLY 1.0; the deployed
//!    engine realizes the contract value, and no high-precision escalation may overturn a
//!    contract point: re-evaluating "the f64 approximation more precisely" answers a
//!    different question than "what happens at pi/2"). Judged WITHOUT the hiprec rescue.
//! 3. SPECIAL SOURCE CONSTANTS (`SPECIAL_CONSTS`): the source-constant battery, with skip
//!    semantics (an instance where the source is nowhere finite, or where no witness
//!    fits, is SKIPPED with no verdict -- only core generic behaviour is existence-binding;
//!    a special instance that DOES bind must then agree at the contract points).
//!
//! NaN rows stay domain-extendable (the `x/x -> 1` limit-completion doctrine),
//! EXCEPT where the extension is a precision ROULETTE rather than a stable fact:
//! `(x^3 + x^2 y)/(x + y)` at (pi/2, -pi/2) evaluates 0/0 = nan in f64 (the two spellings
//! of x^3 round identically) but residue/0 = -inf at 50 decimal digits (they round APART at
//! 169 bits) -- the precision rungs disagree, so the extension is not a stable fact and the
//! rule is rejected. `hiprec::contract_point_verdict` re-judges every deployed-diverging row
//! at fixed precision rungs with the coordinates rendered as transcendentals AT each
//! precision; only a confirmed event class keeps the row tolerated.

use crate::fit::Rng;
use crate::hiprec::ProbeAtom;
use crate::operators::Operators;

/// The deployed-consistency comparison scale: `|a - b| <= 1e-9 * max(1, |a|, |b|)`.
pub const JUDGE_REL: f64 = 1e-9;

/// The per-variable battery of contract points.
pub const BATTERY: [ProbeAtom; 21] = [
    ProbeAtom::Val(0.0),
    ProbeAtom::Val(0.5),
    ProbeAtom::Val(-0.5),
    ProbeAtom::Val(1.0),
    ProbeAtom::Val(-1.0),
    ProbeAtom::Val(2.0),
    ProbeAtom::Val(-2.0),
    ProbeAtom::Val(3.0),
    ProbeAtom::Val(-3.0),
    ProbeAtom::Val(0.25),
    ProbeAtom::Val(-0.25),
    ProbeAtom::Val(1.5),
    ProbeAtom::Val(-1.5),
    ProbeAtom::PiFrac(1, 2),
    ProbeAtom::PiFrac(-1, 2),
    ProbeAtom::PiFrac(1, 1),
    ProbeAtom::PiFrac(1, 4),
    ProbeAtom::PiFrac(1, 3),
    ProbeAtom::PiFrac(1, 6),
    ProbeAtom::E,
    ProbeAtom::Dec17Neg,
];

/// The source-constant battery: the core generic
/// witnesses plus the special rationals and the symbolic transcendental atoms. A fitted or
/// pattern-bound source constant reaches every one of these in deployment.
pub const SPECIAL_CONSTS: [f64; 18] = [
    2.5,
    -1.5,
    3.0,
    0.5,
    -0.7,
    1.0,
    -1.0,
    0.0,
    2.0,
    -2.0,
    4.0,
    -4.0,
    5.0,
    -5.0,
    std::f64::consts::FRAC_PI_2,
    -std::f64::consts::FRAC_PI_2,
    std::f64::consts::PI,
    std::f64::consts::E,
];

/// Cap on battery combinations per source: the slot-product is capped at 500 with a
/// deterministic seeded sample (>= 3-variable sources).
const MAX_COMBOS: usize = 500;

/// The special-point evaluation matrix for ONE source's variable set: full-width columns
/// (unused variables pinned to a generic value, never read by the tapes) plus the per-row
/// symbolic atoms of the USED variables, aligned with `var_names` for the hiprec probe.
pub struct SpecialBattery {
    pub cols: Vec<Vec<f64>>,
    pub n_rows: usize,
    /// per row: full-width atoms (unused variables hold a generic dyadic filler)
    pub atoms: Vec<Vec<ProbeAtom>>,
}

impl SpecialBattery {
    /// Build the battery for the used-variable index set (into a `n_vars`-wide column
    /// space). A variable-free source gets the single EMPTY point -- the source-constant
    /// sweep runs its one empty combo per constant, and `pow(cos(<constant>), inf) -> 0`
    /// is judged exactly there.
    pub fn build(n_vars: usize, used: &[usize]) -> Option<Self> {
        if used.is_empty() {
            let filler = ProbeAtom::Val(1.7);
            return Some(SpecialBattery {
                cols: vec![vec![1.7f64]; n_vars.max(1)],
                n_rows: 1,
                atoms: vec![vec![filler; n_vars.max(1)]],
            });
        }
        let k = used.len();
        let total: usize = BATTERY.len().checked_pow(k as u32)?;
        let picks: Vec<usize> = if total <= MAX_COMBOS {
            (0..total).collect()
        } else {
            // deterministic seeded sample without replacement
            let mut rng = Rng::new(0xBA77E52);
            let mut seen = std::collections::BTreeSet::new();
            while seen.len() < MAX_COMBOS {
                seen.insert((rng.next_u64() % total as u64) as usize);
            }
            seen.into_iter().collect()
        };
        let n_rows = picks.len();
        let mut cols = vec![vec![1.7f64; n_rows]; n_vars];
        let filler = ProbeAtom::Val(1.7);
        let mut atoms = vec![vec![filler; n_vars]; n_rows];
        for (r, &pick) in picks.iter().enumerate() {
            let mut rem = pick;
            for &v in used {
                let a = BATTERY[rem % BATTERY.len()];
                rem /= BATTERY.len();
                cols[v][r] = a.value();
                atoms[r][v] = a;
            }
        }
        Some(SpecialBattery {
            cols,
            n_rows,
            atoms,
        })
    }
}

/// The deployed (f64) row comparison: class equality for nan/inf, and the reference
/// tolerance joined with the mine's own (rtol, atol) row gate for finite pairs.
fn dep_compare(s: f64, c: f64, rtol: f64, atol: f64) -> crate::hiprec::PointCmp {
    use crate::hiprec::PointCmp as P;
    if s.is_nan() && c.is_nan() {
        return P::Eq;
    }
    if s.is_nan() {
        return P::Ext;
    }
    if c.is_nan() {
        return P::Shrink;
    }
    if s.is_infinite() || c.is_infinite() {
        return if s == c { P::Eq } else { P::InfChange };
    }
    // SUM, not max: a downstream consumer checks the rule at an exact (polished) witness,
    // while the miner's fitted constants may sit anywhere in the (rtol, atol) acceptance
    // band -- an exact constant rule with a band-edge witness is off by up to the band
    // width at every row (observed live: C*(x+y)/(x+y) -> <constant> with the witness
    // 1.01e-9 from C). The row passes iff SOME witness inside the band agrees at the
    // reference scale: by the triangle inequality that is the band width plus the
    // reference tolerance.
    let tol = (atol + rtol * s.abs()) + JUDGE_REL * s.abs().max(c.abs()).max(1.0);
    if (s - c).abs() <= tol {
        P::Eq
    } else {
        P::RealChange
    }
}

/// Row semantics at the special points. A row where
/// the deployed f64 sides agree passes outright; a diverging row is judged by the contract
/// point verdict (`hiprec::contract_point_verdict`: precision rungs plus snap semantics):
///   - confirmed REAL-CHANGE -> REJECT (clause (a): kills at any measure, snap included);
///   - EXT / SHRINK / INF-CHANGE -> a tolerated event class, row passes;
///   - contract eq or unresolved, NOT snapped -> REJECT (the deployed algebra structurally
///     diverges where the contract does not object -- this also convicts the
///     precision-roulette seam class);
///   - snapped (symbolic-cancellation / pole-proximity at the point) -> row passes (the f64
///     algebra evaluates a DIFFERENT point there; measurement error, not a value claim).
#[allow(clippy::too_many_arguments)]
pub fn rows_consistent(
    y_src: &[f64],
    y_cand: &[f64],
    battery: &SpecialBattery,
    source: &[String],
    src_params: &[f64],
    cand: &[String],
    cand_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    rtol: f64,
    atol: f64,
) -> bool {
    use crate::hiprec::PointCmp as P;
    for r in 0..battery.n_rows {
        let dv = dep_compare(y_src[r], y_cand[r], rtol, atol);
        if dv == P::Eq {
            continue;
        }
        let (v, snapped) = crate::hiprec::contract_point_verdict(
            source,
            src_params,
            cand,
            cand_params,
            ops,
            var_names,
            &battery.atoms[r],
        );
        let reject = match v {
            Some(P::RealChange) => true,
            Some(P::Ext) | Some(P::Shrink) | Some(P::InfChange) => false,
            Some(P::Eq) | None => !snapped,
        };
        if reject {
            if std::env::var("SIMPLIPY_BATTERY_TRACE").is_ok() {
                eprintln!(
                    "BATTERY-REJECT src=[{}] sp={:?} cand=[{}] cp={:?} row={} coords={:?} s={:e} c={:e} dv={:?} v={:?} snapped={}",
                    source.join(" "), src_params, cand.join(" "), cand_params, r,
                    battery.atoms[r].iter().map(|a| a.value()).collect::<Vec<_>>(),
                    y_src[r], y_cand[r], dv, v, snapped,
                );
            }
            return false;
        }
    }
    true
}

/// Snap fitted constants to the canonical witness targets: nearest integer, else
/// nearest half-integer, within `max(1e-6, 1e-9 |c|)`.
/// `None` when no coordinate moves. PURE arithmetic -- whether the snapped vector is still
/// a valid witness is the caller's decision (`worker::adopt_snapped_witness`, which needs
/// the fit's own accept semantics including the hiprec rescue).
pub fn snap_candidates(params: &[f64]) -> Option<Vec<f64>> {
    let mut snapped = params.to_vec();
    let mut changed = false;
    for v in snapped.iter_mut() {
        for t in [v.round(), (*v * 2.0).round() * 0.5] {
            if t != *v && (t - *v).abs() <= 1e-6_f64.max(1e-9 * v.abs()) {
                *v = t;
                changed = true;
                break;
            }
        }
    }
    if changed {
        Some(snapped)
    } else {
        None
    }
}

/// The used-variable index set of `source` within `var_names`.
pub fn used_variables(source: &[String], var_names: &[String]) -> Vec<usize> {
    var_names
        .iter()
        .enumerate()
        .filter(|(_, v)| source.iter().any(|t| &t == v))
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_shapes_and_determinism() {
        let sb = SpecialBattery::build(2, &[0]).unwrap();
        assert_eq!(sb.n_rows, BATTERY.len());
        assert!(sb.cols[1].iter().all(|&v| v == 1.7)); // unused var pinned
        let sb2 = SpecialBattery::build(2, &[0, 1]).unwrap();
        assert_eq!(sb2.n_rows, BATTERY.len() * BATTERY.len());
        let sb3a = SpecialBattery::build(3, &[0, 1, 2]).unwrap();
        let sb3b = SpecialBattery::build(3, &[0, 1, 2]).unwrap();
        assert_eq!(sb3a.n_rows, 500);
        assert_eq!(sb3a.cols, sb3b.cols); // deterministic sample
                                          // a variable-free source gets the single EMPTY point
        assert_eq!(SpecialBattery::build(1, &[]).unwrap().n_rows, 1);
    }

    #[test]
    fn snap_targets_integers_then_half_integers_within_threshold() {
        assert_eq!(snap_candidates(&[2.999_999_999_999_999_6]), Some(vec![3.0]));
        assert_eq!(snap_candidates(&[1.499_999_999_999_9]), Some(vec![1.5]));
        assert_eq!(snap_candidates(&[-0.999_999_999_997]), Some(vec![-1.0]));
        assert_eq!(snap_candidates(&[1.3e-13]), Some(vec![0.0]));
        // far from any canonical witness: no snap
        assert_eq!(snap_candidates(&[2.71]), None);
        assert_eq!(snap_candidates(&[std::f64::consts::FRAC_PI_2]), None);
        // already canonical: no change to report
        assert_eq!(snap_candidates(&[3.0]), None);
    }
}
