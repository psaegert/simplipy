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

/// The STRUCTURAL deployed band: `|a - b| <= 1e-9 * max(1, |a|, |b|)`, the mirror of
/// `_contract.compare_deployed_structural`. It answers "has the deployed algebra DIVERGED"
/// -- an infinity where a finite value belongs, a gap far outside rounding -- and a few ULP
/// is none of those. Deliberately loose: convicting rounding here renames sound rules as
/// misaligned, measured in Python at 396 against a true 10.
pub const JUDGE_REL: f64 = 1e-9;

/// The deployed-REALISATION bound, in ULP -- the mirror of `_contract.REALISED_ULP`, and
/// the same derivation. IEEE-754 mandates correct rounding only for + - * / and sqrt; the
/// measured libm error reaches 1 ULP on cos, cosh, sinh and tanh; a rewrite has TWO sides,
/// each a composition of up to about four such calls with independent errors that can
/// oppose: 2 x 4 x 1 = 8.
pub const REALISED_ULP: f64 = 8.0;

/// The spacing between `x` and the next double away from zero -- one ULP AT `x`.
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
///
/// SPAN MIRRORS THE JUDGE (audit U8, 2026-08-22). This battery reached |c| <= 5 while
/// the gate's constant quantifier reaches |c| = 1e4 generically (`WIDE_CONSTS`, F107)
/// and probes the f64 attainment magnitudes out to 1e17 (`SATURATION_CONSTS`) -- so
/// the miner minted rows the gate then convicted, and every such row was judged,
/// confirmed and routed before dying. The span below is the judge's own: the 18 core
/// witnesses, the 20 generic decades (both signs), and the 10 saturation magnitudes.
/// Skip semantics make the widening safe: an instance where no witness fits is
/// SKIPPED with no verdict, so new points can only refuse rows the gate would refuse.
pub const SPECIAL_CONSTS: [f64; 48] = [
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
    // WIDE_CONSTS: the generic magnitude span, decade-spaced, both signs (F107)
    10.0,
    -10.0,
    30.0,
    -30.0,
    100.0,
    -100.0,
    300.0,
    -300.0,
    1e3,
    -1e3,
    3e3,
    -3e3,
    1e4,
    -1e4,
    0.1,
    -0.1,
    0.01,
    -0.01,
    0.001,
    -0.001,
    // SATURATION_CONSTS: where f64 ATTAINS a bound mathematics only approaches
    19.0,
    -19.0,
    20.0,
    -20.0,
    50.0,
    -50.0,
    750.0,
    -750.0,
    1e17,
    -1e17,
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

/// The nan/inf half of a deployed comparison, shared by both bars below -- the mirror of
/// `_contract._deployed_classes`.
fn dep_classes(s: f64, c: f64) -> Option<crate::hiprec::PointCmp> {
    use crate::hiprec::PointCmp as P;
    if s.is_nan() && c.is_nan() {
        return Some(P::Eq);
    }
    if s.is_nan() {
        return Some(P::Ext);
    }
    if c.is_nan() {
        return Some(P::Shrink);
    }
    if s.is_infinite() || c.is_infinite() {
        return Some(if s == c { P::Eq } else { P::InfChange });
    }
    None
}

/// The mine's own witness allowance. SUM, not max: a downstream consumer checks the rule at
/// an exact (polished) witness, while the miner's fitted constants may sit anywhere in the
/// (rtol, atol) acceptance band -- an exact constant rule with a band-edge witness is off by
/// up to the band width at every row (observed live: C*(x+y)/(x+y) -> <constant> with the
/// witness 1.01e-9 from C). The row passes iff SOME witness inside the band agrees at the
/// reference scale: by the triangle inequality that is the band width plus the reference
/// tolerance.
fn witness_band(s: f64, rtol: f64, atol: f64) -> f64 {
    atol + rtol * s.abs()
}

/// "Does the deployed f64 engine compute this rewrite?" -- the REALISATION question, the
/// mirror of `_contract.compare_deployed_realised`. Bounded in ULP, with NO absolute floor.
///
/// This is the PRE-SCREEN: a row that passes here never reaches the contract at all. At the
/// structural band it passed `pow np.e sinh (-5) -> 0`, whose deployed sides are
/// 5.942307292381135e-33 and 0.0 -- a rule that is FALSE, minted without its truth ever
/// being asked, and convicted by the gate afterwards. 1e-9 relative is roughly 1e7 ULP and
/// for any value below 1 a pure ABSOLUTE floor; it is verbatim the first of the three
/// answers `compare_deployed_realised` documents as rejected.
fn dep_realised(s: f64, c: f64, rtol: f64, atol: f64) -> crate::hiprec::PointCmp {
    use crate::hiprec::PointCmp as P;
    if let Some(cl) = dep_classes(s, c) {
        return cl;
    }
    // The pre-screen's bar is the fit band PLUS ULP headroom, additively -- and that
    // shape is deliberate, not sloppy (audit U8, resolved 2026-08-22 by measurement).
    // A strict bit-distance OR-form was tried and refused true rules at the band
    // EDGE: `adopt_snapped_witness` legitimately places a witness up to the band
    // boundary, and the boundary case then measured the SNAP displacement (124,875
    // bit-ulps on `rootn exp 7 <constant>` at source constant -5), not realisation
    // -- while the judge, whose witness machinery re-polishes, CERTIFIES the rule
    // core. The additive ulp term is the band's closure under its own arithmetic.
    // Its binade-boundary factor-2 looseness (<= 16 effective bit-ulps) sits inside
    // the REALISED_ULP derivation's own insensitivity envelope ("every bound in
    // [8, 56] gives the identical partition"). The one genuine defect was
    // `ulp_at(f64::MAX) = +inf` -- a literally vacuous bar at the top of the range
    // -- which `ulp_at` now closes by pricing the top magnitude at its lower
    // neighbour's spacing.
    let tol = witness_band(s, rtol, atol) + REALISED_ULP * ulp_at(s.abs().max(c.abs()));
    if (s - c).abs() <= tol {
        P::Eq
    } else {
        P::RealChange
    }
}

/// The ulp SPACING at |x|, finite for every finite input: at `f64::MAX` the next
/// representable is inf, so the spacing is taken to the lower neighbour instead --
/// an infinite tolerance is not a bar (audit U8).
fn ulp_at(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return f64::INFINITY;
    }
    if a == f64::MAX {
        return a - f64::from_bits(a.to_bits() - 1);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// "Has the deployed algebra DIVERGED?" -- the STRUCTURAL question, the mirror of
/// `_contract.compare_deployed_structural`, and a DIFFERENT question from the one above.
/// Python measured what answering both with one comparison costs: 396 misalignments against
/// a true 10. Rust asked it once, so tightening that single bar to the realisation one would
/// have rejected every sound rule whose two sides round a few ULP apart.
fn dep_structural(s: f64, c: f64, rtol: f64, atol: f64) -> crate::hiprec::PointCmp {
    use crate::hiprec::PointCmp as P;
    if let Some(cl) = dep_classes(s, c) {
        return cl;
    }
    let tol = witness_band(s, rtol, atol) + JUDGE_REL * s.abs().max(c.abs()).max(1.0);
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
///
/// `events` (f64 only) marks the rows where an operation of either side overflowed or
/// underflowed (`numeric::RangeKind`): there f64 accepts the true value, so a row the contract
/// finds equal passes however the deployed sides diverge, as in the contract judge's battery.
#[allow(clippy::too_many_arguments)]
pub fn rows_consistent(
    y_src: &[f64],
    y_cand: &[f64],
    events: Option<&[bool]>,
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
        let dv = dep_realised(y_src[r], y_cand[r], rtol, atol);
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
            // The contract does not object, so the only thing left to convict is a
            // DIVERGENCE -- and that is the structural question, not the realisation one.
            // A few ULP between two spellings is rounding; asking the realisation bar here
            // would reject sound rules for it.
            Some(P::Eq) if events.is_some_and(|e| e[r]) => false,
            Some(P::Eq) | None => {
                !snapped && dep_structural(y_src[r], y_cand[r], rtol, atol) != P::Eq
            }
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
    fn the_pre_screen_asks_realisation_and_the_reject_asks_divergence() {
        use crate::hiprec::PointCmp as P;
        // `pow np.e sinh (-5) -> 0`: the deployed sides are 5.94e-33 and 0.0. Under the
        // STRUCTURAL band -- `1e-9 * max(1, |a|, |b|)`, a pure ABSOLUTE floor for anything
        // below 1 -- that reads as equal, so the row was minted and the contract was never
        // asked whether it is true. It is not. In ULP the two are ~4.5e18 apart.
        let (s, c) = (5.942_307_292_381_135e-33, 0.0);
        assert_eq!(dep_realised(s, c, 0.0, 0.0), P::RealChange);
        assert_eq!(dep_structural(s, c, 0.0, 0.0), P::Eq);

        // ...and why the structural bar has to STAY loose: a few ULP between two spellings
        // is rounding, not a divergence, and convicting it renames sound rules as
        // misaligned. Python measured that at 396 against a true 10.
        let a = 1.0_f64;
        for k in [3u64, 100, 1_000_000] {
            let b = f64::from_bits(a.to_bits() + k);
            assert_eq!(
                dep_structural(a, b, 0.0, 0.0),
                P::Eq,
                "{k} ULP is not a divergence"
            );
        }
        assert_eq!(
            dep_realised(a, f64::from_bits(a.to_bits() + 3), 0.0, 0.0),
            P::Eq
        );
        assert_eq!(
            dep_realised(a, f64::from_bits(a.to_bits() + 100), 0.0, 0.0),
            P::RealChange,
            "100 ULP is past the realisation bar"
        );

        // the mine's own witness band still rides on top of both
        assert_eq!(dep_realised(1.0, 1.0 + 1e-6, 1e-5, 0.0), P::Eq);
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

// ===== THE DEPLOYED BAR ========================================================================

/// The probe values of the deployed bar beyond `BATTERY`, the infinities and NaN: where the
/// step heads jump and the special functions meet a cell (a signed zero, the float neighbours of
/// the integers, tiny and huge magnitudes, the negative integers of `harmonic`'s poles, erf's
/// saturation), so that a law of the reals float64 breaks there is seen
/// (`floor(x + 1) = floor(x) + 1` at x = -1e-17, `ceil(x) = floor(x) + 1` at every integer).
pub const DEPLOYED_EXTRA: [f64; 33] = [
    -0.0,
    1e-17,
    -1e-17,
    5e-324,
    -5e-324,
    1e-300,
    -1e-300,
    0.9999999999999999,
    1.0000000000000002,
    -0.9999999999999999,
    -1.0000000000000002,
    1.9999999999999998,
    2.0000000000000004,
    -1.9999999999999998,
    -2.0000000000000004,
    2.9999999999999996,
    3.0000000000000004,
    -2.9999999999999996,
    -3.0000000000000004,
    4.0,
    -4.0,
    2.5,
    -2.5,
    6.0,
    -6.0,
    7.0,
    -7.0,
    1e17,
    -1e17,
    1e300,
    -1e300,
    0.1,
    -0.1,
];

/// Rows of the product past this many are thinned to an even stride (four or more variables).
const DEPLOYED_MAX_ROWS: usize = 250_000;

/// What the deployed bar found: the verdict, the number of probes, and the first probe
/// where the sides differ.
pub struct DeployedExact {
    pub ok: bool,
    pub probes: usize,
    pub first: Option<Vec<f64>>,
}

/// The 0.15 heads. A rule spelled with one answers to the deployed bar for its float64
/// realisation (`f64_bar`): a step head's jump (`floor(x + 1) = floor(x) + 1` fails at
/// x = -1e-17), a zero's sign that `atan2` reads, a pole of `harmonic` -- float64 parts from
/// the reals there, at points the contract's battery does not reach.
pub const NEW_HEADS: [&str; 9] = [
    "floor", "ceil", "erf", "harmonic", "max", "min", "mod", "atan2", "besselj",
];

/// The 0.15 heads whose float64 value jumps (a step, a remainder's wrap): a fitted target
/// constant cannot be judged bit for bit beside them.
const JUMP_HEADS: [&str; 3] = ["floor", "ceil", "mod"];

/// Does `tokens` hold a 0.15 head (as an operator of `ops`)?
pub fn holds_new_head(tokens: &[String], ops: &Operators) -> bool {
    tokens
        .iter()
        .any(|t| ops.arity_of(t).is_some() && NEW_HEADS.contains(&t.as_str()))
}

/// THE f64 REALISATION BAR of a rule spelled with a 0.15 head: `None` when the ordinary
/// realisation axis decides (no 0.15 head; a bare `<constant>` target, whose witness is the
/// source's own value; a fitted target constant beside heads without jumps), else the deployed
/// bar's verdict (`deployed_exact_over`), each source `<constant>` probed as a variable over the
/// finite probes (a fitted constant is a finite float). A fitted target constant beside a jump
/// head cannot be judged bit for bit, so it is refused. The contract judge ANDs this into a
/// rule's `realised` (`_core.f64_bar`) and the f64 mint applies it to every pick, so the two
/// cannot disagree.
pub fn f64_bar(
    lhs: &[String],
    rhs: &[String],
    var_names: &[String],
    ops: &Operators,
) -> Option<DeployedExact> {
    if !holds_new_head(lhs, ops) && !holds_new_head(rhs, ops) {
        return None;
    }
    let is_c = |t: &String| t == "<constant>";
    if rhs.iter().any(is_c) {
        let jumps = lhs
            .iter()
            .chain(rhs)
            .any(|t| ops.arity_of(t).is_some() && JUMP_HEADS.contains(&t.as_str()));
        return if rhs.len() == 1 || !jumps {
            None
        } else {
            Some(DeployedExact {
                ok: false,
                probes: 0,
                first: None,
            })
        };
    }
    let mut names = var_names.to_vec();
    let mut lhs_v = Vec::with_capacity(lhs.len());
    let mut n_c = 0usize;
    for t in lhs {
        if is_c(t) {
            let name = format!("<c{n_c}>");
            names.push(name.clone());
            lhs_v.push(name);
            n_c += 1;
        } else {
            lhs_v.push(t.clone());
        }
    }
    Some(deployed_exact_over(&lhs_v, rhs, &names, n_c, ops))
}

/// THE DEPLOYED BAR: do `lhs` and `rhs` agree BIT FOR BIT in float64 (the sign of a zero
/// included, since an inverse or `atan2` reads it; NaN equal to NaN), with the engine's own
/// kernels, at every probe? The probes are, per
/// variable of `var_names` either side uses, the `BATTERY` points, +-inf, NaN and
/// `DEPLOYED_EXTRA` (their full product); a generic grid per variable with the others at
/// 1.7 and -1.3; and the diagonal of that grid. No probe is excused: no tolerance, no range
/// event, no extension of a NaN. The last `n_finite` names of `var_names` are fitted constants,
/// probed over the finite values only.
///
/// It is the verdict of the f64 bar (`f64_bar`), which the contract judge ANDs into a rule's
/// `realised` and the f64 mint applies to every pick, so the judge licenses exactly what the
/// mint accepts. A side holding a `<constant>`, or one that does not compile, fails.
fn deployed_exact_over(
    lhs: &[String],
    rhs: &[String],
    var_names: &[String],
    n_finite: usize,
    ops: &Operators,
) -> DeployedExact {
    let refused = DeployedExact {
        ok: false,
        probes: 0,
        first: None,
    };
    if lhs.iter().chain(rhs).any(|t| t == "<constant>") {
        return refused;
    }
    let (Ok(tl), Ok(tr)) = (
        crate::eval::Tape::compile(lhs, ops, var_names),
        crate::eval::Tape::compile(rhs, ops, var_names),
    ) else {
        return refused;
    };
    let used: Vec<usize> = (0..var_names.len())
        .filter(|&i| lhs.iter().chain(rhs).any(|t| *t == var_names[i]))
        .collect();
    let mut values: Vec<f64> = Vec::new();
    for v in BATTERY
        .iter()
        .map(|a| a.value())
        .chain([f64::INFINITY, f64::NEG_INFINITY, f64::NAN])
        .chain(DEPLOYED_EXTRA)
    {
        if !values.iter().any(|w| w.to_bits() == v.to_bits()) {
            values.push(v);
        }
    }
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let first_finite = var_names.len().saturating_sub(n_finite);
    let probes_of = |v: usize| if v >= first_finite { &finite } else { &values };
    let width = var_names.len().max(1);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let total = used
        .iter()
        .fold(1usize, |acc, &v| acc.saturating_mul(probes_of(v).len()));
    let stride = total.div_ceil(DEPLOYED_MAX_ROWS).max(1);
    for pick in (0..total).step_by(stride) {
        let mut row = vec![1.7f64; width];
        let mut rem = pick;
        for &v in &used {
            let vals = probes_of(v);
            row[v] = vals[rem % vals.len()];
            rem /= vals.len();
        }
        rows.push(row);
    }
    let grid = generic_grid();
    for cfg in [1.7f64, -1.3] {
        for &v in &used {
            for &g in &grid {
                let mut row = vec![cfg; width];
                row[v] = g;
                rows.push(row);
            }
        }
    }
    if used.len() >= 2 {
        for &g in &grid {
            let mut row = vec![1.7f64; width];
            for &v in &used {
                row[v] = g;
            }
            rows.push(row);
        }
    }
    let n = rows.len();
    let cols: Vec<Vec<f64>> = (0..width)
        .map(|c| rows.iter().map(|r| r[c]).collect())
        .collect();
    let (yl, yr) = (
        tl.eval_columns(&cols, &[], n),
        tr.eval_columns(&cols, &[], n),
    );
    let same = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits();
    match (0..n).find(|&i| !same(yl[i], yr[i])) {
        Some(i) => DeployedExact {
            ok: false,
            probes: n,
            first: Some(used.iter().map(|&v| rows[i][v]).collect()),
        },
        None => DeployedExact {
            ok: true,
            probes: n,
            first: None,
        },
    }
}

/// The generic grid of the deployed bar: every 0.05 in [-25, 25] and a magnitude tail,
/// shifted off the integers. Dense enough to land between a saturation and the next integer,
/// where a step head and a saturating one part (`erf(floor(x))` is below 1 on [5.9216, 6) while
/// `erf(x)` is 1 there; `tanh` likewise on [19.06, 20)).
fn generic_grid() -> Vec<f64> {
    let mut g: Vec<f64> = Vec::new();
    g.extend((0..=1000).map(|i| -25.0 + 0.05 * i as f64));
    g.extend([
        -1e4, -1e3, -300.0, -100.0, -50.0, -30.0, 30.0, 50.0, 100.0, 300.0, 1e3, 1e4,
    ]);
    let mut out: Vec<f64> = Vec::new();
    for v in g.into_iter().map(|v| v + 0.0137421) {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod deployed_bar_tests {
    use super::*;

    fn s(t: &str) -> Vec<String> {
        t.split(' ').map(String::from).collect()
    }

    /// The bar refuses the laws of the reals that float64 breaks at a jump or a pole, and passes
    /// the float64 identities.
    #[test]
    fn deployed_exact_sees_the_jumps() {
        let e = crate::engine::Engine::from_strs(
            include_str!("../assets/engines/acj2-4/config.yaml"),
            "[]",
        )
        .expect("acj2-4 builds");
        let ops = e.operators_ref();
        let v = s("x0 x1");
        for (l, r) in [
            ("+ floor x0 1", "ceil x0"),
            ("floor + x0 1", "+ floor x0 1"),
            ("mod + x0 1 1", "mod x0 1"),
            ("floor mod x0 1", "0"),
            ("mod x0 x0", "0"),
            ("harmonic x0", "- harmonic + x0 1 inv + x0 1"),
            ("besselj 0.5 x0", "besselj 0.5 abs x0"),
            ("* floor x0 0", "0"),
            ("max x0 x1", "* 0.5 + + x0 x1 abs - x0 x1"),
            ("floor <constant>", "<constant>"),
            // a saturation and the next integer part the step head from the smooth one
            ("acosh erf floor x0", "acosh erf x0"),
            ("acosh tanh floor x0", "acosh tanh x0"),
            // a zero's sign is a value an inverse reads: ceil(asinh(-1)) is -0
            ("ceil asinh -1", "0"),
            ("mod floor x0 1", "* 0 x0"),
        ] {
            assert!(
                !deployed_exact_over(&s(l), &s(r), &v, 0, ops).ok,
                "{l} -> {r}"
            );
        }
        for (l, r) in [
            ("floor floor x0", "floor x0"),
            ("ceil neg x0", "neg floor x0"),
            ("max x0 x0", "x0"),
            ("max x0 x1", "max x1 x0"),
            ("min x0 float(\"inf\")", "x0"),
            ("erf neg x0", "neg erf x0"),
        ] {
            let d = deployed_exact_over(&s(l), &s(r), &v, 0, ops);
            assert!(d.ok, "{l} -> {r} at {:?}", d.first);
            assert!(d.probes > 300);
        }
    }

    /// The f64 bar of a rule with a 0.15 head: the deployed bar, a source constant probed as a
    /// finite variable; the ordinary axis for a bare constant target or a fitted target constant
    /// beside smooth heads; refused for a fitted target constant beside a jump head.
    #[test]
    fn f64_bar_reads_constants() {
        let e = crate::engine::Engine::from_strs(
            include_str!("../assets/engines/acj2-4/config.yaml"),
            "[]",
        )
        .expect("acj2-4 builds");
        let ops = e.operators_ref();
        let v = s("x0 x1");
        let bar = |l: &str, r: &str| f64_bar(&s(l), &s(r), &v, ops).map(|d| d.ok);
        assert_eq!(bar("sin x0", "sin x0"), None);
        assert_eq!(bar("floor + x0 1", "+ floor x0 1"), Some(false));
        assert_eq!(bar("floor floor x0", "floor x0"), Some(true));
        // a source constant alone is a variable over the finite probes
        assert_eq!(
            bar("max <constant> float(\"nan\")", "float(\"nan\")"),
            Some(true)
        );
        assert_eq!(bar("floor + <constant> 1", "+ floor x0 1"), Some(false));
        // a fitted target constant beside a jump head is refused
        assert_eq!(
            bar("floor floor <constant>", "floor <constant>"),
            Some(false)
        );
        assert_eq!(bar("floor <constant>", "<constant>"), None);
        assert_eq!(bar("erf * x0 <constant>", "erf * <constant> x0"), None);
        assert_eq!(
            bar("floor * x0 <constant>", "floor * <constant> x0"),
            Some(false)
        );
    }
}
