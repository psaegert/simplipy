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
/// SPAN MIRRORS THE JUDGE. This battery reached |c| <= 5 while
/// the gate's constant quantifier reaches |c| = 1e4 generically (`WIDE_CONSTS`)
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
    // WIDE_CONSTS: the generic magnitude span, decade-spaced, both signs
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

/// The saturation magnitudes of `SPECIAL_CONSTS`, where float64 attains a bound mathematics only
/// approaches: the judge withholds them from a target with constants, whose witness its fit
/// cannot resolve there.
pub const SATURATION_CONSTS: [f64; 10] = [
    19.0, -19.0, 20.0, -20.0, 50.0, -50.0, 750.0, -750.0, 1e17, -1e17,
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
    // shape is deliberate, not sloppy (resolved by measurement).
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
/// an infinite tolerance is not a bar.
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

/// Row semantics at the special points, each mode on its own values, refusing only at a
/// counterexample there.
///
/// - **f64** reads float64's values at the points' doubles, as the judge's f64 lane does, and
///   both of strict's readings there (`verify._contract.strict_reading`). A row where the two
///   sides agree by the mint's row agreement (`strict_row_agrees`: F6, or both finite within the
///   tolerances where the reals give both sides a value) passes. Where an operation of either
///   side overflowed or underflowed (`events`),
///   the true value at the doubles is accepted (the range standard, `hiprec::range_rescue`; a row
///   the table cannot read passes, as the judge skips it). The left side without a value against
///   a value is an extension at a point, measured by the domain gate, and passes. Two finite
///   values that differ pass where the reals give both sides a value at the doubles and agree at
///   the point, which a point can read and a row cannot (strict's second reading, `kind_holds`:
///   `pow(sin(x), inf) -> 0` at pi/2's double, 1 against 0, where the reals' left side is 0).
///   Any other disagreement refuses.
/// - **permissive** reads the reals' values at the points themselves (pi/2 is pi/2), each side
///   as the double nearest to it (P3, `point_doubles`), at every point, as the judge's permissive
///   lane does: float64 agreeing at a point is no reading of the reals there (`asin(1.0)` is
///   pi/2 exactly, where `ceil(cos(asin(x)))` is 0 and float64 gives 1). The two doubles
///   agreeing (F6; a fitted constant within its band) passes, and so does an extension (the left
///   side without a value); anything else refuses: `ceil(asinh(cos(x))) -> ceil(cos(x))` is 0
///   against -1 at x = pi, and `asin(1e-8) -> 1e-8` agrees. A point the table cannot read passes.
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
    mode: crate::engine::RuleMode,
) -> bool {
    use crate::engine::RuleMode;
    use crate::hiprec::PointCmp as P;
    let lane_f64 = matches!(mode, RuleMode::Default);
    let agree = |s: f64, c: f64| {
        if cand_params.is_empty() {
            crate::table::f64_agree(s, c) == crate::table::Cmp::Eq
        } else {
            dep_realised(s, c, rtol, atol) == P::Eq
        }
    };
    // the reals' reading at a point, each side as the double nearest to it (P3): a counterexample
    // there, or none
    let reals_refute = |r: usize| {
        use crate::interval::FloatReading;
        match point_reading(
            source,
            src_params,
            cand,
            cand_params,
            ops,
            var_names,
            &battery.atoms[r],
        ) {
            FloatReading::Agree | FloatReading::Unread => false,
            // a fitted constant agrees within its band, which the enclosure does not read
            FloatReading::Differ => cand_params.is_empty(),
            FloatReading::Doubles(a, b) => !(agree(a, b) || (a.is_nan() && !b.is_nan())),
        }
    };
    // do the reals give both sides a value at the point's doubles (`kind_holds`)?
    let valued = |r: usize| {
        let cols1: Vec<Vec<f64>> = battery.cols.iter().map(|col| vec![col[r]]).collect();
        crate::hiprec::reals_both_valued(
            source,
            src_params,
            cand,
            cand_params,
            ops,
            var_names,
            &cols1,
            0,
        )
    };
    for r in 0..battery.n_rows {
        let (s, c) = (y_src[r], y_cand[r]);
        let reject = if !lane_f64 {
            reals_refute(r)
        } else if strict_row_agrees(s, c, rtol, atol, !cand_params.is_empty(), || valued(r))
            || (kind_holds(s, c, || valued(r)) && !reals_refute(r))
        {
            false
        } else if events.is_some_and(|e| e[r]) {
            let cols1: Vec<Vec<f64>> = battery.cols.iter().map(|col| vec![col[r]]).collect();
            !crate::hiprec::range_rescue(
                &[s],
                &[c],
                Some(&[true]),
                None,
                source,
                src_params,
                cand,
                cand_params,
                ops,
                var_names,
                &cols1,
                rtol,
                atol,
            )
        } else {
            !(s.is_nan() && !c.is_nan())
        };
        if reject {
            if std::env::var("SIMPLIPY_BATTERY_TRACE").is_ok() {
                eprintln!(
                    "BATTERY-REJECT src=[{}] sp={:?} cand=[{}] cp={:?} row={} coords={:?} s={:e} c={:e}",
                    source.join(" "), src_params, cand.join(" "), cand_params, r,
                    battery.atoms[r].iter().map(|a| a.value()).collect::<Vec<_>>(), s, c,
                );
            }
            return false;
        }
    }
    true
}

/// Both sides' values in the reals at a battery point (`atoms`, read exactly: pi/2 is pi/2, each
/// `<constant>` its double), as the doubles nearest to them -- permissive's reading of a value
/// (P3, `interval::float_resolution`: a rigorous enclosure at the point, else the table's
/// readings at the precision the point demands; running out of precision is no reading). The
/// judge's permissive lane reads its points so (`_contract._float_resolution`).
fn point_reading(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    atoms: &[ProbeAtom],
) -> crate::interval::FloatReading {
    match spelled_at(source, src_params, target, tgt_params, var_names, atoms) {
        Some((ls, rs)) => crate::interval::float_resolution(&ls, &rs, &[], &[], ops),
        None => crate::interval::FloatReading::Unread,
    }
}

/// Both sides spelled at a battery point: each variable as its atom's tokens (`atom_tokens`),
/// each `<constant>` as its double; `None` where a parameter is missing.
fn spelled_at(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    var_names: &[String],
    atoms: &[ProbeAtom],
) -> Option<(Vec<String>, Vec<String>)> {
    let spell = |tokens: &[String], params: &[f64]| -> Option<Vec<String>> {
        let mut out = Vec::with_capacity(tokens.len());
        let mut k = 0usize;
        for t in tokens {
            if t == "<constant>" {
                out.push(crate::numeric::py_float_repr(*params.get(k)?));
                k += 1;
            } else if let Some(i) = var_names.iter().position(|v| v == t) {
                out.extend(atom_tokens(atoms.get(i)?));
            } else {
                out.push(t.clone());
            }
        }
        Some(out)
    };
    Some((spell(source, src_params)?, spell(target, tgt_params)?))
}

/// A battery atom spelled as tokens the table reads exactly: a double as its literal, `n pi / d`
/// as `(n * np.pi) / d`, `e` as `np.e`, -1.7 and `n / d` as decimals' quotients (`*` and `/` are
/// the core's own operators, read whatever the configuration holds).
fn atom_tokens(a: &ProbeAtom) -> Vec<String> {
    let s = |x: &str| x.to_string();
    let num = |x: f64| crate::numeric::py_float_repr(x);
    match a {
        ProbeAtom::Val(v) => vec![num(*v)],
        ProbeAtom::PiFrac(n, d) => vec![s("/"), s("*"), num(*n as f64), s("np.pi"), num(*d as f64)],
        ProbeAtom::E => vec![s("np.e")],
        ProbeAtom::Dec17Neg => vec![s("-1.7")],
        ProbeAtom::Rat(n, d) => vec![s("/"), num(*n as f64), num(*d as f64)],
    }
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

    /// The saturation magnitudes are the special constants' last ten, the ones the judge
    /// withholds from a target with constants.
    #[test]
    fn saturation_constants_end_the_special_constants() {
        let tail = &SPECIAL_CONSTS[SPECIAL_CONSTS.len() - SATURATION_CONSTS.len()..];
        assert_eq!(tail, &SATURATION_CONSTS[..]);
    }

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
    fn the_pre_screen_asks_realisation() {
        use crate::hiprec::PointCmp as P;
        // `pow np.e sinh (-5) -> 0`: the deployed sides are 5.94e-33 and 0.0, ~4.5e18 units in
        // the last place apart: no agreement, whatever an absolute floor would read
        let (s, c) = (5.942_307_292_381_135e-33, 0.0);
        assert_eq!(dep_realised(s, c, 0.0, 0.0), P::RealChange);
        let a = 1.0_f64;
        assert_eq!(
            dep_realised(a, f64::from_bits(a.to_bits() + 3), 0.0, 0.0),
            P::Eq
        );
        assert_eq!(
            dep_realised(a, f64::from_bits(a.to_bits() + 100), 0.0, 0.0),
            P::RealChange,
            "100 ULP is past the realisation bar"
        );

        // the mine's own witness band rides on top of it
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
    /// With `defer_range` (`f64_bar_sorted`): the probes where the sides differ and an
    /// operation of either side overflowed or underflowed, left to the caller, which accepts
    /// one where the rewrite gives the true value (the range standard). Empty otherwise.
    pub range_rows: Vec<Vec<f64>>,
}

/// The 0.15 heads. A rule spelled with one answers to the deployed bar for its float64
/// realisation (`f64_bar`): a step head's jump (`floor(x + 1) = floor(x) + 1` fails at
/// x = -1e-17), a zero's sign that `atan2` reads, a pole of `harmonic` -- float64 parts from
/// the reals there, at points the contract's battery does not reach.
pub const NEW_HEADS: [&str; 13] = [
    "floor", "ceil", "erf", "harmonic", "max", "min", "mod", "atan2", "besselj", "erfc", "bessely",
    "besseli", "besselk",
];

/// The heads whose value at a zero operand depends on the zero's sign (F7): `1/(-0)` is -inf,
/// `atan2(-0, -1)` is -pi, `pow(-0, -3)` and `rootn(-0, -3)` are -inf. Where a rule holds one,
/// the deployed bar tells the two zeros apart.
const ZERO_SIGN_READERS: [&str; 5] = ["/", "inv", "pow", "rootn", "atan2"];

/// The 0.15 heads whose float64 value jumps (a step, a remainder's wrap): a fitted target
/// constant cannot be judged bit for bit beside them.
const JUMP_HEADS: [&str; 3] = ["floor", "ceil", "mod"];

/// Does `tokens` hold a 0.15 head (as an operator of `ops`)?
pub fn holds_new_head(tokens: &[String], ops: &Operators) -> bool {
    tokens
        .iter()
        .any(|t| ops.arity_of(t).is_some() && NEW_HEADS.contains(&t.as_str()))
}

/// How the deployed bar reads float64's two sides at a probe: strict's two readings
/// (`verify._contract.strict_reading`). `F6`: they agree (`table::f64_agree`). `Kind`: they agree
/// in kind -- both finite, the same infinity, or both undefined -- which strict's (b) asks of a
/// rule true in the reals: there their float64 difference is rounding, however far a jump or a
/// pole between float64's argument and the real one magnifies it (`floor(x + 1) -> floor(x) + 1`
/// is 1 against 0 at x = -1e-17). In both, a zero's sign counts where a head of the rule reads it
/// (`ZERO_SIGN_READERS`), and a fully numeric pair is bit-exact (`table::f64_same`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Agreement {
    F6,
    Kind,
}

/// STRICT'S SECOND READING AT ONE PROBE of float64's two sides `a` and `b`, the core's one
/// definition (`verify._contract.kind_holds`): both finite, at a point where the reals give both
/// sides a value (`valued()`, `hiprec::reals_both_valued` at the probe's doubles), where their
/// difference is rounding. At any other point they agree only as F6 asks
/// (`tanh(1/(abs(x) - x))` is 1 against `tanh(-1/(x - abs(x)))`'s -1 on x > 0, where the reals
/// give neither side a value).
pub fn kind_holds(a: f64, b: f64, valued: impl FnOnce() -> Option<bool>) -> bool {
    a.is_finite() && b.is_finite() && valued() == Some(true)
}

/// THE MINT'S ROW AGREEMENT in f64, strict's reading at a row of the mine's matrix, its one
/// definition: the source `s` without a value (an extension at the row, which the domain gate
/// measures), or (a) F6 (`table::f64_agree`: the same infinity included), or (b) the two within
/// the tolerance of where the candidate's value `c` came from. An evaluated side (`fitted`
/// false) is off by rounding: within the mine's tolerances (`atol + rtol*|c|`, the prefilter's
/// bound on strict's unbounded (b)) and of one kind where the reals give both sides a value
/// (`kind_holds`, `valued` read only here). A side holding a fitted constant (`fitted`) is off by
/// the fit: within its witness band (`dep_realised`; `besselj(cos(1), C) -> C'` at C = 0.1, a
/// witness 3.2e-12 off). The constant-free rows (`worker::strict_rows_agree`), the range
/// standard's re-reading of the rows where nothing left the range (`hiprec::range_rescue`), the
/// special points (`rows_consistent`) and the overlap gate's prefilter reading
/// (`worker::prefilter_agrees`) read it.
pub fn strict_row_agrees(
    s: f64,
    c: f64,
    rtol: f64,
    atol: f64,
    fitted: bool,
    valued: impl FnOnce() -> Option<bool>,
) -> bool {
    if s.is_nan() || crate::table::f64_agree(s, c) == crate::table::Cmp::Eq {
        return true;
    }
    if fitted {
        return dep_realised(s, c, rtol, atol) == crate::hiprec::PointCmp::Eq;
    }
    (s - c).abs() <= atol + rtol * c.abs() && kind_holds(s, c, valued)
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
    f64_bar_sorted(lhs, rhs, var_names, &[], false, Agreement::F6, ops)
}

/// `f64_bar` over each variable's SORT, the contract judge's reading of it: `sorts[i]` is the
/// slot sort of `var_names[i]` -- `?` (data) and `!` are probed over finite doubles only, `$`
/// over nonzero finite ones, `_` (and a variable with no sort given) over every probe, the
/// infinities and NaN included -- and with `defer_range`, a probe where the sides differ and an
/// operation overflowed or underflowed is returned in `range_rows` instead of failing the bar
/// (the range standard: there the true value is accepted, which only the caller can judge).
/// `ceil(1/cosh(x)) -> 1` holds in f64 so: x = +-inf is no data, and past |x| = 710.48 `cosh`
/// overflows and the rewrite gives the true value 1. `agreement`: how a probe's two sides are read
/// (`Agreement`).
pub fn f64_bar_sorted(
    lhs: &[String],
    rhs: &[String],
    var_names: &[String],
    sorts: &[char],
    defer_range: bool,
    agreement: Agreement,
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
                range_rows: Vec::new(),
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
    Some(deployed_exact_over(
        &lhs_v,
        rhs,
        &names,
        n_c,
        sorts,
        defer_range,
        agreement,
        ops,
    ))
}

/// The most range rows of the f64 bar the mint reads by the table (the judge's
/// `RANGE_ROWS_CAP`); a bar with more is undecided, and fails closed.
const MINT_RANGE_ROWS_CAP: usize = 256;

/// THE MINT'S f64 BAR, the judge's reading of it (`verify._contract`'s deployed check): every
/// mined variable is data, a `?` slot probed over finite doubles only, and at a probe where the
/// sides differ and an operation overflowed or underflowed the rewrite may give the true value
/// (the range standard), which the table decides (`range_row_true`). `true` where no 0.15 head
/// asks the bar. `ceil(1/cosh(x)) -> 1` passes: x = +-inf is no data, and past |x| = 710.48
/// `cosh` overflows and the true value is 1. A pre-filter of strict's judgment, it reads the
/// probes in kind (`Agreement::Kind`): a change of kind refuses the rule in both of strict's
/// readings, while a finite difference is the judge's to decide (strict's (b) admits it where the
/// rule is true in the reals).
pub fn f64_bar_mint(lhs: &[String], rhs: &[String], var_names: &[String], ops: &Operators) -> bool {
    let sorts = vec!['?'; var_names.len()];
    let Some(bar) = f64_bar_sorted(lhs, rhs, var_names, &sorts, true, Agreement::Kind, ops) else {
        return true;
    };
    bar.ok
        && bar.range_rows.len() <= MINT_RANGE_ROWS_CAP
        && bar
            .range_rows
            .iter()
            .all(|row| range_row_true(lhs, rhs, var_names, row, ops))
}

/// THE f64 MAGNITUDE LADDER AT THE MINT (`table::F64_MAGNITUDES`), the judge's f64 lane read the
/// same way (`_contract`'s `magnitudes`): along each variable either side uses, every magnitude of
/// the ladder in both signs, the other variables at 1.7 and then at -1.3. A pre-filter of strict's
/// judgment, it reads a probe by strict's two readings: a probe where float64's two sides agree
/// (F6), or agree in kind where the reals give both a value (`kind_holds`), passes
/// (`sinh(asinh(x)) -> x`, 66 units in the last place apart at 1e100, is the judge's to decide);
/// one where an operation of either side overflowed or underflowed passes
/// where the rewrite gives the true value (`range_row_true`, the table; an unreadable one passes,
/// as the judge skips it); one where the left side alone has no value passes where that is
/// isolated along a variable (`isolated_extension`, the judge's `isolated_f64`); any other change
/// of kind refuses, as it refuses strict in the judge. A pair holding a `<constant>` is left to
/// the judge, which reads the ladder at its own fitted witness.
pub fn f64_magnitudes_mint(
    lhs: &[String],
    rhs: &[String],
    var_names: &[String],
    ops: &Operators,
) -> bool {
    if lhs.iter().chain(rhs).any(|t| t == "<constant>") {
        return true;
    }
    let used: Vec<usize> = (0..var_names.len())
        .filter(|&i| lhs.iter().chain(rhs).any(|t| *t == var_names[i]))
        .collect();
    if used.is_empty() {
        return true;
    }
    let (Ok(tl), Ok(tr)) = (
        crate::eval::Tape::compile(lhs, ops, var_names),
        crate::eval::Tape::compile(rhs, ops, var_names),
    ) else {
        return true; // a side the tape cannot read is the judge's to refuse
    };
    let width = var_names.len().max(1);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for cfg in [1.7f64, -1.3] {
        for &v in &used {
            for &m in &crate::table::F64_MAGNITUDES {
                for x in [m, -m] {
                    let mut row = vec![cfg; width];
                    row[v] = x;
                    rows.push(row);
                }
            }
        }
    }
    let eval = |rows: &[Vec<f64>]| {
        let n = rows.len();
        let cols: Vec<Vec<f64>> = (0..width)
            .map(|c| rows.iter().map(|r| r[c]).collect())
            .collect();
        let (yl, el) = tl.eval_columns_events(&cols, &[], n);
        let (yr, er) = tr.eval_columns_events(&cols, &[], n);
        (
            yl,
            yr,
            el.iter()
                .zip(&er)
                .map(|(a, b)| *a || *b)
                .collect::<Vec<bool>>(),
        )
    };
    let (yl, yr, events) = eval(&rows);
    let valued = |row: &[f64]| {
        let cols1: Vec<Vec<f64>> = row.iter().map(|&x| vec![x]).collect();
        crate::hiprec::reals_both_valued(lhs, &[], rhs, &[], ops, var_names, &cols1, 0)
    };
    for (i, row) in rows.iter().enumerate() {
        if crate::table::f64_agree(yl[i], yr[i]) == crate::table::Cmp::Eq
            || kind_holds(yl[i], yr[i], || valued(row))
        {
            continue;
        }
        if events[i] {
            if range_row_true(lhs, rhs, var_names, row, ops) {
                continue;
            }
            return false;
        }
        if yl[i].is_nan() && !yr[i].is_nan() && isolated_extension(&tl, &tr, row, &used) {
            continue;
        }
        return false;
    }
    true
}

/// Is float64's extension at `row` -- the left side undefined, the right side a value -- ISOLATED,
/// a null set of the data (ruling 15)? It is where, along some variable of `along`, the
/// neighbouring doubles on both sides are no extension: the left side has a value there, or the
/// right side has none, or an operation of either side overflowed or underflowed there, where the
/// true value stands (the range standard). The judge's `isolated_f64`, and the one reading of
/// every f64 probe of the core (the bar and the magnitude ladder): `x/x -> 1` at 0, where
/// 5e-324/5e-324 is 1; `atan2(x0, x1)/atan2(-x0, x1) -> -1` at x0 = 0, x1 = 2, where
/// `atan2(5e-324, 2)` underflows to 0.
fn isolated_extension(
    tl: &crate::eval::Tape,
    tr: &crate::eval::Tape,
    row: &[f64],
    along: &[usize],
) -> bool {
    along.iter().any(|&v| {
        if !row[v].is_finite() {
            return false;
        }
        let cols: Vec<Vec<f64>> = (0..row.len())
            .map(|c| {
                if c == v {
                    vec![
                        crate::ac::rat::next_up(row[c]),
                        crate::ac::rat::next_down(row[c]),
                    ]
                } else {
                    vec![row[c]; 2]
                }
            })
            .collect();
        let (nl, el) = tl.eval_columns_events(&cols, &[], 2);
        let (nr, er) = tr.eval_columns_events(&cols, &[], 2);
        (0..2).all(|k| el[k] || er[k] || !(nl[k].is_nan() && !nr[k].is_nan()))
    })
}

/// Does the rewrite give the TRUE value at one range row of the f64 bar? Both sides read by the
/// table at the row (the left side's `<constant>`s named and valued as `f64_bar_sorted` appends
/// them): the same class, and finite values equal to the working precision. A row the table
/// cannot read says nothing about the true value and passes, as the judge skips it.
fn range_row_true(
    lhs: &[String],
    rhs: &[String],
    var_names: &[String],
    row: &[f64],
    ops: &Operators,
) -> bool {
    use crate::table::{class_of, Ctx, Expr};
    let mut names = var_names.to_vec();
    let mut k = 0usize;
    let lt: Vec<String> = lhs
        .iter()
        .map(|t| {
            if t == "<constant>" {
                let name = format!("<c{k}>");
                names.push(name.clone());
                k += 1;
                name
            } else {
                t.clone()
            }
        })
        .collect();
    let arity: std::collections::HashMap<String, u8> = lt
        .iter()
        .chain(rhs)
        .filter_map(|t| ops.arity_of(t).map(|a| (t.clone(), a)))
        .collect();
    let (Ok(le), Ok(re)) = (
        Expr::compile(&lt, &names, &arity),
        Expr::compile(rhs, &names, &arity),
    ) else {
        return false;
    };
    let env: Vec<crate::table::Val> = row.iter().map(|v| crate::table::val_of_f64(*v)).collect();
    const DPS: i32 = 50;
    let (Some(l), Some(r)) = (
        le.eval_real(&env, DPS).and_then(|o| o.value),
        re.eval_real(&env, DPS).and_then(|o| o.value),
    ) else {
        return true;
    };
    let (lu, li) = class_of(&l);
    let (ru, ri) = class_of(&r);
    if lu || ru || li.is_some() || ri.is_some() {
        return lu == ru && li == ri;
    }
    let Some(mut ctx) = Ctx::new(DPS) else {
        return false;
    };
    let p = crate::hiprec::dps_to_prec(DPS);
    let (a, b) = (ctx.value_bf(&l), ctx.value_bf(&r));
    let d = a.sub(&b, p, crate::hiprec::RM).abs();
    let scale = a.abs().max(&b.abs());
    let tol = scale.mul(&crate::hiprec::pow10(-(DPS - 10), p), p, crate::hiprec::RM);
    matches!(d.cmp(&tol), Some(o) if o <= 0)
}

/// THE DEPLOYED BAR: do `lhs` and `rhs` AGREE in float64 (F6, `table::f64_agree`: NaN with NaN,
/// one infinity, finite values within `REALISED_ULP`, the two zeros as one value), with the
/// engine's own kernels, at every probe? The zeros are told apart by their sign where a head of
/// the rule reads a zero's sign (`ZERO_SIGN_READERS`: `1/x`, `atan2`, ...), and nowhere else:
/// `floor(floor(x) + 0.5) -> floor(x)` is +0 against -0 at x = -0, one value. A probe where the
/// left side alone has no value is excused where it is isolated: a data (`?`) variable's move to
/// its neighbouring double, in each direction, gives the left side a value there, a null set of
/// the data, which float64's constructors fill as the rewrite does (`ceil(atan(x)/x) -> 1` at
/// x = +-0). The probes are, per variable of `var_names` either side uses, the `BATTERY` points,
/// +-inf, NaN and `DEPLOYED_EXTRA` (their full product); a generic grid per variable with the
/// others at 1.7 and -1.3; and the diagonal of that grid. The last `n_finite` names of
/// `var_names` are fitted constants, probed over the finite values only.
///
/// It is the verdict of the f64 bar (`f64_bar`), which the contract judge ANDs into strict's two
/// readings, each read its way (`agreement`), and which the f64 mint applies in kind to every
/// pick, so the mint refuses only what the judge refuses. A side holding a `<constant>`, or one
/// that does not compile, fails.
#[allow(clippy::too_many_arguments)]
fn deployed_exact_over(
    lhs: &[String],
    rhs: &[String],
    var_names: &[String],
    n_finite: usize,
    sorts: &[char],
    defer_range: bool,
    agreement: Agreement,
    ops: &Operators,
) -> DeployedExact {
    let refused = DeployedExact {
        ok: false,
        probes: 0,
        first: None,
        range_rows: Vec::new(),
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
    let nonzero: Vec<f64> = finite.iter().copied().filter(|v| *v != 0.0).collect();
    let first_finite = var_names.len().saturating_sub(n_finite);
    let probes_of = |v: usize| {
        if v >= first_finite {
            return &finite;
        }
        match sorts.get(v) {
            Some('?' | '!') => &finite,
            Some('$') => &nonzero,
            _ => &values,
        }
    };
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
    let reads_zero_sign = lhs
        .iter()
        .chain(rhs)
        .any(|t| ops.arity_of(t).is_some() && ZERO_SIGN_READERS.contains(&t.as_str()));
    // a pair without variables is a fully numeric rewrite, bit-exact in f64 (F8): the same
    // double, a zero's sign included, and no range standard (`table::f64_same`)
    let ground = used.is_empty();
    // do the reals give both sides a value at a probe's doubles (`kind_holds`)?
    let valued = |row: &[f64]| {
        let cols1: Vec<Vec<f64>> = row.iter().map(|&x| vec![x]).collect();
        crate::hiprec::reals_both_valued(lhs, &[], rhs, &[], ops, var_names, &cols1, 0)
    };
    let same = |a: f64, b: f64, row: &[f64]| {
        if ground {
            return crate::table::f64_same(a, b);
        }
        if a == 0.0 && b == 0.0 {
            return !reads_zero_sign || a.to_bits() == b.to_bits();
        }
        crate::table::f64_agree(a, b) == crate::table::Cmp::Eq
            || (agreement == Agreement::Kind && kind_holds(a, b, || valued(row)))
    };
    // the left side's NaN at a probe is isolated along a data variable (`isolated_extension`)
    let data: Vec<usize> = used
        .iter()
        .copied()
        .filter(|&v| v < first_finite && sorts.get(v) == Some(&'?'))
        .collect();
    let isolated_nan = |row: &[f64]| isolated_extension(&tl, &tr, row, &data);
    // The range standard's reading of a differing probe: did an operation of either side
    // overflow or underflow there (`table::Expr::eval_f64`, the same kernels)?
    let range_reader = defer_range
        .then(|| {
            let arity: std::collections::HashMap<String, u8> = lhs
                .iter()
                .chain(rhs)
                .filter_map(|t| ops.arity_of(t).map(|a| (t.clone(), a)))
                .collect();
            let l = crate::table::Expr::compile(lhs, var_names, &arity).ok()?;
            let r = crate::table::Expr::compile(rhs, var_names, &arity).ok()?;
            Some((l, r))
        })
        .flatten();
    let out_of_range = |row: &[f64]| {
        range_reader.as_ref().is_some_and(|(l, r)| {
            let in_l = l.eval_f64(row).is_none_or(|(_, ok)| ok);
            let in_r = r.eval_f64(row).is_none_or(|(_, ok)| ok);
            !(in_l && in_r)
        })
    };
    let mut range_rows: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        if same(yl[i], yr[i], &rows[i])
            || (yl[i].is_nan() && !yr[i].is_nan() && isolated_nan(&rows[i]))
        {
            continue;
        }
        if !ground && out_of_range(&rows[i]) {
            range_rows.push(used.iter().map(|&v| rows[i][v]).collect());
            continue;
        }
        return DeployedExact {
            ok: false,
            probes: n,
            first: Some(used.iter().map(|&v| rows[i][v]).collect()),
            range_rows,
        };
    }
    DeployedExact {
        ok: true,
        probes: n,
        first: None,
        range_rows,
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
            // a zero's sign is a value where a head of the rule reads it: 0/x is -0 for x < 0
            ("* 0 inv x0", "0"),
        ] {
            assert!(
                !deployed_exact_over(&s(l), &s(r), &v, 0, &[], false, Agreement::F6, ops).ok,
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
            // F6: the two zeros are one value where no head of the rule reads a zero's sign,
            // and values 8 units in the last place apart agree
            ("mod floor x0 1", "* 0 x0"),
            ("floor + floor x0 0.5", "floor x0"),
            ("erf * 0.1 x0", "erf / x0 10"),
        ] {
            let d = deployed_exact_over(&s(l), &s(r), &v, 0, &[], false, Agreement::F6, ops);
            assert!(d.ok, "{l} -> {r} at {:?}", d.first);
            assert!(d.probes > 300);
        }
        // a fully numeric rewrite is bit-exact (F8): ceil(asinh(-1)) is -0, not the double 0
        let ground = |r: &str| {
            deployed_exact_over(
                &s("ceil asinh -1"),
                &s(r),
                &v,
                0,
                &[],
                false,
                Agreement::F6,
                ops,
            )
        };
        assert!(!ground("0").ok && ground("-0.0").ok);
    }

    /// An isolated NaN of the left side, where the right side has a value, is excused at a data
    /// (`?`) variable: its neighbouring doubles give the left side a value. A NaN on positive
    /// measure is not, nor one at a slot of another sort (a `_` can be constant on an interval).
    #[test]
    fn the_bar_excuses_an_isolated_nan_of_the_left_side() {
        let e = crate::engine::Engine::from_strs(
            include_str!("../assets/engines/acj2-4/config.yaml"),
            "[]",
        )
        .expect("acj2-4 builds");
        let ops = e.operators_ref();
        let v = s("x0");
        let bar = |l: &str, r: &str, sort: char| {
            deployed_exact_over(&s(l), &s(r), &v, 0, &[sort], false, Agreement::F6, ops).ok
        };
        assert!(bar("ceil / atan x0 x0", "1", '?'));
        assert!(!bar("ceil / atan x0 x0", "1", '_'));
        assert!(!bar("floor acosh x0", "0", '?'));
    }

    /// An extension is isolated where its neighbouring doubles are no extension, an operation
    /// that underflows there included (the range standard): `atan2(x0, x1)/atan2(-x0, x1)` is 0/-0
    /// on x0 = 0, x1 > 0, and beside it `atan2(5e-324, 2)` underflows to 0.
    #[test]
    fn an_extension_isolated_by_an_underflow_is_a_null_set() {
        let e = crate::engine::Engine::from_strs(
            include_str!("../assets/engines/acj2-4/config.yaml"),
            "[]",
        )
        .expect("acj2-4 builds");
        let ops = e.operators_ref();
        let v = s("x0 x1");
        let (l, r) = (s("/ atan2 x0 x1 atan2 neg x0 x1"), s("-1"));
        for agreement in [Agreement::F6, Agreement::Kind] {
            let d = deployed_exact_over(&l, &r, &v, 0, &['?', '?'], false, agreement, ops);
            assert!(d.ok, "{agreement:?} at {:?}", d.first);
        }
    }

    /// In kind the bar reads a finite difference as agreement where the reals give both sides a
    /// value at the probe (strict's second reading, `kind_holds`), and still refuses a change of
    /// kind, a zero's sign a head reads, and a finite difference where the reals give a side no
    /// value (`tanh(1/(abs(x) - x))` is 1 against -1 on x > 0, where 1/0 has none).
    #[test]
    fn in_kind_the_bar_refuses_only_a_change_of_kind() {
        let e = crate::engine::Engine::from_strs(
            include_str!("../assets/engines/acj2-4/config.yaml"),
            "[]",
        )
        .expect("acj2-4 builds");
        let ops = e.operators_ref();
        let v = s("x0 x1");
        let bar = |l: &str, r: &str, agreement| {
            deployed_exact_over(&s(l), &s(r), &v, 0, &['?', '?'], false, agreement, ops).ok
        };
        for (l, r) in [
            ("floor + x0 1", "+ floor x0 1"),
            ("mod + x0 1 1", "mod x0 1"),
            ("floor mod x0 1", "0"),
        ] {
            assert!(
                !bar(l, r, Agreement::F6) && bar(l, r, Agreement::Kind),
                "{l} -> {r}"
            );
        }
        for (l, r) in [
            // tanh(x) is 1.0 on [19.06, 20), where acosh of it is 0 and of tanh(19) NaN
            ("acosh tanh floor x0", "acosh tanh x0"),
            ("* 0 inv x0", "0"),
            ("tanh inv - abs x0 x0", "tanh neg inv - x0 abs x0"),
        ] {
            assert!(!bar(l, r, Agreement::Kind), "{l} -> {r}");
        }
        assert!(kind_holds(1.0, -2.0, || Some(true)));
        assert!(!kind_holds(1.0, -2.0, || Some(false)) && !kind_holds(1.0, -2.0, || None));
        assert!(!kind_holds(f64::INFINITY, 1.0, || Some(true)));
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
