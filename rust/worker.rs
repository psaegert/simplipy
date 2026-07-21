//! The no-constant equivalence test + rule selection for the OFFLINE miner.
//!
//! This is the constant-FREE candidate branch of `find_rule_worker` plus the
//! winner-selection. It adds NO new numerics -- the only math is
//! `allclose_extends` (bit-exact vs numpy) -- so it is a pure control-flow port. Together with the
//! constant-fit branch (`crate::fit`) these are the two halves of the per-candidate decision,
//! assembled over the candidate library + generation + Kruskal prune into the full native miner.

use crate::eval::{allclose_extends, columns_from_row_major, Tape};
use crate::fit::Rng;
use crate::operators::Operators;

/// The non-increasing wildcard-multiplicity condition: a rule `lhs -> rhs` violates it
/// when any wildcard token (`^_\d+$`) occurs MORE times on the rhs than the lhs (it would duplicate a
/// matched subtree). Faithfully matches `^_\d+$`, so on dummy-variable (`x0`..) mining expressions it
/// is inert exactly as in Python -- selection then reduces to "fewest `<constant>` first".
pub fn violates_wildcard_multiplicity(lhs: &[String], rhs: &[String]) -> bool {
    use rustc_hash::FxHashMap;
    let is_wc = |t: &str| -> bool {
        let b = t.as_bytes();
        b.len() >= 2 && b[0] == b'_' && b[1..].iter().all(|c| c.is_ascii_digit())
    };
    let mut lhs_wc: FxHashMap<&str, i32> = FxHashMap::default();
    for t in lhs {
        if is_wc(t) {
            *lhs_wc.entry(t).or_insert(0) += 1;
        }
    }
    let mut rhs_wc: FxHashMap<&str, i32> = FxHashMap::default();
    for t in rhs {
        if is_wc(t) {
            *rhs_wc.entry(t).or_insert(0) += 1;
        }
    }
    rhs_wc
        .iter()
        .any(|(w, &c)| c > lhs_wc.get(w).copied().unwrap_or(0))
}

/// Sign-combination vectors for the source-constant test grid. `n == 0` -> a single empty
/// combo (so a const-free source still runs once).
///
/// The grid is `{-1, +1}^n` (signed-continuous magnitudes), NOT `{-1, 0, 1}^n`. A `0` entry
/// would be a measure-zero ATOM in the test data: it spuriously certifies rules true only at
/// C=0 (`pow(nan,C)->1`, `pow(x,inf)->0` via a const exponent) AND blocks true multiplicative
/// power laws that fail only at the measure-zero point C=0. By the measure-zero principle no
/// special constant value gets weight (neither 0, the additive neutral, nor C=1, the
/// multiplicative one). Sign coverage `{-1,+1}` is kept (continuous, not an atom):
/// `sqrt`/`log`/`pow` need both signs of C tested for domain reasons. Env
/// `SIMPLIPY_ZERO_SIGN=1` restores the old `{-1,0,1}` grid (reference/repro only).
fn sign_combos(n: usize) -> Vec<Vec<f64>> {
    static WITH_ZERO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let signs: &[f64] = if *WITH_ZERO.get_or_init(|| std::env::var("SIMPLIPY_ZERO_SIGN").is_ok()) {
        &[-1.0, 0.0, 1.0]
    } else {
        &[-1.0, 1.0]
    };
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        let mut next = Vec::with_capacity(out.len() * signs.len());
        for base in &out {
            for &s in signs {
                let mut v = base.clone();
                v.push(s);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// Magnitudes straddling the +-1 pole thresholds where a unary op crosses 1 (asin at sin 1 ~= 0.84,
/// atanh at tanh 1 ~= 0.76, mult_k at 1/k, ...), plus small and large tails.
///
/// Retirement was tried and REJECTED: the exact reachability gate does NOT replace this grid.
/// The grid's load-bearing content is not the magnitude sweep at all -- it is the three
/// INTEGER entries {5, 30, 500}. `np.power(negative, C)` is defined iff C is an INTEGER, while
/// `pow1_3`/`pow1_5` are sign-preserving real roots (total on R), so `pow1_3(pow(x, C))` is defined
/// on x<0 exactly when C is an integer, and the rhs constant is pinned to C/k by the x>0 branch and
/// cannot repair it (`pow1_4(pow(x,4))` at x=-2: source +2.0, target -2.0, both DEFINED). That
/// failure set is measure-ZERO in C but positive-measure in x, and the source constant is
/// universally quantified -- so NO continuous sampler can find it (`const_magnitude` hits Z with
/// probability 0) and no value-class analysis sees it either.
///
/// Known costs, kept deliberately until the two fixes below land: the grid rejects some true
/// power laws for FITTER-POWER reasons (at C=+-1000 few rows of `inv(x^C)` carry fittable
/// signal), it catches some artifacts by luck rather than by construction, and it MINTS one
/// artifact of its own -- `pow(div5 C, nan) -> <constant>` exists only because div5(5.0) == 1.0
/// exactly and IEEE says pow(1, nan) = 1.
///
/// To retire it, two exact fixes are needed: (1) INTEGRALITY classes in the source-constant
/// challenge set (integers/parity/k-divisibility -- the grid's only causal content, made principled
/// and complete); (2) a BIDIRECTIONAL reachability gate: it currently checks the candidate reaches
/// every value the source reaches, but not that the candidate stays UNDEFINED where the source is.
const POLE_GRID: [f64; 26] = [
    0.02, 0.1, 0.18, 0.22, 0.24, 0.26, 0.3, 0.34, 0.4, 0.45, 0.49, 0.51, 0.6, 0.7, 0.76, 0.84, 0.9,
    0.95, 0.99, 1.01, 1.2, 1.6, 2.5, 5.0, 30.0, 500.0,
];

/// Per-challenge source-constant MAGNITUDE vectors (multiplied by the sign combos downstream, so
/// only |C| is chosen here). The `POLE_GRID` is swept on EACH constant in turn, with the OTHER
/// constants drawn randomly, so a narrow pole band in ANY single constant is exercised
/// DETERMINISTICALLY (nlc==1 reduces to a plain grid sweep). Missing such a band admits
/// step-function artifacts (`pow(asin(C), inf) -> 0`, and its multi-constant scalings
/// `* C1 pow(asin(C2), inf) -> 0`, which random sampling lands in only ~2%/draw). Fully-random
/// rounds are appended for off-grid decorrelation and joint coverage.
fn source_const_magnitudes(rng: &mut Rng, n_const: usize, challenges: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    // A/B kill-switch (`SIMPLIPY_POLE_GRID=0`): random draws only, no grid sweep. Ablation only.
    static GRID_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let grid_on =
        *GRID_ON.get_or_init(|| std::env::var("SIMPLIPY_POLE_GRID").as_deref() != Ok("0"));
    for j in 0..(if grid_on { n_const } else { 0 }) {
        for &m in POLE_GRID.iter() {
            let mut v: Vec<f64> = (0..n_const).map(|_| rng.const_magnitude()).collect();
            v[j] = m;
            out.push(v);
        }
    }
    for _ in 0..challenges {
        out.push((0..n_const).map(|_| rng.const_magnitude()).collect());
    }
    out
}

/// DIAGNOSTIC: `SIMPLIPY_GATE_TRACE="<source tokens>"` prints every gate rejection for that
/// source (the instance's source constants, the candidate + its fitted constants, the witness
/// width). Debug only -- unset in any real mine.
fn gate_trace() -> Option<&'static String> {
    static T: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    T.get_or_init(|| std::env::var("SIMPLIPY_GATE_TRACE").ok())
        .as_ref()
}

/// A/B kill-switch for the interval domain-preservation gate (`SIMPLIPY_IVL_GATE=0` disables it).
/// Ablation/repro only -- the gate is ON by default and is a soundness requirement, not a knob.
fn ivl_gate_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_IVL_GATE").as_deref() != Ok("0"))
}

/// A/B kill-switch for the exact interval classifier in the all-constant short-circuit
/// (`SIMPLIPY_IVL_CLASS=0` disables it, leaving const-bearing sources to the candidate scan).
fn ivl_class_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_IVL_CLASS").as_deref() != Ok("0"))
}

/// A/B kill-switch for the interval REACHABILITY gate (`SIMPLIPY_IVL_REACH=0` disables it).
/// Independent of `SIMPLIPY_IVL_CLASS` so the grid-retirement factorial can vary one at a time.
fn ivl_reach_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_IVL_REACH").as_deref() != Ok("0"))
}

/// A/B kill-switch for the special-point battery phase (`SIMPLIPY_SPECIAL_BATTERY=0`
/// disables it). Ablation/repro only -- like the interval gate, it is a soundness layer,
/// not a knob (see `rust/battery.rs`).
fn special_battery_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_SPECIAL_BATTERY").as_deref() != Ok("0"))
}

/// WITNESS SNAPPING (see `battery::snap_candidates`): the shipped rule's
/// exists-witness set contains the integer/half-integer snap of the fitted constants
/// whenever the snap still fits, and deployment realizes the rule AT the snap -- so the
/// gates downstream of the fit must test it too (a raw `pow(x, 2.9999999999999996)` witness is NaN on
/// x < 0 and hides the half-line extension the snapped 3.0 creates). Adoption uses the
/// fit's own accept semantics -- `allclose_extends` with the hiprec rescue on near-miss
/// rows (the raw witness itself was often accepted only through the rescue: saturation
/// rows like `log(pow(x, y))` at f64-overflowing powers) -- with ONE relaxation: an
/// infinity's SIGN never vetoes adoption. Under the contract's ONE-zero doctrine (zero
/// sign erased) `pow(0, -1) = +inf`, and the deployed `-0.0` corner rows' sign residue is
/// a known, accepted gap: `exp(neg(log(-0.0))) = +inf` vs `pow(-0.0, -1.0) = -inf` must
/// not veto the witness -1.0 (observed live: the whole `-> pow(?0, -1)` extension family
/// kept its unsnapped raw witness and slipped the domain gate).
#[allow(clippy::too_many_arguments)]
fn adopt_snapped_witness(
    cand_tape: &Tape,
    cand_tokens: &[String],
    source_tokens: &[String],
    src_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    cols: &[Vec<f64>],
    n_rows: usize,
    y_src: &[f64],
    fitted: Vec<f64>,
    rtol: f64,
    atol: f64,
) -> Vec<f64> {
    let Some(snapped) = crate::battery::snap_candidates(&fitted) else {
        return fitted;
    };
    let y_fit = cand_tape.eval_columns(cols, &snapped, n_rows);
    // one-zero relaxation: an (inf, inf) row pair passes regardless of sign -- feed the
    // downstream checks the candidate's own value there so neither vetoes the pair
    let y_adj: Vec<f64> = y_src
        .iter()
        .zip(&y_fit)
        .map(|(&a, &b)| {
            if a.is_infinite() && b.is_infinite() {
                b
            } else {
                a
            }
        })
        .collect();
    let ok = allclose_extends(&y_adj, &y_fit, rtol, atol)
        || crate::hiprec::rescue(
            &y_adj,
            &y_fit,
            source_tokens,
            src_params,
            cand_tokens,
            &snapped,
            ops,
            var_names,
            cols,
            rtol,
            atol,
        );
    if ok {
        snapped
    } else {
        fitted
    }
}

/// Evaluate both sides of one certified instance on the special battery and apply the
/// special-point row semantics (`battery::rows_consistent`). `true` when there is nothing
/// to check (no battery: variable-free source).
#[allow(clippy::too_many_arguments)]
fn battery_rows_ok(
    src_tape: &Tape,
    source_tokens: &[String],
    src_params: &[f64],
    cand_tape: &Tape,
    cand_tokens: &[String],
    cand_params: &[f64],
    battery: Option<&crate::battery::SpecialBattery>,
    ops: &Operators,
    var_names: &[String],
    rtol: f64,
    atol: f64,
) -> bool {
    let Some(sb) = battery else { return true };
    let y_s = src_tape.eval_columns(&sb.cols, src_params, sb.n_rows);
    let y_c = cand_tape.eval_columns(&sb.cols, cand_params, sb.n_rows);
    crate::battery::rows_consistent(
        &y_s,
        &y_c,
        sb,
        source_tokens,
        src_params,
        cand_tokens,
        cand_params,
        ops,
        var_names,
        rtol,
        atol,
    )
}

/// The NO-CONSTANT equivalence test: the candidate has no `<constant>`, so it is
/// a fixed function -- evaluate it once. The SOURCE may carry `n_src_const` constants, and the rule
/// must hold for ALL of them, so we resample the source's constants over the `source_const_magnitudes`
/// rounds and every sign combination and require `allclose_extends(source, candidate)` EVERY time
/// (generic equivalence: source-finite rows bind, source-nonfinite rows are extendable).
/// Rejects on the first mismatch (the guard against a coincidental single-value match).
/// PARITY: a fast-path reject is re-judged by the SAME `hiprec::rescue` the production arm
/// (`candidate_matches`) applies -- the two entry points must never diverge (inf rows are
/// must-match at f64, so without the rescue `log(exp(x)) -> x` would be rejected here while
/// certified there).
#[allow(clippy::too_many_arguments)]
fn equivalent_no_const(
    source: &Tape,
    candidate: &Tape,
    source_tokens: &[String],
    cand_tokens: &[String],
    ops: &Operators,
    var_names: &[String],
    n_src_const: usize,
    x_cols: &[Vec<f64>],
    n_rows: usize,
    challenges: usize,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    rng: &mut Rng,
) -> bool {
    let y_cand = candidate.eval_columns(x_cols, &[], n_rows);
    let combos = sign_combos(n_src_const);
    // A const-free source has exactly one distinct instance (see `source_instances`).
    let eff_challenges = if n_src_const == 0 { 1 } else { challenges };
    // Fast NECESSARY-condition gate: every evidence row needs the candidate finite on it, and
    // evidence rows are UNIQUE rows, so finite(y_cand) bounds the evidence from above. Kills
    // all-NaN/all-inf candidates (the asin(cosh(_0)) -> nan family) outright.
    if crate::eval::count_finite(&y_cand) < min_informative {
        return false;
    }
    // EVIDENCE = UNIQUE rows where SOME instance's source is finite. Unique, NOT accumulated
    // with multiplicity: 16 challenges x sign-combos over the same ~7 defined rows must count
    // as 7 evidence points, else an almost-nowhere-defined const-bearing source (e.g.
    // asin(cosh(C*_0)), finite only at 0 for every C) would clear the gate by repetition.
    let mut evidence = vec![false; n_rows];
    let mags = if n_src_const == 0 {
        vec![vec![]]
    } else {
        source_const_magnitudes(rng, n_src_const, eff_challenges)
    };
    let mut instances: Vec<Vec<f64>> = Vec::with_capacity(mags.len() * combos.len());
    for rc in &mags {
        for combo in &combos {
            let params: Vec<f64> = rc.iter().zip(combo).map(|(r, c)| r * c).collect();
            let y = source.eval_columns(x_cols, &params, n_rows);
            // GENERIC EQUIVALENCE: source-finite rows must match; source-nonfinite
            // rows may be domain-EXTENDED by the candidate (x/x -> 1). a = source, b = candidate.
            // Inf rows are must-match at f64; the near-miss rescue then re-judges the
            // failing rows by escalation (saturation inf resolves finite, a true pole stays).
            if !allclose_extends(&y, &y_cand, rtol, atol)
                && !crate::hiprec::rescue(
                    &y,
                    &y_cand,
                    source_tokens,
                    &params,
                    cand_tokens,
                    &[],
                    ops,
                    var_names,
                    x_cols,
                    rtol,
                    atol,
                )
            {
                return false;
            }
            for (e, v) in evidence.iter_mut().zip(&y) {
                *e |= v.is_finite();
            }
            instances.push(params);
        }
    }
    // EVIDENCE GATE: enough distinct defined points must back the certification, else an
    // (almost-)nowhere-defined source would be rewritten from its corner rows alone.
    if evidence.iter().filter(|&&e| e).count() < min_informative {
        return false;
    }
    // SPECIAL-POINT PHASE (see `rust/battery.rs`). PARITY: identical to the
    // const-free arm of `candidate_matches` -- the two entry points must never diverge.
    if special_battery_on() {
        let used = crate::battery::used_variables(source_tokens, var_names);
        let battery = crate::battery::SpecialBattery::build(var_names.len(), &used);
        for params in &instances {
            if !battery_rows_ok(
                source,
                source_tokens,
                params,
                candidate,
                cand_tokens,
                &[],
                battery.as_ref(),
                ops,
                var_names,
                rtol,
                atol,
            ) {
                return false;
            }
        }
        // the special source-constant sweep (skip semantics: an instance where the
        // source is nowhere finite binds nothing; a binding instance is judged at the
        // CONTRACT points only -- the sweep never binds generic X rows)
        if n_src_const == 1 {
            for &s in crate::battery::SPECIAL_CONSTS.iter() {
                let sp = vec![s];
                let y_s = source.eval_columns(x_cols, &sp, n_rows);
                if crate::eval::count_finite(&y_s) == 0 {
                    continue;
                }
                if !battery_rows_ok(
                    source,
                    source_tokens,
                    &sp,
                    candidate,
                    cand_tokens,
                    &[],
                    battery.as_ref(),
                    ops,
                    var_names,
                    rtol,
                    atol,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// FFI-facing wrapper: compile both expressions and run the no-constant equivalence test. The
/// candidate must be constant-free (its `<constant>` count is 0); the source's constant count drives
/// the resampling. `var_names` columns come from row-major `x_flat`.
#[allow(clippy::too_many_arguments)]
pub fn equivalent_no_const_check(
    ops: &Operators,
    source: &[String],
    candidate: &[String],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    challenges: usize,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    seed: u64,
) -> Result<bool, String> {
    let src_tape = Tape::compile(source, ops, var_names)?;
    let cand_tape = Tape::compile(candidate, ops, var_names)?;
    if cand_tape.n_params != 0 {
        return Err("candidate has <constant> tokens; use exist_constants_fit for constant-bearing candidates".to_string());
    }
    let n_vars = var_names.len();
    if x_flat.len() != n_rows * n_vars {
        return Err("x_flat shape mismatch".to_string());
    }
    let cols = columns_from_row_major(x_flat, n_rows, n_vars);
    let mut rng = Rng::new(seed);
    Ok(equivalent_no_const(
        &src_tape,
        &cand_tape,
        source,
        candidate,
        ops,
        var_names,
        src_tape.n_params,
        &cols,
        n_rows,
        challenges,
        rtol,
        atol,
        min_informative,
        &mut rng,
    ))
}

/// Winner selection: among matched candidates prefer the FEWEST `<constant>`s
/// (stable -> discovery-order tiebreak), skip any that violate wildcard multiplicity, and if the
/// chosen target is bare `<constant>` while the source is all-numeric, fold to the literal value.
#[allow(dead_code)] // rlib surface; exercised by tests
pub fn select_best(
    source: &[String],
    mut matches: Vec<Vec<String>>,
    ops: &Operators,
) -> Option<Vec<String>> {
    if matches.is_empty() {
        return None;
    }
    // stable sort by <constant> count (Vec::sort_by_key is stable, matching Python's sorted()).
    matches.sort_by_key(|c| c.iter().filter(|t| t.as_str() == "<constant>").count());
    for cand in matches {
        if violates_wildcard_multiplicity(source, &cand) {
            continue;
        }
        if cand.len() == 1 && cand[0] == "<constant>" {
            let leaves: Vec<&String> = source.iter().filter(|t| !ops.is_operator(t)).collect();
            if !leaves.is_empty() && leaves.iter().all(|t| crate::utils::is_numeric_string(t)) {
                if let Some(tok) = crate::numeric::evaluate_constant_subtree(source, ops) {
                    return Some(vec![tok]);
                }
            }
        }
        return Some(cand);
    }
    None
}

/// A precompiled candidate (an entry of [`CandidateLibrary`]).
struct CandEntry {
    tokens: Vec<String>,
    var_mask: u32,
    n_const: usize,
    linearity: crate::fit::Linearity,
    tape: Tape,
    y_const_free: Option<Vec<f64>>,
    /// count_finite(y_const_free) precomputed at library build (0 for const-bearing candidates):
    /// the informativeness gate rejects vacuous (all-NaN/inf) candidates before any comparison.
    finite_y: usize,
    /// Precomputed log-linear plan (form + the const-free `g` tape) for nonlinear
    /// `pow(C,g)` / `pow(g,C)` candidates -> closed-form fit instead of the LM.
    loglin: Option<(crate::fit::LogLinForm, Tape)>,
    /// FNV-1a over the tokens: the ORDER-INDEPENDENT ingredient of the per-(candidate, instance)
    /// fit seed (see `candidate_matches`), so a candidate's LM restart stream depends only on the
    /// source seed and the candidate itself -- never on which OTHER candidates are in the library
    /// or where in the scan it sits. This is what makes the fold-filter parity gate exact:
    /// a filtered and an unfiltered mine draw IDENTICAL fit randomness for every shared candidate.
    hash: u64,
}

/// FNV-1a over a token slice, with a separator byte so ["ab","c"] != ["a","bc"].
fn token_hash(tokens: &[String]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for t in tokens {
        for b in t.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Weyl-sequence increment (golden-ratio) used to mix the instance index into the fit seed.
const SEED_GOLDEN: u64 = 0x9E3779B97F4A7C15;

/// Bitmask over `var_names` of the variables appearing in `tokens` (<=32 vars).
fn var_mask(tokens: &[String], var_names: &[String]) -> u32 {
    let mut m = 0u32;
    for (i, v) in var_names.iter().enumerate() {
        if i < 32 && tokens.iter().any(|t| t == v) {
            m |= 1 << i;
        }
    }
    m
}

/// Test one candidate against a source across the challenge/sign-combo loop. Const-free candidates
/// use the `allclose` check (against the precomputed `y_const_free`); constant-bearing candidates
/// use the fit (`exist_constants_fit_prepared`, `retries` restarts) per combo. Rejects on first
/// failure.
#[allow(clippy::too_many_arguments)]
fn candidate_matches(
    y_src: &[(Vec<f64>, Vec<f64>)],
    cand: &CandEntry,
    source: &[String],
    src_tape: &Tape,
    battery: Option<&crate::battery::SpecialBattery>,
    ops: &Operators,
    var_names: &[String],
    cols: &[Vec<f64>],
    n_rows: usize,
    retries: usize,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    fit_seed: u64,
) -> bool {
    // Fast NECESSARY-condition gate, const-free arm (see `equivalent_no_const`): every
    // evidence row needs the candidate finite on it, so finite_y bounds the UNIQUE evidence.
    if cand.n_const == 0 && cand.finite_y < min_informative {
        return false;
    }
    // EVIDENCE = UNIQUE rows where SOME instance's source is finite (BOTH arms; see
    // `equivalent_no_const` for why repetition across instances must not count).
    let mut evidence = vec![false; n_rows];
    // the certified (source-constants, target-witness) pairs, for the special-point phase
    let mut certified: Vec<(Vec<f64>, Option<Vec<f64>>)> = Vec::with_capacity(y_src.len());
    for (inst, (inst_params, y)) in y_src.iter().enumerate() {
        // WITNESS, not just a bool: None = reject; Some(None) = VACUOUS pass (no constants
        // determined -- every row was extendable); Some(Some(c)) = pass AT the constants `c`.
        // The domain gate needs `c` to bind the candidate's `<constant>` leaves.
        let ok: Option<Option<Vec<f64>>> = if cand.n_const == 0 {
            // GENERIC EQUIVALENCE: source-finite rows must match; source-nonfinite
            // rows may be domain-EXTENDED (x/x -> 1). a = source, b = candidate. f64 is a
            // PRE-FILTER: a near-miss reject is re-judged at high precision on exactly the
            // failing rows (`hiprec::rescue`) so that true identities
            // whose f64 round-trip loses precision -- atanh(tanh(x)) -> x -- still certify.
            let y_cand = cand.y_const_free.as_ref().unwrap();
            let hit = allclose_extends(y, y_cand, rtol, atol)
                || crate::hiprec::rescue(
                    y,
                    y_cand,
                    source,
                    inst_params,
                    &cand.tokens,
                    &[],
                    ops,
                    var_names,
                    cols,
                    rtol,
                    atol,
                );
            // A const-free candidate binds nothing, so even a vacuous match has a determined
            // (empty) witness and the gate applies exactly.
            if hit {
                Some(Some(Vec::new()))
            } else {
                None
            }
        } else {
            // ORDER-INDEPENDENT fit seed: a pure function of (per-source seed,
            // candidate tokens, instance index), NOT a draw from the scan-order RNG stream --
            // dropping or reordering OTHER candidates (e.g. the fold-filter) must not change
            // this candidate's LM restarts, or the filtered-vs-unfiltered parity gate would
            // drown in seed-shift noise. `Rng::new(..).next_u64()` is one splitmix64 step,
            // i.e. a full 64-bit mix of the combined key.
            let s = Rng::new(
                fit_seed
                    .wrapping_add(cand.hash)
                    .wrapping_add((inst as u64).wrapping_mul(SEED_GOLDEN)),
            )
            .next_u64();
            match crate::fit::exist_constants_fit_prepared(
                &cand.tape,
                cand.linearity,
                cols,
                n_rows,
                y,
                rtol,
                atol,
                retries,
                s,
                cand.loglin.as_ref().map(|(f, t)| (*f, t)),
            ) {
                crate::fit::FitVerdict::Pass(c) => Some(c),
                crate::fit::FitVerdict::Fail => None,
                // FIT-ARM ESCALATION: a true constant-bearing rule whose SOURCE
                // f64 evaluation is corrupted on a few rows (atanh(tanh(C*x)) -> C*x) fails
                // the fit with CONTAMINATED constants (the corrupted rows pull the least
                // squares), so first recover the majority fit by robust median trimming
                // (`fit::refit_robust`), then re-judge the rows that still fail -- at P bits,
                // at the refit constants (a legitimate "constants exist" witness).
                // `hiprec::rescue` applies its own near-miss cost gate.
                crate::fit::FitVerdict::NearMiss(c0) => {
                    match crate::fit::refit_robust(
                        &cand.tape,
                        cand.linearity,
                        cols,
                        n_rows,
                        y,
                        &c0,
                        rtol,
                        atol,
                    ) {
                        None => None, // no small-outlier consensus: keep the f64 reject
                        Some(c1) => {
                            let y_fit = cand.tape.eval_columns(cols, &c1, n_rows);
                            if crate::hiprec::rescue(
                                y,
                                &y_fit,
                                source,
                                inst_params,
                                &cand.tokens,
                                &c1,
                                ops,
                                var_names,
                                cols,
                                rtol,
                                atol,
                            ) {
                                Some(Some(c1))
                            } else {
                                None
                            }
                        }
                    }
                }
            }
        };
        let Some(fitted) = ok else { return false };
        // witness snapping before the domain gate (see `adopt_snapped_witness`)
        let fitted: Option<Vec<f64>> = fitted.map(|c| {
            if c.is_empty() {
                c
            } else {
                adopt_snapped_witness(
                    &cand.tape,
                    &cand.tokens,
                    source,
                    inst_params,
                    ops,
                    var_names,
                    cols,
                    n_rows,
                    y,
                    c,
                    rtol,
                    atol,
                )
            }
        });
        // DOMAIN-PRESERVATION gate (exact, interval-analytic), applied PER INSTANCE at the
        // witness the fit chose: a rule may complete a MEASURE-ZERO hole (`x/x -> 1` at 0, a
        // removable singularity) but must never define a value across a POSITIVE-MEASURE region
        // where the source is NaN -- that is a DOMAIN BOUNDARY (`exp(log x) -> x` off x>0) and
        // the rewrite would INVENT a function the source never was. The numeric arms cannot see
        // this: `allclose_extends` skips every NaN row regardless of measure.
        // A VACUOUS instance (`None`) determined no constants, so it pins no target function to
        // test -- and it contributes no evidence either, so `min_informative` still decides.
        if let Some(cand_params) = &fitted {
            // ONE horizon per gate decision, shared by the extension witness and BOTH deadness
            // measures. Computing them separately lets a source read "dead" off a smaller box
            // than the harm it exempts (its own constants can be smaller than the pair's), a
            // fail-open in the anti-conservative direction. `gate_horizon` counts a horizon
            // miss ONCE per decision.
            //
            // FAIL-CLOSED: every undecided path REJECTS. An undecidable horizon (interesting
            // region past R_MAX) and an exhausted node budget with no witness (`None` from
            // `domain_extension_p_at`) must never read as "no extension" = accept. The
            // counters (`interval_horizon_misses`, `interval_node_budget_misses`) count
            // fail-closed REJECTIONS; a large per-stratum delta in a mine is the flag to
            // raise `SIMPLIPY_IVL_NODE_BUDGET` / R_MAX for that stratum and re-run it.
            if ivl_gate_on() {
                let (gh_r, gh_d, gh_decidable) = crate::interval::gate_horizon(
                    source,
                    inst_params,
                    &cand.tokens,
                    cand_params,
                    ops,
                );
                let verdict = if gh_decidable {
                    crate::interval::domain_extension_p_at(
                        source,
                        inst_params,
                        &cand.tokens,
                        cand_params,
                        ops,
                        gh_r,
                        gh_d,
                    )
                } else {
                    None
                };
                let reject = match verdict {
                    // Undecided (horizon or budget): cannot certify "no extension" -- reject.
                    None => true,
                    // A proven positive-measure extension fires the gate ...
                    // ... unless the rule keeps a DEAD source dead. ONE clause, stated once:
                    //
                    //   fire  <=>  extension  AND NOT (source dead AND target dead)
                    //
                    // where "dead" = defined on no positive-measure set at these constants. A
                    // dead source has no generic graph to preserve, and an existential witness
                    // that is ITSELF dead buys the rule no capacity the source lacked --
                    // `sqrt(C^x)` at C<0 is NaN a.e. (finite only on the even integers), and the
                    // fit's `C' = -sqrt(2)` is equally NaN a.e., so nothing was invented. But
                    // resurrecting a dead source into a LIVE target injects fittable capacity
                    // from nothing, and that is the artifact.
                    //
                    // This REPLACES two ad-hoc carve-outs. The first was
                    // `cand_params.is_empty()`, justified as "a const-free candidate has no
                    // witness to be arbitrary" and as what "keeps the f64 SATURATION artifacts
                    // out". It does not: the verdict it produces depends on the target's
                    // SPELLING, not its mathematics. `acosh(|tanh x|) -> 0` was rejected while
                    // `acosh(|tanh x|) -> <constant>` was ACCEPTED at the identical witness
                    // width, and `select_best` prefers fewest constants -- so the miner reached
                    // for the rejected form, was refused, and mined the surviving one instead.
                    // Deadness is a property of the FUNCTION at its constants, so it cannot be
                    // dodged by respelling: a fitted `<constant>` binds to a finite real and is
                    // therefore LIVE, exactly like the literal `0` it was standing in for.
                    //
                    // Deadness must be PROVEN (`Some(0.0)` = a COMPLETED search found no
                    // support). An unresolved deadness search (`None`) is treated as live:
                    // letting it read "dead" widened the accept exemption, the
                    // anti-conservative direction.
                    Some(w) if w > 0.0 => {
                        !(crate::interval::defined_measure_p_at(
                            source,
                            inst_params,
                            ops,
                            gh_r,
                            gh_d,
                        ) == Some(0.0)
                            && crate::interval::defined_measure_p_at(
                                &cand.tokens,
                                cand_params,
                                ops,
                                gh_r,
                                gh_d,
                            ) == Some(0.0))
                    }
                    // Completed search, no extension: the gate passes.
                    Some(_) => false,
                };
                if reject {
                    if let Some(t) = gate_trace() {
                        if source.join(" ").starts_with(t.as_str()) {
                            eprintln!(
                                "GATE-REJECT src=[{}] src_c={:?} cand=[{}] cand_c={:?} width={:?} src_measure={:?}",
                                source.join(" "), inst_params, cand.tokens.join(" "), cand_params,
                                verdict,
                                crate::interval::defined_measure_p_at(source, inst_params, ops, gh_r, gh_d));
                        }
                    }
                    return false;
                }
            }
        }
        for (e, v) in evidence.iter_mut().zip(y) {
            *e |= v.is_finite();
        }
        certified.push((inst_params.clone(), fitted));
    }
    if evidence.iter().filter(|&&e| e).count() < min_informative {
        return false;
    }
    // SPECIAL-POINT PHASE (see `rust/battery.rs`): runs only for candidates
    // that certified everything above, so its cost is per accepted (source, candidate) pair.
    if special_battery_on() {
        // 1. Contract points at every certified instance's witness. A vacuous instance pins
        //    no target function and is skipped, exactly like the domain gate above.
        for (inst_params, fitted) in &certified {
            let Some(cand_params) = fitted else { continue };
            if !battery_rows_ok(
                src_tape,
                source,
                inst_params,
                &cand.tape,
                &cand.tokens,
                cand_params,
                battery,
                ops,
                var_names,
                rtol,
                atol,
            ) {
                return false;
            }
        }
        // 2. The special source-constant battery: a pattern-bound source constant reaches
        //    0, pi/2, ... in deployment. Skip semantics: an instance with a nowhere-finite
        //    source, or (const-bearing arm) no fittable witness, binds nothing; a binding
        //    instance must agree at the contract points. Scoped to single-constant sources
        //    -- multi-constant shapes have no ratified special-constant semantics beyond
        //    one constant.
        let n_src_const = y_src.first().map(|(p, _)| p.len()).unwrap_or(0);
        if n_src_const == 1 {
            for (k, &s) in crate::battery::SPECIAL_CONSTS.iter().enumerate() {
                let sp = vec![s];
                let y_s = src_tape.eval_columns(cols, &sp, n_rows);
                if crate::eval::count_finite(&y_s) == 0 {
                    continue;
                }
                let cp: Option<Vec<f64>> = if cand.n_const == 0 {
                    // const-free target: judged at the contract points below (the
                    // source-constant sweep never binds generic X rows)
                    Some(Vec::new())
                } else {
                    // existence is only generically binding: an unfittable special instance
                    // is SKIPPED, a fitted one binds below
                    let sseed = Rng::new(
                        fit_seed ^ cand.hash ^ (0xC0DEu64 + k as u64).wrapping_mul(SEED_GOLDEN),
                    )
                    .next_u64();
                    match crate::fit::exist_constants_fit_prepared(
                        &cand.tape,
                        cand.linearity,
                        cols,
                        n_rows,
                        &y_s,
                        rtol,
                        atol,
                        retries,
                        sseed,
                        cand.loglin.as_ref().map(|(f, t)| (*f, t)),
                    ) {
                        crate::fit::FitVerdict::Pass(Some(c)) => Some(adopt_snapped_witness(
                            &cand.tape,
                            &cand.tokens,
                            source,
                            &sp,
                            ops,
                            var_names,
                            cols,
                            n_rows,
                            &y_s,
                            c,
                            rtol,
                            atol,
                        )),
                        _ => None,
                    }
                };
                let Some(cp) = cp else { continue };
                if !battery_rows_ok(
                    src_tape,
                    source,
                    &sp,
                    &cand.tape,
                    &cand.tokens,
                    &cp,
                    battery,
                    ops,
                    var_names,
                    rtol,
                    atol,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// The source-side challenge instances, evaluated ONCE PER SOURCE and shared across every
/// candidate (re-evaluating inside `candidate_matches` would cost per (candidate, challenge,
/// sign-combo)). A const-free source has exactly one distinct instance, so its challenge count
/// collapses to 1 (identical targets add no evidence and only multiply fit flakiness).
fn source_instances(
    src_tape: &Tape,
    n_src_const: usize,
    combos: &[Vec<f64>],
    cols: &[Vec<f64>],
    n_rows: usize,
    challenges: usize,
    rng: &mut Rng,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let eff_challenges = if n_src_const == 0 { 1 } else { challenges };
    let mags = if n_src_const == 0 {
        vec![vec![]]
    } else {
        source_const_magnitudes(rng, n_src_const, eff_challenges)
    };
    let mut out = Vec::with_capacity(mags.len() * combos.len());
    for rc in &mags {
        for combo in combos {
            let p: Vec<f64> = rc.iter().zip(combo).map(|(r, c)| r * c).collect();
            let y = src_tape.eval_columns(cols, &p, n_rows);
            // params are RETAINED alongside the values so the hiprec rescue can
            // re-evaluate the exact same instance at high precision.
            out.push((p, y));
        }
    }
    out
}

/// A RESIDENT candidate library: every candidate precompiled to a tape and indexed by length,
/// with each const-free candidate's `y` evaluated ONCE (not per source). Built once per mine;
/// reused by `find_rule_with_lib` for every source -- this removes the per-source rebuild that
/// the per-call `find_rule` path pays.
pub struct CandidateLibrary {
    var_names: Vec<String>,
    cols: Vec<Vec<f64>>,
    n_rows: usize,
    by_len: Vec<Vec<CandEntry>>, // index = candidate length
    /// candidates kept in the library (post-filter)
    n_candidates: usize,
    /// variable-free candidates dropped by the fold-filter (0 when the filter is off/inert)
    n_filtered: usize,
}

impl CandidateLibrary {
    /// Row count of the library's shared X (for resolving informativeness-gate defaults).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Candidates kept in the library (post-filter).
    pub fn n_candidates(&self) -> usize {
        self.n_candidates
    }

    /// Variable-free candidates dropped by the fold-filter.
    pub fn n_filtered(&self) -> usize {
        self.n_filtered
    }

    /// Build the resident library. With `fold_filter`, VARIABLE-FREE candidates of length >= 2
    /// are dropped, PROVIDED the bare
    /// `["<constant>"]` candidate is present (otherwise the filter is inert -- conservative).
    ///
    /// SOUNDNESS (the narrow, provable form of "candidate minimization"): a variable-free
    /// candidate evaluates to ONE scalar per constant-assignment (a constant function of X), so
    /// - const-bearing var-free (`sin(<constant>)`, `pow(<constant>,<constant>)`, ...): any
    ///   source instance it matches is (near-)constant-valued, which the length-1 `<constant>`
    ///   candidate also matches -- with a fit family (all of R) that SUPERSET-dominates the
    ///   wrapper's reachable set;
    /// - const-free var-free (`exp(1)`, `neg(np.pi)`, ...): a fixed value; any match implies the
    ///   per-instance `<constant>` fit matches too (and its all-nonfinite members never match:
    ///   the finite-evidence gate rejects them).
    /// Since `find_rule_with_lib` scans lengths SHORTEST-FIRST and returns at the first matching
    /// length, the length-1 `<constant>` match preempts every length>=2 var-free candidate, which
    /// is therefore never selectable and can be dropped without changing any mined rule. The one
    /// theoretical edge is tolerance-band width: `<constant>`'s solved fit sits within <= 2x the
    /// per-row tolerance of the wrapper's (mean vs member), relevant only for sources within
    /// ~rtol of the band edge -- real mine values agree to ~1e-16 vs rtol 1e-9, and the
    /// filtered-vs-unfiltered mine parity test certifies the dominance empirically.
    pub fn build(
        ops: &Operators,
        candidates: &[Vec<String>],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        fold_filter: bool,
    ) -> Result<Self, String> {
        let n_vars = var_names.len();
        if x_flat.len() != n_rows * n_vars {
            return Err("x_flat shape mismatch".to_string());
        }
        let cols = columns_from_row_major(x_flat, n_rows, n_vars);
        let max_len = candidates.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut by_len: Vec<Vec<CandEntry>> = (0..=max_len).map(|_| Vec::new()).collect();
        let filter_active = fold_filter
            && candidates
                .iter()
                .any(|c| c.len() == 1 && c[0] == "<constant>");
        let mut n_candidates = 0usize;
        let mut n_filtered = 0usize;
        for c in candidates {
            let len = c.len();
            if len == 0 {
                continue;
            }
            let vm = var_mask(c, var_names);
            // Filter decision NOT via `vm == 0`: the scan mask truncates at 32 variables, so a
            // candidate whose only variables have index >= 32 would be misclassified as
            // var-free and wrongly dropped. The filter checks ALL var_names.
            let has_var = c.iter().any(|t| var_names.iter().any(|v| v == t));
            if filter_active && len >= 2 && !has_var {
                n_filtered += 1;
                continue;
            }
            let tape = Tape::compile(c, ops, var_names)?;
            let n_const = tape.n_params;
            let y_const_free = if n_const == 0 {
                Some(tape.eval_columns(&cols, &[], n_rows))
            } else {
                None
            };
            let linearity = crate::fit::classify(c, ops)?;
            let loglin = if linearity == crate::fit::Linearity::Nonlinear {
                match crate::fit::detect_log_linear(c, ops) {
                    Some((form, g)) => Some((form, Tape::compile(&g, ops, var_names)?)),
                    None => None,
                }
            } else {
                None
            };
            let finite_y = y_const_free
                .as_ref()
                .map(|y| crate::eval::count_finite(y))
                .unwrap_or(0);
            by_len[len].push(CandEntry {
                tokens: c.clone(),
                var_mask: vm,
                n_const,
                linearity,
                tape,
                y_const_free,
                finite_y,
                loglin,
                hash: token_hash(c),
            });
            n_candidates += 1;
        }
        Ok(CandidateLibrary {
            var_names: var_names.to_vec(),
            cols,
            n_rows,
            by_len,
            n_candidates,
            n_filtered,
        })
    }
}

/// The full native `find_rule_worker` decision over a RESIDENT library:
/// guard -> all-numeric short-circuit -> scan candidates shortest-first (variable-subset filtered),
/// dispatch const-free -> the no-constant test / const-bearing -> the constant fit, break on the
/// first matching length, `select_best`.
#[allow(clippy::too_many_arguments)]
pub fn find_rule_with_lib(
    ops: &Operators,
    source: &[String],
    simplified_length: usize,
    max_target: Option<usize>,
    lib: &CandidateLibrary,
    challenges: usize,
    retries: usize,
    seed: u64,
    rtol: f64,
    atol: f64,
    min_informative: usize,
) -> Result<Option<Vec<String>>, String> {
    // GUARD FIRST (BEFORE the all-constant short-circuit).
    let max_cand_len = match max_target {
        Some(mt) => simplified_length.min(mt + 1),
        None => simplified_length,
    };
    if max_cand_len <= 1 {
        return Ok(None);
    }
    // all-constant short-circuit, reached only after the guard passes.
    // A leaf counts as constant when `leaf_value` resolves it to a FINITE value: numeric
    // literals AND the special constants (`np.pi`, `np.e`, `(-1)`). The previous
    // `is_numeric_string`-only gate sent e.g. `/ (-1) <constant>` to the library scan
    // (inf/nan literals stay excluded -- non-finite algebra is the explicit rules' domain,
    // e.g. `+ float("-inf") <constant> -> float("-inf")`, not absorption).
    let finite_leaf = |t: &str| crate::numeric::leaf_value(t).is_some_and(f64::is_finite);
    if source.len() > 1
        && source
            .iter()
            .all(|t| t == "<constant>" || ops.is_operator(t) || finite_leaf(t))
    {
        let non_ops: Vec<&String> = source.iter().filter(|t| !ops.is_operator(t)).collect();
        if !non_ops.is_empty() && non_ops.iter().all(|t| finite_leaf(t)) {
            if let Some(tok) = crate::numeric::evaluate_constant_subtree(source, ops) {
                return Ok(Some(vec![tok]));
            }
        }
        // A variable-free <constant>-bearing source collapses to a single leaf ONLY when that
        // leaf reproduces it a.e. Classified EXACTLY by interval analysis (`rust/interval.rs`),
        // not by sampling: a POLE -- a finite region together with an inf region, e.g.
        // `pow(0, C)` = 0 for C>0 and +inf for C<0, or `* <constant> inv 0` = C*inf -- is NOT a
        // constant, so returning `<constant>` there is an artifact. Finite -> `<constant>` (nan
        // rows extend); all-NaN -> nan (`* <constant> acos(np.e)`); a.e. +-inf -> that inf literal
        // (`inv(pow(0, exp C))` = +inf everywhere); a mixed pole -> fall through to the candidate
        // scan. The sampled classifier this replaces could only ever be as good as its grid.
        match if ivl_class_on() {
            crate::interval::value_class(source, ops)
        } else {
            None
        } {
            Some(crate::interval::Class::Finite) => {
                return Ok(Some(vec!["<constant>".to_string()]))
            }
            Some(crate::interval::Class::Nan) => {
                return Ok(Some(vec!["float(\"nan\")".to_string()]))
            }
            Some(crate::interval::Class::PosInf) => {
                return Ok(Some(vec!["float(\"inf\")".to_string()]))
            }
            Some(crate::interval::Class::NegInf) => {
                return Ok(Some(vec!["float(\"-inf\")".to_string()]))
            }
            _ => {} // Mixed / Empty / unevaluable -> not a clean collapse: candidate scan below
        }
    }
    let src_tape = Tape::compile(source, ops, &lib.var_names)?;
    let n_src_const = src_tape.n_params;
    let src_mask = var_mask(source, &lib.var_names);
    let combos = sign_combos(n_src_const);

    let mut rng = Rng::new(seed);
    // Source instances shared across the whole candidate scan (see `source_instances`).
    let y_src = source_instances(
        &src_tape,
        n_src_const,
        &combos,
        &lib.cols,
        lib.n_rows,
        challenges,
        &mut rng,
    );
    // REACHABILITY gate (exact, interval-analytic) for a VARIABLE-FREE source: every value
    // component the source takes over its constants must be reachable by the candidate over its
    // own (see `interval::reaches_all_of`). Scoped to variable-free sources -- the class the
    // interval engine is validated on, and exactly the family the sampled POLE_GRID exists for
    // (`pow(asin(C/2), inf) -> 0`, whose +inf band a fixed grid can straddle and miss).
    // Computed ONCE per source, outside the scan.
    let src_reach = if ivl_reach_on() && !source.iter().any(|t| lib.var_names.contains(t)) {
        crate::interval::value_set(source, ops, &crate::interval::Vs::reals())
    } else {
        None
    };
    // Special-point battery over THIS source's variable set (None for a variable-free
    // source), shared across the whole candidate scan; see `rust/battery.rs`.
    let battery = crate::battery::SpecialBattery::build(
        lib.var_names.len(),
        &crate::battery::used_variables(source, &lib.var_names),
    );
    let scan_max = max_cand_len.min(lib.by_len.len());
    for length in 1..scan_max {
        let mut matches: Vec<Vec<String>> = Vec::new();
        for cand in &lib.by_len[length] {
            if cand.var_mask & !src_mask != 0 {
                continue; // candidate uses a variable the source lacks
            }
            if let Some(sv) = &src_reach {
                if let Some(cv) =
                    crate::interval::value_set(&cand.tokens, ops, &crate::interval::Vs::reals())
                {
                    if !crate::interval::reaches_all_of(&cv, sv) {
                        continue; // cannot reach a component the source takes: FALSE, no sampling
                    }
                }
            }
            if candidate_matches(
                &y_src,
                cand,
                source,
                &src_tape,
                battery.as_ref(),
                ops,
                &lib.var_names,
                &lib.cols,
                lib.n_rows,
                retries,
                rtol,
                atol,
                min_informative,
                seed,
            ) {
                // (the domain gate runs INSIDE `candidate_matches`, per instance, where the
                // source's drawn constants and the fit's chosen constants are both in scope)
                matches.push(cand.tokens.clone());
            }
        }
        if !matches.is_empty() {
            if let Some(t) = gate_trace() {
                if source.join(" ").starts_with(t.as_str()) {
                    eprintln!(
                        "TRACE src=[{}] simplified_len={} cand_len={} matches={:?}",
                        source.join(" "),
                        simplified_length,
                        length,
                        matches.iter().map(|m| m.join(" ")).collect::<Vec<_>>()
                    );
                }
            }
            return Ok(select_best(source, matches, ops)); // shortest matching length wins
        }
    }
    if let Some(t) = gate_trace() {
        if source.join(" ").starts_with(t.as_str()) {
            eprintln!(
                "TRACE src=[{}] simplified_len={} -> NO MATCH",
                source.join(" "),
                simplified_length
            );
        }
    }
    Ok(None)
}

/// Per-call convenience: build a `CandidateLibrary` then delegate. Kept for the per-call FFI /
/// tests; the mine uses the resident-library path.
#[allow(clippy::too_many_arguments)]
pub fn find_rule(
    ops: &Operators,
    source: &[String],
    simplified_length: usize,
    max_target: Option<usize>,
    candidates: &[Vec<String>],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    challenges: usize,
    retries: usize,
    seed: u64,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    fold_filter: bool,
) -> Result<Option<Vec<String>>, String> {
    let lib = CandidateLibrary::build(ops, candidates, var_names, x_flat, n_rows, fold_filter)?;
    find_rule_with_lib(
        ops,
        source,
        simplified_length,
        max_target,
        &lib,
        challenges,
        retries,
        seed,
        rtol,
        atol,
        min_informative,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn wildcard_multiplicity() {
        // inert on dummy vars (no _j tokens)
        assert!(!violates_wildcard_multiplicity(
            &s(&["+", "x0", "x0"]),
            &s(&["x0"])
        ));
        // _0 twice on rhs, once on lhs -> violates
        assert!(violates_wildcard_multiplicity(
            &s(&["_0"]),
            &s(&["+", "_0", "_0"])
        ));
        // _0 once each -> ok
        assert!(!violates_wildcard_multiplicity(
            &s(&["+", "_0", "_1"]),
            &s(&["_0"])
        ));
    }

    #[test]
    fn no_const_equiv() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let n = 64usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            xf.push((r as f64) * 0.1 - 3.0);
            xf.push((r as f64) * -0.05 + 1.5);
        }
        // source: neg(neg(x0)) == candidate x0  (const-free both) -> equivalent
        assert!(e
            .equivalent_no_const_check(
                &s(&["neg", "neg", "x0"]),
                &s(&["x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-5,
                1e-8,
                8,
                0
            )
            .unwrap());
        // source: + <constant> x0  vs candidate x0 : holds ONLY for constant 0 -> resampling rejects
        assert!(!e
            .equivalent_no_const_check(
                &s(&["+", "<constant>", "x0"]),
                &s(&["x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-5,
                1e-8,
                8,
                0
            )
            .unwrap());
        // source: * <constant> x0 vs candidate x0 : holds only for constant 1 -> rejected
        assert!(!e
            .equivalent_no_const_check(
                &s(&["*", "<constant>", "x0"]),
                &s(&["x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-5,
                1e-8,
                8,
                0
            )
            .unwrap());
    }

    #[test]
    fn find_rule_basic() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let n = 64usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            xf.push((r as f64) * 0.1 - 3.0);
            xf.push((r as f64) * -0.05 + 1.5);
        }
        let lib = vec![s(&["x0"]), s(&["x1"]), s(&["neg", "x0"]), s(&["abs", "x0"])];
        // neg(neg(x0)) (len 3) simplifies to x0 -> find_rule should return ["x0"]
        let r = e
            .find_rule(
                &s(&["neg", "neg", "x0"]),
                3,
                Some(2),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-5,
                1e-8,
                8,
                true,
            )
            .unwrap();
        assert_eq!(r, Some(s(&["x0"])));
        // sin(x0) (len 2) has no shorter equivalent in the library -> None
        let r2 = e
            .find_rule(
                &s(&["sin", "x0"]),
                2,
                Some(2),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-5,
                1e-8,
                8,
                true,
            )
            .unwrap();
        assert_eq!(r2, None);
    }

    /// REGRESSION (the lesson that removed mine-time accept-certification):
    /// `pow(np.e, x) -> exp(x)` must be minable. np.e is an f64 LITERAL, so the identity
    /// holds to ~|x| * 1e-16 -- true at the mine tolerance. A checker judging finer than
    /// the mine tolerance vetoes this class of true rules while catching nothing.
    #[test]
    fn f64_literal_identities_stay_minable() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0"]);
        let n = 64usize;
        let xf: Vec<f64> = (0..n).map(|i| -8.0 + 0.25 * i as f64).collect();
        let lib = vec![s(&["x0"]), s(&["exp", "x0"]), s(&["neg", "cos", "x0"])];
        let r = e
            .find_rule(
                &s(&["pow", "np.e", "x0"]),
                3,
                Some(2),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-9,
                1e-12,
                8,
                false,
            )
            .unwrap();
        assert_eq!(
            r,
            Some(s(&["exp", "x0"])),
            "pow(np.e, x) -> exp(x) must be mined"
        );
        let r2 = e
            .find_rule(
                &s(&["cos", "+", "np.pi", "x0"]),
                4,
                Some(3),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-9,
                1e-12,
                8,
                false,
            )
            .unwrap();
        assert_eq!(
            r2,
            Some(s(&["neg", "cos", "x0"])),
            "cos(np.pi + x) -> -cos(x) must be mined"
        );
    }

    #[test]
    fn fold_filter_drops_var_free_and_preserves_decisions() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["x0"]);
        let n = 128usize;
        let xf: Vec<f64> = (0..n).map(|r| (r as f64) * 0.11 - 7.0).collect();
        // library incl. the bare <constant> (guard token), var-free const-bearing (sin(<c>),
        // pow2(<c>)), var-free const-free (exp(1)), and var-bearing candidates
        let cands = vec![
            s(&["x0"]),
            s(&["<constant>"]),
            s(&["sin", "<constant>"]),
            s(&["pow2", "<constant>"]),
            s(&["exp", "1"]),
            s(&["neg", "x0"]),
            s(&["*", "<constant>", "x0"]),
        ];
        let filtered = CandidateLibrary::build(ops, &cands, &vars, &xf, n, true).unwrap();
        let raw = CandidateLibrary::build(ops, &cands, &vars, &xf, n, false).unwrap();
        assert_eq!(filtered.n_filtered(), 3); // sin(<c>), pow2(<c>), exp(1)
        assert_eq!(filtered.n_candidates(), 4);
        assert_eq!(raw.n_filtered(), 0);
        assert_eq!(raw.n_candidates(), 7);
        // WITHOUT the bare <constant> the filter must be INERT (conservative guard)
        let no_bare: Vec<Vec<String>> = cands[2..].to_vec();
        let inert = CandidateLibrary::build(ops, &no_bare, &vars, &xf, n, true).unwrap();
        assert_eq!(inert.n_filtered(), 0);
        // decision parity on both a constant-valued and a non-constant source: the dominance
        // lemma says the length-1 <constant> preempts every var-free candidate, so filtered and
        // raw libraries must decide identically (incl. the ORDER-INDEPENDENT fit seeds).
        for src in [
            s(&["-", "x0", "x0"]),    // constant-valued (0) -> matches <constant> at len 1
            s(&["+", "x0", "np.pi"]), // genuinely var-dependent
            s(&["neg", "neg", "x0"]), // reduces to x0
        ] {
            let a = find_rule_with_lib(
                ops,
                &src,
                src.len(),
                Some(2),
                &filtered,
                16,
                16,
                7,
                1e-9,
                1e-12,
                16,
            )
            .unwrap();
            let b = find_rule_with_lib(
                ops,
                &src,
                src.len(),
                Some(2),
                &raw,
                16,
                16,
                7,
                1e-9,
                1e-12,
                16,
            )
            .unwrap();
            assert_eq!(a, b, "filtered vs raw decision diverged on {src:?}");
        }
    }

    #[test]
    fn fold_filter_keeps_var_dependent_candidates_beyond_32_vars() {
        // Regression: the scan var_mask truncates at 32
        // variables, so a candidate whose only variable has index >= 32 has var_mask == 0; the
        // filter must NOT treat it as var-free (it checks all var_names directly).
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars: Vec<String> = (0..33).map(|i| format!("x{i}")).collect();
        let n = 16usize;
        let xf: Vec<f64> = (0..n * 33).map(|i| (i as f64) * 0.01 - 2.0).collect();
        let cands = vec![
            s(&["<constant>"]),
            s(&["sin", "x32"]), // var-DEPENDENT, but var_mask == 0 (index >= 32)
            s(&["sin", "x0"]),
            s(&["exp", "<constant>"]), // genuinely var-free -> filtered
        ];
        let lib = CandidateLibrary::build(ops, &cands, &vars, &xf, n, true).unwrap();
        assert_eq!(lib.n_filtered(), 1, "only exp(<constant>) may be dropped");
        assert_eq!(lib.n_candidates(), 3);
    }

    #[test]
    fn select_prefers_fewest_constants() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let src = s(&["+", "*", "<constant>", "x0", "<constant>"]);
        // two matches: one with a constant, one without -> pick the const-free
        let m = vec![s(&["*", "<constant>", "x0"]), s(&["x0"])];
        assert_eq!(select_best(&src, m, ops), Some(s(&["x0"])));
    }

    /// Signed, mixed-magnitude 1-var grid: negatives probe domain extensions, positives feed
    /// the constant fit.
    fn grid_1var(n: usize) -> Vec<f64> {
        (0..n)
            .map(|r| {
                let t = (r as f64 + 0.5) / n as f64;
                let m = 10f64.powf(-2.0 + 3.0 * t); // 1e-2 .. 10
                if r % 2 == 0 {
                    m
                } else {
                    -m
                }
            })
            .collect()
    }

    /// `adopt_snapped_witness`: adopt when the snap still fits (incl. the one-zero
    /// infinity-sign relaxation), keep the raw witness when the snap is a genuinely
    /// different function.
    #[test]
    fn snapped_witness_adoption() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["x0"]);
        let toks = s(&["pow", "x0", "<constant>"]);
        let tape = Tape::compile(&toks, ops, &vars).unwrap();
        // -0.0 corner row included: exp(neg(log(-0.0))) = +inf vs pow(-0.0, -1) = -inf is
        // the known accepted zero-sign residue and must NOT veto the snap to -1.0
        let mut xs: Vec<f64> = (1..40).map(|i| 0.3 * i as f64).collect();
        xs.push(-0.0);
        let cols = vec![xs];
        let n = cols[0].len();
        let src = s(&["exp", "neg", "log", "x0"]);
        let y: Vec<f64> = cols[0]
            .iter()
            .map(|&x| (-(x.ln())).exp()) // +inf at -0.0 (log(-0.0) = -inf)
            .collect();
        let a = adopt_snapped_witness(
            &tape,
            &toks,
            &src,
            &[],
            ops,
            &vars,
            &cols,
            n,
            &y,
            vec![-0.999_999_999_997_003_7],
            1e-9,
            1e-12,
        );
        assert_eq!(a, vec![-1.0]);
        // y genuinely x^2.9999997: within snap range of 3 but pow(x, 3) misses by ~1e-7
        // rel > rtol on every row -- keep the raw witness
        let yb: Vec<f64> = cols[0].iter().map(|&x| x.powf(2.999_999_7)).collect();
        let b = adopt_snapped_witness(
            &tape,
            &toks,
            &s(&["pow", "x0", "<constant>"]),
            &[2.999_999_7],
            ops,
            &vars,
            &cols,
            n,
            &yb,
            vec![2.999_999_7],
            1e-9,
            1e-12,
        );
        assert_eq!(b, vec![2.999_999_7]);
    }

    /// The certification gaps closed by witness snapping + the special-point phase
    /// (`rust/battery.rs`); each case is a family observed live in (4,3) mining runs.
    #[test]
    fn special_phase_closes_certification_gaps() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0"]);
        let n = 512usize;
        let xf = grid_1var(n);
        let fr = |src: &[&str], cand: &[&str]| {
            let src = s(src);
            let cand = s(cand);
            let slen = src.len();
            let mt = cand.len();
            e.find_rule(
                &src,
                slen,
                Some(mt),
                &[cand],
                &vars,
                &xf,
                n,
                16,
                16,
                7,
                1e-9,
                1e-12,
                32,
                false,
            )
            .unwrap()
        };
        // 1. WITNESS SNAPPING + domain gate: exp(log(x^3)) is defined only on x > 0; the raw
        //    fit (2.9999999999999996) is NaN on x < 0 and hides the extension, the snapped
        //    3.0 is total on R -- a positive-measure domain extension (a 37-rule family in
        //    (4,3) mining runs before this phase existed).
        assert_eq!(
            fr(&["exp", "log", "pow3", "x0"], &["pow", "x0", "<constant>"]),
            None
        );
        // 2. a domain-PRESERVING constant witness stays minable: exp(log(x)/3) = x^(1/3) on
        //    x > 0, and the fitted 1/3 is far from every snap target -- pow(x, 1/3) is NaN
        //    on x < 0 exactly like the source (a certified live rule).
        assert!(fr(&["exp", "div3", "log", "x0"], &["pow", "x0", "<constant>"]).is_some());
        // 3. null-set completion stays allowed (x/x -> 1 at 0: the limit-completion doctrine).
        assert!(fr(&["/", "x0", "x0"], &["1"]).is_some());
        // 4. contract point x = pi/2: f64 sin(pi/2) is EXACTLY 1.0 and pow(1, inf) = 1, not
        //    0 -- clause (a) at a real battery point; the hiprec rescue must not overturn a
        //    contract point (re-evaluating the f64 pi/2 "more precisely" answers a different
        //    question than "what happens at pi/2").
        assert_eq!(fr(&["pow", "sin", "x0", "float(\"inf\")"], &["0"]), None);
        // 5. pow-of-cos in CONSTANT space: pow(cos c, inf) = 1 at c = 0 -- the special
        //    source-constant battery reaches what no random draw does.
        assert_eq!(
            fr(&["pow", "cos", "<constant>", "float(\"inf\")"], &["0"]),
            None
        );
        // 6. deployed-consistency at battery points: atanh(tanh(tan x)) diverges from tan x
        //    by ~3e-7 at x = +-1.5 in f64 (a live (4,3) family) ...
        assert_eq!(fr(&["atanh", "tanh", "tan", "x0"], &["tan", "x0"]), None);
        // ... while atanh(tanh(x)) -> x stays certified (the rescue's flagship identity).
        assert!(fr(&["atanh", "tanh", "x0"], &["x0"]).is_some());
    }

    /// The unresolvable-seam class: a removable
    /// singularity whose cancellation is spelled through DIFFERENT subexpressions cancels
    /// exactly in f64 but leaves a precision-roulette residue at the precision rungs.
    #[test]
    fn seam_class_rejected_stable_null_completion_kept() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let n = 256usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            let t = (r as f64 + 0.5) / n as f64;
            let m = 10f64.powf(-1.0 + 2.0 * t);
            xf.push(if r % 2 == 0 { m } else { -m });
            let u = 10f64.powf(-1.0 + 2.0 * ((r as f64 * 0.37) % 1.0));
            xf.push(if r % 3 == 0 { -u } else { u });
        }
        // (x^3 + x^2 y)/(x + y) -> x^2: nan at (pi/2, -pi/2) in f64 (both spellings of x^3
        // round identically) but residue/0 = inf at 50 decimal digits: a precision seam.
        let seam = s(&[
            "/", "+", "pow3", "x0", "*", "pow2", "x0", "x1", "+", "x0", "x1",
        ]);
        assert!(!e
            .equivalent_no_const_check(
                &seam,
                &s(&["pow2", "x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-9,
                1e-12,
                32,
                7
            )
            .unwrap());
        // (x+y)^3/(x+y)^2 -> x+y: the cancellation is spelled ONCE, cancels exactly at every
        // precision, 0/0 = nan stably -- the ratified null-set-completion class stays.
        let stable = s(&["/", "pow3", "+", "x0", "x1", "pow2", "+", "x0", "x1"]);
        assert!(e
            .equivalent_no_const_check(
                &stable,
                &s(&["+", "x0", "x1"]),
                &vars,
                &xf,
                n,
                16,
                1e-9,
                1e-12,
                32,
                7
            )
            .unwrap());
    }
}
