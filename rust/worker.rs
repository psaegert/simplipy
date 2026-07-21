//! The no-constant equivalence test + rule selection for the OFFLINE miner.
//!
//! This is the constant-FREE candidate branch of `find_rule_worker` (engine.py@0.2.15:2433-2452) plus the
//! winner-selection (engine.py@0.2.15:2489-2510). It adds NO new numerics -- the only math is
//! `allclose_extends` (bit-exact vs numpy) -- so it is a pure control-flow port. Together with the
//! constant-fit branch (`crate::fit`) these are the two halves of the per-candidate decision,
//! assembled over the candidate library + generation + Kruskal prune into the full native miner.

use crate::eval::{allclose_extends, columns_from_row_major, Tape};
use crate::fit::Rng;
use crate::operators::Operators;

/// The non-increasing wildcard-multiplicity condition (utils.py@0.2.15:938): a rule `lhs -> rhs` violates it
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

/// The NO-CONSTANT equivalence test (engine.py@0.2.15:2433-2452): the candidate has no `<constant>`, so it is
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
        }
    }
    // EVIDENCE GATE: enough distinct defined points must back the certification, else an
    // (almost-)nowhere-defined source would be rewritten from its corner rows alone.
    evidence.iter().filter(|&&e| e).count() >= min_informative
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

/// Winner selection (engine.py@0.2.15:2489-2510): among matched candidates prefer the FEWEST `<constant>`s
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
        // DOMAIN-PRESERVATION gate (exact, interval-analytic), applied PER INSTANCE at the
        // witness the fit chose: a rule may complete a MEASURE-ZERO hole (`x/x -> 1` at 0, a
        // removable singularity) but must never define a value across a POSITIVE-MEASURE region
        // where the source is NaN -- that is a DOMAIN BOUNDARY (`exp(log x) -> x` off x>0) and
        // the rewrite would INVENT a function the source never was. The numeric arms cannot see
        // this: `allclose_extends` skips every NaN row regardless of measure.
        // A VACUOUS instance (`None`) determined no constants, so it pins no target function to
        // test -- and it contributes no evidence either, so `min_informative` still decides.
        if let Some(cand_params) = fitted {
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
                    &cand_params,
                    ops,
                );
                let verdict = if gh_decidable {
                    crate::interval::domain_extension_p_at(
                        source,
                        inst_params,
                        &cand.tokens,
                        &cand_params,
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
                                &cand_params,
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
    }
    evidence.iter().filter(|&&e| e).count() >= min_informative
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

/// The full native `find_rule_worker` decision (engine.py@0.2.15:2382-2510) over a RESIDENT library:
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
    // GUARD FIRST (engine.py@0.2.15:2384, BEFORE the short-circuit at :2390).
    let max_cand_len = match max_target {
        Some(mt) => simplified_length.min(mt + 1),
        None => simplified_length,
    };
    if max_cand_len <= 1 {
        return Ok(None);
    }
    // all-constant short-circuit (engine.py@0.2.15:2390-2400), reached only after the guard passes.
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
}
