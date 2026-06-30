//! The no-constant equivalence test + rule selection for the OFFLINE miner (Phase B, Milestone 2).
//!
//! This is the constant-FREE candidate branch of `find_rule_worker` (engine.py:2433-2452) plus the
//! winner-selection (engine.py:2489-2510). It adds NO new numerics -- the only math is `allclose`
//! (already bit-exact vs numpy from M1) -- so it is a pure control-flow port. Together with M3 (the
//! constant-fit branch) these are the two halves of the per-candidate decision; M4 assembles them
//! over the candidate library + generation + Kruskal prune into the full native miner.

use crate::eval::{allclose, columns_from_row_major, Tape};
use crate::fit::Rng;
use crate::operators::Operators;

/// The non-increasing wildcard-multiplicity condition (utils.py:938): a rule `lhs -> rhs` violates it
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

/// All sign-combination vectors in `{-1, 0, 1}^n` (the `product((-1,0,1), repeat=n)` of
/// engine.py:2444). `n == 0` -> a single empty combo (so a const-free source still runs once).
fn sign_combos(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        let mut next = Vec::with_capacity(out.len() * 3);
        for base in &out {
            for s in [-1.0f64, 0.0, 1.0] {
                let mut v = base.clone();
                v.push(s);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// The NO-CONSTANT equivalence test (engine.py:2433-2452): the candidate has no `<constant>`, so it is
/// a fixed function -- evaluate it once. The SOURCE may carry `n_src_const` constants, and the rule
/// must hold for ALL of them, so we resample the source's constants over `challenges` rounds and every
/// sign combination `abs(N(0,5)) * {-1,0,1}` and require `allclose(source, candidate)` EVERY time.
/// Rejects on the first mismatch (the guard against a coincidental single-value match).
#[allow(clippy::too_many_arguments)]
fn equivalent_no_const(
    source: &Tape,
    candidate: &Tape,
    n_src_const: usize,
    x_cols: &[Vec<f64>],
    n_rows: usize,
    challenges: usize,
    rtol: f64,
    atol: f64,
    rng: &mut Rng,
) -> bool {
    let y_cand = candidate.eval_columns(x_cols, &[], n_rows);
    let combos = sign_combos(n_src_const);
    for _ in 0..challenges {
        let rc: Vec<f64> = (0..n_src_const)
            .map(|_| rng.normal(0.0, 5.0).abs())
            .collect();
        for combo in &combos {
            let params: Vec<f64> = rc.iter().zip(combo).map(|(r, c)| r * c).collect();
            let y = source.eval_columns(x_cols, &params, n_rows);
            // engine.py:2446: np.allclose(y, y_candidate) -> a = source, b = candidate.
            if !allclose(&y, &y_cand, rtol, atol) {
                return false;
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
    seed: u64,
) -> Result<bool, String> {
    let src_tape = Tape::compile(source, ops, var_names)?;
    let cand_tape = Tape::compile(candidate, ops, var_names)?;
    if cand_tape.n_params != 0 {
        return Err("candidate has <constant> tokens; use exist_constants_fit (M3)".to_string());
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
        src_tape.n_params,
        &cols,
        n_rows,
        challenges,
        rtol,
        atol,
        &mut rng,
    ))
}

/// Winner selection (engine.py:2489-2510): among matched candidates prefer the FEWEST `<constant>`s
/// (stable -> discovery-order tiebreak), skip any that violate wildcard multiplicity, and if the
/// chosen target is bare `<constant>` while the source is all-numeric, fold to the literal value.
#[allow(dead_code)] // the M4 `find_rule` assembly entry point; exercised by tests today
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

/// A precompiled candidate. Built per `find_rule` call here; M4b will make a `CandidateLibrary`
/// resident across the whole mine (and precompute const-free `y` once instead of per source).
struct CandEntry {
    tokens: Vec<String>,
    var_mask: u32,
    n_const: usize,
    linearity: crate::fit::Linearity,
    tape: Tape,
    y_const_free: Option<Vec<f64>>,
}

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

/// Test one candidate against a source across the challenge/sign-combo loop. Const-free candidates use
/// the M2 `allclose` check (against the precomputed `y_const_free`); constant-bearing candidates use
/// the M3 fit (`exist_constants_fit_prepared`, `retries` restarts) per combo. Rejects on first failure.
#[allow(clippy::too_many_arguments)]
fn candidate_matches(
    src_tape: &Tape,
    n_src_const: usize,
    combos: &[Vec<f64>],
    cand: &CandEntry,
    cols: &[Vec<f64>],
    n_rows: usize,
    challenges: usize,
    retries: usize,
    rtol: f64,
    atol: f64,
    rng: &mut Rng,
) -> bool {
    for _ in 0..challenges {
        let rc: Vec<f64> = (0..n_src_const)
            .map(|_| rng.normal(0.0, 5.0).abs())
            .collect();
        for combo in combos {
            let p: Vec<f64> = rc.iter().zip(combo).map(|(r, c)| r * c).collect();
            let y = src_tape.eval_columns(cols, &p, n_rows);
            let ok = if cand.n_const == 0 {
                // engine.py:2446: allclose(source, candidate)
                allclose(&y, cand.y_const_free.as_ref().unwrap(), rtol, atol)
            } else {
                let s = rng.next_u64();
                crate::fit::exist_constants_fit_prepared(
                    &cand.tape,
                    cand.linearity,
                    cols,
                    n_rows,
                    &y,
                    rtol,
                    atol,
                    retries,
                    s,
                )
            };
            if !ok {
                return false;
            }
        }
    }
    true
}

/// The full native `find_rule_worker` decision (engine.py:2382-2510): all-numeric short-circuit, then
/// scan candidates shortest-first (variable-subset filtered), dispatch const-free -> M2 / const-bearing
/// -> M3, break on the first matching length, `select_best`. `candidates` = every expression up to
/// `max_target`, indexed by length here (M4b makes the index resident). Returns the chosen target.
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
) -> Result<Option<Vec<String>>, String> {
    // GUARD FIRST (engine.py:2384, BEFORE the short-circuit at :2390): allowed candidate lengths =
    // range(min(slen, max_target+1)) | range(slen); max(allowed) = max_cand_len-1 <= 0 -> nothing to
    // do. An all-constant source that simplify folded to length 1 has slen=1 -> caught here, exactly
    // as Python (the driver also Kruskal-prunes such sources before the worker is even called).
    let max_cand_len = match max_target {
        Some(mt) => simplified_length.min(mt + 1),
        None => simplified_length,
    };
    if max_cand_len <= 1 {
        return Ok(None);
    }
    // all-numeric short-circuit (engine.py:2390-2400), reached only after the guard passes.
    if source.len() > 1
        && source
            .iter()
            .all(|t| t == "<constant>" || ops.is_operator(t) || crate::utils::is_numeric_string(t))
    {
        let non_ops: Vec<&String> = source.iter().filter(|t| !ops.is_operator(t)).collect();
        if !non_ops.is_empty() && non_ops.iter().all(|t| crate::utils::is_numeric_string(t)) {
            if let Some(tok) = crate::numeric::evaluate_constant_subtree(source, ops) {
                return Ok(Some(vec![tok]));
            }
        }
        return Ok(Some(vec!["<constant>".to_string()]));
    }
    let n_vars = var_names.len();
    if x_flat.len() != n_rows * n_vars {
        return Err("x_flat shape mismatch".to_string());
    }
    let cols = columns_from_row_major(x_flat, n_rows, n_vars);

    let src_tape = Tape::compile(source, ops, var_names)?;
    let n_src_const = src_tape.n_params;
    let src_mask = var_mask(source, var_names);
    let combos = sign_combos(n_src_const);

    // index candidates by length (only the lengths we will scan).
    let mut by_len: Vec<Vec<CandEntry>> = (0..max_cand_len).map(|_| Vec::new()).collect();
    for c in candidates {
        let len = c.len();
        if len == 0 || len >= max_cand_len {
            continue;
        }
        let tape = Tape::compile(c, ops, var_names)?;
        let n_const = tape.n_params;
        let y_const_free = if n_const == 0 {
            Some(tape.eval_columns(&cols, &[], n_rows))
        } else {
            None
        };
        by_len[len].push(CandEntry {
            tokens: c.clone(),
            var_mask: var_mask(c, var_names),
            n_const,
            linearity: crate::fit::classify(c, ops)?,
            tape,
            y_const_free,
        });
    }

    let mut rng = Rng::new(seed);
    for length in 1..max_cand_len {
        let mut matches: Vec<Vec<String>> = Vec::new();
        for cand in &by_len[length] {
            if cand.var_mask & !src_mask != 0 {
                continue; // candidate uses a variable the source lacks
            }
            if candidate_matches(
                &src_tape,
                n_src_const,
                &combos,
                cand,
                &cols,
                n_rows,
                challenges,
                retries,
                rtol,
                atol,
                &mut rng,
            ) {
                matches.push(cand.tokens.clone());
            }
        }
        if !matches.is_empty() {
            return Ok(select_best(source, matches, ops)); // shortest matching length wins
        }
    }
    Ok(None)
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
            )
            .unwrap();
        assert_eq!(r2, None);
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
