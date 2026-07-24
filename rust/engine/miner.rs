//! The OFFLINE mining surface of the `Engine`: thin delegates into `crate::eval` /
//! `crate::worker` / `crate::fit`, plus the mine driver's `set_rules` / `mine_one_length` /
//! `prune_explicit`.

use rayon::prelude::*;

use crate::rules::CompiledRules;

use super::Engine;

impl Engine {
    /// OFFLINE miner kernel: vectorized evaluation of a prefix expression over a batch of
    /// rows. Variable leaves index `var_names` (column order), `<constant>` leaves bind to `params`
    /// left-to-right, numeric/special literals fold to their value. See `crate::eval`.
    pub fn evaluate_batch(
        &self,
        tokens: &[String],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        params: &[f64],
    ) -> Result<Vec<f64>, String> {
        crate::eval::evaluate_batch(&self.operators, tokens, var_names, x_flat, n_rows, params)
    }

    /// OFFLINE miner: the no-constant equivalence test. See `crate::worker`.
    #[allow(clippy::too_many_arguments)]
    pub fn equivalent_no_const_check(
        &self,
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
        crate::worker::equivalent_no_const_check(
            &self.operators,
            source,
            candidate,
            var_names,
            x_flat,
            n_rows,
            challenges,
            rtol,
            atol,
            min_informative,
            seed,
        )
    }

    /// OFFLINE miner: the full native `find_rule_worker` decision for one source. See
    /// `crate::worker::find_rule`.
    #[allow(clippy::too_many_arguments)]
    pub fn find_rule(
        &self,
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
        crate::worker::find_rule(
            &self.operators,
            source,
            simplified_length,
            max_target,
            candidates,
            var_names,
            x_flat,
            n_rows,
            challenges,
            retries,
            seed,
            rtol,
            atol,
            min_informative,
            fold_filter,
        )
    }

    /// OFFLINE miner: build a RESIDENT candidate library (once per mine). See
    /// `crate::worker::CandidateLibrary` for the `fold_filter` (var-free candidate minimization)
    /// semantics and its soundness argument.
    pub fn build_candidate_library(
        &self,
        candidates: &[Vec<String>],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        fold_filter: bool,
    ) -> Result<crate::worker::CandidateLibrary, String> {
        crate::worker::CandidateLibrary::build(
            &self.operators,
            candidates,
            var_names,
            x_flat,
            n_rows,
            fold_filter,
        )
    }

    /// OFFLINE miner: the `find_rule_worker` decision over a resident library.
    #[allow(clippy::too_many_arguments)]
    pub fn find_rule_with_lib(
        &self,
        source: &[String],
        simplified_length: usize,
        max_target: Option<usize>,
        lib: &crate::worker::CandidateLibrary,
        challenges: usize,
        retries: usize,
        seed: u64,
        rtol: f64,
        atol: f64,
        min_informative: usize,
    ) -> Result<Option<Vec<String>>, String> {
        crate::worker::find_rule_with_lib(
            &self.operators,
            source,
            simplified_length,
            max_target,
            lib,
            challenges,
            retries,
            seed,
            rtol,
            atol,
            min_informative,
        )
    }

    /// OFFLINE (mine driver): replace the engine's rules (recompile). Used by the mine driver
    /// to GROW the Kruskal-prune rule set length-by-length (the Python outer loop dedups/canonicalizes
    /// the found rules into wildcard patterns, then sets them here before the next length's inner loop).
    /// Takes `&mut self`, so extending the token table with any NEW rule tokens is safe
    /// (append-only; existing ids -- including `bang_cache` keys -- stay valid).
    pub fn set_rules(&mut self, raw: Vec<(Vec<String>, Vec<String>)>) {
        self.rules = CompiledRules::compile(raw, &mut self.tokens, &self.operators);
    }

    /// OFFLINE (mine driver): mine ONE source-length IN PARALLEL (rayon, all cores). For each
    /// source: Kruskal-prune via `simplify` (current rules) -- skip if it shortens -- else `find_rule`
    /// on it. Returns the found (source -> target) rules. Within a length the rule set is FIXED (the
    /// Python driver `set_rules` between lengths = the order-dependent barrier), so the parallel map is
    /// pure reads (`&self`), safe and deterministic (per-source seed = base seed + index).
    ///
    /// `relaxed_kruskal=true` (the production default): a source the current rules already
    /// shorten is still searched, with the bound tightened to its simplified length -- i.e. only
    /// targets STRICTLY SHORTER than what simplify already reaches are accepted; such rules fire
    /// top-down as one-step shortcuts. With `false`, the source is SKIPPED (strict Kruskal).
    /// Relaxed mining is the only lever that reaches one-step shortcut rules (degenerate constant
    /// collapses like `C * acos(np.e) -> <constant>` and value-specific rewrites like
    /// `pow(_0, mult5(1)) -> pow5(_0)`).
    #[allow(clippy::too_many_arguments)]
    pub fn mine_one_length(
        &self,
        sources: &[Vec<String>],
        lib: &crate::worker::CandidateLibrary,
        max_target: Option<usize>,
        challenges: usize,
        retries: usize,
        seed: u64,
        rtol: f64,
        atol: f64,
        min_informative: usize,
        relaxed_kruskal: bool,
    ) -> Vec<(Vec<String>, Vec<String>)> {
        sources
            .par_iter()
            .enumerate()
            .filter_map(|(idx, src)| {
                // Kruskal prune: simplify with the current rules; skip if it
                // shortens (strict), or tighten the search bound to the simplified length (relaxed).
                let slen = self.simplify(src, 48, None, true, false).len();
                if slen < src.len() && !relaxed_kruskal {
                    return None;
                }
                let s = seed.wrapping_add(idx as u64);
                match self.find_rule_with_lib(
                    src,
                    slen,
                    max_target,
                    lib,
                    challenges,
                    retries,
                    s,
                    rtol,
                    atol,
                    min_informative,
                ) {
                    Ok(Some(target)) => Some((src.clone(), target)),
                    _ => None,
                }
            })
            .collect()
    }

    /// OFFLINE: prune redundant explicit rules against the compiled rule set `simplify` actually
    /// uses. Mutates the Rust `exact_rules` map: for each explicit `lhs` (in the given asset
    /// order), remove it, `simplify(lhs)` with the deployed config, and keep it removed IFF the
    /// result still equals its `rhs` (i.e. covered by other rules / folding). Serial BY DESIGN --
    /// a pruned rule stays removed for subsequent tests (avoids over-pruning mutually-redundant
    /// pairs). Returns the pruned `lhs` list; leaves the map in the pruned state.
    /// A `lhs` with any token absent from the table cannot be a compiled rule key -> skip
    /// (same "not an explicit rule in this engine" outcome).
    pub fn prune_explicit(
        &mut self,
        ordered_lhs: &[Vec<String>],
        mask_elementary_literals: bool,
    ) -> Vec<Vec<String>> {
        let mut pruned = Vec::new();
        for lhs in ordered_lhs {
            let Some(lhs_t) = self.tokens.lookup_seq(lhs) else {
                continue; // token not interned -> not an explicit rule in this engine (skip)
            };
            let Some(rhs_t) = self.rules.exact_rules.remove(&lhs_t) else {
                continue; // not an explicit rule in this engine (skip)
            };
            let rhs: Vec<String> = rhs_t
                .iter()
                .map(|&t| self.tokens.resolve(t).to_string())
                .collect();
            let simplified = self.simplify(lhs, 48, None, true, false);
            // rhs may be stored in masked (placeholder) form; mask to compare like-for-like.
            let result = if mask_elementary_literals {
                self.mask(&simplified)
            } else {
                simplified
            };
            if result == rhs {
                pruned.push(lhs.clone()); // redundant: keep removed
            } else {
                self.rules.exact_rules.insert(lhs_t, rhs_t); // needed: restore
            }
        }
        pruned
    }

    /// OFFLINE miner: native `exist_constants_that_fit` for affine-in-params candidates
    /// (closed-form least squares + allclose). `None` for nonlinear-in-params (LM path).
    #[allow(clippy::too_many_arguments)]
    pub fn exist_constants_fit_linear(
        &self,
        candidate: &[String],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        y_target: &[f64],
        rtol: f64,
        atol: f64,
    ) -> Result<Option<bool>, String> {
        crate::fit::exist_constants_fit_linear(
            &self.operators,
            candidate,
            var_names,
            x_flat,
            n_rows,
            y_target,
            rtol,
            atol,
        )
    }

    /// OFFLINE miner: native `exist_constants_that_fit` -- affine candidates
    /// via the closed-form path, nonlinear-in-params via `n_restarts` LM solves. See `crate::fit`.
    #[allow(clippy::too_many_arguments)]
    pub fn exist_constants_fit(
        &self,
        candidate: &[String],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        y_target: &[f64],
        rtol: f64,
        atol: f64,
        n_restarts: usize,
        seed: u64,
    ) -> Result<bool, String> {
        crate::fit::exist_constants_fit(
            &self.operators,
            candidate,
            var_names,
            x_flat,
            n_rows,
            y_target,
            rtol,
            atol,
            n_restarts,
            seed,
        )
    }
}
