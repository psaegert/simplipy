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

    /// OFFLINE miner: the `find_rule_worker` decision over a resident library. `accept`
    /// threads the caller's acceptance criterion into the candidate scan (finding F2):
    /// pass the serve-time ordering so the scan itself only yields targets the serving
    /// pass would fire.
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
        accept: Option<crate::worker::AcceptFn<'_>>,
        accept_resolved: Option<crate::worker::AcceptFn<'_>>,
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
            accept,
            accept_resolved,
        )
    }

    /// OFFLINE (mine driver): replace the engine's rules (recompile). Used by the mine driver
    /// to GROW the Kruskal-prune rule set length-by-length (the Python outer loop dedups/canonicalizes
    /// the found rules into wildcard patterns, then sets them here before the next length's inner loop).
    /// Takes `&mut self`, so extending the token table with any NEW rule tokens is safe
    /// (append-only; existing ids -- including `bang_cache` keys -- stay valid).
    pub fn set_rules(&mut self, raw: Vec<(Vec<String>, Vec<String>)>) {
        self.rules = CompiledRules::compile(raw, &mut self.tokens, &self.operators);
        // The AC translation is cached lazily (ac_rules_cell); the AC-judged Kruskal prune
        // must see the rules just installed, so the cache dies with the old rule set.
        self.ac_rules_cell = std::sync::OnceLock::new();
    }

    /// OFFLINE (mine driver): mine ONE source-length IN PARALLEL (rayon, all cores). For each
    /// source: AC-JUDGED Kruskal-prune (`ac_simplify` + semantic complexity, current rules) --
    /// skip if the serving engine already reduces it -- else `find_rule` on it. Returns the found (source -> target) rules. Within a length the rule set is FIXED (the
    /// Python driver `set_rules` between lengths = the order-dependent barrier), so the parallel map is
    /// pure reads (`&self`), safe and deterministic (per-source seed = base seed + index).
    ///
    /// `relaxed_kruskal=true` (the production default): EVERY source is searched with the
    /// mark to beat set to the engine's own result -- the serve-time ordering acceptance
    /// rides the candidate scan (finding F2), so only targets strictly below that mark are
    /// selectable and a refused candidate yields to the next one (next length included);
    /// such rules fire top-down as one-step shortcuts. With `false` (strict Kruskal), a
    /// source is skipped iff the engine already strictly reduces its STATE in the serve
    /// ordering (walk descent), or the state is atomic (provably nothing to mint). A
    /// RESPELL -- canon spelling the same state shorter, `/ 1 exp x0` -> `inv exp x0` --
    /// is neither and is searched in BOTH modes (one coverage ordering, everywhere): the
    /// historical token-shrink skip silently lost such families to the enumeration
    /// alphabet (the respelled form speaks minted literals/`pow`/`inv`, tokens the ladder
    /// never enumerates). Strict mode still loses one-step shortcuts relative to relaxed
    /// where the walk genuinely descends part-way (degenerate constant collapses like
    /// `C * acos(np.e) -> <constant>`).
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
        // Publish the tier size so a driver polling `mine_progress()` sees within-tier progress
        // while this one blocking call runs.
        super::stats::mine_begin(sources.len() as u64);
        sources
            .par_iter()
            .enumerate()
            .filter_map(|(idx, src)| {
                // AC-JUDGED Kruskal prune: the arbiter is the engine that will SERVE the
                // rules. A source the AC engine already strictly reduces -- by exact
                // arithmetic, bag collection, or the rules mined so far -- is skipped
                // (strict) or sets the mark to beat (relaxed). The candidate search keeps
                // its token-length bound (the library is length-organized); FINAL
                // acceptance is the SEMANTIC metric, so every mined rule strictly
                // descends the AC reduction ordering by construction and the serving
                // engine never refuses its own artifact.
                let out = {
                    // Mine sources are engine-enumerated and well-formed by construction;
                    // `None` (the malformed-input signal at the API boundary) cannot occur
                    // here, and treating it as "already reduced" (source unchanged) keeps
                    // this site total without a panic path in the worker threads.
                    let ac_out = self
                        .ac_simplify_proj(src, 48, false, crate::engine::AcForm::Explicit)
                        .unwrap_or_else(|| src.to_vec());
                    // "Already reduced" is judged in the serve-time reduction ordering
                    // (one coverage ordering, everywhere): the walk strictly descended,
                    // or the result is a single atom -- nothing mintable sits strictly
                    // below an atomic state, so canon-owned collapses (`+ x0 0` parses
                    // straight to `x0`) land in that arm. A canon RESPELL (same state,
                    // shorter spelling) is neither and is searched: the token-shrink
                    // arm that used to skip it lost e.g. the `1/exp(t) -> exp(-t)`
                    // family to the enumeration alphabet.
                    let reduced = ac_out.len() == 1
                        || match self.ac_ordered_below(&ac_out, src) {
                            Some(below) => below,
                            None => {
                                // Both spellings are the engine's own language (the
                                // closure invariant); an undecidable ordering is a
                                // bug, never a soft verdict: loud in debug, counted
                                // in release, and the source is SEARCHED, not
                                // skipped on a guess.
                                debug_assert!(
                                    false,
                                    "skip ordering could not parse {src:?} vs {ac_out:?}"
                                );
                                super::stats::ACCEPT_UNDECIDED_REFUSALS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                false
                            }
                        };
                    // ATOMIC MARK ENDS THE SEARCH, even under relaxed Kruskal
                    // (signed-zero finding, 2026-08-02): when the engine folds the
                    // source to a single atom, no sound target exists strictly below
                    // it -- a different atom is a different VALUE, and the ordering
                    // tiebreak happily ranks atoms among themselves (+inf sits below
                    // -inf), so searching mints value-changes. Found live: the f64
                    // challenge layer and the interval classifier both follow IEEE
                    // signed zero (`-1 / -0.0 = +inf`, deployment semantics, which
                    // they must keep for variable-bearing measure questions), while
                    // the engine's ratified one-zero convention folds
                    // `/ (-1) neg 0` to `-inf` -- two artifact rows claimed the
                    // IEEE value for states the engine constructs away, surviving
                    // only because load-normalization rejected them.
                    if ac_out.len() == 1 || (reduced && !relaxed_kruskal) {
                        None
                    } else {
                        let s = seed.wrapping_add(idx as u64);
                        // The token bound stays at SOURCE length: canon can respell a
                        // source shorter without reducing it (`/ 1 exp x0` parses to
                        // `inv exp x0`), and tightening by that respelled length would
                        // exclude true targets like `exp neg x0` (fewer COMPLEXITY, equal
                        // tokens). "Beat what the engine reaches" is enforced exactly by
                        // the ordering acceptance below, not by a token proxy.
                        //
                        // Serve-time acceptance RIDES THE SCAN (finding F2): the target
                        // must sit strictly below the engine's own result in the full
                        // reduction ordering (complexity, then canonical order) -- a
                        // refused candidate asks the scan for the next one, so equal-
                        // complexity canonicalization rules mint iff they fire and a
                        // longer-but-lower spelling is still reachable.
                        //
                        // STAGE-1 CONST-ABSORPTION LICENCE, mint-side twin (owner-ratified
                        // 2026-08-01, mask x special stays unabsorbed; a future theorem of
                        // the unified measure, design/UNIFIED_SIMPLICITY_MEASURE.md §5): a
                        // special may never vanish INTO a fitted constant, so for a
                        // special-bearing source a `<constant>`-bearing target must
                        // preserve every special occurrence. This closes the rules-channel
                        // resurrection of the engine-side licence: without it the scan
                        // minted `+ <constant> np.pi -> <constant>` (plain candidate,
                        // Const-count non-increasing, complexity strictly below the
                        // no-longer-collapsing mark) and the artifact would absorb what
                        // the constructor now refuses. Special-count-preserving targets
                        // and Const-free exact hits (`sin np.pi -> 0`) are untouched.
                        let n_special = |ts: &[String]| {
                            ts.iter().filter(|t| *t == "np.pi" || *t == "np.e").count()
                        };
                        // COUNTED ON THE STATE THE MASK ACTUALLY MEETS (2026-08-07), not on
                        // the source's SPELLING. `exp(1) -> E` in the constructor gave `e` a
                        // second spelling, and `* <constant> exp 1` and `/ <constant> exp (-1)`
                        // carry no `np.e` token while canonicalizing to
                        // `<mul> np.e <constant> </mul>`. The raw scan passed them and the
                        // re-mine minted SEVEN absorptions this very licence forbids -- the
                        // rules channel resurrecting the engine-side refusal a second time,
                        // through a spelling rather than through a channel.
                        //
                        // `ac_out` is the engine's own endpoint for this source, i.e. exactly
                        // the state a bare `<constant>` target would replace, so counting
                        // there is spelling-INDEPENDENT: it closes the channel for every
                        // spelling of a special the canon may grow, not for the ones we
                        // happened to enumerate. The raw count stays in the max because a
                        // rule fires on the SOURCE too, and a special the engine's own pass
                        // removes must still not be absorbed by the target.
                        let src_specials = n_special(src).max(n_special(&ac_out));
                        // DETERMINED sources (var-free, Const-free) never accept a
                        // Const-INTRODUCING target: such a source is a single value
                        // the engine folds for itself (or a special-exact state whose
                        // collapses mint as exact alphabet literals) -- `<constant>`
                        // for it is vacuous abstraction the engine's own semantics
                        // refuse (fold materializes values; it never masks them).
                        // Mirrors the Finite-arm contract gate in worker.rs; without
                        // this the candidate scan's bare-`<constant>` const-fit
                        // re-mints what that arm no longer answers. Variable-bearing
                        // abstraction (`* asin (-1) _0 -> * <constant> _0`, the
                        // artifact-only translation-dropped family) is untouched.
                        let src_determined = !src.iter().any(|x| x == "<constant>")
                            && !src.iter().any(|x| lib.var_names().contains(x));
                        let accept = |t: &[String]| {
                            if src_determined && t.iter().any(|x| x == "<constant>") {
                                return false;
                            }
                            if src_specials > 0
                                && t.iter().any(|x| x == "<constant>")
                                && n_special(t) < src_specials
                            {
                                return false;
                            }
                            match self.ac_ordered_below(t, &ac_out) {
                                Some(below) => below,
                                None => {
                                    // Candidates and the engine's own result always parse
                                    // (the core serialization language is config-
                                    // independent), so an UNDECIDABLE ordering is a bug,
                                    // never a soft refusal: loud in debug, counted in
                                    // release.
                                    debug_assert!(
                                        false,
                                        "ordering acceptance could not parse {t:?} vs {ac_out:?}"
                                    );
                                    super::stats::ACCEPT_UNDECIDED_REFUSALS
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    false
                                }
                            }
                        };
                        // Literal-RESOLVED targets must beat the mark in STRICT mu
                        // (fail closed on an unmeasurable side): resolution recovers
                        // structure whose constant must be computed, never respells.
                        //
                        // STAGE 2: the stage-1 resolution licence (total refusal for
                        // special-bearing sources) is DELETED -- it is now a theorem of
                        // the measure. A materialized rounding of a special-bearing
                        // state is a ~105-unit literal against a symbolic mark priced
                        // in single symbols (`* np.pi x0`: mark 24, materialization
                        // ~117), so the strict-descent gate refuses it by arithmetic,
                        // from every spelling. What the deletion ADDS is the sound
                        // half the fail-closed licence gave up: a resolution whose
                        // correctly-rounded value is genuinely SHORT (an exact
                        // collapse arriving through the resolution channel) descends
                        // and may mint -- the hiprec arbiter computes at 1024 bits, so
                        // an inexact value cannot masquerade as a short rational.
                        // Standing falsifier: no special-bearing rule with a rounded
                        // decimal RHS may appear in any mine
                        // (tests/test_special_constants.py pins it).
                        // RESPELL GUARD (stage-2 reflection, 2026-08-01): under mu a
                        // rounded literal has FEWER BITS than a long exact rational,
                        // so strict descent alone re-licensed the f64-respell disease
                        // against long-literal marks (`(0.3333333333333333)^2 * x0`:
                        // exact 32-digit coefficient ~210 units, its f64 rounding
                        // ~105 -- a mu-descending VALUE CHANGE). A resolved target
                        // skeleton-equal to the mark differs only in literal content:
                        // same value (not strictly below, refused anyway) or a
                        // respell. Structure-recovering resolutions (different
                        // skeleton: `/ acos 0 np.pi -> 0.5`) are untouched. Fail
                        // closed on unparseable sides.
                        let mark_c = self.ac_complexity(&ac_out);
                        let accept_resolved = |t: &[String]| {
                            matches!((self.ac_complexity(t), mark_c), (Some(tc), Some(mc)) if tc < mc)
                                && !self.ac_same_literal_skeleton(t, &ac_out).unwrap_or(true)
                        };
                        match self.find_rule_with_lib(
                            src,
                            src.len(),
                            max_target,
                            lib,
                            challenges,
                            retries,
                            s,
                            rtol,
                            atol,
                            min_informative,
                            Some(&accept),
                            Some(&accept_resolved),
                        ) {
                            Ok(Some(target)) => Some((src.clone(), target)),
                            _ => None,
                        }
                    }
                };
                super::stats::mine_tick(); // count the source whether or not it yielded a rule
                out
            })
            .collect()
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
