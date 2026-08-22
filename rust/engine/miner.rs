//! The OFFLINE mining surface of the `Engine`: thin delegates into `crate::eval` /
//! `crate::worker` / `crate::fit`, plus the mine driver's `set_rules` / `mine_one_length` /
//! `prune_explicit`.

use rayon::prelude::*;

use crate::engine::RuleMode;
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
        )
    }

    /// OFFLINE miner: build a RESIDENT candidate library (once per mine). Var-free
    /// composites are dropped unconditionally -- the library mirrors the emit guard; see
    /// `crate::worker::CandidateLibrary::build`.
    pub fn build_candidate_library(
        &self,
        candidates: &[Vec<String>],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
    ) -> Result<crate::worker::CandidateLibrary, String> {
        // mu is scored HERE because it is an engine-level measure (`ac_complexity` reads the
        // bare context), and the library builder only has the operator table. Bare-context mu
        // depends on no mined rule, so scoring once at build time is stable for the whole mine.
        let mus: Vec<Option<u64>> = candidates.iter().map(|c| self.ac_complexity(c)).collect();
        crate::worker::CandidateLibrary::build(
            &self.operators,
            candidates,
            &mus,
            var_names,
            x_flat,
            n_rows,
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
        max_mu: Option<u64>,
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
            max_mu,
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

    /// The registry's pole entry with the engine's own operator table -- the FFI
    /// channel's entry point (`find_rule_lib`'s accept closure); the native mine
    /// path calls `refusals::pole_refusal` directly with the same table.
    pub fn mint_pole_refusal(
        &self,
        src: &[String],
        target: &[String],
        var_names: &[String],
    ) -> bool {
        super::refusals::pole_refusal(src, target, &self.operators, var_names).is_some()
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
        // ...and so do the other two modes' cells. Not because their SETS changed -- they
        // did not -- but because a mode with NO set of its own SERVES this one, so its
        // answer moved even though its file did not. Retaining them would leave such a
        // mode serving a rule set the engine no longer holds.
        self.ac_real_rules_cell = std::sync::OnceLock::new();
        self.ac_corpus_rules_cell = std::sync::OnceLock::new();
    }

    /// Install ONE MODE'S OWN COMPLETE RULE SET (`rules_real.json` / `rules_corpus.json`),
    /// or -- with `None` -- retract it so that mode falls back to the default set again.
    ///
    /// `Some(vec![])` is NOT the same call as `None`: it installs an EMPTY set, and that
    /// mode then serves nothing. The distinction is the design (see `Engine::ac_rules_for`)
    /// and it survives every layer up to the config loader.
    ///
    /// `RuleMode::Default` routes to `set_rules`, so there is exactly one way to say
    /// "replace the default set" and it keeps that path's invalidation discipline.
    /// Shaped like `set_rules` otherwise: the same `&mut self` token-table extension (the
    /// table is append-only, so existing ids -- including `bang_cache` keys -- stay valid)
    /// and the same lazy-cell invalidation. Only the named mode's cell dies; installing
    /// `real` can never move a `default` or `corpus` answer.
    pub fn set_mode_rules(&mut self, mode: RuleMode, raw: Option<Vec<(Vec<String>, Vec<String>)>>) {
        if mode == RuleMode::Default {
            self.set_rules(raw.unwrap_or_default());
            return;
        }
        let compiled = raw.map(|r| CompiledRules::compile(r, &mut self.tokens, &self.operators));
        match mode {
            RuleMode::Real => {
                self.real_rules = compiled;
                self.ac_real_rules_cell = std::sync::OnceLock::new();
            }
            RuleMode::Corpus => {
                self.corpus_rules = compiled;
                self.ac_corpus_rules_cell = std::sync::OnceLock::new();
            }
            RuleMode::Default => unreachable!("handled above"),
        }
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
                        // UNFOLDED: the mine's canonical form is the finest-grained one,
                        // so no rule is lost at discovery. See `Engine::ac_judge`.
                        .ac_simplify_proj_fold(
                            src,
                            48,
                            RuleMode::Default,
                            crate::engine::AcForm::Explicit,
                            false,
                        )
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
                        // STAGE-1 CONST-ABSORPTION LICENCE, mint-side twin. Doctrine as
                        // AMENDED 2026-08-08 (contract 10.11, supersedes the 2026-08-01
                        // blanket refusal): a special absorbs into an EXISTING `<constant>`
                        // through the bijective constructor sites (Add-shift / Mul-scale;
                        // `C + pi -> C'` folds in the engine now), so such sources reach an
                        // ATOMIC endpoint and never arrive here. What this licence still
                        // refuses is the NON-absorbable remainder: a `<constant>`-bearing
                        // target eating a special the constructor's site list would not
                        // (introduced masks -- `* np.pi _0 -> * <constant> _0`; and
                        // non-bijective positions -- `pow np.pi <constant> -> <constant>`,
                        // whose value set is (0,inf), not "anything"). Historical incident
                        // this closed: the scan minted `+ <constant> np.pi -> <constant>`
                        // (2026-08-01) while the then-doctrine refused it engine-side --
                        // the rules channel resurrecting a constructor refusal, C1.20's
                        // first recorded instance. Special-count-preserving targets and
                        // Const-free exact hits (`sin np.pi -> 0`) are untouched.
                        // THE LICENCE REGISTRY (C1.20's dual, 2026-08-08): the two
                        // syntactic licences that lived inline here -- the determined-
                        // source Const-introduction gate and the special-absorption
                        // count (its full incident history: tokens -> spellings ->
                        // `ac_out` endpoints, F49/F55) -- are RELOCATED verbatim into
                        // `engine::refusals::mint_refusals` (single home, shared with
                        // the AC load consumer). Behavior-identical by construction;
                        // the relocation's falsifier is the triple re-mine's byte
                        // identity. The endpoint-count subtlety (max of raw source and
                        // `ac_out` counts, spelling independence) lives with the
                        // registry now -- see refusals.rs.
                        //
                        // The POLE entry (SWEEP row 3) runs LAST, only on candidates
                        // that pass the cheap licences AND the ordering acceptance:
                        // it constructs the exceptional points of both sides (affine
                        // denominators, exact rational roots) and refuses a candidate
                        // that changes a DEFINED extended-real value there under the
                        // contract evaluator. This is the gate that keeps the a.e.
                        // pole-mirror family (`neg inv - 1 _0 -> inv - _0 1`, verified
                        // mu-descending and otherwise gate-invisible) out of the
                        // artifact when the tiers rise past (4,3). A refused candidate
                        // yields to the next one, exactly like every other licence.
                        let accept = |t: &[String]| {
                            if super::refusals::mint_refusals(src, &ac_out, t, lib.var_names())
                                .is_some()
                            {
                                return false;
                            }
                            let below = match self.ac_ordered_below(t, &ac_out) {
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
                            };
                            if !below {
                                return false;
                            }
                            if super::refusals::pole_refusal(
                                src,
                                t,
                                &self.operators,
                                lib.var_names(),
                            )
                            .is_some()
                            {
                                super::stats::REGISTRY_POLE_REFUSALS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return false;
                            }
                            true
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
                        // THE SEARCH BOUND IS THE ACCEPTANCE MARK (owner ruling, re-mine
                        // 2026-08-20). `mark_c` is mu of the engine's own endpoint for this
                        // source -- exactly what `accept_resolved` above requires a target to
                        // beat -- so passing it as the scan bound removes the last split
                        // between a token bound for SEARCH and mu for ACCEPTANCE. The scan
                        // now stops at the first mu tier that cannot descend, and reaches
                        // longer-but-cheaper targets the token ceiling used to refuse.
                        match self.find_rule_with_lib(
                            src,
                            src.len(),
                            max_target,
                            mark_c,
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
