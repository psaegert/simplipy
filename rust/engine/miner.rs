//! The OFFLINE mining surface of the `Engine`: thin delegates into `crate::eval` /
//! `crate::worker` / `crate::fit`, plus the mine driver's `set_rules` / `mine_one_length` /
//! `prune_explicit`.
//!
//! ONE ACCEPTANCE. Every mint channel -- the batch mine, the proposals' library scan and
//! the hint arms -- asks the same question through the same object: [`MineSpec`] says what
//! the mine mints FOR, [`Engine::mine_mark`] computes the bar a target must beat, and the
//! [`Acceptor`] built from that mark is the only acceptance predicate there is.

use rayon::prelude::*;

use crate::engine::RuleMode;
use crate::rules::CompiledRules;

use super::Engine;

/// One mined rule: (global source index, source, target).
pub type MinedRule = (u64, Vec<String>, Vec<String>);

/// What one mine mints FOR. Everything mode-dependent is derived from `mode`, once, here.
///
/// THE MARK IS THE SERVE ANSWER (`Engine::mine_mark`): a mode's mine asks the engine that
/// will serve its file, `simplify(s, mode)` exactly as a user calls it -- the mode's canon,
/// folds and number domain, its rules so far, the default effort and work budget.
///
/// THE NUMERIC ACCEPTANCE PER MODE (`rescue`). A const-free pair agrees relatively,
/// `|s - c| <= rtol*|c|`, so zero agrees only with zero; a FITTED constant keeps the
/// absolute band `atol + rtol*|c|`, the error of its f64 solve (`worker::candidate_matches`).
/// - `Default` (f64): the deployed evaluator's values decide; nothing is re-judged.
/// - `Real`, `Permissive`: f64 is a pre-filter; a near miss is re-judged at high precision
///   (`hiprec::rescue`), where a zero is a residual that collapses as the precision
///   doubles. Permissive accepts the union, which is real's acceptance.
#[derive(Clone, Copy, Debug)]
pub struct MineSpec {
    pub mode: RuleMode,
    pub rtol: f64,
    pub atol: f64,
    pub min_informative: usize,
    pub challenges: usize,
    pub retries: usize,
}

impl MineSpec {
    /// Whether a near miss is re-judged at high precision: every mode but f64.
    pub fn rescue(&self) -> bool {
        !matches!(self.mode, RuleMode::Default)
    }
}

/// The work budget `simplify` runs a mode's searches under by default: the shared
/// `DEFAULT_WORK`, the same in every mode (none).
pub fn serve_work(_mode: RuleMode) -> Option<u64> {
    super::DEFAULT_WORK
}

/// The engine's own answer for a source: the bar every target must beat. `mu` is `None`
/// only for a state the measure cannot price; the scan then keeps its token bound.
pub struct Mark {
    tokens: Vec<String>,
    mu: Option<u64>,
}

impl Mark {
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// A single atom: no sound target sits strictly below it (see `Acceptor::new`).
    pub fn is_atomic(&self) -> bool {
        self.tokens.len() == 1
    }
}

/// STAGE 2: the independent rows every NUMERIC pick is re-certified on before it may mint
/// (`judge_pair` on these rows, at a per-pair seed). Fresh rows and fresh constant draws
/// kill data-luck accepts -- agreement within tolerance only on the mine's own rows,
/// saturation plateaus, lucky constant draws -- with the same checker the scan runs.
pub struct ConfirmSet {
    cols: Vec<Vec<f64>>,
    n_rows: usize,
    min_informative: usize,
    seed: u64,
}

impl ConfirmSet {
    pub fn new(
        x_flat: &[f64],
        n_rows: usize,
        n_vars: usize,
        min_informative: usize,
        seed: u64,
    ) -> Result<Self, String> {
        Ok(ConfirmSet {
            cols: crate::worker::x_columns(x_flat, n_rows, n_vars)?,
            n_rows,
            min_informative: min_informative.max(1),
            seed,
        })
    }
}

/// Why the [`Acceptor`] refused a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MineRefusal {
    /// The licence registry (`refusals::mint_refusals`), by its wire name.
    Registry(&'static str),
    /// Not strictly below the mark in the serve ordering.
    NotBelow,
    /// The serve ordering could not compare the two sides (a bug; counted).
    Undecided,
    /// The pole entry: a defined value changes at a constructed exceptional point.
    Pole,
    /// A resolved literal that does not strictly descend or only respells the mark.
    NotStructural,
    /// The rule states a law the engine performs itself (`ac::rules::engine_owned_law`,
    /// the load gate's one predicate).
    EngineOwnedLaw,
    /// The judge loop refused this target for this source (its tier is not the mode's).
    Excluded,
    /// Stage 2: the pair does not certify on the independent rows.
    Unconfirmed,
}

impl MineRefusal {
    pub fn name(self) -> &'static str {
        match self {
            MineRefusal::Registry(name) => name,
            MineRefusal::NotBelow => "not_below",
            MineRefusal::Undecided => "undecided",
            MineRefusal::Pole => super::refusals::Refusal::PoleClassChange.name(),
            MineRefusal::NotStructural => "not_structural",
            MineRefusal::EngineOwnedLaw => "engine-owned-law",
            MineRefusal::Excluded => "excluded",
            MineRefusal::Unconfirmed => "unconfirmed",
        }
    }
}

/// THE acceptance predicate (one copy). Built from a source and its [`Mark`]; every mint
/// channel asks it, so a channel cannot accept what another refuses.
pub(crate) struct Acceptor<'a> {
    eng: &'a Engine,
    spec: &'a MineSpec,
    src: &'a [String],
    mark: Mark,
    vars: &'a [String],
    confirm: Option<&'a ConfirmSet>,
    /// Targets the judge loop refused for this source (`Refusal::Excluded`): the scan yields
    /// to the next candidate, so another tier's pick can never shadow a licensed one.
    excluded: &'a [Vec<String>],
}

impl<'a> Acceptor<'a> {
    /// `None` for an ATOMIC mark: nothing is mintable. When the engine folds the source to a
    /// single atom, no sound target exists strictly below it -- a different atom is a
    /// different VALUE, and the ordering tiebreak happily ranks atoms among themselves
    /// (+inf sits below -inf), so searching mints value-changes (signed-zero finding,
    /// 2026-08-02: two artifact rows claimed the IEEE value for states the engine's
    /// one-zero convention constructs away).
    pub(crate) fn new(
        eng: &'a Engine,
        spec: &'a MineSpec,
        src: &'a [String],
        mark: Mark,
        vars: &'a [String],
        confirm: Option<&'a ConfirmSet>,
        excluded: &'a [Vec<String>],
    ) -> Option<Self> {
        if mark.is_atomic() {
            return None;
        }
        Some(Acceptor {
            eng,
            spec,
            src,
            mark,
            vars,
            confirm,
            excluded,
        })
    }

    /// THE SEARCH BOUND IS THE ACCEPTANCE MARK (owner ruling, re-mine 2026-08-20): the
    /// scan stops at the first mu tier that cannot descend below the mark, so a token
    /// ceiling no longer splits search from acceptance.
    pub(crate) fn scan_bound(&self) -> Option<u64> {
        self.mark.mu
    }

    /// The licence registry, then the serve ordering, then the pole entry. A refused
    /// candidate yields to the next one in the scan. THE acceptance of a numeric match is
    /// `license` and then `confirm`; the exact short-circuit arms answer to `license`.
    ///
    /// The registry (`refusals::mint_refusals`) holds the determined-source Const gate
    /// and the special-absorption count, the latter taken on the MAX of the raw source
    /// and the mark (spelling independence, F49). The ordering is the one the rewrite
    /// pass fires under, so the serving engine never refuses a mined rule. The POLE entry
    /// runs LAST, only on candidates that pass the cheap licences and the ordering: it
    /// constructs the exceptional points of both sides and refuses a candidate that
    /// changes a DEFINED extended-real value there (the a.e. pole-mirror family, otherwise
    /// gate-invisible).
    pub(crate) fn license(&self, t: &[String]) -> Result<(), MineRefusal> {
        // The scan is deterministic, so an excluded pick recurs in exactly its spelling.
        if self.excluded.iter().any(|e| e.as_slice() == t) {
            return Err(MineRefusal::Excluded);
        }
        if let Some(r) = super::refusals::mint_refusals(self.src, &self.mark.tokens, t, self.vars) {
            return Err(MineRefusal::Registry(r.name()));
        }
        match self
            .eng
            .ac_ordered_below_in(t, &self.mark.tokens, self.spec.mode)
        {
            Some(true) => {}
            Some(false) => return Err(MineRefusal::NotBelow),
            None => {
                // Candidates and the engine's own result always parse (the core
                // serialization language is config-independent), so an UNDECIDABLE
                // ordering is a bug, never a soft refusal: loud in debug, counted in
                // release.
                debug_assert!(
                    false,
                    "ordering acceptance could not parse {t:?} vs {:?}",
                    self.mark.tokens
                );
                super::stats::ACCEPT_UNDECIDED_REFUSALS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(MineRefusal::Undecided);
            }
        }
        // The law, not a spelling: a mined variable reads as a `?` wildcard (a finite real,
        // what the mine's rows bind), so the predicate the rule load applies to the written
        // rule decides here on the same pattern.
        let pattern = |e: &[String]| -> Vec<String> {
            e.iter()
                .map(|tok| match self.vars.iter().position(|v| v == tok) {
                    Some(i) => format!("?{i}"),
                    None => tok.clone(),
                })
                .collect()
        };
        if self
            .eng
            .law_instance(&pattern(self.src), &pattern(t), self.spec.mode)
            == Some(Some("engine-owned-law"))
        {
            return Err(MineRefusal::EngineOwnedLaw);
        }
        if super::refusals::pole_refusal(self.src, t, &self.eng.operators, self.vars).is_some() {
            super::stats::REGISTRY_POLE_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(MineRefusal::Pole);
        }
        Ok(())
    }

    /// Stage 2: `t` must certify on the confirmation rows too (`judge_pair`, at a seed that
    /// hashes both sides, so the verdict depends on nothing else the scan judged). A pick
    /// that does not confirm yields to the next candidate. Without a `ConfirmSet` (a mine
    /// run with `confirm=False`) every pair passes.
    pub(crate) fn confirm(&self, t: &[String]) -> Result<(), MineRefusal> {
        let Some(c) = self.confirm else {
            return Ok(());
        };
        let verdict = crate::worker::judge_pair(
            &self.eng.operators,
            self.src,
            t,
            self.vars,
            &c.cols,
            c.n_rows,
            self.spec.challenges,
            self.spec.retries,
            crate::worker::pair_seed(c.seed, self.src, t),
            self.spec.rtol,
            self.spec.atol,
            c.min_informative,
            self.spec.rescue(),
        );
        match verdict {
            Ok(None) => Ok(()),
            // a pair the checker cannot even compile fails closed
            _ => Err(MineRefusal::Unconfirmed),
        }
    }

    /// The additional bar for COMPUTED literals (literal-resolved candidates and the
    /// ground short-circuit's evaluation): STRICT mu descent below the mark (fail closed
    /// on an unmeasurable side), and a literal skeleton different from the mark's.
    ///
    /// Resolution recovers structure whose constant must be computed; it never respells.
    /// Under mu a rounded literal has FEWER BITS than a long exact rational, so strict
    /// descent alone would re-license the f64-respell disease against long-literal marks
    /// (`(0.3333333333333333)^2 * x0`: a mu-descending VALUE CHANGE). A resolved target
    /// skeleton-equal to the mark differs only in literal content. Structure-recovering
    /// resolutions (`/ acos 0 np.pi -> 0.5`) are untouched. A materialized rounding of a
    /// special-bearing state is a long literal against a symbolic mark priced in single
    /// symbols, so strict descent refuses it from every spelling; the standing falsifier
    /// is `tests/test_special_constants.py` (no special-bearing rule with a rounded RHS).
    pub(crate) fn accept_resolved(&self, t: &[String]) -> Result<(), MineRefusal> {
        let tc = self.eng.mine_price(t, self.spec.mode);
        let descends = matches!((tc, self.mark.mu), (Some(tc), Some(mc)) if tc < mc);
        if descends
            && !self
                .eng
                .ac_same_literal_skeleton(t, &self.mark.tokens)
                .unwrap_or(true)
        {
            Ok(())
        } else {
            Err(MineRefusal::NotStructural)
        }
    }

    /// The library scan for this source, under this acceptance: the first candidate in
    /// scan order that the certification and the acceptance both pass.
    pub(crate) fn scan(
        &self,
        lib: &crate::worker::CandidateLibrary,
        max_target: Option<usize>,
        seed: u64,
    ) -> Result<Option<Vec<String>>, String> {
        let spec = self.spec;
        // The exact short-circuit arms answer to the licences; numeric matches to the
        // licences and then confirmation (see `worker::find_rule_with_lib`).
        let license = |t: &[String]| self.license(t).is_ok();
        let confirm = |t: &[String]| self.confirm(t).is_ok();
        let accept_resolved = |t: &[String]| self.accept_resolved(t).is_ok();
        // The token bound stays at SOURCE length: canon can respell a source shorter
        // without reducing it (`/ 1 exp x0` parses to `inv exp x0`), and tightening by
        // that respelled length would exclude true targets like `exp neg x0` (fewer
        // COMPLEXITY, equal tokens). It binds only when the mark is unpriceable; the mu
        // bound decides otherwise.
        crate::worker::find_rule_with_lib(
            &self.eng.operators,
            self.src,
            self.src.len(),
            max_target,
            self.scan_bound(),
            lib,
            spec.challenges,
            spec.retries,
            seed,
            spec.rtol,
            spec.atol,
            spec.min_informative,
            spec.rescue(),
            Some(&license),
            Some(&confirm),
            Some(&accept_resolved),
        )
    }
}

impl Engine {
    /// The MARK for a source in `mode`: the engine's own SERVE answer, priced -- the
    /// entry `simplify(s, mode)` takes (`effort=None`, the mode's default work budget),
    /// with the mode's rules installed so far. `None` on malformed input.
    pub fn mine_mark(&self, src: &[String], mode: RuleMode) -> Option<Mark> {
        let tokens = {
            let _work = crate::ac::work::search_budget_scope(serve_work(mode));
            self.ac_explore_proj(src, 48, mode, crate::engine::AcForm::Explicit, usize::MAX)?
        };
        let mu = self.mine_price(&tokens, mode);
        Some(Mark { tokens, mu })
    }

    /// The measure a mine prices marks, candidates and resolved literals in: the mode's own
    /// certified reading (what `complexity(e, mode)` prices and the mode's simplify
    /// descends), so the scan bound, the library tiers and the strict-descent bar agree.
    pub fn mine_price(&self, tokens: &[String], mode: RuleMode) -> Option<u64> {
        self.ac_complexity_certified(tokens, mode, mode)
    }

    /// THE SOURCES of one length in `mode`: each enumerated spelling's canonical
    /// projection as the mode reads it (`ac_mode_projection`), one per class (the first
    /// enumerated), atomic projections dropped (nothing mintable sits below an atom). What
    /// is certified is what is written. An unparseable spelling is kept as it is.
    pub fn mine_sources(&self, enumerated: &[Vec<String>], mode: RuleMode) -> Vec<Vec<String>> {
        let projected: Vec<Option<(Vec<String>, Vec<String>)>> = enumerated
            .par_iter()
            .map(|e| self.ac_mode_projection(e, mode))
            .collect();
        let mut seen: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (e, p) in enumerated.iter().zip(projected) {
            match p {
                None => out.push(e.clone()),
                Some((explicit, key)) => {
                    if explicit.len() > 1 && seen.insert(key) {
                        out.push(explicit);
                    }
                }
            }
        }
        out
    }

    /// The library scan for ONE source (the proposals' channel): the mark is computed
    /// here, from `spec`, so this channel and the batch mine share one acceptance. An
    /// atomic mark mints nothing.
    pub fn find_rule_lib(
        &self,
        src: &[String],
        lib: &crate::worker::CandidateLibrary,
        max_target: Option<usize>,
        spec: &MineSpec,
        confirm: Option<&ConfirmSet>,
        seed: u64,
    ) -> Result<Option<Vec<String>>, String> {
        let mark = self
            .mine_mark(src, spec.mode)
            .ok_or_else(|| "invalid or malformed prefix expression".to_string())?;
        match Acceptor::new(self, spec, src, mark, lib.var_names(), confirm, &[]) {
            None => Ok(None),
            Some(acc) => acc.scan(lib, max_target, seed),
        }
    }

    /// THE HINT ARM: a proposed (source, hint) pair, judged by the same acceptance the scan
    /// uses -- the licences, `judge_pair` on the library's own rows, then stage-2
    /// confirmation. `seed` is the source's scan seed, so the hint is certified exactly as
    /// the scan would certify it as a library candidate. `Ok(None)` is acceptance;
    /// `Ok(Some(reason))` names the first refusal (`atomic_mark` when the engine already
    /// folds the source to an atom).
    pub fn judge_hint(
        &self,
        src: &[String],
        hint: &[String],
        lib: &crate::worker::CandidateLibrary,
        spec: &MineSpec,
        confirm: Option<&ConfirmSet>,
        seed: u64,
    ) -> Result<Option<&'static str>, String> {
        let mark = self
            .mine_mark(src, spec.mode)
            .ok_or_else(|| "invalid or malformed prefix expression".to_string())?;
        let Some(acc) = Acceptor::new(self, spec, src, mark, lib.var_names(), confirm, &[]) else {
            return Ok(Some("atomic_mark"));
        };
        if let Err(r) = acc.license(hint) {
            return Ok(Some(r.name()));
        }
        if let Some(why) = crate::worker::judge_pair(
            &self.operators,
            src,
            hint,
            lib.var_names(),
            lib.cols(),
            lib.n_rows(),
            spec.challenges,
            spec.retries,
            seed,
            spec.rtol,
            spec.atol,
            spec.min_informative,
            spec.rescue(),
        )? {
            return Ok(Some(why));
        }
        Ok(acc.confirm(hint).err().map(MineRefusal::name))
    }

    /// The pair judgment on caller rows (`worker::judge_pair`) under `spec`'s tolerances.
    #[allow(clippy::too_many_arguments)]
    pub fn judge_pair(
        &self,
        src: &[String],
        tgt: &[String],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        spec: &MineSpec,
        seed: u64,
    ) -> Result<Option<&'static str>, String> {
        let cols = crate::worker::x_columns(x_flat, n_rows, var_names.len())?;
        crate::worker::judge_pair(
            &self.operators,
            src,
            tgt,
            var_names,
            &cols,
            n_rows,
            spec.challenges,
            spec.retries,
            seed,
            spec.rtol,
            spec.atol,
            spec.min_informative,
            spec.rescue(),
        )
    }

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
        self.find_rule_in(
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
            true,
        )
    }

    /// [`Engine::find_rule`] with the numeric acceptance of a mode (`rescue`: high-precision
    /// re-judging of near misses, every mode but f64; see `MineSpec::rescue`).
    #[allow(clippy::too_many_arguments)]
    pub fn find_rule_in(
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
        rescue: bool,
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
            rescue,
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
        mode: RuleMode,
    ) -> Result<crate::worker::CandidateLibrary, String> {
        // mu is scored HERE because it is an engine-level measure, and the library builder
        // only has the operator table. It is the measure the marks are priced in
        // (`mine_price`), so the scan bound compares like with like; it depends on no mined
        // rule, so scoring once at build time is stable for the whole mine.
        let mus: Vec<Option<u64>> = candidates
            .iter()
            .map(|c| self.mine_price(c, mode))
            .collect();
        // THE AC-CLASS QUOTIENT (2026-08-22 audit §7.1). The enumeration spells one
        // canonical class many ways -- commutative orderings, odd-sign twins, inv/abs
        // pairs -- and the scan judged every spelling: 31.6% of the variable-carrying
        // acj-4 library was duplicate classes. Two spellings of one class compute the
        // same function, so they match identically; the one the scan would PICK --
        // cheapest mu tier, then earliest in library order -- makes every later
        // spelling with the SAME variable multiset and `<constant>` count unreachable:
        // it sits in a same-or-later tier, sorts identically in select_best (stable by
        // `<constant>` count), and passes the wildcard-multiplicity and mu-acceptance
        // checks whenever the dropped one would. Spellings whose variable multiset or
        // `<constant>` count differ (`* x0 x0` vs `pow2 x0`) are NEVER dropped: their
        // guard behaviour is not interchangeable, and the byte-identity gate only
        // certifies the dominated class.
        let mut keeper: std::collections::HashMap<(Vec<String>, Vec<String>, usize), (u64, usize)> =
            std::collections::HashMap::new();
        let sig = |c: &[String]| -> (Vec<String>, usize) {
            let mut vars: Vec<String> = c
                .iter()
                .filter(|t| var_names.iter().any(|v| v == *t))
                .cloned()
                .collect();
            vars.sort();
            let n_const = c.iter().filter(|t| t.as_str() == "<constant>").count();
            (vars, n_const)
        };
        for (i, c) in candidates.iter().enumerate() {
            let Some(k) = self.ac_canonical_key(c, mode) else {
                continue; // unkeyed spellings are never quotiented
            };
            let (vars, n_const) = sig(c);
            let mu_i = mus[i].unwrap_or(u64::MAX);
            keeper
                .entry((k, vars, n_const))
                .and_modify(|best| {
                    if (mu_i, i) < *best {
                        *best = (mu_i, i);
                    }
                })
                .or_insert((mu_i, i));
        }
        let mut cands_q: Vec<Vec<String>> = Vec::with_capacity(candidates.len());
        let mut mus_q: Vec<Option<u64>> = Vec::with_capacity(candidates.len());
        for (i, c) in candidates.iter().enumerate() {
            let keep = match self.ac_canonical_key(c, mode) {
                None => true,
                Some(k) => {
                    let (vars, n_const) = sig(c);
                    keeper.get(&(k, vars, n_const)).is_none_or(|&(_, j)| j == i)
                }
            };
            if keep {
                cands_q.push(c.clone());
                mus_q.push(mus[i]);
            }
        }
        crate::worker::CandidateLibrary::build(
            &self.operators,
            &cands_q,
            &mus_q,
            var_names,
            x_flat,
            n_rows,
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
        self.ac_permissive_rules_cell = std::sync::OnceLock::new();
    }

    /// Install ONE MODE'S OWN COMPLETE RULE SET (`rules_real.json` / `rules_permissive.json`),
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
    /// `real` can never move a `default` or `permissive` answer.
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
            RuleMode::Permissive => {
                self.permissive_rules = compiled;
                self.ac_permissive_rules_cell = std::sync::OnceLock::new();
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
        spec: &MineSpec,
        confirm: Option<&ConfirmSet>,
        seed: u64,
        index_offset: u64,
        indices: Option<&[u64]>,
        exclusions: &std::collections::HashMap<u64, Vec<Vec<String>>>,
        relaxed_kruskal: bool,
    ) -> Vec<MinedRule> {
        // Publish the tier size so a driver polling `mine_progress()` sees within-tier progress
        // while this one blocking call runs.
        super::stats::mine_begin(sources.len() as u64);
        let none: Vec<Vec<String>> = Vec::new();
        sources
            .par_iter()
            .enumerate()
            .filter_map(|(k, src)| {
                // The GLOBAL source index: it seeds the source and keys its exclusions, so a
                // shard (`index_offset`) or a rescan (`indices`) draws exactly the seeds of a
                // one-shot mine.
                let idx = match indices {
                    Some(ix) => ix[k],
                    None => index_offset.wrapping_add(k as u64),
                };
                let excluded = exclusions.get(&idx).unwrap_or(&none);
                // AC-JUDGED Kruskal prune: the arbiter is the engine that will SERVE the
                // rules. A source the AC engine already strictly reduces -- by exact
                // arithmetic, bag collection, or the rules mined so far -- is skipped
                // (strict) or sets the mark to beat (relaxed). FINAL acceptance is the
                // SEMANTIC metric, so every mined rule strictly descends the AC reduction
                // ordering by construction and the serving engine never refuses its own
                // artifact.
                let out = {
                    // Mine sources are engine-enumerated and well-formed by construction;
                    // `None` (the malformed-input signal at the API boundary) cannot occur
                    // here, and treating it as "already reduced" (source unchanged) keeps
                    // this site total without a panic path in the worker threads.
                    let mark = self.mine_mark(src, spec.mode).unwrap_or_else(|| Mark {
                        tokens: src.to_vec(),
                        mu: self.mine_price(src, spec.mode),
                    });
                    // "Already reduced" is judged in the serve-time reduction ordering
                    // (one coverage ordering, everywhere): the walk strictly descended,
                    // or the result is a single atom -- nothing mintable sits strictly
                    // below an atomic state, so canon-owned collapses (`+ x0 0` parses
                    // straight to `x0`) land in that arm. A canon RESPELL (same state,
                    // shorter spelling) is neither and is searched: the token-shrink
                    // arm that used to skip it lost e.g. the `1/exp(t) -> exp(-t)`
                    // family to the enumeration alphabet.
                    let reduced = mark.is_atomic()
                        || match self.ac_ordered_below_in(mark.tokens(), src, spec.mode) {
                            Some(below) => below,
                            None => {
                                // Both spellings are the engine's own language (the
                                // closure invariant); an undecidable ordering is a
                                // bug, never a soft verdict: loud in debug, counted
                                // in release, and the source is SEARCHED, not
                                // skipped on a guess.
                                debug_assert!(
                                    false,
                                    "skip ordering could not parse {src:?} vs {:?}",
                                    mark.tokens()
                                );
                                super::stats::ACCEPT_UNDECIDED_REFUSALS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                false
                            }
                        };
                    if reduced && !relaxed_kruskal {
                        None
                    } else {
                        // An atomic mark ends the search even under relaxed Kruskal
                        // (`Acceptor::new`).
                        Acceptor::new(self, spec, src, mark, lib.var_names(), confirm, excluded)
                            .and_then(|acc| {
                                match acc.scan(lib, max_target, seed.wrapping_add(idx)) {
                                    Ok(Some(target)) => Some((idx, src.clone(), target)),
                                    _ => None,
                                }
                            })
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
