//! The Rust inline backend for SimpliPy, exposed to Python as `simplipy._core` via PyO3.
//!
//! ## FFI design (load-bearing, per the verified analysis)
//! The ENTIRE simplify fixpoint is ported as ONE FFI unit. A single call into Rust receives the
//! prefix token list and returns the simplified token list. Inside Rust, the whole recursion runs
//! with NO boundary crossings:
//!   cancel_terms -> apply_simplification_rules (parse_subtree + apply_rules_top_down +
//!   match_pattern_with_cert + apply_mapping + the constant fold) -> sort_operands ->
//!   mask_elementary_literals, over a best-first tree search bounded by `node_budget`.
//! ~1.8 boundary crossings/expr; FFI marshalling stays <1% of wall time.
//! Porting the pattern matcher alone is a TRAP (millions of crossings -> the speedup evaporates).
//! Therefore the PyO3 layer here is deliberately THIN: marshal `list[str]` <-> `Vec<String>`,
//! hold the compiled engine, and delegate the whole unit to `engine::Engine::simplify`.
//!
//! ## One engine line
//! The core implements the contract semantics as a single engine line: numeric constant
//! folding (incl. `1/0 -> float("inf")`, via the `numeric` module's f64 + libm evaluator),
//! the corrected conversions, and atomic inf/nan tokens. Byte-exact reproduction of the
//! historical dev_7-3 / v23.0-era behavior is served by installing `simplipy<=0.6.0`.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Reject inputs the recursive kernel cannot safely handle, with a clean `ValueError` BEFORE it runs.
/// Two off-distribution hazards, both of which would otherwise ABORT the interpreter (uncatchable):
///   1. PATHOLOGICALLY LONG input -> the tree recursions (cancel / parse / apply_rules / sort) recurse
///      ~one frame per nesting level and overflow the default stack. Real expressions are tiny
///      (<~100 tokens); cap generously. (Python raises RecursionError in the same regime.)
///   2. MALFORMED prefix (an operator with too few operands) -> `cancel_terms` / `parse_subtree`
///      index-underflow-PANIC, and a panic across the GIL-released `detach` boundary can abort.
///      `is_valid` is the (240k-differential-verified) arity check, false exactly on those shapes.
///      Empty input is the one valid case `is_valid` rejects (`simplify([]) == []`), allowed through.
/// Both checks are O(n) and only fire off the deployed distribution; valid inputs are unaffected. We
/// deliberately do NOT replicate Python's path-dependent exception TYPE -- one clean `ValueError`.
const MAX_TOKENS: usize = 4096;

fn ensure_well_formed(inner: &engine::Engine, tokens: &[String]) -> PyResult<()> {
    if tokens.len() > MAX_TOKENS {
        return Err(PyValueError::new_err(format!(
            "prefix expression too long ({} tokens > {MAX_TOKENS}); refusing to risk a deep-recursion stack overflow",
            tokens.len()
        )));
    }
    if !tokens.is_empty() && !inner.is_valid(tokens) {
        return Err(PyValueError::new_err(
            "invalid or malformed prefix expression",
        ));
    }
    Ok(())
}

mod battery;
mod cancel;
mod convert;
pub mod engine;
mod eval;
mod fit;
mod hiprec;
mod interval;
mod matcher;
mod numeric;
mod operators;
mod parse;
mod rules;
mod sort;
mod tokens;
mod utils;
mod worker;

// The curated rlib surface (Cargo.toml: integration tests / examples link the core directly):
// the engine handle plus the types its pub methods take/return from otherwise-private modules.
pub use convert::Power;
pub use engine::Engine;
pub use worker::CandidateLibrary;

/// Test helper: load the `dev_7-3` engine, or `None` if its HF asset is not staged in the local
/// cache (e.g. a CI job or fresh checkout that did not download it). The asset-dependent parity
/// tests early-return on `None` (skip) rather than panic; the same kernel behaviour is covered end-
/// to-end by the Python `pytest` suite, which downloads the asset.
#[cfg(test)]
pub(crate) fn test_engine() -> Option<Engine> {
    let home = std::env::var("HOME").ok()?;
    let cfg = format!("{home}/.cache/simplipy/engines/dev_7-3/config.yaml");
    if !std::path::Path::new(&cfg).exists() {
        eprintln!("SKIP: dev_7-3 HF asset not staged in ~/.cache/simplipy");
        return None;
    }
    let rules = format!("{home}/.cache/simplipy/engines/dev_7-3/rules.json");
    Some(Engine::from_paths(&cfg, &rules).expect("engine loads"))
}

/// Opaque, compiled engine handle held on the Python side. Construction (parse config.yaml +
/// rules.json, build the bucket index + first-operand filter) happens ONCE; `simplify` is the
/// hot path.
#[pyclass(name = "Engine", module = "simplipy._core")]
struct PyEngine {
    inner: engine::Engine,
}

/// OFFLINE miner: a resident candidate library (built once per mine), passed back to
/// `find_rule_lib`. Opaque handle; all compute is Rust.
#[pyclass(name = "CandidateLibrary", module = "simplipy._core")]
struct PyCandidateLibrary {
    inner: worker::CandidateLibrary,
}

#[pymethods]
impl PyCandidateLibrary {
    /// Candidates kept in the library (post fold-filter).
    #[getter]
    fn n_candidates(&self) -> usize {
        self.inner.n_candidates()
    }

    /// Variable-free candidates dropped by the fold-filter (0 when the filter is off or inert).
    #[getter]
    fn n_filtered(&self) -> usize {
        self.inner.n_filtered()
    }
}

#[pymethods]
impl PyEngine {
    /// Build from already-resolved local asset paths (the Python shim resolves HF-hub/local paths
    /// via simplipy's own asset_manager and hands us the files, so asset resolution stays in ONE
    /// place and the Rust core stays network-free).
    #[staticmethod]
    fn from_paths(config_yaml_path: &str, rules_json_path: &str) -> PyResult<Self> {
        let inner = engine::Engine::from_paths(config_yaml_path, rules_json_path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build from in-memory config YAML + rules JSON text: the shim's direct
    /// `SimpliPyEngine(operators=..., rules=...)` construction serializes its state and
    /// attaches a core here, filesystem-free (there is no pure-Python fallback).
    #[staticmethod]
    fn from_strs(config_yaml: &str, rules_json: &str) -> PyResult<Self> {
        let inner = engine::Engine::from_strs(config_yaml, rules_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// THE hot path and the whole FFI unit. `tokens` is a prefix token list; returns the
    /// EQUIVALENCE-preserving simplified prefix token list. Defaults mirror the deployed call
    /// (`simplify(skeleton, inplace=True, max_pattern_length=None)`); `inplace` is a Python-shim
    /// concern (the shim mutates the caller's list), so it is NOT a kernel parameter here.
    ///
    /// Does NOT mask: masking (literals -> `<constant>`) is a representation step carved out into
    /// [`PyEngine::mask`] -- callers needing placeholders apply it to this output.
    #[pyo3(signature = (tokens, node_budget=48, max_pattern_length=None,
                        apply_simplification_rules=true, wildcard_all=false))]
    fn simplify(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        node_budget: usize,
        max_pattern_length: Option<usize>,
        apply_simplification_rules: bool,
        wildcard_all: bool,
    ) -> PyResult<Py<PyList>> {
        ensure_well_formed(&self.inner, &tokens)?;
        // Release the GIL for the pure-Rust kernel (parallel callers are not serialized on Python's lock).
        let out = py.detach(|| {
            self.inner.simplify(
                &tokens,
                node_budget,
                max_pattern_length,
                apply_simplification_rules,
                wildcard_all,
            )
        });
        Ok(PyList::new(py, out)?.into())
    }

    /// The REPRESENTATION pass: relabel numeric literals to `<constant>` + sort. Apply to
    /// `simplify`'s output when a downstream model needs placeholders; never re-`simplify` the
    /// result (see [`Engine::mask`]).
    fn mask(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        ensure_well_formed(&self.inner, &tokens)?;
        let out = py.detach(|| self.inner.mask(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Validation entry (NOT the shipped surface): the rule-application sub-unit only
    /// (`apply_simplification_rules`).
    #[pyo3(signature = (tokens, max_pattern_length=None))]
    fn apply_rules(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_pattern_length: Option<usize>,
    ) -> PyResult<Py<PyList>> {
        ensure_well_formed(&self.inner, &tokens)?;
        let out = py.detach(|| {
            self.inner
                .apply_simplification_rules(&tokens, max_pattern_length)
        });
        Ok(PyList::new(py, out)?.into())
    }

    /// Validation entry (NOT the shipped surface): the term-cancellation sub-unit only --
    /// `cancel_terms(*collect_multiplicities(tokens))`. `mpl`-independent.
    fn cancel_only(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        ensure_well_formed(&self.inner, &tokens)?;
        let out = py.detach(|| self.inner.cancel_terms(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Validation entry (NOT the shipped surface): the operand-sort sub-unit only --
    /// `sort_operands`.
    fn sort_only(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        ensure_well_formed(&self.inner, &tokens)?;
        let out = py.detach(|| self.inner.sort_operands(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// `is_valid`: is the prefix expression syntactically valid?
    /// Part of the drop-in-engine surface; the most-called simplipy method on the per-candidate
    /// inference path.
    fn is_valid(&self, py: Python<'_>, tokens: Vec<String>) -> bool {
        py.detach(|| self.inner.is_valid(&tokens))
    }

    /// `prefix_to_infix`. `power` in {'func','**'} (default 'func');
    /// `realization` toggles realization-name rendering. Raises `ValueError` on a malformed prefix
    /// (mirrors Python). Part of the drop-in-engine surface.
    #[pyo3(signature = (tokens, power="func", realization=false))]
    fn prefix_to_infix(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        power: &str,
        realization: bool,
    ) -> PyResult<String> {
        let power_mode = match power {
            "func" => convert::Power::Func,
            "**" => convert::Power::StarStar,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "power must be 'func' or '**', got {other:?}"
                )))
            }
        };
        py.detach(|| self.inner.prefix_to_infix(&tokens, power_mode, realization))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// `infix_to_prefix`. Part of the drop-in-engine surface.
    fn infix_to_prefix(&self, py: Python<'_>, infix_expression: &str) -> PyResult<Py<PyList>> {
        let out = py.detach(|| self.inner.infix_to_prefix(infix_expression));
        Ok(PyList::new(py, out)?.into())
    }

    /// `convert_expression`. Raises `ValueError` where Python raises
    /// (the exact exception kind differs; failure-parity, not message text).
    fn convert_expression(&self, py: Python<'_>, prefix_expr: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py
            .detach(|| self.inner.convert_expression(&prefix_expr))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// `parse`. `convert_expression`/`mask_numbers` match the Python
    /// defaults (True/False). Closes the `simplify(str)` + canonicalization path.
    #[pyo3(signature = (infix_expression, convert_expression=true, mask_numbers=false))]
    fn parse(
        &self,
        py: Python<'_>,
        infix_expression: &str,
        convert_expression: bool,
        mask_numbers: bool,
    ) -> PyResult<Py<PyList>> {
        let out = py
            .detach(|| {
                self.inner
                    .parse(infix_expression, convert_expression, mask_numbers)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// `operators_to_realizations`.
    fn operators_to_realizations(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
    ) -> PyResult<Py<PyList>> {
        let out = py.detach(|| self.inner.operators_to_realizations(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// `realizations_to_operators`.
    fn realizations_to_operators(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
    ) -> PyResult<Py<PyList>> {
        let out = py.detach(|| self.inner.realizations_to_operators(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Native numeric constant folding. Returns the result token, or `None` if
    /// the subtree cannot be folded (complex result / unparseable leaf / unknown operator) -- matching
    /// Python `_evaluate_constant_subtree`. Validation entry for the differential.
    fn evaluate_constant_subtree(&self, py: Python<'_>, tokens: Vec<String>) -> Option<String> {
        py.detach(|| self.inner.evaluate_constant_subtree(&tokens))
    }

    /// The CPython-exact `str(float)` formatter alone (the result-formatting half of numeric folding),
    /// exposed for the float-repr fuzz. Static; does not need engine state.
    #[staticmethod]
    fn py_float_repr(x: f64) -> String {
        crate::numeric::py_float_repr(x)
    }

    /// OFFLINE miner kernel: vectorized evaluation of a prefix expression over
    /// `n_rows` rows of row-major `x_flat` (shape `n_rows x len(var_names)`); `<constant>` slots bind
    /// to `params` left-to-right; numeric/special literals fold to their value. Returns the length-
    /// `n_rows` result column. NOT part of the inline (online) surface; replaces the Python
    /// realize->infix->codify->lambda->safe_f residual path inside `find_rule_worker`.
    fn evaluate_batch(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        params: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.inner
                .evaluate_batch(&tokens, &var_names, &x_flat, n_rows, &params)
        })
        .map_err(PyValueError::new_err)
    }

    /// `numpy.allclose(a, b, rtol, atol, equal_nan=True)` -- the miner's accept/reject decision gate.
    /// `b` is the asymmetric reference (second arg), matching the miner's call order. Static.
    #[staticmethod]
    #[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8))]
    fn allclose(a: Vec<f64>, b: Vec<f64>, rtol: f64, atol: f64) -> bool {
        crate::eval::allclose(&a, &b, rtol, atol)
    }

    /// OFFLINE miner: the no-constant equivalence test. The candidate
    /// must be constant-free; the source's constants are resampled over `challenges` rounds
    /// and every sign combination, requiring `allclose(source, candidate)` every time.
    /// `min_informative=None` resolves to `n_rows / 8`: certification requires that many
    /// SOURCE-FINITE evidence rows (accumulated across challenge instances), killing vacuous
    /// all-NaN/inf acceptance. Generic-equivalence semantics: source-finite rows bind; where
    /// the source is NaN/inf the replacement may EXTEND the domain (x/x -> 1).
    #[pyo3(signature = (source, candidate, var_names, x_flat, n_rows, challenges=16, rtol=1e-9, atol=1e-12, min_informative=None, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn equivalent_no_const(
        &self,
        py: Python<'_>,
        source: Vec<String>,
        candidate: Vec<String>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        challenges: usize,
        rtol: f64,
        atol: f64,
        min_informative: Option<usize>,
        seed: u64,
    ) -> PyResult<bool> {
        let mi = min_informative.unwrap_or((n_rows / 8).max(1));
        py.detach(|| {
            self.inner.equivalent_no_const_check(
                &source, &candidate, &var_names, &x_flat, n_rows, challenges, rtol, atol, mi, seed,
            )
        })
        .map_err(PyValueError::new_err)
    }

    /// OFFLINE: the `!`-sort certificate -- is the expression DEFINED AND FINITE a.e. over its
    /// variables (within the standalone horizon box)? Fail-closed: every undecided path is
    /// `false`. The same predicate the engine's `!` pattern matching runs behind its cache.
    fn interval_finite_ae(&self, tokens: Vec<String>) -> bool {
        crate::interval::finite_ae(&tokens, self.inner.operators_ref())
    }

    /// OFFLINE: EXACT value-class of an expression by interval analysis (`rust/interval.rs`) --
    /// the deterministic replacement for the sampled pole grid / classify probe. Returns one of
    /// FINITE / POSINF / NEGINF / NAN / MIXED / EMPTY, or ERROR if unevaluable.
    fn interval_class(&self, tokens: Vec<String>) -> String {
        match crate::interval::value_class(&tokens, self.inner.operators_ref()) {
            Some(c) => format!("{:?}", c).to_uppercase(),
            None => "ERROR".to_string(),
        }
    }

    /// OFFLINE: the raw positive-measure value COMPONENTS of an expression over free constants,
    /// as `(has_finite, pos_inf, neg_inf, nan)` -- the reachability gate's inputs, before `Class`
    /// collapses them. `None` if unevaluable.
    fn interval_value_components(&self, tokens: Vec<String>) -> Option<(bool, bool, bool, bool)> {
        crate::interval::value_set(
            &tokens,
            self.inner.operators_ref(),
            &crate::interval::Vs::reals(),
        )
        .map(|v| (v.has_fin, v.pinf, v.ninf, v.nan))
    }

    /// OFFLINE: how many gate calls fell OUTSIDE the box horizon and were therefore decided by
    /// default (= accepted). The gate is a dyadic witness search over a BOUNDED box; when an
    /// expression's own constants push the interesting region past the cap, it does not know. This
    /// counter is the exposure -- read it after a mine instead of assuming the number is zero.
    fn interval_horizon_misses(&self) -> u64 {
        crate::interval::HORIZON_MISSES.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// OFFLINE: how many subdivision searches exhausted their node budget. Exhaustion returns the
    /// incumbent witness, indistinguishable from a completed "no witness" -- the SECOND fail-open
    /// (the horizon cap is the first). Read after a mine, next to `interval_horizon_misses`.
    fn interval_node_budget_misses(&self) -> u64 {
        crate::interval::NODE_BUDGET_MISSES.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ONLINE observability: snapshot of the process-global simplify hot-path
    /// counters (calls / iterations / exact hits / pattern attempts+fires / cert calls+hits)
    /// plus coarse nanosecond accounting (cancel / rules / cert / mask+sort). Relaxed atomics,
    /// zero behavior change. Aggregates across engines and threads; pair with
    /// `reset_simplify_counters` around a batch to profile it.
    fn simplify_counters(&self) -> std::collections::HashMap<String, u64> {
        crate::engine::stats::snapshot()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    /// ONLINE observability: zero all `simplify_counters` counters.
    fn reset_simplify_counters(&self) {
        crate::engine::stats::reset();
    }

    /// OFFLINE / TEST HOOK: the value set of `tokens` over an explicit per-variable BOX --
    /// variable `i` ranges over `[los[i], his[i]]`. Returns
    /// `(has_finite, pos_inf, neg_inf, nan, fin_lo, fin_hi)`, or `None` if unevaluable.
    ///
    /// Exists so the box path can be validated against an external oracle. The property the whole
    /// domain gate rests on is that this OVER-approximates: every value the expression really takes
    /// on the box must be inside the returned set. Over-approximation is what makes
    /// "no defined value anywhere on this box" conservative, hence a witness always TRUE, hence
    /// the gate unable to reject a sound rule. If this ever under-reports, that proof is void.
    fn interval_value_set_box(
        &self,
        tokens: Vec<String>,
        los: Vec<f64>,
        his: Vec<f64>,
    ) -> Option<(bool, bool, bool, bool, f64, f64)> {
        if los.len() != his.len() || los.is_empty() {
            return None;
        }
        let doms: Vec<crate::interval::Vs> = los
            .iter()
            .zip(his.iter())
            .map(|(lo, hi)| crate::interval::Vs::interval(*lo, *hi, false, false))
            .collect();
        crate::interval::value_set_p(&tokens, self.inner.operators_ref(), &doms, &[])
            .map(|v| (v.has_fin, v.pinf, v.ninf, v.nan, v.lo, v.hi))
    }

    /// OFFLINE: witness WIDTH of a positive-measure region where `source` is NaN but `target`
    /// defines a value (0.0 = no domain extension). This is the exact domain-preservation
    /// gate: a rule must never be grossly domain dependent. `None` = UNDECIDED (horizon past
    /// R_MAX, or node budget exhausted with no witness) -- the gate fails closed on it; a
    /// consumer must treat `None` as "extension not ruled out", never as 0.0.
    fn interval_domain_extension(&self, source: Vec<String>, target: Vec<String>) -> Option<f64> {
        crate::interval::domain_extension(&source, &target, self.inner.operators_ref())
    }

    /// OFFLINE: `interval_domain_extension` with both sides' `<constant>` leaves BOUND to concrete
    /// values (k-th `<constant>` in prefix order -> `params[k]`). This is the form the miner uses:
    /// the gate runs per instance at the source's drawn constants and the constants the fit chose.
    /// `None` = UNDECIDED (fail-closed), as in `interval_domain_extension`.
    fn interval_domain_extension_p(
        &self,
        source: Vec<String>,
        src_params: Vec<f64>,
        target: Vec<String>,
        tgt_params: Vec<f64>,
    ) -> Option<f64> {
        crate::interval::domain_extension_p(
            &source,
            &src_params,
            &target,
            &tgt_params,
            self.inner.operators_ref(),
        )
    }

    /// OFFLINE miner: the wildcard-multiplicity rule guard.
    #[staticmethod]
    fn violates_wildcard_multiplicity(lhs: Vec<String>, rhs: Vec<String>) -> bool {
        crate::worker::violates_wildcard_multiplicity(&lhs, &rhs)
    }

    /// OFFLINE miner: the full native `find_rule_worker` decision for one source --
    /// short-circuit + candidate scan (no-constant test / constant fit) + selection. Returns
    /// the chosen target token list, or None. `candidates` = the candidate library (expressions
    /// up to max_target); for the resident path see `build_candidate_library`/`find_rule_lib`.
    #[pyo3(signature = (source, simplified_length, max_target, candidates, var_names, x_flat, n_rows, challenges=16, retries=16, seed=0, rtol=1e-9, atol=1e-12, min_informative=None, fold_filter=true))]
    #[allow(clippy::too_many_arguments)]
    fn find_rule(
        &self,
        py: Python<'_>,
        source: Vec<String>,
        simplified_length: usize,
        max_target: Option<usize>,
        candidates: Vec<Vec<String>>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        challenges: usize,
        retries: usize,
        seed: u64,
        rtol: f64,
        atol: f64,
        min_informative: Option<usize>,
        fold_filter: bool,
    ) -> PyResult<Option<Vec<String>>> {
        let mi = min_informative.unwrap_or((n_rows / 8).max(1));
        py.detach(|| {
            self.inner.find_rule(
                &source,
                simplified_length,
                max_target,
                &candidates,
                &var_names,
                &x_flat,
                n_rows,
                challenges,
                retries,
                seed,
                rtol,
                atol,
                mi,
                fold_filter,
            )
        })
        .map_err(PyValueError::new_err)
    }

    /// OFFLINE miner: build a RESIDENT candidate library once per mine (precompiles every
    /// candidate's tape + precomputes const-free `y`). Pass the returned handle to
    /// `find_rule_lib`. `fold_filter` (default on) drops var-free candidates of length >= 2 --
    /// the sound "candidate minimization" lever; see `worker::CandidateLibrary::build`.
    #[pyo3(signature = (candidates, var_names, x_flat, n_rows, fold_filter=true))]
    fn build_candidate_library(
        &self,
        py: Python<'_>,
        candidates: Vec<Vec<String>>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        fold_filter: bool,
    ) -> PyResult<PyCandidateLibrary> {
        let inner = py
            .detach(|| {
                self.inner.build_candidate_library(
                    &candidates,
                    &var_names,
                    &x_flat,
                    n_rows,
                    fold_filter,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyCandidateLibrary { inner })
    }

    /// OFFLINE miner: `find_rule_worker` decision over a resident `CandidateLibrary`
    /// (no per-source rebuild, no per-call X marshaling). Returns the chosen target, or None.
    #[pyo3(signature = (source, simplified_length, max_target, library, challenges=16, retries=16, seed=0, rtol=1e-9, atol=1e-12, min_informative=None))]
    #[allow(clippy::too_many_arguments)]
    fn find_rule_lib(
        &self,
        py: Python<'_>,
        source: Vec<String>,
        simplified_length: usize,
        max_target: Option<usize>,
        library: PyRef<'_, PyCandidateLibrary>,
        challenges: usize,
        retries: usize,
        seed: u64,
        rtol: f64,
        atol: f64,
        min_informative: Option<usize>,
    ) -> PyResult<Option<Vec<String>>> {
        let lib = &library.inner;
        let mi = min_informative.unwrap_or((lib.n_rows() / 8).max(1));
        py.detach(|| {
            self.inner.find_rule_with_lib(
                &source,
                simplified_length,
                max_target,
                lib,
                challenges,
                retries,
                seed,
                rtol,
                atol,
                mi,
            )
        })
        .map_err(PyValueError::new_err)
    }

    /// OFFLINE (mine driver): replace the engine's rules (recompile) -- grows the Kruskal-prune
    /// rule set length-by-length during a mine. `rules` = the canonicalized (wildcard) rule list.
    fn set_rules(&mut self, rules: Vec<(Vec<String>, Vec<String>)>) {
        self.inner.set_rules(rules);
    }

    /// OFFLINE (mine driver): mine ONE source-length IN PARALLEL (rayon, all cores) -- Kruskal-
    /// prune each source with the current rules, then `find_rule` on survivors. Returns the found
    /// (source -> target) rules. The Python driver loops lengths, dedups/canonicalizes, and `set_rules`
    /// between them (the order-dependent barrier). GIL released for the parallel work.
    /// `relaxed_kruskal=True` (the default) still searches sources the current rules already
    /// shorten, with the bound tightened to the simplified length (targets must beat what
    /// `simplify` already reaches); `False` skips them (strict Kruskal). See
    /// `Engine::mine_one_length`.
    #[pyo3(signature = (sources, library, max_target, challenges=16, retries=16, seed=0, rtol=1e-9, atol=1e-12, min_informative=None, relaxed_kruskal=true))]
    #[allow(clippy::too_many_arguments)]
    fn mine_one_length(
        &self,
        py: Python<'_>,
        sources: Vec<Vec<String>>,
        library: PyRef<'_, PyCandidateLibrary>,
        max_target: Option<usize>,
        challenges: usize,
        retries: usize,
        seed: u64,
        rtol: f64,
        atol: f64,
        min_informative: Option<usize>,
        relaxed_kruskal: bool,
    ) -> Vec<(Vec<String>, Vec<String>)> {
        let lib = &library.inner;
        let mi = min_informative.unwrap_or((lib.n_rows() / 8).max(1));
        py.detach(|| {
            self.inner.mine_one_length(
                &sources,
                lib,
                max_target,
                challenges,
                retries,
                seed,
                rtol,
                atol,
                mi,
                relaxed_kruskal,
            )
        })
    }

    /// OFFLINE: prune redundant explicit rules with the Rust core. Tests each explicit `lhs`
    /// (in the given asset order) by removing it from the Rust rule map + re-simplifying;
    /// returns the pruned `lhs` list. Must run against the compiled rules `simplify` actually
    /// uses (pruning a divergent rule store over-prunes). Takes `&mut self`.
    #[pyo3(signature = (ordered_lhs, mask_elementary_literals=false))]
    fn prune_explicit(
        &mut self,
        ordered_lhs: Vec<Vec<String>>,
        mask_elementary_literals: bool,
    ) -> Vec<Vec<String>> {
        self.inner
            .prune_explicit(&ordered_lhs, mask_elementary_literals)
    }

    /// OFFLINE miner: native `exist_constants_that_fit` for AFFINE-in-params candidates -- a
    /// closed-form least-squares solve + the `allclose` decision gate (no optimizer,
    /// deterministic). Returns `Some(decision)` for affine candidates, `None` for
    /// nonlinear-in-params ones (the native-LM path). Same accept/reject gate as scipy's path.
    #[pyo3(signature = (candidate, var_names, x_flat, n_rows, y_target, rtol=1e-5, atol=1e-8))]
    fn exist_constants_fit_linear(
        &self,
        py: Python<'_>,
        candidate: Vec<String>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        y_target: Vec<f64>,
        rtol: f64,
        atol: f64,
    ) -> PyResult<Option<bool>> {
        py.detach(|| {
            self.inner.exist_constants_fit_linear(
                &candidate, &var_names, &x_flat, n_rows, &y_target, rtol, atol,
            )
        })
        .map_err(PyValueError::new_err)
    }

    /// OFFLINE miner: native `exist_constants_that_fit`. Affine candidates ->
    /// closed-form (deterministic); nonlinear-in-params -> `n_restarts` LM solves from random N(0,5)
    /// starts (seeded). Accept iff any makes `allclose(y_target, fitted)` pass -- scipy's exact gate.
    #[pyo3(signature = (candidate, var_names, x_flat, n_rows, y_target, rtol=1e-5, atol=1e-8, n_restarts=16, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn exist_constants_fit(
        &self,
        py: Python<'_>,
        candidate: Vec<String>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        y_target: Vec<f64>,
        rtol: f64,
        atol: f64,
        n_restarts: usize,
        seed: u64,
    ) -> PyResult<bool> {
        py.detach(|| {
            self.inner.exist_constants_fit(
                &candidate, &var_names, &x_flat, n_rows, &y_target, rtol, atol, n_restarts, seed,
            )
        })
        .map_err(PyValueError::new_err)
    }

    #[getter]
    fn engine_id(&self) -> &str {
        self.inner.engine_id()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyCandidateLibrary>()?;
    m.add("__build__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
