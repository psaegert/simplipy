//! The Rust inline backend for SimpliPy, exposed to Python as `simplipy._core` via PyO3.
//!
//! ## FFI design (load-bearing, per the verified analysis)
//! The ENTIRE simplify fixpoint is ONE FFI unit. A single call into Rust receives the
//! prefix token list and returns the simplified token list; the whole rewrite loop
//! (parse -> canonical constructors -> rule matching -> projection) runs with NO boundary
//! crossings. FFI marshalling stays <1% of wall time; crossing the boundary per matcher
//! call instead would evaporate the speedup. Therefore the PyO3 layer here is deliberately
//! THIN: marshal `list[str]` <-> `Vec<String>`, hold the compiled engine, and delegate the
//! whole unit to the engine.
//!
//! ## One engine line
//! The core implements the contract semantics as a single engine line: numeric constant
//! folding (incl. `1/0 -> float("inf")`, via the `numeric` module's f64 + libm evaluator),
//! the corrected conversions, and atomic inf/nan tokens. Byte-exact reproduction of the
//! historical dev_7-3 / v23.0-era behavior is served by installing `simplipy<=0.6.0`.
use pyo3::exceptions::PyValueError;

// See the `mimalloc` entry in Cargo.toml: glibc malloc grows RSS steadily under the miner's
// 28-thread sustained churn, and the glibc knobs cost 11.5x throughput at that thread count.
#[cfg(feature = "mimalloc-allocator")]
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Reject inputs the recursive kernel cannot safely handle, with a clean `ValueError` BEFORE it runs.
/// Two off-distribution hazards, both of which would otherwise ABORT the interpreter (uncatchable):
///   1. PATHOLOGICALLY LONG input -> the tree recursions (parse / canonicalize / serialize) recurse
///      ~one frame per nesting level and overflow the default stack. Real expressions are tiny
///      (<~100 tokens); cap generously. (Python raises RecursionError in the same regime.)
///   2. MALFORMED prefix (an operator with too few operands) -> the parser would
///      index-underflow-PANIC, and a panic across the GIL-released `detach` boundary can abort.
///      `is_valid` is the (240k-differential-verified) arity check, false exactly on those shapes.
///      Empty input is the one valid case `is_valid` rejects (`simplify([]) == []`), allowed through.
///
/// Both checks are O(n) and only fire off the deployed distribution; valid inputs are unaffected. We
/// deliberately do NOT replicate Python's path-dependent exception TYPE -- one clean `ValueError`.
pub(crate) const MAX_TOKENS: usize = 4096;

/// `interval_value_set_box`'s row: `(has_finite, pos_inf, neg_inf, nan, fin_lo, fin_hi)`.
type ValueSetBox = (bool, bool, bool, bool, f64, f64);

/// The AC entry accepts BOTH the old grammar and the tagged form; `is_valid` only understands
/// the former, so tag-bearing inputs skip the arity scan (the shared AC parser is the
/// authority there -- a malformed tagged input fails the parse, the kernel returns `None`,
/// and the FFI raises the same ValueError as every other malformed input; audit Tier-2,
/// 2026-08-03: it used to be returned UNCHANGED, silently). The length cap applies to both.
/// The WIRE SPELLING of a rule mode (`ac_simplify_in_mode` and its infix twin), refused
/// loudly on anything else: a mistyped mode must never silently select a different rule
/// set. Same doctrine as `simplify(mode=...)`'s string handling in the Python shim -- a
/// stray string that resolved to a rung by accident is exactly the bug that made
/// `mode='lossy'` run SOUND.
fn parse_rule_mode(name: &str) -> PyResult<engine::RuleMode> {
    engine::RuleMode::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown rule_mode {name:?}: expected 'default', 'real' or 'corpus'"
        ))
    })
}

/// The output projection's wire spelling, factored out of the three FFI entries that
/// accept it so they cannot drift.
fn parse_ac_form(name: &str) -> PyResult<engine::AcForm> {
    match name {
        "tagged" => Ok(engine::AcForm::Tagged),
        "explicit" => Ok(engine::AcForm::Explicit),
        other => Err(PyValueError::new_err(format!(
            "unknown form {other:?}: expected 'tagged' or 'explicit'"
        ))),
    }
}

/// THE ONE simplify implementation behind the FFI. `ac_simplify` (the `wildcard_all`
/// spelling that shipped) and `ac_simplify_in_mode` (the mode spelling) both land here,
/// so the bool is a SPELLING of a mode and never a second mechanism beside it.
fn ac_simplify_impl(
    inner: &engine::Engine,
    py: Python<'_>,
    tokens: Vec<String>,
    max_passes: usize,
    mode: engine::RuleMode,
    form: engine::AcForm,
) -> PyResult<Py<PyList>> {
    // The documented empty-input contract: `simplify([]) == []` (the one valid
    // case `is_valid` rejects). Restored explicitly after the malformed-input
    // hardening accidentally made it raise (hardening H-003, 2026-08-03).
    if tokens.is_empty() {
        return Ok(PyList::empty(py).into());
    }
    ensure_ac_well_formed(inner, &tokens)?;
    let out = py
        .detach(|| inner.ac_simplify_proj(&tokens, max_passes, mode, form))
        .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))?;
    Ok(PyList::new(py, out)?.into())
}

/// The infix twin of [`ac_simplify_impl`], shared by `ac_simplify_infix` and
/// `ac_simplify_infix_in_mode`.
fn ac_simplify_infix_impl(
    inner: &engine::Engine,
    py: Python<'_>,
    tokens: Vec<String>,
    max_passes: usize,
    mode: engine::RuleMode,
) -> PyResult<String> {
    // Empty-input contract, as in `ac_simplify` (H-003): the empty rendering.
    if tokens.is_empty() {
        return Ok(String::new());
    }
    ensure_ac_well_formed(inner, &tokens)?;
    py.detach(|| inner.ac_simplify_infix(&tokens, max_passes, mode))
        .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))
}

fn ensure_ac_well_formed(inner: &engine::Engine, tokens: &[String]) -> PyResult<()> {
    ensure_tokens_are_tokens(tokens)?;
    // Tagged-form tokens and the `rootn` built-in live outside `is_valid`'s config
    // vocabulary; the AC parser is the arbiter for such inputs -- a malformed sequence
    // fails the parse and RAISES at the FFI (kernel `None`). Legacy hyper-operator
    // spellings have NO config-independent reading (strictly clean): they parse only
    // under a legacy CONFIG that still declares them, and legacy corpora are CONVERTED
    // once at the boundary instead.
    if tokens
        .iter()
        .any(|t| t == "<add>" || t == "<mul>" || t == "rootn")
    {
        return Ok(()); // capped + alphabet-checked above; the AC parser is the arbiter
    }
    ensure_well_formed(inner, tokens)
}

/// THE NORMATIVE TOKEN GRAMMAR (H-004 + H-007), enforced at every semantic boundary.
/// A well-formed token (non-empty, whitespace-free -- an empty or whitespace-bearing
/// "token" corrupts every space-joined serialization; H-004, 2026-08-03) is one of:
///   1. a NUMERIC LITERAL -- the canonical spellings every layer reads identically:
///      `Rat::parse_decimal` forms (optional single sign, decimal digits, one optional
///      `.`, optional `e`/`E` exponent with optional sign), the exact fraction `p/q`,
///      the special spellings `np.pi`/`np.e`/`float("inf")`/`float("-inf")`/
///      `float("nan")`, and their parenthesized forms;
///   2. an OPERATOR of the loaded config (or a tagged-form structural token);
///   3. a FREE SYMBOL -- anything else that NO standard numeric reader interprets
///      (`x0`, `foo`, `_0`); symbol algebra applies, under the contract that a free
///      symbol denotes a finite real;
///   4. a RESERVED NUMERIC SPELLING -- REFUSED here (H-007, 2026-08-04): a token that
///      Python `float()`/literal syntax or Rust `f64` reads as a value but the symbolic
///      core would treat as a free symbol (`inf`, `1_000`, `0x10`; see
///      `utils::reserved_numeric_spelling`). Before this guard the SAME input had two
///      contradictory public answers: `simplify(["-","inf","inf"]) == ["0"]` (symbol
///      algebra) while `evaluate_constant_subtree` said `float("nan")` (value reading).
///
/// `is_numeric_string` remains a deliberately over-approximating MASKING heuristic
/// (its excess, e.g. `--5`, is refused by `is_valid`), and `is_valid` remains the pure
/// arity oracle (its 240k-differential pin is untouched); alphabet well-formedness and
/// spelling reservation are the boundary's job, enforced here.
fn ensure_tokens_are_tokens(tokens: &[String]) -> PyResult<()> {
    // RECURSION CAP AT THE CHOKE POINT (H-043, D4): the recursive walkers behind the
    // instrument surfaces (tape compile, constant-subtree eval, interval descent, rule
    // compile) overflow the stack on ~1e5-deep chains, which ABORTS the process --
    // pyo3 maps panics, not SIGSEGV. The cap lives HERE so every entry that checks its
    // alphabet is depth-bounded by construction; before D4 it lived only on the
    // simplify/AC surfaces and twelve other entries aborted (probed, exit -11).
    if tokens.len() > MAX_TOKENS {
        return Err(PyValueError::new_err(format!(
            "prefix expression too long ({} tokens > {MAX_TOKENS}); refusing to risk a \
             deep-recursion stack overflow",
            tokens.len()
        )));
    }
    for t in tokens {
        if t.is_empty() || t.contains(char::is_whitespace) {
            return Err(PyValueError::new_err(format!(
                "invalid token {t:?}: tokens must be non-empty and whitespace-free"
            )));
        }
        if crate::utils::reserved_numeric_spelling(t) {
            return Err(PyValueError::new_err(format!(
                "invalid token {t:?}: reserved numeric spelling -- numeric to Python \
                 (float()/literal syntax) but not a simplipy numeric literal, so the engine \
                 would apply symbol algebra to a value-bearing token (H-007). Use the \
                 canonical spelling instead: decimal/exponent literals ('5', '0.5', '1e-05'), \
                 exact fractions ('1/3'), or the non-finite tokens 'float(\"inf\")', \
                 'float(\"-inf\")', 'float(\"nan\")'."
            )));
        }
    }
    Ok(())
}

fn ensure_well_formed(inner: &engine::Engine, tokens: &[String]) -> PyResult<()> {
    ensure_tokens_are_tokens(tokens)?; // alphabet + the MAX_TOKENS recursion cap
    if !tokens.is_empty() && !inner.is_valid(tokens) {
        return Err(PyValueError::new_err(
            "invalid or malformed prefix expression",
        ));
    }
    Ok(())
}

mod ac;
mod battery;

mod convert;
pub mod engine;
mod eval;
mod fit;
mod forms;
mod hiprec;
mod interval;
mod numeric;
mod operators;
mod rules;
mod tokens;
mod utils;
mod worker;

// The curated rlib surface (Cargo.toml: integration tests / examples link the core directly):
// the engine handle plus the types its pub methods take/return from otherwise-private modules.
pub use convert::Power;
pub use engine::Engine;
pub use worker::CandidateLibrary;

/// Test-suite thinning guard: when `SIMPLIPY_TEST_REQUIRE_ASSETS` is set (any value), an
/// asset-gated test that would silently skip must PANIC instead. Roughly half the Rust suite
/// is gated on the acj-4-3 asset; without this mode a fresh machine (or a CI job whose asset
/// staging broke) reports every one of those tests green while running none of them --
/// `cargo test` has no skip verdict, so the only honest failure mode is a loud one. CI sets
/// the variable right after staging the asset; local verification runs should do the same.
#[cfg(test)]
pub(crate) fn assets_required() -> bool {
    std::env::var_os("SIMPLIPY_TEST_REQUIRE_ASSETS").is_some()
}

/// Test helper: load the `acj-4-3` engine (the GENERATION-2 test universe: the clean
/// 23-operator vocabulary the engine actually serves), or `None` if the HF asset is not staged
/// in the local cache (e.g. a CI job or fresh checkout that did not download it). The
/// asset-dependent tests early-return on `None` (skip) rather than panic; under
/// `SIMPLIPY_TEST_REQUIRE_ASSETS` the skip path panics instead (see [`assets_required`]).
///
/// HISTORY (audit Tier-1 #3, owner ruling 2026-08-03): this used to load the legacy `dev_7-3`
/// asset, which made the RETIRED hyper-operator vocabulary the universe every rust test ran in
/// -- exactly why the binary-vocabulary certificate holes (division towers, 2026-08-03) stayed
/// invisible to the suite. Tests that legitimately need a legacy TABLE (the conversion layer's
/// input language) use the in-repo fixture instead (`ac::convert::tests`), never a cached asset.
#[cfg(test)]
pub(crate) fn test_engine() -> Option<Engine> {
    let skip = |why: &str| {
        assert!(
            !assets_required(),
            "SIMPLIPY_TEST_REQUIRE_ASSETS is set but {why}"
        );
        eprintln!("SKIP: {why}");
    };
    let Ok(home) = std::env::var("HOME") else {
        skip("HOME is unset, so the acj-4-3 asset cache cannot be located");
        return None;
    };
    let cfg = format!("{home}/.cache/simplipy/engines/acj-4-3/config.yaml");
    if !std::path::Path::new(&cfg).exists() {
        skip("acj-4-3 HF asset not staged in ~/.cache/simplipy");
        return None;
    }
    let rules = format!("{home}/.cache/simplipy/engines/acj-4-3/rules.json");
    Some(Engine::from_paths(&cfg, &rules).expect("engine loads"))
}

/// Test helper: the LEGACY-VOCABULARY table (the dev_7-3 operator set, in-repo fixture,
/// no rules, no asset). The retired hyper-operator spellings are a supported INPUT
/// language for tables that declare them; the conversion-layer tests exercise exactly
/// that and run here. Always available (no skip): the fixture ships with the repo.
#[cfg(test)]
pub(crate) fn legacy_sugar_engine() -> Engine {
    let cfg = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/legacy_vocab_config.yaml"
    );
    let yaml = std::fs::read_to_string(cfg).expect("legacy fixture present in-repo");
    Engine::from_strs(&yaml, "[]").expect("legacy fixture engine builds")
}

/// Opaque, compiled engine handle held on the Python side. Construction (parse config.yaml +
/// rules.json, intern the token table; the AC rule translation is lazy) happens ONCE;
/// `simplify` is the hot path.
#[pyclass(name = "Engine", module = "simplipy._core")]
struct PyEngine {
    inner: engine::Engine,
    /// C1.9: a digest of the rules AS PUSHED (from_strs' initial set or the last
    /// `set_rules`), echoed back through [`PyEngine::rules_in_sync`] so the Python
    /// shim's `__getstate__` can refuse to pickle an engine whose rule list and
    /// compiled core have silently diverged. The digest is over the PUSHED
    /// representation (the raw (lhs, rhs) token lists), deliberately not the core's
    /// translated/registry-filtered internal set -- the question is "does the recipe
    /// match what was pushed", not "what did the core keep".
    pushed_rules_digest: u64,
}

/// C1.9: deterministic digest of a pushed rule list (std SipHash with fixed keys --
/// `DefaultHasher::new()` -- stable across processes of one build; both storers and
/// the checker share this one function, so representation drift is impossible).
/// One SERVED rule as the FFI hands it out: `(lhs, rhs, artifact index)`.
type ServedRule = (Vec<String>, Vec<String>, usize);

fn pushed_rules_digest(rules: &[(Vec<String>, Vec<String>)]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    rules.hash(&mut h);
    h.finish()
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
    /// Build from in-memory config YAML + rules JSON text: the shim's direct
    /// `SimpliPyEngine(operators=..., rules=...)` construction serializes its state and
    /// attaches a core here, filesystem-free (there is no pure-Python fallback).
    #[staticmethod]
    fn from_strs(config_yaml: &str, rules_json: &str) -> PyResult<Self> {
        // threatmodel-3: parse and CAP the rules BEFORE the engine build -- rule
        // compile walks each side recursively, and this was the one token-taking
        // FFI entry the H-043 cap never reached: a ~1e5-deep chain in a hostile or
        // corrupt rules.json aborted the whole process (SIGSEGV, not a Python
        // exception) at plain `load()`. Same cap, same choke-point doctrine.
        let rules: Vec<(Vec<String>, Vec<String>)> = serde_json::from_str(rules_json)
            .map_err(|e| PyValueError::new_err(format!("rules JSON: {e}")))?;
        for (lhs, rhs) in &rules {
            ensure_tokens_are_tokens(lhs)?;
            ensure_tokens_are_tokens(rhs)?;
        }
        let inner = engine::Engine::from_strs(config_yaml, rules_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // C1.9: digest the initial rules exactly as `set_rules` digests later pushes --
        // the same (lhs, rhs) list representation the shim serialized into the JSON.
        Ok(Self {
            inner,
            pushed_rules_digest: pushed_rules_digest(&rules),
        })
    }

    /// Simplify through the AC CORE: n-ary add/mul bags with exact rational coefficients,
    /// rules widened to sub-multiset matching, coefficient arithmetic as computation.
    /// `form` selects the output projection: "tagged" (default -- the strict delimited
    /// prefix form, `<add> ... </add>`), "explicit" (sugared old-token diagnostic form),
    /// Input accepts both the old grammar and
    /// the tagged form. `max_passes` bounds the number of outer REWRITE PASSES (not nodes);
    /// the chain stops early at its fixpoint, which measures at 2-4 passes against this
    /// default of 48. `wildcard_all` is the LOSSY switch, exactly as in `simplify`.
    #[pyo3(signature = (tokens, max_passes=48, wildcard_all=false, form="tagged"))]
    fn ac_simplify(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
        wildcard_all: bool,
        form: &str,
    ) -> PyResult<Py<PyList>> {
        // THE MAP, and the only one on this boundary: today's two-valued `simplipy.Mode`
        // becomes a three-valued rule mode here. SOUND -> the default set, LOSSY -> the
        // corpus set. Renaming `Mode` to `f64`/`real`/`corpus` retires this entry in
        // favour of `ac_simplify_in_mode` below; nothing under it has to move.
        ac_simplify_impl(
            &self.inner,
            py,
            tokens,
            max_passes,
            engine::RuleMode::from_wildcard_all(wildcard_all),
            parse_ac_form(form)?,
        )
    }

    /// `ac_simplify` addressing the rule mode DIRECTLY -- `"default"` (`rules.json`),
    /// `"real"` (`rules_real.json`) or `"corpus"` (`rules_corpus.json`). Each mode serves
    /// ONE DISTINCT, COMPLETE set; `"default"`/`"corpus"` are exactly what
    /// `wildcard_all=False`/`True` select above, and `"real"` is the rung today's
    /// `simplipy.Mode` cannot yet name.
    ///
    /// Contracts (empty input, malformed input, forms) are the SAME code as `ac_simplify`
    /// -- both entries are one call into `ac_simplify_impl`.
    #[pyo3(signature = (tokens, max_passes=48, rule_mode="default", form="tagged"))]
    fn ac_simplify_in_mode(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
        rule_mode: &str,
        form: &str,
    ) -> PyResult<Py<PyList>> {
        ac_simplify_impl(
            &self.inner,
            py,
            tokens,
            max_passes,
            parse_rule_mode(rule_mode)?,
            parse_ac_form(form)?,
        )
    }

    /// D39 B1 -- `ac_simplify` with the OPT-IN post-fixpoint exploration phase (ledger
    /// D39): the deterministic chain runs unchanged to its fixpoint, then a budgeted
    /// exploration phase proposes expansion moves through the same certified machinery
    /// and accepts only endpoints STRICTLY below in the serve ordering
    /// (`ac_ordered_below`'s predicate -- measure-agnostic scaffolding, roadmap B1).
    /// `explore_budget` counts candidate descents; 0 (the default) never enters the
    /// phase, so this entry is then byte-identical to `ac_simplify` (the ledger's
    /// effort=0 semantics). The public `effort=` API is deliberately NOT wired here
    /// (roadmap B7); this is the scaffolding's FFI boundary, contracts (empty input,
    /// malformed input, forms) exactly as `ac_simplify`.
    #[pyo3(signature = (tokens, max_passes=48, wildcard_all=false, form="tagged", explore_budget=0))]
    fn ac_simplify_explore(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
        wildcard_all: bool,
        form: &str,
        explore_budget: usize,
    ) -> PyResult<Py<PyList>> {
        let form = parse_ac_form(form)?;
        if tokens.is_empty() {
            return Ok(PyList::empty(py).into());
        }
        ensure_ac_well_formed(&self.inner, &tokens)?;
        let out = py
            .detach(|| {
                self.inner.ac_explore_proj(
                    &tokens,
                    max_passes,
                    engine::RuleMode::from_wildcard_all(wildcard_all),
                    form,
                    explore_budget,
                )
            })
            .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))?;
        Ok(PyList::new(py, out)?.into())
    }

    /// The PRETTY INFIX rendering of the AC-simplified expression:
    /// `x8 + 1.2*x3`, `-x0/3`, `(x0 + 1)^2`, `sin(x0)`. Round-trips through `parse`
    /// (the reserved constant names `pi`/`e`/`inf`/`nan` read back as constants).
    #[pyo3(signature = (tokens, max_passes=48, wildcard_all=false))]
    fn ac_simplify_infix(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
        wildcard_all: bool,
    ) -> PyResult<String> {
        ac_simplify_infix_impl(
            &self.inner,
            py,
            tokens,
            max_passes,
            engine::RuleMode::from_wildcard_all(wildcard_all),
        )
    }

    /// `ac_simplify_infix` addressing the rule mode directly (see `ac_simplify_in_mode`).
    #[pyo3(signature = (tokens, max_passes=48, rule_mode="default"))]
    fn ac_simplify_infix_in_mode(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
        rule_mode: &str,
    ) -> PyResult<String> {
        ac_simplify_infix_impl(
            &self.inner,
            py,
            tokens,
            max_passes,
            parse_rule_mode(rule_mode)?,
        )
    }

    /// Semantic complexity of an expression, measured on its CANONICAL form (the functional
    /// the AC simplify search minimizes): weighted structural count -- atoms 1, k-member bags
    /// k-1, signs free, coefficients +1, powers/functions 1 + operands.
    /// SERVE-TIME ORDERING (internal, for the mining acceptance gate): is `a` strictly
    /// below `b` in the reduction ordering the rewrite pass fires under?
    fn ac_ordered_below(&self, py: Python<'_>, a: Vec<String>, b: Vec<String>) -> PyResult<bool> {
        ensure_tokens_are_tokens(&a)?;
        ensure_tokens_are_tokens(&b)?;
        py.detach(|| self.inner.ac_ordered_below(&a, &b))
            .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))
    }

    /// MINING JUDGE (internal): one fused call -- (canonical input complexity, reduced
    /// complexity, reduced expression in the explicit projection). The AC parser is the
    /// arbiter; only the length cap guards recursion. The scores are reference values:
    /// coverage decisions judge in the serve ordering via `ac_ordered_below`, never by
    /// comparing them.
    fn ac_judge(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
    ) -> PyResult<(u64, u64, Vec<String>)> {
        ensure_tokens_are_tokens(&tokens)?;
        py.detach(|| self.inner.ac_judge(&tokens, max_passes))
            .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))
    }

    /// THE DEDUP KEY (internal): the BARE-CONTEXT canonical serialization of each
    /// expression -- exactly the internal form the rule loader (`AcRules::translate`)
    /// builds for a rule side, in the engine's native tagged projection. Batched: a
    /// ruleset is keyed side-by-side and one shared context serves the whole list.
    /// `None` for an unparseable side, so the caller can fall back to the spelling
    /// instead of failing a whole ruleset on one malformed entry.
    ///
    /// NOT interchangeable with `to_prefix`/`to_tagged`: those run the rewrite pass under
    /// the LOADED RULESET, which on acj-4-3 turns `* (-1) asin _0` into `asin neg _0` --
    /// that rule's own RHS. See `Engine::ac_canonical_key`.
    fn ac_canonical_keys(
        &self,
        py: Python<'_>,
        exprs: Vec<Vec<String>>,
    ) -> PyResult<Vec<Option<Vec<String>>>> {
        for e in &exprs {
            ensure_tokens_are_tokens(e)?;
        }
        Ok(py.detach(|| self.inner.ac_canonical_keys(&exprs)))
    }

    fn ac_complexity(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<u64> {
        ensure_ac_well_formed(&self.inner, &tokens)?;
        py.detach(|| self.inner.ac_complexity(&tokens))
            .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))
    }

    /// Certified-canon complexity (the serve ordering's own pricing; see
    /// `engine::ac::ac_complexity_certified`): `mu(simplify(e)) <= mu(e)` is a
    /// theorem under this pricing, unlike the bare `ac_complexity`.
    fn ac_complexity_certified(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<u64> {
        ensure_ac_well_formed(&self.inner, &tokens)?;
        py.detach(|| self.inner.ac_complexity_certified(&tokens))
            .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))
    }

    /// AC translation audit: (rules kept incl. minted orientation twins, rules subsumed by
    /// exact arithmetic, rules dropped as untranslatable/degenerate, orientation twins
    /// minted). Forces the lazy translation.
    fn ac_rules_info(&self, py: Python<'_>) -> PyResult<(usize, usize, usize, usize)> {
        Ok(py.detach(|| self.inner.ac_rules_info()))
    }

    /// The SERVED rule set -- asset rules AND load-minted orientation twins -- as
    /// `(lhs, rhs, source)` triples in the explicit projection, where `source` indexes the
    /// artifact ruleset. The audit surface for "judge what the engine serves, not what the
    /// artifact says": the twins exist only after translation, so a sweep that reads
    /// `rules.json` never sees them. Forces the lazy translation.
    fn ac_served_rules(&self, py: Python<'_>) -> PyResult<Vec<ServedRule>> {
        Ok(py.detach(|| self.inner.ac_served_rules()))
    }

    /// The four translation counts for ANY mode's served set (`"default"` / `"real"` /
    /// `"corpus"`). `ac_rules_info` stays the DEFAULT mode's tuple, unwidened -- it is
    /// pinned by the corpus gate, and answering it must never force another mode's index
    /// to be built. Forces the named mode's lazy translation, and only that one.
    #[pyo3(signature = (rule_mode="default"))]
    fn ac_rules_info_in_mode(
        &self,
        py: Python<'_>,
        rule_mode: &str,
    ) -> PyResult<(usize, usize, usize, usize)> {
        let mode = parse_rule_mode(rule_mode)?;
        Ok(py.detach(|| self.inner.ac_rules_info_in_mode(mode)))
    }

    /// [`PyEngine::ac_served_rules`] for ANY mode: what that mode actually fires, with
    /// `source` indexing THAT MODE'S OWN file. The judge surface for the triple.
    #[pyo3(signature = (rule_mode="default"))]
    fn ac_served_rules_in_mode(
        &self,
        py: Python<'_>,
        rule_mode: &str,
    ) -> PyResult<Vec<ServedRule>> {
        let mode = parse_rule_mode(rule_mode)?;
        Ok(py.detach(|| self.inner.ac_served_rules_in_mode(mode)))
    }

    /// How many rules this mode's OWN file was loaded with, BEFORE translation -- `None`
    /// when the mode HAS no file of its own and therefore serves the default set.
    ///
    /// `None` vs `Some(0)` is the load-side statement the whole design rests on: "this
    /// mode names no set" vs "this mode names an EMPTY set and serves nothing". Builds no
    /// index -- it is a count of raw loaded pairs.
    #[pyo3(signature = (rule_mode="default"))]
    fn mode_rules_len(&self, rule_mode: &str) -> PyResult<Option<usize>> {
        Ok(self.inner.mode_rules_len(parse_rule_mode(rule_mode)?))
    }

    /// Licence-registry load-consumer audit (C1.20): rules refused at AC load by the
    /// pole entry (a defined extended-real value changed at a constructed exceptional
    /// point). 0 for every artifact the mint-side registry produced; nonzero means a
    /// stale or foreign artifact carried a violation past its own mine. Forces the
    /// lazy translation.
    fn ac_registry_dropped(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(py.detach(|| self.inner.ac_registry_dropped()))
    }

    /// C36: the per-reason rule-drop census as a dict -- what happened to every
    /// non-kept rule of the loaded asset, by name.
    fn ac_rules_drop_census(
        &self,
        py: Python<'_>,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        Ok(py
            .detach(|| self.inner.ac_rules_drop_census())
            .into_iter()
            .collect())
    }

    /// The licence registry for one would-be rule (C1.20): the HINT/PROPOSAL channels'
    /// entry point -- a hint that passes vocabulary/ordering/numeric confirmation is
    /// still a mint, and every mint channel consults the registry. Returns the refusal
    /// name (`"const_introduction"` / `"special_absorption"` / `"pole_class_change"`)
    /// or None. `mark` is the engine's endpoint for the source (the special count is
    /// taken on the max of both sides, F49's spelling-independence); wildcard-shaped
    /// tokens (`_i`/`!i`/`?i`) in the patterns are treated as variables alongside
    /// `var_names`.
    fn registry_mint_refusal(
        &self,
        py: Python<'_>,
        source: Vec<String>,
        mark: Vec<String>,
        target: Vec<String>,
        var_names: Vec<String>,
    ) -> PyResult<Option<String>> {
        Ok(py.detach(|| {
            let mut vars = var_names;
            for t in source.iter().chain(target.iter()) {
                let wild = t.len() >= 2
                    && (t.starts_with('_') || t.starts_with('!') || t.starts_with('?'))
                    && t[1..].chars().all(|c| c.is_ascii_digit());
                if wild && !vars.contains(t) {
                    vars.push(t.clone());
                }
            }
            if let Some(r) = engine::refusals::mint_refusals(&source, &mark, &target, &vars) {
                return Some(r.name().to_string());
            }
            if self.inner.mint_pole_refusal(&source, &target, &vars) {
                engine::stats::REGISTRY_POLE_REFUSALS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(
                    engine::refusals::Refusal::PoleClassChange
                        .name()
                        .to_string(),
                );
            }
            None
        }))
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
        ensure_tokens_are_tokens(&tokens)?;
        py.detach(|| self.inner.prefix_to_infix(&tokens, power_mode, realization))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// SYNTACTIC prefix -> tagged regrouping (`forms.rs`): the notation half of the
    /// conversion/simplification split. Never consults the ruleset.
    fn to_tagged_syntactic(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        ensure_tokens_are_tokens(&tokens)?;
        let out = py
            .detach(|| self.inner.to_tagged_syntactic(&tokens))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// SYNTACTIC tagged -> explicit binary prefix expansion (`forms.rs`).
    fn to_prefix_syntactic(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        ensure_tokens_are_tokens(&tokens)?;
        let out = py
            .detach(|| self.inner.to_prefix_syntactic(&tokens))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// Well-formedness in either token dialect; `ValueError` names the malformation.
    fn check_form(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<()> {
        ensure_tokens_are_tokens(&tokens)?;
        py.detach(|| self.inner.check_form(&tokens))
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
        ensure_tokens_are_tokens(&prefix_expr)?;
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
    fn evaluate_constant_subtree(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
    ) -> PyResult<Option<String>> {
        ensure_tokens_are_tokens(&tokens)?;
        Ok(py.detach(|| self.inner.evaluate_constant_subtree(&tokens)))
    }

    /// The CPython-exact `str(float)` formatter alone (the result-formatting half of numeric folding),
    /// exposed for the float-repr fuzz. Static; does not need engine state.
    /// KEPT with no in-repo caller: fuzz-instrument surface (the float-repr differential fuzz drives it via `_core`; audit 2026-08-03).
    #[staticmethod]
    fn py_float_repr(x: f64) -> String {
        crate::numeric::py_float_repr(x)
    }

    /// OFFLINE miner kernel: vectorized evaluation of a prefix expression over
    /// `n_rows` rows of row-major `x_flat` (shape `n_rows x len(var_names)`); `<constant>` slots bind
    /// to `params` left-to-right; numeric/special literals fold to their value. Returns the length-
    /// `n_rows` result column. NOT part of the inline (online) surface; replaces the Python
    /// realize->infix->codify->lambda->safe_f residual path inside `find_rule_worker`.
    /// KEPT with no in-repo caller: offline-miner instrument surface -- flash-ansr's `experimental/simplipy_offline_miner` scripts drive it via `_core` (audit 2026-08-03).
    fn evaluate_batch(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        var_names: Vec<String>,
        x_flat: Vec<f64>,
        n_rows: usize,
        params: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        ensure_tokens_are_tokens(&tokens)?;
        py.detach(|| {
            self.inner
                .evaluate_batch(&tokens, &var_names, &x_flat, n_rows, &params)
        })
        .map_err(PyValueError::new_err)
    }

    /// `numpy.allclose(a, b, rtol, atol, equal_nan=True)` -- the miner's accept/reject decision gate.
    /// `b` is the asymmetric reference (second arg), matching the miner's call order. Static.
    /// KEPT with no in-repo caller: offline-miner instrument surface -- flash-ansr's `experimental/simplipy_offline_miner` scripts drive it via `_core` (audit 2026-08-03).
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
        // Universal floor (D3, extends H-038): an EXPLICIT 0 would silently disable the
        // evidence gate, and gate-disabling must ride a RECORDED channel (the provenance
        // kill-switches), never a bare numeric argument. Defaults were floored by d2db5dd.
        ensure_tokens_are_tokens(&source)?;
        ensure_tokens_are_tokens(&candidate)?;
        let mi = min_informative.unwrap_or(n_rows / 8).max(1);
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
    /// KEPT with no in-repo caller: certificate-layer probe surface for cross-repo research scripts and ad-hoc soundness audits (queried directly via `_core`; audit 2026-08-03).
    fn interval_finite_ae(&self, tokens: Vec<String>) -> PyResult<bool> {
        ensure_tokens_are_tokens(&tokens)?;
        Ok(crate::interval::finite_ae(
            &tokens,
            self.inner.operators_ref(),
        ))
    }

    /// OFFLINE instrument (F80 E2 design read): the zero-set certificate exactly as
    /// the odd-negative distribution licence consumes it (`zero_set_null_generic_const`,
    /// E1 semantics: `<constant>` as a generic parameter in the joint space). Lets the
    /// design-read scripts ask the LIVE walker+witness about transformed factor
    /// spellings instead of mirroring the machinery in python. Read-only diagnostics.
    fn interval_zero_set_null_generic_const(&self, tokens: Vec<String>) -> PyResult<bool> {
        ensure_tokens_are_tokens(&tokens)?;
        Ok(crate::interval::zero_set_null_generic_const(
            &tokens,
            self.inner.operators_ref(),
        ))
    }

    /// OFFLINE instrument (F80 E2 design read): the witness path's entire-analyticity
    /// gate alone (`entire_analytic_composition`), for splitting gate refusals from
    /// witness/horizon failures in the design read. Read-only diagnostics.
    fn interval_entire_analytic_composition(&self, tokens: Vec<String>) -> PyResult<bool> {
        ensure_tokens_are_tokens(&tokens)?;
        Ok(crate::interval::entire_analytic_composition_probe(
            &tokens,
            self.inner.operators_ref(),
        ))
    }

    /// OFFLINE instrument (F80 E3 read): kept negative-exponent bag carriers
    /// `Pow(Mul[..], n<0)` in the sound simplify endpoint, with per-factor licence
    /// verdicts (`nz_ae_certified`, `certainly_nonneg`) re-asked under the deployed
    /// certificates. Rows `(carrier_prefix, exp_token, odd_neg_int, has_div_factor,
    /// [(factor_prefix, nzae, nonneg), ..])` -- the F80 nesting classification's
    /// bridge. Read-only diagnostics; nothing feeds the chain.
    #[pyo3(signature = (tokens, max_passes=48))]
    fn ac_odd_neg_carriers(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_passes: usize,
    ) -> PyResult<
        Vec<(
            Vec<String>,
            String,
            bool,
            bool,
            Vec<(Vec<String>, bool, bool)>,
        )>,
    > {
        ensure_ac_well_formed(&self.inner, &tokens)?;
        py.detach(|| self.inner.ac_odd_neg_carriers(&tokens, max_passes))
            .ok_or_else(|| PyValueError::new_err("invalid or malformed prefix expression"))
    }

    /// OFFLINE: EXACT value-class of an expression by interval analysis (`rust/interval.rs`) --
    /// the deterministic replacement for the sampled pole grid / classify probe. Returns one of
    /// FINITE / POSINF / NEGINF / NAN / MIXED / EMPTY, or ERROR if unevaluable.
    /// KEPT with no in-repo caller: certificate-layer probe surface for cross-repo research scripts and ad-hoc soundness audits (queried directly via `_core`; audit 2026-08-03).
    fn interval_class(&self, tokens: Vec<String>) -> PyResult<String> {
        ensure_tokens_are_tokens(&tokens)?;
        Ok(
            match crate::interval::value_class(&tokens, self.inner.operators_ref()) {
                Some(c) => format!("{:?}", c).to_uppercase(),
                None => "ERROR".to_string(),
            },
        )
    }

    /// OFFLINE: the raw positive-measure value COMPONENTS of an expression over free constants,
    /// as `(has_finite, pos_inf, neg_inf, nan)` -- the reachability gate's inputs, before `Class`
    /// collapses them. `None` if unevaluable.
    /// KEPT with no in-repo caller: certificate-layer probe surface for cross-repo research scripts and ad-hoc soundness audits (queried directly via `_core`; audit 2026-08-03).
    fn interval_value_components(
        &self,
        tokens: Vec<String>,
    ) -> PyResult<Option<(bool, bool, bool, bool)>> {
        ensure_tokens_are_tokens(&tokens)?;
        Ok(crate::interval::value_set(
            &tokens,
            self.inner.operators_ref(),
            &crate::interval::Vs::reals(),
        )
        .map(|v| (v.has_fin, v.pinf, v.ninf, v.nan)))
    }

    /// OFFLINE: how many gate calls fell OUTSIDE the box horizon and were therefore UNDECIDED
    /// (the caller fails closed: rejected, lost recall not lost soundness). The gate is a dyadic
    /// witness search over a BOUNDED box; when an expression's own constants push the interesting
    /// region past the cap, it does not know. This counter is the exposure -- recorded per mine in
    /// the provenance sidecar (`soundness.interval_undecided`), never assumed zero.
    fn interval_horizon_misses(&self) -> u64 {
        crate::interval::HORIZON_MISSES.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// OFFLINE: how many subdivision searches exhausted their node budget with no witness found.
    /// Since the F0-era fix this is UNDECIDED and the caller fails closed (an exhausted search has
    /// not earned the decided "no witness"). Recorded per mine in the provenance sidecar, next to
    /// `interval_horizon_misses`.
    fn interval_node_budget_misses(&self) -> u64 {
        crate::interval::NODE_BUDGET_MISSES.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// OFFLINE: how many subdivision searches visited an UNANALYZABLE box (`value_set_p`
    /// fail-closed, e.g. rootn with an expression-position index) and found no witness -- each
    /// returned UNDECIDED and its caller rejected (fail-closed). This was the THIRD fail-open
    /// until 2026-07-30 (F0): the abandoned box read as clean and the completed search as a
    /// decided verdict, which is how the five NaN-a.e. rootn rules of the 2026-07-29 acj-4-3
    /// mine shipped. Read after a mine, next to the other two miss counters.
    fn interval_unanalyzable_misses(&self) -> u64 {
        crate::interval::UNANALYZABLE_MISSES.load(std::sync::atomic::Ordering::Relaxed)
    }

    // (`simplify_counters`/`reset_simplify_counters` were removed in D3, 2026-08-05:
    // they reported the DELETED binary kernel's hot-path counters, whose bump sites
    // went with the kernel -- an FFI answering all-zeros regardless of build features
    // is a record that misleads, not observability. See `engine::stats`.)

    /// OFFLINE mining progress: `(sources_done, sources_total)` for the length tier currently
    /// being mined by `mine_one_length`. `mine_one_length` is one blocking, rayon-parallel call,
    /// so a driver polls this from another thread to get within-tier progress (rate / ETA)
    /// instead of only the end-of-tier summary. `(0, 0)` before the first tier starts.
    fn mining_progress(&self) -> (u64, u64) {
        crate::engine::stats::mine_progress()
    }

    /// Licence-registry observability (C1.20): the process-wide count of candidates
    /// the pole entry refused, across every mint channel. A raised-tier mine that
    /// reaches a pole-mirror candidate INCREMENTS this while minting nothing -- the
    /// falsification tests read it to prove the gate FIRED rather than the candidate
    /// never being enumerated ("report it, never silence it", `engine::stats`).
    fn registry_pole_refusals(&self) -> u64 {
        crate::engine::stats::REGISTRY_POLE_REFUSALS.load(std::sync::atomic::Ordering::Relaxed)
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
    ) -> PyResult<Option<ValueSetBox>> {
        ensure_tokens_are_tokens(&tokens)?;
        if los.len() != his.len() || los.is_empty() {
            return Ok(None);
        }
        let doms: Vec<crate::interval::Vs> = los
            .iter()
            .zip(his.iter())
            .map(|(lo, hi)| crate::interval::Vs::interval(*lo, *hi, false, false))
            .collect();
        Ok(
            crate::interval::value_set_p(&tokens, self.inner.operators_ref(), &doms, &[])
                .map(|v| (v.has_fin, v.pinf, v.ninf, v.nan, v.lo, v.hi)),
        )
    }

    /// OFFLINE: witness WIDTH of a positive-measure region where `source` is NaN but `target`
    /// defines a value (0.0 = no domain extension). This is the exact domain-preservation
    /// gate: a rule must never be grossly domain dependent. `None` = UNDECIDED (horizon past
    /// R_MAX, or node budget exhausted with no witness) -- the gate fails closed on it; a
    /// consumer must treat `None` as "extension not ruled out", never as 0.0.
    /// KEPT with no in-repo caller: certificate-layer probe surface for cross-repo research scripts and ad-hoc soundness audits (queried directly via `_core`; audit 2026-08-03).
    fn interval_domain_extension(
        &self,
        source: Vec<String>,
        target: Vec<String>,
    ) -> PyResult<Option<f64>> {
        // H-043/D4: the guard its four sibling probes already run (alphabet + cap).
        ensure_tokens_are_tokens(&source)?;
        ensure_tokens_are_tokens(&target)?;
        Ok(crate::interval::domain_extension(
            &source,
            &target,
            self.inner.operators_ref(),
        ))
    }

    /// OFFLINE: `interval_domain_extension` with both sides' `<constant>` leaves BOUND to concrete
    /// values (k-th `<constant>` in prefix order -> `params[k]`). This is the form the miner uses:
    /// the gate runs per instance at the source's drawn constants and the constants the fit chose.
    /// `None` = UNDECIDED (fail-closed), as in `interval_domain_extension`.
    /// KEPT with no in-repo caller: certificate-layer probe surface for cross-repo research scripts and ad-hoc soundness audits (queried directly via `_core`; audit 2026-08-03).
    fn interval_domain_extension_p(
        &self,
        source: Vec<String>,
        src_params: Vec<f64>,
        target: Vec<String>,
        tgt_params: Vec<f64>,
    ) -> PyResult<Option<f64>> {
        ensure_tokens_are_tokens(&source)?;
        ensure_tokens_are_tokens(&target)?;
        Ok(crate::interval::domain_extension_p(
            &source,
            &src_params,
            &target,
            &tgt_params,
            self.inner.operators_ref(),
        ))
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
        ensure_tokens_are_tokens(&source)?;
        for c in &candidates {
            ensure_tokens_are_tokens(c)?;
        }
        // Universal floor (D3, extends H-038): an EXPLICIT 0 would silently disable the
        // evidence gate, and gate-disabling must ride a RECORDED channel (the provenance
        // kill-switches), never a bare numeric argument. Defaults were floored by d2db5dd.
        let mi = min_informative.unwrap_or(n_rows / 8).max(1);
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
        for c in &candidates {
            ensure_tokens_are_tokens(c)?;
        }
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
    /// `mark` (optional): the engine's own result for the source -- when given, the
    /// serve-time reduction ordering rides the candidate scan (finding F2): only targets
    /// STRICTLY BELOW the mark are selectable, a refused candidate yields to the next one,
    /// and a fully refused length yields to the next length.
    #[pyo3(signature = (source, simplified_length, max_target, library, challenges=16, retries=16, seed=0, rtol=1e-9, atol=1e-12, min_informative=None, mark=None))]
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
        mark: Option<Vec<String>>,
    ) -> PyResult<Option<Vec<String>>> {
        ensure_tokens_are_tokens(&source)?;
        if let Some(m) = &mark {
            ensure_tokens_are_tokens(m)?;
        }
        let lib = &library.inner;
        // Universal floor (D3, extends H-038): see equivalent_no_const.
        let mi = min_informative.unwrap_or(lib.n_rows() / 8).max(1);
        // A user-supplied mark that does not parse would otherwise refuse EVERY candidate
        // silently (`ordered_below` = None on either side failing) -- reject it loudly.
        if let Some(m) = &mark {
            if self.inner.ac_ordered_below(m, m).is_none() {
                return Err(PyValueError::new_err(
                    "mark is not a parseable prefix expression",
                ));
            }
            // ATOMIC MARK ENDS THE SEARCH (signed-zero finding, 2026-08-02; the
            // native twin lives in engine::miner::mine_one_length): no sound target
            // sits strictly below a single atom -- a different atom is a different
            // VALUE, and the ordering tiebreak ranks atoms among themselves, so
            // searching below an atomic mark can only mint value-changes.
            if m.len() == 1 {
                return Ok(None);
            }
        }
        py.detach(|| {
            // THE LICENCE REGISTRY RUNS ON EVERY MINT CHANNEL (C1.20, 2026-08-08).
            // This is the python-facing single-source channel (the hint/proposal
            // paths of `find_rules` -- the exact channel family F49's seven
            // absorptions came through), and it used to carry only the determined-
            // source gate: the special-count licence and the pole entry existed
            // solely in `mine_one_length`'s closure. A channel gated by "the LLM
            // channel is off in the acj configs" is a channel gated by coincidence
            // -- the C1.20 pattern itself -- so the registry now runs here
            // unconditionally: `mint_refusals` with the caller's mark as the
            // endpoint (its absence degrades the special count to the raw source
            // side, never below it), and the pole entry after the ordering, exactly
            // as in the native path. No-mark calls previously ran UNGATED; they now
            // carry the registry too (the ordering acceptance still requires a
            // mark, as before).
            let empty: Vec<String> = Vec::new();
            let mark_for_count: &Vec<String> = mark.as_ref().unwrap_or(&empty);
            let src_ref: &Vec<String> = &source;
            let mark_opt: Option<&Vec<String>> = mark.as_ref();
            let inner = &self.inner;
            let accept = Some(move |t: &[String]| {
                if engine::refusals::mint_refusals(src_ref, mark_for_count, t, lib.var_names())
                    .is_some()
                {
                    return false;
                }
                if let Some(m) = mark_opt {
                    let below = match inner.ac_ordered_below(t, m) {
                        Some(below) => below,
                        None => {
                            // The mark parses (validated above) and candidates are library
                            // spellings: an undecidable ordering here is a bug -- loud in
                            // debug, counted in release (never a silent refusal).
                            debug_assert!(
                                false,
                                "ordering acceptance could not parse candidate {t:?}"
                            );
                            engine::stats::ACCEPT_UNDECIDED_REFUSALS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            false
                        }
                    };
                    if !below {
                        return false;
                    }
                }
                if inner.mint_pole_refusal(src_ref, t, lib.var_names()) {
                    engine::stats::REGISTRY_POLE_REFUSALS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
                true
            });
            // Resolved-literal targets additionally need a STRICT mu descent below
            // the mark AND a DIFFERENT literal-skeleton (see engine::miner): under mu
            // a rounded literal descends from a long exact one by bits alone, so
            // strict descent stopped refusing respells -- the skeleton guard is the
            // doctrine ("resolution recovers structure, never respells literals")
            // made structural. Fail closed on unparseable sides.
            // One mu for the mark, used TWICE: as the resolved-target acceptance threshold
            // and as the scan bound. They were separate criteria (mu vs a token ceiling);
            // the ruling makes them one, so they must read the same number.
            let mark_mu = mark.as_ref().and_then(|m| self.inner.ac_complexity(m));
            let accept_resolved = mark.as_ref().map(|m| {
                let mark_c = mark_mu;
                move |t: &[String]| {
                    matches!(
                        (self.inner.ac_complexity(t), mark_c),
                        (Some(tc), Some(mc)) if tc < mc
                    ) && !self.inner.ac_same_literal_skeleton(t, m).unwrap_or(true)
                }
            });
            self.inner.find_rule_with_lib(
                &source,
                simplified_length,
                max_target,
                mark_mu,
                lib,
                challenges,
                retries,
                seed,
                rtol,
                atol,
                mi,
                accept.as_ref().map(|f| f as &dyn Fn(&[String]) -> bool),
                accept_resolved
                    .as_ref()
                    .map(|f| f as &dyn Fn(&[String]) -> bool),
            )
        })
        .map_err(PyValueError::new_err)
    }

    /// OFFLINE (mine driver): replace the engine's rules (recompile) -- grows the Kruskal-prune
    /// rule set length-by-length during a mine. `rules` = the canonicalized (wildcard) rule list.
    fn set_rules(&mut self, rules: Vec<(Vec<String>, Vec<String>)>) -> PyResult<()> {
        // H-043/D4: rule compile parses each side into a tree (recursive); cap + alphabet
        // per side, like every other token-taking entry.
        for (lhs, rhs) in &rules {
            ensure_tokens_are_tokens(lhs)?;
            ensure_tokens_are_tokens(rhs)?;
        }
        self.pushed_rules_digest = pushed_rules_digest(&rules);
        self.inner.set_rules(rules);
        Ok(())
    }

    /// Install ONE MODE'S OWN COMPLETE RULE SET -- `"real"` (`rules_real.json`) or
    /// `"corpus"` (`rules_corpus.json`), each a self-contained file carrying the core
    /// rules AND that mode's own, never a supplement to `rules.json`.
    ///
    /// `rules=None` RETRACTS the set, and that mode falls back to the default one.
    /// `rules=[]` is a DIFFERENT call: it installs an EMPTY set, and that mode then
    /// serves nothing. Do not collapse them -- the distinction is what makes "the mode's
    /// set is exactly the file it names" a statement rather than an inference.
    ///
    /// `rule_mode="default"` routes to `set_rules`, so there is one way to replace the
    /// default set and it keeps that path's digest bookkeeping.
    ///
    /// The base-rule digest (`rules_in_sync`) deliberately does NOT cover these sets: it
    /// answers "does the core hold the DEFAULT recipe it was pushed", and the other two
    /// sets are separate recipes carried by the Python shim's own attributes, which
    /// re-push them on every core rebuild.
    #[pyo3(signature = (rule_mode, rules))]
    fn set_mode_rules(
        &mut self,
        rule_mode: &str,
        rules: Option<Vec<(Vec<String>, Vec<String>)>>,
    ) -> PyResult<()> {
        let mode = parse_rule_mode(rule_mode)?;
        // H-043/D4: same cap + alphabet as every other token-taking entry -- rule compile
        // parses each side into a tree, recursively.
        if let Some(rules) = &rules {
            for (lhs, rhs) in rules {
                ensure_tokens_are_tokens(lhs)?;
                ensure_tokens_are_tokens(rhs)?;
            }
        }
        if mode == engine::RuleMode::Default {
            return self.set_rules(rules.unwrap_or_default());
        }
        self.inner.set_mode_rules(mode, rules);
        Ok(())
    }

    /// C1.9: does `rules` match what was last PUSHED into this core (`from_strs` or
    /// `set_rules`)? The Python shim's `__getstate__` calls this with the current
    /// `simplification_rules` and refuses to pickle on a mismatch -- a pickle rebuilds
    /// the worker from the LIST, so a diverged parent would otherwise silently spawn
    /// workers running different rules.
    fn rules_in_sync(&self, rules: Vec<(Vec<String>, Vec<String>)>) -> bool {
        pushed_rules_digest(&rules) == self.pushed_rules_digest
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
    ) -> PyResult<Vec<(Vec<String>, Vec<String>)>> {
        for src in &sources {
            ensure_tokens_are_tokens(src)?;
        }
        let lib = &library.inner;
        // Universal floor (D3, extends H-038): see equivalent_no_const.
        let mi = min_informative.unwrap_or(lib.n_rows() / 8).max(1);
        Ok(py.detach(|| {
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
        }))
    }

    /// OFFLINE miner: native `exist_constants_that_fit` for AFFINE-in-params candidates -- a
    /// closed-form least-squares solve + the `allclose` decision gate (no optimizer,
    /// deterministic). Returns `Some(decision)` for affine candidates, `None` for
    /// nonlinear-in-params ones (the native-LM path). Same accept/reject gate as scipy's path.
    #[pyo3(signature = (candidate, var_names, x_flat, n_rows, y_target, rtol=1e-5, atol=1e-8))]
    #[allow(clippy::too_many_arguments)] // the Python API signature: mirrors the deployed call shape
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
        ensure_tokens_are_tokens(&candidate)?;
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
    /// KEPT with no in-repo caller: offline-miner instrument surface -- flash-ansr's `experimental/simplipy_offline_miner` scripts drive it via `_core` (audit 2026-08-03).
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
        ensure_tokens_are_tokens(&candidate)?;
        py.detach(|| {
            self.inner.exist_constants_fit(
                &candidate, &var_names, &x_flat, n_rows, &y_target, rtol, atol, n_restarts, seed,
            )
        })
        .map_err(PyValueError::new_err)
    }
}

/// The CORE SERIALIZATION LANGUAGE, exported so Python reads the SAME table the engine
/// does instead of keeping its own copy.
///
/// `{token: {"arity": int, "precedence": float, "realization": str | None}}`. The audit
/// counted five hand-maintained copies of this data (C1.10); a copy that drifts is not a
/// tidiness problem but a soundness one -- the masking walk treated an unknown operator as
/// a LEAF, so its operands inherited the enclosing bag's role and a `pow` exponent could be
/// masked, the exact accident the role API exists to prevent.
#[pyfunction]
fn core_serialization_ops(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    for (tok, arity, precedence, realization) in crate::operators::CORE_SERIALIZATION_OPS {
        let entry = PyDict::new(py);
        entry.set_item("arity", arity)?;
        entry.set_item("precedence", precedence)?;
        entry.set_item("realization", realization)?;
        out.set_item(tok, entry)?;
    }
    Ok(out.into())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyCandidateLibrary>()?;
    m.add_function(wrap_pyfunction!(core_serialization_ops, m)?)?;
    m.add("__build__", env!("SIMPLIPY_BUILD"))?;
    Ok(())
}
