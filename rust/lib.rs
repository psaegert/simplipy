//! simplipy-rust: pure-Rust faithful re-implementation of the SimpliPy prefix-expression
//! simplifier, exposed to Python as `simplipy_rust._core` via PyO3.
//!
//! ## FFI design (load-bearing, per the verified analysis)
//! The ENTIRE simplify fixpoint is ported as ONE FFI unit. A single call into Rust receives the
//! prefix token list and returns the simplified token list. Inside Rust, the whole recursion runs
//! with NO boundary crossings:
//!   cancel_terms -> apply_simplification_rules (parse_subtree + apply_rules_top_down +
//!   match_pattern + apply_mapping + the constant fold) -> sort_operands ->
//!   mask_elementary_literals -> longer-result guard, iterated to a fixpoint (<= max_iter).
//! ~1.8 boundary crossings/expr; FFI marshalling stays <1% of wall time.
//! Porting `match_pattern` alone is a TRAP (millions of crossings -> the speedup evaporates).
//! Therefore the PyO3 layer here is deliberately THIN: marshal `list[str]` <-> `Vec<String>`,
//! hold the compiled engine, and delegate the whole unit to `engine::Engine::simplify`.
//!
//! ## Faithful-v1 target
//! Reproduce engine-id `dev_7-3` @ simplipy 0.2.15 (git 1fe9b7e) on skeleton inputs at
//! `max_pattern_length=4` (deployed) and `=7` (offline). v1 is a behavioral drop-in; the only
//! constant fold that fires on skeletons collapses an all-`<constant>` subtree to `["<constant>"]`
//! (pure string/tree rewriting, NO float math), so the str(float)/str(int) repr hazard is OUT of
//! scope for v1 (it belongs to the separate `dev_7-3_numeric` track).

// NOTE: PyO3 0.27 (current-stable, pinned in Cargo.toml) Bound API. The exact shapes of
// `PyList::new`, `Python::allow_threads`, and the `#[pymodule]` fn signature shift slightly across
// PyO3 minors; VERIFY them against the pinned version at the first `maturin develop` (the crate
// cannot build until the kernel bodies replace the `unimplemented!()` stubs anyway, so these are
// fixed mechanically on the first real compile, not reconstructed from memory here).
use pyo3::prelude::*;
use pyo3::types::PyList;

mod cancel;
mod convert;
mod engine;
mod numeric;
mod operators;
mod parse;
mod rules;
mod sort;
mod utils;

pub use engine::Engine;

/// Engine-id this build faithfully reproduces. Bump (new engine-id) on any quality-shifting
/// change so prior training results (v23.0) stay reproducible against the old id.
pub const FAITHFUL_ENGINE_ID: &str = "dev_7-3";
/// simplipy reference the faithful port targets (provenance, surfaced to Python).
pub const REFERENCE_SIMPLIPY_VERSION: &str = "0.2.15";
pub const REFERENCE_SIMPLIPY_COMMIT: &str = "1fe9b7e88563368c15063ab726d1468c6bee9869";

/// Opaque, compiled engine handle held on the Python side. Construction (parse config.yaml +
/// rules.json, build the bucket index + first-operand filter) happens ONCE; `simplify` is the
/// hot path.
#[pyclass(name = "Engine", module = "simplipy._core")]
struct PyEngine {
    inner: engine::Engine,
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

    /// THE hot path and the whole FFI unit. `tokens` is a prefix token list; returns the
    /// simplified prefix token list. Defaults mirror the deployed call
    /// (`simplify(skeleton, inplace=True, max_pattern_length=4)`); `inplace` is a Python-shim
    /// concern (the shim mutates the caller's list), so it is NOT a kernel parameter here.
    /// `fold` selects the engine line: `false` (default) = faithful `dev_7-3`; `true` = the numeric
    /// line (constant folding woven into the rule recursion, the published improved engine-id). The
    /// faithful default keeps the frozen-reference parity untouched.
    #[pyo3(signature = (tokens, max_iter=5, max_pattern_length=None, mask_elementary_literals=true,
                        apply_simplification_rules=true, fold=false))]
    fn simplify(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_iter: usize,
        max_pattern_length: Option<usize>,
        mask_elementary_literals: bool,
        apply_simplification_rules: bool,
        fold: bool,
    ) -> PyResult<Py<PyList>> {
        // Release the GIL for the pure-Rust kernel: parallel callers (corpus harness, dataloader
        // workers) are not serialized on Python's lock.
        let out = py.allow_threads(|| {
            self.inner.simplify(
                &tokens,
                max_iter,
                max_pattern_length,
                mask_elementary_literals,
                apply_simplification_rules,
                fold,
            )
        });
        Ok(PyList::new(py, out)?.into())
    }

    /// Validation entry (NOT the shipped surface): the rule-application sub-unit only. `fold=false`
    /// = faithful `apply_simplifcation_rules`; `fold=true` = the numeric line (used by the improved
    /// differential). Used by the stage-(b) differential vs fresh Python.
    #[pyo3(signature = (tokens, max_pattern_length=None, fold=false))]
    fn apply_rules(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        max_pattern_length: Option<usize>,
        fold: bool,
    ) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| {
            self.inner
                .apply_simplification_rules(&tokens, max_pattern_length, fold)
        });
        Ok(PyList::new(py, out)?.into())
    }

    /// Validation entry (NOT the shipped surface): the term-cancellation sub-unit only -- a faithful
    /// `cancel_terms(*collect_multiplicities(tokens))`. Used by the differential vs fresh Python
    /// before the whole `simplify` fixpoint (sort + iterate) is ported. `mpl`-independent.
    fn cancel_only(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| self.inner.cancel_terms(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Validation entry (NOT the shipped surface): the operand-sort sub-unit only -- a faithful
    /// `sort_operands`. Used by the differential vs fresh Python before the whole `simplify` fixpoint
    /// compose.
    fn sort_only(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| self.inner.sort_operands(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Faithful port of `is_valid` (engine.py:354): is the prefix expression syntactically valid?
    /// Part of M2 (the drop-in-engine surface); the most-called simplipy method on the per-candidate
    /// inference path.
    fn is_valid(&self, py: Python<'_>, tokens: Vec<String>) -> bool {
        py.allow_threads(|| self.inner.is_valid(&tokens))
    }

    /// Faithful port of `prefix_to_infix` (engine.py:409). `power` in {'func','**'} (default 'func');
    /// `realization` toggles realization-name rendering. Raises `ValueError` on a malformed prefix
    /// (mirrors Python). Part of M2 (the drop-in-engine surface).
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
        py.allow_threads(|| self.inner.prefix_to_infix(&tokens, power_mode, realization))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Faithful port of `infix_to_prefix` (engine.py:581). Part of M2 (the drop-in-engine surface).
    fn infix_to_prefix(&self, py: Python<'_>, infix_expression: &str) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| self.inner.infix_to_prefix(infix_expression));
        Ok(PyList::new(py, out)?.into())
    }

    /// Faithful port of `convert_expression` (engine.py:655). Raises `ValueError` where Python raises
    /// (the exact exception kind differs -- the differential checks failure-parity). Part of M2.
    fn convert_expression(&self, py: Python<'_>, prefix_expr: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py
            .allow_threads(|| self.inner.convert_expression(&prefix_expr))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// Faithful port of `parse` (engine.py:852). `convert_expression`/`mask_numbers` match the Python
    /// defaults (True/False). Closes the `simplify(str)` + canonicalization path. Part of M2.
    #[pyo3(signature = (infix_expression, convert_expression=true, mask_numbers=false))]
    fn parse(
        &self,
        py: Python<'_>,
        infix_expression: &str,
        convert_expression: bool,
        mask_numbers: bool,
    ) -> PyResult<Py<PyList>> {
        let out = py
            .allow_threads(|| self.inner.parse(infix_expression, convert_expression, mask_numbers))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// Corrected (deliberate-improvement) variants: the conversion-quirk fixes. NOT `dev_7-3`; these
    /// mirror the Python `fix/conversion-quirks` branch and back a future fixed engine-id.
    #[pyo3(signature = (tokens, power="func", realization=false))]
    fn prefix_to_infix_fixed(
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
        py.allow_threads(|| self.inner.prefix_to_infix_fixed(&tokens, power_mode, realization))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn infix_to_prefix_fixed(&self, py: Python<'_>, infix_expression: &str) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| self.inner.infix_to_prefix_fixed(infix_expression));
        Ok(PyList::new(py, out)?.into())
    }

    fn convert_expression_fixed(&self, py: Python<'_>, prefix_expr: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py
            .allow_threads(|| self.inner.convert_expression_fixed(&prefix_expr))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    #[pyo3(signature = (infix_expression, convert_expression=true, mask_numbers=false))]
    fn parse_fixed(
        &self,
        py: Python<'_>,
        infix_expression: &str,
        convert_expression: bool,
        mask_numbers: bool,
    ) -> PyResult<Py<PyList>> {
        let out = py
            .allow_threads(|| self.inner.parse_fixed(infix_expression, convert_expression, mask_numbers))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyList::new(py, out)?.into())
    }

    /// Faithful port of `operators_to_realizations` (engine.py:2547).
    fn operators_to_realizations(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| self.inner.operators_to_realizations(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Faithful port of `realizations_to_operators` (engine.py:2566).
    fn realizations_to_operators(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
        let out = py.allow_threads(|| self.inner.realizations_to_operators(&tokens));
        Ok(PyList::new(py, out)?.into())
    }

    /// Native numeric constant folding (the `numeric` line). Returns the result token, or `None` if
    /// the subtree cannot be folded (complex result / unparseable leaf / unknown operator) -- matching
    /// Python `_evaluate_constant_subtree`. Validation entry for the differential.
    fn evaluate_constant_subtree(&self, py: Python<'_>, tokens: Vec<String>) -> Option<String> {
        py.allow_threads(|| self.inner.evaluate_constant_subtree(&tokens))
    }

    /// The CPython-exact `str(float)` formatter alone (the result-formatting half of numeric folding),
    /// exposed for the float-repr fuzz. Static; does not need engine state.
    #[staticmethod]
    fn py_float_repr(x: f64) -> String {
        crate::numeric::py_float_repr(x)
    }

    #[getter]
    fn engine_id(&self) -> &str {
        self.inner.engine_id()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add("FAITHFUL_ENGINE_ID", FAITHFUL_ENGINE_ID)?;
    m.add("REFERENCE_SIMPLIPY_VERSION", REFERENCE_SIMPLIPY_VERSION)?;
    m.add("REFERENCE_SIMPLIPY_COMMIT", REFERENCE_SIMPLIPY_COMMIT)?;
    m.add("__build__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
