//! The `Engine`: the whole-unit simplify kernel -- the Rust analogue of the removed pure-Python
//! `SimpliPyEngine.simplify` and its callees, ported as ONE FFI unit (see lib.rs).
//! The kernel runs on interned `Tok` ids (see `crate::tokens`); the string boundary is the public
//! `&[String]` methods (intern once at entry, resolve once at exit). Submodules: [`stats`]
//! (hot-path counters), `memo` (cache/ctx state), `ac` (THE engine), `miner` (the OFFLINE
//! mining surface); config deserialization, the `Engine` struct and constructors live here.
//! The old binary kernel is deleted (clean 0.12); masking is the PYTHON-side
//! `simplipy.masking` module, downstream of simplify, never a kernel pass.

mod ac;
mod memo;
mod miner;
pub mod stats;

use std::error::Error;
use std::fs;

use rustc_hash::FxHashMap;

use crate::operators::{OperatorSpec, Operators};
use crate::rules::CompiledRules;
use crate::tokens::{Tok, TokenTable, TokenView};

use memo::{BangCache, SimplifyCtx};

pub use ac::AcForm;

/// Test-only plumbing for the serialization-stability probes (a per-call ctx + view + intern).
#[cfg(test)]
fn memo_ctx_for_tests(e: &Engine) -> SimplifyCtx {
    SimplifyCtx::new(e.tokens.len())
}
#[cfg(test)]
fn intern_for_tests(e: &Engine, tokens: &[String], ctx: &SimplifyCtx) -> Vec<Tok> {
    e.intern_seq(tokens, ctx)
}
#[cfg(test)]
fn view_for_tests<'a>(e: &'a Engine, ctx: &'a SimplifyCtx) -> TokenView<'a> {
    e.view(ctx)
}

/// The on-disk engine config: the operator block + a relative path to rules.json
/// (the config.yaml in `simplipy-assets/engines/dev_7-3/` has `rules: "./rules.json"`).
///
/// Operator ORDER matters: `SimpliPyEngine.__init__` uses the enumeration index as the precedence
/// fallback (`operator_precedence_compat[k] = v.get("precedence", i)`). A plain map would lose
/// that order, so we deserialize the operator block as an ORDER-PRESERVING `serde_yaml_ng::Mapping`
/// and rebuild the (order, specs) pair faithfully.
#[derive(serde::Deserialize)]
struct EngineConfig {
    operators: serde_yaml_ng::Mapping,
    // Schema completeness only, never read here: a config.yaml may carry a `rules:`
    // path, but rule RESOLUTION is the Python shim's job (the core receives rules text).
    #[allow(dead_code)]
    rules: Option<String>,
}

/// (insertion-ordered operator names, name -> spec), preserving config.yaml order.
type OperatorTable = (Vec<String>, FxHashMap<String, OperatorSpec>);

impl EngineConfig {
    fn into_operators(self) -> Result<OperatorTable, Box<dyn Error>> {
        let mut order = Vec::with_capacity(self.operators.len());
        let mut specs = FxHashMap::default();
        for (k, v) in self.operators {
            let name = k.as_str().ok_or("non-string operator key")?.to_string();
            let spec: OperatorSpec = serde_yaml_ng::from_value(v)?;
            order.push(name.clone());
            specs.insert(name, spec);
        }
        Ok((order, specs))
    }
}

pub struct Engine {
    operators: Operators,
    rules: CompiledRules,
    /// Per-Engine token interner: immutable under `&self`; extended append-only by the
    /// `&mut self` entry points (`set_rules`), so ids are stable for the Engine's lifetime.
    tokens: TokenTable,
    /// Memo for the `!`-sort match-time certificate (`interval::finite_ae` by subtree tokens).
    /// Per-ENGINE, not global: certificates depend on this engine's operator semantics, and
    /// tests build many engines with different op sets in one process. Generational (see
    /// `BangCache`): bounded memory, never stops memoizing.
    bang_cache: std::sync::Mutex<BangCache>,
    /// The `$`-sort twin (`interval::finite_nonzero_ae`), same discipline.
    mult_cache: std::sync::Mutex<BangCache>,
    /// The AC core's translated ruleset, built lazily on first `ac_simplify` (see `engine/ac.rs`)
    /// so consumers that never opt in pay nothing at construction.
    ac_rules_cell: std::sync::OnceLock<crate::ac::rules::AcRules>,
}

impl Engine {
    /// Build from resolved local paths (the Python shim resolves HF-hub/local via simplipy's own
    /// asset_manager and hands us files; the Rust core is network-free). REUSES the unchanged
    /// config.yaml + rules.json (single source of truth shared with Python).
    pub fn from_paths(
        config_yaml_path: &str,
        rules_json_path: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let cfg_text = fs::read_to_string(config_yaml_path)?;
        let rules_text = fs::read_to_string(rules_json_path)?;
        Self::from_strs(&cfg_text, &rules_text)
    }

    /// Build from in-memory config/rules text: direct `SimpliPyEngine(operators=..., rules=...)`
    /// construction serializes its in-memory state and attaches a core through here -- with no
    /// pure-Python fallback, every construction path must reach a compiled core without touching
    /// the filesystem.
    pub fn from_strs(cfg_text: &str, rules_text: &str) -> Result<Self, Box<dyn Error>> {
        let cfg: EngineConfig = serde_yaml_ng::from_str(cfg_text)?;

        // rules.json: a JSON list of [lhs_tokens, rhs_tokens] pairs; serde_json maps each
        // 2-element array onto the (Vec<String>, Vec<String>) tuple. Compile interns the
        // pairs in asset order (`CompiledRules::raw`); bucketing/gating happens at the lazy
        // AC translation (`ac::rules::AcRules::translate`), not here.
        let raw: Vec<(Vec<String>, Vec<String>)> = serde_json::from_str(rules_text)?;

        // Operators FIRST: their arity drives parsing each rule's lhs/rhs into a tree at compile.
        let (order, specs) = cfg.into_operators()?;
        // CORE-LANGUAGE GUARD: the serializers emit the core tokens with FIXED meanings
        // (`crate::operators::CORE_SERIALIZATION_OPS`), so a config may declare them --
        // supplying realizations/precedences -- but never repurpose them: a conflicting
        // arity or an alias shadowing a core token would silently change what the engine's
        // own output means. Fail loudly at build instead.
        for (name, spec) in &specs {
            if let Some(core_a) = crate::operators::core_arity(name) {
                if spec.arity != core_a {
                    return Err(format!(
                        "config error: {name:?} is a core serialization token with arity \
                         {core_a}, but the config declares arity {}; core tokens cannot \
                         be repurposed",
                        spec.arity
                    )
                    .into());
                }
            }
            // ...and its PRECEDENCE, for the same reason. The guard used to check arity
            // and aliases only, which left the closure breach open from the other side:
            // the realization rendering is a string PYTHON evaluates, and Python's
            // precedence is fixed. A config declaring `/` at 0.5 renders `(x0+x1)/x2` as
            // `x0 + x1 / x2` -- which this engine re-parses correctly, because its parser
            // shares the same table, while Python reads `x0 + (x1/x2)`: 2.5 where the
            // value is 2.0. All eight shipped precedences already match.
            if let Some(core_p) = crate::operators::core_precedence(name) {
                if spec.precedence.is_some_and(|p| p != core_p) {
                    return Err(format!(
                        "config error: {name:?} is a core serialization token with \
                         precedence {core_p}, but the config declares {}; core tokens \
                         cannot be repurposed (the realization rendering is evaluated by \
                         Python, whose precedence is fixed, so a divergent precedence \
                         silently changes the VALUE)",
                        spec.precedence.unwrap()
                    )
                    .into());
                }
            }
            for alias in &spec.alias {
                if crate::operators::core_arity(alias).is_some() {
                    return Err(format!(
                        "config error: operator {name:?} declares the alias {alias:?}, \
                         which is a core serialization token; core tokens cannot be \
                         shadowed by aliases"
                    )
                    .into());
                }
            }
        }
        let operators = Operators::from_specs(order.clone(), specs);
        // The per-Engine token table (operators + aliases + vocab), then compile interns
        // every rule token into it.
        let mut tokens = TokenTable::build(&order, &operators);
        let compiled = CompiledRules::compile(raw, &mut tokens, &operators);

        Ok(Self {
            operators,
            rules: compiled,
            tokens,
            bang_cache: std::sync::Mutex::new(BangCache::new()),
            mult_cache: std::sync::Mutex::new(BangCache::new()),
            ac_rules_cell: std::sync::OnceLock::new(),
        })
    }

    /// The per-call token view over this engine's table and the ctx's overlay.
    #[inline]
    fn view<'a>(&'a self, ctx: &'a SimplifyCtx) -> TokenView<'a> {
        TokenView::new(&self.tokens, &ctx.overlay)
    }

    /// Boundary helpers: intern a whole string expression / resolve a whole id expression.
    fn intern_seq(&self, tokens: &[String], ctx: &SimplifyCtx) -> Vec<Tok> {
        let view = self.view(ctx);
        tokens.iter().map(|s| view.intern(s)).collect()
    }

    fn resolve_seq(&self, toks: &[Tok], ctx: &SimplifyCtx) -> Vec<String> {
        let view = self.view(ctx);
        toks.iter().map(|&t| view.resolve_owned(t)).collect()
    }

    /// Crate-internal accessor for the operator tables (used by `crate::eval` tests/benches).
    #[allow(dead_code)] // test/bench-only today
    pub(crate) fn operators_ref(&self) -> &Operators {
        &self.operators
    }

    /// The compiled rule set (buckets + operand index). Exposed for the stage-(a) parity tests.
    #[allow(dead_code)] // test-only accessor (parity tests here and in rules.rs)
    pub(crate) fn rules(&self) -> &CompiledRules {
        &self.rules
    }

    /// The per-Engine token table. Exposed for the stage-(a) parity tests, which resolve
    /// compiled-rule ids back to strings.
    #[allow(dead_code)] // test-only accessor (parity tests here and in rules.rs)
    pub(crate) fn token_table(&self) -> &TokenTable {
        &self.tokens
    }

    /// `is_valid`: is the prefix expression syntactically valid
    /// (every operator has exactly its arity of operands, a single root remains)? Uses the plain
    /// `operator_arity` (NOT the sort `_compat` map, so `**` is treated as a leaf here, as in Python).
    /// The reversed scan only ever needs the stack DEPTH (the pushed tokens are never inspected).
    /// Stays on `&[String]`: it never touches the tree types, and it runs at the FFI boundary
    /// BEFORE interning (the validation gate for inputs that may be malformed).
    pub fn is_valid(&self, expression: &[String]) -> bool {
        // A multi-token expression must start with an operator.
        if expression.len() > 1 && !self.operators.is_operator(&expression[0]) {
            return false;
        }

        let mut depth: usize = 0;
        for token in expression.iter().rev() {
            // A numeric-looking token that is not `<constant>` must actually RESOLVE to a
            // value (catches malformed numerics like `--5` / `1e` that pass
            // `is_numeric_string`). Resolution goes through the evaluator's own leaf table
            // (`numeric::leaf_value`), so every literal the engine can evaluate -- floats
            // AND the AC core's exact `p/q` fractions -- validates by the same rule it is
            // later read by.
            if token != "<constant>"
                && crate::utils::is_numeric_string(token)
                && crate::numeric::leaf_value(token).is_none()
            {
                return false;
            }

            if let Some(arity) = self.operators.arity_of(token) {
                let arity = arity as usize;
                if depth < arity {
                    return false; // not enough operands
                }
                depth -= arity;
            }
            depth += 1; // push this token
        }

        depth == 1
    }

    /// `prefix_to_infix`: render a prefix token list to infix. `Err` mirrors Python's `ValueError`
    /// on a malformed prefix (too few / too many operands). No equal-precedence right-operand
    /// flattening, coordinated with `infix_to_prefix`/`parse` so prefix<->infix round-trips.
    pub fn prefix_to_infix(
        &self,
        tokens: &[String],
        power: crate::convert::Power,
        realization: bool,
    ) -> Result<String, String> {
        crate::convert::prefix_to_infix(tokens, &self.operators, power, realization)
    }

    /// `infix_to_prefix`: infix string -> prefix token list via a
    /// right-to-left shunting-yard. Never raises (matches Python on degenerate inputs).
    pub fn infix_to_prefix(&self, infix_expression: &str) -> Vec<String> {
        crate::convert::infix_to_prefix(infix_expression, &self.operators)
    }

    /// `convert_expression`: normalize into the engine's internal form. `Err` mirrors a Python
    /// raise (the dead float-division `int()` ValueError).
    pub fn convert_expression(&self, prefix_expr: &[String]) -> Result<Vec<String>, String> {
        crate::convert::convert_expression(prefix_expr, &self.operators)
    }

    /// Native f64 numeric constant folding: evaluate an all-numeric prefix
    /// subtree to a result token, or `None` if unfoldable. Mirrors `_evaluate_constant_subtree`.
    pub fn evaluate_constant_subtree(&self, tokens: &[String]) -> Option<String> {
        crate::numeric::evaluate_constant_subtree(tokens, &self.operators)
    }

    /// `parse`: infix string -> standardized prefix expression
    /// (infix_to_prefix -> convert_expression -> numbers_to_constant -> remove_pow1).
    pub fn parse(
        &self,
        infix_expression: &str,
        convert: bool,
        mask_numbers: bool,
    ) -> Result<Vec<String>, String> {
        crate::convert::parse(infix_expression, &self.operators, convert, mask_numbers)
    }

    /// `operators_to_realizations`: map each operator NAME to its
    /// realization (`sin` -> `simplipy.operators.sin`, `+` -> `+`); non-operator tokens are kept.
    pub fn operators_to_realizations(&self, expression: &[String]) -> Vec<String> {
        expression
            .iter()
            .map(|t| {
                self.operators
                    .operator_realizations
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| t.clone())
            })
            .collect()
    }

    /// `realizations_to_operators`: the inverse map (realization ->
    /// operator name); tokens not in the map are kept.
    pub fn realizations_to_operators(&self, expression: &[String]) -> Vec<String> {
        expression
            .iter()
            .map(|t| {
                self.operators
                    .realization_to_operator
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| t.clone())
            })
            .collect()
    }
}
