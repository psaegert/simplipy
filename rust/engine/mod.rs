//! The `Engine`: the whole-unit simplify kernel -- the Rust analogue of the removed pure-Python
//! `SimpliPyEngine.simplify` (engine.py@0.2.15:1770) and its callees, ported as ONE FFI unit (see lib.rs).
//! Faithful target: dev_7-3 @ simplipy 0.2.15 / 1fe9b7e, skeleton inputs, mpl in {4, 7}.
//! The kernel runs on interned `Tok` ids (see `crate::tokens`); the string boundary is the public
//! `&[String]` methods (intern once at entry, resolve once at exit). Submodules: [`stats`]
//! (hot-path counters), `memo` (cache/ctx state), `simplify` (the kernel), `miner` (the OFFLINE
//! mining surface); config deserialization, the `Engine` struct and constructors live here.

mod memo;
mod miner;
mod simplify;
pub mod stats;
#[cfg(test)]
mod tests;

use std::error::Error;
use std::fs;

use rustc_hash::FxHashMap;

use crate::operators::{OperatorSpec, Operators};
use crate::rules::CompiledRules;
use crate::tokens::{Tok, TokenTable, TokenView};

use memo::{BangCache, SimplifyCtx};

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
    #[allow(dead_code)]
    rules: Option<String>,
}

impl EngineConfig {
    /// (insertion-ordered operator names, name -> spec), preserving config.yaml order.
    fn into_operators(
        self,
    ) -> Result<(Vec<String>, FxHashMap<String, OperatorSpec>), Box<dyn Error>> {
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
    engine_id: String,
    /// Memo for the `!`-sort match-time certificate (`interval::finite_ae` by subtree tokens).
    /// Per-ENGINE, not global: certificates depend on this engine's operator semantics, and
    /// tests build many engines with different op sets in one process. Generational (see
    /// `BangCache`): bounded memory, never stops memoizing.
    bang_cache: std::sync::Mutex<BangCache>,
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
        // 2-element array onto the (Vec<String>, Vec<String>) tuple. Compile splits the list into
        // pattern/wildcard rules and explicit rules; no dev_7-3 pattern rule has a wildcard at
        // operand[0], so the first-operand index is fully applicable.
        let raw: Vec<(Vec<String>, Vec<String>)> = serde_json::from_str(rules_text)?;

        // Operators FIRST: their arity drives parsing each rule's lhs/rhs into a tree at compile.
        let (order, specs) = cfg.into_operators()?;
        let operators = Operators::from_specs(order.clone(), specs);
        // The per-Engine token table (operators + aliases + vocab), then compile interns
        // every rule token into it.
        let mut tokens = TokenTable::build(&order, &operators);
        let compiled = CompiledRules::compile(raw, &mut tokens, &operators);

        Ok(Self {
            operators,
            rules: compiled,
            tokens,
            engine_id: crate::FAITHFUL_ENGINE_ID.to_string(),
            bang_cache: std::sync::Mutex::new(BangCache::new()),
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
        toks.iter().map(|&t| view.to_string(t)).collect()
    }

    pub fn engine_id(&self) -> &str {
        &self.engine_id
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

    /// Faithful port of `is_valid` (engine.py@0.2.15:354): is the prefix expression syntactically valid
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
            // A numeric-looking token that is not `<constant>` must actually parse as a float
            // (catches malformed numerics like `--5` / `1e` that pass `is_numeric_string`). Rust
            // `f64::from_str` agrees with Python `float()` on every `is_numeric_string`-true token
            // (the underscore/whitespace divergences are filtered out by `is_numeric_string` itself).
            if token != "<constant>"
                && crate::utils::is_numeric_string(token)
                && token.parse::<f64>().is_err()
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

    /// Faithful port of the term-cancellation unit `cancel_terms(*collect_multiplicities(x))`
    /// (engine.py@0.2.15:1290 + 1410), as invoked once per `simplify` fixpoint iteration. Validated in
    /// isolation against fresh Python before the sort + fixpoint compose. Cancel is
    /// `max_pattern_length`-independent (no `mpl` argument).
    pub fn cancel_terms(&self, expression: &[String]) -> Vec<String> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(expression, &ctx);
        let out = crate::cancel::cancel_terms_unit(&toks, &self.operators, &self.view(&ctx));
        self.resolve_seq(&out, &ctx)
    }

    /// Faithful port of `sort_operands` (engine.py@0.2.15:1636) + `operand_key` (2512): the canonical
    /// commutative-operand ordering, the final stage of the `simplify` fixpoint (runs once, after the
    /// loop). Validated in isolation against fresh Python before the whole-unit compose.
    pub fn sort_operands(&self, expression: &[String]) -> Vec<String> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(expression, &ctx);
        let out = crate::sort::sort_operands_unit(&toks, &self.view(&ctx));
        self.resolve_seq(&out, &ctx)
    }

    /// Faithful port of `prefix_to_infix` (engine.py@0.2.15:409). `Err` mirrors Python's `ValueError` on a
    /// malformed prefix (too few / too many operands).
    pub fn prefix_to_infix(
        &self,
        tokens: &[String],
        power: crate::convert::Power,
        realization: bool,
    ) -> Result<String, String> {
        crate::convert::prefix_to_infix(tokens, &self.operators, power, realization, false)
    }

    /// Corrected (`fixed`) render: the #5 render half (no equal-precedence right-operand flattening),
    /// coordinated with `infix_to_prefix_fixed`/`parse_fixed` so prefix<->infix round-trips.
    pub fn prefix_to_infix_fixed(
        &self,
        tokens: &[String],
        power: crate::convert::Power,
        realization: bool,
    ) -> Result<String, String> {
        crate::convert::prefix_to_infix(tokens, &self.operators, power, realization, true)
    }

    /// Faithful port of `infix_to_prefix` (engine.py@0.2.15:581): infix string -> prefix token list via a
    /// right-to-left shunting-yard. Never raises (matches Python on degenerate inputs).
    pub fn infix_to_prefix(&self, infix_expression: &str) -> Vec<String> {
        crate::convert::infix_to_prefix(infix_expression, &self.operators, false)
    }

    /// Faithful port of `convert_expression` (engine.py@0.2.15:655). `Err` mirrors a Python raise (raw
    /// unconfigured `powN` KeyError; the dead float-division `int()` ValueError).
    pub fn convert_expression(&self, prefix_expr: &[String]) -> Result<Vec<String>, String> {
        crate::convert::convert_expression(prefix_expr, &self.operators, false)
    }

    /// Native f64 numeric constant folding (the `numeric` line): evaluate an all-numeric prefix
    /// subtree to a result token, or `None` if unfoldable. Mirrors `_evaluate_constant_subtree`.
    pub fn evaluate_constant_subtree(&self, tokens: &[String]) -> Option<String> {
        crate::numeric::evaluate_constant_subtree(tokens, &self.operators)
    }

    /// Faithful port of `parse` (engine.py@0.2.15:852): infix string -> standardized prefix expression
    /// (infix_to_prefix -> convert_expression -> numbers_to_constant -> remove_pow1).
    pub fn parse(
        &self,
        infix_expression: &str,
        convert: bool,
        mask_numbers: bool,
    ) -> Result<Vec<String>, String> {
        crate::convert::parse(
            infix_expression,
            &self.operators,
            convert,
            mask_numbers,
            false,
        )
    }

    /// Corrected (deliberate-improvement) variants of the conversion surface: the conversion-quirk
    /// fixes (#1 fractional power preserved, #2 `x**0`->`1`, #3 neg-of-literal toggles one minus,
    /// #4 `^` parses unary-minus like `**`, #6 raw `powN` no KeyError). NOT `dev_7-3` -- these back a
    /// future fixed engine-id (mirror of the Python `fix/conversion-quirks` branch).
    pub fn infix_to_prefix_fixed(&self, infix_expression: &str) -> Vec<String> {
        crate::convert::infix_to_prefix(infix_expression, &self.operators, true)
    }

    pub fn convert_expression_fixed(&self, prefix_expr: &[String]) -> Result<Vec<String>, String> {
        crate::convert::convert_expression(prefix_expr, &self.operators, true)
    }

    pub fn parse_fixed(
        &self,
        infix_expression: &str,
        convert: bool,
        mask_numbers: bool,
    ) -> Result<Vec<String>, String> {
        crate::convert::parse(
            infix_expression,
            &self.operators,
            convert,
            mask_numbers,
            true,
        )
    }

    /// Faithful port of `operators_to_realizations` (engine.py@0.2.15:2547): map each operator NAME to its
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

    /// Faithful port of `realizations_to_operators` (engine.py@0.2.15:2566): the inverse map (realization ->
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
