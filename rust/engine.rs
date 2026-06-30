//! The `Engine`: the whole-unit simplify kernel. This is the Rust analogue of
//! `SimpliPyEngine.simplify` (engine.py:1770) and its callees, ported as ONE FFI unit (see lib.rs).
//!
//! Faithful target: dev_7-3 @ simplipy 0.2.15 / 1fe9b7e, skeleton inputs, mpl in {4, 7}.

use std::error::Error;
use std::fs;

use rustc_hash::FxHashMap;

use crate::operators::{OperatorSpec, Operators};
use crate::parse::{parse_subtree, tree_to_prefix, Node};
use crate::rules::CompiledRules;
use crate::utils::{apply_mapping, match_pattern};

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
    engine_id: String,
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
        let cfg: EngineConfig = serde_yaml_ng::from_str(&cfg_text)?;

        // rules.json: a JSON list of [lhs_tokens, rhs_tokens] pairs (verified: top-level list,
        // each element [[..lhs..], [..rhs..]]; 114,000 raw rules ~23 MB -> compile splits into
        // 30,200 pattern/wildcard rules + 83,800 explicit rules; verified 0/30,200 pattern rules
        // have a wildcard at operand[0], so the first-operand index is fully applicable).
        // serde_json maps a JSON 2-element array onto the (Vec<String>, Vec<String>) tuple.
        let rules_text = fs::read_to_string(rules_json_path)?;
        let raw: Vec<(Vec<String>, Vec<String>)> = serde_json::from_str(&rules_text)?;

        // Operators FIRST: their arity drives parsing each rule's lhs/rhs into a tree at compile.
        let (order, specs) = cfg.into_operators()?;
        let operators = Operators::from_specs(order, specs);
        let compiled = CompiledRules::compile(raw, &|t| operators.arity_of(t));

        Ok(Self {
            operators,
            rules: compiled,
            engine_id: crate::FAITHFUL_ENGINE_ID.to_string(),
        })
    }

    pub fn engine_id(&self) -> &str {
        &self.engine_id
    }

    /// Crate-internal accessor for the operator tables (used by `crate::eval` tests/benches).
    #[allow(dead_code)] // test/bench-only today; an M3 entry point
    pub(crate) fn operators_ref(&self) -> &Operators {
        &self.operators
    }

    /// OFFLINE miner kernel (Phase B): vectorized evaluation of a prefix expression over a batch of
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

    /// OFFLINE miner (Phase B, M3): classify a candidate's degree in its `<constant>`s. See `crate::fit`.
    pub fn classify_linearity(&self, tokens: &[String]) -> Result<String, String> {
        crate::fit::classify(tokens, &self.operators).map(|l| l.as_str().to_string())
    }

    /// OFFLINE miner (Phase B, M3a): native `exist_constants_that_fit` for affine-in-params candidates
    /// (closed-form least squares + allclose). `None` for nonlinear-in-params (deferred to M3b).
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

    /// OFFLINE miner (Phase B, M3 complete): native `exist_constants_that_fit` -- affine candidates
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

    /// DEV micro-benchmark (not a shipped surface): compile the tape + take X ONCE, then run
    /// `repeats` resident evaluations over all rows (param[0] perturbed per iter so nothing is
    /// hoisted), returning elapsed seconds. Measures the M3 residual-loop cost (X resident in Rust),
    /// EXCLUDING the per-call FFI list marshaling -- the cost the in-Rust LM actually pays per
    /// residual evaluation.
    pub fn eval_bench_resident(
        &self,
        tokens: &[String],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        params: &[f64],
        repeats: usize,
    ) -> Result<f64, String> {
        let tape = crate::eval::Tape::compile(tokens, &self.operators, var_names)?;
        let n_vars = var_names.len();
        if x_flat.len() != n_rows * n_vars {
            return Err("x_flat shape mismatch".to_string());
        }
        if params.len() < tape.n_params {
            return Err("not enough params".to_string());
        }
        // Build the variable columns ONCE (the M3 LM would hold X resident, column-major).
        let cols = crate::eval::columns_from_row_major(x_flat, n_rows, n_vars);
        let mut p = params.to_vec();
        let p0 = if p.is_empty() { 0.0 } else { p[0] };
        let mut acc = 0.0f64;
        let t = std::time::Instant::now();
        for k in 0..repeats {
            if !p.is_empty() {
                p[0] = p0 + (k as f64) * 1e-12; // perturb so the loop is not hoisted
            }
            let out = tape.eval_columns(&cols, &p, n_rows);
            acc += out[0] + out[n_rows - 1];
        }
        let elapsed = t.elapsed().as_secs_f64();
        std::hint::black_box(acc);
        Ok(elapsed)
    }

    /// The compiled rule set (buckets + operand index). Exposed for the stage-(a) parity tests.
    pub fn rules(&self) -> &CompiledRules {
        &self.rules
    }

    /// Parse a flat prefix token slice into a tree using this engine's operator arities.
    fn parse_prefix(&self, tokens: &[String]) -> Node {
        parse_subtree(tokens, 0, &|t| self.operators.arity_of(t)).0
    }

    /// Faithful port of `apply_simplifcation_rules` (engine.py:1252): the whole-expression
    /// all-`<constant>`/operator fold, then parse -> `apply_rules_top_down` -> flatten back to
    /// prefix. This is the rule-application sub-unit (the `simplify` fixpoint's per-iteration rule
    /// pass); validated in isolation against fresh Python `apply_simplifcation_rules`.
    pub fn apply_simplification_rules(
        &self,
        expression: &[String],
        max_pattern_length: Option<usize>,
        fold: bool,
    ) -> Vec<String> {
        if expression
            .iter()
            .all(|t| t == "<constant>" || self.operators.is_operator(t))
        {
            return vec!["<constant>".to_string()];
        }
        let tree = self.parse_prefix(expression);
        let simplified = self.apply_rules_top_down(tree, max_pattern_length, fold);
        let mut out = Vec::new();
        tree_to_prefix(&simplified, &mut out);
        out
    }

    /// Faithful port of `is_valid` (engine.py:354): is the prefix expression syntactically valid
    /// (every operator has exactly its arity of operands, a single root remains)? Uses the plain
    /// `operator_arity` (NOT the sort `_compat` map, so `**` is treated as a leaf here, as in Python).
    /// The reversed scan only ever needs the stack DEPTH (the pushed tokens are never inspected).
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
    /// (engine.py:1290 + 1410), as invoked once per `simplify` fixpoint iteration. Validated in
    /// isolation against fresh Python before the sort + fixpoint compose. Cancel is
    /// `max_pattern_length`-independent (no `mpl` argument).
    pub fn cancel_terms(&self, expression: &[String]) -> Vec<String> {
        crate::cancel::cancel_terms_unit(expression, &self.operators)
    }

    /// Faithful port of `sort_operands` (engine.py:1636) + `operand_key` (2512): the canonical
    /// commutative-operand ordering, the final stage of the `simplify` fixpoint (runs once, after the
    /// loop). Validated in isolation against fresh Python before the whole-unit compose.
    pub fn sort_operands(&self, expression: &[String]) -> Vec<String> {
        crate::sort::sort_operands_unit(expression, &self.operators)
    }

    /// Faithful port of `prefix_to_infix` (engine.py:409). `Err` mirrors Python's `ValueError` on a
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

    /// Faithful port of `infix_to_prefix` (engine.py:581): infix string -> prefix token list via a
    /// right-to-left shunting-yard. Never raises (matches Python on degenerate inputs).
    pub fn infix_to_prefix(&self, infix_expression: &str) -> Vec<String> {
        crate::convert::infix_to_prefix(infix_expression, &self.operators, false)
    }

    /// Faithful port of `convert_expression` (engine.py:655). `Err` mirrors a Python raise (raw
    /// unconfigured `powN` KeyError; the dead float-division `int()` ValueError).
    pub fn convert_expression(&self, prefix_expr: &[String]) -> Result<Vec<String>, String> {
        crate::convert::convert_expression(prefix_expr, &self.operators, false)
    }

    /// Native f64 numeric constant folding (the `numeric` line): evaluate an all-numeric prefix
    /// subtree to a result token, or `None` if unfoldable. Mirrors `_evaluate_constant_subtree`.
    pub fn evaluate_constant_subtree(&self, tokens: &[String]) -> Option<String> {
        crate::numeric::evaluate_constant_subtree(tokens, &self.operators)
    }

    /// Faithful port of `parse` (engine.py:852): infix string -> standardized prefix expression
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

    /// Faithful port of `operators_to_realizations` (engine.py:2547): map each operator NAME to its
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

    /// Faithful port of `realizations_to_operators` (engine.py:2566): the inverse map (realization ->
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

    /// Faithful port of `apply_rules_top_down` (engine.py:1025), the deployed (no-statistics) path:
    /// per-node all-`<constant>` fold; exact-rule lookup; pattern scan longest-length-first
    /// (first-match-wins, via the operand[0] index); else recurse into operands and re-check exact +
    /// pattern rules on the rebuilt node.
    ///
    /// `fold` selects the engine line. `false` = faithful `dev_7-3`: the early per-node
    /// all-`<constant>` collapse runs BEFORE the rule scan. `true` = the numeric line: that early
    /// collapse is REMOVED and folding runs as a FALLBACK after each rule scan (`try_fold_constants`
    /// at the two sites = engine.py:1432/1480), so a rule that matches an all-`<constant>` subtree
    /// is tried before the subtree is collapsed (the position change the numeric branch introduced).
    fn apply_rules_top_down(
        &self,
        node: Node,
        max_pattern_length: Option<usize>,
        fold: bool,
    ) -> Node {
        let operands = match &node {
            Node::Leaf(_) => return node,
            Node::Op { operands, .. } => operands,
        };
        // Faithful per-node fold (engine.py tag:1059): all DIRECT operands `<constant>` -> collapse.
        // The numeric line removes this (folds as a post-rule fallback instead).
        if !fold
            && operands
                .iter()
                .all(|o| matches!(o, Node::Leaf(t) if t == "<constant>"))
        {
            return Node::Leaf("<constant>".to_string());
        }
        let operator = node.root_token();

        // Exact (no-pattern) rule lookup on the flat prefix.
        let mut flat = Vec::new();
        tree_to_prefix(&node, &mut flat);
        let subtree_length = flat.len();
        if let Some(replacement) = self.rules.no_patterns.get(&flat) {
            let parsed = self.parse_prefix(replacement);
            return self.apply_rules_top_down(parsed, max_pattern_length, fold);
        }

        let subtree_max_pl = match max_pattern_length {
            None => subtree_length.min(self.rules.max_pattern_length),
            Some(m) => m.min(subtree_length).min(self.rules.max_pattern_length),
        };

        // Pattern scan, largest pattern length first; candidates via the operand[0] index.
        let head = operands[0].root_token();
        for pattern_length in (1..=subtree_max_pl).rev() {
            for rule in self.rules.candidates(pattern_length, operator, head) {
                let mut mapping: FxHashMap<String, &Node> = FxHashMap::default();
                if match_pattern(&node, &rule.lhs_tree, &mut mapping) {
                    let replacement = apply_mapping(&rule.rhs_tree, &mapping);
                    return self.apply_rules_top_down(replacement, max_pattern_length, fold);
                }
            }
        }

        // Numeric line: no rule matched -> try constant folding as fallback (engine.py:1432).
        if fold {
            if let Some(folded) = self.try_fold_constants(operator, operands) {
                return folded;
            }
        }

        // No rule at this node: recurse into operands and rebuild.
        let simplified_operands: Vec<Node> = operands
            .iter()
            .map(|o| self.apply_rules_top_down(o.clone(), max_pattern_length, fold))
            .collect();
        let simplified = Node::Op {
            token: operator.to_string(),
            operands: simplified_operands,
        };

        // Re-check exact rules on the rebuilt node.
        let mut flat2 = Vec::new();
        tree_to_prefix(&simplified, &mut flat2);
        if let Some(replacement) = self.rules.no_patterns.get(&flat2) {
            let parsed = self.parse_prefix(replacement);
            return self.apply_rules_top_down(parsed, max_pattern_length, fold);
        }

        // Re-check pattern rules on the rebuilt node.
        let head2 = match &simplified {
            Node::Op { operands, .. } => operands[0].root_token(),
            Node::Leaf(_) => unreachable!(),
        };
        for pattern_length in (1..=subtree_max_pl).rev() {
            for rule in self.rules.candidates(pattern_length, operator, head2) {
                let mut mapping: FxHashMap<String, &Node> = FxHashMap::default();
                if match_pattern(&simplified, &rule.lhs_tree, &mut mapping) {
                    let replacement = apply_mapping(&rule.rhs_tree, &mapping);
                    return self.apply_rules_top_down(replacement, max_pattern_length, fold);
                }
            }
        }

        // Numeric line: no rule after operand simplification -> fold fallback (engine.py:1480).
        if fold {
            if let Node::Op {
                operands: simp_ops, ..
            } = &simplified
            {
                if let Some(folded) = self.try_fold_constants(operator, simp_ops) {
                    return folded;
                }
            }
        }

        simplified
    }

    /// Port of `_try_fold_constants` (engine.py:1314, the numeric line): fold a subtree whose
    /// operands are ALL leaves. If every operand is numeric -> evaluate to the `f64` result token
    /// (`evaluate_constant_subtree`; `None` if unfoldable, e.g. a complex/unparseable result). ELSE
    /// if every operand is numeric / `<constant>` / `np.e` / `np.pi` -> collapse to `<constant>`.
    /// The `if all_numeric { ... } elif all_const_or_num { ... }` ORDER is load-bearing: an
    /// all-numeric but UNFOLDABLE subtree returns `None` (it does NOT fall through to the
    /// `<constant>` collapse).
    fn try_fold_constants(&self, operator: &str, operands: &[Node]) -> Option<Node> {
        let mut values: Vec<&str> = Vec::with_capacity(operands.len());
        for o in operands {
            match o {
                Node::Leaf(t) => values.push(t.as_str()),
                Node::Op { .. } => return None, // not all leaves
            }
        }
        let all_numeric = values.iter().all(|v| crate::utils::is_numeric_string(v));
        if all_numeric {
            let mut flat: Vec<String> = Vec::with_capacity(values.len() + 1);
            flat.push(operator.to_string());
            flat.extend(values.iter().map(|v| v.to_string()));
            return crate::numeric::evaluate_constant_subtree(&flat, &self.operators)
                .map(Node::Leaf);
        }
        let all_const_or_num = values.iter().all(|v| {
            crate::utils::is_numeric_string(v)
                || *v == "<constant>"
                || *v == "np.e"
                || *v == "np.pi"
        });
        if all_const_or_num {
            return Some(Node::Leaf("<constant>".to_string()));
        }
        None
    }

    /// THE whole-unit kernel. Faithful port of the `simplify` fixpoint (engine.py:1815-1924), the
    /// prefix-token-list contract: per iteration `cancel_terms` -> `apply_simplification_rules`
    /// (when enabled), break when the iteration is a no-op vs the previous (`<= max_iter`); then
    /// `sort_operands`; then `mask_elementary_literals` (when enabled); then the LONGER-RESULT GUARD
    /// (engine.py:1886/1893: if the result is longer than the original input, return the ORIGINAL).
    ///
    /// Returns the simplified prefix tokens (the Python `'list'` return). The `inplace` /
    /// return-type machinery (str/tuple/np_array) is a Python-shim concern, not part of this kernel.
    pub fn simplify(
        &self,
        tokens: &[String],
        max_iter: usize,
        max_pattern_length: Option<usize>,
        mask_elementary_literals: bool,
        apply_simplification_rules: bool,
        fold: bool,
    ) -> Vec<String> {
        let length_before = tokens.len();

        // current_expression / new_expression both start as a copy of the input.
        let mut current_expression = tokens.to_vec();
        let mut new_expression = current_expression.clone();

        for _ in 0..max_iter {
            // Cancel any terms (cancel_terms(*collect_multiplicities(new_expression))).
            new_expression = crate::cancel::cancel_terms_unit(&new_expression, &self.operators);

            // Apply simplification rules.
            if apply_simplification_rules {
                new_expression =
                    self.apply_simplification_rules(&new_expression, max_pattern_length, fold);
            }

            // Converged: this iteration produced no change vs the previous iteration's result.
            if new_expression == current_expression {
                break;
            }
            current_expression = new_expression.clone();
        }

        // Sort operands (once, after the loop).
        new_expression = crate::sort::sort_operands_unit(&new_expression, &self.operators);

        // Mask elementary literals (0/1/coefficients -> <constant>).
        if mask_elementary_literals {
            new_expression = crate::utils::mask_elementary_literals(&new_expression);
        }

        // Longer-result guard: a result longer than the input is not a simplification.
        if new_expression.len() > length_before {
            return tokens.to_vec();
        }

        new_expression
    }
}

#[cfg(test)]
mod tests {
    use crate::Engine;
    use std::fs;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    fn load(name: &str) -> Vec<Vec<String>> {
        let p = format!("{}/benchmarks/corpus/{}", env!("CARGO_MANIFEST_DIR"), name);
        serde_json::from_str(&fs::read_to_string(p).expect("corpus fixture present")).unwrap()
    }

    /// Whole-unit gate (a self-contained slice of benchmarks/diff_simplify.py): the composed Rust
    /// `simplify` fixpoint reproduces the frozen dev_7-3 reference on the first 400 corpus skeletons
    /// at mpl=4 AND mpl=7. The full 10000/10000 (both mpl) + 3000-row fresh-Python cross-check live
    /// in the harness; this pins it in CI.
    #[test]
    fn simplify_matches_frozen_reference() {
        // Skip when the (multi-MB, not-vendored) frozen corpus is absent -- the full 10k parity runs
        // in the benchmark harness; this in-crate slice only fires where the fixtures are staged.
        let corpus = format!(
            "{}/benchmarks/corpus/raw_skeletons.json",
            env!("CARGO_MANIFEST_DIR")
        );
        if !std::path::Path::new(&corpus).exists() {
            eprintln!("simplify_matches_frozen_reference: SKIPPED (corpus fixtures not vendored)");
            return;
        }
        let Some(e) = engine() else { return };
        let raw = load("raw_skeletons.json");
        for mpl in [4usize, 7usize] {
            let reference = load(&format!("reference_dev_7-3_mpl{mpl}.json"));
            assert_eq!(raw.len(), reference.len());
            let mut n_changed = 0;
            for (s, r) in raw.iter().zip(reference.iter()).take(400) {
                let out = e.simplify(s, 5, Some(mpl), true, true, false);
                if &out != s {
                    n_changed += 1;
                }
                assert_eq!(&out, r, "mpl={mpl} input={s:?}");
            }
            assert!(
                n_changed > 300,
                "expected most rows to simplify (got {n_changed}/400)"
            );
        }
    }

    /// `is_valid` (engine.py:354) canonical cases, cross-checked against fresh Python (the ~248k-input
    /// corpus+fuzz gate lives in benchmarks/diff_is_valid.py). Pins each reject path + the numeric guard.
    #[test]
    fn is_valid_cases() {
        let Some(e) = engine() else { return };
        let valid: &[&[&str]] = &[
            &["x1"],
            &["<constant>"],
            &["+", "x1", "x2"],
            &["sin", "+", "x1", "x1"],
            &["+", "<constant>", "<constant>"],
            &["+", "0", "1"],
            &["+", "sin", "x1", "neg", "x2"],
        ];
        let invalid: &[&[&str]] = &[
            &[],                      // empty
            &["+"],                   // operator, no operands
            &["sin"],                 // unary, no operand
            &["+", "x1"],             // not enough operands
            &["+", "x1", "x2", "x3"], // leftover stack
            &["x1", "x2"],            // multi-token starting with a leaf
            &["--5"],                 // numeric-looking but float() raises
            &["1e"],                  // ditto
            &["+", "x1", "--5"],      // bad numeric operand
        ];
        for v in valid {
            let t: Vec<String> = v.iter().map(|s| s.to_string()).collect();
            assert!(e.is_valid(&t), "expected valid: {v:?}");
        }
        for v in invalid {
            let t: Vec<String> = v.iter().map(|s| s.to_string()).collect();
            assert!(!e.is_valid(&t), "expected invalid: {v:?}");
        }
    }

    /// `operators_to_realizations` / `realizations_to_operators` (engine.py:2547/2566): operator
    /// names <-> realizations, non-operator tokens untouched, round-trip on canonical prefix.
    #[test]
    fn realizations_round_trip() {
        let Some(e) = engine() else { return };
        let t = |s: &[&str]| -> Vec<String> { s.iter().map(|x| x.to_string()).collect() };
        let expr = t(&["*", "neg", "x1", "pow2", "<constant>"]);
        let fwd = e.operators_to_realizations(&expr);
        assert_eq!(
            fwd,
            t(&[
                "*",
                "simplipy.operators.neg",
                "x1",
                "simplipy.operators.pow2",
                "<constant>"
            ])
        );
        assert_eq!(e.realizations_to_operators(&fwd), expr); // round-trips on canonical names
                                                             // non-operator tokens (vars / numerics / <constant>) are passed through unchanged.
        assert_eq!(
            e.operators_to_realizations(&t(&["x1", "0", "<constant>"])),
            t(&["x1", "0", "<constant>"])
        );
    }
}
