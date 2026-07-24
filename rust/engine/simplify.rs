//! The simplify kernel: the `!`-certificate lookup, `apply_rules_top_down` (with the
//! constant-fold fallback), and the `simplify` fixpoint, all running on interned ids against
//! the per-call [`SimplifyCtx`].

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::matcher::{apply_mapping, match_pattern_with_cert};
use crate::parse::{parse_subtree, tree_to_prefix, Node};
use crate::tokens::Tok;

use super::memo::SimplifyCtx;
use super::stats;
use super::Engine;

/// One memo-wrapped pass of the fixpoint: return the cached output for `input`, or run
/// `compute` and memoize its result (whole-pass input -> output map; see [`SimplifyCtx`]).
fn memoized_pass(
    memo: &RefCell<FxHashMap<Vec<Tok>, Vec<Tok>>>,
    input: &[Tok],
    compute: impl FnOnce() -> Vec<Tok>,
) -> Vec<Tok> {
    if let Some(out) = memo.borrow().get(input) {
        return out.clone();
    }
    let out = compute();
    memo.borrow_mut().insert(input.to_vec(), out.clone());
    out
}

/// Node budget for the cancel/rules search ([`Engine::simplify_search`]): the maximum number of
/// states EXPANDED (popped and given successors) before the search stops and returns the best
/// state it has seen. The distribution is extremely skewed -- on the 64k v23.0 prior the median
/// expression needs 2 expansions and 53% have no cancellation candidate at all, but the p99 is
/// ~186 and a handful of rows would run indefinitely -- so the budget bounds that tail rather
/// than the typical case. `SIMPLIPY_SEARCH_BUDGET` overrides it (0 = greedy only).
fn search_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("SIMPLIPY_SEARCH_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

impl Engine {
    /// The `!`-sort certificate behind the memos: defined-and-finite a.e. per
    /// `interval::finite_ae`. Variable leaves never reach here (match_pattern binds them
    /// freely); everything uncertifiable is simply not bound (fail-closed).
    /// Lookup order: the per-call scratch first (same subtree re-certified across
    /// candidates/scans within one pass), then the generational per-engine memo, then compute.
    /// Memo keys are `Vec<Tok>` -- NO string resolution on the lookup paths. The per-Engine
    /// cache only sees TABLE ids (an overlay-bearing subtree is memoized per-call only);
    /// `finite_ae` itself keeps its `&[String]` interface and is resolved-to at the (rare)
    /// actual-compute boundary.
    fn bang_certified(&self, node: &Node, ctx: &SimplifyCtx) -> bool {
        let scratch = &ctx.cert_scratch;
        let t0 = std::time::Instant::now();
        stats::bump(&stats::CERT_CALLS);
        let mut flat = Vec::new();
        tree_to_prefix(node, &mut flat);
        if let Some(&b) = scratch.borrow().get(&flat) {
            stats::bump(&stats::CERT_HITS);
            stats::add(&stats::NANOS_CERT, t0.elapsed().as_nanos() as u64);
            return b;
        }
        let table_only = flat.iter().all(|&t| self.tokens.is_table_id(t));
        if table_only {
            if let Some(b) = self.bang_cache.lock().unwrap().get_promoting(&flat) {
                scratch.borrow_mut().insert(flat, b);
                stats::bump(&stats::CERT_HITS);
                stats::add(&stats::NANOS_CERT, t0.elapsed().as_nanos() as u64);
                return b;
            }
        }
        let flat_strs = self.resolve_seq(&flat, ctx);
        let b = crate::interval::finite_ae(&flat_strs, &self.operators);
        if table_only {
            self.bang_cache.lock().unwrap().insert(flat.clone(), b);
        }
        scratch.borrow_mut().insert(flat, b);
        stats::add(&stats::NANOS_CERT, t0.elapsed().as_nanos() as u64);
        b
    }

    /// Parse a flat prefix id slice into a tree using this engine's operator arities.
    fn parse_prefix(&self, tokens: &[Tok], ctx: &SimplifyCtx) -> Node {
        let view = self.view(ctx);
        parse_subtree(tokens, 0, &|t| view.arity(t)).0
    }

    /// The rule-application sub-unit `apply_simplification_rules`: the whole-expression
    /// all-`<constant>`/operator fold, then parse -> `apply_rules_top_down` -> flatten back to
    /// prefix. This is the `simplify` fixpoint's per-iteration rule pass.
    pub fn apply_simplification_rules(
        &self,
        expression: &[String],
        max_pattern_length: Option<usize>,
    ) -> Vec<String> {
        let ctx = SimplifyCtx::new(self.tokens.len(), false);
        let toks = self.intern_seq(expression, &ctx);
        let out = self.apply_simplification_rules_with_ctx(&toks, max_pattern_length, &ctx);
        self.resolve_seq(&out, &ctx)
    }

    /// The rules pass against a caller-owned memo context: `simplify` threads ONE ctx
    /// across all fixpoint iterations so the confirming pass and unchanged subtrees are free.
    fn apply_simplification_rules_with_ctx(
        &self,
        expression: &[Tok],
        max_pattern_length: Option<usize>,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        let view = self.view(ctx);
        if expression
            .iter()
            .all(|&t| t == self.tokens.constant || view.is_operator(t))
        {
            return vec![self.tokens.constant];
        }
        let tree = self.parse_prefix(expression, ctx);
        let simplified = self.apply_rules_top_down(tree, max_pattern_length, ctx);
        let mut out = Vec::new();
        tree_to_prefix(&simplified, &mut out);
        out
    }

    /// The shared fire-site protocol of `apply_rules_top_down`, identical at both scan sites:
    /// exact (no-pattern) rule lookup on the flat prefix FIRST, then the pattern scan, largest
    /// pattern length first, candidates in asset order via the operand[0] index, first match
    /// wins (ONE mapping per scan, `clear()`ed per attempt for capacity reuse; the matcher
    /// only requires an EMPTY map).
    /// Returns the replacement node (exact: parsed rhs; pattern: mapped rhs) for the caller to
    /// recurse on, or `None` when no rule fires.
    fn try_rule_rewrite(
        &self,
        node: &Node,
        flat: &[Tok],
        subtree_max_pl: usize,
        ctx: &SimplifyCtx,
    ) -> Option<Node> {
        if let Some(replacement) = self.rules.exact_rules.get(flat) {
            stats::bump(&stats::EXACT_HITS);
            return Some(self.parse_prefix(replacement, ctx));
        }

        let view = self.view(ctx);
        let operator = node.root_token();
        let head = match node {
            Node::Op { operands, .. } => operands[0].root_token(),
            Node::Leaf(_) => unreachable!(),
        };
        let mut mapping: FxHashMap<Tok, &Node> = FxHashMap::default();
        for pattern_length in (1..=subtree_max_pl).rev() {
            for rule in self.rules.candidates(pattern_length, operator, head) {
                stats::bump(&stats::PATTERN_ATTEMPTS);
                mapping.clear();
                if match_pattern_with_cert(
                    node,
                    &rule.lhs_tree,
                    &mut mapping,
                    Some(&|n: &Node| self.bang_certified(n, ctx)),
                    ctx.wildcard_all,
                    &view,
                ) {
                    stats::bump(&stats::PATTERN_FIRES);
                    return Some(apply_mapping(&rule.rhs_tree, &mapping, &view));
                }
            }
        }
        None
    }

    /// `apply_rules_top_down`, the deployed (no-statistics) path:
    /// exact-rule lookup; pattern scan longest-length-first (first-match-wins, via the
    /// operand[0] index); constant folding as a FALLBACK after each rule scan
    /// (`try_fold_constants` at the two sites), so a rule that matches an all-`<constant>`
    /// subtree is tried before the subtree is collapsed; else recurse into operands and
    /// re-check exact + pattern rules (and the fold fallback) on the rebuilt node.
    fn apply_rules_top_down(
        &self,
        node: Node,
        max_pattern_length: Option<usize>,
        ctx: &SimplifyCtx,
    ) -> Node {
        let operands = match &node {
            Node::Leaf(_) => return node,
            Node::Op { operands, .. } => operands,
        };
        let operator = node.root_token();

        let mut flat = Vec::new();
        tree_to_prefix(&node, &mut flat);
        let subtree_length = flat.len();
        // A subtree this call already walked to a fixpoint (no fire anywhere inside)
        // cannot fire now -- return it untouched. Only the changed spine gets re-scanned.
        if ctx.normal_forms.borrow().contains(&flat) {
            return node;
        }

        let subtree_max_pl = match max_pattern_length {
            None => subtree_length.min(self.rules.max_pattern_length),
            Some(m) => m.min(subtree_length).min(self.rules.max_pattern_length),
        };

        // First fire site: exact lookup + pattern scan on the node as-is.
        if let Some(replacement) = self.try_rule_rewrite(&node, &flat, subtree_max_pl, ctx) {
            return self.apply_rules_top_down(replacement, max_pattern_length, ctx);
        }

        // No rule matched -> try constant folding as fallback.
        if let Some(folded) = self.try_fold_constants(operator, operands, ctx) {
            return folded;
        }

        // No rule at this node: recurse into operands and rebuild. `node` is owned -- consume its
        // operand vec instead of deep-cloning every subtree (behaviour-identical).
        let owned_operands = match node {
            Node::Op { operands, .. } => operands,
            Node::Leaf(_) => unreachable!(),
        };
        let simplified_operands: Vec<Node> = owned_operands
            .into_iter()
            .map(|o| self.apply_rules_top_down(o, max_pattern_length, ctx))
            .collect();
        let simplified = Node::Op {
            token: operator,
            operands: simplified_operands,
        };

        // Second fire site: re-check exact + pattern rules on the rebuilt node.
        let mut flat2 = Vec::new();
        tree_to_prefix(&simplified, &mut flat2);
        if let Some(replacement) = self.try_rule_rewrite(&simplified, &flat2, subtree_max_pl, ctx) {
            return self.apply_rules_top_down(replacement, max_pattern_length, ctx);
        }

        // No rule after operand simplification -> fold fallback.
        if let Node::Op {
            operands: simp_ops, ..
        } = &simplified
        {
            if let Some(folded) = self.try_fold_constants(operator, simp_ops, ctx) {
                return folded;
            }
        }

        // `simplified` survived every fire site -> normal form for this call.
        ctx.normal_forms.borrow_mut().insert(flat2);
        simplified
    }

    /// `_try_fold_constants`, extended with NaN
    /// literal propagation (works on ANY operand shapes): fold a subtree whose operands are
    /// ALL leaves. If every operand is VALUED (`numeric::leaf_value` resolves it: numeric
    /// literals, `np.pi`/`np.e`, `(-1)`-style parenthesized literals, the `float("...")`
    /// tokens) -> evaluate to the `f64` result token (`evaluate_constant_subtree`; `None` if
    /// unfoldable). ELSE if every operand is `<constant>` or a FINITE-valued leaf -> collapse to
    /// `<constant>` (inf/nan literals never absorb into `<constant>`: non-finite algebra belongs
    /// to explicit rules such as `+ float("-inf") <constant> -> float("-inf")`).
    /// The `if all_valued { ... } elif ... { ... }` ORDER is load-bearing: an all-valued but
    /// UNFOLDABLE subtree returns `None` (it does NOT fall through to the `<constant>` collapse).
    /// Both gates MUST use `numeric::leaf_value` -- the tape evaluator's own leaf table -- so the
    /// folder and the evaluator agree about which subtrees are constant.
    ///
    /// The leaf-value gates run on the per-id property; strings are only materialized for the
    /// (rare, actual-fold) `evaluate_constant_subtree` boundary call.
    fn try_fold_constants(
        &self,
        operator: Tok,
        operands: &[Node],
        ctx: &SimplifyCtx,
    ) -> Option<Node> {
        let view = self.view(ctx);
        // NaN LITERAL PROPAGATION: IEEE-754 maps a NaN operand to NaN through every
        // operator in this set REGARDLESS of the other operand -- except `pow`, whose
        // C99 edge cases pow(1, NaN) = 1 and pow(NaN, 0) = 1 depend on the other
        // operand's value, so `pow` never propagates structurally (when both operands
        // are literals the all-valued fold below computes those cases exactly). This is
        // operator-table knowledge, not sampling: it collapses `* <constant> nan`,
        // `* x0 nan` and every composition -- rewrites the evidence-based miner PROVABLY
        // cannot certify, because an everywhere-NaN source has no finite evidence (and
        // sampling must never claim all-NaN: a finite pocket at a single point is
        // invisible to random rows).
        if operator != self.tokens.pow
            && operands
                .iter()
                .any(|o| matches!(o, Node::Leaf(t) if *t == self.tokens.nan))
        {
            return Some(Node::Leaf(self.tokens.nan));
        }
        let mut values: Vec<Tok> = Vec::with_capacity(operands.len());
        for o in operands {
            match o {
                Node::Leaf(t) => values.push(*t),
                Node::Op { .. } => return None, // not all leaves
            }
        }
        let all_valued = values.iter().all(|&v| view.leaf_value(v).is_some());
        if all_valued {
            let mut flat: Vec<String> = Vec::with_capacity(values.len() + 1);
            flat.push(view.to_string(operator));
            flat.extend(values.iter().map(|&v| view.to_string(v)));
            return crate::numeric::evaluate_constant_subtree(&flat, &self.operators)
                .map(|s| Node::Leaf(view.intern(&s)));
        }
        let all_const_or_finite = values
            .iter()
            .all(|&v| v == self.tokens.constant || view.leaf_value(v).is_some_and(f64::is_finite));
        if all_const_or_finite {
            return Some(Node::Leaf(self.tokens.constant));
        }
        None
    }

    /// THE whole-unit kernel: the `simplify` fixpoint, the
    /// prefix-token-list contract: per iteration `cancel_terms` -> `apply_simplification_rules`
    /// (when enabled), break when the iteration is a no-op vs the previous (`<= max_iter`);
    /// the loop then yields the BEST (shortest, ties -> later) iterate rather than the endpoint --
    /// every iterate is a.e.-equivalent, and a cancel emit may grow transiently betting on rule
    /// folding the ruleset cannot always pay off; then `mask_elementary_literals` (when enabled);
    /// then `sort_operands` (mask-BEFORE-sort so the canonical operand order is a fixpoint --
    /// idempotent); then the LONGER-RESULT GUARD (if the result is longer than the original
    /// input, return the ORIGINAL -- with best-iterate tracking this is an unreachable
    /// defensive invariant, since the input itself is iterate zero).
    ///
    /// Returns the simplified prefix tokens (the Python `'list'` return). The `inplace` /
    /// return-type machinery (str/tuple/np_array) is a Python-shim concern, not part of this kernel.
    ///
    /// This is the string boundary -- intern once at entry, resolve once at exit; the whole
    /// fixpoint runs on `Tok` ids ([`Engine::simplify_toks`]).
    pub fn simplify(
        &self,
        tokens: &[String],
        max_iter: usize,
        max_pattern_length: Option<usize>,
        mask_elementary_literals: bool,
        apply_simplification_rules: bool,
        wildcard_all: bool,
    ) -> Vec<String> {
        let ctx = SimplifyCtx::new(self.tokens.len(), wildcard_all);
        let toks = self.intern_seq(tokens, &ctx);
        let out = self.simplify_toks(
            &toks,
            max_iter,
            max_pattern_length,
            mask_elementary_literals,
            apply_simplification_rules,
            &ctx,
        );
        self.resolve_seq(&out, &ctx)
    }

    /// The id-level simplify fixpoint (see [`Engine::simplify`] for the contract).
    fn simplify_toks(
        &self,
        tokens: &[Tok],
        max_iter: usize,
        max_pattern_length: Option<usize>,
        mask_elementary_literals: bool,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        let length_before = tokens.len();
        stats::bump(&stats::SIMPLIFY_CALLS);

        // SEARCH over the cancel/rules move graph (see `simplify_search`): one mechanism
        // replacing the former hand-picked candidate policies.
        let mut new_expression = self.simplify_search(
            tokens,
            max_iter,
            max_pattern_length,
            apply_simplification_rules,
            ctx,
        );

        // Mask elementary literals (0/1/coefficients -> <constant>) BEFORE sorting, so the final
        // operand order is computed on the canonical (masked) tokens. Masking still runs AFTER the
        // rule loop, so which rules fire is unchanged; this only reorders the operands of
        // masked-literal cases and makes sort/mask a fixpoint -- fixing the sort-then-mask
        // non-idempotency (a literal's post-sort position differed from <constant>'s, so a re-pass
        // re-sorted the now-masked token).
        let t_post = std::time::Instant::now();
        if mask_elementary_literals {
            new_expression =
                crate::utils::mask_elementary_literals(&new_expression, &self.view(ctx));
        }

        // Sort operands (once, after masking).
        new_expression = crate::sort::sort_operands_unit(&new_expression, &self.view(ctx));
        stats::add(&stats::NANOS_MASK_SORT, t_post.elapsed().as_nanos() as u64);

        // Longer-result guard: a result longer than the input is not a simplification.
        // With best-iterate tracking (`simplify_trajectory`) this is an unreachable defensive
        // invariant -- the input itself is iterate zero, so best <= input always.
        if new_expression.len() > length_before {
            return tokens.to_vec();
        }

        new_expression
    }

    /// THE search over the cancel/rules move graph -- one mechanism in place of the former
    /// hand-picked candidate policies.
    ///
    /// * **State** = a prefix expression.
    /// * **Moves** (a single flat choice set, NOT a fixed cancel->rules alternation):
    ///   `apply_simplification_rules(S)`, and `cancel k-th candidate of S` for every `k`.
    ///   There is one region shape only: `neg`/`inv` are always region-continuing.
    ///   Interleavings the old fixpoint could not express (cancel->cancel->rules, rules-first,
    ///   cancel-only) are ordinary paths here. Ordering is load-bearing: applying the rules pass
    ///   as state *normalization* before the first cancel destroys cancellations.
    /// * **Objective** = the shortest state ever VISITED, not just terminal ones. That single
    ///   choice is what makes "decline to cancel" free -- retaining the parent node IS the old
    ///   `LegacyOpaque` policy, and tracking the running minimum IS best-iterate -- so both stop
    ///   being separate mechanisms.
    ///
    /// Every state is a.e.-equivalent to the input (sound cancel + sound rules compose), so any
    /// visited state is a sound answer and the search is free to minimize over all of them.
    ///
    /// Best-first by length (ties -> insertion order, so it is deterministic), a visited-set on
    /// the exact token sequence (the move graph is a DAG, not a tree -- most branches reconverge),
    /// depth capped at `2 * max_iter` moves (one old fixpoint iteration = cancel + rules = 2
    /// moves), and [`search_budget`] expansions. `best` is SEEDED with the greedy trajectory, so
    /// the result is never longer than the incumbent fixpoint's regardless of budget.
    fn simplify_search(
        &self,
        tokens: &[Tok],
        max_iter: usize,
        max_pattern_length: Option<usize>,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        // Pre-fill `best` with the greedy trajectory. This is ONLY insurance for a budget that
        // cuts the search off early: the greedy path is one path through the same move graph, so
        // with an unbounded budget the search finds it anyway. It costs one trajectory.
        let mut best = self.simplify_trajectory(
            tokens,
            max_iter,
            max_pattern_length,
            apply_simplification_rules,
            ctx,
        );
        let budget = search_budget();
        if budget == 0 {
            return best;
        }

        let max_depth = 2 * max_iter;
        let mut seen: FxHashSet<Vec<Tok>> = FxHashSet::default();
        seen.insert(tokens.to_vec());
        // (length, tie, depth, state) under `Reverse` -> pop the SHORTEST first; `tie` is unique
        // so the state itself is never compared (deterministic, and no Ord cost on the payload).
        let mut frontier: BinaryHeap<Reverse<(usize, usize, usize, Vec<Tok>)>> = BinaryHeap::new();
        frontier.push(Reverse((tokens.len(), 0, 0, tokens.to_vec())));
        let mut tie: usize = 0;
        let mut expanded: usize = 0;

        while let Some(Reverse((_, _, depth, state))) = frontier.pop() {
            if expanded >= budget {
                break;
            }
            expanded += 1;
            if depth >= max_depth {
                continue;
            }
            stats::bump(&stats::SIMPLIFY_ITERS);

            // Successors: the rules move, then every cancellation candidate.
            let mut successors: Vec<Vec<Tok>> = Vec::new();
            if apply_simplification_rules {
                successors.push(memoized_pass(&ctx.rules_memo, &state, || {
                    self.apply_simplification_rules_with_ctx(&state, max_pattern_length, ctx)
                }));
            }
            let t_cancel = std::time::Instant::now();
            let (_, n_candidates) =
                crate::cancel::cancel_nth(&state, &self.operators, &self.view(ctx), None);
            for k in 0..n_candidates {
                successors.push(
                    crate::cancel::cancel_nth(&state, &self.operators, &self.view(ctx), Some(k)).0,
                );
            }
            stats::add(&stats::NANOS_CANCEL, t_cancel.elapsed().as_nanos() as u64);

            for child in successors {
                if !seen.insert(child.clone()) {
                    continue;
                }
                if child.len() < best.len() {
                    best = child.clone();
                }
                tie += 1;
                frontier.push(Reverse((child.len(), tie, depth + 1, child)));
            }
        }

        best
    }

    /// One fixpoint trajectory under a fixed cancel-candidate policy: per iteration
    /// `cancel_terms` -> `apply_simplification_rules` (when enabled), break on no-op vs the
    /// previous iterate (`<= max_iter`).
    ///
    /// Returns the BEST-ITERATE, not the endpoint: every iterate is a.e.-equivalent to the
    /// input (sound cancel + sound rules compose), and a cancel emit may grow the expression
    /// (hyper/inverse tokens) betting on rule folding the ruleset cannot always pay off (sparse
    /// assets like 2-1); the endpoint can then stall LONGER than a mid-trajectory form, and the
    /// old endpoint-vs-input guard threw all banked wins away. Shortest iterate wins, ties
    /// prefer the LATER one (more rule-normalized).
    ///
    fn simplify_trajectory(
        &self,
        tokens: &[Tok],
        max_iter: usize,
        max_pattern_length: Option<usize>,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        // current_expression / new_expression both start as a copy of the input.
        let mut current_expression = tokens.to_vec();
        let mut new_expression = current_expression.clone();
        let mut best_expression = current_expression.clone();

        for _ in 0..max_iter {
            stats::bump(&stats::SIMPLIFY_ITERS);
            // Cancel any terms (cancel_terms(*collect_multiplicities(new_expression))).
            let t_cancel = std::time::Instant::now();
            new_expression = memoized_pass(&ctx.cancel_memo, &new_expression, || {
                crate::cancel::cancel_terms_unit(&new_expression, &self.operators, &self.view(ctx))
            });
            stats::add(&stats::NANOS_CANCEL, t_cancel.elapsed().as_nanos() as u64);

            // Apply simplification rules.
            if apply_simplification_rules {
                let t_rules = std::time::Instant::now();
                new_expression = memoized_pass(&ctx.rules_memo, &new_expression, || {
                    self.apply_simplification_rules_with_ctx(
                        &new_expression,
                        max_pattern_length,
                        ctx,
                    )
                });
                stats::add(&stats::NANOS_RULES, t_rules.elapsed().as_nanos() as u64);
            }

            // Converged: this iteration produced no change vs the previous iteration's result.
            if new_expression == current_expression {
                break;
            }
            if new_expression.len() <= best_expression.len() {
                best_expression = new_expression.clone();
            }
            current_expression = new_expression.clone();
        }

        best_expression
    }
}
