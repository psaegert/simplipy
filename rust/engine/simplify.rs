//! The simplify kernel: the `!`-certificate lookup, `apply_rules_top_down` (with the
//! constant-fold fallback), and the `simplify` tree search, all running on interned ids against
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

/// One memo-wrapped rewrite pass: return the cached output for `input`, or run
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
    /// prefix. This is the rules EDGE of the search graph.
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
    /// across every node it expands, so a node reached by several paths costs one pass.
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

    /// THE whole-unit kernel, on the prefix-token-list contract: run the tree search
    /// ([`Engine::simplify_search`]) to get the shortest a.e.-equivalent form it reaches, then
    /// `mask_elementary_literals` (when enabled), then `sort_operands` -- mask BEFORE sort, so the
    /// canonical operand order is computed on canonical tokens and mask/sort is idempotent. Both
    /// of those are length-neutral, so the search's guarantee (never longer than the input)
    /// survives them and no result guard is needed.
    ///
    /// Returns the simplified prefix tokens (the Python `'list'` return). The `inplace` /
    /// return-type machinery (str/tuple/np_array) is a Python-shim concern, not part of this kernel.
    ///
    /// This is the string boundary -- intern once at entry, resolve once at exit; the whole
    /// search runs on `Tok` ids ([`Engine::simplify_toks`]).
    pub fn simplify(
        &self,
        tokens: &[String],
        node_budget: usize,
        max_pattern_length: Option<usize>,
        mask_elementary_literals: bool,
        apply_simplification_rules: bool,
        wildcard_all: bool,
    ) -> Vec<String> {
        let ctx = SimplifyCtx::new(self.tokens.len(), wildcard_all);
        let toks = self.intern_seq(tokens, &ctx);
        let out = self.simplify_toks(
            &toks,
            node_budget,
            max_pattern_length,
            mask_elementary_literals,
            apply_simplification_rules,
            &ctx,
        );
        self.resolve_seq(&out, &ctx)
    }

    /// The id-level kernel (see [`Engine::simplify`] for the contract).
    fn simplify_toks(
        &self,
        tokens: &[Tok],
        node_budget: usize,
        max_pattern_length: Option<usize>,
        mask_elementary_literals: bool,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        stats::bump(&stats::SIMPLIFY_CALLS);

        // SEARCH over the cancel/rules move graph (see `simplify_search`): one mechanism
        // replacing the former hand-picked candidate policies.
        // Iterate SEARCH -> mask -> sort until the whole pipeline stops moving. Masking and
        // operand sorting are not cosmetic post-processing: they rewrite tokens (literals ->
        // `<constant>`) and reorder operands, which changes WHICH RULES MATCH, so a canonicalised
        // answer can be simplifiable again. Leaving them outside the loop is what made `simplify`
        // non-idempotent -- a second call found more on 0.87% of the 64k prior (979 tokens on
        // 4-3). Both are length-neutral and idempotent, and every round but the last strictly
        // shortens, so this terminates. `ctx` is shared across rounds, so the rules memo makes
        // repeated nodes free.
        let mut current = tokens.to_vec();
        loop {
            let searched = self.simplify_search(
                &current,
                node_budget,
                max_pattern_length,
                apply_simplification_rules,
                ctx,
            );
            let t_post = std::time::Instant::now();
            let mut canonical = if mask_elementary_literals {
                crate::utils::mask_elementary_literals(&searched, &self.view(ctx))
            } else {
                searched
            };
            canonical = crate::sort::sort_operands_unit(&canonical, &self.view(ctx));
            stats::add(&stats::NANOS_MASK_SORT, t_post.elapsed().as_nanos() as u64);
            if canonical == current {
                return current;
            }
            current = canonical;
        }
    }

    /// THE simplification algorithm: a best-first TREE SEARCH over rewrite states, re-rooted
    /// until it reaches a fixed point (see [`Engine::search_once`] for one search).
    ///
    /// One bounded search is not enough on its own: `best` is adopted the moment a node is
    /// GENERATED, but the budget can stop before that node is ever EXPANDED, so the answer may
    /// sit one move away from something shorter -- which is exactly what a user calling
    /// `simplify` twice would discover. (Measured on the 64k v23.0 prior: 0.87% of 4-3 rows
    /// improved on a second call, worth 979 tokens, about half of what the search itself buys.)
    /// Re-rooting the search at its own answer until the answer stops moving closes that, and
    /// makes idempotence hold BY CONSTRUCTION rather than by hope. It terminates because every
    /// round but the last strictly shortens the expression: `best` starts at the root and only
    /// strictly shorter nodes replace it, so a round either returns its root unchanged or
    /// returns something shorter.
    fn simplify_search(
        &self,
        tokens: &[Tok],
        node_budget: usize,
        max_pattern_length: Option<usize>,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        let mut current = tokens.to_vec();
        loop {
            let out = self.search_once(
                &current,
                node_budget,
                max_pattern_length,
                apply_simplification_rules,
                ctx,
            );
            if out == current {
                return current;
            }
            current = out;
        }
    }

    /// ONE bounded best-first search over rewrite states.
    ///
    /// * **Node** = a prefix expression. The **root** is the input.
    /// * **Edges** (one flat move set; there is no cancel->rules alternation): a node's children
    ///   are `apply_simplification_rules(node)` and `cancel(node, k)` for every qualifying
    ///   cancellation candidate `k`. `neg`/`inv` are always region-connected -- one cancel, one
    ///   region shape.
    /// * **Answer** = the shortest node ever VISITED, not merely the leaves. Every node is
    ///   a.e.-equivalent to the root (sound cancel and sound rules compose), so every node is a
    ///   legal answer and minimising over all of them is free. Two properties fall out with no
    ///   extra machinery: declining to cancel is simply keeping the parent, and no answer can
    ///   ever be longer than the input, because the root itself is the first candidate.
    ///
    /// The graph is a DAG rather than a tree -- most branches reconverge -- so a `visited` set on
    /// the exact token sequence collapses it. The frontier is a min-heap on node length; ties go
    /// to insertion order, which keeps the result deterministic.
    ///
    /// ONE bound, about work rather than correctness: `node_budget` caps how many nodes are
    /// EXPANDED. A separate depth cap used to exist and was removed as provably redundant --
    /// depth <= expansions <= budget, so any cap at or above the budget can never bind, and
    /// measurement confirmed the reachable depth saturates well inside it.
    ///
    /// Deliberately absent, each tried and measured on the 64k v23.0 prior:
    /// * a greedy PRE-FILL of `best` -- at matched wall-clock it is a wash (un-pre-filled at
    ///   budget 32 is 1509383 tokens / 78.5 us against pre-filled at 24 with 1509365 / 79.1), and
    ///   it forced a whole second cancel->rules code path to exist purely to produce it;
    /// * frontier PRIORITISATION by the promise of a move -- tie-breaking by candidate kind won
    ///   ~15 tokens per 1.5M for +2.8% time, and a one-ply post-rules lookahead LOST outright at
    ///   matched cost. Spending the time on more expansions beats spending it on a better order.
    fn search_once(
        &self,
        tokens: &[Tok],
        node_budget: usize,
        max_pattern_length: Option<usize>,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Tok> {
        // The root is the first candidate answer; every other node must beat it to be adopted.
        let mut best = tokens.to_vec();
        let mut visited: FxHashSet<Vec<Tok>> = FxHashSet::default();
        visited.insert(tokens.to_vec());
        // (length, tie, node) under `Reverse` -> pop the SHORTEST first; `tie` is unique so the
        // node itself is never compared (deterministic, and no Ord cost on the payload).
        let mut frontier: BinaryHeap<Reverse<(usize, usize, Vec<Tok>)>> = BinaryHeap::new();
        frontier.push(Reverse((tokens.len(), 0, tokens.to_vec())));
        let mut tie: usize = 0;
        let mut expanded: usize = 0;

        while let Some(Reverse((_, _, node))) = frontier.pop() {
            if expanded >= node_budget {
                break;
            }
            expanded += 1;
            stats::bump(&stats::SIMPLIFY_ITERS);

            let children =
                self.children_of(&node, max_pattern_length, apply_simplification_rules, ctx);

            for child in children {
                if !visited.insert(child.clone()) {
                    continue;
                }
                if child.len() < best.len() {
                    best = child.clone();
                }
                tie += 1;
                frontier.push(Reverse((child.len(), tie, child)));
            }
        }

        best
    }

    /// A node's children in the search graph: one rules pass, plus one per qualifying
    /// cancellation candidate.
    fn children_of(
        &self,
        node: &[Tok],
        max_pattern_length: Option<usize>,
        apply_simplification_rules: bool,
        ctx: &SimplifyCtx,
    ) -> Vec<Vec<Tok>> {
        let mut children: Vec<Vec<Tok>> = Vec::new();
        if apply_simplification_rules {
            children.push(memoized_pass(&ctx.rules_memo, node, || {
                self.apply_simplification_rules_with_ctx(node, max_pattern_length, ctx)
            }));
        }
        let t_cancel = std::time::Instant::now();
        children.extend(
            crate::cancel::cancel_successors(node, &self.operators, &self.view(ctx))
                .into_iter()
                .map(|(child, _sum)| child),
        );
        stats::add(&stats::NANOS_CANCEL, t_cancel.elapsed().as_nanos() as u64);
        // Operand SORTING is a legitimate edge -- length-neutral, semantically identity on
        // commutative operands, and it changes which rules match -- and it was TRIED as one. It
        // raises the reachable ceiling: with sorting confined to the pipeline loop the 64k prior
        // saturates at 1508224 tokens (4-3) no matter how large the budget gets, while sorting
        // as an edge reaches 1508057. But it is not efficient, because a sort pass is then paid
        // at EVERY expansion:
        //     ~124us  budget-only 1508255 (b96)  vs  sort-edge 1508656 (b12)
        //     ~150us  budget-only 1508229 (b192) vs  sort-edge 1508301 (b24)
        //     ~174us  budget-only 1508224 (b384) vs  sort-edge 1508158 (b32)   <- crossover
        // Below ~170us/expr the budget wins outright; only past the plateau does sorting pay.
        // At the shipped operating point (~106us) it costs ~2x for 241 tokens in 1.5M, so the
        // cheap way to get sorting's rule-unlocking effect is what `simplify_toks` already does:
        // sort BETWEEN pipeline rounds, a handful of times, instead of at every node.
        children
    }

}
