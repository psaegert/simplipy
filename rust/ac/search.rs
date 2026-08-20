//! D39 B1: opportunistic POST-FIXPOINT EXPLORATION -- the search scaffolding.
//!
//! Ledger D39: the deterministic chain runs
//! UNCHANGED to its fixpoint (the termination theorem is untouched -- this module is
//! never entered by the default entry points), then an explicit, budgeted exploration
//! phase tries EXPANSION moves -- states the descent chain refuses because they ascend
//! the reduction ordering at their node -- runs each candidate through the SAME
//! certified machinery (canonical constructors under the full-certificate [`Cx`], then
//! the [`rewrite_pass`] descent loop), and accepts an endpoint iff it is STRICTLY
//! below the incumbent in the engine's one reduction ordering ([`ordered_below`]).
//!
//! MEASURE-AGNOSTIC by construction (roadmap B1): no measure number appears anywhere
//! in this module -- acceptance is the `ordered_below` predicate, the same
//! Knuth-Bendix orientation the chain and the miner judge in. The numeric caps below
//! are CANDIDATE-SIZE guards (budget-class, like `max_passes`), not measure values.
//!
//! The D39 properties, and where they live:
//! * SOUNDNESS -- every candidate is built through the canonical constructors under
//!   the caller's certificate context, and the moves carry their own licences (see
//!   `distribute_product`): value preservation a.e. exactly as the chain's own
//!   collections. A candidate whose recollection is unlicensed simply settles to a
//!   fat endpoint and is refused by the ordering test -- wasted budget, never
//!   unsoundness.
//! * NEVER-WORSE -- `best` starts at the chain's fixpoint and is only ever replaced
//!   by a state strictly below it: fall back to the fixpoint is the identity case.
//! * TERMINATION -- `budget` bounds candidate descents; independent of the budget,
//!   the frontier only grows on strict descent of a well-founded ordering, so the
//!   loop terminates as a theorem even with the budget effectively infinite.
//! * IDEMPOTENCE -- deterministic order (canonical bag order, pre-order walk, FIFO
//!   frontier, single thread), and a reached valley re-explores to nothing: its own
//!   expansions all settle at or above it.
//!
//! MOVE SET (B1): `distribute` (a Mul bag's Add children, fully distributed -- the
//! row-156 move) and its Pow sibling `pow-expand` (integer power of a sum, expanded
//! through the same product builder). The enumeration in `local_moves` is the single
//! extension point for further kinds (common-denominator is deferred to the B7 lane
//! with its own falsifier). One deliberate scaffolding bound: the frontier extends
//! only from ACCEPTED valleys, so every win is reachable by depth-1 moves from a
//! valley -- composite hills that need two unaccepted expansions in sequence are
//! out of scope here and priced in the B7 acceptance study.
//!
//! Shared-state note: the caller's [`PassCtx`] (normal-form memo, step counter) is
//! reused across candidate descents -- same semantics as the chain's own passes; on
//! very large budgets the defense-in-depth step cap can begin refusing steps, which
//! is sound (candidates then settle high and are refused by the ordering test).

use std::collections::VecDeque;

use super::expr::{add, canon, fun, mul, pow, Cx, Ex};
use super::rules::{no_nested_bags, ordered_below, rewrite_pass, PassCtx};

/// Cap on the addend count of a fully-distributed candidate (cartesian width). A
/// size guard for candidate construction, not a measure: candidates past it are
/// simply not proposed.
const EXPAND_TERM_CAP: usize = 64;

/// pow-expand opens `(sum)^n` for integer `2 <= n <= POW_EXPAND_CAP` only -- the
/// candidate's size is exponential in `n`, and `EXPAND_TERM_CAP` above already
/// bounds the width; this keeps the factor list itself small.
const POW_EXPAND_CAP: i128 = 6;

/// The exploration phase (ledger D39). `fix` is the chain's fixpoint for the calling
/// mode; `pass` is that mode's OWN pass context (sound: the phase-1 pass; lossy: the
/// sentinel-expired phase-2 pass) -- the same certified machinery, byte for byte.
/// `budget` counts candidate descents; 0 disables the phase (the caller's guard makes
/// that the no-call case, so unused exploration is byte-identical behavior).
pub fn explore(fix: Ex, pass: &PassCtx, max_passes: usize, budget: usize) -> Ex {
    if budget == 0 {
        return fix;
    }
    let mut best = fix;
    let mut spent = 0usize;
    let mut frontier: VecDeque<Ex> = VecDeque::new();
    frontier.push_back(best.clone());
    while let Some(state) = frontier.pop_front() {
        for cand in expansion_candidates(&state, pass.cx) {
            if spent >= budget {
                return best;
            }
            spent += 1;
            // The candidate through the chain's own descent: canon under the full
            // certificate context, then rewrite passes to a fixpoint (or the same
            // max_passes truncation the chain itself accepts, which is sound).
            let mut cur = canon(cand, pass.cx);
            for _ in 0..max_passes.max(1) {
                let next = rewrite_pass(cur.clone(), pass);
                if next == cur {
                    break;
                }
                cur = next;
            }
            debug_assert!(
                no_nested_bags(&cur),
                "explored endpoint has nested bags: {cur:?}"
            );
            // THE acceptance: strictly below the incumbent in the serve ordering.
            if ordered_below(&cur, &best, pass.cx.view) {
                best = cur.clone();
                frontier.push_back(cur);
            }
        }
    }
    best
}

/// Every expansion candidate of `e`, as WHOLE STATES (the move applied at one node,
/// ancestors rebuilt through the canonical constructors under `cx`), in
/// deterministic pre-order: the local move at a node first, then its children in
/// canonical bag order.
fn expansion_candidates(e: &Ex, cx: &Cx) -> Vec<Ex> {
    let mut out = Vec::new();
    local_moves(e, cx, &mut out);
    match e {
        Ex::Add(v) => {
            for (i, c) in v.iter().enumerate() {
                for cand in expansion_candidates(c, cx) {
                    let mut items = v.clone();
                    items[i] = cand;
                    out.push(add(items, cx));
                }
            }
        }
        Ex::Mul(v) => {
            for (i, c) in v.iter().enumerate() {
                for cand in expansion_candidates(c, cx) {
                    let mut items = v.clone();
                    items[i] = cand;
                    out.push(mul(items, cx));
                }
            }
        }
        Ex::Pow(b, x) => {
            for cand in expansion_candidates(b, cx) {
                out.push(pow(cand, (**x).clone(), cx));
            }
            for cand in expansion_candidates(x, cx) {
                out.push(pow((**b).clone(), cand, cx));
            }
        }
        Ex::Fun(f, v) => {
            for (i, c) in v.iter().enumerate() {
                for cand in expansion_candidates(c, cx) {
                    let mut items = v.clone();
                    items[i] = cand;
                    out.push(fun(*f, items, cx));
                }
            }
        }
        _ => {}
    }
    out
}

/// The B1 move kinds at one node. THE extension point: a new expansion move is a new
/// arm here (with its own licences), nothing else changes.
fn local_moves(e: &Ex, cx: &Cx, out: &mut Vec<Ex>) {
    match e {
        // DISTRIBUTE (the row-156 move): fully distribute a Mul bag over its Add
        // children.
        Ex::Mul(v) => {
            if let Some(cand) = distribute_product(v, cx) {
                out.push(cand);
            }
        }
        // POW-EXPAND: `(sum)^n` for small integer n, expanded through the same
        // product builder (the factor list is n copies of the base).
        Ex::Pow(b, x) => {
            if let (Ex::Add(_), Ex::Num(r)) = (&**b, &**x) {
                if let Some(n) = r.as_integer() {
                    if (2..=POW_EXPAND_CAP).contains(&n) {
                        let factors: Vec<Ex> = std::iter::repeat_with(|| (**b).clone())
                            .take(n as usize)
                            .collect();
                        if let Some(cand) = distribute_product(&factors, cx) {
                            out.push(cand);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Fully distribute a product over its Add factors: the cartesian expansion
/// `sum over picks of (rest * pick)`, built through the canonical constructors under
/// the caller's certificate context (so licensed collections -- the recollect that
/// makes an expansion worth trying -- happen right here, in the same machinery the
/// chain uses). `None` when there is nothing to distribute or the move is not
/// licensed:
///
/// * VALUE PRESERVATION -- `a*(b+c) = a*b + a*c` can disagree only where an operand
///   is non-finite (e.g. `a = inf, b = 2, c = -1`: `inf*1` vs `inf - inf`), so every
///   factor and every distributed term must carry the finite-a.e. licence
///   (`Cx::fin_licensed` -- the `!`-certificate machinery, blanket-granted in lossy
///   mode exactly like the chain's own collections).
/// * `<constant>` INDEPENDENCE -- every `Const` occurrence is an independent fitted
///   constant; distribution DUPLICATES the factors it multiplies in, and duplicating
///   a `Const`-bearing subtree would mint independent copies of one constant (a
///   semantic widening no certificate covers). Any factor or term that would end up
///   in more than one addend must be `Const`-free.
/// * SIZE -- the cartesian width is capped (`EXPAND_TERM_CAP`).
fn distribute_product(factors: &[Ex], cx: &Cx) -> Option<Ex> {
    let mut adds: Vec<&Vec<Ex>> = Vec::new();
    let mut rest: Vec<Ex> = Vec::new();
    for f in factors {
        match f {
            Ex::Add(v) => adds.push(v),
            other => rest.push(other.clone()),
        }
    }
    if adds.is_empty() {
        return None;
    }
    let count = adds
        .iter()
        .try_fold(1usize, |acc, v| acc.checked_mul(v.len()))
        .filter(|&c| (2..=EXPAND_TERM_CAP).contains(&c))?;
    // Const-independence: refuse duplication of any Const-bearing piece.
    if count > 1 && rest.iter().any(Ex::contains_const) {
        return None;
    }
    for v in &adds {
        // Each term of this Add appears once per pick of the OTHER Adds.
        if count / v.len() > 1 && v.iter().any(|t| t.contains_const()) {
            return None;
        }
    }
    // Finite-a.e. licences for the distribution identity itself.
    if !rest.iter().all(|f| cx.fin_licensed(f))
        || !adds.iter().all(|v| v.iter().all(|t| cx.fin_licensed(t)))
    {
        return None;
    }
    // The cartesian picks, in canonical bag order (deterministic).
    let mut picks: Vec<Vec<Ex>> = vec![rest];
    for v in &adds {
        let mut next = Vec::with_capacity(picks.len() * v.len());
        for p in &picks {
            for t in v.iter() {
                let mut q = p.clone();
                q.push(t.clone());
                next.push(q);
            }
        }
        picks = next;
    }
    Some(add(picks.into_iter().map(|p| mul(p, cx)).collect(), cx))
}

#[cfg(test)]
mod tests {
    use crate::engine::RuleMode;
    use crate::Engine;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    fn t(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    /// Budget 0 is the no-phase case: byte-identical to the plain chain, both modes,
    /// both projections (the ledger's effort=0 semantics at the engine boundary).
    #[test]
    fn d39_budget_zero_is_byte_identical() {
        let Some(e) = engine() else { return };
        let row156 = t(&["*", "x2", "+", "x2", "/", "+", "x1", "1", "x2"]);
        let plain = t(&["+", "x1", "x2"]);
        for toks in [&row156, &plain] {
            for mode in [RuleMode::Default, RuleMode::Corpus] {
                for form in [
                    crate::engine::AcForm::Tagged,
                    crate::engine::AcForm::Explicit,
                ] {
                    assert_eq!(
                        e.ac_explore_proj(toks, 48, mode, form, 0),
                        e.ac_simplify_proj(toks, 48, mode, form),
                    );
                }
            }
        }
    }

    /// Row-156 (the ledger's falsifier row): exploration lands in the valley the
    /// greedy chain cannot reach, and the valley is strictly below in the serve
    /// ordering -- asserted through `ac_ordered_below`, measure-agnostic.
    #[test]
    fn d39_row156_reaches_the_valley() {
        let Some(e) = engine() else { return };
        let row156 = t(&["*", "x2", "+", "x2", "/", "+", "x1", "1", "x2"]);
        let valley = t(&["+", "1", "+", "x1", "pow", "x2", "2"]);
        let hill_fix = e.ac_simplify(&row156, 48, RuleMode::Default).unwrap();
        let valley_fix = e.ac_simplify(&valley, 48, RuleMode::Default).unwrap();
        assert_ne!(hill_fix, valley_fix, "row-156 stopped being a hill");
        let out = e
            .ac_explore_proj(
                &row156,
                48,
                RuleMode::Default,
                crate::engine::AcForm::Tagged,
                32,
            )
            .unwrap();
        assert_eq!(out, valley_fix);
        assert_eq!(e.ac_ordered_below(&out, &hill_fix), Some(true));
    }

    /// A reached valley re-explores to nothing, and two runs are identical
    /// (idempotence + determinism, ledger D39).
    #[test]
    fn d39_idempotent_and_deterministic() {
        let Some(e) = engine() else { return };
        let row156 = t(&["*", "x2", "+", "x2", "/", "+", "x1", "1", "x2"]);
        let form = crate::engine::AcForm::Tagged;
        let out = e
            .ac_explore_proj(&row156, 48, RuleMode::Default, form, 32)
            .unwrap();
        assert_eq!(
            e.ac_explore_proj(&row156, 48, RuleMode::Default, form, 32)
                .unwrap(),
            out
        );
        assert_eq!(
            e.ac_explore_proj(&out, 48, RuleMode::Default, form, 32)
                .unwrap(),
            out
        );
    }
}
