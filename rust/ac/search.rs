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
//!   both phases move only on strict descent of a well-founded ordering (the frontier
//!   grows only by accepted states, the finish only steps to a strictly lower best), so
//!   the search terminates as a theorem even with the budget effectively infinite.
//! * IDEMPOTENCE -- deterministic order (canonical bag order, pre-order walk, FIFO
//!   frontier, single thread), and a reached valley re-explores to nothing: its own
//!   expansions all settle at or above it (an uncapped search ends on a full round of
//!   the answer's own candidates that found nothing). That holds for the answer only when the
//!   search ran until a whole round of the answer's candidates found nothing (the public
//!   default, `effort=None`; owner
//!   2026-10-06): a cap can stop it between two accepted valleys, and a second call then
//!   continues from the first one's answer. On srbf's 125,127 model predictions (f64) a cap
//!   of 4, the old default, leaves 2,701 answers that change on a second call; uncapped, 26
//!   remain, at equal price but one: 23 that also change with the search off, and 3 long
//!   products that re-read with their factors in another order or sign. It also needs the answer to re-read to
//!   itself (L6a, docs/formal.md) and the step cap not to bind. `permissive` selects among
//!   three searches, and its winner is a valley of its own search only (formal.md L6).
//!
//! MOVE SET (B1): `distribute` (a Mul bag's Add children, fully distributed -- the
//! row-156 move) and its Pow sibling `pow-expand` (integer power of a sum, expanded
//! through the same product builder), and their reverse `factor` (terms of a sum that share
//! a factor, as that factor times the quotient sum: the distributive law, which the search
//! owns, see [`factorings`]). The enumeration in `local_moves` is the single
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

use super::expr::{
    add, complexity, distributes_power, fun, mul, orient_term_classes, orientation_near_miss, pow,
    Cx, Ex,
};
use super::rat::Rat;
use super::rules::{no_nested_bags, ordered_below, rewrite_pass, PassCtx};

/// Cap on the addend count of a fully-distributed candidate (cartesian width). A
/// size guard for candidate construction, not a measure: candidates past it are
/// simply not proposed.
const EXPAND_TERM_CAP: usize = 64;

/// pow-expand opens `(sum)^n` for integer `2 <= n <= POW_EXPAND_CAP` only -- the
/// candidate's size is exponential in `n`, and `EXPAND_TERM_CAP` above already
/// bounds the width; this keeps the factor list itself small.
const POW_EXPAND_CAP: i128 = 6;

/// How many candidate descents the breadth-first first phase of [`explore`] runs before
/// the first-improvement finish takes over. On srbf's 125,127 model predictions (f64) the
/// first phase alone returns the final answer on 124,572; the finish improves the other 555
/// (by a median of 58 bits) and runs only where 8 did not settle.
/// The first phase is the capped default's breadth-first search, so per search an answer
/// is never worse than the same build's answer at any effort up to the prefix.
const BFS_PREFIX: usize = 8;

/// The exploration phase (ledger D39). `fix` is the chain's fixpoint for the calling
/// mode; `pass` is that mode's OWN pass context (sound: the phase-1 pass; lossy: the
/// sentinel-expired phase-2 pass) -- the same certified machinery, byte for byte.
/// `budget` counts candidate descents; 0 disables the phase (the caller's guard makes
/// that the no-call case, so unused exploration is byte-identical behavior).
///
/// Two phases, one acceptance (strictly below the incumbent in [`ordered_below`]):
/// 1. BREADTH-FIRST, for the first [`BFS_PREFIX`] descents: every candidate of every
///    accepted state, in pre-order, against the best so far (the capped default's search).
///    If its frontier empties within the prefix, the answer is the breadth-first one.
/// 2. FIRST IMPROVEMENT, from the best: try the best's candidates in pre-order starting at
///    the place that last improved, step to the first one strictly below, and stop after
///    a full round that finds nothing. Breadth-first search re-tries every candidate of
///    every accepted state, so `k` independent improvable places cost it `4k^2 + 1`
///    descents; this finish costs about one round per improvement plus one closing round
///    (16 copies of one srbf prediction: about six times the time of `effort=4`, for an answer
///    a third of the price).
///
/// A candidate is the move applied at one place with its ancestors rebuilt through the
/// canonical constructors ([`rebuild`]): a state's moves are computed when the state is
/// walked ([`site_list`]), and a candidate's whole state is rebuilt only when it is tried,
/// then descended by
/// [`rewrite_pass`] to its fixpoint (or the same `max_passes` truncation the chain itself
/// accepts, which is sound).
///
/// `trace`, when given, receives every state the search accepts, in order, with the number
/// of descents spent when it was accepted (the starting fixpoint is not recorded). A run
/// under a smaller budget is the same walk cut short, so a run capped at `k` ends on the
/// last of these with at most `k` descents (or on the fixpoint): `permissive` uses this to
/// make more budget never end costlier.
///
/// One move set ([`local_moves`]: DISTRIBUTE, POW-EXPAND and FACTOR) under one acceptance. A
/// FACTOR candidate is descended only when its rebuilt state is already below the best: the
/// merges a factored sum enables across its product happen in the constructors that rebuild its
/// ancestors, so a factoring that does not pay there is not walked.
pub fn explore(
    fix: Ex,
    pass: &PassCtx,
    max_passes: usize,
    budget: usize,
    trace: Option<&mut Vec<(usize, Ex)>>,
) -> Ex {
    if budget == 0 {
        return fix;
    }
    // THE WORK BUDGET (`ac::work`): the search may spend `work::search_budget()` units --
    // matcher steps and constructor calls -- and stops when they are spent, exactly as when the
    // descent cap binds: it returns the best state so far. A descent the ceiling cut short is
    // never accepted. The units are a function of the walk alone, so the cut falls at the same
    // place under every `budget`: a run capped at `k` descents is still the same walk cut short,
    // and permissive's selection keeps its three bounds (docs/formal.md).
    let _ceiling = super::work::limit(super::work::search_budget());
    search(fix, pass, max_passes, budget, trace)
}

/// The descent of a candidate to its fixpoint under `pass` (or the `max_passes` truncation
/// the chain itself accepts).
fn descend(mut cur: Ex, pass: &PassCtx, max_passes: usize) -> Ex {
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
    cur
}

/// A candidate: the place of a move, the moved node, and whether it is a FACTOR move (descended
/// only when its rebuilt state is already below the best).
type Site = (Vec<usize>, Ex, bool);

/// The candidate `site` of `state`, rebuilt and descended; `None` for a FACTOR move whose
/// rebuilt state is not below `best`.
fn try_site(state: &Ex, best: &Ex, site: &Site, pass: &PassCtx, max_passes: usize) -> Option<Ex> {
    let (path, moved, probe) = site;
    let rebuilt = rebuild(state, path, moved.clone(), pass.cx);
    if *probe && !ordered_below(&rebuilt, best, pass.cx.view) {
        return None;
    }
    Some(descend(rebuilt, pass, max_passes))
}

/// The search of [`explore`]. Returns the best state.
fn search(
    fix: Ex,
    pass: &PassCtx,
    max_passes: usize,
    budget: usize,
    mut trace: Option<&mut Vec<(usize, Ex)>>,
) -> Ex {
    let mut best = fix;
    let mut spent = 0usize;
    // Phase 1: breadth-first over accepted states.
    let mut frontier: VecDeque<Ex> = VecDeque::new();
    frontier.push_back(best.clone());
    let mut settled = true;
    // When the prefix runs out while walking the best state itself, the candidates of it
    // already tried were refused against that same best: the finish's first round skips them.
    let mut refused_of_best = 0usize;
    'bfs: while let Some(state) = frontier.pop_front() {
        let mut sites: Vec<Site> = Vec::new();
        site_list(&state, pass.cx, &[], None, &mut Vec::new(), &mut sites);
        for (i, site) in sites.iter().enumerate() {
            if spent >= budget || super::work::over() {
                return best;
            }
            if spent >= BFS_PREFIX {
                settled = false;
                // `best` only ever moves strictly below, so if it equals the state being
                // walked, it was this state for the whole walk.
                if state == best {
                    refused_of_best = i;
                }
                break 'bfs;
            }
            spent += 1;
            let Some(cur) = try_site(&state, &best, site, pass, max_passes) else {
                continue;
            };
            if super::work::over() {
                return best;
            }
            if ordered_below(&cur, &best, pass.cx.view) {
                best = cur.clone();
                if let Some(t) = trace.as_deref_mut() {
                    t.push((spent, cur.clone()));
                }
                frontier.push_back(cur);
            }
        }
    }
    if settled {
        return best;
    }
    // Phase 2: first improvement from the best until a full round finds nothing. A round
    // walks the best's candidates cyclically from where the last improvement happened; the
    // first round starts after the candidates phase 1 already refused against this best (the
    // same list: `site_list` is deterministic), so together they still make a full round.
    let mut resume: Vec<usize> = Vec::new();
    let mut skip = refused_of_best;
    loop {
        let mut sites: Vec<Site> = Vec::new();
        site_list(&best, pass.cx, &[], None, &mut Vec::new(), &mut sites);
        let n = sites.len();
        let start = if skip > 0 {
            skip
        } else {
            sites.partition_point(|(p, _, _)| *p < resume)
        };
        let round = n.saturating_sub(skip);
        skip = 0;
        let mut accepted: Option<(Ex, Vec<usize>)> = None;
        for off in 0..round {
            if spent >= budget || super::work::over() {
                return best;
            }
            spent += 1;
            let site = &sites[(start + off) % n];
            let Some(cur) = try_site(&best, &best, site, pass, max_passes) else {
                continue;
            };
            if super::work::over() {
                return best;
            }
            if ordered_below(&cur, &best, pass.cx.view) {
                accepted = Some((cur, site.0.clone()));
                break;
            }
        }
        match accepted {
            Some((cur, path)) => {
                if let Some(t) = trace.as_deref_mut() {
                    t.push((spent, cur.clone()));
                }
                best = cur;
                resume = path;
            }
            None => return best,
        }
    }
}

/// The places of `e` with a move, in deterministic pre-order (the moves at a node first, then its
/// children in canonical bag order), each with its moved node. `near` holds the bases of the
/// product `e` is a factor of, and `power` the integer exponent `e` is the base of, if any (what
/// FACTOR reads, see [`factor_moves`]).
fn site_list(
    e: &Ex,
    cx: &Cx,
    near: &[&Ex],
    power: Option<&Rat>,
    path: &mut Vec<usize>,
    out: &mut Vec<Site>,
) {
    let mut own = Vec::new();
    local_moves(e, cx, near, power, &mut own);
    for (m, probe) in own {
        out.push((path.clone(), m, probe));
    }
    match e {
        Ex::Add(v) | Ex::Fun(_, v) => {
            for (i, c) in v.iter().enumerate() {
                path.push(i);
                site_list(c, cx, &[], None, path, out);
                path.pop();
            }
        }
        Ex::Mul(v) => {
            let bases = term_bases(e);
            for (i, c) in v.iter().enumerate() {
                path.push(i);
                site_list(c, cx, &bases, None, path, out);
                path.pop();
            }
        }
        Ex::Pow(b, x) => {
            let k = match &**x {
                Ex::Num(r) if r.is_integer() => Some(r),
                _ => None,
            };
            path.push(0);
            site_list(b, cx, near, k, path, out);
            path.pop();
            path.push(1);
            site_list(x, cx, &[], None, path, out);
            path.pop();
        }
        _ => {}
    }
}

/// `e` with the node at `path` replaced by `moved`: the WHOLE STATE the candidate is, its
/// ancestors rebuilt through the canonical constructors under `cx`.
fn rebuild(e: &Ex, path: &[usize], moved: Ex, cx: &Cx) -> Ex {
    let Some((&i, rest)) = path.split_first() else {
        return moved;
    };
    match e {
        Ex::Add(v) => {
            let mut items = v.clone();
            items[i] = rebuild(&v[i], rest, moved, cx);
            add(items, cx)
        }
        Ex::Mul(v) => {
            let mut items = v.clone();
            items[i] = rebuild(&v[i], rest, moved, cx);
            mul(items, cx)
        }
        Ex::Fun(f, v) => {
            let mut items = v.clone();
            items[i] = rebuild(&v[i], rest, moved, cx);
            fun(*f, items, cx)
        }
        Ex::Pow(b, x) => {
            if i == 0 {
                pow(rebuild(b, rest, moved, cx), (**x).clone(), cx)
            } else {
                pow((**b).clone(), rebuild(x, rest, moved, cx), cx)
            }
        }
        _ => moved,
    }
}

/// The move kinds at one node, each with whether it is FACTOR. THE extension point: a new move
/// is a new arm here (with its own licences), nothing else changes.
fn local_moves(e: &Ex, cx: &Cx, near: &[&Ex], power: Option<&Rat>, out: &mut Vec<(Ex, bool)>) {
    match e {
        // DISTRIBUTE (the row-156 move): fully distribute a Mul bag over its Add
        // children.
        Ex::Mul(v) => {
            if let Some(cand) = distribute_product(v, cx) {
                out.push((cand, false));
            }
        }
        // POW-EXPAND: `(sum)^n` for small integer n, expanded through the same
        // product builder (the factor list is n copies of the base).
        Ex::Pow(b, x) => {
            if let (Ex::Add(_), Ex::Num(r)) = (&**b, &**x) {
                if let Some(n) = r.small_int() {
                    if (2..=POW_EXPAND_CAP).contains(&n) {
                        let factors: Vec<Ex> = std::iter::repeat_with(|| (**b).clone())
                            .take(n as usize)
                            .collect();
                        if let Some(cand) = distribute_product(&factors, cx) {
                            out.push((cand, false));
                        }
                    }
                }
            }
        }
        // FACTOR: the reverse of DISTRIBUTE, at a sum.
        Ex::Add(v) => {
            for cand in factor_moves(v, cx, near, power) {
                out.push((cand, true));
            }
        }
        _ => {}
    }
}

/// The bases of a term's factors, borrowed, as [`term_factors`] reads them.
fn term_bases(t: &Ex) -> Vec<&Ex> {
    fn base(f: &Ex) -> Option<&Ex> {
        match f {
            Ex::Num(_) => None,
            Ex::Pow(b, x) => match &**x {
                Ex::Num(r) if !r.is_zero() => Some(&**b),
                _ => Some(f),
            },
            _ => Some(f),
        }
    }
    match t {
        Ex::Mul(fs) => fs.iter().filter_map(base).collect(),
        _ => base(t).into_iter().collect(),
    }
}

/// A term split into its coefficient literals and its factors as `(base, exponent)`.
type SplitTerm = (Vec<Ex>, Vec<(Ex, Rat)>);

/// A term's coefficient and its factors as `(base, exponent)`: a literal power `b^r` reads as
/// `(b, r)`, so `x1` and `x1^2` share `x1`, and `1/x1` and `x1` share it at exponent -1; any
/// other factor is itself with exponent 1.
fn term_factors(t: &Ex) -> SplitTerm {
    let one = |f: &Ex| match f {
        Ex::Pow(b, x) => match &**x {
            Ex::Num(r) if !r.is_zero() => ((**b).clone(), r.clone()),
            _ => (f.clone(), Rat::int(1)),
        },
        _ => (f.clone(), Rat::int(1)),
    };
    match t {
        Ex::Mul(fs) => {
            let (coef, rest): (Vec<&Ex>, Vec<&Ex>) =
                fs.iter().partition(|f| matches!(f, Ex::Num(_)));
            (
                coef.into_iter().cloned().collect(),
                rest.into_iter().map(one).collect(),
            )
        }
        Ex::Num(_) => (vec![t.clone()], Vec::new()),
        _ => (Vec::new(), vec![one(t)]),
    }
}

/// The FACTOR moves of the sum `terms` at its place (see [`site_list`]): every factoring
/// ([`factorings`]) whose pieces carry DISTRIBUTE's licences, read backwards: the identity can
/// fail only where a piece is non-finite, so the content and every quotient carry the finite-a.e.
/// licence (the sound certificate for a piece that spells an infinity). Factoring a sum under a
/// negative power is distributing that power read backwards: the factored product then splits
/// under it, so it takes the same licence (`distributes_power`). Where `pow` would hold the
/// product -- a factor whose zero set may hold a region, where the inverse's infinity takes the
/// product's sign -- the sum is not factored either.
///
/// A factoring is proposed where it can pay: the factored sum is cheaper by itself, or a factor
/// it takes out of the whole sum shares a base with the product the sum (or a power of it) is a
/// factor of (`near`). Under a lone inverse the factored spelling can tie the sum
/// (`1/(x1 - x1*x2)` against `-(1/(x2 - 1))/x1`), so the chain never takes it, while the merges
/// it enables across the product pay (`rootn(x1, 2)/(x1 - x1*x2)` becomes
/// `-(1/pow(x1, 1/2))*(1/(x2 - 1))`).
fn factor_moves(terms: &[Ex], cx: &Cx, near: &[&Ex], power: Option<&Rat>) -> Vec<Ex> {
    if terms.len() > EXPAND_TERM_CAP {
        return Vec::new();
    }
    let licensed = |f: &Ex| {
        if f.contains_infinity() {
            cx.fin_certified(f)
        } else {
            cx.fin_licensed(f)
        }
    };
    let cands = factorings(terms, cx, &licensed);
    if cands.is_empty() {
        return Vec::new();
    }
    let whole = complexity(&Ex::Add(terms.to_vec()), cx.view);
    let mut out = Vec::new();
    for (cand, c) in cands {
        let pays = match &cand {
            Ex::Mul(fs) => {
                if let Some(k) = power.filter(|k| k.is_negative()) {
                    if !distributes_power(fs, k, cx) {
                        continue;
                    }
                }
                term_bases(&c).iter().any(|b| near.contains(b))
            }
            _ => false,
        };
        if pays || complexity(&cand, cx.view) < whole {
            out.push(cand);
        }
    }
    out
}

/// THE DISTRIBUTIVE LAW read as factoring, at the sum `terms`: each candidate is the sum with
/// some of its terms replaced by their content `c` times the sum of their quotients, returned with
/// `c`. Every way the engine factors is here, once: the search proposes these moves
/// ([`factor_moves`], with the licences in `licensed`).
///
/// * SHARED BASES -- for every base two or more terms share, those terms factored by everything
///   they share, each base at its least exponent (`x1*x2 + x1*x3 + x4` is `x4 + x1*(x2 + x3)`).
///   A bare term is its own product with the unit coefficient ([`term_factors`]):
///   `x2*x3 + x2` is `x2*(x3 + 1)`. A base counts up to sign, as a repeated wildcard binds a
///   class up to sign: where a base of one term meets in another its negation through a class's
///   orientation (a near miss, `orientation_near_miss`), those terms are factored with every class
///   in one orientation, the signs in the coefficients (`orient_term_classes`):
///   `x1/(x2 - x3) + x4/(x3 - x2)` is `(x1 - x4)/(x2 - x3)`.
/// * A SUM FACTOR SPREAD AMONG THE TERMS -- a term `c*S*q` whose sum factor `S`, or its negation,
///   also stands among the other terms (flattened into the bag, so no term holds it as a factor)
///   factors as `S*(c*q + 1)` or `S*(c*q - 1)`: `x3 - x2 + (x2 - x3)*x4` is `(x2 - x3)*(x4 - 1)`.
///
/// The content holds no `<constant>`, whose occurrences in the terms are independent constants
/// that one factor would merge into one.
pub(crate) fn factorings(terms: &[Ex], cx: &Cx, licensed: &dyn Fn(&Ex) -> bool) -> Vec<(Ex, Ex)> {
    let mut out: Vec<(Ex, Ex)> = Vec::new();
    if terms.len() < 2 {
        return out;
    }
    let whole = Ex::Add(terms.to_vec());
    // The sum with the terms at `members` (ascending) replaced by `group` factored.
    let push = |members: &[usize], group: &[Ex], out: &mut Vec<(Ex, Ex)>| {
        let Some((factored, c)) = factor_shared(group, cx, licensed) else {
            return;
        };
        let mut items: Vec<Ex> = terms
            .iter()
            .enumerate()
            .filter(|(j, _)| members.binary_search(j).is_err())
            .map(|(_, t)| t.clone())
            .collect();
        let cand = if items.is_empty() {
            factored
        } else {
            items.push(factored);
            add(items, cx)
        };
        if cand != whole && !out.iter().any(|(o, _)| *o == cand) {
            out.push((cand, c));
        }
    };
    // Shared bases: the terms holding each base (as spelled, or up to sign), once per group.
    let bases: Vec<Vec<&Ex>> = terms.iter().map(term_bases).collect();
    let mut seen: Vec<&Ex> = Vec::new();
    let mut groups: Vec<(Vec<usize>, bool)> = Vec::new();
    for bs in &bases {
        for &b in bs {
            if seen.contains(&b) {
                continue;
            }
            seen.push(b);
            let mut members = Vec::new();
            let mut signed = false;
            for (j, cs) in bases.iter().enumerate() {
                if cs.contains(&b) {
                    members.push(j);
                } else if cs.iter().any(|c| orientation_near_miss(b, c, cx.view)) {
                    members.push(j);
                    signed = true;
                }
            }
            if members.len() >= 2 && !groups.iter().any(|g| g.0 == members && g.1 == signed) {
                groups.push((members, signed));
            }
        }
    }
    for (members, signed) in &groups {
        let group: Vec<Ex> = members
            .iter()
            .map(|&j| {
                let t = &terms[j];
                if *signed {
                    orient_term_classes(t, cx).unwrap_or_else(|| t.clone())
                } else {
                    t.clone()
                }
            })
            .collect();
        push(members, &group, &mut out);
    }
    // A sum factor spread among the other terms, as itself (`k = 1`) or negated (`k = -1`).
    for (i, t) in terms.iter().enumerate() {
        let (_, fs) = term_factors(t);
        for (s, r) in &fs {
            if !matches!(s, Ex::Add(_)) || r.is_negative() {
                continue;
            }
            for k in [1, -1] {
                let spread = mul(vec![Ex::int(k), s.clone()], cx);
                let Ex::Add(us) = &spread else {
                    continue;
                };
                let mut members = vec![i];
                let found = us.iter().all(|u| {
                    let hit = (0..terms.len()).find(|j| !members.contains(j) && terms[*j] == *u);
                    hit.inspect(|&j| members.push(j)).is_some()
                });
                if !found {
                    continue;
                }
                members.sort_unstable();
                // The spread terms as the one product `k*S` they are, so the content is `S`.
                let group = [t.clone(), Ex::Mul(vec![Ex::int(k), s.clone()])];
                push(&members, &group, &mut out);
            }
        }
    }
    out
}

/// `terms` as their content `c` (every base they all share, each at its least exponent) times
/// the sum of their quotients; returned with `c`. `None` when they share nothing, `c` holds a
/// `<constant>`, or a piece is not `licensed`.
fn factor_shared(terms: &[Ex], cx: &Cx, licensed: &dyn Fn(&Ex) -> bool) -> Option<(Ex, Ex)> {
    let split: Vec<SplitTerm> = terms.iter().map(term_factors).collect();
    // The exponent arithmetic below reads each base once per term, which a canonical product
    // guarantees; a term that holds one base twice is not factored.
    let twice = |fs: &[(Ex, Rat)]| {
        fs.iter()
            .enumerate()
            .any(|(i, (b, _))| fs[i + 1..].iter().any(|(c, _)| c == b))
    };
    if split.iter().any(|(_, fs)| twice(fs)) {
        return None;
    }
    let mut common: Vec<(Ex, Rat)> = split[0].1.clone();
    for (_, fs) in &split[1..] {
        common = common
            .into_iter()
            .filter_map(|(b, r)| {
                let (_, s) = fs.iter().find(|(c, _)| *c == b)?;
                let least = if s.cmp_exact(&r) == std::cmp::Ordering::Less {
                    s.clone()
                } else {
                    r
                };
                Some((b, least))
            })
            .collect();
        if common.is_empty() {
            return None;
        }
    }
    let power = |b: Ex, r: Rat| {
        if r.is_one() {
            b
        } else {
            pow(b, Ex::Num(r), cx)
        }
    };
    let c = mul(
        common
            .iter()
            .map(|(b, r)| power(b.clone(), r.clone()))
            .collect(),
        cx,
    );
    if c.contains_const() {
        return None;
    }
    let mut quotients = Vec::with_capacity(terms.len());
    for (coef, fs) in split {
        let mut items = coef;
        for (b, r) in fs {
            let left = match common.iter().find(|(c, _)| *c == b) {
                Some((_, rc)) => r.checked_add(&rc.checked_neg()?)?,
                None => r,
            };
            if !left.is_zero() {
                items.push(power(b, left));
            }
        }
        quotients.push(if items.is_empty() {
            Ex::int(1)
        } else {
            mul(items, cx)
        });
    }
    if !licensed(&c) || !quotients.iter().all(licensed) {
        return None;
    }
    Some((mul(vec![c.clone(), add(quotients, cx)], cx), c))
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
///   mode exactly like the chain's own collections) -- EXCEPT a piece that spells an
///   infinity, which needs the sound certificate (`Cx::fin_certified`) in every mode.
///   The constructor steps the blanket licenses (opposite-sign cancellation, the zero
///   collapse, an infinity absorbing a term) give a value only where their input has
///   none (NaN); this move runs the other way, and at an infinite piece it turns a
///   defined `+-inf` into NaN on a set of full measure (`-inf*(x1 - 1/2)` becomes
///   `inf - inf*x1`, NaN for every `x1 > 0`). The chain then legitimately fills that NaN
///   (`inf - inf*x1 -> inf`), so the search would accept `inf` for a value that is
///   `-inf` on `x1 > 1/2`. For a piece that spells an infinity the blanket's premise
///   ("finite a.e.") is false everywhere, not on a null set, so the blanket stops there.
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
    // Finite-a.e. licences for the distribution identity itself; the lossy blanket does
    // not cover a piece that spells an infinity (see the doc above). Outside lossy mode
    // both branches are the same certificate, so the sound modes are unchanged.
    let licensed = |f: &Ex| {
        if f.contains_infinity() {
            cx.fin_certified(f)
        } else {
            cx.fin_licensed(f)
        }
    };
    if !rest.iter().all(licensed) || !adds.iter().all(|v| v.iter().all(licensed)) {
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
            for mode in [RuleMode::Default, RuleMode::Permissive] {
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

    /// The move's licence at a spelled infinity, at the constructor level (no assets): the
    /// lossy blanket distributes `x2 * (x1 - 1/2)` but not `-inf * (x1 - 1/2)`, whose
    /// distributed form `inf - inf*x1` is NaN on `x1 > 0` where the product is `+-inf`
    /// (and which the constructors finish to `inf`, wrong on `x1 > 1/2`).
    #[test]
    fn distribute_refuses_a_spelled_infinity_in_every_mode() {
        use super::distribute_product;
        use crate::ac::expr::{add, Cx, Ex};
        use crate::ac::rat::Rat;
        use crate::operators::{OperatorSpec, Operators};
        use crate::tokens::{TokenOverlay, TokenTable, TokenView};
        let mut order = Vec::new();
        let mut specs: rustc_hash::FxHashMap<String, OperatorSpec> = Default::default();
        for (n, arity) in [
            ("+", 2),
            ("-", 2),
            ("*", 2),
            ("/", 2),
            ("pow", 2),
            ("tanh", 1),
        ] {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: vec![],
                    inverse: None,
                    arity,
                    precedence: None,
                    commutative: n == "+" || n == "*",
                },
            );
        }
        let ops = Operators::from_specs(order.clone(), specs);
        let table = TokenTable::build(&order, &ops);
        let overlay = std::cell::RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        let mut lossy = Cx::bare(&view);
        lossy.mode = RuleMode::Permissive;
        lossy.sentinels_expired = true;
        let sound = Cx::bare(&view);
        let x1 = Ex::Leaf(view.intern("x1"));
        let x2 = Ex::Leaf(view.intern("x2"));
        let half = Ex::Num(Rat::new(-1, 2).unwrap());
        let sum = add(vec![x1.clone(), half], &lossy);
        assert!(matches!(sum, Ex::Add(_)), "x1 - 1/2 stays a sum: {sum:?}");
        for inf in [Ex::NegInf, Ex::PosInf] {
            for cx in [&lossy, &sound] {
                assert_eq!(
                    distribute_product(&[inf.clone(), sum.clone()], cx),
                    None,
                    "{inf:?} * (x1 - 1/2) must not distribute (lossy = {})",
                    cx.lossy()
                );
                // An infinity inside a co-factor is the same piece.
                let carrier = crate::ac::expr::mul(vec![inf.clone(), x2.clone()], cx);
                assert_eq!(distribute_product(&[carrier, sum.clone()], cx), None);
            }
        }
        // A finite factor still distributes under the blanket (the move is kept).
        assert!(distribute_product(&[x2.clone(), sum.clone()], &lossy).is_some());
    }

    /// End to end (acj-4): permissive keeps the value of an infinity times a sign-changing
    /// sum at every effort, the repro and its finite-valued relatives included.
    #[test]
    fn permissive_search_keeps_signed_infinity_products() {
        let Some(e) = engine() else { return };
        let form = crate::engine::AcForm::Explicit;
        for (input, wrong) in [
            (
                t(&["*", "float(\"-inf\")", "-", "x1", "/", "1", "2"]),
                t(&["float(\"inf\")"]),
            ),
            (
                t(&[
                    "+",
                    "1",
                    "tanh",
                    "*",
                    "float(\"inf\")",
                    "-",
                    "x1",
                    "/",
                    "1",
                    "2",
                ]),
                t(&["0"]),
            ),
            (
                t(&["*", "float(\"inf\")", "*", "x1", "-", "x2", "1"]),
                t(&["*", "float(\"-inf\")", "x1"]),
            ),
        ] {
            let off = e
                .ac_simplify_proj(&input, 48, RuleMode::Permissive, form)
                .unwrap();
            for budget in [1, 4, usize::MAX] {
                let out = e
                    .ac_explore_proj(&input, 48, RuleMode::Permissive, form, budget)
                    .unwrap();
                assert_ne!(out, wrong, "{input:?} at budget {budget}");
                assert_eq!(
                    out, off,
                    "{input:?} at budget {budget}: the search moved it"
                );
            }
        }
    }
}
