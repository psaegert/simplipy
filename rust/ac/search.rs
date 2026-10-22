//! THE SEARCH: the exploration phase after the chain's fixpoint.
//!
//! The deterministic chain (rules and constructors) runs UNCHANGED to its fixpoint (the
//! termination theorem is untouched), then the search tries EXPANSION moves -- states the
//! descent chain refuses because they ascend the reduction ordering at their node -- runs each
//! candidate through the SAME certified machinery (canonical constructors under the caller's
//! [`Cx`], then the [`rewrite_pass`] descent loop), and accepts an endpoint iff it is STRICTLY
//! below the incumbent in the engine's one reduction ordering ([`ordered_below`]). Every mode
//! runs it, `permissive` included, in its own pass context; `effort=0` never enters it.
//!
//! ONE WALK ([`explore`]): a round takes the best state's candidates -- every move at every
//! place, each rebuilt into the whole state it is -- cheapest rebuilt state first, descends each
//! to its fixpoint, steps to the first endpoint strictly below the best and starts the next
//! round there; a round that accepts nothing ends the search. `effort=k` is the first `k`
//! descents of that walk.
//!
//! MEASURE-AGNOSTIC by construction: no measure number appears anywhere in this module --
//! acceptance is the `ordered_below` predicate, the same Knuth-Bendix orientation the chain
//! and the miner judge in. The numeric caps below are CANDIDATE-SIZE guards (budget-class,
//! like `max_passes`), not measure values.
//!
//! The properties, and where they live:
//! * SOUNDNESS -- every candidate is built through the canonical constructors under
//!   the caller's certificate context, and the moves carry their own licences (see
//!   `distribute_product`): value preservation a.e. exactly as the chain's own
//!   collections. A candidate whose recollection is unlicensed simply settles to a
//!   fat endpoint and is refused by the ordering test -- wasted budget, never
//!   unsoundness.
//! * NEVER-WORSE -- `best` starts at the chain's fixpoint and is only ever replaced
//!   by a state strictly below it: fall back to the fixpoint is the identity case.
//! * TERMINATION -- `budget` bounds candidate descents; independent of the budget,
//!   the search moves only on strict descent of a well-founded ordering (it steps only to a
//!   strictly lower best), so it terminates as a theorem even with the budget effectively
//!   infinite.
//! * IDEMPOTENCE -- deterministic order (canonical bag order, pre-order walk, candidates
//!   ordered by price, single thread), and a reached valley re-explores to nothing: its own
//!   expansions all settle at or above it. That holds for the answer only when the search ran
//!   until a whole round of the answer's candidates found nothing (the default,
//!   `effort=None`): a cap can stop it between two accepted valleys, and a second call then
//!   continues from the first one's answer. It also needs the answer
//!   to re-read to itself (L6a, docs/formal.md) and the step cap not to bind.
//!
//! MOVE SET: the distributive law, which the search owns (the rule load refuses a rule that
//! states it, `rules::engine_owned_law`), both ways: DISTRIBUTE (a Mul bag's Add children,
//! fully distributed, or one factor into one sum factor, or the other factors over one term of
//! a sum factor) and its Pow sibling POW-EXPAND (an integer power of a sum, expanded through the
//! same product builder), and their reverse FACTOR (terms of a sum that share a factor, as that
//! factor times the quotient sum, see [`factorings`]). In `permissive` the RECIPROCAL JOIN
//! ([`reciprocal_joins`]) also joins a product's reciprocal sums, which permissive's `pow`
//! holds split. The enumeration in [`local_moves`] is the single extension point for further
//! kinds. One deliberate bound: the search moves only from ACCEPTED valleys, so every win is
//! reachable by depth-1 moves from a valley -- composite hills that need two unaccepted
//! expansions in sequence are out of scope.
//!
//! Shared-state note: the caller's [`PassCtx`] (normal-form memo, step counter) is
//! reused across candidate descents -- same semantics as the chain's own passes; on
//! very large budgets the defense-in-depth step cap can begin refusing steps, which
//! is sound (candidates then settle high and are refused by the ordering test).

use super::expr::{
    add, complexity, distributes_power, fun, keeps_zero_sign, mul, orient_term_classes,
    orientation_near_miss, pow, Cx, Ex,
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

/// The exploration phase. `fix` is the chain's fixpoint for the calling mode; `pass` is that
/// mode's OWN pass context -- the same certified machinery as its chain, byte for byte.
/// `budget` counts candidate descents; 0 disables the phase (the caller's guard makes
/// that the no-call case, so unused exploration is byte-identical behavior).
///
/// One strategy, one acceptance (strictly below the incumbent in [`ordered_below`]): FIRST
/// IMPROVEMENT. A round takes the best state's candidates -- every move ([`local_moves`]:
/// DISTRIBUTE, POW-EXPAND and FACTOR) at every place, each with its ancestors rebuilt through the
/// canonical constructors ([`rebuild`]) into the WHOLE STATE it is -- cheapest rebuilt state first
/// (ties in pre-order), descends each by [`rewrite_pass`] to its fixpoint (or the same
/// `max_passes` truncation the chain itself accepts, which is sound), steps to the first endpoint
/// strictly below the best, and starts the next round there; a round that finds nothing ends the
/// search. The rebuilt price is the order, not a filter: the merges a move enables across its
/// ancestors happen in the constructors that rebuild them, so a candidate that already pays there
/// is tried first, and one whose payoff appears only in its descent (a rule that fires on the
/// factored sum) is still tried. `budget` counts descents: `effort=k` is the first `k` of them.
///
pub fn explore(fix: Ex, pass: &PassCtx, max_passes: usize, budget: usize) -> Ex {
    if budget == 0 {
        return fix;
    }
    search(fix, pass, max_passes, budget)
}

/// The descent of a candidate to its fixpoint under `pass` (or the `max_passes` truncation
/// the chain itself accepts; 0 runs no pass, as in the chain).
fn descend(mut cur: Ex, pass: &PassCtx, max_passes: usize) -> Ex {
    for _ in 0..max_passes {
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

/// A candidate: the place of a move and the moved node.
type Site = (Vec<usize>, Ex);

/// The search of [`explore`]. Returns the best state.
fn search(fix: Ex, pass: &PassCtx, max_passes: usize, budget: usize) -> Ex {
    let view = pass.cx.view;
    let mut best = fix;
    let mut spent = 0usize;
    loop {
        let mut sites: Vec<Site> = Vec::new();
        site_list(&best, pass.cx, None, &mut Vec::new(), &mut sites);
        // The whole states, cheapest first; a state two moves rebuild alike is tried once, and
        // a move that rebuilds the best itself not at all.
        let mut seen: rustc_hash::FxHashSet<Ex> = rustc_hash::FxHashSet::default();
        let mut round: Vec<(u64, Ex)> = Vec::with_capacity(sites.len());
        for (path, moved) in sites {
            let rebuilt = rebuild(&best, &path, moved, pass.cx);
            if rebuilt != best && seen.insert(rebuilt.clone()) {
                round.push((complexity(&rebuilt, view), rebuilt));
            }
        }
        round.sort_by_key(|(mu, _)| *mu);
        let mut accepted: Option<Ex> = None;
        for (_, rebuilt) in round {
            if spent >= budget {
                return best;
            }
            spent += 1;
            let cur = descend(rebuilt, pass, max_passes);
            if ordered_below(&cur, &best, view) {
                accepted = Some(cur);
                break;
            }
        }
        match accepted {
            Some(cur) => best = cur,
            None => return best,
        }
    }
}

/// The places of `e` with a move, in deterministic pre-order (the moves at a node first, then its
/// children in canonical bag order), each with its moved node. `power` is the integer exponent
/// `e` is the base of, if any (FACTOR's licence, see [`factor_moves`]).
fn site_list(e: &Ex, cx: &Cx, power: Option<&Rat>, path: &mut Vec<usize>, out: &mut Vec<Site>) {
    let mut own = Vec::new();
    local_moves(e, cx, power, &mut own);
    for m in own {
        out.push((path.clone(), m));
    }
    match e {
        // `atan2` reads the signs of its arguments' zeros, which a move inside them may trade
        // with a sum's orientation (`expr::atan2_of`): no exploration move inside one.
        Ex::Fun(f, v) if v.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(*f, "atan2") => {}
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            for (i, c) in v.iter().enumerate() {
                path.push(i);
                site_list(c, cx, None, path, out);
                path.pop();
            }
        }
        Ex::Pow(b, x) => {
            let k = match &**x {
                Ex::Num(r) if r.is_integer() => Some(r),
                _ => None,
            };
            path.push(0);
            site_list(b, cx, k, path, out);
            path.pop();
            path.push(1);
            site_list(x, cx, None, path, out);
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

/// The move kinds at one node. THE extension point: a new move is a new arm here (with its own
/// licences), nothing else changes.
fn local_moves(e: &Ex, cx: &Cx, power: Option<&Rat>, out: &mut Vec<Ex>) {
    match e {
        // DISTRIBUTE (the row-156 move): fully distribute a Mul bag over its Add
        // children; and on a sub-multiset, the reverse of FACTOR on some of a sum's terms: one
        // factor into one sum factor, the other factors left outside
        // (`x2*sqrt(x1)*(sqrt(x1) - x3)` is `x2*(x1 - x3*sqrt(x1))`), and the other factors over
        // one term of a sum factor, the rest of the sum left inside (`(x1 + x2 + x3)/x2` is
        // `(x1 + x3)/x2 + 1`).
        Ex::Mul(v) => {
            if let Some(cand) = distribute_product(v, cx) {
                out.push(cand);
            }
            if cx.lossy() {
                out.extend(reciprocal_joins(v, cx));
            }
            for (si, s) in v.iter().enumerate() {
                let Ex::Add(ts) = s else {
                    continue;
                };
                if v.len() > 2 {
                    for (fi, f) in v.iter().enumerate() {
                        if matches!(f, Ex::Add(_)) {
                            continue;
                        }
                        if let Some(inner) = distribute_product(&[f.clone(), s.clone()], cx) {
                            let mut rest: Vec<Ex> = v
                                .iter()
                                .enumerate()
                                .filter(|&(k, _)| k != si && k != fi)
                                .map(|(_, x)| x.clone())
                                .collect();
                            rest.push(inner);
                            out.push(mul(rest, cx));
                        }
                    }
                }
                // The other factors go into two terms, so DISTRIBUTE's licences: no
                // `<constant>` among them, and every piece licensed.
                let rest: Vec<Ex> = v
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != si)
                    .map(|(_, x)| x.clone())
                    .collect();
                if ts.len() > 2
                    && rest.iter().all(|f| !f.contains_const() && licensed(f, cx))
                    && ts.iter().all(|t| licensed(t, cx))
                {
                    for (ti, t) in ts.iter().enumerate() {
                        let left: Vec<Ex> = ts
                            .iter()
                            .enumerate()
                            .filter(|&(k, _)| k != ti)
                            .map(|(_, x)| x.clone())
                            .collect();
                        let left = add(left, cx);
                        let mut a = rest.clone();
                        a.push(left.clone());
                        let mut b = rest.clone();
                        b.push(t.clone());
                        let (a, b) = (mul(a, cx), mul(b, cx));
                        if splits_a_pole(&rest, &[(&left, &a), (t, &b)], cx) {
                            continue;
                        }
                        out.push(add(vec![a, b], cx));
                    }
                }
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
                            out.push(cand);
                        }
                    }
                }
            }
        }
        // FACTOR: the reverse of DISTRIBUTE, at a sum.
        Ex::Add(v) => {
            out.extend(factor_moves(v, cx, power));
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
/// Every licensed factoring is proposed, whether or not it is cheaper by itself: whether it pays
/// is the whole state's question. Under a lone inverse the factored spelling can tie the sum
/// (`1/(x1 - x1*x2)` against `-(1/(x2 - 1))/x1`), while the merges it enables across the product
/// pay (`rootn(x1, 2)/(x1 - x1*x2)` becomes `-(1/pow(x1, 1/2))*(1/(x2 - 1))`); and a rule can fire
/// on the factored sum alone.
fn factor_moves(terms: &[Ex], cx: &Cx, power: Option<&Rat>) -> Vec<Ex> {
    // A cost bound, not a measure: the groups grow with the square of the terms. No sum on srbf's
    // ground truths, a tenth of its predictions or a tenth of the simplify inputs of flash-ansr's
    // training-data generator reaches it.
    if terms.len() > EXPAND_TERM_CAP {
        return Vec::new();
    }
    let negative = power.filter(|k| k.is_negative());
    factorings(terms, cx, &|f| licensed(f, cx))
        .into_iter()
        .filter(|(cand, _)| match (cand, negative) {
            (Ex::Mul(fs), Some(k)) => {
                distributes_power(fs, k, cx) && !(k.is_odd_integer() && moves_a_zero_sign(fs, cx))
            }
            _ => true,
        })
        .map(|(cand, _)| cand)
        .collect()
}

/// THE DISTRIBUTIVE LAW read as factoring, at the sum `terms`: each candidate is the sum with
/// some of its terms replaced by their content `c` times the sum of their quotients, returned with
/// `c`. Every way the engine factors is here, once: the search proposes these moves
/// ([`factor_moves`], with the licences in `licensed`), and the rule load refuses a rule that
/// states one of them ([`is_factoring`], with the licence read off the pattern), since the
/// search owns the law.
///
/// * SHARED FACTORS -- a group of terms is factored by everything its terms all share, each base
///   at its least exponent (`x1*x2 + x1*x3 + x4` is `x4 + x1*(x2 + x3)`). The groups: the terms
///   holding a base, for every base two or more terms share; every two terms that share a base;
///   and every term holding all the two share, each base at an exponent at least theirs (so
///   `x8 + x3*x8 + 1/x8^3` proposes `x8*(x3 + 1) + 1/x8^3`, not only `(x8^4*(x3 + 1) + 1)/x8^3`).
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
    // Every two terms that share a base, and every term holding all the two share (each base at
    // an exponent at least the least of theirs).
    let splits: Vec<SplitTerm> = terms.iter().map(term_factors).collect();
    for (i, (_, fi)) in splits.iter().enumerate() {
        for (j, (_, fj)) in splits[i + 1..].iter().enumerate() {
            let content: Vec<(&Ex, &Rat)> = fi
                .iter()
                .filter_map(|(b, r)| {
                    let (_, s) = fj.iter().find(|(c, _)| c == b)?;
                    Some((
                        b,
                        if s.cmp_exact(r) == std::cmp::Ordering::Less {
                            s
                        } else {
                            r
                        },
                    ))
                })
                .collect();
            if content.is_empty() {
                continue;
            }
            let members: Vec<usize> = (0..terms.len())
                .filter(|&k| {
                    content.iter().all(|&(b, e)| {
                        splits[k]
                            .1
                            .iter()
                            .any(|(c, s)| c == b && s.cmp_exact(e) != std::cmp::Ordering::Less)
                    })
                })
                .collect();
            if !groups.iter().any(|g| g.0 == members && !g.1) {
                groups.push((members, false));
            }
            let pair = vec![i, i + 1 + j];
            if !groups.iter().any(|g| g.0 == pair && !g.1) {
                groups.push((pair, false));
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

/// Does one FACTOR move -- the search's own ([`factor_moves`], with its licence and its
/// negative-power filter), at any sum of the pattern `lhs` -- turn `lhs` into `rhs`? Both
/// canonical under `cx`. A rule that states this is an instance of the distributive law as the
/// search applies it, so the search owns it, and the rule load refuses it (`AcRules::translate`).
///
/// `cx` is the licence context of the search the rule would serve beside: its mode, and
/// certificates that answer for every binding the wildcards' sorts admit
/// (`Engine::law_instance` builds it). A piece the certificates cannot vouch for at every
/// binding -- a `_` wildcard inside it, say -- is unlicensed, so a rule over it stays a rule.
pub(crate) fn is_factoring(lhs: &Ex, rhs: &Ex, cx: &Cx) -> bool {
    fn walk(
        e: &Ex,
        power: Option<&Rat>,
        root: &Ex,
        rhs: &Ex,
        cx: &Cx,
        path: &mut Vec<usize>,
    ) -> bool {
        if let Ex::Add(v) = e {
            for cand in factor_moves(v, cx, power) {
                if rebuild(root, path, cand, cx) == *rhs {
                    return true;
                }
            }
        }
        // The children with the power each is the base of, as `site_list` reads them.
        let kids: Vec<(&Ex, Option<&Rat>)> = match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().map(|c| (c, None)).collect(),
            Ex::Pow(b, x) => {
                let k = match &**x {
                    Ex::Num(r) if r.is_integer() => Some(r),
                    _ => None,
                };
                vec![(&**b, k), (&**x, None)]
            }
            _ => Vec::new(),
        };
        kids.into_iter().enumerate().any(|(i, (k, p))| {
            path.push(i);
            let hit = walk(k, p, root, rhs, cx, path);
            path.pop();
            hit
        })
    }
    walk(lhs, None, lhs, rhs, cx, &mut Vec::new())
}

/// `terms` as their content `c` (every base they all share, each at its least exponent) times
/// the sum of their quotients; returned with `c`. `None` when they share nothing, `c` holds a
/// `<constant>`, or a piece is not `licensed`.
fn factor_shared(terms: &[Ex], cx: &Cx, licensed: &dyn Fn(&Ex) -> bool) -> Option<(Ex, Ex)> {
    let split: Vec<SplitTerm> = terms.iter().map(term_factors).collect();
    // A term that holds one base twice holds two powers of it its constructor would not merge,
    // so its exponent does not split as `b^r = b^c * b^(r - c)` either: it is not factored
    // (soundness, not cost; on srbf's ground truths and a tenth of its predictions and of the
    // simplify inputs of flash-ansr's training-data generator, no answer depends on it).
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

/// The licence of every move of the distributive law, on each piece the identity multiplies or
/// collects (see [`distribute_product`]): finite a.e. (the `!` certificate, the lossy blanket in a
/// lossy mode), and the sound certificate for a piece that spells an infinity.
fn licensed(f: &Ex, cx: &Cx) -> bool {
    if f.contains_infinity() {
        cx.fin_certified(f)
    } else {
        cx.fin_licensed(f)
    }
}

/// READING A AT A DISTRIBUTED POLE (review P3): does moving the product of `piece` into the
/// addends it `meets` -- each as `(co-piece, the addend built)` -- split a pole whose limit the
/// product keeps? In the sound modes a pole is read by its limit. `P*(a + b)` keeps `P`'s
/// one-signed limit (`1/x^2`, `abs(1/x)`, `log(x)`, `x^(-1/2)`) times `sign(a + b)`, while
/// `P*a + P*b` is `inf - inf`, NaN, wherever `a` and `b` differ in sign there:
/// `(1 + x3 - x1 + 5/3*exp(x2))/abs(x3)` split as
/// `5/3*exp(x2)*abs(1/x3) + (x3 + 1 - x1)*abs(1/x3)` is NaN at `x3 = 0` on the whole slab
/// `1 < x1 < 1 + 5/3*exp(x2)`, where the product is `+inf`. The pole is split only where two or
/// more addends keep it: an addend whose co-piece cancelled it is certainly finite
/// (`(x1 + x2 + x3)/x2` is `(x1 + x3)/x2 + 1`), and where one addend holds the pole the sum takes
/// that addend's value there, which is the product's. Where two or more hold it, the piece carries
/// two-signed poles only ([`two_signed_poles`]), whose product keeps no limit, or the co-pieces of
/// those addends keep one strict sign ([`strict_sign`]), which each then takes at the pole. A
/// lossy mode keeps no value at a pole.
///
/// FACTOR runs the law the other way and needs no such licence: the joined product keeps the
/// limit wherever the quotients' sum is nonzero at the pole, and where the quotients differ in
/// sign there the unjoined sum was NaN.
fn splits_a_pole(piece: &[Ex], meets: &[(&Ex, &Ex)], cx: &Cx) -> bool {
    if cx.lossy() {
        return false;
    }
    let held: Vec<&Ex> = meets
        .iter()
        .filter(|(_, addend)| holds_a_pole_of(addend, piece, cx))
        .map(|(co, _)| *co)
        .collect();
    if held.len() < 2 || two_signed_poles(piece, cx) {
        return false;
    }
    let first = strict_sign(held[0], cx);
    first.is_none() || !held[1..].iter().all(|co| strict_sign(co, cx) == first)
}

/// Are all the poles of the product of `members` two-signed, so that it keeps no limit at any?
/// Every member is certainly finite (`Cx::certainly_finite`) or a negative odd power of a
/// certainly finite base that changes sign at its zeros -- a variable, a sum (as the class
/// reading takes a sum to change sign there), or `sin`, `cos`, `sinh`, `tanh`, `atan` or `asinh`
/// of one -- an odd pole at its zero, and no member shares a variable with another's pole, since a
/// factor vanishing there, or a second pole there, can make the product even (`sin(x)/x^3`).
/// Anything else may hold a one-signed pole: an even or fractional power, `abs`, `log`, a
/// function of something not finite.
fn two_signed_poles(members: &[Ex], cx: &Cx) -> bool {
    // A base that changes sign at its zeros: a variable, a sum, or an odd function or cosine of
    // one (simple zeros; `sin(x2^2)` touches zero without changing sign, so its argument is a
    // variable or a sum).
    let crossing = |b: &Ex| match b {
        Ex::Leaf(_) | Ex::Add(_) => true,
        Ex::Fun(op, args) => {
            args.len() == 1
                && matches!(&args[0], Ex::Leaf(_) | Ex::Add(_))
                && ["sin", "cos", "sinh", "tanh", "atan", "asinh"]
                    .iter()
                    .any(|n| cx.view.tok_is(*op, n))
        }
        _ => false,
    };
    let odd_pole = |f: &Ex| match f {
        Ex::Pow(b, k) => {
            matches!(&**k, Ex::Num(r) if r.is_negative() && r.is_odd_integer())
                && crossing(b)
                && cx.certainly_finite(b)
        }
        _ => false,
    };
    let mut poles: Vec<usize> = Vec::new();
    for (i, f) in members.iter().enumerate() {
        if odd_pole(f) {
            poles.push(i);
        } else if !cx.certainly_finite(f) {
            return false;
        }
    }
    poles.iter().all(|&p| {
        let Ex::Pow(b, _) = &members[p] else {
            unreachable!("an odd pole is a power")
        };
        members
            .iter()
            .enumerate()
            .all(|(i, f)| i == p || !shares_a_variable(f, b))
    })
}

/// Does `addend`, built from the product of `pieces` and a co-piece, still hold one of their
/// poles: anywhere inside it, a member that is not certainly finite, or, for a negative power, a
/// negative power of the same base (`log(x)*log(x)` is `log(x)^2`, which holds `log(x)`)? A
/// co-piece that cancelled the pole leaves none (`x2*x2^-1` is `1`), and a pole of the co-piece's
/// own is not the pieces'.
fn holds_a_pole_of(addend: &Ex, pieces: &[Ex], cx: &Cx) -> bool {
    fn inverse_base(f: &Ex) -> Option<&Ex> {
        match f {
            Ex::Pow(b, k) if matches!(&**k, Ex::Num(r) if r.is_negative()) => Some(&**b),
            _ => None,
        }
    }
    fn holds(e: &Ex, f: &Ex) -> bool {
        let here = match inverse_base(f) {
            Some(b) => inverse_base(e) == Some(b),
            None => e == f,
        };
        here || match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|c| holds(c, f)),
            Ex::Pow(x, y) => holds(x, f) || holds(y, f),
            _ => false,
        }
    }
    !cx.certainly_finite(addend)
        && pieces
            .iter()
            .filter(|f| !cx.certainly_finite(f))
            .any(|f| holds(addend, f))
}

/// Do `a` and `b` hold a common leaf?
fn shares_a_variable(a: &Ex, b: &Ex) -> bool {
    fn leaves<'e>(e: &'e Ex, out: &mut Vec<&'e Ex>) {
        match e {
            Ex::Leaf(_) => out.push(e),
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().for_each(|c| leaves(c, out)),
            Ex::Pow(x, y) => {
                leaves(x, out);
                leaves(y, out);
            }
            _ => {}
        }
    }
    let (mut la, mut lb) = (Vec::new(), Vec::new());
    leaves(a, &mut la);
    leaves(b, &mut lb);
    la.iter().any(|x| lb.contains(x))
}

/// FACTOR UNDER AN INVERSE AT A SUM THAT KEEPS THE SIGN OF ITS ZERO (`expr::keeps_zero_sign`,
/// `f64`): on the region where the quotients' sum vanishes, `1/(c*q1 + c*q2)` is `+inf` (the terms
/// cancel to `+0`) while `1/(c*(q1 + q2))` is the infinity of `c`'s sign: does the factored
/// product `fs` hold such a sum beside members that are not certainly positive (`strict_sign`)?
fn moves_a_zero_sign(fs: &[Ex], cx: &Cx) -> bool {
    fs.iter().enumerate().any(|(i, f)| {
        keeps_zero_sign(f, cx) && {
            let co: Vec<Ex> = fs
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, g)| g.clone())
                .collect();
            strict_sign(&Ex::Mul(co), cx) != Some(false)
        }
    })
}

/// The one sign `e` takes wherever it is defined, finite and nonzero (`Some(true)`: negative),
/// read syntactically: a nonzero literal, `pi`, `e`, `exp` or `cosh` of something certainly
/// finite, and the products, integer powers and sums of such. `None` where that is not read.
fn strict_sign(e: &Ex, cx: &Cx) -> Option<bool> {
    match e {
        Ex::Num(r) => (!r.is_zero()).then(|| r.is_negative()),
        Ex::Pi | Ex::E => Some(false),
        Ex::Fun(op, args)
            if args.len() == 1
                && (cx.view.tok_is(*op, "exp") || cx.view.tok_is(*op, "cosh"))
                && cx.certainly_finite(&args[0]) =>
        {
            Some(false)
        }
        Ex::Mul(fs) => fs
            .iter()
            .try_fold(false, |neg, f| Some(neg ^ strict_sign(f, cx)?)),
        Ex::Pow(b, k) => match &**k {
            Ex::Num(r) if r.is_integer() => {
                let s = strict_sign(b, cx)?;
                Some(s && r.is_odd_integer())
            }
            _ => None,
        },
        Ex::Add(ts) => {
            let first = strict_sign(ts.first()?, cx)?;
            ts.iter()
                .all(|t| strict_sign(t, cx) == Some(first))
                .then_some(first)
        }
        _ => None,
    }
}

/// RECIPROCAL JOIN (permissive): the product's reciprocal sums, joined and distributed inside.
/// Permissive's `pow` distributes a reciprocal over a product unconditionally, so `c/(A*B)` is
/// held as `c * A^-1 * B^-1`: no DISTRIBUTE can reach the product inside the reciprocal, and the
/// coefficient can never merge into a sum it divides. The sound modes keep such a reciprocal
/// joined where it is not certified to split, and their DISTRIBUTE expands inside it. This
/// move gives permissive the same reach, as a candidate the search descends and keeps only when
/// it lands strictly below (never a projection):
///
/// * two reciprocal sums: `A^-1 * B^-1 -> (A*B)^-1`, with `A*B` distributed into one sum;
/// * the coefficient into one reciprocal sum: `c * A^-1 -> (A/c)^-1`, with `1/c` distributed
///   over `A`'s terms (`v1/(76.06*(82165.5 - 84.64*v1))` is `v1/(6249720.55 - 6438.02*v1)`);
/// * any other factor into one reciprocal sum: `f * A^-1 -> (A/f)^-1`, with `1/f` distributed
///   (`x7/(x7 - x5)` is `1/(1 - x5/x7)`).
///
/// Both are identities wherever the pieces are finite, the licence `distribute_product`
/// checks; the distributed sum is built by the same constructors, so its coefficients land
/// once (`expr::land`).
fn reciprocal_joins(v: &[Ex], cx: &Cx) -> Vec<Ex> {
    let recips: Vec<usize> = v
        .iter()
        .enumerate()
        .filter(|(_, f)| match f {
            Ex::Pow(b, x) => {
                matches!(&**b, Ex::Add(_)) && matches!(&**x, Ex::Num(r) if *r == Rat::NEG_ONE)
            }
            _ => false,
        })
        .map(|(i, _)| i)
        .collect();
    if recips.is_empty() {
        return Vec::new();
    }
    let base = |i: usize| match &v[i] {
        Ex::Pow(b, _) => (**b).clone(),
        _ => unreachable!("a reciprocal sum"),
    };
    let joined = |inside: Ex, skip: &[usize]| {
        let mut r: Vec<Ex> = v
            .iter()
            .enumerate()
            .filter(|(k, _)| !skip.contains(k))
            .map(|(_, x)| x.clone())
            .collect();
        r.push(pow(inside, Ex::Num(Rat::NEG_ONE), cx));
        mul(r, cx)
    };
    let mut out = Vec::new();
    for (a, &i) in recips.iter().enumerate() {
        for &j in &recips[a + 1..] {
            if let Some(inside) = distribute_product(&[base(i), base(j)], cx) {
                out.push(joined(inside, &[i, j]));
            }
        }
    }
    if let Some(ci) = v.iter().position(|f| matches!(f, Ex::Num(_))) {
        if let Ex::Num(c) = &v[ci] {
            if let Some(inv) = c.checked_inv() {
                for &i in &recips {
                    let pieces = [Ex::Num(inv.clone()), base(i)];
                    if let Some(inside) = distribute_product(&pieces, cx) {
                        out.push(joined(inside, &[ci, i]));
                    }
                }
            }
        }
    }
    // Any other factor into one reciprocal sum: `f * A^-1 -> (A/f)^-1`
    // (`x7/(x7 - x5)` is `1/(1 - x5/x7)`).
    for (fi, f) in v.iter().enumerate() {
        if matches!(f, Ex::Num(_) | Ex::Add(_)) || recips.contains(&fi) {
            continue;
        }
        let f_inv = pow(f.clone(), Ex::Num(Rat::NEG_ONE), cx);
        for &i in &recips {
            if let Some(inside) = distribute_product(&[f_inv.clone(), base(i)], cx) {
                out.push(joined(inside, &[fi, i]));
            }
        }
    }
    out
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
    if !rest.iter().all(|f| licensed(f, cx))
        || !adds.iter().all(|v| v.iter().all(|f| licensed(f, cx)))
    {
        return None;
    }
    // The cartesian picks, in canonical bag order (deterministic).
    let rest_len = rest.len();
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
    let addends: Vec<Ex> = picks.iter().map(|p| mul(p.clone(), cx)).collect();
    // Reading A at a pole ([`splits_a_pole`]): the pieces that end up in more than one addend,
    // the ones `<constant>` independence asks about above. Over one sum the other factors meet
    // its terms; over several, a piece in more than one addend meets several co-pieces, so where
    // two or more addends hold a pole it carries two-signed poles only.
    let rest = &picks[0][..rest_len];
    if let [v] = adds.as_slice() {
        let meets: Vec<(&Ex, &Ex)> = v.iter().zip(&addends).collect();
        if splits_a_pole(rest, &meets, cx) {
            return None;
        }
    } else if !cx.lossy() {
        let pieces: Vec<Ex> = rest
            .iter()
            .chain(adds.iter().copied().flatten())
            .cloned()
            .collect();
        let held = addends
            .iter()
            .filter(|a| holds_a_pole_of(a, &pieces, cx))
            .count();
        let duplicated_pole = |v: &&Vec<Ex>| {
            count / v.len() > 1
                && v.iter()
                    .any(|t| !two_signed_poles(std::slice::from_ref(t), cx))
        };
        if held > 1 && (!two_signed_poles(rest, cx) || adds.iter().any(duplicated_pole)) {
            return None;
        }
    }
    Some(add(addends, cx))
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
