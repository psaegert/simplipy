//! AC pattern matching -- the SEMANTIC WIDENING.
//!
//! A rule rooted at `Add`/`Mul` matches any SUB-MULTISET of a subject bag; the unmatched
//! remainder is preserved around the replacement. Combined with the bags being flat and sorted,
//! this removes both axes of the old invariance defect: there is no operand order to be
//! sensitive to, and no bracketing to hide a match behind.
//!
//! Wildcard sorts are carried over from the shipped matcher unchanged -- the sort rides in the
//! token: `?N` binds a variable leaf only; `_N` binds anything; `!N` binds a variable leaf
//! freely and a composite only under the finite-a.e. certificate; `$N` likewise under
//! finite-and-nonzero-a.e.; `wildcard_all` (LOSSY) binds everything and skips the certificates.
//! The `<constant>` rebind guard is ported as-is: a placeholder bound to a `Const`-containing
//! expression never matches a second occurrence (independent constants).
//!
//! Bag matching detail: pattern elements are tried STRUCTURED-FIRST (non-wildcard elements bind
//! wildcards through their structure), then wildcard elements. A wildcard already bound to a
//! same-kind bag consumes its elements as a sub-multiset (this is how `* $0 inv $0` fires on
//! `Mul[a, b, Pow(Mul[a, b], -1)]` with `$0 = a*b`, the reach the old tree matcher had); an
//! unbound wildcard tries each single element in canonical order, then -- as the last
//! alternative -- ABSORBS every remaining element as a sub-bag. Backtracking is leftmost-first,
//! deterministic, and assignment-COMPLETE: the search is continuation-passing, so a nested
//! member match is resumable and a binding refused by a LATER sibling (or by the caller's
//! acceptance check) is revised rather than fatal. Reach therefore depends on neither the
//! rule's wildcard numbering nor on where the canonical order places a shared subterm
//! (finding F1), and first-match-wins stays well-defined: the first accepted assignment in
//! canonical enumeration order.

use rustc_hash::FxHashMap;

use crate::tokens::{Tok, TokenView};

use super::expr::Ex;

/// Match-time context: view + the two certificates + the LOSSY switch (mirrors the old
/// `match_pattern_with_cert` plumbing; absent certificates fail closed).
pub struct MCx<'a> {
    pub view: &'a TokenView<'a>,
    pub cert_fin: Option<&'a dyn Fn(&Ex) -> bool>,
    pub cert_finnz: Option<&'a dyn Fn(&Ex) -> bool>,
    pub wildcard_all: bool,
}

pub type Binds = FxHashMap<Tok, Ex>;

fn is_variable_leaf(e: &Ex, view: &TokenView) -> bool {
    matches!(e, Ex::Leaf(t) if view.sigil(*t) == 0)
}

/// A/B diagnostic: with `SIMPLIPY_AC_ABSORB_FIRST=1` an unbound trailing wildcard prefers
/// ABSORBING the whole remaining sub-bag over binding single elements (maximal-binding-first
/// instead of the spec'd minimal-binding-first). Both orders are total and deterministic;
/// this flag exists to MEASURE the preference on the corpus rather than argue about it.
fn absorb_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // ARTIFACT-AFFECTING switch: listed in engine.py::ARTIFACT_ENV_SWITCHES (H-042).
    *ON.get_or_init(|| std::env::var("SIMPLIPY_AC_ABSORB_FIRST").as_deref() == Ok("1"))
}

/// May a wildcard of this sort bind this subject? (The certificates run EAGERLY at bind time --
/// unlike the old matcher's post-walk pass -- because bag backtracking must be able to retry a
/// different assignment after a certificate refusal. The per-engine certificate caches keep the
/// repeat cost negligible.)
fn bind_ok(sigil: u8, subject: &Ex, cx: &MCx) -> bool {
    if cx.wildcard_all || is_variable_leaf(subject, cx.view) {
        return true;
    }
    match sigil {
        b'?' => false, // variable leaves only
        b'_' => true,  // any subtree (the pointwise-certified sort)
        b'!' | b'$' => {
            // GROUND subjects (no variable, no <constant>) have no measure space for the
            // a.e. certificates to quantify over -- `inv 0` is one VALUE, not a function, so
            // "finite a.e." degenerates to plain finiteness. The interval certificates are
            // verified CORRECT on ground poles (`finite_ae("inv 0") == false`), so this guard
            // is defense-in-depth, not a bug fix: it makes soundness on ground independent of
            // interval edge behavior, at the cost of also refusing ground FINITE composites
            // (`+ 1 np.e`) that the old matcher would bind. Since fold unification
            // (2026-08-02) nothing rescues those: rational grounds never reach this gate
            // (exact constructor arithmetic collapses them to literals at build, `+ 1 2`
            // -> `3`), while transcendental-bearing grounds stay composite BY DESIGN (no
            // numeric fold exists) and this refusal is permanent -- an accepted, bounded
            // recall loss on `!`/`$` slots, never a soundness cost. (The
            // shipped engine's actual `- inv 0 inv 0 -> 0` gap is the certificate-FREE
            // `_`-sort rule tier, which no bind gate can reach; see the sort ladder notes.)
            if !subject.has_measure_space(cx.view) {
                return false;
            }
            if sigil == b'!' {
                cx.cert_fin.is_some_and(|f| f(subject))
            } else {
                cx.cert_finnz.is_some_and(|f| f(subject))
            }
        }
        _ => false,
    }
}

fn bind(pk: Tok, subject: &Ex, binds: &mut Binds, cx: &MCx) -> bool {
    match binds.get(&pk) {
        Some(existing) => {
            // The independence guard: never rebind through a <constant>-containing binding.
            if existing.contains_const() {
                false
            } else {
                existing == subject
            }
        }
        None => {
            if !bind_ok(cx.view.sigil(pk), subject, cx) {
                return false;
            }
            binds.insert(pk, subject.clone());
            true
        }
    }
}

/// A continuation invoked at every complete assignment. Return `true` to ACCEPT: the search
/// stops and unwinds immediately with the accepted bindings left in place. Return `false` to
/// ask for the NEXT assignment: the matcher undoes the current one and keeps enumerating.
pub type Cont<'c> = &'c mut dyn FnMut(&mut Binds) -> bool;

/// Whole-subtree match: does `pattern` match ALL of `subject` (extending `binds`)? Bags
/// require full cover here (absorption allowed, remainder not). First-solution wrapper over
/// [`matches_each`]: for a pure match question any accepted assignment will do, and the
/// enumeration underneath makes the answer assignment-complete -- a `false` means NO
/// assignment exists, not merely that the first one found did not extend.
#[cfg(test)]
pub fn matches(subject: &Ex, pattern: &Ex, binds: &mut Binds, cx: &MCx) -> bool {
    matches_each(subject, pattern, binds, cx, &mut |_| true)
}

/// Whole-subtree match, enumerating EVERY assignment through the continuation. On acceptance
/// the bindings are left in place; on exhaustion (return `false`) `binds` is restored to its
/// entry state. Enumeration order is canonical and total -- alternatives at each choice point
/// run in canonical subject order, and a member match exhausts its own internal assignments
/// before the search moves on -- so "the first accepted assignment" is well-defined and every
/// caller stays deterministic.
pub fn matches_each(subject: &Ex, pattern: &Ex, binds: &mut Binds, cx: &MCx, k: Cont) -> bool {
    match pattern {
        Ex::Leaf(pk) if cx.view.sigil(*pk) != 0 => {
            let fresh = !binds.contains_key(pk);
            if !bind(*pk, subject, binds, cx) {
                return false;
            }
            if k(binds) {
                return true;
            }
            if fresh {
                binds.remove(pk);
            }
            false
        }
        Ex::Leaf(pk) => matches!(subject, Ex::Leaf(s) if s == pk) && k(binds),
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const => {
            subject == pattern && k(binds)
        }
        Ex::Fun(pf, pa) => match subject {
            Ex::Fun(sf, sa) if sf == pf && sa.len() == pa.len() => {
                matches_seq(sa, pa, 0, binds, cx, k)
            }
            _ => false,
        },
        Ex::Pow(pb, pe) => match subject {
            Ex::Pow(sb, se) => matches_each(sb, pb, binds, cx, &mut |b| {
                matches_each(se, pe, b, cx, &mut *k)
            }),
            _ => false,
        },
        Ex::Add(pv) => match subject {
            Ex::Add(sv) => match_bag_each(pv, sv, BagKind::Add, false, binds, cx, &mut |b, _| k(b)),
            _ => false,
        },
        Ex::Mul(pv) => match subject {
            Ex::Mul(sv) => match_bag_each(pv, sv, BagKind::Mul, false, binds, cx, &mut |b, _| k(b)),
            _ => false,
        },
    }
}

/// CPS chain over parallel child slices (`Fun` arguments): child `i` enumerates, and each of
/// its assignments continues into child `i + 1`; a later child's failure RESUMES the earlier
/// children's enumeration -- the rollback the old pairwise loop approximated with a bindings
/// snapshot, now exact and allocation-free.
fn matches_seq(sub: &[Ex], pat: &[Ex], i: usize, binds: &mut Binds, cx: &MCx, k: Cont) -> bool {
    if i == pat.len() {
        return k(binds);
    }
    matches_each(&sub[i], &pat[i], binds, cx, &mut |b| {
        matches_seq(sub, pat, i + 1, b, cx, &mut *k)
    })
}

#[derive(Copy, Clone, PartialEq)]
pub enum BagKind {
    Add,
    Mul,
}

fn make_bag(kind: BagKind, mut v: Vec<Ex>) -> Ex {
    debug_assert!(!v.is_empty());
    if v.len() == 1 {
        return v.pop().unwrap();
    }
    match kind {
        BagKind::Add => Ex::Add(v),
        BagKind::Mul => Ex::Mul(v),
    }
}

/// Match a bag pattern against a bag subject; first-solution wrapper over
/// [`match_bag_each`]. Returns the REMAINDER (unmatched subject elements, canonical order
/// preserved) on success. `allow_remainder = true` is the applied-node (root) mode -- the
/// widening; `false` requires full cover (nested positions), which absorption can still
/// achieve.
#[cfg(test)]
pub fn match_bag(
    pv: &[Ex],
    sv: &[Ex],
    kind: BagKind,
    allow_remainder: bool,
    binds: &mut Binds,
    cx: &MCx,
) -> Option<Vec<Ex>> {
    let mut rem: Option<Vec<Ex>> = None;
    match_bag_each(pv, sv, kind, allow_remainder, binds, cx, &mut |_, used| {
        rem = Some(
            sv.iter()
                .zip(used.iter())
                .filter(|(_, &u)| !u)
                .map(|(e, _)| e.clone())
                .collect(),
        );
        true
    });
    rem
}

/// Bag match, enumerating every assignment. The continuation receives the bindings and the
/// consumed-element mask (`used[i]` == subject element `i` is covered by the pattern); the
/// caller derives the remainder from the mask. Acceptance/rollback protocol as in
/// [`matches_each`]. In full-cover mode (`allow_remainder = false`) the mask is all-true by
/// the time the continuation runs.
pub fn match_bag_each(
    pv: &[Ex],
    sv: &[Ex],
    kind: BagKind,
    allow_remainder: bool,
    binds: &mut Binds,
    cx: &MCx,
    k: &mut dyn FnMut(&mut Binds, &[bool]) -> bool,
) -> bool {
    if pv.len() > sv.len() {
        return false;
    }
    // Order pattern elements structured-first (stable within each class): structure binds
    // wildcards cheaply and prunes the search; pure wildcards go last so bound-wildcard
    // consumption is deterministic and unbound choices see the smallest leftover set.
    let mut order: Vec<usize> = (0..pv.len()).collect();
    order.sort_by_key(|&i| matches!(&pv[i], Ex::Leaf(t) if cx.view.sigil(*t) != 0) as u8);

    let mut used = vec![false; sv.len()];
    try_assign(
        &order,
        0,
        pv,
        sv,
        kind,
        allow_remainder,
        &mut used,
        binds,
        cx,
        k,
    )
}

/// Recursive backtracking over pattern elements in `order[at..]`, continuation-passing: a
/// COMPLETE placement calls `k`; `k` returning `false` resumes the search at the innermost
/// choice point. (The pre-CPS version returned `bool` from nested member matches, freezing
/// their first internal assignment -- an assignment refused by a LATER sibling sharing a
/// wildcard, or by the caller's acceptance check, could never be revised, so rule reach
/// depended on where the canonical order placed a shared subterm: finding F1.)
#[allow(clippy::too_many_arguments)]
fn try_assign(
    order: &[usize],
    at: usize,
    pv: &[Ex],
    sv: &[Ex],
    kind: BagKind,
    allow_remainder: bool,
    used: &mut Vec<bool>,
    binds: &mut Binds,
    cx: &MCx,
    k: &mut dyn FnMut(&mut Binds, &[bool]) -> bool,
) -> bool {
    if at == order.len() {
        // All pattern elements placed. In nested (full-cover) mode every subject element must
        // be consumed; in root mode leftovers become the remainder.
        if !allow_remainder && !used.iter().all(|&u| u) {
            return false;
        }
        return k(binds, used);
    }
    let p = &pv[order[at]];
    let is_wild = matches!(p, Ex::Leaf(t) if cx.view.sigil(*t) != 0);

    if is_wild {
        let pk = match p {
            Ex::Leaf(t) => *t,
            _ => unreachable!(),
        };
        if let Some(bound) = binds.get(&pk).cloned() {
            // Bound wildcard: consumption is DETERMINISTIC. A same-kind bag consumes its
            // elements as a sub-multiset; anything else consumes one equal element. The
            // rebind-through-<constant> guard applies as everywhere.
            if bound.contains_const() {
                return false;
            }
            let bag_elems: Option<&[Ex]> = match (&bound, kind) {
                (Ex::Add(b), BagKind::Add) => Some(b),
                (Ex::Mul(b), BagKind::Mul) => Some(b),
                _ => None,
            };
            match bag_elems {
                Some(belems) => {
                    // Multiset-subtract: each bound element consumes one unused equal subject
                    // element.
                    let mut taken: Vec<usize> = Vec::with_capacity(belems.len());
                    'outer: for be in belems {
                        for (i, se) in sv.iter().enumerate() {
                            if !used[i] && !taken.contains(&i) && se == be {
                                taken.push(i);
                                continue 'outer;
                            }
                        }
                        return false;
                    }
                    for &i in &taken {
                        used[i] = true;
                    }
                    if try_assign(
                        order,
                        at + 1,
                        pv,
                        sv,
                        kind,
                        allow_remainder,
                        used,
                        binds,
                        cx,
                        k,
                    ) {
                        return true;
                    }
                    for &i in &taken {
                        used[i] = false;
                    }
                    false
                }
                None => {
                    for i in 0..sv.len() {
                        if used[i] || sv[i] != bound {
                            continue;
                        }
                        used[i] = true;
                        if try_assign(
                            order,
                            at + 1,
                            pv,
                            sv,
                            kind,
                            allow_remainder,
                            used,
                            binds,
                            cx,
                            k,
                        ) {
                            return true;
                        }
                        used[i] = false;
                        // Equal elements are interchangeable: trying further copies of the
                        // same value cannot change the outcome.
                        break;
                    }
                    false
                }
            }
        } else {
            // Unbound wildcard: the spec'd order is each single element (canonical order)
            // first, then absorption of the whole remaining sub-bag as the last alternative
            // (minimal-binding-first). `SIMPLIPY_AC_ABSORB_FIRST=1` measures the reverse.
            let try_singles = |used: &mut Vec<bool>,
                               binds: &mut Binds,
                               k: &mut dyn FnMut(&mut Binds, &[bool]) -> bool|
             -> bool {
                for i in 0..sv.len() {
                    if used[i] {
                        continue;
                    }
                    if !bind(pk, &sv[i], binds, cx) {
                        continue;
                    }
                    used[i] = true;
                    if try_assign(
                        order,
                        at + 1,
                        pv,
                        sv,
                        kind,
                        allow_remainder,
                        used,
                        binds,
                        cx,
                        &mut *k,
                    ) {
                        return true;
                    }
                    used[i] = false;
                    binds.remove(&pk);
                }
                false
            };
            // Absorption: this wildcard takes EVERY remaining element (>= 2, else it is the
            // single-element case). Only meaningful as the last pattern element still to
            // place -- later elements would find nothing left.
            let try_absorb = |used: &mut Vec<bool>,
                              binds: &mut Binds,
                              k: &mut dyn FnMut(&mut Binds, &[bool]) -> bool|
             -> bool {
                if at != order.len() - 1 {
                    return false;
                }
                let free: Vec<usize> = (0..sv.len()).filter(|&i| !used[i]).collect();
                if free.len() < 2 {
                    return false;
                }
                let sub = make_bag(kind, free.iter().map(|&i| sv[i].clone()).collect());
                if !bind(pk, &sub, binds, cx) {
                    return false;
                }
                for &i in &free {
                    used[i] = true;
                }
                if try_assign(
                    order,
                    at + 1,
                    pv,
                    sv,
                    kind,
                    allow_remainder,
                    used,
                    binds,
                    cx,
                    &mut *k,
                ) {
                    return true;
                }
                for &i in &free {
                    used[i] = false;
                }
                binds.remove(&pk);
                false
            };
            // The arms differ in ATTEMPT ORDER only -- but order is semantic here: both
            // calls short-circuit and mutate the search state, so the first strategy tried
            // decides which assignment the continuation sees first (the absorb-first
            // heuristic). `match` spells that out where clippy reads `if a {x||y} else
            // {y||x}` as duplicate blocks.
            match absorb_first() {
                true => try_absorb(used, binds, &mut *k) || try_singles(used, binds, &mut *k),
                false => try_singles(used, binds, &mut *k) || try_absorb(used, binds, &mut *k),
            }
        }
    } else {
        // Structured pattern element: enumerate subject elements AND, per element, every
        // internal assignment of the member match (the F1 fix -- the continuation resumes
        // INSIDE the member before the search moves to the next subject element).
        let mut tried_shapes: Vec<Ex> = Vec::new();
        for i in 0..sv.len() {
            if used[i] {
                continue;
            }
            // Equal elements are interchangeable: skip duplicates of a shape whose whole
            // assignment enumeration already failed this slot under the same binding state.
            if tried_shapes.iter().any(|e| e == &sv[i]) {
                continue;
            }
            used[i] = true;
            let accepted = matches_each(&sv[i], p, binds, cx, &mut |b| {
                try_assign(
                    order,
                    at + 1,
                    pv,
                    sv,
                    kind,
                    allow_remainder,
                    used,
                    b,
                    cx,
                    &mut *k,
                )
            });
            if accepted {
                return true;
            }
            used[i] = false;
            tried_shapes.push(sv[i].clone());
        }
        false
    }
}

/// Substitute bound wildcards into a rule RHS template. A wildcard with no binding panics,
/// mirroring the old `apply_mapping` (the mining invariant guarantees RHS wildcards are a
/// subset of LHS wildcards).
pub fn substitute(template: &Ex, binds: &Binds, view: &TokenView) -> Ex {
    match template {
        Ex::Leaf(t) if view.sigil(*t) != 0 => binds
            .get(t)
            .cloned()
            .expect("rhs wildcard is bound by the lhs"),
        Ex::Leaf(_)
        | Ex::Num(_)
        | Ex::Pi
        | Ex::E
        | Ex::PosInf
        | Ex::NegInf
        | Ex::NaN
        | Ex::Const => template.clone(),
        Ex::Add(v) => Ex::Add(v.iter().map(|e| substitute(e, binds, view)).collect()),
        Ex::Mul(v) => Ex::Mul(v.iter().map(|e| substitute(e, binds, view)).collect()),
        Ex::Pow(b, e) => Ex::Pow(
            Box::new(substitute(b, binds, view)),
            Box::new(substitute(e, binds, view)),
        ),
        Ex::Fun(f, v) => Ex::Fun(*f, v.iter().map(|e| substitute(e, binds, view)).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{OperatorSpec, Operators};
    use crate::tokens::{TokenOverlay, TokenTable, TokenView};
    use std::cell::RefCell;

    use super::super::expr::{fun, mul, pow, Cx};

    fn test_ops() -> (Vec<String>, Operators) {
        let mut order = Vec::new();
        let mut specs: rustc_hash::FxHashMap<String, OperatorSpec> = Default::default();
        for (n, a) in [
            ("+", 2u8),
            ("*", 2),
            ("pow", 2),
            ("sin", 1),
            ("cos", 1),
            ("tan", 1),
            ("log", 1),
        ] {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: vec![],
                    inverse: None,
                    arity: a,
                    precedence: None,
                    commutative: n == "+" || n == "*",
                },
            );
        }
        let ops = Operators::from_specs(order.clone(), specs);
        (order, ops)
    }

    fn with_view<R>(f: impl FnOnce(&TokenView) -> R) -> R {
        let (order, ops) = test_ops();
        let table = TokenTable::build(&order, &ops);
        let overlay = RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        f(&view)
    }

    fn mcx<'a>(view: &'a TokenView<'a>) -> MCx<'a> {
        MCx {
            view,
            cert_fin: None,
            cert_finnz: None,
            wildcard_all: false,
        }
    }

    #[test]
    fn bag_submultiset_with_remainder() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let m = mcx(view);
            let x = Ex::Leaf(view.intern("x0"));
            let w = view.intern("_0");
            // pattern: tan(_0) * cos(_0); subject: a * tan(x) * cos(x) -- remainder [a].
            let a = Ex::Leaf(view.intern("x9"));
            let pat = vec![
                fun(view.intern("tan"), vec![Ex::Leaf(w)], &cx),
                fun(view.intern("cos"), vec![Ex::Leaf(w)], &cx),
            ];
            let subj = match mul(
                vec![
                    a.clone(),
                    fun(view.intern("tan"), vec![x.clone()], &cx),
                    fun(view.intern("cos"), vec![x.clone()], &cx),
                ],
                &cx,
            ) {
                Ex::Mul(v) => v,
                other => panic!("expected Mul, got {other:?}"),
            };
            let mut binds = Binds::default();
            let rem =
                match_bag(&pat, &subj, BagKind::Mul, true, &mut binds, &m).expect("must match");
            assert_eq!(rem, vec![a]);
            assert_eq!(binds.get(&w), Some(&x));
        });
    }

    #[test]
    fn bound_wildcard_consumes_subbag() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let m = mcx(view);
            // pattern bag: [_0, log(_0)] over Add; subject: a + b + log(a + b). The structured
            // element binds _0 := Add[a, b]; the bag wildcard then consumes {a, b} as a
            // sub-multiset -- the reach the old tree matcher had via bracketed subtrees.
            // (An INVERSE-of-product case cannot arise here: integer powers distribute over
            // Mul in canon, so `(a*b)^-1` never survives as a bag element.)
            let w = view.intern("_0");
            let a = Ex::Leaf(view.intern("x0"));
            let b = Ex::Leaf(view.intern("x1"));
            let ab = super::super::expr::add(vec![a.clone(), b.clone()], &cx);
            let log_ab = fun(view.intern("log"), vec![ab.clone()], &cx);
            let subj = vec![a.clone(), b.clone(), log_ab];
            let pat = vec![Ex::Leaf(w), fun(view.intern("log"), vec![Ex::Leaf(w)], &cx)];
            let mut binds = Binds::default();
            let rem = match_bag(&pat, &subj, BagKind::Add, true, &mut binds, &m)
                .expect("must match via sub-bag consumption");
            assert!(rem.is_empty());
            assert_eq!(binds.get(&w), Some(&ab));
            let _ = pow(a, Ex::int(2), &cx); // keep the import exercised
        });
    }

    #[test]
    fn unbound_wildcard_absorbs_rest_when_nested() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let m = mcx(view);
            // pattern log(_0 * _1) -- nested Mul must fully cover, absorption closes it.
            let w0 = view.intern("_0");
            let w1 = view.intern("_1");
            let (a, b, c) = (
                Ex::Leaf(view.intern("x0")),
                Ex::Leaf(view.intern("x1")),
                Ex::Leaf(view.intern("x2")),
            );
            let subj = fun(
                view.intern("log"),
                vec![mul(vec![a.clone(), b.clone(), c.clone()], &cx)],
                &cx,
            );
            let pat = fun(
                view.intern("log"),
                vec![Ex::Mul(vec![Ex::Leaf(w0), Ex::Leaf(w1)])],
                &cx,
            );
            let mut binds = Binds::default();
            assert!(matches(&subj, &pat, &mut binds, &m));
            // _0 takes the first element, _1 absorbs the remaining two.
            assert_eq!(binds.get(&w0), Some(&a));
            assert_eq!(binds.get(&w1), Some(&mul(vec![b, c], &cx)));
        });
    }

    #[test]
    fn sorts_gate_composites() {
        with_view(|view| {
            let m = mcx(view);
            let cx = Cx::bare(view);
            let q = view.intern("?0");
            let bang = view.intern("!0");
            let x = Ex::Leaf(view.intern("x0"));
            let comp = fun(view.intern("sin"), vec![x.clone()], &cx);
            let mut binds = Binds::default();
            // ? binds a variable...
            assert!(matches(&x, &Ex::Leaf(q), &mut binds, &m));
            // ...but never a composite or a literal.
            binds.clear();
            assert!(!matches(&comp, &Ex::Leaf(q), &mut binds, &m));
            binds.clear();
            assert!(!matches(&Ex::int(3), &Ex::Leaf(q), &mut binds, &m));
            // ! without a certifier: variable yes, composite no (fail-closed).
            binds.clear();
            assert!(matches(&x, &Ex::Leaf(bang), &mut binds, &m));
            binds.clear();
            assert!(!matches(&comp, &Ex::Leaf(bang), &mut binds, &m));
            // ! with a certifier: composite binds.
            let yes = |_: &Ex| true;
            let m2 = MCx {
                view,
                cert_fin: Some(&yes),
                cert_finnz: None,
                wildcard_all: false,
            };
            binds.clear();
            assert!(matches(&comp, &Ex::Leaf(bang), &mut binds, &m2));
            // wildcard_all: everything binds.
            let m3 = MCx {
                view,
                cert_fin: None,
                cert_finnz: None,
                wildcard_all: true,
            };
            binds.clear();
            assert!(matches(&comp, &Ex::Leaf(q), &mut binds, &m3));
        });
    }

    #[test]
    fn bag_matching_is_assignment_complete() {
        // The factoring pattern `Add[Mul[_0,_1], Mul[_0,_2]]` matches `t*a + t*b` for EVERY
        // canonical arrangement of the subject bags -- whether the shared factor sorts
        // first, last, or is a composite. A matcher that commits to the first internal
        // assignment of a member match loses the arrangements where the shared factor does
        // not sort first (finding F1: reach must not depend on where the canonical order
        // happens to place the shared subterm).
        with_view(|view| {
            let cx = Cx::bare(view);
            let m = mcx(view);
            let (w0, w1, w2) = (view.intern("_0"), view.intern("_1"), view.intern("_2"));
            let pat = Ex::Add(vec![
                Ex::Mul(vec![Ex::Leaf(w0), Ex::Leaf(w1)]),
                Ex::Mul(vec![Ex::Leaf(w0), Ex::Leaf(w2)]),
            ]);
            let x = |n: usize| Ex::Leaf(view.intern(&format!("x{n}")));
            let subj = |shared: Ex, a: Ex, b: Ex| {
                super::super::expr::add(
                    vec![mul(vec![shared.clone(), a], &cx), mul(vec![shared, b], &cx)],
                    &cx,
                )
            };

            // Control: the shared factor sorts FIRST in the subject Mul bags.
            let lucky = subj(x(0), x(1), x(2));
            let mut binds = Binds::default();
            assert!(
                matches(&lucky, &pat, &mut binds, &m),
                "control: shared-factor-first arrangement must match"
            );
            assert_eq!(binds.get(&w0), Some(&x(0)));

            // Same mathematical shape, renamed so the shared factor sorts LAST.
            let unlucky = subj(x(8), x(0), x(1));
            binds.clear();
            assert!(
                matches(&unlucky, &pat, &mut binds, &m),
                "shared factor sorting last must not gate the match"
            );
            assert_eq!(binds.get(&w0), Some(&x(8)));

            // Shared factor is a composite (sorts after plain variables).
            let sh = fun(view.intern("sin"), vec![x(0)], &cx);
            let comp = subj(sh.clone(), x(1), x(2));
            binds.clear();
            assert!(
                matches(&comp, &pat, &mut binds, &m),
                "composite shared factor must not gate the match"
            );
            assert_eq!(binds.get(&w0), Some(&sh));

            // The applied-node widening (remainder mode) with an extra term, unlucky names.
            let wide = super::super::expr::add(
                vec![mul(vec![x(8), x(0)], &cx), mul(vec![x(8), x(1)], &cx), x(9)],
                &cx,
            );
            let (pv, sv) = match (&pat, &wide) {
                (Ex::Add(pv), Ex::Add(sv)) => (pv, sv),
                other => panic!("expected Add bags, got {other:?}"),
            };
            binds.clear();
            let rem = match_bag(pv, sv, BagKind::Add, true, &mut binds, &m)
                .expect("widening must match modulo the remainder");
            assert_eq!(rem, vec![x(9)]);
            assert_eq!(binds.get(&w0), Some(&x(8)));
        });
    }

    #[test]
    fn const_rebind_guard() {
        with_view(|view| {
            let m = mcx(view);
            let w = view.intern("_0");
            // subject bag: two structurally equal <constant>-containing terms; a nonlinear
            // pattern `_0 + _0` must NOT match through the Const-bearing binding.
            let cxterm = || Ex::Mul(vec![Ex::Const, Ex::Leaf(view.intern("x0"))]);
            let subj = vec![cxterm(), cxterm()];
            let pat = vec![Ex::Leaf(w), Ex::Leaf(w)];
            let mut binds = Binds::default();
            assert!(match_bag(&pat, &subj, BagKind::Add, false, &mut binds, &m).is_none());
        });
    }

    #[test]
    fn substitution_rebuilds() {
        with_view(|view| {
            let w = view.intern("_0");
            let x = Ex::Leaf(view.intern("x0"));
            let mut binds = Binds::default();
            binds.insert(w, x.clone());
            let template = Ex::Mul(vec![Ex::int(2), Ex::Leaf(w)]);
            assert_eq!(
                substitute(&template, &binds, view),
                Ex::Mul(vec![Ex::int(2), x])
            );
        });
    }
}
