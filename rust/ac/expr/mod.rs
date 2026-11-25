//! The AC expression type and its CANONICAL CONSTRUCTORS.
//!
//! Canonical invariants (established by [`add`], [`mul`], [`pow`], [`fun`]; every consumer may
//! rely on them):
//! * `Add(v)`: `v.len() >= 2`; no element is an `Add`; at most one `Num` (0 only in `f64`, the
//!   `+0` term kept where every other term may be -0 on a region, see `signed_zero_null`); at most
//!   one `Const`; like terms collected as far as the soundness gates allow; sorted by [`cmp_ex`].
//! * `Mul(v)`: `v.len() >= 2`; no element is a `Mul`; at most one `Num` (never 1; 0 only in the
//!   uncertified `0 * t` case documented at the zero-coefficient gate); at most one bare `Const`;
//!   like bases collected under the gates, and at most one power of e (`exp(a) exp(b)` is
//!   `exp(a + b)`); sorted by [`cmp_ex`].
//! * `Pow(b, e)`: `e` is never `Num(0)` or `Num(1)`; `b` is never a `Mul` when `e` is a
//!   non-negative or even integer (integer powers distribute) -- for ODD NEGATIVE `e` the
//!   distribution carries a zero-set licence (see `pow`), so a refused `Pow(Mul[..], e)`
//!   is a legal canonical form; rational^rational is folded when exact.
//! * No element anywhere is a nested bag of the same operator (Flat), and bags carry no order
//!   information beyond the canonical sort (Orderless).
//!
//! Soundness: every transform here is licensed either as TOTAL (equal as extended-real
//! evaluations everywhere), or as almost-everywhere equal with the null set certified by the
//! caller-provided certificates (the same two the shipped engine uses: `!` = finite a.e., `$` =
//! finite-and-nonzero a.e.), or as a `forall c_s exists c_t` fitted-constant absorption per the
//! contract. Each gate cites its counterexample. Refusing to transform is always sound; the
//! constructors prefer refusal over any uncertified step.

use crate::engine::RuleMode;
use std::cmp::Ordering;

use crate::tokens::{Tok, TokenView};

use super::laws::{Law, Powers};
use super::rat::{Mantissa, Rat};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;

mod add;
mod coef;
mod cx;
mod ex;
mod flat;
mod mul;
mod order;
mod price;
mod terms;
mod zero_sign;

pub use self::add::*;
pub(crate) use self::coef::*;
pub use self::cx::*;
pub use self::ex::*;
pub(crate) use self::flat::*;
pub use self::mul::*;
pub use self::order::*;
pub use self::price::*;
pub(crate) use self::terms::*;
pub use self::zero_sign::*;

/// Is `e` an odd root (`rootn` of an odd index), or a power of one? Its collection can leave its
/// class (`cbrt(u)^3` is `u`), which the certificate-free parse already decides; a divisor
/// holding one is distributed by its members' form there too (`power_bag`), so its members meet
/// that decision as the same members spelled one by one do.
fn holds_odd_root(e: &Ex, cx: &Cx) -> bool {
    match e {
        Ex::Pow(b, _) => holds_odd_root(b, cx),
        Ex::Fun(op, a) => {
            a.len() == 2
                && cx.view.tok_is(*op, "rootn")
                && matches!(&a[1], Ex::Num(k) if k.is_odd_integer())
        }
        _ => false,
    }
}

/// Is `e` an infinity, or a sum holding one (an infinite or NaN value, never zero)?
fn holds_infinity(e: &Ex) -> bool {
    match e {
        Ex::PosInf | Ex::NegInf => true,
        Ex::Add(ts) => ts
            .iter()
            .any(|t| matches!(t, Ex::PosInf | Ex::NegInf) || is_inf_carrying(t)),
        _ => false,
    }
}

/// Where a product's sign lives once its factors are collected (`mul()`'s tail): the rational
/// coefficient, a bare `Const` (whose refit absorbs any sign), or a bare infinity (which carries
/// it).
enum SignHost {
    Coeff(Rat),
    Const(ConstName),
    Inf(bool),
}

impl SignHost {
    /// The product of `out` with the host's sign toggled when `neg`, assembled as `mul()` does;
    /// `None` where the coefficient's negation overflows.
    fn build(&self, neg: bool, mut out: Vec<Ex>, cx: &Cx) -> Option<Ex> {
        let coeff = match self {
            SignHost::Coeff(c) if neg => c.checked_neg()?,
            SignHost::Coeff(c) => c.clone(),
            SignHost::Const(n) => {
                out.push(Ex::Const(*n));
                Rat::ONE
            }
            SignHost::Inf(pos) => {
                out.push(if *pos != neg { Ex::PosInf } else { Ex::NegInf });
                Rat::ONE
            }
        };
        Some(finish_product(coeff, out, cx))
    }

    fn key(&self) -> HostKey {
        match self {
            SignHost::Coeff(c) => HostKey::Coeff(c.clone()),
            SignHost::Const(n) => HostKey::Const(*n),
            SignHost::Inf(pos) => HostKey::Inf(*pos),
        }
    }
}

/// A sign host as part of a memo key (`OPPOSITE_MEMO`).
#[derive(Clone, PartialEq, Eq, Hash)]
enum HostKey {
    Coeff(Rat),
    Const(ConstName),
    Inf(bool),
}

/// OPPOSITE SUMS (see `mul()`): a class {B, -B} of a product's factors, where `B` is a sum `S`
/// (its powers, and its even roots read as powers) or an odd function of one (`sin(S)`,
/// `tanh(S)`, an odd `rootn(S, k)`, and their powers: `f(-S) = -f(S)`), and `-B` is the same
/// over the negation of `S`. An even function needs no class: `fun` files `cos(S)` and
/// `cos(-S)` in one orientation already.
///
/// A class is read as a TABLE of exponent totals per orientation, whatever factors delivered
/// them: on each side the positive integer exponents add up to `p` and the negative ones to `n`;
/// a fractional exponent gives its integer part (taken toward zero) to `p` or `n` and stays with
/// the rest (`S^(5/2) = S^2 * S^(1/2)`), and a symbolic one stays whole. What may move between
/// the two sides follows from the totals alone:
///
/// * all of `p`, with a sign to the sign host for every odd amount moved
///   (`(-B)^k = (-1)^k B^k`);
/// * the even part of `n`, freely (`(-B)^-2 = B^-2`, `+inf` on both sides at a zero);
/// * nothing else in a sound context. A side whose `n` is odd keeps one `B^-1`: moved, it would
///   flip the sign of the infinity at the zero (the pole trilemma). A lossy context reads a
///   function up to its values at a zero and moves that inverse too, with its sign. A fractional
///   or symbolic power never flips: `(-S)^(1/2)` is undefined where `S^(1/2)` is defined.
///
/// So `(3 - x1)^-3 * sqrt(x1 - 3)` may become `(3 - x1)^-1 * (x1 - 3)^(-3/2)` whether `^-3`
/// arrived as one factor or as `^-2 * ^-1`. Every placement of the movable totals (how each
/// class splits its positive and its movable negative mass between the sides, and on each side
/// between the integer power and the fractional powers of its sign) is priced as the whole
/// product it builds -- each class collected under the context's own licences, the rest of the
/// bag beside them, the sign placed by `finish_product` -- and the cheapest wins; a price tie
/// goes by `sign_place`'s own order (the sign key, then the structurally smaller product). The classes are decided together while their placements combine to at most
/// `JOINT_LIMIT` products, priced in the order of a lower bound on their prices, else one at a
/// time, from a start the tables decide, until none moves; a class with a large total offers its
/// two extreme placements and the one it holds. A product about to be inverted is built without
/// the decision (`mul_undecided`): its members are decided where they meet the inverse.
/// A class whose members all arrived over one sum is decided too where they hold exponents of
/// both signs (`-(x1 - 3) / (x1 - 3)^2` is `1/(3 - x1)`) or fractional powers that may split its
/// mass. Where a collection changes a table (the certified cancellation `S^2 * S^-3 = S^-1`), the
/// new tables are decided again, until none changes; where a collection leaves its class (an odd
/// root's cube is its radicand), the product is decided again from its own members. The decision
/// is a function of the tables, so the product does not depend on the order of its factors or on
/// how they were bracketed, and the product returned is the one `mul()` builds again from its own
/// factors.
///
/// A context without the `$`-certificate (the entry parse, the bare measure, a rule's sides) runs
/// the same decision on the same tables; its collection refuses the merges of opposite-sign
/// exponents the certificate licenses, and that is the only difference. The parse builds a
/// product once from all of its members, so no smaller product a spelling nests is decided on
/// its own first.
fn collect_opposite_sums(host: SignHost, raw: Vec<Ex>, out: Vec<Ex>, cx: &Cx) -> Ex {
    // Under the call's own contexts each distinct product is decided once in the call
    // (`CallDecisions`).
    let ctx_key = trade_cx_key(cx);
    let lived = call_lived(&ctx_key).then(|| (ctx_key, host.key(), raw.clone()));
    if let Some(hit) = lived.as_ref().and_then(|k| {
        CALL_DECISIONS.with(|m| m.borrow().as_ref().and_then(|d| d.opposite.get(k).cloned()))
    }) {
        return hit;
    }
    // Within an enclosing scope (a parse, a re-walk) each distinct product is decided once.
    let key = FLIP_MEMO
        .with(|m| m.borrow().is_some())
        .then(|| (ctx_key, host.key(), raw.clone()));
    if let Some(hit) = key
        .as_ref()
        .and_then(|k| OPPOSITE_MEMO.with(|m| m.borrow().get(k).cloned()))
    {
        return hit;
    }
    // The flip memo spans the whole decision: every priced product places its signs over the
    // same sites.
    let _memo = FlipScope::open();
    let decided = decide_opposite_sums(host, raw, out, cx);
    if let Some(k) = lived {
        CALL_DECISIONS.with(|m| {
            if let Some(d) = m.borrow_mut().as_mut() {
                d.opposite.insert(k, decided.clone());
            }
        });
    }
    if let Some(k) = key {
        OPPOSITE_MEMO.with(|m| m.borrow_mut().insert(k, decided.clone()));
    }
    decided
}

fn decide_opposite_sums(host: SignHost, raw: Vec<Ex>, out: Vec<Ex>, cx: &Cx) -> Ex {
    let mut classes: Vec<Class> = Vec::new();
    for f in &raw {
        let Some(read) = class_read(f, cx) else {
            continue;
        };
        if classes.iter_mut().any(|c| c.place(&read, cx).is_some()) {
            continue;
        }
        classes.push(Class::new(&read));
    }
    // What each class keeps at its zero: the limit, where the product keeps one (`zero_limit`).
    // An isolated zero's sign is no value the answer keeps, and neither is the sign of the infinity
    // a two-signed pole makes of it.
    for c in classes.iter_mut().filter(|c| !c.multiple()) {
        c.kept = class_of(&c.sums[0], cx)
            .map(|(k, _, _)| k)
            .filter(|k| zero_limit(&raw, k, cx).is_some());
    }
    // A class whose factors all arrived over one sum is decided too where they hold exponents of
    // both signs, as the cancellation they meet may be cheaper over the other sum
    // (`-(x1 - 3) / (x1 - 3)^2` is `1/(3 - x1)`), or fractional powers that split its mass
    // (`(x1 - 3)^(1/2) (x1 - 3)^(9/2)` is `(x1 - 3)^(5/2) (x1 - 3)^(5/2)`).
    classes.retain_mut(|c| {
        if !c.sums[1].is_empty() {
            return true;
        }
        let t = c.table(
            raw.iter().filter_map(|f| class_read(f, cx).map(|r| (f, r))),
            cx,
        );
        (t.sides[0].p > 0 && t.sides[0].n < 0 || t.splits_mass(0, cx)) && c.complete(cx)
    });
    // Each class written over its first sum: the structurally smaller one, or for a literal
    // multiple the positive one (as an even power of it files, `pow`).
    for c in classes.iter_mut() {
        let first = if c.multiple() {
            c.sums[1]
                .first()
                .and_then(multiple_literal)
                .is_some_and(|r| !r.is_negative())
        } else {
            cmp_vec(&c.sums[1], &c.sums[0], cx.view) == Ordering::Less
        };
        if first {
            c.sums.swap(0, 1);
        }
    }
    classes.sort_by(|a, b| cmp_ex(&a.base(0, cx.view), &b.base(0, cx.view), cx.view));
    // The rest of the bag: the factors outside every class, collected as `mul()` collects them.
    // It is read from the factors as they arrived, not from `mul()`'s collection, where a class's
    // members can merge into a factor outside it (an odd root's cube is its radicand) that the
    // class's table still counts.
    let outside: Vec<Ex> = raw
        .iter()
        .filter(|f| {
            !class_read(f, cx).is_some_and(|r| classes.iter().any(|c| c.side_of(&r).is_some()))
        })
        .cloned()
        .collect();
    let (rest, spliced) = collect_like_bases(outside, cx);
    if spliced {
        return host
            .build(false, out, cx)
            .expect("the unflipped bag moves no sign");
    }
    // Each class's table, read from the factors as they arrived.
    let mut tables: Vec<Table> = classes
        .iter()
        .map(|c| {
            c.table(
                raw.iter().filter_map(|f| class_read(f, cx).map(|r| (f, r))),
                cx,
            )
        })
        .collect();
    // A class's factors collected until the collection moves nothing (a re-oriented even power
    // meets the factor it merges with on the next round, and a collection that leaves no two
    // factors over one base has nothing left to move); `None` if a rebuilt factor is a product,
    // which no power of a sum or of a function is.
    let pairs = |v: &[Ex]| {
        (0..v.len()).any(|i| (i + 1..v.len()).any(|j| same_collection_base(&v[i], &v[j], cx)))
    };
    let collect = |mut v: Vec<Ex>| -> Option<Vec<Ex>> {
        v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
        for _ in 0..=v.len() {
            let (mut next, spliced) = collect_like_bases(v.clone(), cx);
            if spliced {
                return None;
            }
            next.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            if next == v || !pairs(&next) {
                return Some(next);
            }
            v = next;
        }
        None
    };
    // Cheaper than `(q, m)`: the price, then the tie order (`tie_cmp`: the sign key, then the
    // structure), as the product's sign is placed (`sign_place`), so a product the placement
    // alone would build from either spelling is the one chosen. A normal-form choice: the
    // candidates are admitted by the premises (a placement off a class's value at its zero,
    // `off`, is none), never by the price.
    let better = |mu: u64, p: &Ex, q: &Ex, m: u64| {
        mu < m || (mu == m && tie_cmp(p, q, cx.view) == Ordering::Less)
    };
    // A lower bound on a candidate's price, from its factors alone: `complexity` prices a product
    // as its node, its members and its coefficient, and the sign placement can only choose each
    // member's orientation (at best its cheaper one) and the coefficient's sign (at best the
    // cheaper one). Candidates are priced in the order of their bounds, and none whose bound
    // exceeds the best price found can win. Where an absorption owner of `finish_product` may
    // re-orient a sum of the rest (a `Const`-bearing, negation-absorbing or infinite one) and take
    // the sign or the coefficient into it, the bound counts only the classes' factors and the
    // rest's other factors, at their cheaper orientations.
    // A piece of a literal (`is_piece`) is priced with the coefficient and the product's other
    // pieces as the one literal they denote, which can cost less than its pieces: where the
    // product holds one, the bound counts neither the pieces, nor the coefficient, nor the node.
    let pieces = raw.iter().any(|f| is_piece(f, cx.view));
    let factor_lb = |f: &Ex| {
        if is_piece(f, cx.view) {
            return 0;
        }
        let mu = complexity(f, cx.view);
        match sign_trade_flip(f, cx) {
            Some(nf) => mu.min(complexity(&nf, cx.view)),
            None => mu,
        }
    };
    let absorbing = |f: &Ex| {
        matches!(f, Ex::Add(ts) if ts.iter().any(|t| {
            t.contains_const() || term_absorbs_negation(t) || matches!(t, Ex::PosInf | Ex::NegInf)
        }))
    };
    let bounded = !rest.iter().any(absorbing);
    let rest_lb: u64 = rest
        .iter()
        .filter(|f| !absorbing(f))
        .map(factor_lb)
        .fold(0, u64::saturating_add);
    let (host_lb, host_members, unit) = match &host {
        SignHost::Coeff(c) => {
            let unit = c.is_one() || *c == Rat::NEG_ONE;
            (if unit { 0 } else { mu_rat(&c.abs()) }, 0, unit)
        }
        SignHost::Const(_) => (complexity(&Ex::ANON, cx.view), 1, true),
        SignHost::Inf(_) => (complexity(&Ex::PosInf, cx.view), 1, true),
    };
    let bound = |members: usize, lb: u64| -> u64 {
        if members == 0 {
            return 0;
        }
        if !bounded || pieces {
            return rest_lb.saturating_add(lb);
        }
        let node = if members >= 2 || !unit { mu_mul() } else { 0 };
        node.saturating_add(rest_lb)
            .saturating_add(host_lb)
            .saturating_add(lb)
    };
    // Every combination of one placement per class while there are at most this many; beyond,
    // the classes are decided one at a time, each against the others as they stand.
    const JOINT_LIMIT: usize = 4096;
    // Does a collected placement hold a factor over no side of its class (an odd root's cube is
    // its radicand)?
    let leaves = |k: usize, list: &Option<Vec<Ex>>| {
        list.as_ref().is_some_and(|l| {
            l.iter()
                .any(|f| !class_read(f, cx).is_some_and(|r| classes[k].side_of(&r).is_some()))
        })
    };
    let mut neg = false;
    let mut chosen: Option<Ex> = None;
    let mut escaped = false;
    for _ in 0..8 {
        // Each class's placements, collected (`None` where a collection splices), with the sign
        // each moves relative to the table and the bound of its factors; index 0 is the table's
        // own placement, with its mass beside its fractional powers. Placements that collect to
        // the same factors with the same sign are one option.
        let mut options: Vec<Vec<PlacedClass>> = Vec::with_capacity(classes.len());
        for (c, t) in classes.iter().zip(&tables) {
            let mut o: Vec<PlacedClass> = Vec::new();
            let placed = t
                .placements()
                .into_iter()
                .flat_map(|(pl, odd)| c.factors(t, &pl, cx).into_iter().map(move |v| (v, odd)));
            for (members, odd) in placed {
                let list = collect(members);
                if o.iter().any(|x| x.odd == odd && x.list == list) {
                    continue;
                }
                let lb = list
                    .as_ref()
                    .map_or(0, |l| l.iter().map(factor_lb).fold(0, u64::saturating_add));
                let members = list.as_ref().map_or(0, Vec::len);
                o.push(PlacedClass {
                    list,
                    odd,
                    lb,
                    members,
                });
            }
            options.push(o);
        }
        let held = vec![0usize; classes.len()];
        if chosen.is_none() && options.iter().all(|o| o.len() <= 1) {
            // Nothing can move: the bag as `mul()` collected it.
            escaped = options
                .iter()
                .enumerate()
                .any(|(k, o)| o.first().is_some_and(|x| leaves(k, &x.list)));
            break;
        }
        let bag = |choice: &[usize]| -> Option<(Vec<Ex>, bool)> {
            let mut v = rest.clone();
            let mut odd = false;
            for (k, &i) in choice.iter().enumerate() {
                let x = &options[k][i];
                v.extend(x.list.as_ref()?.iter().cloned());
                odd ^= x.odd;
            }
            Some((v, odd))
        };
        let lower = |choice: &[usize]| -> u64 {
            let (members, lb) = choice.iter().enumerate().fold(
                (rest.len() + host_members, 0u64),
                |(m, b), (k, &i)| {
                    (
                        m + options[k][i].members,
                        b.saturating_add(options[k][i].lb),
                    )
                },
            );
            bound(members, lb)
        };
        // A placement that leaves the product's value at a class's zero off the limit it keeps
        // there (`zero_limit`, as `sign_place` reads it) is priced above every other.
        let off = |p: &Ex| {
            let members = match p {
                Ex::Mul(v) => v.as_slice(),
                p => std::slice::from_ref(p),
            };
            classes
                .iter()
                .filter_map(|c| c.kept.as_ref())
                .any(|k| zero_off(members, k, cx) == Some(true))
        };
        let price = |choice: &[usize]| -> Option<(Ex, u64)> {
            let (v, odd) = bag(choice)?;
            let p = host.build(neg != odd, v, cx)?;
            let mu = if off(&p) {
                u64::MAX
            } else {
                complexity(&p, cx.view)
            };
            debug_assert!(mu >= lower(choice), "a candidate priced below its bound");
            Some((p, mu))
        };
        let Some(mut best) = price(&held) else {
            break;
        };
        let mut pick = held.clone();
        let limit = JOINT_LIMIT;
        let combos = options
            .iter()
            .try_fold(1usize, |n, o| n.checked_mul(o.len()))
            .filter(|n| *n <= limit);
        if let Some(n) = combos {
            let decode = |code: usize| -> Vec<usize> {
                let mut c = code;
                options
                    .iter()
                    .map(|o| {
                        let i = c % o.len();
                        c /= o.len();
                        i
                    })
                    .collect()
            };
            let mut order: Vec<(u64, usize)> =
                (1..n).map(|code| (lower(&decode(code)), code)).collect();
            order.sort_unstable();
            for (lb, code) in order {
                if lb > best.1 {
                    break;
                }
                let choice = decode(code);
                if let Some((p, mu)) = price(&choice) {
                    if better(mu, &p, &best.0, best.1) {
                        best = (p, mu);
                        pick = choice;
                    }
                }
            }
        } else {
            // One class at a time from a start the tables alone decide (each class's option
            // with the smallest bound, a tie by its factors), so the placement the sweeps settle
            // on does not depend on how the members arrived.
            let start: Vec<usize> = options
                .iter()
                .map(|o| {
                    (0..o.len())
                        .filter(|&i| o[i].list.is_some())
                        .min_by(|&a, &b| {
                            o[a].lb
                                .cmp(&o[b].lb)
                                .then_with(|| match (&o[a].list, &o[b].list) {
                                    (Some(x), Some(y)) => cmp_vec(x, y, cx.view),
                                    _ => Ordering::Equal,
                                })
                        })
                        .unwrap_or(0)
                })
                .collect();
            if let Some(s) = price(&start) {
                best = s;
                pick = start;
            }
            loop {
                let mut moved = false;
                for k in 0..options.len() {
                    for i in 0..options[k].len() {
                        if i == pick[k] {
                            continue;
                        }
                        let mut choice = pick.clone();
                        choice[k] = i;
                        if lower(&choice) > best.1 {
                            continue;
                        }
                        if let Some((p, mu)) = price(&choice) {
                            if better(mu, &p, &best.0, best.1) {
                                best = (p, mu);
                                pick = choice;
                                moved = true;
                            }
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        chosen = Some(best.0);
        escaped = (0..classes.len()).any(|k| leaves(k, &options[k][pick[k]].list));
        if escaped || pick == held {
            break;
        }
        // The tables of the chosen placement, read from its collected factors: unchanged unless
        // the collection merged across a total (a cancellation), which is decided again.
        let mut changed = false;
        for k in 0..classes.len() {
            let Some(list) = &options[k][pick[k]].list else {
                break;
            };
            let t = classes[k].table(
                list.iter()
                    .filter_map(|f| class_read(f, cx).map(|r| (f, r))),
                cx,
            );
            neg ^= options[k][pick[k]].odd;
            changed |= !t.same_orbit(&tables[k]);
            tables[k] = t;
        }
        if !changed {
            break;
        }
    }
    let decided = chosen.unwrap_or_else(|| {
        host.build(false, out, cx)
            .expect("the unflipped bag moves no sign")
    });
    // A collection that left its class leaves a product whose classes are not the ones decided,
    // and a placement can leave two odd roots of one index, which `mul()` joins before any
    // decision: either product is built again from its own members, as the parse of its print
    // builds it. Each such round moves a multiple of an odd root's index out of that root's class
    // or joins two odd roots, so it ends.
    match decided {
        Ex::Mul(v) if escaped || holds_joinable_roots(&v, cx) => mul(v, cx),
        d => d,
    }
}

/// Does a product hold two odd roots of one index, which `mul()` joins into one?
fn holds_joinable_roots(v: &[Ex], cx: &Cx) -> bool {
    let mut seen: Vec<&Rat> = Vec::new();
    for f in v {
        if let Ex::Fun(op, a) = f {
            if a.len() == 2 && cx.view.tok_is(*op, "rootn") {
                if let Ex::Num(k) = &a[1] {
                    if k.is_odd_integer() && k.cmp_int(3) != Ordering::Less {
                        if seen.contains(&k) {
                            return true;
                        }
                        seen.push(k);
                    }
                }
            }
        }
    }
    false
}

/// How a factor reads as a member of a class {B, -B} (`collect_opposite_sums`): the head of `B`
/// (`None` for the sum itself, else the odd function and an odd root's index), the terms of its
/// argument (`class_base`; a literal multiple is a sum of one term: `sin(2 x1)` and `sin(-2 x1)`,
/// `sin(x1)` and `sin(-x1)`, `sin(2)` and `sin(-2)`), and the factor's exponent over `B` (`None`
/// when it is symbolic).
struct ClassRead<'a> {
    head: Option<(Tok, Option<&'a Ex>)>,
    terms: &'a [Ex],
    exp: Option<Rat>,
}

/// THE CLASS READING, the one definition every owner asks (the product table `class_read`, the
/// orientation of like terms and of shared factors `orient_base`, the sign trade's classes
/// `site_multiple`, a term read up to sign `odd_signs_out`, and `pow`'s even powers). A class is a
/// pair {B, -B}: a sum and its negation, or an odd head -- an odd function, or an odd root of an
/// odd index of at least 3 -- of a negatable argument and of its negation. A negatable argument is
/// a sum, or a literal multiple `c u` (`multiple_literal`: a bare `u` is `1 u`, a negation `-u` is
/// `-1 u`, a literal is itself). A sum is one only where the sign may pass through it: a sum
/// holding a `Const` or an infinity, or a term that absorbs a negation, has its own owners.
struct ClassBase<'a> {
    head: Option<(Tok, Option<&'a Ex>)>,
    arg: ClassArg<'a>,
}

/// The argument of a class base: a sum's terms, or a literal multiple.
enum ClassArg<'a> {
    Sum(&'a [Ex]),
    Multiple(&'a Ex),
}

impl<'a> ClassArg<'a> {
    /// The argument as the terms of a sum (a literal multiple is a sum of one term).
    fn terms(&self) -> &'a [Ex] {
        match self {
            ClassArg::Sum(ts) => ts,
            ClassArg::Multiple(m) => std::slice::from_ref(*m),
        }
    }
}

fn class_base<'a>(b: &'a Ex, view: &TokenView) -> Option<ClassBase<'a>> {
    let sum = |ts: &'a [Ex]| {
        (!ts.iter()
            .any(|t| t.contains_const() || t.contains_infinity() || term_absorbs_negation(t)))
        .then_some(ClassArg::Sum(ts))
    };
    match b {
        Ex::Add(ts) => Some(ClassBase {
            head: None,
            arg: sum(ts)?,
        }),
        Ex::Fun(op, a) => {
            let (pos, index) = odd_head(view, *op, a)?;
            let arg = match &a[pos] {
                Ex::Add(ts) => sum(ts)?,
                m => {
                    multiple_literal(m)?;
                    ClassArg::Multiple(m)
                }
            };
            Some(ClassBase {
                head: Some((*op, index)),
                arg,
            })
        }
        _ => None,
    }
}

/// The literal of a literal multiple `c u`: `c`, 1 for a bare `u`; `None` for a sum, a zero, and an
/// argument holding a `Const` or an infinity, whose signs have their own owners, and for NaN, which
/// a negation leaves as it is (`-NaN` is NaN), as it leaves a zero: an odd head of either has no
/// other member in its class, so no sign can trade into it.
fn multiple_literal(m: &Ex) -> Option<Rat> {
    if matches!(m, Ex::Add(_) | Ex::NaN) || m.contains_const() || m.contains_infinity() {
        return None;
    }
    let c = match m {
        Ex::Num(r) => r.clone(),
        Ex::Mul(v) => match v.first() {
            Some(Ex::Num(r)) => r.clone(),
            _ => Rat::ONE,
        },
        _ => Rat::ONE,
    };
    (!c.is_zero()).then_some(c)
}

/// `b` over the positive literal of its class's argument, where `b` is an odd head over a literal
/// multiple with a negative literal (`sin(-2 x1)` is `-sin(2 x1)`); raw.
fn positive_head(b: &Ex, view: &TokenView) -> Option<Ex> {
    let ClassBase {
        head: Some(_),
        arg: ClassArg::Multiple(m),
    } = class_base(b, view)?
    else {
        return None;
    };
    let Ex::Fun(op, a) = b else { return None };
    let (pos, _) = odd_head(view, *op, a)?;
    let mut args = a.clone();
    args[pos] = positive_multiple(m)?;
    Some(Ex::Fun(*op, args))
}

fn class_read<'a>(f: &'a Ex, cx: &Cx) -> Option<ClassRead<'a>> {
    let (base, exp) = match f {
        Ex::Pow(b, e) => (
            &**b,
            match &**e {
                Ex::Num(r) => Some(r.clone()),
                _ => None,
            },
        ),
        Ex::Fun(op, a)
            if a.len() == 2
                && cx.view.tok_is(*op, "rootn")
                && matches!(&a[1], Ex::Num(k) if k.is_even_integer() && k.cmp_int(2) != Ordering::Less) =>
        {
            let Ex::Num(k) = &a[1] else { return None };
            (&a[0], Some(k.checked_inv()?))
        }
        other => (other, Some(Rat::ONE)),
    };
    let ClassBase { head, arg } = class_base(base, cx.view)?;
    Some(ClassRead {
        head,
        terms: arg.terms(),
        exp,
    })
}

/// The class of a sign-trade site over a literal multiple (`class_read`'s {B, -B}): `B`, the
/// head over the multiple with its literal made positive, and whether the site spells `-B`.
fn site_multiple(f: &Ex, cx: &Cx) -> Option<(Ex, bool)> {
    let m = match f {
        Ex::Pow(b, q) if matches!(&**q, Ex::Num(_)) => &**b,
        _ => f,
    };
    let ClassBase {
        head: Some(_),
        arg: ClassArg::Multiple(arg),
    } = class_base(m, cx.view)?
    else {
        return None;
    };
    if multiple_literal(arg)?.is_negative() {
        Some((positive_head(m, cx.view)?, true))
    } else {
        Some((m.clone(), false))
    }
}

/// The class a sign-trade site is built over: the sum under the site's head (the sum itself, an
/// odd function or odd root of it, or a power of any of these) in its one orientation, and whether
/// the site spells it in the other one. `None` where the site is over no sum.
fn site_class(f: &Ex, cx: &Cx) -> Option<(Ex, bool)> {
    let m = match f {
        Ex::Pow(b, q) if matches!(&**q, Ex::Num(_)) => &**b,
        _ => f,
    };
    let terms = match m {
        Ex::Add(ts) => ts,
        Ex::Fun(op, a) => match odd_head(cx.view, *op, a).map(|(p, _)| &a[p]) {
            Some(Ex::Add(ts)) => ts,
            _ => return None,
        },
        _ => return None,
    };
    class_of(terms, cx).map(|(k, negative, _)| (k, negative))
}

/// THE CLASS OF A SUM IS ITS ZERO SET: sums equal up to a nonzero constant factor are one class
/// (`6 - 2 x1` is `x1 - 3`'s, `3 - x1` too), and so are sums equal up to a factor every term
/// shares (`3 x2 - x1 x2` is `x2 (3 - x1)`, a member of that class). Its key is the sum's primitive
/// part -- the shared factors divided out, then every term's coefficient divided by the leading
/// term's (the first by the order of the terms' non-numeric parts, a constant term never leading)
/// -- and the sum is that factor times the key: `true` where the constant is negative (the sum is
/// negative where the key is positive), and whether a shared factor was divided out (its sign is
/// not read). `None` where the sum holds a constant or an infinity, or a division overflows.
fn class_of(ts: &[Ex], cx: &Cx) -> Option<(Ex, bool, bool)> {
    if ts
        .iter()
        .any(|t| t.contains_const() || t.contains_infinity())
    {
        return None;
    }
    let mut parts: Vec<(Rat, Ex)> = ts
        .iter()
        .map(|t| match t {
            Ex::Num(r) => Some((r.clone(), Ex::Num(Rat::ONE))),
            // Each term read as the orientation reads it (`odd_signs_out`, as `orient_base` and
            // `up_to_sign_key` do): an odd function of a negated argument is the negated term.
            // A term whose coefficient is a refused partition has no class here.
            t => {
                let (c, k) = term_split(odd_signs_out(t, cx), cx.view);
                Some((c.rat()?, k))
            }
        })
        .collect::<Option<_>>()?;
    // The factors every term shares, divided out.
    let factors = |k: &Ex| -> Vec<Ex> {
        match k {
            Ex::Mul(v) => v.clone(),
            Ex::Num(_) => Vec::new(),
            k => vec![k.clone()],
        }
    };
    let mut shared: Vec<Ex> = factors(&parts[0].1);
    for (_, k) in &parts[1..] {
        let fs = factors(k);
        shared.retain(|f| fs.contains(f));
    }
    let scaled = !shared.is_empty();
    if scaled {
        for (_, k) in parts.iter_mut() {
            let mut fs = factors(k);
            for f in &shared {
                if let Some(i) = fs.iter().position(|g| g == f) {
                    fs.remove(i);
                }
            }
            *k = match fs.len() {
                0 => Ex::Num(Rat::ONE),
                1 => fs.pop().expect("one factor"),
                _ => Ex::Mul(fs),
            };
        }
    }
    parts.sort_by(|a, b| cmp_ex(&a.1, &b.1, cx.view));
    let lead = parts
        .iter()
        .find(|(c, k)| !matches!(k, Ex::Num(_)) && !c.is_zero())?
        .0
        .clone();
    let inv = lead.checked_inv()?;
    let mut key = Vec::with_capacity(parts.len());
    for (c, k) in parts {
        let q = c.checked_mul(&inv)?;
        key.push(match k {
            Ex::Num(_) => Ex::Num(q),
            k if q.is_one() => k,
            k => Ex::Mul(vec![Ex::Num(q), k]),
        });
    }
    Some((Ex::Add(key), lead.is_negative(), scaled))
}

/// THE CLASS'S POLYNOMIAL MULTIPLES: a sum `T` (its terms `ts`) that holds the sum `S` of a class
/// (its key, `class_of`'s) as an exact polynomial factor, `T = S^m R` with `S` not dividing `R`:
/// `9 - x1^2` is `-(x1 - 3)(x1 + 3)`, `x1^2 - 6 x1 + 9` is `(x1 - 3)^2`. Read where `S` is linear in
/// one of its variables `v` (`c v + r`, `c` a nonzero literal, `r` free of `v`, and no other atom
/// holding `v`): such an `S` is irreducible, so `R` does not vanish on its zero set at a generic
/// point, and `T` is odd in `S` for an odd `m` and even for an even one. Returned: whether `m` is
/// odd, and the sign of `R` on the zero set where it is a constant there (`R` with `v = -r/c`;
/// `true`: negative), else `None` (its sign is not read, as for a factor every term shares). The
/// atoms are the factors that are neither literals nor sums nor positive integer powers of either
/// (a variable, a function, any other power), each an indeterminate, so the factor is proven for
/// every value they take; sums inside products and under positive integer powers are expanded,
/// within `POLY_TERM_CAP` terms. `None` where `T` or `S` holds a `Const`, an infinity or a
/// wildcard, or no factor is proven. A float screen comes first, so that the exact division runs
/// only on a sum that vanishes at one point of the zero set of `S` (`linear_point`,
/// `vanishes_at`); it only refuses.
fn class_multiple(ts: &[Ex], class: &Ex, cx: &Cx) -> Option<(bool, Option<bool>)> {
    use super::poly::Poly;
    let Ex::Add(kts) = class else {
        return None;
    };
    // The screen first (`linear_point`, `vanishes_at`): most sums are no multiple of the class.
    let (lv, lx) = linear_point(kts, cx)?;
    if !vanishes_at(ts, lv, lx) {
        return None;
    }
    let unread =
        |e: &Ex| e.contains_const() || e.contains_infinity() || e.contains_wildcard(cx.view);
    if ts.iter().chain(kts).any(unread) {
        return None;
    }
    fn to_poly(e: &Ex, atoms: &mut Vec<Ex>) -> Option<Poly> {
        const POWER_CAP: i128 = 8;
        let mut atom = |a: &Ex, k: u32| -> Option<Poly> {
            let i = match atoms.iter().position(|x| x == a) {
                Some(i) => i,
                None => {
                    atoms.push(a.clone());
                    atoms.len() - 1
                }
            };
            Some(Poly::atom(u16::try_from(i).ok()?, k))
        };
        match e {
            Ex::Num(r) => Some(Poly::constant(r.clone())),
            Ex::Add(v) => v
                .iter()
                .try_fold(Poly::zero(), |acc, t| acc.add(&to_poly(t, atoms)?)),
            Ex::Mul(v) => v.iter().try_fold(Poly::constant(Rat::ONE), |acc, f| {
                acc.mul(&to_poly(f, atoms)?)
            }),
            Ex::Pow(b, q) => {
                let k = match &**q {
                    Ex::Num(r) if r.is_integer() && r.signum() > 0 => r
                        .small_int()
                        .filter(|k| *k <= POWER_CAP)
                        .and_then(|k| u32::try_from(k).ok()),
                    _ => None,
                };
                match k {
                    Some(k) if matches!(&**b, Ex::Add(_) | Ex::Num(_)) => to_poly(b, atoms)?.pow(k),
                    Some(k) => atom(b, k),
                    None => atom(e, 1),
                }
            }
            Ex::Const(_) | Ex::PosInf | Ex::NegInf | Ex::NaN => None,
            other => atom(other, 1),
        }
    }
    let mut atoms: Vec<Ex> = Vec::new();
    let s = to_poly(class, &mut atoms)?;
    // Every atom of `S` is an atom of each of its multiples.
    if !atoms.iter().all(|a| ts.iter().any(|t| poly_occurs(a, t))) {
        return None;
    }
    let n_s = atoms.len();
    let v = (0..n_s).find(|&i| {
        matches!(atoms[i], Ex::Leaf(_))
            && atoms[i].has_measure_space(cx.view)
            && s.linear_in(i as u16).is_some()
            && atoms
                .iter()
                .enumerate()
                .all(|(j, a)| j == i || !poly_occurs(&atoms[i], a))
    })? as u16;
    let terms: Vec<Ex> = ts.iter().map(|t| odd_signs_out(t, cx)).collect();
    let mut rest = to_poly(&Ex::Add(terms), &mut atoms)?;
    if atoms
        .iter()
        .enumerate()
        .any(|(j, a)| j != v as usize && poly_occurs(&atoms[v as usize], a))
    {
        return None;
    }
    let mut m = 0u32;
    while let Some(q) = rest.div_exact(&s) {
        m += 1;
        rest = q;
        if m > 32 {
            return None;
        }
    }
    if m == 0 {
        return None;
    }
    let (c, r) = s.linear_in(v)?;
    let at = Poly::constant(c.checked_inv()?.checked_neg()?).mul(&r)?;
    let sign = rest
        .substitute(v, &at)?
        .as_constant()
        .filter(|k| !k.is_zero())
        .map(|k| k.is_negative());
    Some((m % 2 == 1, sign))
}

/// `poly_screen` of the sum of `ts`.
fn poly_screen_sum(ts: &[Ex], v: &Ex, x: f64) -> Option<(f64, f64)> {
    ts.iter().try_fold((0.0, 0.0), |(a, b), t| {
        let (y, m) = poly_screen(t, v, x)?;
        Some((a + y, b + m))
    })
}

/// A variable `v` a sum is linear in, `c v` one of its terms and `v` in no other, and the value of
/// `v` that puts the sum to zero where every other atom takes its value (`poly_atom_value`): a
/// point of the sum's zero set for `class_multiple`'s screen.
fn linear_point<'a>(ts: &'a [Ex], cx: &Cx) -> Option<(&'a Ex, f64)> {
    ts.iter().enumerate().find_map(|(i, t)| {
        let (c, v) = match t {
            Ex::Leaf(_) => (1.0, t),
            Ex::Mul(fs) => match fs.as_slice() {
                [Ex::Num(c), v @ Ex::Leaf(_)] => (c.to_f64(), v),
                _ => return None,
            },
            _ => return None,
        };
        if !v.has_measure_space(cx.view)
            || ts
                .iter()
                .enumerate()
                .any(|(j, u)| j != i && poly_occurs(v, u))
        {
            return None;
        }
        let (r, _) = poly_screen_sum(&ts[..i], v, 0.0)?;
        let (q, _) = poly_screen_sum(&ts[i + 1..], v, 0.0)?;
        Some((v, -(r + q) / c))
    })
}

/// Does the sum of `ts` vanish, up to rounding, at the point `linear_point` gives (`v = x`, every
/// other atom at its value)? A multiple of a sum vanishes where it does; the screen only refuses.
fn vanishes_at(ts: &[Ex], v: &Ex, x: f64) -> bool {
    poly_screen_sum(ts, v, x).is_some_and(|(y, scale)| scale.is_finite() && y.abs() <= 1e-9 * scale)
}

/// Does `a` occur in `e` (`e` itself included)?
fn poly_occurs(a: &Ex, e: &Ex) -> bool {
    e == a
        || match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|x| poly_occurs(a, x)),
            Ex::Pow(b, q) => poly_occurs(a, b) || poly_occurs(a, q),
            _ => false,
        }
}

/// A value for an atom (`class_multiple`), the same wherever it stands: the screen reads the
/// polynomials at one point of the atoms' space, each atom an independent coordinate.
fn poly_atom_value(a: &Ex) -> f64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    a.hash(&mut h);
    0.55 + (h.finish() >> 11) as f64 / (1u64 << 53) as f64 * 0.9
}

/// THE SCREEN of `class_multiple`: `e` at the atoms' point (`poly_atom_value`) with `x` for the
/// atom `v`, and the same with every term made positive (the scale its rounding error is measured
/// against); `None` where `v` stands inside another atom, where the formal identity would not be
/// the function's.
fn poly_screen(e: &Ex, v: &Ex, x: f64) -> Option<(f64, f64)> {
    let atom = |a: &Ex| -> Option<(f64, f64)> {
        if a == v {
            Some((x, x.abs()))
        } else if poly_occurs(v, a) {
            None
        } else {
            let y = poly_atom_value(a);
            Some((y, y))
        }
    };
    match e {
        Ex::Num(r) => {
            let y = r.to_f64();
            Some((y, y.abs()))
        }
        Ex::Add(ts) => poly_screen_sum(ts, v, x),
        Ex::Mul(fs) => fs.iter().try_fold((1.0, 1.0), |(a, b), f| {
            let (y, m) = poly_screen(f, v, x)?;
            Some((a * y, b * m))
        }),
        Ex::Pow(b, q) => match &**q {
            Ex::Num(r) if r.is_integer() && r.signum() > 0 && r.cmp_int(8) != Ordering::Greater => {
                let k = r.small_int()? as i32;
                let (y, m) = if matches!(&**b, Ex::Add(_) | Ex::Num(_)) {
                    poly_screen(b, v, x)?
                } else {
                    atom(b)?
                };
                Some((y.powi(k), m.powi(k)))
            }
            _ => atom(e),
        },
        Ex::Const(_) | Ex::PosInf | Ex::NegInf | Ex::NaN => None,
        other => atom(other),
    }
}

/// A hash of a sum's terms as `class_of` reads them: the factors every term shares left out, and
/// literals and signs ignored (`shape_sig`), so that two sums of one class always hash alike.
fn class_shape(ts: &[Ex]) -> u64 {
    let factors = |t: &Ex| -> Vec<u64> {
        match t {
            Ex::Num(_) => Vec::new(),
            Ex::Mul(v) => v
                .iter()
                .filter(|f| !matches!(f, Ex::Num(_)))
                .map(shape_sig)
                .collect(),
            t => vec![shape_sig(t)],
        }
    };
    let per: Vec<Vec<u64>> = ts.iter().map(factors).collect();
    let mut shared: Vec<u64> = per.first().cloned().unwrap_or_default();
    for fs in &per[1..] {
        shared.retain(|f| fs.contains(f));
    }
    let mut sig = 0u64;
    for mut fs in per {
        for f in &shared {
            if let Some(i) = fs.iter().position(|g| g == f) {
                fs.remove(i);
            }
        }
        let term = fs.iter().fold(0x9e37_79b9u64, |a, f| a.wrapping_add(*f));
        sig = sig.wrapping_add(term.wrapping_mul(0x0100_0000_01b3));
    }
    sig
}

/// A hash of a sum's shape blind to its literals and signs (`class_of` reads alike over a sum and
/// its constant multiples): two sums of one class always hash alike.
fn shape_sig(e: &Ex) -> u64 {
    use std::hash::{Hash, Hasher};
    fn one<T: Hash>(t: &T) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }
    match e {
        Ex::Num(_) => 0,
        Ex::Add(v) | Ex::Mul(v) => v.iter().map(shape_sig).fold(0, u64::wrapping_add),
        Ex::Fun(op, args) => args.iter().map(shape_sig).fold(one(op), u64::wrapping_add),
        Ex::Pow(b, q) => shape_sig(b).wrapping_add(shape_sig(q)).wrapping_mul(31),
        Ex::Leaf(t) => one(t),
        other => one(&std::mem::discriminant(other)),
    }
}

/// How a factor reads in a class `S` (a sum in its one orientation): free of it, even (alike over
/// `S` and `-S`), odd (negated with it), defined beside the zero of `S` on one side only, or
/// neither. Compositional: `S` and `-S` are odd; a product's parity is the product of its factors'
/// (a factor free of `S` is even); an odd function or an odd root keeps its argument's; an even
/// function of anything in `S`, an even integer power of it, or any function of something even in
/// it, is even; a sum that holds `S` as an exact polynomial factor has the parity of the factor's
/// multiplicity (`class_multiple`), any other sum the parity its terms share. A log, an even root
/// or a fractional or symbolic power of something odd in `S` whose sign beside the zero is read is
/// defined only on the side where that is positive (`Side(true)`: where `S` is negative), and so
/// is a product or an integer power holding one; members defined on opposite sides leave no domain
/// beside the zero, and are not read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Parity {
    Free,
    Even,
    Odd,
    Side(bool),
    Unknown,
}

/// A factor read at the zero of a class `S` (READING A): its parity in `S`; whether its sign at
/// the zero differs from its sign beside the zero on the side where the oriented sum is positive
/// (for a factor defined on one side only, on that side); and that sign beside the zero
/// (negative?). The sum spelled in its other orientation is `+0` at its zero but negative beside
/// it; a product counts its members (a literal or a factor free of `S` signs both alike); an odd
/// power, an odd function and an odd root keep the sign of zero; an even power or an even function
/// differs nowhere, and neither does a one-sided head (`log(±0)` is `-inf`, a fractional power of
/// `±0` is `+0` or `+inf`, each with the sign it has beside the zero). `None` where that is not
/// read (a sum that holds `S` times a factor its terms share, `3 x2 - x1 x2`, or as a polynomial
/// factor beside a cofactor of no constant sign at the zero, any other head).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ZeroRead {
    parity: Parity,
    differs: Option<bool>,
    below: Option<bool>,
}

/// `zero_read`'s verdicts within a simplify call (`RegionScope`), per token view, class and factor.
type ReadMemo =
    rustc_hash::FxHashMap<usize, rustc_hash::FxHashMap<Ex, rustc_hash::FxHashMap<Ex, ZeroRead>>>;

thread_local! {
    static READ_MEMO: std::cell::RefCell<Option<ReadMemo>> = const { std::cell::RefCell::new(None) };
}

/// The one reading of a factor in a class: membership by parity and the sign at the zero, read
/// together and decided once per call for each factor and class however often products meet them.
fn zero_read(e: &Ex, class: &Ex, cx: &Cx) -> ZeroRead {
    let Ex::Add(cts) = class else {
        return ZeroRead {
            parity: Parity::Unknown,
            differs: None,
            below: None,
        };
    };
    let view = cx.view as *const TokenView as usize;
    let hit = READ_MEMO.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|t| t.get(&view)?.get(class)?.get(e).copied())
    });
    if let Some(hit) = hit {
        return hit;
    }
    let r = read_walk(e, class, cts.len(), class_shape(cts), cx);
    READ_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.entry(view)
                .or_default()
                .entry(class.clone())
                .or_default()
                .insert(e.clone(), r);
        }
    });
    r
}

fn parity_in(e: &Ex, class: &Ex, cx: &Cx) -> Parity {
    zero_read(e, class, cx).parity
}

/// The read of a product from its members' reads: its parity is theirs multiplied, and on the side
/// a one-sided member is defined on (the side where the class's key is positive otherwise) the
/// members whose sign at the zero differs from their sign beside it are counted, and so are those
/// negative beside it. An odd member read on the side where the key is negative changes both.
/// Members defined on opposite sides leave no domain beside the zero: not read.
fn product_read(reads: impl Iterator<Item = ZeroRead> + Clone) -> ZeroRead {
    use Parity::{Even, Free, Odd, Side, Unknown};
    let unread = ZeroRead {
        parity: Unknown,
        differs: None,
        below: None,
    };
    let mut side: Option<bool> = None;
    for r in reads.clone() {
        match r.parity {
            Unknown => return unread,
            Side(s) if side.is_some_and(|t| t != s) => return unread,
            Side(s) => side = Some(s),
            _ => {}
        }
    }
    let other = side == Some(true);
    let (mut odd, mut held) = (false, false);
    let (mut differs, mut below) = (Some(false), Some(false));
    for r in reads {
        let turn = match r.parity {
            Odd => {
                odd = !odd;
                held = true;
                other
            }
            Even | Side(_) => {
                held = true;
                false
            }
            _ => false,
        };
        differs = differs.zip(r.differs).map(|(a, b)| a ^ b ^ turn);
        below = below.zip(r.below).map(|(a, b)| a ^ b ^ turn);
    }
    let parity = match (side, odd, held) {
        (Some(s), _, _) => Side(s),
        (None, true, _) => Odd,
        (None, false, true) => Even,
        _ => Free,
    };
    ZeroRead {
        parity,
        differs,
        below,
    }
}

fn read_walk(e: &Ex, class: &Ex, n: usize, nk: u64, cx: &Cx) -> ZeroRead {
    use Parity::{Even, Free, Odd, Side, Unknown};
    let read = |parity, differs, below| ZeroRead {
        parity,
        differs,
        below,
    };
    let unread = read(Unknown, None, None);
    let free = |below| read(Free, Some(false), below);
    let even = read(Even, Some(false), Some(false));
    // A head defined where its argument is positive, over an argument odd in `S` whose sign beside
    // the zero is read: defined on the side where that sign is positive, where the head is
    // positive (a root or a power) or of the sign `below` gives (a log: negative where its argument
    // tends to zero, read for no argument here).
    let one_sided = |r: ZeroRead, below: Option<bool>| match (r.parity, r.below) {
        (Odd, Some(b)) => read(Side(b), Some(false), below),
        (Side(s), Some(false)) => read(Side(s), Some(false), below),
        _ => unread,
    };
    let is_even_fun = |op: Tok| {
        cx.view.tok_is(op, "abs") || cx.view.tok_is(op, "cos") || cx.view.tok_is(op, "cosh")
    };
    match e {
        Ex::Num(r) => free(Some(r.is_negative())),
        Ex::Add(ts) => {
            if ts.len() == n && class_shape(ts) == nk {
                if let Some((k, negative, scaled)) = class_of(ts, cx) {
                    if k == *class {
                        let sign = (!scaled).then_some(negative);
                        return read(Odd, sign, sign);
                    }
                }
            }
            // A polynomial multiple of the class (`class_multiple`): odd or even in it, with the
            // sign of its cofactor at the zero, and `+0` there as an exactly cancelling sum.
            if let Some((odd, sign)) = class_multiple(ts, class, cx) {
                return read(if odd { Odd } else { Even }, sign, sign);
            }
            let reads: Vec<ZeroRead> = ts.iter().map(|t| read_walk(t, class, n, nk, cx)).collect();
            // A sum holding a term defined on one side only reads as its one-sided terms where those
            // are infinite at the zero (of one side and one sign) and the rest finite there: they
            // dominate it at the zero and beside it (`(3 - x1)^(-1/2) + x2*(x1 - 3)`).
            if reads.iter().any(|r| matches!(r.parity, Side(_))) {
                let (mut side, mut below) = (None, None);
                for (t, r) in ts.iter().zip(&reads) {
                    match (r.parity, zero_mag(t, class, cx)) {
                        (Side(s), Mag::Inf) if side.is_none_or(|x| x == s) => {
                            let Some(b) = r.below.filter(|b| below.is_none_or(|x| x == *b)) else {
                                return unread;
                            };
                            (side, below) = (Some(s), Some(b));
                        }
                        (Free | Even | Odd, Mag::Zero | Mag::Finite) => {}
                        _ => return unread,
                    }
                }
                return side.map_or(unread, |s| read(Side(s), Some(false), below));
            }
            let (mut odd, mut even_t, mut free_t) = (false, false, false);
            for r in &reads {
                match r.parity {
                    Unknown | Side(_) => return unread,
                    Odd => odd = true,
                    Even => even_t = true,
                    Free => free_t = true,
                }
            }
            match (odd, even_t || free_t) {
                (true, true) => unread,
                (true, false) => read(Odd, None, None),
                _ if even_t => read(Even, None, None),
                _ => free(None),
            }
        }
        Ex::Mul(fs) => {
            let r = product_read(fs.iter().map(|f| read_walk(f, class, n, nk, cx)));
            if r.parity == Free {
                free(r.below)
            } else {
                r
            }
        }
        Ex::Pow(b, q) => {
            if read_walk(q, class, n, nk, cx).parity != Free {
                return unread;
            }
            let rb = read_walk(b, class, n, nk, cx);
            match (rb.parity, &**q) {
                (Free, Ex::Num(r)) if r.is_odd_integer() => free(rb.below),
                (Free, Ex::Num(r)) if r.is_even_integer() => free(Some(false)),
                (Free, _) => free(None),
                (Unknown, _) => unread,
                (Side(s), Ex::Num(r)) if r.is_even_integer() => {
                    read(Side(s), Some(false), Some(false))
                }
                (_, Ex::Num(r)) if r.is_even_integer() => even,
                (p, Ex::Num(r)) if r.is_odd_integer() => read(p, rb.differs, rb.below),
                (Even, _) => read(Even, None, None),
                // A fractional or symbolic power: defined where its base is positive.
                _ => one_sided(rb, Some(false)),
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && odd_fun(cx.view, *op) => {
            let ra = read_walk(&args[0], class, n, nk, cx);
            match ra.parity {
                Free => free(None),
                Side(_) => unread,
                _ => ra,
            }
        }
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            if read_walk(&args[1], class, n, nk, cx).parity != Free {
                return unread;
            }
            let ra = read_walk(&args[0], class, n, nk, cx);
            let odd_index = matches!(&args[1], Ex::Num(k) if k.is_odd_integer());
            let even_index = matches!(&args[1], Ex::Num(k) if k.is_even_integer());
            match ra.parity {
                Free if odd_index => free(ra.below),
                Free => free(Some(false)),
                Unknown => unread,
                _ if odd_index => ra,
                Even => even,
                _ if even_index => one_sided(ra, Some(false)),
                _ => unread,
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "log") => {
            let ra = read_walk(&args[0], class, n, nk, cx);
            match ra.parity {
                Free => free(None),
                Even => read(Even, None, None),
                // negative beside the zero where its argument vanishes there, positive where it is
                // infinite
                _ => one_sided(
                    ra,
                    match zero_mag(&args[0], class, cx) {
                        Mag::Zero => Some(true),
                        Mag::Inf => Some(false),
                        _ => None,
                    },
                ),
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && is_even_fun(*op) => {
            match read_walk(&args[0], class, n, nk, cx).parity {
                Free => free(None),
                Unknown | Side(_) => unread,
                _ => even,
            }
        }
        Ex::Fun(_, args) => {
            let mut acc = Free;
            for a in args {
                match read_walk(a, class, n, nk, cx).parity {
                    Free => {}
                    Even => acc = Even,
                    _ => return unread,
                }
            }
            if acc == Free {
                free(None)
            } else {
                read(Even, None, None)
            }
        }
        _ => free(None),
    }
}

/// READING A AT THE ZERO OF A CLASS `S`: the value a product keeps at the zero of `S` -- the side
/// whose limit it is (`true`: where the class's key is negative) and whether the factors as written
/// leave the value off that limit -- or `None` where it keeps none. It keeps one where that value
/// is a limit: where the product is even in `S` (two or more odd members, an even count, so its two
/// sides agree), or defined beside the zero on one side only (`Parity::Side`: a log, an even root,
/// a fractional or symbolic power of `S` among its members, which fixes the side), and the product
/// is not NaN there in every orientation (an infinity times a zero, `log(S) S`). An odd product
/// with no one-sided member has a two-signed pole or zero, whose sign is no value the answer keeps.
/// The float value at the zero is the limit exactly where the members whose sign there differs
/// from their sign beside the zero on that side are even in number (`ZeroRead::differs`): written
/// in the orientation positive on that side, every member's base is `+0` there. Where a member is
/// not read (an odd root of a product of unknown sign, `cbrt(x2 (x1 - 3))`, or a class scaled by a
/// factor of unknown sign), the product keeps nothing and its orientation is its cheapest. A lossy
/// context keeps no value at a pole. The product owners all ask this one predicate: the sign trade
/// (`sign_place`), the product table's lone inverse (`Class::table`) and the sum's like terms
/// (`orient_term_classes`).
fn zero_limit(factors: &[Ex], class: &Ex, cx: &Cx) -> Option<(bool, bool)> {
    if cx.permits(Permissive::Agreement) {
        return None;
    }
    let reads: Vec<ZeroRead> = factors.iter().map(|f| zero_read(f, class, cx)).collect();
    let r = product_read(reads.iter().copied());
    let side = match r.parity {
        Parity::Side(s) => s,
        Parity::Even if reads.iter().filter(|r| r.parity == Parity::Odd).count() >= 2 => false,
        _ => return None,
    };
    let off = r.differs?;
    let mags = factors.iter().map(|f| zero_mag(f, class, cx));
    (product_mag(mags) != Mag::Nan).then_some((side, off))
}

/// Whether a product's value at the zero of a class is off the limit it keeps there
/// (`zero_limit`); `None` where it keeps none.
fn zero_off(factors: &[Ex], class: &Ex, cx: &Cx) -> Option<bool> {
    zero_limit(factors, class, cx).map(|(_, off)| off)
}

/// A factor's magnitude at the zero of a class `S` (`zero_limit`'s NaN test): zero, finite and
/// nonzero, infinite, NaN, or not read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mag {
    Zero,
    Finite,
    Inf,
    Nan,
    Unknown,
}

/// The magnitude of a product from its members': NaN where an infinity meets a zero.
fn product_mag(mags: impl Iterator<Item = Mag>) -> Mag {
    let (mut zero, mut inf) = (false, false);
    for m in mags {
        match m {
            Mag::Nan => return Mag::Nan,
            Mag::Unknown => return Mag::Unknown,
            Mag::Zero => zero = true,
            Mag::Inf => inf = true,
            Mag::Finite => {}
        }
    }
    match (zero, inf) {
        (true, true) => Mag::Nan,
        (true, false) => Mag::Zero,
        (false, true) => Mag::Inf,
        _ => Mag::Finite,
    }
}

/// `e`'s magnitude at the zero of a class `S`: a factor free of `S` is finite there (at a generic
/// point), a sum of the class and a sum of terms that vanish there are zero, and the heads carry it:
/// a negative power or a negative root index inverts it, a log of a zero is infinite, an odd
/// function keeps a zero, `exp`, `cos` and `cosh` of a finite argument are finite.
fn zero_mag(e: &Ex, class: &Ex, cx: &Cx) -> Mag {
    let inverse = |m: Mag| match m {
        Mag::Zero => Mag::Inf,
        Mag::Inf => Mag::Zero,
        m => m,
    };
    if parity_in(e, class, cx) == Parity::Free {
        return match e {
            Ex::PosInf | Ex::NegInf => Mag::Inf,
            _ => Mag::Finite,
        };
    }
    match e {
        Ex::Add(ts) if matches!(class_of(ts, cx), Some((k, _, _)) if k == *class) => Mag::Zero,
        Ex::Add(ts) if class_multiple(ts, class, cx).is_some() => Mag::Zero,
        Ex::Add(ts) => {
            let mut all_zero = true;
            for t in ts {
                match zero_mag(t, class, cx) {
                    Mag::Zero => {}
                    Mag::Finite => all_zero = false,
                    _ => return Mag::Unknown,
                }
            }
            if all_zero {
                Mag::Zero
            } else {
                Mag::Finite
            }
        }
        Ex::Mul(fs) => product_mag(fs.iter().map(|f| zero_mag(f, class, cx))),
        Ex::Pow(b, q) => match &**q {
            Ex::Num(r) if r.is_negative() => inverse(zero_mag(b, class, cx)),
            Ex::Num(_) => zero_mag(b, class, cx),
            _ => match zero_mag(b, class, cx) {
                Mag::Finite => Mag::Finite,
                _ => Mag::Unknown,
            },
        },
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => match &args[1] {
            Ex::Num(k) if k.is_negative() => inverse(zero_mag(&args[0], class, cx)),
            Ex::Num(_) => zero_mag(&args[0], class, cx),
            _ => Mag::Unknown,
        },
        Ex::Fun(op, args) if args.len() == 1 => {
            let m = zero_mag(&args[0], class, cx);
            let view = cx.view;
            if view.tok_is(*op, "log") {
                match m {
                    Mag::Zero | Mag::Inf => Mag::Inf,
                    m => m,
                }
            } else if odd_fun(view, *op) || view.tok_is(*op, "abs") {
                match m {
                    Mag::Zero | Mag::Finite => m,
                    _ => Mag::Unknown,
                }
            } else if ["exp", "cos", "cosh"].iter().any(|n| view.tok_is(*op, n)) {
                match m {
                    Mag::Zero | Mag::Finite => Mag::Finite,
                    _ => Mag::Unknown,
                }
            } else {
                Mag::Unknown
            }
        }
        _ => Mag::Unknown,
    }
}

/// The sites of one class in a product (`sign_place`): the class, and per site its index among the
/// sites and whether it spells the class against the class's own orientation.
type ClassGroup = (Option<Ex>, Vec<(usize, bool)>);

/// A class {B, -B}: the head of `B`, the terms of the sum on each side (the second empty until a
/// factor over it is seen), and the key and flip that recognise the negation of the first.
struct Class<'a> {
    head: Option<(Tok, Option<&'a Ex>)>,
    sums: [Vec<Ex>; 2],
    key: u64,
    flip: SumFlip,
    /// The class's key where the product keeps a value at its zero, the limit (`zero_limit`).
    kept: Option<Ex>,
}

impl<'a> Class<'a> {
    fn new(read: &ClassRead<'a>) -> Self {
        Class {
            head: read.head,
            sums: [read.terms.to_vec(), Vec::new()],
            key: negation_key(read.terms),
            flip: SumFlip::new(),
            kept: None,
        }
    }

    /// Is the class's argument a literal multiple (a sum of one term under a head)?
    fn multiple(&self) -> bool {
        self.head.is_some() && self.sums[0].len() == 1 && !matches!(self.sums[0][0], Ex::Add(_))
    }

    /// The second sum of a class whose factors all arrived over the first: its flip, where the
    /// sign may pass through it (`sign_trade_flip`; for a literal multiple, its head's).
    fn complete(&mut self, cx: &Cx) -> bool {
        let first = &self.sums[0];
        let multiple = self.multiple();
        let base = self.base(0, cx.view);
        let flip = self.flip.get_or_init(|| {
            if multiple {
                // The head's own flip where it is a sign-trade site (a literal folds through
                // `fun()`), else the argument negated.
                let negated = match sign_trade_flip(&base, cx) {
                    Some(Ex::Fun(f, a)) => {
                        a[odd_head(cx.view, f, &a).map_or(0, |(p, _)| p)].clone()
                    }
                    _ => negate_term(&first[0], cx)?,
                };
                return multiple_literal(&negated).map(|_| vec![negated]);
            }
            match sign_trade_flip(&Ex::Add(first.clone()), cx) {
                Some(Ex::Add(f)) => Some(f),
                _ => None,
            }
        });
        match flip.clone() {
            Some(f) => {
                self.sums[1] = f;
                true
            }
            None => false,
        }
    }

    fn same_head(&self, read: &ClassRead) -> bool {
        match (&self.head, &read.head) {
            (None, None) => true,
            (Some((a, i)), Some((b, j))) => a == b && i == j,
            _ => false,
        }
    }

    /// The side a factor reads over, if it belongs to the class.
    fn side_of(&self, read: &ClassRead) -> Option<usize> {
        if !self.same_head(read) {
            return None;
        }
        (0..2).find(|&s| !self.sums[s].is_empty() && self.sums[s].as_slice() == read.terms)
    }

    /// The side a factor belongs to, recognising the second sum on first sight.
    fn place(&mut self, read: &ClassRead<'a>, cx: &Cx) -> Option<usize> {
        if let Some(s) = self.side_of(read) {
            return Some(s);
        }
        let opposite = self.same_head(read)
            && self.sums[1].is_empty()
            && negation_key(read.terms) == self.key
            && negates(&self.sums[0], &self.flip, read.terms, cx);
        if opposite {
            self.sums[1] = read.terms.to_vec();
            return Some(1);
        }
        None
    }

    /// The base over one side's sum (a literal multiple as itself).
    fn base(&self, side: usize, view: &TokenView) -> Ex {
        let sum = match self.sums[side].as_slice() {
            [m] if self.head.is_some() => m.clone(),
            ts => Ex::Add(ts.to_vec()),
        };
        match &self.head {
            None => sum,
            Some((op, None)) => Ex::Fun(*op, vec![sum]),
            // `besselj`'s fixed operand (the order) stands first, `rootn`'s (the index) second
            Some((op, Some(k))) if view.tok_is(*op, "besselj") => {
                Ex::Fun(*op, vec![(*k).clone(), sum])
            }
            Some((op, Some(k))) => Ex::Fun(*op, vec![sum, (*k).clone()]),
        }
    }

    /// The table of the class's members among `factors`.
    fn table<'b>(&self, factors: impl Iterator<Item = (&'b Ex, ClassRead<'b>)>, cx: &Cx) -> Table {
        let mut t = Table::default();
        for (f, read) in factors {
            let Some(s) = self.side_of(&read) else {
                continue;
            };
            let side = &mut t.sides[s];
            // An integer exponent is mass; a fractional one gives its integer part, taken toward
            // zero, and keeps the rest (`S^(5/2) = S^2 * S^(1/2)`: a merge of like signs, which a
            // context without the certificate makes, must not hide mass from the certified one);
            // a symbolic one, or one too large to count (or whose parity the f64 reading does not
            // know), stays as it is.
            let Some((whole, frac)) = read.exp.as_ref().and_then(whole_part) else {
                side.fixed.push(f.clone());
                continue;
            };
            let mass = if whole > 0 { &mut side.p } else { &mut side.n };
            match mass.checked_add(whole) {
                Some(m) if m.checked_abs().is_some_and(|a| a <= MASS_LIMIT) => *mass = m,
                _ => {
                    side.fixed.push(f.clone());
                    continue;
                }
            }
            if !frac.is_zero() {
                side.fixed.push(if whole == 0 {
                    f.clone()
                } else {
                    pow(self.base(s, cx.view), Ex::Num(frac), cx)
                });
            }
        }
        // A lone inverse moves, its sign going to the product: the table writes the class in one
        // orientation, so at the zero of its sum (where every member's base is +0) the product's
        // value is the coefficient's infinity, the limit wherever the pole is one-signed
        // (`1/((x1 - 3)(3 - x1))` is `-1/(x1 - 3)^2`); where the product keeps a value at the
        // zero (`zero_limit`), only to a placement that keeps it. A class whose zero set may hold
        // a region keeps its inverse, infinite on the region.
        t.free_inverse = zero_set_null(&self.base(0, cx.view), cx);
        for side in t.sides.iter_mut() {
            side.fixed.sort_by(|a, b| cmp_ex(a, b, cx.view));
        }
        t
    }

    /// The class's factors for a placement of its table: on each side the positive mass, the
    /// movable negative mass, the kept `B^-1` and the factors that never move. A fractional power
    /// may take the mass of its own sign (`B^2 * B^(1/2) * B^(1/2)` is `B^(5/2) * B^(1/2)` or
    /// `B^(3/2) * B^(3/2)`, merges of like signs), so every split of a side's mass between its
    /// integer power and its fractional powers is one spelling while a placement has at most
    /// `SPLIT_LIMIT` of them, else each mass goes whole to its integer power or spread evenly.
    fn factors(&self, t: &Table, pl: &Placement, cx: &Cx) -> Vec<Vec<Ex>> {
        // Per side, its fractional powers by sign (each with its position in `fixed` and its
        // exponent), and per side and sign the splits of the mass: the integer power's share,
        // then each fractional power's.
        let slots: Vec<[Vec<(usize, Rat)>; 2]> = t
            .sides
            .iter()
            .map(|side| {
                let mut by_sign: [Vec<(usize, Rat)>; 2] = [Vec::new(), Vec::new()];
                for (i, f) in side.fixed.iter().enumerate() {
                    if let Some(e) = class_read(f, cx)
                        .and_then(|r| r.exp)
                        .filter(|e| !e.is_integer())
                    {
                        by_sign[usize::from(e.is_negative())].push((i, e));
                    }
                }
                by_sign
            })
            .collect();
        let splits_with = |every: bool| -> Vec<[Vec<Vec<i128>>; 2]> {
            (0..2)
                .map(|s| {
                    [
                        mass_splits(pl.p[s], slots[s][0].len(), every),
                        mass_splits(pl.q[s], slots[s][1].len(), every),
                    ]
                })
                .collect()
        };
        let mut splits = splits_with(true);
        let count = splits
            .iter()
            .flatten()
            .try_fold(1usize, |n, v| n.checked_mul(v.len()));
        if count.is_none_or(|n| n > SPLIT_LIMIT) {
            splits = splits_with(false);
        }
        let mut lists: Vec<Vec<Ex>> = vec![Vec::new()];
        for s in 0..2 {
            let side = &t.sides[s];
            let b = self.base(s, cx.view);
            let int_pow = |k: i128| -> Ex {
                if k == 1 {
                    b.clone()
                } else {
                    Ex::Pow(Box::new(b.clone()), Box::new(Ex::int(k)))
                }
            };
            let mut side_lists: Vec<Vec<Ex>> = Vec::new();
            for pos in &splits[s][0] {
                for neg in &splits[s][1] {
                    let mut v = Vec::new();
                    let mut taken = vec![0i128; side.fixed.len()];
                    for (sign, split) in [(0usize, pos), (1usize, neg)] {
                        for (&(i, _), &k) in slots[s][sign].iter().zip(&split[1..]) {
                            taken[i] = k;
                        }
                        if split[0] != 0 {
                            v.push(int_pow(split[0]));
                        }
                    }
                    let kept = t.kept(side.n);
                    if kept != 0 {
                        v.push(int_pow(kept));
                    }
                    for (i, f) in side.fixed.iter().enumerate() {
                        if taken[i] == 0 {
                            v.push(f.clone());
                            continue;
                        }
                        let e = slots[s]
                            .iter()
                            .flatten()
                            .find(|(j, _)| *j == i)
                            .and_then(|(_, e)| e.checked_add(&Rat::int(taken[i])));
                        match e {
                            Some(e) => v.push(pow(b.clone(), Ex::Num(e), cx)),
                            None => {
                                v.push(int_pow(taken[i]));
                                v.push(f.clone());
                            }
                        }
                    }
                    side_lists.push(v);
                }
            }
            lists = lists
                .into_iter()
                .flat_map(|l| {
                    side_lists.iter().map(move |sl| {
                        let mut l = l.clone();
                        l.extend(sl.iter().cloned());
                        l
                    })
                })
                .collect();
        }
        lists
    }
}

/// The most splits of its mass between integer and fractional powers a placement offers
/// (`Class::factors`).
const SPLIT_LIMIT: usize = 36;

/// The splits of a side's mass of one sign between its integer power and `slots` fractional
/// powers of that sign, the integer power's share first and the whole mass on the integer power
/// the first split: with `every`, every composition (while there are at most `SPLIT_LIMIT`);
/// else that one and the mass spread evenly over the fractional powers.
fn mass_splits(mass: i128, slots: usize, every: bool) -> Vec<Vec<i128>> {
    let mut whole = vec![0i128; slots + 1];
    whole[0] = mass;
    if mass == 0 || slots == 0 {
        return vec![whole];
    }
    let m = mass.unsigned_abs();
    let sign = mass.signum();
    if !every {
        let n = slots as u128;
        let even = (0..n)
            .map(|i| sign * ((m / n + u128::from(i < m % n)) as i128))
            .collect::<Vec<i128>>();
        return vec![whole, [vec![0], even].concat()];
    }
    // Every composition, the integer power's share falling: the whole mass on it comes first.
    fn fill(at: usize, left: u128, sign: i128, cur: &mut Vec<i128>, out: &mut Vec<Vec<i128>>) {
        if at + 1 == cur.len() {
            cur[at] = sign * left as i128;
            out.push(cur.clone());
            return;
        }
        for k in (0..=left).rev() {
            cur[at] = sign * k as i128;
            fill(at + 1, left - k, sign, cur, out);
        }
    }
    // C(m + slots, slots) compositions
    let mut count: u128 = 1;
    for i in 1..=slots as u128 {
        count = count.saturating_mul(m + i) / i;
        if count > SPLIT_LIMIT as u128 {
            return mass_splits(mass, slots, false);
        }
    }
    let mut out = Vec::new();
    fill(0, m, sign, &mut vec![0i128; slots + 1], &mut out);
    out
}

/// An exponent's integer part, taken toward zero, and the rest: `None` for one too large to count
/// or an integer whose parity the f64 reading does not know.
fn whole_part(r: &Rat) -> Option<(i128, Rat)> {
    let (p, q) = r.small_parts()?;
    if q == 1 && r.parity().is_none() {
        return None;
    }
    let whole = p / q;
    let frac = r.checked_add(&Rat::int(-whole))?;
    Some((whole, frac))
}

/// The largest total a table counts (a larger exponent stays as it is): sums and differences of
/// two totals stay far inside `i128`.
const MASS_LIMIT: i128 = 1 << 100;

/// A class's exponent totals on each side (`collect_opposite_sums`): the positive integer mass
/// `p`, the negative `n`, and the factors that never move; and whether a lone inverse moves too
/// (off a region).
#[derive(Default)]
struct Table {
    sides: [TableSide; 2],
    free_inverse: bool,
}

#[derive(Default)]
struct TableSide {
    p: i128,
    n: i128,
    fixed: Vec<Ex>,
}

/// A class's factors under one placement (`collect_opposite_sums`): collected (`None` where a
/// collection splices), whether the placement moves a sign, and the lower bound of their price
/// and their count.
struct PlacedClass {
    list: Option<Vec<Ex>>,
    odd: bool,
    lb: u64,
    members: usize,
}

/// Where a placement puts the movable mass, positive and negative, per side.
struct Placement {
    p: [i128; 2],
    q: [i128; 2],
}

impl Table {
    /// Does a side hold two fractional powers of one sign beside integer mass of that sign, which
    /// they may split between them?
    fn splits_mass(&self, s: usize, cx: &Cx) -> bool {
        let side = &self.sides[s];
        let mut count = [0usize; 2];
        for f in &side.fixed {
            if let Some(e) = class_read(f, cx)
                .and_then(|r| r.exp)
                .filter(|e| !e.is_integer())
            {
                count[usize::from(e.is_negative())] += 1;
            }
        }
        (count[0] >= 2 && side.p > 0) || (count[1] >= 2 && side.n - self.kept(side.n) < 0)
    }

    /// The inverse a side keeps: one `B^-1` where its negative total is odd, none where it may
    /// move (`Table::free_inverse`).
    fn kept(&self, n: i128) -> i128 {
        if self.free_inverse {
            0
        } else {
            n % 2
        }
    }

    /// The placements of the movable mass, the table's own first, each with whether it moves an
    /// odd amount of mass (a sign). Every split while there are few; else the two extremes beside
    /// the table's own.
    fn placements(&self) -> Vec<(Placement, bool)> {
        let [a, b] = &self.sides;
        let total_p = a.p + b.p;
        let q = |n: i128| n - self.kept(n);
        let total_q = q(a.n) + q(b.n);
        let step = if self.free_inverse { 1 } else { 2 };
        let fixed = self.kept(a.n) != 0
            || self.kept(b.n) != 0
            || !a.fixed.is_empty()
            || !b.fixed.is_empty();
        let mut ps: Vec<i128> = if total_p <= 7 {
            (0..=total_p).collect()
        } else {
            vec![0, a.p, total_p]
        };
        let mut qs: Vec<i128> = if -total_q <= 6 {
            (0..=-total_q / step).map(|k| -step * k).collect()
        } else {
            vec![0, q(a.n), total_q]
        };
        ps.sort_unstable();
        ps.dedup();
        qs.sort_unstable();
        qs.dedup();
        let own = (a.p, q(a.n));
        // Without a factor that stays, all the mass on one side mirrors all of it on the other,
        // and with even totals `pow` files the two alike: one of them is priced.
        let ends = [(total_p, total_q), (0, 0)];
        let mirrored = |c: (i128, i128)| {
            !fixed
                && total_p % 2 == 0
                && total_q % 2 == 0
                && ends.contains(&c)
                && (ends.contains(&own) || c == ends[1])
        };
        let mut out = vec![(
            Placement {
                p: [a.p, b.p],
                q: [q(a.n), q(b.n)],
            },
            false,
        )];
        for &x in &ps {
            for &y in &qs {
                if (x, y) == own || mirrored((x, y)) {
                    continue;
                }
                out.push((
                    Placement {
                        p: [x, total_p - x],
                        q: [y, total_q - y],
                    },
                    (x - a.p + y - q(a.n)) % 2 != 0,
                ));
            }
        }
        out
    }

    /// Do the two tables offer the same placements: the same totals, the same `B^-1` kept on
    /// each side, the same factors that never move?
    fn same_orbit(&self, o: &Table) -> bool {
        let [a, b] = &self.sides;
        let [c, d] = &o.sides;
        a.p + b.p == c.p + d.p
            && a.n + b.n == c.n + d.n
            && self.kept(a.n) == o.kept(c.n)
            && self.kept(b.n) == o.kept(d.n)
            && a.fixed == c.fixed
            && b.fixed == d.fixed
    }
}

/// Does the bag hold a class to decide (`collect_opposite_sums`): a factor over a sum, or over an
/// odd function of one, and one over its negation, or two over one of them with exponents of
/// opposite signs? Only sums that share a `negation_key` are compared, and each sum's flip is
/// computed at most once.
fn holds_opposite_sums(factors: &[Ex], cx: &Cx) -> bool {
    let reads: Vec<ClassRead> = factors.iter().filter_map(|f| class_read(f, cx)).collect();
    if reads.len() < 2 {
        return false;
    }
    let sign = |r: &ClassRead| r.exp.as_ref().and_then(whole_part).map(|(w, _)| w.signum());
    let fraction = |r: &ClassRead| r.exp.as_ref().filter(|e| !e.is_integer()).map(Rat::signum);
    let mut keyed: Vec<(u64, usize)> = reads
        .iter()
        .enumerate()
        .map(|(i, r)| (negation_key(r.terms), i))
        .collect();
    keyed.sort_unstable();
    for i in 0..keyed.len() {
        let first = &reads[keyed[i].1];
        let mut class = Class::new(first);
        for j in i + 1..keyed.len() {
            if keyed[j].0 != keyed[i].0 {
                break;
            }
            let other = &reads[keyed[j].1];
            match class.place(other, cx) {
                Some(1) => return true,
                Some(_) if sign(first).zip(sign(other)).is_some_and(|(a, b)| a * b < 0) => {
                    return true
                }
                // two fractional powers of one sign over one sum, which may split its mass
                Some(_) if fraction(first).is_some_and(|a| fraction(other) == Some(a)) => {
                    return true
                }
                _ => {}
            }
        }
    }
    false
}

/// A key a sum shares with its negation: its length and the multiset of its terms' shapes, where a
/// shape ignores every sign a negation can move (a coefficient's, and below it the orientation of
/// any sum) and reads four levels deep. Equal keys are only a necessary condition (`negates`).
fn negation_key(ts: &[Ex]) -> u64 {
    fn mix(x: u64) -> u64 {
        let z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn shape(e: &Ex, depth: u32) -> u64 {
        use std::hash::{Hash, Hasher};
        if depth == 0 {
            return 1;
        }
        let mut h = rustc_hash::FxHasher::default();
        match e {
            Ex::Num(r) => {
                0u8.hash(&mut h);
                if r.is_negative() {
                    r.checked_neg().hash(&mut h);
                } else {
                    Some(r).hash(&mut h);
                }
            }
            Ex::Add(ts) => {
                1u8.hash(&mut h);
                ts.iter()
                    .fold(0u64, |a, t| a.wrapping_add(mix(shape(t, depth - 1))))
                    .hash(&mut h);
            }
            Ex::Mul(fs) => {
                let unit = |f: &&Ex| matches!(f, Ex::Num(r) if r.is_one() || *r == Rat::NEG_ONE);
                if fs.iter().filter(|f| !unit(f)).count() == 1 {
                    if let Some(only) = fs.iter().find(|f| !unit(f)) {
                        return shape(only, depth);
                    }
                }
                2u8.hash(&mut h);
                fs.iter()
                    .filter(|f| !unit(f))
                    .fold(0u64, |a, f| a.wrapping_add(mix(shape(f, depth - 1))))
                    .hash(&mut h);
            }
            Ex::Pow(b, x) => {
                3u8.hash(&mut h);
                shape(b, depth - 1).hash(&mut h);
                x.hash(&mut h);
            }
            Ex::Fun(op, args) => {
                4u8.hash(&mut h);
                op.hash(&mut h);
                for a in args {
                    shape(a, depth - 1).hash(&mut h);
                }
            }
            Ex::PosInf | Ex::NegInf => 5u8.hash(&mut h),
            other => {
                6u8.hash(&mut h);
                other.hash(&mut h);
            }
        }
        h.finish()
    }
    ts.iter().fold(mix(ts.len() as u64), |a, t| {
        a.wrapping_add(mix(shape(t, 4)))
    })
}

/// A sum's flip (`sign_trade_flip`), computed once when a comparison needs it (`negates`).
type SumFlip = std::cell::OnceCell<Option<Vec<Ex>>>;

/// Is the sum `b` the negation of the sum `a`? Term by term (`opposite_sums`) where negation
/// keeps every key. A term holding a sum may take the sign into that sum instead
/// (`-sin(x1 - x2)` is `sin(x2 - x1)`) and so change its key; every other term keeps its key, so
/// those must pair up term by term before the sum's own flip decides (`sign_trade_flip`, computed
/// once into `flip`).
fn negates(a: &[Ex], flip: &SumFlip, b: &[Ex], cx: &Cx) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if opposite_sums(a, b, cx.view) {
        return true;
    }
    if !a.iter().any(holds_sum) {
        return false;
    }
    fn plain(s: &[Ex]) -> Vec<&Ex> {
        s.iter().filter(|t| !holds_sum(t)).collect()
    }
    let (pa, pb) = (plain(a), plain(b));
    pa.len() == pb.len()
        && pa
            .iter()
            .zip(&pb)
            .all(|(x, y)| opposite_terms(x, y, cx.view))
        && flip
            .get_or_init(|| match sign_trade_flip(&Ex::Add(a.to_vec()), cx) {
                Some(Ex::Add(f)) => Some(f),
                _ => None,
            })
            .as_deref()
            == Some(b)
}

/// The base a factor is collected under (`factor_split`'s reading, borrowed): a power's base, an
/// even root's radicand, else the factor itself.
fn collection_base<'a>(f: &'a Ex, cx: &Cx) -> &'a Ex {
    match f {
        Ex::Pow(b, _) => b,
        Ex::Fun(op, args)
            if args.len() == 2
                && matches!(&args[1], Ex::Num(n)
                    if n.is_even_integer() && n.cmp_int(2) != Ordering::Less)
                && cx.view.tok_is(*op, "rootn") =>
        {
            &args[0]
        }
        _ => f,
    }
}

/// Is the sum `b` the negation of the sum `a`, term by term? Both are canonical bags, sorted
/// by their coefficient-stripped keys, so a negation keeps every term in its place.
fn opposite_sums(a: &[Ex], b: &[Ex], view: &TokenView) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| opposite_terms(x, y, view))
}

/// Is the term `y` the term `x` with its coefficient negated?
fn opposite_terms(x: &Ex, y: &Ex, view: &TokenView) -> bool {
    // A term's literal members and the factors that remain (borrowed: no key is built).
    fn split(t: &Ex) -> (&[Ex], &[Ex]) {
        match t {
            Ex::Mul(v) => v.split_at(v.iter().take_while(|f| matches!(f, Ex::Num(_))).count()),
            other => (&[], std::slice::from_ref(other)),
        }
    }
    // The coefficient those members denote, all of them (`Coef`).
    fn coef(lits: &[Ex]) -> Coef {
        Coef::from_members(lits.iter().filter_map(|f| match f {
            Ex::Num(r) => Some(r),
            _ => None,
        }))
    }
    match (x, y) {
        (Ex::Num(p), Ex::Num(q)) => p.checked_neg().is_some_and(|n| n == *q),
        (Ex::Num(_), _) | (_, Ex::Num(_)) => false,
        _ => {
            let ((lx, kx), (ly, ky)) = (split(x), split(y));
            if kx == ky {
                return match (lx, ly) {
                    ([], []) => false,
                    ([Ex::Num(c)], []) | ([], [Ex::Num(c)]) => *c == Rat::NEG_ONE,
                    ([Ex::Num(p)], [Ex::Num(q)]) => p.checked_neg().is_some_and(|n| n == *q),
                    _ => coef(lx).negated() == coef(ly) && !coef(lx).is_zero(),
                };
            }
            // An odd function of a literal carries its sign inside (`sin(-2)`): only
            // `term_split`'s sign-normalized keys can match then.
            let literal_odd =
                |k: &[Ex]| k.iter().any(|f| odd_of_literal_multiple(f, view).is_some());
            if !literal_odd(kx) && !literal_odd(ky) {
                return false;
            }
            let (cx_, kx) = term_split(x.clone(), view);
            let (cy, ky) = term_split(y.clone(), view);
            kx == ky && cx_.negated() == cy
        }
    }
}

/// Does the expression hold a sum anywhere?
fn holds_sum(e: &Ex) -> bool {
    match e {
        Ex::Add(_) => true,
        Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(holds_sum),
        Ex::Pow(b, x) => holds_sum(b) || holds_sum(x),
        _ => false,
    }
}

/// Are the two factors collected under one base (`sign_place`'s collision test)?
fn same_collection_base(f: &Ex, g: &Ex, cx: &Cx) -> bool {
    cmp_ex(collection_base(f, cx), collection_base(g, cx), cx.view) == Ordering::Equal
}

/// Does the sum a site is built over occur again in the product, up to its orientation: as
/// another site's sum or inside any factor? Only then are its classes read (`sign_place`). One
/// pass over the product's sums with a hash of their shapes blind to literals and signs
/// (`shape_sig`'s), so a sum meets its constant multiples too; a hash that merely collides only
/// costs the full reading. A sum meets its polynomial multiples as well (`class_multiple`).
fn classes_meet(out: &[Ex], site_flips: &[(usize, Ex)], cx: &Cx) -> bool {
    shapes_meet(out, site_flips) || multiples_meet(out, site_flips, cx)
}

/// Does another sum of the product, a factor or inside one, hold a site's sum as a polynomial
/// factor (`class_multiple`)?
fn multiples_meet(out: &[Ex], site_flips: &[(usize, Ex)], cx: &Cx) -> bool {
    fn sums<'a>(e: &'a Ex, acc: &mut Vec<&'a [Ex]>) {
        match e {
            Ex::Add(v) => {
                acc.push(v);
                v.iter().for_each(|x| sums(x, acc));
            }
            Ex::Mul(v) | Ex::Fun(_, v) => v.iter().for_each(|x| sums(x, acc)),
            Ex::Pow(b, q) => {
                sums(b, acc);
                sums(q, acc);
            }
            _ => {}
        }
    }
    // Each site's sum with a variable it is linear in and the point of its zero set
    // (`linear_point`): only a sum that vanishes there is read further.
    let points: Vec<(&[Ex], &Ex, f64)> = site_flips
        .iter()
        .filter_map(|(pos, _)| {
            let ts = site_sum_terms(&out[*pos])?;
            let (v, x) = linear_point(ts, cx)?;
            Some((ts, v, x))
        })
        .collect();
    if points.is_empty() {
        return false;
    }
    let mut all: Vec<&[Ex]> = Vec::new();
    out.iter().for_each(|f| sums(f, &mut all));
    if all.len() < 2 {
        return false;
    }
    points.iter().any(|&(ts, v, x)| {
        let mut key: Option<Option<Ex>> = None;
        all.iter().any(|us| {
            if std::ptr::eq(us.as_ptr(), ts.as_ptr()) || !vanishes_at(us, v, x) {
                return false;
            }
            let Some(k) = key.get_or_insert_with(|| class_of(ts, cx).map(|(k, _, _)| k)) else {
                return false;
            };
            !matches!(class_of(us, cx), Some((c, _, _)) if c == *k)
                && class_multiple(us, k, cx).is_some()
        })
    })
}

fn shapes_meet(out: &[Ex], site_flips: &[(usize, Ex)]) -> bool {
    fn one<T: std::hash::Hash>(t: &T) -> u64 {
        use std::hash::Hasher;
        let mut h = rustc_hash::FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }
    fn walk(e: &Ex, sums: &mut Vec<(usize, u64)>) -> u64 {
        match e {
            Ex::Num(_) => 0,
            Ex::Add(v) => {
                let s = v.iter().map(|x| walk(x, sums)).fold(0, u64::wrapping_add);
                sums.push((v.len(), class_shape(v)));
                s
            }
            Ex::Mul(v) => v.iter().map(|x| walk(x, sums)).fold(0, u64::wrapping_add),
            Ex::Fun(op, args) => args
                .iter()
                .map(|x| walk(x, sums))
                .fold(one(op), u64::wrapping_add),
            Ex::Pow(b, q) => walk(b, sums).wrapping_add(walk(q, sums)).wrapping_mul(31),
            Ex::Leaf(t) => one(t),
            other => one(&std::mem::discriminant(other)),
        }
    }
    let class_sites = site_flips
        .iter()
        .filter(|(pos, _)| site_sum_terms(&out[*pos]).is_some())
        .count();
    if class_sites == 0 {
        return false;
    }
    // Most products hold no sum besides the ones their sites are built over, each once: then
    // only two sites over sums of one length can meet.
    fn count(e: &Ex) -> usize {
        match e {
            Ex::Add(v) => 1 + v.iter().map(count).sum::<usize>(),
            Ex::Mul(v) | Ex::Fun(_, v) => v.iter().map(count).sum(),
            Ex::Pow(b, q) => count(b) + count(q),
            _ => 0,
        }
    }
    if out.iter().map(count).sum::<usize>() == class_sites {
        let lens: Vec<usize> = site_flips
            .iter()
            .filter_map(|(pos, _)| site_sum_terms(&out[*pos]).map(<[Ex]>::len))
            .collect();
        if (0..lens.len()).all(|i| lens[i + 1..].iter().all(|l| *l != lens[i])) {
            return false;
        }
    }
    let mut sums: Vec<(usize, u64)> = Vec::new();
    for f in out {
        walk(f, &mut sums);
    }
    if sums.len() <= 1 {
        return false;
    }
    site_flips.iter().any(|(pos, _)| {
        site_sum_terms(&out[*pos]).is_some_and(|ts| {
            let sig = class_shape(ts);
            sums.iter().filter(|&&x| x == (ts.len(), sig)).count() >= 2
        })
    })
}

/// The terms of the sum a site is built over, where it is one (the sum itself, an odd function or
/// odd root of it, a power of either).
fn site_sum_terms(f: &Ex) -> Option<&[Ex]> {
    let m = match f {
        Ex::Pow(b, _) => &**b,
        f => f,
    };
    match m {
        Ex::Add(ts) => Some(ts),
        // the sum under an odd head: its one sum argument (a site's index or order is a literal)
        Ex::Fun(_, a) => a.iter().find_map(|x| match x {
            Ex::Add(ts) => Some(ts.as_slice()),
            _ => None,
        }),
        _ => None,
    }
}

/// Assemble a product from `(coefficient, factor bag)` in its CANONICAL sign
/// placement -- the shared owner behind `mul()`'s final assembly and `term_join`'s
/// negative-coefficient joins, so every site that mints a product builds the SAME
/// spelling the orientation machinery prices (the row-120 law). With no trade site
/// (or a negation overflow, or no configuration that keeps the product's values at its zeros)
/// this is exactly the plain assembly: push the non-unit coefficient, sort, wrap.
fn sign_place(coeff: Rat, out: Vec<Ex>, cx: &Cx) -> Ex {
    sign_place_signed(coeff, out, None, cx)
}

/// The zero of sign `negative` in `f64` (`Some`): `+0` is the literal zero and `-0` the written
/// negative zero. `None`: the one zero.
fn signed_zero(negative: Option<bool>, cx: &Cx) -> Ex {
    match negative {
        Some(true) => Ex::Leaf(cx.view.intern(NEG_ZERO)),
        _ => Ex::Num(Rat::ZERO),
    }
}

/// [`sign_place`] for a product whose coefficient is a zero: in `f64` (`Some(negative)`) the zero
/// carries the product's sign, `+0` or `-0`, and a flip of a site moves it, unless a bare `Const`
/// is beside it (the refit eats every sign); `None`, the one zero eats it.
fn sign_place_zero(negative: Option<bool>, out: Vec<Ex>, cx: &Cx) -> Ex {
    if out.iter().any(|f| matches!(f, Ex::Const(_))) {
        return sign_place(Rat::ZERO, out, cx);
    }
    sign_place_signed(Rat::ZERO, out, negative, cx)
}

/// [`sign_place`], with the zero's sign where the coefficient is a zero that carries one.
fn sign_place_signed(coeff: Rat, out: Vec<Ex>, zero: Option<bool>, cx: &Cx) -> Ex {
    // A placement under the call's own contexts is made once in the call (`CallDecisions`).
    let Some(ctx) = call_decision_key(cx) else {
        return sign_placed(coeff, out, zero, cx);
    };
    let key = (ctx, coeff.clone(), out.clone(), zero);
    if let Some(hit) = CALL_DECISIONS.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|d| d.placements.get(&key).cloned())
    }) {
        return hit;
    }
    let r = sign_placed(coeff, out, zero, cx);
    CALL_DECISIONS.with(|m| {
        if let Some(d) = m.borrow_mut().as_mut() {
            d.placements.insert(key, r.clone());
        }
    });
    r
}

/// [`sign_place_signed`], placed.
fn sign_placed(mut coeff: Rat, mut out: Vec<Ex>, zero: Option<bool>, cx: &Cx) -> Ex {
    // A sum holding a `Const` takes a sign into its terms (`negate_term`: a Const term absorbs it
    // in the refit, every other term negates; `Law::AbsorbSumSign`), so it carries the product's
    // sign exactly, as a negation-absorbing sum does; only a bare `Const` factor or a zero
    // coefficient eats a sign.
    // Beside such a sum the coefficient and the infinities are positive and the sum holds their
    // signs (the Const sign fold, extended to the infinities).
    let const_sum = if !coeff.is_zero() && !out.iter().any(|f| matches!(f, Ex::Const(_))) {
        out.iter()
            .position(|f| matches!(f, Ex::Add(v) if v.iter().any(Ex::contains_const)))
    } else {
        None
    };
    if let Some(i) = const_sum {
        let negs = out.iter().filter(|f| matches!(f, Ex::NegInf)).count();
        if (negs % 2 == 1) != coeff.is_negative() {
            let Ex::Add(terms) = &out[i] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let Some(flipped) = flipped {
                let re = add(flipped, cx);
                if matches!(re, Ex::Add(_)) {
                    out[i] = re;
                    coeff = coeff.abs();
                    for f in out.iter_mut() {
                        if matches!(f, Ex::NegInf) {
                            *f = Ex::PosInf;
                        }
                    }
                }
            }
        } else if negs > 0 || coeff.is_negative() {
            coeff = coeff.abs();
            for f in out.iter_mut() {
                if matches!(f, Ex::NegInf) {
                    *f = Ex::PosInf;
                }
            }
        }
    }
    let assemble = |factors: Vec<Ex>, c: &Rat| -> Ex {
        let mut v = factors;
        // A unit coefficient is no member (`mul.one`, total).
        if !(c.is_one() && Law::MulOne.total()) {
            v.push(Ex::Num(c.clone()));
        }
        match v.len() {
            0 => Ex::Num(Rat::ONE),
            1 => v.pop().unwrap(),
            _ => {
                v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
                Ex::Mul(v)
            }
        }
    };
    // The bag's SIGN CARRIER -- where a sign toggle legally lives (mirroring the
    // bare-inf, Const and negation absorption arms above, which guarantee the ENTRY spelling is
    // carrier-normalized; the owner must return only carrier-normalized spellings
    // too, or the arms and the owner mint two states for one value -- measured live
    // as 9 idem failures at 64k / 50 at 1M when `assemble` parked a raw `-1` beside
    // a bare Const (row 1419) and a bare infinity (row 7320)):
    //   * a bare Const factor: the sign VANISHES into the forall-exists refit -- every
    //     site orientation is value-equal as a family, so orientations are chosen
    //     freely;
    //   * a Const-bearing Add factor: the sign folds into that sum term by term
    //     (`negate_term`; the Const sign fold), the coefficient and the infinities positive;
    //   * a bare +-Inf factor: the infinity carries the sign itself;
    //   * an Add factor with a negation-absorbing member: the sign folds into that
    //     sum member-wise;
    //   * a ZERO coefficient (the kept-zero bag `Mul[0, ..]`): the zero eats every
    //     sign -- `0 * X` and `0 * (-X)` are value-equal in EVERY case (finite -> 0,
    //     +-inf -> nan, nan -> nan), so the whole sign dimension collapses exactly
    //     like a Const-carrier's (fuzz rows 392777/647852:
    //     the kept-zero arm assembled RAW and froze the arrival mirror, so the
    //     direct build and the parse of its own rendering shipped two states);
    //   * otherwise: the rational coefficient.
    {
        // Each site's flip is materialized ONCE (the site test computes it anyway) and
        // reused by every orientation mask below: a flip negates a sum term by term, and
        // each negated term is built through `mul`, which places its own signs the same way,
        // so re-flipping per mask multiplied the work at every nesting level. Each mask's
        // price is the sum of per-factor prices (see `complexity`'s Mul arm), each factor and
        // each flip priced once.
        // A class trades its sign only as a whole: the sites built over one sum (`site_class`)
        // are written in one orientation and flip together, so a flip moves the product's sign
        // by the parity of the class's members. At the zero of the sum every member's base is +0,
        // so the product's value there is the coefficient's infinity -- the limit wherever the
        // pole is one-signed; a single member flipping alone would turn it (`1/(cbrt(u) u)`).
        // Beside a bare `Const`, which absorbs every constant factor of the product and so the
        // orientation sign of every class, every class is a site (`refit_trade_flip`,
        // `Law::AbsorbRefitSign`).
        let refit = out.iter().any(|f| matches!(f, Ex::Const(_)));
        let site_flips: Vec<(usize, Ex)> = out
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let flip = if refit {
                    refit_trade_flip(f, cx)
                } else {
                    sign_trade_flip(f, cx)
                };
                flip.map(|nf| (i, nf))
            })
            .collect();
        // A class is read only where its sum meets another occurrence of itself: two sites over
        // sums of one shape, or a sum inside another factor (`classes_meet`). Elsewhere each site
        // is a class of its own, which nothing pins and no other member checks.
        let meets = classes_meet(&out, &site_flips, cx);
        // Each group: its sites (indices into `site_flips`) and, per site, whether it spells the
        // class against the class's own orientation (`site_class`). A configuration bit picks one
        // orientation for the whole group; the sites spelled against it flip.
        let mut groups: Vec<ClassGroup> = Vec::new();
        for (bit, (pos, _)) in site_flips.iter().enumerate() {
            let class = match site_multiple(&out[*pos], cx) {
                Some(c) => Some(c),
                None if meets => site_class(&out[*pos], cx),
                None => None,
            };
            let joined = class.as_ref().and_then(|(key, flipped)| {
                groups
                    .iter_mut()
                    .find(|(k, _)| k.as_ref() == Some(key))
                    .map(|g| (g, *flipped))
            });
            match joined {
                Some(((_, members), flipped)) => members.push((bit, flipped)),
                None => {
                    let (key, flipped) = class.map_or((None, false), |(k, f)| (Some(k), f));
                    groups.push((key, vec![(bit, flipped)]));
                }
            }
        }
        // Reading A at the zero of each class (`zero_limit`): where the product keeps a value
        // there, a configuration that leaves it off the limit is no candidate. A configuration
        // changes the count of members whose sign at the zero differs from their sign beside it by
        // the sites it flips whose own count changes; a member built over the sum's other
        // orientation inside a product (`sin(-(x1 (x1 - 3)))` for `sin(x1 (3 - x1))`) is counted
        // where it stands, so the product restores the limit an inner orientation moved.
        let toggles_of = |k: &Ex| -> Vec<Option<bool>> {
            site_flips
                .iter()
                .map(|(pos, nf)| {
                    let r = zero_read(&out[*pos], k, cx);
                    if r.parity == Parity::Free {
                        return Some(false);
                    }
                    Some(r.differs? != zero_read(nf, k, cx).differs?)
                })
                .collect()
        };
        let checks: Vec<(bool, Vec<Option<bool>>)> = groups
            .iter()
            .filter_map(|(k, _)| {
                let k = k.as_ref()?;
                let (_, now) = zero_limit(&out, k, cx)?;
                Some((now, toggles_of(k)))
            })
            .collect();
        let factor_price = |f: &Ex| -> u64 {
            match f {
                Ex::Num(r) if r.is_one() => 0,
                Ex::Num(r) if *r == Rat::NEG_ONE => mu_sign(),
                Ex::Num(r) => mu_rat(r),
                _ => complexity(f, cx.view),
            }
        };
        if let (false, Some(nc)) = (site_flips.is_empty(), coeff.checked_neg()) {
            // Priority: Free FIRST -- a bare Const eats EVERY sign (coefficient and
            // bare-infinity signs alike, by the forall-exists refit), so with one present
            // the whole sign dimension collapses and orientations are chosen freely.
            // Checking Inf first split the orbit into two components joined only through
            // the refit (fuzz row 39122: {PosInf, x4-x1} and {PosInf, x1-x4} both stable,
            // one value class). A zero coefficient is Free by the same collapse (see the
            // carrier list above). A Const-bearing sum comes next: it holds the signs of
            // the coefficient and the infinities (normalized above), so every toggle goes
            // into it and the orbit has one component.
            let carrier = if let (Some(negative), true) = (zero, coeff.is_zero()) {
                Carrier::Zero(negative)
            } else if coeff.is_zero() || out.iter().any(|f| matches!(f, Ex::Const(_))) {
                Carrier::Free
            } else if let Some(i) = const_sum {
                Carrier::Absorb(i)
            } else if let Some(i) = out
                .iter()
                .position(|f| matches!(f, Ex::PosInf | Ex::NegInf))
            {
                Carrier::Inf(i)
            } else if let Some(i) = out
                .iter()
                .position(|f| matches!(f, Ex::Add(ts) if ts.iter().any(term_absorbs_negation)))
            {
                Carrier::Absorb(i)
            } else {
                Carrier::Coeff
            };
            let base_price: Vec<u64> = out.iter().map(&factor_price).collect();
            let flip_price: Vec<u64> = site_flips.iter().map(|(_, nf)| factor_price(nf)).collect();
            let orbit = SignOrbit {
                out: &out,
                coeff: &coeff,
                nc: &nc,
                site_flips: &site_flips,
                groups: &groups,
                checks: &checks,
                base_price: &base_price,
                flip_price: &flip_price,
                carrier,
                assemble: &assemble,
            };
            let chosen = sign_place_groups(&orbit, cx);
            // The closed form's price is the enumeration's optimum: checked on every small orbit.
            #[cfg(debug_assertions)]
            if groups.len() <= 6 {
                debug_assert_eq!(
                    chosen.as_ref().map(|c| complexity(c, cx.view)),
                    sign_place_enumerated(&orbit, cx),
                    "the sign orbit's closed form prices off its enumeration"
                );
            }
            if let Some(chosen) = chosen {
                return chosen;
            }
        }
    }
    match zero {
        Some(negative) if coeff.is_zero() => {
            out.push(signed_zero(Some(negative), cx));
            assemble(out, &Rat::ONE)
        }
        _ => assemble(out, &coeff),
    }
}

/// The negation of a sum that carries a product's sign (`sign_place`): a sum holding a `Const`
/// is negated through `add`, which settles a negated member that holds a Const-bearing sum itself
/// (its sign folds into that sum, as `mul()` folds it); any other carrier term by term. `None`
/// where a negation overflows or the sum does not stay one.
fn flip_carrier_sum(f: &Ex, cx: &Cx) -> Option<Ex> {
    let Ex::Add(ts) = f else {
        return None;
    };
    let mut fl = ts
        .iter()
        .map(|x| negate_term(x, cx))
        .collect::<Option<Vec<Ex>>>()?;
    if ts.iter().any(Ex::contains_const) {
        return Some(add(fl, cx)).filter(|s| matches!(s, Ex::Add(_)));
    }
    fl.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    Some(Ex::Add(fl))
}

/// A product's sign carrier (`sign_place`): where a sign toggle lives.
#[derive(Clone, Copy)]
enum Carrier {
    Inf(usize),
    Free,
    Absorb(usize),
    Coeff,
    /// A zero coefficient in `f64`, of the sign given: a toggle turns `+0` into `-0`.
    Zero(bool),
}

/// A product's sign orbit (`sign_place`): its factors and coefficient (and the coefficient's
/// negation), its sign-trade sites with their flips, the classes they form, the zero checks, the
/// prices of every factor and flip, and the sign carrier.
struct SignOrbit<'a> {
    out: &'a [Ex],
    coeff: &'a Rat,
    nc: &'a Rat,
    site_flips: &'a [(usize, Ex)],
    groups: &'a [ClassGroup],
    checks: &'a [(bool, Vec<Option<bool>>)],
    base_price: &'a [u64],
    flip_price: &'a [u64],
    carrier: Carrier,
    assemble: &'a dyn Fn(Vec<Ex>, &Rat) -> Ex,
}

/// One orientation of a group of sites (`SignOrbit::group_options`): the sites it flips, their
/// price, whether it moves the product's sign, and its sites as they then stand (the tie order).
#[derive(Clone)]
struct SiteOption {
    flips: Vec<usize>,
    cost: u64,
    odd: bool,
    key: Vec<Ex>,
    /// The signs of its sites as they then stand (`sign_key`, summed).
    signs: (u32, u32),
}

impl SignOrbit<'_> {
    /// The product the flipped sites give, the carrier taking the sign their count moves, and
    /// whether its coefficient is negative; `None` where the carrier cannot take it.
    fn candidate(&self, flipped: &[bool], cx: &Cx) -> Option<(Ex, bool)> {
        let mut factors = self.out.to_vec();
        let mut toggled = false;
        for (bit, (pos, nf)) in self.site_flips.iter().enumerate() {
            if flipped[bit] {
                factors[*pos] = nf.clone();
                toggled = !toggled;
            }
        }
        if let Carrier::Zero(negative) = self.carrier {
            let negative = negative ^ toggled;
            factors.push(signed_zero(Some(negative), cx));
            return Some(((self.assemble)(factors, &Rat::ONE), negative));
        }
        let c = match self.carrier {
            Carrier::Zero(_) => unreachable!("the zero carrier is assembled above"),
            Carrier::Coeff if toggled => self.nc.clone(),
            Carrier::Coeff => self.coeff.clone(),
            Carrier::Inf(i) => {
                if toggled {
                    factors[i] = match &factors[i] {
                        Ex::PosInf => Ex::NegInf,
                        _ => Ex::PosInf,
                    };
                }
                self.coeff.clone()
            }
            // The refit eats the toggle: every configuration is value-equal as a fitted family.
            // The whole sign dimension goes into the refit: coefficient magnitude only, bare
            // infinities positive.
            Carrier::Free => {
                for f in factors.iter_mut() {
                    if matches!(f, Ex::NegInf) {
                        *f = Ex::PosInf;
                    }
                }
                if self.coeff.is_negative() {
                    self.nc.clone()
                } else {
                    self.coeff.clone()
                }
            }
            Carrier::Absorb(i) => {
                if toggled {
                    factors[i] = flip_carrier_sum(&factors[i], cx)?;
                }
                self.coeff.clone()
            }
        };
        let negative = c.is_negative();
        Some(((self.assemble)(factors, &c), negative))
    }

    /// Each group's orientations: the class written along its key or against it, every site
    /// spelled the other way flipping. An orientation that flips a site onto the base of another
    /// factor is none: the bag would be an unmerged spelling that collection merges, so the
    /// product would not read back as itself (two factors the entry already holds over one base
    /// are how `mul()` emits them, and block nothing).
    fn group_options(&self, cx: &Cx) -> Vec<Vec<SiteOption>> {
        let view = cx.view;
        let mut site_of: Vec<Option<usize>> = vec![None; self.out.len()];
        let mut group_of = vec![0usize; self.site_flips.len()];
        for (bit, (pos, _)) in self.site_flips.iter().enumerate() {
            site_of[*pos] = Some(bit);
        }
        for (g, (_, members)) in self.groups.iter().enumerate() {
            for (bit, _) in members {
                group_of[*bit] = g;
            }
        }
        let state = |bit: usize, flipped: bool| -> &Ex {
            if flipped {
                &self.site_flips[bit].1
            } else {
                &self.out[self.site_flips[bit].0]
            }
        };
        self.groups
            .iter()
            .enumerate()
            .map(|(g, (_, members))| {
                [false, true]
                    .into_iter()
                    .filter_map(|against| {
                        let flips: Vec<usize> = members
                            .iter()
                            .filter(|(_, other)| *other != against)
                            .map(|(bit, _)| *bit)
                            .collect();
                        for &bit in &flips {
                            let (pos, nf) = &self.site_flips[bit];
                            for j in (0..self.out.len()).filter(|j| j != pos) {
                                let hit = match site_of[j] {
                                    Some(b) if group_of[b] == g => {
                                        same_collection_base(nf, state(b, flips.contains(&b)), cx)
                                    }
                                    Some(b) => {
                                        same_collection_base(nf, state(b, false), cx)
                                            || same_collection_base(nf, state(b, true), cx)
                                    }
                                    None => same_collection_base(nf, &self.out[j], cx),
                                };
                                if hit {
                                    return None;
                                }
                            }
                        }
                        let cost = members.iter().fold(0u64, |acc, (bit, _)| {
                            acc.saturating_add(if flips.contains(bit) {
                                self.flip_price[*bit]
                            } else {
                                self.base_price[self.site_flips[*bit].0]
                            })
                        });
                        let mut key: Vec<Ex> = members
                            .iter()
                            .map(|(bit, _)| state(*bit, flips.contains(bit)).clone())
                            .collect();
                        key.sort_by(|a, b| cmp_ex(a, b, view));
                        let signs = key
                            .iter()
                            .map(|f| sign_key(f, view))
                            .fold((0, 0), |a, k| (a.0 + k.0, a.1 + k.1));
                        Some(SiteOption {
                            odd: flips.len() % 2 == 1,
                            flips,
                            cost,
                            key,
                            signs,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Does a configuration (the sites it flips) keep every value the product keeps at a class's
    /// zero (`zero_limit`)? A site whose own count is not read may not flip.
    fn keeps(&self, checks: &[usize], flipped: &[bool]) -> bool {
        checks.iter().all(|&c| {
            let (now, toggles) = &self.checks[c];
            let mut off = *now;
            for (bit, t) in toggles.iter().enumerate() {
                if flipped[bit] {
                    match t {
                        Some(t) => off ^= t,
                        None => return false,
                    }
                }
            }
            !off
        })
    }
}

/// THE SIGN ORBIT'S ARGMIN, for any number of classes. A product's price is the sum of its
/// factors' prices and the carrier's (`complexity`'s Mul arm; pieces of a literal, `is_piece`,
/// price together, which a site that is one reads as its own price), so the cheapest spelling is
/// separable up to the one coupling the classes share, the PARITY of the sites they flip (the
/// sign the carrier takes). Each class flips as a whole and prices as the sum of its members; a
/// zero check (`zero_limit`) constrains the classes whose sites it counts, and the classes one
/// check couples are decided together (a component, its orientations enumerated: in practice a
/// check counts the sites of its own class only). Per component its cheapest orientation, and per
/// parity of the sign the one component whose switch to the other parity costs least: the
/// cheapest product of either sign, priced exactly, the cheaper winning, on an equal price the tie
/// order (`tie_cmp`: the sign key, then the structure). A normal-form choice: the configurations
/// are admitted by the zero checks, never by the price. Ties within a parity resolve by a fixed
/// order that is a function of the orbit, never of the entry spelling: each component's
/// orientation by its sites as they then stand, the switch by the components' smallest site.
/// `None` where no configuration keeps the product's values at its zeros, or a component
/// couples more than `COMPONENT_LIMIT` classes: the caller keeps the entry spelling.
fn sign_place_groups(o: &SignOrbit, cx: &Cx) -> Option<Ex> {
    const COMPONENT_LIMIT: usize = 12;
    let view = cx.view;
    let n = o.site_flips.len();
    let options = o.group_options(cx);
    if options.iter().any(Vec::is_empty) {
        return None;
    }
    let mut group_of = vec![0usize; n];
    for (g, (_, members)) in o.groups.iter().enumerate() {
        for (bit, _) in members {
            group_of[*bit] = g;
        }
    }
    // Components: the groups each check couples (the sites whose flip changes its count, or whose
    // count is not read), joined.
    let mut root: Vec<usize> = (0..o.groups.len()).collect();
    fn find(root: &mut [usize], g: usize) -> usize {
        let mut r = g;
        while root[r] != r {
            r = root[r];
        }
        root[g] = r;
        r
    }
    let mut check_groups: Vec<Vec<usize>> = Vec::with_capacity(o.checks.len());
    for (now, toggles) in o.checks {
        let mut gs: Vec<usize> = toggles
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != Some(false))
            .map(|(bit, _)| group_of[bit])
            .collect();
        gs.sort_unstable();
        gs.dedup();
        if gs.is_empty() && *now {
            return None;
        }
        for w in gs.windows(2) {
            let (a, b) = (find(&mut root, w[0]), find(&mut root, w[1]));
            root[a.max(b)] = a.min(b);
        }
        check_groups.push(gs);
    }
    let mut components: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for g in 0..o.groups.len() {
        let r = find(&mut root, g);
        match components
            .iter_mut()
            .find(|(gs, _)| find(&mut root, gs[0]) == r)
        {
            Some((gs, _)) => gs.push(g),
            None => components.push((vec![g], Vec::new())),
        }
    }
    for (c, gs) in check_groups.iter().enumerate() {
        if let Some(&g) = gs.first() {
            let r = find(&mut root, g);
            if let Some((_, cs)) = components
                .iter_mut()
                .find(|(groups, _)| find(&mut root, groups[0]) == r)
            {
                cs.push(c);
            }
        }
    }
    // Each component's orientations, cheapest first (then by its sites as they stand), and its
    // rank: its smallest site in either orientation.
    let mut decided: Vec<(Ex, Vec<SiteOption>)> = Vec::with_capacity(components.len());
    for (gs, cs) in &components {
        if gs.len() > COMPONENT_LIMIT {
            return None;
        }
        let mut opts: Vec<SiteOption> = vec![SiteOption {
            flips: Vec::new(),
            cost: 0,
            odd: false,
            key: Vec::new(),
            signs: (0, 0),
        }];
        for &g in gs {
            opts = opts
                .into_iter()
                .flat_map(|a| {
                    options[g].iter().map(move |b| {
                        let mut flips = a.flips.clone();
                        flips.extend(b.flips.iter().copied());
                        let mut key = a.key.clone();
                        key.extend(b.key.iter().cloned());
                        SiteOption {
                            flips,
                            cost: a.cost.saturating_add(b.cost),
                            odd: a.odd ^ b.odd,
                            key,
                            signs: (a.signs.0 + b.signs.0, a.signs.1 + b.signs.1),
                        }
                    })
                })
                .collect();
        }
        if !cs.is_empty() {
            opts.retain(|opt| {
                let mut flipped = vec![false; n];
                for &bit in &opt.flips {
                    flipped[bit] = true;
                }
                o.keeps(cs, &flipped)
            });
        }
        if opts.is_empty() {
            return None;
        }
        opts.sort_by(|a, b| {
            (a.cost, a.signs)
                .cmp(&(b.cost, b.signs))
                .then_with(|| cmp_vec(&a.key, &b.key, view))
        });
        let rank = gs
            .iter()
            .flat_map(|g| o.groups[*g].1.iter())
            .flat_map(|(bit, _)| [&o.out[o.site_flips[*bit].0], &o.site_flips[*bit].1])
            .min_by(|a, b| cmp_ex(a, b, view))
            .cloned()
            .unwrap_or(Ex::Num(Rat::ONE));
        decided.push((rank, opts));
    }
    decided.sort_by(|a, b| cmp_ex(&a.0, &b.0, view));
    // Per parity, the cheapest configuration: every component's cheapest orientation, and where
    // their parity is the other one, the cheapest switch of one component to its cheapest
    // orientation of the other parity (among switches of one cost, the one whose product is
    // structurally smallest).
    let best: Vec<&SiteOption> = decided.iter().map(|(_, opts)| &opts[0]).collect();
    let build = |pick: &[&SiteOption]| {
        let mut flipped = vec![false; n];
        for opt in pick {
            for &bit in &opt.flips {
                flipped[bit] = true;
            }
        }
        o.candidate(&flipped, cx)
    };
    // Where the sign is absorbed (a bare `Const`, which absorbs every constant factor of its
    // product, or a zero coefficient) the parity is no constraint: every class takes its own
    // cheapest orientation, ties by its sites as they then stand, whatever else the product holds.
    if matches!(o.carrier, Carrier::Free) {
        return build(&best).map(|(e, _)| e);
    }
    let parity = best.iter().fold(false, |p, opt| p ^ opt.odd);
    type Delta = (u64, i64, i64);
    let switches: Vec<(Delta, usize, &SiteOption)> = decided
        .iter()
        .enumerate()
        .filter_map(|(i, (_, opts))| {
            let other = opts.iter().find(|opt| opt.odd != opts[0].odd)?;
            let delta = (
                other.cost - opts[0].cost,
                i64::from(other.signs.0) - i64::from(opts[0].signs.0),
                i64::from(other.signs.1) - i64::from(opts[0].signs.1),
            );
            Some((delta, i, other))
        })
        .collect();
    let least = switches.iter().map(|(delta, _, _)| *delta).min();
    let mut chosen: Option<(u64, Ex)> = None;
    for target in [false, true] {
        let built = if target == parity {
            build(&best)
        } else {
            switches
                .iter()
                .filter(|(delta, _, _)| Some(*delta) == least)
                .filter_map(|(_, i, other)| {
                    let mut pick = best.clone();
                    pick[*i] = other;
                    build(&pick)
                })
                .min_by(|a, b| tie_cmp(&a.0, &b.0, view))
        };
        let Some((cand, _)) = built else {
            continue;
        };
        let mu = complexity(&cand, view);
        let better = match &chosen {
            None => true,
            Some((bmu, bex)) => {
                mu < *bmu || (mu == *bmu && tie_cmp(&cand, bex, view) == Ordering::Less)
            }
        };
        if better {
            chosen = Some((mu, cand));
        }
    }
    chosen.map(|(_, e)| e)
}

/// The debug build's cross-check of `sign_place_groups`: the cheapest price over every
/// configuration of whole classes, under the same orientations and checks.
#[cfg(debug_assertions)]
fn sign_place_enumerated(o: &SignOrbit, cx: &Cx) -> Option<u64> {
    let n = o.site_flips.len();
    let options = o.group_options(cx);
    let all: Vec<usize> = (0..o.checks.len()).collect();
    let mut best: Option<u64> = None;
    for code in 0usize..(1usize << o.groups.len()) {
        let mut flipped = vec![false; n];
        let mut allowed = true;
        for (g, (_, members)) in o.groups.iter().enumerate() {
            let against = code & (1 << g) != 0;
            let flips: Vec<usize> = members
                .iter()
                .filter(|(_, other)| *other != against)
                .map(|(bit, _)| *bit)
                .collect();
            allowed &= options[g].iter().any(|opt| opt.flips == flips);
            for bit in flips {
                flipped[bit] = true;
            }
        }
        if !allowed || !o.keeps(&all, &flipped) {
            continue;
        }
        if let Some((cand, _)) = o.candidate(&flipped, cx) {
            let mu = complexity(&cand, cx.view);
            best = Some(best.map_or(mu, |b| b.min(mu)));
        }
    }
    best
}

/// Cheap shape test for [`sign_trade_flip`] -- true iff the factor is an odd
/// carrier of a tradeable mixed-sign sum. Must stay the exact mirror of that
/// function's match (this one detects, that one materializes).
fn is_sign_trade_site(f: &Ex, cx: &Cx) -> bool {
    // Delegates to the materializing test: the FILES-BARE gate needs the flipped
    // terms, so a cheap shape-only mirror cannot exist without drifting (the
    // mixed-sign counting attempt regressed 5 corpus rows -- see sign_trade_flip).
    sign_trade_flip(f, cx).is_some()
}

/// Is `f` a sign-trade site -- an odd carrier of a mixed-sign sum -- and if so,
/// the factor with that sum FLIPPED (`None` = not a site, or refused: Const-bearing
/// and negation-absorbing sums have absorption owners, and negation
/// overflow fails safe). The odd carriers, each a TOTAL identity f(-S) = -f(S) on the
/// extended reals: the sum itself; Pow(S, odd n >= 3); rootn(S, odd m >= 3) (the
/// SIGNED total root is an odd bijection); the eight odd functions (`odd_fun`); and
/// Pow(c, odd n >= 3) of either of the last two over a sum. A negative odd power of any of them
/// is one too, its identity failing only at the pole: there the two sides disagree, so the sign
/// of the pole's infinity is no value the answer keeps (reading A). A product flips the sites of
/// one class only together (`sign_place`), so a one-signed pole keeps its infinity.
///
/// MEMOIZED for the dynamic extent of the outermost call: the flip is a pure function of
/// `(f, cx)` and the thread's number domain, and computing it re-enters itself FOUR times per
/// nesting level -- it negates each term through `term_join`, whose site test
/// (`is_sign_trade_site`) and `mul()` assembly (`sign_place`) each ask the flip of the nested
/// sum, and the files-bare gate (`flipped_orientation_wins`) negates the flipped terms again
/// through the same two. Without the memo a sum nested `d` deep cost `4^d` flips (measured:
/// `x1*(x1 + (N + 1)/x1)` nested 9 deep took 7 s with the search off, all of it here).
fn sign_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    trade_flip(f, cx, false)
}

/// The flip of `f` as a factor of a product that holds a bare `Const` (`sign_place`). A
/// `<constant>` absorbs every constant factor of its product, and a class's orientation sign is
/// one, the constant factor -1: beside one, the orientation of EVERY class is free, so the classes
/// whose sign an absorption owner holds elsewhere (a sum holding a `Const`, one with a
/// negation-absorbing member, one with an infinity) are sites too, their sign the constant's.
fn refit_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    trade_flip(f, cx, true)
}

/// [`sign_trade_flip`] and [`refit_trade_flip`], memoized; `refit`: the product's sign is a bare
/// `Const`'s.
fn trade_flip(f: &Ex, cx: &Cx, refit: bool) -> Option<Ex> {
    if !matches!(f, Ex::Add(_) | Ex::Pow(..) | Ex::Fun(..)) {
        return None;
    }
    let key = (trade_cx_key(cx), f.clone(), refit);
    let lived = call_lived(&key.0);
    if lived {
        if let Some(hit) =
            CALL_DECISIONS.with(|m| m.borrow().as_ref().and_then(|d| d.flips.get(&key).cloned()))
        {
            return hit;
        }
    }
    let outer = FLIP_MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_none() {
            *m = Some(rustc_hash::FxHashMap::default());
            true
        } else {
            false
        }
    });
    let _scope = FlipScope(outer);
    if let Some(hit) = FLIP_MEMO.with(|m| m.borrow().as_ref().and_then(|t| t.get(&key).cloned())) {
        return hit;
    }
    let r = sign_trade_flip_raw(f, cx, refit);
    if lived {
        CALL_DECISIONS.with(|m| {
            if let Some(d) = m.borrow_mut().as_mut() {
                d.flips.insert(key.clone(), r.clone());
            }
        });
    }
    FLIP_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.insert(key, r.clone());
        }
    });
    r
}

/// THE CALL'S DECISIONS: the flips ([`trade_flip`]) and the opposite-sums decisions
/// ([`collect_opposite_sums`]) made under the simplify call's own contexts, remembered for the
/// whole call. Each is a function of its key (the factor or the product as it arrived, and the
/// context: its token view, certificate closures, mode, switches and the thread's number
/// domain), so a decision met again in a later product -- the search rebuilds every candidate
/// through the constructors, round after round, over sums whose orientation does not change --
/// is the one already made. A key is remembered beyond its decision only where every closure it
/// names lives as long as the call (`call_lived`): a context a caller builds around a closure of
/// its own (a sum's held sign, `Cx::sign_held`) is keyed by that closure's address, which a later
/// caller's closure may take, so its decisions keep the extent of the decision that made them
/// ([`FlipScope`]).
pub(crate) struct CallDecisions(Option<CallMemo>);

/// The call's decisions and the identities of its own contexts' closures and token view.
pub(crate) struct CallMemo {
    lived: Vec<usize>,
    flips: FlipMemo,
    opposite: rustc_hash::FxHashMap<OppositeKey, Ex>,
    placements: rustc_hash::FxHashMap<(DecisionCxKey, Rat, Vec<Ex>, Option<bool>), Ex>,
    orientations: rustc_hash::FxHashMap<(DecisionCxKey, Vec<Ex>, Wrapper, Tie), bool>,
}

/// The context part of a placement's or an orientation's key: the flip's
/// ([`trade_cx_key`]) and the enclosure certificate's identity, which the constructors a
/// decision builds through read too.
type DecisionCxKey = (FlipCxKey, usize);

/// The context key of a decision under one of the call's own contexts (`call_lived`), `None`
/// under any other.
fn call_decision_key(cx: &Cx) -> Option<DecisionCxKey> {
    let key = trade_cx_key(cx);
    let encl = cx.cert_encl.map_or(0, |f| {
        f as *const dyn Fn(&Ex) -> Option<(f64, f64)> as *const () as usize
    });
    let lived = call_lived(&key)
        && (encl == 0
            || CALL_DECISIONS
                .with(|m| m.borrow().as_ref().is_some_and(|d| d.lived.contains(&encl))));
    lived.then_some((key, encl))
}

thread_local! {
    static CALL_DECISIONS: std::cell::RefCell<Option<CallMemo>> =
        const { std::cell::RefCell::new(None) };
}

impl CallDecisions {
    /// Opens the call's memo for the contexts of `cx` (its token view and certificate closures),
    /// restoring the caller's when it drops.
    pub(crate) fn open(cx: &Cx) -> Self {
        let (view, certs, _) = flip_cx_key(cx);
        let mut lived: Vec<usize> = certs.iter().copied().filter(|&c| c != 0).collect();
        lived.push(view);
        if let Some(f) = cx.cert_encl {
            lived.push(f as *const dyn Fn(&Ex) -> Option<(f64, f64)> as *const () as usize);
        }
        CallDecisions(CALL_DECISIONS.with(|m| {
            m.borrow_mut().replace(CallMemo {
                lived,
                flips: FlipMemo::default(),
                opposite: rustc_hash::FxHashMap::default(),
                placements: rustc_hash::FxHashMap::default(),
                orientations: rustc_hash::FxHashMap::default(),
            })
        }))
    }
}

impl Drop for CallDecisions {
    fn drop(&mut self) {
        let outer = self.0.take();
        CALL_DECISIONS.with(|m| *m.borrow_mut() = outer);
    }
}

/// Does every closure the context key names (and its token view) live as long as the call? Not
/// while a fire log records the folds (`rules::FireLog`): a decision taken from the memo would not
/// record the folds of its constructions again.
fn call_lived(key: &FlipCxKey) -> bool {
    if super::rules::folds_logged() {
        return false;
    }
    CALL_DECISIONS.with(|m| {
        m.borrow().as_ref().is_some_and(|d| {
            d.lived.contains(&key.0) && key.1.iter().all(|&c| c == 0 || d.lived.contains(&c))
        })
    })
}

/// The memo key's context part: everything `sign_trade_flip` reads besides the factor -- the
/// token view, the four certificate closures (by identity), the mode, the two fold switches and
/// the thread's number domain. All of them outlive the outermost call, so no identity is reused
/// within the memo's extent.
type FlipCxKey = (usize, [usize; 6], u8);

type FlipMemo = rustc_hash::FxHashMap<(FlipCxKey, Ex, bool), Option<Ex>>;

/// The opposite-sums decision of a product (`collect_opposite_sums`), keyed like a flip: by the
/// context, the sign host and the factors as they arrived. It lives exactly as long as the flip
/// memo, so a parse or a re-walk decides each distinct product once however often it occurs.
type OppositeKey = (FlipCxKey, HostKey, Vec<Ex>);

thread_local! {
    static FLIP_MEMO: std::cell::RefCell<Option<FlipMemo>> = const { std::cell::RefCell::new(None) };
    static OPPOSITE_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<OppositeKey, Ex>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

/// Drops the memos when the outermost `sign_trade_flip`, opposite-sums decision, parse or
/// re-walk returns (or unwinds).
pub(crate) struct FlipScope(bool);

impl FlipScope {
    /// Holds the memos open until the guard drops (unless an enclosing scope already does), so
    /// the flips of one decision are computed once however many products it prices, and a parse
    /// or re-walk decides each distinct product once.
    pub(crate) fn open() -> Self {
        let outer = FLIP_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            if m.is_none() {
                *m = Some(rustc_hash::FxHashMap::default());
                true
            } else {
                false
            }
        });
        FlipScope(outer)
    }
}

impl Drop for FlipScope {
    fn drop(&mut self) {
        if self.0 {
            FLIP_MEMO.with(|m| *m.borrow_mut() = None);
            OPPOSITE_MEMO.with(|m| m.borrow_mut().clear());
        }
    }
}

fn flip_cx_key(cx: &Cx) -> FlipCxKey {
    fn id(c: Option<&dyn Fn(&Ex) -> bool>) -> usize {
        c.map_or(0, |f| f as *const dyn Fn(&Ex) -> bool as *const () as usize)
    }
    let mode = match cx.mode {
        RuleMode::Default => 0u8,
        RuleMode::Permissive => 2,
    };
    let bits = mode
        | (u8::from(cx.fold_f64) << 2)
        | (u8::from(super::rat::f64_numbers()) << 4)
        | (u8::from(cx.flat_unread) << 6);
    (
        cx.view as *const TokenView as usize,
        [
            id(cx.cert_fin),
            id(cx.cert_finnz),
            id(cx.cert_nzae),
            id(cx.cert_nce),
            id(cx.cert_region),
            id(cx.sign_held),
        ],
        bits,
    )
}

/// [`flip_cx_key`] with whether the context reads a sum's zero (`Cx::zero_sign_unread`): a sign
/// trade and a product's opposite sums depend on it (`sum_holds_its_sign`), the region test does
/// not.
fn trade_cx_key(cx: &Cx) -> FlipCxKey {
    let (view, certs, bits) = flip_cx_key(cx);
    (view, certs, bits | (u8::from(cx.zero_sign_unread) << 5))
}

fn sign_trade_flip_raw(f: &Ex, cx: &Cx, refit: bool) -> Option<Ex> {
    let flip_sum = |ts: &[Ex]| -> Option<Vec<Ex>> {
        // Exclusions = the absorption owners' classes: Const-bearing,
        // absorbing-member, and BARE-infinity terms (the inf carries the
        // sign itself -- `term_absorbs_negation` only matches inf-carrying
        // PRODUCTS, so the bare member needs its own test; missing it let an
        // odd-fun trade flip an inf-bearing sum and inflate mu: fuzz row 131227,
        // the one violation in 200k). Beside a bare `Const` (`refit`) the
        // constant owns every class's sign, theirs too: such a sum negates through
        // `add`, which files it bare (`primitive_sum` never wraps one).
        if ts.iter().any(Ex::contains_const)
            || ts.iter().any(term_absorbs_negation)
            || ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
        {
            if !refit {
                return None;
            }
            let negated = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()?;
            return match add(negated, cx) {
                Ex::Add(v) => Some(v),
                _ => None,
            };
        }
        let mut out = ts
            .iter()
            .map(|t| negate_term(t, cx))
            .collect::<Option<Vec<Ex>>>()?;
        // FILES-BARE gate: the flipped multiset must be a spelling the sum's own orientation
        // (`flipped_orientation_wins`) files bare. Where it would wrap it (`-a - b` rests as
        // `-1 x (a + b)`) the site is none: trading into it would mint the wrapper again, and a
        // power over it would freeze a one-signed sum its constructors do not file. Counting signs
        // is not the same test: an odd literal's sign sits inside a function's argument, where no
        // coefficient shows it.
        if flipped_orientation_wins(&out, Wrapper::Flipped, Tie::Keep, cx) {
            return None;
        }
        // A sign enters a sum only where it holds none (`Law::ClassSignTrade`, as `add.content` in
        // `primitive_sum`): float64's sum is `+0` wherever it cancels, so `-(a - b)` is `-0` there
        // and `b - a` is `+0`, a sign the expression around it may read (`2*(-a - b)` is not
        // `-2*(a + b)` where `a + b` cancels on a region). Asked last.
        if !Law::ClassSignTrade.holds(&Ex::Add(ts.to_vec()), cx) {
            return None;
        }
        out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        Some(out)
    };
    let odd_int_ge3 = |e: &Ex| -> bool {
        matches!(e, Ex::Num(r) if r.is_odd_integer() && r.cmp_int(3) != Ordering::Less)
    };
    // A negative odd power's pole is two-signed, so its orientation is not pinned there
    // (`-1/(x1 - 3)` is `1/(3 - x1)`).
    let odd_power = |e: &Ex| -> bool {
        odd_int_ge3(e) || matches!(e, Ex::Num(r) if r.is_odd_integer() && r.is_negative())
    };
    match f {
        Ex::Add(ts) => flip_sum(ts).map(Ex::Add),
        Ex::Pow(b, e) if odd_power(e) => {
            let flipped = match &**b {
                Ex::Add(ts) => flip_sum(ts).map(|s| Ex::Pow(Box::new(Ex::Add(s)), e.clone())),
                // An odd power of an odd head of a negatable argument (`class_base`) is one too:
                // `f(-u)^n = -f(u)^n`, for a negative `n` as for a positive one. A constant
                // argument keeps its sign on its literal under a power (`1/tanh(-2)`), as the
                // sign key files it.
                Ex::Fun(op, args)
                    if odd_head(cx.view, *op, args)
                        .map_or(args.first(), |(p, _)| args.get(p))
                        .is_some_and(|a| {
                            a.has_measure_space(cx.view) || a.contains_wildcard(cx.view)
                        }) =>
                {
                    trade_flip(b, cx, refit).map(|nb| Ex::Pow(Box::new(nb), e.clone()))
                }
                _ => None,
            }?;
            // A negative power's pole pins the orientation where the base's zero set may hold a
            // region: there the inverse is infinite on the region, a value the answer keeps. Asked
            // last, of a base whose flip exists.
            (cx.permits(Permissive::Agreement)
                || !matches!(&**e, Ex::Num(r) if r.is_negative())
                || zero_set_null(b, cx))
            .then_some(flipped)
        }
        Ex::Fun(op, args)
            if args.len() == 2 && cx.view.tok_is(*op, "rootn") && odd_int_ge3(&args[1]) =>
        {
            let Ex::Add(ts) = &args[0] else {
                return literal_multiple_flip(f, cx);
            };
            flip_sum(ts).map(|s| Ex::Fun(*op, vec![Ex::Add(s), args[1].clone()]))
        }
        Ex::Fun(op, args)
            if (args.len() == 1 && odd_fun(cx.view, *op))
                || (args.len() == 2
                    && cx.view.tok_is(*op, "besselj")
                    && odd_head(cx.view, *op, args).is_some()) =>
        {
            let pos = args.len() - 1;
            let with = |a: Ex| {
                let mut v = args.clone();
                v[pos] = a;
                v
            };
            match &args[pos] {
                Ex::Add(ts) => flip_sum(ts).map(|s| Ex::Fun(*op, with(Ex::Add(s)))),
                // An odd function of a LITERAL is a trade site too -- f(-r) = -f(r)
                // exactly, so the sign relocates between the coefficient slot and the
                // literal, and the orbit argmin (mu, then the NON-NEGATIVE-coefficient tie,
                // then cmp_ex) lands every route on the ruled direction: the literal
                // owns the sign (`-5 * sin(2)` files as `5 * sin(-2)`; `-1 * sin(2)` as
                // `sin(-2)`, the Mul node dissolving). This is the hoist of `term_join`'s
                // former private fusion arm into the shared owner: `mul()`'s direct
                // assembly and the collector's rebuild now price the SAME orbit. Rebuilt
                // through `fun()`, NEVER raw, so a boundary ground keeps folding
                // (`atanh(-1)` must reach `-inf`); a rebuild that FOLDS to a non-Fun is
                // refused as a site (fail-safe: the orbit stays type-stable, and such
                // grounds fold on their own paths).
                //
                // In `f64` an odd function of the literal ZERO is no site: `-sin(0)` is -0 in
                // float64 and `sin(-0)` would need the written negative zero, so the sign stays on
                // the coefficient, where the ground fold of a ground product writes float64's -0
                // and the zero of a product with a variable is the one zero (`-x1*sin(0)` is
                // `0`). Read as the zero again, the flip would drop the sign (`-sin(0)` as
                // `sin(0)`, a +0). `permissive` reads one zero.
                Ex::Num(r) if r.is_zero() && cx.f64_semantics() && cx.f64_numbers() => None,
                Ex::Num(r) => {
                    let flipped = fun(*op, with(Ex::Num(r.checked_neg()?)), cx);
                    matches!(flipped, Ex::Fun(..)).then_some(flipped)
                }
                _ => literal_multiple_flip(f, cx),
            }
        }
        Ex::Fun(..) => literal_multiple_flip(f, cx),
        _ => None,
    }
}

/// The other spelling of an odd function or odd root of a literal multiple
/// (`odd_of_literal_multiple`): the argument negated through `mul()`, rebuilt through `fun()`;
/// `None` where the rebuild leaves the shape (a fold, or a negation the argument's own sites took
/// instead of the literal).
fn literal_multiple_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    let c = odd_of_literal_multiple(f, cx.view)?;
    let Ex::Fun(op, args) = f else { return None };
    let (pos, _) = odd_head(cx.view, *op, args)?;
    let mut items = vec![Ex::Num(c.checked_neg()?)];
    match &args[pos] {
        Ex::Num(_) => {}
        Ex::Mul(v) if matches!(v.first(), Some(Ex::Num(_))) => items.extend(v[1..].iter().cloned()),
        u => items.push(u.clone()),
    }
    let mut flipped_args = args.clone();
    flipped_args[pos] = mul(items, cx);
    let flipped = fun(*op, flipped_args, cx);
    matches!(
        class_base(&flipped, cx.view),
        Some(ClassBase { head: Some(_), .. })
    )
    .then_some(flipped)
}

// The residual equal-mu same-coefficient-sign tie in the
// sign-placement owner breaks on `cmp_ex` -- the ONE canonical total order, which
// compares Leaf/Fun tokens by their STRINGS. The dedicated `ex_struct_cmp` this
// replaced compared them by raw token ID, i.e. by INTERNING ORDER, so on a fully
// mu-tied orbit (the two mirror pairings price alike) the winner was a function of which
// literal the input stream interned first: `-1 * (a-b)*(c-d)` and its factor-swapped
// spelling picked OPPOSITE mirror pairings (extreme-lane fuzz rows 829655/866090, the
// two regressions at 1M rows).

/// LOSSY simplification priced reciprocal products WORSE
/// than sound's on the sound-REFUSED set. The blanket zero-set licence makes the lossy
/// `pow` distribute `(a*b)^-1 -> a^-1 * b^-1` unconditionally -- the match-enabling
/// normal form the `$`-cancellation needs (`x * inv(x*y) -> inv y` only cancels through
/// the flat bag) -- but each distributed factor carries its own `^-1` wrapper, so where
/// NO partner cancels, the lossy endpoint priced strictly above sound's licence-refusal
/// form `(a*b)^-1` (registered row 1414: mu 80 vs 72). Neither mode consulted `mu` when
/// choosing the spelling; sound's smaller form was a fail-closed accident.
///
/// This is an OUTPUT-BOUNDARY projection (`perm.reciprocal_projection`), deliberately NOT a
/// constructor arm. The first
/// build of this fix put the join into `mul`'s final assembly, making the joined form
/// the WORKING representation -- disqualified by the debug stability assert: `pow`'s
/// distribution arm settles its two parts in a nested vacuum `mul` that joined them
/// before the enclosing bag could splice, so construction order leaked into the state
/// (and joined mid-chain states also shift rule reachability and the descent gate's mu
/// ordering). The distributed form is the right working representation (match-enabling,
/// cancellation-visible); the joined form is only ever a better FINAL SPELLING. So the
/// rewrite chain runs entirely in the distributed canon, byte-identical to pre-fix, and
/// [`rejoin_projection`] applies this decision once, bottom-up, to the returned
/// endpoint (measured on the 200k battery: mode-ordering rows 822 -> 780, 42 fixed,
/// 0 added).
///
/// The decision, per `Mul` node: offer the negative-integer-power members back as one
/// joined `(prod f^|n|)^-1`; the strictly `mu`-smaller of the two spellings wins (tie:
/// distributed, the incumbent). The `mu` arithmetic makes the win set exact: for k
/// joined factors the join saves (k-1) `Pow` wrappers and pays one inner `Mul` symbol,
/// so k = 2 INSIDE a wider bag is a tie (distributed stays) while the whole-bag pair
/// (the outer `Mul` unwraps: row 1414's shape) and every k >= 3 join strictly. Two
/// gates keep the decision honest:
///
/// * SOUND ALIGNMENT: the join is offered only where sound-mode `pow` would REFUSE to
///   re-distribute it (`nz_ae_certified` / `certainly_nonneg` per member -- the same
///   conjuncts as the distribution licence, minus the lossy blanket). Where sound
///   distributes, lossy stays distributed too: the two output forms agree there.
///   Idempotence is a funnel property: re-simplifying a joined output re-parses it
///   through `pow`'s blanket distribution back to a working endpoint whose
///   projection must reproduce the output. That endpoint is NOT in general the
///   state the join was priced against (the two construction routes -- positive
///   inner bag vs hidden `^-1` members -- orient trade-site mirrors differently),
///   so the decision is additionally gated on the candidate's RE-SETTLE IMAGE
///   (see the comparison at the return).
/// * ELIGIBILITY excludes BARE-`Const` bases (joining `c^-1 * c'^-1` would let the
///   inner bag's `has_const` flag conflate INDEPENDENT constants -- the flag collapses
///   multiplicity) and literal-`Num` bases (only `0^neg` survives `pow`, and a literal
///   zero never joins a denominator -- the inner product would collapse; cf. the
///   renderer's den rule). DEEP Consts (inside function arguments, `atanh(C*x2)`) are
///   eligible: the inner bag keeps Const-bearing factors verbatim in the opaque bucket,
///   never merged -- the first build excluded them wholesale (`contains_const`) and
///   silently forfeited the join on exactly the sound-refused shapes the projection
///   exists for (measured: whole-bag pairs under `sin(..)` stayed distributed at mu +8
///   over sound's refusal form).
///
/// Termination: the joined node is built RAW (`Ex::Pow`), never through `pow` (whose
/// lossy blanket would immediately re-distribute it); the inner `mul` sees only
/// positive-power members, so no recursion re-enters this decision. The decision
/// itself is guarded by the re-derivability comparison at the return (see there):
/// a join ships only when it strictly beats its own re-settle image, which is what
/// makes the shipped spelling a fixpoint under re-parse.
fn rejoin_reciprocals(settled: Ex, signed: bool, cx: &Cx) -> Ex {
    let Ex::Mul(members) = &settled else {
        return settled;
    };
    let eligible: Vec<usize> = members
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let Ex::Pow(b, e) = f else { return None };
            // A literal-infinity base never joins: `inv(inf)` is a DETERMINED zero
            // (`pow` folds it at construction). Joining it hides that determined
            // fold behind an opaque base -- measured: every net-regressed battery row
            // had `inf` as a direct member of the joined product. DEEP infinities
            // (inside function arguments, row 1414's `acos(inf * x4)`) stay eligible:
            // no fold sees through those either way.
            if matches!(**b, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const(_)) {
                return None;
            }
            let Ex::Num(r) = &**e else { return None };
            (r.is_integer() && r.is_negative()).then_some(i)
        })
        .collect();
    if eligible.len() < 2 {
        return settled;
    }
    let inner_members: Vec<Ex> = eligible
        .iter()
        .map(|&i| {
            let Ex::Pow(b, e) = &members[i] else {
                unreachable!()
            };
            let Ex::Num(r) = &**e else { unreachable!() };
            if *r == Rat::NEG_ONE {
                (**b).clone()
            } else {
                pow((**b).clone(), Ex::Num(r.checked_neg().unwrap()), cx)
            }
        })
        .collect();
    let inner = mul(inner_members.clone(), cx);
    // The joined base's own decision can trade a member's sign into its coefficient (a sum
    // whose other orientation is cheaper once it stands in a product); that sign leaves the base
    // (`joined_base_sign`).
    let (inner, base_sign) = joined_base_sign(inner, cx);
    {
        let Ex::Mul(v) = &inner else {
            // The positive-power bag collapsed below two factors: no joined SHAPE exists.
            return settled;
        };
        let sound_distributes =
            v.iter().all(|f| cx.nz_ae_certified(f)) || v.iter().all(|f| cx.certainly_nonneg(f));
        if sound_distributes {
            return settled;
        }
    }
    let mut rest: Vec<Ex> = members
        .iter()
        .enumerate()
        .filter(|(i, _)| !eligible.contains(i))
        .map(|(_, f)| f.clone())
        .collect();
    if base_sign {
        // The rest of the bag takes the sign, placed by its own decision (`mul()`): the parse of the
        // joined spelling decides the same members with the base's inverted members beside them,
        // which carry no sign of their own.
        rest.push(Ex::Num(Rat::NEG_ONE));
        rest = match mul(rest, cx) {
            Ex::Mul(v) => v,
            Ex::Num(r) if r.is_one() => Vec::new(),
            f => vec![f],
        };
        // A bag that may not take a new sign (`signed`) keeps its working spelling.
        if !signed && gains_a_sign(&rest, &settled) {
            return settled;
        }
    }
    let assemble = |mut cand: Vec<Ex>| match cand.len() {
        1 => cand.pop().unwrap(),
        _ => {
            cand.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            Ex::Mul(cand)
        }
    };
    let joined = Ex::Pow(Box::new(inner.clone()), Box::new(Ex::Num(Rat::NEG_ONE)));
    let mut cand = rest.clone();
    cand.push(joined);
    let mut best = assemble(cand);
    // COEFFICIENT COMPLETION: the bag's rational
    // coefficient can ride INSIDE the joined base, reciprocated -- `0.5 * (x4*S)^-1`
    // IS `(2*x4*S)^-1` -- and when the bag then collapses to the lone joined node, the
    // outer `Mul` unwraps (mu -8, and mu(1/c) vs mu(c) can differ too). Sound's
    // licence-refusal forms keep their coefficient inside the base exactly like this
    // (`inv(-1 * x0 * S)` from `inv(neg(..))` spellings), so without the completion
    // the projection stopped one symbol short of sound's spelling on every such row
    // (23/200k, all gap 8). The funnel stays closed: re-parse distributes the joined
    // base, `pow` folds the literal's reciprocal back into the coefficient, and this
    // decision re-derives.
    //
    // RE-DERIVABILITY GATE: the completion is
    // funnel-stable only if the reciprocated coefficient SURVIVES VERBATIM as a `Num`
    // member of the joined base. `mul()` inside the completion runs the full assembly,
    // and the Const sign fold may CONSUME a negative `cinv` into a Const-bearing Add
    // member -- a strict mu win whose source is the sign fold, not the join, and one
    // the funnel cannot re-derive: the re-parsed (absorbed) state has no Num left in
    // the bag, the completion cannot re-fire, the plain join ties, and
    // tie -> distributed -- so the joined spelling shipped exactly once and the next
    // lossy pass moved (lossy idempotence, 2/200k, fuzz rows 92892 + 115616).
    // When the coefficient survives (positive c always; negative c beside sums the
    // fold does not touch), re-parse re-extracts it through pow's distribution and
    // the completion re-derives -- including sound-aligned `inv(-1 * x0 * S)` joins
    // (blanket refusal of negative c donated a uniform mu gap of 8 on 8/200k rows).
    if let Some(ci) = rest.iter().position(|f| matches!(f, Ex::Num(_))) {
        let Ex::Num(c) = &rest[ci] else {
            unreachable!()
        };
        if let Some(cinv) = c.checked_inv() {
            let inner2 = mul(vec![Ex::Num(cinv.clone()), inner.clone()], cx);
            let survives = matches!(&inner2, Ex::Mul(v)
                if v.iter().any(|f| matches!(f, Ex::Num(n) if *n == cinv)));
            // A NEGATIVE reciprocated coefficient
            // beside a sign-trade site is no longer funnel-stable -- pow()'s
            // odd-negative pre-fold now folds that sign into the site on re-parse
            // (the row-310 pole-orientation cure), so the joined spelling cannot
            // re-derive verbatim (measured: corpus row 33 lossy idempotence). The
            // completion skips exactly the pre-fold's firing condition.
            let funnel_stable = !cinv.is_negative()
                || !matches!(&inner2, Ex::Mul(v)
                    if v.iter().any(|f| is_sign_trade_site(f, cx)));
            if survives && funnel_stable {
                let joined2 = Ex::Pow(Box::new(inner2), Box::new(Ex::Num(Rat::NEG_ONE)));
                let mut cand2: Vec<Ex> = rest
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ci)
                    .map(|(_, f)| f.clone())
                    .collect();
                cand2.push(joined2);
                let candidate2 = assemble(cand2);
                if complexity(&candidate2, cx.view) < complexity(&best, cx.view) {
                    best = candidate2;
                }
            }
        }
    }
    // RE-DERIVABILITY COMPARISON (fuzz rows 303116/472156/603382 and, after the
    // first-cut fix, 59022/194208/308376/593418/713415/790992 -- all introduced or
    // exposed by the sign-placement owner): the join used to be judged against the WORKING state, but the
    // working state is not what a shipped join re-parses to. Parse builds the joined
    // base's inner bag POSITIVELY first, where bare trade sites get re-oriented (the
    // Coeff orbit's argmin, or the Const-side Free flip beside a Const-bearing member)
    // before the blanket re-distributes -- while the division route builds the same
    // value member-by-member with every site HIDDEN inside a `^-1` wrapper (not a
    // carrier: the one-zero pole trilemma), preserving arrival mirrors. Two routes,
    // two working states, and a join priced against one route re-derived into the
    // other: the strict win evaporated into a tie on the second pass and the lossy
    // endpoint moved (lossy idempotence). The gate is the re-derivability principle applied to the
    // WHOLE decision: a join ships only if it strictly beats BOTH the working state
    // (the improvement test, as before) AND its own RE-SETTLE IMAGE -- the state its
    // re-parse provably lands on (inner is `mul`-built == what parse rebuilds, `mul`
    // is idempotent on it, so the image is exact). A shipped join then re-picks
    // itself on every later pass (fixpoint by construction), and a refused join
    // ships the working state, whose rendering re-parses member-by-member through
    // the hidden-site route back to itself.
    let resettle = |cand: &Ex| -> Ex {
        let members: Vec<Ex> = match cand {
            Ex::Mul(v) => v.clone(),
            other => vec![other.clone()],
        };
        let mut parts: Vec<Ex> = Vec::with_capacity(members.len() + 2);
        for m in members {
            match m {
                Ex::Pow(b, e)
                    if matches!(&*e, Ex::Num(r) if *r == Rat::NEG_ONE)
                        && matches!(&*b, Ex::Mul(_)) =>
                {
                    let Ex::Mul(bm) = *b else { unreachable!() };
                    for f in bm {
                        parts.push(pow(f, Ex::Num(Rat::NEG_ONE), cx));
                    }
                }
                other => parts.push(other),
            }
        }
        mul(parts, cx)
    };
    let mu_best = complexity(&best, cx.view);
    let mu_settled = complexity(&settled, cx.view);
    let image = resettle(&best);
    let mu_image = complexity(&image, cx.view);
    if mu_best < mu_settled && mu_best < mu_image {
        best
    } else if base_sign
        && mu_image < mu_settled
        && (signed || !gains_a_sign(std::slice::from_ref(&image), &settled))
    {
        // The base's decision found a cheaper orientation for a member the working state holds
        // under its inverse: the re-settle image keeps it, distributed, and re-derives itself.
        image
    } else {
        settled
    }
}

/// Does `members` (a bag's members, or the bag itself) carry a negative coefficient where the bag
/// `settled` carries none?
fn gains_a_sign(members: &[Ex], settled: &Ex) -> bool {
    let negative = |f: &Ex| match f {
        Ex::Num(r) => r.is_negative(),
        Ex::Mul(v) => v.iter().any(|g| matches!(g, Ex::Num(r) if r.is_negative())),
        _ => false,
    };
    members.iter().any(negative) && !negative(settled)
}

/// The joined base without a negative coefficient of its own, and whether one left it: a sign the
/// joined base's own product placed in its coefficient leaves the base for the rest of the bag
/// (`rejoin_reciprocals`). The base is decided as a product of its own, so a sum the distributed
/// bag held under its inverse can take its cheaper orientation there and hand the sign to the
/// base's coefficient, while the bag keeps the sign it had. Kept in the base, the two signs would
/// print as `(-a)/(-(b (x - u)))`, which the parse reads as `a/(b (x - u))`, so a second call would
/// print another spelling. In the lossy context the projection runs in, the inverse of `-P` is
/// minus the inverse of `P` (they differ only at `P`'s zero), so the sign joins the rest of the bag,
/// whose own decision places it. A bag the parse reads with its sign kept (the answer, or a term of
/// a sum) takes it as it comes; inside a function's argument or a power's base, where the parse
/// moves or drops a product's sign, a bag that would gain a sign keeps its working spelling.
fn joined_base_sign(inner: Ex, cx: &Cx) -> (Ex, bool) {
    if !cx.permits(Permissive::Agreement) {
        return (inner, false);
    }
    let Ex::Mul(mut v) = inner else {
        return (inner, false);
    };
    if !matches!(v.first(), Some(Ex::Num(r)) if r.is_negative()) {
        return (Ex::Mul(v), false);
    }
    let Ex::Num(r) = &v[0] else { unreachable!() };
    match r.checked_neg() {
        Some(p) if p.is_one() => {
            v.remove(0);
        }
        Some(p) => v[0] = Ex::Num(p),
        None => return (Ex::Mul(v), false),
    }
    let base = if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    };
    (base, true)
}

/// The output-boundary rejoin walk (see [`rejoin_reciprocals`]): apply the reciprocal
/// join decision bottom-up to every `Mul` node of the final lossy state. STRUCTURAL
/// rebuild on purpose -- routing through the constructors would re-distribute the joins
/// (`pow`'s lossy blanket). Bags whose members changed are re-sorted under their own
/// comparator so the projected tree spells deterministically.
pub fn rejoin_projection(e: Ex, cx: &Cx) -> Ex {
    rejoin_walk(e, true, cx)
}

/// [`rejoin_projection`] at a node; `signed`: does the parse keep a sign the node's bag takes (the
/// answer itself, a term of a sum, a member of a product), or move or drop it (a function's
/// argument, a power's base or exponent)?
fn rejoin_walk(e: Ex, signed: bool, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let mut terms: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect();
            terms.sort_by(|a, b| add_term_cmp(a, b, cx.view));
            Ex::Add(terms)
        }
        Ex::Mul(v) => {
            let mut m: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect();
            m.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            rejoin_reciprocals(Ex::Mul(m), signed, cx)
        }
        Ex::Pow(b, ex) => Ex::Pow(
            Box::new(rejoin_walk(*b, false, cx)),
            Box::new(rejoin_walk(*ex, false, cx)),
        ),
        // `atan2` reads its arguments' zero signs, and the parse keeps their signs (`atan2_of`).
        Ex::Fun(f, v) if v.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(f, "atan2") => {
            Ex::Fun(f, v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect())
        }
        // A lattice bag keeps its members in the canonical order (`lattice`).
        Ex::Fun(f, v) if lattice_head(f, cx.view) => {
            let mut m: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, false, cx)).collect();
            m.sort_by(|a, b| cmp_ex(a, b, cx.view));
            Ex::Fun(f, m)
        }
        Ex::Fun(f, v) => Ex::Fun(
            f,
            v.into_iter().map(|x| rejoin_walk(x, false, cx)).collect(),
        ),
        leaf => leaf,
    }
}

/// Rebuild `base^exp` (rational or symbolic-with-count) through the power constructor so
/// integer-power distribution and exact folds apply; as a member of the product being built
/// ([`pow_member`]), whose sign placement orients it.
fn rebuild_factor(base: Ex, sym: Option<Ex>, r: Rat, cx: &Cx) -> Ex {
    match sym {
        None => pow_member(base, Ex::Num(r), cx),
        Some(e) => {
            if r.is_one() {
                pow_member(base, e, cx)
            } else {
                let scaled = mul(vec![Ex::Num(r), e], cx);
                pow_member(base, scaled, cx)
            }
        }
    }
}

/// `pow`'s licence to distribute the integer power `e` over a product of `v`: the bag it distributes
/// over (a negative coefficient folded into a sign-trade site first), or `None` where it
/// refuses. The divisor parse and `canon` ask it too (`distributes_power`, `inverted_members`),
/// so a quotient is distributed exactly where `pow` would distribute it.
/// Does `pow` distribute the integer power `e` over the product of `v` (`power_bag`'s licence)?
pub(crate) fn distributes_power(v: &[Ex], e: &Rat, cx: &Cx) -> bool {
    power_bag(v, e, cx).is_some()
}

/// THE PREMISE OF `pow.distribute`: may the integer power `e` distribute over the product of
/// `v`? A non-negative or even power always; an odd negative one where every member is nonzero
/// almost everywhere, a member is infinite, all are certainly `>= 0`, or a member's zero is one
/// whose sign an inverse reads (`inverts_a_signed_zero`).
pub(crate) fn distribution_licensed(v: &[Ex], e: &Rat, cx: &Cx) -> bool {
    !e.is_negative()
        || e.is_even_integer()
        || inverts_a_signed_zero(v, cx)
        || v.iter().any(holds_infinity)
        || v.iter().all(|f| zero_set_null(f, cx))
        || (v.iter().any(|f| holds_odd_root(f, cx))
            && v.iter()
                .all(|f| cx.nz_ae_licensed(f) || zero_set_null_by_form(f, cx)))
        || v.iter().all(|f| cx.certainly_nonneg(f))
}

fn power_bag(v: &[Ex], e: &Rat, cx: &Cx) -> Option<Vec<Ex>> {
    if !Law::PowDistribute.holds_powers(Powers::Product { members: v, n: e }, cx) {
        return None;
    }
    Some(distributed_bag(v, e, cx))
}

/// The licences a distributed reading's bindings must meet: for each product read distributed,
/// its members and power ([`pattern_distributed`]).
pub(crate) type Licences = Vec<(Vec<Ex>, Rat)>;

/// THE DISTRIBUTED READING OF A PATTERN (`pow.distribute` at bind time). A slot may bind a member
/// whose zero set holds a region, so a pattern's product under an odd negative power holds where
/// its members cannot license the distribution (`distribution_licensed`), while its instances
/// distribute wherever their members do: `1/(x1*?0)` stays one member, `1/(x1*x3)` is
/// `x1^-1*x3^-1`. This is `e` with every such product distributed as `pow` distributes a licensed
/// one, and, for each, its members and power: the licence a binding must meet for the reading to
/// match (`AcRule::licences`). `None` where `e` holds no such product.
pub(crate) fn pattern_distributed(e: &Ex, cx: &Cx) -> Option<(Ex, Licences)> {
    fn walk(e: &Ex, cx: &Cx, out: &mut Licences) -> Option<Ex> {
        let kids = |v: &[Ex], out: &mut Licences| -> Option<Vec<Ex>> {
            let mut kids: Option<Vec<Ex>> = None;
            for (i, c) in v.iter().enumerate() {
                if let Some(r) = walk(c, cx, out) {
                    kids.get_or_insert_with(|| v.to_vec())[i] = r;
                }
            }
            kids
        };
        match e {
            Ex::Pow(b, x) => {
                if let (Ex::Mul(v), Ex::Num(n)) = (&**b, &**x) {
                    if n.is_integer()
                        && n.is_negative()
                        && v.iter().any(|f| f.contains_wildcard(cx.view))
                        && !distribution_licensed(v, n, cx)
                    {
                        let members = kids(v, out).unwrap_or_else(|| v.clone());
                        out.push((members.clone(), n.clone()));
                        let raised = distributed_bag(&members, n, cx)
                            .into_iter()
                            .map(|f| pow_member(f, Ex::Num(n.clone()), cx))
                            .collect();
                        return Some(mul(raised, cx));
                    }
                }
                let nb = walk(b, cx, out);
                let nx = walk(x, cx, out);
                (nb.is_some() || nx.is_some()).then(|| {
                    pow(
                        nb.unwrap_or_else(|| (**b).clone()),
                        nx.unwrap_or_else(|| (**x).clone()),
                        cx,
                    )
                })
            }
            Ex::Add(v) => kids(v, out).map(|k| add(k, cx)),
            Ex::Mul(v) => kids(v, out).map(|k| mul(k, cx)),
            Ex::Fun(f, v) => kids(v, out).map(|k| fun(*f, k, cx)),
            _ => None,
        }
    }
    let mut licences = Vec::new();
    walk(e, cx, &mut licences).map(|d| (d, licences))
}

/// The members `pow` raises where it distributes the integer power `e` over the product of `v`
/// ([`power_bag`] without its licence).
fn distributed_bag(v: &[Ex], e: &Rat, cx: &Cx) -> Vec<Ex> {
    // For a NEGATIVE ODD exponent, fold a negative bag coefficient
    // into the first sign-trade site BEFORE distributing (exact at the
    // finite bag level: -c * S == c * (-S)). After distribution the sum
    // sits under a REFUSED odd-negative carrier -- (S)^-n and -((-S))^-n
    // are a.e.-equal but POLE-DIFFERENT, so no owner may reconcile them
    // later; whether the sign had already met the bag decided which
    // rendering the a.e. licence produced (corpus row 310: chain held
    // (1-x4)^-1 while parse of its own serialization built
    // -(x4-1)^-1). The pre-fold is UNCONDITIONAL (not mu-judged):
    // post-distribute there is no legal trade, so entry-independence
    // requires one fixed pre-state.
    let vv: Vec<Ex>;
    let bag: &[Ex] = if e.is_negative() && e.is_odd_integer() && !inverts_a_signed_zero(v, cx) {
        let ci = v
            .iter()
            .position(|f| matches!(f, Ex::Num(r) if r.is_negative()));
        let si = v.iter().position(|f| is_sign_trade_site(f, cx));
        match (ci, si) {
            (Some(ci), Some(si)) => {
                let nc = match &v[ci] {
                    Ex::Num(r) => r.checked_neg().map(Ex::Num),
                    _ => None,
                };
                match (sign_trade_flip(&v[si], cx), nc) {
                    (Some(nf), Some(nc)) => {
                        let mut w = v.to_vec();
                        w[si] = nf;
                        w[ci] = nc;
                        if matches!(&w[ci], Ex::Num(r) if r.is_one()) {
                            w.remove(ci);
                        }
                        vv = w;
                        &vv
                    }
                    _ => v,
                }
            }
            _ => v,
        }
    } else {
        v
    };
    bag.to_vec()
}

/// The canonical power constructor. Folds and gates:
///
/// * `e == 1` -> base. `e == 0` -> 1 (TOTAL under the engine's C99-aligned pow: `pow(x, 0) = 1`
///   for EVERY x including NaN and the infinities).
/// * `base == 1` -> 1 (TOTAL: `pow(1, y) = 1` for every y including NaN).
/// * rational^integer folds exactly (checked; overflow keeps the symbolic form).
///   rational^(p/q) folds exactly when the q-th root is exact (`4^(1/2) -> 2`,
///   `(-8)^(1/3)` is NOT folded: the engine's rational pow is NaN on negative bases with
///   non-integer exponents -- the REAL odd root is the separate `pow1_3` operator).
///   `0^negative` and inexact roots stay symbolic (the rules-pass numeric fold owns IEEE edge
///   parity with the shipped engine).
/// * `(a^r)^s -> a^(r*s)` under the composition licence: sound iff (`r` integer AND (`s` integer
///   OR `r` odd)) OR (`r` non-integer AND `r*s` non-integer) OR the base is certainly
///   non-negative. Counterexamples pinned in tests: `(a^2)^(1/2) = |a| != a`;
///   `(a^(1/2))^2 = a` only on `a >= 0`.
/// * `(a*b)^n` for INTEGER n distributes over the factors (TOTAL as extended-real evaluations;
///   for non-integer exponents `(ab)^(1/2) != a^(1/2) b^(1/2)` on `a, b < 0`).
/// * a power that is a sign-trade site takes the orientation it takes as a product's one factor
///   (`lone_site`).
pub fn pow(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    match pow_member(base, exp, cx) {
        node @ Ex::Pow(..) => lone_site(node, cx),
        other => other,
    }
}

/// The SPELLING `pow(b, e)`, as the parse reads it: [`pow`], except numpy's square root in `f64`.
/// Float64's `b ** 0.5` is `sqrt(b)` (`numeric::aligned_pow`), which keeps a zero's sign
/// (`sqrt(-0) = -0`) where `rootn(b, 2)`, the engine's reading of `b^(1/2)`, gives +0. Where `b`
/// may be a zero with a sign of its own (`sqrt_parts_from_rootn`) the two are different functions
/// there, as an inverse or `atan2` reads them: the written negative zero's root is the written
/// negative zero, and any other such `b` keeps the call as written, a node no exponent arithmetic
/// reads (`1/pow(ceil(x), 1/2)` is -inf on (-1, 0), `1/rootn(ceil(x), 2)` +inf).
pub(crate) fn pow_spelled(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    let pow_tok = cx.view.intern("pow");
    if matches!(&exp, Ex::Num(h) if *h == Rat::new(1, 2).expect("1/2"))
        && cx.view.arity(pow_tok) == Some(2)
        && !Law::PowHalfRoot.holds(&base, cx)
    {
        // `sqrt` keeps a zero, its sign and NaN: the root of the written negative zero, or of a
        // zero product (`mul.zero`), is that zero.
        if is_neg_zero(&base, cx.view) || zero_product(&base, cx) {
            return base;
        }
        return Ex::Fun(pow_tok, vec![base, exp]);
    }
    pow(base, exp, cx)
}

/// [`pow`] for a member of a product, whose sign placement decides the member's orientation with
/// the product's other members (`sign_place`): the same power, a sign-trade site left in the
/// orientation it was built in.
///
/// The laws applied here (`ac::laws::Law`): `pow.refused_power`, `num.zero_pole`, `num.inf_power`,
/// `num.literal_base_inf_exponent`, `num.table_fold`.
pub(crate) fn pow_member(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    // A ZERO PRODUCT UNDER A POSITIVE POWER (`mul.zero`): the power 1/2 is float64's `sqrt`, which
    // keeps a zero, its sign and NaN, so the power is the product itself; any other that is no
    // odd integer is `+0` (or NaN) whatever the zero's sign, so the product drops the factors that
    // only gave it one (`unsigned_zero`).
    let base = match &exp {
        Ex::Num(q) if *q == Rat::new(1, 2).expect("1/2") && zero_product(&base, cx) => {
            return base;
        }
        Ex::Num(q) if q.signum() > 0 && !q.is_odd_integer() => {
            unsigned_zero(&base, cx).unwrap_or(base)
        }
        _ => base,
    };
    // A literal base or exponent beside a non-literal one LANDS here (permissive's ρ,
    // [`land`]), where that one is not ground; a power of two literals folds exactly or stays
    // as written.
    let (base, exp) = if matches!(base, Ex::Num(_)) != matches!(exp, Ex::Num(_))
        && !in_ground([&base, &exp], cx)
    {
        (land_operand(base, cx), land_operand(exp, cx))
    } else {
        (base, exp)
    };
    // In `permissive` a power of two literals is the table's exact value
    // (`table_literal_fold`):
    // `pow(0, -1)` and `pow(0, -2)` are NaN (a division by zero), `pow(1, inf)`, `pow(inf, 0)`
    // and `pow(nan, 0)` NaN, `pow(-inf, 0)` 1 -- before the one-zero arms and the `x^0` and `1^x`
    // identities below, which hold for every finite real.
    if cx.table_values() && table_literal(&base) && table_literal(&exp) {
        if let Some(folded) = table_literal_fold("pow", &[base.clone(), exp.clone()]) {
            return folded;
        }
    }
    if let Ex::Num(e) = &exp {
        if e.is_zero() && Law::PowZero.holds(&base, cx) {
            return Ex::Num(Rat::ONE);
        }
        if e.is_one() && Law::PowOne.holds(&base, cx) {
            return base;
        }
    }
    if matches!(&base, Ex::Num(b) if b.is_one()) && Law::PowBaseOne.holds(&exp, cx) {
        return Ex::Num(Rat::ONE);
    }
    // THE WRITTEN NEGATIVE ZERO under a negative integer power is float64's pole of its sign:
    // `(-0)^-1` is -inf, `(-0)^-2` is +inf (`NEG_ZERO`; `f64` only).
    if cx.f64_semantics() && is_neg_zero(&base, cx.view) {
        if let Ex::Num(k) = &exp {
            if k.is_integer() && k.is_negative() {
                return if k.is_odd_integer() {
                    Ex::NegInf
                } else {
                    Ex::PosInf
                };
            }
        }
    }
    // THE NEGATIVE HALF POWER OF A SIGNED ZERO (`f64` only). Float64 reads `b^(1/2)` the way
    // `rootn(b, 2)` is, +0 at -0 (C's `pow` at every non-integer exponent but numpy's 0.5), and
    // prints `b^(-1/2)` as `inv pow b 1/2`, whose `pow b 0.5` numpy computes as `sqrt(b)`, -0
    // at -0 (`numeric::aligned_pow`): where `b` may be a zero with a sign of its own
    // (`sqrt_parts_from_rootn`), the power is `1/rootn(b, 2)`, which prints as the value it
    // is (+inf at -0). (The SPELLING `pow b 0.5` is numpy's `sqrt`: `pow_spelled`.)
    if let Ex::Num(e) = &exp {
        if *e == Rat::new(-1, 2).expect("-1/2") && !Law::PowHalfRoot.holds(&base, cx) {
            if is_neg_zero(&base, cx.view) {
                return Ex::PosInf;
            }
            let root = fun(cx.view.intern("rootn"), vec![base, Ex::int(2)], cx);
            return pow_member(root, Ex::int(-1), cx);
        }
    }
    // rational^rational exact folds.
    if let (Ex::Num(b), Ex::Num(e)) = (&base, &exp) {
        // DETERMINED POLE: a literal-zero base with a
        // negative literal exponent is exact non-finite arithmetic under the
        // one-zero convention (0^n = +0 for n > 0, so 0^{-n} = 1/+0 = +inf; the
        // same convention the signed-inf licence settled for `inv 0`). Without
        // this arm, exponent collection could BUILD `Pow(0, -2)` (from
        // `inv 0 * inv 0`) -- a state whose serialization `inv pow 0 2` re-reads
        // to `inv 0` because the inner `pow 0 2` folds on the way in: a genuine
        // violation of the canonical-form uniqueness contract, caught by the
        // debug `stable()` assertion (release outputs were value-correct; the
        // contract was not).
        if b.is_zero() && e.is_negative() {
            return Ex::PosInf;
        }
        if e.is_integer() {
            if let Some(r) = b.checked_pow_integer(e) {
                return Ex::Num(r);
            }
        } else if !b.is_negative() {
            // b^(p/q): exact iff the q-th root of b is rational.
            if let Some(root) = b.checked_root_integer(&e.denom()) {
                if let Some(r) = root.checked_pow_integer(&e.numer()) {
                    return Ex::Num(r);
                }
            }
        }
    }
    // DETERMINED RECIPROCAL OF INFINITY (dual of the pole arm): an
    // infinite literal base under a negative literal exponent is the same
    // reciprocal-convention arithmetic as the pole arm above -- inf^q = 1/inf^|q| = 0
    // for every rational q < 0, and (-inf)^{-n} = 1/(+-inf) = 0 for every integer
    // n > 0 (unsigned zero). It is the unique determined case where an infinite
    // ground yields a FINITE value, so the interval classification arm (which folds
    // only non-finite classes) cannot carry it; without it `inv float("inf")`
    // survives as a composite and the mine re-learns 1/inf as data. (-inf)^q with
    // q < 0 NON-integer stays unfolded here: the principal real power of a negative
    // base is NaN territory, which interval classification certifies instead.
    // Infinite EXPONENTS are untouched: b^inf is a limit notion, not arithmetic --
    // that family (exp(-inf) -> 0 etc.) arrives as certified rules by doctrine.
    // In `f64` the zero keeps float64's sign, the dual of the written negative zero's pole above:
    // `(-inf)^-1` is `1/-inf = -0` and `(-inf)^-3` is -0, `(-inf)^-2` is +0, so `1/(x1/-inf)` is
    // `-inf/x1` (a product's zero carries its sign, `mul.zero`). And `+inf` under a positive exponent is `+inf`
    // (`inf^(1/2)` is `sqrt(inf)`), decided here, so a product holding it inverts to the zero of
    // its sign: `1/-(inf^(1/2))` is -0.
    if let Ex::Num(e) = &exp {
        if matches!(base, Ex::PosInf) && e.signum() > 0 {
            return Ex::PosInf;
        }
        if e.is_negative() {
            match &base {
                Ex::PosInf => return Ex::Num(Rat::ZERO),
                Ex::NegInf if e.is_odd_integer() && f64_zero_signs(cx) => {
                    return Ex::Leaf(cx.view.intern(NEG_ZERO))
                }
                Ex::NegInf if e.is_integer() => return Ex::Num(Rat::ZERO),
                _ => {}
            }
        }
    }
    // A NEGATED SITE UNDER A POWER IS THE POWER OF ITS FLIP: `-F` and its flip (`sign_trade_flip`)
    // are one class in its two orientations, one value wherever the flip is a site's other spelling,
    // so `(-F)^s` is `flip(F)^s` for every exponent. A site standing alone may take the negated
    // orientation (`lone_site`); a power over it reads the class as it reads the site spelled
    // without the sign: `(S^3)^-1` is `S^-3` and `(S^3)^(1/2)` is `S^(3/2)` whichever orientation
    // `S^3` stands in. (A negative coefficient beside other members folds into a site where the
    // power distributes, `power_bag`.)
    // An odd negative power inverts the zero of a site that vanishes on a region with the sign
    // its orientation gives it (`region_signed_zero`): `1/-(S^3)` is `-inf` there and
    // `1/flip(S)^3` is `+inf`, so the power distributes instead (`inverts_a_signed_zero`).
    let negated_site = |f: &Ex| match &exp {
        Ex::Num(s) => Law::PowNegatedSite.holds_powers(Powers::Site { site: f, s }, cx),
        _ => true,
    };
    if let Ex::Mul(v) = &base {
        if let [Ex::Num(c), f] | [f, Ex::Num(c)] = v.as_slice() {
            if *c == Rat::NEG_ONE && negated_site(f) {
                if let Some(flipped) = sign_trade_flip(f, cx) {
                    return pow_member(flipped, exp, cx);
                }
            }
        }
    }
    // Integer powers distribute over products -- TOTAL for non-negative and for EVEN
    // integer exponents (at a zero of the product both spellings give 0 resp. +inf; at
    // infinities they agree for every integer). For ODD NEGATIVE exponents the identity
    // (a*b)^n = a^n * b^n FAILS under the one-zero contract wherever the product VANISHES
    // with a negative co-factor: (-1 * (x-|x|))^-1 is +inf on the whole half-line x >= 0
    // (1/0 = +inf) while -1 * (x-|x|)^-1 is -inf there -- a positive-measure
    // infinity-sign change (surfaced by the 1M corpus gate, 2 ppm). The failure set
    // lives entirely on the factors' ZERO SETS (every infinite/NaN configuration agrees
    // on both sides), so the licence is per-factor NONZERO-A.E. (`nz_ae_licensed`:
    // variables and Const qualify -- null hyperplane resp. null {c=0} in c-space --
    // abs-style factors with half-line zero sets are refused by the zero-set
    // certificate; a polynomial or an analytic function of one passes by its form,
    // `zero_set_null_by_form`, which the certificate-free parse reads too), a bag with an
    // infinite member (never zero), or an all-certainly-nonneg bag (at a zero every
    // co-factor is >= 0, so both spellings agree on +inf). Refusal keeps Pow(Mul[..], n):
    // on the refused set the two spellings are genuinely DIFFERENT functions, so
    // canonicalization must not identify them. The licence is `power_bag`.
    if let (Ex::Mul(v), Ex::Num(e)) = (&base, &exp) {
        if e.is_integer() {
            if let Some(bag) = power_bag(v, e, cx) {
                let parts: Vec<Ex> = bag
                    .into_iter()
                    .map(|f| pow_member(f, Ex::Num(e.clone()), cx))
                    .collect();
                return mul(parts, cx);
            }
            // REFUSED: normalize the kept form to the exponent -1 shape so refusal is
            // CONFLUENT. The same object reaches this point structured two ways -- the
            // exponent composed before meeting the bag (pow-of-pow: Pow(Mul[-1,A], -3))
            // or after (parse of the emitted spelling: Pow(Mul[-1, A^3], -1)) -- and a
            // form-dependent refusal oscillates between them across passes (observed:
            // idempotence broke on 3/400 corpus rows). (prod f)^n == (prod f^|n|)^-1 is
            // TOTAL for odd n < 0: x^n == (x^|n|)^-1 pointwise (at 0 both +inf, at
            // +-inf both 0, finite exact) and the inner |n| is a NON-NEGATIVE integer
            // distribution (total, licensed above).
            if e.is_integer() && e.cmp_int(-1) == Ordering::Less {
                let m = e.checked_neg().expect("a negative integer negates");
                let parts: Vec<Ex> = v
                    .iter()
                    .cloned()
                    .map(|f| pow_member(f, Ex::Num(m.clone()), cx))
                    .collect();
                return pow_member(mul(parts, cx), Ex::int(-1), cx);
            }
        }
    }
    // Power-of-power composition under the licence. (x^r)^s -> x^(r*s) is a THEOREM
    // only where both sides agree on the negative half-line; the three sound cases:
    //   * s integer: an integer power of a total function, exact everywhere;
    //   * r odd AND POSITIVE AND r*s NON-integer: on x < 0 the source is NaN (x^r < 0
    //     under a non-integer outer exponent) and the target is NaN too (r*s
    //     non-integer). The non-integer-product conjunct is LOAD-BEARING: with r*s
    //     integer the target is FINITE on the whole negative half-line while the source
    //     is NaN ((x^5)^0.2 -> x would fabricate values). Oddness alone licenses nothing
    //     about the outer principal root.
    //
    //     For a NEGATIVE odd r the base must additionally be unable to reach an infinity
    //     (`never_infinite`), and without that conjunct the step is unsound. The "x^r < 0 on
    //     x < 0" step is false at x = -inf when r is NEGATIVE: there x^r is ZERO (one
    //     zero, contract §9.2 -- an f64 zero's sign is measurement rendering, never a
    //     value), so the source is DEFINED (0^s) while the target x^(r*s) is NaN on a
    //     negative base with a non-integer exponent. That is defined -> undefined, which
    //     soundness forbids at ZERO measure tolerance: the null-set licence covers
    //     undefined -> defined only, "the one asymmetry". Reachable at a REAL valuation,
    //     not merely at the point -inf, because a slot binds an arbitrary SUBTREE and
    //     `neg(inv(x0))` is -inf at x0 = 0 -- the Dirac on a null set that contract v2's
    //     own repair #2 warns about. A LEAF base keeps its fold: §9.1 quantifies variables
    //     over the REALS and §10.8 makes a free constant's infinity a limit, never an
    //     attained value, so only a COMPOUND can reach one. Counterexample:
    //         x1 - rootn(inv(neg(inv(x0))), 4)  ->  x1 - inv(pow(-1/x0, 1/4))
    //         at x0 = 0:  source 1.0,  target nan.
    //     `cert_fin` cannot rescue the negative case: it certifies finiteness ALMOST
    //     EVERYWHERE, and the whole defect lives on the null set.
    //   * a certainly-non-negative base, where every spelling agrees.
    // (r even with r*s non-integer must stay refused: (x^2)^(1/4) = |x|^(1/2) is
    // DEFINED on x < 0 while x^(1/2) is NaN -- the dual failure direction.)
    if let (Some((inner_base, r)), Ex::Num(s)) = (as_rational_power(&base, cx), &exp) {
        {
            let r = &r;
            // A composition does not hide a reader of a held pair's zero
            // (`compose_hides_a_flat_read`): `1/(1/log(exp(x0)))` stays.
            let flat = compose_hides_a_flat_read(inner_base, r, s, cx);
            let licence = !flat
                && Law::PowCompose.holds_powers(
                    Powers::Rational {
                        base: inner_base,
                        r,
                        s,
                    },
                    cx,
                );
            if licence {
                if let Some(p) = r.checked_mul(s) {
                    return pow_member(inner_base.clone(), Ex::Num(p), cx);
                }
            }
        }
    }
    // The SAME composition where the power is spelled as a FUNCTION: `pow(exp a, b)` and
    // `pow(e, b)` collapse into one `exp`. See `compose_e_power` -- the base is positive
    // everywhere, so no licence is needed, and the fold condition is what keeps the arm
    // descending. Placement is load-bearing: it runs BEFORE the unit-fraction -> `rootn`
    // conversion at the bottom of this function, the rewrite that blinded both
    // `pow np.e _0 -> exp _0` and `pow exp (-1) _0 -> exp neg _0` at a unit-fraction
    // exponent (measured before this arm: `pow(exp(-1), 1/2)` shipped as
    // `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is 11,585).
    if let Some(composed) = compose_e_power(&base, &exp, cx) {
        return composed;
    }
    // ROOT/POWER INVERSE under licence: `pow(rootn(b, n), n) -> b`. TOTAL for odd n (the
    // signed root is a bijection on the extended reals: `rootn(-8,3)^3 = -8`); for even n
    // it needs a certainly-non-negative base, because `rootn(b,2)^2` is NaN on `b < 0`
    // where `b` itself is defined -- unlicensed, that would be a positive-measure
    // extension. This restores the licence the pow-of-pow arm used to supply back when
    // even roots were spelled as powers; without it `rootn(abs(x),2)^2` stopped folding
    // to `abs(x)`.
    //
    // The NEGATIVE mirror `pow(rootn(b, n), -n) -> 1/b` carries the same licence by the same
    // argument: reciprocating a total identity is total (`1/nan = nan`, `1/0 = +inf`,
    // `1/+inf = 0`), and the even case needs the same non-negative base. Without it the
    // reduction was ROUTE-DEPENDENT -- `pow(rootn(x,3), -3)` stayed put while its own
    // explicit spelling `inv pow rootn x 3 3` reduced to `inv x`, so the explicit projection
    // was not a fixpoint of the parser.
    if let (Ex::Fun(rop, rargs), Ex::Num(e)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") {
            if let (Ex::Num(idx), true) = (&rargs[1], e.is_integer()) {
                let n = e.abs();
                if n.cmp_int(2) != Ordering::Less
                    && *idx == n
                    && Law::PowRootInverse.holds_powers(
                        Powers::Rational {
                            base: &rargs[0],
                            r: idx,
                            s: e,
                        },
                        cx,
                    )
                {
                    if !e.is_negative() {
                        return rargs[0].clone();
                    }
                    return pow_member(rargs[0].clone(), Ex::int(-1), cx);
                }
            }
        }
    }
    // ODD-ROOT COMPOSITION under licence: `pow(rootn(b, m), s) -> pow(b, s/m)` for an ODD
    // index `m >= 3` and a NON-INTEGER rational `s`.
    //
    // An odd `rootn` is deliberately NOT readable as `b^(1/m)` (`factor_split`), because the
    // signed total root and the principal power are different functions on `b < 0`, so the
    // generic composition licence cannot reach this shape. It is nonetheless TOTAL exactly
    // when `s` is not an integer, and the case split is short: on `b < 0` the left side is a
    // negative real raised to a non-integer power, i.e. NaN, and the right side is `b^(s/m)`
    // whose exponent is also non-integer -- `s = p/q` in lowest terms with `q >= 2` gives
    // `s/m = p/(qm)` with reduced denominator `qm/gcd(p,m) >= q >= 2` -- hence NaN too; on
    // `b > 0` both are `b^(s/m)`; on `b == 0` both are `0^(s/m)`, agreeing at the pole by the
    // sign of the exponent; `+-inf` and NaN follow the same two lines.
    //
    // The INTEGER `s` case is genuinely unsound and stays excluded: `rootn(-8,3)^2 = 4` while
    // `(-8)^(2/3)` is NaN, a positive-measure disagreement.
    if let (Ex::Fun(rop, rargs), Ex::Num(s)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") {
            if let Ex::Num(idx) = &rargs[1] {
                if idx.is_odd_integer()
                    && idx.cmp_int(3) != Ordering::Less
                    && Law::PowOddRootCompose.holds_powers(
                        Powers::Rational {
                            base: &rargs[0],
                            r: idx,
                            s,
                        },
                        cx,
                    )
                {
                    if let Some(t) = idx.checked_inv().and_then(|inv| s.checked_mul(&inv)) {
                        return pow_member(rargs[0].clone(), Ex::Num(t), cx);
                    }
                }
            }
        }
    }
    // A NEGATIVE EVEN UNIT FRACTION over a `rootn` whose index has no known parity (f64 mode,
    // beyond 2^53): `rootn(b, m)^(-1/n) -> 1/rootn(rootn(b, m), n)` for even
    // n. The odd-root composition above needs the parity and refuses, while the printed
    // `inv pow rootn b m 1/n` re-reads through the parity-free even-unit-fraction and
    // root-composition arms; construction takes that route directly. Sound: on R = rootn(b, m)
    // >= 0 both sides are 1/R^(1/n), on R < 0 both are NaN, at R = 0 both are +inf.
    if let (Ex::Fun(rop, rargs), Ex::Num(s)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") && s.is_negative() && !s.is_integer() {
            if let Ex::Num(idx) = &rargs[1] {
                let n = s.denom();
                if idx.is_integer()
                    && idx.parity().is_none()
                    && s.numer() == Rat::NEG_ONE
                    && n.is_even_integer()
                {
                    if let Some(unit) = s.checked_neg() {
                        let root = pow_member(base.clone(), Ex::Num(unit), cx);
                        return pow_member(root, Ex::int(-1), cx);
                    }
                }
            }
        }
    }
    // A LITERAL BASE AT AN INFINITE EXPONENT IS DETERMINED: `b^(+-inf)` is a STEP in |b|,
    // and with `b` a literal the step is decided, so this is exact arithmetic and not a
    // limit notion (the spike-flattener refusal is about a SYMBOLIC base, where the step can
    // be straddled). Judge: CERTIFIED at every case below.
    //
    // The engine already folded the halves that go to +inf (`2^inf`, `(1/2)^(-inf)`) through
    // the determined-reciprocal arm; the halves that go to ZERO were MISSING, and it was the
    // MINER that said so -- the reciprocal-base arm below made `inv pow 2 _0 -> pow 0.5 _0`
    // native, and the re-mine promptly minted eighteen ground rules
    // (`inv pow k float("inf") -> 0`, k in -10..10) to cover what the wildcard rule had been
    // reaching through. An incomplete arm shows up as rules: the same signal that found
    // a missing sign half, a missing write half and the parity arm's abs propagation.
    //
    // |b| = 1 and b = 0 stay OUT: `(-1)^inf` is 1 and `0^inf` is 0, both certified, but
    // neither is reached by the reciprocal arm and neither was asked for here.
    if matches!(&exp, Ex::PosInf | Ex::NegInf) {
        if let Ex::Num(bv) = &base {
            if !bv.is_zero() {
                // |b| against 1, exactly: `Rat` is normalized with a positive denominator,
                // so comparing |num| with den decides it without any division.
                let mag = bv.abs().cmp_int(1);
                let to_zero = matches!(
                    (&exp, mag),
                    (Ex::PosInf, std::cmp::Ordering::Less)
                        | (Ex::NegInf, std::cmp::Ordering::Greater)
                );
                if to_zero {
                    return Ex::Num(Rat::ZERO);
                }
            }
        }
    }
    // A RECIPROCAL BASE AT AN INFINITE EXPONENT MOVES INTO THE EXPONENT'S SIGN:
    // `(1/t)^(+-inf) -> t^(-+inf)`. Judge: CERTIFIED (the general symbolic-base version
    // `pow(1/t, s) -> pow(t, -s)` is only TOLERATED, so this is deliberately the infinite
    // case ALONE, where the pole configurations collapse). The descent is unconditional --
    // the `inv` node disappears and the two infinities cost the same, 32,000 -> 24,000.
    if matches!(&exp, Ex::PosInf | Ex::NegInf) {
        if let Ex::Pow(inner, k) = &base {
            if matches!(&**k, Ex::Num(r) if *r == Rat::NEG_ONE) {
                let flipped = if matches!(&exp, Ex::PosInf) {
                    Ex::NegInf
                } else {
                    Ex::PosInf
                };
                return pow_member((**inner).clone(), flipped, cx);
            }
        }
    }
    // A LITERAL BASE ABSORBS THE EXPONENT'S SIGN: `b^(-t) -> (1/b)^t` for a literal `b`.
    // Judge: CERTIFIED at every literal, including the two the one-zero convention decides
    // (`0^(-t) -> inf^t`, `inf^(-t) -> 0^t`) and negative bases.
    //
    // GATED ON THE FLIP REMOVING A NODE, which is a structural test and NOT a mu guard --
    // the direction is genuinely context-dependent and this is where it turns. Measured:
    //   exponent `Mul[-1, x0]`        26,000 -> 18,585   the Mul wrapper collapses     TAKE
    //   exponent `Mul[-1, sin x0]`    34,000 -> 26,585   same                          TAKE
    //   exponent `Num(-1/2)`          13,585 -> 12,585   sign is free, rootn pays      TAKE
    //   exponent `Num(-1/3)`          14,000 -> 13,585   sign is free                  TAKE
    //   exponent `Mul[-1, x0, x1]`    34,000 -> 34,585   the Mul SURVIVES              REFUSE
    // The reciprocal costs the base exactly one bit (`mu(1/n) = mu(n) + 1`), so the
    // flip pays iff it buys back a whole node. With two or more members the bag survives the
    // flip, nothing is bought, and the arm would ASCEND -- which is why the shipped rules
    // could never be a blanket constructor arm and why this is not one either.
    // The literal's reciprocal, under the one-zero convention the judge priced these against
    // (`1/0 = +inf`). RATIONAL LITERALS ONLY, and both exclusions were paid for:
    //
    //   * `-inf` is REFUSED because the judge KILLs it, bc-positive-measure: `(-inf)^(-t)` is
    //     NaN at every non-integer `t` while `0^t` is 0 there, so the flip would extend
    //     undefined to defined on positive measure. I had written `PosInf | NegInf` from
    //     symmetry WITHOUT asking, and the h045 wall caught it -- the judge-first rule exists
    //     precisely because symmetry is not an argument.
    //   * `+inf` is judge-CERTIFIED but owned by the determined-reciprocal arm above, which
    //     folds `Pow(inf, negative)` to 0 before this arm is reached.
    let literal_reciprocal = |e: &Ex| -> Option<Ex> {
        match e {
            Ex::Num(b) if b.is_zero() => Some(Ex::PosInf),
            Ex::Num(b) => b.checked_inv().map(Ex::Num),
            _ => None,
        }
    };
    if let Some(positive) = cx.carried_negative(&exp) {
        let single_member = match &exp {
            Ex::Num(_) => true,
            Ex::Mul(v) => v.iter().filter(|m| !matches!(m, Ex::Num(_))).count() == 1,
            _ => false,
        };
        if single_member {
            if let Some(r) = literal_reciprocal(&base) {
                return pow_member(r, positive, cx);
            }
        }
    }
    // The `inv` SPELLING of the same fact: `1/(b^t) -> (1/b)^t` at a literal `b`. It arrives
    // as `Pow(Pow(b, t), -1)`, which the pow-of-pow composition above cannot touch -- that
    // arm needs a RATIONAL inner exponent and here `t` is symbolic. No structural gate is
    // needed: the exponent is carried across untouched, so the outer `Pow` node disappears
    // against the base's one bit and the descent is unconditional (26,000 -> 18,585).
    // Judge: CERTIFIED at a literal base (`inv pow 3 _0 -> pow (1/3) _0`), TOLERATED at a
    // symbolic one -- literal only, exactly as above.
    if matches!(&exp, Ex::Num(r) if *r == Rat::NEG_ONE) {
        if let Ex::Pow(ib, ie) = &base {
            if let Some(r) = literal_reciprocal(ib) {
                return pow_member(r, (**ie).clone(), cx);
            }
        }
    }
    // EVEN UNIT-FRACTION EXPONENT -> `rootn`. For an even index the two
    // spellings are the SAME function -- both are the principal root, NaN on negatives --
    // so the orientation is free, and mu now settles it: `rootn x 2` prices 18 against
    // `pow x 1/2` at 19, because a genuine fraction pays for its denominator while the
    // integer index does not. It is also 3 tokens against 7 in the tagged form, and the
    // index stays inside a consumer's integer vocabulary.
    //
    // ODD n must NOT convert, in either direction: `rootn(-8,3) = -2` where `pow(-8,1/3)`
    // is NaN, so those are different functions. Only UNIT fractions convert: `pow(x, 3/2)`
    // is not `rootn(x,2)`, and rewriting it as `pow(rootn(x,2),3)` is a separate question.
    //
    // Runs LAST, so every exact fold, pole, infinity and composition arm above has already
    // had its chance -- `pow(4, 1/2)` still folds to `2` and never reaches here.
    if let Ex::Num(e) = &exp {
        let n = e.denom();
        if e.numer().is_one() && n.is_even_integer() && n.cmp_int(2) != Ordering::Less {
            return fun(cx.view.intern("rootn"), vec![base, Ex::Num(n)], cx);
        }
    }
    // AN EVEN POWER'S BASE: an even integer exponent erases the base's sign, `(-S)^2k = S^2k`
    // exactly, at the poles too (a negative even power is `+inf` at `S = 0` in either orientation,
    // 0 at an infinity), so a mixed-sign sum's orientation is free and both orientations are one
    // value. The base takes the orientation the sum's normal-form choice gives a free site
    // (`flipped_orientation_wins`, `Tie::Lex`: neither spelling distinguished). Const-bearing and
    // absorbing sums keep their absorption owners. Base and exponent are final here, and the flip
    // re-enters `pow` once (the flipped orientation stands, the order being one on the class).
    if let (Ex::Add(ts), Ex::Num(r)) = (&base, &exp) {
        if r.is_even_integer()
            && !r.is_zero()
            && !ts.iter().any(Ex::contains_const)
            && !ts.iter().any(term_absorbs_negation)
            && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
            && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
        {
            if let Some(mut flipped) = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()
            {
                // FILES-BARE gate (same rule as sign_trade_flip): only adopt an
                // orientation primitive_sum itself would file bare.
                if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
                    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                    return pow_member(Ex::Add(flipped), exp, cx);
                }
            }
        }
    }
    // The same orientation for an odd function of a sum under an even exponent: f(-S)^2k =
    // (-f(S))^2k = f(S)^2k exactly, so `tanh(x2 - x1)^2` and `tanh(x1 - x2)^2` are one value,
    // filed with the argument the sum itself would take.
    if let (Ex::Fun(op, args), Ex::Num(r)) = (&base, &exp) {
        let pos = odd_head(cx.view, *op, args).map(|(p, _)| p);
        let odd = pos.is_some();
        let pos = pos.unwrap_or(0);
        if let (true, Ex::Add(ts)) = (odd && r.is_even_integer() && !r.is_zero(), &args[pos]) {
            if !ts.iter().any(Ex::contains_const)
                && !ts.iter().any(term_absorbs_negation)
                && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
                && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
            {
                if let Some(mut flipped) = ts
                    .iter()
                    .map(|t| negate_term(t, cx))
                    .collect::<Option<Vec<Ex>>>()
                {
                    if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
                        flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                        let mut args = args.clone();
                        args[pos] = Ex::Add(flipped);
                        return pow_member(Ex::Fun(*op, args), exp, cx);
                    }
                }
            }
        }
        // And over a negative literal multiple, the positive one: `sin(-2 x1)^2` is
        // `sin(2 x1)^2`, as the class {sin(2 x1), sin(-2 x1)} is written (`collect_opposite_sums`).
        if r.is_even_integer() && !r.is_zero() {
            if let Some(Ex::Fun(fk, fa)) = positive_head(&base, cx.view) {
                return pow_member(fun(fk, fa, cx), exp, cx);
            }
        }
    }
    Ex::Pow(Box::new(base), Box::new(exp))
}

/// ONE ORIENTATION FOR A CLASS, WHEREVER IT STANDS. A sign-trade site (`sign_trade_flip`: an odd
/// power or an odd root of a sum, an odd function of one) spells its class in either
/// orientation, `f(S)` or `-f(-S)`; as a factor of a product the sign placement picks between
/// them over the whole product (`sign_place`: the cheaper, then the sign key, fewest signs and
/// then the outer ones). A site standing alone takes the spelling that placement gives it as the
/// one factor of a product, so `S^3` and `1*S^3` are one answer, and so are `1/S` and `1/(1*S)`:
/// `(x3*(x3 - x1) - x2)^3` is `-(x2 + x3*(x1 - x3))^3`, whose two signs stand further out.
fn lone_site(node: Ex, cx: &Cx) -> Ex {
    // A site standing alone may be the one an inverse reads (`1/S^3`): over a sum that vanishes on
    // a region it keeps the orientation it was written in, which the inverse keeps
    // (`region_signed_zero`, `inverts_a_signed_zero`).
    if region_signed_zero(&node, cx) {
        return node;
    }
    // One scope for the site test and the placement, so the flip is computed once.
    let _memo = FlipScope::open();
    if sign_trade_flip(&node, cx).is_some() {
        sign_place(Rat::ONE, vec![node], cx)
    } else {
        node
    }
}

/// THE PERMISSIVE LITERAL FOLD. An exact rational the constructors produced -- the quotient of two
/// 16-digit decimals, a coefficient cleared into `p/q`, a 30-digit integer -- prices under
/// mu' as the rational codeword alone once its expansion stops terminating, and the
/// emitter must spell it as that fraction, because a spelling denotes the state's EXACT
/// value (`decimal_spelling_wins` chooses among exact codewords only). On the permissive
/// tier the value is allowed to move -- that tier's licence -- so the literal folds to the
/// f64 nearest to it (as that float's shortest decimal) WHEN the exact codewords
/// ([`mu_rat_exact`]) price that spelling strictly cheaper, and every number is priced as it is
/// printed: `2e29/426738538271436458205631863649` (196.8 bits) becomes `0.46867105279529636`
/// (57.1 bits); `1/2`, `15/37` and `4366/8875` stay, their fractions being cheaper than
/// any 17-digit float; a literal that already IS a shortest f64 spelling is its own fold
/// and never moves, so the drawn constants of a training stream are untouched, and the fold
/// is idempotent. It is applied where a literal lands ([`land`]).
///
/// THE FLOAT IT LANDS ON: the value lies between two floats, and permissive promises float
/// precision, not the nearest float, so the literal lands on whichever of the two prints
/// shorter (the cheaper shortest spelling; the nearest on a tie). A literal combined from
/// literals that already landed can sit an ulp from the single rounding of its exact sources,
/// on the float whose shortest spelling carries a digit more (`0.016213652755841255` where the
/// sources give `0.01621365275584125`); the shorter neighbour is that rounding wherever it is
/// one of the two, and no worse a float anywhere. Values whose shortest
/// spelling is no number of the f64 domain refuse (fail-closed);
/// non-finite values
/// never fold.
pub fn lossy_literal(r: &Rat) -> Option<Rat> {
    let (y, other) = r.float_bracket()?;
    // Rust's `{:?}` for f64 is the shortest string that round-trips, which is precisely the
    // spelling whose f64 reading is this value again.
    let snapped = Rat::parse_decimal(&format!("{y:?}"))?;
    if snapped == *r {
        return None;
    }
    let snapped = match other.and_then(|o| Rat::parse_decimal(&format!("{o:?}"))) {
        Some(s) if mu_rat_exact(&s) < mu_rat_exact(&snapped) => s,
        _ => snapped,
    };
    (mu_rat_exact(&snapped) < mu_rat_exact(r)).then_some(snapped)
}

/// ρ, A LITERAL'S ONE ROUNDING: a literal is rounded once, when it becomes part of the
/// expression, not at every operation. Arithmetic BETWEEN
/// literals stays exact -- an all-literal node folds exactly, as in every mode -- and a literal
/// is moved by [`lossy_literal`] at the one place it LANDS as an operand of a composite node:
/// a sum's constant term or a term's coefficient (`add`, `term_join`), a product's coefficient
/// (`finish_product`), a power's literal base or exponent beside a non-literal one (`pow`), a
/// function's argument (`fun`), and the root of a state (the walk, [`land_root`]). So
/// `3*(1/3 + 1e-20)` folds exactly to `1 + 3e-20` and rounds once, to `1`; rounding inside
/// each operation would give `0.9999999999999999`.
///
/// In `strict` a literal lands on float64's own reading of it where that is cheaper
/// ([`strict_literal`]): float64 reads a literal as its nearest double, so writing that double's
/// shortest decimal keeps float64's value bit for bit (`x0*0.1234567890123456789` is
/// `0.12345678901234568*x0`); an integer and a literal no dearer than that decimal stay.
///
/// ρ is idempotent (a moved literal is a shortest float spelling, which neither mode moves
/// again), and never raises a price: it moves a literal only to a strictly cheaper spelling
/// ([`mu_rat`]), so the reduction ordering, termination and every descent argument are
/// unchanged, and every state the walk holds prints the codeword it is priced at. In permissive
/// it moves a value to one of the two floats around it, less than an ulp, inside permissive's
/// licence; it never creates an infinity or a zero (non-finite values refuse, and both floats
/// around a nonzero value are nonzero).
///
/// NO LITERAL LANDS INSIDE A GROUND NODE in permissive ([`in_ground`]): a ground subtree is
/// valued whole by the table and written as one float, rounded once (P3, the engine's ground
/// fold), so `14.81 - cos(c)` with `c` a product of ten literals reads the cosine of the exact
/// product.
///
/// The laws applied here (`ac::laws::Law`): `num.literal_to_float`.
#[inline]
pub fn land(r: Rat, cx: &Cx) -> Rat {
    if !cx.permits(Permissive::LiteralToFloat) {
        return strict_literal(r, cx);
    }
    match lossy_literal(&r) {
        Some(s) => {
            if super::rules::folds_logged() {
                super::rules::log_fold("literal", vec![Ex::Num(r)], Ex::Num(s.clone()));
            }
            s
        }
        None => r,
    }
}

/// Does a literal land among `others` inside a GROUND node: are they all ground (no variable,
/// wildcard or `<constant>`)? Then it does not land ([`land`]): the fully numeric part is valued
/// whole, by the ground evaluator.
fn in_ground<'a>(others: impl IntoIterator<Item = &'a Ex>, cx: &Cx) -> bool {
    others
        .into_iter()
        .all(|e| !e.has_measure_space(cx.view) && !e.contains_wildcard(cx.view))
}

/// [`land`] on a literal operand; any other operand is returned as it is.
#[inline]
fn land_operand(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Num(r) => Ex::Num(land(r, cx)),
        e => e,
    }
}

/// [`land`] at the root of a state: a state that is one literal is that literal landed. And
/// nothing reads the answer's zero: `log(exp(t))` and `acosh(cosh(t))` collapse wherever nothing
/// reads their flat point (`release_flat`; `log(exp(x0))` is `x0` as an answer, while
/// `1/log(exp(x0))` stays), a zero product drops the factors that only gave it its sign
/// (`unsigned_zero`; `0*x0` is 0 as an answer, while `1/(0*x0)` is `inf/x0`), and a sum's sign
/// moves wherever nothing reads its zero (`release_unread`; `-ceil(x0) - floor(x1)` is
/// `-(ceil(x0) + floor(x1))` as an answer, while `1/(-ceil(x0) - floor(x1))` stays).
pub fn land_root(e: Ex, cx: &Cx) -> Ex {
    let e = release_flat(e, cx);
    let e = release_unread(e, cx);
    if let Some(u) = unsigned_zero(&e, cx) {
        return u;
    }
    // ... and a sum of zero products is a zero whose sign nothing reads either.
    if let Ex::Add(v) = &e {
        if v.iter().all(|t| zero_product(t, cx)) {
            let terms: Vec<Ex> = v
                .iter()
                .map(|t| unsigned_zero(t, cx).unwrap_or_else(|| t.clone()))
                .collect();
            let settled = add(terms, cx);
            if settled != e {
                return land_root(settled, cx);
            }
        }
    }
    land_operand(e, cx)
}

/// Is `f` a zero member of a product: the literal zero, or the written negative zero?
fn zero_member(f: &Ex, view: &TokenView) -> bool {
    matches!(f, Ex::Num(r) if r.is_zero()) || is_neg_zero(f, view)
}

/// In `f64`, is `t` a ZERO PRODUCT: a product holding a zero, whose value is `+0`, `-0` or NaN at
/// every point, the zero's sign the sign of its other members (`mul.zero`)?
pub(crate) fn zero_product(t: &Ex, cx: &Cx) -> bool {
    f64_zero_signs(cx) && matches!(t, Ex::Mul(v) if v.iter().any(|f| zero_member(f, cx.view)))
}

/// Is `t` a zero product that may be `-0`: its zero the written negative zero, or a member whose
/// sign is not certainly positive?
fn zero_of_either_sign(t: &Ex, cx: &Cx) -> bool {
    zero_product(t, cx)
        && matches!(t, Ex::Mul(v) if v.iter().any(|f| is_neg_zero(f, cx.view)
            || (!zero_member(f, cx.view) && !cx.certainly_nonneg(f))))
}

/// A zero product whose sign nothing reads, without the factors that only gave it its sign: its
/// members finite almost everywhere go and the zero is the literal zero, built as `mul` builds it
/// (the zero itself where nothing is left; the NaN of `0*log(x0)` stays). `None` where `t` is no
/// zero product, or nothing goes.
pub(crate) fn unsigned_zero(t: &Ex, cx: &Cx) -> Option<Ex> {
    if !zero_product(t, cx) {
        return None;
    }
    let Ex::Mul(v) = t else { return None };
    let carriers: Vec<Ex> = v
        .iter()
        .filter(|f| !zero_member(f, cx.view) && !cx.fin_certified(f))
        .cloned()
        .collect();
    let mut items = vec![Ex::Num(Rat::ZERO)];
    items.extend(carriers);
    let u = mul(items, cx);
    (u != *t).then_some(u)
}

/// A term of a sum that nothing reads the zeros of (`add`'s settled sum), pushed to `out` as the
/// sum takes it: a zero product without the factors that only gave it its sign (`unsigned_zero`),
/// and a zero product over a sum spread over that sum's terms, each taken the same way
/// (`x1 + 0*(2*x0 + sqrt(x0))` is `x1 + 0*sqrt(x0)`). Spread or whole, the zero products are NaN
/// at the same points (`0*inf` and `0*NaN` are NaN, `0*inf + 0*(-inf)` too) and a zero of some
/// sign elsewhere, which the sum does not read. A literal zero is no term.
fn settle_zero_term(t: Ex, out: &mut Vec<Ex>, cx: &Cx) {
    let u = unsigned_zero(&t, cx).unwrap_or(t);
    if let Ex::Mul(v) = &u {
        if let Some((r, inner)) = factored_sum(v) {
            if r.is_zero() {
                let scaled: Option<Vec<Ex>> = inner.iter().map(|s| scale_term(s, &r, cx)).collect();
                if let Some(scaled) = scaled {
                    for s in scaled {
                        settle_zero_term(s, out, cx);
                    }
                    return;
                }
            }
        }
    }
    match u {
        Ex::Num(z) if z.is_zero() => {}
        u => out.push(u),
    }
}

/// A FACTORED SUM: a product of literals and one sum, `r * (a + b)`, as its coefficient (a refused
/// partition's members are one) and the sum's terms.
fn factored_sum(v: &[Ex]) -> Option<(Coef, &[Ex])> {
    match v.split_last() {
        Some((Ex::Add(inner), lits))
            if !lits.is_empty() && lits.iter().all(|f| matches!(f, Ex::Num(_))) =>
        {
            let r = Coef::from_members(lits.iter().map(|f| match f {
                Ex::Num(r) => r,
                _ => unreachable!("literal members"),
            }));
            Some((r, inner.as_slice()))
        }
        _ => None,
    }
}

/// `add.unfactor`'s premise: does a factored sum `r * (a + b)` entering a sum unfactor into
/// `r*a + r*b`? The two agree at every point (a finite rational distributes over extended-real
/// sums) but where the sum's zero is read. In `f64`, float64's `a + b` is `+0` wherever it cancels,
/// and `-0` only where both terms are:
///
/// * a positive `r` keeps every zero's sign: always;
/// * a negative `r` makes `r*(a + b)` `-0` where the sum cancels, while `r*a + r*b` cancels to
///   `+0`: only where the sum holds no sign (`Law::AddUnfactor`; `-2*(ceil(x0) + floor(x1))`
///   stays whole, `-2*(x0 + x1)` unfactors), unless another addend settles the sum (`add`);
/// * a zero gives `0*(a + b)` the sign of `a + b`, which the terms' zeros do not spell
///   (`0*(x0 - x1)` is `-0` wherever `x0 < x1`, `0*x0 + 0*x1` only where both are negative): it
///   stays whole here, and spreads where the sum settles, which reads none of its zeros
///   (`settle_zero_term`).
///
/// `permissive` reads one zero: always.
fn unfactors(r: &Coef, inner: &[Ex], cx: &Cx) -> bool {
    if !f64_zero_signs(cx) || !(r.is_zero() || r.is_negative()) {
        return true;
    }
    !r.is_zero() && Law::AddUnfactor.holds(&Ex::Add(inner.to_vec()), cx)
}

/// STRICT'S NUMBERS, written as the cheapest value strict admits for them: a number the
/// constructors make from literals is either its exact value (the reals' value, which float64's
/// kind keeps: both finite) or the double nearest the same operation on the literals' doubles,
/// computed exactly and rounded once (`rat::rounded_once`; for two literals float64's own value of
/// the operation), written as its shortest decimal (float64 reads it as that double), whichever
/// is cheaper, and the exact value at an equal price.
/// `x0*0.30000000000000004*0.30000000000000004` is
/// `0.09000000000000002*x0` (the exact product has 34 digits), `0.1 + 0.2` is `0.3` (the exact
/// sum is cheaper than float64's `0.30000000000000004`). Float64's value is taken only where it
/// is a normal double, or an exact zero the exact value also is: an overflow or an underflow
/// keeps the exact value. An integer stays exact (`3^40` is `12157665459056928801`): its parity
/// and its exactness are the point of it. The decimal must be a number of the f64 number domain,
/// as a written literal is (`rat::number_domain`): `3.3333333333333334e-301`, whose denominator
/// passes 2^1022, is no number there, and the exact value stays.
pub(crate) fn strict_number(exact: Rat, float: f64) -> Rat {
    if exact.is_integer() || !(float.is_normal() || (float == 0.0 && exact.is_zero())) {
        return exact;
    }
    // The double's shortest decimal is a function of its bits: each thread remembers it, emptying
    // the table at 65,536 entries.
    thread_local! {
        static DECIMALS: std::cell::RefCell<rustc_hash::FxHashMap<u64, Option<Rat>>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    let decimal = DECIMALS.with(|m| m.borrow().get(&float.to_bits()).cloned());
    let decimal = match decimal {
        Some(d) => d,
        None => {
            let d = {
                let _f64 = super::rat::number_domain(true);
                Rat::parse_decimal(&format!("{float:?}"))
            };
            DECIMALS.with(|m| {
                let mut m = m.borrow_mut();
                if m.len() >= 1 << 16 {
                    m.clear();
                }
                m.insert(float.to_bits(), d.clone());
            });
            d
        }
    };
    match decimal {
        Some(d) if mu_rat_exact(&d) < mu_rat_exact(&exact) => d,
        _ => exact,
    }
}

/// The literal partition of a sum (`Rat::partition_sum`), exact in every mode; in `strict`, where
/// the literals combine into one number, that number is written as the cheaper of its exact value
/// and the sum of the literals' doubles, exact and rounded once ([`strict_number`]). Rounding only
/// the combined number keeps every exact cancellation a later fold of the partition reaches.
/// Inside a fully numeric sum (`ground`) the partition stays exact: the part is valued whole.
fn literal_partition_sum(lits: Vec<Rat>, ground: bool, cx: &Cx) -> Vec<Rat> {
    let float = || {
        let ds: Vec<f64> = lits
            .iter()
            .filter(|m| !m.is_zero())
            .map(Rat::to_f64_nearest)
            .collect();
        super::rat::rounded_once('+', &ds).unwrap_or_else(|| ds.iter().fold(0.0, |acc, d| acc + d))
    };
    let combined = lits.iter().filter(|m| !m.is_zero()).count() >= 2;
    let y = if cx.f64_semantics() && cx.fold_f64 && combined && !ground {
        Some(float())
    } else {
        None
    };
    let operands = (combined && !ground && super::rules::folds_logged())
        .then(|| lits.iter().filter(|m| !m.is_zero()).cloned().collect());
    let mut out = Rat::partition_sum(lits);
    if let (Some(y), [one]) = (y, out.as_mut_slice()) {
        *one = strict_number(one.clone(), y);
    }
    log_partition("+", operands, &out);
    out
}

/// Record a literal partition that combined its operands into one number (a [`super::rules::Fold`]
/// of `op`), where a fire log is open.
fn log_partition(op: &'static str, operands: Option<Vec<Rat>>, out: &[Rat]) {
    if let (Some(operands), [one]) = (operands, out) {
        super::rules::log_fold(
            op,
            operands.into_iter().map(Ex::Num).collect(),
            Ex::Num(one.clone()),
        );
    }
}

/// The coefficient partition of a product (`Rat::partition_product`), exact in every mode; in
/// `strict`, where the literals combine into one number, that number is written as the cheaper
/// of its exact value and the product of the literals' doubles, exact and rounded once
/// ([`strict_number`]): `0.30000000000000004*0.30000000000000004` is `0.09000000000000002`,
/// `1/3*1e300*1e-300` is `1/3`, and the reciprocals of `89875517873681764`, `8.854e-12` and
/// `1.380649e-23` make `91019782591670690`, where float64's product taken one rounding at a time
/// gives `91019782591670700`. Inside a fully numeric product (`ground`) the partition stays
/// exact: the part is valued whole.
fn literal_partition_product(nums: Vec<Rat>, ground: bool, cx: &Cx) -> Vec<Rat> {
    let combined = nums.iter().filter(|m| !m.is_one()).count() >= 2;
    let y = if cx.f64_semantics() && cx.fold_f64 && combined && !ground {
        let ds: Vec<f64> = nums
            .iter()
            .filter(|m| !m.is_one())
            .map(Rat::to_f64_nearest)
            .collect();
        Some(
            super::rat::rounded_once('*', &ds)
                .unwrap_or_else(|| ds.iter().fold(1.0, |acc, d| acc * d)),
        )
    } else {
        None
    };
    let operands = (combined && !ground && super::rules::folds_logged())
        .then(|| nums.iter().filter(|m| !m.is_one()).cloned().collect());
    let mut out = Rat::partition_product(nums);
    if let (Some(y), [one]) = (y, out.as_mut_slice()) {
        *one = strict_number(one.clone(), y);
    }
    log_partition("*", operands, &out);
    out
}

/// A WRITTEN LITERAL in `strict`: float64 reads it as its nearest double, so a decimal of more
/// digits than that double needs is written as the double's shortest decimal
/// (`0.1234567890123456789` is `0.12345678901234568`); an integer keeps its exact spelling
/// (its parity and its exact arithmetic are the point of it), and so does a literal whose own
/// spelling is no dearer. In the simplify walk only (`Cx::fold_f64`): a conversion, a pattern
/// and the measure's canon keep every number as written.
pub(crate) fn strict_literal(r: Rat, cx: &Cx) -> Rat {
    if !cx.f64_semantics() || !cx.fold_f64 || r.is_integer() {
        return r;
    }
    let y = r.to_f64_nearest();
    let written = strict_number(r.clone(), y);
    if written != r && super::rules::folds_logged() {
        super::rules::log_fold("literal", vec![Ex::Num(r)], Ex::Num(written.clone()));
    }
    written
}

/// `atan2(y, x)` built from the product members of its arguments, as float64 reads them: it
/// reads the sign of a zero in both (`atan2(-0, -1) = -pi`, `atan2(+0, -0) = pi`). In the
/// float64 modes a negative coefficient of `y` leaves the call (`atan2(-S, x) = -atan2(S, x)`,
/// exact there, and not in `permissive`, whose one zero gives `atan2(0, -1) = pi`), and each
/// argument keeps its members apart (`mul_undecided`), so neither a sum that vanishes on a
/// region nor a written zero changes the sign of its zero by a product's orientation decision.
/// Every construction of an `atan2` node goes through here: the parse, the canonical rebuild (a
/// rule's right-hand side after substitution) and the pass's rebuild.
pub(crate) fn atan2_of(op: Tok, mut ym: Vec<Ex>, xm: Vec<Ex>, cx: &Cx) -> Ex {
    let negative = |m: &Ex| matches!(m, Ex::Num(r) if r.is_negative());
    if !cx.ieee_zeros() {
        return fun(op, vec![mul(ym, cx), mul(xm, cx)], cx);
    }
    // No sign moves between an argument's members either way: a decided product may trade its
    // coefficient's sign with a sum's orientation, which moves the sign of the sum's zero.
    let build = |m: Vec<Ex>| mul_undecided(m, cx);
    if ym.iter().filter(|m| negative(m)).count() % 2 == 1 {
        if let Some(pos) = ym.iter().position(negative) {
            if let Ex::Num(r) = &ym[pos] {
                if let Some(n) = r.checked_neg() {
                    ym[pos] = Ex::Num(n);
                    let inner = fun(op, vec![build(ym), build(xm)], cx);
                    return mul(vec![Ex::Num(Rat::NEG_ONE), inner], cx);
                }
            }
        }
    }
    fun(op, vec![build(ym), build(xm)], cx)
}

/// Does `e`, an argument of `atan2` or a product member of one, VANISH WITH A SIGN OF ITS OWN in
/// `f64`: does one of its product members spell a zero (a literal zero or the written `-0.0`) or
/// vanish on a region with the sign its form gives it (`region_signed_zero`, the bar an inverse
/// keeps, `inverts_a_signed_zero`)? `atan2` reads that sign (`atan2(+0, -1) = pi`,
/// `atan2(-0, -1) = -pi`), so a search move inside an argument is taken only where neither the
/// part it rewrites nor the part it writes does (`search::site_list`): elsewhere the part's zero
/// set is null, and a move that keeps the value almost everywhere keeps its sign there. `false`
/// outside `f64`: `permissive` reads a value up to the sign of a zero.
pub(crate) fn vanishes_with_a_sign(e: &Ex, cx: &Cx) -> bool {
    let signed = |m: &Ex| {
        matches!(m, Ex::Num(r) if r.is_zero())
            || is_neg_zero(m, cx.view)
            || region_signed_zero(m, cx)
    };
    cx.f64_semantics()
        && match e {
            Ex::Mul(v) => v.iter().any(signed),
            other => signed(other),
        }
}

/// The product members of `e` (itself where it is no product).
pub(crate) fn product_members(e: Ex) -> Vec<Ex> {
    match e {
        Ex::Mul(v) => v,
        other => vec![other],
    }
}

/// The float64 value of a GROUND factor (no variable, `<constant>` or wildcard) as the deployed
/// evaluator computes it, the sign of a zero included: its literal-token spelling, the one the
/// ground fold evaluates (`convert::to_prefix_literal_tokens`), through the evaluator's kernels
/// (`numeric::apply_op`). `None` for anything else, or a head no kernel evaluates.
fn ground_value(e: &Ex, cx: &Cx) -> Option<f64> {
    if e.has_measure_space(cx.view) || e.contains_wildcard(cx.view) {
        return None;
    }
    let view = cx.view;
    let tokens = super::convert::to_prefix_literal_tokens(e, &Cx::bare(view));
    fn node(tokens: &[Tok], at: &mut usize, view: &TokenView) -> Option<f64> {
        let t = *tokens.get(*at)?;
        *at += 1;
        let s = view.resolve_owned(t);
        match crate::operators::core_arity(&s).or_else(|| view.arity(t)) {
            Some(n) if n <= 2 => {
                let mut args = [0.0f64; 2];
                for a in args.iter_mut().take(usize::from(n)) {
                    *a = node(tokens, at, view)?;
                }
                crate::numeric::apply_op(&s, &args[..usize::from(n)])
            }
            Some(_) => None,
            None => crate::numeric::leaf_value(&s),
        }
    }
    let mut at = 0;
    let v = node(&tokens, &mut at, view)?;
    (at == tokens.len()).then_some(v)
}

/// The float64 value of a written product that is zero, in `f64`: the product of the signs of its
/// factors (`-1*0`, `0*(-2)` and `-(0)` are -0, `-1*-0.0` is +0), which the exact coefficient,
/// one zero, cannot hold, so `-0.0` (`NEG_ZERO`) where an odd number of the factors are negative
/// or the written negative zero, else `0`. A factor that is no number but ground gives the sign
/// of its float64 value (`ground_value`): `0*sin(-1)` is -0 and `sinh(-0.0)*0` is -0, as float64
/// computes them, where the zero would drop the factor and keep +0; this is the ground fold's
/// reading (`Cx::fold_f64`), so a rule file read in the canon it was certified in (`RuleCanon`),
/// which does not fold, keeps its rows. `None` outside `f64` and for
/// any other product (a factor that is neither a number nor ground and finite, or no zero
/// literal). The parse builds a written product through it; a zero the search derives keeps the
/// one zero (the region rule).
///
/// The laws applied here (`ac::laws::Law`): `num.signed_zero`.
pub(crate) fn signed_zero_product(items: &[Ex], cx: &Cx) -> Option<Ex> {
    if !cx.f64_semantics() {
        return None;
    }
    let mut zero = false;
    let mut negative = false;
    let mut ground: Vec<&Ex> = Vec::new();
    for f in items {
        match f {
            Ex::Num(r) => {
                zero |= r.is_zero();
                negative ^= r.is_negative();
            }
            f if is_neg_zero(f, cx.view) => {
                zero = true;
                negative = !negative;
            }
            f if cx.fold_f64 && !f.has_measure_space(cx.view) && !f.contains_wildcard(cx.view) => {
                ground.push(f)
            }
            _ => return None,
        }
    }
    if !zero {
        return None;
    }
    for f in ground {
        let v = ground_value(f, cx)?;
        if !v.is_finite() {
            return None;
        }
        negative ^= v.is_sign_negative();
    }
    zero.then(|| {
        if negative {
            Ex::Leaf(cx.view.intern(NEG_ZERO))
        } else {
            Ex::Num(Rat::ZERO)
        }
    })
}

/// `floor` (or `ceil`) of a rational, exactly; `None` when the result leaves `Rat`.
fn rat_floor(r: &Rat, ceil: bool) -> Option<Rat> {
    use num_integer::Integer;
    if r.is_integer() {
        return Some(r.clone());
    }
    let (p, q) = r.big_parts();
    let f = p.div_floor(&q);
    Rat::from_big(if ceil { f + 1 } else { f }, BigInt::from(1))
}

/// PERMISSIVE'S LITERAL ARITHMETIC IS THE TABLE'S: the value of the operation `op` at literal
/// arguments (rationals, the infinities, NaN) as THE TABLE reads it in the reals' values
/// (`table::apply`), where it is exact -- a rational, an infinity, or NaN where the limit rule
/// gives no value:
/// `pow(0, -1)`, `pow(0, -2)`, `rootn(0, -2)` and `mod(1, 0)` (a division by zero),
/// `pow(1, inf)`, `pow(inf, 0)`, `pow(nan, 0)` and `mod(0, inf)` are NaN, `pow(-inf, 0)` 1,
/// `exp(-inf)` 0, `tanh(inf)` 1, `acosh(1)` 0. `None` for an approximation (a transcendental
/// value stays symbolic) or an operator the table does not know.
///
/// The laws applied here (`ac::laws::Law`): `num.table_fold`.
fn table_literal_fold(op: &str, args: &[Ex]) -> Option<Ex> {
    use crate::table::{Exact, Val};
    let vals: Option<Vec<Val>> = args
        .iter()
        .map(|a| match a {
            Ex::Num(r) => Some(Val::Exact(Exact::Rat(r.clone()))),
            Ex::PosInf => Some(Val::Exact(Exact::PosInf)),
            Ex::NegInf => Some(Val::Exact(Exact::NegInf)),
            Ex::NaN => Some(Val::Exact(Exact::Undef)),
            _ => None,
        })
        .collect();
    let mut ctx = crate::table::Ctx::new(50)?;
    match crate::table::apply(op, &vals?, &mut ctx)? {
        Val::Exact(Exact::Rat(r)) => Some(Ex::Num(r)),
        Val::Exact(Exact::PosInf) => Some(Ex::PosInf),
        Val::Exact(Exact::NegInf) => Some(Ex::NegInf),
        Val::Exact(Exact::Undef) => Some(Ex::NaN),
        _ => None,
    }
}

/// Is `e` a literal the table reads exactly: a rational, an infinity or NaN?
fn table_literal(e: &Ex) -> bool {
    matches!(e, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::NaN)
}

/// Is `op` a LATTICE head, `max` or `min` (a bag of members, [`lattice`])?
pub(crate) fn lattice_head(op: Tok, view: &TokenView) -> bool {
    view.tok_is(op, "max") || view.tok_is(op, "min")
}

/// A lattice member read as a number in the order `max` and `min` read: a literal, or in `f64`
/// the written negative zero (`NEG_ZERO`), which orders below `+0`: `(value, negative zero)`.
/// `None` for anything else.
fn lattice_number(e: &Ex, cx: &Cx) -> Option<(Rat, bool)> {
    match e {
        Ex::Num(r) => Some((r.clone(), false)),
        e if cx.f64_semantics() && is_neg_zero(e, cx.view) => Some((Rat::ZERO, true)),
        _ => None,
    }
}

/// `a` below `b` in the order of IEEE 754-2019's `maximum` and `minimum`: by value, and `-0`
/// below `+0` (both as [`lattice_number`] reads them).
fn lattice_number_lt(a: &(Rat, bool), b: &(Rat, bool)) -> bool {
    match a.0.cmp_exact(&b.0) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.1 && !b.1,
    }
}

/// A certified rational enclosure `(lo, hi)`, `lo < v < hi`, of `pi` or `e` (20 decimals each,
/// so the two never overlap, and each is irrational, so it ties with no rational); `None` for
/// anything else.
fn lattice_enclosure(e: &Ex) -> Option<(Rat, Rat)> {
    let (lo, hi) = match e {
        Ex::Pi => ("3.14159265358979323846", "3.14159265358979323847"),
        Ex::E => ("2.71828182845904523536", "2.71828182845904523537"),
        _ => return None,
    };
    let _exact = super::rat::number_domain(false);
    Some((Rat::parse_decimal(lo)?, Rat::parse_decimal(hi)?))
}

/// The certified enclosure of a GROUND lattice member outside `f64` ([`Cx::cert_encl`]): a
/// composite whose value is certainly a finite real, as an exact rational box `(lo, hi)`,
/// `lo <= v <= hi`; `None` in `f64` (whose ground fold reads such a member as its float), for
/// a leaf, and wherever the certificate is absent or undecided.
fn lattice_ground_enclosure(e: &Ex, cx: &Cx) -> Option<(Rat, Rat)> {
    if cx.f64_semantics() || !matches!(e, Ex::Add(_) | Ex::Mul(_) | Ex::Fun(..) | Ex::Pow(..)) {
        return None;
    }
    let (lo, hi) = (cx.cert_encl?)(e)?;
    let _exact = super::rat::number_domain(false);
    Some((Rat::from_f64_exact(lo)?, Rat::from_f64_exact(hi)?))
}

/// Is `e` a NUMBER of a lattice bag: a literal, the written negative zero in `f64`
/// ([`lattice_number`]), and outside `f64` `pi`, `e` and every ground composite with a certified
/// enclosure ([`lattice_ground_enclosure`]), which `f64`'s ground fold reads as their floats
/// instead?
pub(crate) fn lattice_numeric(e: &Ex, cx: &Cx) -> bool {
    lattice_number(e, cx).is_some()
        || (!cx.f64_semantics()
            && (matches!(e, Ex::Pi | Ex::E) || lattice_ground_enclosure(e, cx).is_some()))
}

/// The CERTAIN order of two lattice numbers ([`lattice_numeric`]): exact between literals (`-0`
/// below `+0`), and through certified enclosures for `pi`, `e` and the ground composites
/// (`max(3, pi)` is `pi`, `min(e, pi)` is `e`, `max(sin 1, 1/2)` is `sin 1`); `None` where it is
/// not decided (two enclosures that overlap, a literal inside an enclosure, or a member that is
/// no number). Nothing breaks a tie the enclosures leave open: the bag keeps both members.
fn lattice_cmp(a: &Ex, b: &Ex, cx: &Cx) -> Option<Ordering> {
    if let (Some(x), Some(y)) = (lattice_number(a, cx), lattice_number(b, cx)) {
        return Some(if lattice_number_lt(&x, &y) {
            Ordering::Less
        } else if lattice_number_lt(&y, &x) {
            Ordering::Greater
        } else {
            Ordering::Equal
        });
    }
    if cx.f64_semantics() || !lattice_numeric(a, cx) || !lattice_numeric(b, cx) {
        return None;
    }
    if a == b {
        return Some(Ordering::Equal);
    }
    let bounds = |e: &Ex| match e {
        Ex::Num(r) => Some((r.clone(), r.clone())),
        Ex::Pi | Ex::E => lattice_enclosure(e),
        other => lattice_ground_enclosure(other, cx),
    };
    let ((alo, ahi), (blo, bhi)) = (bounds(a)?, bounds(b)?);
    if ahi.cmp_exact(&blo) == Ordering::Less {
        Some(Ordering::Less)
    } else if alo.cmp_exact(&bhi) == Ordering::Greater {
        Some(Ordering::Greater)
    } else {
        None
    }
}

/// `MAX` AND `MIN` ARE LATTICE BAGS, as `+` and `*` are bags: `max(a, max(b, c))`, in any
/// nesting and argument order, is ONE node, `Fun(max, [a, b, c])`, its members flattened,
/// collected and in the canonical order. Every law below holds bit for bit in float64, whose
/// kernels are IEEE 754-2019's `maximum` and `minimum` (NaN-propagating, `-0` below `+0`, both
/// commutative and associative: `max(-0, +0) = +0` in either order), and over the reals, up to
/// the payload of a NaN: like `np.maximum`, the kernels return their FIRST NaN operand
/// (`max(nan, -nan)` keeps the first one's sign bit), which the engine's one NaN does not read.
///
/// * FLATTENING: a member that is itself a bag of the same head gives its members.
/// * NaN: a NaN member makes the bag NaN.
/// * ABSORPTION by the dual bag: a member `min(N)` of a `max` bag (dually `max(N)` of a `min`
///   bag) drops when some `x` in `N` is at most some other member `y` -- `x` is `y`, two numbers
///   in order, `x` is the bottom `-inf`, or `y` is the top `+inf` -- and every other member of
///   `N` is no NaN almost everywhere: `max(a, min(a, b)) = a`, while a NaN `b` makes
///   `min(a, b)` NaN and the bag with it. Every member is read as it stands before the drops
///   below, so the bag does not depend on its nesting: `max(3, max(2, min(2, x)))` and
///   `max(min(2, x), max(2, 3))` are both `3`.
/// * NUMBERS: the literal members fold to their extreme, `-0` below `+0` (in `f64` the written
///   `-0.0`): `max(-0.0, 0) = 0`, `min(-0.0, 0) = -0.0`. Outside `f64`, `pi`, `e` and every
///   ground composite are numbers too, ordered by certified enclosures (`max(3, pi) = pi`,
///   `min(e, pi, 4) = e`, `max(sin 1, 1/2) = sin 1`); an order the enclosures leave open keeps
///   both members. In `f64` the ground fold reads them as their floats.
/// * The ABSORBER takes every member that is no NaN almost everywhere (`max(+inf, a) = +inf`,
///   while `max(+inf, nan) = nan`), and the IDENTITY drops (`max(-inf, a) = a`). "No NaN" is a
///   number, an infinity, or the finite-a.e. licence (`Cx::fin_licensed`).
/// * IDEMPOTENCE: equal members collapse (`max(a, a) = a`), except where they hold a
///   `<constant>` (two `<constant>`s are independent constants).
///
/// A bag left with one member is that member.
///
/// The laws applied here (`ac::laws::Law`): `lattice.bag`, `lattice.absorb`,
/// `num.lattice_numbers`.
pub(crate) fn lattice(op: Tok, args: Vec<Ex>, cx: &Cx) -> Ex {
    let view = cx.view;
    let is_max = view.tok_is(op, "max");
    let dual = view.intern(if is_max { "min" } else { "max" });
    let (top, bottom) = if is_max {
        (Ex::PosInf, Ex::NegInf)
    } else {
        (Ex::NegInf, Ex::PosInf)
    };
    let mut members: Vec<Ex> = Vec::with_capacity(args.len());
    for a in args {
        match a {
            Ex::Fun(f, v) if f == op => members.extend(v),
            other => members.push(other),
        }
    }
    if members.iter().any(|m| matches!(m, Ex::NaN)) {
        return Ex::NaN;
    }
    // The members a law drops are no NaN almost everywhere (`lattice.absorb`).
    let no_nan = |e: &Ex| Law::LatticeAbsorb.holds(e, cx);
    // `x` certainly at most `y` in a `max` bag (at least, in a `min` bag).
    let below = |x: &Ex, y: &Ex| -> bool {
        if *x == bottom {
            return true;
        }
        if *y == top {
            return no_nan(x);
        }
        if x == y {
            return !x.contains_const();
        }
        matches!(
            (lattice_cmp(x, y, cx), is_max),
            (Some(Ordering::Less | Ordering::Equal), true)
                | (Some(Ordering::Greater | Ordering::Equal), false)
        )
    };
    let absorbed: Vec<bool> = members
        .iter()
        .enumerate()
        .map(|(i, m)| match m {
            Ex::Fun(f, inner) if *f == dual => inner.iter().enumerate().any(|(k, x)| {
                members
                    .iter()
                    .enumerate()
                    .any(|(j, y)| j != i && below(x, y))
                    && inner.iter().enumerate().all(|(l, z)| l == k || no_nan(z))
            }),
            _ => false,
        })
        .collect();
    if absorbed.iter().any(|&a| a) {
        members = members
            .into_iter()
            .zip(absorbed)
            .filter(|(_, a)| !a)
            .map(|(m, _)| m)
            .collect();
    }
    // The numbers fold to their extreme: a number another number certainly passes drops (an
    // equal one is a duplicate, collapsed below).
    let beaten = if is_max {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    let dropped: Vec<bool> = members
        .iter()
        .map(|m| {
            lattice_numeric(m, cx)
                && members
                    .iter()
                    .any(|n| lattice_cmp(m, n, cx) == Some(beaten))
        })
        .collect();
    if dropped.iter().any(|&d| d) {
        members = members
            .into_iter()
            .zip(dropped)
            .filter(|(_, d)| !d)
            .map(|(m, _)| m)
            .collect();
    }
    // The absorber takes every member that is no NaN, and the identity drops.
    if members.contains(&top) {
        members.retain(|m| *m == top || !no_nan(m));
    }
    if members.iter().any(|m| *m != bottom) {
        members.retain(|m| *m != bottom);
    }
    members.sort_by(|a, b| cmp_ex(a, b, view));
    members.dedup_by(|a, b| a == b && !a.contains_const());
    if members.len() == 1 {
        return members.pop().expect("one member");
    }
    Ex::Fun(op, members)
}

/// The argument an EVEN function reads in place of `a`, or `None` when `a` is already it: the
/// sign-blind representative (`sign_blind_rep`), or else the other orientation of a sum where it
/// wins.
///
/// Beyond the carried sign `sign_blind_rep` drops, an even function erases its argument's
/// orientation (`cos(x - y)` and `cos(y - x)` are one value), which a mixed-sign sum carries in no
/// top-level sign. As an even power's base, the argument takes the orientation the sum's
/// normal-form choice gives a free site (`flipped_orientation_wins`, `Tie::Lex`); Const-bearing and
/// absorbing sums keep their absorption owners, and the flip re-enters `fun` once.
fn even_argument(a: &Ex, cx: &Cx) -> Option<Ex> {
    if let Some(sign_free) = cx.sign_blind_rep(a) {
        return Some(sign_free);
    }
    let Ex::Add(ts) = a else { return None };
    if ts.iter().any(Ex::contains_const)
        || ts.iter().any(term_absorbs_negation)
        || ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
        || !flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
    {
        return None;
    }
    let mut flipped = ts
        .iter()
        .map(|t| negate_term(t, cx))
        .collect::<Option<Vec<Ex>>>()?;
    // FILES-BARE gate (same rule as sign_trade_flip).
    if flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
        return None;
    }
    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    Some(Ex::Add(flipped))
}

/// The argument an even function's arm moves `a` to (`even_argument`), where it may move: a
/// strictly cheaper argument, or one the arm does not move again (the class's spelling at an equal
/// price: a mirror orientation, an infinity's sign). So the arms that rebuild the call on the moved
/// argument (`fun`'s even arm, `besselj`'s even orders) re-enter `fun` on an argument whose price
/// falls at every re-entry but the last, and the price is well founded (`rules::ordered_below`):
/// they terminate. `None` where `a` stays as it is.
fn even_move(a: &Ex, cx: &Cx) -> Option<Ex> {
    let moved = even_argument(a, cx)?;
    (complexity(&moved, cx.view) < complexity(a, cx.view) || even_argument(&moved, cx).is_none())
        .then_some(moved)
}

/// `besselj(+-inf, x)` at a literal `x`: 0 at `x >= 0`, NaN below (scipy's cells). And
/// `besselj(nu, 0)` at a literal order, exactly: `J_0(0) = 1`, `J_nu(0) = 0` for `nu > 0` and for
/// a negative integer `nu`, and at a negative non-integer order the pole's one-sided limit (the
/// order's domain is `x >= 0`), the infinity of the sign of `1/Gamma(nu + 1)`:
/// `J_{-1/2}(0) = inf`, `J_{-3/2}(0) = -inf` (scipy's cells). In `f64` the value is the
/// deployed kernel's, whose zero carries a sign of its own at some orders (`J_{1.5}(0)` and
/// `J_{-1}(0)` are `-0.0`, `J_{2.5}(0)` is `0.0`), which an inverse reads as an infinity's sign.
///
/// The laws applied here (`ac::laws::Law`): `num.besselj_literals`.
fn besselj_of_literals_at_zero(args: &[Ex], cx: &Cx) -> Option<Ex> {
    // An infinite order (scipy's cells): 0 at a literal x >= 0, NaN below.
    if let [Ex::PosInf | Ex::NegInf, Ex::Num(x)] = args {
        return Some(if x.is_negative() {
            Ex::NaN
        } else {
            Ex::Num(Rat::ZERO)
        });
    }
    let [Ex::Num(nu), Ex::Num(x)] = args else {
        return None;
    };
    if !x.is_zero() {
        return None;
    }
    if cx.f64_semantics() {
        let y = crate::numeric::kernels::besselj(nu.to_f64(), 0.0);
        return if y == 1.0 {
            Some(Ex::Num(Rat::ONE))
        } else if y == 0.0 && y.is_sign_negative() {
            Some(Ex::Leaf(cx.view.intern(NEG_ZERO)))
        } else if y == 0.0 {
            Some(Ex::Num(Rat::ZERO))
        } else if y == f64::INFINITY {
            Some(Ex::PosInf)
        } else if y == f64::NEG_INFINITY {
            Some(Ex::NegInf)
        } else {
            None
        };
    }
    if nu.is_zero() {
        return Some(Ex::Num(Rat::ONE));
    }
    if !nu.is_negative() {
        return Some(Ex::Num(Rat::ZERO));
    }
    if nu.is_integer() {
        return Some(Ex::Num(Rat::ZERO));
    }
    // Gamma(z) is positive for z > 0 and has the sign (-1)^ceil(-z) at a negative non-integer z.
    let z = nu.checked_add(&Rat::ONE)?;
    let positive = !z.is_negative() || !rat_floor(&z.checked_neg()?, true)?.is_odd_integer();
    Some(if positive { Ex::PosInf } else { Ex::NegInf })
}

/// `besselj(n, t)` at an INTEGER literal order `n`: two identities of `J_n` that hold at every
/// value, NaN and the sign of a zero included -- in float64 too, where the kernel reads both
/// (`numeric::besselj`):
///
///   * the reflection `J_{-n}(t) = (-1)^n J_n(t)`: the canonical order is non-negative
///     (`besselj(-3, x)` is `-besselj(3, x)`);
///   * the parity `J_n(-t) = (-1)^n J_n(t)`: an even order is an even function of `t`, so it reads
///     the argument an even function reads (`even_argument`: `besselj(2, -x)` and
///     `besselj(2, |x|)` are `besselj(2, x)`, `besselj(0, y - x)` is `besselj(0, x - y)`); an odd
///     order is an odd head (`odd_head`), so its sign and its sum's orientation are the class
///     owners', as an odd function's are (`besselj(1, -2 x)` is `-besselj(1, 2 x)`,
///     `-besselj(1, x0 - x1)` is `besselj(1, x1 - x0)`).
///
/// A sign the reflection moves out of a GROUND argument under an odd order stays where it is in
/// `f64` (the value is the ground fold's, from the kernel, and a derived zero would lose the sign
/// the kernel writes).
///
/// The laws applied here (`ac::laws::Law`): `fun.besselj_reflection`.
fn besselj_integer_order(op: Tok, args: &[Ex], cx: &Cx) -> Option<Ex> {
    let Ex::Num(n) = &args[0] else { return None };
    if !n.is_integer() {
        return None;
    }
    let ground = !args[1].has_measure_space(cx.view) && !args[1].contains_wildcard(cx.view);
    let negated = |e: Ex| mul(vec![Ex::Num(Rat::NEG_ONE), e], cx);
    if n.is_negative() {
        if n.is_odd_integer() && ground && cx.f64_semantics() {
            return None;
        }
        let j = fun(op, vec![Ex::Num(n.checked_neg()?), args[1].clone()], cx);
        return Some(if n.is_odd_integer() { negated(j) } else { j });
    }
    if n.is_even_integer() {
        let t = even_move(&args[1], cx)?;
        return Some(fun(op, vec![args[0].clone(), t], cx));
    }
    // An odd order is an odd head (`odd_head`): the class owners place its sign.
    None
}

/// `besseli(n, t)` at an integer literal order `n >= 0`: the parity `I_n(-t) = (-1)^n I_n(t)`.
/// An even order is an even function of `t`, bit for bit in float64 too, so it reads the argument
/// an even function reads (`even_move`: `besseli(2, -x)` and `besseli(0, |x|)` are
/// `besseli(2, x)` and `besseli(0, x)`, `besseli(0, y - x)` is `besseli(0, x - y)`). An odd order
/// moves the sign of a negative literal multiple out (`besseli(1, -2 x)` is `-besseli(1, 2 x)`)
/// where its premise holds: the kernel gives `+0` at a zero of either sign. A ground argument
/// under an odd order keeps its sign, as an odd function's does (`besseli(1, -2)`).
///
/// The laws applied here (`ac::laws::Law`): `fun.besseli_parity`.
fn besseli_parity(op: Tok, args: &[Ex], cx: &Cx) -> Option<Ex> {
    let Ex::Num(n) = &args[0] else { return None };
    if !n.is_integer() || n.is_negative() {
        return None;
    }
    if n.is_even_integer() {
        let t = even_move(&args[1], cx)?;
        return Some(fun(op, vec![args[0].clone(), t], cx));
    }
    if !args[1].has_measure_space(cx.view) && !args[1].contains_wildcard(cx.view) {
        return None;
    }
    let u = cx.carried_negative(&args[1])?;
    if !Law::FunBesseliParity.holds(&args[1], cx) {
        return None;
    }
    let i = fun(op, vec![args[0].clone(), u], cx);
    Some(mul(vec![Ex::Num(Rat::NEG_ONE), i], cx))
}

/// The laws applied here (`ac::laws::Law`): `fun.rootn_index`, `fun.abs_nonnegative`,
/// `fun.half_period`, `num.abs_literal`, `num.exp_one`, `num.rootn_index`, and the inverse pairs'
/// (`Cx::inverse_pair_collapse`).
pub fn fun(op: Tok, args: Vec<Ex>, cx: &Cx) -> Ex {
    // A `pow` call (`pow_spelled`: numpy's `sqrt` of a signed zero) is rebuilt as the spelling
    // it is, which decides again whether it stays a call.
    if args.len() == 2 && cx.view.tok_is(op, "pow") {
        let mut it = args.into_iter();
        let (b, e) = (it.next().expect("base"), it.next().expect("exponent"));
        return pow_spelled(b, e, cx);
    }
    // A ZERO PRODUCT UNDER A HEAD, in `f64` (`mul.zero`). A head that keeps a zero, its sign and
    // NaN (`f(-0) = -0`) is the product itself: `atan(0*x0)` is `0*x0`, as float64 gives it. A
    // head that reads no zero's sign (`f(-0) = f(+0)`) takes the product without the factors that
    // only gave it its sign: `exp(0*x0)` is `exp(0)`. An odd root is a head that keeps a zero, an
    // even one reads no sign.
    let mut args = args;
    if args.first().is_some_and(|z| zero_product(z, cx)) {
        let keeps = |h: &&str| cx.view.tok_is(op, h);
        let rootn_index = (args.len() == 2 && cx.view.tok_is(op, "rootn"))
            .then(|| match &args[1] {
                Ex::Num(n) if n.is_integer() && n.signum() > 0 => Some(n.is_odd_integer()),
                _ => None,
            })
            .flatten();
        let one = args.len() == 1;
        if (one
            && [
                "sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "atanh", "erf", "floor",
                "ceil",
            ]
            .iter()
            .any(keeps))
            || rootn_index == Some(true)
        {
            return args.swap_remove(0);
        }
        if (one
            && ["abs", "exp", "cos", "cosh", "log", "acos", "acosh", "erfc"]
                .iter()
                .any(keeps))
            || rootn_index == Some(false)
        {
            if let Some(u) = unsigned_zero(&args[0], cx) {
                args[0] = u;
            }
        }
    }
    // A literal argument LANDS here (ρ, [`land`]), unless the call is ground.
    let args: Vec<Ex> = if !in_ground(&args, cx) {
        args.into_iter().map(|a| land_operand(a, cx)).collect()
    } else {
        args
    };
    // `exp(1) -> E`. The two spellings denote the SAME constant exactly -- this is a
    // choice of symbol, not an evaluation, so the rules-pass doctrine above does not
    // cover it (nothing is rounded and no exact rule can be preempted; `exp(0) -> 1`
    // IS a literal evaluation and stays with the rules, where it already lives).
    //
    // It is the WRITE half of the invariant `compose_e_power` READS: that helper takes
    // the leaf `E` to mean `E^1`, so the canon must have exactly one spelling for the
    // value or the composition can land on the dearer one. Measured, and not
    // hypothetically: with the composition in and this fold out, `inv(inv(e))` and
    // `x0/exp(-1)` both settled on `exp 1` at mu 10,000 where `np.e` is 8,000, and the
    // re-mine dropped 42 rules that had been carrying exactly that respelling in
    // context (`* exp 1 _0 -> * np.e _0`, `- _0 exp 1 -> - _0 np.e`, ...) -- the mine
    // papering over a missing constructor fold, one context at a time.
    //
    // Unconditional: `np.e` is a universal leaf, parsed and emitted with no config
    // declaration behind it (`convert.rs` maps the token both ways for every table),
    // unlike `exp` itself, which is why the arms that BUILD an `exp` are gated and this
    // one is not.
    if args.len() == 1 && cx.view.tok_is(op, "exp") && matches!(&args[0], Ex::Num(r) if r.is_one())
    {
        return Ex::E;
    }
    // `abs` of a rational literal is EXACT arithmetic (|p/q| = |p|/q, no rounding),
    // so it folds in the constructor -- the fold unification deleted
    // the f64 evaluation path, and abs is the one vocabulary function that is exact
    // on rationals (everything else is transcendental and arrives as mined,
    // symbolically certified rules).
    if args.len() == 1 && cx.view.tok_is(op, "abs") {
        if let Ex::Num(r) = &args[0] {
            if r.is_negative() {
                if let Some(a) = r.checked_neg() {
                    return Ex::Num(a);
                }
            } else {
                return Ex::Num(r.clone());
            }
        }
    }
    // `max` AND `min` ARE LATTICE BAGS, as sums and products are bags: `lattice`.
    if !args.is_empty() && lattice_head(op, cx.view) {
        return lattice(op, args, cx);
    }
    // The step heads of literals (`floor`, `ceil`, `mod`) are no constructor's: the ground
    // fold computes them, `f64` with the deployed kernels at a literal's nearest float
    // (`floor(0.99999999999999999)` is 1 there) and `permissive` by the table.
    // In `permissive` an operation on literals is the table's exact value
    // (`table_literal_fold`):
    // `rootn(0, -2)` and `mod(0, inf)` are NaN, `exp(-inf)` 0, `tanh(inf)` 1, `acosh(1)` 0.
    if cx.table_values() && !args.is_empty() && args.iter().all(table_literal) {
        if let Some(folded) = table_literal_fold(&cx.view.resolve_owned(op), &args) {
            return folded;
        }
    }
    let mut args = args;
    if args.len() == 2
        && cx.ieee_zeros()
        && cx.view.tok_is(op, "atan2")
        && matches!(&args[0], Ex::Mul(v) if v.iter().any(|m| matches!(m, Ex::Num(r) if r.is_negative())))
    {
        let x = args.pop().expect("two arguments");
        let y = args.pop().expect("two arguments");
        return atan2_of(op, product_members(y), product_members(x), cx);
    }
    // AN EVEN FUNCTION EATS THE SIGN OF ITS ARGUMENT: `g(-t) -> g(t)`.
    //
    // ONE arm, replacing thirty mined rules -- the entire
    // `{abs, cos, cosh} x {sin, sinh, tan, tanh, asin, asinh, atan, atanh}` cross product
    // plus `cos neg _0` and `cosh neg _0`. Those thirty are not thirty facts: they are TWO
    // (an odd function propagates a sign, an even one absorbs it) multiplied by fifteen
    // pairs, and the mine enumerated the product because the engine had no name for either
    // factor. `drop_sign` is the missing name.
    //
    // It fixes what no rule can. Three separate holes close at once:
    //   * LITERAL arguments. The rules name a `neg` NODE; `neg(3)` folds to the literal
    //     `-3` before any matcher runs, so every one of the thirty is blind at every
    //     numeric binding (`abs asinh neg 3` shipped as `abs asinh -3`, mu 19,000, where
    //     `asinh 3` is 10,000). This arm reads the canonical SIGN instead, so `neg 3`,
    //     `-3`, `(-1)*x` and `neg(sin x)` are one case.
    //   * DEPTH. The mine caps source patterns at length 4, so `abs sin tanh neg _0` (five
    //     tokens) has no rule at all -- it did not reduce even at a VARIABLE. A recursion
    //     has no depth limit.
    //   * The bare case the rules never covered: `cos (-2)` shipped as `cos -2` at 10,585
    //     against 10,000 for `cos 2`.
    //
    // It composes with `certainly_nonneg` without either knowing about the other:
    // `abs asinh -3` --(here)--> `abs asinh 3` --(nonneg licence)--> `asinh 3`.
    //
    // TOTAL, no licence needed: evenness is a pointwise identity on the whole extension,
    // and `is_even_fun` is a certified table, not a config field. It terminates by the price of
    // its argument (`even_move`).
    if args.len() == 1 && cx.is_even_fun(op) {
        if let Some(arg) = even_move(&args[0], cx) {
            return fun(op, vec![arg], cx);
        }
    }
    // `besselj` OF LITERALS AT ZERO, and OF AN INTEGER ORDER: its reflection and its parity.
    if args.len() == 2 && cx.view.tok_is(op, "besselj") {
        if let Some(e) =
            besselj_of_literals_at_zero(&args, cx).or_else(|| besselj_integer_order(op, &args, cx))
        {
            return e;
        }
    }
    // `besseli` OF AN INTEGER ORDER: its argument's parity.
    if args.len() == 2 && cx.view.tok_is(op, "besseli") {
        if let Some(e) = besseli_parity(op, &args, cx) {
            return e;
        }
    }
    // A HALF-PERIOD SHIFT DROPS OUT: `f(t +- pi)` for f in {sin, cos, tan}.
    //
    // Eleven pattern rules say this (`sin|cos|tan` x `+|-` x both argument orders, plus
    // `f(pi - t)`), and every one of them is blind at a LITERAL argument, for TWO different
    // reasons that no rule can fix:
    //   * sin/cos sit at the `_` sort and go blind at a NEGATIVE literal, because the
    //     constructor factors the bag's common sign: `(-2) - pi` is stored as `neg(2 + pi)`,
    //     and the `-pi` member the pattern needs is gone. Family A's mechanism exactly.
    //   * tan sits at `!`, and `bind_ok` REFUSES ground subjects outright -- a documented,
    //     accepted recall loss on certificate-carrying slots, not a defect. So `tan + 2 np.pi`
    //     cannot bind `2` at all, and tan was blind at EVERY literal, both signs.
    // Measured before the arm: of the 18 `{sin,cos,tan} x {+,-} x {x0, 2, (-2)}` states, cos
    // reduced 6/6, sin 5/6, tan only the 2 variable cases.
    //
    // Each function's law is DERIVED, not taken by symmetry, and the derivation
    // is what picks the output spelling:
    //   * sin is ODD:  sin(t +- pi) = -sin(t) = sin(-t), so the sign may go either side, and
    //     WHICH side is a structural question the first build got wrong. Pushing it into the
    //     argument always looked right from the literal case (`sin -2` prices 10,585 where
    //     `neg sin 2` is 18,000, and the wrapped form would be a WORSE answer than the rule it
    //     replaces). At a VARIABLE the two spellings are mu-EQUAL, and there the choice is
    //     decided by what they COMPOSE with: `test_function_rooted_rules_mint_no_twin` failed
    //     on `neg(sin(x0 + pi))`, which the wrapped form collapses to `sin x0` and the pushed-in
    //     form leaves at `neg sin neg x0` -- an outer `neg` cancels a wrapped sign and cannot
    //     reach one buried in the argument. The condition is therefore whether the sign has
    //     something to CANCEL AGAINST: push in iff the remainder already carries a negative
    //     (`carried_negative`), wrap otherwise. STRUCTURAL, not a mu guard -- mu
    //     guards in constructor arms break idempotence. The first build tested `rest is a Num`
    //     instead, which is the same answer on literals but left `sin(pi - t)` at the
    //     uncollapsible `neg sin neg t` and kept a rule alive to say so.
    //   * cos is EVEN: cos(t +- pi) = -cos(t), and evenness gives no argument trick, so the
    //     `neg` stays outside -- which is what the existing rule already emits.
    //   * tan:         tan(t +- pi) = tan(t), no sign at all.
    // The sign of the pi member is irrelevant to all three, so `+pi` and `-pi` are one case.
    //
    // The judge was asked first, at the plain sort and at literals: all of
    // `sin|cos|tan {+,-} _0 np.pi` and the ground `tan +- 2 np.pi`, `sin + (-2) np.pi` come
    // back CERTIFIED. tan's `!` is a miner-side sort choice, not a judge requirement. The law is
    // `fun.half_period`.
    if args.len() == 1 {
        let shift = match cx.view.resolve_owned(op).as_str() {
            "sin" | "cos" | "tan" => Some(cx.view.tok_is(op, "cos")),
            _ => None,
        };
        if let Some(negate_outside) = shift {
            if let Some(rest) = cx.strip_unit_pi(&args[0]) {
                return if cx.view.tok_is(op, "tan") {
                    fun(op, vec![rest], cx)
                } else if !negate_outside {
                    // sin: push the sign INTO the argument exactly when the argument ABSORBS
                    // it without adding a node. Two ways that happens, and both are needed --
                    // testing only one of them was wrong twice:
                    //   * a LITERAL of either sign absorbs it into its own sign
                    //     (`sin(-2)` prices 10,585 where `neg sin 2` is 18,000);
                    //   * a subtree already CARRYING a negative annihilates it
                    //     (`sin(pi - t)` lands on `sin t`, not the uncollapsible
                    //     `neg sin neg t` that kept a rule alive to say so).
                    // A bare symbolic argument absorbs nothing, so it WRAPS -- mu-equal there,
                    // and only the wrapped sign can be cancelled by an outer `neg`.
                    if let Some(positive) = cx.carried_negative(&rest) {
                        fun(op, vec![positive], cx)
                    } else if matches!(rest, Ex::Num(_)) {
                        fun(op, vec![mul(vec![Ex::Num(Rat::NEG_ONE), rest], cx)], cx)
                    } else {
                        mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![rest], cx)], cx)
                    }
                } else {
                    mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![rest], cx)], cx)
                };
            }
        }
    }
    // AN INVERSE PAIR COLLAPSES: `f(g(t)) -> t` where the licence holds. See
    // `inverse_pair_collapse` -- the table and its three licence groups are the contract
    // judge's own verdicts, taken before the arm was written.
    //
    // The engine missed ONE of these (`exp log abs _0 -> abs _0`). The artifact was
    // carrying **101** identity-shaped inverse-pair rules, which is the same picture other families
    // showed: the mine enumerating a fact the engine has no name for, one member and one
    // LITERAL at a time (`asinh sinh 1`, `asinh sinh 2`, ... `log exp 10`, `acosh cosh np.pi`).
    // A pair is a pair at every argument, so a table costs what one arm costs.
    if args.len() == 1 {
        if let Ex::Fun(inner, iargs) = &args[0] {
            if iargs.len() == 1 {
                if let Some(collapsed) = cx.inverse_pair_collapse(op, *inner, &iargs[0]) {
                    return collapsed;
                }
            }
        }
    }
    // THE LOG OF AN INVERSE (`fun.log_inverse`): `log(1/t) = -log(t)` and
    // `log(abs(1/t)) = -log(abs(t))` where its law holds, at the root's walk in both modes
    // (`release_flat`; in `f64` over a monomial `t`, where nothing reads the sign of a zero); a
    // fully numeric `t` is read for its value first.
    if args.len() == 1 && cx.view.tok_is(op, "log") {
        if let Some((t, in_abs)) = log_inverse_read(&args[0], cx.view) {
            if (t.has_measure_space(cx.view) || t.contains_wildcard(cx.view))
                && Law::FunLogInverse.holds(t, cx)
            {
                let read = if in_abs {
                    fun(cx.view.intern("abs"), vec![t.clone()], cx)
                } else {
                    t.clone()
                };
                return mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![read], cx)], cx);
            }
        }
    }
    // `rootn` index normalization lives HERE, in the constructor, so parse-time and
    // mid-pass rebuilds agree: an index subtree that a numeric fold collapses mid-pass
    // (`cos 0 -> 1`) must normalize in the SAME call, or idempotence breaks. Under the
    // IEEE `rootn` semantics every arm is a sound identity:
    //   rootn(x, 1) = x;  rootn(x, even) = pow(x, 1/even) (both principal, NaN on
    //   negatives);  rootn(x, -n) = pow(rootn(x, n), -1);  rootn(x, 0) and a
    //   provably-non-integer index = NaN everywhere (invalid operation, total).
    // Only binary Fun heads are inspected (rootn is the sole binary function), so the
    // name check stays off the unary hot path. `Const` and compound indices stay
    // symbolic: the evaluators fail closed on them.
    if args.len() == 2 && cx.view.resolve_owned(op) == "rootn" {
        let index = &args[1];
        match index {
            Ex::Num(r) => {
                if r.is_integer() {
                    let n = r.clone();
                    if n.is_zero() {
                        return Ex::NaN;
                    }
                    if n.is_one() && Law::RootnOne.holds(&args[0], cx) {
                        return args.into_iter().next().unwrap();
                    }
                    // A ROOT of `exp` is an `exp`: `rootn(exp a, n) -> exp(a/n)`, which is
                    // `compose_e_power` with the exponent `1/n` -- the SAME call `pow` makes
                    // with `b`, so the two spellings of a power reach the identical
                    // reduction (route independence) and the shape is
                    // unreachable by a mined rule either way: `rootn np.e 2 -> exp 0.5` has
                    // a var-free target, which `candidate_fold_filter` drops at mine time.
                    //
                    // `n` is a nonzero integer here (`n == 0` and `n == 1` returned above),
                    // so `1/n` is a finite nonzero rational and the sign follows through the
                    // reciprocal for free: `rootn(e, -m) = 1/e^(1/m) = exp(-1/m)`. The
                    // first version of this arm handled the LEAF `e` only, and the
                    // census that came out of it found
                    // its sibling still blind on the very next base: `pow(exp(-1), 1/2)` was
                    // shipping as `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is
                    // 11,585. One base is a patch; the family is the principle.
                    if let Some(composed) = n
                        .checked_inv()
                        .and_then(|inv| compose_e_power(&args[0], &Ex::Num(inv), cx))
                    {
                        return composed;
                    }
                    // n >= 2 EVEN: kept as `rootn`. The conversion to `pow(x, 1/n)` used
                    // to live here; it now runs the OTHER way (see `pow`), because mu
                    // prefers the root spelling once a genuine fraction pays for its
                    // denominator -- `rootn x 2` prices 18 against `pow x 1/2` at 19.
                    if n.is_negative() {
                        // checked_neg, not `-n`: a MIN index cannot be negated (release
                        // wraps silently -> the still-negative index would recurse here
                        // forever). Unreachable while the no-MIN Rat invariant holds;
                        // refusing to fold (stay symbolic) is sound either way.
                        if let Some(pos) = n.checked_neg() {
                            let base = args.into_iter().next().unwrap();
                            let inner = fun(op, vec![base, Ex::Num(pos)], cx);
                            return pow(inner, Ex::int(-1), cx);
                        }
                    }
                    // ROOT COMPOSITION: rootn(rootn(x, m), n) -> rootn(x, m*n). Sound at
                    // every parity. m even: the inner root is NaN on x < 0 and m*n is even
                    // too, so both sides are NaN there. m odd with n even: the inner keeps
                    // the sign and the outer even root is NaN on it, and m*n is even.
                    // Both odd: signs pass through on both sides. This composition used to
                    // arrive for free through the pow-of-pow arm (even roots WERE pows);
                    // now that `rootn` survives, it has to be explicit or nested roots stop
                    // composing (`rootn(rootn(x,2),2)` measured 28 against 19 for the
                    // fourth root).
                    if let Ex::Fun(inner_op, inner_args) = &args[0] {
                        if inner_args.len() == 2 && cx.view.tok_is(*inner_op, "rootn") {
                            if let Ex::Num(m) = &inner_args[1] {
                                if m.is_integer() && m.cmp_int(2) != Ordering::Less {
                                    {
                                        if let Some(prod) = m.checked_mul(&n) {
                                            let inner_base = inner_args[0].clone();
                                            return fun(op, vec![inner_base, Ex::Num(prod)], cx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // POWER-THROUGH-ROOT at an EVEN index: `rootn(A^k, n) -> A^(k/n)` under the
                    // SAME licence as the pow-of-pow composition in `pow`. An even `rootn` IS the
                    // principal power `.^(1/n)`, so this IS that composition, merely reached
                    // through the root spelling; sharing the licence is what makes the two routes
                    // agree. Without it the constructor and the rejoin path disagreed:
                    // `rootn(1/cosh(t), 4)` stayed put at construction but reduced to
                    // `cosh(t)^(-1/4)` after a projection round-trip, so the explicit form was not
                    // a fixpoint (64k row 51607).
                    //
                    // The licence also keeps the step sound. A NEGATIVE odd `k` is refused
                    // unless the base is certainly non-negative, because at `A = -inf` the inner
                    // `A^k` is ZERO and the source is then DEFINED while `A^(k/n)` is NaN --
                    // defined -> undefined, which soundness forbids at zero measure tolerance.
                    // `cosh` clears it on the certainly-non-negative disjunct, a bare variable
                    // does not. `s = 1/n` is never an integer here, so that disjunct is vacuous.
                    if n.is_even_integer() {
                        let composed = if let Ex::Pow(inner, ex) = &args[0] {
                            match &**ex {
                                Ex::Num(k) => n.checked_inv().and_then(|s| {
                                    let licensed = Law::PowCompose.holds_powers(
                                        Powers::Rational {
                                            base: inner,
                                            r: k,
                                            s: &s,
                                        },
                                        cx,
                                    );
                                    k.checked_mul(&s).filter(|_| licensed)
                                }),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some(p) = composed {
                            let Ex::Pow(inner, _) = args.into_iter().next().unwrap() else {
                                unreachable!()
                            };
                            return pow(*inner, Ex::Num(p), cx);
                        }
                    }
                    // n >= 3 odd: the genuine signed root. The contract-blessed
                    // inverse composition folds FIRST (SIMPLIFICATION_CONTRACT_v2 s5
                    // Corollaries: pow1_3(pow3 t) = t at EVERY value including nan and
                    // +-inf, hence _-sound): rootn(t^n, n) -> t for the SAME odd n.
                    //
                    // The ODD guard is load-bearing and newly EXPLICIT: at even n the
                    // identity is false -- rootn(t^2, 2) is |t|, not t. It was previously
                    // unreachable at even n only because even indices converted to `pow`
                    // above; with them surviving, the parity has to be tested here.
                    if n.is_odd_integer() {
                        if let Ex::Pow(b, ex) = &args[0] {
                            if let Ex::Num(rr) = &**ex {
                                if *rr == n
                                    && Law::PowRootInverse.holds_powers(
                                        Powers::Rational {
                                            base: b,
                                            r: &n,
                                            s: &n,
                                        },
                                        cx,
                                    )
                                {
                                    let Ex::Pow(inner, _) = args.into_iter().next().unwrap() else {
                                        unreachable!()
                                    };
                                    return *inner;
                                }
                            }
                        }
                    }
                    // EXACT RATIONAL ODD ROOT: a
                    // rational base that is a perfect n-th power folds exactly --
                    // rootn(8,3) = 2, rootn(1,k) = 1, rootn(-8,3) = -2 (checked_root
                    // carries the sign through odd k). This is rational ARITHMETIC,
                    // not transcendental evaluation, so fold unification mandates it
                    // in the constructor (the deleted f64 arm had been carrying it,
                    // same story as abs; the mine cannot cover it -- index/base
                    // literals beyond the alphabet). Inexact bases keep the Fun.
                    if let Ex::Num(r) = &args[0] {
                        if let Some(root) = r.checked_root_integer(&n) {
                            return Ex::Num(root);
                        }
                    }
                    // Otherwise keep the Fun.
                } else {
                    return Ex::NaN; // exact non-integer index: invalid, NaN everywhere
                }
            }
            Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => return Ex::NaN,
            _ => {}
        }
    }
    // Constructor-level abs elimination: `abs t -> t` on a certainly-non-negative operand
    // (`fun.abs_nonnegative`). Both sides share t's domain exactly (abs(nan) = nan,
    // abs(+inf) = +inf) and t's value is >= 0 wherever defined; in `f64` t is not `-0` on a region
    // too, since `|-0|` is `+0` (numpy's `sqrt` keeps `ceil(x0)`'s `-0` on (-1, 0), so
    // `|sqrt(ceil(x0))|` stays). Living HERE rather than in the mined
    // ruleset makes the fold reachable from every entry path and structurally immune to
    // serve-order shadowing: a row diff found the old `inv abs t -> abs inv t`
    // orientation respelling |x^2| into |x^-2| before the exponent-literal abs rules could
    // match, sticking a dead abs in 9/65,536 (64k) + 25/1M outputs. The abs rules this
    // subsumes self-retire at the next mine (mint-vs-mark: the mark now carries the fold).
    // `tok_is` keeps the head dispatch allocation-free on the unary hot path.
    if args.len() == 1 && cx.view.tok_is(op, "abs") && Law::FunAbsNonnegative.holds(&args[0], cx) {
        return args.into_iter().next().unwrap();
    }
    // THE SIGN OF AN ODD HEAD: `f(-u)` and `-f(u)` are one value for an odd head of a negatable
    // argument (`class_base`). A normal-form choice: the cheaper spelling stands, and on an equal
    // price the tie order (`tie_cmp`: fewer signs, then the outer one, so the sign stands outside;
    // then the structure). A product the head stands in decides again over all its members
    // (`sign_place`). A constant argument keeps its sign on its literal (`sin(-2)`).
    let head = Ex::Fun(op, args);
    let pos = match &head {
        Ex::Fun(f, a) => odd_head(cx.view, *f, a).map_or(0, |(p, _)| p),
        _ => 0,
    };
    if let (Some(r), Ex::Fun(_, a)) = (odd_of_literal_multiple(&head, cx.view), &head) {
        if r.is_negative()
            && (a[pos].has_measure_space(cx.view) || a[pos].contains_wildcard(cx.view))
        {
            if let Some(p) = positive_head(&head, cx.view) {
                // The positive head as the constructor builds it: an inverse pair inside it
                // collapses (`sinh(-asinh(x0))` is `-x0`), and the product is that.
                if let Ex::Fun(f, a) = &p {
                    let built = fun(*f, a.clone(), cx);
                    if built != p {
                        return mul(vec![Ex::Num(Rat::NEG_ONE), built], cx);
                    }
                }
                let mut v = vec![Ex::Num(Rat::NEG_ONE), p];
                v.sort_by(|x, y| mul_factor_cmp(x, y, cx.view));
                let outside = Ex::Mul(v);
                let (inside_mu, outside_mu) =
                    (complexity(&head, cx.view), complexity(&outside, cx.view));
                if outside_mu < inside_mu
                    || (outside_mu == inside_mu
                        && tie_cmp(&outside, &head, cx.view) == Ordering::Less)
                {
                    return outside;
                }
            }
        }
    }
    // A head of a sum is a class site: alone it takes the orientation it takes as a product's one
    // factor (`lone_site`). Its flip negates the sum in place, so asking it builds no head.
    if matches!(&head, Ex::Fun(_, a) if matches!(a.get(pos), Some(Ex::Add(_)))) {
        return lone_site(head, cx);
    }
    head
}

/// Full recursive re-canonicalization: rebuild `e` bottom-up through the canonical
/// constructors. Used on freshly substituted rule RHS instances and at the parse boundary.
pub fn canon(e: Ex, cx: &Cx) -> Ex {
    // One scope for the whole re-walk: a product met again is decided once (`FlipScope`).
    let _memo = FlipScope::open();
    canon_walk(e, cx)
}

fn canon_walk(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let parts = v.into_iter().map(|x| canon_walk(x, cx)).collect();
            add(parts, cx)
        }
        Ex::Mul(v) => {
            // A member that is a product under a negative integer power gives the enclosing
            // product its own members, each raised (`inverted_members`), so that one product
            // decides them all.
            let mut parts = Vec::with_capacity(v.len());
            for x in v {
                match x {
                    Ex::Pow(b, k) if inverted_product(&b, &k) => {
                        let (Ex::Mul(w), Ex::Num(k)) = (*b, *k) else {
                            unreachable!()
                        };
                        match inverted_members(w, &k, cx) {
                            Ok(raised) => parts.extend(raised),
                            Err(kept) => parts.push(kept),
                        }
                    }
                    x => parts.push(canon_walk(x, cx)),
                }
            }
            mul(parts, cx)
        }
        Ex::Pow(b, k) if inverted_product(&b, &k) => {
            let (Ex::Mul(w), Ex::Num(k)) = (*b, *k) else {
                unreachable!()
            };
            match inverted_members(w, &k, cx) {
                Ok(raised) => mul(raised, cx),
                Err(kept) => kept,
            }
        }
        Ex::Pow(b, ex) => {
            let b = canon_walk(*b, cx);
            let ex = canon_walk(*ex, cx);
            pow(b, ex, cx)
        }
        Ex::Fun(f, v) if v.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(f, "atan2") => {
            let mut members = v.into_iter().map(|a| match a {
                Ex::Mul(w) => w.into_iter().map(|m| canon_walk(m, cx)).collect(),
                other => vec![canon_walk(other, cx)],
            });
            let ym: Vec<Ex> = members.next().expect("two arguments");
            let xm: Vec<Ex> = members.next().expect("two arguments");
            atan2_of(f, ym, xm, cx)
        }
        Ex::Fun(f, v) => {
            let parts = v.into_iter().map(|x| canon_walk(x, cx)).collect();
            fun(f, parts, cx)
        }
        leaf => leaf,
    }
}

/// Is `b^k` a product under a negative integer power?
fn inverted_product(b: &Ex, k: &Ex) -> bool {
    matches!((b, k), (Ex::Mul(_), Ex::Num(k)) if k.is_integer() && k.is_negative())
}

/// The members of a product `w` raised to the negative integer `k`, for the product that decides
/// them (`raised_members`), so the members meet the power as they would spelled one by one. Where
/// `pow` does not distribute it, the product stays one member under the power (`Err`), built as
/// every product is (`mul`): the canonical state its print reads back to.
fn inverted_members(w: Vec<Ex>, k: &Rat, cx: &Cx) -> Result<Vec<Ex>, Ex> {
    let w: Vec<Ex> = w.into_iter().map(|x| canon_walk(x, cx)).collect();
    raised_members(w.clone(), k, cx).map_err(|_| pow(mul(w, cx), Ex::Num(k.clone()), cx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{OperatorSpec, Operators};
    use crate::tokens::{TokenOverlay, TokenTable, TokenView};
    use std::cell::RefCell;

    fn test_ops() -> (Vec<String>, Operators) {
        let unary = ["sin", "cos", "abs", "exp", "log", "cosh"];
        let mut order = Vec::new();
        let mut specs: rustc_hash::FxHashMap<String, OperatorSpec> = Default::default();
        for n in ["+", "-", "*", "/", "pow"] {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: vec![],
                    inverse: None,
                    arity: 2,
                    precedence: None,
                    commutative: n == "+" || n == "*",
                },
            );
        }
        for n in unary {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: vec![],
                    inverse: None,
                    arity: 1,
                    precedence: None,
                    commutative: false,
                },
            );
        }
        let ops = Operators::from_specs(order.clone(), specs);
        (order, ops)
    }

    /// Run `f` with a fresh view over a minimal operator universe.
    fn with_view<R>(f: impl FnOnce(&TokenView) -> R) -> R {
        let (order, ops) = test_ops();
        let table = TokenTable::build(&order, &ops);
        let overlay = RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        f(&view)
    }

    fn x(view: &TokenView) -> Ex {
        Ex::Leaf(view.intern("x0"))
    }

    fn y(view: &TokenView) -> Ex {
        Ex::Leaf(view.intern("x1"))
    }

    #[test]
    fn add_collects_and_orders() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // x + x -> 2x (same-sign, no licence needed)
            let e = add(vec![x(view), x(view)], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(2), x(view)]));
            // 2x + 3x -> 5x
            let e = add(
                vec![
                    mul(vec![Ex::int(2), x(view)], &cx),
                    mul(vec![Ex::int(3), x(view)], &cx),
                ],
                &cx,
            );
            assert_eq!(e, Ex::Mul(vec![Ex::int(5), x(view)]));
            // Order invariance: y + x == x + y
            let a = add(vec![y(view), x(view)], &cx);
            let b = add(vec![x(view), y(view)], &cx);
            assert_eq!(a, b);
            // Flattening: (x + y) + x -> 2x + y
            let inner = add(vec![x(view), y(view)], &cx);
            let e = add(vec![inner, x(view)], &cx);
            // Stripped term order: the 2x term (key x0) sorts before y (key x1).
            assert_eq!(
                e,
                Ex::Add(vec![Ex::Mul(vec![Ex::int(2), x(view)]), y(view)])
            );
            // Literals fold exactly: 1/2 + 1/3 = 5/6
            let e = add(
                vec![
                    Ex::Num(Rat::new(1, 2).unwrap()),
                    Ex::Num(Rat::new(1, 3).unwrap()),
                ],
                &cx,
            );
            assert_eq!(e, Ex::Num(Rat::new(5, 6).unwrap()));
        });
    }

    #[test]
    fn add_sign_cancel_gate() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // x - x -> 0: variables are certainly finite, licence-free.
            let neg_x = mul(vec![Ex::int(-1), x(view)], &cx);
            assert_eq!(add(vec![x(view), neg_x], &cx), Ex::int(0));
            // log(x) - log(x): NO certificate in a bare context -> refuses to cancel.
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let neg_lg = mul(vec![Ex::int(-1), lg.clone()], &cx);
            let e = add(vec![lg.clone(), neg_lg.clone()], &cx);
            assert!(
                matches!(e, Ex::Add(_)),
                "uncertified cancel must refuse: {e:?}"
            );
            // With the certificate it cancels.
            let yes = |_: &Ex| true;
            let cx2 = Cx {
                view,
                cert_fin: Some(&yes),
                cert_finnz: None,
                cert_nzae: None,
                cert_nce: None,
                cert_region: None,
                cert_encl: None,
                mode: RuleMode::Default,
                fold_f64: Cx::folds_for(RuleMode::Default),
                ground_first: None,
                zero_sign_unread: false,
                flat_unread: false,
                sign_held: None,
            };
            assert_eq!(add(vec![lg.clone(), neg_lg], &cx2), Ex::int(0));
            // Same-sign merging never needs the licence: log(x) + log(x) -> 2 log(x).
            let e = add(vec![lg.clone(), lg.clone()], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(2), lg]));
        });
    }

    #[test]
    fn add_extended_values() {
        with_view(|view| {
            let cx = Cx::bare(view);
            assert_eq!(add(vec![Ex::NaN, x(view)], &cx), Ex::NaN);
            assert_eq!(add(vec![Ex::PosInf, Ex::NegInf], &cx), Ex::NaN);
            assert_eq!(add(vec![Ex::PosInf, Ex::int(5)], &cx), Ex::PosInf);
            assert_eq!(add(vec![Ex::PosInf, x(view)], &cx), Ex::PosInf); // variable: finite
                                                                         // inf + log(x): log(x) can be -inf (x=0) -- refuses without certificate.
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let e = add(vec![Ex::PosInf, lg], &cx);
            assert!(matches!(e, Ex::Add(_)));
        });
    }

    #[test]
    fn add_const_independence_and_absorption() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // C + C -> C (one constant absorbs, forall-exists).
            assert_eq!(add(vec![Ex::ANON, Ex::ANON], &cx), Ex::ANON);
            // C + 5 -> C.
            assert_eq!(add(vec![Ex::ANON, Ex::int(5)], &cx), Ex::ANON);
            // 2C + x + 3 -> C + x.
            let e = add(
                vec![mul(vec![Ex::int(2), Ex::ANON], &cx), x(view), Ex::int(3)],
                &cx,
            );
            // Constant-like terms sort LAST under the stripped order (x + C, not C + x).
            assert_eq!(e, Ex::Add(vec![x(view), Ex::ANON]));
            // C*x + C*x must NOT merge (independent constants).
            let cx_term = || Ex::Mul(vec![Ex::ANON, x(view)]);
            let e = add(vec![cx_term(), cx_term()], &cx);
            assert_eq!(e, Ex::Add(vec![cx_term(), cx_term()]));
        });
    }

    #[test]
    fn a_constant_absorbs_the_orientation_sign_of_every_class() {
        with_view(|view| {
            let sin = view.intern("sin");
            for mode in [RuleMode::Default, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
                // A sum holding a `<constant>` takes a sign into its terms: -(x0 + x1^c) is
                // -x0 - x1^c.
                let s = add(vec![x(view), pow(y(view), Ex::ANON, &cx)], &cx);
                let neg_s = neg(s.clone());
                assert!(matches!(&neg_s, Ex::Add(_)), "{mode:?}: {neg_s:?}");
                // Beside a bare `<constant>`, which absorbs every constant factor of its
                // product, that sign is the constant's, whatever route built the product, and the
                // sum takes the orientation with the fewer signs: c*-(x0 + x1^c) is c*(x0 + x1^c).
                let want = mul(vec![Ex::ANON, s.clone()], &cx);
                assert_eq!(want, Ex::Mul(vec![Ex::ANON, s.clone()]), "{mode:?}");
                assert_eq!(mul(vec![neg_s.clone(), Ex::ANON], &cx), want, "{mode:?}");
                assert_eq!(
                    mul(vec![Ex::ANON, Ex::int(-1), s.clone()], &cx),
                    want,
                    "{mode:?}"
                );
                // So does an odd function of one: c*sin(c - x0) is c*sin(x0 + c).
                let t = add(vec![x(view), Ex::ANON], &cx);
                assert_eq!(
                    mul(vec![Ex::ANON, fun(sin, vec![neg(t.clone())], &cx)], &cx),
                    mul(vec![Ex::ANON, fun(sin, vec![t], &cx)], &cx),
                    "{mode:?}"
                );
                // Without a bare `<constant>` the sum keeps the sign the product gave it.
                let kept = mul(vec![y(view), neg_s.clone()], &cx);
                assert!(
                    matches!(&kept, Ex::Mul(v) if v.contains(&neg_s)),
                    "{mode:?}: {kept:?}"
                );
            }
        });
    }

    #[test]
    fn a_lone_site_takes_the_orientation_of_a_products_one_factor() {
        with_view(|view| {
            let sin = view.intern("sin");
            let z = || Ex::Leaf(view.intern("x2"));
            let half = Ex::Num(Rat::new(1, 2).unwrap());
            for mode in [RuleMode::Default, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
                // x0 - x1 - x2: two signs; -(x1 + x2 - x0) has two too, the outer one
                let s = add(vec![x(view), neg(y(view)), neg(z())], &cx);
                let raw_cube = Ex::Pow(Box::new(s.clone()), Box::new(Ex::int(3)));
                let raw_sin = Ex::Fun(sin, vec![s.clone()]);
                // alone, as the one factor of a product: (x0 - x1 - x2)^3 is
                // -(x1 + x2 - x0)^3, and sin(x0 - x1 - x2) is -sin(x1 + x2 - x0)
                let cube = pow(s.clone(), Ex::int(3), &cx);
                assert_eq!(
                    cube,
                    mul(vec![Ex::int(1), raw_cube.clone()], &cx),
                    "{mode:?}"
                );
                assert!(
                    matches!(&cube, Ex::Mul(v) if v.contains(&Ex::int(-1))),
                    "{mode:?}: {cube:?}"
                );
                let sine = fun(sin, vec![s.clone()], &cx);
                assert_eq!(
                    sine,
                    mul(vec![Ex::int(1), raw_sin.clone()], &cx),
                    "{mode:?}"
                );
                // a power over a lone site reads its class, whichever orientation it stands in
                assert_eq!(
                    pow(cube.clone(), Ex::int(-1), &cx),
                    pow(s.clone(), Ex::int(-3), &cx),
                    "{mode:?}"
                );
                assert_eq!(
                    pow(cube, half.clone(), &cx),
                    pow(s.clone(), Ex::Num(Rat::new(3, 2).unwrap()), &cx),
                    "{mode:?}"
                );
                assert_eq!(
                    pow(sine, Ex::int(-1), &cx),
                    pow(raw_sin, Ex::int(-1), &cx),
                    "{mode:?}"
                );
            }
        });
    }

    #[test]
    fn a_zero_keeps_its_ieee_sign_where_it_is_inverted() {
        with_view(|view| {
            let lg = |cx: &Cx| fun(view.intern("log"), vec![x(view)], cx);
            for mode in [RuleMode::Default, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let ieee = matches!(mode, RuleMode::Default);
                let inverse = |members: Vec<Ex>| -> Ex {
                    let raised = raised_members(members, &Rat::NEG_ONE, &cx)
                        .unwrap_or_else(|held| vec![pow(held, Ex::int(-1), &cx)]);
                    mul(raised, &cx)
                };
                // 1/(x0*0) is inf/x0 in f64 (-inf wherever x0 < 0); in permissive 1/0 has no
                // value (division by zero is undefined however written)
                let want = if ieee {
                    mul(vec![Ex::PosInf, pow(x(view), Ex::int(-1), &cx)], &cx)
                } else {
                    Ex::NaN
                };
                assert_eq!(inverse(vec![x(view), Ex::int(0)]), want, "{mode:?}");
                // 1/(-1*0) = 1/-0 is -inf in f64
                let want = if ieee { Ex::NegInf } else { Ex::NaN };
                assert_eq!(inverse(vec![Ex::int(-1), Ex::int(0)]), want, "{mode:?}");
                // a product held whole around its zero distributes the inverse too
                let held = pow(Ex::Mul(vec![Ex::int(0), lg(&cx)]), Ex::int(-1), &cx);
                if ieee {
                    let want = mul(vec![Ex::PosInf, pow(lg(&cx), Ex::int(-1), &cx)], &cx);
                    assert_eq!(held, want, "{mode:?}");
                } else {
                    // permissive cancels `0*log(x0)` to 0 without a definedness check, and
                    // then reads 1/0 by the table: no value
                    assert_eq!(held, Ex::NaN, "{mode:?}");
                }
            }
        });
    }

    #[test]
    fn a_zero_term_stays_where_the_rest_may_be_neg_zero_on_a_region() {
        with_view(|view| {
            for mode in [RuleMode::Default, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let x1 = Ex::Leaf(view.intern("x1"));
                let ab = fun(view.intern("abs"), vec![x(view)], &cx);
                // t = x1*(x0 - |x0|): -0 wherever x0 >= 0 and x1 < 0, a region
                let s = add(vec![x(view), mul(vec![Ex::int(-1), ab], &cx)], &cx);
                let t = mul(vec![x1.clone(), s], &cx);
                let plus_zero = add(vec![t.clone(), Ex::int(0)], &cx);
                // 2t - t with both terms over the one key (a product holding the sum: `mul`
                // would turn the negated sum around, a class in its other orientation)
                let scaled = |c: i128| match &t {
                    Ex::Mul(v) => Ex::Mul([vec![Ex::int(c)], v.clone()].concat()),
                    other => panic!("a product expected: {other:?}"),
                };
                let collected = add(vec![scaled(2), scaled(-1)], &cx);
                if matches!(mode, RuleMode::Default) {
                    // t + 0 and 2t - t are +0 there, as float64 sums them
                    let kept = Ex::Add(vec![t.clone(), Ex::int(0)]);
                    assert_eq!(plus_zero, kept);
                    assert_eq!(collected, kept);
                    // the sum keeps its orientation: -(t + 0) is no sum of negated terms
                    assert_eq!(negate_term(&Ex::int(0), &cx), None);
                    // the class in its two orientations does not collect where it may vanish
                    let turned = add(vec![t.clone(), mul(vec![Ex::int(-1), t.clone()], &cx)], &cx);
                    assert!(matches!(&turned, Ex::Add(v) if v.len() == 2), "{turned:?}");
                } else {
                    assert_eq!(plus_zero, t, "{mode:?}");
                    assert_eq!(collected, t, "{mode:?}");
                }
                // x0 + 0 is x0 in every mode: its zero set is null
                assert_eq!(add(vec![x(view), Ex::int(0)], &cx), x(view));
                // x1 + t + 0 is x1 + t: x1 is -0 only at a point
                let sum = add(vec![x1.clone(), t.clone()], &cx);
                assert_eq!(add(vec![sum.clone(), Ex::int(0)], &cx), sum, "{mode:?}");
            }
        });
    }

    #[test]
    fn mul_collects_exponents() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // x * x -> x^2
            let e = mul(vec![x(view), x(view)], &cx);
            assert_eq!(e, Ex::Pow(Box::new(x(view)), Box::new(Ex::int(2))));
            // x^2 * x^-1 -> x (variable base: finite-nonzero a.e. licence is syntactic)
            let x2 = pow(x(view), Ex::int(2), &cx);
            let xm1 = pow(x(view), Ex::int(-1), &cx);
            assert_eq!(mul(vec![x2, xm1], &cx), x(view));
            // x * 1/x -> 1 (full cancel on a variable)
            let xm1 = pow(x(view), Ex::int(-1), &cx);
            assert_eq!(mul(vec![x(view), xm1], &cx), Ex::int(1));
            // log(x)^1 * log(x)^-1: composite base, no certificate -> refuses.
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let lgm1 = pow(lg.clone(), Ex::int(-1), &cx);
            let e = mul(vec![lg.clone(), lgm1], &cx);
            assert!(
                matches!(e, Ex::Mul(_)),
                "uncertified exponent cancel: {e:?}"
            );
            // 2 * 3 * x -> 6x; coefficient exact.
            let e = mul(vec![Ex::int(2), Ex::int(3), x(view)], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(6), x(view)]));
            // Zero: 0 * x^2 -> 0 (finite and never negative); 0 * x keeps x in `f64`, whose sign
            // is the zero's (float64's 0*x is -0 for x < 0); 0 * log(x) stays.
            let sq = pow(x(view), Ex::int(2), &cx);
            assert_eq!(mul(vec![Ex::int(0), sq], &cx), Ex::int(0));
            assert_eq!(
                mul(vec![Ex::int(0), x(view)], &cx),
                Ex::Mul(vec![Ex::int(0), x(view)])
            );
            let e = mul(vec![Ex::int(0), lg.clone()], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(0), lg]));
            // ... and in `permissive`, which reads one zero, 0 * x -> 0.
            let mut lossy = Cx::bare(view);
            lossy.mode = RuleMode::Permissive;
            assert_eq!(mul(vec![Ex::int(0), x(view)], &lossy), Ex::int(0));
        });
    }

    #[test]
    fn powers_of_e_collect_by_their_exponents() {
        with_view(|view| {
            let exp = view.intern("exp");
            let y = || Ex::Leaf(view.intern("x1"));
            for mode in [RuleMode::Default, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let e = |a: Ex| fun(exp, vec![a], &cx);
                let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
                // exp(a) exp(b) = exp(a + b)
                assert_eq!(
                    mul(vec![e(x(view)), e(y())], &cx),
                    e(add(vec![x(view), y()], &cx)),
                    "{mode:?}"
                );
                // exp(u) exp(-u) = exp(0) = 1, u finite almost everywhere
                assert_eq!(mul(vec![e(x(view)), e(neg(x(view)))], &cx), Ex::int(1));
                // e is e^1, and a rational power of an exp is a power of e
                assert_eq!(
                    mul(vec![Ex::E, e(x(view))], &cx),
                    e(add(vec![x(view), Ex::int(1)], &cx))
                );
                let sq = pow(e(x(view)), Ex::int(2), &cx);
                assert_eq!(
                    mul(vec![sq.clone(), e(y())], &cx),
                    e(add(vec![mul(vec![Ex::int(2), x(view)], &cx), y()], &cx))
                );
                // a quotient of powers of e, inside a wider product
                let q = mul(vec![y(), e(x(view)), pow(e(y()), Ex::int(-1), &cx)], &cx);
                assert_eq!(q, mul(vec![y(), e(add(vec![x(view), neg(y())], &cx))], &cx));
                // f64's exp(800) exp(-800) is inf * 0 = NaN, an overflow, where the true value
                // is given: exp(0) = 1
                assert_eq!(
                    mul(vec![e(Ex::int(800)), e(Ex::int(-800))], &cx),
                    Ex::int(1)
                );
                // a lone power of e stays, and one base's powers collect over that base as
                // `pow` spells them: exp(x) exp(x) is exp(x)^2
                assert_eq!(
                    mul(vec![sq.clone(), Ex::int(3)], &cx),
                    Ex::Mul(vec![Ex::int(3), sq.clone()])
                );
                assert_eq!(mul(vec![e(x(view)), e(x(view))], &cx), sq);
            }
        });
    }

    #[test]
    fn mul_branch_cut_gate() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let half = Ex::Num(Rat::new(1, 2).unwrap());
            // x^(1/2) * x^(1/2) must NOT become x (wrong on x < 0). The PROPERTY is what
            // matters, not the shape: with even roots spelled `rootn`, equal factors now
            // collect into `pow(rootn(x,2), 2)`, which is value-exact (NaN on x < 0 both
            // sides) and judge-OK. Asserting "still a Mul" was a proxy that only held while
            // the sound merge did not exist.
            let s = pow(x(view), half.clone(), &cx);
            let e = mul(vec![s.clone(), s.clone()], &cx);
            assert_ne!(e, x(view), "branch-cut merge must refuse: {e:?}");
            // abs(x)^(1/2) * abs(x)^(1/2) -> abs(x): certainly non-negative base.
            let ax = fun(view.intern("abs"), vec![x(view)], &cx);
            let s = pow(ax.clone(), half.clone(), &cx);
            assert_eq!(mul(vec![s.clone(), s], &cx), ax);
            // x^(1/2) * x^1 -> x^(3/2): sum stays non-integer, sound. `factor_split` reads
            // the even root as `x^(1/2)`, so the root spelling does not cost the merge.
            let s = pow(x(view), half, &cx);
            let e = mul(vec![s, x(view)], &cx);
            assert_eq!(
                e,
                Ex::Pow(
                    Box::new(x(view)),
                    Box::new(Ex::Num(Rat::new(3, 2).unwrap()))
                )
            );
        });
    }

    #[test]
    fn abs_elimination_on_certainly_nonneg() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let abs_t = view.intern("abs");
            // |x^2| -> x^2 and |x^-2| -> x^-2: even exponents of EITHER sign (the
            // old serve-order shadow lived exactly at the -2 spelling).
            for n in [2, -2] {
                let p = pow(x(view), Ex::int(n), &cx);
                assert_eq!(fun(abs_t, vec![p.clone()], &cx), p);
            }
            // |1/|x|| -> 1/|x|: certainly-non-negative base under ANY exponent.
            let ax = fun(abs_t, vec![x(view)], &cx);
            let inv_ax = pow(ax.clone(), Ex::int(-1), &cx);
            assert_eq!(fun(abs_t, vec![inv_ax.clone()], &cx), inv_ax);
            // |cos(cos x)| -> cos(cos x): range positivity via the unit-bounded argument.
            let cos_t = view.intern("cos");
            let cc = fun(cos_t, vec![fun(cos_t, vec![x(view)], &cx)], &cx);
            assert_eq!(fun(abs_t, vec![cc.clone()], &cx), cc);
            // `|sinh |x||` settles at `|sinh x|`, NOT at `sinh |x|`. Both are
            // the same function and both price 24,000 -- the abs elimination this test
            // guards is not what changed. What changed is which representative wins: the
            // parity arm reaches the argument FIRST and strips the inner `abs` (an `abs`
            // inside a sign-blind context is a no-op), leaving `sinh x`, whose sign is
            // indefinite, so the OUTER abs correctly stays. Before, the inner `abs` survived
            // and made the argument certainly-non-negative, so the outer one dropped
            // instead. Same node count, same mu, one canonical form rather than two.
            let sh_inner = fun(view.intern("sinh"), vec![ax.clone()], &cx);
            let sh_outer = fun(
                abs_t,
                vec![fun(view.intern("sinh"), vec![x(view)], &cx)],
                &cx,
            );
            assert_eq!(fun(abs_t, vec![sh_inner], &cx), sh_outer);
            // Bag closure: a product of non-negatives sheds the outer abs.
            let prod = mul(vec![pow(x(view), Ex::int(2), &cx), ax.clone()], &cx);
            assert_eq!(fun(abs_t, vec![prod.clone()], &cx), prod);
            // Controls: sign-indefinite operands KEEP the abs. (`abs Const` folding is the
            // Const channel's absorption in the pipeline, deliberately NOT this fold --
            // certainly_nonneg(Const) is false because a fitted value can be negative.)
            for keep in [
                x(view),
                fun(view.intern("sin"), vec![x(view)], &cx),
                fun(cos_t, vec![x(view)], &cx), // cos of an UNBOUNDED argument
                pow(x(view), Ex::int(3), &cx),
                // periodic, not monotone -- so `certainly_nonneg` refuses and the abs
                // stays. The argument is a BARE variable: it used to be
                // `|x|`, and the parity arm now strips that inner `abs` (a no-op inside a
                // sign-blind context), which was testing the parity arm rather than this
                // control's subject.
                fun(view.intern("tan"), vec![x(view)], &cx),
                Ex::ANON,
            ] {
                let e = fun(abs_t, vec![keep.clone()], &cx);
                assert_eq!(e, Ex::Fun(abs_t, vec![keep]), "abs must stay");
            }
        });
    }

    #[test]
    fn mul_extended_and_const() {
        with_view(|view| {
            let cx = Cx::bare(view);
            assert_eq!(mul(vec![Ex::NaN, x(view)], &cx), Ex::NaN);
            assert_eq!(mul(vec![Ex::int(0), Ex::PosInf], &cx), Ex::NaN);
            assert_eq!(mul(vec![Ex::int(-2), Ex::PosInf], &cx), Ex::NegInf);
            assert_eq!(mul(vec![Ex::NegInf, Ex::NegInf], &cx), Ex::PosInf);
            // C * C -> C; C * 5 -> C; C * 0 keeps C in `f64`, whose sign is the zero's, and is 0
            // in `permissive`.
            assert_eq!(mul(vec![Ex::ANON, Ex::ANON], &cx), Ex::ANON);
            assert_eq!(mul(vec![Ex::ANON, Ex::int(5)], &cx), Ex::ANON);
            assert_eq!(
                mul(vec![Ex::ANON, Ex::int(0)], &cx),
                Ex::Mul(vec![Ex::int(0), Ex::ANON])
            );
            let mut lossy = Cx::bare(view);
            lossy.mode = RuleMode::Permissive;
            assert_eq!(mul(vec![Ex::ANON, Ex::int(0)], &lossy), Ex::int(0));
            // inf * C: C = 0 reachable -> keep both factors.
            let e = mul(vec![Ex::PosInf, Ex::ANON], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::PosInf, Ex::ANON]));
        });
    }

    #[test]
    fn pow_folds_and_licences() {
        with_view(|view| {
            let cx = Cx::bare(view);
            assert_eq!(pow(x(view), Ex::int(1), &cx), x(view));
            assert_eq!(pow(x(view), Ex::int(0), &cx), Ex::int(1));
            assert_eq!(pow(Ex::int(1), x(view), &cx), Ex::int(1));
            assert_eq!(pow(Ex::int(2), Ex::int(10), &cx), Ex::int(1024));
            assert_eq!(
                pow(Ex::int(4), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
                Ex::int(2)
            );
            assert_eq!(
                pow(Ex::int(2), Ex::int(-2), &cx),
                Ex::Num(Rat::new(1, 4).unwrap())
            );
            // Inexact EVEN root stays symbolic -- as a `rootn`, which is now the canonical
            // spelling for an even index.
            let e = pow(Ex::int(2), Ex::Num(Rat::new(1, 2).unwrap()), &cx);
            assert!(matches!(e, Ex::Fun(..)), "expected rootn(2,2), got {e:?}");
            // (-8)^(1/3) is NOT the real root in Pow semantics -> stays symbolic.
            let e = pow(Ex::int(-8), Ex::Num(Rat::new(1, 3).unwrap()), &cx);
            assert!(matches!(e, Ex::Pow(..)));
            // (x^2)^3 -> x^6 (both integers).
            let e = pow(pow(x(view), Ex::int(2), &cx), Ex::int(3), &cx);
            assert_eq!(e, Ex::Pow(Box::new(x(view)), Box::new(Ex::int(6))));
            // (x^2)^(1/2) must NOT become x (|x| != x).
            let e = pow(
                pow(x(view), Ex::int(2), &cx),
                Ex::Num(Rat::new(1, 2).unwrap()),
                &cx,
            );
            // Refusal is the PROPERTY: (x^2)^(1/2) must not become x. It is now spelled
            // `rootn(x^2, 2)` (= |x|), which is exactly the refusal, so assert the property
            // rather than the old Pow-of-Pow shape.
            assert_ne!(e, x(view), "sign-erasing composition must refuse: {e:?}");
            // (x^3)^(1/3) must NOT become x in POW semantics: on x < 0 the source is
            // NaN (principal non-integer power of a negative value) while x is finite
            // -- a value fabrication; oddness of the inner exponent licenses
            // nothing about the OUTER principal root. The sound spelling of the
            // inverse composition is rootn (contract s5), tested below.
            let e = pow(
                pow(x(view), Ex::int(3), &cx),
                Ex::Num(Rat::new(1, 3).unwrap()),
                &cx,
            );
            assert!(
                matches!(&e, Ex::Pow(b, _) if matches!(&**b, Ex::Pow(..))),
                "integer-product composition must refuse: {e:?}"
            );
            // (x^5)^0.2 must refuse too.
            let e = pow(
                pow(x(view), Ex::int(5), &cx),
                Ex::Num(Rat::new(1, 5).unwrap()),
                &cx,
            );
            assert!(matches!(&e, Ex::Pow(b, _) if matches!(&**b, Ex::Pow(..))));
            // (x^3)^(1/5) -> x^(3/5): odd inner AND non-integer product -- both sides
            // are NaN on the whole negative half-line, so the fold is exact.
            let e = pow(
                pow(x(view), Ex::int(3), &cx),
                Ex::Num(Rat::new(1, 5).unwrap()),
                &cx,
            );
            assert_eq!(
                e,
                Ex::Pow(
                    Box::new(x(view)),
                    Box::new(Ex::Num(Rat::new(3, 5).unwrap()))
                )
            );
            // rootn(x^n, n) -> x for the SAME odd n: total at every value including
            // nan and +-inf (SIMPLIFICATION_CONTRACT_v2 s5 Corollaries), so the
            // CONSTRUCTOR folds it -- the composition pow refuses lives here soundly.
            let rootn_tok = view.intern("rootn");
            let e = fun(
                rootn_tok,
                vec![pow(x(view), Ex::int(3), &cx), Ex::int(3)],
                &cx,
            );
            assert_eq!(e, x(view));
            let e = fun(
                rootn_tok,
                vec![pow(x(view), Ex::int(5), &cx), Ex::int(5)],
                &cx,
            );
            assert_eq!(e, x(view));
            // ...but only for the MATCHING index: rootn(x^3, 5) keeps the Fun.
            let e = fun(
                rootn_tok,
                vec![pow(x(view), Ex::int(3), &cx), Ex::int(5)],
                &cx,
            );
            assert!(matches!(&e, Ex::Fun(..)), "{e:?}");
            // (x^(1/2))^2 -> must NOT become x (would fabricate a value on x < 0). The
            // inner is spelled `rootn(x,2)` now, so assert the PROPERTY: the licence arm
            // refuses without a non-negativity certificate.
            let e = pow(
                pow(x(view), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
                Ex::int(2),
                &cx,
            );
            assert_ne!(
                e,
                x(view),
                "unlicensed root/power inverse must refuse: {e:?}"
            );
            // ...but (x^(1/2))^3 -> x^(3/2) (still non-integer: NaN region preserved). The
            // composition arm reads the even root as a rational power via
            // `as_rational_power`, so the licence logic applies unchanged.
            let e = pow(
                pow(x(view), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
                Ex::int(3),
                &cx,
            );
            assert_eq!(
                e,
                Ex::Pow(
                    Box::new(x(view)),
                    Box::new(Ex::Num(Rat::new(3, 2).unwrap()))
                )
            );
            // (2x)^2 -> 4x^2 (integer power distributes).
            let e = pow(mul(vec![Ex::int(2), x(view)], &cx), Ex::int(2), &cx);
            assert_eq!(
                e,
                Ex::Mul(vec![
                    Ex::int(4),
                    Ex::Pow(Box::new(x(view)), Box::new(Ex::int(2)))
                ])
            );
        });
    }

    /// A NEGATIVE exponent over an ODD `rootn` had no arm at all,
    /// so the reduction was route-dependent -- the explicit projection re-spelled it and the
    /// re-parse then reduced, leaving the projection outside its own fixpoint (and tripping
    /// the serialization-stability `debug_assert` in `engine/ac.rs`).
    #[test]
    fn odd_rootn_negative_exponent_composes() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let rootn_tok = view.intern("rootn");
            let root = |b: Ex, n: i128| fun(rootn_tok, vec![b, Ex::int(n)], &cx);
            let r = |p: i128, q: i128| Ex::Num(Rat::new(p, q).unwrap());

            // ODD-ROOT COMPOSITION: sound for every NON-INTEGER exponent, because both sides
            // are NaN on a negative base. `rootn(x,3)^(-1/2) -> x^(-1/6)`, matching what the
            // reciprocal spelling already produced.
            assert_eq!(
                pow(root(x(view), 3), r(-1, 2), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(r(-1, 6)))
            );
            assert_eq!(
                pow(root(x(view), 5), r(-1, 2), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(r(-1, 10)))
            );
            assert_eq!(
                pow(root(x(view), 3), r(-3, 2), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(r(-1, 2)))
            );
            // The positive direction is unchanged and still lands on the root spelling.
            assert_eq!(pow(root(x(view), 3), r(1, 2), &cx), root(x(view), 6));

            // INTEGER exponents stay excluded -- `rootn(-8,3)^2 = 4` while `(-8)^(2/3)` is
            // NaN, a positive-measure disagreement. The `rootn` must survive.
            assert!(matches!(
                pow(root(x(view), 3), Ex::int(2), &cx),
                Ex::Pow(..)
            ));
            assert!(matches!(
                pow(root(x(view), 3), Ex::int(-2), &cx),
                Ex::Pow(..)
            ));

            // ROOT/POWER INVERSE, negative mirror: `rootn(x,n)^(-n) -> 1/x` for odd n.
            assert_eq!(
                pow(root(x(view), 3), Ex::int(-3), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(Ex::int(-1)))
            );
            assert_eq!(
                pow(root(x(view), 5), Ex::int(-5), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(Ex::int(-1)))
            );
            // ...positive mirror unchanged, and only for the MATCHING index.
            assert_eq!(pow(root(x(view), 3), Ex::int(3), &cx), x(view));
            assert!(matches!(
                pow(root(x(view), 3), Ex::int(-5), &cx),
                Ex::Pow(..)
            ));
            // EVEN index needs a certainly-non-negative base, in both directions.
            assert!(matches!(
                pow(root(x(view), 2), Ex::int(-2), &cx),
                Ex::Pow(..)
            ));
        });
    }

    /// Permissive's one rounding lands on whichever of the two floats around the value prints
    /// shorter: a value between 0.1 and the next float, nearer the next, lands on 0.1 (the
    /// nearest, 0.10000000000000002, prints a digit more). A float's own shortest spelling never
    /// moves.
    #[test]
    fn a_landed_literal_takes_the_shorter_of_its_two_floats() {
        let r = Rat::from_big(
            num_bigint::BigInt::from(28823037615171179u64),
            num_bigint::BigInt::from(288230376151711744u64),
        )
        .unwrap();
        assert_eq!(r.to_f64_nearest(), 0.10000000000000002);
        assert_eq!(lossy_literal(&r), Rat::parse_decimal("0.1"));
        assert_eq!(lossy_literal(&Rat::parse_decimal("0.1").unwrap()), None);
        assert_eq!(
            lossy_literal(&Rat::parse_decimal("0.10000000000000002").unwrap()),
            None
        );
    }

    /// ONE PRICE FOR A SIGN: one bit wherever it sits -- on a literal, on a product's
    /// coefficient beside other factors, or a negation standing alone (a sign, not a product,
    /// so no `Mul` head). `+1` is no sign, and a reciprocal is division, not a sign.
    #[test]
    fn a_sign_costs_one_bit_wherever_it_sits() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let c = |e: &Ex| super::complexity(e, view);
            let (x_, y_) = (x(view), y(view));
            let neg = |e: Ex| mul(vec![Ex::int(-1), e], &cx);
            let sin = |e: Ex| fun(view.intern("sin"), vec![e], &cx);
            // a negation standing alone: its operand and one bit
            assert_eq!(c(&neg(x_.clone())), c(&x_) + 1_000);
            assert_eq!(c(&neg(x_.clone())), 7_000);
            // the term of a difference
            let diff = add(vec![x_.clone(), neg(y_.clone())], &cx);
            let sum = add(vec![x_.clone(), y_.clone()], &cx);
            assert_eq!(c(&diff), c(&sum) + 1_000);
            // the -1 coefficient of a product beside other factors
            let prod = mul(vec![x_.clone(), y_.clone()], &cx);
            assert_eq!(c(&neg(prod.clone())), c(&prod) + 1_000);
            // the sign of a literal, a coefficient and an exponent: unchanged, one bit each
            assert_eq!(c(&Ex::int(-2)), c(&Ex::int(2)) + 1_000);
            let three_x = mul(vec![Ex::int(3), x_.clone()], &cx);
            let minus_three_x = mul(vec![Ex::int(-3), x_.clone()], &cx);
            assert_eq!(c(&minus_three_x), c(&three_x) + 1_000);
            assert_eq!(
                c(&pow(x_.clone(), Ex::int(-2), &cx)),
                c(&pow(x_.clone(), Ex::int(2), &cx)) + 1_000
            );
            // a reciprocal is division, not a sign
            assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 9_000);
            // so the two placements of a constant's sign cost the same, and the sign key
            // decides: the sign stays on the constant's literal
            let outside = Ex::Mul(vec![
                Ex::int(-1),
                x_.clone(),
                Ex::Fun(view.intern("sin"), vec![Ex::int(3)]),
            ]);
            let inside = Ex::Mul(vec![
                x_.clone(),
                Ex::Fun(view.intern("sin"), vec![Ex::int(-3)]),
            ]);
            assert_eq!(c(&outside), c(&inside));
            let built = mul(vec![Ex::int(-1), x_.clone(), sin(Ex::int(3))], &cx);
            assert_eq!(built, inside);
        });
    }

    #[test]
    fn complexity_weight_table() {
        // The mu weight table in MILLI-BITS, at the SYMBOL TABLE: a leaf 6
        // bits, Add/Mul/Pow 3, Pi/E 4, an elementary head 6, a transcendental head 8,
        // an infinity 8 -- where mu' charged a flat 8 for every one of them.
        // Unchanged by the table: a priced literal pays ONE selector bit + its cheaper codeword
        // (rational L(p)[+L(q)] vs decimal-scientific max(2,L(m))+L(|k|)) + a bit for a
        // negative sign, mantissa floored at two bits; magnitude-1 coefficient and
        // exponent slots stay free (a bare sign writes no number, so no selector
        // either); <constant> = MU_FREE_WORST_CASE_F64 = 67 bits (the sup-construction
        // re-derived under the mu' codebook -- DERIVED, not chosen).
        with_view(|view| {
            let cx = Cx::bare(view);
            let c = |e: &Ex| super::complexity(e, view);
            // x - y = 16: Add(3) + x(6) + the negation of y (a sign, one bit, and no Mul
            // head) + y(6).
            // x*y = 15; x/y = 18 (the Pow(y, -1) wrapper at 3, exponent -1 free).
            let x_ = x(view);
            let y_ = y(view);
            let x_minus_y = add(
                vec![x_.clone(), mul(vec![Ex::int(-1), y_.clone()], &cx)],
                &cx,
            );
            assert_eq!(c(&x_minus_y), 16_000);
            assert_eq!(c(&mul(vec![x_.clone(), y_.clone()], &cx)), 15_000);
            let x_over_y = mul(vec![x_.clone(), pow(y_.clone(), Ex::int(-1), &cx)], &cx);
            assert_eq!(c(&x_over_y), 18_000);
            // 2x = 11.585 (the coefficient PAYS selector + its bits: 3 + 2.585 + 6),
            // x^2 = 11.585, 1/x = 9, sin(x) = 12. The literal floor is ONE bit,
            // so cost(2) = 2.585 clears it and no longer ties cost(3).
            assert_eq!(c(&mul(vec![Ex::int(2), x_.clone()], &cx)), 11_585);
            assert_eq!(c(&pow(x_.clone(), Ex::int(2), &cx)), 11_585);
            // THE EXPONENTS ARE ORDERED AGAIN. Under the two-bit floor `x^2`, `x^3` and
            // `x^4` all priced alike, because the clamp swallowed cost(2), cost(3) and
            // cost(4) together -- the measure could not tell a square from a cube.
            assert!(
                c(&pow(x_.clone(), Ex::int(2), &cx)) < c(&pow(x_.clone(), Ex::int(3), &cx))
                    && c(&pow(x_.clone(), Ex::int(3), &cx)) < c(&pow(x_.clone(), Ex::int(4), &cx))
            );
            assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 9_000);
            assert_eq!(c(&fun(view.intern("sin"), vec![x_.clone()], &cx)), 12_000);
            // x^-2 costs as x^2 plus the FULL sign bit (the -2 exponent is a priced
            // literal: selector + L(2) + sign = 1 + 1.585 + 1, all of it clear of the
            // one-bit floor; under the two-bit floor the clamp absorbed all but 0.585).
            assert_eq!(c(&pow(x_.clone(), Ex::int(-2), &cx)), 12_585);
            // Mul(3) + <constant>(67_000) + x(6) = 76_000. The free constant is a LITERAL
            // price, so the table does not touch it: it now outweighs its own structure by
            // more than it did, which is the direction the ordering wants.
            assert_eq!(
                c(&mul(vec![Ex::ANON, x_.clone()], &cx)),
                mu_mul() + MU_FREE_WORST_CASE_F64 + mu_leaf()
            );
            assert_eq!(mu_free(), 67 * MU_MILLI);
            assert_eq!(c(&Ex::int(-5)), 4_585); // selector + L(5) + 1 for the sign
                                                // x8 + 1.2*x3 = 3 + 6 + (3 + cost(6/5) + 6).
                                                // cost(6/5) = selector + min(L(6)+L(5) = 5.392, max(2,L(12))+L(1) = 4.700)
                                                // = 5.700: the decimal-scientific codeword takes it (the fraction-only
                                                // rule is gone).
            let t = mul(vec![Ex::Num(Rat::new(6, 5).unwrap()), y_.clone()], &cx);
            assert_eq!(c(&add(vec![x_.clone(), t], &cx)), 23_700);
            // The stage-2 point, in one line: an E factor undercuts its f64 image --
            // and by MORE than before, since the table cheapened the structure the
            // symbolic spelling is made of and left the image's 56.272 bits alone.
            let sym = c(&mul(vec![Ex::E, x_.clone()], &cx));
            let mat = c(&mul(
                vec![
                    Ex::Num(Rat::new(2718281828459045, 1000000000000000).unwrap()),
                    x_.clone(),
                ],
                &cx,
            ));
            assert!(sym == 13_000 && sym < mat, "{sym} vs {mat}");
        });
    }

    #[test]
    fn canon_is_idempotent_on_samples() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let samples = vec![
                add(vec![x(view), y(view), x(view)], &cx),
                mul(vec![Ex::int(2), x(view), lg.clone(), x(view)], &cx),
                pow(add(vec![x(view), Ex::int(1)], &cx), Ex::int(2), &cx),
                add(vec![Ex::PosInf, lg], &cx),
                mul(
                    vec![Ex::int(0), fun(view.intern("log"), vec![y(view)], &cx)],
                    &cx,
                ),
            ];
            for s in samples {
                assert_eq!(canon(s.clone(), &cx), s, "canon not idempotent on {s:?}");
            }
        });
    }

    /// PIN: `mu_numeric_str` reads fraction-shaped beyond-`Rat`
    /// spellings through the component model (L(p) + L(q) from the digit strings),
    /// not through the decimal path with the slash counted as a significand digit.
    /// The boundary properties that matter: far above every in-range literal, monotone
    /// in either component's digit count, and the decimal path untouched.
    #[test]
    fn mu_numeric_str_fraction_component_model() {
        let big41 = "1".repeat(41); // beyond i128 (39 digits max)
        let frac = format!("{big41}/7");
        let c = mu_numeric_str(&frac);
        // Component model, both components PAID (the implicit-denominator discount applies
        // only when there is no genuine denominator), plus the mu' selector bit; the
        // denominator 7 offers no decimal codeword: 1 + L(111...1) + L(7) = 137.031 bits.
        assert_eq!(c, 137_031);
        // Far above any in-range literal; the boundary that matters is the ~105-BIT
        // 52-bit dyadics.
        assert!(c > 105 * MU_MILLI);
        // Monotone in each component.
        assert!(mu_numeric_str(&format!("{big41}1/7")) > c);
        assert!(mu_numeric_str(&format!("{big41}/71")) > c);
        // The sign costs exactly one bit here, as it does in range;
        // a zero numerator floors at the one-bit codeword floor (+ the selector) --
        // the one-bit floor, mirrored across the i128 boundary.
        assert_eq!(mu_numeric_str(&format!("-{big41}/7")), c + MU_MILLI);
        assert_eq!(mu_numeric_str("0/7"), 2 * MU_MILLI);
        // The decimal path is case-insensitive in the exponent marker.
        assert_eq!(
            mu_numeric_str("4.159653437657682e-35"),
            mu_numeric_str("4.159653437657682E-35")
        );
        // The second codeword RESTORES the minimum over codes: `1e-40` takes its
        // decimal-scientific codeword, selector + max(floor, L(1)) + L(40) = 7.358 bits
        // under the one-bit floor, not the 133.880 the fraction-only rule charged. The boundary property this test
        // exists for -- the token pricer agrees with `mu_rat`'s rule, and the SAME value
        // prices the SAME from both spellings -- now holds EXACTLY: the fraction-only code
        // pinned a one-milli-bit seam between these two spellings (133.880 vs 133.879,
        // the rounded-L10 residue of the fraction code); both now take the decimal
        // codeword through the same `l_millibits(40)` and the seam is CLOSED.
        let e_notation = mu_numeric_str("1e-40");
        let as_fraction = mu_numeric_str(&format!("1/1{}", "0".repeat(40)));
        assert_eq!(e_notation, 7_358);
        assert_eq!(as_fraction, 7_358);
        assert_eq!(
            e_notation, as_fraction,
            "the two spellings of 1e-40 drifted apart: {e_notation} vs {as_fraction}"
        );
    }

    /// c_free' derivation: `MU_FREE_WORST_CASE_F64` is the CEILED sup of mu'
    /// over f64 round-trip spellings plus the sign bit -- recomputed here from the
    /// engine's own token pricer, so the published constant cannot drift from the code
    /// that defines it. The sweep: every f64 exponent field at six mantissa patterns
    /// (binade edges and mid-grid probes), plus the recorded argmax
    /// `8.9002954340287245e-308` (found offline by ulp-walking every binade top in the
    /// boundary decades; the analytic feasibility cap -- a shortest repr's mantissa is
    /// below `10 * 2^53` and its scale within +-343 -- bounds the sup under 64.92
    /// bits, so the 67-bit ceiling is robust to argmax misses).
    #[test]
    fn c_free_prime_derivation() {
        let price = |x: f64| -> u64 {
            let tok = format!("{x:e}");
            mu_numeric_str(&tok) - MU_MILLI // the codeword min, without the selector
        };
        let mantissas: [u64; 6] = [
            0,                     // the binade bottom (a power of two)
            1,                     // one ulp up
            0x000F_FFFF_FFFF_FFFF, // the binade top
            0x0008_0000_0000_0000,
            0x0005_5555_5555_5555,
            0x000A_AAAA_AAAA_AAAA,
        ];
        let mut sup = 0u64;
        for exp in 0..2047u64 {
            for m in mantissas {
                let x = f64::from_bits((exp << 52) | m);
                if x == 0.0 || !x.is_finite() {
                    continue;
                }
                sup = sup.max(price(x));
            }
        }
        // The recorded argmax reproduces the recorded supremum, and nothing in the
        // sweep prices above it (the load-bearing safety direction).
        let rec = price(8.9002954340287245e-308);
        assert_eq!(rec, 64_649);
        assert!(
            sup <= rec,
            "the sweep beat the recorded supremum: {sup} > {rec}"
        );
        assert!(
            sup > rec - 2_000,
            "the sweep never reached the extreme region: {sup}"
        );
        // + the sign bit + the selector bit, ceiled: the published constant, re-derived.
        let derived = (rec + MU_MILLI + MU_MILLI).div_ceil(MU_MILLI) * MU_MILLI;
        assert_eq!(derived, MU_FREE_WORST_CASE_F64);
        assert_eq!(mu_free(), 67 * MU_MILLI);
    }

    /// The astronomic knee, both halves. (1) PRICING: the linear schedule `|scale| * log2(10)` exhausts
    /// u64 at |scale| ~ 5.553e15, and saturating there collided every larger scale --
    /// and every beyond-i64 exponent -- on the single price `u64::MAX`, so mu could not
    /// tell `1e5553000000000000` from `1e10^30`. The plan's acceptance line is asserted
    /// verbatim, then a strict ladder across the whole astronomic range in both
    /// exponent signs. (2) ACCUMULATION: a `u64::MAX`-priced leaf made the tree sums in
    /// `complexity()` overflow -- a debug-build panic and a release-build WRAP, and a
    /// wrapped sum prices a composite below its own parts, which is the direction that
    /// licenses false mu-descents. Raw `Ex` shells (no canon) exercise the sums
    /// directly.
    #[test]
    fn astronomic_literals_stay_ordered_and_sums_never_wrap() {
        // (1) the plan's acceptance, verbatim
        assert_ne!(mu_numeric_str("1e5553000000000000"), u64::MAX);

        // strictly ordered across the old ceiling, the i64 boundary, and into
        // digit-string-only exponents; both signs ride the same schedule
        let huge = format!("1e1{}", "0".repeat(30));
        let huge_neg = format!("1e-1{}", "0".repeat(30));
        let ladder = [
            "1e308", // f64's edge: the exact linear regime
            "1e4294967296",
            "1e5553000000000000", // the old saturating_mul ceiling
            "1e9223372036854775807",
            "1e99999999999999999999", // exponent past i64: digit string only
            huge.as_str(),
        ];
        for w in ladder.windows(2) {
            assert!(
                mu_numeric_str(w[0]) < mu_numeric_str(w[1]),
                "mu not strictly increasing: mu({}) = {} !< mu({}) = {}",
                w[0],
                mu_numeric_str(w[0]),
                w[1],
                mu_numeric_str(w[1])
            );
        }
        let neg_ladder = [
            "1e-308",
            "1e-4294967296",
            "1e-5553000000000000",
            "1e-99999999999999999999",
            huge_neg.as_str(),
        ];
        for w in neg_ladder.windows(2) {
            assert!(
                mu_numeric_str(w[0]) < mu_numeric_str(w[1]),
                "mu not strictly increasing on the denominator side: {} vs {}",
                w[0],
                w[1]
            );
        }

        // (2) no wrap: a composite always prices at or above every part, and never
        // below a bare variable, whatever its leaves cost
        with_view(|view| {
            let a = Ex::Leaf(view.intern("1e5553000000000000"));
            let b = Ex::Leaf(view.intern("1e9223372036854775807"));
            let (ca, cb) = (complexity(&a, view), complexity(&b, view));
            let bare = complexity(&x(view), view);
            let shells = [
                Ex::Add(vec![a.clone(), b.clone()]),
                Ex::Mul(vec![a.clone(), b.clone()]),
                Ex::Pow(Box::new(a.clone()), Box::new(b.clone())),
                Ex::Fun(view.intern("sin"), vec![a.clone(), b.clone()]),
            ];
            for s in &shells {
                let cs = complexity(s, view);
                assert!(
                    cs >= ca.max(cb),
                    "composite priced below a part: {cs} < max({ca}, {cb}) on {s:?}"
                );
                assert!(cs >= bare, "composite priced below a bare variable: {cs}");
            }
            // The saturation CROSSING is no longer materializable (mu'): under
            // the two-codeword pricing every leaf is bounded by ~3.32 bits per DIGIT
            // of its own print (the scientific codeword prices the scale as an
            // integer, so the old ~1.4e13-milli-bit astronomic leaves now price a few
            // dozen bits -- this leaf is {ca} ~ 55 bits), and reaching u64::MAX would
            // take a tree whose serialization is petabytes. The old crossing pin
            // (`vec![leaf; u64::MAX / ca]`, ~1.3M members under the fraction-only code) would today demand
            // ~3e14 members and is deleted as physically unreachable; the
            // `saturating_add` accumulation stays as pure defense-in-depth. What
            // remains checkable at reachable scale: totals are exact (no wrap, no
            // saturation) and strictly monotone in member count on the largest
            // feasible astronomic bags.
            let q = 100_000usize;
            let mut prev = 0u64;
            for n in [q - 1, q, q + 1] {
                let cs = complexity(&Ex::Add(vec![a.clone(); n]), view);
                assert!(
                    cs > prev,
                    "the Add total failed to grow at {n} members: {prev} -> {cs}"
                );
                assert_eq!(
                    cs,
                    mu_add() + ca * n as u64,
                    "an astronomic Add bag mispriced at {n} members"
                );
                prev = cs;
            }
            for bag in [
                Ex::Mul(vec![a.clone(); q]),
                Ex::Fun(view.intern("sin"), vec![a.clone(); q]),
            ] {
                let cs = complexity(&bag, view);
                assert!(
                    cs >= ca && cs >= bare,
                    "a large astronomic bag priced below its parts: {cs}"
                );
            }
        });
    }

    /// ONE owner for the odd-function literal sign, on every construction
    /// route. Before the hoist, `term_join` (the Add-collector's rebuild) fused
    /// `-5 * sin(2)` to `5 * sin(-2)` while `mul()`'s direct assembly did not -- two
    /// fixpoints for one value, differing by construction history (the invariance
    /// defect class). And the collector could not SPLIT the sign back out, so
    /// `sin(-2) + sin(2)` needed a render/re-parse cycle to cancel.
    #[test]
    fn odd_literal_sign_is_route_independent() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let sin = view.intern("sin");
            let sin2 = fun(sin, vec![Ex::int(2)], &cx);
            let sin_neg2 = fun(sin, vec![Ex::int(-2)], &cx);

            // The two routes must agree, and on the ruled direction (the
            // literal owns the sign): -5 * sin(2) spells 5 * sin(-2).
            let via_mul = mul(vec![Ex::int(-5), sin2.clone()], &cx);
            let via_join = term_join(Coef::of(&Rat::int(-5)), sin2.clone(), &cx);
            assert_eq!(
                via_mul, via_join,
                "mul() and term_join disagree on the sign"
            );
            assert_eq!(
                via_mul,
                mul(vec![Ex::int(5), sin_neg2.clone()], &cx),
                "the fused spelling is not the canonical one"
            );
            // The other example: -1 * sin(2) files as sin(-2) (the Mul node itself
            // dissolves).
            assert_eq!(mul(vec![Ex::int(-1), sin2.clone()], &cx), sin_neg2);

            // The collector conflates the pair IN ONE PASS -- sin(-2)
            // and sin(2) share a key with opposite coefficients and cancel exactly.
            assert_eq!(
                add(vec![sin_neg2.clone(), sin2.clone()], &cx),
                Ex::int(0),
                "sin(-2) + sin(2) must cancel at collection, not via re-parse"
            );

            // The collision guard: flipping sin(-2) beside sin(2) would mint an
            // unmerged duplicate-base bag; the orbit must skip such masks and keep
            // the entry spelling stable (idempotent).
            let prod = mul(vec![sin_neg2.clone(), sin2.clone()], &cx);
            assert_eq!(canon(prod.clone(), &cx), prod, "ground product not stable");
        });
    }

    /// Resolved-string rendering for CROSS-VIEW comparisons: two views may intern the
    /// same names to different ids, so `Ex` equality cannot compare their outputs --
    /// the resolved shape can.
    fn ex_str(e: &Ex, view: &TokenView) -> String {
        match e {
            Ex::Num(r) => format!("{r:?}"),
            Ex::Pi => "pi".into(),
            Ex::E => "e".into(),
            Ex::PosInf => "inf".into(),
            Ex::NegInf => "-inf".into(),
            Ex::NaN => "nan".into(),
            Ex::Const(_) => "C".into(),
            Ex::Leaf(t) => view.resolve_owned(*t),
            Ex::Fun(f, args) => {
                let a: Vec<String> = args.iter().map(|x| ex_str(x, view)).collect();
                format!("{}({})", view.resolve_owned(*f), a.join(","))
            }
            Ex::Pow(b, x) => format!("pow({},{})", ex_str(b, view), ex_str(x, view)),
            Ex::Add(v) => {
                let a: Vec<String> = v.iter().map(|x| ex_str(x, view)).collect();
                format!("add[{}]", a.join(","))
            }
            Ex::Mul(v) => {
                let a: Vec<String> = v.iter().map(|x| ex_str(x, view)).collect();
                format!("mul[{}]", a.join(","))
            }
        }
    }

    /// The sign-placement tie-break is a function of CONTENT, never of token
    /// interning order. The mu-tied orbit of `-1 * (a-b) * (c-d)` (its two mirror
    /// pairings price alike) must land on the same resolved spelling
    /// when the same four opaque literals are interned in OPPOSITE orders and the
    /// factors arrive swapped -- an older `ex_struct_cmp` compared `Leaf`/`Fun`
    /// tokens by raw id and flipped the mirror decision with the interning (extreme
    /// fuzz rows 829655/866090, the two regressions at 1M rows).
    #[test]
    fn f72_tiebreak_is_interning_independent() {
        fn build(intern_first: &[&str], swap_factors: bool) -> String {
            with_view(|view| {
                for s in intern_first {
                    view.intern(s);
                }
                let cx = Cx::bare(view);
                let leaf = |s: &str| Ex::Leaf(view.intern(s));
                let neg = |e: Ex, cx: &Cx| mul(vec![Ex::int(-1), e], cx);
                let ab = add(vec![leaf("lit_a"), neg(leaf("lit_b"), &cx)], &cx);
                let cd = add(vec![leaf("lit_c"), neg(leaf("lit_d"), &cx)], &cx);
                let items = if swap_factors {
                    vec![Ex::int(-1), cd, ab]
                } else {
                    vec![Ex::int(-1), ab, cd]
                };
                ex_str(&mul(items, &cx), view)
            })
        }
        let base = build(&["lit_a", "lit_b", "lit_c", "lit_d"], false);
        assert_eq!(base, build(&["lit_d", "lit_c", "lit_b", "lit_a"], false));
        assert_eq!(base, build(&["lit_c", "lit_d", "lit_a", "lit_b"], true));
        assert_eq!(base, build(&["lit_b", "lit_a", "lit_d", "lit_c"], true));
    }

    /// With the rational content PARTITIONED across several `Num` members (the
    /// product is not representable), the sign has exactly ONE canonical host, a function
    /// of the value -- not of which member carried it on arrival. Before the
    /// sign-factored accumulation, `-N * P` kept the sign on the `-N` partial while
    /// the negation of `N * P` hosted it on the coefficient slot: two stable states
    /// for one value (the extreme lane's idempotence family, 88 -> 30 rows).
    #[test]
    fn f72_partition_sign_host_is_route_invariant() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // 0.9999999999999999 and 1e-300: the product's denominator 10^316 exceeds the
            // f64 domain's 2^1022 (phase 2c; it used to be an i128 overflow), so the bag keeps
            // BOTH as members.
            let n = Rat::new(9999999999999999, 10000000000000000).unwrap();
            let p = Rat::from_big(
                num_bigint::BigInt::from(1),
                num_traits::pow(num_bigint::BigInt::from(10), 300),
            )
            .unwrap();
            let route_signed_member = mul(
                vec![Ex::Num(n.checked_neg().unwrap()), Ex::Num(p.clone())],
                &cx,
            );
            let route_negated_bag =
                negate_term(&mul(vec![Ex::Num(n.clone()), Ex::Num(p.clone())], &cx), &cx).unwrap();
            let route_other_host =
                term_join(Coef::of(&p.checked_neg().unwrap()), Ex::Num(n.clone()), &cx);
            let route_swapped = mul(vec![Ex::Num(p), Ex::Num(n.checked_neg().unwrap())], &cx);
            assert_eq!(route_signed_member, route_negated_bag);
            assert_eq!(route_signed_member, route_other_host);
            assert_eq!(route_signed_member, route_swapped);
            // The chosen state is a fixpoint of its own constructor.
            let Ex::Mul(v) = route_signed_member.clone() else {
                panic!("partition bag expected, got {route_signed_member:?}");
            };
            assert_eq!(mul(v, &cx), route_signed_member);
        });
    }

    /// A term's coefficient is its whole literal partition (`Coef`). With only the first member
    /// as the coefficient, `-N*N*x` kept a key of its own beside `N*N*x`, `N*N*x` and `N*y`
    /// shared the content `N`, and a sum's content and like terms depended on which terms a
    /// construction saw together: a state and the parse of its own print differed.
    #[test]
    fn refused_partition_is_one_coefficient() {
        for mode in [RuleMode::Default, RuleMode::Permissive] {
            with_view(|view| {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                // 10^300 * 10^300 is beyond 2^1022: refused in the f64 domain.
                let n = Rat::from_big(
                    num_traits::pow(num_bigint::BigInt::from(10), 300),
                    num_bigint::BigInt::from(1),
                )
                .unwrap();
                let leaf = |s: &str| Ex::Leaf(view.intern(s));
                let (x, y, z) = (leaf("x1"), leaf("x2"), leaf("x3"));
                let nn = |e: Ex| mul(vec![Ex::Num(n.clone()), Ex::Num(n.clone()), e], &cx);
                let neg = |e: &Ex| negate_term(e, &cx).unwrap();
                let t = nn(x.clone());
                assert!(
                    matches!(&t, Ex::Mul(v) if v.len() == 3),
                    "partition bag expected: {t:?}"
                );
                let (c, k) = term_split(t.clone(), view);
                assert_eq!((c.rat(), &k), (None, &x));
                assert_eq!(term_split(neg(&t), view), (c.negated(), x.clone()));
                // The sign is written on the smallest member, not as a `-1` beside them, however
                // it arrives.
                let minus_n = Ex::Num(n.checked_neg().unwrap());
                let written = Ex::Mul(vec![minus_n.clone(), Ex::Num(n.clone()), x.clone()]);
                assert_eq!(neg(&t), written);
                assert_eq!(mul(vec![Ex::int(-1), t.clone()], &cx), written);
                assert_eq!(nn(mul(vec![Ex::int(-1), x.clone()], &cx)), written);
                assert_eq!(mul(vec![written.clone()], &cx), written);
                assert_eq!(
                    mul(vec![Ex::Num(n.clone()), minus_n, x.clone()], &cx),
                    written
                );
                // Like terms cancel, and add over their shared member: {2N, N} - {N, N} = {N, N}.
                assert_eq!(add(vec![t.clone(), neg(&t)], &cx), Ex::Num(Rat::ZERO));
                let two_t = mul(vec![Ex::int(2), t.clone()], &cx);
                assert_eq!(add(vec![two_t, neg(&t)], &cx), t);
                // The grouping does not decide: flat, a sub-sum first, and factored.
                let flat = add(vec![nn(x.clone()), nn(y.clone()), z.clone()], &cx);
                let nested = add(
                    vec![z.clone(), add(vec![nn(x.clone()), nn(y.clone())], &cx)],
                    &cx,
                );
                let xy = add(vec![x.clone(), y.clone()], &cx);
                let factored = add(vec![nn(xy.clone()), z.clone()], &cx);
                assert_eq!(flat, nested);
                assert_eq!(flat, factored);
                // A member of a partition is no content shared with a one-member coefficient.
                let mixed = add(
                    vec![nn(x.clone()), mul(vec![Ex::Num(n.clone()), y.clone()], &cx)],
                    &cx,
                );
                assert!(matches!(&mixed, Ex::Add(_)), "{mixed:?}");
                // The negated pair's content is the partition, its sign the partition's.
                let negated = add(vec![neg(&nn(x.clone())), neg(&nn(y.clone()))], &cx);
                assert_eq!(negated, neg(&nn(xy)));
            });
        }
    }

    /// Integers beyond 128 bits are integers. The map of the 128-bit boundary found the sites
    /// that read "does not fit i128" as "is not an integer", each unsound once such an integer
    /// reaches it. The f64 domain knows no parity beyond 2^53, and every site must then refuse
    /// (in `f64`: permissive composes powers naively, ignoring the sign).
    #[test]
    fn big_integers_keep_their_integer_reading() {
        with_view(|view| big_integer_sites(view, RuleMode::Default));
    }

    fn big_integer_sites(view: &TokenView, mode: RuleMode) {
        {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let big =
                |p: num_bigint::BigInt| Rat::from_big(p, num_bigint::BigInt::from(1)).unwrap();
            let ten40 = big(num_traits::pow(num_bigint::BigInt::from(10), 40));
            let odd = ten40.checked_add(&Rat::ONE).unwrap(); // 10^40 + 1
            assert!(ten40.small_int().is_none() && ten40.is_integer());
            assert!(ten40.parity().is_none() && odd.parity().is_none());
            let rootn = view.intern("rootn");

            // 1. rootn(x, 10^40) is a root with an integer index, not the invalid operation.
            let r = fun(rootn, vec![x(view), Ex::Num(ten40.clone())], &cx);
            assert!(matches!(&r, Ex::Fun(..)), "{r:?}");

            // 2. Pow-of-pow: (x^(10^40))^(1/3) must not become x^(10^40/3). The inner power is
            //    an even power, defined and non-negative at x < 0, where x^(10^40/3) is NaN.
            let third = Rat::new(1, 3).unwrap();
            let inner = pow(x(view), Ex::Num(ten40.clone()), &cx);
            let composed = pow(inner, Ex::Num(third.clone()), &cx);
            let wrong = pow(x(view), Ex::Num(ten40.checked_mul(&third).unwrap()), &cx);
            assert_ne!(composed, wrong);

            // 3. Power through an even root: rootn(x^(2(10^40+1)), 4) must not become
            //    x^((10^40+1)/2), for the same reason.
            let k = odd.checked_mul(&Rat::int(2)).unwrap();
            let through = fun(
                rootn,
                vec![pow(x(view), Ex::Num(k.clone()), &cx), Ex::int(4)],
                &cx,
            );
            let wrong = pow(
                x(view),
                Ex::Num(k.checked_mul(&Rat::new(1, 4).unwrap()).unwrap()),
                &cx,
            );
            assert_ne!(through, wrong);
        }
    }

    /// Zero is priced by its fraction codeword alone, at the floor: it has no decimal codeword and
    /// no reciprocal.
    #[test]
    fn zero_has_the_fraction_codeword_alone() {
        assert_eq!(
            mu_rat_codeword_totals(&Rat::ZERO),
            (MU_RAT_FLOOR, None, None)
        );
        assert_eq!(mu_rat_exact(&Rat::ZERO), MU_MILLI + MU_RAT_FLOOR);
    }

    /// Every number is priced as it is written: the long exact product costs its digits, and the
    /// 16-digit decimal of 1/3 costs its 16 digits.
    #[test]
    fn a_literal_is_priced_as_it_is_written() {
        let long = {
            let _exact = crate::ac::rat::number_domain(false);
            Rat::parse_decimal("0.0766541268471677307861497420516056347394916180597539647375669841")
                .unwrap()
        };
        let short = Rat::shortest_reading_as(long.to_f64_nearest()).unwrap();
        assert_eq!(mu_rat(&long), mu_rat_exact(&long));
        assert!(mu_rat(&short) < mu_rat(&long));
        let third = Rat::parse_decimal("0.3333333333333333").unwrap();
        assert_eq!(mu_rat(&third), mu_rat_exact(&third));
        assert!(mu_rat(&Rat::new(1, 3).unwrap()) < mu_rat(&third));
        for t in ["1000", "0.2", "-7.5", "1e-40", "2.177697277405848"] {
            let r = Rat::parse_decimal(t).unwrap();
            assert_eq!(mu_rat(&r), mu_rat_exact(&r), "{t}");
        }
    }

    /// Strict writes a number two literals make as the cheaper of its exact value and float64's
    /// value of the operation, the exact one at an equal price.
    #[test]
    fn strict_writes_the_cheaper_of_the_exact_and_the_float64_number() {
        let d = |t: &str| Rat::parse_decimal(t).unwrap();
        with_view(|view| {
            // the simplify walk's context
            let mut cx = Cx::bare(view);
            cx.fold_f64 = true;
            let prod = |v: Vec<Rat>| literal_partition_product(v, false, &cx);
            let sum = |v: Vec<Rat>| literal_partition_sum(v, false, &cx);
            let a = d("0.30000000000000004");
            assert_eq!(prod(vec![a.clone(), a]), vec![d("0.09000000000000002")]);
            assert_eq!(
                prod(vec![d("0.7234567890123457"), d("1.3821548250604163")]),
                vec![d("0.9999292916561291")],
                "float64's product of the two doubles, not the double nearest the exact product"
            );
            assert_eq!(sum(vec![d("0.1"), d("0.2")]), vec![d("0.3")]);
            assert_eq!(prod(vec![d("0.1"), d("3")]), vec![d("0.3")]);
            // the partition is exact, and only its one combined number is rounded
            assert_eq!(
                prod(vec![Rat::new(1, 3).unwrap(), d("1e300"), d("1e-300")]),
                vec![Rat::new(1, 3).unwrap()]
            );
            // a fully numeric product stays exact: the part is valued whole
            let a = d("0.30000000000000004");
            assert_eq!(
                literal_partition_product(vec![a.clone(), a], true, &cx),
                vec![d("0.0900000000000000240000000000000016")]
            );
            // a conversion keeps the exact product
            let bare = Cx::bare(view);
            let a = d("0.30000000000000004");
            assert_eq!(
                literal_partition_product(vec![a.clone(), a], false, &bare),
                vec![d("0.0900000000000000240000000000000016")]
            );
            let mut p = Cx::bare(view);
            p.mode = RuleMode::Permissive;
            let a = d("0.30000000000000004");
            assert_eq!(
                literal_partition_product(vec![a.clone(), a], false, &p),
                vec![d("0.0900000000000000240000000000000016")],
                "permissive rounds where the number lands (`land`)"
            );
        });
        // no float64 value replaces an exact one past the range of the normal doubles
        let third = Rat::new(1, 3).unwrap();
        assert_eq!(strict_number(third.clone(), f64::INFINITY), third);
        assert_eq!(strict_number(third.clone(), 0.0), third);
        assert_eq!(strict_number(third.clone(), 1e-310), third);
        // an integer stays exact
        let big = Rat::int(12157665459056928801);
        assert_eq!(strict_number(big.clone(), 1.2157665459056929e19), big);
        // a written literal of more digits than its double needs, in the simplify walk
        with_view(|view| {
            let mut cx = Cx::bare(view);
            cx.fold_f64 = true;
            assert_eq!(
                strict_literal(d("0.1234567890123456789"), &cx),
                d("0.12345678901234568")
            );
            assert_eq!(
                strict_literal(d("0.1234567890123456789"), &Cx::bare(view)),
                d("0.1234567890123456789"),
                "a conversion keeps the literal as written"
            );
            assert_eq!(strict_literal(d("0.1"), &cx), d("0.1"));
            let mut p = Cx::bare(view);
            p.mode = RuleMode::Permissive;
            assert_eq!(
                strict_literal(d("0.1234567890123456789"), &p),
                d("0.1234567890123456789"),
                "permissive rounds where a literal lands (`land`)"
            );
        });
    }

    /// The shortest decimal reads back as the float.
    #[test]
    fn the_shortest_decimal_reads_back_as_its_double() {
        for y in [
            0.1,
            1.0 / 3.0,
            std::f64::consts::PI,
            1e-300,
            5e-324,
            f64::MAX,
            -0.75,
            1e22,
            0.45445993041202365,
        ] {
            let d = Rat::shortest_reading_as(y).unwrap();
            assert_eq!(d.to_f64_nearest(), y, "{y:?}");
        }
        assert_eq!(Rat::shortest_reading_as(f64::INFINITY), None);
    }

    /// A literal held in pieces costs at most what its value costs as one number, alone or
    /// beside other factors.
    #[test]
    fn a_literal_held_in_pieces_costs_at_most_its_value() {
        with_view(|view| {
            let cx = Cx::bare(view); // f64's number domain
            let b = Rat::parse_decimal("9.876543210987654e-10").unwrap();
            let inv_b = match pow(Ex::Num(b.clone()), Ex::int(-1), &cx) {
                Ex::Num(r) => r,
                other => panic!("not a number: {other:?}"),
            };
            let exact = {
                let _exact = crate::ac::rat::number_domain(false);
                Rat::parse_decimal("1.2345678901234567e-305")
                    .unwrap()
                    .checked_mul(&inv_b)
                    .unwrap()
            };
            // a's denominator, 10^321, passes 2^1022: a stays a leaf as written, beside 1/b
            let leaf = Ex::Leaf(view.intern("1.2345678901234567e-305"));
            let quotient = mul(
                vec![leaf.clone(), pow(Ex::Num(b.clone()), Ex::int(-1), &cx)],
                &cx,
            );
            assert_eq!(
                quotient,
                Ex::Mul(vec![Ex::Num(inv_b.clone()), leaf.clone()])
            );
            // the product's head and its pieces: the leaf as written and 1/b, priced as its
            // divisor (the reciprocal codeword), below the one exact number
            let pieces = mu_mul() + mu_numeric_str("1.2345678901234567e-305") + mu_rat(&inv_b);
            assert!(pieces < mu_rat(&exact));
            assert_eq!(complexity(&quotient, view), pieces);
            // A product deciding opposite sums beside a piece prices its candidates against
            // bounds below their prices (`decide_opposite_sums`' debug assertion).
            let s = add(
                vec![x(view), Ex::Num(Rat::parse_decimal("-9.6").unwrap())],
                &cx,
            );
            let t = mul(vec![Ex::int(-1), s.clone()], &cx);
            let p = mul(vec![Ex::int(3), leaf, s, t], &cx);
            let Ex::Mul(v) = &p else {
                panic!("not a product: {p:?}")
            };
            assert_eq!(v.iter().filter(|f| is_piece(f, view)).count(), 2, "{p:?}");
            let literal = bag_literal_value(v, false, view).unwrap();
            let rest: u64 = v
                .iter()
                .filter(|f| !is_piece(f, view))
                .map(|f| complexity(f, view))
                .sum();
            let structural: u64 = v
                .iter()
                .filter(|f| is_piece(f, view))
                .map(|f| complexity(f, view))
                .sum();
            assert_eq!(
                complexity(&p, view),
                mu_mul() + structural.min(mu_rat(&literal)) + rest
            );
        });
    }

    #[test]
    fn an_odd_head_of_nan_trades_no_sign_and_an_even_head_over_it_terminates() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let sin_nan = fun(view.intern("sin"), vec![Ex::NaN], &cx);
            // NaN is no negatable argument: a negation leaves it as it is, so the odd head has
            // no other member in its class and no sign trades into it.
            assert_eq!(multiple_literal(&Ex::NaN), None);
            assert_eq!(sign_trade_flip(&sin_nan, &cx), None);
            assert_eq!(
                mul(vec![Ex::int(-2), sin_nan.clone()], &cx),
                Ex::Mul(vec![Ex::int(-2), sin_nan.clone()])
            );
            // `sin(nan) / -inf` beside the written negative zero, under each even head: the arm
            // drops the sign and does not move again.
            let neg_zero = Ex::Leaf(view.intern(NEG_ZERO));
            for even in ["abs", "cos", "cosh"] {
                for coefficient in [1, -1] {
                    let arg = mul(
                        vec![Ex::int(coefficient), neg_zero.clone(), sin_nan.clone()],
                        &cx,
                    );
                    let built = fun(view.intern(even), vec![arg], &cx);
                    let Ex::Fun(_, a) = &built else {
                        panic!("not a call: {built:?}")
                    };
                    assert_eq!(even_argument(&a[0], &cx), None, "{built:?}");
                }
            }
        });
    }
}
