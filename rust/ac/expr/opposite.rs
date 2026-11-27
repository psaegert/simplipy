//! OPPOSITE SUMS: a product's classes {B, -B}, the mass tables that decide what each class keeps at
//! its zero, and the tests of a sum's negation.

use super::*;

/// Where a product's sign lives once its factors are collected (`mul()`'s tail): the rational
/// coefficient, a bare `Const` (whose refit absorbs any sign), or a bare infinity (which carries
/// it).
pub(super) enum SignHost {
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
pub(super) enum HostKey {
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
pub(super) fn collect_opposite_sums(host: SignHost, raw: Vec<Ex>, out: Vec<Ex>, cx: &Cx) -> Ex {
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
    let key = (FLIP_MEMO.with(|m| m.borrow().is_some()) && !memo_free(cx))
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
pub(super) fn holds_joinable_roots(v: &[Ex], cx: &Cx) -> bool {
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
pub(super) fn holds_opposite_sums(factors: &[Ex], cx: &Cx) -> bool {
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
pub(super) fn same_collection_base(f: &Ex, g: &Ex, cx: &Cx) -> bool {
    cmp_ex(collection_base(f, cx), collection_base(g, cx), cx.view) == Ordering::Equal
}
