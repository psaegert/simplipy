//! `max` and `min` as lattice bags ([`lattice`]).

use super::*;

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
    let _exact = crate::ac::rat::number_domain(false);
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
    let _exact = crate::ac::rat::number_domain(false);
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
