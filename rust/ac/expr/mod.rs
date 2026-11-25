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
mod classes;
mod coef;
mod cx;
mod ex;
mod flat;
mod limit;
mod literals;
mod mul;
mod opposite;
mod order;
mod pow;
mod price;
mod sign_place;
mod terms;
mod zero_sign;

pub use self::add::*;
use self::classes::*;
pub(crate) use self::coef::*;
pub use self::cx::*;
pub use self::ex::*;
pub(crate) use self::flat::*;
use self::limit::*;
pub use self::literals::*;
pub use self::mul::*;
use self::opposite::*;
pub use self::order::*;
pub use self::pow::*;
pub use self::price::*;
pub(crate) use self::sign_place::*;
pub(crate) use self::terms::*;
pub use self::zero_sign::*;

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
