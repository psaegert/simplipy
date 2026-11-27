//! THE REGION RULE: where a sum's zero may hold a region and keeps its sign in float64, where
//! nothing reads it and it is released, and the zero-set tests the licences read; the memos a
//! simplify call holds open for them ([`RegionScope`]).

use super::*;

/// The licence of an odd negative power's distribution over a product (`pow`): is `e` nonzero
/// almost everywhere by the certificate (`nz_ae_licensed`)? The parse, which has no certificate,
/// holds a divisor product, and `canon` distributes it where the certificate licenses it; a product
/// the certificate refuses stays one member, which the search may expand (`1/((a x + b)(c x + d))`
/// as one polynomial). Only a divisor holding an odd root is distributed by its form
/// (`power_bag`).
///
/// A negative power `b^-q` is zero only where `b` is infinite, so its zero set is null wherever `b`
/// is finite almost everywhere (`fin_licensed`); where `b` is undefined the power is too, and an
/// inverse of an undefined product is undefined as each member's is. So the inverse of a quotient
/// distributes as the quotient's members would: `1/(x12^(-1/2)/x14)` is `x14 x12^(1/2)`.
///
/// The verdict is a function of `e` and the context alone, so within a simplify call
/// (`RegionScope`) each distinct `e` is decided once per context however often the products
/// holding it are built.
pub(super) fn zero_set_null(e: &Ex, cx: &Cx) -> bool {
    if cx.permits(Permissive::Agreement) {
        return true;
    }
    let key = flip_cx_key(cx);
    let hit = REGION_MEMO.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|t| t.get(&key).and_then(|v| v.get(e).copied()))
    });
    if let Some(hit) = hit {
        return hit;
    }
    let r = cx.nz_ae_licensed(e)
        || matches!(e, Ex::Pow(b, q) if matches!(&**q, Ex::Num(r) if r.is_negative())
            && cx.fin_licensed(b));
    REGION_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.entry(key).or_default().insert(e.clone(), r);
        }
    });
    r
}

/// `zero_set_null`'s verdicts, per context (`flip_cx_key`) and expression. The key holds whether a
/// confinement holds (`CONFINED_BIT`) but not its bases (`Cx::confined`), and needs none: the
/// verdict reads the certificate closures (`nz_ae_licensed`, `fin_licensed`), never the bases,
/// which only `Cx::confined_nonneg` reads (the composition of powers in `mul` and `laws`). So a
/// verdict under one confinement is the verdict under any other and under none (the canon review of
/// 2026-11-27 measured 0 re-ask differences over 300,000 random expressions in shuffled order).
type RegionMemo = rustc_hash::FxHashMap<FlipCxKey, rustc_hash::FxHashMap<Ex, bool>>;

thread_local! {
    static REGION_MEMO: std::cell::RefCell<Option<RegionMemo>> = const { std::cell::RefCell::new(None) };
}

/// [`strict_price`]'s prices within a simplify call (`RegionScope`), per spelling and number
/// domain (the token view is the call's own). No other part of the context is in the key because
/// the price reads none: it is strict's canonical form priced under the BARE context of the call's
/// token view (`Cx::bare`: no certificate, no mode's permissions, no confinement).
type StrictPriceMemo = rustc_hash::FxHashMap<(Ex, bool), u64>;

thread_local! {
    pub(super) static STRICT_PRICE_MEMO: std::cell::RefCell<Option<StrictPriceMemo>> =
        const { std::cell::RefCell::new(None) };
}

/// Holds a fresh region memo (`zero_set_null`), class-read memo (`zero_read`) and strict-price
/// memo (`strict_price`) open for one simplify call and restores the
/// caller's when it drops. The memo's key holds the context's identities (its token view and
/// certificate closures), which live as long as the call that opens the scope, so no identity is
/// reused within one memo's extent; a nested call keys its own contexts in its own memo.
pub(crate) struct RegionScope(
    Option<RegionMemo>,
    Option<ReadMemo>,
    Option<StrictPriceMemo>,
);

impl RegionScope {
    pub(crate) fn open() -> Self {
        RegionScope(
            REGION_MEMO.with(|m| m.borrow_mut().replace(RegionMemo::default())),
            READ_MEMO.with(|m| m.borrow_mut().replace(ReadMemo::default())),
            STRICT_PRICE_MEMO.with(|m| m.borrow_mut().replace(StrictPriceMemo::default())),
        )
    }
}

impl Drop for RegionScope {
    fn drop(&mut self) {
        let outer = self.0.take();
        REGION_MEMO.with(|m| *m.borrow_mut() = outer);
        let reads = self.1.take();
        READ_MEMO.with(|m| *m.borrow_mut() = reads);
        let prices = self.2.take();
        STRICT_PRICE_MEMO.with(|m| *m.borrow_mut() = prices);
    }
}

/// A SUM KEEPS THE SIGN OF ITS ZERO WHERE ITS ZERO SET MAY HOLD A REGION, in `f64`. Float64 adds
/// to `+0` wherever a sum cancels, however it is written: `x1 - |x1|` and `|x1| - x1` are both
/// `+0` for every `x1 >= 0`, so the negation `-(|x1| - x1)` is `-0` there. A sum and its terms
/// negated one by one (its other orientation) are one value everywhere else, the rounding being
/// symmetric in sign. Where the zero set is null its sign is no value the answer keeps (reading
/// A: poles are read by their limits); where it may hold a region, the sign is a value where the
/// zero is inverted (`region_signed_zero`, `inverts_a_signed_zero`). The region test is reading
/// A's (`zero_set_null`: the certificate where the context has one; a context without it reads
/// the form, `zero_set_null_by_form`). A sum without a region of the variables' space
/// (`Ex::has_measure_space`: a ground sum, or a pattern's) has no zero set to read, and a sum
/// whose sign an absorption owner takes into its terms whatever its zero set -- one holding a
/// `<constant>` (the refit), an infinity term or a term absorbing a negation
/// -- keeps its owner. `permissive` reads a value up to the sign of a zero.
pub(crate) fn keeps_zero_sign(s: &Ex, cx: &Cx) -> bool {
    signed_zero_sum(s, cx) && !region_zero_null(s, cx)
}

/// A sum whose zero's sign is its own in `f64`: one over a region of the variables' space that no
/// absorption owner takes the sign of (`keeps_zero_sign`).
fn signed_zero_sum(s: &Ex, cx: &Cx) -> bool {
    let owned = s.contains_const()
        || matches!(s, Ex::Add(ts) if ts.iter().any(|t| {
            matches!(t, Ex::PosInf | Ex::NegInf) || term_absorbs_negation(t)
        }));
    cx.f64_semantics() && s.has_measure_space(cx.view) && !owned
}

/// [`keeps_zero_sign`] read by the form alone, without the certificate: a sum the form clears
/// (`zero_set_null_by_form`, or a separated variable, `zero_set_null_by_separation`) keeps no
/// sign, one it does not may. The certificate is an interval analysis, dear on a long sum, and
/// the form is read wherever a sum is built.
fn may_keep_zero_sign(s: &Ex, cx: &Cx) -> bool {
    signed_zero_sum(s, cx) && !zero_set_null_by_form(s, cx) && !zero_set_null_by_separation(s, cx)
}

/// A SUM HOLDS ITS SIGN where its zero's sign is a value: where it may vanish on a region
/// (`keeps_zero_sign`) and the expression around it reads its zero (`zero_read_below`). Its sign
/// does not leave it there (`add.content`: `-a - b` stays, it is not `-(a + b)`), no negative
/// coefficient spreads over it (`add.unfactor`), no sign trades into it (`2*(-a - b)` is not
/// `-2*(a + b)`), and a printer writes its negation as the negation it is (`neg - a b`, not
/// `b - a`). Float64's sum is `+0` wherever it cancels, so each of those moves changes the zero's
/// sign there (`1/(-ceil(x0) - floor(x1))` is `+inf` wherever `ceil(x0) = -floor(x1)`,
/// `-1/(ceil(x0) + floor(x1))` is `-inf`).
///
/// A construction does not see what is around it, so it holds every sum whose form does not show
/// its zero set null (`may_keep_zero_sign`), and the root's walk ([`release_unread`]) gives a held
/// sum the orientation its constructors give it where nothing reads its zero or the certificate
/// shows its zero set null. A printer reads the held sums of the state it prints
/// (`Cx::sign_held`). A context without a zero-set certificate (a key's, a rule side's) holds
/// none, and neither does one where nothing reads the zero (`Cx::zero_sign_unread`).
pub(crate) fn sum_holds_its_sign(s: &Ex, cx: &Cx) -> bool {
    if cx.zero_sign_unread {
        return false;
    }
    if let Some(held) = cx.sign_held {
        return held(s);
    }
    (cx.cert_region.is_some() || cx.cert_nzae.is_some()) && may_keep_zero_sign(s, cx)
}

/// Is `t` a subtracted term of a sum: a negative literal, or a product whose coefficient is one?
/// A sign leaves a sum only where it holds one.
fn subtracted_term(t: &Ex) -> bool {
    match t {
        Ex::Num(r) => r.is_negative(),
        Ex::Mul(f) => matches!(f.first(), Some(Ex::Num(r)) if r.is_negative()),
        _ => false,
    }
}

/// The sums of `e` that hold their sign (`sum_holds_its_sign`), for a printer (`Cx::sign_held`):
/// each sum holding a subtracted term at a place that reads its zero (`zero_read_below`; the root
/// reads it where `read`) whose form does not show its zero set null (`may_keep_zero_sign`). Its
/// negation written as it is reads back to the same sum, so the answer reads back to its state.
pub fn held_sums(e: &Ex, read: bool, cx: &Cx) -> rustc_hash::FxHashSet<Ex> {
    fn walk(e: &Ex, read: bool, cx: &Cx, out: &mut rustc_hash::FxHashSet<Ex>) {
        if read
            && matches!(e, Ex::Add(ts) if ts.iter().any(subtracted_term))
            && !out.contains(e)
            && may_keep_zero_sign(e, cx)
        {
            out.insert(e.clone());
        }
        // A held sum is written with its groups as they are, as the root's walk keeps them
        // (`released`): a group's sum holding a subtracted term is written as it is there.
        if read && matches!(e, Ex::Add(_)) && may_keep_zero_sign(e, cx) {
            if let Ex::Add(ts) = e {
                for t in ts {
                    if let Ex::Mul(f) = t {
                        if let (Some(_), Some(inner @ Ex::Add(its))) = (factored_sum(f), f.last()) {
                            if its.iter().any(subtracted_term) {
                                out.insert(inner.clone());
                            }
                        }
                    }
                }
            }
        }
        match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
                for (i, c) in v.iter().enumerate() {
                    walk(c, zero_read_below(e, i, read, cx), cx, out);
                }
            }
            Ex::Pow(b, x) => {
                walk(b, zero_read_below(e, 0, read, cx), cx, out);
                walk(x, false, cx, out);
            }
            _ => {}
        }
    }
    let mut out = rustc_hash::FxHashSet::default();
    if f64_zero_signs(cx) {
        walk(e, read, cx, &mut out);
    }
    out
}

/// WHERE NOTHING READS A SUM'S ZERO, ITS SIGN MOVES: `e` with every sum a construction held
/// (`sum_holds_its_sign`) rebuilt in the orientation its constructors give it where its zero set
/// is null, wherever nothing reads the sign of its zero (`zero_read_below`) or the certificate
/// shows its zero set null (`keeps_zero_sign`). The root of a state reads no zero (`land_root`),
/// and below it a sum, a product, an odd power or an odd head carries the sign up to what reads
/// it; a head blind to it (`exp`, `cos`, `abs`, an even power) does not. So
/// `-ceil(x0) - floor(x1)` is `-(ceil(x0) + floor(x1))` as an answer and under `exp`, and stays
/// as it is under an inverse, where float64's `+0` and the negation's `-0` are `+inf` and `-inf`.
/// The nodes are rebuilt from their children, the released places in a context that holds no
/// sign (`Cx::zero_sign_unread`).
pub(crate) fn release_unread(e: Ex, cx: &Cx) -> Ex {
    if !f64_zero_signs(cx)
        || cx.zero_sign_unread
        || (cx.cert_region.is_none() && cx.cert_nzae.is_none())
    {
        return e;
    }
    let free = Cx {
        zero_sign_unread: true,
        ..*cx
    };
    released(&e, false, cx, &free).0.unwrap_or(e)
}

/// [`release_unread`] at `e`, read or not (`read`): the node where it changes (`None` where
/// nothing does), and whether its subtree holds a sum the form does not clear
/// (`may_keep_zero_sign`), whose sign a construction may have held.
fn released(e: &Ex, read: bool, cx: &Cx, free: &Cx) -> (Option<Ex>, bool) {
    let mut held = false;
    let rebuilt = match e {
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            let mut kids: Option<Vec<Ex>> = None;
            for (i, c) in v.iter().enumerate() {
                let (r, h) = released(c, zero_read_below(e, i, read, cx), cx, free);
                held |= h;
                if let Some(r) = r {
                    kids.get_or_insert_with(|| v.clone())[i] = r;
                }
            }
            kids.map(|k| match e {
                Ex::Add(_) => add(k, cx),
                Ex::Mul(_) => mul(k, cx),
                // `atan2` is built from its arguments' members (`atan2_of`), as the parse builds it.
                Ex::Fun(f, _) if k.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(*f, "atan2") => {
                    let mut members: Vec<Vec<Ex>> = k.into_iter().map(product_members).collect();
                    let xm = members.pop().expect("two arguments");
                    let ym = members.pop().expect("two arguments");
                    atan2_of(*f, ym, xm, cx)
                }
                Ex::Fun(f, _) => fun(*f, k, cx),
                _ => unreachable!("a node with members"),
            })
        }
        Ex::Pow(b, x) => {
            let (nb, hb) = released(b, zero_read_below(e, 0, read, cx), cx, free);
            let (nx, hx) = released(x, false, cx, free);
            held = hb || hx;
            (nb.is_some() || nx.is_some()).then(|| {
                pow(
                    nb.unwrap_or_else(|| (**b).clone()),
                    nx.unwrap_or_else(|| (**x).clone()),
                    cx,
                )
            })
        }
        _ => return (None, false),
    };
    let node = rebuilt.as_ref().unwrap_or(e);
    held |= matches!(node, Ex::Add(_)) && may_keep_zero_sign(node, cx);
    if !held {
        return (rebuilt, false);
    }
    // The node rebuilt where no sum holds its sign; a node is its constructors' own (they build
    // it again), so it differs where a sum held its sign, its own or one a sign reaches through
    // its members. Where it differs, again where only the sums read in its subtree that keep
    // their sign by the certificate hold it (asked last).
    let build = |free: &Cx| match node {
        Ex::Add(v) => Some(add(v.clone(), free)),
        Ex::Mul(v) => Some(mul(v.clone(), free)),
        Ex::Pow(b, x) => Some(pow((**b).clone(), (**x).clone(), free)),
        Ex::Fun(f, v) if v.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(*f, "atan2") => {
            let mut members: Vec<Vec<Ex>> = v.iter().cloned().map(product_members).collect();
            let xm = members.pop().expect("two arguments");
            let ym = members.pop().expect("two arguments");
            Some(atan2_of(*f, ym, xm, free))
        }
        Ex::Fun(f, v) => Some(fun(*f, v.clone(), free)),
        _ => None,
    };
    let Some(freed) = build(free) else {
        return (rebuilt, true);
    };
    if freed == *node {
        return (rebuilt, true);
    }
    // A sum whose zero is read is named by its spelling, and so is each group it holds
    // (`group_shared_numbers`): a group a sum keeps is spelled with it, so the group keeps its
    // sign there even where nothing reads the group's own zero (`ceil(x2) + 2*(-ceil(x0) -
    // floor(x1))` under an inverse is that sum, and releasing the group's sign would spell
    // another).
    fn kept_below<'e>(e: &'e Ex, read: bool, cx: &Cx, out: &mut Vec<&'e Ex>) {
        if read && matches!(e, Ex::Add(_)) && keeps_zero_sign(e, cx) {
            out.push(e);
            if let Ex::Add(ts) = e {
                for t in ts {
                    if let Ex::Mul(f) = t {
                        if let (Some(_), Some(inner)) = (factored_sum(f), f.last()) {
                            out.push(inner);
                        }
                    }
                }
            }
        }
        match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
                for (i, c) in v.iter().enumerate() {
                    kept_below(c, zero_read_below(e, i, read, cx), cx, out);
                }
            }
            Ex::Pow(b, x) => {
                kept_below(b, zero_read_below(e, 0, read, cx), cx, out);
                kept_below(x, false, cx, out);
            }
            _ => {}
        }
    }
    let mut kept: Vec<&Ex> = Vec::new();
    kept_below(node, read, cx, &mut kept);
    if kept.is_empty() {
        return (Some(freed), true);
    }
    let is_kept = |s: &Ex| kept.contains(&s);
    let selective = Cx {
        sign_held: Some(&is_kept),
        ..*cx
    };
    match build(&selective) {
        Some(freed) if freed != *node => (Some(freed), true),
        _ => (rebuilt, true),
    }
}

/// May the expression around child `i` of `e` read the sign of the child's zero, given `read` for
/// `e`'s own (in `f64`; float64's signed zeros)? A sum reads it only where every other term may be
/// -0 on a region (a sum is -0 only where every term is: `add.zero`'s test), a product carries it
/// into its own, a power or a head that carries a zero's sign (an odd power, `sqrt`, an odd head
/// in its odd argument, `floor`, `ceil`) carries it, an inverse (a negative odd power) reads it, a
/// head blind to it (an even power, `exp`, `cos`, `abs`, `rootn` of an even index, ...) does not,
/// and any other head is taken to read it.
pub(crate) fn zero_read_below(e: &Ex, i: usize, read: bool, cx: &Cx) -> bool {
    if !cx.ieee_zeros() {
        return false;
    }
    match e {
        Ex::Add(v) => {
            read && !v
                .iter()
                .enumerate()
                .any(|(j, t)| j != i && signed_zero_null(t, true, cx))
        }
        Ex::Mul(_) => read,
        Ex::Pow(_, x) => match &**x {
            Ex::Num(k) if k.is_integer() && k.is_odd_integer() => read || k.is_negative(),
            Ex::Num(k) if k.is_integer() => false,
            Ex::Num(k) if *k == Rat::new(1, 2).expect("1/2") => read,
            Ex::Num(_) => false,
            _ => true,
        },
        Ex::Fun(f, a) => {
            let is = |s: &str| cx.view.tok_is(*f, s);
            // An odd head in its odd argument (`odd_head`: an odd function, `besselj` of an odd
            // order, `rootn` of an odd index), `floor` and `ceil` carry the sign; `rootn` of an
            // even index is `+0` at both zeros.
            let carries = odd_head(cx.view, *f, a).is_some_and(|(p, _)| p == i)
                || (a.len() == 1 && (is("floor") || is("ceil")));
            let blind = (a.len() == 1
                && ["abs", "cos", "cosh", "exp", "log", "acos", "acosh", "erfc"]
                    .iter()
                    .any(|s| is(s)))
                || (i == 0
                    && is("rootn")
                    && matches!(a.get(1), Some(Ex::Num(k)) if k.is_integer() && !k.is_odd_integer()));
            if carries {
                read
            } else {
                !blind
            }
        }
        _ => true,
    }
}

/// The region test of `f64`'s zero signs (`keeps_zero_sign`, `signed_zero_null`): is the zero set
/// of `e` null, by its form (`zero_set_null_by_form`) or by the certificate -- the context's region
/// certificate where it has one (`Cx::cert_region`, the parse's), else `zero_set_null`?
fn region_zero_null(e: &Ex, cx: &Cx) -> bool {
    zero_set_null_by_form(e, cx)
        || match cx.cert_region {
            Some(f) => f(e),
            None => zero_set_null(e, cx),
        }
}

/// Is `f64` the construction, in the run's float64 numbers? A construction in the exact number
/// domain (`Cx::f64_numbers`) reads one zero.
pub(crate) fn f64_zero_signs(cx: &Cx) -> bool {
    cx.f64_semantics() && cx.f64_numbers()
}

/// In `f64`, may `e` vanish on a region: its zero set not shown null by the region test? A ground
/// expression and a pattern have no region (as in `keeps_zero_sign`).
pub(crate) fn vanishes_on_a_region(e: &Ex, cx: &Cx) -> bool {
    f64_zero_signs(cx)
        && e.has_measure_space(cx.view)
        && !e.contains_wildcard(cx.view)
        && !region_zero_null(e, cx)
}

/// A SUM OF ZEROS IS -0 ONLY WHERE EVERY TERM IS, in float64: `-0 + 0` is `+0`, and so is any
/// cancellation of nonzero terms. Is the set where `e` is float64's zero of one sign null --
/// where it is `-0` (`neg`), else where it is `+0` -- PROVEN, in `f64`? Fail closed: `false`
/// wherever it is not shown. The sum constructor reads it (`add`): it drops a `+0` term, and
/// collects two like terms of opposite signs, only where what remains cannot be `-0` on a region.
///
/// `ceil(x1)` is `-0` on the whole interval (-1, 0), so `ceil(x1) + 0` is `+0` there and
/// `1/(ceil(x1) + 0)` is `+inf`, while `1/ceil(x1)` is `-inf`. The heads and carriers, as the
/// evaluator computes them: a term whose zero set is null has no such region (the region test,
/// `region_zero_null`); `floor` is the sign of its argument's zero (`floor(-0) = -0`) and `+0` on
/// [0, 1), `ceil` the sign of its argument's (`ceil(+0) = +0`) and `-0` on (-1, 0); an odd
/// function, an odd root or an odd power keeps its argument's sign; `abs`, `cos`, `cosh`, `exp`,
/// `log`, `acos`, `acosh`, `erfc` (whose zero is `+0`, by underflow), an even root and an even or
/// fractional power are never `-0` (the power
/// `1/2` is numpy's `sqrt`, `sqrt(-0) = -0`, and keeps it), and so is a negative power of an even
/// or non-integer exponent, while an odd one is `-0` only where its base is negative (at `-inf`,
/// or by underflow), so a base that is never negative keeps it from `-0`; `max` and `min` are one
/// of their members; `mod` takes the sign of its divisor and `atan2` of its first argument
/// (read where its second is finite: `atan2(y, +inf)` is `+-0`); a product's zero has the sign of
/// its members, so a product is read through its one member that may vanish on a region where
/// every other member is positive, and a product holding a zero is that zero everywhere, of the
/// members' sign; a sum is `-0` only where every term is. `harmonic`, `besselj`
/// (`besselj(1.5, 0) = -0`) and any other head are read by the region test alone. A ground term
/// is its float64 value (`ground_value`); a pattern has no region (as in `keeps_zero_sign`).
pub(crate) fn signed_zero_null(e: &Ex, neg: bool, cx: &Cx) -> bool {
    let view = cx.view;
    if e.contains_wildcard(view) {
        return true;
    }
    if !e.has_measure_space(view) {
        return match e {
            Ex::Num(r) => neg || !r.is_zero(),
            _ if is_neg_zero(e, view) => !neg,
            _ => ground_value(e, cx).is_some_and(|v| v != 0.0 || v.is_sign_negative() != neg),
        };
    }
    if region_zero_null(e, cx) {
        return true;
    }
    let rec = |g: &Ex, n: bool| signed_zero_null(g, n, cx);
    // `b^k` as the evaluator's `pow` computes it (`simplipy.operators.pow`).
    let power = |b: &Ex, k: &Rat| {
        if k.is_integer() && k.signum() > 0 {
            if k.is_even_integer() {
                neg
            } else {
                rec(b, neg)
            }
        } else if *k == Rat::new(1, 2).expect("1/2") {
            rec(b, neg)
        } else if k.signum() < 0 {
            // a negative power vanishes only where `b` is infinite or the power underflows; an
            // even or non-integer one is never -0, and an odd one is -0 only where `b` is
            // negative (at -inf, or by underflow), so a base that is never negative keeps it from
            // -0 (`1/abs(t)`)
            neg && (!k.is_integer() || k.is_even_integer() || cx.certainly_nonneg(b))
        } else {
            // a fractional power of a zero is +0
            k.signum() > 0 && neg
        }
    };
    match e {
        Ex::Add(ts) => neg && ts.iter().any(|t| rec(t, true)),
        Ex::Mul(v) => {
            // The zero of a product carries the members' signs. `sign` is the sign asked, turned
            // by each negative ground member; the other members must be positive wherever they
            // are defined and nonzero (a power of a non-integer exponent is NaN below 0).
            let positive = |f: &Ex| {
                cx.certainly_nonneg(f)
                    || matches!(f, Ex::Pow(_, k) if matches!(&**k, Ex::Num(k) if !k.is_integer()))
            };
            let mut sign = neg;
            let mut zero = false;
            let mut carrier: Option<&Ex> = None;
            let mut unsigned = false;
            for f in v {
                if !f.has_measure_space(view) {
                    // a ground member: its float64 value, a zero making the product one everywhere
                    match ground_value(f, cx) {
                        Some(x) if x.is_finite() => {
                            zero |= x == 0.0;
                            sign ^= x.is_sign_negative();
                        }
                        _ => return false,
                    }
                } else if region_zero_null(f, cx) {
                    unsigned |= !positive(f);
                } else if carrier.is_none() {
                    carrier = Some(f);
                } else {
                    // a second member that may vanish on a region
                    return false;
                }
            }
            if zero {
                // zero wherever it is defined, of the members' sign: never the sign asked where
                // every other member is positive and never -0
                sign && !unsigned && carrier.is_none_or(|g| positive(g) && rec(g, true))
            } else {
                carrier.is_none_or(|g| !unsigned && rec(g, sign))
            }
        }
        Ex::Pow(b, k) => matches!(&**k, Ex::Num(k) if power(b, k)),
        Ex::Fun(op, a) => {
            let is = |s: &str| view.tok_is(*op, s);
            match a.as_slice() {
                [g] if odd_fun(view, *op) => rec(g, neg),
                [_] if ["abs", "cos", "cosh", "exp", "log", "acos", "acosh", "erfc"]
                    .iter()
                    .any(|s| is(s)) =>
                {
                    neg
                }
                [g] if is("floor") => neg && rec(g, true),
                [g] if is("ceil") => !neg && rec(g, false),
                [b, Ex::Num(k)] if is("rootn") => {
                    if k.is_integer() && k.signum() > 0 {
                        if k.is_even_integer() {
                            neg
                        } else {
                            rec(b, neg)
                        }
                    } else {
                        false
                    }
                }
                [b, Ex::Num(k)] if is("pow") => power(b, k),
                [_, b] if is("mod") => {
                    if neg {
                        cx.certainly_nonneg(b)
                    } else {
                        cx.carried_negative(b)
                            .is_some_and(|p| cx.certainly_nonneg(&p))
                    }
                }
                [y, x] if is("atan2") => rec(y, neg) && cx.fin_certified(x),
                ms if lattice_head(*op, view) => ms.iter().all(|m| rec(m, neg)),
                _ => false,
            }
        }
        _ => false,
    }
}

/// Does `f` vanish on a region with the sign its orientation gives it: a sum that keeps the sign
/// of its zero (`keeps_zero_sign`), or an odd power, an odd function or an odd root of one, each
/// odd at a signed zero as everywhere else?
pub(crate) fn region_signed_zero(f: &Ex, cx: &Cx) -> bool {
    signed_zero_of(f, cx, true)
}

/// [`region_signed_zero`], reading a sum's zero (`keeps_zero_sign`) where `sums`, and no sum's
/// where not: the heads that vanish with a sign of their own and the odd carriers of them.
fn signed_zero_of(f: &Ex, cx: &Cx, sums: bool) -> bool {
    let rec = |g: &Ex| signed_zero_of(g, cx, sums);
    match f {
        Ex::Add(_) => sums && keeps_zero_sign(f, cx),
        Ex::Pow(b, e) => {
            matches!(&**e, Ex::Num(r) if r.is_odd_integer() && !r.is_negative()) && rec(b)
        }
        Ex::Fun(op, a) if a.len() == 1 && odd_fun(cx.view, *op) => rec(&a[0]),
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            matches!(&a[1], Ex::Num(k) if k.is_odd_integer()) && rec(&a[0])
        }
        // The 0.15 heads. A step or lattice head vanishes on whole intervals with a sign of its
        // own (`ceil` is -0 on (-1, 0), `mod` takes the divisor's, `max(x, -0)` and `min(x, 0)`
        // hold a zero on a half-line), wherever it has a region to vanish on, and so do the scipy
        // heads in float64: `erfc` is +0 from x = 26.6 and `besselk` from 697.9, `besseli`
        // underflows to a zero of its order's parity near 0 (-0 at x < 0 under an odd order),
        // and `bessely` is +-0 on 7.2e8 <= x <= 2^51 from order 86. `harmonic` is odd at its
        // zero, `atan2` in its first argument and `besselj` (under an odd order) in its second,
        // so each keeps its argument's.
        Ex::Fun(op, _)
            if [
                "floor", "ceil", "max", "min", "mod", "erfc", "bessely", "besseli", "besselk",
            ]
            .iter()
            .any(|h| cx.view.tok_is(*op, h)) =>
        {
            cx.f64_semantics() && f.has_measure_space(cx.view)
        }
        Ex::Fun(op, a) if a.len() == 1 && cx.view.tok_is(*op, "harmonic") => rec(&a[0]),
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "atan2") => rec(&a[0]),
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "besselj") => rec(&a[1]),
        // `pow(b, 1/2)` kept as the call it is (`pow_spelled`): `sqrt` keeps the sign of its
        // argument's zero.
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "pow") => {
            matches!(&a[1], Ex::Num(h) if *h == Rat::new(1, 2).expect("1/2"))
                && sqrt_parts_from_rootn(&a[0], cx)
        }
        _ => false,
    }
}

/// In `f64`, may `b` be a zero with a sign of its own where `b ** 0.5` (numpy's `sqrt`, which keeps
/// it) and `rootn(b, 2)` (+0) part: the written negative zero, a ground composite whose float64
/// value is -0 (in the reading that folds ground nodes, `Cx::fold_f64`, as `signed_zero_product`
/// reads one), a head that vanishes on a region with a sign of its own or an odd carrier of one
/// (`signed_zero_of`), or a product holding such a member (its zero takes the members' signs)?
/// A sum is not read: its orientation is canonical before a power sees it, so the sign of its
/// zero is not the written one, and a sum whose zero set the certificate cannot show null (most
/// quotients) would keep every square root of it as a call.
pub(crate) fn sqrt_parts_from_rootn(b: &Ex, cx: &Cx) -> bool {
    cx.f64_semantics()
        && (is_neg_zero(b, cx.view)
            || signed_zero_of(b, cx, false)
            || zero_of_either_sign(b, cx)
            || matches!(b, Ex::Mul(v) if v.iter().any(|f| sqrt_parts_from_rootn(f, cx)))
            || (cx.fold_f64
                && matches!(b, Ex::Add(_) | Ex::Mul(_) | Ex::Fun(..) | Ex::Pow(..))
                && ground_value(b, cx).is_some_and(|v| v == 0.0 && v.is_sign_negative())))
}

/// Is the zero set of `e` null by its form alone, without the certificate (which the parse has
/// not)? A variable, a nonzero literal, a `Const`, an infinity, a product of such factors, a
/// positive power or a root of one, a negative power of a function whose level sets are null
/// (zero only where that function is infinite), or such a function itself (`level_sets_null`).
/// A sufficient test: where it fails the certificate may still hold.
pub(super) fn zero_set_null_by_form(e: &Ex, cx: &Cx) -> bool {
    match e {
        Ex::Num(r) => !r.is_zero(),
        Ex::Pi | Ex::E | Ex::Const(_) | Ex::PosInf | Ex::NegInf => true,
        Ex::Leaf(_) => form_variable(e, cx),
        Ex::Mul(v) => v.iter().all(|f| zero_set_null_by_form(f, cx)),
        Ex::Pow(b, q) => match &**q {
            Ex::Num(q) if q.signum() > 0 => zero_set_null_by_form(b, cx),
            Ex::Num(q) if q.is_negative() => level_sets_null(b, cx),
            _ => false,
        },
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            zero_set_null_by_form(&a[0], cx)
        }
        _ => level_sets_null(e, cx),
    }
}

/// Is the zero set of the sum `s` null by a SEPARATED VARIABLE: a variable `v` that only one term
/// holds, as one factor `g(v)` of no other variable whose level sets are null
/// (`level_sets_null`), beside factors whose zero sets are null by form? For almost every value
/// of the other variables the sum is then `a*g(v) + b` with `a` nonzero, which vanishes only where
/// `g(v) = -b/a`, a null set of `v`; so the zero set is null (Fubini): `x2 - 2*ceil(x0)` and
/// `0.03*x1 + 18*(0.58 - 0.48*x2)*sin(x2)` are. `g` holds no `tanh` or `atan`, which float64
/// holds at one level on a whole tail (`tanh(20)` is 1), so that `x2 + tanh(x1) - 1` vanishes on
/// a region. A sufficient test, which a sum's hold reads before the certificate
/// (`may_keep_zero_sign`).
fn zero_set_null_by_separation(s: &Ex, cx: &Cx) -> bool {
    fn holds(e: &Ex, v: Tok) -> bool {
        match e {
            Ex::Leaf(t) => *t == v,
            Ex::Add(w) | Ex::Mul(w) | Ex::Fun(_, w) => w.iter().any(|x| holds(x, v)),
            Ex::Pow(b, x) => holds(b, v) || holds(x, v),
            _ => false,
        }
    }
    fn one_variable(e: &Ex, cx: &Cx, v: &mut Option<Tok>) -> bool {
        match e {
            Ex::Leaf(t) if form_variable(e, cx) => match v {
                Some(u) => *u == *t,
                None => {
                    *v = Some(*t);
                    true
                }
            },
            Ex::Leaf(_) | Ex::Const(_) => false,
            Ex::Add(w) | Ex::Mul(w) | Ex::Fun(_, w) => w.iter().all(|x| one_variable(x, cx, v)),
            Ex::Pow(b, x) => one_variable(b, cx, v) && one_variable(x, cx, v),
            _ => true,
        }
    }
    // A head float64 holds at one level on a whole tail.
    fn saturates(e: &Ex, cx: &Cx) -> bool {
        match e {
            Ex::Fun(f, w) => {
                cx.view.tok_is(*f, "tanh")
                    || cx.view.tok_is(*f, "atan")
                    || w.iter().any(|x| saturates(x, cx))
            }
            Ex::Add(w) | Ex::Mul(w) => w.iter().any(|x| saturates(x, cx)),
            Ex::Pow(b, x) => saturates(b, cx) || saturates(x, cx),
            _ => false,
        }
    }
    let Ex::Add(ts) = s else { return false };
    if !f64_zero_signs(cx) || s.contains_wildcard(cx.view) {
        return false;
    }
    ts.iter().enumerate().any(|(i, t)| {
        let factors: &[Ex] = match t {
            Ex::Mul(w) => w,
            f => std::slice::from_ref(f),
        };
        factors.iter().enumerate().any(|(j, g)| {
            let mut v = None;
            if !one_variable(g, cx, &mut v) || saturates(g, cx) || !level_sets_null(g, cx) {
                return false;
            }
            let Some(v) = v else { return false };
            factors
                .iter()
                .enumerate()
                .all(|(k, f)| k == j || (!holds(f, v) && zero_set_null_by_form(f, cx)))
                && ts.iter().enumerate().all(|(m, u)| m == i || !holds(u, v))
        })
    })
}

/// A variable of the form tests: a leaf that is no wildcard and no numeral (a literal the f64
/// domain keeps as a leaf may read as 0 or as an infinity).
fn form_variable(e: &Ex, cx: &Cx) -> bool {
    matches!(e, Ex::Leaf(t) if cx.view.sigil(*t) == 0
        && !cx.view.with_str(*t, crate::utils::looks_numeric))
}

/// Is every level set of `e` null: a variable, a nonconstant polynomial in variables (integer
/// exponents, rational coefficients; a `Const` term is a generic parameter, so each of its
/// values shifts one level), or a nonconstant analytic function, a nonzero power or a root of
/// such a function, each of whose levels is reached on countably many levels of its argument?
fn level_sets_null(e: &Ex, cx: &Cx) -> bool {
    const ANALYTIC: [&str; 14] = [
        "sin", "cos", "tan", "exp", "log", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh",
        "acosh", "atanh",
    ];
    let variable = |f: &Ex| form_variable(f, cx);
    let power_of_variable = |f: &Ex| {
        variable(f)
            || matches!(f, Ex::Pow(b, q) if variable(b)
                && matches!(&**q, Ex::Num(q) if q.is_integer() && !q.is_zero()))
    };
    // A term of a polynomial: a literal or `Const` coefficient times powers of variables; does it
    // hold a variable?
    let monomial = |t: &Ex| -> Option<bool> {
        match t {
            Ex::Num(_) | Ex::Const(_) => Some(false),
            Ex::Mul(v) => v
                .iter()
                .all(|f| matches!(f, Ex::Num(_) | Ex::Const(_)) || power_of_variable(f))
                .then(|| v.iter().any(power_of_variable)),
            f => power_of_variable(f).then_some(true),
        }
    };
    match e {
        Ex::Leaf(_) => variable(e),
        Ex::Add(_) | Ex::Mul(_) => {
            let terms: &[Ex] = match e {
                Ex::Add(ts) => ts,
                _ => std::slice::from_ref(e),
            };
            let read: Option<Vec<bool>> = terms.iter().map(monomial).collect();
            read.is_some_and(|r| r.into_iter().any(|x| x))
        }
        Ex::Pow(b, q) => matches!(&**q, Ex::Num(q) if !q.is_zero()) && level_sets_null(b, cx),
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            level_sets_null(&a[0], cx)
        }
        Ex::Fun(op, a) => {
            a.len() == 1
                && ANALYTIC.iter().any(|s| cx.view.tok_is(*op, s))
                && level_sets_null(&a[0], cx)
        }
        _ => false,
    }
}
