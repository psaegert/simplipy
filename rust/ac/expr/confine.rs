//! THE CONFINEMENT OF A STATE ([`release_confined`], the law `confine.sibling`, `f64` only).
//!
//! A sum or a product is NaN wherever one of its members is: float64's `+` and `*` give NaN for a
//! NaN operand whatever the other one is (`NaN + inf`, `NaN * 0`). So where a member is NaN at every
//! point where a base `t` is negative (`x^(5/2)`, `log(x)` and `rootn(x, 2)` are, for `t = x`), the
//! whole node is NaN there, whatever its other members compute, and those members may be rewritten
//! by any identity that holds where `t >= 0`: the node keeps its value at every point. The laws
//! that merge fractional powers of `t`, whose two sides differ only where `t` is negative, then
//! apply to `t` throughout the node (`Cx::confined`): `x^(3/2)*(x^(3/2) + x)` is
//! `x^(5/2) + x^3`, as `x*x*(x^(1/2) + x)` is, where the product `x^(3/2)*x^(3/2)` alone keeps its
//! halves (`x^3` has a value on x < 0). A law that folds a sign is not read so: `abs(x)` and
//! `(x^2)^(1/2)` stay beside `x^(1/2)`.
//!
//! The confinement is read from the node as it stands and must hold of the node rebuilt: a merge
//! may take the member that confined the node (`x^(1/2)*x^(1/2) + x^(3/2)*x^(3/2)` confines x by
//! each product, and merging both gives `x + x^3`, which does not), so the confined bases shrink
//! until the rebuilt node is still NaN wherever each is negative. A merge over one base never
//! removes another base's confinement (it changes exponents, never a base), so the bases kept are
//! the largest set the rebuilt node confines, a function of the node alone: the step is unique and
//! idempotent. It is a step of the root's landing (`land_root`), read top-down, so a node confined
//! by an enclosing one reads that confinement too.
//!
//! What propagates NaN is what the deployed evaluator computes (`simplipy.operators`, measured at
//! every special value of the other operand): `+`, `*`, a power's base under a nonzero NUMERIC
//! exponent (`pow(NaN, 0)` is 1, and `pow(1, NaN)` is 1, so neither a zero exponent, nor a
//! symbolic one, which may be zero, nor the exponent propagates), every unary head, `rootn`'s radicand, both operands of `max`, `min`, `mod`, `atan2`,
//! `besselj`, `bessely` and `besseli`, and `besselk`'s argument (`besselk(NaN, 0)` is `+inf`). An
//! operator outside the list propagates nothing (fail closed). What confines `t` to `t >= 0`
//! (NaN at every `t < 0`, `-inf` included): `t^r` with an exponent float64 reads as a non-integer
//! (`aligned_pow` is NaN at every negative base there), an even root `rootn(t, 2*k)`, `log(t)` and
//! `acosh(t)` (NaN below 1). A base that is `-0` on a region is not confined: its powers part on
//! the sign of that zero (`ceil(x)^(3/2)*ceil(x)^(3/2)` is `+0` on (-1, 0), `ceil(x)^3` is `-0`).

use super::*;

/// The heads through whose operand `i` (of `n`) a NaN propagates in the deployed evaluator, at
/// every value of the other operands (`simplipy.operators`, measured; `max` and `min` are IEEE
/// 754-2019's NaN-propagating `maximum` and `minimum`).
fn propagates(cx: &Cx, f: Tok, i: usize, n: usize) -> bool {
    let name = cx.view.resolve_owned(f);
    match (name.as_str(), n) {
        (
            "neg" | "inv" | "abs" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh"
            | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "exp" | "log" | "floor" | "ceil"
            | "erf" | "erfc" | "harmonic",
            1,
        ) => true,
        ("rootn", 2) => i == 0,
        ("max" | "min", _) => true,
        ("mod" | "atan2" | "besselj" | "bessely" | "besseli", 2) => true,
        ("besselk", 2) => i == 1,
        _ => false,
    }
}

/// Does float64 read the exponent `r` as a non-integer, so that `t^r` is NaN at every `t < 0`?
/// A non-integer `p/q` with `|p|` and `q` below 2^52 stays one whether the evaluator reads it as
/// the double nearest it or divides the two doubles: its distance to an integer, at least `1/q`,
/// exceeds half a unit in the last place of `p/q`.
fn reads_non_integer(r: &Rat) -> bool {
    const BOUND: u128 = 1 << 52;
    !r.is_integer()
        && r.small_parts()
            .is_some_and(|(p, q)| p.unsigned_abs() < BOUND && q.unsigned_abs() < BOUND)
}

/// Is `f` a head that confines its first operand to `>= 0` (NaN at every negative value, `-inf`
/// included): `log`, `acosh` (NaN below 1), or an even root?
fn confining_head(f: Tok, args: &[Ex], cx: &Cx) -> bool {
    match args {
        [_] => cx.view.tok_is(f, "log") || cx.view.tok_is(f, "acosh"),
        [_, Ex::Num(k)] => cx.view.tok_is(f, "rootn") && k.is_even_integer() && !k.is_zero(),
        _ => false,
    }
}

/// THE BASES `e` CONFINES: each base `t` such that `e` is NaN at every point where `t < 0`
/// (`-inf` included), as float64 evaluates it, read through the operations that propagate NaN.
/// Appended to `out` once each.
pub(crate) fn confining_bases(e: &Ex, cx: &Cx, out: &mut Vec<Ex>) {
    let push = |t: &Ex, out: &mut Vec<Ex>| {
        if !out.contains(t) {
            out.push(t.clone());
        }
    };
    match e {
        Ex::Add(v) | Ex::Mul(v) => v.iter().for_each(|m| confining_bases(m, cx, out)),
        Ex::Pow(b, x) => {
            if let Ex::Num(r) = &**x {
                if !r.is_zero() {
                    if reads_non_integer(r) {
                        push(b, out);
                    }
                    confining_bases(b, cx, out);
                }
            }
        }
        Ex::Fun(f, args) => {
            if confining_head(*f, args, cx) {
                push(&args[0], out);
            }
            for (i, a) in args.iter().enumerate() {
                if propagates(cx, *f, i, args.len()) {
                    confining_bases(a, cx, out);
                }
            }
        }
        _ => {}
    }
}

/// The base a member of a product collects over, as the like-base collection reads it (closely
/// enough to say where it may merge): `t` for `t^r` and for a root of `t` or of a power of `t`,
/// the member itself otherwise.
fn power_base<'e>(m: &'e Ex, cx: &Cx) -> &'e Ex {
    match m {
        Ex::Pow(b, _) => b,
        Ex::Fun(f, args) if args.len() == 2 && cx.view.tok_is(*f, "rootn") => match &args[0] {
            Ex::Pow(b, _) => b,
            t => t,
        },
        m => m,
    }
}

/// May a constructor confined to the bases `k` change the node `e`, beyond what its children do?
/// A product with two members over one base of `k` (`mul.collect`, `mul.collect_symbolic`,
/// `pow.odd_root_compose` in a product), a power of a power or of a root of such a base
/// (`pow.compose`, `pow.root_inverse`), and a root of a power of one (`pow.compose`).
fn sensitive(e: &Ex, k: &[Ex], cx: &Cx) -> bool {
    if k.is_empty() {
        return false;
    }
    match e {
        Ex::Mul(v) => k
            .iter()
            .any(|t| v.iter().filter(|m| power_base(m, cx) == t).count() >= 2),
        Ex::Pow(b, _) => match &**b {
            Ex::Pow(t, _) => k.contains(t),
            Ex::Fun(f, args) if args.len() == 2 && cx.view.tok_is(*f, "rootn") => {
                k.contains(power_base(b, cx))
            }
            _ => false,
        },
        Ex::Fun(f, args) if args.len() == 2 && cx.view.tok_is(*f, "rootn") => {
            matches!(&args[0], Ex::Pow(t, _) if k.contains(t))
        }
        _ => false,
    }
}

/// Is there anything the step could change: a confining member somewhere and a place where a
/// confined constructor may merge (`sensitive`, over every base)?
fn may_confine(e: &Ex, cx: &Cx) -> bool {
    fn any(e: &Ex, p: &dyn Fn(&Ex) -> bool) -> bool {
        p(e) || match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|c| any(c, p)),
            Ex::Pow(b, x) => any(b, p) || any(x, p),
            _ => false,
        }
    }
    let confines = |n: &Ex| match n {
        Ex::Pow(_, x) => matches!(&**x, Ex::Num(r) if reads_non_integer(r)),
        Ex::Fun(f, args) => confining_head(*f, args, cx),
        _ => false,
    };
    let merges = |n: &Ex| match n {
        Ex::Mul(v) => v.iter().enumerate().any(|(i, a)| {
            v[i + 1..]
                .iter()
                .any(|b| power_base(a, cx) == power_base(b, cx))
        }),
        Ex::Pow(b, _) => {
            matches!(&**b, Ex::Pow(..))
                || matches!(&**b, Ex::Fun(f, a) if a.len() == 2 && cx.view.tok_is(*f, "rootn"))
        }
        Ex::Fun(f, args) => {
            args.len() == 2 && cx.view.tok_is(*f, "rootn") && matches!(&args[0], Ex::Pow(..))
        }
        _ => false,
    };
    any(e, &confines) && any(e, &merges)
}

/// THE CONFINEMENT STEP at the root of a state (`confine.sibling`, `f64` only): every node whose
/// members propagate NaN reads the bases its members confine, and its members are rebuilt in a
/// context confined to them (and to what the enclosing nodes confine), as long as the rebuilt node
/// still confines each. `x1^(5/2) + x1^(3/2)*x1^(3/2)` is `x1^(5/2) + x1^3`;
/// `log(x1)*x1^(3/2)*x1^(3/2)` is `x1^3*log(x1)`; `x1^(3/2)*x1^(3/2)` stays.
pub(crate) fn release_confined(e: Ex, cx: &Cx) -> Ex {
    if !cx.f64_semantics()
        || !cx.confined.is_empty()
        || e.contains_wildcard(cx.view)
        || !may_confine(&e, cx)
    {
        return e;
    }
    confined_walk(&e, &[], cx).unwrap_or(e)
}

/// Does the node read its members' confinement: a sum, a product, or a head of two or more
/// operands (a unary head's operand reads its own)?
fn reads_its_members(e: &Ex) -> bool {
    match e {
        Ex::Add(_) | Ex::Mul(_) => true,
        Ex::Fun(_, v) => v.len() >= 2,
        _ => false,
    }
}

/// [`release_confined`] at `e`, which stands confined to the bases `k` by the nodes above it: the
/// node where it changes, `None` where nothing does.
fn confined_walk(e: &Ex, k: &[Ex], cx: &Cx) -> Option<Ex> {
    if !reads_its_members(e) {
        return rebuilt(e, k, cx);
    }
    let mut found = Vec::new();
    confining_bases(e, cx, &mut found);
    let mut own: Vec<Ex> = found
        .into_iter()
        .filter(|t| {
            !k.contains(t) && t.has_measure_space(cx.view) && Law::ConfineSibling.holds(t, cx)
        })
        .collect();
    loop {
        let kd: Vec<Ex> = k.iter().chain(own.iter()).cloned().collect();
        let r = rebuilt(e, &kd, cx);
        if own.is_empty() {
            return r;
        }
        let mut after = Vec::new();
        confining_bases(r.as_ref().unwrap_or(e), cx, &mut after);
        let kept: Vec<Ex> = own.iter().filter(|t| after.contains(t)).cloned().collect();
        if kept.len() == own.len() {
            return r;
        }
        own = kept;
    }
}

/// `e` rebuilt from its children, each walked under `k`, by its constructor in a context confined
/// to `k`: `None` where nothing changes.
fn rebuilt(e: &Ex, k: &[Ex], cx: &Cx) -> Option<Ex> {
    let cxk = Cx { confined: k, ..*cx };
    let out = match e {
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            let mut kids: Option<Vec<Ex>> = None;
            for (i, c) in v.iter().enumerate() {
                if let Some(r) = confined_walk(c, k, cx) {
                    kids.get_or_insert_with(|| v.clone())[i] = r;
                }
            }
            if kids.is_none() && !sensitive(e, k, cx) {
                return None;
            }
            let parts = kids.unwrap_or_else(|| v.clone());
            match e {
                Ex::Add(_) => add(parts, &cxk),
                Ex::Mul(_) => mul(parts, &cxk),
                Ex::Fun(f, _)
                    if parts.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(*f, "atan2") =>
                {
                    let mut members = parts.into_iter().map(|a| match a {
                        Ex::Mul(w) => w,
                        other => vec![other],
                    });
                    let ym = members.next().expect("two arguments");
                    let xm = members.next().expect("two arguments");
                    atan2_of(*f, ym, xm, &cxk)
                }
                Ex::Fun(f, _) => fun(*f, parts, &cxk),
                _ => unreachable!(),
            }
        }
        Ex::Pow(b, x) => {
            let nb = confined_walk(b, k, cx);
            let nx = confined_walk(x, k, cx);
            if nb.is_none() && nx.is_none() && !sensitive(e, k, cx) {
                return None;
            }
            pow(
                nb.unwrap_or_else(|| (**b).clone()),
                nx.unwrap_or_else(|| (**x).clone()),
                &cxk,
            )
        }
        _ => return None,
    };
    (out != *e).then_some(out)
}
