//! THE FLAT PAIRS: the pairs whose float64 round trip is flat at a point (`log o exp`, `acosh o
//! cosh`), and where a context reads that point.

use super::*;

/// The two pairs whose float64 round trip is flat at 0 (`Cx::flat_unread`): `log o exp`
/// (`fun.log_exp`) and `acosh o cosh` (`fun.acosh_cosh`), each with the context premise the root's
/// walk reads (`Cx::flat_unread`); and the log of an inverse (`fun.log_inverse`), held to the root's
/// walk in both modes, where it collapses everywhere, after the constructors have taken
/// `exp(log(1/t)) = 1/t`: float64's `log(1/t)` is `+0` where `-log(t)` is `-0`, at t = 1, which
/// for a monomial is an isolated point, whose zero's sign is no value the answer keeps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FlatPair {
    LogExp = 0,
    AcoshCosh = 1,
    LogInverse = 2,
}

/// The pairs whose flat point a context may read (`flat_read_below`), in their index order: the
/// read state of a walk is one flag per pair here. The log of an inverse has none: nothing reads
/// its zero (`flat_read_below`'s note), so it collapses wherever the root's walk meets it.
const READ_PAIRS: [FlatPair; 2] = [FlatPair::LogExp, FlatPair::AcoshCosh];

/// Whether `read` reads `pair`'s flat point: never the log of an inverse's.
fn reads(read: [bool; 2], pair: FlatPair) -> bool {
    pair != FlatPair::LogInverse && read[pair as usize]
}

/// Is `e` a pair whose collapse waits for the root's walk: in `f64` `log(exp(t))` or
/// `acosh(cosh(t))`, flat at 0, or `log(1/t)`; in `permissive` `log(1/t)`?
fn flat_pair(e: &Ex, cx: &Cx) -> Option<FlatPair> {
    let view = cx.view;
    let Ex::Fun(o, a) = e else { return None };
    if a.len() != 1 {
        return None;
    }
    // the log of an inverse waits only where its law may collapse it: over a part with a
    // variable, a slot or a fitted constant, in `f64` a monomial
    if view.tok_is(*o, "log") {
        if let Some((t, _)) = log_inverse_read(&a[0], view) {
            if (t.has_measure_space(view) || t.contains_wildcard(view))
                && (!cx.f64_semantics() || log_inverse_argument(t, view))
            {
                return Some(FlatPair::LogInverse);
            }
        }
    }
    if !cx.f64_semantics() {
        return None;
    }
    let Ex::Fun(i, b) = &a[0] else {
        return None;
    };
    if b.len() != 1 {
        return None;
    }
    if view.tok_is(*o, "log") && view.tok_is(*i, "exp") {
        Some(FlatPair::LogExp)
    } else if view.tok_is(*o, "acosh") && view.tok_is(*i, "cosh") {
        Some(FlatPair::AcoshCosh)
    } else {
        None
    }
}

/// [`flat_read_below`] for both flat-point pairs at once, each with its own `read`.
fn flat_read_below_all(e: &Ex, i: usize, read: [bool; 2], cx: &Cx) -> [bool; 2] {
    READ_PAIRS.map(|p| flat_read_below(e, i, read[p as usize], p, cx))
}

/// May the expression around child `i` of `e` read the FLAT POINT of a `pair` in that child --
/// float64's exact 0 where the pair's argument is a nonzero number inside the flat band, `+0`
/// where the argument may be `-0` or negative -- given `read` for `e`'s own? The readers are the
/// judge's, measured as `C[pair(t)] -> C[t]` over the contexts `C` of the operators: refused at
/// `C`'s root, `C` reads the zero (an inverse, a divisor, a negative power, `log`, `rootn` of a
/// negative index, `mod`'s divisor, `bessely`, `besseli`, `besselk`, `harmonic`, ...); held at the
/// root and refused under an inverse, `C` carries it to what reads it (an odd or bounded function
/// that keeps a zero, `abs`, a sum, a product beside numbers, a positive integer power, ...); held
/// in both, `C` is blind to it (`exp`, `cos`, `cosh`, `acosh`, `erfc`, an exponent over a positive
/// base, a sum beside a nonzero number). Where the pairs differ the table is each pair's:
/// `log o exp` turns a negative argument into `+0`, so the heads that read a zero's sign or a
/// domain at 0 read it (`ceil`, a fractional power, `rootn` of an even index, a product or `max`
/// beside a variable, `atan2` beside a variable), while `floor` and `atanh` carry it;
/// `acosh o cosh` drifts below `|t|` (`acosh(cosh(3))` is 2.9999999999999996), so `floor` and
/// `atanh` read it. A head the table does not name reads it.
///
/// The log of an inverse of a monomial is a zero only at points, where its value parts from the
/// collapse's in the sign of that zero alone: an isolated zero's sign, no value the answer keeps,
/// so nothing reads it, and it has no read state (`READ_PAIRS`).
pub(crate) fn flat_read_below(e: &Ex, i: usize, read: bool, pair: FlatPair, cx: &Cx) -> bool {
    // `permissive` reads one zero
    if !cx.f64_semantics() || pair == FlatPair::LogInverse {
        return false;
    }
    let view = cx.view;
    let le = pair == FlatPair::LogExp;
    let ground = |x: &Ex| !x.has_measure_space(view) && !x.contains_wildcard(view);
    match e {
        // a sum carries it, but beside a nonzero number, which a zero leaves as it is
        Ex::Add(v) => {
            read && !v
                .iter()
                .enumerate()
                .any(|(j, t)| j != i && matches!(t, Ex::Num(r) if !r.is_zero()))
        }
        // a product carries it beside numbers; `log o exp`'s zero meets a variable's sign
        Ex::Mul(v) => read || (le && v.iter().enumerate().any(|(j, f)| j != i && !ground(f))),
        Ex::Pow(_, x) if i == 0 => match &**x {
            Ex::Num(k) if k.is_integer() && k.signum() > 0 => read,
            Ex::Num(k) if !k.is_integer() && k.signum() > 0 && !le => read,
            _ => true,
        },
        // an exponent: blind over a positive finite base, read over any other (`inf^0` is 1)
        Ex::Pow(b, _) => match ground_value(b, cx) {
            Some(v) if v > 0.0 && v.is_finite() => false,
            Some(v) if v == 0.0 && !le => read,
            _ => true,
        },
        Ex::Fun(f, a) => {
            let is = |s: &str| view.tok_is(*f, s);
            let others_ground = a.iter().enumerate().all(|(j, m)| j == i || ground(m));
            if a.len() == 1 {
                let blind = ["cos", "cosh", "acosh", "exp", "erfc"]
                    .iter()
                    .any(|s| is(s))
                    || (le && is("acos"));
                let carries = [
                    "abs", "sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "erf",
                ]
                .iter()
                .any(|s| is(s))
                    || (le && (is("floor") || is("atanh")))
                    || (!le && (is("acos") || is("ceil") || is("harmonic")));
                return if blind {
                    false
                } else if carries {
                    read
                } else {
                    true
                };
            }
            if is("rootn") {
                return match (i, a.get(1)) {
                    (0, Some(Ex::Num(k)))
                        if k.is_integer() && k.signum() > 0 && (k.is_odd_integer() || !le) =>
                    {
                        read
                    }
                    _ => true,
                };
            }
            if is("min") {
                return read;
            }
            if is("max") {
                return (le && !others_ground) || read;
            }
            if is("mod") {
                return i != 0 || !others_ground || read;
            }
            if is("atan2") && a.len() == 2 {
                let other = &a[1 - i];
                return match (i, ground_value(other, cx)) {
                    (0, Some(x)) if x > 0.0 => read,
                    (0, Some(_)) => false,
                    (0, None) => le || read,
                    (_, Some(_)) => false,
                    _ => le,
                };
            }
            if is("besselj") && a.len() == 2 {
                return match (i, &a[0]) {
                    (1, Ex::Num(n)) if n.is_zero() => false,
                    (1, Ex::Num(n)) if n.is_integer() && n.signum() > 0 => read,
                    (0, _) if ground(&a[1]) => false,
                    _ => true,
                };
            }
            true
        }
        _ => true,
    }
}

/// Does composing `(b^r)^s` into `b^(r*s)` hide a reader of a pair in `b` (`flat_read_below`):
/// the two powers read a pair's zero that the one power does not, in `f64` where the pairs are
/// held? Float64's value has turned on that zero at the inner power: `1/(1/log(exp(x0)))` and
/// `1/sqrt(log(1/x0))` stay (`sqrt` keeps `-0`, which `1/...` reads, and `pow(-0, -1/2)` is
/// `+inf` like `pow(+0, -1/2)`). Nothing reads the log of an inverse's zero (`flat_read_below`),
/// so `1/(1/log(1/x0))` is `log(1/x0)`.
pub(crate) fn compose_hides_a_flat_read(b: &Ex, r: &Rat, s: &Rat, cx: &Cx) -> bool {
    if !cx.f64_semantics() || cx.flat_unread {
        return false;
    }
    let Some(rs) = r.checked_mul(s) else {
        return false;
    };
    let power = |k: Rat| Ex::Pow(Box::new(Ex::NaN), Box::new(Ex::Num(k)));
    let outer = Ex::Pow(Box::new(power(r.clone())), Box::new(Ex::Num(s.clone())));
    let at_inner = flat_read_below_all(&outer, 0, [false; 2], cx);
    let before = flat_read_below_all(&power(r.clone()), 0, at_inner, cx);
    let after = flat_read_below_all(&power(rs), 0, [false; 2], cx);
    let hidden = READ_PAIRS.map(|p| before[p as usize] && !after[p as usize]);
    flat_reads_a_pair_at(b, hidden, cx)
}

/// Does `e` hold a pair whose zero is read (`flat_read_below`), `e`'s own zero read as `read` says,
/// pair by pair?
fn flat_reads_a_pair_at(e: &Ex, read: [bool; 2], cx: &Cx) -> bool {
    fn walk(e: &Ex, read: [bool; 2], cx: &Cx) -> bool {
        if !read.iter().any(|r| *r) {
            return false;
        }
        if let Some(p) = flat_pair(e, cx) {
            if reads(read, p) {
                return true;
            }
        }
        let below = |i: usize| flat_read_below_all(e, i, read, cx);
        match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
                v.iter().enumerate().any(|(i, c)| walk(c, below(i), cx))
            }
            Ex::Pow(b, x) => walk(b, below(0), cx) || walk(x, below(1), cx),
            _ => false,
        }
    }
    walk(e, read, cx)
}

/// The inverse the log of an inverse reads (`fun.log_inverse`): `t` of `1/t`, and of `abs(1/t)`,
/// which is `1/abs(t)` in every mode, float64 included (`1/t` and `1/(-t)` are one magnitude,
/// zeros and infinities included), with whether it stood in an absolute value.
pub(super) fn log_inverse_read<'a>(e: &'a Ex, view: &TokenView) -> Option<(&'a Ex, bool)> {
    let inverse = |x: &'a Ex| match x {
        Ex::Pow(t, k) if matches!(&**k, Ex::Num(r) if *r == Rat::NEG_ONE) => Some(&**t),
        _ => None,
    };
    match e {
        Ex::Fun(f, a) if a.len() == 1 && view.tok_is(*f, "abs") => {
            inverse(&a[0]).map(|t| (t, true))
        }
        _ => inverse(e).map(|t| (t, false)),
    }
}

/// Is `t` a MONOMIAL, the argument `fun.log_inverse` reads: a product of variables, fitted
/// constants and nonzero numbers, each perhaps under an integer power or `abs`, a slot that binds
/// a variable (`?`) among them? Float64's value of a monomial is 1 only at points, where the
/// judge reads `log(1/t)` and `-log(t)` as one: a sum, a root and a function round to 1 on a
/// region (`exp(x0)` is 1.0 wherever |x0| < 1.1e-16, `ceil(x0)` on (0, 1]), where `log(1/t)` is
/// `+0` and `-log(t)` is `-0`.
pub(crate) fn log_inverse_argument(t: &Ex, view: &TokenView) -> bool {
    match t {
        Ex::Const(_) | Ex::Pi | Ex::E => true,
        Ex::Num(r) => !r.is_zero(),
        Ex::Leaf(tok) => {
            if t.contains_wildcard(view) {
                view.with_str(*tok, |s| s.starts_with('?'))
            } else {
                !is_neg_zero(t, view)
            }
        }
        Ex::Mul(v) => v.iter().all(|f| log_inverse_argument(f, view)),
        Ex::Pow(b, k) => {
            matches!(&**k, Ex::Num(k) if k.is_integer()) && log_inverse_argument(b, view)
        }
        Ex::Fun(f, a) => {
            a.len() == 1 && view.tok_is(*f, "abs") && log_inverse_argument(&a[0], view)
        }
        _ => false,
    }
}

/// WHERE NOTHING READS ITS FLAT POINT, THE PAIR COLLAPSES (`fun.log_exp`, `fun.acosh_cosh`,
/// `fun.log_inverse`): `e` with every pair a construction held (`flat_pair`) rebuilt where nothing
/// around reads its flat point (`flat_read_below`). The root of a state reads none, so
/// `log(exp(x0))` is `x0` and `sin(log(exp(x0)))` is `sin(x0)` as answers, while
/// `1/log(exp(x0))` and `1/sin(log(exp(x0)))` stay; in `permissive`, which reads no zero's sign,
/// every `log(1/t)` collapses. The nodes are rebuilt from their children, the released pairs in a
/// context that reads no flat point (`Cx::flat_unread`).
pub(crate) fn release_flat(e: Ex, cx: &Cx) -> Ex {
    fn holds_a_pair(e: &Ex, cx: &Cx) -> bool {
        flat_pair(e, cx).is_some()
            || match e {
                Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|c| holds_a_pair(c, cx)),
                Ex::Pow(b, x) => holds_a_pair(b, cx) || holds_a_pair(x, cx),
                _ => false,
            }
    }
    if cx.flat_unread || !holds_a_pair(&e, cx) {
        return e;
    }
    let free = Cx {
        flat_unread: true,
        ..*cx
    };
    flat_released(&e, [false; 2], cx, &free).unwrap_or(e)
}

/// [`release_flat`] at `e`, each pair's flat point read or not (`read`, by [`READ_PAIRS`]): the
/// node where it changes, `None` where nothing does.
fn flat_released(e: &Ex, read: [bool; 2], cx: &Cx, free: &Cx) -> Option<Ex> {
    let below = |i: usize| flat_read_below_all(e, i, read, cx);
    let rebuilt = match e {
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            let mut kids: Option<Vec<Ex>> = None;
            for (i, c) in v.iter().enumerate() {
                if let Some(r) = flat_released(c, below(i), cx, free) {
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
            let nb = flat_released(b, below(0), cx, free);
            let nx = flat_released(x, below(1), cx, free);
            (nb.is_some() || nx.is_some()).then(|| {
                pow(
                    nb.unwrap_or_else(|| (**b).clone()),
                    nx.unwrap_or_else(|| (**x).clone()),
                    cx,
                )
            })
        }
        _ => return None,
    };
    let node = rebuilt.as_ref().unwrap_or(e);
    if let (Some(p), Ex::Fun(f, a)) = (flat_pair(node, cx), node) {
        if !reads(read, p) {
            let freed = fun(*f, a.clone(), free);
            if freed != *node {
                return Some(freed);
            }
        }
    }
    rebuilt
}
