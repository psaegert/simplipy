//! A sum's terms and a product's factors, read and rebuilt: coefficient and key ([`term_split`]),
//! the orientation of the classes a term carries, the odd heads, the joins of a coefficient
//! ([`term_join`]) and a factor's base and exponent ([`factor_split`]).

use super::*;

/// `ms` without the members of `sub` (both sorted), or `None` where `sub` is no sub-multiset.
pub(super) fn take_members(ms: &[Rat], sub: &[Rat]) -> Option<Vec<Rat>> {
    let (shared, rest, left) = common_members(ms, sub);
    (shared.len() == sub.len() && left.is_empty()).then_some(rest)
}

/// The members two sorted lists share (as multisets), and what remains of each.
pub(super) fn common_members(a: &[Rat], b: &[Rat]) -> (Vec<Rat>, Vec<Rat>, Vec<Rat>) {
    let (mut shared, mut ra, mut rb) = (Vec::new(), Vec::new(), Vec::new());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp_exact(&b[j]) {
            Ordering::Equal => {
                shared.push(a[i].clone());
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                ra.push(a[i].clone());
                i += 1;
            }
            Ordering::Greater => {
                rb.push(b[j].clone());
                j += 1;
            }
        }
    }
    ra.extend(a[i..].iter().cloned());
    rb.extend(b[j..].iter().cloned());
    (shared, ra, rb)
}

/// Split an addend into `(rational coefficient, term key)`. The key is what like-term collection
/// compares: `Mul[3, x, y] -> (3, Mul[x, y])`, `Mul[x, y] -> (1, Mul[x, y])`, `sin(x) -> (1,
/// sin(x))`. The coefficient is every literal member of the product ([`Coef`]): a refused
/// partition splits as `Mul[-1, 1e160, 1e170, x] -> (-{1e160, 1e170}, x)`, and a product of
/// literals alone keys as `1`. A `Mul` whose coefficient is 0 (the uncertified `0 * t` form)
/// is NOT split -- it stays an opaque term so collection cannot manufacture a licence it does
/// not have.
///
/// The key is SIGN-NORMALIZED: an odd function's negative-literal sign
/// lifts out of the key into the coefficient (`sin(-2)` splits as `(-1, sin(2))`;
/// `Mul[5, sin(-2), x0]` as `(-5, Mul[sin(2), x0])`), so the two spellings of one
/// value share a collection key and `sin(-2) + sin(2)` cancels IN ONE PASS instead of
/// needing a render/re-parse cycle. `term_join` re-sinks the sign through the
/// trade-site route (`mul()` -> `sign_place`), so split/join still round-trips onto
/// the canonical spelling. Scope: top-level factors with a literal argument only -- a
/// sign under a `Pow` base is structure, not a spelling choice, and stays put.
pub(super) fn term_split(e: Ex, view: &TokenView) -> (Coef, Ex) {
    // Flip `f(-r)` -> `f(r)` in place for odd f; true iff a sign was extracted. RAW
    // rebuild on purpose: keys are comparison objects, and every re-JOIN routes
    // through `fun()` (which folds boundary grounds); a fold here would change the
    // key's shape mid-collection.
    fn unsign(e: &mut Ex, view: &TokenView) -> bool {
        // A positive odd power of such a function carries its sign out (`f(-u)^3 = -f(u)^3`).
        if let Ex::Pow(b, x) = e {
            if matches!(&**x, Ex::Num(n) if n.is_odd_integer() && n.signum() > 0)
                && odd_of_literal_multiple(b, view).is_some_and(|r| r.is_negative())
            {
                return unsign(b, view);
            }
            return false;
        }
        // An odd head of a negative literal multiple: the multiple's sign (`positive_head`).
        match odd_of_literal_multiple(e, view) {
            Some(r) if r.is_negative() => match positive_head(e, view) {
                Some(p) => {
                    *e = p;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
    let (c, mut key) = match e {
        Ex::Mul(v) => {
            // The literal members lead a canonical product (`mul_factor_cmp`).
            let n = v.iter().take_while(|f| matches!(f, Ex::Num(_))).count();
            if n == 0
                || v[..n]
                    .iter()
                    .any(|f| matches!(f, Ex::Num(r) if r.is_zero()))
            {
                (Coef::ONE, Ex::Mul(v))
            } else {
                let mut lits = v;
                let mut rest = lits.split_off(n);
                let c = Coef::from_members(lits.iter().map(|f| match f {
                    Ex::Num(r) => r,
                    _ => unreachable!("a literal prefix"),
                }));
                let key = match rest.len() {
                    0 => Ex::Num(Rat::ONE),
                    1 => rest.pop().expect("one factor"),
                    _ => Ex::Mul(rest), // still flat + sorted: the literals led
                };
                (c, key)
            }
        }
        other => (Coef::ONE, other),
    };
    let mut flips = 0usize;
    match &mut key {
        Ex::Mul(v) => {
            for f in v.iter_mut() {
                if unsign(f, view) {
                    flips += 1;
                }
            }
            if flips > 0 {
                // A flip edits a factor's content; re-sort so the key stays canonical
                // for equality comparison against keys split from other spellings.
                v.sort_by(|a, b| mul_factor_cmp(a, b, view));
            }
        }
        k => {
            if unsign(k, view) {
                flips += 1;
            }
        }
    }
    (if flips % 2 == 1 { c.negated() } else { c }, key)
}

/// Negate an addend exactly. Literals and infinities negate DIRECTLY; everything else
/// negates its coefficient through [`term_split`]/[`term_join`]. TOTAL: `(-1) * t` IS
/// coefficient negation in the bag representation. `None` only on the i128::MIN edge.
/// Used by the DISPLAY emitters for sign redistribution of `Mul[-1, Add]`.
/// A term with every class it holds as a factor -- a sum, an odd function of one or an odd root
/// of one, possibly under an integer power -- in the class's one orientation (the lower of `S`
/// and `-S` in the canonical order), the signs the flips move out gathered into the term's
/// coefficient. `None` where nothing flips. Raw, for matching: a repeated wildcard then binds a
/// class whichever orientation each of its terms carries (`?0/?2 + ?1/?2` reaches
/// `a/(x - y) + b/(y - x)`), and the replacement is re-canonicalized.
/// Are `x` and `y` (which differ) negations of each other through the orientation of the classes
/// they hold? Then a repeated wildcard bound to one meets the other: a near miss, which the
/// up-to-sign retry (`orient_term_classes`) may resolve. A sign-blind multiset of their leaves is
/// compared first, so a pair that differs in content is refused without building a key.
pub(crate) fn orientation_near_miss(x: &Ex, y: &Ex, view: &TokenView) -> bool {
    if leaf_sig(x) != leaf_sig(y) {
        return false;
    }
    let cx = Cx::bare(view);
    let (kx, mx) = up_to_sign_key(x, &cx);
    let (ky, my) = up_to_sign_key(y, &cx);
    (mx || my) && kx == ky
}

/// An order-free hash of the leaves of `e`, literals by magnitude and unit literals left out, so
/// that a sum and its negation, or a product and its negation, hash alike.
pub(super) fn leaf_sig(e: &Ex) -> u64 {
    use std::hash::{Hash, Hasher};
    fn one<T: Hash>(t: &T) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }
    match e {
        Ex::Leaf(t) => one(t),
        Ex::Num(r) => {
            if r.is_one() || r.checked_neg().is_some_and(|n| n.is_one()) {
                0
            } else if r.is_negative() {
                r.checked_neg().map_or(1, |n| one(&n))
            } else {
                one(r)
            }
        }
        Ex::Add(v) | Ex::Mul(v) => v.iter().map(leaf_sig).fold(0, u64::wrapping_add),
        Ex::Fun(op, args) => args.iter().map(leaf_sig).fold(one(op), u64::wrapping_add),
        Ex::Pow(b, q) => leaf_sig(b).wrapping_add(leaf_sig(q)).wrapping_mul(31),
        other => one(&std::mem::discriminant(other)),
    }
}

/// `x` read up to sign: every class factor (a sum, an odd function or odd root of one, under an
/// integer power or not) in its one orientation, the coefficient's sign dropped; and whether a
/// class moved.
fn up_to_sign_key(x: &Ex, cx: &Cx) -> (Ex, bool) {
    let fs: Vec<&Ex> = match x {
        Ex::Mul(v) => v.iter().collect(),
        _ => vec![x],
    };
    let mut moved = false;
    let mut out: Vec<Ex> = Vec::with_capacity(fs.len());
    for f in fs {
        match f {
            Ex::Num(r) => {
                let a = if r.is_negative() {
                    match r.checked_neg() {
                        Some(a) => a,
                        None => return (x.clone(), false),
                    }
                } else {
                    r.clone()
                };
                if !a.is_one() {
                    out.push(Ex::Num(a));
                }
            }
            Ex::Pow(b, q) if matches!(&**q, Ex::Num(r) if r.is_integer()) => {
                match orient_base(b, cx, &[]) {
                    Some((nb, _)) => {
                        moved = true;
                        out.push(Ex::Pow(Box::new(nb), q.clone()));
                    }
                    None => out.push(f.clone()),
                }
            }
            _ => match orient_base(f, cx, &[]) {
                Some((nb, _)) => {
                    moved = true;
                    out.push(nb);
                }
                None => out.push(f.clone()),
            },
        }
    }
    out.sort_by(|a, b| cmp_ex(a, b, cx.view));
    let key = match out.len() {
        0 => Ex::Num(Rat::int(1)),
        1 => out.pop().expect("one member"),
        _ => Ex::Mul(out),
    };
    (key, moved)
}

pub(crate) fn orient_term_classes(t: &Ex, cx: &Cx) -> Option<Ex> {
    let (factors, coef): (Vec<Ex>, Rat) = match t {
        Ex::Mul(fs) => {
            let mut c = Rat::int(1);
            let mut v = Vec::new();
            for f in fs {
                match f {
                    Ex::Num(r) => c = r.clone(),
                    _ => v.push(f.clone()),
                }
            }
            (v, c)
        }
        Ex::Num(_) => return None,
        _ => (vec![t.clone()], Rat::int(1)),
    };
    // A class is reoriented only as a whole: each class the factors carry is written in its own
    // orientation (`orient_base`), unless the term keeps a value at the class's zero
    // (`zero_limit`) that this orientation leaves off the limit. Then it is written in its other
    // orientation where that restores the limit (an odd count of odd members, as beside a log of
    // the class, whose side fixes the orientation), else as it stands.
    let mut classes: Vec<Ex> = Vec::new();
    for f in &factors {
        carried_classes(f, cx, &mut classes);
    }
    let orient = |turns: &[(Ex, Turn)]| -> Vec<(Ex, bool, bool)> {
        factors
            .iter()
            .map(|f| match orient_factor(f, cx, turns) {
                Some((g, s)) => (g, true, s),
                None => (f.clone(), false, false),
            })
            .collect()
    };
    let mut turns: Vec<(Ex, Turn)> = Vec::new();
    let mut oriented = orient(&turns);
    for _ in 0..2 * classes.len() {
        let now: Vec<Ex> = oriented.iter().map(|(g, _, _)| g.clone()).collect();
        let off = classes.iter().find(|k| {
            !turns.contains(&((*k).clone(), Turn::Keep)) && zero_off(&now, k, cx) == Some(true)
        });
        let Some(k) = off else {
            break;
        };
        match turns.iter_mut().find(|(c, _)| c == k) {
            Some(t) => t.1 = Turn::Keep,
            None => turns.push((k.clone(), Turn::Other)),
        }
        oriented = orient(&turns);
    }
    let mut odd = false;
    let mut changed = false;
    let mut out: Vec<Ex> = Vec::with_capacity(oriented.len());
    for (g, moved, s) in oriented {
        changed |= moved;
        odd ^= s;
        out.push(g);
    }
    if !changed {
        return None;
    }
    let c = if odd { coef.checked_neg()? } else { coef };
    let mut v = Vec::with_capacity(out.len() + 1);
    if !c.is_one() {
        v.push(Ex::Num(c));
    }
    v.extend(out);
    Some(if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    })
}

/// A class written against its own orientation (`orient_term_classes`): in the other one, or as
/// it stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Turn {
    Other,
    Keep,
}

/// `f` in its class's orientation (`orient_base`), and whether the flip moved a sign out of it;
/// `None` where it already is, or holds no class.
fn orient_factor(f: &Ex, cx: &Cx, turns: &[(Ex, Turn)]) -> Option<(Ex, bool)> {
    let (b, k) = match f {
        Ex::Pow(b, e) => match &**e {
            Ex::Num(r) if r.is_integer() => (&**b, Some(r)),
            _ => return None,
        },
        _ => (f, None),
    };
    // A class under a negative power whose zero set may hold a region keeps its spelling: there
    // its inverse is infinite on the region, a value the answer keeps, and the sign of that
    // infinity follows the spelling (the products' fat-zero-set refusal, `power_bag`).
    if k.is_some_and(|r| r.is_negative()) && !zero_set_null(b, cx) {
        return None;
    }
    let (nb, flipped) = orient_base(b, cx, turns)?;
    let odd = k.is_none_or(|r| r.is_odd_integer());
    let g = match k {
        None => nb,
        Some(r) => Ex::Pow(Box::new(nb), Box::new(Ex::Num(r.clone()))),
    };
    Some((g, flipped && odd))
}

/// A class base (`class_base`: a sum, or an odd head of a negatable argument) in its class's
/// orientation, and whether it flipped (each is odd in its argument, so a flip moves a sign out);
/// `None` where it already has it. A sum's terms are read with every odd head of a negative literal
/// multiple turned positive (`sin(-x1)` as `-sin(x1)`), so that a sum and its negation spell their
/// terms alike; the sum's orientation is the structurally smaller one, or as `turns` gives it. An
/// odd head's argument takes its own orientation where it holds a class (`sin(-(x1 (x1 - 3)))` is
/// `-sin(x1 (x1 - 3))`), else its positive literal (`sin(-x1)` is `-sin(x1)`).
fn orient_base(b: &Ex, cx: &Cx, turns: &[(Ex, Turn)]) -> Option<(Ex, bool)> {
    match b {
        Ex::Add(_) => {
            let Some(ClassBase {
                arg: ClassArg::Sum(ts),
                ..
            }) = class_base(b, cx.view)
            else {
                return None;
            };
            let turn = if turns.is_empty() {
                None
            } else {
                class_of(ts, cx)
                    .and_then(|(k, _, _)| turns.iter().find(|(c, _)| *c == k).map(|(_, t)| *t))
            };
            if turn == Some(Turn::Keep) {
                return None;
            }
            let mut pos: Vec<Ex> = ts.iter().map(|t| odd_signs_out(t, cx)).collect();
            pos.sort_by(|x, y| add_term_cmp(x, y, cx.view));
            let mut neg: Vec<Ex> = pos
                .iter()
                .map(raw_negate_term)
                .collect::<Option<Vec<Ex>>>()?;
            neg.sort_by(|x, y| add_term_cmp(x, y, cx.view));
            let (pb, nb) = (Ex::Add(pos), Ex::Add(neg));
            if (cmp_ex(&nb, &pb, cx.view) == Ordering::Less) != (turn == Some(Turn::Other)) {
                Some((nb, true))
            } else {
                (pb != *b).then_some((pb, false))
            }
        }
        Ex::Fun(fk, args) => {
            let ClassBase { head: Some(_), arg } = class_base(b, cx.view)? else {
                return None;
            };
            let (pos, _) = odd_head(cx.view, *fk, args)?;
            let (a, s) = match (orient_base(&args[pos], cx, turns), arg) {
                (Some(o), _) => o,
                (None, ClassArg::Multiple(m)) if multiple_literal(m)?.is_negative() => {
                    (positive_multiple(m)?, true)
                }
                _ => return None,
            };
            let mut v = args.clone();
            v[pos] = a;
            Some((Ex::Fun(*fk, v), s))
        }
        // An integer power of a class carrier: odd powers move the sign out, even ones do not. A
        // negative power whose base's zero set may hold a region keeps its spelling
        // (`orient_factor`).
        Ex::Pow(pb, q) => {
            let Ex::Num(r) = &**q else { return None };
            if !r.is_integer() || (r.is_negative() && !zero_set_null(pb, cx)) {
                return None;
            }
            let (nb, s) = orient_base(pb, cx, turns)?;
            Some((Ex::Pow(Box::new(nb), q.clone()), s && r.is_odd_integer()))
        }
        // A product holding a class carrier: each carrier in its orientation and a negative
        // coefficient moved out, so that a product's sign is read as one (`sin(-(x1 (x1 - 3)))`
        // is `-sin(x1 (x1 - 3))`).
        Ex::Mul(fs) if fs.iter().any(|f| carries_class(f, cx)) => {
            let mut flipped = false;
            let mut changed = false;
            let mut v = Vec::with_capacity(fs.len());
            for f in fs {
                match f {
                    Ex::Num(r) if r.is_negative() => {
                        let a = r.checked_neg()?;
                        flipped = !flipped;
                        changed = true;
                        if !a.is_one() {
                            v.push(Ex::Num(a));
                        }
                    }
                    _ => match orient_base(f, cx, turns) {
                        Some((g, s)) => {
                            flipped ^= s;
                            changed = true;
                            v.push(g);
                        }
                        None => v.push(f.clone()),
                    },
                }
            }
            if !changed {
                return None;
            }
            v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            Some((
                if v.len() == 1 {
                    v.pop().expect("one member")
                } else {
                    Ex::Mul(v)
                },
                flipped,
            ))
        }
        _ => None,
    }
}

/// Does `f` carry a class that `orient_base` reorients: a class base (`class_base`), or an integer
/// power or product of one?
fn carries_class(f: &Ex, cx: &Cx) -> bool {
    match f {
        Ex::Pow(b, q) => matches!(&**q, Ex::Num(r) if r.is_integer()) && carries_class(b, cx),
        Ex::Mul(fs) => fs.iter().any(|g| carries_class(g, cx)),
        f => class_base(f, cx.view).is_some(),
    }
}

/// The sums `f` carries as classes where `orient_base` reaches them (`carries_class`), each in its
/// one orientation (`class_of`, the key a class's zero is read by).
fn carried_classes(f: &Ex, cx: &Cx, out: &mut Vec<Ex>) {
    match f {
        Ex::Pow(b, q) if matches!(&**q, Ex::Num(r) if r.is_integer()) => {
            carried_classes(b, cx, out)
        }
        Ex::Mul(fs) => fs.iter().for_each(|g| carried_classes(g, cx, out)),
        f => match class_base(f, cx.view).map(|c| c.arg) {
            Some(ClassArg::Sum(ts)) => {
                if let Some((k, _, _)) = class_of(ts, cx) {
                    if !out.contains(&k) {
                        out.push(k);
                    }
                }
            }
            Some(ClassArg::Multiple(m)) => carried_classes(m, cx, out),
            None => {}
        },
    }
}

/// `a` with its negative literal coefficient made positive, raw; `None` where it has none.
pub(super) fn positive_multiple(a: &Ex) -> Option<Ex> {
    match a {
        Ex::Num(r) if r.is_negative() => Some(Ex::Num(r.checked_neg()?)),
        Ex::Mul(fs) => match fs.first() {
            Some(Ex::Num(r)) if r.is_negative() => {
                let c = r.checked_neg()?;
                let mut v: Vec<Ex> = Vec::with_capacity(fs.len());
                if !c.is_one() {
                    v.push(Ex::Num(c));
                }
                v.extend(fs[1..].iter().cloned());
                Some(if v.len() == 1 {
                    v.pop().expect("one member")
                } else {
                    Ex::Mul(v)
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// A term with each odd head of a negative literal multiple (possibly under an integer power)
/// turned positive (`positive_head`), the signs moved into the term's coefficient. Raw.
pub(super) fn odd_signs_out(t: &Ex, cx: &Cx) -> Ex {
    let flip_fun = |f: &Ex| positive_head(f, cx.view);
    let (mut c, factors): (Rat, Vec<Ex>) = match t {
        Ex::Mul(fs) => match fs.first() {
            Some(Ex::Num(r)) => (r.clone(), fs[1..].to_vec()),
            _ => (Rat::int(1), fs.clone()),
        },
        Ex::Num(_) => return t.clone(),
        _ => (Rat::int(1), vec![t.clone()]),
    };
    let mut changed = false;
    let mut out: Vec<Ex> = Vec::with_capacity(factors.len());
    for f in factors {
        let (g, sign) = match &f {
            Ex::Pow(b, e) => match (&**e, flip_fun(b)) {
                (Ex::Num(k), Some(nb)) if k.is_integer() => {
                    (Some(Ex::Pow(Box::new(nb), e.clone())), k.is_odd_integer())
                }
                _ => (None, false),
            },
            _ => match flip_fun(&f) {
                Some(nf) => (Some(nf), true),
                None => (None, false),
            },
        };
        match g {
            Some(g) => {
                changed = true;
                if sign {
                    match c.checked_neg() {
                        Some(n) => c = n,
                        None => return t.clone(),
                    }
                }
                out.push(g);
            }
            None => out.push(f),
        }
    }
    if !changed {
        return t.clone();
    }
    let mut v: Vec<Ex> = Vec::with_capacity(out.len() + 1);
    if !c.is_one() {
        v.push(Ex::Num(c));
    }
    v.extend(out);
    if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    }
}

/// A term negated in its coefficient, raw (no spelling choice: `-1 * f` stays a product).
fn raw_negate_term(t: &Ex) -> Option<Ex> {
    Some(match t {
        Ex::Num(r) => Ex::Num(r.checked_neg()?),
        Ex::Mul(fs) => match fs.first() {
            Some(Ex::Num(r)) => {
                let c = r.checked_neg()?;
                if c.is_one() {
                    if fs.len() == 2 {
                        fs[1].clone()
                    } else {
                        Ex::Mul(fs[1..].to_vec())
                    }
                } else {
                    let mut v = fs.clone();
                    v[0] = Ex::Num(c);
                    Ex::Mul(v)
                }
            }
            _ => {
                let mut v = Vec::with_capacity(fs.len() + 1);
                v.push(Ex::int(-1));
                v.extend(fs.iter().cloned());
                Ex::Mul(v)
            }
        },
        _ => Ex::Mul(vec![Ex::int(-1), t.clone()]),
    })
}

/// `None` on a negation overflow, and for the zero term an `f64` sum holds (`add`): float64's
/// `-(t + 0)` is `-0` wherever `t` vanishes, which no sum of the negated terms spells (`-t + -0` is
/// `-t`), so a sum holding that zero has no other orientation.
pub(crate) fn negate_term(t: &Ex, cx: &Cx) -> Option<Ex> {
    match t {
        Ex::Num(r) if r.is_zero() && f64_zero_signs(cx) => None,
        Ex::Num(r) => Some(Ex::Num(r.checked_neg()?)),
        Ex::PosInf => Some(Ex::NegInf),
        Ex::NegInf => Some(Ex::PosInf),
        Ex::NaN => Some(Ex::NaN),
        _ => {
            let (c, key) = term_split(t.clone(), cx.view);
            Some(term_join(c.negated(), key, cx))
        }
    }
}

/// Scale an addend by a coefficient, exactly. The distribution step of the Add flatten (see the
/// `Mul[Num.., Add]` arm there). A product a fold refuses keeps both members in the term's
/// coefficient (`1e160 * (x + 1e170*y)` scales `1e170*y` to `1e160*1e170*y`), so only `0 * inf`
/// returns `None`.
///
/// The ZERO product routes through `mul()`: `term_join` is canonical-form-
/// preserving only for NONZERO coefficients -- `term_join(0, x0)` would mint `Mul[0, x0]`,
/// a product `mul()` itself collapses (x0 is certainly finite), i.e. a locally
/// non-canonical transient. Every exposure surface happened to heal such terms bottom-up
/// (entry `canon` re-walks the parse output, the pass rebuild re-walks every child), so
/// this was never a live defect -- but the canonicity of scaled terms should be local,
/// not a consequence of who re-walks them. `mul()` owns the kept-zero licence logic
/// (licensed factors collapse, unlicensed keep `Mul[0, t]`), so the zero case lands on
/// exactly the state the healing used to reach.
pub(super) fn scale_term(t: &Ex, r: &Coef, cx: &Cx) -> Option<Ex> {
    match t {
        Ex::Num(v) => Some(match Coef::of(v).times(r) {
            p if p.rat().is_some() => Ex::Num(p.rat().expect("one member")),
            p => mul(p.items(), cx),
        }),
        Ex::Const(n) => {
            if r.is_zero() {
                // 0 * c = 0 EXACTLY (masked constants are FINITE by doctrine): the
                // structural zero must not launder into the fittable class -- the
                // forall direction fails there (the output realizes c' + .. while the
                // input realizes only .. a.e.). The sibling zero guards (term_join's
                // kept-zero absorption) carry exactly this exemption; this arm was the
                // one path around them (found by a constructor-licence sweep).
                return Some(Ex::Num(Rat::ZERO));
            }
            Some(Ex::Const(*n)) // r * c refits to c' (forall-exists; `Law::AbsorbProduct`)
        }
        Ex::PosInf | Ex::NegInf => {
            if r.is_zero() {
                return None; // 0 * inf: leave to the caller's exact paths
            }
            let flip = r.is_negative();
            Some(match (t, flip) {
                (Ex::PosInf, false) | (Ex::NegInf, true) => Ex::PosInf,
                _ => Ex::NegInf,
            })
        }
        Ex::NaN => Some(Ex::NaN),
        _ => {
            let (c, key) = term_split(t.clone(), cx.view);
            let p = c.times(r);
            if p.is_zero() {
                // r == 0 (term_split never yields c == 0): the kept-zero licence
                // decision belongs to mul() -- see the doc above.
                return Some(mul(vec![Ex::Num(Rat::ZERO), key], cx));
            }
            Some(term_join(p, key, cx))
        }
    }
}

/// Rebuild an addend from `(coefficient, key)` -- the inverse of [`term_split`], preserving
/// canonical form without a full re-canon (the key is canonical and `Num` sorts first).
/// The odd-function table, shared by the constructors (`term_join`'s sign fusion), the
/// parity machinery and the infix renderer's sign hoist -- ONE table, or it is the
/// five-arity-tables disease again.
pub(crate) fn odd_fun(view: &TokenView, f: Tok) -> bool {
    // `erf` is odd bit for bit in float64 too, signed zero included: its kernel evaluates
    // |x| and restores the sign.
    const ODD: [&str; 9] = [
        "sin", "sinh", "tan", "tanh", "asin", "asinh", "atan", "atanh", "erf",
    ];
    ODD.iter().any(|s| view.tok_is(f, s))
}

/// THE ODD HEADS, the one table every class owner reads: `f` is odd in one argument, at every value
/// and in float64 bit for bit, the sign of a zero included. The position of that argument and the
/// head's fixed operand: an odd function (`odd_fun`) in its argument; `rootn` of an odd literal
/// index of at least 3 in its radicand (the index); `besselj` of an odd integer literal order in
/// its argument (the order: `J_n(-t) = -J_n(t)`, which the kernel reads bit for bit).
pub(crate) fn odd_head<'a>(
    view: &TokenView,
    op: Tok,
    a: &'a [Ex],
) -> Option<(usize, Option<&'a Ex>)> {
    match a {
        [_] if odd_fun(view, op) => Some((0, None)),
        [_, k]
            if view.tok_is(op, "rootn")
                && matches!(k, Ex::Num(k) if k.is_odd_integer() && k.cmp_int(3) != Ordering::Less) =>
        {
            Some((0, Some(k)))
        }
        [n, _] if view.tok_is(op, "besselj") && matches!(n, Ex::Num(n) if n.is_odd_integer()) => {
            Some((1, Some(n)))
        }
        _ => None,
    }
}

/// The literal of an odd head of a literal multiple (`class_base`'s `{B, -B}` over a multiple):
/// `f(c u)` with `f` odd (`odd_fun`) or `rootn` of an odd index of at least 3, `c` the multiple's
/// literal (`multiple_literal`: 1 for a bare `u`, -1 for a negation). Such a head is a sign-trade
/// site: `f(-c u) = -f(c u)` exactly, so the sign moves between the product's coefficient and the
/// argument, and one owner (`sign_trade_flip`, `sign_place`) prices both spellings, the tie going
/// by the one sign key (`sign_key`). An odd head of a sum is a site of its own kind (its sum
/// flips).
pub(super) fn odd_of_literal_multiple(e: &Ex, view: &TokenView) -> Option<Rat> {
    match class_base(e, view)? {
        ClassBase {
            head: Some(_),
            arg: ClassArg::Multiple(m),
        } => multiple_literal(m),
        _ => None,
    }
}

/// Does a unit-coefficient join of `key` have to go through `mul()`? Only a plain product
/// (no literal, infinity or Const member: those carry or eat the sign themselves) or a lone
/// factor that holds a sign-trade site. The shape screen keeps `sign_trade_flip` off the keys
/// that cannot hold one.
fn unit_join_trades(key: &Ex, cx: &Cx) -> bool {
    let may_be_site = |f: &Ex| match f {
        Ex::Add(_) => true,
        Ex::Pow(b, _) => matches!(**b, Ex::Add(_) | Ex::Fun(..)),
        Ex::Fun(_, args) => !args.is_empty(),
        _ => false,
    };
    match key {
        Ex::Mul(v) => {
            !v.iter().any(|f| {
                matches!(f, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const(_))
                    || f.contains_const()
            }) && v
                .iter()
                .any(|f| may_be_site(f) && is_sign_trade_site(f, cx))
        }
        // A lone sum is `add()`'s to orient (`primitive_sum`), never a product's.
        Ex::Add(_) => false,
        k => may_be_site(k) && !k.contains_const() && is_sign_trade_site(k, cx),
    }
}

pub(super) fn term_join(c: Coef, key: Ex, cx: &Cx) -> Ex {
    // A coefficient a fold refused joins through `mul()`, which owns the partition and its sign
    // host, and folds it into a constant or an infinity of the key.
    let c = match c.rat() {
        Some(c) => c,
        None => {
            let mut items = c.items();
            match key {
                Ex::Mul(v) => items.extend(v),
                k => items.push(k),
            }
            let placed = mul_keeping_sums(items, cx);
            debug_assert!(
                !matches!(placed, Ex::Add(_)),
                "term_join produced a bare Add term: {placed:?}"
            );
            return placed;
        }
    };
    // The coefficient LANDS on its key here (permissive's ρ, [`land`]); a literal key is a
    // product of literals, which `mul` keeps exact, and a ground key a ground term.
    let c = if matches!(key, Ex::Num(_)) || in_ground([&key], cx) {
        c
    } else {
        land(c, cx)
    };
    // An ODD function of a LITERAL argument owns any adjacent SIGN:
    // `-1 * sin(2)` joins as `sin(-2)`, and `-5 * sin(2)` as
    // `5 * sin(-2)`. HOISTED: the fusion is no longer a private arm
    // here -- an odd function of a literal is a SIGN-TRADE SITE
    // (`sign_trade_flip`'s literal arm), so a negative coefficient joining such a
    // key routes through the full `mul()` assembly below (the trade-site
    // conditions), where the shared orbit owner prices every route onto ONE
    // spelling. The private arm fused only on THIS path, so `mul()`-built and
    // collector-built states disagreed (`-5 * sin(2)` vs `5 * sin(-2)`: two
    // fixpoints for one value, construction-history dependence).
    if c.is_one() {
        // A UNIT coefficient still decides the orientation of the key's sign-trade sites:
        // a key oriented for its old coefficient (`k * (-a - b) * x` keeps the negated sum: at
        // |k| != 1 it ties with `-k * (a + b) * x`, and the tie goes to the positive
        // coefficient) is not the cheapest spelling once the coefficient is 1 -- `mul()`
        // reads `(-a - b) * x` as `-(a + b) * x`. Returned raw, the term printed
        // `-(a + b)/..` inside the sum and re-read with the sign on its coefficient (srbf
        // prediction 117892, permissive). Such a key takes the full assembly; a key without
        // a site cannot trade and is returned as it is.
        if unit_join_trades(&key, cx) {
            let items = match key {
                Ex::Mul(v) => v,
                k => vec![k],
            };
            // A term stays a term: its key is assembled as a product, not multiplied out over a
            // sum it holds (`mul_keeping_sums`), which the sum around it decides.
            let placed = mul_keeping_sums(items, cx);
            debug_assert!(
                !matches!(placed, Ex::Add(_)),
                "term_join produced a bare Add term: {placed:?}"
            );
            return placed;
        }
        return key;
    }
    // A bare `Const` FACTOR absorbs any rational coefficient (`c * C` refits to `C'`, the
    // contract's forall-exists direction), so a Const-bearing product term canonically
    // carries NO coefficient -- and in particular no SIGN. Joining through here (collection
    // rebuilds, scaling, negation) must respect that, or raw `Mul[-1, Const, ...]` terms
    // leak into bags and the serialized sign flip-flops across parse (a term that prints
    // subtracted re-parses added once the constructor absorbs the -1).
    match key {
        // c * nan = nan for every rational c: total, fold outright.
        Ex::NaN => Ex::NaN,
        // A bare `Const` absorbs a NONZERO coefficient (the contract's forall-exists
        // refit, `Law::AbsorbProduct`) -- but an exact ZERO stays OUTSIDE, mirroring
        // `mul()`: absorbing a STRUCTURAL zero into the constant class would hand the fitter
        // a fittable constant where mathematics has an exact 0 (the masking-soundness
        // principle).
        Ex::Const(n) if !c.is_zero() => Ex::Const(n),
        Ex::Const(n) => Ex::Mul(vec![Ex::Num(c), Ex::Const(n)]),
        // An infinity absorbs the coefficient's magnitude and takes its sign, EXACTLY
        // (`c * inf = sign(c) * inf` for rational c != 0): joining a raw `Num` next to an
        // Inf factor would leave an unfolded product whose serialization is not
        // parse-stable (the constructors fold it on re-parse).
        Ex::PosInf | Ex::NegInf => {
            // 0 * (+-inf) = nan: the IEEE invalid operation, total -- no certificate
            // needed. The zero must never be absorbed sign-only.
            if c.is_zero() {
                return Ex::NaN;
            }
            let pos = matches!(key, Ex::PosInf);
            if c.is_negative() == pos {
                Ex::NegInf
            } else {
                Ex::PosInf
            }
        }
        Ex::Mul(v) => {
            // A KEPT-ZERO product (`Mul[0, t]`, the unlicensed zero-collapse) absorbs any
            // rational coefficient EXACTLY: `c * (0*t) = (c*0)*t = 0*t` pointwise, and the
            // contract has ONE zero, so there is no sign to keep. Threading the coefficient
            // through instead would mint a non-canonical `Mul[c, 0, t]` twin: `term_split`
            // reads a kept-zero term as coefficient +1 (its `!r.is_zero()` guard), so the
            // decorated twin reads `c` -- and with `c = -1` from `negate_term` the sum
            // orientation comparison loses flip-antisymmetry, leaving TWO stable sign
            // spellings around such terms (the 5 idempotence failures across the 64k/1M
            // gates: rows 29663/37873/59042 + 274133/514869).
            if matches!(v.first(), Some(Ex::Num(r)) if r.is_zero()) {
                return Ex::Mul(v);
            }
            if let Some(pos) = v.iter().position(|f| matches!(f, Ex::PosInf | Ex::NegInf)) {
                // 0 * (inf * t) = nan EVERYWHERE: t finite-nonzero gives 0 * (+-inf),
                // t = 0 gives 0 * nan, t = nan propagates. Pointwise-total, so the raw
                // path may fold it without a certificate; the zero must never be
                // absorbed sign-only.
                if c.is_zero() {
                    return Ex::NaN;
                }
                // Fold the coefficient into the Inf factor (sign only; magnitude absorbed).
                if c.is_negative() {
                    let mut v = v;
                    v[pos] = if matches!(v[pos], Ex::PosInf) {
                        Ex::NegInf
                    } else {
                        Ex::PosInf
                    };
                    return Ex::Mul(v);
                }
                return Ex::Mul(v);
            }
            if v.iter().any(|f| matches!(f, Ex::Const(_))) {
                // A NONZERO coefficient is absorbed by the constant (forall-exists);
                // an exact ZERO stays OUTSIDE, mirroring `mul()` (see the bare-Const
                // arm: a structural zero must never launder into the constant class).
                if c.is_zero() {
                    let mut v = v;
                    v.insert(0, Ex::Num(c));
                    return Ex::Mul(v);
                }
                return Ex::Mul(v);
            }
            // A NEGATIVE coefficient joining a bag that holds a sign-trade site
            // must land in the canonical placement, or this join mints the spelling
            // the orientation machinery prices away -- negate_term/scale_term/
            // divide_terms all build through here, and a priced-vs-built gap here is
            // construction-history dependence one level down (corpus row 120). The
            // route is the FULL `mul()` assembly, not `sign_place` alone: the Const and
            // negation absorption arms outrank the trade owner, and a join that skips
            // them rests the sign beside an absorbing sum that `mul()`-built states
            // fold it into (caught by the h030 confluence pin). Safe: `mul()` never
            // calls `term_join` (only `add()` does), so the recursion strictly
            // descends. Positive coefficients never trade (a trade would mint the
            // sign it sheds), so the raw join stays.
            //
            // A key that itself CARRIES Num members (an i128 overflow partition
            // bag) also takes the full assembly, for BOTH coefficient signs: the raw
            // coefficient-first insert is canonical only under "Num sorts first",
            // and with several Nums in one bag their relative order and the sign
            // HOST are `mul()`'s sign-factored accumulation's to decide -- a raw
            // `Mul[-P, N]` beside the canonical `Mul[-N, P]` is exactly the
            // route-dependence one join away.
            // A negative coefficient beside a sum that takes the sign itself (one holding an
            // infinity, a constant or a negation-absorbing member) also takes the full
            // assembly: `mul()` folds the sign into that sum, and `add()` re-settles such a
            // term the same way when a sum is read, so a raw join here would spell a term the
            // second read respells (`-x8 (log(x16) - inf)` is `x8 (inf - log(x16))`).
            let folds_sign = |f: &Ex| {
                matches!(f, Ex::Add(ts) if ts.iter().any(|m| {
                    m.contains_const()
                        || matches!(m, Ex::PosInf | Ex::NegInf)
                        || term_absorbs_negation(m)
                }))
            };
            if v.iter().any(|f| matches!(f, Ex::Num(_)))
                || (c.is_negative() && v.iter().any(|f| is_sign_trade_site(f, cx) || folds_sign(f)))
            {
                let mut items = Vec::with_capacity(v.len() + 1);
                items.push(Ex::Num(c));
                items.extend(v);
                let placed = mul_keeping_sums(items, cx);
                debug_assert!(
                    !matches!(placed, Ex::Add(_)),
                    "term_join produced a bare Add term: {placed:?}"
                );
                return placed;
            }
            let mut v = v;
            v.insert(0, Ex::Num(c));
            Ex::Mul(v)
        }
        k => {
            if matches!(k, Ex::Num(_)) || (c.is_negative() && is_sign_trade_site(&k, cx)) {
                // A bare-Num key (the other partition-bag shape) folds or
                // re-hosts through `mul()` exactly like the Num-bearing Mul above.
                let placed = mul_keeping_sums(vec![Ex::Num(c), k], cx);
                debug_assert!(
                    !matches!(placed, Ex::Add(_)),
                    "term_join produced a bare Add term: {placed:?}"
                );
                return placed;
            }
            Ex::Mul(vec![Ex::Num(c), k])
        }
    }
}

/// Split a factor into `(base, rational exponent | symbolic exponent occurrence)`.
pub(super) enum FactorExp {
    Rat(Rat),
    /// A symbolic exponent, counted by occurrence: `x^y * x^y -> (x, Sym(y), count 2)`.
    Sym(Ex),
}

pub(super) fn factor_split(e: Ex, cx: &Cx) -> (Ex, FactorExp) {
    // An EVEN `rootn(b, n)` reads as the rational power `b^(1/n)`: at an even index the
    // root IS the principal root, so the two agree pointwise (both NaN on negatives) and
    // exponent collection may treat it as a power -- this is what keeps
    // `sqrt(x) * x -> x^(3/2)` working now that even roots are no longer spelled as powers.
    // An ODD root must NEVER be read this way: `rootn(-8,3) = -2` where `(-8)^(1/3)` is
    // NaN, so they are different functions and collecting them would fabricate values.
    // In `f64` a square root of a base that may be a signed zero is not `b^(1/2)`, the `pow` call
    // (`pow_member`'s half powers: `sqrt(-0) = -0`, `rootn(-0, 2) = +0`).
    if let Ex::Fun(op, args) = &e {
        if args.len() == 2 && cx.view.tok_is(*op, "rootn") {
            if let Ex::Num(idx) = &args[1] {
                if idx.is_even_integer()
                    && idx.cmp_int(2) != Ordering::Less
                    && !(idx.cmp_int(2) == Ordering::Equal && sqrt_parts_from_rootn(&args[0], cx))
                {
                    {
                        if let Some(r) = idx.checked_inv() {
                            let Ex::Fun(_, args) = e else { unreachable!() };
                            let mut it = args.into_iter();
                            return (it.next().unwrap(), FactorExp::Rat(r));
                        }
                    }
                }
            }
        }
    }
    match e {
        Ex::Pow(b, ex) => match *ex {
            Ex::Num(r) => (*b, FactorExp::Rat(r)),
            sym => (*b, FactorExp::Sym(sym)),
        },
        other => (other, FactorExp::Rat(Rat::ONE)),
    }
}

/// `(base, rational exponent)` when `e` reads faithfully as a rational power. Same rule as
/// `factor_split`: a `Pow` with a rational exponent always, an EVEN `rootn` as `b^(1/n)`,
/// an odd `rootn` never.
pub(super) fn as_rational_power<'a>(e: &'a Ex, cx: &Cx) -> Option<(&'a Ex, Rat)> {
    match e {
        Ex::Pow(b, ex) => match &**ex {
            Ex::Num(r) => Some((&**b, r.clone())),
            _ => None,
        },
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            let Ex::Num(idx) = &args[1] else {
                return None;
            };
            if idx.is_even_integer()
                && idx.cmp_int(2) != Ordering::Less
                && !(idx.cmp_int(2) == Ordering::Equal && sqrt_parts_from_rootn(&args[0], cx))
            {
                idx.checked_inv().map(|r| (&args[0], r))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A POWER OF `exp` IS AN `exp`: `(e^a)^b = e^(a*b)`. Returns the composed `exp(a*b)` when
/// `base` reads faithfully as a power of Euler's number AND the product folds to a single
/// node; `None` otherwise. `exp(a)` IS `E^a` and the leaf `E` is `E^1`, so this is the same
/// "reads faithfully as a power" idea `as_rational_power` implements for `Pow` and `rootn`
/// -- it needs its own entry point only because an `exp` exponent is an arbitrary SUBTREE,
/// not a rational, so it does not fit that helper's signature.
///
/// Both power spellings call it: `pow(base, b)` passes `b`, and `rootn(base, n)` passes
/// `1/n`. That is what makes the reduction route-independent and what
/// makes `rootn(e, n) -> exp(1/n)` -- once a single-BASE arm -- one
/// INSTANCE of a principle rather than a patch for one shape.
///
/// THE FOLD CONDITION IS BOTH GATES AT ONCE, which is why the arm needs no licence:
///
///   * mu. With a symbolic `a` the product needs a `Mul` node and the composition stops
///     paying: `pow(exp x0, x1)` and `exp(x0*x1)` both price 24,000. UNDER THE SYMBOL
///     TABLE the LITERAL-exponent readings moved and no longer all agree:
///     `pow(exp x0, 3)` ties its composition at 18,000, but `rootn(exp x0, 3)` is 21,000
///     against 19,000 for `exp(x0/3)` -- a 2,000 DESCENT the arm declines, because a
///     `rootn` head (6, elementary in the table) is dearer than the `Mul` plus unit
///     fraction the composed spelling pays, where the flat unit priced them the other way
///     round (26,000 against 27,000). The arm's gate was never mu -- it is the FOLD
///     CONDITION, and that has not moved -- so this is a MISSED descent, not an unsound
///     one, and it stays where it already lives: with the RULES. `pow exp (-1) _0 ->
///     exp neg _0`
///     (a non-unit literal `a` against a symbolic `b`) is shipped and still needed. The
///     `E`-base rule `pow np.e _0 -> exp _0` is NOT -- `a = 1` needs no fold, so the arm
///     covers every exponent and the mine can no longer mint it.
///     Inside the fold condition the descent is universal, not merely measured: one whole
///     operator node disappears, and §10.10's `L(n) = log2(1+|n|)` is subadditive on the
///     numerator and the denominator alike, so the surviving literal never costs more than
///     the two it replaces.
///   * soundness. The one configuration where `(e^a)^b` and `e^(a*b)` disagree is `0 * inf`
///     in the exponent: at `a = -inf, b = 0` the source is `0^0 = 1` while the target is
///     `e^nan = nan` -- defined -> undefined, which soundness forbids at ZERO measure
///     tolerance (the same trap the pow-of-pow arm above guards with `never_infinite`).
///     `Ex::Num` is finite by construction (`PosInf`/`NegInf` are their own variants) and
///     `a = 1` is finite, so inside the fold condition that configuration is unreachable.
///
/// TOTAL wherever it fires: `exp` is strictly positive on the whole real line, so the odd
/// signed root and the even principal root coincide with the power on this base, and every
/// non-finite argument agrees on both sides (`a = -inf` gives 0 = 0 for `b > 0` and
/// +inf = +inf for `b < 0` under the one-zero convention; NaN propagates on both).
///
/// GATED on the config declaring `exp`: the rules this subsumes were gated for free by the
/// config's own vocabulary, and a constructor arm is not.
///
/// The laws applied here (`ac::laws::Law`): `pow.exp_compose`.
pub(super) fn compose_e_power(base: &Ex, exponent: &Ex, cx: &Cx) -> Option<Ex> {
    // `None` marks the bare leaf `E`, i.e. `a = 1`, where the product IS `exponent` for
    // ANY exponent -- no `Mul` node is created, so the fold condition holds unconditionally
    // and this is the arm that drops the `E` leaf outright (`pow(e, x0)` 24,000 -> 16,000).
    let a = match base {
        Ex::E => None,
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "exp") => Some(&args[0]),
        _ => return None,
    };
    let composed = match a {
        None => exponent.clone(),
        // `exp(1)` denotes the same value as `E` and takes the same unconditional path.
        // UNREACHABLE while the write half holds -- `fun` folds `exp(1)` to `E`, so no
        // such node exists -- and kept anyway: this helper's whole contract is that the
        // two spellings behave identically, and an arm that states it cannot go stale
        // the way a comment can.
        Some(Ex::Num(ra)) if ra.is_one() => exponent.clone(),
        Some(Ex::Num(ra)) => match exponent {
            Ex::Num(rb) => Ex::Num(ra.checked_mul(rb)?),
            _ => return None,
        },
        Some(_) => return None,
    };
    let exp_tok = cx.view.intern("exp");
    if cx.view.arity(exp_tok) != Some(1) {
        return None;
    }
    // Through `fun`, so a composed exponent of exactly 1 comes back out as the leaf `E`
    // rather than as `exp(1)` -- the two halves of the invariant meeting.
    Some(fun(exp_tok, vec![composed], cx))
}
