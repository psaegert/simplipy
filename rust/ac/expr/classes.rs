//! THE CLASS READING: the class {B, -B} a factor is read in, a class's polynomial multiples, and
//! where the sum of a sign-trade site meets another.

use super::*;

/// How a factor reads as a member of a class {B, -B} (`collect_opposite_sums`): the head of `B`
/// (`None` for the sum itself, else the odd function and an odd root's index), the terms of its
/// argument (`class_base`; a literal multiple is a sum of one term: `sin(2 x1)` and `sin(-2 x1)`,
/// `sin(x1)` and `sin(-x1)`, `sin(2)` and `sin(-2)`), and the factor's exponent over `B` (`None`
/// when it is symbolic).
pub(super) struct ClassRead<'a> {
    pub(super) head: Option<(Tok, Option<&'a Ex>)>,
    pub(super) terms: &'a [Ex],
    pub(super) exp: Option<Rat>,
}

/// THE CLASS READING, the one definition every owner asks (the product table `class_read`, the
/// orientation of like terms and of shared factors `orient_base`, the sign trade's classes
/// `site_multiple`, a term read up to sign `odd_signs_out`, and `pow`'s even powers). A class is a
/// pair {B, -B}: a sum and its negation, or an odd head -- an odd function, or an odd root of an
/// odd index of at least 3 -- of a negatable argument and of its negation. A negatable argument is
/// a sum, or a literal multiple `c u` (`multiple_literal`: a bare `u` is `1 u`, a negation `-u` is
/// `-1 u`, a literal is itself). A sum is one only where the sign may pass through it: a sum
/// holding a `Const` or an infinity, or a term that absorbs a negation, has its own owners.
pub(super) struct ClassBase<'a> {
    pub(super) head: Option<(Tok, Option<&'a Ex>)>,
    pub(super) arg: ClassArg<'a>,
}

/// The argument of a class base: a sum's terms, or a literal multiple.
pub(super) enum ClassArg<'a> {
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

pub(super) fn class_base<'a>(b: &'a Ex, view: &TokenView) -> Option<ClassBase<'a>> {
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
pub(super) fn multiple_literal(m: &Ex) -> Option<Rat> {
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
pub(super) fn positive_head(b: &Ex, view: &TokenView) -> Option<Ex> {
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

pub(super) fn class_read<'a>(f: &'a Ex, cx: &Cx) -> Option<ClassRead<'a>> {
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
pub(super) fn site_multiple(f: &Ex, cx: &Cx) -> Option<(Ex, bool)> {
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
pub(super) fn site_class(f: &Ex, cx: &Cx) -> Option<(Ex, bool)> {
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
pub(super) fn class_of(ts: &[Ex], cx: &Cx) -> Option<(Ex, bool, bool)> {
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
pub(super) fn class_multiple(ts: &[Ex], class: &Ex, cx: &Cx) -> Option<(bool, Option<bool>)> {
    use crate::ac::poly::Poly;
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
pub(super) fn class_shape(ts: &[Ex]) -> u64 {
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

/// Does the sum a site is built over occur again in the product, up to its orientation: as
/// another site's sum or inside any factor? Only then are its classes read (`sign_place`). One
/// pass over the product's sums with a hash of their shapes blind to literals and signs
/// (`shape_sig`'s), so a sum meets its constant multiples too; a hash that merely collides only
/// costs the full reading. A sum meets its polynomial multiples as well (`class_multiple`).
pub(super) fn classes_meet(out: &[Ex], site_flips: &[(usize, Ex)], cx: &Cx) -> bool {
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
