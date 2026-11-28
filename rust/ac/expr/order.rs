//! The canonical orders: the total order of expressions ([`cmp_ex`]), the bag orders of sums and
//! products, and the tie order of equally priced spellings ([`sign_key`], [`tie_cmp`]).

use super::*;

/// Rank for the canonical total order (cross-kind comparisons decide on this alone).
fn rank(e: &Ex) -> u8 {
    match e {
        Ex::Num(_) => 0,
        Ex::Pi => 1,
        Ex::E => 2,
        Ex::PosInf => 3,
        Ex::NegInf => 4,
        Ex::NaN => 5,
        Ex::Const(_) => 6,
        Ex::Leaf(_) => 7,
        Ex::Fun(..) => 8,
        Ex::Pow(..) => 9,
        Ex::Add(_) => 10,
        Ex::Mul(_) => 11,
    }
}

/// The canonical total order. View-dependent because `Leaf`/`Fun` tokens compare by their
/// STRINGS (stable across engines/interning orders), not by id. Order is a CANONICALIZATION
/// concern only -- bags denote the same function in any order -- so no soundness rests here,
/// but determinism does: this order is the whole fix for the ORDER axis of the invariance
/// defect, so it must be total and stable.
///
/// NAMES LAST. Two expressions compare first with every named constant's name read as the same
/// (`<constant_a>*x1` and `<constant_b>*x1` tie there), and only where they tie so, by the names
/// (a fitted constant's name too, `ConstName`; `<constant>` first). So a consistent renaming of
/// named constants changes no decision this order takes between expressions that differ in more
/// than their names: which part a sum's factoring takes out, which orientation a class keeps.
/// An expression without a named constant compares as before.
pub fn cmp_ex(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    let named = std::cell::Cell::new(false);
    match cmp_ex_in(a, b, view, Some(&named)) {
        Ordering::Equal if named.get() => cmp_ex_in(a, b, view, None),
        o => o,
    }
}

/// [`cmp_ex`]'s two readings: `blind` (names read as the same, recording whether two differed),
/// or by the names (`None`).
fn cmp_ex_in(a: &Ex, b: &Ex, view: &TokenView, blind: Option<&std::cell::Cell<bool>>) -> Ordering {
    let (ra, rb) = (rank(a), rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Ex::Num(x), Ex::Num(y)) => x.cmp_exact(y),
        (Ex::Leaf(x), Ex::Leaf(y)) => match blind {
            Some(seen) if x != y && view.is_named(*x) && view.is_named(*y) => {
                seen.set(true);
                Ordering::Equal
            }
            _ => view.str_cmp(*x, *y),
        },
        (Ex::Const(m), Ex::Const(n)) => match (blind, m.0, n.0) {
            (Some(seen), _, _) => {
                if m != n {
                    seen.set(true);
                }
                Ordering::Equal
            }
            (None, Some(x), Some(y)) => view.str_cmp(x, y),
            (None, x, y) => x.is_some().cmp(&y.is_some()),
        },
        (Ex::Fun(fa, va), Ex::Fun(fb, vb)) => view
            .str_cmp(*fa, *fb)
            .then_with(|| cmp_vec_in(va, vb, view, blind)),
        (Ex::Pow(ba, ea), Ex::Pow(bb, eb)) => {
            cmp_ex_in(ba, bb, view, blind).then_with(|| cmp_ex_in(ea, eb, view, blind))
        }
        (Ex::Add(va), Ex::Add(vb)) | (Ex::Mul(va), Ex::Mul(vb)) => cmp_vec_in(va, vb, view, blind),
        _ => Ordering::Equal, // same rank, no payload (Pi, E, infinities, NaN)
    }
}

fn cmp_vec_in(
    a: &[Ex],
    b: &[Ex],
    view: &TokenView,
    blind: Option<&std::cell::Cell<bool>>,
) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match cmp_ex_in(x, y, view, blind) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    a.len().cmp(&b.len())
}

/// [`cmp_ex`] on two sequences, position by position, then by length: names last.
pub(super) fn cmp_vec(a: &[Ex], b: &[Ex], view: &TokenView) -> Ordering {
    let named = std::cell::Cell::new(false);
    match cmp_vec_in(a, b, view, Some(&named)) {
        Ordering::Equal if named.get() => cmp_vec_in(a, b, view, None),
        o => o,
    }
}

/// STRIPPED comparators -- the bag orders of the canonical form.
///
/// Add terms compare by their COEFFICIENT-STRIPPED structural key (the `term_split` key),
/// with constant-like terms (pure literals, `Const`) LAST, ties broken by the coefficient.
/// Mul factors compare by their EXPONENT-STRIPPED base (the `factor_split` base), with the
/// rational coefficient FIRST, ties broken by the exponent.
///
/// Stripping is LOAD-BEARING for the normal form, not aesthetics: under the raw structural
/// order a term's position depends on its coefficient's sign (`Mul[-2, a]` sorts before
/// `Mul[3, b]` because -2 < 3), so "make the lead coefficient positive" ALTERNATES on bags
/// like `-2a + 3b` -- negating every term re-sorts a different term to the front, negative
/// again. With stripped keys the lead term of a bag and of its negation is the SAME term,
/// and the sign rule of the primitive form (below) is well-defined. Two side benefits: sign
/// or exponent edits no longer relocate members (edit locality), and sums read in the
/// classical polynomial order (`x + 1`, not `1 + x`).
fn is_constlike(e: &Ex) -> bool {
    matches!(
        e,
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const(_)
    )
}

pub fn add_term_cmp(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    let (ca, ka) = term_split(a.clone(), view);
    let (cb, kb) = term_split(b.clone(), view);
    let ga = is_constlike(&ka) as u8;
    let gb = is_constlike(&kb) as u8;
    ga.cmp(&gb)
        .then_with(|| cmp_ex(&ka, &kb, view))
        .then_with(|| ca.order(&cb))
}

pub fn mul_factor_cmp(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    // The rational coefficient (a bare Num) leads the product; everything else by base.
    let ga = !matches!(a, Ex::Num(_)) as u8;
    let gb = !matches!(b, Ex::Num(_)) as u8;
    if ga != gb {
        return ga.cmp(&gb);
    }
    let (ba, ea) = factor_split_ref(a);
    let (bb, eb) = factor_split_ref(b);
    cmp_ex(ba, bb, view).then_with(|| cmp_ex(ea, eb, view))
}

/// Borrowing (base, exponent) view of a factor for the comparator (`x` is `x^1`).
fn factor_split_ref(e: &Ex) -> (&Ex, &Ex) {
    static ONE: Ex = Ex::Num(Rat::ONE);
    match e {
        Ex::Pow(b, ex) => (b, ex),
        other => (other, &ONE),
    }
}

/// THE SIGN TIE KEY (one key after the price, for every placement tie: the sign trade, the product
/// table, the reduction ordering): fewest signs, then outermost, then the structural order
/// (`tie_cmp`). A sign is a negative number -- a product's or a sum term's negative coefficient (a
/// negation is the coefficient -1) or a negative literal argument. A literal exponent is not a
/// sign (a negative one is a reciprocal, written as a division), nor is an infinity's sign, which
/// is its value (`-inf` and `log(0)` are one constant, not a placement). A sign inside a ground
/// argument of a function (no variable, wildcard or `<constant>`) belongs to that constant and is
/// not counted, so a constant keeps its sign on its literal (`sin(-2)` stays, `-sin(2)` is filed
/// as `sin(-2)`). The key is the number of the counted signs and the sum of their depths (the
/// root at depth 0): fewer, then outer.
pub(crate) fn sign_key(e: &Ex, view: &TokenView) -> (u32, u32) {
    fn walk(e: &Ex, depth: u32, view: &TokenView, acc: &mut (u32, u32)) {
        match e {
            Ex::Num(r) if r.is_negative() => {
                acc.0 += 1;
                acc.1 += depth;
            }
            Ex::Add(v) | Ex::Mul(v) => v.iter().for_each(|x| walk(x, depth + 1, view, acc)),
            Ex::Pow(b, q) => {
                walk(b, depth + 1, view, acc);
                if !matches!(**q, Ex::Num(_)) {
                    walk(q, depth + 1, view, acc);
                }
            }
            Ex::Fun(_, args) => {
                for a in args {
                    if a.has_measure_space(view) || a.contains_wildcard(view) {
                        walk(a, depth + 1, view, acc);
                    }
                }
            }
            _ => {}
        }
    }
    let mut acc = (0, 0);
    walk(e, 0, view, &mut acc);
    acc
}

/// The order of two spellings of an equal price (`sign_key`, then `cmp_ex`).
pub(crate) fn tie_cmp(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    sign_key(a, view)
        .cmp(&sign_key(b, view))
        .then_with(|| cmp_ex(a, b, view))
}
