//! Full re-canonicalization ([`canon`]): an expression rebuilt bottom-up through the constructors.

use super::*;

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
