//! A PRODUCT AT THE ZERO OF A CLASS: a factor's parity and sign there, the limit the product keeps
//! ([`zero_limit`]) and its magnitude.

use super::*;

/// How a factor reads in a class `S` (a sum in its one orientation): free of it, even (alike over
/// `S` and `-S`), odd (negated with it), defined beside the zero of `S` on one side only, or
/// neither. Compositional: `S` and `-S` are odd; a product's parity is the product of its factors'
/// (a factor free of `S` is even); an odd function or an odd root keeps its argument's; an even
/// function of anything in `S`, an even integer power of it, or any function of something even in
/// it, is even; a sum that holds `S` as an exact polynomial factor has the parity of the factor's
/// multiplicity (`class_multiple`), any other sum the parity its terms share. A log, an even root
/// or a fractional or symbolic power of something odd in `S` whose sign beside the zero is read is
/// defined only on the side where that is positive (`Side(true)`: where `S` is negative), and so
/// is a product or an integer power holding one; members defined on opposite sides leave no domain
/// beside the zero, and are not read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Parity {
    Free,
    Even,
    Odd,
    Side(bool),
    Unknown,
}

/// A factor read at the zero of a class `S` (READING A): its parity in `S`; whether its sign at
/// the zero differs from its sign beside the zero on the side where the oriented sum is positive
/// (for a factor defined on one side only, on that side); and that sign beside the zero
/// (negative?). The sum spelled in its other orientation is `+0` at its zero but negative beside
/// it; a product counts its members (a literal or a factor free of `S` signs both alike); an odd
/// power, an odd function and an odd root keep the sign of zero; an even power or an even function
/// differs nowhere, and neither does a one-sided head (`log(±0)` is `-inf`, a fractional power of
/// `±0` is `+0` or `+inf`, each with the sign it has beside the zero). `None` where that is not
/// read (a sum that holds `S` times a factor its terms share, `3 x2 - x1 x2`, or as a polynomial
/// factor beside a cofactor of no constant sign at the zero, any other head).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct ZeroRead {
    pub(super) parity: Parity,
    pub(super) differs: Option<bool>,
    below: Option<bool>,
}

/// `zero_read`'s verdicts within a simplify call (`RegionScope`), per token view, class and factor.
/// No other part of the context is in the key because the reading reads none: `read_walk`,
/// `class_of`, `class_multiple` and `zero_mag` read the token view and the expression's structure
/// only, never a certificate, the mode or a confinement (`Cx::confined`).
pub(super) type ReadMemo =
    rustc_hash::FxHashMap<usize, rustc_hash::FxHashMap<Ex, rustc_hash::FxHashMap<Ex, ZeroRead>>>;

thread_local! {
    pub(super) static READ_MEMO: std::cell::RefCell<Option<ReadMemo>> = const { std::cell::RefCell::new(None) };
}

/// The one reading of a factor in a class: membership by parity and the sign at the zero, read
/// together and decided once per call for each factor and class however often products meet them.
pub(super) fn zero_read(e: &Ex, class: &Ex, cx: &Cx) -> ZeroRead {
    let Ex::Add(cts) = class else {
        return ZeroRead {
            parity: Parity::Unknown,
            differs: None,
            below: None,
        };
    };
    let view = cx.view as *const TokenView as usize;
    let hit = READ_MEMO.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|t| t.get(&view)?.get(class)?.get(e).copied())
    });
    if let Some(hit) = hit {
        return hit;
    }
    let r = read_walk(e, class, cts.len(), class_shape(cts), cx);
    READ_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.entry(view)
                .or_default()
                .entry(class.clone())
                .or_default()
                .insert(e.clone(), r);
        }
    });
    r
}

fn parity_in(e: &Ex, class: &Ex, cx: &Cx) -> Parity {
    zero_read(e, class, cx).parity
}

/// The read of a product from its members' reads: its parity is theirs multiplied, and on the side
/// a one-sided member is defined on (the side where the class's key is positive otherwise) the
/// members whose sign at the zero differs from their sign beside it are counted, and so are those
/// negative beside it. An odd member read on the side where the key is negative changes both.
/// Members defined on opposite sides leave no domain beside the zero: not read.
fn product_read(reads: impl Iterator<Item = ZeroRead> + Clone) -> ZeroRead {
    use Parity::{Even, Free, Odd, Side, Unknown};
    let unread = ZeroRead {
        parity: Unknown,
        differs: None,
        below: None,
    };
    let mut side: Option<bool> = None;
    for r in reads.clone() {
        match r.parity {
            Unknown => return unread,
            Side(s) if side.is_some_and(|t| t != s) => return unread,
            Side(s) => side = Some(s),
            _ => {}
        }
    }
    let other = side == Some(true);
    let (mut odd, mut held) = (false, false);
    let (mut differs, mut below) = (Some(false), Some(false));
    for r in reads {
        let turn = match r.parity {
            Odd => {
                odd = !odd;
                held = true;
                other
            }
            Even | Side(_) => {
                held = true;
                false
            }
            _ => false,
        };
        differs = differs.zip(r.differs).map(|(a, b)| a ^ b ^ turn);
        below = below.zip(r.below).map(|(a, b)| a ^ b ^ turn);
    }
    let parity = match (side, odd, held) {
        (Some(s), _, _) => Side(s),
        (None, true, _) => Odd,
        (None, false, true) => Even,
        _ => Free,
    };
    ZeroRead {
        parity,
        differs,
        below,
    }
}

fn read_walk(e: &Ex, class: &Ex, n: usize, nk: u64, cx: &Cx) -> ZeroRead {
    use Parity::{Even, Free, Odd, Side, Unknown};
    let read = |parity, differs, below| ZeroRead {
        parity,
        differs,
        below,
    };
    let unread = read(Unknown, None, None);
    let free = |below| read(Free, Some(false), below);
    let even = read(Even, Some(false), Some(false));
    // A head defined where its argument is positive, over an argument odd in `S` whose sign beside
    // the zero is read: defined on the side where that sign is positive, where the head is
    // positive (a root or a power) or of the sign `below` gives (a log: negative where its argument
    // tends to zero, read for no argument here).
    let one_sided = |r: ZeroRead, below: Option<bool>| match (r.parity, r.below) {
        (Odd, Some(b)) => read(Side(b), Some(false), below),
        (Side(s), Some(false)) => read(Side(s), Some(false), below),
        _ => unread,
    };
    let is_even_fun = |op: Tok| {
        cx.view.tok_is(op, "abs") || cx.view.tok_is(op, "cos") || cx.view.tok_is(op, "cosh")
    };
    match e {
        Ex::Num(r) => free(Some(r.is_negative())),
        Ex::Add(ts) => {
            if ts.len() == n && class_shape(ts) == nk {
                if let Some((k, negative, scaled)) = class_of(ts, cx) {
                    if k == *class {
                        let sign = (!scaled).then_some(negative);
                        return read(Odd, sign, sign);
                    }
                }
            }
            // A polynomial multiple of the class (`class_multiple`): odd or even in it, with the
            // sign of its cofactor at the zero, and `+0` there as an exactly cancelling sum.
            if let Some((odd, sign)) = class_multiple(ts, class, cx) {
                return read(if odd { Odd } else { Even }, sign, sign);
            }
            let reads: Vec<ZeroRead> = ts.iter().map(|t| read_walk(t, class, n, nk, cx)).collect();
            // A sum holding a term defined on one side only reads as its one-sided terms where those
            // are infinite at the zero (of one side and one sign) and the rest finite there: they
            // dominate it at the zero and beside it (`(3 - x1)^(-1/2) + x2*(x1 - 3)`).
            if reads.iter().any(|r| matches!(r.parity, Side(_))) {
                let (mut side, mut below) = (None, None);
                for (t, r) in ts.iter().zip(&reads) {
                    match (r.parity, zero_mag(t, class, cx)) {
                        (Side(s), Mag::Inf) if side.is_none_or(|x| x == s) => {
                            let Some(b) = r.below.filter(|b| below.is_none_or(|x| x == *b)) else {
                                return unread;
                            };
                            (side, below) = (Some(s), Some(b));
                        }
                        (Free | Even | Odd, Mag::Zero | Mag::Finite) => {}
                        _ => return unread,
                    }
                }
                return side.map_or(unread, |s| read(Side(s), Some(false), below));
            }
            let (mut odd, mut even_t, mut free_t) = (false, false, false);
            for r in &reads {
                match r.parity {
                    Unknown | Side(_) => return unread,
                    Odd => odd = true,
                    Even => even_t = true,
                    Free => free_t = true,
                }
            }
            match (odd, even_t || free_t) {
                (true, true) => unread,
                (true, false) => read(Odd, None, None),
                _ if even_t => read(Even, None, None),
                _ => free(None),
            }
        }
        Ex::Mul(fs) => {
            let r = product_read(fs.iter().map(|f| read_walk(f, class, n, nk, cx)));
            if r.parity == Free {
                free(r.below)
            } else {
                r
            }
        }
        Ex::Pow(b, q) => {
            if read_walk(q, class, n, nk, cx).parity != Free {
                return unread;
            }
            let rb = read_walk(b, class, n, nk, cx);
            match (rb.parity, &**q) {
                (Free, Ex::Num(r)) if r.is_odd_integer() => free(rb.below),
                (Free, Ex::Num(r)) if r.is_even_integer() => free(Some(false)),
                (Free, _) => free(None),
                (Unknown, _) => unread,
                (Side(s), Ex::Num(r)) if r.is_even_integer() => {
                    read(Side(s), Some(false), Some(false))
                }
                (_, Ex::Num(r)) if r.is_even_integer() => even,
                (p, Ex::Num(r)) if r.is_odd_integer() => read(p, rb.differs, rb.below),
                (Even, _) => read(Even, None, None),
                // A fractional or symbolic power: defined where its base is positive.
                _ => one_sided(rb, Some(false)),
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && odd_fun(cx.view, *op) => {
            let ra = read_walk(&args[0], class, n, nk, cx);
            match ra.parity {
                Free => free(None),
                Side(_) => unread,
                _ => ra,
            }
        }
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            if read_walk(&args[1], class, n, nk, cx).parity != Free {
                return unread;
            }
            let ra = read_walk(&args[0], class, n, nk, cx);
            let odd_index = matches!(&args[1], Ex::Num(k) if k.is_odd_integer());
            let even_index = matches!(&args[1], Ex::Num(k) if k.is_even_integer());
            match ra.parity {
                Free if odd_index => free(ra.below),
                Free => free(Some(false)),
                Unknown => unread,
                _ if odd_index => ra,
                Even => even,
                _ if even_index => one_sided(ra, Some(false)),
                _ => unread,
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "log") => {
            let ra = read_walk(&args[0], class, n, nk, cx);
            match ra.parity {
                Free => free(None),
                Even => read(Even, None, None),
                // negative beside the zero where its argument vanishes there, positive where it is
                // infinite
                _ => one_sided(
                    ra,
                    match zero_mag(&args[0], class, cx) {
                        Mag::Zero => Some(true),
                        Mag::Inf => Some(false),
                        _ => None,
                    },
                ),
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && is_even_fun(*op) => {
            match read_walk(&args[0], class, n, nk, cx).parity {
                Free => free(None),
                Unknown | Side(_) => unread,
                _ => even,
            }
        }
        Ex::Fun(_, args) => {
            let mut acc = Free;
            for a in args {
                match read_walk(a, class, n, nk, cx).parity {
                    Free => {}
                    Even => acc = Even,
                    _ => return unread,
                }
            }
            if acc == Free {
                free(None)
            } else {
                read(Even, None, None)
            }
        }
        _ => free(None),
    }
}

/// READING A AT THE ZERO OF A CLASS `S`: the value a product keeps at the zero of `S` -- the side
/// whose limit it is (`true`: where the class's key is negative) and whether the factors as written
/// leave the value off that limit -- or `None` where it keeps none. It keeps one where that value
/// is a limit: where the product is even in `S` (two or more odd members, an even count, so its two
/// sides agree), or defined beside the zero on one side only (`Parity::Side`: a log, an even root,
/// a fractional or symbolic power of `S` among its members, which fixes the side), and the product
/// is not NaN there in every orientation (an infinity times a zero, `log(S) S`). An odd product
/// with no one-sided member has a two-signed pole or zero, whose sign is no value the answer keeps.
/// The float value at the zero is the limit exactly where the members whose sign there differs
/// from their sign beside the zero on that side are even in number (`ZeroRead::differs`): written
/// in the orientation positive on that side, every member's base is `+0` there. Where a member is
/// not read (an odd root of a product of unknown sign, `cbrt(x2 (x1 - 3))`, or a class scaled by a
/// factor of unknown sign), the product keeps nothing and its orientation is its cheapest. A lossy
/// context keeps no value at a pole. The product owners all ask this one predicate: the sign trade
/// (`sign_place`), the product table's lone inverse (`Class::table`) and the sum's like terms
/// (`orient_term_classes`).
pub(super) fn zero_limit(factors: &[Ex], class: &Ex, cx: &Cx) -> Option<(bool, bool)> {
    if cx.permits(Permissive::Agreement) {
        return None;
    }
    let reads: Vec<ZeroRead> = factors.iter().map(|f| zero_read(f, class, cx)).collect();
    let r = product_read(reads.iter().copied());
    let side = match r.parity {
        Parity::Side(s) => s,
        Parity::Even if reads.iter().filter(|r| r.parity == Parity::Odd).count() >= 2 => false,
        _ => return None,
    };
    let off = r.differs?;
    let mags = factors.iter().map(|f| zero_mag(f, class, cx));
    (product_mag(mags) != Mag::Nan).then_some((side, off))
}

/// Whether a product's value at the zero of a class is off the limit it keeps there
/// (`zero_limit`); `None` where it keeps none.
pub(super) fn zero_off(factors: &[Ex], class: &Ex, cx: &Cx) -> Option<bool> {
    zero_limit(factors, class, cx).map(|(_, off)| off)
}

/// A factor's magnitude at the zero of a class `S` (`zero_limit`'s NaN test): zero, finite and
/// nonzero, infinite, NaN, or not read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mag {
    Zero,
    Finite,
    Inf,
    Nan,
    Unknown,
}

/// The magnitude of a product from its members': NaN where an infinity meets a zero.
fn product_mag(mags: impl Iterator<Item = Mag>) -> Mag {
    let (mut zero, mut inf) = (false, false);
    for m in mags {
        match m {
            Mag::Nan => return Mag::Nan,
            Mag::Unknown => return Mag::Unknown,
            Mag::Zero => zero = true,
            Mag::Inf => inf = true,
            Mag::Finite => {}
        }
    }
    match (zero, inf) {
        (true, true) => Mag::Nan,
        (true, false) => Mag::Zero,
        (false, true) => Mag::Inf,
        _ => Mag::Finite,
    }
}

/// `e`'s magnitude at the zero of a class `S`: a factor free of `S` is finite there (at a generic
/// point), a sum of the class and a sum of terms that vanish there are zero, and the heads carry it:
/// a negative power or a negative root index inverts it, a log of a zero is infinite, an odd
/// function keeps a zero, `exp`, `cos` and `cosh` of a finite argument are finite.
fn zero_mag(e: &Ex, class: &Ex, cx: &Cx) -> Mag {
    let inverse = |m: Mag| match m {
        Mag::Zero => Mag::Inf,
        Mag::Inf => Mag::Zero,
        m => m,
    };
    if parity_in(e, class, cx) == Parity::Free {
        return match e {
            Ex::PosInf | Ex::NegInf => Mag::Inf,
            _ => Mag::Finite,
        };
    }
    match e {
        Ex::Add(ts) if matches!(class_of(ts, cx), Some((k, _, _)) if k == *class) => Mag::Zero,
        Ex::Add(ts) if class_multiple(ts, class, cx).is_some() => Mag::Zero,
        Ex::Add(ts) => {
            let mut all_zero = true;
            for t in ts {
                match zero_mag(t, class, cx) {
                    Mag::Zero => {}
                    Mag::Finite => all_zero = false,
                    _ => return Mag::Unknown,
                }
            }
            if all_zero {
                Mag::Zero
            } else {
                Mag::Finite
            }
        }
        Ex::Mul(fs) => product_mag(fs.iter().map(|f| zero_mag(f, class, cx))),
        Ex::Pow(b, q) => match &**q {
            Ex::Num(r) if r.is_negative() => inverse(zero_mag(b, class, cx)),
            Ex::Num(_) => zero_mag(b, class, cx),
            _ => match zero_mag(b, class, cx) {
                Mag::Finite => Mag::Finite,
                _ => Mag::Unknown,
            },
        },
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => match &args[1] {
            Ex::Num(k) if k.is_negative() => inverse(zero_mag(&args[0], class, cx)),
            Ex::Num(_) => zero_mag(&args[0], class, cx),
            _ => Mag::Unknown,
        },
        Ex::Fun(op, args) if args.len() == 1 => {
            let m = zero_mag(&args[0], class, cx);
            let view = cx.view;
            if view.tok_is(*op, "log") {
                match m {
                    Mag::Zero | Mag::Inf => Mag::Inf,
                    m => m,
                }
            } else if odd_fun(view, *op) || view.tok_is(*op, "abs") {
                match m {
                    Mag::Zero | Mag::Finite => m,
                    _ => Mag::Unknown,
                }
            } else if ["exp", "cos", "cosh"].iter().any(|n| view.tok_is(*op, n)) {
                match m {
                    Mag::Zero | Mag::Finite => Mag::Finite,
                    _ => Mag::Unknown,
                }
            } else {
                Mag::Unknown
            }
        }
        _ => Mag::Unknown,
    }
}
