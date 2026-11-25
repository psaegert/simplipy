//! The expression type [`Ex`], its leaves, and the name a fitted constant is written with.

use super::*;

/// THE NAME A FITTED CONSTANT IS WRITTEN WITH: `None` for `<constant>`, the token of a named
/// constant `<constant_NAME>` the expression uses once (`TokenView::is_free_name`) otherwise. A
/// name used once is exactly a `<constant>` (one unknown finite real nothing else refers to), so
/// the engine treats the two alike; the name is how the answer writes the constant. Where
/// fitted constants become one (`c1 + c2 = c`, `5*c = c`, `log(c) = c`), the one keeps a name if
/// any of them had one, the least in string order ([`ConstName::merge`]).
///
/// Part of the expression's equality, so a memo never hands one name's state to another; an
/// anonymous constant hashes as the unit variant did (`Hash` writes nothing for `None`), so
/// every expression without a named constant hashes, compares and orders as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConstName(pub Option<Tok>);

impl std::hash::Hash for ConstName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if let Some(t) = self.0 {
            t.hash(state);
        }
    }
}

impl ConstName {
    /// `<constant>`.
    pub const ANON: ConstName = ConstName(None);

    /// The name of the one constant that `self` and `other` become: a name where either has
    /// one, the least in string order where both do.
    pub fn merge(self, other: ConstName, view: &TokenView) -> ConstName {
        match (self.0, other.0) {
            (None, _) => other,
            (_, None) => self,
            (Some(a), Some(b)) => {
                if view.str_cmp(a, b) == Ordering::Greater {
                    other
                } else {
                    self
                }
            }
        }
    }
}

/// An AC-core expression. Leaves carry interned [`Tok`] ids (variables `x0..`, wildcards
/// `_0`/`?0`/`!0`/`$0`); everything numeric is EXACT ([`Rat`], or the symbolic
/// transcendentals/extended values as their own variants).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ex {
    /// An exact rational literal.
    Num(Rat),
    /// pi -- a transcendental FACTOR, never absorbed into the rational coefficient.
    Pi,
    /// e, likewise.
    E,
    PosInf,
    NegInf,
    NaN,
    /// `<constant>` -- a fitted constant, existentially quantified over R. EVERY occurrence is an
    /// INDEPENDENT constant: two `Const`s are structurally equal but semantically distinct, which
    /// is why `Const`-containing keys never merge in collection. A named constant the expression
    /// uses once is one too, written with its name ([`ConstName`]).
    Const(ConstName),
    /// A variable or wildcard leaf (sigil-classified by the token, exactly as the old matcher).
    Leaf(Tok),
    /// n-ary flat sorted addition bag.
    Add(Vec<Ex>),
    /// n-ary flat sorted multiplication bag.
    Mul(Vec<Ex>),
    Pow(Box<Ex>, Box<Ex>),
    /// A named function application (`sin`, `cos`, `log`, `abs`, ..., and the real odd roots
    /// `pow1_3`/`pow1_5` which deliberately do NOT desugar to `Pow`).
    Fun(Tok, Vec<Ex>),
}

impl Ex {
    /// `<constant>`, the anonymous fitted constant.
    pub const ANON: Ex = Ex::Const(ConstName::ANON);

    pub fn int(n: i128) -> Ex {
        Ex::Num(Rat::int(n))
    }

    /// The names of the fitted constants of `self` (`Ex::Const(_)` carrying a name), merged into one
    /// ([`ConstName::merge`]): the name a constant standing for all of them is written with.
    pub fn merged_const_name(&self, view: &TokenView) -> ConstName {
        match self {
            Ex::Const(n) => *n,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Leaf(_) => {
                ConstName::ANON
            }
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().fold(ConstName::ANON, |n, x| {
                n.merge(x.merged_const_name(view), view)
            }),
            Ex::Pow(b, e) => b
                .merged_const_name(view)
                .merge(e.merged_const_name(view), view),
        }
    }

    /// Does this expression contain a `<constant>` anywhere? (The independence guard: such
    /// subtrees never participate in key-equality collection, and the matcher refuses to rebind
    /// a placeholder to one -- both carried over from the shipped engine.)
    pub fn contains_const(&self) -> bool {
        match self {
            Ex::Const(_) => true,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Leaf(_) => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(Ex::contains_const),
            Ex::Pow(b, e) => b.contains_const() || e.contains_const(),
        }
    }

    /// Does this expression SPELL an infinity anywhere (an `Ex::PosInf`/`Ex::NegInf` node)?
    /// Such a subtree can be infinite on a set of full measure, so the lossy blanket's
    /// premise ("finite almost everywhere") is not an approximation there but false (see
    /// `ac::search::distribute_product`).
    pub fn contains_infinity(&self) -> bool {
        match self {
            Ex::PosInf | Ex::NegInf => true,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::NaN | Ex::Const(_) | Ex::Leaf(_) => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(Ex::contains_infinity),
            Ex::Pow(b, e) => b.contains_infinity() || e.contains_infinity(),
        }
    }

    /// Does this expression contain a VARIABLE leaf or a `<constant>` -- i.e., does it have a
    /// measure space for "almost everywhere" to quantify over? The a.e. certificates are
    /// statements about null sets in (variable, constant)-space; a fully ground expression
    /// (`inv 0`) has a SINGLE value, so a.e. tolerance degenerates to exactness. The interval
    /// certificates handle this correctly themselves (`finite_ae("inv 0") == false`,
    /// verified); the guard makes the licences independent of that edge behavior and hands
    /// ground arithmetic to the numeric fold, which owns it exactly. (The shipped engine's
    /// `- inv 0 inv 0 -> 0` gap is its certificate-FREE `_`-sort rule tier, not the
    /// certificates.)
    pub fn has_measure_space(&self, view: &TokenView) -> bool {
        match self {
            // A sigil-free leaf is a variable UNLESS it merely SPELLS a number the exact
            // parser could not represent (beyond-i128 mantissas like `1e39`, minted by the
            // engine's own fold re-parse). Such a leaf denotes ONE real value: it is
            // ground, and treating it as a variable would hand the a.e. licences a
            // measure space that does not exist.
            Ex::Leaf(t) => {
                view.sigil(*t) == 0 && !crate::utils::is_numeric_string(&view.resolve_owned(*t))
            }
            Ex::Const(_) => true,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|e| e.has_measure_space(view)),
            Ex::Pow(b, e) => b.has_measure_space(view) || e.has_measure_space(view),
        }
    }

    /// Does this expression contain any wildcard leaf? (Pattern detection.)
    pub fn contains_wildcard(&self, view: &TokenView) -> bool {
        match self {
            Ex::Leaf(t) => view.sigil(*t) != 0,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const(_) => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|e| e.contains_wildcard(view)),
            Ex::Pow(b, e) => b.contains_wildcard(view) || e.contains_wildcard(view),
        }
    }
}

/// The spelling of float64's negative zero in `f64`: a numeral leaf, since a number has one
/// zero. Python reads `-0.0` as -0 and `-0` as the integer zero, +0.
pub(crate) const NEG_ZERO: &str = "-0.0";

/// Is `e` the written negative zero (`NEG_ZERO`)?
pub(crate) fn is_neg_zero(e: &Ex, view: &TokenView) -> bool {
    matches!(e, Ex::Leaf(t) if view.sigil(*t) == 0 && view.with_str(*t, |s| s == NEG_ZERO))
}

/// Is `s` a NUMERIC-STRING leaf (a literal the engine holds as written)? A first-byte filter
/// skips the full numeric parse for ordinary vocabulary leaves (`x0`, function names), which
/// never start with a digit, dot, or sign.
#[inline]
pub(crate) fn numeric_leaf(s: &str) -> bool {
    s.as_bytes()
        .first()
        .is_some_and(|b| b.is_ascii_digit() || *b == b'-' || *b == b'.' || *b == b'+')
        && crate::utils::is_numeric_string(s)
}
