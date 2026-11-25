//! A term's rational coefficient [`Coef`]: its sign and its literal partition.

use super::*;

/// A term's rational coefficient ([`term_split`]): a sign and the magnitudes of the term's
/// literal members, which `mul()` keeps as their canonical product partition
/// (`Rat::partition_product`): one member wherever the product is representable, none for a
/// unit, several where a fold is refused (`1e160*1e170*x` in f64 has the coefficient
/// `{1e160, 1e170}` and the key `x`). The whole partition is the coefficient. With only its
/// first member as the coefficient and the others left in the key, `1e160*1e170*x` and
/// `1e160*x` shared the content 1e160 and `-1e160*1e170*x` kept a key of its own, so a sum's
/// content and like terms depended on which terms a construction saw together, and a second
/// call could split them again.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Coef {
    neg: bool,
    /// Positive, sorted ascending (`cmp_exact`), no unit; `[0]` is the zero.
    mags: Vec<Rat>,
}

impl Coef {
    pub(crate) const ONE: Coef = Coef {
        neg: false,
        mags: Vec::new(),
    };

    fn zero() -> Coef {
        Coef {
            neg: false,
            mags: vec![Rat::ZERO],
        }
    }

    /// The coefficient `r`.
    pub(crate) fn of(r: &Rat) -> Coef {
        Coef::from_members(std::iter::once(r))
    }

    /// The product of literal members (of any sign), as they stand: a canonical product holds
    /// its partition already, and a re-partition here could run in another number domain.
    pub(super) fn from_members<'a>(members: impl IntoIterator<Item = &'a Rat>) -> Coef {
        let mut neg = false;
        let mut mags: Vec<Rat> = Vec::new();
        for r in members {
            if r.is_zero() {
                return Coef::zero();
            }
            neg ^= r.is_negative();
            if !r.is_one() && *r != Rat::NEG_ONE {
                mags.push(r.abs());
            }
        }
        mags.sort_unstable_by(|a, b| a.cmp_exact(b));
        Coef { neg, mags }
    }

    pub(crate) fn is_one(&self) -> bool {
        !self.neg && self.mags.is_empty()
    }

    pub(super) fn is_neg_one(&self) -> bool {
        self.neg && self.mags.is_empty()
    }

    pub(crate) fn is_zero(&self) -> bool {
        matches!(self.mags.as_slice(), [m] if m.is_zero())
    }

    pub(crate) fn is_negative(&self) -> bool {
        self.neg
    }

    /// The coefficient as one rational, where its partition has at most one member.
    pub(crate) fn rat(&self) -> Option<Rat> {
        let m = match self.mags.as_slice() {
            [] => Rat::ONE,
            [m] => m.clone(),
            _ => return None,
        };
        if self.neg {
            m.checked_neg()
        } else {
            Some(m)
        }
    }

    pub(crate) fn negated(&self) -> Coef {
        Coef {
            neg: !self.neg && !self.is_zero(),
            mags: self.mags.clone(),
        }
    }

    pub(super) fn abs(&self) -> Coef {
        Coef {
            neg: false,
            mags: self.mags.clone(),
        }
    }

    /// The product of two coefficients, partitioned in the current number domain.
    pub(super) fn times(&self, o: &Coef) -> Coef {
        if self.is_zero() || o.is_zero() {
            return Coef::zero();
        }
        let mut mags = self.mags.clone();
        mags.extend(o.mags.iter().cloned());
        Coef {
            neg: self.neg != o.neg,
            mags: Rat::partition_product(mags),
        }
    }

    /// The quotient by a nonzero `o`, exactly: the members left once `o`'s are taken out, or
    /// the rational quotient of two one-member coefficients; `None` otherwise.
    pub(super) fn over(&self, o: &Coef) -> Option<Coef> {
        if o.is_zero() {
            return None;
        }
        if let Some(rest) = take_members(&self.mags, &o.mags) {
            return Some(Coef {
                neg: (self.neg != o.neg) && !self.is_zero(),
                mags: rest,
            });
        }
        let (a, b) = (self.rat()?, o.rat()?);
        Some(Coef::of(&a.checked_mul(&b.checked_inv()?)?))
    }

    /// The sum of two coefficients of one key, where it is a coefficient of the domain: the
    /// rational sum of two one-member coefficients, or the members both share times the sum
    /// of what remains of each (`{2e300, 1e300} - {1e300, 1e300}` is `{1e300, 1e300}`, and
    /// `{1e160, 1e170} + {1e160, 1e170}` is `{2e160, 1e170}`); `None` where neither applies.
    /// A function of the pair, so a pool folds by [`partition_by`] as a literal pool does.
    pub(super) fn checked_add(&self, o: &Coef) -> Option<Coef> {
        if let (Some(a), Some(b)) = (self.rat(), o.rat()) {
            if let Some(s) = a.checked_add(&b) {
                return Some(Coef::of(&s));
            }
        }
        let (shared, p, q) = common_members(&self.mags, &o.mags);
        if shared.is_empty() {
            return None;
        }
        let rest = |neg: bool, ms: &[Rat]| -> Option<Rat> {
            let m = match ms {
                [] => Rat::ONE,
                [m] => m.clone(),
                _ => return None,
            };
            if neg {
                m.checked_neg()
            } else {
                Some(m)
            }
        };
        let s = rest(self.neg, &p)?.checked_add(&rest(o.neg, &q)?)?;
        if s.is_zero() {
            return Some(Coef::zero());
        }
        let mut mags = shared;
        mags.push(s.abs());
        Some(Coef {
            neg: s.is_negative(),
            mags: Rat::partition_product(mags),
        })
    }

    /// The canonical order: by exact value (the products formed without a cap), then by the
    /// members. On one-member coefficients it is the rational order.
    pub(crate) fn order(&self, o: &Coef) -> Ordering {
        if let (Some(a), Some(b)) = (self.rat(), o.rat()) {
            return a.cmp_exact(&b);
        }
        let sign = |c: &Coef| {
            if c.is_zero() {
                0
            } else if c.neg {
                -1
            } else {
                1
            }
        };
        let (sa, sb) = (sign(self), sign(o));
        if sa != sb {
            return sa.cmp(&sb);
        }
        let product = |ms: &[Rat]| {
            ms.iter()
                .fold((BigInt::from(1u8), BigInt::from(1u8)), |(p, q), m| {
                    let (a, b) = m.big_parts();
                    (p * a, q * b)
                })
        };
        let ((p1, q1), (p2, q2)) = (product(&self.mags), product(&o.mags));
        let mag = (p1 * q2).cmp(&(p2 * q1)).then_with(|| {
            self.mags
                .iter()
                .zip(&o.mags)
                .map(|(a, b)| a.cmp_exact(b))
                .find(|c| c.is_ne())
                .unwrap_or_else(|| self.mags.len().cmp(&o.mags.len()))
        });
        if sa < 0 {
            mag.reverse()
        } else {
            mag
        }
    }

    /// The literal members of a product with this coefficient, as `mul()` writes them: the sign
    /// on the one member, or on the smallest of several ([`host_partition_sign`]).
    pub(crate) fn items(&self) -> Vec<Ex> {
        let mut v: Vec<Ex> = self.mags.iter().cloned().map(Ex::Num).collect();
        if self.neg {
            match v.first_mut() {
                Some(Ex::Num(m)) => *m = m.checked_neg().expect("a Rat negates"),
                _ => v.push(Ex::Num(Rat::NEG_ONE)),
            }
        }
        v
    }
}
