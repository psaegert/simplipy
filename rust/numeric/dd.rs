//! Double-double arithmetic: a value carried as an unevaluated sum `hi + lo` of two f64 with
//! `|lo| <= ulp(hi)/2`, about 106 significant bits. The new operators' kernels compute their
//! intermediates in it, so a result's only error worth counting is its final rounding to f64.
//!
//! Error-free transformations only, no fused multiply-add: Rust never contracts `a * b + c`,
//! and the product splits by Dekker's method, exact for operands whose magnitudes stay below
//! about 2^996 (every caller here stays far inside that). The operations are the classical
//! ones (Dekker 1971; Knuth TAOCP vol. 2, 4.2.2; Shewchuk 1997), written out here.

/// 2^27 + 1: splits an f64 into two halves of at most 26 significant bits each.
const SPLITTER: f64 = 134_217_729.0;

/// `a + b = s + e` exactly.
#[inline]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a + b = s + e` exactly, given `|a| >= |b|` (or `a` zero).
#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline]
fn split(a: f64) -> (f64, f64) {
    let t = SPLITTER * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

/// `a * b = p + e` exactly (Dekker), barring overflow and underflow.
#[inline]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// A double-double `hi + lo`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Dd {
    pub(crate) hi: f64,
    pub(crate) lo: f64,
}

impl Dd {
    pub(crate) const fn new(hi: f64, lo: f64) -> Dd {
        Dd { hi, lo }
    }

    pub(crate) const fn from_f64(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    /// `a + b` of two doubles, exactly.
    #[inline]
    pub(crate) fn sum(a: f64, b: f64) -> Dd {
        let (hi, lo) = two_sum(a, b);
        Dd { hi, lo }
    }

    /// The f64 nearest to the value (one rounding).
    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    #[inline]
    pub(crate) fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    #[inline]
    pub(crate) fn add(self, b: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, b.hi);
        let (hi, lo) = quick_two_sum(s, e + (self.lo + b.lo));
        Dd { hi, lo }
    }

    #[inline]
    pub(crate) fn add_f64(self, b: f64) -> Dd {
        let (s, e) = two_sum(self.hi, b);
        let (hi, lo) = quick_two_sum(s, e + self.lo);
        Dd { hi, lo }
    }

    #[inline]
    pub(crate) fn mul(self, b: Dd) -> Dd {
        let (p, e) = two_prod(self.hi, b.hi);
        let (hi, lo) = quick_two_sum(p, e + (self.hi * b.lo + self.lo * b.hi));
        Dd { hi, lo }
    }

    #[inline]
    pub(crate) fn mul_f64(self, b: f64) -> Dd {
        let (p, e) = two_prod(self.hi, b);
        let (hi, lo) = quick_two_sum(p, e + self.lo * b);
        Dd { hi, lo }
    }

    /// `1 / x` of an f64: one division, corrected by its exact residual `1 - q x`.
    #[inline]
    pub(crate) fn recip_f64(x: f64) -> Dd {
        let q = 1.0 / x;
        let (p, e) = two_prod(q, x);
        let (hi, lo) = quick_two_sum(q, ((1.0 - p) - e) * q);
        Dd { hi, lo }
    }

    /// `1 / self`, as `recip_f64` with the low part's share of the residual.
    #[inline]
    pub(crate) fn recip(self) -> Dd {
        let q = 1.0 / self.hi;
        let (p, e) = two_prod(q, self.hi);
        let (hi, lo) = quick_two_sum(q, (((1.0 - p) - e) - q * self.lo) * q);
        Dd { hi, lo }
    }

    /// `self / b`: the quotient of the leading parts, corrected once by the remainder.
    #[inline]
    pub(crate) fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        let r = self.add(b.mul_f64(q1).neg());
        let q2 = r.hi / b.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Dd { hi, lo }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transformations are exact: the error terms recover what the rounded result lost.
    #[test]
    fn error_free_transformations_are_exact() {
        let (s, e) = two_sum(1.0, 1e-20);
        assert_eq!((s, e), (1.0, 1e-20));
        let (p, e) = two_prod(1.0 + f64::EPSILON, 1.0 + f64::EPSILON);
        // (1 + 2^-52)^2 = 1 + 2^-51 + 2^-104
        assert_eq!(p, 1.0 + 2.0 * f64::EPSILON);
        assert_eq!(e, f64::EPSILON * f64::EPSILON);
    }

    /// One third to double-double precision: 3 * (1/3) - 1 vanishes to ~2^-106.
    #[test]
    fn division_carries_twice_the_precision() {
        let third = Dd::from_f64(1.0).div(Dd::from_f64(3.0));
        let back = third.mul_f64(3.0).add_f64(-1.0);
        assert!(back.to_f64().abs() < 1e-31, "{back:?}");
        assert_eq!(third.to_f64(), 1.0 / 3.0);
    }
}
