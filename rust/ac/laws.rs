//! THE ZERO AND ONE LAWS, each a named step that carries its own premise, per mode.
//!
//! A law is an identity of the reals with the mode's extended values. Where it holds at every
//! binding it applies everywhere (a total law); where it does not, it applies only where its
//! premise holds of the operand its statement names, and each mode states its own premise:
//! `f64` reads float64's values (a step is admitted where float64's two sides agree wherever
//! the input has a value, or where the step holds in the reals and the two sides agree in kind),
//! `permissive` reads the table's values and cancels like terms without a definedness check.
//! No flag is shared between laws: a site that applies a law asks [`Law::holds`] by the law's
//! name.
//!
//! A fully numeric operand never reaches a law: its value is read first (a part without one is
//! NaN, which absorbs), so `0*tan(asin(1))` is NaN in `permissive` and `pow(tan(asin(1)), 0)`
//! too, and every premise below speaks of parts with a variable or a `<constant>`.
//!
//! | law | statement | premise, `f64` | premise, `permissive` |
//! |---|---|---|---|
//! | `add.zero` | t + 0 = t | t is not `-0` on a region | none |
//! | `mul.one` | 1*t = t | none | none |
//! | `mul.zero` | 0*t = 0 | t finite almost everywhere and never negative | none |
//! | `add.cancel` | t - t = 0 | t finite almost everywhere | none |
//! | `mul.cancel` | t/t = 1 | t finite and nonzero almost everywhere | none |
//! | `pow.zero` | t^0 = 1 | none | none |
//! | `pow.one` | t^1 = t | none | none |
//! | `pow.base_one` | 1^t = 1 | none | none |
//! | `rootn.one` | rootn(t, 1) = t | none | none |
//! | `add.infinity_absorbs` | inf + t = inf | t finite almost everywhere | none, after the like terms |

use super::expr::{Cx, Ex, Permissive};

/// A law of the catalogue: its name, its statement and, per mode, its premise and its stage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Law {
    /// `t + 0 = t`. In `f64` the zero stays where every other term may be float64's `-0` on a
    /// region: the sum is `+0` there, where `t` alone is `-0`, and an inverse reads the sign
    /// (`1/(ceil(x0) + 0)` is `+inf` on (-1, 0), `1/ceil(x0)` is `-inf`). The premise is read
    /// per term: a sum is `-0` only where every term is, so the zero goes where one term is never
    /// `-0` on a region (`x0 + 0` is `x0`).
    AddZero,
    /// `1*t = t`: a unit coefficient is no member. Total.
    MulOne,
    /// `0*t = 0`, per factor: a factor whose premise fails stays beside the zero. In `f64`, t
    /// finite almost everywhere: float64's `0*t` is NaN where t is infinite or undefined, which
    /// the step may fill only on a null set (`0*log(x0)` is NaN for every x0 < 0 and stays;
    /// `0*tanh(floor(x0)/floor(x0))` is NaN on all of [0, 1) and stays); and t never negative:
    /// float64's `0*t` is -0 wherever t < 0, a sign an inverse reads (`1/(0*x0)` is `inf/x0`), so
    /// a factor whose sign may vary stays beside the zero, which carries the product's sign (`-0`
    /// where the literals' sign is negative), and a head that keeps a zero keeps the product
    /// (`1/atan(0*x0)` is `inf/x0`). Where nothing reads the sign, the product drops the factors
    /// that only gave it its sign (`expr::unsigned_zero`): the answer itself, a sum's zero term
    /// (`add.zero`), a head blind to the sign (`exp(0*x0)` is 1). In `permissive`, none.
    MulZero,
    /// Like terms of opposite signs cancel: `t - t = 0`, `2t - t = t`. In `f64`, t finite almost
    /// everywhere: `t - t` is NaN where t is infinite or undefined (`log(x0) - log(x0)` is NaN
    /// for every x0 < 0 and stays). In `permissive`, none.
    AddCancel,
    /// Factors of one base with exponents of opposite signs merge: `t/t = 1`, `t^2/t = t`. In
    /// `f64`, t finite and nonzero almost everywhere (`floor(x0)/floor(x0)` is NaN on all of
    /// [0, 1) and stays), and the exponents' branch cut (`x0^(1/2)/x0^(1/2)` has no value for
    /// x0 < 0), which every merge of two exponents reads. In `permissive`, none.
    MulCancel,
    /// `t^0 = 1`. Total: numpy's `pow(t, 0)` is 1 at every t, NaN and the infinities included.
    PowZero,
    /// `t^1 = t`. Total.
    PowOne,
    /// `1^t = 1`. Total: numpy's `pow(1, t)` is 1 at every t, NaN included.
    PowBaseOne,
    /// `rootn(t, 1) = t`. Total.
    RootnOne,
    /// An infinity absorbs every finite term of its sum: `inf + t = inf`, and an
    /// infinity-carrying product absorbs such a term (`x0 + inf*x0 = inf*x0`); the literals and
    /// a `<constant>` are finite and always absorbed. In `f64`, t finite almost everywhere
    /// (`inf + log(x0)` is NaN for every x0 < 0 and stays), at construction. In `permissive`,
    /// none, as a settling step: a part with a variable is one term to the like-term
    /// cancellations, which run first, so `(x0 + inf)/(x0 + inf)` is 1,
    /// `2*(x0 + inf)/(x0 + inf)` is 2 and `(x0 + inf) - (x0 + inf)` is 0; what survives them is
    /// absorbed (`x0 + inf` is `inf`).
    InfinityAbsorbs,
}

impl Law {
    /// The whole catalogue, in its order.
    pub const ALL: [Law; 10] = [
        Law::AddZero,
        Law::MulOne,
        Law::MulZero,
        Law::AddCancel,
        Law::MulCancel,
        Law::PowZero,
        Law::PowOne,
        Law::PowBaseOne,
        Law::RootnOne,
        Law::InfinityAbsorbs,
    ];

    /// The law's stable name, `family.law`.
    pub const fn name(self) -> &'static str {
        match self {
            Law::AddZero => "add.zero",
            Law::MulOne => "mul.one",
            Law::MulZero => "mul.zero",
            Law::AddCancel => "add.cancel",
            Law::MulCancel => "mul.cancel",
            Law::PowZero => "pow.zero",
            Law::PowOne => "pow.one",
            Law::PowBaseOne => "pow.base_one",
            Law::RootnOne => "rootn.one",
            Law::InfinityAbsorbs => "add.infinity_absorbs",
        }
    }

    /// The identity, in the reals with the mode's extended values.
    pub const fn statement(self) -> &'static str {
        match self {
            Law::AddZero => "t + 0 = t",
            Law::MulOne => "1*t = t",
            Law::MulZero => "0*t = 0",
            Law::AddCancel => "t - t = 0",
            Law::MulCancel => "t/t = 1",
            Law::PowZero => "t^0 = 1",
            Law::PowOne => "t^1 = t",
            Law::PowBaseOne => "1^t = 1",
            Law::RootnOne => "rootn(t, 1) = t",
            Law::InfinityAbsorbs => "inf + t = inf",
        }
    }

    /// Is the law total: an identity at every binding in both modes, so its premise is none?
    pub const fn total(self) -> bool {
        matches!(
            self,
            Law::MulOne | Law::PowZero | Law::PowOne | Law::PowBaseOne | Law::RootnOne
        )
    }

    /// Does the law's premise hold of `t`, the operand its statement names, in `cx`'s mode?
    /// A total law holds of every operand. `f64`'s premises are discharged by certificates
    /// (`Cx::fin_certified`, `Cx::finnz_certified`, the zero-sign test), conservatively: a
    /// premise not shown refuses the step, which is always sound.
    #[inline]
    pub(crate) fn holds(self, t: &Ex, cx: &Cx) -> bool {
        match self {
            law if law.total() => true,
            Law::AddZero => {
                cx.permits(Permissive::LikeTerms) || super::expr::signed_zero_null(t, true, cx)
            }
            Law::MulZero => {
                cx.permits(Permissive::LikeTerms)
                    || (cx.fin_certified(t)
                        && (!super::expr::f64_zero_signs(cx) || cx.certainly_nonneg(t)))
            }
            Law::AddCancel | Law::InfinityAbsorbs => {
                cx.permits(Permissive::LikeTerms) || cx.fin_certified(t)
            }
            Law::MulCancel => cx.permits(Permissive::LikeTerms) || cx.finnz_certified(t),
            _ => unreachable!("every partial law states its premise"),
        }
    }

    /// Is the law a SETTLING step in `cx`'s mode, applied by the pass to a built node, rather
    /// than a step of the constructors? Only the absorption of infinities, in `permissive`: it
    /// waits for the like-term cancellations, which the constructors take.
    #[inline]
    pub(crate) fn settles(self, cx: &Cx) -> bool {
        matches!(self, Law::InfinityAbsorbs) && cx.permits(Permissive::LikeTerms)
    }

    /// Does the constructor apply the law to `t` here: the premise holds and the law is no
    /// settling step in this mode?
    #[inline]
    pub(crate) fn constructs(self, t: &Ex, cx: &Cx) -> bool {
        !self.settles(cx) && self.holds(t, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::Law;

    #[test]
    fn every_law_has_one_name_and_a_statement() {
        let mut names: Vec<&str> = Law::ALL.iter().map(|l| l.name()).collect();
        assert!(Law::ALL.iter().all(|l| !l.statement().is_empty()));
        assert!(names.iter().all(|n| n.contains('.')));
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Law::ALL.len());
    }
}
