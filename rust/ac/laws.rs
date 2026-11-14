//! THE LAWS OF THE CATALOGUE, each a named step that carries its own premise, per mode: the
//! numbers and their ground values, the zero and one laws, the absorptions of a fitted constant,
//! the parse's and the bags' structure, and the functions' identities.
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
//! | `absorb.sum` | c1 + r*c2 + q = c | none | none |
//! | `absorb.product` | r*c = c, r a nonzero rational | none | none |
//! | `absorb.sum_sign` | -k*(s + c) = k*(-s + c) | none | none |
//! | `absorb.refit_sign` | c*(a - b) = c*(b - a) | none | none |
//! | `absorb.ground_shift` | c + g = c, c*g = c, g ground and finite (nonzero in a product) | none | none |
//! | `absorb.collapse` | L(c) = c, L variable-free | none | none |
//! | `absorb.nan_family` | L(c) = NaN, L variable-free and NaN at every c | none | none |
//! | `num.literal_arith` | r1 + r2 and r1*r2 are exact | none | none |
//! | `num.abs_literal` | abs(r) = r or -r | none | none |
//! | `num.exp_one` | exp(1) = e | none | none |
//! | `num.zero_pole` | 0^q = inf, q < 0 | none | no step (no value) |
//! | `num.inf_arith` | c*inf = sign(c)*inf, 0*inf = inf - inf = NaN | none | none |
//! | `num.inf_power` | inf^q = inf (q > 0), inf^q = 0 (q < 0) | none | none |
//! | `num.literal_base_inf_exponent` | b^inf = 0 (abs(b) < 1), b^-inf = 0 (abs(b) > 1) | none | none |
//! | `num.table_fold` | f(r1, r2) = the table's value | no step (float64's values) | none |
//! | `num.ground_class_fold` | g = NaN or g = +-inf, g ground | none | none |
//! | `num.ground_float_fold` | g = its cheapest admitted value, g ground | none | none |
//! | `num.rootn_index` | rootn(t, n) = NaN, n no nonzero integer | none | none |
//! | `num.pow_parity` | (-inf)^k = (-1)^k*inf | none | none |
//! | `num.besselj_literals` | J_nu(0), J_(+-inf)(x) at literals | none | none |
//! | `num.lattice_numbers` | max(r1, r2) = the greater | none | none |
//! | `num.literal_to_float` | r = its shorter float | no step | none |
//! | `num.signed_zero` | t + (-0) = t, (-0)^-k = -inf (k odd) | none | no step (one zero) |
//! | `num.ground_first` | a fully numeric part is its value | none | none |
//! | `parse.sub` | a - b = a + (-1)*b | none | none |
//! | `parse.div` | a/b = a*b^-1 | none | none |
//! | `parse.neg_inv` | neg a = (-1)*a, inv a = a^-1 | none | none |
//! | `parse.signed_literal` | -r = (-1)*r | none | none |
//! | `add.flatten` | (a + b) + c = a + (b + c) | none | none |
//! | `mul.flatten` | (a*b)*c = a*(b*c) | none | none |
//! | `lattice.bag` | max(max(a, b), c) = max(a, b, c), max(a, a) = a | none | none |
//! | `add.collect` | r1*t + r2*t = (r1 + r2)*t | none | none |
//! | `add.unfactor` | r*(a + b) + c = r*a + r*b + c | r <= 0: the sum holds no sign | none |
//! | `add.content` | u*a + u*b = u*(a + b), -a - b = -(a + b) | a sign moves: no held sign | none |
//! | `mul.exp_merge` | exp(a)*exp(b) = exp(a + b) | none | none |
//! | `mul.rootn_join` | rootn(a, n)*rootn(b, n) = rootn(a*b, n) | none | none |
//! | `mul.infinity_sign` | -k*(s + inf) = k*(-s - inf) | none | none |
//! | `pow.exp_compose` | e^t = exp(t), exp(a)^b = exp(a*b) | none | none |
//! | `pow.refused_power` | (a*b)^-k = ((a*b)^k)^-1 | none | none |
//! | `fun.rootn_index` | rootn(t, -n) = 1/rootn(t, n) | none | none |
//! | `fun.atanh_tanh` | atanh(tanh(t)) = t | abs(t) at most 19: tanh(t) is no +-1 | none |
//! | `fun.log_exp` | log(exp(t)) = t | t is not `-0` on a region, and nothing around reads its flat point | none |
//! | `fun.asinh_sinh` | asinh(sinh(t)) = t | none | none |
//! | `fun.sinh_asinh` | sinh(asinh(t)) = t | none | none |
//! | `fun.acosh_cosh` | acosh(cosh(t)) = abs(t) | nothing around reads its flat point | none |
//! | `fun.exp_log` | exp(log(t)) = t | t >= 0, not `-0` on a region | none |
//! | `fun.unit_inverses` | cos(acos(t)) = t, sin(asin(t)) = t, tanh(atanh(t)) = t | t in [-1, 1] | none |
//! | `perm.inverse_pairs` | f(g(t)) = t, g the inverse of f | no step | none |
//! | `fun.abs_nonnegative` | abs(t) = t | t >= 0, not `-0` on a region | t >= 0 |
//! | `fun.half_period` | sin(t + pi) = -sin(t), cos(t + pi) = -cos(t), tan(t + pi) = tan(t) | none | none |
//! | `fun.besselj_reflection` | J_(-n)(t) = (-1)^n J_n(t) | none | none |
//! | `lattice.absorb` | max(a, min(a, b)) = a, max(inf, a) = inf, max(-inf, a) = a | the dropped members no NaN almost everywhere | none |
//!
//! An absorption is the specification's 3.2, a statement about families: for every value of
//! the constants on the left there is one on the right, `d = c1 + r*c2 + q`, `d = r*c`,
//! `d = -c`, `d = L(c)`, which is why none needs a premise. A fitted constant is finite: a real
//! number, in `f64` a finite double (-0.0 included), so a value of the constants at which the
//! constant part has none is no member of the family (`log(c)` at c = 0, `1/c` at 0, `atanh(c)`
//! at 1), and an overflow at a finite constant is a range event: `tanh(x0 + log(c))` is
//! `tanh(x0 + c)`. A collapse needs the evidence every step needs (the node is finite on a set of
//! positive measure of its constants in the mode's values: `(-1)^c` has none in `f64`), which
//! `permissive` does without.

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
    /// A sum's fitted constants and literals are one constant: `c1 + 2*c2 + 5 = c`. Premise
    /// none in both modes.
    AbsorbSum,
    /// A fitted constant absorbs a nonzero rational coefficient: `5*c = c`, a partition's
    /// members included. An exact 0 stays outside: `0*c` is 0, never a fitted constant. Premise
    /// none in both modes.
    AbsorbProduct,
    /// A negative coefficient beside a sum holding a fitted constant joins the sum:
    /// `-2*(x0 + c) = 2*(c - x0)`. Premise none in both modes.
    AbsorbSumSign,
    /// Beside a fitted constant every orientation of a class is free: `c*(x0 - x1) =
    /// c*(x1 - x0)`. Premise none in both modes.
    AbsorbRefitSign,
    /// A sum or product holding a fitted constant absorbs a ground member whose value is finite,
    /// and certainly nonzero in a product: `c + cosh(1) = c`, `c*sin(1) = c`. Premise none in
    /// both modes.
    AbsorbGroundShift,
    /// A variable-free node holding a fitted constant is one: `log(c) = c`, wherever it stands
    /// (`tanh(x0 + log(c))` is `tanh(x0 + c)`): a fitted constant is finite, so the constant
    /// part's value is the witness wherever it has one, and the values of the constants at which
    /// it has none (`log(0)`, `1/0`, `atanh(1)`) are no members. Premise none in both modes; the
    /// collapse licence (`interval::collapse_licence`) reads the node's evidence in `f64`
    /// (`(-1)^c`, finite only at the integers, has none).
    AbsorbCollapse,
    /// A variable-free node holding a fitted constant whose value is NaN at every constant is
    /// NaN: `acosh(acos(cosh(c)))`. Premise none in both modes.
    AbsorbNanFamily,
    /// A decimal is the rational it spells, and `+`, `*`, integer powers and exact roots between
    /// numbers are exact rational arithmetic (`0.1 + 0.2` is 3/10). Total; a number is written as
    /// the cheapest value its mode admits (`num.ground_float_fold`).
    NumLiteralArith,
    /// `|r|` is `r` or `-r` for a rational literal. Total.
    NumAbsLiteral,
    /// `exp(1) = e`. Total: float64's `exp(1.0)` is numpy's `e`.
    NumExpOne,
    /// `0^q = +inf` for a literal `q < 0`, IEEE's pole. A step of `f64` alone: `permissive` reads
    /// the table, where `0^-1` has no value.
    NumZeroPole,
    /// IEEE's arithmetic of the infinities and NaN: `c*inf` is the infinity of `c`'s sign, `0*inf`
    /// and `inf - inf` are NaN, and NaN absorbs. Total.
    NumInfArith,
    /// `inf^q` is `inf` for `q > 0` and 0 for `q < 0`, and `(-inf)^k` is 0 for a negative integer
    /// `k` (`-0` at an odd `k` in `f64`). Total.
    NumInfPower,
    /// `b^inf = 0` for `|b| < 1` and `b^-inf = 0` for `|b| > 1`, `b` a nonzero literal. Total.
    NumLiteralBaseInfExponent,
    /// An operation on numbers (the rationals, the infinities, NaN) is the table's exact value:
    /// `pow(0, -1)`, `pow(1, inf)` and `mod(0, inf)` have none (NaN). A step of `permissive` alone:
    /// `f64` keeps float64's values.
    NumTableFold,
    /// A fully numeric part whose value is NaN or an infinity is that value: in `f64` its
    /// float64 value's kind, the infinity's sign float64's; in `permissive` the table's. Total.
    NumGroundClassFold,
    /// A finite fully numeric part is written as the cheapest value its mode admits: in `f64`
    /// float64's double, or the real value where float64's kind agrees; in `permissive` the
    /// table's value at float resolution (the ground evaluator, `ground`). Total.
    NumGroundFloatFold,
    /// `rootn(t, n)` is NaN where `n` is no nonzero integer: 0, a non-integer literal, `pi`, `e`,
    /// an infinity, NaN, or a ground index whose value holds none. Total.
    NumRootnIndex,
    /// `(-inf)^k` is the infinity of the parity of an integer literal `k` beyond 2^53, read
    /// exactly. Total.
    NumPowParity,
    /// `besselj` at literals: `J_nu(0)` is 1 at `nu = 0`, 0 at `nu > 0` and at a negative integer,
    /// and the pole's infinity at a negative non-integer; `J_(+-inf)(x)` is 0 for a literal
    /// `x >= 0`. Total: the kernel's values in `f64`, the table's in `permissive`.
    NumBesseljLiterals,
    /// The numbers of a `max` or `min` bag fold to their extreme (`-0` below `+0` in `f64`).
    /// Total.
    NumLatticeNumbers,
    /// A literal beside a non-literal lands on the shorter of the two floats around it where that
    /// prints cheaper. A step of `permissive` alone: one of its own simplifications, within float
    /// resolution.
    NumLiteralToFloat,
    /// The written `-0.0` is float64's negative zero: `t + (-0) = t`, `(-0)^k` is `-inf` at an odd
    /// `k < 0`, and a product of numbers that is zero takes float64's sign (`-1*0 = -0.0`). A step
    /// of `f64` alone: `permissive` reads one zero.
    NumSignedZero,
    /// A fully numeric part takes its value before any law, rule or simplification reaches it,
    /// and a part without a value is NaN, which absorbs. Total: it is how every other law reads
    /// a number.
    NumGroundFirst,
    /// `a - b = a + (-1)*b`, read at the parse. Total: float64 subtracts by adding the negation,
    /// bit for bit.
    ParseSub,
    /// `a/b = a*b^-1`, read at the parse. Total: the quotient and the product by the reciprocal
    /// are one value at every binding, the infinities and a zero's sign included, and a
    /// reciprocal that overflows is a range event.
    ParseDiv,
    /// `neg a = (-1)*a` and `inv a = a^-1`, read at the parse. Total.
    ParseNegInv,
    /// A numeral with a sign beyond the exact reader (`-1e309`) is -1 times its unsigned leaf.
    /// Total.
    ParseSignedLiteral,
    /// A sum is one flat bag in the canonical order: `(a + b) + c = a + (b + c)`. Total: the
    /// reals' value is one in any order, float64's rounding is a range of its own, and a sum is
    /// `-0` only where every term is, in any order.
    AddFlatten,
    /// A product is one flat bag in the canonical order: `(a*b)*c = a*(b*c)`. Total.
    MulFlatten,
    /// `max` and `min` are flat bags: `max(max(a, b), c) = max(a, b, c)` and `max(a, a) = a`, and
    /// NaN absorbs. Total: IEEE's `maximum` and `minimum`, the zeros' tie included.
    LatticeBag,
    /// Like terms of one sign merge: `2t + 3t = 5t`. Total.
    AddCollect,
    /// A factored sum entering a sum spreads: `r*(a + b) + c = r*a + r*b + c`, `r` a literal (a
    /// refused partition's members are one coefficient). In `f64` a coefficient that is no
    /// positive number reads the sum's zero: float64's `a + b` is `+0` wherever it cancels, so a
    /// negative `r` makes `r*(a + b)` `-0` there while `r*a + r*b` cancels to `+0`. A negative `r`
    /// spreads where the sum holds no sign (`expr::sum_holds_its_sign`: its zero set is null, or
    /// nothing reads its zero, `expr::release_unread`), or where another addend is never `-0` on
    /// a region, which keeps the whole sum off `-0` there: under an inverse,
    /// `ceil(x2) - 2*(ceil(x0) + floor(x1))` stays as it is and `x2 - 2*(ceil(x0) + floor(x1))`
    /// spreads; as the answer both spread. A zero `r` gives `0*(a + b)` the sign of `a + b`, which
    /// `0*a + 0*b` does not spell: the zero product stays whole, and spreads where its sum
    /// settles, which reads none of its zeros (`x1 + 0*(2*x0 + sqrt(x0))` is `x1 + 0*sqrt(x0)`).
    /// [`Law::holds`] reads the sum. In `permissive`, none: one zero.
    AddUnfactor,
    /// A sum carries no unanimous coefficient magnitude and takes its cheaper orientation:
    /// `u*a + u*b = u*(a + b)`, `-a - b = -(a + b)`. In `f64` a sign leaves a sum, or enters it,
    /// only where the sum holds no sign (`expr::sum_holds_its_sign`: its zero set is null, or
    /// nothing reads its zero, `expr::release_unread`): float64's `-a - b` is `+0` wherever the
    /// sum cancels and `-(a + b)` is `-0` there, a sign the expression around it may read
    /// (`1/(-ceil(x0) - floor(x1))` is `+inf` wherever `ceil(x0) = -floor(x1)`, and stays as it
    /// is, while the answer `-ceil(x0) - floor(x1)` is `-(ceil(x0) + floor(x1))`). A positive
    /// content moves no sign. [`Law::holds`] reads the sum. In `permissive`, none.
    AddContent,
    /// `exp(a)*exp(b) = exp(a + b)`, and `e = exp(1)` beside one. Total: an overflow is a range
    /// event.
    MulExpMerge,
    /// Odd roots of one index join: `rootn(a, n)*rootn(b, n) = rootn(a*b, n)`, `n` odd. Total.
    MulRootnJoin,
    /// A negative coefficient beside a sum holding an infinity joins the sum:
    /// `-k*(s + inf) = k*(-s - inf)`. Total.
    MulInfinitySign,
    /// A power of `e` is an `exp`: `e^t = exp(t)`, `exp(a)^b = exp(a*b)` and
    /// `rootn(exp(a), n) = exp(a/n)` for literals. Total: `exp` is positive.
    PowExpCompose,
    /// A negative odd power of a product the distribution refuses is the inverse of the positive
    /// power: `(a*b)^-k = ((a*b)^k)^-1`. Total.
    PowRefusedPower,
    /// `rootn(t, -n) = 1/rootn(t, n)` and `rootn(rootn(t, m), n) = rootn(t, m*n)`. Total.
    FunRootnIndex,
    /// `atanh(tanh(t)) = t`. In `f64` where `|t|` is at most 19: float64's `tanh` is `+-1` from
    /// `|t|` = 19.0615 on, where `atanh` gives an infinity against `t`; within the band the
    /// rounding of `tanh` keeps the kind (`Cx::certainly_in_band`). In `permissive`, none.
    FunAtanhTanh,
    /// `log(exp(t)) = t`. In `f64` where `t` is not `-0` on a region: `log(exp(-0))` is `+0`
    /// (`log(exp(ceil(x0)))` is `+0` on (-1, 0), where `ceil(x0)` is `-0`); `exp`'s overflow and
    /// underflow are range events. And where nothing around reads its flat point
    /// (`Cx::flat_unread`, `flat_read_below`): float64's `log(exp(t))` is `+0` wherever
    /// 0 < |t| < 1.1e-16, so `1/log(exp(x0))` is `+inf` there, where `1/x0` is finite. In
    /// `permissive`, none.
    FunLogExp,
    /// `asinh(sinh(t)) = t`. Total: `sinh`'s overflow is a range event.
    FunAsinhSinh,
    /// `sinh(asinh(t)) = t`. Total: `asinh`'s rounding keeps the kind, and `sinh` keeps a zero's
    /// sign.
    FunSinhAsinh,
    /// `acosh(cosh(t)) = |t|`: `cosh` is even, `cosh`'s overflow is a range event, and both sides
    /// are `+0` at a zero of either sign. In `f64` where nothing around reads its flat point
    /// (`Cx::flat_unread`, `flat_read_below`): float64's `acosh(cosh(t))` is 0 wherever
    /// |t| < 1.5e-8 and drifts below `|t|` elsewhere (`acosh(cosh(3))` is 2.9999999999999996).
    /// In `permissive`, none.
    FunAcoshCosh,
    /// `exp(log(t)) = t` where `t >= 0` (on `t < 0` the left side has no value and the right side
    /// has, `Cx::certainly_nonneg`), and in `f64` where `t` is not `-0` on a region:
    /// `exp(log(-0))` is `+0`.
    FunExpLog,
    /// `cos(acos(t)) = t`, `sin(asin(t)) = t` and `tanh(atanh(t)) = t` where `t` is in [-1, 1]
    /// (`sin`, `cos` or `tanh` of something; off it the left side has no value). Both sides keep a
    /// zero's sign but `cos(acos(-0))`, which is 6e-17 against `-0`, both finite.
    FunUnitInverses,
    /// An operation cancels its own inverse in both orders, ignoring domain and branch:
    /// `asin(sin(t)) = t`, `exp(log(t)) = t`, `acosh(cosh(t)) = t`. A step of `permissive` alone,
    /// one of its own simplifications (`Permissive::InversePairs`).
    PermInversePairs,
    /// `|t| = t` where `t >= 0` wherever it has a value (`Cx::certainly_nonneg`), and in `f64`
    /// where `t` is not `-0` on a region: `|-0|` is `+0` (`|sqrt(ceil(x0))|` is `+0` on (-1, 0),
    /// where numpy's `sqrt` keeps `ceil(x0)`'s `-0`).
    FunAbsNonnegative,
    /// A unit half-period shift drops out: `sin(t +- pi) = -sin(t)`, `cos(t +- pi) = -cos(t)`,
    /// `tan(t +- pi) = tan(t)`, `pi` with the coefficient 1 or -1. Total.
    FunHalfPeriod,
    /// `J_(-n)(t) = (-1)^n J_n(t)` for an integer order `n`. Total: the kernel reads it bit for
    /// bit.
    FunBesseljReflection,
    /// The dual bag absorbs, `max(a, min(a, b)) = a`; the absorber takes, `max(inf, a) = inf`; the
    /// identity drops, `max(-inf, a) = a` (dually for `min`). Where the members it drops are no
    /// NaN almost everywhere (a number, an infinity, or finite almost everywhere): a NaN member
    /// makes the bag NaN. [`Law::holds`] reads one dropped member. In `permissive`, none.
    LatticeAbsorb,
}

impl Law {
    /// The whole catalogue, in its order.
    pub const ALL: [Law; 62] = [
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
        Law::AbsorbSum,
        Law::AbsorbProduct,
        Law::AbsorbSumSign,
        Law::AbsorbRefitSign,
        Law::AbsorbGroundShift,
        Law::AbsorbCollapse,
        Law::AbsorbNanFamily,
        Law::NumLiteralArith,
        Law::NumAbsLiteral,
        Law::NumExpOne,
        Law::NumZeroPole,
        Law::NumInfArith,
        Law::NumInfPower,
        Law::NumLiteralBaseInfExponent,
        Law::NumTableFold,
        Law::NumGroundClassFold,
        Law::NumGroundFloatFold,
        Law::NumRootnIndex,
        Law::NumPowParity,
        Law::NumBesseljLiterals,
        Law::NumLatticeNumbers,
        Law::NumLiteralToFloat,
        Law::NumSignedZero,
        Law::NumGroundFirst,
        Law::ParseSub,
        Law::ParseDiv,
        Law::ParseNegInv,
        Law::ParseSignedLiteral,
        Law::AddFlatten,
        Law::MulFlatten,
        Law::LatticeBag,
        Law::AddCollect,
        Law::AddUnfactor,
        Law::AddContent,
        Law::MulExpMerge,
        Law::MulRootnJoin,
        Law::MulInfinitySign,
        Law::PowExpCompose,
        Law::PowRefusedPower,
        Law::FunRootnIndex,
        Law::FunAtanhTanh,
        Law::FunLogExp,
        Law::FunAsinhSinh,
        Law::FunSinhAsinh,
        Law::FunAcoshCosh,
        Law::FunExpLog,
        Law::FunUnitInverses,
        Law::PermInversePairs,
        Law::FunAbsNonnegative,
        Law::FunHalfPeriod,
        Law::FunBesseljReflection,
        Law::LatticeAbsorb,
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
            Law::AbsorbSum => "absorb.sum",
            Law::AbsorbProduct => "absorb.product",
            Law::AbsorbSumSign => "absorb.sum_sign",
            Law::AbsorbRefitSign => "absorb.refit_sign",
            Law::AbsorbGroundShift => "absorb.ground_shift",
            Law::AbsorbCollapse => "absorb.collapse",
            Law::AbsorbNanFamily => "absorb.nan_family",
            Law::NumLiteralArith => "num.literal_arith",
            Law::NumAbsLiteral => "num.abs_literal",
            Law::NumExpOne => "num.exp_one",
            Law::NumZeroPole => "num.zero_pole",
            Law::NumInfArith => "num.inf_arith",
            Law::NumInfPower => "num.inf_power",
            Law::NumLiteralBaseInfExponent => "num.literal_base_inf_exponent",
            Law::NumTableFold => "num.table_fold",
            Law::NumGroundClassFold => "num.ground_class_fold",
            Law::NumGroundFloatFold => "num.ground_float_fold",
            Law::NumRootnIndex => "num.rootn_index",
            Law::NumPowParity => "num.pow_parity",
            Law::NumBesseljLiterals => "num.besselj_literals",
            Law::NumLatticeNumbers => "num.lattice_numbers",
            Law::NumLiteralToFloat => "num.literal_to_float",
            Law::NumSignedZero => "num.signed_zero",
            Law::NumGroundFirst => "num.ground_first",
            Law::ParseSub => "parse.sub",
            Law::ParseDiv => "parse.div",
            Law::ParseNegInv => "parse.neg_inv",
            Law::ParseSignedLiteral => "parse.signed_literal",
            Law::AddFlatten => "add.flatten",
            Law::MulFlatten => "mul.flatten",
            Law::LatticeBag => "lattice.bag",
            Law::AddCollect => "add.collect",
            Law::AddUnfactor => "add.unfactor",
            Law::AddContent => "add.content",
            Law::MulExpMerge => "mul.exp_merge",
            Law::MulRootnJoin => "mul.rootn_join",
            Law::MulInfinitySign => "mul.infinity_sign",
            Law::PowExpCompose => "pow.exp_compose",
            Law::PowRefusedPower => "pow.refused_power",
            Law::FunRootnIndex => "fun.rootn_index",
            Law::FunAtanhTanh => "fun.atanh_tanh",
            Law::FunLogExp => "fun.log_exp",
            Law::FunAsinhSinh => "fun.asinh_sinh",
            Law::FunSinhAsinh => "fun.sinh_asinh",
            Law::FunAcoshCosh => "fun.acosh_cosh",
            Law::FunExpLog => "fun.exp_log",
            Law::FunUnitInverses => "fun.unit_inverses",
            Law::PermInversePairs => "perm.inverse_pairs",
            Law::FunAbsNonnegative => "fun.abs_nonnegative",
            Law::FunHalfPeriod => "fun.half_period",
            Law::FunBesseljReflection => "fun.besselj_reflection",
            Law::LatticeAbsorb => "lattice.absorb",
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
            Law::AbsorbSum => "c1 + r*c2 + q = c",
            Law::AbsorbProduct => "r*c = c",
            Law::AbsorbSumSign => "-k*(s + c) = k*(-s + c)",
            Law::AbsorbRefitSign => "c*(a - b) = c*(b - a)",
            Law::AbsorbGroundShift => "c + g = c, c*g = c",
            Law::AbsorbCollapse => "L(c) = c",
            Law::AbsorbNanFamily => "L(c) = NaN",
            Law::NumLiteralArith => "r1 + r2 and r1*r2 are exact",
            Law::NumAbsLiteral => "abs(r) = r or -r",
            Law::NumExpOne => "exp(1) = e",
            Law::NumZeroPole => "0^q = inf, q < 0",
            Law::NumInfArith => "c*inf = sign(c)*inf, 0*inf = inf - inf = NaN",
            Law::NumInfPower => "inf^q = inf (q > 0), inf^q = 0 (q < 0)",
            Law::NumLiteralBaseInfExponent => "b^inf = 0 (abs(b) < 1), b^-inf = 0 (abs(b) > 1)",
            Law::NumTableFold => "f(r1, r2) = the table's value",
            Law::NumGroundClassFold => "g = NaN or g = +-inf, g ground",
            Law::NumGroundFloatFold => "g = its cheapest admitted value, g ground",
            Law::NumRootnIndex => "rootn(t, n) = NaN, n no nonzero integer",
            Law::NumPowParity => "(-inf)^k = (-1)^k*inf",
            Law::NumBesseljLiterals => "J_nu(0), J_(+-inf)(x) at literals",
            Law::NumLatticeNumbers => "max(r1, r2) = the greater",
            Law::NumLiteralToFloat => "r = its shorter float",
            Law::NumSignedZero => "t + (-0) = t, (-0)^-k = -inf (k odd)",
            Law::NumGroundFirst => "a fully numeric part is its value",
            Law::ParseSub => "a - b = a + (-1)*b",
            Law::ParseDiv => "a/b = a*b^-1",
            Law::ParseNegInv => "neg a = (-1)*a, inv a = a^-1",
            Law::ParseSignedLiteral => "-r = (-1)*r",
            Law::AddFlatten => "(a + b) + c = a + (b + c)",
            Law::MulFlatten => "(a*b)*c = a*(b*c)",
            Law::LatticeBag => "max(max(a, b), c) = max(a, b, c), max(a, a) = a",
            Law::AddCollect => "r1*t + r2*t = (r1 + r2)*t",
            Law::AddUnfactor => "r*(a + b) + c = r*a + r*b + c",
            Law::AddContent => "u*a + u*b = u*(a + b), -a - b = -(a + b)",
            Law::MulExpMerge => "exp(a)*exp(b) = exp(a + b)",
            Law::MulRootnJoin => "rootn(a, n)*rootn(b, n) = rootn(a*b, n)",
            Law::MulInfinitySign => "-k*(s + inf) = k*(-s - inf)",
            Law::PowExpCompose => "e^t = exp(t), exp(a)^b = exp(a*b)",
            Law::PowRefusedPower => "(a*b)^-k = ((a*b)^k)^-1",
            Law::FunRootnIndex => "rootn(t, -n) = 1/rootn(t, n)",
            Law::FunAtanhTanh => "atanh(tanh(t)) = t",
            Law::FunLogExp => "log(exp(t)) = t",
            Law::FunAsinhSinh => "asinh(sinh(t)) = t",
            Law::FunSinhAsinh => "sinh(asinh(t)) = t",
            Law::FunAcoshCosh => "acosh(cosh(t)) = abs(t)",
            Law::FunExpLog => "exp(log(t)) = t",
            Law::FunUnitInverses => "cos(acos(t)) = t, sin(asin(t)) = t, tanh(atanh(t)) = t",
            Law::PermInversePairs => "f(g(t)) = t, g the inverse of f",
            Law::FunAbsNonnegative => "abs(t) = t",
            Law::FunHalfPeriod => {
                "sin(t + pi) = -sin(t), cos(t + pi) = -cos(t), tan(t + pi) = tan(t)"
            }
            Law::FunBesseljReflection => "J_(-n)(t) = (-1)^n J_n(t)",
            Law::LatticeAbsorb => "max(a, min(a, b)) = a, max(inf, a) = inf, max(-inf, a) = a",
        }
    }

    /// Is the law's premise none in both modes: an identity at every binding, or, for the
    /// absorptions, at every value of the constants on the left?
    pub const fn total(self) -> bool {
        matches!(
            self,
            Law::MulOne
                | Law::PowZero
                | Law::PowOne
                | Law::PowBaseOne
                | Law::RootnOne
                | Law::AbsorbSum
                | Law::AbsorbProduct
                | Law::AbsorbSumSign
                | Law::AbsorbRefitSign
                | Law::AbsorbGroundShift
                | Law::AbsorbCollapse
                | Law::AbsorbNanFamily
                | Law::NumLiteralArith
                | Law::NumAbsLiteral
                | Law::NumExpOne
                | Law::NumZeroPole
                | Law::NumInfArith
                | Law::NumInfPower
                | Law::NumLiteralBaseInfExponent
                | Law::NumTableFold
                | Law::NumGroundClassFold
                | Law::NumGroundFloatFold
                | Law::NumRootnIndex
                | Law::NumPowParity
                | Law::NumBesseljLiterals
                | Law::NumLatticeNumbers
                | Law::NumLiteralToFloat
                | Law::NumSignedZero
                | Law::NumGroundFirst
                | Law::ParseSub
                | Law::ParseDiv
                | Law::ParseNegInv
                | Law::ParseSignedLiteral
                | Law::AddFlatten
                | Law::MulFlatten
                | Law::LatticeBag
                | Law::AddCollect
                | Law::MulExpMerge
                | Law::MulRootnJoin
                | Law::MulInfinitySign
                | Law::PowExpCompose
                | Law::PowRefusedPower
                | Law::FunRootnIndex
                | Law::FunAsinhSinh
                | Law::FunSinhAsinh
                | Law::FunHalfPeriod
                | Law::FunBesseljReflection
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
            // `t` is the sum: may a coefficient that is no positive number spread over it, may a
            // sign leave or enter it?
            Law::AddUnfactor | Law::AddContent => !super::expr::sum_holds_its_sign(t, cx),
            // `t` is the argument of the inner function.
            Law::FunAtanhTanh => !cx.f64_semantics() || cx.certainly_in_band(t, 19),
            Law::FunLogExp => {
                !cx.f64_semantics()
                    || (cx.flat_unread && super::expr::signed_zero_null(t, true, cx))
            }
            Law::FunAcoshCosh => !cx.f64_semantics() || cx.flat_unread,
            Law::FunExpLog | Law::FunAbsNonnegative => {
                cx.certainly_nonneg(t)
                    && (!cx.f64_semantics() || super::expr::signed_zero_null(t, true, cx))
            }
            Law::FunUnitInverses => cx.certainly_in_unit(t),
            Law::PermInversePairs => cx.permits_on(Permissive::InversePairs, t),
            // `t` is a member the bag drops.
            Law::LatticeAbsorb => {
                matches!(t, Ex::PosInf | Ex::NegInf)
                    || super::expr::lattice_numeric(t, cx)
                    || cx.fin_licensed(t)
            }
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
