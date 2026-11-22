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
//! | `add.group` | c*a + c*b + d = c*(a + b) + d, the terms grouped by the exact value of c, where the group costs no more | none (a sign leaving a group: `add.content`) | none |
//! | `mul.exp_merge` | exp(a)*exp(b) = exp(a + b) | none | none |
//! | `mul.rootn_join` | rootn(a, n)*rootn(b, n) = rootn(a*b, n) | none | none |
//! | `mul.infinity_sign` | -k*(s + inf) = k*(-s - inf) | none | none |
//! | `pow.exp_compose` | e^t = exp(t), exp(a)^b = exp(a*b) | none | none |
//! | `pow.refused_power` | (a*b)^-k = ((a*b)^k)^-1 | none | none |
//! | `pow.distribute` | (a*b)^n = a^n*b^n, n an integer | n >= 0 or even; else every member nonzero almost everywhere, or one infinite, or all >= 0, or a zero whose sign an inverse reads | as `f64` |
//! | `parse.inverse_of_product` | 1/(a*b) = (1/a)*(1/b) as a divisor is read | as `pow.distribute` | as `pow.distribute` |
//! | `pow.compose` | (t^r)^s = t^(r*s) | s an integer; r an odd integer (positive, or t never infinite) and r*s no integer; r and r*s no integers; or t >= 0 | none |
//! | `pow.root_inverse` | rootn(t, n)^n = t, rootn(t, n)^-n = 1/t, rootn(t^n, n) = t | n odd, or t >= 0 | none |
//! | `pow.odd_root_compose` | rootn(t, m)^s = t^(s/m), m odd | s no integer | s no integer: the root's class keeps one spelling |
//! | `pow.unknown_parity_root` | rootn(t, m)^(-1/n) = 1/rootn(rootn(t, m), n), n even | none | none |
//! | `pow.literal_base_sign` | r^(-t) = (1/r)^t, 1/(r^t) = (1/r)^t, (1/t)^(+-inf) = t^(-+inf) | none | none |
//! | `pow.half_root` | t^(1/(2k)) = rootn(t, 2k) | t^(1/2) as numpy's `sqrt`: t is no zero with a sign of its own | none |
//! | `mul.collect` | t^a*t^b = t^(a+b) | a and b integers float64 reads exactly; or t >= 0; or a + b a nonzero non-integer | none |
//! | `mul.collect_symbolic` | t^y*t^y = t^(2*y) | t not `-0` on a region, and t >= 0 or the level sets of y null | none |
//! | `fun.rootn_index` | rootn(t, -n) = 1/rootn(t, n) | none | none |
//! | `fun.atanh_tanh` | atanh(tanh(t)) = t | abs(t) at most 19: tanh(t) is no +-1 | none |
//! | `fun.log_exp` | log(exp(t)) = t | t is not `-0` on a region, and nothing around reads its flat point | none |
//! | `fun.asinh_sinh` | asinh(sinh(t)) = t | none | none |
//! | `fun.sinh_asinh` | sinh(asinh(t)) = t | none | none |
//! | `fun.acosh_cosh` | acosh(cosh(t)) = abs(t) | nothing around reads its flat point | none |
//! | `fun.exp_log` | exp(log(t)) = t | t >= 0, not `-0` on a region | none |
//! | `fun.log_inverse` | log(1/t) = -log(t), log(abs(1/t)) = -log(abs(t)) | t a monomial: 1 only at points | none |
//! | `fun.unit_inverses` | cos(acos(t)) = t, sin(asin(t)) = t, tanh(atanh(t)) = t | t in [-1, 1] | none |
//! | `perm.inverse_pairs` | f(g(t)) = t, g the inverse of f | no step | none |
//! | `fun.abs_nonnegative` | abs(t) = t | t >= 0, not `-0` on a region | t >= 0 |
//! | `fun.half_period` | sin(t + pi) = -sin(t), cos(t + pi) = -cos(t), tan(t + pi) = tan(t) | none | none |
//! | `fun.besselj_reflection` | J_(-n)(t) = (-1)^n J_n(t) | none | none |
//! | `lattice.absorb` | max(a, min(a, b)) = a, max(inf, a) = inf, max(-inf, a) = a | the dropped members no NaN almost everywhere | none |
//! | `class.sign_trade` | f(-S) = -f(S), f an odd carrier of a sum S | a sign enters S only where S holds none (as `add.content`) | none |
//! | `class.pole_orientation` | f(-S)^-k = -f(S)^-k, k odd | S holds no sign (as `class.sign_trade`); a one-signed pole keeps its limit | none |
//! | `class.opposite_sums` | S^a*(-S)^b = (-1)^b*S^(a+b) | a + b < 0: as `class.pole_orientation` | none |
//! | `class.like_terms_up_to_sign` | u*f(S) + v*f(-S) = (u - v)*f(S), f odd | f(S) infinite at S's zero: as `class.pole_orientation` | none |
//! | `pow.even_orientation` | (-S)^(2k) = S^(2k) | none | none |
//! | `pow.negated_site` | (-F)^s = flip(F)^s | the flip as `class.sign_trade`'s; s odd and negative: F's zero no region with a sign of its own | none |
//! | `fun.even` | g(-t) = g(t), g even | none | none |
//! | `fun.odd_sign` | f(-u) = -f(u), f odd | none | none |
//! | `fun.atan2_sign` | atan2(-y, x) = -atan2(y, x) | none | no step (one zero) |
//! | `fun.besseli_parity` | I_n(-t) = (-1)^n I_n(t), n an integer | n odd: t vanishes on no region | none |
//! | `search.distribute` | A*(B + C) = A*B + A*C | A infinite only on a null set, at most one addend keeping each pole (or those that do of one strict sign), and a pole's reading of the product's zero kept | a spelled infinity's |
//! | `search.pow_expand` | (a + b)^n = the expansion, 2 <= n <= 6 | as `search.distribute`, every term a shared piece | a spelled infinity's |
//! | `search.factor` | c*q1 + c*q2 = c*(q1 + q2) | c infinite only on a null set; under a pole's reading, c positive and never zero | a spelled infinity's |
//! | `search.reciprocal_join` | A^-1*B^-1 = (A*B)^-1, f*A^-1 = (A/f)^-1 | no step | f of one strict sign |
//! | `search.composite_step` | a refused rebuild is kept where the whole descends | none | none |
//! | `settle.rule` | a mined rule fires where its left side binds | none (the rule's own) | none (the rule's own) |
//! | `settle.negation_twin` | L -> T entails -L -> -T | none | none |
//! | `perm.reciprocal_projection` | a^-1*b^-1 = (a*b)^-1, as written | no step | none |
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
use super::rat::Rat;

/// What a law of powers reads beside its base ([`Law::holds_powers`]).
#[derive(Clone, Copy)]
pub(crate) enum Powers<'a> {
    /// `(base^r)^s`, `rootn(base, r)^s`, `base^r*base^s`: two rational exponents.
    Rational {
        base: &'a Ex,
        r: &'a Rat,
        s: &'a Rat,
    },
    /// `base^y*base^y`: one symbolic exponent.
    Symbolic { base: &'a Ex, y: &'a Ex },
    /// `(-site)^s`: a negated sign-trade site under a rational exponent.
    Site { site: &'a Ex, s: &'a Rat },
    /// `(members[0]*members[1]*...)^n`: a product under an integer power.
    Product { members: &'a [Ex], n: &'a Rat },
}

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
    /// A sum writes a number its terms share once: its terms grouped by the exact value of their
    /// coefficient up to sign, each group of two or more `c*(group)` where that costs no more than
    /// its terms distributed, `c*a + c*b + d = c*(a + b) + d`. Total: a positive coefficient
    /// moves no zero's sign, and float64 only rounds the two spellings apart. Each group is the
    /// whole-sum factoring of its terms, so a sign leaves it as `add.content` says (in `f64` only
    /// where the group's sum holds none).
    AddGroup,
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
    /// `(a*b)^n = a^n*b^n` for an integer `n`. An odd negative power of a product that vanishes
    /// with a negative member is a pole of one sign where the members' poles give the other
    /// (`(-1*(x0 - abs(x0)))^-1` is `+inf` on x0 >= 0, `-1*(x0 - abs(x0))^-1` is `-inf`), so it
    /// distributes where every member is nonzero almost everywhere, a member is infinite, all are
    /// `>= 0`, or a member's zero is one whose sign an inverse reads (`expr::power_bag`). Both
    /// modes read the same licence; `permissive`'s certificates admit every member.
    PowDistribute,
    /// The divisor of a written product is read as its members' inverses where `pow.distribute`
    /// holds: `x2/(x0*x1)` is `x2*x0^-1*x1^-1` (`convert::powered_members`); else the product
    /// stays one member under the power.
    ParseInverseOfProduct,
    /// `(t^r)^s = t^(r*s)`. True where both sides agree on t < 0: an integer `s`; an odd integer
    /// `r` with `r*s` no integer (the source is NaN on t < 0 as the target is), `r` positive or
    /// `t` never infinite (at t = -inf a negative odd power is 0, whose non-integer power is 0,
    /// where the target is NaN); `r` and `r*s` no integers; or `t >= 0`. `(t^2)^(1/2)` is
    /// `abs(t)`, `(t^(1/2))^2` is `t` only on t >= 0. In `permissive`, none (powers compose).
    PowCompose,
    /// `rootn(t, n)^n = t`, `rootn(t, n)^-n = 1/t` and `rootn(t^n, n) = t`, where `n` is odd
    /// (the odd root is a bijection of the extended line) or `t >= 0`: `rootn(t, 2)^2` has no
    /// value on t < 0. In `permissive`, none (powers compose).
    PowRootInverse,
    /// `rootn(t, m)^s = t^(s/m)` for an odd index `m >= 3` and a non-integer `s`: both sides are
    /// NaN on t < 0. An integer `s` keeps the root in both modes: on t < 0 it is no identity
    /// (`rootn(-8, 3)^2` is 4, `(-8)^(2/3)` has no value), and where it is one (t >= 0, or in
    /// `permissive`, which composes powers ignoring the sign) the root's class keeps one spelling:
    /// `rootn(t, m)^k` collects with `rootn(t, m)` over one base, `t^(k/m)` would not.
    PowOddRootCompose,
    /// `rootn(t, m)^(-1/n) = 1/rootn(rootn(t, m), n)` for an even `n` over an index of no known
    /// parity (beyond 2^53, `f64`). Total: both sides are `1/R^(1/n)` on R = `rootn(t, m)` >= 0,
    /// NaN on R < 0 and `+inf` at R = 0.
    PowUnknownParityRoot,
    /// A literal base absorbs the exponent's sign: `r^(-t) = (1/r)^t` (`0^(-t) = inf^t`) where
    /// the flip removes a node, `1/(r^t) = (1/r)^t`, and `(1/t)^(+-inf) = t^(-+inf)`. Total over
    /// the rational literals; `-inf` is no such base (`(-inf)^(-t)` has no value at a non-integer
    /// `t`, where `0^t` is 0).
    PowLiteralBaseSign,
    /// An even unit fraction is a root: `t^(1/(2k)) = rootn(t, 2k)`, one principal root. In
    /// `f64` the spelling `pow(t, 0.5)` is numpy's `sqrt`, which keeps a zero's sign
    /// (`sqrt(-0) = -0`, `rootn(-0, 2) = +0`), so it stays a call where `t` may be a zero with a
    /// sign of its own (`expr::sqrt_parts_from_rootn`): `1/pow(ceil(x0), 1/2)` is `-inf` on
    /// (-1, 0), `1/rootn(ceil(x0), 2)` is `+inf`. In `permissive`, none (one zero).
    PowHalfRoot,
    /// Factors of one base merge their exponents: `t^a*t^b = t^(a+b)`, where both are integers
    /// (in `f64` integers float64 reads exactly, parity known), `t >= 0`, or the sum is a nonzero
    /// non-integer (both sides NaN on t < 0): `t^(1/2)*t^(1/2)` is `t` only on t >= 0. Exponents
    /// of opposite signs share one pool where `mul.cancel` holds of `t`. In `permissive`, none
    /// (powers compose by agreement).
    MulCollect,
    /// Occurrences of one symbolic power merge: `t^y*t^y = t^(2*y)`. The two sides part on
    /// {t < 0} x {y in (1/2)Z \ Z}, NaN against a value, so `t >= 0` or the level sets of `y`
    /// null (`y` nonconstant entire, `interval::nonconstant_entire`, the certificate the judge
    /// reads too). In `f64` `t` is not `-0` on a region either: where `t` is `-0` and `2*y` an odd
    /// integer `(-0)^y*(-0)^y` is `+0` or `+inf` and `(-0)^(2*y)` is `-0` or `-inf`
    /// (`ceil(x0)^x1*ceil(x0)^x1` is `+inf` at x0 = -0.5, x1 = -0.5, `ceil(x0)^(2*x1)` is
    /// `-inf`), a value where the input has one. In `permissive`, none (agreement).
    MulCollectSymbolic,
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
    /// `log(1/t) = -log(t)`. Float64's `log(1/t)` is `+0` where `-log(t)` is `-0`, wherever `t`
    /// is 1, so in `f64` it holds where `t` is a monomial (`expr::log_inverse_argument`:
    /// variables, fitted constants and nonzero numbers under integer powers and `abs`), whose
    /// float64 value is 1 only at points: the sign of a zero at an isolated point, and the sign
    /// of the infinity a pole makes of it, is no value the answer keeps (`1/log(1/x0)` is
    /// `-1/log(x0)`, `+inf` against `-inf` at x0 = 1 alone). Where `t` rounds to 1 on a region
    /// (`exp(x0)`, `ceil(x0)`) the sign stays, and the law does not hold. `permissive` reads one
    /// zero, and holds it everywhere. In both modes the collapse waits for the root's walk
    /// (`release_flat`), so the constructors take `exp(log(1/t)) = 1/t` first: `exp(-log(t))` is
    /// no inverse pair.
    FunLogInverse,
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
    /// A product's sign trades between its coefficient and an odd carrier of a sum `S` (the sum
    /// itself, an odd positive power, an odd root, an odd function or an odd `besselj` order of
    /// one): `f(-S) = -f(S)`, the cheaper placement standing (`expr::sign_place`), the sites over
    /// one sum flipping as a whole. Float64 negates exactly but where `S` cancels, which is `+0`
    /// in either orientation, so `-f(S)` is `-0` there and `f(-S)` is `+0`. In `f64` a sign enters
    /// `S` only where `S` holds no sign (`expr::sum_holds_its_sign`, as `add.content`): where it
    /// may vanish on a region and the expression around it reads its zero, the sign stays on the
    /// coefficient (`1/(2*(-ceil(x0) - floor(x1)))` is `+inf` wherever `ceil(x0) = -floor(x1)`,
    /// `-1/(2*(ceil(x0) + floor(x1)))` is `-inf`). A configuration also keeps the value the product
    /// keeps at the zero of each class (`expr::zero_limit`). [`Law::holds`] reads the sum. In
    /// `permissive`, none.
    ClassSignTrade,
    /// The orientation of a sum `S` under an odd negative power: `f(-S)^-k = -f(S)^-k`
    /// (`-1/(x0 - 3) = 1/(3 - x0)`), at a lone inverse, a pole that distribution splits, a negated
    /// site and like terms through a pole alike. The two sides agree wherever `S` is nonzero. At an
    /// isolated zero float64 gives each side the infinity of the sign written (a sum cancels to
    /// `+0` in either orientation), the sign of a zero at a point, which `f64` does not keep: a
    /// two-signed pole has no value the answer keeps, and a one-signed one keeps its limit
    /// (`expr::zero_limit`, asked by the sign placement). Where `S` may vanish on a region and the
    /// expression around it reads its zero, `S` holds its sign (`expr::sum_holds_its_sign`, as
    /// `class.sign_trade`). [`Law::holds`] reads the class's sum. In `permissive`, none: the reals
    /// give `1/0` no value.
    ClassPoleOrientation,
    /// Factors over a sum and its negation are one class, decided on the product's exponent
    /// table: `S^a*(-S)^b = (-1)^b*S^(a+b)` (`(9.6 - x)*(x - 9.6) = -(x - 9.6)^2`,
    /// `(x - 3)/(3 - x) = -1`). A positive total is free. In `f64` a negative total keeps the limit
    /// of a one-signed pole (`expr::zero_limit`): `1/((x0 - 3)*(3 - x0))` is `-1/(x0 - 3)^2`,
    /// `-inf` beside x0 = 3 on both sides, the isolated zero's own sign aside
    /// (`class.pole_orientation`). [`Law::holds`] reads the class's sum, asked of a negative total.
    /// In `permissive`, none.
    ClassOppositeSums,
    /// Like terms are like up to the orientation of their classes: `1/(x3 - x2)` and
    /// `-1/(x2 - x3)` are one term. In `f64` a term keeps the limit its class's zero has where it
    /// keeps one (`expr::zero_limit`), and a two-signed pole's infinity is an isolated zero's sign
    /// (`class.pole_orientation`): `1/(x0 - 3) + 1/(3 - x0)` is 0. [`Law::holds`] reads the class's
    /// sum, asked of a term infinite at its zero. In `permissive`, none.
    ClassLikeTermsUpToSign,
    /// An even power erases its base's orientation: `(-S)^(2k) = S^(2k)`, `f(-S)^(2k) = f(S)^(2k)`
    /// and `f(-c*u)^(2k) = f(c*u)^(2k)` for an odd head `f`. Total: bit for bit, zeros and poles
    /// included.
    PowEvenOrientation,
    /// A negated site under a power is the power of its flip: `(-F)^s = flip(F)^s`, the flip a
    /// sign trade into `F`'s sum (`class.sign_trade`'s premise). In `f64` an odd negative `s` reads
    /// the sign of `F`'s zero: where `F` may vanish on a region with a sign of its own
    /// (`expr::region_signed_zero`) `1/-(S^3)` is `-inf` there and `1/flip(S)^3` is `+inf`, and the
    /// power distributes instead; at an isolated zero the pole's sign is no value the answer keeps
    /// (`class.pole_orientation`). [`Law::holds_powers`] reads the site and the exponent. In
    /// `permissive`, none.
    PowNegatedSite,
    /// An even function eats its argument's sign and orientation: `g(-t) = g(t)` for `g` in
    /// {`abs`, `cos`, `cosh`} and `besselj` of an even order, through odd heads, `abs` and integer
    /// powers (`|sin(-x)| = |sin(x)|`, `cos(y - x) = cos(x - y)`). Total: bit for bit.
    FunEven,
    /// An odd head of a negative literal multiple takes the cheaper spelling: `f(-u) = -f(u)`
    /// (`sin(-2*x0)` is `-sin(2*x0)`; a ground argument keeps its sign, `sin(-2)`). Total: odd bit
    /// for bit, a zero's sign included.
    FunOddSign,
    /// `atan2(-y, x) = -atan2(y, x)`: odd in `y` bit for bit, signed zeros included. A step of
    /// `f64` alone: `permissive` reads one zero, where `atan2(0, -1)` is `pi` and `-atan2(0, -1)`
    /// is `-pi`.
    FunAtan2Sign,
    /// `I_n(-t) = (-1)^n I_n(t)` for an integer order `n >= 0`: an even order reads the argument an
    /// even function reads (`besseli(2, -x)` is `besseli(2, x)`), an odd order moves the sign of a
    /// negative literal multiple out (`besseli(1, -2*x)` is `-besseli(1, 2*x)`). The kernel is even
    /// bit for bit at an even order; at an odd one it is odd but at a zero, where it gives `+0`
    /// for either sign, so in `f64` an odd order moves the sign only where `t` vanishes on no
    /// region (`besseli(1, -ceil(x0))` is `+0` on (-1, 0), `-besseli(1, ceil(x0))` is `-0`).
    /// [`Law::holds`] reads the argument, asked of an odd order. In `permissive`, none.
    FunBesseliParity,
    /// THE DISTRIBUTIVE LAW, as the search applies it (DISTRIBUTE, `ac::search`):
    /// `A*(B + C) = A*B + A*C`, a product over its sums, one factor into one sum, or the other
    /// factors over one term of a sum. The two sides differ only where a piece the law multiplies
    /// into two or more addends (`A`) is infinite: at `A = +inf`, `B = 1`, `C = 0` the left side
    /// is `+inf` and the right side `inf + NaN`. In `f64` such a piece is infinite only on a null
    /// set (`Cx::inf_null`; it may have no value on a region, as the product then has none), and
    /// at each of its poles, read by its limit, at most one addend keeps the pole, each addend
    /// that keeps it carries two-signed poles only, whose infinity's sign is an isolated zero's
    /// (`class.pole_orientation`), or those addends take one strict sign:
    /// `(1/x0)*(1 + x0)` is `1/x0 + 1`, one move whose constructors cancel `x0/x0`, and
    /// `(x1 + x2)/x0` may split into `x1/x0 + x2/x0`, while `(x1 + x2)/abs(x0)` stays: its pole is
    /// `+inf` on both sides of x0 = 0 where x1 + x2 > 0, and the split is `inf - inf` there
    /// wherever x1 and x2 differ in sign; so does `(x2 + 2*sqrt(x1))/x1`, whose co-factor
    /// `sqrt(x1)` makes the pole one-sided.
    /// [`Law::holds`] reads a shared piece. In `permissive`, none, but a piece that spells an
    /// infinity needs the sound certificate in every mode (`-inf*(x1 - 1/2)` is not
    /// `inf - inf*x1`).
    SearchDistribute,
    /// An integer power of a sum expands, `(a + b)^n` for `2 <= n <= 6`: the distributive law over
    /// `n` copies of the sum, each term a shared piece (`search.distribute`'s premise):
    /// `(1/x0 + x1)^2` stays, its pole `+inf` on both sides of x0 = 0, where `1/x0^2 + 2*x1/x0`
    /// is NaN for x1 < 0. [`Law::holds`] reads a term.
    SearchPowExpand,
    /// THE DISTRIBUTIVE LAW READ BACKWARDS, as the search applies it (FACTOR):
    /// `c*q1 + c*q2 = c*(q1 + q2)`, by what a group of terms shares (shared bases, up to a class's
    /// orientation, a sum factor spread among the terms). The two sides differ only where the
    /// content `c` is infinite: where the quotients take one sign there both are that infinity,
    /// and where they differ in sign the sum has no value and the product may have one. In `f64`
    /// the content is infinite only on a null set (`Cx::inf_null`: a value gained on a null set,
    /// never a region) and may have no value on a region (`x1*sqrt(x0) + sqrt(x0)` is
    /// `sqrt(x0)*(x1 + 1)`); a quotient needs nothing. Under an inverse the sum's zero is a pole
    /// whose sign is an isolated zero's (`class.pole_orientation`): `1/(x1 + x1*x2)` is
    /// `1/(x1*(x2 + 1))`; where the sum keeps its zero's sign on a region and a reader reads it,
    /// the content leaves the sum only where it is positive and never zero
    /// (`1/(2*asin(ceil(x1)) - asin(ceil(x1)))` stays). [`Law::holds`] reads the content. In
    /// `permissive`, none but a spelled infinity's, as `search.distribute`.
    SearchFactor,
    /// `permissive`'s search joins a product's reciprocal sums (RECIPROCAL JOIN), which its `pow`
    /// holds split: `A^-1*B^-1 = (A*B)^-1`, `c*A^-1 = (A/c)^-1` and `f*A^-1 = (A/f)^-1`, each
    /// joined sum distributed. The last loses a value at a zero of `f` (the left side 0,
    /// `(A/0)^-1` no value), so `f` takes one strict sign wherever it has a value
    /// (`ac::search::strict_sign`): `2*tan(x0)/(1 - tan(x0)^2)` stays. [`Law::holds`] reads `f`. In
    /// `f64`, no step: its `pow` keeps such a reciprocal joined where it is not certified to split,
    /// and DISTRIBUTE expands inside it.
    SearchReciprocalJoin,
    /// THE COMPOSITE STEP, the search's: a rebuild the settling refused, because it rises at its
    /// node and in the whole expression, is proposed as the state it builds and kept where that
    /// state's descent lands below (an integer power distributing over a product the settling
    /// mints, whose literal factor then folds). An acceptance, no identity of its own: every step
    /// of the descent is one. Total.
    SearchCompositeStep,
    /// A mined rule fires where its left side binds (`?` a variable, `!` a part finite almost
    /// everywhere, `$` finite and nonzero almost everywhere, `_` anything; in `permissive` any slot
    /// binds anything) and the whole expression's price falls (`rules::rewrite_pass`). Its
    /// premise is the rule's own, judged at mining and read at its slots' sorts at bind time.
    /// Total as a step of the catalogue.
    SettleRule,
    /// An admitted rule `L -> T` entails `-L -> -T`, minted where the negation is absorbed
    /// (`rules::AcRules::translate`): negation is exact in both modes. Total.
    SettleNegationTwin,
    /// `permissive`'s answer writes a product's negative powers joined, `(a*b)^-1`, where that is
    /// cheaper than `a^-1*b^-1` and the sound modes would not split it (`expr::rejoin_reciprocals`):
    /// one value, the sign that leaves the base read by permissive's agreement. A spelling of the
    /// answer, after the walk. Total; `f64` takes no such step.
    PermReciprocalProjection,
}

impl Law {
    /// The whole catalogue, in its order.
    pub const ALL: [Law; 92] = [
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
        Law::AddGroup,
        Law::MulExpMerge,
        Law::MulRootnJoin,
        Law::MulInfinitySign,
        Law::PowExpCompose,
        Law::PowRefusedPower,
        Law::FunRootnIndex,
        Law::PowDistribute,
        Law::ParseInverseOfProduct,
        Law::PowCompose,
        Law::PowRootInverse,
        Law::PowOddRootCompose,
        Law::PowUnknownParityRoot,
        Law::PowLiteralBaseSign,
        Law::PowHalfRoot,
        Law::MulCollect,
        Law::MulCollectSymbolic,
        Law::FunAtanhTanh,
        Law::FunLogExp,
        Law::FunAsinhSinh,
        Law::FunSinhAsinh,
        Law::FunAcoshCosh,
        Law::FunExpLog,
        Law::FunLogInverse,
        Law::FunUnitInverses,
        Law::PermInversePairs,
        Law::FunAbsNonnegative,
        Law::FunHalfPeriod,
        Law::FunBesseljReflection,
        Law::LatticeAbsorb,
        Law::ClassSignTrade,
        Law::ClassPoleOrientation,
        Law::ClassOppositeSums,
        Law::ClassLikeTermsUpToSign,
        Law::PowEvenOrientation,
        Law::PowNegatedSite,
        Law::FunEven,
        Law::FunOddSign,
        Law::FunAtan2Sign,
        Law::FunBesseliParity,
        Law::SearchDistribute,
        Law::SearchPowExpand,
        Law::SearchFactor,
        Law::SearchReciprocalJoin,
        Law::SearchCompositeStep,
        Law::SettleRule,
        Law::SettleNegationTwin,
        Law::PermReciprocalProjection,
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
            Law::AddGroup => "add.group",
            Law::MulExpMerge => "mul.exp_merge",
            Law::MulRootnJoin => "mul.rootn_join",
            Law::MulInfinitySign => "mul.infinity_sign",
            Law::PowExpCompose => "pow.exp_compose",
            Law::PowRefusedPower => "pow.refused_power",
            Law::FunRootnIndex => "fun.rootn_index",
            Law::PowDistribute => "pow.distribute",
            Law::ParseInverseOfProduct => "parse.inverse_of_product",
            Law::PowCompose => "pow.compose",
            Law::PowRootInverse => "pow.root_inverse",
            Law::PowOddRootCompose => "pow.odd_root_compose",
            Law::PowUnknownParityRoot => "pow.unknown_parity_root",
            Law::PowLiteralBaseSign => "pow.literal_base_sign",
            Law::PowHalfRoot => "pow.half_root",
            Law::MulCollect => "mul.collect",
            Law::MulCollectSymbolic => "mul.collect_symbolic",
            Law::FunAtanhTanh => "fun.atanh_tanh",
            Law::FunLogExp => "fun.log_exp",
            Law::FunAsinhSinh => "fun.asinh_sinh",
            Law::FunSinhAsinh => "fun.sinh_asinh",
            Law::FunAcoshCosh => "fun.acosh_cosh",
            Law::FunExpLog => "fun.exp_log",
            Law::FunLogInverse => "fun.log_inverse",
            Law::FunUnitInverses => "fun.unit_inverses",
            Law::PermInversePairs => "perm.inverse_pairs",
            Law::FunAbsNonnegative => "fun.abs_nonnegative",
            Law::FunHalfPeriod => "fun.half_period",
            Law::FunBesseljReflection => "fun.besselj_reflection",
            Law::LatticeAbsorb => "lattice.absorb",
            Law::ClassSignTrade => "class.sign_trade",
            Law::ClassPoleOrientation => "class.pole_orientation",
            Law::ClassOppositeSums => "class.opposite_sums",
            Law::ClassLikeTermsUpToSign => "class.like_terms_up_to_sign",
            Law::PowEvenOrientation => "pow.even_orientation",
            Law::PowNegatedSite => "pow.negated_site",
            Law::FunEven => "fun.even",
            Law::FunOddSign => "fun.odd_sign",
            Law::FunAtan2Sign => "fun.atan2_sign",
            Law::FunBesseliParity => "fun.besseli_parity",
            Law::SearchDistribute => "search.distribute",
            Law::SearchPowExpand => "search.pow_expand",
            Law::SearchFactor => "search.factor",
            Law::SearchReciprocalJoin => "search.reciprocal_join",
            Law::SearchCompositeStep => "search.composite_step",
            Law::SettleRule => "settle.rule",
            Law::SettleNegationTwin => "settle.negation_twin",
            Law::PermReciprocalProjection => "perm.reciprocal_projection",
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
            Law::AddGroup => "c*a + c*b + d = c*(a + b) + d",
            Law::MulExpMerge => "exp(a)*exp(b) = exp(a + b)",
            Law::MulRootnJoin => "rootn(a, n)*rootn(b, n) = rootn(a*b, n)",
            Law::MulInfinitySign => "-k*(s + inf) = k*(-s - inf)",
            Law::PowExpCompose => "e^t = exp(t), exp(a)^b = exp(a*b)",
            Law::PowRefusedPower => "(a*b)^-k = ((a*b)^k)^-1",
            Law::FunRootnIndex => "rootn(t, -n) = 1/rootn(t, n)",
            Law::PowDistribute => "(a*b)^n = a^n*b^n",
            Law::ParseInverseOfProduct => "1/(a*b) = (1/a)*(1/b)",
            Law::PowCompose => "(t^r)^s = t^(r*s)",
            Law::PowRootInverse => "rootn(t, n)^n = t",
            Law::PowOddRootCompose => "rootn(t, m)^s = t^(s/m)",
            Law::PowUnknownParityRoot => "rootn(t, m)^(-1/n) = 1/rootn(rootn(t, m), n)",
            Law::PowLiteralBaseSign => "r^(-t) = (1/r)^t",
            Law::PowHalfRoot => "t^(1/(2*k)) = rootn(t, 2*k)",
            Law::MulCollect => "t^a*t^b = t^(a+b)",
            Law::MulCollectSymbolic => "t^y*t^y = t^(2*y)",
            Law::FunAtanhTanh => "atanh(tanh(t)) = t",
            Law::FunLogExp => "log(exp(t)) = t",
            Law::FunAsinhSinh => "asinh(sinh(t)) = t",
            Law::FunSinhAsinh => "sinh(asinh(t)) = t",
            Law::FunAcoshCosh => "acosh(cosh(t)) = abs(t)",
            Law::FunExpLog => "exp(log(t)) = t",
            Law::FunLogInverse => "log(1/t) = -log(t), log(abs(1/t)) = -log(abs(t))",
            Law::FunUnitInverses => "cos(acos(t)) = t, sin(asin(t)) = t, tanh(atanh(t)) = t",
            Law::PermInversePairs => "f(g(t)) = t, g the inverse of f",
            Law::FunAbsNonnegative => "abs(t) = t",
            Law::FunHalfPeriod => {
                "sin(t + pi) = -sin(t), cos(t + pi) = -cos(t), tan(t + pi) = tan(t)"
            }
            Law::FunBesseljReflection => "J_(-n)(t) = (-1)^n J_n(t)",
            Law::LatticeAbsorb => "max(a, min(a, b)) = a, max(inf, a) = inf, max(-inf, a) = a",
            Law::ClassSignTrade => "f(-S) = -f(S)",
            Law::ClassPoleOrientation => "f(-S)^-k = -f(S)^-k, k odd",
            Law::ClassOppositeSums => "S^a*(-S)^b = (-1)^b*S^(a+b)",
            Law::ClassLikeTermsUpToSign => "u*f(S) + v*f(-S) = (u - v)*f(S)",
            Law::PowEvenOrientation => "(-S)^(2*k) = S^(2*k)",
            Law::PowNegatedSite => "(-F)^s = flip(F)^s",
            Law::FunEven => "g(-t) = g(t)",
            Law::FunOddSign => "f(-u) = -f(u)",
            Law::FunAtan2Sign => "atan2(-y, x) = -atan2(y, x)",
            Law::FunBesseliParity => "I_n(-t) = (-1)^n I_n(t)",
            Law::SearchDistribute => "A*(B + C) = A*B + A*C",
            Law::SearchPowExpand => "(a + b)^n = its expansion, 2 <= n <= 6",
            Law::SearchFactor => "c*q1 + c*q2 = c*(q1 + q2)",
            Law::SearchReciprocalJoin => "A^-1*B^-1 = (A*B)^-1, f*A^-1 = (A/f)^-1",
            Law::SearchCompositeStep => "a refused rebuild is kept where the whole descends",
            Law::SettleRule => "a mined rule L -> T fires where L binds",
            Law::SettleNegationTwin => "L -> T entails -L -> -T",
            Law::PermReciprocalProjection => "a^-1*b^-1 = (a*b)^-1, as written",
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
                | Law::AddGroup
                | Law::MulExpMerge
                | Law::MulRootnJoin
                | Law::MulInfinitySign
                | Law::PowExpCompose
                | Law::PowRefusedPower
                | Law::FunRootnIndex
                | Law::PowUnknownParityRoot
                | Law::PowLiteralBaseSign
                | Law::FunAsinhSinh
                | Law::FunSinhAsinh
                | Law::FunHalfPeriod
                | Law::FunBesseljReflection
                | Law::PowEvenOrientation
                | Law::FunEven
                | Law::FunOddSign
                | Law::FunAtan2Sign
                | Law::SearchCompositeStep
                | Law::SettleRule
                | Law::SettleNegationTwin
                | Law::PermReciprocalProjection
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
            Law::AddUnfactor | Law::AddContent | Law::ClassSignTrade => {
                !super::expr::sum_holds_its_sign(t, cx)
            }
            // `t` is the class's sum (its key): a sign moves through it where it holds none.
            Law::ClassPoleOrientation | Law::ClassOppositeSums | Law::ClassLikeTermsUpToSign => {
                !super::expr::sum_holds_its_sign(t, cx)
            }
            // `t` is the argument of an odd order.
            Law::FunBesseliParity => !super::expr::vanishes_on_a_region(t, cx),
            // `t` is a piece the law multiplies into two or more addends, or the content it
            // collects out of them; the poles and a pole's reading are asked at the place
            // (`ac::search`).
            Law::SearchDistribute | Law::SearchPowExpand | Law::SearchFactor => {
                if t.contains_infinity() {
                    cx.fin_certified(t)
                } else {
                    cx.permits(Permissive::LikeTerms) || cx.inf_null(t)
                }
            }
            // `t` is the factor joined into a reciprocal sum.
            Law::SearchReciprocalJoin => {
                cx.permits(Permissive::Agreement) && super::search::strict_sign(t, cx).is_some()
            }
            // `t` is the argument of the inner function.
            Law::FunAtanhTanh => !cx.f64_semantics() || cx.certainly_in_band(t, 19),
            Law::FunLogExp => {
                !cx.f64_semantics()
                    || (cx.flat_unread && super::expr::signed_zero_null(t, true, cx))
            }
            Law::FunAcoshCosh => !cx.f64_semantics() || cx.flat_unread,
            Law::FunLogInverse => {
                cx.flat_unread
                    && (!cx.f64_semantics() || super::expr::log_inverse_argument(t, cx.view))
            }
            Law::FunExpLog | Law::FunAbsNonnegative => {
                cx.certainly_nonneg(t)
                    && (!cx.f64_semantics() || super::expr::signed_zero_null(t, true, cx))
            }
            Law::FunUnitInverses => cx.certainly_in_unit(t),
            // `t` is the base of the spelled square root.
            Law::PowHalfRoot => !super::expr::sqrt_parts_from_rootn(t, cx),
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

    /// [`Law::holds`] for a law of powers, whose premise reads the exponents beside the base
    /// ([`Powers`]).
    pub(crate) fn holds_powers(self, p: Powers, cx: &Cx) -> bool {
        use super::expr::{distribution_licensed, signed_zero_null};
        let composes = |t: &Ex| cx.permits_on(Permissive::PowersCompose, t);
        match (self, p) {
            (Law::PowDistribute | Law::ParseInverseOfProduct, Powers::Product { members, n }) => {
                distribution_licensed(members, n, cx)
            }
            (Law::PowCompose, Powers::Rational { base, r, s }) => {
                let rs = r.checked_mul(s);
                composes(base)
                    || cx.certainly_nonneg(base)
                    || if r.is_integer() {
                        s.is_integer()
                            || (r.is_odd_integer()
                                && (r.signum() > 0 || cx.never_infinite(base))
                                && rs.is_some_and(|p| !p.is_integer()))
                    } else {
                        rs.is_some_and(|p| !p.is_integer())
                    }
            }
            // `r` is the index, `s` the exponent (`n` or `-n`).
            (Law::PowRootInverse, Powers::Rational { base, r, .. }) => {
                composes(base) || r.is_odd_integer() || cx.certainly_nonneg(base)
            }
            // `r` is the odd index, `s` the exponent.
            // An integer `s` keeps the root, in both modes: `rootn(t, m)^k` collects with
            // `rootn(t, m)` over one base, where `t^(k/m)` would not.
            (Law::PowOddRootCompose, Powers::Rational { s, .. }) => !s.is_integer(),
            (Law::MulCollect, Powers::Rational { base, r, s }) => {
                // Integers merge totally, in `f64` only while float64 reads them exactly (parity
                // known, |n| <= 2^53): it reads `t^(2^53+1)` as `t^(2^53)`.
                let exact_integers = r.parity().is_some()
                    && s.parity().is_some()
                    && r.checked_add(s).is_some_and(|q| q.parity().is_some());
                cx.permits(Permissive::Agreement)
                    || exact_integers
                    || cx.certainly_nonneg(base)
                    || r.checked_add(s)
                        .is_some_and(|q| !q.is_integer() && !q.is_zero())
            }
            // `site` is the negated site, `s` the exponent.
            (Law::PowNegatedSite, Powers::Site { site, s }) => {
                !(s.is_negative() && s.is_odd_integer())
                    || !super::expr::region_signed_zero(site, cx)
            }
            (Law::MulCollectSymbolic, Powers::Symbolic { base, y }) => {
                cx.permits(Permissive::Agreement)
                    || ((cx.certainly_nonneg(base) || cx.nonconstant_entire(y))
                        && (!cx.f64_semantics() || signed_zero_null(base, true, cx)))
            }
            (law, _) => unreachable!("{} reads no such powers", law.name()),
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
