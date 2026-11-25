//! THE FUNCTION CONSTRUCTOR [`fun`]: the even and odd arms, `atan2`, and the Bessel identities.

use super::*;

/// `atan2(y, x)` built from the product members of its arguments, as float64 reads them: it
/// reads the sign of a zero in both (`atan2(-0, -1) = -pi`, `atan2(+0, -0) = pi`). In the
/// float64 modes a negative coefficient of `y` leaves the call (`atan2(-S, x) = -atan2(S, x)`,
/// exact there, and not in `permissive`, whose one zero gives `atan2(0, -1) = pi`), and each
/// argument keeps its members apart (`mul_undecided`), so neither a sum that vanishes on a
/// region nor a written zero changes the sign of its zero by a product's orientation decision.
/// Every construction of an `atan2` node goes through here: the parse, the canonical rebuild (a
/// rule's right-hand side after substitution) and the pass's rebuild.
pub(crate) fn atan2_of(op: Tok, mut ym: Vec<Ex>, xm: Vec<Ex>, cx: &Cx) -> Ex {
    let negative = |m: &Ex| matches!(m, Ex::Num(r) if r.is_negative());
    if !cx.ieee_zeros() {
        return fun(op, vec![mul(ym, cx), mul(xm, cx)], cx);
    }
    // No sign moves between an argument's members either way: a decided product may trade its
    // coefficient's sign with a sum's orientation, which moves the sign of the sum's zero.
    let build = |m: Vec<Ex>| mul_undecided(m, cx);
    if ym.iter().filter(|m| negative(m)).count() % 2 == 1 {
        if let Some(pos) = ym.iter().position(negative) {
            if let Ex::Num(r) = &ym[pos] {
                if let Some(n) = r.checked_neg() {
                    ym[pos] = Ex::Num(n);
                    let inner = fun(op, vec![build(ym), build(xm)], cx);
                    return mul(vec![Ex::Num(Rat::NEG_ONE), inner], cx);
                }
            }
        }
    }
    fun(op, vec![build(ym), build(xm)], cx)
}

/// Does `e`, an argument of `atan2` or a product member of one, VANISH WITH A SIGN OF ITS OWN in
/// `f64`: does one of its product members spell a zero (a literal zero or the written `-0.0`) or
/// vanish on a region with the sign its form gives it (`region_signed_zero`, the bar an inverse
/// keeps, `inverts_a_signed_zero`)? `atan2` reads that sign (`atan2(+0, -1) = pi`,
/// `atan2(-0, -1) = -pi`), so a search move inside an argument is taken only where neither the
/// part it rewrites nor the part it writes does (`search::site_list`): elsewhere the part's zero
/// set is null, and a move that keeps the value almost everywhere keeps its sign there. `false`
/// outside `f64`: `permissive` reads a value up to the sign of a zero.
pub(crate) fn vanishes_with_a_sign(e: &Ex, cx: &Cx) -> bool {
    let signed = |m: &Ex| {
        matches!(m, Ex::Num(r) if r.is_zero())
            || is_neg_zero(m, cx.view)
            || region_signed_zero(m, cx)
    };
    cx.f64_semantics()
        && match e {
            Ex::Mul(v) => v.iter().any(signed),
            other => signed(other),
        }
}

/// The product members of `e` (itself where it is no product).
pub(crate) fn product_members(e: Ex) -> Vec<Ex> {
    match e {
        Ex::Mul(v) => v,
        other => vec![other],
    }
}

/// The argument an EVEN function reads in place of `a`, or `None` when `a` is already it: the
/// sign-blind representative (`sign_blind_rep`), or else the other orientation of a sum where it
/// wins.
///
/// Beyond the carried sign `sign_blind_rep` drops, an even function erases its argument's
/// orientation (`cos(x - y)` and `cos(y - x)` are one value), which a mixed-sign sum carries in no
/// top-level sign. As an even power's base, the argument takes the orientation the sum's
/// normal-form choice gives a free site (`flipped_orientation_wins`, `Tie::Lex`); Const-bearing and
/// absorbing sums keep their absorption owners, and the flip re-enters `fun` once.
pub(super) fn even_argument(a: &Ex, cx: &Cx) -> Option<Ex> {
    if let Some(sign_free) = cx.sign_blind_rep(a) {
        return Some(sign_free);
    }
    let Ex::Add(ts) = a else { return None };
    if ts.iter().any(Ex::contains_const)
        || ts.iter().any(term_absorbs_negation)
        || ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
        || !flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
    {
        return None;
    }
    let mut flipped = ts
        .iter()
        .map(|t| negate_term(t, cx))
        .collect::<Option<Vec<Ex>>>()?;
    // FILES-BARE gate (same rule as sign_trade_flip).
    if flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
        return None;
    }
    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    Some(Ex::Add(flipped))
}

/// The argument an even function's arm moves `a` to (`even_argument`), where it may move: a
/// strictly cheaper argument, or one the arm does not move again (the class's spelling at an equal
/// price: a mirror orientation, an infinity's sign). So the arms that rebuild the call on the moved
/// argument (`fun`'s even arm, `besselj`'s even orders) re-enter `fun` on an argument whose price
/// falls at every re-entry but the last, and the price is well founded (`rules::ordered_below`):
/// they terminate. `None` where `a` stays as it is.
fn even_move(a: &Ex, cx: &Cx) -> Option<Ex> {
    let moved = even_argument(a, cx)?;
    (complexity(&moved, cx.view) < complexity(a, cx.view) || even_argument(&moved, cx).is_none())
        .then_some(moved)
}

/// `besselj(+-inf, x)` at a literal `x`: 0 at `x >= 0`, NaN below (scipy's cells). And
/// `besselj(nu, 0)` at a literal order, exactly: `J_0(0) = 1`, `J_nu(0) = 0` for `nu > 0` and for
/// a negative integer `nu`, and at a negative non-integer order the pole's one-sided limit (the
/// order's domain is `x >= 0`), the infinity of the sign of `1/Gamma(nu + 1)`:
/// `J_{-1/2}(0) = inf`, `J_{-3/2}(0) = -inf` (scipy's cells). In `f64` the value is the
/// deployed kernel's, whose zero carries a sign of its own at some orders (`J_{1.5}(0)` and
/// `J_{-1}(0)` are `-0.0`, `J_{2.5}(0)` is `0.0`), which an inverse reads as an infinity's sign.
///
/// The laws applied here (`ac::laws::Law`): `num.besselj_literals`.
fn besselj_of_literals_at_zero(args: &[Ex], cx: &Cx) -> Option<Ex> {
    // An infinite order (scipy's cells): 0 at a literal x >= 0, NaN below.
    if let [Ex::PosInf | Ex::NegInf, Ex::Num(x)] = args {
        return Some(if x.is_negative() {
            Ex::NaN
        } else {
            Ex::Num(Rat::ZERO)
        });
    }
    let [Ex::Num(nu), Ex::Num(x)] = args else {
        return None;
    };
    if !x.is_zero() {
        return None;
    }
    if cx.f64_semantics() {
        let y = crate::numeric::kernels::besselj(nu.to_f64(), 0.0);
        return if y == 1.0 {
            Some(Ex::Num(Rat::ONE))
        } else if y == 0.0 && y.is_sign_negative() {
            Some(Ex::Leaf(cx.view.intern(NEG_ZERO)))
        } else if y == 0.0 {
            Some(Ex::Num(Rat::ZERO))
        } else if y == f64::INFINITY {
            Some(Ex::PosInf)
        } else if y == f64::NEG_INFINITY {
            Some(Ex::NegInf)
        } else {
            None
        };
    }
    if nu.is_zero() {
        return Some(Ex::Num(Rat::ONE));
    }
    if !nu.is_negative() {
        return Some(Ex::Num(Rat::ZERO));
    }
    if nu.is_integer() {
        return Some(Ex::Num(Rat::ZERO));
    }
    // Gamma(z) is positive for z > 0 and has the sign (-1)^ceil(-z) at a negative non-integer z.
    let z = nu.checked_add(&Rat::ONE)?;
    let positive = !z.is_negative() || !rat_floor(&z.checked_neg()?, true)?.is_odd_integer();
    Some(if positive { Ex::PosInf } else { Ex::NegInf })
}

/// `besselj(n, t)` at an INTEGER literal order `n`: two identities of `J_n` that hold at every
/// value, NaN and the sign of a zero included -- in float64 too, where the kernel reads both
/// (`numeric::besselj`):
///
///   * the reflection `J_{-n}(t) = (-1)^n J_n(t)`: the canonical order is non-negative
///     (`besselj(-3, x)` is `-besselj(3, x)`);
///   * the parity `J_n(-t) = (-1)^n J_n(t)`: an even order is an even function of `t`, so it reads
///     the argument an even function reads (`even_argument`: `besselj(2, -x)` and
///     `besselj(2, |x|)` are `besselj(2, x)`, `besselj(0, y - x)` is `besselj(0, x - y)`); an odd
///     order is an odd head (`odd_head`), so its sign and its sum's orientation are the class
///     owners', as an odd function's are (`besselj(1, -2 x)` is `-besselj(1, 2 x)`,
///     `-besselj(1, x0 - x1)` is `besselj(1, x1 - x0)`).
///
/// A sign the reflection moves out of a GROUND argument under an odd order stays where it is in
/// `f64` (the value is the ground fold's, from the kernel, and a derived zero would lose the sign
/// the kernel writes).
///
/// The laws applied here (`ac::laws::Law`): `fun.besselj_reflection`.
fn besselj_integer_order(op: Tok, args: &[Ex], cx: &Cx) -> Option<Ex> {
    let Ex::Num(n) = &args[0] else { return None };
    if !n.is_integer() {
        return None;
    }
    let ground = !args[1].has_measure_space(cx.view) && !args[1].contains_wildcard(cx.view);
    let negated = |e: Ex| mul(vec![Ex::Num(Rat::NEG_ONE), e], cx);
    if n.is_negative() {
        if n.is_odd_integer() && ground && cx.f64_semantics() {
            return None;
        }
        let j = fun(op, vec![Ex::Num(n.checked_neg()?), args[1].clone()], cx);
        return Some(if n.is_odd_integer() { negated(j) } else { j });
    }
    if n.is_even_integer() {
        let t = even_move(&args[1], cx)?;
        return Some(fun(op, vec![args[0].clone(), t], cx));
    }
    // An odd order is an odd head (`odd_head`): the class owners place its sign.
    None
}

/// `besseli(n, t)` at an integer literal order `n >= 0`: the parity `I_n(-t) = (-1)^n I_n(t)`.
/// An even order is an even function of `t`, bit for bit in float64 too, so it reads the argument
/// an even function reads (`even_move`: `besseli(2, -x)` and `besseli(0, |x|)` are
/// `besseli(2, x)` and `besseli(0, x)`, `besseli(0, y - x)` is `besseli(0, x - y)`). An odd order
/// moves the sign of a negative literal multiple out (`besseli(1, -2 x)` is `-besseli(1, 2 x)`)
/// where its premise holds: the kernel gives `+0` at a zero of either sign. A ground argument
/// under an odd order keeps its sign, as an odd function's does (`besseli(1, -2)`).
///
/// The laws applied here (`ac::laws::Law`): `fun.besseli_parity`.
fn besseli_parity(op: Tok, args: &[Ex], cx: &Cx) -> Option<Ex> {
    let Ex::Num(n) = &args[0] else { return None };
    if !n.is_integer() || n.is_negative() {
        return None;
    }
    if n.is_even_integer() {
        let t = even_move(&args[1], cx)?;
        return Some(fun(op, vec![args[0].clone(), t], cx));
    }
    if !args[1].has_measure_space(cx.view) && !args[1].contains_wildcard(cx.view) {
        return None;
    }
    let u = cx.carried_negative(&args[1])?;
    if !Law::FunBesseliParity.holds(&args[1], cx) {
        return None;
    }
    let i = fun(op, vec![args[0].clone(), u], cx);
    Some(mul(vec![Ex::Num(Rat::NEG_ONE), i], cx))
}

/// The laws applied here (`ac::laws::Law`): `fun.rootn_index`, `fun.abs_nonnegative`,
/// `fun.half_period`, `num.abs_literal`, `num.exp_one`, `num.rootn_index`, and the inverse pairs'
/// (`Cx::inverse_pair_collapse`).
pub fn fun(op: Tok, args: Vec<Ex>, cx: &Cx) -> Ex {
    // A `pow` call (`pow_spelled`: numpy's `sqrt` of a signed zero) is rebuilt as the spelling
    // it is, which decides again whether it stays a call.
    if args.len() == 2 && cx.view.tok_is(op, "pow") {
        let mut it = args.into_iter();
        let (b, e) = (it.next().expect("base"), it.next().expect("exponent"));
        return pow_spelled(b, e, cx);
    }
    // A ZERO PRODUCT UNDER A HEAD, in `f64` (`mul.zero`). A head that keeps a zero, its sign and
    // NaN (`f(-0) = -0`) is the product itself: `atan(0*x0)` is `0*x0`, as float64 gives it. A
    // head that reads no zero's sign (`f(-0) = f(+0)`) takes the product without the factors that
    // only gave it its sign: `exp(0*x0)` is `exp(0)`. An odd root is a head that keeps a zero, an
    // even one reads no sign.
    let mut args = args;
    if args.first().is_some_and(|z| zero_product(z, cx)) {
        let keeps = |h: &&str| cx.view.tok_is(op, h);
        let rootn_index = (args.len() == 2 && cx.view.tok_is(op, "rootn"))
            .then(|| match &args[1] {
                Ex::Num(n) if n.is_integer() && n.signum() > 0 => Some(n.is_odd_integer()),
                _ => None,
            })
            .flatten();
        let one = args.len() == 1;
        if (one
            && [
                "sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "atanh", "erf", "floor",
                "ceil",
            ]
            .iter()
            .any(keeps))
            || rootn_index == Some(true)
        {
            return args.swap_remove(0);
        }
        if (one
            && ["abs", "exp", "cos", "cosh", "log", "acos", "acosh", "erfc"]
                .iter()
                .any(keeps))
            || rootn_index == Some(false)
        {
            if let Some(u) = unsigned_zero(&args[0], cx) {
                args[0] = u;
            }
        }
    }
    // A literal argument LANDS here (ρ, [`land`]), unless the call is ground.
    let args: Vec<Ex> = if !in_ground(&args, cx) {
        args.into_iter().map(|a| land_operand(a, cx)).collect()
    } else {
        args
    };
    // `exp(1) -> E`. The two spellings denote the SAME constant exactly -- this is a
    // choice of symbol, not an evaluation, so the rules-pass doctrine above does not
    // cover it (nothing is rounded and no exact rule can be preempted; `exp(0) -> 1`
    // IS a literal evaluation and stays with the rules, where it already lives).
    //
    // It is the WRITE half of the invariant `compose_e_power` READS: that helper takes
    // the leaf `E` to mean `E^1`, so the canon must have exactly one spelling for the
    // value or the composition can land on the dearer one. Measured, and not
    // hypothetically: with the composition in and this fold out, `inv(inv(e))` and
    // `x0/exp(-1)` both settled on `exp 1` at mu 10,000 where `np.e` is 8,000, and the
    // re-mine dropped 42 rules that had been carrying exactly that respelling in
    // context (`* exp 1 _0 -> * np.e _0`, `- _0 exp 1 -> - _0 np.e`, ...) -- the mine
    // papering over a missing constructor fold, one context at a time.
    //
    // Unconditional: `np.e` is a universal leaf, parsed and emitted with no config
    // declaration behind it (`convert.rs` maps the token both ways for every table),
    // unlike `exp` itself, which is why the arms that BUILD an `exp` are gated and this
    // one is not.
    if args.len() == 1 && cx.view.tok_is(op, "exp") && matches!(&args[0], Ex::Num(r) if r.is_one())
    {
        return Ex::E;
    }
    // `abs` of a rational literal is EXACT arithmetic (|p/q| = |p|/q, no rounding),
    // so it folds in the constructor -- the fold unification deleted
    // the f64 evaluation path, and abs is the one vocabulary function that is exact
    // on rationals (everything else is transcendental and arrives as mined,
    // symbolically certified rules).
    if args.len() == 1 && cx.view.tok_is(op, "abs") {
        if let Ex::Num(r) = &args[0] {
            if r.is_negative() {
                if let Some(a) = r.checked_neg() {
                    return Ex::Num(a);
                }
            } else {
                return Ex::Num(r.clone());
            }
        }
    }
    // `max` AND `min` ARE LATTICE BAGS, as sums and products are bags: `lattice`.
    if !args.is_empty() && lattice_head(op, cx.view) {
        return lattice(op, args, cx);
    }
    // The step heads of literals (`floor`, `ceil`, `mod`) are no constructor's: the ground
    // fold computes them, `f64` with the deployed kernels at a literal's nearest float
    // (`floor(0.99999999999999999)` is 1 there) and `permissive` by the table.
    // In `permissive` an operation on literals is the table's exact value
    // (`table_literal_fold`):
    // `rootn(0, -2)` and `mod(0, inf)` are NaN, `exp(-inf)` 0, `tanh(inf)` 1, `acosh(1)` 0.
    if cx.table_values() && !args.is_empty() && args.iter().all(table_literal) {
        if let Some(folded) = table_literal_fold(&cx.view.resolve_owned(op), &args) {
            return folded;
        }
    }
    let mut args = args;
    if args.len() == 2
        && cx.ieee_zeros()
        && cx.view.tok_is(op, "atan2")
        && matches!(&args[0], Ex::Mul(v) if v.iter().any(|m| matches!(m, Ex::Num(r) if r.is_negative())))
    {
        let x = args.pop().expect("two arguments");
        let y = args.pop().expect("two arguments");
        return atan2_of(op, product_members(y), product_members(x), cx);
    }
    // AN EVEN FUNCTION EATS THE SIGN OF ITS ARGUMENT: `g(-t) -> g(t)`.
    //
    // ONE arm, replacing thirty mined rules -- the entire
    // `{abs, cos, cosh} x {sin, sinh, tan, tanh, asin, asinh, atan, atanh}` cross product
    // plus `cos neg _0` and `cosh neg _0`. Those thirty are not thirty facts: they are TWO
    // (an odd function propagates a sign, an even one absorbs it) multiplied by fifteen
    // pairs, and the mine enumerated the product because the engine had no name for either
    // factor. `drop_sign` is the missing name.
    //
    // It fixes what no rule can. Three separate holes close at once:
    //   * LITERAL arguments. The rules name a `neg` NODE; `neg(3)` folds to the literal
    //     `-3` before any matcher runs, so every one of the thirty is blind at every
    //     numeric binding (`abs asinh neg 3` shipped as `abs asinh -3`, mu 19,000, where
    //     `asinh 3` is 10,000). This arm reads the canonical SIGN instead, so `neg 3`,
    //     `-3`, `(-1)*x` and `neg(sin x)` are one case.
    //   * DEPTH. The mine caps source patterns at length 4, so `abs sin tanh neg _0` (five
    //     tokens) has no rule at all -- it did not reduce even at a VARIABLE. A recursion
    //     has no depth limit.
    //   * The bare case the rules never covered: `cos (-2)` shipped as `cos -2` at 10,585
    //     against 10,000 for `cos 2`.
    //
    // It composes with `certainly_nonneg` without either knowing about the other:
    // `abs asinh -3` --(here)--> `abs asinh 3` --(nonneg licence)--> `asinh 3`.
    //
    // TOTAL, no licence needed: evenness is a pointwise identity on the whole extension,
    // and `is_even_fun` is a certified table, not a config field. It terminates by the price of
    // its argument (`even_move`).
    if args.len() == 1 && cx.is_even_fun(op) {
        if let Some(arg) = even_move(&args[0], cx) {
            return fun(op, vec![arg], cx);
        }
    }
    // `besselj` OF LITERALS AT ZERO, and OF AN INTEGER ORDER: its reflection and its parity.
    if args.len() == 2 && cx.view.tok_is(op, "besselj") {
        if let Some(e) =
            besselj_of_literals_at_zero(&args, cx).or_else(|| besselj_integer_order(op, &args, cx))
        {
            return e;
        }
    }
    // `besseli` OF AN INTEGER ORDER: its argument's parity.
    if args.len() == 2 && cx.view.tok_is(op, "besseli") {
        if let Some(e) = besseli_parity(op, &args, cx) {
            return e;
        }
    }
    // A HALF-PERIOD SHIFT DROPS OUT: `f(t +- pi)` for f in {sin, cos, tan}.
    //
    // Eleven pattern rules say this (`sin|cos|tan` x `+|-` x both argument orders, plus
    // `f(pi - t)`), and every one of them is blind at a LITERAL argument, for TWO different
    // reasons that no rule can fix:
    //   * sin/cos sit at the `_` sort and go blind at a NEGATIVE literal, because the
    //     constructor factors the bag's common sign: `(-2) - pi` is stored as `neg(2 + pi)`,
    //     and the `-pi` member the pattern needs is gone. Family A's mechanism exactly.
    //   * tan sits at `!`, and `bind_ok` REFUSES ground subjects outright -- a documented,
    //     accepted recall loss on certificate-carrying slots, not a defect. So `tan + 2 np.pi`
    //     cannot bind `2` at all, and tan was blind at EVERY literal, both signs.
    // Measured before the arm: of the 18 `{sin,cos,tan} x {+,-} x {x0, 2, (-2)}` states, cos
    // reduced 6/6, sin 5/6, tan only the 2 variable cases.
    //
    // Each function's law is DERIVED, not taken by symmetry, and the derivation
    // is what picks the output spelling:
    //   * sin is ODD:  sin(t +- pi) = -sin(t) = sin(-t), so the sign may go either side, and
    //     WHICH side is a structural question the first build got wrong. Pushing it into the
    //     argument always looked right from the literal case (`sin -2` prices 10,585 where
    //     `neg sin 2` is 18,000, and the wrapped form would be a WORSE answer than the rule it
    //     replaces). At a VARIABLE the two spellings are mu-EQUAL, and there the choice is
    //     decided by what they COMPOSE with: `test_function_rooted_rules_mint_no_twin` failed
    //     on `neg(sin(x0 + pi))`, which the wrapped form collapses to `sin x0` and the pushed-in
    //     form leaves at `neg sin neg x0` -- an outer `neg` cancels a wrapped sign and cannot
    //     reach one buried in the argument. The condition is therefore whether the sign has
    //     something to CANCEL AGAINST: push in iff the remainder already carries a negative
    //     (`carried_negative`), wrap otherwise. STRUCTURAL, not a mu guard -- mu
    //     guards in constructor arms break idempotence. The first build tested `rest is a Num`
    //     instead, which is the same answer on literals but left `sin(pi - t)` at the
    //     uncollapsible `neg sin neg t` and kept a rule alive to say so.
    //   * cos is EVEN: cos(t +- pi) = -cos(t), and evenness gives no argument trick, so the
    //     `neg` stays outside -- which is what the existing rule already emits.
    //   * tan:         tan(t +- pi) = tan(t), no sign at all.
    // The sign of the pi member is irrelevant to all three, so `+pi` and `-pi` are one case.
    //
    // The judge was asked first, at the plain sort and at literals: all of
    // `sin|cos|tan {+,-} _0 np.pi` and the ground `tan +- 2 np.pi`, `sin + (-2) np.pi` come
    // back CERTIFIED. tan's `!` is a miner-side sort choice, not a judge requirement. The law is
    // `fun.half_period`.
    if args.len() == 1 {
        let shift = match cx.view.resolve_owned(op).as_str() {
            "sin" | "cos" | "tan" => Some(cx.view.tok_is(op, "cos")),
            _ => None,
        };
        if let Some(negate_outside) = shift {
            if let Some(rest) = cx.strip_unit_pi(&args[0]) {
                return if cx.view.tok_is(op, "tan") {
                    fun(op, vec![rest], cx)
                } else if !negate_outside {
                    // sin: push the sign INTO the argument exactly when the argument ABSORBS
                    // it without adding a node. Two ways that happens, and both are needed --
                    // testing only one of them was wrong twice:
                    //   * a LITERAL of either sign absorbs it into its own sign
                    //     (`sin(-2)` prices 10,585 where `neg sin 2` is 18,000);
                    //   * a subtree already CARRYING a negative annihilates it
                    //     (`sin(pi - t)` lands on `sin t`, not the uncollapsible
                    //     `neg sin neg t` that kept a rule alive to say so).
                    // A bare symbolic argument absorbs nothing, so it WRAPS -- mu-equal there,
                    // and only the wrapped sign can be cancelled by an outer `neg`.
                    if let Some(positive) = cx.carried_negative(&rest) {
                        fun(op, vec![positive], cx)
                    } else if matches!(rest, Ex::Num(_)) {
                        fun(op, vec![mul(vec![Ex::Num(Rat::NEG_ONE), rest], cx)], cx)
                    } else {
                        mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![rest], cx)], cx)
                    }
                } else {
                    mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![rest], cx)], cx)
                };
            }
        }
    }
    // AN INVERSE PAIR COLLAPSES: `f(g(t)) -> t` where the licence holds. See
    // `inverse_pair_collapse` -- the table and its three licence groups are the contract
    // judge's own verdicts, taken before the arm was written.
    //
    // The engine missed ONE of these (`exp log abs _0 -> abs _0`). The artifact was
    // carrying **101** identity-shaped inverse-pair rules, which is the same picture other families
    // showed: the mine enumerating a fact the engine has no name for, one member and one
    // LITERAL at a time (`asinh sinh 1`, `asinh sinh 2`, ... `log exp 10`, `acosh cosh np.pi`).
    // A pair is a pair at every argument, so a table costs what one arm costs.
    if args.len() == 1 {
        if let Ex::Fun(inner, iargs) = &args[0] {
            if iargs.len() == 1 {
                if let Some(collapsed) = cx.inverse_pair_collapse(op, *inner, &iargs[0]) {
                    return collapsed;
                }
            }
        }
    }
    // THE LOG OF AN INVERSE (`fun.log_inverse`): `log(1/t) = -log(t)` and
    // `log(abs(1/t)) = -log(abs(t))` where its law holds, at the root's walk in both modes
    // (`release_flat`; in `f64` over a monomial `t`, where nothing reads the sign of a zero); a
    // fully numeric `t` is read for its value first.
    if args.len() == 1 && cx.view.tok_is(op, "log") {
        if let Some((t, in_abs)) = log_inverse_read(&args[0], cx.view) {
            if (t.has_measure_space(cx.view) || t.contains_wildcard(cx.view))
                && Law::FunLogInverse.holds(t, cx)
            {
                let read = if in_abs {
                    fun(cx.view.intern("abs"), vec![t.clone()], cx)
                } else {
                    t.clone()
                };
                return mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![read], cx)], cx);
            }
        }
    }
    // `rootn` index normalization lives HERE, in the constructor, so parse-time and
    // mid-pass rebuilds agree: an index subtree that a numeric fold collapses mid-pass
    // (`cos 0 -> 1`) must normalize in the SAME call, or idempotence breaks. Under the
    // IEEE `rootn` semantics every arm is a sound identity:
    //   rootn(x, 1) = x;  rootn(x, even) = pow(x, 1/even) (both principal, NaN on
    //   negatives);  rootn(x, -n) = pow(rootn(x, n), -1);  rootn(x, 0) and a
    //   provably-non-integer index = NaN everywhere (invalid operation, total).
    // Only binary Fun heads are inspected (rootn is the sole binary function), so the
    // name check stays off the unary hot path. `Const` and compound indices stay
    // symbolic: the evaluators fail closed on them.
    if args.len() == 2 && cx.view.resolve_owned(op) == "rootn" {
        let index = &args[1];
        match index {
            Ex::Num(r) => {
                if r.is_integer() {
                    let n = r.clone();
                    if n.is_zero() {
                        return Ex::NaN;
                    }
                    if n.is_one() && Law::RootnOne.holds(&args[0], cx) {
                        return args.into_iter().next().unwrap();
                    }
                    // A ROOT of `exp` is an `exp`: `rootn(exp a, n) -> exp(a/n)`, which is
                    // `compose_e_power` with the exponent `1/n` -- the SAME call `pow` makes
                    // with `b`, so the two spellings of a power reach the identical
                    // reduction (route independence) and the shape is
                    // unreachable by a mined rule either way: `rootn np.e 2 -> exp 0.5` has
                    // a var-free target, which `candidate_fold_filter` drops at mine time.
                    //
                    // `n` is a nonzero integer here (`n == 0` and `n == 1` returned above),
                    // so `1/n` is a finite nonzero rational and the sign follows through the
                    // reciprocal for free: `rootn(e, -m) = 1/e^(1/m) = exp(-1/m)`. The
                    // first version of this arm handled the LEAF `e` only, and the
                    // census that came out of it found
                    // its sibling still blind on the very next base: `pow(exp(-1), 1/2)` was
                    // shipping as `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is
                    // 11,585. One base is a patch; the family is the principle.
                    if let Some(composed) = n
                        .checked_inv()
                        .and_then(|inv| compose_e_power(&args[0], &Ex::Num(inv), cx))
                    {
                        return composed;
                    }
                    // n >= 2 EVEN: kept as `rootn`. The conversion to `pow(x, 1/n)` used
                    // to live here; it now runs the OTHER way (see `pow`), because mu
                    // prefers the root spelling once a genuine fraction pays for its
                    // denominator -- `rootn x 2` prices 18 against `pow x 1/2` at 19.
                    if n.is_negative() {
                        // checked_neg, not `-n`: a MIN index cannot be negated (release
                        // wraps silently -> the still-negative index would recurse here
                        // forever). Unreachable while the no-MIN Rat invariant holds;
                        // refusing to fold (stay symbolic) is sound either way.
                        if let Some(pos) = n.checked_neg() {
                            let base = args.into_iter().next().unwrap();
                            let inner = fun(op, vec![base, Ex::Num(pos)], cx);
                            return pow(inner, Ex::int(-1), cx);
                        }
                    }
                    // ROOT COMPOSITION: rootn(rootn(x, m), n) -> rootn(x, m*n). Sound at
                    // every parity. m even: the inner root is NaN on x < 0 and m*n is even
                    // too, so both sides are NaN there. m odd with n even: the inner keeps
                    // the sign and the outer even root is NaN on it, and m*n is even.
                    // Both odd: signs pass through on both sides. This composition used to
                    // arrive for free through the pow-of-pow arm (even roots WERE pows);
                    // now that `rootn` survives, it has to be explicit or nested roots stop
                    // composing (`rootn(rootn(x,2),2)` measured 28 against 19 for the
                    // fourth root).
                    if let Ex::Fun(inner_op, inner_args) = &args[0] {
                        if inner_args.len() == 2 && cx.view.tok_is(*inner_op, "rootn") {
                            if let Ex::Num(m) = &inner_args[1] {
                                if m.is_integer() && m.cmp_int(2) != Ordering::Less {
                                    {
                                        if let Some(prod) = m.checked_mul(&n) {
                                            let inner_base = inner_args[0].clone();
                                            return fun(op, vec![inner_base, Ex::Num(prod)], cx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // POWER-THROUGH-ROOT at an EVEN index: `rootn(A^k, n) -> A^(k/n)` under the
                    // SAME licence as the pow-of-pow composition in `pow`. An even `rootn` IS the
                    // principal power `.^(1/n)`, so this IS that composition, merely reached
                    // through the root spelling; sharing the licence is what makes the two routes
                    // agree. Without it the constructor and the rejoin path disagreed:
                    // `rootn(1/cosh(t), 4)` stayed put at construction but reduced to
                    // `cosh(t)^(-1/4)` after a projection round-trip, so the explicit form was not
                    // a fixpoint (64k row 51607).
                    //
                    // The licence also keeps the step sound. A NEGATIVE odd `k` is refused
                    // unless the base is certainly non-negative, because at `A = -inf` the inner
                    // `A^k` is ZERO and the source is then DEFINED while `A^(k/n)` is NaN --
                    // defined -> undefined, which soundness forbids at zero measure tolerance.
                    // `cosh` clears it on the certainly-non-negative disjunct, a bare variable
                    // does not. `s = 1/n` is never an integer here, so that disjunct is vacuous.
                    if n.is_even_integer() {
                        let composed = if let Ex::Pow(inner, ex) = &args[0] {
                            match &**ex {
                                Ex::Num(k) => n.checked_inv().and_then(|s| {
                                    let licensed = Law::PowCompose.holds_powers(
                                        Powers::Rational {
                                            base: inner,
                                            r: k,
                                            s: &s,
                                        },
                                        cx,
                                    );
                                    k.checked_mul(&s).filter(|_| licensed)
                                }),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some(p) = composed {
                            let Ex::Pow(inner, _) = args.into_iter().next().unwrap() else {
                                unreachable!()
                            };
                            return pow(*inner, Ex::Num(p), cx);
                        }
                    }
                    // n >= 3 odd: the genuine signed root. The contract-blessed
                    // inverse composition folds FIRST (SIMPLIFICATION_CONTRACT_v2 s5
                    // Corollaries: pow1_3(pow3 t) = t at EVERY value including nan and
                    // +-inf, hence _-sound): rootn(t^n, n) -> t for the SAME odd n.
                    //
                    // The ODD guard is load-bearing and newly EXPLICIT: at even n the
                    // identity is false -- rootn(t^2, 2) is |t|, not t. It was previously
                    // unreachable at even n only because even indices converted to `pow`
                    // above; with them surviving, the parity has to be tested here.
                    if n.is_odd_integer() {
                        if let Ex::Pow(b, ex) = &args[0] {
                            if let Ex::Num(rr) = &**ex {
                                if *rr == n
                                    && Law::PowRootInverse.holds_powers(
                                        Powers::Rational {
                                            base: b,
                                            r: &n,
                                            s: &n,
                                        },
                                        cx,
                                    )
                                {
                                    let Ex::Pow(inner, _) = args.into_iter().next().unwrap() else {
                                        unreachable!()
                                    };
                                    return *inner;
                                }
                            }
                        }
                    }
                    // EXACT RATIONAL ODD ROOT: a
                    // rational base that is a perfect n-th power folds exactly --
                    // rootn(8,3) = 2, rootn(1,k) = 1, rootn(-8,3) = -2 (checked_root
                    // carries the sign through odd k). This is rational ARITHMETIC,
                    // not transcendental evaluation, so fold unification mandates it
                    // in the constructor (the deleted f64 arm had been carrying it,
                    // same story as abs; the mine cannot cover it -- index/base
                    // literals beyond the alphabet). Inexact bases keep the Fun.
                    if let Ex::Num(r) = &args[0] {
                        if let Some(root) = r.checked_root_integer(&n) {
                            return Ex::Num(root);
                        }
                    }
                    // Otherwise keep the Fun.
                } else {
                    return Ex::NaN; // exact non-integer index: invalid, NaN everywhere
                }
            }
            Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => return Ex::NaN,
            _ => {}
        }
    }
    // Constructor-level abs elimination: `abs t -> t` on a certainly-non-negative operand
    // (`fun.abs_nonnegative`). Both sides share t's domain exactly (abs(nan) = nan,
    // abs(+inf) = +inf) and t's value is >= 0 wherever defined; in `f64` t is not `-0` on a region
    // too, since `|-0|` is `+0` (numpy's `sqrt` keeps `ceil(x0)`'s `-0` on (-1, 0), so
    // `|sqrt(ceil(x0))|` stays). Living HERE rather than in the mined
    // ruleset makes the fold reachable from every entry path and structurally immune to
    // serve-order shadowing: a row diff found the old `inv abs t -> abs inv t`
    // orientation respelling |x^2| into |x^-2| before the exponent-literal abs rules could
    // match, sticking a dead abs in 9/65,536 (64k) + 25/1M outputs. The abs rules this
    // subsumes self-retire at the next mine (mint-vs-mark: the mark now carries the fold).
    // `tok_is` keeps the head dispatch allocation-free on the unary hot path.
    if args.len() == 1 && cx.view.tok_is(op, "abs") && Law::FunAbsNonnegative.holds(&args[0], cx) {
        return args.into_iter().next().unwrap();
    }
    // THE SIGN OF AN ODD HEAD: `f(-u)` and `-f(u)` are one value for an odd head of a negatable
    // argument (`class_base`). A normal-form choice: the cheaper spelling stands, and on an equal
    // price the tie order (`tie_cmp`: fewer signs, then the outer one, so the sign stands outside;
    // then the structure). A product the head stands in decides again over all its members
    // (`sign_place`). A constant argument keeps its sign on its literal (`sin(-2)`).
    let head = Ex::Fun(op, args);
    let pos = match &head {
        Ex::Fun(f, a) => odd_head(cx.view, *f, a).map_or(0, |(p, _)| p),
        _ => 0,
    };
    if let (Some(r), Ex::Fun(_, a)) = (odd_of_literal_multiple(&head, cx.view), &head) {
        if r.is_negative()
            && (a[pos].has_measure_space(cx.view) || a[pos].contains_wildcard(cx.view))
        {
            if let Some(p) = positive_head(&head, cx.view) {
                // The positive head as the constructor builds it: an inverse pair inside it
                // collapses (`sinh(-asinh(x0))` is `-x0`), and the product is that.
                if let Ex::Fun(f, a) = &p {
                    let built = fun(*f, a.clone(), cx);
                    if built != p {
                        return mul(vec![Ex::Num(Rat::NEG_ONE), built], cx);
                    }
                }
                let mut v = vec![Ex::Num(Rat::NEG_ONE), p];
                v.sort_by(|x, y| mul_factor_cmp(x, y, cx.view));
                let outside = Ex::Mul(v);
                let (inside_mu, outside_mu) =
                    (complexity(&head, cx.view), complexity(&outside, cx.view));
                if outside_mu < inside_mu
                    || (outside_mu == inside_mu
                        && tie_cmp(&outside, &head, cx.view) == Ordering::Less)
                {
                    return outside;
                }
            }
        }
    }
    // A head of a sum is a class site: alone it takes the orientation it takes as a product's one
    // factor (`lone_site`). Its flip negates the sum in place, so asking it builds no head.
    if matches!(&head, Ex::Fun(_, a) if matches!(a.get(pos), Some(Ex::Add(_)))) {
        return lone_site(head, cx);
    }
    head
}
