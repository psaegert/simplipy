//! THE PRODUCT CONSTRUCTOR [`mul`]: the coefficient, like bases and their exponents, the powers of
//! e, and the product's assembly.

use super::*;

/// The canonical n-ary multiplication constructor. Gates:
///
/// * NaN factor -> NaN (TOTAL: `nan * t = nan` for every extended-real `t`, including 0 and inf).
/// * Rational coefficients multiply exactly; a coefficient meeting an infinity folds exactly
///   (`r * inf` keeps the sign algebra; `0 * inf = nan`, TOTAL).
/// * Coefficient 0 (no infinity): the product IS 0 wherever every other factor is finite --
///   collapse licensed per-factor by the `!`-certificate; unlicensed factors keep `Mul[0, t]`
///   (`0 * (1/0)` is NaN on full measure, and the certified rule `* 0 !0 -> 0` still gets its
///   chance later).
/// * Like bases merge exponents: SAME-SIGN rational exponents freely -- with the branch-cut
///   licence when either is non-integer (`x^(1/2) * x^(1/2) -> x` is wrong on `x < 0`; requires
///   a certainly-non-negative base OR the merged sum staying non-integer). OPPOSITE-SIGN needs
///   the `$`-licence on the base (`t^2 * t^-1 -> t` fills the NaN at `t = 0`).
/// * `Const` independence as in `add`; bare `Const` factors collapse into ONE `Const`, absorbing
///   a NONZERO rational coefficient (`c * r = c'`, forall-exists; 0 stays outside).
pub fn mul(items: Vec<Ex>, cx: &Cx) -> Ex {
    mul_with(items, cx, true)
}

/// The members of a product under the negative integer power `k`, each raised, for the product
/// that decides them, exactly where `pow` distributes the power over the product they build
/// (`power_bag`): each member as it stands, or, where `pow` first folds a negative coefficient of
/// another magnitude than 1 into a sign-trade site (the one zero's pole of `1/(-c S)` is
/// `+inf`, and `-1/(c S)` would be `-inf` there), the members of that bag; members that collect to
/// one factor give that factor raised. A pure sign stays a member, as a negated sum spells it.
/// `Err` holds their product, built without its decision (`mul_undecided`), where `pow` refuses.
pub(crate) fn raised_members(members: Vec<Ex>, k: &Rat, cx: &Cx) -> Result<Vec<Ex>, Ex> {
    let ex = Ex::Num(k.clone());
    if inverts_a_signed_zero(&members, cx) {
        return Ok(members
            .into_iter()
            .map(|f| pow_member(f, ex.clone(), cx))
            .collect());
    }
    let magnitude = |v: &[Ex]| {
        v.iter()
            .any(|f| matches!(f, Ex::Num(r) if r.is_negative() && *r != Rat::NEG_ONE))
    };
    match mul_undecided(members.clone(), cx) {
        Ex::Mul(v) => match power_bag(&v, k, cx) {
            Some(bag) if bag != v && magnitude(&v) => Ok(bag
                .into_iter()
                .map(|f| pow_member(f, ex.clone(), cx))
                .collect()),
            Some(_) => Ok(members
                .into_iter()
                .map(|f| pow_member(f, ex.clone(), cx))
                .collect()),
            None => Err(Ex::Mul(v)),
        },
        f => Ok(vec![pow_member(f, ex, cx)]),
    }
}

/// A ZERO KEEPS ITS SIGN WHERE IT IS INVERTED, in float64's signed zeros (`Cx::ieee_zeros`): does
/// the product of `v` hold an exact zero, whose sign is then the sign of its other members (`x*0`
/// is -0 wherever x < 0, `-(x - x)` is -0)? A negative integer power distributes over such a
/// product exactly in IEEE 754 arithmetic: the exact zero is +0, so `(0*f)^k` is `0^k * f^k`, that
/// is `inf * f^k`, for every `f`, a zero, an infinity and NaN included. Each member keeps its own
/// power and the zero the pole, so the answer keeps the infinity's sign: `1/(x1*(x2 - x2))` is
/// `inf/x1`, `1/-(x1 - x1)` is `-inf` and `x1/(0/x2)` is `inf*x1*x2`. Asked where the members of
/// an inverted product are still apart (`raised_members`), and by `pow`'s distribution licence
/// (`power_bag`), where a product that holds a zero is inverted whole.
///
/// So with a member that vanishes on a region with the sign its orientation gives it, in `f64`
/// (`region_signed_zero`): `1/(x2*-(|x1| - x1))` is `-1/x2 * 1/(|x1| - x1)`, `-inf/x2` for every
/// `x1 >= 0`, and each member's inverse keeps the orientation it was written in, a negative power
/// whose zero set may hold a region being no sign-trade site (`sign_trade_flip`).
pub(crate) fn inverts_a_signed_zero(v: &[Ex], cx: &Cx) -> bool {
    cx.ieee_zeros()
        && v.iter().any(|f| {
            matches!(f, Ex::Num(r) if r.is_zero())
                || is_neg_zero(f, cx.view)
                || region_signed_zero(f, cx)
        })
}

/// `mul()` without the opposite-sums decision, for a product about to be raised to a negative
/// integer power (`pow`, `canon`, the parse of a divisor). Its classes are decided where its
/// members meet that power: moved positive mass makes the product's zero `-0`, which the
/// inverse turns into the other infinity (`1/((x1 - 3)(3 - x1))` is no `-1/(x1 - 3)^2` at
/// `x1 = 3`), while each member's own inverse keeps its pole (`collect_opposite_sums`).
///
/// Its members over one base merge before a member that is itself a product (a negated sum is
/// `-1` times the sum) is spread into the bag, as their inverses merge in the product that
/// spells them one by one: `1/((1 - x)(1 - x)^(1/3))` is `(1 - x)^(-4/3)` as
/// `(1 - x)^-1 (1 - x)^(-1/3)` is, where spreading first would hand the sign to the coefficient
/// and leave `x - 1` beside `(1 - x)^(1/3)`.
pub(crate) fn mul_undecided(items: Vec<Ex>, cx: &Cx) -> Ex {
    if !items.iter().any(|f| matches!(f, Ex::Mul(_))) {
        return mul_with(items, cx, false);
    }
    let (mut plain, rest): (Vec<Ex>, Vec<Ex>) = items.into_iter().partition(|f| {
        matches!(
            f,
            Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const(_) | Ex::NaN
        )
    });
    let (merged, _) = collect_like_bases(rest, cx);
    plain.extend(merged);
    mul_with(plain, cx, false)
}

fn mul_with(items: Vec<Ex>, cx: &Cx, decide: bool) -> Ex {
    let e = host_partition_sign(mul_assembled(items, cx, decide));
    // A product about to be raised to a negative power (`mul_undecided`) is a divisor, which
    // keeps the form it comes in.
    if decide {
        own_powers_multiplied_out(e, cx)
    } else {
        e
    }
}

/// NO NESTING OF A QUANTITY'S OWN POWERS: a product of a sum with a factor that
/// reappears in it, itself or the base of one of its powers, as a multiplicative factor of a term,
/// is multiplied out over that sum, the reappearing factors only: `x1*(x1 + 1)` is `x1^2 + x1`,
/// `x1*(x1*(0.4*x1 + 2.7) + 1.3)` is `0.4*x1^3 + 2.7*x1^2 + 1.3*x1`, `sin(x1)*(sin(x1) + 1)` is
/// `sin(x1)^2 + sin(x1)`, `x1^(1/2)*(x1 + 1)` is `x1^(3/2) + x1^(1/2)`, while `x2*(x1 + 1)`,
/// `x1*(sin(x1) + cos(x1))` (x1 inside the functions, no factor of a term) and `exp(x1)*(x1 + 1)`
/// keep their factor out, and `x2*x1*(x1 + 1)` is `x2*(x1^2 + x1)`. So nested input reaches the
/// expanded form, and the search takes no move that nests one (`search::explore`, with
/// [`own_power_nestings`]): `x1^2 + 3*x1 + 2` stays, never `x1*(x1 + 3) + 2`. Products and powers of sums
/// stay products (`(x1 + 1)*(x1 + 2)`, `x1*(x1 + 1)*(x1 + 2)`, `x1*(x1 + 1)^2`), and a divisor
/// keeps the form it comes in (`(x1 + 1)/x1^2`): a factor under a negative power is no factor
/// multiplied out. The step is the distributive law with its premises (`search.distribute`,
/// [`crate::ac::search::distribute_product`], read where the sum's zero may be read), strict's in
/// both modes; where they refuse it the product stays. An even root is a power of its radicand (`rootn(x1, 2)` of `x1`).
fn own_powers_multiplied_out(e: Ex, cx: &Cx) -> Ex {
    let Ex::Mul(v) = &e else {
        return e;
    };
    let mut sum_at = None;
    for (i, f) in v.iter().enumerate() {
        if matches!(f, Ex::Add(_)) {
            if sum_at.is_some() {
                return e;
            }
            sum_at = Some(i);
        }
    }
    let Some(si) = sum_at else {
        return e;
    };
    let Ex::Add(ts) = &v[si] else {
        unreachable!("the sum member")
    };
    let (mut own, kept): (Vec<Ex>, Vec<Ex>) = v
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != si)
        .map(|(_, f)| f.clone())
        .partition(|f| {
            own_power_base(f, cx).is_some_and(|b| ts.iter().any(|t| own_factor_of(b, t, cx)))
        });
    if own.is_empty() {
        return e;
    }
    own.push(v[si].clone());
    // The premises are strict's in every mode, so both modes read one form: where strict keeps a
    // product (a pole it would split, an infinity it would spread), permissive keeps it too, and
    // its answer stays no dearer than strict's on strict's yardstick. A zero's sign is no value
    // permissive reads (nor a premise its context certifies), so that premise is not asked.
    if matches!(cx.mode, RuleMode::Permissive) {
        let strict = Cx {
            mode: RuleMode::Strict,
            zero_sign_unread: true,
            ..*cx
        };
        if crate::ac::search::distribute_product(&own, &strict, false, Law::SearchDistribute)
            .is_none()
        {
            return e;
        }
    }
    let Some(spread) = crate::ac::search::distribute_product(&own, cx, true, Law::SearchDistribute)
    else {
        return e;
    };
    let mut items = kept;
    items.push(spread);
    mul(items, cx)
}

/// `mul` without its last step ([`own_powers_multiplied_out`]): the product of a factoring as it
/// writes it (`search::factor_shared`), which the state it is rebuilt into reads.
pub(crate) fn mul_keeping_sums(items: Vec<Ex>, cx: &Cx) -> Ex {
    host_partition_sign(mul_assembled(items, cx, true))
}

/// How many products of `e`, at any depth, nest a quantity's own powers ([`nests_own_powers`]).
pub(crate) fn own_power_nestings(e: &Ex, cx: &Cx) -> usize {
    let own = usize::from(matches!(e, Ex::Mul(_)) && nests_own_powers(e, cx));
    own + match e {
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            v.iter().map(|c| own_power_nestings(c, cx)).sum()
        }
        Ex::Pow(b, x) => own_power_nestings(b, cx) + own_power_nestings(x, cx),
        _ => 0,
    }
}

/// Does `e` (a product, or a sum of terms) hold a product that nests a quantity's own powers: a
/// sum, or a power of one, beside a factor that reappears in it ([`own_powers_multiplied_out`]'s
/// trigger, which leaves a power of a sum a product)? The search
/// takes no move that adds one (`search::explore`), whatever the distributive law's premises say of
/// multiplying it out.
pub(crate) fn nests_own_powers(e: &Ex, cx: &Cx) -> bool {
    // The sum a member holds: itself, or the base of a positive power of it (the search's
    // factoring under a power: `(x1*(x2 + x1))^3` is `x1^3*(x2 + x1)^3`).
    fn sum_of(f: &Ex) -> Option<&Vec<Ex>> {
        match f {
            Ex::Add(ts) => Some(ts),
            Ex::Pow(b, q) => match (&**b, &**q) {
                (Ex::Add(ts), Ex::Num(r)) if !r.is_negative() && !r.is_zero() => Some(ts),
                _ => None,
            },
            _ => None,
        }
    }
    let nests = |p: &Ex| -> bool {
        let Ex::Mul(v) = p else {
            return false;
        };
        let mut sums = v.iter().filter(|f| sum_of(f).is_some());
        let (Some(s), None) = (sums.next(), sums.next()) else {
            return false;
        };
        let ts = sum_of(s).expect("a sum");
        v.iter().any(|f| {
            sum_of(f).is_none()
                && own_power_base(f, cx).is_some_and(|b| ts.iter().any(|t| own_factor_of(b, t, cx)))
        })
    };
    match e {
        Ex::Add(ts) => ts.iter().any(nests),
        p => nests(p),
    }
}

/// The quantity a factor of a product is a power of, for [`own_powers_multiplied_out`]: the base
/// of a positive numeric power or the radicand of a root, the factor itself otherwise; none for a
/// number, a fitted constant, a sum or a power of one, or a divisor (a negative power).
pub(crate) fn own_power_base<'e>(f: &'e Ex, cx: &Cx) -> Option<&'e Ex> {
    match f {
        Ex::Num(_) | Ex::Const(_) | Ex::Pi | Ex::E | Ex::NaN | Ex::PosInf | Ex::NegInf => None,
        Ex::Add(_) => None,
        Ex::Pow(b, q) => match &**q {
            Ex::Num(r) if !r.is_negative() && !r.is_zero() => own_power_base(b, cx),
            _ => None,
        },
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => match &a[1] {
            Ex::Num(n) if !n.is_negative() && !n.is_zero() => own_power_base(&a[0], cx),
            _ => None,
        },
        other => Some(other),
    }
}

/// Does `b`, or a power of it, stand as a multiplicative factor of the term `t`, or of a term of
/// the one sum `t` holds as a factor (that sum is multiplied out with it)?
pub(crate) fn own_factor_of(b: &Ex, t: &Ex, cx: &Cx) -> bool {
    let power_of_b = |m: &Ex| -> bool {
        match m {
            Ex::Pow(base, q) if matches!(**q, Ex::Num(_)) => **base == *b,
            Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => a[0] == *b,
            other => other == b,
        }
    };
    match t {
        Ex::Mul(ms) => {
            if ms.iter().any(power_of_b) {
                return true;
            }
            let mut sums = ms.iter().filter(|m| matches!(m, Ex::Add(_)));
            match (sums.next(), sums.next()) {
                (Some(Ex::Add(us)), None) => us.iter().any(|u| own_factor_of(b, u, cx)),
                _ => false,
            }
        }
        other => power_of_b(other),
    }
}

/// A product's sign, where its coefficient is a partition of several literals (a fold refused),
/// written on the smallest of them: `-1e300*1e300*x1`, not `-1*1e300*1e300*x1`. The assembly
/// decides every placement with the sign on the coefficient slot (`-1` beside the
/// magnitudes), and a re-read factors the signs of the members out again, so the decisions are
/// the same however the sign is written; the written product carries it on a literal, as a
/// ground constant carries its own sign, one sign and no extra member.
fn host_partition_sign(e: Ex) -> Ex {
    let Ex::Mul(mut v) = e else {
        return e;
    };
    // The literals lead (`mul_factor_cmp`); `-1` sorts before the positive magnitudes.
    let n = v.iter().take_while(|f| matches!(f, Ex::Num(_))).count();
    if n >= 2 && matches!(&v[0], Ex::Num(r) if *r == Rat::NEG_ONE) {
        if let Ex::Num(m) = &v[1] {
            if !m.is_negative() && !m.is_zero() {
                let host = m.checked_neg().expect("a Rat negates");
                v.remove(0);
                v[0] = Ex::Num(host);
                return if v.len() == 1 {
                    v.pop().expect("one member")
                } else {
                    Ex::Mul(v)
                };
            }
        }
    }
    Ex::Mul(v)
}

/// The laws applied here (`ac::laws::Law`): `mul.flatten`, `mul.rootn_join`, `num.inf_arith`;
/// [`mul_with`] then `search.distribute` ([`own_powers_multiplied_out`]).
fn mul_assembled(items: Vec<Ex>, cx: &Cx, decide: bool) -> Ex {
    // a fully numeric product is valued whole: its literals combine exactly
    let ground_bag = in_ground(&items, cx);
    let mut nums: Vec<Rat> = Vec::new();
    let mut coeff_overflow: Vec<Ex> = Vec::new();
    let mut inf_sign: Option<bool> = None; // Some(true) = +inf so far, Some(false) = -inf
    let mut has_const = false;
    // The name the product's one fitted constant is written with (`ConstName::merge`).
    let mut const_name = ConstName::ANON;
    let mut factors: Vec<Ex> = Vec::new();
    // In `f64` a written negative zero is a zero with its sign (`zero_sign`).
    let signed_zeros = f64_zero_signs(cx);
    let mut neg_zeros = false;
    let mut stack: Vec<Ex> = items;
    stack.reverse();
    loop {
        while let Some(e) = stack.pop() {
            match e {
                Ex::NaN => return Ex::NaN,
                Ex::Mul(v) => stack.extend(v.into_iter().rev()),
                Ex::Num(r) => nums.push(r),
                Ex::PosInf => inf_sign = Some(inf_sign.unwrap_or(true)),
                Ex::NegInf => inf_sign = Some(!inf_sign.unwrap_or(true)),
                Ex::Const(n) => {
                    has_const = true;
                    const_name = const_name.merge(n, cx.view);
                }
                e if signed_zeros && is_neg_zero(&e, cx.view) => {
                    nums.push(Rat::ZERO);
                    neg_zeros = !neg_zeros;
                }
                other => factors.push(other),
            }
        }
        // ODD-ROOT JOIN: rootn factors sharing the
        // same odd literal index merge their arguments -- rootn(a,n) * rootn(b,n) =
        // rootn(a*b, n) is a TOTAL identity (the real odd root is completely
        // multiplicative over the extended reals, including +-inf and nan), so no
        // certificate is needed. This is the sound form of the exponent-collection
        // behavior rootn deliberately opted out of by not being a power (the
        // principal rational power differs on negative bases); the joining rule is
        // unminable (7-token source > the length-4 mine). The joined `fun` result
        // re-enters the dispatch (it may collapse further: rootn(x^2,3)*rootn(x,3)
        // -> rootn(x^3,3) -> x, or fold exactly via the odd-root arm); each join strictly
        // reduces total node count, so the loop terminates by well-foundedness.
        // One scan collects EVERY odd index's occurrence list (a single-latch scan
        // missed joins whose index differs from the first rootn seen); the loop
        // joins one group per round and re-scans, so the fixpoint -- every group
        // reduced to one factor -- is reached regardless of group order (products
        // are AC, so the per-group result is order-independent).
        // A product about to be inverted keeps its roots apart: the inverse reaches each of
        // them, and inverted roots are never joined (`mul_undecided`).
        if !decide {
            break;
        }
        let mut by_index: Vec<(Rat, Vec<usize>)> = Vec::new();
        for (i, f) in factors.iter().enumerate() {
            if let Ex::Fun(op, fargs) = f {
                if fargs.len() == 2 && cx.view.tok_is(*op, "rootn") {
                    if let Ex::Num(r) = &fargs[1] {
                        if r.is_odd_integer() && r.cmp_int(3) != Ordering::Less {
                            match by_index.iter_mut().find(|(gn, _)| gn == r) {
                                Some((_, idxs)) => idxs.push(i),
                                None => by_index.push((r.clone(), vec![i])),
                            }
                        }
                    }
                }
            }
        }
        match by_index.into_iter().find(|(_, idxs)| idxs.len() >= 2) {
            Some((n, idxs)) => {
                let mut args = Vec::with_capacity(idxs.len());
                let mut op_tok = None;
                for i in idxs.into_iter().rev() {
                    let Ex::Fun(op, mut fargs) = factors.remove(i) else {
                        unreachable!()
                    };
                    op_tok = Some(op);
                    args.push(fargs.swap_remove(0));
                }
                stack.push(fun(op_tok.unwrap(), vec![mul(args, cx), Ex::Num(n)], cx));
            }
            _ => break,
        }
    }

    // DETERMINISTIC coefficient accumulation: folding rational
    // factors in ARRIVAL order made the overflow partition order-dependent -- which
    // pairs merged before an i128 overflow depended on the input spelling, yielding two
    // sound-but-different canonical states for one multiset. Sorting first (cmp_exact
    // is exact at every magnitude) makes the merge sequence a function of the
    // MULTISET: permutation invariance by construction. On overflow the accumulated
    // value is emitted as a factor and accumulation restarts.
    //
    // SIGN-FACTORED: the accumulation runs on ABSOLUTE values and the
    // net sign lands on the final coefficient, which `sign_place` owns. Accumulating
    // SIGNED rationals let the sign ride whichever partial the sorted fold left it in,
    // so with an overflow partition the sign HOST was a function of the arrival
    // spelling: `-N * P` hosted it on the `-N` partial while the re-parse of its own
    // rendering hosted it on the coefficient slot -- two canonical-looking states for
    // one value (extreme-lane fuzz rows; the 7-token falsifier in test_ac_core). The
    // sign of a product IS a bag-level attribute: |a*b| = |a|*|b| and the partition of
    // the UNSIGNED multiset is spelling-invariant. (`checked_neg` on a Rat is total
    // outside i128::MIN, which `Rat::new`'s normalization never stores.)
    let mut net_neg = false;
    for r in nums.iter_mut() {
        if r.is_negative() {
            match r.checked_neg() {
                Some(a) => {
                    net_neg = !net_neg;
                    *r = a;
                }
                None => debug_assert!(false, "Rat holding i128::MIN reached mul()"),
            }
        }
    }
    nums.sort_unstable_by(|a, b| a.cmp_exact(b));
    // CANONICAL coefficient partition (`Rat::partition_product`): the magnitudes
    // multiplied as far as the products stay representable, as a fixed point a re-read
    // reproduces; the sign stays on the coefficient slot below.
    let mut coeff = Rat::ONE;
    let mut nums = literal_partition_product(nums, ground_bag, cx);
    if nums.len() == 1 {
        coeff = nums.pop().unwrap();
    } else {
        coeff_overflow.extend(nums.into_iter().map(Ex::Num));
    }
    if net_neg {
        match coeff.checked_neg() {
            Some(nc) => coeff = nc,
            None => debug_assert!(false, "positive Rat negation overflowed"),
        }
    }

    // Coefficient x infinity folds exactly and totally.
    if let Some(sign) = inf_sign {
        if coeff.is_zero() {
            return Ex::NaN; // 0 * inf = nan, and nan * rest = nan: TOTAL
        }
        let sign = if coeff.is_negative() { !sign } else { sign };
        coeff = Rat::ONE;
        // The members a refused coefficient partition kept are positive finite magnitudes
        // (the sign is on `coeff`): each is absorbed like the coefficient.
        coeff_overflow.clear();
        inf_sign = Some(sign);
        if has_const {
            // inf * c: c = 0 is reachable (0 * inf = nan), so the constant does NOT absorb the
            // infinity; keep both factors.
            factors.push(Ex::Const(const_name));
            has_const = false;
        }
    }

    // OPPOSITE SUMS: factors whose sums are each other's negation -- `(9.6 - x) * (x - 9.6)`,
    // and odd functions of such sums, `tanh(x1 - x2) * tanh(x2 - x1)` -- are one base up to a
    // sign, so their classes {B, -B} are decided on the exponent totals of the whole product and
    // collected again: `-(x - 9.6)^2`, and under the usual exponent licence
    // `(x - 9.6) / (9.6 - x)` to `-1` (`collect_opposite_sums`). Without this
    // the pair survived wherever it met in one bag (`sign_place` skips a flip that collides
    // with another base) but collected wherever the parse built a smaller product first, in
    // which the site traded alone: a state and the parse of its own print differed (srbf
    // ground truth 4174, in all three modes). A flip's sign goes to the bag's sign host: the
    // coefficient, a bare infinity, or a bare Const that absorbs it. A zero coefficient (which
    // eats every sign) and a coefficient partition keep their own owners. Only the scan runs on
    // every bag.
    let opposite = (decide
        && !coeff.is_zero()
        && (coeff_overflow.is_empty() || has_const)
        && holds_opposite_sums(&factors, cx))
    .then(|| factors.clone());
    // A zero coefficient keeps only the factors it cannot drop (below), read after the
    // collection and before it.
    let zero_members = (inf_sign.is_none() && coeff.is_zero() && coeff_overflow.is_empty())
        .then(|| factors.clone());

    let (mut out, spliced) = collect_like_bases(factors, cx);
    if has_const {
        // Const absorbs every nonzero rational factor (c * r = c'), the members a refused
        // coefficient partition kept included.
        coeff_overflow.clear();
    }
    out.extend(coeff_overflow);
    // A spliced factor collects against the rest on a new pass, and so do two odd roots of one
    // index the collection left (`cbrt(x)^2 / cbrt(x)` beside `cbrt(y)`), which the join above
    // only takes on one.
    if spliced || (decide && holds_joinable_roots(&out, cx)) {
        if !coeff.is_one() {
            out.push(Ex::Num(coeff));
        }
        if has_const {
            out.push(Ex::Const(const_name));
        }
        if let Some(sign) = inf_sign {
            out.push(if sign { Ex::PosInf } else { Ex::NegInf });
        }
        return mul_with(out, cx, decide);
    }
    if let Some(raw) = opposite {
        let host = match inf_sign {
            Some(sign) => SignHost::Inf(sign),
            None if has_const => SignHost::Const(const_name),
            None => SignHost::Coeff(coeff),
        };
        return collect_opposite_sums(host, raw, out, cx);
    }

    // Coefficient 0 (no infinity): collapse where licensed (see the gate doc above). A bare
    // Const vanishes here (0 * c = 0 -- Const is certainly finite), except in `f64`, where its
    // sign is the zero's (`0*c` is -0 for c < 0, which an inverse reads).
    if inf_sign.is_none() && coeff.is_zero() {
        // THE ZERO'S SIGN, in `f64`: float64's product of a zero is `-0` where its members' signs
        // are negative, and an inverse reads it, so the zero carries the literals' sign (`-0` is
        // the written negative zero) and keeps every factor whose sign may vary (`mul.zero`).
        let zero_sign = signed_zeros.then_some(net_neg ^ neg_zeros);
        // Each LICENSED factor drops individually (`0 * f == 0` wherever `f` is finite --
        // the same finite-a.e. licence as the full collapse, applied per-factor). Keeping
        // licensed factors would make the kept-zero form depend on ASSOCIATION order
        // (`(0 * x) * (1/D)` collapses the inner pair, `0 * (x * (1/D))` would not) and
        // break serialization stability; only the unlicensed factors stay with the 0.
        let mut v: Vec<Ex> = out
            .into_iter()
            .filter(|f| !Law::MulZero.holds(f, cx))
            .collect();
        if signed_zeros && has_const {
            v.push(Ex::Const(const_name));
        }
        if v.is_empty() {
            return signed_zero(zero_sign, cx);
        }
        // Assemble through the sign-placement owner, not raw: the orbit files ONE orientation
        // for every trade-site factor where the raw assembly froze whichever mirror arrived
        // (direct build vs parse-of-own-rendering shipped two states; row-120 law), the zero
        // taking the sign a flip moves (in `f64`) or eating it.
        let kept = sign_place_zero(zero_sign, v, cx);
        // The licensed factors drop before the collection too: a collection can merge one into
        // a factor the zero cannot drop (`0 * x^4 * sqrt(x)` collects to `0 * x^(9/2)`, while
        // dropping first keeps `0 * sqrt(x)`), as it can cancel what dropping first keeps
        // (`0 * x / x`). Both read the product's members alone: a normal-form choice, the cheaper
        // kept, the tie order on an equal price (`tie_cmp`).
        if let Some(members) = zero_members {
            let w: Vec<Ex> = members
                .into_iter()
                .filter(|f| !Law::MulZero.holds(f, cx))
                .collect();
            let (w, spliced) = collect_like_bases(w, cx);
            if !spliced {
                let mut w: Vec<Ex> = w
                    .into_iter()
                    .filter(|f| !Law::MulZero.holds(f, cx))
                    .collect();
                if signed_zeros && has_const {
                    w.push(Ex::Const(const_name));
                }
                if w.is_empty() {
                    return signed_zero(zero_sign, cx);
                }
                let dropped = sign_place_zero(zero_sign, w, cx);
                let (md, mk) = (complexity(&dropped, cx.view), complexity(&kept, cx.view));
                if md < mk || (md == mk && tie_cmp(&dropped, &kept, cx.view) == Ordering::Less) {
                    return dropped;
                }
            }
        }
        return kept;
    }

    if has_const {
        // Bare Const absorbs the (nonzero) rational coefficient: c * r = c'.
        coeff = Rat::ONE;
        out.push(Ex::Const(const_name));
    }
    if let Some(sign) = inf_sign {
        out.push(if sign { Ex::PosInf } else { Ex::NegInf });
    }
    if !decide {
        // No sign moves between the members either: a sign a site took would reach the
        // inverse as a flipped pole (`mul_undecided`).
        if !coeff.is_one() {
            out.push(Ex::Num(coeff));
        }
        return match out.len() {
            0 => Ex::Num(Rat::ONE),
            1 => out.pop().expect("one member"),
            _ => {
                out.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
                Ex::Mul(out)
            }
        };
    }
    finish_product(coeff, out, cx)
}

/// Like-base exponent collection, `mul()`'s step after the coefficient: the factors with one
/// base (and one symbolic exponent) merge their exponents, and each result is rebuilt through
/// [`pow`]; then the powers of e left over distinct bases merge into one (`merge_e_powers`).
/// Returns the rebuilt factors and whether one of them must re-enter the constructor
/// (`spliced`: a product whose members are in the list, or a factor of another kind, and the
/// caller re-runs the constructor on it).
pub(super) fn collect_like_bases(factors: Vec<Ex>, cx: &Cx) -> (Vec<Ex>, bool) {
    // Like-base exponent collection. The BRANCH-CUT licence gates every single merge step,
    // including same-sign ones: merging exponents `a` and `b` into `a + b` is sound iff both
    // are integers (TOTAL up to null poles), OR the base is certainly non-negative, OR the sum
    // stays a nonzero non-integer (the `base < 0` region is NaN on both sides). Counterexample
    // that forces the same-sign gate: `x^(1/2) * x^(1/2) -> x` fabricates values on `x < 0`.
    struct FBucket {
        base: Ex,
        sym: Option<Ex>, // Some(e): occurrences of base^e; None: rational exponent pool
        /// ALL rational exponents for this (base, sym), raw; the licence-gated fold
        /// happens at rebuild in SORTED order (determinism).
        exps: Vec<Rat>,
    }
    let mut buckets: Vec<FBucket> = Vec::new();
    let mut opaque: Vec<Ex> = Vec::new(); // Const-containing and merge-refused factors, verbatim
    for f in factors {
        let (base, fe) = factor_split(f, cx);
        let (sym, r) = match fe {
            FactorExp::Rat(r) => (None, r),
            FactorExp::Sym(s) => (Some(s), Rat::ONE),
        };
        if base.contains_const() || sym.as_ref().is_some_and(Ex::contains_const) {
            // Independent constants: keep verbatim, never merged with anything.
            opaque.push(rebuild_factor(base, sym, r, cx));
            continue;
        }
        match buckets.iter_mut().find(|b| b.base == base && b.sym == sym) {
            Some(b) => b.exps.push(r),
            None => buckets.push(FBucket {
                base,
                sym,
                exps: vec![r],
            }),
        }
    }

    // AN ODD ROOT'S POWERS MERGE WITH ITS RADICAND'S (`pow.odd_root_compose` in a product):
    // `rootn(t, k)^m` is `t^(m/k)` wherever t >= 0, while `rootn` is the real root on t < 0
    // (`rootn(-8, 3)` is -2, `(-8)^(1/3)` has none). So beside a power of `t` the root's bucket
    // joins `t`'s, its exponents scaled by `1/k` (a radicand `t^p` by `p/k`): in `f64` only where
    // the product is confined to t >= 0, a power of `t` with a non-integer exponent beside it
    // having no value for t < 0, so the product has none there either and the two readings
    // differ by rounding alone (`t^(3/2)/rootn(t, 3)^2` is `t^(5/6)`; an even root is such a
    // power, `rootn(t, 4)*rootn(t, 3)` is `t^(7/12)`); in `permissive` always, as powers compose
    // there ignoring the sign (`t*rootn(t, 3)` is `t^(4/3)`, `rootn(t, 3)*rootn(t, 5)` is
    // `t^(8/15)`). A root no other power of `t` meets keeps its spelling (`rootn(t, 3)^2`), and
    // in `f64` so does one beside integer powers only (`t*rootn(t, 3)` has a value for t < 0).
    // The collection's licence then folds the merged pool as any other.
    let mut i = 0;
    while i < buckets.len() {
        let Some((rad, scale)) = buckets[i]
            .sym
            .is_none()
            .then(|| odd_root_of_a_power(&buckets[i].base, cx))
            .flatten()
        else {
            i += 1;
            continue;
        };
        let radicand_at = |b: &FBucket| b.sym.is_none() && b.base == rad;
        let target = buckets.iter().position(radicand_at);
        // Where the state confines the product to t >= 0 (`confine.sibling`: an enclosing node is
        // NaN wherever t < 0), the root composes as in `permissive`: on t >= 0 both readings agree.
        let composes = if cx.permits(Permissive::PowersCompose) || cx.confined_nonneg(&rad) {
            target.is_some()
                || buckets.iter().enumerate().any(|(j, b)| {
                    j != i
                        && b.sym.is_none()
                        && odd_root_of_a_power(&b.base, cx).is_some_and(|(r2, _)| r2 == rad)
                })
        } else {
            target.is_some_and(|t| buckets[t].exps.iter().any(|r| !r.is_integer()))
        };
        let scaled: Option<Vec<Rat>> = buckets[i]
            .exps
            .iter()
            .map(|m| m.checked_mul(&scale))
            .collect();
        let (true, Some(scaled)) = (composes, scaled) else {
            i += 1;
            continue;
        };
        // Only where its powers merge: the radicand's pool folds into fewer factors with the
        // root's exponents than the two buckets give apart (`t^(2/3)*rootn(t, 3)` keeps the root
        // in `f64`, where `t^(2/3 + 1/3)` would give `t` a value on t < 0).
        let apart = fold_exponents(&buckets[i].base, None, buckets[i].exps.clone(), cx).len()
            + target.map_or(0, |t| {
                fold_exponents(&rad, None, buckets[t].exps.clone(), cx).len()
            });
        let mut pooled = target.map_or_else(Vec::new, |t| buckets[t].exps.clone());
        pooled.extend(scaled.iter().cloned());
        if target.is_some() && fold_exponents(&rad, None, pooled, cx).len() >= apart {
            i += 1;
            continue;
        }
        buckets.remove(i);
        match buckets.iter_mut().find(|b| radicand_at(b)) {
            Some(b) => b.exps.extend(scaled),
            None => buckets.push(FBucket {
                base: rad,
                sym: None,
                exps: scaled,
            }),
        }
    }

    // A rebuilt factor can itself be a `Mul` (pow distributes integer exponents over Mul
    // bases; the Add sign-flip wraps odd powers in `Mul[-1, ...]`): SPLICE those and re-run the
    // constructor at the end, exactly as `add` does -- the spliced factors must collect against
    // the other buckets, and the Flat invariant must hold.
    let mut out: Vec<Ex> = opaque;
    let mut spliced = false;
    let push_factor = |out: &mut Vec<Ex>, spliced: &mut bool, f: Ex| match f {
        Ex::Mul(v) => {
            out.extend(v);
            *spliced = true;
        }
        f => out.push(f),
    };
    // A rebuilt factor over another base (`cbrt(x)^3` is `x`) collects against the bucket of
    // that base on one more pass.
    let mut rebased = false;
    for b in buckets {
        let FBucket { base, sym, exps } = b;
        for r in fold_exponents(&base, sym.as_ref(), exps, cx) {
            let f = rebuild_factor(base.clone(), sym.clone(), r, cx);
            if !matches!(f, Ex::Mul(_)) && factor_split(f.clone(), cx).0 != base {
                rebased = true;
            }
            push_factor(&mut out, &mut spliced, f);
        }
    }
    if rebased {
        let (out, again) = collect_like_bases(out, cx);
        return (out, spliced || again);
    }
    let merged = merge_e_powers(&mut out, cx);
    (out, spliced || merged)
}

/// THE LIKE-BASE FOLD of one bucket (`collect_like_bases`): the exponents of `base` (and its
/// symbolic exponent `sym`, if any) after every merge the branch-cut licence admits, in the order
/// they are rebuilt. A DETERMINISTIC licence-gated fold (`Rat::partition_with`): two exponents
/// merge when the licence admits the pair and the sum is a number, until no two merge -- a
/// function of the exponent MULTISET that re-reads to itself in any order (a greedy fold left
/// pieces the next pass merged: `x^1e-306*x^1000*x^1001`). Opposite signs share one pool only on
/// a finite-nonzero base (`x^a * x^-a = 1` needs the licence); a zero sum drops the factor.
fn fold_exponents(base: &Ex, sym: Option<&Ex>, exps: Vec<Rat>, cx: &Cx) -> Vec<Rat> {
    let sym_ok = match sym {
        None => true,
        Some(y) => cx.sym_merge_licensed(base, y),
    };
    let merge = |a: &Rat, b: &Rat| -> Option<Rat> {
        if sym_ok && Law::MulCollect.holds_powers(Powers::Rational { base, r: a, s: b }, cx) {
            a.checked_add(b)
        } else {
            None
        }
    };
    let mixed = exps.iter().any(|r| r.is_negative()) && exps.iter().any(|r| !r.is_negative());
    let pools: Vec<Vec<Rat>> = if mixed && sym_ok && Law::MulCancel.holds(base, cx) {
        vec![exps]
    } else {
        let (negs, poss): (Vec<Rat>, Vec<Rat>) = exps.into_iter().partition(|r| r.is_negative());
        vec![poss, negs]
    };
    let mut out = Vec::new();
    for pool in pools {
        let mut parts = Rat::partition_with(pool, merge, Rat::is_zero);
        // THE POWERS LEFT MERGE AS A WHOLE where no two of them did: the licence reads the
        // merged exponent, which two members can make an integer their whole does not, so
        // `x^(1/2)*x^(1/2)*x^(1/2)` is `x^(3/2)` while `x^(1/2)*x^(1/2)` keeps its halves in
        // f64 (`x` has a value where they have none). The first k of the parts merge, for the
        // largest k of three or more the licence admits, until none does.
        while parts.len() >= 3 {
            let merged = (3..=parts.len()).rev().find_map(|k| {
                let rest = parts[1..k]
                    .iter()
                    .try_fold(Rat::ZERO, |acc, r| acc.checked_add(r))?;
                merge(&parts[0], &rest).map(|total| (k, total))
            });
            let Some((k, total)) = merged else {
                break;
            };
            parts.drain(..k);
            if !total.is_zero() {
                parts.push(total);
            }
            parts.sort_by(Rat::cmp_exact);
        }
        out.extend(parts);
    }
    out
}

/// An odd root of a power, `rootn(t^p, k)` for an odd index k >= 3 (`p` 1 for a radicand that is no
/// rational power), as `(t, p/k)`: the base and exponent the like-base collection reads its powers
/// with where it may (`collect_like_bases`).
fn odd_root_of_a_power(f: &Ex, cx: &Cx) -> Option<(Ex, Rat)> {
    let Ex::Fun(op, args) = f else {
        return None;
    };
    if args.len() != 2 || !cx.view.tok_is(*op, "rootn") {
        return None;
    }
    let Ex::Num(k) = &args[1] else {
        return None;
    };
    if !k.is_odd_integer() || k.cmp_int(3) == Ordering::Less {
        return None;
    }
    let inv = k.checked_inv()?;
    match as_rational_power(&args[0], cx) {
        Some((t, p)) => Some((t.clone(), p.checked_mul(&inv)?)),
        None => Some((args[0].clone(), inv)),
    }
}

/// The exponent of `f` read as a power of Euler's number: `exp(a)` is `e^a`, the leaf `E` is
/// `e^1`, and a rational power of either is `e^(a r)` -- a `Pow` with a literal exponent, or an
/// even root, which `factor_split` reads as a rational power too. `None` for anything else,
/// a symbolic power included: `(e^a)^b` is `e^(a b)` only where `a b` is no `0 * inf`.
fn e_exponent(f: &Ex, cx: &Cx) -> Option<Ex> {
    let (base, r) = as_rational_power(f, cx).unwrap_or((f, Rat::ONE));
    let a = match base {
        Ex::E => return Some(Ex::Num(r)),
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "exp") => &args[0],
        _ => return None,
    };
    Some(if r.is_one() {
        a.clone()
    } else {
        mul(vec![Ex::Num(r), a.clone()], cx)
    })
}

/// POWERS OF e COLLECT BY THEIR EXPONENTS: `exp(a) exp(b) = exp(a + b)`. `exp` is a power of e,
/// so this is the like-base collection that merges `x^a x^b` into `x^(a + b)`, over the one base
/// e every power of e shares, whatever `exp` spells its exponent with: `collect_like_bases` merges
/// the powers over one `exp(a)` as over any base, and this merges what remains over distinct
/// ones. The base is positive, finite and nonzero, so every licence of that collection holds for
/// it (the branch cut and the finite-nonzero licence of a cancelling merge); the exponents sum
/// through [`add`], whose own licences decide what cancels: `exp(u) exp(-u)` is `exp(0) = 1` where
/// `u` is finite almost everywhere, as `exp(u)/exp(u)` is.
///
/// TOTAL on the extended real line, so no mode needs a licence of its own: at `a = +inf`,
/// `b = -inf` both sides are NaN (`inf * 0`, and `exp(inf - inf)`), and at one infinite exponent
/// both are that exponent's limit. Under the f64 evaluator the two sides differ where an `exp`
/// overflows or underflows (`exp(800) exp(-800)` is `inf * 0 = NaN`, while `exp(0)` is 1), which
/// is f64's range event (`numeric::RangeKind`): there a rewrite may give the true value, and the
/// merge gives it.
///
/// Rebuilt through [`pow`] over the base e, so the merged exponent takes the spelling `pow` gives
/// any power of e (`compose_e_power`): `exp(x + 1)`, or `E` and `1` at the exponents 1 and 0. A
/// single power of e stays as it stands. Returns whether the merged factor must re-enter the
/// constructor: when it is no power of e (a number, a product, a factor over another base).
///
/// The laws applied here (`ac::laws::Law`): `mul.exp_merge`.
fn merge_e_powers(out: &mut Vec<Ex>, cx: &Cx) -> bool {
    let at: Vec<(usize, Ex)> = out
        .iter()
        .enumerate()
        .filter_map(|(i, f)| e_exponent(f, cx).map(|x| (i, x)))
        .collect();
    if at.len() < 2 {
        return false;
    }
    let mut exponents = Vec::with_capacity(at.len());
    for (i, x) in at.into_iter().rev() {
        out.remove(i);
        exponents.push(x);
    }
    let merged = pow(Ex::E, add(exponents, cx), cx);
    let again = e_exponent(&merged, cx).is_none();
    out.push(merged);
    again
}

/// `mul()`'s last step: the sign owners, then the assembly, for a nonzero coefficient and a
/// collected factor list (bare `Const` and infinities included). `collect_opposite_sums` prices
/// its candidates through it, so a priced product is exactly the product built.
///
/// The laws applied here (`ac::laws::Law`): `mul.infinity_sign`.
pub(super) fn finish_product(mut coeff: Rat, mut out: Vec<Ex>, cx: &Cx) -> Ex {
    // The coefficient LANDS beside the other factors here (permissive's ρ, [`land`]); a
    // product of literals alone stays exact, and a product of ground factors.
    if !out.is_empty() && !in_ground(&out, cx) {
        coeff = land(coeff, cx);
    }
    // CANONICAL SIGN ORIENTATION for `Mul[-1, Add[S]]`:
    // the orientation of a sum class {S, -S} is owned by `flipped_orientation_wins`, but
    // two assembly paths could ship the LOSING orientation wrapped in a -1 coefficient:
    // (a) infinity-bearing sums bypass `primitive_sum` entirely (the one-infinity exit
    // in `add()`), so nothing ever re-orients them; (b) this constructor bagged
    // `Num(-1) x Add[S]` verbatim, whatever the orientation. The tagged renderer's
    // display redistribution (`emit_tagged`'s Mul arm) prints the factored form and the
    // flipped sum IDENTICALLY, so parse rebuilds the DISTRIBUTED state and the two
    // spellings were distinct derivation-reachable canonical states sharing one
    // serialization -- invisible to the stability assert (it compares spellings, which
    // agree). Observable: a rule with a `Mul[-1, F(..)]`-shaped LHS fires on the
    // distributed member but cannot see inside the factored node, so one-call fixpoints
    // differed from re-entry fixpoints (15/200k fuzz rows, every one
    // infinity-bearing). The fix has two clauses, one per abdicating owner:
    //
    // * BARE-INFINITY sums (the class `primitive_sum` never sees -- `add`'s
    //   one-infinity exit returns its flat bag with no orientation enforcement): the
    //   sign lives IN the sum, for ANY negative coefficient. A negative rational
    //   coefficient never coexists with a bare-inf Add factor: `Mul[-c, Add[S]]` and
    //   `Mul[c, Add[flip S]]` are one value, and the projections conflate them (the
    //   infix renderer spells the coefficient sign as a leading `-(..)`, which
    //   re-parses as the sum flip -- fuzz row 10985). The distributed positive-coeff
    //   spelling is the one parse builds and the one whose members are visible rule
    //   sites (the sign-pull family fires on a `Mul[-1, F(..)]` MEMBER, never through
    //   a wrapper); `add(flipped)` re-enters the same inf-exit and returns flat, and
    //   the negated coefficient is positive, so no bounce is possible.
    // * Everything else, for the lone `-1 x Add` shape: consult the SAME orientation
    //   decision `primitive_sum` uses -- flip exactly when the flipped orientation
    //   wins. Antisymmetry gives termination (the flipped bag's own check is false),
    //   and `add`'s primitive-sum tail call always passes the WINNING orientation, so
    //   this arm is a no-op on it. General negative coefficients on non-inf sums stay:
    //   they ARE the content form (`u` extraction).
    //
    // A `flipped_orientation_wins`/`negate_term`/negation overflow refusal keeps the
    // factored form, whose `neg <add>` / literal-coefficient display is injective.
    if coeff.is_negative() {
        let inf_add = out.iter().position(
            |f| matches!(f, Ex::Add(v) if v.iter().any(|t| matches!(t, Ex::PosInf | Ex::NegInf))),
        );
        if let Some(pos) = inf_add {
            let Ex::Add(terms) = &out[pos] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let (Some(flipped), Some(nc)) = (flipped, coeff.checked_neg()) {
                let re = add(flipped, cx);
                // Provable: merging/cancellation decisions ride STRIPPED keys,
                // which negation preserves (the coefficient sign lives outside the
                // key), so no licence fires on the flipped bag that did not fire on
                // the original; a canonical Add carries exactly ONE infinity term
                // (both-signs folds to NaN, same-sign merges, NaN absorbs -- all at
                // construction) and `negate_term` maps it 1:1 to the opposite sign.
                // The flip therefore re-enters `add`'s one-infinity exit with the
                // same term count >= 2: always a flat Add.
                debug_assert!(
                    matches!(re, Ex::Add(_)),
                    "flip of a bare-inf sum must stay a flat Add: {re:?}"
                );
                out[pos] = re;
                coeff = nc;
            }
        }
    }
    // A NEGATIVE coefficient never coexists with a
    // Const-bearing Add factor -- the sign folds INTO the sum, wherever the sign sits.
    // `negate_term` is exact family algebra per term (a Const-carrying term absorbs the
    // sign through `term_join`'s forall-exists refit; every other term negates its
    // coefficient as ordinary structure), so `-c x Add[S]` IS `c x Add[flip S]` as a
    // fitted family, and the subtracted Const TERM already loses its sign at
    // construction (the `has_const` absorption above) -- this clause completes that
    // doctrine at the sum level. Two failed narrowings pinned the required scope:
    // keeping the wrapper handed the projections a sign no spelling can carry on a
    // Const term (`-<constant>` is not a token: the display redistribution silently ate
    // it under pow bases, fuzz rows 148693/166766), and folding only the LONE
    // `-1 x Add` shape was not a CONGRUENCE -- the same value routes the sign through a
    // negative divisor literal (`S/(-3.89)`, bag coefficient) or a wider bag
    // (`-(S)/rootn(..)`), while the infix display pulls it out front and re-parse folds
    // through the lone shape: two canonical states, one family (67 fuzz rows
    // at 200k under the narrow arm). Folding at the BAG level makes the
    // decision a function of the bag alone: every construction route converges, and
    // displays never face an unspellable Const sign because the coefficient beside a
    // Const-bearing sum is always positive. The first Const-bearing Add under the bag
    // order takes the sign (deterministic; bags rarely hold two). Termination: the
    // negated coefficient is positive, and `add` of the flipped terms re-enters
    // `primitive_sum`, whose Const-bearing skip returns the flat bag -- no re-wrap, no
    // bounce. A `negate_term`/negation overflow refusal (i128::MIN) keeps the negative
    // coefficient, whose literal display is injective.
    if coeff.is_negative() {
        let const_add = out
            .iter()
            .position(|f| matches!(f, Ex::Add(v) if v.iter().any(Ex::contains_const)));
        if let Some(pos) = const_add {
            let Ex::Add(terms) = &out[pos] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let (Some(flipped), Some(nc)) = (flipped, coeff.checked_neg()) {
                let re = add(flipped, cx);
                debug_assert!(
                    matches!(re, Ex::Add(_)),
                    "flip of a Const-bearing sum must stay a flat Add: {re:?}"
                );
                out[pos] = re;
                coeff = nc;
            }
        }
    }
    // A NEGATIVE coefficient never coexists with an Add factor
    // holding a negation-ABSORBING member (`term_absorbs_negation`: bare-inf products,
    // inf-Add-factor products, and their recursive nestings) -- the sign folds INTO
    // that sum, member-wise, for ANY coefficient and ANY co-factor multiset, exactly
    // like the bare-inf and Const arms above. The wrapper spelling is
    // never re-parse-stable for this class: the display splits co-factors (fuzz row
    // 18943: `-(S)/0.0795` re-parses as `(-1 x S) / den`, the lone unit distributes,
    // and the two routes diverged), and a rewrite deep inside a factor can mint the
    // absorbing member AFTER the wrapper was legitimately built (fuzz rows
    // 212777/914946: the spine rebuild kept a wrapper `flipped_orientation_wins`
    // judged by the non-absorbed member spelling). `add(flipped)` re-settles the
    // absorbing members through `mul()` (term-level congruence) and `primitive_sum`'s
    // absorbing skip never re-wraps: no bounce, and the negated coefficient is positive.
    // A `negate_term` overflow refusal keeps the factored form, whose display is
    // injective.
    if coeff.is_negative() {
        let abs_add = out
            .iter()
            .position(|f| matches!(f, Ex::Add(ts) if ts.iter().any(term_absorbs_negation)));
        if let Some(pos) = abs_add {
            let Ex::Add(terms) = &out[pos] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let (Some(flipped), Some(nc)) = (flipped, coeff.checked_neg()) {
                let re = add(flipped, cx);
                debug_assert!(
                    matches!(re, Ex::Add(_)),
                    "flip of an absorbing-member sum must stay a flat Add: {re:?}"
                );
                out[pos] = re;
                coeff = nc;
            }
        }
    }
    // THE PRODUCT'S SIGN (`sign_place`). A product's sign trades between its coefficient and its
    // sign-trade sites -- a mixed-sign sum, an odd power or odd root of one, an odd function of
    // one (`f(-S) = -f(S)`); the sites of one class flip together. Of the spellings the flips
    // give, the product files the cheapest admitted one, the tie order on an equal price: a
    // normal-form choice, admitted by the premises (the zero checks; a sum that holds its sign is
    // no site), never by the price. The spelling priced is the spelling built, so the choice is a
    // function of the orbit, not of the entry spelling. A negation overflow keeps the entry
    // spelling. The result places its signs again as it stands (a flipped inner sum files bare,
    // `flipped_orientation_wins`), and the absorption arms above never meet these bags (their
    // sums are no sites).
    sign_place(coeff, out, cx)
}
