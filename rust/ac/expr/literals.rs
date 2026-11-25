//! WHERE NUMBERS LAND: a literal's one rounding ([`land`]), the zero products, strict's numbers,
//! the literal partitions, the float64 value of a ground factor, and permissive's literal
//! arithmetic.

use super::*;

/// THE PERMISSIVE LITERAL FOLD. An exact rational the constructors produced -- the quotient of two
/// 16-digit decimals, a coefficient cleared into `p/q`, a 30-digit integer -- prices under
/// mu' as the rational codeword alone once its expansion stops terminating, and the
/// emitter must spell it as that fraction, because a spelling denotes the state's EXACT
/// value (`decimal_spelling_wins` chooses among exact codewords only). On the permissive
/// tier the value is allowed to move -- that tier's licence -- so the literal folds to the
/// f64 nearest to it (as that float's shortest decimal) WHEN the exact codewords
/// ([`mu_rat_exact`]) price that spelling strictly cheaper, and every number is priced as it is
/// printed: `2e29/426738538271436458205631863649` (196.8 bits) becomes `0.46867105279529636`
/// (57.1 bits); `1/2`, `15/37` and `4366/8875` stay, their fractions being cheaper than
/// any 17-digit float; a literal that already IS a shortest f64 spelling is its own fold
/// and never moves, so the drawn constants of a training stream are untouched, and the fold
/// is idempotent. It is applied where a literal lands ([`land`]).
///
/// THE FLOAT IT LANDS ON: the value lies between two floats, and permissive promises float
/// precision, not the nearest float, so the literal lands on whichever of the two prints
/// shorter (the cheaper shortest spelling; the nearest on a tie). A literal combined from
/// literals that already landed can sit an ulp from the single rounding of its exact sources,
/// on the float whose shortest spelling carries a digit more (`0.016213652755841255` where the
/// sources give `0.01621365275584125`); the shorter neighbour is that rounding wherever it is
/// one of the two, and no worse a float anywhere. Values whose shortest
/// spelling is no number of the f64 domain refuse (fail-closed);
/// non-finite values
/// never fold.
pub fn lossy_literal(r: &Rat) -> Option<Rat> {
    let (y, other) = r.float_bracket()?;
    // Rust's `{:?}` for f64 is the shortest string that round-trips, which is precisely the
    // spelling whose f64 reading is this value again.
    let snapped = Rat::parse_decimal(&format!("{y:?}"))?;
    if snapped == *r {
        return None;
    }
    let snapped = match other.and_then(|o| Rat::parse_decimal(&format!("{o:?}"))) {
        Some(s) if mu_rat_exact(&s) < mu_rat_exact(&snapped) => s,
        _ => snapped,
    };
    (mu_rat_exact(&snapped) < mu_rat_exact(r)).then_some(snapped)
}

/// ρ, A LITERAL'S ONE ROUNDING: a literal is rounded once, when it becomes part of the
/// expression, not at every operation. Arithmetic BETWEEN
/// literals stays exact -- an all-literal node folds exactly, as in every mode -- and a literal
/// is moved by [`lossy_literal`] at the one place it LANDS as an operand of a composite node:
/// a sum's constant term or a term's coefficient (`add`, `term_join`), a product's coefficient
/// (`finish_product`), a power's literal base or exponent beside a non-literal one (`pow`), a
/// function's argument (`fun`), and the root of a state (the walk, [`land_root`]). So
/// `3*(1/3 + 1e-20)` folds exactly to `1 + 3e-20` and rounds once, to `1`; rounding inside
/// each operation would give `0.9999999999999999`.
///
/// In `strict` a literal lands on float64's own reading of it where that is cheaper
/// ([`strict_literal`]): float64 reads a literal as its nearest double, so writing that double's
/// shortest decimal keeps float64's value bit for bit (`x0*0.1234567890123456789` is
/// `0.12345678901234568*x0`); an integer and a literal no dearer than that decimal stay.
///
/// ρ is idempotent (a moved literal is a shortest float spelling, which neither mode moves
/// again), and never raises a price: it moves a literal only to a strictly cheaper spelling
/// ([`mu_rat`]), so the reduction ordering, termination and every descent argument are
/// unchanged, and every state the walk holds prints the codeword it is priced at. In permissive
/// it moves a value to one of the two floats around it, less than an ulp, inside permissive's
/// licence; it never creates an infinity or a zero (non-finite values refuse, and both floats
/// around a nonzero value are nonzero).
///
/// NO LITERAL LANDS INSIDE A GROUND NODE in permissive ([`in_ground`]): a ground subtree is
/// valued whole by the table and written as one float, rounded once (P3, the engine's ground
/// fold), so `14.81 - cos(c)` with `c` a product of ten literals reads the cosine of the exact
/// product.
///
/// The laws applied here (`ac::laws::Law`): `num.literal_to_float`.
#[inline]
pub fn land(r: Rat, cx: &Cx) -> Rat {
    if !cx.permits(Permissive::LiteralToFloat) {
        return strict_literal(r, cx);
    }
    match lossy_literal(&r) {
        Some(s) => {
            if crate::ac::rules::folds_logged() {
                crate::ac::rules::log_fold("literal", vec![Ex::Num(r)], Ex::Num(s.clone()));
            }
            s
        }
        None => r,
    }
}

/// Does a literal land among `others` inside a GROUND node: are they all ground (no variable,
/// wildcard or `<constant>`)? Then it does not land ([`land`]): the fully numeric part is valued
/// whole, by the ground evaluator.
pub(super) fn in_ground<'a>(others: impl IntoIterator<Item = &'a Ex>, cx: &Cx) -> bool {
    others
        .into_iter()
        .all(|e| !e.has_measure_space(cx.view) && !e.contains_wildcard(cx.view))
}

/// [`land`] on a literal operand; any other operand is returned as it is.
#[inline]
pub(super) fn land_operand(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Num(r) => Ex::Num(land(r, cx)),
        e => e,
    }
}

/// [`land`] at the root of a state: a state that is one literal is that literal landed. And
/// nothing reads the answer's zero: `log(exp(t))` and `acosh(cosh(t))` collapse wherever nothing
/// reads their flat point (`release_flat`; `log(exp(x0))` is `x0` as an answer, while
/// `1/log(exp(x0))` stays), a zero product drops the factors that only gave it its sign
/// (`unsigned_zero`; `0*x0` is 0 as an answer, while `1/(0*x0)` is `inf/x0`), and a sum's sign
/// moves wherever nothing reads its zero (`release_unread`; `-ceil(x0) - floor(x1)` is
/// `-(ceil(x0) + floor(x1))` as an answer, while `1/(-ceil(x0) - floor(x1))` stays).
pub fn land_root(e: Ex, cx: &Cx) -> Ex {
    let e = release_flat(e, cx);
    let e = release_unread(e, cx);
    if let Some(u) = unsigned_zero(&e, cx) {
        return u;
    }
    // ... and a sum of zero products is a zero whose sign nothing reads either.
    if let Ex::Add(v) = &e {
        if v.iter().all(|t| zero_product(t, cx)) {
            let terms: Vec<Ex> = v
                .iter()
                .map(|t| unsigned_zero(t, cx).unwrap_or_else(|| t.clone()))
                .collect();
            let settled = add(terms, cx);
            if settled != e {
                return land_root(settled, cx);
            }
        }
    }
    land_operand(e, cx)
}

/// Is `f` a zero member of a product: the literal zero, or the written negative zero?
fn zero_member(f: &Ex, view: &TokenView) -> bool {
    matches!(f, Ex::Num(r) if r.is_zero()) || is_neg_zero(f, view)
}

/// In `f64`, is `t` a ZERO PRODUCT: a product holding a zero, whose value is `+0`, `-0` or NaN at
/// every point, the zero's sign the sign of its other members (`mul.zero`)?
pub(crate) fn zero_product(t: &Ex, cx: &Cx) -> bool {
    f64_zero_signs(cx) && matches!(t, Ex::Mul(v) if v.iter().any(|f| zero_member(f, cx.view)))
}

/// Is `t` a zero product that may be `-0`: its zero the written negative zero, or a member whose
/// sign is not certainly positive?
pub(super) fn zero_of_either_sign(t: &Ex, cx: &Cx) -> bool {
    zero_product(t, cx)
        && matches!(t, Ex::Mul(v) if v.iter().any(|f| is_neg_zero(f, cx.view)
            || (!zero_member(f, cx.view) && !cx.certainly_nonneg(f))))
}

/// A zero product whose sign nothing reads, without the factors that only gave it its sign: its
/// members finite almost everywhere go and the zero is the literal zero, built as `mul` builds it
/// (the zero itself where nothing is left; the NaN of `0*log(x0)` stays). `None` where `t` is no
/// zero product, or nothing goes.
pub(crate) fn unsigned_zero(t: &Ex, cx: &Cx) -> Option<Ex> {
    if !zero_product(t, cx) {
        return None;
    }
    let Ex::Mul(v) = t else { return None };
    let carriers: Vec<Ex> = v
        .iter()
        .filter(|f| !zero_member(f, cx.view) && !cx.fin_certified(f))
        .cloned()
        .collect();
    let mut items = vec![Ex::Num(Rat::ZERO)];
    items.extend(carriers);
    let u = mul(items, cx);
    (u != *t).then_some(u)
}

/// A term of a sum that nothing reads the zeros of (`add`'s settled sum), pushed to `out` as the
/// sum takes it: a zero product without the factors that only gave it its sign (`unsigned_zero`),
/// and a zero product over a sum spread over that sum's terms, each taken the same way
/// (`x1 + 0*(2*x0 + sqrt(x0))` is `x1 + 0*sqrt(x0)`). Spread or whole, the zero products are NaN
/// at the same points (`0*inf` and `0*NaN` are NaN, `0*inf + 0*(-inf)` too) and a zero of some
/// sign elsewhere, which the sum does not read. A literal zero is no term.
pub(super) fn settle_zero_term(t: Ex, out: &mut Vec<Ex>, cx: &Cx) {
    let u = unsigned_zero(&t, cx).unwrap_or(t);
    if let Ex::Mul(v) = &u {
        if let Some((r, inner)) = factored_sum(v) {
            if r.is_zero() {
                let scaled: Option<Vec<Ex>> = inner.iter().map(|s| scale_term(s, &r, cx)).collect();
                if let Some(scaled) = scaled {
                    for s in scaled {
                        settle_zero_term(s, out, cx);
                    }
                    return;
                }
            }
        }
    }
    match u {
        Ex::Num(z) if z.is_zero() => {}
        u => out.push(u),
    }
}

/// A FACTORED SUM: a product of literals and one sum, `r * (a + b)`, as its coefficient (a refused
/// partition's members are one) and the sum's terms.
pub(super) fn factored_sum(v: &[Ex]) -> Option<(Coef, &[Ex])> {
    match v.split_last() {
        Some((Ex::Add(inner), lits))
            if !lits.is_empty() && lits.iter().all(|f| matches!(f, Ex::Num(_))) =>
        {
            let r = Coef::from_members(lits.iter().map(|f| match f {
                Ex::Num(r) => r,
                _ => unreachable!("literal members"),
            }));
            Some((r, inner.as_slice()))
        }
        _ => None,
    }
}

/// `add.unfactor`'s premise: does a factored sum `r * (a + b)` entering a sum unfactor into
/// `r*a + r*b`? The two agree at every point (a finite rational distributes over extended-real
/// sums) but where the sum's zero is read. In `f64`, float64's `a + b` is `+0` wherever it cancels,
/// and `-0` only where both terms are:
///
/// * a positive `r` keeps every zero's sign: always;
/// * a negative `r` makes `r*(a + b)` `-0` where the sum cancels, while `r*a + r*b` cancels to
///   `+0`: only where the sum holds no sign (`Law::AddUnfactor`; `-2*(ceil(x0) + floor(x1))`
///   stays whole, `-2*(x0 + x1)` unfactors), unless another addend settles the sum (`add`);
/// * a zero gives `0*(a + b)` the sign of `a + b`, which the terms' zeros do not spell
///   (`0*(x0 - x1)` is `-0` wherever `x0 < x1`, `0*x0 + 0*x1` only where both are negative): it
///   stays whole here, and spreads where the sum settles, which reads none of its zeros
///   (`settle_zero_term`).
///
/// `permissive` reads one zero: always.
pub(super) fn unfactors(r: &Coef, inner: &[Ex], cx: &Cx) -> bool {
    if !f64_zero_signs(cx) || !(r.is_zero() || r.is_negative()) {
        return true;
    }
    !r.is_zero() && Law::AddUnfactor.holds(&Ex::Add(inner.to_vec()), cx)
}

/// STRICT'S NUMBERS, written as the cheapest value strict admits for them: a number the
/// constructors make from literals is either its exact value (the reals' value, which float64's
/// kind keeps: both finite) or the double nearest the same operation on the literals' doubles,
/// computed exactly and rounded once (`rat::rounded_once`; for two literals float64's own value of
/// the operation), written as its shortest decimal (float64 reads it as that double), whichever
/// is cheaper, and the exact value at an equal price.
/// `x0*0.30000000000000004*0.30000000000000004` is
/// `0.09000000000000002*x0` (the exact product has 34 digits), `0.1 + 0.2` is `0.3` (the exact
/// sum is cheaper than float64's `0.30000000000000004`). Float64's value is taken only where it
/// is a normal double, or an exact zero the exact value also is: an overflow or an underflow
/// keeps the exact value. An integer stays exact (`3^40` is `12157665459056928801`): its parity
/// and its exactness are the point of it. The decimal must be a number of the f64 number domain,
/// as a written literal is (`rat::number_domain`): `3.3333333333333334e-301`, whose denominator
/// passes 2^1022, is no number there, and the exact value stays.
pub(crate) fn strict_number(exact: Rat, float: f64) -> Rat {
    if exact.is_integer() || !(float.is_normal() || (float == 0.0 && exact.is_zero())) {
        return exact;
    }
    // The double's shortest decimal is a function of its bits: each thread remembers it, emptying
    // the table at 65,536 entries.
    thread_local! {
        static DECIMALS: std::cell::RefCell<rustc_hash::FxHashMap<u64, Option<Rat>>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    let decimal = DECIMALS.with(|m| m.borrow().get(&float.to_bits()).cloned());
    let decimal = match decimal {
        Some(d) => d,
        None => {
            let d = {
                let _f64 = crate::ac::rat::number_domain(true);
                Rat::parse_decimal(&format!("{float:?}"))
            };
            DECIMALS.with(|m| {
                let mut m = m.borrow_mut();
                if m.len() >= 1 << 16 {
                    m.clear();
                }
                m.insert(float.to_bits(), d.clone());
            });
            d
        }
    };
    match decimal {
        Some(d) if mu_rat_exact(&d) < mu_rat_exact(&exact) => d,
        _ => exact,
    }
}

/// The literal partition of a sum (`Rat::partition_sum`), exact in every mode; in `strict`, where
/// the literals combine into one number, that number is written as the cheaper of its exact value
/// and the sum of the literals' doubles, exact and rounded once ([`strict_number`]). Rounding only
/// the combined number keeps every exact cancellation a later fold of the partition reaches.
/// Inside a fully numeric sum (`ground`) the partition stays exact: the part is valued whole.
pub(super) fn literal_partition_sum(lits: Vec<Rat>, ground: bool, cx: &Cx) -> Vec<Rat> {
    let float = || {
        let ds: Vec<f64> = lits
            .iter()
            .filter(|m| !m.is_zero())
            .map(Rat::to_f64_nearest)
            .collect();
        crate::ac::rat::rounded_once('+', &ds)
            .unwrap_or_else(|| ds.iter().fold(0.0, |acc, d| acc + d))
    };
    let combined = lits.iter().filter(|m| !m.is_zero()).count() >= 2;
    let y = if cx.f64_semantics() && cx.fold_f64 && combined && !ground {
        Some(float())
    } else {
        None
    };
    let operands = (combined && !ground && crate::ac::rules::folds_logged())
        .then(|| lits.iter().filter(|m| !m.is_zero()).cloned().collect());
    let mut out = Rat::partition_sum(lits);
    if let (Some(y), [one]) = (y, out.as_mut_slice()) {
        *one = strict_number(one.clone(), y);
    }
    log_partition("+", operands, &out);
    out
}

/// Record a literal partition that combined its operands into one number (a [`crate::ac::rules::Fold`]
/// of `op`), where a fire log is open.
fn log_partition(op: &'static str, operands: Option<Vec<Rat>>, out: &[Rat]) {
    if let (Some(operands), [one]) = (operands, out) {
        crate::ac::rules::log_fold(
            op,
            operands.into_iter().map(Ex::Num).collect(),
            Ex::Num(one.clone()),
        );
    }
}

/// The coefficient partition of a product (`Rat::partition_product`), exact in every mode; in
/// `strict`, where the literals combine into one number, that number is written as the cheaper
/// of its exact value and the product of the literals' doubles, exact and rounded once
/// ([`strict_number`]): `0.30000000000000004*0.30000000000000004` is `0.09000000000000002`,
/// `1/3*1e300*1e-300` is `1/3`, and the reciprocals of `89875517873681764`, `8.854e-12` and
/// `1.380649e-23` make `91019782591670690`, where float64's product taken one rounding at a time
/// gives `91019782591670700`. Inside a fully numeric product (`ground`) the partition stays
/// exact: the part is valued whole.
pub(super) fn literal_partition_product(nums: Vec<Rat>, ground: bool, cx: &Cx) -> Vec<Rat> {
    let combined = nums.iter().filter(|m| !m.is_one()).count() >= 2;
    let y = if cx.f64_semantics() && cx.fold_f64 && combined && !ground {
        let ds: Vec<f64> = nums
            .iter()
            .filter(|m| !m.is_one())
            .map(Rat::to_f64_nearest)
            .collect();
        Some(
            crate::ac::rat::rounded_once('*', &ds)
                .unwrap_or_else(|| ds.iter().fold(1.0, |acc, d| acc * d)),
        )
    } else {
        None
    };
    let operands = (combined && !ground && crate::ac::rules::folds_logged())
        .then(|| nums.iter().filter(|m| !m.is_one()).cloned().collect());
    let mut out = Rat::partition_product(nums);
    if let (Some(y), [one]) = (y, out.as_mut_slice()) {
        *one = strict_number(one.clone(), y);
    }
    log_partition("*", operands, &out);
    out
}

/// A WRITTEN LITERAL in `strict`: float64 reads it as its nearest double, so a decimal of more
/// digits than that double needs is written as the double's shortest decimal
/// (`0.1234567890123456789` is `0.12345678901234568`); an integer keeps its exact spelling
/// (its parity and its exact arithmetic are the point of it), and so does a literal whose own
/// spelling is no dearer. In the simplify walk only (`Cx::fold_f64`): a conversion, a pattern
/// and the measure's canon keep every number as written.
pub(crate) fn strict_literal(r: Rat, cx: &Cx) -> Rat {
    if !cx.f64_semantics() || !cx.fold_f64 || r.is_integer() {
        return r;
    }
    let y = r.to_f64_nearest();
    let written = strict_number(r.clone(), y);
    if written != r && crate::ac::rules::folds_logged() {
        crate::ac::rules::log_fold("literal", vec![Ex::Num(r)], Ex::Num(written.clone()));
    }
    written
}

/// The float64 value of a GROUND factor (no variable, `<constant>` or wildcard) as the deployed
/// evaluator computes it, the sign of a zero included: its literal-token spelling, the one the
/// ground fold evaluates (`convert::to_prefix_literal_tokens`), through the evaluator's kernels
/// (`numeric::apply_op`). `None` for anything else, or a head no kernel evaluates.
pub(super) fn ground_value(e: &Ex, cx: &Cx) -> Option<f64> {
    if e.has_measure_space(cx.view) || e.contains_wildcard(cx.view) {
        return None;
    }
    let view = cx.view;
    let tokens = crate::ac::convert::to_prefix_literal_tokens(e, &Cx::bare(view));
    fn node(tokens: &[Tok], at: &mut usize, view: &TokenView) -> Option<f64> {
        let t = *tokens.get(*at)?;
        *at += 1;
        let s = view.resolve_owned(t);
        match crate::operators::core_arity(&s).or_else(|| view.arity(t)) {
            Some(n) if n <= 2 => {
                let mut args = [0.0f64; 2];
                for a in args.iter_mut().take(usize::from(n)) {
                    *a = node(tokens, at, view)?;
                }
                crate::numeric::apply_op(&s, &args[..usize::from(n)])
            }
            Some(_) => None,
            None => crate::numeric::leaf_value(&s),
        }
    }
    let mut at = 0;
    let v = node(&tokens, &mut at, view)?;
    (at == tokens.len()).then_some(v)
}

/// The float64 value of a written product that is zero, in `f64`: the product of the signs of its
/// factors (`-1*0`, `0*(-2)` and `-(0)` are -0, `-1*-0.0` is +0), which the exact coefficient,
/// one zero, cannot hold, so `-0.0` (`NEG_ZERO`) where an odd number of the factors are negative
/// or the written negative zero, else `0`. A factor that is no number but ground gives the sign
/// of its float64 value (`ground_value`): `0*sin(-1)` is -0 and `sinh(-0.0)*0` is -0, as float64
/// computes them, where the zero would drop the factor and keep +0; this is the ground fold's
/// reading (`Cx::fold_f64`), so a rule file read in the canon it was certified in (`RuleCanon`),
/// which does not fold, keeps its rows. `None` outside `f64` and for
/// any other product (a factor that is neither a number nor ground and finite, or no zero
/// literal). The parse builds a written product through it; a zero the search derives keeps the
/// one zero (the region rule).
///
/// The laws applied here (`ac::laws::Law`): `num.signed_zero`.
pub(crate) fn signed_zero_product(items: &[Ex], cx: &Cx) -> Option<Ex> {
    if !cx.f64_semantics() {
        return None;
    }
    let mut zero = false;
    let mut negative = false;
    let mut ground: Vec<&Ex> = Vec::new();
    for f in items {
        match f {
            Ex::Num(r) => {
                zero |= r.is_zero();
                negative ^= r.is_negative();
            }
            f if is_neg_zero(f, cx.view) => {
                zero = true;
                negative = !negative;
            }
            f if cx.fold_f64 && !f.has_measure_space(cx.view) && !f.contains_wildcard(cx.view) => {
                ground.push(f)
            }
            _ => return None,
        }
    }
    if !zero {
        return None;
    }
    for f in ground {
        let v = ground_value(f, cx)?;
        if !v.is_finite() {
            return None;
        }
        negative ^= v.is_sign_negative();
    }
    zero.then(|| {
        if negative {
            Ex::Leaf(cx.view.intern(NEG_ZERO))
        } else {
            Ex::Num(Rat::ZERO)
        }
    })
}

/// `floor` (or `ceil`) of a rational, exactly; `None` when the result leaves `Rat`.
pub(super) fn rat_floor(r: &Rat, ceil: bool) -> Option<Rat> {
    use num_integer::Integer;
    if r.is_integer() {
        return Some(r.clone());
    }
    let (p, q) = r.big_parts();
    let f = p.div_floor(&q);
    Rat::from_big(if ceil { f + 1 } else { f }, BigInt::from(1))
}

/// PERMISSIVE'S LITERAL ARITHMETIC IS THE TABLE'S: the value of the operation `op` at literal
/// arguments (rationals, the infinities, NaN) as THE TABLE reads it in the reals' values
/// (`table::apply`), where it is exact -- a rational, an infinity, or NaN where the limit rule
/// gives no value:
/// `pow(0, -1)`, `pow(0, -2)`, `rootn(0, -2)` and `mod(1, 0)` (a division by zero),
/// `pow(1, inf)`, `pow(inf, 0)`, `pow(nan, 0)` and `mod(0, inf)` are NaN, `pow(-inf, 0)` 1,
/// `exp(-inf)` 0, `tanh(inf)` 1, `acosh(1)` 0. `None` for an approximation (a transcendental
/// value stays symbolic) or an operator the table does not know.
///
/// The laws applied here (`ac::laws::Law`): `num.table_fold`.
pub(super) fn table_literal_fold(op: &str, args: &[Ex]) -> Option<Ex> {
    use crate::table::{Exact, Val};
    let vals: Option<Vec<Val>> = args
        .iter()
        .map(|a| match a {
            Ex::Num(r) => Some(Val::Exact(Exact::Rat(r.clone()))),
            Ex::PosInf => Some(Val::Exact(Exact::PosInf)),
            Ex::NegInf => Some(Val::Exact(Exact::NegInf)),
            Ex::NaN => Some(Val::Exact(Exact::Undef)),
            _ => None,
        })
        .collect();
    let mut ctx = crate::table::Ctx::new(50)?;
    match crate::table::apply(op, &vals?, &mut ctx)? {
        Val::Exact(Exact::Rat(r)) => Some(Ex::Num(r)),
        Val::Exact(Exact::PosInf) => Some(Ex::PosInf),
        Val::Exact(Exact::NegInf) => Some(Ex::NegInf),
        Val::Exact(Exact::Undef) => Some(Ex::NaN),
        _ => None,
    }
}

/// Is `e` a literal the table reads exactly: a rational, an infinity or NaN?
pub(super) fn table_literal(e: &Ex) -> bool {
    matches!(e, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::NaN)
}
