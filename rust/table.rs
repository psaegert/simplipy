//! THE TABLE: each mode's values, defined once.
//!
//! Every step that asks "what is this expression's value here?" or "do these two values agree?"
//! reads this module: how a literal is read, what each operation gives in the mode (including at
//! its poles and at infinite arguments), how two values are compared, and which bindings each
//! slot sort quantifies over. The contract judge reads it through `_core.TableExpr` and the
//! `table_*` functions; nothing in Python keeps a copy.
//!
//! **the reals** (permissive's values):
//!
//! - The values are the reals, +inf and -inf; undefined (NaN) is the absence of a value. There is
//!   one zero.
//! - A literal is exact: a decimal is the rational it spells, `np.pi` is pi, `np.e` is e.
//! - An operation is the real function on its real domain, principal branch. Where it has no
//!   value by its formula (a pole, a boundary, an infinite argument) its value is THE LIMIT
//!   RULE's: the limit of the operation as its arguments approach the point from inside its
//!   domain, taken in all arguments together, if every direction gives the same limit; otherwise
//!   undefined. `log(0)` is -inf (one side in the domain), `1/0` and `tan(pi/2)` are undefined
//!   (the sides disagree), `pow(1, inf)` is undefined (along x = 1 + 1/y the value tends to e),
//!   `mod(0, inf)` is undefined (along (1/n, n) the remainder tends to 0, along (-1/n, n) to
//!   +inf), `pow(-inf, 0)` is 1 (a negative base admits only integer exponents, and near 0 the
//!   only one is 0).
//! - Division by zero is undefined however it is written, even where a limit exists: `1/0`,
//!   `inv(0)`, a negative power of zero (`pow(0, -2)`, though 1/x^2 tends to +inf at 0), a
//!   remainder by zero (`mod(1, 0)`), a root whose index is not a finite integer
//!   (`rootn(2, inf)`). So `x^-2` and `1/x^2` agree everywhere.
//! - Undefined propagates: an operation with an undefined operand is undefined.
//!
//! The evaluator carries a value EXACTLY wherever it can -- rationals, rational multiples of pi,
//! the infinities, undefined -- so every point the table decides is decided: `ceil(0.5)` is
//! exactly 1, `asin(1)` exactly pi/2, and `tan` there is a pole, never a number near 1.6e16. A
//! value it cannot carry exactly is a high-precision approximation, judged with the
//! working-precision honesty of the contract's ladder: a computed value inside the precision band
//! of a boundary cannot be told from it, and the point is Unresolved (skipped, never convicted).
//!
//! **f64**: the values are IEEE-754 doubles; a literal is the double nearest to it (a written
//! `-0.0` is -0.0); an operation is what the engine's own float64 kernels compute
//! (`numeric::apply_op`), and the range standard (`numeric::RangeKind`) marks where an
//! operation overflowed or underflowed. Two f64 values agree when both are undefined, the same
//! infinity, or finite within 8 units in the last place (the two zeros as one value).
//!
//! **The sorts** (`sort_atoms`): `?` quantifies over real data (finite values only); `!` over
//! finite values, its +-inf atoms excused (they occur on a null set); `$` over nonzero finite
//! values, its 0 and +-inf excused; `_` over every value, +-inf and undefined included and none
//! excused. Permissive's guarantee covers real-valued bindings only.

use astro_float::{BigFloat, Consts, Radix, INF_NEG, INF_POS, NAN};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::ac::rat::{number_domain, token_rational, Rat};
use crate::hiprec::{
    bessel, besselj, cmp_gt, cmp_lt, dps_to_prec, is_pos, one, pow_finite_pos, small_int, special,
    RM,
};

// ------------------------------------------------------------------------------- values

/// A value the table carries exactly.
#[derive(Clone, Debug, PartialEq)]
pub enum Exact {
    /// a rational number
    Rat(Rat),
    /// a rational multiple of pi, `q * pi` (never zero: zero is `Rat`)
    Pi(Rat),
    PosInf,
    NegInf,
    /// no value
    Undef,
}

/// A value at a node: exact where the table knows it, else a finite high-precision
/// approximation.
#[derive(Clone, Debug)]
pub enum Val {
    Exact(Exact),
    Approx(BigFloat),
}

/// A leaf of a compiled expression.
#[derive(Clone, Debug)]
pub enum Leaf {
    /// an exact literal
    Exact(Exact),
    /// a numeral too large for the exact form: read at the working precision
    Big(BigInt, BigInt),
    /// a decimal `m * 10^k` whose exponent lies past the exact readers' bound
    /// (`rat::token_rational`): read at the working precision
    Dec(BigInt, i32),
    /// Euler's number
    E,
}

/// The largest decimal exponent a literal may carry (`Leaf::Dec`): the engine keeps a written
/// `2.827881176741212e-244756` as it is, so its reader must too, and a bound keeps an untrusted
/// rule file from asking for `10^(10^9)`.
const DEC_EXPONENT_MAX: i64 = 1_000_000;

/// A decimal numeral as `(m, k)` with its value `m * 10^k`, at any exponent up to
/// `DEC_EXPONENT_MAX` and at most `rat::READER_DIGITS` mantissa digits; `None` for anything else.
fn big_decimal(t: &str) -> Option<(BigInt, i32)> {
    if !crate::utils::is_decimal_numeral(t) {
        return None;
    }
    let (mant, exp) = match t.find(['e', 'E']) {
        Some(i) => (&t[..i], t[i + 1..].parse::<i64>().ok()?),
        None => (t, 0),
    };
    let (negative, mant) = match mant.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mant.strip_prefix('+').unwrap_or(mant)),
    };
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let digits: String = int_part.chars().chain(frac_part.chars()).collect();
    if digits.is_empty() || digits.len() > crate::ac::rat::READER_DIGITS {
        return None;
    }
    let k = exp.checked_sub(frac_part.len() as i64)?;
    if k.abs() > DEC_EXPONENT_MAX {
        return None;
    }
    let m: BigInt = digits.parse().ok()?;
    Some((if negative { -m } else { m }, i32::try_from(k).ok()?))
}

/// A literal as real reads it: `None` for a token that is no literal. The grammar is f64's
/// (`literal_f64`): a token is a literal in one mode exactly where it is one in the other, and
/// only its value differs.
fn real_leaf(tok: &str) -> Option<Leaf> {
    let t = tok.trim();
    crate::numeric::leaf_value(t)?;
    match t {
        "np.pi" => return Some(Leaf::Exact(Exact::Pi(Rat::int(1)))),
        "np.e" => return Some(Leaf::E),
        "float(\"inf\")" => return Some(Leaf::Exact(Exact::PosInf)),
        "float(\"-inf\")" => return Some(Leaf::Exact(Exact::NegInf)),
        "float(\"nan\")" => return Some(Leaf::Exact(Exact::Undef)),
        _ => {}
    }
    let (negate, body) = match t.strip_prefix("(-").and_then(|s| s.strip_suffix(')')) {
        Some(inner) => (true, inner),
        None => (false, t),
    };
    let Some((p, q)) = token_rational(body) else {
        let (m, k) = big_decimal(body)?;
        return Some(Leaf::Dec(if negate { -m } else { m }, k));
    };
    let p = if negate { -p } else { p };
    let _exact = number_domain(false);
    Some(match Rat::from_big(p.clone(), q.clone()) {
        Some(r) => Leaf::Exact(Exact::Rat(r)),
        None => Leaf::Big(p, q),
    })
}

/// The real reading of a literal token, for the judge's own parser: `("q", p, q)` for a number
/// (unreduced), `("pi",)`, `("e",)`, `("inf", sign)`, `("nan",)`, or `None` for a token that is no
/// literal.
pub fn literal_real(tok: &str) -> Option<(String, String, String)> {
    let leaf = real_leaf(tok)?;
    Some(match leaf {
        Leaf::Exact(Exact::Rat(r)) => ("q".into(), r.numer_string(), r.denom_string()),
        Leaf::Exact(Exact::Pi(_)) => ("pi".into(), String::new(), String::new()),
        Leaf::Exact(Exact::PosInf) => ("inf".into(), "+".into(), String::new()),
        Leaf::Exact(Exact::NegInf) => ("inf".into(), "-".into(), String::new()),
        Leaf::Exact(Exact::Undef) => ("nan".into(), String::new(), String::new()),
        Leaf::Big(p, q) => ("q".into(), p.to_string(), q.to_string()),
        Leaf::Dec(m, k) => ("d".into(), m.to_string(), k.to_string()),
        Leaf::E => ("e".into(), String::new(), String::new()),
    })
}

/// The f64 reading of a literal token (the double nearest to it, a written `-0.0` keeping its
/// sign): `None` for a token that is no literal.
pub fn literal_f64(tok: &str) -> Option<f64> {
    crate::numeric::leaf_value(tok.trim())
}

// --------------------------------------------------------------------- the evaluator

/// The per-evaluation context: precision, the dps-derived honesty band and pole ceiling, and
/// what the evaluation observed.
pub struct Ctx {
    p: usize,
    dps: i32,
    /// 10^(-dps+10): the symbolic-cancellation floor and the boundary-honesty band
    floor: BigFloat,
    /// 10^(dps-10): the pole ceiling, and the resolution limit of the step heads
    ceil: BigFloat,
    /// a symbolic-cancellation snap or a pole at working precision fired
    pub snapped: bool,
    /// some operation produced an infinity
    pub inf_node: bool,
    /// the largest finite intermediate
    pub max_mag: BigFloat,
    /// the smallest nonzero finite intermediate (sums and remainders excluded)
    pub min_mag: BigFloat,
    /// the smallest distance to its bound of a bounded function left inside its band
    /// (`note_distance`), zero where none was
    pub band: BigFloat,
    /// THE FLOAT-RESOLUTION READING (P3): a bounded function inside its boundary band
    /// (`tanh` within 1e-40 of +-1 at 50 digits) reads as the working float's value there,
    /// whose nearest double is the limit's, where the exact reading is Unresolved (it cannot
    /// tell the value from the limit). For a double's worth of digits only, never for an exact
    /// comparison.
    pub saturate: bool,
    cc: PooledConsts,
}

/// 10^k at precision `p` by binary powering (a band edge: a few ulps of rounding are immaterial,
/// a linear product of thousands of factors at thousands of bits is not).
fn ten_to(k: i32, p: usize) -> BigFloat {
    let t = small_int(10, p).powi(k.unsigned_abs() as usize, p, RM);
    if k < 0 {
        one(p).div(&t, p, RM)
    } else {
        t
    }
}

thread_local! {
    /// Constant caches kept across evaluations: pi, e and ln 2 at a working precision are
    /// computed once per thread, not once per point.
    static CONSTS: std::cell::RefCell<Vec<Consts>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A constant cache borrowed from the thread's pool, returned to it on drop.
struct PooledConsts(Option<Consts>);

impl PooledConsts {
    fn take() -> Option<PooledConsts> {
        let cached = CONSTS.with(|c| c.borrow_mut().pop());
        Some(PooledConsts(Some(match cached {
            Some(cc) => cc,
            None => Consts::new().ok()?,
        })))
    }
}

impl Drop for PooledConsts {
    fn drop(&mut self) {
        if let Some(cc) = self.0.take() {
            CONSTS.with(|c| c.borrow_mut().push(cc));
        }
    }
}

impl std::ops::Deref for PooledConsts {
    type Target = Consts;
    fn deref(&self) -> &Consts {
        self.0.as_ref().expect("a pooled cache is held until drop")
    }
}

impl std::ops::DerefMut for PooledConsts {
    fn deref_mut(&mut self) -> &mut Consts {
        self.0.as_mut().expect("a pooled cache is held until drop")
    }
}

impl Ctx {
    pub fn new(dps: i32) -> Option<Ctx> {
        let p = dps_to_prec(dps);
        Some(Ctx {
            p,
            dps,
            floor: ten_to(-dps + 10, p),
            ceil: ten_to(dps - 10, p),
            snapped: false,
            inf_node: false,
            max_mag: small_int(0, p),
            min_mag: small_int(0, p),
            band: small_int(0, p),
            saturate: false,
            cc: PooledConsts::take()?,
        })
    }

    fn zero(&self) -> BigFloat {
        small_int(0, self.p)
    }

    /// A finite value as a float at the working precision (the miner's row comparison).
    pub fn value_bf(&mut self, v: &Val) -> BigFloat {
        self.bf(v)
    }

    fn pi(&mut self) -> BigFloat {
        self.cc.pi(self.p, RM)
    }

    /// A big integer at the working precision (rounded once).
    fn big(&mut self, n: &BigInt) -> BigFloat {
        if let Some(v) = n.to_i64() {
            let b = BigFloat::from_u64(v.unsigned_abs(), self.p.max(64));
            let b = b.add(&self.zero(), self.p, RM);
            return if v < 0 { b.neg() } else { b };
        }
        BigFloat::parse(&n.to_string(), Radix::Dec, self.p + 64, RM, &mut self.cc).add(
            &self.zero(),
            self.p,
            RM,
        )
    }

    fn ratio(&mut self, p: &BigInt, q: &BigInt) -> BigFloat {
        let g = self.p + 64;
        let (a, b) = (self.big_at(p, g), self.big_at(q, g));
        a.div(&b, self.p, RM)
    }

    /// `m * 10^k` at the working precision (`Leaf::Dec`).
    fn decimal(&mut self, m: &BigInt, k: i32) -> BigFloat {
        let g = self.p + 64;
        let a = self.big_at(m, g);
        a.mul(&ten_to(k, g), self.p, RM)
    }

    fn big_at(&mut self, n: &BigInt, g: usize) -> BigFloat {
        if let Some(v) = n.to_i64() {
            let b = BigFloat::from_u64(v.unsigned_abs(), g.max(64));
            return if v < 0 { b.neg() } else { b };
        }
        BigFloat::parse(&n.to_string(), Radix::Dec, g, RM, &mut self.cc)
    }

    /// The working-precision reading of an exact value.
    pub fn bf(&mut self, v: &Val) -> BigFloat {
        match v {
            Val::Approx(b) => b.clone(),
            Val::Exact(Exact::Rat(r)) => {
                let (p, q) = r.big_parts();
                if q.is_one() {
                    self.big(&p)
                } else {
                    self.ratio(&p, &q)
                }
            }
            Val::Exact(Exact::Pi(r)) => {
                let (p, q) = r.big_parts();
                let g = self.p + 64;
                let pi = self.cc.pi(g, RM);
                let num = self.big_at(&p, g);
                let den = self.big_at(&q, g);
                pi.mul(&num, g, RM).div(&den, self.p, RM)
            }
            Val::Exact(Exact::PosInf) => INF_POS,
            Val::Exact(Exact::NegInf) => INF_NEG,
            Val::Exact(Exact::Undef) => NAN,
        }
    }

    fn note_mag(&mut self, v: &Val, additive: bool) {
        let b = match v {
            Val::Exact(Exact::PosInf | Exact::NegInf | Exact::Undef) => return,
            _ => self.bf(v),
        };
        if b.is_nan() || b.is_inf() {
            return;
        }
        let a = b.abs();
        if cmp_gt(&a, &self.max_mag) {
            self.max_mag = a.clone();
        }
        if !additive && !a.is_zero() && (self.min_mag.is_zero() || cmp_lt(&a, &self.min_mag)) {
            self.min_mag = a;
        }
    }

    /// A bounded function (`tanh`, `erf`, `erfc`) whose value lies inside the band of its bound
    /// is unresolved at this precision, and it is its distance `d > 0` to the bound that a
    /// reading must carry: recorded (`band`), so the judge's ladder can size a rung that
    /// resolves it where it owes the point a reading. `floor(erf(exp(3)))` is 0, read at about
    /// 400 digits (`erf(exp(3))` is 1 - 1.7e-177). The float-resolution reading, which reads such
    /// a value as its bound (`saturate`), records nothing.
    fn note_distance(&mut self, d: &BigFloat) {
        if self.saturate || d.is_zero() || d.is_nan() || d.is_inf() {
            return;
        }
        let a = d.abs();
        if self.band.is_zero() || cmp_lt(&a, &self.band) {
            self.band = a;
        }
    }

    /// `note_distance` of `erf(x)` to +-1, and of `erfc(x)` to 2 at x < 0: `erfc(|x|)`, read by
    /// its bound (`special::erfc_bound`).
    fn note_erfc_distance(&mut self, x: &BigFloat) {
        if !self.saturate {
            let d = special::erfc_bound(&x.abs(), 64, &mut self.cc);
            self.note_distance(&d);
        }
    }

    /// The symbolic-cancellation snap of a trig output, relative to what its argument predicts
    /// (see the contract judge's `_snap`).
    fn snap_trig(&mut self, v: BigFloat, arg: &BigFloat) -> Val {
        if v.is_zero() {
            return zero();
        }
        let a = v.abs();
        let mut predicted = one(self.p);
        let aa = arg.abs();
        if cmp_lt(&aa, &predicted) {
            predicted = aa;
        }
        let floor = predicted.mul(&self.floor, self.p, RM);
        if cmp_lt(&a, &floor) {
            self.snapped = true;
            return zero();
        }
        if cmp_gt(&a, &self.ceil) {
            self.snapped = true;
            return undef();
        }
        approx(v)
    }
}

fn zero() -> Val {
    Val::Exact(Exact::Rat(Rat::ZERO))
}

fn undef() -> Val {
    Val::Exact(Exact::Undef)
}

fn rat(r: Rat) -> Val {
    Val::Exact(Exact::Rat(r))
}

fn int(n: i128) -> Val {
    rat(Rat::int(n))
}

fn pi_times(q: Rat) -> Val {
    if q.is_zero() {
        zero()
    } else {
        Val::Exact(Exact::Pi(q))
    }
}

fn inf(positive: bool) -> Val {
    Val::Exact(if positive {
        Exact::PosInf
    } else {
        Exact::NegInf
    })
}

/// An approximation as a value: NaN and the infinities become exact, and the sign of a zero is
/// erased (one zero).
fn approx(b: BigFloat) -> Val {
    if b.is_nan() {
        undef()
    } else if b.is_inf() {
        inf(!b.is_negative())
    } else if b.is_zero() {
        zero()
    } else {
        Val::Approx(b)
    }
}

fn is_undef(v: &Val) -> bool {
    matches!(v, Val::Exact(Exact::Undef))
}

/// `Some(true)` for +inf, `Some(false)` for -inf.
fn inf_sign(v: &Val) -> Option<bool> {
    match v {
        Val::Exact(Exact::PosInf) => Some(true),
        Val::Exact(Exact::NegInf) => Some(false),
        _ => None,
    }
}

fn is_inf(v: &Val) -> bool {
    inf_sign(v).is_some()
}

fn is_zero(v: &Val) -> bool {
    match v {
        Val::Exact(Exact::Rat(r)) => r.is_zero(),
        Val::Approx(b) => b.is_zero(),
        _ => false,
    }
}

/// The sign of a defined value: -1, 0 or 1.
fn sign(v: &Val) -> i32 {
    match v {
        Val::Exact(Exact::Rat(r)) | Val::Exact(Exact::Pi(r)) => r.signum(),
        Val::Exact(Exact::PosInf) => 1,
        Val::Exact(Exact::NegInf) => -1,
        Val::Exact(Exact::Undef) => 0,
        Val::Approx(b) => {
            if b.is_zero() {
                0
            } else if b.is_negative() {
                -1
            } else {
                1
            }
        }
    }
}

fn exact_rat(v: &Val) -> Option<&Rat> {
    match v {
        Val::Exact(Exact::Rat(r)) => Some(r),
        _ => None,
    }
}

/// Is a finite value an integer, as far as the table can tell? Exact: exactly; an
/// approximation: integral at working precision.
fn is_integer(v: &Val) -> bool {
    match v {
        Val::Exact(Exact::Rat(r)) => r.is_integer(),
        Val::Approx(b) => b.is_int(),
        _ => false,
    }
}

/// The parity of an integer value: `Some(true)` odd, `None` beyond what can be told (an
/// approximation past 1e15, whose integrality is not a fact).
fn odd(v: &Val, ctx: &mut Ctx) -> Option<bool> {
    match v {
        Val::Exact(Exact::Rat(r)) => r.parity(),
        Val::Approx(b) => {
            if cmp_gt(&b.abs(), &crate::hiprec::bf(1e15, ctx.p)) || !b.is_int() {
                return None;
            }
            let h = b.div(&small_int(2, ctx.p), ctx.p + 64, RM);
            Some(!h.is_int())
        }
        _ => None,
    }
}

/// Compare two finite values: `a < b` -> -1.
fn compare(a: &Val, b: &Val, ctx: &mut Ctx) -> i32 {
    if let (Some(x), Some(y)) = (exact_rat(a), exact_rat(b)) {
        return match x.cmp_exact(y) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        };
    }
    if let (Val::Exact(Exact::Pi(x)), Val::Exact(Exact::Pi(y))) = (a, b) {
        return match x.cmp_exact(y) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        };
    }
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    if cmp_lt(&x, &y) {
        -1
    } else if cmp_gt(&x, &y) {
        1
    } else {
        0
    }
}

/// Are two finite values closer than the working precision can separate (32 bits of guard
/// below it)? Every approximation carries an error there, so their order is no proof.
fn indistinct(a: &Val, b: &Val, ctx: &mut Ctx) -> bool {
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    if x.is_nan() || x.is_inf() || y.is_nan() || y.is_inf() {
        return false;
    }
    let d = x.sub(&y, ctx.p, RM);
    if d.is_zero() {
        return true;
    }
    let scale = match (x.exponent(), y.exponent()) {
        (Some(ex), Some(ey)) => i64::from(ex.max(ey)),
        (Some(e), None) | (None, Some(e)) => i64::from(e),
        (None, None) => return true,
    };
    d.exponent()
        .is_none_or(|ed| scale - i64::from(ed) > ctx.p as i64 - 32)
}

/// `r` reduced modulo `m` into [0, m).
fn rat_mod(r: &Rat, m: &Rat) -> Option<Rat> {
    let q = r.checked_mul(&m.checked_inv()?)?;
    let (p, d) = q.big_parts();
    let fl = num_integer::Integer::div_floor(&p, &d);
    let k = Rat::from_big(fl, BigInt::one())?;
    r.checked_add(&k.checked_mul(m)?.checked_neg()?)
}

// ------------------------------------------------------------------------- operations

/// `None` is Unresolved: the point cannot be judged honestly at this precision.
type Out = Option<Val>;

fn add(a: &Val, b: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(a) || is_undef(b) {
        return Some(undef());
    }
    match (inf_sign(a), inf_sign(b)) {
        (Some(x), Some(y)) => return Some(if x == y { inf(x) } else { undef() }),
        (Some(x), None) | (None, Some(x)) => return Some(inf(x)),
        _ => {}
    }
    match (a, b) {
        (Val::Exact(Exact::Rat(x)), Val::Exact(Exact::Rat(y))) => {
            if let Some(s) = x.checked_add(y) {
                return Some(rat(s));
            }
        }
        (Val::Exact(Exact::Pi(x)), Val::Exact(Exact::Pi(y))) => {
            if let Some(s) = x.checked_add(y) {
                return Some(pi_times(s));
            }
        }
        _ => {}
    }
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    Some(approx(x.add(&y, ctx.p, RM)))
}

fn neg(a: &Val) -> Val {
    match a {
        Val::Exact(Exact::Rat(r)) => r.checked_neg().map(rat).unwrap_or_else(|| a.clone()),
        Val::Exact(Exact::Pi(r)) => r.checked_neg().map(pi_times).unwrap_or_else(|| a.clone()),
        Val::Exact(Exact::PosInf) => inf(false),
        Val::Exact(Exact::NegInf) => inf(true),
        Val::Exact(Exact::Undef) => undef(),
        Val::Approx(b) => approx(b.neg()),
    }
}

fn mul(a: &Val, b: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(a) || is_undef(b) {
        return Some(undef());
    }
    if is_inf(a) || is_inf(b) {
        let (sa, sb) = (sign(a), sign(b));
        if sa == 0 || sb == 0 {
            return Some(undef()); // 0 * inf
        }
        return Some(inf(sa * sb > 0));
    }
    match (a, b) {
        (Val::Exact(Exact::Rat(x)), Val::Exact(Exact::Rat(y))) => {
            if let Some(s) = x.checked_mul(y) {
                return Some(rat(s));
            }
        }
        (Val::Exact(Exact::Rat(x)), Val::Exact(Exact::Pi(y)))
        | (Val::Exact(Exact::Pi(y)), Val::Exact(Exact::Rat(x))) => {
            if let Some(s) = x.checked_mul(y) {
                return Some(pi_times(s));
            }
        }
        _ => {}
    }
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    Some(approx(x.mul(&y, ctx.p, RM)))
}

/// Division. A zero divisor is undefined: the quotient tends to +inf on one side of it and to
/// -inf on the other (V4).
fn div(a: &Val, b: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(a) || is_undef(b) || is_zero(b) {
        return Some(undef());
    }
    match (inf_sign(a), inf_sign(b)) {
        (Some(_), Some(_)) => return Some(undef()),
        (None, Some(_)) => return Some(zero()),
        (Some(x), None) => return Some(inf(x == (sign(b) > 0))),
        _ => {}
    }
    match (a, b) {
        (Val::Exact(Exact::Rat(x)), Val::Exact(Exact::Rat(y))) => {
            if let Some(s) = y.checked_inv().and_then(|i| x.checked_mul(&i)) {
                return Some(rat(s));
            }
        }
        (Val::Exact(Exact::Pi(x)), Val::Exact(Exact::Rat(y))) => {
            if let Some(s) = y.checked_inv().and_then(|i| x.checked_mul(&i)) {
                return Some(pi_times(s));
            }
        }
        (Val::Exact(Exact::Pi(x)), Val::Exact(Exact::Pi(y))) => {
            if let Some(s) = y.checked_inv().and_then(|i| x.checked_mul(&i)) {
                return Some(rat(s));
            }
        }
        _ => {}
    }
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    Some(approx(x.div(&y, ctx.p, RM)))
}

fn abs_val(a: &Val) -> Val {
    if sign(a) < 0 {
        neg(a)
    } else {
        a.clone()
    }
}

/// `a^b` at the real numbers, by the limit rule at its boundaries (V4, V5, V6).
fn pow(a: &Val, b: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(a) || is_undef(b) {
        return Some(undef());
    }
    // x^0 = 1 has a value at every finite x; at -inf the base admits only integer exponents and
    // near 0 the only one is 0, so the limit is 1; inf^0 is an indeterminate form
    if is_zero(b) {
        return Some(if inf_sign(a) == Some(true) {
            undef()
        } else {
            int(1)
        });
    }
    // 1^y = 1 for finite y; 1^inf is indeterminate
    if exact_rat(a).is_some_and(|r| r.is_one()) {
        return Some(if is_inf(b) { undef() } else { int(1) });
    }
    if let Some(up) = inf_sign(b) {
        if let Some(apos) = inf_sign(a) {
            return Some(match (apos, up) {
                (true, true) => inf(true),
                (true, false) => zero(),
                (false, true) => undef(), // (-inf)^inf: the sign alternates along the integers
                (false, false) => zero(),
            });
        }
        if is_zero(a) {
            // 0^inf = 0; 0^-inf: +inf over positive bases, +-inf alternating over negative ones
            return Some(if up { zero() } else { undef() });
        }
        let mag = compare(&abs_val(a), &int(1), ctx);
        if mag == 0 {
            return Some(undef()); // (-1)^inf
        }
        let negative = sign(a) < 0;
        let small = mag < 0;
        return Some(match (small, up) {
            (true, true) | (false, false) => zero(),
            (true, false) | (false, true) => {
                if negative {
                    undef() // the magnitude grows and the sign alternates
                } else {
                    inf(true)
                }
            }
        });
    }
    if let Some(apos) = inf_sign(a) {
        // b finite nonzero
        if apos {
            return Some(if sign(b) > 0 { inf(true) } else { zero() });
        }
        if !is_integer(b) {
            return Some(undef());
        }
        if sign(b) < 0 {
            return Some(zero());
        }
        return Some(match odd(b, ctx) {
            Some(true) => inf(false),
            Some(false) => inf(true),
            None => return None,
        });
    }
    if is_zero(a) {
        // a negative power of zero is a division by zero: undefined however it is written, so
        // `pow(0, -2)` and `1/pow(0, 2)` agree
        return Some(if sign(b) > 0 { zero() } else { undef() });
    }
    if sign(a) < 0 {
        if let Val::Approx(e) = b {
            if cmp_gt(&e.abs(), &crate::hiprec::bf(1e15, ctx.p)) {
                return None; // integrality is unknowable beyond the cap
            }
        }
        if !is_integer(b) {
            return Some(undef()); // a negative base with a non-integer exponent
        }
        let o = odd(b, ctx)?;
        let m = pow(&abs_val(a), b, ctx)?;
        return Some(if o { neg(&m) } else { m });
    }
    // a > 0 finite, b finite nonzero
    if let (Some(x), Some(y)) = (exact_rat(a), exact_rat(b)) {
        if y.is_integer() {
            if let Some(n) = y.small_int().filter(|n| n.abs() <= 4096) {
                if let Some(v) = x.checked_pow_int(n) {
                    return Some(rat(v));
                }
            }
        } else if let Some(v) = rat_pow_frac(x, y) {
            return Some(rat(v));
        }
    }
    if let Val::Approx(e) = b {
        if cmp_gt(&e.abs(), &crate::hiprec::bf(1e6, ctx.p)) {
            return None;
        }
    }
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    if cmp_gt(&y.abs(), &crate::hiprec::bf(1e6, ctx.p)) && !y.is_int() {
        return None;
    }
    // A power past the working float's exponent range (`2^(1e40)`) is finite and unreadable,
    // never an infinity: Unresolved.
    let bits = x
        .exponent()
        .map_or(0, |e| i64::from(e).unsigned_abs())
        .max(1) as f64;
    let y_mag = bf_to_f64(&y.abs());
    if !y_mag.is_finite() || y_mag * bits > 1e9 {
        return None;
    }
    Some(approx(pow_finite_pos(&x, &y, ctx.p, &mut ctx.cc)))
}

/// `x^(p/q)` exactly, for a non-negative rational `x` with a rational `q`-th root.
fn rat_pow_frac(x: &Rat, y: &Rat) -> Option<Rat> {
    let (p, q) = y.small_parts()?;
    if q > 64 || p.abs() > 4096 {
        return None;
    }
    x.checked_root(q)?.checked_pow_int(p)
}

/// `rootn(a, n)`: the real n-th root for an integer index -- the signed root for an odd index,
/// the principal root (NaN on negatives) for an even one, the reciprocal for a negative index --
/// undefined for an index that is zero or no finite integer, and at a zero radicand under a
/// negative index (a division by zero).
fn rootn(a: &Val, n: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(a) || is_undef(n) || is_inf(n) || is_zero(n) || !is_integer(n) {
        return Some(undef());
    }
    if let Val::Approx(e) = n {
        if cmp_gt(&e.abs(), &crate::hiprec::bf(1e15, ctx.p)) {
            return Some(undef());
        }
    }
    let negative_index = sign(n) < 0;
    let k = abs_val(n);
    let odd_k = odd(&k, ctx)?;
    if exact_rat(&k).is_some_and(|r| r.is_one()) {
        return if negative_index {
            div(&int(1), a, ctx)
        } else {
            Some(a.clone())
        };
    }
    if negative_index && is_zero(a) {
        return Some(undef());
    }
    let root = if let Some(up) = inf_sign(a) {
        if up || odd_k {
            inf(up)
        } else {
            undef()
        }
    } else if sign(a) < 0 && !odd_k {
        undef()
    } else if is_zero(a) {
        zero()
    } else {
        let exact = match (exact_rat(a), exact_rat(&k)) {
            (Some(x), Some(kk)) => kk
                .small_int()
                .and_then(|kn| x.abs().checked_root(kn))
                .and_then(|r| {
                    if sign(a) < 0 {
                        r.checked_neg()
                    } else {
                        Some(r)
                    }
                }),
            _ => None,
        };
        match exact {
            Some(r) => rat(r),
            None => {
                let x = ctx.bf(a);
                let kb = ctx.bf(&k);
                let r = one(ctx.p).div(&kb, ctx.p, RM);
                let m = pow_finite_pos(&x.abs(), &r, ctx.p, &mut ctx.cc);
                approx(if x.is_negative() { m.neg() } else { m })
            }
        }
    };
    if negative_index {
        div(&int(1), &root, ctx)
    } else {
        Some(root)
    }
}

/// Is `v` within the honesty band of `target` (a nonzero bound)?
fn in_band(v: &BigFloat, target: &BigFloat, ctx: &Ctx) -> bool {
    let d = v.sub(target, ctx.p, RM).abs();
    cmp_lt(&d, &ctx.floor)
}

/// The rational multiple of pi a value is, when it is one.
fn pi_part(v: &Val) -> Option<Rat> {
    match v {
        Val::Exact(Exact::Pi(q)) => Some(q.clone()),
        Val::Exact(Exact::Rat(r)) if r.is_zero() => Some(Rat::ZERO),
        _ => None,
    }
}

fn r(p: i128, q: i128) -> Rat {
    Rat::new(p, q).expect("a small rational")
}

/// `sin(q pi)` exactly where it is rational.
fn sin_pi(q: &Rat) -> Option<Val> {
    let x = rat_mod(q, &Rat::int(2))?;
    let table: [(Rat, i128, i128); 8] = [
        (r(0, 1), 0, 1),
        (r(1, 6), 1, 2),
        (r(1, 2), 1, 1),
        (r(5, 6), 1, 2),
        (r(1, 1), 0, 1),
        (r(7, 6), -1, 2),
        (r(3, 2), -1, 1),
        (r(11, 6), -1, 2),
    ];
    table
        .iter()
        .find(|(k, _, _)| *k == x)
        .map(|(_, p, d)| rat(r(*p, *d)))
}

fn unary(name: &str, a: &Val, ctx: &mut Ctx) -> Option<Out> {
    if is_undef(a) {
        return Some(Some(undef()));
    }
    let p = ctx.p;
    let out: Out = match name {
        "neg" => Some(neg(a)),
        "abs" => Some(abs_val(a)),
        "inv" => div(&int(1), a, ctx),
        "mult2" | "mult3" | "mult4" | "mult5" => {
            mul(a, &int((name.as_bytes()[4] - b'0') as i128), ctx)
        }
        "div2" | "div3" | "div4" | "div5" => div(a, &int((name.as_bytes()[3] - b'0') as i128), ctx),
        "pow2" | "pow3" | "pow4" | "pow5" => pow(a, &int((name.as_bytes()[3] - b'0') as i128), ctx),
        "pow1_2" => pow(a, &rat(r(1, 2)), ctx),
        "pow1_4" => pow(a, &rat(r(1, 4)), ctx),
        "pow1_3" => rootn(a, &int(3), ctx),
        "pow1_5" => rootn(a, &int(5), ctx),
        "exp" => match inf_sign(a) {
            Some(true) => Some(inf(true)),
            Some(false) => Some(zero()),
            None if is_zero(a) => Some(int(1)),
            None => {
                let x = ctx.bf(a);
                if !cmp_lt(&x.abs(), &crate::hiprec::bf(1e5, p)) {
                    None
                } else {
                    Some(approx(x.exp(p, RM, &mut ctx.cc)))
                }
            }
        },
        "log" => match inf_sign(a) {
            Some(true) => Some(inf(true)),
            Some(false) => Some(undef()),
            None if is_zero(a) => Some(inf(false)),
            None if sign(a) < 0 => Some(undef()),
            None if exact_rat(a).is_some_and(|r| r.is_one()) => Some(zero()),
            None => {
                let x = ctx.bf(a);
                Some(approx(x.ln(p, RM, &mut ctx.cc)))
            }
        },
        "sin" | "cos" | "tan" => {
            if is_inf(a) {
                Some(undef())
            } else if let Some(q) = pi_part(a) {
                match name {
                    "sin" => sin_pi(&q)
                        .map(Some)
                        .unwrap_or_else(|| trig_approx(name, a, ctx)),
                    "cos" => {
                        let shifted = q.checked_add(&r(1, 2));
                        shifted
                            .as_ref()
                            .and_then(sin_pi)
                            .map(Some)
                            .unwrap_or_else(|| trig_approx(name, a, ctx))
                    }
                    _ => {
                        let x = rat_mod(&q, &Rat::int(1));
                        match x {
                            Some(x) if x.is_zero() => Some(zero()),
                            Some(x) if x == r(1, 4) => Some(int(1)),
                            Some(x) if x == r(3, 4) => Some(int(-1)),
                            Some(x) if x == r(1, 2) => Some(undef()), // a two-sided pole
                            _ => trig_approx(name, a, ctx),
                        }
                    }
                }
            } else {
                trig_approx(name, a, ctx)
            }
        }
        "asin" | "acos" => {
            if is_inf(a) {
                Some(undef())
            } else if let Some(x) = exact_rat(a) {
                let found = [
                    (r(-1, 1), r(-1, 2), r(1, 1)),
                    (r(-1, 2), r(-1, 6), r(2, 3)),
                    (r(0, 1), r(0, 1), r(1, 2)),
                    (r(1, 2), r(1, 6), r(1, 3)),
                    (r(1, 1), r(1, 2), r(0, 1)),
                ]
                .into_iter()
                .find(|(k, _, _)| k == x);
                if let Some((_, s, c)) = found {
                    Some(pi_times(if name == "asin" { s } else { c }))
                } else if x.abs().cmp_int(1) == std::cmp::Ordering::Greater {
                    Some(undef())
                } else {
                    asin_approx(name, a, ctx)
                }
            } else {
                asin_approx(name, a, ctx)
            }
        }
        "atan" => match inf_sign(a) {
            Some(up) => Some(pi_times(r(if up { 1 } else { -1 }, 2))),
            None => {
                if let Some(x) = exact_rat(a) {
                    if x.is_zero() {
                        return Some(Some(zero()));
                    }
                    if x.abs().is_one() {
                        return Some(Some(pi_times(r(x.signum() as i128, 4))));
                    }
                }
                let x = ctx.bf(a);
                let v = x.atan(p, RM, &mut ctx.cc);
                let half_pi = ctx.pi().div(&small_int(2, p), p, RM);
                if in_band(&v.abs(), &half_pi, ctx) && !ctx.saturate {
                    None
                } else {
                    Some(approx(v))
                }
            }
        },
        "sinh" | "cosh" => match inf_sign(a) {
            Some(up) => Some(if name == "sinh" { inf(up) } else { inf(true) }),
            None if is_zero(a) => Some(if name == "sinh" { zero() } else { int(1) }),
            None => {
                let x = ctx.bf(a);
                if !cmp_lt(&x.abs(), &crate::hiprec::bf(1e5, p)) {
                    None
                } else if name == "sinh" {
                    Some(approx(x.sinh(p, RM, &mut ctx.cc)))
                } else {
                    Some(approx(x.cosh(p, RM, &mut ctx.cc)))
                }
            }
        },
        "tanh" => match inf_sign(a) {
            Some(up) => Some(int(if up { 1 } else { -1 })),
            None if is_zero(a) => Some(zero()),
            None => {
                let x = ctx.bf(a);
                let t = x.tanh(p, RM, &mut ctx.cc);
                if in_band(&t.abs(), &one(p), ctx) && !ctx.saturate {
                    // 1 - |tanh x| = 2 / (e^(2|x|) + 1)
                    let w = 64;
                    let e = x.abs().mul(&small_int(2, w), w, RM).exp(w, RM, &mut ctx.cc);
                    let d = small_int(2, w).div(&e.add(&one(w), w, RM), w, RM);
                    ctx.note_distance(&d);
                    None
                } else {
                    Some(approx(t))
                }
            }
        },
        "asinh" => match inf_sign(a) {
            Some(up) => Some(inf(up)),
            None if is_zero(a) => Some(zero()),
            None => {
                let x = ctx.bf(a);
                Some(approx(x.asinh(p, RM, &mut ctx.cc)))
            }
        },
        "acosh" => match inf_sign(a) {
            Some(true) => Some(inf(true)),
            Some(false) => Some(undef()),
            None => {
                if let Some(x) = exact_rat(a) {
                    if x.is_one() {
                        return Some(Some(zero()));
                    }
                    if x.cmp_int(1) == std::cmp::Ordering::Less {
                        return Some(Some(undef()));
                    }
                }
                let x = ctx.bf(a);
                if in_band(&x, &one(p), ctx) {
                    None
                } else if cmp_lt(&x, &one(p)) {
                    Some(undef())
                } else {
                    Some(approx(x.acosh(p, RM, &mut ctx.cc)))
                }
            }
        },
        "atanh" => {
            if is_inf(a) {
                Some(undef())
            } else if let Some(x) = exact_rat(a) {
                if x.is_zero() {
                    Some(zero())
                } else if x.abs().is_one() {
                    Some(inf(x.signum() > 0)) // only the inner side is in the domain
                } else if x.abs().cmp_int(1) == std::cmp::Ordering::Greater {
                    Some(undef())
                } else {
                    let v = ctx.bf(a);
                    Some(approx(v.atanh(p, RM, &mut ctx.cc)))
                }
            } else {
                let x = ctx.bf(a);
                if in_band(&x.abs(), &one(p), ctx) {
                    None
                } else if cmp_gt(&x.abs(), &one(p)) {
                    Some(undef())
                } else {
                    Some(approx(x.atanh(p, RM, &mut ctx.cc)))
                }
            }
        }
        "floor" | "ceil" => step(name == "floor", a, ctx),
        "erf" => match inf_sign(a) {
            Some(up) => Some(int(if up { 1 } else { -1 })),
            None if is_zero(a) => Some(zero()),
            None => {
                let x = ctx.bf(a);
                if cmp_gt(&x.abs(), &crate::hiprec::bf(1e5, p))
                    || special::erfc_below(&x.abs(), &ctx.floor, p, &mut ctx.cc)
                {
                    // past the band erf is +-1 to every digit the reading carries; its distance
                    // to the bound is erfc(|x|)
                    ctx.note_erfc_distance(&x);
                    ctx.saturate.then(|| {
                        let l = one(p);
                        approx(if x.is_negative() { l.neg() } else { l })
                    })
                } else {
                    match special::erf(&x, p, &mut ctx.cc) {
                        Some(v) if !in_band(&v.abs(), &one(p), ctx) || ctx.saturate => {
                            Some(approx(v))
                        }
                        Some(_) => {
                            ctx.note_erfc_distance(&x);
                            None
                        }
                        None => None,
                    }
                }
            }
        },
        // erfc's limits are 0 and 2 (the limit rule); toward +inf it is a tiny value read to its
        // last place (Unresolved past `ERFC_X_CAP`), toward -inf it nears 2 from below, erf's -1
        // seen from `erfc = 1 - erf`, and inside the band of 2 the value is lost
        "erfc" => match inf_sign(a) {
            Some(up) => Some(int(if up { 0 } else { 2 })),
            None if is_zero(a) => Some(int(1)),
            None => {
                let x = ctx.bf(a);
                let two = small_int(2, p);
                if x.is_negative()
                    && (cmp_gt(&x.abs(), &crate::hiprec::bf(special::ERFC_X_CAP, p))
                        || special::erfc_below(&x.abs(), &ctx.floor, p, &mut ctx.cc))
                {
                    // its distance to 2 is erfc(|x|)
                    ctx.note_erfc_distance(&x);
                    ctx.saturate.then(|| approx(two))
                } else {
                    match special::erfc(&x, p, &mut ctx.cc) {
                        Some(v) if !in_band(&v, &two, ctx) || ctx.saturate => Some(approx(v)),
                        Some(_) => {
                            ctx.note_erfc_distance(&x);
                            None
                        }
                        None => None,
                    }
                }
            }
        },
        "harmonic" => harmonic(a, ctx),
        _ => return None,
    };
    Some(out)
}

fn trig_approx(name: &str, a: &Val, ctx: &mut Ctx) -> Out {
    let p = ctx.p;
    let x = ctx.bf(a);
    // The argument carries `dps` significant digits: past 10^(dps-10) its uncertainty reaches
    // the period, and the point is Unresolved at this precision (the step heads' resolution
    // limit). Below it the reduction is exact to the working precision (astro-float reduces by
    // pi at the argument's own magnitude), so `sin(5e14)` is decided.
    if !cmp_lt(&x.abs(), &ctx.ceil) {
        return None;
    }
    match name {
        "tan" => {
            // pole PROXIMITY, not output magnitude: where cos vanishes at working precision
            let c = x.cos(p, RM, &mut ctx.cc);
            if cmp_lt(&c.abs(), &ctx.floor) {
                ctx.snapped = true;
                return Some(undef());
            }
            let s = x.sin(p, RM, &mut ctx.cc);
            Some(ctx.snap_trig(s.div(&c, p, RM), &x))
        }
        "sin" => {
            let v = x.sin(p, RM, &mut ctx.cc);
            Some(ctx.snap_trig(v, &x))
        }
        _ => {
            let v = x.cos(p, RM, &mut ctx.cc);
            Some(ctx.snap_trig(v, &x))
        }
    }
}

fn asin_approx(name: &str, a: &Val, ctx: &mut Ctx) -> Out {
    let p = ctx.p;
    let x = ctx.bf(a);
    let ax = x.abs();
    if cmp_gt(&ax, &one(p)) {
        return if in_band(&ax, &one(p), ctx) {
            None
        } else {
            Some(undef())
        };
    }
    Some(approx(if name == "asin" {
        x.asin(p, RM, &mut ctx.cc)
    } else {
        x.acos(p, RM, &mut ctx.cc)
    }))
}

/// `floor` / `ceil`: exact on an exact value; on an approximation, decided only outside the band
/// of every integer and inside the working resolution. An approximation in an integer's band,
/// ON the integer included, is undecided: one that rounds onto an integer at the working
/// precision may lie on either side of it (`tanh(400)` is the fifty-digit 1.0 and below 1, so
/// its floor is 0), and a finer rung decides it or the caller fails closed.
fn step(floor: bool, a: &Val, ctx: &mut Ctx) -> Out {
    if let Some(up) = inf_sign(a) {
        return Some(inf(up));
    }
    if let Some(x) = exact_rat(a) {
        let (p, q) = x.big_parts();
        let v = if floor {
            num_integer::Integer::div_floor(&p, &q)
        } else {
            -num_integer::Integer::div_floor(&-p, &q)
        };
        return Rat::from_big(v, BigInt::one()).map(rat).or_else(|| {
            let b = ctx.bf(a);
            Some(approx(if floor { b.floor() } else { b.ceil() }))
        });
    }
    let x = ctx.bf(a);
    if !cmp_lt(&x.abs(), &ctx.ceil) {
        return None;
    }
    let nearest = x.round(0, RM);
    let d = x.sub(&nearest, ctx.p, RM).abs();
    let scale = if cmp_gt(&x.abs(), &one(ctx.p)) {
        x.abs()
    } else {
        one(ctx.p)
    };
    if cmp_lt(&d, &scale.mul(&ctx.floor, ctx.p, RM)) {
        return None;
    }
    let v = if floor { x.floor() } else { x.ceil() };
    Some(match bf_integer(&v) {
        Some(n) => Rat::from_big(n, BigInt::one())
            .map(rat)
            .unwrap_or_else(|| approx(v)),
        None => approx(v),
    })
}

/// An integer-valued BigFloat as a BigInt (its decimal reading).
fn bf_integer(v: &BigFloat) -> Option<BigInt> {
    if !v.is_int() || v.is_inf() || v.is_nan() {
        return None;
    }
    if v.is_zero() {
        return Some(BigInt::zero());
    }
    let s = format!("{v}");
    // astro-float prints `1.5e+3`-style decimals; read the mantissa and exponent exactly
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i64>().ok()?),
        None => (s.as_str(), 0),
    };
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let digits: String = ip.chars().chain(fp.chars()).collect();
    let shift = exp - fp.len() as i64;
    let mut n: BigInt = digits.parse().ok()?;
    if shift >= 0 {
        n *= num_traits::pow(BigInt::from(10), shift as usize);
    } else {
        let d = num_traits::pow(BigInt::from(10), (-shift) as usize);
        if !(n.clone() % &d).is_zero() {
            return None;
        }
        n /= d;
    }
    Some(if neg { -n } else { n })
}

/// The harmonic number H_n of a nonnegative integer `n` as the exact rational it is, while it fits
/// the exact form (`None` past that, or past n = 64).
fn harmonic_exact(n: &Rat) -> Option<Rat> {
    let (p, q) = n.big_parts();
    if q != BigInt::one() {
        return None;
    }
    let n = p.to_i64()?;
    if !(0..=64).contains(&n) {
        return None;
    }
    let mut h = Rat::ZERO;
    for k in 1..=n {
        h = h.checked_add(&Rat::new(1, k as i128)?)?;
    }
    Some(h)
}

fn harmonic(a: &Val, ctx: &mut Ctx) -> Out {
    match inf_sign(a) {
        Some(true) => return Some(inf(true)),
        Some(false) => return Some(undef()),
        None => {}
    }
    if is_zero(a) {
        return Some(zero());
    }
    if let Some(x) = exact_rat(a) {
        if x.is_negative() && x.is_integer() {
            return Some(undef()); // the one-sided limits at a pole differ
        }
        // at a positive integer the value is exact, 1 + 1/2 + ... + 1/n: `harmonic(1)` is 1, so
        // `atanh(harmonic(ceil(x)))` meets atanh's pole at 1 exactly where ceil(x) = 1, as the
        // judge reads it, where an approximation of 1 is unreadable at every rung
        if x.is_integer() && !x.is_negative() {
            if let Some(h) = harmonic_exact(x) {
                return Some(rat(h));
            }
        }
    }
    if ctx.dps > 2000 {
        return None;
    }
    let x = ctx.bf(a);
    if x.is_negative() {
        if !cmp_lt(&x.abs(), &ctx.ceil) {
            return None;
        }
        if x.is_int() {
            return Some(undef());
        }
        if cmp_lt(&x, &crate::hiprec::bf(-0.5, ctx.p)) {
            let nearest = x.round(0, RM);
            let d = x.sub(&nearest, ctx.p, RM).abs();
            let scale = x.abs();
            if !d.is_zero() && cmp_lt(&d, &scale.mul(&ctx.floor, ctx.p, RM)) {
                return None;
            }
        }
    }
    special::harmonic(&x, ctx.p, &mut ctx.cc).map(approx)
}

fn binary(name: &str, a: &Val, b: &Val, ctx: &mut Ctx) -> Option<Out> {
    let out: Out = match name {
        "+" => add(a, b, ctx),
        "-" => add(a, &neg(b), ctx),
        "*" => mul(a, b, ctx),
        "/" => div(a, b, ctx),
        "pow" => pow(a, b, ctx),
        "rootn" => rootn(a, b, ctx),
        "max" | "min" => {
            if is_undef(a) || is_undef(b) {
                Some(undef())
            } else {
                let c = match (inf_sign(a), inf_sign(b)) {
                    (Some(x), Some(y)) => (x as i32) - (y as i32),
                    (Some(x), None) => {
                        if x {
                            1
                        } else {
                            -1
                        }
                    }
                    (None, Some(y)) => {
                        if y {
                            -1
                        } else {
                            1
                        }
                    }
                    _ => compare(a, b, ctx),
                };
                let pick_a = if name == "max" { c >= 0 } else { c <= 0 };
                let picked = if pick_a { a } else { b };
                // An exact member picked over an approximation the precision cannot tell
                // apart from it is no proof of the exact value: the result is approximate.
                let exact_pair = matches!((a, b), (Val::Exact(_), Val::Exact(_)));
                if !exact_pair && matches!(picked, Val::Exact(_)) && indistinct(a, b, ctx) {
                    Some(approx(ctx.bf(picked)))
                } else {
                    Some(picked.clone())
                }
            }
        }
        "mod" => modulo(a, b, ctx),
        "atan2" => atan2(a, b, ctx),
        "besselj" => besselj_val(a, b, ctx),
        "bessely" | "besseli" | "besselk" => bessel_val(name, a, b, ctx),
        _ => return None,
    };
    Some(out)
}

/// numpy's floored remainder at the reals: undefined at a zero divisor (a division by zero) and
/// an infinite dividend; against an infinite divisor its limit (the dividend where the signs
/// agree, the divisor's infinity otherwise), undefined at a zero dividend, where the limit taken
/// in both arguments together disagrees (0 along (1/n, n), +inf along (-1/n, n)).
fn modulo(a: &Val, b: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(a) || is_undef(b) || is_zero(b) || is_inf(a) {
        return Some(undef());
    }
    if let Some(bpos) = inf_sign(b) {
        if is_zero(a) {
            return Some(undef());
        }
        return Some(if (sign(a) > 0) == bpos {
            a.clone()
        } else {
            inf(bpos)
        });
    }
    if let (Some(x), Some(y)) = (exact_rat(a), exact_rat(b)) {
        if let Some(m) = rat_mod(x, &y.abs()) {
            // the remainder takes the divisor's sign
            let m = if y.is_negative() && !m.is_zero() {
                m.checked_add(y)
            } else {
                Some(m)
            };
            if let Some(m) = m {
                return Some(rat(m));
            }
        }
    }
    let (x, y) = (ctx.bf(a), ctx.bf(b));
    let yy = y.abs();
    if !cmp_lt(&x.abs(), &yy.mul(&ctx.ceil, ctx.p, RM)) {
        return None;
    }
    let m = special::modulo(&x, &y, ctx.p);
    if m.is_zero() {
        return Some(zero());
    }
    let band = yy.mul(&ctx.floor, ctx.p, RM);
    let ma = m.abs();
    if cmp_lt(&ma, &band) || cmp_lt(&yy.sub(&ma, ctx.p, RM).abs(), &band) {
        return None;
    }
    Some(approx(m))
}

/// `atan2(y, x)`, the principal value in (-pi, pi]: decided exactly on the axes, undefined at
/// the origin (no direction has a value there), and on the negative x axis pi (one zero); an
/// approximate y inside the band of 0 there is Unresolved (its residue's sign picks the branch).
fn atan2(y: &Val, x: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(y) || is_undef(x) {
        return Some(undef());
    }
    let (sy, sx) = (sign(y), sign(x));
    match (inf_sign(y), inf_sign(x)) {
        (Some(yu), Some(xu)) => {
            let q = if xu { r(1, 4) } else { r(3, 4) };
            return Some(pi_times(if yu { q } else { q.checked_neg()? }));
        }
        (Some(yu), None) => return Some(pi_times(r(if yu { 1 } else { -1 }, 2))),
        (None, Some(xu)) => {
            return Some(if xu {
                zero()
            } else if sy >= 0 {
                pi_times(r(1, 1))
            } else {
                pi_times(r(-1, 1))
            })
        }
        _ => {}
    }
    let exact_y = matches!(y, Val::Exact(_));
    if sy == 0 && exact_y {
        return Some(match sx {
            1 => zero(),
            -1 => pi_times(r(1, 1)),
            _ => undef(),
        });
    }
    if sx == 0 && matches!(x, Val::Exact(_)) && sy != 0 {
        return Some(pi_times(r(sy as i128, 2)));
    }
    let (yb, xb) = (ctx.bf(y), ctx.bf(x));
    if yb.is_zero() && !is_pos(&xb) {
        return None;
    }
    if xb.is_negative() && cmp_lt(&yb.abs(), &xb.abs().mul(&ctx.floor, ctx.p, RM)) {
        return None;
    }
    Some(approx(special::atan2(&yb, &xb, ctx.p, &mut ctx.cc)))
}

/// `besselj(nu, x)` at the reals: the limit rule at infinite arguments (J tends to 0 as x grows,
/// and as the order grows at a fixed argument), the kernel's cells elsewhere.
fn besselj_val(nu: &Val, x: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(nu) || is_undef(x) {
        return Some(undef());
    }
    let integer = is_integer(nu);
    if let Some(xu) = inf_sign(x) {
        if is_inf(nu) {
            return Some(undef());
        }
        return Some(if xu || integer { zero() } else { undef() });
    }
    if let Some(nu_up) = inf_sign(nu) {
        return Some(if nu_up { zero() } else { undef() });
    }
    let n = bessel_order(nu, sign(nu) < 0 || sign(x) < 0, ctx)?;
    let xb = ctx.bf(x);
    besselj::besselj(&n, &xb, ctx.p, &mut ctx.cc).map(|v| bessel_value(v, x))
}

/// A Bessel kernel's value as the table carries it. At a zero argument the kernels give the
/// formula's values (`J_0(0) = I_0(0) = 1`, else 0, a pole's infinity or none), which are exact,
/// as `pow(0, 0) = 1` is: `atanh(besseli(0, 0))` is atanh(1) = +inf, whether the order is written
/// or bound. Elsewhere the value is the kernel's approximation.
fn bessel_value(v: BigFloat, x: &Val) -> Val {
    if is_zero(x) && v.is_int() && !v.is_zero() && !v.is_inf() {
        let p = v.precision().unwrap_or(64);
        if !cmp_gt(&v, &one(p)) && !cmp_lt(&v, &one(p)) {
            return int(1);
        }
    }
    approx(v)
}

/// A Bessel order read at the precision its distance `e` to the nearest integer needs: every
/// kernel reads `e`. Near a negative integer -n, J and I swing between the size of `J_n` and
/// that of the term `e` carries (`J_-(n+e)(x)` is about `(-1)^n (J_n(x) + e pi Y_n(x))`, and
/// `Y_n(x)` may exceed `J_n(x)` by hundreds of digits); Y and K at a non-integer order divide by
/// `sin(nu pi)`. So an exact order is read at the working precision plus the bits `e` lies
/// below it (Unresolved past the kernels' `BITS_CAP`), and an approximate one as it stands,
/// except where `fragile` (J and I at a negative order, or at x < 0, where only the integer
/// orders have a value) and it lies within the working band of an integer, where the order's own
/// error decides the value: Unresolved.
fn bessel_order(nu: &Val, fragile: bool, ctx: &mut Ctx) -> Option<BigFloat> {
    match nu {
        Val::Exact(Exact::Rat(r)) if !r.is_integer() => {
            let (p, q) = r.big_parts();
            let two = BigInt::from(2);
            let n = num_integer::Integer::div_floor(&(&p * &two + &q), &(&q * &two));
            let e = &p - &n * &q;
            let below = (p.bits() as usize).saturating_sub(e.bits() as usize);
            let g = ctx.p + 64 + below;
            if g > besselj::BITS_CAP {
                return None;
            }
            let (a, b) = (ctx.big_at(&p, g), ctx.big_at(&q, g));
            Some(a.div(&b, g, RM))
        }
        Val::Approx(b) if fragile => {
            let half = one(64).div(&small_int(2, 64), 64, RM);
            let n = b.add(&half, ctx.p + 64, RM).floor();
            (!in_band(b, &n, ctx)).then(|| b.clone())
        }
        _ => Some(ctx.bf(nu)),
    }
}

/// `bessely`, `besseli`, `besselk` at the reals: the limit rule at an infinite operand, in all
/// arguments together (undefined where both are infinite: the limit depends on the path), the
/// kernels' cells and values at finite ones (`hiprec::bessel`).
///
/// * At `x = +inf`: Y and K tend to 0, I to +inf; at `x = -inf` Y and K have no value, nor I at a
///   non-integer order, and `I_n(-inf) = (-1)^n inf` (the domain near it is the line of that
///   order, as `pow(-inf, 0)`).
/// * At an infinite order and finite x: `Y_+inf = -inf` and `K_+-inf = +inf` where x >= 0 (no
///   value at x < 0); `I_+inf = 0`; `Y_-inf` and `I_-inf` have no value (they oscillate without
///   bound), and an infinite negative order has none at x < 0 either, as for `besselj`.
/// * At x = 0, Y under a negative half-integer order has no value (the kernel's cell), and an
///   approximate order within the working band of one cannot be told from it: Unresolved.
fn bessel_val(name: &str, nu: &Val, x: &Val, ctx: &mut Ctx) -> Out {
    if is_undef(nu) || is_undef(x) {
        return Some(undef());
    }
    match (inf_sign(nu), inf_sign(x)) {
        (Some(_), Some(_)) => return Some(undef()),
        (None, Some(x_up)) => {
            return match name {
                "besseli" if x_up => Some(inf(true)),
                "besseli" if is_integer(nu) => odd(nu, ctx).map(|o| inf(!o)),
                _ if x_up && name != "besseli" => Some(zero()),
                _ => Some(undef()),
            }
        }
        (Some(nu_up), None) => {
            let nonneg = sign(x) >= 0;
            return match name {
                "bessely" => Some(if nu_up && nonneg { inf(false) } else { undef() }),
                "besselk" => Some(if nonneg { inf(true) } else { undef() }),
                _ if nu_up => Some(zero()),
                _ => Some(undef()),
            };
        }
        (None, None) => {}
    }
    // Y at a zero argument has no value at a negative half-integer order and an infinity of either
    // sign beside it: an approximate order within the working band of one is undecided
    if name == "bessely" && is_zero(x) && sign(nu) < 0 {
        if let Val::Approx(b) = nu {
            let half = one(64).div(&small_int(2, 64), 64, RM);
            if in_band(b, &b.floor().add(&half, ctx.p + 64, RM), ctx) {
                return None;
            }
        }
    }
    let fragile = name == "besseli" && (sign(nu) < 0 || sign(x) < 0);
    let n = bessel_order(nu, fragile, ctx)?;
    let xb = ctx.bf(x);
    let p = ctx.p;
    match name {
        "bessely" => bessel::bessely(&n, &xb, p, &mut ctx.cc),
        "besseli" => bessel::besseli(&n, &xb, p, &mut ctx.cc),
        _ => bessel::besselk(&n, &xb, p, &mut ctx.cc),
    }
    .map(|v| bessel_value(v, x))
}

// ---------------------------------------------------------------- compiled expressions

#[derive(Clone, Debug)]
enum Node {
    Op(String, Vec<usize>),
    Var(usize),
    Lit(Leaf, String),
}

/// An expression compiled against the table: its tokens' tree, its free names in order.
pub struct Expr {
    nodes: Vec<Node>,
    root: usize,
    pub names: Vec<String>,
}

/// What a real evaluation found.
pub struct RealOutcome {
    /// `None`: Unresolved at this precision
    pub value: Option<Val>,
    pub snapped: bool,
    pub inf_node: bool,
    pub max_mag: BigFloat,
    pub min_mag: BigFloat,
    /// the smallest distance to its bound of a bounded function left inside its band, zero where
    /// none was (`Ctx::note_distance`)
    pub band: BigFloat,
}

impl Expr {
    /// Compile `tokens` (prefix) with the operator arities `arity`; a token in `names` is a
    /// free name (a slot, a variable, a constant symbol). `Err` names the first token that is
    /// neither an operator, a name nor a literal.
    pub fn compile(
        tokens: &[String],
        names: &[String],
        arity: &std::collections::HashMap<String, u8>,
    ) -> Result<Expr, String> {
        let mut nodes = Vec::new();
        let mut pos = 0usize;
        let root = Self::build(tokens, &mut pos, names, arity, &mut nodes)?;
        if pos != tokens.len() {
            return Err("trailing tokens".into());
        }
        Ok(Expr {
            nodes,
            root,
            names: names.to_vec(),
        })
    }

    fn build(
        tokens: &[String],
        pos: &mut usize,
        names: &[String],
        arity: &std::collections::HashMap<String, u8>,
        nodes: &mut Vec<Node>,
    ) -> Result<usize, String> {
        let tok = tokens.get(*pos).ok_or("truncated expression")?.clone();
        *pos += 1;
        let node = if let Some(&k) = arity.get(&tok) {
            let mut kids = Vec::with_capacity(k as usize);
            for _ in 0..k {
                kids.push(Self::build(tokens, pos, names, arity, nodes)?);
            }
            Node::Op(tok, kids)
        } else if let Some(i) = names.iter().position(|n| *n == tok) {
            Node::Var(i)
        } else {
            let leaf =
                real_leaf(&tok).ok_or_else(|| format!("not an accepted literal: {tok:?}"))?;
            Node::Lit(leaf, tok)
        };
        nodes.push(node);
        Ok(nodes.len() - 1)
    }

    /// The real value at `env` (one value per name), at `dps` decimal digits.
    pub fn eval_real(&self, env: &[Val], dps: i32) -> Option<RealOutcome> {
        self.eval_real_reading(env, dps, false)
    }

    /// `eval_real` in the float-resolution reading (`Ctx::saturate`): a value to round to a
    /// double, never one to compare exactly.
    pub fn eval_float_resolution(&self, env: &[Val], dps: i32) -> Option<RealOutcome> {
        self.eval_real_reading(env, dps, true)
    }

    fn eval_real_reading(&self, env: &[Val], dps: i32, saturate: bool) -> Option<RealOutcome> {
        let mut ctx = Ctx::new(dps)?;
        ctx.saturate = saturate;
        let _exact = number_domain(false);
        let value = self.node_real(self.root, env, &mut ctx);
        Some(RealOutcome {
            value,
            snapped: ctx.snapped,
            inf_node: ctx.inf_node,
            max_mag: ctx.max_mag,
            min_mag: ctx.min_mag,
            band: ctx.band,
        })
    }

    fn node_real(&self, i: usize, env: &[Val], ctx: &mut Ctx) -> Out {
        match &self.nodes[i] {
            // A leaf is an operand like any intermediate: the precision a computation demands is
            // set by the magnitudes it combines, and `1 + 1e-3000` demands 3,000 digits however
            // the 1e-3000 is spelled.
            Node::Var(k) => {
                let v = env.get(*k)?.clone();
                ctx.note_mag(&v, false);
                Some(v)
            }
            Node::Lit(leaf, _) => {
                let v = match leaf {
                    Leaf::Exact(e) => Val::Exact(e.clone()),
                    Leaf::Big(p, q) => approx(ctx.ratio(p, q)),
                    Leaf::Dec(m, k) => approx(ctx.decimal(m, *k)),
                    Leaf::E => approx(ctx.cc.e(ctx.p, RM)),
                };
                ctx.note_mag(&v, false);
                Some(v)
            }
            Node::Op(name, kids) => {
                let a = self.node_real(kids[0], env, ctx)?;
                let v = if kids.len() == 1 {
                    unary(name, &a, ctx)??
                } else {
                    let b = self.node_real(kids[1], env, ctx)?;
                    binary(name, &a, &b, ctx)??
                };
                if is_inf(&v) {
                    ctx.inf_node = true;
                }
                ctx.note_mag(&v, matches!(name.as_str(), "+" | "-" | "mod"));
                Some(v)
            }
        }
    }

    /// The f64 value at `env`, and whether no operation overflowed or underflowed on the way
    /// (the range standard).
    pub fn eval_f64(&self, env: &[f64]) -> Option<(f64, bool)> {
        let mut in_range = true;
        let v = self.node_f64(self.root, env, &mut in_range)?;
        Some((v, in_range))
    }

    fn node_f64(&self, i: usize, env: &[f64], in_range: &mut bool) -> Option<f64> {
        match &self.nodes[i] {
            Node::Var(k) => env.get(*k).copied(),
            Node::Lit(_, tok) => literal_f64(tok),
            Node::Op(name, kids) => {
                let mut args = Vec::with_capacity(kids.len());
                for &k in kids {
                    args.push(self.node_f64(k, env, in_range)?);
                }
                let v = crate::numeric::apply_op(name, &args)?;
                if crate::numeric::range_event(name, &args, v) == Some(true) {
                    *in_range = false;
                }
                Some(v)
            }
        }
    }
}

// ------------------------------------------------------------- point readings

/// THE TABLE's real value of one operation on exact or approximate arguments: `None` for an
/// operator the table does not know, or a point it cannot judge honestly at `ctx`'s precision.
/// The interval layer reads it where a subexpression is one point across a cell
/// (`interval::with_real_reading`).
pub fn apply(name: &str, args: &[Val], ctx: &mut Ctx) -> Option<Val> {
    let _exact = number_domain(false);
    match args {
        [a] => unary(name, a, ctx)?,
        [a, b] => binary(name, a, b, ctx)?,
        _ => None,
    }
}

/// THE TABLE's real value of a literal token at `ctx`'s precision (`None`: no literal).
pub fn literal_val(tok: &str, ctx: &mut Ctx) -> Option<Val> {
    let leaf = real_leaf(tok)?;
    let _exact = number_domain(false);
    Some(match leaf {
        Leaf::Exact(e) => Val::Exact(e),
        Leaf::Big(p, q) => approx(ctx.ratio(&p, &q)),
        Leaf::Dec(m, k) => approx(ctx.decimal(&m, k)),
        Leaf::E => approx(ctx.cc.e(ctx.p, RM)),
    })
}

/// A double as an exact value: a finite double is the dyadic rational it is, an infinity is
/// itself, NaN is undefined.
pub fn val_of_f64(x: f64) -> Val {
    if x.is_nan() {
        return undef();
    }
    if x.is_infinite() {
        return inf(x > 0.0);
    }
    let _exact = number_domain(false);
    match Rat::from_f64_exact(x) {
        Some(r) => rat(r),
        None => approx(BigFloat::from_f64(x, 64)),
    }
}

/// Is the value undefined, `Some(sign)` for an infinity, `None` otherwise.
pub fn class_of(v: &Val) -> (bool, Option<bool>) {
    (is_undef(v), inf_sign(v))
}

/// Is `v` exactly zero (the exact rational 0, `tan(pi)`), not a value computed as zero?
pub fn is_exact_zero(v: &Val) -> bool {
    matches!(v, Val::Exact(Exact::Rat(r)) if r.is_zero())
}

/// A float64 enclosure of a FINITE value: `(lo, hi, lo_open, hi_open)`, a point when the value
/// is a double, otherwise doubles bracketing it (a value beyond the doubles' range is enclosed
/// by the largest double and an open infinite end, a nonzero value below them by zero and the
/// smallest subnormal, zero excluded). `None` for a value that is not finite.
pub fn f64_enclosure(v: &Val, ctx: &mut Ctx) -> Option<(f64, f64, bool, bool)> {
    let _exact = number_domain(false);
    if let Val::Exact(Exact::Rat(r)) = v {
        if r.is_zero() {
            return Some((0.0, 0.0, false, false));
        }
        if let Some((y, other)) = r.float_bracket() {
            return Some(match other {
                Some(o) => (y.min(o), y.max(o), false, false),
                // the largest double has no double past it: a value beyond it is no point
                None if y.abs() == f64::MAX
                    && Rat::from_f64_exact(y).is_none_or(|e| r.cmp_exact(&e).is_ne()) =>
                {
                    if y > 0.0 {
                        (f64::MAX, f64::INFINITY, true, true)
                    } else {
                        (f64::NEG_INFINITY, -f64::MAX, true, true)
                    }
                }
                None => (y, y, false, false),
            });
        }
    }
    let b = match v {
        Val::Exact(Exact::PosInf | Exact::NegInf | Exact::Undef) => return None,
        _ => ctx.bf(v),
    };
    if b.is_nan() || b.is_inf() {
        return None;
    }
    if b.is_zero() {
        return Some((0.0, 0.0, false, false));
    }
    let neg = b.is_negative();
    let y = bf_to_f64(&b.abs());
    let (lo, hi, lo_open, hi_open) = if y.is_infinite() {
        (f64::MAX, f64::INFINITY, false, true)
    } else if b.exponent().is_some_and(|e| e <= -1074) {
        // below 2^-1074, the smallest subnormal: `1e-3000` lies in (0, 5e-324]
        (0.0, f64::from_bits(1), true, false)
    } else if y < 4.0 * f64::MIN_POSITIVE {
        (0.0, 4.0 * f64::MIN_POSITIVE, true, false)
    } else {
        // two units in the last place each way: the reading truncates the mantissa below its
        // top word and rounds once; past the largest double the end is open
        let down = |x: f64| f64::from_bits(x.to_bits() - 2);
        let up = f64::from_bits(y.to_bits() + 2);
        if up.is_finite() {
            (down(y), up, false, false)
        } else {
            (down(y), f64::INFINITY, false, true)
        }
    };
    Some(if neg {
        (-hi, -lo, hi_open, lo_open)
    } else {
        (lo, hi, lo_open, hi_open)
    })
}

/// The double nearest to a FINITE value (a value beyond the doubles' range reads as an
/// infinity, which the caller refuses): exact for a rational with a certified rounding, else
/// the working float's top 128 mantissa bits rounded once to 53. `None` for an infinity or an
/// undefined value. Permissive writes its ground values so (P3).
pub fn nearest_f64(v: &Val, ctx: &mut Ctx) -> Option<f64> {
    let _exact = number_domain(false);
    match v {
        Val::Exact(Exact::PosInf | Exact::NegInf | Exact::Undef) => return None,
        Val::Exact(Exact::Rat(r)) => {
            if r.is_zero() {
                return Some(0.0);
            }
            if let Some(y) = r.to_f64_nearest_certified() {
                return Some(y);
            }
        }
        _ => {}
    }
    let b = ctx.bf(v);
    if b.is_nan() || b.is_inf() {
        return None;
    }
    if b.is_zero() {
        return Some(0.0);
    }
    let neg = b.is_negative();
    let a = b.abs();
    let (Some(words), Some(e)) = (a.mantissa_digits(), a.exponent()) else {
        return None;
    };
    let n = words.len();
    let top = u128::from(*words.last()?);
    let next = if n >= 2 { u128::from(words[n - 2]) } else { 0 };
    // value = 0.m * 2^e = m * 2^(e - 128), the u128 rounded to nearest once
    let mut y = ((top << 64) | next) as f64;
    let mut k = i64::from(e) - 128;
    while k > 0 && y.is_finite() {
        let step = k.min(512);
        y *= 2f64.powi(step as i32);
        k -= step;
    }
    while k < 0 && y > 0.0 {
        let step = (-k).min(512);
        y /= 2f64.powi(step as i32);
        k += step;
    }
    Some(if neg { -y } else { y })
}

/// A positive finite BigFloat read to a double from its top mantissa word (within an ulp or
/// two; `f64_enclosure` widens it).
fn bf_to_f64(b: &BigFloat) -> f64 {
    let (Some(words), Some(e)) = (b.mantissa_digits(), b.exponent()) else {
        return f64::NAN;
    };
    let Some(&top) = words.last() else {
        return 0.0;
    };
    // value = 0.top... * 2^e = top * 2^(e - 64)
    let mut y = top as f64;
    let mut k = i64::from(e) - 64;
    while k > 0 && y.is_finite() {
        let step = k.min(512);
        y *= 2f64.powi(step as i32);
        k -= step;
    }
    while k < 0 && y > 0.0 {
        let step = (-k).min(512);
        y /= 2f64.powi(step as i32);
        k += step;
    }
    y
}

/// THE TABLE's real value of a ground expression (no free name) at `dps` digits, in the
/// float-resolution reading (`Ctx::saturate`): permissive's ground fold, which writes the double
/// nearest to the value (P3). `None` where it holds a token the table cannot read or the point
/// is Unresolved at that precision.
pub fn ground_value_float(
    tokens: &[String],
    ops: &crate::operators::Operators,
    dps: i32,
) -> Option<Val> {
    let arity: std::collections::HashMap<String, u8> = tokens
        .iter()
        .filter_map(|t| ops.arity_of(t).map(|a| (t.clone(), a)))
        .collect();
    let e = Expr::compile(tokens, &[], &arity).ok()?;
    e.eval_real_reading(&[], dps, true)?.value
}

// -------------------------------------------------------------------------- comparison

/// The comparison classes of two values at one point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    /// they agree
    Eq,
    /// the left side is undefined and the right has a value: an extension
    Ext,
    /// the left side has a value and the right is undefined: a loss
    Shrink,
    /// an infinity is involved and the two differ
    InfChange,
    /// both finite and different
    RealChange,
}

impl Cmp {
    pub fn name(self) -> &'static str {
        match self {
            Cmp::Eq => "eq",
            Cmp::Ext => "EXT",
            Cmp::Shrink => "SHRINK",
            Cmp::InfChange => "INF-CHANGE",
            Cmp::RealChange => "REAL-CHANGE",
        }
    }
}

/// The class half of a comparison, `kind` being `"nan"`, `"+inf"`, `"-inf"` or `"fin"` per side:
/// everything undefined or infinite settles here; `None` when both are finite, where the
/// magnitude of the gap decides (real: the precision ladder; f64: `f64_agree`).
pub fn compare_classes(l: &str, r: &str) -> Option<Cmp> {
    match (l, r) {
        ("nan", "nan") => Some(Cmp::Eq),
        ("nan", _) => Some(Cmp::Ext),
        (_, "nan") => Some(Cmp::Shrink),
        ("fin", "fin") => None,
        (a, b) => Some(if a == b { Cmp::Eq } else { Cmp::InfChange }),
    }
}

fn kind_f64(v: f64) -> &'static str {
    if v.is_nan() {
        "nan"
    } else if v == f64::INFINITY {
        "+inf"
    } else if v == f64::NEG_INFINITY {
        "-inf"
    } else {
        "fin"
    }
}

/// THE f64 AGREEMENT (F6): both undefined, the same infinity, or both finite and at most
/// `REALISED_ULP` units in the last place apart, the two zeros counting as one value.
pub fn f64_agree(a: f64, b: f64) -> Cmp {
    if let Some(c) = compare_classes(kind_f64(a), kind_f64(b)) {
        return c;
    }
    if ulp_gap(a, b) <= REALISED_ULP {
        Cmp::Eq
    } else {
        Cmp::RealChange
    }
}

/// The structural f64 comparison: has the deployed algebra DIVERGED (an infinity where a finite
/// value belongs, a gap far outside rounding)? Within `1e-9` of the larger magnitude (at least 1)
/// is no divergence.
pub fn f64_structural(a: f64, b: f64) -> Cmp {
    if let Some(c) = compare_classes(kind_f64(a), kind_f64(b)) {
        return c;
    }
    if (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0) {
        Cmp::Eq
    } else {
        Cmp::RealChange
    }
}

/// The f64 agreement bound, in units in the last place: a rewrite has two sides, each up to about
/// four library calls of at most one unit of error, which can oppose: 2 x 4 x 1.
pub const REALISED_ULP: u64 = 8;

/// THE f64 AGREEMENT OF A FULLY NUMERIC REWRITE (F8, the semantics' 3.3): the same double, the
/// sign of a zero included, both undefined agreeing; no tolerance, and no range standard, since a
/// ground rule's one point is its whole domain and a fully numeric rewrite is bit-exact.
/// `exp(sin(np.pi)) -> 1` is refused: the left side is 1.0000000000000002.
pub fn f64_same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The ordinary working precision of a reading of the reals, in decimal digits
/// (`_contract.BASE_DPS`).
pub const BASE_DPS: i32 = 50;

/// The precision ceiling of a reading of the reals (`_contract.MAX_DPS`): a point that demands
/// more is unread, and readings that agree past it are no equality.
pub const MAX_DPS: i32 = 2000;

/// THE PRECISION A READING DEMANDS, from the magnitudes its operations combine (the judge's
/// ladder rule, `_contract._demanded_dps`): `BASE_DPS + 2k`, `k` the decades of the largest
/// intermediate above 1 or of the smallest nonzero one below 1. A cancellation between magnitudes
/// 10^k apart loses about 2k digits before any correct digit survives: `x + 1e-200` demands 450
/// digits, `1 + 1e-3000` 6,050.
pub fn demanded_dps(max_mag: &BigFloat, min_mag: &BigFloat) -> i64 {
    let log10 = |b: &BigFloat| -> Option<f64> {
        if b.is_zero() || b.is_nan() || b.is_inf() {
            return None;
        }
        let a = b.abs();
        let y = bf_to_f64(&a);
        if y.is_finite() && y > 0.0 {
            return Some(y.log10());
        }
        a.exponent()
            .map(|e| f64::from(e) * std::f64::consts::LOG10_2)
    };
    let mut k = 0i64;
    if let Some(l) = log10(max_mag).filter(|l| *l > 0.0) {
        k = k.max(l.floor() as i64 + 1);
    }
    if let Some(l) = log10(min_mag).filter(|l| *l < 0.0) {
        k = k.max(-(l.floor() as i64));
    }
    i64::from(BASE_DPS) + 2 * k
}

/// THE f64 MAGNITUDE LADDER: the magnitudes, from the smallest normal double to the largest, at
/// which f64's agreement is read along each variable (both signs, the other variables at a generic
/// value), past the reach of a sample grid. Rounding harmless at moderate magnitudes is not
/// harmless at every one: `sinh(asinh(x))` is 66 units in the last place off at 1e100, and
/// `log(exp(x))` and `acosh(cosh(x))` lose their argument at small ones. Where the core keeps one
/// of its own inverse collapses to a safe magnitude (`sinh o asinh` to |t| <= 710, `log o exp` to
/// t >= 1, `acosh o cosh` to |t| >= 1, `atanh o tanh` to |t| <= 2), the ladder runs past the limit.
/// One ladder, read by the judge's f64 lane and the f64 mint (`battery::f64_magnitudes_mint`).
pub const F64_MAGNITUDES: [f64; 21] = [
    2.2250738585072014e-308,
    1e-300,
    1e-200,
    1e-100,
    1e-50,
    1e-16,
    1e-8,
    1e5,
    1e7,
    1e10,
    1e15,
    1e20,
    1e30,
    1e50,
    1e100,
    1e150,
    1e200,
    1e250,
    1e300,
    1e307,
    f64::MAX,
];

/// The units in the last place between two finite doubles of the same sign (the zeros as one);
/// `u64::MAX` across a sign change.
pub fn ulp_gap(a: f64, b: f64) -> u64 {
    let a = if a == 0.0 { 0.0 } else { a };
    let b = if b == 0.0 { 0.0 } else { b };
    if a == b {
        return 0;
    }
    if (a < 0.0) != (b < 0.0) {
        return u64::MAX;
    }
    (a.abs().to_bits() as i64 - b.abs().to_bits() as i64).unsigned_abs()
}

// ------------------------------------------------------------------------------ the sorts

/// The values every sort quantifies over, before its own excusals (`sort_atoms`): the symbolic
/// points where expressions meet their coincidences.
fn real_atoms() -> Vec<Leaf> {
    let q = |p, d| Leaf::Exact(Exact::Rat(r(p, d)));
    let pi = |p, d| Leaf::Exact(Exact::Pi(r(p, d)));
    vec![
        q(0, 1),
        q(1, 2),
        q(-1, 2),
        q(1, 1),
        q(-1, 1),
        q(2, 1),
        q(-2, 1),
        q(3, 1),
        q(-3, 1),
        q(1, 4),
        q(-1, 4),
        q(3, 2),
        q(-3, 2),
        pi(1, 2),
        pi(-1, 2),
        pi(1, 1),
        pi(1, 4),
        pi(1, 3),
        pi(1, 6),
        Leaf::E,
        q(-17, 10),
    ]
}

/// What a sort's atom is to the criterion: `"real"`, a real binding where clause (a) applies;
/// `"atom"`, a non-finite binding in the sort's domain, judged with no excuse; `"null"`, a
/// non-finite or zero binding the sort's certificate puts on a set of measure zero, owed nothing.
pub fn sort_atoms(sort: char, permissive: bool) -> Vec<(Leaf, &'static str)> {
    if permissive {
        // every slot matches any subexpression, and the guarantee covers the real-valued
        // bindings, zero included
        return real_atoms().into_iter().map(|l| (l, "real")).collect();
    }
    let mut out: Vec<(Leaf, &'static str)> = real_atoms()
        .into_iter()
        .filter(|l| sort != '$' || !matches!(l, Leaf::Exact(Exact::Rat(x)) if x.is_zero()))
        .map(|l| (l, "real"))
        .collect();
    let nonfinite = [Exact::PosInf, Exact::NegInf];
    match sort {
        '_' => {
            out.extend(nonfinite.iter().cloned().map(|e| (Leaf::Exact(e), "atom")));
            out.push((Leaf::Exact(Exact::Undef), "atom"));
        }
        '!' => out.extend(nonfinite.iter().cloned().map(|e| (Leaf::Exact(e), "null"))),
        '$' => {
            out.push((Leaf::Exact(Exact::Rat(Rat::ZERO)), "null"));
            out.extend(nonfinite.iter().cloned().map(|e| (Leaf::Exact(e), "null")));
        }
        _ => {}
    }
    out
}

/// A leaf as the interchange spec the Python judge carries: `(kind, a, b)`.
pub fn leaf_spec(l: &Leaf) -> (String, String, String) {
    match l {
        Leaf::Exact(Exact::Rat(x)) => ("q".into(), x.numer_string(), x.denom_string()),
        Leaf::Exact(Exact::Pi(x)) => ("pi".into(), x.numer_string(), x.denom_string()),
        Leaf::Exact(Exact::PosInf) => ("inf".into(), "+".into(), String::new()),
        Leaf::Exact(Exact::NegInf) => ("inf".into(), "-".into(), String::new()),
        Leaf::Exact(Exact::Undef) => ("nan".into(), String::new(), String::new()),
        Leaf::Big(p, q) => ("q".into(), p.to_string(), q.to_string()),
        Leaf::Dec(m, k) => ("d".into(), m.to_string(), k.to_string()),
        Leaf::E => ("e".into(), String::new(), String::new()),
    }
}

/// A value from its interchange spec: `("q", p, q)`, `("pi", p, q)`, `("e", "", "")`,
/// `("inf", "+"|"-", "")`, `("nan", "", "")`, or `("m", man, exp)` for an approximation
/// `man * 2^exp` (`man` a signed hexadecimal integer).
pub fn val_from_spec(kind: &str, a: &str, b: &str, ctx: &mut Ctx) -> Option<Val> {
    let _exact = number_domain(false);
    Some(match kind {
        "q" | "pi" => {
            let p: BigInt = a.parse().ok()?;
            let q: BigInt = b.parse().ok()?;
            match Rat::from_big(p.clone(), q.clone()) {
                Some(x) if kind == "q" => rat(x),
                Some(x) => pi_times(x),
                None if kind == "q" => approx(ctx.ratio(&p, &q)),
                None => {
                    let v = ctx.ratio(&p, &q);
                    let pi = ctx.pi();
                    approx(v.mul(&pi, ctx.p, RM))
                }
            }
        }
        "e" => approx(ctx.cc.e(ctx.p, RM)),
        "inf" => inf(a != "-"),
        "nan" => undef(),
        "m" => {
            let (neg, digits) = match a.strip_prefix('-') {
                Some(d) => (true, d),
                None => (false, a),
            };
            let man = BigInt::parse_bytes(digits.as_bytes(), 16)?;
            let exp: i64 = b.parse().ok()?;
            if man.is_zero() {
                return Some(zero());
            }
            // man * 2^exp, exactly when the precision holds the mantissa
            let bits = man.bits() as usize;
            let g = ctx.p.max(bits) + 64;
            let m = ctx.big_at(&man, g);
            let two = small_int(2, g);
            let scale = if exp >= 0 {
                two.powi(exp as usize, g, RM)
            } else {
                one(g).div(&two.powi((-exp) as usize, g, RM), g, RM)
            };
            let v = m.mul(&scale, ctx.p.max(bits), RM);
            approx(if neg { v.neg() } else { v })
        }
        _ => return None,
    })
}

/// A value's interchange spec: exact values as `val_from_spec` reads them, an approximation as
/// `("m", man, exp)`.
pub fn val_spec(v: &Val) -> (String, String, String) {
    match v {
        Val::Exact(Exact::Rat(x)) => ("q".into(), x.numer_string(), x.denom_string()),
        Val::Exact(Exact::Pi(x)) => ("pi".into(), x.numer_string(), x.denom_string()),
        Val::Exact(Exact::PosInf) => ("inf".into(), "+".into(), String::new()),
        Val::Exact(Exact::NegInf) => ("inf".into(), "-".into(), String::new()),
        Val::Exact(Exact::Undef) => ("nan".into(), String::new(), String::new()),
        Val::Approx(b) => bf_spec(b),
    }
}

/// A finite BigFloat as `("m", man, exp)`: `man * 2^exp` exactly.
pub fn bf_spec(b: &BigFloat) -> (String, String, String) {
    if b.is_zero() {
        return ("q".into(), "0".into(), "1".into());
    }
    let (Some(words), Some(e)) = (b.mantissa_digits(), b.exponent()) else {
        return ("nan".into(), String::new(), String::new());
    };
    // value = 0.m * 2^e, the words little-endian
    let mut man = BigInt::zero();
    for w in words.iter().rev() {
        man = (man << 64) + BigInt::from(*w);
    }
    let exp = i64::from(e) - 64 * words.len() as i64;
    let hex = man.to_str_radix(16);
    let hex = if b.is_negative() {
        format!("-{hex}")
    } else {
        hex
    };
    ("m".into(), hex, exp.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arity() -> std::collections::HashMap<String, u8> {
        let mut m = std::collections::HashMap::new();
        for u in [
            "neg", "abs", "inv", "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "sinh",
            "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil", "erf", "erfc", "harmonic",
        ] {
            m.insert(u.to_string(), 1);
        }
        for b in [
            "+", "-", "*", "/", "pow", "rootn", "max", "min", "mod", "atan2", "besselj", "bessely",
            "besseli", "besselk",
        ] {
            m.insert(b.to_string(), 2);
        }
        m
    }

    fn real(expr: &str, env: &[Val]) -> Option<Val> {
        let toks: Vec<String> = expr.split(' ').map(String::from).collect();
        let names: Vec<String> = (0..env.len()).map(|i| format!("x{i}")).collect();
        let e = Expr::compile(&toks, &names, &arity()).unwrap();
        e.eval_real(env, 50).unwrap().value
    }

    fn class(v: &Option<Val>) -> &'static str {
        match v {
            None => "unresolved",
            Some(Val::Exact(Exact::Undef)) => "nan",
            Some(Val::Exact(Exact::PosInf)) => "+inf",
            Some(Val::Exact(Exact::NegInf)) => "-inf",
            Some(_) => "fin",
        }
    }

    /// The limit rule's poles and boundaries (V4) and infinite arguments (V5), and undefined
    /// propagating (V6).
    #[test]
    fn the_limit_rule() {
        for (expr, want) in [
            ("inv 0", "nan"),
            ("/ 1 0", "nan"),
            ("pow 0 -1", "nan"),
            ("rootn 0 -3", "nan"),
            ("pow 0 -2", "nan"),
            ("inv pow 0 2", "nan"),
            ("pow 0 -0.5", "nan"),
            ("pow 0 float(\"-inf\")", "nan"),
            ("rootn 0 -2", "nan"),
            ("mod 1 0", "nan"),
            ("mod 0 float(\"inf\")", "nan"),
            ("mod 0 float(\"-inf\")", "nan"),
            ("mod -1 float(\"inf\")", "+inf"),
            ("rootn 2 float(\"inf\")", "nan"),
            ("rootn 1 atanh -1", "nan"),
            ("log 0", "-inf"),
            ("atanh 1", "+inf"),
            ("atanh -1", "-inf"),
            ("tan asin 1", "nan"),
            ("tan / np.pi 2", "nan"),
            ("exp float(\"inf\")", "+inf"),
            ("pow 1 float(\"inf\")", "nan"),
            ("pow float(\"inf\") 0", "nan"),
            ("pow float(\"nan\") 0", "nan"),
            ("pow 1 float(\"nan\")", "nan"),
            ("- float(\"inf\") float(\"inf\")", "nan"),
            ("* 0 float(\"inf\")", "nan"),
            ("sin float(\"inf\")", "nan"),
            ("exp tan asin 1", "nan"),
            ("pow -8 / 1 3", "nan"),
        ] {
            assert_eq!(class(&real(expr, &[])), want, "{expr}");
        }
        for (expr, want) in [
            ("exp float(\"-inf\")", int(0)),
            ("tanh float(\"inf\")", int(1)),
            ("pow 0 0", int(1)),
            ("pow float(\"-inf\") 0", int(1)),
            ("mod 1 float(\"inf\")", int(1)),
            ("rootn -8 3", int(-2)),
            ("acosh 1", int(0)),
            ("ceil 0.5", int(1)),
            ("asin 1", pi_times(r(1, 2))),
            ("atan float(\"inf\")", pi_times(r(1, 2))),
            ("cos np.pi", int(-1)),
            ("sin np.pi", int(0)),
            ("+ 0.1 0.2", rat(r(3, 10))),
        ] {
            let got = real(expr, &[]);
            match (&got, &want) {
                (Some(Val::Exact(a)), Val::Exact(b)) => assert_eq!(a, b, "{expr}"),
                _ => panic!("{expr}: {got:?}"),
            }
        }
    }

    /// The cells of `erfc` and the Bessel functions of the second kind and the modified ones:
    /// outside the domain undefined, the limit rule at x = 0 and at infinite operands, the
    /// formula's value where it has one, and no value where the joint limit does not exist (Y at
    /// x = 0 under a negative half-integer order, an infinite negative order).
    #[test]
    fn the_scipy_heads_cells() {
        let (inf, ninf, nan) = ("float(\"inf\")", "float(\"-inf\")", "float(\"nan\")");
        for (expr, want) in [
            (format!("erfc {nan}"), "nan"),
            ("bessely 1 -1".into(), "nan"),
            ("besselk 1 -1".into(), "nan"),
            ("besseli 0.5 -1".into(), "nan"),
            ("besseli 3 -1".into(), "fin"),
            ("bessely 0 0".into(), "-inf"),
            ("bessely 2.5 0".into(), "-inf"),
            ("bessely -1 0".into(), "+inf"),
            ("bessely -2 0".into(), "-inf"),
            ("bessely -1.25 0".into(), "+inf"),
            ("bessely -0.25 0".into(), "-inf"),
            ("bessely -0.5 0".into(), "nan"),
            ("bessely -1.5 0".into(), "nan"),
            ("bessely -2.5 0".into(), "nan"),
            // an approximate order within the band of -1/2 cannot be told from it
            ("bessely neg log rootn np.e 2 0".into(), "unresolved"),
            ("besselk 0 0".into(), "+inf"),
            ("besselk -2.5 0".into(), "+inf"),
            ("besseli -0.5 0".into(), "+inf"),
            ("besseli -1.5 0".into(), "-inf"),
            ("besseli -2.5 0".into(), "+inf"),
            (format!("bessely 1 {ninf}"), "nan"),
            (format!("bessely {inf} 1"), "-inf"),
            (format!("bessely {inf} 0"), "-inf"),
            (format!("bessely {ninf} 1"), "nan"),
            (format!("bessely {inf} {inf}"), "nan"),
            (format!("besseli 1 {inf}"), "+inf"),
            (format!("besseli 2 {ninf}"), "+inf"),
            (format!("besseli 3 {ninf}"), "-inf"),
            (format!("besseli 0.5 {ninf}"), "nan"),
            (format!("besseli {ninf} 1"), "nan"),
            (format!("besseli {ninf} -1"), "nan"),
            (format!("besseli {ninf} -2.5"), "nan"),
            (format!("besseli {inf} {inf}"), "nan"),
            (format!("besselk 1 {ninf}"), "nan"),
            (format!("besselk {inf} 1"), "+inf"),
            (format!("besselk {ninf} 1"), "+inf"),
            (format!("besselk {inf} 0"), "+inf"),
            (format!("besselk {inf} -1"), "nan"),
            (format!("besselk {inf} {inf}"), "nan"),
            (format!("bessely {nan} 1"), "nan"),
            (format!("besselk {nan} 0"), "nan"),
        ] {
            assert_eq!(class(&real(&expr, &[])), want, "{expr}");
        }
        for (expr, want) in [
            (format!("erfc {inf}"), int(0)),
            (format!("erfc {ninf}"), int(2)),
            ("erfc 0".into(), int(1)),
            ("besseli 1 0".into(), int(0)),
            ("besseli -1 0".into(), int(0)),
            ("besseli 2.5 0".into(), int(0)),
            (format!("bessely 1 {inf}"), int(0)),
            (format!("besseli {inf} 1"), int(0)),
            (format!("besseli {inf} -1"), int(0)),
            (format!("besselk 1 {inf}"), int(0)),
        ] {
            let got = real(&expr, &[]);
            match (&got, &want) {
                (Some(Val::Exact(a)), Val::Exact(b)) => assert_eq!(a, b, "{expr}"),
                _ => panic!("{expr}: {got:?}"),
            }
        }
        // I_0(0) = 1 by the series, and the parity of an integer order at x < 0
        let mut ctx = Ctx::new(50).unwrap();
        let mut read = |expr: &str| ctx.bf(&real(expr, &[]).expect(expr));
        let (i00, even, odd) = (
            read("besseli 0 0"),
            read("besseli 2 -1"),
            read("besseli 3 -1"),
        );
        let (e2, o3) = (read("besseli 2 1"), read("besseli 3 1"));
        let p = ctx.p;
        assert!(i00.sub(&one(p), p, RM).is_zero(), "{i00}");
        assert!(even.sub(&e2, p, RM).is_zero() && odd.add(&o3, p, RM).is_zero());
    }

    /// At a zero argument a Bessel head's value is the formula's, exact, whether the order is
    /// written or bound: `J_0(0) = I_0(0) = 1`, so `atanh` of it is +inf, not a value inside
    /// atanh's pole band.
    #[test]
    fn a_bessel_head_at_a_zero_argument_is_exact() {
        let zero = [int(0)];
        for head in ["besselj", "besseli"] {
            for (expr, env) in [
                (format!("{head} 0 0"), &[][..]),
                (format!("{head} x0 0"), &zero[..]),
            ] {
                match real(&expr, env) {
                    Some(Val::Exact(Exact::Rat(r))) => assert!(r == Rat::int(1), "{expr}"),
                    got => panic!("{expr}: {got:?}"),
                }
            }
            assert_eq!(class(&real(&format!("atanh {head} 0 0"), &[])), "+inf");
            assert_eq!(class(&real(&format!("atanh {head} x0 0"), &zero)), "+inf");
            // a nonzero order at a zero argument is 0, exactly
            assert!(matches!(real(&format!("{head} 2 0"), &[]),
                             Some(Val::Exact(Exact::Rat(r))) if r.is_zero()));
        }
    }

    /// A point the table decides is decided, through a variable: `tan(asin(ceil x))` at x = 0.5
    /// is a pole, `atanh(ceil x)` +inf.
    #[test]
    fn exact_through_a_variable() {
        let half = rat(r(1, 2));
        assert_eq!(
            class(&real("tan asin ceil x0", std::slice::from_ref(&half))),
            "nan"
        );
        assert_eq!(class(&real("atanh ceil x0", &[half])), "+inf");
    }

    #[test]
    fn specs_round_trip() {
        let mut ctx = Ctx::new(50).unwrap();
        let v = approx(ctx.cc.pi(ctx.p, RM));
        let (k, a, b) = val_spec(&v);
        let back = val_from_spec(&k, &a, &b, &mut ctx).unwrap();
        let (x, y) = (ctx.bf(&v), ctx.bf(&back));
        assert!(x.sub(&y, ctx.p, RM).is_zero());
    }

    #[test]
    fn f64_sameness() {
        assert!(f64_same(1.0, 1.0) && f64_same(f64::NAN, -f64::NAN));
        assert!(!f64_same(1.0000000000000002, 1.0));
        assert!(!f64_same(-0.0, 0.0));
        assert!(f64_same(f64::INFINITY, f64::INFINITY) && !f64_same(f64::INFINITY, f64::MAX));
    }

    #[test]
    fn f64_agreement() {
        assert_eq!(f64_agree(0.30000000000000004, 0.3), Cmp::Eq);
        assert_eq!(f64_agree(1.2246467991473532e-16, 0.0), Cmp::RealChange);
        assert_eq!(f64_agree(-0.0, 0.0), Cmp::Eq);
        assert_eq!(f64_agree(f64::NAN, 1.0), Cmp::Ext);
        assert_eq!(f64_agree(f64::INFINITY, 1.0), Cmp::InfChange);
    }

    /// A step head over an approximation in an integer's band, on the integer included, is
    /// undecided at the working precision, and a finer rung decides it: `tanh(400)` is the
    /// fifty-digit 1.0 and below 1, so its floor is 0, never 1.
    #[test]
    fn a_step_of_an_approximation_on_an_integer_is_undecided() {
        let toks: Vec<String> = "floor tanh 400".split(' ').map(String::from).collect();
        let e = Expr::compile(&toks, &[], &arity()).unwrap();
        assert_eq!(class(&e.eval_real(&[], 50).unwrap().value), "unresolved");
        let v = e.eval_real(&[], 400).unwrap().value.unwrap();
        assert!(is_exact_zero(&v), "{v:?}");
        assert_eq!(class(&real("ceil + 1 * -1 exp -200", &[])), "unresolved");
        // an exact value is exact, and one off every integer's band is decided
        assert_eq!(class(&real("floor 2.5", &[])), "fin");
        assert_eq!(class(&real("floor tanh 1", &[])), "fin");
    }
}
