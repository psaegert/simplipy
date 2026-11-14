//! THE GROUND EVALUATOR: the value of a fully numeric expression in each mode, defined once.
//!
//! The engine's parse and fold, every rule step on a ground subject, the miner's Dirac refuter,
//! the judge's ground check and verify-mine's ground gate read a ground value here, and nowhere
//! else.
//!
//! **Two readings of one expression.**
//!
//! - *Float64's*: the shipped realizations evaluated in the written order, each literal at the
//!   double nearest to it, the sign of a zero, the infinities and NaN included, and whether an
//!   operation overflowed or underflowed on the way (a range event, `numeric::RangeKind`).
//!   Strict's float64 value of an elementary function (`sin`, `exp`, `tanh`, ...) is
//!   `numeric::elementary`'s, the one place that value is decided: a ruling on it changes what
//!   that module calls and nothing here.
//! - *The reals'*: the table's value (`table::ground_value`), read at the judge's precision
//!   ladder: exact where the table carries it exactly (rationals, rational multiples of pi, the
//!   infinities, no value), a high-precision approximation elsewhere.
//!
//! **strict.** An expression `R` may stand for a part `P` (the part's admitted values) where
//! (a) float64 gives `R` the double it gives `P`, or (b) the reals give `R` the value they give
//! `P` and float64's two values agree in kind: both finite (two zeros of one sign), the same
//! infinity, both NaN, or a range event on either side whose true value is of the other's kind
//! (the range standard). (b) reads only where the reals give `P` a value. So `floor(tanh(100))`
//! is 0 by (b) (float64 floors 1.0 to 1), `tanh(100)` is 1 by (a), `exp(1000)/exp(1000)` is 1 by
//! (b) (float64's inf/inf is a range event), and `tan(asin(1))` keeps float64's 1.633e16 (the
//! reals give it no value).
//!
//! **permissive.** The value is the table's, at float resolution: `R` may stand for `P` where the
//! table gives neither a value, the same infinity, or finite values with one nearest double.
//! `floor(tanh(100))` is 0, `asin(sin(4))` is pi - 4, `tan(asin(1))` has no value (NaN).

use std::cell::RefCell;

use rustc_hash::FxHashMap;

use crate::operators::Operators;
use crate::table::{self, Exact, Val};

/// The two modes' readings. `Strict` is the engine's `RuleMode::Default`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Strict,
    Permissive,
}

/// One ground expression's readings, computed once: float64's at once, the reals' on first use.
#[derive(Clone, Debug)]
pub struct Reading {
    /// float64's value, `None` where the evaluator cannot read a token
    pub float: Option<f64>,
    /// whether an operation of float64's evaluation overflowed or underflowed, or a literal lies
    /// past the doubles' range (`1e400` read as inf, `1e-400` as 0)
    pub range_event: bool,
    /// whether float64's evaluation hands a negative zero to an operation that reads a zero's
    /// sign (`atan2(ceil(-0.5), -1)` is -pi, where the reals' one zero gives pi)
    pub reads_a_negative_zero: bool,
    /// permissive's value at float resolution where the exact ladder reads none and the root is
    /// a bounded function within the band of its bound (`table::Expr::float_resolution_at_a_bounded_root`)
    float_resolution: RefCell<Option<Option<Val>>>,
    /// the reals' value and the digits it was read at; `None` inside: no rung reads it
    real: RefCell<Option<Option<(Val, i32)>>>,
    tokens: Vec<String>,
}

impl Reading {
    /// The reals' value and the digits it was read at (the table at its ladder), `None` where no
    /// rung up to the ladder's ceiling reads it.
    pub fn real(&self, ops: &Operators) -> Option<(Val, i32)> {
        if let Some(v) = self.real.borrow().as_ref() {
            return v.clone();
        }
        let v = compile(&self.tokens, ops).and_then(|e| table::ground_value(&e));
        *self.real.borrow_mut() = Some(v.clone());
        v
    }

    /// Permissive's value at float resolution: the reals' value (`real`), or, where no rung reads
    /// it, the bound of a bounded root within its band, read for its nearest double only.
    pub fn permissive_value(&self, ops: &Operators) -> Option<(Val, i32)> {
        if let Some(v) = self.real(ops) {
            return Some(v);
        }
        if let Some(v) = self.float_resolution.borrow().as_ref() {
            return v.clone().map(|v| (v, table::BASE_DPS));
        }
        let v = compile(&self.tokens, ops).and_then(|e| e.float_resolution_at_a_bounded_root());
        *self.float_resolution.borrow_mut() = Some(v.clone());
        v.map(|v| (v, table::BASE_DPS))
    }
}

/// Is a literal token read past the doubles' range: a nonzero value float64 reads as an
/// infinity, as zero or as a subnormal (`1e400`, `1e-400`, `1e-310`)?
fn literal_out_of_range(tok: &str) -> bool {
    if !crate::utils::looks_numeric(tok) {
        return false;
    }
    let Some(v) = crate::numeric::leaf_value(tok) else {
        return false;
    };
    if v.is_infinite() {
        return true;
    }
    if v == 0.0 || v.is_subnormal() {
        let body = tok.trim_start_matches('(').trim_end_matches(')');
        return crate::ac::rat::token_rational(body.trim_start_matches(['+', '-']))
            .is_some_and(|(p, _)| !num_traits::Zero::is_zero(&p));
    }
    false
}

fn compile(tokens: &[String], ops: &Operators) -> Option<table::Expr> {
    let arity: std::collections::HashMap<String, u8> = tokens
        .iter()
        .filter_map(|t| ops.arity_of(t).map(|a| (t.clone(), a)))
        .collect();
    table::Expr::compile(tokens, &[], &arity).ok()
}

thread_local! {
    /// Each thread remembers the readings it computed (a reading is a function of the tokens),
    /// emptying the table at 16,384 entries.
    static READINGS: RefCell<FxHashMap<Box<str>, std::rc::Rc<Reading>>> =
        RefCell::new(FxHashMap::default());
}

/// THE READING of a ground expression (prefix tokens, each literal one token). `None` for a
/// token sequence that is no ground expression of the table and the evaluator.
///
/// The laws applied here (`ac::laws::Law`): `num.ground_first`, `num.ground_class_fold`,
/// `num.ground_float_fold`.
pub fn read(tokens: &[String], ops: &Operators) -> Option<std::rc::Rc<Reading>> {
    let key: Box<str> = tokens.join("\u{1}").into_boxed_str();
    if let Some(r) = READINGS.with(|m| m.borrow().get(&key).cloned()) {
        return Some(r);
    }
    let e = compile(tokens, ops)?;
    let (float, in_range) = match e.eval_f64(&[]) {
        Some((v, ok)) => (Some(v), ok),
        None => (None, true),
    };
    let literal_event = tokens
        .iter()
        .any(|t| ops.arity_of(t).is_none() && literal_out_of_range(t));
    let r = std::rc::Rc::new(Reading {
        float,
        range_event: !in_range || literal_event,
        reads_a_negative_zero: e.reads_a_negative_zero(&[]),
        real: RefCell::new(None),
        float_resolution: RefCell::new(None),
        tokens: tokens.to_vec(),
    });
    READINGS.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= 1 << 14 {
            m.clear();
        }
        m.insert(key, r.clone());
    });
    Some(r)
}

/// Do two float64 values agree in kind: both NaN, the same infinity, or both finite, two zeros
/// carrying one sign?
fn float_kinds_agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    if a == 0.0 && b == 0.0 {
        return a.is_sign_negative() == b.is_sign_negative();
    }
    true
}

/// Is a float64 value of the kind of the real value `v` (finite against finite, an infinity
/// against the same infinity)? The range standard's other side.
fn float_of_real_kind(f: f64, v: &Val) -> bool {
    match table::class_of(v) {
        (true, _) => false,
        (_, Some(up)) => f == if up { f64::INFINITY } else { f64::NEG_INFINITY },
        _ => f.is_finite(),
    }
}

/// THE ADMISSION: may `cand` stand for `part` in `mode`? See the module documentation. Every
/// undecided reading refuses (a reading the evaluator or the table cannot make admits nothing).
pub fn admits(mode: Mode, part: &Reading, cand: &Reading, ops: &Operators) -> bool {
    match mode {
        Mode::Strict => {
            let (Some(p), Some(c)) = (part.float, cand.float) else {
                return false;
            };
            // (a) the same double
            if p.to_bits() == c.to_bits() || (p.is_nan() && c.is_nan()) {
                return true;
            }
            same_real_in_kind(part, cand, ops)
        }
        Mode::Permissive => {
            let (Some((rp, dp)), Some((rc, dc))) =
                (part.permissive_value(ops), cand.permissive_value(ops))
            else {
                return false;
            };
            same_at_float_resolution(&rp, dp, &rc, dc)
        }
    }
}

/// The reals' values of two readings at ONE working precision, for comparing them: an
/// approximation read at a coarser rung than the other side's is read again at the finer one, so
/// the two agree only as far as both are read (`rootn(exp(800.5), 2)` is read at about 360 digits
/// and `exp(400.25)` at about 190: compared at 360 digits, the coarser reading would differ in
/// its 190th). `None` where a side has no reading or the second reading reaches none.
fn reals_at_one_precision(a: &Reading, b: &Reading, ops: &Operators) -> Option<(Val, Val, i32)> {
    let (va, da) = a.real(ops)?;
    let (vb, db) = b.real(ops)?;
    let dps = da.max(db);
    let at = |r: &Reading, v: Val, d: i32| -> Option<Val> {
        if d == dps || matches!(v, Val::Exact(_)) {
            return Some(v);
        }
        compile(&r.tokens, ops)?.eval_real_ground(dps)?.value
    };
    Some((at(a, va, da)?, at(b, vb, db)?, dps))
}

/// Strict's (b): the reals give both readings one value, and float64's two values agree in kind
/// (the range standard included). Where the reals give the part no value, (b) does not apply.
fn same_real_in_kind(part: &Reading, cand: &Reading, ops: &Operators) -> bool {
    let (Some(p), Some(c)) = (part.float, cand.float) else {
        return false;
    };
    // a zero's sign read inside either side is a kind the reals' one zero does not carry
    if part.reads_a_negative_zero || cand.reads_a_negative_zero {
        return false;
    }
    let Some((rp, rc, dps)) = reals_at_one_precision(part, cand, ops) else {
        return false;
    };
    if matches!(rp, Val::Exact(Exact::Undef)) || !table::same_real_value(&rp, &rc, dps) {
        return false;
    }
    float_kinds_agree(p, c)
        || (part.range_event && float_of_real_kind(c, &rp))
        || (cand.range_event && float_of_real_kind(p, &rp))
        || (part.range_event && cand.range_event)
}

/// The evaluator's verdict on a FULLY NUMERIC RULE `lhs -> rhs`, every reading the judge's ground
/// check reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleVerdict {
    /// strict's (a): float64 gives both sides one double
    pub same_double: bool,
    /// strict's (b): the reals give both sides one value and float64's kinds agree
    pub same_real_in_kind: bool,
    /// the reals' values: one value (`Some(true)`), two (`Some(false)`), unread (`None`)
    pub reals: Option<bool>,
    /// permissive: the table's values at float resolution
    pub permissive: bool,
}

/// THE GROUND CHECK of a fully numeric rule (the judge's and verify-mine's): each mode's
/// admission of `rhs` for `lhs` (`admits`), with the reals' comparison beside it. `None` where a
/// side is no ground expression the evaluator reads.
pub fn rule_verdict(lhs: &[String], rhs: &[String], ops: &Operators) -> Option<RuleVerdict> {
    let (l, r) = (read(lhs, ops)?, read(rhs, ops)?);
    let (lf, rf) = (l.float?, r.float?);
    let same_double = lf.to_bits() == rf.to_bits() || (lf.is_nan() && rf.is_nan());
    let reals = reals_at_one_precision(&l, &r, ops).map(|(a, b, dps)| {
        let (ua, ub) = (table::class_of(&a).0, table::class_of(&b).0);
        if ua || ub {
            ua && ub
        } else {
            table::same_real_value(&a, &b, dps)
        }
    });
    Some(RuleVerdict {
        same_double,
        same_real_in_kind: same_real_in_kind(&l, &r, ops),
        reals,
        permissive: admits(Mode::Permissive, &l, &r, ops),
    })
}

/// Permissive's agreement of two real values at float resolution: neither has a value, the same
/// infinity, one exact value, or nearest doubles that agree as float64's values do (at most 8
/// units in the last place apart, the two zeros as one value: `tanh(-19)` and -1).
pub fn same_at_float_resolution(a: &Val, da: i32, b: &Val, db: i32) -> bool {
    let (ca, cb) = (table::class_of(a), table::class_of(b));
    if ca.0 || cb.0 {
        return ca.0 && cb.0;
    }
    if ca.1.is_some() || cb.1.is_some() {
        return ca.1 == cb.1;
    }
    if let (Val::Exact(x), Val::Exact(y)) = (a, b) {
        if x == y {
            return true;
        }
    }
    match (nearest_double(a, da), nearest_double(b, db)) {
        (Some(x), Some(y)) => table::f64_agree(x, y) == table::Cmp::Eq,
        _ => false,
    }
}

/// The double nearest to a finite real value (`table::nearest_f64`).
pub fn nearest_double(v: &Val, dps: i32) -> Option<f64> {
    let mut ctx = table::Ctx::new(dps.max(table::BASE_DPS))?;
    table::nearest_f64(v, &mut ctx)
}

// ------------------------------------------------------------------- the recognition list

/// THE RECOGNITION BOUNDS: `k*pi/m` and `k*e/m` for coprime `1 <= k, m <= RECOGNISED_K`;
/// `k/m` for coprime `m <= RECOGNISED_K` whose decimal does not terminate, with
/// `|k/m| <= RECOGNISED_FRACTION_MAX`.
pub const RECOGNISED_K: i64 = 24;
/// See [`RECOGNISED_K`].
pub const RECOGNISED_FRACTION_MAX: i64 = 100;

/// An entry of the recognition list: a short symbolic number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// `k*pi/m`
    Pi(i64, i64),
    /// `k*e/m`
    E(i64, i64),
    /// `k/m`, a fraction whose decimal does not terminate
    Fraction(i64, i64),
    /// `log(n)`, n in 2, 3, 10
    Log(i64),
    /// `sqrt(n)`, n in 2, 3, 5
    Sqrt(i64),
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Every entry under the doubles float64 may compute it as (`k*pi/m` as `(k*pi)/m`,
/// `(k/m)*pi` and `k*(pi/m)`; `log(n)`; `sqrt(n)`), the keys a literal is looked up by. A key
/// only finds an entry: whether a literal is read as it is decided by evaluation, on the entry
/// as the engine spells it (`spelled_double`). No two entries share a double within the bounds
/// (checked by `recognition_doubles_are_distinct`).
fn entries() -> &'static FxHashMap<u64, Entry> {
    static LIST: std::sync::OnceLock<FxHashMap<u64, Entry>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut out = FxHashMap::default();
        let mut put = |d: f64, e: Entry| {
            out.entry(d.to_bits()).or_insert(e);
        };
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        for m in 1..=RECOGNISED_K {
            for k in 1..=RECOGNISED_K {
                if gcd(k, m) == 1 {
                    let (kf, mf) = (k as f64, m as f64);
                    for (c, entry) in [(pi, Entry::Pi(k, m)), (e, Entry::E(k, m))] {
                        put(kf * c / mf, entry);
                        put((kf / mf) * c, entry);
                        put(kf * (c / mf), entry);
                    }
                }
            }
            let mut rest = m;
            for p in [2, 5] {
                while rest % p == 0 {
                    rest /= p;
                }
            }
            if rest != 1 {
                for k in 1..=RECOGNISED_FRACTION_MAX * m {
                    if gcd(k, m) == 1 {
                        put((k as f64) / (m as f64), Entry::Fraction(k, m));
                    }
                }
            }
        }
        for n in [2i64, 3, 10] {
            put(crate::numeric::elementary::log(n as f64), Entry::Log(n));
        }
        for n in [2i64, 3, 5] {
            put((n as f64).sqrt(), Entry::Sqrt(n));
        }
        out
    })
}

/// FLOAT64'S VALUE OF A SPELLING (prefix tokens, every literal one token), as the shipped
/// realizations compute it in written order: an operator's arity read off the core's own
/// kernels (`numeric::apply_op`), so it needs no configuration. The recognition list's check by
/// evaluation reads it. `None` for a token the kernels and the literal reader do not know.
pub fn spelled_double(tokens: &[String]) -> Option<f64> {
    fn node(tokens: &[String], i: &mut usize) -> Option<f64> {
        let t = tokens.get(*i)?;
        *i += 1;
        let arity = if crate::numeric::binary_fn(t).is_some() {
            2
        } else if crate::numeric::unary_fn(t).is_some() {
            1
        } else {
            return crate::numeric::leaf_value(t);
        };
        let mut args = [0.0f64; 2];
        for a in args.iter_mut().take(arity) {
            *a = node(tokens, i)?;
        }
        crate::numeric::apply_op(t, &args[..arity])
    }
    let mut i = 0;
    let v = node(tokens, &mut i)?;
    (i == tokens.len()).then_some(v)
}

/// The entry whose double is `|d|`, with the sign `d` carries outside it: `Some((entry,
/// negative))`. A candidate only: the caller checks it by evaluation.
pub fn recognise(d: f64) -> Option<(Entry, bool)> {
    if !d.is_finite() || d == 0.0 {
        return None;
    }
    entries()
        .get(&d.abs().to_bits())
        .map(|e| (*e, d.is_sign_negative()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognition_doubles_are_distinct() {
        // every entry kept its own double: none was shadowed by an earlier one
        let mut n = 0usize;
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        for m in 1..=RECOGNISED_K {
            for k in 1..=RECOGNISED_K {
                if gcd(k, m) == 1 {
                    assert_eq!(
                        recognise((k as f64) * pi / (m as f64)),
                        Some((Entry::Pi(k, m), false))
                    );
                    assert_eq!(
                        recognise((k as f64) * e / (m as f64)),
                        Some((Entry::E(k, m), false))
                    );
                    n += 2;
                }
            }
        }
        assert!(n > 0);
        assert_eq!(recognise(-pi), Some((Entry::Pi(1, 1), true)));
        assert_eq!(recognise(1.0 / 3.0), Some((Entry::Fraction(1, 3), false)));
        assert_eq!(
            recognise(0.5),
            None,
            "a terminating fraction is its own decimal"
        );
        assert_eq!(recognise(2f64.sqrt()), Some((Entry::Sqrt(2), false)));
        assert_eq!(
            recognise(std::f64::consts::LN_2),
            Some((Entry::Log(2), false))
        );
        assert_eq!(recognise(0.3), None);
    }

    #[test]
    fn kinds_and_the_range_standard() {
        assert!(float_kinds_agree(1.0, 0.0), "finite against finite");
        assert!(!float_kinds_agree(0.0, -0.0), "two zeros keep their signs");
        assert!(!float_kinds_agree(f64::INFINITY, 1.0));
        assert!(float_kinds_agree(f64::NAN, f64::NAN));
    }
}
