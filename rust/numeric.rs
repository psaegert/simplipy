//! Numeric constant folding (the `numeric` engine line): a native f64 evaluator + a CPython-exact
//! `str(float)` formatter, replacing the Python `_evaluate_constant_subtree` which
//! went through `operators_to_realizations` -> `prefix_to_infix` -> `codify` -> `code_to_lambda`.
//!
//! Parity policy ("numerically correct"): the evaluator uses the SAME primitives as
//! Python -- libm (`powf`/`sin`/...) and exact f64 arithmetic, with the engine's custom operator
//! semantics (the `div`/`inv` zero handling, the real cube/fifth roots, negative-base powers ->
//! NaN). One deliberate deviation from raw libm (`aligned_pow`): `pow(negative base incl. -inf,
//! finite non-integer exponent)` is NaN -- the real-analysis semantics -- overriding C99's
//! magnitude convention at the -inf-base row only; infinite exponents keep the C99
//! magnitude-step cells. Values otherwise match Python bit-for-bit on a given build, and the
//! float repr matches `str()` byte-for-byte. Result `None` mirrors Python returning `None`
//! (unfoldable: unparseable leaf, unknown operator).

use crate::operators::Operators;

mod besselj;
mod dd;
pub(crate) mod elementary;
mod harmonic;
#[rustfmt::skip]
mod harmonic_tables;
pub mod kernels;

/// The libm `atanh`, for the ONE consumer outside this module's own kernels: the interval
/// arm (`interval.rs::u_atanh`) must evaluate endpoint images with the SAME function the
/// fold ships, because its outward-rounding budget (`ULP_LIBM` = 8 ulps) only covers
/// libm's own rounding error -- not the unbounded std-vs-libm composition gap documented
/// in `elementary` (audit B6: the fold at -0.999 sat 161 ulps below the std-computed
/// box's lower bound, an enclosure violation).
#[inline]
pub(crate) fn libm_atanh(x: f64) -> f64 {
    elementary::atanh(x)
}

/// Contract-aligned power: `pow(negative base — including −∞ — , finite non-integer
/// exponent)` is undefined (NaN), the real-analysis semantics. libm already returns NaN
/// for finite negative bases; C99's special case `pow(−∞, non-integer) = ±∞` (magnitude
/// convention) is deliberately overridden, so the override changes only the −∞-base row.
/// Infinite exponents keep the C99 magnitude-step semantics (`pow(t, ±∞)` decided by
/// |t| — including negative `t`), which the contract ratifies as-is.
#[inline]
pub(crate) fn aligned_pow(x: f64, y: f64) -> f64 {
    if x < 0.0 && y.is_finite() && y.fract() != 0.0 {
        return f64::NAN;
    }
    elementary::pow(x, y)
}

/// Port of `_evaluate_constant_subtree`: evaluate an all-numeric prefix
/// subtree to a single result token, or `None` if it cannot be folded (matches Python's `None`).
pub fn evaluate_constant_subtree(tokens: &[String], ops: &Operators) -> Option<String> {
    let mut idx = 0;
    let value = eval_node(tokens, &mut idx, ops)?;
    if idx != tokens.len() {
        return None; // malformed (extra tokens) -> unfoldable
    }
    // Fold to the IEEE-754 f64 result -- including inf/nan (`1/0 -> float("inf")`, `sqrt(-1) ->
    // float("nan")`): the value Rust f64 computes natively and what the engine is designed to do
    // (test_division_by_zero_produces_inf). `py_float_repr` emits the `float("inf")`/`float("nan")`
    // tokens. Deterministic + identical to the Python f64+libm reference (NUMERIC_FOLDING_PARITY.md).
    Some(py_float_repr(value))
}

/// The float64 value of an all-numeric prefix subtree as the evaluator computes it, the sign
/// of a zero included (`evaluate_constant_subtree` writes a token, which keeps no zero sign).
pub(crate) fn evaluate_constant_value(tokens: &[String], ops: &Operators) -> Option<f64> {
    let mut idx = 0;
    let value = eval_node(tokens, &mut idx, ops)?;
    (idx == tokens.len()).then_some(value)
}

/// Recursive prefix evaluation: an operator consumes `arity` operands; a leaf is `float(token)`.
fn eval_node(tokens: &[String], idx: &mut usize, ops: &Operators) -> Option<f64> {
    let tok = tokens.get(*idx)?.clone();
    *idx += 1;
    match ops.arity_of(&tok) {
        Some(arity) => {
            let mut args = [0.0f64; 2];
            let arity = arity as usize;
            if arity > 2 {
                return None; // the evaluator handles arity <= 2 (every supported vocabulary)
            }
            for a in args.iter_mut().take(arity) {
                *a = eval_node(tokens, idx, ops)?;
            }
            apply_op(&tok, &args[..arity])
        }
        // Leaf: any token `leaf_value` resolves -- numeric literals (Python `float(token)`; Rust
        // `parse::<f64>` agrees on every `is_numeric_string`-true form the grammar emits), the
        // special constants (`np.pi`, `np.e`, the `float("...")` tokens) and parenthesized
        // literals (`(-1)`). ONE leaf table with the tape evaluator (the previous
        // `parse_pyfloat`-only leaf made `cosh(np.pi)` unfoldable HERE while the OFFLINE tape
        // resolved it, so the miner and the folder disagreed about which subtrees are constant).
        // An unresolvable leaf -> None (Python would raise inside the eval -> None).
        None => leaf_value(&tok),
    }
}

/// Map a non-operator, non-`<constant>` leaf token to the f64 value Python's eval of the realized
/// infix would produce: the special constants (`np.pi`, `np.e`, the `float("...")` tokens, the
/// parenthesized `(-1)` form from `extra_internal_terms`), then the general `float(token)` numeric
/// literals. `None` for a token that is neither (an unknown leaf). Used by the OFFLINE evaluator.
pub(crate) fn leaf_value(tok: &str) -> Option<f64> {
    match tok {
        "np.pi" => return Some(std::f64::consts::PI),
        "np.e" => return Some(std::f64::consts::E),
        "float(\"inf\")" => return Some(f64::INFINITY),
        "float(\"-inf\")" => return Some(f64::NEG_INFINITY),
        "float(\"nan\")" => return Some(f64::NAN),
        _ => {}
    }
    // Bare or parenthesized, decimal or exact-fraction: the wrapping composes with
    // both grammars. The old paren branch `return`ed parse_pyfloat(inner) EVEN ON
    // None, so "(-1/3)" never reached the fraction arm below it (hardening H-012,
    // 2026-08-03 -- the engine-side reader had the same hole, kept aligned).
    fn bare_value(t: &str) -> Option<f64> {
        if let Some(v) = parse_pyfloat(t) {
            // An INTEGER spelling is an integer to the deployed evaluator (Python reads `-0`
            // as the int 0, `simplipy.utils.evaluator_literal`), and an integer has one zero:
            // +0.0. A float spelling keeps its IEEE sign (`-0.0` is -0.0, as Python's float
            // literal is).
            let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
            let integer = !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit());
            return Some(if integer && v == 0.0 { 0.0 } else { v });
        }
        // The AC core's exact-fraction literal `p/q` (`1/3`): the realized infix reads it as
        // Python `int / int` true division, which rounds ONCE to the nearest f64.
        fraction_value(t).map(|(v, _)| v)
    }
    if let Some(v) = bare_value(tok) {
        return Some(v);
    }
    if let Some(inner) = tok.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        return bare_value(inner);
    }
    None
}

/// The value of an exact-fraction leaf `p/q` as `(f64, certified)`, `None` if `t` is not one.
/// `certified`: the f64 is the correctly rounded value of the fraction -- what Python's
/// `int / int` gives -- established by exact midpoint tests via `Rat` when p and q fit
/// `i128`, and from the exact big-integer components beyond that (up to
/// `rat::READER_DIGITS` digits; `10^400/10^399` is 10.0, not inf/inf). Only beyond that limit
/// is the f64 `float(p) / float(q)`, which three roundings can leave up to three ulps from the
/// fraction (the old reader, B6: `673107593011939307760027002528 /
/// 810572757194796821120128085049` read 1.5 ulps low, outside the interval kernel's
/// one-ulp leaf bracket).
pub(crate) fn fraction_value(t: &str) -> Option<(f64, bool)> {
    let (p, q) = crate::utils::split_fraction(t)?;
    let (pv, qv) = (p.parse::<f64>().ok()?, q.parse::<f64>().ok()?);
    let nearest = match (p.parse::<i128>(), q.parse::<i128>()) {
        (Ok(pi), Ok(qi)) => {
            crate::ac::rat::Rat::new(pi, qi).and_then(|r| r.to_f64_nearest_certified())
        }
        _ => crate::ac::rat::token_nearest_f64(t),
    };
    Some(match nearest {
        Some(v) => (v, true),
        None => (pv / qv, false),
    })
}

/// Whether `leaf_value(tok)` is certified to be the f64 NEAREST to what the token denotes:
/// true for every leaf except an exact fraction `fraction_value` could not certify. (Decimal
/// literals are read by Rust's correctly rounding float parser; the special tokens have their
/// own enclosures.) The interval kernel's one-ulp leaf bracket is sound only around the
/// nearest f64.
pub(crate) fn leaf_value_is_nearest(tok: &str) -> bool {
    let t = tok
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(tok);
    parse_pyfloat(t).is_some() || fraction_value(t).is_none_or(|(_, certified)| certified)
}

/// H-045-R (owner Option B, 2026-08-05): sign and parity of an INTEGER-denoting decimal
/// literal, read from the SPELLING -- exact at every magnitude, no `i128`/f64 limit.
/// `Some((negative, odd))` iff the token is a decimal/e-notation literal whose denoted
/// value is a nonzero integer: the digit string `D` with decimal shift `s` denotes
/// `sign * D * 10^s`; an integer iff `s >= 0` or the last `|s|` digits are all zero.
/// Parity: any positive shift makes it even (a factor of 10); at shift 0 the last
/// effective digit decides. Zero mantissas, non-integers, fractions `p/q`, and anything
/// outside the plain decimal grammar return `None` -- callers refuse (fail closed).
pub(crate) fn integer_literal_parity(tok: &str) -> Option<(bool, bool)> {
    let t = tok
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(tok);
    let (neg, t) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let (mant, exp) = match t.find(['e', 'E']) {
        Some(i) => (&t[..i], t[i + 1..].parse::<i64>().ok()?),
        None => (t, 0i64),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], &mant[i + 1..]),
        None => (mant, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let digits: Vec<u8> = int_part.bytes().chain(frac_part.bytes()).collect();
    if digits.iter().all(|&b| b == b'0') {
        return None; // zero: no parity claim, callers refuse
    }
    let shift = exp.checked_sub(frac_part.len() as i64)?;
    let effective_last = if shift >= 0 {
        if shift > 0 {
            return Some((neg, false)); // a positive power of 10 factor: even
        }
        *digits.last().unwrap()
    } else {
        let drop = usize::try_from(-shift).ok()?;
        if drop >= digits.len() || digits[digits.len() - drop..].iter().any(|&b| b != b'0') {
            return None; // trailing digits carry mass: not an integer
        }
        digits[digits.len() - drop - 1]
    };
    Some((neg, (effective_last - b'0') % 2 == 1))
}

/// Python `float(token)` for the literal forms `is_numeric_string` admits. Handles a trailing dot
/// (`1.` -> 1.0) which Rust's `parse` rejects.
fn parse_pyfloat(tok: &str) -> Option<f64> {
    if let Ok(v) = tok.parse::<f64>() {
        return Some(v);
    }
    // `1.` / `-2.` : Rust rejects the trailing dot, Python accepts it.
    if let Some(stripped) = tok.strip_suffix('.') {
        if let Ok(v) = stripped.parse::<f64>() {
            return Some(v);
        }
    }
    None
}

/// Resolve a unary canonical operator to its scalar kernel as a (capture-free) function pointer, or
/// `None` if `name` is not a unary operator. Resolving ONCE (at tape-compile) lets the OFFLINE
/// vectorized evaluator (`crate::eval`) run a tight per-column loop with NO per-element string
/// dispatch -- the key to matching numpy's vectorized speed. Same semantics as before (exact
/// arithmetic, custom `inv` zero handling, real odd roots, libm transcendentals via the C symbols).
pub(crate) fn unary_fn(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        // The deployed vocabulary: the deployed kernels themselves (`kernels`), so the fold,
        // the tape and `simplipy.operators` compute one function.
        "neg" => kernels::neg,
        // inv: plain IEEE 1/x (1/+0 -> +inf, 1/-0 -> -inf) == the deployed realization.
        "inv" => kernels::inv,
        "abs" => kernels::abs,
        "sin" => kernels::sin,
        "cos" => kernels::cos,
        "tan" => kernels::tan,
        "asin" => kernels::asin,
        "acos" => kernels::acos,
        "atan" => kernels::atan,
        "sinh" => kernels::sinh,
        "cosh" => kernels::cosh,
        "tanh" => kernels::tanh,
        "asinh" => kernels::asinh,
        "acosh" => kernels::acosh,
        "atanh" => kernels::atanh,
        "exp" => kernels::exp,
        "log" => kernels::log,
        // The 0.15 heads: the deployed kernels themselves, like the rest.
        "floor" => kernels::floor,
        "ceil" => kernels::ceil,
        "harmonic" => kernels::harmonic,
        "erf" => kernels::erf,
        // The retired hyper-operator vocabulary (input support for configs that declare it).
        "mult2" => |x| 2.0 * x,
        "mult3" => |x| 3.0 * x,
        "mult4" => |x| 4.0 * x,
        "mult5" => |x| 5.0 * x,
        "div2" => |x| x / 2.0,
        "div3" => |x| x / 3.0,
        "div4" => |x| x / 4.0,
        "div5" => |x| x / 5.0,
        // pow{N} = `x ** N` (C pow, matches Python float.__pow__); integer exponent -> always real
        "pow2" => |x| aligned_pow(x, 2.0),
        "pow3" => |x| aligned_pow(x, 3.0),
        "pow4" => |x| aligned_pow(x, 4.0),
        "pow5" => |x| aligned_pow(x, 5.0),
        // pow1_2 / pow1_4 = `x ** 0.5` / `x ** 0.25`: negative base (incl. -inf) -> NaN.
        "pow1_2" => |x| aligned_pow(x, 0.5),
        "pow1_4" => |x| aligned_pow(x, 0.25),
        // pow1_3 / pow1_5: real cube/fifth root, sign-folded, always real (the live spelling is
        // `rootn`, whose odd arm matches this exactly)
        "pow1_3" => |x| real_odd_root(x, 1.0 / 3.0),
        "pow1_5" => |x| real_odd_root(x, 1.0 / 5.0),
        _ => return None,
    })
}

/// Resolve a binary canonical operator to its scalar kernel as a function pointer, or `None`.
pub(crate) fn binary_fn(name: &str) -> Option<fn(f64, f64) -> f64> {
    Some(match name {
        "+" => kernels::add,
        "-" => kernels::sub,
        "*" => kernels::mul,
        // `/`: plain IEEE x / y, the sign of a zero divisor participating
        "/" => kernels::div,
        // binary `pow` = `x ** y` = C pow (invalid -> NaN, overflow -> inf)
        "pow" => kernels::pow,
        // `rootn(x, n)`: IEEE-754 rootn, honest for EVERY integer index.
        "rootn" => kernels::rootn,
        // The 0.15 heads. `besselj(nu, x)`: the order is the first operand, a literal or
        // any expression.
        "max" => kernels::max,
        "min" => kernels::min,
        "mod" => kernels::modulo,
        "atan2" => kernels::atan2,
        "besselj" => kernels::besselj,
        _ => return None,
    })
}

/// THE f64 RANGE EVENT: the one definition the miner's scan (`eval::Tape::eval_columns_events`)
/// and the contract judge (`verify._contract.out_of_range`, through `_core.range_event`) share.
/// f64 promises the evaluator's value wherever nothing overflows or underflows, and overflows
/// are a special case: at a point where an operation of either side overflows or underflows, a
/// rewrite may give the true value instead of the evaluator's, as the constructors do
/// (`x1*x2/x2 -> x1`, `exp(x1)/exp(x1) -> 1`).
///
/// The two events are IEEE 754's. An operation OVERFLOWS when its operands are finite and its
/// exact value is finite, but the float is +-inf (`exp(800)`, `1e200*1e200`); it UNDERFLOWS
/// when its operands are finite and its exact value is nonzero, but the float is zero or
/// subnormal (`exp(-800)`, `1e-200*1e-200`). Neither is a POLE, an exact value that is itself
/// infinite (`1/0`, `log(0)`, `atanh(1)`), nor an EXACT ZERO (`x - x`, `log(1)`): those are
/// what the operator computes. So `atanh(tanh(20))` holds no event: `tanh` saturates to
/// exactly 1.0 inside the range, and `atanh(1.0)` is a pole. Each kind names where its
/// operators have a pole and an exact zero at finite operands; an operator outside the table
/// has no events, so its points are compared as any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RangeKind {
    /// `+ - neg abs mult2..mult5`: a result below the normal range is always exact.
    ExactWhenTiny,
    /// `exp cos cosh`: neither a pole nor an exact zero.
    Plain,
    /// `*`: an exact zero when an operand is zero.
    Mul,
    /// `/`: a pole at a zero divisor, an exact zero at a zero dividend.
    Div,
    /// `inv`: a pole at zero.
    Inv,
    /// `log`: a pole at zero, an exact zero at one.
    Log,
    /// `atanh`: poles at +-1, an exact zero at zero.
    Atanh,
    /// `pow`, `rootn`: a pole at a zero base under a negative exponent or index, an exact
    /// zero at a zero base.
    PowLike,
    /// `acos acosh`: an exact zero at one.
    ZeroAtOne,
    /// `sin tan asin atan sinh tanh asinh erf harmonic`, `div2..div5`, `pow2..pow5`,
    /// `pow1_2..pow1_5`: an exact zero at zero. (`harmonic` is NaN, not infinite, at its
    /// poles, so it has none.)
    ZeroAtZero,
    /// `atan2(y, x)`: an exact zero at `y = 0` with `x` positive or `+0`; no pole.
    Atan2,
    /// `besselj(nu, x)`: a pole at `x = 0` under a negative non-integer order, an exact zero
    /// at `x = 0` under any other nonzero order.
    Besselj,
}

impl RangeKind {
    /// The kind of an operator token, `None` outside the table.
    pub(crate) fn of(name: &str) -> Option<RangeKind> {
        use RangeKind::*;
        Some(match name {
            "+" | "-" | "neg" | "abs" | "mult2" | "mult3" | "mult4" | "mult5" => ExactWhenTiny,
            // integers or +-0, an operand, or an exact remainder moved by one addition of
            // the divisor (a subnormal sum is exact): never a rounded tiny result
            "floor" | "ceil" | "max" | "min" | "mod" => ExactWhenTiny,
            "erf" | "harmonic" => ZeroAtZero,
            "atan2" => Atan2,
            "besselj" => Besselj,
            "exp" | "cos" | "cosh" => Plain,
            "*" => Mul,
            "/" => Div,
            "inv" => Inv,
            "log" => Log,
            "atanh" => Atanh,
            "pow" | "rootn" => PowLike,
            "acos" | "acosh" => ZeroAtOne,
            "sin" | "tan" | "asin" | "atan" | "sinh" | "tanh" | "asinh" | "div2" | "div3"
            | "div4" | "div5" | "pow2" | "pow3" | "pow4" | "pow5" | "pow1_2" | "pow1_3"
            | "pow1_4" | "pow1_5" => ZeroAtZero,
            _ => return None,
        })
    }

    /// Is the exact value infinite at these finite operands?
    fn pole(self, args: &[f64]) -> bool {
        match self {
            RangeKind::Div => args[1] == 0.0,
            RangeKind::Inv | RangeKind::Log => args[0] == 0.0,
            RangeKind::Atanh => args[0] == 1.0 || args[0] == -1.0,
            RangeKind::PowLike => args[0] == 0.0 && args[1] < 0.0,
            RangeKind::Besselj => args[1] == 0.0 && args[0] < 0.0 && args[0].fract() != 0.0,
            _ => false,
        }
    }

    /// Is the exact value zero at these finite operands?
    fn exact_zero(self, args: &[f64]) -> bool {
        match self {
            RangeKind::Mul => args[0] == 0.0 || args[1] == 0.0,
            RangeKind::Div | RangeKind::PowLike | RangeKind::Atanh | RangeKind::ZeroAtZero => {
                args[0] == 0.0
            }
            RangeKind::Log | RangeKind::ZeroAtOne => args[0] == 1.0,
            RangeKind::Atan2 => {
                args[0] == 0.0 && (args[1] > 0.0 || (args[1] == 0.0 && args[1].is_sign_positive()))
            }
            RangeKind::Besselj => {
                args[1] == 0.0 && args[0] != 0.0 && (args[0] > 0.0 || args[0].fract() == 0.0)
            }
            RangeKind::ExactWhenTiny | RangeKind::Plain | RangeKind::Inv => false,
        }
    }

    /// Did this operation, on `args`, overflow or underflow to `value`?
    pub(crate) fn event(self, args: &[f64], value: f64) -> bool {
        if !args.iter().all(|a| a.is_finite()) {
            return false;
        }
        if value.is_infinite() {
            return !self.pole(args);
        }
        if value == 0.0 || value.abs() < f64::MIN_POSITIVE {
            return self != RangeKind::ExactWhenTiny && !self.exact_zero(args);
        }
        false
    }
}

/// `RangeKind::event` by operator name: `None` for an operator outside the table.
pub(crate) fn range_event(name: &str, args: &[f64], value: f64) -> Option<bool> {
    RangeKind::of(name).map(|k| k.event(args, value))
}

/// Apply one canonical operator to its numeric operands as IEEE-754 f64 + libm. Thin dispatcher over
/// `unary_fn`/`binary_fn` (ONE semantics source of truth, shared by the inline folder and the OFFLINE
/// vectorized evaluator). `None` for an unknown operator or wrong operand count.
pub(crate) fn apply_op(name: &str, a: &[f64]) -> Option<f64> {
    match a.len() {
        1 => unary_fn(name).map(|f| f(a[0])),
        2 => binary_fn(name).map(|f| f(a[0], a[1])),
        _ => None,
    }
}

/// IEEE-754 `rootn(x, n)`, honest for every integer index:
/// odd index = the signed root (total on R, `-0` at `-0`); even = the principal root (NaN on
/// negatives, `+0` at either zero, identical to `pow(x, 1/n)`); n == 1 = the identity; n < 0 =
/// the reciprocal of the positive-index root; n == 0 or a non-integer index = NaN (invalid
/// operation).
/// One semantics, five surfaces: this table, `operators.py::rootn`, `interval.rs`, and
/// both `hiprec.rs` evaluators -- pinned against each other by the parity tests.
pub(crate) fn rootn_f64(x: f64, n: f64) -> f64 {
    if n.fract() != 0.0 || n == 0.0 || !n.is_finite() {
        return f64::NAN;
    }
    let k = n.abs();
    // Parity via exact float remainder: every representable float >= 2^53 is an even
    // integer (spacing >= 2), so `% 2.0` is exact at every magnitude.
    let root = if k == 1.0 {
        x
    } else if k % 2.0 == 1.0 {
        real_odd_root(x, 1.0 / k)
    } else if x < 0.0 {
        f64::NAN
    } else {
        elementary::pow(x, 1.0 / k)
    };
    if n < 0.0 {
        1.0 / root
    } else {
        root
    }
}

/// The real odd root (`rootn`'s odd arm; the retired `pow1_3`/`pow1_5` spellings): `|x|**r`
/// carrying the sign of `x`, the sign of zero included (`rootn(-0, 3) = -0`, as IEEE-754 `rootn`
/// and every other odd function give).
fn real_odd_root(x: f64, r: f64) -> f64 {
    elementary::pow(x.abs(), r).copysign(x)
}

/// CPython-exact `str(float)` as the result formatter uses it:
/// nan/inf -> `float("...")` tokens; integer-valued -> `str(int(x))`; else `str(float)`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "float(\"nan\")".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 {
            "float(\"-inf\")"
        } else {
            "float(\"inf\")"
        }
        .to_string();
    }
    if x == x.trunc() {
        // `result == int(result)` -> `str(int(result))` (the exact integer; -0.0 -> "0").
        if x == 0.0 {
            return "0".to_string();
        }
        return format!("{x:.0}"); // exact integer digits (matches Python str(int) incl. large values)
    }
    // non-integer finite: Python `str(float)`. Get the shortest correctly-rounded significant digits
    // from ryu (round-half-even, matching CPython's dtoa; Rust std `{}`/`{:e}` differs on the rare
    // equidistant tie), then lay them out per Python's notation (sci iff leading-digit exponent < -4;
    // the >=16 threshold never applies -- non-integer floats are < 2^52 < 1e16).
    let mut buf = ryu::Buffer::new();
    let (neg, digits, dexp) = shortest_digits(buf.format_finite(x));
    let sign = if neg { "-" } else { "" };
    if dexp < -4 {
        // scientific: d[0].d[1..]e±NN
        let mant = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{sign}{mant}e{}{:02}",
            if dexp < 0 { "-" } else { "+" },
            dexp.unsigned_abs()
        )
    } else if dexp >= 0 {
        // fixed, |x| >= 1: integer part is digits[..dexp+1] (no padding needed -- non-integer means a
        // fractional digit exists, so digits.len() > dexp+1), fraction is the rest.
        let split = (dexp as usize) + 1;
        format!("{sign}{}.{}", &digits[..split], &digits[split..])
    } else {
        // fixed, 0 < |x| < 1: "0." + (-dexp-1) zeros + digits
        format!("{sign}0.{}{}", "0".repeat((-dexp - 1) as usize), digits)
    }
}

/// Decompose a ryu-formatted finite decimal into `(negative, significant_digits, leading_exp)` where
/// the value is `±d[0].d[1..] x 10^leading_exp` (digits stripped of leading/trailing zeros). ryu emits
/// forms like `1.5`, `0.001`, `1.23e-5`, `1e100`.
fn shortest_digits(s: &str) -> (bool, String, i32) {
    let neg = s.starts_with('-');
    let s = s.strip_prefix('-').unwrap_or(s);
    let (mant, eexp) = match s.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (s, 0),
    };
    let (ipart, fpart) = mant.split_once('.').unwrap_or((mant, ""));
    let raw = format!("{ipart}{fpart}");
    // value = int(raw) x 10^k, with k = eexp - len(fpart)
    let mut k = eexp - fpart.len() as i32;
    let stripped = raw.trim_start_matches('0'); // leading zeros don't change the integer value
    let trailing = stripped.len() - stripped.trim_end_matches('0').len();
    let digits = stripped.trim_end_matches('0').to_string();
    k += trailing as i32; // each dropped trailing zero raises the exponent of the last digit
    let digits = if digits.is_empty() {
        "0".to_string()
    } else {
        digits
    };
    let dexp = k + digits.len() as i32 - 1; // exponent of the leading digit
    (neg, digits, dexp)
}

#[cfg(test)]
mod tests {
    use super::py_float_repr;
    use crate::Engine;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    fn ev(e: &Engine, toks: &[&str]) -> Option<String> {
        e.evaluate_constant_subtree(&toks.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    /// Exact (non-transcendental) folds + the custom operator semantics. Transcendental folds go
    /// through the host libm, so their last bit is the platform's; they are not asserted here.
    #[test]
    fn evaluate_exact_cases() {
        let Some(e) = engine() else { return };
        assert_eq!(ev(&e, &["+", "2", "3"]).as_deref(), Some("5"));
        assert_eq!(ev(&e, &["-", "3", "5"]).as_deref(), Some("-2"));
        assert_eq!(ev(&e, &["/", "7", "2"]).as_deref(), Some("3.5"));
        assert_eq!(ev(&e, &["neg", "-5"]).as_deref(), Some("5"));
        assert_eq!(ev(&e, &["pow", "-3", "2"]).as_deref(), Some("9"));
        assert_eq!(ev(&e, &["rootn", "4", "2"]).as_deref(), Some("2"));
        assert_eq!(ev(&e, &["rootn", "-8", "3"]).as_deref(), Some("-2"));
        assert_eq!(ev(&e, &["*", "2", "3"]).as_deref(), Some("6"));
        // fold to the IEEE-754 f64 result, including inf/nan tokens.
        assert_eq!(ev(&e, &["/", "1", "0"]).as_deref(), Some("float(\"inf\")"));
        assert_eq!(
            ev(&e, &["/", "-1", "0"]).as_deref(),
            Some("float(\"-inf\")")
        );
        assert_eq!(ev(&e, &["/", "0", "0"]).as_deref(), Some("float(\"nan\")"));
        assert_eq!(ev(&e, &["inv", "0"]).as_deref(), Some("float(\"inf\")"));
        assert_eq!(
            ev(&e, &["rootn", "-1", "2"]).as_deref(),
            Some("float(\"nan\")")
        );
        // sqrt(-1) = nan
    }

    /// The CPython-exact float formatter (ryu digits + Python notation): integer-valued -> str(int),
    /// fixed for exp in [-4, ..), sci for exp < -4; nan/inf tokens; -0.0 -> "0".
    #[test]
    fn float_repr_python_exact() {
        assert_eq!(py_float_repr(0.0), "0");
        assert_eq!(py_float_repr(-0.0), "0");
        assert_eq!(py_float_repr(3.0), "3");
        assert_eq!(py_float_repr(2.5), "2.5");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(9.9e-5), "9.9e-05");
        assert_eq!(py_float_repr(0.30000000000000004), "0.30000000000000004");
        assert_eq!(py_float_repr(1e16), "10000000000000000");
        assert_eq!(py_float_repr(1e20), "100000000000000000000");
        assert_eq!(py_float_repr(f64::NAN), "float(\"nan\")");
        assert_eq!(py_float_repr(f64::INFINITY), "float(\"inf\")");
        assert_eq!(py_float_repr(f64::NEG_INFINITY), "float(\"-inf\")");
    }
}

#[cfg(test)]
mod aligned_pow_tests {
    use super::*;

    /// THE f64 RANGE EVENT: every operator with a kernel has a kind, and the kinds read
    /// IEEE 754's overflow and underflow, never a pole, an exact zero or a saturation.
    #[test]
    fn range_events_are_ieee_overflow_and_underflow() {
        let unary = [
            "neg", "inv", "abs", "mult2", "mult3", "mult4", "mult5", "div2", "div3", "div4",
            "div5", "pow2", "pow3", "pow4", "pow5", "pow1_2", "pow1_3", "pow1_4", "pow1_5", "sin",
            "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh",
            "atanh", "exp", "log", "floor", "ceil", "harmonic", "erf",
        ];
        for op in unary {
            assert!(
                unary_fn(op).is_some() && RangeKind::of(op).is_some(),
                "{op}"
            );
        }
        for op in [
            "+", "-", "*", "/", "pow", "rootn", "max", "min", "mod", "atan2", "besselj",
        ] {
            assert!(
                binary_fn(op).is_some() && RangeKind::of(op).is_some(),
                "{op}"
            );
        }
        assert_eq!(RangeKind::of("no-such-operator"), None);
        let ev = |op: &str, args: &[f64]| {
            let v = apply_op(op, args).unwrap();
            range_event(op, args, v).unwrap()
        };
        // overflow and underflow
        for (op, args) in [
            ("exp", vec![800.0]),
            ("exp", vec![-800.0]),
            ("exp", vec![-740.0]), // subnormal
            ("cosh", vec![800.0]),
            ("*", vec![1e200, 1e200]),
            ("*", vec![1e-200, 1e-200]),
            ("+", vec![1e308, 1e308]),
            ("pow", vec![10.0, 400.0]),
            ("inv", vec![1e308]),
            ("rootn", vec![5e-324, -1.0]),
            ("erf", vec![5e-324]),
            ("atan2", vec![1e-300, 1e300]),
            ("besselj", vec![200.0, 1e-3]),
        ] {
            assert!(ev(op, &args), "{op}{args:?} must be a range event");
        }
        // poles, exact zeros, saturation inside the range, non-finite operands
        for (op, args) in [
            ("atanh", vec![1.0]),
            ("log", vec![0.0]),
            ("/", vec![1.0, 0.0]),
            ("pow", vec![0.0, -1.0]),
            ("rootn", vec![0.0, -2.0]),
            ("inv", vec![0.0]),
            ("-", vec![0.5, 0.5]),
            ("log", vec![1.0]),
            ("acos", vec![1.0]),
            ("*", vec![0.0, 1e-200]),
            ("tanh", vec![20.0]),
            ("exp", vec![f64::INFINITY]),
            ("exp", vec![700.0]),
            ("erf", vec![0.0]),
            ("harmonic", vec![0.0]),
            ("harmonic", vec![-2.0]),
            ("ceil", vec![-0.5]),
            ("mod", vec![3.0, 3.0]),
            ("mod", vec![5e-324, 1e-320]),
            ("max", vec![5e-324, -1.0]),
            ("atan2", vec![0.0, 1.0]),
            ("atan2", vec![0.0, 0.0]),
            ("besselj", vec![1.0, 0.0]),
            ("besselj", vec![-2.0, 0.0]),
            ("besselj", vec![-0.5, 0.0]),
        ] {
            assert!(!ev(op, &args), "{op}{args:?} must not be a range event");
        }
    }

    /// The zero spellings read as the deployed evaluator reads them: an integer has one zero
    /// (+0), a float spelling keeps its sign, a fraction is `int / int` (+0). The Python
    /// side's twin is `tests/test_evaluator_literals.py`.
    #[test]
    fn an_integer_has_one_zero() {
        let sign = |t: &str| leaf_value(t).unwrap().is_sign_negative();
        for t in ["0", "-0", "+0", "00", "-00", "(-0)", "0/5", "-0/5"] {
            assert!(!sign(t), "{t} must read +0");
        }
        for t in ["-0.0", "-.0", "-0.", "-0e0", "(-0.0)"] {
            assert!(sign(t), "{t} must read -0");
        }
        for t in ["0.0", "+0.0", "0e5"] {
            assert!(!sign(t), "{t} must read +0");
        }
    }

    /// The contract's normative pow cells at negative and infinite bases: NaN for
    /// negative base (incl. −∞) with finite non-integer exponent; integer exponents
    /// and the magnitude-step infinite-exponent cells unchanged from C99.
    #[test]
    fn negative_infinite_base_non_integer_exponent_is_nan() {
        for op in ["pow1_2", "pow1_4"] {
            let f = unary_fn(op).unwrap();
            assert!(f(f64::NEG_INFINITY).is_nan(), "{op}(-inf) must be NaN");
            assert!(f(-1.0).is_nan(), "{op}(-1) must be NaN");
        }
        let p = binary_fn("pow").unwrap();
        assert!(p(f64::NEG_INFINITY, 0.25).is_nan());
        assert!(p(f64::NEG_INFINITY, 0.5).is_nan());
        assert!(p(-2.0, 0.5).is_nan());
    }

    #[test]
    fn ratified_cells_unchanged() {
        let p = binary_fn("pow").unwrap();
        assert_eq!(p(f64::NEG_INFINITY, 2.0), f64::INFINITY);
        assert_eq!(p(f64::NEG_INFINITY, 3.0), f64::NEG_INFINITY);
        // §9.4 magnitude-step at infinite exponents, including negative t:
        assert_eq!(p(f64::NEG_INFINITY, f64::INFINITY), f64::INFINITY);
        assert_eq!(p(f64::NEG_INFINITY, f64::NEG_INFINITY), 0.0);
        assert_eq!(p(-2.0, f64::INFINITY), f64::INFINITY);
        assert_eq!(p(-0.5, f64::INFINITY), 0.0);
        assert_eq!(p(-1.0, f64::INFINITY), 1.0);
        // IEEE-definitional nan-pow cells (ratified):
        assert_eq!(p(1.0, f64::NAN), 1.0);
        assert_eq!(p(f64::NAN, 0.0), 1.0);
        // -0.0 is not a negative base, and a fractional power has no odd part to sign: +0
        assert_eq!(p(-0.0, 0.5).to_bits(), 0.0f64.to_bits());
        // odd real roots stay sign-folded and defined:
        let r3 = unary_fn("pow1_3").unwrap();
        assert_eq!(r3(-8.0), -2.0);
    }

    /// IEEE-754 `rootn` at a zero: an odd index keeps the sign of zero (and a negative odd index
    /// turns it into the infinity of that sign), an even index gives `+0` (`+inf`) at either zero.
    /// Compared by bits, since `-0.0 == 0.0`.
    #[test]
    fn rootn_keeps_the_sign_of_zero_under_an_odd_index() {
        let r = binary_fn("rootn").unwrap();
        let bits = |v: f64| v.to_bits();
        for (n, at_neg, at_pos) in [
            (1.0, -0.0, 0.0),
            (3.0, -0.0, 0.0),
            (5.0, -0.0, 0.0),
            (2.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (-1.0, f64::NEG_INFINITY, f64::INFINITY),
            (-3.0, f64::NEG_INFINITY, f64::INFINITY),
            (-2.0, f64::INFINITY, f64::INFINITY),
        ] {
            assert_eq!(bits(r(-0.0, n)), bits(at_neg), "rootn(-0, {n})");
            assert_eq!(bits(r(0.0, n)), bits(at_pos), "rootn(+0, {n})");
        }
        assert_eq!(bits(unary_fn("pow1_3").unwrap()(-0.0)), bits(-0.0));
    }
}
