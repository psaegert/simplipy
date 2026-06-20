//! Numeric constant folding (the `numeric` engine line): a native f64 evaluator + a CPython-exact
//! `str(float)` formatter, replacing the Python `_evaluate_constant_subtree` (engine.py:1073) which
//! went through `operators_to_realizations` -> `prefix_to_infix` -> `codify` -> `code_to_lambda`.
//!
//! Parity policy (user decision: "numerically correct"): the evaluator uses the SAME primitives as
//! Python -- libm (`powf`/`sin`/...) and exact f64 arithmetic, with the engine's custom operator
//! semantics (the `div`/`inv` zero handling, the real cube/fifth roots, `pow1_2`/`pow1_4` of a
//! negative base yielding a COMPLEX result -> no fold, binary `pow` via `np.power` -> NaN). This is
//! the same platform-libm parity Python itself has; values match bit-for-bit on a given build, and
//! the float repr matches `str()` byte-for-byte. Result `None` mirrors Python returning `None`
//! (unfoldable: complex result, unparseable leaf, unknown operator).

use crate::operators::Operators;

/// The transcendentals are evaluated through the SYSTEM libm (the same `<math.h>` functions Python's
/// `math` module calls), NOT Rust's `f64::*` methods. Rust's std calls the platform libm for most of
/// them (so they already match Python at 0 ULP), but `f64::atanh` is Rust's OWN implementation and
/// differs from glibc by up to ~86 ULP. Binding the C symbols directly guarantees Python==Rust on every
/// platform (both use that platform's libm) -- the determinism the f64+libm folding spec needs.
mod cmath {
    extern "C" {
        pub fn sin(x: f64) -> f64;
        pub fn cos(x: f64) -> f64;
        pub fn tan(x: f64) -> f64;
        pub fn asin(x: f64) -> f64;
        pub fn acos(x: f64) -> f64;
        pub fn atan(x: f64) -> f64;
        pub fn sinh(x: f64) -> f64;
        pub fn cosh(x: f64) -> f64;
        pub fn tanh(x: f64) -> f64;
        pub fn asinh(x: f64) -> f64;
        pub fn acosh(x: f64) -> f64;
        pub fn atanh(x: f64) -> f64;
        pub fn exp(x: f64) -> f64;
        pub fn log(x: f64) -> f64;
        pub fn pow(x: f64, y: f64) -> f64;
    }
}

/// `simplipy.operators.pow` / the `pow{N}` ops / the real roots all reduce to `x ** y` = C `pow`
/// (CPython `float.__pow__`). Route through the system `pow` for guaranteed Python parity.
///
/// `black_box` the operands so the optimizer cannot specialize a constant exponent into a different
/// instruction sequence: LLVM rewrites `pow(x, 2.0)` -> `x * x` (and, under fast-math, `pow(x, 0.5)`
/// -> `sqrt`), which is ~1 ULP off the libm `pow` result. Without this, `pow2` would silently diverge
/// from both the binary `pow` operator and the pure-Python folder, and the divergence would depend on
/// the opt-level / LLVM version. Forcing the real call keeps every `pow{N}` realization bit-identical
/// to `pow(x, N)` and to the Python `_apply_numeric_op` (which routes through the same libm `pow`).
#[inline]
fn cpow(x: f64, y: f64) -> f64 {
    unsafe { cmath::pow(std::hint::black_box(x), std::hint::black_box(y)) }
}

/// Faithful port of `_evaluate_constant_subtree` (engine.py:1073): evaluate an all-numeric prefix
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

/// Recursive prefix evaluation: an operator consumes `arity` operands; a leaf is `float(token)`.
fn eval_node(tokens: &[String], idx: &mut usize, ops: &Operators) -> Option<f64> {
    let tok = tokens.get(*idx)?.clone();
    *idx += 1;
    match ops.arity_of(&tok) {
        Some(arity) => {
            let mut args = [0.0f64; 2];
            let arity = arity as usize;
            if arity > 2 {
                return None; // no arity>2 operator exists in dev_7-3
            }
            for a in args.iter_mut().take(arity) {
                *a = eval_node(tokens, idx, ops)?;
            }
            apply_op(&tok, &args[..arity])
        }
        // Leaf: Python `float(token)`. Rust `parse::<f64>` agrees on every `is_numeric_string`-true
        // token the grammar emits (the `1.`/`.5`/`1e3` forms); a non-parseable leaf -> None (Python
        // would raise inside the eval -> None).
        None => parse_pyfloat(&tok),
    }
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

/// Apply one canonical operator to its numeric operands as IEEE-754 f64 + libm: exact arithmetic, the
/// custom `div`/`inv` signed-zero results, the real cube/fifth roots, and libm transcendentals. inf/nan
/// propagate naturally and are filtered by the finiteness gate in `evaluate_constant_subtree`. `None`
/// only for an unknown operator.
fn apply_op(name: &str, a: &[f64]) -> Option<f64> {
    let v = match name {
        // bare arithmetic realizations
        "+" => a[0] + a[1],
        "-" => a[0] - a[1],
        "*" => a[0] * a[1],
        // `/` realization is `operators.div`: scalar zero-divisor -> signed inf / nan (operators.py:29)
        "/" => op_div(a[0], a[1]),
        // binary `pow` = `x ** y` = C pow (invalid -> NaN, overflow -> inf)
        "pow" => cpow(a[0], a[1]),
        "neg" => -a[0],
        // inv: x == 0 (incl -0.0) -> +inf (operators.py:15)
        "inv" => {
            if a[0] == 0.0 {
                f64::INFINITY
            } else {
                1.0 / a[0]
            }
        }
        "abs" => a[0].abs(),
        "mult2" => 2.0 * a[0],
        "mult3" => 3.0 * a[0],
        "mult4" => 4.0 * a[0],
        "mult5" => 5.0 * a[0],
        "div2" => a[0] / 2.0,
        "div3" => a[0] / 3.0,
        "div4" => a[0] / 4.0,
        "div5" => a[0] / 5.0,
        // pow{N} = `x ** N` (C pow, matches Python float.__pow__); integer exponent -> always real
        "pow2" => cpow(a[0], 2.0),
        "pow3" => cpow(a[0], 3.0),
        "pow4" => cpow(a[0], 4.0),
        "pow5" => cpow(a[0], 5.0),
        // pow1_2 / pow1_4 = `x ** 0.5` / `x ** 0.25`: a negative base gives NaN -> caught by the
        // finiteness gate (-> no fold), which is what we want (keep `sqrt(-1)` symbolic).
        "pow1_2" => cpow(a[0], 0.5),
        "pow1_4" => cpow(a[0], 0.25),
        // pow1_3 / pow1_5: real cube/fifth root (operators.py:121,164) -- sign-folded, always real
        "pow1_3" => real_odd_root(a[0], 1.0 / 3.0),
        "pow1_5" => real_odd_root(a[0], 1.0 / 5.0),
        // transcendentals via the SYSTEM libm (same functions Python's `math` calls)
        "sin" => unsafe { cmath::sin(a[0]) },
        "cos" => unsafe { cmath::cos(a[0]) },
        "tan" => unsafe { cmath::tan(a[0]) },
        "asin" => unsafe { cmath::asin(a[0]) },
        "acos" => unsafe { cmath::acos(a[0]) },
        "atan" => unsafe { cmath::atan(a[0]) },
        "sinh" => unsafe { cmath::sinh(a[0]) },
        "cosh" => unsafe { cmath::cosh(a[0]) },
        "tanh" => unsafe { cmath::tanh(a[0]) },
        "asinh" => unsafe { cmath::asinh(a[0]) },
        "acosh" => unsafe { cmath::acosh(a[0]) },
        "atanh" => unsafe { cmath::atanh(a[0]) },
        "exp" => unsafe { cmath::exp(a[0]) },
        "log" => unsafe { cmath::log(a[0]) },
        _ => return None, // unknown operator -> unfoldable
    };
    Some(v)
}

/// `operators.div` scalar branch (operators.py:29-53): `y==0` -> `x>0`:+inf, `x<0`:-inf, `x==0`:nan.
fn op_div(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        if x > 0.0 {
            f64::INFINITY
        } else if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        }
    } else {
        x / y
    }
}

/// `pow1_3`/`pow1_5` real odd root (operators.py:121,164): `x<0 -> -(-x)**r` else `x**r`.
fn real_odd_root(x: f64, r: f64) -> f64 {
    if x < 0.0 {
        -cpow(-x, r)
    } else {
        cpow(x, r)
    }
}

/// CPython-exact `str(float)` as the result formatter uses it (engine.py:1085-1093):
/// nan/inf -> `float("...")` tokens; integer-valued -> `str(int(x))`; else `str(float)`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "float(\"nan\")".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "float(\"-inf\")" } else { "float(\"inf\")" }.to_string();
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
        format!("{sign}{mant}e{}{:02}", if dexp < 0 { "-" } else { "+" }, dexp.unsigned_abs())
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
    let digits = if digits.is_empty() { "0".to_string() } else { digits };
    let dexp = k + digits.len() as i32 - 1; // exponent of the leading digit
    (neg, digits, dexp)
}

#[cfg(test)]
mod tests {
    use super::py_float_repr;
    use crate::Engine;

    fn engine() -> Engine {
        let home = std::env::var("HOME").unwrap();
        Engine::from_paths(
            &format!("{home}/.cache/simplipy/engines/dev_7-3/config.yaml"),
            &format!("{home}/.cache/simplipy/engines/dev_7-3/rules.json"),
        )
        .expect("engine loads")
    }

    fn ev(e: &Engine, toks: &[&str]) -> Option<String> {
        e.evaluate_constant_subtree(&toks.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    /// Exact (non-transcendental) folds + the custom operator semantics. Transcendental folds are
    /// numerically-correct-modulo-numpy-ULP (see benchmarks/diff_numeric.py); not asserted here.
    #[test]
    fn evaluate_exact_cases() {
        let e = engine();
        assert_eq!(ev(&e, &["+", "2", "3"]).as_deref(), Some("5"));
        assert_eq!(ev(&e, &["-", "3", "5"]).as_deref(), Some("-2"));
        assert_eq!(ev(&e, &["/", "7", "2"]).as_deref(), Some("3.5"));
        assert_eq!(ev(&e, &["neg", "-5"]).as_deref(), Some("5"));
        assert_eq!(ev(&e, &["pow2", "-3"]).as_deref(), Some("9"));
        assert_eq!(ev(&e, &["pow1_2", "4"]).as_deref(), Some("2"));
        assert_eq!(ev(&e, &["pow1_3", "-8"]).as_deref(), Some("-2"));
        assert_eq!(ev(&e, &["mult2", "3"]).as_deref(), Some("6"));
        // fold to the IEEE-754 f64 result, including inf/nan tokens.
        assert_eq!(ev(&e, &["/", "1", "0"]).as_deref(), Some("float(\"inf\")"));
        assert_eq!(ev(&e, &["/", "-1", "0"]).as_deref(), Some("float(\"-inf\")"));
        assert_eq!(ev(&e, &["/", "0", "0"]).as_deref(), Some("float(\"nan\")"));
        assert_eq!(ev(&e, &["inv", "0"]).as_deref(), Some("float(\"inf\")"));
        assert_eq!(ev(&e, &["pow1_2", "-1"]).as_deref(), Some("float(\"nan\")")); // sqrt(-1) = nan
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
