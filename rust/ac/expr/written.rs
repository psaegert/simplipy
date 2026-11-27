//! THE WRITTEN PRICE: [`complexity`]'s code book applied to an expression exactly as written
//! ([`written_complexity`]).
//!
//! [`complexity`] prices the CANONICAL state a mode reads an expression into; the written price
//! reads nothing into any state. It charges every operator, function, variable, sign and number
//! the expression writes, each at its entry of the one code book (`mu_leaf`, `mu_add`, `mu_mul`,
//! `mu_pow`, the head table, `MU_SIGN`, the literal code), and makes only the identifications
//! the infix notation forces (owner's ruling relayed 2026-11-26: srbf measures ground truths and
//! predictions "exactly as written, without any modifications that either benefit the benchmark
//! or the model"). Where the notation leaves a choice, the identification is the one the code
//! book prices (the overseer's decisions M1, S1, K and L, 2026-11-27), and a quotient of two
//! numbers is their division (owner's ruling 2026-11-27: "As written, fractions should cost their
//! division, and numbers their value").

use super::*;
use crate::operators::Operators;

/// A node of the expression as written: a leaf token, or a head over its operands.
enum Written<'a> {
    Leaf(&'a str),
    Node(&'a str, Vec<Written<'a>>),
}

/// A one-token fraction `p/q` with `|p| != 1` (`3/4`, `(-3/4)`) as the quotient it writes, the
/// leaves `p` and `q` under `/`: it costs its division, as the infix `3/4` and the prefix
/// `/ 3 4` do (the fraction ruling). `1/q` and `-1/q` stay one literal (`None`).
fn fraction_quotient(token: &str) -> Option<Written<'_>> {
    let s = unwrapped(token);
    let (p, q) = s.split_once('/')?;
    if !crate::utils::is_numeric_string(s) || literal_value(p)?.abs().is_one() {
        return None;
    }
    Some(Written::Node("/", vec![Written::Leaf(p), Written::Leaf(q)]))
}

/// Read explicit prefix tokens into the tree they write. A head is a declared or core operator
/// or a defined function (`Operators::syntax_arity`), by its own name: an alias or `**` is
/// refused, as every other prefix entry refuses it ([`complexity`] among them; infix text reads
/// an alias as its target before it gets here). Every other token is a leaf, a one-token
/// fraction the quotient it writes ([`fraction_quotient`]).
fn read<'a>(tokens: &'a [String], ops: &'a Operators) -> Result<Written<'a>, String> {
    if let Some(t) = tokens.iter().find(|t| {
        matches!(
            t.as_str(),
            "<add>"
                | "</add>"
                | "<mul>"
                | "</mul>"
                | "<sub>"
                | "<div>"
                | "<max>"
                | "</max>"
                | "<min>"
                | "</min>"
        )
    }) {
        return Err(format!(
            "written_complexity reads infix or explicit prefix tokens, not the tagged token \
             {t:?}; convert the sequence with to_prefix first"
        ));
    }
    fn one<'a>(
        tokens: &'a [String],
        at: usize,
        ops: &'a Operators,
    ) -> Result<(Written<'a>, usize), String> {
        let token = tokens
            .get(at)
            .ok_or("written_complexity: an operand is missing (malformed expression)")?
            .as_str();
        let target = if token == "**" {
            Some("pow")
        } else {
            ops.operator_aliases.get(token).map(String::as_str)
        };
        if let Some(target) = target {
            return Err(format!(
                "written_complexity: the prefix token {token:?} is an alias of {target:?}: prefix \
                 tokens name the engine's operators as every prefix entry reads them (infix text \
                 reads an alias as its target)"
            ));
        }
        match ops.syntax_arity(token) {
            None => Ok((
                fraction_quotient(token).unwrap_or(Written::Leaf(token)),
                at + 1,
            )),
            Some(n) => {
                let mut kids = Vec::with_capacity(usize::from(n));
                let mut next = at + 1;
                for _ in 0..n {
                    let (kid, after) = one(tokens, next, ops)?;
                    kids.push(kid);
                    next = after;
                }
                Ok((Written::Node(token, kids), next))
            }
        }
    }
    if tokens.is_empty() {
        return Err("written_complexity: the expression is empty".to_string());
    }
    let (tree, next) = one(tokens, 0, ops)?;
    if next != tokens.len() {
        return Err(format!(
            "written_complexity: {} token(s) after a complete expression: the tokens are not \
             one expression over this engine's operators and defined functions (an undeclared \
             function head is refused, never guessed)",
            tokens.len() - next
        ));
    }
    Ok(tree)
}

/// A literal's own spelling inside the parentheses the engine's prefix dialect may wrap a
/// negative literal in (`(-3)`).
fn unwrapped(s: &str) -> &str {
    s.strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .filter(|x| crate::utils::is_numeric_string(x))
        .unwrap_or(s)
}

/// Is a leaf token a numeric literal (the numeral grammar, signed, fractions included)?
fn literal(s: &str) -> bool {
    numeric_leaf(unwrapped(s))
}

/// THE LITERAL CODE on the exact value written, every digit ([`mu_numeric_leaf`]: [`mu_rat`] on
/// the exact rational the token spells, its print's price beyond the 1,100-bit cap); a written
/// minus is one sign bit on the magnitude, so `-0` costs the zero and a sign, and every other
/// negative literal costs exactly what [`mu_rat`] charges its value.
fn literal_price(s: &str) -> u64 {
    let s = unwrapped(s);
    match s.strip_prefix('-') {
        Some(magnitude) => MU_SIGN.saturating_add(literal_price(magnitude)),
        None => mu_numeric_leaf(s),
    }
}

/// The exact value of a literal token, in the exact number domain (`None` beyond its cap).
fn literal_value(s: &str) -> Option<Rat> {
    let _exact = crate::ac::rat::number_domain(false);
    let (p, q) = crate::ac::rat::token_rational(unwrapped(s))?;
    Rat::from_big(p, q)
}

/// The exact value of a literal LEAF (`None` for any other node, and beyond the exact domain).
fn leaf_value(n: &Written) -> Option<Rat> {
    match n {
        Written::Leaf(s) if literal(s) => literal_value(s),
        _ => None,
    }
}

/// Is a written node the literal one (whatever its spelling: `1`, `1.0`)?
fn is_unit(n: &Written) -> bool {
    leaf_value(n).is_some_and(|v| v.is_one())
}

/// Is a written node the literal minus one (`-1`, `-1.0`, `(-1)`, or the sign on the literal one,
/// `neg(1)`, which costs what `-1` costs)?
fn is_minus_one(n: &Written) -> bool {
    match n {
        Written::Node("neg", k) => is_unit(&k[0]),
        _ => leaf_value(n).is_some_and(|v| v.abs().is_one() && v.is_negative()),
    }
}

/// The exact value of a literal leaf or of the sign on one (`neg(c)` is the number `-c`, as `-c`
/// is), `None` for any other node.
fn signed_value(n: &Written) -> Option<Rat> {
    match n {
        Written::Node("neg", k) => {
            let v = leaf_value(&k[0])?;
            let _exact = crate::ac::rat::number_domain(false);
            v.checked_neg()
        }
        _ => leaf_value(n),
    }
}

/// The exact value of a node written as ONE NUMBER, or `None`: a literal (`c`, `-c`, `neg(c)`),
/// and the reciprocal of one (`1/c`, `inv(c)`: the literal 1/c, its 1 the notation's) and its
/// sign (`-1/c`, M1). A written minus on the divisor of `1/c` or `-1/c` is no part of one number:
/// `-1/-3` is the sign of the reciprocal of `-3`, two signs written. Any other quotient of numbers
/// is a division, not one number (owner's ruling 2026-11-27: `3/4` is `3` and the literal `1/4`,
/// as `4*x0/3` is `4`, `x0` and the literal `1/3`). A literal beyond the exact domain has no value
/// here (it is priced by its spelling). The value is exact: it is read and computed in the exact
/// number domain.
fn number(n: &Written) -> Option<Rat> {
    let _exact = crate::ac::rat::number_domain(false);
    match n {
        Written::Leaf(_) | Written::Node("neg", _) => signed_value(n),
        Written::Node("inv", k) => signed_value(&k[0])?.checked_inv(),
        Written::Node("/", k) => {
            let (p, q) = (signed_value(&k[0])?, signed_value(&k[1])?);
            if p.abs().is_one() && !q.is_negative() {
                p.checked_mul(&q.checked_inv()?)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The exact reciprocal of [`number`] (`None` for zero too), in the exact number domain.
fn number_reciprocal(n: &Written) -> Option<Rat> {
    let v = number(n)?;
    let _exact = crate::ac::rat::number_domain(false);
    v.checked_inv()
}

/// A leaf as [`complexity`] prices it: `np.pi`, `np.e`, the infinities and NaN, `<constant>`
/// and a named constant `<constant_NAME>` at their symbol-table entries, a literal at the
/// literal code, any other symbol (a variable) at `mu_leaf`.
fn leaf_price(s: &str) -> u64 {
    match s {
        "np.pi" => mu_pi(),
        "np.e" => mu_e(),
        "float(\"inf\")" | "float(\"-inf\")" | "float(\"nan\")" => mu_infinity(),
        "<constant>" => mu_free(),
        _ if crate::tokens::is_named_constant(s) => mu_free(),
        _ if literal(s) => literal_price(s),
        _ => mu_leaf(),
    }
}

/// Is a written node the positive infinity (`inf`, read as `float("inf")`)? With a sign on it,
/// or as a subtrahend, it is the symbol `float("-inf")` (S1), as [`complexity`] prices it.
fn is_infinity(n: &Written) -> bool {
    matches!(n, Written::Leaf("float(\"inf\")"))
}

/// The addends of a run of `+` and `-`: a left operand and a `+` operand continue the run, a
/// subtrahend stays whole and carries a sign.
fn addends<'a>(n: &'a Written<'a>, out: &mut Vec<(bool, &'a Written<'a>)>) {
    match n {
        Written::Node("+", k) => {
            addends(&k[0], out);
            addends(&k[1], out);
        }
        Written::Node("-", k) => {
            addends(&k[0], out);
            out.push((true, &k[1]));
        }
        _ => out.push((false, n)),
    }
}

/// One member of a run of `*` and `/`: a factor, a divisor (stays whole), or the sign a
/// numerator `-1` writes (M1).
enum Factor<'a> {
    Times(&'a Written<'a>),
    Divisor(&'a Written<'a>),
    Sign,
}

/// The factors of a run of `*` and `/`: a left operand and a `*` operand continue the run, a
/// divisor stays whole; `1/y` (and `inv(y)`, the same) is the reciprocal of `y`, its 1 the
/// notation's, and `-1/y` is the SIGN of the reciprocal, -(1/y) (M1). One number (`1/3`, `-1/3`:
/// [`number`]) is one factor; `3/4` is the factor `3` and the divisor `4`.
fn factors<'a>(n: &'a Written<'a>, out: &mut Vec<Factor<'a>>) {
    match n {
        Written::Node("*", k) => {
            factors(&k[0], out);
            factors(&k[1], out);
        }
        Written::Node("/" | "inv", _) if number(n).is_some() => out.push(Factor::Times(n)),
        Written::Node("/", k) => {
            if is_minus_one(&k[0]) {
                out.push(Factor::Sign);
            } else if !is_unit(&k[0]) {
                factors(&k[0], out);
            }
            out.push(Factor::Divisor(&k[1]));
        }
        Written::Node("inv", k) => out.push(Factor::Divisor(&k[0])),
        _ => out.push(Factor::Times(n)),
    }
}

/// The price of the exponent `-k` when a written power `y^k` divides (K): the division's sign
/// merges into a literal `k` (`x/y^2` holds y^-2, `x/y^-2` holds y^2, and `x/y^0` holds y^0, no
/// sign written or held); `None` when `k` is not one number ([`number`]: `neg(c)` is one, a
/// fraction `p/q` is a division), whose power is then a reciprocal.
fn merged_exponent(k: &Written) -> Option<u64> {
    if let Written::Leaf(s) = k {
        if literal(s) {
            // by its spelling, which prices a literal beyond the exact domain too
            return Some(match unwrapped(s).strip_prefix('-') {
                Some(magnitude) => literal_price(magnitude),
                None if literal_value(s).is_some_and(|v| v.is_zero()) => literal_price(s),
                None => MU_SIGN.saturating_add(literal_price(s)),
            });
        }
    }
    let negated = {
        let v = number(k)?;
        let _exact = crate::ac::rat::number_domain(false);
        v.checked_neg()?
    };
    Some(mu_rat(&negated))
}

/// A divisor's price: one number `c` is the literal `1/c` at the literal code (as [`complexity`]
/// prices `x/3`: the literal 1/3); a written power `y^k` is `y^-k` when `k` is a literal (K: the
/// division's sign merges into it) and the reciprocal of the power otherwise; any other divisor
/// is a reciprocal, the `Pow` head over it (`x^-1` is division and its exponent is free).
fn divisor_price(d: &Written, ops: &Operators) -> u64 {
    if let Some(r) = number_reciprocal(d) {
        return mu_rat(&r);
    }
    match d {
        Written::Node("pow", k) => match merged_exponent(&k[1]) {
            Some(exponent) => mu_pow()
                .saturating_add(price(&k[0], ops))
                .saturating_add(exponent),
            None => mu_pow().saturating_add(price(d, ops)),
        },
        // a reciprocal, also of the literal zero and of a literal beyond the exact domain
        _ => mu_pow().saturating_add(price(d, ops)),
    }
}

/// The written price of a node (see [`written_complexity`]).
fn price(n: &Written, ops: &Operators) -> u64 {
    match n {
        Written::Leaf(s) => leaf_price(s),
        Written::Node("+" | "-", _) => {
            let mut run = Vec::new();
            addends(n, &mut run);
            run.iter().fold(mu_add(), |acc, (negated, m)| {
                acc.saturating_add(match (*negated, is_infinity(m)) {
                    (true, true) => mu_infinity(),
                    (true, false) => MU_SIGN.saturating_add(price(m, ops)),
                    (false, _) => price(m, ops),
                })
            })
        }
        Written::Node("*" | "/" | "inv", _) => {
            if let Some(v) = number(n) {
                return mu_rat(&v);
            }
            let mut run = Vec::new();
            factors(n, &mut run);
            let members = run.iter().filter(|f| !matches!(f, Factor::Sign)).count();
            let head = if members > 1 { mu_mul() } else { 0 };
            run.iter().fold(head, |acc, f| {
                acc.saturating_add(match f {
                    Factor::Times(m) => price(m, ops),
                    Factor::Divisor(d) => divisor_price(d, ops),
                    Factor::Sign => MU_SIGN,
                })
            })
        }
        Written::Node("neg", k) if is_infinity(&k[0]) => mu_infinity(),
        Written::Node("neg", k) => MU_SIGN.saturating_add(price(&k[0], ops)),
        Written::Node("pow", k) => mu_pow()
            .saturating_add(price(&k[0], ops))
            .saturating_add(price(&k[1], ops)),
        Written::Node(head @ ("max" | "min"), _) => {
            // L: the reader writes the call max(a, b, c) as max(a, max(b, c)), and the printer
            // writes that chain back as the one call: a right operand of the same head
            // continues the call, which pays its head once
            let mut total = mu_head(head);
            let mut rest = n;
            while let Written::Node(h, k) = rest {
                if h != head || k.len() != 2 {
                    break;
                }
                total = total.saturating_add(price(&k[0], ops));
                rest = &k[1];
            }
            total.saturating_add(price(rest, ops))
        }
        Written::Node(head, k) => {
            let head_price = if ops.defined.contains_key(*head) {
                mu_defined_head(head)
            } else {
                mu_head(head)
            };
            k.iter()
                .fold(head_price, |acc, m| acc.saturating_add(price(m, ops)))
        }
    }
}

/// THE WRITTEN PRICE of an expression, in milli-bits: [`complexity`]'s code book applied to the
/// expression EXACTLY AS WRITTEN. Nothing is read into a canonical state and nothing is
/// simplified: every operator, function, variable, sign and number the expression writes is
/// charged, and the only identifications are the ones infix notation forces, each read as the
/// code book prices it.
///
/// * A run of `+` and `-` is ONE sum, a run of `*` and `/` ONE product, whatever the
///   bracketing: a left operand and a `+` (`*`) operand continue the run; a bracketed
///   subtrahend or divisor stays whole. Operands are unordered. A sum pays `mu_add` once, a
///   product `mu_mul` once (a lone reciprocal is no product). A call of `max` or `min` with
///   more than two arguments, `max(a, b, c)`, is ONE call (L): the reader nests it to the
///   right, `max(a, max(b, c))`, and a right operand of the same head continues the call.
/// * `-y` and `neg(y)` are a SIGN: one bit (`MU_SIGN`) on `y`, as is a subtrahend. A written
///   `-inf` and a subtrahend `inf` are the symbol `float("-inf")` (S1).
/// * `1/y` is the RECIPROCAL of `y`, its 1 the notation's, and `inv(y)` is the same: a divisor.
///   `-1/y` is the sign of the reciprocal, -(1/y) (M1): a numerator `-1` is a sign. A divisor
///   pays `mu_pow` over itself (`x^-1` is division, its exponent free), except that dividing
///   by a written power `y^k` with a LITERAL exponent is `y^-k`, the sign merging into the
///   literal (K: `x/y^2` holds y^-2, `x/y^-2` holds y^2, `x/y^0` holds y^0 and pays no sign,
///   and `x/y^z` divides by the power, `mu_pow` over `y^z`), and dividing by one number `c` is
///   the literal `1/c` (`x/3` holds the literal 1/3, as [`complexity`] prices it). A literal
///   with a sign, `-c` or `neg(c)`, is one number in each of these places.
/// * A NUMBER pays the literal code on the exact value written, with every digit ([`mu_rat`];
///   its print's price beyond the 1,100-bit cap): a long literal is not rounded to its double.
///   A written minus on a literal is its sign bit. A FRACTION written `p/q` costs its
///   division, `p` and the literal `1/q`, as any number divided by a number does (owner's
///   ruling 2026-11-27: `3/4` 10.322 bits, `3/4*x0` and `4*x0/3` both 16.322; a decimal costs
///   its value, `0.75` 5.322), whether it is written as three tokens (`/ 3 4`) or as the one
///   token `3/4`; only `1/q` and `-1/q` are one literal, the reciprocal's 1 the notation's, and
///   a minus written on their `q` is a sign of its own (`-1/-3` pays two).
/// * `np.pi`, `np.e`, the infinities, NaN, `<constant>` and named constants (`<constant_NAME>`)
///   pay their entries, as [`complexity`] prices them; every other symbol is a variable
///   (`mu_leaf`).
/// * A function pays its head's entry (the head table, by name) and its operands; `pow` pays
///   `mu_pow`, base and exponent; an alias in infix text pays its target's price (the reader
///   reads it as its target), and a prefix token list names the operators themselves: an alias
///   token (`ln`, `div`, `**`) is refused there, as [`complexity`] refuses it. A DEFINED FUNCTION
///   (`crate::defined`) pays its own head's entry ([`mu_defined_head`]: `sqrt`, `cot`, `sec` and
///   `csc` at the elementary tier, `acot`, `asec` and `acsc` at the transcendental). A head that
///   is neither an operator nor a defined function of the engine is refused, never guessed.
///
/// The written price of a canonical form is not promised to equal [`complexity`]: the
/// canonical serializations write some states otherwise than they are held (a product of
/// reciprocals as the reciprocal of a product, a distributed negation, a coefficient p/q as
/// `p*x/q`); the measured agreement is in `SimpliPyEngine.written_complexity`'s docstring.
pub fn written_complexity(tokens: &[String], ops: &Operators) -> Result<u64, String> {
    let tree = read(tokens, ops)?;
    Ok(price(&tree, ops))
}

#[cfg(test)]
mod tests {
    fn engine() -> crate::Engine {
        let cfg = include_str!("../../../tests/fixtures/vocab32/config.yaml");
        crate::Engine::from_strs(cfg, "[]").unwrap()
    }

    fn written(e: &crate::Engine, infix: &str) -> u64 {
        e.written_complexity(&e.parse(infix, true, false, true).unwrap())
            .unwrap()
    }

    fn tokens(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    /// The brief's values (milli-bits), and the identifications of the runs.
    #[test]
    fn the_written_price_charges_what_is_written() {
        let e = engine();
        for (infix, price) in [
            ("x0 - x0/1 + 2", 23_585),
            ("x0 + 0", 11_000),
            ("x0 + x0", 15_000),
            ("x0*x0", 15_000),
            ("exp(1)", 8_000),
            ("-(x0 - x1)", 17_000),
            ("sqrt(x0)", 12_000),
            ("cot(x0)", 12_000),
            ("acot(x0)", 14_000),
            ("sec(x0)", 12_000),
            ("csc(x0)", 12_000),
            ("asec(x0)", 14_000),
            ("acsc(x0)", 14_000),
            ("x0**0.5", 12_585),
            ("rootn(x0, 2)", 14_585),
            ("6.674e-11*x0*x1/x2**2", 45_197),
            ("x0*x1/(4*np.pi*x2*x3**2)", 45_907),
            ("x0 + (x1 + x2)", 21_000),
            ("x0/x1/x2", 27_000),
            ("x0/(x1*x2)", 27_000),
            ("1/(x1*x2)", 18_000),
            ("x0 - (x1 + x2)", 25_000),
            ("Abs(x0) + ln(x1)", 27_000),
        ] {
            assert_eq!(written(&e, infix), price, "{infix}");
        }
    }

    /// The overseer's decisions (2026-11-27), each case with its price and the price it had
    /// before: M1 `-1/y` is the sign of the reciprocal, S1 a written `-inf` is the symbol, K a
    /// divisor `y^k` is `y^-k` only for a literal `k`, L a variadic `max`/`min` is one call; and
    /// the owner's ruling (2026-11-27) that a fraction `p/q` costs its division.
    #[test]
    fn the_decisions_m1_s1_k_l_f() {
        let e = engine();
        for (infix, price, before) in [
            ("-1/x0", 10_000, 15_000),
            ("x0*(-1/x1)", 19_000, 21_000),
            ("-1/x0/x1", 22_000, 24_000),
            ("-1/3", 5_000, 10_000),
            ("-inf", 8_000, 9_000),
            ("x0 - inf", 17_000, 18_000),
            ("x0 + -inf", 17_000, 18_000),
            ("2*(-inf)", 13_585, 14_585),
            ("x0/x1**x2", 27_000, 25_000),
            ("x0/exp(x1)**x2", 33_000, 31_000),
            ("x0/x1**(-2)", 20_585, 22_585),
            ("x0/x1**2", 21_585, 21_585),
            ("max(x0, x1, x2)", 26_000, 34_000),
            ("min(x0, x1, x2, x3)", 32_000, 48_000),
            ("max(max(x0, x1), x2)", 34_000, 34_000),
        ] {
            assert_eq!(written(&e, infix), price, "{infix} (was {before})");
        }
        // a fraction costs its division (owner, 2026-11-27): (expression, price, the price before
        // the decisions); `1/q` stays the literal 1/q and `0.75` costs its value
        for (infix, price, before) in [
            ("3/4", 10_322, 10_322),
            ("x0**(3/2)", 18_585, 18_585),
            ("x0**(-3/2)", 19_585, 19_585),
            ("x0/x1**(3/2)", 30_585, 28_585),
            ("x0/(3/4)", 22_322, 22_322),
            ("6/4", 11_129, 11_129),
            ("3/4*x0", 16_322, 16_322),
            ("4*x0/3", 16_322, 16_322),
            ("2*x0*3", 14_585, 14_585),
            ("1/4", 4_322, 4_322),
            ("1/2", 3_585, 3_585),
            ("0.75", 5_322, 5_322),
        ] {
            assert_eq!(written(&e, infix), price, "{infix} (was {before})");
        }
        // the numerator -1 in any spelling, and the infinities as prefix tokens
        assert_eq!(
            e.written_complexity(&tokens("/ neg 1 x0")),
            e.written_complexity(&tokens("/ -1 x0"))
        );
        assert_eq!(
            e.written_complexity(&tokens("neg float(\"inf\")")),
            e.written_complexity(&tokens("float(\"-inf\")"))
        );
        assert_eq!(
            e.written_complexity(&tokens("- x0 float(\"-inf\")")),
            Ok(18_000)
        );
    }

    #[test]
    fn an_undeclared_head_or_a_tagged_sequence_is_refused() {
        let e = engine();
        assert!(e.written_complexity(&tokens("zeta x0")).is_err());
        assert!(e.written_complexity(&tokens("<add> x0 x1 </add>")).is_err());
        assert!(e.written_complexity(&tokens("+ x0")).is_err());
        assert!(e.written_complexity(&[]).is_err());
    }

    /// The written price's consistency (pre-audit review W-1..W-5): each spelling costs what the
    /// same writing costs in its other spellings; (tokens, price, the spelling it equals).
    #[test]
    fn one_writing_one_price_w1_to_w5() {
        let e = engine();
        let w = |s: &str| e.written_complexity(&tokens(s));
        for (spelling, price, other) in [
            // W-1: a one-token fraction costs its division, also in the exponent merge
            ("3/4", 10_322, "/ 3 4"),
            ("(-3/4)", 11_322, "/ -3 4"),
            ("/ x0 pow x1 3/2", 30_585, "/ x0 pow x1 / 3 2"),
            ("1/4", 4_322, "inv 4"),
            ("-1/4", 5_322, "/ -1 4"),
            // W-2: a sign on a literal is one number, as a divisor, a merged exponent, a 1/q part
            ("/ x0 neg 3", 14_000, "/ x0 -3"),
            ("/ x0 pow x1 neg 2", 20_585, "/ x0 pow x1 -2"),
            ("/ 1 neg 3", 5_000, "/ 1 -3"),
            ("inv neg 3", 5_000, "inv -3"),
            // W-3: -1/-3 writes two minus signs: the sign of the reciprocal of -3
            ("/ -1 -3", 6_000, "neg / 1 -3"),
            // W-4: x/y**0 writes no sign
            ("/ x0 pow x1 0", 20_000, "* x0 pow x1 0"),
        ] {
            assert_eq!(w(spelling), Ok(price), "{spelling}");
            assert_eq!(w(other), Ok(price), "{other}");
        }
        assert_eq!(written(&e, "-1/-3"), 6_000);
        assert_eq!(written(&e, "-1/3"), 5_000);
        // W-5: a prefix alias token is refused, as complexity refuses it; infix reads its target
        for alias in ["ln x0", "div x0 x1", "** x0 2", "Max x0 x1", "power x0 2"] {
            let error = w(alias).unwrap_err();
            assert!(error.contains("is an alias of"), "{alias}: {error}");
        }
        assert_eq!(written(&e, "ln(x0)"), written(&e, "log(x0)"));
        assert_eq!(written(&e, "x0**2"), w("pow x0 2").unwrap());
    }
}
