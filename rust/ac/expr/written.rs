//! THE WRITTEN PRICE: [`complexity`]'s code book applied to an expression exactly as written
//! ([`written_complexity`]).
//!
//! [`complexity`] prices the CANONICAL state a mode reads an expression into; the written price
//! reads nothing into any state. It charges every operator, function, variable, sign and number
//! the expression writes, each at its entry of the one code book (`mu_leaf`, `mu_add`, `mu_mul`,
//! `mu_pow`, the head table, `MU_SIGN`, the literal code), and makes only the identifications
//! the infix notation forces (owner's ruling relayed 2026-11-26: srbf measures ground truths and
//! predictions "exactly as written, without any modifications that either benefit the benchmark
//! or the model").

use super::*;
use crate::operators::Operators;

/// A node of the expression as written: a leaf token, or a head (its canonical name, an alias
/// resolved) over its operands.
enum Written<'a> {
    Leaf(&'a str),
    Node(&'a str, Vec<Written<'a>>),
}

/// The canonical name of a written head: an alias is its target (`ln` is `log`, `Abs` is
/// `abs`, `div` is `/`, `power` is `pow`) and `**` is `pow`.
fn canonical<'a>(token: &'a str, ops: &'a Operators) -> &'a str {
    if token == "**" {
        return "pow";
    }
    ops.operator_aliases
        .get(token)
        .map(String::as_str)
        .unwrap_or(token)
}

/// Read explicit prefix tokens into the tree they write. A head is a declared or core operator
/// or a defined function (`Operators::syntax_arity`); every other token is a leaf.
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
        let head = canonical(token, ops);
        let arity = if token == "**" {
            Some(2)
        } else {
            ops.syntax_arity(head)
        };
        match arity {
            None => Ok((Written::Leaf(token), at + 1)),
            Some(n) => {
                let mut kids = Vec::with_capacity(usize::from(n));
                let mut next = at + 1;
                for _ in 0..n {
                    let (kid, after) = one(tokens, next, ops)?;
                    kids.push(kid);
                    next = after;
                }
                Ok((Written::Node(head, kids), next))
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

/// One factor of a run of `*` and `/`: a factor, or a divisor (stays whole).
enum Factor<'a> {
    Times(&'a Written<'a>),
    Divisor(&'a Written<'a>),
}

/// The factors of a run of `*` and `/`: a left operand and a `*` operand continue the run, a
/// divisor stays whole; `1/y` (and `inv(y)`, the same) is the reciprocal of `y`, its 1 the
/// notation's.
fn factors<'a>(n: &'a Written<'a>, out: &mut Vec<Factor<'a>>) {
    match n {
        Written::Node("*", k) => {
            factors(&k[0], out);
            factors(&k[1], out);
        }
        Written::Node("/", k) => {
            if !is_unit(&k[0]) {
                factors(&k[0], out);
            }
            out.push(Factor::Divisor(&k[1]));
        }
        Written::Node("inv", k) => out.push(Factor::Divisor(&k[0])),
        _ => out.push(Factor::Times(n)),
    }
}

/// Is a written node the literal one (whatever its spelling: `1`, `1.0`)?
fn is_unit(n: &Written) -> bool {
    matches!(n, Written::Leaf(s) if literal(s) && literal_value(s).is_some_and(|v| v.is_one()))
}

/// A divisor's price: a literal `c` is the literal `1/c` at the literal code (as [`complexity`]
/// prices `x/3`: the literal 1/3); a written power `y^k` is `y^-k`, the division the exponent's
/// sign; any other divisor is a reciprocal, the `Pow` head over it (`x^-1` is division and its
/// exponent is free).
fn divisor_price(d: &Written, ops: &Operators) -> u64 {
    match d {
        Written::Leaf(s) if literal(s) => match literal_value(s).and_then(|v| v.checked_inv()) {
            Some(r) => mu_rat(&r),
            // zero, or beyond the exact domain: a reciprocal of the literal as written
            None => mu_pow().saturating_add(literal_price(s)),
        },
        Written::Node("pow", k) => mu_pow()
            .saturating_add(price(&k[0], ops))
            .saturating_add(MU_SIGN)
            .saturating_add(price(&k[1], ops)),
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
                acc.saturating_add(if *negated { MU_SIGN } else { 0 })
                    .saturating_add(price(m, ops))
            })
        }
        Written::Node("*" | "/" | "inv", _) => {
            let mut run = Vec::new();
            factors(n, &mut run);
            let head = if run.len() > 1 { mu_mul() } else { 0 };
            run.iter().fold(head, |acc, f| {
                acc.saturating_add(match f {
                    Factor::Times(m) => price(m, ops),
                    Factor::Divisor(d) => divisor_price(d, ops),
                })
            })
        }
        Written::Node("neg", k) => MU_SIGN.saturating_add(price(&k[0], ops)),
        Written::Node("pow", k) => mu_pow()
            .saturating_add(price(&k[0], ops))
            .saturating_add(price(&k[1], ops)),
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
/// charged, and the only identifications are the ones infix notation forces.
///
/// * A run of `+` and `-` is ONE sum, a run of `*` and `/` ONE product, whatever the
///   bracketing: a left operand and a `+` (`*`) operand continue the run; a bracketed
///   subtrahend or divisor stays whole. Operands are unordered. A sum pays `mu_add` once, a
///   product `mu_mul` once (a lone reciprocal is no product).
/// * `-y` and `neg(y)` are a SIGN: one bit (`MU_SIGN`) on `y`, as is a subtrahend.
/// * `1/y` is the RECIPROCAL of `y`, its 1 the notation's, and `inv(y)` is the same: a divisor.
///   A divisor pays `mu_pow` over itself (`x^-1` is division, its exponent free), except that
///   dividing by a written power `y^k` is `y^-k` (`mu_pow`, `y`, a sign and `k`), and a literal
///   divisor `c` is the literal `1/c` (`x/3` holds the literal 1/3, as [`complexity`] prices it).
/// * A NUMBER pays the literal code on the exact value written, with every digit ([`mu_rat`];
///   its print's price beyond the 1,100-bit cap): a long literal is not rounded to its double.
///   A written minus on a literal is its sign bit.
/// * `np.pi`, `np.e`, the infinities, NaN, `<constant>` and named constants (`<constant_NAME>`)
///   pay their entries, as [`complexity`] prices them; every other symbol is a variable
///   (`mu_leaf`).
/// * A function pays its head's entry (the head table, by name) and its operands; `pow` pays
///   `mu_pow`, base and exponent; an alias pays its target's price. A DEFINED FUNCTION
///   (`crate::defined`) pays its own head's entry ([`mu_defined_head`]: `sqrt`, `cot`, `sec` and
///   `csc` at the elementary tier, `acot`, `asec` and `acsc` at the transcendental). A head that is
///   neither an operator nor a
///   defined function of the engine is refused, never guessed.
///
/// On an expression already in canonical form the written price is meant to equal
/// [`complexity`] in that mode; `tests/` and the measurement in the 0.15 notes record where the
/// serialization of a canonical form and this reading part.
pub fn written_complexity(tokens: &[String], ops: &Operators) -> Result<u64, String> {
    let tree = read(tokens, ops)?;
    Ok(price(&tree, ops))
}

#[cfg(test)]
mod tests {
    fn engine() -> crate::Engine {
        let cfg = include_str!("../../../assets/engines/acj2-4/config.yaml");
        crate::Engine::from_strs(cfg, "[]").unwrap()
    }

    fn written(e: &crate::Engine, infix: &str) -> u64 {
        e.written_complexity(&e.parse(infix, true, false, true).unwrap())
            .unwrap()
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

    #[test]
    fn an_undeclared_head_or_a_tagged_sequence_is_refused() {
        let e = engine();
        let t = |s: &str| -> Vec<String> { s.split_whitespace().map(str::to_string).collect() };
        assert!(e.written_complexity(&t("zeta x0")).is_err());
        assert!(e.written_complexity(&t("<add> x0 x1 </add>")).is_err());
        assert!(e.written_complexity(&t("+ x0")).is_err());
        assert!(e.written_complexity(&[]).is_err());
    }
}
