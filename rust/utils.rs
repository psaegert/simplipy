//! Pure string/number helpers mirroring `simplipy/utils.py`: is_numeric_string,
//! numbers_to_constant, remove_pow1, factorize_to_at_most.

/// Port of `is_numeric_string`, the predicate `mask_elementary_literals`
/// uses. It is a string-munging check, NOT `float()`:
/// `s.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e+', '', 1)
/// .replace('e', '', 1).isdigit()`.
/// Order matters (`.` then `e-` then `e+` then `e`); each `replace(..., 1)` is
/// first-occurrence-only; `lstrip('-')` strips ALL leading `-`; `isdigit()` is false on the
/// empty string. The `e+` arm closes the predicate under the engine's own emissions:
/// `py_float_repr` spells big magnitudes `1e+16` exactly as Python `repr` does (H-007).
/// The munge OVER-approximates (`--5`, `1e-5.5` pass); the exact readers
/// (`Rat::parse_decimal`, `leaf_value`) decide evaluability, and the excess is refused by
/// `is_valid`. (Distinct from `operand_key`'s `float()`-based numeric test in `sort.rs`.)
pub fn is_numeric_string(s: &str) -> bool {
    let t = s.trim_start_matches('-');
    let t = replace_first(t, ".");
    let t = replace_first(&t, "e-");
    let t = replace_first(&t, "e+");
    let t = replace_first(&t, "e");
    // `str.isdigit()`: non-empty and every char a digit. The grammar's tokens are ASCII, so
    // `is_ascii_digit` matches Python `isdigit()` for every token that can occur.
    if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    // The AC core's exact-fraction literal `p/q` (`1/3`, `-7/4`): integer '/' integer,
    // one slash. Without this arm, `mask` left the fraction unmasked and the evaluators
    // refused it.
    if let Some((p, q)) = s.split_once('/') {
        let p = p.strip_prefix('-').unwrap_or(p);
        return !p.is_empty()
            && !q.is_empty()
            && p.chars().all(|c| c.is_ascii_digit())
            && q.chars().all(|c| c.is_ascii_digit());
    }
    false
}

/// `s.replace(pat, '', 1)`: remove the FIRST occurrence of `pat`, else return `s` unchanged.
fn replace_first(s: &str, pat: &str) -> String {
    match s.find(pat) {
        Some(idx) => format!("{}{}", &s[..idx], &s[idx + pat.len()..]),
        None => s.to_string(),
    }
}

/// H-007: a spelling that a STANDARD numeric reader interprets but the engine's symbolic
/// core does not. Such a token has two contradictory readings -- the AC canon would treat
/// it as a free symbol and apply symbol algebra (`inf - inf -> 0`) while Python `float()`,
/// Rust `f64::from_str`, and the engine's own `leaf_value` read it as a value
/// (`inf - inf = nan`) -- so the boundary refuses it outright (`ensure_tokens_are_tokens`).
/// Three families:
///   1. textual non-finites, any case, optional single sign: `inf`, `-Infinity`, `NAN`
///      (Python `float()` and Rust `f64` both accept them; the canonical spellings are the
///      `float("inf")`/`float("-inf")`/`float("nan")` tokens);
///   2. underscore digit groupings: `1_000`, `1_000.5`, `1e1_0` (Python `float()` reads
///      them; `Rat::parse_decimal` does not). The `_` must sit BETWEEN two digits, exactly
///      Python's placement rule -- `_0` (rule placeholders) and `x_0` stay free symbols;
///   3. base-prefixed integer literals: `0x10`/`0o17`/`0b101` any case, optional sign,
///      optional underscore grouping (Python source syntax: a realized-infix `eval` would
///      read 16 where the engine sees a symbol).
///
/// Everything else is either a canonical numeric literal (`Rat::parse_decimal` forms, the
/// exact fraction `p/q`, the special spellings `np.pi`/`np.e`/`float("...")` and their
/// parenthesized forms) or a genuine free symbol, which no numeric reader misreads.
pub fn reserved_numeric_spelling(t: &str) -> bool {
    let core = t.strip_prefix(['+', '-']).unwrap_or(t);
    // Family 1: textual non-finites.
    if core.eq_ignore_ascii_case("inf")
        || core.eq_ignore_ascii_case("infinity")
        || core.eq_ignore_ascii_case("nan")
    {
        return true;
    }
    // Family 3: base-prefixed integer literals. Underscore grouping is allowed loosely
    // (malformed grouping like `0x__10` is still reserved: fail closed -- a "symbol" of
    // that shape is a numeric spelling to every human and most readers).
    let lower = core.to_ascii_lowercase();
    type RadixCheck = fn(char) -> bool;
    let families: [(&str, RadixCheck); 3] = [
        ("0x", |c| c.is_ascii_hexdigit()),
        ("0o", |c| ('0'..='7').contains(&c)),
        ("0b", |c| c == '0' || c == '1'),
    ];
    for (prefix, digits_ok) in families {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let digits: String = rest.chars().filter(|&c| c != '_').collect();
            if !digits.is_empty() && digits.chars().all(digits_ok) {
                return true;
            }
        }
    }
    // Family 2: underscore digit groupings. Enforce Python's placement rule (every `_`
    // strictly between two ASCII digits), then require the cleaned spelling to be a
    // float literal (Rust `f64` parse; non-finite words were handled above, so this is
    // exactly the sign/digits/`.`/exponent grammar).
    if t.contains('_') {
        let bytes = t.as_bytes();
        let placed = t.char_indices().filter(|&(_, c)| c == '_').all(|(i, _)| {
            i > 0
                && i + 1 < bytes.len()
                && bytes[i - 1].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit()
        });
        if placed {
            let cleaned: String = t.chars().filter(|&c| c != '_').collect();
            if cleaned.parse::<f64>().is_ok() {
                return true;
            }
        }
    }
    false
}

/// Port of `numbers_to_constant`: replace every token for which Python's
/// `float(token)` SUCCEEDS with `<constant>`. NOTE this uses `float()` (try/except), NOT
/// `is_numeric_string` -- so it is a DIFFERENT predicate from [`mask_elementary_literals`]
/// (e.g. `float('1e3')` succeeds, `float('inf')`/`float('nan')` succeed). `parse` calls this when
/// `mask_numbers=True`. Rust `f64::from_str` matches Python `float()` on every token the
/// `infix_to_prefix` tokenizer can emit (its number regex has no `_`/whitespace, and Rust accepts
/// `inf`/`nan` case-insensitively exactly as Python does); the only divergences (underscore digit
/// grouping, surrounding whitespace) are unreachable on tokenizer output.
pub fn numbers_to_constant(prefix_expression: &[String]) -> Vec<String> {
    prefix_expression
        .iter()
        .map(|t| {
            if t.parse::<f64>().is_ok() {
                "<constant>".to_string()
            } else {
                t.clone()
            }
        })
        .collect()
}

/// Port of `remove_pow1`: drop every `pow1` token (raising-to-1 identity)
/// and rewrite `pow_1` (raising-to-(-1)) as `inv`; all other tokens pass through. The final cleanup
/// step of `parse`.
pub fn remove_pow1(prefix_expression: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(prefix_expression.len());
    for token in prefix_expression {
        if token == "pow1" {
            continue;
        }
        if token == "pow_1" {
            out.push("inv".to_string());
            continue;
        }
        out.push(token.clone());
    }
    out
}

/// Port of `utils.factorize_to_at_most`. Decomposes `p` into factors
/// each `<= max_factor` whose product is `p`, in discovery order (NOT sorted). Returns `Err(())`
/// where Python raises `ValueError` -- the surviving callers (the legacy power-chain
/// conversion paths in `convert.rs`; the historical one, `cancel_terms`, is deleted) branch
/// on the raise to a `*`/`pow`-coefficient fallback, so the raise conditions are CONTROL
/// FLOW and must match exactly:
/// `p < 1`, `max_factor < 2`, a `divisor > max_factor` mid-loop, a prime `remaining > max_factor`,
/// or `processed_factors > max_iter`. `max_iter` defaults to 1000 at the Python call sites.
// The final `flush_current!()` resets `current_factor` to 1 that is then never read -- a faithful
// port of the Python nonlocal closure's reset, structurally part of the macro, not a bug.
#[allow(unused_assignments)]
pub fn factorize_to_at_most(p: i128, max_factor: i64, max_iter: i64) -> Result<Vec<i64>, ()> {
    // `p` is i128 (not i64): a `pow<N>` exponent / chain product can exceed i64 (e.g. `x ** 2^63`),
    // where Python's arbitrary-precision `prod`/`factorize` still decomposes faithfully. The emitted
    // factors are all <= max_factor (a small int), so the output stays Vec<i64>. (Beyond i128 is the
    // documented out-of-domain boundary, unreachable on any real expression.)
    let max_factor_i128 = max_factor as i128;
    if p < 1 {
        return Err(());
    }
    if max_factor < 2 {
        return Err(());
    }
    if p == 1 {
        return Ok(Vec::new());
    }

    let mut remaining = p;
    let mut factors: Vec<i64> = Vec::new();
    let mut current_factor: i128 = 1;
    let mut processed_factors: i64 = 0;

    // `flush_current()` (the Python nonlocal closure): emit the accumulated factor if > 1, reset.
    macro_rules! flush_current {
        () => {
            if current_factor > 1 {
                factors.push(current_factor as i64); // current_factor <= max_factor -> fits i64
                current_factor = 1;
            }
        };
    }

    let mut divisor: i128 = 2;
    while divisor * divisor <= remaining {
        while remaining % divisor == 0 {
            processed_factors += 1;
            if processed_factors > max_iter {
                return Err(());
            }
            if divisor > max_factor_i128 {
                return Err(());
            }
            if current_factor * divisor <= max_factor_i128 {
                current_factor *= divisor;
            } else {
                flush_current!();
                current_factor = divisor;
            }
            remaining /= divisor;
        }
        divisor = if divisor == 2 { 3 } else { divisor + 2 };
    }

    if remaining > 1 {
        // remaining is prime at this point
        if remaining > max_factor_i128 {
            return Err(());
        }
        if current_factor * remaining <= max_factor_i128 {
            current_factor *= remaining;
        } else {
            flush_current!();
            current_factor = remaining;
        }
    }

    flush_current!();

    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::{factorize_to_at_most, is_numeric_string, reserved_numeric_spelling};

    #[test]
    fn reserved_spellings_h007() {
        // Family 1: textual non-finites, any case, optional sign.
        for t in [
            "inf", "Inf", "INF", "infinity", "Infinity", "+inf", "-inf", "nan", "NaN", "+nan",
        ] {
            assert!(reserved_numeric_spelling(t), "{t}");
        }
        // Family 2: underscore digit groupings (Python float() placement rule).
        for t in ["1_000", "1_000.5", "1e1_0", "1_0"] {
            assert!(reserved_numeric_spelling(t), "{t}");
        }
        // Family 3: base-prefixed integer literals.
        for t in ["0x10", "0X10", "0o17", "0b101", "+0x10", "-0x10", "0x_10"] {
            assert!(reserved_numeric_spelling(t), "{t}");
        }
        // Canonical literals and genuine symbols are NOT reserved.
        for t in [
            "5",
            "+5",
            "-5",
            ".5",
            "5.",
            "1e-05",
            "1e+16",
            "1/3",
            "-7/4",
            "np.pi",
            "float(\"inf\")",
            "(-1)",
            "x0",
            "foo",
            "_0",
            "x_0",
            "inf2",
            "nanx",
            "in",
            "_",
            "0x",
        ] {
            assert!(!reserved_numeric_spelling(t), "{t}");
        }
    }

    #[test]
    fn numeric_string_covers_engine_emitter() {
        // The 'e+' munge arm (H-007): py_float_repr spells 1e16 as "1e+16" (Python repr
        // parity); the masking predicate must recognize the engine's own emissions.
        assert!(is_numeric_string("1e+16"));
        assert!(is_numeric_string("1e-05"));
        assert!(is_numeric_string("1e5"));
        assert!(!is_numeric_string("1_000"));
        assert!(!is_numeric_string("inf"));
        assert!(!is_numeric_string("+5")); // the munge does not strip '+'; parse_decimal reads it
    }

    #[test]
    fn factorize_matches_python_examples() {
        assert_eq!(factorize_to_at_most(100, 10, 1000), Ok(vec![4, 5, 5]));
        assert_eq!(factorize_to_at_most(18, 5, 1000), Ok(vec![2, 3, 3]));
        assert_eq!(factorize_to_at_most(1, 5, 1000), Ok(vec![]));
        // Historical cancel_terms-era vectors (max_factor = max_power = 5), kept as
        // parity coverage for the surviving convert.rs callers.
        assert_eq!(factorize_to_at_most(2, 5, 1000), Ok(vec![2]));
        assert_eq!(factorize_to_at_most(3, 5, 1000), Ok(vec![3]));
        assert_eq!(factorize_to_at_most(6, 5, 1000), Ok(vec![2, 3]));
    }

    #[test]
    fn factorize_raises_match_python() {
        // p < 1 / max_factor < 2 guards.
        assert_eq!(factorize_to_at_most(0, 5, 1000), Err(()));
        assert_eq!(factorize_to_at_most(5, 1, 1000), Err(()));
        // A prime factor larger than max_factor cannot be decomposed -> ValueError (the fallback trigger).
        assert_eq!(factorize_to_at_most(7, 5, 1000), Err(())); // 7 prime > 5
        assert_eq!(factorize_to_at_most(10, 3, 1000), Err(())); // needs factor 5 > 3
    }
}
