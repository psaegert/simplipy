//! Pure helpers mirroring `simplipy/utils.py`: match_pattern, apply_mapping, the operand sort key
//! (`engine.operand_key`, engine.py:2564), mask_elementary_literals, is_numeric_string.
//!
//! ## Sort-determinism parity trap (called out in HARNESS_SPEC)
//! Python's `sort_operands` uses `operand_key` -> a tuple key and Timsort STABILITY. The Rust port
//! MUST replicate BOTH the tuple-key comparison semantics AND stable tie-breaking:
//!   - Use a STABLE sort (`slice::sort_by`, not `sort_unstable_by`).
//!   - Reproduce Python's leaf key `(1, float(token))` ordering, including NaN / -0.0 / float
//!     precision edge cases (the `sort_determinism` adversarial corpus category exercises this).
//!   - Equal keys MUST preserve input order to match Timsort.

use rustc_hash::FxHashMap;

use crate::parse::Node;

/// Does this subtree contain a `<constant>` token anywhere? Mirrors
/// `"<constant>" in flatten_nested_list(existing)` in `match_pattern`'s placeholder-rebind guard.
pub fn contains_constant(node: &Node) -> bool {
    match node {
        Node::Leaf(t) => t == "<constant>",
        Node::Op { token, operands } => {
            token == "<constant>" || operands.iter().any(contains_constant)
        }
    }
}

/// Faithful port of `match_pattern` (utils.py:800). Matches a query subtree `tree` against a rule
/// LHS `pattern`; on success `mapping` binds each placeholder name to the matched query subtree
/// (borrowed from `tree`). The bare-string-operand early-return branch in the Python source is DEAD
/// for `prefix_to_tree`-built patterns (operands are always subtrees), so it is intentionally omitted.
pub fn match_pattern<'a>(
    tree: &'a Node,
    pattern: &Node,
    mapping: &mut FxHashMap<String, &'a Node>,
) -> bool {
    match pattern {
        // Elementary (leaf) pattern.
        Node::Leaf(pkey) => {
            if pkey.starts_with('_') {
                // Placeholder. Python keys on startswith('_'), not the ^_\d+$ regex.
                match mapping.get(pkey.as_str()) {
                    None => {
                        mapping.insert(pkey.clone(), tree);
                        true
                    }
                    Some(existing) => {
                        // Cannot rebind a subtree that contains an (independent) <constant>.
                        if contains_constant(existing) {
                            false
                        } else {
                            *existing == tree
                        }
                    }
                }
            } else {
                // Concrete literal leaf: structural equality (Python `tree == pattern`).
                tree == pattern
            }
        }
        // Tree-structured pattern.
        Node::Op {
            token: p_op,
            operands: p_operands,
        } => match tree {
            // Leaf tree vs non-leaf pattern -> mismatch (Python len(tree)==1 & pattern_length!=1).
            Node::Leaf(_) => false,
            Node::Op {
                token: t_op,
                operands: t_operands,
            } => {
                if t_op != p_op {
                    return false;
                }
                for (t_operand, p_operand) in t_operands.iter().zip(p_operands.iter()) {
                    if !match_pattern(t_operand, p_operand, mapping) {
                        return false;
                    }
                }
                true
            }
        },
    }
}

/// Faithful port of `apply_mapping` (utils.py:762): substitute each placeholder leaf (`_N`) in the
/// replacement template with its bound subtree, returning a fresh tree. Concrete leaves and operator
/// structure are copied. A placeholder with no binding panics (mirrors Python's `KeyError`); the
/// wildcard-multiplicity invariant guarantees RHS placeholders are a subset of bound LHS ones.
pub fn apply_mapping(template: &Node, mapping: &FxHashMap<String, &Node>) -> Node {
    match template {
        Node::Leaf(t) => {
            if t.starts_with('_') {
                (*mapping.get(t.as_str()).expect("rhs placeholder is bound")).clone()
            } else {
                template.clone()
            }
        }
        Node::Op { token, operands } => Node::Op {
            token: token.clone(),
            operands: operands.iter().map(|o| apply_mapping(o, mapping)).collect(),
        },
    }
}

// `operand_key` + its ordered key type are ported in `src/sort.rs` (co-located with `sort_operands`,
// their only consumer).

/// Faithful port of `is_numeric_string` (utils.py:552), the predicate `mask_elementary_literals`
/// uses. It is a string-munging check, NOT `float()`:
/// `s.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e', '', 1).isdigit()`.
/// Order matters (`.` then `e-` then `e`); each `replace(..., 1)` is first-occurrence-only;
/// `lstrip('-')` strips ALL leading `-`; `isdigit()` is false on the empty string. (Distinct from
/// `operand_key`'s `float()`-based numeric test in `sort.rs`.)
pub fn is_numeric_string(s: &str) -> bool {
    let t = s.trim_start_matches('-');
    let t = replace_first(t, ".");
    let t = replace_first(&t, "e-");
    let t = replace_first(&t, "e");
    // `str.isdigit()`: non-empty and every char a digit. The grammar's tokens are ASCII, so
    // `is_ascii_digit` matches Python `isdigit()` for every token that can occur.
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
}

/// `s.replace(pat, '', 1)`: remove the FIRST occurrence of `pat`, else return `s` unchanged.
fn replace_first(s: &str, pat: &str) -> String {
    match s.find(pat) {
        Some(idx) => format!("{}{}", &s[..idx], &s[idx + pat.len()..]),
        None => s.to_string(),
    }
}

/// Faithful port of `mask_elementary_literals` (utils.py:679): replace every token for which
/// [`is_numeric_string`] holds (e.g. `0`, `1`, `14`, `3.14`) with `<constant>`. The final step of
/// `simplify` (after sort), abstracting the literal coefficients/neutrals that cancellation emits.
pub fn mask_elementary_literals(expression: &[String]) -> Vec<String> {
    expression
        .iter()
        .map(|t| {
            if is_numeric_string(t) {
                "<constant>".to_string()
            } else {
                t.clone()
            }
        })
        .collect()
}

/// Faithful port of `numbers_to_constant` (utils.py:259): replace every token for which Python's
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

/// Faithful port of `remove_pow1` (utils.py:898): drop every `pow1` token (raising-to-1 identity)
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

/// Faithful port of `utils.is_prime` (utils.py:396). NOTE: this mirrors the SOURCE exactly,
/// quirks included -- it is only ever invoked from `cancel_terms` with `abs(sum) > 5`, but the
/// faithful contract is exact replication, not a "correct" primality test. Python:
/// ```python
/// if n % 2 == 0 and n > 2: return False
/// return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))
/// ```
/// `int(math.sqrt(n))` and Rust `(n as f64).sqrt() as i64` are the SAME platform-libm sqrt
/// truncated toward zero, so the perfect-square float edge (if it ever bit) bites both identically.
pub fn is_prime(n: i64) -> bool {
    if n % 2 == 0 && n > 2 {
        return false;
    }
    let limit = (n as f64).sqrt() as i64; // == int(math.sqrt(n)) for n >= 0
    let mut i = 3i64;
    while i <= limit {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

/// Faithful port of `utils.factorize_to_at_most` (utils.py:583). Decomposes `p` into factors
/// each `<= max_factor` whose product is `p`, in discovery order (NOT sorted). Returns `Err(())`
/// where Python raises `ValueError` -- the caller (`cancel_terms`) branches on the raise to a
/// `*`/`pow`-coefficient fallback, so the raise conditions are CONTROL FLOW and must match exactly:
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
    use super::{factorize_to_at_most, is_prime};

    #[test]
    fn is_prime_matches_python() {
        // Python docstring examples + the values cancel_terms actually feeds (abs(sum) > 5).
        assert!(is_prime(29));
        assert!(!is_prime(30));
        assert!(is_prime(7));
        assert!(!is_prime(9));
        assert!(is_prime(11));
        assert!(!is_prime(15));
        assert!(is_prime(13));
        assert!(!is_prime(25)); // perfect square: sqrt truncation must still catch the divisor 5
    }

    #[test]
    fn factorize_matches_python_examples() {
        assert_eq!(factorize_to_at_most(100, 10, 1000), Ok(vec![4, 5, 5]));
        assert_eq!(factorize_to_at_most(18, 5, 1000), Ok(vec![2, 3, 3]));
        assert_eq!(factorize_to_at_most(1, 5, 1000), Ok(vec![]));
        // Values cancel_terms reaches with max_factor = max_power = 5.
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
