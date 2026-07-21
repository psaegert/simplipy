//! The pattern matcher mirroring `simplipy/utils.py`: `match_pattern_with_cert` (the deployed
//! matcher entry), `apply_mapping`, and the placeholder-binding predicates. (`utils.rs` keeps
//! the string/number helpers.)
//! The tree helpers operate on interned [`Tok`] ids; the per-token predicates (sigil class,
//! `leaf_value`) are PRECOMPUTED per id and consulted through the [`TokenView`] -- same
//! functions, evaluated once at intern instead of at every use.

use rustc_hash::FxHashMap;

use crate::parse::Node;
use crate::tokens::{Tok, TokenView};

/// Does this subtree contain a `<constant>` token anywhere? Mirrors
/// `"<constant>" in flatten_nested_list(existing)` in `match_pattern`'s placeholder-rebind guard.
pub fn contains_constant(node: &Node, constant: Tok) -> bool {
    match node {
        Node::Leaf(t) => *t == constant,
        Node::Op { token, operands } => {
            *token == constant || operands.iter().any(|o| contains_constant(o, constant))
        }
    }
}

/// Diagnostic flag: with `SIMPLIPY_LEAF_WILDCARDS=1` every `_i` placeholder binds variable
/// leaves only (demoting all wildcards to the stricter `?`-sort semantics; a leaf slot's
/// pushforward is the identity) -- used to isolate whether a behavior difference comes from
/// composite wildcard bindings. Default off: the deployed subtree semantics, byte-identical.
fn leaf_wildcards_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_LEAF_WILDCARDS").as_deref() == Ok("1"))
}

/// A VARIABLE leaf, defined by EXCLUSION, never by name pattern: a leaf token that has no
/// literal value (`numeric::leaf_value` is the one shared table, per-id via the view) and is not
/// `<constant>` -- which denotes a constant, whose pushforward is a Dirac, the exact binding the
/// `?`-sort exists to forbid.
fn is_variable_leaf(node: &Node, view: &TokenView) -> bool {
    match node {
        Node::Leaf(tok) => *tok != view.table.constant && view.leaf_value(*tok).is_none(),
        Node::Op { .. } => false,
    }
}

/// `match_pattern_with_cert` is the faithful port of `match_pattern` (utils.py@0.2.15:800) plus a
/// `!`-sort certifier: it matches a query subtree `tree` against a rule LHS `pattern`; on
/// success `mapping` binds each placeholder id to the matched query subtree (borrowed from `tree`).
/// The bare-string-operand early-return branch in the Python source is DEAD for
/// `prefix_to_tree`-built patterns (operands are always subtrees), so it is intentionally omitted.
/// The certifier `cert(subtree)` answers "is this subtree defined-and-finite a.e.?" (the engine
/// passes `interval::finite_ae` behind its per-engine cache). With `None`, `!` slots bind
/// VARIABLE LEAVES only -- fail-closed: a caller that cannot certify loses `!`-subtree
/// capability, never soundness.
///
/// The walk is purely syntactic (`!` binds composites freely when a certifier exists); the
/// certificates run ONCE per COMPLETED syntactic match, over the final `!`-bindings -- the same
/// conjunction as certifying mid-walk (identical verdicts), but never paid on attempts that
/// fail syntactically elsewhere.
pub fn match_pattern_with_cert<'a>(
    tree: &'a Node,
    pattern: &Node,
    mapping: &mut FxHashMap<Tok, &'a Node>,
    cert: Option<&dyn Fn(&Node) -> bool>,
    view: &TokenView,
) -> bool {
    if !match_pattern_impl(
        tree,
        pattern,
        mapping,
        leaf_wildcards_only(),
        cert.is_some(),
        view,
    ) {
        return false;
    }
    for (pkey, bound) in mapping.iter() {
        if view.sigil(*pkey) == b'!' && !is_variable_leaf(bound, view) {
            match cert {
                Some(f) if f(bound) => {}
                _ => return false, // uncertified or uncertifiable: no binding (fail-closed)
            }
        }
    }
    true
}

fn match_pattern_impl<'a>(
    tree: &'a Node,
    pattern: &Node,
    mapping: &mut FxHashMap<Tok, &'a Node>,
    leaf_only: bool,
    bang_binds_composite: bool,
    view: &TokenView,
) -> bool {
    match pattern {
        // Elementary (leaf) pattern.
        Node::Leaf(pkey) => {
            let sigil = view.sigil(*pkey);
            if sigil != 0 {
                // Placeholder. Python keys on the sigil prefix, not the full ^[_?!]\d+$ regex.
                // The SORT is the sigil: `?` always binds
                // a variable leaf or nothing; `_` binds any subtree unless the
                // SIMPLIPY_LEAF_WILDCARDS diagnostic demotes it; `!` binds a variable leaf
                // freely, or a SUBTREE only when the
                // certifier proves it defined-and-finite a.e.
                if (sigil == b'?' || leaf_only) && !is_variable_leaf(tree, view) {
                    return false;
                }
                if sigil == b'!' && !is_variable_leaf(tree, view) && !bang_binds_composite {
                    return false; // no certifier available: `!` binds variable leaves only
                }
                match mapping.get(pkey) {
                    None => {
                        mapping.insert(*pkey, tree);
                        true
                    }
                    Some(existing) => {
                        // Cannot rebind a subtree that contains an (independent) <constant>.
                        if contains_constant(existing, view.table.constant) {
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
                    if !match_pattern_impl(
                        t_operand,
                        p_operand,
                        mapping,
                        leaf_only,
                        bang_binds_composite,
                        view,
                    ) {
                        return false;
                    }
                }
                true
            }
        },
    }
}

/// Faithful port of `apply_mapping` (utils.py@0.2.15:762): substitute each placeholder leaf (`_N`) in the
/// replacement template with its bound subtree, returning a fresh tree. Concrete leaves and operator
/// structure are copied. A placeholder with no binding panics (mirrors Python's `KeyError`); the
/// wildcard-multiplicity invariant guarantees RHS placeholders are a subset of bound LHS ones.
pub fn apply_mapping(template: &Node, mapping: &FxHashMap<Tok, &Node>, view: &TokenView) -> Node {
    match template {
        Node::Leaf(t) => {
            if view.sigil(*t) != 0 {
                (*mapping.get(t).expect("rhs placeholder is bound")).clone()
            } else {
                template.clone()
            }
        }
        Node::Op { token, operands } => Node::Op {
            token: *token,
            operands: operands
                .iter()
                .map(|o| apply_mapping(o, mapping, view))
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::match_pattern_impl;
    use crate::operators::{OperatorSpec, Operators};
    use crate::parse::{parse_subtree, Node};
    use crate::tokens::{TokenOverlay, TokenTable, TokenView};
    use rustc_hash::FxHashMap;
    use std::cell::RefCell;

    /// A minimal operator universe for the matcher tests (arity only).
    fn test_ops() -> (Vec<String>, Operators) {
        let names = [
            "+", "-", "*", "/", "pow", "log", "exp", "sin", "abs", "inv", "pow3", "mult3",
        ];
        let binary = ["+", "-", "*", "/", "pow"];
        let mut order = Vec::new();
        let mut specs: FxHashMap<String, OperatorSpec> = FxHashMap::default();
        for n in names {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: Vec::new(),
                    inverse: None,
                    arity: if binary.contains(&n) { 2 } else { 1 },
                    precedence: None,
                    commutative: false,
                },
            );
        }
        let ops = Operators::from_specs(order.clone(), specs);
        (order, ops)
    }

    fn tree(tokens: &[&str], view: &TokenView) -> Node {
        let toks: Vec<crate::tokens::Tok> = tokens.iter().map(|t| view.intern(t)).collect();
        let (node, consumed) = parse_subtree(&toks, 0, &|t| view.arity(t));
        assert_eq!(consumed, toks.len(), "trailing tokens in test expression");
        node
    }

    fn matches(query: &[&str], pattern: &[&str], leaf_only: bool) -> bool {
        let (order, ops) = test_ops();
        let table = TokenTable::build(&order, &ops);
        let overlay = RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        let (q, p) = (tree(query, &view), tree(pattern, &view));
        let mut m = FxHashMap::default();
        match_pattern_impl(&q, &p, &mut m, leaf_only, false, &view)
    }

    /// Under `?`-sort semantics (leaf_only=true) a slot binds a variable leaf or nothing.
    /// The rows pin the exact boundary: variables bind in both modes; composite subtrees,
    /// literals, and `<constant>` bind ONLY under the deployed subtree semantics.
    #[test]
    fn leaf_wildcards_bind_variables_only() {
        // variable leaf: binds in BOTH modes (x/x -> 1 stays golden)
        assert!(matches(&["/", "x0", "x0"], &["/", "_0", "_0"], false));
        assert!(matches(&["/", "x0", "x0"], &["/", "_0", "_0"], true));
        // composite subtree: subtree mode only
        assert!(matches(
            &["/", "log", "x0", "log", "x0"],
            &["/", "_0", "_0"],
            false
        ));
        assert!(!matches(
            &["/", "log", "x0", "log", "x0"],
            &["/", "_0", "_0"],
            true
        ));
        // literal 0 (a Dirac on the hole): subtree mode only
        assert!(matches(&["/", "0", "0"], &["/", "_0", "_0"], false));
        assert!(!matches(&["/", "0", "0"], &["/", "_0", "_0"], true));
        // <constant> denotes a constant -> Dirac -> excluded under ?-sort
        assert!(matches(&["*", "0", "<constant>"], &["*", "0", "_0"], false));
        assert!(!matches(&["*", "0", "<constant>"], &["*", "0", "_0"], true));
        // np.pi is a literal, not a variable (exclusion test, not name pattern)
        assert!(!matches(&["*", "0", "np.pi"], &["*", "0", "_0"], true));
        // non-linear pattern still requires the SAME variable under leaf mode
        assert!(!matches(&["/", "x0", "x1"], &["/", "_0", "_0"], true));
        // abs(pow(x, mult3(1))) -> pow(x, mult3(1)): `_1 := mult3(1)` is a COMPOSITE subtree
        // (a Dirac on an odd integer) -- the deployed semantics accepts it, the ?-sort kills it.
        assert!(matches(
            &["abs", "pow", "x0", "mult3", "1"],
            &["abs", "pow", "_0", "_1"],
            false
        ));
        assert!(!matches(
            &["abs", "pow", "x0", "mult3", "1"],
            &["abs", "pow", "_0", "_1"],
            true
        ));
        // two DISTINCT variable slots: untouched by the flag
        assert!(matches(
            &["abs", "pow", "x0", "x1"],
            &["abs", "pow", "_0", "_1"],
            true
        ));
    }

    /// The two sorts side by side: `?N` binds a variable leaf REGARDLESS of the global flag;
    /// `_N` binds any subtree at default. The sort rides in the token.
    #[test]
    fn sort_sigils_decide_binding() {
        // `?` refuses composite subtrees, literals, and <constant> even with the flag OFF
        assert!(matches(&["/", "x0", "x0"], &["/", "?0", "?0"], false));
        assert!(!matches(
            &["/", "log", "x0", "log", "x0"],
            &["/", "?0", "?0"],
            false
        ));
        assert!(!matches(&["/", "0", "0"], &["/", "?0", "?0"], false));
        assert!(!matches(
            &["*", "0", "<constant>"],
            &["*", "0", "?0"],
            false
        ));
        // `_` keeps the deployed subtree semantics at default
        assert!(matches(
            &["/", "log", "x0", "log", "x0"],
            &["/", "_0", "_0"],
            false
        ));
        // mixed pattern: each slot by its own sigil
        assert!(matches(
            &["+", "x0", "log", "x1"],
            &["+", "?0", "_1"],
            false
        ));
        assert!(!matches(
            &["+", "log", "x1", "x0"],
            &["+", "?0", "_1"],
            false
        ));
    }
}
