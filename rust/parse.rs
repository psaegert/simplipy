//! Prefix <-> tree parsing, mirroring `engine.parse_subtree` (engine.py:1033) and
//! `engine.prefix_to_tree` (engine.py:935).
//!
//! The tree node is the Rust analogue of simplipy's nested-list representation
//! `[op, [child], [child], ...]` with leaves as single-element lists `[token]`. We use a flat
//! arena keyed by index to avoid Box churn in the hot recursion; faithful traversal order
//! (depth-first, operands left-to-right) is preserved exactly.

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// A leaf token: variable (x1..), `<constant>`, named const ((-1), np.pi, np.e), or numeric.
    Leaf(String),
    /// An operator with its operand subtrees, in order.
    Op { token: String, operands: Vec<Node> },
}

impl Node {
    #[inline]
    pub fn root_token(&self) -> &str {
        match self {
            Node::Leaf(t) => t,
            Node::Op { token, .. } => token,
        }
    }
}

/// Parse one prefix subtree starting at `start_idx`; returns (node, next_index).
///
/// Faithful port of `parse_subtree` (engine.py:1033). A token with a known arity is an operator and
/// consumes that many operand subtrees; anything else is a leaf. For dev_7-3 this is also a faithful
/// stand-in for `prefix_to_tree` (used on rule patterns): VERIFIED that every operator has arity 1
/// or 2 (no arity-0 operators) and no alias / `**` token appears in the rules or skeletons, so the
/// two parsers agree on every token that can occur.
pub fn parse_subtree(
    tokens: &[String],
    start_idx: usize,
    arity_of: &dyn Fn(&str) -> Option<u8>,
) -> (Node, usize) {
    let token = &tokens[start_idx];
    match arity_of(token) {
        Some(arity) => {
            let mut operands = Vec::with_capacity(arity as usize);
            let mut idx = start_idx + 1;
            for _ in 0..arity {
                let (operand, next) = parse_subtree(tokens, idx, arity_of);
                operands.push(operand);
                idx = next;
            }
            (
                Node::Op {
                    token: token.clone(),
                    operands,
                },
                idx,
            )
        }
        None => (Node::Leaf(token.clone()), start_idx + 1),
    }
}

/// Flatten a tree back to a prefix token list (inverse of `parse_subtree`): emit the operator, then
/// each operand left-to-right. Equivalent to `flatten_nested_list(tree)[::-1]` in Python, computed
/// directly in prefix order.
pub fn tree_to_prefix(node: &Node, out: &mut Vec<String>) {
    match node {
        Node::Leaf(t) => out.push(t.clone()),
        Node::Op { token, operands } => {
            out.push(token.clone());
            for operand in operands {
                tree_to_prefix(operand, out);
            }
        }
    }
}
