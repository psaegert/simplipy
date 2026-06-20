//! Faithful port of `sort_operands` (engine.py:1636) + `operand_key` (engine.py:2512): the canonical
//! operand ordering for commutative operators (`+`, `*`), the last stage of the `simplify` fixpoint
//! (it runs ONCE, after the loop). Public entry: [`sort_operands_unit`].
//!
//! ## Mechanism (and one deliberate quirk)
//! Right-to-left scan builds subtrees bottom-up. For a commutative node it:
//!  1. **Rotation special case** (engine.py:1667): if the LEFT operand is a composite Op with the
//!     SAME operator (`op(op(A,B), C)`), right-rotate to `op(A, op(B,C))` and `continue` -- which
//!     SKIPS the sort for this node. This is faithful: left-nested chains at the root come out only
//!     partially sorted (verified: `sort(['+','+','x2','x3','x1']) == ['+','x2','+','x3','x1']`). Do
//!     NOT "fix" this to fully canonicalize.
//!  2. Otherwise: gather the maximal same-operator chain's BOUNDARY operands (leaves or
//!     different-operator composites) as index PATHS, sort the paths lexicographically, sort the
//!     operands by [`operand_key`] (STABLE), and place sorted-operand[i] at sorted-path[i] in a clone.
//!
//! BFS-vs-recursion order is irrelevant: the paths are sorted afterward, so only the SET of boundary
//! paths matters (advisor). Stability IS load-bearing (composite duplicates survive cancel).
//!
//! ## `operand_key` (engine.py:2512) -- a heterogeneous tuple, lifted to [`Key`]
//!  * non-numeric leaf  -> `(0, token)`        -> [`Key::Var`]  (sorts FIRST)
//!  * numeric leaf       -> `(1, float(token))` -> [`Key::Num`]  (sorts SECOND)
//!  * composite node     -> `(2, len, child_keys, op)` -> [`Key::Node`] (sorts LAST)
//! Cross-tag order is fixed by the leading int (0<1<2), so no heterogeneous comparison ever happens.
//! For dev_7-3 the only `float()`-parseable leaves corpus-wide are `0`/`1` (inf/nan/`(-1)`/`np.pi`
//! all fail `float()` -> tag 0), so the f64-not-`Ord` hazard is moot; we use `total_cmp` anyway.

use std::cmp::Ordering;

use crate::operators::Operators;
use crate::parse::{tree_to_prefix, Node};

/// The `operand_key` value (engine.py:2512), as an ordered enum mirroring the Python tuple.
enum Key {
    /// `(0, token)` -- a non-numeric leaf (variable / `<constant>` / named const / `(-1)`).
    Var(String),
    /// `(1, float(token))` -- a numeric leaf.
    Num(f64),
    /// `(2, len(flatten(node)), child_keys, op)` -- a composite subtree.
    Node { len: usize, children: Vec<Key>, op: String },
}

#[inline]
fn tag(k: &Key) -> u8 {
    match k {
        Key::Var(_) => 0,
        Key::Num(_) => 1,
        Key::Node { .. } => 2,
    }
}

/// Compare two keys exactly as Python compares the `operand_key` tuples: the leading int tag orders
/// across kinds (0<1<2); within a kind, `Var` by string (byte order == code-point order for the
/// ASCII grammar tokens), `Num` by `total_cmp` (only 0.0/1.0 ever occur), `Node` by
/// `(len, children, op)` -- where `Vec<Key>` is lexicographic with shorter<longer on a common
/// prefix, matching Python tuple comparison of unequal-length key tuples.
fn key_cmp(a: &Key, b: &Key) -> Ordering {
    match (a, b) {
        (Key::Var(x), Key::Var(y)) => x.cmp(y),
        (Key::Num(x), Key::Num(y)) => x.total_cmp(y),
        (Key::Node { len: l1, children: c1, op: o1 }, Key::Node { len: l2, children: c2, op: o2 }) => {
            l1.cmp(l2)
                .then_with(|| vec_key_cmp(c1, c2))
                .then_with(|| o1.cmp(o2))
        }
        _ => tag(a).cmp(&tag(b)),
    }
}

/// Lexicographic comparison of two key tuples (Python tuple comparison: element-wise, then
/// shorter<longer on a common prefix).
fn vec_key_cmp(a: &[Key], b: &[Key]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match key_cmp(x, y) {
            Ordering::Equal => continue,
            non_eq => return non_eq,
        }
    }
    a.len().cmp(&b.len())
}

/// Total token count of a subtree's prefix (`len(flatten_nested_list(node))`): the operator plus
/// every descendant token.
fn prefix_len(node: &Node) -> usize {
    match node {
        Node::Leaf(_) => 1,
        Node::Op { operands, .. } => 1 + operands.iter().map(prefix_len).sum::<usize>(),
    }
}

/// Faithful port of `operand_key` (engine.py:2512).
fn operand_key(node: &Node) -> Key {
    match node {
        // Node: `(2, len(flatten(operands)), tuple(operand_key(c) for c in operands[1]), operands[0])`.
        Node::Op { token, operands } => Key::Node {
            len: prefix_len(node),
            children: operands.iter().map(operand_key).collect(),
            op: token.clone(),
        },
        // Leaf: `try (1, float(token)) except ValueError: (0, token)`.
        Node::Leaf(t) => match python_float(t) {
            Some(f) => Key::Num(f),
            None => Key::Var(t.clone()),
        },
    }
}

/// `float(token)` with Python's accept/reject behaviour for the tokens that occur on the deployment
/// path. Rust `f64::from_str` agrees with Python `float()` on every leaf in the corpus (`0`/`1`
/// parse; `<constant>` / `x*` / `(-1)` / `np.pi` / `float("inf")` all reject -> `Var`). The two
/// known divergences -- Python accepts digit underscores (`1_0`) and surrounding whitespace -- do
/// not occur in the grammar's leaf tokens.
fn python_float(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

/// Gather the BOUNDARY operand paths of the maximal same-`operator` chain rooted at `node`: a child
/// that is a leaf, or a composite with a DIFFERENT operator, is a boundary (recorded); a composite
/// with the SAME operator is followed. Order is irrelevant (paths are sorted by the caller).
fn collect_positions(node: &Node, operator: &str, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    let operands = match node {
        Node::Op { operands, .. } => operands,
        Node::Leaf(_) => return,
    };
    for (child_index, child) in operands.iter().enumerate() {
        path.push(child_index);
        match child {
            Node::Op { token, .. } if token == operator => {
                collect_positions(child, operator, path, out);
            }
            _ => out.push(path.clone()),
        }
        path.pop();
    }
}

/// Navigate to the node at `path` (a sequence of operand indices).
fn node_at<'a>(mut node: &'a Node, path: &[usize]) -> &'a Node {
    for &idx in path {
        node = match node {
            Node::Op { operands, .. } => &operands[idx],
            Node::Leaf(_) => unreachable!("path indexes into a leaf"),
        };
    }
    node
}

/// Navigate to a mutable handle on the node at `path`.
fn node_at_mut<'a>(mut node: &'a mut Node, path: &[usize]) -> &'a mut Node {
    for &idx in path {
        node = match node {
            Node::Op { operands, .. } => &mut operands[idx],
            Node::Leaf(_) => unreachable!("path indexes into a leaf"),
        };
    }
    node
}

/// The commutative-node sort (engine.py:1673-1722): gather boundary paths, lex-sort them, key-sort
/// the operands (STABLE), and place sorted-operand[i] at sorted-path[i] in a clone of the subtree.
fn sort_commutative_node(subtree: Node, operator: &str) -> Node {
    let mut positions: Vec<Vec<usize>> = Vec::new();
    collect_positions(&subtree, operator, &mut Vec::new(), &mut positions);
    positions.sort(); // lexicographic, == Python `sorted(..., key=position-tuple)`

    let operands_to_sort: Vec<&Node> = positions.iter().map(|p| node_at(&subtree, p)).collect();
    let keys: Vec<Key> = operands_to_sort.iter().map(|n| operand_key(n)).collect();

    // STABLE sort of indices by key: Python `sorted(operands_to_sort, key=operand_key)` over operands
    // already in lex-position order. `sort_by` is stable.
    let mut order: Vec<usize> = (0..operands_to_sort.len()).collect();
    order.sort_by(|&i, &j| key_cmp(&keys[i], &keys[j]));
    let sorted_operands: Vec<Node> = order.iter().map(|&k| operands_to_sort[k].clone()).collect();

    let mut new_subtree = subtree.clone();
    for (position, operand) in positions.iter().zip(sorted_operands.into_iter()) {
        *node_at_mut(&mut new_subtree, position) = operand;
    }
    new_subtree
}

/// Faithful port of `sort_operands` (engine.py:1636). On a malformed expression that does not
/// collapse to a single root, returns the input unchanged (out of the deployment contract).
pub fn sort_operands_unit(expression: &[String], ops: &Operators) -> Vec<String> {
    let mut stack: Vec<Node> = Vec::new();

    let mut i = expression.len() as isize - 1;
    while i >= 0 {
        let token = &expression[i as usize];

        if let Some((operator, arity)) = ops.sort_resolve(token) {
            if stack.len() < arity {
                return expression.to_vec(); // malformed (arity underflow) -> passthrough
            }
            // operands = list(reversed(stack[-arity:]))
            let mut operands: Vec<Node> = stack.split_off(stack.len() - arity);
            operands.reverse();

            if ops.is_commutative(&operator) {
                // Rotation special case: left operand is a composite Op with the SAME operator.
                // `op(op(A,B), C)` -> `op(A, op(B,C))`, then SKIP the sort (continue).
                let rotate = matches!(&operands[0], Node::Op { token, .. } if *token == operator);
                if rotate {
                    let (a, b) = match &operands[0] {
                        Node::Op { operands: lops, .. } => (lops[0].clone(), lops[1].clone()),
                        Node::Leaf(_) => unreachable!(),
                    };
                    let c = operands[1].clone();
                    let inner = Node::Op { token: operator.clone(), operands: vec![b, c] };
                    stack.push(Node::Op { token: operator.clone(), operands: vec![a, inner] });
                    i -= 1;
                    continue;
                }

                let subtree = Node::Op { token: operator.clone(), operands };
                stack.push(sort_commutative_node(subtree, &operator));
                i -= 1;
                continue;
            }

            // Non-commutative operator: rebuild with operands in order (alias-canonicalized operator).
            stack.push(Node::Op { token: operator, operands });
        } else {
            // Leaf.
            stack.push(Node::Leaf(token.clone()));
        }

        i -= 1;
    }

    // `flatten_nested_list(stack)[::-1]` == the single root's prefix.
    if stack.len() == 1 {
        let mut out = Vec::new();
        tree_to_prefix(&stack[0], &mut out);
        out
    } else {
        expression.to_vec() // malformed (multi-root) -> passthrough
    }
}

#[cfg(test)]
mod tests {
    use crate::Engine;

    fn engine() -> Engine {
        let home = std::env::var("HOME").unwrap();
        Engine::from_paths(
            &format!("{home}/.cache/simplipy/engines/dev_7-3/config.yaml"),
            &format!("{home}/.cache/simplipy/engines/dev_7-3/rules.json"),
        )
        .expect("engine loads")
    }

    fn t(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    /// Canonical sort cases cross-checked against fresh Python (see benchmarks/diff_sort.py for the
    /// 10037+ corpus gate). Pins: simple swap, non-commutative no-op, the tag ordering (var < num),
    /// stability on equal composite operands, and the rotation-SKIPS-sort quirk (faithful, NOT
    /// "cleaned up" -- left-nested chains come out only partially sorted).
    #[test]
    fn sort_canonical_cases() {
        let e = engine();
        let cases: &[(&[&str], &[&str])] = &[
            (&["+", "x3", "x1"], &["+", "x1", "x3"]),
            (&["*", "x3", "x1"], &["*", "x1", "x3"]),
            (&["-", "x3", "x1"], &["-", "x3", "x1"]),         // non-commutative: unchanged
            (&["+", "0", "x1"], &["+", "x1", "0"]),            // var(tag0) < num(tag1)
            (&["+", "<constant>", "x1"], &["+", "<constant>", "x1"]), // '<' < 'x'
            (&["+", "sin", "x1", "sin", "x1"], &["+", "sin", "x1", "sin", "x1"]), // stable, equal
            // rotation fires -> SKIPS sort (x2,x3,x1 NOT fully sorted):
            (&["+", "+", "x2", "x3", "x1"], &["+", "x2", "+", "x3", "x1"]),
            // right-nested chain is fully sorted:
            (&["+", "x1", "+", "x3", "x2"], &["+", "x1", "+", "x2", "x3"]),
            // composite operand sorts after leaves (tag 2):
            (&["*", "x2", "+", "x1", "x3"], &["*", "x2", "+", "x1", "x3"]),
        ];
        for (input, expected) in cases {
            assert_eq!(e.sort_operands(&t(input)), t(expected), "input {input:?}");
        }
    }
}
