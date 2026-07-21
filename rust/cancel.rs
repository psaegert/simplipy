//! Faithful port of the term-cancellation unit: `collect_multiplicities` (engine.py@0.2.15:1290) feeding
//! `cancel_terms` (engine.py@0.2.15:1410). In the `simplify` fixpoint these always run as a pair
//! (`cancel_terms(*collect_multiplicities(expr))`), so the public entry here is the fused unit
//! [`cancel_terms_unit`].
//!
//! ## What it does (the actual mechanism, not the docstring)
//! Within a maximal *connected region* of one connection class -- additive (`+`/`-`) or
//! multiplicative (`*`/`/`) -- it counts the SIGNED multiplicity of each LEAF token and, if some
//! leaf's `sum(|pos|,|neg|) > 1`, merges its occurrences: the first occurrence becomes the merged
//! term (`mult{k}`/`pow{k}` hyper-operator, or a `neg`/`inv` inverse prefix, or a `*k`/`pow ... k`
//! coefficient fallback) and every later occurrence becomes the class neutral (`0`/`1`). Only ONE
//! cancellation candidate is taken per call; the fixpoint re-invokes until convergence. Composite
//! subtrees do NOT register as cancellable hashes (only leaf `(token,)` hashes propagate), so the
//! `(a*b)+(a*b)` docstring example is not what the code actually cancels.
//!
//! ## Faithfulness traps (verified against the source)
//! * The annotation dicts are Python dicts iterated in **insertion order** -> replicated with an
//!   insertion-ordered `Vec<(key,[i64;2])>`, NOT a hash map.
//! * Candidate selection does **NOT break**: it runs the full `[add, mult] x dict.items()` nest, so
//!   the candidate is the **last** qualifying match, not the first.
//! * `factorize_to_at_most` raising `ValueError` is control flow -> the `Err` arm selects the
//!   `*`/`pow`-coefficient fallback.
//! * The `[::-1]` reversed-flatten label of a subtree is exactly its prefix-token sequence (leaves
//!   are `(token,)`), so prefix token ids serve as the label directly.
//!
//! Labels/hashes are `Vec<Tok>` (interning is injective within a call, so
//! `Vec<Tok>` equality == the old `Vec<String>` equality); the class operators/neutrals/inverses
//! compare against the distinguished table ids; the emitted hyper/coefficient tokens are interned
//! at emission (`view.intern`).

use crate::operators::Operators;
use crate::tokens::{Tok, TokenTable, TokenView};

// Connection classes, hardcoded exactly as engine.py@0.2.15:172-176 (NOT config-derived). Iteration order
// of `self.connection_classes` is the dict insertion order [add, mult]; we index by these consts.
const CC_ADD: usize = 0;
const CC_MULT: usize = 1;
const N_CC: usize = 2;

/// `connection_classes[cc][0]` -- the operator set of each class (positive op first).
#[inline]
fn connection_ops(cc: usize, tt: &TokenTable) -> [Tok; 2] {
    match cc {
        CC_ADD => [tt.plus, tt.minus],
        _ => [tt.star, tt.slash],
    }
}

/// `connection_classes[cc][1]` -- the neutral element of each class (`0` / `1`).
#[inline]
fn neutral(cc: usize, tt: &TokenTable) -> Tok {
    match cc {
        CC_ADD => tt.zero,
        _ => tt.one,
    }
}

/// `connection_classes_inverse` (engine.py@0.2.15:174): the unary inverse operator of each class.
#[inline]
fn cc_inverse(cc: usize, tt: &TokenTable) -> Tok {
    match cc {
        CC_ADD => tt.neg,
        _ => tt.inv,
    }
}

/// `connection_classes_hyper` (engine.py@0.2.15:175): the hyper-operator PREFIX of each class (the
/// emitted token is `format!("{hyper}{k}")`, interned at emission).
const CC_HYPER: [&str; N_CC] = ["mult", "pow"];

/// `token in self.binary_connectable_operators` (engine.py@0.2.15:176).
#[inline]
fn is_binary_connectable(token: Tok, tt: &TokenTable) -> bool {
    token == tt.plus || token == tt.minus || token == tt.star || token == tt.slash
}

/// `self.operator_to_class[operator]` (engine.py@0.2.15:173) for a binary-connectable operator.
#[inline]
fn operator_to_class(op: Tok, tt: &TokenTable) -> usize {
    if op == tt.plus || op == tt.minus {
        CC_ADD
    } else if op == tt.star || op == tt.slash {
        CC_MULT
    } else {
        unreachable!("operator_to_class called on a non-connectable operator")
    }
}

/// An insertion-ordered multiplicity dict: `{ token-tuple hash -> [pos, neg] }`. A `Vec` of pairs
/// (not a hash map) so iteration order matches Python dict insertion order exactly (load-bearing for
/// candidate selection). The dicts are tiny (a handful of leaf hashes), so linear lookup is cheap.
type Ann = Vec<(Vec<Tok>, [i64; 2])>;

/// `if hash not in dict: dict[hash] = [0,0]` then return a mutable handle to `dict[hash]`.
/// Preserves insertion order (new keys appended at the end).
fn ann_entry<'a>(dict: &'a mut Ann, hash: &[Tok]) -> &'a mut [i64; 2] {
    if let Some(pos) = dict.iter().position(|(k, _)| k.as_slice() == hash) {
        &mut dict[pos].1
    } else {
        dict.push((hash.to_vec(), [0, 0]));
        &mut dict.last_mut().unwrap().1
    }
}

/// The annotated subtree: the fused analogue of Python's parallel `stack` / `stack_annotations` /
/// `stack_labels` entries. Bundling them removes the desync hazard of mirroring three `Vec`s.
struct AnnNode {
    /// `None` for a leaf (`len(subtree) == 1`), `Some(operator)` for a composite node.
    op: Option<Tok>,
    /// Leaf token, or (for a composite) the operator token. Equals `subtree[0]`.
    token: Tok,
    /// Child subtrees, left-to-right (empty for a leaf).
    operands: Vec<AnnNode>,
    /// The node's OWN annotation dict per connection class (`subtree_annotation[0]`).
    own: [Ann; N_CC],
    /// `subtree_labels[0]`: the subtree's prefix-token sequence (`flatten([op,operands])[::-1]`).
    label: Vec<Tok>,
}

/// Faithful port of `collect_multiplicities` (engine.py@0.2.15:1290). Right-to-left scan building a stack
/// of annotated subtrees; for a well-formed prefix expression the stack ends with a single root,
/// which is returned. Mirrors the leaf / binary-connectable / general-operator branches exactly.
fn collect_multiplicities(expression: &[Tok], view: &TokenView) -> Option<AnnNode> {
    let tt = view.table;
    let mut stack: Vec<AnnNode> = Vec::new();

    let mut i = expression.len() as isize - 1;
    while i >= 0 {
        let token = expression[i as usize];

        if is_binary_connectable(token, tt) {
            let operator = token;
            let arity = 2usize;
            // operands = list(reversed(stack[-arity:])): pop the top `arity`, restore left->right.
            let mut operands: Vec<AnnNode> = stack.split_off(stack.len() - arity);
            operands.reverse();

            let cc = operator_to_class(operator, tt);
            let is_inverse_op = operator == tt.minus || operator == tt.slash; // operator in {'-','/'}

            // Carry over annotations from operand nodes (only the operator's OWN class is populated).
            let mut own: [Ann; N_CC] = [Vec::new(), Vec::new()];
            for (branch, operand) in operands.iter().enumerate() {
                // operand_annotations_dict[0][cc] -- iterate in the operand's insertion order.
                for (subtree_hash, val) in &operand.own[cc] {
                    let entry = ann_entry(&mut own[cc], subtree_hash);
                    if is_inverse_op && branch == 1 {
                        // reversed: operator_dict[1-p] += operand[p]  ->  [1]+=val[0], [0]+=val[1]
                        entry[1] += val[0];
                        entry[0] += val[1];
                    } else {
                        entry[0] += val[0];
                        entry[1] += val[1];
                    }
                }
            }

            let label = build_label(operator, &operands);
            stack.push(AnnNode {
                op: Some(operator),
                token,
                operands,
                own,
                label,
            });
            i -= 1;
            continue;
        }

        if let Some(arity) = view.arity(token) {
            // General (non-connectable) operator: empty annotation in BOTH classes.
            let arity = arity as usize;
            let mut operands: Vec<AnnNode> = stack.split_off(stack.len() - arity);
            operands.reverse();
            let label = build_label(token, &operands);
            stack.push(AnnNode {
                op: Some(token),
                token,
                operands,
                own: [Vec::new(), Vec::new()],
                label,
            });
            i -= 1;
            continue;
        }

        // Leaf: registers itself with multiplicity [1,0]. Cancellation assumes the group
        // axioms, so a leaf registers ONLY in the connection classes where they hold:
        //   - VARIABLES in both classes (`x - x -> 0` is total; `x/x -> 1` fills a null hole);
        //   - a LITERAL where its value is a group element: nonzero finite in both classes; 0 in
        //     the ADDITIVE class only (`0/0 -> 1` and the sign-of-zero family are wrong answers);
        //     nan/+-inf NOWHERE (`nan/nan -> 1`, `inf - inf -> 0`: truth nan in all four);
        //   - `<constant>` keeps its historical registration.
        // Not registering is the same state a composite subtree has always had (own = empty), so
        // the cancel phase needs no other change; it also UN-SHADOWS the already-mined correct
        // ground rules (`('/','0','0') -> float("nan")` et al.) that this pass pre-empted.
        let leaf_hash = vec![token];
        let lv = if token == tt.constant {
            None
        } else {
            view.leaf_value(token)
        };
        let reg = |allowed: bool| {
            if allowed {
                vec![(leaf_hash.clone(), [1, 0])]
            } else {
                Vec::new()
            }
        };
        let own = [
            reg(lv.map(|v| v.is_finite()).unwrap_or(true)),
            reg(lv.map(|v| v.is_finite() && v != 0.0).unwrap_or(true)),
        ];
        stack.push(AnnNode {
            op: None,
            token,
            operands: Vec::new(),
            own,
            label: leaf_hash,
        });
        i -= 1;
    }

    // A well-formed prefix expression collapses to exactly one root subtree.
    if stack.len() == 1 {
        stack.pop()
    } else {
        None
    }
}

/// `tuple(flatten_nested_list([operator, operands])[::-1])` -- the subtree's prefix token sequence:
/// the operator followed by each operand's (already-prefix) label, left-to-right.
fn build_label(operator: Tok, operands: &[AnnNode]) -> Vec<Tok> {
    let mut label = Vec::with_capacity(1 + operands.iter().map(|o| o.label.len()).sum::<usize>());
    label.push(operator);
    for operand in operands {
        label.extend_from_slice(&operand.label);
    }
    label
}

/// A cancel_terms work-stack frame (the fused analogue of the parallel `stack` / `stack_parity` /
/// `stack_still_connected` entries; the node's annotation + label travel inside the `AnnNode`).
struct Frame<'a> {
    node: &'a AnnNode,
    parity: [i64; N_CC],
    still_connected: bool,
}

/// Faithful port of `cancel_terms` (engine.py@0.2.15:1410), the deployed `collect_statistics=False` path.
fn cancel_terms(root: &AnnNode, ops: &Operators, view: &TokenView) -> Vec<Tok> {
    let tt = view.table;
    let mut expression: Vec<Tok> = Vec::new();

    // (argmax_class, cancelled_subtree, cancelled_multiplicity_sum). Set at most once.
    let mut cancellation_candidate: Option<(usize, Vec<Tok>, i64)> = None;
    let mut n_replaced: i64 = 0;

    // stack initialized to the single root; parity {add:1, mult:1}; still_connected = False.
    let mut stack: Vec<Frame> = vec![Frame {
        node: root,
        parity: [1, 1],
        still_connected: false,
    }];

    while let Some(frame) = stack.pop() {
        let subtree = frame.node;
        let subtree_parities = frame.parity;
        let mut still_connected = frame.still_connected;

        if let Some((argmax_class, cancelled_subtree, cancelled_multiplicity_sum)) =
            cancellation_candidate
                .as_ref()
                .map(|(a, s, m)| (*a, s.clone(), *m))
        {
            // still_connected stays true only along operators of the cancellation class (or leaves).
            let st0 = subtree.token;
            let in_class = connection_ops(argmax_class, tt).contains(&st0);
            let not_operator = !view.is_operator(st0);
            still_connected = still_connected && (in_class || not_operator);

            if still_connected && cancelled_subtree == subtree.label {
                let neutral_element = neutral(argmax_class, tt);

                let (first_replacement, other_replacements): (Vec<Tok>, Vec<Tok>) =
                    if cancelled_subtree.as_slice() == [tt.constant] {
                        // A single <constant>: keep one, neutralize the rest.
                        (vec![tt.constant], vec![neutral_element])
                    } else {
                        let current_parity = subtree_parities[argmax_class];
                        let inverse_operator = cc_inverse(argmax_class, tt);

                        // Negative parity and negative multiplicity cancel out (engine.py@0.2.15:1483).
                        let inverse_operator_prefix: Vec<Tok> =
                            if current_parity * cancelled_multiplicity_sum >= 0 {
                                Vec::new()
                            } else {
                                vec![inverse_operator]
                            };
                        // `double_inverse_operator_prefix` (engine.py@0.2.15:1485/1488) is computed in the
                        // source but never consumed -> intentionally omitted (verified dead).

                        let mut fr: Vec<Tok> = Vec::new();
                        let mut orr: Vec<Tok> = Vec::new();

                        if cancelled_multiplicity_sum == 0 {
                            // Cancelled entirely: every occurrence -> neutral element.
                            fr = vec![neutral_element];
                            orr = vec![neutral_element];
                        }
                        if cancelled_multiplicity_sum.abs() == 1 {
                            // Occurs once: first keeps the (possibly inverted) term, rest neutral.
                            fr = inverse_operator_prefix.clone();
                            fr.extend(cancelled_subtree.iter().copied());
                            orr = vec![neutral_element];
                        }
                        if cancelled_multiplicity_sum.abs() > 1 {
                            // Occurs multiple times: first becomes a hyper-power of the term.
                            let hyper_operator = CC_HYPER[argmax_class];
                            let pos_operator = connection_ops(argmax_class, tt)[0]; // positive-multiplicity op
                            let magnitude = cancelled_multiplicity_sum.abs();

                            let built: Result<Vec<Tok>, ()> = if cancelled_multiplicity_sum > 5
                                && crate::utils::is_prime(magnitude)
                            {
                                crate::utils::factorize_to_at_most(
                                    (magnitude - 1) as i128,
                                    ops.max_power,
                                    1000,
                                )
                                .map(|powers| {
                                    let mut r = inverse_operator_prefix.clone();
                                    r.push(pos_operator);
                                    for p in &powers {
                                        r.push(view.intern(&format!("{hyper_operator}{p}")));
                                    }
                                    r.extend(cancelled_subtree.iter().copied());
                                    r.extend(cancelled_subtree.iter().copied());
                                    r
                                })
                            } else {
                                crate::utils::factorize_to_at_most(
                                    magnitude as i128,
                                    ops.max_power,
                                    1000,
                                )
                                .map(|powers| {
                                    let mut r = inverse_operator_prefix.clone();
                                    for p in &powers {
                                        r.push(view.intern(&format!("{hyper_operator}{p}")));
                                    }
                                    r.extend(cancelled_subtree.iter().copied());
                                    r
                                })
                            };

                            fr = match built {
                                Ok(r) => r,
                                Err(()) => {
                                    // ValueError fallback: explicit integer coefficient.
                                    let coefficient_token = view.intern(&magnitude.to_string());
                                    let mut r = inverse_operator_prefix.clone();
                                    if argmax_class == CC_ADD {
                                        r.push(tt.star);
                                        r.push(coefficient_token);
                                        r.extend(cancelled_subtree.iter().copied());
                                    } else {
                                        r.push(tt.pow);
                                        r.extend(cancelled_subtree.iter().copied());
                                        r.push(coefficient_token);
                                    }
                                    r
                                }
                            };
                            orr = vec![neutral_element];
                        }

                        (fr, orr)
                    };

                if n_replaced == 0 {
                    expression.extend(first_replacement);
                } else {
                    expression.extend(other_replacements);
                }
                n_replaced += 1;
                continue;
            }
        }

        // Leaf node (`len(subtree) == 1`).
        if subtree.op.is_none() {
            expression.push(subtree.token);
            continue;
        }

        // Non-leaf node.
        let operator = subtree.token;

        if is_binary_connectable(operator, tt) {
            // Propagate parities into the two operands.
            let mut prop: [[i64; N_CC]; 2] = [[0, 0], [0, 0]];
            for cc in [CC_ADD, CC_MULT] {
                // `operator_inverses[operator_set0]` for '+' / '*', precomputed per class.
                let flips = tt.cc_inverse_op[cc] == Some(operator);
                let sign = if flips { -1 } else { 1 };
                if still_connected {
                    prop[0][cc] = subtree_parities[cc];
                    prop[1][cc] = subtree_parities[cc] * sign;
                } else {
                    prop[0][cc] = 1;
                    prop[1][cc] = sign;
                }
            }

            // Try to find a cancellation candidate in THIS subtree (only if none yet). No break:
            // the full nest runs, so the LAST qualifying (cc, hash) wins.
            if cancellation_candidate.is_none() {
                for cc in [CC_ADD, CC_MULT] {
                    for (subtree_hash, multiplicity) in &subtree.own[cc] {
                        let abs_sum = multiplicity[0].abs() + multiplicity[1].abs();
                        let has_constant = subtree_hash.iter().any(|&t| t == tt.constant);
                        if abs_sum > 1 && (!has_constant || subtree_hash.len() == 1) {
                            cancellation_candidate =
                                Some((cc, subtree_hash.clone(), multiplicity[0] - multiplicity[1]));
                            still_connected = true;
                        }
                    }
                }
            }

            expression.push(operator);

            // Push children: zip(reversed(operands), reversed(propagated_parities)).
            // operands = [left, right] -> push (right, prop[1]) then (left, prop[0]); pop order
            // restores left-before-right (prefix output).
            stack.push(Frame {
                node: &subtree.operands[1],
                parity: prop[1],
                still_connected,
            });
            stack.push(Frame {
                node: &subtree.operands[0],
                parity: prop[0],
                still_connected,
            });
        } else {
            // General operator: children get reset parity {add:1, mult:1}; still_connected carried.
            expression.push(operator);
            for operand in subtree.operands.iter().rev() {
                stack.push(Frame {
                    node: operand,
                    parity: [1, 1],
                    still_connected,
                });
            }
        }
    }

    expression
}

/// The fused public entry: `cancel_terms(*collect_multiplicities(expression))`. On a malformed
/// expression (`collect_multiplicities` does not collapse to a single root) returns the input
/// unchanged -- the deployed skeleton path only ever feeds well-formed prefix expressions.
pub fn cancel_terms_unit(expression: &[Tok], ops: &Operators, view: &TokenView) -> Vec<Tok> {
    match collect_multiplicities(expression, view) {
        Some(root) => cancel_terms(&root, ops, view),
        None => expression.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use crate::Engine;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    fn toks(s: &[&str]) -> Vec<String> {
        s.iter().map(|t| t.to_string()).collect()
    }

    /// Canonical cancellation cases, cross-checked against fresh Python (see benchmarks/diff_cancel.py
    /// for the full 10k+18k corpus gate). These pin the mechanism: hyper-operator factorization
    /// (`mult{k}`/`pow{k}`), the `neg`/`inv` parity prefix, the `<constant>` special case, and
    /// the additive-vs-multiplicative neutral element.
    #[test]
    fn cancel_canonical_cases() {
        let Some(e) = engine() else { return };
        let cases: &[(&[&str], &[&str])] = &[
            (&["+", "x1", "x1"], &["+", "mult2", "x1", "0"]),
            (&["-", "x1", "x1"], &["-", "0", "0"]),
            (&["*", "x1", "x1"], &["*", "pow2", "x1", "1"]),
            (&["+", "x1", "x2"], &["+", "x1", "x2"]),
            (
                &["+", "+", "x1", "x1", "x1"],
                &["+", "+", "mult3", "x1", "0", "0"],
            ),
            (
                &["-", "x1", "+", "x1", "x1"],
                &["-", "neg", "x1", "+", "0", "0"],
            ),
            (
                &["+", "<constant>", "<constant>"],
                &["+", "<constant>", "0"],
            ),
            // 6 occurrences -> factorize(6,5) = [2,3] -> mult2 mult3.
            (
                &["+", "+", "+", "+", "+", "x1", "x1", "x1", "x1", "x1", "x1"],
                &[
                    "+", "+", "+", "+", "+", "mult2", "mult3", "x1", "0", "0", "0", "0", "0",
                ],
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                e.cancel_terms(&toks(input)),
                toks(expected),
                "input {input:?}"
            );
        }
    }

    /// Group-axiom gate: nan/inf leaves (and zero in the multiplicative class) must NOT cancel;
    /// variables and nonzero-finite literals must keep cancelling exactly as before.
    #[test]
    fn cancel_r5_group_carve_out() {
        let Some(e) = engine() else { return };
        let untouched: &[&[&str]] = &[
            &["/", "0", "0"],                           // was -> 1; truth nan
            &["/", "float(\"nan\")", "float(\"nan\")"], // was -> 1; truth nan
            &["/", "float(\"inf\")", "float(\"inf\")"], // was -> 1; truth nan
            &["-", "float(\"inf\")", "float(\"inf\")"], // was -> 0; truth nan
            &["-", "float(\"nan\")", "float(\"nan\")"], // was -> 0; truth nan
            &["*", "0", "0"],                           // 0 forbidden in the mult class
        ];
        for input in untouched {
            assert_eq!(e.cancel_terms(&toks(input)), toks(input), "input {input:?}");
        }
        let still_cancel: &[(&[&str], &[&str])] = &[
            (&["-", "x1", "x1"], &["-", "0", "0"]), // variable: total identity
            (&["/", "x1", "x1"], &["/", "1", "1"]), // variable: fills the null hole
            (&["-", "0", "0"], &["-", "0", "0"]),   // additive 0 is a group element
            (&["/", "2", "2"], &["/", "1", "1"]),   // nonzero finite literal
        ];
        for (input, expected) in still_cancel {
            assert_eq!(
                e.cancel_terms(&toks(input)),
                toks(expected),
                "input {input:?}"
            );
        }
    }
}
