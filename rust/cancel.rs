//! Port of the term-cancellation unit: `collect_multiplicities` feeding
//! `cancel_terms`. These always run as a pair (`cancel_terms(*collect_multiplicities(expr))`),
//! so the public entry here is the fused unit [`cancel_terms_unit`]; the tree search enumerates
//! a node's cancellation children through [`cancel_successors`].
//!
//! ## What it does (the actual mechanism, not the docstring)
//! Within a maximal *connected region* of one connection class -- additive (`+`/`-`) or
//! multiplicative (`*`/`/`) -- it counts the SIGNED multiplicity of each LEAF token and, if some
//! leaf's `sum(|pos|,|neg|) > 1`, merges its occurrences: the first occurrence becomes the merged
//! term (`mult{k}`/`pow{k}` hyper-operator, or a `neg`/`inv` inverse prefix, or a `*k`/`pow ... k`
//! coefficient fallback) and every later occurrence becomes the class neutral (`0`/`1`). Only ONE
//! cancellation candidate is taken per call -- the search branches over the choice, and reaching
//! a further cancellation is simply another edge deeper in the graph. Composite
//! subtrees do NOT register as cancellable hashes (only leaf `(token,)` hashes propagate), so the
//! `(a*b)+(a*b)` docstring example is not what the code actually cancels.
//!
//! ## Faithfulness traps (verified against the source)
//! * The annotation dicts are Python dicts iterated in **insertion order** -> replicated with an
//!   insertion-ordered `Vec<(key,[i64;2])>`, NOT a hash map.
//! * Candidate selection does **NOT break**: it runs the full `[add, mult] x dict.items()` nest, so
//!   the candidate is the **last** qualifying match, not the first.
//! * Which candidate to take is NOT decided here. Cancellation is non-confluent -- taking
//!   candidate A can destroy candidate B, and which choice ends shortest depends on what the
//!   rule set can fold afterwards -- so `Engine::simplify_search` BRANCHES over every candidate
//!   (via [`cancel_successors`]) and keeps the shortest node it reaches. It is searched, never
//!   guessed; the `select_nth` argument below just names which branch to emit.
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

// Connection classes, hardcoded (NOT config-derived). Iteration order
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

/// `connection_classes_inverse`: the unary inverse operator of each class.
#[inline]
fn cc_inverse(cc: usize, tt: &TokenTable) -> Tok {
    match cc {
        CC_ADD => tt.neg,
        _ => tt.inv,
    }
}

/// `connection_classes_hyper`: the hyper-operator PREFIX of each class (the
/// emitted token is `format!("{hyper}{k}")`, interned at emission).
const CC_HYPER: [&str; N_CC] = ["mult", "pow"];

/// `token in self.binary_connectable_operators`.
#[inline]
fn is_binary_connectable(token: Tok, tt: &TokenTable) -> bool {
    token == tt.plus || token == tt.minus || token == tt.star || token == tt.slash
}

/// `self.operator_to_class[operator]` for a binary-connectable operator.
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

/// Port of `collect_multiplicities`. Right-to-left scan building a stack
/// of annotated subtrees; for a well-formed prefix expression the stack ends with a single root,
/// which is returned. Mirrors the leaf / binary-connectable / general-operator branches exactly.
fn collect_multiplicities(expression: &[Tok], view: &TokenView, wildcard_all: bool) -> Option<AnnNode> {
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
            let arity = arity as usize;
            let mut operands: Vec<AnnNode> = stack.split_off(stack.len() - arity);
            operands.reverse();
            // The class INVERSE operators (`cc_inverse`) are made region-CONTINUING in THEIR OWN
            // class, symmetric with how `cancel_terms` already EMITS them: `neg` = additive inverse
            // (flip CC_ADD multiplicities); `inv` = multiplicative inverse (flip CC_MULT). Each is
            // OPAQUE in the other class. (`neg` as a cross-class `-1` MULTIPLICATIVE factor is NOT
            // handled here: it needs the `-1` sign preserved when the inner content fully cancels,
            // which the emit does not currently thread -- deferred; any other general operator: empty.)
            let own: [Ann; N_CC] = if token == tt.neg && arity == 1 {
                let mut add: Ann = Vec::new();
                for (h, v) in &operands[0].own[CC_ADD] {
                    let e = ann_entry(&mut add, h);
                    e[0] += v[1];
                    e[1] += v[0];
                }
                [add, Vec::new()]
            } else if token == tt.inv && arity == 1 {
                let mut mult: Ann = Vec::new();
                for (h, v) in &operands[0].own[CC_MULT] {
                    let e = ann_entry(&mut mult, h);
                    e[0] += v[1];
                    e[1] += v[0];
                }
                [Vec::new(), mult]
            } else {
                [Vec::new(), Vec::new()]
            };
            let label = build_label(token, &operands);
            stack.push(AnnNode {
                op: Some(token),
                token,
                operands,
                own,
                label,
            });
            i -= 1;
            continue;
        }

        // Leaf: registers itself with multiplicity [1,0]. In SOUND mode cancellation assumes the
        // group axioms, so a leaf registers ONLY in the connection classes where they hold:
        //   - VARIABLES in both classes (`x - x -> 0` is total; `x/x -> 1` fills a null hole);
        //   - a LITERAL where its value is a group element: nonzero finite in both classes; 0 in
        //     the ADDITIVE class only (`0/0 -> 1` and the sign-of-zero family are wrong answers);
        //     nan/+-inf NOWHERE (`nan/nan -> 1`, `inf - inf -> 0`: truth nan in all four);
        //   - `<constant>` keeps its historical registration.
        // Not registering is the same state a composite subtree has always had (own = empty), so
        // the cancel phase needs no other change; it also UN-SHADOWS the already-mined correct
        // ground rules (`('/','0','0') -> float("nan")` et al.) that this pass pre-empted.
        //
        // LOSSY mode (`wildcard_all`) drops this group-axiom gate, exactly as it drops the rule
        // matcher's `!`-certificate and the constant-fold's finiteness gate: EVERY leaf registers
        // in BOTH classes, so `0/0 -> 1`, `inf/inf -> 1`, `inf - inf -> 0` cancel structurally. Not
        // sound; the training-corpus-canonicalisation contract (data generated FROM the simplified
        // form, target == data) is what makes the three relaxations coherent.
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
        let own = if wildcard_all {
            [reg(true), reg(true)]
        } else {
            [
                reg(lv.map(|v| v.is_finite()).unwrap_or(true)),
                reg(lv.map(|v| v.is_finite() && v != 0.0).unwrap_or(true)),
            ]
        };
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

/// Port of `cancel_terms`, the deployed `collect_statistics=False` path.
/// The second return value is the CHOSEN candidate's sum (`None` when no cancellation fired).
fn cancel_terms(
    root: &AnnNode,
    ops: &Operators,
    view: &TokenView,
    select_nth: Option<usize>,
) -> (Vec<Tok>, Option<i64>, usize) {
    let tt = view.table;
    let mut expression: Vec<Tok> = Vec::new();

    // (argmax_class, cancelled_subtree, cancelled_multiplicity_sum). Set at most once.
    let mut cancellation_candidate: Option<(usize, Vec<Tok>, i64)> = None;
    let mut n_replaced: i64 = 0;
    // Total qualifying (node, cc, hash) triples seen -- the state's branching factor.
    let mut n_candidates: usize = 0;

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
            // neg/inv are region-continuing class inverses (see collect_multiplicities): treat them
            // as in-class so still_connected survives across them (neg additive; inv mult only),
            // mirroring the collect.
            let in_class = connection_ops(argmax_class, tt).contains(&st0)
                || (st0 == tt.neg && argmax_class == CC_ADD)
                || (st0 == tt.inv && argmax_class == CC_MULT);
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

                        // Negative parity and negative multiplicity cancel out.
                        let inverse_operator_prefix: Vec<Tok> =
                            if current_parity * cancelled_multiplicity_sum >= 0 {
                                Vec::new()
                            } else {
                                vec![inverse_operator]
                            };
                        // `double_inverse_operator_prefix` is computed in the
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
            //
            // `select_nth = Some(k)`: enumerate every qualifying (node, cc, hash) triple in walk
            // order and select the k-th -- this is how the tree search names a child (each k is
            // one cancellation edge);
            // `n_candidates` keeps counting past the selection so the caller learns the full
            // branching factor of this state in one pass.
            if select_nth.is_some() || cancellation_candidate.is_none() {
                for cc in [CC_ADD, CC_MULT] {
                    for (subtree_hash, multiplicity) in &subtree.own[cc] {
                        let abs_sum = multiplicity[0].abs() + multiplicity[1].abs();
                        let has_constant = subtree_hash.iter().any(|&t| t == tt.constant);
                        if abs_sum > 1 && (!has_constant || subtree_hash.len() == 1) {
                            let take = match select_nth {
                                Some(k) => n_candidates == k,
                                None => true,
                            };
                            n_candidates += 1;
                            if take {
                                cancellation_candidate = Some((
                                    cc,
                                    subtree_hash.clone(),
                                    multiplicity[0] - multiplicity[1],
                                ));
                                still_connected = true;
                            }
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
        } else if operator == tt.neg && subtree.operands.len() == 1 {
            // neg: additive inverse (flip CC_ADD parity); the multiplicative region resets inside.
            expression.push(operator);
            stack.push(Frame {
                node: &subtree.operands[0],
                parity: [-subtree_parities[CC_ADD], 1],
                still_connected,
            });
        } else if operator == tt.inv && subtree.operands.len() == 1 {
            // inv: multiplicative inverse (flip CC_MULT parity); the additive region resets inside.
            expression.push(operator);
            stack.push(Frame {
                node: &subtree.operands[0],
                parity: [1, -subtree_parities[CC_MULT]],
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

    let chosen_sum = cancellation_candidate.as_ref().map(|(_, _, s)| *s);
    (expression, chosen_sum, n_candidates)
}

/// SEARCH branching operator: every successor of `expression` under the cancel move, each
/// paired with the SIGNED multiplicity sum of the candidate it took (the caller ranks on it).
///
/// The annotation tree is collected ONCE and the per-candidate emit walks reuse it; enumerating
/// via a per-candidate entry point re-collected it `b + 1` times per expansion.
///
/// There is exactly ONE region shape: `neg`/`inv` are ALWAYS region-continuing class inverses.
/// Treating them as opaque was the original defect (the unit emitted the class inverses but
/// never consumed them); the pre-0.8.0 outputs that relied on that asymmetry are not preserved.
pub fn cancel_successors(
    expression: &[Tok],
    ops: &Operators,
    view: &TokenView,
    wildcard_all: bool,
) -> Vec<(Vec<Tok>, i64)> {
    let Some(root) = collect_multiplicities(expression, view, wildcard_all) else {
        return Vec::new();
    };
    // select_nth = usize::MAX selects nothing and just counts the qualifying triples.
    let (_, _, n_candidates) = cancel_terms(&root, ops, view, Some(usize::MAX));
    (0..n_candidates)
        .map(|k| {
            let (out, sum, _) = cancel_terms(&root, ops, view, Some(k));
            (out, sum.unwrap_or(0))
        })
        .collect()
}

/// The fused public entry: `cancel_terms(*collect_multiplicities(expression))` under the
/// engine's own candidate selection (historical last-qualifying-at-the-root-most-node)
/// -- i.e. exactly the step [`crate::Engine::simplify`]'s
/// greedy seed takes, so this validation entry and the kernel cannot drift apart. On a malformed
/// expression (`collect_multiplicities` does not collapse to a single root) the input is
/// returned unchanged; the deployed skeleton path only ever feeds well-formed prefix
/// expressions.
pub fn cancel_terms_unit(expression: &[Tok], ops: &Operators, view: &TokenView) -> Vec<Tok> {
    // The diagnostic unit entry (`cancel_only`) stays SOUND (group-axiom gate on).
    match collect_multiplicities(expression, view, false) {
        Some(root) => cancel_terms(&root, ops, view, None).0,
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

    /// Canonical cancellation cases. These pin the mechanism: hyper-operator factorization
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

    /// neg/inv are region-continuing class inverses (symmetric with `cc_inverse` on EMIT): a leaf
    /// cancels through the unary inverse OF ITS OWN CLASS. `inv` = multiplicative inverse; `neg` =
    /// additive inverse (across a composite too). Cross-class `neg` as a `-1` multiplicative factor
    /// is deliberately NOT handled (sign preservation on full inner cancellation) -- see
    /// `collect_multiplicities` -- so `neg(x)/x` is left to a rule, not cancelled here.
    #[test]
    fn cancel_neg_inv_symmetric() {
        let Some(e) = engine() else { return };
        let cases: &[(&[&str], &[&str])] = &[
            // inv = multiplicative inverse: x / inv(x) = x^2 ; x * inv(x) = 1.
            (&["/", "x0", "inv", "x0"], &["/", "pow2", "x0", "inv", "1"]),
            (&["*", "x0", "inv", "x0"], &["*", "1", "inv", "1"]),
            // neg = additive inverse across a composite: x0 + neg(x0 + x1) -> -x1.
            (
                &["+", "x0", "neg", "+", "x0", "x1"],
                &["+", "0", "neg", "+", "0", "x1"],
            ),
            // NO false cancel: x0 / inv(x0 + x1) keeps x0 (inner x0 is an ADDITIVE sub-leaf, not a
            // multiplicative leaf of the outer region).
            (
                &["/", "x0", "inv", "+", "x0", "x1"],
                &["/", "x0", "inv", "+", "x0", "x1"],
            ),
            // cross-class neg is NOT cancelled here (deferred): neg(x0) / x0 stays for a rule.
            (
                &["/", "neg", "x0", "x0"],
                &["/", "neg", "x0", "x0"],
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

    /// `cancel_terms_unit` (behind the public `cancel_only`) applies ONE cancellation using the
    /// default selection. It is a unit-inspection entry, NOT "what simplify does" -- the tree
    /// search branches over every candidate rather than privileging one, so the two answer
    /// different questions and must not be read as equivalent. This pins the unit's own contract
    /// on a tree with two candidates: `(x1/x1) * inv(x1)` has an inner annihilation (x1:[1,1])
    /// and a root-level x1:[1,2]; the walk selects the root-most.
    #[test]
    fn cancel_unit_selects_the_root_most_candidate() {
        let Some(e) = engine() else { return };
        assert_eq!(
            e.cancel_terms(&toks(&["*", "/", "x1", "x1", "inv", "x1"])),
            toks(&["*", "/", "inv", "x1", "1", "inv", "1"]),
        );
    }
}
