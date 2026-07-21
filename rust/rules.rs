//! Rule storage, compilation, matching + the (length, root_operator) bucket index with the
//! first-operand-symbol FILTER.
//!
//! Mirrors `compile_rules` (engine.py@0.2.15:189), `construct_rule_patterns` (engine.py@0.2.15:936),
//! `apply_rules_top_down` (engine.py@0.2.15:1025), the pattern matcher `match_pattern_with_cert`
//! (port of utils.py@0.2.15:800), `apply_mapping` (utils.py@0.2.15:762).
//!
//! `deduplicate_rules` is a VERIFIED NO-OP on the dev_7-3 rules.json (114k -> 114k, identical
//! order + tokens), so the rules are consumed directly; no remap/dedup port is needed. The
//! first-operand index (ported from the Python implementation) is OUTPUT-IDENTICAL to the linear scan:
//! it filters provable fast-fails while preserving first-match-wins (asset-order merge of the
//! concrete-head subset and the wildcard residual).
//!
//! Rules compile to interned [`Tok`] sequences/trees. Every rule token is
//! interned into the per-Engine [`TokenTable`] at compile, so rule keys/heads are always TABLE
//! ids -- a query token that only exists in a per-call overlay can, by injectivity, never equal
//! any rule token (the lookups fail exactly as the string comparisons did).

use rustc_hash::FxHashMap;

use crate::operators::Operators;
use crate::parse::{parse_subtree, Node};
use crate::tokens::{Tok, TokenTable};

/// A compiled rule: prefix lhs/rhs token ids plus their pre-parsed trees. The trees are built once
/// at compile time via `parse_subtree` (faithful to `construct_rule_patterns` calling
/// `prefix_to_tree`; for dev_7-3 the two parsers agree -- no arity-0 operators, no aliases in the
/// rules).
#[derive(Debug, Clone)]
pub struct Rule {
    pub lhs: Vec<Tok>,
    #[allow(dead_code)] // read by the stage-(a) parity tests only (crate-internal since the
    // `Engine::rules()` accessor became pub(crate))
    pub rhs: Vec<Tok>,
    pub lhs_tree: Node,
    pub rhs_tree: Node,
}

/// The first-operand-head index for ONE (pattern_length, root_operator) bucket. The
/// values are POSITIONS into the bucket's `Vec<Rule>` (no rule/tree duplication). `by_head[H]` is the
/// asset-ORDER-preserving merge of the concrete-operand[0]-head==H positions and the wildcard
/// residual positions; `wild_only` is the residual alone (for query heads absent from `by_head`).
#[derive(Debug, Default, Clone)]
pub struct OperandIndex {
    pub by_head: FxHashMap<Tok, Vec<usize>>,
    pub wild_only: Vec<usize>,
}

/// Compiled rule set: explicit (no-wildcard) rules in an exact-match map, pattern rules bucketed by
/// (pattern_length, root_operator) in asset (rules.json) order, and the per-bucket operand index.
#[derive(Debug, Default)]
pub struct CompiledRules {
    /// `simplification_rules_no_patterns`: exact lhs token-seq -> rhs.
    pub exact_rules: FxHashMap<Vec<Tok>, Vec<Tok>>,
    /// `simplification_rules_patterns`: (len, root_op) -> ordered rules (asset order preserved).
    pub patterns: FxHashMap<(usize, Tok), Vec<Rule>>,
    /// Per-bucket first-operand-head index (positions into the bucket). Output-identical to the scan.
    pub operand_index: FxHashMap<(usize, Tok), OperandIndex>,
    pub max_pattern_length: usize,
}

/// Mirror of `_WILDCARD_RE = re.compile(r'^[_?]\d+$')` (utils.py): a slot token is a SORT SIGIL
/// followed by one or more ASCII digits. Two sorts: `_N` binds an arbitrary SUBTREE (the
/// pointwise-certified sort); `?N` binds a
/// VARIABLE LEAF only (the sort the miner's certification actually establishes). The sort rides
/// in the token, so a sorted rule set is just a rules.json with different spellings -- no schema
/// change anywhere. (String form; the per-id `is_wildcard` property is computed from this at intern.)
#[inline]
pub fn is_wildcard(token: &str) -> bool {
    let b = token.as_bytes();
    b.len() >= 2
        && (b[0] == b'_' || b[0] == b'?' || b[0] == b'!')
        && b[1..].iter().all(u8::is_ascii_digit)
}

/// Classify a pattern rule by the head of its operand[0]. In flat prefix the root operator is
/// `lhs[0]` and operand[0]'s root token is `lhs[1]`. `Some(head)` = concrete (only matches a query
/// whose operand[0] head == head); `None` = wildcard/degenerate (tried for every query head).
#[inline]
pub fn operand0_head(lhs: &[Tok], table: &TokenTable) -> Option<Tok> {
    if lhs.len() < 2 {
        return None;
    }
    let h = lhs[1];
    if table.is_wildcard_tok(h) {
        None
    } else {
        Some(h)
    }
}

fn build_operand_index(bucket: &[Rule], table: &TokenTable) -> OperandIndex {
    let mut wild_only: Vec<usize> = Vec::new();
    let mut concrete: FxHashMap<Tok, Vec<usize>> = FxHashMap::default();
    for (pos, r) in bucket.iter().enumerate() {
        match operand0_head(&r.lhs, table) {
            Some(h) => concrete.entry(h).or_default().push(pos),
            None => wild_only.push(pos),
        }
    }
    let mut by_head: FxHashMap<Tok, Vec<usize>> = FxHashMap::default();
    for (h, cpos) in concrete {
        // Asset-order merge of {concrete-h} U {wildcard residual}. Positions within a bucket are
        // distinct, so ascending sort reproduces the original bucket subsequence exactly.
        let mut merged: Vec<usize> = cpos.into_iter().chain(wild_only.iter().copied()).collect();
        merged.sort_unstable();
        by_head.insert(h, merged);
    }
    OperandIndex { by_head, wild_only }
}

impl CompiledRules {
    /// Faithful port of `compile_rules` + `construct_rule_patterns` (engine.py@0.2.15:189/936). Consumes the
    /// raw (lhs, rhs) prefix pairs from rules.json (dedup is a verified no-op). A rule is a pattern
    /// iff any lhs token matches `^_\d+$`; pattern rules bucket by (len(lhs), lhs[0]) in asset order
    /// (Python's group-by-op + stable-sort-by-len nets to the same per-bucket order). Each pattern
    /// rule's lhs/rhs is pre-parsed to a tree over interned ids; every rule token is interned into
    /// the (per-Engine, `&mut` here) token table.
    pub fn compile(
        raw: Vec<(Vec<String>, Vec<String>)>,
        table: &mut TokenTable,
        ops: &Operators,
    ) -> Self {
        let mut exact_rules: FxHashMap<Vec<Tok>, Vec<Tok>> = FxHashMap::default();
        let mut patterns: FxHashMap<(usize, Tok), Vec<Rule>> = FxHashMap::default();
        let mut max_pattern_length = 0usize;
        for (lhs, rhs) in raw {
            let lhs_t: Vec<Tok> = lhs.iter().map(|s| table.intern(s, ops)).collect();
            let rhs_t: Vec<Tok> = rhs.iter().map(|s| table.intern(s, ops)).collect();
            if lhs.iter().any(|t| is_wildcard(t)) {
                let plen = lhs_t.len();
                if plen > max_pattern_length {
                    max_pattern_length = plen;
                }
                let key = (plen, lhs_t[0]);
                let arity_of = |t: Tok| table.arity(t);
                let (lhs_tree, _) = parse_subtree(&lhs_t, 0, &arity_of);
                let (rhs_tree, _) = parse_subtree(&rhs_t, 0, &arity_of);
                patterns.entry(key).or_default().push(Rule {
                    lhs: lhs_t,
                    rhs: rhs_t,
                    lhs_tree,
                    rhs_tree,
                });
            } else {
                exact_rules.insert(lhs_t, rhs_t);
            }
        }
        let operand_index = patterns
            .iter()
            .map(|(k, bucket)| (*k, build_operand_index(bucket, table)))
            .collect();
        Self {
            exact_rules,
            patterns,
            operand_index,
            max_pattern_length,
        }
    }

    /// Candidate rules for a node at (pattern_length, operator) whose operand[0] head is `head`, in
    /// asset order, via the operand[0] index. Behaviour-identical subsequence of the bucket; mirrors
    /// the Python `_candidate_rules` (by_head.get(head) else the wildcard residual).
    pub fn candidates<'a>(
        &'a self,
        plen: usize,
        op: Tok,
        head: Tok,
    ) -> impl Iterator<Item = &'a Rule> + 'a {
        let key = (plen, op);
        let bucket = self.patterns.get(&key);
        let positions: &'a [usize] = match self.operand_index.get(&key) {
            Some(idx) => idx
                .by_head
                .get(&head)
                .map(Vec::as_slice)
                .unwrap_or(idx.wild_only.as_slice()),
            None => &[],
        };
        positions
            .iter()
            .filter_map(move |&p| bucket.and_then(|b| b.get(p)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use std::collections::HashMap;
    use std::fs;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    #[derive(serde::Deserialize)]
    struct GroundTruth {
        n_pattern_rules: usize,
        n_no_pattern_rules: usize,
        n_buckets: usize,
        buckets: HashMap<String, Vec<(Vec<String>, Vec<String>)>>,
    }

    fn ground_truth() -> GroundTruth {
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/corpus/_py_rules_groundtruth.json"
        );
        serde_json::from_str(&fs::read_to_string(p).expect("ground truth present")).unwrap()
    }

    /// Stage (a) gate: Rust buckets byte-identical to Python's compiled buckets -- same keys, same
    /// contents, same WITHIN-BUCKET ORDER (the first-match-wins-critical property).
    #[test]
    fn rust_buckets_match_python_ground_truth() {
        // Skip when the (not-vendored) Python ground-truth fixture is absent (see the engine.rs note).
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/corpus/_py_rules_groundtruth.json"
        );
        if !std::path::Path::new(p).exists() {
            eprintln!("rust_buckets_match_python_ground_truth: SKIPPED (ground-truth fixture not vendored)");
            return;
        }
        let Some(eng) = engine() else { return };
        let c = eng.rules();
        let tt = eng.token_table();
        let gt = ground_truth();
        let resolve = |toks: &[Tok]| -> Vec<String> {
            toks.iter().map(|&t| tt.resolve(t).to_string()).collect()
        };

        let rust_pattern_rules: usize = c.patterns.values().map(Vec::len).sum();
        assert_eq!(rust_pattern_rules, gt.n_pattern_rules, "pattern-rule count");
        assert_eq!(
            c.exact_rules.len(),
            gt.n_no_pattern_rules,
            "no_pattern count"
        );
        assert_eq!(c.patterns.len(), gt.n_buckets, "bucket count");

        for (key, gt_rules) in &gt.buckets {
            let (lstr, op) = key.split_once(',').unwrap();
            let plen: usize = lstr.parse().unwrap();
            let op_tok = tt.lookup(op).unwrap_or_else(|| panic!("op {op} interned"));
            let rust_bucket = c
                .patterns
                .get(&(plen, op_tok))
                .unwrap_or_else(|| panic!("rust missing bucket {key}"));
            assert_eq!(rust_bucket.len(), gt_rules.len(), "bucket {key} size");
            for (i, (gl, gr)) in gt_rules.iter().enumerate() {
                assert_eq!(
                    &resolve(&rust_bucket[i].lhs),
                    gl,
                    "bucket {key} idx {i} lhs"
                );
                assert_eq!(
                    &resolve(&rust_bucket[i].rhs),
                    gr,
                    "bucket {key} idx {i} rhs"
                );
            }
        }
        for key in c.patterns.keys() {
            let k = format!("{},{}", key.0, tt.resolve(key.1));
            assert!(gt.buckets.contains_key(&k), "rust has extra bucket {k}");
        }
    }

    /// Stage (a) gate: the exhaustive (bucket, head) operand-index invariant on the RUST index --
    /// candidates() == the asset-ordered bucket filtered to {wildcard} U {concrete head == query}.
    #[test]
    fn operand_index_invariant_holds() {
        let Some(eng) = engine() else { return };
        let c = eng.rules();
        let tt = eng.token_table();
        // An id no table token has: by injectivity it can never equal a concrete head.
        let absent: Tok = Tok(u32::MAX);
        let mut checks = 0usize;
        for (key, bucket) in &c.patterns {
            let (plen, op) = (key.0, key.1);
            let mut heads: Vec<Tok> = bucket
                .iter()
                .filter_map(|r| operand0_head(&r.lhs, tt))
                .collect();
            heads.sort();
            heads.dedup();
            for qh in heads.iter().copied().chain(std::iter::once(absent)) {
                let expected: Vec<&Rule> = bucket
                    .iter()
                    .filter(|r| match operand0_head(&r.lhs, tt) {
                        None => true,
                        Some(h) => h == qh,
                    })
                    .collect();
                let actual: Vec<&Rule> = c.candidates(plen, op, qh).collect();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "bucket {key:?} head {qh:?} len"
                );
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert_eq!(a.lhs, e.lhs, "bucket {key:?} head {qh:?} order lhs");
                    assert_eq!(a.rhs, e.rhs, "bucket {key:?} head {qh:?} order rhs");
                }
                checks += 1;
            }
        }
        assert!(checks > 0);
        eprintln!("operand_index invariant: {checks} (bucket,head) checks passed");
    }
}
