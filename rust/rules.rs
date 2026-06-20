//! Rule storage, compilation, matching + the (length, root_operator) bucket index with the
//! first-operand-symbol FILTER.
//!
//! Mirrors `compile_rules` (engine.py:189), `construct_rule_patterns` (engine.py:936),
//! `apply_rules_top_down` (engine.py:1025), `match_pattern` (utils.py:800), `apply_mapping`
//! (utils.py:762).
//!
//! `deduplicate_rules` is a VERIFIED NO-OP on the dev_7-3 rules.json (114k -> 114k, identical
//! order + tokens), so the rules are consumed directly; no remap/dedup port is needed. The
//! first-operand index (ported from the Python P0 work) is OUTPUT-IDENTICAL to the linear scan:
//! it filters provable fast-fails while preserving first-match-wins (asset-order merge of the
//! concrete-head subset and the wildcard residual).

use rustc_hash::FxHashMap;

use crate::parse::{parse_subtree, Node};

/// A compiled rule: prefix lhs/rhs tokens plus their pre-parsed trees. The trees are built once at
/// compile time via `parse_subtree` (faithful to `construct_rule_patterns` calling `prefix_to_tree`;
/// for dev_7-3 the two parsers agree -- no arity-0 operators, no aliases in the rules).
#[derive(Debug, Clone)]
pub struct Rule {
    pub lhs: Vec<String>,
    pub rhs: Vec<String>,
    pub lhs_tree: Node,
    pub rhs_tree: Node,
}

/// The first-operand-head index for ONE (pattern_length, root_operator) bucket (the P0 index). The
/// values are POSITIONS into the bucket's `Vec<Rule>` (no rule/tree duplication). `by_head[H]` is the
/// asset-ORDER-preserving merge of the concrete-operand[0]-head==H positions and the wildcard
/// residual positions; `wild_only` is the residual alone (for query heads absent from `by_head`).
#[derive(Debug, Default, Clone)]
pub struct OperandIndex {
    pub by_head: FxHashMap<String, Vec<usize>>,
    pub wild_only: Vec<usize>,
}

/// Compiled rule set: explicit (no-wildcard) rules in an exact-match map, pattern rules bucketed by
/// (pattern_length, root_operator) in asset (rules.json) order, and the per-bucket operand index.
#[derive(Debug, Default)]
pub struct CompiledRules {
    /// `simplification_rules_no_patterns`: exact lhs token-seq -> rhs.
    pub no_patterns: FxHashMap<Vec<String>, Vec<String>>,
    /// `simplification_rules_patterns`: (len, root_op) -> ordered rules (asset order preserved).
    pub patterns: FxHashMap<(usize, String), Vec<Rule>>,
    /// Per-bucket first-operand-head index (positions into the bucket). Output-identical to the scan.
    pub operand_index: FxHashMap<(usize, String), OperandIndex>,
    pub max_pattern_length: usize,
}

/// Mirror of `_WILDCARD_RE = re.compile(r'^_\d+$')` (utils.py:935): a metavariable token is `_`
/// followed by one or more ASCII digits.
#[inline]
pub fn is_wildcard(token: &str) -> bool {
    let b = token.as_bytes();
    b.len() >= 2 && b[0] == b'_' && b[1..].iter().all(u8::is_ascii_digit)
}

/// Classify a pattern rule by the head of its operand[0]. In flat prefix the root operator is
/// `lhs[0]` and operand[0]'s root token is `lhs[1]`. `Some(head)` = concrete (only matches a query
/// whose operand[0] head == head); `None` = wildcard/degenerate (tried for every query head).
#[inline]
pub fn operand0_head(lhs: &[String]) -> Option<&str> {
    if lhs.len() < 2 {
        return None;
    }
    let h = &lhs[1];
    if is_wildcard(h) {
        None
    } else {
        Some(h.as_str())
    }
}

fn build_operand_index(bucket: &[Rule]) -> OperandIndex {
    let mut wild_only: Vec<usize> = Vec::new();
    let mut concrete: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    for (pos, r) in bucket.iter().enumerate() {
        match operand0_head(&r.lhs) {
            Some(h) => concrete.entry(h.to_string()).or_default().push(pos),
            None => wild_only.push(pos),
        }
    }
    let mut by_head: FxHashMap<String, Vec<usize>> = FxHashMap::default();
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
    /// Faithful port of `compile_rules` + `construct_rule_patterns` (engine.py:189/936). Consumes the
    /// raw (lhs, rhs) prefix pairs from rules.json (dedup is a verified no-op). A rule is a pattern
    /// iff any lhs token matches `^_\d+$`; pattern rules bucket by (len(lhs), lhs[0]) in asset order
    /// (Python's group-by-op + stable-sort-by-len nets to the same per-bucket order). Each pattern
    /// rule's lhs/rhs is pre-parsed to a tree via `arity_of`.
    pub fn compile(
        raw: Vec<(Vec<String>, Vec<String>)>,
        arity_of: &dyn Fn(&str) -> Option<u8>,
    ) -> Self {
        let mut no_patterns: FxHashMap<Vec<String>, Vec<String>> = FxHashMap::default();
        let mut patterns: FxHashMap<(usize, String), Vec<Rule>> = FxHashMap::default();
        let mut max_pattern_length = 0usize;
        for (lhs, rhs) in raw {
            if lhs.iter().any(|t| is_wildcard(t)) {
                let plen = lhs.len();
                if plen > max_pattern_length {
                    max_pattern_length = plen;
                }
                let key = (plen, lhs[0].clone());
                let (lhs_tree, _) = parse_subtree(&lhs, 0, arity_of);
                let (rhs_tree, _) = parse_subtree(&rhs, 0, arity_of);
                patterns.entry(key).or_default().push(Rule {
                    lhs,
                    rhs,
                    lhs_tree,
                    rhs_tree,
                });
            } else {
                no_patterns.insert(lhs, rhs);
            }
        }
        let operand_index = patterns
            .iter()
            .map(|(k, bucket)| (k.clone(), build_operand_index(bucket)))
            .collect();
        Self {
            no_patterns,
            patterns,
            operand_index,
            max_pattern_length,
        }
    }

    /// The pattern bucket for a node's (pattern_length, operator), if any.
    #[inline]
    pub fn bucket(&self, plen: usize, op: &str) -> Option<&Vec<Rule>> {
        self.patterns.get(&(plen, op.to_string()))
    }

    /// Candidate rules for a node at (pattern_length, operator) whose operand[0] head is `head`, in
    /// asset order, via the operand[0] index. Behaviour-identical subsequence of the bucket; mirrors
    /// the Python `_candidate_rules` (by_head.get(head) else the wildcard residual).
    pub fn candidates<'a>(
        &'a self,
        plen: usize,
        op: &str,
        head: &str,
    ) -> impl Iterator<Item = &'a Rule> + 'a {
        let key = (plen, op.to_string());
        let bucket = self.patterns.get(&key);
        let positions: &'a [usize] = match self.operand_index.get(&key) {
            Some(idx) => idx
                .by_head
                .get(head)
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

    fn cfg_path() -> String {
        format!(
            "{}/.cache/simplipy/engines/dev_7-3/config.yaml",
            std::env::var("HOME").unwrap()
        )
    }
    fn rules_path() -> String {
        format!(
            "{}/.cache/simplipy/engines/dev_7-3/rules.json",
            std::env::var("HOME").unwrap()
        )
    }

    fn engine() -> Engine {
        Engine::from_paths(&cfg_path(), &rules_path()).expect("engine loads")
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
        let eng = engine();
        let c = eng.rules();
        let gt = ground_truth();

        let rust_pattern_rules: usize = c.patterns.values().map(Vec::len).sum();
        assert_eq!(rust_pattern_rules, gt.n_pattern_rules, "pattern-rule count");
        assert_eq!(
            c.no_patterns.len(),
            gt.n_no_pattern_rules,
            "no_pattern count"
        );
        assert_eq!(c.patterns.len(), gt.n_buckets, "bucket count");

        for (key, gt_rules) in &gt.buckets {
            let (lstr, op) = key.split_once(',').unwrap();
            let plen: usize = lstr.parse().unwrap();
            let rust_bucket = c
                .patterns
                .get(&(plen, op.to_string()))
                .unwrap_or_else(|| panic!("rust missing bucket {key}"));
            assert_eq!(rust_bucket.len(), gt_rules.len(), "bucket {key} size");
            for (i, (gl, gr)) in gt_rules.iter().enumerate() {
                assert_eq!(&rust_bucket[i].lhs, gl, "bucket {key} idx {i} lhs");
                assert_eq!(&rust_bucket[i].rhs, gr, "bucket {key} idx {i} rhs");
            }
        }
        for key in c.patterns.keys() {
            let k = format!("{},{}", key.0, key.1);
            assert!(gt.buckets.contains_key(&k), "rust has extra bucket {k}");
        }
    }

    /// Stage (a) gate: the exhaustive (bucket, head) operand-index invariant on the RUST index --
    /// candidates() == the asset-ordered bucket filtered to {wildcard} U {concrete head == query}.
    #[test]
    fn operand_index_invariant_holds() {
        let eng = engine();
        let c = eng.rules();
        const ABSENT: &str = "\0__absent_head__\0";
        let mut checks = 0usize;
        for (key, bucket) in &c.patterns {
            let (plen, op) = (key.0, key.1.as_str());
            let mut heads: Vec<&str> = bucket
                .iter()
                .filter_map(|r| operand0_head(&r.lhs))
                .collect();
            heads.sort();
            heads.dedup();
            for qh in heads.iter().copied().chain(std::iter::once(ABSENT)) {
                let expected: Vec<&Rule> = bucket
                    .iter()
                    .filter(|r| match operand0_head(&r.lhs) {
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
