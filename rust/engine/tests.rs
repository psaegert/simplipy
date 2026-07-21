use crate::Engine;
use std::fs;

fn engine() -> Option<Engine> {
    crate::test_engine()
}

/// `prune_explicit` must (a) NOT over-prune everything, and (b) every rule it prunes must STILL
/// simplify lhs->rhs via the remaining rules (genuinely redundant).
#[test]
fn prune_explicit_is_correct() {
    let Some(mut e) = engine() else {
        return;
    };
    let sample: Vec<Vec<String>> = e
        .rules()
        .exact_rules
        .keys()
        .take(300)
        .map(|k| {
            k.iter()
                .map(|&t| e.token_table().resolve(t).to_string())
                .collect()
        })
        .collect();
    if sample.is_empty() {
        return;
    }
    let rhs_of: std::collections::HashMap<Vec<String>, Vec<String>> = sample
        .iter()
        .map(|l| {
            let lhs_t = e.token_table().lookup_seq(l).unwrap();
            let rhs_t = e.rules().exact_rules.get(&lhs_t).unwrap();
            let rhs: Vec<String> = rhs_t
                .iter()
                .map(|&t| e.token_table().resolve(t).to_string())
                .collect();
            (l.clone(), rhs)
        })
        .collect();
    let pruned = e.prune_explicit(&sample, false);
    assert!(
        pruned.len() < sample.len(),
        "prune must not remove everything (got {}/{})",
        pruned.len(),
        sample.len()
    );
    for lhs in &pruned {
        let r = e.simplify(lhs, 5, None, false, true);
        assert_eq!(
            &r,
            rhs_of.get(lhs).unwrap(),
            "pruned rule must stay derivable"
        );
    }
}

fn load(name: &str) -> Vec<Vec<String>> {
    let p = format!("{}/benchmarks/corpus/{}", env!("CARGO_MANIFEST_DIR"), name);
    serde_json::from_str(&fs::read_to_string(p).expect("corpus fixture present")).unwrap()
}

/// Whole-unit regression gate: the composed Rust `simplify` fixpoint reproduces the frozen
/// reference outputs on the first 400 corpus skeletons at mpl=4 AND mpl=7. The reference is the
/// 0.7.0 ALIGNED engine line (real pow semantics at a -inf base; single engine line) -- it was
/// regenerated from this engine when the faithful dev_7-3 reproduction line was removed. The
/// corpus is the first 400 skeletons of the 65,536-expression training-prior benchmark. To
/// reproduce the historical dev_7-3 / v23.0-era outputs byte-for-byte, install simplipy<=0.6.0.
#[test]
fn simplify_matches_frozen_reference() {
    // Skip when the frozen corpus is absent -- this in-crate slice only fires where the
    // fixtures are staged (benchmarks/corpus/).
    let corpus = format!(
        "{}/benchmarks/corpus/raw_skeletons.json",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&corpus).exists() {
        eprintln!("simplify_matches_frozen_reference: SKIPPED (corpus fixtures not vendored)");
        return;
    }
    let Some(e) = engine() else { return };
    let raw = load("raw_skeletons.json");
    for mpl in [4usize, 7usize] {
        let reference = load(&format!("reference_aligned_mpl{mpl}.json"));
        assert_eq!(raw.len(), reference.len());
        let mut n_changed = 0;
        for (s, r) in raw.iter().zip(reference.iter()).take(400) {
            let out = e.simplify(s, 5, Some(mpl), true, true);
            if &out != s {
                n_changed += 1;
            }
            assert_eq!(&out, r, "mpl={mpl} input={s:?}");
        }
        assert!(
            n_changed > 300,
            "expected most rows to simplify (got {n_changed}/400)"
        );
    }
}

/// `is_valid` canonical cases. Pins each reject path + the numeric guard.
#[test]
fn is_valid_cases() {
    let Some(e) = engine() else { return };
    let valid: &[&[&str]] = &[
        &["x1"],
        &["<constant>"],
        &["+", "x1", "x2"],
        &["sin", "+", "x1", "x1"],
        &["+", "<constant>", "<constant>"],
        &["+", "0", "1"],
        &["+", "sin", "x1", "neg", "x2"],
    ];
    let invalid: &[&[&str]] = &[
        &[],                      // empty
        &["+"],                   // operator, no operands
        &["sin"],                 // unary, no operand
        &["+", "x1"],             // not enough operands
        &["+", "x1", "x2", "x3"], // leftover stack
        &["x1", "x2"],            // multi-token starting with a leaf
        &["--5"],                 // numeric-looking but float() raises
        &["1e"],                  // ditto
        &["+", "x1", "--5"],      // bad numeric operand
    ];
    for v in valid {
        let t: Vec<String> = v.iter().map(|s| s.to_string()).collect();
        assert!(e.is_valid(&t), "expected valid: {v:?}");
    }
    for v in invalid {
        let t: Vec<String> = v.iter().map(|s| s.to_string()).collect();
        assert!(!e.is_valid(&t), "expected invalid: {v:?}");
    }
}

/// `operators_to_realizations` / `realizations_to_operators`: operator
/// names <-> realizations, non-operator tokens untouched, round-trip on canonical prefix.
#[test]
fn realizations_round_trip() {
    let Some(e) = engine() else { return };
    let t = |s: &[&str]| -> Vec<String> { s.iter().map(|x| x.to_string()).collect() };
    let expr = t(&["*", "neg", "x1", "pow2", "<constant>"]);
    let fwd = e.operators_to_realizations(&expr);
    assert_eq!(
        fwd,
        t(&[
            "*",
            "simplipy.operators.neg",
            "x1",
            "simplipy.operators.pow2",
            "<constant>"
        ])
    );
    assert_eq!(e.realizations_to_operators(&fwd), expr); // round-trips on canonical names
                                                         // non-operator tokens (vars / numerics / <constant>) are passed through unchanged.
    assert_eq!(
        e.operators_to_realizations(&t(&["x1", "0", "<constant>"])),
        t(&["x1", "0", "<constant>"])
    );
}

/// NaN literal propagation in the numeric fold (`C * acos(np.e)` = C * NaN = NaN): once a
/// `nan` literal exists (here from folding the constant subtree acos(np.e)), it propagates
/// through every operator except `pow`, whose C99 edge cases pow(1, NaN) = 1 /
/// pow(NaN, 0) = 1 depend on the other operand and must NOT fold structurally. The pow case
/// runs with an EMPTY rule set: dev_7-3 still carries a defective legacy
/// `pow(<constant>, nan) -> nan` rule (wrong at C = 1) that would shadow the fold's correct
/// refusal; a re-mine retires that rule.
#[test]
fn nan_literal_propagates_in_numeric_fold() {
    let Some(mut e) = crate::test_engine() else {
        return;
    };
    let t = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
    let nan = t(&["float(\"nan\")"]);
    // C * acos(np.e) -> nan (constant-subtree fold + propagation)
    assert_eq!(
        e.simplify(
            &t(&["*", "<constant>", "acos", "np.e"]),
            5,
            None,
            true,
            true
        ),
        nan
    );
    // propagation reaches VARIABLE contexts, which the evidence-based miner cannot:
    assert_eq!(
        e.simplify(&t(&["*", "x0", "acos", "np.e"]), 5, None, true, true),
        nan
    );
    // pow does not propagate structurally (pow(1, NaN) = 1):
    e.set_rules(Vec::new());
    let kept = e.simplify(
        &t(&["pow", "<constant>", "acos", "np.e"]),
        5,
        None,
        true,
        true,
    );
    assert_ne!(kept, nan, "pow(<constant>, nan) must not fold to nan");
    assert_eq!(
        kept,
        t(&["pow", "<constant>", "float(\"nan\")"]),
        "the constant subtree folds; pow itself must not"
    );
}
