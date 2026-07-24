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
        let r = e.simplify(lhs, 5, None, true, false);
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

/// Whole-unit gate over the 400-skeleton corpus slice, asserting the search's INVARIANTS
/// rather than a stored answer sheet.
///
/// It used to compare against frozen per-row outputs. That is the wrong shape of test for a
/// bounded search: the outputs are a function of `node_budget`, so the expectations had to be
/// regenerated on every tuning change and would fail confusingly for anyone running a different
/// budget -- reporting "regression" when the engine had merely searched harder. What is actually
/// contractual is asserted here instead, and holds at ANY budget:
///   * no result is longer than its input (the root is the first candidate answer);
///   * the EQUIVALENCE LOOP (mask=false) is idempotent (a second pass cannot beat the first);
///   * it is deterministic (ties break on insertion order);
///   * a larger budget never yields a longer result (more search cannot hurt);
///   * the engine actually does something (most rows change);
///   * the TERMINAL MASK (mask=true) is length-neutral -- it only relabels literals to
///     `<constant>`, never grows or shrinks. Idempotence is asserted on the equivalence loop,
///     NOT on the masked result: masking mints free constants from structural literals
///     (`x-x -> 0 -> <constant>`), so a masked output re-fed to `simplify` can collect
///     redundant constants (~0.4% of rows). That is by design -- masking is a terminal
///     representation step for the downstream model, never re-simplified -- so requiring the
///     masked form to be a search fixpoint would re-introduce the unsound in-loop fold it
///     was split out to avoid.
#[test]
fn simplify_holds_its_invariants_on_the_corpus() {
    let corpus = format!(
        "{}/benchmarks/corpus/raw_skeletons.json",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&corpus).exists() {
        eprintln!("simplify_holds_its_invariants_on_the_corpus: SKIPPED (fixtures not vendored)");
        return;
    }
    let Some(e) = engine() else { return };
    let raw = load("raw_skeletons.json");
    for mpl in [4usize, 7usize] {
        let mut n_changed = 0;
        for s in raw.iter().take(400) {
            // The equivalence loop: the sound, idempotent, deterministic core (no masking).
            let out = e.simplify(s, 48, Some(mpl), true, false);
            assert!(
                out.len() <= s.len(),
                "grew: mpl={mpl} {} -> {} input={s:?}",
                s.len(),
                out.len()
            );
            let again = e.simplify(&out, 48, Some(mpl), true, false);
            assert_eq!(&again, &out, "not idempotent: mpl={mpl} input={s:?}");
            let repeat = e.simplify(s, 48, Some(mpl), true, false);
            assert_eq!(&repeat, &out, "not deterministic: mpl={mpl} input={s:?}");
            let richer = e.simplify(s, 256, Some(mpl), true, false);
            assert!(
                richer.len() <= out.len(),
                "more budget gave a longer result: mpl={mpl} input={s:?}"
            );
            // The separate mask pass is length-neutral (it only relabels literals + sorts).
            let masked = e.mask(&out);
            assert_eq!(
                masked.len(),
                out.len(),
                "mask changed length: mpl={mpl} input={s:?}"
            );
            if &out != s {
                n_changed += 1;
            }
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
        e.simplify(&t(&["*", "<constant>", "acos", "np.e"]), 5, None, true, false),
        nan
    );
    // propagation reaches VARIABLE contexts, which the evidence-based miner cannot:
    assert_eq!(
        e.simplify(&t(&["*", "x0", "acos", "np.e"]), 5, None, true, false),
        nan
    );
    // pow does not propagate structurally (pow(1, NaN) = 1):
    e.set_rules(Vec::new());
    let kept = e.simplify(&t(&["pow", "<constant>", "acos", "np.e"]), 5, None, true, false);
    assert_ne!(kept, nan, "pow(<constant>, nan) must not fold to nan");
    assert_eq!(
        kept,
        t(&["pow", "<constant>", "float(\"nan\")"]),
        "the constant subtree folds; pow itself must not"
    );
}
