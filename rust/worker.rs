//! The no-constant equivalence test + rule selection for the OFFLINE miner.
//!
//! This is the constant-FREE candidate branch of `find_rule_worker` plus the
//! winner-selection. It adds NO new numerics -- the only math is
//! `allclose_extends` (bit-exact vs numpy) -- so it is a pure control-flow port. Together with the
//! constant-fit branch (`crate::fit`) these are the two halves of the per-candidate decision,
//! assembled over the candidate library + generation + Kruskal prune into the full native miner.

use crate::eval::{allclose_extends, columns_from_row_major, Tape};
use crate::fit::Rng;
use crate::operators::Operators;

/// The non-increasing wildcard-multiplicity condition: a rule `lhs -> rhs` violates it
/// when any wildcard token (`^_\d+$`) occurs MORE times on the rhs than the lhs (it would duplicate a
/// matched subtree). Faithfully matches `^_\d+$`, so on dummy-variable (`x0`..) mining expressions it
/// is inert exactly as in Python -- selection then reduces to "fewest `<constant>` first".
pub fn violates_wildcard_multiplicity(lhs: &[String], rhs: &[String]) -> bool {
    use rustc_hash::FxHashMap;
    let is_wc = |t: &str| -> bool {
        let b = t.as_bytes();
        b.len() >= 2 && b[0] == b'_' && b[1..].iter().all(|c| c.is_ascii_digit())
    };
    let mut lhs_wc: FxHashMap<&str, i32> = FxHashMap::default();
    for t in lhs {
        if is_wc(t) {
            *lhs_wc.entry(t).or_insert(0) += 1;
        }
    }
    let mut rhs_wc: FxHashMap<&str, i32> = FxHashMap::default();
    for t in rhs {
        if is_wc(t) {
            *rhs_wc.entry(t).or_insert(0) += 1;
        }
    }
    rhs_wc
        .iter()
        .any(|(w, &c)| c > lhs_wc.get(w).copied().unwrap_or(0))
}

/// Sign-combination vectors for the source-constant test grid. `n == 0` -> a single empty
/// combo (so a const-free source still runs once).
///
/// The grid is `{-1, +1}^n` (signed-continuous magnitudes), NOT `{-1, 0, 1}^n`. A `0` entry
/// would be a measure-zero ATOM in the test data: it spuriously certifies rules true only at
/// C=0 (`pow(nan,C)->1`, `pow(x,inf)->0` via a const exponent) AND blocks true multiplicative
/// power laws that fail only at the measure-zero point C=0. By the measure-zero principle no
/// special constant value gets weight (neither 0, the additive neutral, nor C=1, the
/// multiplicative one). Sign coverage `{-1,+1}` is kept (continuous, not an atom):
/// `sqrt`/`log`/`pow` need both signs of C tested for domain reasons. Env
/// `SIMPLIPY_ZERO_SIGN=1` restores the old `{-1,0,1}` grid (reference/repro only).
// ARTIFACT-AFFECTING switch: listed in engine.py::ARTIFACT_ENV_SWITCHES (H-042).
fn sign_combos(n: usize) -> Vec<Vec<f64>> {
    static WITH_ZERO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let signs: &[f64] = if *WITH_ZERO.get_or_init(|| std::env::var("SIMPLIPY_ZERO_SIGN").is_ok()) {
        &[-1.0, 0.0, 1.0]
    } else {
        &[-1.0, 1.0]
    };
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        let mut next = Vec::with_capacity(out.len() * signs.len());
        for base in &out {
            for &s in signs {
                let mut v = base.clone();
                v.push(s);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// Magnitudes straddling the +-1 pole thresholds where a unary op crosses 1 (asin at sin 1 ~= 0.84,
/// atanh at tanh 1 ~= 0.76, mult_k at 1/k, ...), plus small and large tails.
///
/// Retirement was tried and REJECTED: the exact reachability gate does NOT replace this grid.
/// The grid's load-bearing content is not the magnitude sweep at all -- it is the three
/// INTEGER entries {5, 30, 500}. `np.power(negative, C)` is defined iff C is an INTEGER, while
/// `pow1_3`/`pow1_5` are sign-preserving real roots (total on R), so `pow1_3(pow(x, C))` is defined
/// on x<0 exactly when C is an integer, and the rhs constant is pinned to C/k by the x>0 branch and
/// cannot repair it (`pow1_4(pow(x,4))` at x=-2: source +2.0, target -2.0, both DEFINED). That
/// failure set is measure-ZERO in C but positive-measure in x, and the source constant is
/// universally quantified -- so NO continuous sampler can find it (`const_magnitude` hits Z with
/// probability 0) and no value-class analysis sees it either.
///
/// Known costs, kept deliberately until the two fixes below land: the grid rejects some true
/// power laws for FITTER-POWER reasons (at C=+-1000 few rows of `inv(x^C)` carry fittable
/// signal), it catches some artifacts by luck rather than by construction, and it MINTS one
/// artifact of its own -- `pow(div5 C, nan) -> <constant>` exists only because div5(5.0) == 1.0
/// exactly and IEEE says pow(1, nan) = 1.
///
/// To retire it, two exact fixes are needed: (1) INTEGRALITY classes in the source-constant
/// challenge set (integers/parity/k-divisibility -- the grid's only causal content, made principled
/// and complete); (2) a BIDIRECTIONAL reachability gate: it currently checks the candidate reaches
/// every value the source reaches, but not that the candidate stays UNDEFINED where the source is.
const POLE_GRID: [f64; 26] = [
    0.02, 0.1, 0.18, 0.22, 0.24, 0.26, 0.3, 0.34, 0.4, 0.45, 0.49, 0.51, 0.6, 0.7, 0.76, 0.84, 0.9,
    0.95, 0.99, 1.01, 1.2, 1.6, 2.5, 5.0, 30.0, 500.0,
];

/// Per-challenge source-constant MAGNITUDE vectors (multiplied by the sign combos downstream, so
/// only |C| is chosen here). The `POLE_GRID` is swept on EACH constant in turn, with the OTHER
/// constants drawn randomly, so a narrow pole band in ANY single constant is exercised
/// DETERMINISTICALLY (nlc==1 reduces to a plain grid sweep). Missing such a band admits
/// step-function artifacts (`pow(asin(C), inf) -> 0`, and its multi-constant scalings
/// `* C1 pow(asin(C2), inf) -> 0`, which random sampling lands in only ~2%/draw). Fully-random
/// rounds are appended for off-grid decorrelation and joint coverage.
// ARTIFACT-AFFECTING switch (SIMPLIPY_POLE_GRID): listed in
// engine.py::ARTIFACT_ENV_SWITCHES (H-042).
fn source_const_magnitudes(rng: &mut Rng, n_const: usize, challenges: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    // A/B kill-switch (`SIMPLIPY_POLE_GRID=0`): random draws only, no grid sweep. Ablation only.
    static GRID_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let grid_on =
        *GRID_ON.get_or_init(|| std::env::var("SIMPLIPY_POLE_GRID").as_deref() != Ok("0"));
    for j in 0..(if grid_on { n_const } else { 0 }) {
        for &m in POLE_GRID.iter() {
            let mut v: Vec<f64> = (0..n_const).map(|_| rng.const_magnitude()).collect();
            v[j] = m;
            out.push(v);
        }
    }
    for _ in 0..challenges {
        out.push((0..n_const).map(|_| rng.const_magnitude()).collect());
    }
    out
}

/// §5 DIRAC REFUTER for a LITERAL target. Refuter ONLY: it can subtract a rule, never license
/// one, so it is sound by construction and unsound only if read as a certificate.
///
/// WHY IT EXISTS. A variable-free `<constant>`-bearing source is quantified over its constants,
/// and §5's proven theorem says a constant slot binds as a DIRAC: one disagreeing constant value
/// carries FULL data measure, so R2's "where the source is DEFINED the target must be defined and
/// exactly equal" applies with ZERO tolerance. The interval classifier answers a MEASURE question
/// ("what is the value class almost everywhere?") and the all-constant short-circuit then
/// republished that answer as an ATTAINABILITY claim ("the source is nan, full stop"). Measured
/// consequences of that republication: `pow (-1) <constant> -> nan` (false at every integer C,
/// where the source is +-1) and the IEEE absorption pairs `pow <constant> acos np.e -> nan`
/// (false at C=1, since pow(1, nan) = 1) and `pow acos np.e <constant> -> nan` (false at C=0,
/// since pow(nan, 0) = 1) -- 22 of which shipped in the queued production artifact.
///
/// A measure abstraction over C subset of R provably cannot express "C is an integer", so this
/// question does not belong to the interval layer at all: it belongs to the EXACT-POINT layer.
/// `battery::SPECIAL_CONSTS` already carries every witness needed (0, +-1, +-2, +-4, +-5, 3,
/// pi/2, pi, e) -- the integer exponents AND both absorption points -- so this adds no new tuned
/// constants, it routes an existing question to the instrument that can decide it.
///
/// SCOPE, deliberately narrow: LITERAL targets only. A `<constant>` target is NOT judged here.
/// Under a strict §5 reading `inv <constant> -> <constant>` would die at the witness C = 0
/// (inv(0) = +inf under the ratified one-zero convention, and +inf is DEFINED), taking the whole
/// ratified `inv`/`log`/`atanh <constant>` family with it. Whether the constant axis inherits
/// §9.8.3's data-null tolerance is an owner ratification, not a decision for this function.
/// The exact mathematical inverse of each LIVE unary operator, BY NAME. Used only to widen the
/// Dirac refuter's witness set; the engine already implements every one of these, so this is a
/// name mapping and not new numerics. The retired hyper-operator ring (`mult3`, `div4`, `pow2`,
/// `pow1_2`, …) is gone from this map with the vocabulary itself: its generation-2 spellings are
/// BINARY nodes with a determined operand, handled tree-aware by `sections` below. The live odd
/// roots `pow1_3`/`pow1_5` are also NOT here: their inverses have no unary spelling in any live
/// vocabulary and are spelled with binary `pow` in `slot_witnesses`.
///
/// WHY: `battery::SPECIAL_CONSTS` holds the algebraic corner points of a value (0, ±1, π/2, e, …),
/// but a source wraps its slot in operators, so the corner point of the WHOLE expression sits at
/// the INVERSE IMAGE of a corner point, not at a corner point itself. `acosh(sin(C/3))` is finite
/// only where sin(C/3) >= 1, i.e. at C = 3(π/2 + 2kπ) — an isolated set no sweep over the corner
/// points, and no dense sampling, will ever hit. Evaluating instead at 3·(π/2) finds it exactly.
/// The dev-era acceptance gate discovered this family (24 rules) precisely by composing inverses,
/// so the refuter needs the same closure to see what the gate sees.
///
/// NOTE the engine's own operator table cannot serve here: it declares no inverse for `abs`,
/// and its `acosh`/`atanh` entries say `cos`/`tan`, which are wrong.
fn inverse_unary_name(op: &str) -> Option<&'static str> {
    Some(match op {
        "neg" => "neg",
        "inv" => "inv",
        "abs" => "abs", // abs(±s) = s: `abs` itself reaches every attainable value
        "sin" => "asin",
        "cos" => "acos",
        "tan" => "atan",
        "asin" => "sin",
        "acos" => "cos",
        "atan" => "tan",
        "sinh" => "asinh",
        "cosh" => "acosh",
        "tanh" => "atanh",
        "asinh" => "sinh",
        "acosh" => "cosh",
        "atanh" => "tanh",
        "exp" => "log",
        "log" => "exp",
        _ => return None,
    })
}

/// End of the prefix subtree starting at `start` (one past its last token). Saturates at
/// `tokens.len()` on malformed input; the caller checks completeness.
fn prefix_subtree_end(tokens: &[String], start: usize, ops: &Operators) -> usize {
    let mut idx = start;
    let mut need = 1usize;
    while need > 0 && idx < tokens.len() {
        need -= 1;
        need += ops.arity_of(&tokens[idx]).unwrap_or(0) as usize;
        idx += 1;
    }
    idx
}

/// A subtree the engine can fold to one value with no free axis: no `<constant>` slot and
/// `evaluate_constant_subtree` succeeds (which excludes variables and wildcards -- their
/// tokens are not resolvable leaves).
fn determined_operand(tokens: &[String], ops: &Operators) -> bool {
    !tokens.iter().any(|t| t == "<constant>")
        && crate::numeric::evaluate_constant_subtree(tokens, ops).is_some()
}

/// One-operand SECTIONS of the source's binary nodes: a binary node `op(a, b)` with exactly one
/// determined operand is the unary map x ↦ op(x, b̂) (or x ↦ op(â, x)) on its free side. The
/// retired hyper-operator ring spelled exactly these sections as unary NAMES (`mult3(x)` for
/// `* x 3`, `pow2(x)` for `pow x 2`); generation 2 spells them as trees, so the witness closure
/// must read them from the tree. Returns `(op, determined-operand tokens, operand-is-first)`.
fn sections(source: &[String], ops: &Operators) -> Vec<(String, Vec<String>, bool)> {
    let mut out = Vec::new();
    for i in 0..source.len() {
        if ops.arity_of(&source[i]) != Some(2) {
            continue;
        }
        let a0 = i + 1;
        let a1 = prefix_subtree_end(source, a0, ops);
        let b1 = prefix_subtree_end(source, a1, ops);
        if b1 > source.len() || a1 > source.len() {
            continue; // malformed tail: not a section
        }
        let (a, b) = (&source[a0..a1], &source[a1..b1]);
        let (da, db) = (determined_operand(a, ops), determined_operand(b, ops));
        // both determined -> the node folds whole (no free side); neither -> no section
        if da != db {
            let (lit, lit_first) = if da {
                (a.to_vec(), true)
            } else {
                (b.to_vec(), false)
            };
            out.push((source[i].clone(), lit, lit_first));
        }
    }
    out
}

/// Push a finite, not-yet-present witness value.
fn push_witness(w: &mut Vec<f64>, v: f64) {
    if v.is_finite() && !w.contains(&v) {
        w.push(v);
    }
}

/// Evaluate an inverse-image expression with the engine's own numerics; `None` = unevaluable,
/// which simply contributes no witness (witnesses are sound by construction: every pushed value
/// is a legitimate constant instantiation, however it was found).
fn eval_witness(expr: Vec<String>, ops: &Operators) -> Option<f64> {
    crate::numeric::evaluate_constant_subtree(&expr, ops)
        .and_then(|t| crate::numeric::leaf_value(&t))
}

/// Witnesses for one `<constant>` slot: the special constants themselves, PLUS the inverse image
/// of each special constant under every unary operator AND every one-operand binary section that
/// appears in the source. One level of inversion is what the dev-era acceptance gate's `atom`
/// tier used and what closes the observed family.
fn slot_witnesses(source: &[String], ops: &Operators) -> Vec<f64> {
    // CORNER VALUES first, then their inverse images. Keeping the two stages separate is what
    // makes the closure reach `acosh(cos(exp(C)))`, whose witness is log(2*pi): 2*pi is a corner
    // of `cos`, and the witness is that corner pulled back through `exp`. Enriching AFTER the
    // inversion cannot produce it.
    // The trig corners are k*pi/2 -- exactly the zeros and extrema of sin and cos, i.e. the
    // operators' own structure. SPECIAL_CONSTS carries only pi/2 and pi, so `cos` never reaches
    // its +1 corners (2k*pi) without this.
    let mut corners: Vec<f64> = crate::battery::SPECIAL_CONSTS.to_vec();
    for k in -5i32..=5 {
        let c = f64::from(k) * std::f64::consts::FRAC_PI_2;
        if !corners.contains(&c) {
            corners.push(c);
        }
    }
    let mut w: Vec<f64> = corners.clone();
    for tok in source {
        // LIVE odd roots: inverse spelled with BINARY pow (`pow3` is a retired name that no
        // live engine evaluates; `pow s 3` is the same function on every engine).
        let odd_power: Option<&str> = match tok.as_str() {
            "pow1_3" => Some("3"),
            "pow1_5" => Some("5"),
            _ => None,
        };
        if let Some(k) = odd_power {
            for &s in corners.iter() {
                let expr = vec![
                    "pow".to_string(),
                    crate::numeric::py_float_repr(s),
                    k.to_string(),
                ];
                if let Some(v) = eval_witness(expr, ops) {
                    push_witness(&mut w, v);
                }
            }
            continue;
        }
        let Some(inv) = inverse_unary_name(tok) else {
            continue;
        };
        for &s in corners.iter() {
            // Evaluate the inverse operator with the engine's own numerics.
            let expr = vec![inv.to_string(), crate::numeric::py_float_repr(s)];
            if let Some(v) = eval_witness(expr, ops) {
                push_witness(&mut w, v);
                // Inverting a PERIODIC operator returns only the principal branch, so add the
                // period shifts too -- again the operator's own structure, not a tuned value.
                let period = match inv {
                    "asin" | "acos" => Some(std::f64::consts::TAU),
                    "atan" => Some(std::f64::consts::PI),
                    _ => None,
                };
                if let Some(p) = period {
                    for k in [1.0f64, -1.0, 2.0] {
                        push_witness(&mut w, v + k * p);
                    }
                }
            }
        }
    }
    // BINARY SECTIONS (the generation-2 spellings of the retired unary ring): pull each corner
    // back through x ↦ op(x, ĉ) / x ↦ op(ĉ, x). The determined operand's tokens are spliced
    // VERBATIM into the inverse expression so the engine evaluates exactly what the source
    // spells (`np.pi`, `neg 2`, folded subtrees -- all resolve natively).
    for (op, lit, lit_first) in sections(source, ops) {
        for &s in corners.iter() {
            let sr = crate::numeric::py_float_repr(s);
            let mut exprs: Vec<Vec<String>> = Vec::new();
            let prefix = |head: &[&str], tail_lit: bool| -> Vec<String> {
                // tail_lit: [head..., lit...] ; else [head[0], lit..., head[1..]...]
                let mut e: Vec<String> = Vec::new();
                if tail_lit {
                    e.extend(head.iter().map(|t| t.to_string()));
                    e.extend(lit.iter().cloned());
                } else {
                    e.push(head[0].to_string());
                    e.extend(lit.iter().cloned());
                    e.extend(head[1..].iter().map(|t| t.to_string()));
                }
                e
            };
            match (op.as_str(), lit_first) {
                // c*x and x*c: preimage s/c
                ("*", _) => exprs.push(prefix(&["/", &sr], true)),
                // x/c: preimage s*c;  c/x: preimage c/s
                ("/", false) => exprs.push(prefix(&["*", &sr], true)),
                ("/", true) => exprs.push(prefix(&["/", &sr], false)),
                // c+x and x+c: preimage s-c
                ("+", _) => exprs.push(prefix(&["-", &sr], true)),
                // x-c: preimage s+c;  c-x: preimage c-s
                ("-", false) => exprs.push(prefix(&["+", &sr], true)),
                ("-", true) => exprs.push(prefix(&["-", &sr], false)),
                // pow(x, c): integer c -> rootn(s, c); non-integer c -> pow(s, 1/c)
                // (whichever arm applies evaluates; the other folds to nan and is dropped)
                ("pow", false) => {
                    exprs.push(prefix(&["rootn", &sr], true));
                    exprs.push(prefix(&["pow", &sr, "/", "1"], true));
                }
                // pow(c, x): preimage log_c(s) = log(s)/log(c)
                ("pow", true) => exprs.push(prefix(&["/", "log", &sr, "log"], true)),
                // rootn(x, c): preimage pow(s, c). (rootn(c, x) needs only INTEGER witnesses
                // -- the index is nan off the integers -- and SPECIAL_CONSTS carries those.)
                ("rootn", false) => exprs.push(prefix(&["pow", &sr], true)),
                _ => {}
            }
            for expr in exprs {
                if let Some(v) = eval_witness(expr, ops) {
                    push_witness(&mut w, v);
                    // The even-power/even-root twin preimage: pow(-v, 2k) = pow(v, 2k). For the
                    // other sections -v is merely one more sound probe value.
                    push_witness(&mut w, -v);
                }
            }
        }
    }
    w
}

fn dirac_refutes_literal(source: &[String], target: &[String], ops: &Operators) -> bool {
    if target.len() != 1 {
        return false;
    }
    let Some(want) = crate::numeric::leaf_value(&target[0]) else {
        return false; // target is not a literal (e.g. `<constant>`): out of scope, see above
    };
    let slots: usize = source.iter().filter(|t| t.as_str() == "<constant>").count();
    // Every `<constant>` occurrence is its own fitted parameter, so the witness set is a product.
    // Capped at 2 slots: 18^2 = 324 folds is already generous for a mint-time refuter.
    if slots == 0 || slots > 2 {
        return false;
    }
    // One slot: the closed witness set (specials + inverse images). Two slots: the closure would
    // be quadratic, so the cross product stays on the plain specials and the closure is applied to
    // the first slot only -- the observed family is single-slot.
    let closed = slot_witnesses(source, ops);
    let w: &[f64] = if slots == 1 {
        &closed
    } else {
        &crate::battery::SPECIAL_CONSTS
    };
    let sp = &crate::battery::SPECIAL_CONSTS;
    let total = if slots == 1 {
        w.len()
    } else {
        w.len() * sp.len()
    };
    for idx in 0..total {
        let (a, b) = if slots == 1 {
            (w[idx], 0.0)
        } else {
            (w[idx % w.len()], sp[idx / w.len()])
        };
        let mut seen = 0usize;
        let mut toks: Vec<String> = Vec::with_capacity(source.len());
        for t in source {
            if t == "<constant>" {
                toks.push(crate::numeric::py_float_repr(if seen == 0 { a } else { b }));
                seen += 1;
            } else {
                toks.push(t.clone());
            }
        }
        // Fold with the engine's own numeric semantics; `None` = unfoldable, which is no evidence.
        let Some(tok) = crate::numeric::evaluate_constant_subtree(&toks, ops) else {
            continue;
        };
        let Some(got) = crate::numeric::leaf_value(&tok) else {
            continue;
        };
        // Extended equality: nan matches nan, an infinity must match SIGN included (R2 counts
        // +-inf as DEFINED), and finite values must be exactly equal.
        let agrees = if got.is_nan() || want.is_nan() {
            got.is_nan() && want.is_nan()
        } else {
            got == want
        };
        if !agrees {
            return true; // one Dirac witness disagrees => REFUTED
        }
    }
    false
}

/// DIAGNOSTIC: `SIMPLIPY_GATE_TRACE="<source tokens>"` prints every gate rejection for that
/// source (the instance's source constants, the candidate + its fitted constants, the witness
/// width). Debug only -- unset in any real mine.
fn gate_trace() -> Option<&'static String> {
    static T: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    T.get_or_init(|| std::env::var("SIMPLIPY_GATE_TRACE").ok())
        .as_ref()
}

/// A/B kill-switch for the interval domain-preservation gate (`SIMPLIPY_IVL_GATE=0` disables it).
/// Ablation/repro only -- the gate is ON by default and is a soundness requirement, not a knob.
fn ivl_gate_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_IVL_GATE").as_deref() != Ok("0"))
}

/// A/B kill-switch for the exact interval classifier in the all-constant short-circuit
/// (`SIMPLIPY_IVL_CLASS=0` disables it, leaving const-bearing sources to the candidate scan).
fn ivl_class_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_IVL_CLASS").as_deref() != Ok("0"))
}

/// A/B kill-switch for the interval REACHABILITY gate (`SIMPLIPY_IVL_REACH=0` disables it).
/// Independent of `SIMPLIPY_IVL_CLASS` so the grid-retirement factorial can vary one at a time.
fn ivl_reach_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_IVL_REACH").as_deref() != Ok("0"))
}

/// A/B kill-switch for the special-point battery phase (`SIMPLIPY_SPECIAL_BATTERY=0`
/// disables it). Ablation/repro only -- like the interval gate, it is a soundness layer,
/// not a knob (see `rust/battery.rs`).
fn special_battery_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_SPECIAL_BATTERY").as_deref() != Ok("0"))
}

/// WITNESS SNAPPING (see `battery::snap_candidates`): the shipped rule's
/// exists-witness set contains the integer/half-integer snap of the fitted constants
/// whenever the snap still fits, and deployment realizes the rule AT the snap -- so the
/// gates downstream of the fit must test it too (a raw `pow(x, 2.9999999999999996)` witness is NaN on
/// x < 0 and hides the half-line extension the snapped 3.0 creates). Adoption uses the
/// fit's own accept semantics -- `allclose_extends` with the hiprec rescue on near-miss
/// rows (the raw witness itself was often accepted only through the rescue: saturation
/// rows like `log(pow(x, y))` at f64-overflowing powers) -- with ONE relaxation: an
/// infinity's SIGN never vetoes adoption. Under the contract's ONE-zero doctrine (zero
/// sign erased) `pow(0, -1) = +inf`, and the deployed `-0.0` corner rows' sign residue is
/// a known, accepted gap: `exp(neg(log(-0.0))) = +inf` vs `pow(-0.0, -1.0) = -inf` must
/// not veto the witness -1.0 (observed live: the whole `-> pow(?0, -1)` extension family
/// kept its unsnapped raw witness and slipped the domain gate).
#[allow(clippy::too_many_arguments)]
fn adopt_snapped_witness(
    cand_tape: &Tape,
    cand_tokens: &[String],
    source_tokens: &[String],
    src_params: &[f64],
    ops: &Operators,
    var_names: &[String],
    cols: &[Vec<f64>],
    n_rows: usize,
    y_src: &[f64],
    fitted: Vec<f64>,
    rtol: f64,
    atol: f64,
) -> Vec<f64> {
    let Some(snapped) = crate::battery::snap_candidates(&fitted) else {
        return fitted;
    };
    let y_fit = cand_tape.eval_columns(cols, &snapped, n_rows);
    // one-zero relaxation: an (inf, inf) row pair passes regardless of sign -- feed the
    // downstream checks the candidate's own value there so neither vetoes the pair
    let y_adj: Vec<f64> = y_src
        .iter()
        .zip(&y_fit)
        .map(|(&a, &b)| {
            if a.is_infinite() && b.is_infinite() {
                b
            } else {
                a
            }
        })
        .collect();
    let ok = allclose_extends(&y_adj, &y_fit, rtol, atol)
        || crate::hiprec::rescue(
            &y_adj,
            &y_fit,
            source_tokens,
            src_params,
            cand_tokens,
            &snapped,
            ops,
            var_names,
            cols,
            rtol,
            atol,
        );
    if ok {
        snapped
    } else {
        fitted
    }
}

/// F3 literal resolution: pin a matched Const-bearing candidate's slots to literal
/// tokens, for a const-FREE source (whose slot values are fixed reals, not existential
/// families). The fit's witness only SEEDS the pin -- its last bits are least-squares
/// noise (measured: the log-linear closed form landed ~20 ulp off `e^pi`) -- so the
/// shipped value is found by ROOT-FINDING at the miner's arbiter precision
/// (`hiprec::point_diff`, 1024 bits): on a probe row, `g(c) = target(c) - source` is
/// sign-bisected over the f64 grid (monotone ulp-key space) inside a geometrically
/// expanded bracket around the witness, converging to the adjacent f64 pair straddling
/// the true constant; the smaller-residual side is the correctly-rounded literal. A
/// SECOND probe row must then confirm the choice against both ulp neighbors.
/// Deterministic: a pure function of (witness, candidate, source, X). Fail closed
/// (`None`): a vacuous witness, a non-finite or multi-slot witness (a multi-slot fit on
/// a const-free source carries no per-slot identifiability certificate), no bracketing
/// probe row, or a confirm row that prefers a neighbor.
fn resolve_const_slots(
    cand_tokens: &[String],
    witness: Option<&[f64]>,
    source: &[String],
    ops: &Operators,
    var_names: &[String],
    x_cols: &[Vec<f64>],
    y_src: &[f64],
) -> Option<Vec<String>> {
    let w = witness?;
    let n_slots = cand_tokens
        .iter()
        .filter(|t| t.as_str() == "<constant>")
        .count();
    if n_slots != 1 || w.len() != 1 || !w[0].is_finite() {
        return None;
    }
    let g = |c: f64, row: usize| -> Option<crate::hiprec::ArbFloat> {
        crate::hiprec::point_diff(source, &[], cand_tokens, &[c], ops, var_names, x_cols, row)
    };
    // Try to PIN on one row, CONFIRM on a later one; a row that cannot bracket (target
    // locally insensitive to the slot, or unevaluable) is skipped. At most a handful of
    // rows are consulted -- resolution runs once per accepted candidate.
    const MAX_PIN_ROWS: usize = 8;
    let mut pinned: Option<f64> = None;
    let mut rows_tried = 0usize;
    for (row, ys) in y_src.iter().enumerate() {
        if !ys.is_finite() {
            continue;
        }
        match pinned {
            None => {
                rows_tried += 1;
                if rows_tried > MAX_PIN_ROWS {
                    return None;
                }
                match pin_root_on_row(&g, w[0], row) {
                    Some(c) => pinned = Some(c),
                    None => {
                        if gate_trace().is_some_and(|t| source.join(" ").starts_with(t.as_str())) {
                            eprintln!("RESOLVE-DBG pin failed row={row} w={}", w[0]);
                        }
                    }
                }
            }
            Some(c) => {
                // Confirm: this row must not strictly prefer either ulp neighbor.
                let (r, ru, rd) = (
                    g(c, row)?,
                    g(key_f64(f64_key(c).wrapping_add(1)), row)?,
                    g(key_f64(f64_key(c).wrapping_sub(1)), row)?,
                );
                if crate::hiprec::residual_lt(&ru, &r) || crate::hiprec::residual_lt(&rd, &r) {
                    if gate_trace().is_some_and(|t| source.join(" ").starts_with(t.as_str())) {
                        eprintln!("RESOLVE-DBG confirm refused row={row} c={c}");
                    }
                    return None;
                }
                let tok = crate::numeric::py_float_repr(c);
                return Some(
                    cand_tokens
                        .iter()
                        .map(|t| {
                            if t == "<constant>" {
                                tok.clone()
                            } else {
                                t.clone()
                            }
                        })
                        .collect(),
                );
            }
        }
    }
    None
}

/// Monotone bijection f64 <-> u64 over the finite total order (negative floats' bit
/// patterns order in reverse; flipping them makes ulp arithmetic a plain +-1 on keys).
fn f64_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1u64 << 63)
    }
}

fn key_f64(k: u64) -> f64 {
    if k >> 63 == 1 {
        f64::from_bits(k & !(1u64 << 63))
    } else {
        f64::from_bits(!k)
    }
}

/// Sign-bisection of `g` on one probe row: expand a bracket around the witness key
/// geometrically (1, 2, 4, ... ulp, capped at 2^24 -- least-squares witnesses sit within
/// rtol of the truth, orders of magnitude inside the cap) until the endpoints' signs
/// differ, then bisect the integer key interval to the adjacent f64 pair and return the
/// smaller-residual side. `None`: unevaluable g, a non-finite bracket endpoint, or no
/// sign change inside the cap (this row cannot see the root).
fn pin_root_on_row(
    g: &dyn Fn(f64, usize) -> Option<crate::hiprec::ArbFloat>,
    w: f64,
    row: usize,
) -> Option<f64> {
    let sign = |v: &crate::hiprec::ArbFloat| -> i8 {
        if v.is_zero() {
            0
        } else if v.is_negative() {
            -1
        } else {
            1
        }
    };
    let gw = g(w, row)?;
    if sign(&gw) == 0 {
        return Some(w); // exact at arbiter precision
    }
    let wk = f64_key(w);
    let (mut lo, mut hi): (u64, u64);
    let (mut g_lo, mut g_hi): (crate::hiprec::ArbFloat, crate::hiprec::ArbFloat);
    let mut step = 1u64;
    const MAX_STEP: u64 = 1 << 24;
    loop {
        let (nlo, nhi) = (wk.saturating_sub(step), wk.saturating_add(step));
        let (clo, chi) = (key_f64(nlo), key_f64(nhi));
        if !clo.is_finite() || !chi.is_finite() {
            return None;
        }
        (g_lo, g_hi) = (g(clo, row)?, g(chi, row)?);
        (lo, hi) = (nlo, nhi);
        if sign(&g_lo) == 0 {
            return Some(clo);
        }
        if sign(&g_hi) == 0 {
            return Some(chi);
        }
        if sign(&g_lo) != sign(&g_hi) {
            break;
        }
        if step >= MAX_STEP {
            return None; // no bracket: this row cannot see the root
        }
        step <<= 1;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let gm = g(key_f64(mid), row)?;
        match sign(&gm) {
            0 => return Some(key_f64(mid)),
            s if s == {
                if g_lo.is_negative() {
                    -1
                } else {
                    1
                }
            } =>
            {
                lo = mid;
                g_lo = gm;
            }
            _ => {
                hi = mid;
                g_hi = gm;
            }
        }
    }
    Some(if crate::hiprec::residual_lt(&g_lo, &g_hi) {
        key_f64(lo)
    } else {
        key_f64(hi)
    })
}

/// Evaluate both sides of one certified instance on the special battery and apply the
/// special-point row semantics (`battery::rows_consistent`). `true` when there is nothing
/// to check (no battery: variable-free source).
#[allow(clippy::too_many_arguments)]
fn battery_rows_ok(
    src_tape: &Tape,
    source_tokens: &[String],
    src_params: &[f64],
    cand_tape: &Tape,
    cand_tokens: &[String],
    cand_params: &[f64],
    battery: Option<&crate::battery::SpecialBattery>,
    ops: &Operators,
    var_names: &[String],
    rtol: f64,
    atol: f64,
) -> bool {
    let Some(sb) = battery else { return true };
    let y_s = src_tape.eval_columns(&sb.cols, src_params, sb.n_rows);
    let y_c = cand_tape.eval_columns(&sb.cols, cand_params, sb.n_rows);
    crate::battery::rows_consistent(
        &y_s,
        &y_c,
        sb,
        source_tokens,
        src_params,
        cand_tokens,
        cand_params,
        ops,
        var_names,
        rtol,
        atol,
    )
}

/// The NO-CONSTANT equivalence test: the candidate has no `<constant>`, so it is
/// a fixed function -- evaluate it once. The SOURCE may carry `n_src_const` constants, and the rule
/// must hold for ALL of them, so we resample the source's constants over the `source_const_magnitudes`
/// rounds and every sign combination and require `allclose_extends(source, candidate)` EVERY time
/// (generic equivalence: source-finite rows bind, source-nonfinite rows are extendable).
/// Rejects on the first mismatch (the guard against a coincidental single-value match).
/// PARITY: a fast-path reject is re-judged by the SAME `hiprec::rescue` the production arm
/// (`candidate_matches`) applies -- the two entry points must never diverge (inf rows are
/// must-match at f64, so without the rescue `log(exp(x)) -> x` would be rejected here while
/// certified there).
#[allow(clippy::too_many_arguments)]
fn equivalent_no_const(
    source: &Tape,
    candidate: &Tape,
    source_tokens: &[String],
    cand_tokens: &[String],
    ops: &Operators,
    var_names: &[String],
    n_src_const: usize,
    x_cols: &[Vec<f64>],
    n_rows: usize,
    challenges: usize,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    rng: &mut Rng,
) -> bool {
    let y_cand = candidate.eval_columns(x_cols, &[], n_rows);
    let combos = sign_combos(n_src_const);
    // A const-free source has exactly one distinct instance (see `source_instances`).
    let eff_challenges = if n_src_const == 0 { 1 } else { challenges };
    // Fast NECESSARY-condition gate: every evidence row needs the candidate finite on it, and
    // evidence rows are UNIQUE rows, so finite(y_cand) bounds the evidence from above. Kills
    // all-NaN/all-inf candidates (the asin(cosh(_0)) -> nan family) outright.
    if crate::eval::count_finite(&y_cand) < min_informative {
        return false;
    }
    // EVIDENCE = UNIQUE rows where SOME instance's source is finite. Unique, NOT accumulated
    // with multiplicity: 16 challenges x sign-combos over the same ~7 defined rows must count
    // as 7 evidence points, else an almost-nowhere-defined const-bearing source (e.g.
    // asin(cosh(C*_0)), finite only at 0 for every C) would clear the gate by repetition.
    let mut evidence = vec![false; n_rows];
    let mags = if n_src_const == 0 {
        vec![vec![]]
    } else {
        source_const_magnitudes(rng, n_src_const, eff_challenges)
    };
    let mut instances: Vec<Vec<f64>> = Vec::with_capacity(mags.len() * combos.len());
    for rc in &mags {
        for combo in &combos {
            let params: Vec<f64> = rc.iter().zip(combo).map(|(r, c)| r * c).collect();
            let y = source.eval_columns(x_cols, &params, n_rows);
            // GENERIC EQUIVALENCE: source-finite rows must match; source-nonfinite
            // rows may be domain-EXTENDED by the candidate (x/x -> 1). a = source, b = candidate.
            // Inf rows are must-match at f64; the near-miss rescue then re-judges the
            // failing rows by escalation (saturation inf resolves finite, a true pole stays).
            if !allclose_extends(&y, &y_cand, rtol, atol)
                && !crate::hiprec::rescue(
                    &y,
                    &y_cand,
                    source_tokens,
                    &params,
                    cand_tokens,
                    &[],
                    ops,
                    var_names,
                    x_cols,
                    rtol,
                    atol,
                )
            {
                return false;
            }
            for (e, v) in evidence.iter_mut().zip(&y) {
                *e |= v.is_finite();
            }
            instances.push(params);
        }
    }
    // EVIDENCE GATE: enough distinct defined points must back the certification, else an
    // (almost-)nowhere-defined source would be rewritten from its corner rows alone.
    if evidence.iter().filter(|&&e| e).count() < min_informative {
        return false;
    }
    // SPECIAL-POINT PHASE (see `rust/battery.rs`). PARITY: identical to the
    // const-free arm of `candidate_matches` -- the two entry points must never diverge.
    if special_battery_on() {
        let used = crate::battery::used_variables(source_tokens, var_names);
        let battery = crate::battery::SpecialBattery::build(var_names.len(), &used);
        for params in &instances {
            if !battery_rows_ok(
                source,
                source_tokens,
                params,
                candidate,
                cand_tokens,
                &[],
                battery.as_ref(),
                ops,
                var_names,
                rtol,
                atol,
            ) {
                return false;
            }
        }
        // the special source-constant sweep (skip semantics: an instance where the
        // source is nowhere finite binds nothing; a binding instance is judged at the
        // CONTRACT points only -- the sweep never binds generic X rows)
        if n_src_const == 1 {
            for &s in crate::battery::SPECIAL_CONSTS.iter() {
                let sp = vec![s];
                let y_s = source.eval_columns(x_cols, &sp, n_rows);
                if crate::eval::count_finite(&y_s) == 0 {
                    continue;
                }
                if !battery_rows_ok(
                    source,
                    source_tokens,
                    &sp,
                    candidate,
                    cand_tokens,
                    &[],
                    battery.as_ref(),
                    ops,
                    var_names,
                    rtol,
                    atol,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// FFI-facing wrapper: compile both expressions and run the no-constant equivalence test. The
/// candidate must be constant-free (its `<constant>` count is 0); the source's constant count drives
/// the resampling. `var_names` columns come from row-major `x_flat`.
#[allow(clippy::too_many_arguments)]
pub fn equivalent_no_const_check(
    ops: &Operators,
    source: &[String],
    candidate: &[String],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    challenges: usize,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    seed: u64,
) -> Result<bool, String> {
    let src_tape = Tape::compile(source, ops, var_names)?;
    let cand_tape = Tape::compile(candidate, ops, var_names)?;
    if cand_tape.n_params != 0 {
        return Err("candidate has <constant> tokens; use exist_constants_fit for constant-bearing candidates".to_string());
    }
    let n_vars = var_names.len();
    if x_flat.len() != n_rows * n_vars {
        return Err("x_flat shape mismatch".to_string());
    }
    let cols = columns_from_row_major(x_flat, n_rows, n_vars);
    let mut rng = Rng::new(seed);
    Ok(equivalent_no_const(
        &src_tape,
        &cand_tape,
        source,
        candidate,
        ops,
        var_names,
        src_tape.n_params,
        &cols,
        n_rows,
        challenges,
        rtol,
        atol,
        min_informative,
        &mut rng,
    ))
}

/// Winner selection: among matched candidates prefer the FEWEST `<constant>`s
/// (stable -> discovery-order tiebreak), skip any that violate wildcard multiplicity, and if the
/// chosen target is bare `<constant>` while the source is all-numeric, fold to the literal value.
///
/// `accept` is the CALLER's acceptance criterion (the serve-time reduction ordering in the
/// mine and certify paths), threaded INTO the selection: a refused candidate asks for the
/// next one instead of vetoing the whole selection (finding F2 -- the enumeration doctrine
/// of finding F1 applied to candidate selection). It judges the candidate's FINAL spelling,
/// i.e. after the bare-`<constant>` literal fold.
#[allow(dead_code)] // rlib surface; exercised by tests
pub fn select_best(
    source: &[String],
    mut matches: Vec<Vec<String>>,
    ops: &Operators,
    accept: Option<AcceptFn<'_>>,
) -> Option<Vec<String>> {
    if matches.is_empty() {
        return None;
    }
    // stable sort by <constant> count (Vec::sort_by_key is stable, matching Python's sorted()).
    matches.sort_by_key(|c| c.iter().filter(|t| t.as_str() == "<constant>").count());
    for cand in matches {
        if violates_wildcard_multiplicity(source, &cand) {
            continue;
        }
        let mut chosen = cand;
        if chosen.len() == 1 && chosen[0] == "<constant>" {
            let leaves: Vec<&String> = source.iter().filter(|t| !ops.is_operator(t)).collect();
            if !leaves.is_empty() && leaves.iter().all(|t| crate::utils::is_numeric_string(t)) {
                if let Some(tok) = crate::numeric::evaluate_constant_subtree(source, ops) {
                    chosen = vec![tok];
                }
            }
        }
        if let Some(f) = accept {
            if !f(&chosen) {
                continue;
            }
        }
        return Some(chosen);
    }
    None
}

/// A caller-supplied target-acceptance predicate (`find_rule_with_lib`'s `accept` /
/// `accept_resolved`).
pub type AcceptFn<'a> = &'a dyn Fn(&[String]) -> bool;

/// A precompiled candidate (an entry of [`CandidateLibrary`]).
struct CandEntry {
    tokens: Vec<String>,
    var_mask: u32,
    n_const: usize,
    linearity: crate::fit::Linearity,
    tape: Tape,
    y_const_free: Option<Vec<f64>>,
    /// count_finite(y_const_free) precomputed at library build (0 for const-bearing candidates):
    /// the informativeness gate rejects vacuous (all-NaN/inf) candidates before any comparison.
    finite_y: usize,
    /// Precomputed log-linear plan (form + the const-free `g` tape) for nonlinear
    /// `pow(C,g)` / `pow(g,C)` candidates -> closed-form fit instead of the LM.
    loglin: Option<(crate::fit::LogLinForm, Tape)>,
    /// mu of this candidate in the BARE context -- the SEARCH ORDER and the SEARCH BOUND.
    /// `u64::MAX` for a candidate the measure cannot score (unparseable): it sorts last and
    /// no finite bound admits it, which matches acceptance, since that also requires a
    /// computable mu (`accept_resolved`'s `Some(tc)`).
    mu: u64,
    /// FNV-1a over the tokens: the ORDER-INDEPENDENT ingredient of the per-(candidate, instance)
    /// fit seed (see `candidate_matches`), so a candidate's LM restart stream depends only on the
    /// source seed and the candidate itself -- never on which OTHER candidates are in the library
    /// or where in the scan it sits. This is what makes the fold-filter parity gate exact:
    /// a filtered and an unfiltered mine draw IDENTICAL fit randomness for every shared candidate.
    hash: u64,
}

/// FNV-1a over a token slice, with a separator byte so ["ab","c"] != ["a","bc"].
fn token_hash(tokens: &[String]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for t in tokens {
        for b in t.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Weyl-sequence increment (golden-ratio) used to mix the instance index into the fit seed.
const SEED_GOLDEN: u64 = 0x9E3779B97F4A7C15;

/// Bitmask over `var_names` of the variables appearing in `tokens` (<=32 vars).
fn var_mask(tokens: &[String], var_names: &[String]) -> u32 {
    let mut m = 0u32;
    for (i, v) in var_names.iter().enumerate() {
        if i < 32 && tokens.iter().any(|t| t == v) {
            m |= 1 << i;
        }
    }
    m
}

/// Test one candidate against a source across the challenge/sign-combo loop. Const-free candidates
/// use the `allclose` check (against the precomputed `y_const_free`); constant-bearing candidates
/// use the fit (`exist_constants_fit_prepared`, `retries` restarts) per combo. Rejects on first
/// failure (`None`). A match returns the FIRST instance's target witness (`Some(c)` = the fitted
/// constants for the candidate's `<constant>` slots, `None` = a vacuous instance that determined
/// nothing) -- for a const-FREE source there is exactly one instance, so this IS the candidate's
/// witness, which literal resolution (`resolve_const_slots`) pins to tokens.
#[allow(clippy::too_many_arguments)]
fn candidate_matches(
    y_src: &[(Vec<f64>, Vec<f64>)],
    cand: &CandEntry,
    source: &[String],
    src_tape: &Tape,
    battery: Option<&crate::battery::SpecialBattery>,
    ops: &Operators,
    var_names: &[String],
    cols: &[Vec<f64>],
    n_rows: usize,
    retries: usize,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    fit_seed: u64,
) -> Option<Option<Vec<f64>>> {
    // Fast NECESSARY-condition gate, const-free arm (see `equivalent_no_const`): every
    // evidence row needs the candidate finite on it, so finite_y bounds the UNIQUE evidence.
    if cand.n_const == 0 && cand.finite_y < min_informative {
        return None;
    }
    // EVIDENCE = UNIQUE rows where SOME instance's source is finite (BOTH arms; see
    // `equivalent_no_const` for why repetition across instances must not count).
    let mut evidence = vec![false; n_rows];
    // the certified (source-constants, target-witness) pairs, for the special-point phase
    let mut certified: Vec<(Vec<f64>, Option<Vec<f64>>)> = Vec::with_capacity(y_src.len());
    for (inst, (inst_params, y)) in y_src.iter().enumerate() {
        // WITNESS, not just a bool: None = reject; Some(None) = VACUOUS pass (no constants
        // determined -- every row was extendable); Some(Some(c)) = pass AT the constants `c`.
        // The domain gate needs `c` to bind the candidate's `<constant>` leaves.
        let ok: Option<Option<Vec<f64>>> = if cand.n_const == 0 {
            // GENERIC EQUIVALENCE: source-finite rows must match; source-nonfinite
            // rows may be domain-EXTENDED (x/x -> 1). a = source, b = candidate. f64 is a
            // PRE-FILTER: a near-miss reject is re-judged at high precision on exactly the
            // failing rows (`hiprec::rescue`) so that true identities
            // whose f64 round-trip loses precision -- atanh(tanh(x)) -> x -- still certify.
            let y_cand = cand.y_const_free.as_ref().unwrap();
            let hit = allclose_extends(y, y_cand, rtol, atol)
                || crate::hiprec::rescue(
                    y,
                    y_cand,
                    source,
                    inst_params,
                    &cand.tokens,
                    &[],
                    ops,
                    var_names,
                    cols,
                    rtol,
                    atol,
                );
            // A const-free candidate binds nothing, so even a vacuous match has a determined
            // (empty) witness and the gate applies exactly.
            if hit {
                Some(Some(Vec::new()))
            } else {
                None
            }
        } else {
            // ORDER-INDEPENDENT fit seed: a pure function of (per-source seed,
            // candidate tokens, instance index), NOT a draw from the scan-order RNG stream --
            // dropping or reordering OTHER candidates (e.g. the fold-filter) must not change
            // this candidate's LM restarts, or the filtered-vs-unfiltered parity gate would
            // drown in seed-shift noise. `Rng::new(..).next_u64()` is one splitmix64 step,
            // i.e. a full 64-bit mix of the combined key.
            let s = Rng::new(
                fit_seed
                    .wrapping_add(cand.hash)
                    .wrapping_add((inst as u64).wrapping_mul(SEED_GOLDEN)),
            )
            .next_u64();
            match crate::fit::exist_constants_fit_prepared(
                &cand.tape,
                cand.linearity,
                cols,
                n_rows,
                y,
                rtol,
                atol,
                retries,
                s,
                cand.loglin.as_ref().map(|(f, t)| (*f, t)),
            ) {
                crate::fit::FitVerdict::Pass(c) => Some(c),
                crate::fit::FitVerdict::Fail => None,
                // FIT-ARM ESCALATION: a true constant-bearing rule whose SOURCE
                // f64 evaluation is corrupted on a few rows (atanh(tanh(C*x)) -> C*x) fails
                // the fit with CONTAMINATED constants (the corrupted rows pull the least
                // squares), so first recover the majority fit by robust median trimming
                // (`fit::refit_robust`), then re-judge the rows that still fail -- at P bits,
                // at the refit constants (a legitimate "constants exist" witness).
                // `hiprec::rescue` applies its own near-miss cost gate.
                crate::fit::FitVerdict::NearMiss(c0) => {
                    match crate::fit::refit_robust(
                        &cand.tape,
                        cand.linearity,
                        cols,
                        n_rows,
                        y,
                        &c0,
                        rtol,
                        atol,
                    ) {
                        None => None, // no small-outlier consensus: keep the f64 reject
                        Some(c1) => {
                            let y_fit = cand.tape.eval_columns(cols, &c1, n_rows);
                            if crate::hiprec::rescue(
                                y,
                                &y_fit,
                                source,
                                inst_params,
                                &cand.tokens,
                                &c1,
                                ops,
                                var_names,
                                cols,
                                rtol,
                                atol,
                            ) {
                                Some(Some(c1))
                            } else {
                                None
                            }
                        }
                    }
                }
            }
        };
        let fitted = match ok {
            Some(f) => f,
            None => {
                if gate_trace().is_some_and(|t| source.join(" ").starts_with(t.as_str())) {
                    let nf = y.iter().filter(|v| v.is_finite()).count();
                    eprintln!(
                        "TRACE-CM fit-reject inst={inst} cand=[{}] n_finite={nf}",
                        cand.tokens.join(" ")
                    );
                }
                return None;
            }
        };
        // witness snapping before the domain gate (see `adopt_snapped_witness`)
        let fitted: Option<Vec<f64>> = fitted.map(|c| {
            if c.is_empty() {
                c
            } else {
                adopt_snapped_witness(
                    &cand.tape,
                    &cand.tokens,
                    source,
                    inst_params,
                    ops,
                    var_names,
                    cols,
                    n_rows,
                    y,
                    c,
                    rtol,
                    atol,
                )
            }
        });
        // DOMAIN-PRESERVATION gate (exact, interval-analytic), applied PER INSTANCE at the
        // witness the fit chose: a rule may complete a MEASURE-ZERO hole (`x/x -> 1` at 0, a
        // removable singularity) but must never define a value across a POSITIVE-MEASURE region
        // where the source is NaN -- that is a DOMAIN BOUNDARY (`exp(log x) -> x` off x>0) and
        // the rewrite would INVENT a function the source never was. The numeric arms cannot see
        // this: `allclose_extends` skips every NaN row regardless of measure.
        // A VACUOUS instance (`None`) determined no constants, so it pins no target function to
        // test -- and it contributes no evidence either, so `min_informative` still decides.
        if let Some(cand_params) = &fitted {
            // ONE horizon per gate decision, shared by the extension witness and BOTH deadness
            // measures. Computing them separately lets a source read "dead" off a smaller box
            // than the harm it exempts (its own constants can be smaller than the pair's), a
            // fail-open in the anti-conservative direction. `gate_horizon` counts a horizon
            // miss ONCE per decision.
            //
            // FAIL-CLOSED: every undecided path REJECTS. An undecidable horizon (interesting
            // region past R_MAX) and an exhausted node budget with no witness (`None` from
            // `domain_extension_p_at`) must never read as "no extension" = accept. The
            // counters (`interval_horizon_misses`, `interval_node_budget_misses`) count
            // fail-closed REJECTIONS; a large per-stratum delta in a mine is the flag to
            // raise `SIMPLIPY_IVL_NODE_BUDGET` / R_MAX for that stratum and re-run it.
            if ivl_gate_on() {
                let (gh_r, gh_d, gh_decidable) = crate::interval::gate_horizon(
                    source,
                    inst_params,
                    &cand.tokens,
                    cand_params,
                    ops,
                );
                let verdict = if gh_decidable {
                    crate::interval::domain_extension_p_at(
                        source,
                        inst_params,
                        &cand.tokens,
                        cand_params,
                        ops,
                        gh_r,
                        gh_d,
                    )
                } else {
                    None
                };
                let reject = match verdict {
                    // Undecided (horizon or budget): cannot certify "no extension" -- reject.
                    None => true,
                    // A proven positive-measure extension fires the gate ...
                    // ... unless the rule keeps a DEAD source dead. ONE clause, stated once:
                    //
                    //   fire  <=>  extension  AND NOT (source dead AND target dead)
                    //
                    // where "dead" = defined on no positive-measure set at these constants. A
                    // dead source has no generic graph to preserve, and an existential witness
                    // that is ITSELF dead buys the rule no capacity the source lacked --
                    // `sqrt(C^x)` at C<0 is NaN a.e. (finite only on the even integers), and the
                    // fit's `C' = -sqrt(2)` is equally NaN a.e., so nothing was invented. But
                    // resurrecting a dead source into a LIVE target injects fittable capacity
                    // from nothing, and that is the artifact.
                    //
                    // This REPLACES two ad-hoc carve-outs. The first was
                    // `cand_params.is_empty()`, justified as "a const-free candidate has no
                    // witness to be arbitrary" and as what "keeps the f64 SATURATION artifacts
                    // out". It does not: the verdict it produces depends on the target's
                    // SPELLING, not its mathematics. `acosh(|tanh x|) -> 0` was rejected while
                    // `acosh(|tanh x|) -> <constant>` was ACCEPTED at the identical witness
                    // width, and `select_best` prefers fewest constants -- so the miner reached
                    // for the rejected form, was refused, and mined the surviving one instead.
                    // Deadness is a property of the FUNCTION at its constants, so it cannot be
                    // dodged by respelling: a fitted `<constant>` binds to a finite real and is
                    // therefore LIVE, exactly like the literal `0` it was standing in for.
                    //
                    // Deadness must be PROVEN (`Some(0.0)` = a COMPLETED search found no
                    // support). An unresolved deadness search (`None`) is treated as live:
                    // letting it read "dead" widened the accept exemption, the
                    // anti-conservative direction.
                    Some(w) if w > 0.0 => {
                        !(crate::interval::defined_measure_p_at(
                            source,
                            inst_params,
                            ops,
                            gh_r,
                            gh_d,
                        ) == Some(0.0)
                            && crate::interval::defined_measure_p_at(
                                &cand.tokens,
                                cand_params,
                                ops,
                                gh_r,
                                gh_d,
                            ) == Some(0.0))
                    }
                    // Completed search, no extension: the gate passes.
                    Some(_) => false,
                };
                if reject {
                    if let Some(t) = gate_trace() {
                        if source.join(" ").starts_with(t.as_str()) {
                            eprintln!(
                                "GATE-REJECT src=[{}] src_c={:?} cand=[{}] cand_c={:?} width={:?} src_measure={:?}",
                                source.join(" "), inst_params, cand.tokens.join(" "), cand_params,
                                verdict,
                                crate::interval::defined_measure_p_at(source, inst_params, ops, gh_r, gh_d));
                        }
                    }
                    return None;
                }
            }
        }
        for (e, v) in evidence.iter_mut().zip(y) {
            *e |= v.is_finite();
        }
        certified.push((inst_params.clone(), fitted));
    }
    if evidence.iter().filter(|&&e| e).count() < min_informative {
        return None;
    }
    // SPECIAL-POINT PHASE (see `rust/battery.rs`): runs only for candidates
    // that certified everything above, so its cost is per accepted (source, candidate) pair.
    if special_battery_on() {
        // 1. Contract points at every certified instance's witness. A vacuous instance pins
        //    no target function and is skipped, exactly like the domain gate above.
        for (inst_params, fitted) in &certified {
            let Some(cand_params) = fitted else { continue };
            if !battery_rows_ok(
                src_tape,
                source,
                inst_params,
                &cand.tape,
                &cand.tokens,
                cand_params,
                battery,
                ops,
                var_names,
                rtol,
                atol,
            ) {
                return None;
            }
        }
        // 2. The special source-constant battery: a pattern-bound source constant reaches
        //    0, pi/2, ... in deployment. Skip semantics: an instance with a nowhere-finite
        //    source, or (const-bearing arm) no fittable witness, binds nothing; a binding
        //    instance must agree at the contract points. Scoped to single-constant sources
        //    -- multi-constant shapes have no ratified special-constant semantics beyond
        //    one constant.
        let n_src_const = y_src.first().map(|(p, _)| p.len()).unwrap_or(0);
        if n_src_const == 1 {
            for (k, &s) in crate::battery::SPECIAL_CONSTS.iter().enumerate() {
                let sp = vec![s];
                let y_s = src_tape.eval_columns(cols, &sp, n_rows);
                if crate::eval::count_finite(&y_s) == 0 {
                    continue;
                }
                let cp: Option<Vec<f64>> = if cand.n_const == 0 {
                    // const-free target: judged at the contract points below (the
                    // source-constant sweep never binds generic X rows)
                    Some(Vec::new())
                } else {
                    // existence is only generically binding: an unfittable special instance
                    // is SKIPPED, a fitted one binds below
                    let sseed = Rng::new(
                        fit_seed ^ cand.hash ^ (0xC0DEu64 + k as u64).wrapping_mul(SEED_GOLDEN),
                    )
                    .next_u64();
                    match crate::fit::exist_constants_fit_prepared(
                        &cand.tape,
                        cand.linearity,
                        cols,
                        n_rows,
                        &y_s,
                        rtol,
                        atol,
                        retries,
                        sseed,
                        cand.loglin.as_ref().map(|(f, t)| (*f, t)),
                    ) {
                        crate::fit::FitVerdict::Pass(Some(c)) => Some(adopt_snapped_witness(
                            &cand.tape,
                            &cand.tokens,
                            source,
                            &sp,
                            ops,
                            var_names,
                            cols,
                            n_rows,
                            &y_s,
                            c,
                            rtol,
                            atol,
                        )),
                        _ => None,
                    }
                };
                let Some(cp) = cp else { continue };
                if !battery_rows_ok(
                    src_tape,
                    source,
                    &sp,
                    &cand.tape,
                    &cand.tokens,
                    &cp,
                    battery,
                    ops,
                    var_names,
                    rtol,
                    atol,
                ) {
                    return None;
                }
            }
        }
    }
    Some(certified.into_iter().next().and_then(|(_, w)| w))
}

/// The source-side challenge instances, evaluated ONCE PER SOURCE and shared across every
/// candidate (re-evaluating inside `candidate_matches` would cost per (candidate, challenge,
/// sign-combo)). A const-free source has exactly one distinct instance, so its challenge count
/// collapses to 1 (identical targets add no evidence and only multiply fit flakiness).
fn source_instances(
    src_tape: &Tape,
    n_src_const: usize,
    combos: &[Vec<f64>],
    cols: &[Vec<f64>],
    n_rows: usize,
    challenges: usize,
    rng: &mut Rng,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let eff_challenges = if n_src_const == 0 { 1 } else { challenges };
    let mags = if n_src_const == 0 {
        vec![vec![]]
    } else {
        source_const_magnitudes(rng, n_src_const, eff_challenges)
    };
    let mut out = Vec::with_capacity(mags.len() * combos.len());
    for rc in &mags {
        for combo in combos {
            let p: Vec<f64> = rc.iter().zip(combo).map(|(r, c)| r * c).collect();
            let y = src_tape.eval_columns(cols, &p, n_rows);
            // params are RETAINED alongside the values so the hiprec rescue can
            // re-evaluate the exact same instance at high precision.
            out.push((p, y));
        }
    }
    out
}

/// A RESIDENT candidate library: every candidate precompiled to a tape and indexed by length,
/// with each const-free candidate's `y` evaluated ONCE (not per source). Built once per mine;
/// reused by `find_rule_with_lib` for every source -- this removes the per-source rebuild that
/// the per-call `find_rule` path pays.
pub struct CandidateLibrary {
    var_names: Vec<String>,
    cols: Vec<Vec<f64>>,
    n_rows: usize,
    by_len: Vec<Vec<CandEntry>>, // index = candidate length
    /// mu TIERS, ascending: (mu, [(length, index into by_len[length])]). The scan walks these
    /// while mu < the caller's mark, so the search bound IS the acceptance predicate rather
    /// than a token-length proxy for it.
    by_mu: Vec<(u64, Vec<(usize, usize)>)>,
    /// Highest mu tier holding a Const-BEARING candidate. Those are priced by what they
    /// RESOLVE to, not by their own mu (`<constant>` is the priciest atom, yet it resolves
    /// to `1/2`), so the scan bound cannot refuse them on their unresolved mu -- it must
    /// keep walking to here and let `accept_resolved` price the resolved spelling.
    max_const_mu: u64,
    /// candidates kept in the library (post-filter)
    n_candidates: usize,
    /// variable-free candidates dropped by the fold-filter (0 when the filter is off/inert)
    n_filtered: usize,
}

impl CandidateLibrary {
    /// Row count of the library's shared X (for resolving informativeness-gate defaults).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The variable vocabulary the library was built over (for var-freeness tests on
    /// sources, e.g. the mining acceptance's determined-source clause).
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Candidates kept in the library (post-filter).
    pub fn n_candidates(&self) -> usize {
        self.n_candidates
    }

    /// Variable-free candidates dropped by the fold-filter.
    pub fn n_filtered(&self) -> usize {
        self.n_filtered
    }

    /// Build the resident library. With `fold_filter`, VARIABLE-FREE candidates of length >= 2
    /// are dropped, PROVIDED the bare
    /// `["<constant>"]` candidate is present (otherwise the filter is inert -- conservative).
    ///
    /// SOUNDNESS (the narrow, provable form of "candidate minimization"): a variable-free
    /// candidate evaluates to ONE scalar per constant-assignment (a constant function of X), so
    /// - const-bearing var-free (`sin(<constant>)`, `pow(<constant>,<constant>)`, ...): any
    ///   source instance it matches is (near-)constant-valued, which the length-1 `<constant>`
    ///   candidate also matches -- with a fit family (all of R) that SUPERSET-dominates the
    ///   wrapper's reachable set;
    /// - const-free var-free (`exp(1)`, `neg(np.pi)`, ...): a fixed value; any match implies the
    ///   per-instance `<constant>` fit matches too (and its all-nonfinite members never match:
    ///   the finite-evidence gate rejects them).
    ///
    /// Since `find_rule_with_lib` scans lengths SHORTEST-FIRST and returns at the first matching
    /// length, the length-1 `<constant>` match preempts every length>=2 var-free candidate, which
    /// is therefore never selectable and can be dropped without changing any mined rule. The one
    /// theoretical edge is tolerance-band width: `<constant>`'s solved fit sits within <= 2x the
    /// per-row tolerance of the wrapper's (mean vs member), relevant only for sources within
    /// ~rtol of the band edge -- real mine values agree to ~1e-16 vs rtol 1e-9, and the
    /// filtered-vs-unfiltered mine parity test certifies the dominance empirically.
    pub fn build(
        ops: &Operators,
        candidates: &[Vec<String>],
        mus: &[Option<u64>],
        folds_to_leaf: &[bool],
        var_names: &[String],
        x_flat: &[f64],
        n_rows: usize,
        fold_filter: bool,
    ) -> Result<Self, String> {
        if mus.len() != candidates.len() {
            return Err("mus/candidates length mismatch".to_string());
        }
        let n_vars = var_names.len();
        if x_flat.len() != n_rows * n_vars {
            return Err("x_flat shape mismatch".to_string());
        }
        let cols = columns_from_row_major(x_flat, n_rows, n_vars);
        let max_len = candidates.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut by_len: Vec<Vec<CandEntry>> = (0..=max_len).map(|_| Vec::new()).collect();
        let filter_active = fold_filter
            && candidates
                .iter()
                .any(|c| c.len() == 1 && c[0] == "<constant>");
        // WHAT A VAR-FREE CANDIDATE MUST BE TO EARN ITS PLACE (revised 2026-08-21, after
        // the mu-dominance criterion below it was measured UNSOUND).
        //
        // The original lemma -- "the length-1 `<constant>` preempts every var-free
        // candidate" -- held only because the scan was ordered by TOKEN LENGTH, and the
        // mu bound broke it. The repair attempt, "drop those with mu >= mu(`<constant>`)",
        // is worse than the disease: `<constant>` prices at 67000 while `asin sin 2` is
        // 19000 and `* 0 <constant>` is 3000, so the whole class of var-free composites
        // sailed through. A live mine under that criterion produced 57 rules of the form
        // `pi - 2 -> asin(sin(2))` in its first 212: true by value, and an obfuscation
        // rather than a simplification.
        //
        // THE CRITERION IS WHAT THE CONSTRUCTOR MAKES OF IT, not what the measure charges.
        // A var-free candidate is admitted iff the constructor folds it to a SINGLE LEAF:
        //
        //     exp 1            -> np.e     KEPT   (a leaf the engine can name exactly)
        //     * 0 <constant>   -> 0        KEPT
        //     asin sin 2       -> unchanged  DROPPED (a composite spelling of pi - 2)
        //     log <constant>   -> unchanged  DROPPED (the universal absorber the filter
        //                                    was written for: it ranges over every real
        //                                    AND nan, so it "matches" any constant-family
        //                                    source and mints vacuous rules)
        //
        // This keeps the fix the mu criterion was reaching for -- dropping `exp 1` forced
        // the source through `<constant>` to the ROUNDED literal 2.7182818260590453, the
        // respell the mint doctrine forbids -- while excluding everything that made the
        // mu version unsound. The falsifier is TestFoldFilter::test_dominance_holds_at_
        // the_band_edge plus the no-composite-var-free-RHS invariant asserted on the mine.
        let mut n_candidates = 0usize;
        let mut n_filtered = 0usize;
        for (ci, c) in candidates.iter().enumerate() {
            let len = c.len();
            if len == 0 {
                continue;
            }
            let vm = var_mask(c, var_names);
            // Filter decision NOT via `vm == 0`: the scan mask truncates at 32 variables, so a
            // candidate whose only variables have index >= 32 would be misclassified as
            // var-free and wrongly dropped. The filter checks ALL var_names.
            let has_var = c.iter().any(|t| var_names.iter().any(|v| v == t));
            if filter_active && len >= 2 && !has_var && !folds_to_leaf[ci] {
                // NOTE: this filter carries more weight than "candidate
                // minimization". A var-free candidate of length >= 2 can be a UNIVERSAL ABSORBER:
                // `log <constant>` ranges over every real AND nan, so under the rule semantics
                // (for every source constant there EXISTS a target constant) it "matches" any
                // constant-family source at all -- `cosh asin <constant> -> log <constant>` is
                // formally sound and completely vacuous. Dropping these candidates is what keeps
                // such matches out; an experiment that kept the non-`Finite`-class ones to
                // preserve filtered/unfiltered parity promptly shipped exactly that rule and was
                // reverted. Parity is now asserted as an INCLUSION, not an equality
                // (tests/test_mining_soundness.py::TestFoldFilter).
                n_filtered += 1;
                continue;
            }
            let tape = Tape::compile(c, ops, var_names)?;
            let n_const = tape.n_params;
            let y_const_free = if n_const == 0 {
                Some(tape.eval_columns(&cols, &[], n_rows))
            } else {
                None
            };
            let linearity = crate::fit::classify(c, ops)?;
            let loglin = if linearity == crate::fit::Linearity::Nonlinear {
                match crate::fit::detect_log_linear(c, ops) {
                    Some((form, g)) => Some((form, Tape::compile(&g, ops, var_names)?)),
                    None => None,
                }
            } else {
                None
            };
            let finite_y = y_const_free
                .as_ref()
                .map(|y| crate::eval::count_finite(y))
                .unwrap_or(0);
            by_len[len].push(CandEntry {
                tokens: c.clone(),
                var_mask: vm,
                n_const,
                linearity,
                tape,
                y_const_free,
                finite_y,
                loglin,
                hash: token_hash(c),
                mu: mus[ci].unwrap_or(u64::MAX),
            });
            n_candidates += 1;
        }
        // The mu tiers, built once: every candidate keyed by its mu, ascending.
        let mut tiers: std::collections::BTreeMap<u64, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for (len, bucket) in by_len.iter().enumerate() {
            for (idx, e) in bucket.iter().enumerate() {
                tiers.entry(e.mu).or_default().push((len, idx));
            }
        }
        let by_mu: Vec<(u64, Vec<(usize, usize)>)> = tiers.into_iter().collect();
        let max_const_mu = by_len
            .iter()
            .flatten()
            .filter(|e| e.n_const > 0)
            .map(|e| e.mu)
            .max()
            .unwrap_or(0);
        Ok(CandidateLibrary {
            var_names: var_names.to_vec(),
            cols,
            n_rows,
            by_len,
            by_mu,
            max_const_mu,
            n_candidates,
            n_filtered,
        })
    }
}

/// The full native `find_rule_worker` decision over a RESIDENT library:
/// guard -> all-numeric short-circuit -> scan candidates shortest-first (variable-subset filtered),
/// dispatch const-free -> the no-constant test / const-bearing -> the constant fit, stop at the
/// first length with an ACCEPTED match, `select_best`.
///
/// `accept` (optional) is the caller's acceptance criterion, threaded into the scan: when every
/// match at a length is refused, the scan CONTINUES to the next length -- token count is not the
/// reduction ordering, so a longer spelling can still sit strictly below the caller's mark
/// (finding F2: the old accept-after-scan shape returned one representative and let the caller
/// veto the whole search).
///
/// `accept_resolved` (optional) is an ADDITIONAL criterion for COMPUTED-literal targets:
/// literal-RESOLVED candidates (see `resolve_const_slots`) and the evaluate
/// short-circuit's f64 answer alike. Mining passes strict-semantic-complexity-below-mark:
/// both channels exist to recover values that must be computed
/// (`exp(pi*x) -> pow(23.14..., x)`), never to respell literals -- without the strict
/// gate the full lex ordering's literal tiers accept 1-ulp respells of the engine's own
/// fold results (the arbiter's correctly-rounded constant can differ from the fold's
/// libm value and happen to print SHORTER; the f64 evaluation of a source the engine
/// folds by exact rational arithmetic can differ the same way, `+ 1 exp (-1)`) and
/// reciprocal respells (`x * acos(0) -> x / 0.6366...`, whose canonical coefficient is
/// a giant exact rational). Resolved targets are also DEFERRED behind same-length
/// symbolic matches: when an exact spelling certifies
/// (`abs(pow(x, np.e)) -> pow(x, np.e)`), it is preferred over materializing the
/// constant.
#[allow(clippy::too_many_arguments)]
pub fn find_rule_with_lib(
    ops: &Operators,
    source: &[String],
    simplified_length: usize,
    max_target: Option<usize>,
    max_mu: Option<u64>,
    lib: &CandidateLibrary,
    challenges: usize,
    retries: usize,
    seed: u64,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    accept: Option<AcceptFn<'_>>,
    accept_resolved: Option<AcceptFn<'_>>,
) -> Result<Option<Vec<String>>, String> {
    // GUARD FIRST (BEFORE the all-constant short-circuit).
    let max_cand_len = match max_target {
        Some(mt) => simplified_length.min(mt + 1),
        None => simplified_length,
    };
    if max_cand_len <= 1 {
        return Ok(None);
    }
    // Every return path honors the caller's acceptance -- the short-circuits included: a
    // refused short-circuit target falls through to the candidate scan instead of being
    // returned for the caller to veto (there is no caller-side veto anymore).
    let accepted = |t: &[String]| match accept {
        Some(f) => f(t),
        None => true,
    };
    // all-constant short-circuit, reached only after the guard passes.
    // A leaf counts as constant when `leaf_value` resolves it to a FINITE value: numeric
    // literals AND the special constants (`np.pi`, `np.e`, `(-1)`). The previous
    // `is_numeric_string`-only gate sent e.g. `/ (-1) <constant>` to the library scan
    // (inf/nan literals stay excluded -- non-finite algebra is the explicit rules' domain,
    // e.g. `+ float("-inf") <constant> -> float("-inf")`, not absorption).
    let finite_leaf = |t: &str| crate::numeric::leaf_value(t).is_some_and(f64::is_finite);
    // STAGE-1 GROUND-FOLD LICENCE, miner side (owner-ratified 2026-07-31; a future
    // theorem of the unified measure, design/UNIFIED_SIMPLICITY_MEASURE.md §5): a
    // special-bearing source never short-circuits to its f64 evaluation -- that literal
    // is a ROUNDING of an exact symbolic value (`rootn np.e (-1) ->
    // 0.36787944117144233`). The refusal is by CHANNEL, not by target shape: exact
    // finite hits (`sin np.pi -> 0`) still mint through the candidate scan below, whose
    // equivalence machinery reads specials at high precision (hiprec rescue), while a
    // rounded value finds no alphabet literal to match. The non-finite class arms stay
    // OPEN to special-bearing sources: nan/inf collapse is exact (the ratified
    // nan/inf x special absorption rows), certified by intervals plus the dirac refuter.
    // SPELLING-COMPLETE, not spelling-literal (2026-08-07). This tests whether the source
    // DENOTES a special, and since 2026-08-07 the token scan alone no longer decides that:
    // the constructor folds `exp 1` to the leaf `e` (`fun`, the write half of
    // `compose_e_power`), so `* <constant> exp 1` is the canonical state
    // `<mul> np.e <constant> </mul>` while carrying no `np.e` token. Testing the raw tokens
    // alone let exactly that source through and the re-mine minted
    // `* <constant> exp 1 -> <constant>` plus its +, - and / mirrors -- an absorption the
    // licence forbids and the CONSTRUCTOR itself refuses, so the mine was routing around a
    // ratified refusal through a second spelling of one value.
    //
    // `exp 1` is the ONLY spelling that needs adding, and that is a checkable claim rather
    // than a hope: `Ex::Pi` has no constructor arm producing it at all, and `Ex::E` is
    // produced by exactly one -- `exp(Num(1))`. In prefix a unary head is immediately
    // followed by its argument, so consecutive `exp 1` IS the subterm.
    //
    // This guard covers the GROUND-FOLD channel (the short-circuit just below). The
    // ABSORPTION channel has its own twin in `miner.rs`, and that one could not be fixed
    // this way: `/ <constant> exp (-1)` denotes `<constant> * e` through no `exp 1` subterm
    // at all, so the mint-side licence counts specials on the engine's own ENDPOINT for the
    // source instead of on any spelling. Here the spelling test is enough and provably
    // complete, because this channel only ever sees the source's own leaves.
    //
    // The general question both leave open is recorded in the audit (F49, register C1.20):
    // a gate keyed on a spelling breaks whenever the canon unifies two spellings, and the
    // miner has no mechanism that asks the engine "would you refuse this?" -- it only asks
    // "do you already derive it?" (the mint-pristine property). The dual is missing.
    let src_special = source.iter().any(|t| t == "np.pi" || t == "np.e")
        || source.windows(2).any(|w| w[0] == "exp" && w[1] == "1");
    let src_has_const = source.iter().any(|t| t == "<constant>");
    if source.len() > 1
        && source
            .iter()
            .all(|t| t == "<constant>" || ops.is_operator(t) || finite_leaf(t))
    {
        let non_ops: Vec<&String> = source.iter().filter(|t| !ops.is_operator(t)).collect();
        if !src_special && !non_ops.is_empty() && non_ops.iter().all(|t| finite_leaf(t)) {
            if let Some(tok) = crate::numeric::evaluate_constant_subtree(source, ops) {
                // DENOTES A SPECIAL BY VALUE, NOT BY SPELLING (2026-08-08; register B, and the
                // dual C1.20 records above). The licence's own prose already states the right
                // test -- whether the source DENOTES a special -- and its token scan cannot
                // see one that arrives as a computed VALUE: `acos (-1)` carries no `np.pi`
                // token and no `exp 1` subterm, yet it IS pi. So `exp 1 is the ONLY spelling
                // that needs adding` above is complete for CONSTRUCTOR-PRODUCED leaves and
                // silent about evaluated ones; this is the other half.
                //
                // Left alone, the short-circuit answers with the f64 decimal
                // 3.141592653589793 -- a ROUNDING of an exact symbolic value, exactly what
                // this licence exists to refuse -- and since that decimal is mu-expensive
                // nothing descends, no rule mints, and the candidate scan never runs. The
                // corpus paid for it in FORTY-EIGHT rules, every one a context of the single
                // fact `acos(-1) = pi`: `* acos (-1) _0 -> * np.pi _0` and its +, -, /, pow
                // and both-argument-order mirrors, `cos acos (-1) -> -1`, `/ acos (-1) np.pi
                // -> 1` -- with no bare `acos (-1) -> np.pi` among them, because that is the
                // one rule this channel prevented. Same shape as F49/F51/F52/F53: an arm the
                // engine lacks shows up as rules.
                //
                // Declining routes the source to the CANDIDATE SCAN -- the channel that
                // already mints `sin np.pi -> 0` -- whose equivalence machinery reads specials
                // at high precision. The f64 comparison is only the TRIGGER; hiprec is the
                // proof. A value that merely SHARES pi's f64 without being pi finds no
                // alphabet match there and stays symbolic, so the failure direction is closed
                // rather than open. Both signs trigger: the sign is carried by the same
                // rounding argument, and which of them the mine actually reaches is left to
                // the re-mine to answer rather than prescribed here.
                let denotes_special = crate::numeric::leaf_value(&tok).is_some_and(|x| {
                    x.abs() == std::f64::consts::PI || x.abs() == std::f64::consts::E
                });
                // A NON-FINITE f64 verdict must be corroborated by the honest interval
                // CLASS (H-019/H-021, 2026-08-04): f64 evaluation cannot distinguish an
                // exact boundary hit from a true domain escape -- `acosh(acos(cos 1))`
                // evaluates nan at EVERY float precision (the argument rounds below 1)
                // while its exact value is acosh(1) = 0. Under the old point-precision
                // interval arms the engine's class fold unsoundly collapsed such grounds
                // itself, so they never reached this short-circuit; the outward-rounded
                // arms refuse (MIXED), the sources arrive here, and without this gate the
                // same lie re-minted as sixteen explicit nan rules. On a class mismatch
                // the source falls through: the class arms below cannot speak either
                // (not a clean class), and the candidate scan's class-preservation guard
                // refuses the literal -- fail closed, the ground stays symbolic.
                let corroborated = match crate::numeric::leaf_value(&tok).filter(|x| !x.is_finite())
                {
                    None => true, // finite verdicts keep the existing channel semantics
                    Some(x) => match crate::interval::value_class(source, ops) {
                        // UNANALYZABLE: the honest layer has NO opinion (rootn's
                        // non-integer-index nan family lives here) -- the channel keeps
                        // its pre-H-019 semantics. A registered residual, strictly no
                        // wider than the ungated channel ever was.
                        None => true,
                        // An analyzed class must AGREE. Mismatches are exactly the two
                        // defect families: MIXED boundary grazes (`acosh(acos(cos 1))`
                        // is 0, not the f64 nan) and rendered-overflow poles
                        // (`cosh(tan(acos 0))` composes through tan's exact pole --
                        // undefined -- while f64 renders tan(f64(pi/2)) as a huge
                        // finite and overflows cosh to inf).
                        Some(cls) => {
                            (x.is_nan() && cls == crate::interval::Class::Nan)
                                || (x == f64::INFINITY && cls == crate::interval::Class::PosInf)
                                || (x == f64::NEG_INFINITY && cls == crate::interval::Class::NegInf)
                        }
                    },
                };
                if corroborated && !denotes_special {
                    let t = vec![tok];
                    // An f64-EVALUATED literal is a COMPUTED value, not an alphabet
                    // spelling, so it answers to the caller's resolved-target criterion
                    // exactly like the resolution channel below (mining passes
                    // strict-semantic-complexity-below-mark). Without this, the full
                    // ordering's literal tie tiers minted 1-ulp respells of the engine's
                    // own fold results: the engine reduces `+ 1 exp (-1)` by EXACT
                    // decimal-rational arithmetic to `1.36787944117144233`, f64
                    // arithmetic gives the DIFFERENT value `1.3678794411714423`, and the
                    // shorter print won the lit_size tier -- measured 2026-08-01 at
                    // ~1.4k mint-then-drop rules per canonical mine, contained only by
                    // sort promotion's UNSUPPORTED bucket (and `promote_sorts` defaults
                    // to OFF, where they shipped as serve-time rounding rewrites).
                    // Callers without the criterion (stage-2 confirmation, the public
                    // query path) keep the short-circuit as-is.
                    let resolved_ok = match accept_resolved {
                        Some(f) => f(&t),
                        None => true,
                    };
                    if resolved_ok && accepted(&t) {
                        return Ok(Some(t));
                    }
                }
            }
        }
        // A variable-free <constant>-bearing source collapses to a single leaf ONLY when that
        // leaf reproduces it a.e. Classified EXACTLY by interval analysis (`rust/interval.rs`),
        // not by sampling: a POLE -- a finite region together with an inf region, e.g.
        // `pow(0, C)` = 0 for C>0 and +inf for C<0, or `* <constant> inv 0` = C*inf -- is NOT a
        // constant, so returning `<constant>` there is an artifact. Finite -> `<constant>` (nan
        // rows extend); all-NaN -> nan (`* <constant> acos(np.e)`); a.e. +-inf -> that inf literal
        // (`inv(pow(0, exp C))` = +inf everywhere); a mixed pole -> fall through to the candidate
        // scan. The sampled classifier this replaces could only ever be as good as its grid.
        match if ivl_class_on() {
            crate::interval::value_class(source, ops)
        } else {
            None
        } {
            // The Finite -> `<constant>` arm is gated on SPECIAL-FREE sources (stage-1
            // Const-absorption licence at the channel, owner 2026-08-01): a special
            // vanishing into a fitted constant is the absorption the policy forbids, for
            // EVERY caller. This is also what makes stage-2 confirmation honest for the
            // exact-collapse family: confirmation re-runs this function with the target
            // as the only candidate, and this arm used to HIJACK the answer for
            // `cos np.pi` (returning `<constant>` instead of testing `(-1)`), scoring
            // every bare ground exact collapse "unconfirmed". Gated, the source falls
            // through to the scan, which verifies the asked target -- exact hits
            // (`cos np.pi -> (-1)`, owner: "cos(pi) should be simplified to -1") mint
            // and confirm; inexact values still find no alphabet literal to match.
            //
            // ALSO gated on `<constant>`-BEARING sources -- the arm's contract as
            // documented above ("a variable-free `<constant>`-bearing source"): a
            // DETERMINED source (var-free, Const-free, all leaves finite literals) is
            // its own reduction -- the engine folds it to its exact value and never
            // absorbs a determined value into a mask -- so `<constant>` is not an
            // answer for it. Determined sources only ever reached this arm shadowed:
            // the evaluate short-circuit answered first (value-agreeing sources fell
            // through when their answer equaled the mark; value-disagreeing ones fell
            // through once the strict resolved-target gate closed the respell leak),
            // and the arm minted `+ 1 exp (-1) -> <constant>` -- a Const-INTRODUCING
            // ground rule that died only because confirmation's answer-shape differs
            // (measured 2026-08-01 on the canonical mine). The mining acceptance
            // carries the matching refusal so the candidate scan cannot re-mint what
            // this arm no longer answers.
            Some(crate::interval::Class::Finite) => {
                if !src_special && src_has_const {
                    let t = vec!["<constant>".to_string()];
                    if accepted(&t) {
                        return Ok(Some(t));
                    }
                }
            }
            // The three LITERAL arms publish a measure verdict as an attainability claim, so each
            // is checked against the exact-point layer before it may speak. A disagreement does
            // not just drop the literal -- it FALLS THROUGH to the candidate scan, where the same
            // refuter guards whatever the scan proposes.
            Some(crate::interval::Class::Nan) => {
                let t = vec!["float(\"nan\")".to_string()];
                if !dirac_refutes_literal(source, &t, ops) && accepted(&t) {
                    return Ok(Some(t));
                }
            }
            Some(crate::interval::Class::PosInf) => {
                let t = vec!["float(\"inf\")".to_string()];
                if !dirac_refutes_literal(source, &t, ops) && accepted(&t) {
                    return Ok(Some(t));
                }
            }
            Some(crate::interval::Class::NegInf) => {
                let t = vec!["float(\"-inf\")".to_string()];
                if !dirac_refutes_literal(source, &t, ops) && accepted(&t) {
                    return Ok(Some(t));
                }
            }
            _ => {} // Mixed / Empty / unevaluable -> not a clean collapse: candidate scan below
        }
    }
    let src_tape = Tape::compile(source, ops, &lib.var_names)?;
    let n_src_const = src_tape.n_params;
    let src_mask = var_mask(source, &lib.var_names);
    let combos = sign_combos(n_src_const);

    let mut rng = Rng::new(seed);
    // Source instances shared across the whole candidate scan (see `source_instances`).
    let y_src = source_instances(
        &src_tape,
        n_src_const,
        &combos,
        &lib.cols,
        lib.n_rows,
        challenges,
        &mut rng,
    );
    // REACHABILITY gate (exact, interval-analytic) for a VARIABLE-FREE source: every value
    // component the source takes over its constants must be reachable by the candidate over its
    // own (see `interval::reaches_all_of`). Scoped to variable-free sources -- the class the
    // interval engine is validated on, and exactly the family the sampled POLE_GRID exists for
    // (`pow(asin(C/2), inf) -> 0`, whose +inf band a fixed grid can straddle and miss).
    // Computed ONCE per source, outside the scan.
    let src_reach = if ivl_reach_on() && !source.iter().any(|t| lib.var_names.contains(t)) {
        crate::interval::value_set(source, ops, &crate::interval::Vs::reals())
    } else {
        None
    };
    // Special-point battery over THIS source's variable set (None for a variable-free
    // source), shared across the whole candidate scan; see `rust/battery.rs`.
    let battery = crate::battery::SpecialBattery::build(
        lib.var_names.len(),
        &crate::battery::used_variables(source, &lib.var_names),
    );
    // THE SEARCH BOUND. `max_mu` (the caller's mark, i.e. mu of the engine's own endpoint
    // for this source) makes the search bound the SAME predicate as the acceptance gate:
    // walk mu tiers ascending and stop at the first tier that cannot descend. The token
    // ceiling is then NOT applied -- it is retired as a criterion, which is the point. A
    // longer target can be strictly cheaper (`1/5` -> `0.2`, `pow(inf,x)` -> `exp(inf*x)`)
    // and the length bound was silently refusing exactly those.
    // With `max_mu = None` (the per-call FFI and its tests) the legacy token bound stands.
    let scan_max = max_cand_len.min(lib.by_len.len());
    for (tier_mu, members) in lib.by_mu.iter() {
        // Beyond the mark only Const-BEARING candidates remain eligible (see `max_const_mu`):
        // their price is the RESOLVED spelling's, which `accept_resolved` checks in strict mu.
        let mut resolvable_only = false;
        if let Some(cap) = max_mu {
            // `>`, NOT `>=`. The acceptance predicate is `ordered_below`, which is mu
            // comparison with a CANONICAL TIE-BREAK: at equal mu it still accepts when
            // `cmp_ex(t, mark) == Less`. Breaking at the equal tier would prune targets
            // acceptance would have taken -- the bound must be exactly as permissive as
            // the gate it stands in for, or it is just a different arbitrary criterion.
            // Strictly-greater tiers are refused by `ordered_below` outright, so stopping
            // there loses nothing.
            if *tier_mu > cap {
                if *tier_mu > lib.max_const_mu {
                    break; // tiers ascend: no resolvable candidate remains either
                }
                resolvable_only = true;
            }
        }
        let length = *tier_mu; // for the trace below
        let mut matches: Vec<Vec<String>> = Vec::new();
        // Resolved-literal matches rank BEHIND every symbolic match of the same length
        // (appended after the loop): materializing a constant is the fallback, never
        // preferred over an exact spelling that also certifies.
        let mut resolved_matches: Vec<Vec<String>> = Vec::new();
        for &(clen, cidx) in members.iter() {
            if max_mu.is_none() && (clen < 1 || clen >= scan_max) {
                continue; // legacy token bound, only when no mu mark was given
            }
            let cand = &lib.by_len[clen][cidx];
            if resolvable_only && cand.n_const == 0 {
                continue; // its own mu is its price, and it is already above the mark
            }
            if cand.var_mask & !src_mask != 0 {
                continue; // candidate uses a variable the source lacks
            }
            // The Const-count invariant at the mint (its serve twin is the translate-time
            // count gate; one invariant, both layers): a fire may never INCREASE the number
            // of `<constant>` placeholders. For a const-BEARING source, a candidate with
            // more slots than the source could only mint a count-increasing rule -- its
            // slot values are per-challenge functions of the drawn source constants, not
            // fixed reals, so literal resolution cannot apply either. For a const-FREE
            // source, Const-bearing candidates stay IN the scan: their slots are fixed
            // reals, resolved to literal tokens below.
            if n_src_const > 0 && cand.n_const > n_src_const {
                continue;
            }
            if let Some(sv) = &src_reach {
                if let Some(cv) =
                    crate::interval::value_set(&cand.tokens, ops, &crate::interval::Vs::reals())
                {
                    if !crate::interval::reaches_all_of(&cv, sv) {
                        continue; // cannot reach a component the source takes: FALSE, no sampling
                    }
                }
            }
            if let Some(witness) = candidate_matches(
                &y_src,
                cand,
                source,
                &src_tape,
                battery.as_ref(),
                ops,
                &lib.var_names,
                &lib.cols,
                lib.n_rows,
                retries,
                rtol,
                atol,
                min_informative,
                seed,
            ) {
                // (the domain gate runs INSIDE `candidate_matches`, per instance, where the
                // source's drawn constants and the fit's chosen constants are both in scope)
                // LITERAL RESOLUTION (the Const-count invariant's constructive arm): a
                // const-FREE source that matched a Const-bearing candidate did so at FIXED
                // slot values -- abstraction manufactured at mint time, which is masking's
                // job downstream, never the miner's. The slots are pinned to literal tokens
                // (fail closed: an unpinnable slot skips the candidate and the scan
                // continues), so the minted rule carries the VALUE, not a mask. The
                // downstream guards and the caller's ordering acceptance then judge the
                // RESOLVED spelling -- a resolution the engine already reaches natively
                // (the fold family) lands on the mark's own state and is refused there.
                let is_resolution = n_src_const == 0 && cand.n_const > 0;
                let target: Vec<String> = if is_resolution {
                    let resolved = y_src.first().and_then(|inst| {
                        resolve_const_slots(
                            &cand.tokens,
                            witness.as_deref(),
                            source,
                            ops,
                            &lib.var_names,
                            &lib.cols,
                            &inst.1,
                        )
                    });
                    let resolved = resolved.filter(|t| match accept_resolved {
                        Some(f) => f(t),
                        None => true,
                    });
                    match resolved {
                        Some(t) => t,
                        None => {
                            if let Some(t) = gate_trace() {
                                if source.join(" ").starts_with(t.as_str()) {
                                    eprintln!(
                                        "RESOLVE-FAIL src=[{}] cand=[{}] witness={witness:?}",
                                        source.join(" "),
                                        cand.tokens.join(" ")
                                    );
                                }
                            }
                            continue;
                        }
                    }
                } else {
                    cand.tokens.clone()
                };
                // The scan reaches literal targets the short-circuit never sees: `finite_leaf`
                // excludes any source carrying an inf/nan literal from the short-circuit
                // entirely, so guarding only the arms would leave that whole family unjudged.
                // Judged on the RESOLVED spelling: a resolved bare literal is exactly the
                // single-token collapse this refuter exists for.
                if dirac_refutes_literal(source, &target, ops) {
                    continue;
                }
                // COLLAPSE-TO-`<constant>` guard, the ratified zoo-collapse predicate applied at
                // the MINT rather than only at the online fold: a source may collapse to a free
                // finite constant only if its finite part has POSITIVE MEASURE. `inv <constant>`
                // qualifies (finite off the measure-zero pole, which is why the family is
                // ratified); `pow (-1) <constant>` does not (finite only at the integers, a null
                // set), and without this the literal refuter above merely diverts it into an
                // equally false `-> <constant>`. Same predicate as the engine's value-set
                // collapse fold (`engine/ac.rs::ac_fold`, `cls()==Finite && fin_pm()`), so
                // the two paths cannot disagree about what "collapses to a constant" means.
                // CLASS PRESERVATION for a VARIABLE-FREE source (ratified licence).
                // Such a source is a constant-family: its whole behaviour is a function of its
                // `<constant>` slots, so a rewrite must preserve the value CLASS, which is a
                // measure statement over the constant axis. `reaches_all_of` above deliberately
                // ignores nan (a nan row is extendable, "decided by measure rather than by
                // reachability") -- this is where that measure decision is made.
                //   `exp log <constant>` is MIXED (finite on C>0, undefined on C<0) while
                //   `exp <constant>` is FINITE: the rewrite would DEFINE a positive-measure
                //   undefined region, which §4 R3 and §9.8.3(b) put in the kill set.
                //   `pow 0 <constant>` (MIXED: 0 on C>0, +inf on C<0) likewise cannot become a
                //   bare `<constant>`, and `pow (-1) <constant>` (nan a.e.) cannot become
                //   anything finite.
                // Null-set exceptions stay tolerated because the CLASS ignores them: `inv
                // <constant>` is FINITE (its +inf lives only at C=0) and keeps collapsing, per
                // §9.8.3(c) and the §9.8.4 degenerate-literal carve-out.
                if let Some(sv) = &src_reach {
                    let same_class =
                        crate::interval::value_set(&target, ops, &crate::interval::Vs::reals())
                            .is_some_and(|cv| cv.cls() == sv.cls());
                    if !same_class {
                        continue;
                    }
                    // A source finite only on a NULL set additionally may not collapse to a free
                    // constant (the ratified zoo-collapse predicate, mint-side).
                    if target.len() == 1 && target[0] == "<constant>" && !sv.fin_pm() {
                        continue;
                    }
                }
                if is_resolution {
                    resolved_matches.push(target);
                } else {
                    matches.push(target);
                }
            }
        }
        matches.extend(resolved_matches);
        if !matches.is_empty() {
            if let Some(t) = gate_trace() {
                if source.join(" ").starts_with(t.as_str()) {
                    eprintln!(
                        "TRACE src=[{}] simplified_len={} cand_mu={} matches={:?}",
                        source.join(" "),
                        simplified_length,
                        length,
                        matches.iter().map(|m| m.join(" ")).collect::<Vec<_>>()
                    );
                }
            }
            // CHEAPEST ACCEPTED mu tier wins. When select_best refuses every match in this
            // tier (wildcard multiplicity or the caller's acceptance), the scan continues:
            // a costlier spelling can still sit strictly below the caller's mark.
            if let Some(best) = select_best(source, matches, ops, accept) {
                return Ok(Some(best));
            }
        }
    }
    if let Some(t) = gate_trace() {
        if source.join(" ").starts_with(t.as_str()) {
            eprintln!(
                "TRACE src=[{}] simplified_len={} -> NO MATCH",
                source.join(" "),
                simplified_length
            );
        }
    }
    Ok(None)
}

/// Per-call convenience: build a `CandidateLibrary` then delegate. Kept for the per-call FFI /
/// tests; the mine uses the resident-library path.
#[allow(clippy::too_many_arguments)]
pub fn find_rule(
    ops: &Operators,
    source: &[String],
    simplified_length: usize,
    max_target: Option<usize>,
    candidates: &[Vec<String>],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    challenges: usize,
    retries: usize,
    seed: u64,
    rtol: f64,
    atol: f64,
    min_informative: usize,
    fold_filter: bool,
) -> Result<Option<Vec<String>>, String> {
    // Per-call path: no engine here to score mu, so the library is unscored and the legacy
    // token bound stands (`max_mu = None`). The MINE goes through the resident-library path.
    let mus = vec![None; candidates.len()];
    let folds = vec![false; candidates.len()];
    let lib = CandidateLibrary::build(
        ops, candidates, &mus, &folds, var_names, x_flat, n_rows, fold_filter,
    )?;
    find_rule_with_lib(
        ops,
        source,
        simplified_length,
        max_target,
        None,
        &lib,
        challenges,
        retries,
        seed,
        rtol,
        atol,
        min_informative,
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// The generation-2 witness closure: the retired hyper-operator ring spelled fixed
    /// scalings/shifts/powers as UNARY names (`mult3`, `div4`, `pow2`); generation 2 spells
    /// them as BINARY nodes with a determined operand, so the pullback must read the TREE.
    /// Every expected value below is computed with the engine's own numerics, so the
    /// assertions cannot drift from the evaluator.
    #[test]
    fn slot_witnesses_generation2_sections() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let ev = |expr: &[&str]| {
            crate::numeric::evaluate_constant_subtree(&s(expr), ops)
                .and_then(|t| crate::numeric::leaf_value(&t))
                .unwrap()
        };
        let has = |w: &[f64], v: f64| w.contains(&v);
        let pi2 = std::f64::consts::FRAC_PI_2;
        let pi2r = crate::numeric::py_float_repr(pi2);

        // `* <constant> 4`: the corner pi/2 pulls back to (pi/2)/4 (exact: power-of-two scale)
        let w = slot_witnesses(&s(&["sin", "*", "<constant>", "4"]), ops);
        assert!(has(&w, pi2 / 4.0), "*-section pullback missing");
        // `/ <constant> 4`: pulls back to (pi/2)*4
        let w = slot_witnesses(&s(&["sin", "/", "<constant>", "4"]), ops);
        assert!(has(&w, pi2 * 4.0), "/-section pullback missing");
        // `+ <constant> 1`: pulls back to pi/2 - 1
        let w = slot_witnesses(&s(&["sin", "+", "<constant>", "1"]), ops);
        assert!(has(&w, pi2 - 1.0), "+-section pullback missing");
        // `- <constant> 1`: pulls back to pi/2 + 1
        let w = slot_witnesses(&s(&["sin", "-", "<constant>", "1"]), ops);
        assert!(has(&w, pi2 + 1.0), "--section pullback missing");
        // `pow <constant> 3`: pulls back to rootn(pi/2, 3)
        let w = slot_witnesses(&s(&["sin", "pow", "<constant>", "3"]), ops);
        assert!(
            has(&w, ev(&["rootn", &pi2r, "3"])),
            "pow-section pullback missing"
        );
        // `rootn <constant> 3`: pulls back to pow(pi/2, 3)
        let w = slot_witnesses(&s(&["sin", "rootn", "<constant>", "3"]), ops);
        assert!(
            has(&w, ev(&["pow", &pi2r, "3"])),
            "rootn-section pullback missing"
        );
        // exponent-side `pow 2 <constant>`: pulls back to log2(pi/2)
        let w = slot_witnesses(&s(&["sin", "pow", "2", "<constant>"]), ops);
        assert!(
            has(&w, ev(&["/", "log", &pi2r, "log", "2"])),
            "exponent-side pow-section pullback missing"
        );
        // LIVE odd root `pow1_3` (not retired): its inverse has no unary spelling in any
        // live vocabulary and must be spelled with BINARY pow.
        let w = slot_witnesses(&s(&["sin", "pow1_3", "<constant>"]), ops);
        assert!(
            has(&w, ev(&["pow", &pi2r, "3"])),
            "pow1_3 live-spelled inverse missing"
        );
    }

    /// End-to-end: `acosh(sin(4C))` is defined ONLY on the isolated set sin(4C) = 1
    /// (value 0 there), so its a.e. class is nan -- but `-> nan` is REFUTED at the
    /// witness C = (pi/2)/4, which only the *-section pullback reaches. Deterministic
    /// in f64: the power-of-two scale round-trips exactly and sin(f64 pi/2) saturates
    /// to 1.0, so acosh evaluates to 0.0 != nan.
    #[test]
    fn dirac_refuter_sees_generation2_spellings() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        assert!(dirac_refutes_literal(
            &s(&["acosh", "sin", "*", "<constant>", "4"]),
            &s(&["float(\"nan\")"]),
            ops
        ));
    }

    #[test]
    fn wildcard_multiplicity() {
        // inert on dummy vars (no _j tokens)
        assert!(!violates_wildcard_multiplicity(
            &s(&["+", "x0", "x0"]),
            &s(&["x0"])
        ));
        // _0 twice on rhs, once on lhs -> violates
        assert!(violates_wildcard_multiplicity(
            &s(&["_0"]),
            &s(&["+", "_0", "_0"])
        ));
        // _0 once each -> ok
        assert!(!violates_wildcard_multiplicity(
            &s(&["+", "_0", "_1"]),
            &s(&["_0"])
        ));
    }

    #[test]
    fn no_const_equiv() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let n = 64usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            xf.push((r as f64) * 0.1 - 3.0);
            xf.push((r as f64) * -0.05 + 1.5);
        }
        // source: neg(neg(x0)) == candidate x0  (const-free both) -> equivalent
        assert!(e
            .equivalent_no_const_check(
                &s(&["neg", "neg", "x0"]),
                &s(&["x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-5,
                1e-8,
                8,
                0
            )
            .unwrap());
        // source: + <constant> x0  vs candidate x0 : holds ONLY for constant 0 -> resampling rejects
        assert!(!e
            .equivalent_no_const_check(
                &s(&["+", "<constant>", "x0"]),
                &s(&["x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-5,
                1e-8,
                8,
                0
            )
            .unwrap());
        // source: * <constant> x0 vs candidate x0 : holds only for constant 1 -> rejected
        assert!(!e
            .equivalent_no_const_check(
                &s(&["*", "<constant>", "x0"]),
                &s(&["x0"]),
                &vars,
                &xf,
                n,
                16,
                1e-5,
                1e-8,
                8,
                0
            )
            .unwrap());
    }

    #[test]
    fn find_rule_basic() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let n = 64usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            xf.push((r as f64) * 0.1 - 3.0);
            xf.push((r as f64) * -0.05 + 1.5);
        }
        let lib = vec![s(&["x0"]), s(&["x1"]), s(&["neg", "x0"]), s(&["abs", "x0"])];
        // neg(neg(x0)) (len 3) simplifies to x0 -> find_rule should return ["x0"]
        let r = e
            .find_rule(
                &s(&["neg", "neg", "x0"]),
                3,
                Some(2),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-5,
                1e-8,
                8,
                true,
            )
            .unwrap();
        assert_eq!(r, Some(s(&["x0"])));
        // sin(x0) (len 2) has no shorter equivalent in the library -> None
        let r2 = e
            .find_rule(
                &s(&["sin", "x0"]),
                2,
                Some(2),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-5,
                1e-8,
                8,
                true,
            )
            .unwrap();
        assert_eq!(r2, None);
    }

    /// REGRESSION (the lesson that removed mine-time accept-certification):
    /// `pow(np.e, x) -> exp(x)` must be minable. np.e is an f64 LITERAL, so the identity
    /// holds to ~|x| * 1e-16 -- true at the mine tolerance. A checker judging finer than
    /// the mine tolerance vetoes this class of true rules while catching nothing.
    #[test]
    fn f64_literal_identities_stay_minable() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0"]);
        let n = 64usize;
        let xf: Vec<f64> = (0..n).map(|i| -8.0 + 0.25 * i as f64).collect();
        let lib = vec![s(&["x0"]), s(&["exp", "x0"]), s(&["neg", "cos", "x0"])];
        let r = e
            .find_rule(
                &s(&["pow", "np.e", "x0"]),
                3,
                Some(2),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-9,
                1e-12,
                8,
                false,
            )
            .unwrap();
        assert_eq!(
            r,
            Some(s(&["exp", "x0"])),
            "pow(np.e, x) -> exp(x) must be mined"
        );
        let r2 = e
            .find_rule(
                &s(&["cos", "+", "np.pi", "x0"]),
                4,
                Some(3),
                &lib,
                &vars,
                &xf,
                n,
                16,
                16,
                0,
                1e-9,
                1e-12,
                8,
                false,
            )
            .unwrap();
        assert_eq!(
            r2,
            Some(s(&["neg", "cos", "x0"])),
            "cos(np.pi + x) -> -cos(x) must be mined"
        );
    }

    #[test]
    fn fold_filter_drops_var_free_and_preserves_decisions() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["x0"]);
        let n = 128usize;
        let xf: Vec<f64> = (0..n).map(|r| (r as f64) * 0.11 - 7.0).collect();
        // library incl. the bare <constant> (guard token), var-free const-bearing (sin(<c>),
        // pow(<c>, 2)), var-free const-free (exp(1)), and var-bearing candidates
        let cands = vec![
            s(&["x0"]),
            s(&["<constant>"]),
            s(&["sin", "<constant>"]),
            s(&["pow", "<constant>", "2"]),
            s(&["exp", "1"]),
            s(&["neg", "x0"]),
            s(&["*", "<constant>", "x0"]),
        ];
        let filtered = CandidateLibrary::build(ops, &cands, &vec![None; cands.len()], &vec![false; cands.len()], &vars, &xf, n, true).unwrap();
        let raw = CandidateLibrary::build(ops, &cands, &vec![None; cands.len()], &vec![false; cands.len()], &vars, &xf, n, false).unwrap();
        assert_eq!(filtered.n_filtered(), 3); // sin(<c>), pow(<c>, 2), exp(1)
        assert_eq!(filtered.n_candidates(), 4);
        assert_eq!(raw.n_filtered(), 0);
        assert_eq!(raw.n_candidates(), 7);
        // WITHOUT the bare <constant> the filter must be INERT (conservative guard)
        let no_bare: Vec<Vec<String>> = cands[2..].to_vec();
        let inert = CandidateLibrary::build(ops, &no_bare, &vec![None; no_bare.len()], &vec![false; no_bare.len()], &vars, &xf, n, true).unwrap();
        assert_eq!(inert.n_filtered(), 0);
        // decision parity on both a constant-valued and a non-constant source: the dominance
        // lemma says the length-1 <constant> preempts every var-free candidate, so filtered and
        // raw libraries must decide identically (incl. the ORDER-INDEPENDENT fit seeds).
        for src in [
            s(&["-", "x0", "x0"]),    // constant-valued (0) -> matches <constant> at len 1
            s(&["+", "x0", "np.pi"]), // genuinely var-dependent
            s(&["neg", "neg", "x0"]), // reduces to x0
        ] {
            let a = find_rule_with_lib(
                ops,
                &src,
                src.len(),
                Some(2),
                None,
                &filtered,
                16,
                16,
                7,
                1e-9,
                1e-12,
                16,
                None,
                None,
            )
            .unwrap();
            let b = find_rule_with_lib(
                ops,
                &src,
                src.len(),
                Some(2),
                None,
                &raw,
                16,
                16,
                7,
                1e-9,
                1e-12,
                16,
                None,
                None,
            )
            .unwrap();
            assert_eq!(a, b, "filtered vs raw decision diverged on {src:?}");
        }
    }

    #[test]
    fn fold_filter_keeps_var_dependent_candidates_beyond_32_vars() {
        // Regression: the scan var_mask truncates at 32
        // variables, so a candidate whose only variable has index >= 32 has var_mask == 0; the
        // filter must NOT treat it as var-free (it checks all var_names directly).
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars: Vec<String> = (0..33).map(|i| format!("x{i}")).collect();
        let n = 16usize;
        let xf: Vec<f64> = (0..n * 33).map(|i| (i as f64) * 0.01 - 2.0).collect();
        let cands = vec![
            s(&["<constant>"]),
            s(&["sin", "x32"]), // var-DEPENDENT, but var_mask == 0 (index >= 32)
            s(&["sin", "x0"]),
            s(&["exp", "<constant>"]), // genuinely var-free -> filtered
        ];
        let lib = CandidateLibrary::build(ops, &cands, &vec![None; cands.len()], &vec![false; cands.len()], &vars, &xf, n, true).unwrap();
        assert_eq!(lib.n_filtered(), 1, "only exp(<constant>) may be dropped");
        assert_eq!(lib.n_candidates(), 3);
    }

    #[test]
    fn select_prefers_fewest_constants() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let src = s(&["+", "*", "<constant>", "x0", "<constant>"]);
        // two matches: one with a constant, one without -> pick the const-free
        let m = vec![s(&["*", "<constant>", "x0"]), s(&["x0"])];
        assert_eq!(select_best(&src, m, ops, None), Some(s(&["x0"])));
    }

    /// Finding F2: the caller's acceptance rides the selection -- a refused candidate
    /// yields to the NEXT one instead of vetoing the whole selection (the old shape
    /// returned one representative and let the caller throw the search away).
    #[test]
    fn select_acceptance_moves_to_the_next_candidate() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let src = s(&["*", "exp", "x0", "exp", "x0"]);
        let m = vec![s(&["pow", "exp", "x0", "2"]), s(&["exp", "+", "x0", "x0"])];
        // no acceptance: discovery order wins (both const-free, stable sort)
        assert_eq!(
            select_best(&src, m.clone(), ops, None),
            Some(s(&["pow", "exp", "x0", "2"]))
        );
        // acceptance refuses the first: the second is selected, not None
        let not_pow = |t: &[String]| t[0] != "pow";
        assert_eq!(
            select_best(&src, m.clone(), ops, Some(&not_pow)),
            Some(s(&["exp", "+", "x0", "x0"]))
        );
        // acceptance refuses everything: None, honestly
        let nothing = |_: &[String]| false;
        assert_eq!(select_best(&src, m, ops, Some(&nothing)), None);
        // the bare-<constant> literal fold happens BEFORE the acceptance judges: the
        // acceptance sees the FINAL spelling (the folded literal), never the raw token
        let all_num = s(&["+", "1", "1"]);
        let folded = select_best(
            &all_num,
            vec![s(&["<constant>"])],
            ops,
            Some(&|t: &[String]| t == ["2.0"] || t == ["2"]),
        );
        assert!(
            folded.is_some(),
            "acceptance must judge the folded literal, not bare <constant>"
        );
    }

    /// Finding F2, the across-length half: when every match at the shortest matching
    /// length is refused, the scan CONTINUES -- token count is not the reduction
    /// ordering, so a longer spelling can sit strictly below the caller's mark.
    #[test]
    fn scan_continues_past_a_fully_refused_length() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["x0"]);
        let n = 64usize;
        let xf = grid_1var(n);
        // Both candidates are numerically equivalent to the source x0+x0+x0 = 3*x0:
        // `* 3 x0` at length 3, `neg * -3 x0` at length 4.
        let cands = vec![s(&["*", "3", "x0"]), s(&["neg", "*", "-3", "x0"])];
        let lib = CandidateLibrary::build(ops, &cands, &vec![None; cands.len()], &vec![false; cands.len()], &vars, &xf, n, true).unwrap();
        let src = s(&["+", "+", "x0", "x0", "x0"]);
        // Without acceptance the length-3 spelling wins.
        let free = find_rule_with_lib(
            ops,
            &src,
            src.len(),
            None,
            None,
            &lib,
            16,
            16,
            7,
            1e-9,
            1e-12,
            16,
            None,
            None,
        )
        .unwrap();
        assert_eq!(free, Some(s(&["*", "3", "x0"])));
        // With an acceptance that refuses the length-3 spelling, the scan must
        // continue and take the length-4 one.
        let not_star_head = |t: &[String]| t[0] != "*";
        let deep = find_rule_with_lib(
            ops,
            &src,
            src.len(),
            None,
            None,
            &lib,
            16,
            16,
            7,
            1e-9,
            1e-12,
            16,
            Some(&not_star_head),
            None,
        )
        .unwrap();
        assert_eq!(deep, Some(s(&["neg", "*", "-3", "x0"])));
    }

    /// Signed, mixed-magnitude 1-var grid: negatives probe domain extensions, positives feed
    /// the constant fit.
    fn grid_1var(n: usize) -> Vec<f64> {
        (0..n)
            .map(|r| {
                let t = (r as f64 + 0.5) / n as f64;
                let m = 10f64.powf(-2.0 + 3.0 * t); // 1e-2 .. 10
                if r % 2 == 0 {
                    m
                } else {
                    -m
                }
            })
            .collect()
    }

    /// `adopt_snapped_witness`: adopt when the snap still fits (incl. the one-zero
    /// infinity-sign relaxation), keep the raw witness when the snap is a genuinely
    /// different function.
    #[test]
    fn snapped_witness_adoption() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["x0"]);
        let toks = s(&["pow", "x0", "<constant>"]);
        let tape = Tape::compile(&toks, ops, &vars).unwrap();
        // -0.0 corner row included: exp(neg(log(-0.0))) = +inf vs pow(-0.0, -1) = -inf is
        // the known accepted zero-sign residue and must NOT veto the snap to -1.0
        let mut xs: Vec<f64> = (1..40).map(|i| 0.3 * i as f64).collect();
        xs.push(-0.0);
        let cols = vec![xs];
        let n = cols[0].len();
        let src = s(&["exp", "neg", "log", "x0"]);
        let y: Vec<f64> = cols[0]
            .iter()
            .map(|&x| (-(x.ln())).exp()) // +inf at -0.0 (log(-0.0) = -inf)
            .collect();
        let a = adopt_snapped_witness(
            &tape,
            &toks,
            &src,
            &[],
            ops,
            &vars,
            &cols,
            n,
            &y,
            vec![-0.999_999_999_997_003_7],
            1e-9,
            1e-12,
        );
        assert_eq!(a, vec![-1.0]);
        // y genuinely x^2.9999997: within snap range of 3 but pow(x, 3) misses by ~1e-7
        // rel > rtol on every row -- keep the raw witness
        let yb: Vec<f64> = cols[0].iter().map(|&x| x.powf(2.999_999_7)).collect();
        let b = adopt_snapped_witness(
            &tape,
            &toks,
            &s(&["pow", "x0", "<constant>"]),
            &[2.999_999_7],
            ops,
            &vars,
            &cols,
            n,
            &yb,
            vec![2.999_999_7],
            1e-9,
            1e-12,
        );
        assert_eq!(b, vec![2.999_999_7]);
    }

    /// The certification gaps closed by witness snapping + the special-point phase
    /// (`rust/battery.rs`); each case is a family observed live in (4,3) mining runs.
    #[test]
    fn special_phase_closes_certification_gaps() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0"]);
        let n = 512usize;
        let xf = grid_1var(n);
        let fr = |src: &[&str], cand: &[&str]| {
            let src = s(src);
            let cand = s(cand);
            let slen = src.len();
            let mt = cand.len();
            e.find_rule(
                &src,
                slen,
                Some(mt),
                &[cand],
                &vars,
                &xf,
                n,
                16,
                16,
                7,
                1e-9,
                1e-12,
                32,
                false,
            )
            .unwrap()
        };
        // 1. WITNESS SNAPPING + domain gate: exp(log(x^3)) is defined only on x > 0; the raw
        //    fit (2.9999999999999996) is NaN on x < 0 and hides the extension, the snapped
        //    3.0 is total on R -- a positive-measure domain extension (a 37-rule family in
        //    (4,3) mining runs before this phase existed).
        assert_eq!(
            fr(
                &["exp", "log", "pow", "x0", "3"],
                &["pow", "x0", "<constant>"]
            ),
            None
        );
        // 2. a domain-PRESERVING constant witness stays minable: exp(log(x)/3) = x^(1/3) on
        //    x > 0, and the fitted 1/3 is far from every snap target -- pow(x, 1/3) is NaN
        //    on x < 0 exactly like the source (a certified live rule).
        assert!(fr(
            &["exp", "/", "log", "x0", "3"],
            &["pow", "x0", "<constant>"]
        )
        .is_some());
        // 3. null-set completion stays allowed (x/x -> 1 at 0: the limit-completion doctrine).
        assert!(fr(&["/", "x0", "x0"], &["1"]).is_some());
        // 4. contract point x = pi/2: f64 sin(pi/2) is EXACTLY 1.0 and pow(1, inf) = 1, not
        //    0 -- clause (a) at a real battery point; the hiprec rescue must not overturn a
        //    contract point (re-evaluating the f64 pi/2 "more precisely" answers a different
        //    question than "what happens at pi/2").
        assert_eq!(fr(&["pow", "sin", "x0", "float(\"inf\")"], &["0"]), None);
        // 5. pow-of-cos in CONSTANT space: pow(cos c, inf) = 1 at c = 0 -- the special
        //    source-constant battery reaches what no random draw does.
        assert_eq!(
            fr(&["pow", "cos", "<constant>", "float(\"inf\")"], &["0"]),
            None
        );
        // 6. deployed-consistency at battery points: atanh(tanh(tan x)) diverges from tan x
        //    by ~3e-7 at x = +-1.5 in f64 (a live (4,3) family) ...
        assert_eq!(fr(&["atanh", "tanh", "tan", "x0"], &["tan", "x0"]), None);
        // ... while atanh(tanh(x)) -> x stays certified (the rescue's flagship identity).
        assert!(fr(&["atanh", "tanh", "x0"], &["x0"]).is_some());
    }

    /// The unresolvable-seam class: a removable
    /// singularity whose cancellation is spelled through DIFFERENT subexpressions cancels
    /// exactly in f64 but leaves a precision-roulette residue at the precision rungs.
    #[test]
    fn seam_class_rejected_stable_null_completion_kept() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let n = 256usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            let t = (r as f64 + 0.5) / n as f64;
            let m = 10f64.powf(-1.0 + 2.0 * t);
            xf.push(if r % 2 == 0 { m } else { -m });
            let u = 10f64.powf(-1.0 + 2.0 * ((r as f64 * 0.37) % 1.0));
            xf.push(if r % 3 == 0 { -u } else { u });
        }
        // (x^3 + x^2 y)/(x + y) -> x^2: nan at (pi/2, -pi/2) in f64 (both spellings of x^3
        // round identically) but residue/0 = inf at 50 decimal digits: a precision seam.
        let seam = s(&[
            "/", "+", "pow", "x0", "3", "*", "pow", "x0", "2", "x1", "+", "x0", "x1",
        ]);
        assert!(!e
            .equivalent_no_const_check(
                &seam,
                &s(&["pow", "x0", "2"]),
                &vars,
                &xf,
                n,
                16,
                1e-9,
                1e-12,
                32,
                7
            )
            .unwrap());
        // (x+y)^3/(x+y)^2 -> x+y: the cancellation is spelled ONCE, cancels exactly at every
        // precision, 0/0 = nan stably -- the ratified null-set-completion class stays.
        let stable = s(&[
            "/", "pow", "+", "x0", "x1", "3", "pow", "+", "x0", "x1", "2",
        ]);
        assert!(e
            .equivalent_no_const_check(
                &stable,
                &s(&["+", "x0", "x1"]),
                &vars,
                &xf,
                n,
                16,
                1e-9,
                1e-12,
                32,
                7
            )
            .unwrap());
    }
}
