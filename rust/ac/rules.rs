//! Rule translation (old prefix grammar -> AC patterns) and the top-down rewrite pass.
//!
//! ## Translation
//!
//! Every rule of the loaded asset is parsed through the SAME boundary converter as subjects
//! ([`super::convert::from_prefix`]) and canonicalized with a BARE context (no certificates).
//! That context performs exactly the collections subject-side canon performs UNCONDITIONALLY,
//! so patterns stay aligned with canonical subjects by construction: `+ _0 _0` becomes
//! `Mul[2, _0]` -- precisely the shape a subject `t + t` always canonicalizes to -- while
//! gated collections (`- !0 !0`, `* $0 inv $0`) are refused without certificates and survive
//! as patterns, exactly as intended.
//!
//! A rule whose translated LHS and RHS are IDENTICAL is dead -- the engine's own arithmetic
//! already performs the rewrite -- and is dropped. This is what removes the 82
//! coefficient-arithmetic rules (`* div2 _0 div2 _1 -> * div4 _0 _1` translates to
//! `Mul[1/4, _0, _1] -> Mul[1/4, _0, _1]`) and every hyper-operator spelling variant: exact
//! rational arithmetic replaces the sampled certificates that used to back them.
//!
//! ## The rewrite pass
//!
//! `apply_rules_top_down`, ported to bags: at every node, try the rules for that root kind in
//! ASSET ORDER (first-match-wins), with the bag rules matching SUB-MULTISETS and preserving the
//! remainder (the widening). If no rule fires, the NUMERIC FOLD fallback runs -- the exact
//! ordering of the shipped engine, which is load-bearing: `sin(pi) -> 0` (a rule) must beat
//! `sin(3.14159...) -> 1.22e-16` (the f64 fold). Then recurse into children, rebuild
//! canonically, and re-enter when the rebuild descends the reduction ordering (else refuse
//! it). Every accepted step -- fire or rebuild -- strictly descends that ordering at its
//! node, and the ordering is WELL-FOUNDED, so termination is a theorem; the hard step cap
//! is defense-in-depth against ordering-invariant bugs (see `ordered_below` and
//! docs/formal.md).

use std::cell::{Cell, RefCell};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::tokens::{Tok, TokenView};

use super::convert::from_prefix;
use super::expr::{add, canon, cmp_ex, complexity, mul, Cx, Ex};
use super::matcher::{match_bag_each, matches_each, substitute, BagKind, Binds, MCx};
use super::rat::Rat;

/// One translated rule. `lhs`/`rhs` are canonical-with-bare-context AC expressions over TABLE
/// token ids only (translation never mints tokens, so storing them per-Engine is safe).
pub struct AcRule {
    pub lhs: Ex,
    pub rhs: Ex,
    /// Bloom signature of the lhs's concrete atoms: a subject that lacks a required atom can
    /// never match (wildcards contribute nothing). False positives only -- a prefilter, never
    /// an authority.
    pub sig: u64,
    /// Does the LHS contain wildcards? EXACT (wildcard-free) rules are tried BEFORE pattern
    /// rules within a bucket -- the shipped engine's protocol (exact lookup first, then the
    /// pattern scan), which desugaring makes LOAD-BEARING: `/ inf inf -> nan` (exact) and
    /// `* _0 inv _0 -> 1` (pattern) lived in different buckets of the binary engine but share
    /// the `Mul` bucket here, and the pattern rule is unsound exactly on the ground poles the
    /// exact rules exist for.
    pub is_pattern: bool,
}

/// The translated rule set, bucketed by root kind. Bucket vectors hold indices into `rules` in
/// ascending = ASSET order (first-match-wins preserved globally within each bucket).
#[derive(Default)]
pub struct AcRules {
    pub rules: Vec<AcRule>,
    /// Provenance, parallel to `rules`: the index of the ARTIFACT rule each served entry
    /// derives from. An asset rule reports its own position; a minted twin reports the
    /// position of the rule it was negated from. This is what lets a consumer that judges
    /// the SERVED set (`Engine::ac_served_rules`) act on the artifact -- a twin has no
    /// artifact row of its own, so a verdict against it is attributed to its source.
    pub src: Vec<usize>,
    pub add_idx: Vec<usize>,
    pub mul_idx: Vec<usize>,
    pub pow_idx: Vec<usize>,
    pub fun_idx: FxHashMap<Tok, Vec<usize>>,
    /// Patterns rooted at a leaf kind (literals, `Const`): matched by whole-node equality.
    pub leaf_idx: Vec<usize>,
    /// Rules dropped as arithmetic-subsumed (translated lhs == rhs) -- the count is reported,
    /// the rules are simply gone.
    pub n_subsumed: usize,
    /// Rules dropped as untranslatable or degenerate (unparseable side, or a bare-wildcard LHS
    /// with a differing RHS -- a catch-all no sound asset contains).
    pub n_dropped: usize,
    /// Orientation twins MINTED at translate (not asset rules): for an admitted rule
    /// `L -> T`, the negated identity `-L -> -T`, added when the flipped LHS absorbs the
    /// sign and every admission gate passes. Counted separately so the asset-rule
    /// accounting (`n_subsumed`/`n_dropped`) keeps its meaning.
    pub n_twins: usize,
    /// Rules refused by the licence registry's load consumer (C1.20, SWEEP row 3): the
    /// rule changes a DEFINED extended-real value at a constructed exceptional point.
    /// The engine pre-filters the raw rules BEFORE translate (the caller sets this
    /// count); 0 for every artifact the mint-side registry produced -- a nonzero count
    /// means a stale or foreign artifact carried a violation past its own mine.
    pub n_registry_dropped: usize,
    /// C36: the PER-REASON drop census -- the six translate-time drop reasons (plus
    /// the registry's, filled by the caller) used to collapse into `n_dropped` with
    /// no accessor, so a user could not find out what was dropped from their own
    /// mine. Keys are stable strings; the FFI exposes this as a dict.
    pub drop_census: Vec<(&'static str, usize)>,
}

impl AcRules {
    /// Count a drop under its reason (and in the aggregate `n_dropped`).
    fn note_drop(&mut self, reason: &'static str) {
        self.n_dropped += 1;
        match self.drop_census.iter_mut().find(|(r, _)| *r == reason) {
            Some((_, n)) => *n += 1,
            None => self.drop_census.push((reason, 1)),
        }
    }
}

/// Fold the concrete atoms of an expression into a 64-bit signature. Wildcard leaves contribute
/// nothing (they bind anything); structure contributes nothing (only atoms are required).
pub fn atom_sig(e: &Ex, view: &TokenView) -> u64 {
    match e {
        Ex::Leaf(t) => {
            if view.sigil(*t) != 0 {
                0
            } else {
                1u64 << (t.0 & 63)
            }
        }
        Ex::Num(r) => {
            let h = (r.num().wrapping_mul(31)).wrapping_add(r.den()) as u64;
            1u64 << (h & 63)
        }
        Ex::Pi => 1u64 << 57,
        Ex::E => 1u64 << 58,
        Ex::PosInf => 1u64 << 59,
        Ex::NegInf => 1u64 << 60,
        Ex::NaN => 1u64 << 61,
        Ex::Const => 1u64 << 62,
        Ex::Fun(f, v) => v
            .iter()
            .fold(1u64 << (f.0 & 63), |acc, x| acc | atom_sig(x, view)),
        Ex::Add(v) | Ex::Mul(v) => v.iter().fold(0, |acc, x| acc | atom_sig(x, view)),
        Ex::Pow(b, ex) => atom_sig(b, view) | atom_sig(ex, view),
    }
}

/// `<constant>` leaf count of an expression (the Const-count invariant's measure; see the
/// translate-time gate below). Counted on canonical sides only.
fn const_count(e: &Ex) -> usize {
    match e {
        Ex::Const => 1,
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().map(const_count).sum(),
        Ex::Pow(b, ex) => const_count(b) + const_count(ex),
        _ => 0,
    }
}

/// Collect the wildcard leaves of an expression. Translation must verify that every RHS
/// wildcard is bound by the CANONICAL lhs -- `substitute` panics on a free one, and canon
/// can ERASE an lhs wildcard a raw containment check would count (an lhs subterm
/// `pow(_1, 0)` folds to `1` unconditionally), so the check runs post-canon.
fn collect_wildcards(e: &Ex, view: &TokenView, out: &mut FxHashSet<Tok>) {
    match e {
        Ex::Leaf(t) => {
            if view.sigil(*t) != 0 {
                out.insert(*t);
            }
        }
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const => {}
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            for x in v {
                collect_wildcards(x, view, out);
            }
        }
        Ex::Pow(b, ex) => {
            collect_wildcards(b, view, out);
            collect_wildcards(ex, view, out);
        }
    }
}

impl AcRules {
    /// Translate a compiled ruleset's raw (asset-order) rules. `view` must be a table-backed
    /// view; translation is read-only on the token side (parse + canon mint no tokens), so the
    /// resulting `Ex`s hold table ids only and are safe to store per-Engine.
    pub fn translate(raw: &[(Vec<Tok>, Vec<Tok>)], view: &TokenView) -> AcRules {
        let cx = Cx::bare(view);
        let mut out = AcRules::default();
        // Inverse-pair guard: the asset contains BOTH directions of some length-neutral
        // canonicalizations (`* float("inf") !0 <-> / float("inf") !0`), which root at
        // DIFFERENT operators in the old tree but at the same `Mul` after desugaring -- a
        // deterministic ping-pong. First-match-wins resolves the direction: the earlier rule
        // owns it, the reversed later rule is dropped.
        let mut seen_pairs: FxHashSet<(Ex, Ex)> = FxHashSet::default();
        for (raw_idx, (lhs_t, rhs_t)) in raw.iter().enumerate() {
            let (Some(lhs0), Some(rhs0)) = (from_prefix(lhs_t, &cx), from_prefix(rhs_t, &cx))
            else {
                out.note_drop("unparseable-side");
                continue;
            };
            let lhs = canon(lhs0, &cx);
            let rhs = canon(rhs0, &cx);
            if lhs == rhs {
                out.n_subsumed += 1;
                continue;
            }
            // The Const-COUNT invariant (its mint twin is the count gate in the candidate
            // scan; one invariant, both layers): a fire may never INCREASE the number of
            // `<constant>` placeholders. The classic violators are the masking-era
            // CONST-INTRODUCING rules (a `<constant>` in the RHS with none in the LHS):
            // they REPLACE an exact literal by an abstract constant
            // (`neg div5 inv !0 -> * <constant> inv !0`). In the old engine they fired on
            // one exact tree shape; widened to sub-multiset matching they would erase exact
            // coefficients everywhere (`(-1/5) * x * log(y) / z` loses its -1/5). Masking is
            // a separate REPRESENTATION pass -- the shipped engine carved it out of simplify
            // for the same reason -- so abstraction-manufacturing rules do not belong in the
            // AC simplifier at all. The count form also refuses the Const-bearing analogue
            // (one slot in the LHS, two in the RHS), which the old boolean gate let through;
            // counts are judged on the CANONICAL sides (canon may absorb Const arithmetic,
            // e.g. `C+C -> C`, changing the spelled count).
            if const_count(&rhs) > const_count(&lhs) {
                out.note_drop("const-count-increase");
                continue;
            }
            // An LHS that canon collapsed to a NON-COMPOSITE (a bare wildcard, leaf, literal or
            // `Const`) is a catch-all: every mined LHS was composite, so the rule's distinctions
            // live below the arithmetic waterline and the arithmetic already owns them.
            if !matches!(&lhs, Ex::Add(_) | Ex::Mul(_) | Ex::Pow(..) | Ex::Fun(..)) {
                out.note_drop("catch-all-lhs");
                continue;
            }
            if seen_pairs.contains(&(rhs.clone(), lhs.clone())) {
                out.note_drop("inverse-of-earlier-rule"); // the exact inverse of an earlier rule: ping-pong fuel
                continue;
            }
            // Sixth drop reason: an RHS wildcard the canonical LHS does not bind would
            // panic `substitute` at rewrite time -- a delayed, data-dependent crash
            // pointing nowhere near the defective rule. Fail closed at load instead.
            let mut lhs_wilds = FxHashSet::default();
            let mut rhs_wilds = FxHashSet::default();
            collect_wildcards(&lhs, view, &mut lhs_wilds);
            collect_wildcards(&rhs, view, &mut rhs_wilds);
            if !rhs_wilds.is_subset(&lhs_wilds) {
                out.note_drop("unbound-rhs-wildcard");
                continue;
            }
            // Seventh drop reason: the Knuth-Bendix orientation, checked statically on
            // the canonical pattern sides (gate G7, docs/formal.md). The complexity
            // functional is not additive, so pattern-level descent neither implies nor is
            // implied by instance-level descent -- the runtime `oriented` gate remains the
            // enforcement per fire; this gate keeps every LOADED rule aligned with the
            // ordering the pass fires under, and re-vets every asset on every load (an
            // order change re-judges the rules automatically).
            if !ordered_below(&rhs, &lhs, view) {
                out.note_drop("not-ordered-below");
                continue;
            }
            seen_pairs.insert((lhs.clone(), rhs.clone()));
            out.src.push(raw_idx);
            let sig = atom_sig(&lhs, view);
            let is_pattern = lhs.contains_wildcard(view);
            let idx = out.rules.len();
            match &lhs {
                Ex::Add(_) => out.add_idx.push(idx),
                Ex::Mul(_) => out.mul_idx.push(idx),
                Ex::Pow(..) => out.pow_idx.push(idx),
                Ex::Fun(f, _) => out.fun_idx.entry(*f).or_default().push(idx),
                _ => out.leaf_idx.push(idx),
            }
            out.rules.push(AcRule {
                lhs,
                rhs,
                sig,
                is_pattern,
            });
        }
        // ORIENTATION CLOSURE (rules fire modulo the sign orientation class): an admitted
        // rule `L -> T` entails `-L -> -T` -- negation is total, so the twin inherits the
        // rule's soundness outright, and sub-multiset semantics carry over term-by-term
        // with the remainder untouched. A sum and its negation are DIFFERENT canonical
        // states (negating redistributes the sign into the term coefficients, so the
        // original Add never survives as a subtree); without the twin a rule only ever
        // met subjects in the orientation it was mined or proposed in --
        // `1 - 2sin^2 x -> cos 2x` fired while `2sin^2 x - 1` (= -cos 2x) stayed
        // unrewritten. With both orientations loaded, matching is complete modulo the
        // class as a property of the DATA: any subject either matches L or its negation
        // does.
        //
        // A twin is minted only when the flipped LHS ABSORBS the sign (a distributed
        // Add, or a Mul coefficient other than -1). When an explicit -1 factor survives,
        // the twin is redundant: sub-multiset matching absorbs the -1 into the remainder
        // (coefficient-free Mul), or the original LHS survives as an intact subtree the
        // source rule already fires on (Fun/Pow roots, kept-primitive sums). The sign
        // symmetry is PARTIAL in this algebra (licence-refused primitives, ordering
        // ties), which is exactly why the closure materializes per-rule through the SAME
        // admission gates instead of quotienting the representation: each twin is
        // individually justified, and an unorientable one simply drops. Twins append
        // AFTER all asset rules (lowest priority within their bucket) and count
        // separately (`n_twins`), so the asset accounting keeps its meaning.
        let mut seen_rules: FxHashSet<(Ex, Ex)> = out
            .rules
            .iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone()))
            .collect();
        let n_raw_kept = out.rules.len();
        for i in 0..n_raw_kept {
            let neg = |e: &Ex| canon(mul(vec![Ex::int(-1), e.clone()], &cx), &cx);
            let tl = neg(&out.rules[i].lhs);
            let absorbed = match &tl {
                Ex::Add(_) => true,
                Ex::Mul(v) => !v
                    .iter()
                    .any(|m| matches!(m, Ex::Num(r) if *r == Rat::NEG_ONE)),
                _ => false,
            };
            if !absorbed {
                continue;
            }
            let tr = neg(&out.rules[i].rhs);
            // The Const gate and the composite gate are inherited (negation introduces
            // no `<constant>`, and an absorbed twin is bag-rooted by construction); the
            // remaining static gates run exactly as for asset rules.
            if tl == tr || seen_rules.contains(&(tl.clone(), tr.clone())) {
                continue; // arithmetic-subsumed, or the asset already carries the twin
            }
            if seen_pairs.contains(&(tr.clone(), tl.clone())) {
                continue; // the exact inverse of a loaded rule: ping-pong fuel
            }
            let mut lhs_wilds = FxHashSet::default();
            let mut rhs_wilds = FxHashSet::default();
            collect_wildcards(&tl, view, &mut lhs_wilds);
            collect_wildcards(&tr, view, &mut rhs_wilds);
            if !rhs_wilds.is_subset(&lhs_wilds) {
                continue;
            }
            // The KB orientation genuinely can differ from the source rule's at
            // complexity ties (the tie-break is a total order, not sign-invariant):
            // a twin that does not orient is dropped, conservative.
            if !ordered_below(&tr, &tl, view) {
                continue;
            }
            let idx = out.rules.len();
            match &tl {
                Ex::Add(_) => out.add_idx.push(idx),
                Ex::Mul(_) => out.mul_idx.push(idx),
                _ => continue, // unreachable given `absorbed`; fail closed
            }
            seen_pairs.insert((tl.clone(), tr.clone()));
            seen_rules.insert((tl.clone(), tr.clone()));
            out.src.push(out.src[i]);
            let sig = atom_sig(&tl, view);
            let is_pattern = tl.contains_wildcard(view);
            out.rules.push(AcRule {
                lhs: tl,
                rhs: tr,
                sig,
                is_pattern,
            });
            out.n_twins += 1;
        }
        // Exact-before-pattern within every bucket (stable: asset order within each class).
        let rules = &out.rules;
        let partition = |v: &mut Vec<usize>| {
            let (exact, pattern): (Vec<usize>, Vec<usize>) =
                v.iter().partition(|&&i| !rules[i].is_pattern);
            v.clear();
            v.extend(exact);
            v.extend(pattern);
        };
        partition(&mut out.add_idx);
        partition(&mut out.mul_idx);
        partition(&mut out.pow_idx);
        partition(&mut out.leaf_idx);
        for v in out.fun_idx.values_mut() {
            partition(v);
        }
        out
    }

    fn bucket_for(&self, e: &Ex) -> &[usize] {
        match e {
            Ex::Add(_) => &self.add_idx,
            Ex::Mul(_) => &self.mul_idx,
            Ex::Pow(..) => &self.pow_idx,
            Ex::Fun(f, _) => self.fun_idx.get(f).map(Vec::as_slice).unwrap_or(&[]),
            _ => &self.leaf_idx,
        }
    }
}

/// Per-pass context for the rewrite walk.
pub struct PassCtx<'a> {
    pub rules: &'a AcRules,
    /// Canonicalization context WITH the engine certificates (substituted RHS instances are
    /// re-canonicalized under the full gates).
    pub cx: &'a Cx<'a>,
    pub mcx: &'a MCx<'a>,
    /// The numeric-fold fallback (engine glue): all-literal evaluation at f64 parity with the
    /// shipped engine, plus the licence-gated `<constant>` collapse. `None` result = no fold.
    pub fold: &'a dyn Fn(&Ex) -> Option<Ex>,
    /// Count of accepted DESCENT STEPS (rule fires and oriented rebuilds) across the whole
    /// call, and the operand of the release-mode step cap (`STEP_CAP`). The well-founded
    /// reduction ordering makes every run finite as a theorem (T6, docs/formal.md); the
    /// cap is defense-in-depth: `rewrite_pass` stops accepting steps -- soundly -- if the
    /// counter ever reaches it. Never observed to bind; a binding cap indicates a bug in
    /// the ordering invariant, not a large input.
    pub fires: Cell<u64>,
    /// Normal-form memo: subtrees this pass already walked to a fixpoint cannot fire and are
    /// returned untouched.
    pub normal: RefCell<FxHashSet<Ex>>,
    /// Composite-step exploration switch (docs/formal.md, the rebuild case of the step
    /// relation). `true` for the engine's pass: a rebuild that fails the immediate
    /// ordering test gets ONE tentative walk from the rebuilt candidate, and commits iff
    /// the settled endpoint is strictly below. `false` inside that tentative walk itself:
    /// one level of exploration is what keeps oscillating reassemblies refused (their
    /// endpoint is the starting node, never strictly below) instead of regressing forever.
    pub explore: bool,
}

/// The REDUCTION ORDERING: is `a` strictly smaller than `b` in the lexicographic pair
/// (mu, canonical total order)? This one predicate is the Knuth-Bendix orientation of
/// the whole system: `translate` drops rules whose canonical pattern sides do not
/// descend it (gate G7, docs/formal.md), and `rewrite_pass` refuses any FIRE or
/// REBUILD whose result does not descend it at the rewritten node (Lemma L2).
///
/// Stage 2 (design/UNIFIED_SIMPLICITY_MEASURE.md): the first component is the unified
/// simplicity measure mu (`ac::expr::complexity`), which ABSORBS the old lit_size
/// middle tier -- mu's literal component IS the bit-length content lit_size carried,
/// so the ordering loses a layer and the dense-literal hazard its own tier existed
/// for (T7: `Mul[3/2^k, x]` now strictly ASCENDS in k instead of sitting at one
/// complexity level).
///
/// The pair is a strict total order (mu is a u64; cmp_ex is total by construction
/// with EXACT literal comparison) that is WELL-FOUNDED: mu can strictly drop only
/// finitely often, and at a FIXED mu value only finitely many terms exist -- mu
/// bounds the node count (every structural node and leaf costs >= 8 except zero-cost
/// magnitude-1 coefficient/exponent slots, of which each bag and pow carries at most
/// one), bounds every literal's bit length (a literal pays its bits), and the
/// vocabulary of `Leaf`/`Fun` tokens is finite -- so the total cmp_ex admits no
/// infinite descent within a level. Hence EVERY descending chain is finite:
/// termination of the pass and of the outer loop are theorems (T6, docs/formal.md),
/// no state is ever revisited (L4), and the step cap and iteration budget are
/// defense-in-depth against implementation bugs, not load-bearing bounds.
pub fn ordered_below(a: &Ex, b: &Ex, view: &TokenView) -> bool {
    match complexity(a, view).cmp(&complexity(b, view)) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => cmp_ex(a, b, view) == std::cmp::Ordering::Less,
    }
}

fn oriented(next: &Ex, node: &Ex, p: &PassCtx) -> bool {
    ordered_below(next, node, p.cx.view)
}

/// Diagnostic: with `SIMPLIPY_AC_TRACE=1` every rule fire is logged (rule index, subject).
fn trace_fires() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_AC_TRACE").as_deref() == Ok("1"))
}

/// Diagnostic: with `SIMPLIPY_AC_MEMO_TRACE=1` every normal-form-memo hit and insert is
/// logged (H-014 instrumentation: the memo is the only walk state that changes
/// rule-attempt control flow, so shielded rule sites show up here and nowhere else).
fn trace_memo() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_AC_MEMO_TRACE").as_deref() == Ok("1"))
}

fn memo_log(what: &str, e: &Ex) {
    let s = format!("{e:?}");
    let t: String = s.chars().take(220).collect();
    eprintln!("AC memo {what}: {t}");
}

/// Try every applicable rule at this node (asset order, sig-prefiltered). On a bag fire the
/// remainder joins the substituted RHS in a fresh canonical bag.
fn try_rules_at(e: &Ex, p: &PassCtx) -> Option<Ex> {
    let node_sig = atom_sig(e, p.cx.view);
    for &ri in p.rules.bucket_for(e) {
        let rule = &p.rules.rules[ri];
        if rule.sig & !node_sig != 0 {
            continue;
        }
        // Acceptance rides the assignment enumeration (finding F1): a disoriented
        // assignment asks the matcher for the NEXT one, so a rule yields to later rules
        // only when NO assignment produces an oriented fire. (Shipped rules have unique
        // assignments, where this is exactly the old fire-then-check behavior.)
        let fired: Option<Ex> = match (&rule.lhs, e) {
            (Ex::Add(pv), Ex::Add(sv)) => {
                let mut binds = Binds::default();
                let mut out: Option<Ex> = None;
                match_bag_each(
                    pv,
                    sv,
                    BagKind::Add,
                    true,
                    &mut binds,
                    p.mcx,
                    &mut |b, used| {
                        let replacement = substitute(&rule.rhs, b, p.cx.view);
                        let mut parts: Vec<Ex> = sv
                            .iter()
                            .zip(used.iter())
                            .filter(|(_, &u)| !u)
                            .map(|(x, _)| x.clone())
                            .collect();
                        parts.push(replacement);
                        let next = add(parts.into_iter().map(|x| canon(x, p.cx)).collect(), p.cx);
                        if oriented(&next, e, p) {
                            out = Some(next);
                            true
                        } else {
                            false
                        }
                    },
                );
                out
            }
            (Ex::Mul(pv), Ex::Mul(sv)) => {
                let mut binds = Binds::default();
                let mut out: Option<Ex> = None;
                match_bag_each(
                    pv,
                    sv,
                    BagKind::Mul,
                    true,
                    &mut binds,
                    p.mcx,
                    &mut |b, used| {
                        let replacement = substitute(&rule.rhs, b, p.cx.view);
                        let mut parts: Vec<Ex> = sv
                            .iter()
                            .zip(used.iter())
                            .filter(|(_, &u)| !u)
                            .map(|(x, _)| x.clone())
                            .collect();
                        parts.push(replacement);
                        let next = mul(parts.into_iter().map(|x| canon(x, p.cx)).collect(), p.cx);
                        if oriented(&next, e, p) {
                            out = Some(next);
                            true
                        } else {
                            false
                        }
                    },
                );
                out
            }
            _ => {
                let mut binds = Binds::default();
                let mut out: Option<Ex> = None;
                matches_each(e, &rule.lhs, &mut binds, p.mcx, &mut |b| {
                    let next = canon(substitute(&rule.rhs, b, p.cx.view), p.cx);
                    if oriented(&next, e, p) {
                        out = Some(next);
                        true
                    } else {
                        false
                    }
                });
                out
            }
        };
        if let Some(next) = fired {
            if trace_fires() {
                eprintln!(
                    "AC fire rule#{ri}: lhs={:?} rhs={:?}\n  subject={e:?}\n  ->={next:?}",
                    rule.lhs, rule.rhs
                );
                debug_assert!(
                    no_nested_bags(&next),
                    "NESTED BAG after rule#{ri}: {next:?}"
                );
            }
            return Some(next);
        }
    }
    None
}

/// Hard bound on accepted descent steps (fires + oriented rebuilds) per simplify call.
/// Since the reduction ordering is well-founded (see `ordered_below`), every run is
/// finite as a THEOREM and this cap is defense-in-depth against bugs in the ordering
/// invariant itself, not a load-bearing bound. Orders of magnitude above any observed
/// run; when it binds, the pass completes descent-free, which is sound.
const STEP_CAP: u64 = 1_000_000;

/// One top-down rewrite pass. The protocol at each node: fire site on the node as-is ->
/// fold fallback -> recurse into children + canonical rebuild -> re-enter on an ORIENTED
/// rebuild (else give the rebuild its one COMPOSITE chance, else refuse it) -> normal
/// form.
///
/// Every state change strictly descends the reduction ordering (`ordered_below`) at this
/// node: fires and rebuilds are gated explicitly (a composite step is gated on its
/// ENDPOINT), and a fold replaces a composite node (complexity >= 2) by a single leaf or
/// `Const` (complexity 1). Consequently the pass returns a result <= its input in the
/// ordering, equal exactly when nothing changed (Lemma L3, docs/formal.md) -- which is
/// what makes the outer loop's fixpoint detection and its no-cycle argument (L4) sound.
///
/// The NORMAL-FORM memo (per simplify call, shared across the outer iterations) marks
/// subtrees walked to a true fixpoint, plus refused-rebuild nodes, which are fixpoints of
/// this pass by decision; marked subtrees return untouched. Marks are NOT placed once the
/// step cap has bound: a capped walk skipped fire sites, so its results are not certified
/// normal forms (the cap is permanent for the call, so this is hygiene of the memo's
/// meaning, not a behavioral necessity).
pub fn rewrite_pass(e: Ex, p: &PassCtx) -> Ex {
    let mut node = e;
    loop {
        if matches!(
            node,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const
        ) {
            return node; // literal leaves: nothing to do
        }
        if p.normal.borrow().contains(&node) {
            if trace_memo() {
                memo_log("HIT", &node);
            }
            return node;
        }

        // First fire site. Every accepted fire strictly descends the reduction ordering at
        // this node (gated inside `try_rules_at`); the step cap bounds the tier the descent
        // argument leaves open (see STEP_CAP).
        if p.fires.get() < STEP_CAP {
            if let Some(next) = try_rules_at(&node, p) {
                p.fires.set(p.fires.get() + 1);
                node = next;
                continue;
            }
        }
        // Fold fallback, GOVERNED BY THE MEASURE (stage 2; owner 2026-07-31, option
        // (a)): the fold fires exactly when its result descends the reduction
        // ordering. The old justification "a fold result is a leaf: always
        // ordering-decreasing" is FALSE under mu -- a literal pays its bits, so
        // folding `exp 1` into a ~105-unit decimal ASCENDS from the 10-unit
        // symbolic state and is refused, while exact arithmetic keeps folding
        // (`+ 2 3 -> 5`, `cos 0 -> 1`: small results are genuinely cheap). This one
        // inequality is what dissolves the symbol/literal seam (design doc finding
        // (b)): `exp(1)` stays symbolic, so no rule can serve on a state whose
        // literal reading differs from its symbolic certificate.
        if let Some(folded) = (p.fold)(&node) {
            if oriented(&folded, &node, p) {
                return folded;
            }
        }

        // Recurse into children and rebuild canonically.
        let rebuilt = match node.clone() {
            Ex::Add(v) => add(v.into_iter().map(|x| rewrite_pass(x, p)).collect(), p.cx),
            Ex::Mul(v) => mul(v.into_iter().map(|x| rewrite_pass(x, p)).collect(), p.cx),
            Ex::Pow(b, ex) => super::expr::pow(rewrite_pass(*b, p), rewrite_pass(*ex, p), p.cx),
            Ex::Fun(f, v) => {
                super::expr::fun(f, v.into_iter().map(|x| rewrite_pass(x, p)).collect(), p.cx)
            }
            leaf => leaf,
        };

        // A rebuild that changed anything must re-enter the full walk: the canonical
        // constructors can MINT new subterms during the rebuild (an integer power
        // distributing over a product, collection merging terms), and only a fresh
        // top-down walk visits them -- a root-only re-check would leave a freshly minted
        // foldable node unvisited and the normal-form memo would lock the incomplete
        // result in. But the re-entry must satisfy the SAME reduction ordering as a rule
        // fire: the canonical RE-ASSEMBLY can re-mint a redex it just consumed, and an
        // unconditional re-entry would climb forever, one level deeper per round. A
        // disoriented rebuild is REFUSED exactly like a disoriented fire: the pre-rebuild
        // node (the smaller element in the ordering) is the pass's answer.
        if rebuilt != node {
            if p.fires.get() < STEP_CAP && oriented(&rebuilt, &node, p) {
                p.fires.set(p.fires.get() + 1);
                node = rebuilt;
                continue;
            }
            // COMPOSITE-STEP acceptance (docs/formal.md, step relation): the reassembly
            // may be an uphill INTERMEDIATE that pays for itself one fold later -- an
            // integer power distributing over a freshly minted product is the canonical
            // case (the distribution costs complexity, the all-literal factor it exposes
            // folds it right back). Judge the SETTLED ENDPOINT, not the intermediate: run
            // the ordinary pass from the rebuilt candidate with exploration DISABLED
            // (one level -- an oscillating reassembly walks back to `node` itself, never
            // strictly below, and is refused without regress) and a PRIVATE memo (its
            // refusal-decision marks must not leak into the engine walk, which explores
            // where the tentative walk may not), and commit iff the endpoint descends.
            // Every COMMITTED composite step strictly descends the ordering at this node,
            // so L2/L3/T6 hold verbatim; a discarded exploration is bounded wasted work.
            if p.explore && p.fires.get() < STEP_CAP {
                let sub = PassCtx {
                    rules: p.rules,
                    cx: p.cx,
                    mcx: p.mcx,
                    fold: p.fold,
                    fires: Cell::new(p.fires.get()),
                    normal: RefCell::new(FxHashSet::default()),
                    explore: false,
                };
                let endpoint = rewrite_pass(rebuilt, &sub);
                p.fires.set(sub.fires.get());
                if p.fires.get() < STEP_CAP && oriented(&endpoint, &node, p) {
                    p.fires.set(p.fires.get() + 1);
                    if trace_fires() {
                        eprintln!(
                            "AC composite step committed:\n  node={node:?}\n  endpoint={endpoint:?}"
                        );
                    }
                    node = endpoint;
                    continue;
                }
                if trace_fires() {
                    eprintln!(
                        "AC rebuild refused (endpoint not below):\n  node={node:?}\n  endpoint={endpoint:?}"
                    );
                }
            } else if trace_fires() {
                eprintln!(
                    "AC rebuild refused (disoriented):\n  node={node:?}\n  rebuilt={rebuilt:?}"
                );
            }
            if p.fires.get() < STEP_CAP {
                if trace_memo() {
                    memo_log("INSERT(refused-rebuild)", &node);
                }
                p.normal.borrow_mut().insert(node.clone());
            }
            return node;
        }

        // A true fixpoint of one full walk: children unchanged, no fire, no fold anywhere.
        if p.fires.get() < STEP_CAP {
            if trace_memo() {
                memo_log("INSERT(fixpoint)", &rebuilt);
            }
            p.normal.borrow_mut().insert(rebuilt.clone());
        }
        return rebuilt;
    }
}

/// Diagnostic: no Add directly inside an Add, no Mul directly inside a Mul.
pub fn no_nested_bags(e: &Ex) -> bool {
    match e {
        Ex::Add(v) => !v.iter().any(|x| matches!(x, Ex::Add(_))) && v.iter().all(no_nested_bags),
        Ex::Mul(v) => !v.iter().any(|x| matches!(x, Ex::Mul(_))) && v.iter().all(no_nested_bags),
        Ex::Pow(b, x) => no_nested_bags(b) && no_nested_bags(x),
        Ex::Fun(_, v) => v.iter().all(no_nested_bags),
        _ => true,
    }
}
