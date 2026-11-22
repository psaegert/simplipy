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
//! `apply_rules_top_down`, ported to bags: at every node, try the rules for that root kind
//! MOST SPECIFIC FIRST (exact rules, then patterns, each by `specificity`, asset order among
//! equals; first-match-wins), with the bag rules matching SUB-MULTISETS and preserving the
//! remainder (the widening). If no rule fires, the NUMERIC FOLD fallback runs -- an
//! ordering that is load-bearing: `sin(pi) -> 0` (a rule) must beat
//! `sin(3.14159...) -> 1.22e-16` (the f64 fold). Then recurse into children, rebuild
//! canonically, and re-enter when the rebuild descends the reduction ordering (else refuse
//! it). Every accepted step -- fire or rebuild -- strictly descends that ordering at its
//! node, and the ordering is WELL-FOUNDED, so termination is a theorem; the hard step cap
//! is a resource limit (a terminating descent can still be long; see `ordered_below` and
//! docs/formal.md).

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::tokens::{Tok, TokenView};

use super::convert::from_prefix;
use super::expr::{
    add, canon, complexity, distribution_licensed, fun, lattice_head, mul, orient_term_classes,
    pattern_distributed, tie_cmp, Cx, Ex, Licences,
};
use super::matcher::{
    match_bag_each_cls, matches_each, substitute, substitute_named, BagKind, Binds, MCx,
};
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
    /// rules within a bucket (exact lookup first, then the pattern scan), which desugaring makes
    /// LOAD-BEARING: `/ inf inf -> nan` (exact) and `* _0 inv _0 -> 1` (pattern) share the `Mul`
    /// bucket, and the pattern rule is unsound exactly on the ground poles the exact rules exist
    /// for.
    pub is_pattern: bool,
    /// Does a wildcard occur more than once in the LHS? Then it binds a class (a sum, an odd
    /// function or odd root of one) up to sign: a sum pattern whose first match met a near miss
    /// (a binding refused by its repeat's negation, `MCx::near_misses`) tries its subject again
    /// with every class in one orientation (`orient_term_classes`). `Repeat::Class` where the repeated
    /// wildcard's sort can bind a class in every mode (`_`, `!`, `$`); `Repeat::Leaf` where only
    /// a pass that binds any subtree (permissive) lets it, `?` binding variable leaves otherwise.
    pub repeat: Repeat,
    /// The licences a binding must meet for the rule to fire (`expr::pattern_distributed`): for
    /// each product of the LHS read distributed, its members and power, which the bound members
    /// must let `pow.distribute` distribute (`expr::distribution_licensed`), as the instance's
    /// own product was. Empty for a rule read as written.
    pub licences: Licences,
}

impl AcRule {
    /// Do the binding's members meet the licences of the reading this rule matches
    /// ([`AcRule::licences`])?
    fn licensed(&self, b: &Binds, cx: &Cx) -> bool {
        self.licences.iter().all(|(members, n)| {
            let bound: Vec<Ex> = members
                .iter()
                .map(|m| canon(substitute(m, b, cx.view), cx))
                .collect();
            distribution_licensed(&bound, n, cx)
        })
    }
}

/// Whether, and through which wildcard sorts, an LHS repeats a wildcard (`AcRule::repeat`).
#[derive(Clone, Copy, PartialEq)]
pub enum Repeat {
    No,
    Leaf,
    Class,
}

/// How `lhs` repeats a wildcard (`AcRule::repeat`).
fn repeated_wildcard(lhs: &Ex, view: &TokenView) -> Repeat {
    let mut ws = FxHashSet::default();
    collect_wildcards(lhs, view, &mut ws);
    let mut out = Repeat::No;
    for &w in ws.iter().filter(|&&w| leaf_count(lhs, w) > 1) {
        if view.sigil(w) == b'?' {
            out = Repeat::Leaf;
        } else {
            return Repeat::Class;
        }
    }
    out
}

/// The translated rule set, bucketed by root kind. Bucket vectors hold indices into `rules` in
/// FIRE order: exact rules before patterns, each class most specific first (`specificity`),
/// ascending = ASSET order among equals.
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
    /// O(1) EXACT dispatch (exact lookup first): canonical wildcard-free LHS -> rule indices
    /// in asset order, for NON-BAG
    /// roots only. Sound because a wildcard-free pattern at a non-bag root matches a
    /// canonical subject iff it structurally equals it: every `matches_each` arm without a
    /// wildcard is strict equality, and nested bags match full-cover as multisets, which for
    /// two bags sorted by the total order `cmp_ex` is vector equality. Bag ROOTS (`Add`/
    /// `Mul`) match sub-multisets with a remainder, so they stay on the linear scan.
    pub exact_map: FxHashMap<Ex, Vec<usize>>,
    /// Partition points: `bucket[..n]` is the exact head (all wildcard-free), `bucket[n..]`
    /// the pattern tail. Only meaningful for the non-bag buckets the exact map serves.
    pub pow_exact_end: usize,
    pub leaf_exact_end: usize,
    pub fun_exact_end: FxHashMap<Tok, usize>,
    /// Rules dropped as arithmetic-subsumed (translated lhs == rhs) -- the count is reported,
    /// the rules are simply gone.
    pub n_subsumed: usize,
    /// Rules dropped as untranslatable or degenerate (unparseable side, or a bare-wildcard LHS
    /// with a differing RHS -- a catch-all no sound asset contains).
    pub n_dropped: usize,
    /// Orientation twins MINTED at translate (not asset rules): for an admitted rule
    /// `L -> T`, the negated identity `-L -> -T`, added when the flipped LHS absorbs the
    /// sign and every admission gate passes. Counted separately so the asset-rule accounting
    /// (`n_subsumed`/`n_dropped`) keeps its meaning.
    pub n_twins: usize,
    /// Distributed readings LOADED at translate beside the rule they read (not asset rules): a
    /// rule whose LHS holds a product under an odd negative power that a slot keeps whole is
    /// also read with that product distributed, matched where the binding licenses it
    /// (`expr::pattern_distributed`). Counted apart, so the asset accounting keeps its meaning.
    pub n_distributed: usize,
    /// Rules refused by the licence registry's load consumer (SWEEP row 3): the
    /// rule changes a DEFINED extended-real value at a constructed exceptional point.
    /// The engine pre-filters the raw rules BEFORE translate (the caller sets this
    /// count); 0 for every artifact the mint-side registry produced -- a nonzero count
    /// means a stale or foreign artifact carried a violation past its own mine.
    pub n_registry_dropped: usize,
    /// The PER-REASON drop census -- the translate-time drop reasons (plus
    /// the registry's, filled by the caller) used to collapse into `n_dropped` with
    /// no accessor, so a user could not find out what was dropped from their own
    /// mine. Keys are stable strings; the FFI exposes this as a dict.
    pub drop_census: Vec<(&'static str, usize)>,
    /// The SHADOW CENSUS: every translate-time skip caused by CROSS-RULE state, i.e.
    /// every entry that would (re)appear if some OTHER artifact row were removed from
    /// the input. `(victim_row, owner_row, owner_was_twin, kind)`, with one kind,
    /// `"twin-shadow"`: the orientation twin of `victim_row` skipped because
    /// `owner_row`'s entry (its asset rule, or its own twin when `owner_was_twin`)
    /// already carries the identical pair (`seen_rules`).
    ///
    /// No other cross-rule skip exists. The reverse of a loaded rule (asset or twin) never
    /// loads, but by gate G7 alone (`ordered_below` is a strict total order, so a pair and
    /// its reverse cannot both descend it), and removing the owner resurrects nothing.
    ///
    /// Twin victims are recorded only when every OTHER admission gate passes, so an
    /// entry states "removing the owner resurrects this rewrite", not merely "a hash
    /// collided". This is what lets a consumer that needs the engine WITHOUT one rule
    /// (the promotion refund's derivability probe) know for which rows fire-site
    /// suppression alone is NOT behaviorally a rebuild.
    pub shadow_census: Vec<(usize, usize, bool, &'static str)>,
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
            let h = match r.small_parts() {
                Some((p, q)) => (p.wrapping_mul(31)).wrapping_add(q) as u64,
                None => {
                    use std::hash::{Hash, Hasher};
                    let mut s = std::collections::hash_map::DefaultHasher::new();
                    r.hash(&mut s);
                    s.finish()
                }
            };
            1u64 << (h & 63)
        }
        Ex::Pi => 1u64 << 57,
        Ex::E => 1u64 << 58,
        Ex::PosInf => 1u64 << 59,
        Ex::NegInf => 1u64 << 60,
        Ex::NaN => 1u64 << 61,
        Ex::Const(_) => 1u64 << 62,
        Ex::Fun(f, v) => v
            .iter()
            .fold(1u64 << (f.0 & 63), |acc, x| acc | atom_sig(x, view)),
        Ex::Add(v) | Ex::Mul(v) => v.iter().fold(0, |acc, x| acc | atom_sig(x, view)),
        Ex::Pow(b, ex) => atom_sig(b, view) | atom_sig(ex, view),
    }
}

/// `<constant>` leaf count of an expression (the Const-count invariant's measure; see the
/// translate-time gates below).
fn const_count(e: &Ex) -> usize {
    match e {
        Ex::Const(_) => 1,
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
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const(_) => {}
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

/// The SPECIFICITY of a canonical LHS, the rule's priority at its root: its node count less
/// its distinct wildcards -- the nodes a match must find beyond one binding per wildcard.
/// Every proper syntactic instance of a pattern is more specific than it: binding a wildcard
/// to a leaf, to a larger subtree, or to another wildcard of the pattern each raises it by
/// at least one. (Narrowing a wildcard's sort keeps it; asset order decides those.) The
/// legacy engine scanned the longest pattern first for the same reason; the count is taken
/// on the canonical side, the shape the matcher meets.
fn specificity(lhs: &Ex, view: &TokenView) -> usize {
    fn nodes(e: &Ex) -> usize {
        match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => 1 + v.iter().map(nodes).sum::<usize>(),
            Ex::Pow(b, x) => 1 + nodes(b) + nodes(x),
            _ => 1,
        }
    }
    let mut ws = FxHashSet::default();
    collect_wildcards(lhs, view, &mut ws);
    nodes(lhs) - ws.len()
}

/// How often the leaf `tok` occurs in `e`.
fn leaf_count(e: &Ex, tok: Tok) -> usize {
    match e {
        Ex::Leaf(t) => usize::from(*t == tok),
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().map(|x| leaf_count(x, tok)).sum(),
        Ex::Pow(b, ex) => leaf_count(b, tok) + leaf_count(ex, tok),
        _ => 0,
    }
}

/// The context a rule side of `mode` is READ in: that mode's fold discipline and number
/// semantics, without certificates. A mode's rules are read the way the mode reads its
/// inputs, so an f64 pattern carrying a ground subtree f64 folds is folded here too and can
/// match; the miner keys its classes and reads its sources in the same context.
///
/// A PATTERN IS NOT ITS INSTANCES. What the mode's canon does to a ground subtree it does
/// to every instance, so folds apply. A licence that passes a gate uncertified does not:
/// permissive cancels `/ $0 $0` on the pattern, but its instance `inf / inf` is NaN (a fully
/// numeric part is read for its value first). So permissive's rule sides are read with
/// certified licences (its fold discipline is the bare one).
pub(crate) fn rule_cx<'a>(view: &'a TokenView<'a>, mode: crate::engine::RuleMode) -> Cx<'a> {
    let mut cx = Cx::bare(view);
    if !mode.wildcard_all() {
        cx.mode = mode;
    }
    cx.fold_f64 = Cx::folds_for(mode);
    cx
}

/// THE CANON A RULE SET WAS CERTIFIED IN, and so the one it is read in. A rule is judged on
/// the pattern its mine wrote, read in its mine's canon; read in another canon, its LHS can
/// become a different pattern, which nothing judged. A canon that cancels `atanh(tanh(t))`
/// reads the f64 fact `rootn 2 atanh tanh 3 -> nan` (f64's `atanh(tanh(3))` is not 3) as
/// `rootn 2 3 -> nan`, which is false. Read in the canon it was mined in, the pattern stays the
/// one judged, and a pattern the serving canon never produces never fires.
///
/// Numerals are read in the SERVING mode's number domain whatever the canon (a numeral
/// denotes the number it spells); the canon decides the structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleCanon {
    /// A per-mode mine of this mode (0.15 on), and every set handed over in memory: the
    /// mode's own reading ([`rule_cx`]).
    Mode(crate::engine::RuleMode),
    /// A mine from before per-mode mines (0.14 and earlier): one mode-blind canon for all
    /// its files, Default's, unfolded and uncertified ([`Cx::bare`]), the reading every
    /// file was loaded in until per-mode mines.
    Legacy,
}

impl RuleCanon {
    /// The context a rule side of this canon is read in.
    pub(crate) fn cx<'a>(self, view: &'a TokenView<'a>) -> Cx<'a> {
        match self {
            RuleCanon::Mode(mode) => rule_cx(view, mode),
            RuleCanon::Legacy => Cx::bare(view),
        }
    }

    /// The wire spelling: `"legacy"`, or a rule mode's (`RuleMode::parse`).
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "legacy" => Some(RuleCanon::Legacy),
            other => crate::engine::RuleMode::parse(other).map(RuleCanon::Mode),
        }
    }

    /// The wire spelling [`RuleCanon::parse`] reads back.
    pub fn name(self) -> &'static str {
        use crate::engine::RuleMode;
        match self {
            RuleCanon::Legacy => "legacy",
            RuleCanon::Mode(RuleMode::Default) => "default",
            RuleCanon::Mode(RuleMode::Permissive) => "permissive",
        }
    }
}

/// THE ENGINE-OWNED LAW (gate G8), one definition for the rule load and the miner: does the
/// rule `lhs -> rhs` (canonical sides) state a law the engine itself performs? The distributive
/// law as the search applies it (one FACTOR move, `search::is_factoring`), and a zero factor
/// absorbing the product (`mul.zero`; in `permissive` without a definedness check, the step
/// `Permissive::ZeroAbsorbs`): a rule whose right side is 0 and whose left side is a product
/// holding a literal 0, where the constructors read both sides' instances to one state, so
/// `* 0 !0 -> 0` and `/ 0 $0 -> 0` in `permissive`, though their patterns, whose slot may stand
/// for what no instance spells, are not read to `0`. Strict keeps a product float64 gives no
/// value, so no such rule is its own there. A rule stating a like-term cancellation
/// (`- !0 !0 -> 0`, `/ $0 $0 -> 1`) stays a rule: the walk takes it where the constructors'
/// merge does not, and answers change without it.
/// The load drops such a rule ("engine-owned-law"); the miner refuses to mint it
/// (`Engine::law_instance`, the miner's acceptance). A law the constructors perform on the
/// pattern itself, the collection of powers of e (`exp(a) exp(b) = exp(a + b)`) among them, has
/// one canonical state on both written sides, which the load refuses as subsumed before this gate
/// and `Engine::law_instance` names `canon`.
///
/// The search meets a rule's INSTANCES, so the law is judged on one: every `?`, `!` and `$`
/// wildcard becomes a variable of its own ([`pattern_instance`]), and the sides are read in
/// `law`, the serving search's licence context (`Engine::with_law_cx`), whose certificates
/// answer for every binding the sorts admit. The FACTOR move is then the search's own.
pub(crate) fn engine_owned_law(lhs: &Ex, rhs: &Ex, law: &Cx) -> bool {
    let mut fresh = FxHashMap::default();
    let l = canon(pattern_instance(lhs, law.view, &mut fresh), law);
    let r = canon(pattern_instance(rhs, law.view, &mut fresh), law);
    let zero = |e: &Ex| matches!(e, Ex::Num(z) if z.is_zero());
    let zero_absorbs = zero(rhs) && matches!(lhs, Ex::Mul(v) if v.iter().any(zero));
    (zero_absorbs && l == r) || super::search::is_factoring(&l, &r, law)
}

/// The spelling prefix of the variables [`pattern_instance`] puts in place of wildcards: the
/// wildcard's own spelling follows it (`PATTERN_VAR` + `?0`). No token of a table spells it.
pub(crate) const PATTERN_VAR: &str = "\u{1}";

/// `e` with each `?`, `!` and `$` wildcard replaced by a variable leaf of its own, minted in
/// `view`'s overlay (`fresh` keeps one per wildcard across both sides). A `?` binds a variable,
/// so this is its instance up to renaming; `!` and `$` bind subjects that carry their sort's
/// certificate, which the licence context's certificates read off the spelling. `_` binds any
/// subtree, so it stays a wildcard and no licence holds for it.
fn pattern_instance(e: &Ex, view: &TokenView, fresh: &mut FxHashMap<Tok, Tok>) -> Ex {
    match e {
        Ex::Leaf(t) if matches!(view.sigil(*t), b'?' | b'!' | b'$') => {
            Ex::Leaf(*fresh.entry(*t).or_insert_with(|| {
                view.intern(&format!("{PATTERN_VAR}{}", view.resolve_owned(*t)))
            }))
        }
        Ex::Add(v) => Ex::Add(v.iter().map(|x| pattern_instance(x, view, fresh)).collect()),
        Ex::Mul(v) => Ex::Mul(v.iter().map(|x| pattern_instance(x, view, fresh)).collect()),
        Ex::Fun(f, v) => Ex::Fun(
            *f,
            v.iter().map(|x| pattern_instance(x, view, fresh)).collect(),
        ),
        Ex::Pow(b, x) => Ex::Pow(
            Box::new(pattern_instance(b, view, fresh)),
            Box::new(pattern_instance(x, view, fresh)),
        ),
        other => other.clone(),
    }
}

impl AcRules {
    /// Translate a compiled ruleset's raw (asset-order) rules, read in the canon they were
    /// certified in ([`RuleCanon`]); `law` is the serving search's licence context, which
    /// gate G8 judges in ([`engine_owned_law`]). `view` must be a table-backed
    /// view; translation is read-only on the token side (parse + canon mint no tokens), so the
    /// resulting `Ex`s hold table ids only and are safe to store per-Engine.
    pub fn translate(
        raw: &[(Vec<Tok>, Vec<Tok>)],
        view: &TokenView,
        read_in: RuleCanon,
        law: &Cx,
    ) -> AcRules {
        let cx = read_in.cx(view);
        let mut out = AcRules::default();
        for (raw_idx, (lhs_t, rhs_t)) in raw.iter().enumerate() {
            let (Some(lhs0), Some(rhs0)) = (from_prefix(lhs_t, &cx), from_prefix(rhs_t, &cx))
            else {
                out.note_drop("unparseable-side");
                continue;
            };
            // The ground-composite policy (the specification's 4.3): a right side without a
            // variable is one leaf. `asin(1) -> acos(0)` holds and is refused. The load checks
            // policies, never truth.
            let rhs_tokens: Vec<String> = rhs_t.iter().map(|t| view.resolve_owned(*t)).collect();
            if crate::engine::refusals::ground_composite_rule(&rhs_tokens) {
                out.note_drop("ground-composite");
                continue;
            }
            // A left side may hold several `<constant>`s: the matcher binds each to a value of
            // its own, and the judge reads each as its own symbol (the specification's 3.2).
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
            // (`neg div5 inv !0 -> * <constant> inv !0`). Under sub-multiset matching they
            // would erase exact coefficients everywhere (`(-1/5) * x * log(y) / z` loses its
            // -1/5). Masking is a separate REPRESENTATION pass, so abstraction-manufacturing
            // rules do not belong in the AC simplifier at all. The count form also refuses the
            // Const-bearing analogue (one slot in the LHS, two in the RHS); counts are judged
            // on the CANONICAL sides (canon may absorb Const arithmetic, e.g. `C+C -> C`,
            // changing the spelled count).
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
            // Seventh drop reason: an uphill rule, checked statically on the canonical
            // pattern sides by price alone (gate G7, docs/formal.md). The complexity
            // functional is not additive, so pattern-level descent neither implies nor is
            // implied by instance-level descent -- the runtime `oriented` gate is the
            // enforcement per fire, under the full ordering. Two sides of one price are
            // therefore no orientation: the rule loads, and each application fires only where
            // it descends (a literal instance that is cheaper, or the sign key's order on a
            // tie). Both directions of such a rewrite may load (`* float("inf") !0 <-> / float("inf")
            // !0` roots at the same `Mul` after desugaring); the order is strict and total, so
            // on any instance at most one of them descends and fires, and neither undoes the
            // other. The gate re-vets every asset on every load.
            if complexity(&rhs, view) > complexity(&lhs, view) {
                out.note_drop("not-ordered-below");
                continue;
            }
            // Eighth drop reason: the rule states the distributive law, which the search owns
            // (one FACTOR move turns its LHS into its RHS, `search::is_factoring`): judged by
            // the law, not by spelling. The search proposes every factoring such a rule makes,
            // under whole-state acceptance; in the chain the rule would take the first factoring
            // that lowers its own node, undo the search's DISTRIBUTE candidates, and pre-empt
            // the other factorings and the merges they enable.
            if engine_owned_law(&lhs, &rhs, law) {
                out.note_drop("engine-owned-law");
                continue;
            }
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
            let repeat = repeated_wildcard(&lhs, view);
            // A product under an odd negative power that a slot keeps whole is read distributed
            // too, matched where the binding licenses the distribution (`pow.distribute` at bind
            // time): `1/(x1*?0) -> t` fires on `1/(x1*x3)`, whose product distributes.
            let distributed = pattern_distributed(&lhs, &cx).filter(|(d, _)| {
                *d != lhs && *d != rhs && complexity(&rhs, view) <= complexity(d, view)
            });
            out.rules.push(AcRule {
                lhs,
                rhs: rhs.clone(),
                sig,
                is_pattern,
                repeat,
                licences: Vec::new(),
            });
            if let Some((d, licences)) = distributed {
                let idx = out.rules.len();
                match &d {
                    Ex::Add(_) => out.add_idx.push(idx),
                    Ex::Mul(_) => out.mul_idx.push(idx),
                    Ex::Pow(..) => out.pow_idx.push(idx),
                    Ex::Fun(f, _) => out.fun_idx.entry(*f).or_default().push(idx),
                    _ => out.leaf_idx.push(idx),
                }
                out.src.push(raw_idx);
                out.rules.push(AcRule {
                    sig: atom_sig(&d, view),
                    is_pattern: true,
                    repeat: repeated_wildcard(&d, view),
                    lhs: d,
                    rhs,
                    licences,
                });
                out.n_distributed += 1;
            }
        }
        // ORIENTATION CLOSURE (rules fire modulo the sign orientation class,
        // `settle.negation_twin`): an admitted
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
        // AFTER all asset rules (last among the rules of their specificity in their
        // bucket) and count separately (`n_twins`), so the asset accounting keeps its
        // meaning.
        let mut seen_rules: FxHashMap<(Ex, Ex), (usize, bool)> = FxHashMap::default();
        for (r, &row) in out.rules.iter().zip(out.src.iter()) {
            seen_rules
                .entry((r.lhs.clone(), r.rhs.clone()))
                .or_insert((row, false));
        }
        let n_raw_kept = out.rules.len();
        for i in 0..n_raw_kept {
            // A fully numeric rule has no orientation class to close: its one instance is
            // itself, and the negation of it is another fully numeric part, whose value the
            // ground evaluator reads before any rule (in `f64` the zero of `0*tan(4)` and of its
            // negation are two values).
            let lhs = &out.rules[i].lhs;
            if !lhs.has_measure_space(view) && !lhs.contains_wildcard(view) {
                continue;
            }
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
            if tl == tr {
                continue; // arithmetic-subsumed
            }
            // The PER-TWIN gates run before the cross-rule guards (they are pure checks,
            // so the reordering cannot change which twins mint): a twin the gates refuse
            // resurrects under NO removal, and must not enter the shadow census.
            let mut lhs_wilds = FxHashSet::default();
            let mut rhs_wilds = FxHashSet::default();
            collect_wildcards(&tl, view, &mut lhs_wilds);
            collect_wildcards(&tr, view, &mut rhs_wilds);
            if !rhs_wilds.is_subset(&lhs_wilds) {
                continue;
            }
            // The twin's price can differ from the source rule's (negating redistributes
            // signs into coefficients): an uphill twin is dropped, as G7 drops a rule.
            if complexity(&tr, view) > complexity(&tl, view) {
                continue;
            }
            if let Some(&(owner, owner_was_twin)) = seen_rules.get(&(tl.clone(), tr.clone())) {
                // The asset (or an earlier twin) already carries this exact rewrite;
                // removing its owner would mint this twin instead -- census.
                out.shadow_census
                    .push((out.src[i], owner, owner_was_twin, "twin-shadow"));
                continue;
            }
            let idx = out.rules.len();
            match &tl {
                Ex::Add(_) => out.add_idx.push(idx),
                Ex::Mul(_) => out.mul_idx.push(idx),
                _ => continue, // unreachable given `absorbed`; fail closed
            }
            seen_rules
                .entry((tl.clone(), tr.clone()))
                .or_insert((out.src[i], true));
            out.src.push(out.src[i]);
            let sig = atom_sig(&tl, view);
            let is_pattern = tl.contains_wildcard(view);
            let repeat = repeated_wildcard(&tl, view);
            let licences = out.rules[i].licences.clone();
            out.rules.push(AcRule {
                lhs: tl,
                rhs: tr,
                sig,
                is_pattern,
                repeat,
                licences,
            });
            out.n_twins += 1;
        }
        // Exact-before-pattern within every bucket, and MOST SPECIFIC FIRST within each class
        // (`specificity`; stable, so asset order decides among equals). First-match-wins
        // otherwise lets a general rule take the instances a more specific rule was mined
        // for: with `sin acos _0 -> cos asin _0` first in the file, `sin acos tanh _0 ->
        // inv cosh _0` never fired, and `sin acos tanh x` ended at `cos asin tanh x`. The
        // order is a function of each rule alone, so a set without some rows orders the
        // survivors as this one does (the suppression probes rely on it).
        let rules = &out.rules;
        let spec: Vec<usize> = rules.iter().map(|r| specificity(&r.lhs, view)).collect();
        let partition = |v: &mut Vec<usize>| {
            let (mut exact, mut pattern): (Vec<usize>, Vec<usize>) =
                v.iter().partition(|&&i| !rules[i].is_pattern);
            exact.sort_by_key(|&i| std::cmp::Reverse(spec[i]));
            pattern.sort_by_key(|&i| std::cmp::Reverse(spec[i]));
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
        // Exact-dispatch map over the non-bag buckets' exact heads (asset order preserved:
        // bucket vectors are ascending, so same-LHS twins land in the map in fire order).
        out.pow_exact_end = out
            .pow_idx
            .iter()
            .take_while(|&&i| !rules[i].is_pattern)
            .count();
        out.leaf_exact_end = out
            .leaf_idx
            .iter()
            .take_while(|&&i| !rules[i].is_pattern)
            .count();
        let mut fun_exact_end: FxHashMap<Tok, usize> = FxHashMap::default();
        for (f, v) in &out.fun_idx {
            fun_exact_end.insert(*f, v.iter().take_while(|&&i| !rules[i].is_pattern).count());
        }
        let mut exact_map: FxHashMap<Ex, Vec<usize>> = FxHashMap::default();
        {
            let mut insert = |idx: &[usize]| {
                for &i in idx {
                    exact_map.entry(rules[i].lhs.clone()).or_default().push(i);
                }
            };
            insert(&out.pow_idx[..out.pow_exact_end]);
            insert(&out.leaf_idx[..out.leaf_exact_end]);
            for (f, v) in &out.fun_idx {
                insert(&v[..fun_exact_end[f]]);
            }
        }
        out.fun_exact_end = fun_exact_end;
        out.exact_map = exact_map;
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

/// A step's admission on a fully numeric node (`PassCtx::ground_step`): `(node, result)`.
pub type GroundStep<'a> = &'a dyn Fn(&Ex, &Ex) -> bool;

/// THE NORMAL-FORM MEMO'S KEYS: each distinct subtree a pass meets is one id, made from its own head
/// and its children's ids, so two subtrees share an id exactly when they are equal (`Ex`'s
/// structural equality, a number by its exact value) and the memo stores ids, not copies: a chain
/// of depth `d` costs memory linear in `d` (a copy per node cost `d^2/2` nodes).
#[derive(Default)]
pub struct NodeIds {
    ids: FxHashMap<Box<[u32]>, u32>,
    nums: FxHashMap<Rat, u32>,
    stack: Vec<u32>,
}

impl NodeIds {
    /// The id of `e`, given the first time `e`'s structure is met.
    pub fn id(&mut self, e: &Ex) -> u32 {
        let base = self.stack.len();
        match e {
            Ex::Num(r) => {
                let n = self.nums.len() as u32;
                let k = *self.nums.entry(r.clone()).or_insert(n);
                self.stack.extend([0, k]);
            }
            Ex::Pi => self.stack.push(1),
            Ex::E => self.stack.push(2),
            Ex::PosInf => self.stack.push(3),
            Ex::NegInf => self.stack.push(4),
            Ex::NaN => self.stack.push(5),
            Ex::Const(_) => self.stack.push(6),
            Ex::Leaf(t) => self.stack.extend([7, t.0]),
            Ex::Add(v) | Ex::Mul(v) => {
                self.stack.push(if matches!(e, Ex::Add(_)) { 8 } else { 9 });
                self.stack.push(v.len() as u32);
                for c in v {
                    let i = self.id(c);
                    self.stack.push(i);
                }
            }
            Ex::Pow(b, x) => {
                self.stack.push(10);
                let i = self.id(b);
                self.stack.push(i);
                let j = self.id(x);
                self.stack.push(j);
            }
            Ex::Fun(f, v) => {
                self.stack.extend([11, f.0, v.len() as u32]);
                for c in v {
                    let i = self.id(c);
                    self.stack.push(i);
                }
            }
        }
        let key = &self.stack[base..];
        let id = match self.ids.get(key) {
            Some(&i) => i,
            None => {
                let n = self.ids.len() as u32;
                self.ids.insert(key.into(), n);
                n
            }
        };
        self.stack.truncate(base);
        id
    }
}

/// Per-pass context for the rewrite walk.
pub struct PassCtx<'a> {
    pub rules: &'a AcRules,
    /// Canonicalization context WITH the engine certificates (substituted RHS instances are
    /// re-canonicalized under the full gates).
    pub cx: &'a Cx<'a>,
    pub mcx: &'a MCx<'a>,
    /// The numeric-fold fallback (engine glue): all-literal evaluation in float64, plus the
    /// licence-gated `<constant>` collapse. `None` result = no fold.
    pub fold: &'a dyn Fn(&Ex) -> Option<Ex>,
    /// Count of accepted DESCENT STEPS (rule fires and oriented rebuilds) across the whole
    /// call, and the operand of the release-mode step cap (`STEP_CAP`). The well-founded
    /// reduction ordering makes every run finite as a theorem (T6, docs/formal.md), not short:
    /// the cap is a resource limit, and `rewrite_pass` stops accepting steps -- soundly -- if
    /// the counter reaches it, and the call says it was interrupted ([`take_caps`]).
    pub fires: Cell<u64>,
    /// Normal-form memo: subtrees this pass already walked to a fixpoint cannot fire and are
    /// returned untouched. Keyed by the subtree's id ([`NodeIds`]), so it holds no copy of one.
    pub normal: RefCell<FxHashSet<u32>>,
    /// How many refusals the walk met so far (`visit`): a fire or a rebuild that does not descend
    /// at its node is taken where the whole expression's price falls, which reads the ancestors,
    /// so a node above a refusal is normal only in its place as it stands (`normal_at`).
    pub contextual: Cell<u64>,
    /// The memo of nodes normal in their place: a node above a refusal, with the walk's `epoch`
    /// when it was marked. It is normal while nothing changed since; after a change it is walked
    /// again, its refusals reading the ancestors as they then stand.
    pub normal_at: RefCell<FxHashMap<u32, u64>>,
    /// The ids of the subtrees the memo has met ([`NodeIds`]).
    pub ids: RefCell<NodeIds>,
    /// How many changes the walk made so far (steps, folds and carried steps), in the whole call.
    pub epoch: Cell<u64>,
    /// THE COMPOSITE STEP, for the search: rebuilds the settling refused, as `(node, rebuilt)`,
    /// the first `REFUSED_KEPT` of the call. A rebuild that rises at its node and in the whole
    /// can still pay once the rebuilt node settles (an integer power distributing over a freshly
    /// minted product whose literal factor then folds); the search proposes the rebuilt node at
    /// every place of its state that holds the node, and keeps it where the whole state's
    /// descent lands below (`ac::search`).
    pub refused: RefCell<Vec<(Ex, Ex)>>,
    /// SUPPRESSED ARTIFACT ROWS (`AcRules::src` values), checked at the single fire site
    /// (`try_rules_at`): a suppressed rule and every orientation twin minted from it (the
    /// twin reports its source's row) never fires, so the walk is BEHAVIORALLY the walk
    /// of an engine translated without those artifact rows -- translation is per-rule,
    /// bucket order filtered in place is the order a rebuild would give the survivors,
    /// and constructors/certificates/folds never consult the rule set. `None` (every
    /// pre-existing entry point) is byte-identical to the field not existing. Consumer:
    /// the promotion refund's per-candidate "engine WITHOUT this rule" probes.
    pub suppressed: Option<&'a FxHashSet<usize>>,
    /// THE GROUND EVALUATOR's admission of a step on a FULLY NUMERIC node (`ground`): may the
    /// step's result stand for the node in the mode? A rule changes no ground value, so a rule
    /// fires on a ground node only where the evaluator admits its result (a ground rule of a file
    /// verify-mine passed always is); and a ground node rebuilt after its operands moved is
    /// judged whole, so a step inside a fully numeric part is admitted for the part, not only for
    /// the operand (`1/sin(np.pi)` keeps float64's 8.2e15: `sin(np.pi)` is 0 alone, but `1/0` is
    /// no value of the whole). `None`: every oriented step is taken.
    pub ground_step: Option<GroundStep<'a>>,
}

/// The REDUCTION ORDERING: is `a` strictly smaller than `b` in the lexicographic order
/// (mu, the sign key, canonical total order -- `tie_cmp`)? This one predicate orients the
/// whole system: `rewrite_pass` takes a FIRE or a REBUILD where its result descends it at the
/// rewritten node (Lemma L2), or where the whole expression's does (`whole_falls`), and the
/// search accepts only states below its best;
/// `translate` drops only rules whose canonical pattern sides climb in mu (gate G7,
/// docs/formal.md).
///
/// Stage 2 (design/UNIFIED_SIMPLICITY_MEASURE.md): the first component is the unified
/// simplicity measure mu (`ac::expr::complexity`), whose literal component prices a literal
/// as it is written, so the ordering needs no separate literal tier. The dense-literal chain
/// (`Mul[3/2^k, x]`) ascends in k; it stays finite through the 1,100-bit cap (below).
///
/// The pair is a strict total order (mu is a u64; cmp_ex is total by construction
/// with EXACT literal comparison) that is WELL-FOUNDED: mu can strictly drop only
/// finitely often, and at a FIXED mu value only finitely many terms exist -- mu
/// bounds the node count (every structural node and leaf costs >= 8 except zero-cost
/// magnitude-1 coefficient/exponent slots, of which each bag and pow carries at most
/// one), literals are finitely many under the 1,100-bit cap (and a literal's price, its
/// written spelling, bounds its digits), and the
/// vocabulary of `Leaf`/`Fun` tokens is finite -- so the total cmp_ex admits no
/// infinite descent within a level. Hence EVERY descending chain is finite:
/// termination of the pass and of the outer loop are theorems (T6, docs/formal.md),
/// no state is ever revisited (L4), and the step cap and iteration budget are
/// defense-in-depth against implementation bugs, not load-bearing bounds.
pub fn ordered_below(a: &Ex, b: &Ex, view: &TokenView) -> bool {
    match complexity(a, view).cmp(&complexity(b, view)) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => tie_cmp(a, b, view) == std::cmp::Ordering::Less,
    }
}

fn oriented(next: &Ex, node: &Ex, p: &PassCtx) -> bool {
    ordered_below(next, node, p.cx.view)
}

/// `oriented` with the node's complexity precomputed: the node is FIXED across every
/// candidate attempt at a visit, so `try_rules_at` prices it once (lazily) instead of
/// per candidate. Pure caching of a pure function -- behavior-identical.
fn oriented_mu(next: &Ex, node_mu: u64, node: &Ex, p: &PassCtx) -> bool {
    let below = match complexity(next, p.cx.view).cmp(&node_mu) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => tie_cmp(next, node, p.cx.view) == std::cmp::Ordering::Less,
    };
    if !below {
        // A candidate of the whole expression's test (`visit`).
        REFUSED_FIRES.with(|r| {
            let mut r = r.borrow_mut();
            if r.len() < REFUSED_FIRES_KEPT {
                r.push(next.clone());
            }
        });
    }
    below && ground_step_admitted(node, next, p)
}

/// Is a rule step from `node` to `next` admitted where `node` is fully numeric (no variable, no
/// `<constant>`, no wildcard)? The ground evaluator decides (`PassCtx::ground_step`); a step on a
/// subject with a variable or a constant is no ground step.
fn ground_step_admitted(node: &Ex, next: &Ex, p: &PassCtx) -> bool {
    match p.ground_step {
        Some(admits)
            if !node.has_measure_space(p.cx.view) && !node.contains_wildcard(p.cx.view) =>
        {
            admits(node, next)
        }
        _ => true,
    }
}

thread_local! {
    /// The rule steps of the calls inside a [`FireLog`] scope on this thread.
    static FIRE_LOG: RefCell<Option<Vec<(Ex, Ex)>>> = const { RefCell::new(None) };
    /// The folds of the calls inside a [`FireLog`] scope on this thread.
    static FOLD_LOG: RefCell<Option<Vec<Fold>>> = const { RefCell::new(None) };
}

/// A FOLD: numbers written as one number, `op(operands) -> written`. `op` is `*` or `+` for the
/// literals a product's coefficient or a sum's constant term combines, in the order they are
/// combined (float64's value is computed in that order); `literal` for one number written as
/// another (a written literal at its double, `permissive`'s landing on a float, an entry of the
/// recognition list); `ground` for a fully numeric node written as its value by the pass.
#[derive(Clone, Debug)]
pub struct Fold {
    pub op: &'static str,
    pub operands: Vec<Ex>,
    pub written: Ex,
}

/// THE RULE STEPS AND THE FOLDS OF A SIMPLIFY CALL: within the scope, every fire is recorded as
/// its rule's instance, the left side as matched and the right side as substituted (before the
/// constructors place it), in firing order -- what F8 holds each step to (the deployed monitor
/// judges each one in the mode) -- and every fold of numbers into one number ([`Fold`]), exact or
/// rounded, in the order made, so a fold can be judged as the ground rule
/// `op(m1, m2, ...) -> written`. The folds of constructions the walk then discards (a
/// candidate the search or the settling refuses) are recorded too. Restores the enclosing logs
/// when dropped; recording changes nothing the call computes.
pub struct FireLog(Option<Vec<(Ex, Ex)>>, Option<Vec<Fold>>);

impl FireLog {
    pub fn open() -> Self {
        FireLog(
            FIRE_LOG.with(|l| l.replace(Some(Vec::new()))),
            FOLD_LOG.with(|l| l.replace(Some(Vec::new()))),
        )
    }

    /// The steps recorded so far in this scope (the log is emptied).
    pub fn take(&self) -> Vec<(Ex, Ex)> {
        FIRE_LOG.with(|l| {
            l.borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }

    /// The folds recorded so far in this scope (the log is emptied).
    pub fn take_folds(&self) -> Vec<Fold> {
        FOLD_LOG.with(|l| {
            l.borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }
}

impl Drop for FireLog {
    fn drop(&mut self) {
        let outer = self.0.take();
        FIRE_LOG.with(|l| *l.borrow_mut() = outer);
        let outer = self.1.take();
        FOLD_LOG.with(|l| *l.borrow_mut() = outer);
    }
}

/// Is a [`FireLog`] open on this thread, so that a fold is recorded ([`log_fold`])?
#[inline]
pub(crate) fn folds_logged() -> bool {
    FOLD_LOG.with(|l| l.borrow().is_some())
}

/// Record a fold where a [`FireLog`] is open.
pub(crate) fn log_fold(op: &'static str, operands: Vec<Ex>, written: Ex) {
    FOLD_LOG.with(|l| {
        if let Some(folds) = l.borrow_mut().as_mut() {
            folds.push(Fold {
                op,
                operands,
                written,
            });
        }
    });
}

/// Record a fire's rule instance where a [`FireLog`] is open.
fn log_fire(lhs: &Ex, rhs: &Ex, binds: &Binds, view: &TokenView) {
    FIRE_LOG.with(|l| {
        if let Some(steps) = l.borrow_mut().as_mut() {
            steps.push((substitute(lhs, binds, view), substitute(rhs, binds, view)));
        }
    });
}

/// Diagnostic: with `SIMPLIPY_AC_TRACE=1` every rule fire is logged (rule index, subject).
fn trace_fires() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SIMPLIPY_AC_TRACE").as_deref() == Ok("1"))
}

/// Diagnostic: with `SIMPLIPY_AC_MEMO_TRACE=1` every normal-form-memo hit and insert is
/// logged (the memo is the only walk state that changes
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

/// Try every applicable rule at this node (bucket fire order, sig-prefiltered): a mined rule's
/// step, `settle.rule` (`ac::laws::Law::SettleRule`). On a bag fire the remainder joins the
/// substituted RHS in a fresh canonical bag.
fn try_rules_at(e: &Ex, p: &PassCtx) -> Option<Ex> {
    let node_sig = atom_sig(e, p.cx.view);
    // Lazy per-visit complexity of the subject (first candidate that needs it pays it).
    let node_mu_cell: std::cell::Cell<Option<u64>> = std::cell::Cell::new(None);
    let node_mu = || match node_mu_cell.get() {
        Some(m) => m,
        None => {
            let m = complexity(e, p.cx.view);
            node_mu_cell.set(Some(m));
            m
        }
    };
    // Candidate selection. Non-bag roots: probe the exact map (one hash of the node) for
    // the exact-head rules whose LHS equals the node -- the only exact rules that can fire
    // there -- then scan just the bucket's pattern tail. Bag roots scan the whole bucket.
    const EMPTY: &[usize] = &[];
    let (exact_hits, scan): (&[usize], &[usize]) = match e {
        Ex::Add(_) | Ex::Mul(_) => (EMPTY, p.rules.bucket_for(e)),
        Ex::Fun(f, _) if lattice_head(*f, p.cx.view) => (EMPTY, p.rules.bucket_for(e)),
        _ => {
            let bucket = p.rules.bucket_for(e);
            let end = match e {
                Ex::Pow(..) => p.rules.pow_exact_end,
                Ex::Fun(f, _) => p.rules.fun_exact_end.get(f).copied().unwrap_or(0),
                _ => p.rules.leaf_exact_end,
            };
            let hits = if end == 0 {
                EMPTY
            } else {
                p.rules.exact_map.get(e).map(Vec::as_slice).unwrap_or(EMPTY)
            };
            (hits, &bucket[end..])
        }
    };
    // The subject bag's equality classes, computed by the first rule whose match needs them.
    let bag_cls: std::cell::OnceCell<Vec<usize>> = std::cell::OnceCell::new();
    // A sum subject's terms with every class in one orientation (`None`: none flips), and their
    // equality classes, computed by the first rule with a repeated wildcard that needs them.
    let oriented_terms: std::cell::OnceCell<Option<Vec<Ex>>> = std::cell::OnceCell::new();
    let oriented_cls: std::cell::OnceCell<Vec<usize>> = std::cell::OnceCell::new();
    for &ri in exact_hits.iter().chain(scan.iter()) {
        let rule = &p.rules.rules[ri];
        if rule.sig & !node_sig != 0 {
            continue;
        }
        if let Some(sup) = p.suppressed {
            // Suppression by ARTIFACT row: `src[ri]` attributes minted orientation twins
            // to the rule they were negated from, so suppressing a row silences the rule
            // AND its twins -- exactly what a build without that row would not serve.
            if sup.contains(&p.rules.src[ri]) {
                continue;
            }
        }
        // Acceptance rides the assignment enumeration: a disoriented
        // assignment asks the matcher for the NEXT one, so a rule yields to later rules
        // only when NO assignment produces an oriented fire. (Shipped rules have unique
        // assignments, where this is fire-then-check.)
        let fired: Option<Ex> = match (&rule.lhs, e) {
            (Ex::Add(pv), Ex::Add(sv)) => {
                let fire = |sv: &[Ex], cls: &std::cell::OnceCell<Vec<usize>>| -> Option<Ex> {
                    let mut binds = Binds::default();
                    let mut out: Option<Ex> = None;
                    match_bag_each_cls(
                        pv,
                        sv,
                        cls,
                        BagKind::Add,
                        true,
                        &mut binds,
                        p.mcx,
                        &mut |b, used| {
                            if !rule.licensed(b, p.cx) {
                                return false;
                            }
                            let replacement = substitute_named(
                                &rule.rhs,
                                b,
                                sv.iter()
                                    .zip(used.iter())
                                    .filter(|(_, &u)| u)
                                    .map(|(x, _)| x),
                                p.cx.view,
                            );
                            let mut parts: Vec<Ex> = sv
                                .iter()
                                .zip(used.iter())
                                .filter(|(_, &u)| !u)
                                .map(|(x, _)| x.clone())
                                .collect();
                            parts.push(replacement);
                            let next =
                                add(parts.into_iter().map(|x| canon(x, p.cx)).collect(), p.cx);
                            if oriented_mu(&next, node_mu(), e, p) {
                                log_fire(&rule.lhs, &rule.rhs, b, p.cx.view);
                                out = Some(next);
                                true
                            } else {
                                false
                            }
                        },
                    );
                    out
                };
                // A repeated wildcard binds a class up to sign: where the first match met a
                // binding refused by its repeat's negation (a near miss, `MCx::near_misses`),
                // the same terms with every class in one orientation, the signs in the
                // coefficients, are tried too.
                let misses = p.mcx.near_misses.get();
                fire(sv, &bag_cls).or_else(|| {
                    let class_binds = match rule.repeat {
                        Repeat::No => false,
                        Repeat::Leaf => p.mcx.wildcard_all,
                        Repeat::Class => true,
                    };
                    if !class_binds {
                        return None;
                    }
                    if p.mcx.near_misses.get() == misses {
                        return None;
                    }
                    let oriented = oriented_terms
                        .get_or_init(|| {
                            let mut changed = false;
                            let v: Vec<Ex> = sv
                                .iter()
                                .map(|t| match orient_term_classes(t, p.cx) {
                                    Some(o) => {
                                        changed = true;
                                        o
                                    }
                                    None => t.clone(),
                                })
                                .collect();
                            changed.then_some(v)
                        })
                        .as_ref()?;
                    fire(oriented, &oriented_cls)
                })
            }
            (Ex::Mul(pv), Ex::Mul(sv)) => {
                let mut binds = Binds::default();
                let mut out: Option<Ex> = None;
                match_bag_each_cls(
                    pv,
                    sv,
                    &bag_cls,
                    BagKind::Mul,
                    true,
                    &mut binds,
                    p.mcx,
                    &mut |b, used| {
                        if !rule.licensed(b, p.cx) {
                            return false;
                        }
                        let replacement = substitute_named(
                            &rule.rhs,
                            b,
                            sv.iter()
                                .zip(used.iter())
                                .filter(|(_, &u)| u)
                                .map(|(x, _)| x),
                            p.cx.view,
                        );
                        let mut parts: Vec<Ex> = sv
                            .iter()
                            .zip(used.iter())
                            .filter(|(_, &u)| !u)
                            .map(|(x, _)| x.clone())
                            .collect();
                        parts.push(replacement);
                        let next = mul(parts.into_iter().map(|x| canon(x, p.cx)).collect(), p.cx);
                        if oriented_mu(&next, node_mu(), e, p) {
                            log_fire(&rule.lhs, &rule.rhs, b, p.cx.view);
                            out = Some(next);
                            true
                        } else {
                            false
                        }
                    },
                );
                out
            }
            // A lattice bag (`expr::lattice`) matches any sub-bag of its members, the remainder
            // kept beside the replacement, as a sum or a product does.
            (Ex::Fun(pf, pv), Ex::Fun(sf, sv)) if pf == sf && lattice_head(*sf, p.cx.view) => {
                let mut binds = Binds::default();
                let mut out: Option<Ex> = None;
                match_bag_each_cls(
                    pv,
                    sv,
                    &bag_cls,
                    BagKind::Lat(*sf),
                    true,
                    &mut binds,
                    p.mcx,
                    &mut |b, used| {
                        if !rule.licensed(b, p.cx) {
                            return false;
                        }
                        let replacement = substitute_named(
                            &rule.rhs,
                            b,
                            sv.iter()
                                .zip(used.iter())
                                .filter(|(_, &u)| u)
                                .map(|(x, _)| x),
                            p.cx.view,
                        );
                        let mut parts: Vec<Ex> = sv
                            .iter()
                            .zip(used.iter())
                            .filter(|(_, &u)| !u)
                            .map(|(x, _)| x.clone())
                            .collect();
                        parts.push(replacement);
                        let next = fun(
                            *sf,
                            parts.into_iter().map(|x| canon(x, p.cx)).collect(),
                            p.cx,
                        );
                        if oriented_mu(&next, node_mu(), e, p) {
                            log_fire(&rule.lhs, &rule.rhs, b, p.cx.view);
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
                    if !rule.licensed(b, p.cx) {
                        return false;
                    }
                    let next = canon(
                        substitute_named(&rule.rhs, b, std::iter::once(e), p.cx.view),
                        p.cx,
                    );
                    if oriented_mu(&next, node_mu(), e, p) {
                        log_fire(&rule.lhs, &rule.rhs, b, p.cx.view);
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
            }
            debug_assert!(
                no_nested_bags(&next),
                "NESTED BAG after rule#{ri}: {next:?}"
            );
            return Some(next);
        }
    }
    None
}

/// Hard bound on accepted descent steps (fires + oriented rebuilds) per simplify call.
/// Since the reduction ordering is well-founded (see `ordered_below`), every run is
/// finite as a THEOREM, but finite is not short: a chain of n rules can take 2^n - 1 steps.
/// So the cap is a RESOURCE LIMIT, not proof of a bug when it binds. Orders of magnitude above
/// any observed run; when it binds, the pass completes descent-free, which is sound -- but
/// the answer may then be a non-fixpoint, so every binding is counted ([`step_cap_binds`]).
const STEP_CAP: u64 = 1_000_000;

/// Descent steps refused because `STEP_CAP` had bound, process-wide. 0 on every corpus the
/// gates measure; a binding is a resource limit (see `STEP_CAP`), which the call reports.
static STEP_CAP_BINDS: AtomicU64 = AtomicU64::new(0);

/// How many descent steps `STEP_CAP` has refused in this process.
pub fn step_cap_binds() -> u64 {
    STEP_CAP_BINDS.load(Relaxed)
}

/// May the walk take one more descent step? A refusal is a binding of the cap, counted and
/// reported for the call ([`CAP_STEPS`]).
#[inline]
fn step_allowed(p: &PassCtx) -> bool {
    let allowed = p.fires.get() < STEP_CAP;
    if !allowed {
        STEP_CAP_BINDS.fetch_add(1, Relaxed);
        note_cap(CAP_STEPS);
    }
    allowed
}

/// THE CALL'S STATUS: the resource caps that bound during the simplify call on this thread,
/// as bits. A call that hits one returns the state it reached -- by sound steps, possibly not a
/// fixpoint, possibly not the search's endpoint -- and says it was interrupted: the caller reads
/// the bits once the call returns ([`take_caps`], `simplify(..., return_status=True)`).
/// `STEP_CAP` bound.
pub const CAP_STEPS: u8 = 1;
/// `max_passes` bound: the chain's passes, a candidate's descent or the search's rounds ended
/// before a pass confirmed the fixpoint.
pub const CAP_PASSES: u8 = 2;
/// The search's budget (`effort`) bound with candidates left to descend.
pub const CAP_EFFORT: u8 = 4;

thread_local! {
    /// The caps that bound in the current call on this thread ([`CAP_STEPS`], ...).
    static CAPS_BOUND: Cell<u8> = const { Cell::new(0) };
}

/// Record that the cap `cap` bound in the current call.
pub fn note_cap(cap: u8) {
    CAPS_BOUND.with(|c| c.set(c.get() | cap));
}

/// The caps that bound since the last read on this thread, and reset them: read once before a
/// call (to start clean) and once after it.
pub fn take_caps() -> u8 {
    CAPS_BOUND.with(|c| c.replace(0))
}

/// One top-down rewrite pass. The protocol at each node ([`visit`]): fire site on the node
/// as-is -> fold fallback -> recurse into children + canonical rebuild -> re-enter on an
/// ORIENTED rebuild -> normal form; a fire or a rebuild that does not descend at its node is
/// taken where the WHOLE expression's price falls ([`whole_falls`]), carried to the
/// ancestor where it descends.
///
/// Every state change strictly descends the reduction ordering (`ordered_below`): a fire or a
/// rebuild at the node it rewrites, a fold by replacing a composite node (complexity >= 2) by a
/// single leaf or `Const` (complexity 1), and a carried step at the ancestor it is carried to,
/// whose rebuild is gated there. The root is its own ancestor, so the pass returns a result <=
/// its input in the ordering, equal exactly when nothing changed (Lemma L3, docs/formal.md) --
/// which is what makes the outer loop's fixpoint detection and its no-cycle argument (L4) sound.
///
/// The NORMAL-FORM memo (per simplify call, shared across the outer iterations) marks
/// subtrees walked to a true fixpoint, plus refused-rebuild nodes, which are fixpoints of
/// this pass by decision; marked subtrees return untouched. A node is not marked where a step
/// below it was refused by the whole expression's price ([`PassCtx::contextual`]): that refusal
/// reads the ancestors, which a later step can change, so the node is walked again there. Marks
/// are NOT placed once the step cap has bound: a capped walk skipped fire sites, so its results
/// are not certified normal forms (the cap is permanent for the call, so this is hygiene of the
/// memo's meaning, not a behavioral necessity).
///
/// The owned loop re-enters on every accepted step, so fire chains iterate instead of
/// recursing; [`rewrite_pass_opt`] is the borrow-side entry the child walk uses.
pub fn rewrite_pass(e: Ex, p: &PassCtx) -> Ex {
    walk(e, p, None).0
}

/// Borrow-side single visit: `None` = this subtree is unchanged by the pass (no clone was
/// made, no constructor was run); `Some(x)` = it rewrote to `x` (strictly below in the
/// reduction ordering, by the same gates as [`rewrite_pass`]). The owned loop handles
/// everything past the first change, so fire chains never recurse.
pub fn rewrite_pass_opt(node0: &Ex, p: &PassCtx) -> Option<Ex> {
    walk_opt(node0, p, None).map(|(x, _)| x)
}

/// A PLACE OF THE WALK: the ancestor `node` (as the walk visits it), the index of the child the
/// walk is in, the ancestor's depth (the root's is 0) and its own place. The whole expression's
/// test ([`whole_falls`]) rebuilds the ancestors from it. An `atan2` is rebuilt from its
/// arguments' members (`expr::atan2_of`), not by a constructor of its children: below it the
/// place is a BARRIER (`idx == usize::MAX`), where the test stops.
pub(crate) struct Up<'a> {
    node: &'a Ex,
    idx: usize,
    depth: usize,
    up: Option<&'a Up<'a>>,
}

impl Up<'_> {
    /// The ancestor with its child at this place replaced by `child`, rebuilt as the walk
    /// rebuilds it. `None` at a barrier.
    fn rebuild(&self, child: Ex, p: &PassCtx) -> Option<Ex> {
        let swap = |v: &[Ex]| {
            let mut w = v.to_vec();
            w[self.idx] = child.clone();
            w
        };
        match self.node {
            _ if self.idx == usize::MAX => None,
            Ex::Add(v) => Some(add(swap(v), p.cx)),
            Ex::Mul(v) => Some(mul(swap(v), p.cx)),
            Ex::Pow(b, x) => Some(if self.idx == 0 {
                super::expr::pow(child, (**x).clone(), p.cx)
            } else {
                super::expr::pow((**b).clone(), child, p.cx)
            }),
            Ex::Fun(f, v) => Some(super::expr::fun(*f, swap(v), p.cx)),
            _ => None,
        }
    }
}

/// THE WHOLE EXPRESSION'S PRICE: replacing the node at `up` by `next` -- a step its own node
/// refuses, because it does not descend there -- does the whole expression fall? Each ancestor is
/// rebuilt with the new child through its constructor and compared with itself as visited, up
/// to the root, whose comparison is the whole's. `Some(depth)` of the lowest ancestor where the
/// step descends, where the whole falls: the step is carried there ([`Visit::Carry`]), whose
/// rebuild then descends at its own level. `None` where the whole does not fall, at the root
/// (nothing above it), or at a barrier.
///
/// The ancestors are read as the walk visited them; a sibling rewritten since is read at its
/// visit too. The rebuild at the ancestor the step is carried to is gated on the real state, so
/// the reading decides only where to carry, never what is accepted.
fn whole_falls(next: Ex, up: Option<&Up>, p: &PassCtx) -> Option<usize> {
    let view = p.cx.view;
    let mut frame = up?;
    let mut cur = next;
    let mut target = None;
    loop {
        let rebuilt = frame.rebuild(cur, p)?;
        let below = ordered_below(&rebuilt, frame.node, view);
        if below && target.is_none() {
            target = Some(frame.depth);
        }
        match frame.up {
            Some(f) => {
                cur = rebuilt;
                frame = f;
            }
            None => return target.filter(|_| below),
        }
    }
}

/// The owned loop at one place of the walk (`up`: its ancestors). Returns the state reached and,
/// where its last step was carried to an ancestor, that ancestor's depth.
fn walk(e: Ex, p: &PassCtx, up: Option<&Up>) -> (Ex, Option<usize>) {
    let mut node = e;
    loop {
        let v = visit(&node, p, up);
        if !matches!(v, Visit::Normal) {
            p.epoch.set(p.epoch.get() + 1);
        }
        match v {
            Visit::Normal => return (node, None),
            Visit::Folded(x) => return (x, None),
            Visit::Step(next) => node = next,
            Visit::Carry(x, t) => return (x, Some(t)),
        }
    }
}

/// [`walk`] from a borrowed node: `None` where nothing changed.
fn walk_opt(node0: &Ex, p: &PassCtx, up: Option<&Up>) -> Option<(Ex, Option<usize>)> {
    let v = visit(node0, p, up);
    if !matches!(v, Visit::Normal) {
        p.epoch.set(p.epoch.get() + 1);
    }
    match v {
        Visit::Normal => None,
        Visit::Folded(x) => Some((x, None)),
        Visit::Step(next) => Some(walk(next, p, up)),
        Visit::Carry(x, t) => Some((x, Some(t))),
    }
}

/// The higher of two carries (the lower depth): the ancestor both steps reach.
fn carry_to(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, y) => x.or(y),
    }
}

/// Debug check of the pass contract (Lemma L3 plus re-entry): is `e` a fixpoint of a
/// FRESH pass -- the same rules, contexts and fold, but an empty normal-form memo, so the
/// check does not trust the marks the checked pass placed itself? `rewrite_pass`
/// re-enters every node it changes, so one pass ends where nothing fires, folds or
/// rebuilds, and its output must pass this check. (Called from `debug_assert!` only.)
pub fn is_pass_fixpoint(e: &Ex, p: &PassCtx) -> bool {
    let fresh = PassCtx {
        rules: p.rules,
        cx: p.cx,
        mcx: p.mcx,
        fold: p.fold,
        fires: Cell::new(0),
        normal: RefCell::new(FxHashSet::default()),
        contextual: Cell::new(0),
        normal_at: RefCell::new(FxHashMap::default()),
        ids: RefCell::new(NodeIds::default()),
        epoch: Cell::new(0),
        refused: RefCell::new(Vec::new()),
        suppressed: p.suppressed,
        ground_step: p.ground_step,
    };
    rewrite_pass_opt(e, &fresh).is_none()
}

/// What one visit of a node decided.
enum Visit {
    /// The node is a normal form of this pass (memo-marked unless the cap has bound or a refusal
    /// below read the ancestors).
    Normal,
    /// The numeric fold replaced the node by a leaf: final, no re-entry. (A fold whose result
    /// is no leaf -- an absorbed bag, `mod(a, b) = a - k b` -- is a [`Visit::Step`]: its
    /// members have not been visited yet.)
    Folded(Ex),
    /// An accepted descent step (a fire or an oriented rebuild): the walk re-enters on the new
    /// node.
    Step(Ex),
    /// A step accepted by the whole expression's price, carried to the ancestor at the
    /// depth given, where it descends: final at this place, which every ancestor below that one
    /// takes as it is; the ancestor's rebuild is gated as any.
    Carry(Ex, usize),
}

/// THE PER-NODE PROTOCOL, written once for both entries ([`walk`], [`walk_opt`]); see
/// [`rewrite_pass`] for the ordering argument. `up` is the node's place. Inlined into both, so
/// the walk's recursion keeps the frame layout of the former two copies.
#[inline(always)]
fn visit(node: &Ex, p: &PassCtx, up: Option<&Up>) -> Visit {
    if matches!(
        node,
        Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const(_)
    ) {
        return Visit::Normal; // literal leaves: nothing to do
    }
    if matches!(node, Ex::Num(_)) {
        // a number the fold spells otherwise (the recognition list), where that descends
        return match (p.fold)(node) {
            Some(folded) if oriented(&folded, node, p) => {
                if folds_logged() {
                    log_fold("literal", vec![node.clone()], folded.clone());
                }
                if matches!(folded, Ex::Add(_) | Ex::Mul(_) | Ex::Fun(..) | Ex::Pow(..)) {
                    Visit::Step(folded)
                } else {
                    Visit::Folded(folded)
                }
            }
            _ => Visit::Normal,
        };
    }
    let id = p.ids.borrow_mut().id(node);
    if p.normal.borrow().contains(&id) || p.normal_at.borrow().get(&id) == Some(&p.epoch.get()) {
        if trace_memo() {
            memo_log("HIT", node);
        }
        return Visit::Normal;
    }
    let depth = up.map_or(0, |u| u.depth + 1);
    // The refusals by the whole expression's price so far: one below this node keeps it unmarked.
    let contextual = p.contextual.get();

    // First fire site. Every accepted fire strictly descends the reduction ordering at this
    // node (gated inside `try_rules_at`); the step cap bounds the tier the descent argument
    // leaves open (see STEP_CAP). A fire refused there is taken where the whole expression's
    // price falls.
    if step_allowed(p) {
        REFUSED_FIRES.with(|r| r.borrow_mut().clear());
        if let Some(next) = try_rules_at(node, p) {
            p.fires.set(p.fires.get() + 1);
            return Visit::Step(next);
        }
        // A refusal reads the node's place: a node above it is normal there only as it stands
        // (`PassCtx::normal_at`), at the root too, since the node can stand below another later
        // (a search's candidate).
        let refused = REFUSED_FIRES.with(|r| std::mem::take(&mut *r.borrow_mut()));
        if !refused.is_empty() {
            p.contextual.set(p.contextual.get() + 1);
        }
        for next in refused {
            if up.is_none() || !ground_step_admitted(node, &next, p) {
                continue;
            }
            if let Some(t) = whole_falls(next.clone(), up, p) {
                if step_allowed(p) {
                    p.fires.set(p.fires.get() + 1);
                    if trace_fires() {
                        eprintln!("AC fire carried to depth {t}:\n  node={node:?}\n  ->={next:?}");
                    }
                    return Visit::Carry(next, t);
                }
            }
        }
    }
    // Fold fallback, GOVERNED BY THE MEASURE: the
    // fold fires exactly when its result descends the reduction ordering (literal bits are
    // priced, so `exp 1` refuses its ~105-unit decimal while exact small arithmetic folds).
    if let Some(folded) = (p.fold)(node) {
        if oriented(&folded, node, p) {
            if folds_logged() {
                let op = if matches!(node, Ex::Leaf(_)) {
                    "literal"
                } else {
                    "ground"
                };
                log_fold(op, vec![node.clone()], folded.clone());
            }
            return if matches!(folded, Ex::Add(_) | Ex::Mul(_) | Ex::Fun(..) | Ex::Pow(..)) {
                Visit::Step(folded)
            } else {
                Visit::Folded(folded)
            };
        }
    }

    // Recurse into children, CLONE-ON-CHANGE: an unchanged child is never cloned and an
    // unchanged node never re-runs its canonical constructor. Skipping the constructor on
    // unchanged children is sound because constructors are idempotent on canonical input --
    // every node in this walk IS constructor output under this same `cx` (from_prefix+canon
    // at entry, constructors everywhere after), so re-running them reproduces the node. The
    // normal-form memo relies on the same fact when it returns marked subtrees untouched.
    // A child's step carried above this node comes back with the depth it is carried to.
    let (rebuilt, carry): (Option<Ex>, Option<usize>) = match node {
        Ex::Add(v) => match rewrite_children(v, p, node, depth, up) {
            Some((k, c)) => (Some(add(k, p.cx)), c),
            None => (None, None),
        },
        Ex::Mul(v) => match rewrite_children(v, p, node, depth, up) {
            Some((k, c)) => (Some(mul(k, p.cx)), c),
            None => (None, None),
        },
        Ex::Pow(b, ex) => {
            let (fb, fx) = (
                Up {
                    node,
                    idx: 0,
                    depth,
                    up,
                },
                Up {
                    node,
                    idx: 1,
                    depth,
                    up,
                },
            );
            let nb = walk_opt(b, p, Some(&fb));
            let nx = in_exact(|| walk_opt(ex, p, Some(&fx)));
            if nb.is_none() && nx.is_none() {
                (None, None)
            } else {
                let c = carry_to(nb.as_ref().and_then(|x| x.1), nx.as_ref().and_then(|x| x.1));
                (
                    Some(super::expr::pow(
                        nb.map_or_else(|| (**b).clone(), |x| x.0),
                        nx.map_or_else(|| (**ex).clone(), |x| x.0),
                        p.cx,
                    )),
                    c,
                )
            }
        }
        // `atan2` is rebuilt from its arguments' members (`expr::atan2_of`): a negated argument
        // keeps its members apart, so its rewritten members are not re-decided as a product.
        // Below it the whole expression's test stops (a barrier, [`Up`]).
        Ex::Fun(f, v) if v.len() == 2 && p.cx.ieee_zeros() && p.cx.view.tok_is(*f, "atan2") => {
            let f = *f;
            let barrier = Up {
                node,
                idx: usize::MAX,
                depth,
                up,
            };
            let members = |a: &Ex| -> (Vec<Ex>, bool) {
                match a {
                    Ex::Mul(w) => match rewrite_children(w, p, a, depth + 1, Some(&barrier)) {
                        Some((k, _)) => (k, true),
                        None => (w.clone(), false),
                    },
                    other => match walk_opt(other, p, Some(&barrier)) {
                        Some((n, _)) => (vec![n], true),
                        None => (vec![other.clone()], false),
                    },
                }
            };
            let ((ym, cy), (xm, cxm)) = (members(&v[0]), members(&v[1]));
            (
                (cy || cxm).then(|| super::expr::atan2_of(f, ym, xm, p.cx)),
                None,
            )
        }
        Ex::Fun(f, v) => {
            let f = *f;
            match exact_argument(f, v.len(), p.cx.view) {
                Some(k) => {
                    let mut out: Option<Vec<Ex>> = None;
                    let mut c = None;
                    for (i, x) in v.iter().enumerate() {
                        let here = Up {
                            node,
                            idx: i,
                            depth,
                            up,
                        };
                        let r = if i == k {
                            in_exact(|| walk_opt(x, p, Some(&here)))
                        } else {
                            walk_opt(x, p, Some(&here))
                        };
                        if let Some((r, rc)) = r {
                            out.get_or_insert_with(|| v.to_vec())[i] = r;
                            c = carry_to(c, rc);
                        }
                    }
                    out.map(|k| (k, c))
                }
                None => rewrite_children(v, p, node, depth, up),
            }
            .map_or((None, None), |(k, c)| {
                (Some(super::expr::fun(f, k, p.cx)), c)
            })
        }
        _ => (None, None), // Leaf: no children
    };

    match rebuilt {
        // A true fixpoint of one full walk: children unchanged, no fire, no fold.
        None => mark_normal(node, p, contextual, "INSERT(fixpoint)"),
        // A rebuild that changed anything must re-enter the full walk: the canonical
        // constructors can MINT new subterms during the rebuild (an integer power
        // distributing over a product, collection merging terms), and only a fresh top-down
        // walk visits them. The re-entry must satisfy the SAME reduction ordering as a rule
        // fire; a rebuild that does not descend here is taken where a step below was carried
        // above this node, or where the whole expression's price falls, else REFUSED and
        // proposed to the search as the state it builds (`PassCtx::refused`).
        Some(rebuilt) if rebuilt != *node => {
            let admitted = ground_step_admitted(node, &rebuilt, p);
            if oriented(&rebuilt, node, p) && admitted && step_allowed(p) {
                p.fires.set(p.fires.get() + 1);
                return Visit::Step(rebuilt);
            }
            if admitted {
                if let Some(t) = carry.filter(|&t| t < depth) {
                    return Visit::Carry(rebuilt, t);
                }
                if let Some(t) = whole_falls(rebuilt.clone(), up, p) {
                    if step_allowed(p) {
                        p.fires.set(p.fires.get() + 1);
                        if trace_fires() {
                            eprintln!(
                                "AC rebuild carried to depth {t}:\n  node={node:?}\n  rebuilt={rebuilt:?}"
                            );
                        }
                        return Visit::Carry(rebuilt, t);
                    }
                }
            }
            if trace_fires() {
                eprintln!("AC rebuild refused:\n  node={node:?}\n  rebuilt={rebuilt:?}");
            }
            p.contextual.set(p.contextual.get() + 1);
            // Proposed to the search where the ground evaluator admits it (a fully numeric node
            // rebuilt is judged whole: `1/sin(np.pi)` is no `1/0`).
            let mut refused = p.refused.borrow_mut();
            if admitted && refused.len() < REFUSED_KEPT && !refused.iter().any(|(n, _)| n == node) {
                refused.push((node.clone(), rebuilt));
            }
            drop(refused);
            mark_normal(node, p, contextual, "INSERT(refused-rebuild)")
        }
        // rebuilt == node with changed children: the constructor undid the child moves
        // (equality decided on the representation). A fixpoint by decision.
        Some(_) => mark_normal(node, p, contextual, "INSERT(fixpoint)"),
    }
}

/// How many refused rebuilds a call keeps for the search (`PassCtx::refused`): a size guard,
/// not a measure.
const REFUSED_KEPT: usize = 64;

thread_local! {
    /// The fires `try_rules_at` refused at the node it last tried because they do not descend
    /// there (`oriented_mu`), the first [`REFUSED_FIRES_KEPT`]: the candidates of the whole
    /// expression's test ([`visit`]).
    static REFUSED_FIRES: RefCell<Vec<Ex>> = const { RefCell::new(Vec::new()) };
}

/// How many refused fires a visit tries by the whole expression's price: the first in the rules'
/// fire order. A cost bound, not a measure: each try rebuilds every ancestor, and trying four
/// took a sixth more time on the training-data inputs in `permissive`, with no answer of srbf's
/// ground truths, a tenth of those inputs or a tenth of its predictions changed.
const REFUSED_FIRES_KEPT: usize = 1;

/// Mark `node` a normal form of this pass, unless the step cap has bound, and report it so: in
/// its place as it stands where a refusal happened below it since `contextual` was read
/// ([`PassCtx::normal_at`]), else wherever it stands.
fn mark_normal(node: &Ex, p: &PassCtx, contextual: u64, why: &str) -> Visit {
    // a node normal inside an exponent, a root index or a Bessel order may not be outside it
    // (`in_exact_position`)
    if p.fires.get() < STEP_CAP && !in_exact_position() {
        if trace_memo() {
            memo_log(why, node);
        }
        let id = p.ids.borrow_mut().id(node);
        if p.contextual.get() == contextual {
            p.normal.borrow_mut().insert(id);
        } else {
            p.normal_at.borrow_mut().insert(id, p.epoch.get());
        }
    }
    Visit::Normal
}

thread_local! {
    /// Whether the walk is inside an exponent or a root index (`in_exact_position`).
    static EXACT_POSITION: Cell<bool> = const { Cell::new(false) };
}

/// IS THE WALK INSIDE AN EXPONENT, A ROOT INDEX OR A BESSEL ORDER? Those read more than a part's
/// kind: they read its integrality and parity (a negative base raised to a non-integer power is
/// NaN, an even root is NaN below 0), so a fully numeric part there may stand only for a value
/// float64 gives it bit for bit, never for its value in the reals: `rootn(x1, acos(cos(1)))`
/// stays, its index 0.9999999999999999 in float64, where the reals give 1. The ground fold and a
/// rule step on a fully numeric node read it.
pub(crate) fn in_exact_position() -> bool {
    EXACT_POSITION.with(|c| c.get())
}

/// `f` with the walk inside an exponent or a root index (`in_exact_position`).
fn in_exact<T>(f: impl FnOnce() -> T) -> T {
    let prev = EXACT_POSITION.with(|c| c.replace(true));
    let out = f();
    EXACT_POSITION.with(|c| c.set(prev));
    out
}

/// The argument of `f` (with `n` arguments) that reads its value exactly: `pow`'s exponent,
/// `rootn`'s index and a Bessel function's order (NaN at a negative argument for an order that is
/// no integer).
fn exact_argument(f: Tok, n: usize, view: &TokenView) -> Option<usize> {
    if n != 2 {
        None
    } else if view.tok_is(f, "pow") || view.tok_is(f, "rootn") {
        Some(1)
    } else if ["besselj", "bessely", "besseli", "besselk"]
        .iter()
        .any(|s| view.tok_is(f, s))
    {
        Some(0)
    } else {
        None
    }
}

/// Rewrite the members of `parent` (at `depth`, its place `up`) clone-on-change: `None` = no
/// member moved (nothing allocated); else the members, with the carry of a member's step above
/// `parent` ([`carry_to`]).
fn rewrite_children(
    v: &[Ex],
    p: &PassCtx,
    parent: &Ex,
    depth: usize,
    up: Option<&Up>,
) -> Option<(Vec<Ex>, Option<usize>)> {
    let mut out: Option<Vec<Ex>> = None;
    let mut carry = None;
    for (i, x) in v.iter().enumerate() {
        let here = Up {
            node: parent,
            idx: i,
            depth,
            up,
        };
        match walk_opt(x, p, Some(&here)) {
            Some((nx, c)) => {
                out.get_or_insert_with(|| v[..i].to_vec()).push(nx);
                carry = carry_to(carry, c);
            }
            None => {
                if let Some(o) = out.as_mut() {
                    o.push(x.clone());
                }
            }
        }
    }
    out.map(|o| (o, carry))
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
