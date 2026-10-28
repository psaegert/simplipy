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
//! remainder (the widening). If no rule fires, the NUMERIC FOLD fallback runs -- the exact
//! ordering of the shipped engine, which is load-bearing: `sin(pi) -> 0` (a rule) must beat
//! `sin(3.14159...) -> 1.22e-16` (the f64 fold). Then recurse into children, rebuild
//! canonically, and re-enter when the rebuild descends the reduction ordering (else refuse
//! it). Every accepted step -- fire or rebuild -- strictly descends that ordering at its
//! node, and the ordering is WELL-FOUNDED, so termination is a theorem; the hard step cap
//! is defense-in-depth against ordering-invariant bugs (see `ordered_below` and
//! docs/formal.md).

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::tokens::{Tok, TokenView};

use super::convert::from_prefix;
use super::expr::{
    add, canon, complexity, fun, lattice_head, mul, orient_term_classes, tie_cmp, Cx, Ex,
};
use super::matcher::{match_bag_each_cls, matches_each, substitute, BagKind, Binds, MCx};
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
    /// Does a wildcard occur more than once in the LHS? Then it binds a class (a sum, an odd
    /// function or odd root of one) up to sign: a sum pattern whose first match met a near miss
    /// (a binding refused by its repeat's negation, `MCx::near_misses`) tries its subject again
    /// with every class in one orientation (`orient_term_classes`). `Repeat::Class` where the repeated
    /// wildcard's sort can bind a class in every mode (`_`, `!`, `$`); `Repeat::Leaf` where only
    /// a pass that binds any subtree (permissive) lets it, `?` binding variable leaves otherwise.
    pub repeat: Repeat,
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
    /// O(1) EXACT dispatch (the old engine's exact-lookup-first protocol, restored for the
    /// AC engine): canonical wildcard-free LHS -> rule indices in asset order, for NON-BAG
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
        Ex::Const => 1u64 << 62,
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
/// permissive cancels `/ $0 $0` on the pattern, but its instance `inf / inf` is NaN, and the
/// rule `/ $0 $0 -> 1` is the sentinel doctrine that defines that instance. So permissive's
/// rule sides are read with certified licences (its fold discipline is the bare one).
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
/// become a different pattern, which nothing judged. Real's canon reads the f64 fact
/// `rootn 2 atanh tanh 3 -> nan` (f64's `atanh(tanh(3))` is not 3) as `rootn 2 3 -> nan`,
/// which is false. Read in the canon it was mined in, the pattern stays the one judged, and a
/// pattern the serving canon never produces never fires.
///
/// Numerals are read in the SERVING mode's number domain whatever the canon (a numeral
/// denotes the number it spells, and real reads it exactly); the canon decides the structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleCanon {
    /// A per-mode mine of this mode (0.15 on), and every set handed over in memory: the
    /// mode's own reading ([`rule_cx`]).
    Mode(crate::engine::RuleMode),
    /// A mine from before per-mode mines (0.14 and earlier): one mode-blind canon for all
    /// three files, Default's, unfolded and uncertified ([`Cx::bare`]), the reading every
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
            RuleCanon::Mode(RuleMode::Real) => "real",
            RuleCanon::Mode(RuleMode::Permissive) => "permissive",
        }
    }
}

/// THE ENGINE-OWNED LAW (gate G8), one definition for the rule load and the miner: does the
/// rule `lhs -> rhs` (canonical sides) state a law the engine itself performs? Today that is
/// the distributive law as the search applies it (one FACTOR move, `search::is_factoring`).
/// The load drops such a rule ("engine-owned-law"); the miner refuses to mint it
/// (`Engine::law_instance`, the miner's acceptance). A law the constructors perform needs no
/// arm here, the collection of powers of e (`exp(a) exp(b) = exp(a + b)`) among them: every
/// canon a rule side is read in performs it, so a rule stating it, in either direction, has one
/// canonical state on both sides, which the load refuses as subsumed before this gate and
/// `Engine::law_instance` names `canon`.
///
/// The search meets a rule's INSTANCES, so the law is judged on one: every `?`, `!` and `$`
/// wildcard becomes a variable of its own ([`pattern_instance`]), and the sides are read in
/// `law`, the serving search's licence context (`Engine::with_law_cx`), whose certificates
/// answer for every binding the sorts admit. The FACTOR move is then the search's own.
pub(crate) fn engine_owned_law(lhs: &Ex, rhs: &Ex, law: &Cx) -> bool {
    let mut fresh = FxHashMap::default();
    let l = canon(pattern_instance(lhs, law.view, &mut fresh), law);
    let r = canon(pattern_instance(rhs, law.view, &mut fresh), law);
    super::search::is_factoring(&l, &r, law)
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
            out.rules.push(AcRule {
                lhs,
                rhs,
                sig,
                is_pattern,
                repeat,
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
            out.rules.push(AcRule {
                lhs: tl,
                rhs: tr,
                sig,
                is_pattern,
                repeat,
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
    /// SUPPRESSED ARTIFACT ROWS (`AcRules::src` values), checked at the single fire site
    /// (`try_rules_at`): a suppressed rule and every orientation twin minted from it (the
    /// twin reports its source's row) never fires, so the walk is BEHAVIORALLY the walk
    /// of an engine translated without those artifact rows -- translation is per-rule,
    /// bucket order filtered in place is the order a rebuild would give the survivors,
    /// and constructors/certificates/folds never consult the rule set. `None` (every
    /// pre-existing entry point) is byte-identical to the field not existing. Consumer:
    /// the promotion refund's per-candidate "engine WITHOUT this rule" probes.
    pub suppressed: Option<&'a FxHashSet<usize>>,
}

/// The REDUCTION ORDERING: is `a` strictly smaller than `b` in the lexicographic order
/// (mu, the sign key, canonical total order -- `tie_cmp`)? This one predicate orients the
/// whole system: `rewrite_pass` refuses any FIRE or REBUILD whose result does not descend it
/// at the rewritten node (Lemma L2), and the search accepts only states below its best;
/// `translate` drops only rules whose canonical pattern sides climb in mu (gate G7,
/// docs/formal.md).
///
/// Stage 2 (design/UNIFIED_SIMPLICITY_MEASURE.md): the first component is the unified
/// simplicity measure mu (`ac::expr::complexity`), which ABSORBS the old lit_size
/// middle tier -- mu's literal component carries a literal's content up to float
/// precision (the float cap), so the ordering loses a layer. The dense-literal chain its
/// own tier existed for (`Mul[3/2^k, x]`) ascends in k up to k = 55 and then sits at
/// 64-72 bits up to k = 200; it stays finite through the 1,100-bit cap (below).
///
/// The pair is a strict total order (mu is a u64; cmp_ex is total by construction
/// with EXACT literal comparison) that is WELL-FOUNDED: mu can strictly drop only
/// finitely often, and at a FIXED mu value only finitely many terms exist -- mu
/// bounds the node count (every structural node and leaf costs >= 8 except zero-cost
/// magnitude-1 coefficient/exponent slots, of which each bag and pow carries at most
/// one), literals are finitely many under the 1,100-bit cap (a literal's price is
/// capped at its float's shortest decimal, so mu no longer bounds its bit length), and the
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
    match complexity(next, p.cx.view).cmp(&node_mu) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => tie_cmp(next, node, p.cx.view) == std::cmp::Ordering::Less,
    }
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

/// Try every applicable rule at this node (bucket fire order, sig-prefiltered). On a bag fire the
/// remainder joins the substituted RHS in a fresh canonical bag.
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
        // assignments, where this is exactly the old fire-then-check behavior.)
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
                            let replacement = substitute(&rule.rhs, b, p.cx.view);
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
                        let replacement = substitute(&rule.rhs, b, p.cx.view);
                        let mut parts: Vec<Ex> = sv
                            .iter()
                            .zip(used.iter())
                            .filter(|(_, &u)| !u)
                            .map(|(x, _)| x.clone())
                            .collect();
                        parts.push(replacement);
                        let next = mul(parts.into_iter().map(|x| canon(x, p.cx)).collect(), p.cx);
                        if oriented_mu(&next, node_mu(), e, p) {
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
                        let replacement = substitute(&rule.rhs, b, p.cx.view);
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
                    if oriented_mu(&next, node_mu(), e, p) {
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
/// finite as a THEOREM and this cap is defense-in-depth against bugs in the ordering
/// invariant itself, not a load-bearing bound. Orders of magnitude above any observed
/// run; when it binds, the pass completes descent-free, which is sound -- but the answer
/// may then be a non-fixpoint, so every binding is counted ([`step_cap_binds`]).
const STEP_CAP: u64 = 1_000_000;

/// Descent steps refused because `STEP_CAP` had bound, process-wide. Expected 0 on every
/// input: a nonzero count means the ordering invariant has a bug (see `STEP_CAP`).
static STEP_CAP_BINDS: AtomicU64 = AtomicU64::new(0);

/// How many descent steps `STEP_CAP` has refused in this process (expected 0).
pub fn step_cap_binds() -> u64 {
    STEP_CAP_BINDS.load(Relaxed)
}

/// May the walk take one more descent step? A refusal is a binding of the cap, counted.
#[inline]
fn step_allowed(p: &PassCtx) -> bool {
    let allowed = p.fires.get() < STEP_CAP;
    if !allowed {
        STEP_CAP_BINDS.fetch_add(1, Relaxed);
    }
    allowed
}

/// One top-down rewrite pass. The protocol at each node ([`visit`]): fire site on the node
/// as-is -> fold fallback -> recurse into children + canonical rebuild -> re-enter on an
/// ORIENTED rebuild (else give the rebuild its one COMPOSITE chance, else refuse it) ->
/// normal form.
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
///
/// The owned loop re-enters on every accepted step, so fire chains iterate instead of
/// recursing; [`rewrite_pass_opt`] is the borrow-side entry the child walk uses.
pub fn rewrite_pass(e: Ex, p: &PassCtx) -> Ex {
    let mut node = e;
    loop {
        match visit(&node, p) {
            Visit::Normal => return node,
            Visit::Folded(x) => return x,
            Visit::Step(next) => node = next,
        }
    }
}

/// Borrow-side single visit: `None` = this subtree is unchanged by the pass (no clone was
/// made, no constructor was run); `Some(x)` = it rewrote to `x` (strictly below in the
/// reduction ordering, by the same gates as [`rewrite_pass`]). The owned loop handles
/// everything past the first change, so fire chains never recurse.
pub fn rewrite_pass_opt(node0: &Ex, p: &PassCtx) -> Option<Ex> {
    match visit(node0, p) {
        Visit::Normal => None,
        Visit::Folded(x) => Some(x),
        Visit::Step(next) => Some(rewrite_pass(next, p)),
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
        explore: p.explore,
        suppressed: p.suppressed,
    };
    rewrite_pass_opt(e, &fresh).is_none()
}

/// What one visit of a node decided.
enum Visit {
    /// The node is a normal form of this pass (memo-marked unless the cap has bound).
    Normal,
    /// The numeric fold replaced the node by a leaf: final, no re-entry. (A fold whose result
    /// is no leaf -- an absorbed bag, `mod(a, b) = a - k b` -- is a [`Visit::Step`]: its
    /// members have not been visited yet.)
    Folded(Ex),
    /// An accepted descent step (a fire, an oriented rebuild, or a committed composite
    /// step): the walk re-enters on the new node.
    Step(Ex),
}

/// THE PER-NODE PROTOCOL, written once for both entries ([`rewrite_pass`],
/// [`rewrite_pass_opt`]); see [`rewrite_pass`] for the ordering argument. Inlined into
/// both, so the walk's recursion keeps the frame layout of the former two copies.
#[inline(always)]
fn visit(node: &Ex, p: &PassCtx) -> Visit {
    if matches!(
        node,
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const
    ) {
        return Visit::Normal; // literal leaves: nothing to do
    }
    if p.normal.borrow().contains(node) {
        if trace_memo() {
            memo_log("HIT", node);
        }
        return Visit::Normal;
    }

    // First fire site. Every accepted fire strictly descends the reduction ordering at this
    // node (gated inside `try_rules_at`); the step cap bounds the tier the descent argument
    // leaves open (see STEP_CAP).
    if step_allowed(p) {
        if let Some(next) = try_rules_at(node, p) {
            p.fires.set(p.fires.get() + 1);
            return Visit::Step(next);
        }
    }
    // Fold fallback, GOVERNED BY THE MEASURE: the
    // fold fires exactly when its result descends the reduction ordering (literal bits are
    // priced, so `exp 1` refuses its ~105-unit decimal while exact small arithmetic folds).
    if let Some(folded) = (p.fold)(node) {
        if oriented(&folded, node, p) {
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
    let rebuilt: Option<Ex> = match node {
        Ex::Add(v) => rewrite_children(v, p).map(|k| add(k, p.cx)),
        Ex::Mul(v) => rewrite_children(v, p).map(|k| mul(k, p.cx)),
        Ex::Pow(b, ex) => {
            let nb = rewrite_pass_opt(b, p);
            let nx = rewrite_pass_opt(ex, p);
            if nb.is_none() && nx.is_none() {
                None
            } else {
                Some(super::expr::pow(
                    nb.unwrap_or_else(|| (**b).clone()),
                    nx.unwrap_or_else(|| (**ex).clone()),
                    p.cx,
                ))
            }
        }
        // `atan2` is rebuilt from its arguments' members (`expr::atan2_of`): a negated argument
        // keeps its members apart, so its rewritten members are not re-decided as a product.
        Ex::Fun(f, v) if v.len() == 2 && p.cx.ieee_zeros() && p.cx.view.tok_is(*f, "atan2") => {
            let f = *f;
            let members = |a: &Ex| -> (Vec<Ex>, bool) {
                match a {
                    Ex::Mul(w) => match rewrite_children(w, p) {
                        Some(k) => (k, true),
                        None => (w.clone(), false),
                    },
                    other => match rewrite_pass_opt(other, p) {
                        Some(n) => (vec![n], true),
                        None => (vec![other.clone()], false),
                    },
                }
            };
            let ((ym, cy), (xm, cxm)) = (members(&v[0]), members(&v[1]));
            (cy || cxm).then(|| super::expr::atan2_of(f, ym, xm, p.cx))
        }
        Ex::Fun(f, v) => {
            let f = *f;
            rewrite_children(v, p).map(|k| super::expr::fun(f, k, p.cx))
        }
        _ => None, // Leaf: no children
    };

    match rebuilt {
        // A true fixpoint of one full walk: children unchanged, no fire, no fold.
        None => mark_normal(node, p, "INSERT(fixpoint)"),
        // A rebuild that changed anything must re-enter the full walk: the canonical
        // constructors can MINT new subterms during the rebuild (an integer power
        // distributing over a product, collection merging terms), and only a fresh top-down
        // walk visits them. The re-entry must satisfy the SAME reduction ordering as a rule
        // fire (a disoriented rebuild is REFUSED), with the composite step as its one
        // second chance.
        Some(rebuilt) if rebuilt != *node => {
            if oriented(&rebuilt, node, p) && step_allowed(p) {
                p.fires.set(p.fires.get() + 1);
                return Visit::Step(rebuilt);
            }
            if let Some(endpoint) = composite_step(rebuilt, node, p) {
                return Visit::Step(endpoint);
            }
            mark_normal(node, p, "INSERT(refused-rebuild)")
        }
        // rebuilt == node with changed children: the constructor undid the child moves
        // (equality decided on the representation). A fixpoint by decision.
        Some(_) => mark_normal(node, p, "INSERT(fixpoint)"),
    }
}

/// COMPOSITE-STEP acceptance (docs/formal.md, step relation) for a rebuild that failed the
/// immediate ordering test: judge the SETTLED ENDPOINT of the rebuilt candidate under a
/// one-level, private-memo walk, and commit iff the endpoint strictly descends below
/// `node`. `None` = refused (or exploration is off at this level, or the cap has bound).
#[inline(always)]
fn composite_step(rebuilt: Ex, node: &Ex, p: &PassCtx) -> Option<Ex> {
    if !(p.explore && step_allowed(p)) {
        if trace_fires() {
            eprintln!("AC rebuild refused (disoriented):\n  node={node:?}\n  rebuilt={rebuilt:?}");
        }
        return None;
    }
    let sub = PassCtx {
        rules: p.rules,
        cx: p.cx,
        mcx: p.mcx,
        fold: p.fold,
        fires: Cell::new(p.fires.get()),
        normal: RefCell::new(FxHashSet::default()),
        explore: false,
        suppressed: p.suppressed,
    };
    let endpoint = rewrite_pass(rebuilt, &sub);
    p.fires.set(sub.fires.get());
    if oriented(&endpoint, node, p) && step_allowed(p) {
        p.fires.set(p.fires.get() + 1);
        if trace_fires() {
            eprintln!("AC composite step committed:\n  node={node:?}\n  endpoint={endpoint:?}");
        }
        return Some(endpoint);
    }
    if trace_fires() {
        eprintln!(
            "AC rebuild refused (endpoint not below):\n  node={node:?}\n  endpoint={endpoint:?}"
        );
    }
    None
}

/// Mark `node` a normal form of this pass (unless the step cap has bound) and report it so.
fn mark_normal(node: &Ex, p: &PassCtx, why: &str) -> Visit {
    if p.fires.get() < STEP_CAP {
        if trace_memo() {
            memo_log(why, node);
        }
        p.normal.borrow_mut().insert(node.clone());
    }
    Visit::Normal
}

/// Rewrite a bag's members clone-on-change: `None` = no member moved (nothing allocated).
fn rewrite_children(v: &[Ex], p: &PassCtx) -> Option<Vec<Ex>> {
    let mut out: Option<Vec<Ex>> = None;
    for (i, x) in v.iter().enumerate() {
        match rewrite_pass_opt(x, p) {
            Some(nx) => out.get_or_insert_with(|| v[..i].to_vec()).push(nx),
            None => {
                if let Some(o) = out.as_mut() {
                    o.push(x.clone());
                }
            }
        }
    }
    out
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
