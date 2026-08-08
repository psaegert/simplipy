//! Engine glue for the AC core: certificates, the numeric-fold fallback, lazy rule
//! translation, and the `ac_simplify` entry point.
//!
//! The AC core is representation + algorithm; everything ENGINE-flavored lives here:
//! * the two soundness certificates run through the SAME per-engine generational caches as the
//!   old kernel (`bang_cache` / `mult_cache`), keyed by the serialized explicit form -- one
//!   trust base, two engines;
//! * the serve-time fold is EXACT-ONLY (fold unification, 2026-08-02): rational arithmetic
//!   lives in the constructors, the `<constant>` value-set collapse below, and NOTHING
//!   transcendental folds at serve -- those identities are mined, symbolically certified rules;
//! * the `<constant>` collapse carries the owner-ratified licence verbatim (`cls() == Finite`,
//!   positive-measure finite part) through `interval::value_set`;
//! * the loaded ruleset is TRANSLATED once, lazily, on first AC use -- construction cost is
//!   untouched for consumers that never opt in.
//!
//! ## The simplify loop, and why it is so much simpler than the old one
//!
//! The old kernel needs a best-first tree SEARCH because cancellation is non-confluent there:
//! which cancel candidate fires, and in which order, changes which rules can fire later, so the
//! engine must branch. In the AC core, cancellation IS canonicalization (like-term collection
//! in bags, computed by one deterministic function) -- there is nothing to branch over. What
//! remains is a deterministic chain: rewrite-pass -> canonical form -> repeat. Every pass
//! output descends the reduction ordering (strictly when anything changed), and the ordering
//! is WELL-FOUNDED (the measure mu, then the canonical order -- see `ordered_below`), so the
//! chain never revisits a state and reaches its fixpoint in finitely many passes as a THEOREM;
//! the step cap and the iteration budget are defense-in-depth against ordering-invariant bugs
//! (the full ledger of what is theorem, what is enforced, and what is empirical:
//! docs/formal.md). The answer is the chain's FINAL state -- by pass contraction (L3) it
//! carries the chain's minimum complexity, and it is the fixpoint whenever the budget did not
//! truncate (the input's canonical form is the first state, so the result never exceeds it);
//! a fixpoint run is idempotent by determinism, gated empirically on the corpora.

use std::cell::{Cell, RefCell};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ac::convert::{from_prefix, to_infix_pretty, to_prefix, to_prefix_tagged};
use crate::ac::expr::{canon, complexity, rejoin_projection, Cx, Ex};
use crate::ac::matcher::MCx;
use crate::ac::rules::{rewrite_pass, AcRules, PassCtx};
use crate::tokens::{TokenOverlay, TokenView};

use super::memo::SimplifyCtx;
use super::Engine;

/// Contains a special constant (pi / e) anywhere: the PERMANENT mask-x-special policy
/// gate -- a special never vanishes into a fitted constant (owner-ratified 2026-08-01,
/// held against mu's own gradient for the post-fit rationalizer).
fn has_special(e: &Ex) -> bool {
    match e {
        Ex::Pi | Ex::E => true,
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(has_special),
        Ex::Pow(b, ex) => has_special(b) || has_special(ex),
        _ => false,
    }
}

/// H-051 (2026-08-05): does this expression contain a `pow` whose exponent denotes
/// an exact INTEGER the interval layer can only BRACKET -- a beyond-2^53 `Num`, a
/// beyond-i128 literal spelling, or its structural negation? The composite-level
/// class fold must not trust a Nan verdict computed over such a node: the interval
/// walk re-brackets the exponent internally and its continuum convention asserts Nan
/// for what is truly a parity-decided value (fuzz-extreme row 1756). Conservative
/// over-approximation (some 2^53..2^64 integers are f64-exact and would classify
/// fine); the consequence of a hit is a fold REFUSAL, which is always sound.
fn contains_bracket_poisoned_pow(e: &Ex, view: &TokenView) -> bool {
    // A poisoned LITERAL: an exact integer the interval leaf read can only bracket
    // (or reads as an f64 image away from the denoted integer).
    fn poisoned_literal(e: &Ex, view: &TokenView) -> bool {
        match e {
            Ex::Num(r) => r.is_integer() && r.num().unsigned_abs() > (1u128 << 53),
            Ex::Leaf(t) => view
                .with_str(*t, crate::numeric::integer_literal_parity)
                .is_some(),
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|c| poisoned_literal(c, view)),
            Ex::Pow(b, ex) => poisoned_literal(b, view) || poisoned_literal(ex, view),
            _ => false,
        }
    }
    match e {
        // The WHOLE exponent subtree counts: `pow(-3, 1e19 + (-1e40))` carries the
        // poisoned leaf inside an Add exponent, brackets just the same, and asserted
        // Nan for a parity-decided finite value (fuzz-extreme 200k survey).
        Ex::Pow(b, ex) => {
            poisoned_literal(ex, view)
                || contains_bracket_poisoned_pow(b, view)
                || contains_bracket_poisoned_pow(ex, view)
        }
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => {
            v.iter().any(|c| contains_bracket_poisoned_pow(c, view))
        }
        _ => false,
    }
}

/// Recursively GROUND: no `<constant>` and no variable leaf anywhere (numeric-string
/// leaves are literals and qualify). The certified-absorption arm's scope guard.
fn is_ground(e: &Ex, view: &TokenView) -> bool {
    match e {
        Ex::Const => false,
        Ex::Leaf(t) => view.with_str(*t, |s| {
            s.as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit() || *b == b'-' || *b == b'.' || *b == b'+')
                && crate::utils::is_numeric_string(s)
        }),
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => true,
        Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().all(|x| is_ground(x, view)),
        Ex::Pow(b, ex) => is_ground(b, view) && is_ground(ex, view),
    }
}

impl Engine {
    /// The translated AC ruleset, built once on first use (never during construction).
    pub(crate) fn ac_rules(&self) -> &AcRules {
        self.ac_rules_cell.get_or_init(|| {
            let overlay = RefCell::new(TokenOverlay::new(self.tokens.len()));
            let view = TokenView::new(&self.tokens, &overlay);
            AcRules::translate(&self.rules.raw, &view)
        })
    }

    /// (kept, subsumed-by-arithmetic, dropped, orientation-twins) counts of the AC
    /// translation -- the audit surface for "the coefficient-rule family stops needing
    /// to exist". `kept` is the SERVING rule count and includes the minted twins;
    /// `subsumed`/`dropped` account for asset rules only.
    pub fn ac_rules_info(&self) -> (usize, usize, usize, usize) {
        let r = self.ac_rules();
        (r.rules.len(), r.n_subsumed, r.n_dropped, r.n_twins)
    }

    /// An AC-side certificate through the shared per-engine caches: serialize the expression
    /// into the explicit form and consult the SAME `finite_ae` / `finite_nonzero_ae` interval
    /// analysis (and the same generational memo) as the old kernel.
    fn ac_cert(&self, e: &Ex, ctx: &SimplifyCtx, mult: bool) -> bool {
        let view = self.view(ctx);
        let bare = Cx::bare(&view);
        let flat = to_prefix(e, &bare);
        let scratch = if mult {
            &ctx.cert_mult_scratch
        } else {
            &ctx.cert_scratch
        };
        if let Some(&b) = scratch.borrow().get(&flat) {
            return b;
        }
        let table_only = flat.iter().all(|&t| self.tokens.is_table_id(t));
        if table_only {
            let cache = if mult {
                &self.mult_cache
            } else {
                &self.bang_cache
            };
            if let Some(b) = cache.lock().unwrap().get_promoting(&flat) {
                scratch.borrow_mut().insert(flat, b);
                return b;
            }
        }
        let strs = self.resolve_seq(&flat, ctx);
        let b = if mult {
            crate::interval::finite_nonzero_ae(&strs, &self.operators)
        } else {
            crate::interval::finite_ae(&strs, &self.operators)
        };
        if table_only {
            let cache = if mult {
                &self.mult_cache
            } else {
                &self.bang_cache
            };
            cache.lock().unwrap().insert(flat.clone(), b);
        }
        scratch.borrow_mut().insert(flat, b);
        b
    }

    /// The zero-set certificate behind `Cx::cert_nzae`: NONZERO a.e., decided by the
    /// structural zero-set analysis (`interval::zero_set_null` -- identity-theorem
    /// witness for analytic compositions, fail-closed on everything else). Uncached:
    /// the structural recursion is cheap and the witness budget is small.
    fn ac_zsn(&self, e: &Ex, ctx: &SimplifyCtx) -> bool {
        let view = self.view(ctx);
        let bare = Cx::bare(&view);
        let flat = to_prefix(e, &bare);
        let strs = self.resolve_seq(&flat, ctx);
        crate::interval::zero_set_null(&strs, &self.operators)
    }

    /// The nonconstant-entire certificate behind `Cx::cert_nce` (the symbolic-exponent
    /// merge licence): two disjoint rigorous value intervals prove the exponent is not
    /// identically constant, so its level sets are null (identity theorem).
    fn ac_nce(&self, e: &Ex, ctx: &SimplifyCtx) -> bool {
        let view = self.view(ctx);
        let bare = Cx::bare(&view);
        let flat = to_prefix(e, &bare);
        let strs = self.resolve_seq(&flat, ctx);
        crate::interval::nonconstant_entire(&strs, &self.operators)
    }

    /// The `<constant>` value-set fold: a `<constant>`-and-finite-literals node collapses
    /// to `<constant>` under the ratified value-set licence (unconditionally under LOSSY).
    ///
    /// FOLD UNIFICATION (owner-approved 2026-08-02): the engine folds ONLY what it
    /// computes EXACTLY -- rational arithmetic, which lives in the constructors (bag
    /// coefficient merging, integer/rational pow, exact roots) -- and this fallback's
    /// former all-literal arm, the ONE remaining f64-evaluation path at serve, is
    /// DELETED. Every transcendental identity, special-bearing or not, is a MINED rule
    /// certified by the symbolic judge (`simplipy.verify`, precision-stable residuals):
    /// `cos 0 -> 1` and `cos np.pi -> -1` now arrive the same way. This removes the
    /// special/special-free asymmetry (the old `has_special` exactness guard goes with
    /// the arm it guarded) and closes the residual accidental-short-hit hole: an f64
    /// evaluation landing exactly on a cheap literal while truly differing at 1e-17
    /// can no longer fold at serve, because nothing transcendental folds at serve.
    ///
    /// The `<constant>` arm's policy is unchanged (mask x special STAYS UNABSORBED --
    /// permanent policy against mu's own gradient, for the post-fit rationalizer;
    /// rationals still absorb; `e^C` via the exact exp identity is the one exception).
    /// Structural exact symbol algebra (bag cancellation `pi/pi -> 1`) is outside the
    /// fold and deliberately unaffected.
    /// H-045-R (owner Option B, 2026-08-05): exact-parity classification for
    /// `pow(<negative ground>, <beyond-certification integer literal>)`. Returns
    /// `Some(verdict)` when this arm DECIDES -- `Some(Some(lit))` folds, `Some(None)`
    /// refuses (the true class is Finite, which this fold never materializes) --
    /// and `None` to fall through to the interval class. Doctrine at the call site.
    ///
    /// H-051 (2026-08-05, extreme-literal lane): the arm also reads `Ex::Num`
    /// exponents -- an i128-parseable integer literal beyond 2^53 (e.g.
    /// `10000000000000000001`) parses to `Num`, whose f64 image differs from the
    /// denoted integer, so the interval layer brackets it and the continuum
    /// convention asserted Nan for a genuinely finite value
    /// (`pow(-9007199254740994, 10^19+1) -> nan` live, fuzz-extreme row 1756). The
    /// rational IS exact: integrality, sign, and parity come straight off it, at
    /// every magnitude. This is the "constructor parity fold on Num exponents"
    /// direction the H-045-R closure recorded.
    fn h045_exact_pow_class(&self, e: &Ex, bare: &Cx, ctx: &SimplifyCtx) -> Option<Option<Ex>> {
        let Ex::Pow(b, ex) = e else { return None };
        let view = bare.view;
        // The exponent: an exact rational, a bare numeric-string Leaf, or the Leaf's
        // `Mul[-1, Leaf]` negation (the only spellings the emitters produce for `pow`
        // second arguments; every other shape falls through and keeps today's
        // behavior). A non-integer Num falls through: the interval class's Nan for a
        // negative base at a non-integer exponent is CORRECT.
        let (neg, odd) = match &**ex {
            Ex::Num(r) => {
                if !r.is_integer() {
                    return None;
                }
                (r.is_negative(), r.num() % 2 != 0)
            }
            Ex::Leaf(t) => view.with_str(*t, crate::numeric::integer_literal_parity)?,
            Ex::Mul(v) if v.len() == 2 => match (&v[0], &v[1]) {
                (Ex::Num(r), Ex::Leaf(t)) if *r == crate::ac::rat::Rat::NEG_ONE => {
                    let (n, o) = view.with_str(*t, crate::numeric::integer_literal_parity)?;
                    (!n, o)
                }
                _ => return None,
            },
            _ => return None,
        };
        let exp_negative = neg;
        // The base's certified disposition, from its own rigorous value set.
        let flat_b = to_prefix(b, bare);
        let strs_b = self.resolve_seq(&flat_b, ctx);
        let vs =
            crate::interval::value_set(&strs_b, &self.operators, &crate::interval::Vs::reals())?;
        if vs.ninf && !vs.has_fin && !vs.pinf && !vs.nan {
            if exp_negative {
                return Some(None); // (-inf)^{-int} = 0: FINITE -- refuse here
            }
            // (-inf)^{+int} is +-inf by parity: extended-real exact arithmetic.
            return Some(Some(if odd { Ex::NegInf } else { Ex::PosInf }));
        }
        if vs.has_fin && vs.hi < 0.0 && !vs.pinf && !vs.ninf && !vs.nan {
            // A finite negative real to ANY integer power is a finite real (however
            // astronomic): class Finite -- refuse; never the a.e. arms' asserted Nan.
            return Some(None);
        }
        None
    }

    fn ac_fold(
        &self,
        e: &Ex,
        ctx: &SimplifyCtx,
        lossy: bool,
        class_memo: &RefCell<FxHashMap<Vec<crate::tokens::Tok>, Option<Ex>>>,
    ) -> Option<Ex> {
        let children: Vec<&Ex> = match e {
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().collect(),
            Ex::Pow(b, ex) => vec![b, ex],
            _ => return None,
        };
        let view = self.view(ctx);
        let bare = Cx::bare(&view);
        // EXTENDED-REAL GROUND CLASSIFICATION (owner-approved option B, 2026-08-02;
        // generalized from bag absorption): exact arithmetic BY CERTIFICATION, never
        // by evaluation. A GROUND composite (no variables, no Const anywhere) whose
        // rigorous interval CLASS is Nan / +inf / -inf folds to that literal: the
        // class of a ground expression is an exact fact (no finite value is read, so
        // no rounding can occur -- the fold-unification principle stands), and the
        // interval layer already carries every operator's domain and limit behavior
        // (`atanh(e*e)` classifies Nan from atanh's domain; `cosh(-inf)` classifies
        // +inf). Finite classes NEVER fold here -- finite ground transcendentals are
        // exactly where rounding lives, and they stay symbolic (mined, symbolically
        // certified rules carry the exact hits). Uncertifiable cases (Mixed, or a
        // pole reachable only through symbolic exactness like tan(acos 0)) stay
        // symbolic; the miner's hiprec channel still mints those rows as rules.
        let is_composite = matches!(e, Ex::Add(_) | Ex::Mul(_) | Ex::Fun(..) | Ex::Pow(..));
        if is_composite && is_ground(e, &view) {
            // PER-CALL MEMO (2026-08-02): a ground composite that survives (Finite/
            // Mixed class) is re-visited by the fold on EVERY pass, and interval
            // classification is the expensive step -- without the memo the arm cost
            // ~24us of the 74us corpus-walk mean. Keyed on the interned
            // serialization (per-call ctx, so Tok values are call-consistent); a
            // ground node's class fold is independent of `lossy`. The None verdict
            // is memoized too -- the common, repeated case is exactly the miss.
            let flat = to_prefix(e, &bare);
            if let Some(hit) = class_memo.borrow().get(&flat) {
                // Ground nodes carry no Const, so the Const arm below is a no-op
                // for them and returning the memoized None early is equivalent.
                return hit.clone();
            }
            let strs = self.resolve_seq(&flat, ctx);
            // H-045-R OVERRIDE (owner Option B, 2026-08-05): `pow` with a NEGATIVE
            // ground base and a beyond-certification INTEGER literal exponent (an
            // overlay Leaf: the spelling's denoted integer exceeds `i128`, so it is
            // neither a `Num` nor an interval-certifiable point -- the bracket spans
            // many integers and the a.e. arms would ASSERT Nan). The spelling itself
            // carries the exact facts: sign and parity (`integer_literal_parity`).
            //   * base = pure -inf:  (-inf)^{+int} IS +-inf by parity (extended-real
            //     exact arithmetic); a negative exponent gives 0 -- FINITE, so it
            //     REFUSES here (finite classes never fold in this arm).
            //   * base = certified negative FINITE: the true value is a finite real
            //     at every integer exponent (astronomic, but finite) -- class
            //     Finite, so the honest verdict is REFUSE, never the old Nan.
            //   * anything else (non-integer spelling, positive/mixed/nan base)
            //     falls through to `value_class` unchanged.
            // The interval layer keeps its documented continuum convention; this
            // arm reads the one thing the interval cannot see: the token. (Vs
            // point-support provenance is the RECORDED design direction if more
            // point-derived-enclosure consumers appear -- register H-045.)
            let verdict = match self.h045_exact_pow_class(e, &bare, ctx) {
                Some(decided) => decided,
                None => match crate::interval::value_class(&strs, &self.operators) {
                    // H-051 completion (2026-08-05): a Nan verdict on a composite
                    // CONTAINING a bracket-poisoned pow exponent (a Num or literal
                    // spelling whose exact integer the interval can only bracket) is
                    // untrustworthy -- the pow-node arm above cannot help once the
                    // poisoned node sits INSIDE the classified composite (the
                    // interval walk re-brackets it internally and its continuum
                    // convention asserts Nan for what is truly a parity-decided
                    // finite/inf value; fuzz-extreme row 1756's div chain). REFUSE
                    // the fold. Inf verdicts stay trusted: a poisoned pow can only
                    // leak Nan (never a definite infinity) through the walk, so a
                    // pure-inf class cannot have come from the bracket.
                    Some(crate::interval::Class::Nan) => {
                        if contains_bracket_poisoned_pow(e, &view) {
                            None
                        } else {
                            Some(Ex::NaN)
                        }
                    }
                    Some(crate::interval::Class::PosInf) => Some(Ex::PosInf),
                    Some(crate::interval::Class::NegInf) => Some(Ex::NegInf),
                    _ => None,
                },
            };
            class_memo.borrow_mut().insert(flat, verdict.clone());
            return verdict;
        }
        // GENERALIZED CONST-SHIFT ABSORPTION (P3', owner-approved 2026-08-02): an
        // Add/Mul bag holding a `<constant>` absorbs any SPECIAL-FREE GROUND member
        // whose interval-certified value is finite (Add) resp. finite and provably
        // nonzero (Mul): `C + g -> C'` is a bijective reparametrization of the fitted
        // family for every finite real g, and `C * g -> C'` for every finite nonzero
        // g. The mined const-absorption rules cover only the SPELLINGS the length-4
        // enumeration reaches -- the family is structural and closed under signs and
        // nesting the enumerator never sees (found live: `C + cosh(1)` absorbed by a
        // mined rule while `C - cosh(1)` survived for want of a sign twin). Members
        // with specials NEVER absorb (mask-x-special, permanent policy); `<constant>`
        // -bearing members are not ground and never qualify (Const-independence).
        if matches!(e, Ex::Add(_) | Ex::Mul(_)) && children.iter().any(|c| matches!(c, Ex::Const)) {
            let is_mul = matches!(e, Ex::Mul(_));
            let absorbable = |m: &Ex| -> bool {
                if matches!(m, Ex::Const) || !is_ground(m, &view) || has_special(m) {
                    return false;
                }
                let flat = to_prefix(m, &bare);
                let strs = self.resolve_seq(&flat, ctx);
                match crate::interval::value_set(
                    &strs,
                    &self.operators,
                    &crate::interval::Vs::reals(),
                ) {
                    Some(vs) => {
                        let fin = vs.cls() == crate::interval::Class::Finite && vs.fin_pm();
                        // Mul additionally needs g != 0: a zero factor collapses the
                        // product, which `C' ` cannot represent. Certify only strictly
                        // signed bounds; a straddle (or a NaN endpoint) fails closed.
                        fin && (!is_mul || vs.lo > 0.0 || vs.hi < 0.0)
                    }
                    None => false,
                }
            };
            let absorbed: Vec<bool> = children.iter().map(|c| absorbable(c)).collect();
            if absorbed.iter().any(|&b| b) {
                let kept: Vec<Ex> = children
                    .iter()
                    .zip(&absorbed)
                    .filter(|(_, &a)| !a)
                    .map(|(c, _)| (*c).clone())
                    .collect();
                return Some(if is_mul {
                    crate::ac::expr::mul(kept, &bare)
                } else {
                    crate::ac::expr::add(kept, &bare)
                });
            }
        }
        let const_or_fin = children.iter().all(|c| matches!(c, Ex::Const | Ex::Num(_)))
            && children.iter().any(|c| matches!(c, Ex::Const));
        if const_or_fin {
            if lossy {
                return Some(Ex::Const);
            }
            let flat = to_prefix(e, &bare);
            let strs = self.resolve_seq(&flat, ctx);
            let collapses =
                crate::interval::value_set(&strs, &self.operators, &crate::interval::Vs::reals())
                    .is_some_and(|vs| vs.cls() == crate::interval::Class::Finite && vs.fin_pm());
            if collapses {
                return Some(Ex::Const);
            }
        }
        None
    }

    /// Simplify through the AC core, returning the STRICT TAGGED prefix form -- the AC
    /// engine's native serialization (`<add> ... </add>` / `<mul> ... </mul>` bags, plain
    /// `pow`/function prefix, one-token exact literals, no sugar operators). Input accepts
    /// both the old grammar and the tagged form (one shared parser). `node_budget` bounds the
    /// outer rewrite iterations; `wildcard_all` is the LOSSY switch (training-corpus
    /// canonicalisation only). A malformed input is returned unchanged.
    pub fn ac_simplify(
        &self,
        tokens: &[String],
        node_budget: usize,
        wildcard_all: bool,
    ) -> Option<Vec<String>> {
        self.ac_simplify_proj(tokens, node_budget, wildcard_all, AcForm::Tagged)
    }

    /// [`Engine::ac_simplify`] with an explicit output projection. The SEARCH runs on the
    /// UNIQUE internal canonical form (stripped bag orders + primitive sums, `ac::expr`);
    /// the form is serialization only.
    pub fn ac_simplify_proj(
        &self,
        tokens: &[String],
        node_budget: usize,
        wildcard_all: bool,
        form: AcForm,
    ) -> Option<Vec<String>> {
        let (ctx, best) = self.ac_simplify_ex(tokens, node_budget, wildcard_all);
        // Malformed input is `None` -- the AC parser is the arbiter, and the FAILURE
        // SIGNAL propagates to the caller (the FFI raises ValueError). The old contract
        // returned the input unchanged, which silently passed garbage through the one
        // entry point whose inputs skip `is_valid` (audit Tier-2, 2026-08-03).
        let best = best?;
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let toks = match form {
            AcForm::Explicit => to_prefix(&best, &bare),
            AcForm::Tagged => to_prefix_tagged(&best, &bare),
        };
        Some(self.resolve_seq(&toks, &ctx))
    }

    /// The PRETTY INFIX rendering of the simplified expression: `x8 + 1.2*x3`, `-x0/3`,
    /// `(x0 + 1)^2`, `sin(x0)`. Round-trips through the infix parser (reserved constant
    /// names + core-symbol precedences; see `ac::convert::to_infix_pretty`).
    pub fn ac_simplify_infix(
        &self,
        tokens: &[String],
        node_budget: usize,
        wildcard_all: bool,
    ) -> Option<String> {
        let (ctx, best) = self.ac_simplify_ex(tokens, node_budget, wildcard_all);
        // `None` on malformed input, exactly as `ac_simplify_proj` (the FFI raises).
        let best = best?;
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        Some(to_infix_pretty(&best, &bare))
    }

    /// The SERVE-TIME ORDERING, exposed for the miner's acceptance gate: is `a` strictly
    /// below `b` in the lexicographic reduction ordering (semantic complexity, literal
    /// size, canonical total order -- `ac::rules::ordered_below`) the rewrite pass fires
    /// under? A mined rule whose target is NOT below what the engine already reaches
    /// would be dead on arrival -- the pass would refuse to fire it.
    pub fn ac_ordered_below(&self, a: &[String], b: &[String]) -> Option<bool> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let ta = self.intern_seq(a, &ctx);
        let tb = self.intern_seq(b, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let ea = canon(from_prefix(&ta, &bare)?, &bare);
        let eb = canon(from_prefix(&tb, &bare)?, &bare);
        Some(crate::ac::rules::ordered_below(&ea, &eb, &view))
    }

    /// The MINING JUDGE, fused: parse once, simplify, and return
    /// `(canonical input complexity, best complexity, best in the explicit projection)`.
    /// The AC parser is the arbiter (`None` on malformed input) -- no config-vocabulary
    /// gate, so the judge works under any mining vocabulary, including minimal test
    /// configs whose `is_valid` does not span the AC output language.
    /// The two scores are REFERENCE values (the input one is measured on the bare
    /// canonical form): coverage decisions -- prune, the strict-Kruskal skip, proposal
    /// verdicts -- judge in the serve ordering via `ac_ordered_below`, never by
    /// comparing these scores (one coverage ordering, everywhere).
    pub fn ac_judge(
        &self,
        tokens: &[String],
        node_budget: usize,
    ) -> Option<(u64, u64, Vec<String>)> {
        let src_c = self.ac_complexity(tokens)?;
        let (ctx, best) = self.ac_simplify_ex(tokens, node_budget, false);
        let best = best?;
        let view = self.view(&ctx);
        let best_c = complexity(&best, &view);
        let bare = Cx::bare(&view);
        let toks = to_prefix(&best, &bare);
        Some((src_c, best_c, self.resolve_seq(&toks, &ctx)))
    }

    /// The semantic complexity of an expression (either grammar), measured on its canonical
    /// form -- the same functional the simplify search minimizes (`ac::expr::complexity`).
    /// `None` on malformed input.
    pub fn ac_complexity(&self, tokens: &[String]) -> Option<u64> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(tokens, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let e = from_prefix(&toks, &bare)?;
        Some(complexity(&canon(e, &bare), &view))
    }

    /// CERTIFIED-canon complexity: the measure of the state the simplify CHAIN starts
    /// from (canon under the full certificate-carrying Cx, sound mode) -- the serve
    /// ordering's own pricing. For a simplify output the round-trip re-canon is the
    /// final state itself (the per-state `stable()` contract), so
    /// `mu_cert(simplify(e)) <= mu_cert(e)` is a THEOREM (chain descent, L3). The
    /// bare pricing above cannot promise that: a certificate-licensed respelling the
    /// bare context cannot re-derive keeps its own (possibly higher) measure --
    /// found live as 0.48% of 64k corpus rows measuring above ratio 1, in quanta of
    /// one symbol unit (2026-08-02).
    pub fn ac_complexity_certified(&self, tokens: &[String]) -> Option<u64> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(tokens, &ctx);
        let view = self.view(&ctx);
        let cf = |e: &Ex| self.ac_cert(e, &ctx, false);
        let cfz = |e: &Ex| self.ac_cert(e, &ctx, true);
        let czn = |e: &Ex| self.ac_zsn(e, &ctx);
        let cnc = |e: &Ex| self.ac_nce(e, &ctx);
        let cx = Cx {
            view: &view,
            cert_fin: Some(&cf),
            cert_finnz: Some(&cfz),
            cert_nzae: Some(&czn),
            cert_nce: Some(&cnc),
            lossy: false,
            sentinels_expired: false,
        };
        let bare = Cx::bare(&view);
        let e = from_prefix(&toks, &bare)?;
        Some(complexity(&canon(e, &cx), &view))
    }

    /// Canonical skeleton equality MODULO literal content (`ac::expr::eq_mod_nums`),
    /// for the resolution respell guard: a resolved target skeleton-equal to its mark
    /// differs only in literal values -- a respell, never structural recovery.
    /// `None` on malformed input (callers fail CLOSED: treat as skeleton-equal).
    pub fn ac_same_literal_skeleton(&self, a: &[String], b: &[String]) -> Option<bool> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let ta = self.intern_seq(a, &ctx);
        let tb = self.intern_seq(b, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let ea = canon(from_prefix(&ta, &bare)?, &bare);
        let eb = canon(from_prefix(&tb, &bare)?, &bare);
        Some(crate::ac::expr::eq_mod_nums(&ea, &eb, &view))
    }

    /// The simplify chain itself, entirely on the N-ARY canonical representation: ONE parse
    /// at the input boundary, then Ex-valued states throughout -- identity (the `seen` cycle
    /// guard, fixpoint detection) is decided by `Ex` equality, never by any serialization.
    /// The METRIC is the semantic complexity functional on the canonical form itself
    /// (`ac::expr::complexity`) -- no serialization plays any runtime role in the chain.
    /// Debug builds additionally assert per state that
    /// parse -> canon -> serialize is the identity (the uniqueness contract of the canonical
    /// form) and that no nested bags survive (the Flat invariant).
    ///
    /// Returns the per-call ctx and the best (shortest-measuring) canonical state; `None` on
    /// a malformed input.
    fn ac_simplify_ex(
        &self,
        tokens: &[String],
        node_budget: usize,
        wildcard_all: bool,
    ) -> (SimplifyCtx, Option<Ex>) {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(tokens, &ctx);
        let view = self.view(&ctx);

        let cf = |e: &Ex| self.ac_cert(e, &ctx, false);
        let cfz = |e: &Ex| self.ac_cert(e, &ctx, true);
        let czn = |e: &Ex| self.ac_zsn(e, &ctx);
        let cnc = |e: &Ex| self.ac_nce(e, &ctx);
        let cx = Cx {
            view: &view,
            cert_fin: Some(&cf),
            cert_finnz: Some(&cfz),
            cert_nzae: Some(&czn),
            cert_nce: Some(&cnc),
            lossy: wildcard_all,
            sentinels_expired: false,
        };
        // Parse with the CALL's MODE: a bare (sound) parse let sound-only
        // constructor arms destroy structure lossy passes need -- `inv
        // float("inf")` folded to 0 AT PARSE, so the mul bag never saw the
        // x * x^-1 pair the relaxed $-certificate cancels (`(inf/inf) * x0`,
        // 2026-08-02). The parse Cx stays CERT-LESS: `canon(e0, &cx)` below
        // immediately re-runs full cert-carrying construction, so certs at parse
        // are pure double work (measured +15us corpus-walk mean).
        let mut pbare = Cx::bare(&view);
        pbare.lossy = wildcard_all;
        let Some(e0) = from_prefix(&toks, &pbare) else {
            return (ctx, None);
        };
        let mcx = MCx {
            view: &view,
            cert_fin: Some(&cf),
            cert_finnz: Some(&cfz),
            wildcard_all,
        };
        let class_memo: RefCell<FxHashMap<Vec<crate::tokens::Tok>, Option<Ex>>> =
            RefCell::new(FxHashMap::default());
        let fold = |e: &Ex| self.ac_fold(e, &ctx, wildcard_all, &class_memo);
        let pass = PassCtx {
            rules: self.ac_rules(),
            cx: &cx,
            mcx: &mcx,
            fold: &fold,
            fires: Cell::new(0),
            normal: RefCell::new(FxHashSet::default()),
            explore: true,
        };

        let stable_in = |e: &Ex, pb: &Cx, cxx: &Cx| {
            let t = to_prefix(e, cxx);
            // Re-parse under the PHASE's MODE (cert-less, like the entry parse):
            // serialization stability is a PER-MODE contract -- phase 2 states must
            // round-trip under the sentinel-expired canon, phase 1 states under the
            // sentinel-keeping one.
            //
            // STATE-level identity, not token-level (H-014 hardening, 2026-08-03): two
            // DISTINCT canonical states can share one serialization (the display-
            // redistribution class), and a token comparison certifies both -- exactly
            // how the factored/distributed sign-orientation pair hid from this assert.
            // The round-trip must land on the SAME Ex.
            from_prefix(&t, pb)
                .map(|p| {
                    let p = canon(p, cxx);
                    if p != *e {
                        let s = |v: &[crate::tokens::Tok]| {
                            v.iter()
                                .map(|x| view.resolve_owned(*x))
                                .collect::<Vec<_>>()
                                .join(" ")
                        };
                        eprintln!(
                            "STABLE-DIFF (state)\n  T : {}\n  T': {}\n  e : {e:?}\n  p : {p:?}",
                            s(&t),
                            s(&to_prefix(&p, cxx))
                        );
                    }
                    p == *e
                })
                .unwrap_or(false)
        };
        let stable = |e: &Ex| stable_in(e, &pbare, &cx);

        let mut current = canon(e0, &cx);
        debug_assert!(
            stable(&current),
            "canonical state is not serialization-stable: {current:?}"
        );
        // Every pass output descends the reduction ordering at the root -- fires, folds and
        // rebuilds are all descent-gated (Lemma L3, docs/formal.md) -- so the chain of pass
        // outputs strictly descends until a fixpoint: cycles are impossible, no seen-set is
        // needed (L4), and because the ordering is well-founded the fixpoint arrives in
        // finitely many passes as a theorem (T6). This budget is defense-in-depth, like the
        // pass-level step cap. The answer is the FINAL state: by L3 it carries the chain's
        // minimum complexity anyway, and returning anything earlier ships a non-fixpoint
        // (the former `best`-tracking did exactly that -- an earlier state of equal
        // complexity but higher literal-size/canonical rank -- a non-fixpoint, so
        // idempotence would break). A budget-truncated run returns the last state
        // reached, which is sound.
        for _ in 0..node_budget.max(1) {
            let next = rewrite_pass(current.clone(), &pass);
            debug_assert!(
                crate::ac::rules::no_nested_bags(&next),
                "pass output has nested bags: {next:?}"
            );
            debug_assert!(
                stable(&next),
                "pass output is not serialization-stable: {next:?}"
            );
            if next == current {
                break; // fixpoint, decided on the representation itself
            }
            current = next;
        }
        // SENTINEL EXPIRY, phase 2 (H-015 class (a), 2026-08-04): the lossy parse keeps
        // `inv(inf)`-class reciprocals unfolded so the `$`-cancel can find its partner
        // (mask-sentinel doctrine) -- a licence for the SEARCH, not the answer. At the
        // phase-1 fixpoint every surviving sentinel is unpartnered and IS the value 0,
        // and keeping it both inflates the endpoint and BLOCKS the rules its unfolded
        // shape hides from (measured: 609/750 mode-ordering rows carried an inf/nan
        // the sound endpoint lacked -- a kept `C * inv(inf)` term froze its sum short
        // of the shape the cos-parity rule needed). Phase 2 re-canonicalizes the
        // fixpoint with the fold re-enabled and keeps descending under the SAME lossy
        // licences; its states satisfy the per-mode stability contract under the
        // expired canon. Idempotence: the output carries no foldable sentinel, so a
        // fresh call's phase 1 reproduces it and its phase 2 is a no-op; if a phase-1
        // rewrite ever re-mints a sentinel from sentinel-free input, phase 2 folds it
        // again -- the pipeline, not the phase, is the projection consumers see.
        if wildcard_all {
            let cx2 = Cx {
                view: &view,
                cert_fin: Some(&cf),
                cert_finnz: Some(&cfz),
                cert_nzae: Some(&czn),
                cert_nce: Some(&cnc),
                lossy: true,
                sentinels_expired: true,
            };
            let mut pbare2 = Cx::bare(&view);
            pbare2.lossy = true;
            pbare2.sentinels_expired = true;
            let pass2 = PassCtx {
                rules: self.ac_rules(),
                cx: &cx2,
                mcx: &mcx,
                fold: &fold,
                fires: Cell::new(0),
                normal: RefCell::new(FxHashSet::default()),
                explore: true,
            };
            let stable2 = |e: &Ex| stable_in(e, &pbare2, &cx2);
            current = canon(current, &cx2);
            debug_assert!(
                stable2(&current),
                "expired canonical state is not serialization-stable: {current:?}"
            );
            for _ in 0..node_budget.max(1) {
                let next = rewrite_pass(current.clone(), &pass2);
                debug_assert!(
                    crate::ac::rules::no_nested_bags(&next),
                    "phase-2 pass output has nested bags: {next:?}"
                );
                debug_assert!(
                    stable2(&next),
                    "phase-2 pass output is not serialization-stable: {next:?}"
                );
                if next == current {
                    break;
                }
                current = next;
            }
        }
        // LOSSY OUTPUT PROJECTION (H-015, 2026-08-04): the returned endpoint gets the
        // mu-cheaper reciprocal spelling (`(a*b)^-1` vs `a^-1 * b^-1`) where the joined
        // form wins. Boundary-only by design: the chain above runs entirely in the
        // distributed working canon (per-state `stable()` holds for THAT canon), and the
        // projected state deliberately never re-enters it -- re-simplifying the projected
        // spelling re-parses through `pow`'s blanket distribution back to the same
        // working endpoint, whose projection reproduces the output (idempotence as a
        // funnel property; see `ac::expr::rejoin_reciprocals`).
        let current = if wildcard_all {
            rejoin_projection(current, &cx)
        } else {
            current
        };
        (ctx, Some(current))
    }
}

/// Output projections of the AC canonical state (see `ac::convert`): `Tagged` is the native
/// strict prefix form; `Explicit` is the sugared old-token diagnostic form (the internal
/// canonical form's sugared spelling, parseable by the binary engine -- the
/// differential-testing oracle);
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AcForm {
    Tagged,
    Explicit,
}

#[cfg(test)]
mod tests {
    use crate::Engine;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    fn t(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    /// H-051 (2026-08-05, extreme-literal lane): `Ex::Num` exponents classify by the
    /// exact rational -- an i128-parseable beyond-2^53 literal parses to Num, whose
    /// f64 image differs from the denoted integer; the interval bracket then made the
    /// continuum convention assert Nan for a genuinely FINITE value
    /// (`pow(-9007199254740994, 10^19+1) -> nan` live, fuzz-extreme row 1756).
    #[test]
    fn h051_num_exponents_classify_exactly() {
        let Some(e) = engine() else { return };
        let s = |toks: &[&str]| {
            e.ac_simplify_proj(&t(toks), 48, false, super::AcForm::Explicit)
                .unwrap()
        };
        let odd = "10000000000000000001"; // 10^19 + 1: Num, not f64-representable
        let even = "10000000000000000000"; // 10^19
        assert_eq!(s(&["pow", "float(\"-inf\")", odd]), t(&["float(\"-inf\")"]));
        assert_eq!(s(&["pow", "float(\"-inf\")", even]), t(&["float(\"inf\")"]));
        // (-inf)^{-odd}: the constructor decomposes pow(x, -n) = inv(pow(x, n)),
        // this arm folds the inner to -inf, and inv(-inf) = 0 EXACTLY (total
        // extended-real arithmetic) -- unlike the opaque-Leaf negative case, which
        // cannot decompose and refuses.
        assert_eq!(
            s(&["pow", "float(\"-inf\")", "-10000000000000000001"]),
            t(&["0"])
        );
        // finite-negative-base at ANY integer exponent is a FINITE class: refusal,
        // never the asserted Nan (row 1756's chain).
        assert_eq!(
            s(&["pow", "-", "-3", "9007199254740991", odd]),
            t(&["pow", "-9007199254740994", odd])
        );
        // non-integer Num exponents keep the interval class's correct Nan
        assert_eq!(
            s(&["pow", "-9007199254740994", "/", odd, "3"]),
            t(&["float(\"nan\")"])
        );
        // The ENCLOSING-composite guard (`contains_bracket_poisoned_pow`): row
        // 1756's full div chain -- the outer ground node's interval class asserts
        // Nan through the internally re-bracketed exponent, and the fold must
        // refuse rather than ship it (the pow-node arm cannot see the outer node).
        let chain = [
            "/",
            "pow",
            "-",
            "-3",
            "9007199254740991",
            odd,
            "-0.9999999999999999",
        ];
        let kept = e.ac_simplify(&t(&chain), 48, false).unwrap();
        assert_eq!(
            kept,
            // H-059 (2026-08-07): the coefficient now takes the DIVISOR side, per the
            // ratified §8 rule restored for out-of-bound fractions -- `0.9999999999999999`
            // (18 chars) against `10000000000000000/9999999999999999` (34). Same value,
            // spelling only; the classification this test guards is the `pow` pair, which
            // is unchanged. The sign splits out as `-1` (H-020: never inside `<div>`).
            t(&[
                "<mul>",
                "-1",
                "pow",
                "-9007199254740994",
                odd,
                "<div>",
                "0.9999999999999999",
                "</mul>"
            ])
        );
        assert_eq!(e.ac_simplify(&kept, 48, false).unwrap(), kept);
    }

    /// H-045-R (owner Option B, 2026-08-05): `pow` with a negative ground base and a
    /// beyond-i128 INTEGER literal exponent classifies EXACTLY from the spelling's
    /// sign and parity -- the old path asserted Nan through the interval bracket's
    /// a.e. convention (the bracket at 1e40 spans many integers, so the H-029
    /// single-integer honesty gate could not fire). Finite negative bases REFUSE
    /// (the true value is a finite real at every integer exponent); (-inf) folds by
    /// parity; non-integer spellings keep the correct Nan; the certified-point
    /// (i128-reachable) and +inf paths are untouched.
    #[test]
    fn h045_beyond_i128_integer_exponents_classify_exactly() {
        let Some(e) = engine() else { return };
        let s = |toks: &[&str]| {
            e.ac_simplify_proj(&t(toks), 48, false, super::AcForm::Explicit)
                .unwrap()
        };
        let odd_huge = "10000000000000000000000000000000000000001"; // 10^40 + 1
                                                                    // (-inf)^{+int}: extended-real exact, sign by parity.
        assert_eq!(
            s(&["pow", "float(\"-inf\")", "1e40"]),
            t(&["float(\"inf\")"])
        );
        assert_eq!(
            s(&["pow", "float(\"-inf\")", odd_huge]),
            t(&["float(\"-inf\")"])
        );
        // (-inf)^{-int} = 0 is FINITE: sound refusal, in both negation spellings.
        // H-048: the SIGNED leaf spelling "-1e40" splits into structure at parse
        // (the sign is never inside an opaque leaf), so the kept form is the
        // structural negation -- identical to the `neg`-spelled twin below.
        assert_eq!(
            s(&["pow", "float(\"-inf\")", "-1e40"]),
            t(&["pow", "float(\"-inf\")", "neg", "1e40"])
        );
        assert_eq!(
            s(&["pow", "float(\"-inf\")", "neg", "1e40"]),
            t(&["pow", "float(\"-inf\")", "neg", "1e40"])
        );
        // Finite negative bases: finite class, REFUSE (the old path shipped nan).
        assert_eq!(s(&["pow", "(-2)", "1e40"]), t(&["pow", "-2", "1e40"]));
        assert_eq!(s(&["pow", "(-1)", "1e40"]), t(&["pow", "-1", "1e40"]));
        // A ground COMPOSITE base classifying -inf takes the same arm.
        assert_eq!(s(&["pow", "log", "0", "1e40"]), t(&["float(\"inf\")"]));
        // Untouched paths: +inf base; non-integer spelling keeps the correct Nan;
        // the i128-certified point path (H-045 proper).
        assert_eq!(
            s(&["pow", "float(\"inf\")", "1e40"]),
            t(&["float(\"inf\")"])
        );
        assert_eq!(s(&["pow", "(-2)", "1.5e-40"]), t(&["float(\"nan\")"]));
        assert_eq!(
            s(&["pow", "float(\"-inf\")", "1e19"]),
            t(&["float(\"inf\")"])
        );
    }

    /// REGRESSION (64k/1M-gate idempotence rows 29663/37873/59042 + 274133/514869): a
    /// KEPT-ZERO term (`Mul[0, t]`, unlicensed zero-collapse) is sign-free, but
    /// `negate_term` used to thread `-1` through `term_join` and mint a non-canonical
    /// `Mul[-1, 0, t]` twin whose coefficient read -1 while the canonical spelling reads
    /// +1 -- the sum orientation comparison lost flip-antisymmetry and BOTH sign
    /// spellings of the enclosing sum were stable. `term_join` now lets a zero head
    /// absorb any coefficient; this input (the tagged output of 64k row 29663) must
    /// reach its unique spelling in one call.
    #[test]
    fn ac_kept_zero_sign_regression() {
        // Constructor-level mechanism (term_join's zero-head absorption): reproduces on
        // the generation-2 test engine; the legacy 4-3 pairing this row was FOUND on is
        // refused at load since the generation gate (audit Tier-1 #1/#3).
        let Some(e) = crate::test_engine() else {
            return;
        };
        let t1: Vec<String> = "<add> <sub> <mul> 0 <div> <add> x3 <sub> pow <add> x3 asinh x3 </add> 0.25 </add> </mul> <mul> x3 atanh x3 </mul> </add>"
            .split_whitespace().map(str::to_string).collect();
        let t2 = e.ac_simplify(&t1, 48, false).unwrap();
        let t3 = e.ac_simplify(&t2, 48, false).unwrap();
        assert_eq!(t2, t3, "kept-zero sign spelling must be unique");
        // The sign-free zero term displays in the POSITIVE section: `<add> <mul> 0 ...`.
        assert_eq!(&t2[..2], &["<add>".to_string(), "<mul>".to_string()]);
    }

    /// PIN (H-031, C1 spec-writing read, 2026-08-05): a kept-zero product over an
    /// UNLICENSED sum (`0 * (x0 + log x1)`) entering a wider Add distributes through the
    /// factored-sum flatten arm with r = 0. The scaled terms used to be rebuilt by raw
    /// `term_join` -- canonical-form-preserving only for NONZERO coefficients -- minting
    /// locally non-canonical `Mul[0, x0]` transients that every exposure surface then
    /// healed bottom-up (entry `canon` re-walks the parse output; the pass rebuild
    /// re-walks every child), so no state ever escaped un-healed: NOT a live defect,
    /// but a five-step global argument where a local invariant belongs. `scale_term`
    /// now routes the zero-coefficient case through `mul()`, which owns the kept-zero
    /// licence logic, so scaled terms are canonical by construction and the healing
    /// argument is unnecessary. This test pins the exposed contract either way.
    #[test]
    fn ac_kept_zero_distribution_entry_state_is_stable() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        // In debug builds the stable() assert inside the chain is the real check; the
        // release-observable contract is idempotence plus the licensed collapse of the
        // finite part (0*x0 drops, the unlicensed 0*log(x1) survives).
        let t1 = t(&["+", "x2", "*", "0", "+", "x0", "log", "x1"]);
        let s1 = e.ac_simplify(&t1, 48, false).unwrap();
        let s2 = e.ac_simplify(&s1, 48, false).unwrap();
        assert_eq!(s1, s2, "kept-zero distribution must be idempotent");
        assert_eq!(
            s1,
            t(&["<add>", "x2", "<mul>", "0", "log", "x1", "</mul>", "</add>"])
        );
        // LOSSY mode: every factor is licence-blanketed, the zero collapses outright.
        let l1 = e.ac_simplify(&t1, 48, true).unwrap();
        assert_eq!(l1, t(&["x2"]));
    }

    /// REGRESSION (64k-gate row 65198, shipped 4-3): a sum containing an uncertified
    /// flip-symmetric pair plus an asymmetric term (`acosh x16 - acosh x16 - x17*(x16+x17)`)
    /// made the former positive-lead orientation rule ill-defined -- canon flip-flopped
    /// between two spellings and the rebuild re-walk climbed forever (stack overflow).
    /// Fixed twice over: `flipped_orientation_wins` decides the sign on the orientation
    /// CLASS (confluent constructors), and a disoriented rebuild is refused in
    /// `rewrite_pass` (termination stays a theorem regardless).
    #[test]
    fn ac_flip_symmetric_orientation_regression() {
        // Constructor-level mechanism (orientation-class sign decision + disoriented-
        // rebuild refusal): reproduces on the generation-2 test engine (mult3 respelled
        // as its generation-2 section `* 3 x16`).
        let Some(e) = crate::test_engine() else {
            return;
        };
        let toks: Vec<String> =
            "/ sinh - x16 x17 + * x16 x17 / * 3 x16 sin - - acosh x16 acosh x16 * + x17 x16 x17"
                .split_whitespace()
                .map(str::to_string)
                .collect();
        let out = e.ac_simplify(&toks, 48, false).unwrap();
        // Terminates, is idempotent, and keeps the cert-refused acosh-acosh pair.
        assert_eq!(e.ac_simplify(&out, 48, false).unwrap(), out);
        assert_eq!(out.iter().filter(|t| *t == "acosh").count(), 2);
    }

    /// EMIT-PARSE CLOSURE (the serialization-language family): the engine must re-read its
    /// own output under ANY config. The serializers emit a FIXED core language
    /// (`+ - * / neg inv pow rootn`, tags, literal spellings) regardless of the config
    /// vocabulary, so every core spelling is an engine built-in on the parse side too.
    /// Adversarial battery: each core sugar token individually missing from an
    /// otherwise-full config, plus two minimal vocabularies; the COMPLETE L<=3 prefix
    /// universe of each config's own vocabulary (the exhaustive L<=4 twin runs in the
    /// Python suite against the release build); both prefix projections; a full-vocabulary
    /// referee engine as the semantic oracle.
    #[test]
    fn serialization_closure_under_degenerate_configs() {
        const SPECS: [(&str, &str); 11] = [
            (
                "+",
                "{realization: '+', arity: 2, precedence: 1, commutative: true}",
            ),
            ("-", "{realization: '-', arity: 2, precedence: 1}"),
            (
                "neg",
                "{realization: simplipy.operators.neg, arity: 1, precedence: 2.5}",
            ),
            (
                "*",
                "{realization: '*', arity: 2, precedence: 2, commutative: true}",
            ),
            (
                "/",
                "{realization: simplipy.operators.div, arity: 2, precedence: 2}",
            ),
            (
                "inv",
                "{realization: simplipy.operators.inv, arity: 1, precedence: 4}",
            ),
            (
                "pow",
                "{realization: simplipy.operators.pow, arity: 2, precedence: 3}",
            ),
            ("exp", "{realization: np.exp, arity: 1, precedence: 3}"),
            ("log", "{realization: np.log, arity: 1, precedence: 3}"),
            ("sin", "{realization: np.sin, arity: 1, precedence: 3}"),
            (
                "abs",
                "{realization: simplipy.operators.abs, arity: 1, precedence: 3}",
            ),
        ];
        fn build(ops: &[&str]) -> Engine {
            let body: String = SPECS
                .iter()
                .filter(|(n, _)| ops.contains(n))
                .map(|(n, s)| format!("  {n:?}: {s}\n"))
                .collect();
            Engine::from_strs(&format!("operators:\n{body}"), "[]").expect("engine builds")
        }
        fn universe(ops: &[(String, u8)], budget: usize, seen: &mut Vec<Vec<String>>) {
            fn gen(ops: &[(String, u8)], budget: usize) -> Vec<Vec<String>> {
                let mut v: Vec<Vec<String>> = ["x0", "x1", "2", "-1", "<constant>"]
                    .iter()
                    .map(|l| vec![l.to_string()])
                    .collect();
                for (op, a) in ops {
                    let a = *a as usize;
                    if budget < 1 + a {
                        continue;
                    }
                    if a == 1 {
                        for sub in gen(ops, budget - 1) {
                            let mut e = vec![op.clone()];
                            e.extend(sub);
                            v.push(e);
                        }
                    } else {
                        for left in gen(ops, budget - 2) {
                            let rem = budget - 1 - left.len();
                            if rem == 0 {
                                continue;
                            }
                            for right in gen(ops, rem) {
                                let mut e = vec![op.clone()];
                                e.extend(left.iter().cloned());
                                e.extend(right);
                                v.push(e);
                            }
                        }
                    }
                }
                v
            }
            for e in gen(ops, budget) {
                if !seen.contains(&e) {
                    seen.push(e);
                }
            }
        }
        let full: Vec<&str> = SPECS.iter().map(|(n, _)| *n).collect();
        let referee = build(&full);
        let state = |toks: &[String]| referee.ac_simplify(toks, 48, false).unwrap();
        let mut batteries: Vec<Vec<&str>> = ["neg", "inv", "-", "/", "+", "*"]
            .iter()
            .map(|drop| full.iter().copied().filter(|n| n != drop).collect())
            .collect();
        batteries.push(vec!["+", "*", "exp", "sin"]);
        batteries.push(vec!["*", "inv", "log"]);
        for ops in &batteries {
            let e = build(ops);
            let arities: Vec<(String, u8)> = SPECS
                .iter()
                .filter(|(n, _)| ops.contains(n))
                .map(|(n, s)| (n.to_string(), if s.contains("arity: 2") { 2 } else { 1 }))
                .collect();
            let mut uni = Vec::new();
            universe(&arities, 3, &mut uni);
            for src in &uni {
                let want = state(src);
                let tagged = e.ac_simplify(src, 48, false).unwrap();
                let explicit = e
                    .ac_simplify_proj(src, 48, false, super::AcForm::Explicit)
                    .unwrap();
                for out in [&tagged, &explicit] {
                    assert!(
                        e.ac_complexity(out).is_some(),
                        "cfg {ops:?}: {src:?} -> {out:?} does not re-parse"
                    );
                    assert_eq!(
                        &e.ac_simplify(out, 48, false).unwrap(),
                        &tagged,
                        "cfg {ops:?}: {src:?} -> {out:?} is not a fixpoint spelling"
                    );
                    assert_eq!(
                        state(out),
                        want,
                        "cfg {ops:?}: {src:?} -> {out:?} changed canonical state"
                    );
                }
                // The boundary validator accepts the explicit projection (the gate that
                // used to raise "invalid or malformed prefix expression" on own output).
                assert!(
                    e.is_valid(&explicit),
                    "cfg {ops:?}: is_valid rejects own output {explicit:?}"
                );
                // The mining acceptance path DECIDES on own output (None = silent refusal).
                assert_eq!(
                    e.ac_ordered_below(&explicit, &explicit),
                    Some(false),
                    "cfg {ops:?}: ordering acceptance cannot read own output {explicit:?}"
                );
            }
        }
    }

    /// ORIENTATION CLOSURE (rules fire modulo the sign orientation class): a rule
    /// `L -> T` entails `-L -> -T`, but a sum and its negation are DIFFERENT canonical
    /// states -- negating a sum redistributes the sign into the term coefficients, so
    /// the original Add never survives as a subtree and the rule never met the flipped
    /// subject. Translate now mints the negated twin of every admitted rule whose
    /// flipped LHS absorbs the sign, through the same gates every rule passes.
    #[test]
    fn rules_fire_modulo_sum_orientation() {
        let cfg = r#"operators:
  "+": {realization: '+', arity: 2, precedence: 1, commutative: true}
  "-": {realization: '-', arity: 2, precedence: 1}
  "*": {realization: '*', arity: 2, precedence: 2, commutative: true}
  neg: {realization: simplipy.operators.neg, arity: 1, precedence: 2.5}
  pow: {realization: simplipy.operators.pow, arity: 2, precedence: 3}
  sin: {realization: np.sin, arity: 1, precedence: 3}
  cos: {realization: np.cos, arity: 1, precedence: 3}
"#;
        let rules = r#"[[["-","1","*","2","pow","sin","?0","2"],["cos","*","2","?0"]]]"#;
        let e = Engine::from_strs(cfg, rules).expect("engine builds");
        let t = |s: &str| -> Vec<String> { s.split_whitespace().map(str::to_string).collect() };
        let ex = |toks: &[String]| {
            e.ac_simplify_proj(toks, 48, false, super::AcForm::Explicit)
                .unwrap()
        };
        // The rule's own orientation fires (sanity).
        assert_eq!(ex(&t("- 1 * 2 pow sin x0 2")), t("cos * 2 x0"));
        // The FLIPPED subject 2sin^2(x) - 1 == -cos(2x) fires via the minted twin.
        assert_eq!(ex(&t("- * 2 pow sin x0 2 1")), t("neg cos * 2 x0"));
        // Sub-multiset semantics carry over: a remainder rides along untouched.
        assert_eq!(
            ex(&t("+ x1 - * 2 pow sin x0 2 1")),
            ex(&t("+ x1 neg cos * 2 x0"))
        );
        // Accounting: 1 raw + 1 twin serve; loading BOTH orientations explicitly
        // dedupes the twin pass to the identical serving set.
        let (kept, _, _, twins) = e.ac_rules_info();
        assert_eq!((kept, twins), (2, 1));
        let both = r#"[[["-","1","*","2","pow","sin","?0","2"],["cos","*","2","?0"]],
                       [["-","*","2","pow","sin","?0","2","1"],["neg","cos","*","2","?0"]]]"#;
        let e2 = Engine::from_strs(cfg, both).expect("engine builds");
        let (kept2, _, _, twins2) = e2.ac_rules_info();
        assert_eq!((kept2, twins2), (2, 0));
    }

    /// Translation loses NOTHING (`dropped == 0`, invariant), and SUBSUMPTION measures how
    /// far the engine has advanced past the artifact's mint-era canon: the mine refuses to
    /// mint what the engine derives natively, so a matched pair translates mint-pristine
    /// and an artifact mined by an OLDER engine carries exactly the rules the constructors
    /// have since absorbed. (The dev-era claim -- "exact arithmetic subsumes the
    /// coefficient-rule family" -- retired with the dev_7-3 asset; the census numbers are
    /// pinned in `remine/gate_acj.py` REFS, this in-crate twin pins the PROPERTY.)
    ///
    /// The fixture is the PUBLISHED asset in `~/.cache/simplipy`, and since the 2026-08-08
    /// HF republish the pair is MATCHED again (asset mined 2026-08-08: 6661 rules), so both
    /// counts sit at their pristine 0. History of the lag this pin surfaced while the pair
    /// was unmatched: the 2026-08-04 asset (995 rules) fell 88 behind on 2026-08-07 (33 to
    /// "a power of `exp` is an `exp`" incl. its `exp(1) -> E` write-half, 55 to the parity
    /// lattice `sign_blind_rep`), then ran to 143 subsumed + 1 dropped as the inverse-pair
    /// table, the reciprocal-base arms, the ground-fold value licence and the half-period
    /// arm landed. The single drop was `tan - np.pi !0 -> neg tan !0`, whose LHS the
    /// half-period arm canonicalizes to `tan(neg !0)` at the SAME mu -- a mu-neutral
    /// respell, no capability lost, and the re-mined artifact no longer contains it.
    /// If either count moves off 0 again, the published asset lags the engine:
    /// REPUBLISH, do not re-pin.
    #[test]
    fn ac_translation_stats() {
        let Some(e) = engine() else { return };
        let (kept, subsumed, dropped, twins) = e.ac_rules_info();
        eprintln!(
            "AC translation: kept={kept} subsumed={subsumed} dropped={dropped} twins={twins}"
        );
        assert!(kept > 0, "no rules survived translation");
        assert_eq!(
            dropped, 0,
            "published asset must translate loss-free; a drop means it lags the engine -- \
             republish, do not re-pin"
        );
        assert_eq!(
            subsumed, 0,
            "published asset (2026-08-08) is mint-pristine against this engine; subsumption \
             means the canon moved past it -- republish, do not re-pin"
        );
        let _ = twins;
    }

    /// The campaign's two defect axes, closed at the representation level. Self-contained:
    /// an inline engine carrying exactly the rule the ORDER axis needs (the historical
    /// `* tan A cos A -> sin A`, whose 5-token source exceeds the acj-4-3 length-4 mine
    /// universe -- the axis is an ENGINE property, not an artifact property).
    #[test]
    fn ac_defect_axes_closed() {
        let cfg = r#"operators:
  "+": {realization: '+', arity: 2, precedence: 1, commutative: true}
  "-": {realization: '-', arity: 2, precedence: 1}
  "*": {realization: '*', arity: 2, precedence: 2, commutative: true}
  "/": {realization: '/', arity: 2, precedence: 2}
  neg: {realization: simplipy.operators.neg, arity: 1, precedence: 2.5}
  pow: {realization: simplipy.operators.pow, arity: 2, precedence: 3}
  sin: {realization: np.sin, arity: 1, precedence: 3}
  cos: {realization: np.cos, arity: 1, precedence: 3}
  tan: {realization: np.tan, arity: 1, precedence: 3}
"#;
        let rules = r#"[[["*","tan","?0","cos","?0"],["sin","?0"]]]"#;
        let e = Engine::from_strs(cfg, rules).expect("engine builds");
        // ORDER: `* tan(A) cos(A) -> sin(A)` fired in the old engine while `* cos(A) tan(A)`
        // did not. Both spellings must now give the same (and simplified) answer.
        let a = e
            .ac_simplify(&t(&["*", "tan", "x0", "cos", "x0"]), 48, false)
            .unwrap();
        let b = e
            .ac_simplify(&t(&["*", "cos", "x0", "tan", "x0"]), 48, false)
            .unwrap();
        assert_eq!(a, b, "order invariance");
        assert_eq!(a, t(&["sin", "x0"]), "the rule fires through the bag");
        // ADJACENCY: `+ x3 (+ x8 (x3/5))` never collected in the old engine because the two
        // x3 spellings were never adjacent. The bag collects them by construction. The native
        // output is the strict tagged form.
        let out = e
            .ac_simplify(&t(&["+", "x3", "+", "x8", "/", "x3", "5"]), 48, false)
            .unwrap();
        assert_eq!(
            out,
            t(&["<add>", "<mul>", "6", "x3", "<div>", "5", "</mul>", "x8", "</add>"])
        );
    }

    /// The explicit form never spells a hyper-operator, and outputs stay valid old-grammar
    /// prefix expressions. Runs on the legacy-sugar fixture: the hyper-operator INPUTS
    /// need a table that declares them (in-repo, no asset).
    #[test]
    fn ac_output_language() {
        let e = crate::legacy_sugar_engine();
        let cases: &[&[&str]] = &[
            &["mult4", "x0"],
            &["pow3", "+", "x0", "x1"],
            &["div2", "mult4", "x0"],
            &[
                "+", "x0", "+", "x0", "+", "x0", "+", "x0", "+", "x0", "+", "x0", "x0",
            ],
        ];
        for c in cases {
            let out = e.ac_simplify(&t(c), 48, false).unwrap();
            for tok in &out {
                let hyper = tok.starts_with("mult")
                    || tok.starts_with("div")
                    || (tok.starts_with("pow") && tok.len() > 3);
                assert!(!hyper, "hyper-operator {tok} in output {out:?} for {c:?}");
            }
            // Tagged validity = the shared parser round-trips it (is_valid only knows the
            // old grammar).
            let again = e.ac_simplify(&out, 48, false).unwrap();
            assert_eq!(again, out, "tagged output fails round-trip for {c:?}");
        }
        // 7x arrives as an explicit literal coefficient in a tagged bag.
        let out = e
            .ac_simplify(
                &t(&[
                    "+", "x0", "+", "x0", "+", "x0", "+", "x0", "+", "x0", "+", "x0", "x0",
                ]),
                48,
                false,
            )
            .unwrap();
        assert_eq!(out, t(&["<mul>", "7", "x0", "</mul>"]));
    }

    /// The three projections and the pretty infix rendering of one canonical answer.
    /// Runs on the legacy-sugar fixture: the inputs deliberately use the retired sugar
    /// (`div5`, `mult2`, `div3`) whose desugar-and-respell IS part of the subject.
    #[test]
    fn ac_projections() {
        let e = crate::legacy_sugar_engine();
        let expr = t(&["+", "x3", "+", "x8", "div5", "x3"]);
        assert_eq!(
            e.ac_simplify_proj(&expr, 48, false, super::AcForm::Tagged)
                .unwrap(),
            t(&["<add>", "<mul>", "6", "x3", "<div>", "5", "</mul>", "x8", "</add>"])
        );
        // The TAGGED projection above spells 6/5 structurally (both components inside the
        // vocabulary bound); the EXPLICIT one takes the argmin token, and a 5-carrying
        // denominator wins as a decimal -- §10.10(1), owner-ratified 2026-08-07. Two
        // dialects, one value; mu is spelling-independent.
        assert_eq!(
            e.ac_simplify_proj(&expr, 48, false, super::AcForm::Explicit)
                .unwrap(),
            t(&["+", "*", "1.2", "x3", "x8"])
        );
        // ...and the user-facing infix agrees with the explicit dialect: `1.2` is the
        // argmin token for 6/5, so the reader gets `1.2*x3` rather than `6*x3/5`.
        assert_eq!(
            e.ac_simplify_infix(&expr, 48, false).unwrap(),
            "1.2*x3 + x8"
        );
        // The inverse SECTIONS: subtraction and division spell as `<sub>`/`<div>` sections
        // of their bags; fractions are single tokens.
        assert_eq!(
            e.ac_simplify(&t(&["-", "x0", "x1"]), 48, false).unwrap(),
            t(&["<add>", "x0", "<sub>", "x1", "</add>"])
        );
        assert_eq!(
            e.ac_simplify(&t(&["/", "x0", "x1"]), 48, false).unwrap(),
            t(&["<mul>", "x0", "<div>", "x1", "</mul>"])
        );
        // Divisor-side spelling (2026-08-01, design §8): x0/3 spells through the
        // `<div>` section, not as the `1/3` coefficient token (integer reciprocal is
        // the strictly shorter exact spelling; ties like 22/7 stay coefficient-side).
        assert_eq!(
            e.ac_simplify(&t(&["div3", "x0"]), 48, false).unwrap(),
            t(&["<mul>", "x0", "<div>", "3", "</mul>"])
        );
        // (2*x1)/(x2*x3): one bag, numerator and denominator sections.
        assert_eq!(
            e.ac_simplify(
                &t(&["/", "*", "mult2", "x1", "1", "*", "x2", "x3"]),
                48,
                false
            )
            .unwrap(),
            t(&["<mul>", "2", "x1", "<div>", "x2", "x3", "</mul>"])
        );
        // Standalone unary spellings: `neg` in a function argument, `inv` for a lone
        // reciprocal; both parse back (liberal input).
        assert_eq!(
            e.ac_simplify(&t(&["tan", "neg", "x0"]), 48, false).unwrap(),
            t(&["tan", "neg", "x0"])
        );
        assert_eq!(
            e.ac_simplify(&t(&["inv", "x0"]), 48, false).unwrap(),
            t(&["inv", "x0"])
        );
        // In-bag `neg` on input maps to the section spelling on output.
        // (constant-like terms sort LAST under the stripped order: x2 + 2.3, not 2.3 + x2)
        assert_eq!(
            e.ac_simplify(
                &t(&["<add>", "x2", "2.3", "neg", "x1", "</add>"]),
                48,
                false
            )
            .unwrap(),
            t(&["<add>", "x2", "2.3", "<sub>", "x1", "</add>"])
        );
        assert_eq!(
            e.ac_simplify_infix(&t(&["div3", "neg", "x0"]), 48, false)
                .unwrap(),
            "-x0/3"
        );
        assert_eq!(
            e.ac_simplify_infix(&t(&["/", "x0", "x1"]), 48, false)
                .unwrap(),
            "x0/x1"
        );
        // Tagged input parses back through the shared parser: the retired coefficient
        // spelling stays LIBERAL INPUT for the same state, whose canonical emission is
        // now divisor-side (2026-08-01, design §8) -- and that emission is the fixpoint.
        let old_spelling = t(&["<mul>", "1/3", "x0", "</mul>"]);
        let tagged = t(&["<mul>", "x0", "<div>", "3", "</mul>"]);
        assert_eq!(e.ac_simplify(&old_spelling, 48, false).unwrap(), tagged);
        assert_eq!(e.ac_simplify(&tagged, 48, false).unwrap(), tagged);
    }

    /// The general signed root: pow1_3/pow1_5 desugar into `rootn` (input compat only --
    /// the pow1_k vocabulary is deleted), even/unit/negative indices normalize away, and
    /// EVERY projection emits `rootn` natively.
    #[test]
    fn ac_rootn() {
        let Some(e) = engine() else { return };
        // Legacy odd roots survive as rootn (a table DECLARING the sugar: the in-repo
        // legacy fixture); every projection emits rootn natively.
        let sugar = crate::legacy_sugar_engine();
        assert_eq!(
            sugar.ac_simplify(&t(&["pow1_3", "x0"]), 48, false).unwrap(),
            t(&["rootn", "x0", "3"])
        );
        assert_eq!(
            sugar
                .ac_simplify_proj(&t(&["pow1_3", "x0"]), 48, false, super::AcForm::Explicit)
                .unwrap(),
            t(&["rootn", "x0", "3"])
        );
        assert_eq!(
            e.ac_simplify_proj(
                &t(&["rootn", "x0", "5"]),
                48,
                false,
                super::AcForm::Explicit
            )
            .unwrap(),
            t(&["rootn", "x0", "5"])
        );
        // Normalizations: even index == principal power; index 1 == identity; negative
        // index inverts (rendered as the division of the odd root).
        assert_eq!(
            e.ac_simplify(&t(&["rootn", "x0", "2"]), 48, false).unwrap(),
            t(&["rootn", "x0", "2"])
        );
        assert_eq!(
            e.ac_simplify(&t(&["rootn", "x0", "1"]), 48, false).unwrap(),
            t(&["x0"])
        );
        assert_eq!(
            e.ac_simplify_proj(
                &t(&["rootn", "x0", "-3"]),
                48,
                false,
                super::AcForm::Explicit
            )
            .unwrap(),
            t(&["inv", "rootn", "x0", "3"])
        );
        // Degenerate indices fold in the CONSTRUCTOR: the
        // normalization lives in expr::fun, so parse-time and MID-PASS rebuilds agree.
        // The killer case: an index subtree the numeric fold collapses mid-pass
        // (`cos 0 -> 1`) must normalize in the SAME call -- this was the idempotence
        // break (simplify twice used to differ from simplify once).
        let once = e
            .ac_simplify(&t(&["rootn", "x0", "cos", "0"]), 48, false)
            .unwrap();
        assert_eq!(
            once,
            t(&["x0"]),
            "index fold + unit normalization in ONE call"
        );
        let abs_case = e
            .ac_simplify(&t(&["rootn", "x1", "abs", "(-2)"]), 48, false)
            .unwrap();
        assert_eq!(abs_case, t(&["rootn", "x1", "2"]));
        // Invalid indices are NaN everywhere (IEEE): zero and provably-non-integer.
        assert_eq!(
            e.ac_simplify(&t(&["rootn", "x0", "0"]), 48, false).unwrap(),
            t(&["float(\"nan\")"])
        );
        assert_eq!(
            e.ac_simplify(&t(&["rootn", "x0", "0.5"]), 48, false)
                .unwrap(),
            t(&["float(\"nan\")"])
        );
        // Honest even root on a literal: rootn(8, 2) = sqrt(8), via the pow spelling.
        // STAGE 2 (mu-governed fold): sqrt(8) is IRRATIONAL -- materializing
        // `2.8284271247461903` is a ~105-unit rounding of a 14-unit exact state, so
        // the fold refuses and the pow spelling stands ("simplify is lossless";
        // pre-mu this pin read `2.8284271247461903`).
        assert_eq!(
            e.ac_simplify(&t(&["rootn", "8", "2"]), 48, false).unwrap(),
            t(&["rootn", "8", "2"])
        );
        // Arbitrary odd roots exist now (no legacy spelling; tagged carries them).
        assert_eq!(
            e.ac_simplify(&t(&["rootn", "x0", "7"]), 48, false).unwrap(),
            t(&["rootn", "x0", "7"])
        );
        // Complexity parity with the principal power: rootn(x,3) and pow(x, 1/2) both
        // price as Pow with a cheap rational exponent -- mu (stage 2): 8 + 8 +
        // cost(1/3 or 1/2) = 2, so 18 each. Parity preserved, at the mu scale.
        // (The pow1_3 spelling needs the sugar-declaring fixture table to parse.)
        assert_eq!(sugar.ac_complexity(&t(&["pow1_3", "x0"])), Some(18_000));
        assert_eq!(e.ac_complexity(&t(&["rootn", "x0", "2"])), Some(18_000));
        // The pretty infix is function-call style (x^(1/3) would claim the WRONG function).
        assert_eq!(
            sugar
                .ac_simplify_infix(&t(&["pow1_3", "x0"]), 48, false)
                .unwrap(),
            "rootn(x0, 3)"
        );
    }

    /// H-014 (2026-08-03): sign-orientation canonical uniqueness. `Mul[-1, Add[S]]` and the
    /// sign-flipped Add were two derivation-reachable states sharing one serialization
    /// (the renderer's display redistribution), so one-call fixpoints differed from
    /// re-entry fixpoints. Three orientation owners now cover every sum class -- the
    /// `mul` constructor's sign arm (bare-inf sums + the lone `-1 x Add` shape via
    /// `flipped_orientation_wins`), and `primitive_sum` (whose constlike skip narrowed
    /// to Const-bearing sums). These specimens are the minimized fuzz counterexamples;
    /// each must be idempotent THROUGH its own serialization in one call.
    #[test]
    fn sign_orientation_canonical_uniqueness() {
        let Some(e) = engine() else { return };
        let specimens: [&[&str]; 4] = [
            // inf-bearing sum under an odd-function sign-pull (fuzz row 17001 family)
            &["neg", "-", "tanh", "neg", "acos", "x0", "float(\"inf\")"],
            // factored wrap built by a rewrite chain (row 7333 family)
            &["/", "-", "float(\"-inf\")", "acosh", "x1", "inv", "-2"],
            // atom-bearing constlike sum: primitive_sum's old skip left it unoriented
            &["cos", "inv", "-", "-2", "np.e"],
            // negative coefficient with an inf-bearing Add factor (row 10985 family)
            &["/", "-", "float(\"inf\")", "inv", "asinh", "x0", "-3.253"],
        ];
        for s in specimens {
            let toks: Vec<String> = s.iter().map(|t| t.to_string()).collect();
            let once = e.ac_simplify(&toks, 48, false).unwrap();
            let twice = e.ac_simplify(&once, 48, false).unwrap();
            assert_eq!(
                twice, once,
                "not idempotent through serialization: {toks:?}"
            );
        }
    }

    /// H-030 (2026-08-05): the H-014 orientation congruence completed for the
    /// negation-ABSORBING class (`term_absorbs_negation`: bare-inf products,
    /// inf-Add-factor products, recursive nestings). Three coordinated owners --
    /// `mul()`'s H-030 sign arm (any negative coefficient, any co-factor multiset),
    /// `primitive_sum`'s sign-decision skip, and `add()`'s widened term re-settle --
    /// so every construction route lands on the distributed orientation in ONE call.
    /// Specimens: the 1M fuzz counterexamples (rows 212777 / 914946, the term-level
    /// parking) and 200k row 18943 (the NON-lone wrapper `-1 x Add[S_abs] x den`,
    /// which the first lone-shape-gated fix missed: the display splits divisor-side
    /// co-factors, so re-parse associates the sign with the sum alone), regenerated
    /// verbatim; each pinned fixpoint is engine-verified.
    #[test]
    fn h030_inf_product_sign_orientation_confluence() {
        let Some(e) = engine() else { return };
        let cases: [(&[&str], &[&str]); 3] = [
            (
                &[
                    "-",
                    "neg",
                    "log",
                    "0",
                    "*",
                    "+",
                    "(-1/3)",
                    "tanh",
                    "x4",
                    "+",
                    "/",
                    "float(\"-inf\")",
                    "(-0.5)",
                    "pow",
                    "x4",
                    "float(\"nan\")",
                ],
                &[
                    "<add>",
                    "<mul>",
                    "<add>",
                    "tanh",
                    "x4",
                    "<sub>",
                    "<mul>",
                    "1",
                    "<div>",
                    "3",
                    "</mul>",
                    "</add>",
                    "<add>",
                    "<sub>",
                    "pow",
                    "x4",
                    "float(\"nan\")",
                    "float(\"inf\")",
                    "</add>",
                    "</mul>",
                    "float(\"inf\")",
                    "</add>",
                ],
            ),
            (
                &[
                    "+", "0", "-", "*", "-", "atanh", "/", "x1", "x1", "pow", "pow", "x0", "x2",
                    "x3", "+", "pow", "x0", "cos", "x0", "+", "-", "1", "x3", "/", "x3", "3", "-2",
                ],
                &[
                    "<add>",
                    "<mul>",
                    "<add>",
                    "pow",
                    "pow",
                    "x0",
                    "x2",
                    "x3",
                    "<sub>",
                    "float(\"inf\")",
                    "</add>",
                    "<add>",
                    "<mul>",
                    "2",
                    "x3",
                    "<div>",
                    "3",
                    "</mul>",
                    "<sub>",
                    "pow",
                    "x0",
                    "cos",
                    "x0",
                    "1",
                    "</add>",
                    "</mul>",
                    "2",
                    "</add>",
                ],
            ),
            (
                &[
                    "/",
                    "+",
                    "*",
                    "-",
                    "/",
                    "float(\"-inf\")",
                    "*",
                    "2",
                    "2",
                    "pow",
                    "+",
                    "x4",
                    "1",
                    "/",
                    "5",
                    "float(\"inf\")",
                    "pow",
                    "rootn",
                    "x2",
                    "inv",
                    "x4",
                    "inv",
                    "x0",
                    "/",
                    "cosh",
                    "*",
                    "neg",
                    "-3",
                    "x4",
                    "acos",
                    "/",
                    "/",
                    "-0.03",
                    "2",
                    "+",
                    "x4",
                    "x2",
                    "*",
                    "0.5",
                    "-0.159",
                ],
                // H-059 (2026-08-07): the `2000/159` coefficient now takes the DIVISOR
                // side as `0.0795` (6 chars against 8), per the ratified §8 rule restored
                // for out-of-bound fractions. Same value, spelling only -- the sign
                // orientation this test guards is the `<add>` body, which is unchanged.
                &[
                    "<mul>",
                    "<add>",
                    "<mul>",
                    "float(\"inf\")",
                    "pow",
                    "rootn",
                    "x2",
                    "inv",
                    "x4",
                    "inv",
                    "x0",
                    "</mul>",
                    "<sub>",
                    "<mul>",
                    "cosh",
                    "<mul>",
                    "3",
                    "x4",
                    "</mul>",
                    "<div>",
                    "acos",
                    "<mul>",
                    "-0.015",
                    "<div>",
                    "<add>",
                    "x2",
                    "x4",
                    "</add>",
                    "</mul>",
                    "</mul>",
                    "</add>",
                    "<div>",
                    "0.0795",
                    "</mul>",
                ],
            ),
        ];
        for (input, fixpoint) in cases {
            let toks: Vec<String> = input.iter().map(|s| s.to_string()).collect();
            let want: Vec<String> = fixpoint.iter().map(|s| s.to_string()).collect();
            let once = e.ac_simplify(&toks, 48, false).unwrap();
            assert_eq!(
                once, want,
                "pass-1 must ship the mul()-conformant orientation: {toks:?}"
            );
            let twice = e.ac_simplify(&once, 48, false).unwrap();
            assert_eq!(
                twice, once,
                "not idempotent through serialization: {toks:?}"
            );
            let exp = e
                .ac_simplify_proj(&toks, 48, false, super::AcForm::Explicit)
                .unwrap();
            let exp2 = e
                .ac_simplify_proj(&exp, 48, false, super::AcForm::Explicit)
                .unwrap();
            assert_eq!(exp2, exp, "P7-explicit round-trip unstable: {toks:?}");
        }
    }

    /// H-015 class (b) (2026-08-04): the lossy OUTPUT projection rejoins reciprocal
    /// products where the joined spelling prices strictly smaller AND sound's
    /// distribution licence would refuse to re-split it (`ac::expr::rejoin_projection`).
    /// Pins the three regimes: the registered row-1414 shape joins (mu 80 -> 72, now
    /// IDENTICAL to sound's licence-refusal output); a cancellation partner still
    /// cancels through the distributed working canon; a certified pair stays
    /// distributed in BOTH modes (sound alignment). Idempotence through the projected
    /// serialization is the funnel property the boundary design must keep.
    #[test]
    fn lossy_reciprocal_rejoin_projection() {
        let Some(e) = engine() else { return };
        let row1414 = t(&[
            "inv",
            "*",
            "acos",
            "*",
            "float(\"inf\")",
            "x4",
            "atan",
            "asinh",
            "x0",
        ]);
        let sound = e.ac_simplify(&row1414, 48, false).unwrap();
        let lossy = e.ac_simplify(&row1414, 48, true).unwrap();
        assert_eq!(
            lossy, sound,
            "joined output must equal sound's refusal form"
        );
        assert_eq!(e.ac_complexity(&lossy), Some(72_000));

        let partner = t(&["*", "acos", "x0", "inv", "*", "acos", "x0", "atan", "x1"]);
        assert_eq!(
            e.ac_simplify(&partner, 48, true).unwrap(),
            t(&["inv", "atan", "x1"]),
            "partner cancellation must survive the projection"
        );

        let certified = t(&["inv", "*", "acos", "x0", "atan", "x1"]);
        let cs = e.ac_simplify(&certified, 48, false).unwrap();
        let cl = e.ac_simplify(&certified, 48, true).unwrap();
        assert_eq!(cl, cs, "certified pair: both modes stay distributed");
        assert!(cs.contains(&"<div>".to_string()));

        // A literal-infinity member never joins: `inv(inf)` is a determined zero
        // (the mask-sentinel cancellation partner), and hiding it inside a joined
        // base loses the fold. The spelling keeps `float("inf")` as its own divisor.
        let infbag = t(&[
            "inv",
            "*",
            "float(\"inf\")",
            "*",
            "acos",
            "*",
            "x4",
            "x1",
            "atan",
            "x0",
        ]);
        let il = e.ac_simplify(&infbag, 48, true).unwrap();
        assert!(
            !il.starts_with(&["inv".to_string(), "<mul>".to_string()]),
            "literal-inf member must not be joined: {il:?}"
        );

        for toks in [&row1414, &partner, &certified, &infbag] {
            let once = e.ac_simplify(toks, 48, true).unwrap();
            let twice = e.ac_simplify(&once, 48, true).unwrap();
            assert_eq!(
                twice, once,
                "lossy not idempotent through projection: {toks:?}"
            );
        }
    }

    /// H-027 (2026-08-05, found by the B3+ P1-lossy-idempotence oracle, fuzz rows
    /// 92892 + 115616): the rejoin projection's coefficient completion is admissible
    /// only when the reciprocated coefficient SURVIVES VERBATIM as a member of the
    /// joined base. `mul()` inside the completion runs the H-020 sign fold, so a
    /// negative `cinv` beside a Const-bearing Add member was CONSUMED into the sum --
    /// a strict mu win whose source was the sign fold, not the join, and one the
    /// funnel cannot re-derive (the re-parsed state has no Num left, the completion
    /// cannot re-fire, the plain join ties toward distributed): the joined spelling
    /// shipped exactly once and the second lossy pass moved. Where the coefficient
    /// survives (positive c; negative c beside fold-untouched sums), re-parse
    /// re-extracts it and the completion re-derives, including the sound-aligned
    /// `inv(-1 * ...)` joins.
    #[test]
    fn lossy_completion_rederivability_gate() {
        let Some(e) = engine() else { return };
        // minimal repro: -x0 / (x1 * (C - x2))
        let minimal = t(&["/", "neg", "x0", "*", "x1", "-", "<constant>", "x2"]);
        // fuzz row 115616: ((-x0)/asinh(x4)) / (C - rootn(x2^x4, x2))
        let row115616 = t(&[
            "/",
            "/",
            "neg",
            "x0",
            "asinh",
            "x4",
            "-",
            "<constant>",
            "rootn",
            "pow",
            "x2",
            "x4",
            "x2",
        ]);
        // positive-coefficient completion class stays live and stable
        let positive = t(&["*", "0.5", "inv", "*", "x4", "-", "x3", "np.pi"]);
        for toks in [&minimal, &row115616, &positive] {
            let once = e.ac_simplify(toks, 48, true).unwrap();
            let twice = e.ac_simplify(&once, 48, true).unwrap();
            assert_eq!(twice, once, "lossy not idempotent: {toks:?}");
        }
        // Const-bearing class: the sign stays SPELLED (the H-020 fold inside the
        // completion would consume it -- absorbing it there was the defect).
        let l = e.ac_simplify(&minimal, 48, true).unwrap();
        assert!(
            l.contains(&"-1".to_string()) || l.contains(&"<sub>".to_string()),
            "the sign must survive the projection: {l:?}"
        );
        // Non-Const class (fuzz row 147398): the -1 SURVIVES as a member of the
        // joined base, so the negative completion is re-derivable and fires --
        // lossy reaches sound's `inv(-1 * pi * rootn(3, x1))` refusal form exactly.
        let nonconst = t(&["inv", "*", "neg", "np.pi", "rootn", "3", "x1"]);
        let ns = e.ac_simplify(&nonconst, 48, false).unwrap();
        let nl = e.ac_simplify(&nonconst, 48, true).unwrap();
        assert_eq!(
            nl, ns,
            "re-derivable negative completion must join to sound's form"
        );
        let nl2 = e.ac_simplify(&nl, 48, true).unwrap();
        assert_eq!(nl2, nl, "and stay idempotent: {nonconst:?}");
    }

    /// H-015 class (a) (2026-08-04): the two lossy endpoint mechanisms beyond the plain
    /// rejoin. SENTINEL EXPIRY: the `inv(inf)` keep is a search licence that expires at
    /// the phase-1 fixpoint -- an unpartnered sentinel folds to its determined 0 and
    /// the chain continues descending (it had BLOCKED downstream rules: a kept
    /// `C * inv(inf)` term froze its sum short of the cos-parity shape). The mask
    /// doctrine is untouched: partnered sentinels cancel in phase 1 BEFORE expiry.
    /// COEFFICIENT COMPLETION: the rejoin candidate set includes the bag coefficient
    /// reciprocated into the joined base (`0.5 * (x4*S)^-1` IS `(2*x4*S)^-1`), which
    /// sound's licence-refusal spellings carry inside exactly like that. Measured
    /// together with the narrowed Const-exclusion and the doctrine-exempt P6 oracle:
    /// mode-ordering rows 822 (true baseline) -> 110/200k, all remaining rows
    /// characterized (69 greedy rule-reachability + 41 inf/nan mode-semantics).
    #[test]
    fn lossy_sentinel_expiry_and_coefficient_completion() {
        let Some(e) = engine() else { return };
        // Unpartnered sentinel: expiry folds it and the chain CONTINUES (cos parity
        // fires on the unfrozen sum) -- lossy lands exactly on sound's endpoint.
        let sentinel = t(&[
            "sin",
            "cos",
            "+",
            "*",
            "<constant>",
            "inv",
            "float(\"inf\")",
            "neg",
            "x2",
        ]);
        let s = e.ac_simplify(&sentinel, 48, false).unwrap();
        let l = e.ac_simplify(&sentinel, 48, true).unwrap();
        assert_eq!(l, s, "expired sentinel must reach the sound endpoint");
        assert_eq!(l, t(&["sin", "cos", "x2"]));
        // The mask doctrine survives: a PARTNERED sentinel cancels in phase 1.
        let mask = t(&["*", "/", "float(\"inf\")", "float(\"inf\")", "x0"]);
        assert_eq!(e.ac_simplify(&mask, 48, true).unwrap(), t(&["x0"]));
        // Coefficient completion: the rational coefficient rides inside the joined
        // base, reciprocated -- the outer Mul unwraps.
        let coeff = t(&[
            "*", "0.5", "inv", "*", "x4", "+", "x3", "+", "tan", "x0", "/", "1", "3",
        ]);
        let lc = e.ac_simplify(&coeff, 48, true).unwrap();
        assert_eq!(
            lc,
            t(&[
                "inv", "<mul>", "2", "x4", "<add>", "x3", "tan", "x0", "<mul>", "1", "<div>", "3",
                "</mul>", "</add>", "</mul>",
            ]),
            "coefficient must reciprocate into the joined base"
        );
        for toks in [&sentinel, &mask, &coeff] {
            let once = e.ac_simplify(toks, 48, true).unwrap();
            let twice = e.ac_simplify(&once, 48, true).unwrap();
            assert_eq!(twice, once, "lossy not idempotent: {toks:?}");
        }
    }

    /// H-020 (2026-08-04, owner-ruled): the sign of a negative coefficient folds INTO a
    /// Const-bearing Add factor -- Const-carrying terms absorb it by the forall-exists
    /// refit, every other term negates as ordinary structure -- at EVERY parking (lone
    /// `-1 x Add` wrapper, bag coefficient, collected term), so no projection ever
    /// faces a sign it cannot spell on a Const term (`-<constant>` is not a token; the
    /// infix display silently ate it under pow bases -- fuzz rows 148693/166766,
    /// P7-infix). The sign-parking congruence took three narrowings to get right: the
    /// lone-shape-only fold left the bag-coefficient and divisor-literal routes
    /// diverging (47 P7-infix + 20 P7-explicit at 200k), and `divisor_side` had to stop
    /// parking signed literals in den groups.
    /// H-019 option (a) (2026-08-04, owner-ruled) + H-021: outward-rounded interval
    /// arms with exact-hit carve-outs, and the corroboration-gated evaluate channel.
    /// Pins the three behavior families the hardening settled:
    /// * exact-boundary grounds the OLD arms folded UNSOUNDLY now reduce by the exact
    ///   rules to their true values (`acosh(cos(0))` = 0 shipped as nan; `pow(-1,
    ///   cosh 0)` read a widened continuum exponent and folded nan over tanh(-1));
    /// * the IEEE total folds survive exact hits (`pow(nan, tan 0)` = 1 by x^0 = 1);
    /// * exactness-beyond-instruments grounds stay SYMBOLIC -- no fold, no rule --
    ///   rather than adopting either the f64 rendering or a phantom class
    ///   (`acosh(acos(cos 1))` is 0 but no honest instrument can certify it;
    ///   `atanh(pow(pow(atan 0.5, -1/2), nan))` once shipped inf through the
    ///   `{pinf, nan}` cls() gap).
    #[test]
    fn outward_rounded_interval_honesty() {
        let Some(e) = engine() else { return };
        // Healed unsound folds: the exact rules now win. (D2 finding: the first case
        // shipped as `acosh acos cos 0` -- a stray `acos` changed the value to nan, and
        // instead of the typo being caught the case was EXCLUDED from the loop, so the
        // doc's `acosh(cos(0))` bullet had no covering assertion. Restored to the
        // intended 3-token form; the 4-token variant is pinned separately below as the
        // exact-hit nan chain it actually is.)
        let cases: [(&[&str], &[&str]); 5] = [
            (&["acosh", "cos", "0"], &["0"]),
            (
                &["tanh", "pow", "/", "-3", "3", "cosh", "0"],
                &["tanh", "-1"],
            ),
            (&["pow", "float(\"nan\")", "tan", "0"], &["1"]),
            (
                &[
                    "/", "x16", "acosh", "cos", "*", "-", "x4", "x4", "tan", "x4",
                ],
                &["<mul>", "float(\"inf\")", "x16", "</mul>"],
            ),
            // acosh(acos(cos 0)) = acosh(acos 1) = acosh(0) = nan: the whole chain is
            // exact hits, so the fold carries all the way to the IEEE nan.
            (&["acosh", "acos", "cos", "0"], &["float(\"nan\")"]),
        ];
        for (src, want) in cases {
            let toks: Vec<String> = src.iter().map(|s| s.to_string()).collect();
            let out = e.ac_simplify(&toks, 48, false).unwrap();
            let want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
            assert_eq!(out, want, "healed fold regressed: {src:?}");
        }
        // Exactness-beyond-instruments: symbolic, never a false literal.
        // `acosh atanh tanh 1` was a third specimen here until 2026-08-07. The
        // inverse-pair table gives the engine an EXACT route to it -- `atanh(tanh 1)`
        // collapses to `1` by a judge-CERTIFIED total identity, and `acosh 1 -> 0` is a
        // certified rule -- so the answer 0 is exact algebra, not an instrument-derived
        // literal, and the specimen no longer exercises this wall's subject. Removed
        // rather than re-pinned: the two that remain still have no exact route.
        for src in [
            vec!["acosh", "acos", "cos", "1"],
            vec!["cosh", "tan", "acos", "0"],
        ] {
            let toks: Vec<String> = src.iter().map(|s| s.to_string()).collect();
            let out = e.ac_simplify(&toks, 48, false).unwrap();
            assert!(
                out.len() > 1,
                "exactness-beyond-instruments ground must stay symbolic: {src:?} -> {out:?}"
            );
        }
    }

    /// H-028 + H-029 (2026-08-05, found by the B3+ in-run judge on 1M fuzz): the
    /// b_pow honesty holes the tenth-scale run exposed.
    /// * H-028 (fuzz row 540516): the general-path infinite-exponent fallback read
    ///   `pinf = true` regardless of base magnitude, so `pow(exp(e), -inf/x2)`
    ///   (exponent attaining BOTH infinities) classified PosInf-only, its inverse
    ///   classified FINITE, and the zero-absorption licence folded
    ///   `0 / pow(exp(np.e), -inf/x2)` to 0 -- the true value is nan on a half-line.
    ///   Now the magnitude step runs per component, the class is Mixed, absorption
    ///   refuses.
    /// * H-029 (fuzz rows 809604 + 662833): an outward-rounded ENCLOSURE of a ground
    ///   exceptional exponent (a bracket around 3 from `atan(0) + 3`; around 0 from
    ///   `sin(sin(acos 1))`) was resolved to the enclosure-slack's a.e. class (nan)
    ///   although the ground's true exponent IS the exceptional point. Tight single-
    ///   integer brackets now union the slice value: the class reads Mixed, the
    ///   ground fold REFUSES, and the exact rule path finishes where it can
    ///   (`pow(log 0, atan 0 + 3)` = (-inf)^3 folds -inf; `pow(nan-ground, ~0)`
    ///   stays symbolic -- its true value 1 has no reachable exact spelling).
    #[test]
    fn b_pow_exceptional_point_honesty() {
        let Some(e) = engine() else { return };
        // H-028: the zero-absorption must refuse the mixed-class cofactor.
        let s1 = t(&["/", "0", "pow", "exp", "np.e", "/", "float(\"-inf\")", "x2"]);
        let out = e.ac_simplify(&s1, 48, false).unwrap();
        assert_ne!(
            out,
            t(&["0"]),
            "0-absorption against a {{0, inf}} cofactor is unsound"
        );
        // H-029 integer bracket: the exact path finishes with the true value.
        let s2 = t(&["pow", "log", "0", "+", "atan", "0", "3"]);
        assert_eq!(
            e.ac_simplify(&s2, 48, false).unwrap(),
            t(&["float(\"-inf\")"]),
            "(-inf)^3 must fold through the exact path, not ship the enclosure nan"
        );
        // H-029 zero bracket: refusal, never the a.e. nan.
        let s3 = t(&["pow", "float(\"nan\")", "sin", "sin", "acos", "1"]);
        let out = e.ac_simplify(&s3, 48, false).unwrap();
        assert!(
            out.len() > 1,
            "nan-base pow over a ~0 enclosure must stay symbolic: {out:?}"
        );
    }

    #[test]
    fn const_sum_sign_absorption() {
        let Some(e) = engine() else { return };
        // Fuzz row 166766's shape: the wrapped Const sum under a fractional pow. The
        // canonical state is the unsigned sum (both constants refit).
        let powbase = t(&[
            "pow",
            "neg",
            "+",
            "*",
            "<constant>",
            "x4",
            "*",
            "<constant>",
            "acos",
            "<constant>",
            "/",
            "1",
            "3",
        ]);
        let out = e.ac_simplify(&powbase, 48, false).unwrap();
        assert!(
            !out.contains(&"neg".to_string()),
            "sign must fold into the Const sum: {out:?}"
        );
        // Mixed sum: the Const term refits, the non-Const term carries its sign as
        // ordinary structure -- no wrapper survives.
        let mixed = t(&["neg", "+", "*", "<constant>", "x4", "tanh", "x1"]);
        let m = e.ac_simplify(&mixed, 48, false).unwrap();
        assert!(
            !m.starts_with(&["neg".to_string()]),
            "wrapper must fold: {m:?}"
        );
        // Every parking is idempotent through its own serialization (bag coefficient
        // via the division spelling included).
        let bag = t(&[
            "/",
            "neg",
            "+",
            "*",
            "<constant>",
            "x0",
            "atan",
            "x1",
            "3.89",
        ]);
        for toks in [&powbase, &mixed, &bag] {
            let once = e.ac_simplify(toks, 48, false).unwrap();
            let twice = e.ac_simplify(&once, 48, false).unwrap();
            assert_eq!(
                twice, once,
                "not idempotent through serialization: {toks:?}"
            );
        }
    }

    /// LOSSY-mode corpus gates. LOSSY is the training-corpus canonicalisation mode: every
    /// certificate is skipped, so it rewrites MORE, and its soundness contract is
    /// self-referential -- training data is generated FROM the simplified form (target ==
    /// data), so parity with any other engine's lossy output is a NON-goal. What must hold
    /// are the same structural invariants as SOUND mode: determinism, idempotence, and
    /// permutation invariance -- a training corpus canonicalizer that maps equal functions to
    /// different targets reintroduces exactly the multi-modality it exists to remove.
    #[test]
    fn ac_lossy_corpus_gates() {
        let Some(e) = engine() else { return };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/corpus/raw_skeletons_nv.json"
        );
        let corpus: Vec<Vec<String>> =
            serde_json::from_str(&std::fs::read_to_string(path).expect("corpus present"))
                .expect("corpus parses");
        let mut comp_sound = 0u64;
        let mut comp_lossy = 0u64;
        let mut n_lossy_wins = 0usize;
        for expr in corpus.iter() {
            let sound = e.ac_simplify(expr, 48, false).unwrap();
            let lossy = e.ac_simplify(expr, 48, true).unwrap();
            // Idempotence and permutation invariance under LOSSY.
            assert_eq!(
                e.ac_simplify(&lossy, 48, true).unwrap(),
                lossy,
                "lossy not idempotent on {expr:?}"
            );
            let swapped = swap_commutative(expr, &e);
            assert_eq!(
                e.ac_simplify(&swapped, 48, true).unwrap(),
                lossy,
                "lossy permutation variance on {expr:?}"
            );
            let cs = e.ac_complexity(&sound).expect("sound parses");
            let cl = e.ac_complexity(&lossy).expect("lossy parses");
            comp_sound += cs;
            comp_lossy += cl;
            if cl < cs {
                n_lossy_wins += 1;
            }
        }
        // Reported, not asserted: lossy rewrites more, so it should rarely be worse -- but
        // rule-path divergence makes a hard <= assert too strong.
        eprintln!(
            "AC lossy corpus: complexity sound={comp_sound} lossy={comp_lossy}, lossy strictly smaller on {n_lossy_wins}/{} rows",
            corpus.len()
        );
    }

    /// Corpus gates, amortized over one engine build: output validity, idempotence,
    /// commutative-permutation invariance, complexity accounting, and no `<constant>`
    /// minting. (Numeric EQUIVALENCE gating is cross-installation territory by the clean-
    /// release doctrine: the old kernel is gone from this crate, and the remine/ harness
    /// adjudicates equivalence against a pinned previous release.)
    #[test]
    fn ac_corpus_gates() {
        let Some(e) = engine() else { return };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/corpus/raw_skeletons_nv.json"
        );
        let corpus: Vec<Vec<String>> =
            serde_json::from_str(&std::fs::read_to_string(path).expect("corpus present"))
                .expect("corpus parses");
        assert!(!corpus.is_empty());

        let mut n_shorter = 0usize;
        let mut tok_in = 0usize;
        let mut tok_out = 0usize;
        let mut comp_ac = 0u64;
        for expr in corpus.iter() {
            // The EXPLICIT projection is gated for validity; the TAGGED default is gated
            // separately: it must round-trip through the shared parser (idempotence feeds
            // it back in).
            let out = e
                .ac_simplify_proj(expr, 48, false, super::AcForm::Explicit)
                .unwrap();
            assert!(e.is_valid(&out), "invalid output {out:?} for {expr:?}");
            let tagged = e.ac_simplify(expr, 48, false).unwrap();
            // Idempotence in the native form: feeding the tagged answer back reproduces it.
            let again = e.ac_simplify(&tagged, 48, false).unwrap();
            assert_eq!(again, tagged, "not idempotent on {expr:?}");
            // The tagged form is strict: no binary sugar operators (bags + sections own
            // that structure); `neg`/`inv` are the sanctioned STANDALONE unary spellings.
            for t in &tagged {
                assert!(
                    !matches!(t.as_str(), "-" | "/" | "+" | "*"),
                    "sugar token {t} in tagged output {tagged:?}"
                );
            }
            // Permutation invariance: swapping the operands of every binary commutative node
            // must not change the answer (the ORDER axis, corpus-wide).
            let swapped = swap_commutative(expr, &e);
            let out_swapped = e
                .ac_simplify_proj(&swapped, 48, false, super::AcForm::Explicit)
                .unwrap();
            assert_eq!(out_swapped, out, "permutation variance on {expr:?}");

            tok_in += expr.len();
            tok_out += out.len();
            if out.len() < expr.len() {
                n_shorter += 1;
            }

            if expr.iter().any(|t| t == "<constant>") {
                continue;
            }
            // A constant-free input must never mint a <constant>: <constant>-collapse and the
            // Const absorptions all REQUIRE a Const source. Surface the row if it happens.
            assert!(
                !out.iter().any(|t| t == "<constant>"),
                "minted <constant>: {expr:?} -> {out:?}"
            );
            // Complexity accounting on the canonical form.
            {
                let ctx2 = super::super::memo_ctx_for_tests(&e);
                let toks_a = super::super::intern_for_tests(&e, &out, &ctx2);
                let view2 = super::super::view_for_tests(&e, &ctx2);
                let bare2 = crate::ac::expr::Cx::bare(&view2);
                if let Some(p) = crate::ac::convert::from_prefix(&toks_a, &bare2) {
                    comp_ac +=
                        crate::ac::expr::complexity(&crate::ac::expr::canon(p, &bare2), &view2);
                }
            }
        }
        eprintln!(
            "AC corpus: {} exprs, {} shorter-than-input, tokens input={} ac={}, COMPLEXITY (const-free rows) ac={}",
            corpus.len(),
            n_shorter,
            tok_in,
            tok_out,
            comp_ac
        );
    }

    /// Swap the operands of every binary commutative (`+`/`*`) node -- a deterministic
    /// worst-case permutation of the input spelling.
    fn swap_commutative(expr: &[String], e: &Engine) -> Vec<String> {
        fn parse(toks: &[String], i: &mut usize, e: &Engine) -> Vec<String> {
            let t = toks[*i].clone();
            *i += 1;
            let arity = e.operators_ref().arity_of(&t).unwrap_or(0) as usize;
            let mut parts: Vec<Vec<String>> = Vec::with_capacity(arity);
            for _ in 0..arity {
                parts.push(parse(toks, i, e));
            }
            let commutative = t == "+" || t == "*";
            let mut out = vec![t];
            if commutative && parts.len() == 2 {
                parts.swap(0, 1);
            }
            for p in parts {
                out.extend(p);
            }
            out
        }
        let mut i = 0;
        let out = parse(expr, &mut i, e);
        assert_eq!(i, expr.len());
        out
    }
}
#[cfg(test)]
mod probe_stab {
    use crate::ac::convert::{from_prefix, to_prefix};
    use crate::ac::expr::{canon, Cx};

    #[test]
    #[ignore]
    fn find_unstable() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/corpus/raw_skeletons_nv.json"
        );
        let corpus: Vec<Vec<String>> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for (i, expr) in corpus.iter().enumerate() {
            let ctx = super::super::memo_ctx_for_tests(&e);
            let toks = super::super::intern_for_tests(&e, expr, &ctx);
            let view = super::super::view_for_tests(&e, &ctx);
            let bare = Cx::bare(&view);
            let Some(ex) = from_prefix(&toks, &bare) else {
                continue;
            };
            let c1 = canon(ex, &bare);
            let t1 = to_prefix(&c1, &bare);
            let Some(p) = from_prefix(&t1, &bare) else {
                eprintln!("row {i}: REPARSE FAILED");
                continue;
            };
            let c2 = canon(p, &bare);
            let t2 = to_prefix(&c2, &bare);
            if t1 != t2 {
                let s1: Vec<String> = t1.iter().map(|&t| view.resolve_owned(t)).collect();
                let s2: Vec<String> = t2.iter().map(|&t| view.resolve_owned(t)).collect();
                eprintln!(
                    "row {i} UNSTABLE:\n  t1: {}\n  t2: {}",
                    s1.join(" "),
                    s2.join(" ")
                );
                return;
            }
        }
        eprintln!("all stable (bare ctx)");
    }
}
