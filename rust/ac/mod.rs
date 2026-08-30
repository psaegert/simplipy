//! The AC core: n-ary `Add`/`Mul` (Flat + Orderless bags), binary `Pow`, unary function
//! applications, and unrestricted exact literals.
//!
//! ## Why this exists
//!
//! The shipped engine represents expressions as binary trees over a finite operator vocabulary in
//! which 16 of 38 operators are baked-in coefficients (`mult2..5`, `div2..5`, `pow2..5`,
//! `pow1_2..5`) -- a workaround for the missing numeric literal. That representation caused a
//! measured commutative-order invariance defect (4.58% of eligible expressions simplify
//! differently under operand permutation) with TWO axes: ORDER (which operand spelling a rule
//! was mined in) and ADJACENCY (`+ x3 (+ x8 (div5 x3))` never fires even though both operand
//! spellings of the rule exist, because the matching is strictly local to one binary node).
//!
//! The AC core removes both axes at the representation level:
//! * `Add`/`Mul` are FLAT SORTED BAGS -- there is no operand order and no bracketing for a rule
//!   to be sensitive to.
//! * Rule matching selects a SUB-MULTISET of a bag and preserves the remainder (see
//!   `ac::matcher`), so locality is gone too: this is the ratified "semantic widening".
//! * Coefficients are EXACT RATIONALS composed explicitly (`Mul[7, x]`, `Pow(x, 3)`), never
//!   absorbed into operator names -- coefficient arithmetic becomes exact computation instead of
//!   82 numerically-mined, sampling-certified rules.
//! * `-`, `/`, `neg`, `inv` and the whole hyper-operator family desugar at the boundary
//!   (`ac::convert`) and do not exist in the core. `pow1_3`/`pow1_5` do NOT desugar: the engine
//!   realizes them as REAL odd roots, while `Pow(x, 1/3)` denotes the NaN-on-negatives rational
//!   power -- they differ on the full-measure set `x < 0`, so they stay opaque unary functions.
//!
//! The internal canonical form is UNIQUE per derivation-reachable class: bags order by
//! STRIPPED keys (Add terms by coefficient-stripped structural key, constant-like terms last;
//! Mul factors by exponent-stripped base) and every canonical `Add` is PRIMITIVE -- positive
//! lead coefficient, no unanimous coefficient magnitude, extracted content wrapping the sum
//! (`-a - b -> -1 * (a + b)`, `a/3 + b/3 -> 1/3 * (a + b)`). Factored sums entering an Add
//! context unfactor, so extraction decisions are made once per complete bag and association
//! order cannot leak. Serializations are bijective-on-classes projections, asserted stable
//! (parse -> canon -> serialize == identity) on every chain state in debug builds.
//!
//! Like-term/like-factor collection inside the canonical constructors SUBSUMES the old engine's
//! cancellation unit -- and makes it confluent: bags eliminate the adjacency choices that made
//! cancellation a branching search dimension. The soundness discipline is carried over 1:1 and
//! generalized: the additive sign-cancel gate is the `!`-certificate (`finite_ae`), the
//! multiplicative exponent-cancel gate is the `$`-certificate (`finite_nonzero_ae`), and
//! `<constant>` independence (every occurrence is a fresh constant) blocks all collection of
//! `<constant>`-containing keys, with the exact `forall c_s exists c_t` absorptions the contract
//! licenses handled explicitly.
//!
//! ## Supported ruleset pairing
//!
//! The AC core is supported with SORT-PROMOTED rulesets (the grid-cell artifact family):
//! their cancellation rules carry the `!`/`$` certificate sorts, which the matcher gates per
//! fire, so pole identities resolve exactly. The legacy pre-promotion `dev_7-3` asset is
//! certificate-free (`_`-sorted throughout); on the unified AC representation its cancel
//! rules reach pole spellings the binary engine's tree bucketing happened to shield, so that
//! asset stays paired with the binary engine and older simplipy versions.
//!
//! Serialization is pluggable and stays INSIDE the old token language (`ac::convert::to_prefix`):
//! binary nesting, explicit literal coefficients, `-`//`/`/`neg`/`inv` sugar, no hyper-operators.
//! Every output remains parseable, evaluable and certifiable by the shipped engine, which keeps
//! the old engine available as a differential-testing oracle.

pub mod convert;
pub mod expr;
pub mod matcher;
pub mod rat;
pub mod rules;
pub mod search;
