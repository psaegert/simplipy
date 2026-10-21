# The Simplification Engine (Formal)

This page is the mathematical specification of the AC simplification core
(`SimpliPyEngine.simplify`): how mined rule sets are *applied* — how they are
*discovered* is described in [Creating rulesets](rules.md). It covers: the term algebra, the
normalization function, the rewrite relation, the reduction ordering, and exactly which
properties hold with which strength.

**Faithfulness convention.** Every claim carries one of five tags, and the tag is part of
the claim:

| Tag | Meaning |
|---|---|
| **[THEOREM]** | Proven here from stated premises; the proof is in this page. |
| **[BY CONSTRUCTION]** | Holds because the code checks or computes it on every run (the check is cited); audited at the source level, not mechanized. |
| **[ENFORCED]** | An explicit runtime bound or gate makes violation impossible, without claiming the bound is ever reached. |
| **[EMPIRICAL]** | Measured on named corpora/tests; re-measured by CI or the release gates; not claimed beyond the measured data. |
| **[OPEN]** | Believed plausible, **not proven**; the operational cover for the gap is named. |

Semantic soundness (that a rewrite preserves the denoted function, with which
almost-everywhere licences) is out of scope here; it is governed by the operator
documentation and the per-fold licences cited in `rust/ac/expr.rs`. This page covers
*structure*: well-definedness, termination, canonicity.

---

## 1. Architecture: a two-layer system

The published system (arXiv:2602.08885, §4.1) was formalized as "a length-reducing
rule-based term rewriting system (TRS) $R$ *paired with* a multiplicity-based cancellation
procedure" — already a two-layer object, with the proof covering $R$ only. The AC core
keeps the two-layer shape and redraws the layers:

1. **The normalization function** $\mathrm{nf}$ (§3) — the canonical constructors. It
   absorbs the old cancellation procedure *and* the arithmetic rule family: like-term and
   like-factor collection, exact rational coefficient/exponent arithmetic, and the licensed
   structural folds. It is a *function*, not a rewrite relation: deterministic by
   construction, with no proof obligation about rule application order.
2. **The rule layer** $R$ (§4) — the mined rules, applied *modulo* the bag structure:
   sub-multiset matching on flattened associative-commutative arguments with the unmatched
   remainder preserved. This is rewriting modulo AC in the sense of Peterson–Stickel
   extension rules; the composite system (rules applied between renormalizations) is a
   *normalized rewriting* system in the sense of Marché (1996).

Both layers fire under one Knuth–Bendix-style orientation: the reduction ordering of §5.

## 2. Terms

The term set $T$ is the canonical expression algebra (`rust/ac/expr.rs`, `Ex`):

$$ t ::= q \mid \pi \mid e \mid +\infty \mid -\infty \mid \mathrm{NaN} \mid \diamond
   \mid x \mid \mathrm{Add}\,M \mid \mathrm{Mul}\,M \mid \mathrm{Pow}(t, t)
   \mid f(t_1, \dots, t_n) $$

where $q$ ranges over exact rationals (`Rat`: $p/q$ with $|p|$ and $q$ of at most 1,100 bits
each, `CAP_BITS`; normalized $q > 0$, $\gcd(|p|, q) = 1$; held as `i128` whenever it fits, never
`i128::MIN`; in the `f64` and `permissive` modes -- "f64 mode" below, for the number domain --
only zero and the numbers whose numerator and denominator are both at most $2^{1022}$, so that
the number and its reciprocal are normal float64 values and every printed spelling reads back as
the same number, and in real mode every number within the cap; any other literal stays a token
leaf as written: `1e400` in every mode, and in f64 mode also the subnormal `5e-324` and
`5e307`), $\diamond$ is the abstract fitted constant
(`Const`), $x$ over vocabulary leaves, $f$ over operator tokens, and $M$ over multisets of
terms with $|M| \geq 2$.

**Canonical-form invariants** [BY CONSTRUCTION — established by the constructors
`add`/`mul`/`pow`/`fun`, spot-checked by the debug assertions `no_nested_bags` and the
serialization-stability check `stable()` in `ac_simplify_ex`]:

- **I1 (flatness):** no $\mathrm{Add}$ directly inside an $\mathrm{Add}$, no $\mathrm{Mul}$
  directly inside a $\mathrm{Mul}$.
- **I2 (bag order):** $\mathrm{Add}$ members are sorted by their coefficient-stripped
  structural key (constant-like terms last); $\mathrm{Mul}$ members by their
  exponent-stripped base (rational coefficient first). Stripping makes the order invariant
  under sign and exponent edits, which is what makes the sign orientation of sums
  well-defined (`rust/ac/expr.rs`, "STRIPPED comparators").
- **I3 (coefficient normal form):** a $\mathrm{Mul}$ carries one rational factor and an
  $\mathrm{Add}$ one rational term, unless a fold is refused. The literals of a bag, the
  coefficients of one like-term bucket and the exponents of one like-base pool fold to a
  CANONICAL PARTITION (`Rat::partition_with`): sorted ascending, each member folds into the
  lowest kept member it folds with, until no two members fold. A fold refuses beyond the cap,
  and in f64 mode where the result is not admissible (numerator and denominator at most
  $2^{1022}$, a set closed under negation and reciprocals); an exponent merge also needs its
  branch-cut licence. No two members of a partition fold, so it is the partition of each of its
  subsets and re-reads to itself in any printed order. *Known residuals* [EMPIRICAL]:
  the partition makes each group re-read to itself, not every form around it. Measured against
  the same code with 128-bit literals, two boundary fuzzes with literals at the 128-bit,
  float64-range and cap boundaries gave fewer canonical-form idempotence failures (at effort 0)
  in every mode: 3 f64, 0 real, 1 permissive over 60,000 calls per mode (128 bits: 6, 6, 3), and
  10, 7, 5 over another 52,000 (128 bits: 16, 16, 19; exact McNemar p = 0.26, 0.064, 0.001, so
  only permissive's gap is significant). The classes: sums whose sign orientation stays open
  because they hold literals that stay opaque leaves (beyond the cap, or in f64 mode outside the
  admissible set: `1e400`, subnormals, `1.7976931348623157e308`, and the decimal
  `2.2250738585072014e-308`, whose exact denominator is $10^{324}$) or stand beside a
  $\pm\infty$ factor; a like term whose coefficient is itself a refused two-member partition,
  whose coefficient and key can split differently on re-read (also with 128-bit literals); and
  odd-function sign extraction. A class independent of literal size is closed: a product with
  more than six sign-trade sites kept the placement it entered with, two sums that are each
  other's negation stayed apart, and a join whose coefficient became 1 kept its key's signs, so
  a state and the parse of its own print could differ (0.14.7: 28 of srbf's 125,127 model
  predictions change on a second call in f64 with the search off). Each is now decided by the
  orbit (§3), and no f64 or real answer changes on a second call on the 129,490 simplify inputs
  of flash-ansr's training-data generator, srbf's ground truths or its predictions, with the
  search off or at the default effort. The sign placement keeps the entry
  spelling where an orientation would collide with another factor's base or a negation overflows
  (§3). At a refusal every grouping of a product partitions the same multiset of coefficients,
  because the parse builds the product once from all of its members:
  `(1e200*1e100)*(1e100*x1)` and `1e200*(1e100*(1e100*x1))` reach one canonical form (pinned
  in tests). Uncapped (the default in every mode; L6), the exploration phase can reach states of
  the literal classes above that the chain alone never visits. A capped search, or one a work
  budget stops, can stop between two improvements, which a second call continues.
- **I4 (fold normal form):** the licensed structural folds of §3 have been applied; e.g. no
  $\mathrm{Pow}(t, 1)$, no $\mathrm{rootn}(t, k)$ with $k \leq 0$ or $|k| = 1$, no
  all-literal composite that the constructors fold — and the sign placement between a
  product's coefficient slot and an odd function's literal argument is decided by ONE
  shared pricing orbit on every construction route (the sign-trade owner): $\mu$ picks the cheaper spelling, an exact tie goes
  to the sign key (§3), which leaves a constant's sign on its literal. So $-1 \cdot \sin(2)$ files as $\sin(-2)$ (the product
  node itself dissolves), $-5 \cdot \sin(2)$ as $5 \cdot \sin(-2)$, and
  $-1 \cdot x_0 \cdot \sin(2)$ as $x_0 \cdot \sin(-2)$: a sign costs one bit wherever it sits
  (§5), so the placements tie and the sign key decides.
  **An EVEN index is canonical**: `rootn(t, 2)` and `rootn(t, 4)` are normal forms, not reducible ones --
  `rootn` is the canonical spelling of a principal even root and $\mu$ ties it with
  $\mathrm{Pow}(t, 1/k)$, so neither orientation is forced.

**Route invariance** [EMPIRICAL — deliberately *not* claimed as a theorem]:
$\mathrm{nf}$ is deterministic *per route*, but a state built incrementally through
the constructors and the same value's parse-route canon are kept equal by shared-owner
design plus a live instrument, not by proof. The instrument: every pass output must
round-trip (serialize → parse → canon) onto the *same state*, debug-asserted on every
`simplify` call and exercised by the full suite and the corpus gates. The one measured
divergence class (the odd-function literal-sign pair: `mul()`-built vs collector-built
spellings of one value) was removed by the shared sign-trade owner; zero
specimens remain. The instrument itself printed each state with the mode's certificate
context, a spelling no caller receives (every caller prints with the bare context), which
raised false alarms and hid three real classes (the closed I3 class above); it now prints as
the callers do, and in a debug build it holds on every state of srbf's ground truths and
predictions and of the 129,490 simplify inputs of flash-ansr's training-data generator, in every
mode. A second measured class sat not in `canon()` but in the *pricing instrument*:
`complexity()` once parsed without the folds a mode's chain applies at parse, so on 10 of 65,536
benchmark rows it priced a different state than the chain descended from and
$\mu(\mathrm{simplify}(e)) \le \mu(e)$ failed *as measured* while holding per route. Closed by
construction: the instruments price each mode through that mode's own parse route. No proof
that `canon()` is route-invariant is claimed; the assert's failure diagnostics tell a
*value-changing* divergence at entry canon from a μ-equal one.

$T_{\mathrm{can}} \subset T$ denotes the terms satisfying I1–I4. Serialization
(`to_prefix`) and parsing (`from_prefix`) connect $T_{\mathrm{can}}$ to token sequences;
round-trip stability is debug-asserted per pass and covered by the corpus gates
[EMPIRICAL].

## 3. The normalization function

$\mathrm{nf} : T \to T_{\mathrm{can}}$ is defined by structural recursion through the four
canonical constructors (`canon` composes them). Each constructor performs its flattening,
sorting, exact arithmetic, and *licensed* folds — a fold whose soundness needs a
certificate (finiteness, non-vanishing, null infinite set, non-negativity, branch-cut
conditions) fires only when the certificate holds; refusing a fold is always available
and always sound. Each certificate is exactly as sharp as the fold's true disagreement
set: e.g. splitting a reciprocal factor needs only `{A = ±inf}` null — a NaN of `A`
agrees on both spellings, so a fat NaN domain does not refuse it.

**The zero-set certificate reads a zero set on the expression's domain.** A zero is a value,
so only where an expression is defined can it have one; a NaN region holds no zero. The
certificate of a null zero set that the core reads (reading A, the distribution licences,
orientation) applies the identity theorem there. Every operator is real-analytic on an open
domain: the entire ones on R; `log`, an even root and a fractional or symbolic power where
their argument is positive; `asin`, `acos`, `atanh` on (-1, 1); `acosh` above 1; an inverse, a
quotient, a negative power, `abs` and an odd root where their argument is nonzero. Off that
domain an expression is NaN, or its argument sits on the domain's edge. Where each restricted
argument is affine in the variables, or its domain holds over the whole space, the edges are
hyperplanes, a null set, and the domain is a union of convex pieces, one per sign of the
arguments that must be nonzero. The zero set is null when interval evaluation finds in every
piece a cell on which every restricted argument lies in its domain and the expression is
nonzero: an analytic function that vanished on a set of positive measure in a connected piece
would vanish on all of it. So `log(x2) - x2`, `sqrt(x2) - x2`, `log(x3) - log(x2)`,
`asin(x1) - x1` and `sin(x3)/x1 - sqrt(x2)` are null, while `x1 - |x1|` (zero on all of
`x1 > 0`) and `asin(sin(x1)) - x1` are not. The certificate fails closed on a piece where it
finds no such cell, on a restricted argument that is not affine, on `tan` (whose poles cut the
line into infinitely many pieces) and on more than three arguments that must be nonzero. The
plain certificate behind the `$`-cancel keeps its strength.

**In `f64` the zero-set certificate reads each function on its float64 range.** Two functions of
the vocabulary are exactly constant on a whole tail in float64: `tanh(u)` is ±1 for every
|u| ≥ 18.990341103219276 (the deployed evaluator; the engine's own saturates at 19.06) and
`atan(u)` is ±fl(π/2) for every |u| ≥ 5805358775541310. This was measured on every operator,
with both evaluators, out to the largest float. Every other operator keeps changing on its tails,
or overflows to an infinity or underflows to 0 (`exp`, `sinh`, `cosh`, `pow`), which the overflow
reading covers, or is NaN off its domain. On a tail such a function is not analytic, and a sum
that cancels its limit vanishes on a region: `x2*tanh(x1) - x2` is `+0` wherever x1 ≥ 18.99, so
its inverse is `+inf` there. In `f64` the certificate therefore splits the space. Off the tails the
expression is the real one, so its zeros there are among the real expression's, which the real
certificate shows null. On a tail it is exactly the expression with that function replaced by
its limit, and the tail adds no zero set when that expression's zero set is null in float64 on
the whole space (a stronger claim, which fails closed). A function whose argument is ground, or
never leaves the middle (`tanh(sin(x1))`), has no tail to read; more than three such functions
fail closed. So `x2*tanh(x1) - x1`, `x2*tanh(x1*x3) - x1`, `sin(tanh(x1^8))` and `atan(x1) - x1`
stay null, while `x2*tanh(x1) - x2`, `tanh(x1 - x3) + 1`, `atan(x1) - fl(π/2)`,
`tanh(tanh(x1)) - tanh(1)` and `1/(1/(tanh(x1) - 1))` are not. `real` and `permissive` read the
real functions, which never reach their limits. Rounding plateaus are not covered by this reading:
`cos` is exactly 1 within 1.05e-8 of every multiple of 2π, `cosh` and `exp` are exactly 1 near 0,
an addend falls off a much larger one (`(x1 + 1) - x1`), and an argument that tends to a plateau
stretches it over a tail (`cos(1/x1)` is exactly 1 for every |x1| ≥ 1e8). Like all rounding, the
certificate reads these as the real functions.

**Lemma L1 (nf is well-defined and terminating)** [BY CONSTRUCTION — audited per call
site]. Every recursive constructor call is on arguments of strictly smaller size, with
exactly two audited exceptions carrying their own strictly decreasing measures:

- *power composition* $(t^{r})^{s} \to t^{r \cdot s}$ recurses on the strict subterm $t$;
- *rootn index normalization* $\mathrm{rootn}(t, -n) \to \mathrm{Pow}(\mathrm{rootn}(t, n), -1)$
  makes one self-call that lands in the non-negative-index branch and cannot re-enter the
  negative branch (a non-negatable index refuses the fold and stays symbolic).

**Determinism** [BY CONSTRUCTION]: $\mathrm{nf}$ makes no random or order-of-arrival
choices; bag merges fold in sorted order (I3).

**What is *not* claimed:** confluence of an underlying rewrite relation. $\mathrm{nf}$ is
canonical *by definition* — the representative of an expression is whatever $\mathrm{nf}$
returns — and the meaningful uniqueness property is invariance: syntactically different
spellings of the same bag reach the same representative. That is gated, not proven:
idempotence and commutative-permutation invariance are pytest properties and corpus gates
(currently 0 violations on the 400-expression canonical corpus and the permutation
property tests; see §7) [EMPIRICAL].

**Convention (infinity-bearing sums are stored flat).** A sum carrying one unabsorbed
infinity is returned as its flat bag without the primitive-sum orientation/content pass.
This loses nothing, by a two-part argument [BY CONSTRUCTION + EMPIRICAL]: *content*
extraction requires a unanimous coefficient magnitude, and an infinity pins the magnitude
multiset at 1 (it absorbs scalar magnitudes exactly), so a unanimous content $\neq 1$
cannot exist; and the *sign* family is closed without the pass, because $\pm 1$
coefficients distribute totally into the bag (the scaling flips the infinity's sign) and
per-term signs are carried by the serialization sections, so every negation spelling of
the same function lands in the same flat bag. Probed with a 2,000-case orientation fuzz
(every sum against its negated-flip spelling): zero divergence. Two conventions coexist knowingly: a finite
mixed-orientation sum files in whichever of its two orientations prices LOWER under
$\mu$ — the $-1$ wrapper (a negation, one bit) appears exactly when the wrapped spelling is the cheaper one.
By design, mirrored subtractions score equal, and a strictly larger expression
never scores the same as a smaller one through a hidden wrapper surcharge.
Infinity-bearing sums never wrap; each family has a single representative, so no
canonicity is at stake.

**Convention (sign placement, three tiers).**
Signs in a product trade legally across ODD carriers — a bare mixed-sign `Add` factor,
$\mathrm{Pow}(S, n)$ for odd integer $n \ge 3$, $\mathrm{rootn}(S, m)$ for odd
$m \ge 3$, and the eight odd functions — because $f(-S) = -f(S)$ is total on those
carriers — and negative odd powers of them, as members of a class that trades its sign as a
whole (the pole convention below). The
sign-placement owner (`ac::expr::sign_place`, shared by `mul()`'s final assembly and
`term_join`'s negative joins so the priced spelling is always the built spelling)
keeps the $\mu$-argmin of every reachable placement, for any number of sites, without
materializing the orbit (`sign_place_groups`): each class flips as a whole and prices as the sum
of its members, a placement's price is the sum of its classes' prices, coupled only by the parity
of the flips (the sign the carrier takes), so the cheapest placement of either sign is each
class's cheaper orientation, with the one class whose switch costs least switched where the
parity differs, and the cheaper of the two is built and priced exactly. A zero check (the pole
convention below) restricts the orientations of the classes it counts, which are decided
together where one check counts several. An orientation in which a flipped factor lands on the
base another factor is collected under (an even root counts as a power of its radicand) is no
candidate. Exact ties go to the sign key (below) and then to the structural order, both
functions of the orbit, never of the entry spelling. The debug build checks the price against
the enumeration of every placement on orbits of up to six classes.

**Convention (opposite classes, one table per product).** A sum $S$ and its negation are one
base up to a sign, and so are $f(S)$ and $f(-S)$ for an odd function $f$ and
$\mathrm{rootn}(S, m)$ and $\mathrm{rootn}(-S, m)$ for odd $m$: each pair $\{B, -B\}$ is one
class (an even function needs none: `fun` files `cos(S)` and `cos(-S)` in one orientation). The
parse builds a product once from all of its members, whatever brackets, `/`, `neg` or the tagged
`<mul>`/`<div>` spell it, so no bracket is decided on its own. A divisor that is itself a product
gives the enclosing product its members' inverses wherever `pow` would distribute the power (one
licence for `pow`, the parse and `canon`); where the certificate refuses, it stays one inverted
product, decided as a whole. `mul()` reads each class as a table of exponent totals per
orientation: the positive integer mass $p$, the negative integer mass $n$ (a fractional exponent
gives its integer part, toward zero, to $p$ or $n$: $S^{5/2} = S^2 S^{1/2}$) and the fixed
factors (fractional remainders, symbolic powers, totals beyond $2^{100}$). All of $p$ may move to
the other orientation, an odd amount with a sign to the bag's host (the coefficient, or a bare
infinity or a bare Const that carries it); the even part of $n$ moves freely, and an odd $n$
moves with its class as a whole under the pole convention below; fractional and symbolic parts
never move.
Every placement, including how a side's mass splits between its integer and its fractional
powers, is priced as the whole product, and the cheapest wins, an exact tie going to the sign
key and then the structurally smaller product, as in `sign_place`. The
classes are decided together while their placements combine to at most 4,096 products (priced
in the order of a lower bound on their prices), else one at a time until none moves. The
decision is a function of the multiset of members, so the product does not depend on how its
factors were bracketed or ordered: `(x1 - x2)*(x2 - x1)` is `-(x1 - x2)^2`,
`(x1 - x2)^2/(x2 - x1)/(x1 - x2)` is `-1`, and `tanh(x1 - x2)*(tanh(x1 - x2)*(-1))` is
`-tanh(x1 - x2)^2`, as written flat. The bare
(certificate-free) context, such as the entry parse, the bare measure or a rule's sides, uses
the same tables and rule and differs only in the collections it may license. A class has one
definition (`class_base`), which the tables, the like terms, the sign trade and the search's shared
factors all read: a sum, or an odd function or odd root (of an odd index of at least 3) of a
negatable argument, a sum or a literal multiple `c u` (a bare `u` is `1 u`, a negation `-1 u`).
So an odd function of a literal multiple is a sign-trade site wherever it stands, alone as in a
product (`sin(-x1)` is `-sin(x1)`, `x2*sin(-x1)` is `-x2*sin(x1)`), and an odd function of a
literal multiple and of its negation are one class: `sin(2*x1)/sin(-2*x1)` and
`sin(x1)/sin(-x1)` are `-1`, `sin(2*x1)*sin(-2*x1)` is `-sin(2*x1)^2`, and an even power of one over a negative literal files
the positive one (`sin(-2*x1)^2` is `sin(2*x1)^2`); `x2*atan(-2*x1)` and `-x2*atan(2*x1)` are
one answer, so their sum is `0`, and `x3*sin(x1) - sin(x1)` is `sin(x1)*(x3 - 1)`. A join whose
coefficient becomes 1 re-places its key's
signs (`term_join`). The decision is two-tier:

1. $\mu$ decides where it can (strict argmin over the materialized orbit);
2. an exact $\mu$ tie goes to the sign key, and a tie in that to the structural order `cmp_ex`.

**Convention (the sign key: fewest signs, then outermost).** A *sign* is a negative number of
the canonical form: a negative coefficient of a product or of a sum's term (a negation is the
coefficient $-1$) or a negative literal argument. A literal exponent is not a sign (a negative one
is a reciprocal, printed as a division), nor is the sign of an infinity, which is its value
($-\infty$ and `log(0)` are one constant, not two placements). A sign inside a *ground* argument of a
function (one without a variable, a pattern variable or $\diamond$) belongs to that constant and
is not counted, so a constant keeps its sign on its literal: `-sin(2)` is `sin(-2)` and
`x1 - sin(2)` is the state `x1 + sin(-2)`. The sign key of a form is the number of its signs,
then the sum of their depths (the root at depth 0). Wherever two spellings cost the same $\mu$ --
a sign-trade site, the product table, a sum's orientation, and the reduction ordering (§5), so
every rule application -- the one with fewer signs wins, then the one whose signs stand
further out, then `cmp_ex`: `sin(-x1)` is `-sin(x1)`, `x1 + sin(-x1)` is `x1 - sin(x1)`, and
`-sin(x1 - x2)` is `sin(x2 - x1)` (one sign, not two). A site standing alone is decided as a
product's one factor, so a class has one orientation wherever it stands: `(x3(x3 - x1) - x2)^3`
is `-(x2 + x3(x1 - x3))^3` alone as beside other factors (two signs either way, the second's
further out). A power over a negated site reads its class, $(-F)^s = \bar F^s$ for the flip
$\bar F$ of $F$, so `1/S^3` and `(S^3)^(1/2)` are read over one class whichever orientation `S^3`
stands in. A rule whose two sides differ only in
where the core places a sign has one canonical state on both sides and is refused at load as
subsumed (G2).

The toggle rides the bag's SIGN CARRIER, mirroring the absorption arms: a Const
carrier dominates (the forall-exists refit eats every sign — coefficient and bare
infinities normalize positive, orientations are then free), else a bare-infinity
factor carries the sign itself, else an absorbing sum takes it member-wise, else the
rational coefficient. Sums under the absorption owners (Const-bearing, absorbing-member,
or holding a bare-infinity term) are never trade sites — absorption owns their signs.
Distributing $\mathrm{Pow}(\mathrm{Mul}, n)$ for negative odd $n$, a negative bag coefficient
pre-folds into the first trade site before distribution, so the spellings $(S)^{-n}$ and
$-(-S)^{-n}$ are chosen entry-independently.

**Convention (poles are read by their limits).** Where the two sides of a pole disagree, the
value there is not part of the function (`1/(x1 - 3)` at `x1 = 3`); where they agree, the answer
has that limit. A *class* is the set of factors of a product that are odd in one sum $S$, read
compositionally (`zero_read`): $S$, $-S$ and $cS$ for a constant $c \neq 0$ (sums equal up to a
constant factor and to factors every term shares are one class, keyed by the primitive part, an
odd function of a negated argument read as the negated term); the parity of a product is the
product of its factors' parities; an odd function or odd root of an odd argument is odd; an even
function or an even integer power is even; a sum has a parity only if all its terms share it.
A class is written in one orientation with its sign in the coefficient, in every mode, and is a
sign-trade site only as a whole, so a single member never flips: `-1/(x1 - 3)` and `1/(3 - x1)`
are one answer, and `1/((x1 - 3)*(3 - x1))` is `-1/(x1 - 3)^2`. At the zero of $S$ each member's
float value is read from its parts (an exactly cancelling sum evaluates to $+0$); a placement is a
candidate only if the product's value at the zero is its limit, so an even product with an odd
count of members whose zero sign differs is refused and no sign moves through a pole whose two
sides agree. A member defined beside the zero on one side only (a log, an even root, a fractional
or symbolic power of an odd member, or a sum such members dominate there because they are infinite
at the zero and its other terms finite) fixes that side: the class is written in the orientation
positive there, so every base over $S$ is $+0$ at the zero and the float value is that side's
limit (`(3 - x1)^(-1/2)/(x1 - 3)` and `-(3 - x1)^(-1/2)/(3 - x1)` are `-1/(3 - x1)^(3/2)`,
`-inf` at `x1 = 3` like the limit from `x1 < 3`, where the first input gives `+inf`). Members defined on opposite sides leave no domain and fix nothing, and a product that is
NaN at the zero in every orientation (an infinity times a zero, `log(x1 - 3)*(3 - x1)`) keeps no
value there. The sign trade, the product table's placements and the sum's like terms ask one
predicate (`zero_limit`); where a member's value at the zero is not read, the product keeps no
value there and the class takes its cheapest orientation. A lossy context keeps no value at a
point. A sign never moves where the class's zero set may hold a region (`zero_set_null`, decided
once per base and call). Like terms collect up to sign (`add()`
keys a term by its classes in one orientation), and a pattern variable that occurs twice binds a
class up to sign (in every mode for `_`, `!` and `$`, where they bind a class; for `?` only where
it binds any subtree, in `permissive`), the matcher retrying in one orientation only after a
binding was refused by its own negation; the search's FACTOR move (§4) takes a factor shared up
to sign the same way. So `x1/(x2 - x3) + x4/(x3 - x2)` is `(x1 - x4)/(x2 - x3)` and
`x1 + sin(x3 - x2) - sin(x3 - x2)` is `x1`. Residuals [EMPIRICAL]: where a member's value at the
zero is not read, the class's cheapest orientation can leave that value off the limit (a sum holding
the class among terms free of it: with the search off, `1/((x1 - 3 - (x1 - 3)^3)*(3 - x1))` is
`+inf` at `x1 = 3` where its limit is `-inf`; a class scaled by a factor of unknown sign:
`log(x2*(x1 - 3))/(3 - x1)` is `+inf` at `x1 = 3, x2 = -1` where its limit from `x1 < 3` is
`-inf`); a sum whose terms share a factor that is undefined on a region is not factored in `f64`
and `real`, since FACTOR asks its content to be finite almost everywhere, which `log(3 - x3)` is
not (it is NaN for `x3 > 3`), although the law fails only where the content is infinite: so
`log(3 - x3)/(x3 - 3) + log(3 - x3)` stays two terms and is NaN at `x3 = 3`, each term reading
its own limit (`-inf + inf`), where the limit is `+inf`; joined, the product would hold the class
inside a sum among terms free of it, the residual above, and the cheapest orientation of that sum,
`log(3 - x3)*(1/(x3 - 3) + 1)`, is `-inf` there, which is `permissive`'s answer (the joined
`log(3 - x3)*(1 - 1/(3 - x3))` keeps the limit); one class split across the
terms of a sum with a dominant even term can give NaN at its zero where the limit is infinite;
where two singularities meet, the float value can differ from the joint limit (most such points
have none); polynomial factors are not classes (`9 - x1^2` does not hold `3 - x1`), and a sum that
vanishes at a class's zero only through a term that vanishes faster there
(`(3 - x1) + x2*exp(-1/(x1 - 3)^2)`) is a class of its own, so its sign can move alone. An odd
root keeps the sign of zero like every odd function (IEEE-754 `rootn(-0, n) = -0` for odd `n`), so
it reads at the zero as its radicand does.

A zero that holds everywhere (`x2 - x2`) has no limit to read. In `f64` and `permissive`, whose
evaluator is float64, it is IEEE-754's signed zero: where a product holding an exact zero is
inverted, the inverse distributes over its members, the zero giving `+inf` (it is `+0`) and each
other member its own power, so the answer keeps the sign the members give the zero:
`1/(x1*(x2 - x2))` is `inf/x1`, `1/(-(x1 - x1))` is `-inf` and `x1/(0/x2)` is `inf*x1*x2`.
`real` reads the one zero of the reals, whose inverse is `+inf` (`c/0 = sign(c)*inf`): there the
three are `inf`, `inf` and `inf*x1`. A zero's sign survives only as a member of the product
that is inverted, since the canonical form holds one exact zero: through an odd function or a
sum it is lost, so `1/tan(-(x1 - x1))` and `1/(0*(-2) - 0)` are `+inf` in every mode, where
float64 gives `-inf`.

**Convention (factored vs. flat scaling, finite and infinite alike).** The constructors
extract a sum's content only in the unanimity case (it must never lengthen a term), so with
the search off (`effort=0`) a factored spelling like `-2*(x1 - 5/2)` and its distributed equal
`5 - 2*x1` are *both* stable representatives. The search (§4) joins such a pair where
distributing is strictly cheaper: at the default effort both are `5 - 2*x1`, in every mode. It
does not factor a numeric content out of a sum (FACTOR takes shared bases, not a shared
coefficient), so `2*x1 + 6` stays as written although `2*(x1 + 3)` prices lower, and both remain
stable representatives. This is a knowing restriction on cross-spelling convergence of
*scaled* families, identical for finite and infinity-bearing sums — not an infinity-specific
defect.

## 4. The rule layer

A **rule** is a pair $(\ell, \rho)$ of canonical patterns — terms over $T$ extended with
sorted wildcard leaves (sorts `_`, `!`, `$`, `?` bind under certificate side conditions;
see the matcher). Rules come from a mined asset in raw prefix form and pass through
**translation** (`AcRules::translate`), which parses both sides with the *same* boundary
converter as subjects, canonicalizes them with a bare (certificate-free) context in the
reading of the mode whose file it is, and applies seven load-time gates (numbered G1–G4 and
G6–G8; there is no G5):

| Gate | Drops / normalizes | Why |
|---|---|---|
| G1 | unparseable sides | fail closed |
| G2 | $\mathrm{nf}(\ell) = \mathrm{nf}(\rho)$ (arithmetic-subsumed) | the engine's own arithmetic already performs the rewrite |
| G3 | rules whose canonical RHS holds more $\diamond$ than their canonical LHS | masking is representation, not simplification |
| G4 | non-composite LHS | catch-alls below the arithmetic waterline |
| G6 | RHS wildcards not bound by the **canonical** LHS | an unbound wildcard would panic substitution at rewrite time; canon can erase LHS wildcards, so the raw check is insufficient |
| G7 | $\mu(\mathrm{nf}(\rho)) > \mu(\mathrm{nf}(\ell))$ (uphill patterns) | no loaded rule climbs; sides of one price load, and the `oriented` gate decides each application under the full ordering (§5) |
| G8 | one FACTOR move of the search turns $\mathrm{nf}(\ell)$ into $\mathrm{nf}(\rho)$, under the search's licence at every binding the wildcard sorts admit | the search owns the distributive law (below); a rule stating it would take the first factoring that lowers its own node. A law the constructors own, such as the collection of powers of e ($e^a e^b = e^{a+b}$), needs no gate of its own: a rule stating it, in either direction, has $\mathrm{nf}(\ell) = \mathrm{nf}(\rho)$ and G2 refuses it |

Both directions of a rewrite whose sides cost the same may load (on the shipped `acj-5-4-llm`,
`acos(0) - _0` and `asin(1) - _0`): $<_o$ is strict and total, so on any instance at most one of
them descends it, and the `oriented` gate fires only that one; neither undoes the other, and no
separate check for the inverse of an earlier rule is needed. Each admitted rule $\ell \to \rho$ whose negated LHS absorbs the sign
also loads its orientation twin $-\ell \to -\rho$, through the same gates. Before translation,
the licence registry refuses a rule that changes a defined extended-real value at a pole.
Every drop is counted by reason (`ac_rules_drop_census`). On the shipped `acj-5-4-llm`
[EMPIRICAL — re-verified on every load, since translation runs the gates each time]: <!-- re-measure after the re-mine: the translation counts below -->

| mode's file | rows reaching translation | G2 | G4 | G7 | G8 | twins | served |
|---|---|---|---|---|---|---|---|
| `f64` | 138,334 | 83 | 0 | 0 | 0 | 337 | 138,588 |
| `real` | 133,294 | 838 | 44 | 43 | 5 | 390 | 132,754 |
| `permissive` | 139,299 | 122 | 0 | 0 | 5 | 457 | 139,629 |

No row is dropped by G1, G3 or G6, and the registry refuses none.

**How a rule is mined, per mode.** Each mode $m$'s file is mined in $m$. For a source $s$
(the canonical projection of its class in $m$'s reading) the mark is $m$'s own answer
$\mathrm{simplify}_m(s)$ under the rules of the shorter lengths, and a candidate $t$ is
*accepted* iff it passes, in order: the licences (no `<constant>` introduced into a
determined source, no special constant absorbed, $\mathrm{nf}_m(t) <_o$ the mark in $m$'s
serve ordering, no instance of an engine-owned law — G8's predicate — and no defined
extended-real value changed at a pole), the numeric agreement on the mine's matrix (f64
values in `f64`; a near miss re-judged at high precision in `real` and `permissive`), and
the same agreement on an independent confirm matrix. The scan visits candidates in a fixed
order (μ tiers ascending, library order inside a tier) with per-source seeds that depend
only on the source's global index, so it is a deterministic function $\mathrm{pick}(s, E)$
of the source and a set $E$ of excluded canonical classes, returning the first accepted
candidate whose class is not in $E$.

**The judge loop** [THEOREM]. Let $J$ be the contract judge and $L_m$ the tiers $m$
licenses. For each source, start with $E = \emptyset$; while $p = \mathrm{pick}(s, E)$
exists and some spelling of $p$ (its pattern, or an entry $m$ serves from it) has
$J \notin L_m$, add $p$'s class to $E$. Since $\mathrm{pick}$ only skips excluded classes,
the loop ends at the first candidate in scan order that is accepted and whose spellings
$m$ licenses — the result of a scan that asks $J$ about every candidate — or at none. The
candidate set is finite, so the loop terminates; the implementation caps it at 8 rounds
per length and fails the mine there. Only the loop's survivors are installed before the
next length's marks, so a rule $J$ refuses in $m$ never changes a mark of $m$. The
proposals are one more length of the same loop. The file gates then re-judge the
post-processed file (sort promotion respells) and drop what $m$ does not license, so every
shipped row satisfies $J \in L_m$ in its file spelling and in every entry $m$ serves from it.

**The step relation.** For canonical $t$, one *step* $t \Rightarrow t'$ is one of:

- **(fire)** at some node $u$ of $t$: a rule $(\ell, \rho)$ matches $u$ — for bag nodes, a
  sub-multiset match binding $\ell$'s members and leaving a remainder $\mathrm{rem}$
  (Peterson–Stickel extension behavior); the candidate is
  $u' = \mathrm{nf}(\mathrm{bag}(\mathrm{rem} \cup \{\rho\sigma\}))$, and the step is
  **accepted only if** $u' <_o u$;
- **(fold)** at a node $u$ that is *ground with a certified non-finite value class*,
  $\diamond$-collapsible, or (in `permissive` only) ground with a finite value: $u' =$ the
  class literal (`nan`/$\pm$`inf`), $\diamond$, resp. the literal of that value,
  **accepted only if** $u' <_o u$. Exact rational arithmetic lives in the constructors (bag
  coefficient merging, integer/rational `pow`, exact roots, the two determined reciprocal arms
  $0^{-k} = +\infty$ and $(\pm\infty)^{-k} = 0$), and a ground composite whose rigorous
  interval *class* is non-finite folds to that class literal in every mode (a class is an
  exact fact — no finite value is read, so no rounding can occur). Two shapes beside it fold to
  `nan` the same way: a `rootn` whose index is ground and whose rigorous value set reaches no
  nonzero integer is `nan` at every radicand (`rootn(x1, asinh(tanh(4)))`, the ground form of
  the constructor's literal arm `rootn(x1, 2.5)`; an index whose enclosure holds an integer
  stays, so `rootn(x1, acos(cos(1)))` is kept even in `f64`, whose evaluator rounds that index
  off 1), and a variable-free composite holding a $\diamond$ whose class over every value of its
  constants is `nan` (`acosh(acos(cosh(c)))`, the class dropping a finite part on a null set of
  the constants as for any family). The interval layer reads a positive base under a
  non-integer exponent by its four corner powers, so `acosh(1/pi^(1/6))` classifies as `nan`.
  Where a finite value would
  be read, the modes differ, because each reads a different function. *`real` evaluates
  nothing*: a finite transcendental identity (`cos(0)` $\to$ `1`, `cos(pi)` $\to$ `-1`,
  `exp(-inf)` $\to$ `0`) arrives as a mined, symbolically certified *rule*, and `asin(1e-8)`,
  `sin(-1)` and `exp(9.55893790547399)` stay as written. *`f64` and `permissive` fold in this
  step*, and no constructor folds a transcendental in any mode: a ground node no rule rewrote is
  evaluated by the deployed f64 evaluator (the system libm, each literal at its nearest float)
  and offered as the shortest decimal that reads back as a normal (or zero) result, kept only
  where it descends. The pass reaches an enclosing node before its children, so
  `rootn(exp(19.11787581094798), 2)` is `14170.787488305119` (in `f64` too: the constructor fold
  it replaces made the inner `exp` a 9-digit literal first); `asin(1e-8)` is `1e-8`,
  `sin(91618929858.17857)` is `-0.9807991153359534`, and `sin(1)` stays (its float costs more).
  `atanh(tanh(30))` is `30` in `permissive`, whose constructor identity reads the shape first,
  and `inf` in `f64`, where the identity does not hold (`tanh` saturates): the whole node's float
  is no number, so the fold moves on to `tanh(30)`, whose float is `1`, as the evaluator computes
  it. `complexity` prices the reading, so a ground constant is priced as written in every mode. Respelling one exact symbol as another is
  a constructor's job in every mode: `exp(1)` $\to$ `e` chooses between two exact names for
  one value and nothing is rounded;
- **(rebuild)** at a node $u$ whose children were rewritten: $u' = \mathrm{nf}(u[\text{new
  children}])$, **accepted immediately if** $u' <_o u$. If the immediate test fails, the
  rebuild gets one **composite** chance: the ordinary pass — with composite exploration
  *disabled* and a private memo — is run from $u'$, and the composite step
  $u \Rightarrow \mathrm{endpoint}$ is accepted **iff** the settled endpoint is strictly
  below $u$. Otherwise the rebuild is refused and $u$ stands (the children's changes at
  this position are discarded). One level of exploration is deliberate: an oscillating
  reassembly's endpoint is $u$ itself — never strictly below — so oscillators are refused
  without regress, while an uphill intermediate that pays for itself (an integer power
  distributing over a freshly minted product, whose all-literal factor then folds) is
  committed on the strength of where it lands.

Acceptance is checked on the *instantiated, renormalized* result at the actual node —
never predicted from the pattern (see §5 for why prediction is impossible).

**The search (exploration phase, `effort`).** After the chain's fixpoint, the search proposes
moves the step relation refuses, because they ascend at their node: the distributive law, in
both directions and on sub-multisets, owned by the engine and not by any rule (G8).
**DISTRIBUTE** multiplies a product out over its sum factors, moves one factor into one sum
factor, or the other factors over one term of a sum factor (and **POW-EXPAND** expands an
integer power of a sum); **FACTOR**, at a sum, takes what a group of its terms share out of
exactly those terms, by everything they share, and leaves the other terms. The groups are the
terms holding a base two or more terms share (as spelled, or up to a class's orientation, §3),
every two terms sharing a base, and every term holding all that two share; a bare term is its own
product with the unit coefficient, and a term $c \cdot S \cdot q$ whose sum factor $S$, or $-S$,
stands spread among the other terms factors as $S \cdot (c q \pm 1)$. In `permissive` the
search also proposes **RECIPROCAL JOIN**: permissive's `pow` distributes a reciprocal over a
product unconditionally, so `c/(A*B)` is held as `c * A^-1 * B^-1`, where no DISTRIBUTE reaches
the product inside the reciprocal; the move joins two reciprocal sums (`A^-1 * B^-1` becomes
`(A*B)^-1`, with `A*B` distributed into one sum), or moves the coefficient or another factor into
one reciprocal sum (`c * A^-1` becomes `(A/c)^-1`, with `1/c` distributed over `A`'s terms). So
`x2/(x2 - x1)` is `1/(1 - x1/x2)` in `permissive`; `f64` and `real` keep such a reciprocal
joined where it is not certified to split, and their DISTRIBUTE reaches inside it. Each
move carries its licence (every piece finite a.e., so the identity cannot meet $\infty - \infty$;
no `<constant>` duplicated or merged). A candidate is the move at one place with its ancestors
rebuilt through the constructors, so it is the **whole state**; it is descended by the step
relation to its fixpoint and accepted **iff** that endpoint is strictly below the current state
in $<_o$ — the one acceptance for every move kind. A round tries the current state's candidates
cheapest rebuilt state first (ties in pre-order), steps to the first accepted one and starts
again there; a round that accepts nothing ends the search. `effort=k` is the first $k$ descents
of this walk, and `effort=0` is the chain alone (rules and constructors), which does not factor.
Whole-state acceptance is what the law needs: factoring is not confluent (`x1*x2 + x1*x3 + x2*x4`
has two factorings, and which pays is decided by the merges it enables in the enclosing product),
so it is not a rule, whose acceptance is local to its node.

## 5. The reduction ordering

$$ a <_o b \iff \big(\mu(a), \sigma(a), a\big) <_{\mathrm{lex}} \big(\mu(b), \sigma(b), b\big) $$

where $\mu$ is the **unified simplicity measure** (`ac::expr::complexity` — description
length under an exactness-respecting cost model). The integer carrier is the
**milli-bit** ($\tfrac{1}{1000}$ bit): a literal's cost is the real
quantity $L(n) = \log_2(1+|n|)$ rather than a bit count — bit lengths quantise exactly
where the ordering must discriminate — while the ordering stays integer, no float ever
entering a comparison. In bits, each grammar symbol carries its own price -- the
**symbol table**:

| symbol | bits | | symbol | bits |
|---|---|---|---|---|
| variable leaf | 6 | | $\pi$, $e$ | 4 |
| `Add`, `Mul`, `Pow` bag or head | 3 | | $\pm\infty$, NaN | 8 |
| elementary head (`exp log abs sin cos tan rootn`) | 6 | | any other head | 8 |
| inverse/hyperbolic head (`asin acos atan sinh cosh tanh asinh acosh atanh`) | 8 | | a sign | 1 |

The entries are read against the $1/8$-bit unit and scale with it, so the
symbol-vs-literal *ratio* — the sensitivity axis that matters — is independent of the unit. Two of them are fixed by
laws a frequency count cannot see: a **named constant costs less than the cheapest
expression denoting it** ($\mathrm{acos}(-1) = 11$ bits, $\exp(1) = 8$, so $4$ has
room), and a **leaf must also name which variable it is** -- the class `Leaf` carries
$1.65$ bits of the corpus census, a specific variable among eighteen carries $5.82$, and
it is the second number the code has to charge for Kraft's inequality to hold and for
collection ($x + x \to 2x$) to stay a descent. A numeric literal is priced **on its
exact value**, spelling-free, by a
**two-codeword codebook behind a one-bit selector**:

$$ \mu(v) = 1 + \min\Big( \underbrace{\max\big(1,\; L(|p|) + [q \neq 1]\,L(q) + [v < 0]\big)}_{\text{rational}},\;
   \underbrace{\max\big(1,\; L(m)\big) + L(|k|) + [v < 0]}_{\text{decimal-scientific}} \Big) $$

where $p/q$ is the reduced rational — an integer's denominator is implicit and free, a
genuine fraction pays both components — and $m \cdot 10^{k}$ is the shortest exact
decimal spelling, which exists iff $q = 2^a 5^b$ and $k \neq 0$ (for an integer, $k$
is its count of trailing zeros: $1000$ is the codeword $(1, 3)$; $k = 0$ degenerates
to the rational codeword and is refused as redundant). Each codeword total carries its
own **sign bit** ($\mu$ must tell a number from its negation) and a one-bit mantissa
floor (at one bit it binds only where a codeword could not be shorter anyway, and
$\mu$ still tells $x^2$ from $x^3$); a value
with no terminating decimal pays the rational codeword alone — still plus the
selector, so every priced numeric leaf pays exactly one selector bit and the codebook
is a genuine prefix code. An integer, and a value whose decimal codeword is the cheaper
one, prints from the same decomposition $\pm m \cdot 10^{k}$ (`Rat::decimal_parts`): positionally for
$10^{-4} \leq |v| < 10^{16}$, and as `<m>e<exponent>` with one digit before the point
outside that range where that is shorter (`1e-300`, `1e16`, `2.5e-21`; `0.001`, `1000` and
`12345678901234567890` keep their digits). **A sign costs one bit wherever it sits**: the
sign bit of a literal (a number, a coefficient, an exponent), the coefficient $-1$ of a
product beside its other factors ($-x_1 x_2$ costs a bit more than $x_1 x_2$), and a negation
standing alone, which is a sign and not a product: $-x$ pays $x$ and one bit, and no `Mul`
head ($x_1 - x_2$ costs a bit more than $x_1 + x_2$). The bit is absolute, like a literal's,
and does not scale with the symbol unit. So where a sign may stand is never a question of
price, and the sign key decides it ($-x_1 \tanh(3)$ and $x_1 \tanh(-3)$ cost the same; the
second is filed). The coefficient $+1$ is no sign and costs $0$, and a rational exponent of
magnitude $1$ is a reciprocal, priced as division ($x^{-1}$ is the `Pow` head and $x$); every
other coefficient or rational-exponent slot pays its literal cost; a numeric literal that is no number to the engine (beyond the
cap, or in f64 mode not admissible) lives as a token string and pays the same two-codeword
rule parsed from its canonical print (`mu_numeric_str`, monotone in significand and
scale within each codeword, with the astronomic knee at scale $2^{32}$ keeping the
codomain inside `u64`); and the
free placeholder $\diamond$ costs $c_{\mathrm{free}} = 67$ — **derived, not
chosen**: the supremum of the codeword
minimum over f64 round-trip spellings ($64.649$ bits, attained at
$8.9002954340287245 \times 10^{-308}$ — the largest 17-digit mantissa a shortest
repr can carry, $m < 10 \cdot 2^{53}$, meeting the deepest scale a 17-digit spelling
reaches) plus the sign bit and the selector bit, ceiled. The ceiling is robust to the
exact argmax — analytically $L(m) < 56.48$ and $|k| \leq 343$ put the supremum below
$64.92$ — and the supremum is scoped to the f64 range on purpose: $\diamond$ stands
for a value a fit will supply, and dominating exactly those is what the construction
guarantees (a beyond-f64 literal can price below it — $\mu(10^{-400}) = 10.647$
via its scientific codeword — and nothing rests on outpricing those). An exact $\mu$ tie goes
to the sign key $\sigma$ (§3: the number of signs, then the sum of their depths, compared as a
pair: fewest signs, then outermost), and a tie in both to the canonical total order `cmp_ex`
(rank, then structural lexicographic comparison, with *exact* rational comparison — the 256-bit
`cmp_exact`). There is no separate literal-size tier: $\mu$'s literal component carries its
content up to float precision (the float cap below), and the 1,100-bit cap keeps the literals
finite (L5).

**The float cap.** The evaluator reads every literal as its nearest float64, so a literal's
price is capped at that float's shortest round-trip decimal $d(v)$ (at most 17 significant
digits), priced by the same codebook: $\mu'(v) = \min\big(\mu(v),\, \mu(d(v))\big)$ for a
finite nonzero nearest float, $\mu(v)$ otherwise. The cap holds however a term holds a
literal: f64 mode admits a number only where its numerator and denominator are at most
$2^{1022}$ and holds any other value in *pieces* -- a numeric string as written, or a sum,
product or integer power of numbers that would leave the domain. Pieces whose exact value
is a number within the 1,100-bit cap price as $\min(\text{their own prices}, \mu'(v))$, the
price real mode gives the one number, never below one bit per node (`piece_price`); a
numeric string beyond the cap keeps its print's price (`mu_numeric_str`). Every literal that
is its own float's
shortest decimal keeps its price (every constant drawn for flash-ansr's training data among
them), and no price rises; a literal carrying more digits than its float needs (the 64-digit
fourth power of a 16-digit constant, or a fitted constant printed with 17 digits where 16
suffice) costs its float's shortest decimal. The cap is a decimal, never a fraction: a fraction that reads as the same
float is no candidate unless it is the literal itself, so `0.3333333333333333` keeps its
16-digit price while `1/3` costs 4 bits. Below, $\mu$ denotes the capped price wherever it
prices a term.

**Two codewords, one value.** The minimum is over *codes for the same exact value*,
never over values — $\mu$ stays spelling-free, and the symmetry the codebook buys is
at the codeword level: $\mu(1000) = \mu(0.001) = 4$, both being the codeword
$(1, 3)$, where a fraction-only code would price them $9.967$ against $10.967$ — a scale
asymmetry with no informational content, since fitted constants arrive as
decimal-printed f64s and a fraction-only code forces them through "two arbitrary
integers" at $\log_2 10 \approx 3.32$ bits per decimal shift.

**The emitted spelling is the priced spelling, up to the float cap** [BY CONSTRUCTION]. The serializer
chooses each literal's print by the argmin over the *same* two codeword totals the
measure minimizes (`decimal_spelling_wins`), so $\mu(t)$ is the description length
of the representation actually emitted, not of a hypothetical one — except where the float
cap applies: the state is exact and prints its exact value, which then costs more than its
price; the capped price is that of the float's shortest decimal (what Python's `repr`
prints for it, up to the last digit on an exact tie). An exact codeword
tie goes to the fraction — the structurally distinguished caller-dialect member,
mirroring tier 2 of the sign-placement convention (§3); states carry no spelling, so
the tie-break must be spelling-free, and either member realizes the same $\mu$. The
clause is defensive: no exact tie exists in the reachable `i128` lattice [EMPIRICAL —
exhaustive scan over all 3,563 denominators $2^a 5^b < 2^{127}$ at dense-plus-spread
numerator samples; the closest observed gap is 3–4 milli-bits; numbers beyond 128 bits
were not scanned].

- $\mu$ takes values in $\mathbb{N}$ [THEOREM — trivially].
- `cmp_ex` is a strict total order on $T_{\mathrm{can}}$ [BY CONSTRUCTION — rank +
  structural induction; antisymmetry holds because normalized `Rat`s are equal iff
  identical and interned tokens are equal iff their strings are; exactness of the literal
  comparison is oracle-tested against an independent Euclidean-descent comparator].
- $\sigma$ takes values in $\mathbb{N}^2$, ordered lexicographically [BY CONSTRUCTION].
- Hence $<_o$ is a strict total order on $T_{\mathrm{can}}$ [THEOREM, given the above].

**Lemma L5 (finite level sets)** [THEOREM]. For every $\mu_0$, only finitely many
canonical terms $t$ have $\mu(t) = \mu_0$. *Proof.* $\mu$ bounds the node count
(every node and leaf contributes $\geq 1$ bit -- a negation standing alone is one bit, every
other node and leaf $\geq 2$ -- except at most one zero-cost coefficient slot $+1$ per bag and
one zero-cost exponent slot per `Pow`, and the pieces of a literal priced together pay at
least one bit per node, so $\#\mathrm{nodes} \leq \mu_0$ in bits — the carrier
being milli-bits scales both sides by $1000$),
hence finitely many shapes; the non-leaf alphabet (operators) and the variable/special
vocabulary are finite; and there are finitely many numbers at all: a number's numerator
and denominator have at most 1,100 bits each. $\mu_0$ does not bound a number's codeword
components (the float cap prices every literal at most at its float's shortest decimal), so
the finiteness of the literals comes from the 1,100-bit cap, not from the price. A numeric-string
leaf not admissible in f64 mode but within the cap denotes one of those numbers; one beyond the
cap pays a cost that grows with its
canonical print (`mu_numeric_str` is strictly monotone in significand digits and in
the scale within each codeword), so $\mu_0$ bounds the print's significand length
and scale magnitude too, leaving finitely many leaf choices per slot.
$\square$

**Theorem T-wf ($<_o$ is well-founded)** [THEOREM]. There is no infinite strictly
$<_o$-descending sequence. *Proof.* Along such a sequence $\mu$ is non-increasing, so
eventually constant; the tail then lives in one finite level set (L5) and descends the
strict total order $(\sigma, \texttt{cmp\_ex})$ on it, so it is finite. $\square$

A dense-literal chain ascends while its literals carry no more than float precision: the
chain $\mathrm{Mul}[3/2^k, x]$ grows strictly in $\mu$ with $k$ up to $k = 55$; at $k = 56$
and for $58 \le k \le 1022$ in f64 mode ($1076$ in real mode), $3/2^k$'s exact
spelling costs more than its float's shortest decimal, so the literal costs that decimal's
price (the term 64–72 bits up to $k = 200$), which no longer ascends strictly. It stays finite all the same — $k$ is
bounded by the 1,100-bit cap, and T-wf holds through L5 — and a rewrite cannot walk it
anyway: every step preserves the term's value, and changing one literal alone changes it
(pinned in `tests/test_unified_measure.py`: the chain ascends up to the float cap's onset, and
simplifying the chain's members settles and re-reads to itself). A pure re-sort or re-orientation preserves
$\mu$ (the leaf multiset and shape are unchanged) and is decided by the sign key, then
`cmp_ex`, exactly as before.

**Why numeric-string leaves are priced.** The numeric fold interns its result as a
token; a term can therefore carry literals that exist only as opaque numeric *strings*
(values beyond the 1,100-bit cap, or in f64 mode not admissible). Pricing them at one
vocabulary symbol would break L5 (unboundedly many strings at one level), invert the
ordering at the number/leaf boundary, and license deep-magnitude roundings;
`mu_numeric_str` closes all three.

**The property that fails, and why it matters.** $<_o$ is **not** closed under
substitution or context: $\mu$ is *not additive* (coefficients and exponents carry
positional costs), and every fire renormalizes. Example: the hypothetical rule
$\mathrm{Pow}(\_0, 2) \to \mathrm{Mul}[2, \_0]$ ties on patterns ($\mu = 11.585$ both, no
signs, decided by `cmp_ex`), but the instance $\_0 \mapsto 5$ folds both sides to literals
($25$ at $\mu = 5.7$, $10$ at $\mu = 3$) whose comparison the pattern cannot see. This is why orientation is enforced **per
instance at the fire site** (the `oriented` gate), and why G7's static pattern check is an
*alignment* gate, not the termination mechanism: it refuses only a rule whose pattern sides climb
in $\mu$, and a rule whose sides cost the same loads, its applications each decided by the
`oriented` gate under $<_o$ (a literal instance that is cheaper fires, a tie fires only down the
sign key). It also means the published system's
static termination conditions (non-duplication + size decrease of the rule) do not
transfer as a proof device — see §8.

## 6. Properties

Throughout, "step" means an accepted fire, a fold, or an accepted rebuild (§4), and a
"run" is the full `ac_simplify_ex` execution: $t_0 = \mathrm{nf}(\text{input})$, then
passes $t_{i+1} = \mathrm{pass}(t_i)$ until fixpoint or budget.

**Lemma L2 (step descent)** [BY CONSTRUCTION]. Every step strictly descends $<_o$ at its
node: all three step kinds carry an explicit `oriented` gate — fires and rebuilds in
`try_rules_at` / `rewrite_pass`, and the fold by §4's own acceptance clause ("accepted
only if $u' <_o u$"; the class literal or $\diamond$ is checked
against the ordering, not assumed smaller).

**Lemma L3 (pass contraction)** [THEOREM, from L2]. For every canonical $t$:
$\mathrm{pass}(t) \leq_o t$, with equality iff the pass changed nothing. *Proof.* The
pass's state at each position evolves only through accepted steps (each strictly
$<_o$-descending at that position, hence — by induction up the spine, since a child change
survives only through an accepted rebuild at the parent — descending at the root) or
through refusals, which leave the pre-refusal state. $\square$

**Lemma L4 (no revisits)** [THEOREM, from L3 and totality of $<_o$]. The run's chain
$t_0 >_o t_1 >_o \cdots$ is strictly descending in a strict total order, hence all states
are pairwise distinct: cycles are impossible, and no seen-set is needed — at any level
(per position, per pass, per run).

**Theorem T5 (complexity-tier bound)** [THEOREM]. Any run contains fewer than $\mu(t_0)$
steps that strictly decrease $\mu$ (each drops $\mu$ by $\geq 1$ within
$\mathbb{N}_{\geq 1}$, and $\mu$ never increases along the chain by L2/L3).

**Theorem T6 (unconditional termination)** [THEOREM, from L2/L3 + T-wf]. Every call
terminates without any fuel: within a pass, each position's chain is a strictly
$<_o$-descending sequence (finite by T-wf) and the walk is structurally recursive; across
passes, the chain of pass outputs is strictly $<_o$-descending while it changes (L3),
hence reaches its fixpoint after finitely many passes. *With composite steps:* the
argument is two-tier. The exploration-disabled pass contains no tentative walks and
terminates by the argument above directly; the engine's exploration-enabled pass adds, at
refusal sites, one tentative walk each — a terminating exploration-disabled pass — and
its *committed* steps (immediate or composite) all strictly descend $<_o$ at their node,
so L2/L3 and the descent argument hold verbatim. A discarded exploration is finite wasted
work that changes no state.

*Defense-in-depth* [ENFORCED]. Two bounds remain in the code although T6 makes them
non-load-bearing: the release-mode step cap (`STEP_CAP` = $10^6$ accepted steps per call)
and the outer pass budget (`max_passes`, default 48; one pass reaches the fixpoint, asserted in debug builds against a fresh memo, so the second pass only confirms it). They exist to fail closed against a *bug*
in the ordering invariant (a mis-implemented gate would otherwise loop), not against any
legitimate input; neither has ever been observed to bind [EMPIRICAL], and by T6 a binding
bound now *proves* an implementation bug. When one binds, rewriting stops and the state
reached is returned — sound, possibly non-minimal.

**Equal-$\mu$ ties, measured** [EMPIRICAL, on the 128-bit engine — instrumented build over 50,800 simplify
calls: the 400-expression mined corpus in the `f64` and `permissive` modes, plus 50,000 fuzz
expressions biased toward coefficient/exponent cost shifts and i128-boundary literals].
Tie **fires**: none occurred (0). Tie **rebuilds**: 169 calls (0.3%), every one on a
fuzz input carrying overflow-magnitude literals, at most 4 per call, none from the
mined corpus. Because literals pay their bits in $\mu$'s first component, equal-$\mu$
level sets are finite (L5) and every tie is decided by the sign key and `cmp_ex`; pure re-sorts — the
observed benign tie class — preserve $\mu$ and descend the canonical order.

**Soundness of refusal** [BY CONSTRUCTION]. Every state in the chain is a sound form of
the input (each step is a sound rewrite; refusing further steps at cap or budget
exhaustion merely returns an intermediate state). Cap- or budget-truncated outputs are
correct, just possibly non-minimal.

**$\mu$-descent is not a token bound** [EMPIRICAL — the 400-expression canonical corpus
(`simplipy.verify.default_corpus()`), artifact `acj-5-4-llm`, the default effort]. $\mu$
prices description length, not serialization length, so the chain's $\mu$-non-increase (L3)
implies nothing about the number of tokens emitted. Measured over the corpus, each mode in its
own measure <!-- re-measure after the re-mine: this table -->:

| mode | $\mu(\mathrm{simplify}(e)) \leq \mu(e)$ | strictly cheaper | longer, explicit dialect | of these, at an exact $\mu$ tie | longer, tagged dialect |
|---|---|---|---|---|---|
| `f64` | 400/400 | 138 | 73 | 54 | 34 |
| `real` | 400/400 | 148 | 71 | 51 | 34 |
| `permissive` | 400/400 | 197 | 62 | 36 | 32 |

The longer outputs not at a tie are strictly $\mu$-cheaper yet longer: a cheaper literal or
factoring can cost more tokens to spell. The tagged dialect writes the bag delimiters
explicitly. No answer changes on a second call in any mode. The engine's carried guarantee is
the $\mu$-non-increase, soundness and idempotence; no output-token bound is claimed in either
serialization. (The user-facing statement of this is in
[the simplification guide](guides/simplify.md).)

**Lemma L6a (serialization injectivity — the certificate-cache premise)** [THEOREM,
conditional on `stable()`]. The finiteness-certificate caches are keyed on
`to_prefix(t)`; a collision between distinct canonical states would silently alias
certificate verdicts. On the set of canonical states where the round-trip identity
holds ($\mathrm{canon}(\mathrm{parse}(\mathrm{serialize}(t))) = t$, the `stable()`
assertion), `to_prefix` has a left inverse and is therefore injective — two states
sharing a serialization would be mapped back to the same state by the left inverse.
The identity is exercised per state in debug builds (the full suites run green under
debug, so every state reached by the tests and the mini-mines satisfies it, and so does every
state of srbf's ground truths and model predictions and of the 129,490 simplify inputs of
flash-ansr's training-data generator, in every mode). It fails on the documented I3 residuals at literals that stay leaves. These
residuals scope the cache guarantee too.

**Lemma L6 (conditional idempotence)** [THEOREM, conditional]. If a run reaches a pass
fixpoint within budget ($\mathrm{pass}(t_k) = t_k$ — by T6 the fixpoint *exists* and is
reached whenever the budget suffices) *and* the output serialization round-trips
($\mathrm{parse}(\mathrm{serialize}(t_k)) = t_k$ — the `stable()` assertion),
then re-simplifying the output returns it unchanged. *Proof.* By the round-trip premise,
the re-run's start state — parse then $\mathrm{nf}$ of the serialized output — is $t_k$
itself; its first pass is $\mathrm{pass}(t_k) = t_k$, so the fixpoint test fires
immediately, and determinism gives the identical serialization back ($\mathrm{nf}$, asset
order, and the certificate analyses are all deterministic; the pass memo is fresh per
call). $\square$ Whether the premises hold on real data is what the gates measure [EMPIRICAL —
§7]. The run returns its *final* state — which by L3 carries the chain's minimum
complexity — so L6's fixpoint premise is structural whenever the budget does not
truncate. **The exploration phase** (`effort`) keeps L6 when it runs until a round finds
nothing, the default since 0.15.0: its answer is a valley whose own expansions were all
refused against it, the re-run's chain returns that valley (the premises above), and its
search proposes the same candidates in the same order, which descend to the same endpoints
and are refused again. Two further premises: the step cap does not bind (a capped walk
skips fire sites), and the descent of a candidate does not depend on the memo's contents
(the memo only marks fixpoints). A capped search (`effort=k`) can stop between two accepted
valleys, and the re-run then continues it [EMPIRICAL:
srbf's 125,127 model predictions change on a second call 1,632 times in `f64` at cap 4, the former
default, every one to a cheaper form (`real` the same); uncapped, never, in `f64` and `real`, on these and on
srbf's ground truths and the training-data generator's inputs]. Where
the round-trip premise fails (L6a) the re-run starts from a different state. A further
premise: each accepted candidate's descent reaches its fixpoint within `max_passes`.
`permissive` is one walk like every mode (its chain, then the search, in its own canon and
rules), so L3, L6 and the prefix argument for `effort` cover it as they cover every mode. Its
two steps of its own are ordinary steps: the ground fold is the pass's fold (tried after the
rules at a node, kept only when the result descends, so L3 holds; `f64` takes the same step), and
the
landing rounding ρ moves a literal to one of the two floats around it, the one whose shortest
spelling is cheaper (permissive promises float precision, not the nearest float), and only where
that spelling is cheaper as an exact codeword than the literal: the float cap prices the literal
at its nearest float's spelling, so ρ never raises a price and the descent arguments and T6 are
unchanged (idempotent, since a float's shortest spelling is never moved). Its
search move that joins reciprocal sums is a candidate like the others, accepted only strictly
below. A literal rounded where it lands is exact from then on, so a later step that combines it
rounds again: an answer's literal can sit an ulp or more from the correctly rounded value of the
input's exact arithmetic (each landing moves less than one ulp), and print a digit longer or
shorter than a single rounding of that value would.

**Canonicity across spellings** [EMPIRICAL]. That all spellings of the same bag (operand
permutations, re-bracketings) reach the same representative is measured, not proven:
commutative-permutation and adjacency-collection property tests, plus the corpus
permutation gate (0 violations / 400 canonical corpus expressions). The known residual class
is the I3 residuals above. Re-bracketings of a product reach one representative: the parse
builds a product once from all of its members and `mul()` decides its opposite classes from
their exponent totals, a function of the multiset of members (§3). On 1,653 generated
products over opposite sums, odd functions and odd roots, each in up to eleven spellings (four
bracketings and the tagged bag for all; two division spellings, the inverse of a product, a
chain of divisions and the tagged bag with `<div>` where the product has a divisor; a negation
for 72), no product in `f64` and `real` and one in `permissive` reaches two answers (0.14.7:
616 in `f64` and `real`, 687 in `permissive`), and in `f64` and `real` none changes on a second
call at the same effort <!-- re-measure after the re-mine: the 1,653-product counts -->; on the
route fuzz of 150 products none depends on the nesting in any mode, with the search off or at
the default (0.14.7 with the search off: 87, 87 and 82). Where the certificate refuses to split an inverse over a product (a
factor that vanishes on a region, such as `abs(x2) - x2`), `1/(a*b)` and `1/a*1/b` are different
functions and keep different answers. (The scale gates held 0 idempotence and 0 permutation
failures when last run, on the 128-bit engine.)

**Permissive mode.** `permissive` is the mode for canonicalizing a training corpus: it
accepts rewrites that hold only up to sign or domain, lets every wildcard sort bind any
subtree, and widens the $\diamond$-collapse licence. It relaxes *soundness* licences, never
the ordering: fires, folds and rebuilds remain `oriented`-gated and the search accepts only a
strictly lower state, so L2–T6 hold verbatim in permissive mode.

## 7. Where each property is checked

| Property | Code site (enforcement) | Test / gate (evidence) |
|---|---|---|
| I1–I4 canonical invariants | constructors in `ac/expr.rs` | `no_nested_bags` + `stable()` debug asserts; property tests in `tests/test_ac_core.py` |
| L1 nf termination | structural recursion; the two audited self-calls (`pow`, `fun`/rootn) | full suites exercise both self-call sites |
| G1–G4, G6–G8 translation gates | `AcRules::translate` | poisoned-asset test (`test_free_rhs_wildcards_are_dropped_at_translation`): G6 both flavors + G7 dropped at load; translation counts pinned by `test_translation_audit_surface`; per-reason counts from `ac_rules_drop_census` |
| L2 step descent | `oriented` in `try_rules_at` + the rebuild gate in `rewrite_pass` | `SIMPLIPY_AC_TRACE=1` logs every accepted fire and every rebuild decision (refused, or committed as a composite step) |
| L5/T-wf ($\mu$ literal component) | `mu_rat`/`mu_numeric_str`/`piece_price` inside `complexity` | `tests/test_unified_measure.py` + `tests/test_mu_prime.py`: the weight table and the codeword pins against independently computed expectations; the dyadic chain ascends up to the float cap's onset and its members settle; a value f64 holds in pieces prices as real's one number |
| composite acceptance | the exploration branch in `rewrite_pass` (one level, private memo) | two-spelling convergence test: the distribution-refusal specimen and its pre-simplified spelling reach the SAME form |
| T6 defense-in-depth | `STEP_CAP` in `rewrite_pass`; `max_passes` loop in `ac_simplify_ex` | by T6 a binding bound proves an implementation bug; each binding of the step cap is counted (`step_cap_binds`, pinned at 0 on the corpus in every mode); fixpoint break covered by idempotence gates |
| L6 premises | `stable()` debug assert; fixpoint break | corpus idempotence gate: 0/400 |
| ordering exactness | `Rat::cmp_exact` (256-bit) | Euclidean-oracle fuzz (60k pairs, 3 magnitude regimes) + transitivity triples |

## 8. Relation to the published TRS, and what a new proof must contain

| arXiv:2602.08885 | AC core | Status of the transfer |
|---|---|---|
| terms: binary prefix trees | canonical bags (I1–I4) | redefinition required (done here, §2) |
| cancellation procedure outside the TRS | $\mathrm{nf}$ (§3), much larger | same architectural role; L1 replaces the informal argument |
| rule conditions: $\mathrm{Vars}(\rho) \subseteq \mathrm{Vars}(\ell)$, $|\rho| < |\ell|$, non-duplication | G6; G7; (non-duplication measured on the f64 file of `acj-5-4-llm`: 138,640 of 138,651 rows, not required) | **the static proof device does not transfer** — $c$ is non-additive and fires renormalize (§5), so orientation is enforced per instance instead |
| termination: length is a reduction order | T-wf/T6: the pair $(\mu, \mathrm{cmp\_ex})$ is a well-founded strict total order (no separate literal-size tier — literal content lives in $\mu$, §5) | **new proof, done here, unconditional** |
| iteration cap $K = 5$ | `max_passes` | demoted to defense-in-depth: T6 guarantees the fixpoint in finitely many passes |
| syntactic matching | AC sub-multiset matching with remainder | Peterson–Stickel extension rules; matching soundness is the matcher's contract |

One research direction remains open, now about *completeness* rather than termination.
The composite step heals refusals whose uphill intermediate pays for itself within one
settled walk (measured incidence of refusals before the fix: 0 on the mined corpus, 1 in
50,000 adversarial fuzz calls — that one now commits and the two spellings of its function
converge). What remains refusable: reassemblies whose payoff would need a *further*
exploration inside the tentative walk (nested composite steps are deliberately disabled to
refuse oscillators), and true oscillators themselves. An ordering under which every
$\mathrm{nf}$ fold is non-increasing — e.g. a polynomial interpretation handling power
distribution — would make even those acceptances unconditional and the exploration
machinery unnecessary. That is genuine research on the weight table; the G7 gate would
re-vet any asset against a new order automatically on load, and T-wf-style
well-foundedness would need re-establishing for the new order.

## 9. Change impact

| If you change… | You must re-establish… |
|---|---|
| the rule asset | nothing — the translation gates re-vet on every load (drops are counted by reason, visible in `ac_rules_info_in_mode` and `ac_rules_drop_census`) |
| the $\mu$ weight table, the sign key or `cmp_ex` | T-wf (well-foundedness of the new triple — L5's finite-level-set argument must survive, including the numeric-string-leaf clause); G7 re-vets every asset on load; re-run the corpus gates AND the unit-sensitivity grid: the *reachable* normal forms change |
| a constructor (`add`/`mul`/`pow`/`fun`) | L1 (check any new self-call has a decreasing measure), I1–I4; the fold needs no descent premise (it is `oriented`-gated at the pass); then the corpus gates |
| the matcher | the step-relation definition of §4 (binding soundness); L2 is unaffected (orientation checks the result, not the match) |
| `rewrite_pass` control flow | L2 (every state change gated), L3 (no ungated mutation path), memo semantics (§6 L6 caveat) |
| serialization | L6's round-trip premise (`stable()`), the corpus gates |
