# The Simplification Engine (Formal)

This page is the mathematical specification of the AC simplification core
(`SimpliPyEngine.simplify` since 0.12.0) — the companion to
[The Mining Algorithm (Formal)](algorithm.md), which specifies how rule sets are
*discovered*. This page specifies how they are *applied*: the term algebra, the
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

where $q$ ranges over exact rationals (`Rat`: $p/q$ over `i128`, normalized $q > 0$,
$\gcd(|p|, q) = 1$, $p \neq$ `i128::MIN`), $\diamond$ is the abstract fitted constant
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
- **I3 (coefficient normal form):** a $\mathrm{Mul}$ carries at most one rational factor; an
  $\mathrm{Add}$ carries at most one rational term; rational arithmetic inside bags is
  folded in a deterministic (sorted) merge order, exactly (`Rat` checked ops; overflow
  refuses the merge and keeps the operands symbolic). *Known residual at the overflow
  boundary* [EMPIRICAL, tracked]: when a merge is refused, the bag holds two-plus literal
  members whose relative order can depend on construction history — measured as 2
  idempotence violations in 50,000 adversarial fuzz calls (i128-scale literal pairs only;
  0 on the mined corpus; predates the composite step, verified on the prior build). This
  is the ratified i128-boundedness design boundary; the cure (bignum content + a widened
  token boundary) is an owner-level decision.
- **I4 (fold normal form):** the licensed structural folds of §3 have been applied; e.g. no
  $\mathrm{Pow}(t, 1)$, no $\mathrm{rootn}(t, k)$ with $k \leq 0$ or $|k| = 1$, no
  all-literal composite that the constructors fold — and the sign placement between a
  product's coefficient slot and an odd function's literal argument is decided by ONE
  shared pricing orbit on every construction route (the sign-trade owner; B5+B19,
  2026-08-15): $\mu$ picks the cheaper spelling, exact ties resolve to the
  non-negative coefficient. So $-1 \cdot \sin(2)$ files as $\sin(-2)$ (the product
  node itself dissolves) and $-5 \cdot \sin(2)$ as $5 \cdot \sin(-2)$ — while a
  magnitude-1 coefficient beside *other* factors keeps the sign, because it rides
  $\mu$-free where the literal's sign bit would cost ($-1 \cdot x_0 \cdot \sin(2)$ is
  canonical as spelled). **An EVEN index is canonical** (corrected
  2026-08-07): `rootn(t, 2)` and `rootn(t, 4)` are normal forms, not reducible ones --
  `rootn` is the canonical spelling of a principal even root and $\mu$ ties it with
  $\mathrm{Pow}(t, 1/k)$, so neither orientation is forced. The earlier wording ("or even
  $k$") predates that change and contradicted §5 of this same page.

**Route invariance** [EMPIRICAL, D7 — deliberately *not* claimed as a theorem]:
$\mathrm{nf}$ is deterministic *per route*, but a state built incrementally through
the constructors and the same value's parse-route canon are kept equal by shared-owner
design plus a live instrument, not by proof. The instrument: every pass output must
round-trip (serialize → parse → canon) onto the *same state*, debug-asserted on every
`simplify` call and exercised by the full suite and the corpus gates. The one measured
divergence class (the odd-function literal-sign pair: `mul()`-built vs collector-built
spellings of one value) was removed in 0.13.0 by the shared sign-trade owner; zero
specimens remain. A route-invariant `canon()` proof is deferred with a named trigger
(D7): a *value-changing* divergence at entry canon reopens it immediately; a μ-equal
divergence reopens the deferred route-invariance work. The assert's failure
diagnostics classify any future specimen along exactly that line.

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
(every sum against its negated-flip spelling): zero divergence, at head *and* on the
pre-hardening build — the "canonical doublet" once conjectured here never existed; the
defensive infinity handling inside `primitive_sum` was dead on arrival and has been
removed. Two conventions coexist knowingly: a finite
mixed-orientation sum files in whichever of its two orientations prices LOWER under
$\mu$ — the $-1$ wrapper appears exactly when the wrapped spelling is the cheaper one.
Ruled 2026-08-08: mirrored subtractions score equal, and a strictly larger expression
never scores the same as a smaller one through a hidden wrapper surcharge.
Infinity-bearing sums never wrap; each family has a single representative, so no
canonicity is at stake.

**Convention (sign placement, three tiers — owner-ruled 2026-08-08, full family).**
Signs in a product trade legally across ODD carriers — a bare mixed-sign `Add` factor,
$\mathrm{Pow}(S, n)$ for odd integer $n \ge 3$, $\mathrm{rootn}(S, m)$ for odd
$m \ge 3$, and the eight odd functions — because $f(-S) = -f(S)$ is total on those
carriers (negative odd exponents are NOT carriers: the pole trilemma). The
sign-placement owner (`ac::expr::sign_place`, shared by `mul()`'s final assembly and
`term_join`'s negative joins so the priced spelling is always the built spelling)
materializes every reachable placement and keeps the $\mu$-argmin. The decision is
three-tier:

1. $\mu$ decides where it can (strict argmin over the materialized orbit);
2. an exact $\mu$ tie at a SIGN-TRADE site goes to the structurally distinguished
   member — the positive-coefficient / bare spelling ("what you typed survives; a
   leading minus is only ever minted when strictly cheaper");
3. the historical sorted-coefficient lexicographic comparison survives ONLY at
   free-orientation sites (even carriers: even integer powers and the even functions
   `abs`/`cos`/`cosh`, whose argument orientation is value-free), where neither
   spelling is structurally distinguished.

The toggle rides the bag's SIGN CARRIER, mirroring the absorption arms: a Const
carrier dominates (the forall-exists refit eats every sign — coefficient and bare
infinities normalize positive, orientations are then free), else a bare-infinity
factor carries the sign itself, else an absorbing sum takes it member-wise, else the
rational coefficient. Sums under the absorption owners (Const-bearing, absorbing-member,
or holding a bare-infinity term) are never trade sites — absorption owns their signs.
At the one boundary where a licensed a.e. rewrite meets a refused carrier —
distributing $\mathrm{Pow}(\mathrm{Mul}, n)$ for negative odd $n$ — a negative bag
coefficient pre-folds into the first trade site before distribution, so the
pole-different spellings $(S)^{-n}$ and $-(-S)^{-n}$ are chosen entry-independently
and never conflated.

**Convention (factored vs. flat scaling, finite and infinite alike).** Content extraction
is deliberately restricted to the unanimity case (it must never lengthen a term), so a
factored input spelling like $-2 \cdot (x - 5/2)$ and its distributed equal $5 - 2x$ are
*both* stable representatives. This is a knowing restriction on cross-spelling
convergence of *scaled* families, identical for finite and infinity-bearing sums — not an
infinity-specific defect.

## 4. The rule layer

A **rule** is a pair $(\ell, \rho)$ of canonical patterns — terms over $T$ extended with
sorted wildcard leaves (sorts `_`, `!`, `$`, `?` bind under certificate side conditions;
see the matcher). Rules come from a mined asset in raw prefix form and pass through
**translation** (`AcRules::translate`), which parses both sides with the *same* boundary
converter as subjects, canonicalizes them with a bare (certificate-free) context, and
applies seven load-time gates:

| Gate | Drops / normalizes | Why |
|---|---|---|
| G1 | unparseable sides | fail closed |
| G2 | $\mathrm{nf}(\ell) = \mathrm{nf}(\rho)$ (arithmetic-subsumed) | the engine's own arithmetic already performs the rewrite |
| G3 | $\diamond$-introducing rules | masking is representation, not simplification |
| G4 | non-composite LHS | catch-alls below the arithmetic waterline |
| G5 | exact inverse of an earlier rule | ping-pong fuel |
| G6 | RHS wildcards not bound by the **canonical** LHS | an unbound wildcard would panic substitution at rewrite time; canon can erase LHS wildcards, so the raw check is insufficient |
| G7 | $\mathrm{nf}(\rho) \not<_o \mathrm{nf}(\ell)$ (disoriented patterns) | keep every loaded rule aligned with the ordering the pass fires under (§5) |

All seven are counted; the shipped asset (`acj-4-3`, 6,671 raw rules, mined under
$\mu$ itself on the deployed literal alphabet) loses **zero** rules
and gains ten orientation twins, keeping 6,681
[EMPIRICAL — re-verified on every load, since translation runs the gates each time; a
mine whose acceptance rides the serve ordering produces an artifact the serve ordering
fully admits].

**The step relation.** For canonical $t$, one *step* $t \Rightarrow t'$ is one of:

- **(fire)** at some node $u$ of $t$: a rule $(\ell, \rho)$ matches $u$ — for bag nodes, a
  sub-multiset match binding $\ell$'s members and leaving a remainder $\mathrm{rem}$
  (Peterson–Stickel extension behavior); the candidate is
  $u' = \mathrm{nf}(\mathrm{bag}(\mathrm{rem} \cup \{\rho\sigma\}))$, and the step is
  **accepted only if** $u' <_o u$;
- **(fold)** at a node $u$ that is either *ground with a certified non-finite value
  class* or $\diamond$-collapsible: $u' =$ the class literal (`nan`/$\pm$`inf`) resp.
  $\diamond$, **accepted only if** $u' <_o u$. Since fold unification (2026-08-02) the
  fold performs **no numeric evaluation**: exact rational arithmetic lives in the
  constructors (bag coefficient merging, integer/rational `pow`, exact roots, the two
  determined reciprocal arms $0^{-k} = +\infty$ and $(\pm\infty)^{-k} = 0$), a ground
  composite whose rigorous interval *class* is non-finite folds to that class literal
  (a class is an exact fact — no finite value is read, so no rounding can occur), and
  every finite transcendental identity (`cos 0` $\to$ `1`, `cos pi` $\to$ `-1`,
  `exp(-inf)` $\to$ `0`) arrives as a mined, symbolically certified *rule*. `exp(1)`,
  `sin(-1)`, `sqrt(8)` stay symbolic *structurally* — no arm exists that could
  evaluate them (previously this was enforced by the measure refusing a $\sim$105-unit
  rounded literal; now it is unrepresentable). Respelling one exact symbol as another
  is a different thing and *is* a constructor's job: `exp(1)` $\to$ `np.e` (2026-08-07)
  chooses between two exact names for one value and nothing is rounded, which is what
  makes it admissible where evaluation is not;
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

## 5. The reduction ordering

$$ a <_o b \iff \big(\mu(a), a\big) <_{\mathrm{lex}} \big(\mu(b), b\big) $$

where $\mu$ is the **unified simplicity measure** (`ac::expr::complexity` — description
length under an exactness-respecting cost model). The integer carrier is the
**milli-bit** ($\tfrac{1}{1000}$ bit; §10.10, 2026-08-06): a literal's cost is the real
quantity $L(n) = \log_2(1+|n|)$ rather than a bit count — bit lengths quantise exactly
where the ordering must discriminate — while the ordering stays integer, no float ever
entering a comparison. In bits: a structural node (bag, `Pow`, function head), a
variable leaf, and a special constant ($\pi$, $e$, the infinities, NaN) each cost $8$;
a numeric literal $p/q$ costs $\max(2,\; L(|p|) + [q \neq 1]\,L(q) + [p < 0])$ **on
its exact value**, spelling-free — an integer's denominator is implicit and free, a
genuine fraction pays both components, a negative literal pays one **sign bit** (the
former blanket "sign is free" doctrine is revoked for literals: $\mu$ must tell a
number from its negation), and the two-bit floor covers $0, \pm 1, 2$; a magnitude-$1$
coefficient or rational-exponent slot is a bare sign and costs $0$, every other such
slot pays its literal cost; a numeric literal whose
exact rational exceeds `i128` lives as a token string and pays a description length
parsed from its canonical print (`mu_numeric_str`, monotone in significand and scale,
with the astronomic knee at scale $2^{32}$ keeping the codomain inside `u64`); and the
free placeholder $\diamond$ costs $c_{\mathrm{free}} = 1133$ — **derived, not
chosen** (§10.10(5)): the supremum of $\mu$ over f64 round-trip spellings
($1131.931$ bits, attained at $5.5605781537525765 \times 10^{-308}$) plus the sign
bit, ceiled — the priciest atom by construction (the former asserted floor of $128$
was beaten $8\times$ by $\mu(10^{308}) = 1024.154$). Final ties are broken
by the canonical total order `cmp_ex` (rank, then structural lexicographic comparison,
with *exact* rational comparison — the 256-bit `cmp_exact`). The former middle
literal-size tier is **absorbed**: $\mu$'s literal component carries its content, so
the ordering is a pair, not a triple.

- $\mu$ takes values in $\mathbb{N}$ [THEOREM — trivially].
- `cmp_ex` is a strict total order on $T_{\mathrm{can}}$ [BY CONSTRUCTION — rank +
  structural induction; antisymmetry holds because normalized `Rat`s are equal iff
  identical and interned tokens are equal iff their strings are; exactness of the literal
  comparison is oracle-tested against an independent Euclidean-descent comparator].
- Hence $<_o$ is a strict total order on $T_{\mathrm{can}}$ [THEOREM, given the above].

**Lemma L5 (finite level sets)** [THEOREM]. For every $\mu_0$, only finitely many
canonical terms $t$ have $\mu(t) = \mu_0$. *Proof.* $\mu$ bounds the node count
(every node and leaf contributes $\geq 2$ bits except at most one zero-cost coefficient
slot per bag and one zero-cost exponent slot per `Pow`, so $\#\mathrm{nodes} \leq \mu_0$
in bits — the carrier being milli-bits scales both sides by $1000$),
hence finitely many shapes; the non-leaf alphabet (operators) and the variable/special
vocabulary are finite; $\mu_0$ bounds every in-range literal's bit size; and a
beyond-`i128` numeric-string leaf pays a cost that grows with its digit count
(`mu_numeric_str` is strictly monotone in significand digits and decimal scale), so
$\mu_0$ bounds its string length too, leaving finitely many leaf choices per slot.
$\square$

**Theorem T-wf ($<_o$ is well-founded)** [THEOREM]. There is no infinite strictly
$<_o$-descending sequence. *Proof.* Along such a sequence $\mu$ is non-increasing, so
eventually constant; the tail then lives in one finite level set (L5) and descends the
strict total order `cmp_ex`, so it is finite. $\square$

Because a literal pays its bits, the dense-literal hazard that motivated the former
middle tier now **ascends** outright: the chain $\mathrm{Mul}[3/2^k, x]$ grows strictly
in $\mu$ with $k$ (the former Open O1's specimen class cannot be a reduction sequence;
pinned in `tests/test_unified_measure.py`). A pure re-sort or re-orientation preserves
$\mu$ (the leaf multiset and shape are unchanged) and is decided by `cmp_ex`, exactly
as before.

**Why numeric-string leaves are priced.** The numeric fold interns its result as a
token; a term can therefore carry literals that exist only as opaque numeric *strings*
(values whose exact rational exceeds `i128`). Pricing them at one vocabulary symbol
would break L5 (unboundedly many strings at one level), invert the ordering at the
`i128` boundary, and license deep-magnitude roundings; `mu_numeric_str` closes all
three.

**The property that fails, and why it matters.** $<_o$ is **not** closed under
substitution or context: $\mu$ is *not additive* (coefficients and exponents carry
positional costs), and every fire renormalizes. Example: the hypothetical rule
$\mathrm{Pow}(\_0, 2) \to \mathrm{Mul}[2, \_0]$ ties on patterns ($\mu = 18$ both,
decided by `cmp_ex`), but the instance $\_0 \mapsto 5$ folds both sides to literals
($25$ at $\mu = 5$, $10$ at $\mu = 4$) whose comparison the pattern cannot see. This is why orientation is enforced **per
instance at the fire site** (the `oriented` gate), and why G7's static pattern check is an
*alignment* gate, not the termination mechanism. It also means the published system's
static termination conditions (non-duplication + size decrease of the rule) do not
transfer as a proof device — see §8.

## 6. Properties

Throughout, "step" means an accepted fire, a fold, or an accepted rebuild (§4), and a
"run" is the full `ac_simplify_ex` execution: $t_0 = \mathrm{nf}(\text{input})$, then
passes $t_{i+1} = \mathrm{pass}(t_i)$ until fixpoint or budget.

**Lemma L2 (step descent)** [BY CONSTRUCTION]. Every step strictly descends $<_o$ at its
node: all three step kinds carry an explicit `oriented` gate — fires and rebuilds in
`try_rules_at` / `rewrite_pass`, and the fold by §4's own acceptance clause ("accepted
only if $u' <_o u$"; since the $\mu$ ship the class literal or $\diamond$ is checked
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
and the outer iteration budget (`node_budget`). They exist to fail closed against a *bug*
in the ordering invariant (a mis-implemented gate would otherwise loop), not against any
legitimate input; neither has ever been observed to bind [EMPIRICAL], and by T6 a binding
bound now *proves* an implementation bug. When one binds, rewriting stops and the state
reached is returned — sound, possibly non-minimal.

**History: the former Open O1, and the measurement that shaped the fix** [EMPIRICAL —
instrumented build, 2026-07-28]. Under the original *pair* ordering $(c, \mathrm{cmp\_ex})$,
equal-complexity descent had no proven bound: the literal order is dense, and fires mint
literals, so equal-complexity level sets were not provably finite (an earlier in-code
claim that "the reachable atom set is finite" was wrong for exactly this reason). An
instrumented build over 50,800 simplify calls (the 400-expression mined corpus in strict
and LOSSY modes, plus 50,000 fuzz expressions biased toward coefficient/exponent cost
shifts and i128-boundary literals) located the tie tier precisely: tie **fires** never
occurred (0); tie **rebuilds** occurred in 169 calls (0.3%), every one on a fuzz input
carrying overflow-magnitude literals, at most 4 per call, none from the mined corpus. The
infinitude therefore lived entirely in literal *values* — first bounded by a dedicated
literal-size tier (2026-07-28), then absorbed outright when the unified measure $\mu$
made literals pay their bits in the *first* component (2026-08-01, stage 2 of
`design/UNIFIED_SIMPLICITY_MEASURE.md`), turning the open item into T-wf/T6 with one
fewer ordering layer. Pure re-sorts (the observed benign tie class) preserve $\mu$ and
behave exactly as before.

**Soundness of refusal** [BY CONSTRUCTION]. Every state in the chain is a sound form of
the input (each step is a sound rewrite; refusing further steps at cap or budget
exhaustion merely returns an intermediate state). Cap- or budget-truncated outputs are
correct, just possibly non-minimal.

**Lemma L6a (serialization injectivity — the certificate-cache premise)** [THEOREM,
conditional on `stable()`]. The finiteness-certificate caches are keyed on
`to_prefix(t)`; a collision between distinct canonical states would silently alias
certificate verdicts. On the set of canonical states where the round-trip identity
holds ($\mathrm{canon}(\mathrm{parse}(\mathrm{serialize}(t))) = t$, the `stable()`
assertion), `to_prefix` has a left inverse and is therefore injective — two states
sharing a serialization would be mapped back to the same state by the left inverse.
The identity is exercised per state in debug builds (2026-08-02: the full suites run
green under debug after the determined-pole fix, so every state reached by the tests
and the mini-mines satisfies it); its one known exception class is the documented I3
i128-boundary residual (2/50k fuzz, corpus-unreachable), which therefore also scopes
the cache guarantee — the same boundary, the same bignum cure if ever needed.

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
§7]. *Historical caveat, resolved:* the code formerly returned the first
*minimum-complexity* state rather than the final state; the two differ exactly when a
run's tail descends only the $s$/`cmp_ex` tiers, in which case the returned state is not
a fixpoint. This was dormant until composite steps made such tails reachable (4
idempotence violations in 50,000 adversarial fuzz calls, 0 on the corpus), and was fixed
by returning the final state — which by L3 carries the chain's minimum complexity anyway,
so nothing is lost and L6's fixpoint premise is structural whenever the budget does not
truncate.

**Canonicity across spellings** [EMPIRICAL]. That all spellings of the same bag (operand
permutations, re-bracketings) reach the same representative is measured, not proven:
commutative-permutation and adjacency-collection property tests, plus the corpus
permutation gate (0 violations / 400 canonical corpus expressions at head). One
registered residual class at the 10^6 fuzz scale: the I3 i128-boundary bag-order class
above. (An earlier second class — sign-orientation non-confluence on infinity-bearing
sums under an outer product sign, 2 rows per 10^6 — was closed by making
negation-absorption recursive across its three owning constructors; the scale gates hold
0 idempotence / 0 permutation failures at head.)

**LOSSY mode.** `wildcard_all` widens matching and the $\diamond$-collapse licence
(training-corpus canonicalization). It relaxes *soundness* licences, never the ordering:
fires and rebuilds remain `oriented`-gated, so L2–T6 hold verbatim in LOSSY mode.

## 7. Where each property is checked

| Property | Code site (enforcement) | Test / gate (evidence) |
|---|---|---|
| I1–I4 canonical invariants | constructors in `ac/expr.rs` | `no_nested_bags` + `stable()` debug asserts; property tests in `tests/test_ac_core.py` |
| L1 nf termination | structural recursion; the two audited self-calls (`pow`, `fun`/rootn) | full suites exercise both self-call sites |
| G1–G7 translation gates | `AcRules::translate` | poisoned-asset test (`test_free_rhs_wildcards_are_dropped_at_translation`): G6 both flavors + G7 dropped at load; shipped counts pinned by `test_translation_audit_surface` |
| L2 step descent | `oriented` in `try_rules_at` + the rebuild gate in `rewrite_pass` | `SIMPLIPY_AC_TRACE=1` logs every accepted step |
| L5/T-wf ($\mu$ literal component) | `mu_rat`/`mu_numeric_str` inside `complexity` | `tests/test_unified_measure.py`: the weight table against independently computed expectations; the dyadic chain ascends |
| composite acceptance | the exploration branch in `rewrite_pass` (one level, private memo) | two-spelling convergence test: the distribution-refusal specimen and its pre-simplified spelling reach the SAME form |
| T6 defense-in-depth | `STEP_CAP` in `rewrite_pass`; `node_budget` loop in `ac_simplify_ex` | by T6 a binding bound proves an implementation bug; fixpoint break covered by idempotence gates |
| L6 premises | `stable()` debug assert; fixpoint break | corpus idempotence gate: 0/400 at head |
| ordering exactness | `Rat::cmp_exact` (256-bit) | Euclidean-oracle fuzz (60k pairs, 3 magnitude regimes) + transitivity triples |

## 8. Relation to the published TRS, and what a new proof must contain

| arXiv:2602.08885 | AC core | Status of the transfer |
|---|---|---|
| terms: binary prefix trees | canonical bags (I1–I4) | redefinition required (done here, §2) |
| cancellation procedure outside the TRS | $\mathrm{nf}$ (§3), much larger | same architectural role; L1 replaces the informal argument |
| rule conditions: $\mathrm{Vars}(\rho) \subseteq \mathrm{Vars}(\ell)$, $|\rho| < |\ell|$, non-duplication | G6; G7; (non-duplication measured: 503/503, not required) | **the static proof device does not transfer** — $c$ is non-additive and fires renormalize (§5), so orientation is enforced per instance instead |
| termination: length is a reduction order | T-wf/T6: the pair $(\mu, \mathrm{cmp\_ex})$ is a well-founded strict total order (the former literal-size middle tier is absorbed into $\mu$, §5) | **new proof, done here, unconditional** |
| iteration cap $K = 5$ | `node_budget` | demoted to defense-in-depth: T6 guarantees the fixpoint in finitely many passes |
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
| the rule asset | nothing — G1–G7 re-vet on every load (drops are counted, visible in `ac_rules_info`) |
| the $\mu$ weight table or `cmp_ex` | T-wf (well-foundedness of the new pair — L5's finite-level-set argument must survive, including the numeric-string-leaf clause); G7 re-vets every asset on load; re-run the corpus gates AND the unit-sensitivity grid (P-R3): the *reachable* normal forms change |
| a constructor (`add`/`mul`/`pow`/`fun`) | L1 (check any new self-call has a decreasing measure), I1–I4; the fold needs no descent premise (it is `oriented`-gated at the pass since the $\mu$ ship); then the corpus gates |
| the matcher | the step-relation definition of §4 (binding soundness); L2 is unaffected (orientation checks the result, not the match) |
| `rewrite_pass` control flow | L2 (every state change gated), L3 (no ungated mutation path), memo semantics (§6 L6 caveat) |
| serialization | L6's round-trip premise (`stable()`), the corpus gates |
