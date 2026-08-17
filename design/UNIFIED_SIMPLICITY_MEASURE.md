# The unified simplicity measure: description length as the one objective

> **2026-08-08 amendment note (contract §10.11):** "mask × special stays unabsorbed" is
> SUPERSEDED — the owner confirmed the deferred leaning-ruling the other way. Specials
> now absorb into an existing `<constant>` at the bijective bag sites (Add-shift /
> Mul-scale), exactly like every other finite ground; introduced masks and non-bag
> positions (`pi^C`) still refuse. The stage-1/stage-2 "stays unabsorbed" language
> below is kept as the historical record.

**Status: STAGE 2 IMPLEMENTED 2026-08-01** — μ is the ordering (mu + canonical order,
lit_size absorbed), the fold is μ-governed, the resolution licence is deleted (a
theorem now, on every channel incl. the hint channel stage 1 left open), the
special-READ refusal + Const-absorption licence + determined-source clause stay
(policy/contract, not theorems). Measured same day: T1–T7 battery 15/15; 454 pytest +
109 cargo green with every pin re-earned; reference-gate translation IDENTICAL 466/32/39
(total re-pinned 97911 μ-units); 64k/1M gates 0/0/0 with the F64_FOLD verdict class
EMPTY at both scales (finding (b) dissolved, measured); ECDF mean 50µs == baseline
(after a 2.4× alloc regression was fixed with non-allocating leaf reads); canonical
re-mine 1942 rules (+210/−21 vs stage-1 1753: value-set over newly-symbolic factors +
exact special quotients `/ acos 0 np.pi → 0.5`, `* _0 acos (-1) → * np.pi _0`; losses
= compositional subsumptions + one sort upgrade). Implementation deltas from this
doc, both flagged: (1) beyond-`Rat` literals (string leaves, q > 2^128) pay a
description length parsed from the canonical print (`mu_numeric_str`) — the doc's
table only covered `Rat`-range values; (2) the negation WRAPPER (`x − y` =
Add[x, Mul[−1, y]]) prices its structural node (+8) — Σ_nodes taken literally; sign
stays free in literal atoms and coefficient/exponent slots. P-R3 MEASURED (2026-08-01, both
canonical mines at HEAD): the 1/8 artifact (1942) is a STRICT SUBSET of the 1/16
artifact (1947) — zero reorderings, and the 5 extra 1/16 rules are all POLICY
VIOLATIONS: special-bearing ground states materializing to ~50-bit roundings
(`abs - np.e np.pi -> 0.423310825130748`, `- np.pi atan float("-inf") ->
4.71238898038469` = 3π/2 rounded, `- np.e exp np.e`, `- np.e acosh np.e`,
`- np.pi asinh np.pi`). Mechanism: a minimal special state is ~5 symbols; at
8 bits/symbol the mark (40) refuses a ~50-bit literal, at 16 (80) the literal
descends. The unit window therefore has a MEASURED policy ceiling — symbol cost
must stay below ~10 bits/symbol or minimal special states lose to their own f64
prints — and 1/8 sits inside it, 1/16 outside. Not the spec's "windows were
wrong" stop-clause: rule membership moved only ACROSS the policy boundary, never
within legitimate content, converting the 1/8 pin from a choice into a bounded
one. c_free scaled as 16x the symbol in the run (`SIMPLIPY_MU_SYM`, read-once
env, production never sets it). **STAGE 1 SHIPPED 2026-08-01** (commits `0c1160b` licences,
`0ea5258` divisor-side emitter) — all owner decisions in: fold-by-μ (§4 P-R2, option
a), mask×special stays unabsorbed, c_free = 128 accepted as initial pin,
stage-1-first, divisor-side spelling ratified as a user-facing aesthetics mechanism.
§4 predictions measured — see the dated blocks inline. Internal document (excluded
from sdists); the public docs get a distilled version if and when this ships.

## 1. Where this comes from

The 2026-07-31 special-constant policy discussion surfaced a fact the codebase had been
dancing around: the engine optimizes *node count*, but the owner's rulings optimize
something else. The moment that made it visible: `exp(π·x)` is PREFERRED over
`pow(23.14069263277927, x)` — a deliberate refusal of a node-count reduction, the first
in the project's history. The hidden second objective has a name: **description length
under an exactness-respecting cost model**. π is one symbol with zero degrees of freedom;
`23.14069263277927` is a ~52-bit object; `<constant>` is a full real degree of freedom.

The slogan form: **simplify is lossless compression.** Rounding an exact object into bits
is lossy, and lossy steps belong to the fitter and to the (planned) post-fit
quantization/rationalization stage — never to simplify.

Ratified inputs this document builds on (owner, 2026-07-31):

- special ∘ special stays symbolic (`π·e`, `e^π`, `exp(π·x)` — no materialization);
- integer × special and rational × special stay symbolic;
- `f(special)` collapses only to EXACT values (`sin π → 0`, `cos π → −1`, `log e → 1`),
  and these ship as MINED certified rules, not predefined tables;
- nan/±inf absorption stays as-is (already exact, constructor-level);
- **`cost(<constant>) ≥ cost(special)`** — the entropy argument: a free degree of
  freedom generates higher-entropy data than a fixed named constant;
- masking: `mask_all` masks specials too (landed, `a30204a`); flash-ansr training masks
  all constants, so specials are not in the fa vocabulary;
- mask × special: leaning-ruling "stays unabsorbed" in generic simplipy use (the
  post-fit rationalizer may want the structure) — confirm before stage 1 lands.

## 2. The measure, formally

Let `e` be a canonical expression (the AC form). Define

    μ(e) = Σ_nodes c_struct(node) + Σ_atoms c_atom(atom)

over the canonical tree, with costs in units of 1/8 (scaled integers, so all arithmetic
stays exact and the order is discrete — no float comparisons anywhere in the ordering).

| atom / node | cost (units of 1/8) | rationale |
|---|---|---|
| structural node (bag, Pow, Fun head) | 8 | one grammar symbol |
| variable leaf | 8 | one vocabulary symbol |
| special constant (π, e) | 8 | one vocabulary symbol, zero dof |
| numeric literal q = p/q′ (exact value, spelling-independent) | max(2, bits(p) + bits(q′)) | its description length; small ints ≈ 2–4, a 52-bit dyadic ≈ 105 |
| bag COEFFICIENT slot (the Rat riding a Mul/Add) | the literal cost above, NOT free | removes the current coefficient-rides-free asymmetry |
| `<constant>` | c_free = 128 → **1133 (amended 2026-08-06: DERIVED, not floored** — sup over f64 round-trip spellings = 1131.931 bits at 5.5605781537525765e-308, + 1 sign bit, ceiled; the 128 floor was beaten 8× by μ(1e308) = 1024.154)** | a free real dof; the priciest atom by construction |

Two structural decisions worth pinning:

1. **Literal cost is a function of the VALUE, not the token spelling** — `0.5` and `1/2`
   cost the same; the exact rational behind a decimal spelling is what gets measured.
2. **The coefficient slot pays.** Today `ac_complexity(Mul[2.718…, x]) = 2` <
   `ac_complexity(Mul[E, x]) = 3` because the Rat coefficient rides free — this single
   asymmetry is what minted the 6 coefficient-materialization rules through the
   spelling-dependent skip. Under μ the coefficient pays its description length.

The reduction ordering becomes (μ, then the existing cmp tiers with symbolic-first). The
`lit_size` tie tier is absorbed into μ's first component — one fewer ordering layer.

## 3. What the measure derives (the point of it)

Each policy row becomes an inequality instead of a licence. With the table above
(units of 1/8; op/leaf = 8):

- **T1, coefficient preservation**: μ(`Mul[E, x]`) = 8+8+8 = 24 < μ(`Mul[fl(e), x]`)
  ≈ 8+105+8 = 121. Materializing a special coefficient is a steep ASCENT. Kills the
  6-rule fork *by arithmetic* — the spelling-dependent skip loses its sting because the
  candidate is no longer below any mark, from any spelling.
- **T2, `exp(π·x)` over `pow(fl(e^π), x)`**: 8+8+8+8 = 32 vs 8+105+8 = 121. The owner's
  preference is the measure's theorem.
- **T3, exact collapses still fire**: μ(`sin π`) = 16 > μ(`0`) = 2. `sin π → 0` is a
  descent; so are `cos π → −1` and `log e → 1`. Exactness is not blocked — only
  *rounding* is.
- **T4, ordinary algebra unharmed**: `x + x → 2x`: 8+8+8 = 24 > 8+2+8 = 18 ✓ descent.
  `x·x → x²`: 24 > 8+8+2 = 18 ✓. Collection and folding of small integers keep working
  because small integers are genuinely cheap. **Constraint this imposes**: for every
  collection `n·x` minted from n-fold sums, need cost(literal n) < cost(variable), i.e.
  bits-cost of alphabet integers < 8 ✓ (≤ 4 for |n| ≤ 8).
- **T5, real × special resolves itself**: `2.5·π` (8+4+8 = 20) stays symbolic (vs
  literal ≈ 105); `2.5069314582…·π` (8+105+8) vs one materialized literal (≈105) —
  materializing WINS for long literals. The rational-vs-real boundary the type system
  cannot express becomes exactly the point where the cost comparison flips.
- **T6, the F3 count invariant aligns**: `<constant>` is the priciest atom, so any
  rewrite that removes a Const descends and any that manufactures one ascends. The count
  gate stays ENFORCED (it is a hard invariant, not a preference), but the measure now
  pushes in the same direction instead of against it.
- **T7, termination posture improves**: the audit's O1 concern was DENSE literal tiers
  (`Mul[3/2^k, x]` for all k sat at one complexity level; a hang was reproduced). Under
  μ, cost grows with bits(2^k) — the chain ASCENDS and cannot be a reduction sequence.
  Levels are discrete (1/8-integer), bounded below → well-founded per level.

## 4. Pre-registered predictions

To be checked against measurement when implemented, in the H6 style (predict first,
measure second, explain every miss).

**Benefits (P-B):**

1. The 6 coefficient-materialization rules become unmintable from every spelling; the
   stage-1 mint/fold licences become theorems of μ and can be DELETED.
2. At the re-mine, the F3-era literal-resolution family dies except exact hits:
   predicted dead: `exp(πx)/log(π^x)/π^(−x)` materializations and all 35 sign/inverse
   materializations; predicted alive: exact collapses (`sin π → 0` class) and every
   Const-free structural rule. Artifact moves toward pure structure.
3. `sin np.pi → 1.2246e-16` disappears as a *policy* consequence (the fold licence /
   measure refuses inexact materialization) — no 1024-bit fold machinery needed; the
   hiprec-fold queue item shrinks to a numerical-QA tool.
4. Ordering simplification: lit_size tier absorbed; one objective explains
   preservation, collection, masking costs, and the count-invariant direction.
5. Termination: the dense-literal chain class provably ascends (T7).

**Risks / downsides (P-R):**

1. **Global ordering shift.** Every serve decision, mint mark, prune verdict and
   select_best comparison re-judges under μ. Prediction: most corpus outputs unchanged;
   equal-complexity laterals may flip orientation (quantify by row-diff buckets, same
   protocol as 7-31). All gate pins (10140-era complexity numbers) become
   incommensurable and must be re-earned; cross-measure comparison happens ONLY via
   row-level diff classes, never by comparing totals across measures.
2. **The fold tension — RESOLVED (owner 2026-07-31: option (a); "then sin(-1) should
   stay as is. It makes sense under the MDL").** The fold is governed by μ: it fires
   exactly when it descends. All EXACT arithmetic keeps folding (`2+3 → 5`,
   `cos 0 → 1`, cancellations); transcendental roundings (`sin(-1) → 0.841…`) are
   refused — no special case needed, it is the same inequality as everything else.
   Behavior change on numeric grounds is accepted; deployment is unaffected (fa
   evaluates numerically at runtime).
3. **Parameter sensitivity.** c_free = 128 and the 1/8 unit are choices inside derived
   windows, not derived values. Pre-registered check: mine at unit 1/8 AND 1/16; the
   artifact diff must be empty or fully explained. If rule membership is sensitive to
   the unit, the windows in §3 were wrong — stop and re-derive.
4. **Plumbing.** bits() per literal per comparison — memoize per Rat; scaled-integer
   overflow audited (i64 headroom: 2^63 / 8 units ≫ any tree). Prediction: ≤ 5%
   latency on the 400-row ECDF, restored by memoization if exceeded.
5. **Behavioral regressions vs today** on rows where materialized forms were shorter:
   the sign-materialization rows print 4 tokens instead of 3 (`neg * np.pi x0`).
   Accepted cost of the ratified policy; quantify the exact row counts at 64k/1M.
6. **Const-cost coupling.** c_free = 128 > any single-literal cost means a rewrite
   trading one Const for any one literal descends — consistent with resolution; but
   audit every Const-bearing serve rule under μ for direction flips (predicted: none,
   since Const-count is gated independently).

### Stage-1 measurement (2026-08-01, licence forms — the μ-form re-check happens at stage 2)

- **P-B1 HIT.** The 6 coefficient-materialization rules are unmintable from every
  spelling (resolution licence refuses the source, not the spelling); the fresh mine
  contains zero. The pre-registered "state-level Kruskal-skip fix" DISSOLVED on
  inspection: `ac_ordered_below` already canons both sides (state-level); the fork's
  actual channel was `accept_resolved` + the coefficient-rides-free asymmetry, and the
  licence closes it channel-wide.
- **P-B2 HIT with one instructive miss — miss RESOLVED same day (owner: "cos(pi)
  should be simplified to -1").** Verification re-mine 617 → 585: literal
  materializations 45 → 0 (`exp(pi*x) -> pow(23.14069263277927, x)` dead in all four
  spellings; `log(pi^x)`, `pi^(-x)`, all sign/inverse materializations dead);
  special-absorbing Const rules 8 → 0. Predicted-alive exact collapses arrived first
  as 21 compositional rules (`* _0 cos np.pi -> neg _0`, `+ _0 sin np.pi -> _0`,
  `pow _0 log np.e -> _0`) while the BARE ground collapses (`cos np.pi -> (-1)`)
  did not mint. The first root-cause attribution (dirac cannot certify transcendental
  exactness) was WRONG: a gate-trace showed the mint and dirac both PASSED — hiprec
  residuals of genuinely exact identities underflow the working precision (cos of
  1024-bit π is −1 + ~2^-2049, correctly rounded to −1 exactly), so the numeric layer
  certifies them fine. The real refusers were TWO stacked post-mine stages. (1)
  Stage-2 CONFIRMATION re-runs `find_rule` with the target as the only candidate, and
  the var-free short-circuit's `Class::Finite -> <constant>` arm hijacked the answer
  for special-bearing sources (returning `<constant>` instead of testing `(-1)`),
  which the result-must-equal-target guard rightly scored unconfirmed. Fix: that arm
  is now gated on special-free sources — the Const-absorption licence applied at the
  channel (a special vanishing into a fitted constant is forbidden for EVERY caller).
  (2) The sort-promotion ladder's ground tier judges exactly the
  one-`<constant>`-slot skeleton family; zero-slot pure-literal rules scored
  UNSUPPORTED (a scoping refusal, not a verdict) and silently vanished — this is why
  NO artifact has ever carried a length-2 LHS rule. Fix: special-bearing pure-literal
  ground rules bypass promotion (no wildcard to carry a sort; one instantiation,
  certified at mint + confirmation + the hiprec arbiter). Bare collapses now mint,
  confirm, survive the post-pass, and serve (`cos np.pi -> -1`); inexact values still
  cannot (no alphabet literal matches, resolution refuses, hiprec refutes
  near-misses).

  **Two findings surfaced by the promotion audit — (a) FIXED 2026-08-01, (b) is a
  stage-2 item:**
  (a) **FIXED at the mint (2026-08-01).** The UNSUPPORTED drop had been silently
  containing SPECIAL-FREE pure-literal ground mint-then-drop rules, among them
  f64-respells (`+ 1 exp (-1) -> 1.3678794411714423` maps the engine's exact
  `1.36787944117144233` to its f64 rounding) admitted through the lit_size tie tier
  of the evaluate short-circuit's ordering acceptance — the same disease
  `accept_resolved`'s strict-complexity guard closes for the resolution channel. The
  short-circuit now honors `accept_resolved` (an f64-evaluated literal is a COMPUTED
  value); the fix unmasked the interval-class `Finite -> <constant>` arm answering
  for DETERMINED sources (var-free, Const-free — outside its documented contract),
  which was closed the same day with the determined-source clause: such a source
  never accepts a Const-introducing target, at the arm and in the mining acceptance
  closures alike. Canonical re-mine: rules.json BYTE-IDENTICAL 1753; mined proposals
  16,506 -> 4,262 (stage-2 confirmation had been eating a 12k determined->Const
  flood per mine, pre-dating the guard); stage-2 rejections 12,394 -> 360. Under μ
  the family additionally dies by arithmetic (transcendental folds refuse, marks
  stay symbolic and cheap, and a literal's value-cost exceeds them), so the strict
  tie-refusal becomes a theorem — but the determined-source clause is CONTRACT, not
  licence (the miner agreeing with the engine's own semantics), and stays.
  (b) The symbol/literal seam: rules like `- np.e exp 1 -> 0` are hiprec-EXACT under
  the symbolic reading (exp(1) IS e) but serve on the canon state
  `e - 2.718281828459045` (the stage-1-retained special-free fold materializes
  exp(1)), where the literal reading differs by ~4.4e-17 — the F64_FOLD verdict
  class, deployment-tolerated (fa evaluates at f64; the pre-policy engine equated the
  two spellings outright by materializing both). Stage 2's μ-governed fold dissolves
  the seam: exp(1) stays symbolic and the exact rule `exp 1 -> np.e` becomes
  mintable.
- **P-B3 HIT.** `sin np.pi -> 1.2246e-16` gone as a policy consequence; the hiprec-fold
  queue item shrinks to a numerical-QA tool as predicted.
- **P-R5 QUANTIFIED, far below the worst case.** 1M corpus: 495 special-bearing rows,
  13 changed, delta histogram {+1: 10, +2: 3}, total +16 complexity (gate cross-check:
  complexity_out 25,282,581 → 25,282,597, exactly +16). 64k corpus has zero
  special-bearing rows (policy delta structurally 0 there). Every changed row is the
  ratified preservation (`13.591... -> 5*e`, `1.359... -> 0.5*e`, three `C - e`
  absorption refusals).
- **Gates.** the acj-4-3 400-row reference gate VERBATIM (rows=400 idem=0 perm=0 complexity=10140
  translation=466/32/39 twins=1 — twice: after the licences, after the emitter);
  the 64k scale gate (stage 1) identical to the abs baseline on every aggregate (345 flags =
  343 OK + 2 F64_FOLD); stage1-1M identical except the +16 (1710 flags = 1709 OK +
  1 F64_FOLD, judge_non_ok empty, disagree histogram identical). Latency ECDF
  p50=44us p90=82us p99=160us mean=50us max=252us vs the 52us shipped baseline — no
  regression (P-R4 moot for stage 1).
- **§8 emitter.** 6,857 respelled rows at 64k / 102,096 at 1M, every one the same
  canonical state (old output re-simplifies to the new spelling), state complexity
  delta identically 0 — the emission-only guardrail held by measurement, not just by
  construction. One doctrine simplification fell out: the `-1/3` neg-wrapper special
  case is gone (the sign rides the relocated divisor literal; pretty infix still leads
  with the sign, `-x0/3`).

## 5. Staging

- **Stage 1 — licences (no measure change, small blast radius):** ground-fold licence
  (fold only special-free compounds), resolution licence (no inexact literals for
  special-bearing states), state-level fix for the spelling-dependent Kruskal skip,
  mask×special non-absorption (pending the owner's confirmation of "stays"). Full
  ladder + re-mine + row-diff quantification. Behavior becomes policy-correct
  immediately; the licences are explicitly marked as future theorems of μ.
- **Stage 2 — the measure:** implement μ, delete the stage-1 licences it subsumes,
  re-earn every pin, double re-mine, sensitivity run (P-R3), termination probes
  (`Mul[3/2^k, x]` chain), full falsifier battery with T1–T7 as red-first tests.

Stage 2 is foundational surgery on the load-bearing ordering; it does not ship without
the full battery, and stage 1 does not wait for it.

## 6. Open questions for the owner — ALL RESOLVED 2026-08-01

1. ~~Fold governance under μ~~ — **RESOLVED 2026-07-31: option (a), fold governed by
   the measure; `sin(-1)` stays symbolic** (§4 P-R2).
2. ~~Mask × special~~ — **RESOLVED: stays unabsorbed** ("2 ok"). Implemented in stage 1;
   one refinement discovered at implementation: `e^C -> C` still collapses, because the
   exact identity `pow(e,t) == exp(t)` eliminates the special BEFORE any mask meets it
   and `exp(C) -> C` is ordinary special-free absorption — the licence protects specials
   from rounding and absorption, not from exact identities. Pinned in
   tests/test_special_constants.py.
3. ~~c_free = 128~~ — **RESOLVED: accepted as the initial pin**, subject to the §4 P-R3
   sensitivity run at stage 2.
4. ~~Staging~~ — **RESOLVED: stage 1 first** (owner delegated; cleaner attribution — the
   corpus is policy-correct before the measure lands, so the stage-2 row diff isolates
   the measure change alone). Stage 1 shipped 2026-08-01.

## 7. Literal spellings in output (owner question, answered 2026-07-31)

"Will simplified expressions contain `0.333…`s or `1/3`s?" — **both, and never
interchangeably, because they are different numbers.** Every literal token denotes an
exact rational (`0.3333333333333333` = 3333333333333333/10^16 ≠ 1/3), each VALUE has
one canonical spelling (the shortest exact one: finite-decimal rationals print as
decimals — `0.5`, `0.2`, `0.3333333333333333` round-trips as itself; infinite-decimal
rationals print as p/q — `1/3`, and `1/0.333…` prints as the honest
`10000000000000000/3333333333333333`), and simplify NEVER converts between values —
that would be lossy. The decimal→nice-fraction step (`0.3333… → 1/3`, `3.14159… → π`)
is the post-fit quantization/rationalization stage: explicit, tolerance-parameterized,
optional, downstream. μ prices the honest third cheap and the fake third expensive,
which is correct (different description lengths) and rewards pipelines that
rationalize early — without ever letting simplify falsify a value.

## 8. Divisor-side spelling (proposal, 2026-07-31 — owner aesthetic finding)

`x0 / 0.3333333333333333` prints as `<mul> 10000000000000000/3333333333333333 x0
</mul>`. The value IS the owner's intuited `3.000…003` — as the infinite repeating
decimal 3.(0000000000000003), which no finite digit string spells exactly; the p/q
fraction is its only finite exact spelling, and truncating or f64-rounding it would be
a silent value change (the quantizer's move, never simplify's). The canonical fold
itself is correct: reciprocals of rationals are exact and drop a node.

The fix lives one layer up: the STATE and its SPELLING are different layers, and the
tagged form already has a `<div>` section. Proposal: the emitter spells a rational
coefficient on whichever side has the SHORTER exact spelling —
`<mul> x0 <div> 0.3333333333333333 </mul>` here (the reciprocal has no finite decimal;
the original does). Same state, same value, cosmetic only. Guardrail: prettiness lives
in EMISSION ONLY — spelling must never enter μ or any mint/skip decision
(spelling-dependence is exactly the Kruskal-skip disease diagnosed 7-31). Scope note:
flash-ansr never hits this path (fitted constants substitute into masked coefficient
slots; quantization precedes re-simplification) — it matters for direct library input.
Needs: emitter tie-break rules (prefer coefficient side on ties), parse-closure check
(the parser already reads `<div>` sections), corpus spelling re-pins, masking role
check (`Role.COEFFICIENT` already covers `/` operands).

**RATIFIED + SHIPPED 2026-08-01** (owner: "we handle this over user-facing aesthetics
mechanisms"; commit `0ea5258`) — one `divisor_side` helper drives all three emitters
(tagged, explicit, infix), guarded on a plain numerator factor remaining; the sign
rides the relocated literal (retiring the `-1/3` neg-wrapper special case). The
owner's MDL confusion that preceded ratification is worth keeping: a decimal literal
IS a fraction of two big integers in disguise (`3.0000000000000003` =
30000000000000003/10^16, ~109 units), the monster fraction is ~106 — MDL is symmetric
under reciprocal, which is exactly why the spelling side is free to choose.
Measurement in §4's stage-1 block.
