# Can we make a full 7-4 rule mine feasible? — findings (2026-07-11)

Verdict from a literature scan + adversarial design research (27-agent Workflow) plus direct
verification of the load-bearing claims. Companion to `EQUIVALENCE_AUDIT_2026-07-10.md`.

## Headline

- **Literal-full 7-4 (every source of length <=7) is infeasible** — ~3,100 years on 64 cores and
  ~5 TB to even enumerate. Not closeable by any sound lever in the explored design space.
- **Effectively-full 7-4** — complete at lengths 2-5, saturating-sampled at 6-7 with a measured
  completeness proxy — **is feasible in ~4-8 wall-days on 64 cores** and is the honest target.

## Two corrections to numbers used earlier this session (both verified here)

### Correction 1 — the real universe is 13 leaves, not 5 (~3.8x bigger). VERIFIED.
Earlier session estimates used a 5-leaf universe (`x0..x3` + `<constant>`). The shipped `dev_7-3`
was mined with **13 leaves**: 4 dummy variables + **9 constant leaves**, recovered directly from
the shipped `rules.json` token inventory: `<constant>, 0, 1, (-1), np.e, np.pi, float("inf"),
float("-inf"), float("nan")`. (These 9 come from the mine's `find_rules(extra_internal_terms=...)`
argument, NOT from the engine config, which holds only `operators`+`rules`, and NOT auto-computed —
the engine only auto-adds the 4 dummy vars at engine.py:2656-2658. The research workflow's "13 auto-
computed" attribution was imprecise, but the count is right.) Exact count DP:

| length | 13-leaf (REAL, matches dev_7-3) | 5-leaf (earlier session figure) |
|---|---|---|
| 4 | 550,836 | 192,060 |
| 5 | 21,048,053 | 6,752,605 |
| 6 | 830,553,009 | 241,629,465 |
| 7 | **33,632,882,647** | 8,783,426,095 |
| candidate lib (len<=4) | **566,280** | 197,800 |

Every earlier session wall-clock/universe figure was 5-leaf; multiply sources by ~3.8x and the
candidate library by ~2.9x for the real config. (Whether a NEW mine SHOULD include `inf/nan/e/pi`
as source leaves is a config choice; matching dev_7-3 means yes.)

### Correction 2 — "7-4 ~300 days / 64c" was the SAMPLED plan, not literal-full. VERIFIED.
The ~19,000 CPU-days / ~300-days-on-64c figure quoted earlier was for len 6-7 **sampled at 1M each**
(it is internally consistent: 19,164 CPU-days / 64 = 299 days — the workflow's claim that this is
"inconsistent by 64x" is a workflow ERROR). Literal-full 7-4 is 3.36e10 x ~185 CPU-s/source ≈
**72M CPU-days ≈ 3,100 years / 64c** (5-leaf: ~805 yr). The "300 days" should never have been
attached to the word "full."

## Why literal-full is dead (compute AND memory, independently)

- **Compute**: even at the most optimistic sound per-source cost anyone measured (~1.0 CPU-s/source
  after candidate minimization; a single 5-leaf measurement, unverified here) x Kruskal keep-fraction
  0.675, len-7 alone is ~263,000 CPU-days ≈ 11 years / 64c. The only two source-count reducers that
  could close orders of magnitude — normal-form-only enumeration and raw cvec dedup — were MEASURED
  to collapse the count by only ~1.6x (normal-form fraction ~62% at len-7; the grammar is 33/38 unary
  and length-decreasing rules are sparse, so most long terms are genuine normal forms). That
  measurement is the empirical death certificate.
- **Memory**: `enumerate_expressions` builds a full per-length Python set. Len-7 = ~5 TB (13-leaf),
  len-6 = ~125 GB, len-5 = ~3 GB. Len-5 is the last length that fits without streaming.

## What IS feasible: effectively-full, in tiers

Wall-clock at ~1.0 CPU-s/source (post-minimization, PROTOTYPE-gated), Kruskal x0.675, 64 cores:

| tier | scope | wall-clock (13-leaf) | soundness |
|---|---|---|---|
| A (commit) | complete len 2-5 | ~2.6 wall-days (needs streaming at len-5) | complete, no loss |
| B (offer) | complete len-6, streamed | ~100 wall-days (~3 wk) | complete, no loss |
| C (effectively-full) | sampled len 6-7, saturation-stop | ~1.8-5.4 wall-days | SOUND (sampling >J verified 2026-07-11); reduces completeness only |

**Recommended: A + C ≈ 4-8 wall-days / 64c**, reporting the coupon-collector saturation curve
(marginal new-rule rate vs draws) as the headline completeness number. Every figure needs the
calibration prototype (below); the ~1.0 CPU-s anchor will RISE on the real 566k-candidate library.

## Ranked SOUND levers

1. **Candidate-library minimization — NARROW (collapse-to-bare-`<constant>`) whitelist only.**
   Drop const-bearing candidates whose fixed structural simplify collapses to bare `<constant>`
   (`sin(<constant>)`, `exp(<constant>)`, `pow(<constant>,<constant>)`, `inv(<constant>)`,
   `log(exp(<constant>))` -> `<constant>`; premise verified this session). Sound because bare
   `<constant>` reaches all of R, is length-1, and dominates the wrapper's whole fit family.
   Claimed ~20-40x / one 5-leaf measurement showed ~200x on the nonlinear candidate set.
   ⚠ The GENERAL rule "drop any candidate whose simplify is strictly shorter" is NOT proven sound
   and must not be shipped blind: soundness requires the shorter form's fit family to DOMINATE the
   longer's, which holds for collapse-to-`<constant>` but not necessarily for collapse-to-other.
   (The workflow's specific falsifier, `abs(pow(x0,C))->pow(x0,C)`, did NOT reproduce — simplify
   does not shorten `abs(pow(x0,<constant>))` — so it is not a live counterexample, but it is not a
   proof of general soundness either.) Ship the enumerated safe-collapse whitelist + the
   byte-identical-parity gate below, which catches any family-incomparability bug empirically.
2. **Streaming normal-form enumeration — value is MEMORY, not speed.** A tree-automaton redex
   recognizer (TATA Ch. 3.4) built from the left-linear length-decreasing LHSs streams only
   irreducible sources, never materializing the set — this is what makes len-6 and sampled-len-7
   possible at all (kills the 125 GB / 5 TB blowup). Compute bonus only ~1.6x. Non-left-linear
   length-decreasing rules (`div(_0,_0)->1` etc.) stay caught by the existing per-source Kruskal
   mop-up. Sound (shortest-first + frozen-R).
3. **Iterative length-by-length Kruskal congruence prune** (Ruler's `run_rewrites` structure;
   compound the existing single-shot Kruskal prune up the length ladder). ~1.5x, mostly built.
4. **Above-target sampling with saturation-stopping** — the engine of Tier C; verified sound
   2026-07-11 (never a wrong/non-minimal rule).

## Six FALSIFIED levers (the design-space-exhaustion proof)

Each empirically killed this session — this is what makes the "no" to literal-full definitive:
- **Frozen fingerprint/invariant index as an accept gate** — false-negatives on domain-extension
  rows; the gate binds on the source-specific finite rows S, unknown at index-build time.
- **cvec bucketing as a reject prefilter** — true rules (`div(x0,x0)->1`, `log(exp(x0))->x0`, ...)
  have source/target finite-mask differences of 11-780 rows; no fixed key groups them.
- **VarPro (variable projection)** — 0 applicable candidates in the len<=4 library; separating a
  linear-in-C part needs >=6 tokens and 100% of nonlinear-with-variable candidates carry one
  constant. A no-op at 7-4.
- **1-D deterministic global sweep replacing LM restarts** — the `sin(C*x)` accepting basin is
  ~3e-3 wide and the gate needs the constant to <1e-8; a coarse sweep drops `sin(3x)` (a rule the
  16-restart LM finds); the sweep that recovers it is more expensive than the LM.
- **Subset-rows fit prefilter** — "no fit on subset" != "no fit exists": LM is a heuristic not a
  decision procedure; 96/96 subset-rejects flipped to ACCEPT at 256 restarts.
- **Raw observational-equivalence / normal-form collapse as a COMPUTE lever** — measured 1.6x, not
  the hoped 2-3 orders.

## Key literature

- **Ruler** (Nandi et al., OOPSLA 2021, arXiv:2108.10436) — the blueprint: `run_rewrites` = the
  sound generalization of Kruskal-prune; cvec-match with null=don't-care is the reject-only,
  per-source-recomputed pattern. Its No-RR ablation (>24h -> 350s) shows enumeration-modulo-
  equivalence is the feasibility line — but our measured 1.6x collapse (vs their orders of magnitude
  on Boolean/bitvector domains) is why literal-full still dies for rationals/transcendentals.
- **Enumo** (Pal et al., OOPSLA 2023) — theory-exploration scheduling on top of Ruler.
- **egg** (Willsey et al., POPL 2021, arXiv:2004.03082) — the e-graph + e-class-analysis backbone.
- **TATA** (Comon et al., Ch. 3.4) — constructive backbone for streaming NF enumeration.
- **EUSolver** (Alur et al., TACAS 2017) / **Probe** (Barke et al., OOPSLA 2020) — observational-
  equivalence dedup with a minimality theorem.
- **Golub & Pereyra VarPro** (2003) / SAGE-Fit (arXiv:2605.23272) — why VarPro is a no-op here.

## Go/no-go prototype (the single calibration)

Build lever 1 (narrow-whitelist minimization) + lever 2's streaming NF sampler, run a complete
len-4 AND len-5 calibration mine on the REAL 13-leaf `dev_7-4` config, measuring exactly:
1. **Per-source CPU-s** on the real 566,280-candidate library (the ~1.0 s anchor WILL rise from the
   5-leaf 197,800-lib measurement; every wall-clock above scales linearly with it).
2. **Rule-set parity**: on N random len-4/5 sources, diff minimized-library output vs the
   unminimized reference miner. Zero divergence certifies the whitelist sound in practice (guards
   the family-incomparability class).

Decision rule: per-source <= ~2 CPU-s AND exact parity -> commit A+C (~4-8 wall-days), offer B;
per-source > ~5 CPU-s -> drop B, run A + sampled-6/7 only.
