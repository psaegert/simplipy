# 7-4 calibration verdict — GO (commit tiers A+C) — 2026-07-11

Ran on **solomon** (Ryzen 9 9950X, 30 rayon threads, nice 5), simplipy `357c795`
(0.5.0 + fold-filter + affine-recall fixes), the real 13-leaf dev_7-4 config
(4 dummies + 9 constant leaves, X=1024, challenges=retries=16, rtol=1e-9/atol=1e-12,
stage-2 confirm on). 12:17:48–12:42:00 UTC = **24 minutes end to end**.
Raw: `calib_out_solomon/CALIB_RESULTS.json` (+ `calibration.log`, `calib_l4_rules.json`).

## Measured (verbatim from CALIB_RESULTS.json)

| quantity | value |
|---|---|
| P1 complete L<=4 mine (fold-filter ON) | **6,849 rules** (35 L2 + 309 L3 + 6,505 L4) in **39.5 s wall** / 969 CPU-s |
| candidate library | 566,280 total -> **192,249 kept** (374,031 var-free dropped) |
| P2 parity (1,000 L4 + 500 L5 sampled, identical fit seeds) | 655 survivors, **0 mismatches -> EXACT** (1,360.7 s wall, dominated by the unfiltered reference scans) |
| P3 cost anchor (2,000 uniform L5 sources) | survivors 778 (38.9%), 22 rules found, **per-source CPU 1.377 s** (per-call wall: median 0.34 s, p90 4.0 s, max 8.9 s) |

## Decision (pre-registered rule from MINE_7_4_READINESS_2026-07-11.md)

`1.377 CPU-s <= ~2` AND `parity EXACT` -> **commit tiers A+C** (complete lengths 2–5 +
saturation-sampled lengths 6–7), tier B (complete length-6) formally "offered".

## Projections from the measured anchors (label: projections, not measurements)

- **Tier A, complete L5** = 21,048,053 sources x 0.389 survivors x 1.377 CPU-s
  ~ 11.3M CPU-s ~ **4.4 wall-days on solomon's 30 threads**. (L<=4 adds ~16 CPU-min.)
- **L5 rule yield**: 22/2,000 sampled -> ~1.1% of the universe -> **~230k L5 rules**
  (Poisson n=22 -> wide 95% band, roughly 150k–330k). The shipped dev_7-3 has 33,922
  L5 rules from its ~2% enumeration coverage; completeness is worth roughly an order
  of magnitude here.
- **Tier C** (sampled L6/L7): same library, so ~1.4 CPU-s/source-survivor scale;
  ~0.5–1.5 wall-days per million sampled sources; the saturation curve governs when
  to stop.
- **Tier B, complete L6** = 830,553,009 x ~0.39 x 1.377 ~ 446M CPU-s ~ 5,200 CPU-days
  ~ **172 days on solomon (81 days even on 64 cores) -> declined as impractical**,
  consistent with Doc A's stretch framing.

## Cross-machine determinism

Solomon and valkyrie smoke runs were byte-identical (344 rules, parity EXACT).
The full-scale replica is queued on valkyrie behind the hybrid-grid timing run;
its P1 ruleset must equal solomon's byte-for-byte (same seeds).

## Stratified feasibility (source-class x length; solomon probe 2026-07-11, N=300/stratum)

User question: exhaust const-free L6/L7 sources, sample const-bearing? Measured answer: the
numbers flip the idea. Exact counts: const-free = 88-92% of every length (12-of-13-leaf DP).
Measured (per-SURVIVOR CPU-s on the production filtered library, pruning with the L<=4 ruleset;
`strata_out/STRATA_RESULTS.json`):

| stratum | N (exact) | survive | CPU-s/surv | exhaustive (solomon, 28 thr) |
|---|---|---|---|---|
| L5 const-free | 19,021,932 | 0.41 | 1.330 | 4.3 days (tier A) |
| L5 const-bearing | 2,026,121 | 0.09 | 1.186 | 0.1 days (tier A) |
| L6 const-free | 742,627,116 | 0.35 | 1.125 | **120.9 days -> NO** |
| L6 const-bearing | 87,925,893 | 0.07 | 1.087 | **2.8 days -> YES** |
| L7 const-free | 29,729,866,428 | 0.33 | 1.035 | 4,197 days -> NO |
| L7 const-bearing | 3,903,016,219 | 0.10 | 0.537 | 86.6 days -> NO |

Key mechanics: per-survivor cost is ~1 CPU-s in EVERY stratum because ~99.5% of any scan is
the const-bearing CANDIDATE arm (LM fits), which soundness forbids dropping (cf-min lemma
FALSE: `7*x0 -> x0/C`). Feasibility is therefore governed by source count x survivor fraction
-- and const-bearing sources are both few (8-12%) and heavily pruned (7-10% survive), making
them the EXHAUSTIBLE stratum at L6. The boundary: everything at L<=5; only const-bearing at
L6; nothing at L7. (Upper bounds: production pruning also carries the ~230k L5 rules ->
survivor fractions drop further.)

The cf-source x cf-candidate cell (the hypothetical cheap sweep): 0.006-0.009 CPU-s/surv ->
L6-cf ~0.6 days, L7-cf ~38 days. Usable ONLY as a stage-1 screen (misses the cb-target-only
family; hits must be re-derived against the full library for certified-minimal targets) --
optional lever, not part of the committed plan.

RAM note (solomon 60 GB): L5 enumeration ~3 GB fine; complete-L6 enumerations exceed RAM
(full ~125 GB, cf ~111 GB) but L6-const-bearing (~13-18 GB) fits, and slice-wise generation
(per root operator/composition) keeps any of them at a few GB.

**Recommended mine shape (upgrade over plain A+C): tier A (complete <=5, ~4.4 d) +
EXHAUSTIVE L6-const-bearing (~+2.8 d, slice-enumerated) + stratified-sampled L6-cf and
L7-both with saturation stop.** Completeness claim becomes: complete through L5, complete
L6-const-bearing, saturation-sampled remainder.

## Remaining pre-launch items (readiness doc items 4–6)

1. **Provenance in the mined artifact** (item 5, the important one): rules.json is a
   bare list; the production mine should record seed, X spec, tolerances, challenges,
   simplipy sha, config into the artifact (or a sidecar).
2. pow-of-(-inf) miner/deployment kernel parity fork (item 4; not exploitable through
   the gate, close for consistency).
3. Count-DP cross-check on the sampler path per run (item 6).
4. RAM check for the L5 enumeration on solomon (~3 GB expected for 21M expressions).
