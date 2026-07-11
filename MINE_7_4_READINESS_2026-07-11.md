# 7-4 re-mine readiness — consolidated overview (2026-07-11)

> **STATUS UPDATE 2026-07-11 (later the same day): BLOCKERS 1 and 2 are RESOLVED, then
> ADVERSARIALLY VERIFIED and hardened** (see the Unreleased CHANGELOG entry; cargo 35/35,
> pytest 253/253 incl. an end-to-end filtered-vs-unfiltered mine parity test). The verification
> round (5 refutation lenses + cross-checks) REFUTED the first dominance argument (least-squares
> mean vs interval-intersection gate at the band edge) and found a weighted-design overflow
> false-reject; both fixed (exact bare-`<constant>` feasibility decision; scale-before-weight +
> capped weights + near-integer snap), plus the >=33-variable filter defect and the
> min_informative=0 floor. Full detail in the CHANGELOG.
> - **BLOCKER 1 (fold-filter)**: built as the provable generalization of the whitelist -- ALL
>   variable-free candidates of length >= 2 are dropped (gated on the bare `<constant>` candidate
>   being present; inert otherwise). On the real 13-leaf config the filter drops 374,031 of the
>   566,280 len<=4 candidates (66%); hard const-free non-reducing probe sources (len 5-8) measure
>   **374-474 s/source unfiltered -> 5.0-8.5 s/source filtered (55-88x)** at identical decisions.
>   Fit seeds are now order-independent (pure function of source seed + candidate tokens +
>   instance), making the parity gate exact.
> - **BLOCKER 2 (affine growing-basis recall)**: root cause was NOT QR conditioning but row
>   weighting -- rows with |y| ~ 1e21 carry f64 rounding noise (eps*|y|) larger than an O(1)
>   intercept, so ANY unweighted solve buries it. Fixed with a row-weighted LS (weights mirror
>   the relative accept gate: 1/(atol + rtol*|y_r|), pre-scaled columns, capped weights) + 2
>   rounds of iterative refinement on the retained QR factor + a near-integer snap re-gate.
>   Recall 0/4 -> 4/4 on C0*f(x)+C1 and f(x)+C1 for f in exp/cosh/sinh/pow3/pow4/pow5; the
>   adversarial cosh/sinh cancellation slate went old 30/40 -> new 40/40 with zero regression
>   flips; accept gate unchanged; negatives still reject.
> - **BLOCKER 3 (calibration) DONE 2026-07-11 on solomon -- VERDICT: GO, COMMIT TIERS A+C.**
>   24 minutes end-to-end at simplipy `357c795` on the real 13-leaf config: complete L<=4 mine
>   = 6,849 rules in 39.5 s wall (969 CPU-s); sampled parity (1,000 L4 + 500 L5, identical fit
>   seeds) = 655 survivors, ZERO mismatches; cost anchor (2,000 uniform L5) = survivors 38.9%,
>   **1.377 CPU-s/source** (median wall 0.34 s, p90 4.0 s, max 8.9 s). Rule: 1.377 <= ~2 AND
>   parity EXACT -> commit A+C. Tier B (complete L6) DECLINED as impractical (~446M CPU-s ~
>   172 days on solomon's 30 threads, ~81 days even at 64 cores). Projections: complete L5
>   ~4.4 wall-days on solomon; ~230k L5 rules (n=22 Poisson, roughly 150k-330k) vs the shipped
>   dev_7-3's 33,922. Full record: fa-lab
>   `experimental/simplipy_offline_miner/calibration_74/CALIB_VERDICT.md` (+ raw JSON).
>   Remaining pre-launch: artifact PROVENANCE (item 5), pow-of-(-inf) kernel parity (item 4),
>   sampler count-DP cross-check (item 6), L5-enumeration RAM check on solomon.

Fuses three parallel work streams from 2026-07-10/11 and re-verifies the load-bearing claims of
each against the installed 0.5.0 core (HEAD `479c429`). Supersedes none of them; reconciles all:

1. **Checker / mine-readiness review** — 8-dimension soundness review of the 0.5.0 hardening
   (this session's Workflow; finder + adversarial verifier per dimension).
2. **`MINE_7_4_FEASIBILITY_2026-07-11.md`** (Doc A, simplipy) — 27-agent feasibility + literature scan.
3. **`research/experimental/simplipy_offline_miner/74_FEASIBILITY_2026-07-11.md`** (Doc B) — 9-agent
   feasibility; cf-min lemma + fold-filter findings.

## Headline

- The **equivalence checker is sound** and empirically closes every audited false-accept class. The
  fix is AVAILABLE but not DEPLOYED by design: the `dev_7-3` asset is unchanged and deploys only when
  the re-mine ships.
- **Literal-full 7-4 is dead** (millennia / TB-scale, independently agreed). The honest target is
  **effectively-full: complete at low lengths + source-sampled at 6-7 with a saturation-completeness
  proxy.**
- **Three things block launching that mine**, below. None is a checker-soundness defect; they are a
  cost prerequisite, a recall gap, and a calibration gate.

## Verified reconciliation (three buckets)

### AGREED across sources (and independently confirmed here)
- **Checker soundness.** All five audited false-accept classes reject on the 0.5.0 core
  (`asin(cosh)->nan`, `pow(-1)->nan`, saturation `->1`, linearization `sin(x/K)->x/K`, signed-zero
  div); true rules keep recall; enumeration is provably complete (DP == count-DP == brute force); the
  universe sampler is exactly uniform; the mine is byte-reproducible across `PYTHONHASHSEED` and
  thread count.
- **Literal-full 7-4 infeasible; effectively-full is the deliverable.** Complete low lengths +
  sampled 6-7. Sample **sources**, never **targets** (target-sampling breaks the shortest-first
  minimality certificate and is unnecessary — the target/candidate universe is small and enumerable).
- **Design space is exhausted for a "make full feasible" trick** (Doc A's 6 falsified levers;
  Doc B's excluded levers). Observational-equivalence / normal-form collapse buys only ~1.6x on this
  grammar (33/38 unary, sparse length-decreasing rules), not the orders of magnitude it gives on
  Boolean/bitvector domains (Ruler).

### CONTRADICTIONS — resolved here by independent verification
- **Universe size: 13 leaves (Doc A) vs 5 leaves (Doc B). RESOLVED: 13 is correct.** Recovered the
  shipped mine's leaf inventory directly from `dev_7-3/rules.json`: 4 dummy vars + **9 constant
  leaves** (`<constant>, 0, 1, (-1), np.e, np.pi, float("inf"), float("-inf"), float("nan")`, from
  `find_rules(extra_internal_terms=...)`). Every Doc B wall-clock/universe figure is 5-leaf and must
  be scaled: sources x~3.8, candidate library x~2.9. Corrected count DP: L4=550,836; L5=21,048,053;
  L6=830,553,009; L7=33,632,882,647; candidate lib (len<=4)=566,280.
- **"Const-free is the cheap half" (early intuition) vs "the expensive half" (Doc B). RESOLVED: Doc B
  is correct.** A const-free source's minimal <=4 target is NOT always const-free (cf-min lemma
  FALSE, e.g. `7*x0 -> x0/C`, `x0/x0 -> <constant>`), so a const-free source must scan the
  const-bearing (LM) candidate arm. The "const-free-only allclose sweep" is unsound (silently loses
  the constant-collapse + non-ladder scale/affine/power families).

### OPEN TENSION — needs the calibration prototype to close
- **Complete-L6 cost: Doc A "~100 wall-days streamed" vs Doc B "~5 node-days/128c after R1".** The
  gap is leaf count (13 vs 5) x core count x whether the candidate filter is wired. Both agree
  complete-L6 is the *stretch* and complete-L5 + sampled-L6/7 is the *base*. The real number needs
  the L4+L5 calibration mine on the real 13-leaf config (see BLOCKER 3).

## Feasibility envelope (corrected, 13-leaf)

| tier | scope | soundness |
|---|---|---|
| A (commit) | complete len 2-5 | complete, no loss (needs streaming at L5) |
| B (offer) | complete len-6, streamed | complete, no loss |
| C (effectively-full) | source-sampled len 6-7, saturation-stop | SOUND (source-sampling above the candidate range verified 2026-07-11) |

Recommended target **A + C**; report the coupon-collector saturation curve as the completeness
headline. Every wall-clock figure is gated on the calibration prototype — the current unfiltered
per-source cost is ~130-200 CPU-s (measured), far above Doc A's ~1s post-filter anchor.

## Prioritized gap-closing list

### BLOCKERS — before the mine can launch at a feasible cost / acceptable quality
1. **Wire the candidate fold-filter** (Doc A lever 1 / Doc B R1). CONFIRMED not present in the mine
   path (`engine.py:2468-2475` feeds every enumerated candidate into `build_candidate_library`;
   no collapse filter). ~40-200x per-source speedup; without it effectively-full is out of reach.
   Soundness scope: ship only the *narrow* collapse-to-bare-`<constant>` whitelist
   (`sin(<c>)`, `exp(<c>)`, `pow(<c>,<c>)`, `inv(<c>)`, `log(exp(<c>))`), NOT the general
   "drop any candidate whose simplify is shorter" (family-incomparability not proven). Guard with the
   byte-identical parity gate below.
2. **Fix the affine growing-basis recall gap** (checker review, MAJOR). The Householder-QR affine
   fit rejects `C0*f(x)+C1` for fast-growing `f` (exp, cosh, sinh, pow3/4/5) even for exactly-true
   constants (verified: `C0*exp(x0)+C1` 0/4, `C0*pow3(x0)+C1` 0/4, `C0*x0+C1` 4/4). This is the LM
   arm that dominates const-free source cost (Doc B), so the loss compounds: the new asset silently
   drops the whole growing-basis affine family. Likely needs column scaling / whitening of the
   intercept+basis design before the QR, or a log-domain path for exponential bases.
3. **Run the go/no-go calibration prototype** (Doc A). Complete L4 AND L5 mine on the real 13-leaf
   `dev_7-4` config with lever 1 + streaming NF, measuring (a) per-source CPU-s on the real
   566,280-candidate library and (b) rule-set parity vs the unfiltered reference miner (zero
   divergence certifies the whitelist sound in practice). Decision rule: <=~2 CPU-s AND exact parity
   -> commit A+C, offer B; >~5 CPU-s -> A + sampled-6/7 only. NOT yet run.

### IMPORTANT — soundness / reproducibility / quality
4. **pow1_2 / pow of -inf parity fork** (checker review, MAJOR-titled, contained). Miner kernel
   returns `+inf` where deployment numpy array returns `nan`; the audit's "IEEE aligned across all
   surfaces" claim did not cover pow-of-(-inf). NOT exploitable through the equivalence gate (the
   `min_informative` evidence gate blocks it — verified `pow1_2(-inf)==inf -> False`), but the
   constant-folder path and miner<->deployment consistency should be closed.
5. **Record provenance in the mined artifact** (checker review, MAJOR). `rules.json` is a bare list
   with no seed / X spec / tolerances / challenges / version. Also: when X is passed as an explicit
   array (not a seed), the mine is not reproducible from the seed and X is recorded nowhere.
6. **Extend the count-DP cross-check to the sampler path.** The cross-check guards enumerated lengths
   only; the top-down sampler for infeasible lengths is verified uniform once but has no per-run
   assertion.

### LOWER — scale headroom + test/doc hygiene
7. **Sampler integer ceiling.** `rng.integers(counts[length])` raises `ValueError` above 2^63
   (~9.2e18). 13-leaf L7 = 3.36e10 is fine; a length-8+ mine crashes.
8. **Test gaps.** `test_vacuous_nan_pair_rejected` and `test_saturation_tower_rejected` pass for a
   different reason than their docstrings assert (do not pin what they claim); sampler uniformity,
   fit-recall (5->16), and the actual `-0.0` signed-zero case are unpinned.
9. **Doc hygiene.** The audit's "STATUS: RESOLVED — all six fixes" banner is premature (3 of the 5
   follow-up commits are fixes-of-fixes for regressions the hardening introduced); CHANGELOG says the
   mine-X log-uniform tier is `1e-6..1e6` where the code uses `1e-4..1e3`; propagate the 13-leaf
   correction into any doc still quoting 5-leaf figures (Doc B).
10. **R3 — enumeration-modulo-rules** (Doc B). The real lever on the L7 source-count wall (prune
    upstream of enumeration; stratify L7 into rule-length sub-passes so any L7 source reducible by
    <=6 rules is never materialized). Larger effort; the path to better-than-sampled L7. Post-launch.

## Recommended critical path to the re-mine

R1 (fold-filter + parity gate) and BLOCKER 2 (affine recall) in parallel -> BLOCKER 3 (L4+L5
calibration on the real 13-leaf config) -> read per-source cost + parity -> commit A+C, launch the
node-week mine with provenance recording -> confirm-stage + saturation curve -> version + release the
new `dev_7-4` engine (deploys the checker fix). Items 4-6 fold in before launch; 7-10 are hygiene /
post-launch.
