# Phase B — porting the OFFLINE phase (rule mining) to Rust

**Status:** feasibility + plan (2026-06-30). **Milestone 1 BUILT + validated** (branch `feat/offline-rust-miner`); the multi-day rest awaits the user's launch call.
**Predecessor:** Phase A (online `simplify` + conversions) shipped as Rust in simplipy 0.3.0.
**Question answered:** can the offline phase (`find_rules`) be Rust-ported like the online phase, and does that let us mine *more / larger* patterns in the same wall-clock?

> ⚠️ **COURSE-CORRECTION from Milestone-1 measurements (see the M1 RESULTS section at the bottom — these supersede the pre-build speedup framing in this TL;DR):** the vectorized evaluator is at **numpy PARITY (~1×), not a speedup source** — numpy is already compiled C. The offline hot path is `scipy.curve_fit` (compiled C/Fortran), so a *faithful* Rust port yields only a **modest** speedup (unlike the inline phase, whose hot path was interpreted Python). The real lever is **algorithmic** — the user's "maybe no optimizer" instinct: linear-in-params constants need a closed-form solve (measured ~5× vs curve_fit AND deterministic), not an LM.

---

## TL;DR

- **Feasible — yes**, and a natural Phase-A successor. The hard numeric foundation (`numeric.rs` f64 evaluator, libm-exact transcendentals, ryu byte-exact float repr, the whole Rust `simplify`) already exists.
- **All the cost is in the constant-fitting path:** a worker-aware in-process profile (below) shows `exist_constants_that_fit` is **85.9 % of verification time**, ~492 curve_fit calls per surviving source. But *inside* that 86 %, MINPACK's LM core (`_lmdif`) is only ~5 s — **the bulk is the Python residual/eval layer** (`func_wrapped`→`pred_function`→`safe_f`, ~17 s) + scipy dispatch. **Decoupling consequence:** the *speedup* comes from the Rust **tape evaluator** eliminating Python in the residual loop (validatable at Milestone 1, before the LM); the native LM's 4/5 difficulty is about **accept/reject correctness, not speed**.
- The Rust `simplify` Kruskal-prune (Phase A) **already kills 94 % of generated expressions at 0.048 ms each**; generation is 4 s of Python for millions of expressions. Neither is the bottleneck.
- **"More/larger" is a frontier, not a single multiple** — and the multiple itself can only be pinned down by a tape-evaluator prototype (bounded ~10–50×, not assertable from a profile). The thoroughness axis (`challenges`/`retries`/`n_samples`) is simultaneously the *soundness* lever and sits on the hot path — so the speedup most naturally buys **sounder** mining (or ~one `max_source` step), not unbounded rule-count growth.
- **Soundness is the binding constraint and the optimizer is NOT soundness-neutral** (corrected from an earlier draft): `X` is sampled *once*; scipy's LM occasionally *failing to converge* is an accidental conservative filter. A different Rust LM can mint *new* domain-restricted false equivalences. Verification target = **match scipy's accept/reject decision both ways**, and any Rust-only rule is a soundness regression to audit. (User reaffirmed 2026-06-30: keep the engine conservative.)

---

## The offline algorithm (what we are porting)

`SimpliPyEngine.find_rules` (engine.py:2518), two phases:

1. **Generation** — `construct_expressions` (utils.py:711): combinatorial level-by-level enumeration of all valid prefix expressions, dedup by set per length. Pure Python.
2. **Verification** — `find_rule_worker` (engine.py:2337), multiprocessing workers. Per generated source:
   - **Kruskal prune** (driver, engine.py:2722/2782): `self.simplify(expr, max_iter=5)` (Rust); if it shortens, skip.
   - else test numerical equivalence vs every shorter candidate (grouped by length & variable-subset):
     - candidate **without** constants: `np.allclose` over `challenges(16) × 3^n_const` sign-combos (engine.py:2433);
     - candidate **with** constants: `exist_constants_that_fit` → scipy `curve_fit` (LM), looped `challenges × 3^n_const × retries(16)` with early breaks (engine.py:2454).
   - select target by wildcard-multiplicity (engine.py:2489); periodic dedup + JSON save.

`dev_7-3` mining params (configs/create_dev_7-3.yaml): `max_source=7, max_target=3, n_samples=1024, challenges=16, retries=16`; 38 ops (5 binary + 33 unary) + 9 extra terms. Output = 114 k rules / 23 MB. **`dev_7-3` is FROZEN (the v23.0 training anchor); a larger mine ships as a NEW engine_id (e.g. `dev_9-4`) and a new package version.**

---

## Empirical profile (the discriminator) — RESOLVED 2026-06-30

Worker-aware, **in-process** (the real `find_rule_worker` called synchronously under cProfile, so curve_fit — normally only in forked children — is captured). Single-threaded (`OMP/OPENBLAS/MKL=1`). Calibration = dev_5-2 settings on the standard 38-op engine with the full 114 k dev_7-3 ruleset loaded. Script: `scratchpad/profile_offline_miner.py`.

**A — Generation (true pre-prune counts):** 4.06 s to produce **6,399,156** expressions. Per-length: len4 71 k, len5 870 k, len6 2.85 M, len7 2.59 M; growth ratio ~3–35× per length. (Note: the `{2:262 … 7:21502}` distribution cited earlier is *mined-rule* counts from rules.json, **not** generation counts — generation is ~50× larger.)

**B — Kruskal prune:** **94.2 % of generated expressions are killed** by Rust `simplify` before verification, at **0.048 ms/call**. Kill-fraction rises with length (len4 90 %, len5 94 %, len6/7 94 %). ~6 % survive to verification. (Measured with the mature ruleset loaded; during a cold mine the rule set grows from empty, so early kill-fraction is lower.)

**C — Verification cost (the headline):** worker wall 56.98 s for 180 survivors = **316.6 ms/source**.

| bucket | cumtime | share of worker | calls |
|---|---|---|---|
| `exist_constants_that_fit` (→ scipy LM) | 48.96 s | **85.9 %** | 97,977 |
| `np.allclose` (no-const path + post-fit) | ~6.0 s | ~10 % | 145,780 |
| `safe_f` / eval + codify/compile/`prefix_to_infix` | overlapping, ~15–25 % (much *inside* curve_fit residuals) | — | millions |

**~492 curve_fit calls per surviving source.** Most verify time is spent **failing** (rejecting non-equivalences) — every constant-bearing candidate that does not match still pays a full challenge×retry curve_fit loop before bailing.

**Inside the 86 %:** MINPACK `_lmdif` core ≈ 5 s; Python residual/wrapper layer (`_memoized_func`→`func_wrapped`→`pred_function`→`safe_f`) ≈ 17 s; scipy dispatch the rest. **The speedup is in the residual layer (the tape evaluator), not the LM iteration core.**

**Population caveat (carry on the 86 % itself):** these survivors passed a *mature* 114 k-rule Kruskal prune → 0 rules found → the hard, failing tail, which over-weights curve_fit vs a cold mine (where some rules are still discovered via the cheaper no-constant path). **The verdict is robust to halving** — even at ~43 % curve_fit still dominates and the LM stays on the critical path. A cold-start (empty-rules) bracket is optional, not blocking.

**Verdict: curve_fit DOMINATES.** Porting everything *except* curve_fit is Amdahl-capped at ~1.16×. The native LM is the critical path for *correctness*; the tape evaluator is the critical path for *speed*.

---

## Component plan

| component | on critical path | difficulty | notes |
|---|---|---|---|
| **Native Levenberg-Marquardt** + `exist_constants_that_fit` wrapper | **YES (correctness, not speed)** | 4/5 | **Recommend porting MINPACK's `lmdif` specifically** (its damping/scaling/convergence tests), NOT a generic `argmin` LM: dev_7-3's conservative frontier *was defined by MINPACK*, so a faithful `lmdif` matches accept/reject **by construction** and the doubled soundness target comes free. A different LM forces empirically re-establishing that frontier. Finite-diff Jacobian over the tape; curve_fit-matched stops (ftol/xtol/gtol ≈1.49e-8) + max-fev; non-convergence → `False`. Wrap with finite-row mask + the `n_const>n_valid` bail (scipy #13969) + post-fit allclose. The 4/5 difficulty is **accept/reject parity**, not throughput (the speedup is the tape evaluator). |
| **Vectorized tape evaluator** (vars + free-param leaves over `n_samples` rows) | YES (feeds the LM) | 3/5 | Compile each prefix expr ONCE to a flat stack tape; reuse `numeric.rs apply_op` (exact IEEE/special-value semantics) + `operators.rs arity_of`. Removes the per-candidate `realize→infix→codify→compile` recompile that runs in the inner loop today. |
| **Native `np.allclose`** (the real decision gate) | YES | 2/5 | Asymmetric `|a-b| <= atol + rtol*|b|`, b = 2nd arg, rtol=1e-5, atol=1e-8, `equal_nan=True`. This — not the LM — is the finite-arithmetic parity surface; last-ULP libm diffs are absorbed by rtol. |
| No-constant equivalence path (3a) + wildcard-multiplicity selection | YES | 2/5 | Pure-logic port; covers the *decision* for 87.8 % of mined rules (source has no constant) but **not** 87.8 % of compute — the early empty-var `<constant>` candidate forces curve_fit on most sources regardless. |
| `rayon` parallelism (replace Process+SharedMemory+Queue) | no | 2/5 | Parallelize **within a length, hard barrier between lengths** (the Kruskal prune uses the growing rule set, so mined output is order-dependent — flat parallelism would confound the rediscovery check). |
| `construct_expressions` generation | no | 2/5 | Pure-Python combinatorics; port last for a fully-native miner. Not the bottleneck. |
| RNG for the challenge protocol | no | 1/5 | `rand`/`rand_distr`; will NOT match numpy's MT stream — expected, validate by equivalence not byte-parity. |

---

## Payoff frontier — "more / larger in the same time"

Disentangle the three axes the question collapses:

- **Axis i — `max_source` 7→8/9 ("longer" sources):** generation grows ~order-of-magnitude per step *and* per-source verify cost grows (more shorter candidates to scan). The most expensive axis.
- **Axis ii — `max_target` 3→4/5 ("larger" targets):** enlarges the candidate set per source and weakens the early-break → super-linear verify cost.
- **Axis iii — thoroughness (`n_samples`/`challenges`/`retries`):** the **soundness** lever — more of it mints FEWER but SOUNDER rules — and it sits directly on the curve_fit hot path (linear in challenges×retries). Note the two soundness mechanisms are *different levers* (see below).

**Soundness mechanism — two distinct paths (do not conflate):**
- **No-constant candidates:** the LM is never invoked. Domain-restricted false equivalences are governed by the **X distribution + `allclose`**. Example: `sin(asin(x))→x` is *rejected* by dev_7-3 only because `X ~ N(0,5)` samples `|x|>1` → `asin` returns nan → mismatch. The thoroughness lever here is **`n_samples` and the X range**, NOT `challenges`. This path is optimizer-independent → a Rust port cannot change its conservatism (given identical X).
- **Constant-bearing candidates:** the **LM's convergence frontier** governs which equivalences get minted — *this* is where a different/stronger Rust LM can newly catch a fit scipy missed and reduce conservatism. `challenges` here guards against constant-*value* coincidence (it resamples the constants, X stays fixed), not X-domain restriction. → this is exactly why the LM must replicate MINPACK's accept/reject frontier, not merely "find fits."

**Grounded affordability:** the realized speedup is bounded by how much faster a native LM + tape evaluator is than scipy `curve_fit` + the Python recompile/eval per call. On 1024-row arrays the cost is Python/scipy dispatch, not BLAS, so the headroom is large (plausibly 10–50× on the constant path — **defer the exact number to a Rust micro-benchmark of the ported LM vs curve_fit**). As a frontier, that speedup roughly "pays for" **one** axis-i step *or* a generous thoroughness increase within the current dev_7-3 wall-clock — **not** a simultaneous big `max_source` and `max_target` jump.

**Feedback cost (must price in):** a much larger `rules.json` makes the ONLINE `simplify` scan more candidate rules per node, partially eating the Phase-A inline win (operand-index + Rust mitigate, do not eliminate). See the pruning TODO.

---

## Verification strategy (replaces Phase-A byte-parity — the miner is stochastic)

1. **Kernel-first on the 114 k dev_7-3 oracle:** the Rust evaluator + native LM must certify each known rule as a true equivalence AND reject negatives. Strengthen negatives beyond "easy" ones: instrument the Python miner to log a sample of REJECTED constant-bearing near-misses and confirm the Rust kernel also rejects them (probes the real LM-divergence boundary).
2. **Miner-level rediscovery on dev_5-2 (then dev_7-2):** run the native miner, compare to the Python mine by equivalence/coverage. **Hard soundness gate:** any rule the Rust miner mints that Python did not is a soundness regression requiring human audit. Preserve processing order (within-length parallel, between-length barrier) so any divergence is attributable to the LM, not reordering.
3. **No automated check certifies a NEW dev_9-4 ruleset's soundness** (the oracle is positives-only; new false positives are by construction absent). dev_9-4's new rules need the same conservative human discipline as the original mine.

---

## Sequenced milestones

0. **(blocks launch — DONE 2026-06-30)** worker-aware profile → curve_fit dominates → native LM is the critical path. ✅
1. Rust vectorized tape evaluator (var+param leaves) + native `np.allclose`; validate kernel-first on the 114 k oracle. **Micro-benchmark the tape evaluator inside a residual loop vs scipy's `func_wrapped`/`safe_f` here — this pins down most of the speedup multiple *before* the LM.**
2. No-constant path + wildcard selection + `rayon` (within-length parallel, between-length barrier); keep Rust `simplify` for Kruskal. Re-validate kernel-first.
3. **Native MINPACK-`lmdif` port** + `exist_constants_that_fit` wrapper (the correctness hard core). Goal = match scipy's accept/reject frontier both ways (faithful `lmdif` ⇒ by construction); speed already secured by M1's evaluator.
4. Miner-level rediscovery on dev_5-2 (recall + soundness-regression audit) before trusting any larger mine.
5. (optional) rediscovery on dev_7-2; port `construct_expressions` for a fully-native miner.
6. **(user launch call)** a NEW engine_id (e.g. dev_9-4) mine at the chosen frontier point — recommend favoring thoroughness over raw max_source/max_target per soundness. Ship as a new package version; `dev_7-3` stays frozen.

---

## Tracked TODO — pruning (user note, 2026-06-30)

`prune_redundant_rules` (the `--prune` flag) is currently OFF for dev_7-3. Investigate later, **both directions**: (1) **offline** — does pruning during mining lower total mine time (smaller rule set → cheaper Kruskal `simplify`) net of the O(rules²)-ish redundancy pass? (2) **online** — a smaller `rules.json` means the online `simplify` scans fewer candidate rules per node → directly mitigates the larger-ruleset feedback cost; this is the lever that could let a thorough dev_9-4 stay LARGER without proportionally inflating online cost. Measure: (a) mine wall-clock with/without `--prune` on dev_5-2/dev_7-2; (b) online `simplify` throughput on a fixed corpus, pruned vs unpruned; (c) confirm prune removes only truly-redundant rules. Do this **before** a dev_9-4 launch so the frontier decision accounts for pruning.

---

## Open questions to confirm before the multi-day launch

- The dev_9-4 **frontier point** (max_source vs max_target vs thoroughness) — recommendation: favor thoroughness, per soundness + the conservative reaffirmation.
- Acceptance of the new-engine_id + new-package-version path (dev_7-3 frozen).
- Whether to run `--prune` for dev_9-4 (see TODO).
- **LM design driver:** confirm a faithful MINPACK-`lmdif` port reproduces curve_fit's accept/reject frontier on dev_5-2/dev_7-2 (recommended over a generic `argmin` LM precisely to make this hold by construction). This — not raw fit success — is the LM acceptance criterion.
- The realized speedup **multiple** — pin it down with the M1 tape-evaluator micro-benchmark; it is the one number not assertable from the profile. **[ANSWERED in M1 RESULTS: eval is at numpy parity; the multiple comes from the curve_fit replacement, not the evaluator.]**

---

## Milestone 1 — RESULTS + course-correction (built + measured 2026-06-30)

Branch `feat/offline-rust-miner`. New Rust: `rust/eval.rs` (column-wise tape evaluator, fn-pointer dispatch resolved at compile) + native `allclose`; `numeric.rs` refactored so `apply_op` delegates to reusable `unary_fn`/`binary_fn` (one operator-semantics source of truth for inline + offline); FFI `evaluate_batch` / `allclose` (+ a dev `eval_bench_resident`). 19/19 cargo tests, `cargo fmt` clean, warning-free build.

**Validation (Python oracle harness `scratchpad/oracle_eval_m1.py`, env `flash-ansr`):**
- **Native `allclose` vs `numpy.allclose`: 0 / 5000 disagreements** (incl. nan/inf/signed-zero) — exact.
- **Kernel parity (Rust `evaluate_batch` vs numpy eval): ~99.95 %** (12 / 23,988 real-valued differ). **Every** difference is in the inf / signed-zero division corner (the improved-vs-faithful IEEE boundary from Phase A), plus 12 complex-scalar cases where Rust gives `nan` (the improved behavior, matching the shipped folder). **Zero genuine finite-value errors.**
- **Oracle equivalence (constant-free wildcard-bearing dev_7-3 rules, Rust native `allclose`): 99.53 %** (2133 / 2143). The 10 "disagreements" are **all** inf/0 signed-zero-division pseudo-equivalences that are sign-dependent — the improved evaluator correctly declines them. These are exactly the IEEE-pathology rules the soundness review flagged for human audit (a real, pre-existing soundness observation about dev_7-3).

**THE COURSE-CORRECTION (measured, supersedes the earlier "tape evaluator is the speedup" framing):**
- **Vectorized Rust eval ≈ numpy parity** (sin-expr 1.1×, arithmetic 0.7–1.0×, exp 0.5×). numpy is already vectorized C; a Rust tape matches it, it does not beat it. **The evaluator is NOT a speedup source.**
- **Naive per-call FFI is a trap:** marshaling a per-call X list makes Rust ~0.2–0.5×. The eval must stay resident in Rust (the M3 architecture); an FFI-per-residual bridge would be *slower* than Python.
- **The cost is `scipy.curve_fit` itself:** measured **~782 µs per nonlinear fit**, **~103 µs per linear fit**, vs **~11 µs per residual eval**. So ~90–98 % of a fit is scipy / MINPACK-Fortran-bridge / Python-wrapper overhead, not math.
- **Consequence:** a *faithful* Rust offline port gives only a **modest** speedup, because its hot path (`curve_fit`) is already compiled C/Fortran — unlike the inline phase, whose hot path was interpreted-Python tree-matching (hence 17–99×). **The offline win must come from ALGORITHM, not language.**

**The real lever (the user's "maybe we don't even need LM" instinct — now quantified):**
- **Linear-in-params candidates** (constants appear only affinely: `C`, `C0*_0+C1`, `C0*f(_0)+C1`, …) need **no iterative optimizer** — one closed-form least-squares / normal-equations solve decides equivalence. Measured **~5× faster than curve_fit AND deterministic** (no random `p0` → the 16 retries collapse to 1 → a further ~16× on those candidates). Determinism also answers the "more reliable" part of the user's note.
- **Genuinely-nonlinear-in-params candidates** (a constant inside `sin`/`exp`/…) still need an optimizer; a native LM removes the ~782 µs scipy/Fortran-bridge overhead per fit.
- Plus two language-level wins that *do* apply: eliminating the **per-candidate Python recompile** (tape compiled once, cheaply) and the **multiprocessing IPC** (rayon over a resident X).

**Revised M3 (the primary speedup milestone, now structure-aware):** (1) statically classify each constant-bearing candidate as linear-in-params vs nonlinear-in-params (does any `<constant>` pass through a nonlinear op?); (2) linear → closed-form solve in Rust (deterministic, no retries); (3) nonlinear → native MINPACK-`lmdif` port (accept/reject-faithful, removes scipy overhead); (4) tape reuse + rayon. The "more/larger in the same time" payoff is set by (1)–(2), not by the evaluator or a faithful port.

**Build/run notes:** `cargo build --release` → copy `target/release/lib_core.so` → `src/simplipy/_core.abi3.so` (gitignored), import via `PYTHONPATH=src` with the `flash-ansr` (py3.13) env — leaves the env's installed simplipy 0.3.0 untouched. The env's installed package is NOT modified by this branch.

---

## Milestone 3a — RESULTS: linear-in-params closed-form fit (built + measured 2026-06-30)

New `rust/fit.rs`: a `<constant>`-degree classifier (`ConstFree` / `Affine` / `Nonlinear`) + a closed-form affine fitter (build the design matrix by `eval(C=0)` and `eval(C=e_j)`, solve ridge-regularized normal equations on the finite-row mask, accept via `allclose`). FFI `classify_linearity` + `exist_constants_fit_linear` (returns `Some(decision)` for affine, `None` for nonlinear → deferred to M3b). 21/21 cargo tests, fmt clean, warning-free.

**Validated against scipy `curve_fit` on dev_7-3 (harness `scratchpad/m3a_validate.py`):**
- **Coverage: 76.1 %** of constant-bearing fit targets are **affine** (16,129 / 21,194) → ~3 in 4 `curve_fit` calls are replaceable by a deterministic closed-form solve.
- **Decision parity: 99.83 %** positive / **99.93 %** negative vs `scipy.exist_constants_that_fit`. The ~0.17 % disagreements are all **pure-constant sources** (e.g. `(C⁴)^π`) — which the real miner short-circuits via constant-folding *before* `exist_constants_that_fit`, and where scipy's scalar-return quirk wrongly returns `False` while Rust correctly fits. Not real-mining inputs; Rust arguably more correct.
- **Speed: 8.2× faster** per call (163.7 µs scipy → 19.8 µs Rust incl. FFI) **and deterministic** → in the worker the 16-retry loop collapses to 1 on affine candidates (~16× fewer fit attempts on top of the 8×).

**Net:** the user's "maybe no optimizer" lever is confirmed and large — 76 % of constant-fits become a ~8×-faster, retry-free, deterministic closed-form decision at >99.8 % parity. The remaining **24 % nonlinear-in-params** candidates still need an optimizer → **M3b (native MINPACK-`lmdif` port)**, which also removes the ~782 µs scipy overhead on those.

**Milestone map (updated):** M1 ✅ (evaluator + allclose) · M3a ✅ (affine closed-form) · M3b ✅ (native LM, below) · **M2** = the no-constant equivalence path (challenges × sign-combos × allclose) + wildcard selection + `rayon` · **M4** = compose M2+M3 into the full native `find_rule_worker` + driver (Kruskal prune via Rust `simplify`, incremental dedup/save). M2 and M3 are complementary halves of the candidate loop; both required.

---

## Milestone 3b — RESULTS: native Levenberg-Marquardt (built + measured 2026-06-30)

`rust/fit.rs` extended with a native LM (`lm_fit`: Marquardt diagonal scaling, forward-difference Jacobian, early-exit on `allclose`) + a seeded splitmix64/Box-Muller PRNG for random restarts, wired into the complete native `exist_constants_that_fit` (`exist_constants_fit`: affine → closed-form, nonlinear → `n_restarts` LM solves). FFI `exist_constants_fit`. 22/22 cargo tests, fmt clean, warning-free. **M3 (the whole constant-fit kernel) is now native — no scipy on the fit path.**

**Validated against scipy `curve_fit` on nonlinear dev_7-3 targets (R=16 restarts, harness `scratchpad/m3b_validate.py`):**
- **Decision parity: 97.5 % positive / 100.0 % negative.**
- **Soundness intact:** the 100 % NEGATIVE agreement (target vs an unrelated source's output) means the native LM never accepts a false fit scipy rejects — directly answering the review's "a different LM could mint new false equivalences" concern.
- **Directionality:** the 2.5 % positive disagreements (10 Rust-finds-more, 5 Rust-misses) are **all** the ill-conditioned `pow(C, _0)` family (fitting an exponential *base* `C^x`), where convergence is restart-luck for *both* optimizers. Net, Rust recovers slightly more genuine rules. (Future "no-optimizer" refinement: `C^x` log-linearizes to `log(y)=x·log(C)` → closed form, same trick as M3a.)
- **Speed: ~1× on nonlinear** (both run iterative LM) — as predicted; the speed win is the affine 76 %, M3b's job is completing the kernel natively with faithful decisions.

**Why M3 is still a big win despite nonlinear being ~1×:** weighting by coverage, 76 % of fits get the ~8× deterministic closed-form (with the 16-retry loop collapsing to 1) and 24 % stay ~par. On top, going fully native unlocks the M4 structural wins this isolated micro-benchmark does NOT capture — no per-candidate Python recompile, no multiprocessing IPC/pickle, no GIL — which apply to ALL candidates. The realized miner speedup will exceed the per-fit numbers.

---

## Milestone 2 — RESULTS: no-constant equivalence + selection primitives (built + measured 2026-06-30)

New `rust/worker.rs` (the constant-FREE candidate branch + selection): `equivalent_no_const` (the source-constant resampling test, engine.py:2433-2452: eval the const-free candidate once, then require `allclose(source, candidate)` across `challenges` resamplings × every `{-1,0,1}` sign combo), `violates_wildcard_multiplicity` (utils.py:938 port), and `select_best` (fewest-`<constant>` stable, skip wildcard-violators, all-numeric fold). Adds NO new numerics — only `allclose` (M1). FFI `equivalent_no_const` + `violates_wildcard_multiplicity`. 25/25 cargo tests, fmt clean, warning-free.

**Validated against Python (harness `scratchpad/m2_validate.py`):**
- **`violates_wildcard_multiplicity`: 0 disagreements / 134,000 pairs** (all 114k rule pairs, which carry `_j` wildcards, + 20k synthetic) — exact.
- **No-constant equivalence: 99.76 % positive / 99.96 % negative** parity vs a faithful Python replication; directionality balanced (3 Rust-more, 4 Rust-less). Every disagreement is the inf/nan/signed-zero/`(-1)`-fractional-power corner — the same improved-vs-faithful IEEE boundary from M1, not new behavior. On finite non-pathological expressions: effectively 100 %.

Note (faithful, deliberate): `violates_wildcard_multiplicity` matches `^_\d+$`, so on dummy-variable (`x0`..) MINING expressions it is INERT exactly as in Python → selection reduces to "fewest `<constant>` first" during mining (the `_j` form only exists after `deduplicate_rules` canonicalizes).

**Remaining for a runnable native miner (M4):** the `find_rule` scan assembly (iterate the candidate library by length / variable-subset, dispatch const-free → M2 / constant-bearing → M3, early-break on the first matching length, `select_best`) + the driver (generation, Kruskal-prune via the Rust `simplify`, `rayon` over sources WITHIN a length with a barrier BETWEEN lengths to preserve the order-dependent Kruskal/dedup, incremental JSON save). Then the end-to-end gate: re-mine **dev_5-2** natively and compare rule-rediscovery vs the Python mine. All per-candidate DECISION primitives (M1+M2+M3) are now native and validated; M4 is assembly + driver + the real "more/larger patterns" measurement.
