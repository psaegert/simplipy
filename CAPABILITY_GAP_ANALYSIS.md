# simplipy capability-gap characterization — where does dev_7-3 leave simplifications on the table?

**Date:** 2026-07-01 · **Goal:** for the "make simplipy maximally capable" mission, find where dev_7-3's simplify misses reductions, bucket into {greedy-application, missing-rule, canonical-difference}, and pick the lever. Method: dev_7-3 simplify vs a sympy oracle (object-faithful: each `<constant>` → a DISTINCT sympy symbol, so sympy can't merge independent fittable constants) on 2500 realistic random expressions.

## Result: gap = 4.7% of outputs have a sympy-shorter equivalent — but most is NOT safely fixable

| bucket | % of gap | % of corpus | fixable? |
|---|---|---|---|
| vocab: coeff/power/rational outside the engine's {2..5} compound ops (`*16`, `/8`, `**10`, `3/(5x)`) | 55% | 2.6% | NO — would need new operators (a different engine + model vocab) |
| candidate genuine miss (in-vocab shorter form) | 32% | 1.5% | PARTLY (see below) |
| complex / non-real arithmetic (sympy uses ℂ: `re()`, `I`, `sqrt(-1)`) | 10% | 0.5% | NO — dev is real-f64 by design |
| const-fold (arith, degenerate) | 2.5% | 0.1% | NO |

**The 32% "candidate genuine miss" splits again**, mostly into things dev is RIGHT to leave alone:
- **Domain-restricted inverse pairs** (`exp∘log` needs x>0, `cos∘acos` / `sin∘asin` need |x|≤1, `cosh∘acosh` needs x≥1, `tanh∘atanh` needs |x|<1): dev **deliberately keeps** these (verified) — collapsing them unconditionally on a skeleton is UNSOUND. NOT a fixable miss; it's a correctness feature.
- **`count_ops` metric artifacts** (`mult3(inv(x))` vs sympy `3/x` — same 2-op dev form, sympy counts fewer): not real misses.
- **Odd-function symmetry** (`sin(-x)→-sin(x)`, and tan/sinh/tanh/asin/atan/asinh/atanh): the one genuinely-safe fixable class — see below.

## The one clean, safe capability win: the 8 odd-function symmetry rules

dev_7-3 handles EVEN functions (`cos(-x)→cos(x)`, `cosh(-x)→cosh(x)` — verified reduced) but is **missing all 8 ODD-function rules** (`sin/tan/sinh/tanh/asin/atan/asinh/atanh (neg(x)) → neg(f(x))` — verified UNCHANGED).

**Why the asymmetry exists (root cause):** `cos(-x)→cos(x)` is length-*reducing* (3→2 tokens; neg disappears), so the length-driven miner found it. `sin(-x)→-sin(x)` is length-*neutral* (3→3; neg moves outward), so the miner **skipped it by design**. But it is a valuable *canonicalization*: pushing `neg` outward exposes downstream reductions (measured example: `+ x2 sin neg x2` → `x2 - sin(x2)`) and improves dedup (canonical sign placement). Domain-safe (odd-function identity holds everywhere in-domain).

## Lever verdict

- **Better application algorithm (e-graph / saturation): NOT warranted.** Greedy-application misses are ~0 — separately established: the sort-in-loop and the full mask+sort+rules joint-fixpoint experiments both came back negligible-to-regressing. The greedy top-down Kruskal application is not leaking reductions.
- **General mining of more/larger rules: NOT warranted.** Combinatorially walled (see `PHASE_B_OFFLINE_PLAN.md`), and the gap it could address is mostly not-safely-fixable (vocab limits, complex arith, domain-unsafe inverses).
- **RECOMMENDED: add the 8 odd-function symmetry canonicalization rules** (hand-written, not mined — the miner can't, they're length-neutral). Cheap, domain-safe, principled (removes the even/odd asymmetry), and enables downstream reductions. A new engine version (dev_7-3 stays the frozen v23.0 anchor).

## Caveats
- The 4.7% is on a RANDOM corpus that over-stacks unary ops (`mult4(mult4(x))`, `exp(log(tanh(x)))`); real SR expressions do this far less, so the real-world gap is smaller and even more vocab/inverse-dominated. The model-generated-candidate distribution (the survey's Experiment 1) was not measured here (needs an inference run).
- `count_ops` (sympy) vs prefix-node-count are only approximately comparable; the buckets are robust to this, individual borderline cases less so.

---

## UPDATE 2026-07-01 — the real lever is RE-MINING (hand-adding rules rejected by the user)

User constraint: rules must be produced ALGORITHMICALLY (no hand-writing) and must be LENGTH-REDUCING (not same-length canonicalization). That kills the odd-function-*symmetry* idea above (`sin(-x)→-sin(x)` is length-NEUTRAL). But it surfaced the actual lever:

**The shipped `dev_7-3` ruleset is INCOMPLETE relative to the current Rust miner.** Verified chain:
- `neg(f(neg(x)))→f(x)` for all 8 ODD functions is a length-REDUCING (4→2 tokens), general domain-safe, within-config (source 4≤7, target 2≤3) rule. Shipped rules.json lacks all 8 (grep: none). **The current Rust miner produces all 8** when run on those sources (from its length-reduction criterion — no hand-writing).
- Sizing: mined a 6000-source sample of length-4 sources with the current miner (context = shipped rules with LHS≤3) → **37% of the canonical length-4 rules are ABSENT from shipped** (78 new / 212). Genuine, NOT a generation-leaf artifact (shipped itself uses inf/nan leaves — 86166/114000 rules contain inf/nan). ~half the new rules are clean (real-expression-relevant: `- _0 neg _1 → + _1 _0` [a−(−b)→b+a], `/ _0 pow5 _0 → inv pow4 _0` [x/x⁵→1/x⁴], `- 1 neg _0 → - _0 (-1)`, `+ div4 _0 _0 → div4 mult5 _0`, `mult3 * np.pi _0 → * _0 <constant>`); ~half are inf/nan-degenerate (valid, rarely fire on real data). All mined rules are sound (100% soundness gate from the M4 work).
- Likely cause: the original dev_7-3 was mined long ago under a weaker/time-limited configuration (fewer challenges/samples, or a `--timeout`), so it never found these. The current Rust miner (16 challenges × 16 retries × 1024 samples, no timeout) is more thorough.

**LEVER VERDICT (final):** the algorithmic, length-reducing capability improvement is to **RE-MINE `dev_7-3` with the current Rust miner** — it produces the missing length-reducing rules as a pure function of the algorithm. This is exactly what the Rust offline port enables: a full re-mine is ~2.8 days on a single free box (was cluster-only). Produces a new engine version; `dev_7-3` stays the frozen v23.0 anchor. NOT e-graph (greedy misses ~0), NOT hand-adding, NOT bigger configs (walled), NOT same-length canonicalization.

**To do before committing the ~2.8 d:** (1) confirm the original dev_7-3 mining params to explain the gap (weaker mine?) — check `find-simplifications` invocation / any recorded config; (2) the exact real-world gain (how much shorter real expressions get) is quantified BY the re-mine + a before/after simplification comparison on realistic + benchmark expressions.
