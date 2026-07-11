# Changelog

## Unreleased — 7-4 re-mine readiness: candidate fold-filter + growing-basis affine recall

The two launch blockers from the internal readiness review (blockers 1 and 2). Mining-path
only; the inline `simplify` engine and the shipped `dev_7-3` asset are untouched.

### Added (BLOCKER 1 — candidate library minimization, the "fold-filter")
- `find_rules(candidate_fold_filter=True)` / `_core.build_candidate_library(..., fold_filter=True)`:
  VARIABLE-FREE candidates of length >= 2 (e.g. `sin(<constant>)`, `pow(<constant>,<constant>)`,
  `exp(1)`) are dropped from the mining candidate library. Sound (the narrow, provable form of
  candidate minimization): a var-free candidate evaluates to one scalar per constant-assignment,
  so any source instance it matches is constant-valued -- which the length-1 `<constant>`
  candidate also matches at a strictly shorter length, preempting it in the shortest-first scan;
  the filter is gated on the bare `<constant>` candidate being present (inert otherwise).
  This removes the bulk of the constant-bearing (LM-fit) candidate arm -- the dominant per-source
  cost for const-free sources. Measured on the real
  13-leaf dev config (complete len<=4 library = 566,280 candidates): the filter drops 374,031
  var-free candidates (66%), and hard const-free non-reducing probe sources (len 5-8) go from
  374-474 s/source to 5.0-8.5 s/source -- a 55-88x per-source speedup at identical decisions.
  The library reports `n_candidates` / `n_filtered`; `find_rules(verbose=True)` logs both.
- **Order-independent fit seeds.** The per-(candidate, instance) LM restart seed is now a pure
  function of (per-source seed, candidate tokens, instance index) instead of a draw from the
  scan-order RNG stream. Same-seed mines remain byte-reproducible; the mined output of a given
  seed changes vs 0.5.0 only through restart-luck on marginal nonlinear fits. This makes the
  filtered-vs-unfiltered parity gate exact: both runs draw identical fit randomness for every
  shared candidate, so any ruleset diff would be attributable to the filter alone
  (`TestFoldFilter::test_mine_parity_filtered_vs_unfiltered` certifies zero diff end-to-end).

### Fixed (BLOCKER 2 — growing-basis affine recall)
- The affine closed-form constant fit solves a ROW-WEIGHTED least-squares system (weights
  `1/(atol + rtol*|y_r|)`, mirroring the RELATIVE per-row accept gate) with column max-abs
  equilibration and two fixed rounds of iterative refinement on the retained Householder-QR
  factor. Before: with a fast-growing basis (`exp`/`cosh`/`sinh`/`pow3+`), rows with |y| ~ 1e21
  dominated the unweighted solve in absolute terms and their f64 rounding (eps*|y|) buried an
  O(1) additive constant -- `C0*f(x)+C1` and `f(x)+C1` rejected on exactly-true constants (0/4
  recall) while the interceptless `C0*f(x)` passed. Now 4/4 across the family. The accept gate
  (`allclose_extends` at the solved constants, over all rows) is unchanged: the fix is
  recall-only, negatives still reject.

### Hardened by the adversarial verification round (5 refutation lenses + independent cross-checks)
A 20+-agent adversarial verification of the two changes above REFUTED the first version of the
dominance argument and found one real recall regression; all findings are fixed and regression-
tested (cargo 35, pytest 253):
- **Exact interval decision for the bare `<constant>` candidate.** The accept gate's feasible set
  in v is an intersection of per-row intervals; a (weighted or not) least-squares mean is NOT its
  Chebyshev center, so for skewed near-constant sources (63 rows at `e-2.4e-9`, one at `e+2.4e-9`;
  all within the band of v = e) `<constant>` rejected while the var-free `exp(1)` accepted --
  filtered and unfiltered mines DIVERGED. The bare `<constant>` fit now solves the feasibility
  problem exactly (branch-tracked interval intersection), which restores the fold-filter dominance
  as a theorem (residual edge: ~1-ulp rounding of the interval endpoints, 7 orders below the band).
- **Weighted-design overflow false-reject fixed.** A huge-but-finite basis entry (`sinh(705)` ~
  7.5e305) times a large row weight (1/atol at exact-cancellation rows like `cosh(x)-sinh(x)` = 0.0)
  overflowed to inf and NaN'd the solve. Columns are now equilibrated BEFORE weighting (entries
  <= 1), weights are capped against the residual target, and the second (post-weighting)
  equilibration was removed -- its combined scale product itself overflowed f64 and zeroed the
  solution. Adversarial slate: old solver 30/40 accepts, new 40/40, zero old->new regression flips.
- **Near-integer snap rescue.** Exact-cancellation rows tolerate ~atol while d(fitted)/dC ~ 1e303,
  so they demand constants BITWISE equal to the true ones -- any solver's +-1-ulp output was a coin
  flip. Solutions within ~rtol of integers are re-gated at the snapped integers (still
  gate-certified; deterministic accept for the integer-constant rule class).
- **Fold-filter var-detection fixed for >= 33 variables** (the scan `var_mask` truncates at 32; the
  filter now checks all `var_names`, so a candidate whose only variable is `x32` is never dropped).
- **`min_informative` default floored at 1** (integer division gave 0 for n_rows < 8, disabling the
  evidence gate; unreachable at the mine's n_rows = 1024, hardened anyway).
- Known bounded non-goal (documented, not fixed): a pair equivalent ONLY through bitwise-identical
  float evaluation under heavy cancellation (y built from the candidate's own eval, e.g.
  `3*cosh + 3*sinh` at negative x where the true value is below one ulp of the summands) can still
  reject. Real mining pairs evaluate through structurally different trees, so their rounding noise
  differs and the gate rejects such rows regardless of the solver; the old path passed this class
  only by ulp-luck.
- Verified HOLDS by the same round: end-to-end filtered-vs-unfiltered mine parity (three
  independent mines, identical rulesets), byte-determinism across processes / PYTHONHASHSEED /
  rayon thread counts, seed order-independence (0/160 divergences), refinement inertness on
  inconsistent systems, deterministic rejection on NaN/inf designs.

### Docs
- 0.5.0 entry corrected: the mine-X log-uniform tier is 1e-4..1e3 (as the code has always
  shipped), not 1e-6..1e6.

## 0.5.0 — 2026-07-11 — sound rule-mine checker (2026-07-10 audit) + Python mining mirror removed

The pre-0.5.0 checker shipped ~8.3% defective
rules in the dev_7-3 sample (52/840), including ~5,125 vacuous all-NaN wildcard rules such as
`asin(cosh(_0)) -> nan` (false at 0, where the source is pi/2).

### Fixed (checker soundness)
- **Vacuous equal_nan acceptance closed.** Certification now requires at least `min_informative`
  (default `n_rows / 8`) SOURCE-FINITE evidence rows, accumulated across challenge instances; a
  precomputed candidate-finite count serves as a fast necessary-condition gate for const-free
  candidates.
- **Tolerances tightened** `rtol` 1e-5 -> 1e-9, `atol` 1e-8 -> 1e-12 (0 borderline rules in the
  840-rule audit sample; tanh/exp saturation towers are no longer "equal" to constants).
- **Heavy-tailed, seeded evaluation matrix** (`_mining_sample_x`): 40% N(0,5) + 25% U(-50,50) +
  25% signed log-uniform magnitudes 1e-4..1e3 + 10% exact corner points ({+-0.0, +-0.1, +-0.5,
  +-1, +-2, +-e, +-pi, +-10}). (This entry mis-stated the tier as 1e-6..1e6 until 2026-07-11;
  the code always shipped 1e-4..1e3 -- the wider tail was tried and reverted for fit
  conditioning, see the `_mining_sample_x` docstring.)
- **GENERIC-EQUIVALENCE semantics with domain extension** (`allclose_extends`, the single accept
  gate everywhere: const-free compare, affine/log-linear/LM fit accepts). Rows where the SOURCE is
  finite bind: the replacement must be finite and equal within tolerance -- the exact corner points
  refute wrong-VALUE identities (`asin(cosh(_0)) -> nan` is false AT 0, where the source is pi/2).
  Rows where the source is NaN/inf are extendable: the replacement may complete them with the
  generic/limit value (`div(_0, _0) -> 1` certifies; `log(exp(_0)) -> _0` certifies under f64
  overflow). The evidence gate (`min_informative`) counts SOURCE-FINITE rows accumulated across
  challenge instances, so an (almost-)nowhere-defined source can never be rewritten from its
  corner rows alone.
- **Stage-2 confirmation** (`confirm=True`): every mined pair is re-verified on an independent,
  twice-as-wide X with fresh constant draws and seeds before it can enter the Kruskal cascade.
- **IEEE inv/div semantics** aligned across the Rust kernels, the scalar Python operators and the
  constant folder (`1/±0 = ±inf`, `x/0` sign-correct): the mine now certifies exactly the semantics
  the deployed numpy engine executes.
- **Complete Phase-1 enumeration.** The old pass-based closure stopped once the maximum length was
  REACHED, not SATURATED: the dev_7-3 mine saw 444,865 of the 9.0e9 length<=7 expressions and missed
  e.g. ALL 179,685 triple-unary chains at length 4. Phase 1 is now a bottom-up DP per length,
  complete by construction and cross-checked against an exact count DP every run.
- **Universe policy for infeasible lengths** (`source_sample_per_length`): lengths whose complete
  universe explodes (dev set: 2.4e8 at length 6, 8.8e9 at length 7) are drawn uniformly from the
  complete universe via a count-weighted top-down sampler; coverage is always logged, and sampling
  inside the candidate range warns (the library then no longer certifies minimality).
- **Full determinism**: seeded master RNG (`seed=42`), sorted set iteration for sources and the
  candidate library, derived per-length seed blocks. Same seed => identical rule set.
- **Fit-path completeness**: `constants_fit_challenges` / `constants_fit_retries` defaults raised
  5 -> 16 end to end (the old find_rules passed 5 against an FFI default of 16; measured fit-path
  completeness at 5/5 was ~24%).

- **Affine-fit conditioning fix**: the whole `C0*f(x)+C1` family silently REJECTED before. Two
  causes, both from the audit's own hardening: a GLOBAL trace-scaled Tikhonov ridge biased the
  intercept, and the normal-equations solve squared the condition number on the wide-magnitude X
  so an exact affine relationship's intercept came out ~5e-9 off (past rtol). Now the affine path
  solves by Householder-QR least-squares (no `A^T A`, no ridge -- works at cond(A), not its square),
  and the mine X's log-uniform tier is capped at 1e3 (from 1e6): 1e3 still exercises every
  saturation and f64 overflow the tier is FOR, but 1e6 wrecked the fit conditioning. Found by the
  generic-equivalence adversarial verification (a CONTESTED finding, confirmed by probe: 0/30 ->
  64/72 affine cases certify, 0 false-accepts; the residual misses have constants spanning >1e6).
- **Log-linear recall fix**: for `pow(<constant>, g)` candidates, only a closed-form log-space
  ACCEPT short-circuits; an imprecise `Some(false)` solve (its ~1e-10 base error amplified past
  rtol by large exponents on the heavy-tailed X) now SEEDS the LM restart instead of rejecting,
  restoring the declared const-bearing policy ("accept iff constants EXIST that fit"). Surfaced by
  a 20-agent adversarial verification of the generic-equivalence change (which found NO soundness
  defects).

### Changed
- **Perf**: the source expression is now evaluated once per source (per challenge instance) and
  shared across the whole candidate scan, instead of per (candidate, challenge, sign-combo); a
  const-free source collapses to a single challenge instance (identical targets add no evidence
  and only multiplied fit flakiness).

### Removed (BREAKING)
- **The pure-Python mining mirror is gone**: `find_rule_worker`, `exist_constants_that_fit` and the
  fork/SharedMemory pool (with the `n_workers` parameter). It duplicated `rust/worker.rs`/`fit.rs`
  and repeatedly desynced from them (the audit's IEEE inv/div fork). `find_rules` now requires the
  compiled core and raises `RuntimeError` on a bare engine; parallelism is rayon's
  (`RAYON_NUM_THREADS`).

### Compatibility
- Shipped rule assets (`dev_7-3` etc.) are unchanged by this release; they remain the v23 anchor.
  A re-mine with the hardened checker is a separate, explicitly-versioned artifact.


## 0.4.1 — 2026-07-02 — find_rules works with the Rust core + safe concurrent asset installs

### Fixed
- **`find_rules` now mines natively on the Rust core.** With `_core` attached (any engine from
  `load()` / `from_config()`), the fork-based Python pool mined **0 rules**: the workers mutate
  Python-side rule state while `simplify` runs on the immutable core — the same class of bug as the
  0.4.0 `prune_redundant_rules` fix. Phase 2 now delegates to the native mine when the core is
  attached (candidate library + per-length `mine_one_length` + `set_rules` between lengths,
  rayon-parallel; cap with `RAYON_NUM_THREADS`; `n_workers` only applies to the pure-Python
  fallback). Mined rules are now always canonicalized to the wildcard (`_j`) form.
- **`find_rules(X=<ndarray>)`** no longer raises `NameError` (the documented array form was never
  assigned to the internal data variable).
- **Concurrent cold-cache asset installs are now safe.** Installs serialize per asset via a
  `FileLock`, and an asset only counts as installed when **every** file in its manifest is present,
  so a partially-downloaded asset is correctly treated as not installed. New dependency: `filelock`.

### Changed
- Offline-mining research artifacts (the Phase-B plan, the capability-gap analysis, and the mine
  grid/validation drivers) moved out of the released repo into the research archive.

## 0.4.0 — 2026-07-01 — native offline rule-miner + simplify fixes + CLI fix

### Added
- **Native (Rust-core) offline rule mining.** An all-cores native mine driver (with a grid timing
  harness) supersedes the pure-Python `find_rules` for offline discovery against the Rust `_core`.
- **Closed-form `pow(C, x)` / `pow(x, C)` log-linearization** in the offline pipeline.

### Fixed
- **`sort_operands` is now idempotent** — canonical operand order is a fixpoint (rotation iterated to
  convergence; mask-before-sort), so simplifying a simplified expression no longer changes it. This can
  change the canonical operand ordering of some expressions vs 0.3.x (verified not to affect the
  downstream symbolic-data / flash-ansr / srbf test goldens).
- **`prune_redundant_rules` corrected against the Rust core** (the pure-Python prune produced 0 rules
  with `_core`); offline grid mining caps sources at length ≤ i and chunks the per-config budget.
- **`simplipy install <name>` / `simplipy remove <name>` now work.** They previously passed the asset
  *type* as the asset name (and the name as `force`/`quiet`), so every invocation raised
  `ValueError: Unknown asset: 'engine'`. They now take just an asset name with clean error handling
  (message + exit 1, no traceback); the vestigial `--type` flag is dropped from `install`/`remove`
  (`list` keeps it).

### Docs
- Documented the normalization helpers in the API reference; added a CLI reference for
  `list` / `prune-rules` / `resolve-rules` / `install` / `remove`; qualified `SimpliPyEngine`
  methods in the component overview; named `symbolic-data` as the direct downstream in the family DAG.

## 0.3.1 — 2026-06-28 — expression-token normalization helpers

Adds pure-Python `normalize_skeleton`, `normalize_expression`, and `normalize_variable_token`
to the package root (also available as `simplipy.normalization`). These canonicalize a prefix
token sequence -- variable tokens (`v1`/`x1`, case-insensitive) to a stable `x{n}`, and numeric
literals to a `<constant>` placeholder (skeleton form) or kept as-is (expression form) -- so two
expressions that are "the same" up to variable renaming / constant values compare equal.

Relocated from flash-ansr (behavior-identical) so the canonicalizer lives at the shared
expression-engine leaf that downstream packages (symbolic-data, flash-ansr, srbf) all depend on,
keeping holdout-matching and symbolic-recovery scoring consistent by construction. No change to
the Rust inline backend (`simplipy._core`) or any existing API; purely additive.

## 0.3.0 — 2026-06-21 — Rust inline backend + the "numeric" engine line (MAJOR behavior change)

This release rewrites SimpliPy's **inline phase** (`simplify`, the prefix/infix conversions, and
validation) as a compiled Rust extension (`simplipy._core`) and makes the improved **numeric** engine
line the default. The **offline phase** (rule mining + `curve_fit`) stays pure Python.

### ⚠ Breaking — behavior changes vs 0.2.x

The default `simplify`/conversion behavior is now the corrected ("numeric") line. The engine-id is
**unchanged** (`dev_7-3` still identifies the rule-mining parameters: max source length 7, max target
length 3; the `rules.json` asset is byte-identical). What changed is the engine **code**:

1. **Numeric constant folding** is now applied during simplification. All-numeric subtrees evaluate to
   their `f64` result (e.g. `1/0 → float("inf")`, `sqrt(-1) → float("nan")`); the folding fires as a
   fallback after rule matching, so a rule that applies to an all-`<constant>` subtree is tried before
   the subtree is collapsed.
2. **Six conversion bug fixes**, most notably:
   a fractional-power child (`pow1_M`) is no longer silently dropped by `convert_expression`; left/right
   associativity in `infix_to_prefix`/`prefix_to_infix` is corrected (and round-trip-preserving);
   `x**0`, `--5`, `^`-vs-`**` unary-minus, and raw `powN` are handled correctly.
3. **`float("inf")` / `float("-inf")` / `float("nan")` are atomic tokens** in the conversion tokenizer,
   so a folded constant round-trips through prefix↔infix instead of fragmenting.

> Reproducibility: to reproduce results generated with the pre-0.3 engine (e.g. models trained against
> `dev_7-3` under SimpliPy 0.2.x), pin `simplipy<0.3`. The 0.2.x behavior is the frozen anchor; 0.3 is a
> deliberate, documented quality improvement, not a silent drift.

### Robustness (review-driven)

- **ndarray return no longer truncates folded tokens.** `simplify(np.array([...]))` re-infers the result
  string width (keeping the input dtype *kind*); previously a fold that emitted a token wider than the
  input tokens (e.g. `1/0 -> float("inf")` from a `<U1` input) was silently truncated. Affected both the
  Rust-routed and pure-Python paths.
- **Malformed / pathological input raises a clean `ValueError`** instead of an uncatchable abort. A
  malformed prefix (an operator with too few operands) and an excessively long expression
  (`> 4096` tokens, which would recurse deep enough to overflow the stack) are now rejected up front.
  Empty input is still valid (`simplify([]) == []`). Valid inputs are unaffected (verified: 0 diffs vs
  the pre-fix engine across 112,040 corpus comparisons).
- **`simplipy._core` load failures now warn** (`RuntimeWarning`) rather than silently degrading to the
  slower pure-Python path.
- The pow-chain exponent product is computed in `i128` (Python's `prod` is arbitrary-precision), pushing
  the divergence boundary past any reachable exponent.

### Notes

- **Offline mining now uses the Rust inline methods internally** (e.g. `find_rules` calls the improved
  `simplify`/conversions). For Phase-A this is intended; it does not re-mine or change shipped assets,
  but mining a *new* asset under 0.3 would inherit the numeric-line behavior.
- **Folding precision:** the f64 evaluator folds through the platform's system `libm` (not NumPy). On a
  given platform it is byte-identical between the Rust and the pure-Python fallback, but folded constants
  can differ **sub-ULP** from the previous NumPy-based folding. No bit-identity vs 0.2.x folded constants
  is promised (or expected).
- **Pure-Python fallback:** if the compiled `simplipy._core` extension is unavailable, the engine
  transparently falls back to pure-Python implementations (correct, slower). `import simplipy` never
  depends on the extension or on `ctypes`/`libm` being present.

### Build / packaging

- Build backend switched from setuptools to **maturin** (mixed layout: Python at `src/simplipy/`,
  Rust crate under `rust/`, compiled module `simplipy._core`). One abi3 wheel per platform/arch for
  CPython ≥ 3.11.
- **New runtime dependency:** `platformdirs` (resolves the per-user asset cache directory).
