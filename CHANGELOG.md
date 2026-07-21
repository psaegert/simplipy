# Changelog

## 0.7.0 (unreleased)

Real-semantics pow alignment and the removal of the faithful dev_7-3 reproduction line:
the package now ships ONE engine line.

### Added
- **LLM-proposal channel in the miner**: `find_rules(..., proposals=...)` accepts a path
  to a proposals JSON (consolidated `{"proposals": [...]}` artifact or a bare list of
  `{source, target?}` objects) or the equivalent in-memory object. After the mining
  length loop and before the optional prune, every proposal runs the exact
  `certify_rules` chain against the just-mined rule state with the mine's evaluation
  matrices, tolerances and master-seed-derived, content-derived per-proposal seeds;
  certified proposals join the ruleset through the same `deduplicate_rules` path. The
  find-rules YAML forwards a `proposals:` key, and the provenance sidecar records the
  proposals file, its sha256, and per-outcome counts
  (`certified` / `already_covered` / `rejected` / `duplicate`). This makes a
  mined-plus-proposed ruleset reproducible from one config and one command.

### Changed
- **Pow alignment (real semantics)**: `pow` at a `-inf` base with a finite non-integer
  exponent now evaluates to NaN (real semantics) across the constant folder, the operator
  realizations, and the interval engine. Magnitude-step cells at infinite exponents are
  unchanged.
- **`engine_id` now reports the package version** (e.g. `simplipy-0.7.0`) instead of the
  frozen reference id `dev_7-3`, so provenance records identify the exact engine build.

### Removed
- **BREAKING: the faithful dev_7-3 reproduction line is removed.** The `fold` parameter of
  `_core.Engine.simplify` / `apply_rules` / `prune_explicit` is gone (the numeric
  constant-folding behavior is now the only one), the legacy quirk-preserving conversion
  variants are gone (the corrected conversions are now exposed under the plain
  `prefix_to_infix` / `infix_to_prefix` / `convert_expression` / `parse` names, which is
  what the Python API always routed), and the frozen byte-identical-to-0.2.15 parity
  fixtures and reference constants (`FAITHFUL_ENGINE_ID`, `REFERENCE_SIMPLIPY_VERSION`,
  `REFERENCE_SIMPLIPY_COMMIT`) are gone. The parity regression test is re-baselined
  against the 0.7.0 aligned engine line. To reproduce v23.0/dev_7-3-era behavior
  byte-for-byte, install `simplipy<=0.6.0`.

## 0.6.0

A performance overhaul of the simplify hot path (byte-identical outputs), sorted rule
placeholders with match-time certificates, ruleset pruning and observability tools, and
a round of rule-mining improvements and constant-folding coherence fixes. The shipped
`dev_7-3` asset is unchanged.

### Added
- **Sorted rule placeholders** (`_` / `?` / `!`): every placeholder in a rule carries a
  *sort* — the binding claim its sigil encodes, enforced by the matcher at apply time.
  `_i` binds any subtree; `?i` binds only a bare variable leaf; `!i` binds a variable
  leaf freely, but a composite subtree only when a match-time certificate proves it
  defined and finite almost everywhere (an adaptive interval analysis over the reals;
  fail-closed — what cannot be certified is not bound). Sorts let a rule that is
  value-sound only for certified operands make the wider claim without giving up
  soundness. Newly mined rulesets emit sorted patterns; existing assets (explicit and
  `_`-wildcard rules) behave exactly as before. See the rule-authoring guide
  (`docs/rules.md`) for details.
- **Covered-rule pruning** (`SimpliPyEngine.prune_covered_rules`, CLI
  `simplipy prune-covered-rules`): removes any rule — wildcard-pattern rules included —
  whose effect the remaining rules already achieve compositionally. A rule is covered
  only if every instantiation variant of its source (distinct variable leaves, literal
  `<constant>`, composite probes for wide placeholder slots) still simplifies to at
  most the length of the corresponding target without it. Deterministic batch
  remove-and-repair, longest sources first; greedy (the result is valid, not
  necessarily minimal). Complementary to `prune_redundant_rules`, which removes only
  explicit rules shadowed by pattern rules under an equality criterion. `find_rules`
  gains `prune='covered'` to run this prune after discovery instead of the
  redundant-rule prune.
- **Simplify observability**: `SimpliPyEngine.simplify_counters()` /
  `reset_simplify_counters()` expose the process-global simplify hot-path counters
  (calls, iterations, pattern attempts/fires, certificate calls/hits, and coarse
  per-stage nanosecond accounting) for profiling a batch.
- **Formal mining-algorithm documentation**: a formally typeset specification of
  `find_rules` (the discovery loop and the equivalence certification), with PDF/LaTeX
  downloads, an algorithm-line-to-configuration reading guide, and the determinism
  contract (`docs/algorithm.md`).
- **Relaxed Kruskal search** (`mine_one_length(..., relaxed_kruskal=...)`): with
  `True`, a source the current rules already shorten is still searched, with the
  candidate bound tightened to its simplified length (only targets strictly shorter
  than what `simplify` reaches are accepted). Relaxed rules are one-step shortcuts that
  are capability-redundant wherever the reduct's own rule exists; for sampled
  universes, re-mining the simplify-fixpoints of skipped sources generalizes better (a
  fixpoint rule covers every source that reduces to it). The Rust-level parameter
  defaults to the strict skip, but `find_rules(relaxed_kruskal=True)` is the
  Python-side production default (a capability comparison across four mined corpora,
  ~780k expressions total, found the relaxed mine strictly shorter on 735 of them and
  never longer, at ~+6% mine cost).
- **High-precision rescue for const-free equivalence** (`astro-float`, pure Rust, 1024-bit):
  f64 remains the fast pre-filter, but a const-free candidate that fails the tolerance on
  only a small fraction of its binding rows (<= 25%, <= 256 rows) is re-evaluated on
  exactly those rows at 1024 bits and re-judged under the same generic-equivalence
  semantics. True identities whose f64 round-trip loses precision now certify --
  `atanh(tanh(x)) -> x` (f64 error reaches 2e-1 for |x| in [10, 19.6]) and, via domain
  extension, `tanh(atanh(x)) -> x`, both previously unminable at the strict rtol=1e-9.
  Every f64 accept is unchanged; plateau falsities keep failing their near-corner rows and
  never reach the escalation. Applies to the mine scan and (through the same code path)
  stage-2 confirmation and `certify_rules`.
- **Candidate library minimization** (`find_rules(candidate_fold_filter=True)`, default on):
  variable-free candidates of length >= 2 are excluded from the mining candidate library.
  This is provably behavior-preserving (any source they could match is already matched by
  the length-1 `<constant>` candidate, which is scanned first) and removes the bulk of the
  constant-fitting work: measured 55-88x faster per source on the dev configuration at
  identical mining decisions. The library reports `n_candidates`/`n_filtered`.
- **Provenance sidecar**: `find_rules(output_file=...)` now writes
  `<output>.provenance.json` recording all parameters, derived seeds, the evaluation-matrix
  specification, and per-length universe coverage, so a mined ruleset is reproducible from
  its artifact alone.
- **Per-run sampler validation**: sampled sources are checked for universe membership
  (length, vocabulary, well-formedness) on every run.

### Changed
- **NaN literals propagate through constant folding**: a subtree with a
  `float("nan")` operand now folds directly to `float("nan")` (every operator maps a
  NaN operand to NaN; the IEEE `pow` edge cases whose value does not depend on the NaN
  operand are excluded), and `inf`/`nan` literals are never absorbed into a fittable
  `<constant>` — a `<constant>` claims a finite value exists to fit, and NaN is not
  one. This is operator-table knowledge, not sampling: it also keeps the miner from
  certifying rules that rewrite an everywhere-NaN expression, for which no finite
  evidence can exist.
- **Constant-fit restart seeds are order-independent** (a pure function of the source seed,
  candidate, and challenge instance). Mines remain byte-reproducible for a fixed seed; the
  output of a given seed may differ from 0.5.0 on marginal nonlinear fits.

### Performance
- **Certificate economics**: `!`-sort certificates are evaluated only after a
  *completed* syntactic match (instead of during every candidate attempt), memoized
  per call and in a generational per-engine cache that never stops memoizing. This
  replaces the old hard-capped certificate cache, which silently stopped memoizing
  once full and degraded at scale (measured on a 65,536-expression training-prior
  benchmark: hit rate falling from 99.3% to 36.6% and per-expression engine cost from
  ~600 µs to ~7,400 µs once the cap was hit). Certificate verdicts are unchanged.
- **Fixpoint memoization**: the simplify fixpoint memoizes whole cancel/rules passes
  and rule-normal subtrees per call, so the convergence-confirming iteration and
  unchanged subexpressions cost hash lookups instead of re-scans.
- **Interned token representation**: the hot path runs on interned token ids with a
  per-engine property table (and a per-call overlay for unknown tokens), reducing
  allocations per simplify call by ~20×.
- Net effect on a 65,536-expression training-prior benchmark: large
  certificate-bearing rulesets simplify ~59× faster than 0.5.0, with byte-identical
  outputs; certificate-free rulesets run ~2.3× faster.
- **Mining**: the source expression is evaluated once per source (per challenge
  instance) and shared across the whole candidate scan, instead of per
  (candidate, challenge, sign-combo); a const-free source collapses to a single
  challenge instance (identical targets add no evidence and only multiplied fit
  flakiness).

### Removed
- The `scipy` runtime dependency: dead since the 0.5.0 native constant-fit cutover
  (no `scipy` import remained anywhere in the package).
- 14 dead `SimpliPyEngine` attributes left over from the removed pure-Python simplify
  engine, each verified to have no remaining consumers: `operator_inverses`,
  `inverse_base`, `inverse_unary`, `inverse_binary`, `unary_mult_div_operators`,
  `commutative_operators`, `realization_to_operator`, `operator_precedence_compat`,
  `max_power`, `max_fractional_power`, `connection_classes`, `operator_to_class`,
  `connection_classes_inverse`, `connection_classes_hyper`,
  `binary_connectable_operators`.
- `simplipy.utils.leaf_value`: unused Python duplicate of the engine's single
  leaf-value table.
- Two undocumented, dev-only methods on the compiled core (`eval_bench_resident`,
  `classify_linearity`) and, on the Rust side, the unused `match_pattern` wrapper and
  the unused `criterion` dev-dependency.

### Fixed
- **Term cancellation respects non-finite algebra**: with an empty ruleset (or rule
  application disabled), `0/0`, `nan/nan`, and `inf/inf` previously canceled to `1`,
  and `inf - inf` / `nan - nan` to `0` — all wrong answers (the true value is NaN).
  Cancellation assumes the group axioms, so a leaf now registers as cancellable only
  in the connection classes where they hold: nonzero finite literals in both, `0` in
  the additive class only, `nan`/`±inf` in neither. Variables and nonzero-finite
  literals cancel exactly as before.
- **One leaf-value table everywhere**: constant folding (`_try_fold_constants` /
  `evaluate_constant_subtree`), the miner's all-constant short-circuit, and the offline
  tape evaluator now share a single leaf resolution (Rust `numeric::leaf_value`):
  numeric literals, `np.pi`, `np.e`, parenthesized literals such
  as `(-1)`, and the `float("...")` tokens. Previously the folder and short-circuit used
  the numeric-string predicate only, so e.g. `cosh(np.pi)` never folded exactly and
  `(-1)/cosh(np.pi)` simplified no further than `(-1)/<constant>` even though every leaf
  has a known value. The `<constant>` absorption collapse now admits any FINITE-valued
  leaf (still never inf/nan literals: non-finite algebra remains the explicit rules'
  domain).
- **Generic-equivalence vacuity in the constant-bearing fit arm**: an equivalence-check
  instance whose source evaluates nowhere finite (e.g. the sign-combo `C = 0`
  instantiation of a constant-denominator source, `y = k/0`) binds no rows and is now
  treated as domain-extendable — matching the declared semantics and the const-free arm —
  instead of falling into the underdetermined-fit bail that returned `False` and vetoed
  the candidate. This single instance previously killed every constant-denominator rule:
  `x0*<constant>/<constant> -> x0*<constant>` was unminable, and (with the short-circuit
  gap above) so was the whole `k/<constant> -> <constant>` absorption family. Evidence
  accounting is unchanged: vacuous instances contribute zero rows toward
  `min_informative`.
- **Affine constant-fit recall with fast-growing bases**: fits of the form `C0*f(x)+C1`
  and `f(x)+C1` for `f` in `exp`/`cosh`/`sinh`/`pow3`-`pow5` were spuriously rejected at
  exactly-true constants. The closed-form solver now uses row weights matched to the
  acceptance tolerance, overflow-safe column equilibration, iterative refinement, an exact
  feasibility decision for the bare-`<constant>` candidate, and integer snapping for
  exact-cancellation cases. The acceptance gate itself is unchanged (recall-only fix;
  non-equivalences are still rejected).
- Fold-filter variable detection is correct for configurations with more than 32 variables.
- `min_informative` defaults are floored at 1 for very small evaluation matrices.

### Docs
- 0.5.0 entry corrected: the mine evaluation matrix's log-uniform tier is 1e-4..1e3, not
  1e-6..1e6.

## 0.5.0 — 2026-07-11 — sound rule-mine checker + Python mining mirror removed

The pre-0.5.0 checker shipped ~8.3% defective
rules in the dev_7-3 sample (52/840), including ~5,125 vacuous all-NaN wildcard rules such as
`asin(cosh(_0)) -> nan` (false at 0, where the source is pi/2).

### Fixed (checker soundness)
- **Vacuous equal_nan acceptance closed.** Certification now requires at least `min_informative`
  (default `n_rows / 8`) SOURCE-FINITE evidence rows, accumulated across challenge instances; a
  precomputed candidate-finite count serves as a fast necessary-condition gate for const-free
  candidates.
- **Tolerances tightened** `rtol` 1e-5 -> 1e-9, `atol` 1e-8 -> 1e-12 (0 borderline rules in an
  840-rule review sample; tanh/exp saturation towers are no longer "equal" to constants).
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
  causes, both introduced by earlier hardening of the same path: a GLOBAL trace-scaled Tikhonov ridge biased the
  intercept, and the normal-equations solve squared the condition number on the wide-magnitude X
  so an exact affine relationship's intercept came out ~5e-9 off (past rtol). Now the affine path
  solves by Householder-QR least-squares (no `A^T A`, no ridge -- works at cond(A), not its square),
  and the mine X's log-uniform tier is capped at 1e3 (from 1e6): 1e3 still exercises every
  saturation and f64 overflow the tier is FOR, but 1e6 wrecked the fit conditioning. Confirmed by
  probe: 0/30 -> 64/72 affine cases certify, 0 false-accepts; the residual misses have constants
  spanning >1e6.
- **Log-linear recall fix**: for `pow(<constant>, g)` candidates, only a closed-form log-space
  ACCEPT short-circuits; an imprecise `Some(false)` solve (its ~1e-10 base error amplified past
  rtol by large exponents on the heavy-tailed X) now SEEDS the LM restart instead of rejecting,
  restoring the declared const-bearing policy ("accept iff constants EXIST that fit"). Surfaced by
  extensive adversarial re-verification of the generic-equivalence change (which found NO soundness
  defects).

### Changed
- **Perf**: the source expression is now evaluated once per source (per challenge instance) and
  shared across the whole candidate scan, instead of per (candidate, challenge, sign-combo); a
  const-free source collapses to a single challenge instance (identical targets add no evidence
  and only multiplied fit flakiness).

### Removed (BREAKING)
- **The pure-Python mining mirror is gone**: `find_rule_worker`, `exist_constants_that_fit` and the
  fork/SharedMemory pool (with the `n_workers` parameter). It duplicated `rust/worker.rs`/`fit.rs`
  and repeatedly desynced from them (e.g. diverging IEEE inv/div semantics). `find_rules` now requires the
  compiled core and raises `RuntimeError` on a bare engine; parallelism is rayon's
  (`RAYON_NUM_THREADS`).

### Compatibility
- Shipped rule assets (`dev_7-3` etc.) are unchanged by this release; they remain the frozen
  reference that downstream training pipelines pin for reproducibility. A re-mine with the
  hardened checker is a separate, explicitly-versioned artifact.


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
- Internal offline-mining scaffolding (mine grid and validation drivers) moved out of the
  released repo.

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

### Robustness

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
