# Changelog

## 0.11.0 — 2026-07-30

### Added
- **Engines are picklable.** `SimpliPyEngine` now supports `pickle` round-trips,
  `copy.deepcopy`, and `multiprocessing` spawn contexts. The compiled core
  (`simplipy._core.Engine`) has no serialization surface, but it is derived state: the
  pickle carries the construction recipe (operator config + rule list, including rules
  added at runtime via `compile_rules`), and unpickling rebuilds the core exactly as
  `__init__` would — realization modules first, then the compiled engine. Previously any
  engine crossing a process boundary (for example inside a data-generation catalog handed
  to spawn workers) failed with `TypeError: cannot pickle 'simplipy._core.Engine' object`.

## 0.10.1 — 2026-07-26

### Added
- **Per-candidate verdict trail for the proposal channel.** `find_rules(proposals=...)` now
  writes `proposals.trail` into the provenance sidecar: one entry per proposal, in file order,
  as `{source, target, verdict, stage, certificate}`. The aggregate tally it accompanies cannot
  be audited on its own -- "93 rejected" never says which candidate died at which gate, so a
  reviewer cannot separate a correctly killed hallucination from a wrongly killed identity.
  `stage` names the deciding gate: `vocabulary` (token outside the alphabet, or malformed),
  `covered` (the mined rules already shorten the source), `search` (no library target and no
  verifiable hint), `confirm` (failed independent stage-2 re-verification), `merge` (certified
  but folded away as a duplicate), or `accepted`. No certification semantics change.

## 0.10.0 — 2026-07-26

### Added
- **The certificate algebra: structural null-measure certificates.** A new predicate family in
  the interval engine — `zero_set_null` (identity-theorem witness for entire-analytic
  compositions plus structural recursion), `nonfinite_null` (pole-set tracking through the
  domain-restricted unaries), `finite_nonzero_ae` (= the two combined: the regular domain of
  multiplication), and `positive_ae` (v1) — closing the class of binding-dependent side
  conditions. `finite_ae` now certifies through the structural path as well as box subdivision.
- **The `$` wildcard sort: certified multiplicative cancellation.** `$N` binds composite
  subtrees gated by `finite_nonzero_ae`, licensing `A/A -> 1`, `A·inv(A) -> 1`, and
  `0/A -> 0` exactly where they are sound (`cosh(x)/cosh(x) -> 1`; `asin(x)/asin(x)` and
  `0/0`-bearing trees refuse). Ships with `judge_bang_mult` (the promotion bar on the
  finite-nonzero atom lattice) and four ladder seeds; deployed artifacts are unchanged until
  the next production mine.

### Fixed
- **Promotion oracle non-termination (B4).** `_literal_spans` could loop forever with unbounded
  memory when the operator table was empty or contained unknown ops (unvisited `end[]` entries
  sent the scan backwards). A termination guard plus a fail-fast `RuntimeError` in `prefold`
  when the oracle is unconfigured close both the hang and the silent misconfiguration path.

### Removed
- **BREAKING: the `max_pattern_length` parameter of `simplify` (and `apply_rules`) is gone.**
  Rule application always considers every pattern in the loaded artifact; the scan window is
  the ruleset's own longest pattern, an intrinsic property rather than a caller knob. All
  deployed callers already passed `None`; passing the keyword now raises `TypeError`.

## 0.9.1 — 2026-07-25

### Added
- **Within-tier mining progress.** `find_rules(verbose=True)` now reports progress WHILE a length
  tier is mining, not only at the end. The Rust miner publishes a `(sources_done, sources_total)`
  counter (`Engine.mining_progress()` on the compiled core) that a daemon monitor polls, printing
  `Length L: X/N sources (P%) | R src/s | ETA T | elapsed E | RSS M` at 20 / 60 / 180 s and then
  every `SIMPLIPY_MINE_PROGRESS_INTERVAL` seconds (default 600). A long tier -- the length-5+
  combinatorial wall where a mine can sit for days -- is no longer an opaque wait: its rate, ETA,
  and memory are visible as it runs. Zero effect when `verbose=False`.

## 0.9.0 — 2026-07-24

A new simplify kernel. `simplify` is now a best-first cancellation SEARCH instead of a fixed
cancellation order; masking (numeric literals -> `<constant>`) is separated from the
equivalence loop so representation is no longer entangled with rewriting; and the constant-fold
fallback is brought under the same finite-a.e. certificate the rules and cancellation already
enforce.

### Added
- **Cancellation search** (`Engine::simplify_search`). `simplify` no longer commits to one
  cancellation order. Cancel is non-confluent -- taking candidate A can destroy candidate B,
  and which choice ends shortest depends on what the ruleset can fold -- so the kernel now
  SEARCHES a small move graph instead of guessing. State = an expression; the moves are a flat
  choice set (apply the rules pass, or cancel any one qualifying candidate); the answer is the
  shortest state visited. Every state is a.e.-equivalent to the input and the input is state
  zero, so a result can never be longer than its own input. Best-first by
  length with a visited-set, pre-filled with the greedy result so a truncated budget can never
  do worse than the plain fixpoint. The node budget (`SIMPLIPY_SEARCH_BUDGET`, default 24;
  0 restores the plain greedy fixpoint and its speed exactly) bounds a rare heavy tail -- the
  median expression expands 2 nodes and 53% have no cancellation candidate at all, but the p99
  is ~186. The default is the measured elbow of the returns curve on the 64k v23.0 prior: below
  24 an extra microsecond buys ~14 output tokens, above it ~2.7, and by 256 only 0.7. On that
  corpus the search shortens ~2860/1000/350 expressions (2-1/3-2/4-3) that no previous version
  could reach, at ~2.3x the simplify wall time; raise the budget for offline corpus
  canonicalisation, lower it for latency.
- **Symmetric `neg`/`inv` cancellation** (BEHAVIOUR CHANGE). The cancellation unit already
  EMITTED the class inverses but never CONSUMED them: a leaf under its own class inverse was
  shielded, so `x * inv(x)` and `x + neg(x)` did not cancel through the inverse. Each inverse is
  now region-continuing in its own class (`neg` additive, `inv` multiplicative), symmetric with
  the emit path, and there is exactly one region shape -- the asymmetry is not preserved behind
  a flag. Consequence on sparse rule sets: a handful of expressions come out ONE token longer
  than in 0.7.x, because the old opaque treatment happened to leave a sign arrangement those
  rule sets can re-fold and the connected treatment does not (6 of 65536 on the 2-1/3-2-class
  `3-2` set; none on `4-3` or richer, and never longer than the input). Cancelling through the
  class inverse is worth far more than it costs: the same corpus gets 1005 expressions SHORTER
  on `3-2` and 350 on `4-3`.
- **Soundness `Mode` (`simplify(expr, mode=Mode.SOUND)`).** A single ordinal soundness axis,
  exposed as `simplipy.Mode`. `Mode.SOUND` (the default) is equivalence-preserving and
  idempotent -- the deployed inference/scoring path, byte-identical to the historical default.
  `Mode.LOSSY` trades soundness for recall: every rule placeholder binds any subtree (the
  `!`-sort finite-a.e. certificate is skipped) AND the constant-fold's finiteness gate is
  relaxed, so a non-finite-a.e. subtree such as `<constant>/0` collapses to `<constant>` too.
  `Mode.LOSSY` is for training-corpus canonicalisation ONLY -- the training data is generated
  FROM the simplified form (target == data) -- and must never run on an inference or scoring
  path. The decided full ordering is `EXACT <= SOUND <= AE <= LOSSY`; only `SOUND` and `LOSSY`
  are implemented.

### Changed
- **BREAKING: masking is separated from `simplify`.** Masking numeric literals to the generic
  `<constant>` placeholder is a REPRESENTATION step for downstream models that cannot consume
  literals, not an equivalence-preserving rewrite. `simplify` no longer masks; it is now the
  equivalence loop only (search + sort to a fixpoint), sound and idempotent by construction. The
  `mask_elementary_literals` parameter of `simplify` is removed; use the new terminal
  `Engine.mask()` (relabel literals + one sort, no re-simplify) on `simplify`'s output when
  placeholders are needed, and do NOT re-`simplify` a masked expression. Entangling masking with
  the search fixpoint was also the sole cause of the former non-idempotence: masking mints a free
  `<constant>` from a structural literal (`x - x -> 0 -> <constant>`), and re-searching could
  fold that constant into a denominator, dropping the reachable `C = 0` the input required.

### Fixed
- **Constant folding respects the finiteness certificate.** The constant-fold fallback collapsed
  any subtree whose operands are all `<constant>` or finite literals to a free `<constant>`, on a
  syntactic test that assumed finite operands compose to a finite result. That is false at a pole
  (`<constant> / 0` is +-inf/nan for every constant), so a structural zero reaching a denominator
  could be folded to a finite free constant -- unsound (it revives a structurally zeroed term).
  This was the one search edge that skipped the finiteness certification the rule matcher and the
  cancellation already enforce. The fold now consults the same value-set analysis and collapses to
  `<constant>` only when the subtree has a positive-measure finite part, keeping
  unbounded-but-finite-a.e. folds (`1/C`, `tan C`, `cosh(C + 5)`) and refusing non-finite-a.e.
  ones (`C / 0`, `C * inv(0)`).

## 0.8.0 — 2026-07-22

Makes the public package produce the BEST rule sets from one command: mining now natively
promotes every rule to the strongest sound sort, and a public verification API can
independently gate + monitor any rule set.

### Added
- **Native sort promotion** (`simplipy.promotion`; `find_rules(promote_sorts=True)`, CLI
  config key `promote_sorts`). After mining and pruning, every rule (mined + proposed) is
  re-certified at the stronger sorts and shipped at the strongest SOUND one: `_` (arbitrary
  subtree), `!` (certified-finite subtree, enforced by a match-time defined-and-finite-a.e.
  certificate), or `?` (variable-leaf, the fallback). Five stages: a pointwise exact bar on
  an atom lattice, a const-bearing witness-map bar, an exact-arbiter overturn of finite-draw
  demotions, a subsumption/derivability refund, and the `_`→`!`→`?` ladder with a
  moving-spike structural refusal. Promotion is fail-safe: an uncertifiable rule stays `?`
  and loses only composite-subtree recall, never soundness. This recovers the simplification
  power that conservative `?`-only rulesets leave unrealized (a mined ruleset promoted this
  way matches a far larger legacy engine's reduction at a fraction of the rules), because the
  limiting factor was sort generality, not the number of rules.
- **Independent verification API** (`simplipy.verify`): `verify_ruleset` gates a rule set by
  judging every rule at its own symbolic trigger points under an arbitrary-precision contract
  evaluator (eight-bucket classification, 100% coverage by construction); `monitor_ruleset`
  runs the deployed engine over an adversarial+sampled corpus and attributes any
  deployed-value violation to the responsible rule under an independent high-precision
  evaluator; `verify_rule` gives a single-rule verdict. Deliberately implemented
  independently of the compiled core so it cross-checks the miner rather than echoing it.
  Both carry poison self-tests.

### Changed
- `mpmath` and `scipy` are now runtime dependencies (offline mining + verification only; the
  inline simplify path remains the compiled core).

## 0.7.1 — 2026-07-21

### Fixed
- **CLI `find-rules` honors the full config**: the `prune` and `relaxed_kruskal` keys in a
  find-rules YAML were silently ignored (`prune` fell back to the `--prune` CLI flag,
  i.e. `False`; `relaxed_kruskal` to its default), so a config declaring
  `prune: covered` produced an unpruned artifact that did not match its own claims.
  Both keys are now forwarded (a config `prune:` key takes precedence over `--prune`),
  and the CLI rejects unknown config keys fail-closed, naming the offending key — a
  mis-spelled or unsupported key is an error, never a silent no-op (`confirm_rules:`
  was such a silent no-op; the honored key is `confirm:`).
- The provenance sidecar now records `relaxed_kruskal` alongside the other mine
  parameters.

## 0.7.0 — 2026-07-21

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
- **Special-point certification phase in the miner** (`rust/battery.rs`): rule
  certification now additionally checks every accepted (source, candidate) pair at a
  battery of symbolic special points (0, ±1/2, ±1, ..., ±π/2, π, e) per variable, sweeps a
  battery of special source-constant values, and snaps fitted witness constants to nearby
  integers / half-integers before the domain-preservation gate. This closes three gaps
  where a rule could certify on random evaluation matrices yet be unsound at points
  deployed expressions actually reach: a fitted exponent like `2.9999999999999996` is NaN
  on the negative half-line and hides the positive-measure domain extension its snapped
  witness `3.0` creates (`exp(log(pow3 x)) → pow(x, C)`); a rule can be false exactly at a
  symbolic coincidence (`pow(sin x, inf) → 0` is 1 at x = π/2); and a pattern-bound
  source constant reaches special values in deployment (`pow(cos C, inf) → 0` at C = 0).
  Rows where deployed f64 evaluation diverges from contract semantics are re-judged at
  three fixed precision rungs (50/120/250 decimal digits) with symbolic coordinates
  rendered at each precision, so stable limit completions (`x/x → 1`) keep certifying
  while precision-dependent cancellation seams are rejected fail-closed.

### Changed
- **Stricter mining certification (see the special-point phase above)**: `find_rules`
  rejects rule families that earlier versions could mine. Rulesets mined with
  `simplipy <= 0.6.0` may therefore contain rules that 0.7.0 refuses to re-mine;
  re-mining under 0.7.0 is the recommended migration.
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
