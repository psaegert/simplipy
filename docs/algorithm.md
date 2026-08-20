# The mining algorithm (formal)

This page is the typeset specification of `SimpliPyEngine.find_rules`, and the
companion to the prose walkthrough in [Creating rulesets](rules.md). Algorithm 1
is the discovery loop; Algorithm 2 is the per-pair equivalence certification
(`Equivalent⁺`).

**Scope.** The typeset pages specify the kernel of the 0.6.0–0.9.x line. The
discovery loop's shape is unchanged since — complete source universes per
length, constant fitting, pairwise certification, covered-pruning — and
Algorithm 2's numeric certification still describes what the miner does per
pair. Four things the pages predate are now part of the pipeline:

- Certification runs against the current engine under the unified simplicity
  measure, with stronger gates than the ones typeset here: interval-corroborated ground
  folds, the special-constant policy, class preservation, and sort promotion.
  These are specified in
  [The simplification engine (formal)](formal.md).
- Acceptance rides the serve-time reduction ordering: a target must strictly
  descend the simplicity measure below the engine's own mark for the source
  ([formal.md](formal.md) §5).
- Matched `<constant>` slots on constant-free sources are resolved to literal
  tokens at the 1024-bit arbiter, behind a structural respell guard.
- Every mined rule passes stage-2 confirmation on an independent matrix, and the
  surviving set is covered-pruned and sort-promoted before it ships.

Read this page as the loop's skeleton and its historical certification. The
prose walkthrough in [Creating rulesets](rules.md) describes the current
pipeline in full; a re-typeset against the current miner is not yet done.

Downloads: [PDF](assets/algorithm/simplipy_mining_algorithm.pdf) ·
[LaTeX source](assets/algorithm/simplipy_mining_algorithm.tex)

![SimpliPy Rule Discovery — Algorithm 1](assets/algorithm/page1.svg)

![Expression Equivalence Check — Algorithm 2 and historical note](assets/algorithm/page2.svg)

## Reading guide: algorithm line → configuration knob

| Algorithm element | Configuration |
|---|---|
| challenges $K$, retries $R$ | `constants_fit_challenges`, `constants_fit_retries` |
| tolerances rtol/atol | `rtol`, `atol` |
| master seed $s$ | `seed` (reproduces the mine byte for byte) |
| evaluation matrix $X$ (heavy-tailed mixture) | `X` (`n_samples` in a mining config) + seeded mixture spec |
| evidence floor $m_{\min}$ | `min_informative` |
| candidate fold filter | `candidate_fold_filter` |
| relaxed Kruskal search | `relaxed_kruskal` (default on) |
| universe policy per length | `source_sample_per_length` |
| stage-2 confirmation | `confirm` |
| provenance sidecar | written next to the output as `<output>.provenance.json` |

## Determinism contract

A mine is a pure function of its configuration. One master seed derives the
evaluation matrices and every per-length, per-source and per-rule seed. Chunked
and monolithic runs are bit-identical (a per-source seed is the length seed plus
the index), stage-2 verdicts are order-independent (per-rule seeds are derived
from content), and the enumeration is asserted against an exact counting
recurrence on every run: a mine that cannot prove its universe complete, or its
sample valid, aborts rather than under-covering silently.
