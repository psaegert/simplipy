# The Mining Algorithm (Formal)

This page is the formally typeset specification of `SimpliPyEngine.find_rules` — the
companion to the prose walkthrough in [Creating Rulesets](rules.md). The two algorithms
below are the current implementation's ground truth: Algorithm 1 is the discovery loop,
Algorithm 2 the per-pair equivalence certification (`Equivalent⁺`). Both were typeset at
0.6.0 and are unchanged through 0.9.x (later releases add within-tier progress reporting
and tighten the `simplify` subroutine's soundness gates, but leave the discovery loop and
certification exactly as specified here).

Downloads: [PDF](assets/algorithm/simplipy_mining_algorithm.pdf) ·
[LaTeX source](assets/algorithm/simplipy_mining_algorithm.tex)

![SimpliPy Rule Discovery — Algorithm 1](assets/algorithm/page1.svg)

![Expression Equivalence Check — Algorithm 2 and historical note](assets/algorithm/page2.svg)

## Reading guide: algorithm line → configuration knob

| Algorithm element | Configuration |
|---|---|
| challenges $K$, retries $R$ | `constants_fit_challenges`, `constants_fit_retries` |
| tolerances rtol/atol | `rtol`, `atol` |
| master seed $s$ | `seed` (reproduces the mine byte-for-byte) |
| evaluation matrix $X$ (heavy-tailed mixture) | `X` (`n_samples` in a mining config) + seeded mixture spec |
| evidence floor $m_{\min}$ | `min_informative` |
| candidate fold filter | `candidate_fold_filter` |
| relaxed Kruskal search | `relaxed_kruskal` (default on) |
| universe policy per length | `source_sample_per_length` |
| stage-2 confirmation | `confirm` |
| provenance sidecar | written next to the output as `<output>.provenance.json` |

## Determinism contract

A mine is a pure function of its configuration: one master seed derives the evaluation
matrices and every per-length, per-source, and per-rule seed. Chunked and monolithic
runs are bit-identical (per-source seed = length-seed + index), stage-2 verdicts are
order-independent (content-derived per-rule seeds), and the enumeration is asserted
against an exact counting recurrence on every run — a mine that cannot prove its
universe complete (or its sample valid) aborts rather than under-covering silently.
