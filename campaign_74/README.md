# The 7-4 rule-mine campaign — procedure, artifacts, reproducibility

End-to-end procedure for producing the next simplipy engine asset ("7-4": sources
mined to length 7+, targets certified-minimal to length 4, plus the LLM identity
packs). Everything runs on simplipy `868bbe7` (0.5.0 + fold-filter + affine-recall
fixes + provenance); every step below names its script, inputs, outputs, and its
measured duration.

## Reproducibility contract

- **Deterministic steps** (everything except LLM proposal *generation*): fully
  reproducible from `seed=42` — same seed => byte-identical rulesets, verified
  across processes, PYTHONHASHSEED, thread counts, AND across machines
  (valkyrie/solomon smoke: byte-identical). Every mined artifact carries a
  provenance sidecar (params, derived seeds, X spec, universe coverage).
- **LLM proposal generation is NOT seed-reproducible** (model sampling); it is
  reproducible at the ARTIFACT level: the exact prompts are versioned in
  `llm_prompts/`, the raw proposal batches are committed
  (`llm_proposals_raw.json`, `llm_proposals_round2.json`), and everything
  downstream of those files is deterministic given a rules snapshot.
- Certified LLM rules are exactly as sound as mined rules (identical gate +
  independent wide-X confirmation); provenance tags them as a proposal stratum,
  and `hint_verified` rules additionally as possibly-non-minimal.

## Procedure + measured durations

| # | step | script / tool | compute | measured duration | output |
|---|------|--------------|---------|-------------------|--------|
| 0 | build + deploy simplipy wheel | `maturin build --release`, scp, pip | valkyrie 1 core | ~1 min | manylinux wheel @ pinned sha |
| 1 | go/no-go calibration (complete L<=4 mine + 1,500-source filtered-vs-unfiltered parity + 2,000-source L5 cost anchor) | `calibrate_74_l4l5.py` | solomon 30 thr | **24 min** | `calib_out_solomon/CALIB_RESULTS.json` (decision: 1.377 CPU-s/src, parity EXACT -> GO) |
| 2 | stratified feasibility probe (cf/cb x L5/6/7 cost matrix) | `stratified_cost_probe.py` | solomon 28 thr | **~1 min** | `calib_out_solomon/STRATA_RESULTS.json` (boundary: all L<=5; only cb at L6; nothing L7) |
| 3 | production mine, phase A: complete L2-4 | `mine_74_production.py` (find_rules verbatim) | solomon 30 thr | **40 s** | 6,849 rules + provenance sidecar |
| 4 | phase A5: complete L5, chunked (43 x 500k sources, per-chunk checkpoint) | same driver | solomon 30 thr | **~4.3 days** (~2.4 h/chunk) | ~+230k rules (projected), `found_A5_L5.jsonl` |
| 5 | phase B: exhaustive L6-const-bearing (87,925,893 sources, slice-generated, count-validated) | same driver | solomon 30 thr | **~2.8 days** | `found_B_L6cb.jsonl` + barrier integrate |
| 6 | phase C: stratified sampled L6-cf / L7-cb / L7-cf, saturation stop (<5 new x 2 batches of 250k) | same driver | solomon 30 thr | **<=3.6 days** (budgets 36/24/30 h, less if saturated) | saturation curves in campaign provenance |
| 7 | LLM proposal wave (family-diversified agents, grammar-constrained) | `llm_prompts/round*_proposal_workflow.js` (Workflow tool) | LLM API | **~15 min/wave** | `llm_proposals_*.json` (591 + 1,863 raw) |
| 8 | proposal certification (syntax -> Kruskal -> minimal-<=4 target -> wide-X confirm) | `validate_llm_proposals.py` | valkyrie 8 low-prio cores | **~1.5 s/proposal** (591 -> 3 min; 1,863 -> ~45 min) | `llm_round*/certified.json` (192 + 611 certified) |
| 9 | long-target lift for `no_rule_found` (minimal-<=5 library tier, then hint-verified unlimited tier) | `certify_long_targets.py` | valkyrie 8 low-prio cores | lib build ~15 min + **~30-60 s/proposal** (161 -> ~2-3 h) | `llm_round2_long/certified_long.json` |
| 10 | phase D: integrate certified proposal packs at the final barrier (dedup, shortest-target-wins) | driver extension — **TODO** | solomon | minutes | final `rules_74.json` |
| 11 | telemetry: pull ruleset + paired l_max 0-7+None sweep vs dev_7-3 on the paper's 4,096-skeleton corpus | `pull_and_bench_74.py` via `bench_loop.sh` | valkyrie 1 pinned core, nice 10 | **~11 s-few min/checkpoint**, every 5 h | `bench74/bench_timeline.jsonl` + per-pull pickles |
| 12 | SymPy reference leg (one-time; comparable token-equivalent complexity metric, 10 s cap) | `sympy_reference.py` | valkyrie 1 pinned core | **~45 min** | `sympy_reference.pkl` (median 64.2 ms; complexity INFLATES 35.4 -> 38.5) |
| 13 | figures: paper-style ECDFs + progress dashboard | ECDF snippet (this dir) / dashboard — **TODO(notebook)** | valkyrie | seconds | `ecdf_74_checkpoint1.png` |

Total wall-clock for the mine proper (3-6): **~10-12 solomon-days**; the entire
LLM channel (7-9): **~4 h** end to end.

## Key certified numbers so far

- Calibration: complete L<=4 = 6,849 rules in 39.5 s; fold-filter drops 374,031 of
  566,280 candidates (66%); 55-88x per-source speedup at identical decisions.
- LLM round 1: 591 proposals -> 192 certified (74 additive at L6-L7); hint accuracy 90%.
- LLM round 2 (length-uncapped): 1,863 -> 611 certified (461 at L6+, 175 with
  sources of length 8-13); 900 already covered; only 7 (0.4%) failed confirmation;
  hint accuracy 84%.

## Known gaps (state honestly)

1. Phase D (pack integration) not yet in the driver.
2. The trajectory dashboard + a user-executable notebook for the ECDF/timeline
   figures are pending (`ecdf_74_checkpoint1.png` was produced by an inline
   script; the code should live in a notebook per repo convention).
3. Automated operand-order variant EXPANSION at certification is designed but not
   implemented (round-2 prompts requested variants from the agents instead).
4. LLM waves ran via session Workflow scripts; re-running them requires the
   Workflow harness (prompts preserved verbatim in `llm_prompts/`).
