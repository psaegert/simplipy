# Creating Rulesets

SimpliPy's simplification rules are not hand-written: they are **mined** — discovered by
exhaustively enumerating candidate expressions and certifying, numerically, which longer
expressions are equivalent to shorter ones. This page explains the procedure and shows a
complete mining configuration.

## How mining works

`SimpliPyEngine.find_rules` runs two phases:

**Phase 1 — building the source universe.** All valid prefix expressions up to
`max_source_pattern_length` are enumerated bottom-up, and the enumeration is cross-checked
against an exact counting recurrence — the two must agree exactly, or the mine aborts.
Lengths whose complete universe is too large to enumerate can instead be represented by a
uniform sample from the complete universe (`source_sample_per_length`); coverage is always
reported, and the drawn sample is validated per run.

**Phase 2 — certifying rules, shortest sources first.** For each source expression:

1. **Prune**: if the rules found so far already shorten the source, it is skipped.
2. **Scan**: otherwise the source is compared against every candidate replacement, in
   order of increasing length, so the first match is a *minimal* target. The candidate
   library is built once per mine; variable-free candidates of length ≥ 2 are excluded
   (`candidate_fold_filter`, default on) — a provably behavior-preserving optimization,
   since any source they could match is already matched by the length-1 `<constant>`
   candidate.
3. **Certify**: a candidate matches only if it reproduces the source's values on a
   heavy-tailed, seeded evaluation matrix — across `constants_fit_challenges` re-drawings
   of the source's constants, with constants in candidates fitted by a deterministic
   closed-form solver where possible and a restarted optimizer otherwise. Rows where the
   source is finite must agree within `rtol`/`atol`; rows where the source is undefined
   may be completed by the replacement (so `x/x → 1` certifies, in the sense of the limit),
   and a minimum number of informative rows is required (`min_informative`).
4. **Confirm**: every mined pair is re-verified on an independent, twice-as-wide
   evaluation matrix with fresh seeds before it enters the rule set.
5. **Deduplicate**: rules are canonicalized into wildcard patterns, keeping the shortest
   target per source.

The whole procedure is **deterministic**: a fixed `seed` reproduces the ruleset
byte-for-byte, independent of process, hash randomization, or thread count. Alongside the
output, a **provenance sidecar** (`<output>.provenance.json`) records every parameter,
derived seed, the evaluation-matrix specification, and per-length universe coverage, so a
published ruleset is reproducible from its artifact alone.

## Running a mine

```sh
simplipy find-rules -e "path/to/my_config.yaml" -c "path/to/create_my_config.yaml" -o "path/to/my_rules.json" -v --reset-rules
```

- `-e` is the path to the engine configuration file to use as a backend
- `-c` is the path to the configuration file containing parameters for finding rules
- `-o` is the output path for the collected rules
- `-v` enables verbose output
- `--reset-rules` will start with an empty rule set, otherwise it will append to the existing rules loaded with the engine

A complete mining configuration:

```yaml
# Special symbols available as expression leaves (beyond the dummy variables).
# <constant> is the wildcard that matches any fitted constant.
extra_internal_terms: [
  '<constant>',
  '0',
  '1',
  '(-1)',
  'np.pi',
  'np.e',
  'float("inf")',
  'float("-inf")',
  'float("nan")'
]

# Number of dummy variables (null = derived from max_source_pattern_length)
dummy_variables: null

# Maximum number of tokens in a source (left-hand side) expression
max_source_pattern_length: 7

# Maximum number of tokens in a target (replacement) expression.
# Targets are never sampled: the candidate library must stay complete,
# or the minimality guarantee is lost.
max_target_pattern_length: 4

# Universe policy: lengths whose complete universe is infeasible to enumerate
# are drawn uniformly from the complete universe instead. With the operator
# set above, lengths through 5 are enumerable; 6 and 7 are sampled.
source_sample_per_length:
  6: 1000000
  7: 1000000

# Rows of the evaluation matrix (heavy-tailed mixture, drawn from `seed`)
n_samples: 1024

# Re-drawings of source constants per certification (guards against
# coincidental matches at particular constant values)
constants_fit_challenges: 16

# Optimizer restarts per constant fit (accounts for non-convergence)
constants_fit_retries: 16

# Acceptance tolerances (relative / absolute, per row)
rtol: 1.0e-9
atol: 1.0e-12

# Master seed: reproduces the entire mine, byte-for-byte
seed: 42

# Re-verify every mined rule on an independent evaluation matrix
confirm: true

# Exclude variable-free candidates from the library (behavior-preserving speedup)
candidate_fold_filter: true
```

Complete enumeration through length 5 covers tens of millions of sources and is a
multi-day run on a modern many-core CPU; the sampled lengths add time proportional to
their sample sizes. Progress, per-length rule counts, and universe coverage are printed
as the mine advances, and the output file plus its provenance sidecar are updated after
every completed length.

# Post-processing Rules

Two commands refine an existing rule set loaded from an engine and write the
result to a JSON file (they do not modify the installed asset in place):

```sh
# Remove explicit rules that are already subsumed by wildcard-pattern rules
simplipy prune-rules -e "dev_7-3" -o "path/to/pruned_rules.json" -v

# Replace <constant> placeholders with concrete numeric values in all-numeric rules
simplipy resolve-rules -e "dev_7-3" -o "path/to/resolved_rules.json" -v
```

- `-e` is the engine name (e.g. `dev_7-3`) or a path to an engine configuration file
- `-o` is the output path for the post-processed rules
- `-v` enables verbose progress output

# Managing Engine Assets

List the engines (and test-data assets) available on Hugging Face and which are
already installed locally:

```sh
simplipy list --type engine
# --- Available Assets ---
# - dev_7-3         [installed]  Development engine 7-3 for mathematical expression simplification.
# - dev_7-2                      Development engine 7-2 for mathematical expression simplification.

simplipy list --installed        # only assets already downloaded
```

Install or remove an asset by name:

```sh
simplipy install dev_7-3         # download an asset from Hugging Face (--force to reinstall)
simplipy remove dev_7-3          # remove a locally installed asset
```

The same operations are available from Python (this is also what the engine loader uses under the hood):

```python
import simplipy as sp

sp.install("dev_7-3")     # download an asset from Hugging Face
sp.uninstall("dev_7-3")   # remove a locally installed asset
sp.get_path("dev_7-3", install=True)  # resolve a local path, installing if needed
```

`sp.SimpliPyEngine.load("dev_7-3", install=True)` installs the engine
on demand as part of loading.