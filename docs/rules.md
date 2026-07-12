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

## LLM-proposed rules

Mining guarantees completeness where enumeration or sampling reaches, but the source
universe grows so fast with expression length (billions at length 7, and worse beyond)
that uniform sampling essentially never draws the *mathematically salient* long
identities — `sin²x + cos²x → 1` is a length-7 source with a ~0.03% chance of appearing
in a million-draw sample. A language model, by contrast, can name such identities
directly. SimpliPy therefore supports a complementary channel:

**an LLM proposes candidate source expressions; the engine certifies them with the
exact same gates as mined rules.** Proposals only ever *add* source expressions, so a
certified proposal is precisely as sound as a mined rule — the model's correctness is
never trusted, only its taste in candidates. Wrong proposals cost about a CPU-second
each and are rejected.

```python
import simplipy as sp
from simplipy.utils import deduplicate_rules

engine = sp.SimpliPyEngine.load("dev_7-3")

proposals = [
    ["+", "pow2", "sin", "x0", "pow2", "cos", "x0"],   # sin^2 + cos^2
    ["mult2", "*", "sin", "x0", "cos", "x0"],          # 2 sin cos
    ["*", "-", "x0", "x1", "-", "x1", "x0"],           # (x-y)(y-x)
]
hints = [None, None, ["neg", "pow2", "-", "x0", "x1"]]  # optional target suggestions

extra_terms = ['<constant>', '0', '1', '(-1)', 'np.pi', 'np.e']  # target vocabulary
certified = engine.certify_rules(proposals, hints,
                                 extra_internal_terms=extra_terms, verbose=True)
# -> [(source, target, 'minimal' | 'verified'), ...]

dummies = ["x0", "x1"]
engine.simplification_rules = deduplicate_rules(
    engine.simplification_rules + [(s, t) for s, t, _ in certified], dummies)
engine.compile_rules()
```

For each proposal, `certify_rules` checks validity, skips sources the engine already
reduces, searches the complete candidate library for a certified-**minimal** target,
and re-verifies the winning pair on an independent evaluation matrix. If no
library-sized target exists, an optional per-proposal **hint** is verified instead —
sound, but marked `'verified'` rather than `'minimal'` since no shorter form was ruled
out. Targets certified this way may have any length, as long as they are shorter than
the source.

### How we use this (and what to expect)

The rule packs for the upcoming engine asset were proposed by Claude, prompted with the
exact grammar (the operator inventory, the leaf symbols including the `<constant>`
wildcard, prefix arity rules, and a source-length window) and split across identity
families — trigonometric, hyperbolic, exponential/logarithmic, algebraic cancellations,
powers and roots, constant arithmetic, multi-variable symmetry, and special values —
with each family asked to enumerate systematically and to include operand-order and
factored variants as separate entries (rule matching is syntactic, so distinct tree
spellings of one identity are distinct rules).

Observed over ~2,400 proposals: generating a wave takes minutes; certification runs at
roughly 1–2 seconds per proposal; about half of all proposals are true identities the
engine already covers (rejected as redundant); under 1% fail numerical verification
outright; and roughly a third certify — the majority at source lengths that uniform
sampling would never reach, with the model's own target suggestion matching the
certified-minimal target about 85–90% of the time.

## Replicating a large ruleset end to end

1. **Mine** with the configuration above (`simplipy find-rules ...`). Complete
   enumeration through length 5 plus one-million-source samples at lengths 6 and 7 is
   roughly a week on a modern 16-core CPU; the provenance sidecar records everything
   needed to reproduce the run from its seed.
2. **Propose** long identities with an LLM of your choice, prompting with your engine's
   grammar; ask for systematic family-by-family enumeration and tree-spelling variants.
3. **Certify** the proposals with `certify_rules` against the freshly mined engine
   (minutes per thousand proposals), and merge the accepted pairs with
   `deduplicate_rules`, which keeps the shortest target per source.
4. **Post-process** (optional): `prune-rules` / `resolve-rules`, below.

Steps 1 and 3–4 are deterministic given the seed; step 2 is inherently not, so keep the
accepted proposal list under version control — the certified pairs, not the model
transcript, are the reproducible artifact.

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