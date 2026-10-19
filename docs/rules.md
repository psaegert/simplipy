# Creating Rulesets

SimpliPy's simplification rules are discovered in a one-time offline phase by
exhaustively enumerating candidate expressions and certifying numerically which shorter
expressions are equivalent to longer ones. This page explains the procedure and shows a
complete mining configuration.

## How mining works

`SimpliPyEngine.find_rules` mines three files in one run, one per mode:
`rules_f64.json` (`Mode.f64`), `rules_real.json` (`Mode.real`) and `rules_permissive.json`
(`Mode.permissive`). Each file is mined in its own mode from start to end: the mode reads
each source, the mode's own answer is the mark a target must beat, the mode's numerics
accept a target, and the contract judge's tier must be one the mode licenses. The real and
permissive mines run against the finished f64 set, as their files are served beside it.

**Phase 1 — building the source universe.** All valid prefix expressions up to
`max_source_pattern_length` are enumerated bottom-up, and the enumeration is cross-checked
against an exact counting recurrence.

**Phase 2 — certifying rules, shortest sources first, in each mode.** For each length:

1. **Sources**: each enumerated expression is read in the mode, and one representative per
   canonical class (its canonical projection) is mined; a source the mode reads as a single
   atom yields nothing.
2. **Mark**: the mode's answer for the source (`simplify` in that mode, with the rules of
   the shorter lengths installed). A target must lie strictly below it in the mode's serve
   ordering, the ordering the rewrite pass fires under. An atomic mark mints nothing.
3. **Scan**: the candidates are tried cheapest μ first. Acceptance is decided by the serve
   ordering, so a target may carry more tokens than its source and still certify. A
   candidate passes the licences (no constant introduced into a determined source, no
   special constant absorbed, no instance of a law the engine owns, no defined value
   changed at a pole) before it is judged numerically.
4. **Certify**: the candidate must reproduce the source on a heavy-tailed, seeded evaluation
   matrix across `constants_fit_challenges` re-drawings of the source's constants
   (constants in candidates are fitted by a closed-form solver where possible and a
   restarted optimizer otherwise). A row agrees relatively (`rtol`; zero agrees only with
   zero), rows where the source is undefined may be completed by the target (so
   `x/x → 1` certifies, in the sense of the limit), and a minimum number of informative
   rows is required (`min_informative`). In `f64` the deployed evaluator's values decide,
   except on a row where an operation of either side overflows or underflows: there the
   true value is accepted too, so such a row is re-judged at high precision (the judge
   reads the same event, one definition in the core). In `real` and `permissive` a near
   miss (at most half of the points failing in f64) is re-judged at high precision.
5. **Confirm**: the pick is re-judged on an independent, twice-as-wide matrix with fresh
   seeds; a pick that fails yields to the next candidate.
6. **Judge**: the length's picks go to the contract judge (`simplipy.verify`), each as the
   pattern the file will carry and as every entry the mode serves from it. A pick whose
   judged tier the mode does not license (`f64`: core, f64; `real`: core, real;
   `permissive`: core, f64, real), or that the mode does not serve, is excluded by class
   and only its source is scanned again. The scan is deterministic, so the result is the
   scan that consults the judge per candidate: a target another mode would keep never
   hides one this mode licenses. The loop settles within 8 rounds or fails the mine.
7. **Install**: the licensed picks are deduplicated into wildcard patterns (of two targets
   for one canonical source, the one lower in the mode's ordering stays), rows the mode's
   load would mint anyway from another row's orientation twin are dropped, and the set is
   installed before the next length's marks are taken, so a refused rule never prunes a
   longer source.

After the last length, each mode runs its post-pass: the proposals, if any (one more length
of the judge loop), the symbolic gate on the entries the mode serves, the covered prune
(asked of the mode's answer, on the instances each slot binds in that mode), sort
promotion, the covered prune again, and the file gates — every file spelling and served
entry the mode does not license is dropped, and so is a ground rule the mode's constructor
already performs; both are listed in the sidecar. Then the file and its sidecar are
written.

The permissive mine runs last, and its post-pass is also offered the finished f64 and real
files (`offer_final_rules`, on by default). Permissive promises less than either mode, so
their rules hold there too, and its answer must never cost more than theirs, while a
per-mode mine finds only what its own scan reaches. Every rule of the two files runs through
the proposal lane right after the proposals, certified by permissive's own acceptance, and
enters only where permissive's answer is not already as low.

The procedure is **deterministic**: a fixed `seed` reproduces the three files byte-for-byte,
independent of process, hash randomization, thread count or the judge's process count. One
master seed derives the evaluation matrices and every per-length, per-source and per-rule
seed, and a source's seed and exclusions follow its global index, so a length mined in
shards equals the length mined at once. A mine writes `<file>.checkpoint.json` after every
completed length; a rerun of the same mine (the same parameters, mode, universe, starting
rules and code) resumes after the last completed length, any other mine refuses to start
over it, and the checkpoint is removed once the file is final.

Each file has a **provenance sidecar** (`<file>.provenance.json`) recording every parameter
(`interval_node_budget` included), the derived seeds, the evaluation-matrix specification,
the universe coverage, the mode, how the mode accepted (`acceptance`: tolerances, whether
and at which gate a near miss is re-judged, the licensed tiers, the judge's digest), per
length the classes, picks, judge rounds, exclusions and twin-shadowed rows (`per_length`),
the file gates (`gates`), the digests of the code that mined it (`code_digest`), in the
permissive sidecar the f64 and real files it was offered, pinned by name and sha256, with the
lane's outcomes (`offered`), the
versions that serve it (`requires_simplipy`) and every `SIMPLIPY_*` variable set during the
mine (`soundness.env_overrides`), so a published file is reproducible from its artifact.

## Rule sorts: `_`, `?`, `!`, and `$`

Every placeholder in a shipped rule carries a *sort*: the binding claim its sigil
encodes, enforced by the matcher at apply time:

- **`_i` — any subtree.** The widest claim: the slot binds an arbitrary expression, including subtrees that may evaluate to `nan`.
- **`?i` — variable leaf only.** The narrowest claim: the slot binds a bare variable,
  never a composite subtree, a literal, or `<constant>`.
- **`!i` — finite a.e. subtree.** Binds a variable leaf freely; a composite subtree binds
  only when a match-time certificate proves it defined and finite almost everywhere
  (an adaptive interval analysis over the reals). Fail-closed: what cannot be
  certified is not bound.
- **`$i` — finite a.e. & non-zero a.e. subtree.** Binds a variable leaf freely; a composite subtree
  binds only when the certificate proves it finite **and nonzero** almost everywhere.
  This is the licence behind cancelling a factor against its own inverse
  (`$0 / $0 → 1`, `$0 * inv($0) → 1`): where the bound subtree is zero the source is
  `0/0` while the target says `1`, so nonzero-a.e. is exactly the extra obligation
  beyond `!`. Fail-closed like `!`.

The four claims nest — everything a narrower sort binds, the wider sorts bind too:

    ?i  ⊂  $i  ⊂  !i  ⊂  _i

- `?i` — a bare variable leaf only (narrowest claim);
- `$i` — additionally any composite certified finite **and nonzero** almost everywhere;
- `!i` — any composite certified finite almost everywhere;
- `_i` — any subtree (widest claim).

Sorts exist because a rewrite can be value-sound when a slot holds a variable yet
unsound when it holds a composite carrying poles or infinities into the pattern.
Take `!0 - !0 → 0`: with a bare variable the source is `0` everywhere, but with
`log(x)` in the slot the source is `nan - nan` on the entire negative half-line while
the target says `0` — so the certificate refuses `log(x)`, which is not finite almost
everywhere, and the rule never sees it.

These sort gates define the default **`f64`** apply-time contract. `Mode.permissive`
(see [Soundness modes](guides/simplify.md#soundness-modes)) relaxes them together — every `!`/`$`
placeholder then binds any subtree with the certificate skipped which recovers extra
reductions for training-data canonicalization at the cost of equivalence. Mined rules are
always certified under the sound gates regardless. The mode only changes how they bind
at apply time.

## Running a mine

```sh
simplipy find-rules -e "path/to/my_config.yaml" -c "path/to/create_my_config.yaml" -o "path/to/my_rules_f64.json" -v --reset-rules
```

- `-e` is the path to the engine configuration file to use as a backend
- `-c` is the path to the configuration file containing parameters for finding rules
- `-o` is the output path for the collected rules
- `-v` enables verbose output
- `--reset-rules` will start with an empty rule set, otherwise it will append to the existing rules loaded with the engine

The published `acj-5-4-llm` artifact is this machinery at its shipped scale:
complete sources through length 5, targets through length 4, plus the
LLM-augmented proposal lane, certified by the same gates as the mined rules.
It is mined as a triple and independently gate-verified — 138,651 rules in
`rules_f64.json`, 133,294 in `rules_real.json`, 139,299 in
`rules_permissive.json`, all three modes clean, with 13 real-tier constant
folds relocated out of the f64 set by the gate (each present in the real and
permissive sets). The exact mine configuration ships with the asset as
`mine.yaml`, and `simplipy find-rules -e config.yaml -c mine.yaml -o rules_f64.json -v --reset-rules`
reproduces the triple byte-for-byte at the recorded environment. The
configuration format, annotated:

```yaml
# Special symbols available as expression leaves (beyond the dummy variables).
# <constant> is the wildcard that matches any fitted constant; the integer span
# matches the downstream vocabulary.
extra_internal_terms: ['<constant>', '(-10)', '(-9)', '(-8)', '(-7)', '(-6)', '(-5)',
                       '(-4)', '(-3)', '(-2)', '(-1)', '0', '1', '2', '3', '4', '5',
                       '6', '7', '8', '9', '10', 'np.e', 'np.pi',
                       'float("inf")', 'float("-inf")', 'float("nan")']

# Dummy variables for pattern slots: ceil(max_source_pattern_length / 2) — beyond
# that, a source can only repeat variables it already carries.
dummy_variables: 2

# Maximum number of tokens in a source (left-hand side) expression
max_source_pattern_length: 4

# Maximum number of tokens in a target (replacement) expression. The cap bounds
# only the candidate LIBRARY; acceptance is decided by mu(target) < mu(source),
# so a target may carry more tokens than its source and still certify.
max_target_pattern_length: 4

# Rows of the evaluation matrix (heavy-tailed mixture, drawn from `seed`)
n_samples: 1024

# Re-drawings of source constants per certification (guards against
# coincidental matches at particular constant values)
constants_fit_challenges: 16

# Optimizer restarts per constant fit (accounts for non-convergence)
constants_fit_retries: 16

# Acceptance tolerances (relative / absolute, per row). The rtol window is
# sensitivity-measured on the published mines: 1e-9 admits a tanh-saturation
# near-miss family (residuals ~2e-10, not sound), 1e-7 admits the next shelf
# AND degrades literal resolution; 1e-11 holds.
rtol: 1.0e-11
atol: 1.0e-12

# Evidence floor: minimum informative rows per certification (null = built-in)
min_informative: null

# Master seed: reproduces the entire mine, byte-for-byte
seed: 42

# Re-verify every mined rule on an independent evaluation matrix
confirm: true

# Tighten (rather than skip) sources the rules found so far already shorten
relaxed_kruskal: true

# Post-mine stages: the covered prune and sort promotion
prune: covered
promote_sorts: true

# Optional: the interval layer's subdivision node budget (default 400,000). An
# exhausted search refuses its candidate; raise it when the sidecar's
# interval_undecided counters flag systematic exhaustion. Recorded in the sidecar.
# interval_node_budget: 800000

# Offer the permissive mine the finished f64 and real files through its proposal
# lane (the default); its sidecar pins both files.
offer_final_rules: true

# Optional: LLM/human-proposed rules, certified against the mined state at the
# end of the run (see "LLM-proposed rules" below). Paths starting with ./ are
# resolved relative to this config file. The published artifact's LLM lane is
# this key, enabled in its shipped mine.yaml.
# proposals: ./llm_proposals.json
```

With this operator set the complete universes through length 4 hold 413,772 sources
(29 / 493 / 13,427 / 399,823 at lengths 1–4), the enumeration is cross-checked
against the counting recurrence, and the length-4 tier dominates: a full mine of
this cell completes in about 25 minutes on a 16-core desktop host. Cost climbs
steeply with either cap. Length 5 — the published artifact's source tier — adds
roughly 13 million sources, and the target cap sizes the candidate library every
source is judged against.
Progress, per-length rule counts, judge rounds and universe coverage are printed as
the mine advances, and the checkpoint is updated after every completed length.

## Verifying a mine

`simplipy verify-mine` checks one mined file against the re-mine gates in its own mode:

```sh
simplipy verify-mine rules_real.json --mode real -e config.yaml
```

| gate | holds when |
|---|---|
| `twin_shadow` | no row is one the mode's load mints anyway from another row's orientation twin |
| `engine_owned_law` | no rule respells one canonical state or states the distributive law as the search factors (a sign pull such as `neg sin _0 -> sin neg _0` still passes) |
| `unlicensed` | every file spelling and every served entry carries a judged tier the mode licenses |
| `preempted` | no ground rule is one the rule-less engine already performs, such as a ground fact it folds (`log (-3) → nan`) |
| `translation` | the mode's load subsumes and drops nothing |
| `fingerprint` | the sidecar's measure record prices every probe as this engine does |
| `env_switches` | the mine ran, and the check runs, with no `SIMPLIPY_*` switch set |
| `offered` | (`permissive` only) the sidecar pins the f64 and real files the mine was offered, and each lies beside the file with the pinned sha256 |

It prints one line per gate and exits non-zero when any gate fails (`--report` writes the
full report as JSON). `simplipy.mining.verify_mine` is the same check from Python.

## LLM-proposed rules

Mining guarantees completeness where enumeration (or sampling) reaches, but the
source universe grows by roughly a factor of thirty per token so the
*mathematically salient* long identities sit far beyond any enumerable horizon:
`sin²x + cos²x → 1` is a nine-token source in this vocabulary
(`+ pow sin x 2 pow cos x 2`). A language model,
by contrast, can name such identities directly. SimpliPy therefore supports a complementary channel:

**an LLM proposes candidate source expressions; the engine certifies them with the
exact same gates as mined rules.** Proposals only ever add source expressions, so a
certified proposal is precisely as sound as a mined rule. The LLM's correctness is
never trusted, only its taste in candidates. Wrong proposals cost about a CPU-second
each and are rejected.

### The reproduction path: `proposals:` in the find-rules config

The channel is wired into the miner itself, so reproducing a mined-plus-proposed
ruleset is **one config and one command**: add a `proposals:` key to the find-rules
YAML and run `find-rules` as usual.

```yaml
# ... the mining configuration from above ...
proposals: ./llm_proposals.json
```

The proposals file is a JSON object with a `"proposals"` key whose entries carry a
`"source"` (a prefix token list) and an optional `"target"` (a prefix token list, used
as the certification *hint*); any other keys (`why`, `family`, `tier`, ...) are
ignored. A bare list of such `{source, target?}` objects is accepted too.

```json
{"proposals": [
  {"source": ["+", "pow", "sin", "x0", "2", "pow", "cos", "x0", "2"],
   "target": ["1"]}
]}
```

```sh
simplipy find-rules -e "path/to/my_config.yaml" -c "path/to/create_my_config.yaml" -o "path/to/my_rules_f64.json" -v --reset-rules
```

After the mining length loop completes and before the optional prune, every proposal is certified against
the just-mined rule state with the exact machinery of the mine: the same evaluation
matrices, constant challenges, tolerances, and seeds derived from the master `seed`.
A proposal the mined rules already shorten is skipped exactly like an
already-reducible source, and a certified proposal joins the ruleset through the same
deduplication path (shortest target per canonical source).

Each proposal ends in exactly one of four outcomes — `certified` (joined the
ruleset), `already_covered` (the mined rules already shorten it), `rejected`
(invalid, no shorter equivalent found, or failed numerical verification), or
`duplicate` (certified, but canonically identical to an earlier certified proposal)
and the provenance sidecar records the proposals file, its sha256, and the
per-outcome counts. The pass is deterministic: proposals are processed in file order
with content-derived per-proposal seeds, so editing the file never rerolls the
certification of untouched proposals, and two runs from the same seed and file are
byte-identical.

### Programmatic alternative: `certify_rules`

To certify proposals against an already-built engine without re-mining, use the
`certify_rules` API directly:

```python
import simplipy as sp
from simplipy.utils import deduplicate_rules

engine = sp.SimpliPyEngine.load("acj-5-4-llm", install=True)

proposals = [
    ["+", "pow", "sin", "x0", "2", "pow", "cos", "x0", "2"],           # sin^2 + cos^2
    ["*", "2", "*", "sin", "x0", "cos", "x0"],                         # 2 sin cos
    ["-", "pow", "+", "x0", "x1", "2", "pow", "-", "x0", "x1", "2"],   # (x+y)^2 - (x-y)^2
]
hints = [None, None, ["*", "4", "*", "x0", "x1"]]  # optional target suggestions

extra_terms = ['<constant>', '0', '1', '(-1)', 'np.pi', 'np.e']  # target vocabulary
certified = engine.certify_rules(proposals, hints,
                                 extra_internal_terms=extra_terms, verbose=True)
# certified: a list of (source, target, 'minimal' | 'verified') triples

dummies = ["x0", "x1"]
engine.simplification_rules = deduplicate_rules(
    engine.simplification_rules + [(s, t) for s, t, _ in certified], dummies,
    engine=engine, rule_mode="f64")   # keys on the mode's internal form, not the spelling
engine.compile_rules()
```

For each proposal, `certify_rules` checks validity, searches the complete candidate
library for a certified-**minimal** target (a source the engine already reduces is
still searched — coverage is decided by the search, with the engine's own result as
the ordering mark to beat), and re-verifies the winning pair on an independent
evaluation matrix. If no library-sized target exists, an optional per-proposal
**hint** is verified instead — sound, but marked `'verified'` rather than `'minimal'`
since no shorter form was ruled out. Targets certified this way may have any
length; what is enforced is the serve-time reduction ordering — the target must
price strictly below the source's mark. Finally, every accepted pair is
re-judged by the independent symbolic verifier (`verify_ruleset`, the
precision-stability discriminator) and KILLs are dropped — the instrument that
separates true exact identities from impostors below every usable numeric tolerance,
such as `tanh(exp(e)) → 1`, false by a stable `1.37e-13`. The mining channel above
runs this same chain per proposal; the only difference is bookkeeping (the mine's
matrices and seeds, and the provenance record).

### How we use this (and what to expect)

The rule packs for the earlier `acj-4-3-llm` engine asset were proposed by two
independent language models under the frozen prompt spec shipped with the asset,
prompted with the
exact grammar (the operator inventory, the leaf symbols including the `<constant>`
wildcard, prefix arity rules, and a source-length window) and split across identity
families — trigonometric, hyperbolic, exponential/logarithmic, algebraic cancellations,
powers and roots, constant arithmetic, multi-variable symmetry, and special values —
with each family asked to enumerate systematically and to include operand-order and
factored variants as separate entries (rule matching is syntactic, so distinct tree
spellings of one identity are distinct rules).

Observed on the published batch (recorded in its provenance sidecar): 169 proposals
from the two models; 73 certified against the freshly mined state — 54 served after
the covered prune and sort promotion, 19 dropped by those later sound stages — 26
already covered by the mine, 70 refused by the gates, and none lost to numerical
confirmation. Generating a wave takes minutes; certification runs at roughly 1–2
seconds per proposal, and the certified rules land at source lengths beyond the
mine's enumerated horizon.

## Replicating a large ruleset end to end

1. **Propose** long identities with an LLM of your choice, prompting with your engine's
   grammar; ask for systematic family-by-family enumeration and tree-spelling variants.
   Save the proposals as a JSON file (either schema above).
2. **Mine + certify** with the configuration above, including its `proposals:` key —
   one command (`simplipy find-rules ...`) runs the mine and then certifies every
   proposal against the freshly mined state (roughly 1–2 seconds per proposal).
   The complete 4→4 mine is a half-hour run on a 16-core host; the provenance
   sidecar records everything needed to reproduce the run from its seed,
   including the proposal file's sha256 and per-outcome counts.
3. **Post-process** (optional): `prune-covered-rules` / `resolve-rules`, below.

Steps 2–3 are deterministic given the seed and the proposals file; step 1 is
inherently not, so keep the proposals file under version control — it, not the model
transcript, is the reproducible artifact.

# Post-processing Rules

Two commands refine an existing rule set loaded from an engine and write the
result to a JSON file (they do not modify the installed asset in place):

```sh
# Remove rules that the remaining rules already cover compositionally
simplipy prune-covered-rules -e "acj-5-4-llm" -o "path/to/pruned_rules.json" -v

# Replace <constant> placeholders with concrete numeric values in all-numeric rules
simplipy resolve-rules -e "acj-5-4-llm" -o "path/to/resolved_rules.json" -v
```

- `-e` is the engine name (e.g. `acj-5-4-llm`) or a path to an engine configuration file
- `-o` is the output path for the post-processed rules
- `-v` enables verbose progress output

## Pruning rulesets

One prune shrinks a ruleset without changing what the engine can simplify:

- **`SimpliPyEngine.prune_covered_rules(rule_mode=...)`** (CLI: `prune-covered-rules`,
  which prunes the default set) is the compositional prune: it removes any rule of the
  named mode's set whose effect the remaining rules achieve on their own. A rule is
  covered only if every instantiation variant of its source still reaches, in that mode,
  a state at most the corresponding target in the mode's serve ordering: each slot is
  instantiated as a distinct variable leaf, `<constant>` is kept literal so native
  constant folding cannot fake coverage, wide slots (`_`/`!`/`$` sigils) are additionally
  probed with composite subtrees, since leaf-only instantiation under-tests wide-sort
  claims, and in `permissive`, where every slot binds any subtree, each slot is also
  probed with the non-finite literals (`$0 / $0 → 1` takes `inf/inf` to `1` there, which
  permissive's constructor takes to `nan`).
  Rules are processed in source-length waves, longest first; each wave is tentatively
  removed and then repaired to a fixpoint against an engine rebuilt from the kept
  rules, so the result is deterministic for a given rule list and greedy: valid,
  not necessarily minimal.

The prune is also available at the end of a mine through the `prune` parameter:
`find_rules(prune='covered')`, or a `prune: 'covered'` key in the mine configuration.

Note that the covered prune trades ruleset size against one-step reachability: a
covered rule's source still simplifies at least as short, but possibly through
intermediate rewrites inside the same `simplify` call. When set-level closure
guarantees matter, re-verify the pruned engine on a benchmark corpus and check that no
meaningful fraction of outputs got longer.

# Managing Engine Assets

List the engines (and test-data assets) available on Hugging Face and which are
already installed locally:

```sh
simplipy list --type engine
# --- Available engine assets ---
# - acj-5-4-llm [installed] Complete AC-judged rule mine of the clean 23-operator
#                         vocabulary (sources to length 5, targets to length 4) plus the
#                         LLM-augmented proposal lane, mined as a triple and independently
#                         gate-verified ... Pairs with simplipy >= 0.14.
# - acj-4                 Complete AC-judged rule mine ... (sources and targets to length 4), ...
# - base                  Bare 23-operator engine configuration (no rules): the clean-vocabulary
#                         starting point for fresh mining. Pairs with simplipy >= 0.12.
# - ...                   (older assets remain listed for older installs; generation-1 artifacts
#                         refuse to load on >= 0.12)

simplipy list --installed        # only assets already downloaded
```

Install or remove an asset by name:

```sh
simplipy install acj-5-4-llm     # download an asset from Hugging Face (--force to reinstall)
simplipy remove acj-5-4-llm      # remove a locally installed asset
```

The same operations are available from Python (this is also what the engine loader uses under the hood):

<!-- docs-example: skip: cache-mutating -- uninstall/install rewrite the user's shared asset cache; each call is covered by tests/test_asset_manager.py -->
```python
import simplipy as sp

sp.install("acj-5-4-llm")       # download an asset from Hugging Face
sp.uninstall("acj-5-4-llm")     # remove a locally installed asset
sp.get_path("acj-5-4-llm", install=True)  # resolve a local path, installing if needed
```

`sp.SimpliPyEngine.load("acj-5-4-llm", install=True)` installs the engine
on demand as part of loading.