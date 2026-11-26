# Artifacts and assets

Every published engine asset carries `config.yaml` (the operator table and engine
configuration), the rule sets that config names. A mine produces one file per mode,
`rules_strict.json` and `rules_permissive.json`, and `mine.yaml` (the exact mine
configuration). Each artifact is byte-deterministically reproducible from the config with one
`simplipy find-rules` command at the recorded environment
(see [environment qualification](../method/environment-qualification.md)).


## The rule files

| file | mode | contains |
|---|---|---|
| `rules_strict.json` | `Mode.strict` (the default) | every rule the deployed float64 evaluator reproduces |
| `rules_permissive.json` | `Mode.permissive` | every rule that holds in permissive: every rule that holds in strict (permissive includes strict), and the rules that agree with the reals' values up to float resolution or give a value where the input has none |

The two files carry their mode in their name. `config.yaml` names its rule files
explicitly and the loader serves whatever the config declares, so a file named `rules.json`
(the convention before 0.15) loads too. The `rules_permissive:` config key is optional (the
key `rules_corpus:` of earlier artifacts is read as it), and a mode naming no set of its own
serves the default (strict) set. A config naming a `rules_real:` set is refused: the real mode
is retired.

From 0.15 the loader refuses a rule file mined before 0.15, by the simplipy version its
provenance sidecar records: 0.15 changed the canonical forms, the measure and the rule judge,
and serves only the rule sets mined for it (`acj2-4-llm`). The error names the version that
mined the file. A rule file without a sidecar is a hand-made set and loads.

Loading is per-mode on request: `SimpliPyEngine.load(..., modes=('strict',))` builds only the
named sets, each compiled before `load` returns — the profile an inference worker runs, at
about 0.45 GB instead of about 0.7 GB with both sets of acj-5-4-llm — and any other
config-named set builds on its mode's first use, once, in about 3–4 s, announced with one log
record. The default
`modes='all'` builds everything. `engine.unload_mode(mode)` is the paired operational RAM knob:
it drops that mode's built structures (about 100 MB per set with acj-5-4-llm) and the mode's next
use lazily reloads them; the default strict set is always present and refuses to unload.

**The two files are the unit of mining, pinning and distribution.** One mine run writes
both, each mined in its own mode (see [How mining works](../rules.md#how-mining-works)),
and the byte-identity promise covers both: a re-mine at the recorded environment
reproduces them. Each file's sidecar records its own mine: the mode, how it accepted
(`acceptance`), per length the judge loop's rounds and exclusions (`per_length`), the
sources the judge loop gave up on (`given_up`), and the
file gates (`gates`: the rows dropped as unlicensed in the mode or as already performed by
its constructor) — a rule the mine finds and the mode does not license is recorded there,
never silently absent.

Verify a shipped artifact with `simplipy.verify.verify_artifact`, which sweeps each file
against its own mode's contract, and a freshly mined file with `simplipy verify-mine <file>
--mode <m> -e <engine>`, which also checks the re-mine gates (twin-shadowed rows, engine-owned laws,
translation, the measure fingerprint, environment switches; see
[Verifying a mine](../rules.md#verifying-a-mine)). Cleanliness is per mode:
`atanh(tanh t) → t` is exactly what belongs in `rules_permissive.json` and would be a
defect in `rules_strict.json`.

The provenance
sidecar records how the ruleset came to be: the mine parameters, the core build stamp
(package version plus git revision of the compiled core) and the digests of the code that
mined it (`code_digest`: the compiled core and the package's sources), the environment (python,
platform, libc, numpy/scipy/mpmath versions, and a `libm_fingerprint` — a digest of a
fixed probe battery evaluated through the deployed folding path, so two sidecars are
comparable-or-not by inspection), the soundness state at mine
time (every `SIMPLIPY_*` variable set during the mine, recorded verbatim — no
artifact-affecting switch is read since 0.15 — and the interval layer's fail-closed miss
counters), and the
measure fingerprint (the μ constants and the probe values, priced in the file's own mode,
with a digest) — so artifacts mined under different orderings are distinguishable from
provenance alone. Each rule file has its own sidecar. Loading reads every
sidecar present and warns, naming the probes, when a recorded probe now prices
differently; a probe added after the file was mined is not recorded and not a mismatch.

## Defined functions

A config may define functions beside its operators. Each has a name, an arity and a definition
written in infix over the config's operators, its arguments written `_0`, `_1`, ...:

```yaml
defined_functions:
  sqrt:
    arity: 1
    definition: "rootn(_0, 2)"
  cot:
    arity: 1
    definition: "1/tan(_0)"
  acot:
    arity: 1
    definition: "atan(1/_0)"   # SymPy's convention: range (-pi/2, pi/2], acot(-1) = -pi/4
```

The config is a floor for what the engine supports, not a ceiling: a defined function is
supported without being mined. `read_infix`, the conversions and `is_valid` read its head and keep
it as written (`to_infix(['sqrt', 'x0'])` is `sqrt(x0)`); `simplify`, `complexity`, `prettify`,
`mask` and every evaluation read its definition, so `simplify('cot(x0)')` answers `1/tan(x0)`,
`simplify('sqrt(x0)')` answers what `simplify('rootn(x0, 2)')` answers, the mined `rootn`, `tan`
and `atan` rules apply, and a value is the definition's, bit for bit. `written_complexity` prices
the head as written. A defined function is not an operator: it is in neither `operator_arity`
nor the operator table the miner, the monitor and the rule files' fingerprints read, so a mine is
the same with or without the section. A definition may call only the config's operators (never
another defined function) and hold numbers, `np.pi`, `np.e` and its arguments, each at least
once; a name that is already an operator or an alias is refused, when the engine is built. Every
config this repository ships defines these three.

## Identity

An installed artifact is identified by its manifest `revision` (the pinned
upstream commit) and per-file `sha256` digests, enforced at install **and at
cache resolution**: a corrupted or swapped cached file makes resolution raise
rather than silently serve, and a partially-installed asset is correctly
treated as not installed.

## Managing assets

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-5-4-llm", install=True)   # resolve, installing on demand
```

<!-- docs-example: skip: cache-mutating -- installs into and removes from the user's shared asset cache -->
```python
sp.install("acj-5-4-llm")      # explicit install (alias of asset_manager.install_asset)
sp.get_path("acj-5-4-llm")     # resolve an installed asset to its entrypoint path
sp.list_assets("engine")       # list available and installed engine assets
sp.uninstall("acj-5-4-llm")    # remove (alias of asset_manager.uninstall_asset)
```

An installed asset resolves without the network: `get_path` (and so `load`) reads the
local copy of the manifest, checks that the asset's files are present and their digests
match, and goes online only for an asset that is not installed or when no local manifest
exists. Only a cold cache with no network fails, loudly. Progress and failures are records
on the `simplipy` logger, not printed output.

## Compatibility

Compatibility is enforced at load, not documented and hoped for (`simplipy.compat`):
artifacts carry an `engine_generation` pin in `config.yaml` (generation 2 is the AC
engine's clean 23-operator vocabulary), the package carries the allowlist of
generations it serves, and the refusal is mutual and actionable — a generation-1
artifact on 0.12 raises with `pin the legacy package to load it: pip install
"simplipy<0.12"`, and a too-new artifact points at upgrading simplipy. Configs
without a pin are classified by vocabulary: any retired hyper-operator token means
generation 1, so already-published legacy artifacts refuse without republishing.

Within a generation, canonical forms change between minor releases, and a rule file
is minimal and correctly ordered only under the forms it was mined with. A config may
therefore carry `requires_simplipy`, a PEP 440 specifier such as `">=0.15,<0.16"`: a
version outside it refuses to load the artifact with `IncompatibleArtifactError` and
names the pin to install. The miner records the specifier for the version that ran
the mine (its minor series) in the provenance sidecar, and the published config
carries it. A pre-release counts as the release it precedes, and a config without the
key loads as before.

For a fleet operator, `simplipy.compat.SUPPORTED_ENGINE_GENERATIONS` is the
machine-readable statement of what this package loads, and
`IncompatibleArtifactError` is the exception to catch to distinguish "wrong
artifact generation" from an I/O failure. The full rules live on the
[compatibility policy](../compatibility.md) page.
