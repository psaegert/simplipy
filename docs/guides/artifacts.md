# Artifacts and assets

Every published engine asset carries `config.yaml` (the operator table and engine
configuration), the rule sets that config names — a 0.14.0 mine produces the full
**TRIPLE**, `rules.json`, `rules_real.json` and `rules_corpus.json` — `mine.yaml` (the exact mine
configuration — each artifact is byte-deterministically reproducible from it with one
`simplipy find-rules` command **at the recorded environment**; what "recorded
environment" means, and why it must be said, is the
[environment qualification](../method/environment-qualification.md) page), and
`rules.json.provenance.json`.
## The triple

One distinct, complete rule set per mode, not a base plus overlays:

| file | mode | contains |
|---|---|---|
| `rules.json` | `Mode.f64` (the default) | every rule the deployed f64 evaluator reproduces |
| `rules_real.json` | `Mode.real` | every rule that is true over ℝ |
| `rules_corpus.json` | `Mode.corpus` | the permissive superset |

`rules.json` keeps its name, so a config written before the triple goes on loading
unchanged: the other two keys are optional, and a mode naming no set of its own serves
the default one. `Mode.real` is the exception — it **fails closed** rather than fall
back, because its only divergence from `f64` is which rules are certified, so serving
it the f64 set would answer a request for mathematical soundness with rules that are
f64-exact and mathematically false.

**The triple is the unit of mining, pinning and distribution.** A mine run is valid only
if all three fall out of it; a partial triple is not shippable. The provenance sidecar
covers the triple as a whole, and so does D29 byte-identity. Rules the mine finds and
can license in no mode are **recorded** in the sidecar's drop census rather than
silently absent.

Verify a shipped triple with `simplipy.verify.verify_triple`, which sweeps each file
against **its own** mode's contract. Cleanliness is per mode: `atanh(tanh t) → t` is
exactly what belongs in `rules_real.json` and would be a defect in `rules.json`.

The provenance
sidecar records how the ruleset came to be: the mine parameters, the core build stamp
(package version plus git revision of the compiled core), the environment (python,
platform, libc, numpy/scipy/mpmath versions, and a `libm_fingerprint` — a digest of a
fixed probe battery evaluated through the deployed folding path, so two sidecars are
comparable-or-not by inspection), the soundness state at mine
time (certificate kill-switch states, every artifact-affecting environment override
recorded verbatim, and the interval layer's fail-closed miss counters), and the
measure fingerprint (the μ constants and probe values with a digest) — so artifacts
mined under different orderings are distinguishable from provenance alone.

## Identity

An installed artifact is identified by its manifest `revision` (the pinned
upstream commit) and per-file `sha256` digests, enforced at install **and at
cache resolution**: a corrupted or swapped cached file makes resolution raise
rather than silently serve, and a partially-installed asset is correctly
treated as not installed.

## Managing assets

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4-3", install=True)   # resolve, installing on demand
```

<!-- docs-example: skip: cache-mutating -- installs into and removes from the user's shared asset cache -->
```python
sp.install("acj-4-3")          # explicit install (alias of asset_manager.install_asset)
sp.get_path("acj-4-3")         # resolve an installed asset to its entrypoint path
sp.list_assets("engine")       # list available and installed engine assets
sp.uninstall("acj-4-3")        # remove (alias of asset_manager.uninstall_asset)
```

Resolution works offline once installed: a network failure falls back to the
last cached manifest copy, so installed assets resolve on a plane; only a cold
cache with no network fails, loudly.

## Compatibility

Compatibility is enforced at load, not documented and hoped for (`simplipy.compat`):
artifacts carry an `engine_generation` pin in `config.yaml` (generation 2 is the AC
engine's clean 23-operator vocabulary), the package carries the allowlist of
generations it serves, and the refusal is mutual and actionable — a generation-1
artifact on 0.12 raises with `pin the legacy package to load it: pip install
"simplipy<0.12"`, and a too-new artifact points at upgrading simplipy. Configs
without a pin are classified by vocabulary: any retired hyper-operator token means
generation 1, so already-published legacy artifacts refuse without republishing.

For a fleet operator, `simplipy.compat.SUPPORTED_ENGINE_GENERATIONS` is the
machine-readable statement of what this package loads, and
`IncompatibleArtifactError` is the exception to catch to distinguish "wrong
artifact generation" from an I/O failure. The full rules live on the
[compatibility policy](../compatibility.md) page.
