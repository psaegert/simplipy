# Artifacts and assets

Every published engine asset carries `config.yaml` (the operator table and engine
configuration), the rule sets that config names. A 0.14.0 mine produces the full
**TRIPLE**, `rules_f64.json`, `rules_real.json` and `rules_permissive.json`, and `mine.yaml` (the exact mine
configuration). Each artifact is byte-deterministically reproducible from the config with one
`simplipy find-rules` command at the recorded environment
(see [environment qualification](../method/environment-qualification.md)).


## The triple

| file | mode | contains |
|---|---|---|
| `rules_f64.json` | `Mode.f64` (the default) | every rule the deployed f64 evaluator reproduces |
| `rules_real.json` | `Mode.real` | every rule that is true over ℝ |
| `rules_permissive.json` | `Mode.permissive` | the permissive superset |

The three files carry their mode in their name. Artifacts published before this
convention (acj-4 among them) name the f64 file `rules.json`; they load unchanged,
because `config.yaml` names its rule files explicitly and the loader serves whatever
the config declares. The `rules_real:` and `rules_permissive:` config keys are optional
(artifacts published before the rename declare the latter as `rules_corpus:`, which the
loader reads unchanged),
and a mode naming no set of its own serves the default (f64) set.

`Mode.real` is the exception to that fallback: on an artifact without a real set,
`simplify(mode=Mode.real)` **raises** instead of quietly using the f64 set. The f64
set contains rules that floating point reproduces exactly but that are false as
mathematics — `asin(1e-8) → 1e-8` is bit-identical in f64 and wrong over ℝ by the
cubic term. A caller selecting `Mode.real` is asking precisely for those rules to be
absent, so the fallback would serve them the one thing they opted out of.

Loading is per-mode on request: `SimpliPyEngine.load(..., modes=('f64', 'permissive'))`
builds only the named sets eagerly — the profile a training worker runs, inference plus
corpus canonicalisation, at 469 MB instead of 569 MB on acj-5-4-llm — and any other
config-named set builds on its mode's first use, once, announced with one log line. The
default `modes='all'` builds everything, exactly as always. `engine.unload_mode(mode)`
is the paired operational RAM knob: it drops that mode's built structures (about 130 MB
per used set on acj-5-4-llm) and the mode's next use lazily reloads them; the default
f64 set is always present and refuses to unload.

**The triple is the unit of mining, pinning and distribution.** A mine run is valid only
if all three fall out of it; a partial triple is not shippable. The provenance sidecar
covers the triple as a whole, and so does the byte-identity promise: a re-mine at the
recorded environment reproduces all three files. Rules the mine finds and
can license in no mode are recorded in the sidecar's drop census rather than
silently absent.

Verify a shipped triple with `simplipy.verify.verify_triple`, which sweeps each file
against its own mode's contract. Cleanliness is per mode: `atanh(tanh t) → t` is
exactly what belongs in `rules_real.json` and would be a defect in `rules_f64.json`.

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

engine = sp.SimpliPyEngine.load("acj-5-4-llm", install=True)   # resolve, installing on demand
```

<!-- docs-example: skip: cache-mutating -- installs into and removes from the user's shared asset cache -->
```python
sp.install("acj-5-4-llm")      # explicit install (alias of asset_manager.install_asset)
sp.get_path("acj-5-4-llm")     # resolve an installed asset to its entrypoint path
sp.list_assets("engine")       # list available and installed engine assets
sp.uninstall("acj-5-4-llm")    # remove (alias of asset_manager.uninstall_asset)
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
