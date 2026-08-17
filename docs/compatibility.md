# Compatibility policy

This page states what is stable, what may change, and how you will find out.
It binds to the declared public surface — the `__all__` declarations
introduced in 0.13.0.

## What is stable

The public API of `simplipy` is exactly the set of names declared in
`__all__` — at the package root and in each public module. Those names, their
signatures, and their documented behaviour follow the versioning rules below.

Names *reachable* but not declared (helpers, `_`-prefixed modules,
`simplipy._core`, anything imported into a module's namespace from elsewhere)
carry no stability promise, whether or not a documentation generator has ever
rendered them. Being rendered in the API reference is not, and has never
been, a stability promise; 0.13.0 narrows the rendered surface to the
declared one.

The declared surface has three tiers:

- **Public** — stable under this policy.
- **Power-user** (`simplipy.verify`, `simplipy.promotion`, `simplipy.mining`,
  `codify`, `SimpliPyEngine.code_to_lambda`) — same stability rules, but each
  carries a documented sharp edge (environment qualification of verdicts,
  one-miner-per-process, compile-to-source semantics). Read the caveat before
  depending on behaviour details.
- **Internal** — everything else; may change or vanish in any release.

## Versioning

While the major version is 0: breaking changes to declared names happen only
at a **minor** version bump (0.x → 0.x+1), are listed in the CHANGELOG under
"Changed"/"Removed", and — where a replacement exists — the old spelling
warns for at least one minor release before removal. Patch releases (0.13.x)
never break declared names or change artifact semantics.

Deprecations currently running:

- `numbers_to_constant` — has warned since 0.12.0; **removed in 0.14.0**.
  Replacement: `explicit_constant_placeholders`.
- `substitude_constants` (misspelling) — alias of `substitute_constants`;
  warns from 0.13.0, removal not before 0.15.0 (a shipped downstream imports
  it).
- `explicit_constant_placeholders(convert_numbers_to_constant=)` — the
  parameter is keyword-only and **required** in 0.13.x (its default silently
  flipped in the past; requiring it makes every call site state its intent).
  A default may return in 0.14.0.

Python: 0.13.0 supports Python ≥ 3.12. New minor releases may raise the
floor to the oldest Python receiving full upstream support ("new but stable"
policy).

## Artifacts

An installed artifact is identified by its manifest `revision` (the pinned
upstream commit) and per-file `sha256` digests, enforced at install and at
cache resolution. The engine refuses artifacts from unsupported generations:
`simplipy.compat.SUPPORTED_ENGINE_GENERATIONS` is the machine-readable
statement of what this version loads, and `IncompatibleArtifactError` is the
exception a deployment catches to distinguish "wrong artifact generation"
from an I/O failure.

Retired operator tokens (`simplipy.compat.RETIRED_OPERATOR_TOKENS`) are
refused at config load. A config that loads today keeps loading within the
same minor version.

Mined-artifact reproducibility is **environment-qualified**: a mine's
provenance sidecar records the engine build, the numeric environment
(including the libm fingerprint), and every artifact-affecting environment
switch that was set. Byte-identical re-mines are promised only on an
environment matching that record — see
[environment qualification](method/environment-qualification.md).

## Environment variables

The environment switches that can change a mined artifact or a certification
verdict are exactly those listed in `simplipy.engine.ARTIFACT_ENV_SWITCHES`;
they are recorded in the provenance sidecar of any mine run under them.
`SIMPLIPY_TRUSTED_MODULES` extends the realization trust model as documented.
All other `SIMPLIPY_*` switches are observability/debug knobs: undocumented
and unstable. See [environment variables](environment.md).

## Downstream pins

The three known consumers (`symbolic-data`, `flash-ansr`, `srbf`) pin
`simplipy>=0.3.1` with no upper bound. One (`srbf`) already breaks on 0.12.0
via names removed with the hyper-operator vocabulary. Consumers are advised
to cap at `<0.14` until they build against the declared surface; the declared
surface was chosen so that no current downstream import of an existing name
breaks in 0.13.0.
