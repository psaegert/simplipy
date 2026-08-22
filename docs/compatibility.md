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

Removed in 0.14.0:

- `simplipy.utils.numbers_to_constant` — warned since 0.12.0, announced for
  0.14.0 in the 0.13.0 changelog, removed here. Replacement:
  `simplipy.masking.mask(tokens, engine, policy)` or the `engine.mask` front
  door, with the policy that states the intent (`mask_all` for the legacy
  mask-everything behaviour, `mask_fittable` for what a constant optimizer can
  fit). The replacement is not a drop-in and is not meant to be: the removed
  helper classified by a bare `float()` probe, so it minted a
  finite-by-doctrine `<constant>` for `inf`/`nan`/`1_000` (now a loud
  `ValueError`) and walked past `np.pi`/`np.e` and the exact fraction `1/3`
  (now masked). Pass `collect=False` for the positional 1:1 substitution the
  helper approximated. Unaffected and unchanged: the
  `explicit_constant_placeholders(convert_numbers_to_constant=)` keyword and
  `read_infix(..., mask_numbers=True)`, which share the name but not the
  surface.

No deprecations are currently running. 0.14.0 retires every alias that was
outstanding, so a retired name raises rather than warning:

| removed in 0.14.0 | replacement |
|---|---|
| `Mode.SOUND`, `Mode.LOSSY` | `Mode.f64`, `Mode.corpus` |
| `mode='sound'`, `mode='lossy'` | `mode='f64'`, `mode='corpus'` (matched case-insensitively) |
| `SimpliPyEngine.parse` | `read_infix` |
| `simplify(..., form=)` | convert first: `simplify(to_tagged(x))` |
| `simplify(..., node_budget=)` | `max_passes=` — it bounds outer rewrite passes, and never counted nodes |
| `SimpliPyEngine.load(path=)` | `load(engine=)` |
| `normalize_skeleton`, `normalize_expression` | `to_skeleton`, `to_expression`, which now require `engine=` |
| `masking.mask_values_keep_structure` | `masking.mask_fittable` (the same object throughout) |
| `utils.substitude_constants` | `utils.substitute_constants` |

`Mode` also stopped being an `IntEnum`, so `<` between modes raises `TypeError`: the
modes are an axis, not rungs of one ordering, and comparing them is a category error
rather than a question with a wrong answer.

Still supported and unchanged: `explicit_constant_placeholders(convert_numbers_to_constant=)`
remains keyword-only and required, so every call site states its intent.

Behaviour changes in 0.14.0 that are **not** deprecations, because no name changed:

- **`Mode.f64` no longer folds `sin(np.pi)` to `0`.** It is exactly `0` in mathematics
  and `1.2246467991473532e-16` in f64, so the rewrite changes what the deployed
  evaluator computes. It moves to `Mode.real` and `Mode.corpus`, which still fold it.
  102 rules are affected, nearly all of the same symbolic-cancellation family.
- **`f64` mode preserves what the deployed evaluator computes for every rewrite it
  applies, but not your evaluation order.** The canonical form flattens sums and
  products into bags and re-emits them, and IEEE-754 addition commutes but does not
  associate. The association is deterministic and well-conditioned expressions keep
  their value; ill-conditioned ones may not.

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
