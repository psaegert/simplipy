# Environment qualification

"Byte-deterministically reproducible" is a claim about a machine, not just a
mine configuration. This page states exactly which environment facts qualify
a mined artifact, why they must, and how two artifacts are compared.

## The libm dependence, measured

The engine's transcendental constant folds (`exp`, `log`, `cosh`, …) go
through the **system libm**, which resolves on the running machine. Two
correct, standards-conforming libms may round a handful of values one ulp
apart — and a one-ulp difference at a fold is enough to change what the miner
can prove. This is not hypothetical; it was measured during the 0.13.0
audit:

> `cosh(acosh(2))` rounds to exactly `2` on glibc 2.43 and to a value one
> ulp away on glibc 2.39, so a glibc 2.43 host mints `cosh acosh 2 → 2` — a
> rule the publish host does not mint. Both artifacts are sound; they are
> simply different, and only the environment record makes the difference
> attributable.

The exact-rational core is unaffected (rational arithmetic has no libm); the
qualification is about the fold path and the numeric confirmation/judging
stages.

## What the sidecar records

Every mine writes a provenance sidecar (`rules.json.provenance.json`)
recording, alongside the mine parameters and the core build stamp:

- **The environment**: python, platform, machine, libc, and the
  numpy/scipy/mpmath versions.
- **The `libm_fingerprint`**: a sha256 digest over a fixed battery of 36
  probes evaluated **through the deployed folding path** — not through a
  separately-linked libm, so the fingerprint measures the code path that
  actually minted the artifact. Two sidecars are comparable-or-not by
  inspecting one field.
- **The soundness state**: certificate kill-switch states, every
  artifact-affecting environment override recorded verbatim
  (see [environment variables](../environment.md)), and the interval
  layer's fail-closed miss counters.
- **The measure fingerprint**: the μ constants and probe values with a
  digest, so artifacts mined under different reduction orderings are
  distinguishable from provenance alone.

## What the qualification means in practice

- A re-mine from an artifact's `mine.yaml` is promised byte-identical **only
  on an environment matching the sidecar** — same fingerprint, same
  versions, no artifact-affecting overrides.
- On load, the engine cross-checks the artifact's recorded measure
  fingerprint against its own and warns on mismatch — a served ruleset is
  never silently re-interpreted under a different ordering.
- Verification verdicts (`simplipy.verify`) inherit the same qualification:
  a judge running on a different libm may resolve a borderline witness
  differently. The judge's arbitrary-precision contract evaluator narrows
  this to the deployed-check comparisons; refusals (the fatal buckets) are
  the honest answer where evaluation itself is environment-marginal.

The [formal treatment](../formal.md) covers the measure and the certificate
system this qualification protects.
