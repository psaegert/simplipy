# Environment qualification

"Byte-deterministically reproducible" is a claim about a machine, not just a
mine configuration. This page states exactly which environment facts qualify
a mined artifact, why they must, and how two artifacts are compared.

## The libm dependence, measured

The engine's transcendental constant folds (`exp`, `log`, `cosh`, …) go
through the **system libm**, which resolves on the running machine. Two
correct, standards-conforming libms may round a handful of values one ulp
apart — and a one-ulp difference at a fold is enough to change what the miner
can prove. This is not hypothetical; it is measured:

> `cosh(acosh(2))` rounds to exactly `2` on glibc 2.43 and to a value one
> ulp away on glibc 2.39, so a glibc 2.43 host mints `cosh acosh 2 → 2` — a
> rule the publish host does not mint. Both artifacts are sound; they are
> simply different, and only the environment record makes the difference
> attributable.

The exact-rational core is unaffected (rational arithmetic has no libm); the
qualification is about the fold path and the numeric confirmation/judging
stages.

## Complex arithmetic in `besselj`, by platform

`besselj` evaluates in float64 as the C++ it ports, and part of that
computation is complex. C leaves the product and quotient of two complex
numbers to runtime routines, `__muldc3` and `__divdc3`, and the engine calls
the platform's:

| Platform | The engine calls | scipy's compiled kernels call |
|---|---|---|
| Linux (x86_64, aarch64) | libgcc's | libgcc's, the same routines |
| macOS (x86_64, arm64) | LLVM's | LLVM's on x86_64 (the system's); none on arm64 |
| Windows (x64) | LLVM's | libgcc's, linked into scipy's build |

The products are the same everywhere; the quotients round differently. With
LLVM's routines, `besselj` differs from Linux at about 9% of a million points
spanning orders and arguments from tiny to huge, by 4.5e-16 relative at the
median and at most 1e-10, and never in class (a finite value stays finite, an
infinity or NaN stays one). So macOS and Windows differ from Linux in the last
bits, and on Windows the engine also differs from scipy. A mine that folds
`besselj` inherits its platform like any other fold, and its sidecar's
platform record says which.

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
- **The soundness state**: every `SIMPLIPY_*` variable set during the mine,
  recorded verbatim (no variable changes what the core mines, certifies or
  serves, so `simplipy verify-mine` requires this record to be empty; see
  [environment variables](../environment.md)), and the interval layer's
  fail-closed miss counters.
- **The measure fingerprint**: the μ constants and probe values with a
  digest, so artifacts mined under different reduction orderings are
  distinguishable from provenance alone.

## What the qualification means in practice

- A re-mine from an artifact's `mine.yaml` is promised byte-identical **only
  on an environment matching the sidecar** — same fingerprint, same
  versions, no `SIMPLIPY_*` variable recorded.
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
