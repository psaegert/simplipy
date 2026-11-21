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

## Complex arithmetic in the Bessel functions

`besselj`, `bessely` and `besselk` evaluate in float64 as the C++ they port
(scipy's AMOS code), and part of that computation is complex. C leaves the
product and quotient of two complex numbers to runtime routines, `__muldc3` and
`__divdc3`, and runtimes differ between machines in the last bits of a
quotient. The engine calls none of them: the core computes both itself, with
the algorithms of GCC's runtime (C99 Annex G's product, Smith's quotient with
the scaling of Baudin and Smith, 2012), so these three functions give the same
bits on every platform. Against GCC's runtime on Linux, 29 million operand sets
per function, drawn to reach every branch (ordinary, subnormal and huge parts,
zero divisors, infinite and NaN parts, both orientations), give the same values;
a NaN may carry another operand's sign and payload, which no value reads.
(`erfc` and `besseli`, scipy's Cephes code, compute in real arithmetic.)

scipy's compiled kernels call the platform's routines, so where those compute
otherwise, the engine's Bessel values differ from that platform's scipy in the
last bits:

| Platform | scipy's compiled kernels call | The engine against that scipy |
|---|---|---|
| Linux (x86_64, aarch64) | the system's libgcc | the same values where libgcc computes as GCC 15.2's and 16's do; Ubuntu 26.04's rounds some quotients of the scaling branches (subnormal and huge parts) one unit in the last place apart, `(1e-310 + 3i)/(1e-300 + 7e-301i)` to a real part of 1.4093959731543626e300 against 1.4093959731543623e300 |
| macOS (x86_64, arm64) | LLVM's on x86_64 (the system's); none on arm64 | LLVM's quotient rounds differently (its product is the same): `besselj` differs at about 9% of a million points spanning orders and arguments from tiny to huge, by 4.5e-16 relative at the median and at most 1e-10, and never in class (a finite value stays finite, an infinity or NaN stays one) |
| Windows (x64) | libgcc's, linked into scipy's build | as on Linux, by the runtime scipy's build links |

`bessely` and `besselk` call the same product and quotient as `besselj`; their
identity with scipy is measured on Linux. A fold of one of these functions no
longer depends on the platform's runtime; it depends on the platform through
the libm, as every fold does (above).

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
