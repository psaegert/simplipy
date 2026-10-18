# Environment variables

Two parts of SimpliPy's environment surface are documented and stable. Every
other `SIMPLIPY_*` switch is an observability or debug knob: undocumented and
subject to change.

## `SIMPLIPY_TRUSTED_MODULES`

A comma-separated list of module roots that a config may name in its operator
realizations, beyond the default `('math', 'np', 'scipy', 'simplipy')`. This is
the deployment-side way to extend the trust model; see
[Trust and deployment](guides/trust.md).

```bash
SIMPLIPY_TRUSTED_MODULES=mylab python run_service.py
```

## The artifact-affecting registry

`simplipy.engine.ARTIFACT_ENV_SWITCHES` is the registry of every environment
switch that can change a mined artifact, a certification verdict or a served answer.
The compiled core declares it (its read sites name their variables from the same
list). The registry is also a recording contract: a mine run with any of these set
records the override verbatim in its provenance sidecar, under
`soundness.env_overrides`, so an artifact mined under a non-default instrument says
so itself.

| Switch | What it changes |
|---|---|
| `SIMPLIPY_IVL_NODE_BUDGET` | the interval layer's node budget, and so which candidates the miner can decide |
| `SIMPLIPY_HIPREC_FRAC` | the high-precision near-miss escalation gate of the miner |
| `SIMPLIPY_TAGGED_FRACTION_MAX` | largest numerator and denominator (magnitude) of a decimal the tagged form spells as a bag (every fraction is one); changes tagged answers and mined output |

These exist for instrumentation and reproduction studies. Do not set them in
production: they change what the engine mines, certifies or answers, and an
artifact produced under them is not comparable to one produced without them.

The compiled core reads its switches once per process, at first use. Changing
`os.environ` between the first engine call and a mine therefore leaves the
sidecar's record unfaithful — set them before the process starts, or not at all.
