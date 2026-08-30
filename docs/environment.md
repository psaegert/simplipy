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
switch that can change a mined artifact or a certification verdict. The registry
is also a recording contract: a mine run with any of these set records the
override verbatim in its provenance sidecar, under `soundness.env_overrides`, so
an artifact mined under a non-default instrument says so itself.

| Switch | What it changes |
|---|---|
| `SIMPLIPY_IVL_GATE` | interval domain gate layer |
| `SIMPLIPY_IVL_CLASS` | interval value-class layer |
| `SIMPLIPY_IVL_REACH` | interval reachability layer |
| `SIMPLIPY_SPECIAL_BATTERY` | special-point battery layer |
| `SIMPLIPY_IVL_NODE_BUDGET` | interval node-budget override |
| `SIMPLIPY_AC_ABSORB_FIRST` | bag-match attempt order (affects served output and mined rules) |
| `SIMPLIPY_MU_SYM` | symbol unit of the simplicity measure, and so the reduction ordering |
| `SIMPLIPY_MU_FREE` | `<constant>` cost in the measure, and so the ordering |
| `SIMPLIPY_ZERO_SIGN` | miner sign-combo grid |
| `SIMPLIPY_POLE_GRID` | miner magnitude-grid ablation |
| `SIMPLIPY_HIPREC_FRAC` | high-precision near-miss escalation gate |
| `SIMPLIPY_TAGGED_FRACTION_MAX` | tagged structural-fraction bound (changes mined output) |

These exist for instrumentation and reproduction studies. Do not set them in
production: they change what the engine mines or certifies, and an artifact
produced under them is not comparable to one produced without them.

The compiled core reads its switches once per process, at first use. Changing
`os.environ` between the first engine call and a mine therefore leaves the
sidecar's record unfaithful — set them before the process starts, or not at all.
