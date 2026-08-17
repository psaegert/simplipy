# Environment variables

SimpliPy's environment surface has exactly two documented parts; every other
`SIMPLIPY_*` switch is an observability/debug knob — undocumented and
unstable.

## `SIMPLIPY_TRUSTED_MODULES`

Extends the realization trust model for a deployment (see
[Trust and deployment](guides/trust.md)): a comma-separated list of module
roots a config may name beyond the default
`('math', 'np', 'scipy', 'simplipy')`.

```bash
SIMPLIPY_TRUSTED_MODULES=mylab python run_service.py
```

## The artifact-affecting registry

`simplipy.engine.ARTIFACT_ENV_SWITCHES` is the machine-readable registry of
every environment switch that can change a **mined artifact** or a
**certification verdict**. The registry is also the recording contract: a
mine run with any of these set records the override verbatim in its
provenance sidecar (`soundness.env_overrides`), so an artifact mined under a
non-default instrument says so itself.

| Switch | What it changes |
|---|---|
| `SIMPLIPY_IVL_GATE` | interval domain gate layer |
| `SIMPLIPY_IVL_CLASS` | interval value-class layer |
| `SIMPLIPY_IVL_REACH` | interval reachability layer |
| `SIMPLIPY_SPECIAL_BATTERY` | special-point battery layer |
| `SIMPLIPY_IVL_NODE_BUDGET` | interval node-budget override |
| `SIMPLIPY_AC_ABSORB_FIRST` | bag-match attempt order (serve outputs and mined rules) |
| `SIMPLIPY_MU_SYM` | μ symbol unit (the reduction ordering itself) |
| `SIMPLIPY_MU_FREE` | μ `<constant>` cost (the ordering) |
| `SIMPLIPY_ZERO_SIGN` | miner sign-combo grid (reference/reproduction) |
| `SIMPLIPY_POLE_GRID` | miner magnitude-grid ablation |
| `SIMPLIPY_HIPREC_FRAC` | high-precision near-miss escalation gate (calibration) |
| `SIMPLIPY_TAGGED_FRACTION_MAX` | tagged structural-fraction bound (changes mined output) |

These exist for instrumentation and reproduction studies. **Do not set them
in production**: they change what the engine mines or certifies, which is the
definition of an unqualified artifact. The Rust side reads its switches once
per process (at first use), so mutating `os.environ` between first engine use
and a mine makes the sidecar record unfaithful — do not do that either.
