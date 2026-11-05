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
list), and it is empty: no environment variable changes what the core mines,
certifies or serves. The interval layer's node budget, the one instrument a mine may
need to raise, is the `find_rules(interval_node_budget=...)` argument, recorded in
the sidecar's parameters.

A mine still records every `SIMPLIPY_*` variable set while it runs (the diagnostics,
the test-asset switch, the build stamp, the progress interval and the trusted-module
list excepted) in its provenance sidecar, under `soundness.env_overrides`, and
`simplipy verify-mine` requires that record, and its own environment, to be empty.
