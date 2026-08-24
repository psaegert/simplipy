# SimpliPy

SimpliPy simplifies mathematical expressions at corpus scale: millions of
machine-generated expressions in symbolic-regression and ML-training
pipelines, where per-expression cost is multiplied by the size of the dataset.

One rule drives the engine. Expressions are canonicalized into flat AC bags
with exact rational arithmetic, and a rewrite is applied only when the
description length μ strictly drops. Termination, loop-freedom,
never-worse-than-input, and idempotence follow from that rule.

Tokens are parsed straight into the
canonical state (interned ids in flat bags) and serialized straight back
out, with no round trip through a general CAS object model. SimpliPy is the
expression-engine used for
[symbolic-data](https://github.com/psaegert/symbolic-data),
[flash-ansr](https://github.com/psaegert/flash-ansr),
and [srbf](https://github.com/psaegert/srbf).

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4", install=True)

engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# -> '<constant>/x3'
```

## Where to go

- **Use it** — [Getting started](getting-started.md), then the guides:
  [simplifying](guides/simplify.md), [artifacts](guides/artifacts.md),
  [masking](guides/masking.md), [trust](guides/trust.md),
  [mining](rules.md), [verifying](guides/verify.md).
- **Look it up** — the [API reference](api.md), the
  [compatibility policy](compatibility.md), the
  [environment variables](environment.md).
- **Check the claims** — the formal specification of the
  [engine](formal.md), and
  [environment qualification](method/environment-qualification.md): what
  "reproducible" means on real machines, stated precisely.