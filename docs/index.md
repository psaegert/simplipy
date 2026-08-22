# SimpliPy

SimpliPy simplifies mathematical expressions the way a compiler optimizes
code: fast, deterministically, and with a stated guarantee. One idea drives
the whole engine — **expressions are canonicalized into flat bags with exact
rational arithmetic, and a rewrite is taken only when a single integer price
(the description length μ) strictly drops.** Termination, no rewrite loops,
never-worse-than-the-input, and idempotence are not separate features; they
all fall out of that one rule.

It exists for workloads where classic computer-algebra tools struggle:
millions of machine-generated expressions in symbolic-regression and
ML-training pipelines, where expressions live as prefix token lists and every
millisecond per expression is multiplied by a corpus. Instead of converting
tokens into heavyweight objects and back, SimpliPy keeps them as token lists
end to end — it is the expression-engine leaf under
[symbolic-data](https://github.com/psaegert/symbolic-data), and through it
feeds [flash-ansr](https://github.com/psaegert/flash-ansr) training and the
srbf benchmark framework. Measured comparisons against SymPy live in the
[simplification guide](guides/simplify.md); the plots, not this page, report
what they show.

Thirty seconds of it:

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4-3", install=True)

engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# -> '<constant>/x3'
```

The guarantee is honest about what it measures: the output is never costlier
than the input **under μ** — which is a description length, not a token
count — the default `f64` mode is sound as the deployed evaluator computes,
and everything the engine ships (mined rulesets, soundness certificates,
verification verdicts) carries provenance that says exactly where it came
from and on what machine it holds.

## Where to go

- **Use it** — [Getting started](getting-started.md), then the guides:
  [simplifying](guides/simplify.md), [artifacts](guides/artifacts.md),
  [masking](guides/masking.md), [trust](guides/trust.md),
  [mining](rules.md), [verifying](guides/verify.md).
- **Look it up** — the [API reference](api.md), the
  [compatibility policy](compatibility.md), the
  [environment variables](environment.md).
- **Check the claims** — the formal pages on the
  [engine](formal.md) and the [mining algorithm](algorithm.md), and
  [environment qualification](method/environment-qualification.md): what
  "reproducible" means on real machines, stated precisely.
