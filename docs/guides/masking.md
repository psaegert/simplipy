# Masking

Masking is **downstream policy, not engine behavior**: the engine simplifies
and delivers the full expression with explicit constants; consumers decide
what to abstract. `simplipy.masking` is the mechanism. It walks the engine's
output (the native tagged form or the explicit binary form) and tells a policy
the structural **role** of every literal it meets — a multiplicative
coefficient, an additive constant, a `pow` exponent, a `rootn` index — so a
policy like "mask constants, but keep in-vocabulary integer exponents" is a
single conditional, and position-blind accidents (masking a `rootn` index into
a skeleton that is NaN almost everywhere) are impossible to write by accident.

Masking is a separate **terminal** step: apply it to `simplify`'s output.

```python
import simplipy as sp
from simplipy import masking

engine = sp.SimpliPyEngine.load("acj-4-3", install=True)

masking.mask(engine.simplify(['+', 'x1', '3.14']), engine,
             masking.mask_values_keep_structure)
# -> ['<add>', 'x1', '<constant>', '</add>']
```

## The shipped policies

Choosing what to abstract is a real decision with real failure modes, so the
kinds ship in the module rather than being re-invented by every consumer:

- **`mask_values_keep_structure`** — numeric literal *values* become
  `<constant>` while structural literals stay.
- **`mask_all`** — every number, including either half of a fraction and the
  special constants. For structural comparison.
- **`mask_fittable`** — every number a constant optimizer can actually fit,
  *keeping* the ones it cannot (`pow` exponents, `rootn` indices). For
  training data.

A custom `(value, role) -> str | None` policy remains the primitive
(`masking.Role` is the role enum, `masking.literal_sites` enumerates the
literal positions with their roles), so any other rule — mask only floats,
keep a given vocabulary — is a one-line function.

## One constant per degree of freedom

Masking runs **two** stages: (a) the positional substitution the policy
drives, then (b) a **collect** pass that re-runs the engine over the result.
Stage (b) is what makes the invariant *one `<constant>` per degree of freedom*
hold: the serializers can split one state node across several tokens (a
rational is one node but prints as `<mul> 2 x0 <div> 3 </mul>`), so a
positional pass alone would abstract it into two `<constant>`s for one degree
of freedom — a flat direction for any downstream refiner. The collect pass
also collapses `c1*c2` and `c1+c2` (one degree of freedom each) while
correctly leaving `c1*x0 + c2*x1` alone (two), and normalizes the shape the
way a human would write it (`x0 / <constant>` → `<constant> * x0`).
`mask(..., collect=False)` gives the raw positional substitution instead.

## Mask once

Masking **destroys information**: feed `mask` an expression, never a
skeleton. The collect pass can turn structure into a literal (a `<sub>`
becoming a `-1` coefficient inside an odd function), and a second masking
pass then abstracts that literal into a degree of freedom the original never
had — measured at ~0.15% of skeletons over 20k rows. Mask exactly once, as
the last step.
