# Getting started

## Install

```bash
pip install simplipy
```

Python ≥ 3.12. The package ships a required compiled Rust core
(`simplipy._core`); there is no pure-Python fallback. Rule mining and
verification pull the full numeric stack; plain simplification does not.

## First simplification

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4-3", install=True)   # the published AC-engine artifact

expr = ['/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3']

# Simplify prefix expressions: `simplify` answers in the form it was GIVEN
engine.simplify(expr)
# -> ['/', '<constant>', 'log', 'x3']            (explicit binary prefix: C/log(x3))

# To change the NOTATION, convert -- `to_infix` / `to_prefix` / `to_tagged` are pure
# syntactic conversions that never simplify -- and compose the two calls:
engine.simplify(engine.to_tagged(expr))
# -> ['<mul>', '<constant>', '<div>', 'log', 'x3', '</mul>']   (the native tagged form)

engine.to_infix(engine.simplify(expr))
# -> '<constant> / log(x3)'

# Simplify infix expressions
engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# -> '<constant>/x3'
```

`simplify`'s default `Mode.f64` is sound as the deployed f64 evaluator computes;
`Mode.real` is sound as mathematics defines; the training-only `Mode.corpus` trades
soundness for recall. The
[simplification guide](guides/simplify.md) covers the modes, the search
budget, and the guarantee the engine actually makes.

## Engine assets

Available engines can be browsed and downloaded from Hugging Face. The
asset manager handles listing, installing, and uninstalling:

```python
sp.list_assets("engine")
# --- Available engine assets ---
# - acj-4-3  [installed]  Complete AC-judged rule mine of the clean 23-operator vocabulary
#                         (sources to length 4, targets to length 3), ... Pairs with simplipy >= 0.12.
# - acj-3-2               Complete AC-judged rule mine ... (sources to length 3, targets to length 2), ...
# - acj-2-1               Complete AC-judged rule mine ... (sources to length 2, targets to length 1), ...
# - base                  Bare 23-operator engine configuration (no rules): the clean-vocabulary
#                         starting point for fresh mining. Pairs with simplipy >= 0.12.
# - ...                   (pre-0.12 assets remain listed for older installs; they refuse to load on 0.12)
```

Every published artifact is identity-pinned (a manifest revision plus
per-file sha256 digests, enforced at install and at cache resolution) — see
[Artifacts and assets](guides/artifacts.md).

## Where next

- [Simplifying expressions](guides/simplify.md) — modes, budget, the μ guarantee.
- [Masking](guides/masking.md) — literals → `<constant>` for models, done once, done right.
- [Mining rulesets](rules.md) — build your own artifact.
- [API reference](api.md) — the declared surface.
