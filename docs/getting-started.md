# Getting started

## Install

```bash
pip install simplipy
```

Python ≥ 3.12. The package ships a required compiled Rust core
(`simplipy._core`); there is no pure-Python fallback. Rule mining and
verification are the only paths that use scipy and mpmath at runtime; plain
simplification touches only numpy (all three install as required dependencies).

## First simplification

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4-llm_0.15", install=True)   # the published default rule set

# Simplify infix expressions: a str in, a str out
engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# -> '<constant>/x3'

# Pipelines carry expressions as prefix token lists; `simplify` answers in the
# form it was GIVEN
expr = ['/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3']

engine.simplify(expr)
# -> ['/', '<constant>', 'log', 'x3']            (explicit binary prefix: C/log(x3))

# To change the NOTATION, convert -- `to_infix` / `to_prefix` / `to_tagged` are pure
# syntactic conversions that never simplify -- and compose the two calls:
engine.simplify(engine.to_tagged(expr))
# -> ['<mul>', '<constant>', '<div>', 'log', 'x3', '</mul>']   (the native tagged form)

engine.to_infix(engine.simplify(expr))
# -> '<constant> / log(x3)'
```

`simplify`'s default `Mode.strict` is for expressions you will evaluate in float64. Every step
it takes is either reproduced by float64 itself (within 8 units in the last place), or true in
the real numbers with float64 giving both sides the same kind of value (both finite, the same
infinity, or both undefined). That promise is made per step, not for the whole answer: where
float64 rounds badly (near a pole, a jump or a saturation) the answer can differ from float64's
evaluation of your input by more than rounding. The training-only `Mode.permissive` reads the
reals' values and trades soundness for recall: `atanh(tanh(t)) → t` is true for every real `t`
yet returns `inf` in float64 once `tanh` saturates to exactly `1.0` at large `|t|`, so strict
does not serve it and permissive does. The `real` mode is retired: naming it raises. The
[simplification guide](guides/simplify.md) covers the modes, the search
budget, and the guarantee the engine actually makes.

## Engine assets

Available engines can be browsed and downloaded from Hugging Face. The
asset manager handles listing, installing, and uninstalling:

<!-- PENDING the release: the manifest's own descriptions of the 0.15 assets. -->
```python
sp.list_assets("engine")
# --- Available engine assets ---
# - acj-4-llm_0.15 [installed] Complete AC-judged rule mine of the 36-operator 0.15
#                         vocabulary (sources and targets to length 4) plus the LLM and
#                         manual proposal lane ... Pairs with simplipy 0.15.
# - base                  Bare 23-operator engine configuration (no rules): the clean-vocabulary
#                         starting point for fresh mining. Pairs with simplipy >= 0.12.
# - ...                   (older assets remain listed for older installs; a rule file mined
#                         before 0.15 refuses to load on >= 0.15, a generation-1 artifact
#                         on >= 0.12)
```

`base` loads with a warning: an engine without rules parses, evaluates and
returns canonical forms, but no simplification rule ever fires.

Every published artifact is identity-pinned (a manifest revision plus
per-file sha256 digests, enforced at install and at cache resolution) — see
[Artifacts and assets](guides/artifacts.md).

## Where next

- [Upgrading to 0.15](upgrading.md) — for users of 0.14: what changes and what to do.
- [Simplifying expressions](guides/simplify.md) — modes, budget, the μ guarantee.
- [Masking](guides/masking.md) — literals → `<constant>` for models, done once, done right.
- [Mining rulesets](rules.md) — build your own artifact.
- [API reference](api.md) — the declared surface.
