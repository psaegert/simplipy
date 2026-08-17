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

# Simplify prefix expressions
engine.simplify(['/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3'])
# -> ['<mul>', '<constant>', '<div>', 'log', 'x3', '</mul>']   (the native tagged form: C/log(x3))

# The `form` parameter re-projects the same canonical answer: 'infix' renders it,
# 'explicit' gives the binary-prefix dialect that is_valid / prefix_to_infix consume
# (tagged output itself is accepted back as input by simplify/complexity/masking)
engine.simplify(['/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3'], form='infix')
# -> '<constant>/log(x3)'

# Simplify infix expressions
engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# -> '<constant>/x3'
```

`simplify`'s default `Mode.SOUND` is equivalence-preserving and idempotent;
the training-only `Mode.LOSSY` trades soundness for recall. The
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
# - acj-3-2  [installed]  Complete AC-judged rule mine ... (sources to length 3, targets to length 2), ...
# - acj-2-1  [installed]  Complete AC-judged rule mine ... (sources to length 2, targets to length 1), ...
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
