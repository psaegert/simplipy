# Trust and deployment

## What a config can execute

An operator config declares a **realization** for each operator: the Python
callable that computes it, such as `simplipy.operators.sin` or `np.sin`. When an
engine is built, it imports the modules those realizations name, because the
Python evaluation path (`prefix_to_infix(realization=True)` → `codify` →
`code_to_lambda`) resolves the names against them. The compiled core needs none
of it — it dispatches on canonical operator names through a native table, so
`simplify`, `complexity` and the miner are unaffected.

Importing a module runs its top-level code. Naming a module in a config is
therefore enough to execute code, before any expression is evaluated. Configs
travel: a colleague's file, a Hugging Face asset loaded by name, a cloned
repository. So SimpliPy restricts which module roots a config may name.

## The default allowlist

```python
from simplipy.trust import DEFAULT_TRUSTED_MODULES

DEFAULT_TRUSTED_MODULES
# -> ('math', 'np', 'scipy', 'simplipy')
```

Each module has one accepted spelling: `np`, not `numpy`. `np.pi` and `np.e` are
part of the token grammar, so `np` is the canonical form, and a realization
written `numpy.sin` is refused with a hint. A realization naming any root
outside the list refuses at engine construction, and the message names both the
operator that asked and how to allow it.

## Extending it

Trust is granted from outside the config, never by the config. A
`trusted_modules:` key inside the YAML is ignored. Two surfaces grant it:

<!-- docs-example: skip: illustrative -- references the reader's own config file and module -->
```python
engine = SimpliPyEngine.from_config('their_engine.yaml', trusted_modules=['mylab'])
```
```bash
SIMPLIPY_TRUSTED_MODULES=mylab python run_service.py   # for deployments
```

Each engine evaluates in its own namespace, containing the modules its own
realizations needed plus `np`. One engine's imports are not reachable from
another engine's expressions.

## What this does not cover

The allowlist makes a config safe to *load*, and it scopes what a compiled
expression can see. It does not make `codify` or `code_to_lambda` safe against a
hostile *expression*: compiling and evaluating attacker-supplied source is
unsafe in Python regardless of which modules are in scope. Treat expression
strings from untrusted sources the way you would treat any other code.

Catch `simplipy.trust.UntrustedModuleError` to tell a refused config apart from
a broken one. Report a path that executes code without tripping the trust model
through the repository's private security reporting — see `SECURITY.md`.
