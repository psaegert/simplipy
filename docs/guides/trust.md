# Trust and deployment

## What a config is allowed to do

An operator config declares a **realization** for each operator — the Python callable
that computes it (`simplipy.operators.sin`, `np.sin`). The engine imports the modules
those realizations name, because the Python evaluation path
(`prefix_to_infix(realization=True)` → `codify` → `code_to_lambda`) resolves the names
against them. The Rust core needs none of it: it dispatches on canonical operator names
through a native table, so `simplify`, `complexity` and the miner are unaffected.

Importing a module **runs its top-level code**, so naming a module in a config is enough
to execute code — before any expression is evaluated. Since configs travel (a colleague's
file, a Hugging Face asset via `SimpliPyEngine.load`, a cloned repository), simplipy
restricts which module roots a config may name:

```python
DEFAULT_TRUSTED_MODULES = ('math', 'np', 'scipy', 'simplipy')
```

One spelling per module: **`np`, never `numpy`** — `np.pi` and `np.e` are normative token
grammar, so the alias is the canonical form and a realization written `numpy.sin` is
refused with a hint. Anything outside the list refuses at construction, naming the
operator that asked and how to allow it.

Trust is granted from **outside** the config, never by the config. A `trusted_modules:`
key inside the YAML would be worthless — the author of the dangerous line is the author
of the permission line — so simplipy ignores it and offers two surfaces instead:

<!-- docs-example: skip: illustrative -- references the reader's own config file and module -->
```python
engine = SimpliPyEngine.from_config('their_engine.yaml', trusted_modules=['mylab'])
```
```bash
SIMPLIPY_TRUSTED_MODULES=mylab python run_service.py   # for deployments
```

Each engine also evaluates in **its own namespace**: the modules its own realizations
needed, plus `np`. Before 0.13 every engine shared `simplipy.engine`'s module globals, so
one engine's imports were reachable from another's expressions along with simplipy's own
(`os`, `importlib`, …); that is closed.

**The limit, stated plainly:** this makes a config safe to *load*, and it scopes what a
compiled expression can *see*. It does not make `codify`/`code_to_lambda` safe against a
hostile **expression** — compiling and evaluating attacker-supplied source is unsafe by
construction in Python, and no allowlist changes that. Treat expression strings from
untrusted sources as you would any other code.

Deployments catch `simplipy.trust.UntrustedModuleError` to distinguish a
refused config from a broken one. Vulnerability reports (a path that executes
code *without* tripping the trust model) go through the repository's private
security reporting — see `SECURITY.md`.
