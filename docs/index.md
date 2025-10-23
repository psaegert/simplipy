# SimpliPy Documentation

SimpliPy is a high-throughput symbolic simplifier built for workloads where
classic tools like SymPy struggle—think millions of expressions in the pre-training of
Flash-ANSR's prefix-based transformer models. Instead of converting tokens into
heavyweight objects and back again, SimpliPy keeps expressions as lightweight
prefix lists, enabling rapid rewriting and direct integration with machine
learning pipelines.


## Why SimpliPy Exists

SymPy excels at exact algebra, but its object graph and string parsing introduce
costs that dominate at scale. SimpliPy was created to remove those bottlenecks:

- **Prefix-first representation** – Expressions stay as token lists the entire
	time, so there's no repeated parsing or AST allocation.
- **Deterministic pipelines** – Rule application, operand sorting, and literal
	masking always produce the same layout, which keeps downstream caches warm.
- **GPU-friendly integration** – Outputs map directly into Flash-ANSR's input
	space without any conversion step, making it practical to simplify millions of
	candidates per minute.


## Simplification Pipeline (Pseudo-Algorithm)

```text
function simplify(expr, max_iter=5):
    tokens = parse(expr)  # infix→prefix or validate existing prefix
    tokens = normalize(tokens)  # power folding, unary handling

    for _ in range(max_iter):
        tokens = cancel_terms(tokens)  # additive/multiplicative multiplicities
        tokens = apply_rules(tokens)   # compiled rewrite patterns
        tokens = sort_operands(tokens) # canonical order for commutative ops
        tokens = mask_literals(tokens) # collapse trivial numerics to <constant>

        if converged(tokens):
            break

    return finalize(tokens)  # prefix list or infix string, caller’s choice
```

This loop is intentionally lightweight: each pass performs a handful of pure
list transformations, giving you predictable performance even on nested or noisy
expressions.


## Key Components

- **Parsing & normalization** – `SimpliPyEngine.parse` and
	`convert_expression` convert infix input, harmonize power operators, and
	propagate unary negation without losing prefix fidelity.
- **Term cancellation** – `collect_multiplicities` and `cancel_terms` identify
	subtrees that appear with opposite parity or redundant factors, pruning them
	before any rules run.
- **Rule execution** – `compile_rules` turns machine-discovered or human-authored
	simplifications into tree patterns. `apply_simplifcation_rules` then performs
	fast top-down matching in each iteration.
- **Canonical ordering** – `sort_operands` imposes a stable ordering for
	commutative operators, ensuring identical expressions share identical token
	layouts.
- **Rule discovery workflow** – `find_rules` explores expression space in
	parallel worker processes, confirms identities with numeric sampling, and
	writes back deduplicated rulesets that future engines can load instantly.


## Quickstart

```bash
pip install simplipy
```

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("dev_7-3", install=True)

# Simplify prefix expressions
engine.simplify(['/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3'])
# -> ['/', '<constant>', 'log', 'x3']

# Simplify infix expressions
engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# -> '<constant> / x3'
```

Available engines can be browsed and downloaded from Hugging Face.
The SimpliPy Asset Manager handles listing, installing, and uninstalling assets:

```python
sp.list_assets("engine")
# --- Available Assets ---
# - dev_7-3         [installed]  Development engine 7-3 for mathematical expression simplification.
# - dev_7-2                      Development engine 7-2 for mathematical expression simplification.
```

## Where to go next

- Explore the [API reference](api.md) for function-level details.
- Read the [rule authoring guide](rules.md) to build simplification rule sets.

Happy simplifying!