<h1 align="center" style="margin-top: 0px;">SimpliPy:<br>Efficient Simplification of Mathematical Expressions</h1>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/simplipy.svg)](https://pypi.org/project/simplipy/)
[![PyPI license](https://img.shields.io/pypi/l/simplipy.svg)](https://pypi.org/project/simplipy/)
[![Documentation Status](https://readthedocs.org/projects/simplipy/badge/?version=latest)](https://simplipy.readthedocs.io/en/latest/?badge=latest)

</div>

<div align="center">

[![pytest](https://github.com/psaegert/simplipy/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/simplipy/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/simplipy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/simplipy/actions/workflows/pre-commit.yml)
[![CodeQL Advanced](https://github.com/psaegert/simplipy/actions/workflows/codeql.yaml/badge.svg)](https://github.com/psaegert/simplipy/actions/workflows/codeql.yaml)


</div>

# Publications
- Saegert & Köthe 2026, _Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression_ (ICML 2026) [https://arxiv.org/abs/2602.08885](https://arxiv.org/abs/2602.08885)


# Usage

```sh
pip install simplipy
```

> The compiled Rust extension (`simplipy._core`) is **required**: the inline phase (`simplify`,
> conversions, validation) runs on it exclusively, and there is no pure-Python fallback. Prebuilt
> wheels are published for Linux (x86_64/aarch64), macOS (x86_64/arm64) and Windows (x64) on
> CPython ≥ 3.12, so `pip install simplipy` does not compile anything for most users. Installing
> from the **source distribution** (an unsupported platform, or `--no-binary`) requires a Rust
> toolchain (`rustup`, MSRV 1.83). If the extension is missing at runtime, constructing an engine
> raises `ImportError`.

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4-3", install=True)   # a published ruleset artifact

# Simplify prefix expressions
engine.simplify(('/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3'))
# > ('<mul>', '<constant>', '<div>', 'log', 'x3', '</mul>')

# Simplify infix expressions
engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# > '<constant>/x3'
```

Token input returns the engine's native **tagged** form by default (n-ary `+`/`*` bags are
delimited: `<add> ... </add>`, `<mul> ... </mul>`; tagged output is accepted back as input).
The `form` parameter selects a different projection of the same canonical answer:

```python
expr = ('/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3')

engine.simplify(expr, form='infix')      # the pretty rendering (a str)
# > '<constant>/log(x3)'

engine.simplify(expr, form='explicit')   # binary prefix -- what is_valid / prefix_to_infix read
# > ('/', '<constant>', 'log', 'x3')
```

## Normalization

The root-exported `normalize_skeleton`, `normalize_expression`, and
`normalize_variable_token` helpers (also available as `simplipy.normalization`)
canonicalize a prefix token sequence so that two expressions that are "the same"
up to variable renaming / constant values compare equal. They are pure-string
helpers with no engine state, so consumers such as holdout matching and
symbolic-recovery scoring share identical behavior by construction.

```python
import simplipy as sp

# Skeleton form: variables -> x{n}, numeric literals -> <constant>
sp.normalize_skeleton(['+', 'v1', '2.5'])
# > ['+', 'x1', '<constant>']

# Expression form: variables canonicalized, numeric literals kept intact
sp.normalize_expression(['+', 'V1', '2.5'])
# > ['+', 'x1', '2.5']

# Classify / canonicalize a single token -> (normalized_token, is_variable)
sp.normalize_variable_token('X3')
# > ('x3', True)
sp.normalize_variable_token('sin')
# > ('sin', False)
```

More examples can be found in the [documentation](https://simplipy.readthedocs.io/).

# Performance

On a 65,536-expression symbolic-regression benchmark, paired per-row against SymPy's `simplify` (serial single-core):

| | SimpliPy | SymPy |
|---|---:|---:|
| Rows won head-to-head | **18.7%** | 17.1% |
| Mean size ratio (lower is better) | **0.98** | 1.07 |
| Expressions made bigger | **0.00%** | 40.5% |
| Median per-row speedup | **≈780×** | 1× |

Full results, figures, and methodology: [simplify guide](https://simplipy.readthedocs.io/en/stable/guides/simplify/) · [paper](https://arxiv.org/abs/2602.08885).

# Development

## Setup
To set up the development environment, run the following commands:

```sh
pip install -e .[dev]
pre-commit install
```

## Tests

Test the package with `pytest`:

```sh
pytest tests --cov src --cov-report html
```

or to skip integration tests,

```sh
pytest tests --cov src --cov-report html -m "not integration"
```

# Citation
```bibtex
@inproceedings{saegert2026breakingsimplificationbottleneckamortized,
  title   = {Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression},
  author  = {Paul Saegert and Ullrich Köthe},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year    = {2026},
  eprint  = {2602.08885},
  archivePrefix =  {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2602.08885},
}

% Optionally
@software{simplipy-2025,
    author = {Paul Saegert},
    title = {Efficient Simplification of Mathematical Expressions},
    year = 2026,
    publisher = {GitHub},
    version = {0.13.1},
    url = {https://github.com/psaegert/simplipy}
}
```
