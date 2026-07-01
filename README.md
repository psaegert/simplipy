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
- Saegert & Köthe 2026, _Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression_ (preprint, under review) [https://arxiv.org/abs/2602.08885](https://arxiv.org/abs/2602.08885)


# Usage

```sh
pip install simplipy
```

> As of 0.3.0 the inline phase (`simplify`, conversions, validation) is a compiled Rust extension
> (`simplipy._core`). Prebuilt wheels are published for Linux (x86_64/aarch64), macOS (x86_64/arm64)
> and Windows (x64) on CPython ≥ 3.11, so `pip install simplipy` does not compile anything for most
> users. Installing from the **source distribution** (an unsupported platform, or `--no-binary`)
> requires a Rust toolchain (`rustup`, MSRV 1.83). If the extension is unavailable at runtime, the
> package transparently falls back to a slower pure-Python implementation.

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("dev_7-3", install=True)

# Simplify prefix expressions
engine.simplify(('/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3'))
# > ('/', '<constant>', 'log', 'x3')

# Simplify infix expressions
engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# > '<constant> / x3'
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

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/psaegert/simplipy/main/assets/images/simplification_comparison_sympy_python_rust.svg" alt="Simplification time and ratio ECDFs: SymPy vs SimpliPy (Python 0.2.15) vs SimpliPy (Rust 0.3.0)" width="680">
      <p><strong>Top row:</strong> SimpliPy <code>0.3.0</code> (Rust inline engine, green). <strong>Bottom row:</strong> SimpliPy <code>0.2.15</code> (pure Python, blue). <strong>Left:</strong> Empirical Cumulative Distribution Functions (ECDFs) of simplification wall-clock time across maximum pattern lengths L<sub>max</sub> = 0–7, with the SymPy <a href="https://peerj.com/articles/cs-103/">[Meurer et al. 2017]</a> baseline (orange, red). The Rust inline engine is roughly 5× to 100× faster than the pure-Python engine at the same L<sub>max</sub> (≈ 15× at L<sub>max</sub> = 4), and both are orders of magnitude faster than SymPy. <strong>Right:</strong> ECDF of the simplification ratio |τ ∗|/|τ | (inset: zoom on the low-ratio region where the L<sub>max</sub> curves separate); the Rust and Python engines produce near-identical simplification-ratio distributions, so the Rust rewrite buys the speed-up without sacrificing simplification quality. (0.3.0 does deliberately change behaviour on a small fraction of inputs via the conversion-quirk fixes and numeric folding; see the <a href="https://github.com/psaegert/simplipy/blob/main/CHANGELOG.md">CHANGELOG</a>.)<br>
      Source expressions are sampled with 0 to 17 unique variables and 1 to 35 symbols <a href="https://arxiv.org/abs/2602.08885">[Saegert & Köthe 2026]</a></p>
    </td>
  </tr>
</table>

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
@misc{saegert2026breakingsimplificationbottleneckamortized,
  title   = {Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression},
  author  = {Paul Saegert and Ullrich Köthe},
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
    year = 2025,
    publisher = {GitHub},
    version = {0.3.1},
    url = {https://github.com/psaegert/simplipy}
}
```
