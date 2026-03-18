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

More examples can be found in the [documentation](https://simplipy.readthedocs.io/).

# Performance

<!-- <img src="https://raw.githubusercontent.com/psaegert/simplipy/main/assets/images/simplification_comparison_simplipy_sympy.svg" alt="Original vs Simplified Length and Simplification Time"/>

> Simplification efficacy and efficiency for different maximum pattern lengths (Engine: `dev_7-3`).
> Original expressions sampled with the [Lample-Charton Algorithm](https://arxiv.org/abs/1912.01412) using the following parameters:
> - 0 to 3 variables
> - 0 to 20 operators (corresponding to lengths of 0 to 41)
> - Operators:
>   - with relative weight 10: `+`, `-`, `*`, `/`
>   - with relative weight 1: `abs`, `inv`, `neg`, `pow2`, `pow3`, `pow4`, `pow5`, `pow1_2`, `pow1_3`, `pow1_4`, `pow1_5`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`, `mult2`, `mult3`, `mult4`, `mult5`, `div2`, `div3`, `div4`, `div5`
>
> Points show bootstrapped mean and 95% confidence interval (N = 10,000).
> Orange points are within the 95% confidence interval of the shortest simplified length for the respective original length.
> Using patterns beyond a length of 4 tokens does not yield significant improvements and comes at a cost of increased simplification time. -->


<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/psaegert/simplipy/main/assets/images/simplification_comparison_simplipy_sympy.svg" alt="Original vs Simplified Length and Simplification Time" width="500">
      <p><strong>Left:</strong> Empirical Cumulative Distribution Functions (ECDFs) of simplification wall-clock time. Our SimpliPy rewriting engine (shades of blue, varying Lmax) operates in the low to moderate millisecond regime, orders of magnitude faster than the SymPy <a href="https://peerj.com/articles/cs-103/">[Meurer et al. 2017]</a> baseline (orange, red). <strong>Right:</strong> ECDF of the simplification ratio |τ ∗|/|τ |. The inset highlights the tail of the distribution. Our method with Lmax ≥ 5 achieves simplification ratios comparable to the SymPy baseline while maintaining high throughput.<br>
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
    version = {0.2.14},
    url = {https://github.com/psaegert/simplipy}
}
```
