<h1 align="center" style="margin-top: 0px;">SimpliPy:<br>Efficient Simplification of Mathematical Expressions</h1>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/simplipy.svg)](https://pypi.org/project/simplipy/)
[![PyPI license](https://img.shields.io/pypi/l/simplipy.svg)](https://pypi.org/project/simplipy/)

</div>

<div align="center">

[![pytest](https://github.com/psaegert/simplipy/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/simplipy/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/simplipy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/simplipy/actions/workflows/pre-commit.yml)
[![CodeQL Advanced](https://github.com/psaegert/simplipy/actions/workflows/codeql.yaml/badge.svg)](https://github.com/psaegert/simplipy/actions/workflows/codeql.yaml)


</div>


# Usage

```sh
pip install simplipy
```

```python
import simplipy as sp

engine = sp.SimpliPyEngine.from_config(sp.utils.get_path('configs', 'dev_7-3.yaml'))

# Simplify prefix expressions
engine.simplify(('/', '<constant>', '*', '/', '*', 'x3', '<constant>', 'x3', 'log', 'x3'))
# > ('/', '<constant>', 'log', 'x3')

engine.simplify('x3 * sin(<constant> + 1) / (x3 * x3)')
# > '(<constant> / x3)'
```

# Performance

<img src="https://raw.githubusercontent.com/psaegert/simplipy/main/assets/images/dev_7-3_multi_simplification_length_histogram.png" alt="Original vs Simplified Length and Simplification Time"/>

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
> Using patterns beyond a length of 4 tokens does not yield significant improvements and comes at a cost of increased simplification time.


# Collecting Rules

```sh
simplipy find-rules -e "{{ROOT}}/configs/my_config.yaml" -c "{{ROOT}}/configs/create_my_config.yaml" -o "{{ROOT}}/data/rules/my_config.json" -v --reset-rules
```

- `-e` is the path to the engine configuration file to use as a backend
- `-c` is the path to the configuration file containing parameters for finding rules
- `-o` is the output path for the collected rules
- `-v` enables verbose output
- `--reset-rules` will start with an empty rule set, otherwise it will append to the existing rules loaded with the engine

A configuration file for collecting rules could look like this:

```yaml
# This includes special symbols for intermediate simplification steps
extra_internal_terms: [
  '<constant>',
  '0',
  '1',
  '(-1)',
  'np.pi',
  'np.e',
  'float("inf")',
  'float("-inf")',
  'float("nan")'
]

# Number of dummy variables used to create equations
# null means the number of dummy variables is determined automatically based on max source pattern length
dummy_variables: null  

# Maximum number of tokens in the source equation
max_source_pattern_length: 3

# Maximum number of tokens in the target equation
max_target_pattern_length: 2

# Number of data points to compute the image of the equation
n_samples: 1024

# Number of combinations of constants to sample for each equation
# This prevents false positives due to unlucky constant combinations
constants_fit_challenges: 16

# Number of retries for each challenge to find a valid constant combination
# This accounts for optimization problems that may not converge
constants_fit_retries: 16
```

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
pytest -v
```

# Citation
```bibtex
@software{simplipy-2025,
    author = {Paul Saegert},
    title = {Efficient Simplification of Mathematical Expressions},
    year = 2025,
    publisher = {GitHub},
    version = {0.2.2},
    url = {https://github.com/psaegert/simplipy}
}
```
