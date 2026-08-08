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

> The compiled Rust extension (`simplipy._core`) is **required**: the inline phase (`simplify`,
> conversions, validation) runs on it exclusively, and there is no pure-Python fallback. Prebuilt
> wheels are published for Linux (x86_64/aarch64), macOS (x86_64/arm64) and Windows (x64) on
> CPython ≥ 3.11, so `pip install simplipy` does not compile anything for most users. Installing
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

As of 0.6.0 the simplify hot path defers match-time certificates to completed matches
(memoized generationally, never stopping memoization), memoizes whole fixpoint passes and
rule-normal subtrees, and runs on interned token ids (~20× fewer allocations per call) —
all at byte-identical outputs. On a 65,536-expression training-prior benchmark (measured
at 0.11.0), large certificate-bearing rulesets simplify ~59× faster than 0.5.0 and
certificate-free rulesets ~2.3× faster; see the [CHANGELOG](https://github.com/psaegert/simplipy/blob/main/CHANGELOG.md)
for details. Since 0.7.0 there is a single compiled engine line; the published ruleset
artifacts (`acj-2-1`, `acj-3-2`, `acj-4-3`, …) are the distinguishing factor between engines. Rule
application always considers every pattern in the loaded artifact (the former
`max_pattern_length` knob was removed in 0.10.0). (To reproduce the historical dev_7-3 /
v23.0-era behavior byte-for-byte, install `simplipy<=0.6.0`.)

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/psaegert/simplipy/main/assets/images/simplipy_vs_sympy.svg" alt="Simplification time and ratio ECDFs: SimpliPy 0.11.0 vs SymPy across three axes (mined rulesets, safe vs aggressive, search budget) on 64k Lample-Charton expressions from the Flash-ANSR v23.0 prior" width="900">
      <p>Empirical Cumulative Distribution Functions (ECDFs) of simplification wall-clock time (<strong>top row</strong>) and simplification ratio <code>|simp| / |orig|</code> in prefix tokens (<strong>bottom row</strong>, inset: zoom on the low-ratio tail), over 65,536 randomly generated Lample-Charton expressions sampled from the Flash-ANSR v23.0 training prior (0 to 17 unique variables, 1 to 35 symbols <a href="https://arxiv.org/abs/2602.08885">[Saegert & Köthe 2026]</a>). Three axes vary SimpliPy (green) while <strong>SymPy</strong> <a href="https://peerj.com/articles/cs-103/">[Meurer et al. 2017]</a> (orange <code>ratio=None</code> / red <code>ratio=1</code>) is the fixed reference: <strong>Mined Rulesets</strong> — the <code>2-1</code>/<code>3-2</code>/<code>4-3</code> ruleset artifacts, every pattern active (darker = larger); <strong>Safe vs Aggressive</strong> — <code>4-3</code> in the deployed <code>SOUND</code> mode vs the training-only <code>LOSSY</code> mode; <strong>Search Budget</strong> — <code>4-3</code> SOUND at node budgets 1 to 48. SimpliPy is timed per call (<code>perf_counter</code>, gc off); SymPy is given each <code>&lt;constant&gt;</code> as a free symbol and simplified symbolically inside a per-expression worker with a 1 s timeout, then scored by its native prefix length; SymPy&#39;s workers run 24-wide while SimpliPy is timed single-threaded. Expressions SymPy does not finish inside its budget are scored as what they are &mdash; infinite time, ratio 1 (not simplified) &mdash; and stay in the denominator, so its time curve plateaus at the fraction it completed and its ratio curve carries a step at exactly 1; renormalising over only its successes would inflate its low-ratio tail. The measured quantities are the two ECDFs; the plot, not this caption, reports what they show.</p>
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
    year = 2026,
    publisher = {GitHub},
    version = {0.12.0},
    url = {https://github.com/psaegert/simplipy}
}
```
