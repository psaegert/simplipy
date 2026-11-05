"""The public helpers: operator wrappers dispatch on both operands and recognise Tensor
subclasses; generated parameter names avoid the supplied ones; a placeholder is a whole
identifier; uniform sampling works beyond int64 counts."""
import os

import numpy as np
import pytest
import yaml

import simplipy.operators as ops
from simplipy import SimpliPyEngine, explicit_constant_placeholders
from simplipy.utils import (_uniform_below, codify, count_expressions, is_constant_placeholder,
                            sample_expression, substitute_constants)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACJ2_4 = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')))[
    'operators']


def test_a_tensor_on_either_side_takes_torch() -> None:
    torch = pytest.importorskip('torch')
    t = torch.tensor([2.0, 3.0], dtype=torch.float64)
    got = ops.pow(2.0, t)
    assert isinstance(got, torch.Tensor)
    assert torch.equal(got, torch.tensor([4.0, 8.0], dtype=torch.float64))
    got = ops.rootn(-8.0, torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert isinstance(got, torch.Tensor)
    assert torch.equal(got, torch.tensor([-8.0, -2.0], dtype=torch.float64))


def test_a_tensor_with_a_gradient_and_a_parameter_keep_their_graph() -> None:
    torch = pytest.importorskip('torch')
    g = torch.tensor([2.0, 3.0], requires_grad=True)
    out = ops.pow(2.0, g)
    assert out.requires_grad
    out.sum().backward()
    assert torch.allclose(g.grad, torch.tensor([4.0, 8.0]) * float(np.log(2.0)))
    p = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
    assert ops._is_tensor(p)
    s = ops.sin(p)
    assert isinstance(s, torch.Tensor) and s.requires_grad
    assert torch.equal(s.detach(), torch.sin(p).detach())


def test_array_like_operands_of_rootn() -> None:
    assert np.array_equal(ops.rootn([-8.0, -27.0], 3.0), [-2.0, -3.0])
    assert np.array_equal(ops.rootn(-8.0, [1.0, 3.0]), [-8.0, -2.0])
    assert isinstance(ops.rootn(-8.0, 3.0), float)


def test_generated_names_avoid_the_supplied_ones() -> None:
    expr, names = explicit_constant_placeholders(['+', '<constant>', '<constant>'],
                                                 constants=['C_0'])
    assert expr == ['+', 'C_0', 'C_1'] and names == ['C_0', 'C_1']
    f = codify('C_0 + C_1', variables=names)
    assert f is not None
    expr, names = explicit_constant_placeholders(['+', '<constant>', '*', '<constant>', '<constant>'],
                                                 constants=['C_1', 'C_0'])
    assert names == ['C_1', 'C_0', 'C_2'] and len(set(names)) == 3


def test_a_placeholder_is_a_whole_identifier() -> None:
    x = ['+', 'C_0feature', 'x0']
    assert explicit_constant_placeholders(x) == (['+', 'C_0feature', 'x0'], [])
    assert substitute_constants(x, [17]) == x
    assert not is_constant_placeholder('C_0feature') and is_constant_placeholder('C_12')
    assert substitute_constants(['*', 'C_0', 'C_0feature'], [17]) == ['*', '17', 'C_0feature']


def test_sampling_is_uniform_beyond_int64_counts() -> None:
    arity = {name: row['arity'] for name, row in ACJ2_4.items()}
    leaves = ['x0', 'x1', '<constant>']
    counts = count_expressions(len(leaves), arity, 15)
    assert counts[15] > np.iinfo(np.int64).max
    draws = [sample_expression(15, leaves, arity, counts, np.random.default_rng(42)) for _ in range(2)]
    assert draws[0] == draws[1] and len(draws[0]) == 15, 'deterministic in the seed'
    engine = SimpliPyEngine(operators=ACJ2_4, rules=[])
    assert engine.is_valid(list(draws[0]))
    # below 2^63 the draw is numpy's own, so every seed already in use keeps its expression
    assert _uniform_below(10 ** 6, np.random.default_rng(7)) == \
        int(np.random.default_rng(7).integers(10 ** 6))
    # beyond it: unbiased, below the bound, deterministic
    n = 3 * 2 ** 70 + 1
    rng = np.random.default_rng(0)
    xs = [_uniform_below(n, rng) for _ in range(3000)]
    assert all(0 <= x < n for x in xs)
    thirds = np.bincount([min(2, 3 * x // n) for x in xs], minlength=3)
    assert thirds.min() > 900, thirds
