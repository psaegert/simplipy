"""The deployed realizations (`simplipy.operators`) are the engine's own kernels.

Every realization evaluates the float64 kernel the engine's evaluator uses for its token
(`simplipy._core.kernel_unary`/`kernel_binary`, the functions `numeric::unary_fn`/`binary_fn`
resolve to), so the deployed value, the constant fold, the miner's evaluator and the rule
judge compute the same bits. These tests pin that on dense grids and every special cell, for
the whole vocabulary, plus the realization-level conventions: scalar types, broadcasting,
complex operands, torch tensors.
"""
import math

import numpy as np
import pytest

import simplipy.operators as ops
from test_operator_kernels import bits_equal, core1, core2, dense_binary_grid, dense_unary_grid

F = np.float64
RNG = np.random.default_rng(1021)
UNARY = ['neg', 'inv', 'abs', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
         'asinh', 'acosh', 'atanh', 'exp', 'log', 'floor', 'ceil', 'harmonic', 'erf']
BINARY = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': 'pow', 'rootn': 'rootn',
          'max': 'max', 'min': 'min', 'mod': 'mod', 'atan2': 'atan2', 'besselj': 'besselj'}


def unary_grid() -> np.ndarray:
    return np.concatenate([dense_unary_grid(), RNG.uniform(-1, 1, 20_000), RNG.uniform(-30, 30, 20_000),
                           RNG.uniform(-800, 800, 5_000)])


# --------------------------------------------------------------------------- one function

@pytest.mark.parametrize('name', UNARY)
def test_unary_realizations_are_the_kernels(name) -> None:
    x = unary_grid()
    with np.errstate(all='ignore'):
        same = bits_equal(getattr(ops, name)(x), core1(name, x))
    assert same.all(), x[~same][:5]


@pytest.mark.parametrize('name, token', BINARY.items())
def test_binary_realizations_are_the_kernels(name, token) -> None:
    a, b = dense_binary_grid()
    if name == 'rootn':
        b = np.where(RNG.random(b.size) < 0.8, np.round(b), b)  # mostly integer indices
    with np.errstate(all='ignore'):
        same = bits_equal(getattr(ops, name)(a, b), core2(token, a, b))
    assert same.all(), list(zip(a[~same][:5], b[~same][:5]))


def test_scalar_operands_take_the_same_kernels() -> None:
    """A Python scalar goes through the kernel's scalar entry: the same function."""
    x = unary_grid()[::577]
    for name in UNARY:
        for u in x:
            assert bits_equal(getattr(ops, name)(float(u)), core1(name, [u])).all(), (name, u)
    a, b = dense_binary_grid()
    for name, token in BINARY.items():
        for u, v in zip(a[::4001], b[::4001]):
            assert bits_equal(getattr(ops, name)(float(u), float(v)), core2(token, u, v)).all(), (name, u, v)


def test_long_arrays_split_over_threads_give_the_same_bits() -> None:
    big = RNG.uniform(-30, 30, 300_000)
    serial = np.concatenate([ops.tanh(chunk) for chunk in np.array_split(big, 300)])  # 1000 each: one thread
    assert bits_equal(ops.tanh(big), serial).all()
    assert bits_equal(ops.atan2(big, big[::-1]), np.concatenate(
        [ops.atan2(u, v) for u, v in zip(np.array_split(big, 300), np.array_split(big[::-1], 300))])).all()


def test_the_judge_reads_the_kernels() -> None:
    """The rule judge's deployed lane evaluates the realizations, so it reads the same bits."""
    from simplipy.verify._contract import DEP_OPS
    x = unary_grid()[::97]
    for name in UNARY:
        if name not in DEP_OPS:
            continue
        with np.errstate(all='ignore'):
            got = np.array([DEP_OPS[name](F(u)) for u in x])
        assert bits_equal(got, core1(name, x)).all(), name


def test_tanh_saturates_where_the_engine_reads_its_float64_tail() -> None:
    """The deployed tanh is the engine's: exactly 1.0 from 19.061547465398498 (glibc's
    threshold, the interval layer's TANH_SAT), and short of 1 below it."""
    sat = 19.061547465398498
    assert ops.tanh(sat) == 1.0 and ops.tanh(-sat) == -1.0
    assert ops.tanh(np.nextafter(sat, 0.0)) < 1.0
    assert ops.atan(5805358775541310.0) == np.pi / 2 and ops.atan(5805358775541309.0) < np.pi / 2


# --------------------------------------------------------------------------- conventions

def test_cells() -> None:
    inf, nan = np.inf, np.nan
    assert bits_equal(ops.max(-0.0, 0.0), 0.0) and bits_equal(ops.max(0.0, -0.0), 0.0)
    assert bits_equal(ops.min(-0.0, 0.0), -0.0) and bits_equal(ops.min(0.0, -0.0), -0.0)
    assert np.isnan(ops.max(nan, 1.0)) and np.isnan(ops.max(1.0, nan)) and np.isnan(ops.min(1.0, nan))
    assert ops.mod(-7, 3) == 2.0 and ops.mod(7, -3) == -2.0 and ops.mod(-3.0, inf) == inf
    assert np.isnan(ops.mod(5, 0)) and np.isnan(ops.mod(inf, 2.0))
    assert bits_equal(ops.mod(-0.0, 3.0), 0.0) and bits_equal(ops.mod(0.0, -3.0), -0.0)
    assert bits_equal(ops.floor(-0.0), -0.0) and bits_equal(ops.ceil(-0.5), -0.0)
    assert bits_equal(ops.harmonic(-0.0), -0.0) and ops.harmonic(3) == 11 / 6
    assert np.isnan(ops.harmonic(-2)) and np.isnan(ops.harmonic(-inf)) and ops.harmonic(inf) == inf
    assert bits_equal(ops.erf(-0.0), -0.0) and ops.erf(inf) == 1.0 and ops.erf(-inf) == -1.0
    assert ops.atan2(0.0, -0.0) == np.pi and bits_equal(ops.atan2(-0.0, 0.0), -0.0)
    assert ops.inv(-0.0) == -inf and ops.div(1.0, -0.0) == -inf and np.isnan(ops.div(0.0, 0.0))
    assert np.isnan(ops.pow(-inf, 0.5)) and ops.pow(-inf, 3.0) == -inf and bits_equal(ops.pow(-0.0, 0.5), 0.0)
    assert ops.rootn(-8.0, 3.0) == -2.0 and np.isnan(ops.rootn(-8.0, 2.0)) and bits_equal(ops.rootn(-0.0, 3), -0.0)


def test_scalar_types() -> None:
    """Scalars give numpy float64 scalars, as numpy's functions do; `abs` keeps a Python
    scalar's own type and `rootn` gives a Python float, as before."""
    for f, args in [(ops.sin, (0.5,)), (ops.tanh, (2,)), (ops.harmonic, (2.0,)), (ops.floor, (2.5,)),
                    (ops.inv, (np.float64(4.0),)), (ops.add, (1, 2)), (ops.max, (1, 2)), (ops.mod, (7, 3)),
                    (ops.pow, (2.0, 0.5)), (ops.atan2, (1.0, 1.0)), (ops.sin, (np.array(0.5),))]:
        assert type(f(*args)) is np.float64, f.__name__
    assert type(ops.abs(-3)) is int and type(ops.abs(-2.5)) is float
    assert type(ops.rootn(8.0, 3.0)) is float and ops.rootn(8.0, 3.0) == 2.0


def test_integer_operands_are_float64() -> None:
    """Integer operands are taken as float64: mod by zero is NaN, not numpy's integer 0."""
    out = ops.mod(np.array([5, -7]), np.array([0, 3]))
    assert out.dtype == F and np.isnan(out[0]) and out[1] == 2.0
    assert ops.floor(np.array([3])).dtype == F and ops.max(np.array([1]), 2).dtype == F
    assert ops.sin(np.array([0, 1])).dtype == F


def test_arrays_broadcast_and_keep_their_shape() -> None:
    m = np.arange(6, dtype=F).reshape(2, 3)
    h = ops.harmonic(m)
    assert h.shape == (2, 3) and bits_equal(h, core1('harmonic', m.ravel()).reshape(2, 3)).all()
    out = ops.max(m, np.array([[2.5], [-1.0]]))
    assert out.shape == (2, 3) and out[0, 0] == 2.5 and out[1, 2] == 5.0
    z = ops.max(np.zeros((2, 2)), -np.zeros((1, 2)))
    assert not np.signbit(z).any()
    z = ops.min(-np.zeros((2, 1)), np.zeros(3))
    assert z.shape == (2, 3) and np.signbit(z).all()
    assert ops.mod(m, 4.0)[1, 1] == 0.0 and ops.add(m, 1.0)[1, 2] == 6.0
    assert ops.pow(2.0, np.array([1.0, 3.0])).tolist() == [2.0, 8.0]
    assert ops.harmonic(np.array([])).shape == (0,) and ops.add(np.array([]), 1.0).shape == (0,)
    assert bits_equal(ops.sin(m[:, ::2]), ops.sin(np.ascontiguousarray(m[:, ::2]))).all()
    assert ops.sin([0.0, 1.0]).shape == (2,)


def test_complex_operands() -> None:
    """The realizations of functions with a complex continuation take numpy's on a complex
    operand (the kernels are real); the 0.15 operators refuse one."""
    z = np.array([1 + 1j, -0.5 + 2j])
    for name, f in [('sin', np.sin), ('exp', np.exp), ('log', np.log), ('tanh', np.tanh)]:
        assert np.array_equal(getattr(ops, name)(z), f(z)), name
    assert ops.abs(3 + 4j) == 5.0 and np.array_equal(ops.abs(z), np.abs(z))
    assert np.array_equal(ops.add(z, 1.0), z + 1.0) and np.array_equal(ops.neg(z), -z)
    for f, args in [(ops.harmonic, (1j,)), (ops.floor, (np.array([1 + 1j]),)), (ops.max, (1.0, 2j)),
                    (ops.mod, (1j, 2.0)), (ops.erf, (1j,)), (ops.atan2, (1j, 1.0)),
                    (ops.besselj, (1.0, 1j))]:
        with pytest.raises(TypeError, match='real numbers'):
            f(*args)


def test_torch_branches() -> None:
    torch = pytest.importorskip('torch')
    t = torch.tensor([-2.5, -0.0, 0.0, 1.5, 3.0], dtype=torch.float64)
    assert torch.equal(ops.floor(t), torch.floor(t)) and torch.equal(ops.ceil(t), torch.ceil(t))
    assert torch.equal(ops.sin(t), torch.sin(t)) and torch.equal(ops.add(t, t), t + t)
    pos, neg = torch.tensor([0.0, -0.0], dtype=torch.float64), torch.tensor([-0.0, 0.0], dtype=torch.float64)
    assert not torch.signbit(ops.max(pos, neg)).any() and torch.signbit(ops.min(pos, neg)).all()
    a = torch.tensor([-7.0, 7.0, -3.0, -0.0, 5.0], dtype=torch.float64)
    b = torch.tensor([3.0, -3.0, float('inf'), 3.0, 0.0], dtype=torch.float64)
    assert bits_equal(ops.mod(a, b).numpy(), ops.mod(a.numpy(), b.numpy())).all()
    h = ops.harmonic(torch.tensor([0.0, -0.0, 1.0, -2.0, 2.5], dtype=torch.float64))
    assert h[0] == 0 and torch.signbit(h[1]) and abs(float(h[2]) - 1) < 1e-15 and torch.isnan(h[3])
    assert abs(float(h[4]) - float(ops.harmonic(2.5))) < 1e-14
    assert math.isclose(float(ops.erf(torch.tensor(0.5, dtype=torch.float64))), float(ops.erf(0.5)), rel_tol=1e-15)
    with pytest.raises(TypeError, match='no torch realization'):
        ops.besselj(t, t)


def test_the_acj2_4_config_is_acj_4_plus_the_new_operators() -> None:
    """acj2-4's vocabulary is acj-4's plus the nine 0.15 operators, every realization one of
    `simplipy.operators`; it loads (without rule sets until its mines write them) and reads
    the new calls."""
    import importlib
    import os
    import warnings

    import yaml

    import simplipy as sp

    root = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines')
    with open(os.path.join(root, 'acj-4', 'config.yaml')) as f:
        old = yaml.safe_load(f)['operators']
    path = os.path.join(root, 'acj2-4', 'config.yaml')
    with open(path) as f:
        new = yaml.safe_load(f)['operators']
    added = {'floor', 'ceil', 'erf', 'harmonic', 'max', 'min', 'mod', 'atan2', 'besselj'}
    assert set(new) == set(old) | added
    for token, spec in new.items():
        module, name = spec['realization'].rsplit('.', 1)
        assert module == 'simplipy.operators', token
        assert callable(getattr(importlib.import_module(module), name)), token
        if token in old:
            assert {k: v for k, v in spec.items() if k != 'realization'} == \
                {k: v for k, v in old[token].items() if k != 'realization'}, token
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        engine = sp.SimpliPyEngine.from_config(path)
    assert engine.simplify('max(x1, x2, x3)') == 'max(x1, max(x2, x3))'
    assert engine.simplify('besselj(2, x1) + 0') == 'besselj(2, x1)'
