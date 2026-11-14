# mypy: ignore-errors
"""The contract's composition standing obligation, discharged per vocabulary.

SIMPLIFICATION_CONTRACT_v2.md section 5 (composition, [PROVEN / MACHINE-CHECKED]):
inner-context rule application preserves the null-extension order over THIS operator
table because the only nan-absorbing operator slots -- pow(., 0) and pow(1, .) -- are
constant in the absorbing slot. That check was machine-verified for the LEGACY
35-operator vocabulary and ratified as a STANDING OBLIGATION: "any vocabulary change
must re-check it, because one non-constant nan-absorbing operator breaks composition
silently."

The 0.12.0 vocabulary change (23 config operators + the AC built-in ``rootn``; the
hyper-operator family deleted) had no discharge on record. This module IS the discharge, pinned: it sweeps every
operator slot of the CURRENT serving vocabulary under BOTH algebras -- the contract
evaluator (``verify._contract.c_eval``, mp) and the deployed realization
(``verify._contract.d_eval``, numpy) -- and asserts that

  * every nan-absorbing cell (nan in one slot, non-nan output) is one of the two
    ratified IEEE-definitional cells: pow(., 0) = 1 and pow(1, .) = 1, and
  * each absorbing cell is CONSTANT in the absorbing slot (the property the
    composition proof actually consumes).

A future vocabulary change that adds a non-constant nan-absorbing operator fails
this test instead of breaking composition silently.
"""
import math

import numpy as np
import pytest

from simplipy.verify import _contract as C

# The serving vocabulary: the 23 operators of the shipped config. Enumerated here
# EXPLICITLY -- the point of the pin is that a vocabulary change must come HERE and
# re-justify itself, not inherit the discharge silently; `test_the_lists_are_the_served_
# vocabulary` fails the moment the config and these lists disagree.
UNARY = [
    'neg', 'abs', 'inv', 'exp', 'log',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
]
BINARY = ['+', '-', '*', '/', 'pow', 'rootn']

# The two ratified nan-absorbing cells (v2 section 2: IEEE-definitional, kept) of the deployed
# float64 algebra: (operator, index of the nan slot, value in the other slot, absorbed output).
# The contract's real values absorb nothing: undefined propagates through every operation
# (the table's V6), so `pow(nan, 0)` and `pow(1, nan)` are undefined there and the obligation is
# vacuous in that algebra.
RATIFIED_CELLS = {('pow', 0, 0.0, 1.0), ('pow', 1, 1.0, 1.0)}
ABSORBING = {'contract': set(), 'deployed': RATIFIED_CELLS}

GRID = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, float('inf'), float('-inf')]


def _c_point(op, args):
    toks = [op] + [f'?{i}' for i in range(len(args))]
    tree = C.parse(toks)
    env = {f'?{i}': (a if isinstance(a, float) and (math.isnan(a) or math.isinf(a))
                     else C.mpf(repr(a))) for i, a in enumerate(args)}
    try:
        v = C.c_eval(tree, env)
    except C.Unresolved:
        return None
    return v


def _d_point(op, args):
    toks = [op] + [f'?{i}' for i in range(len(args))]
    tree = C.parse(toks)
    env = {f'?{i}': a for i, a in enumerate(args)}
    with np.errstate(all='ignore'):
        return float(C.d_eval(tree, env))


def _is_nan(v):
    try:
        return C.misnan(v)
    except TypeError:
        return isinstance(v, float) and math.isnan(v)


def _absorbing_cells(point):
    """All (op, nan_slot, other_value, output) cells where nan in produces non-nan out."""
    nan = float('nan')
    cells = []
    for op in UNARY:
        v = point(op, [nan])
        if v is not None and not _is_nan(v):
            cells.append((op, 0, None, float(v)))
    for op in BINARY:
        for slot in (0, 1):
            for other in GRID:
                args = [nan, other] if slot == 0 else [other, nan]
                v = point(op, args)
                if v is not None and not _is_nan(v):
                    cells.append((op, slot, other, float(v)))
    return cells


@pytest.mark.parametrize('algebra,point', [('contract', _c_point), ('deployed', _d_point)])
def test_nan_absorbing_slots_are_exactly_the_ratified_cells(algebra, point):
    cells = _absorbing_cells(point)
    assert set(cells) == ABSORBING[algebra], (
        f'{algebra} algebra: nan-absorption differs from the ratified set -- '
        f'the section-5 composition proof no longer covers this vocabulary: {cells}'
    )


@pytest.mark.parametrize('algebra,point', [('contract', _c_point), ('deployed', _d_point)])
def test_absorbing_cells_are_constant_in_the_absorbing_slot(algebra, point):
    # The property the composition proof consumes: at the absorbing configuration the
    # operator is CONSTANT in the absorbing slot, so a nan flowing into it cannot
    # change the value an outer context sees.
    for op, slot, other, absorbed in sorted(ABSORBING[algebra]):
        outs = set()
        for v in GRID + [float('nan')]:
            args = [v, other] if slot == 0 else [other, v]
            r = point(op, args)
            assert r is not None, f'{algebra}: {op} unresolved at {args}'
            outs.add(float(r))
        assert outs == {absorbed}, (
            f'{algebra} algebra: {op} slot {slot} at other={other} is not constant: {outs}'
        )


def test_the_lists_are_the_served_vocabulary():
    """The discharge covers exactly what is served: every operator of the shipped
    config, with its arity, and nothing else."""
    from conftest import acj_config_path, require_or_skip
    from simplipy.io import load_config
    require_or_skip(acj_config_path(), 'the acj-4 config is not staged')
    served = {name: spec['arity'] for name, spec in load_config(acj_config_path())['operators'].items()}
    assert {**{op: 1 for op in UNARY}, **{op: 2 for op in BINARY}} == served
