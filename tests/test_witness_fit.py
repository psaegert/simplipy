"""The judge's witness fit: its seeds are the left side's own constants, each assignment of them
to the right side's constants generated directly (never the n! permutations they come from),
and the search in the reals solves a root again from an environment it misses, where the
environments it was solved at do not pin it."""
import itertools
import math

import numpy as np
import pytest

from simplipy.verify import _contract


def _reference_seeds(cl_env, m):
    """The seeds as they were read off every permutation of the left side's constants: each
    permutation's first `m` (padded with 1.0), the first of equal ones kept."""
    lv = [float(v) for v in cl_env.values()]
    own = list(dict.fromkeys(tuple((list(p) + [1.0] * m)[:m])
                             for p in itertools.permutations(lv)))
    return [list(p) for p in own] + [[1.0] * m, [0.5] * m, [-1.0] * m, [2.0] * m,
                                     [float(i + 1) for i in range(m)], [0.0] * m]


def _cases():
    rng = np.random.default_rng(20261101)
    pools = [[2.5, -1.5, 3.0, 0.5, -0.7, 1.0, -1.0, 0.0], [1.0, 1.0, 2.0, 2.0, -0.0, 0.0, 3.0, 3.0]]
    out = []
    for n in range(0, 9):
        for m in range(1, max(n, 3) + 1):
            for pool in pools:
                out.append((list(rng.choice(pool, size=n)), m))
                out.append((pool[:n], m))
    return out


@pytest.mark.parametrize('values, m', _cases())
def test_the_seeds_are_the_prefix_assignments(values, m) -> None:
    """The same seeds, in the same order, as every permutation's prefix: on a generated set of
    left sides of up to eight constants, with repeated values and both zeros."""
    cl_env = {f'<C_L{k}>': v for k, v in enumerate(values)}
    got = _contract._fit_seeds(cl_env, m)
    want = _reference_seeds(cl_env, m)
    assert [[repr(x) for x in s] for s in got] == [[repr(x) for x in s] for s in want]


def test_no_permutation_of_the_left_sides_constants_is_enumerated(monkeypatch) -> None:
    """Eight distinct constants and two on the right: 56 assignments, generated directly; the n!
    = 40,320 permutations they used to be read off are never visited."""
    def refuse(*args, **kwargs):
        raise AssertionError('itertools.permutations was called')
    monkeypatch.setattr(_contract.itertools, 'permutations', refuse)
    cl_env = {f'<C_L{k}>': float(k + 1) for k in range(8)}
    seeds = _contract._fit_seeds(cl_env, 2)
    assert len(seeds) == 56 + 6
    assert seeds[:3] == [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]


def test_the_witness_search_starts_again_where_a_root_misses() -> None:
    """`max(x0, log(C)) -> max(D, x0)`: at C = 0.001 the witness is log(0.001) = -6.9, below every
    generic value of x0, so the first environment where the left side is finite (x0 = 1.7) does
    not pin it; one ten times farther out (x0 = -13) does, and the search solves there after the
    roots of the first ones miss it. The rule is refused only at C = 0, where `max(x0, log(0))`
    is x0, a value no finite D gives: there the search ends at D = -13, which no other
    environment reads, and finds no witness."""
    lhs, rhs = 'max ?0 log <constant>'.split(), 'max <constant> ?0'.split()
    tl = _contract.parse(lhs, '<C_L>', number=True)
    tr = _contract.parse(rhs, '<C_R>', number=True)
    slots = set(_contract.slots_of(lhs, '<C_L>', number=True)) - set(
        _contract._constant_names(lhs, '<C_L>', True))
    generic = [_contract._gen_env(slots, k) for k in range(len(_contract.GEN))]
    farther = [{n: np.float64(float(v) * 10) for n, v in env.items()} for env in generic]
    names = _contract._constant_names(rhs, '<C_R>', True)
    lnames = _contract._constant_names(lhs, '<C_L>', True)
    with _contract._own_precision():
        found = _contract._fit_multi_mp(tl, tr, generic + farther,
                                        _contract._cl_env({lnames[0]: 0.001}), names, strict=True)
        boundary = _contract._fit_multi_mp(tl, tr, generic + farther,
                                           _contract._cl_env({lnames[0]: 0.0}), names, strict=True)
    assert found is not None and math.isclose(found[0][0], math.log(0.001), rel_tol=1e-12), found
    assert boundary is None, boundary
    r = _contract.judge_rule('max x0 log <constant>'.split(), 'max <constant> x0'.split())
    assert r['verdict'] == 'NO-WITNESS' and 'cl=[(0.0,)]' in r['detail'], r
