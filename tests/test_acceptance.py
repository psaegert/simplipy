"""Refusal at acceptance: every rule the mine considers -- a pick, a proposal, a completion, a
promoted sort, a seed -- is decided where the mine first considers it, by the licence (the
judge's tier and the one policy predicate), the entries it serves and liveness, and is never
dropped afterwards. Finalization fails, naming the offender, on a file a rule of which would
fail a gate."""
import json

import pytest

from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner, _dead_rules, judge_tiers

_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1,
          "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2,
          "commutative": True},
    "asin": {"realization": "np.arcsin", "alias": [], "inverse": "sin", "arity": 1,
             "precedence": 3, "commutative": False},
    "acos": {"realization": "np.arccos", "alias": [], "inverse": "cos", "arity": 1,
             "precedence": 3, "commutative": False},
}

#: True, and refused by the ground-composite policy: two ground spellings of pi/2.
SEED = (('asin', '1'), ('acos', '0'))


def _mine(tmp_path, eng, **kwargs):
    out = str(tmp_path / 'mined.json')
    eng.find_rules(max_source_pattern_length=2, dummy_variables=1,
                   extra_internal_terms=['0', '1'], X=128, seed=7, verbose=False,
                   output_file=out, reset_rules=False, modes='f64', judge_processes=1,
                   **kwargs)
    return out


def test_a_refused_seed_is_never_written(tmp_path) -> None:
    """A continuation mine seeded with `asin(1) -> acos(0)`, true and refused by the ground
    composite policy, starts without it: the seed goes through the mine's acceptance, the
    sidecar names the refusal, and the file never holds it."""
    eng = SimpliPyEngine(operators=_OPERATORS, rules=[SEED])
    out = _mine(tmp_path, eng)
    rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(open(out))}
    assert SEED not in rules
    seeds = json.load(open(out + '.provenance.json'))['seeds']
    assert seeds['given'] == 1 and seeds['refused'] == [[list(SEED[0]), list(SEED[1]), 'licence']]


def test_finalization_fails_naming_an_offender(tmp_path, monkeypatch) -> None:
    """Nothing is dropped after acceptance: where a rule the file would hold fails a gate (here
    the refused seed, let through by a planted acceptance that refuses nothing), finalization
    fails and names it, and no file is written."""
    monkeypatch.setattr(RuleMiner, '_refusals', lambda self, *args, **kwargs: {})
    eng = SimpliPyEngine(operators=_OPERATORS, rules=[SEED])
    with pytest.raises(RuntimeError, match=r"unlicensed.*asin"):
        _mine(tmp_path, eng)
    assert not (tmp_path / 'mined.json').exists()


@pytest.fixture(scope='module')
def acj() -> SimpliPyEngine:
    import os
    import yaml
    config = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4',
                          'config.yaml')
    return SimpliPyEngine(operators=yaml.safe_load(open(config))['operators'], rules=[],
                          rules_permissive=[])


@pytest.mark.parametrize('src, hint, mode, policy', [
    # below the mark in permissive's reading (the inverse pair cancels), not ordered below as
    # written: the load's ordering refuses it
    ('abs x0', 'acosh cosh x0', 'permissive', 'ordering'),
    # a ground left side without a value has no evidence (4.1)
    ('cos acos acos -1', 'float("nan")', 'permissive', 'evidence'),
])
def test_the_mint_refuses_what_the_licence_refuses(acj, src, hint, mode, policy) -> None:
    """The mint's acceptance reads the licence's policies on the pick's written pattern -- the
    load's predicate and evidence -- so a pick the licence refuses is never proposed."""
    import numpy as np
    from simplipy.verify import licensed
    x = RuleMiner(acj)._mining_sample_x(256, 1, np.random.default_rng(3))
    lib = acj._core.build_candidate_library([['x0']], ['x0'], x.flatten().tolist(), x.shape[0],
                                            rule_mode=mode)
    ok, why = acj._core.judge_hint(src.split(), hint.split(), lib, rule_mode=mode)
    assert (ok, why) == (False, policy)
    pattern = [t.replace('x0', '?0') for t in src.split()]
    target = [t.replace('x0', '?0') for t in hint.split()]
    got = licensed(pattern, target, 'f64' if mode == 'default' else mode, acj)
    assert policy in got['refused_by'], got


def test_a_promoted_sort_that_does_not_fire_is_refused(acj) -> None:
    """In permissive, `pow(0, inv(?0)) -> 0` holds at every sort, and at `_` it does not fire on
    its own generic instance: `0^(1/(1/(x0 + x1)))` is cancelled to `0^(x0 + x1)` before the
    pattern can match. The `_` rung goes through the mine's acceptance, which refuses it, and the
    rule ships at a sort where it fires."""
    from simplipy.promotion import promote
    miner = RuleMiner(acj)
    rule = (('pow', '0', 'inv', '?0'), ('0',))

    def accept(pairs, context):
        return set(miner._refusals([(tuple(lhs), tuple(rhs)) for lhs, rhs in pairs],
                                   'permissive', judge_tiers, {}, context))

    kept, _ = promote([rule], acj, rule_mode='permissive', accept=accept,
                      run_positive_controls=False)
    kept = [(tuple(lhs), tuple(rhs)) for lhs, rhs in kept]
    assert (('pow', '0', 'inv', '_0'), ('0',)) not in kept, kept
    assert any(lhs[:3] == ('pow', '0', 'inv') for lhs, _ in kept), kept
    miner._install('permissive', kept)
    try:
        assert _dead_rules(acj, 'permissive', kept) == []
    finally:
        miner._install('permissive', [])
