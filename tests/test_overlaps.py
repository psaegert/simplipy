"""Overlaps of a rule set: the instances where two rules' left sides unify (the critical pairs
of completion), the ones whose two paths end in different states, and the completion between
them."""
import os

import yaml

from simplipy import SimpliPyEngine
from simplipy.engine import _CORE_SERIALIZATION_ARITIES
from simplipy.mining import (RuleMiner, _cell_literals, _critical_instances, _overlap_gate,
                             _unjoined_overlaps)

VOCAB32 = os.path.join(os.path.dirname(__file__), 'fixtures', 'vocab32', 'config.yaml')


def _engine(rule_mode, rules):
    ops = yaml.safe_load(open(VOCAB32))['operators']
    engine = SimpliPyEngine(operators=ops, rules=[])
    RuleMiner(engine)._install(rule_mode, rules)
    return engine


def test_a_critical_instance_specializes_a_slot():
    """`acos(cos(asin(_0)))` and `cos(asin(sin(_0)))` overlap where the first rule's slot is
    `sin(x)`: unification finds the instance no generic instance of either rule holds."""
    rules = [(('acos', 'cos', 'asin', '_0'), ('abs', 'asin', '_0')),
             (('cos', 'asin', 'sin', '_0'), ('abs', 'cos', '_0'))]
    ops = yaml.safe_load(open(VOCAB32))['operators']
    arity = {**_CORE_SERIALIZATION_ARITIES, **{k: v['arity'] for k, v in ops.items()}}
    found = {(i, ' '.join(s), ' '.join(r)) for i, s, r in _critical_instances(rules, arity)}
    assert (0, 'acos cos asin sin x0', 'abs asin sin x0') in found, found


def test_a_variable_slot_binds_no_composite():
    """A `?` slot binds a variable leaf only, so no instance binds it to a composite."""
    rules = [(('acos', 'cos', 'asin', '?0'), ('abs', 'asin', '?0')),
             (('cos', 'asin', 'sin', '_0'), ('abs', 'cos', '_0'))]
    ops = yaml.safe_load(open(VOCAB32))['operators']
    arity = {**_CORE_SERIALIZATION_ARITIES, **{k: v['arity'] for k, v in ops.items()}}
    assert not any(i == 0 for i, _, _ in _critical_instances(rules, arity))


def test_the_unjoined_overlap_names_its_completion():
    """`sin(acos(tanh(x)))`: the longer rule takes it to `1/cosh(x)`, the shorter to
    `cos(asin(tanh(x)))`, which nothing continues; the completion runs from the dearer state to
    the cheaper, and once the file holds it the overlap joins."""
    rules = [(('sin', 'acos', '_0'), ('cos', 'asin', '_0')),
             (('sin', 'acos', 'tanh', '_0'), ('inv', 'cosh', '_0'))]
    found = _unjoined_overlaps(_engine('default', rules), 'default', rules)
    assert {(' '.join(o['completion'][0]), ' '.join(o['completion'][1])) for o in found} \
        == {('cos asin tanh x0', 'inv cosh x0')}, found
    rules.append((('cos', 'asin', 'tanh', '_0'), ('inv', 'cosh', '_0')))
    assert _unjoined_overlaps(_engine('default', rules), 'default', rules) == []


def test_the_gate_judges_the_cell_and_counts_the_rest():
    """The completion `cos asin tanh x0 -> inv cosh x0` has a four-token source: a mine of
    sources to length 4 holds it, so the gate judges it; one of sources to length 3 does not
    enumerate it, so the gate counts it beyond the cell and judges nothing."""
    rules = [(('sin', 'acos', '_0'), ('cos', 'asin', '_0')),
             (('sin', 'acos', 'tanh', '_0'), ('inv', 'cosh', '_0'))]
    asked = []

    def judge(pairs):
        asked.extend(pairs)
        return ['core'] * len(pairs)

    engine = _engine('default', rules)
    missing, refused, beyond, _ = _overlap_gate(engine, 'default', rules, judge, max_source=4)
    assert [' '.join(o['completion'][0]) for o in missing] == ['cos asin tanh x0'], missing
    assert refused == [] and beyond == [] and len(asked) == 1
    asked.clear()
    missing, refused, beyond, _ = _overlap_gate(engine, 'default', rules, judge, max_source=3)
    assert missing == [] and refused == [] and asked == []
    assert [' '.join(o['completion'][0]) for o in beyond] == ['cos asin tanh x0'], beyond


def test_a_number_the_canon_folds_lies_beyond_the_cell():
    """`floor(x) + 1 -> floor(1 + x)` and `floor(10 - x) -> 10 - ceil(x)` overlap on
    `floor(10 - x) + 1`. The first rule's path ends in `floor(11 - x)`, the canon writing 1 + 10
    as one number, so the completion `floor(11 - x0) -> 11 - ceil(x0)` has a source of four
    tokens with a literal the universe lacks: a mine of sources to length 4 over the literals
    0..10 does not enumerate it, and the gate counts it beyond the cell and judges nothing."""
    rules = [(('+', 'floor', '_0', '1'), ('floor', '+', '1', '_0')),
             (('floor', '-', '10', '_0'), ('-', '10', 'ceil', '_0'))]
    asked = []

    def judge(pairs):
        asked.extend(pairs)
        return ['core'] * len(pairs)

    engine = _engine('default', rules)
    found = {(' '.join(o['completion'][0]), ' '.join(o['completion'][1]))
             for o in _unjoined_overlaps(engine, 'default', rules)}
    assert ('floor - 11 x0', '- 11 ceil x0') in found, found
    literals = _cell_literals(engine, ['x0'] + [str(k) for k in range(11)], 'default')
    missing, refused, beyond, _ = _overlap_gate(engine, 'default', rules, judge, max_source=4,
                                                literals=literals)
    assert 'floor - 11 x0' in [' '.join(o['completion'][0]) for o in beyond], beyond
    assert not any('11' in o['completion'][0] for o in missing + refused)
    assert not any('11' in lhs for lhs, _ in asked)
    missing, refused, _, _ = _overlap_gate(engine, 'default', rules, judge, max_source=4)
    assert 'floor - 11 x0' in [' '.join(o['completion'][0]) for o in missing + refused]
