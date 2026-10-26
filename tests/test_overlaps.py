"""Overlaps of a rule set: the instances where two rules' left sides unify (the critical pairs
of completion), the ones whose two paths end in different states, and the completion between
them."""
import os

import yaml

from simplipy import SimpliPyEngine
from simplipy.engine import _CORE_SERIALIZATION_ARITIES
from simplipy.mining import RuleMiner, _critical_instances, _unjoined_overlaps

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')


def _engine(rule_mode, rules):
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    engine = SimpliPyEngine(operators=ops, rules=[])
    RuleMiner(engine)._install(rule_mode, rules)
    return engine


def test_a_critical_instance_specializes_a_slot():
    """`acos(cos(asin(_0)))` and `cos(asin(sin(_0)))` overlap where the first rule's slot is
    `sin(x)`: unification finds the instance no generic instance of either rule holds."""
    rules = [(('acos', 'cos', 'asin', '_0'), ('abs', 'asin', '_0')),
             (('cos', 'asin', 'sin', '_0'), ('abs', 'cos', '_0'))]
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    arity = {**_CORE_SERIALIZATION_ARITIES, **{k: v['arity'] for k, v in ops.items()}}
    found = {(i, ' '.join(s), ' '.join(r)) for i, s, r in _critical_instances(rules, arity)}
    assert (0, 'acos cos asin sin x0', 'abs asin sin x0') in found, found


def test_a_variable_slot_binds_no_composite():
    """A `?` slot binds a variable leaf only, so no instance binds it to a composite."""
    rules = [(('acos', 'cos', 'asin', '?0'), ('abs', 'asin', '?0')),
             (('cos', 'asin', 'sin', '_0'), ('abs', 'cos', '_0'))]
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    arity = {**_CORE_SERIALIZATION_ARITIES, **{k: v['arity'] for k, v in ops.items()}}
    assert not any(i == 0 for i, _, _ in _critical_instances(rules, arity))


def test_the_unjoined_overlap_names_its_completion():
    """`sin(acos(tanh(x)))`: the longer rule takes it to `1/cosh(x)`, the shorter to
    `cos(asin(tanh(x)))`, which nothing continues; the completion runs from the dearer state to
    the cheaper, and once the file holds it the overlap joins."""
    rules = [(('sin', 'acos', '_0'), ('cos', 'asin', '_0')),
             (('sin', 'acos', 'tanh', '_0'), ('inv', 'cosh', '_0'))]
    found = _unjoined_overlaps(_engine('real', rules), 'real', rules)
    assert {(' '.join(o['completion'][0]), ' '.join(o['completion'][1])) for o in found} \
        == {('cos asin tanh x0', 'inv cosh x0')}, found
    rules.append((('cos', 'asin', 'tanh', '_0'), ('inv', 'cosh', '_0')))
    assert _unjoined_overlaps(_engine('real', rules), 'real', rules) == []
