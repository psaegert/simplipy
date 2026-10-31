"""One policy implementation: the load, `licensed`, `verify_ruleset` and `verify_artifact` read
the same predicate (the load's gates: the constant count, the ground composite, the ordering,
what the engine does itself, well-formedness). The load checks policies, never truth.

`asin(1) -> acos(0)` holds in every mode (both sides are pi/2) and is refused by the
ground-composite policy: a rule file holding it loads without it, and no gate calls the file
clean.
"""
import json
import os

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.verify import licensed, load_policy_refusal, verify_artifact, verify_ruleset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPERATORS = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')))[
    'operators']
L, R = ['asin', '1'], ['acos', '0']
RULE_MODE = {'f64': 'default', 'permissive': 'permissive'}


@pytest.fixture(scope='module')
def loaded(tmp_path_factory) -> SimpliPyEngine:
    """An artifact whose two files hold the true, policy-refused rule, each with its mode."""
    folder = tmp_path_factory.mktemp('policy')
    config = {'engine_generation': 2, 'operators': OPERATORS}
    for mode, key in (('f64', 'rules'), ('permissive', 'rules_permissive')):
        name = f'rules_{mode}.json'
        config[key] = name
        (folder / name).write_text(json.dumps([[L, R]]))
        (folder / (name + '.provenance.json')).write_text(json.dumps({'mode': mode}))
    (folder / 'config.yaml').write_text(yaml.safe_dump(config))
    return SimpliPyEngine.load(str(folder / 'config.yaml'))


@pytest.mark.parametrize('mode', ['f64', 'permissive'])
def test_the_load_refuses_a_ground_composite(loaded, mode) -> None:
    rule_mode = RULE_MODE[mode]
    assert loaded._core.ac_served_rules_in_mode(rule_mode=rule_mode) == []
    assert loaded._core.ac_rules_drop_census(rule_mode=rule_mode) == {'ground-composite': 1}
    assert loaded.simplify(list(L), mode=mode, effort=0) != R
    assert loaded._core.load_refusal(L, R, rule_mode=rule_mode) == 'ground-composite'


@pytest.mark.parametrize('mode', ['f64', 'permissive'])
def test_licensed_reads_the_loads_predicate(loaded, mode) -> None:
    assert load_policy_refusal(L, R, mode, loaded) == 'ground_composite'
    got = licensed(L, R, mode, loaded)
    assert got['holds'] is True and got['licensed'] is False, got
    assert got['refused_by'] == ['ground_composite'], got


@pytest.mark.parametrize('mode', ['f64', 'permissive'])
def test_verify_ruleset_reads_the_same_predicate(mode) -> None:
    report = verify_ruleset([[L, R]], mode=mode, judge_timeout_s=0)
    assert report['is_clean'] is False
    [offender] = report['offenders']
    assert offender['tier'] == 'core' and offender['refused_by'] == ['ground_composite'], offender


def test_verify_artifact_reads_the_same_predicate() -> None:
    report = verify_artifact([[L, R]], [[L, R]], judge_timeout_s=0)
    assert report['is_clean'] is False
    assert all(not r['is_clean'] for r in report['modes'].values())


def test_the_load_checks_policies_not_truth(loaded) -> None:
    """A hand-written false rule loads: truth is decided when a file is mined and verified."""
    assert loaded._core.load_refusal(['sin', '?0'], ['?0'], rule_mode='default') is None
    assert load_policy_refusal(['sin', '?0'], ['?0'], 'f64', loaded) is None
    report = verify_ruleset([[['sin', '?0'], ['?0']]], mode='f64', judge_timeout_s=0)
    assert [o['refused_by'] for o in report['offenders']] == [[]]
