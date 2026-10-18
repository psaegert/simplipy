"""`simplipy verify-mine`: the re-mine gates of one mined file in its own mode (miner.md 3).
A clean mine passes every gate; each gate fails on a planted violation of its own."""
import copy
import json

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.mining import verify_mine

_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
}
FILES = {'f64': 'mined.json', 'real': 'mined_real.json', 'permissive': 'mined_permissive.json'}


def core_judge(pairs):
    return ['core'] * len(pairs)


@pytest.fixture(scope='module')
def mine(tmp_path_factory):
    d = tmp_path_factory.mktemp('verify_mine')
    (d / 'rules.json').write_text('[]')
    (d / 'config.yaml').write_text(yaml.safe_dump(
        {'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(d / 'config.yaml'))
    engine.find_rules(max_source_pattern_length=3, max_target_pattern_length=3, dummy_variables=1,
                      extra_internal_terms=['0', '1', '2'], X=256, seed=7, verbose=False,
                      output_file=str(d / 'mined.json'), judge_processes=1)
    return d


def _plant(tmp_path, mine, mode, rules, sidecar_edit=None):
    """A copy of the clean file of `mode` with `rules` in place of its rows; its sidecar is
    the clean one, edited by `sidecar_edit`."""
    path = tmp_path / FILES[mode]
    path.write_text(json.dumps([[list(lhs), list(rhs)] for lhs, rhs in rules]))
    side = json.load(open(mine / (FILES[mode] + '.provenance.json')))
    if sidecar_edit is not None:
        sidecar_edit(side)
    (tmp_path / (FILES[mode] + '.provenance.json')).write_text(json.dumps(side))
    return str(path)


@pytest.mark.parametrize('mode', ['f64', 'real', 'permissive'])
def test_a_clean_mine_passes_every_gate(mine, mode) -> None:
    report = verify_mine(str(mine / FILES[mode]), mode, _OPERATORS, judge_processes=1)
    assert report['ok'], report['gates']
    assert json.load(open(mine / FILES[mode])), 'vacuous: an empty file'


def _gate(report, name):
    return report['gates'][name]


def test_the_twin_shadow_gate(tmp_path, mine) -> None:
    from conftest import acj_config_path, require_or_skip
    require_or_skip(acj_config_path(), 'needs the acj vocabulary')
    ops = yaml.safe_load(open(acj_config_path()))['operators']
    sib = [(('-', 'acosh', 'atanh', '?0', '?1'), ('-', '?0', '?1')),
           (('-', '?1', 'acosh', 'atanh', '?0'), ('-', '?1', '?0'))]
    path = _plant(tmp_path, mine, 'f64', sib)
    gate = _gate(verify_mine(path, 'f64', ops, judge=core_judge), 'twin_shadow')
    assert not gate['ok'] and gate['examples'] == [[list(sib[1][0]), list(sib[1][1])]]
    path = _plant(tmp_path, mine, 'f64', sib[:1])
    assert _gate(verify_mine(path, 'f64', ops, judge=core_judge), 'twin_shadow')['ok']


def test_the_engine_owned_law_gate(tmp_path, mine) -> None:
    """The distributive law (factoring) and a canon-owned rule are the engine's."""
    factor = (('+', '*', '?0', '?1', '*', '?2', '?1'), ('*', '+', '?0', '?2', '?1'))
    canon = (('*', '?0', '1'), ('?0',))
    mined = (('*', 'exp', '?0', 'exp', '?0'), ('exp', '*', '2', '?0'))  # a rule of its own
    path = _plant(tmp_path, mine, 'real', [factor, canon, mined])
    gate = _gate(verify_mine(path, 'real', _OPERATORS, judge=core_judge), 'engine_owned_law')
    assert gate['examples'] == [[list(factor[0]), list(factor[1]), 'engine-owned-law'],
                                [list(canon[0]), list(canon[1]), 'canon']], gate


def test_the_engine_owned_law_gate_catches_the_sign_pull(tmp_path, mine) -> None:
    """A sign pull through an odd function is the sign orbit, which the engine owns: the core
    places that sign itself, so both sides are one canonical state and the gate flags the rule
    as the canon's."""
    ops = dict(_OPERATORS, asin={"realization": "np.arcsin", "alias": [], "inverse": "sin",
                                 "arity": 1, "precedence": 3, "commutative": False})
    pull = (('neg', 'asin', '_0'), ('asin', 'neg', '_0'))
    path = _plant(tmp_path, mine, 'f64', [pull])
    gate = _gate(verify_mine(path, 'f64', ops, judge=core_judge), 'engine_owned_law')
    assert not gate['ok'], gate


def test_the_unlicensed_gate(tmp_path, mine) -> None:
    """An f64 file holding a rule the judge files `real`."""
    rule = (('log', 'exp', '?0'), ('?0',))
    path = _plant(tmp_path, mine, 'f64', [rule])

    def judge(pairs):
        return ['real' if tuple(map(tuple, p)) == rule else 'core' for p in pairs]

    gate = _gate(verify_mine(path, 'f64', _OPERATORS, judge=judge), 'unlicensed')
    assert not gate['ok'] and gate['examples'] == [[list(rule[0]), list(rule[1]), 'real']]
    # the same file read as `real`, which licenses the tier
    path = _plant(tmp_path, mine, 'real', [rule])
    assert _gate(verify_mine(path, 'real', _OPERATORS, judge=judge), 'unlicensed')['ok']


def test_the_preempted_gate(tmp_path, mine) -> None:
    """A ground rule f64's constructor performs (`exp 0 -> 1`)."""
    path = _plant(tmp_path, mine, 'f64', [(('exp', '0'), ('1',))])
    gate = _gate(verify_mine(path, 'f64', _OPERATORS, judge=core_judge), 'preempted')
    assert not gate['ok'] and gate['count'] == 1, gate


def test_the_translation_gate(tmp_path, mine) -> None:
    """A rule the load subsumes by exact arithmetic (`* 1 ?0 -> ?0`)."""
    path = _plant(tmp_path, mine, 'f64', [(('*', '1', '?0'), ('?0',))])
    gate = _gate(verify_mine(path, 'f64', _OPERATORS, judge=core_judge), 'translation')
    assert not gate['ok'] and gate['examples'][0]['subsumed'] == 1, gate


def test_the_fingerprint_gate(tmp_path, mine) -> None:
    rules = json.load(open(mine / FILES['real']))

    def move_a_probe(side):
        probe = sorted(side['measure']['probes'])[0]
        side['measure']['probes'][probe] += 1

    path = _plant(tmp_path, mine, 'real', rules, move_a_probe)
    gate = _gate(verify_mine(path, 'real', _OPERATORS, judge=core_judge), 'fingerprint')
    assert not gate['ok'] and gate['count'] == 1, gate


def _unfinished(side):
    side['progress']['final'] = False


def _interrupted(side):
    side['progress']['interrupted'] = True


def _another_mode(side):
    side['mode'] = 'real'


def _no_probes(side):
    side['measure'] = {'mode': 'permissive', 'probes': {}}


def _no_sidecar(side):
    side.clear()


@pytest.mark.parametrize('edit, gate', [
    (_unfinished, 'sidecar'), (_interrupted, 'sidecar'), (_another_mode, 'sidecar'),
    (_no_probes, 'fingerprint'), (_no_sidecar, 'sidecar')])
def test_a_vacuous_or_foreign_sidecar_fails(tmp_path, mine, edit, gate) -> None:
    """S6/F3: verify-mine passes only a finished mine of the mode it is asked about."""
    rules = json.load(open(mine / FILES['permissive']))
    path = _plant(tmp_path, mine, 'permissive', rules, edit)
    if edit is _no_sidecar:
        (tmp_path / (FILES['permissive'] + '.provenance.json')).unlink()
    report = verify_mine(path, 'permissive', _OPERATORS, judge=core_judge)
    assert not report['ok'] and not report['gates'][gate]['ok'], report['gates'][gate]
    # the clean copy passes
    path = _plant(tmp_path, mine, 'permissive', rules)
    assert verify_mine(path, 'permissive', _OPERATORS, judge=core_judge)['ok']


def test_an_empty_file_fails(tmp_path, mine) -> None:
    path = _plant(tmp_path, mine, 'f64', [])
    report = verify_mine(path, 'f64', _OPERATORS, judge=core_judge)
    assert not report['ok'] and not report['gates']['rules']['ok'], report['gates']


def test_the_env_switch_gate(tmp_path, mine, monkeypatch) -> None:
    rules = json.load(open(mine / FILES['permissive']))

    def set_a_switch(side):
        side['soundness']['env_overrides'] = {'SIMPLIPY_IVL_NODE_BUDGET': '9'}

    path = _plant(tmp_path, mine, 'permissive', rules, set_a_switch)
    gate = _gate(verify_mine(path, 'permissive', _OPERATORS, judge=core_judge), 'env_switches')
    assert not gate['ok'] and gate['examples'] == [['SIMPLIPY_IVL_NODE_BUDGET', '9']]
    # ... and the check itself runs under no switch either
    path = _plant(tmp_path, mine, 'permissive', rules)
    assert _gate(verify_mine(path, 'permissive', _OPERATORS, judge=core_judge), 'env_switches')['ok']
    monkeypatch.setenv('SIMPLIPY_SOMETHING', '1')
    monkeypatch.setenv('SIMPLIPY_AC_TRACE', '1')  # a diagnostic is inert
    gate = _gate(verify_mine(path, 'permissive', _OPERATORS, judge=core_judge), 'env_switches')
    assert gate['examples'] == [['SIMPLIPY_SOMETHING', '1']]


def test_the_cli_exit_code(tmp_path, mine) -> None:
    from simplipy.__main__ import main
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'engine_generation': 2, 'operators': _OPERATORS,
                                      'rules': str(mine / 'rules.json')}))
    report = tmp_path / 'report.json'
    with pytest.raises(SystemExit) as done:
        main(['verify-mine', str(mine / FILES['real']), '--mode', 'real', '-e', str(config),
              '--judge-processes', '1', '--report', str(report)])
    assert done.value.code == 0 and json.load(open(report))['ok']
    planted = _plant(tmp_path, mine, 'real', [(('*', '1', '?0'), ('?0',))])
    with pytest.raises(SystemExit) as done:
        main(['verify-mine', planted, '--mode', 'real', '-e', str(config), '--judge-processes', '1'])
    assert done.value.code == 1


def test_a_sidecar_records_the_acceptance_and_the_code(mine) -> None:
    side = json.load(open(mine / (FILES['real'] + '.provenance.json')))
    assert side['acceptance']['rescue'] is True
    assert side['acceptance']['near_miss_frac'] == 0.5
    assert side['acceptance']['tiers'] == ['core', 'real']
    assert len(side['acceptance']['judge_digest']) == 64
    assert set(side['code_digest']) == {'core', 'python'}
    f64 = json.load(open(mine / (FILES['f64'] + '.provenance.json')))
    assert f64['acceptance']['rescue'] is False and f64['acceptance']['near_miss_frac'] is None
    assert f64['params']['interval_node_budget'] is None
    assert copy.deepcopy(f64['soundness']['env_overrides']) == {}
