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
    the clean one, edited by `sidecar_edit`. A permissive file lies beside the clean f64
    and real files its sidecar pins (the `offered` gate)."""
    if mode == 'permissive':
        for other in ('f64', 'real'):
            (tmp_path / FILES[other]).write_bytes((mine / FILES[other]).read_bytes())
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
    assert not gate['ok'] and gate['examples'] == [[list(rule[0]), list(rule[1]), 'real', []]]
    # the same file read as `permissive`, which licenses the tier
    path = _plant(tmp_path, mine, 'permissive', [rule])
    assert _gate(verify_mine(path, 'permissive', _OPERATORS, judge=judge), 'unlicensed')['ok']
    # in `real` the tier holds and a named policy refuses: the real canon cancels log(exp(x))
    # itself, so the rule is the engine's own
    path = _plant(tmp_path, mine, 'real', [rule])
    gate = _gate(verify_mine(path, 'real', _OPERATORS, judge=judge), 'unlicensed')
    assert gate['examples'] == [[list(rule[0]), list(rule[1]), 'real', ['engine_owned']]], gate


def test_the_unlicensed_gate_reads_the_named_policies(tmp_path, mine) -> None:
    """A rule that holds in every mode and a named policy refuses: `2*log(3) -> log(9)` has a
    ground composite on its right (`verify.licensed`), so a file holding it fails the gate in
    every mode."""
    rule = (('*', '2', 'log', '3'), ('log', '9'))
    for mode in ('f64', 'real', 'permissive'):
        path = _plant(tmp_path, mine, mode, [rule])
        gate = _gate(verify_mine(path, mode, _OPERATORS, judge=core_judge), 'unlicensed')
        want = [list(rule[0]), list(rule[1]), 'core']
        assert not gate['ok'] and gate['examples'][0][:3] == want, (mode, gate)
        assert 'ground_composite' in gate['examples'][0][3], (mode, gate)


def test_the_preempted_gate(tmp_path, mine) -> None:
    """A ground rule f64's constructor performs (`exp 0 -> 1`)."""
    path = _plant(tmp_path, mine, 'f64', [(('exp', '0'), ('1',))])
    gate = _gate(verify_mine(path, 'f64', _OPERATORS, judge=core_judge), 'preempted')
    assert not gate['ok'] and gate['count'] == 1, gate


def test_the_dead_gate(tmp_path, mine) -> None:
    """A rule another rule takes from its own instance, ending dearer, never decides an answer:
    here two rows of one left side, the first ending at `2*x` where the second promises `x`."""
    first, second = (('log', 'exp', '_0'), ('*', '2', '_0')), (('log', 'exp', '_0'), ('_0',))
    path = _plant(tmp_path, mine, 'f64', [first, second])
    gate = _gate(verify_mine(path, 'f64', _OPERATORS, judge=core_judge), 'dead')
    assert not gate['ok'] and [e[:2] for e in gate['examples']] == [[list(second[0]),
                                                                     list(second[1])]], gate
    assert gate['examples'][0][2:] == [['log', 'exp', 'x0'], ['*', '2', 'x0']], gate


def test_a_longer_rule_is_live_beside_its_prefix(tmp_path, mine) -> None:
    """The most specific rule fires first: with `sin acos _0 -> cos asin _0` first in the file,
    `sin acos tanh _0 -> inv cosh _0` used to be dead (`sin(acos(tanh(x)))` ended at
    `cos(asin(tanh(x)))`)."""
    rules = [(('sin', 'acos', '_0'), ('cos', 'asin', '_0')),
             (('sin', 'acos', 'tanh', '_0'), ('inv', 'cosh', '_0'))]
    path = _plant(tmp_path, mine, 'real', rules)
    ops = _acj_operators()
    assert _gate(verify_mine(path, 'real', ops, judge=core_judge), 'dead')['ok']
    engine = SimpliPyEngine(operators=ops, rules=[], rules_real=rules)
    assert engine.simplify('sin acos tanh x0'.split(), mode='real') == ['inv', 'cosh', 'x0']
    assert engine.simplify('sin acos x0'.split(), mode='real') == ['cos', 'asin', 'x0']


def _length_4(side) -> None:
    side['params']['max_source_pattern_length'] = 4


def test_the_overlaps_gate(tmp_path, mine) -> None:
    """Two rules overlap on `sin(acos(tanh(x)))`: one takes it to `1/cosh(x)`, the other to
    `cos(asin(tanh(x)))`, and no rule joins the two. The completion the judge licenses,
    `cos asin tanh x0 -> inv cosh x0`, is missing from a file of sources to length 4, and the
    gate fails until it is there; a file of sources to length 3 does not reach its source, so
    there it lies beyond the cell and fails nothing."""
    rules = [(('sin', 'acos', '_0'), ('cos', 'asin', '_0')),
             (('sin', 'acos', 'tanh', '_0'), ('inv', 'cosh', '_0'))]
    ops = _acj_operators()
    path = _plant(tmp_path, mine, 'real', rules, _length_4)
    gate = _gate(verify_mine(path, 'real', ops, judge=core_judge), 'overlaps')
    assert not gate['ok'], gate
    assert {(' '.join(e['completion'][0]), ' '.join(e['completion'][1]))
            for e in gate['examples']} == {('cos asin tanh x0', 'inv cosh x0')}, gate
    completed = rules + [(('cos', 'asin', 'tanh', '_0'), ('inv', 'cosh', '_0'))]
    path = _plant(tmp_path, mine, 'real', completed, _length_4)
    assert _gate(verify_mine(path, 'real', ops, judge=core_judge), 'overlaps')['ok']

    def refuse(pairs):
        return ['reject'] * len(pairs)

    path = _plant(tmp_path, mine, 'real', rules, _length_4)
    report = verify_mine(path, 'real', ops, judge=refuse)
    assert _gate(report, 'overlaps')['ok'] and len(report['overlaps_refused']) == 1, report
    path = _plant(tmp_path, mine, 'real', rules)
    report = verify_mine(path, 'real', ops, judge=core_judge)
    assert _gate(report, 'overlaps')['ok'] and report['overlaps_beyond_cell'] == 1, report


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
    """Verify-mine passes only a finished mine of the mode it is asked about."""
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
    assert side['acceptance']['tiers'] == ['core', 'real', 'real-only']
    assert len(side['acceptance']['judge_digest']) == 64
    assert set(side['code_digest']) == {'core', 'python'}
    f64 = json.load(open(mine / (FILES['f64'] + '.provenance.json')))
    assert f64['acceptance']['rescue'] is False and f64['acceptance']['near_miss_frac'] is None
    assert f64['params']['interval_node_budget'] is None
    assert copy.deepcopy(f64['soundness']['env_overrides']) == {}


def test_the_offered_gate(tmp_path, mine) -> None:
    """The permissive sidecar pins the f64 and real files it was offered: each must lie
    beside it with the pinned bytes, and a sidecar that pins nothing fails."""
    import shutil
    for name in FILES.values():
        shutil.copy(mine / name, tmp_path / name)
        shutil.copy(mine / (name + '.provenance.json'), tmp_path / (name + '.provenance.json'))
    path = str(tmp_path / FILES['permissive'])

    def offered():
        return _gate(verify_mine(path, 'permissive', _OPERATORS, judge=core_judge), 'offered')

    assert offered()['ok'], offered()
    assert 'offered' not in verify_mine(str(tmp_path / FILES['f64']), 'f64', _OPERATORS,
                                        judge=core_judge)['gates']
    # the real file changes after the mine
    real = json.load(open(tmp_path / FILES['real']))
    (tmp_path / FILES['real']).write_text(json.dumps(real[1:], indent=4))
    gate = offered()
    assert not gate['ok'] and gate['examples'] == [['real', 'sha256', str(tmp_path / FILES['real'])]]
    # the f64 file is gone
    (tmp_path / FILES['f64']).unlink()
    assert [e[:2] for e in offered()['examples']] == [['f64', 'missing'], ['real', 'sha256']]
    # a permissive mine that was offered nothing
    side_path = tmp_path / (FILES['permissive'] + '.provenance.json')
    side = json.load(open(side_path))
    del side['offered']
    side_path.write_text(json.dumps(side))
    gate = offered()
    assert not gate['ok'] and gate['count'] == 1, gate


def _acj_operators():
    from conftest import acj_config_path, require_or_skip
    require_or_skip(acj_config_path(), 'needs the acj vocabulary')
    return yaml.safe_load(open(acj_config_path()))['operators']


def test_a_ground_fact_the_engine_folds_is_no_rule(tmp_path, mine) -> None:
    """`preempted` is the census of ground facts: a ground rule whose left side the rule-less
    engine already takes to its right side's state. `log(-3)` folds to `nan` today."""
    fact = (('log', '(-3)'), ('float("nan")',))
    path = _plant(tmp_path, mine, 'f64', [fact])
    gate = _gate(verify_mine(path, 'f64', _acj_operators(), judge=core_judge), 'preempted')
    assert not gate['ok'] and gate['examples'] == [[list(fact[0]), list(fact[1])]], gate


def test_a_ground_nan_fact_of_a_non_integer_root_index_is_no_rule(tmp_path, mine) -> None:
    """The bulk of the published files: 106,093 of acj-5-4-llm's 138,651 f64 rules are ground
    facts with a `nan` right side, 105,085 of them `rootn` at a non-integer ground index. The
    core folds them itself, so the gate refuses every one, and a mine mints none (its mark is
    the fold)."""
    fact = (('rootn', '(-5)', 'asinh', 'tanh', '4'), ('float("nan")',))
    path = _plant(tmp_path, mine, 'f64', [fact])
    gate = _gate(verify_mine(path, 'f64', _acj_operators(), judge=core_judge), 'preempted')
    assert not gate['ok'] and gate['examples'] == [[list(fact[0]), list(fact[1])]], gate
