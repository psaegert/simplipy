"""A mine's per-length operator set (`operator_max_length`): an operator bounded to n tokens
appears in no source and no candidate longer than n. One definition builds the universe at every
site that builds one, the mine's cell is that universe, and the setting is part of what a mine
records and resumes by."""
import json

import pytest
import yaml

import simplipy.mining as mining
from simplipy import SimpliPyEngine
from simplipy.__main__ import main
from simplipy.mining import RuleMiner, _in_cell, _operator_max_length, _universe
from simplipy.utils import count_expressions, enumerate_expressions
from simplipy.verify import policy_refusals

_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "pow2": {"realization": "simplipy.operators.pow2", "alias": [], "inverse": "sqrt", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
}
_ARITY = {name: spec['arity'] for name, spec in _OPERATORS.items()}
_LEAVES = ['x0', '0', '1', '<constant>']


def _engine(directory):
    directory.mkdir(exist_ok=True)
    (directory / 'rules.json').write_text('[]')
    (directory / 'config.yaml').write_text(yaml.safe_dump(
        {'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
    return SimpliPyEngine.from_config(str(directory / 'config.yaml'))


def _mine(directory, length=3, extra=('0', '1', '<constant>'), **kw):
    """A small mine in `directory`; its output is `directory/mined.json`."""
    _engine(directory).find_rules(
        max_source_pattern_length=length, max_target_pattern_length=length, dummy_variables=1,
        extra_internal_terms=list(extra), X=256, seed=7, promote_sorts=False,
        output_file=str(directory / 'mined.json'), **kw)
    return directory / 'mined.json'


def test_without_a_bound_the_universe_is_the_enumeration() -> None:
    full = enumerate_expressions(_LEAVES, _ARITY, 4)
    counts = count_expressions(len(_LEAVES), _ARITY, 4)
    for setting in (None, {}, {'exp': 4, 'log': 9}):
        assert _universe(_LEAVES, _ARITY, 4, setting) == (full, counts), setting


@pytest.mark.parametrize('bounds', [{'exp': 2}, {'exp': 3}, {'exp': 2, 'log': 4}, {'+': 3, 'neg': 1}])
def test_a_bound_removes_exactly_the_longer_expressions_that_hold_it(bounds) -> None:
    """At each length the universe is the complete enumeration less exactly the expressions that
    hold an operator bounded below that length, as many as the counting recurrence over all the
    operators exceeds the recurrence over those left."""
    full = enumerate_expressions(_LEAVES, _ARITY, 5)
    expressions, counts = _universe(_LEAVES, _ARITY, 5, bounds)
    removed_somewhere = False
    for length in range(1, 6):
        excluded = {op for op, bound in bounds.items() if bound < length}
        kept = {e for e in full[length] if not excluded & set(e)}
        assert expressions[length] == kept, length
        assert counts[length] == len(kept), length
        allowed = {op: arity for op, arity in _ARITY.items() if op not in excluded}
        assert len(full[length]) - len(kept) == (
            count_expressions(len(_LEAVES), _ARITY, length)[length]
            - count_expressions(len(_LEAVES), allowed, length)[length]), length
        removed_somewhere |= len(kept) < len(full[length])
    assert removed_somewhere, 'vacuous: the bound removes nothing'


def test_the_setting_is_validated_and_normalized() -> None:
    assert _operator_max_length(None, _ARITY) == {}
    normalized = _operator_max_length({'log': 3, 'exp': 2}, _ARITY)
    assert normalized == {'exp': 2, 'log': 3} and list(normalized) == ['exp', 'log']
    for bad in ({'sin': 2}, {'exp': 0}, {'exp': True}, {'exp': 2.0}, {'exp': '2'}, ['exp']):
        with pytest.raises(ValueError, match='operator_max_length'):
            _operator_max_length(bad, _ARITY)


def test_a_mine_reads_the_setting_wherever_it_builds_the_universe(tmp_path, monkeypatch) -> None:
    """find_rules and certify_rules build their universe through `_universe` with the setting;
    the length loop's sources and the candidate library come from that universe, so no source or
    candidate longer than 2 tokens holds `exp`; both sidecars record the setting, and the cell
    the overlap gate of verify-mine reads is the sidecar's."""
    universes: list = []
    real_universe = mining._universe

    def universe_spy(*args):
        universes.append(args[3] if len(args) > 3 else None)
        return real_universe(*args)

    seen: list = []
    real_native = RuleMiner._find_rules_native

    def native_spy(self, expressions_of_length, *args, **kwargs):
        seen.append({length: set(found) for length, found in expressions_of_length.items()})
        return real_native(self, expressions_of_length, *args, **kwargs)

    monkeypatch.setattr(mining, '_universe', universe_spy)
    monkeypatch.setattr(RuleMiner, '_find_rules_native', native_spy)
    out = _mine(tmp_path / 'm', operator_max_length={'exp': 2})
    assert universes == [{'exp': 2}]
    expected, counts = real_universe(_LEAVES, _ARITY, 3, {'exp': 2})
    assert len(seen) == 2 and all(found == expected for found in seen)  # f64, then permissive
    assert any('exp' in e for e in expected[2])
    assert not any('exp' in e for e in expected[3])
    for name in ('mined.json', 'mined_permissive.json'):
        sidecar = json.load(open(tmp_path / 'm' / f'{name}.provenance.json'))
        assert sidecar['params']['operator_max_length'] == {'exp': 2}
        assert {int(k): v['complete_count'] for k, v in sidecar['universe'].items()} == counts

    engine = _engine(tmp_path / 'c')
    engine.certify_rules([['exp', 'log', 'x0']], mode='f64', max_target_pattern_length=3, X=256,
                         operator_max_length={'exp': 2})
    assert universes[-1] == {'exp': 2}

    cells: list = []
    monkeypatch.setattr(mining, '_overlap_gate',
                        lambda *args, **kwargs: cells.append(kwargs) or ([], [], [], []))
    mining.verify_mine(str(out), 'f64', _OPERATORS,
                       judge=lambda pairs: ['core'] * len(pairs))
    assert cells and cells[0]['max_source'] == 3
    assert cells[0]['operator_max_length'] == {'exp': 2}


def test_the_cell_is_the_universe(tmp_path) -> None:
    """The cell test of the completion step and the overlap gate (`_in_cell`, the policy
    `cell_bound`) holds what the universe holds: a source no longer than the mine's, with no
    operator bounded below its length."""
    longer = {'completion': [['exp', 'exp', 'x0'], ['x0']]}
    shorter = {'completion': [['exp', 'x0'], ['x0']]}
    assert _in_cell(longer, 4) and _in_cell(shorter, 4)
    assert not _in_cell(longer, 4, {'exp': 2}) and _in_cell(shorter, 4, {'exp': 2})
    assert not _in_cell(longer, 2)
    engine = _engine(tmp_path)
    assert 'cell_bound' in policy_refusals(['exp', 'exp', '?0'], ['?0'], 'f64', engine,
                                           max_source=4, operator_max_length={'exp': 2})
    assert 'cell_bound' not in policy_refusals(['exp', 'exp', '?0'], ['?0'], 'f64', engine,
                                               max_source=4)


def test_an_empty_setting_mines_the_same_files(tmp_path) -> None:
    """No setting, an empty one and one whose every bound is past the mine's longest source give
    the same rule files, byte for byte."""
    kw: dict = {'length': 4, 'extra': ['2']}
    runs = [_mine(tmp_path / 'none', **kw), _mine(tmp_path / 'empty', operator_max_length={}, **kw),
            _mine(tmp_path / 'past', operator_max_length={'exp': 4, 'log': 9}, **kw)]
    for name in ('mined.json', 'mined_permissive.json'):
        files = [(run.parent / name).read_bytes() for run in runs]
        assert files[0] == files[1] == files[2], name
        assert json.loads(files[0]), 'vacuous: an empty file'


def test_a_checkpoint_of_another_setting_is_not_resumed(tmp_path) -> None:
    """A mine that dies inside a length leaves its checkpoint; a mine under another per-length
    operator set refuses it and leaves it in place, and the same mine resumes it."""
    def crashing_mine(crash_at=None, **kw):
        calls: list = []
        original = RuleMiner._mine_length

        def spy(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
            if rule_mode == 'default':
                calls.append(len(calls))
                if crash_at is not None and len(calls) == crash_at:
                    raise KeyboardInterrupt('simulated crash')
            return original(self, sources, mine_, rule_mode, judge_, judged, dummy_variables)

        RuleMiner._mine_length = spy  # type: ignore[method-assign]
        try:
            _mine(tmp_path / 'a', **kw)
        finally:
            RuleMiner._mine_length = original  # type: ignore[method-assign]
        return calls

    with pytest.raises(KeyboardInterrupt):
        crashing_mine(crash_at=3)
    checkpoint = tmp_path / 'a' / 'mined.json.checkpoint.json'
    assert json.load(open(checkpoint))['completed_length'] == 2
    for other in ({'exp': 2}, {'exp': 3}):
        with pytest.raises(RuntimeError, match='different mine'):
            crashing_mine(operator_max_length=other)
        assert checkpoint.exists(), 'a refused checkpoint is never overwritten'
    assert len(crashing_mine()) == 1, 'the same mine resumes'


def test_the_cli_forwards_the_config_key(tmp_path) -> None:
    _engine(tmp_path)
    mine_cfg = tmp_path / 'find_rules.yaml'
    mine_cfg.write_text(yaml.safe_dump({
        'max_source_pattern_length': 3, 'max_target_pattern_length': 3, 'dummy_variables': 1,
        'extra_internal_terms': ['0', '1', '<constant>'], 'n_samples': 256,
        'constants_fit_retries': 16, 'seed': 7, 'promote_sorts': False,
        'operator_max_length': {'log': 2, 'exp': 2}}))
    out = tmp_path / 'mined.json'
    main(['find-rules', '-e', str(tmp_path / 'config.yaml'), '-c', str(mine_cfg), '-o', str(out)])
    params = json.load(open(str(out) + '.provenance.json'))['params']
    assert params['operator_max_length'] == {'exp': 2, 'log': 2}
