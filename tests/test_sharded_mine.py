"""A length mined in shards: runs that stop before it, scan its shards and merge them write the
files and sidecars of a single-machine mine (save the timings and the record of the runs), and
the merge refuses a missing, doubled or foreign shard."""
import json
import os
import shutil

import pytest
import yaml

from simplipy import SimpliPyEngine

_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "pow2": {"realization": "simplipy.operators.pow2", "alias": [], "inverse": "sqrt", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    # gives a length-4 source a chain of candidates the f64 mint accepts, so the judge loop
    # re-scans it, takes three rounds and gives it up at a cap of 2 (`log(exp(abs(x0)))`)
    "abs": {"realization": "simplipy.operators.abs", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
}
FILES = ('mined.json', 'mined_permissive.json')

#: What differs between runs of one mine: when it ran, and the record of the runs it took.
TIMINGS = ('created', 'resumed')


def comparable(sidecar: dict) -> dict:
    """A sidecar without its timings and the record of the runs."""
    out = {k: v for k, v in sidecar.items() if k not in TIMINGS}
    out['progress'] = {k: v for k, v in out['progress'].items() if k != 'updated'}
    out['per_length'] = {k: {f: x for f, x in v.items() if f != 'wall_s'}
                         for k, v in out['per_length'].items()}
    out['completion'] = [{f: x for f, x in r.items() if f != 'seconds'}
                         for r in out.get('completion') or []]
    return out


def _mine(directory, **kw):
    directory.mkdir(exist_ok=True)
    if not (directory / 'config.yaml').exists():
        (directory / 'rules.json').write_text('[]')
        (directory / 'config.yaml').write_text(yaml.safe_dump(
            {'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
    SimpliPyEngine.from_config(str(directory / 'config.yaml')).find_rules(
        max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
        extra_internal_terms=['2'], X=256, seed=7, promote_sorts=False,
        output_file=str(directory / 'mined.json'), **kw)


def _split(directory, shards: int, **kw):
    """The mine with its last length split `shards` ways, mode by mode."""
    _mine(directory, stop_after_length=3, **kw)
    for _mode in FILES:
        for i in range(shards):
            _mine(directory, shard=(i, shards), **kw)
        _mine(directory, merge=[], stop_after_length=3, **kw)


@pytest.fixture(scope='module')
def single(tmp_path_factory):
    directory = tmp_path_factory.mktemp('single')
    _mine(directory)
    return directory


@pytest.mark.parametrize('shards', [2, 3])
def test_a_split_length_mines_the_single_machine_files(single, tmp_path, shards) -> None:
    _split(tmp_path, shards)
    for name in FILES:
        assert (tmp_path / name).read_bytes() == (single / name).read_bytes(), name
        split = json.load(open(tmp_path / f'{name}.provenance.json'))
        one = json.load(open(single / f'{name}.provenance.json'))
        assert comparable(split) == comparable(one), name
        assert split['resumed'][-1]['merged'] == {'length': 4, 'shards': shards}
        assert one['per_length']['4']['picks'], 'vacuous: no pick at the split length'
        assert not (tmp_path / f'{name}.checkpoint.json').exists()


def test_a_shard_needs_the_checkpoint_of_the_length_before(tmp_path) -> None:
    with pytest.raises(RuntimeError, match='checkpoint of the length before'):
        _mine(tmp_path, shard=(0, 2))
    with pytest.raises(ValueError, match='shard'):
        _mine(tmp_path, shard=(2, 2))
    with pytest.raises(ValueError, match='neither merge'):
        _mine(tmp_path, shard=(0, 2), merge=[])


def test_the_merge_refuses_a_missing_doubled_or_foreign_shard(single, tmp_path) -> None:
    _mine(tmp_path, stop_after_length=3)
    for i in range(3):
        _mine(tmp_path, shard=(i, 3))
    base = str(tmp_path / 'mined.json')
    paths = [f'{base}.shard-4-{i}-of-3.json' for i in range(3)]
    with pytest.raises(RuntimeError, match=r'\[1\] of the 3-way split .* missing'):
        _mine(tmp_path, merge=[paths[0], paths[2]])
    doubled = f'{base}.copy.json'
    shutil.copy(paths[1], doubled)
    with pytest.raises(RuntimeError, match='shard 1 of 3 is there twice'):
        _mine(tmp_path, merge=paths + [doubled])
    os.remove(doubled)
    _mine(tmp_path, shard=(0, 2))  # a shard of another split, beside the others
    with pytest.raises(RuntimeError, match='is a shard of a [23]-way split, the others of a [23]-way'):
        _mine(tmp_path, merge=[])
    os.remove(f'{base}.shard-4-0-of-2.json')
    foreign = json.load(open(paths[2]))
    foreign['fingerprint'] = '0' * 64
    json.dump(foreign, open(f'{base}.foreign.json', 'w'))
    with pytest.raises(RuntimeError, match='not a shard of length 4 of this mine: its mine differ'):
        _mine(tmp_path, merge=paths[:2] + [f'{base}.foreign.json'])
    assert (tmp_path / 'mined.json.checkpoint.json').exists(), 'a refused merge keeps the checkpoint'
    # a shard that died is run again alone, and the merge goes through
    os.remove(paths[1])
    _mine(tmp_path, shard=(1, 3))
    _mine(tmp_path, merge=paths)
    assert (tmp_path / 'mined.json').read_bytes() == (single / 'mined.json').read_bytes()


def test_a_resumed_mine_counts_the_misses_of_the_runs_before(tmp_path, monkeypatch) -> None:
    """The sidecar's fail-closed miss counters cover the whole mine, however many runs it took:
    the checkpoint carries those of the lengths before it, a ladder snapshot's finalization
    included. A stand-in counter moves at every scan and every final gate (the snapshot's
    and the file's)."""
    import simplipy.mining as mining
    from simplipy.mining import RuleMiner
    tally = [0]
    monkeypatch.setattr(mining, '_undecided', lambda core: (tally[0], 0, 0, 0))
    real_gate, real_scan, real_length = (RuleMiner._final_gate, RuleMiner._scan_length,
                                         RuleMiner._mine_length)

    def gate(self, *args, **kwargs):
        tally[0] += 1
        return real_gate(self, *args, **kwargs)

    def scan(*args, **kwargs):
        tally[0] += 10
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(RuleMiner, '_final_gate', gate)
    monkeypatch.setattr(RuleMiner, '_scan_length', staticmethod(scan))

    def mine(directory, crash_at=None):
        calls: list = []

        def length(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
            calls.append(1)
            if len(calls) == crash_at:
                raise KeyboardInterrupt('simulated crash')
            return real_length(self, sources, mine_, rule_mode, judge_, judged, dummy_variables)

        monkeypatch.setattr(RuleMiner, '_mine_length', length)
        _mine(directory, modes=['f64'], snapshot_at={3: str(directory / 'snap.json')})

    def counted(directory):
        return [json.load(open(directory / f'{name}.provenance.json'))['soundness']
                ['interval_undecided']['horizon'] for name in ('snap.json', 'mined.json')]

    mine(tmp_path / 'once')
    with pytest.raises(KeyboardInterrupt):
        mine(tmp_path / 'resumed', crash_at=4)
    assert json.load(open(tmp_path / 'resumed' / 'mined.json.checkpoint.json'))[
        'completed_length'] == 3
    mine(tmp_path / 'resumed')
    once = counted(tmp_path / 'once')
    assert once[0] >= 1 and once[1] > once[0], 'vacuous: the snapshot moved no counter'
    assert counted(tmp_path / 'resumed') == once
    assert (tmp_path / 'resumed' / 'mined.json').read_bytes() == \
        (tmp_path / 'once' / 'mined.json').read_bytes()


def test_a_split_mine_gives_up_on_what_a_single_one_does(tmp_path, monkeypatch) -> None:
    """The judge loop gives up on a source in the merge as on one machine: the same files and
    sidecars, the incidents (`given_up`) included."""
    import simplipy.mining
    monkeypatch.setattr(simplipy.mining, '_JUDGE_ROUNDS_MAX', 1)

    def judge(pairs):        # every length-4 pick refused in f64
        return ['permissive' if len(p[0]) >= 4 else 'core' for p in pairs]

    _mine(tmp_path / 'single', judge=judge)
    _split(tmp_path / 'split', 3, judge=judge)
    for name in FILES:
        assert (tmp_path / 'split' / name).read_bytes() == \
            (tmp_path / 'single' / name).read_bytes(), name
        split = json.load(open(tmp_path / 'split' / f'{name}.provenance.json'))
        one = json.load(open(tmp_path / 'single' / f'{name}.provenance.json'))
        assert comparable(split) == comparable(one), name
    given_up = json.load(open(tmp_path / 'single' / 'mined.json.provenance.json'))['given_up']
    assert 4 in {g['length'] for g in given_up}, 'vacuous: nothing given up in the merge'


def test_a_mine_resumed_after_a_ladder_snapshot_offers_the_snapshot_to_the_next_mode(
        tmp_path, monkeypatch) -> None:
    """A mine stopped after its ladder snapshot takes the f64 cell it finished as it is on the
    resume, so the permissive snapshot is offered it, and every file equals a run through."""
    from simplipy.mining import RuleMiner
    real_length = RuleMiner._mine_length

    def mine(directory, crash_at=None):
        calls: list = []

        def length(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
            calls.append(rule_mode)
            if crash_at is not None and calls.count('default') == crash_at:
                raise KeyboardInterrupt('simulated crash')
            return real_length(self, sources, mine_, rule_mode, judge_, judged, dummy_variables)

        monkeypatch.setattr(RuleMiner, '_mine_length', length)
        _mine(directory, snapshot_at={3: str(directory / 'snap.json')})

    mine(tmp_path / 'once')
    with pytest.raises(KeyboardInterrupt):
        mine(tmp_path / 'resumed', crash_at=4)
    assert (tmp_path / 'resumed' / 'snap.json').exists()
    mine(tmp_path / 'resumed')
    for name in FILES + ('snap.json', 'snap_permissive.json'):
        assert (tmp_path / 'resumed' / name).read_bytes() == \
            (tmp_path / 'once' / name).read_bytes(), name


def _short_target_judge(pairs):
    """Refuses every length-4 pick in both modes: each refused source is re-scanned until its
    candidates run out or the loop gives it up, so the length takes several rounds."""
    return ['reject' if len(lhs) >= 4 else 'core' for lhs, rhs in pairs]


def _pending_round(directory, name):
    """The judge round the mode's checkpoint stopped after (`split_loop`), or None."""
    path = directory / f'{name}.checkpoint.json'
    loop = json.load(open(path)).get('loop') if path.exists() else None
    return None if loop is None else loop['rounds']


def _split_loop(directory, shards: int, **kw) -> list:
    """The mine with its last length's first scan and every judge round of it split `shards`
    ways, mode by mode. -> the rounds the merges stopped after, per mode."""
    _mine(directory, stop_after_length=3, **kw)
    stopped: list = []
    for name in FILES:
        for i in range(shards):
            _mine(directory, shard=(i, shards), **kw)
        _mine(directory, merge=[], stop_after_length=3, split_loop=True, **kw)
        rounds = []
        while (pending := _pending_round(directory, name)) is not None:
            rounds.append(pending)
            for i in range(shards):
                _mine(directory, shard=(i, shards), **kw)
            _mine(directory, merge=[], stop_after_length=3, split_loop=True, **kw)
        stopped.append(rounds)
    return stopped


def _looped(directory, monkeypatch, cap: int = 3):
    """A single-machine mine of `_short_target_judge`, the loop giving a source up after `cap`
    refusals: length 4 takes three judge rounds in f64 at the cap of 3, and gives a source up
    at the cap of 2."""
    import simplipy.mining
    monkeypatch.setattr(simplipy.mining, '_JUDGE_ROUNDS_MAX', cap)
    _mine(directory, judge=_short_target_judge)
    return directory


@pytest.mark.parametrize('cap', [2, 3])
def test_a_split_judge_loop_mines_the_single_machine_files(tmp_path, monkeypatch, cap) -> None:
    """Every judge round of the split length re-scanned in shards and judged in a merge: the
    files and sidecars of a single-machine mine, the sources given up on included."""
    looped = _looped(tmp_path / 'single', monkeypatch, cap)
    stopped = _split_loop(tmp_path / 'split', 3, judge=_short_target_judge)
    for name, rounds in zip(FILES, stopped):
        assert (tmp_path / 'split' / name).read_bytes() == (looped / name).read_bytes(), name
        split = json.load(open(tmp_path / 'split' / f'{name}.provenance.json'))
        one = json.load(open(looped / f'{name}.provenance.json'))
        assert comparable(split) == comparable(one), name
        assert rounds == list(range(1, one['per_length']['4']['judge_rounds'])), (name, rounds)
        if rounds:
            merged = [r['merged'] for r in split['resumed'] if 'merged' in r]
            assert {'length': 4, 'shards': 3, 'round': 2} in merged, merged
    if cap == 3:
        assert any(len(rounds) >= 2 for rounds in stopped), 'vacuous: no length took 3 rounds'
    else:
        given_up = json.load(open(looped / 'mined.json.provenance.json')).get('given_up')
        assert given_up and given_up[0]['length'] == 4, 'vacuous: no source given up on'


def test_a_split_judge_loop_finishes_on_one_machine_after_any_round(tmp_path,
                                                                    monkeypatch) -> None:
    """A run without shard or merge re-scans the pending round itself: stopped after its first
    round, the f64 length finishes on one machine, to the single-machine files."""
    looped = _looped(tmp_path / 'single', monkeypatch)
    split = tmp_path / 'split'
    _mine(split, stop_after_length=3, judge=_short_target_judge)
    for i in range(3):
        _mine(split, shard=(i, 3), judge=_short_target_judge)
    _mine(split, merge=[], stop_after_length=3, split_loop=True, judge=_short_target_judge)
    assert _pending_round(split, 'mined.json') == 1, 'vacuous: the merge left no round'
    _mine(split, judge=_short_target_judge)
    for name in FILES:
        assert (split / name).read_bytes() == (looped / name).read_bytes(), name
        assert comparable(json.load(open(split / f'{name}.provenance.json'))) == \
            comparable(json.load(open(looped / f'{name}.provenance.json'))), name


def test_a_round_merge_refuses_a_missing_doubled_or_foreign_shard(tmp_path, monkeypatch) -> None:
    """The merge of a judge round refuses as the merge of a first scan does, and refuses a shard
    of another round or of another request; the round is merged once its shards are right."""
    import simplipy.mining
    monkeypatch.setattr(simplipy.mining, '_JUDGE_ROUNDS_MAX', 3)
    kw = {'judge': _short_target_judge}
    _mine(tmp_path, stop_after_length=3, **kw)
    for i in range(3):
        _mine(tmp_path, shard=(i, 3), **kw)
    _mine(tmp_path, merge=[], stop_after_length=3, split_loop=True, **kw)
    assert _pending_round(tmp_path, 'mined.json') == 1
    base = str(tmp_path / 'mined.json')
    paths = [f'{base}.shard-4-round-2-{i}-of-3.json' for i in range(3)]
    _mine(tmp_path, shard=(0, 3), **kw)
    _mine(tmp_path, shard=(2, 3), **kw)
    with pytest.raises(RuntimeError, match=r'\[1\] of the 3-way split of length 4, judge round 2'):
        _mine(tmp_path, merge=[], split_loop=True, **kw)
    _mine(tmp_path, shard=(1, 3), **kw)
    doubled = f'{base}.copy.json'
    shutil.copy(paths[1], doubled)
    with pytest.raises(RuntimeError, match='shard 1 of 3 is there twice'):
        _mine(tmp_path, merge=paths + [doubled], split_loop=True, **kw)
    for field, value in (('round', 3), ('request', '0' * 64)):
        foreign = json.load(open(paths[2]))
        foreign[field] = value
        json.dump(foreign, open(doubled, 'w'))
        with pytest.raises(RuntimeError, match=f'not a shard of length 4, judge round 2 of '
                                               f'this mine: its {field} differ'):
            _mine(tmp_path, merge=paths[:2] + [doubled], split_loop=True, **kw)
    with pytest.raises(RuntimeError, match='its round, request differ'):
        _mine(tmp_path, merge=paths[:2] + [f'{base}.shard-4-2-of-3.json'], split_loop=True, **kw)
    assert _pending_round(tmp_path, 'mined.json') == 1, 'a refused merge keeps the round'
    _mine(tmp_path, merge=paths, split_loop=True, **kw)
    assert _pending_round(tmp_path, 'mined.json') != 1
