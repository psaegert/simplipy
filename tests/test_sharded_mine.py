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


def _split(directory, shards: int):
    """The mine with its last length split `shards` ways, mode by mode."""
    _mine(directory, stop_after_length=3)
    for _mode in FILES:
        for i in range(shards):
            _mine(directory, shard=(i, shards))
        _mine(directory, merge=[], stop_after_length=3)


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
    with pytest.raises(RuntimeError, match='not a shard of this mine: its mine differ'):
        _mine(tmp_path, merge=paths[:2] + [f'{base}.foreign.json'])
    assert (tmp_path / 'mined.json.checkpoint.json').exists(), 'a refused merge keeps the checkpoint'
    # a shard that died is run again alone, and the merge goes through
    os.remove(paths[1])
    _mine(tmp_path, shard=(1, 3))
    _mine(tmp_path, merge=paths)
    assert (tmp_path / 'mined.json').read_bytes() == (single / 'mined.json').read_bytes()
