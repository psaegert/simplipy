"""The mine's resume key (`checkpoint_fp` in `RuleMiner._find_rules_native`) against the arguments
that change a finished file: a rerun reuses a finished file (`_finished_cell`) and resumes a
checkpoint only under the key, which holds the sidecar's `params`, the mode, the code, the operator
table, the data, the numeric environment, the universe, the starting and other sets and the
proposal batch's sha256. Pre-audit review of 0.15 (au-mine, M1 and M2): `promote_sorts`,
`symbolic_gate` and the proposal batch change the file, or its sidecar's pin of the batch, and
were not in the key, so a rerun that changed only them took the other mine's work as its own; the
three tests below were strict xfails until the 0.15 fix round put them in (`promote_sorts` and
`symbolic_gate` through the sidecar's params)."""
import hashlib
import json
import shutil

import pytest

from simplipy import SimpliPyEngine

_OPERATORS = {
    "+": {"realization": "simplipy.operators.add", "alias": [], "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "simplipy.operators.mul", "alias": [], "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "arity": 1, "precedence": 2.5, "commutative": False},
    "abs": {"realization": "simplipy.operators.abs", "alias": [], "arity": 1, "precedence": 3, "commutative": False},
    "asin": {"realization": "simplipy.operators.asin", "alias": [], "arity": 1, "precedence": 2, "commutative": False},
    "tanh": {"realization": "simplipy.operators.tanh", "alias": [], "arity": 1, "precedence": 2, "commutative": False},
    "atanh": {"realization": "simplipy.operators.atanh", "alias": [], "arity": 1, "precedence": 2, "commutative": False},
}

#: Two proposal batches, which differ in content and in count.
_PROPOSALS = (
    [{"source": ["abs", "neg", "asin", "x0"], "target": ["abs", "asin", "x0"]}],
    [{"source": ["abs", "neg", "asin", "x0"], "target": ["abs", "asin", "x0"]},
     {"source": ["asin", "abs", "neg", "x0"], "target": ["abs", "asin", "x0"]}],
)


def _mine(directory, **kw) -> list | None:
    """A strict mine of the length-3 cell into `directory`; the rules of the file it leaves (None
    for a run that stops before the file, leaving its checkpoint)."""
    directory.mkdir(exist_ok=True)
    SimpliPyEngine(operators=_OPERATORS, rules=[]).find_rules(
        max_source_pattern_length=3, max_target_pattern_length=3, dummy_variables=1,
        extra_internal_terms=['0', '1'], X=256, seed=7, constants_fit_challenges=4,
        constants_fit_retries=4, rtol=1e-11, atol=1e-12, prune='covered', modes=['strict'],
        judge_processes=1, output_file=str(directory / 'mined.json'), **kw)
    if not (directory / 'mined.json').exists():
        return None
    return [(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(open(directory / 'mined.json'))]


def _batch(path, entries) -> str:
    """Write a proposal batch; its sha256, as a sidecar pins it."""
    path.write_text(json.dumps(entries))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned(directory) -> dict:
    return json.load(open(directory / 'mined.json.provenance.json')).get('proposals') or {}


@pytest.fixture(scope='module')
def cells(tmp_path_factory):
    """The cell mined without promotion, and with it."""
    root = tmp_path_factory.mktemp('resume_key')
    return root, _mine(root / 'raw', promote_sorts=False), _mine(root / 'promoted')


def test_the_premise_promotion_changes_the_cell(cells) -> None:
    _, raw, promoted = cells
    assert raw and raw != promoted


def test_the_premise_a_mine_pins_its_proposal_batch(tmp_path) -> None:
    sha = _batch(tmp_path / 'proposals.json', _PROPOSALS[0])
    _mine(tmp_path / 'a', promote_sorts=False, proposals=str(tmp_path / 'proposals.json'))
    assert _pinned(tmp_path / 'a').get('sha256') == sha


def test_a_finished_file_is_reused_only_by_a_mine_that_promotes_as_it_did(cells, tmp_path) -> None:
    root, _, promoted = cells
    shutil.copytree(root / 'raw', tmp_path / 'a')
    assert _mine(tmp_path / 'a') == promoted
    side = json.load(open(tmp_path / 'a' / 'mined.json.provenance.json'))
    assert (side['params']['promote_sorts'], side['params']['symbolic_gate']) == (True, True)


def test_a_finished_file_is_reused_only_by_a_mine_of_its_proposal_batch(cells, tmp_path) -> None:
    root, _, _ = cells
    shutil.copytree(root / 'raw', tmp_path / 'a')
    sha = _batch(tmp_path / 'proposals.json', _PROPOSALS[0])
    _mine(tmp_path / 'a', promote_sorts=False, proposals=str(tmp_path / 'proposals.json'))
    assert _pinned(tmp_path / 'a').get('sha256') == sha


def test_a_resumed_mine_pins_the_proposal_batch_it_certifies(tmp_path) -> None:
    """A checkpoint written under one batch is another mine's under another batch: it is refused,
    not resumed (it used to resume, certify the new batch and pin the old one, M2), and the mine
    started anew pins the batch it certifies."""
    _batch(tmp_path / 'first.json', _PROPOSALS[0])
    second = _batch(tmp_path / 'second.json', _PROPOSALS[1])
    _mine(tmp_path / 'a', promote_sorts=False, proposals=str(tmp_path / 'first.json'),
          stop_after_length=2)
    with pytest.raises(RuntimeError, match='was written by a different mine'):
        _mine(tmp_path / 'a', promote_sorts=False, proposals=str(tmp_path / 'second.json'))
    (tmp_path / 'a' / 'mined.json.checkpoint.json').unlink()
    _mine(tmp_path / 'a', promote_sorts=False, proposals=str(tmp_path / 'second.json'))
    pin = _pinned(tmp_path / 'a')
    assert (pin.get('sha256'), pin.get('count')) == (second, len(_PROPOSALS[1]))
