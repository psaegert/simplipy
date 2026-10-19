"""The mine's per-length judge loop (miner.md 2.6, 2.7): the judge's tier is decided inside
the scan, a refused rule never prunes a longer source, and the scan is a function of the
global source index alone."""
import json

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner

_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "pow2": {"realization": "simplipy.operators.pow2", "alias": [], "inverse": "sqrt", "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
}

SOURCE = ['*', 'exp', 'x0', 'exp', 'x0']

# A source with two accepted candidates of DIFFERENT classes (equal on the reals, both
# realised by f64 on the mining sample), in scan order.
WITNESS = ['sinh', 'asinh', 'x0']
FIRST = ['x0']
SECOND = ['tan', 'atan', 'x0']


@pytest.fixture(scope='module')
def eng() -> SimpliPyEngine:
    return SimpliPyEngine(operators=_OPERATORS, rules=[])


@pytest.fixture(scope='module')
def acj() -> SimpliPyEngine:
    from conftest import acj_config_path, require_or_skip
    require_or_skip(acj_config_path(), 'needs the acj vocabulary')
    return SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))['operators'], rules=[])


def _scan(eng, mode='default'):
    """`mine(batch, indices, exclusions)` over a library of hand-picked candidates."""
    x = RuleMiner(eng)._mining_sample_x(512, 1, np.random.default_rng(7))
    lib = eng._core.build_candidate_library(
        [FIRST, ['*', '1', 'x0'], ['sin', 'x0'], SECOND], ['x0'],
        x.flatten(order='C').tolist(), x.shape[0], rule_mode=mode)

    def mine(batch, indices, exclusions):
        return eng._core.mine_one_length(batch, lib, None, 16, 16, 7, 1e-9, 1e-12, None, True,
                                         rule_mode=mode, indices=indices, exclusions=exclusions)
    return mine


def test_the_premise_two_classes_in_scan_order_and_exclusion_by_class(acj) -> None:
    mine = _scan(acj)
    assert mine([WITNESS], None, None) == [(0, WITNESS, FIRST)]
    assert mine([WITNESS], None, {0: [FIRST]}) == [(0, WITNESS, SECOND)]
    # an exclusion names a CLASS: another spelling of the refused target excludes it too
    assert mine([WITNESS], None, {0: [['*', '1', 'x0']]}) == [(0, WITNESS, SECOND)]
    assert mine([WITNESS], None, {0: [FIRST, SECOND]}) == []


def test_a_refused_pick_yields_to_the_next_licensed_candidate(acj) -> None:
    """NO SHADOWING: the judge files the cheapest pick under another mode, and the loop
    must return the next candidate the mode licenses -- not drop the source."""
    refused = (('sinh', 'asinh', '?0'), ('?0',))

    def judge_as(tier):
        return lambda pairs: [tier if tuple(map(tuple, p)) == refused else 'core' for p in pairs]

    found, stats = RuleMiner(acj)._mine_length([WITNESS], _scan(acj), 'default',
                                               judge_as('real'), {}, ['x0'])
    assert found == [(WITNESS, SECOND)]
    assert stats['judge_rounds'] == 2
    assert stats['excluded'] == [[WITNESS, FIRST]]
    # a tier f64 licenses: no exclusion, the first pick stands
    found, stats = RuleMiner(acj)._mine_length([WITNESS], _scan(acj), 'default',
                                               judge_as('f64'), {}, ['x0'])
    assert found == [(WITNESS, FIRST)] and stats['judge_rounds'] == 1


def test_the_judge_memo_is_consulted_before_the_judge(acj) -> None:
    calls = []

    def judge(pairs):
        calls.append(list(pairs))
        return ['core'] * len(pairs)

    judged: dict = {}
    miner = RuleMiner(acj)
    miner._mine_length([WITNESS], _scan(acj), 'default', judge, judged, ['x0'])
    miner._mine_length([WITNESS], _scan(acj), 'default', judge, judged, ['x0'])
    assert len(calls) == 1 and judged


def test_two_shards_equal_one_shot(acj) -> None:
    """The scan of a source is a function of its GLOBAL index (its seed, its exclusions),
    never of the batch it arrives in: two shards with `index_offset` -- or explicit
    `indices` -- reproduce the one-shot mine, exclusions included."""
    mine = _scan(acj)
    sources = [WITNESS, WITNESS, WITNESS]
    excl = {2: [FIRST]}
    one_shot = mine(sources, None, excl)
    assert [t for _, _, t in one_shot] == [FIRST, FIRST, SECOND]
    shards = mine(sources[:2], None, excl) + mine(sources[2:], [2], excl)
    assert shards == one_shot
    assert _offset_scan(acj, sources[2:], 2, excl) == one_shot[2:]


def _offset_scan(eng, batch, offset, exclusions):
    """The shard by `index_offset` rather than explicit `indices`."""
    x = RuleMiner(eng)._mining_sample_x(512, 1, np.random.default_rng(7))
    lib = eng._core.build_candidate_library(
        [FIRST, ['*', '1', 'x0'], ['sin', 'x0'], SECOND], ['x0'],
        x.flatten(order='C').tolist(), x.shape[0], rule_mode='default')
    return eng._core.mine_one_length(batch, lib, None, 16, 16, 7, 1e-9, 1e-12, None, True,
                                     rule_mode='default', index_offset=offset,
                                     exclusions=exclusions)


PROBE = ['pow2', 'exp', 'x0']


def _gated_mine(tmp_path, judge, marks):
    """A tiny mine whose length loop records, at the start of each f64 length (lengths
    1..4, in order), the f64 serve answer of `pow2 exp x0` -- the mark a longer source
    containing it mines against."""
    original = RuleMiner._mine_length

    def spy(self, sources, mine, rule_mode, judge_, judged, dummy_variables):
        if rule_mode == 'default':
            marks.append(self.engine._core.mine_mark(PROBE, rule_mode=rule_mode))
        return original(self, sources, mine, rule_mode, judge_, judged, dummy_variables)

    tmp_path.mkdir()
    (tmp_path / 'rules.json').write_text('[]')
    (tmp_path / 'config.yaml').write_text(
        yaml.safe_dump({'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
    RuleMiner._mine_length = spy  # type: ignore[method-assign]
    try:
        engine.find_rules(max_source_pattern_length=4, max_target_pattern_length=4,
                          dummy_variables=1, extra_internal_terms=['2'], X=256, seed=7,
                          promote_sorts=False, verbose=False, judge=judge,
                          output_file=str(tmp_path / 'mined.json'))
    finally:
        RuleMiner._mine_length = original  # type: ignore[method-assign]
    return json.load(open(tmp_path / 'mined.json'))


def test_a_rule_refused_at_one_length_never_moves_a_longer_mark(tmp_path) -> None:
    """GATED BEFORE INSTALLED (miner.md 2.7): what the judge refuses at a length is never
    installed, so no longer source's mark moves for it. Anti-vacuity: with the rule
    licensed, the same mark DOES move."""
    bare = SimpliPyEngine(operators=_OPERATORS, rules=[])._core.mine_mark(
        PROBE, rule_mode='default')
    lhs = {('pow2', 'exp', '?0'), tuple(bare[:1]) + ('exp', '?0') + tuple(bare[3:])}

    def refuse(pairs):
        return ['reject' if tuple(p[0]) in lhs else 'core' for p in pairs]

    licensed: list = []
    _gated_mine(tmp_path / 'licensed', lambda pairs: ['core'] * len(pairs), licensed)
    assert len(licensed) == 4 and licensed[:3] == [bare] * 3, licensed
    assert licensed[3] != bare, f'vacuous: no length-3 rule moves the mark ({licensed})'
    refused: list = []
    mined = _gated_mine(tmp_path / 'refused', refuse, refused)
    assert not [r for r in mined if tuple(r[0]) in lhs], mined
    assert refused == [bare] * 4, refused


def test_a_crashed_mine_resumes_from_its_last_length_to_the_same_files(tmp_path) -> None:
    """RESUME FROM DISK FIRST: a mine that dies inside a length leaves the checkpoint of the
    last completed one, and the rerun continues from there to byte-identical files."""
    def mine(directory, crash_at=None):
        calls: list = []
        original = RuleMiner._mine_length

        def spy(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
            if rule_mode == 'default':
                calls.append(len(calls))
                if crash_at is not None and len(calls) == crash_at:
                    raise KeyboardInterrupt('simulated crash')
            return original(self, sources, mine_, rule_mode, judge_, judged, dummy_variables)

        directory.mkdir(exist_ok=True)
        (directory / 'rules.json').write_text('[]')
        (directory / 'config.yaml').write_text(yaml.safe_dump(
            {'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
        engine = SimpliPyEngine.from_config(str(directory / 'config.yaml'))
        RuleMiner._mine_length = spy  # type: ignore[method-assign]
        try:
            engine.find_rules(max_source_pattern_length=4, max_target_pattern_length=4,
                              dummy_variables=1, extra_internal_terms=['2'], X=256, seed=7,
                              promote_sorts=False, verbose=False,
                              output_file=str(directory / 'mined.json'))
        finally:
            RuleMiner._mine_length = original  # type: ignore[method-assign]
        return calls

    assert len(mine(tmp_path / 'once')) == 4
    with pytest.raises(KeyboardInterrupt):
        mine(tmp_path / 'resumed', crash_at=4)
    checkpoint = json.load(open(tmp_path / 'resumed' / 'mined.json.checkpoint.json'))
    assert checkpoint['completed_length'] == 3
    assert len(mine(tmp_path / 'resumed')) == 1, 'the rerun must not re-mine lengths 1..3'
    assert not (tmp_path / 'resumed' / 'mined.json.checkpoint.json').exists()
    for name in ('mined.json', 'mined_real.json', 'mined_permissive.json'):
        assert (tmp_path / 'once' / name).read_bytes() == \
            (tmp_path / 'resumed' / name).read_bytes(), name


def test_the_dedup_tie_break_is_the_modes_ordering_not_token_count(eng) -> None:
    """Two targets for one canonical source: `pow2 exp ?0` is one token shorter, but it
    reads as the source itself (`pow exp ?0 2`), while `exp * 2 ?0` is strictly below it
    in the serve ordering -- the order the rewrite fires under. The ordering decides, in
    either input order."""
    from simplipy.utils import deduplicate_rules
    below = (tuple(SOURCE), ('exp', '*', '2', 'x0'))
    shorter = (tuple(SOURCE), ('pow2', 'exp', 'x0'))
    assert eng._core.ac_ordered_below_in(list(below[1]), list(shorter[1]), rule_mode='default')
    want = [(('*', 'exp', '?0', 'exp', '?0'), ('exp', '*', '2', '?0'))]
    for rules in ([below, shorter], [shorter, below]):
        assert deduplicate_rules(rules, ['x0'], engine=eng, rule_mode='default') == want, rules


def test_a_twin_shadowed_row_is_dropped_and_nothing_it_served_is_lost() -> None:
    """TWIN-SHADOW REMOVAL (miner.md 2.7): two rows that are each other's orientation twin
    serve one rewrite pair twice over. The later row goes, the earlier row's twin mints in
    its place, and the served set is exactly what it was."""
    from conftest import acj_config_path, require_or_skip
    require_or_skip(acj_config_path(), 'needs the acj vocabulary')
    ops = yaml.safe_load(open(acj_config_path()))['operators']
    sib = [(('-', 'acosh', 'atanh', '?0', '?1'), ('-', '?0', '?1')),
           (('-', '?1', 'acosh', 'atanh', '?0'), ('-', '?1', '?0'))]
    engine = SimpliPyEngine(operators=ops, rules=[[list(lhs), list(rhs)] for lhs, rhs in sib])

    def served():
        return sorted((tuple(lhs), tuple(rhs))
                      for lhs, rhs, _ in engine._core.ac_served_rules_in_mode(rule_mode='default'))

    before = served()
    assert engine._core.ac_shadow_census(rule_mode='default'), 'vacuous: no twin-shadow'
    kept, dropped = RuleMiner(engine)._deshadow('default', sib)
    assert kept == [sib[0]] and dropped == [sib[1]]
    assert engine._core.ac_shadow_census(rule_mode='default') == []
    assert served() == before


def _crashing_mine(directory, crash_at=None, **kw):
    """A small mine in `directory`; with `crash_at`, it dies inside that f64 length."""
    calls: list = []
    original = RuleMiner._mine_length

    def spy(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
        if rule_mode == 'default':
            calls.append(len(calls))
            if crash_at is not None and len(calls) == crash_at:
                raise KeyboardInterrupt('simulated crash')
        return original(self, sources, mine_, rule_mode, judge_, judged, dummy_variables)

    directory.mkdir(exist_ok=True)
    (directory / 'rules.json').write_text('[]')
    (directory / 'config.yaml').write_text(yaml.safe_dump(
        {'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(directory / 'config.yaml'))
    RuleMiner._mine_length = spy  # type: ignore[method-assign]
    try:
        engine.find_rules(max_source_pattern_length=4, max_target_pattern_length=4,
                          dummy_variables=1, extra_internal_terms=['2'], X=256, seed=7,
                          promote_sorts=False, verbose=False,
                          output_file=str(directory / 'mined.json'), **kw)
    finally:
        RuleMiner._mine_length = original  # type: ignore[method-assign]
    return calls


def test_a_resume_across_code_changes_refuses(tmp_path, monkeypatch) -> None:
    """A checkpoint resumes only the mine that wrote it: other code (the compiled core or
    the package's sources) or other parameters refuse, loudly, and leave it in place."""
    import simplipy.mining as mining
    with pytest.raises(KeyboardInterrupt):
        _crashing_mine(tmp_path / 'a', crash_at=4)
    checkpoint = tmp_path / 'a' / 'mined.json.checkpoint.json'
    assert checkpoint.exists()
    real_digest = mining.code_digest()
    monkeypatch.setattr(mining, 'code_digest', lambda: dict(real_digest, core='0' * 64))
    with pytest.raises(RuntimeError, match='different mine'):
        _crashing_mine(tmp_path / 'a')
    assert checkpoint.exists(), 'a refused checkpoint is never overwritten'
    monkeypatch.setattr(mining, 'code_digest', lambda: real_digest)
    with pytest.raises(RuntimeError, match='different mine'):
        _crashing_mine(tmp_path / 'a', interval_node_budget=12345)
    assert len(_crashing_mine(tmp_path / 'a')) == 1, 'the same mine resumes'


def test_a_mine_sets_the_node_budget_for_itself(tmp_path) -> None:
    """`interval_node_budget` is the mine's, recorded in its parameters, and restored after."""
    from simplipy import _core
    seen: list = []
    original = RuleMiner._mine_length

    def spy(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
        previous = _core.set_interval_node_budget(None)
        _core.set_interval_node_budget(previous)
        seen.append(previous)
        return original(self, sources, mine_, rule_mode, judge_, judged, dummy_variables)

    RuleMiner._mine_length = spy  # type: ignore[method-assign]
    try:
        _crashing_mine(tmp_path / 'b', interval_node_budget=12345)
    finally:
        RuleMiner._mine_length = original  # type: ignore[method-assign]
    assert set(seen) == {12345}
    assert _core.set_interval_node_budget(None) == 400_000
    side = json.load(open(tmp_path / 'b' / 'mined.json.provenance.json'))
    assert side['params']['interval_node_budget'] == 12345


def test_a_proposal_whose_target_the_mode_refuses_reads_rejected_at_the_judge(tmp_path) -> None:
    """THE PROPOSALS ARE ONE MORE LENGTH OF THE JUDGE LOOP (miner.md 2.11): the certified
    target of `* exp x0 exp x0` goes to the mine's judge, which files its class `real`;
    the f64 mine excludes it, finds no other class, and the trail says why."""
    proposal = {'source': SOURCE, 'target': ['exp', '*', '2', 'x0']}
    refused_lhs = ('*', 'exp', '?0', 'exp', '?0')

    def judge(pairs):
        return ['real' if tuple(p[0]) == refused_lhs else 'core' for p in pairs]

    (tmp_path / 'rules.json').write_text('[]')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(
        {'engine_generation': 2, 'operators': _OPERATORS, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=4,
                      dummy_variables=1, extra_internal_terms=['2'], X=256, seed=7,
                      promote_sorts=False, verbose=False, proposals=[proposal], judge=judge,
                      output_file=str(tmp_path / 'mined.json'))
    side = {mode: json.load(open(tmp_path / f'mined{suffix}.json.provenance.json'))
            for mode, suffix in (('f64', ''), ('real', '_real'))}
    entry, = side['f64']['proposals']['trail']
    assert (entry['verdict'], entry['stage']) == ('rejected', 'judge'), entry
    loop = side['f64']['proposals']['judge_loop']
    assert loop['judge_rounds'] == 2 and len(loop['excluded']) == 1, loop
    # `real` licenses the tier: the proposal certifies there in one round
    entry, = side['real']['proposals']['trail']
    assert entry['verdict'] == 'certified', entry
    assert side['real']['proposals']['judge_loop'] == {'judge_rounds': 1, 'excluded': []}


def test_the_mine_judges_with_no_timeout_under_a_watchdog(monkeypatch, caplog) -> None:
    """S2: a verdict never depends on the machine's load. The judge runs with no wall-clock
    timeout; a judgment past `warn_s` is logged, one past `fail_s` fails the mine by name."""
    import logging
    import time
    import simplipy.verify
    from simplipy.mining import judge_tiers
    real_verify = simplipy.verify.verify_ruleset
    seen = []

    def slow_verify(rules, **kwargs):
        seen.append(kwargs.get('judge_timeout_s'))
        if rules[0][0] == ['exp', 'log', '?0']:
            for _ in range(200):
                time.sleep(0.01)
        return real_verify(rules, **kwargs)

    monkeypatch.setattr(simplipy.verify, 'verify_ruleset', slow_verify)
    rule, slow = (['log', 'exp', '?0'], ['?0']), (['exp', 'log', '?0'], ['?0'])
    with caplog.at_level(logging.WARNING, logger='simplipy.mining'):
        assert judge_tiers([rule], processes=1, warn_s=0.05, fail_s=60) == judge_tiers(
            [rule], processes=1)
        judge_tiers([slow], processes=1, warn_s=0.05, fail_s=60)
    assert seen and set(seen) == {0}, seen
    assert any('exp log ?0 -> ?0' in r.getMessage() for r in caplog.records), caplog.text
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r'exp log \?0 -> \?0 ran past 0.3 s'):
        judge_tiers([rule, slow], processes=1, warn_s=0.1, fail_s=0.3)
    assert time.monotonic() - started < 1.5, 'the watchdog did not interrupt the judgment'


class _Crash(Exception):
    pass


def test_a_checkpoint_resumes_only_its_own_mine(tmp_path, monkeypatch) -> None:
    """S3: the checkpoint key holds the operator table, the data's content and the numeric
    environment, and a resume keeps the mine's origin (`created`, `host`) and records itself."""
    import platform
    ops = {k: v for k, v in _OPERATORS.items() if k != 'pow2'}
    out = str(tmp_path / 'mined.json')
    rng = np.random.default_rng(1)
    x_a, x_b = rng.normal(size=(128, 1)), rng.normal(size=(128, 1))
    kwargs = dict(max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
                  seed=7, promote_sorts=False, symbolic_gate=False, output_file=out)

    def crashing(pairs):  # the length-4 pick `log exp log ?0 -> log ?0`
        if any(len(lhs) >= 4 for lhs, _ in pairs):
            raise _Crash
        return ['core'] * len(pairs)

    def core(pairs):
        return ['core'] * len(pairs)

    with pytest.raises(_Crash):
        SimpliPyEngine(operators=ops, rules=[]).find_rules(X=x_a, judge=crashing, **kwargs)
    ckpt = json.load(open(out + '.checkpoint.json'))
    assert ckpt['completed_length'] == 3
    origin = (ckpt['provenance']['created'], ckpt['provenance']['host'])
    other_ops = dict(ops, exp=dict(ops['exp'], precedence=4))
    for engine_ops, x in ((other_ops, x_a), (ops, x_b)):
        with pytest.raises(RuntimeError, match='different mine'):
            SimpliPyEngine(operators=engine_ops, rules=[]).find_rules(X=x, judge=core, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(SimpliPyEngine, 'libm_fingerprint', lambda self: 'another libm')
        with pytest.raises(RuntimeError, match='different mine'):
            SimpliPyEngine(operators=ops, rules=[]).find_rules(X=x_a, judge=core, **kwargs)
    monkeypatch.setattr(platform, 'node', lambda: 'elsewhere')
    SimpliPyEngine(operators=ops, rules=[]).find_rules(X=x_a, judge=core, **kwargs)
    side = json.load(open(out + '.provenance.json'))
    assert (side['created'], side['host']) == origin
    assert [(r['host'], r['after_length']) for r in side['resumed']] == [('elsewhere', 3)]
    assert 'resumed' not in json.load(open(str(tmp_path / 'mined_real.json.provenance.json')))


def test_the_rounds_cap_is_per_source(acj, monkeypatch) -> None:
    """S5: a source the judge keeps refusing is given up on by itself; the others converge."""
    import simplipy.mining
    monkeypatch.setattr(simplipy.mining, '_JUDGE_ROUNDS_MAX', 2)
    other = ['tan', 'atan', 'sin', 'x0']
    stubborn = ('sinh', 'asinh', '?0')

    def judge(pairs):
        return ['real' if tuple(p[0]) == stubborn else 'core' for p in pairs]

    found, stats = RuleMiner(acj)._mine_length([WITNESS, other], _scan(acj), 'default',
                                               judge, {}, ['x0'])
    assert found == [(other, ['sin', 'x0'])], found
    assert stats['stubborn'] == [[WITNESS, [FIRST, SECOND]]], stats


def test_a_stubborn_source_fails_the_mine_by_name_after_the_checkpoint(tmp_path, monkeypatch) -> None:
    import simplipy.mining
    monkeypatch.setattr(simplipy.mining, '_JUDGE_ROUNDS_MAX', 1)
    ops = {k: v for k, v in _OPERATORS.items() if k != 'pow2'}
    out = str(tmp_path / 'mined.json')

    def judge(pairs):  # refuses the one length-4 pick, `log exp log ?0 -> log ?0`
        return ['real' if len(p[0]) >= 4 else 'core' for p in pairs]

    with pytest.raises(RuntimeError, match=r"gave up on 1 source\(s\) of length 4.*'log', 'exp', 'log', 'x0'"):
        SimpliPyEngine(operators=ops, rules=[]).find_rules(
            max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
            X=np.random.default_rng(1).normal(size=(128, 1)), seed=7, promote_sorts=False,
            symbolic_gate=False, output_file=out, judge=judge)
    ckpt = json.load(open(out + '.checkpoint.json'))
    assert ckpt['completed_length'] == 3
    assert [['log', 'exp', 'log', '?0'], ['log', '?0'], 'real'] in ckpt['judged']
