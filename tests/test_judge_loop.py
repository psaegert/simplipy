"""The mine's per-length judge loop (miner.md 2.6, 2.7): the judge's tier is decided inside
the scan, a refused rule never prunes a longer source, and the scan is a function of the
global source index alone."""
import contextlib
import json
import logging
import threading
import time

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


class _HeldJudge:
    """The judge (`verify_ruleset`) with each judgment of the rule `lhs` HELD: it returns only
    when the test releases it, so nothing the watchdog decides depends on how fast the machine
    judges. Every other rule goes to the real judge, which the tests run under the default
    thresholds only; the short ones are for held judgments, which cannot finish in time.
    `ended_by` is the exception that ended the last held judgment (the watchdog's), or None
    for one that was released; `timeouts` is the wall-clock timeout of every judgment."""

    #: A held judgment nobody releases returns after this long, so a watchdog that never fires
    #: fails the test instead of hanging it. A watchdog that fires never waits for it.
    GUARD_S = 30.0

    def __init__(self, monkeypatch, lhs: list) -> None:
        import simplipy.verify
        self.real, self.lhs = simplipy.verify.verify_ruleset, lhs
        self.release = threading.Event()
        self.ended_by: type | None = None
        self.timeouts: list = []
        monkeypatch.setattr(simplipy.verify, 'verify_ruleset', self)

    def __call__(self, rules: list, **kwargs) -> dict:
        self.timeouts.append(kwargs.get('judge_timeout_s'))
        if rules[0][0] == self.lhs:
            self.ended_by = None
            deadline = time.monotonic() + self.GUARD_S
            try:
                # Short waits, so the watchdog's exception lands between two of them.
                while not self.release.wait(0.01) and time.monotonic() < deadline:
                    pass
            except BaseException as error:
                self.ended_by = type(error)
                raise
            finally:
                self.release.clear()
        return self.real(rules, **kwargs)

    @contextlib.contextmanager
    def released_by_its_warning(self):
        """Release a held judgment when the watchdog logs it as still running."""
        held, release = f"judgment of {' '.join(self.lhs)} -> ", self.release

        class _Release(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if held in record.getMessage() and 'still running' in record.getMessage():
                    release.set()

        handler = _Release(logging.WARNING)
        log = logging.getLogger('simplipy.mining')
        log.addHandler(handler)
        try:
            yield
        finally:
            log.removeHandler(handler)


def test_the_mine_judges_with_no_timeout_under_a_watchdog(monkeypatch, caplog) -> None:
    """A verdict never depends on the machine's load. The judge runs with no wall-clock
    timeout; a judgment past `warn_s` is logged, one past `fail_s` fails the mine by name.
    Which judgment runs long is the test's choice (`_HeldJudge`), not the machine's."""
    import simplipy.mining
    from simplipy.mining import judge_tiers
    rule, held = (['log', 'exp', '?0'], ['?0']), (['exp', 'log', '?0'], ['?0'])
    judge = _HeldJudge(monkeypatch, held[0])
    with caplog.at_level(logging.WARNING, logger='simplipy.mining'):
        assert judge_tiers([rule], processes=1, warn_s=0.05) == judge_tiers([rule], processes=1)
        # held until the watchdog logs it as still running, then judged: logged, not failed
        with judge.released_by_its_warning():
            judge_tiers([held], processes=1, warn_s=0.05)
    assert judge.ended_by is None, judge.ended_by
    assert judge.timeouts and set(judge.timeouts) == {0}, judge.timeouts
    assert any('judgment of exp log ?0 -> ?0 is still running' in r.getMessage()
               for r in caplog.records), caplog.text
    # held until the watchdog ends it: it fails the mine by the rule's name
    with pytest.raises(RuntimeError, match=r'exp log \?0 -> \?0 ran past 0.3 s'):
        judge_tiers([held], processes=1, warn_s=0.1, fail_s=0.3)
    assert judge.ended_by is simplipy.mining._JudgmentOverrun, 'the watchdog did not interrupt the judgment'


class _Crash(Exception):
    pass


def test_a_checkpoint_resumes_only_its_own_mine(tmp_path, monkeypatch) -> None:
    """The checkpoint key holds the operator table, the data's content and the numeric
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
    """A source the judge keeps refusing is given up on by itself; the others converge."""
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


def test_the_final_gate_judges_before_every_prune_and_what_ships(tmp_path, monkeypatch) -> None:
    """A prune leans only on licensed rules (a rule pruned as covered by one the final gate
    then drops would be lost), and the final gate judges the set that ships."""
    events = []
    real_gate, real_prune = RuleMiner._final_gate, SimpliPyEngine.prune_covered_rules

    def gate(self, rule_mode, judge, judged):
        events.append(('gate', rule_mode, len(self._mode_rules(rule_mode))))
        return real_gate(self, rule_mode, judge, judged)

    def prune(self, verbose=False, *, rule_mode):
        events.append(('prune', rule_mode, None))
        return real_prune(self, verbose, rule_mode=rule_mode)

    monkeypatch.setattr(RuleMiner, '_final_gate', gate)
    monkeypatch.setattr(SimpliPyEngine, 'prune_covered_rules', prune)
    # (`sin`: the prune probes a wide slot with `sin(x_i)`, so its vocabulary needs one)
    ops = dict({k: v for k, v in _OPERATORS.items() if k != 'pow2'},
               sin={"realization": "np.sin", "alias": [], "arity": 1, "precedence": 3,
                    "commutative": False})
    engine = SimpliPyEngine(operators=ops, rules=[])
    engine.find_rules(max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
                      X=np.random.default_rng(1).normal(size=(128, 1)), seed=7, prune='covered',
                      judge=lambda pairs: ['core'] * len(pairs))
    f64 = [kind for kind, mode, _ in events if mode == 'default']
    assert f64 == ['gate', 'prune', 'gate', 'prune', 'gate'], events
    assert events[[e[1] for e in events].index('real') - 1][0] == 'gate'


def test_a_stubborn_source_fails_the_mine_by_name_after_the_checkpoint(tmp_path, monkeypatch) -> None:
    import simplipy.mining
    monkeypatch.setattr(simplipy.mining, '_JUDGE_ROUNDS_MAX', 1)
    ops = {k: v for k, v in _OPERATORS.items() if k != 'pow2'}
    out = str(tmp_path / 'mined.json')

    # refuses the length-4 picks, `log exp log ?0 -> log ?0` and `log exp neg ?0 -> neg ?0`
    # (the latter passes f64's special points where exp leaves the range, 750 and 1e17)
    def judge(pairs):
        return ['real' if len(p[0]) >= 4 else 'core' for p in pairs]

    with pytest.raises(RuntimeError, match=r"gave up on 2 source\(s\) of length 4.*"
                                           r"'log', 'exp', 'log', 'x0'.*'log', 'exp', 'neg', 'x0'"):
        SimpliPyEngine(operators=ops, rules=[]).find_rules(
            max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
            X=np.random.default_rng(1).normal(size=(128, 1)), seed=7, promote_sorts=False,
            symbolic_gate=False, output_file=out, judge=judge)
    ckpt = json.load(open(out + '.checkpoint.json'))
    assert ckpt['completed_length'] == 3
    assert [['log', 'exp', 'log', '?0'], ['log', '?0'], 'real'] in ckpt['judged']


def test_the_watchdog_fails_the_mine_from_any_thread(monkeypatch) -> None:
    """A mine may run off the main thread, and a judgment that never returns must still fail
    it at `fail_s`: the watchdog interrupts the judgment in its own thread, not only when it
    returns. The judgment is held until released (`_HeldJudge`): it never returns before
    the watchdog fires, however loaded the machine."""
    import simplipy.mining
    from simplipy.mining import judge_tiers
    rule, held = (['log', 'exp', '?0'], ['?0']), (['exp', 'log', '?0'], ['?0'])
    expected = judge_tiers([rule], processes=1)
    judge = _HeldJudge(monkeypatch, held[0])
    outcome: dict = {}

    def mine() -> None:
        try:
            judge_tiers([held], processes=1, warn_s=0.1, fail_s=0.3)
        except RuntimeError as error:
            outcome['error'] = str(error)
        outcome['ended_by'] = judge.ended_by
        # the thread carries no exception the watchdog left behind
        outcome['after'] = judge_tiers([rule], processes=1)

    worker = threading.Thread(target=mine)
    worker.start()
    worker.join(2 * judge.GUARD_S)
    assert not worker.is_alive()
    assert 'exp log ?0 -> ?0 ran past 0.3 s' in outcome.get('error', ''), outcome
    assert outcome['ended_by'] is simplipy.mining._JudgmentOverrun, 'the watchdog waited for the judgment to return'
    assert outcome['after'] == expected


def test_a_judgment_without_a_timeout_leaves_the_alarm_alone() -> None:
    """The mine judges with no wall-clock alarm, and it neither arms one nor cancels or
    replaces the process's own: a SIGALRM handler and a pending alarm survive a judgment."""
    import signal
    from simplipy.verify import verify_ruleset

    def handler(signum, frame):
        raise AssertionError('the test alarm fired')

    previous = signal.signal(signal.SIGALRM, handler)
    try:
        signal.alarm(100)
        verify_ruleset([[['log', 'exp', '?0'], ['?0']]], judge_timeout_s=0)
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.alarm(0) > 0, 'the pending alarm was cancelled'
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _mode_mine(directory, crash_in=None, **kw):
    """A small three-mode mine in `directory`, recording which mode each mined length
    belongs to; with `crash_in`, it dies when that mode's first length starts."""
    modes: list = []
    original = RuleMiner._mine_length

    def spy(self, sources, mine_, rule_mode, judge_, judged, dummy_variables):
        modes.append(rule_mode)
        if rule_mode == crash_in:
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
    return modes, engine


_FILES = ('mined.json', 'mined_real.json', 'mined_permissive.json')


def test_a_crash_in_a_later_mode_reuses_the_finished_files(tmp_path) -> None:
    """RESUME FROM DISK FIRST, a finished mode: the f64 and real files are final when the
    permissive mine dies, and their sidecars record this mine's key and each file's sha256.
    The rerun takes both as they are and mines permissive alone, to the files a one-shot mine
    writes."""
    import hashlib
    once, _ = _mode_mine(tmp_path / 'once')
    assert set(once) == {'default', 'real', 'permissive'}
    with pytest.raises(KeyboardInterrupt):
        _mode_mine(tmp_path / 'crashed', crash_in='permissive')
    finished = {name: (tmp_path / 'crashed' / name).read_bytes() for name in _FILES[:2]}
    assert json.loads(finished['mined.json']), 'vacuous: an empty f64 file'
    side = json.load(open(tmp_path / 'crashed' / 'mined.json.provenance.json'))
    assert side['resume']['rules_sha256'] == hashlib.sha256(finished['mined.json']).hexdigest()
    rerun, engine = _mode_mine(tmp_path / 'crashed')
    assert set(rerun) == {'permissive'}, rerun
    for name in _FILES:
        assert (tmp_path / 'once' / name).read_bytes() == \
            (tmp_path / 'crashed' / name).read_bytes(), name
    for name, raw in finished.items():
        assert (tmp_path / 'crashed' / name).read_bytes() == raw, name
    # the engine ends holding the three files, as after a one-shot mine
    assert [(tuple(lhs), tuple(rhs)) for lhs, rhs in engine.simplification_rules] == \
        [(tuple(lhs), tuple(rhs)) for lhs, rhs in json.loads(finished['mined.json'])]


def test_a_finished_file_is_reused_only_by_its_own_mine(tmp_path) -> None:
    """The reuse is a resume, so it answers to the mine's key and the file's bytes: other
    parameters, or a file edited after the mine, mine the mode again."""
    _mode_mine(tmp_path / 'a')
    again, _ = _mode_mine(tmp_path / 'a')
    assert again == [], 'vacuous: the same mine reuses all three files'
    other, _ = _mode_mine(tmp_path / 'a', interval_node_budget=12345)
    assert set(other) == {'default', 'real', 'permissive'}, other
    rules = json.load(open(tmp_path / 'a' / 'mined.json'))
    assert rules, 'vacuous: an empty file edits to the same bytes'
    (tmp_path / 'a' / 'mined.json').write_text(json.dumps(rules[1:], indent=4))
    edited, _ = _mode_mine(tmp_path / 'a', interval_node_budget=12345)
    # f64 mines again, to the same rules: the later modes' inputs are unchanged, so they are
    # reused
    assert set(edited) == {'default'}, edited
    assert json.load(open(tmp_path / 'a' / 'mined.json')) == rules
