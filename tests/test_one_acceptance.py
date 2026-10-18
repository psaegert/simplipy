"""One acceptance for every mint channel, and the pair judgment that stage 2 and the hint
arms run.

The batch mine (``mine_one_length``), the proposals' library scan (``find_rule_lib``) and the
hint arm (``judge_hint``) all build one ``Acceptor`` from the engine's own mark: the licence
registry, the serve ordering, the pole entry and stage-2 confirmation. A pair one channel
accepts, every channel accepts.

``judge_pair`` is the scan's per-candidate certification for exactly one pair: no library,
no token bound, no short-circuit. Stage 2 runs it on the confirmation rows, inside the scan.
"""
import numpy as np
import pytest

from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner

from conftest import acj_config_path

VARS = ['x0', 'x1']
SEED = 11


@pytest.fixture(scope='module')
def eng():
    # A RULE-FREE engine over the shipped vocabulary: the mine's own situation.
    shipped = SimpliPyEngine.from_config(acj_config_path())
    return SimpliPyEngine(operators=shipped._operators_config, rules=[])


@pytest.fixture(scope='module')
def rows(eng):
    rng = np.random.default_rng(0)
    miner = RuleMiner(eng)
    x = miner._mining_sample_x(1024, len(VARS), rng)
    x_confirm = miner._mining_sample_x(2048, len(VARS), rng)
    return x, x_confirm


def _confirm(x_confirm):
    return {'confirm_x': x_confirm.flatten(order='C').tolist(),
            'confirm_rows': x_confirm.shape[0], 'confirm_seed': 5}


def _verdicts(eng, x, x_confirm, src, cand):
    """The three channels' verdicts on one (source, candidate) pair, over a library that
    holds exactly the candidate."""
    core = eng._core
    lib = core.build_candidate_library([list(cand)], VARS, x.flatten(order='C').tolist(),
                                       x.shape[0], rule_mode='default')
    fit = dict(challenges=16, retries=16, seed=SEED, rtol=1e-11, atol=1e-12)
    scan = core.find_rule_lib(list(src), lib, rule_mode='default', **fit, **_confirm(x_confirm))
    mine = core.mine_one_length([list(src)], lib, None, 16, 16, SEED, 1e-11, 1e-12, None, True,
                                rule_mode='default', **_confirm(x_confirm))
    hint_ok, why = core.judge_hint(list(src), list(cand), lib, rule_mode='default', **fit,
                                   **_confirm(x_confirm))
    return scan == list(cand), mine == [(list(src), list(cand))], hint_ok, why


# (source, candidate, accepted, the hint arm's reason)
PAIRS = [
    # accepted: a true rule strictly below the engine's mark
    (('/', 'sin', 'x0', 'cos', 'x0'), ('tan', 'x0'), True, 'accepted'),
    (('log', 'exp', '+', 'x0', 'x1'), ('+', 'x0', 'x1'), True, 'accepted'),
    # numerically false
    (('sin', 'x0'), ('x0',), False, 'uncertified'),
    # true but not strictly below the mark (the engine already collects the bag)
    (('+', '*', 'x0', 'x1', '*', 'x0', 'x1'), ('*', '2', '*', 'x0', 'x1'), False, 'not_below'),
    # SPECIAL ABSORPTION, by spelling and by the mark: the special sits in the source's
    # tokens (`np.pi`), or only in the engine's endpoint (`exp 1` reads as `np.e`). The
    # library scan without a mark used to count specials on the raw source alone.
    (('+', 'x0', '*', 'np.pi', '<constant>'), ('+', 'x0', '<constant>'), False,
     'special_absorption'),
    (('pow', '<constant>', 'exp', '1'), ('<constant>',), False, 'special_absorption'),
    # POLE: a.e. equal and below the mark, but the value at x0 = 0 changes (0 -> nan)
    (('*', 'x0', 'log', '1'), ('log', '/', 'x0', 'x0'), False, 'pole_class_change'),
]


@pytest.mark.parametrize('src,cand,accepted,why', PAIRS, ids=[' '.join(p[0]) for p in PAIRS])
def test_one_acceptance_across_channels(eng, rows, src, cand, accepted, why):
    x, x_confirm = rows
    scan, mine, hint, reason = _verdicts(eng, x, x_confirm, src, cand)
    assert scan == mine == hint == accepted, (scan, mine, hint, reason)
    assert reason == why


def _judge(eng, x, src, tgt, seed=SEED):
    return eng._core.judge_pair(list(src), list(tgt), VARS, x.flatten(order='C').tolist(),
                                x.shape[0], 16, 16, seed, 1e-11, 1e-12, None,
                                rule_mode='default')


class TestJudgePair:
    def test_the_old_hijack_case_is_refused(self, eng, rows):
        # `find_rule`'s short-circuit answered `<constant>` for this source and the pair
        # "confirmed" until the answer was compared with the target. No short-circuit here.
        assert _judge(eng, rows[1], ('exp', '<constant>'), ('0',)) == (False, 'uncertified')

    def test_exact_ground_collapse_is_accepted(self, eng, rows):
        assert _judge(eng, rows[1], ('cos', 'np.pi'), ('(-1)',)) == (True, 'accepted')

    def test_bag_collection_is_accepted(self, eng, rows):
        assert _judge(eng, rows[1], ('+', 'x0', 'x0'), ('*', '2', 'x0')) == (True, 'accepted')

    def test_saturation_identity_keeps_todays_verdict(self, eng, rows):
        # True over the reals, false in f64 near saturation: the high-precision rescue
        # accepts it in every mode today. The per-mode numerics (f64 refuses) come later.
        assert _judge(eng, rows[1], ('atanh', 'tanh', 'x0'), ('x0',)) == (True, 'accepted')

    def test_a_target_needing_resolution_is_not_the_pair_asked(self, eng, rows):
        # The scan would emit the resolved literal, a different rule from the one asked.
        assert _judge(eng, rows[1], ('/', 'x0', '*', '2', 'x0'), ('<constant>',)) \
            == (False, 'respelled')

    def test_a_ground_identity_is_judged_not_reevaluated(self, eng, rows):
        """Stage 2 used to re-run `find_rule`, whose ground short-circuit answered with the
        source's own f64 evaluation (`tan atan 7` -> `7.000000000000001`) instead of judging
        the target, so every inexactly-evaluating ground identity was refused. Measured on
        the acj-3-2c reference: 46 such picks. The pair judgment accepts the true one."""
        core = eng._core
        assert core.evaluate_constant_subtree(['tan', 'atan', '7']) != '7'
        assert _judge(eng, rows[1], ('tan', 'atan', '7'), ('7',)) == (True, 'accepted')
