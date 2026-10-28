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
# The channels are compared in `real`: its serve answer leaves the specials standing
# (permissive absorbs and folds them), so every acceptance entry gets to speak.
MODE = 'real'


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
                                       x.shape[0], rule_mode=MODE)
    fit = dict(challenges=16, retries=16, seed=SEED, rtol=1e-11, atol=1e-12)
    scan = core.find_rule_lib(list(src), lib, rule_mode=MODE, **fit, **_confirm(x_confirm))
    mine = core.mine_one_length([list(src)], lib, None, 16, 16, SEED, 1e-11, 1e-12, None, True,
                                rule_mode=MODE, **_confirm(x_confirm))
    hint_ok, why = core.judge_hint(list(src), list(cand), lib, rule_mode=MODE, **fit,
                                   **_confirm(x_confirm))
    return (scan == list(cand), [(s, t) for _, s, t in mine] == [(list(src), list(cand))],
            hint_ok, why)


# (source, candidate, accepted, the hint arm's reason)
PAIRS = [
    # accepted: a true rule strictly below the engine's mark
    (('/', 'sin', 'x0', 'cos', 'x0'), ('tan', 'x0'), True, 'accepted'),
    # true, but permissive's own serve answer already performs it (the mark is the target)
    (('log', 'exp', '+', 'x0', 'x1'), ('+', 'x0', 'x1'), False, 'not_below'),
    # numerically false
    (('sin', 'x0'), ('x0',), False, 'uncertified'),
    # true but not strictly below the mark (the engine already collects the bag)
    (('+', '*', 'x0', 'x1', '*', 'x0', 'x1'), ('*', '2', '*', 'x0', 'x1'), False, 'not_below'),
    # SPECIAL ABSORPTION, by spelling and by the mark: the special sits in the source's
    # tokens (`np.pi`), or only in the engine's endpoint (`exp 1` reads as `np.e`, and
    # `x0^e` with `C = e` is true). The library scan without a mark used to count specials
    # on the raw source alone.
    (('+', 'x0', '*', 'np.pi', '<constant>'), ('+', 'x0', '<constant>'), False,
     'special_absorption'),
    (('pow', 'x0', 'exp', '1'), ('pow', 'x0', '<constant>'), False, 'special_absorption'),
    # A constant-only source the engine collapses itself (`C^e` is finite for C >= 0) has
    # an atomic mark: nothing is below it.
    (('pow', '<constant>', 'exp', '1'), ('<constant>',), False, 'atomic_mark'),
    # `log(1)` is 0 in the table, so real's mark of `x0*log(1)` is the atomic 0 (the pole
    # reason, a value at x0 = 0 changing from 0 to nan, is pinned in test_licence_registry)
    (('*', 'x0', 'log', '1'), ('log', '/', 'x0', 'x0'), False, 'atomic_mark'),
]


@pytest.mark.parametrize('src,cand,accepted,why', PAIRS, ids=[' '.join(p[0]) for p in PAIRS])
def test_one_acceptance_across_channels(eng, rows, src, cand, accepted, why):
    x, x_confirm = rows
    scan, mine, hint, reason = _verdicts(eng, x, x_confirm, src, cand)
    assert scan == mine == hint == accepted, (scan, mine, hint, reason)
    assert reason == why


def _judge(eng, x, src, tgt, seed=SEED, mode='default'):
    return eng._core.judge_pair(list(src), list(tgt), VARS, x.flatten(order='C').tolist(),
                                x.shape[0], 16, 16, seed, 1e-11, 1e-12, None,
                                rule_mode=mode)


class TestJudgePair:
    def test_the_old_hijack_case_is_refused(self, eng, rows):
        # `find_rule`'s short-circuit answered `<constant>` for this source and the pair
        # "confirmed" until the answer was compared with the target. No short-circuit here.
        assert _judge(eng, rows[1], ('exp', '<constant>'), ('0',)) == (False, 'uncertified')

    def test_exact_ground_collapse_is_accepted(self, eng, rows):
        assert _judge(eng, rows[1], ('cos', 'np.pi'), ('(-1)',)) == (True, 'accepted')

    def test_bag_collection_is_accepted(self, eng, rows):
        assert _judge(eng, rows[1], ('+', 'x0', 'x0'), ('*', '2', 'x0')) == (True, 'accepted')

    def test_saturation_identity_is_real_not_f64(self, eng, rows):
        # True over the reals, false in f64 near saturation: f64 never re-judges at high
        # precision, real and permissive do.
        pair = (('atanh', 'tanh', 'x0'), ('x0',))
        assert _judge(eng, rows[1], *pair, mode='default') == (False, 'uncertified')
        assert _judge(eng, rows[1], *pair, mode='real') == (True, 'accepted')
        assert _judge(eng, rows[1], *pair, mode='permissive') == (True, 'accepted')

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


class TestZeroIsZero:
    """A const-free pair agrees relatively, so zero agrees only with zero: in f64 on the
    deployed evaluator's values, in real and permissive at high precision, where a zero is a
    residual that collapses as the precision doubles."""

    @pytest.mark.parametrize('mode', ['default', 'real', 'permissive'])
    def test_a_tiny_value_is_not_zero(self, eng, rows, mode):
        # exp(sinh(-5)) = 5.9e-33: inside the old absolute band, false in every mode.
        assert _judge(eng, rows[1], ('exp', 'sinh', '(-5)'), ('0',), mode=mode) \
            == (False, 'uncertified')

    def test_an_f64_underflow_is_zero_in_f64(self, eng, rows):
        # exp(sinh(-10)) underflows to exactly 0.0: the deployed evaluator realises `-> 0`.
        assert _judge(eng, rows[1], ('exp', 'sinh', '(-10)'), ('0',)) == (True, 'accepted')

    def test_an_exact_zero_is_zero_at_high_precision(self, eng, rows):
        # sin(pi) is 0 exactly; f64 computes 1.2e-16 from the rounded pi.
        pair = (('sin', 'np.pi'), ('0',))
        assert _judge(eng, rows[1], *pair, mode='default') == (False, 'uncertified')
        assert _judge(eng, rows[1], *pair, mode='real') == (True, 'accepted')


@pytest.mark.parametrize('mode', ['default', 'real'])
def test_resolution_and_the_pair_judgment_agree(eng, rows, mode):
    """One numeric verdict for "this target equals this source": the scan resolves
    `<constant>` in `+ x0 <constant>` for `x0 + atanh(tanh 8)` to the literal `8`, and ships
    `+ x0 8` iff `judge_pair` accepts that pair on the same rows. f64 computes
    atanh(tanh 8) = 8 - 1.0e-10, which fails the relative bar where |x0 + 8| < 10 (refused);
    the value is 8, and with those rows a minority real re-judges them (accepts). Judged
    without the mint acceptance (`find_rule`), whose serve mark would fold the ground part."""
    n = 64
    x0 = [100.0 + i for i in range(n - 8)] + [0.0, 0.5, 1.0, -1.0, 2.0, -2.0, 0.25, 3.0]
    x = np.array([[v, 1.0] for v in x0])
    core = eng._core
    src = ['+', 'x0', 'atanh', 'tanh', '8']
    scan = core.find_rule(src, len(src), None, [['+', 'x0', '<constant>']], VARS,
                          x.flatten(order='C').tolist(), n, 16, 16, SEED, 1e-11,
                          1e-12, None, rule_mode=mode)
    ok, _ = _judge(eng, x, src, ['+', 'x0', '8'], mode=mode)
    assert (scan == ['+', 'x0', '8']) == ok, scan
    assert ok == (mode == 'real')
