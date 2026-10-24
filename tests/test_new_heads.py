"""The 0.15 heads (floor, ceil, max, min, mod, erf, harmonic, atan2, besselj) in the engine:
functions of literals fold under each mode's convention, every layer without a sound treatment
of a head refuses it, and the miner mines with them."""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.verify import _contract, _monitor

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')
MODES = ('f64', 'real', 'permissive')


@pytest.fixture(scope='module')
def bare():
    """The acj2-4 operators with an empty rule set per mode: the core on its own."""
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])


@pytest.mark.parametrize('mode', MODES)
def test_step_and_lattice_heads_of_literals_fold(bare, mode):
    s = lambda t: bare.simplify(t.split(), mode=mode)  # noqa: E731
    assert s('floor 2.5') == ['2']
    assert s('ceil -2.5') == ['-2']
    assert s('max 1 / 1 3') == ['1']
    assert s('mod 7 -3') == ['-2']
    assert s('mod 1 0') == ['float("nan")']
    assert s('floor float("inf")') == ['float("inf")']
    # a transcendental head of a literal stays written where its float is dearer
    assert s('erf 1') == ['erf', '1']


def test_f64_reads_a_literal_at_its_float_and_real_at_its_rational(bare):
    assert bare.simplify(['floor', '0.99999999999999999'], mode='f64') == ['1']
    assert bare.simplify(['floor', '0.99999999999999999'], mode='real') == ['0']
    assert bare.simplify(['erf', '0'], mode='f64') == ['0']
    assert bare.simplify(['besselj', '0', '0'], mode='f64') == ['1']
    # erf is odd in float64 bit for bit, so its sign leaves it
    assert bare.simplify(['erf', 'neg', 'x0'], mode='f64') == ['neg', 'erf', 'x0']


def test_the_evaluator_folds_with_the_deployed_kernels(bare):
    import simplipy.operators as ops
    ev = bare._core.evaluate_constant_subtree
    for toks, f, args in ((['erf', '0.5'], ops.erf, (0.5,)), (['harmonic', '2.5'], ops.harmonic, (2.5,)),
                          (['besselj', '1', '2'], ops.besselj, (1.0, 2.0)),
                          (['atan2', '-3', '0.5'], ops.atan2, (-3.0, 0.5)),
                          (['mod', '-7', '3'], ops.mod, (-7.0, 3.0))):
        want = float(f(*[np.array([a]) for a in args])[0])
        assert float(ev(toks)) == want, toks


def test_the_judge_reads_the_new_heads_on_both_axes():
    """The contract decides whether a rule is true over the reals; float64's realisation of a
    rule with a 0.15 head also answers to the core's f64 bar (the jumps, the signed zeros)."""
    j = _contract.judge_rule
    for lhs, rhs in (('floor floor _0', 'floor _0'), ('max _0 _0', '_0'),
                     ('ceil neg _0', 'neg floor _0'), ('erf neg _0', 'neg erf _0'),
                     ('mod _0 1', '- _0 floor _0'), ('atan2 neg _0 1', 'neg atan2 _0 1')):
        r = j(lhs.split(), rhs.split())
        assert r['verdict'] in ('CERTIFIED', 'TOLERATED') and r['tier'] == 'core', (lhs, rhs, r)
    # true over the reals, broken by float64 at x = -1e-17: the real tier, never f64's
    for lhs, rhs in (('floor + _0 1', '+ floor _0 1'), ('mod + _0 1 1', 'mod _0 1'),
                     ('floor mod ?0 1', '0')):
        r = j(lhs.split(), rhs.split())
        assert r['verdict'] in ('CERTIFIED', 'TOLERATED') and r['tier'] == 'real', (lhs, rhs, r)
        assert r['f64_bar']['ok'] is False
    # false over the reals at a jump, and in float64: no tier
    for lhs, rhs in (('+ floor _0 1', 'ceil _0'), ('mod _0 1', '_0'), ('harmonic _0', 'log _0')):
        r = j(lhs.split(), rhs.split())
        assert r['verdict'] == 'KILL' and r['tier'] == 'reject', (lhs, rhs, r)
    # the constant lane: a fitted witness, as for every other head
    for lhs, rhs in (('floor <constant>', '<constant>'), ('max <constant> 1', '<constant>'),
                     ('harmonic <constant>', '<constant>'), ('atan2 <constant> 1', '<constant>')):
        r = j(lhs.split(), rhs.split())
        assert r['verdict'] in ('CERTIFIED', 'TOLERATED') and r['tier'] == 'core', (lhs, rhs, r)
    # besselj has no contract kernel yet: the deployed lane alone, and no constant
    ok = j('besselj 0 neg _0'.split(), 'besselj 0 _0'.split())
    assert ok['verdict'] == 'F64-ONLY' and ok['tier'] == 'f64', ok
    assert j('besselj 1 <constant>'.split(), '<constant>'.split())['verdict'] == 'UNSUPPORTED-SHAPE'
    # a head the judge does not know at all stays refused
    assert j('gamma _0'.split(), '_0'.split())['verdict'] == 'UNSUPPORTED-SHAPE'
    assert _contract._tier('F64-ONLY', None) == 'reject'


def test_the_constant_witness_is_polished_where_it_is_determined():
    """A target constant's witness is polished at the generic point its fit was determined at
    (a lattice head is flat in it at the first one), and re-rounded at every rung of the measure
    lane: these identities were convicted."""
    j = _contract.judge_rule
    for lhs, rhs in (('max x0 acos <constant>', 'max <constant> x0'),
                     ('max x0 log <constant>', 'max <constant> x0'),
                     ('atan2 x0 acosh <constant>', 'atan2 x0 <constant>'),
                     ('+ x0 log <constant>', '+ <constant> x0')):
        r = j(lhs.split(), rhs.split())
        assert r['verdict'] == 'CERTIFIED', (lhs, rhs, r)


def test_the_fit_keeps_the_witness_it_found():
    """A float64 fit's witness is snapped only where the snapped value reproduces the source
    at least as well (6e-44 is no 0, 0.99999999999846 no 1), and a fitted witness is no
    written bound for the saturation test: these constant collapses are true and realised."""
    j = _contract.judge_rule
    for lhs in ('asinh exp <constant>', 'harmonic exp <constant>', 'asin sin erf <constant>',
                'erf <constant>', 'tanh <constant>'):
        r = j(lhs.split(), ['<constant>'])
        assert (r['verdict'], r['tier']) == ('CERTIFIED', 'core'), (lhs, r)
    # a written bound stays one: tanh never attains 1 at a finite argument
    assert j('tanh <constant>'.split(), ['1'])['verdict'] == 'KILL'


def test_contract_kernels_and_their_bands():
    """The contract's 0.15 kernels against mpmath at a higher precision, and the bands: a value
    inside the working-precision band of a jump, a pole, an asymptote or atan2's cut is
    Unresolved, an exact one is read exactly."""
    from mpmath import mp, mpf
    ops, U = _contract.OPS, _contract.Unresolved
    old = mp.dps
    try:
        mp.dps = 50
        for x in ('1e-300', '1e-20', '0.5', '-0.5', '-2.25', '10.25', '1e17', '-0.999'):
            with mp.workdps(400):
                want = mp.harmonic(mpf(x))
            got = ops['harmonic'](mpf(x))
            assert abs(got - want) <= abs(want) * mpf(10) ** -45, (x, got, want)
        assert _contract.misnan(ops['harmonic'](mpf(-2))) and ops['harmonic'](mpf(0)) == 0
        assert ops['floor'](mpf(-1)) == -1 and ops['ceil'](mpf('-0.5')) == 0
        assert ops['mod'](mpf(-7), mpf(3)) == 2 and ops['mod'](mpf(7), mpf(-3)) == -2
        assert ops['mod'](mpf(-3), mpf('inf')) == mpf('inf')
        assert _contract.misnan(ops['mod'](mpf(1), mpf(0)))
        assert ops['atan2'](mpf(1), mpf(0)) == mp.pi / 2
        tiny = mpf(10) ** -45
        for f, args in (('floor', (1 + tiny,)), ('ceil', (-tiny,)), ('harmonic', (-2 + tiny,)),
                        ('mod', (3 - tiny, mpf(3))), ('mod', (tiny, mpf(3))),
                        ('atan2', (mpf(0), mpf(-1))), ('atan2', (tiny, mpf(-1))),
                        ('atan2', (mpf(0), mpf(0))), ('erf', (mpf(12),))):
            with pytest.raises(U):
                ops[f](*args)
        # outside the band the value is read: erf(9) is 1 - 4e-37, harmonic near a pole is large
        assert 1 - ops['erf'](mpf(9)) > 0
        assert ops['harmonic'](-2 + mpf(10) ** -30) < -1e29
    finally:
        mp.dps = old


def test_monitor_and_oracle_read_the_new_heads():
    """The corpus monitor and sort promotion's high-precision oracle evaluate the eight heads
    with the contract's kernels (the monitor with IEEE's zero signs); besselj stays unknown to
    both."""
    assert not _monitor.unknown_tokens(['floor', 'mod', 'x0', 'atan2', 'x1', 'erf', 'x0'])
    assert 'besselj' in _monitor.unknown_tokens(['besselj', '1', 'x0'])
    neg0 = _monitor._NegZero(0)
    assert _monitor._zero_signbit('ceil', [-0.5]) and not _monitor._zero_signbit('floor', [0.5])
    assert _monitor._zero_signbit('mod', [3.0, -3.0]) and _monitor._zero_signbit('atan2', [neg0, 1.0])
    assert not _monitor._zero_signbit('max', [neg0, 0.0]) and _monitor._zero_signbit('min', [neg0, 0.0])
    with pytest.raises(_monitor.Unresolved):
        _monitor._zero_signbit('besselj', [1.0, 0.0])
    from mpmath import mpf
    from simplipy.promotion._hp_equiv import evaluate
    assert evaluate(['floor', 'x0'], {'x0': mpf(1.5)}, []) == 1
    with pytest.raises(ValueError):
        evaluate(['besselj', '1', 'x0'], {'x0': mpf(1.5)}, [])


def test_a_small_mine_mines_new_head_rules():
    """End to end, in every mode: the rules true over the reals and realised in float64 appear
    in all three files."""
    full = yaml.safe_load(open(ACJ2_4))['operators']
    ops = {k: full[k] for k in ('neg', 'floor', 'ceil', 'max')}
    e = SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])
    e.find_rules(max_source_pattern_length=3, max_target_pattern_length=3, dummy_variables=2,
                 extra_internal_terms=['0', '1'], X=128, seed=3, reset_rules=True,
                 promote_sorts=False, judge_processes=1)

    def has(rules, lhs, rhs):
        strip = lambda t: [x if x[0] not in '?_!$' or not x[1:].isdigit() else '#' + x[1:]  # noqa: E731
                           for x in t]
        return any(strip(list(a)) == lhs.split() and strip(list(b)) == rhs.split() for a, b in rules)

    for rules in (e.simplification_rules, e.real_simplification_rules,
                  e.permissive_simplification_rules):
        assert has(rules, 'floor floor #0', 'floor #0'), rules
        assert has(rules, 'ceil floor #0', 'floor #0'), rules
        # max(x, x) = x and the argument order are the engine's: no rule states them
        assert not any(a[0] == 'max' and b[0] == 'max' for a, b in rules), rules
        assert not has(rules, 'max #0 #0', '#0'), rules


def test_max_and_min_are_order_independent_and_idempotent(bare):
    for mode in MODES:
        s = lambda t: bare.simplify(t.split(), mode=mode)  # noqa: E731
        assert s('max x0 1') == s('max 1 x0') and s('min sin x0 x1') == s('min x1 sin x0')
        assert s('max x0 x0') == ['x0'] and s('min erf x1 erf x1') == ['erf', 'x1']
        # two constants are independent: the pair is no `max(a, a)`
        assert s('max * <constant> x0 * <constant> x0')[0] == 'max'


def test_f64_writes_the_negative_zero_and_reads_it(bare):
    """f64 follows float64: a -0 is written `-0.0` (Python's `-0` is the integer zero, +0), a
    ground fold writes its float exactly, and an inverse or `atan2` reads the sign. `real` and
    `permissive` have one zero."""
    f64 = lambda t: bare.simplify(t.split(), mode='f64')  # noqa: E731
    for src, want in (('ceil -0.5', ['-0.0']), ('mod 3 -3', ['-0.0']),
                      ('inv mod 3 -3', ['float("-inf")']), ('inv ceil -0.5', ['float("-inf")']),
                      ('inv -0.0', ['float("-inf")']), ('pow -0.0 -2', ['float("inf")']),
                      ('atan2 0 -0.0', ['np.pi']), ('+ x0 -0.0', ['x0']), ('-0', ['0']),
                      # a product or negation of numbers that is zero has float64's sign
                      ('* -1 0', ['-0.0']), ('* 0 -1', ['-0.0']), ('* -1 0.0', ['-0.0']),
                      ('neg 0', ['-0.0']), ('/ 0 -2', ['-0.0']), ('* -1 -0.0', ['0']),
                      ('* -1 * -1 0', ['0'])):
        assert f64(src) == want, (src, f64(src))
    for mode in ('real', 'permissive'):
        assert bare.simplify(['inv', '-0.0'], mode=mode) == ['float("inf")']
        assert bare.simplify(['ceil', '-0.5'], mode=mode) == ['0']
    # one sign bit on top of the zero
    assert bare.complexity(['-0.0'], mode='f64') == bare.complexity(['0'], mode='f64') + 1000


def test_atan2_reads_both_zero_signs_however_it_is_built(bare):
    import simplipy.operators as ops
    env = {**vars(ops), 'np': np, 'simplipy': __import__('simplipy')}

    def same(src, tgt):
        fs, ft = (eval('lambda x0: ' + bare.prefix_to_infix(list(t), realization=True), env)
                  for t in (src, tgt))
        for x in (0.0, 1.5, 3.0):
            a, b = (float(np.asarray(f(np.array([x]))).ravel()[0]) for f in (fs, ft))
            assert a == b and np.signbit(a) == np.signbit(b), (src, tgt, x, a, b)
    # parsed: a negated sum that vanishes on x0 >= 0 keeps its zero's sign in either argument
    for src in ('atan2 neg - x0 abs x0 -1', 'atan2 neg - x0 x0 -1', 'atan2 ceil -0.5 -1',
                'atan2 0 neg - x0 abs x0', 'atan2 neg - x0 abs x0 neg - x0 abs x0',
                'atan2 0 * 2 neg - x0 abs x0'):
        same(src.split(), bare.simplify(src.split(), mode='f64'))
    # built by substitution into a rule's right-hand side
    cfg = yaml.safe_load(open(ACJ2_4))['operators']
    for rule, src, want in (
            ((('erf', 'atan2', '_0', '_1'), ('atan2', '_0', 'neg', '_1')),
             'erf atan2 0 - x0 abs x0', ['atan2', '0', 'neg', '-', 'x0', 'abs', 'x0']),
            ((('erf', 'atan2', '_0', 'sin', '_1'), ('atan2', 'neg', '_0', '_1')),
             'erf atan2 - x0 abs x0 sin 1', ['neg', 'atan2', '-', 'x0', 'abs', 'x0', '1'])):
        e = SimpliPyEngine(operators=cfg, rules=[rule], rules_real=[], rules_permissive=[])
        assert e.simplify(src.split(), mode='f64') == want


def test_the_mint_refuses_laws_float64_breaks_at_a_jump_in_f64_alone(bare):
    """A law of the reals float64 breaks at a jump is minted in real and permissive and refused
    in f64; a rule false at a jump over the reals too is refused in every mode."""
    from simplipy.mining import RuleMiner
    X = RuleMiner(bare)._mining_sample_x(256, 2, np.random.default_rng(0))

    def certified(src, tgt, mode):
        return bare._core.judge_pair(src.split(), tgt.split(), ['x0', 'x1'], X.flatten().tolist(),
                                     X.shape[0], rtol=1e-11, atol=1e-12, rule_mode=mode)[0]
    for src, tgt in (('floor + x0 1', '+ floor x0 1'), ('mod + x0 1 1', 'mod x0 1'),
                     ('floor mod x0 1', '0')):
        assert not certified(src, tgt, 'default'), (src, tgt)
        assert certified(src, tgt, 'real') and certified(src, tgt, 'permissive'), (src, tgt)
    for mode in ('default', 'real', 'permissive'):
        assert not certified('+ floor x0 1', 'ceil x0', mode)
        assert certified('floor floor x0', 'floor x0', mode)


@pytest.mark.parametrize('mode', MODES)
def test_sort_promotion_reads_the_new_heads(bare, mode):
    """Rules with the eight heads are promoted to the sorts they certify at, the contract atom
    bar reading their contract values, and a rule false only where a step head's argument
    crosses an integer at no atom (sinh(x) + 1/2 = k) is killed at that preimage."""
    from simplipy.promotion import promote
    rules = [(lhs.split(), rhs.split()) for lhs, rhs in (
        ('floor floor ?0', 'floor ?0'), ('mod ?0 1', '- ?0 floor ?0'),
        ('atan2 neg ?0 1', 'neg atan2 ?0 1'), ('erf neg ?0', 'neg erf ?0'),
        ('- ceil + sinh ?0 0.5 floor + sinh ?0 0.5', '1'))]
    kept, report = promote(rules, bare, rule_mode=mode, run_positive_controls=False)
    kept = {' '.join(a) + ' -> ' + ' '.join(b) for a, b in kept}
    for rule in ('floor floor _0 -> floor _0', 'mod _0 1 -> - _0 floor _0',
                 'atan2 neg _0 1 -> neg atan2 _0 1', 'erf neg _0 -> neg erf _0'):
        assert rule in kept, (rule, kept)
    assert not any(r.startswith('- ceil') for r in kept), kept
    assert any('PREIMAGE' in str(k[2]) for k in report['killed_q']), report['killed_q']
