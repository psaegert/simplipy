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
    # besselj too has its contract kernel: an even order's parity is core
    ok = j('besselj 0 neg _0'.split(), 'besselj 0 _0'.split())
    assert ok['verdict'] == 'CERTIFIED' and ok['tier'] == 'core', ok
    # a head the judge does not know at all stays refused
    assert j('gamma _0'.split(), '_0'.split())['verdict'] == 'UNSUPPORTED-SHAPE'


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
    """The corpus monitor and sort promotion's high-precision oracle evaluate the nine heads
    with the contract's kernels (the monitor with IEEE's zero signs, and besselj with its own
    kernel of the same cells)."""
    assert not _monitor.unknown_tokens(['floor', 'mod', 'x0', 'atan2', 'x1', 'erf', 'x0'])
    assert not _monitor.unknown_tokens(['besselj', '1', 'x0'])
    neg0 = _monitor._NegZero(0)
    assert _monitor._zero_signbit('ceil', [-0.5]) and not _monitor._zero_signbit('floor', [0.5])
    assert _monitor._zero_signbit('mod', [3.0, -3.0]) and _monitor._zero_signbit('atan2', [neg0, 1.0])
    assert not _monitor._zero_signbit('max', [neg0, 0.0]) and _monitor._zero_signbit('min', [neg0, 0.0])
    from mpmath import mpf
    assert not _monitor._zero_signbit('besselj', [mpf(1), mpf(0)])
    from simplipy.promotion._hp_equiv import evaluate
    assert evaluate(['floor', 'x0'], {'x0': mpf(1.5)}, []) == 1
    assert abs(evaluate(['besselj', '1', 'x0'], {'x0': mpf(1.5)}, []) - mpf('0.5579365079100996')) < 1e-15


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


def test_max_and_min_are_lattice_bags(bare):
    """Every nesting and order of one bag is one answer: the right-nested chain of its members in
    the explicit form, a delimited bag in the tagged form, one call in infix."""
    for mode in MODES:
        s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
        for src in ('max x0 max x1 x2', 'max max x2 x0 x1', 'max x1 max x2 max x0 x1'):
            assert s(src) == 'max x0 max x1 x2', (mode, src)
            assert bare.simplify(bare.to_tagged(src.split()), mode=mode) == \
                ['<max>', 'x0', 'x1', 'x2', '</max>']
        assert s('<min> x2 <min> x0 x1 </min> </min>') == '<min> x0 x1 x2 </min>'
        assert bare.simplify('min(x2, min(x1, x0), x1)', mode=mode) == 'min(x0, x1, x2)'
        # a two-member bag is tagged too, as a two-term sum is
        assert bare.simplify(['<max>', 'x1', 'x0', '</max>'], mode=mode) == ['<max>', 'x0', 'x1', '</max>']
    # a bag costs one head whatever its size, as a sum does
    c = lambda t: bare.complexity(t.split())  # noqa: E731
    assert c('max x0 max x1 x2') - c('max x0 x1') == c('x2')
    # the syntactic conversions regroup and expand a bag without simplifying it
    assert bare.to_tagged(['max', 'x2', 'max', 'x0', 'x0']) == ['<max>', 'x2', 'x0', 'x0', '</max>']
    assert bare.to_prefix(['<max>', 'x2', 'x0', 'x1', '</max>']) == ['max', 'x2', 'max', 'x0', 'x1']


@pytest.mark.parametrize('src, want', [
    # absorption by the dual bag, where the other members are no NaN almost everywhere
    ('max x0 min x0 x1', 'x0'), ('min x1 max x0 x1', 'x1'),
    ('max x0 min x0 log x1', 'max x0 min x0 log x1'),
    # numbers: their extreme, -0 below +0 (f64 writes the negative zero), compared with a
    # member of the dual bag too
    ('max 3 max x0 2', 'max 3 x0'), ('max -0.0 0', '0'), ('min -0.0 0', '-0.0'),
    ('max 3 max 2 min 2 x0', '3'), ('max min 2 x0 max 2 3', '3'),
    # the infinities: identity and absorber; a NaN propagates
    ('max float("-inf") x0', 'x0'), ('min float("inf") x0', 'x0'),
    ('max float("inf") x0', 'float("inf")'), ('max float("inf") log x0', 'max float("inf") log x0'),
    ('max float("inf") float("nan")', 'float("nan")'), ('min x0 float("nan")', 'float("nan")'),
])
def test_lattice_laws(bare, src, want):
    assert ' '.join(bare.simplify(src.split(), mode='f64')) == want


@pytest.mark.parametrize('mode', ('real', 'permissive'))
def test_pi_and_e_are_numbers_of_a_lattice_bag(bare, mode):
    """Outside f64, pi and e fold with the literal members, each comparison decided by a
    certified enclosure; f64's ground fold reads them as their floats instead."""
    s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
    assert s('max 3 np.pi') == 'np.pi'
    assert s('min np.e min np.pi 4') == 'np.e'
    assert s('max np.e np.pi') == 'np.pi' and s('min np.pi np.e') == 'np.e'
    assert s('max x0 max 3 np.pi') == s('max x0 np.pi')
    assert s('min x0 max np.pi 4') == s('min x0 4')
    # a rational that close to pi is decided by the enclosure
    assert s('max np.pi 3.1415926535897932') == 'np.pi'
    # (permissive reads a literal at its float, 3.141592653589793, which is below pi)
    assert s('min np.pi 3.1415926535897933') == ('np.pi' if mode == 'real' else '3.141592653589793')
    # the dual bag is absorbed through the same order: min(e, x0) <= e < pi
    assert s('max np.pi min np.e x0') == 'np.pi'


def test_lattice_laws_are_the_judges():
    """The laws the bag applies unconditionally are true over the reals and float64 identities
    bit for bit, a zero's sign and a NaN included (the contract's `max` and `min` and the f64 bar
    agree with the bag's folds): `core`. The absorber and the absorption are refused only at a
    NaN operand, which is why the bag takes them under the no-NaN licence alone."""
    j = lambda lhs, rhs: _contract.judge_rule(lhs.split(), rhs.split())  # noqa: E731
    for lhs, rhs in (('max _0 max _1 _2', 'max max _0 _1 _2'), ('min _0 _1', 'min _1 _0'),
                     ('max _0 _0', '_0'), ('max float("-inf") _0', '_0'), ('max -0.0 0', '0'),
                     ('min 0 -0.0', '-0.0'), ('max _0 float("nan")', 'float("nan")')):
        assert j(lhs, rhs)['tier'] == 'core', (lhs, rhs)
    for lhs, rhs in (('max float("inf") _0', 'float("inf")'), ('max _0 min _0 _1', '_0')):
        v = j(lhs, rhs)
        assert v['tier'] == 'reject' and any(np.isnan(x) for x in v['points'][0].values()), v


def test_rules_match_any_sub_bag(bare):
    cfg = yaml.safe_load(open(ACJ2_4))['operators']
    rule = (('max', 'abs', '_0', '_0'), ('abs', '_0'))
    e = SimpliPyEngine(operators=cfg, rules=[rule], rules_real=[], rules_permissive=[])
    s = lambda t: ' '.join(e.simplify(t.split(), mode='f64'))  # noqa: E731
    assert s('max max abs x0 x1 x0') == 'max x1 abs x0'
    assert s('max x2 max abs x0 max x1 x0') == 'max x1 max x2 abs x0'
    assert s('sin max x1 max x0 abs x0') == 'sin max x1 abs x0'


def test_masking_reads_lattice_bags(bare):
    from simplipy import masking
    sites = [(v, r) for _, v, r in masking.literal_sites(['<max>', '2', 'x0', '</max>'], bare)]
    assert sites == [('2', masking.Role.VALUE)]
    assert masking.mask(['<max>', '2', 'x0', '</max>'], bare, masking.mask_fittable) == \
        ['<max>', '<constant>', 'x0', '</max>']


def test_the_search_moves_inside_atan2(bare):
    """The search rewrites atan2's arguments as it rewrites any other head's; in `f64` it
    makes no move whose part vanishes with a sign of its own, which atan2 reads."""
    s = lambda t, m='f64': ' '.join(bare.simplify(t.split(), mode=m))  # noqa: E731
    for mode in MODES:
        assert s('atan2 - pow + x0 1 2 + pow x0 2 1 x1', mode) == 'atan2 * 2 x0 x1'
        assert s('atan2 x1 - pow + x0 1 2 + pow x0 2 1', mode) == 'atan2 x1 * 2 x0'
        assert s('atan2 + * x0 x1 * x0 x2 x3', mode) == 'atan2 * x0 + x1 x2 x3'
    # `x0*x1 - x0*|x1|` is +0 for every x1 >= 0; factored, `x0*(x1 - |x1|)` is -0 where x0 < 0,
    # and atan2(-0, -1) is -pi: f64 keeps the sum, which real and permissive factor
    assert s('atan2 - * x0 x1 * x0 abs x1 -1') == 'atan2 - * x0 x1 * x0 abs x1 -1'
    assert s('atan2 - * x0 x1 * x0 abs x1 -1', 'real') == 'atan2 * x0 - x1 abs x1 -1'
    # a move beside such an argument is taken
    assert s('atan2 * x2 - pow + x0 1 2 pow x0 2 - x1 abs x1') == 'atan2 * x2 + * 2 x0 1 - x1 abs x1'


def test_f64_ground_folds_keep_float64s_zero_sign(bare):
    """A negated or scaled odd function of 0 and a written product of ground zeros are what
    float64 computes, the sign of the zero included."""
    f64 = lambda t: ' '.join(bare.simplify(t.split(), mode='f64'))  # noqa: E731
    for head in ('sin', 'tan', 'asin', 'atan', 'sinh', 'tanh', 'asinh', 'atanh', 'erf'):
        assert f64(f'neg {head} 0') == '-0.0', head
        assert f64(f'* -2 {head} 0') == '-0.0', head
        assert f64(f'neg {head} -0.0') == '0', head
    assert f64('neg rootn 0 3') == '-0.0' and f64('neg ceil -0.5') == '0'
    assert f64('* 0 sin -1') == '-0.0' and f64('* sinh -0.0 min 0 2') == '-0.0'
    assert f64('* ceil -0.0 neg 0') == '0' and f64('neg * atanh -0.0 neg 0') == '-0.0'
    assert f64('max x2 neg sin 0') == 'max -0.0 x2'
    # real reads one zero, and a zero the search derives beside a variable keeps it
    assert ' '.join(bare.simplify('neg sin 0'.split(), mode='real')) == 'sin 0'
    assert f64('* x1 0') == '0'


def test_f64_ground_fold_is_the_evaluator_at_zero_and_small_inputs(bare):
    """Every unary and binary head at +-0 and small ground inputs, bare, negated, scaled and in
    products of zeros: the f64 answer evaluates bit for bit as the input does. (A negative root
    index is excluded: `rootn(x, -n)` is written `1/rootn(x, n)`, which float64 can round 1 ulp
    apart.)"""
    import itertools
    import struct
    arity = dict(bare.operator_arity)
    leaves = ['0', '-0.0', '1', '-1', '0.5', '-0.5', '2', '-2', '3', '-3']
    grounds = [[h, a] for h, n in arity.items() if n == 1 for a in leaves]
    grounds += [[h, a, b] for h, n in arity.items() if n == 2 for a, b in itertools.product(leaves, leaves)
                if not (h == 'rootn' and b.startswith('-') and b != '-0.0')]

    def ev(t):
        return float(np.asarray(bare._core.evaluate_batch(t, ['x0'], [0.0], 1, []))[0])

    def bits(v):
        return 'nan' if v != v else struct.pack('<d', v)
    zeros = [g for g in grounds if ev(g) == 0.0]
    srcs = [c + g for g in grounds for c in ([], ['neg'], ['*', '-1'], ['*', '2'], ['*', '-2'])]
    srcs += [p + g + h for g in zeros[::7] for h in zeros[::11] for p in (['*'], ['neg', '*'])]
    bad = [(s, a) for s in srcs if bits(ev(s)) != bits(ev(a := bare.simplify(s, mode='f64')))]
    assert not bad, (len(bad), bad[:5])


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
    """Rules with the nine heads are promoted to the sorts they certify at, the contract atom
    bar reading their contract values, and a rule false only where a step head's argument
    crosses an integer at no atom (sinh(x) + 1/2 = k) is killed at that preimage."""
    from simplipy.promotion import promote
    rules = [(lhs.split(), rhs.split()) for lhs, rhs in (
        ('floor floor ?0', 'floor ?0'), ('mod ?0 1', '- ?0 floor ?0'),
        ('atan2 neg ?0 1', 'neg atan2 ?0 1'), ('erf neg ?0', 'neg erf ?0'),
        ('besselj 1 neg ?0', 'neg besselj 1 ?0'), ('besselj 0.5 neg ?0', 'besselj 0.5 ?0'),
        ('- ceil + sinh ?0 0.5 floor + sinh ?0 0.5', '1'))]
    kept, report = promote(rules, bare, rule_mode=mode, run_positive_controls=False)
    kept = {' '.join(a) + ' -> ' + ' '.join(b) for a, b in kept}
    for rule in ('floor floor _0 -> floor _0', 'mod _0 1 -> - _0 floor _0',
                 'atan2 neg _0 1 -> neg atan2 _0 1', 'erf neg _0 -> neg erf _0',
                 'besselj 1 neg _0 -> neg besselj 1 _0'):
        assert rule in kept, (rule, kept)
    assert not any(r.startswith('- ceil') or r.startswith('besselj 0.5') for r in kept), kept
    assert any('PREIMAGE' in str(k[2]) for k in report['killed_q']), report['killed_q']


def test_a_constant_collapse_through_a_band_survives_promotion(bare):
    """The ground tier probes a constant collapse at the finite atoms; where a contract kernel
    cannot decide a probe (`erf(1e20)` inside the band of its asymptote) the probe is skipped,
    not read as a failure to evaluate: the licensed rule used to be dropped there."""
    from simplipy.promotion import promote
    rules = [(lhs.split(), ['<constant>']) for lhs in
             ('acos erf <constant>', 'floor <constant>', 'harmonic exp <constant>',
              'acos erf sinh <constant>', 'harmonic <constant>')]
    kept, report = promote(rules, bare, rule_mode='real', run_positive_controls=False)
    kept = {' '.join(a) for a, _ in kept}
    assert {'acos erf <constant>', 'floor <constant>', 'harmonic exp <constant>',
            'acos erf sinh <constant>'} <= kept
    # harmonic(c) has no finite witness at its poles: refused, as before
    assert 'harmonic <constant>' not in kept and report['killed_ground']


# --------------------------------------------------------------------------- besselj, complete


@pytest.mark.parametrize('mode', MODES)
def test_besselj_reflection_and_parity_in_every_mode(bare, mode):
    s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
    assert s('besselj -3 x0') == 'neg besselj 3 x0'          # J_-n = (-1)^n J_n
    assert s('besselj -2 x0') == 'besselj 2 x0'
    assert s('besselj 2 neg x0') == 'besselj 2 x0'           # an even order is even
    assert s('besselj 0 abs x0') == 'besselj 0 x0'
    assert s('besselj 2 sin neg x0') == 'besselj 2 sin x0'
    assert s('besselj 0 - x1 x0') == s('besselj 0 - x0 x1')
    assert s('besselj 3 neg x0') == 'neg besselj 3 x0'       # an odd order carries a sign out
    assert s('besselj 3 * -2 x0') == 'neg besselj 3 * 2 x0'
    assert s('besselj -1 neg x0') == 'besselj 1 x0'
    assert s('besselj 0.5 neg x0') == 'besselj / 1 2 neg x0'  # a non-integer order has no parity
    assert s('besselj 1 - x1 x0') == 'besselj 1 - x1 x0'     # a sum alone keeps its orientation
    assert s('neg besselj 1 - x0 x1') == 'besselj 1 - x1 x0'  # ... and a sign trades into it
    assert s('besselj 2 - x1 x0') == s('besselj 2 - x0 x1')
    assert s('besselj 1 -2') == 'besselj 1 -2'               # a constant keeps its sign
    for t in ('besselj -3 x0', 'besselj 3 * -2 x0', 'besselj 0 - x1 x0'):
        once = bare.simplify(t.split(), mode=mode)
        assert bare.simplify(once, mode=mode) == once


@pytest.mark.parametrize('mode', MODES)
def test_besselj_of_literals_at_zero_folds_exactly(bare, mode):
    s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
    assert s('besselj 0 0') == '1'
    assert s('besselj 2.5 0') == '0' and s('besselj -2 0') == '0'
    assert s('besselj -1 0') == ('-0.0' if mode == 'f64' else '0')
    # the pole's one-sided limit, the sign of 1/Gamma(nu + 1), as scipy has it
    assert s('besselj -0.5 0') == 'float("inf")'
    assert s('besselj -1.5 0') == 'float("-inf")'
    assert s('besselj -2.5 0') == 'float("inf")'
    # an infinite order: 0 at x >= 0, NaN below (scipy's cells)
    assert s('besselj atanh 1 2') == '0' and s('besselj atanh 1 -2') == 'float("nan")'


def test_besselj_identities_hold_bit_for_bit_in_float64():
    from simplipy._core import f64_bar
    ext = _contract.ARITY
    for lhs, rhs in (('besselj -3 x0', 'neg besselj 3 x0'), ('besselj -2 x0', 'besselj 2 x0'),
                     ('besselj 2 neg x0', 'besselj 2 x0'), ('besselj 0 abs x0', 'besselj 0 x0'),
                     ('besselj 0 - x0 x1', 'besselj 0 - x1 x0'), ('besselj 3 neg x0', 'neg besselj 3 x0'),
                     ('besselj 1 * -2 x0', 'neg besselj 1 * 2 x0'), ('besselj -1 neg x0', 'besselj 1 x0')):
        names = sorted({t for t in (lhs + ' ' + rhs).split() if t in ('x0', 'x1')})
        assert f64_bar(lhs.split(), rhs.split(), names, ext)[0], (lhs, rhs)
    # the bar can fail: no parity at a non-integer order, and an odd order's sum keeps its zero
    assert not f64_bar('besselj 0.5 neg x0'.split(), 'besselj 0.5 x0'.split(), ['x0'], ext)[0]
    assert not f64_bar('besselj 1 - x0 x1'.split(), 'neg besselj 1 - x1 x0'.split(), ['x0', 'x1'],
                       ext)[0]


def test_besselj_is_judged_by_the_contract():
    j = lambda lhs, rhs: _contract.judge_rule(lhs.split(), rhs.split())  # noqa: E731
    v = j('besselj 1 neg _0', 'neg besselj 1 _0')
    assert v['verdict'] == 'CERTIFIED' and v['tier'] == 'core', v
    assert j('besselj -0.5 0', 'float("inf")')['tier'] == 'core'
    # true over the reals, a few ulp apart in float64
    assert j('besselj 0.5 _0', '* rootn / 2 * np.pi _0 2 / sin _0 1')['tier'] == 'real'
    assert j('besselj 0.5 neg _0', 'besselj 0.5 _0')['verdict'] == 'KILL'
    assert j('besselj _0 0', '0')['verdict'] == 'KILL'          # J_0(0) = 1
    # a <constant> is witnessed by the contract
    assert j('* <constant> besselj 0 _0', '* <constant> besselj 0 neg _0')['verdict'] == 'CERTIFIED'
    # the cells
    c = _contract.c_besselj
    mpf = _contract.mpf
    assert c(mpf(-1.5), mpf(0)) == _contract.NINF and c(mpf(-0.5), mpf(0)) == _contract.PINF
    assert _contract.misnan(c(mpf(0.5), mpf(-1))) and _contract.misnan(c(mpf(1), _contract.PINF))
    assert c(mpf(3), mpf(-2)) == -c(mpf(3), mpf(2)) == c(mpf(-3), mpf(2))
    with pytest.raises(_contract.Unresolved):
        c(mpf(2000), mpf(1))


def test_monitor_and_oracle_read_besselj():
    from mpmath import mpf
    from simplipy.promotion._hp_equiv import equiv, evaluate
    assert 'besselj' not in _monitor.unknown_tokens(['besselj', '1', 'x0'])
    neg_zero = _monitor._NegZero(0)
    assert _monitor._zero_signbit('besselj', [mpf(1), neg_zero]) is True      # J_1(-0) = -0
    assert _monitor._zero_signbit('besselj', [mpf(2), neg_zero]) is False
    assert _monitor._zero_signbit('besselj', [mpf(-1), mpf(0)]) is True       # J_-1(+0) = -0
    j = lambda t: evaluate(t.split(), {'x0': mpf(2)}, [])  # noqa: E731
    assert abs(j('besselj 1 neg x0') + j('besselj 1 x0')) < 1e-14
    assert equiv('besselj 1 neg _0'.split(), 'neg besselj 1 _0'.split())['ok']
    assert not equiv('besselj 2 neg _0'.split(), 'neg besselj 2 _0'.split())['ok']


@pytest.mark.parametrize('mode', MODES)
def test_an_odd_order_is_an_odd_head(bare, mode):
    """besselj of an odd order places its sign and orients its sum as an odd function does: each
    shape gives sin's answer, with sin spelled besselj(1, .)."""
    shapes = ('neg H - x0 x1', '* -3 H - x0 x1', '* x2 H neg - x0 x1', 'pow H - x1 x0 2', 'pow H neg x0 3',
              'abs H neg - x0 x1', '+ H - x0 x1 H - x1 x0', '+ H -2 H 2', 'neg H * -2 - x0 x1',
              '* -1 H - abs x1 x1', 'cos H neg x0')
    for shape in shapes:
        want = ' '.join(bare.simplify(shape.replace('H', 'sin').split(), mode=mode))
        got = bare.simplify(shape.replace('H', 'besselj 1').split(), mode=mode)
        assert ' '.join(got) == want.replace('sin', 'besselj 1'), (shape, got, want)
        assert bare.simplify(got, mode=mode) == got


def test_real_folds_a_ground_value_a_certified_enclosure_decides(bare):
    """In real, floor, ceil, mod, besselj at 0 and atan2 on the positive axis fold a ground
    argument to the exact number its enclosure decides; an undecided enclosure stays as written."""
    s = lambda t: ' '.join(bare.simplify(t.split(), mode='real'))  # noqa: E731
    assert s('floor tanh 1') == '0' and s('ceil tanh 1') == '1' and s('floor asinh asin -1') == '-2'
    assert s('ceil cos 0') == '1' and s('ceil tanh 40') == '1'
    assert s('mod 1 sinh 1') == '1' and s('mod -1 tan -1') == '-1' and s('mod 0 sin 1') == '0'
    assert s('mod cosh 1 1') == s('- cosh 1 1')
    assert s('besselj sin 1 0') == '0' and s('besselj cosh 1 0') == '0'
    assert s('atan2 0 cosh 1') == '0' and s('atan2 1 float("inf")') == '0'
    assert s('floor atan2 1 0') == '1' and s('ceil atan2 -1 0') == '-1' and s('ceil besselj 0 1') == '1'
    assert s('erf float("-inf")') == '-1'
    # an infinite divisor (numpy's cells, D7): the dividend where it is zero or has its sign
    assert s('mod -1 log 0') == '-1' and s('mod 0 log 0') == '0' and s('mod 1 float("inf")') == '1'
    # fail closed: float64 and the reals disagree inside these enclosures (f64: 1 and 0)
    assert s('floor tanh 40') == 'floor tanh 40' and s('ceil exp -800') == 'ceil exp -800'
    # f64 keeps its own ground fold, with float64's zero sign
    f64 = lambda t: ' '.join(bare.simplify(t.split(), mode='f64'))  # noqa: E731
    assert f64('floor tanh 40') == '1' and f64('mod 0 log 0') == '-0.0'
    assert f64('besselj cosh 1 0') == '-0.0'


@pytest.mark.parametrize('mode', ('real', 'permissive'))
def test_ground_members_of_a_lattice_bag_are_ordered_by_enclosure(bare, mode):
    s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
    assert s('max sin 1 0.5') == 'sin 1' and s('min sin 1 cos 1') == 'cos 1'
    assert s('max 1 tanh 1') == '1' and s('min -1 asinh 1') == '-1'
    assert s('max x0 max sin 1 0.5') == s('max x0 sin 1')
    assert s('max sin 1 min 0.5 x0') == 'sin 1'
    # overlapping enclosures leave both members: no tie-breaking (permissive then folds the
    # bag to its float)
    if mode == 'real':
        assert s('max sin 1 + 1 neg 0.1585290151921035') == 'max 0.8414709848078965 sin 1'
    # f64 reads the bag through its ground fold, unchanged
    assert ' '.join(bare.simplify('max sin 1 0.5'.split(), mode='f64')) == 'max / 1 2 sin 1'


@pytest.mark.parametrize('mode', MODES)
def test_ground_nan_and_inf_the_domain_decides(bare, mode):
    """harmonic is NaN at -inf and at a negative integer (D6) and enclosed between its poles;
    mod's infinite cells are numpy's (D7); besselj of an integer order is bounded by 1/sqrt(2)
    and signed on 0 < |x| <= 2: each decides a ground NaN or infinity."""
    s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
    nan, inf = 'float("nan")', 'float("inf")'
    for src in ('harmonic atanh ceil -1', 'harmonic log 0', 'harmonic -2', 'acos harmonic sinh -1',
                'log harmonic asin -1', 'mod log 0 1', 'mod atanh 1 0', 'acosh besselj -1 1',
                'acosh besselj -1 -1', 'log besselj -1 1'):
        assert s(src) == nan, src
    assert s('mod 1 atanh -1') == 'float("-inf")'
    assert s('mod -1 atanh 1') == inf


def test_f64_besselj_at_zero_is_the_kernel_with_its_zero_sign(bare):
    """`besselj(nu, 0)` at a literal order is the deployed kernel's value in f64, the sign of a
    zero included, and the inverse of a zero the infinity of that sign, at every order swept."""
    import struct

    from simplipy._core import kernel_binary_scalar
    orders = [str(n) for n in range(-12, 13)] + [f'{n}.5' for n in range(-6, 12)]
    orders += [f'{n}.{q}' for n in range(0, 5) for q in (25, 75, 1, 9)] + ['1e-300', '123.456', '999.5']

    def bits(v):
        return 'nan' if v != v else struct.pack('<d', v)

    def ev(t):
        return float(np.asarray(bare._core.evaluate_batch(t, ['x0'], [0.0], 1, []))[0])
    for nu in orders:
        want = kernel_binary_scalar('besselj', float(nu), 0.0)
        got = bare.simplify(['besselj', nu, '0'], mode='f64')
        assert bits(ev(got)) == bits(want), (nu, got, want)
        if want == 0:
            inv = bare.simplify(['inv', 'besselj', nu, '0'], mode='f64')
            assert bits(ev(inv)) == bits(np.copysign(np.inf, want)), (nu, inv)


@pytest.mark.parametrize('mode', ('f64', 'real'))
def test_match_time_certificates_read_the_new_heads(bare, mode):
    """The `!` (finite a.e.) and `$` (finite and nonzero a.e.) certificates read each 0.15 head by
    its own argument: a cancellation or a zero product is licensed where the head is finite (and
    nonzero) off a null set, and refused where it vanishes, is NaN or has a pole on a region."""
    s = lambda t: ' '.join(bare.simplify(t.split(), mode=mode))  # noqa: E731
    for head in ('sin x2', 'erf x2', 'besselj 0 x2', 'besselj 1 x2', 'harmonic exp x2', 'max x2 x3',
                 'min x2 x3', 'mod x2 x3', 'atan2 x2 x3', 'floor + exp x2 2'):
        assert s(f'* x1 / {head} {head}') == 'x1', head
    for head in ('erf x1', 'tan x1', 'mod x1 x2', 'harmonic exp x1', 'atan2 x1 x2', 'max x1 x2'):
        assert s(f'* {head} 0') == '0', head
    # a step head vanishes on an interval, a non-integer order of besselj is NaN below 0,
    # `mod` by a literal divisor is 0 on float64's whole ray past 2^52 times it
    for head in ('floor x2', 'ceil x2', 'besselj 1.5 x2', 'besselj -1.5 x2', 'mod x2 2'):
        assert s(f'* x1 / {head} {head}') != 'x1', head
    for head in ('besselj 1.5 x1', 'besselj -1.5 x1', 'mod x1 floor x2'):
        assert s(f'* {head} 0') != '0', head
    # harmonic's poles at the negative integers are a null set over the reals; float64's
    # harmonic is NaN at every float at or below -2^52, a whole ray
    assert (s('* x1 / harmonic x2 harmonic x2') == 'x1') == (mode == 'real')
    assert (s('* harmonic x1 0') == '0') == (mode == 'real')


def test_f64_square_root_spellings_keep_float64s_zero_signs(bare):
    """float64's `b ** 0.5` is numpy's `sqrt(b)`, -0 at -0, where `rootn(b, 2)` and `b ** -0.5`
    read +0 there: where the base may be a zero with a sign of its own (the written -0, a ground
    -0, a step head on the interval it vanishes on), each spelling's `f64` answer evaluates as the
    input does, under an inverse and `atan2` too."""
    import struct
    bases = ['ceil x0', '* x1 ceil x0', 'floor neg x0', 'max x0 -0.0', '-0.0', 'ceil -0.5', 'sin -0.0', 'x0']
    shapes = ['pow {} 0.5', 'rootn {} 2', 'pow {} -0.5', 'inv pow {} 0.5', 'inv rootn {} 2',
              'atan2 pow {} 0.5 -1', 'atan2 rootn {} 2 -1', 'inv * x1 pow {} 0.5', 'neg inv pow {} 0.5']
    X = np.array([[a, b] for a in (-0.75, -0.25, 0.3, 1.7, 2.5) for b in (-2.0, 3.0)])

    def ev(t):
        return np.asarray(bare._core.evaluate_batch(t, ['x0', 'x1'], X.flatten().tolist(), len(X), []))

    def bits(v):
        return ['nan' if x != x else struct.pack('<d', x) for x in v]
    for b in bases:
        for shape in shapes:
            src = shape.format(b).split()
            got = bare.simplify(src, mode='f64')
            want, have = ev(src), ev(got)
            # the zeros and infinities bit for bit (their signs); other values to rounding
            exact = (want == 0) | np.isinf(want) | np.isnan(want)
            assert bits(have[exact]) == bits(want[exact]), (src, got)
            assert np.allclose(have[~exact], want[~exact], rtol=1e-15, atol=0), (src, got)
    f64 = lambda t: ' '.join(bare.simplify(t.split(), mode='f64'))  # noqa: E731
    assert f64('pow -0.0 0.5') == '-0.0' and f64('rootn -0.0 2') == '0'
    assert f64('pow x0 0.5') == 'rootn x0 2'
