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


def test_judge_lane_for_heads_without_a_contract_kernel():
    j = _contract.judge_rule
    ok = j('floor floor _0'.split(), 'floor _0'.split())
    assert ok['verdict'] == 'F64-ONLY' and ok['tier'] == 'f64', ok
    assert j('max _0 _0'.split(), '_0'.split())['tier'] == 'f64'
    # a law of the reals that float64 breaks (x = -1e-17): no tier at all
    bad = j('floor + _0 1'.split(), '+ floor _0 1'.split())
    assert bad['verdict'] == 'F64-ONLY' and bad['tier'] == 'reject', bad
    # a constant needs the contract's witness: refused
    assert j('floor <constant>'.split(), '<constant>'.split())['verdict'] == 'UNSUPPORTED-SHAPE'
    # a head the judge does not know at all stays refused
    assert j('gamma _0'.split(), '_0'.split())['verdict'] == 'UNSUPPORTED-SHAPE'
    assert _contract._tier('F64-ONLY', None) == 'reject'


def test_monitor_and_oracle_refuse_unknown_heads():
    assert 'floor' in _monitor.unknown_tokens(['floor', 'x0'])
    with pytest.raises(_monitor.Unresolved):
        _monitor._zero_signbit('floor', [0.0])
    from simplipy.promotion._hp_equiv import evaluate
    with pytest.raises(ValueError):
        evaluate(['floor', 'x0'], {'x0': 1.5}, [])


def test_a_small_mine_mines_new_head_rules():
    """End to end, in every mode: rules the float64 comparison certifies appear in the f64 and
    permissive files; the real file holds none (the contract cannot judge the heads)."""
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

    for rules in (e.simplification_rules, e.permissive_simplification_rules):
        assert has(rules, 'floor floor #0', 'floor #0'), rules
        assert has(rules, 'ceil floor #0', 'floor #0'), rules
        # max(x, x) = x and the argument order are the engine's: no rule states them
        assert not any(a[0] == 'max' and b[0] == 'max' for a, b in rules), rules
        assert not has(rules, 'max #0 #0', '#0'), rules
    heads = {'floor', 'ceil', 'max'}
    assert not any(heads & set(list(a) + list(b)) for a, b in e.real_simplification_rules)


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


def test_lattice_laws_are_the_judges():
    """The laws the bag applies unconditionally are float64 identities bit for bit by the judge;
    the absorber and the absorption are refused there only at a NaN operand, which is why the
    bag takes them under the no-NaN licence alone."""
    j = lambda lhs, rhs: _contract.judge_rule(lhs.split(), rhs.split())  # noqa: E731
    for lhs, rhs in (('max _0 max _1 _2', 'max max _0 _1 _2'), ('min _0 _1', 'min _1 _0'),
                     ('max _0 _0', '_0'), ('max float("-inf") _0', '_0'), ('max -0.0 0', '0'),
                     ('min 0 -0.0', '-0.0'), ('max _0 float("nan")', 'float("nan")')):
        assert j(lhs, rhs)['tier'] == 'f64', (lhs, rhs)
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


def test_the_mint_refuses_laws_float64_breaks_at_a_jump(bare):
    from simplipy.mining import RuleMiner
    X = RuleMiner(bare)._mining_sample_x(256, 2, np.random.default_rng(0))

    def certified(src, tgt):
        return bare._core.judge_pair(src.split(), tgt.split(), ['x0', 'x1'], X.flatten().tolist(),
                                     X.shape[0], rtol=1e-11, atol=1e-12, rule_mode='default')[0]
    for src, tgt in (('floor + x0 1', '+ floor x0 1'), ('mod + x0 1 1', 'mod x0 1'),
                     ('floor mod x0 1', '0'), ('+ floor x0 1', 'ceil x0')):
        assert not certified(src, tgt), (src, tgt)
    assert certified('floor floor x0', 'floor x0')
