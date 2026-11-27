"""Pre-audit reinforcement of the canon and simplify (canon angle, 2026-11-27, frozen at 3a6ed3a).

Each class here was probed by differential fuzzing against float64 and an independent mpmath
oracle; the tests pin what holds, with fixed seeds, on the constructors alone (acj2-4's operators,
no rules), so they are fast and need no staged asset:

* ``confine.sibling`` (FX-76) is sound: a member confines only the base it is NaN below, read only
  through the operations that propagate NaN, never through an exponent or ``besselk``'s order, never
  a base that is ``+0`` on a region where the merge would cancel exponents (``t^(3/2)*t^(-1/2)``
  fills a NaN there), never a base that is ``-0`` on a region, and never the merge's own confiner.
* The confinement never leaks between calls: no decision taken under one is remembered
  (``memo_free``), so an answer does not depend on what the engine simplified before.
* A seeded fuzz over roots, fractional powers, ``log``/``acosh`` confiners, zeros, signs, poles and
  nested sums and products: every strict answer keeps float64's kind at random points of the input
  space, every answer is a fixpoint, and no answer prices above its input in strict.
* Known defects found by the audit are pinned as strict ``xfail`` tests, so a fix flips them
  (canon-1 to canon-8 in the audit report): strict reads the members of a product one by one, so a
  fully numeric factor loses the value strict gives it alone; permissive's complexity reads a fully
  numeric part as written while simplify reads its value first; (-0.0)^3 stays unfolded; a negative
  power of a zero product re-reads to another state; a negative half power of a sum that is -0 on a
  region turns +inf into -inf; the infix printer writes pow(-0.0, t) as -0.0^t; an infinity a fold
  makes absorbs a numeric factor only on a second call; a negated sum inside max prices one sign
  above its input.
"""
import os
import random
import warnings

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARS = ['x1', 'x2', 'x3']


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')) as f:
        ops = yaml.safe_load(f)['operators']
    for name, arity in {'besselk': 2}.items():
        ops.setdefault(name, {'realization': f'simplipy.operators.{name}', 'alias': [],
                              'arity': arity, 'commutative': False})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


def values(engine: SimpliPyEngine, expression: str, points: np.ndarray) -> np.ndarray:
    f = engine.as_callable(expression, variables=VARS)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = f(*[points[:, i].copy() for i in range(3)])
    v = np.asarray(v, dtype=float)
    return np.full(len(points), float(v)) if v.ndim == 0 else v


def kinds(v: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(v), 0, np.where(np.isinf(v), np.sign(v) * 2, 1))


def grid(seed: int, n: int = 64) -> np.ndarray:
    """Random points of the input space, none of them special: every variable on both sides of 0
    and of 1, at several magnitudes."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-4, 4, size=(n, 3))
    pts[: n // 4] = rng.choice([-1, 1], size=(n // 4, 3)) * 10 ** rng.uniform(-3, 2, size=(n // 4, 3))
    return pts


def assert_strict_keeps_kinds(engine: SimpliPyEngine, expression: str, answer: str,
                              points: np.ndarray) -> None:
    """Strict's answer keeps float64's kind of the input at every point (a region-level property:
    the points are random, so none of them is an isolated point of the input)."""
    vi, vo = values(engine, expression, points), values(engine, answer, points)
    bad = np.nonzero(kinds(vi) != kinds(vo))[0]
    assert not len(bad), (f'{expression} -> {answer}: kind changes at {points[bad[0]].tolist()} '
                          f'({vi[bad[0]]} against {vo[bad[0]]})')
    both = np.isfinite(vi) & np.isfinite(vo)
    with np.errstate(all='ignore'):
        close = np.abs(vi - vo) <= 1e-9 * np.maximum(np.abs(vi), np.abs(vo)) + 1e-12
    bad = np.nonzero(both & ~close)[0]
    assert not len(bad), (f'{expression} -> {answer}: value changes at {points[bad[0]].tolist()} '
                          f'({vi[bad[0]]} against {vo[bad[0]]})')


def test_the_kind_check_fails_on_a_planted_unsound_merge(engine) -> None:
    """The gate can fail: merging a product's own halves gives x1^3 a value on x1 < 0."""
    with pytest.raises(AssertionError, match='kind changes'):
        assert_strict_keeps_kinds(engine, 'x1^(3/2)*x1^(3/2)', 'x1^3', grid(1))
    with pytest.raises(AssertionError, match='kind changes'):
        assert_strict_keeps_kinds(engine, '(ceil(x1) + ceil(x2))^(-1/2)',
                                  '1/(ceil(x1) + ceil(x2))^(1/2)',
                                  np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]))


# ---------------------------------------------------------------- confinement (FX-76)

# Each sibling below is NaN below a base other than x1, or not NaN wherever x1 < 0 (an exponent,
# besselk's order, a symbolic exponent), so it licenses no merge of x1's powers: the product
# x1^(3/2)*x1^(3/2) keeps its halves (x1^3 has a value on x1 < 0, where the input has none).
NOT_A_CONFINER_OF_X1 = [
    'log(-x1)', 'log(x1^2)', 'log(abs(x1))', 'rootn(x1^2, 2)', 'log(x1 + 1)', 'acosh(x1^2 + 1)',
    'rootn(-x1, 2)', '(x1 - 1)^(1/2)', 'x2^(rootn(x1, 2))', 'besselk(rootn(x1, 2), x2)',
    'rootn(x1, 2)^x2', 'log(exp(x1))', 'exp(x1)^(1/2)', 'rootn(x1, 3)',
]


@pytest.mark.parametrize('sibling', NOT_A_CONFINER_OF_X1)
def test_a_sibling_that_does_not_confine_x1_licenses_no_merge(engine, sibling) -> None:
    expression = f'{sibling} + x1^(3/2)*x1^(3/2)'
    answer = engine.simplify(expression, mode='strict')
    assert 'x1^3' not in answer.replace(' ', ''), answer
    assert_strict_keeps_kinds(engine, expression, answer, grid(1))


@pytest.mark.parametrize('expression', [
    # each product's own halves confine x1, and merging both would take both confiners
    'x1^(1/2)*x1^(1/2) + x1^(3/2)*x1^(3/2)',
    'x1^(1/2)*x1^(1/2) + sin(x1^(3/2)*x1^(3/2))',
    '(x1^(1/2)*x1^(1/2) + 1)*(x1^(3/2)*x1^(3/2) + 1)',
    # a base +0 on a region: an exponent-cancelling merge would fill float64's 0*inf there
    'rootn(floor(x1), 2) + floor(x1)^(3/2)*floor(x1)^(-1/2)',
    'rootn(max(x1, 0), 2) + max(x1, 0)^(3/2)*max(x1, 0)^(-1/2)',
    'rootn(abs(x1) - x1, 2) + (abs(x1) - x1)^(3/2)*(abs(x1) - x1)^(-1/2)',
    'rootn(floor(x1), 2) + floor(x1)^(1/2)*floor(x1)^(-1/2)',
    # a base -0 on a region (ceil on (-1, 0)): its powers part on that zero's sign
    'rootn(ceil(x1), 2) + ceil(x1)^(3/2)*ceil(x1)^(3/2)',
    '1/(rootn(ceil(x1), 2) + ceil(x1)^(3/2)*ceil(x1)^(3/2))',
])
def test_confinement_never_gives_the_input_a_value_on_a_region(engine, expression) -> None:
    answer = engine.simplify(expression, mode='strict')
    assert_strict_keeps_kinds(engine, expression, answer, grid(2))
    assert engine.simplify(answer, mode='strict') == answer


@pytest.mark.parametrize('expression', [
    'x1^(3/2)*(x1^(3/2) + x1)', 'x1^(5/2) + x1^(3/2)*x1^(3/2)', 'log(x1)*x1^(1/2)*x1^(1/2)',
    'max(rootn(x1, 2), x1^(3/2)*x1^(3/2))', 'atan2(rootn(x1, 2), x1^(3/2)*x1^(3/2))',
    'mod(x1^(3/2)*x1^(3/2), rootn(x1, 2) + 1)', 'besselk(x2, rootn(x1, 2)) + x1^(3/2)*x1^(3/2)',
    '0*rootn(x1, 2) + x1^(3/2)*x1^(3/2)', 'acosh(x1) + x1^(1/3)*x1^(2/3)',
    'rootn(x1, 2) + (x1^(3/2))^(2/3)', 'rootn(x1, 2) + rootn(x1, 2)^2',
    'rootn(x1, 2) + (rootn(x1, 3))^(3/2)', 'rootn(x1, 2)*rootn(x1, 3)^2',
    'rootn(x1, 4)*rootn(x1, 3)', 'x1^(3/2)/rootn(x1, 3)^2', 'rootn(x1, 4) + (x1^(1/2))^(-2)',
    'sin(rootn(x1, 2)) + exp(x1^(3/2)*x1^(3/2))', 'x2*rootn(x1, 2) + x2*x1^(3/2)*x1^(3/2)',
])
def test_confined_merges_keep_float64s_kinds(engine, expression) -> None:
    answer = engine.simplify(expression, mode='strict')
    assert_strict_keeps_kinds(engine, expression, answer, grid(3))
    assert engine.simplify(answer, mode='strict') == answer


def test_no_decision_under_a_confinement_leaks_into_another_call(engine) -> None:
    """The memos hold no confined base (``memo_free``): a confined product asked before, after and
    between unconfined ones gets one answer, and so does every unconfined one."""
    confined = ['x1^(5/2) + x1^(3/2)*x1^(3/2)', 'log(x1) + x1^(3/2)*x1^(3/2)',
                'rootn(x1, 2) + x1*x1^(1/2)*x1^(1/2)', 'rootn(x1, 2) + (x1^(3/2))^(2/3)']
    plain = ['x1^(3/2)*x1^(3/2)', 'x1^(3/2)*x1^(3/2) + 1', 'x1*x1^(1/2)*x1^(1/2)',
             '(x1^(3/2))^(2/3)', 'sin(x1^(3/2)*x1^(3/2))', 'x1^(1/2)*x1^(1/2) - x1']
    order = confined + plain
    for mode in ('strict', 'permissive'):
        first = {e: engine.simplify(e, mode=mode) for e in order}
        for e in reversed(order):
            assert engine.simplify(e, mode=mode) == first[e], (mode, e)
        rng = random.Random(5)
        for _ in range(3):
            shuffled = order[:]
            rng.shuffle(shuffled)
            for e in shuffled:
                assert engine.simplify(e, mode=mode) == first[e], (mode, e)
    # and the plain products stay unmerged in strict, whatever came before
    assert 'x1^3' not in engine.simplify('x1^(3/2)*x1^(3/2)', mode='strict').replace(' ', '')


# ---------------------------------------------------------------- seeded fuzz

NUMS = ['1', '2', '3', '-1', '1/2', '1/3', '3/2', '0.5', '2.7']
FRAC = ['1/2', '3/2', '1/3', '2/3', '5/2', '-1/2', '-3/2', '4/3', '5/6', '1/4']
HEADS = ['log', 'acosh', 'exp', 'abs', 'sin', 'cos', 'atan', 'sinh', 'tanh', 'asinh', 'floor',
         'ceil', 'neg']


def _random_expression(r: random.Random, d: int) -> str:
    """A random expression in which every subexpression holds a variable (no fully numeric part,
    whose strict value is ruling 24's)."""
    if d <= 0 or r.random() < 0.2:
        return r.choice(['x1', 'x1', 'x2', 'x3'])
    k = r.choice(['+', '+', '-', '*', '*', '/', 'powf', 'powi', 'rootn', 'head', 'head', 'head',
                  'max', 'zero', 'share', 'mulout', 'confine', 'confine'])
    sub = lambda: _random_expression(r, d - 1)  # noqa: E731
    if k in '+-*/':
        return f'({sub()} {k} {sub()})'
    if k == 'powf':
        return f'({sub()})^({r.choice(FRAC)})'
    if k == 'powi':
        return f'({sub()})^({r.choice(["2", "3", "-1", "-2"])})'
    if k == 'rootn':
        return f'rootn({sub()}, {r.choice(["2", "3", "4", "-2", "-3"])})'
    if k == 'head':
        h = r.choice(HEADS)
        return f'(-{sub()})' if h == 'neg' else f'{h}({sub()})'
    if k == 'max':
        return f'{r.choice(["max", "min", "atan2"])}({sub()}, {sub()})'
    if k == 'zero':
        return f'(0*{sub()})'
    if k == 'share':
        c = r.choice(['2', '1/3', '3.71649'])
        return f'({c}*{sub()} + {c}*{sub()})'
    if k == 'mulout':
        t = _random_expression(r, d - 2)
        return f'({t}*({t} + {_random_expression(r, d - 2)}))'
    t = r.choice(['x1', 'x2', _random_expression(r, max(d - 3, 0))])
    conf = r.choice([f'({t})^(1/2)', f'log({t})', f'acosh({t})', f'rootn({t}, 4)'])
    a, b = r.choice(FRAC), r.choice(FRAC + ['2', '-1'])
    merge = r.choice([f'({t})^({a})*({t})^({b})', f'(({t})^({a}))^({b})',
                      f'({t})^({a})*rootn({t}, 3)^2', f'({t})^({a})*(({t})^({b}) + {sub()})'])
    wrap = r.choice(['+', '*', 'max', 'atan2'])
    return f'({conf} {wrap} {merge})' if wrap in '+*' else f'{wrap}({conf}, {merge})'


def _fuzz_cases(seed: int, n: int) -> list[str]:
    r = random.Random(seed)
    return [_random_expression(r, r.choice([2, 3, 4])) for _ in range(n)]


@pytest.mark.parametrize('seed', [11, 12, 13])
def test_fuzz_strict_keeps_kinds_and_is_a_fixpoint(engine, seed) -> None:
    points = grid(seed, 48)
    for expression in _fuzz_cases(seed, 120):
        answer = engine.simplify(expression, mode='strict')
        assert engine.simplify(answer, mode='strict') == answer, expression
        assert engine.complexity(answer, mode='strict') <= engine.complexity(expression, mode='strict'), \
            expression
        vi, vo = values(engine, expression, points), values(engine, answer, points)
        bad = np.nonzero(kinds(vi) != kinds(vo))[0]
        assert not len(bad), (f'{expression} -> {answer}: kind changes at '
                              f'{points[bad[0]].tolist()} ({vi[bad[0]]} against {vo[bad[0]]})')


@pytest.mark.parametrize('seed', [21, 22])
def test_fuzz_permissive_is_a_fixpoint_and_never_dearer(engine, seed) -> None:
    for expression in _fuzz_cases(seed, 120):
        answer = engine.simplify(expression, mode='permissive')
        assert engine.simplify(answer, mode='permissive') == answer, expression
        assert engine.complexity(answer, mode='permissive') <= \
            engine.complexity(expression, mode='permissive'), expression


# ---------------------------------------------------------------- known defects (xfail strict)

@pytest.mark.xfail(strict=True, reason='audit canon-1: strict folds the members of a product one by '
                   'one, so a fully numeric part loses the value strict itself gives it '
                   '(sin(np.pi)/sin(np.pi) is 1, x1*(sin(np.pi)/sin(np.pi)) is 0)')
@pytest.mark.parametrize('expression, answer', [
    ('x1*(sin(np.pi)/sin(np.pi))', 'x1'),
    ('2*x1 + x1*(sin(np.pi)/sin(np.pi))', '3*x1'),
    ('x1*(floor(tanh(100))/floor(tanh(100)))', 'x1'),
])
def test_a_fully_numeric_factor_keeps_its_strict_value_inside_a_product(
        engine, expression, answer) -> None:
    assert engine.simplify('sin(np.pi)/sin(np.pi)', mode='strict') == '1'
    assert engine.simplify('floor(tanh(100))/floor(tanh(100))', mode='strict') == '1'
    assert engine.simplify(expression, mode='strict') == answer


@pytest.mark.xfail(strict=True, reason='audit canon-5: a negative fractional power of a sum that is -0 '
                   'on a region is written 1/(sum)^(1/2), and float64 reads pow(-0, 0.5) as -0, so the '
                   'answer is -inf where the input is +inf, on all of (-1, 0)^2')
@pytest.mark.parametrize('expression', ['(ceil(x1) + ceil(x2))^(-1/2)', 'rootn(ceil(x1) + ceil(x2), -2)',
                                        'rootn(0*x1 + 0*x2, -2)'])
def test_a_negative_half_power_keeps_the_sign_of_its_pole_on_a_region(engine, expression) -> None:
    points = np.array([[-0.5, -0.5, 0.0], [-0.25, -0.75, 0.0], [-0.9, -0.1, 0.0]])
    answer = engine.simplify(expression, mode='strict')
    assert np.array_equal(values(engine, expression, points), values(engine, answer, points))


@pytest.mark.xfail(strict=True, reason='audit canon-6: the infix printer writes a power of the literal '
                   '-0.0 without parentheses (-0.0^(x2)), which reads back as -(0.0^x2)')
@pytest.mark.parametrize('expression', ['(0*(-2))^x2', 'rootn((0*(-np.pi))^(0*x3), 2)'])
def test_an_infix_answer_reads_back_to_its_own_value(engine, expression) -> None:
    points = grid(4, 16)
    answer = engine.simplify(expression, mode='strict')
    tokens = engine.simplify(engine.to_prefix(expression), mode='strict')
    assert engine.to_prefix(answer) == list(tokens)
    vi, vo = values(engine, expression, points), values(engine, answer, points)
    assert np.array_equal(kinds(vi), kinds(vo))


@pytest.mark.xfail(strict=True, reason='audit canon-7: an infinity a fold makes (log(0)) absorbs a '
                   'positive fully numeric factor (pi) only on a second call')
@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_an_infinity_absorbs_its_numeric_factors_in_one_call(engine, mode) -> None:
    answer = engine.simplify('x1*log(0)*np.pi', mode=mode)
    assert engine.simplify(answer, mode=mode) == answer


@pytest.mark.xfail(strict=True, reason='audit canon-8: strict answers a negated sum inside max one sign '
                   'dearer than its input (the infix spelling -a - b re-reads to another state)')
def test_a_negated_sum_inside_a_lattice_head_never_prices_above_its_input(engine) -> None:
    expression = 'max(x1, -(min(x1, -0.5) + 1))'
    answer = engine.simplify(expression, mode='strict')
    assert engine.complexity(answer, mode='strict') <= engine.complexity(expression, mode='strict')


@pytest.mark.xfail(strict=True, reason='audit canon-3: the pass folds a ground node only where the '
                   'fold descends, and pow(-0.0, 3) prices as -0.0, so the fully numeric (-0.0)^3 '
                   'is answered unfolded (dearer than its input, and -0.0 on a second call)')
@pytest.mark.parametrize('expression', ['(-0.0)^3', '(0*(-2))^3'])
def test_a_fully_numeric_odd_power_of_a_negative_zero_folds(engine, expression) -> None:
    answer = engine.simplify(expression, mode='strict')
    assert answer == '-0.0'
    assert engine.simplify(answer, mode='strict') == answer


@pytest.mark.xfail(strict=True, reason='audit canon-4: a negative fractional power of a zero product is '
                   'written 1/(0*t)^(1/2), which re-reads to another state (inf/t): the answer is '
                   'not a fixpoint, and in strict prices above its input')
@pytest.mark.parametrize('expression', ['(0*acosh(x1))^(-1/2)', '(0*x3)^(-3/2)'])
def test_a_negative_power_of_a_zero_product_is_a_fixpoint(engine, expression) -> None:
    answer = engine.simplify(expression, mode='strict')
    assert engine.simplify(answer, mode='strict') == answer


@pytest.mark.xfail(strict=True, reason='audit canon-2: permissive complexity reads a fully numeric '
                   'part as written (0*tan(asin(1)) absorbs to 0) while simplify reads its value '
                   'first (NaN), so complexity(simplify(e)) > complexity(e)')
@pytest.mark.parametrize('expression', ['x1 + 0*tan(asin(1))', 'x1*tan(asin(1))/tan(asin(1))',
                                        'x1 + 0/sin(np.pi)'])
def test_permissive_answer_never_prices_above_a_fully_numeric_input(engine, expression) -> None:
    answer = engine.simplify(expression, mode='permissive')
    assert engine.complexity(answer, mode='permissive') <= engine.complexity(expression, mode='permissive')


@pytest.mark.xfail(strict=True, reason='audit canon-9: an odd root and a fractional power of one base '
                   'merge (FX-72) or compose only on a second call, when the root first collects '
                   'with its own powers in the same product, or stands as rootn(t, 2) under a power')
@pytest.mark.parametrize('expression', [
    'rootn(x2, 3)/(x2^(3/4)*rootn(x2, 3)^2*floor(log(x2)))',
    '(min(sin(x1^1.5*rootn(x1, 3)), acosh(x1))^(-1/2))^(1/4)',
])
def test_a_root_merges_with_its_radicands_powers_in_one_call(engine, expression) -> None:
    answer = engine.simplify(expression, mode='strict')
    assert engine.simplify(answer, mode='strict') == answer


@pytest.mark.xfail(strict=True, reason='audit canon-10: permissive multiplies a factor out over a sum '
                   'that holds it (ruling 38, R1) only on a second call, where strict does it in one')
@pytest.mark.parametrize('expression', ['(x1*x2)*((5/2 - x1)^2 + (x1/x2 + x1))',
                                        '(x1*x2)*((x1 - 1)^2 + x1/x2 + x1)'])
def test_permissive_multiplies_out_in_one_call(engine, expression) -> None:
    answer = engine.simplify(expression, mode='permissive')
    assert engine.simplify(answer, mode='permissive') == answer
