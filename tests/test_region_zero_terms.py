"""A sum keeps float64's zero sign where its terms vanish on a region, in `f64`.

float64's sum of zeros is -0 only where every term is -0: `-0 + 0` is `+0`, and so is any
cancellation. `ceil(x1)` is -0 on the whole interval (-1, 0), so `ceil(x1) + 0` and
`2*ceil(x1) - ceil(x1)` are `+0` there and their inverses `+inf`, where `ceil(x1)` and its inverse
are -0 and `-inf`. The sum constructor drops a zero term, and collects like terms of opposite signs
into one, only where what remains cannot be -0 on a region; otherwise the sum keeps a `+0` term
(`ceil(x1) + 0`). Each term below vanishes on a region (or is -0 there) in float64: the answers of
the constructs over it must evaluate bit for bit like the input at points inside that region.

Terms whose negation the canonical form turns around before the sum is built are left out of the
constructs that negate them: a sum that cancels on a region (`ceil(x1) + ceil(x2)` is `1 - 1 = +0`
on (0, 1) x (-2, -1)) and a product holding one (`x2*(x1 - |x1|)`). Their negation is held in one
orientation whatever the zero's sign (see "In `f64` a sum that vanishes on a region keeps its
zero's sign where it is inverted").
"""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')
NAMES = ['x1', 'x2', 'x3']

# Terms that are -0 (or +0, then negated) on a set of positive measure in float64.
TERMS = [
    'ceil x1', 'ceil neg abs x1', '* ceil x1 x2', '/ ceil x1 x2', 'floor ceil x1',
    'neg floor x1', '* -2 floor x1', '/ floor x1 x2', '* floor x1 x2', 'neg max x1 0',
    'neg abs - x1 abs x1', 'neg - x1 abs x1', 'min x1 -0.0', 'max x1 -0.0', 'max neg abs x1 -0.0',
    'mod floor x1 -1', 'mod - abs x1 x1 -3', 'mod * 2 floor x1 x2', 'mod 0 x2',
    'sin ceil x1', 'tan ceil x1', 'sinh ceil x1', 'tanh ceil x1', 'asin ceil x1', 'atan ceil x1',
    'asinh ceil x1', 'atanh ceil x1', 'erf ceil x1', 'harmonic ceil x1', 'besselj 1 ceil x1',
    'besselj 1.5 ceil x1', 'atan2 ceil x1 x2', 'atan2 x2 inv floor x1', 'rootn ceil x1 3',
    'pow ceil x1 3', 'pow ceil x1 0.5', '* ceil x1 ceil x2', 'neg exp inv ceil x1',
]
# ... and the ones whose negation is a sum turned around (module doc).
TURNED = ['+ ceil x1 ceil x2', '* x2 - x1 abs x1', '/ - x1 abs x1 x2']

PLUS_ZERO = ['+ {t} 0']
NEGATING = ['- * 2 {t} {t}', '- {t} * 2 {t}', '- 0 {t}', '- * 3 {t} * 2 {t}', '- + {t} {t} {t}']


@pytest.fixture(scope='module')
def bare():
    """The acj2-4 operators with an empty rule set per mode: the constructors on their own."""
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])


@pytest.fixture(scope='module')
def points():
    """Continuous samples only: a zero met there lies inside a region, never at a single point."""
    rng = np.random.default_rng(3)
    cols = [np.concatenate([rng.uniform(-1, 0, 300), rng.uniform(0, 1, 300), rng.uniform(-3, 3, 300)])
            for _ in NAMES]
    for c in cols:
        rng.shuffle(c)
    return np.stack(cols, axis=1)


def evaluate(eng, tokens, X):
    return np.asarray(eng._core.evaluate_batch(list(tokens), NAMES, X.flatten().tolist(), len(X), []),
                      dtype=float)


def cases(terms, constructs):
    for t in terms:
        for c in constructs:
            for inverted in (False, True):
                yield t, ('inv ' if inverted else '') + c.format(t=t)


@pytest.mark.parametrize('term, expression',
                         list(cases(TERMS + TURNED, PLUS_ZERO)) + list(cases(TERMS, NEGATING)))
def test_f64_answer_keeps_the_zero_sign_on_the_region(bare, points, term, expression):
    region = evaluate(bare, term.split(), points) == 0
    assert region.sum() >= 30, f'{term} vanishes on too few sample points'
    answer = bare.simplify(expression.split(), mode='f64')
    a = evaluate(bare, expression.split(), points)[region]
    b = evaluate(bare, answer, points)[region]
    same = (np.isnan(a) & np.isnan(b)) | (a.view(np.int64) == b.view(np.int64))
    assert same.all(), (f'{" ".join(answer)} at {points[region][~same][0]}: '
                        f'{a[~same][0]!r} vs {b[~same][0]!r}')
    assert bare.simplify(answer, mode='f64') == answer


@pytest.mark.parametrize('expression, expected', [
    ('+ ceil x1 0', '+ ceil x1 0'),
    ('- * 2 ceil x1 ceil x1', '+ ceil x1 0'),
    ('inv - * 2 ceil x1 ceil x1', 'inv + ceil x1 0'),
    ('- 0 floor x1', '- 0 floor x1'),
    ('- floor x1 * 2 floor x1', '- 0 floor x1'),
    ('+ min x1 -0.0 0', '+ min -0.0 x1 0'),
    # the remaining term is never -0 on a region: `-ceil(x1)` is +0 on (-1, 0), `|ceil(x1)|` and
    # `floor(x1)` are never -0 but at a point
    ('- 0 ceil x1', 'neg ceil x1'),
    ('+ abs ceil x1 0', 'abs ceil x1'),
    ('- * 2 floor x1 floor x1', 'floor x1'),
    # `ceil(|x1|)` vanishes only where x1 does
    ('+ ceil abs x1 0', 'ceil abs x1'),
    # a term whose zero set is null: -0 at the one point x1 = -0 (the documented f64 limit)
    ('+ x1 0', 'x1'),
    ('+ sin x1 0', 'sin x1'),
    ('- * 2 sin x1 sin x1', 'sin x1'),
    # a ground zero is its float: `0 - sin(0)` is +0
    ('- 0 sin 0', '0'),
])
def test_f64_spells_the_kept_zero_term(bare, expression, expected):
    assert ' '.join(bare.simplify(expression.split(), mode='f64')) == expected


@pytest.mark.parametrize('mode', ('real', 'permissive'))
@pytest.mark.parametrize('expression, expected', [
    ('+ ceil x1 0', 'ceil x1'),
    ('- * 2 ceil x1 ceil x1', 'ceil x1'),
    ('- 0 floor x1', 'neg floor x1'),
])
def test_real_and_permissive_drop_the_zero_term(bare, mode, expression, expected):
    assert ' '.join(bare.simplify(expression.split(), mode=mode)) == expected
