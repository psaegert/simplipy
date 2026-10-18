"""The sign an odd function's literal multiple carries, and the joined reciprocals of a lossy answer.

An odd function or odd root of a literal multiple owns an adjacent sign as an odd function of a literal does
(`-1 * sin(2 x)` is `sin(-2 x)`), and a product in a sign-blind context gives each of its sites the cheaper spelling. So
two spellings of one value give one answer. In a lossy context a joined reciprocal keeps no sign of its own, so its
answer reads back as itself. Every answer is simplified again.
"""
import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
MODES = ('f64', 'real', 'permissive')


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj config not staged')
    return SimpliPyEngine.from_config(CONFIG, modes='all')


def answer(eng, src, mode, effort):
    a = list(eng.simplify(src.split(), mode=mode, effort=effort))
    assert list(eng.simplify(a, mode=mode, effort=effort)) == a
    return a


ONE_VALUE = [
    # the sign of an odd function or odd root of a literal multiple, a sum's orientation priced with it
    ('- x1 sin * 2 x2', '+ x1 sin * -2 x2'),
    ('* 0.5 pow atan - 2.89 rootn * -2.6 + x2 x3 5 5', '* 0.5 pow atan + 2.89 rootn * 2.6 + x2 x3 5 5'),
    ('* -0.5 pow atan - rootn * -2.6 + x2 x3 5 2.89 5', '* 0.5 pow atan + 2.89 rootn * 2.6 + x2 x3 5 5'),
    # a sign-blind context: the site's cheaper spelling, whatever sign it leaves
    ('abs * 3 pow atan - x2 0.66 5', 'abs * 3 pow atan - 0.66 x2 5'),
]


@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('a, b', ONE_VALUE)
def test_one_value_one_answer(eng, mode, effort, a, b):
    assert answer(eng, a, mode, effort) == answer(eng, b, mode, effort)


# A joined reciprocal keeps no sign of its own: the lossy answer reads back as itself.
@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('src', [
    '/ - x15 + / x17 * * x9 x3 - / x3 * acos x15 sin x17 x9 x15 x9',
    '/ / / x8 x8 * x8 x8 - * / tan x8 * x8 x8 log * x8 x8 x8',
    '* * / x3 / * x4 / - * x1 x6 rootn x11 2 / x5 x12 x7 / - x15 x5 x10 x9',
    '+ x13 atan / / x13 + x4 - rootn x12 4 + * x6 x12 x4 - / x4 x6 x6',
])
def test_a_joined_reciprocal_reads_back_as_itself(eng, effort, src):
    answer(eng, src, 'permissive', effort)
