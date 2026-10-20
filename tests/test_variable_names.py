"""An answer does not depend on what the variables are called.

The certificates (`rust/interval.rs`) read an expression's variables off its tokens. They took a
variable to be a dummy, `x<k>` or `_<k>`, so over `x_1`, `v1` or `alpha` every certificate failed
closed: `2*sin(1/x_1) - sin(1/x_1)` stayed as it was while `2*sin(1/x1) - sin(1/x1)` was
`sin(1/x1)`, in every mode the certificate licenses the collection in. A variable is what the
engine's vocabulary says it is, and the layer spells every other variable as a dummy of its own.
"""
import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
MODES = ('f64', 'real', 'permissive')
SCHEMES = {
    'x_k': lambda k: f'x_{k}',
    'vk': lambda k: f'v{k}',
    'names': lambda k: ('alpha', 'beta', 'gamma')[k - 1],
}
EXPRESSIONS = [
    '- * 2 sin inv x1 sin inv x1',
    '- * 3 tan x1 * 2 tan x1',
    '* x3 inv * x1 - x2 sin x2',
    'pow * x1 - x2 exp x2 -1',
    '/ * sin inv x1 x2 sin inv x1',
    'neg - - x1 sin x2 3',
    'inv neg - exp x2 x1',
]


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj config not staged')
    return SimpliPyEngine.from_config(CONFIG, modes='all')


def rename(tokens, scheme):
    return [scheme(int(t[1:])) if t in ('x1', 'x2', 'x3') else t for t in tokens]


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('scheme', list(SCHEMES))
@pytest.mark.parametrize('expression', EXPRESSIONS)
def test_an_answer_does_not_depend_on_the_variable_names(eng, mode, scheme, expression):
    f = SCHEMES[scheme]
    want = eng.simplify(expression.split(), mode=mode)
    assert eng.simplify(rename(expression.split(), f), mode=mode) == rename(want, f)


@pytest.mark.parametrize('name', ['x1', 'x_1', 'v1', 'alpha', 'x0'])
def test_the_certificates_read_any_variable(eng, name):
    assert eng._core.interval_finite_ae(['sin', name])
    assert eng._core.interval_finite_ae(['inv', name])
    assert not eng._core.interval_finite_ae(['log', name])
    assert eng._core.interval_class(['log', name]) == 'MIXED'
