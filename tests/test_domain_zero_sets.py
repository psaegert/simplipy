"""A zero set is read on the expression's domain.

The certificate of a null zero set (reading A, the distribution licences, orientation) applied the
identity theorem only to expressions analytic on all of R^n, so a sum over `log`, a root or `asin`
was never shown null and kept the sign it was written with under an inverse: `-1/(x1 - log(x1))`
stayed as written in `f64` and `real`, where `permissive` (which needs no certificate) wrote
`1/(log(x1) - x1)`. A NaN region holds no zero, so the certificate reads the zero set where the
expression is defined. A sum that vanishes on a region of its domain (`x1 - |x1|`) keeps its sign.
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


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('expression, expected', [
    ('* -1 inv - x1 log x1', 'inv - log x1 x1'),
    ('+ x2 neg inv - x1 log x1', '+ x2 inv - log x1 x1'),
    ('* -1 inv - x1 rootn x1 2', 'inv - rootn x1 2 x1'),
    ('* -1 inv - x1 asin x1', 'inv - asin x1 x1'),
])
def test_a_sign_moves_through_a_sum_null_on_its_domain(eng, mode, expression, expected):
    assert ' '.join(eng.simplify(expression.split(), mode=mode)) == expected


@pytest.mark.parametrize('mode, expected', [
    ('f64', '/ -1 - x1 abs x1'),
    ('real', '/ -1 - x1 abs x1'),
    ('permissive', 'inv - abs x1 x1'),
])
def test_a_sum_zero_on_a_region_of_its_domain_keeps_its_sign(eng, mode, expected):
    assert ' '.join(eng.simplify('* -1 inv - x1 abs x1'.split(), mode=mode)) == expected
