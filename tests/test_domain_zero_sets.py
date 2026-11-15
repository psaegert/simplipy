"""A zero set is read on the expression's domain.

The certificate of a null zero set (the distribution licences, the region tests) applied the
identity theorem only to expressions analytic on all of R^n, so a sum over `log`, a root or `asin`
was never shown null. A NaN region holds no zero, so the certificate reads the zero set where the
expression is defined. A sum that vanishes on a region of its domain (`x1 - |x1|`) keeps its sign.

Under an inverse `f64` reads the value float64 gives the pole at an exact zero of the sum, the
infinity of the sign written (`class.pole_orientation`): a sum is `+0` at its zero in either
orientation, so the sign stays out of a sum that may vanish exactly, as `x1 - sqrt(x1)` does at 0
and 1 and `x1 - asin(x1)` at 0. `x1 - log(x1)` is never zero (it is at least 1), which the premise,
failing closed, does not show: it keeps its orientation too. `permissive` reads no value at a pole
and writes the cheaper orientation.
"""
import pytest

from conftest import acj_config_path
from conftest import legacy_from_config  # noqa: E402

CONFIG = acj_config_path()
MODES = ('f64', 'permissive')


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj config not staged')
    return legacy_from_config(CONFIG, modes='all')


@pytest.mark.parametrize('expression, f64, permissive', [
    ('* -1 inv - x1 log x1', '/ -1 - x1 log x1', 'inv - log x1 x1'),
    ('+ x2 neg inv - x1 log x1', '- x2 inv - x1 log x1', '+ x2 inv - log x1 x1'),
    ('* -1 inv - x1 rootn x1 2', '/ -1 - x1 rootn x1 2', 'inv - rootn x1 2 x1'),
    ('* -1 inv - x1 asin x1', '/ -1 - x1 asin x1', 'inv - asin x1 x1'),
])
def test_a_sign_moves_through_a_sum_null_on_its_domain_where_no_pole_is_read(
        eng, expression, f64, permissive):
    for mode, expected in (('f64', f64), ('permissive', permissive)):
        assert ' '.join(eng.simplify(expression.split(), mode=mode)) == expected, mode


@pytest.mark.parametrize('mode, expected', [
    ('f64', '/ -1 - x1 abs x1'),
    ('permissive', 'inv - abs x1 x1'),
])
def test_a_sum_zero_on_a_region_of_its_domain_keeps_its_sign(eng, mode, expected):
    assert ' '.join(eng.simplify('* -1 inv - x1 abs x1'.split(), mode=mode)) == expected
