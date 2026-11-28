"""The canonical form reads like a law, in every mode: a quantity's own powers are never nested.
A product of a sum with a factor that reappears in it, itself or the base of one of its powers, as
a multiplicative factor of a term, is multiplied out, the reappearing factors only, so nested
input reaches the expanded form, and the search, which builds its factorings with the same
constructors, never nests one. A factor that does not reappear inside is taken out; products and
powers of sums stay products, and a quotient by a power keeps the form it comes in.
"""
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path, legacy_from_config


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return legacy_from_config(acj_config_path())


#: (input, strict's answer, permissive's answer)
ROWS = [
    # no nesting of a quantity's own powers: multiplied out, both spellings reach one form
    ('x1*(x1 + 1)', 'x1 + x1^2', 'x1 + x1^2'),
    ('x1**2 + x1', 'x1 + x1^2', 'x1 + x1^2'),
    ('x1*(x1*(0.4*x1 + 2.7) + 1.3)', '1.3*x1 + 2.7*x1^2 + 0.4*x1^3', '1.3*x1 + 2.7*x1^2 + 0.4*x1^3'),
    ('0.4*x1**3 + 2.7*x1**2 + 1.3*x1', '1.3*x1 + 2.7*x1^2 + 0.4*x1^3', '1.3*x1 + 2.7*x1^2 + 0.4*x1^3'),
    ('sin(x1)*(sin(x1) + 1)', 'sin(x1) + sin(x1)^2', 'sin(x1) + sin(x1)^2'),
    ('x1*(x1 + sin(x1))', 'x1^2 + x1*sin(x1)', 'x1^2 + x1*sin(x1)'),
    ('x1**0.5*(x1 + 1)', 'rootn(x1, 2) + x1^(3/2)', 'rootn(x1, 2) + x1^(3/2)'),
    ('x1*x2*(x1 + x2)', 'x1*x2^2 + x1^2*x2', 'x1*x2^2 + x1^2*x2'),
    ('x2*(1.5*x1 + 0.3*x2)', '0.3*x2^2 + 3*x1*x2/2', '0.3*x2^2 + 3*x1*x2/2'),
    ('sin(x1*(2.7*x1 + 1.3))', 'sin(1.3*x1 + 2.7*x1^2)', 'sin(1.3*x1 + 2.7*x1^2)'),
    # a factor that does not reappear is taken out, and stays out beside one that does
    ('x1*(x2 + x3)', 'x1*(x2 + x3)', 'x1*(x2 + x3)'),
    ('x1*x2 + x1*x3', 'x1*(x2 + x3)', 'x1*(x2 + x3)'),
    ('x1*(x2 + x3) + x4', 'x4 + x1*(x2 + x3)', 'x4 + x1*(x2 + x3)'),
    ('x2*(x1 + 1)', 'x2*(x1 + 1)', 'x2*(x1 + 1)'),
    ('x1*(sin(x1) + cos(x1))', 'x1*(cos(x1) + sin(x1))', 'x1*(cos(x1) + sin(x1))'),
    ('exp(x1)*(x1 + 1)', 'exp(x1)*(x1 + 1)', 'exp(x1)*(x1 + 1)'),
    ('log(x2)*(x1 + 1)', 'log(x2)*(x1 + 1)', 'log(x2)*(x1 + 1)'),
    ('x1*(x2*(x1 + 0.5) + 2) - 1', '2*x1 + x2*(x1/2 + x1^2) - 1', '2*x1 + x2*(x1/2 + x1^2) - 1'),
    # a shared number groups, the polynomial expanded inside
    ('2*x1*(x1 + 1)', '2*(x1 + x1^2)', '2*(x1 + x1^2)'),
    ('2*(x1**2 + x1)', '2*(x1 + x1^2)', '2*(x1 + x1^2)'),
    # products and powers of sums stay products, and their expanded spellings stay expanded
    ('(x1 + x2)**2', '(x1 + x2)^2', '(x1 + x2)^2'),
    ('(x1 + 1)*(x1 + 2)', '(x1 + 1)*(x1 + 2)', '(x1 + 1)*(x1 + 2)'),
    ('(x1 - 0.3)**2', '(x1 - 0.3)^2', '(x1 - 0.3)^2'),
    ('x1**2 + 3*x1 + 2', '3*x1 + x1^2 + 2', '3*x1 + x1^2 + 2'),
    ('x1**2 - 0.6*x1 + 0.09', 'x1^2 + 0.09 - 0.6*x1', 'x1^2 + 0.09 - 0.6*x1'),
    ('x1*(x1 + 1)*(x1 + 2)', 'x1*(x1 + 1)*(x1 + 2)', 'x1*(x1 + 1)*(x1 + 2)'),
    ('x1*(x1 + 1)**2', 'x1*(x1 + 1)^2', 'x1*(x1 + 1)^2'),
    # a quotient by a power keeps the form it comes in
    ('(x1 + 1)/x1**2', '(x1 + 1)/x1^2', '1/x1^2 + 1/x1'),
    ('x1 + 1/x1', 'x1 + 1/x1', 'x1 + 1/x1'),
]


@pytest.mark.parametrize('expression, strict, permissive', ROWS)
def test_the_answer_reads_like_a_law(engine: SimpliPyEngine, expression: str, strict: str,
                                     permissive: str) -> None:
    for mode, want in (('strict', strict), ('permissive', permissive)):
        answer = engine.simplify(expression, mode=mode)
        assert answer == want, (mode, expression)
        assert engine.simplify(answer, mode=mode) == answer
        assert engine.complexity(answer, mode=mode) <= engine.complexity(expression, mode=mode)
