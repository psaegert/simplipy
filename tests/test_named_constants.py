"""Named constants: ``<constant_NAME>``, NAME alphanumeric, is one unknown finite real per name;
distinct names are independent, and ``<constant>`` stays an anonymous constant per occurrence.

* Simplification reads a name the expression uses more than once as it reads a variable (every
  answer holds for every value; never folded with numbers), in both modes.
* A name used once is exactly a ``<constant>`` and may be absorbed, keeping its name; where
  fitted constants become one, the one keeps a name if any had one, the least.
* Masking's collection of degrees of freedom turns a ground part into one fresh constant only if
  it holds a parameter used nowhere else.
* Code generation, masking and the list of an expression's variables treat a named constant as a
  parameter, not a variable.
* Answers keep the names.
"""
import numpy as np
import pytest

from simplipy import SimpliPyEngine
from simplipy.utils import (explicit_constant_placeholders, is_constant_placeholder,
                            is_named_constant, substitute_constants)
from conftest import acj_config_path, legacy_from_config

MODES = ('strict', 'permissive')


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return legacy_from_config(acj_config_path())


def anonymous(tokens: list[str]) -> list[str]:
    return ['<constant>' if is_named_constant(t) else t for t in tokens]


class TestTokens:
    @pytest.mark.parametrize('token', ['<constant_a>', '<constant_k12>', '<constant_1>', '<constant_X1y>'])
    def test_named(self, token: str) -> None:
        assert is_named_constant(token) and is_constant_placeholder(token)

    @pytest.mark.parametrize('token', ['<constant>', '<constant_>', '<constant_a_b>', '<constant_a-b>',
                                       'constant_a', 'C_0', 'x1'])
    def test_not_named(self, token: str) -> None:
        assert not is_named_constant(token)

    def test_infix_reads_and_writes_a_name(self, engine: SimpliPyEngine) -> None:
        assert engine.to_prefix('<constant_a>*x1 + <constant_b2>') == \
            ['+', '*', '<constant_a>', 'x1', '<constant_b2>']
        assert engine.simplify('x1*<constant_x1>') == '<constant_x1>*x1'


class TestNameUsedMoreThanOnce:
    """Read as a variable: every answer holds for every value."""

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression, answer', [
        ('<constant_a>*x1 + <constant_a>*x2 + x3', 'x3 + <constant_a>*(x1 + x2)'),
        ('<constant_a>*x1 + 2*<constant_a>*x2', '<constant_a>*(x1 + 2*x2)'),
        ('<constant_a>*x1 + sin(<constant_a>)', 'sin(<constant_a>) + <constant_a>*x1'),
        ('<constant_1>*x1 + <constant_1>*x2', '<constant_1>*(x1 + x2)'),
        ('<constant_a> - <constant_a>', '0'),
        ('<constant_a>*x1 - <constant_a>*x1', '0'),
    ])
    def test_answers(self, engine: SimpliPyEngine, expression: str, answer: str, mode: str) -> None:
        assert engine.simplify(expression, mode=mode) == answer

    @pytest.mark.parametrize('mode', MODES)
    def test_never_absorbed(self, engine: SimpliPyEngine, mode: str) -> None:
        # absorbing either use would cut the tie between them
        answer = engine.simplify(['+', '*', '<constant_a>', 'x1', 'sin', '<constant_a>'], mode=mode)
        assert answer.count('<constant_a>') == 2 and 'sin' in answer

    def test_never_folded_with_numbers(self, engine: SimpliPyEngine) -> None:
        # the 2 stays a number beside the name
        assert engine.simplify('<constant_a>*x1 + 2*<constant_a>*x2') == '<constant_a>*(x1 + 2*x2)'


class TestNameUsedOnce:
    """Exactly a ``<constant>``: absorbed as one, keeping its name."""

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression, answer', [
        ('sin(<constant_a>)', '<constant_a>'),
        ('sin(<constant_a>)*x1', '<constant_a>*x1'),
        ('3.7*<constant_a>', '<constant_a>'),
        ('<constant_a> + 5', '<constant_a>'),
        ('sin(<constant_a>) + sin(1)', '<constant_a>'),
        # constants that become one keep a name if any had one, the least in string order
        ('<constant_a> + <constant_b>', '<constant_a>'),
        ('<constant_b> + <constant_a>', '<constant_a>'),
        ('<constant_a> + <constant> + x1', 'x1 + <constant_a>'),
        ('exp(<constant_a>)*x1 + log(<constant_b>)', '<constant_a>*x1 + <constant_b>'),
    ])
    def test_answers(self, engine: SimpliPyEngine, expression: str, answer: str, mode: str) -> None:
        assert engine.simplify(expression, mode=mode) == answer

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression', [
        ['sin', '<constant_a>'],
        ['+', '*', '<constant_a>', 'x1', '*', '<constant_b>', 'x2'],
        ['+', '*', 'exp', '<constant_a>', 'x1', 'log', '<constant_b>'],
        ['/', '+', 'x1', '<constant_a>', '*', '<constant_b>', 'x2'],
        ['*', '<constant_a>', 'pow', '+', 'x1', '<constant_b>', '2'],
    ])
    def test_as_a_constant(self, engine: SimpliPyEngine, expression: list[str], mode: str) -> None:
        # the answer is the anonymous expression's answer, written with names
        named = engine.simplify(expression, mode=mode)
        assert anonymous(named) == engine.simplify(anonymous(expression), mode=mode)
        assert all(not is_named_constant(t) or t in expression for t in named)


class TestFixpoint:
    """A name the answer uses once is a fitted constant, so the answer is a fixpoint."""

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression, answer', [
        ('<constant_a>*<constant_a>', '<constant_a>'),
        ('exp(<constant_a>) + exp(<constant_a>)*x1', '<constant_a>*(x1 + 1)'),
    ])
    def test_answers(self, engine: SimpliPyEngine, expression: str, answer: str, mode: str) -> None:
        assert engine.simplify(expression, mode=mode) == answer

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression', [
        '<constant_a>*x1 + <constant_a>*x2 + x3',
        '<constant_a>*x1 - <constant_a>*x2',
        '<constant_a>*<constant_b>*x1 + <constant_a>*x2 + <constant_b>*x3',
        '3.7*<constant_a>*x1 + <constant_a>*x2',
        'exp(<constant_a>) + exp(<constant_a>)*x1',
        '(x1 + <constant_a>)/(x1 + <constant_a>)',
        'log(<constant_a>*x1) + log(<constant_a>)',
    ])
    def test_idempotent(self, engine: SimpliPyEngine, expression: str, mode: str) -> None:
        answer = engine.simplify(expression, mode=mode)
        assert engine.simplify(answer, mode=mode) == answer


class TestNames:
    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression', [
        '<constant_a>*x1 + <constant_a>*x2 + x3',
        '<constant_b>*x1 + <constant_a>*x2',
        '<constant_a>*<constant_b>*x1 + <constant_a>*x2',
        '3.7*<constant_a>*x1 + <constant_a>*x2',
    ])
    def test_renaming(self, engine: SimpliPyEngine, expression: str, mode: str) -> None:
        swap = {'<constant_a>': '<constant_b>', '<constant_b>': '<constant_a>'}
        tokens = engine.to_prefix(expression)
        answer = engine.simplify(tokens, mode=mode)
        renamed = engine.simplify([swap.get(t, t) for t in tokens], mode=mode)
        assert [swap.get(t, t) for t in renamed] == answer

    def test_names_are_kept(self, engine: SimpliPyEngine) -> None:
        answer = engine.simplify('<constant_k1>*x1 + <constant_k1>*x2 + <constant_k2>*x3')
        assert '<constant_k1>' in answer and '<constant_k2>' in answer

    def test_price(self, engine: SimpliPyEngine) -> None:
        # a name is priced as a <constant> at every occurrence, however often it is used
        for mode in MODES:
            assert engine.complexity('<constant_a>*x1 + <constant_a>*x2', mode=mode) == \
                engine.complexity('<constant>*x1 + <constant>*x2', mode=mode)
            assert engine.complexity('<constant_a>*x1', mode=mode) == \
                engine.complexity('<constant>*x1', mode=mode)

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('expression', [
        '<constant_a>*x1 + <constant_a>*x2 + 2*x3 + 2*x4',
        '<constant_a>*<constant_a>',
        'exp(<constant_a>) + exp(<constant_a>)*x1',
        '<constant_a>*<constant_b>*x1 + <constant_a>*x2',
        '(<constant_a>/x4 - <constant_a>/x3)*x1*x2',
    ])
    def test_never_above_the_input(self, engine: SimpliPyEngine, expression: str, mode: str) -> None:
        # an answer that leaves a name in one place reads it as a fitted constant, at the price
        # the name had at each of its uses
        answer = engine.simplify(expression, mode=mode)
        assert engine.complexity(answer, mode=mode) <= engine.complexity(expression, mode=mode)


class TestMasking:
    @pytest.mark.parametrize('policy', ['all', 'fittable'])
    @pytest.mark.parametrize('expression, masked', [
        # a ground part holding a parameter used nowhere else is one fresh constant
        ('<constant_a>*<constant_b>*x1 + <constant_a>*x2', '<constant> * x1 + <constant_a> * x2'),
        ('3.7*<constant_a>*x1 + <constant_a>*x2', '<constant> * x1 + <constant_a> * x2'),
        ('(<constant_a> + 1)*x1 + <constant_a>*x2', '<constant> * x1 + <constant_a> * x2'),
        ('exp(<constant_a>*<constant_b>)*x1 + <constant_a>*x2', '<constant> * x1 + <constant_a> * x2'),
        # each name is used elsewhere: nothing is collected
        ('<constant_a>*<constant_b>*x1 + <constant_a>*x2 + <constant_b>*x3',
         '<constant_a> * <constant_b> * x1 + <constant_a> * x2 + <constant_b> * x3'),
        # nothing to merge
        ('<constant_a>*x1 + <constant_a>*x2', '<constant_a> * x1 + <constant_a> * x2'),
        ('<constant_a>*x1 + <constant_b>*x2', '<constant_a> * x1 + <constant_b> * x2'),
    ])
    def test_degrees_of_freedom(self, engine: SimpliPyEngine, expression: str, masked: str,
                                policy: str) -> None:
        assert engine.mask(expression, policy) == masked

    def test_dialects(self, engine: SimpliPyEngine) -> None:
        tokens = engine.to_prefix('<constant_a>*<constant_b>*x1 + <constant_a>*x2')
        assert engine.mask(tokens) == ['+', '*', '<constant>', 'x1', '*', '<constant_a>', 'x2']
        assert engine.mask(engine.to_tagged(tokens)) == \
            ['<add>', '<mul>', '<constant>', 'x1', '</mul>', '<mul>', '<constant_a>', 'x2', '</mul>', '</add>']

    def test_a_name_is_no_literal(self, engine: SimpliPyEngine) -> None:
        from simplipy import masking
        tokens = ['+', '*', '<constant_a>', 'x1', '2']
        assert [v for _, v, _ in masking.literal_sites(tokens, engine)] == ['2']


class TestParameters:
    def test_not_a_variable(self, engine: SimpliPyEngine) -> None:
        assert engine.expression_variables('<constant_a>*x1 + <constant_a>*x2') == ['x1', 'x2']

    def test_not_folded(self, engine: SimpliPyEngine) -> None:
        assert engine.evaluate_constants('sin(<constant_a>) + sin(1)') == \
            'sin(<constant_a>) + 0.8414709848078965'

    def test_as_callable(self, engine: SimpliPyEngine) -> None:
        f = engine.as_callable('<constant_a>*x1 + <constant_a>*x2')
        assert f.__code__.co_varnames[:f.__code__.co_argcount] == ('x1', 'x2', 'constant_a')
        assert f(np.array([1.0]), np.array([2.0]), 3.0)[0] == 9.0
        g = engine.as_callable(['*', '<constant_a>', 'x1'], ['<constant_a>', 'x1'])
        assert g(2.0, 5.0) == 10.0
        with pytest.raises(ValueError, match='parameters'):
            engine.as_callable(['*', '<constant_a>', 'x1'], ['x1'])

    def test_placeholders(self) -> None:
        expression = ['+', '*', '<constant_a>', 'x1', '+', '*', '<constant>', 'x2', '*', '<constant_a>', 'x3']
        renamed, names = explicit_constant_placeholders(expression)
        assert renamed == ['+', '*', 'C_0', 'x1', '+', '*', 'C_1', 'x2', '*', 'C_0', 'x3']
        assert names == ['C_0', 'C_1']
        assert substitute_constants(expression, [3.0, 4.0]) == \
            ['+', '*', '3.0', 'x1', '+', '*', '4.0', 'x2', '*', '3.0', 'x3']
