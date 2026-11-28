"""One effective operator schema. The core reads and writes the operators of its serialization
language (`+`, `-`, `*`, `/`, `neg`, `inv`, `pow`, `rootn`) whatever the config declares: an engine
over `+`, `*` and `sin` answers `x0*x0` with `pow(x0, 2)`. Every helper that walks an expression
reads the core's operators beside the config's, so it reads that answer as the core wrote it."""
import pytest

from simplipy import SimpliPyEngine

_OPERATORS = {
    '+': {'realization': '+', 'alias': [], 'arity': 2, 'precedence': 1, 'commutative': True},
    '*': {'realization': '*', 'alias': [], 'arity': 2, 'precedence': 2, 'commutative': True},
    'sin': {'realization': 'np.sin', 'alias': [], 'arity': 1, 'precedence': 3, 'commutative': False},
}


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine(operators=_OPERATORS, rules=[])


def test_the_core_writes_its_own_operators(engine: SimpliPyEngine) -> None:
    assert engine.simplify(['*', 'x0', 'x0']) == ['pow', 'x0', '2']


def test_the_schema_is_the_cores_operators_and_the_configs(engine: SimpliPyEngine) -> None:
    assert engine._effective_arity == {'+': 2, '-': 2, '*': 2, '/': 2, 'neg': 1, 'inv': 1,
                                       'pow': 2, 'rootn': 2, 'sin': 1}
    assert engine.operator_arity == {'+': 2, '*': 2, 'sin': 1}


def test_expression_variables_reads_a_core_operator_as_an_operator(engine: SimpliPyEngine) -> None:
    assert engine.expression_variables(['neg', 'pow', 'x0', '2']) == ['x0']
    assert engine.expression_variables(engine.simplify(['*', 'x1', 'x1'])) == ['x1']


def test_evaluate_constants_folds_under_a_core_operator(engine: SimpliPyEngine) -> None:
    assert engine.evaluate_constants(['+', 'x0', 'pow', '2', '0.5']) \
        == ['+', 'x0', repr(2 ** 0.5)]
    assert engine.evaluate_constants(['*', 'x0', 'neg', 'sin', '1']) \
        == ['*', 'x0', engine.evaluate_constants(['neg', 'sin', '1'])[0]]


def test_resolve_constant_rules_reads_a_core_operator_as_an_operator() -> None:
    engine = SimpliPyEngine(operators=_OPERATORS, rules=[(('pow', '2', '2'), ('<constant>',))])
    assert engine.resolve_constant_rules() == 1
    assert engine.simplification_rules[0][1] != ('<constant>',)
