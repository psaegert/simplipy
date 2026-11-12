"""A fully numeric part takes its value first, read by one ground evaluator per mode, and is written
as the cheapest value its mode admits for it.

- strict: float64's double, or the reals' value where float64's kind agrees (the range standard
  included): `floor(tanh(100))` is 0 (float64 floors its 1.0 to 1), `tanh(100)` is 1,
  `exp(sin(np.pi))` is 1, and `tan(asin(1))` keeps float64's 1.633e16, the reals giving it no
  value.
- permissive: the table's value at float resolution; a part without a value is NaN, and absorbs
  the expression around it (`0 * tan(asin(1))` is NaN).

The engine's parse and fold, every rule step on a fully numeric subject, the mint, the judge's
ground check and verify-mine's ground gate read the same evaluator (`_core.ground_rule_verdict`).
Numbers are priced as they are written, so strict writes float64's value of a long product where
that is cheaper, and a literal float64 reads as exactly the double of a short symbolic number
(k*pi/m, k*e/m, k/m, log 2/3/10, sqrt 2/3/5) is read as that number."""
import json
import os
import re

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine, _core
from simplipy.verify import _contract, verify_rule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


def _simplify(engine, expr, mode):
    return ' '.join(engine.simplify(expr.split(), mode=mode))


# --------------------------------------------------------------------- the engine's answers

@pytest.mark.parametrize('expr, strict, permissive', [
    # float64's value or the reals' where float64's kind agrees; the table in permissive
    ('floor tanh 100', '0', '0'),
    ('floor tanh 40', '0', '0'),
    ('+ x0 floor tanh 100', 'x0', 'x0'),
    ('tanh 100', '1', '1'),
    ('sin np.pi', '0', '0'),
    ('* x0 sin np.pi', '0', '0'),
    ('exp sin np.pi', '1', '1'),
    ('floor 0.99999999999999999', '0', '0'),
    # where the reals give no value float64's value stays (spelled as written where cheaper)
    ('tan asin 1', 'tan asin 1', 'float("nan")'),
    ('/ 1 sin np.pi', 'inv sin np.pi', 'float("nan")'),
    ('log sin np.pi', 'log sin np.pi', 'float("-inf")'),
    # a part without a value is read before any law reaches it
    ('* 0 tan asin 1', '0', 'float("nan")'),
    ('pow tan asin 1 0', '1', 'float("nan")'),
    ('pow 1 tan asin 1', '1', 'float("nan")'),
    ('/ tan asin 1 tan asin 1', '1', 'float("nan")'),
    ('/ float("inf") float("inf")', 'float("nan")', 'float("nan")'),
    ('/ exp tan asin 1 exp tan asin 1', 'float("nan")', 'float("nan")'),
    ('- exp tan asin 1 exp tan asin 1', 'float("nan")', 'float("nan")'),
    ('+ x0 tan asin 1', '+ x0 tan asin 1', 'float("nan")'),
    # a range event is no part without a value: the reals' finite value stands beside
    ('* x0 / exp 1000 exp 1000', 'x0', 'x0'),
    # permissive's own simplifications never reach a fully numeric operand: pi - 4 stays
    ('asin sin 4', 'asin sin 4', 'asin sin 4'),
    ('* x0 asin sin 4', '* x0 asin sin 4', '* x0 asin sin 4'),
    ('asin sin x0', 'asin sin x0', 'x0'),
])
def test_a_fully_numeric_part_takes_its_value_first(engine, expr, strict, permissive) -> None:
    assert _simplify(engine, expr, 'f64') == strict
    assert _simplify(engine, expr, 'permissive') == permissive


@pytest.mark.parametrize('expr, strict, permissive', [
    # every number is priced as written: strict writes float64's product of the literals'
    # doubles, permissive the double around the exact product
    ('* x0 * 0.30000000000000004 0.30000000000000004', '* 0.09000000000000002 x0',
     '* 0.09000000000000002 x0'),
    ('* * x0 0.7234567890123457 1.3821548250604163', '* 0.9999292916561291 x0',
     '* 0.9999292916561292 x0'),
    ('0.1234567890123456789', '0.12345678901234568', '0.12345678901234568'),
    ('* x0 0.1234567890123456789', '* 0.12345678901234568 x0', '* 0.12345678901234568 x0'),
    # an exact sum stays where it is the cheaper spelling; float64's sum of the doubles of
    # 0.30000000000000004 and 0.1 is a tie that rounds to 0.4's double, cheaper than the exact
    # 0.40000000000000004
    ('+ 0.1 0.2', '0.3', '0.3'),
    ('+ * 0.30000000000000004 x0 * 0.1 x0', '* 0.4 x0', '* 0.4 x0'),
])
def test_a_number_is_written_as_the_cheapest_value_its_mode_admits(engine, expr, strict,
                                                                   permissive) -> None:
    assert _simplify(engine, expr, 'f64') == strict
    assert _simplify(engine, expr, 'permissive') == permissive


@pytest.mark.parametrize('expr, both', [
    ('* 3.141592653589793 x0', '* np.pi x0'),
    ('* 6.283185307179586 x0', '* 2 * np.pi x0'),
    ('* 0.3333333333333333 x0', '/ x0 3'),
    ('+ x0 0.6931471805599453', '+ x0 log 2'),
    ('* x0 1.4142135623730951', '* x0 rootn 2 2'),
    ('pow 2.718281828459045 x0', 'exp x0'),
    ('sin 3.141592653589793', '0'),
    ('- 3.141592653589793 np.pi', '0'),
    # a literal that already spells its value, or no entry's double, is read as written
    ('* 0.5 x0', '/ x0 2'),
    ('* 3.14159 x0', '* 3.14159 x0'),
])
def test_a_literal_on_the_recognition_list_is_read_as_its_entry(engine, expr, both) -> None:
    assert _simplify(engine, expr, 'f64') == both
    assert _simplify(engine, expr, 'permissive') == both


def test_groundness_belongs_to_the_tree_not_its_spelling(engine) -> None:
    """A tagged spelling of a ground tree is ground: `1/(2/-inf)` is 1/(-0), -inf in float64."""
    x = ['inv', '/', '2', 'float("-inf")']
    assert engine.simplify(x, mode='f64') == ['float("-inf")']
    assert engine.to_prefix(engine.simplify(engine.to_tagged(x), mode='f64')) == ['float("-inf")']


def test_a_literal_binds_no_variable_slot(engine) -> None:
    """A literal of more digits than the exact domain holds is a literal: it binds no `?` slot, so
    `(?0 - 1)/(?0 - 1) -> 1` does not turn float64's NaN into 1 (the literal reads as 1.0)."""
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    rule = (['/', '-', '?0', '1', '-', '?0', '1'], ['1'])
    loaded = SimpliPyEngine(operators=ops, rules=[rule], rules_permissive=[])
    s = '0.' + '9' * 308
    x = ['/', '-', s, '1', '-', s, '1']
    assert loaded.simplify(x, mode='f64') == ['float("nan")']
    assert loaded.simplify(['/', '-', 'x0', '1', '-', 'x0', '1'], mode='f64') == ['1']


def test_a_rule_changes_no_ground_value() -> None:
    """A rule fires on a fully numeric subject only where the ground evaluator admits its result:
    `asin(sin(_0)) -> _0` serves `asin(sin(x0))` and leaves `asin(sin(4))`, whose value is
    pi - 4."""
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    rule = (['asin', 'sin', '_0'], ['_0'])
    loaded = SimpliPyEngine(operators=ops, rules=[], rules_permissive=[rule])
    assert loaded.simplify('asin sin x0'.split(), mode='permissive') == ['x0']
    assert loaded.simplify('asin sin 4'.split(), mode='permissive') == 'asin sin 4'.split()


# ------------------------------------------------------------------------ the judge's reading

@pytest.mark.parametrize('lhs, rhs, strict, permissive, tier', [
    ('+ 0.1 0.2', '0.3', True, True, 'core'),
    ('* 0.1 3', '0.3', True, True, 'core'),
    ('exp sin np.pi', '1', True, True, 'core'),
    ('sin np.pi', '0', True, True, 'core'),
    ('exp log 3', '3', True, True, 'core'),
    ('floor tanh 100', '0', True, True, 'core'),
    ('/ exp 1000 exp 1000', '1', True, True, 'core'),
    ('cos np.pi', '(-1)', True, True, 'core'),
    # strict's float64 value the table refuses: each mode reads its own ground value, and a
    # ground value is the fold's, so no file needs a rule only strict holds
    ('floor tanh 100', '1', True, False, 'reject'),
    ('* 0 tan asin 1', '0', True, False, 'reject'),
    ('pow tan asin 1 0', '1', True, False, 'reject'),
    ('pow 0 -1', 'float("inf")', True, False, 'reject'),
    # the table's value float64 refuses
    ('tan asin 1', 'float("nan")', False, True, 'permissive'),
    ('sin -0.0', '0', False, True, 'permissive'),
    # neither
    ('asin sin 4', '4', False, False, 'reject'),
    ('/ exp tan asin 1 exp tan asin 1', '1', False, False, 'reject'),
    ('/ 1 sin np.pi', 'float("inf")', False, False, 'reject'),
])
def test_the_judge_reads_a_fully_numeric_rule_by_each_modes_value(lhs, rhs, strict, permissive,
                                                                   tier) -> None:
    d = verify_rule(lhs.split(), rhs.split())
    assert d['realised'] is strict and d['permissive'] is permissive and d['tier'] == tier, d
    assert d['basis'] == {'f64': 'proved' if strict else 'refused',
                          'permissive': 'proved' if permissive else 'refused'}, d


def test_the_judge_and_the_engine_read_one_ground_evaluator() -> None:
    """strict's (a) the same double, (b) the same real value with float64's kinds kept, the
    reals' comparison, and permissive's table at float resolution."""
    arity = dict(_contract.ARITY)
    assert _core.ground_rule_verdict(['exp', 'sin', 'np.pi'], ['1'], arity) == \
        (False, True, True, True)
    assert _core.ground_rule_verdict(['cos', 'np.pi'], ['-1'], arity) == (True, True, True, True)
    assert _core.ground_rule_verdict(['floor', 'tanh', '100'], ['1'], arity) == \
        (True, False, False, False)
    # the range standard: float64's inf/inf against the reals' 1
    assert _core.ground_rule_verdict(['/', 'exp', '1000', 'exp', '1000'], ['1'], arity)[1]
    assert _core.ground_rule_verdict(['x0'], ['1'], arity) is None


def test_the_strict_mint_reads_the_ground_evaluator(engine) -> None:
    """The mint's pair judgment decides a fully numeric pair as the judge does:
    `exp(sin(np.pi)) -> 1` holds in strict by the reals' value (float64's 1.0000000000000002 and
    1 are both finite), `exp(sin(np.pi)) -> 0.5` holds nowhere. (The scan meets neither: the
    engine's own fold writes `exp(sin(np.pi))` as 1, so no rule is owed.)"""
    data = _contract.mine_matrix(42, 512, 2, False)
    flat = np.asarray(data).flatten().tolist()
    for target, held in ((['1'], True), (['0.5'], False)):
        ok, why = engine._core.judge_pair('exp sin np.pi'.split(), target, ['x0', 'x1'], flat,
                                          data.shape[0], rtol=1e-11, atol=1e-12,
                                          rule_mode='default')
        assert ok is held, (target, why)
    assert engine.simplify('exp sin np.pi'.split(), mode='f64') == ['1']


def test_a_coincidence_that_reaches_no_answer_is_no_counterexample() -> None:
    """Two `?` slots that coincide make an instance of their own; where that instance reaches no
    answer (its literals lie past the table's reach), the rule is unresolved there, never
    convicted."""
    path = os.path.join(REPO, 'tests', 'fixtures', 'coincidence_unresolved.json')
    lhs, rhs = json.load(open(path))
    slot = lambda t: re.sub(r'^x(\d+)$', r'?\1', t)  # noqa: E731
    d = verify_rule([slot(t) for t in lhs], [slot(t) for t in rhs])
    assert d['verdict'] == 'UNRESOLVED-COVERAGE', d
    assert d['coincidence']['verdict'] == 'UNRESOLVED-COVERAGE', d
