"""What a variable is called decides nothing to the judge, as to the engine: every variable is read
as a dummy `x<k>`, a data slot, past x2 and whatever its name (`v1`, `x_1`, `alpha`). A rule over
four variables is judged; a token the judge reads as nothing is refused as a shape."""
import pytest

from simplipy.verify import _contract, verify_rule


@pytest.mark.parametrize('lhs, rhs, renamed', [
    ('sin * x3 x5', 'sin * x5 x3', ('sin * x3 x5', 'sin * x5 x3')),
    ('+ v1 alpha', '+ alpha v1', ('+ x0 x1', '+ x1 x0')),
    ('* x2 x_1', '* x_1 x2', ('* x2 x3', '* x3 x2')),
])
def test_every_variable_reads_as_a_dummy(lhs, rhs, renamed) -> None:
    got = _contract.variables_as_dummies(lhs.split(), rhs.split())
    assert got == (renamed[0].split(), renamed[1].split())
    r = verify_rule(lhs.split(), rhs.split())
    assert r['verdict'] == 'CERTIFIED' and r['tier'] == 'core', r


def test_four_variables_are_judged_and_a_false_rule_over_them_is_refused() -> None:
    assert verify_rule('+ * x3 x4 * x5 x6'.split(), '+ * x4 x3 * x6 x5'.split())['tier'] == 'core'
    r = verify_rule('+ * v1 v2 v3'.split(), '* v1 + v2 v3'.split())
    assert r['verdict'] == 'KILL' and r['tier'] == 'reject', r


def test_literals_operators_and_slots_keep_their_reading() -> None:
    lhs, rhs = 'pow + ?0 np.pi 2'.split(), 'pow + np.pi ?0 2'.split()
    assert _contract.variables_as_dummies(lhs, rhs) == (lhs, rhs)


def test_a_token_that_is_no_name_is_refused_as_a_shape() -> None:
    assert verify_rule(['sin', 'x0', 'x1'], ['x0'])['verdict'] == 'UNSUPPORTED-SHAPE'
    assert verify_rule(['sin', 'a+b'], ['x0'])['verdict'] == 'UNSUPPORTED-SHAPE'
