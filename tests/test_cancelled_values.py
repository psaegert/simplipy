"""The judge's decay test where both sides cancel: a side whose value falls away between two rungs as
residue does was residue at the coarse rung, as one that vanished there is, and the intermediates
set the scale. So a step is judged alike alone and inside an expression that cancels it:
`x0/|x1| -> x0*|1/x1|` holds, and so does `atan2(x0, |x1|) - atan(x0/|x1|)` rewritten by it, in
both modes. A fixed gap is still read: the value does not fall away there."""
import pytest

from simplipy.verify import verify_rule


@pytest.mark.parametrize('lhs, rhs', [
    ('/ x0 abs x1', '* x0 abs inv x1'),
    ('- atan2 x0 abs x1 atan / x0 abs x1', '- atan2 x0 abs x1 atan * x0 abs inv x1'),
])
def test_a_step_holds_alone_and_in_a_cancelling_context(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['verdict'] == 'CERTIFIED' and r['tier'] == 'core', r
    assert r['realised'] is True and r['permissive'] is True, r


@pytest.mark.parametrize('lhs, rhs', [
    # a fixed gap of 3.5e-65 beside pi/2, under an intermediate of 2.85e64
    ('acos inv exp exp 5', 'div2 np.pi'),
    # 1.03e-4783, tiny and fixed
    ('pow np.e sinh -10', '0'),
])
def test_a_fixed_gap_stays_a_counterexample(lhs, rhs) -> None:
    assert verify_rule(lhs.split(), rhs.split())['verdict'] == 'KILL'
