"""Eligibility (the specification's 3.2): a constant tuple owes a witness only where the left side
is finite on a set of bindings of positive measure there. The judge and the mint read it with the
one evidence function at the tuple (`_core.table_owes_at`), and over-approximate only where it
cannot decide: finite at a point is no positive measure."""
import pytest

from simplipy import _core
from simplipy.verify import _contract, verify_rule


@pytest.mark.parametrize('tokens, c, f64, owes', [
    # C^x at C < 0 is finite at the integers alone (the even ones under log): a null set
    ('log pow <constant> x0', -1.5, True, False),
    ('log pow <constant> x0', -1.5, False, False),
    ('log pow <constant> x0', 2.5, True, True),
    ('inv pow <constant> x0', -1.0, True, False),
    ('inv pow <constant> x0', -1.0, False, False),
    # 1/0^x: float64's 1/inf is 0 on x < 0; in the reals 0^x has no value at x < 0, nor 1/0
    ('inv pow <constant> x0', 0.0, True, True),
    ('inv pow <constant> x0', 0.0, False, False),
    # without a variable, the one point decides
    ('pow <constant> acos -1', -1.5, True, False),
    ('pow <constant> acos -1', 2.5, False, True),
])
def test_eligibility_reads_positive_measure_at_the_tuple(tokens, c, f64, owes) -> None:
    assert _core.table_owes_at(tokens.split(), [c], dict(_contract.ARITY), f64) is owes


def test_the_judge_owes_nothing_where_the_left_side_is_finite_on_a_null_set() -> None:
    """`log(C^x) -> C*x`: at every C < 0 of the battery the left side is finite at the even
    integers alone, so no witness is owed and no clause is read there (f64 refused the rule for
    an extension at C = -1.5). In f64 the rule stays refused for float64's own rounding:
    `log(2.5^x)` is 57 units in the last place from `log(2.5)*x` at x = 0.0137."""
    d = verify_rule('log pow <constant> x0'.split(), '* <constant> x0'.split())
    negative = [c for c in _contract.CONSTS + [-1.0, -2.0, -4.0, -5.0, -10.0] if c < 0]
    assert all((c,) in d['skipped_cl'] for c in negative), d['skipped_cl']
    assert 'EXT' not in (d.get('f64_kinds') or []), d.get('f64_kinds')
    assert d['basis']['permissive'] in ('proved', 'evidence'), d['basis']
