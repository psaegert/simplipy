"""OPERATORS SUPPORTED BUT NOT MINED: `operator_max_length: {op: 0}` keeps an operator out of
every source and candidate of a mine, at every length, out of the proposals' vocabulary and out of
the mine's cell; the sidecar records the mined operators, and verify-mine refuses a file whose rule
mentions one (`unmined`)."""
import pytest

from simplipy.mining import _excluded_at_length, _in_cell, _operator_max_length, _universe, _unmined

OPERATORS = {'+': 2, 'erf': 1, 'erfc': 1, 'harmonic': 1, 'exp': 1}


def test_a_bound_of_zero_is_read() -> None:
    oml = _operator_max_length({'erfc': 0, 'harmonic': 0, 'exp': 3}, OPERATORS)
    assert oml == {'erfc': 0, 'exp': 3, 'harmonic': 0}
    assert _unmined(oml) == {'erfc', 'harmonic'}
    assert _unmined({}) == set() and _unmined(None) == set()
    for length in (1, 2, 5):
        assert {'erfc', 'harmonic'} <= _excluded_at_length(length, oml)


@pytest.mark.parametrize('bound', [-1, 1.5, True, '0'])
def test_a_bound_is_a_non_negative_integer(bound) -> None:
    with pytest.raises(ValueError, match='non-negative integer'):
        _operator_max_length({'erfc': bound}, OPERATORS)


def test_the_universe_holds_no_unmined_operator() -> None:
    oml = _operator_max_length({'erfc': 0, 'harmonic': 0}, OPERATORS)
    expressions, counts = _universe(['x0', '1'], OPERATORS, 4, oml)
    held = {t for exprs in expressions.values() for e in exprs for t in e}
    assert {'erf', 'exp', '+'} <= held and not held & {'erfc', 'harmonic'}
    assert all(len(expressions[n]) == counts[n] for n in counts)


def test_the_cell_holds_no_unmined_operator() -> None:
    oml = {'erfc': 0}
    assert not _in_cell({'completion': [['erfc', 'x0'], ['erf', 'x0']]}, 4, oml)
    assert _in_cell({'completion': [['erf', 'erf', 'x0'], ['erf', 'x0']]}, 4, oml)
