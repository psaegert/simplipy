"""Permissive decides float resolution (P3) without exact equality: a point is read under a
rigorous bound (the interval layer's enclosure at the point) or at the precision the point's
magnitudes demand, and running out of precision never reads as equality."""
from fractions import Fraction

import pytest

from simplipy import _core
from simplipy.verify import _contract, verify_rule


def reading(lhs: str, rhs: str, **at) -> tuple | None:
    """The core's reading of `lhs` against `rhs` with each named slot at its value."""
    names, extra = {}, {}
    trees = [_contract.parse(side.split()) for side in (lhs, rhs)]
    tokens = [_contract._tree_tokens(t, names, extra) for t in trees]
    order = list(names)
    specs = [_contract._spec(extra[n] if n in extra else at[n]) for n in order]
    return _core.table_float_resolution(tokens[0], tokens[1], order, specs,
                                        dict(_contract.ARITY))


def test_the_reading_shares_the_judges_precision_bounds() -> None:
    assert _core.TABLE_BASE_DPS == _contract.BASE_DPS
    assert _core.TABLE_MAX_DPS == _contract.MAX_DPS


def test_a_rigorous_enclosure_decides_a_summand_below_float_resolution() -> None:
    """`1e-3000` changes no double near 3.89: the enclosures of both sides decide it, where the
    reals' readings would demand 6,050 digits."""
    assert reading('+ + pow _0 2 1 1e-3000', '+ pow _0 2 1', _0=Fraction(1, 2))[0] == 'agree'
    assert reading('+ _0 1e-3000', '_0', _0=Fraction(0))[0] == 'agree'
    # 1e-20 against 0 is far past float resolution, at an exact point
    assert reading('+ _0 1e-20', '_0', _0=Fraction(0))[0] == 'differ'


def test_running_out_of_precision_is_no_reading() -> None:
    """`((x + 1e-3000) - x) * 1e3000` is 1 in the reals and 0 at every affordable precision: its
    enclosure is unbounded and the point demands more than the ceiling, so nothing is read."""
    assert reading('* - + _0 1e-3000 _0 1e3000', '0', _0=Fraction(1, 2)) is None


def test_a_reading_is_made_at_the_precision_the_point_demands() -> None:
    """`((x + pi + 1e-200) - (x + pi)) * 1e200` is 1: its magnitudes demand 450 digits, and read
    there it is 1, where fifty digits and a hundred and twenty read 0."""
    assert reading('* - + + _0 np.pi 1e-200 + _0 np.pi 1e200', '0',
                   _0=Fraction(1, 2)) == ('doubles', 1.0, 0.0)


@pytest.mark.parametrize('lhs, rhs', [('+ ?0 1e-3000', '?0'), ('+ _0 1e-3000', '_0')])
def test_a_summand_below_float_resolution_holds_by_permissives_own_criterion(lhs, rhs) -> None:
    d = verify_rule(lhs.split(), rhs.split())
    assert d['permissive'] is True and d['realised'] is True, d
    assert d['verdict'] == 'UNRESOLVED-COVERAGE', d   # the reals' diagnostic stays undecided


@pytest.mark.parametrize('lhs, rhs', [('+ + pow ?0 2 1 1e-3000', '+ pow ?0 2 1'),
                                      ('+ + pow _0 2 1 1e-3000', '+ pow _0 2 1')])
def test_an_atom_the_enclosure_cannot_read_leaves_permissives_own_criterion_no_answer(
        lhs, rhs) -> None:
    """At the atom e, which enters the enclosure as an approximation, `x^2 + 1` encloses wider
    than float resolution, and the summand demands 6,050 digits: permissive's own reading leaves
    an owed point unread and reaches no answer (`permissive_own`). The rule holds in permissive
    through f64, whose two sides are bit-identical."""
    d = verify_rule(lhs.split(), rhs.split())
    assert d['realised'] is True and d['tier'] == 'core' and d['permissive_own'] is None, d
    assert d['permissive'] is True, d
    slot = lhs.split()[3]
    assert d['unread_owed'] == {'permissive': [{slot: 2.718281828459045}]}, d


@pytest.mark.parametrize('lhs', ['+ ?0 1e-20', '* - + ?0 1e-3000 ?0 1e3000'])
def test_a_summand_past_float_resolution_or_an_amplified_one_stays_refused(lhs) -> None:
    rhs = '?0' if lhs.startswith('+') else '0'
    d = verify_rule(lhs.split(), rhs.split())
    assert d['tier'] == 'reject' and d['permissive'] is not True, d


def test_a_cancellation_read_at_the_precision_it_demands_is_no_excuse() -> None:
    """`((x + pi + 1e-200) - (x + pi)) * 1e200 -> 0` is false by 1 everywhere: the ladder reads
    the gap at the precision the point demands, and the excuse of float resolution reads there
    too (two readings at fifty and a hundred and twenty digits agreed on 0 and excused it)."""
    d = verify_rule('* - + + ?0 np.pi 1e-200 + ?0 np.pi 1e200'.split(), ['0'])
    assert d['permissive'] is False and d['tier'] == 'reject', d
