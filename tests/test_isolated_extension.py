"""float64's extension at a point -- the left side undefined, the right side a value -- is a null set
where it is isolated along a data variable: its neighbouring doubles on both sides are no
extension, the left side having a value there or an operation having overflowed or underflowed
there (the range standard). The judge's lane and the core's bar read it the same way.
`atan2(x0, x1)/atan2(-x0, x1)` is undefined only on x0 = 0, x1 > 0 (0/-0), where `atan2(5e-324, 2)`
underflows to 0, and holds in strict as -1."""
from simplipy import _core
from simplipy.verify import _contract, verify_rule

LHS, RHS = '/ atan2 x0 x1 atan2 neg x0 x1'.split(), ['-1']


def test_the_bar_excuses_an_extension_isolated_by_an_underflow() -> None:
    arity = dict(_contract.ARITY)
    for kinds in (False, True):
        ok, probes, first, _ = _core.f64_bar_sorted(LHS, RHS, ['x0', 'x1'], '??', arity,
                                                    kinds=kinds)
        assert ok, (kinds, first)


def test_strict_holds_the_quotient_of_odd_atan2s() -> None:
    r = verify_rule(LHS, RHS)
    assert r['realised'] is True and r['tier'] == 'core', r
    # the reals' one zero reads atan2(0, x1 < 0) as pi on both sides: a diagnostic
    assert r['verdict'] == 'KILL', r


def test_an_extension_on_a_region_stays_refused() -> None:
    r = verify_rule('exp log x0'.split(), ['x0'])
    assert r['realised'] is False, r
