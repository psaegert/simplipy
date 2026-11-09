"""erfc, bessely, besseli and besselk in the core: the engine reads them, the table's real values
agree with mpmath (at orders a hair off an integer too, where every Bessel function reads the
order's distance to it), the interval layer's enclosures hold every value over boxes in both
readings, and the judge decides rules over them."""
import math
import random
from fractions import Fraction

import mpmath
import pytest
import yaml

from simplipy import SimpliPyEngine, _core
from simplipy.verify import _contract
from test_new_heads import ACJ2_4

ARITY = dict(_contract.ARITY)
NEW = {'erfc': 1, 'bessely': 2, 'besseli': 2, 'besselk': 2}
REF = {'erfc': mpmath.erfc, 'besselj': mpmath.besselj, 'bessely': mpmath.bessely,
       'besseli': mpmath.besseli, 'besselk': mpmath.besselk}


@pytest.fixture(scope='module')
def engine():
    """acj2-4's operators and the four, with an empty rule set per mode."""
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    for name, arity in NEW.items():
        ops[name] = {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
                     'commutative': False}
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


def exact(q) -> tuple[str, str, str]:
    q = Fraction(q)
    return ('q', str(q.numerator), str(q.denominator))


def table(op: str, args, dps: int):
    """The table's real value of `op` at exact operands, an mpf; None where Unresolved."""
    names = [f'x{i}' for i in range(len(args))]
    spec, unresolved, *_ = _core.TableExpr([op] + names, names, ARITY).eval_real(
        [exact(a) for a in args], dps)
    if unresolved:
        return None
    with mpmath.workdps(dps + 20):
        return _contract._from_spec(spec)


def test_the_engine_reads_them(engine) -> None:
    import simplipy.operators as ops
    for mode in ('f64', 'permissive'):
        s = lambda t: engine.simplify(t, mode=mode)  # noqa: E731
        assert s('erfc(x1)') == 'erfc(x1)'
        assert s('besseli(2.5, x1*x2)') == 'besseli(5/2, x1*x2)'
        assert s('besselk(0, x1)/1') == 'besselk(0, x1)'
        assert s('besseli(0, 0)') == '1' and s('besselk(1, 0)') == 'inf' and s('bessely(0, 0)') == '-inf'
        assert s('erfc(inf)') == '0'
        # an order is structure, as an exponent is: printed exactly beside a float argument, and
        # no number the display reads (a function of a float folds to it, whatever the order)
        assert s('bessely(1.5, 0.37)') == 'bessely(3/2, 0.37)'
        assert engine.prettify(s('besselk(2, 7452/1615)')) == '0.008339089152780824'
    # float64 keeps a zero's sign the scipy heads give it on whole regions; permissive drops it
    assert engine.simplify('bessely(2, x1) + 0', mode='f64') == 'bessely(2, x1) + 0'
    assert engine.simplify('bessely(2, x1) + 0', mode='permissive') == 'bessely(2, x1)'
    ev = engine._core.evaluate_constant_subtree
    for toks, f, args in ((['erfc', '0.5'], ops.erfc, (0.5,)), (['bessely', '1', '2'], ops.bessely, (1.0, 2.0)),
                          (['besseli', '2.5', '3'], ops.besseli, (2.5, 3.0)),
                          (['besselk', '0', '700'], ops.besselk, (0.0, 700.0))):
        assert float(ev(toks)) == float(f(*args)), toks


#: Points across the methods of each table value: erfc's `1 - erf` and its asymptotic series; the
#: power series of I, of Y and K through J_-+nu and I_-+nu, DLMF 10.8.1 and 10.31.1 at integer
#: orders, the Hankel expansions at large x, negative orders, an integer order at x < 0 for I.
POINTS = {
    'erfc': [(-3,), (Fraction(-1, 2),), (Fraction(1, 10 ** 10),), (Fraction(1, 2),), (1,), (3,), (10,), (27,), (100,)],
    'bessely': [(0, Fraction(1, 2)), (1, 2), (Fraction(5, 2), 10), (Fraction(-3, 2), 3), (Fraction(3, 10), 50),
                (7, Fraction(1, 100)), (50, 100), (Fraction(-9, 4), Fraction(7, 10)), (100, 30), (2, 200)],
    'besseli': [(0, Fraction(1, 2)), (1, 2), (Fraction(5, 2), 10), (Fraction(-3, 2), 3), (Fraction(3, 10), 50),
                (7, Fraction(1, 100)), (50, 100), (Fraction(-9, 4), Fraction(7, 10)), (3, -2), (-4, -5)],
    'besselk': [(0, Fraction(1, 2)), (1, 2), (Fraction(5, 2), 10), (Fraction(-3, 2), 3), (Fraction(3, 10), 50),
                (7, Fraction(1, 100)), (50, 100), (Fraction(-9, 4), Fraction(7, 10)), (2, 200), (0, 600)],
}


@pytest.mark.parametrize('op', sorted(POINTS))
def test_the_tables_values_are_the_functions(op) -> None:
    for args in POINTS[op]:
        got = table(op, args, 40)
        assert got is not None, (op, args)
        with mpmath.workdps(120):
            want = REF[op](*[mpmath.mpf(a.numerator) / a.denominator if isinstance(a, Fraction) else mpmath.mpf(a)
                             for a in args])
            assert abs((got - want) / want) < mpmath.mpf(10) ** -39, (op, args, got, want)


@pytest.mark.parametrize('op', ['besselj', 'bessely', 'besseli', 'besselk'])
def test_an_order_a_hair_off_an_integer_is_read_to_its_distance(op) -> None:
    """Every Bessel function reads the order's distance to the nearest integer: near a negative
    integer J and I swing by hundreds of digits with it (`J_-(300+e)(1.7)` is about 1.5e575 at
    e = 1e-58, `J_300(1.7)` 2e-636), and Y and K divide by `sin(nu pi)`. An exact order keeps
    that distance at every working precision; a reading that rounded the order to the working
    precision took J_-n's value, or divided by a sine read as 0."""
    eps = Fraction(1, 10 ** 58)
    for base in (3, -3, Fraction(7, 2), Fraction(-7, 2), 300, -300):
        for nu in (base - eps, base + eps):
            with mpmath.workdps(120):
                want = REF[op](mpmath.mpf(nu.numerator) / nu.denominator, mpmath.mpf(17) / 10)
            for dps in (20, 40):
                got = table(op, (nu, Fraction(17, 10)), dps)
                assert got is not None and abs((got - want) / want) < mpmath.mpf(10) ** (-dps + 1), \
                    (op, float(nu), dps, got, want)


def kernel(op: str, args) -> float:
    if len(args) == 1:
        return _core.kernel_unary_scalar(op, *args)
    return _core.kernel_binary_scalar(op, *args)


def contains(box, v) -> bool:
    has_fin, pinf, ninf, nan, lo, hi = box
    if v != v:
        return nan
    if v in (math.inf, mpmath.inf):
        return pinf
    if v in (-math.inf, -mpmath.inf):
        return ninf
    return has_fin and mpmath.mpf(lo) <= v <= mpmath.mpf(hi)


def null_set(op: str, lo, hi, p) -> bool:
    """A value the box takes on a null set only, which an enclosure need not hold: the pole at
    x = 0 at the edge of a wider argument range, and besseli's integer orders at x < 0 inside a
    wider order range."""
    if op == 'erfc':
        return False
    return (p[1] == 0.0 and lo[1] < hi[1]) or (op == 'besseli' and p[1] < 0 and p[0] == int(p[0]) and lo[0] < hi[0])


@pytest.mark.parametrize('op', sorted(NEW))
def test_every_enclosure_holds_every_value(op, engine) -> None:
    """The interval layer's enclosure of each head over a box holds the kernel's value at every
    point of it in float64's reading, and the table's in the reals': boxes at tiny, moderate and
    huge arguments, near AMOS's cuts and overflows, at small, integer, half-integer and large
    orders."""
    core, rnd = engine._core, random.Random(20261103)

    def rand_range(kind):
        if kind == 'order':
            a = rnd.choice([rnd.uniform(-60, 60), round(rnd.uniform(-10, 10)) + rnd.choice([0, 0.5]),
                            rnd.uniform(0, 3), rnd.uniform(45, 95), rnd.uniform(-1, 1) * 1e-6])
        else:
            a = rnd.choice([10 ** rnd.uniform(-12, 3), rnd.uniform(-5, 5), rnd.uniform(0, 30),
                            rnd.uniform(650, 760), 10 ** rnd.uniform(-310, -295)])
        width = rnd.choice([0.0, abs(a) * 1e-12, 0.01, 0.5, 3.0, abs(a)])
        return a, a + width

    checked = 0
    for _ in range(40):
        if op == 'erfc':
            xr = rand_range('x') if rnd.random() < 0.7 else (rnd.uniform(-8, 30), 0.0)
            lo, hi = [min(xr)], [max(xr)]
        else:
            nr, xr = rand_range('order'), rand_range('x')
            lo, hi = [nr[0], xr[0]], [nr[1], xr[1]]
        tokens = [op] + [f'x{i}' for i in range(len(lo))]
        for real in (False, True):
            box = core.interval_value_set_box(tokens, lo, hi, real)
            if box is None:
                continue  # refused: no claim to hold
            for i in range(5):
                p = [a if i == 0 else b if i == 1 else rnd.uniform(a, b) for a, b in zip(lo, hi)]
                if real:
                    v = table(op, [Fraction(t) for t in p], 30)
                    if v is None:
                        continue
                else:
                    v = kernel(op, p)
                checked += 1
                assert contains(box, v) or null_set(op, lo, hi, p), (real, op, lo, hi, p, v, box)
    assert checked > 200


def test_the_judge_decides_rules_over_them() -> None:
    j = _contract.judge_rule
    for lhs, rhs in (('besselk neg _0 _1', 'besselk _0 _1'), ('besseli 2 neg _0', 'besseli 2 _0'),
                     ('besseli 3 neg _0', 'neg besseli 3 _0'), ('+ erfc ?0 erf ?0', '1')):
        r = j(lhs.split(), rhs.split())
        assert r['verdict'] == 'CERTIFIED' and r['tier'] == 'core', (lhs, rhs, r)
    # true over the reals, not within F6 in float64, and its two sides finite at every point:
    # strict holds it by its second reading
    r = j('erfc neg _0'.split(), '- 2 erfc _0'.split())
    assert r['verdict'] == 'CERTIFIED' and r['tier'] == 'core', r
    assert r['realised_f6'] is False and r['realised_kinds'] is True, r
    # false
    r = j('bessely 0.5 _0'.split(), ['_0'])
    assert r['verdict'] == 'KILL' and r['tier'] == 'reject', r


#: Where the joint limit does not exist: Y at x = 0 under a negative half-integer order (the
#: orders beside it tend to infinities of both signs), and I at an infinite negative order (no
#: value at any argument, as for besselj); float64 keeps scipy's cells there.
NO_VALUE = [('bessely(-0.5, 0)', ('bessely', -0.5, 0.0), -math.inf),
            ('bessely(-1.5, 0)', ('bessely', -1.5, 0.0), math.inf),
            ('bessely(-2.5, 0)', ('bessely', -2.5, 0.0), -math.inf),
            ('besseli(-inf, -1)', ('besseli', -math.inf, -1.0), 0.0),
            ('besseli(-inf, -2.5)', ('besseli', -math.inf, -2.5), 0.0)]


@pytest.mark.parametrize('expr, call, f64', NO_VALUE)
def test_where_the_joint_limit_does_not_exist_there_is_no_value(engine, expr, call, f64) -> None:
    """Permissive's fold reads the table: no value, written NaN. f64's value is scipy's cell."""
    import simplipy.operators as ops
    assert engine.simplify(expr, mode='permissive') == 'nan', expr
    assert float(getattr(ops, call[0])(*call[1:])) == f64, expr
    assert engine.simplify(expr, mode='f64') != 'nan', expr


@pytest.mark.parametrize('order', [Fraction(-1, 2), Fraction(-3, 2), Fraction(-5, 2)])
def test_the_table_has_no_value_for_y_at_zero_under_a_negative_half_integer_order(order) -> None:
    spec, unresolved, *_ = _core.TableExpr(['bessely', 'x0', 'x1'], ['x0', 'x1'], ARITY).eval_real(
        [exact(order), exact(0)], 50)
    assert not unresolved and spec[0] == 'nan', spec
    # an approximate order within the working band of -1/2 cannot be told from it
    spec, unresolved, *_ = _core.TableExpr('bessely neg log rootn np.e 2 0'.split(), [], ARITY) \
        .eval_real([], 50)
    assert unresolved


def test_the_table_has_no_value_at_an_infinite_negative_order() -> None:
    for x in ('-1', '-2.5', '0', '1'):
        spec, unresolved, *_ = _core.TableExpr(['besseli', 'float("-inf")', x], [], ARITY) \
            .eval_real([], 50)
        assert not unresolved and spec[0] == 'nan', (x, spec)


def test_a_rule_whose_points_had_no_reading_there_holds_in_permissive() -> None:
    """`1/bessely(_0, 0) -> 0`: Y_nu(0) is an infinity at every order but the negative
    half-integers, where it has no value and the right side extends it (permissive's own
    criterion is agreement where the left side has a value). Read there, the rule holds."""
    r = _contract.judge_rule('inv bessely _0 0'.split(), ['0'])
    assert r['permissive'] is True and 'unread_owed' not in r, r
