# mypy: ignore-errors
"""The judge's exact routes: a rule decided over its whole domain, with no sampling.

Equality is undecidable for the language a rule is spelled in (Richardson), so the judge's
general procedure is a battery of points; on the fragments below it decides exactly, and there
its verdict is a proof or a refutation, not evidence.

- THE PIECEWISE-AFFINE ROUTE (`piecewise_affine`). A side built from one variable, rational
  literals, `+`, `-`, `neg`, products and quotients where one factor is constant on every cell,
  `abs`, `max` and `min` is a continuous piecewise-affine function of the variable: finitely
  many rational breakpoints and an affine piece between consecutive ones. Each operation acts
  on the pieces exactly (a `max` splits a cell at the one point where its two pieces cross), so
  the two sides of a rule are equal on every cell, and the rule holds at every real binding, or
  their difference is a nonzero affine function on some cell, and the rule fails on an interval
  of positive length, at a rational witness. `max(0, 1e-200 - abs(x - 0.123456789)) -> 0` is
  refuted at x = 0.123456789. Several variables would need polyhedral cells; a rule with more
  than one variable is outside the route. Past `MAX_BREAKPOINTS` breakpoints or `MAX_BITS` bits
  in a coefficient the route answers unknown.
- THE WHOLE-DOMAIN ENCLOSURE (`whole_domain_literal`). A left side whose interval enclosure
  over every finite binding of its variables and constants is one value and nothing else, the
  right side's literal, is that value everywhere: `0*exp(x0) -> 0`.
- THE GROUND POINT (`ground`). A rule with no variable and no constant has one point. Where the
  table's values of both sides there are exact (a rational, a rational multiple of pi, an
  infinity, undefined) the comparison is exact; float64's values are always exact, its kernels
  being the realizations.
- THE CONSTANT'S OWN VALUE (`own_value`). `L(C) -> C` with no variable holds in every mode: the
  witness is the left side's own value wherever it is finite, and nothing is owed elsewhere.

Each route reports per mode: 'proved', 'refuted' or None (unknown). The reals' answer is the
route's own. Permissive's follows it, a refutation only where the two real values at the
witness round to doubles more than 8 units in the last place apart (P3). f64's is proved only
where every operation of the rule is exact in float64 (`neg`, `abs`, `max`, `min` on doubles,
literals that are doubles), and refuted where float64 itself disagrees at a double near the
witness with no operation leaving the range.
"""
import bisect
import math
from fractions import Fraction

import numpy as np

#: The most breakpoints a side's function may have before the route answers unknown.
MAX_BREAKPOINTS = 256
#: The most bits a coefficient or breakpoint may carry before the route answers unknown.
MAX_BITS = 4096

#: The operations of the fragment.
_FRAGMENT = frozenset({'+', '-', 'neg', '*', '/', 'abs', 'max', 'min'})
#: The operations float64 computes exactly on doubles: a rule spelled with these alone (and
#: literals that are doubles) has float64 values equal to its real values at every double.
_F64_EXACT = frozenset({'neg', 'abs', 'max', 'min'})


class Unknown(Exception):
    """The route cannot decide the rule: outside the fragment, or past a size limit."""


def as_double(q):
    """The double nearest to the exact value `q`, or None past the doubles' range (`1e400`):
    there it has no double, and no float resolution to be compared at."""
    try:
        return float(q)
    except OverflowError:
        return None


def _check_size(*values):
    for v in values:
        if v.numerator.bit_length() > MAX_BITS or v.denominator.bit_length() > MAX_BITS:
            raise Unknown('a coefficient past the size limit')


class _PA:
    """A continuous piecewise-affine function: breakpoints `b[0] < ... < b[k-1]` and `k + 1`
    pieces `(a, c)`, the function `a*x + c` on the cell between consecutive breakpoints (the
    first and last cells unbounded)."""
    __slots__ = ('b', 'p')

    def __init__(self, b, p):
        if len(b) > MAX_BREAKPOINTS:
            raise Unknown('more breakpoints than the size limit')
        self.b, self.p = b, p

    def constant(self):
        """The function's value if it is one constant, else None."""
        return self.p[0][1] if not self.b and self.p[0][0] == 0 else None

    def at(self, x):
        a, c = self.p[bisect.bisect_right(self.b, x)]
        return a * x + c

    def cells(self):
        """`(lo, hi, piece)` per cell, an unbounded side as None."""
        edges = [None] + self.b + [None]
        return [(edges[i], edges[i + 1], self.p[i]) for i in range(len(self.p))]


def _point_in(lo, hi):
    """A point inside the cell `(lo, hi)` (None: unbounded on that side)."""
    if lo is None and hi is None:
        return Fraction(0)
    if lo is None:
        return hi - 1
    if hi is None:
        return lo + 1
    return (lo + hi) / 2


def _refined(f, b):
    """`f`'s pieces on the finer breakpoints `b` (a superset of `f.b`)."""
    return [f.p[bisect.bisect_right(f.b, _point_in(lo, hi))]
            for lo, hi in zip([None] + b, b + [None])]


def _normal(b, p):
    """Drop the breakpoints between two equal pieces (the function is continuous)."""
    nb, np_ = [], [p[0]]
    for x, piece in zip(b, p[1:]):
        if piece == np_[-1]:
            continue
        nb.append(x)
        np_.append(piece)
    for a, c in np_:
        _check_size(a, c)
    for x in nb:
        _check_size(x)
    return _PA(nb, np_)


def _pointwise(f, g, op):
    b = sorted(set(f.b) | set(g.b))
    return _normal(b, [op(u, v) for u, v in zip(_refined(f, b), _refined(g, b))])


def _scaled(f, k):
    return _normal(list(f.b), [(k * a, k * c) for a, c in f.p])


def _max(f, g):
    """The pointwise maximum: each common cell split where its two pieces cross."""
    b = sorted(set(f.b) | set(g.b))
    fs, gs = _refined(f, b), _refined(g, b)
    nb, np_ = [], []
    for (lo, hi), u, v in zip(zip([None] + b, b + [None]), fs, gs):
        da, dc = u[0] - v[0], u[1] - v[1]
        cuts = [lo]
        if da != 0:
            r = -dc / da
            if (lo is None or lo < r) and (hi is None or r < hi):
                cuts.append(r)
        cuts.append(hi)
        for s, t in zip(cuts, cuts[1:]):
            x = _point_in(s, t)
            pick = u if u[0] * x + u[1] >= v[0] * x + v[1] else v
            if np_:
                nb.append(s)
            np_.append(pick)
    return _normal(nb, np_)


def _neg(f):
    return _normal(list(f.b), [(-a, -c) for a, c in f.p])


def _product(f, g):
    """A product affine on every cell: one factor's slope is zero wherever the other's is not."""
    b = sorted(set(f.b) | set(g.b))
    out = []
    for (a1, c1), (a2, c2) in zip(_refined(f, b), _refined(g, b)):
        if a1 * a2 != 0:
            raise Unknown('a product of two varying factors')
        out.append((a1 * c2 + a2 * c1, c1 * c2))
    return _normal(b, out)


def _build(tree, var):
    """`tree` (the judge's parse) as a piecewise-affine function of the slot `var`."""
    head = tree[0]
    if head == 'slot':
        if tree[1] != var:
            raise Unknown('another variable or a constant')
        return _PA([], [(Fraction(1), Fraction(0))])
    if head == 'lit':
        v = tree[1]
        if not isinstance(v, (Fraction, int)) or isinstance(v, bool):
            raise Unknown('a literal that is no rational')
        v = Fraction(v)
        _check_size(v)
        return _PA([], [(Fraction(0), v)])
    if head not in _FRAGMENT:
        raise Unknown(f'{head} is outside the piecewise-affine fragment')
    args = [_build(t, var) for t in tree[1:]]
    if head == 'neg':
        return _neg(args[0])
    if head == 'abs':
        return _max(args[0], _neg(args[0]))
    if head == '+':
        return _pointwise(args[0], args[1], lambda u, v: (u[0] + v[0], u[1] + v[1]))
    if head == '-':
        return _pointwise(args[0], args[1], lambda u, v: (u[0] - v[0], u[1] - v[1]))
    if head == 'max':
        return _max(args[0], args[1])
    if head == 'min':
        return _neg(_max(_neg(args[0]), _neg(args[1])))
    if head == '*':
        return _product(args[0], args[1])
    k = args[1].constant()
    if k is None or k == 0:
        raise Unknown('a quotient by a varying or zero divisor')
    return _scaled(args[0], 1 / k)


def _ops_and_slots(tree, ops, slots):
    if tree[0] == 'slot':
        slots.add(tree[1])
    elif tree[0] == 'lit':
        ops.add(('lit', tree[1]))
    else:
        ops.add(tree[0])
        for t in tree[1:]:
            _ops_and_slots(t, ops, slots)


def _f64_exact(trees):
    """Is every operation of the rule exact in float64, every literal a double?"""
    for tree in trees:
        ops, slots = set(), set()
        _ops_and_slots(tree, ops, slots)
        for op in ops:
            if isinstance(op, tuple):
                v = op[1]
                if not isinstance(v, (Fraction, int)) or as_double(v) is None \
                        or Fraction(v) != Fraction(as_double(v)):
                    return False
            elif op not in _F64_EXACT:
                return False
    return True


def _witness(d):
    """A point where the nonzero piecewise-affine `d` is largest in magnitude among the ends
    and middles of the cells where it is not zero."""
    best, at = None, None
    for lo, hi, (a, c) in d.cells():
        if (a, c) == (0, 0):
            continue
        for x in (lo, hi, _point_in(lo, hi)):
            if x is None:
                continue
            v = abs(a * x + c)
            if best is None or v > best:
                best, at = v, x
    return at


def piecewise_affine(tl, tr, d_eval, f64_agree):
    """The piecewise-affine route on the judge's parsed sides `tl`, `tr`: `None` where the rule is
    outside the fragment or past a size limit, else `{'route', 'reals', 'permissive', 'f64',
    'witness'}`, each mode 'proved', 'refuted' or None. `d_eval(tree, env, in_range)` is the
    judge's float64 evaluator and `f64_agree(a, b)` F6's agreement."""
    slots = set()
    for t in (tl, tr):
        _ops_and_slots(t, set(), slots)
    variables = [s for s in slots if not s.startswith('<C')]
    if len(variables) != 1 or len(slots) != 1:
        return None
    var = variables[0]
    try:
        d = _pointwise(_build(tl, var), _build(tr, var),
                       lambda u, v: (u[0] - v[0], u[1] - v[1]))
    except (Unknown, ZeroDivisionError, OverflowError):
        return None
    out = {'route': 'piecewise-affine', 'witness': None}
    if not d.b and d.p[0] == (0, 0):
        out.update(reals='proved', permissive='proved',
                   f64='proved' if _f64_exact((tl, tr)) else None)
        return out
    x = _witness(d)
    out.update(reals='refuted', witness=x)
    lv, rv = (as_double(_build(t, var).at(x)) for t in (tl, tr))
    out['permissive'] = 'refuted' if None not in (lv, rv) and f64_agree(lv, rv) != 'eq' \
        else None
    out['f64'] = None
    xf = as_double(x)
    for y in () if xf is None else (xf, math.nextafter(xf, math.inf),
                                    math.nextafter(xf, -math.inf)):
        in_range = [True]
        with np.errstate(all='ignore'):
            dl, dr = d_eval(tl, {var: np.float64(y)}, in_range), d_eval(tr, {var: np.float64(y)},
                                                                        in_range)
        if in_range[0] and f64_agree(float(dl), float(dr)) != 'eq':
            out['f64'] = 'refuted'
            out['witness_f64'] = y
            break
    return out


def whole_domain_literal(lhs_spelled, tr, value_set, literal_f64):
    """The whole-domain enclosure route: `{'route', 'reals', 'permissive', 'f64'}` where the left
    side's enclosure over every finite binding is exactly the right side's literal (`tr` a
    literal), each mode 'proved' or None; `None` where the right side is no literal.
    `value_set(tokens, f64)` is the layer's enclosure, `literal_f64(token)` F2's reading."""
    if tr[0] != 'lit' or len(tr) < 3:
        return None
    tok, v = tr[2], tr[1]

    def is_exactly(vs, value):
        if vs is None:
            return False
        has_fin, pinf, ninf, nan, lo, hi = vs
        if isinstance(value, float) and math.isnan(value):
            return nan and not (has_fin or pinf or ninf)
        if isinstance(value, float) and math.isinf(value):
            return (pinf if value > 0 else ninf) and not (has_fin or nan or (ninf if value > 0
                                                                             else pinf))
        return (has_fin and not (pinf or ninf or nan) and lo == hi
                and Fraction(lo) == Fraction(value))

    real = v if isinstance(v, (Fraction, int)) or (isinstance(v, float) and not math.isfinite(v)) \
        else None
    out = {'route': 'whole-domain', 'reals': None, 'permissive': None, 'f64': None}
    if real is not None and is_exactly(value_set(lhs_spelled, False), real):
        out['reals'] = out['permissive'] = 'proved'
    fv = literal_f64(tok)
    if fv is not None and is_exactly(value_set(lhs_spelled, True), fv):
        out['f64'] = 'proved'
    return out if any(out[m] for m in ('reals', 'permissive', 'f64')) else None


def ground(tl, tr, verdict):
    """The ground-point route: `None` where the rule has a variable or a constant, else
    `{'route', 'reals', 'permissive', 'f64', 'readings'}`. A fully numeric rule has one point,
    and the core's ground evaluator reads it (`verdict()`, `_core.ground_rule_verdict`: the one
    definition the engine's parse and fold and verify-mine's gate read). It holds in strict where
    float64 gives both sides one double (a), or the reals give both one value and float64's two
    values agree in kind (b): `exp(sin(np.pi)) -> 1` and `0.1 + 0.2 -> 0.3` hold by (b),
    `floor(tanh(100)) -> 1` by (a). It holds in permissive where the table's two values agree
    at float resolution, permissive's own value of a fully numeric part:
    `0 * tan(asin(1)) -> 0` holds in strict and not in permissive, whose left side has no value.
    `readings` keeps strict's (a) and (b) apart; every answer is None where a side cannot be
    read."""
    slots = set()
    for t in (tl, tr):
        _ops_and_slots(t, set(), slots)
    if slots:
        return None
    out = {'route': 'ground', 'reals': None, 'permissive': None, 'f64': None, 'readings': None}
    v = verdict()
    if v is None:
        return out
    same_double, same_real_in_kind, reals, permissive = v
    out['f64'] = 'proved' if same_double or same_real_in_kind else 'refuted'
    out['permissive'] = 'proved' if permissive else 'refuted'
    out['reals'] = None if reals is None else ('proved' if reals else 'refuted')
    out['readings'] = {'same_double': same_double, 'same_real_in_kind': same_real_in_kind}
    return out


def own_value(tl, tr):
    """The constant's-own-value route: `{'route', 'reals', 'permissive', 'f64'}`, every mode
    proved, where the right side is one `<constant>` and the left side has a constant and no
    variable; else `None`."""
    if tr[0] != 'slot' or not tr[1].startswith('<C'):
        return None
    slots = set()
    _ops_and_slots(tl, set(), slots)
    if not slots or any(not s.startswith('<C') for s in slots):
        return None
    return {'route': 'own-value', 'reals': 'proved', 'permissive': 'proved', 'f64': 'proved'}
