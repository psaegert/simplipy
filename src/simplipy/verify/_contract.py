# mypy: ignore-errors
#!/usr/bin/env python
"""The contract judge: `holds(L, R, mode)` for a rewrite rule, on THE TABLE.

A single implementation of the criterion (the specification's section 3): every soundness
instrument (the rule-completeness gate, the mine's gates, verify-mine) reads it. Every value at a
point is THE TABLE's (`rust/table.rs`, read through `_core.TableExpr`): real's exact or
high-precision values with the limit rule at poles and infinite arguments (`1/0` and
`pow(0, -1)` are undefined, `pow(0, -2)` is +inf, `log(0)` is -inf), and f64's evaluator values
with the range standard. The judge keeps its procedure -- the precision ladder, the constant
fitting, the sampled points -- and owns none of the values.

The criterion, per mode:
  - (a) agreement: at every binding the sorts quantify over where `L` has a value, `R` has an
    agreeing one (real: equal; f64: within 8 units in the last place, or the true value at a
    range event). A disagreement at any such point fails the rule, whatever its measure.
  - (b) no invented function: the bindings where `L` is undefined and `R` has a value form a
    set of measure zero. Measure is decided by certified cells of the interval layer, read in
    the table's values (`_core.table_extension_width`), never by a fraction of a grid. A `_`
    slot bound to +-inf or NaN is a point of its sort's domain: nothing is excused there.
  - permissive: clause (a) alone, over the real-valued bindings (every slot any real number).
  - constants are independent symbols; at constants where `L` is not finite nothing is owed.
  - a ground rule has one point, and no tolerance: if `L` is undefined, `R` must be too.

Precision honesty:
  - Battery points are exact table values (pi/2 is pi/2, rebuilt per rung); f64 renderings of
    transcendentals are NOT the contract's points.
  - Every non-eq point verdict is CONFIRMED at a second rung; rung disagreement ->
    Unresolved (the point is skipped, never convicted).
"""
import itertools
import math
import re
from fractions import Fraction
import numpy as np
from mpmath import mp, mpf, isnan as misnan, isinf as misinf

# No `np.seterr(all='ignore')` here -- that call is PROCESS-WIDE and ran at
# import, silently changing the embedder's numpy error handling because this module
# was imported. The deployed-check evaluator scopes `np.errstate` around its own
# operator applications instead (see `d_eval`); the host's configuration survives.
F = np.float64

try:  # the CORE serialization language, read from the engine's own table:
    from simplipy._core import core_serialization_ops as _core_ops  # type: ignore[attr-defined]
    _CORE_ARITY = {t: s['arity'] for t, s in _core_ops().items()}
except Exception:  # pragma: no cover  (missing/unbuilt extension)
    _CORE_ARITY = {}

#: The judge's vocabulary is DELIBERATELY broader than the engine's: it keeps the deleted
#: 0.11 hyper-operator spellings, because a generation-2 config may still declare those
#: names as operators (the mining test vocabularies declare `pow2`, and the Rust evaluator
#: has kernels for them), so rules spelled with them stay judgeable (pinned by
#: `test_legacy_hyperop_spellings_remain_judgeable`). Only the CORE half is derived --
#: that is the half that must never drift from the engine; the rest is this module's own
#: and is listed explicitly.
ARITY = {**_CORE_ARITY,
         'abs': 1, 'exp': 1, 'log': 1,
         'sin': 1, 'cos': 1, 'tan': 1, 'asin': 1, 'acos': 1, 'atan': 1,
         'sinh': 1, 'cosh': 1, 'tanh': 1, 'asinh': 1, 'acosh': 1, 'atanh': 1,
         'pow2': 1, 'pow3': 1, 'pow4': 1, 'pow5': 1,
         'pow1_2': 1, 'pow1_3': 1, 'pow1_4': 1, 'pow1_5': 1,
         'div2': 1, 'div3': 1, 'div4': 1, 'div5': 1,
         'mult2': 1, 'mult3': 1, 'mult4': 1, 'mult5': 1,
         # the 0.15 heads
         'floor': 1, 'ceil': 1, 'erf': 1, 'harmonic': 1,
         'max': 2, 'min': 2, 'mod': 2, 'atan2': 2, 'besselj': 2}
SLOT_RE = re.compile(r'^([?!_$])(\d+)$')

#: The 0.15 heads. A rule spelled with one answers to the core's f64 bar for its float64
#: realisation (`_core.f64_bar`, ANDed into `realised` by `judge_rule`): float64 parts from the
#: reals at a step head's jump (`floor(x + 1)` at x = -1e-17), at a zero's sign `atan2` reads,
#: and at a pole of `harmonic`, points the battery and the grid below do not reach.
NEW_HEADS = frozenset({'floor', 'ceil', 'erf', 'harmonic', 'max', 'min', 'mod', 'atan2',
                       'besselj'})


class Unresolved(Exception):
    """the point cannot be judged honestly at working precision -- skip, never convict"""


#: The ordinary working precision, and the ceiling an operand-scaled escalation may
#: reach. Past the ceiling a point is UNRESOLVED -- never convicted -- because the
#: judge has run out of evidence, and "skip, never convict" is the house rule.
BASE_DPS = 50
MAX_DPS = 2000

SNAP_EVENTS = [0]  # incremented when a symbolic-cancellation snap / pole-guard fires;
                   # at such points the f64 deployed algebra evaluates a DIFFERENT point
                   # (a measurement artifact), so the deployed check is skipped


def source_defined(value):
    """THE CONSTANT-COLLAPSE CRITERION, the one definition every gate reads: is the SOURCE
    defined at this value, i.e. finite (element-wise over an array; mpmath values too)?

    A target `<constant>` is fitted to data, and a fitted expression is finite on the data,
    so a source constant at which the source is NaN or infinite is never the fit. A collapse
    therefore owes a witness exactly where the source is finite, and nothing elsewhere: the
    evidence gate's rule, which counts source-finite rows only. `acos(c) -> c` holds for
    |c| <= 1 and is vacuous beyond. A source finite at no probed constant
    (`log(neg(cosh(c)))`) has no evidence at all and gets no rule. Read by the witness fit
    below (`_fit_target_env`), the sort ladder's ground tier and the const-bearing witness
    bar (`simplipy.promotion`)."""
    with np.errstate(over='ignore', invalid='ignore'):
        return np.isfinite(np.asarray(value, dtype=np.float64))


# ---------------------------------------------------------------- literal acceptor
class UnsupportedToken(ValueError):
    """a leaf token outside the accepted numeric grammar -- REFUSED, never evaluated.

    This module adjudicates rule sets it did not produce (`verify_ruleset` documents a
    path to a JSON file as input), so a leaf token is untrusted data, not code. The
    grammar below is the whole of what a literal may be; anything else is refused
    fail-closed and the rule is bucketed UNSUPPORTED-SHAPE."""


#: the longest literal read: a fraction of two 4,300-digit components, signs and parentheses
_MAX_LITERAL_CHARS = 2 * 4300 + 8

try:  # THE TABLE: each mode's values, defined once in the compiled core (`rust/table.rs`)
    from simplipy import _core as _table  # type: ignore[attr-defined]
    _TABLE_OK = hasattr(_table, 'TableExpr')
except Exception:  # pragma: no cover  (missing/unbuilt extension)
    _table, _TABLE_OK = None, False


def literal_value(t):
    """The exact value a literal token denotes, as THE TABLE reads it in real
    (`_core.table_literal_real`): a decimal or `p/q` numeral as the exact `Fraction` it spells
    (`-0.1` is -1/10, not the double 0.1000000000000000055511151231257827), `np.pi` and
    `np.e` as their constants, the `float("...")` spellings as the infinities and NaN. Total:
    a value or a refusal (`UnsupportedToken`), never an `eval`. A rule file is untrusted input,
    so the spelling is bounded before anything is built: more than 4,300 digits or a decimal
    exponent beyond 1,000,000 is refused (every spelling the engine emits reads: it keeps a
    written `2.827881176741212e-244756` as it is)."""
    if len(t) > _MAX_LITERAL_CHARS:
        raise UnsupportedToken(f'literal too long to read: {t[:40]!r}...')
    spec = _table.table_literal_real(t)
    if spec is None:
        raise UnsupportedToken(f'not an accepted literal: {t!r}')
    kind, a, b = spec
    if kind == 'q':
        return Fraction(int(a), int(b))
    if kind == 'd':  # m * 10^k, an exponent past the exact readers' 4,000 (at most 1e6)
        k = int(b)
        return Fraction(int(a)) * (Fraction(10) ** k)
    if kind == 'pi':
        return math.pi
    if kind == 'e':
        return math.e
    if kind == 'inf':
        return math.inf if a == '+' else -math.inf
    return math.nan


# ---------------------------------------------------------------- parse
def _constant_names(tokens, cname, number):
    """The name of each `<constant>` of `tokens`, in order: `cname` for all of them, or with
    `number` one symbol each (`<C_L>` gives `<C_L0>`, `<C_L1>`, ...: every constant is its own
    symbol, specification 3.2)."""
    names, k = [], 0
    for t in tokens:
        if t == '<constant>':
            names.append(f'{cname[:-1]}{k}>' if number else cname)
            k += 1
    return names


def parse(tokens, cname='<C>', arity=None, number=False):
    arity = ARITY if arity is None else arity
    consts = iter(_constant_names(tokens, cname, number))
    toks = [next(consts) if t == '<constant>' else t for t in tokens]
    pos = 0

    def rec():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t in arity:
            return (t,) + tuple(rec() for _ in range(arity[t]))
        if SLOT_RE.match(t) or t.startswith('<C'):
            return ('slot', t)
        if t in ('x0', 'x1', 'x2'):
            return ('slot', '?' + t[1:])
        return ('lit', literal_value(t), t)

    tree = rec()
    if pos != len(toks):
        raise ValueError('trailing tokens')
    return tree


def slots_of(tokens, cname='<C>', number=False):
    out = {}
    consts = iter(_constant_names(tokens, cname, number))
    for t in tokens:
        m = SLOT_RE.match(t)
        if m:
            out[t] = m.group(1)
        elif t in ('x0', 'x1', 'x2'):
            out['?' + t[1:]] = '?'
        elif t == '<constant>':
            out[next(consts)] = 'c'
    return out


# ---------------------------------------------------------------- contract evaluator
NAN, PINF, NINF = mpf('nan'), mpf('inf'), mpf('-inf')


class TV:
    """A value THE TABLE carries exactly -- a rational, a rational multiple of pi, e, an infinity
    or undefined -- as its interchange spec `(kind, a, b)` (`table::val_spec`). The judge binds
    the sorts' atoms and the constant battery as `TV`s, so a point the table decides is decided
    (`asin(1)` is exactly pi/2, and `tan` there a pole). `float()` is its nearest double, `+v`
    is `v`, and arithmetic is mpmath's on its reading at the current precision."""
    __slots__ = ('spec',)

    def __init__(self, spec):
        self.spec = tuple(spec)

    def mpf(self):
        return _from_spec(self.spec)

    def __float__(self):
        kind, a, b = self.spec
        if kind == 'q':
            return float(Fraction(int(a), int(b)))
        if kind == 'inf':
            return math.inf if a == '+' else -math.inf
        if kind == 'nan':
            return math.nan
        with mp.workdps(40):
            return float(_from_spec(self.spec))

    def __pos__(self):
        return self

    def __neg__(self):
        return -self.mpf()

    def __repr__(self):
        return f'TV{self.spec}'

    def __eq__(self, other):
        return isinstance(other, TV) and self.spec == other.spec

    def __hash__(self):
        return hash(self.spec)

    def _other(self, o):
        return o.mpf() if isinstance(o, TV) else o

    def __add__(self, o):
        return self.mpf() + self._other(o)

    def __radd__(self, o):
        return self._other(o) + self.mpf()

    def __sub__(self, o):
        return self.mpf() - self._other(o)

    def __rsub__(self, o):
        return self._other(o) - self.mpf()

    def __mul__(self, o):
        return self.mpf() * self._other(o)

    def __rmul__(self, o):
        return self._other(o) * self.mpf()

    def __truediv__(self, o):
        return self.mpf() / self._other(o)

    def __rtruediv__(self, o):
        return self._other(o) / self.mpf()

    def __lt__(self, o):
        return self.mpf() < self._other(o)

    def __gt__(self, o):
        return self.mpf() > self._other(o)


def tv_of_float(x):
    """A double as the exact value it is (`Fraction(x)`), or its infinity / NaN."""
    x = float(x)
    if x != x:
        return TV(('nan', '', ''))
    if math.isinf(x):
        return TV(('inf', '+' if x > 0 else '-', ''))
    q = Fraction(x)
    return TV(('q', str(q.numerator), str(q.denominator)))


def _spec(v):
    """A binding's interchange spec: a `TV` as it is, an mpf as its exact binary value, a
    float, int or Fraction as the exact number it is."""
    if isinstance(v, TV):
        return v.spec
    if isinstance(v, (Fraction, int)) and not isinstance(v, bool):
        q = Fraction(v)
        return ('q', str(q.numerator), str(q.denominator))
    if not isinstance(v, type(mpf(0))):
        return tv_of_float(v).spec
    if misnan(v):
        return ('nan', '', '')
    if misinf(v):
        return ('inf', '+' if v > 0 else '-', '')
    sign, man, exp, _bc = v._mpf_
    if man == 0:
        return ('q', '0', '1')
    return ('m', ('-' if sign else '') + format(int(man), 'x'), str(exp))


def _from_spec(spec):
    """A spec's value as an mpf at the current precision."""
    kind, a, b = spec
    if kind == 'q':
        return mpf(int(a)) if b == '1' else mpf(int(a)) / mpf(int(b))
    if kind == 'pi':
        return mpf(int(a)) * mp.pi / mpf(int(b))
    if kind == 'e':
        return +mp.e
    if kind == 'inf':
        return PINF if a == '+' else NINF
    if kind == 'nan':
        return NAN
    if kind == 'd':
        return mpf(int(a)) * mpf(10) ** int(b)
    return mp.ldexp(mpf(int(a, 16)), int(b))


def _tree_tokens(tree, names, extra):
    """`tree` back as prefix tokens for the table: a slot is its name, a literal its spelling
    (or, built without one, its exact value), and a literal holding an mpf a hidden name whose
    value rides in `extra`."""
    op = tree[0]
    if op == 'slot':
        names.setdefault(tree[1], None)
        return [tree[1]]
    if op == 'lit':
        if len(tree) > 2:
            return [tree[2]]
        v = tree[1]
        if isinstance(v, float) and v == math.pi:
            return ['np.pi']
        if isinstance(v, float) and v == math.e:
            return ['np.e']
        if isinstance(v, (Fraction, int, float)) and not isinstance(v, bool):
            if isinstance(v, float) and (v != v or math.isinf(v)):
                return ['float("nan")' if v != v else
                        ('float("inf")' if v > 0 else 'float("-inf")')]
            q = Fraction(v)
            return [f'{q.numerator}/{q.denominator}']
        name = f'<lit{len(extra)}>'
        extra[name] = v
        names.setdefault(name, None)
        return [name]
    out = [op]
    for child in tree[1:]:
        out += _tree_tokens(child, names, extra)
    return out


_COMPILED: dict = {}


def _compiled(tree):
    """`tree` compiled against the table (`_core.TableExpr`), with its free names in order and
    the values of its hidden literal names. Cached per tree."""
    hit = _COMPILED.get(tree)
    if hit is None:
        names, extra = {}, {}
        tokens = _tree_tokens(tree, names, extra)
        expr = _table.TableExpr(tokens, list(names), dict(ARITY))
        hit = (expr, tuple(names), extra)
        if len(_COMPILED) > 50_000:
            _COMPILED.clear()
        _COMPILED[tree] = hit
    return hit


def c_eval(tree, env):
    """Evaluate under the contract semantics: THE TABLE's real values (`_core.TableExpr`), at the
    current precision. `env` binds each slot to a `TV`, an mpf or a number. Raises `Unresolved`
    where the table cannot judge the point honestly at this precision; records a snap
    (`SNAP_EVENTS`) and the intermediates' magnitudes (`_MAG_SINK`) for the ladder."""
    expr, names, extra = _compiled(tree)
    args = [_spec(extra[n] if n in extra else env[n]) for n in names]
    spec, unresolved, snapped, inf_node, mag, small = expr.eval_real(args, mp.dps)
    if snapped:
        SNAP_EVENTS[0] += 1
    if _MAG_SINK:
        m, sm = _from_spec(mag), _from_spec(small)
        if m > _MAG_SINK[-1][0]:
            _MAG_SINK[-1][0] = m
        if sm != 0 and (_MAG_SINK[-1][1] == 0 or sm < _MAG_SINK[-1][1]):
            _MAG_SINK[-1][1] = sm
    if unresolved:
        raise Unresolved()
    return _from_spec(spec)


def _table_reading(tree, env, dps):
    """The table's value of `tree` at `env` and `dps` in its float-resolution reading
    (`TableExpr.eval_float_resolution`: a saturation inside the boundary band reads as the
    working float's value, `tanh(cosh(-10))` as 1 to fifty digits), the value P3 rounds to a
    double; None where the table cannot read it at this precision."""
    expr, names, extra = _compiled(tree)
    args = [_spec(extra[n] if n in extra else env[n]) for n in names]
    spec = expr.eval_float_resolution(args, dps)
    return None if spec is None else _from_spec(spec)


#: Filled by `c_eval` when a `_measured_dps` scope is open; see `_required_dps`.
_MAG_SINK: list = []


def _required_dps(mag, small=None):
    """Precision that a cancellation between intermediates of size `mag` (the largest)
    and `small` (the smallest nonzero) can demand.

    Adding two numbers of magnitude 10^k whose true sum is 10^-k destroys about 2k
    significant digits, so the working precision has to carry them before any correct
    digit survives. Measured against the case that motivated this: at t = -20,
    `cosh(25t) + sinh(25t)` has intermediates near e^500 ~ 10^217, and the sum only
    becomes representable somewhere past dps 400 -- which is what `2 * 217 + 50` gives.

    A SMALL INTERMEDIATE COSTS THE SAME AS A LARGE ONE, and for the same reason: it is
    the RATIO the precision has to carry, not the size. `cos x = 1 - x^2/2 + ...` at
    x = exp(-300) = 5.1e-131 puts the whole content of the answer 261 digits below the 1
    it sits next to, so the output is exactly 1.0 at dps 50, at dps 120 and at dps 250,
    and `acos` of that is 0 at all three. Three rungs reading a manufactured 0 is not
    three opinions; it convicted `acos(cos(exp(c))) -> c_t`, true for every c by
    construction. Sized from the small end, rung 2 lands at 312 and the identity
    resolves.

    Derived, not tuned: the 2 is the two-sided digit loss, `k` is measured from the
    actual intermediates rather than guessed from the expression, and `BASE_DPS` is the
    ordinary working precision that must survive on top.
    """
    return min(MAX_DPS, _demanded_dps(mag, small))


def _demanded_dps(mag, small=None):
    """`_required_dps` BEFORE the ceiling: what the point asks for, not what it gets.

    The two differ exactly where the ceiling binds, and there the difference is the
    whole finding -- a rung clamped to `MAX_DPS` is not the rung that was asked for, and
    reading it as though it were is how the judge convicted past its own evidence.
    """
    k = 0
    try:
        if mag is not None and mag > 1:
            k = max(k, int(mp.floor(mp.log10(mag))) + 1)
        if small is not None and 0 < small < 1:
            k = max(k, -int(mp.floor(mp.log10(small))))
    except Exception:
        return BASE_DPS
    return BASE_DPS + 2 * max(0, k)


# ---------------------------------------------------------------- deployed evaluator
#: THE f64 STANDARD: f64 promises the evaluator's value wherever nothing overflows or
#: underflows. The constructors already work this way: `x1*x2/x2 -> x1`,
#: `exp(x1)/exp(x1) -> 1` and `rootn(exp(800.5), 2) -> exp(1601/4)` give the TRUE value where an
#: intermediate leaves the float range. So at a sample point where an operation of EITHER side
#: overflows or underflows, a rewrite may give the evaluator's value or the true one: wherever
#: the contract finds the two sides equal at such a point, the deployed lane reads it as
#: agreement without comparing the floats. A rewrite that gives neither value at a range event
#: (`exp(1000) -> 5`) is still refused, and one that reproduces the evaluator there
#: (`exp(1000) -> inf`) is still realised. The events are IEEE 754's and defined once, in the
#: core (`numeric::RangeKind`): an overflow (finite operands and finite exact value, float
#: +-inf: `exp(800)`) or an underflow (nonzero exact value, float zero or subnormal:
#: `exp(-800)`); neither a pole (`1/0`, `log(0)`, `atanh(1)`) nor an exact zero (`x - x`) is
#: one, so `atanh(tanh(20))` is compared: `tanh` saturates to 1.0 inside the range.


def d_eval(tree, env, in_range=None):
    """Evaluate under the deployed engine's float64 algebra: THE TABLE's f64 values (the core's
    own kernels, each literal the double nearest to it, a written `-0.0` keeping its sign).

    With `in_range` (a one-element list), an operation that overflows or underflows (the range
    standard, `numeric::RangeKind`) sets `in_range[0] = False`."""
    expr, names, extra = _compiled(tree)
    v, ok = expr.eval_f64([float(extra[n] if n in extra else env[n]) for n in names])
    if in_range is not None and not ok:
        in_range[0] = False
    return F(v)


# ---------------------------------------------------------------- comparison
def cls_mp(v):
    if misnan(v):
        return ('nan',)
    if misinf(v):
        return ('inf', 1 if v > 0 else -1)
    return ('fin', v)


def cls_np(v):
    v = F(v)
    if np.isnan(v):
        return ('nan',)
    if np.isinf(v):
        return ('inf', 1 if v > 0 else -1)
    return ('fin', float(v))  # quotient zeros: -0 == 0


#: The gap ladder for the FINITE case. Rung 1 is `BASE_DPS`; the rest climb far enough
#: that a residue's decay is unmistakable beside an analytic gap's stability.
GAP_RUNGS = (BASE_DPS, 120, 250)

#: Residual target for the exists-witness fit. Deeper than the deepest rung, so a fitted
#: `<C_R>` is never the thing the decay test is measuring.
_WIT_RESID = mpf(10) ** (-(250 + 20))


def _kind(c):
    """A class tuple's kind as the table names it: 'nan', '+inf', '-inf' or 'fin'."""
    if c[0] == 'inf':
        return '+inf' if c[1] > 0 else '-inf'
    return c[0]


def _num(c):
    """A class tuple's float."""
    if c[0] == 'nan':
        return math.nan
    if c[0] == 'inf':
        return math.inf if c[1] > 0 else -math.inf
    return float(c[1])


def compare_classes(cl, cr):
    """The class half of a contract comparison -- everything nan/inf settles by itself
    (`_core.table_compare_classes`, THE TABLE's comparison).

    -> 'eq' | 'EXT' (nan on the left: an extension) | 'SHRINK' (nan on the right: a loss) |
    'INF-CHANGE', or None when BOTH sides are finite, where the gap ladder decides instead.
    """
    return _table.table_compare_classes(_kind(cl), _kind(cr))


def gap_reading(cl, cr):
    """One rung's FINITE reading -> `(|l - r|, scale)`, both ABSOLUTE.

    THIS USED TO RETURN THE QUOTIENT, AND AGAINST AN EXACT ZERO THE
    QUOTIENT IS 1.0 AT EVERY PRECISION. The decay test compares this reading at two
    rungs, so any normaliser that is the SAME at both cancels out of the ratio and
    cannot change a verdict. `max(|l|, |r|)` is not the same at both: when one side is
    exactly 0 and the other is pure precision residue, the scale IS the residue, the
    quotient is 1.0 by construction, and a gap that cannot move reads as an analytic
    one. Measured over 85 true zero-side identities, that convicted 25.

    So the separation is reported RAW and the scale beside it, leaving the ladder to
    fix ONE scale across both rungs -- where, being rung-invariant, it can only set the
    zero-floor and never manufacture a ratio of its own.
    """
    a, b = mpf(cl[1]), mpf(cr[1])
    return abs(a - b), max(abs(a), abs(b))


# THE DECAY TEST: THE FINITE CASE HAS NO MAGNITUDE BAR AT ALL. It asks how the gap
# RESPONDS TO PRECISION, which is the only question that separates the two populations.
#
# Every fixed tolerance was the wrong instrument, and the artifact proves it by ordering.
# On acj-4-3 the EXACT identity `cos asin tanh _0 -> inv cosh _0` shows a dps-50 gap of
# 1.8e-26, while the FALSE `tanh pow np.pi 3 -> 1` shows 2.3e-27 -- the true rule's gap is
# the LARGER one. Any bar between them convicts the identity and spares the falsehood.
# Nor is the false family clustered: `tanh(x) -> 1` is false at every finite x by exactly
# 2/(e^2x+1), a smooth continuum measured from 1.8e-26 (x=30) through 2.5e-47 (x=54) and
# below. There is no valley to put a threshold in, because there is no threshold there.
#
# What separates them is decay. Residue is a fact about the ARITHMETIC and falls with the
# working precision; an analytic gap is a fact about the FUNCTIONS and does not move.
# Measured over dps 50 -> 400: the tanh rows above are bit-identical at all four rungs,
# while every true identity's gap falls ~10^350. The bar is half the ADDED precision, and
# the two populations clear it by >=30 decades on either side -- a separator, not a knob.
#
# Crucially, the cancellation depth cancels out of a RATIO. `log(cosh y + sinh y) -> y` at
# y = -50 -- exactly true, and an identity a magnitude bar wrongly KILLs -- carries a
# dps-50 residue of 8.5e-21 that ANY magnitude bar convicts; its gap still falls 10^69 by
# dps 120, so the ratio spares it. That is the failure the old comment called NOT YET
# FIXED, and it is what a size test structurally cannot see.
#
# This supersedes both earlier attempts. An ABSOLUTE 1e-25 floor is gone because it
# read `e**sinh(-5) -> 0` and `asin(1e-8) -> 1e-8` as equal; the RELATIVE 1e-25 that
# replaced it still spared the whole tanh saturation family. Both are gone. `asin(1e-8) ->
# 1e-8` and `sin(1e-3) -> 1e-3` now convict on decay, not on size.
#
# Known limit, stated rather than hidden: a gap below the deepest rung's resolution is
# invisible, so `tanh(400) -> 1` (gap 1e-348) still reads eq. That is the "saturated at P
# is extendable, not wrong" case `hiprec.rs` already documents; it ships today too.


#: Bounded functions, and the values they NEVER attain at a FINITE argument. Every
#: bound here is exactly representable, which is the whole point: the test below must
#: not need a precision to reach its answer.
SATURATING = {
    'tanh': (-1.0, 1.0),   # |tanh a| < 1 strictly, for every finite a
    'exp': (0.0,),         # exp a > 0 strictly, for every finite a
    'inv': (0.0,),         # 1/a reaches 0 only in the limit
    'erf': (-1.0, 1.0),    # |erf a| < 1 strictly, for every finite a
}


def saturation_verdict(tl, tr, env_mp):
    """A BOUNDED FUNCTION NEVER ATTAINS ITS LIMIT AT A FINITE ARGUMENT, so a
    rewrite claiming it does is false -- by a fact about the FUNCTION, at no precision
    at all. -> 'REAL-CHANGE' when the claim is refuted, else None (no opinion).

    This is the family the gap ladder cannot reach, and depth is not the answer.
    `tanh(cosh(10))` differs from 1 by 2e-9566, inside the boundary band at every
    rung anyone can afford, so the honest numeric verdict is Unresolved forever;
    `tanh(exp(10))` differs by 1e-19132. Measured, one more rung bought ten rows for
    ~3x the cost of EVERY comparison. A symbolic fact costs nothing and settles all of
    them.

    Consulted only where the ladder has already refused, so it can add a verdict but
    never change one.

    TWO CONDITIONS, both required, both exact. The argument must be FINITE -- at an
    infinite one the limit IS attained (`tanh(inf) = 1`, `exp(-inf) = 0`) and the claim
    is true. And the bound must be WRITTEN, as a literal or a slot's bound value, never
    computed: `1 - 2*exp(-2*cosh(10))` IS `tanh(cosh(10))`, and it rounds to exactly 1.0
    at any affordable precision, so a computed side would convict an identity. That is
    the same literal-provenance doctrine `c_eval` states for `atanh`/`acosh`.
    """
    for src, tgt in ((tl, tr), (tr, tl)):
        bounds = SATURATING.get(src[0]) if len(src) == 2 else None
        if not bounds:
            continue
        snap0 = SNAP_EVENTS[0]
        try:
            arg = c_eval(src[1], env_mp())
            if tgt[0] == 'lit':
                lim = c_eval(tgt, env_mp())
            elif tgt[0] == 'slot' and tgt[1].startswith('<C_R'):
                # a FITTED witness is no written bound: `erf(30) -> c_t` has the real witness
                # 1 - 2.6e-393, which no float64 fit can carry, so the fit reads 1.0 and this
                # test convicted a rule true by construction
                continue
            elif tgt[0] == 'slot':
                lim = env_mp()[tgt[1]]
            else:
                continue                     # computed: provenance is not a written bound
        except (Unresolved, KeyError):
            continue
        if SNAP_EVENTS[0] != snap0:
            continue                         # a snapped value is a DIFFERENT point
        if misnan(arg) or misinf(arg) or misnan(lim) or misinf(lim):
            continue                         # the limit is attained at an infinite arg
        if any(lim == mpf(b) for b in bounds):
            return 'REAL-CHANGE'
    return None


#: The deployed-realisation bound, in ULP (`table::REALISED_ULP`): a rewrite has two sides, each a
#: composition of up to about four library calls of at most one ULP of error, which can oppose:
#: 2 x 4 x 1 = 8.
REALISED_ULP = int(_table.TABLE_REALISED_ULP) if _TABLE_OK else 8


def _ulp_gap(a, b):
    """ULP distance between two finite doubles of the same sign (the two zeros as one); inf
    across a sign change (`_core.table_ulp_gap`)."""
    g = _table.table_ulp_gap(float(a), float(b))
    return float('inf') if g is None else g


def compare_deployed_realised(cl, cr):
    """The REALISATION comparison, THE TABLE's f64 agreement (`_core.table_f64_agree`, F6):
    both undefined, the same infinity, or both finite and at most `REALISED_ULP` units in the
    last place apart, the two zeros counting as one value. It decides `realised`, and so the
    `f64` tier. Why 8: over the 390 acj-4-3 rules bit-exactness moves, the gap distribution is
    4 at 0 ULP, 310 at 1, then 5/2/4/0/2/2 at 2..7 and nothing until 57, so every bound in
    [8, 56] gives the same partition."""
    return _table.table_f64_agree(_num(cl), _num(cr))


def compare_deployed_structural(cl, cr):
    """The STRUCTURAL deployed comparison, for ENGINE-MISALIGN only (`_core.table_f64_structural`):
    has the deployed algebra DIVERGED -- an infinity where a finite value belongs, a nan where a
    number belongs, or a gap beyond 1e-9 of the larger magnitude? A few ULP is no divergence."""
    return _table.table_f64_structural(_num(cl), _num(cr))


# ---------------------------------------------------------------- batteries
#: THE TABLE's sort tags, as the judge names them: a real binding where clause (a) applies; a
#: non-finite binding in the sort's domain, judged with no excuse; one its certificate puts on a
#: set of measure zero, owed nothing.
_TAGS = {'real': 'real', 'atom': 'atom', 'null': 'nullx'}


def battery_for(sort, permissive=False):
    """-> list of (builder() -> TV, tag) over THE TABLE's quantifier of `sort`
    (`_core.table_sort_atoms`): `?` binds real data only; `!` adds +-inf, owed nothing (its
    certificate puts them on a null set); `$` drops 0 from the reals and adds 0 and +-inf, owed
    nothing; `_` adds +-inf and undefined, judged with no excuse, since a subexpression can be
    infinite or undefined on a whole interval. `permissive` keeps the real-valued bindings only
    (its guarantee)."""
    return [((lambda s=spec: TV(s)), _TAGS[tag])
            for spec, tag in _table.table_sort_atoms(sort, permissive)]


#: The battery slot-product cap: the size of the FULL two-slot product, derived from
#: the batteries themselves rather than written down, so that adding a battery point
#: cannot silently turn an exhaustive two-slot sweep back into a sample.
BATTERY_CAP = max(len(battery_for(s)) for s in '?_!$') ** 2


CONSTS = [2.5, -1.5, 3.0, 0.5, -0.7]      # witness-fitting battery (generic values)


#: The f64 SATURATION magnitudes: where the deployed evaluator ATTAINS a bound that
#: mathematics only approaches. `tanh(19)` is exactly 1.0 in f64; `cosh(exp(-20))` and
#: `cosh(inv(1e17))` round to exactly 1.0; `exp(-750)` underflows to 0.0 and `cosh(750)`
#: overflows to inf. A rule that RELIES on non-attainment -- `acosh(neg(tanh(c)))` is nan
#: only because |tanh| < 1 -- is true mathematics and false in f64 at exactly these
#: magnitudes, and the ordinary battery (|c| <= 5, pi, e) cannot see it. Both signs: the
#: quantifier ranges over R and a one-sided probe is not a probe of R.
SATURATION_CONSTS = [c * s for c in (19., 20., 50., 750., 1e17) for s in (1., -1.)]


#: The GENERIC magnitude span, decade-spaced, both signs.
#: `CONSTS` above reaches |c| <= 5, and the MINER does not: the Rust side
#: sweeps `POLE_GRID` to 500 and draws its fitted constants log-uniform over [1e-3, 1e3],
#: so the quantifier explored a span the gate never checked. A `forall c over R` that
#: only ever looks inside one decade is not a check of R; it is a check of [-5, 5].
#:
#: Deliberately NOT the saturation set (that is `SATURATION_CONSTS`, which probes where
#: f64 ATTAINS a bound) and not a denser re-run of |c| <= 5 -- more points inside a span
#: already covered buy nothing. This is SPAN, not resolution.
#:
#: Measured against the shipped acj-4-3 file with
#: the old ladder it convicted 97 rows and failed to witness 2 more, and every one was an
#: instrument artifact -- six distinct defects. After
#: those fixes: 0 convictions, 0 unresolved.
WIDE_CONSTS = [c * s for c in (10., 30., 100., 300., 1e3, 3e3, 1e4, 0.1, 0.01, 0.001)
               for s in (1., -1.)]


def judge_cl_battery(saturation=True):
    """judging battery for a SOURCE-side constant (forall c_s over R). Includes the
    special rationals (pow(1,nan)=1 class) AND the SYMBOLIC transcendental atoms,
    built at the CURRENT dps: a fitted constant reaches pi/2, and pow(sin(c),inf)
    at c = pi/2 is the powsin class in constant space -- an f64 pi/2 misses the
    coincidence by 5e-33 (a false rescue can otherwise slip through; the symbolic
    constant atoms kill it correctly). A cl whose witness is unfittable (degenerate
    LHS there) is skipped with a tally; only core-CONSTS failures mean NO-WITNESS.

    The `saturation` arm carries BOTH the f64-attainment magnitudes and the generic
    span out to |c| = 1e4 (`WIDE_CONSTS`), because they share a failure mode and
    therefore share the caller's withholding rule: where a TARGET constant has to be
    re-fitted at the probe, the exists-witness machinery cannot resolve it, and the fit
    returns the edge of an interval instead of a point. At `pow(abs ?0, 10000)` with
    `?0 = -0.8` the deployed LHS underflows to exactly 0.0, so `0.8^c - 0.0` vanishes for
    every c past ~3339 and the fit answered 3339.25 against a true 10000 -- an IDENTITY
    convicted. `judge_rule` withholds this arm exactly where that can happen and records
    that it did (`saturation_probed`)."""
    base = ([(lambda c=c: mpf(c)) for c in CONSTS] +
            [(lambda: mpf(1)), (lambda: mpf(-1)), (lambda: mpf(0)),
             (lambda: mpf(2)), (lambda: mpf(-2)), (lambda: mpf(4)),
             (lambda: mpf(-4)), (lambda: mpf(5)), (lambda: mpf(-5)),
             (lambda: mp.pi / 2), (lambda: -mp.pi / 2), (lambda: mp.pi),
             (lambda: mp.e)])
    base = base + [(lambda c=c: mpf(c)) for c in WIDE_CONSTS]
    if not saturation:
        return base
    return base + [(lambda c=c: mpf(c)) for c in SATURATION_CONSTS]
#: The generic grid: more points of clause (a), past the battery's magnitudes, and the deployed
#: lane's reach to where float64 parts from the reals. Never a measure (clause (b) is decided by
#: certified cells).
GRID = np.concatenate([np.linspace(-3, 3, 121), np.linspace(-20, 20, 41),
                       np.array([-1e4, -1e3, -300., -100., -50., -30.,
                                 30., 50., 100., 300., 1e3, 1e4])]) + 0.0137421
# dense center (0.05 spacing: sub-unit undefined intervals like (0.2, 0.8) are visible),
# unit-spaced mid-range, and a magnitude tail (half-line violations opening at |x|>20,
# e.g. exp-shift constructions)
# DEDUPED: `linspace(-3,3,121)` and `linspace(-20,20,41)` overlap, so seven core
# values (-2.9862579, -1.9862579, -0.9862579, 0.0137421, 1.0137421, 2.0137421, 3.0137421)
# were double-weighted -- 174 entries for 167 distinct points, silently giving those seven
# twice the say in `meas`. Order preserved; the point SET is unchanged.
GRID = list(dict.fromkeys(GRID))

# ---------------------------------------------------------------- witness fitting
GEN = [1.7, -1.3, 0.6, 2.3, -0.8]


def _gen_env(names, k):
    return {n: F(GEN[(k + i) % len(GEN)]) for i, n in enumerate(sorted(names))}


#: The COARSE TAIL of the exists-witness search grid, out to the f64 ceiling. A witness
#: only has to be BRACKETED on this grid -- 300 bisection steps do the rest -- and the
#: dense body below stops at 1e12, which cannot bracket the perfectly ordinary
#: constant-folds `pow(exp(9), 5) = 3.5e19` or `pow(cosh(7), 5) = 5.0e13`. 15 rules of
#: the shipped `rules_real.json` were reported witness-less for that reason alone, and
#: `skipped_cl` swallowed the report. Density is not delicate: measured over the whole
#: 296-rule constant population, every tail from 1 point per 12 decades to 1 per decade
#: recovers the SAME 27 fits and moves NO witness that already fitted.
#: ...AND IT RUNS TO THE f64 CEILING, not to a round 1e300. A search that stops eight
#: decades short has no bracket for `cosh(-3)^300 = 7.6e300`, and a rule whose witness
#: cannot be bracketed is reported NO-WITNESS -- a conviction, for a row true by
#: construction.
_XS_TAIL = np.logspace(12, 308, 49)[1:]

#: The dense body is the shipped grid, unchanged -- the tail only adds reach.
_XS = np.concatenate([-_XS_TAIL[::-1], -np.logspace(12, -3, 120),
                      np.linspace(-3, 3, 81), np.logspace(-3, 12, 120), _XS_TAIL])


def _cl_env(cl_val, cl_name='<C_L>'):
    """The left side's constants as a float environment: a dict of them as it is, a single value
    as `cl_name`'s."""
    if cl_val is None:
        return {}
    if isinstance(cl_val, dict):
        return {n: F(v) for n, v in cl_val.items()}
    return {cl_name: F(cl_val)}


def _fit_target_env(tl, tr, shared, cl_val=None, rname='<C_R>'):
    """The generic env at which a witness can be DETERMINED, with its residual scanned.

    Returns `(env0, L0, g, vals, ok)`, or None if no env determines anything.

    TWO CONDITIONS, where there used to be one. The deployed LHS must be FINITE there --
    as before -- AND the residual `rhs(c) - L0` must vanish at no more than one grid
    point. The second is what makes the widened constant battery safe: where the deployed
    LHS has SATURATED, the equation holds on an INTERVAL, and an interval names no
    constant. `pow(abs ?0, 10000)` is inf at `?0 = 1.7` and at `-1.3`, and underflows to
    exactly 0.0 at `-0.8` -- the only finite env -- so `0.8^c - 0.0` is zero for every c
    past ~3339. The old code took the first finite env, fitted the edge of that interval
    at 3339.25 against a true 10000, and the judge convicted an IDENTITY.

    EVERY candidate env is tried, not just the first finite one: indeterminacy is a
    property of the env, and another env may still pin the witness. When none does, the
    cl is degenerate for fitting -- which is exactly what `skipped_cl` records, and why
    `judge_rule` asks THIS function rather than re-deriving "is the LHS defined" from
    finiteness alone (a saturated LHS is finite and determines nothing).
    """
    cl_env = _cl_env(cl_val)

    def lhs_at(env):
        return d_eval(tl, {**env, **cl_env})

    def rhs_at(env, c):
        return d_eval(tr, {**env, **cl_env, rname: F(c)})

    for k in (0, 1, 4):
        try:
            env0 = _gen_env(shared, k)
            L0 = lhs_at(env0)
        except Exception:
            continue
        if not source_defined(L0):
            continue

        def g(c, _env0=env0, _L0=L0):
            try:
                v = rhs_at(_env0, c)
                return v - _L0 if np.isfinite(v) else np.nan
            except Exception:
                return np.nan

        vals = np.array([g(x) for x in _XS])
        ok = np.isfinite(vals)
        if int(np.count_nonzero(ok & (vals == 0))) > 1:
            continue          # the residual vanishes on an interval: nothing to pin
        return env0, L0, g, vals, ok
    return None


def fit_witness(tl, tr, shared, cl_val=None, env_out=None, rname='<C_R>'):
    """fit the exists-witness for a RHS <constant> on generic reals (deployed algebra);
    validated on two further generic envs; None if no witness exists/found. With `env_out` (a
    list), the generic env the witness was determined at is appended to it, for `mp_polish`.
    `cl_val` binds the left side's constants (a dict by name, or one value for `<C_L>`), and
    `rname` names the right side's constant."""
    cl_env = _cl_env(cl_val)

    def lhs_at(env):
        return d_eval(tl, {**env, **cl_env})

    def rhs_at(env, c):
        return d_eval(tr, {**env, **cl_env, rname: F(c)})

    found = _fit_target_env(tl, tr, shared, cl_val, rname)
    if found is None:
        return None
    env0, L0, g, vals, ok = found
    if env_out is not None:
        env_out.append(env0)
    best = None
    if tr == ('slot', rname):
        # BARE-<C_R> SHORTCUT: the whole target IS the
        # constant, so the residual is g(c) = c - L0 exactly -- linear, monotone,
        # witness c_t = L0, nothing to search. 272 of the 343 constant-carrying
        # shipped rules have this shape, and each was paying the 500-point bracket
        # scan and up to 300 bisections per (rule, cl) to approximate the number the
        # identity hands over. Beyond the cost, the scan machinery is where the
        # first-bracket and sign-of-zero misfires live -- this shape cannot reach
        # them any more. Every downstream check still applies: the snap candidates,
        # the two-env validation, and the caller's mp polish.
        best = float(L0)
    for i in range(len(_XS) - 1) if best is None else ():
        if ok[i] and vals[i] == 0:
            best = _XS[i]
            break
        if ok[i] and ok[i + 1] and np.sign(vals[i]) != np.sign(vals[i + 1]):
            lo, hi, vlo = _XS[i], _XS[i + 1], vals[i]
            for _ in range(300):
                mid = 0.5 * (lo + hi)
                vm = g(mid)
                if not np.isfinite(vm):
                    break
                if np.sign(vm) == np.sign(vlo):
                    lo, vlo = mid, vm
                else:
                    hi = mid
            best = 0.5 * (lo + hi)
            break
    if best is None:
        # THE SEARCH MUST INCLUDE THE CONSTANT THE QUANTIFIER IS ASKING ABOUT. `_XS` is a
        # fixed global grid, and its density collapses exactly where the deployed algebra
        # saturates: `pow(abs ?0, c)` at `?0 = -0.8` overflows below c ~ -3178, so every
        # finite grid point near the true witness -3000 lies on ONE side of it and the
        # bracketing finds no sign change at all. For a rule whose source carries a
        # constant, `c_R = c_L` is the first candidate worth trying -- it is the answer
        # for every identity -- and it costs one residual evaluation. It weakens nothing:
        # the candidate still has to satisfy the same residual bound and the same
        # two-env validation as any other. (A lane that cannot
        # evaluate where the question is asked answers a different question.)
        seeds = [float(v) for v in cl_env.values()]
        for c in seeds + [1., -1., 2., -2., 0.5, -0.5, math.e, -math.e,
                          math.pi, -math.pi]:
            v = g(c)
            if np.isfinite(v) and abs(v) <= 1e-9 * max(1.0, abs(L0)):
                best = c
                break
    # A NON-FINITE FIT IS NOT A WITNESS. `<constant>` stands for a value a FIT will
    # supply, i.e. an f64 -- that scoping is the whole basis of `c_free`'s derivation --
    # and no fit produces an infinity. The bisection can land on one when both ends
    # overflow, and `round(inf)` then raises OverflowError out of the middle of the
    # judge. Found by widening the constant battery to |c| = 1e4: at
    # `pow(exp(9), c)` the deployed LHS is inf there, so the fit had nothing finite to
    # aim at. Returning None routes it to the ordinary no-witness tally, which is what a
    # cl with no fittable witness has always meant.
    if best is None or not np.isfinite(best):
        return None
    # ...and the SNAP CANDIDATES have to survive the same scoping. `round(best)` is fine
    # for a finite `best`, but `best * 2` overflows at the top of the f64 range and
    # `round(inf)` raises out of the judge -- so the half-integer candidate is only
    # offered when the doubling stays finite.
    snaps = [round(best)] if abs(best) < 2.0 ** 1023 else []
    if np.isfinite(best * 2):
        snaps.append(round(best * 2) / 2)
    for c in [1., -1., 2., -2., 3., -3., 0.5, -0.5, math.e, -math.e, math.pi, -math.pi,
              *snaps]:
        # RELATIVE snap only (1e-15 floor = f64 fit noise around an exact zero): the old
        # absolute 1e-6 tolerance flattened every tiny TRUE witness to round(best) = 0
        # (pow(exp(-5), pi) = 1.5e-7 snapped to 0), fabricating clause-(a) kills of
        # sound constant-space rules
        #
        # AND ONLY WHERE THE SNAPPED VALUE IS AT LEAST AS GOOD A WITNESS. A snap exists to
        # remove the fit's own noise (a bisection's 2.9999999999999996 for an exact 3); it
        # must never replace a witness the fit found exactly. `asinh(exp(-100)) -> c_t` fits
        # c_t = 3.7e-44, its own value, with residual 0, and the 1e-15 floor snapped it to 0;
        # `erf(5) -> c_t` fits 0.9999999999984626 and the relative band snapped it to 1. The
        # deployed lane then compared 0 and 1 against the source and found the true rule
        # unrealised. So a candidate is taken only where its residual is no larger.
        if abs(c - best) < max(1e-9 * max(abs(best), abs(c)), 1e-15):
            rc, rb = g(float(c)), g(best)
            if np.isfinite(rc) and (not np.isfinite(rb) or abs(rc) <= abs(rb)):
                best = float(c)
            break
    for k in (2, 3):
        env = _gen_env(shared, k)
        try:
            Lv, Rv = lhs_at(env), rhs_at(env, best)
        except Exception:
            return None
        if np.isfinite(Lv) != np.isfinite(Rv):
            return None
        if np.isfinite(Lv) and abs(Lv - Rv) > 1e-4 * max(1.0, abs(Lv)):
            return None
    return float(best)


def _norm(v):
    """The 2-norm of `v` with an exactly rounded sum: the same bits on every machine. A BLAS dot
    product sums in the order of the kernel the CPU dispatches, so `np.linalg.norm`, `@` and
    `np.linalg.solve` give different last bits on different runners, and a verdict that
    reads them depends on the machine (the CI saw it as a Python-version split)."""
    return math.sqrt(math.fsum(float(x) * float(x) for x in v))


def _solve(A, b):
    """`A x = b` for the fit's small dense system: Gaussian elimination with partial pivoting
    in plain float arithmetic (the same bits on every machine); None where `A` is singular."""
    n = len(b)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(M[i][col]))
        if not math.isfinite(M[piv][col]) or M[piv][col] == 0.0:
            return None
        M[col], M[piv] = M[piv], M[col]
        for i in range(col + 1, n):
            f = M[i][col] / M[col][col]
            for j in range(col, n + 1):
                M[i][j] -= f * M[col][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - math.fsum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return np.array(x)


def fit_witness_multi(tl, tr, shared, cl_val, rnames, env_out=None):
    """The exists-witness for SEVERAL right-side constants (`rnames`), each its own symbol:
    values d with `R(x; d) = L(x; c)` at the generic environments where the deployed left side
    is finite, found by damped Gauss-Newton from a few seeds (the left side's own constants
    first), snapped to a nearby simple value where that is no worse, and validated on two
    further environments as `fit_witness` validates. `None` if no witness is found; with
    `env_out`, the environments the fit used are appended to it, for `mp_polish_multi`.

    The arithmetic is the same on every machine (`_norm`, `_solve`, exactly rounded sums for
    the normal equations): a fit that reads a BLAS kernel's summation order lands a few ulps
    apart on different CPUs, and a verdict must not depend on the machine."""
    cl_env = _cl_env(cl_val)
    envs, targets = [], []
    for k in range(len(GEN)):
        env = _gen_env(shared, k)
        try:
            v = d_eval(tl, {**env, **cl_env})
        except Exception:
            continue
        if source_defined(v) and all(env != e for e in envs):
            envs.append(env)
            targets.append(float(v))
    if not envs:
        return None
    m = len(rnames)
    scale = 1.0 + max(abs(t) for t in targets)

    def resid(d):
        out = []
        for env, t in zip(envs, targets):
            try:
                with np.errstate(all='ignore'):
                    v = d_eval(tr, {**env, **cl_env, **{n: F(x) for n, x in zip(rnames, d)}})
            except Exception:
                return None
            if not np.isfinite(v):
                return None
            out.append(float(v) - t)
        return np.array(out)

    lv = [float(v) for v in cl_env.values()]
    seeds = [(lv + [1.0] * m)[:m], [1.0] * m, [0.5] * m, [-1.0] * m, [2.0] * m,
             [float(i + 1) for i in range(m)], [0.0] * m]
    best, best_n = None, None
    for seed in seeds:
        d = np.array(seed, dtype=float)
        r = resid(d)
        if r is None:
            continue
        lam = 1e-3
        for _ in range(200):
            n0 = _norm(r)
            if n0 <= 1e-13 * scale:
                break
            J = np.empty((len(r), m))
            ok = True
            for j in range(m):
                h = 1e-7 * max(1.0, abs(d[j]))
                dp = d.copy()
                dp[j] += h
                rp = resid(dp)
                if rp is None:
                    ok = False
                    break
                J[:, j] = (rp - r) / h
            if not ok:
                break
            A = [[math.fsum(J[:, i] * J[:, j]) for j in range(m)] for i in range(m)]
            g = [math.fsum(J[:, i] * r) for i in range(m)]
            moved = False
            for _ in range(12):
                step = _solve([[A[i][j] + (lam * (A[i][i] + 1e-12) if i == j else 0.0)
                                for j in range(m)] for i in range(m)], [-x for x in g])
                if step is None:
                    lam *= 10
                    continue
                rn = resid(d + step)
                if rn is not None and _norm(rn) < n0:
                    d, r, lam, moved = d + step, rn, max(lam / 10, 1e-12), True
                    break
                lam *= 10
            if not moved:
                break
        n = _norm(r)
        if best_n is None or n < best_n:
            best, best_n = d, n
        if n <= 1e-9 * scale:
            break
    if best is None or best_n > 1e-6 * scale or not np.all(np.isfinite(best)):
        return None
    d = best.copy()

    def snapped(x):
        """The simple value (an integer, a half, or one of the left side's own constants,
        the answer for every respelling) within fit noise of `x`, or `x`."""
        for c in (*lv, round(x), round(x * 2) / 2):
            if abs(c - x) < max(1e-9 * max(abs(x), abs(c)), 1e-15):
                return float(c)
        return float(x)

    # all coordinates together first (the fit's noise is shared between them), then each, every
    # trial from the CURRENT fit: a coordinate trial built from the unsnapped fit undid the joint
    # snap on a tie (`<C_L0> = 0` in `(x + c)^2 -> x^2 + c x + c`: the fit's -2.5e-17 came back)
    for j in range(-1, m):
        trial = np.array([snapped(x) for x in d]) if j < 0 else \
            np.where(np.arange(m) == j, snapped(d[j]), d)
        rt, rd = resid(trial), resid(d)
        if rt is not None and rd is not None and _norm(rt) <= _norm(rd):
            d = trial
    for k in (2, 3):
        env = _gen_env(shared, k)
        try:
            Lv = d_eval(tl, {**env, **cl_env})
            Rv = d_eval(tr, {**env, **cl_env, **{n: F(x) for n, x in zip(rnames, d)}})
        except Exception:
            return None
        if np.isfinite(Lv) != np.isfinite(Rv):
            return None
        if np.isfinite(Lv) and abs(Lv - Rv) > 1e-4 * max(1.0, abs(Lv)):
            return None
    if env_out is not None:
        env_out.extend(envs)
    return [float(x) for x in d]


def mp_polish_multi(tl, tr, nvars, cl_builders, d0, envs):
    """`mp_polish` for several right-side constants: Newton in the contract's values at the
    fit's environments, at a precision past the ladder's deepest rung; the f64 fit where the
    landscape defeats it. Returns `{name: mpf}`."""
    names = list(d0)
    old = mp.dps
    _POLISH_DPS[0] = None
    try:
        mp.dps = 2 * GAP_RUNGS[-1] + 50

        def env_at(env):
            e = {n: mpf(float(v)) for n, v in env.items()}
            e.update({n: b() for n, b in cl_builders.items()})
            return e

        try:
            targets = [c_eval(tl, env_at(env)) for env in envs]
        except Unresolved:
            return {n: mpf(d0[n]) for n in names}
        if any(misnan(t) or misinf(t) for t in targets):
            return {n: mpf(d0[n]) for n in names}

        def f(*d):
            out = []
            for env, t in zip(envs, targets):
                e = env_at(env)
                e.update(dict(zip(names, d)))
                out.append(c_eval(tr, e) - t)
            return out[:len(names)] if len(out) >= len(names) else out + [mpf(0)] * (len(names) - len(out))

        try:
            sol = mp.findroot(f, [mpf(d0[n]) for n in names], tol=mpf(10) ** (-(mp.dps - 30)),
                              maxsteps=100)
            sol = [mpf(x) for x in ([sol] if len(names) == 1 else list(sol))]
            # an exact-zero witness stays exact, as `mp_polish` keeps it, where zero fits as well
            # at every environment: Newton's residue of ~1e-557 for a true 0 is a nonzero value
            # where both sides are exactly zero (x = 0 in `(x + 0)^2 = x^2 + d0 x + d1`)
            zeroed = [mpf(0) if abs(x) < mpf(10) ** (-(mp.dps - 25)) else x for x in sol]
            if zeroed != sol:
                tol = mpf(10) ** (-(mp.dps - 30))
                try:
                    res = []
                    for env, t in zip(envs, targets):
                        e = env_at(env)
                        e.update(dict(zip(names, zeroed)))
                        res.append(abs(c_eval(tr, e) - t) <= tol * max(1, abs(t)))
                    if all(res):
                        sol = zeroed
                except Unresolved:
                    pass
            _POLISH_DPS[0] = mp.dps - 30
            return dict(zip(names, sol))
        except (Unresolved, ZeroDivisionError, ValueError, TypeError):
            return {n: mpf(d0[n]) for n in names}
    finally:
        mp.dps = old


#: The precision (decimal digits) the last `mp_polish` / `mp_polish_multi` witness is good to,
#: or None where the polish fell back to its seed (the f64 fit). A comparison at a rung finer
#: than the witness measures the fit, not the functions: the judge reads such a point as
#: Unresolved (skip, never convict).
_POLISH_DPS = [None]


def mp_polish(tl, tr, nvars, clb, c0, env0=None, rname='<C_R>'):
    """Refine an f64-fitted exists-witness to mp precision against the CONTRACT
    evaluator (the quantifier ranges over R; the f64 fit is only a starting point --
    left at f64 it manufactures 1e-17 'real changes' against exact transcendental
    literals like np.e, producing false clause-(a) kills). mp secant at a generic
    point; falls back to the f64 value if the landscape defeats it (any residual kill
    then surfaces in the sweep for investigation, never silently).

    THE GENERIC POINT IS THE ONE THE FIT WAS DETERMINED AT (`env0`, from `fit_witness`). A
    lattice head can make the target flat in the witness at the first generic point:
    `max(x, acos(c)) -> max(c_t, x)` at x = 1.7 reads 1.7 for every c_t below it, so the secant
    found no slope there, kept the float64 seed, and the frozen 1e-17 gap convicted the
    identity at x = 0. The fit already moved on to a point that pins the witness."""
    old = mp.dps
    gen = None if env0 is None else {n: float(v) for n, v in env0.items()}
    _POLISH_DPS[0] = None

    def done(v, ok):
        _POLISH_DPS[0] = (mp.dps - 30) if ok else None
        return v

    try:
        # THE WITNESS MUST OUT-RESOLVE THE LADDER. `<C_L>` is rebuilt exactly at each
        # rung while `<C_R>` is this fitted value merely re-rounded, so a witness good to
        # only 45 digits puts a FROZEN ~1e-45 gap between the two sides -- one that does
        # not decay, and that the decay test therefore reads as analytic. That
        # convicted the identity `pow abs ?0 <constant> -> pow abs ?0 <constant>`, whose
        # two sides are the same tokens. Fitting deeper than the deepest rung keeps the
        # comparison a statement about the FUNCTIONS rather than about this fit.
        def build_env():
            e = {}
            for i, n in enumerate(sorted(nvars)):
                e[n] = mpf(GEN[i % len(GEN)] if gen is None else gen[n])
            # the EXACT battery value (symbolic builder), NOT its f64 rendering: a
            # witness polished against f64-pi differs from the judge's exact mp.pi by
            # ~4e-17, producing false clause-(a) kills of the witness family. Rebuilt
            # after the depth is settled, since a symbolic atom is only as good as
            # the dps it was rendered at.
            if isinstance(clb, dict):
                e.update({n: b() for n, b in clb.items()})
            elif clb is not None:
                e['<C_L>'] = clb()
            return e

        # ...AND THE LADDER'S DEPTH IS OPERAND-SCALED, so `GAP_RUNGS[-1]` stopped
        # naming its deepest rung. Size the rule's own intermediates first and match the
        # depth the ladder will actually climb to. Measured on `log(cosh(C)) -> c_t` at
        # C = 710: cosh(710) ~ 1.1e308 puts rung 2 at dps 668 and rung 3 at dps 1336,
        # against a witness frozen at 300 -- so the SAME ~1e-300 residue sat at both
        # rungs, did not decay, and the decay test read it as an analytic gap. A
        # clause-(a) KILL of a rule that is true by construction. Costs nothing for an
        # ordinary rule: small intermediates give back exactly the old 300.
        mp.dps = GAP_RUNGS[-1] + 50
        _MAG_SINK.append([mpf(0), mpf(0)])
        try:
            c_eval(tl, build_env())
        except Unresolved:
            # NOT a fallback -- this probe exists to MEASURE, and a refusal is one of
            # its readings. `acosh(exp(exp(-1000)))` is 1.0e-217, but at dps 300 the
            # inner `exp` rounds to exactly 1.0 and `acosh` refuses at its boundary
            # band; returning the f64 seed here froze the witness at 0.0 and clause (a)
            # convicted the identity. The intermediates it DID see (exp(-1000) =
            # 5.1e-435) are enough to size the depth, and the target evaluation below
            # already falls back for real if the deeper rung refuses too.
            pass
        finally:
            _mag, _small = _MAG_SINK.pop()
        mp.dps = max(mp.dps,
                     max(GAP_RUNGS[2],
                         2 * max(GAP_RUNGS[1], _required_dps(_mag, _small))) + 50)
        # `_WIT_RESID` was 10^-270 = "deeper than the deepest rung" when the deepest rung
        # was a fixed 250. It has to follow the ladder for the same reason the depth does.
        # A TARGET THE TABLE CANNOT READ AT THIS DEPTH is read at a shallower one: the witness
        # is then good to fewer digits, and `_POLISH_DPS` says so, so the rungs finer than it
        # skip the point instead of convicting a frozen gap (`erf` reads to 8,192 bits, and
        # `erf(1)^10000 -> c_t` asks for 3,122 digits).
        while True:
            env = build_env()
            try:
                target = c_eval(tl, env)
                break
            except Unresolved:
                if mp.dps <= GAP_RUNGS[-1] + 50:
                    return done(mpf(c0), False)
                mp.dps = max(GAP_RUNGS[-1] + 50, mp.dps // 2)
        _wit_resid = mpf(10) ** (-(mp.dps - 30))
        if misnan(target) or misinf(target):
            return done(mpf(c0), False)

        _residue: list = []

        def target_is_residue():
            """Is `target` the RESIDUE of a cancellation, or a value in its own right?

            The decay question, asked of the target itself: a residue SHRINKS when the
            working precision rises (`sin(pi)` reads ~1e-300 at dps 300 and ~1e-600 at
            dps 600), a value STANDS STILL. Only for a residue is an absolute noise
            floor the right instrument -- `pow(acos(0), -3000)` is 4.37e-589 at every
            precision, out of one power and no subtraction at all, and the floor
            accepted 0 as its witness. The contract then judges by pure relative decay,
            sees a 100% frozen gap, and KILLs a rule true by construction. Measured
            over the widened constant quantifier: 75 of 97 such kills.

            Asked ONCE, lazily, and only where the answer can change the verdict, so an
            ordinary witness never pays for the extra depth.
            """
            if not _residue:
                lo = mp.dps
                try:
                    mp.dps = 2 * lo
                    t2 = c_eval(tl, build_env())
                except (Unresolved, ValueError):
                    t2 = None          # cannot tell: keep the floor (the old answer)
                finally:
                    mp.dps = lo
                _residue.append(
                    t2 is None or target == 0 or t2 == 0
                    or abs(t2) < abs(target) * mpf(10) ** (-(lo // 2)))
            return _residue[0]

        def accepts(residual):
            """Is this residual inside the noise of THIS computation?

            The test used to be `|r| <= _WIT_RESID * max(1, |target|)`, which for every
            target below 1 is a pure ABSOLUTE floor -- the shape removed from the
            contract's own comparison, left behind in the test that accepts a witness.
            It accepted 0 as the witness for `asin(inv(cosh(710))) = 8.9e-309`, since
            8.9e-309 is under 1e-270 absolutely though it is 100% wrong relatively; the
            contract then judged by pure relative decay, saw a frozen 8.9e-309 gap and
            KILLed a rule true by construction.

            Two scales, taken as a MAXIMUM rather than multiplied. The target's own size
            carries the relative claim. The noise floor -- the largest intermediate
            scaled by the working precision, the same quantity `_required_dps` reasons
            about -- carries the other half: below it a value is not distinguishable
            from zero, which is what keeps a CANCELLATION zero (`sin(pi)` reads ~1e-300
            at dps 300, not 0) from being chased as though it were a real witness.

            AND THE NOISE FLOOR IS ASKED ONLY WHERE IT MEANS SOMETHING. It is an
            ABSOLUTE band, so applied unconditionally it swallows every honestly tiny
            target as well; `target_is_residue` is what tells the two apart.
            """
            if abs(residual) <= _wit_resid * abs(target):
                return True
            noise = max(_mag, mpf(1)) * mpf(10) ** (-(mp.dps - 25))
            return abs(residual) <= noise and target_is_residue()

        def h(c):
            e = dict(env)
            e[rname] = c
            v = c_eval(tr, e)
            if misnan(v) or misinf(v):
                raise Unresolved()
            return v - target

        if c0 == 0:
            # an exact-zero witness stays exact (the secant otherwise drifts to ~1e-62
            # and crosses pow's discontinuity at (0,0), fabricating phantom events on
            # identities) -- but only VERIFIED: f64 saturation flattens tiny TRUE
            # witnesses to 0.0 (acos(tanh(cosh(4))) = 2.2e-12 reads exactly 0 in f64;
            # tanh saturates at ~19), and freezing that zero fabricated clause-(a)
            # kills. If the contract rejects 0, fall through to the secant (the
            # +-1e-12 bracket recovers a tiny witness; h is linear for a
            # bare-<constant> RHS).
            try:
                if accepts(h(mpf(0))):
                    return done(mpf(0), True)
            except Unresolved:
                return done(mpf(0), True)
        try:
            a, b = mpf(c0) * (1 - mpf('1e-8')) - mpf('1e-12'), mpf(c0) * (1 + mpf('1e-8')) + mpf('1e-12')
            fa, fb = h(a), h(b)
            # Track the best iterate. Tightening the target must never turn a GOOD
            # witness into the f64 seed: falling back to `c0` would put a ~1e-17 frozen
            # gap between the sides, which is the same defect two orders worse.
            best, best_r = None, None
            for _ in range(200):
                if fb == fa:
                    break
                c = b - fb * (b - a) / (fb - fa)
                fc = h(c)
                a, fa, b, fb = b, fb, c, fc
                if best_r is None or abs(fc) < best_r:
                    best, best_r = c, abs(fc)
                if accepts(fc):
                    if abs(c) < mpf(10) ** (-(mp.dps - 25)):
                        try:
                            if accepts(h(mpf(0))):
                                return done(mpf(0), True)   # snap near-zero polish results to exact 0
                        except Unresolved:
                            pass
                    return done(c, True)
            if best is not None and best_r is not None:
                try:
                    if best_r < abs(h(mpf(c0))):
                        return done(best, False)        # short of target, but better than the seed
                except Unresolved:
                    return done(best, False)
        except (Unresolved, ZeroDivisionError):
            pass
        return done(mpf(c0), False)
    finally:
        mp.dps = old


# ---------------------------------------------------------------- the judge
#: The left side's kind ('fin', 'nan', '+inf', '-inf') at the last point `_point_verdict`
#: resolved: whether the left side is finite there decides what a constant owes (3.2).
_LHS_KIND = [None]


def _point_verdict(tl, tr, env_mp):
    """confirmed contract verdict at one point -> (verdict|None, snapped)

    Rung agreement alone is necessary and NOT sufficient, and this is where that used to
    bite. Confirming the CLASS of a verdict ('REAL-CHANGE' at both rungs) says nothing
    about whether the disagreement is real, because catastrophic cancellation swamps both
    rungs alike: mpmath at dps 50 and at dps 120 can agree perfectly that two equal
    numbers differ. Measured, the identity `log(cosh(25t) + sinh(25t)) = 25t` -- exactly
    true on all of R -- was KILLed at bc-positive-measure with 56 of 167 grid points
    "disagreeing" at dps 50, collapsing to 22, 17, 8, 4, 3 as the precision doubled.

    That monotone collapse IS the answer, and it is now what the finite half asks: a
    residue decays with the working precision and an analytic gap does not (the decay test, above
    `compare_classes`). The identity above is spared on decay -- its dps-50 residue of
    8.5e-21 would fail any magnitude bar, and still falls 10^69 by dps 120.

    The class half is unchanged and keeps the two/three-rung rule, because a class has no
    gap to watch. It is also the remaining hole: where cancellation drives a sum to a
    computed EXACT ZERO the left side reads `log(0) = -inf`, so the disagreement is a
    CLASS change and no decay test can fire on it. Rung 2's OPERAND SCALING is what
    addresses that -- at t = -20 the cancellation is ~217 digits deep, and the rung sized
    from the operands actually seen reaches past it where a fixed 120 never could.

    A SECOND hole, which the decay test opened wider before the boundary band closed it. The three EXACT
    identities that were three grid points from conviction -- `cos asin tanh _0 ->
    inv cosh _0` and two siblings -- moved TOWARD the bar rather than away from it: the bc
    measure rose from under 0.0167 to 0.03593 against a kill bar of 0.05. The decay test
    was never the cause at the point level; at the convicting grid point
    (_0 = -19.9862579) both the contract evaluator and bare mpmath put the gap at 10^-34.5
    (dps 50) and 10^-106.2 (dps 120), a drop of 71.6 that reads as residue and returns
    'eq'. The cause was SATURATION at `tanh`'s asymptote, where a rounded +-1 makes a
    relative gap of 1.0 that is identical at every affordable rung -- stability the decay
    test reads as analytic. It is refused at the asymptote now (the table's boundary band) and the
    rows that refuse at rung 1 are settled by the CLIMB below. Measured after both: those
    identities sit at 0 of 167 grid points, better than the 2 of 167 they scored before
    the decay test existed.
    """
    def once(dps, track=False):
        """One reading at `dps` -> (class_verdict|None, gap|None), or None if Unresolved.

        `class_verdict` is set whenever nan/inf is involved -- there the classes are the
        whole finding. Otherwise both sides are finite and `gap` carries it. With `track`,
        also report the largest finite intermediate either side produced, which is what
        sizes a cancellation."""
        mp.dps = dps
        if track:
            _MAG_SINK.append([mpf(0), mpf(0)])
        try:
            cl = cls_mp(c_eval(tl, env_mp()))
            cr = cls_mp(c_eval(tr, env_mp()))
        except Unresolved:
            return None
        finally:
            if track:
                once.mag, once.small = _MAG_SINK.pop()
        _LHS_KIND[0] = _kind(cl)
        v = compare_classes(cl, cr)
        return (v, None) if v is not None else (None, gap_reading(cl, cr))

    old = mp.dps
    snap0 = SNAP_EVENTS[0]
    _LHS_KIND[0] = None
    try:
        once.mag = once.small = None
        r1 = once(BASE_DPS, track=True)
        snapped1 = SNAP_EVENTS[0] > snap0
        # RUNG 2 IS OPERAND-SCALED. A fixed 120 confirms nothing when the intermediates
        # are 10^217: both rungs are swamped alike, agree on a manufactured verdict, and
        # the class comparison reads that agreement as evidence. The precision now comes
        # from the operands that were actually seen, so rung 2 is a second opinion rather
        # than a second copy of the first.
        dps2 = max(GAP_RUNGS[1], _required_dps(getattr(once, 'mag', None),
                                              getattr(once, 'small', None)))
        dps3 = max(GAP_RUNGS[2], 2 * dps2)

        # CLIMB PAST AN UNRESOLVED RUNG. A rung refuses when an intermediate lands inside
        # its boundary-honesty band, and that band NARROWS as the precision rises:
        # tanh(60) is inside it at dps 50 (band 1e-40) and well outside at dps 120
        # (1e-110). Bailing out at rung 1 therefore abstained on rows a higher rung can
        # settle -- measured, when the band went on tanh, that short-circuit sent 275
        # rules of rules_real.json to UNRESOLVED-COVERAGE and dropped the kills from 65 to
        # 27. An Unresolved is a statement about the RUNG, not about the rule.
        rungs = (BASE_DPS, dps2, dps3)
        r_lo = lo_dps = hi_dps = None
        for i in range(len(rungs) - 1):
            # A climbed rung measures the intermediates afresh: the refusing rung stopped at
            # the node it could not read, so it never saw the operands beyond it
            # (`tanh(100)*cosh(100) - sinh(100)` stops at `tanh` at dps 50 and never meets
            # the 10^43 the cancellation is scaled by).
            r = r1 if i == 0 else once(rungs[i], track=True)
            if r is not None:
                r_lo, lo_dps, hi_dps = r, rungs[i], rungs[i + 1]
                break
        if r_lo is None:
            # Refused at every rung we can afford -- so ask the one question that needs
            # no rung at all (the saturation rule). Reached only here, so it cannot move a verdict the
            # ladder was able to reach.
            return saturation_verdict(tl, tr, env_mp), snapped1
        cls1, gap1 = r_lo

        if cls1 is not None:
            # nan/inf: a CLASS is confirmed by agreement at a second rung -- there is no
            # gap to watch decay, so this half keeps the two/three-rung rule unchanged.
            # PAST THE CEILING, a class is no reading either: where the large intermediates
            # alone demand more digits than any rung carries, a sum of them reads an exact
            # zero at every affordable rung, and `log` of it a manufactured -inf
            # (`log(cosh(x) + sinh(x)) - x -> 0` at x = -1e4 cancels two numbers near
            # 10^4342). Skip, never convict.
            if _demanded_dps(getattr(once, 'mag', None), None) > MAX_DPS:
                return saturation_verdict(tl, tr, env_mp), snapped1
            r2 = once(hi_dps)
            if r2 is None:
                return None, snapped1
            if r2[0] == cls1 and not snapped1:
                return cls1, snapped1
            # a snapped agreement can be fabricated, and a disagreement is unsettled:
            # both take one rung higher.
            r3 = once(max(GAP_RUNGS[2], 2 * hi_dps))
            cls3 = r3[0] if r3 is not None else None
            return (r2[0] if r2[0] == cls3 else None), snapped1

        # BOTH FINITE -- decided by DECAY, never by size (see the decay test above).
        hi = once(hi_dps)
        if hi is None or hi[0] is not None or hi[1] is None:
            return None, snapped1            # class flipped / Unresolved: never convict
        d_lo, s_lo = gap1
        d_hi, s_hi = hi[1]
        # A RUNG THAT LOST A SIDE ENTIRELY CANNOT DATE A DECAY. `d == scale` (with a
        # nonzero scale) says exactly one side read 0 while the other did not: the rung
        # saw no agreement at all, and a rate measured from a reading pinned at the
        # instrument's rail is not a rate. `acos(cos(exp(-100)))` is 3.7e-44, but at dps
        # 50 `cos` of it rounds to exactly 1.0 and `acos` gives 0 -- so rung 1 read a
        # 100% gap, rung 2 read 1.1e-78, and the drop came to 34.5 against a bar of 35.
        # A true identity convicted by three tenths of a decade.
        #
        # Climb, and measure the rate between the two rungs that BOTH saw something.
        # This can only ever DELAY a conviction to a stricter pair, never prevent one:
        # where the vanished side is genuinely zero -- a written 0, as in
        # `exp(sinh(-10)) -> 0` -- it is still zero at the higher rung, the rail reading
        # repeats, and the KILL stands.
        if d_lo == s_lo != 0 and d_hi != s_hi:
            lo_dps, gap1 = hi_dps, hi[1]
            hi_dps = max(GAP_RUNGS[2], 2 * hi_dps)
            hi = once(hi_dps)
            if hi is None or hi[0] is not None or hi[1] is None:
                return None, snapped1
            d_lo, s_lo = gap1
            d_hi, s_hi = hi[1]
        # THE FINER RUNG FOUND NO SEPARATION AT ALL. That is the deepest evidence of
        # agreement the ladder can produce, and it is not something the floor below
        # should be allowed to argue with: at dps 918 the two sides of
        # `asin(inv(cosh(1000))) -> c_t` are bit-identical, yet flooring that exact zero
        # by the computation's own magnitude (cosh(1000) ~ 1e434) put a 9.9e-485 gap
        # where the rung had measured none, larger than the 2.2e-485 the COARSER rung
        # reported -- a negative drop, and a clause-(a) KILL of an identity.
        #
        # A false rule does not reach here: a fixed gap stays visible at the finer rung,
        # which is why `pow np.e sinh -10 -> 0` still reads 1.03e-4783 at every rung and
        # is still convicted below. This also subsumes the exactly-zero-on-both-sides
        # case, which is why the scale below can no longer be zero.
        if d_hi == 0:
            return 'eq', snapped1
        # ONE SCALE FOR BOTH RUNGS, AND IT IS THE VALUE'S, NEVER THE
        # COMPUTATION'S. The separations are absolute; the scale is
        # fixed here, so it divides out of the ratio and touches nothing but the floor.
        # This used to take the largest INTERMEDIATE into the max (`_MAG_SINK`'s
        # `once.mag`), on the argument that an exactly-cancelling source offers no
        # scale from its own value -- and that scaled the floor by everything the
        # computation passed through: `acos(inv(exp(exp 5))) -> div2 np.pi` has a
        # FIXED gap of -3.5e-65, the intermediate `exp(exp 5)` = 2.85e64 put the
        # rung-1 floor at 2.85e14, and the drop read 78.9 decades against a bar of 65
        # -- a false rule CERTIFIED core, in both authorities, for the whole
        # `acos inv cosh c` family. The value scale (pi/2 here) puts the floor at
        # 1.57e-50 and the fixed gap stands still beside it, as a fixed gap must.
        # The exactly-cancelling source keeps its intermediate scale, and ONLY it:
        # when the coarse rung's whole reading vanished (`s_lo == 0` -- total
        # cancellation, `log(exp x * exp -x)` at x = 1e4 rounds through 1.0 to an
        # exact 0), the sides offer no value to scale by, the residue that appears
        # at the finer rung is residue OF the intermediate, and flooring by the
        # value scale (the residue itself) convicts the true identity at 19.8% of
        # the grid. The distinguisher is s_lo itself: the families the gap exists for arrive with
        # `s_lo ~ pi/2` -- the coarse rung SAW the value, so the value sets the
        # floor and the fixed 3.5e-65 gap stands. `_MAG_SINK` keeps its other
        # consumer, the precision-demand estimator, either way.
        mag = getattr(once, 'mag', None) or mpf(0)
        scale = max(s_lo, s_hi) if s_lo != 0 else max(s_hi, abs(mag))
        # A separation of exactly zero is still not a special VERDICT -- it enters as
        # the rung's own resolution, "closer than this rung can see", which is what
        # keeps `log np.e -> 1` and 31 other exact identities resolvable when their
        # sides agree to full precision at one rung and differ by a rounding at the
        # next. What changed is that the resolution is now taken AT THIS SCALE.
        #
        # AND NOTHING NONZERO IS FLOORED, which is the half that does the work. An
        # absolute 10^-dps floor acquits `pow np.e sinh -10 -> 0`: the true value is
        # 1.03e-4783 at every precision -- tiny, but FIXED, and false -- so a floor
        # that swamps it manufactures a decay it does not have. Left unfloored it
        # stands still beside its own scale and clause (a) convicts it.
        g_lo = d_lo if d_lo != 0 else scale * mpf(10) ** (-lo_dps)
        g_hi = d_hi
        # Residue falls with the added precision; an analytic gap does not move, and a
        # gap that only BECOMES visible higher up (deep saturation) drops by less than
        # nothing. Half the added precision separates them with >=30 decades to spare.
        drop = mp.log10(g_lo / g_hi)
        if drop > (hi_dps - lo_dps) / 2:
            return 'eq', snapped1
        # PAST THE CEILING, UNRESOLVED -- NEVER CONVICTED. That is what `MAX_DPS`
        # documents, and the ladder did not do it: it read a rung clamped to 2000 as
        # though it were the rung the point asked for. At c = -3000, `exp(c) = 1e-1303`
        # needs dps 2656 before `cos` of it is anything but exactly 1.0, so every
        # affordable rung reads `acos(1) = 0`, the gap never moves, and clause (a)
        # convicts `acos(cos(exp(c))) -> c_t` -- true for every c by construction.
        #
        # Only the CONVICTION is withheld. A decay the clamped rungs could still see is
        # evidence of agreement and keeps its 'eq'; running out of precision is a
        # statement about the judge, and the house rule for that is skip, never convict.
        #
        # AND NOT WHERE A SIDE IS WRITTEN. The same LITERAL PROVENANCE that governs the
        # boundary bands governs this: a side the rule SPELLS is exact at every rung, so
        # a gap against it is not one more precision could ever close. `10^sinh(-10)` is
        # 5.8e-11014 -- far below any affordable rung, and no closer to the 0 the rule
        # writes than it was at dps 50. Withholding there would abstain on 26 shipped
        # rows the judge convicts correctly, which is the exact-zero family exactly.
        if (_demanded_dps(getattr(once, 'mag', None),
                          getattr(once, 'small', None)) > MAX_DPS
                and tl[0] != 'lit' and tr[0] != 'lit'):
            return saturation_verdict(tl, tr, env_mp), snapped1
        # ...BUT A WRITTEN SIDE DOES NOT MAKE A CANCELLATION READABLE. Where the LARGE
        # intermediates alone demand more than the ceiling and both sides read inside the
        # noise those intermediates leave at the rung, the computed side is that noise:
        # `cosh(x)^2 - sinh(x)^2 -> 1` at x = 1e4 cancels two numbers near 10^8686, which
        # no affordable rung resolves, and the written 1 beside it says nothing about the
        # difference. A large value standing above the noise (`exp(exp(10)) -> 0`) is read.
        big = abs(mag)
        if (_demanded_dps(big, None) > MAX_DPS
                and max(s_lo, s_hi) <= big * mpf(10) ** (-(lo_dps - 10))):
            return saturation_verdict(tl, tr, env_mp), snapped1
        return 'REAL-CHANGE', snapped1
    finally:
        mp.dps = old


#: The cells of the soundness LATTICE, one per combination of the three modes' criteria. The
#: notions are INCOMPARABLE -- a rule can be true and unrealised (`atanh(tanh t) -> t`, exact on
#: R, `inf` in f64) or realised and untrue (`asin(1e-8) -> 1e-8`, bit-identical in f64, wrong by
#: 1.7e-25) -- so this is an axis, not a rung on the Mode ladder.
#:
#:   core        holds in real AND in f64            served by every mode
#:   real        holds in real and permissive, not in f64
#:   real-only   holds in real only (a `$` slot owed nothing at 0 in real, where permissive's
#:               real-valued guarantee reaches it)
#:   f64         holds in f64 and permissive, not in real   f64 and permissive (P3: real's
#:               values up to float resolution, except at isolated points)
#:   f64-only    holds in f64, not in real or permissive (float64's own value: `floor(tanh(x))`
#:               at a saturation)
#:   permissive  holds in permissive only            (an extension: `exp(log x) -> x`)
#:   reject      holds in no mode
def _tier(verdict, realised, permissive=None):
    """The tier of a judged rule: `verdict` is real's (CERTIFIED / TOLERATED / ENGINE-MISALIGN
    hold), `realised` f64's (`True` only: `None` means the deployed lane never got an
    observation, and absent evidence supports no claim -- `sin(np.pi) -> 0` is exactly true while
    f64 computes 1.2246467991473532e-16), `permissive` permissive's (`None`: not judged, read as
    real's)."""
    if verdict not in ('CERTIFIED', 'TOLERATED', 'ENGINE-MISALIGN', 'KILL', 'UNDECIDED-MEASURE'):
        # NO-WITNESS, UNRESOLVED-COVERAGE, UNSUPPORTED-SHAPE, JUDGE-TIMEOUT: the judge reached
        # no answer, and holds in no mode
        return 'reject'
    real = verdict in ('CERTIFIED', 'TOLERATED', 'ENGINE-MISALIGN')
    # an undecided measure refuses real and f64, which both owe clause (b); permissive does not
    # owe it (the owner's ruling, specification 4.10)
    f64 = realised is True and verdict not in ('ENGINE-MISALIGN', 'UNDECIDED-MEASURE')
    perm = real if permissive is None else bool(permissive)
    if real and f64:
        return 'core'
    if real:
        return 'real' if perm else 'real-only'
    if f64:
        return 'f64' if perm else 'f64-only'
    return 'permissive' if perm else 'reject'


def judge_rule(lhs, rhs, deployed_check=True):
    """-> dict: verdict CERTIFIED | TOLERATED | KILL | ENGINE-MISALIGN | NO-WITNESS |
    UNRESOLVED-COVERAGE | UNDECIDED-MEASURE | UNSUPPORTED-SHAPE, with clause/class, points, the
    certified measure (`_judge_contract`), and the tier the three modes' criteria give
    (`_tier`).

    A rule spelled with a 0.15 head (`NEW_HEADS`) is realised in float64 only where the core's
    f64 bar agrees too (`_core.f64_bar_sorted`: bit for bit at the jumps, the signed zeros and
    the poles float64 parts from the reals at, which the battery and the grid do not reach), so
    `floor(x + 1) -> floor(x) + 1` is CERTIFIED and not realised (float64 breaks it at
    x = -1e-17): tier `real`. The bar probes each slot over its sort's values (a `?` slot over
    finite doubles only: +-inf is no data), and a probe where an operation overflowed or
    underflowed agrees where the rewrite gives the true value (the range standard), judged
    here by the contract."""
    r = _judge_contract(lhs, rhs, deployed_check)
    if not deployed_check or 'tier' not in r or not NEW_HEADS & (set(lhs) | set(rhs)):
        return r
    from simplipy._core import f64_bar_sorted  # type: ignore[attr-defined]
    slot = {}
    for t in list(lhs) + list(rhs):
        key = '?' + t[1:] if t in ('x0', 'x1', 'x2') else t
        if SLOT_RE.match(key) and key not in slot:
            slot[key] = None
    keys = sorted(slot)
    names = {key: f'x{i}' for i, key in enumerate(keys)}
    spell = lambda ts: [names.get('?' + t[1:] if t in ('x0', 'x1', 'x2') else t, t)  # noqa: E731
                        for t in ts]
    bar = f64_bar_sorted(spell(lhs), spell(rhs), list(names.values()),
                         ''.join(k[0] for k in keys), ARITY)
    if bar is None:
        return r
    ok, probes, first, range_rows = bar
    range_ok = True
    if ok and range_rows:
        # the range standard: the rewrite gives the true value where an operation overflowed
        # or underflowed -- the contract's agreement there, at each such probe
        tl = parse(lhs, '<C_L>', number=True)
        tr = parse(rhs, '<C_R>', number=True)
        lnames = _constant_names(lhs, '<C_L>', True)
        for row in range_rows[:RANGE_ROWS_CAP]:
            env = {k: mpf(v) for k, v in zip(keys + lnames, row)}
            v, _ = _point_verdict(tl, tr, lambda env=env: {k: +x for k, x in env.items()})
            # a probe the contract cannot resolve says nothing about the true value, and is
            # skipped as the deployed lane skips an unresolved grid point (`cosh(1e17)` is
            # beyond every rung); one it resolves to a disagreement fails the bar
            if v is not None and v != 'eq':
                range_ok, first = False, list(row)
                break
        if len(range_rows) > RANGE_ROWS_CAP:
            range_ok = False          # more than the judge reads: undecided, fail closed
    ok = ok and range_ok
    realised = r.get('realised') is not False and ok
    return dict(r, realised=realised,
                tier=_tier(r['verdict'], realised, r.get('permissive')),
                f64_bar={'ok': ok, 'probes': probes, 'range_probes': len(range_rows),
                         'first': None if first is None else dict(zip(keys, first))})


#: The most range-event probes of the f64 bar the judge reads by the contract (each is a
#: ladder evaluation); a bar with more is undecided, and fails closed.
RANGE_ROWS_CAP = 256


#: Each `<constant>` is its own symbol (3.2): with two or more on the left, the battery is the
#: product of the generic constants plus each constant over the full battery with the others at
#: distinct generic values (the full product of the full battery is out of reach).
def _cl_combos(lnames, saturation):
    """The left side's constant bindings the judge quantifies over: `[(builders, key)]`, the
    builders by name (exact values, rebuilt per rung), the key their float tuple."""
    if not lnames:
        return [({}, ())]
    single = judge_cl_battery(saturation=saturation)
    if len(lnames) == 1:
        return [({lnames[0]: b}, (float(b()),)) for b in single]
    gen = [(lambda c=c: mpf(c)) for c in CONSTS]
    out, seen = [], set()

    def add(bs):
        key = tuple(float(b()) for b in bs)
        if key not in seen:
            seen.add(key)
            out.append((dict(zip(lnames, bs)), key))

    for combo in itertools.product(gen, repeat=len(lnames)):
        add(combo)
    for i in range(len(lnames)):
        for b in single:
            bs = [gen[(j + 1) % len(gen)] for j in range(len(lnames))]
            bs[i] = b
            add(bs)
    return out


def _generic_key(lnames):
    """The generic binding of the left side's constants (distinct values for distinct
    constants), where the grid and the measure are read."""
    if len(lnames) == 1:
        return (float(CONSTS[0]),)
    return tuple(float(CONSTS[(j + 1) % len(CONSTS)]) for j in range(len(lnames)))


def _spelled(tokens, names):
    """`tokens` with each slot spelled as the interval layer's dummy (`names`)."""
    return [names.get('?' + t[1:] if t in ('x0', 'x1', 'x2') else t, t) for t in tokens]


def _set_partitions(items):
    """Every partition of `items` into blocks, the order of `items` kept within a block."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        yield [[first]] + part
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]


def _renamed(tokens, rename):
    """`tokens` with each slot (a dummy `x<k>` read as `?<k>`) renamed by `rename`."""
    out = []
    for t in tokens:
        key = '?' + t[1:] if t in ('x0', 'x1', 'x2') else t
        out.append(rename.get(key, t))
    return out


#: The most `?` slots whose coincidences the judge enumerates (Bell(4) = 15 partitions); a rule
#: with more is undecided there and refused (fail closed).
MAX_COINCIDENT_SLOTS = 4


def _extension_width(lhs, lp, rhs, rp, names):
    """Clause (b)'s measure: the volume of the largest certified cell where the left side is
    undefined and the right side has a value, in the table's values (`None`: undecided)."""
    return _table.table_extension_width(_spelled(lhs, names), [float(c) for c in lp],
                                        _spelled(rhs, names), [float(c) for c in rp],
                                        dict(ARITY))


def _judge_contract(lhs, rhs, deployed_check=True, coincide=True):
    """-> dict: real's verdict, with clause/class, points and the certified measure, f64's
    `realised` and permissive's `permissive`; `judge_rule` adds the tier.

    real's verdict (the criterion of the specification's section 3):
      KILL               a disagreement at a point where the left side has a value (clause (a):
                         a real binding, a `_` slot's +-inf or NaN, any constant owed); any
                         disagreement at a `_` slot's non-finite binding; or a certified cell
                         of positive measure where the left side is undefined and the right
                         side has a value (clause (b)); a ground rule's one point, no tolerance;
      UNDECIDED-MEASURE  clause (b) could not be decided (the interval layer's horizon or budget
                         ran out): refused, fail closed;
      ENGINE-MISALIGN    the contract holds, the deployed algebra diverges structurally;
      TOLERATED          the contract holds with extensions on a null set only;
      CERTIFIED          the contract holds with no event at all.
    permissive: clause (a) alone at the real-valued bindings (every slot any real number).
    f64: the deployed lane (`realised`), each side's own fitted constant, the range standard.

    The measure-zero allowances -- an extension on a null set (clause (b)), and nothing owed
    where the left side is +-inf on a null set (clause (c), rules with variables only) -- hold
    only where a slot guards them (the owner's rulings, specification 3.1): a `?` slot, which
    binds a variable, after every way the `?` slots can coincide is judged too (`coincide`);
    `!` and `$` slots guard their certificates' null events, which they owe nothing at. An
    event that depends on any other slot is a positive-measure event in some instance (a `_`
    bound to a constant subexpression) and refuses the rule."""
    try:
        tl = parse(lhs, '<C_L>', number=True)
        tr = parse(rhs, '<C_R>', number=True)
    except UnsupportedToken as ex:
        # a token outside the numeric grammar: refuse the SHAPE, do not evaluate it
        return {'verdict': 'UNSUPPORTED-SHAPE', 'detail': str(ex)}
    sl = slots_of(lhs, '<C_L>', number=True)
    sr = slots_of(rhs, '<C_R>', number=True)
    slots = dict(sl)
    slots.update({k: v for k, v in sr.items() if k not in slots})
    nvars = {n: s for n, s in slots.items() if not n.startswith('<C')}
    lnames = _constant_names(lhs, '<C_L>', True)
    rnames = _constant_names(rhs, '<C_R>', True)
    has_cl, has_cr = bool(lnames), bool(rnames)
    names = sorted(nvars)
    spelling = {n: f'x{i}' for i, n in enumerate(names)}
    ground = not nvars

    # THE SATURATION PROBES ARE WITHHELD WHERE A WITNESS MUST BE RE-FITTED AT THEM.
    # Measured over the shipped rules.json: on the 71 rows whose constant appears only on
    # the SOURCE side, the probes move exactly 16 rows and every one moves cleanly from
    # `core` to `real` -- they are the rules found unrealised in f64. On
    # the 272 rows that also fit a TARGET constant they produce 88 clause-(a) KILLs, and
    # every one diagnosed is an artifact of the exists-witness machinery at magnitudes it
    # cannot resolve, not a false rule. So the probe is put where its answer can be read
    # and withheld where it cannot; `saturation_probed` says which happened.
    saturation_probed = not has_cr
    cl_battery = _cl_combos(lnames, saturation_probed)
    core = {float(c) for c in CONSTS}
    gkey = _generic_key(lnames) if has_cl else ()

    # TWO witness maps, one per algebra: the contract compares against the mp-polished
    # witness, the deployed check against the raw f64 fit -- deployment fits its OWN
    # constant, so judging the f64 algebra with the mp witness manufactures false
    # ENGINE-MISALIGNs across f64 saturation cliffs (asin(tanh(exp(3))): f64 sees exactly
    # pi/2, the contract pi/2 - 2.3e-9).
    witness, witness_f64, witness_dps = {}, {}, {}
    skipped_cl, unwitnessed_cl = [], []

    def fitted(key):
        """The polished witness at `key`, re-rounded at the current rung -- Unresolved at a
        rung finer than the witness itself (`_POLISH_DPS`)."""
        if not has_cr:
            return {}
        limit = witness_dps.get(key)
        if limit is not None and mp.dps > limit:
            raise Unresolved()
        return {n: +w for n, w in witness.get(key, {}).items()}

    if has_cr:
        kept = []
        for builders, key in cl_battery:
            cl_env = dict(zip(lnames, key))
            fit_env = []
            if len(rnames) == 1:
                w = fit_witness(tl, tr, set(nvars), cl_env, fit_env, rnames[0])
                ws = None if w is None else [w]
            else:
                ws = fit_witness_multi(tl, tr, set(nvars), cl_env, rnames, fit_env)
            if ws is None:
                # LHS UNDEFINED at this constant for the generic envs: nothing is owed there
                # (3.2), a skip with a tally. LHS defined (DETERMINATE, not merely finite: a
                # saturated LHS pins nothing, see `_fit_target_env`) and no witness: the
                # rule fails at a constant it owes.
                if len(rnames) == 1:
                    defined = _fit_target_env(tl, tr, set(nvars), cl_env, rnames[0]) is not None
                else:
                    defined = any(source_defined(d_eval(tl, {**_gen_env(set(nvars), k),
                                                             **_cl_env(cl_env)}))
                                  for k in range(len(GEN)))
                if defined:
                    if all(c in core for c in key):
                        return {'verdict': 'NO-WITNESS', 'detail': f'cl={key} (LHS defined)',
                                'permissive': False}
                    unwitnessed_cl.append(key)
                    continue
                skipped_cl.append(key)
                continue
            if len(rnames) == 1:
                witness[key] = {rnames[0]: mp_polish(tl, tr, nvars, builders, ws[0],
                                                     fit_env[0] if fit_env else None,
                                                     rnames[0])}
            else:
                witness[key] = mp_polish_multi(tl, tr, nvars, builders,
                                               dict(zip(rnames, ws)), fit_env)
            witness_dps[key] = _POLISH_DPS[0]
            witness_f64[key] = dict(zip(rnames, ws))
            kept.append((builders, key))
        if not kept:
            return {'verdict': 'NO-WITNESS', 'detail': 'no cl with a finite LHS/witness',
                    'permissive': False}
        if unwitnessed_cl:
            # forall-c exists-d is falsified at a constant where the LHS is defined. Reported
            # for ALL such constants: one is a corner, a spread is a false rule.
            return {'verdict': 'NO-WITNESS', 'skipped_cl': skipped_cl,
                    'detail': f'cl={sorted(unwitnessed_cl)} (LHS defined)',
                    'permissive': False}
        cl_battery = kept

    # battery sweep (cap the slot-product deterministically)
    combos = [{}]
    for n in names:
        nxt = []
        for c in combos:
            for builder, tag in battery_for(nvars[n]):
                e = dict(c)
                e[n] = (builder, tag)
                nxt.append(e)
        combos = nxt
        if len(combos) > BATTERY_CAP:
            # the cap is the FULL two-slot product, so every <=2-slot rule is judged
            # exhaustively and a >=3-slot sample is drawn from the COMPLETE two-slot base
            rng = np.random.default_rng(11)
            combos = [combos[i]
                      for i in rng.choice(len(combos), BATTERY_CAP, replace=False)]

    a_kills, atom_kills, ext_events, perm_kills = [], [], [], []
    ext_keys = set()
    # The REALISATION axis, recorded with no gate (`dep_bad`), and ENGINE-MISALIGN's
    # structural divergence where the contract agrees (`dep_div`).
    dep_bad, dep_div, dep_seen, dep_out_of_range = [], [], [False], [0]
    resolved_pts = attempted_pts = 0

    def deployed(tl_env, v, point, all_real, snapped, out):
        """The f64 lane at one point: append `(kind, point)` to `out['bad']` / `out['div']`."""
        if not deployed_check or snapped:
            return
        try:
            in_range = [True]
            with np.errstate(all='ignore'):
                _dl, _dr = d_eval(tl, tl_env, in_range), d_eval(tr, tl_env, in_range)
            out['seen'] = True
            out['finite_f64'] = out['finite_f64'] or bool(np.isfinite(_dl))
            if not in_range[0] and v == 'eq':
                # THE f64 STANDARD: an operation of either side overflowed or underflowed,
                # and the rewrite gives the true value -- an agreeing observation
                out['oor'] += 1
                return
            _dcl, _dcr = cls_np(_dl), cls_np(_dr)
            dv = compare_deployed_realised(_dcl, _dcr)
            if dv != 'eq':
                out['bad'].append((dv, point))
            # ENGINE-MISALIGN keeps its gate: the contract agrees, every binding real, and
            # a few-ULP rounding is no divergence
            if compare_deployed_structural(_dcl, _dcr) != 'eq' and v in (None, 'eq') \
                    and all_real:
                out['div'].append((dv, point))
        except Exception:
            pass

    def lane():
        return {'a': [], 'atom': [], 'ext': [], 'c': [], 'perm': [], 'finite': False, 'known': 0,
                'resolved': 0, 'attempted': 0,
                'bad': [], 'div': [], 'seen': False, 'oor': 0, 'finite_f64': False}

    def judge_point(acc, v, point, kind, real_valued, where):
        """File one resolved comparison: `kind` is 'real' (every binding a real number) or
        'atom' (a `_` slot at +-inf or NaN); `real_valued` says permissive's guarantee
        reaches the point; `where` is `(constant key, slot values)`, to move a slot from."""
        lhs_kind = _LHS_KIND[0]
        if kind == 'real' and lhs_kind is not None:
            acc['known'] += 1
            acc['finite'] = acc['finite'] or lhs_kind == 'fin'
        if v is None or v == 'eq':
            return
        if kind == 'atom':
            acc['atom'].append((v, point))
        elif v == 'EXT':
            acc['ext'].append((point, where))
        elif not ground and lhs_kind in ('+inf', '-inf'):
            # the left side is +-inf here: clause (c) decides, after the sweep
            acc['c'].append((v, point, where, real_valued))
        else:
            acc['a'].append((v, point))
            if real_valued:
                acc['perm'].append((v, point, where))

    def owed(acc):
        """3.2: is anything owed at this constant? Where the left side is finite at no real
        point the judge read, nothing is (a fitted constant never lands there: the f64 fit
        may still have found one at the constant's float, `log(|cos(c)|)` at fl(pi/2), but the
        exact pi/2 gives -inf); without constants, always. A constant where the left side was
        read nowhere is owed: unread is not nowhere finite."""
        return not has_cl or acc['finite'] or acc['known'] == 0

    vacuous, owed_keys = [], set()
    ext_where, c_events = [], []

    def merge(acc, key):
        nonlocal resolved_pts, attempted_pts
        resolved_pts += acc['resolved']
        attempted_pts += acc['attempted']
        # f64 owes at a constant by ITS values: where the deployed left side is finite
        # somewhere (`pow(sin(c), inf)` at fl(pi/2) is pow(1.0, inf) = 1 in float64, though
        # undefined at the exact pi/2), whatever real owes there
        if not has_cl or acc['finite_f64']:
            dep_bad.extend(acc['bad'])
            dep_div.extend(acc['div'])
        dep_seen[0] = dep_seen[0] or acc['seen']
        dep_out_of_range[0] += acc['oor']
        if not owed(acc):
            # nothing owed here, which the resolved points decided: they count as coverage
            skipped_cl.append(key)
            vacuous.append(key)
            return
        owed_keys.add(key)
        a_kills.extend(acc['a'])
        atom_kills.extend(acc['atom'])
        perm_kills.extend(acc['perm'])
        if acc['ext']:
            ext_events.extend(p for p, _ in acc['ext'])
            ext_where.extend(acc['ext'])
            ext_keys.add(key)
        c_events.extend(acc['c'])

    for builders, key in cl_battery:
        cr_mp = witness.get(key, {}) if has_cr else {}
        cr_f64 = witness_f64.get(key, {}) if has_cr else {}
        acc = lane()
        for combo in combos:
            tags = {n: t for n, (b, t) in combo.items()}
            values = {n: b() for n, (b, t) in combo.items()}
            finite = {n: math.isfinite(float(x)) for n, x in values.items()}
            real_valued = all(finite.values())
            skip_real = any(t == 'nullx' for t in tags.values())

            def env_mp(combo=combo, key=key, builders=builders):
                e = {n: b() for n, (b, t) in combo.items()}
                e.update({n: b() for n, b in builders.items()})  # exact at the CURRENT dps
                e.update(fitted(key))                             # re-rounded per rung
                return e

            if skip_real and not real_valued:
                continue          # owed nothing in real, outside permissive's guarantee
            v, snapped = _point_verdict(tl, tr, env_mp)
            point = {n: float(x) for n, x in values.items()}
            point.update(dict(zip(lnames, key)))
            point.update({n: float(w) for n, w in cr_mp.items()})
            if skip_real:
                # a `$` slot at 0: owed nothing in real (its certificate), a real-valued
                # binding permissive's guarantee covers
                if v not in (None, 'eq', 'EXT'):
                    acc['perm'].append((v, point, (key, values)))
                continue
            kind = 'atom' if any(t == 'atom' for t in tags.values()) else 'real'
            acc['attempted'] += 1
            if v is not None:
                acc['resolved'] += 1
            if ground and not has_cl and v == 'EXT':
                # a GROUND rule has one point, its whole domain: an extension there is no
                # null event (no tolerance, 3.3)
                acc['a'].append((v, point))
            else:
                judge_point(acc, v, point, kind, real_valued, (key, values))
            ed = {n: F(float(x)) for n, x in values.items()}
            ed.update({n: F(c) for n, c in zip(lnames, key)})
            ed.update({n: F(w) for n, w in cr_f64.items()})   # each algebra its OWN constant
            deployed(ed, v, point, kind == 'real', snapped, acc)
        merge(acc, key)

    # the GENERIC GRID: more points of clause (a) past the battery's magnitudes, and the
    # deployed lane where float64 parts from the reals (tanh saturates at 19.06, exp overflows
    # at 709.78 and underflows at -745.13); every slot real
    grid_key = gkey if has_cl else ()
    grid_ready = (not has_cl or any(k == grid_key for _, k in cl_battery)) and \
        (not has_cr or grid_key in witness)
    gbuilders = next((b for b, k in cl_battery if k == grid_key), {})
    grid_ext = False
    old = mp.dps
    mp.dps = BASE_DPS
    try:
        if names and grid_ready:
            cr_mp = witness.get(grid_key, {}) if has_cr else {}
            cr_f64 = witness_f64.get(grid_key, {}) if has_cr else {}
            acc = lane()
            lanes = [(cfg, n) for cfg in (1.7, -1.3) for n in names]
            for cfg, n in lanes + ([(None, None)] if len(names) >= 2 else []):
                for g in GRID:
                    if n is None:
                        # THE DIAGONAL: every slot bound to one value -- an instance family
                        # the engine reaches (`(a - b)/(b - a)` with a = b = log(x))
                        vals = {m: float(g) for m in names}
                    else:
                        vals = {m: cfg for m in names}
                        vals[n] = float(g)

                    def env_mp(vals=vals):
                        e = {m: mpf(x) for m, x in vals.items()}
                        e.update({k: b() for k, b in gbuilders.items()})
                        e.update(fitted(grid_key))
                        return e

                    v, snapped = _point_verdict(tl, tr, env_mp)
                    point = dict(vals)
                    point.update(dict(zip(lnames, grid_key)))
                    judge_point(acc, v, point, 'real', True,
                                (grid_key, {m: mpf(x) for m, x in vals.items()}))
                    if v is None or n is None:
                        # the deployed lane reads the per-slot grid where the contract
                        # resolved: an unresolved point says nothing about the true value a
                        # range event accepts (`ceil(1/cosh(1e4))` needs 4,353 digits)
                        continue
                    ed = {m: F(x) for m, x in vals.items()}
                    ed.update({k: F(c) for k, c in zip(lnames, grid_key)})
                    ed.update({k: F(w) for k, w in cr_f64.items()})
                    deployed(ed, v, point, True, snapped, acc)
            # the grid's points count toward nothing in coverage (that is the battery's)
            acc['resolved'] = acc['attempted'] = 0
            grid_ext = bool(acc['ext'])
            merge(acc, grid_key)
    finally:
        mp.dps = old

    # CLAUSE (b): measure from certified cells, read in the table's values -- at the generic
    # constants (always, for a rule with variables) and at every owed constant where an
    # extension was seen. (Coincident `?` slots are judged as their own rules, below.)
    b_kill, undecided, width = None, [], 0.0
    if not ground:
        checks = []
        if (not has_cr or grid_key in witness) and (not has_cl or grid_key in owed_keys):
            checks.append(grid_key)
        checks += sorted(k for k in ext_keys if k != grid_key and (not has_cr or k in witness))
        for key in checks:
            rp = [witness[key][n] for n in rnames] if has_cr else []
            w = _extension_width(lhs, key, rhs, rp, spelling)
            if w is None:
                undecided.append(key)
            elif w > 0:
                width = max(width, w)
                b_kill = b_kill or {'cl': list(key), 'width': w}
            elif key == grid_key and grid_ext:
                # NO CERTIFIED CELL IS NO PROOF OF MEASURE ZERO: an enclosure too loose to
                # certify one (`(x - x)/(x - x)`, whose numerator encloses as [-w, w]) finds
                # none, while the grid, which a null event's isolated point does not land
                # on, reads the extension at its points. The check cannot decide: refused.
                undecided.append(('cells-vs-grid',) + tuple(key))

    # THE GUARD (the owner's ruling): a measure-zero allowance holds only where a slot guards
    # it. An event (an extension, or an isolated infinity of the left side) that depends on a
    # slot other than `?` -- it disappears when that slot moves to a generic value -- is an
    # event on positive measure in the instance where the slot is bound to a subexpression
    # constant at that value, and refuses the rule.
    by_key = {k: b for b, k in cl_battery}
    open_slots = [n for n in names if nvars[n] != '?']

    def event_env(key, values):
        def env():
            e = {n: +x for n, x in values.items()}
            e.update({n: b() for n, b in by_key.get(key, gbuilders).items()})
            e.update(fitted(key))
            return e
        return env

    def depends_on_open_slot(kind, key, values):
        for n in open_slots:
            for alt in (mpf(17) / 10, mpf(-13) / 10):
                moved = dict(values)
                moved[n] = alt
                v2, _ = _point_verdict(tl, tr, event_env(key, moved))
                if kind == 'b':
                    persists = v2 == 'EXT'
                else:
                    persists = _LHS_KIND[0] in ('+inf', '-inf') and v2 not in (None, 'eq')
                if not persists:
                    return n
        return None

    b_unguarded, seen = [], set()
    for point, (key, values) in ext_where:
        sig = repr(sorted(point.items()))
        if sig in seen or not open_slots:
            continue
        seen.add(sig)
        slot = depends_on_open_slot('b', key, values)
        if slot is not None:
            b_unguarded.append(dict(point, unguarded=slot))
    # CLAUSE (c): where the left side is +-inf only on a set of measure zero the right side
    # owes nothing there -- certified by the infinite-set certificate, and guarded as above
    c_tolerated, c_kills, c_perm, seen = [], [], [], set()
    infinite_null = bool(c_events) and _table.table_infinite_null(
        _spelled(lhs, spelling), dict(ARITY))
    for v, point, (key, values), real_valued in c_events:
        sig = repr(sorted(point.items()))
        if sig in seen:
            continue
        seen.add(sig)
        slot = depends_on_open_slot('c', key, values) if open_slots else None
        if infinite_null and slot is None:
            c_tolerated.append((v, point))
        else:
            c_kills.append((v, dict(point, unguarded=slot) if slot else point))
            if real_valued:
                c_perm.append((v, point, (key, values)))

    # COINCIDENT `?` SLOTS: two `?` slots bound to one variable are an instance the engine
    # rewrites, on which a set that is null in the slots' joint space can have positive
    # measure (`(?0 - ?1)/(?1 - ?0) -> -1` is undefined everywhere at ?0 = ?1). Every way the
    # `?` slots can coincide is judged as its own rule; real and f64 owe it.
    correlated, correlated_f64 = None, None
    q_slots = [n for n in names if nvars[n] == '?']
    if coincide and len(q_slots) >= 2:
        if len(q_slots) > MAX_COINCIDENT_SLOTS:
            undecided.append(('coincidences',))
        else:
            for blocks in _set_partitions(q_slots):
                if all(len(b) == 1 for b in blocks):
                    continue
                rename = {n: b[0] for b in blocks for n in b}
                sub_r = _judge_contract(_renamed(lhs, rename), _renamed(rhs, rename),
                                        deployed_check, coincide=False)
                held = sub_r.get('verdict') in ('CERTIFIED', 'TOLERATED', 'ENGINE-MISALIGN')
                if not held and correlated is None:
                    correlated = {'coincide': [b for b in blocks if len(b) > 1],
                                  'verdict': sub_r.get('verdict'),
                                  'clause': sub_r.get('clause'),
                                  'points': sub_r.get('points') or sub_r.get('cell')}
                if sub_r.get('realised') is False and correlated_f64 is None:
                    correlated_f64 = [b for b in blocks if len(b) > 1]

    # ONE f64 STANDARD: the constructors fill a NaN hole on a null set (`x1/x1` is `1` in
    # f64), so a TOLERATED rule whose every deployed disagreement is such a fill -- the left
    # side NaN where the right side has a value (`EXT`), at a point the contract itself
    # tolerated as a null event -- is realised in f64 as well.
    _realised = (not dep_bad) if dep_seen[0] else None
    if correlated_f64 is not None:
        _realised = False
    fills_nan = bool(dep_bad) and all(k == 'EXT' and p in ext_events for k, p in dep_bad)
    # nothing owed at any constant (the left side finite nowhere the judge read it): the rule
    # holds vacuously (3.2), which is no coverage failure; evidence (4.1) refuses it
    vacuous_rule = has_cl and bool(vacuous) and not owed_keys
    covered = resolved_pts > 0 and resolved_pts / max(1, attempted_pts) >= 0.25

    # P3 (the owner's ruling): permissive also serves the rules that agree with real's values up
    # to float resolution, except at isolated points. A disagreement is excused where the two
    # real values round to doubles at most REALISED_ULP apart (F6), or where it is isolated: it
    # vanishes when a `?` slot moves off the point, and on no other slot's move (a `_` slot bound
    # to a constant subexpression would hold it on positive measure: unguarded).
    def doubles(env):
        """Both sides' values as the doubles nearest to them, read by the table at two
        precisions, or None where the table cannot read a side or the doubles differ between
        the two (a residue, not a value)."""
        old_dps = mp.dps
        try:
            seen = []
            for dps in (BASE_DPS, 120):
                mp.dps = dps
                e = env()
                vals = [_table_reading(t, e, dps) for t in (tl, tr)]
                if any(v is None for v in vals):
                    return None
                seen.append(tuple(_num(cls_mp(v)) for v in vals))
        finally:
            mp.dps = old_dps
        (l1, r1), (l2, r2) = seen
        same = all(a == b or (a != a and b != b) for a, b in ((l1, l2), (r1, r2)))
        return (l2, r2) if same else None

    def float_close(env):
        d = doubles(env)
        if d is not None:
            return _table.table_f64_agree(*d) == 'eq'
        # Past the table's reach (a value beyond the doubles' range, `exp(sinh(-50))` at
        # 10^(-1.1e21)): the range standard. Where an operation overflowed or underflowed,
        # the double nearest to the true value is float64's, 0 or an infinity.
        try:
            ed = {n: F(float(x.mpf() if hasattr(x, 'mpf') else x)) for n, x in env().items()}
            in_range = [True]
            with np.errstate(all='ignore'):
                dl, dr = d_eval(tl, ed, in_range), d_eval(tr, ed, in_range)
        except Exception:
            return False
        return not in_range[0] and compare_deployed_realised(cls_np(dl), cls_np(dr)) == 'eq'

    def disagrees(key, values):
        env = event_env(key, values)
        v2, _ = _point_verdict(tl, tr, env)
        if v2 == 'eq':
            return False
        return v2 is None or not float_close(env)

    def isolated(key, values):
        depends = []
        for n, x in values.items():
            x = x.mpf() if hasattr(x, 'mpf') else mpf(x)
            if misnan(x) or misinf(x):
                return False
            for alt in (x * (1 + mpf(10) ** -6) + mpf(10) ** -9,
                        x * (1 - mpf(10) ** -6) - mpf(10) ** -9):
                moved = dict(values)
                moved[n] = alt
                if not disagrees(key, moved):
                    depends.append(n)
                    break
        return bool(depends) and all(nvars.get(n) == '?' for n in depends)

    def excused(event):
        _, _, (key, values) = event
        return float_close(event_env(key, values)) or isolated(key, values)

    perm_left, seen_perm = [], set()
    for event in perm_kills + c_perm:
        sig = repr(sorted(event[1].items()))
        if sig in seen_perm:
            continue
        seen_perm.add(sig)
        if not excused(event):
            perm_left.append(event)
            break
    permissive = covered and not perm_left
    base = {'measure': width, 'skipped_cl': skipped_cl, 'saturation_probed': saturation_probed,
            'realised': _realised, 'out_of_range': dep_out_of_range[0],
            'permissive': permissive, 'resolved_points': resolved_pts}
    if perm_left:
        base['permissive_points'] = [p for _, p, _ in perm_left[:3]]
    if correlated_f64 is not None:
        base['unrealised_coincidence'] = correlated_f64
    if dep_bad:
        base['unrealised'] = dep_bad[:3]

    def verdict(name, **extra):
        out = dict(base, verdict=name, **extra)
        out['tier'] = _tier(name, out['realised'], out['permissive'])
        return out

    if a_kills:
        return verdict('KILL', clause='a-real-change', points=[p for _, p in a_kills[:3]],
                       kinds=sorted({k for k, _ in a_kills}))
    if atom_kills:
        return verdict('KILL', clause='a-nonfinite-binding',
                       points=[p for _, p in atom_kills[:3]],
                       kinds=sorted({k for k, _ in atom_kills}))
    if c_kills:
        return verdict('KILL', clause='c-infinity', points=[p for _, p in c_kills[:3]],
                       kinds=sorted({k for k, _ in c_kills}))
    if b_unguarded:
        return verdict('KILL', clause='b-unguarded', points=b_unguarded[:3], kinds=['EXT'])
    if b_kill is not None:
        return verdict('KILL', clause='b-positive-measure', cell=b_kill,
                       tolerated_events=len(ext_events))
    if correlated is not None and correlated['verdict'] != 'UNDECIDED-MEASURE':
        return verdict('KILL', clause='coincident-slots', coincidence=correlated)
    if not covered:
        # "skip, never convict" must not become "skip everything, acquit": a rule whose
        # battery is almost entirely Unresolved has no evidence either way (e.g. a
        # 1e15-argument trig rule certifying from one resolved point). The bar is a
        # FRACTION of attempted points: a ground rule with its single point resolved has
        # full coverage.
        return {'verdict': 'UNRESOLVED-COVERAGE', 'resolved_points': resolved_pts,
                'attempted_points': attempted_pts, 'skipped_cl': skipped_cl,
                'permissive': False}
    if correlated is not None:
        undecided.append(('coincidence', str(correlated['coincide'])))
    if undecided:
        # clause (b) could not be decided: fail closed (4.10), counted
        return verdict('UNDECIDED-MEASURE', undecided=[list(k) for k in undecided[:3]])
    if dep_div:
        # contract certifies; deployed diverges: a genuine realisation split
        return verdict('ENGINE-MISALIGN', points=[p for _, p in dep_div[:3]],
                       kinds=sorted({k for k, _ in dep_div}))
    if ext_events or c_tolerated:
        base['realised'] = True if fills_nan and correlated_f64 is None else _realised
        events = [('EXT', p) for p in ext_events[:3]] + [(k, p) for k, p in c_tolerated[:3]]
        classes = (['EXT'] if ext_events else []) + (['INF'] if c_tolerated else [])
        return verdict('TOLERATED', classes=classes, events=events)
    if vacuous_rule:
        return verdict('CERTIFIED', vacuous=True)
    return verdict('CERTIFIED')


# ---------------------------------------------------------------- self-test
TOUCHSTONES = [
    # (lhs, rhs, expected verdict, why -- the mathematical reason)
    ('+ ?0 pow sin ?0 float("inf")', '?0', 'TOLERATED',
     'powsin under the limit rule: pow(1, inf) is undefined (V5), so at x = pi/2 the left side '
     'has no value and the right side extends it on a null set only; elsewhere |sin x| < 1 and '
     'the power is 0'),
    ('rootn pow ?0 2 2', '?0', 'KILL',
     'sqrt(x^2)->x: real values differ on x<0 (clause a)'),
    ('atan tan asin _0', 'asin _0', 'KILL',
     'tan(pi/2) is undefined at _0 = +-1, and a `_` slot guards no null set: bound to '
     'floor(x) + 1 it is 1 on all of [0, 1), where the right side invents pi/2 (the guard)'),
    ('atan tan asin ?0', 'asin ?0', 'TOLERATED',
     'the same extension over data: a `?` slot binds a variable, so +-1 is a null set'),
    ('abs log cos ?0', 'log inv cos ?0', 'TOLERATED',
     'clause (c): the left side is +inf at pi/2 + k pi only, a null set, where the right side '
     'meets 1/0 and owes nothing; they agree wherever the left side is finite'),
    ('/ - ?0 ?1 - ?1 ?0', '-1', 'UNDECIDED-MEASURE',
     'with ?0 = ?1 the left side is undefined everywhere and -1 is not: the coincidence is '
     'an instance, judged as its own rule, whose cells cannot see through x - x (refused)'),
    ('/ !0 !0', '1', 'KILL',
     'a `!` slot guards only its infinities: bound to floor(x) it is 0 on [0, 1), where 0/0 '
     'has no value and 1 does'),
    ('/ ?0 ?0', '1', 'TOLERATED',
     'x/x->1: undefined->defined at 0 only (clause b, null set)'),
    ('neg inv neg !0', 'inv !0', 'CERTIFIED',
     'one zero and 1/0 undefined (V4): both sides undefined at 0, equal elsewhere; the `!` '
     'slot owes nothing at +-inf'),
    ('* 0 !0', '0', 'CERTIFIED',
     'absorption: exact at every real value; the `!` slot owes nothing at +-inf'),
    ('inv sin np.pi', 'float("inf")', 'KILL',
     'sin(pi) is exactly 0 and 1/0 is undefined (V4): a ground rule has no tolerance (3.3)'),
    ('pow inv rootn ?0 4 4', 'pow ?0 <constant>', 'KILL',
     'even-root extension on the negative half-line (clause b, a certified cell)'),
    ('rootn pow float("-inf") ?0 2', 'pow float("inf") ?0', 'KILL',
     'pow(-inf, data): undefined a.e. (clause b, positive measure)'),
    ('rootn rootn atanh !0 2 2', 'rootn atanh !0 4', 'CERTIFIED',
     'rootn(-inf, 4) = nan = sqrt(sqrt(-inf)): both spellings agree with the contract'),
    ('/ 0 * 2 ?0', '0', 'TOLERATED',
     '0/0 at x=0 is an undefined->defined null-set extension (the x/x->1 class)'),
    ('neg + (-1) _0', '- 1 _0', 'CERTIFIED',
     'algebraic identity, exact everywhere incl the cancellation point and +-inf (one zero)'),
    ('pow sin <constant> float("inf")', '0', 'CERTIFIED',
     'powsin in CONSTANT space under the limit rule: at c = pi/2 the left side is undefined '
     '(pow(1, inf), V5) and owes nothing (3.2); everywhere else it is 0'),
    ('rootn pow _0 3 3', '_0', 'CERTIFIED',
     'cbrt(t^3) = t exactly on all of R incl negatives and +-inf (odd-root exactness)'),
    ('* 3 log rootn _0 3', 'log _0', 'CERTIFIED',
     '3*log(cbrt t) = log t for t>0; both undefined for t<0; -inf vs -inf equal at 0; '
     '3*inf = inf at +inf: no disagreement event anywhere -- exactly certified'),
    ('atanh tanh !0', '!0', 'ENGINE-MISALIGN',
     'BOTH halves, and they are independent. CONTRACT half: exactly true on ALL of R. '
     'DEPLOYED half: f64 tanh attains exactly 1.0 from 19.061547465398498, so the shipped '
     'engine returns inf where the rule says the argument. The rule is TRUE and NOT REALISED, '
     'which is precisely what ENGINE-MISALIGN names. The f64 standard leaves overflows and '
     'underflows out, and none occurs here: tanh saturates INSIDE the range.'),
    ('rootn exp ?0 2', 'exp / ?0 2', 'CERTIFIED',
     'the f64 standard: true on R, and f64 computes it wherever exp stays in range; past '
     'exp\'s overflow (709.78) and underflow (-745.13) the rule gives the true value, which '
     'the deployed lane accepts'),
    # -- regression guards --
    ('+ ?0 sin exp -100', '?0', 'KILL',
     'sin(e^-100) = 3.720076e-44, so at ?0 = 0 the LHS is 3.7e-44 against an RHS of 0: a '
     'fixed gap, which the decay test convicts'),
    ('rootn * - ?0 0.2 - ?0 0.8 2', 'rootn abs * - ?0 0.2 - ?0 0.8 2', 'KILL',
     'undefined on (0.2, 0.8), measure 0.6: a certified cell'),
    ('+ ?0 * pow exp neg pow - <constant> 2 2 float("inf") ?0', '?0', 'CERTIFIED',
     'at c = 2 the power is pow(1, inf), undefined (V5): the left side is undefined for every '
     'x there and owes nothing (3.2); at every other c it is x'),
    ('pow abs ?0 <constant>', 'pow abs ?0 <constant>', 'CERTIFIED',
     'identity: the polisher must not de-snap the exact-zero witness into 1e-62 and '
     'fabricate a pow(0,0)-crossing phantom'),
    ('sin * ?0 * 1e15 + 1 1e-20', 'sin * ?0 1e15', 'KILL',
     'the table reads sin(5e14 * (1 + 1e-20)) exactly: the sides differ by about 5e-6 at '
     '?0 = 1/2, a decided point'),
    ('- <constant> <constant>', '0', 'KILL',
     'two constants are two symbols (3.2): at (2.5, -1.5) the left side is 4'),
    ('* exp + ?0 <constant> <constant>', '* <constant> exp ?0', 'CERTIFIED',
     'for all c1, c2: d = c2 * e^c1 (3.2, independent constants)'),
    ('- _0 _0', '0', 'KILL',
     '`_` quantifies +-inf and undefined with no excuse: inf - inf and NaN - NaN have no '
     'value, and 0 does'),
    ('* 0 tanh _0', '* 0 _0', 'KILL',
     'at _0 = +inf the left side is 0 * 1 = 0 and the right side 0 * inf is undefined (N3)'),
    ('tan asin ceil ?0', 'atanh ceil ?0', 'KILL',
     'for x in (0, 1], ceil x = 1 and tan(pi/2) is undefined while atanh(1) = +inf: an '
     'extension on a certified cell of positive measure (3.1)'),
    ('+ ?0 * 0 rootn - pow ?0 2 0.01 2', '?0', 'KILL',
     'undefined on (-0.1, 0.1), where the right side has a value: a certified cell, however '
     'small a fraction of any grid'),
    ('log -1', '0', 'KILL',
     'a ground rule has one point and no tolerance: the left side is undefined (3.3)'),
    ('/ 1 0', 'float("inf")', 'KILL',
     '1/0 is undefined (V4): +inf from the right, -inf from the left'),
    ('besselj 3 neg _0', 'besselj -3 _0', 'CERTIFIED',
     'J_3(-x) = -J_3(x) = J_-3(x) at every value: an integer order reads its parity and its '
     'reflection, in the contract and bit for bit in float64'),
    ('besselj 0.5 neg _0', 'besselj 0.5 _0', 'KILL',
     'a non-integer order is NaN on the whole negative half-line, where the right side is '
     'defined (clause b, positive measure)'),
]


def selftest(verbose=True):
    ok = True
    for lhs, rhs, want, why in TOUCHSTONES:
        r = judge_rule(lhs.split(), rhs.split())
        got = r['verdict']
        mark = 'ok ' if got == want else 'FAIL'
        if got != want:
            ok = False
        if verbose:
            print(f'  [{mark}] {lhs}  ->  {rhs}: {got} (want {want})')
            if got != want:
                print(f'         {r}')
                print(f'         reason: {why}')
    return ok
