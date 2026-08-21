# mypy: ignore-errors
#!/usr/bin/env python
"""Canonical mpmath judge for the simplification-soundness contract.

A single implementation of the contract semantics: every soundness instrument
(the rule-completeness gate, a deployed-realization sweep, a manual review gate)
derives from this module. The judge evaluates both sides of a candidate rule
under an arbitrary-precision contract algebra and compares the results across a
battery of symbolic points and a generic grid.

Semantics implemented here:
  - One zero: neg(0) = 0, 1/0 = +inf (c/0 = sign(c)*inf, derived from c*(1/0)),
    0/0 undefined.
  - Limit-completion: tanh(+-inf) = +-1, exp(-inf) = 0, log(0) = -inf,
    atanh(+-1) = +-inf, 1/+-inf = 0; tan poles are UNDEFINED (two-sided);
    pow(negative including -inf, non-integer) is UNDEFINED.
  - The kill bar:
      (a) both-sides-REAL disagreement -> KILL at any measure;
      (b)/(c) disagreements involving nan/+-inf -> judged by MEASURE over data
          (positive measure kills; null / Dirac-degenerate events are tolerated
          and classified).

Precision honesty:
  - Battery points are SYMBOLIC (mp.pi/2 etc.), rebuilt per dps rung; f64
    renderings of transcendentals are NOT the contract's points.
  - Trig outputs with |v| < 10^(-dps+10) snap to exact 0 (symbolic-cancellation
    floor; battery/grid inputs are >= 1e-3 in magnitude so genuine tiny trig
    values cannot occur); |v| > 10^(dps-10) is a pole -> undefined.
  - Every non-eq point verdict is CONFIRMED at a second rung (dps 120); rung
    disagreement -> Unresolved (the point is skipped, never convicted).
  - mpmath wedge caps: |exponent| > 1e6 in pow, |arg| > 1e12 in trig ->
    Unresolved.
"""
import math
import re
from fractions import Fraction
import numpy as np
from mpmath import mp, mpf, isnan as misnan, isinf as misinf

# C35: no `np.seterr(all='ignore')` here -- that call is PROCESS-WIDE and ran at
# import, silently changing the embedder's numpy error handling because this module
# was imported. The deployed-check evaluator scopes `np.errstate` around its own
# operator applications instead (see `d_eval`); the host's configuration survives.
F = np.float64

try:  # the CORE serialization language, read from the engine's own table (C1.10):
    from simplipy._core import core_serialization_ops as _core_ops  # type: ignore[attr-defined]
    _CORE_ARITY = {t: s['arity'] for t, s in _core_ops().items()}
except Exception:  # pragma: no cover  (missing/unbuilt extension)
    _CORE_ARITY = {}

#: The judge's vocabulary is DELIBERATELY broader than the engine's: it keeps the deleted
#: 0.11 hyper-operator spellings so legacy-engine rewrites stay judgeable (pinned by
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
         'mult2': 1, 'mult3': 1, 'mult4': 1, 'mult5': 1}
SLOT_RE = re.compile(r'^([?!_$])(\d+)$')


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


#: The judge's statement of NON-JURISDICTION over the multi-Const family (B7/D36):
#: a rule whose LHS binds more than one independent `<constant>` is outside what the
#: one-shared-symbol contract can model, and is refused before judging with exactly
#: this detail. The engine's gate sites exempt this detail -- and ONLY this detail --
#: from the fatal-bucket drop, because the family's soundness authority is the
#: Const-channel chain, not this instrument.
CONST_CHANNEL_DETAIL = 'multiple LHS <constant>'


# ---------------------------------------------------------------- literal acceptor
class UnsupportedToken(ValueError):
    """a leaf token outside the accepted numeric grammar -- REFUSED, never evaluated.

    This module adjudicates rule sets it did not produce (`verify_ruleset` documents a
    path to a JSON file as input), so a leaf token is untrusted data, not code. The
    grammar below is the whole of what a literal may be; anything else is refused
    fail-closed and the rule is bucketed UNSUPPORTED-SHAPE."""


# `1`, `-1`, `0.5`, `-1e-09`, `.5`  -- sign, digits, optional exponent. Nothing else.
_NUM_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
# `(-10)`: the parenthesized-negative spelling. 26.98% of literal occurrences in the
# shipped artifact carry it (3,981 of 6,803 rules) -- an acceptor without it refuses
# the majority of the corpus it exists to gate.
_PAREN_NEG_RE = re.compile(r'^\(-((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)$')
# `float("inf")` / `float('-inf')` / `float("nan")` -- the canonical special spellings.
_FLOAT_CALL_RE = re.compile(r'^float\((["\'])([^"\']*)\1\)$')
_SPECIALS = {'inf': math.inf, '+inf': math.inf, '-inf': -math.inf, 'nan': math.nan}
_NAMED = {'np.pi': math.pi, 'np.e': math.e}


def _exact(t):
    """A decimal literal as the EXACT rational it spells -- `-0.1` is -1/10, not the
    double 0.1000000000000000055511151231257827.

    F97. Literals used to enter the tree through `float()`, so the engine's exact `Rat`
    arrived at an arbitrary-precision judge already wrong in the 17th digit, and `c_eval`
    then did `mpf(that_double)` and preserved the error at every rung. It convicted
    `inv(-10/t) -> -0.1*t` -- an exact identity -- as a clause-(a) REAL-CHANGE at measure
    1.0. A mis-rendered literal is wrong by the same 5.5e-17 at EVERY rung, so the gap does
    not decay and the judge reads it as analytic -- which, given that literal, is exactly
    what it is. No tolerance ever hid this; only the right point does.
    That is the same principle the module header already states for battery points: an f64
    rendering is NOT the contract's point.

    The DEPLOYED lane must keep seeing the double, because that is what deployment
    computes -- `d_eval` converts back with `F()`, so its semantics are unchanged.
    """
    return Fraction(t.strip())


def literal_value(t):
    """The accepted numeric grammar, evaluated without `eval`. Total: value or refusal."""
    if t in _NAMED:
        return _NAMED[t]
    m = _FLOAT_CALL_RE.match(t)
    if m:
        inner = m.group(2).strip()
        if inner in _SPECIALS:
            return _SPECIALS[inner]
        if _NUM_RE.match(inner):
            return _exact(inner)
        raise UnsupportedToken(f'float() of a non-numeric literal: {t!r}')
    m = _PAREN_NEG_RE.match(t)
    if m:
        return -_exact(m.group(1))
    if _NUM_RE.match(t):
        return _exact(t)
    if t.count('/') == 1:  # `p/q`: not in the shipped artifact, legal in a foreign one
        p, q = t.split('/')
        if _NUM_RE.match(p.strip()) and _NUM_RE.match(q.strip()):
            den = _exact(q)
            if den == 0:
                raise UnsupportedToken(f'zero denominator: {t!r}')
            return _exact(p) / den
    raise UnsupportedToken(f'not an accepted literal: {t!r}')


# ---------------------------------------------------------------- parse
def parse(tokens, cname='<C>'):
    toks = [cname if t == '<constant>' else t for t in tokens]
    pos = 0

    def rec():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t in ARITY:
            return (t,) + tuple(rec() for _ in range(ARITY[t]))
        if SLOT_RE.match(t) or t.startswith('<C'):
            return ('slot', t)
        if t in ('x0', 'x1', 'x2'):
            return ('slot', '?' + t[1:])
        return ('lit', literal_value(t))

    tree = rec()
    if pos != len(toks):
        raise ValueError('trailing tokens')
    return tree


def slots_of(tokens, cname='<C>'):
    out = {}
    for t in tokens:
        m = SLOT_RE.match(t)
        if m:
            out[t] = m.group(1)
        elif t in ('x0', 'x1', 'x2'):
            out['?' + t[1:]] = '?'
        elif t == '<constant>':
            out[cname] = 'c'
    return out


# ---------------------------------------------------------------- contract evaluator
NAN, PINF, NINF = mpf('nan'), mpf('inf'), mpf('-inf')


def _z(v):
    """one zero -- every zero result is THE zero (sign erased)"""
    return mpf(0) if v == 0 else v


def _snap(v):
    """symbolic-cancellation floor: trig output below the dps noise floor is exact 0;
    above the pole floor it is a pole (undefined); the band between is ambiguous."""
    if misnan(v) or misinf(v):
        return v
    floor = mpf(10) ** (-mp.dps + 10)
    ceil = mpf(10) ** (mp.dps - 10)
    a = abs(v)
    if a == 0:
        return mpf(0)
    if a < floor:
        SNAP_EVENTS[0] += 1
        return mpf(0)
    if a > ceil:
        SNAP_EVENTS[0] += 1
        return NAN
    return v


def _int_or_none(b):
    if misnan(b) or misinf(b) or abs(b) > mpf('1e15'):
        return None
    ib = mp.floor(b)
    return int(ib) if b == ib else None


def c_div(a, b):
    if misnan(a) or misnan(b):
        return NAN
    if b == 0:
        return NAN if a == 0 else (PINF if a > 0 else NINF)  # c/0 = c*(1/0), 0/0 undef
    if misinf(a) and misinf(b):
        return NAN
    if misinf(b):
        return mpf(0)  # one zero
    return _z(a / b)


def c_pow(a, b):
    if not misnan(a) and a == 1:
        return mpf(1)
    if b == 0:
        return mpf(1)  # x^0 = 1 incl nan^0
    if misnan(a) or misnan(b):
        return NAN
    if a == 0:
        return mpf(0) if b > 0 else PINF  # 0^neg = 1/0^pos = +inf
    if not misinf(a) and abs(a) == 1 and misinf(b):
        return mpf(1)  # pow(+-1, +-inf) = 1
    if misinf(b):
        # pow(t, +-inf) uses the spike-step semantics (|t|<1 -> 0, |t|=1 -> 1,
        # |t|>1 -> +inf), magnitude-based INCLUDING negative bases -- an explicit
        # exception to naive limit-completion (the sign oscillates along the integer
        # lattice; the step takes the magnitude limit, deployed-consistent)
        aa = abs(a)
        if aa < 1:
            return mpf(0) if b > 0 else PINF
        return PINF if b > 0 else mpf(0)
    if misinf(a):
        if a > 0:
            return PINF if b > 0 else mpf(0)
        ib = _int_or_none(b)
        if ib is None:
            return NAN  # pow(-inf, non-integer) undefined
        if b > 0:
            return NINF if ib % 2 else PINF
        return mpf(0)  # one zero
    if a < 0:
        if abs(b) > mpf('1e15'):
            raise Unresolved()  # integrality/parity unknowable beyond the cap: an exact
                                # even integer 1e16 exists and is DEFINED, so returning
                                # nan there would be a wrong definite value
        ib = _int_or_none(b)
        if ib is None:
            return NAN  # non-integer exponent on a negative base: undefined
        v = mp.power(abs(a), b)
        return _z(-v if ib % 2 else v)
    if abs(b) > mpf('1e6'):
        raise Unresolved()
    return _z(mp.power(a, b))


def _trig(fn):
    def g(a):
        if misnan(a) or misinf(a):
            return NAN
        if abs(a) > mpf('1e12'):
            raise Unresolved()
        if fn is mp.tan:
            # pole PROXIMITY, not output magnitude: tan(atan(1e43)) = 1e43 is a genuine
            # huge value, not a pole -- an output-magnitude ceiling would snap it to nan
            # and manufacture spurious nonzero-measure disagreements. A pole is where
            # cos vanishes at working precision.
            c = mp.cos(a)
            if abs(c) < mpf(10) ** (-mp.dps + 10):
                SNAP_EVENTS[0] += 1   # a pole at working precision is a symbolic event:
                                      # f64 evaluates a DIFFERENT point (tan(f64-pi/2) =
                                      # 1.6e16), so the deployed check must be skipped --
                                      # omitting this flag manufactures false
                                      # ENGINE-MISALIGNs
                return NAN
            return _snap(mp.sin(a) / c)
        return _snap(fn(a))
    return g


def _oddroot(k):
    # the exponent 1/k MUST be built at CALL time: a module-level mpf(1)/3 freezes a
    # dps-15 (import-time) approximation of 1/3 into every evaluation, producing false
    # clause-(a) kills of exact odd-root identities (the dyadic roots 1/2, 1/4 are exact
    # and unaffected).
    def g(a):
        if misnan(a):
            return NAN
        if misinf(a):
            return a
        return _z(mp.sign(a) * mp.power(abs(a), mpf(1) / k))
    return g


def c_rootn(a, b):
    """IEEE rootn (the 0.12 general root operator) under the contract. The index must be
    a finite nonzero integer -- n == 0 or a non-integer index is an invalid operation
    (nan); n == 1 is the identity; a negative index is 1/rootn(a, -n) (one zero:
    rootn(0, -n) = +inf); odd n is the signed total root; even n is the principal root,
    nan on negatives including -inf."""
    if misnan(a) or misnan(b):
        return NAN
    ib = _int_or_none(b)
    if ib is None or ib == 0:
        return NAN
    if ib < 0:
        return c_div(mpf(1), c_rootn(a, mpf(-ib)))
    if ib == 1:
        return a
    if ib % 2 == 0:
        if a < 0:
            return NAN
        if misinf(a):
            return PINF
        return _z(mp.power(a, mpf(1) / ib))
    if misinf(a):
        return a
    return _z(mp.sign(a) * mp.power(abs(a), mpf(1) / ib))


OPS = {
    '+': lambda a, b: NAN if (misnan(a) or misnan(b)) else
         (NAN if (misinf(a) and misinf(b) and mp.sign(a) != mp.sign(b)) else _z(a + b)),
    '-': lambda a, b: NAN if (misnan(a) or misnan(b)) else
         (NAN if (misinf(a) and misinf(b) and mp.sign(a) == mp.sign(b)) else _z(a - b)),
    '*': lambda a, b: NAN if (misnan(a) or misnan(b)) else
         (NAN if ((a == 0 and misinf(b)) or (b == 0 and misinf(a))) else _z(a * b)),
    '/': c_div, 'inv': lambda a: c_div(mpf(1), a), 'pow': c_pow, 'rootn': c_rootn,
    'neg': lambda a: NAN if misnan(a) else _z(-a),
    'abs': lambda a: NAN if misnan(a) else _z(abs(a)),
    'exp': lambda a: NAN if misnan(a) else
           (mpf(0) if a == NINF else (PINF if a == PINF else
            (_z(mp.exp(a)) if abs(a) < mpf('1e5') else (_ for _ in ()).throw(Unresolved())))),
    'log': lambda a: NAN if misnan(a) else
           (NINF if a == 0 else (NAN if a < 0 else (PINF if a == PINF else _z(mp.log(a))))),
    'sin': _trig(mp.sin), 'cos': _trig(mp.cos), 'tan': _trig(mp.tan),
    'asin': lambda a: NAN if misnan(a) else
            ((_ for _ in ()).throw(Unresolved())
             if (not misinf(a) and abs(a) > 1 and abs(abs(a) - 1) < mpf(10) ** (-mp.dps + 10))
             else (NAN if abs(a) > 1 else _z(mp.asin(a)))),
    'acos': lambda a: NAN if misnan(a) else
            ((_ for _ in ()).throw(Unresolved())
             if (not misinf(a) and abs(a) > 1 and abs(abs(a) - 1) < mpf(10) ** (-mp.dps + 10))
             else (NAN if abs(a) > 1 else _z(mp.acos(a)))),
    'atan': lambda a: NAN if misnan(a) else
            (_z(mp.sign(a) * mp.pi / 2) if misinf(a) else _z(mp.atan(a))),
    'sinh': lambda a: NAN if misnan(a) else
            (a if misinf(a) else (_z(mp.sinh(a)) if abs(a) < mpf('1e5')
             else (_ for _ in ()).throw(Unresolved()))),
    'cosh': lambda a: NAN if misnan(a) else
            (PINF if misinf(a) else (_z(mp.cosh(a)) if abs(a) < mpf('1e5')
             else (_ for _ in ()).throw(Unresolved()))),
    # F104: BOUNDARY HONESTY AT THE ASYMPTOTE, the same doctrine `atanh` states below.
    # `tanh` is where the information is lost, so this is where the refusal belongs. Once
    # |tanh(x)| is inside the working-precision band of 1 the value is indistinguishable
    # from 1, and everything downstream inherits that: `asin(1) = pi/2` is returned as
    # though definite, `cos(pi/2)` is a ~1e-50 residue, and against a true `sech(1000)` of
    # 1e-434 that is a RELATIVE gap of 1.0 -- at EVERY rung, because tanh saturates at
    # every affordable precision. F103's decay test reads that stability as analytic, which
    # is how the EXACT `cos asin tanh _0 -> inv cosh _0` reached 6 of 167 grid points.
    # Saturation makes a false rule look true AND a true rule look false; this is the
    # second half.
    #
    # Refusing at `asin`/`acos` instead was tried and reverted: their +-1 is an ordinary
    # point with a finite value, so a band there also refuses the legitimate exact
    # `asin(-1) = -pi/2` -- 67 rules to UNRESOLVED-COVERAGE and three broken tests.
    #
    # This does NOT spare the shallow rows: at dps 50 the band is 1e-40 and
    # 1 - tanh(30) = 1.75e-26 sits far outside it, so `tanh * (-10) (-3) -> 1` is still
    # judged and still convicted. Rows that fall inside the band at rung 1 are settled by
    # the CLIMB in `_point_verdict` -- the band narrows as the precision rises.
    'tanh': lambda a: NAN if misnan(a) else (_z(mp.sign(a)) if misinf(a) else
            ((_ for _ in ()).throw(Unresolved())
             if abs(abs(_z(mp.tanh(a))) - 1) < mpf(10) ** (-mp.dps + 10)
             else _z(mp.tanh(a)))),
    'asinh': lambda a: NAN if misnan(a) else (a if misinf(a) else _z(mp.asinh(a))),
    'acosh': lambda a: NAN if misnan(a) else
             ((_ for _ in ()).throw(Unresolved())
              if (not misinf(a) and abs(a - 1) < mpf(10) ** (-mp.dps + 10)) else
              (NAN if a < 1 else (PINF if a == PINF else _z(mp.acosh(a))))),
    # BOUNDARY HONESTY: an argument within the working-precision band of a closed
    # boundary is indistinguishable from the boundary -- dps-50 rounds tanh(100) to
    # EXACTLY 1, and limit-completing that would fabricate atanh(tanh(x)) = +inf vs x,
    # false-killing an exact identity on the magnitude tail. Within the band:
    # Unresolved, never a verdict.
    'atanh': lambda a: NAN if misnan(a) else
             ((_ for _ in ()).throw(Unresolved())
              if (not misinf(a) and abs(abs(a) - 1) < mpf(10) ** (-mp.dps + 10)) else
              (NAN if abs(a) > 1 else _z(mp.atanh(a)))),
    'pow2': lambda a: c_pow(a, mpf(2)), 'pow3': lambda a: c_pow(a, mpf(3)),
    'pow4': lambda a: c_pow(a, mpf(4)), 'pow5': lambda a: c_pow(a, mpf(5)),
    'pow1_2': lambda a: c_pow(a, mpf(1) / 2), 'pow1_4': lambda a: c_pow(a, mpf(1) / 4),
    'pow1_3': _oddroot(3), 'pow1_5': _oddroot(5),
    'div2': lambda a: c_div(a, mpf(2)), 'div3': lambda a: c_div(a, mpf(3)),
    'div4': lambda a: c_div(a, mpf(4)), 'div5': lambda a: c_div(a, mpf(5)),
    'mult2': lambda a: OPS['*'](a, mpf(2)), 'mult3': lambda a: OPS['*'](a, mpf(3)),
    'mult4': lambda a: OPS['*'](a, mpf(4)), 'mult5': lambda a: OPS['*'](a, mpf(5)),
}


def c_eval(tree, env):
    """evaluate under the contract semantics (env values: mpf/mp expressions)"""
    op = tree[0]
    if op == 'slot':
        return env[tree[1]]
    if op == 'lit':
        v = tree[1]
        if v == math.pi:
            return mp.pi
        if v == math.e:
            return mp.e
        if isinstance(v, Fraction):
            # rebuilt per rung, like the symbolic battery points: the exact rational is
            # the contract's value, and its mpf reading follows whatever dps is current.
            return mpf(v.numerator) / mpf(v.denominator)
        return mpf(v)
    # LITERAL PROVENANCE: a boundary literal WRITTEN in the rule is the exact boundary
    # point. The honesty band exists for COMPUTED values that merely round into the
    # precision band of the boundary; it cannot apply to a token the rule itself spells.
    # atanh at a written +-1 limit-completes to +-inf; acosh at a written 1 is 0.
    if op in ('atanh', 'acosh') and len(tree) == 2 and tree[1][0] == 'lit':
        lv = tree[1][1]
        if op == 'atanh' and lv == 1.0:
            return PINF
        if op == 'atanh' and lv == -1.0:
            return NINF
        if op == 'acosh' and lv == 1.0:
            return mpf(0)
    args = [c_eval(c, env) for c in tree[1:]]
    v = OPS[op](*args)
    if _NODE_SINK and misinf(v):
        _NODE_SINK[-1][0] = True
    if _MAG_SINK:
        # The size of the largest INTERMEDIATE, which is what decides how much precision
        # a cancellation will eat. Only finite values count: an inf tells us nothing
        # about digits, and `misinf` above already records it.
        try:
            if not misinf(v):
                a = abs(v)
                if a > _MAG_SINK[-1][0]:
                    _MAG_SINK[-1][0] = a
        except Exception:
            pass
    return v


_NODE_SINK: list = []

#: Filled by `c_eval` when a `_measured_dps` scope is open; see `_required_dps`.
_MAG_SINK: list = []


def _required_dps(mag):
    """Precision that a cancellation between operands of size `mag` can demand.

    Adding two numbers of magnitude 10^k whose true sum is 10^-k destroys about 2k
    significant digits, so the working precision has to carry them before any correct
    digit survives. Measured against the case that motivated this: at t = -20,
    `cosh(25t) + sinh(25t)` has intermediates near e^500 ~ 10^217, and the sum only
    becomes representable somewhere past dps 400 -- which is what `2 * 217 + 50` gives.

    Derived, not tuned: the 2 is the two-sided digit loss, `k` is measured from the
    actual intermediates rather than guessed from the expression, and `BASE_DPS` is the
    ordinary working precision that must survive on top.
    """
    if mag is None or mag <= 1:
        return BASE_DPS
    try:
        k = int(mp.floor(mp.log10(mag))) + 1
    except Exception:
        return BASE_DPS
    return min(MAX_DPS, BASE_DPS + 2 * max(0, k))


def lhs_singular(tree, env):
    """True iff evaluating `tree` at env crosses an internal +-inf (a COMPUTED node,
    not a written literal): the point lies on the expression's singular manifold.

    The twin of ``_monitor._input_singular``; both instruments implement the same
    9.8.3(a) boundary and must move together. Exceptions fail False -- toward strict
    authority."""
    flag = [False]
    _NODE_SINK.append(flag)
    try:
        c_eval(tree, env)
    except Exception:
        return False
    finally:
        _NODE_SINK.pop()
    return flag[0]


# ---------------------------------------------------------------- deployed evaluator
import simplipy.operators as _spo


def _wrap_arr(f):
    # the numpy ARRAY path is the deployed consumer path (the 0-d scalar path
    # diverges at e.g. pow(-inf, 0.5): nan vs inf)
    def g(*a):
        r = f(*[np.array([x], dtype=F) for x in a])
        return F(np.asarray(r).ravel()[0])
    return g


def _legacy_even_root(p):
    # pre-0.12 pow1_2/pow1_4: x**p with an explicit nan branch for a -inf base
    # (IEEE (-inf)**0.5 is +inf; the realization was contract-aligned real semantics)
    def g(a):
        with np.errstate(invalid='ignore'):
            return F(np.where(np.isneginf(a), np.nan, np.asarray(a, dtype=F) ** F(p)))
    return g


def _legacy_odd_root(p):
    # pre-0.12 pow1_3/pow1_5: the real-valued (sign-preserving) root
    def g(a):
        a = np.asarray(a, dtype=F)
        with np.errstate(invalid='ignore'):
            return F(np.where(a < 0, -(-a) ** F(p), a ** F(p)))
    return g


DEP_OPS = {
    '+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
    '/': lambda a, b: F(np.divide(F(a), F(b))),
    'inv': lambda a: F(np.divide(F(1.0), F(a))),
    'neg': lambda a: -a, 'abs': np.abs,
    'div2': lambda a: a / F(2), 'div3': lambda a: a / F(3),
    'div4': lambda a: a / F(4), 'div5': lambda a: a / F(5),
    'mult2': lambda a: a * F(2), 'mult3': lambda a: a * F(3),
    'mult4': lambda a: a * F(4), 'mult5': lambda a: a * F(5),
    # The hyper-op realizations were deleted from simplipy.operators at 0.12 ("the
    # hyper-operator vocabulary is deleted, once and for all"), which silently emptied
    # these DEP_OPS slots via the getattr fallback below and let a planted gate poison
    # evaporate into NO-WITNESS (E1). Hand-coded here, transplanted from the pre-0.12
    # module, so the gate keeps judging LEGACY artifacts and its legacy-spelled poisons.
    'pow2': lambda a: F(a) ** F(2), 'pow3': lambda a: F(a) ** F(3),
    'pow4': lambda a: F(a) ** F(4), 'pow5': lambda a: F(a) ** F(5),
    'pow1_2': _legacy_even_root(0.5), 'pow1_4': _legacy_even_root(0.25),
    'pow1_3': _legacy_odd_root(1 / 3), 'pow1_5': _legacy_odd_root(1 / 5),
}
for _n in ARITY:
    _fn = getattr(_spo, _n, None)
    if callable(_fn):
        DEP_OPS[_n] = _wrap_arr(_fn)


def d_eval(tree, env):
    """evaluate under the deployed engine's operator algebra (numpy array path)"""
    op = tree[0]
    if op == 'slot':
        return env[tree[1]]
    if op == 'lit':
        # deployment parses the token as a double; the exact rational is the CONTRACT's
        # reading, never this lane's.
        return F(float(tree[1]))
    args = [d_eval(c, env) for c in tree[1:]]
    # C35: the ignore-everything float semantics this judge needs are SCOPED to the
    # operator application, never set process-wide at import.
    with np.errstate(all='ignore'):
        return F(DEP_OPS[op](*args))


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


def compare_classes(cl, cr):
    """The class half of a contract comparison -- everything nan/inf settles by itself.

    -> 'eq' | 'EXT' | 'SHRINK' (nan vs defined: clause b) | 'INF-CHANGE' (clause c),
    or None when BOTH sides are finite, where the gap ladder decides instead.
    """
    if cl[0] == 'nan' and cr[0] == 'nan':
        return 'eq'
    if cl[0] == 'nan':
        return 'EXT'
    if cr[0] == 'nan':
        return 'SHRINK'
    if cl[0] == 'inf' or cr[0] == 'inf':
        return 'eq' if cl == cr else 'INF-CHANGE'
    return None


def gap_reading(cl, cr):
    """One rung's FINITE reading -> `(|l - r|, scale)`, both ABSOLUTE.

    F105 (2026-08-21): THIS USED TO RETURN THE QUOTIENT, AND AGAINST AN EXACT ZERO THE
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


# F103 (2026-08-21): THE FINITE CASE HAS NO MAGNITUDE BAR AT ALL. It asks how the gap
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
# y = -50 -- exactly true, and the identity F99 recorded as wrongly KILLed -- carries a
# dps-50 residue of 8.5e-21 that ANY magnitude bar convicts; its gap still falls 10^69 by
# dps 120, so the ratio spares it. That is the failure the old comment called NOT YET
# FIXED, and it is what a size test structurally cannot see.
#
# This supersedes both earlier attempts. F99 deleted an ABSOLUTE 1e-25 floor because it
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
}


def saturation_verdict(tl, tr, env_mp):
    """F106: A BOUNDED FUNCTION NEVER ATTAINS ITS LIMIT AT A FINITE ARGUMENT, so a
    rewrite claiming it does is false -- by a fact about the FUNCTION, at no precision
    at all. -> 'REAL-CHANGE' when the claim is refuted, else None (no opinion).

    This is the family the gap ladder cannot reach, and depth is not the answer.
    `tanh(cosh(10))` differs from 1 by 2e-9566, inside F104's boundary band at every
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


#: The deployed-realisation bound, in ULP. DERIVED, not chosen -- see
#: `compare_deployed_realised`. Reproduce with the libm ULP measurement.
REALISED_ULP = 8


def _ulp_gap(a, b):
    """ULP distance between two finite doubles of the same sign; inf across a sign
    change, which is never a rounding difference."""
    if a == b:
        return 0
    if (a < 0) != (b < 0):
        return float('inf')
    return int(abs(np.float64(a).view(np.int64) - np.float64(b).view(np.int64)))


def _deployed_classes(cl, cr):
    """The nan/inf half of a deployed comparison, shared by both bars below."""
    if cl[0] == 'nan' and cr[0] == 'nan':
        return 'eq'
    if cl[0] == 'nan':
        return 'EXT'
    if cr[0] == 'nan':
        return 'SHRINK'
    if cl[0] == 'inf' or cr[0] == 'inf':
        return 'eq' if cl == cr else 'INF-CHANGE'
    return None


def compare_deployed_realised(cl, cr):
    """The REALISATION comparison: does the deployed f64 engine compute this rewrite?

    Answers `realised`, which decides the `f64` tier and so what ships in `rules.json`.
    Bounded at `REALISED_ULP` ULP, with NO absolute floor.

    WHY A BOUND, AND WHY THIS ONE. Three rejected answers first, because each names a
    constraint on the fourth:

    * `1e-9 * max(1.0, |a|, |b|)`, what this used to be: roughly 10^7 ULP, and for any
      value below 1 a pure ABSOLUTE floor. It read `pow np.e sinh (-5) -> 0` as an f64
      equality when the sides are 5.942307292381135e-33 and 0.0. Not a model of f64. It
      was harmless only while every fatal verdict was DELETED; once the symbolic gate
      began ROUTING on the tier it became an admission path into the default rule set,
      and 8 acj-4-3 rules took it.
    * Bit-exact: right in principle, wrong in practice. It convicts 390 acj-4-3 rules,
      310 of them over a SINGLE ULP, and it makes those 310 depend on the build host's
      libm -- so a pinned triple would not classify identically on another machine, which
      D29 byte-identity cannot survive.
    * 4 ULP, the first proposal: rejected precisely because the number had to be
      defensible. The observed rounding cluster runs 1..7 ULP, so a bar at 4 splits one
      phenomenon down the middle.

    THE DERIVATION. IEEE-754 mandates correct rounding only for + - * / and sqrt; every
    transcendental is implementation-defined, and libm documents an error rather than
    promising exactness. Measured on this host (glibc 2.43) against mpmath at dps 60 over
    the judge's OWN grid (the libm ULP measurement): every function is correctly rounded
    except `cos`, `cosh`, `sinh` and `tanh`, which reach 1 ULP. A rewrite has TWO sides,
    each a composition of up to about four such calls, with independent errors that can
    oppose: 2 x 4 x 1 = 8 ULP.

    WHY 8 IS NOT TUNED. Over the 390 rules bit-exactness moves, the gap distribution is
    4 at 0 ULP (class changes), 310 at 1 ULP, then 5/2/4/0/2/2 at 2..7 -- and then
    NOTHING until 57. Every bound in [8, 56] gives the IDENTICAL partition, so the
    classification does not depend on where in that range the bar sits. That
    insensitivity is what makes this a derivation rather than a preference.

    NOT used for ENGINE-MISALIGN -- see `compare_deployed_structural`. Answering both
    questions with one comparison gave 396 ENGINE-MISALIGNs against a true 10.
    """
    c = _deployed_classes(cl, cr)
    if c is not None:
        return c
    return 'eq' if _ulp_gap(cl[1], cr[1]) <= REALISED_ULP else 'REAL-CHANGE'


def compare_deployed_structural(cl, cr, tol_rel=1e-9):
    """The STRUCTURAL deployed comparison, for ENGINE-MISALIGN only.

    ENGINE-MISALIGN means "the contract certifies but the deployed algebra structurally
    DIVERGES" -- an infinity where a finite value belongs, a nan where a number belongs,
    or a gap far outside rounding. A few ULP is none of those, and convicting it would
    rename ~400 sound rules as misaligned (measured: 396 against a true 10). So this
    keeps the band `compare` used to carry for f64, and realisation gets the ULP bar
    above. Two questions, two comparisons.
    """
    c = _deployed_classes(cl, cr)
    if c is not None:
        return c
    a, b = cl[1], cr[1]
    return 'eq' if abs(a - b) <= tol_rel * max(1.0, abs(a), abs(b)) else 'REAL-CHANGE'


# ---------------------------------------------------------------- batteries
def battery_reals():
    """symbolic contract points, rebuilt at the CURRENT dps (per-rung builders)"""
    return [mpf(0), mpf(1) / 2, -mpf(1) / 2, mpf(1), mpf(-1), mpf(2), mpf(-2),
            mpf(3), mpf(-3), mpf(1) / 4, -mpf(1) / 4, mpf(3) / 2, -mpf(3) / 2,
            mp.pi / 2, -mp.pi / 2, mp.pi, mp.pi / 4, mp.pi / 3, mp.pi / 6,
            mp.e, mpf('-1.7')]


def battery_for(sort):
    """-> list of (builder() -> mp value, tag); tags: real | dirac-inf | nullx"""
    reals = [((lambda v=i: battery_reals()[v]), 'real') for i in range(len(battery_reals()))]
    if sort == '?':
        return reals  # data is real; +-inf is outside the quantifier
    if sort == '_':
        return reals + [((lambda: PINF), 'dirac-inf'), ((lambda: NINF), 'dirac-inf')]
    if sort == '!':
        return reals + [((lambda: PINF), 'nullx'), ((lambda: NINF), 'nullx')]
    if sort == '$':
        # the mult-certified sort: expressions certified FINITE AND NONZERO a.e.
        # (`mult_certified` -- the licence behind f/f -> 1). The quantifier's bulk is
        # nonzero reals; 0 and +-inf occur on null sets only, so events there are
        # null-excused (same doctrine as `!`'s +-inf atoms).
        nz = [(b, t) for b, t in reals if b() != 0]
        return nz + [((lambda: mpf(0)), 'nullx'),
                     ((lambda: PINF), 'nullx'), ((lambda: NINF), 'nullx')]
    raise ValueError(sort)


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


def judge_cl_battery(saturation=True):
    """judging battery for a SOURCE-side constant (forall c_s over R). Includes the
    special rationals (pow(1,nan)=1 class) AND the SYMBOLIC transcendental atoms,
    built at the CURRENT dps: a fitted constant reaches pi/2, and pow(sin(c),inf)
    at c = pi/2 is the powsin class in constant space -- an f64 pi/2 misses the
    coincidence by 5e-33 (a false rescue can otherwise slip through; the symbolic
    constant atoms kill it correctly). A cl whose witness is unfittable (degenerate
    LHS there) is skipped with a tally; only core-CONSTS failures mean NO-WITNESS."""
    base = ([(lambda c=c: mpf(c)) for c in CONSTS] +
            [(lambda: mpf(1)), (lambda: mpf(-1)), (lambda: mpf(0)),
             (lambda: mpf(2)), (lambda: mpf(-2)), (lambda: mpf(4)),
             (lambda: mpf(-4)), (lambda: mpf(5)), (lambda: mpf(-5)),
             (lambda: mp.pi / 2), (lambda: -mp.pi / 2), (lambda: mp.pi),
             (lambda: mp.e)])
    if not saturation:
        return base
    return base + [(lambda c=c: mpf(c)) for c in SATURATION_CONSTS]
GRID = np.concatenate([np.linspace(-3, 3, 121), np.linspace(-20, 20, 41),
                       np.array([-1e4, -1e3, -300., -100., -50., -30.,
                                 30., 50., 100., 300., 1e3, 1e4])]) + 0.0137421
# dense center (0.05 spacing: sub-unit undefined intervals like (0.2, 0.8) are visible),
# unit-spaced mid-range, and a magnitude tail (half-line violations opening at |x|>20,
# e.g. exp-shift constructions)
# DEDUPED (F96): `linspace(-3,3,121)` and `linspace(-20,20,41)` overlap, so seven core
# values (-2.9862579, -1.9862579, -0.9862579, 0.0137421, 1.0137421, 2.0137421, 3.0137421)
# were double-weighted -- 174 entries for 167 distinct points, silently giving those seven
# twice the say in `meas`. Order preserved; the point SET is unchanged.
GRID = list(dict.fromkeys(GRID))

MEASURE_KILL = 0.05  # fraction of the generic grid; a genuine positive-measure set
                     # shows as many points; single-point flukes never reach this


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
_XS_TAIL = np.logspace(12, 300, 49)[1:]

#: The dense body is the shipped grid, unchanged -- the tail only adds reach.
_XS = np.concatenate([-_XS_TAIL[::-1], -np.logspace(12, -3, 120),
                      np.linspace(-3, 3, 81), np.logspace(-3, 12, 120), _XS_TAIL])


def fit_witness(tl, tr, shared, cl_val=None):
    """fit the exists-witness for a RHS <constant> on generic reals (deployed algebra);
    validated on two further generic envs; None if no witness exists/found"""
    def lhs_at(env):
        e = dict(env)
        if cl_val is not None:
            e['<C_L>'] = F(cl_val)
        return d_eval(tl, e)

    def rhs_at(env, c):
        e = dict(env)
        e['<C_R>'] = F(c)
        if cl_val is not None:
            e['<C_L>'] = F(cl_val)
        return d_eval(tr, e)

    env0, L0 = None, None
    for k in (0, 1, 4):
        try:
            e0 = _gen_env(shared, k)
            v = lhs_at(e0)
            if np.isfinite(v):
                env0, L0 = e0, v
                break
        except Exception:
            pass
    if env0 is None:
        return None

    def g(c):
        try:
            v = rhs_at(env0, c)
            return v - L0 if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    vals = np.array([g(x) for x in _XS])
    ok = np.isfinite(vals)
    best = None
    for i in range(len(_XS) - 1):
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
        for c in [1., -1., 2., -2., 0.5, -0.5, math.e, -math.e, math.pi, -math.pi]:
            v = g(c)
            if np.isfinite(v) and abs(v) <= 1e-9 * max(1.0, abs(L0)):
                best = c
                break
    if best is None:
        return None
    for c in [1., -1., 2., -2., 3., -3., 0.5, -0.5, math.e, -math.e, math.pi, -math.pi,
              round(best), round(best * 2) / 2]:
        # RELATIVE snap only (1e-15 floor = f64 fit noise around an exact zero): the old
        # absolute 1e-6 tolerance flattened every tiny TRUE witness to round(best) = 0
        # (pow(exp(-5), pi) = 1.5e-7 snapped to 0), fabricating clause-(a) kills of
        # sound constant-space rules
        if abs(c - best) < max(1e-9 * max(abs(best), abs(c)), 1e-15):
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


def mp_polish(tl, tr, nvars, clb, c0):
    """Refine an f64-fitted exists-witness to mp precision against the CONTRACT
    evaluator (the quantifier ranges over R; the f64 fit is only a starting point --
    left at f64 it manufactures 1e-17 'real changes' against exact transcendental
    literals like np.e, producing false clause-(a) kills). mp secant at a generic
    point; falls back to the f64 value if the landscape defeats it (any residual kill
    then surfaces in the sweep for investigation, never silently)."""
    old = mp.dps
    try:
        # THE WITNESS MUST OUT-RESOLVE THE LADDER. `<C_L>` is rebuilt exactly at each
        # rung while `<C_R>` is this fitted value merely re-rounded, so a witness good to
        # only 45 digits puts a FROZEN ~1e-45 gap between the two sides -- one that does
        # not decay, and that the F103 decay test therefore reads as analytic. That
        # convicted the identity `pow abs ?0 <constant> -> pow abs ?0 <constant>`, whose
        # two sides are the same tokens. Fitting deeper than the deepest rung keeps the
        # comparison a statement about the FUNCTIONS rather than about this fit.
        def build_env():
            e = {}
            for i, n in enumerate(sorted(nvars)):
                e[n] = mpf(GEN[i % len(GEN)])
            if clb is not None:
                # the EXACT battery value (symbolic builder), NOT its f64 rendering: a
                # witness polished against f64-pi differs from the judge's exact mp.pi by
                # ~4e-17, producing false clause-(a) kills of the witness family. Rebuilt
                # after the depth is settled, since a symbolic atom is only as good as
                # the dps it was rendered at.
                e['<C_L>'] = clb()
            return e

        # ...AND THE LADDER'S DEPTH IS OPERAND-SCALED (F96), so `GAP_RUNGS[-1]` stopped
        # naming its deepest rung. Size the rule's own intermediates first and match the
        # depth the ladder will actually climb to. Measured on `log(cosh(C)) -> c_t` at
        # C = 710: cosh(710) ~ 1.1e308 puts rung 2 at dps 668 and rung 3 at dps 1336,
        # against a witness frozen at 300 -- so the SAME ~1e-300 residue sat at both
        # rungs, did not decay, and the F103 decay test read it as an analytic gap. A
        # clause-(a) KILL of a rule that is true by construction. Costs nothing for an
        # ordinary rule: small intermediates give back exactly the old 300.
        mp.dps = GAP_RUNGS[-1] + 50
        _MAG_SINK.append([mpf(0)])
        try:
            c_eval(tl, build_env())
        except Unresolved:
            return mpf(c0)
        finally:
            _mag = _MAG_SINK.pop()[0]
        mp.dps = max(mp.dps,
                     max(GAP_RUNGS[2], 2 * max(GAP_RUNGS[1], _required_dps(_mag))) + 50)
        # `_WIT_RESID` was 10^-270 = "deeper than the deepest rung" when the deepest rung
        # was a fixed 250. It has to follow the ladder for the same reason the depth does.
        _wit_resid = mpf(10) ** (-(mp.dps - 30))
        env = build_env()
        try:
            target = c_eval(tl, env)
        except Unresolved:
            return mpf(c0)
        if misnan(target) or misinf(target):
            return mpf(c0)

        def accepts(residual):
            """Is this residual inside the noise of THIS computation?

            The test used to be `|r| <= _WIT_RESID * max(1, |target|)`, which for every
            target below 1 is a pure ABSOLUTE floor -- the shape F99 deleted from the
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
            """
            noise = max(_mag, mpf(1)) * mpf(10) ** (-(mp.dps - 25))
            return abs(residual) <= max(_wit_resid * abs(target), noise)

        def h(c):
            e = dict(env)
            e['<C_R>'] = c
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
                    return mpf(0)
            except Unresolved:
                return mpf(0)
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
                                return mpf(0)   # snap near-zero polish results to exact 0
                        except Unresolved:
                            pass
                    return c
            if best is not None and best_r is not None:
                try:
                    if best_r < abs(h(mpf(c0))):
                        return best        # short of target, but better than the seed
                except Unresolved:
                    return best
        except (Unresolved, ZeroDivisionError):
            pass
        return mpf(c0)
    finally:
        mp.dps = old


# ---------------------------------------------------------------- the judge
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
    residue decays with the working precision and an analytic gap does not (F103, above
    `compare_classes`). The identity above is spared on decay -- its dps-50 residue of
    8.5e-21 would fail any magnitude bar, and still falls 10^69 by dps 120.

    The class half is unchanged and keeps the two/three-rung rule, because a class has no
    gap to watch. It is also the remaining hole: where cancellation drives a sum to a
    computed EXACT ZERO the left side reads `log(0) = -inf`, so the disagreement is a
    CLASS change and no decay test can fire on it. Rung 2's OPERAND SCALING is what
    addresses that -- at t = -20 the cancellation is ~217 digits deep, and the rung sized
    from the operands actually seen reaches past it where a fixed 120 never could.

    A SECOND hole, and F103 opened it wider before F104 closed it. The three EXACT
    identities that were three grid points from conviction -- `cos asin tanh _0 ->
    inv cosh _0` and two siblings -- moved TOWARD the bar rather than away from it: the bc
    measure rose from under 0.0167 to 0.03593 against a kill bar of 0.05. The decay test
    was never the cause at the point level; at the convicting grid point
    (_0 = -19.9862579) both the contract evaluator and bare mpmath put the gap at 10^-34.5
    (dps 50) and 10^-106.2 (dps 120), a drop of 71.6 that reads as residue and returns
    'eq'. The cause was SATURATION at `tanh`'s asymptote, where a rounded +-1 makes a
    relative gap of 1.0 that is identical at every affordable rung -- stability the decay
    test reads as analytic. It is refused at the asymptote now (the band in `OPS`) and the
    rows that refuse at rung 1 are settled by the CLIMB below. Measured after both: those
    identities sit at 0 of 167 grid points, better than the 2 of 167 they scored before
    F103 existed.
    """
    def once(dps, track=False):
        """One reading at `dps` -> (class_verdict|None, gap|None), or None if Unresolved.

        `class_verdict` is set whenever nan/inf is involved -- there the classes are the
        whole finding. Otherwise both sides are finite and `gap` carries it. With `track`,
        also report the largest finite intermediate either side produced, which is what
        sizes a cancellation."""
        mp.dps = dps
        if track:
            _MAG_SINK.append([mpf(0)])
        try:
            cl = cls_mp(c_eval(tl, env_mp()))
            cr = cls_mp(c_eval(tr, env_mp()))
        except Unresolved:
            return None
        finally:
            if track:
                once.mag = _MAG_SINK.pop()[0]
        v = compare_classes(cl, cr)
        return (v, None) if v is not None else (None, gap_reading(cl, cr))

    old = mp.dps
    snap0 = SNAP_EVENTS[0]
    try:
        once.mag = None
        r1 = once(BASE_DPS, track=True)
        snapped1 = SNAP_EVENTS[0] > snap0
        # RUNG 2 IS OPERAND-SCALED. A fixed 120 confirms nothing when the intermediates
        # are 10^217: both rungs are swamped alike, agree on a manufactured verdict, and
        # the class comparison reads that agreement as evidence. The precision now comes
        # from the operands that were actually seen, so rung 2 is a second opinion rather
        # than a second copy of the first.
        dps2 = max(GAP_RUNGS[1], _required_dps(getattr(once, 'mag', None)))
        dps3 = max(GAP_RUNGS[2], 2 * dps2)

        # CLIMB PAST AN UNRESOLVED RUNG. A rung refuses when an intermediate lands inside
        # its boundary-honesty band, and that band NARROWS as the precision rises:
        # tanh(60) is inside it at dps 50 (band 1e-40) and well outside at dps 120
        # (1e-110). Bailing out at rung 1 therefore abstained on rows a higher rung can
        # settle -- measured, when F104 put the band on tanh, that short-circuit sent 275
        # rules of rules_real.json to UNRESOLVED-COVERAGE and dropped the kills from 65 to
        # 27. An Unresolved is a statement about the RUNG, not about the rule.
        rungs = (BASE_DPS, dps2, dps3)
        r_lo = lo_dps = hi_dps = None
        for i in range(len(rungs) - 1):
            r = r1 if i == 0 else once(rungs[i])
            if r is not None:
                r_lo, lo_dps, hi_dps = r, rungs[i], rungs[i + 1]
                break
        if r_lo is None:
            # Refused at every rung we can afford -- so ask the one question that needs
            # no rung at all (F106). Reached only here, so it cannot move a verdict the
            # ladder was able to reach.
            return saturation_verdict(tl, tr, env_mp), snapped1
        cls1, gap1 = r_lo

        if cls1 is not None:
            # nan/inf: a CLASS is confirmed by agreement at a second rung -- there is no
            # gap to watch decay, so this half keeps the two/three-rung rule unchanged.
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

        # BOTH FINITE -- decided by DECAY, never by size (see F103 above).
        hi = once(hi_dps)
        if hi is None or hi[0] is not None or hi[1] is None:
            return None, snapped1            # class flipped / Unresolved: never convict
        d_lo, s_lo = gap1
        d_hi, s_hi = hi[1]
        # ONE SCALE FOR BOTH RUNGS (F105). The separations are absolute; the scale is
        # fixed here, so it divides out of the ratio and touches nothing but the floor.
        # It is the largest of what either rung SAW and what the computation was made
        # of -- an exactly-cancelling source offers no scale from its own value, and
        # `_MAG_SINK`'s intermediate is then the only honest one available.
        mag = getattr(once, 'mag', None) or mpf(0)
        scale = max(s_lo, s_hi, abs(mag))
        if scale == 0:
            return 'eq', snapped1            # exactly zero on both sides at both rungs
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
        g_hi = d_hi if d_hi != 0 else scale * mpf(10) ** (-hi_dps)
        # Residue falls with the added precision; an analytic gap does not move, and a
        # gap that only BECOMES visible higher up (deep saturation) drops by less than
        # nothing. Half the added precision separates them with >=30 decades to spare.
        drop = mp.log10(g_lo / g_hi)
        return ('eq' if drop > (hi_dps - lo_dps) / 2
                else 'REAL-CHANGE'), snapped1
    finally:
        mp.dps = old


#: The four cells of the soundness LATTICE. The two notions are INCOMPARABLE -- a rule
#: can be true and unrealised (`atanh(tanh t) -> t`, exact on R, `inf` in f64) or
#: realised and untrue (`asin(1e-8) -> 1e-8`, bit-identical in f64, wrong by 1.7e-17) --
#: so this is an axis, not a rung on the Mode ladder.
#:
#:   core    true AND realised     served by every mode
#:   real    true, NOT realised    exact arithmetic is the authority
#:   f64     realised, NOT true    the deployed evaluator is the authority
#:   reject  neither               dropped everywhere
def _tier(verdict, realised):
    if verdict in ('CERTIFIED', 'TOLERATED'):
        # `real`, NOT `f64`. The contract accepting a rule means it is TRUE; `realised is
        # False` means the deployed evaluator disagrees. "True, not realised" is the `real`
        # row of the table above, and filing it as `f64` was wrong in both directions at
        # once -- it put the rule in the one mode whose authority CONTRADICTS it and
        # removed it from the mode that honours it. Reachable, and measured: 6 rules of
        # acj-4-3 land here, all TOLERATED, e.g. `/ $0 $0 -> 1` (true off the null set
        # {0}, and f64 answers nan at 0 and at both infinities).
        # `core` now requires realised to be TRUE, not merely "not False" (owner ruling,
        # 2026-08-20). `None` means the deployed lane never got an observation at all --
        # ABSENT EVIDENCE -- and that cannot support a claim of f64 soundness. 102
        # acj-4-3 rules take this path, dominated by the symbolic-cancellation snap
        # family: `sin(np.pi) -> 0` is exactly true in mathematics while f64 computes
        # 1.2246467991473532e-16, so serving it in f64 mode CHANGES the answer.
        return 'core' if realised is True else 'real'
    if verdict == 'ENGINE-MISALIGN':
        return 'real'
    if verdict == 'KILL':
        return 'f64' if realised is True else 'reject'
    return 'reject'


def judge_rule(lhs, rhs, deployed_check=True):
    """-> dict: verdict CERTIFIED | TOLERATED | KILL | ENGINE-MISALIGN | NO-WITNESS,
    with clause/class, points, measures. Implements the kill bar:
      KILL   iff a REAL-CHANGE exists at any resolved point (clause a)
             or the generic-grid disagreement measure exceeds MEASURE_KILL (b/c);
      ENGINE-MISALIGN iff the contract certifies but the deployed algebra
             structurally diverges at a non-gap battery point / on the grid;
      TOLERATED else-if any null-event disagreement exists (documented class);
      CERTIFIED otherwise."""
    try:
        tl = parse(lhs, '<C_L>')
        tr = parse(rhs, '<C_R>')
    except UnsupportedToken as ex:
        # a token outside the numeric grammar: refuse the SHAPE, do not evaluate it
        return {'verdict': 'UNSUPPORTED-SHAPE', 'detail': str(ex)}
    sl = slots_of(lhs, '<C_L>')
    sr = slots_of(rhs, '<C_R>')
    slots = dict(sl)
    slots.update({k: v for k, v in sr.items() if k not in slots})
    nvars = {n: s for n, s in slots.items() if not n.startswith('<C')}
    has_cl, has_cr = '<C_L>' in slots, '<C_R>' in slots
    if lhs.count('<constant>') > 1:
        # the deployed matcher binds multiple <constant> leaves INDEPENDENTLY; this
        # judge models one shared symbol (diagonal only) -- rather than silently
        # under-judging the off-diagonal, the shape is refused fail-closed. The
        # detail is the SENTINEL the gate sites key their jurisdiction carve-out on
        # (B7/D36): this return is a statement of non-jurisdiction made BEFORE any
        # judging (parse has already succeeded, so an unknown token cannot reach it
        # -- that raises UnsupportedToken above with its own detail), and the
        # multi-Const family's soundness authority is the Const-channel chain
        # (constant fitting + the translation count gate + the licence registry).
        return {'verdict': 'UNSUPPORTED-SHAPE', 'detail': CONST_CHANNEL_DETAIL}

    # cl battery entries: (builder, f64key). builder() gives the EXACT value at the
    # current dps (symbolic atoms rebuild per rung); f64key indexes the witness maps
    # and feeds the f64 fitter / deployed check. TWO witness maps, one per algebra:
    # the contract compares against the mp-polished witness, the deployed check against
    # the raw f64 fit -- deployment fits its OWN constant, so judging the f64 algebra
    # with the mp witness manufactures false ENGINE-MISALIGNs across f64 saturation
    # cliffs (asin(tanh(exp(3))): f64 sees exactly pi/2, the contract pi/2 - 2.3e-9).
    witness = {}
    witness_f64 = {}
    # THE SATURATION PROBES ARE WITHHELD WHERE A WITNESS MUST BE RE-FITTED AT THEM.
    # Measured over the shipped rules.json: on the 71 rows whose constant appears only on
    # the SOURCE side, the probes move exactly 16 rows and every one moves cleanly from
    # `core` to `real` -- they are the rules the deep audit found unrealised in f64. On
    # the 272 rows that also fit a TARGET constant they produce 88 clause-(a) KILLs, and
    # every one diagnosed is an artifact of the exists-witness machinery at magnitudes it
    # cannot resolve, not a false rule. So the probe is put where its answer can be read
    # and withheld where it cannot; `saturation_probed` says which happened, because a
    # withheld question that nobody records is the `skipped_cl` defect again.
    saturation_probed = not has_cr
    if has_cl:
        cl_battery = [(b, float(b()))
                      for b in judge_cl_battery(saturation=saturation_probed)]
    else:
        cl_battery = [(None, None)]
    core = {float(c) for c in CONSTS}
    skipped_cl = []
    unwitnessed_cl = []
    if has_cr:
        kept_cl = []
        any_fit = False
        for b, key in cl_battery:
            w = fit_witness(tl, tr, set(nvars), key)
            if w is None:
                # distinguish: LHS UNDEFINED at this cl for the generic envs -> the rows
                # are tolerated degenerate extensions (the acos(pow(c,-inf)) class),
                # skip WITH tally; LHS defined but unfittable -> a genuine exists-failure
                lhs_defined = False
                for k in (0, 1, 4):
                    try:
                        v = d_eval(tl, dict(_gen_env(set(nvars), k),
                                            **({'<C_L>': F(key)} if key is not None else {})))
                        if np.isfinite(v):
                            lhs_defined = True
                            break
                    except Exception:
                        pass
                if lhs_defined:
                    # the LHS HAS a value here and no c_t reproduces it: an
                    # exists-FAILURE, which is a different fact from the degenerate skip
                    # below. A core cl convicts immediately, as before. A NON-core cl
                    # used to be appended to `skipped_cl` and forgotten -- and
                    # `skipped_cl` is returned by every verdict yet read by nobody, so
                    # the rule went on to be CERTIFIED on the strength of the constants
                    # that happened to fit. Recorded apart, and made verdict-bearing.
                    if key is None or key in core:
                        return {'verdict': 'NO-WITNESS',
                                'detail': f'cl={key} (LHS defined)'}
                    unwitnessed_cl.append(key)
                    continue
                # LHS UNDEFINED here: the rule is vacuous at this cl, a real skip
                skipped_cl.append(key)
                continue
            any_fit = True
            witness[key] = mp_polish(tl, tr, nvars, b, w)
            witness_f64[key] = w
            kept_cl.append((b, key))
        if has_cr and not any_fit:
            return {'verdict': 'NO-WITNESS', 'detail': 'no cl with a finite LHS/witness'}
        if unwitnessed_cl:
            # forall-c_s exists-c_t is falsified at a constant where the LHS is defined.
            # Reported for ALL such constants rather than the first, because the set is
            # the evidence: one is a corner, a spread is a false rule.
            return {'verdict': 'NO-WITNESS', 'skipped_cl': skipped_cl,
                    'detail': f'cl={sorted(unwitnessed_cl)} (LHS defined)'}
        cl_battery = kept_cl if has_cl else [(None, None)]

    a_kills, tolerated, dep_div = [], [], []
    # F100, the REALISATION axis. `dep_div` answers "the contract certifies but
    # deployment diverges" and is therefore gated on the contract agreeing -- which
    # means a KILLed rule's realisation was never recorded, and the SOUND/LOSSY tier
    # split needs exactly that. `dep_bad`/`dep_seen` record the same comparison with no
    # gate, so a verdict and a realisation can be reported independently.
    dep_bad, dep_seen = [], [False]
    resolved_pts = 0
    attempted_pts = 0
    # battery sweep (cap the slot-product deterministically)
    names = sorted(nvars)
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
            # exhaustively and a >=3-slot sample is drawn from the COMPLETE two-slot
            # base. The literal 500 this replaces was 29 short of 23^2 = 529 while its
            # comment claimed the opposite. Two consequences, not one: it dropped 29
            # two-slot pairs outright, and -- because the cap applies once per slot as
            # the product is built -- combos extending those 29 could never enter a
            # 3-slot sample either. A systematic blind spot, not a smaller sample.
            rng = np.random.default_rng(11)
            combos = [combos[i]
                      for i in rng.choice(len(combos), BATTERY_CAP, replace=False)]
    for clb, clkey in cl_battery:
        cr = witness.get(clkey) if has_cr else None
        crd = witness_f64.get(clkey) if has_cr else None
        for combo in combos:
            tags = {n: t for n, (b, t) in combo.items()}

            def env_mp():
                e = {n: b() for n, (b, t) in combo.items()}
                if clb is not None:
                    e['<C_L>'] = clb()  # exact symbolic constant at the CURRENT dps
                if cr is not None:
                    e['<C_R>'] = +cr    # mpf witness, re-rounded to the current dps
                return e

            v, snapped = _point_verdict(tl, tr, env_mp)
            attempted_pts += 1
            if v is not None:
                resolved_pts += 1
            point = {n: float(b()) for n, (b, t) in combo.items()}
            if clkey is not None:
                point['<C_L>'] = clkey
            if cr is not None:
                point['<C_R>'] = float(cr)
            if v == 'REAL-CHANGE':
                if all(t == 'real' for t in tags.values()):
                    # SINGULAR-INPUT gate (2026-08-11), the twin of the monitor's:
                    # clause (a)'s authority is REAL points where mathematics answers.
                    # A real INPUT BINDING can still land on the LHS's own singular
                    # manifold (x1 = x3 zeroes a denominator); there the LHS value is a
                    # convention-completion, spelling-dependent under ANY +-inf-valued
                    # extension, and a bounded compressor above it renders the
                    # disagreement real-vs-real. Clause (c) territory, like @ext.
                    try:
                        singular = lhs_singular(tl, env_mp())
                    except Exception:
                        singular = False
                    if singular:
                        tolerated.append(('REAL-CHANGE@singular', point, tags))
                    else:
                        a_kills.append(point)
                else:
                    # a real-VALUED disagreement at an extension-INPUT point (+-inf
                    # binding) is convention-mediated (it arose through extended ops:
                    # e.g. one-zero's inv(-inf)=0 vs the composition limit) -- clause (c)
                    # territory, not clause (a): clause (a)'s authority is REAL points,
                    # where mathematics answers.
                    tolerated.append(('REAL-CHANGE@ext', point, tags))
            elif v in ('EXT', 'SHRINK', 'INF-CHANGE'):
                tolerated.append((v, point, tags))
            all_real = all(t == 'real' for t in tags.values())
            if deployed_check and not snapped:
                try:
                    ed = {n: F(float(b())) for n, (b, t) in combo.items()}
                    if clkey is not None:
                        ed['<C_L>'] = F(clkey)
                    if crd is not None:
                        ed['<C_R>'] = F(crd)   # the f64-fitted witness: each algebra
                                               # is judged with its OWN constant
                    _dcl = cls_np(d_eval(tl, ed))
                    _dcr = cls_np(d_eval(tr, ed))
                    dv = compare_deployed_realised(_dcl, _dcr)
                    dep_seen[0] = True
                    if dv != 'eq':
                        dep_bad.append((dv, point))
                    # ENGINE-MISALIGN keeps its ORIGINAL gate AND its own STRUCTURAL
                    # comparison: a divergence where the contract also disagrees is not
                    # misalignment, one at an inf binding is convention-mediated, and a
                    # few-ULP rounding is not a divergence at all.
                    if compare_deployed_structural(_dcl, _dcr) != 'eq' \
                            and v in (None, 'eq') and all_real:
                        dep_div.append((dv, point))
                except Exception:
                    pass

    # measure scan: generic grid per slot, identity binding, contract semantics
    meas = 0.0
    old = mp.dps
    mp.dps = 50
    try:
        for cfg in (1.7, -1.3):
            for n in names:
                res = []
                for g in GRID:
                    e = {m: mpf(cfg) for m in names}
                    e[n] = mpf(float(g))
                    if has_cl:
                        e['<C_L>'] = mpf(CONSTS[0])  # measure scan: generic cl suffices
                    if has_cr:
                        wv = witness.get(float(CONSTS[0]) if has_cl else None)
                        if wv is None:
                            continue
                        e['<C_R>'] = +wv
                    # PRECISION-COUPLED (F96). This lane took a SINGLE dps-50 reading,
                    # while `_point_verdict` -- the battery's own comparison -- has always
                    # confirmed each non-eq verdict at a second rung and returned
                    # Unresolved on rung disagreement. The gap was measurable:
                    # `log(cosh t + sinh t) -> t` scores 5/174 at dps 50, 3/174 at 120,
                    # 2/174 at 400 and 0/174 at 9000, because mpmath loses ~0.87|t| digits
                    # to catastrophic cancellation there. Every one of those points is a
                    # TRUE identity, and the only positive measure this judge has ever
                    # recorded on a shipped artifact was that noise -- four more negative
                    # tail points and it would have convicted a true rule. Escalate, and
                    # never convict on a reading a second rung will not repeat.
                    snap0 = SNAP_EVENTS[0]
                    cv, _snapped = _point_verdict(tl, tr, lambda e=e: e)
                    if cv is None:
                        continue
                    res.append(cv)
                    # DEPLOYED LANE (F95). ENGINE-MISALIGN asks "does the shipped f64
                    # engine realise what the contract certified?", and until now the only
                    # place that question was ever put ran over the BATTERY, whose largest
                    # magnitude is 3.0 -- below every point where f64 departs from the
                    # extended reals (tanh saturates at 18.99, exp overflows at 709.78 and
                    # underflows at -745.13, cosh/sinh overflow at 710.48). The verdict was
                    # therefore unreachable in principle, not merely unfired. This grid
                    # already reaches 1e4, so the same comparison the battery lane performs
                    # is put at every grid point too.
                    #
                    # Guards mirror the battery lane exactly: only where the CONTRACT is
                    # 'eq' (misalignment is contract-certifies-but-deployment-diverges, so
                    # a contract disagreement is a different verdict), and never across a
                    # symbolic snap, where f64 evaluates a genuinely different point and a
                    # divergence would be a measurement artifact. `<C_R>` binds the RAW f64
                    # fit, never the mp-polished witness: deployment fits its OWN constant.
                    if not (deployed_check and SNAP_EVENTS[0] == snap0):
                        continue
                    ed = {m: F(cfg) for m in names}
                    ed[n] = F(float(g))
                    if has_cl:
                        ed['<C_L>'] = F(CONSTS[0])
                    if has_cr:
                        wf = witness_f64.get(float(CONSTS[0]) if has_cl else None)
                        if wf is None:
                            continue
                        ed['<C_R>'] = F(wf)
                    try:
                        with np.errstate(all='ignore'):
                            _gcl = cls_np(d_eval(tl, ed))
                            _gcr = cls_np(d_eval(tr, ed))
                            dv = compare_deployed_realised(_gcl, _gcr)
                    except Exception:
                        continue
                    dep_seen[0] = True
                    if dv != 'eq':
                        dep_bad.append((dv, {'slot': n, 'grid': float(g)}))
                    if cv == 'eq' and compare_deployed_structural(_gcl, _gcr) != 'eq':
                        dep_div.append((dv, {'slot': n, 'grid': float(g)}))
                if res:
                    # DENOMINATOR IS THE GRID, not the resolved subset (F96). Dropping
                    # Unresolved points from the divisor made the bar a moving target:
                    # two shipped rules already run at 154, and at a denominator of 19 a
                    # SINGLE disagreeing point scores 0.0526 and kills -- precisely the
                    # "single-point fluke" the MEASURE_KILL comment promises can never
                    # reach the bar. An unresolved point is absence of evidence and
                    # counts toward neither side.
                    meas = max(meas, sum(1 for c in res if c != 'eq') / len(GRID))
        # D15 (X11's opposite-polarity hole): the DIAGONAL lane -- every wildcard
        # slot bound to ONE varying value. Independent-slot sampling makes
        # {_i = _j} a null event, but for a REWRITE RULE the diagonal is an
        # engine-reachable instance family: with _0 = _1 = log(x0) the collector
        # refuses the nan-capable t - t -> 0, the pattern (a-b)/(b-a) still
        # matches, and the engine rewrote a nan-EVERYWHERE expression to -1 while
        # this scan read the event as null-tolerated EXT. An off-'eq' fraction on
        # the diagonal is positive measure ON THAT FAMILY, judged by the same bar.
        if len(names) >= 2:
            res = []
            for g in GRID:
                e = {m: mpf(float(g)) for m in names}
                if has_cl:
                    e['<C_L>'] = mpf(CONSTS[0])
                if has_cr:
                    wv = witness.get(float(CONSTS[0]) if has_cl else None)
                    if wv is None:
                        continue
                    e['<C_R>'] = +wv
                # PRECISION-COUPLED like every other lane (F96 completed 2026-08-20).
                # This took a single bare reading at rung 1 and fed the SAME `meas` that
                # KILLs, so one dps-50 rounding on the diagonal could convict a rule that
                # the per-slot lane -- two-rung since F96 -- would have cleared. The
                # asymmetry was invisible because both lanes write into one accumulator.
                v, _ = _point_verdict(tl, tr, lambda e=e: e)
                if v is not None:
                    res.append(v)
            if res:
                meas = max(meas, sum(1 for c in res if c != 'eq') / len(GRID))
    finally:
        mp.dps = old

    _realised = (not dep_bad) if dep_seen[0] else None

    if a_kills:
        return {'verdict': 'KILL', 'clause': 'a-real-change', 'points': a_kills[:3],
                'measure': meas, 'skipped_cl': skipped_cl,
                'realised': _realised, 'tier': _tier('KILL', _realised)}
    if meas > MEASURE_KILL:
        return {'verdict': 'KILL', 'clause': 'bc-positive-measure', 'measure': meas,
                'tolerated_events': len(tolerated), 'skipped_cl': skipped_cl,
                'realised': _realised, 'tier': _tier('KILL', _realised)}
    if resolved_pts == 0 or resolved_pts / max(1, attempted_pts) < 0.25:
        # "skip, never convict" must not become "skip everything, acquit": a rule whose
        # battery is almost entirely Unresolved has no evidence either way (e.g. a
        # 1e15-argument trig rule certifying from one resolved point). The bar is a
        # FRACTION of attempted points: a ground rule with its single point resolved has
        # full coverage.
        return {'verdict': 'UNRESOLVED-COVERAGE', 'resolved_points': resolved_pts,
                'attempted_points': attempted_pts, 'skipped_cl': skipped_cl}
    for r in ({},):
        pass
    if dep_div:
        # contract certifies; deployed diverges. Gap class (literal-zero division /
        # zero-sign) never reaches here because cls_np uses quotient zeros; what
        # remains is a genuine realization split.
        return {'verdict': 'ENGINE-MISALIGN', 'points': [p for _, p in dep_div[:3]],
                'kinds': sorted({k for k, _ in dep_div}), 'measure': meas,
                'realised': _realised, 'tier': _tier('ENGINE-MISALIGN', _realised)}
    if tolerated:
        kinds = sorted({v for v, _, _ in tolerated})
        return {'verdict': 'TOLERATED', 'classes': kinds,
                'saturation_probed': saturation_probed,
            'realised': _realised, 'tier': _tier('TOLERATED', _realised),
                'events': [(v, p) for v, p, _ in tolerated[:3]], 'measure': meas,
                'skipped_cl': skipped_cl, 'resolved_points': resolved_pts}
    return {'verdict': 'CERTIFIED', 'measure': meas, 'skipped_cl': skipped_cl,
            'saturation_probed': saturation_probed,
            'realised': _realised, 'tier': _tier('CERTIFIED', _realised),
            'resolved_points': resolved_pts}


# ---------------------------------------------------------------- self-test
TOUCHSTONES = [
    # (lhs, rhs, expected verdict, why -- the mathematical reason)
    ('+ ?0 pow sin ?0 float("inf")', '?0', 'KILL',
     'powsin: real value x+1 changed at exact pi/2 (clause a, null measure irrelevant)'),
    ('pow1_2 pow2 ?0', '?0', 'KILL',
     'sqrt(x^2)->x: real values differ on x<0 (clause a + measure)'),
    ('atan tan asin _0', 'asin _0', 'TOLERATED',
     'nan-caused zero-measure extension at +-1 is NOT the powsin class'),
    ('/ ?0 ?0', '1', 'TOLERATED',
     'x/x->1: undefined->defined at 0 only (clause b, null set)'),
    ('neg inv neg !0', 'inv !0', 'TOLERATED',
     'sign-flip family: inf-artifact at 0-events only (clause c, finite-null class)'),
    ('* 0 !0', '0', 'TOLERATED',
     'absorption: exact everywhere on reals; 0*inf indeterminate at nullx events only'),
    ('inv sin np.pi', 'float("inf")', 'CERTIFIED',
     'sin(pi) is exactly 0 (symbolic point + snap), 1/0 = +inf'),
    ('pow4 inv pow1_4 ?0', 'pow ?0 <constant>', 'KILL',
     'even-root extension on the negative half-line (clause b, positive measure)'),
    ('pow1_2 pow float("-inf") ?0', 'pow float("inf") ?0', 'KILL',
     'pow(-inf, data): undefined a.e. (clause b, positive measure)'),
    ('pow1_2 pow1_2 atanh !0', 'pow1_4 atanh !0', 'CERTIFIED',
     'pow1_4(-inf) = nan = sqrt(sqrt(-inf)): both spellings agree with the contract'),
    ('/ 0 mult2 ?0', '0', 'TOLERATED',
     '0/0 at x=0 is an undefined->defined null-set extension (the x/x->1 class); the '
     'deployed zero-sign residue is a documented gap by design'),
    ('neg + (-1) _0', '- 1 _0', 'CERTIFIED',
     'algebraic identity, exact everywhere incl the cancellation point (one zero)'),
    ('pow sin <constant> float("inf")', '0', 'KILL',
     'powsin in CONSTANT space: at c = pi/2 (symbolic atom) the value is 1, not 0 -- '
     'clause (a); fitted constants reach pi/2'),
    ('pow1_3 pow3 _0', '_0', 'CERTIFIED',
     'cbrt(t^3) = t exactly on all of R incl negatives and +-inf (odd-root exactness)'),
    ('mult3 log pow1_3 _0', 'log _0', 'CERTIFIED',
     '3*log(cbrt t) = log t for t>0; both undefined for t<0; -inf vs -inf equal at 0; '
     '3*inf = inf at +inf: no disagreement event anywhere -- exactly certified'),
    ('atanh tanh !0', '!0', 'ENGINE-MISALIGN',
     'BOTH halves, and they are independent. CONTRACT half unchanged: exactly true on ALL '
     'of R, and the dps-50 rounding of tanh(100) to 1 must still go Unresolved at the '
     'boundary rather than fabricate +inf (boundary-honesty band) -- a CERTIFIED contract '
     'lane is what lets the second half be reached at all. DEPLOYED half, new with F95: '
     'f64 tanh attains exactly 1.0 from 18.990341103219276, so the shipped engine returns '
     'inf where the rule says the argument. The rule is TRUE and NOT REALISED, which is '
     'precisely what ENGINE-MISALIGN names. A regression to CERTIFIED here means the '
     'deployed lane stopped reaching the grid; a KILL means the contract lane started '
     'fabricating.'),
    # -- regression guards --
    ('+ ?0 sin exp -100', '?0', 'KILL',
     'F99. The invariant this pins is unchanged and now holds at EVERY rung rather '
     'than only at dps 50: the tolerance floor is tied to the snap threshold '
     '(10**(-dps+10)), so a snap-absorbable value is always within tolerance and the '
     'snap can never fabricate an equality the tolerance would refuse. What changed is '
     'the VERDICT, because the rule is genuinely false and the old fixed 1e-25 floor '
     'was hiding it: sin(e^-100) = 3.720076e-44, so at the battery point ?0 = 0 the LHS '
     'is 3.7e-44 against an RHS of 0 -- a 100% relative disagreement in exact arithmetic '
     'AND in f64, where 0.0 + sin(exp(-100)) == 0.0 is False. The snap absorbs it at '
     'dps 50 (threshold 1e-40) and not at dps 120 (1e-110), so rungs 2 and 3 decide, '
     'which is the escalation working as designed.'),
    ('pow1_2 * - ?0 0.2 - ?0 0.8', 'pow1_2 abs * - ?0 0.2 - ?0 0.8', 'KILL',
     'undefined on (0.2, 0.8), measure 0.6: invisible to a coarse unit-spaced grid'),
    ('+ ?0 * pow exp neg pow2 - <constant> 2 float("inf") ?0', '?0', 'KILL',
     'violates exactly at c = 2: requires the constant battery to include even integers '
     '(powsin-in-constant-space at an ordinary value)'),
    ('pow abs ?0 <constant>', 'pow abs ?0 <constant>', 'CERTIFIED',
     'identity: the polisher must not de-snap the exact-zero witness into 1e-62 and '
     'fabricate a pow(0,0)-crossing phantom'),
    ('sin * ?0 * 1e15 + 1 1e-20', 'sin * ?0 1e15', 'UNRESOLVED-COVERAGE',
     'nearly every probe is beyond the trig wedge: no evidence either way must not '
     'certify'),
    ('- <constant> <constant>', '0', 'UNSUPPORTED-SHAPE',
     'two LHS constants bind independently in the deployed matcher; the one-symbol '
     'diagonal model must refuse, fail-closed'),
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
