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
import numpy as np
from mpmath import mp, mpf, isnan as misnan, isinf as misinf

np.seterr(all='ignore')
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


SNAP_EVENTS = [0]  # incremented when a symbolic-cancellation snap / pole-guard fires;
                   # at such points the f64 deployed algebra evaluates a DIFFERENT point
                   # (a measurement artifact), so the deployed check is skipped


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
        v = eval(t, {'np': np, 'float': float})
        return ('lit', float(v))

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


def _dom(fn, lo=None, hi=None, lo_open=False, hi_open=False):
    def g(a):
        if misnan(a):
            return NAN
        if lo is not None and (a < lo or (lo_open and a == lo)):
            return NAN
        if hi is not None and (a > hi or (hi_open and a == hi)):
            return NAN
        v = fn(a)
        if isinstance(v, mp.mpc):
            return NAN
        return _z(v)
    return g


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
    'tanh': lambda a: NAN if misnan(a) else (_z(mp.sign(a)) if misinf(a) else _z(mp.tanh(a))),
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
    return OPS[op](*args)


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
        return F(tree[1])
    args = [d_eval(c, env) for c in tree[1:]]
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


def compare(cl, cr, rel=mpf('1e-25')):
    """-> 'eq' | 'REAL-CHANGE' (both fin, differ: clause a) |
          'EXT' | 'SHRINK' (nan vs defined: clause b) |
          'INF-CHANGE' (an infinity involved: clause c)"""
    if cl[0] == 'nan' and cr[0] == 'nan':
        return 'eq'
    if cl[0] == 'nan':
        return 'EXT'
    if cr[0] == 'nan':
        return 'SHRINK'
    if cl[0] == 'inf' or cr[0] == 'inf':
        return 'eq' if cl == cr else 'INF-CHANGE'
    a, b = cl[1], cr[1]
    if a == b:
        return 'eq'
    tol = rel * max(mpf(1), abs(mpf(a)), abs(mpf(b))) if not isinstance(a, float) \
        else 1e-9 * max(1.0, abs(a), abs(b))
    return 'eq' if abs(a - b) <= tol else 'REAL-CHANGE'


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


CONSTS = [2.5, -1.5, 3.0, 0.5, -0.7]      # witness-fitting battery (generic values)


def judge_cl_battery():
    """judging battery for a SOURCE-side constant (forall c_s over R). Includes the
    special rationals (pow(1,nan)=1 class) AND the SYMBOLIC transcendental atoms,
    built at the CURRENT dps: a fitted constant reaches pi/2, and pow(sin(c),inf)
    at c = pi/2 is the powsin class in constant space -- an f64 pi/2 misses the
    coincidence by 5e-33 (a false rescue can otherwise slip through; the symbolic
    constant atoms kill it correctly). A cl whose witness is unfittable (degenerate
    LHS there) is skipped with a tally; only core-CONSTS failures mean NO-WITNESS."""
    return ([(lambda c=c: mpf(c)) for c in CONSTS] +
            [(lambda: mpf(1)), (lambda: mpf(-1)), (lambda: mpf(0)),
             (lambda: mpf(2)), (lambda: mpf(-2)), (lambda: mpf(4)),
             (lambda: mpf(-4)), (lambda: mpf(5)), (lambda: mpf(-5)),
             (lambda: mp.pi / 2), (lambda: -mp.pi / 2), (lambda: mp.pi),
             (lambda: mp.e)])
GRID = np.concatenate([np.linspace(-3, 3, 121), np.linspace(-20, 20, 41),
                       np.array([-1e4, -1e3, -300., -100., -50., -30.,
                                 30., 50., 100., 300., 1e3, 1e4])]) + 0.0137421
# dense center (0.05 spacing: sub-unit undefined intervals like (0.2, 0.8) are visible),
# unit-spaced mid-range, and a magnitude tail (half-line violations opening at |x|>20,
# e.g. exp-shift constructions)
MEASURE_KILL = 0.05  # fraction of the generic grid; a genuine positive-measure set
                     # shows as many points; single-point flukes never reach this


# ---------------------------------------------------------------- witness fitting
GEN = [1.7, -1.3, 0.6, 2.3, -0.8]


def _gen_env(names, k):
    return {n: F(GEN[(k + i) % len(GEN)]) for i, n in enumerate(sorted(names))}


_XS = np.concatenate([-np.logspace(12, -3, 120), np.linspace(-3, 3, 81),
                      np.logspace(-3, 12, 120)])


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
        mp.dps = 60
        env = {}
        for i, n in enumerate(sorted(nvars)):
            env[n] = mpf(GEN[i % len(GEN)])
        if clb is not None:
            # the EXACT battery value (symbolic builder), NOT its f64 rendering: a
            # witness polished against f64-pi differs from the judge's exact mp.pi by
            # ~4e-17, producing false clause-(a) kills of the witness family
            env['<C_L>'] = clb()
        try:
            target = c_eval(tl, env)
        except Unresolved:
            return mpf(c0)
        if misnan(target) or misinf(target):
            return mpf(c0)

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
                if abs(h(mpf(0))) <= mpf('1e-45') * max(1, abs(target)):
                    return mpf(0)
            except Unresolved:
                return mpf(0)
        try:
            a, b = mpf(c0) * (1 - mpf('1e-8')) - mpf('1e-12'), mpf(c0) * (1 + mpf('1e-8')) + mpf('1e-12')
            fa, fb = h(a), h(b)
            for _ in range(120):
                if fb == fa:
                    break
                c = b - fb * (b - a) / (fb - fa)
                fc = h(c)
                a, fa, b, fb = b, fb, c, fc
                if abs(fc) <= mpf('1e-45') * max(1, abs(target)):
                    if abs(c) < mpf('1e-50'):
                        try:
                            if abs(h(mpf(0))) <= mpf('1e-45') * max(1, abs(target)):
                                return mpf(0)   # snap near-zero polish results to exact 0
                        except Unresolved:
                            pass
                    return c
        except (Unresolved, ZeroDivisionError):
            pass
        return mpf(c0)
    finally:
        mp.dps = old


# ---------------------------------------------------------------- the judge
def _point_verdict(tl, tr, env_mp):
    """two-rung confirmed contract verdict at one point -> (verdict|None, snapped)"""
    def once():
        try:
            return compare(cls_mp(c_eval(tl, env_mp())), cls_mp(c_eval(tr, env_mp())))
        except Unresolved:
            return None
    old = mp.dps
    snap0 = SNAP_EVENTS[0]
    try:
        mp.dps = 50
        v1 = once()
        snapped1 = SNAP_EVENTS[0] > snap0
        if v1 is None:
            return None, snapped1
        if v1 == 'eq' and not snapped1:
            return 'eq', False
        # non-eq, OR an 'eq' that involved a snap (the snap can FABRICATE equality:
        # sin(exp(-100)) = 3.7e-44 snaps to 0 at dps 50): confirm at rung 2; a
        # snapped-eq that changes class at dps 120 takes rung 3.
        mp.dps = 120
        v2 = once()
        if v1 == v2:
            return v1, snapped1
        if v1 == 'eq':
            mp.dps = 250
            v3 = once()
            return (v2 if v2 == v3 else None), snapped1
        return None, snapped1
    finally:
        mp.dps = old


def judge_rule(lhs, rhs, deployed_check=True):
    """-> dict: verdict CERTIFIED | TOLERATED | KILL | ENGINE-MISALIGN | NO-WITNESS,
    with clause/class, points, measures. Implements the kill bar:
      KILL   iff a REAL-CHANGE exists at any resolved point (clause a)
             or the generic-grid disagreement measure exceeds MEASURE_KILL (b/c);
      ENGINE-MISALIGN iff the contract certifies but the deployed algebra
             structurally diverges at a non-gap battery point / on the grid;
      TOLERATED else-if any null-event disagreement exists (documented class);
      CERTIFIED otherwise."""
    tl = parse(lhs, '<C_L>')
    tr = parse(rhs, '<C_R>')
    sl = slots_of(lhs, '<C_L>')
    sr = slots_of(rhs, '<C_R>')
    slots = dict(sl)
    slots.update({k: v for k, v in sr.items() if k not in slots})
    nvars = {n: s for n, s in slots.items() if not n.startswith('<C')}
    has_cl, has_cr = '<C_L>' in slots, '<C_R>' in slots
    if lhs.count('<constant>') > 1:
        # the deployed matcher binds multiple <constant> leaves INDEPENDENTLY; this
        # judge models one shared symbol (diagonal only) -- rather than silently
        # under-judging the off-diagonal, the shape is refused fail-closed
        return {'verdict': 'UNSUPPORTED-SHAPE', 'detail': 'multiple LHS <constant>'}

    # cl battery entries: (builder, f64key). builder() gives the EXACT value at the
    # current dps (symbolic atoms rebuild per rung); f64key indexes the witness maps
    # and feeds the f64 fitter / deployed check. TWO witness maps, one per algebra:
    # the contract compares against the mp-polished witness, the deployed check against
    # the raw f64 fit -- deployment fits its OWN constant, so judging the f64 algebra
    # with the mp witness manufactures false ENGINE-MISALIGNs across f64 saturation
    # cliffs (asin(tanh(exp(3))): f64 sees exactly pi/2, the contract pi/2 - 2.3e-9).
    witness = {}
    witness_f64 = {}
    if has_cl:
        cl_battery = [(b, float(b())) for b in judge_cl_battery()]
    else:
        cl_battery = [(None, None)]
    core = {float(c) for c in CONSTS}
    skipped_cl = []
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
                if lhs_defined and (key is None or key in core):
                    return {'verdict': 'NO-WITNESS', 'detail': f'cl={key} (LHS defined)'}
                skipped_cl.append(key)
                continue
            any_fit = True
            witness[key] = mp_polish(tl, tr, nvars, b, w)
            witness_f64[key] = w
            kept_cl.append((b, key))
        if has_cr and not any_fit:
            return {'verdict': 'NO-WITNESS', 'detail': 'no cl with a finite LHS/witness'}
        cl_battery = kept_cl if has_cl else [(None, None)]

    a_kills, tolerated, dep_div = [], [], []
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
        if len(combos) > 500:
            # 500 covers every <=2-slot rule exhaustively (23^2 = 529 ~ capped edge);
            # sampling below that drops single-point clause-(a) violations of 2-var
            # rules; >=3-slot rules keep the seeded sample
            rng = np.random.default_rng(11)
            combos = [combos[i] for i in rng.choice(len(combos), 500, replace=False)]
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
            if deployed_check and v in (None, 'eq') and not snapped and all_real:
                try:
                    ed = {n: F(float(b())) for n, (b, t) in combo.items()}
                    if clkey is not None:
                        ed['<C_L>'] = F(clkey)
                    if crd is not None:
                        ed['<C_R>'] = F(crd)   # the f64-fitted witness: each algebra
                                               # is judged with its OWN constant
                    dv = compare(cls_np(d_eval(tl, ed)), cls_np(d_eval(tr, ed)))
                    if dv != 'eq':
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
                    try:
                        res.append(compare(cls_mp(c_eval(tl, e)), cls_mp(c_eval(tr, e))))
                    except Unresolved:
                        pass
                if res:
                    bad = sum(1 for c in res if c != 'eq') / len(res)
                    meas = max(meas, bad)
    finally:
        mp.dps = old

    if a_kills:
        return {'verdict': 'KILL', 'clause': 'a-real-change', 'points': a_kills[:3],
                'measure': meas, 'skipped_cl': skipped_cl}
    if meas > MEASURE_KILL:
        return {'verdict': 'KILL', 'clause': 'bc-positive-measure', 'measure': meas,
                'tolerated_events': len(tolerated), 'skipped_cl': skipped_cl}
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
                'kinds': sorted({k for k, _ in dep_div}), 'measure': meas}
    if tolerated:
        kinds = sorted({v for v, _, _ in tolerated})
        return {'verdict': 'TOLERATED', 'classes': kinds,
                'events': [(v, p) for v, p, _ in tolerated[:3]], 'measure': meas,
                'skipped_cl': skipped_cl, 'resolved_points': resolved_pts}
    return {'verdict': 'CERTIFIED', 'measure': meas, 'skipped_cl': skipped_cl,
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
    ('atanh tanh !0', '!0', 'CERTIFIED',
     'exactly true on ALL of R; dps-50 rounding of tanh(100) to 1 must go Unresolved at '
     'the boundary, never fabricate +inf (boundary-honesty band)'),
    # -- regression guards --
    ('+ ?0 sin exp -100', '?0', 'CERTIFIED',
     'sin(e^-100) = 3.7e-44 differs from 0 BELOW the judge resolution floor (rel 1e-25):'
     ' any snap-absorbable value (<1e-40) is also below the floor, so the snap cannot'
     ' fabricate an equality the tolerance would not grant; f64-mined rules cannot'
     ' express sub-1e-16 discrepancies (rung-2 confirmation kept as belt-and-braces)'),
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
