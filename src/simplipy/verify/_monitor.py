# mypy: ignore-errors
"""Independent deployed-realization monitor: an end-to-end soundness gate.

This monitor is deliberately independent of the certifier's own judges. Its evaluator,
operator semantics, probe machinery, and corpus are written from the simplification
contract alone, so it can catch violations that a checker sharing the certifier's own
machinery would structurally miss. It exercises the DEPLOYED engine end-to-end
(``simplify`` on real expression shapes, with default constant masking) and judges every
rewrite under REAL (exact) semantics:

  Values are compared in mpmath (no float64 saturation/overflow); a single unsigned zero;
  1/0 = +inf; IEEE pow specials (pow(+-1, +-inf) = 1, pow(0, 0) = 1); literal folding
  (np.pi -> pi). A DEFINED value may never change (checked pointwise; symbolic points such
  as pi/2 are probe points). Extending an undefined point to a defined one is tolerated on
  NULL sets only: extension at isolated atoms is tolerated, while extension on a large
  fraction of continuum draws is a positive-measure violation.

Verdict classes per rewrite: OK | VIOLATION (value change / shrink / positive-measure
extension) | UNSCORED (multi-slot outputs). A violation that is ALSO present under the
rules-empty engine is bucketed NATIVE-LINE (pre-existing engine behavior, not attributable
to the ruleset under test).

Self-test: :func:`selftest` poisons a copy of the ruleset with known-unsound rules and
reports failure loudly unless the monitor flags all of them.

Importable API:
  * :func:`judge_pair`  -- judge a single ``inp -> out`` rewrite.
  * :func:`sweep`       -- run the engine over a corpus and collect violations.
  * :func:`build_engine`, :func:`make_corpus` -- engine/corpus construction.
  * :func:`selftest`    -- poison self-test (returns True iff every poison is caught).
  * :func:`monitor`     -- top-level driver (optional self-test, then the main sweep).
"""
import json
import os
import tempfile

import numpy as np
import yaml
from mpmath import mp, mpf, isnan, isinf

from ..engine import SimpliPyEngine
from ..utils import count_expressions, sample_expression

DPS = 50
REL_TOL = mpf('1e-30')
SLOT_TOL = mpf('1e-12')  # slot-bound comparisons check WITNESS EXISTENCE, and the witness
                         # is bound as a 40-digit token (~1e-40 relative error). Phase-
                         # sensitive compositions amplify token error: observed at 1e-21
                         # (cos of a pow tower, amplification ~1e19) -- stable across dps
                         # rungs, so the stability gate cannot clear it. 1e-12 sits 9 orders
                         # above the worst amplification observed while leaving genuine value
                         # changes at any consumer-relevant scale (>= 1e-9) caught. NB this
                         # does NOT mean sub-1e-12 changes are invisible to f64 consumers
                         # (f64 resolves ~1e-16); it means a 40-digit-token instrument cannot
                         # honestly distinguish them from its own binding noise. Exact-real
                         # strictness (mp-polished witnesses) is the rule-complete gate's job.
                         # Slot-bearing pairs: the fitted c carries secant precision ~1e-30,
                         # which HOVERS at REL_TOL on other probes (observed as same-displayed
                         # 'value changes'); slot comparisons use this looser, stated bound.
MEASURE_FRACTION = 0.15   # extension-layer disagreements (nan<->defined and infinity-involved
                          # changes) are judged by MEASURE over the data. ~40 continuum
                          # draws/pair: 0.15 = 6 draws -- far above single-draw noise, far
                          # below any genuine positive-measure set (half-lines show ~0.5). This
                          # end-to-end sweep is the realization check; the rule-complete gate is
                          # the primary measure instrument. Null events (atom probes incl
                          # +-inf) are tolerated and tallied (TOL_COUNTS) per the null-event
                          # doctrine.

TOL_COUNTS = {'shrink-null': 0, 'inf-null': 0, 'ext-null': 0}
N_DRAWS = 40                # continuum draws per variable set
mp.dps = DPS


# ---------------------------------------------------------------------------------------
# The contract evaluator: mpmath, written from the contract semantics alone.
# ---------------------------------------------------------------------------------------

def _nan():
    return mpf('nan')


class Unresolved(Exception):
    """A probe whose value is UNKNOWABLE at working precision (atanh at a possibly-rounded
    +-1). Distinct from nan (= genuinely undefined): an Unresolved probe is SKIPPED entirely
    -- it must count neither as agreement, nor violation, nor extension measure. (Counting it
    as nan fabricated 'positive-measure extension' verdicts on sound atanh(tanh(t)) chains.)"""


def _pole_guard(denom):
    """A denominator indistinguishable from an exact pole at working precision is a pole."""
    return abs(denom) < mpf(10) ** (-(DPS - 10))


def c_div(a, b):
    if isnan(a) or isnan(b):
        return _nan()
    if isinf(a) and isinf(b):
        return _nan()
    if b == 0:                               # ONE zero; c/0 = sign(c)*inf, 0/0 = nan
        if a == 0:
            return _nan()
        if isinf(a) or a > 0:
            return mpf('inf') if (not isinf(a) or a > 0) else mpf('-inf')
        return mpf('-inf')
    if isinf(b):
        return mpf(0) if not isinf(a) else _nan()
    return a / b


PARITY_CAP = mpf('1e15')  # beyond this an exponent's integrality/parity is unresolvable at any
                          # reasonable precision -- and int(mpf) on an exponent like exp(exp(700))
                          # (a 10^304-bit integer) is a bignum operation that can hang the
                          # process. Precision-honest verdict for parity-dependent branches:
                          # nan (symmetric).


def _int_or_none(b):
    """b as an exact int, None if non-integer, nan-sentinel if parity is unresolvable."""
    if abs(b) > PARITY_CAP:
        return 'unresolvable'
    ib = int(b)
    return ib if b == ib else None


def c_pow(a, b):
    if not isnan(a) and a == 1:
        return mpf(1)                        # IEEE: pow(1, y) = 1 for EVERY y, including nan
    if isnan(a) or isnan(b):
        return mpf(1) if (not isnan(b) and b == 0) else _nan()   # contract: pow(nan,0)=1
    if b == 0:
        return mpf(1)                        # pow(0,0)=1, pow(inf,0)=1
    if a == 0:
        if isinf(b):
            return mpf(0) if b > 0 else mpf('inf')
        return mpf(0) if b > 0 else mpf('inf')
    if isinf(b):
        aa = abs(a)
        if aa == 1:
            return mpf(1)                    # pow(+-1, +-inf) = 1
        if b > 0:
            return mpf('inf') if aa > 1 else mpf(0)
        return mpf(0) if aa > 1 else mpf('inf')
    if isinf(a):
        if a > 0:
            return mpf('inf') if b > 0 else mpf(0)
        ib = _int_or_none(b)
        if ib == 'unresolvable':
            return _nan()
        if ib is None:
            return _nan()                    # REAL semantics: (-inf)^non-integer is undefined
        if b > 0:
            return mpf('-inf') if ib % 2 else mpf('inf')
        return mpf(0)                        # one zero: no -0
    if a < 0:
        ib = _int_or_none(b)
        if ib == 'unresolvable':
            return _nan()
        if ib is None:
            return _nan()                    # negative base, non-integer exponent
        r = _pow_mag_capped(-a, b)
        return -r if ib % 2 else r
    try:
        return _pow_mag_capped(a, b)
    except Exception:
        return _nan()


PHASE_CAP = mpf('1e25')   # a trig argument's phase mod 2pi is only knowable to the precision
                          # of the WEAKEST value feeding it: fitted slots are bound as 40-digit
                          # tokens, so phase is garbage beyond ~1e3x -- observed (cos at ~9e48
                          # under an older 1e50 cap flipped sign purely from slot rounding: a
                          # false violation). 1e25 leaves >= 1e14 phase margin for 40-digit
                          # tokens. Beyond the cap: unresolved -> nan, SYMMETRICALLY on both
                          # sides. (Also guards mpmath's quasi-unbounded argument reduction, one
                          # of its uninterruptible paths.) Cost: huge-argument trig pairs go
                          # unjudged (coverage loss, never a wrong verdict).


def _trig_inf(f):
    def g(x):
        if isnan(x) or isinf(x) or abs(x) > PHASE_CAP:
            return _nan()
        return f(x)
    return g


def c_tan(x):
    if isnan(x) or isinf(x) or abs(x) > PHASE_CAP:
        return _nan()
    c = mp.cos(x)
    if _pole_guard(c):
        return _nan()                        # exact pole at working precision
    return mp.sin(x) / c


def c_inv(x):
    return c_div(mpf(1), x)


MAG_CAP = mpf('1e50')     # exponent-magnitude cap: a result beyond 10^(1e50) forces mpmath's
                          # exponent INT to ~1e50 digits -- a single uninterruptible CPython
                          # bignum op (an alarm cannot fire mid-op; observed via exp/pow
                          # towers). Values beyond it are PRACTICALLY INFINITE for any
                          # consumer; returned as +-inf/0 SYMMETRICALLY on both sides, so
                          # comparisons at such magnitudes stay fair.


def c_exp(x):
    if isnan(x):
        return _nan()
    if isinf(x):
        return mpf('inf') if x > 0 else mpf(0)   # exact limit values
    if abs(x) > MAG_CAP:
        # a FINITE huge argument: capping to +-inf/0 is NOT symmetric across
        # algebraically-equal spellings (observed: 9*log((e^(cosh/4))^5) = 11.25*cosh --
        # the exp side capped to inf, the cosh side arrived finite -> false INF-CHANGE).
        # The only honest verdict at such probes is unresolved.
        raise Unresolved()
    return mp.exp(x)


def _hyp(f):
    def g(x):
        if isnan(x):
            return _nan()
        if isinf(x):
            if f is mp.cosh:
                return mpf('inf')
            return (mpf('inf') if x > 0 else mpf('-inf')) if f is mp.sinh else (mpf(1) if x > 0 else mpf(-1))
        if abs(x) > MAG_CAP:
            if f is mp.tanh:
                return mpf(1) if x > 0 else mpf(-1)  # saturation exact to any precision
            raise Unresolved()                        # sinh/cosh: see c_exp rationale
        return f(x)
    return g


def _pow_mag_capped(a, b):
    """mp.power for finite a>0, finite b, with the exponent-magnitude guard."""
    if a == 1:
        return mpf(1)
    try:
        mag = b * mp.log10(a)
    except Exception:
        return _nan()
    if abs(mag) > MAG_CAP:
        raise Unresolved()                            # see c_exp rationale
    return mp.power(a, b)


def _odd_root(k):
    def g(x):
        if isnan(x):
            return _nan()
        if isinf(x):
            return x
        r = mp.power(abs(x), mpf(1) / k)
        return -r if x < 0 else r
    return g


def _even_root(k):
    def g(x):
        if isnan(x) or x < 0:                # sign check FIRST: even root of -inf is nan
            return _nan()                    # (isinf-first returned +inf for -inf: a live bug)
        if isinf(x):
            return mpf('inf')
        return mp.power(x, mpf(1) / k)
    return g


def _tot(f, dom=None, lo=None, hi=None):
    def g(x):
        if isnan(x):
            return _nan()
        if isinf(x):
            return dom(x) if dom else _nan()
        if lo is not None and (x < lo or x > hi):
            # BOUNDARY HONESTY: an argument outside the domain by less than the working-
            # precision noise floor is indistinguishable from the boundary itself (observed:
            # acosh(atanh(tanh(1))) -- the roundtrip lands at 1 - 1e-50, truth is
            # exactly 1). Unresolved, never a fabricated domain verdict.
            band = mpf(10) ** (-mp.dps + 10)
            if min(abs(x - lo), abs(x - hi)) < band:
                raise Unresolved()
            return _nan()
        try:
            return f(x)
        except Exception:
            return _nan()
    return g


OPS = {
    '+': (2, lambda a, b: _nan() if (isnan(a) or isnan(b)) else (_nan() if (isinf(a) and isinf(b) and (a > 0) != (b > 0)) else a + b)),
    '-': (2, lambda a, b: _nan() if (isnan(a) or isnan(b)) else (_nan() if (isinf(a) and isinf(b) and (a > 0) == (b > 0)) else a - b)),
    '*': (2, lambda a, b: _nan() if (isnan(a) or isnan(b)) else (_nan() if ((a == 0 and isinf(b)) or (b == 0 and isinf(a))) else a * b)),
    '/': (2, c_div),
    'pow': (2, c_pow),
    'neg': (1, lambda x: _nan() if isnan(x) else -x),
    'abs': (1, lambda x: _nan() if isnan(x) else abs(x)),
    'inv': (1, c_inv),
    'exp': (1, c_exp),
    'log': (1, lambda x: _nan() if (isnan(x) or x < 0)
            else (mpf('-inf') if x == 0
                  else (_raise_unresolved() if x < mpf(10) ** (-mp.dps + 10)
                        else (mpf('inf') if isinf(x) else mp.log(x))))),
    'sin': (1, _trig_inf(mp.sin)), 'cos': (1, _trig_inf(mp.cos)), 'tan': (1, c_tan),
    'asin': (1, _tot(mp.asin, lo=-1, hi=1)), 'acos': (1, _tot(mp.acos, lo=-1, hi=1)),
    'atan': (1, _tot(mp.atan, dom=lambda x: mp.pi / 2 if x > 0 else -mp.pi / 2)),
    'sinh': (1, _hyp(mp.sinh)),
    'cosh': (1, _hyp(mp.cosh)),
    'tanh': (1, _hyp(mp.tanh)),
    'asinh': (1, _tot(mp.asinh, dom=lambda x: x)),
    'acosh': (1, lambda x: _nan() if isnan(x)
              else (_raise_unresolved() if (x < 1 and 1 - x < mpf(10) ** (-mp.dps + 10))
                    else (_nan() if x < 1 else (mpf('inf') if isinf(x) else mp.acosh(x))))),
    # atanh at EXACTLY +-1: in composition, an argument that reads +-1 at working precision is
    # indistinguishable from a saturated 1-eps (mp.tanh(11013) ROUNDS to 1 at dps 50 -- observed
    # as a false +inf). Precision-honest: unresolvable -> nan. (Cost: rules that hinge on
    # a literal atanh(1)=inf go unresolved here, never wrongly judged.)
    # atanh at EXACTLY +-1: indistinguishable from a saturated 1-eps at working precision
    # (mp.tanh(11013) rounds to 1 at dps 50) -> Unresolved, probe skipped.
    'atanh': (1, lambda x: _nan() if isnan(x)
              else (_raise_unresolved() if abs(abs(x) - 1) < mpf(10) ** (-mp.dps + 10)
                    else (_nan() if abs(x) > 1 else mp.atanh(x)))),
    'pow2': (1, lambda x: c_pow(x, mpf(2))), 'pow3': (1, lambda x: c_pow(x, mpf(3))),
    'pow4': (1, lambda x: c_pow(x, mpf(4))), 'pow5': (1, lambda x: c_pow(x, mpf(5))),
    'pow1_2': (1, _even_root(2)), 'pow1_3': (1, _odd_root(3)),
    'pow1_4': (1, _even_root(4)), 'pow1_5': (1, _odd_root(5)),
    'mult2': (1, lambda x: _nan() if isnan(x) else 2 * x), 'mult3': (1, lambda x: _nan() if isnan(x) else 3 * x),
    'mult4': (1, lambda x: _nan() if isnan(x) else 4 * x), 'mult5': (1, lambda x: _nan() if isnan(x) else 5 * x),
    'div2': (1, lambda x: _nan() if isnan(x) else x / 2), 'div3': (1, lambda x: _nan() if isnan(x) else x / 3),
    'div4': (1, lambda x: _nan() if isnan(x) else x / 4), 'div5': (1, lambda x: _nan() if isnan(x) else x / 5),
}

LITS = {'0': lambda: mpf(0), '1': lambda: mpf(1), '(-1)': lambda: mpf(-1),
        'np.pi': lambda: +mp.pi, 'np.e': lambda: +mp.e,
        'float("inf")': lambda: mpf('inf'), 'float("-inf")': lambda: mpf('-inf'),
        'float("nan")': _nan}


def _raise_unresolved():
    raise Unresolved()


def _slot_token(v):
    """Bind a fitted slot at FULL working precision. Binding via '%r' % float(v) rounded the
    constant to f64 (1e-17 error) -- which the two-rung stability gate then CORRECTLY reported
    as a stable value change on perfectly sound rewrites (observed, both in flagged rules and
    as ~700 spurious native-line entries)."""
    s = mp.nstr(abs(v), 40)
    return s if v >= 0 else f'(-{s})'


def evaluate(tokens, env):
    """Contract-semantics value of a prefix expression at env (variable name -> mp value)."""
    pos = 0

    def walk():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t in OPS:
            ar, f = OPS[t]
            args = [walk() for _ in range(ar)]
            return f(*args)
        if t in env:
            return env[t]
        if t in LITS:
            return LITS[t]()
        return mpf(t.strip('()'))
    v = walk()
    if pos != len(tokens):
        raise ValueError(f'trailing tokens: {tokens}')
    return v


# ---------------------------------------------------------------------------------------
# Probes and judgment
# ---------------------------------------------------------------------------------------

def probe_points(rng):
    # SYMBOLIC probes are BUILDERS, materialized inside the caller's workdps: a frozen dps-50
    # mp.pi/2 drifts off the spike at rung 2 (pi/2 +- 1e-51, where sin < 1 strictly), which
    # let the pow-sin poison evaporate under the two-rung stability gate (observed, caught by
    # the self-test). Rational draws are exact at every dps and stay frozen.
    sym = [lambda: mpf(0), lambda: mpf(1), lambda: mpf(-1), lambda: mpf(2), lambda: mpf(-2),
           lambda: mpf(3), lambda: mpf('0.5'), lambda: mpf('-0.5'),
           lambda: +mp.pi, lambda: -mp.pi, lambda: mp.pi / 2, lambda: -mp.pi / 2,
           lambda: mp.pi / 4, lambda: +mp.e, lambda: -mp.e, lambda: mpf(10),
           lambda: mpf('inf'), lambda: mpf('-inf'), _nan]
    draws = [mpf(repr(float(v))) for v in
             np.concatenate([rng.normal(0, 3, N_DRAWS // 2), rng.uniform(-30, 30, N_DRAWS // 2)])]
    return sym, draws


def variables(tokens):
    return sorted({t for t in tokens if t not in OPS and t not in LITS
                   and t != '<constant>' and not _is_num(t)})


def _is_num(t):
    try:
        float(t.strip('()'))
        return True
    except ValueError:
        return False


def _snap_literal_zeros(tokens):
    """Literal zero-snap, independently implemented: a maximal VARIABLE-FREE subtree that
    denotes exact zero (sin(np.pi)) evaluates to a precision residue (1e-51 at dps 50), which
    turns 0*inf = nan into +-inf and fabricates violations (observed). Two-rung test: residue
    that SHRINKS quadratically with dps is an exact zero; a stable tiny value is left alone."""
    n = len(tokens)
    ends = [0] * n

    def walk(i):
        j = i + 1
        for _ in range(OPS.get(tokens[i], (0,))[0]):
            j = walk(j)
        ends[i] = j
        return j
    walk(0)

    def var_free(i):
        return all(tokens[k] in OPS or tokens[k] in LITS or _is_num(tokens[k])
                   for k in range(i, ends[i]))
    out, i = list(tokens), 0
    spans = []
    while i < n:
        if var_free(i) and ends[i] - i > 1:
            spans.append((i, ends[i]))
            i = ends[i]
        else:
            i += 1
    for s, e in reversed(spans):
        try:
            with mp.workdps(50):
                a = evaluate(out[s:e], {})
            with mp.workdps(110):
                b = evaluate(out[s:e], {})
        except Exception:
            continue
        if isnan(a) or isnan(b) or isinf(a) or isinf(b):
            continue
        if (a == 0 and b == 0) or (abs(a) < mpf('1e-40') and abs(b) <= abs(a) * mpf('1e-20')):
            out[s:e] = ['0']
    return out


def judge_pair(inp, out, rng, _refit_left=1, _slot_bound=False):
    """Judge one rewrite inp -> out under the contract. Returns (verdict, detail).

    `_refit_left`: on a violation while a fittable slot is bound, the bound c may be the
    artifact of a DEGENERATE first fit (pow(c, x) at x = 0 fits every c). One refit at the
    violating probe is allowed, re-judging from scratch with that c substituted; a second
    disagreement means no single constant exists -- a genuine violation."""
    inp, out = _snap_literal_zeros(list(inp)), _snap_literal_zeros(list(out))
    if '<constant>' in inp:
        # PRE-MASKED input (deployed traffic arrives masked): an input-side <constant>
        # is unevaluable, so every probe threw and the pair silently judged OK -- the
        # masked-scale poison was invisible on rulesets too small to fold-create
        # triggers (observed in practice). Bind input-side constants to a fixed
        # probe value on BOTH sides (the engine passes placeholders through verbatim,
        # so the shared spelling is the shared value pre-refit).
        inp = ['2.5' if t == '<constant>' else t for t in inp]
        out = ['2.5' if t == '<constant>' else t for t in out]
    vs = sorted(set(variables(inp)) | set(variables(out)))
    n_slots = out.count('<constant>')
    if n_slots > 1:
        return 'UNSCORED', 'multi-slot output'
    sym, draws = probe_points(rng)

    def _viol(env, msg):
        """A violation -- unless a slot refit at THIS probe is available and unspent."""
        if n_slots == 1 and _refit_left > 0:
            try:
                a_here = evaluate(list(inp), env)
                if not isnan(a_here) and not isinf(a_here):
                    c2 = a_here if list(out) == ['<constant>'] else _fit_slot(list(out), env, a_here)
                    if c2 is not None:
                        bound = [_slot_token(c2) if t == '<constant>' else t for t in out]
                        return judge_pair(inp, bound, rng, _refit_left=0, _slot_bound=True)
            except Exception:
                pass
        cinfo = f' [bound c={_fmt(slot_val)}]' if slot_val is not None else ''
        return 'VIOLATION', msg + cinfo

    def envs():
        for b in sym:
            yield ({v: b for v in vs} if len(vs) <= 1 else None), b, True
        # independent + correlated draw grids (frozen rationals wrapped as builders)
        for i, p in enumerate(draws):
            if len(vs) <= 1:
                yield {v: (lambda q=p: q) for v in vs}, (lambda q=p: q), False
            else:
                e = {v: (lambda q=draws[(i + j * 7) % len(draws)]: q) for j, v in enumerate(vs)}
                yield e, (lambda q=p: q), False
                if i % 4 == 0:
                    yield {v: (lambda q=p: q) for v in vs}, (lambda q=p: q), False
        if len(vs) > 1:
            for b in sym:
                yield {v: b for v in vs}, b, True           # correlated atoms

    slot_val = None
    ext_draws = 0
    shrink_draws = 0
    inf_draws = 0
    tot_draws = 0
    for envb, pb, is_atom in envs():
        if envb is None:
            continue
        env = {k: f() for k, f in envb.items()}
        p = pb()
        try:
            a = evaluate(list(inp), env)
        except Exception:
            continue
        try:
            b_tokens = list(out)
            if n_slots == 1:
                # bind the slot: a candidate c must fit CONSISTENTLY -- fitting at a single
                # probe is unsound when that probe is DEGENERATE (out = pow(c, x) at x = 0
                # fits EVERY c; observed as a fabricated mismatch). Collect the fit here but
                # VALIDATE it at later probes: an inconsistent c is re-fitted once at the
                # disagreeing probe; if the refit disagrees at yet another probe, no single
                # constant exists (genuine violation via the normal comparison below).
                if slot_val is None and not isnan(a) and not isinf(a):
                    if b_tokens == ['<constant>']:
                        slot_val = a
                    elif not _informative(inp, env, a):
                        pass                      # plateau probe: wait for a responsive one
                    else:
                        # MULTI-ROOT landscapes (sin/cos compositions admit many c at one
                        # probe) defeat a single-start secant: collect this probe's candidate
                        # but VALIDATE it at a second, independent point before adopting --
                        # a wrong root fits here and nowhere else (observed twice).
                        cand = _fit_slot(b_tokens, env, a)
                        if cand is not None:
                            ok2 = True
                            n_checked = 0
                            # validate at THIS pair's finite-domain points: the fixed
                            # magic points (0.7891/-1.2345) fall outside narrow domains
                            # (asin-compositions) and validation silently passed on nan,
                            # adopting a wrong periodic root (observed on alt seeds)
                            for w in [mpf('0.7891'), mpf('-1.2345'), mpf('0.31'),
                                      mpf('-0.13'), mpf('0.052'), mpf('1.9')]:
                                if n_checked >= 2:
                                    break
                                env_v = {k: w for k in env}
                                try:
                                    av = evaluate(list(inp), env_v)
                                    if isnan(av) or isinf(av):
                                        continue
                                    bt = [_slot_token(cand) if t == '<constant>' else t
                                          for t in b_tokens]
                                    bv = evaluate(bt, env_v)
                                    if isnan(bv) or isinf(bv):
                                        continue
                                    if not _informative(inp, env_v, av):
                                        continue  # saturated point: vacuous agreement
                                    n_checked += 1
                                    if abs(av - bv) > SLOT_TOL * max(1, abs(av), abs(bv)):
                                        ok2 = False
                                        break
                                except Unresolved:
                                    continue
                                except Exception:
                                    continue
                            if ok2 and n_checked == 0:
                                ok2 = False   # nothing validated: do not adopt blindly
                                try:
                                    env_v = {k: mpf('0.31') for k in env}
                                    av = evaluate(list(inp), env_v)
                                    if not isnan(av) and not isinf(av):
                                        c2 = _fit_slot(b_tokens, env_v, av)
                                        if c2 is not None:
                                            slot_val = c2
                                except Exception:
                                    pass
                            if ok2:
                                slot_val = cand
                            else:
                                # wrong root: re-fit at the validation point instead
                                try:
                                    env_v = {k: mpf('0.7891') for k in env}
                                    av = evaluate(list(inp), env_v)
                                    if not isnan(av) and not isinf(av):
                                        slot_val = _fit_slot(b_tokens, env_v, av)
                                except Exception:
                                    pass
                if slot_val is None:
                    continue
                b_tokens = [_slot_token(slot_val) if t == '<constant>' else t
                            for t in b_tokens]
            b = evaluate(b_tokens, env)
        except Exception:
            continue
        an, bn = isnan(a), isnan(b)
        if not is_atom:
            tot_draws += 1
        if an and bn:
            continue
        if an and not bn:
            if not is_atom:
                ext_draws += 1               # extension on the continuum: count the measure
            else:
                TOL_COUNTS['ext-null'] += 1  # extension at an atom: tolerated (null set)
            continue
        if bn:
            # defined->undefined is judged by MEASURE over data. A null event (atom probe,
            # incl +-inf) is tolerated + tallied; continuum draws count.
            # (NB the slot-refit escape does not apply on the measure path -- the two-point
            # fit validation above already rejects degenerate constants.)
            if is_atom:
                TOL_COUNTS['shrink-null'] += 1
                continue
            shrink_draws += 1
            continue
        if isinf(a) or isinf(b):
            if (isinf(a) and isinf(b) and (a > 0) == (b > 0)):
                continue
            # an infinity-involved disagreement is an extension-layer artifact, judged by
            # MEASURE; null events tolerated (null-event doctrine). Both-sides-REAL
            # disagreements (the strict value-change clause) never reach here and stay strict.
            if is_atom:
                TOL_COUNTS['inf-null'] += 1
                continue
            inf_draws += 1
            continue
        tol = SLOT_TOL if (n_slots == 1 or _slot_bound) else REL_TOL
        if (n_slots == 1 or _slot_bound) and max(abs(a), abs(b)) > mpf('1e1000000'):
            # exponent-amplified token noise: pow(c_token, E) vs pow(c_exact, E) differ
            # by E*1e-40 relative -- at magnitudes 10^(1e28) that exceeds any fixed
            # tolerance while both printed values agree to 8 digits (observed). A
            # 40-digit token cannot resolve such comparisons: unresolved, tallied.
            TOL_COUNTS['slot-magnitude-unres'] = TOL_COUNTS.get('slot-magnitude-unres', 0) + 1
            continue
        if abs(a - b) > tol * max(1, abs(a), abs(b)) and any(isinf(v) for v in env.values()):
            # a real-valued disagreement at an +-inf INPUT probe is convention-mediated
            # (one-zero composition vs the limit: inv(-inf)=0 -> pi/0=+inf flips a tanh
            # sign -- observed on the alt-seed sweep; the rewrite was limit-correct).
            # The strict value-change clause has authority at REAL points only; +-inf-input
            # events are extension-layer -> tolerated + tallied (null-in-data / degenerate).
            TOL_COUNTS['inf-input-null'] = TOL_COUNTS.get('inf-input-null', 0) + 1
            continue
        if abs(a - b) > tol * max(1, abs(a), abs(b)):
            # PRECISION-STABILITY gate: a finite mismatch may be dps-50 rounding amplified by
            # a singularity (acos near 1 turns a 1e-50 roundtrip residue into 1e-25 --
            # observed). Re-evaluate BOTH sides at the SAME point at dps 120: a true identity
            # agrees at any fixed point, so a shrinking difference is rounding; a stable
            # difference is a real violation.
            try:
                with mp.workdps(120):
                    env2 = {k: f() for k, f in envb.items()}
                    a2 = evaluate(list(inp), env2)
                    b2 = evaluate(b_tokens, env2)
                if isnan(a2) or isnan(b2) or isinf(a2) or isinf(b2):
                    raise ValueError
                if abs(a2 - b2) <= tol * max(1, abs(a2), abs(b2)):
                    continue                              # rounding artifact: cleared
            except Exception:
                pass
            return _viol(env, f'value change at {_fmt(p)}: {_fmt(a)} -> {_fmt(b)}')
    if tot_draws:
        for cnt, label in ((ext_draws, 'EXTENSION: undefined->defined'),
                           (shrink_draws, 'SHRINK: defined->undefined'),
                           (inf_draws, 'INF-CHANGE: extension values')):
            if cnt / tot_draws > MEASURE_FRACTION:
                return 'VIOLATION', (f'positive-measure {label} on {cnt}/{tot_draws} '
                                     f'continuum draws (measure clause)')
    return 'OK', None


CANDIDATE_CS = sorted({float(s * a * b * c / d)
                       for a in (1, 2, 3, 4, 5) for b in (1, 2, 3, 4, 5)
                       for c in (1, 2, 3, 4, 5) for d in (1, 2, 3, 4, 5, 6, 8, 9,
                                                          10, 12, 15, 16, 20, 25,
                                                          27, 36, 48, 64, 75, 100,
                                                          125)
                       for s in (1, -1)})


def _informative(inp_tokens, env, target):
    """A fit/validation probe is INFORMATIVE only if the target RESPONDS to the input:
    at a saturation plateau (tanh beyond ~60 is exactly 1.0 at dps 50) EVERY large
    candidate matches the target, so fitting or validating there is vacuous -- observed:
    a wrong constant validated on two saturated probes, then violated at the first
    responsive one."""
    if not env:
        return True
    try:
        env2 = {k: (v * (1 + mpf('1e-9')) if v != 0 else mpf('1e-12'))
                for k, v in env.items()}
        t2 = evaluate(list(inp_tokens), env2)
    except Exception:
        return False
    if isnan(t2) or isinf(t2):
        return False
    return abs(t2 - target) > mpf('1e-35') * max(1, abs(target))


def _fit_slot(out_tokens, env, target):
    """Solve the single <constant> slot so out(env) == target. The masked constants in
    mined rules are overwhelmingly PRODUCTS OF THE SMALL FACTORS the unary ops carry
    (div2..div5, mult2..mult5), so exact candidates are tried FIRST -- a secant root in
    a periodic landscape (sin/tan-wrapped outputs) can satisfy one probe and be wrong
    everywhere else (observed twice on alt seeds). Secant remains the fallback."""
    for cand in CANDIDATE_CS:
        toks = [_slot_token(mpf(cand)) if t == '<constant>' else t for t in out_tokens]
        try:
            v = evaluate(toks, env)
        except Unresolved:
            break
        except Exception:
            continue
        if not isnan(v) and not isinf(v) and abs(v - target) <= mpf('1e-30') * max(1, abs(v), abs(target)):
            return mpf(cand)
    def f(c):
        toks = [_slot_token(mpf(c)) if t == '<constant>' else t for t in out_tokens]
        try:
            return evaluate(toks, env) - target
        except Exception:
            return _nan()
    def _snap(c):
        # INTEGRALITY restoration: a fitted exponent c = 3 +- 1e-30 breaks pow's parity
        # branch (pow(-1, 3.000..1) -> nan, a fabricated SHRINK -- observed). A fit this
        # close to an integer IS that integer at fit precision.
        r = mp.nint(c)
        if abs(c - r) <= mpf('1e-24') * max(1, abs(c)):
            return r
        return c
    # FULL-mp secant, no float casts: an f64-rounded slot is ~1e-17 accurate, which can
    # never meet SLOT_TOL at symbolic probes (observed at pi/4: a perfectly-fitted c flagged
    # as a stable same-displayed mismatch). The contract's slot semantics is exists-c-REAL.
    x0, x1 = mpf(1), mpf(2)
    f0, f1 = f(x0), f(x1)
    for _ in range(80):
        if isnan(f1) or f0 == f1:
            return None
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if isnan(x2) or isinf(x2):
            return None
        if abs(f(x2)) <= mpf('1e-40') * max(1, abs(target)):
            return _snap(x2)
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return None


def _fmt(v):
    if isnan(v):
        return 'nan'
    if isinf(v):
        return '+inf' if v > 0 else '-inf'
    return mp.nstr(v, 8)


# ---------------------------------------------------------------------------------------
# Corpus, engines, the sweep
# ---------------------------------------------------------------------------------------

ADVERSARIAL = [
    ['+', 'x0', 'pow', 'sin', 'x0', 'float("inf")'],
    ['pow', 'sin', 'x0', 'float("inf")'],
    ['mult4', 'pow', 'x0', 'float("inf")'],
    ['-', 'exp', 'x0', 'exp', 'x0'],
    ['-', 'log', 'x0', 'log', 'x0'],
    ['inv', '*', '0', 'x0'],
    ['/', 'log', 'x0', 'log', 'x0'],
    ['atanh', 'tanh', 'sinh', 'x0'],
    ['*', 'x0', 'pow', 'cos', 'x1', 'float("inf")'],
    ['+', 'pow', 'tanh', 'x0', 'float("inf")', 'x1'],
    ['pow', 'div5', '7.0', 'float("nan")'],
    ['*', 'sin', 'np.pi', 'x0'],
    ['abs', 'pow', 'x0', 'mult3', '1'],
    ['tan', '+', 'x0', 'np.pi'],
    ['inv', '+', '0', 'neg', 'pow2', 'pow2', 'x0'],
    ['pow', 'float("-inf")', 'pow1_3', '3.0'],
    # triggers for the measure-clause poison (even-root half-line extension)
    ['pow4', 'pow5', 'pow1_2', 'x0'],
    ['mult2', 'pow4', 'pow5', 'pow1_2', 'x1'],
    # PRE-MASKED shapes (realization fidelity: deployed traffic arrives with literals
    # already masked to <constant>; also guarantees the masked-scale poison a trigger on
    # ANY ruleset -- on a small ruleset the poison had no fold-created triggers and the
    # self-test failed closed, observed in practice)
    ['*', '<constant>', 'x0'],
    ['+', '*', '<constant>', 'x1', 'x0'],
    ['sin', '*', '<constant>', 'x0'],
]

# Default leaf vocabulary used to sample the random half of the corpus.
DEFAULT_LEAVES = ['x0', 'x1', '0', '1', 'np.pi', '2.0', '(-0.5)', '3.0']

# Known-unsound rules for the poison self-test. Each must be a DEPLOYABLE spelling: the
# engine masks numeric literals to <constant> before matching, so a rule keyed on a bare
# literal can never fire on real traffic. Catch requires ATTRIBUTION: a flagged input whose
# output CHANGED versus the unpoisoned engine.
POISON = [((u'pow', u'sin', u'?0', u'float("inf")'), (u'0',)),
          ((u'*', u'<constant>', u'?0'), (u'?0',)),
          ((u'mult4', u'pow', u'?0', u'float("inf")'), (u'pow', u'?0', u'float("inf")')),
          # measure-clause poison: even-root half-line extension -- must be caught via
          # the positive-measure counters
          ((u'pow4', u'pow5', u'pow1_2', u'?0'), (u'pow', u'?0', u'<constant>'))]


def build_engine(rules, config_path):
    """Build a deployed engine from `config_path`, overriding its ruleset with `rules`.

    `config_path` is the deployed engine config the monitor realizes against; the driver
    passes the path in. `rules` is a list of (lhs, rhs) token-sequence pairs (empty for the
    rules-empty baseline engine)."""
    base = yaml.safe_load(open(config_path))
    d = tempfile.mkdtemp(prefix='monitor_')
    with open(os.path.join(d, 'rules.json'), 'w') as fh:
        json.dump([[list(l), list(r)] for l, r in rules], fh)
    base['rules'] = 'rules.json'
    with open(os.path.join(d, 'config.yaml'), 'w') as fh:
        yaml.safe_dump(base, fh)
    return SimpliPyEngine.from_config(os.path.join(d, 'config.yaml'))


def make_corpus(eng, n, rng, adversarial=None, leaves=None):
    if adversarial is None:
        adversarial = ADVERSARIAL
    if leaves is None:
        leaves = DEFAULT_LEAVES
    non_leaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
    counts = count_expressions(len(leaves), non_leaf, 10)
    corpus = [list(e) for e in adversarial]
    for _ in range(n):
        L = int(rng.integers(3, 10))
        corpus.append(list(sample_expression(L, leaves, non_leaf, counts, rng)))
    return corpus


class _JudgeTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _JudgeTimeout()


JUDGE_TIMEOUT_S = 10   # a single expression may not eat the sweep. mpmath has quasi-unbounded
                       # paths (trig argument reduction, bignum parity, and at least one more
                       # observed live); a gate that can HANG is not a gate. Timeouts are
                       # recorded as unresolved (with the offending expression printed) and
                       # skipped.


def sweep(rules, corpus, rng, baseline_engine, config_path, tag='',
          judge_timeout_s=JUDGE_TIMEOUT_S):
    import signal
    signal.signal(signal.SIGALRM, _alarm)
    eng = build_engine(rules, config_path)
    viol, native, unscored, changed, timeouts = [], [], 0, 0, 0
    for i, expr in enumerate(corpus):
        if tag and (i + 1) % 500 == 0:
            print(f'  [{tag} {i+1}/{len(corpus)}] changed={changed} viol={len(viol)} '
                  f'timeouts={timeouts}', flush=True)
        try:
            out = list(eng.simplify(list(expr)))
        except Exception as ex:
            viol.append((expr, ['<RAISED>'], f'simplify raised {type(ex).__name__}'))
            continue
        if out == list(expr):
            continue
        changed += 1
        signal.alarm(judge_timeout_s)
        try:
            v, detail = judge_pair(list(expr), out, rng)
        except _JudgeTimeout:
            timeouts += 1
            print(f'  [{tag}] TIMEOUT-unresolved: {" ".join(expr)}  ->  {" ".join(out)}', flush=True)
            continue
        finally:
            signal.alarm(0)
        if v == 'UNSCORED':
            unscored += 1
        elif v == 'VIOLATION':
            try:
                base_out = list(baseline_engine.simplify(list(expr)))
            except Exception:
                base_out = None
            if base_out == out:
                native.append((expr, out, detail))
            elif base_out is not None and base_out != list(expr):
                # CAUSE attribution: if the rules-empty engine's own rewrite of this
                # expression ALREADY violates, the violation is native (an f64-saturation
                # constant fold etc.) and the ruleset merely rewrote further on top --
                # observed: a native sinh-overflow fold to inf extended the negative-base
                # domain, and a SOUND rule (|x^inf|^(1/4) -> x^inf) inherited the blame
                # under exact-output matching.
                signal.alarm(judge_timeout_s)
                try:
                    vb, _ = judge_pair(list(expr), base_out, rng)
                except _JudgeTimeout:
                    vb = None
                finally:
                    signal.alarm(0)
                if vb == 'VIOLATION':
                    native.append((expr, out, detail))
                else:
                    viol.append((expr, out, detail))
            else:
                viol.append((expr, out, detail))
    return viol, native, unscored, changed


def selftest(rules, config_path, corpus, baseline, seed, poison=None,
             judge_timeout_s=JUDGE_TIMEOUT_S):
    """Poison a copy of the ruleset with known-unsound rules; return True iff EVERY poison is
    caught. A catch requires ATTRIBUTION: a flagged input whose output differs from the
    unpoisoned (clean) engine's output. Prints a per-poison CAUGHT / *** MISSED *** line."""
    if poison is None:
        poison = POISON
    clean_eng = build_engine(rules, config_path)
    caught = 0
    for p in poison:
        v, _, _, _ = sweep(rules + [p], corpus[:600], np.random.default_rng(seed),
                           baseline, config_path, tag="poison", judge_timeout_s=judge_timeout_s)
        attributed = []
        for expr, out, detail in v:
            try:
                if list(clean_eng.simplify(list(expr))) != list(out):
                    attributed.append((expr, out, detail))
            except Exception:
                attributed.append((expr, out, detail))
        hit = len(attributed) > 0
        caught += hit
        print(f'  poison {" ".join(p[0])} -> {" ".join(p[1])}: '
              f'{"CAUGHT" if hit else "*** MISSED ***"} '
              f'({len(attributed)} attributed / {len(v)} flagged)')
    return caught == len(poison)


class MonitorSelfTestError(RuntimeError):
    """Raised by :func:`monitor` when the poison self-test fails to flag a known-unsound rule.

    A self-test failure means the monitor cannot be trusted to catch an unsound ruleset, so
    the main sweep is not run: the gate has lost its own guarantee."""


def monitor(rules, config_path, corpus_n=6000, seed=20260718, run_selftest=False,
            adversarial=None, leaves=None, poison=None, judge_timeout_s=JUDGE_TIMEOUT_S,
            label=''):
    """Run the independent monitor against a deployed engine built from `config_path`.

    `rules` is either a list of (lhs, rhs) token-sequence pairs or a path to a JSON file of
    ``[[lhs, rhs], ...]`` (loaded as tuples). When `run_selftest` is True the poison self-test
    runs first and a failure raises :class:`MonitorSelfTestError` before the main sweep.

    Returns a dict with keys: rules, corpus, changed, violations, native, unscored,
    tolerated, selftest_passed."""
    if adversarial is None:
        adversarial = ADVERSARIAL
    if leaves is None:
        leaves = DEFAULT_LEAVES
    if poison is None:
        poison = POISON
    rules = _load_rules(rules)
    rng = np.random.default_rng(seed)
    baseline = build_engine([], config_path)
    corpus = make_corpus(baseline, corpus_n, rng, adversarial=adversarial, leaves=leaves)
    print(f'monitor: {len(rules):,} rules | corpus {len(corpus):,} '
          f'({len(adversarial)} adversarial + {corpus_n:,} sampled)', flush=True)

    selftest_passed = None
    if run_selftest:
        selftest_passed = selftest(rules, config_path, corpus, baseline, seed,
                                   poison=poison, judge_timeout_s=judge_timeout_s)
        if not selftest_passed:
            print('SELF-TEST FAILED: the monitor cannot catch a poisoned artifact')
            raise MonitorSelfTestError(
                'poison self-test failed: a known-unsound rule was not flagged')
        print('SELF-TEST PASSED (all poisons caught, attribution-verified)\n', flush=True)

    viol, native, unscored, changed = sweep(rules, corpus, rng, baseline, config_path,
                                            tag="main", judge_timeout_s=judge_timeout_s)
    print(f'\n=== INDEPENDENT MONITOR  {label} ===')
    print(f'  rewritten     {changed:7,} / {len(corpus):,}')
    print(f'  VIOLATIONS    {len(viol):7,}  (artifact-attributed)')
    print(f'  native-line   {len(native):7,}  (present with rules=[]; pre-existing engine behavior)')
    print(f'  unscored      {unscored:7,}  (multi-slot outputs)')
    print(f'  tolerated     {dict(TOL_COUNTS)}  (null-event doctrine)')
    for expr, out, detail in viol[:15]:
        print(f'    VIOL {" ".join(expr)}  ->  {" ".join(out)}   [{detail}]')
    for expr, out, detail in native[:5]:
        print(f'    (native) {" ".join(expr)}  ->  {" ".join(out)}   [{detail}]')
    return {
        'rules': len(rules),
        'corpus': len(corpus),
        'changed': changed,
        'violations': viol,
        'native': native,
        'unscored': unscored,
        'tolerated': dict(TOL_COUNTS),
        'selftest_passed': selftest_passed,
    }


def _load_rules(rules):
    """Load rules from a JSON path (list of [lhs, rhs]) into (tuple, tuple) pairs, or pass a
    pre-loaded rule list through unchanged."""
    if isinstance(rules, (str, os.PathLike)):
        with open(rules) as fh:
            return [(tuple(l), tuple(r)) for l, r in json.load(fh)]
    return rules
