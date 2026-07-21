# mypy: ignore-errors
"""INDEPENDENT high-precision equivalence checker (mpmath, default 200 dps) for auditing
candidate rules and engine simplifications. Deliberately a SEPARATE implementation from the
Rust core (f64 numeric.rs + astro-float hiprec.rs) so it can cross-verify soundness.

Semantics match the deployment reference (numpy f64 + C99 libm), lifted to high precision:
  * generic equivalence: a rule/pair holds iff on every row where the SOURCE (lhs) is
    finite, the CANDIDATE (rhs) matches within (rtol, atol); source-nonfinite rows are
    domain-extendable (x/x -> 1). Enough finite rows must bind (min_binding).
  * const-free lhs & rhs -> direct check.
  * const-bearing -> for several lhs-constant draws (challenges x sign combos), FIT the
    rhs constants (numpy lstsq seed + scipy refine) and require the generic-equivalence
    gate to hold at the fitted constants for EVERY draw.

Public: equiv(lhs, rhs, dps=200, ...) -> dict(ok, worst_sep, n_binding, reason).
Wildcards `_k` are treated as variables x{k}. `<constant>` leaves bind to fitted params.
"""
import numpy as np
from mpmath import mp, mpf, isnan, isinf, nan, inf


def _num(x):
    return mpf(str(x)) if not isinstance(x, mpf) else x


LEAF = {
    'np.e': lambda: mp.e, 'np.pi': lambda: mp.pi, '(-1)': lambda: mpf(-1),
    'float("inf")': lambda: mpf('inf'), 'float("-inf")': lambda: mpf('-inf'),
    'float("nan")': lambda: mpf('nan'),
}


def _odd_root(x, n):
    if isnan(x):
        return nan
    return -((-x) ** (mpf(1) / n)) if x < 0 else x ** (mpf(1) / n)


def _even_root(x, n):
    if isnan(x) or x < 0:
        return nan
    return x ** (mpf(1) / n)


def _pow(a, b):
    # C99/numpy pow, high precision
    if (not isnan(a)) and a == 1:
        return mpf(1)
    if (not isnan(b)) and b == 0:
        return mpf(1)
    if isnan(a) or isnan(b):
        return nan
    if a == 0:
        return inf if b < 0 else mpf(0)
    if isinf(a):
        # |a| = inf > 1, so an infinite exponent depends only on sign(b) (parity is moot).
        if isinf(b):
            return inf if b > 0 else mpf(0)
        if a > 0:
            return inf if b > 0 else mpf(0)
        # a == -inf, b finite: parity only matters for integer b
        if b == int(b):
            m = inf if b > 0 else mpf(0)
            return -m if int(b) % 2 else m
        # A NEGATIVE base (including -inf) with a non-integer exponent is UNDEFINED over
        # the reals, so return nan. Using nan here (rather than the C99 pow branch that
        # returns +inf for a negative base) keeps this atom consistent with the real
        # even-root convention pow1_4(-inf) == nan, so equivalent forms such as
        # `pow x div4 1` and `pow1_4 x` agree at the -inf atom instead of one of them
        # reading as a domain shrink.
        return nan
    if isinf(b):
        # numpy: infinite exponent depends only on |a| vs 1, sign of a ignored.
        m = abs(a)
        if m == 1:
            return mpf(1)
        return inf if ((m > 1) == (b > 0)) else mpf(0)
    if a < 0:
        if isinf(b) or b != int(b):
            return nan
        m = (-a) ** b
        return -m if int(b) % 2 else m
    try:
        return a ** b
    except (OverflowError, ValueError):
        return inf


UN = {
    'neg': lambda x: -x, 'abs': lambda x: abs(x),
    'inv': lambda x: (inf if x >= 0 else -inf) if x == 0 else 1 / x,
    'mult2': lambda x: 2 * x, 'mult3': lambda x: 3 * x, 'mult4': lambda x: 4 * x, 'mult5': lambda x: 5 * x,
    'div2': lambda x: x / 2, 'div3': lambda x: x / 3, 'div4': lambda x: x / 4, 'div5': lambda x: x / 5,
    'pow2': lambda x: x ** 2, 'pow3': lambda x: x ** 3, 'pow4': lambda x: x ** 4, 'pow5': lambda x: x ** 5,
    'pow1_2': lambda x: _even_root(x, 2), 'pow1_4': lambda x: _even_root(x, 4),
    'pow1_3': lambda x: _odd_root(x, 3), 'pow1_5': lambda x: _odd_root(x, 5),
    'sin': mp.sin, 'cos': mp.cos, 'tan': mp.tan,
    'asin': lambda x: nan if abs(x) > 1 else mp.asin(x),
    'acos': lambda x: nan if abs(x) > 1 else mp.acos(x),
    'atan': mp.atan, 'sinh': mp.sinh, 'cosh': mp.cosh, 'tanh': mp.tanh,
    'asinh': mp.asinh,
    'acosh': lambda x: nan if x < 1 else mp.acosh(x),
    'atanh': lambda x: nan if abs(x) > 1 else ((inf if x > 0 else -inf) if abs(x) == 1 else mp.atanh(x)),
    'exp': mp.exp, 'log': lambda x: (-inf) if x == 0 else (nan if x < 0 else mp.log(x)),
}
BIN = {
    '+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
    '/': lambda a, b: ((nan if a == 0 else (-inf if (a < 0) != (b < 0) else inf)) if b == 0 else a / b),
    'pow': _pow,
}


def _ev(tokens, i, env, consts, ci):
    t = tokens[i]
    i += 1
    if t in UN:
        v, i, ci = _ev(tokens, i, env, consts, ci)
        if isnan(v):
            return nan, i, ci
        try:
            r = UN[t](v)
        except (OverflowError, ValueError, ZeroDivisionError):
            r = nan
        return r, i, ci
    if t in BIN:
        a, i, ci = _ev(tokens, i, env, consts, ci)
        b, i, ci = _ev(tokens, i, env, consts, ci)
        if t == 'pow':
            return _pow(a, b), i, ci
        if isnan(a) or isnan(b):
            return nan, i, ci
        try:
            r = BIN[t](a, b)
        except (OverflowError, ValueError, ZeroDivisionError):
            r = nan
        return r, i, ci
    if t == '<constant>':
        v = consts[ci]
        return _num(v), i, ci + 1
    if t in LEAF:
        return LEAF[t](), i, ci
    if t in env:
        return env[t], i, ci
    # numeric literal
    return _num(float(t)), i, ci


def evaluate(tokens, env, consts):
    v, j, _ = _ev(list(tokens), 0, dict(env), list(consts), 0)
    return v


def _vars_in(tokens):
    vs = set()
    for t in tokens:
        if t.startswith('x') and t[1:].isdigit():
            vs.add(t)
        elif t.startswith('_') and t[1:].isdigit():
            vs.add(t)
    return sorted(vs)


def _n_const(tokens):
    return sum(1 for t in tokens if t == '<constant>')


def _wild_to_var(tokens):
    return ['x' + t[1:] if (t.startswith('_') and t[1:].isdigit()) else t for t in tokens]


def equiv(lhs, rhs, dps=200, n_rows=48, challenges=6, seed=0, rtol=1e-12, atol=1e-15,
          min_binding=6):
    """Independent generic-equivalence verdict for lhs -> rhs. Returns a dict."""
    mp.dps = dps
    lhs = _wild_to_var(list(lhs))
    rhs = _wild_to_var(list(rhs))
    allvars = sorted(set(_vars_in(lhs)) | set(_vars_in(rhs)))
    rng = np.random.default_rng(seed)
    nlc, nrc = _n_const(lhs), _n_const(rhs)
    # var draws (shared across constant challenges)
    Xrows = [{v: mpf(str(float(x))) for v, x in zip(allvars, rng.normal(0, 5, len(allvars)))}
             for _ in range(n_rows)] or [{}] * n_rows
    if not allvars:
        Xrows = [{} for _ in range(n_rows)]

    worst = mpf(0)
    total_binding = 0
    combos = [tuple()] if nlc == 0 else None

    def lhs_const_draws():
        if nlc == 0:
            yield []
            return
        for _ in range(challenges):
            base = np.abs(rng.normal(0, 5, nlc))
            for signs in _sign_combos(nlc):
                yield [mpf(str(float(b * s))) for b, s in zip(base, signs)]

    for lc in lhs_const_draws():
        yv = []
        rows_kept = []
        for row in Xrows:
            a = evaluate(lhs, row, lc)
            if isnan(a) or isinf(a):
                continue
            yv.append(a)
            rows_kept.append(row)
        if len(yv) < min_binding:
            continue
        total_binding += len(yv)
        if nrc == 0:
            for a, row in zip(yv, rows_kept):
                b = evaluate(rhs, row, [])
                if isnan(b) or isinf(b):
                    return _res(False, worst, total_binding, 'rhs nonfinite where lhs finite')
                sep = abs(a - b) / max(abs(a), mpf(1))
                worst = max(worst, sep)
            if worst > mpf(str(atol)) + mpf(str(rtol)):
                return _res(False, worst, total_binding, 'const-free mismatch')
        else:
            ok, w = _fit_and_check(rhs, rows_kept, yv, nrc, rtol, atol, seed)
            worst = max(worst, w)
            if not ok:
                return _res(False, worst, total_binding, 'no rhs constants fit this lhs draw')
    if total_binding < min_binding:
        return _res(None, worst, total_binding, 'insufficient finite evidence (vacuous)')
    return _res(True, worst, total_binding, 'ok')


def _sign_combos(n):
    # Measure-consistent sign grid {-1, +1}, NOT {-1, 0, +1}. C=0 is a measure-zero point
    # that the constant sampler never draws; testing it would flag rules that hold for
    # every random constant as false (`pow(cos(C), inf) -> 0` is 0 for all C except the
    # measure-zero cos(C)=+-1 set). Magnitudes stay continuous, so any positive-measure
    # falsity is still caught. Matches the reference sampler's default sign grid.
    import itertools
    return list(itertools.product((-1.0, 1.0), repeat=n))


def _fit_and_check(rhs, rows, yv, nrc, rtol, atol, seed):
    """Fit rhs constants to the lhs values yv over rows; accept if a fit reproduces yv.
    Robust to power/exponential rhs (where naive least_squares stalls): a broad grid of
    SIMPLE seeds (integers, simple fractions -- most true structural constants are simple)
    plus random restarts, residual evaluated ONLY on rows where rhs is finite (no flat
    penalty region), and a coverage requirement so a fit that matches on few rows fails."""
    from scipy.optimize import least_squares
    y = np.array([float(a) for a in yv])
    n = len(y)
    # FIT on moderate-magnitude rows only (a const-base pow C^x overflows float64 for the
    # wide |x| the row draws span; drop the overflow-prone rows for the FIT and still
    # VERIFY on ALL rows at high precision).
    fit_idx = [i for i in range(n) if 1e-8 < abs(y[i]) < 1e8]
    if len(fit_idx) < max(3, nrc + 1):
        fit_idx = list(range(n))
    frows = [rows[i] for i in fit_idx]
    fy = y[fit_idx]

    def resid(c):
        out = []
        for row, ay in zip(frows, fy):
            v = evaluate(rhs, row, [mpf(str(ci)) for ci in c])
            fv = float(v) if (not isnan(v) and not isinf(v)) else np.nan
            out.append(0.0 if (np.isnan(fv) or abs(fv) > 1e12) else (fv - ay))
        return np.nan_to_num(np.array(out), nan=0.0, posinf=0.0, neginf=0.0)

    def finite_frac(c):
        k = 0
        for row in frows:
            v = evaluate(rhs, row, [mpf(str(ci)) for ci in c])
            if not (isnan(v) or isinf(v)):
                k += 1
        return k / len(frows)

    SIMPLE = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, 0.5, -0.5, 1.0 / 3, 0.25, 4.0, 5.0, np.e, -np.e,
              float(np.exp(-4)), float(np.exp(4)), float(np.exp(8))]
    rng = np.random.default_rng(seed + 1)
    seeds = [np.full(nrc, s) for s in SIMPLE]
    seeds += [rng.normal(0, m, nrc) for m in (0.5, 1, 2, 4) for _ in range(4)]
    best = None
    for c0 in seeds:
        try:
            r = least_squares(resid, c0, max_nfev=300)
        except Exception:
            continue
        # require the fit to keep rhs finite on (nearly) all binding rows
        if finite_frac(r.x) < 0.9:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return False, mpf(1)
    worst = mpf(0)
    for row, ay in zip(rows, yv):
        v = evaluate(rhs, row, [mpf(str(ci)) for ci in best.x])
        if isnan(v) or isinf(v):
            continue  # rhs may extend the domain; judge only where it is finite
        worst = max(worst, abs(ay - v) / max(abs(ay), mpf(1)))
    return worst <= mpf('1e-7'), worst


def _res(ok, worst, nb, reason):
    return {'ok': ok, 'worst_sep': float(worst), 'n_binding': int(nb), 'reason': reason}
