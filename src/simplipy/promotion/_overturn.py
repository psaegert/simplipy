# mypy: ignore-errors
"""Exact re-arbitration of finite-draw demotions using arbitrary-precision arithmetic.

A pointwise DEMOTE that was decided at a FINITE numeric draw is only float64 evidence, and
float64 saturation can spuriously kill an exact identity (for example ``atanh(tanh(x)) = x``,
whose float64 round-trip saturates at large-magnitude draws and reads as a mismatch). This
module re-judges exactly those finite-draw demotions in mpmath at adaptive precision:

  - each finite draw is evaluated at ADAPTIVE dps = 50 + int(max|v|) + 10, and again at twice
    that precision. The ``+ int(max|v|)`` term adds one working decimal digit per unit of
    |v|, which comfortably exceeds the ~0.87 digits per unit that the saturating round-trip's
    conditioning actually loses; a draw counts as evidence only if BOTH precisions agree
    (a precision-stability gate against precision-cliff artefacts);
  - a draw whose required adaptive dps exceeds ``DPS_CAP`` is UNDECIDABLE, in which case the
    demotion STANDS (fail-safe: never overturn on evidence that could not be computed).

A rule is overturned (DEMOTE -> PROMOTE) only when every valuation agrees exactly under this
arbitration. Atom-only kills are not re-judged here: they are special-value algebra that is
already exact in float64, so a driver should exclude any demotion whose valuation is drawn
entirely from ``ATOMS`` before calling ``judge_exact``.
"""
import numpy as np

from ._pointwise import ATOMS, wildcards, valuations_for
from ._hp_equiv import evaluate as hp_eval, _restores_dps
from mpmath import mp, mpf, isnan, isinf, inf, nan, fabs

SEED = 20260717
DPS_CAP = 800

ATOM_VALS = {'0.0': mpf(0), '-0.0': mpf(0), '1.0': mpf(1), '(-1.0)': mpf(-1),
             'float("inf")': inf, 'float("-inf")': -inf, 'float("nan")': nan}


def tok_to_mp(t):
    if t in ATOM_VALS:
        return ATOM_VALS[t]
    return mpf(t.strip('()'))


def eq(a, b):
    if isnan(a) and isnan(b):
        return True
    if isnan(a) != isnan(b):
        return False
    if isinf(a) or isinf(b):
        return a == b
    return fabs(a - b) <= mpf('1e-30') * max(1, fabs(a), fabs(b))


@_restores_dps
def judge_exact(lhs, rhs, valuations):
    """Re-judge a rule over the supplied valuation set in mpmath.

    Returns ``('PROMOTE', None)`` when every valuation agrees exactly; ``('DEMOTE-EXACT',
    wmap)`` when a valuation disagrees under the precision-stability gate; or
    ``('UNDECIDABLE', wmap)`` when a valuation needs more precision than ``DPS_CAP`` allows
    or evaluation raises. The paired ``wmap`` is the offending valuation.
    """
    for wmap in valuations:
        vals = {w: tok_to_mp(t) for w, t in wmap.items()}
        finmax = max([abs(float(v)) for v in vals.values() if not (isnan(v) or isinf(v))] or [0.0])
        need = 50 + int(finmax) + 10
        if need > DPS_CAP:
            return 'UNDECIDABLE', dict(wmap)
        verdicts = []
        for dps in (need, 2 * need):
            mp.dps = dps
            env = {w: (v if (isnan(v) or isinf(v)) else mpf(repr(float(v)))) for w, v in vals.items()}
            try:
                a = hp_eval(list(lhs), env, [])
                b = hp_eval(list(rhs), env, [])
            except Exception:
                return 'UNDECIDABLE', dict(wmap)
            verdicts.append(eq(a, b))
        if not (verdicts[0] and verdicts[1]):
            return 'DEMOTE-EXACT', dict(wmap)
    return 'PROMOTE', None
