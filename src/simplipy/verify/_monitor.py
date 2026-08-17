# mypy: ignore-errors
"""Independent deployed-realization monitor: an end-to-end soundness gate.

This monitor is deliberately independent of the certifier's own judges. Its evaluator,
operator semantics, probe machinery, and corpus are written from the simplification
contract alone, so it can catch violations that a checker sharing the certifier's own
machinery would structurally miss. It exercises the DEPLOYED engine end-to-end
(``simplify`` on real expression shapes; masking is EXPLICIT in 0.12, so the corpus
carries both literal and pre-masked shapes) and judges every rewrite under REAL (exact)
semantics. The judge speaks the engine's full 0.12 output language: n-ary bag tags
(``<add> .. <sub> .. </add>``, ``<mul> .. <div> .. </mul>``), ``rootn``, and one-token
exact-rational coefficients (``1/3``). Tokens OUTSIDE that language fail CLOSED
(UNSCORED) -- an output the judge cannot read must never silently pass as OK.

  Values are compared in mpmath (no float64 saturation/overflow); a single unsigned zero;
  1/0 = +inf; IEEE pow specials (pow(+-1, +-inf) = 1, pow(0, 0) = 1); literal folding
  (np.pi -> pi). A DEFINED value may never change (checked pointwise; symbolic points such
  as pi/2 are probe points). Extending an undefined point to a defined one is tolerated on
  NULL sets only: extension at isolated atoms is tolerated, while extension on a large
  fraction of continuum draws is a positive-measure violation.

  F74 (2026-08-10, the honest-instrument upgrade): pure-rational variable-free spans fold
  EXACTLY in Fraction arithmetic before any mpf sees them; each pair is judged at a working
  precision adapted to its own literal exponent span; the exact-snap resolves variable-free
  transcendental spans to provable zeros/small integers on an escalating precision ladder
  (never at fixed rungs); refusals (Unresolved) propagate honestly out of c_pow instead of
  fabricating nan; and values whose distance from a decision boundary is smaller than ANY
  working precision (tanh saturation) refuse rather than fabricate the boundary value.

Verdict classes per rewrite: OK | VIOLATION (value change / shrink / positive-measure
extension) | UNSCORED (multi-slot outputs, unknown tokens, no evaluable probes). A
violation that is ALSO present under the rules-empty engine is bucketed NATIVE-LINE
(pre-existing engine behavior, not attributable to the ruleset under test).

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
import re
import tempfile
from decimal import Decimal, InvalidOperation
from fractions import Fraction

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
                          # doctrine, as are correlated all-equal (diagonal) draws for DATA
                          # variables -- a Lebesgue-null subset (see judge_pair).

TOL_COUNTS = {'shrink-null': 0, 'inf-null': 0, 'ext-null': 0, 'singular-real-null': 0}
N_DRAWS = 40                # continuum draws per variable set


def _ambient_dps(fn):
    """C35: establish the monitor's working precision at the ENTRY POINTS and restore
    the host's on exit -- the former module-level `mp.dps = DPS` mutated the embedder's
    mpmath state at import. The evaluator deliberately reads the AMBIENT `mp.dps`
    (F74: the bands ride it, so the adaptive ladder can flex precision mid-judgment);
    this decorator is what makes 'ambient' start at DPS for every public entry.
    Direct importers of `evaluate` (the fuzz harness) manage their own precision."""
    def _wrap(*a, **k):
        old = mp.dps
        mp.dps = DPS
        try:
            return fn(*a, **k)
        finally:
            mp.dps = old
    return _wrap


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
    """A denominator indistinguishable from an exact pole at working precision is a pole.
    (F74: the band rides mp.dps, not the ambient DPS constant -- under the adaptive
    per-pair precision the working dps is the pair's, and a band frozen at the ambient
    value would be 10^600 too loose on an extreme-literal pair judged at dps 700.)"""
    return abs(denom) < mpf(10) ** (-(mp.dps - 10))


def _residue_guard(x):
    """A nonzero finite value indistinguishable from an exact ZERO at working precision.
    A symbolic cancellation (x17 + x15 - x15 - x17) leaves a ~1e-50 residue whose SIGN is
    rounding noise; feeding it to a domain-edged or root-amplifying function fabricates
    verdicts two ways (both observed, 2026-07-29 corpus adjudication):
      * even root / non-integer pow: the noise sign decides defined-vs-nan, so ~half the
        draws report 'input undefined' -- a fabricated positive-measure EXTENSION;
      * odd root: |residue|^(1/k) AMPLIFIES 1e-50 to 1e-10, a 'stable value change' that
        the dps-120 recheck cannot clear (the amplified residue shrinks too slowly).
    Same precision-honesty doctrine as _pole_guard: skip (Unresolved), never convict."""
    return x != 0 and not isinf(x) and abs(x) < mpf(10) ** (-(mp.dps - 10))


def c_div(a, b):
    if isnan(a) or isnan(b):
        return _nan()
    if isinf(a) and isinf(b):
        return _nan()
    if b != 0 and not isinf(b) and _pole_guard(b):
        # A denominator indistinguishable from an EXACT pole at working precision: a
        # symbolic cancellation (log(exp(x)) - x) leaves a ~1e-50 residue whose SIGN is
        # rounding noise, and 1/residue then differs from the exact 1/0 = +inf by that
        # noise sign -- a fabricated stable 'violation' against a sound fold (audit
        # finding). Same precision-honesty doctrine as the atanh/acosh bands: skip,
        # never convict.
        raise Unresolved()
    if b == 0:                               # ONE zero; c/0 = sign(c)*inf, 0/0 = nan
        if a == 0:
            return _nan()
        if isinf(a) or a > 0:
            return mpf('inf') if (not isinf(a) or a > 0) else mpf('-inf')
        return mpf('-inf')
    if isinf(b):
        return mpf(0) if not isinf(a) else _nan()
    return a / b


def _int_or_none(b):
    """b as an exact int, None if non-integer, 'unresolvable' when integrality is
    unknowable at working precision (H-050, extreme-literal lane 2026-08-05; refined
    after 1M row 294377).

    The resolvability criterion is REPRESENTATIONAL: mpmath normalizes the mantissa
    ODD, so for finite nonzero b = (-1)^s * man * 2^exp with bc = bitcount(man), the
    value's full integer part occupies bc + max(exp, 0) bits. While that fits the
    working precision (with guard bits), the mpf IS the exact value it claims and
    exp < 0 <=> non-integer, exp >= 0 <=> integer with parity read off man/exp --
    exact for every literal the token grammar carries (9007199254740993 at 54 bits,
    10^19+1 at 67, 2^127 at 127, 1e40 at 133: all resolve at dps 50 / prec 166,
    fixing the convictions of sound spelling-parity engine folds, smoke rows
    32/301/796). BEYOND the precision budget the mpf is in general a ROUNDING of its
    true value, and representation-integrality says nothing: exp(2.703^5) ~ 6e63 is
    PROVABLY irrational (Lindemann) yet its 166-bit mpf reads as an even integer --
    the read fabricated definedness for pow(-1, <that>) and convicted the engine's
    correct Nan (1M row 294377). Those return 'unresolvable' -> nan, the old
    PARITY_CAP's honest verdict, now stated in representation terms instead of a
    magnitude constant (int(b) is precision-bounded, so the bignum hang the old cap
    also guarded against cannot occur).
    """
    _sign, man, exp, bc = b._mpf_
    if man == 0:
        return None  # zero/specials (callers pre-filter)
    if bc + max(exp, 0) > mp.prec - 10:
        return 'unresolvable'
    if exp < 0:
        return None
    return int(b)


def _int_value_or_none(b):
    """The exact int VALUE of b; None when non-integer or unresolvable at working
    precision. Consumers that compute WITH the integer (c_rootn's index) refuse
    honestly on the unresolvable class."""
    ib = _int_or_none(b)
    return ib if isinstance(ib, int) else None


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
    if _residue_guard(a):
        # a base indistinguishable from an exact ZERO: only a positive-integer exponent
        # is noise-immune (residue^k is a yet-smaller residue, exact-zero-consistent).
        # A non-integer exponent turns the residue's noise SIGN into defined-vs-nan
        # (fabricated measure events); a negative exponent turns it into a noise-signed
        # huge value where the exact base gives 0^-k = +inf (fabricated INF-CHANGE).
        ib = _int_or_none(b)
        if not (isinstance(ib, int) and ib > 0):
            raise Unresolved()
    if isinf(a):
        if a > 0:
            return mpf('inf') if b > 0 else mpf(0)
        ib = _int_or_none(b)
        if ib == 'unresolvable':
            return _nan()                    # integrality unknowable at working precision
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
    except Unresolved:
        # F74 (D1): this catch-all used to SWALLOW the Unresolved refusal raised by
        # _pow_mag_capped, converting "cannot evaluate at working precision" into a
        # fabricated nan -- which then read as input-side undefinedness and convicted
        # correct rewrites as positive-measure EXTENSION/SHRINK on every draw (~150
        # of the 190 live extreme-lane convictions). The negative-base branch above
        # always propagated; refusals now propagate on both branches.
        raise
    except Exception:
        return _nan()


PHASE_CAP = mpf('1e25')   # a trig argument's phase mod 2pi is only knowable to the precision
                          # of the WEAKEST value feeding it: fitted slots are bound as 40-digit
                          # tokens, so phase is garbage beyond ~1e3x -- observed (cos at ~9e48
                          # under an older 1e50 cap flipped sign purely from slot rounding: a
                          # false violation). 1e25 leaves >= 1e14 phase margin for 40-digit
                          # tokens. (Also guards mpmath's quasi-unbounded argument reduction,
                          # one of its uninterruptible paths.) F74: beyond the cap the verdict
                          # is Unresolved (probe skipped), NOT nan -- sin at a huge finite
                          # exact literal is a defined real the instrument cannot COMPUTE, and
                          # returning nan declared it UNDEFINED, feeding the extension-measure
                          # clause: sin(huge_int * -2) * pow(0, y) convicted the engine's
                          # correct 0 as positive-measure EXTENSION on every draw (extreme row
                          # 966406). The old "symmetric nan" defense fails exactly when the
                          # other side's structure never evaluates the trig at all. Cost:
                          # huge-argument trig pairs go unjudged, never wrongly verdicted.


def _trig_inf(f):
    def g(x):
        if isnan(x) or isinf(x):
            return _nan()                    # no limit at +-inf: genuinely undefined
        if abs(x) > PHASE_CAP:
            raise Unresolved()               # uncomputable phase, never fabricated nan
        return f(x)
    return g


def c_tan(x):
    if isnan(x) or isinf(x):
        return _nan()
    if abs(x) > PHASE_CAP:
        raise Unresolved()                   # see PHASE_CAP: uncomputable != undefined
    c = mp.cos(x)
    if _pole_guard(c):
        return _nan()                        # exact pole at working precision
    return mp.sin(x) / c


def c_inv(x):
    return c_div(mpf(1), x)


MAG_CAP = mpf('1e2000')   # exponent-magnitude cap (F74, raised from 1e50): the cost driver
                          # is the SIZE of the result's exponent integer, which mpmath's
                          # pure-python backend manipulates during argument reduction --
                          # measured on this host: mag 1e312 ~ 5-23 ms, 1e1000 ~ 0.1-0.2 s,
                          # 1e2000 ~ 0.6-1.2 s, 1e4000 ~ 3.5-5 s, 1e9999 ~ 44 s (a
                          # near-hang; SIGALRM cannot fire mid-bignum-op). 1e2000 keeps the
                          # worst single op ~1 s while covering everything the extreme
                          # token grammar builds (single extreme pow ~10^(3e311), and
                          # 1e309-exponent towers stay under 1e2000 through four levels).
                          # The old cap refused magnitudes the backend evaluates exactly in
                          # milliseconds, and c_pow converted the refusal into a fabricated
                          # nan (see c_pow) -- together the mechanism behind most of the
                          # extreme lane's fabricated convictions. Beyond the cap: raise
                          # Unresolved SYMMETRICALLY on both sides (probe skipped, never a
                          # wrong verdict).


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
    v = mp.exp(x)
    if v == 1 and x != 0:
        # F74 D5 sibling (found by the upgrade's own row-set diff, extreme row
        # 924168): exp(x) = 1 iff x = 0 over the reals, so an EXACT 1 on a nonzero
        # argument is always a rounding erasure -- exp(exp(-1e309)) sits
        # 10^{-4.34e308} ABOVE 1, beyond any working precision, and the erased side
        # decided asin's domain: the snap ladder accepted the span as exactly 1 and
        # the judge convicted the engine's CORRECT nan. Refuse, never fabricate the
        # boundary value (same provable-erasure argument as the tanh/cosh guards).
        raise Unresolved()
    return v


def _hyp(f):
    def g(x):
        if isnan(x):
            return _nan()
        if isinf(x):
            if f is mp.cosh:
                return mpf('inf')
            return (mpf('inf') if x > 0 else mpf('-inf')) if f is mp.sinh else (mpf(1) if x > 0 else mpf(-1))
        if abs(x) > MAG_CAP:
            raise Unresolved()   # sinh/cosh: see c_exp rationale; tanh: side erasure below
        v = f(x)
        # F74 (D5) saturation-side honesty: |tanh(x)| < 1 STRICTLY for every finite x,
        # so mp.tanh returning EXACTLY +-1 is always a rounding erasure -- the true
        # value sits 2e^{-2|x|} inside the boundary, below working precision (tanh(9e15)
        # is 10^{-8e15} below 1: no dps resolves it). The erased side/gap decides
        # side-sensitive consumers (pow with an infinite exponent reads pow(1,-inf)=1
        # where the truth is pow(1-,-inf)=+inf; acosh reads 0 where the truth is nan) --
        # both observed as fabricated convictions of correct engine outputs. Same for
        # cosh(x) = 1 with x != 0 (cosh > 1 strictly off zero). Unresolved, never a
        # fabricated boundary value; resolvable cases resolve under the adaptive
        # per-pair precision before this fires. (The old tanh beyond-cap branch
        # returned the exact limit +-1 -- the same erasure spelled at the cap.)
        if f is mp.tanh and abs(v) == 1:
            raise Unresolved()
        if f is mp.cosh and v == 1 and x != 0:
            raise Unresolved()
        return v
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
    v = mp.power(a, b)
    if v == 1:
        # a^b = 1 with a != 1 (guarded above) iff b = 0, and b = 0 never reaches
        # this helper (c_pow returns 1 first) -- an exact-1 result here is the same
        # provable rounding erasure as c_exp's (pow(2, exp(exp(-1e309)))-shapes).
        raise Unresolved()
    return v


def _odd_root(k):
    def g(x):
        if isnan(x):
            return _nan()
        if isinf(x):
            return x
        if _residue_guard(x):
            raise Unresolved()
        r = mp.power(abs(x), mpf(1) / k)
        return -r if x < 0 else r
    return g


def _even_root(k):
    def g(x):
        if not isnan(x) and _residue_guard(x):
            raise Unresolved()               # residue sign decides defined-vs-nan: unknowable
        if isnan(x) or x < 0:                # sign check FIRST: even root of -inf is nan
            return _nan()                    # (isinf-first returned +inf for -inf: a live bug)
        if isinf(x):
            return mpf('inf')
        return mp.power(x, mpf(1) / k)
    return g


def c_rootn(x, n):
    """IEEE rootn (the 0.12 general root operator): the index must be a finite nonzero
    integer -- n == 0 or a non-integer index is an invalid operation (nan); n == 1 is
    the identity; a negative index is 1/rootn(x, -n) (one zero: rootn(0, -n) = +inf);
    odd n is the signed total root, even n the principal root (nan on negatives,
    including -inf)."""
    if isnan(x) or isnan(n) or isinf(n):
        return _nan()
    ni = _int_value_or_none(n)  # the index is COMPUTED WITH: value-strict (H-050)
    if ni is None or ni == 0:
        return _nan()
    if ni < 0:
        return c_inv(c_rootn(x, mpf(-ni)))
    if ni == 1:
        return x
    return (_even_root(ni) if ni % 2 == 0 else _odd_root(ni))(x)


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
    'rootn': (2, c_rootn),
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
    # -- retired generation-1 vocabulary (simplipy <= 0.11) below (except the live odd
    # roots pow1_3/pow1_5): kept so the monitor INSTRUMENT stays vocabulary-complete for
    # auditing legacy-spelled rule sets (raw-table construction is a sanctioned boundary;
    # the engine itself refuses generation-1 artifacts at load, simplipy.compat).
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

# The 0.12 engine's n-ary output grammar: `<add> pos.. [<sub> mag..] </add>` (a bag never
# contains negative literals; subtracted terms sit behind `<sub>` as magnitudes) and
# `<mul> num.. [<div> den..] </mul>` (negative-rational-exponent factors sit behind
# `<div>` with the exponent flipped). Coefficients are one-token exact rationals
# (`7`, `1.2`, `1/3`).
BAG_OPEN = {'<add>': ('<sub>', '</add>', '+', '-'),
            '<mul>': ('<div>', '</mul>', '*', '/')}
BAG_TOKENS = frozenset(BAG_OPEN) | {'<sub>', '</add>', '<div>', '</mul>'}

RAT_RE = re.compile(r'^-?\d+/\d+$')      # the exact-rational coefficient spelling
VAR_RE = re.compile(r'^x\d+$')
SORT_RE = re.compile(r'^[_?!$]\d+$')     # rule-sort slots, judged as quantified reals


def _raise_unresolved():
    raise Unresolved()


def _slot_token(v):
    """Bind a fitted slot at FULL working precision. Binding via '%r' % float(v) rounded the
    constant to f64 (1e-17 error) -- which the two-rung stability gate then CORRECTLY reported
    as a stable value change on perfectly sound rewrites (observed, both in flagged rules and
    as ~700 spurious native-line entries)."""
    s = mp.nstr(abs(v), 40)
    return s if v >= 0 else f'(-{s})'


def evaluate(tokens, env, node_hook=None):
    """Contract-semantics value of a prefix expression at env (variable name -> mp value).
    Speaks both the binary spelling and the engine's n-ary bag grammar. `node_hook`, when
    given, receives every COMPUTED node value: op results, bag member-inverses and bag
    fold intermediates. Leaves (env values, literals) are deliberately excluded -- a
    literal float("inf") is a DENOTED conventional constant, not a singularity crossing
    (see the singular-input gate)."""
    pos = 0

    def walk():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t in BAG_OPEN:
            marker, close, fold_op, _join_op = BAG_OPEN[t]
            fold = OPS[fold_op][1]
            main, sec = [], []
            cur = main
            while pos < len(tokens) and tokens[pos] != close:
                if tokens[pos] == marker:
                    pos += 1
                    cur = sec
                    continue
                cur.append(walk())
            if pos >= len(tokens):
                raise ValueError(f'unclosed {t}: {tokens}')
            pos += 1                                       # the close tag
            # The sections denote PER-MEMBER inverse elements folded into ONE flat bag --
            # the internal form is Mul[num.., Pow(d1,-1), Pow(d2,-1)] / Add[pos.., -m1..],
            # never num/(d1*d2). The grouping matters at extension points: with d2 = 0 and
            # d1 < 0, d1^-1 * d2^-1 = -inf while num/(d1*d2) = num/0 = +inf (the sign of
            # d1 is erased through d1*0 = 0) -- the grouped spelling fabricated a
            # positive-measure INF-CHANGE verdict against a sound rewrite (observed, 1M
            # corpus row 26797). Negation distributes totally over the clause-guarded
            # addition, so the <sub> section may negate per member for the same
            # faithfulness. Folds within the flat bag are order-invariant on the
            # special-value lattice {0, +-inf, nan}.
            inv_member = c_inv if fold_op == '*' else (lambda v: OPS['neg'][1](v))
            neutral = mpf(0) if fold_op == '+' else mpf(1)
            acc = main[0] if main else neutral
            for v in main[1:]:
                acc = fold(acc, v)
                if node_hook is not None:
                    node_hook(acc)
            for v in sec:
                iv = inv_member(v)
                if node_hook is not None:
                    node_hook(iv)
                acc = fold(acc, iv)
                if node_hook is not None:
                    node_hook(acc)
            return acc
        if t in OPS:
            ar, f = OPS[t]
            args = [walk() for _ in range(ar)]
            v = f(*args)
            if node_hook is not None:
                node_hook(v)
            return v
        if t in env:
            return env[t]
        if t in LITS:
            return LITS[t]()
        s = t.strip('()')
        if RAT_RE.match(s):
            p, q = s.split('/')
            return mpf(p) / mpf(q)                         # exact int/int at working dps
        return mpf(s)
    v = walk()
    if pos != len(tokens):
        raise ValueError(f'trailing tokens: {tokens}')
    return v


def _input_singular(tokens, env):
    """True iff evaluating `tokens` at env crosses an internal +-inf: some COMPUTED node
    (op result or bag fold step) is infinite. mpmath at working dps does not overflow to
    inf at reachable magnitudes, so an internal inf marks a genuine singular manifold of
    the expression (1/0 under one-zero, log 0, atanh +-1) -- the probe point carries NO
    mathematical value for the expression, and any real number downstream is
    convention-completion (9.8.1 one-zero + 9.8.2 limit-completion). Leaf literals
    (float("inf")) do NOT taint: a denoted inf constant has its ratified limit-completed
    value, so the strict clause keeps authority over it. Internal nan-to-real
    (pow(nan, 0) = 1) is likewise NOT tainted: nan-pow cells are ratified DEFINED values
    (v2 section 2), not completions. Exceptions fail False -- toward strict authority."""
    seen = [False]

    def hook(v):
        if isinf(v):
            seen[0] = True

    try:
        evaluate(list(tokens), env, node_hook=hook)
    except Exception:
        return False
    return seen[0]


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
    """STRICT variable classification (x-vars and rule-sort slots only). The pre-fix
    catch-all -- 'anything not an op/literal is a variable' -- silently turned unknown
    OPERATOR tokens into free variables and judged garbage; unknowns now go through
    unknown_tokens() and fail closed."""
    return sorted({t for t in tokens if VAR_RE.match(t) or SORT_RE.match(t)})


def unknown_tokens(tokens):
    """Tokens outside the judge's language. The fail-closed hinge: anything unknown must
    surface as UNSCORED, never quietly become a free variable or a skipped probe (the
    pre-fix judge OK'd every bag-tagged AC output through exactly that hole)."""
    return sorted({t for t in tokens
                   if t not in OPS and t not in LITS and t not in BAG_TOKENS
                   and t != '<constant>' and not VAR_RE.match(t)
                   and not SORT_RE.match(t) and not _is_num(t)})


def _is_num(t):
    s = t.strip('()')
    if RAT_RE.match(s):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _subtree_ends(tokens):
    """ends[i] = one past the last token of the subtree rooted at position i (bag
    grammar included). Positions holding section markers / close tags keep ends[i]=0."""
    n = len(tokens)
    ends = [0] * n

    def walk(i):
        j = i + 1
        if tokens[i] in BAG_OPEN:
            marker, close = BAG_OPEN[tokens[i]][:2]
            while j < n and tokens[j] != close:
                if tokens[j] == marker:
                    j += 1
                else:
                    j = walk(j)
            if j < n:
                j += 1                       # the close tag
        else:
            for _ in range(OPS.get(tokens[i], (0,))[0]):
                j = walk(j)
        ends[i] = j
        return j
    walk(0)
    return ends


def _var_free_subtree_spans(tokens):
    """EVERY variable-free subtree span [s, e) with at least two tokens, innermost
    (shortest) first. F74: the pre-upgrade snap scanned MAXIMAL spans only, so an
    inner exact zero (tan pi) was never snapped when the enclosing span evaluated
    cleanly -- and its residue then propagated as a fabricated non-zero through
    pow(nan, .) (extreme row 636919). Innermost-first ordering makes the caller's
    snap-and-rescan loop settle inner exactness before outer spans are judged."""
    ends = _subtree_ends(tokens)

    def var_free(i):
        return all(tokens[k] in OPS or tokens[k] in LITS or tokens[k] in BAG_TOKENS
                   or _is_num(tokens[k]) for k in range(i, ends[i]))
    spans = [(i, e) for i, e in enumerate(ends) if e - i >= 2 and var_free(i)]
    spans.sort(key=lambda se: (se[1] - se[0], se[0]))
    return spans


def _fold_noise_floor(tokens, env, base):
    """The pair's own f64 fold-noise floor at this probe. The engine's constant folder
    works in DOUBLES, so every printed non-integer numeric literal in its output carries
    ulp-scale fold rounding (correctly rounded: <= 0.5 ulp; multi-op folds a few ulp) --
    and the fold SOURCE need not be a variable-free subtree the judge could canonicalize
    (observed: asin(x14/x14) -> the f64 pi/2 literal, via a licensed x/x -> 1 rewrite
    first). By local linearity, the summed response to a 2^-52-relative wiggle of each
    such literal (4x the correctly-rounded bound) is an upper envelope of what fold
    rounding alone can move the value -- co-scaling with any downstream condition
    number, which a fixed end-to-end band cannot (observed: asin-1's 1.2e-17 offset
    amplified to a stable 1e-14 'value change' by a tan-cos composition). Exact
    integers and exact-rational coefficients ride the engine's exact paths: no noise."""
    noise = mpf(0)
    eps = mpf(2) ** -52
    for k, t in enumerate(tokens):
        s = t.strip('()')
        if not _is_num(t) or RAT_RE.match(s):
            continue
        v = mpf(s)
        if v == 0 or v == mp.nint(v):
            continue                         # exact in f64: the fold adds no rounding
        toks2 = list(tokens)
        toks2[k] = mp.nstr(v * (1 + eps), 40)
        try:
            v2 = evaluate(toks2, env)
        except Exception:
            continue
        if isnan(v2) or isinf(v2):
            continue
        noise += abs(v2 - base)
    return noise


# -- F74: exact-rational folding + the adaptive exact-snap ------------------------------
#
# The pre-upgrade zero-snap tested maximal variable-free spans at HARDCODED dps 50/110.
# Two measured defects (extreme lane, 2026-08-09/10):
#   * a true nonzero difference whose resolution needs more digits than both rungs
#     (1e309 + (-1e309 - 2.55e38): 272 digits) read EXACTLY zero on both, so the judge
#     rewrote the input span to '0' and convicted the correct engine at any ambient
#     precision (row 13391 and the whole value-change family);
#   * only MAXIMAL spans were tested (row 636919, see _var_free_subtree_spans).
# The replacement: PURE-RATIONAL spans fold exactly in Fraction arithmetic first
# (unbounded integers: no rung can be fooled), and only transcendental-bearing spans
# go to a precision LADDER that escalates until the span resolves -- snapping to '0'
# or to a stable small integer (sinh(asinh 2) = 2: the mpf residue 2 +- 1e-49 read as
# a non-integer rootn index and fabricated nan -- the base table's GAP class).

FOLD_OPS = frozenset({'+', '-', '*', '/', 'neg', 'inv', 'abs', 'pow'})
FOLD_BITS_BUDGET = 15900   # ~2400 decimal digits per side of the emitted p/q token:
                           # bounded work, and safely under CPython's 4300-digit
                           # int<->str conversion limit (emission uses str(int)).


class _FoldRefused(Exception):
    """A span whose exact-rational fold is out of scope (non-rational leaf, pole,
    non-integer or over-budget exponent): left verbatim for the mpf/contract path."""


def _rat_leaf(t):
    """The exact Fraction a rational-literal token denotes, else None."""
    s = t.strip('()')
    if RAT_RE.match(s):
        p, q = s.split('/')
        return Fraction(int(p), int(q))
    if not _is_num(t):
        return None
    try:
        return Fraction(Decimal(s))          # exact for every decimal spelling
    except (InvalidOperation, ValueError):
        return None


def _fold_budget(f):
    if f.numerator.bit_length() + f.denominator.bit_length() > FOLD_BITS_BUDGET:
        raise _FoldRefused()
    return f


def _fold_exact_rational_spans(tokens):
    """Fold every variable-free PURE-RATIONAL subtree (>= 2 tokens) to its exact value
    token, in unbounded Fraction arithmetic. Contract-semantics extension points stay
    out of scope by construction -- division by an exact zero, non-integer or
    over-budget pow exponents, and any span touching pi/e/inf/nan refuse and are left
    verbatim for the mpf evaluator. Untouched leaves keep their original spelling."""
    pos = 0

    def emit(f):
        if f.denominator == 1:
            return str(f.numerator)
        return f'{f.numerator}/{f.denominator}'          # RAT_RE spelling

    def walk():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t in BAG_OPEN:
            marker, close = BAG_OPEN[t][:2]
            is_mul = t == '<mul>'
            parts, main, sec = [t], [], []
            cur, ok = main, True
            while pos < len(tokens) and tokens[pos] != close:
                if tokens[pos] == marker:
                    parts.append(marker)
                    cur = sec
                    pos += 1
                    continue
                sub, v = walk()
                parts.extend(sub)
                if v is None:
                    ok = False
                else:
                    cur.append(v)
            if pos < len(tokens):
                parts.append(close)
                pos += 1
            if not ok:
                return parts, None
            try:
                if is_mul:
                    acc = Fraction(1)
                    for v in main:
                        acc = _fold_budget(acc * v)
                    for v in sec:
                        acc = _fold_budget(acc / v)      # exact nonzero: == per-member inv
                else:
                    acc = Fraction(0)
                    for v in main:
                        acc = _fold_budget(acc + v)
                    for v in sec:
                        acc = _fold_budget(acc - v)
            except (_FoldRefused, ZeroDivisionError):
                return parts, None
            return [emit(acc)], acc
        if t in OPS:
            ar = OPS[t][0]
            parts, vals = [t], []
            for _ in range(ar):
                sub, v = walk()
                parts.extend(sub)
                vals.append(v)
            if t not in FOLD_OPS or any(v is None for v in vals):
                return parts, None
            try:
                if t == '+':
                    acc = _fold_budget(vals[0] + vals[1])
                elif t == '-':
                    acc = _fold_budget(vals[0] - vals[1])
                elif t == '*':
                    acc = _fold_budget(vals[0] * vals[1])
                elif t == '/':
                    acc = _fold_budget(vals[0] / vals[1])
                elif t == 'neg':
                    acc = -vals[0]
                elif t == 'abs':
                    acc = abs(vals[0])
                elif t == 'inv':
                    acc = _fold_budget(Fraction(1) / vals[0])
                else:                                    # pow
                    b = vals[1]
                    if b.denominator != 1 or abs(b.numerator) > 10000:
                        raise _FoldRefused()             # roots / unbounded: mpf path
                    e = b.numerator
                    base_bits = max(vals[0].numerator.bit_length(),
                                    vals[0].denominator.bit_length(), 1)
                    if abs(e) * base_bits > FOLD_BITS_BUDGET:
                        raise _FoldRefused()
                    acc = _fold_budget(vals[0] ** e)     # Fraction(0)**0 == 1: contract
            except (_FoldRefused, ZeroDivisionError):
                return parts, None
            return [emit(acc)], acc
        v = _rat_leaf(t)
        return [t], v                                    # leaves keep their spelling
    out, _v = walk()
    if pos != len(tokens):
        raise ValueError(f'trailing tokens: {tokens}')
    return out


def _pair_dps(tokens):
    """Working precision sufficient for this token list's literal exponent span.
    F74 (D4): at a FIXED dps 50, cos(1e-320) rounds to exactly 1 (the true value is
    1 - 5e-641), fabricating pow(1, -inf) = 1 against a correct +inf. The span
    hi..lo covers each literal's leading exponent down to its last written place;
    2x covers the squares that first-order compositions produce (cos ~ 1 - x^2/2),
    a flat guard covers residue bands. Small-literal pairs keep the ambient dps
    (the entire pre-F74 corpus population: byte-for-byte the old behavior)."""
    hi, lo = 0, 0
    for t in tokens:
        s = t.strip('()')
        if RAT_RE.match(s):
            p, q = s.split('/')
            top = len(p.lstrip('-')) - len(q) + 1
            hi = max(hi, top)
            lo = min(lo, top - (len(p.lstrip('-')) + len(q)) - 10)
            continue
        if not _is_num(t):
            continue
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):
            continue
        if d == 0:
            continue
        hi = max(hi, d.adjusted())
        lo = min(lo, d.as_tuple().exponent)
    extra = hi - lo
    if extra <= 20:
        return mp.dps
    return min(3000, 2 * extra + 70)


SNAP_DPS_CAP = 6000        # ladder ceiling: an all-rungs-exact-zero span is accepted as
                           # exact zero only after resolving nothing through ~6000 digits
                           # -- cancellations engineered past that need pow towers beyond
                           # the token grammar's reach. Short var-free spans only: the
                           # dps-6000 rungs are microseconds (no display conversions).


def _magnitude_resolved(v, dps) -> bool:
    """Is `v`'s own MAGNITUDE resolved at `dps` digits?

    An mpf carries `dps` significant digits of its decimal exponent as well as of its
    mantissa. When |log10|v|| exceeds 10**dps the exponent's last digits are themselves
    unresolved, so the value's magnitude is not pinned at this precision and any
    cross-rung magnitude comparison is meaningless. Exact zeros are resolved by
    definition (nothing to place). Fails CLOSED (returns False -> the caller refuses to
    snap) if the logarithm cannot be taken."""
    if v == 0:
        return True
    if isnan(v) or isinf(v):
        return False
    try:
        return abs(mp.log10(abs(v))) <= mpf(10) ** dps
    except Exception:
        return False


def _span_exact_token(toks):
    """The exact-value token this variable-free span provably denotes ('0' or a small
    integer), else None. A ladder of dps rungs starting at the working precision; at
    each pair of cleanly-evaluated rungs (lower value `a` at r_lo, higher `v` at r):

      * a > its rung floor and the offset to the nearest small integer SHRINKS by the
        rung ratio -> that integer is exact (a precision residue dies with dps; a true
        offset is rung-stable and refuses -- exp(-200) stays 1.4e-87 on every rung);
      * a sits BELOW its own rung floor (reads exactly n, offset 0): undecidable at
        this pair -- escalate so a true sub-floor offset gets RESOLVED and then fails
        the shrink test at the next pair; exact-at-every-rung through the cap is
        accepted as exact (a cancellation engineered past ~6000 digits is beyond the
        token grammar's reach);
      * nan/inf, values beyond small-integer scope, and rung-stable offsets: None.

    Rungs that refuse (Unresolved) are skipped, not fatal -- a pole-guarded rung at
    dps 50 may evaluate cleanly at 250. Pure-rational spans never reach this ladder
    (they fold exactly in Fraction first)."""
    prev = None
    r = mp.dps
    while True:
        try:
            with mp.workdps(r):
                v = evaluate(list(toks), {})
        except Exception:
            v = None                          # refused rung: skip, not fatal
        if v is not None and (isnan(v) or isinf(v)):
            return None                       # semantic specials: never snapped
        if v is not None and prev is not None:
            a, r_lo = prev
            with mp.workdps(r):
                n = mp.nint(v)
                # Scope: the integer must be FULLY RESOLVED at the lower rung (every
                # digit inside r_lo - 20 precision) -- that guard is what refuses the
                # provably-irrational-but-integer-reading trap (exp(2.703^5) ~ 6e63 at
                # dps 50 has unresolved trailing digits, and its nint differs between
                # rungs). 1e300 additionally bounds token emission (str(int)) well
                # under CPython's 4300-digit conversion limit; extreme rootn indices
                # ((A^A)^(1/A) = A ~ 1.2e27, row 690235) sit comfortably inside.
                if abs(n) > mpf('1e300') or abs(a) > mpf(10) ** (r_lo - 20):
                    return None
                da, db = abs(a - n), abs(v - n)
                if da > 0:
                    # MAGNITUDE-RESOLUTION guard. The shrink test below reads a large
                    # rung-to-rung ratio as "a precision residue dying with dps". That
                    # inference is only licensed when each rung RESOLVED the value's
                    # magnitude. A value whose own decimal exponent needs more digits
                    # than the rung carries is not resolved at all: exp(-1e309) renders
                    # as 10^(-4.34e308), and that exponent is correct to 50 digits at
                    # dps 50 and to 121 at dps 121 -- so the two rungs differ by a
                    # factor near 10^(1e258) and the ratio carries NO information about
                    # residue-vs-value. Unguarded, the test read that as a dying residue
                    # and snapped a provably POSITIVE value (exp > 0 everywhere) to '0';
                    # reachable through judge_pair with small literals
                    # (`exp(neg exp(1000)) -> 0` judged OK), and the source of the
                    # ambient-precision dependence in the exp-erasure guard. Refuse:
                    # unknowable at this precision, never a fabricated zero.
                    if not _magnitude_resolved(a, r_lo) or not _magnitude_resolved(v, r):
                        return None
                    if da < max(1, abs(n)) * mpf(10) ** (-(r_lo - 10)) \
                            and db <= da * mpf('1e-20'):
                        return '0' if n == 0 else str(int(n))
                    return None               # rung-stable offset: a true value
                if db == 0 and r >= SNAP_DPS_CAP:
                    return '0' if n == 0 else str(int(n))   # exact through the ladder
                # da == 0: undecidable at this pair (see docstring) -- escalate
        if v is not None:
            prev = (v, r)
        if r >= SNAP_DPS_CAP:
            return None
        r = min(int(r * 2.3) + 7, SNAP_DPS_CAP)


def _snap_literal_exacts(tokens):
    """Snap variable-free subtree spans that provably denote an exact zero or small
    integer to that literal, innermost first, to a fixpoint. Exact-rational spans are
    folded first in Fraction (see _fold_exact_rational_spans); the ladder then only
    faces transcendental-bearing spans (sin(np.pi) -> '0', sinh(asinh 2) -> '2'),
    whose precision residues would otherwise fabricate verdicts at extension points
    (0*inf, pow(nan, 0), rootn index integrality)."""
    out = _fold_exact_rational_spans(list(tokens))
    while True:
        for s, e in _var_free_subtree_spans(out):
            if e - s == 1:
                continue
            rep = _span_exact_token(out[s:e])
            if rep is not None:
                out[s:e] = [rep]
                break                          # token list moved: rescan
        else:
            return out


@_ambient_dps
def judge_pair(inp, out, rng, _refit_left=1, _slot_bound=False):
    """Judge one rewrite inp -> out under the contract. Returns (verdict, detail).

    `_refit_left`: on a violation while a fittable slot is bound, the bound c may be the
    artifact of a DEGENERATE first fit (pow(c, x) at x = 0 fits every c). One refit at the
    violating probe is allowed, re-judging from scratch with that c substituted; a second
    disagreement means no single constant exists -- a genuine violation.

    F74: exact-rational spans fold in Fraction first, and the whole judgment runs at a
    working precision adapted to the pair's own literal exponent span (never below the
    ambient mp.dps, so external escalation harnesses stay meaningful). Small-literal
    pairs -- the entire pre-F74 corpus population -- keep the ambient precision."""
    unk = unknown_tokens(list(inp) + list(out))
    if unk:
        # FAIL CLOSED on language gaps: a token the judge cannot read must surface, never
        # silently skip every probe and fall through to OK (the E1 blindness mechanism).
        return 'UNSCORED', 'unknown tokens: ' + ' '.join(unk)
    inp = _fold_exact_rational_spans(list(inp))
    out = _fold_exact_rational_spans(list(out))
    with mp.workdps(max(mp.dps, _pair_dps(list(inp) + list(out)))):
        return _judge_pair_at(inp, out, rng, _refit_left, _slot_bound)


def _judge_pair_at(inp, out, rng, _refit_left, _slot_bound):
    inp, out = _snap_literal_exacts(list(inp)), _snap_literal_exacts(list(out))
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
        # yields (env_builder, point_builder, is_atom, is_diag); is_diag marks the
        # correlated all-variables-EQUAL continuum draws, a codimension >= 1 subset
        # whenever more than one variable exists
        for b in sym:
            yield ({v: b for v in vs} if len(vs) <= 1 else None), b, True, False
        # independent + correlated draw grids (frozen rationals wrapped as builders)
        for i, p in enumerate(draws):
            if len(vs) <= 1:
                yield {v: (lambda q=p: q) for v in vs}, (lambda q=p: q), False, False
            else:
                e = {v: (lambda q=draws[(i + j * 7) % len(draws)]: q) for j, v in enumerate(vs)}
                yield e, (lambda q=p: q), False, False
                if i % 4 == 0:
                    yield {v: (lambda q=p: q) for v in vs}, (lambda q=p: q), False, True
        if len(vs) > 1:
            for b in sym:
                yield {v: b for v in vs}, b, True, False    # correlated atoms

    # Rule-sort slots (SORT_RE) are quantified REALS: their all-equal diagonal is in
    # scope and keeps measure authority. DATA variables (x-vars) get the contract's
    # null-set tolerance (R3): the correlated all-equal draws are a Lebesgue-null
    # subset, and counting their 10 copies in the measure denominator let any
    # variable-coincidence artifact score 10/50 = 20% > MEASURE_FRACTION -- fabricated
    # positive-measure verdicts (observed, 2026-07-29 corpus adjudication). The strict
    # value-change clause is untouched: it is pointwise (R2), so diagonal draws still
    # feed it.
    has_sort = any(SORT_RE.match(v) for v in vs)
    slot_val = None
    ext_draws = 0
    shrink_draws = 0
    inf_draws = 0
    tot_draws = 0
    judged = 0            # probes where BOTH sides evaluated (incl. nan results)
    for envb, pb, is_atom, is_diag in envs():
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
        judged += 1
        an, bn = isnan(a), isnan(b)
        null_probe = is_atom or (is_diag and not has_sort)
        if not null_probe:
            tot_draws += 1
        if an and bn:
            continue
        if an and not bn:
            if not null_probe:
                ext_draws += 1               # extension on the continuum: count the measure
            else:                            # atom or data-variable diagonal: null set
                key = 'ext-null' if is_atom else 'ext-diag-null'
                TOL_COUNTS[key] = TOL_COUNTS.get(key, 0) + 1
            continue
        if bn:
            # defined->undefined is judged by MEASURE over data. A null event (atom probe,
            # incl +-inf) is tolerated + tallied; continuum draws count.
            # (NB the slot-refit escape does not apply on the measure path -- the two-point
            # fit validation above already rejects degenerate constants.)
            if null_probe:
                key = 'shrink-null' if is_atom else 'shrink-diag-null'
                TOL_COUNTS[key] = TOL_COUNTS.get(key, 0) + 1
                continue
            shrink_draws += 1
            continue
        if isinf(a) or isinf(b):
            if (isinf(a) and isinf(b) and (a > 0) == (b > 0)):
                continue
            # an infinity-involved disagreement is an extension-layer artifact, judged by
            # MEASURE; null events tolerated (null-event doctrine). Both-sides-REAL
            # disagreements (the strict value-change clause) never reach here and stay strict.
            if null_probe:
                key = 'inf-null' if is_atom else 'inf-diag-null'
                TOL_COUNTS[key] = TOL_COUNTS.get(key, 0) + 1
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
            # SINGULAR-INPUT gate (the 9.8.3(a) boundary, second rung; 2026-08-11).
            # Clause (a)'s authority is "where mathematics has an answer". At a REAL
            # probe on the INPUT's singular manifold -- its evaluation crosses an
            # internal +-inf (1/0 under one-zero, log 0, atanh +-1) -- mathematics has
            # none: the real the input evaluates to is a convention-completion, and
            # two a.e.-identical spellings compose the conventions to DIFFERENT
            # completions (1/(0/t) = +inf but (1/0)*t = -inf for t < 0; atan renders
            # both REAL as -+pi/2).
            #
            # This is NOT fixable by choosing better conventions, which is why the
            # clause must be scoped instead. Under any extension that gives poles
            # +-inf values (the ratified one) the pole sign is spelling-dependent:
            # 1/(x1-x3) vs -1/(x3-x1) are a.e. identical and give +inf vs -inf, and
            # a bounded compressor above them (atan, tanh, exp+pow) turns that into a
            # REAL-vs-REAL disagreement -- inside clause (a)'s own jurisdiction. IEEE's
            # signed zero does NOT close it (x-x is +0 for every finite x, so the zero
            # cannot record which spelling formed it); only nan-poisoning poles would,
            # and that is excluded by the ratified limit-completion and the D1 folds.
            # Verified against the engine's own deployed realizations: both spellings
            # agree there, so the disagreement is an artifact of the abstraction, not
            # of the rewrite.
            #
            # Routed exactly like the env-inf rung above (inf-input-null): null probes
            # tolerated + tallied, continuum draws counted into the INF-CHANGE measure,
            # so a POSITIVE-MEASURE singular disagreement still kills. An OUTPUT-side
            # singularity with a CLEAN input keeps strict authority -- there the
            # input's value IS mathematics' answer.
            #
            # ORDER: this fires LAST, on the path that would otherwise CONVICT -- after
            # the precision and f64-fold classifiers below. A disagreement they can
            # explain is rounding, not a completion clash, and routing it into the
            # INF-CHANGE measure would let precision noise accumulate fake measure
            # against a sound rewrite.
            #
            # PRECISION-STABILITY gate: a finite mismatch may be working-dps rounding
            # amplified by a singularity (acos near 1 turns a 1e-50 roundtrip residue into
            # 1e-25 -- observed). Re-evaluate BOTH sides at the SAME point at ~2.4x the
            # working precision (F74: was a flat 120, which sat BELOW the working dps of
            # extreme-literal pairs and could not clear anything there): a true identity
            # agrees at any fixed point, so a shrinking difference is rounding; a stable
            # difference is a real violation.
            try:
                with mp.workdps(max(240, mp.dps * 24 // 10)):
                    env2 = {k: f() for k, f in envb.items()}
                    a2 = evaluate(list(inp), env2)
                    b2 = evaluate(b_tokens, env2)
                if isnan(a2) or isnan(b2) or isinf(a2) or isinf(b2):
                    raise ValueError
                if abs(a2 - b2) <= tol * max(1, abs(a2), abs(b2)):
                    continue                              # rounding artifact: cleared
            except Unresolved:
                continue                     # the recheck landed in a residue/pole band:
                                             # unknowable at higher precision -- skip,
                                             # never convict on a degenerate recheck
            except Exception:
                pass
            if n_slots == 0 and not _slot_bound:
                # F64_FOLD -- the engine's compile-time constant folds are DOUBLES
                # (asin -1 -> the f64 pi/2 literal, rel ~1.2e-17; see _slot_token),
                # which this mp instrument resolves as a stable value change. Two
                # certificates, either suffices; a named class, not a conviction --
                # and rules keep strict authority (slot paths and bound-slot rejudges
                # still return VIOLATION):
                #  * magnitude: the stable offset sits below f64 fold resolution at
                #    the observed values -- nothing an f64 consumer can see;
                #  * sensitivity: the offset sits within the output's own fold-noise
                #    floor (_fold_noise_floor) -- fold-attributable even when a
                #    downstream condition number amplifies it past any fixed band.
                rel = abs(a - b) / max(1, abs(a), abs(b))
                if rel <= mpf('1e-15'):
                    return 'F64_FOLD', (f'f64 constant fold at {_fmt(p)}: '
                                        f'{_fmt(a)} -> {_fmt(b)} (rel {_fmt(rel)})')
                try:
                    if abs(a - b) <= _fold_noise_floor(b_tokens, env, b):
                        return 'F64_FOLD', (f'f64 constant fold at {_fmt(p)}: '
                                            f'{_fmt(a)} -> {_fmt(b)} (rel {_fmt(rel)}, '
                                            f'within fold-noise floor)')
                except Exception:
                    pass
            # SINGULAR-INPUT gate (see the block above for the full derivation): the
            # last classifier before a conviction. The disagreement survived the
            # precision and fold explanations, so it is stable -- but if the INPUT is
            # singular at this probe it is a completion clash, not a value change.
            if _input_singular(inp, env):
                if null_probe:
                    TOL_COUNTS['singular-real-null'] = \
                        TOL_COUNTS.get('singular-real-null', 0) + 1
                else:
                    inf_draws += 1
                continue
            return _viol(env, f'value change at {_fmt(p)}: {_fmt(a)} -> {_fmt(b)}')
    if tot_draws:
        for cnt, label in ((ext_draws, 'EXTENSION: undefined->defined'),
                           (shrink_draws, 'SHRINK: defined->undefined'),
                           (inf_draws, 'INF-CHANGE: extension values')):
            if cnt / tot_draws > MEASURE_FRACTION:
                return 'VIOLATION', (f'positive-measure {label} on {cnt}/{tot_draws} '
                                     f'continuum draws (measure clause)')
    if judged == 0:
        # every probe was skipped: the judge never actually compared the two sides.
        # FAIL CLOSED -- 'OK' here is exactly the E1 blindness (a pair whose output threw
        # at every probe silently passed).
        return 'UNSCORED', 'no evaluable probes'
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
    try:
        return mp.nstr(v, 8)
    except Exception:
        # CPython's 4300-digit int<->str limit trips nstr on very-high-dps mantissas;
        # a display helper must never take the judge down with it.
        _sign, _man, e, bc = v._mpf_
        return f'~2^{e + bc}'


# ---------------------------------------------------------------------------------------
# Corpus, engines, the sweep
# ---------------------------------------------------------------------------------------

ADVERSARIAL = [
    # Spelled in the LIVE 0.12 vocabulary (23 operators + rootn). The pre-fix list spoke
    # the deleted hyper-op vocabulary (mult4, pow1_2, ...), which the 0.12 parser rejects:
    # 7 rows raised ValueError on EVERY sweep, polluting the report with self-inflicted
    # '<RAISED>' violations while the shapes they were built to exercise went untested.
    ['+', 'x0', 'pow', 'sin', 'x0', 'float("inf")'],
    ['pow', 'sin', 'x0', 'float("inf")'],
    ['*', '4', 'pow', 'x0', 'float("inf")'],
    ['-', 'exp', 'x0', 'exp', 'x0'],
    ['-', 'log', 'x0', 'log', 'x0'],
    ['inv', '*', '0', 'x0'],
    ['/', 'log', 'x0', 'log', 'x0'],
    ['atanh', 'tanh', 'sinh', 'x0'],
    ['*', 'x0', 'pow', 'cos', 'x1', 'float("inf")'],
    ['+', 'pow', 'tanh', 'x0', 'float("inf")', 'x1'],
    ['pow', '/', '7.0', '5', 'float("nan")'],
    ['*', 'sin', 'np.pi', 'x0'],
    ['abs', 'pow', 'x0', '*', '3', '1'],
    ['tan', '+', 'x0', 'np.pi'],
    ['inv', '+', '0', 'neg', 'pow', 'x0', '4'],
    ['pow', 'float("-inf")', 'rootn', '3.0', '3'],
    # triggers for the measure-clause poison (even-root round-trip via rootn)
    ['pow', 'rootn', 'x0', '2', '2'],
    ['*', '2', 'pow', 'rootn', 'x1', '2', '2'],
    # trigger for the bag-output poison (a product over a sum: the rewrite's output
    # keeps mul/add structure, so the engine emits it bag-tagged)
    ['*', 'x0', '+', 'x1', 'x2'],
    # PRE-MASKED shapes (realization fidelity: deployed flash-ansr traffic arrives with
    # literals already masked to <constant> -- 0.12 masking is explicit, upstream of
    # simplify; also guarantees the masked-scale poisons a trigger on ANY ruleset -- on
    # a small ruleset the poison had no fold-created triggers and the self-test failed
    # closed, observed in practice)
    ['*', '<constant>', 'x0'],
    ['+', '*', '<constant>', 'x1', 'x0'],
    ['sin', '*', '<constant>', 'x0'],
    ['*', '<constant>', 'pow', 'x0', 'float("inf")'],
]

# Default leaf vocabulary used to sample the random half of the corpus.
DEFAULT_LEAVES = ['x0', 'x1', '0', '1', 'np.pi', '2.0', '(-0.5)', '3.0']

# Known-unsound rules for the poison self-test, spelled in the LIVE 0.12 vocabulary.
# Each must be DEPLOYABLE end-to-end: the AC translation must accept it (reduction-ordered,
# no <constant>-introducing RHS -- the ratified translation gate refuses those) and a
# trigger shape must exist in ADVERSARIAL. Catch requires ATTRIBUTION: a flagged input
# whose output CHANGED versus the unpoisoned engine.
POISON = [
    # value change at a spike: pow(sin x, inf) -> 0 is wrong on the sin(x) = +-1 lattice
    (('pow', 'sin', '?0', 'float("inf")'), ('0',)),
    # masked-scale: drops the coefficient of pre-masked traffic
    (('*', '<constant>', '?0'), ('?0',)),
    # structural coefficient drop (fires on the pre-masked pow shape)
    (('*', '<constant>', 'pow', '?0', 'float("inf")'), ('pow', '?0', 'float("inf")')),
    # measure-clause poison: the even-root round-trip pow(rootn(x,2),2) -> x extends the
    # x < 0 half-line -- must be caught via the positive-measure counters
    (('pow', 'rootn', '?0', '2', '2'), ('?0',)),
    # BAG-OUTPUT poison: x*(y+z) -> y+z drops a factor and its output RETAINS additive
    # structure, so the engine emits it bag-tagged (`<add> y z </add>`). This is the
    # output class the pre-fix judge silently OK'd (E1); it guards the bag evaluator.
    (('*', '?0', '+', '?1', '?2'), ('+', '?1', '?2')),
]


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
    # Timeout protection is SIGALRM-based and therefore main-thread-only. Install for
    # the duration of the sweep and RESTORE on the way out (hardening H-010,
    # 2026-08-03: this used to clobber the process handler permanently, and to crash
    # any off-main-thread caller -- e.g. a worker-thread mine's finalize gate, whose
    # sweep now runs WITHOUT per-expression timeouts; the mpmath hang class is rare
    # and a wedged worker mine is recoverable, an always-crashing one is not).
    import signal
    import threading
    on_main = threading.current_thread() is threading.main_thread()
    if on_main:
        prev_handler = signal.signal(signal.SIGALRM, _alarm)
    else:
        print(f'  [{tag}] off-main-thread sweep: judge timeout protection unavailable '
              f'(SIGALRM is main-thread-only)', flush=True)
    try:
        return _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag,
                            judge_timeout_s if on_main else 0)
    finally:
        if on_main:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)


def _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag,
                 judge_timeout_s):
    import signal
    eng = build_engine(rules, config_path)
    viol, native, unscored, changed, timeouts, f64_folds = [], [], 0, 0, 0, 0
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
        elif v == 'F64_FOLD':
            # the named f64 constant-fold class (see judge_pair): tolerated, counted.
            # These rows previously surfaced as native-line entries via the rules-empty
            # attribution below, so the native count drops by exactly this population.
            f64_folds += 1
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
    return viol, native, unscored, changed, f64_folds


@_ambient_dps
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
        v, *_ = sweep(rules + [p], corpus[:600], np.random.default_rng(seed),
                      baseline, config_path, tag="poison", judge_timeout_s=judge_timeout_s)
        attributed = []
        for expr, out, detail in v:
            try:
                clean_out = list(clean_eng.simplify(list(expr)))
            except Exception:
                clean_out = ['<RAISED>']
            # attribution = the poisoned engine BEHAVES DIFFERENTLY from the clean one on
            # this input. Both raising is the SAME behavior: the pre-fix version counted
            # a clean-side raise as attribution unconditionally, which let corpus rows
            # that crash EVERY engine launder uncatchable poisons into 'CAUGHT'.
            if clean_out != list(out):
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


class MonitorLivenessError(RuntimeError):
    """Raised by :func:`monitor` when the main sweep rewrites NOTHING on a non-trivial corpus.

    A sweep that judges only rewrites is vacuous when no rewrites happen: `changed == 0`
    over hundreds of sampled expressions means the engine under test is not applying
    anything (broken asset load, dead core plumbing) -- an instrument failure that must
    not be reported as 'clean' (0 violations of 0 rewrites)."""


@_ambient_dps
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

    viol, native, unscored, changed, f64_folds = sweep(
        rules, corpus, rng, baseline, config_path,
        tag="main", judge_timeout_s=judge_timeout_s)
    if changed == 0 and len(corpus) >= 100:
        raise MonitorLivenessError(
            f'main sweep rewrote 0 / {len(corpus)} expressions: the engine under test '
            f'applied nothing (broken asset load or dead core plumbing) -- a sweep that '
            f'judges only rewrites is vacuous here and must not report clean')
    print(f'\n=== INDEPENDENT MONITOR  {label} ===')
    print(f'  rewritten     {changed:7,} / {len(corpus):,}')
    print(f'  VIOLATIONS    {len(viol):7,}  (artifact-attributed)')
    print(f'  native-line   {len(native):7,}  (present with rules=[]; pre-existing engine behavior)')
    print(f'  unscored      {unscored:7,}  (multi-slot / unknown-token / no-evaluable-probe)')
    print(f'  f64-folds     {f64_folds:7,}  (correctly-rounded f64 constant folds; tolerated)')
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
        'f64_folds': f64_folds,
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
