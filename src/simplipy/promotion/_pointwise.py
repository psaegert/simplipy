# mypy: ignore-errors
"""Pointwise certifier for the `_`-sort (subtree-sort) promotion bar.

Soundness bar: a rule whose slots are subtree-sort is sound iff S(v) = T(v) for
EVERY consistent valuation v of its wildcards over R u {+-0, +-inf, nan}, where
nan == nan counts as equal, in BOTH directions (a nan hole may not be filled; a
value may not be lost or changed). There is no measure tolerance in the subtree
sort: a constant subtree is a Dirac, so a single disagreeing valuation refutes.

The atoms are TOTAL singular bindings (0, 1, -1, +-inf, nan, and finite draws):
they surface both directions of failure -- nan -> value (a filled hole) and
value -> wrong value. DISTINCT wildcards receive the full product of values;
repeated wildcard names share a value (consistent valuations: the diagonal).

Verdicts (per rule):
  PROMOTE   -- all valuations agree; ships as `_`-sort (subtree sort).
  DEMOTE    -- a killing valuation exists (recorded); ships as `?`-sort
               (variable-leaf sort), losing nothing it was ever certified for.
  UNDECIDED -- only near-tolerance finite disagreements remain; f64 rounding may
               not convict, so these are deferred to an exact arbiter.

Scope: const-free wildcard rules. `<constant>`-bearing rules need a
forall-exists witness map and are handled by a separate promotion stage.
Acceptance is sampling-based: a PROMOTE is falsification-survival at these
valuations, with the false-promote rate bounded by the valuation count.
"""
import json

import numpy as np

from ._f64_eval import evaluate, ARITY, LITERALS, _num

# THE ATOM LATTICE -- the shared list every certifier imports. Killing valuations
# for numeric rules cluster on the integers and on rule-correlated points, so the
# lattice must include them explicitly:
#   * integers >= 2 and half-integers expose pow's odd/even-parity artifacts;
#   * +-pi, +-e, +-pi/2, +-2pi, +-pi/4 (f64 renderings -- used only as refutation
#     probes, so exact transcendence is not needed) expose trig/log coincidences.
# Capped at |10|: past ~20, tanh/exp towers saturate f64 EXACTLY (tanh(20) == 1.0)
# and a true identity would inf-mismatch at the atom -- a wrong kill this atom
# class is never re-judged for.
#
# Quantifiers range over the REALS, so transcendental probe points (x = pi/2 where
# sin = 1, x = pi/4 where tan = 1, ...) are LEGITIMATE: a rule wrong at x = pi/2 is
# wrong at a real point even though f64 never lands on it. The high-precision arm
# binds these tokens SYMBOLICALLY; the f64 arm binds their f64 renderings, whose
# saturation (sin(f64 pi/2) == 1.0) usefully MIMICS the real spike at the atom.
# Known incompleteness: coincidences that MOVE with a fitted constant
# (pow(sin(c*x), inf) at x = pi/(2c)) cannot be covered by any fixed lattice; a
# separate spike-flattening refusal covers that family structurally.
#
# '-0.0' is intentionally absent: it denotes the same real value as '0.0', so it
# probes nothing, and its f64 rendering carries non-normative sign effects.
ATOMS = ['0.0', '1.0', '(-1.0)', 'float("inf")', 'float("-inf")', 'float("nan")',
         '2.0', '(-2.0)', '3.0', '(-3.0)', '5.0', '(-5.0)', '10.0', '(-10.0)',
         '0.5', '(-0.5)', '1.5', '(-1.5)', '2.5', '(-2.5)',
         '3.141592653589793', '(-3.141592653589793)',
         '2.718281828459045', '(-2.718281828459045)',
         '1.5707963267948966', '(-1.5707963267948966)',
         '6.283185307179586', '(-6.283185307179586)',
         '0.7853981633974483', '(-0.7853981633974483)']
N_RAND = 24            # random finite draws appended per wildcard (mixture, seeded)
MAX_PRODUCT = 32768    # cap on the valuation product (30 atoms: full product through k=3)
SEED = 20260717


def wildcards(tokens):
    """DISTINCT slot names, all three sorts (`_` subtree, `!` certified subtree, `?` leaf)."""
    seen = []
    for t in tokens:
        if t.startswith(('_', '?', '!')) and t not in seen:
            seen.append(t)
    return seen


def is_const_free(tokens):
    return '<constant>' not in tokens


def rand_pool(rng):
    vals = list(np.round(rng.normal(0, 2, N_RAND // 2), 6)) + \
           list(np.round(rng.uniform(-40, 40, N_RAND - N_RAND // 2), 6))
    return [repr(float(v)) if v >= 0 else f'({float(v)!r})' for v in vals]


MAX_CORRELATED = 512   # cap on the per-rule correlated-atom product

# OP-EMBEDDED constants: the op NAMES carry the numbers that make their fixed
# points special -- e.g. `pow float("-inf") div4 _0 -> pow float("inf") _0` is
# killed exactly at _0 = 4 (pow(-inf, 1) = -inf vs pow(inf, 4) = +inf), a value
# no literal in the rule spells -- so those embedded constants must be probed too.
OP_CONSTANTS = {
    'mult2': 2.0, 'mult3': 3.0, 'mult4': 4.0, 'mult5': 5.0,
    'div2': 2.0, 'div3': 3.0, 'div4': 4.0, 'div5': 5.0,
    'pow2': 2.0, 'pow3': 3.0, 'pow4': 4.0, 'pow5': 5.0,
    'pow1_2': 2.0, 'pow1_3': 3.0, 'pow1_4': 4.0, 'pow1_5': 5.0,
}


def correlated_valuations(lhs, rhs, ws):
    """Rule-correlated atom valuations: killing points of a rule with numeric
    literals are often IMAGES of its own constants. For each finite nonzero literal
    c in the rule -- and each constant EMBEDDED in an op name (mult3 -> 3) -- probe
    {c, -c, 1/c, -1/c}; full product over the slots, capped."""
    vals = set()
    for t in list(lhs) + list(rhs):
        v = LITERALS.get(t, _num(t))
        if v is None:
            v = OP_CONSTANTS.get(t)
        if v is not None and np.isfinite(v) and v != 0.0:
            for u in (v, -v, 1.0 / v, -1.0 / v):
                if np.isfinite(u) and abs(u) <= 1e6:
                    vals.add(round(float(u), 12))
    if not vals or not ws:
        return []
    pool = [repr(v) if v >= 0 else f'({v!r})' for v in sorted(vals)]
    k = len(ws)
    if len(pool) ** k <= MAX_CORRELATED:
        idx = np.stack(np.meshgrid(*[np.arange(len(pool))] * k, indexing='ij'), -1).reshape(-1, k)
    else:
        idx = np.random.default_rng(SEED).integers(0, len(pool), size=(MAX_CORRELATED, k))
    return [{w: pool[j] for w, j in zip(ws, row)} for row in idx]


def substitute(tokens, wmap):
    return [wmap.get(t, t) for t in tokens]


def _literal_spans(tokens):
    """(start, end) spans of MAXIMAL subtrees containing only operators/literals/numerics
    (no wildcards, no variables, no `<constant>`)."""
    n = len(tokens)
    end = [0] * n

    def walk(i):
        t = tokens[i]
        j = i + 1
        for _ in range(ARITY.get(t, 0)):
            j = walk(j)
        end[i] = j
        return j
    walk(0)

    def is_lit(i):
        return all(tokens[k] in ARITY or tokens[k] in LITERALS or _num(tokens[k]) is not None
                   for k in range(i, end[i]))
    spans, i = [], 0
    while i < n:
        if is_lit(i):
            spans.append((i, end[i]))
            i = end[i]
        else:
            i += 1
    return spans


_PREFOLD_CACHE = {}


def prefold(tokens):
    """ZERO-SNAP: replace literal-only subtrees that denote EXACT zero by '0'.

    Transcendental literals denote their real value: `sin(np.pi)` IS zero, but f64
    renders it as 1.2e-16 -- harmless in the finite rel-band, but WRONG at the inf
    atoms (1.2e-16 * inf = inf where 0 * inf must be nan, a spurious extension). The
    sound `* sin np.pi ?0 -> 0` family would be killed at the inf atom without this
    snap.

    The snap criterion is a precision-stability gate, not a bare threshold: evaluate
    the subtree at dps 60 AND 120. An exact zero's rounding residue SHRINKS with
    precision (~10^-dps), while a legitimately tiny value (exp(-exp(pow(e, e))) ~
    10^-1.6M) is precision-STABLE and must NOT be snapped. Zeros are the only snapped
    class: finite nonzero coincidences are already protected by the rel-band, and
    only zero flips an algebraic class (0*inf, 0/0)."""
    key = tuple(tokens)
    if key in _PREFOLD_CACHE:
        return _PREFOLD_CACHE[key]
    out = list(tokens)
    spans = [s for s in _literal_spans(out) if s[1] - s[0] > 1]
    if spans:
        from mpmath import mp, mpf, isnan as mp_isnan, isinf as mp_isinf
        from ._hp_equiv import evaluate as hp_eval
        for s, e in reversed(spans):
            sub = out[s:e]
            try:
                with mp.workdps(60):
                    a = hp_eval(list(sub), {}, [])
                with mp.workdps(120):
                    b = hp_eval(list(sub), {}, [])
            except Exception:
                continue
            if mp_isnan(a) or mp_isnan(b) or mp_isinf(a) or mp_isinf(b):
                continue
            exact_zero = (a == 0 and b == 0) or \
                         (abs(a) < mpf('1e-40') and abs(b) <= abs(a) * mpf('1e-20'))
            if exact_zero:
                out[s:e] = ['0']
    _PREFOLD_CACHE[key] = out
    return out


def eval_point(tokens):
    """Evaluate a GROUND prefix expression (no variables) to one f64."""
    env = {'__dummy__': np.zeros(1)}
    return float(np.asarray(evaluate(list(tokens), env)).reshape(-1)[0])


def judge(lhs, rhs, valuations):
    """PROMOTE / DEMOTE(+witness) / UNDECIDED / EVAL-ERR over the given valuations.
    Rules are zero-snap prefolded first (exact-zero literal subtrees -> '0'), so the
    f64 evaluation below judges the intended denotation, not f64's rendering of it.
    Rule-correlated atom valuations are appended to whatever grid the caller
    supplies."""
    lhs, rhs = prefold(lhs), prefold(rhs)
    valuations = list(valuations) + correlated_valuations(lhs, rhs, wildcards(list(lhs) + list(rhs)))
    near = 0
    for wmap in valuations:
        try:
            a = eval_point(substitute(lhs, wmap))
            b = eval_point(substitute(rhs, wmap))
        except Exception as ex:
            return 'EVAL-ERR', f'{type(ex).__name__}: {ex}'
        an, bn = np.isnan(a), np.isnan(b)
        if an and bn:
            continue
        if an != bn:
            return 'DEMOTE', (dict(wmap), a, b)          # hole filled or value lost: exact kill
        if np.isinf(a) or np.isinf(b):
            if a == b:
                continue
            return 'DEMOTE', (dict(wmap), a, b)          # inf sign/finiteness mismatch: exact kill
        # ONE ZERO: there is a single zero and 1/0 = +inf; the sign of an f64 zero
        # is measurement rendering, never a value difference. No sign tie-break is
        # applied to a `-> 0` collapse -- float == already treats -0.0 == +0.0.
        rel = abs(a - b) / max(1.0, abs(a), abs(b))
        if rel <= 1e-9:
            continue
        if rel > 1e-6:
            return 'DEMOTE', (dict(wmap), a, b)          # gross finite disagreement
        near += 1                                        # rounding band: not convicting evidence (f64 lacks authority here)
    return ('UNDECIDED', near) if near else ('PROMOTE', None)


def valuations_for(ws, rng):
    pool = ATOMS + rand_pool(rng)
    k = len(ws)
    total = len(pool) ** k
    if total <= MAX_PRODUCT:
        idx = np.stack(np.meshgrid(*[np.arange(len(pool))] * k, indexing='ij'), -1).reshape(-1, k)
    else:
        na = len(ATOMS) ** k
        if na <= MAX_PRODUCT:
            # full ATOM product (the exact kills live there) + random completions
            ai = np.stack(np.meshgrid(*[np.arange(len(ATOMS))] * k, indexing='ij'), -1).reshape(-1, k)
            ri = rng.integers(0, len(pool), size=(MAX_PRODUCT - na, k))
            idx = np.vstack([ai, ri])
        else:
            # k >= 4: even the full ATOM product overflows MAX_PRODUCT -- SAMPLED
            # atom coverage (seeded), a stated bound, not a silent truncation.
            idx = rng.integers(0, len(ATOMS), size=(MAX_PRODUCT, k))
    return [{w: pool[j] for w, j in zip(ws, row)} for row in idx]


def load_rules(path):
    raw = json.load(open(path))
    pairs = raw.items() if isinstance(raw, dict) else ((r[0], r[1]) for r in raw)
    return [(tuple(k.split()) if isinstance(k, str) else tuple(k),
             tuple(v.split()) if isinstance(v, str) else tuple(v)) for k, v in pairs]
