# mypy: ignore-errors
"""Const-bearing witness-map certifier for wildcard rewrite rules.

Certifies a rewrite ``S -> T`` whose subtrees carry ``<constant>`` slots. The soundness
question, per rule, is::

    for all c_s  exists c_t  for all v:  S[v]_{c_s} = T[v]_{c_t}

pointwise over R u {+-0, +-1, +-inf, nan} (nan = nan counts as equal). The target
constants c_t are fixed per application: c_t may depend on the source constants c_s but
never on the wildcard valuation v.

MECHANISM:
  1. Draw c_s vectors adversarially: integer atoms first ({-2,-1,0,1,2,3}), because the
     singular behaviour that breaks a candidate identity concentrates on the integers,
     then half-integers and continuous draws.
  2. Find the existential witness c_t: a STRUCTURAL MENU (c1*c2, c1+c2, +-sqrt|c1|, 1/c1,
     ...), exact for the constant-merge/absorb family that dominates recall, then a
     bounded numeric least-squares fallback fitted on finite wildcard draws.
  3. Hold the INSTANTIATED pair (constants bound, wildcards free) to the SAME pointwise
     atom bar as the const-free certifier (``_pointwise.judge``).

VERDICTS:
  PROMOTE    -- every sampled c_s has a witness passing all valuations (sampling-based,
                so acceptance is falsification-survival at these draws, with the
                false-promote rate bounded by the sample count).
  DEMOTE     -- some c_s where a witness reproduces the finite behaviour but dies at an
                atom, or every tried witness dies; recorded with the killing
                (c_s, c_t, valuation).
  NO-WITNESS -- no candidate even reproduces the finite behaviour (solver failure or a
                dead source); fail-safe.

DEMOTE and NO-WITNESS both cost only composite-binding recall (the rule ships in its
weaker sort, exactly what its certification established). PROMOTE is the only verdict
that changes deployment.
"""
import numpy as np

from ._pointwise import judge, valuations_for, wildcards
from ._f64_eval import evaluate

SEED = 20260717
# The constant axis gets atom probes too -- appended at the END so the `[:5]` product
# slice below (used for the two-constant case) keeps its original coverage. Constants
# range over the reals, so the transcendental spike points (pi/2, pi, e) join as f64
# renderings; numpy's saturation (sin(f64 pi/2) == 1.0) usefully mimics the real spike
# at the atom.
CS_ATOMS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.5, -0.5, -3.0, 5.0, -5.0,
            1.5707963267948966, -1.5707963267948966, 3.141592653589793, 2.718281828459045]
N_CS_RAND = 4
N_V_FIT = 48          # finite wildcard draws for witness fitting/screening
FIT_TOL = 1e-7        # a witness must reproduce finite behaviour to this rel tol
MAX_SOLVER_STARTS = 6


def bind(tokens, consts):
    out, k = [], 0
    for t in tokens:
        if t == '<constant>':
            v = consts[k]; k += 1
            out.append(repr(float(v)) if v >= 0 else f'({float(v)!r})')
        else:
            out.append(t)
    return out


def n_consts(tokens):
    return sum(1 for t in tokens if t == '<constant>')


def eval_on(tokens, ws, V):
    """Evaluate tokens (constants already bound) on wildcard-value matrix V (n, len(ws))."""
    env = {w: V[:, i] for i, w in enumerate(ws)} if ws else {'__d__': np.zeros(len(V))}
    with np.errstate(all='ignore'):
        return np.broadcast_to(np.asarray(evaluate(list(tokens), env), np.float64), (len(V),)).copy()


def finite_match(y_src, y_tgt):
    """Does the target reproduce the source's FINITE behaviour (and its nan pattern) on the fit
    draws? Necessary screen before the atom bar; nan-pattern mismatch on finite draws is already
    a pointwise kill."""
    sn, tn = np.isnan(y_src), np.isnan(y_tgt)
    if (sn != tn).any():
        return False
    both = ~sn
    if not both.any():
        return True                     # nan-everywhere on finite draws: pattern agrees
    a, b = y_src[both], y_tgt[both]
    fin = np.isfinite(a) & np.isfinite(b)
    if (np.isinf(a) != np.isinf(b)).any() or (np.isinf(a) & (a != b))[fin == False].any():  # noqa: E712
        return False
    rel = np.abs(a[fin] - b[fin]) / np.maximum(1.0, np.maximum(np.abs(a[fin]), np.abs(b[fin])))
    return bool((rel <= FIT_TOL).all())


def witness_menu(cs, nt):
    """Structural candidates for c_t given c_s. Exact for the merge/absorb families."""
    cs = list(cs)
    singles = set()
    for c in cs:
        singles.update([c, -c])
        if c != 0:
            singles.update([1.0 / c])
        if c >= 0:
            singles.add(float(np.sqrt(c)))
        singles.update([float(np.sqrt(abs(c))), -float(np.sqrt(abs(c))), c * c])
    if len(cs) >= 2:
        a, b = cs[0], cs[1]
        singles.update([a * b, a + b, a - b, b - a])
        if b != 0:
            singles.add(a / b)
        if a != 0:
            singles.add(b / a)
    pool = sorted({float(v) for v in singles if np.isfinite(v)})
    if nt == 1:
        return [[v] for v in pool]
    if nt == 2:
        base = [[a, b] for a in pool[:8] for b in pool[:8]]
        return base[:64]
    return [list(cs[:nt]) + [1.0] * max(0, nt - len(cs))]


def solve_witness(lhs_b, rhs, ws, nt, y_src, V, rng):
    """Bounded least-squares fallback: fit c_t so T reproduces y_src on the finite draws."""
    try:
        from scipy.optimize import least_squares
    except ImportError as ex:
        # C23c-loud (audit N3, G2 gate): this `return []` used to be SILENT, and
        # scipy's absence then changed the mined artifact with no record and no
        # error -- a certifiable rule read NO-WITNESS instead of PROMOTE. The
        # mine's numeric stack is part of the artifact's identity (the sidecar's
        # `environment` block records it, R5/C23c-prov); a missing member is a
        # hard failure, never a silent degradation.
        raise RuntimeError(
            'simplipy.promotion requires scipy for witness solving; refusing to '
            'silently mine a different artifact without it (C23c-loud)') from ex
    defined = ~np.isnan(y_src) & np.isfinite(y_src)
    if defined.sum() < max(4, nt + 2):
        return []
    Vd, yd = V[defined], y_src[defined]

    def resid(theta):
        y = eval_on(bind(rhs, theta), ws, Vd)
        bad = ~np.isfinite(y)
        y = np.where(bad, 1e12, y)
        return (y - yd) / np.maximum(1.0, np.abs(yd))

    out = []
    for _ in range(MAX_SOLVER_STARTS):
        x0 = rng.normal(0, 2, nt)
        try:
            r = least_squares(resid, x0, method='lm', max_nfev=200)
        except Exception:
            continue
        if r.success and np.max(np.abs(r.fun)) < FIT_TOL:
            out.append([float(v) for v in r.x])
    return out


def certify_rule(lhs, rhs, rng, judge_fn=None, vals_override=None):
    """The `forall c_s exists c_t` certifier. `judge_fn` and `vals_override` let a caller
    hold the instantiated pairs to a different bar (e.g. an atom-only, extension-tolerant
    valuation set) instead of the default `_pointwise.judge` bar -- the witness machinery
    is identical."""
    if judge_fn is None:
        judge_fn = judge
    ws = wildcards(list(lhs) + list(rhs))
    ns, nt = n_consts(lhs), n_consts(rhs)
    cs_draws = [[c] * ns if ns == 1 else None for c in CS_ATOMS]
    cs_list = ([[c] for c in CS_ATOMS] if ns == 1 else
               [[a, b] for a in CS_ATOMS[:5] for b in (-2.0, 1.0, 2.0)] if ns == 2 else
               [list(rng.normal(0, 2, ns)) for _ in range(8)])
    cs_list = cs_list + [list(np.round(rng.normal(0, 3, ns), 4)) for _ in range(N_CS_RAND)]
    V = np.column_stack([np.concatenate([rng.normal(0, 2, N_V_FIT // 2),
                                         rng.uniform(-30, 30, N_V_FIT - N_V_FIT // 2)])
                         for _ in ws]) if ws else np.zeros((N_V_FIT, 1))
    vals = vals_override if vals_override is not None else valuations_for(ws, rng)
    n_no_witness = 0
    for cs in cs_list:
        lhs_b = bind(lhs, cs)
        try:
            y_src = eval_on(lhs_b, ws, V)
        except Exception as ex:
            return 'EVAL-ERR', f'{type(ex).__name__}: {ex}'
        found, atom_kill = None, None
        cands = witness_menu(cs, nt) if nt else [[]]
        seen_finite_match = False
        for ct in cands:
            try:
                y_t = eval_on(bind(rhs, ct), ws, V)
            except Exception:
                continue
            if not finite_match(y_src, y_t):
                continue
            seen_finite_match = True
            v, info = judge_fn(bind(lhs, cs), bind(rhs, ct), vals)
            if v in ('PROMOTE', 'PASS'):
                found = ct
                break
            atom_kill = (cs, ct, info)
        if found is None and not seen_finite_match and nt:
            for ct in solve_witness(lhs_b, rhs, ws, nt, y_src, V, rng):
                y_t = eval_on(bind(rhs, ct), ws, V)
                if not finite_match(y_src, y_t):
                    continue
                seen_finite_match = True
                v, info = judge_fn(bind(lhs, cs), bind(rhs, ct), vals)
                if v in ('PROMOTE', 'PASS'):
                    found = ct
                    break
                atom_kill = (cs, ct, info)
        if found is None:
            if atom_kill is not None:
                return 'DEMOTE', atom_kill
            n_no_witness += 1
            if n_no_witness >= 2:        # dead source / unfittable family: fail-safe, stays leaf-sort
                return 'NO-WITNESS', cs
    if n_no_witness:
        return 'NO-WITNESS', None
    return 'PROMOTE', None
