# mypy: ignore-errors
"""Independent f64 oracle for domain-extension checks on rewrite rules.

This module decides, for a candidate rule ``lhs -> rhs``, whether the rewrite
turns an UNDEFINED point into a DEFINED one (a "domain extension"). It does so
with an oracle that shares no code with the interval-arithmetic engine: the
deployed numpy realizations (``simplipy.operators.*``, resolved through the
engine's own ``realization`` fields), evaluated pointwise on sampled inputs.

Why an independent oracle. Interval evaluation pushes ONE interval through the
whole tree, so every variable is bound to the SAME interval and only the
diagonal cube ``[lo, hi]^n`` is ever explored. A NaN region whose shape is
relational (e.g. ``x0 < x1``) is therefore invisible to it. Sampling each
variable INDEPENDENTLY has no such blind spot, so this oracle can detect
extensions the interval engine cannot.

The verdict is asymmetric. Pointwise sampling is a LOWER bound on domain
extension: a narrow band (a constant inside a thin sliver, or a NaN region
thinner than the sample spacing) can be missed. So a detected extension is
real (up to the f64 saturation caveat below), but the absence of a detected
extension is only "no extension at this sampling density", never a proof of
soundness.

Known blind spot of THIS oracle: f64 saturation. ``acosh(abs(tanh x))`` is NaN
in exact arithmetic (``tanh x < 1`` strictly) but reads ``0.0`` in f64 once
``tanh`` saturates to ``1.0``; numpy then calls the source defined and no
extension is reported. That family is out of scope here and needs an
extended-precision (mpmath) evaluator instead. It is documented, not silently
swallowed.
"""
import json
import math
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..engine import SimpliPyEngine

N_PTS = int(os.environ.get('N_PTS', 200_000))
THRESH = float(os.environ.get('THRESH', 0.005))   # positive-measure floor
SEED = 20260716


def build_oracle(engine: "SimpliPyEngine"):
    """Resolve the engine's ``realization`` strings to the deployed numpy callables.

    Nothing here touches the Rust core: ``simplipy.operators`` is pure
    Python/numpy, and it is what the engine names as each token's realization.
    Infix realizations (``+ - * /``) map to the corresponding ``operator``
    builtins; ``simplipy.operators.<name>`` realizations resolve by attribute
    lookup. An unrecognised realization raises, so no token is silently dropped.

    Returns ``(arity, fns)``: the engine's operator-arity table and a mapping
    from token to its numpy callable.
    """
    import operator as _op

    from .. import operators as spo

    infix = {'+': _op.add, '-': _op.sub, '*': _op.mul, '/': _op.truediv}
    fns = {}
    for tok, real in engine.operator_realizations.items():
        if real in infix:
            fns[tok] = infix[real]
        elif real.startswith('simplipy.operators.'):
            fns[tok] = getattr(spo, real.rsplit('.', 1)[1])
        else:
            raise RuntimeError(f'unresolved realization for {tok!r}: {real!r}')
    return engine.operator_arity, fns


# Populated by ``configure(engine)`` (or by assigning the result of
# ``build_oracle`` to these names) before any evaluation. Kept as module-level
# state so the pure evaluation helpers below stay free of oracle plumbing.
ARITY: dict = {}
FNS: dict = {}


def configure(engine: "SimpliPyEngine"):
    """Install the oracle for ``engine`` into the module globals.

    Resolves ``engine`` through :func:`build_oracle` and binds ``ARITY`` and
    ``FNS`` so that :func:`evaluate`, :func:`variables`, :func:`domain_extension`
    and :func:`audit` operate against that engine's realizations. Must be called
    before those functions are used.
    """
    # Mutate the dicts IN PLACE rather than rebinding: sibling modules bind these
    # objects at import time (``from ._f64_eval import ARITY``), so a rebind would
    # leave them pointing at the empty originals. In-place update keeps every holder
    # of the object in sync.
    ar, fns = build_oracle(engine)
    ARITY.clear(); ARITY.update(ar)
    FNS.clear(); FNS.update(fns)
    return ARITY, FNS


LITERALS = {
    '0': 0.0, '1': 1.0, '(-1)': -1.0, 'np.e': math.e, 'np.pi': math.pi,
    'float("inf")': math.inf, 'float("-inf")': -math.inf, 'float("nan")': math.nan,
}


def _num(tok):
    try:
        return float(tok.strip('()'))
    except ValueError:
        return None


def evaluate(tokens, env):
    """Evaluate a prefix expression on numpy arrays via the deployed realizations."""
    pos = 0

    def walk():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok in ARITY:
            args = [walk() for _ in range(ARITY[tok])]
            with np.errstate(all='ignore'):
                return FNS[tok](*args)
        if tok in env:
            return env[tok]
        if tok in LITERALS:
            return np.full_like(next(iter(env.values())), LITERALS[tok])
        v = _num(tok)
        if v is None:
            raise KeyError(f'unknown leaf token: {tok!r}')
        return np.full_like(next(iter(env.values())), v)

    out = walk()
    if pos != len(tokens):
        raise ValueError(f'trailing tokens in {tokens!r}')
    return np.asarray(out, dtype=np.float64)


def variables(tokens):
    """Every leaf that is not an operator, a literal, a number or ``<constant>`` is a VARIABLE.

    Defined by EXCLUSION on purpose. Matching a name pattern instead (``x0``,
    say) is unsafe: rules may use ``_0``/``_1`` style wildcard leaf names, so a
    name-pattern match would classify them as variable-free and skip them
    silently. An unknown leaf must become a loud ``KeyError`` in ``evaluate``,
    never an empty set here.
    """
    return sorted({t for t in tokens
                   if t not in ARITY and t not in LITERALS
                   and t != '<constant>' and _num(t) is None})


def sample_env(varnames, rng):
    """Broad INDEPENDENT draws per variable -- the axis on which the interval engine is blind.

    Independent (not diagonal) is the whole reason this oracle can see relational NaN regions like
    `x0 < x1`. The mixture keeps mass near the +-1 branch points where inverse round-trips flip,
    while the wide arm reaches the tails.
    """
    env = {}
    for v in varnames:
        u = rng.random(N_PTS)
        x = np.where(u < 0.5, rng.uniform(-2.0, 2.0, N_PTS), rng.uniform(-8.0, 8.0, N_PTS))
        env[v] = x
    return env


def domain_extension(lhs, rhs):
    """Fraction of rows where the SOURCE is undefined but the TARGET is defined.

    This is the asymmetric-accept principle's undefined side: filling a measure-ZERO hole is a
    removable singularity and licit; filling a POSITIVE-measure region invents a function.
    Returns None if the rule is out of this oracle's scope.
    """
    varnames = variables(list(lhs) + list(rhs))
    if not varnames:
        return None                       # variable-free: no x-measure to speak of
    rng = np.random.default_rng(SEED)
    env = sample_env(varnames, rng)
    s = evaluate(list(lhs), env)
    t = evaluate(list(rhs), env)
    return float(np.mean(np.isnan(s) & ~np.isnan(t)))


def _pairs(d):
    return list(d.items()) if isinstance(d, dict) else [(r[0], r[1]) for r in d]


def load(path):
    out = {}
    for k, v in _pairs(json.load(open(path))):
        lhs = tuple(k.split()) if isinstance(k, str) else tuple(k)
        rhs = tuple(v.split()) if isinstance(v, str) else tuple(v)
        out[lhs] = rhs
    return out


def audit(path, label):
    rules = load(path)
    scope = {k: v for k, v in rules.items()
             if '<constant>' not in k and '<constant>' not in v}
    hits, skipped = [], 0
    for lhs, rhs in scope.items():
        try:
            f = domain_extension(lhs, rhs)
        except Exception as e:                       # never silently: a skip is REPORTED
            skipped += 1
            if skipped <= 3:
                print(f'    [skip] {" ".join(lhs)} -> {" ".join(rhs)}: {type(e).__name__}: {e}')
            continue
        if f is not None and f > THRESH:
            hits.append((lhs, rhs, f))
    hits.sort(key=lambda r: -r[2])
    print(f'{label:18s} {len(rules):5,d} rules | {len(scope):5,d} const-free '
          f'| {len(hits):3d} EXTEND a domain | {skipped} un-evaluable')
    return hits, len(scope), skipped
