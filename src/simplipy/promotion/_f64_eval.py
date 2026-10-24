# mypy: ignore-errors
"""Independent f64 evaluator for the sort-promotion checks on rewrite rules.

It evaluates a rule side pointwise with the deployed numpy realizations
(``simplipy.operators.*``, resolved through the engine's own ``realization``
fields), sharing no code with the interval-arithmetic engine. Interval
evaluation pushes ONE interval through the whole tree, so every variable is
bound to the SAME interval and a relational NaN region (``x0 < x1``) is
invisible to it; evaluating each variable INDEPENDENTLY has no such blind spot.

Known blind spot of THIS evaluator: f64 saturation. ``acosh(abs(tanh x))`` is NaN
in exact arithmetic (``tanh x < 1`` strictly) but reads ``0.0`` in f64 once
``tanh`` saturates to ``1.0``; numpy then calls the source defined and no
extension is reported. That family is out of scope here and needs an
extended-precision (mpmath) evaluator instead. It is documented, not silently
swallowed.
"""
import math
from typing import TYPE_CHECKING

import numpy as np

from simplipy.utils import literal_float, reserved_numeric_spelling

if TYPE_CHECKING:
    from ..engine import SimpliPyEngine


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

    infix = {'+': _op.add, '-': _op.sub, '*': _op.mul, '/': _op.truediv,
             '**': _op.pow}
    fns = {}
    for tok, real in engine.operator_realizations.items():
        if real in infix:
            fns[tok] = infix[real]
        elif real.startswith('simplipy.operators.'):
            fns[tok] = getattr(spo, real.rsplit('.', 1)[1])
        else:
            # General dotted realization, resolved the way the ENGINE resolves it
            # (import the leading module, walk the attribute path; `np` is the
            # numpy alias the configs use). The engine accepts `np.cos`-style
            # realizations, so the oracle must too -- with promotion on by
            # default (2026-08-01) every small non-canonical config takes this
            # path. Anything unresolvable still raises: no token is silently
            # dropped.
            parts = real.split('.')
            obj = {'np': np, 'numpy': np, 'math': math}.get(parts[0])
            if obj is None:
                import importlib
                try:
                    obj = importlib.import_module(parts[0])
                except ImportError:
                    obj = None
            for attr in parts[1:]:
                if obj is None:
                    break
                obj = getattr(obj, attr, None)
            if obj is None or not callable(obj):
                raise RuntimeError(f'unresolved realization for {tok!r}: {real!r}')
            fns[tok] = obj
    return engine.operator_arity, fns


# Populated by ``configure(engine)`` (or by assigning the result of
# ``build_oracle`` to these names) before any evaluation. Kept as module-level
# state so the pure evaluation helpers below stay free of oracle plumbing.
ARITY: dict = {}
FNS: dict = {}


def configure(engine: "SimpliPyEngine"):
    """Install the oracle for ``engine`` into the module globals.

    Resolves ``engine`` through :func:`build_oracle` and binds ``ARITY`` and
    ``FNS`` so that :func:`evaluate` operates against that engine's realizations.
    Must be called before it is used.
    """
    # Mutate the dicts IN PLACE rather than rebinding: sibling modules bind these
    # objects at import time (``from ._f64_eval import ARITY``), so a rebind would
    # leave them pointing at the empty originals. In-place update keeps every holder
    # of the object in sync.
    ar, fns = build_oracle(engine)
    ARITY.clear(); ARITY.update(ar)
    FNS.clear(); FNS.update(fns)
    # Engine-DEPENDENT caches must die with the old oracle: the prefold zero-snap
    # parses spans through ARITY, so an entry computed under one vocabulary is
    # WRONG under another (measured 2026-08-01, promotion-on-by-default: an
    # all-infix mini engine cached `* sin np.pi ?0` UNSNAPPED -- `sin` absent from
    # its ARITY -- and the canonical promote later replayed the stale entry,
    # killing its own positive control at the inf atom with 1.2246e-16 * inf).
    from . import _pointwise
    _pointwise._PREFOLD_CACHE.clear()
    return ARITY, FNS


LITERALS = {
    '0': 0.0, '1': 1.0, '(-1)': -1.0, 'np.e': math.e, 'np.pi': math.pi,
    'float("inf")': math.inf, 'float("-inf")': -math.inf, 'float("nan")': math.nan,
}


def _num(tok):
    # H-007: never bare float() -- a reserved spelling ('inf', '1_000') is neither a
    # literal nor a symbol; the boundary forbids it, so seeing one here is a defect.
    core = tok.strip('()')
    if reserved_numeric_spelling(core):
        raise ValueError(f"reserved numeric spelling {tok!r} reached the f64 oracle")
    # fractions and huge integers read as the compiled expression reads them
    return literal_float(core)


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
