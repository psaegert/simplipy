"""The two canonical expression FORMS: the expression form and the skeleton form.

Both name their OUTPUT, the way the engine's ``to_infix``/``to_prefix``/``to_tagged``
name theirs -- but along a different axis. Those three name a DIALECT (how an
expression is spelled); these two name an ABSTRACTION LEVEL:

* :func:`to_expression` -- variable tokens (``v1``/``x1`` style, case-insensitive)
  canonicalised to ``x{n}`` and the constant placeholder folded to one spelling,
  numeric literals KEPT. Two expressions that are "the same" compare equal.
* :func:`to_skeleton` -- the same, plus EVERY numeric literal replaced by
  ``<constant>``. Every one: ``pow`` exponents and ``rootn`` indices included, which
  is exactly what the old name ``normalize_skeleton`` hid. ``pow(x0, 3)`` becomes
  ``pow(x0, <constant>)``, a family no constant optimizer can fit (a non-integer
  exponent is ``nan`` wherever the base is negative). If you want the fittable
  literals abstracted and the structural ones kept, that is
  ``engine.mask(expr, 'fittable')``, not a skeleton.

Both take all three external forms -- an infix ``str``, an explicit binary-prefix
token sequence, a tagged token sequence -- and RETURN THE CALLER'S DIALECT
(``str`` -> ``str``, prefix -> prefix, tagged -> tagged). Abstraction level and
dialect are independent, so neither function may silently re-spell.

THE CONTENT IS DIALECT-INVARIANT, and that is not free
-------------------------------------------------------
Independent does not mean unrelated. Masking the caller's OWN spelling, one token
at a time, gives a different number of ``<constant>`` slots for the same
expression depending on which dialect it arrived in -- in BOTH directions, because
the two token dialects disagree about how many literals an expression even has::

    -x0*x1   explicit `neg * x0 x1`           -> 0 <constant>   sign is STRUCTURE
             tagged   `<mul> -1 x0 x1 </mul>` -> 1 <constant>   bag folds it to -1
    -1/x0    explicit `/ -1 x0`               -> 1 <constant>
             tagged   `neg inv x0`            -> 0 <constant>   reciprocal is STRUCTURE

Measured at 35/400 rows (8.8%) of the repo corpus for the pre-0.14 positional
pass, and at the same 35/400 for a role-aware ``mask(mask_all)`` applied to each
dialect separately -- role-awareness does not help, because the disagreement is
about what the SITES ARE, not about their roles. A skeleton is what downstream
holdout and decontamination KEY on, so a dialect-dependent answer is a
research-integrity hazard: a tagged-era pipeline silently fails to match
explicit-era keys and reports clean.

So every input is canonicalised into the ONE internal AC state first, abstracted
THERE, and only then rendered back into the caller's dialect. The masking
substrate is the canonical EXPLICIT binary prefix -- the dialect the rule
artifacts, the miner's universe and every current caller are already defined on,
and the one where a sign and a reciprocal stay structure instead of being folded
into a literal that masking would mistake for a fitted parameter. That choice is
load-bearing: it is what decides the constant count for the two shapes above.

Requiring an ``engine`` is the cost of the guarantee. There is no engine-free
path and deliberately no fall-back to the positional pass: silently returning a
dialect-dependent key is the failure this module exists to prevent.

ONE MASKING PATH
----------------
:func:`to_skeleton` is not a third masking implementation. It IS
:func:`to_expression` followed by the ratified front door
``engine.mask(..., policy='all')`` -- the same mechanism, the same role walk, the
same one-``<constant>``-per-degree-of-freedom collect stage (``2*x0/3`` is ONE
free value, not two). Two consequences worth knowing:

* the collect stage re-runs the engine, so simplification RULES fire; a skeleton
  therefore depends on the engine ARTIFACT, not only on the expression;
* ``engine.mask(to_expression(e, engine), 'all', collect=False)`` is the escape
  for a rules-free, strictly positional abstraction of the canonical spelling.

:func:`normalize_variable_token` keeps its name: it is a single-TOKEN helper that
answers ``(token, is_variable)``. It produces no form, reads no dialect and needs
no engine, so the ``to_*`` family naming would misdescribe it.

These helpers were relocated here (0.3.1) from flash-ansr so the canonicalizer
lives at the shared expression-engine leaf that symbolic-data, flash-ansr and
srbf all depend on.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, cast

import numpy as np

# The tag delimiters, from the ONE place they are defined (the engine names the
# dialect with this set in its own diagnostics); a second copy of a doctrine
# constant is a second thing to keep in sync.
from .engine import SimpliPyEngine, _TAGGED_DIALECT_TOKENS

__all__ = ["normalize_variable_token", "to_expression", "to_skeleton"]

_VAR_TOKEN_PATTERN = re.compile(r"^[vx](\d+)$", re.IGNORECASE)

# Two spellings of ONE placeholder. Folding them is canonicalisation of the same
# kind as the variable renaming, so it belongs to the EXPRESSION form -- which is
# what keeps `to_skeleton == to_expression + mask('all')` exactly true.
_CONSTANT_PLACEHOLDER_SPELLINGS = frozenset({"<constant>", "<c>"})



def normalize_variable_token(token: str) -> tuple[str, bool]:
    """Return ``(normalized_token, is_variable)``.

    Recognizes tokens like ``v1`` or ``x2`` (case-insensitive) and returns them as
    ``x{n}``. Non-variable tokens are returned unchanged with ``is_variable=False``.

    A single-TOKEN helper, not a form producer: it reads no dialect, needs no
    engine, and answers about one token. That is why it keeps its ``normalize_``
    name while the two form producers became :func:`to_expression` /
    :func:`to_skeleton`.
    """
    match = _VAR_TOKEN_PATTERN.match(token)
    if match:
        return f"x{int(match.group(1))}", True
    return token, False


def to_expression(expression: str | list[str] | tuple[str, ...] | np.ndarray | None,
                  engine: SimpliPyEngine) -> str | list[str] | None:
    """The canonical EXPRESSION form: variables canonicalised, literals kept.

    Variable tokens (``v1``/``V1``/``x1``) become ``x{n}`` and the constant
    placeholder folds to one spelling (``<c>`` -> ``<constant>``); numeric
    literals are left exactly as written, so the form still carries concrete
    values. The expression is canonicalised through the engine's internal AC
    state, so the RESULT does not depend on the dialect the input arrived in --
    see the module docstring for why that is load-bearing rather than tidy.

    Parameters
    ----------
    expression : str, list[str], tuple[str, ...], 1-D ndarray or None
        An infix ``str``, an explicit binary-prefix token sequence, or a tagged
        token sequence; the form is detected the way the conversion API detects
        it (design D2: type first, then a liberal read of the tokens). ``None``
        maps to ``None``.
    engine : SimpliPyEngine
        Supplies the canonical state. Required: there is no engine-free path.

    Returns
    -------
    str or list[str] or None
        THE CALLER'S DIALECT: an infix ``str`` for an infix ``str``, a tagged
        token list for tagged tokens, an explicit binary-prefix token list
        otherwise. Token containers are not mirrored (a tuple or a 1-D ndarray in
        is a list out) -- the dialect is the contract, the container is not.

    Raises
    ------
    ValueError
        On a malformed expression, undeclared vocabulary, or a reserved numeric
        spelling (``inf``, ``1_000``) -- the core's own token grammar (H-007).
    TypeError
        On a non-str/list/tuple/1-D-ndarray expression.
    """
    if expression is None:
        return None
    dialect = _input_dialect(expression)
    return _render(_canonical_expression(expression, engine), dialect, engine)


def to_skeleton(expression: str | list[str] | tuple[str, ...] | np.ndarray | None,
                engine: SimpliPyEngine) -> str | list[str] | None:
    """The SKELETON form: the canonical expression with EVERY numeric masked.

    Equivalent, exactly, to :func:`to_expression` followed by the ratified front
    door ``engine.mask(..., policy='all')`` -- it is not a third masking path.
    "Every numeric" means every one: coefficients, addends, function arguments,
    the special constants ``np.pi``/``np.e``, and ``pow`` EXPONENTS and ``rootn``
    INDICES too. ``pow(x0, 3)`` becomes ``pow(x0, <constant>)``, which is a family
    no constant optimizer can fit; ``engine.mask(expr, 'fittable')`` is the kind
    that keeps the structural literals.

    Because the front door's collect stage re-runs the engine (that is what
    enforces one ``<constant>`` per degree of freedom -- ``2*x0/3`` is ONE free
    value), simplification RULES fire, so a skeleton depends on the engine
    ARTIFACT and not only on the expression. For a rules-free, strictly positional
    abstraction of the canonical spelling, call
    ``engine.mask(to_expression(e, engine), 'all', collect=False)``.

    Accepts all three external forms and returns the caller's dialect; the
    CONTENT is dialect-invariant (module docstring). Same parameters, return
    contract and error taxonomy as :func:`to_expression`.
    """
    if expression is None:
        return None
    dialect = _input_dialect(expression)
    canonical = _canonical_expression(expression, engine)
    # Nothing to mask, and the role walk refuses an empty sequence outright.
    masked = cast("list[str]", engine.mask(canonical, "all")) if canonical else canonical
    return _render(masked, dialect, engine)


def _input_dialect(expression: str | list[str] | tuple[str, ...] | np.ndarray) -> str:
    """Which external form ``expression`` is written in -- the conversion API's
    own detection (design D2): a ``str`` is INFIX; a token sequence is TAGGED when
    it carries a bag delimiter or an inverse-section marker, and the explicit
    binary PREFIX otherwise.

    ``neg``/``inv`` are legal in BOTH token dialects and carry no delimiter, so a
    sequence spelled only with them reads as explicit. That is a call about the
    OUTPUT SPELLING only: the content is decided in the canonical state, so it is
    the same skeleton either way.
    """
    if isinstance(expression, str):
        return "infix"
    if isinstance(expression, np.ndarray):
        tokens = [str(t) for t in expression.ravel().tolist()]
    elif isinstance(expression, (list, tuple)):
        tokens = [str(t) for t in expression]
    else:
        raise TypeError(
            f"expected an infix str or a token sequence (list, tuple or 1-D "
            f"ndarray), not {type(expression).__name__}")
    return "tagged" if _TAGGED_DIALECT_TOKENS.intersection(tokens) else "explicit"


def _canonical_expression(expression: str | list[str] | tuple[str, ...] | np.ndarray,
                          engine: SimpliPyEngine) -> list[str]:
    """The ONE internal representation both forms are computed on: the canonical
    explicit binary prefix, with variables and the constant placeholder
    canonicalised."""
    # to_prefix is a pure NOTATION conversion since the split (2026-08-18), so the
    # CANONICALISATION has to be asked for explicitly -- `simplify` is what folds,
    # collects and sorts. Converting first and simplifying second is the exact
    # composition: it reaches the same canonical state from all three dialects,
    # which is the dialect-invariance this module promises.
    canonical = cast("list[str]", engine.simplify(engine.to_prefix(expression)))
    renamed = [_canonical_leaf(token) for token in canonical]
    if renamed == canonical:
        return canonical
    # A rename can move a leaf out of canonical ORDER (`* v9 x1` renames to
    # `* x9 x1`, which no longer sorts), so the state is rebuilt from the renamed
    # spelling rather than patched in place.
    return cast("list[str]", engine.simplify(renamed))


def _canonical_leaf(token: str) -> str:
    normalized, is_variable = normalize_variable_token(token)
    if is_variable:
        return normalized
    return "<constant>" if token in _CONSTANT_PLACEHOLDER_SPELLINGS else token


def _render(tokens: list[str], dialect: str, engine: SimpliPyEngine) -> str | list[str]:
    """Project the canonical explicit tokens back into the caller's dialect."""
    if dialect == "infix":
        return engine.to_infix(tokens)
    if dialect == "tagged":
        return engine.to_tagged(tokens)
    return tokens
