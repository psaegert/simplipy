"""Structure-aware masking toolkit.

Masking is DOWNSTREAM POLICY, not engine behavior: the engine simplifies and delivers the
full expression with explicit constants; consumers decide what to abstract. This module is
the MECHANISM. It walks the engine's output (the native tagged form or the explicit binary
form) and tells a policy the structural ROLE of every literal it meets -- a multiplicative
coefficient, an additive constant, a ``pow`` exponent, a ``rootn`` index -- so a policy
like "mask constants, but keep in-vocabulary integer exponents" is a single conditional,
and position-blind accidents (masking a ``rootn`` index into a skeleton that is NaN almost
everywhere) are impossible to write by accident.

THE KINDS SIMPLIPY SHIPS. Choosing what to abstract is a real decision with real failure
modes, so the kinds live here rather than being re-invented by every consumer:

* :func:`mask_all` -- every number, including either half of a fraction and the special
  constants. For structural comparison.
* :func:`mask_fittable` -- every number a constant optimizer can actually fit, KEEPING the
  ones it cannot (``pow`` exponents, ``rootn`` indices). For training data.

A custom ``(value, role) -> str | None`` policy remains the primitive, so any other rule
(mask only floats, keep a given vocabulary, ...) is a one-line function.

ONE CONSTANT PER DEGREE OF FREEDOM -- the invariant every kind must satisfy. Masking is
TWO stages: (a) the positional substitution the policy drives, then (b) a COLLECT pass
that re-runs the engine over the result. Stage (b) is what makes the invariant hold, and
it is needed because the serializers split ONE state node across several tokens: a
rational is one node but prints as ``<mul> 2 x0 <div> 3 </mul>``, so a positional pass
alone would abstract it into TWO ``<constant>``s for ONE degree of freedom, and
``(c1*x)/c2`` is a flat direction for the refiner. Re-running the engine also collapses
``c1*c2`` and ``c1+c2`` (one degree of freedom each) while correctly LEAVING
``c1*x0 + c2*x1`` alone (two), because each ``<constant>`` occurrence is an independent
free value, not a shared symbol.

Stage (b) additionally normalizes the SHAPE, which is why a masked coefficient reads the
way a human would write it: ``x0 / <constant>`` becomes ``<constant> * x0`` and
``x0 - <constant>`` becomes ``x0 + <constant>`` -- for a free constant the reciprocal and
the negation are just another free constant, so the engine's own ordering prefers the
multiplicative/additive form.

This is a DELIBERATE reversal of the older "never re-simplify a masked expression" rule,
which was written when masking promised a 1:1 token correspondence with the emission. That
promise is what produced the redundant constants; the degree-of-freedom invariant is worth
more. ``mask(..., collect=False)`` still gives the raw positional substitution.

MASK ONCE. Masking DESTROYS information, so chaining it has consequences that are easy to
miss: feed :func:`mask` an expression, never a skeleton. Concretely, stage (b) can turn
STRUCTURE into a literal, and a second pass then abstracts that literal into a degree of
freedom the original never had::

    <add> <sub> atan <mul> pow x4 3 pow abs x4 3 </mul> sin <mul> 0 log x4 </mul> </add>
    once   ->  <add> atan <mul> -1 pow x4 3 ... </mul> sin <mul> <constant> log x4 </mul> </add>
    twice  ->  <add> atan <mul> <constant> pow x4 3 ... </mul> sin <mul> <constant> ... </mul> </add>
    constants: 0 -> 1 -> 2

The ``-1`` after one pass IS the original ``<sub>``: collect moved the negation inside
``atan`` (an odd function), so it is structure wearing a literal's clothes. Masking it
invents a coefficient. Measured at ~0.15% of skeletons over 20k rows.

KEEPING A SYMBOLIC CONSTANT BESIDE A MASKED ONE IS POINTLESS (owner, 2026-08-07). A policy
may keep ``np.pi`` while masking its coefficient, and the mechanism will honour it -- but
the result ``<constant> * np.pi`` is worth nothing: the constant optimizer just fits an
arbitrary float, and an arbitrary float times pi is an arbitrary float, so the
factorization it looks like it preserves is not there. The shipped kinds mask both and
collapse them to ONE ``<constant>``, which is the honest skeleton. Keep a symbol only if
the DOWNSTREAM consumer can predict its coefficient exactly (an integer or a fraction);
if those coefficients are masked anyway, keeping the symbol buys nothing.

Skeleton SIGN placement is decided entirely by the serializers' shared sign doctrine,
so masking needs no sign handling of its own: a coefficient's sign lives inside the
literal in both dialects (``* -0.84 x0``, ``<mul> -0.84 x0 </mul>``), so masking it
leaves no wrapper behind (``* <constant> x0``); additive term signs are STRUCTURE (the
``<sub>`` section, the explicit binary minus) and stay as spelled -- ``x - C`` and
``x + C`` are distinct canonical states, and collapsing the masked families
(``- x0 <constant>`` == ``+ x0 <constant>`` as skeletons) is downstream skeleton
normalization, deliberately outside this module's positional contract.

Example (the flash-ansr-style policy)::

    from simplipy import masking

    def keep_structure(value: str, role: masking.Role) -> str | None:
        if role in (masking.Role.EXPONENT, masking.Role.ROOT_INDEX):
            return None            # keep structural integers: pow x0 2 stays pow x0 2
        return "<constant>"        # abstract every value-position literal

    skeleton = masking.mask(engine.simplify(tokens), engine, keep_structure)
"""
import enum
from typing import TYPE_CHECKING, Callable, Optional

from simplipy.utils import is_numeric_string, reserved_numeric_spelling

if TYPE_CHECKING:
    from simplipy.engine import SimpliPyEngine

__all__ = ["Role", "literal_sites", "mask", "mask_all", "mask_fittable",
           "mask_values_keep_structure"]


class Role(enum.Enum):
    """The structural position of a literal inside a canonical expression."""

    COEFFICIENT = "coefficient"  # a factor in a multiplicative context (incl. / operands)
    ADDEND = "addend"            # a term in an additive context (incl. - operands)
    EXPONENT = "exponent"        # the second operand of ``pow``
    ROOT_INDEX = "root_index"    # the second operand of ``rootn``
    VALUE = "value"              # everything else: function argument, pow base, lone literal


_OPENERS = {"<add>": ("</add>", "<sub>", Role.ADDEND),
            "<mul>": ("</mul>", "<div>", Role.COEFFICIENT)}
# Sign/reciprocal wrappers are role-transparent: the literal in ``neg 2`` plays whatever
# role the wrapped position plays.
_TRANSPARENT = {"neg", "inv"}
# Special constants are SITES like any numeric literal (owner ruling 2026-07-31: "mask
# all should mask all, even pi or e or other specials") -- the POLICY decides their fate,
# which keeps a future pi-vocabulary pipeline a one-line policy away. Poles and nan are
# deliberately NOT here: a fitted `<constant>` is finite by doctrine, so masking an
# infinity would manufacture an unfittable skeleton.
_SPECIAL_CONSTANTS = {"np.pi", "np.e"}


def literal_sites(tokens: list[str], engine: "SimpliPyEngine") -> list[tuple[int, str, Role]]:
    """Every numeric literal AND special constant (``np.pi``/``np.e``) in ``tokens``
    with its index and structural :class:`Role`.

    Accepts both the tagged and the explicit projection (and any mix the shared parser
    accepts). ``engine`` supplies the config's function arities; the tag delimiters and
    the core serialization language (``pow``/``rootn``/``*``/``/``/``+``/``-``/``neg``/
    ``inv``) are understood natively. Raises ``ValueError`` on a malformed sequence.
    """
    # Read the table DIRECTLY: a wrong `engine` must fail here at the boundary, never
    # degrade the walk to an empty arity map -- function tokens would read as leaves,
    # giving misleading "malformed" errors on explicit input and silently wrong roles
    # on tagged input (the argument literal takes the enclosing bag's role).
    # H-007: the same token-boundary ruling as the core -- a reserved numeric spelling
    # (``inf``, ``1_000``, ``0x10``) must never reach the role walk as a "symbol".
    for t in tokens:
        if reserved_numeric_spelling(t):
            raise ValueError(
                f"invalid token {t!r}: reserved numeric spelling -- numeric to Python but "
                f"not a simplipy numeric literal (H-007); use the canonical spelling "
                f"('5', '0.5', '1e-05', '1/3', float(\"inf\"), float(\"-inf\"), float(\"nan\"))")
    arity = dict(engine.operator_arity)
    sites: list[tuple[int, str, Role]] = []

    def walk(i: int, role: Role) -> int:
        if i >= len(tokens):
            raise ValueError("malformed prefix expression: ran out of tokens")
        t = tokens[i]
        if t in _OPENERS:
            closer, section, child_role = _OPENERS[t]
            i += 1
            while True:
                if i >= len(tokens):
                    raise ValueError(f"malformed tagged expression: unclosed {t}")
                if tokens[i] == closer:
                    return i + 1
                if tokens[i] == section:
                    i += 1
                    continue
                i = walk(i, child_role)
        if t == "pow":
            i = walk(i + 1, Role.VALUE)
            return walk(i, Role.EXPONENT)
        if t == "rootn":
            i = walk(i + 1, Role.VALUE)
            return walk(i, Role.ROOT_INDEX)
        if t in ("*", "/"):
            i = walk(i + 1, Role.COEFFICIENT)
            return walk(i, Role.COEFFICIENT)
        if t in ("+", "-"):
            i = walk(i + 1, Role.ADDEND)
            return walk(i, Role.ADDEND)
        if t in _TRANSPARENT:
            return walk(i + 1, role)
        n = arity.get(t)
        if n:
            j = i + 1
            for _ in range(int(n)):
                j = walk(j, Role.VALUE)
            return j
        if t != "<constant>" and (t in _SPECIAL_CONSTANTS or is_numeric_string(t)):
            sites.append((i, t, role))
        return i + 1

    end = walk(0, Role.VALUE)
    if end != len(tokens):
        raise ValueError("malformed prefix expression: trailing tokens")
    return sites


def _is_tagged(tokens: list[str]) -> bool:
    return any(t in ("<mul>", "<add>", "</mul>", "</add>", "<div>", "<sub>") for t in tokens)


def mask(tokens: list[str], engine: "SimpliPyEngine",
         policy: Callable[[str, Role], Optional[str]], *, collect: bool = True) -> list[str]:
    """Apply a masking ``policy``: ``policy(value, role)`` returns the replacement token,
    or ``None`` to keep the literal as written.

    Two stages. (a) The policy is applied POSITIONALLY, one literal at a time, so a policy
    is a pure ``(value, role)`` decision and nothing else. (b) ``collect`` then re-runs the
    engine over the substituted tokens, which is what enforces ONE ``<constant>`` PER
    DEGREE OF FREEDOM -- see the module docstring. The emitted dialect is preserved.

    Pass ``collect=False`` for the raw positional substitution (a strict 1:1 token
    correspondence with ``tokens``), at the cost of the degree-of-freedom guarantee.

    ONE-SHOT, BY DESIGN -- feed it an unmasked expression, never a skeleton. ``mask`` is
    deliberately NOT idempotent: stage (b) can turn STRUCTURE into a literal, and masking
    that literal would widen the family. Measured on 20k rows, ~0.15% of skeletons: a
    structural negation (`<sub> atan(A)`) becomes an explicit `-1` coefficient once the
    odd-function rule fires (`-atan(A)` -> `atan(-1*A)`), and a second pass would abstract
    that `-1` into a free constant -- a degree of freedom the original never had.
    """
    out = list(tokens)
    for idx, value, role in literal_sites(tokens, engine):
        replacement = policy(value, role)
        if replacement is not None:
            out[idx] = replacement
    if collect and out != list(tokens):
        out = engine.simplify(out, form="tagged" if _is_tagged(tokens) else "explicit")
    return out


def mask_all(value: str, role: Role) -> Optional[str]:
    """KIND 1 -- mask everything. Every number becomes ``<constant>``: a standalone
    integer, either half of a fraction, a float, and the special constants
    (``np.pi``/``np.e``) too.

    This masks EXPONENTS and ROOT INDICES as well, which makes the skeleton hard or
    impossible to fit (see :func:`mask_fittable` for why). Use it when the skeleton is
    for structural comparison rather than for a constant optimizer.
    """
    return "<constant>"


def mask_fittable(value: str, role: Role) -> Optional[str]:
    """KIND 2 -- mask every number a constant optimizer can actually fit, and KEEP the
    ones it cannot.

    The criterion is not a taste call: a literal is UNFITTABLE when its value controls the
    expression's DOMAIN rather than its magnitude, so moving it off its exact value does
    not make the fit worse, it makes the function undefined.

    * ``rootn`` INDEX -- ``rootn(x, 3.5)`` is ``nan`` for EVERY ``x``. The index lives on
      the integers, so a continuous optimizer has no gradient anywhere to descend.
    * ``pow`` EXPONENT -- ``pow(x, 2.5)`` is ``nan`` wherever ``x < 0``. The exponent's
      integrality decides the domain, so masking it silently deletes half the data.

    Everything else is smooth in its literal (``c*x``, ``x+c``, ``log(c*x)``) and is
    masked. This is the policy for training-data preparation.
    """
    if role in (Role.EXPONENT, Role.ROOT_INDEX):
        return None
    return "<constant>"


#: Historical name for :func:`mask_fittable`, kept so existing callers keep working.
mask_values_keep_structure = mask_fittable
