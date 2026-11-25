"""The compiled core and the vocabulary it fixes: the core serialization language's arities, the
operator spec every config entry carries, the tokens of the tagged dialect, and the instances a
rule's promise is read on."""

from itertools import product
from typing import Any

import numpy as np

from simplipy.mining import _CONSTLIKE_LEAVES
from simplipy.utils import is_numeric_string


try:
    # The compiled INLINE core (the simplipy._core extension): simplify + conversions + validation
    # + the offline miner. REQUIRED (see the module docstring): a missing/unbuilt extension is a
    # hard error at engine construction.
    from simplipy._core import Engine as _RustEngine  # type: ignore[attr-defined]
    from simplipy._core import core_serialization_ops as _core_serialization_ops  # type: ignore[attr-defined]
    _CORE_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover  (missing/unbuilt extension)
    _RustEngine = None  # type: ignore[assignment, misc]
    _core_serialization_ops = None  # type: ignore[assignment]
    _CORE_IMPORT_ERROR = _exc


def _is_slot_token(token: str) -> bool:
    """A sorted-rule slot token: a sort sigil (``_`` / ``?`` / ``!`` / ``$``) followed by
    digits."""
    return bool(token) and token[0] in '_?!$' and token[1:].isdigit()


#: The values outside a variable's generic reading. Permissive binds every slot to any
#: subtree, a literal included, and its constructor reads a variable as finite and nonzero
#: almost everywhere -- which says nothing about these three.
_NON_FINITE_LITERALS = ('float("inf")', 'float("-inf")', 'float("nan")')


def _coverage_variants(
        lhs: tuple[str, ...],
        rhs: tuple[str, ...],
        rule_mode: str,
        non_finite: bool = True,
        composite: bool = True) -> list[tuple[list[str], list[str]]]:
    """Instantiation variants a rule must survive to count as covered in ``rule_mode``:
    the instances its slots bind there.

    Every slot in ``lhs`` is instantiated as a distinct variable leaf ``x{i}``;
    ``<constant>`` stays LITERAL (substituting a numeral would let native constant
    folding fake coverage). Additionally, every subset of up to 3 wide slots
    (``_``/``!``/``$`` sigils, which bind composites) is probed with a composite
    (``composite=False`` probes none): leaf-only instantiation under-tests what a wide slot
    claims to match. The composite is ``1/(x{i} + x{k})``, ``x{k}`` a variable of its own:
    in the core language every engine reads whatever its alphabet, finite and nonzero almost
    everywhere, so every wide sort binds it, and no bag, so a narrower slot cannot take it
    apart -- a bare sum it can: ``?0 + 1`` rewrites ``(x{i} + x{k}) + 1`` with ``?0`` bound
    to ``x{i}`` alone, which would read a ``_0 + 1`` rule as covered by it.
    In ``permissive``, where every slot binds any subtree, each non-finite literal is a
    variant too (all slots at once): `$0 / $0 -> 1` is what takes `inf/inf` to `1` there,
    which the constructor takes to `nan`. (In ``strict`` the ground fold reads such an instance
    before any rule.) ``non_finite=False`` leaves that tier out: the
    GENERIC instances only, which the liveness gate asks a rule to fire on.
    """
    slots = list(dict.fromkeys(t for t in lhs if _is_slot_token(t)))
    wide = [s for s in slots if s[0] in '_!$'][:3] if composite else []
    # the composite's own variables: past every index the rule's slots take
    fresh = 1 + max((int(s[1:]) for s in slots if s[1:].isdigit()), default=-1)
    bindings_list = []
    for mask in product((0, 1), repeat=len(wide)):
        bindings = {s: ['x' + s[1:]] for s in slots}
        for k, (slot, wrapped) in enumerate(zip(wide, mask)):
            if wrapped:
                bindings[slot] = ['inv', '+', 'x' + slot[1:], f'x{fresh + k}']
        bindings_list.append(bindings)
    if non_finite and rule_mode == 'permissive' and slots:
        bindings_list += [{s: [value] for s in slots} for value in _NON_FINITE_LITERALS]
    return [([out for t in lhs for out in b.get(t, [t])],
             [out for t in rhs for out in b.get(t, [t])]) for b in bindings_list]


def _reaches_promise(core: Any, rule_mode: str, variant_lhs: list[str],
                     variant_rhs: list[str], arity: dict[str, int]) -> bool:
    """Does ``core``'s serve answer in ``rule_mode`` take the instance ``variant_lhs`` to at
    most the promised ``variant_rhs`` in the serve-time reduction ordering?

    ``probe_out <= rhs`` is spelled ``!(rhs < probe_out)`` in the total order, so equality is
    state identity: the answer literally reaches the promise or something below it, never
    merely something that SCORES the same. One tier needs more than the ordering: a
    ``<constant>`` in a promise is EXISTENTIAL, so an answer reaching an INSTANCE of the
    promise (each Const slot bound to a ground subterm -- ``0`` fulfills ``<constant>``,
    ``* 2.71... x0`` fulfills ``* <constant> x0``) has fulfilled it, though the total order
    sees only unrelated leaf kinds there. The covered-rule prune asks it of the engine
    WITHOUT the rule, the liveness gate of the engine WITH it.
    """
    probe_out = core.mine_mark(list(variant_lhs), rule_mode=rule_mode)
    if not core.ac_ordered_below_in(list(variant_rhs), probe_out, rule_mode=rule_mode):
        return True
    if '<constant>' in variant_rhs:
        promise = core.mine_mark(list(variant_rhs), rule_mode=rule_mode)
        return _fulfills_constant_promise(promise, probe_out, arity)
    return False


# The core serialization language's arities (config-independent built-ins): the
# canonical explicit projection emits these regardless of the config vocabulary,
# so any walk over projection output needs them alongside `operator_arity`.


#: Arities of the CORE SERIALIZATION LANGUAGE, read from the engine's own table rather
#: than restated here. A hand-maintained copy of this data that drifts
#: is a soundness problem, not a tidiness one: the
#: masking walk treats an unknown operator as a LEAF, so its operands inherit the
#: enclosing bag's role and a `pow` exponent can be masked, exactly the accident the
#: role API exists to make impossible.
#: Empty only when the extension is missing, in which case engine construction raises
#: anyway (`_CORE_IMPORT_ERROR`) -- so there is never a stale literal to drift.
_CORE_SERIALIZATION_ARITIES = (
    {tok: spec['arity'] for tok, spec in _core_serialization_ops().items()}
    if _core_serialization_ops is not None else {})


# The bag delimiters and inverse-section markers of the TAGGED serialization --
# `simplify`'s default token output (`<add> ... <sub> ... </add>`, and the lattice bags
# `<max> ... </max>` / `<min> ... </min>` of an engine that declares `max` / `min`). The
# explicit-dialect entry points (`is_valid`, `prefix_to_infix`) do not read this form; they
# use this set to NAME the dialect in their diagnostics instead of failing with a bare arity
# error.
_TAGGED_DIALECT_TOKENS = frozenset({'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>',
                                    '<max>', '</max>', '<min>', '</min>'})

#: Every key an operator spec must carry. `precedence` is deliberately NOT
#: required: non-core operators render as function calls, which never consult it; a
#: core serialization token missing the key gets the core table's own value (a
#: PRESENT-and-conflicting value still refuses in the core-token guard -- absence
#: used to slip past that guard entirely).
# `inverse` left the REQUIRED set -- nothing reads
# it (parse-only since the relic layer's deletion), and requiring pure ceremony made
# every wild config fail construction. Declared values are still accepted silently.
_REQUIRED_OPERATOR_KEYS = ('realization', 'alias', 'arity', 'commutative')


def _normalized_operator_specs(operators: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate operator specs LOUDLY and normalize them.

    Three failure modes are refused by name instead of failing silently: a spec missing
    a key (it would die as a bare ``KeyError: 'alias'`` naming neither the operator nor
    the file); ``commutative: true`` on a non-core operator (only ``+`` and ``*`` are AC
    bags, so ``f(a, b)`` and ``f(b, a)`` would stay two canonical states while the
    config claims one value); and a core serialization token declared WITHOUT
    ``precedence:`` (it would bypass the conflicting-precedence guard, which compares
    only present values).
    """
    normalized: dict[str, dict[str, Any]] = {}
    core = _core_serialization_ops() if _core_serialization_ops is not None else {}
    for token, spec in operators.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f'config error: operator {token!r} must map to a spec mapping, '
                f'got {type(spec).__name__}')
        missing = [key for key in _REQUIRED_OPERATOR_KEYS if key not in spec]
        if missing:
            raise ValueError(
                f'config error: operator {token!r} is missing required key(s) '
                f'{", ".join(map(repr, missing))}; every operator spec carries '
                f'{", ".join(map(repr, _REQUIRED_OPERATOR_KEYS))} '
                "('precedence' is optional for non-core operators: they render as "
                'function calls, which never consult it)')
        if spec.get('commutative') and token not in ('+', '*'):
            raise ValueError(
                f"config error: operator {token!r} declares 'commutative: true', which the "
                "engine cannot honor: only '+' and '*' canonicalize under commutativity "
                f'(the AC bags), so {token}(a, b) and {token}(b, a) would stay two distinct '
                "canonical states while the config claims one value. Declare "
                "'commutative: false' (rendering and evaluation are unaffected)")
        out = dict(spec)
        if 'precedence' not in out and token in core:
            out['precedence'] = core[token]['precedence']
        normalized[token] = out
    return normalized


def _fulfills_constant_promise(pattern: list[str], subject: list[str],
                               arity: dict[str, int]) -> bool:
    """Does ``subject`` instantiate ``pattern``, reading each ``<constant>`` leaf as
    an EXISTENTIAL promise for some constant VALUE (the forall/exists Const
    doctrine)? Both sides are canonical explicit projections of canonical states, so
    exact token agreement outside the Const slots is state agreement; a ``<constant>``
    in ``pattern`` binds exactly one constlike LEAF of the subject (a numeric
    literal, a named constant, a pole, or ``<constant>`` itself). Any other
    disagreement is a plain non-match -- the caller treats that conservatively
    (the rule is NOT covered)."""

    def walk(pi: int, si: int) -> 'tuple[int, int] | None':
        if pi >= len(pattern) or si >= len(subject):
            return None
        token = pattern[pi]
        if token == '<constant>':
            leaf = subject[si]
            if leaf in _CONSTLIKE_LEAVES or is_numeric_string(leaf):
                return pi + 1, si + 1
            return None
        if token != subject[si]:
            return None
        pi, si = pi + 1, si + 1
        for _ in range(arity.get(token, 0)):
            step = walk(pi, si)
            if step is None:
                return None
            pi, si = step
        return pi, si

    return walk(0, 0) == (len(pattern), len(subject))


#: Leaves that denote a VALUE rather than a free variable.
_SPECIAL_LEAVES = {'np.pi', 'np.e', 'float("inf")', 'float("-inf")', 'float("nan")'}


def _token_of(t: Any) -> str:
    """One element of a token sequence as its token: a ``str`` as itself, a number as its
    ``str``. Anything else is refused: ``str(True)`` and ``str(None)`` are words, not the tokens
    of anything the element denotes."""
    if isinstance(t, str):
        return t
    if isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, (bool, np.bool_)):
        return str(t)
    raise ValueError(f'a token must be a str or a number, not {type(t).__name__} ({t!r})')
