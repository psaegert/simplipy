"""Core simplification engine for symbolic expressions in prefix notation.

Defines :class:`SimpliPyEngine`, which parses, converts, evaluates, and simplifies
mathematical expressions given as prefix token lists using a configurable set of
operators and rewrite rules. The engine dispatches all simplification, conversion,
validation, and mining work to the compiled Rust extension (``simplipy._core``);
the compiled core is REQUIRED; there is no pure-Python fallback.
"""
import hashlib
import importlib
import os
import warnings
from itertools import product
from types import CodeType, FunctionType
from typing import Callable
from pathlib import Path
from typing import Any, Literal, cast
from copy import deepcopy
from enum import Enum, EnumMeta

import numpy as np
import json
import yaml

from simplipy.utils import (
    is_numeric_string,
    deduplicate_rules)
from simplipy.trust import check_realization, check_root, package_for, resolve_trusted
from simplipy.io import load_config
from simplipy.asset_manager import get_path

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
    """A sorted-rule slot token: a sort sigil (``_`` / ``?`` / ``!``) followed by digits."""
    return bool(token) and token[0] in '_?!' and token[1:].isdigit()


def _coverage_variants(
        lhs: tuple[str, ...],
        rhs: tuple[str, ...]) -> list[tuple[list[str], list[str]]]:
    """Instantiation variants a rule must survive to count as covered.

    Every slot in ``lhs`` is instantiated as a distinct variable leaf ``x{i}``;
    ``<constant>`` stays LITERAL (substituting a numeral would let native constant
    folding fake coverage). Additionally, every subset of up to 3 wide slots
    (``_``/``!`` sigils) is probed with the composite ``sin(x{i})``: leaf-only
    instantiation under-tests what a wide slot claims to match.
    """
    slots = list(dict.fromkeys(t for t in lhs if _is_slot_token(t)))
    wide = [s for s in slots if s[0] in '_!'][:3]
    variants = []
    for mask in product((0, 1), repeat=len(wide)):
        bindings = {s: ['x' + s[1:]] for s in slots}
        for slot, composite in zip(wide, mask):
            if composite:
                bindings[slot] = ['sin', 'x' + slot[1:]]

        def substitute(tokens: tuple[str, ...]) -> list[str]:
            return [out for t in tokens for out in bindings.get(t, [t])]

        variants.append((substitute(lhs), substitute(rhs)))
    return variants


# The core serialization language's arities (config-independent built-ins): the
# canonical explicit projection emits these regardless of the config vocabulary,
# so any walk over projection output needs them alongside `operator_arity`.
# One mine per process, enforced (hardening H-002/H-008, 2026-08-03): the interval
# soundness counters the provenance sidecar reads are PROCESS-GLOBAL, so a second
# concurrent mine would cross-contaminate both sidecars' per-mine deltas -- and would
# only contend for the cores the first mine already saturates. Serve traffic is safe
# alongside a mine (measured: `simplify` never touches the counters).


#: Arities of the CORE SERIALIZATION LANGUAGE, read from the engine's own table rather
#: than restated here. The audit counted five hand-maintained copies of this data
#: (C1.10) -- and a copy that drifts is a soundness problem, not a tidiness one: the
#: masking walk treats an unknown operator as a LEAF, so its operands inherit the
#: enclosing bag's role and a `pow` exponent can be masked, exactly the accident the
#: role API exists to make impossible.
#: Empty only when the extension is missing, in which case engine construction raises
#: anyway (`_CORE_IMPORT_ERROR`) -- so there is never a stale literal to drift.
_CORE_SERIALIZATION_ARITIES = (
    {tok: spec['arity'] for tok, spec in _core_serialization_ops().items()}
    if _core_serialization_ops is not None else {})
# Constlike LEAF spellings a `<constant>` promise-slot may bind: named constants and
# the poles/nan, plus (checked separately) any numeric literal. NEVER a compound
# variable-free subtree: collapsing `acos(cos(C))` to one Const leaf is a strict
# serve-ordering descent (real rule content), while renaming a LEAF constant to
# `<constant>` is a complexity tie (pure abstraction -- masking's job, downstream).


# The bag delimiters and inverse-section markers of the TAGGED serialization --
# `simplify`'s default token output (`<add> ... <sub> ... </add>`). The explicit-dialect
# entry points (`is_valid`, `prefix_to_infix`) do not read this form; they use this set
# to NAME the dialect in their diagnostics instead of failing with a bare arity error (B1).
_TAGGED_DIALECT_TOKENS = frozenset({'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>'})

#: The EXACT migration for each deprecated `simplify(form=...)` value: convert first,
#: then simplify. Measured byte-identical to the parameter on every corpus row.
_FORM_RECIPE = {'tagged': 'to_tagged', 'explicit': 'to_prefix', 'infix': 'to_infix'}

#: (C1.18) Every key an operator spec must carry. `precedence` is deliberately NOT
#: required: non-core operators render as function calls, which never consult it; a
#: core serialization token missing the key gets the core table's own value (a
#: PRESENT-and-conflicting value still refuses in the core-token guard -- absence
#: used to slip past that guard entirely).
# D12 (owner-ratified 2026-08-16): `inverse` left the REQUIRED set -- nothing reads
# it (parse-only since the relic layer's deletion), and requiring pure ceremony made
# every wild config fail construction. Declared values are still accepted silently.
_REQUIRED_OPERATOR_KEYS = ('realization', 'alias', 'arity', 'commutative')


def _normalized_operator_specs(operators: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate operator specs LOUDLY and normalize them (C1.18).

    Three silent failure modes closed, none of which the suite's DEGENERATE battery
    used to build: a spec missing a key died as a bare ``KeyError: 'alias'`` naming
    neither the operator nor the file; ``commutative: true`` on a non-core operator
    was consumed by NOTHING (only ``+`` and ``*`` are AC bags, so ``f(a, b)`` and
    ``f(b, a)`` stayed two canonical states while the config claimed one value); and
    a core serialization token declared WITHOUT ``precedence:`` bypassed the C1.8
    conflicting-precedence guard, which compares only present values.
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


def _validate_ndarray_input(expression: 'np.ndarray') -> None:
    """The ndarray input contract for ``simplify`` (engine.py): 1-D, string-like dtype.
    Validated BEFORE the core call so malformed inputs fail with a clean ValueError."""
    if expression.ndim != 1:
        raise ValueError('`simplify` expects a one-dimensional numpy array of tokens')
    if expression.dtype.kind not in {'U', 'S', 'O'}:
        raise ValueError('`simplify` expects a numpy array of string-like tokens')


# C23a (D27): the mine half lives in `simplipy.mining` as RuleMiner; the
# engine keeps `find_rules`/`certify_rules` as thin delegators so the public
# API does not move. These aliases keep the module surface stable (tests and
# downstream reach ARTIFACT_ENV_SWITCHES and _MINE_LOCK here, and D28
# requires the lock to stay ONE object across both modules).
from .mining import (  # noqa: E402,F401
    ARTIFACT_ENV_SWITCHES, RuleMiner, _MINE_LOCK,
    _CONSTLIKE_LEAVES, _load_proposals, _tokens_in_vocabulary,
)

# D11 column R27 (owner-ratified 2026-08-16): ARTIFACT_ENV_SWITCHES is the
# documented machine-readable registry of artifact-affecting env switches.
__all__ = ['SimpliPyEngine', 'Mode', 'ARTIFACT_ENV_SWITCHES']


#: THE ARTIFACT THIS VERSION WAS BUILT AND TESTED AGAINST, resolved when `load()` is
#: called with no engine. Pinned HERE, in the package, and not in the hosted manifest:
#: "what do I get by default" must be answerable offline and must not change under a
#: user without a simplipy release. The manifest still supplies the files; this decides
#: WHICH ones.
#:
#: WHY A REVISION AND NOT JUST A NAME. One name can mean different rules across simplipy
#: versions -- `acj-4` mined under 0.14's judge is not `acj-4` mined under 0.15's, and
#: both are legitimately called `acj-4`. The manifest is revision-addressed, so the pair
#: (name, revision) names exactly one artifact, and each simplipy release pins the pair
#: it was tested against. `None` means "whatever the manifest currently points at", which
#: is the right answer only for a release that ships no artifact of its own.
#:
#: The measure-fingerprint check (D25) remains the safety net for an artifact loaded by
#: NAME across a measure change; this pin is what stops that happening by default.
DEFAULT_ENGINE = 'acj-4-3'
DEFAULT_ENGINE_REVISION: str | None = None


class _ModeMeta(EnumMeta):
    """Serves the retired ``SOUND``/``LOSSY`` spellings with a deprecation notice.

    They are resolved HERE rather than declared as enum aliases in the body, because a
    body alias (``SOUND = 'f64'``) is indistinguishable from the member it aliases and
    could never warn. ``__getattr__`` runs only when normal lookup fails, so live
    members never take this path.
    """


#: Leaves that denote a VALUE rather than a free variable.
_SPECIAL_LEAVES = {'np.pi', 'np.e', 'float("inf")', 'float("-inf")', 'float("nan")'}


_MODE_RENAME_WHY = (
    "`SOUND` claimed one notion of soundness where there are two incomparable ones: "
    "`f64` is sound as the deployed f64 evaluator computes, `real` is sound as "
    "mathematics defines, and neither implies the other"
)


class Mode(Enum, metaclass=_ModeMeta):
    """Which soundness the simplification preserves. An AXIS, not an ordering.

    The old ``Mode`` was an ``IntEnum`` on the premise that soundness is ordinal --
    ``EXACT <= SOUND <= AE <= LOSSY``, a higher rung permitting strictly more. Measurement
    refuted the premise. "Mathematically true" and "realised in f64" are INCOMPARABLE:
    ``inv(acosh(cosh x)) -> abs(inv x)`` is true and NOT f64-realised (the deployed
    evaluator diverges), while ``asin(1e-8) -> 1e-8`` is f64-exact and mathematically
    FALSE (the cubic term is 1.667e-25). Neither rule is "more sound" than the other,
    so there is no rung to put them on, and ``<`` between modes no longer type-checks.

    - ``f64`` (the default): sound as the DEPLOYED f64 evaluator computes. Byte-identical
      to the historical ``SOUND``. This is the inference/scoring mode -- it is what makes
      a rewrite safe for a model that will be evaluated in f64.
    - ``real``: sound as MATHEMATICS defines, independent of any float format. Use when
      the rewrite must hold symbolically -- proofs, exact-arithmetic backends, or
      publication -- and accept that some of it is not what f64 will compute.
    - ``corpus``: both of the above UNION-ed, trading soundness for recall -- every rule
      placeholder binds any subtree (the ``!``-sort finite-a.e. certificate is skipped)
      and the constant-fold's finiteness gate is relaxed (so ``<constant>/0`` collapses
      to ``<constant>``). For training-corpus canonicalisation ONLY: the training data is
      generated FROM the simplified form, so the target equals the data and there is no
      external function to violate. Do NOT use on an inference or scoring path.

    Each mode names ONE DISTINCT, COMPLETE rule set -- ``rules.json`` / ``rules_real.json``
    / ``rules_corpus.json`` -- so selecting a mode selects a file, and what is loaded IS
    what is served with nothing unioned at serve time.

    ``SOUND`` and ``LOSSY`` still resolve, with a ``DeprecationWarning``, to ``f64`` and
    ``corpus``.
    """

    f64 = 'f64'
    real = 'real'
    corpus = 'corpus'


#: THE MAP from the public ``Mode`` onto the core's RULE MODE -- the one place the two
#: vocabularies meet on the Python side, and the twin of ``RuleMode::from_wildcard_all``
#: on the Rust side. ``'default'`` is the core's name for the set in ``rules.json``.
_RULE_MODE: dict[Mode, str] = {Mode.f64: 'default', Mode.real: 'real', Mode.corpus: 'corpus'}

#: The retired STRING spellings, accepted by ``simplify(mode=...)`` with a notice. Kept
#: beside ``_ModeMeta._DEPRECATED`` deliberately: the enum path and the string path are
#: two doors onto the same rename and must not drift.


class SimpliPyEngine:
    """Manages and manipulates symbolic expressions.

    This class provides a comprehensive toolkit for parsing, transforming,
    and simplifying mathematical expressions. It operates on expressions in
    prefix notation (a list of tokens) and uses a customizable set of
    operators and simplification rules.

    Parameters
    ----------
    operators : dict[str, dict[str, Any]]
        A dictionary defining the operators. Each key is the operator's
        canonical name (e.g., 'add', 'sin'), and the value is another
        dictionary specifying its properties like 'arity', 'realization'
        (the corresponding Python function), 'inverse', etc.
    rules : list[tuple] or None, optional
        A list of simplification rules. Each rule is a tuple containing two
        lists of strings: the pattern to match and the replacement expression,
        both in prefix notation. If None, the engine is initialized with no
        rules.
    rules_real : list[tuple] or None, optional
        The ``real`` mode's OWN COMPLETE rule set -- core rules AND the real-only
        rules, in one self-contained list, NOT a supplement to ``rules``. Owner ruling
        (2026-08-19): "one distinct rule set for each mode", so what is loaded IS what
        is served and the engine never computes a union to decide what fires.
        ``None`` (the default) means this engine names no ``real`` set of its own and
        that mode serves ``rules``; ``[]`` is the different, sayable statement "the
        ``real`` mode serves nothing".
    rules_corpus : list[tuple] or None, optional
        The ``corpus`` mode's OWN COMPLETE rule set, same discipline as ``rules_real``.
        This is the set ``Mode.corpus`` serves.
    trusted_modules : list[str] or None, optional
        Extra module roots this engine may import for its operator realizations,
        on top of the defaults (``math``, ``np``, ``scipy``, ``simplipy``) and
        anything named by ``SIMPLIPY_TRUSTED_MODULES``. Importing a module runs its
        top-level code, so trust is granted HERE -- by the caller -- and never by
        the config, which could otherwise authorize itself. See :mod:`simplipy.trust`.

    Attributes
    ----------
    operator_tokens : list[str]
        A list of all defined operator names.
    operator_arity : dict[str, int]
        A mapping from operator names to their arity (number of arguments).
    simplification_rules : list[tuple]
        The DEFAULT mode's rule set, as loaded into the engine (mirrored into the
        compiled core by :meth:`compile_rules`). This one list is what
        ``Mode.f64`` serves; the other two modes' sets are separate attributes and
        never appear here, so a consumer reading this reads exactly what it is served.
    real_simplification_rules : list[tuple] or None
        The ``real`` mode's own complete rule set, or ``None`` when this engine names
        none and that mode serves ``simplification_rules``.
    corpus_simplification_rules : list[tuple] or None
        The ``corpus`` mode's own complete rule set, same convention.
    modules : list[str]
        The importable package names this engine's realizations need (``numpy`` is
        always present: ``np.pi``/``np.e`` are token grammar). These are PACKAGE
        names, not the canonical config spellings -- ``np.sin`` appears here as
        ``numpy``.
    """
    def __init__(self, operators: dict[str, dict[str, Any]], rules: list[tuple] | None = None, *,
                 rules_real: list[tuple] | None = None,
                 rules_corpus: list[tuple] | None = None,
                 trusted_modules: list[str] | None = None) -> None:
        # C1.18: loud spec validation + normalization BEFORE anything reads a key --
        # the first consumer used to be a bare `KeyError: 'alias'`.
        operators = _normalized_operator_specs(operators)
        # Cache operator metadata for quick access during parsing and evaluation.
        self.operator_tokens = list(operators.keys())
        self.operator_aliases = {alias: operator for operator, properties in operators.items() for alias in properties['alias']}

        self.operator_realizations = {k: v["realization"] for k, v in operators.items()}

        self.operator_arity = {k: v["arity"] for k, v in operators.items()}
        self.operator_arity_compat = deepcopy(self.operator_arity)
        self.operator_arity_compat['**'] = 2
        self.operators = list(self.operator_arity.keys())

        # THE REALIZATION TRUST MODEL (register C1.12, owner-ruled 2026-08-09).
        # `_realization_roots` maps each CANONICAL module spelling to the operators
        # asking for it, so a refusal can point at the config line rather than at the
        # engine; `modules` keeps its historical meaning (importable PACKAGE names,
        # numpy always included) because downstream consumers iterate it and import
        # each entry. The trusted set is resolved HERE, at construction, from the
        # caller and the environment -- never from the config -- and travels with the
        # engine through pickling so a spawn worker cannot silently widen it.
        self._realization_roots: dict[str, list[str]] = {}
        for operator, realization in self.operator_realizations.items():
            # TOTAL by construction: returns a root, returns None for the shapes that
            # reference nothing, or REFUSES. The old `realization_root(...) is not None`
            # test silently passed everything else through to code_to_lambda (B14).
            root = check_realization(str(realization), str(operator))
            if root is not None:
                self._realization_roots.setdefault(root, []).append(operator)
        self._trusted_modules = resolve_trusted(trusted_modules)
        self.modules = sorted({package_for(root) for root in self._realization_roots} | {'numpy'})
        self.import_modules()

        # The raw operator config, kept for core (re)construction.
        self._operators_config = deepcopy(operators)

        # Normalize the incoming rule list and eliminate duplicate patterns.
        #
        # ORDER (owner ruling 2026-08-18): dedup keys on the engine's INTERNAL FORM, so it
        # needs a core -- and the core is built FROM the deduplicated rules. The knot is
        # cut by the fact that the internal form is a function of the OPERATOR TABLE
        # alone: a rules-less core over this very config is a complete and correct
        # canonicaliser, and it is the cheap half of the build (nothing to intern beyond
        # the vocabulary). So: keying core, dedup, real core.
        dummy_variables = [f'x{i}' for i in range(100)]
        if rules is None:
            self.simplification_rules = []
        else:
            self._core = self._build_core(self._operators_config, [])
            self.simplification_rules = deduplicate_rules(
                rules, dummy_variables=dummy_variables, engine=self)

        # THE OTHER TWO MODES' SETS. Normalized to the same (tuple, tuple) shape as the
        # default list, and deliberately NOT deduplicated against it: each set is
        # COMPLETE and self-contained, so there is no cross-set relationship for a dedup
        # to express. Within a set, the core's own translation applies the arbiters that
        # already exist (global first-match-wins ordering, the `lhs == rhs` subsumption
        # drop), exactly as it does for the default set.
        #
        # `None` is preserved as `None` and never flattened to `[]`: `None` means "this
        # engine names no set for that mode, which therefore serves the default one",
        # `[]` means "that mode serves nothing". Collapsing them would make an empty set
        # unsayable -- the same computed-instead-of-stated mistake the triple exists to
        # remove.
        self.real_simplification_rules = self._normalized_mode_rules(rules_real)
        self.corpus_simplification_rules = self._normalized_mode_rules(rules_corpus)

        # Build the compiled core (REQUIRED; see the module docstring): every
        # construction path (from_config/load AND direct in-memory construction) attaches it
        # here, from the SAME in-memory state, so no path can exist without a core.
        self._core = self._build_core(
            self._operators_config, self.simplification_rules,
            self.real_simplification_rules, self.corpus_simplification_rules)

    @staticmethod
    def _normalized_mode_rules(rules: list[tuple] | None) -> list[tuple] | None:
        """One mode's own rule set in the engine's (tuple, tuple) shape -- ``None``
        through unchanged, because ``None`` (no set of its own) and ``[]`` (an empty
        set) are different statements at every layer."""
        if rules is None:
            return None
        return [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]

    @staticmethod
    def _build_core(operators: dict[str, dict[str, Any]], rules: list[tuple],
                    rules_real: list[tuple] | None = None,
                    rules_corpus: list[tuple] | None = None) -> Any:
        """Build the compiled core (``simplipy._core``) from in-memory config + rules.

        The core is REQUIRED: a missing extension or a load failure is a hard error --
        the pure-Python engine was removed.

        ``rules_real``/``rules_corpus`` are those modes' OWN COMPLETE sets, pushed after
        construction through the core's ``set_mode_rules``. ``None`` means the push does
        not happen at all, so a config naming neither key builds byte-for-byte the call
        this function has always made -- the no-op property holds because there is
        nothing extra to execute, not because something extra is a no-op. The pushes are
        separate statements rather than more arguments to ``from_strs`` because the mine
        driver needs exactly this surface too (it installs a freshly minted set on a live
        core), and one entry point is easier to keep honest than two.
        """
        if _RustEngine is None:
            raise ImportError(
                'simplipy._core is required (the pure-Python engine was removed): the compiled '
                f'extension failed to import ({_CORE_IMPORT_ERROR!r})')
        config_text = yaml.safe_dump({'operators': operators}, sort_keys=False)
        rules_text = json.dumps([[list(lhs), list(rhs)] for lhs, rhs in rules])
        core = _RustEngine.from_strs(config_text, rules_text)
        for mode_name, mode_rules in (('real', rules_real), ('corpus', rules_corpus)):
            if mode_rules is not None:
                core.set_mode_rules(
                    mode_name, [(list(lhs), list(rhs)) for lhs, rhs in mode_rules])
        return core

    def __getstate__(self) -> dict[str, Any]:
        """Pickle support: the engine serializes WITHOUT its compiled core.

        The core (``simplipy._core.Engine``) is a Rust object with no
        serialization surface, but it is derived state -- fully determined by
        the operator config and rule list this wrapper carries. Dropping it
        here (and rebuilding it in :meth:`__setstate__`) makes engines work
        with ``pickle``, ``copy.deepcopy`` and ``multiprocessing`` spawn
        contexts, where every worker receives the recipe and builds its own
        core. Rules pushed to the core directly (bypassing
        ``simplification_rules`` + :meth:`compile_rules`) are not part of the
        recipe, matching the documented mutation contract.

        The evaluation namespace goes the same way: module objects have no pickle
        surface, and it is derived state -- :meth:`import_modules` rebuilds it in the
        worker from the same recipe, re-running the trust check there rather than
        carrying imported modules across the process boundary.

        C1.9: the recipe is REFUSED when it is ambiguous. The core echoes a digest of
        the rules it was actually given (:meth:`compile_rules` or construction), and a
        mismatch with ``simplification_rules`` here means the two have diverged -- the
        list was mutated without :meth:`compile_rules`, or the core's rules were set
        directly. A pickle rebuilds the worker from the LIST, so a diverged parent
        would silently spawn workers running different rules (measured before the fix:
        50/400 corpus rows simplified differently between a parent and its own
        unpickled worker, with no warning). Loud beats silent: sync first, then pickle.
        """
        if not self._core.rules_in_sync(
                [(list(lhs), list(rhs)) for lhs, rhs in self.simplification_rules]):
            raise ValueError(
                'cannot pickle a SimpliPyEngine whose compiled core and '
                'simplification_rules disagree (the list was mutated without '
                'compile_rules(), or rules were pushed to the core directly): the '
                'pickle would rebuild workers from the list and parent and workers '
                'would silently run different rule sets. Call compile_rules() to sync '
                'the core to the list first.')
        state = self.__dict__.copy()
        del state['_core']
        state.pop('_eval_namespace', None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the engine from pickled state, exactly as ``__init__`` would.

        Mirrors the construction order: realization modules first (a spawn
        worker unpickles into a fresh interpreter), then the compiled core from
        the same in-memory config + rules. The trusted set travels IN the state, so
        a worker imports exactly what the parent was allowed to import -- widening
        trust needs the caller or the environment, in the parent, as everywhere else.
        """
        self.__dict__.update(state)
        self.import_modules()
        # The other two modes' sets travel IN the state (plain attributes), so the recipe
        # a worker unpickles is complete. `setdefault(..., None)` keeps pickles written
        # before the triple existed loadable: they rebuild with no set of their own,
        # which is exactly what they were.
        self._core = self._build_core(
            self._operators_config, self.simplification_rules,
            self.__dict__.setdefault('real_simplification_rules', None),
            self.__dict__.setdefault('corpus_simplification_rules', None))

    def compile_rules(self) -> None:
        """Sync the compiled core's rule set from ``self.simplification_rules``.

        Public contract (unchanged since the pure-Python engine): after mutating
        ``self.simplification_rules``, call this so ``simplify`` sees the new rules.

        COPY-ON-WRITE (conc-1): a fresh core is built from the in-memory state and
        the reference is swapped atomically -- an in-place ``set_rules`` push
        mutated the core under a concurrent reader's feet mid-``simplify``. A
        reader holding the old core finishes on a consistent (old) ruleset; the
        ~20 ms rebuild is mutation-API cost only, never on the hot path.

        The other two modes' sets are re-pushed alongside: a fresh core starts with no
        set of its own for either, so NOT re-pushing them would silently retract both on
        every default-rule sync.
        """
        self._core = self._build_core(
            self._operators_config, self.simplification_rules,
            self.real_simplification_rules, self.corpus_simplification_rules)

    def _replace_rules(self, rules: list) -> None:
        """Build-first-or-unchanged (conc-2): compile a fresh core from the CANDIDATE
        ruleset, and only on success assign both the list and the core. The shim
        mutators used to rewrite ``simplification_rules`` before pushing, so a failed
        push left the wrapper and the core silently diverged; this turns that into
        loud-and-unchanged."""
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
        new_core = self._build_core(
            self._operators_config, rules,
            self.real_simplification_rules, self.corpus_simplification_rules)
        self.simplification_rules = rules
        self._core = new_core

    def prune_covered_rules(self, verbose: bool = False) -> int:
        """Remove rules that the remaining rules already cover behaviorally.

        A rule ``(lhs, rhs)`` is *covered* if the engine WITHOUT it still
        reaches, on every instantiation variant of its ``lhs``, a state at most
        the promised ``rhs``-variant in the serve-time reduction ordering
        (semantic complexity, literal size, canonical total order) -- the same
        ordering the rewrite pass fires under and mint acceptance judges by
        (ONE coverage ordering, everywhere). Equality in that total order is
        state identity, so "covered" means the probe engine literally reaches
        the promise or better, never merely something that SCORES the same.
        Variants: each slot instantiated as a
        distinct variable leaf ``x{i}``, with ``<constant>`` kept LITERAL
        (native constant folding must not fake coverage), plus every subset of
        up to 3 wide slots (``_``/``!`` sigils) probed with the composite
        ``sin(x{i})`` -- leaf-only instantiation under-tests wide-sort claims.
        A rule is covered only if ALL variants pass.

        Batch remove-and-repair, longest sources first: rules are processed in
        source-length waves DESCENDING (down to length 3). Per wave, the whole
        wave is tentatively removed, then repaired to a fixpoint: a probe
        engine is built from the kept rules in ORIGINAL rule-list order
        (first-match-wins makes rule order behavioral, so set iteration order
        must never leak into the build), every still-removed rule is
        re-verified against it, failures are re-added; capped at 12 repair
        rounds per wave, and on cap the wave's remaining removals are
        abandoned (re-added), so every removal stands verified against the
        final rule set. Greedy: the result is valid, not necessarily minimal.

        Removes ANY rule (pattern rules included) that the other rules cover
        compositionally under a <=-length criterion. (Its former sibling
        ``prune_redundant_rules`` -- explicit rules shadowed by pattern rules
        under an equality criterion -- died with the legacy kernel; this is
        the one prune.)

        Pruning is intentionally corpus-free. Pair it with a closure-quality
        check on a benchmark corpus of your own (verify outputs do not
        lengthen, e.g. fail if more than 0.5% of expressions get LONGER
        outputs than with the unpruned engine) before deploying a pruned
        ruleset; that check needs a benchmark corpus and stays OUTSIDE this
        method.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints per-wave progress and a summary. Defaults to False.

        Returns
        -------
        int
            The number of rules that were pruned.
        """
        full = [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.simplification_rules]
        if not full:
            return 0

        arity = {**_CORE_SERIALIZATION_ARITIES, **self.operator_arity}

        def covered(core: Any, lhs: tuple[str, ...], rhs: tuple[str, ...]) -> bool:
            # State-coverage in the serve-time reduction ordering: the probe engine
            # must reach at most the promised rhs-state -- probe_out <= rhs, spelled
            # !(rhs < probe_out) in the total order. The complexity-score judge this
            # replaces was blind on the tie tier (equal-score DIFFERENT states:
            # `abs neg ?0 -> abs ?0` read 2 <= 2 "covered" while nothing performed
            # the rewrite), deleting rules mint acceptance had certified as strict
            # improvements. One tier needs more than the ordering: `<constant>` in a
            # promise is EXISTENTIAL, so a probe reaching an INSTANCE of the promise
            # (each Const slot bound to a ground subterm -- `0` fulfills
            # `<constant>`, `* 2.71... x0` fulfills `* <constant> x0`) has fulfilled
            # it, though the total order sees only unrelated leaf kinds there.
            # (ac_judge/ac_ordered_below: the AC parser is the arbiter, no
            # config-vocabulary gate; an undecidable ordering raises.)
            def fulfilled(variant_lhs: list[str], variant_rhs: list[str]) -> bool:
                probe_out = core.ac_judge(variant_lhs, 5)[2]
                if not core.ac_ordered_below(list(variant_rhs), probe_out):
                    return True
                if '<constant>' in variant_rhs:
                    promise = core.ac_judge(list(variant_rhs), 0)[2]
                    return _fulfills_constant_promise(promise, probe_out, arity)
                return False

            return all(fulfilled(variant_lhs, variant_rhs)
                       for variant_lhs, variant_rhs in _coverage_variants(lhs, rhs))

        kept = set(full)
        for wave_length in range(max(len(lhs) for lhs, _ in full), 2, -1):
            wave = [rule for rule in full if rule in kept and len(rule[0]) == wave_length]
            if not wave:
                continue
            kept -= set(wave)
            pending = set(wave)
            rounds = 0
            while True:
                rounds += 1
                core = self._build_core(self._operators_config, [rule for rule in full if rule in kept])
                readd = [rule for rule in pending if not covered(core, *rule)]
                if not readd:
                    if verbose:
                        print(f'Wave length {wave_length}: removed {len(pending)} / {len(wave)} '
                              f'rules (round {rounds}, stable)')
                    break
                kept |= set(readd)
                pending -= set(readd)
                if verbose:
                    print(f'Wave length {wave_length} round {rounds}: re-added {len(readd)}, '
                          f'still removed {len(pending)}')
                if rounds >= 12:
                    # Fail-safe on the repair-round cap: the still-pending removals were
                    # last verified against a now-superseded kept set, so abandon them
                    # (re-add) rather than ship a removal that was never verified against
                    # the final rule set.
                    if verbose and pending:
                        print(f'Wave length {wave_length}: repair-round cap reached; '
                              f're-added {len(pending)} unverified removals')
                    kept |= pending
                    pending = set()
                    break

        self._replace_rules([rule for rule in full if rule in kept])
        n_pruned = len(full) - len(self.simplification_rules)
        if verbose:
            print(f'Pruned {n_pruned} covered rules '
                  f'({len(self.simplification_rules)} rules remaining)')
        return n_pruned

    def resolve_constant_rules(self, verbose: bool = False) -> int:
        """Replace ``<constant>`` with the actual numeric value in all-numeric rules.

        Some rules discovered by :meth:`find_rules` map an expression whose
        leaves are all numeric literals to the generic ``<constant>`` token
        (e.g. ``mult2(1) → <constant>``).  This method evaluates every such
        rule and replaces the ``<constant>`` replacement with the concrete
        numeric result (e.g. ``mult2(1) → 2``), allowing downstream constant
        folding to continue folding with exact values.

        Only rules whose left-hand side leaves **all** pass
        :func:`~simplipy.utils.is_numeric_string` are affected.  Rules
        involving named constants (``np.e``, ``np.pi``, ``(-1)``), generic
        ``<constant>`` placeholders, or wildcard variables are left unchanged.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints each resolved rule.  Defaults to False.

        Returns
        -------
        int
            The number of rules whose replacement was resolved.
        """
        n_resolved = 0
        candidate = list(self.simplification_rules)  # conc-2: mutate a copy, swap on success
        for i, rule in enumerate(candidate):
            lhs, rhs = rule
            if tuple(rhs) != ('<constant>',):
                continue
            leaves = [t for t in lhs if t not in self.operator_arity]
            if not leaves or not all(is_numeric_string(t) for t in leaves):
                continue
            result_token = self._core.evaluate_constant_subtree(list(lhs))
            if result_token is not None:
                if verbose:
                    print(f'Resolved: {list(lhs)} -> ["{result_token}"] (was ["<constant>"])')
                candidate[i] = (tuple(lhs), (result_token,))
                n_resolved += 1
        if n_resolved > 0:
            self._replace_rules(candidate)
        if verbose:
            print(f'Resolved {n_resolved} constant rules '
                  f'({len(self.simplification_rules)} rules total)')
        return n_resolved

    def import_modules(self) -> None:
        """Import the modules this engine's operator realizations need.

        Each realization's leading component is checked against the trusted set
        BEFORE anything is imported (importing runs top-level code, so the check has
        to come first), then imported into this engine's own evaluation namespace --
        not into shared module globals, so engines cannot contaminate each other's
        expression namespaces. ``np`` is always bound, because ``np.pi`` and ``np.e``
        are token grammar and may appear in any expression regardless of config.

        Raises
        ------
        simplipy.trust.UntrustedModuleError
            If a realization names a module root that is neither trusted by default
            nor opted into via ``trusted_modules=`` or ``SIMPLIPY_TRUSTED_MODULES``.
            See :mod:`simplipy.trust` for why the opt-in cannot live in the config.
        """
        namespace: dict[str, Any] = {'np': np}
        for root, operators in sorted(self._realization_roots.items()):
            check_root(root, operators, self._trusted_modules)
            namespace[root] = importlib.import_module(package_for(root))
        self._eval_namespace = namespace

    @classmethod
    def from_config(cls, config_path: str, *, trusted_modules: list[str] | None = None) -> "SimpliPyEngine":
        """Creates a SimpliPyEngine instance from a YAML configuration file.

        The configuration file should specify the `operators` and can
        optionally provide a path to a `rules` file.

        Parameters
        ----------
        config_path : str
            The absolute or relative path to the YAML configuration file.
        trusted_modules : list[str] or None, optional
            Extra module roots the config is allowed to import (see
            :mod:`simplipy.trust`). Deliberately an argument HERE and not a key in
            the config: a config that granted itself permission would grant nothing.

        Returns
        -------
        SimpliPyEngine
            A new instance of the engine configured as per the file.
        """
        config_path = os.path.abspath(config_path)
        config = load_config(config_path)
        # ARTIFACT-GENERATION GATE (owner ruling 2026-08-03): generation-1 artifacts
        # (retired hyper-operator vocabulary) serve only on simplipy <= 0.11; refusing
        # here -- the choke point of every artifact-loading path, including
        # SimpliPyEngine.load -- is what makes the compatibility matrix TRUE. Raw
        # in-memory construction stays open (see simplipy.compat).
        from .compat import check_config
        check_config(config, source=config_path)

        def resolve_artifact_path(declared: str) -> str:
            """A config-declared artifact path: relative entries resolve against the
            CONFIG's own directory, absolute entries are taken as given. ONE function, so
            the three rule keys cannot drift apart -- "the same relative-path handling"
            is then a property of the code, not of a comment."""
            if os.path.isabs(declared):
                return declared
            return os.path.normpath(os.path.join(os.path.dirname(config_path), declared))

        rules = []
        rules_path = None
        rules_file = config.get('rules')
        # Why the engine ends up with the rules it has -- carried to the state check
        # below so one warning can name both the OUTCOME and its CAUSE.
        cause: str
        if not rules_file:
            cause = "the config declares no 'rules' key"
        else:
            rules_path = resolve_artifact_path(rules_file)
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules = json.load(f)
                cause = f"the rules file '{rules_path}' contains none"
            else:
                # BOTH spellings, deliberately: the literal value says what the config
                # asked for, the absolute path says where the engine actually looked.
                # Neither alone is enough to fix a broken artifact layout.
                cause = (f"the configured rules path {rules_file!r} could not be "
                         f"resolved (looked for '{rules_path}')")
                rules_path = None

        # THE OTHER TWO THIRDS OF THE TRIPLE (`rules_real:` / `rules_corpus:`), read by
        # the same resolver as `rules:` above. Each names a COMPLETE, self-contained set
        # for its mode -- not a supplement to `rules.json` -- so what is loaded is what
        # is served (owner ruling, 2026-08-19). Only triples are mined and distributed;
        # the keys are nonetheless OPTIONAL here, because every artifact that shipped
        # before the ruling is a single file and must keep loading unchanged.
        #
        # An ABSENT key leaves the set at `None`, i.e. that mode serves `rules.json` --
        # byte-for-byte the engine this loader has always built. A configured-but-MISSING
        # file is deliberately NOT an error: an engine without it is fully functional and
        # simply serves that mode the default set, and refusing would make a
        # partially-synced asset directory unloadable. It DOES warn, for the same reason
        # the missing `rules:` file does -- a silently-ignored artifact path is how a
        # broken layout survives to production.
        #
        # An EMPTY file is honoured as an empty set, not as an absent one: `[]` says
        # "this mode serves nothing" and the loader has no business overruling it.
        mode_rules: dict[str, list | None] = {'real': None, 'corpus': None}
        for mode_name in mode_rules:
            declared = config.get(f'rules_{mode_name}')
            if not declared:
                continue
            mode_path = resolve_artifact_path(declared)
            if os.path.exists(mode_path):
                with open(mode_path, 'r') as f:
                    mode_rules[mode_name] = json.load(f)
            else:
                warnings.warn(
                    f"the config '{config_path}' declares a {mode_name!r}-mode rule set "
                    f"{declared!r} that could not be resolved (looked for "
                    f"'{mode_path}'): the engine is built WITHOUT it, so Mode "
                    f"{mode_name!r} serves the default rule set instead. The other "
                    f"modes are unaffected -- each mode's set is its own file.",
                    UserWarning)
        engine = cls(operators=config['operators'], rules=rules,
                     rules_real=mode_rules['real'], rules_corpus=mode_rules['corpus'],
                     trusted_modules=trusted_modules)
        # WARNING ON THE RESULTING STATE (owner ruling 2026-08-18: "Warning on engine
        # without rules"), broadened from the missing-file case it replaces: an engine
        # that ends up with zero rules says so however it got there. NON-FATAL by the
        # same ruling -- such an engine is fully functional, it simply never rewrites,
        # and that is exactly why the silence was dangerous.
        #
        # Deliberately scoped to the CONFIG-DRIVEN path (this classmethod, and `load`,
        # which funnels through it): direct `SimpliPyEngine(operators=..., rules=[])`
        # construction is the sanctioned bare-engine idiom and the caller asked for it
        # in as many words, so warning there would be crying wolf.
        if not engine.simplification_rules:
            warnings.warn(
                f"the engine built from '{config_path}' has NO simplification rules: "
                f"{cause}. It will still parse, evaluate and return expressions in "
                f"canonical form, but no simplification rule can ever fire.",
                UserWarning)
        # (The compiled core self-attaches in __init__ from the SAME in-memory state.)
        # D25 (R6's warn-on-mismatch half): the provenance sidecar records the measure
        # fingerprint the ruleset was MINED under, and until this check nothing ever
        # read it back -- recording-without-reading is the inert-control pattern. A
        # ruleset served under a different measure than it was mined with is exactly
        # what the fingerprint exists to catch; warn loudly, do not refuse (the rules
        # stay sound -- their MINIMALITY claims are what the measure change voids).
        if rules_path is not None:
            sidecar_path = rules_path + '.provenance.json'
            if os.path.exists(sidecar_path):
                try:
                    with open(sidecar_path, 'r') as f:
                        recorded = json.load(f).get('measure', {}).get('digest')
                except (OSError, ValueError):
                    recorded = None
                if recorded is not None:
                    current = engine._measure_fingerprint()['digest']
                    if recorded != current:
                        warnings.warn(
                            f"measure fingerprint mismatch: the ruleset at '{rules_path}' was "
                            f"mined under measure digest {recorded} but this engine computes "
                            f"{current}. The rules stay sound, but their minimality/ordering "
                            f"claims were certified under a different measure (D25/R6).",
                            UserWarning)
        return engine

    @classmethod
    def load(cls, engine: str | None = None, install: bool = False,
             local_dir: Path | str | None = None, repo_id: str | None = None,
             manifest_filename: str | None = None, *,
             trusted_modules: list[str] | None = None) -> "SimpliPyEngine":
        """Loads a pre-defined engine configuration from the asset manager.

        This provides a convenient way to load standard engine configurations
        distributed with the `simplipy` package.

        Parameters
        ----------
        engine : str or None, optional
            The NAME of an official engine artifact (e.g. ``'acj-4-3'``). Defaults to
            :data:`simplipy.DEFAULT_ENGINE`, the artifact this simplipy version was
            built and tested against; when it is used implicitly, the choice is
            announced, because a silent default is one a user cannot reproduce.
            Replaces the old ``path`` argument, which named neither a path nor a
            version.
        install : bool, optional
            If True, forces the download of the asset if not found locally.
            Defaults to False.
        local_dir : Path or str or None, optional
            A local directory to search for the assets. Defaults to None,
            which uses the default asset directory.
        repo_id : str or None, optional
            The Hugging Face repository ID where the manifest is stored. If None, the default repository ID is used.
        manifest_filename : str or None, optional
            The filename of the manifest file. If None, the default filename is used.
        trusted_modules : list[str] or None, optional
            Extra module roots the fetched config is allowed to import (see
            :mod:`simplipy.trust`). A hosted asset is third-party input like any
            other config: without this it may only name the default roots.

        Returns
        -------
        SimpliPyEngine
            A new instance of the engine.
        """
        if engine is None:
            engine = DEFAULT_ENGINE
            # Deferred: `__version__` lives in the package __init__, which imports THIS
            # module, so it cannot be imported at module level.
            from . import __version__
            # ANNOUNCED, and only on the implicit path. A default a user did not choose
            # is one they cannot reproduce unless they are told which it was.
            print(f"simplipy: loading the default engine {engine!r}"
                  + (f" @ {DEFAULT_ENGINE_REVISION}" if DEFAULT_ENGINE_REVISION else "")
                  + f" (simplipy {__version__}); pass engine=... to choose another")
        # Known generation-1 names refuse BEFORE any download (the config-level gate
        # in from_config covers everything else, including local paths).
        if not os.path.exists(engine):
            from .compat import check_asset_name
            check_asset_name(engine)
        return cls.from_config(
            get_path(engine, install=install, local_dir=local_dir, repo_id=repo_id,
                     manifest_filename=manifest_filename),
            trusted_modules=trusted_modules)

    def is_valid(self, prefix_expression: 'str | list[str] | tuple[str, ...] | np.ndarray',
                 verbose: bool = False) -> bool:
        """Checks if an expression is syntactically valid, in ANY of the three forms.

        An expression is valid if it reads as a well-formed expression over THIS
        engine's vocabulary: every operator gets exactly its arity of operands, a
        single root remains, and every tag section closes.

        All three external forms are accepted and VERIFIED (ruling 2026-08-18);
        the form is detected exactly as the conversion API detects it (D2), by
        TYPE first and then by dialect:

        * an infix ``str`` (``'2*atan(x0)/3'``) -- read, then verified;
        * an explicit binary-prefix token sequence (``['*', '2', 'atan', 'x0']``);
        * a TAGGED token sequence (``['<mul>', '2', 'atan', 'x0', '</mul>']``) --
          the engine's own native serialization, which this predicate used to
          refuse with a False verdict rather than read.

        A ``list``, ``tuple`` or 1-D ``ndarray`` is a token sequence; tag-bearing
        sequences are verified by the shared liberal parser (the arbiter for that
        dialect, exactly as in :meth:`simplify`), everything else by the arity
        oracle. Note the difference from :meth:`read_infix`, which TOLERATES
        undeclared vocabulary: ``read_infix('sqrt(x0)')`` reads, while
        ``is_valid('sqrt(x0)')`` is False if this engine declares no ``sqrt`` --
        this method verifies against the vocabulary rather than merely reading.

        Malformed input is a False VERDICT, never an exception -- that is what a
        predicate promises. Only a type that is no expression at all raises.

        One inherited tolerance is worth knowing about on the infix path: the
        reader silently DROPS unmatched OPENING parentheses (``'((x0'`` reads as
        ``x0``), so such a string validates. Unmatched CLOSING parentheses survive
        as a stray token and are rejected. This method reports on what the reader
        reads rather than keeping a second, partial infix grammar of its own.

        Parameters
        ----------
        prefix_expression : str or list[str] or tuple[str, ...] or np.ndarray
            The expression, in any of the three forms.
        verbose : bool, optional
            If True, prints the reason for invalidity. Defaults to False.

        Returns
        -------
        bool
            True if the expression is valid, False otherwise.

        Raises
        ------
        TypeError
            If the input is neither an infix ``str`` nor a token sequence.
        """
        tokens = self._validation_tokens(prefix_expression)
        if tokens is None:
            # The infix reader refused the string outright: nothing to verify, and
            # nothing the diagnostic printer could say about tokens that never were.
            if verbose:
                print(f'Invalid expression {prefix_expression!r}: not a readable '
                      f'infix expression')
            return False
        if _TAGGED_DIALECT_TOKENS.intersection(tokens):
            # TAGGED: the shared AC parser is the arbiter for this dialect (the arity
            # oracle has no reading of the bag delimiters and would reject every one
            # of them). A budget-0 projection parses and serializes WITHOUT running
            # the rewrite loop, so this is a pure well-formedness question; malformed
            # input fails the parse and the FFI raises, which is the False verdict.
            try:
                self._core.ac_simplify(tokens, 0, False, 'tagged')
            except ValueError:
                if verbose:
                    self._explain_invalid(tokens)
                return False
            return True
        # EXPLICIT (and everything else): the verdict ALWAYS comes from the compiled core
        # (single implementation); the pure-Python loop below only survives as a DIAGNOSTIC
        # printer for the (rare) verbose call, and only runs when the core has already
        # rejected the expression. This is the pre-existing path, unchanged, so every call
        # that could be written before this method grew its other two forms answers exactly
        # as it did.
        valid = self._core.is_valid(tokens)
        if verbose and not valid:
            self._explain_invalid(tokens)
        return valid

    def _validation_tokens(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray') -> 'list[str] | None':
        """Read :meth:`is_valid`'s input into tokens, mirroring the conversion API's
        detection (D2). Returns ``None`` when an infix ``str`` does not read at all --
        an invalid expression, not an error. Refuses a non-expression TYPE loudly."""
        if isinstance(expression, str):
            try:
                return self._core.parse(expression, True, False)
            except ValueError:
                return None
        if isinstance(expression, np.ndarray):
            if expression.ndim != 1 or expression.dtype.kind not in {'U', 'S', 'O'}:
                raise TypeError(
                    'is_valid expects an infix str or a token sequence '
                    '(list, tuple or 1-D ndarray of string-like tokens)')
            return [str(t) for t in expression.tolist()]
        if isinstance(expression, (list, tuple)):
            return list(expression)
        raise TypeError(
            f'is_valid expects an infix str or a token sequence '
            f'(list, tuple or 1-D ndarray), not {type(expression).__name__}')

    def _explain_invalid(self, prefix_expression: list[str]) -> None:
        """Print WHY an expression failed :meth:`is_valid` (diagnostics only, no verdict)."""
        stack: list[str] = []
        tagged = _TAGGED_DIALECT_TOKENS.intersection(prefix_expression)
        if tagged:
            # Without this check the walk below misreads the bag delimiters as variables
            # and prints 'Variable must be leaf node' -- actively misleading (B1). is_valid
            # now READS this dialect, so reaching here means the sequence is malformed AS
            # tagged, not that the dialect was refused.
            print(f'Invalid expression {prefix_expression}: malformed '
                  f'tagged-serialization sequence (carries {sorted(tagged)}). Every '
                  f'section must open and close in order and every operator must get '
                  f"its operands; re-project a valid expression with "
                  f"simplify(expr, form='explicit') or to_tagged(expr).")
            return
        if len(prefix_expression) > 1 and prefix_expression[0] not in self.operator_arity:
            print(f'Invalid expression {prefix_expression}: Variable must be leaf node')
            return
        for token in reversed(prefix_expression):
            if token != '<constant>' and is_numeric_string(token):
                try:
                    float(token)
                except ValueError:
                    print(f'Invalid token {token} in expression {prefix_expression}')
                    return
            if token in self.operator_arity:
                if len(stack) < self.operator_arity[token]:
                    print(f'Not enough operands for operator {token} in expression {prefix_expression}')
                    return
                for _ in range(self.operator_arity[token]):
                    stack.pop()
            stack.append(token)
        if len(stack) != 1:
            print(f'Stack is not empty after parsing the expression {prefix_expression}')

    def prefix_to_infix(self, tokens: list[str], power: Literal['func', '**'] = 'func', realization: bool = False) -> str:
        """Converts a prefix expression to an infix string with minimal parentheses.

        Parameters
        ----------
        tokens : list[str]
            The prefix expression to render.
        power : {'func', '**'}, optional
            Controls how power operators are emitted. ``'func'`` keeps canonical
            engine names such as ``pow3(x)``, while ``'**'`` renders Python-style
            exponentiation.
        realization : bool, optional
            If True, operator tokens are replaced with their runtime
            realizations (for example, ``'sin'`` becomes ``'np.sin'``), so the
            output can be compiled directly.

        Returns
        -------
        str
            The formatted infix expression.

        Raises
        ------
        ValueError
            If the provided tokens do not form a well-formed prefix expression --
            including the TAGGED serialization (``simplify``'s default token output),
            which this converter does not read: render the canonical state with
            ``simplify(expr, form='infix')``, or request the binary-chain form with
            ``simplify(expr, form='explicit')``.
        """
        tagged = _TAGGED_DIALECT_TOKENS.intersection(tokens)
        if tagged:
            raise ValueError(
                f"prefix_to_infix reads the explicit binary-prefix dialect, but the input "
                f"carries tagged-serialization tokens {sorted(tagged)} (simplify's default "
                f"token output). Render the canonical state directly with "
                f"simplify(expr, form='infix'), or request the binary-chain form with "
                f"simplify(expr, form='explicit').")
        return self._core.prefix_to_infix(list(tokens), power, realization)

    def infix_to_prefix(self, infix_expression: str) -> list[str]:
        """Converts an infix expression string to prefix notation.

        This method uses a standard algorithm (related to Shunting-yard) to
        parse the infix string, respecting operator precedence and parentheses.

        Parameters
        ----------
        infix_expression : str
            The mathematical expression in infix notation.

        Returns
        -------
        list[str]
            A list of tokens representing the expression in prefix notation.
        """
        # Regex to tokenize expression properly (handles floating-point numbers and scientific notation)
        return self._core.infix_to_prefix(infix_expression)

    def convert_expression(self, prefix_expr: list[str]) -> list[str]:
        """Normalizes an expression into the engine's standard internal format.

        This method performs several key conversions:
        1.  Converts standard binary operators like `**` into the engine's
            unary power operators (e.g., `pow2`, `pow1_3`).
        2.  Combines chained power operators (e.g., `pow2(pow3(x))` becomes
            `pow6(x)`).
        3.  Handles unary negation, applying it directly to numbers where
            possible.

        Parameters
        ----------
        prefix_expr : list[str]
            The prefix expression to convert.

        Returns
        -------
        list[str]
            The normalized prefix expression.
        """
        return self._core.convert_expression(list(prefix_expr))

    def read_infix(
            self,
            infix_expression: str,
            convert_expression: bool = True,
            mask_numbers: bool = False) -> list[str]:
        """Read an infix string into the explicit binary-prefix dialect, TOLERANTLY
        and PRESERVING ITS SPELLING.

        The name states the contract because the contract is a CAPABILITY that no
        other public entry offers (renamed from ``parse``, ruling 2026-08-18):

        * **TOLERANT of unknown vocabulary.** A function this engine's config does
          not declare survives as a bare leaf --
          ``read_infix('sqrt(x0)') == ['sqrt', 'x0']`` -- so a corpus written in a
          wider vocabulary can be read first and desugared afterwards. Every
          conversion entry (:meth:`to_prefix`, :meth:`to_infix`, :meth:`to_tagged`)
          and :meth:`simplify` REFUSE such input: they parse into the canonical
          state, which only knows declared operators.
        * **SPELLING-PRESERVING: it does NOT canonicalise.** This is a reader, not a
          simplification -- it never enters the canonical state, so like terms are not
          collected and constants are not folded::

              read_infix('x0+x0')            ==  ['+', 'x0', 'x0']   # as written
              to_prefix('x0+x0')             ==  ['+', 'x0', 'x0']   # notation only
              simplify(read_infix('x0+x0'))  ==  ['*', '2', 'x0']    # canonical state

          Since the conversion/simplification split the CONVERSIONS preserve the
          spelling too -- they are notation, not content -- so the contrast that
          remains is with :meth:`simplify`, the one entry that canonicalises.
          ``read_infix``'s exclusive capability is the vocabulary TOLERANCE above;
          :meth:`to_prefix` refuses what this reader accepts.

        Mechanically it is :meth:`infix_to_prefix` plus optional
        ``convert_expression`` normalization, with a ``remove_pow1`` cleanup that
        drops redundant ``pow1_1`` occurrences.

        Parameters
        ----------
        infix_expression : str
            The mathematical expression in infix notation.
        convert_expression : bool, optional
            If True, the expression is normalized using `convert_expression`
            (``**`` becomes the engine's power operators). Defaults to True.
        mask_numbers : bool, optional
            If True, all numerical literals in the expression are replaced
            with a generic '<constant>' token. Defaults to False.

        Returns
        -------
        list[str]
            The processed prefix expression after conversion, masking (if
            enabled), and `remove_pow1` cleanup.
        """

        return self._core.parse(infix_expression, convert_expression, mask_numbers)

    def _conversion_input(
            self, expression: str | list[str] | tuple[str, ...] | np.ndarray
    ) -> tuple[str, list[str]]:
        """``(form, tokens)`` -- the shared input reading of the conversion API.

        A ``str`` is INFIX and is read with the tolerant reader; a list, tuple or 1-D
        ``ndarray`` is a token sequence, TAGGED when it carries a bag delimiter or an
        inverse-section marker and the explicit binary PREFIX otherwise. Any other type
        refuses loudly. The reading does not validate -- each conversion validates
        through the one arbiter, the syntactic parser in ``rust/forms.rs``.
        """
        if isinstance(expression, str):
            return 'infix', self._core.parse(expression, True, False)
        if isinstance(expression, np.ndarray):
            if expression.ndim != 1:
                raise ValueError('conversion expects a one-dimensional numpy array of tokens')
            if expression.dtype.kind not in {'U', 'S', 'O'}:
                raise ValueError('conversion expects a numpy array of string-like tokens')
            tokens = [str(t) for t in expression.tolist()]
        elif isinstance(expression, (list, tuple)):
            tokens = [str(t) for t in expression]
        else:
            raise TypeError(
                f'conversion expects an infix str or a token sequence '
                f'(list, tuple or 1-D ndarray), not {type(expression).__name__}')
        # REALIZATION is a fourth NOTATION, not a fourth semantics: the same expression
        # with each operator spelled as the callable it runs (`neg` ->
        # `simplipy.operators.neg`). Detected, never guessed -- only realizations that
        # DIFFER from their operator token can identify the form, and those are dotted
        # names no operator vocabulary contains. Operators realized as themselves (`+`,
        # `-`, `*`) make the two spellings identical, so there is nothing to detect and
        # nothing to get wrong.
        if self._realization_only_tokens().intersection(tokens):
            tokens = self.realizations_to_operators(tokens)
        return ('tagged' if _TAGGED_DIALECT_TOKENS.intersection(tokens) else 'prefix'), tokens

    def evaluate_constants(
            self, expression: str | list[str] | tuple[str, ...] | np.ndarray,
            form: str | None = None) -> str | list[str]:
        """Fold every variable-free subtree to its f64 value -- EXPLICITLY, on request.

        This is the front door for the numeric evaluation :meth:`simplify` refuses to
        do. The engine folds only what it computes EXACTLY (rational arithmetic, in the
        constructors); the serve path's last f64-evaluation arm was deleted on
        2026-08-02 because "an f64 evaluation landing exactly on a cheap literal while
        truly differing at 1e-17" is not a simplification, it is a rounding. So
        ``simplify(['tan', '1'])`` returns ``tan 1`` in every mode, and this returns
        ``1.5574077246549023``.

        Having it as its own method is the whole point: the capability exists, it is
        named for what it does, it is opt-in, and it can never contaminate a canonical
        form. Someone who wants a number asks for one; someone canonicalising a training
        corpus does not get one by accident.

        A subtree is foldable when it contains no variable and no slot -- a
        ``<constant>`` is a FITTED degree of freedom, not a value, so a subtree carrying
        one is left alone. Folding is MAXIMAL: the largest foldable subtrees go first,
        so ``tan(1) + x`` folds the ``tan(1)`` and stops.

        A subtree whose value is NOT FINITE is left unevaluated. Substituting
        ``float("inf")`` or ``float("nan")`` would inject a non-finite token into an
        expression that did not have one, which is the defect class that put a
        non-finite guard into flash-ansr; refusing is the safe direction and the caller
        can still evaluate the expression themselves.

        NEVER called by :meth:`simplify`, in any mode. Returns the input's own dialect
        unless ``form`` says otherwise, exactly as :meth:`simplify` does.
        """
        in_form, tokens = self._conversion_input(expression)
        if not tokens:
            return '' if in_form == 'infix' else []
        prefix = (self._core.to_prefix_syntactic(tokens) if in_form == 'tagged'
                  else self._core.parse(expression, True, False) if in_form == 'infix'
                  else tokens)
        self._core.check_form(prefix)

        arity = self.operator_arity

        def span(i: int) -> int:
            j = i + 1
            for _ in range(arity.get(prefix[i], 0)):
                j = span(j)
            return j

        def foldable(sub: list[str]) -> bool:
            return not any(t == '<constant>' or t[:1] in '_!?$' and t[1:].isdigit()
                           or (t.startswith('x') and t[1:].isdigit()) for t in sub)

        out: list[str] = []
        i = 0
        while i < len(prefix):
            j = span(i)
            sub = prefix[i:j]
            if len(sub) > 1 and foldable(sub):
                try:
                    value = float(np.asarray(
                        self._core.evaluate_batch(sub, ['x0'], [0.0], 1, [])).ravel()[0])
                except Exception:
                    value = float('nan')
                if np.isfinite(value):
                    out.append(repr(value))
                    i = j
                    continue
            out.append(prefix[i])
            i += 1

        target = form or ('infix' if in_form == 'infix' else
                          'tagged' if in_form == 'tagged' else 'explicit')
        if target == 'explicit':
            return out
        if target == 'tagged':
            return cast(list, self.to_tagged(out))
        if target == 'infix':
            return cast(str, self.to_infix(out))
        raise ValueError(f"unknown form {form!r}: expected 'tagged', 'explicit' or 'infix'")

    def to_infix(self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> str:
        """Convert an expression to the INFIX form -- NOTATION ONLY.

        One of the three PURE conversions (design
        the research harness): a syntactic re-notation, never a
        simplification. No canonical state is built, no rule fires, nothing is
        collected, folded, reordered or re-spelled, and the answer does not depend on
        the engine ARTIFACT. Use :meth:`simplify` for content; compose them as
        ``simplify(to_infix(x))``.

        A ``str`` input is already in the target notation and comes back VERBATIM
        (validated). Infix erases one distinction the token dialects carry: it has no
        ``inv`` glyph, so ``inv X`` renders ``1/X`` and reads back as ``/ 1 X``.

        Raises ``ValueError`` on a malformed expression or undeclared vocabulary
        (conversions are strict; :meth:`read_infix` stays the tolerant reader),
        ``TypeError`` on a non-expression type.
        """
        form, tokens = self._conversion_input(expression)
        if form == 'infix':
            if tokens:
                self._core.check_form(tokens)
            return cast(str, expression)
        if not tokens:
            return ''
        if form == 'tagged':
            tokens = self._core.to_prefix_syntactic(tokens)
        else:
            self._core.check_form(tokens)
        return self._core.prefix_to_infix(tokens, 'func', False)

    def to_prefix(self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
        """Convert an expression to the explicit binary PREFIX form -- NOTATION ONLY.

        See :meth:`to_infix` for the shared contract. A prefix token sequence is
        already in the target form and comes back BYTE-IDENTICAL (validated only):
        this conversion expands bags and touches nothing else.

        Expanding a bag has to pick an association, and the choice is stated: the
        positive part is a RIGHT-nested chain (``<add> a b c </add>`` becomes
        ``+ a + b c``, the association the canonical explicit emitter already
        produces), while an inverse SECTION becomes a LEFT-nested chain
        (``<mul> a <div> c d </mul>`` becomes ``/ / a c d``) because a section
        inverts each member individually -- only a left chain spells that without
        inventing the product ``c*d``.
        """
        form, tokens = self._conversion_input(expression)
        if not tokens:
            return []
        if form == 'tagged':
            return cast(list, self._core.to_prefix_syntactic(tokens))
        self._core.check_form(tokens)
        return tokens

    def to_tagged(self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
        """Convert an expression to the TAGGED form -- NOTATION ONLY.

        See :meth:`to_infix` for the shared contract. Bag members keep their
        source ORDER (sorting is canonicalisation, which is :meth:`simplify`'s job)
        and literals keep their spelling, so a tagged sequence already in normal form
        comes back byte-identical.

        The regrouping is CONSERVATIVE: a chain flattens only across nodes of the same
        kind at the same POLARITY. The right operand of ``-``/``/`` and the operand of
        ``neg``/``inv`` become ONE section member, never a flattened run, because
        ``1/(a*b)`` and ``(1/a)*(1/b)`` are different expressions -- they part company
        at ``0`` and ``inf``, which is why the AC core gates that distribution behind a
        nonzero certificate. A lone group inverse keeps its unary spelling (``neg x0``,
        ``inv x0``): the grammar has no one-member bag.
        """
        form, tokens = self._conversion_input(expression)
        if not tokens:
            return []
        return cast(list, self._core.to_tagged_syntactic(tokens))

    def mask(self, expression: str | list[str],
             policy: 'str | Callable[[str, Any], str | None]' = 'all',
             *, collect: bool = True) -> str | list[str]:
        """Mask literals: the front door over :mod:`simplipy.masking` (design D8).

        Pure delegation to the toolkit -- the mechanism, the role walk, and the
        one-constant-per-degree-of-freedom collect stage are all
        :func:`simplipy.masking.mask`'s; this method only reads the input form and
        names the shipped policies.

        Parameters
        ----------
        expression : str or list[str]
            An infix ``str`` (masked skeleton returned as an infix ``str``) or a
            token list in either dialect (masked skeleton returned as a token list
            in the SAME dialect, exactly as the toolkit emits it).
        policy : str or callable, optional
            ``'all'`` (:func:`~simplipy.masking.mask_all`, the default),
            ``'fittable'`` (:func:`~simplipy.masking.mask_fittable`), ``'values'``
            (an accepted spelling of ``'fittable'`` -- one policy, see the
            :mod:`simplipy.masking` docstring), or any
            ``(value, role) -> str | None`` callable.
        collect : bool, keyword-only, optional
            Forwarded to :func:`simplipy.masking.mask`. ``True`` (the default) runs
            the COLLECT stage: the engine is re-run over the substituted tokens,
            which is what enforces ONE ``<constant>`` per degree of freedom --
            ``2*x0/3`` is one free value, and a positional pass alone would abstract
            it into two. Because the collect stage IS a ``simplify`` call,
            simplification RULES fire and the canonical ORDER is re-imposed, so
            terms may be RE-ORDERED and shapes normalized (``x0 / <constant>``
            becomes ``<constant> * x0``) relative to the input spelling. Pass
            ``collect=False`` for the raw positional substitution, which keeps a
            strict 1:1 token correspondence with the input at the cost of the
            degree-of-freedom guarantee.

            NOTE for ``str`` input: the masked tokens are rendered back to infix
            through :meth:`to_infix`, which parses into the canonical state, so an
            infix result is canonically ordered either way. ``collect=False`` still
            differs there -- it suppresses the rule-firing re-simplification, not
            the rendering's canonical construction.

        Raises
        ------
        ValueError
            On an unknown policy name or a malformed expression.
        TypeError
            On a non-str/list expression or a non-str/callable policy.
        """
        from . import masking as _masking
        if callable(policy):
            policy_fn = policy
        elif isinstance(policy, str):
            # 'values' and 'fittable' name the SAME policy (the toolkit's historic
            # `mask_values_keep_structure` was renamed to `mask_fittable`); both
            # resolve to the surviving function, so the front door never touches
            # the deprecated spelling.
            named = {'all': _masking.mask_all,
                     'fittable': _masking.mask_fittable,
                     'values': _masking.mask_fittable}
            if policy not in named:
                raise ValueError(
                    f"unknown masking policy {policy!r}: expected 'all', 'fittable', "
                    f"'values' or a (value, role) -> str | None callable")
            policy_fn = named[policy]
        else:
            raise TypeError(
                f'policy must be a policy name or a (value, role) -> str | None '
                f'callable, not {type(policy).__name__}')
        if isinstance(expression, str):
            return self.to_infix(
                _masking.mask(self._core.parse(expression, True, False), self, policy_fn,
                              collect=collect))
        if isinstance(expression, list):
            return _masking.mask(expression, self, policy_fn, collect=collect)
        raise TypeError(
            f'mask expects an infix str or a token list, '
            f'not {type(expression).__name__}')

    def simplify(
            self,
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
            *,
            max_passes: int | None = None,
            mode: Mode | str = Mode.f64) -> str | list[str] | tuple[str, ...] | np.ndarray:
        """Simplify through the AC CORE: the n-ary associative-commutative engine.

        The AC core represents ``+`` and ``*`` as flat, sorted n-ary bags with EXACT rational
        coefficients composed explicitly (``* 7 x``, ``pow x 3``) instead of the hyper-operator
        vocabulary (``mult2..5``, ``div2..5``, ``pow2..5``, ``pow1_2/1_4``; the real odd roots
        ``pow1_3``/``pow1_5`` remain genuine operators). Rules are widened to SUB-MULTISET
        matching with the unmatched remainder preserved, so a rule fires wherever the algebra
        permits, regardless of operand order or bracketing -- both axes of the binary engine's
        commutative-order invariance defect are removed at the representation level, and the
        result is invariant under permutation of commutative operands by construction.

        Like-term/like-factor collection (the AC form of term cancellation) runs inside the
        canonical constructors under the same soundness certificates as the rule matcher
        (finite-a.e. for sign-cancelling addition, finite-and-nonzero-a.e. for
        exponent-cancelling multiplication), and coefficient arithmetic is exact rational
        computation rather than mined rules. The output is idempotent
        (``simplify(simplify(x)) == simplify(x)``).

        .. note::
            Pair this engine with SORT-PROMOTED rulesets (the ``2-1``/``3-2``/``4-3``
            artifact family and later). The legacy ``dev_7-3`` asset predates the certificate
            sorts (its pattern rules are uniformly ``_``-sorted, gated only by mining-time
            sampling); it is supported only by simplipy <= 0.11 as a separate pinned install,
            and its certificate-free cancellation rules are unsound on pole spellings that the
            AC core's unified representation makes reachable.

        Parameters
        ----------
        expression : str | list[str] | tuple[str, ...] | np.ndarray
            The expression to simplify: a ``str`` is parsed as infix; a list, tuple or
            1-D ``ndarray`` is a prefix token sequence (old grammar and tagged form are
            both accepted -- one shared parser).
        max_passes : int, optional
            Bound on the number of outer REWRITE PASSES -- not on nodes, and not on rule
            firings. One pass is one full rewrite sweep over the whole expression; the
            chain stops as soon as a pass changes nothing (the fixpoint). Defaults to 48,
            which is far above the observed convergence of 2-4 passes, so the bound is
            defense-in-depth against an ordering bug (T6 proves the fixpoint is reached in
            finitely many passes) rather than a tuning knob. ``max_passes=0`` is treated as
            1: at least one pass always runs.

            * ``'tagged'`` -- the STRICT prefix form, the AC engine's native serialization
              (default for token inputs): n-ary bags are delimited (``<add> ... </add>``,
              ``<mul> ... </mul>``) and carry their group inverse as a SECTION -- terms
              after ``<sub>`` subtract, factors after ``<div>`` divide, so
              ``(2*x1)/(x2*x3)`` is ``<mul> 2 x1 <div> x2 x3 </mul>``. ``pow`` and the
              unary functions stay plain prefix; ``neg``/``inv`` exist only as the
              standalone unary spellings (``tan neg x0``, ``inv x0``) -- inside bags the
              sections own all inverses. Exact literals are one token each: ``7``, ``0.2``,
              ``1/3``. Tagged output is accepted back as input (one shared, liberal parser).
            * ``'infix'`` -- the PRETTY human-readable rendering (default for ``str``
              inputs; always returns ``str``): ``x8 + 1.2*x3``, ``-x0/3``, ``(x0 + 1)^2``.
              Round-trips: feeding the rendering back as a ``str`` input reaches the same
              canonical state. The bare names ``pi``, ``e``, ``inf`` and ``nan`` are
              RESERVED constant spellings of the infix language (the renderer emits them
              for the canonical constants, so the parser reads them back as constants,
              never as variables).
            * ``'explicit'`` -- the binary-chain form (``* 6 x``, ``- a b``, literal
              coefficients only -- the hyper-operator vocabulary is deleted). This is
              the project's INTERNAL dialect, not a recommended consumer format:
              rule artifacts are stored in it, the miner's enumeration universe and
              the certificate tapes are defined on it, and it stays available here
              for debugging and compatibility. ``'tagged'`` and ``'infix'`` are the
              public output forms. Sign placement follows the same doctrine as the
              tagged form: a sign lives in the numeric coefficient literal whenever
              one is emitted (``* -2 x``), additive term signs are structural
              (binary ``-``), and ``neg`` spells only the pure sign (``neg x0``,
              ``neg * np.pi x0``).

        Returns
        -------
        str | list[str] | tuple[str, ...] | np.ndarray
            The simplified expression, in the same format as the input.
        """
        if max_passes is None:
            max_passes = 48
        if max_passes < 0:
            # would otherwise surface as a raw pyo3 OverflowError at the usize
            # conversion (hardening H-006, 2026-08-03)
            raise ValueError(f"max_passes must be non-negative, got {max_passes}")
        # A STRING mode must coerce, never silently compare unequal to the enum:
        # `mode='lossy'` used to run the default because `'lossy' == Mode.LOSSY` was False
        # (audit Tier-2, 2026-08-03). Accept the enum, its names (any case), and its
        # integer values; everything else raises.
        if isinstance(mode, str):
            key = mode.strip().lower()
            # Match on the member NAME case-insensitively: the documented spellings are
            # lower-case (`'f64'`), and the old code upper-cased before lookup, so a
            # bare `.upper()` would now miss every one of them.
            match = {m.name.lower(): m for m in Mode}.get(key)
            if match is None:
                raise ValueError(
                    f"unknown mode {mode!r}: expected one of "
                    f"{[m.name for m in Mode]} (or a simplipy.Mode)") from None
            mode = match
        elif not isinstance(mode, Mode):
            # api-1/fmux-mode-1: a bare int (or numpy float) used to COERCE -- mode=3
            # from a JSON config silently selected LOSSY, the rung that trades
            # soundness for recall. Only Mode members and the documented names bind.
            raise TypeError(
                f"mode must be a simplipy.Mode or one of "
                f"{[m.name for m in Mode]}, not {type(mode).__name__} ({mode!r})")

        if isinstance(expression, str):
            tokens = self._core.parse(expression, True, False)
        elif isinstance(expression, np.ndarray):
            _validate_ndarray_input(expression)
            tokens = expression.tolist()
        else:
            tokens = list(expression)

        # DIALECT-PRESERVING (owner ruling 2026-08-18): `simplify` only simplifies and
        # answers in the form it was handed -- a str is infix, a token sequence carrying
        # a bag delimiter is tagged, anything else is the explicit binary prefix. This is
        # what closed the old "tagged leak" (a prefix-in token list used to come back
        # tagged); to change the NOTATION, convert: `simplify(to_tagged(x))`.
        if isinstance(expression, str):
            form = 'infix'
        elif _TAGGED_DIALECT_TOKENS.intersection(tokens):
            form = 'tagged'
        else:
            form = 'explicit'
        if mode is Mode.real and self._core.mode_rules_len('real') is None:
            # FAIL CLOSED. A mode naming no set of its own falls back to the default
            # set, which is right for `corpus` (its divergence is search semantics, so
            # the fallback reproduces today's LOSSY exactly) and WRONG for `real`, whose
            # only divergence IS which rules are certified. Silently serving the f64 set
            # here would answer a request for mathematical soundness with rules known to
            # be mathematically false -- the 18-rule `f(+-10^-k) -> +-10^-k` family is
            # f64-exact and false by a cubic term -- which is the exact over-claim the
            # mode split exists to prevent. An artifact predating the triple cannot
            # answer this question, so it says so.
            raise ValueError(
                "mode='real' needs a ruleset mined for it, and this artifact has none: "
                "it predates the rules_f64/rules_real/rules_corpus triple. Falling back "
                "to the default set would serve f64-certified rules under a claim of "
                "MATHEMATICAL soundness, and the two are incomparable -- some f64 rules "
                "are mathematically false. Use mode='f64' for the deployed semantics, or "
                "load an artifact whose config names a real ruleset.")

        # ONE lookup, both branches: the rule mode is decided here and the core is asked
        # for it by name, so the two output paths cannot end up serving different sets.
        rule_mode = _RULE_MODE[mode]
        if form == 'infix':
            return self._core.ac_simplify_infix_in_mode(tokens, max_passes, rule_mode)

        out = self._core.ac_simplify_in_mode(tokens, max_passes, rule_mode, form)

        if isinstance(expression, str):
            # The old infix converter cannot render the tagged form; a str input asking for
            # a token form gets the token list.
            return out
        return self._denormalize(out, expression)

    def complexity(
            self,
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
            certified: bool = True) -> int:
        """The SEMANTIC COMPLEXITY of an expression, measured on its canonical form.

        This is the functional :meth:`simplify` minimizes (the unified measure mu),
        measured by default on the CERTIFIED canonical state -- the same
        certificate-carrying canonicalization the simplify chain runs on, so
        ``complexity(simplify(e)) <= complexity(e)`` is a THEOREM (chain descent,
        docs/formal.md L3). With ``certified=False`` the expression is priced on the
        bare (certificate-less, fail-closed) canonicalization instead: still invariant
        to operand order, bracketing and serialization sugar, but a certificate-licensed
        respelling the bare context cannot re-derive keeps its own measure, so a
        simplify output can price ABOVE its input (measured live: 0.48% of 64k corpus
        rows, in quanta of one symbol unit; the certified default closes exactly this).

        Invariant either way: mu prices structure and information, not spelling --
        signs and magnitude-1 coefficient/exponent slots are free, literals pay their
        description length on the exact value, symbols pay one symbol unit.
        """
        if isinstance(expression, str):
            tokens = self._core.parse(expression, True, False)
        elif isinstance(expression, np.ndarray):
            tokens = expression.tolist()
        else:
            tokens = list(expression)
        if certified:
            return self._core.ac_complexity_certified(tokens)
        return self._core.ac_complexity(tokens)

    def _denormalize(
            self,
            out: list[str],
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
    ) -> str | list[str] | tuple[str, ...] | np.ndarray:
        """Map a core prefix-token result back to the input's type (``simplify``'s tail)."""
        if isinstance(expression, str):
            return self._core.prefix_to_infix(out, '**', False)
        if isinstance(expression, np.ndarray):
            # Re-infer the string WIDTH from the result, keeping only the input dtype KIND: a fold
            # can emit a token wider than any input token (e.g. `1/0 -> float("inf")`), and a fixed
            # `dtype=expression.dtype` (whose width numpy sized to the inputs) would silently truncate.
            return np.array(out).astype(expression.dtype.kind)
        if isinstance(expression, tuple):
            return tuple(out)
        return out

    def find_rules(self, *args: Any, **kwargs: Any) -> Any:
        """Mine rewrite rules into this engine's rule set.

        Thin delegator to :meth:`simplipy.mining.RuleMiner.find_rules` (C23a,
        D27) — the signature, semantics, and single-flight lock are the
        miner's; the public API is unchanged."""
        return RuleMiner(self).find_rules(*args, **kwargs)

    # the delegators carry the miner's REAL signature for inspect/docs/tests
    find_rules.__wrapped__ = RuleMiner.find_rules  # type: ignore[attr-defined]

    def certify_rules(self, *args: Any, **kwargs: Any) -> Any:
        """Certify proposed rules against this engine.

        Thin delegator to :meth:`simplipy.mining.RuleMiner.certify_rules`
        (C23a, D27); the public API is unchanged."""
        return RuleMiner(self).certify_rules(*args, **kwargs)

    certify_rules.__wrapped__ = RuleMiner.certify_rules  # type: ignore[attr-defined]

    _MEASURE_PROBES = (
        # the LITERAL codebook: an integer (the L-formula), a unit fraction and a
        # non-unit one (the fraction code and the inversion bit), a decimal whose
        # denominator carries a five (the print/argmin split), and `<constant>`.
        ('1000',), ('1/2',), ('355/113',), ('0.2',), ('<constant>',),
        # the SYMBOL TABLE, one probe per entry (2026-08-21). Before these, six of the
        # nine entries were invisible to the fingerprint: changing `Pow` from 4 bits to
        # 3 left the digest at `355f6ba90801f603`, so an artifact mined under one table
        # was indistinguishable from one mined under another -- which is the exact
        # failure this fingerprint exists to prevent. Keyed on the first token, so each
        # probe needs a distinct head.
        ('x0',),                    # leaf
        ('+', 'x0', 'x1'),          # Add
        ('*', 'x0', 'x1'),          # Mul
        ('pow', 'x0', '2'),         # Pow
        ('np.pi',), ('np.e',),      # the named constants
        ('sin', 'x0'),              # an elementary head
        ('asin', 'x0'),             # a transcendental head
        ('float("inf")',),          # the infinities
    )

    def _measure_fingerprint(self) -> dict:
        """A fingerprint of the REDUCTION MEASURE, for the provenance sidecar.

        Without it an artifact mined under one measure is indistinguishable from one mined
        under another: the sidecar records the version, the build and the parameters, and
        every one of those can be identical across a measure change (worse, uncommitted
        work stamps the same `-dirty` build string). Audit C1.4 flagged this as a BLOCKER to
        fix BEFORE a re-mine; three artifacts were re-mined without it, so the values here
        are also what identifies those retroactively.

        The fingerprint is BEHAVIOURAL, not a version string: it records what the measure
        actually charges on probes chosen to separate the changes it has undergone. Any
        change to `L`, to the fraction/decimal codes, to `mu_free`, or to ANY ENTRY OF THE
        SYMBOL TABLE moves at least one entry -- the last clause is why there is one probe
        per table entry rather than the single bare symbol this used to carry.
        """
        probes: dict[str, int | None] = {}
        for tokens in self._MEASURE_PROBES:
            try:
                probes[tokens[0]] = int(self.complexity(list(tokens)))
            except Exception:  # a config that cannot express a probe records it as absent
                probes[tokens[0]] = None
        return {
            'unit': 'milli-bits',
            # `mu_sym` is the historical key and stays, for sidecars already published;
            # what it holds is the LEAF entry, which is what a bare symbol costs.
            'mu_sym': probes.get('x0'),
            'mu_leaf': probes.get('x0'),
            'mu_free': probes.get('<constant>'),
            'probes': probes,
            'digest': hashlib.sha256(
                repr(sorted(probes.items())).encode()).hexdigest()[:16],
        }

    # The libm probe battery (R5, audit L9): expressions whose folded values are
    # decided by the SYSTEM libm the deployed folder calls -- the C symbols bound in
    # rust/numeric.rs resolve on THIS machine at load, so the exposure is runtime, not
    # build-time, and no wheel qualification can cover it. Chosen to discriminate real
    # builds: the cosh(acosh(k)) family (>= 5 shipped acj-4-3 rules exist only because
    # glibc 2.43 is 1 ulp wrong exactly there), argument-reduction-sensitive trig at a
    # large argument, atanh near its edge (libm-routed since B6), exp/log/pow at
    # spread magnitudes, and ulp-amplifying compositions. Fixed and ordered: the
    # fingerprint must be comparable across sidecars.
    _LIBM_PROBES: tuple[tuple[str, ...], ...] = (
        ('cosh', 'acosh', '2'), ('cosh', 'acosh', '5'), ('cosh', 'acosh', '8'),
        ('cosh', 'acosh', 'np.e'), ('cosh', 'acosh', 'np.pi'),
        ('sin', '1'), ('sin', '2'), ('sin', '1000000000'),
        ('cos', '1'), ('cos', '1000000000'), ('tan', '1'), ('tan', '100'),
        ('asin', '0.5'), ('asin', '0.9999'), ('acos', '0.5'), ('atan', '2'),
        ('sinh', '1'), ('sinh', '20'), ('cosh', '1'), ('tanh', '1'),
        ('asinh', '2'), ('acosh', '1.5'), ('atanh', '0.5'), ('atanh', '0.999'),
        ('exp', '1'), ('exp', '40'), ('exp', '-40'), ('log', '2'), ('log', 'np.pi'),
        ('pow', 'np.pi', 'np.e'), ('pow', '2', '0.5'), ('pow', '10', '-7'),
        ('rootn', '2', '3'), ('sin', 'exp', '3'), ('log', 'cosh', '10'),
        ('tan', 'exp', '2'),
    )

    def libm_fingerprint(self) -> str:
        """A sha256 fingerprint of the host's libm, computed through the deployed
        folding path (R5, audit L9).

        The transcendental folds go through the SYSTEM libm (the C symbols
        `rust/numeric.rs` binds), which resolves on the running machine -- so two
        hosts with identical wheels and identical `pip freeze` can mine different
        artifacts (measured: the `cosh(acosh(2)) -> 2` rule exists on glibc 2.43 and
        not on the publish host). The fingerprint makes that difference a recorded,
        comparable fact: it is stored in every mine's provenance sidecar
        (`environment.libm_fingerprint`), and two sidecars with different values are
        mines over different arithmetic.
        """
        assert self._core is not None
        lines = []
        for probe in self._LIBM_PROBES:
            try:
                out = self._core.evaluate_constant_subtree(list(probe))
            except Exception:
                out = None
            lines.append(f"{' '.join(probe)}={out}")
        return hashlib.sha256('\n'.join(lines).encode()).hexdigest()

    def _realization_only_tokens(self) -> set[str]:
        """Realization spellings that are NOT also operator tokens -- the detectable half.

        Cached on the instance: the operator table is fixed at construction.
        """
        cached = getattr(self, '_realization_only_cache', None)
        if cached is None:
            ops = set(self._operators_config)
            cached = {spec['realization'] for spec in self._operators_config.values()
                      if spec['realization'] not in ops}
            self._realization_only_cache = cached
        return cached

    def _assert_realizations_are_invertible(self) -> None:
        """Reading realization notation back REQUIRES an injective realization map.

        Two operators may legally share a realization -- nothing in the config format
        forbids `abs` and `absolute` both running `np.abs` -- and then the spelling
        `np.abs` names both, so no reader can recover which was meant. Injectivity is a
        property of a CONFIG, not a guarantee of the format, so it is checked when it
        matters and refused loudly rather than resolved by picking one.
        """
        seen: dict[str, str] = {}
        for op, spec in self._operators_config.items():
            r = spec['realization']
            if r in seen:
                raise ValueError(
                    f"this engine's realization map is not invertible: operators "
                    f"{seen[r]!r} and {op!r} both realize as {r!r}, so realization "
                    f"notation cannot be read back. Convert from operator notation "
                    f"instead, or give the two operators distinct realizations.")
            seen[r] = op

    def expression_variables(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray') -> list[str]:
        """The free variables of an expression, IN ORDER OF FIRST APPEARANCE.

        Order is the contract, not an accident: it is the signature
        :meth:`as_callable` binds, so `f(*values)` has to mean what the expression
        reads. A token is a variable when it is none of the things that are not one --
        an operator, a numeric literal, a special constant, or a slot. Slots are
        excluded because a `<constant>` is a fitted degree of freedom, not an argument.
        """
        from .utils import is_numeric_string
        tokens = self.to_prefix(expression)
        out: list[str] = []
        for t in tokens:
            if t in self._operators_config or t in out:
                continue
            if is_numeric_string(t) or t in _SPECIAL_LEAVES:
                continue
            # The PARENTHESIZED negative literal (`(-3)`) is the engine's own spelling and
            # is not caught by `is_numeric_string`; `!` is a slot sigil like the rest.
            # Missing both made 3,389 of the shipped artifact's 10,638 rule sides report a
            # literal as a free variable, and `as_callable` then compiled a bad signature.
            if t.startswith('(') and t.endswith(')') and is_numeric_string(t[1:-1]):
                continue
            if t.startswith(('<', '_', '$', '?', '!')):
                continue
            out.append(t)
        return out

    def to_realization(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray') -> list[str]:
        """The expression in REALIZATION notation: prefix tokens, operators spelled as
        the callables they run (``sin`` -> ``np.sin``).

        The fourth notation of the conversion family, and reversible like the other
        three: :meth:`to_prefix`, :meth:`to_infix` and :meth:`to_tagged` all accept
        realization-spelled input and read it back. It is what the compile pipeline
        consumes, and it is occasionally what you want directly -- to see which callable
        an operator actually resolves to, without running anything.

        Reading realization notation back needs an injective realization map; writing it
        does not, so this direction never refuses.
        """
        return self.operators_to_realizations(self.to_prefix(expression))

    def as_code(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray',
            variables: list[str] | None = None) -> CodeType:
        """Compile the expression to a Python code object.

        ``as_``, not ``to_``: this is TERMINAL. Every ``to_*`` in this API is a notation
        that converts back, and a code object has no syntax to recover -- naming it
        ``to_compiled`` would advertise a round-trip that cannot exist.

        .. warning::
           Compiling runs the expression's realizations through :func:`compile`, so the
           usual trust rules apply -- the evaluation namespace is scoping, not a sandbox
           (:mod:`simplipy.trust`).
        """
        from .utils import codify
        prefix = self.to_prefix(expression)
        if variables is None:
            variables = self.expression_variables(prefix)
        return codify(self.prefix_to_infix(prefix, realization=True), variables)

    def as_callable(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray',
            variables: list[str] | None = None) -> Callable[..., float]:
        """Compile the expression to a callable bound to THIS engine's namespace.

        The one-step form of :meth:`as_code` followed by :meth:`code_to_lambda`, which
        is the pipeline nearly every caller wanted. Terminal, for the reason given on
        :meth:`as_code`.
        """
        return self.code_to_lambda(self.as_code(expression, variables))

    def operators_to_realizations(self, prefix_expression: list[str] | tuple[str, ...]) -> list[str]:
        """Convert canonical operator names to their runtime realizations (e.g. ``'sin'`` -> ``'np.sin'``)."""
        return self._core.operators_to_realizations(list(prefix_expression))

    def realizations_to_operators(self, prefix_expression: list[str]) -> list[str]:
        """Convert realization tokens (e.g. ``'np.sin'``) back to canonical operator names."""
        return self._core.realizations_to_operators(list(prefix_expression))

    def code_to_lambda(self, code: CodeType) -> Callable[..., float]:
        """Converts a Python code object into an executable lambda function.

        The compiled expression is bound to THIS engine's evaluation namespace: the
        modules its own realizations needed, plus ``np`` for the ``np.pi``/``np.e``
        token grammar. Until 0.13 every engine shared ``simplipy.engine``'s module
        globals, so one engine's imports were reachable from another's expressions,
        along with simplipy's own imports (``os``, ``importlib``, ...); register
        C1.12 closed that. Python still injects builtins, and a hostile EXPRESSION is
        still unsafe to compile -- the namespace is scoping, not a sandbox
        (:mod:`simplipy.trust`).

        Compatibility: this was a ``staticmethod`` before 0.13. Instance calls
        (``engine.code_to_lambda(code)``, the documented and only in-tree usage) are
        unaffected; a call on the CLASS now needs an instance.

        Parameters
        ----------
        code : CodeType
            The compiled code object to convert.

        Returns
        -------
        Callable[..., float]
            An executable lambda function.
        """
        return FunctionType(code, self._eval_namespace)()
