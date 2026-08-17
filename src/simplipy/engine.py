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
import signal
import threading
from itertools import product
from types import CodeType, FunctionType
from typing import Callable
from pathlib import Path
from typing import Any, Literal
from copy import deepcopy
from enum import IntEnum

import numpy as np
import json
import yaml

from simplipy.utils import (
    is_numeric_string,
    deduplicate_rules,
    enumerate_expressions, count_expressions, sample_expression,
    remap_expression,
    violates_wildcard_multiplicity)
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

class Mode(IntEnum):
    """The simplification soundness mode: an ORDINAL axis where a higher rung permits
    strictly more aggressive (less sound) rewrites.

    The decided full ordering is ``EXACT <= SOUND <= AE <= LOSSY``; only the two implemented
    rungs are exposed. ``EXACT`` (0) and ``AE`` (2) are reserved positions in that ordering,
    not yet implemented -- the gaps in the integer values keep the ordinal stable when they
    are added.

    - ``SOUND`` (the default): equivalence-preserving and idempotent. The deployed
      inference/scoring mode. Byte-identical to the historical default.
    - ``LOSSY``: trades soundness for recall -- every rule placeholder binds any subtree (the
      ``!``-sort finite-a.e. certificate is skipped) AND the constant-fold's finiteness gate is
      relaxed (so e.g. ``<constant>/0`` collapses to ``<constant>``). For training-corpus
      canonicalisation ONLY: the training data is generated FROM the simplified form, so the
      target equals the data and there is no external function to violate. Do NOT use on an
      inference or scoring path.
    """
    SOUND = 1
    LOSSY = 3


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
        The list of simplification rules loaded into the engine (mirrored into the
        compiled core by :meth:`compile_rules`).
    modules : list[str]
        The importable package names this engine's realizations need (``numpy`` is
        always present: ``np.pi``/``np.e`` are token grammar). These are PACKAGE
        names, not the canonical config spellings -- ``np.sin`` appears here as
        ``numpy``.
    """
    def __init__(self, operators: dict[str, dict[str, Any]], rules: list[tuple] | None = None, *,
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

        # Normalize the incoming rule list and eliminate duplicate patterns.
        dummy_variables = [f'x{i}' for i in range(100)]
        if rules is None:
            self.simplification_rules = []
        else:
            self.simplification_rules = deduplicate_rules(rules, dummy_variables=dummy_variables)

        # The raw operator config, kept for core (re)construction.
        self._operators_config = deepcopy(operators)
        # Build the compiled core (REQUIRED; see the module docstring): every
        # construction path (from_config/load AND direct in-memory construction) attaches it
        # here, from the SAME in-memory state, so no path can exist without a core.
        self._core = self._build_core(self._operators_config, self.simplification_rules)

    @staticmethod
    def _build_core(operators: dict[str, dict[str, Any]], rules: list[tuple]) -> Any:
        """Build the compiled core (``simplipy._core``) from in-memory config + rules.

        The core is REQUIRED: a missing extension or a load failure is a hard error --
        the pure-Python engine was removed.
        """
        if _RustEngine is None:
            raise ImportError(
                'simplipy._core is required (the pure-Python engine was removed): the compiled '
                f'extension failed to import ({_CORE_IMPORT_ERROR!r})')
        config_text = yaml.safe_dump({'operators': operators}, sort_keys=False)
        rules_text = json.dumps([[list(lhs), list(rhs)] for lhs, rhs in rules])
        return _RustEngine.from_strs(config_text, rules_text)

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
        self._core = self._build_core(self._operators_config, self.simplification_rules)

    def compile_rules(self) -> None:
        """Sync the compiled core's rule set from ``self.simplification_rules``.

        Public contract (unchanged since the pure-Python engine): after mutating
        ``self.simplification_rules``, call this so ``simplify`` sees the new rules.

        COPY-ON-WRITE (conc-1): a fresh core is built from the in-memory state and
        the reference is swapped atomically -- an in-place ``set_rules`` push
        mutated the core under a concurrent reader's feet mid-``simplify``. A
        reader holding the old core finishes on a consistent (old) ruleset; the
        ~20 ms rebuild is mutation-API cost only, never on the hot path.
        """
        self._core = self._build_core(self._operators_config, self.simplification_rules)

    def _replace_rules(self, rules: list) -> None:
        """Build-first-or-unchanged (conc-2): compile a fresh core from the CANDIDATE
        ruleset, and only on success assign both the list and the core. The shim
        mutators used to rewrite ``simplification_rules`` before pushing, so a failed
        push left the wrapper and the core silently diverged; this turns that into
        loud-and-unchanged."""
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
        new_core = self._build_core(self._operators_config, rules)
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
        rules = []
        rules_path = None
        rules_file = config.get('rules')
        if rules_file:
            if not os.path.isabs(rules_file):
                config_dir = os.path.dirname(config_path)
                rules_path = os.path.join(config_dir, rules_file)
            else:
                rules_path = rules_file
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules = json.load(f)
            else:
                warnings.warn(f"Rules file '{rules_path}' specified in config not found.", UserWarning)
                rules_path = None
        engine = cls(operators=config['operators'], rules=rules, trusted_modules=trusted_modules)
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
    def load(cls, path: str, install: bool = False, local_dir: Path | str | None = None, repo_id: str | None = None,
             manifest_filename: str | None = None, *, trusted_modules: list[str] | None = None) -> "SimpliPyEngine":
        """Loads a pre-defined engine configuration from the asset manager.

        This provides a convenient way to load standard engine configurations
        distributed with the `simplipy` package.

        Parameters
        ----------
        path : str
            The name of the configuration to load (e.g., 'default').
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
        # Known generation-1 names refuse BEFORE any download (the config-level gate
        # in from_config covers everything else, including local paths).
        if not os.path.exists(path):
            from .compat import check_asset_name
            check_asset_name(path)
        return cls.from_config(
            get_path(path, install=install, local_dir=local_dir, repo_id=repo_id, manifest_filename=manifest_filename),
            trusted_modules=trusted_modules)

    def is_valid(self, prefix_expression: list[str], verbose: bool = False) -> bool:
        """Checks if a prefix expression is syntactically valid.

        An expression is valid if every operator has the correct number of
        operands according to its defined arity.

        Parameters
        ----------
        prefix_expression : list[str]
            The expression in prefix notation.
        verbose : bool, optional
            If True, prints the reason for invalidity. Defaults to False.

        Returns
        -------
        bool
            True if the expression is valid, False otherwise.
        """
        # The verdict ALWAYS comes from the compiled core (single implementation); the pure-Python
        # loop below only survives as a DIAGNOSTIC printer for the (rare) verbose call, and only
        # runs when the core has already rejected the expression.
        valid = self._core.is_valid(list(prefix_expression))
        if verbose and not valid:
            self._explain_invalid(prefix_expression)
        return valid

    def _explain_invalid(self, prefix_expression: list[str]) -> None:
        """Print WHY an expression failed :meth:`is_valid` (diagnostics only, no verdict)."""
        stack: list[str] = []
        tagged = _TAGGED_DIALECT_TOKENS.intersection(prefix_expression)
        if tagged:
            # Without this check the walk below misreads the bag delimiters as variables
            # and prints 'Variable must be leaf node' -- actively misleading (B1).
            print(f'Invalid expression {prefix_expression}: carries tagged-serialization '
                  f'tokens {sorted(tagged)} (simplify\'s default token output). is_valid '
                  f'reads the explicit binary-prefix dialect; tagged output is re-accepted '
                  f'by simplify()/complexity()/masking, or re-projected via '
                  f"simplify(expr, form='explicit').")
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

    def parse(
            self,
            infix_expression: str,
            convert_expression: bool = True,
            mask_numbers: bool = False) -> list[str]:
        """Parses an infix string into a standardized prefix expression.

        This is a high-level parsing utility that combines `infix_to_prefix`
        with optional canonicalization and number masking. The resulting token
        list is additionally cleaned up via `remove_pow1` to drop redundant
        ``pow1_1`` occurrences.

        Parameters
        ----------
        infix_expression : str
            The mathematical expression in infix notation.
        convert_expression : bool, optional
            If True, the expression is normalized using `convert_expression`.
            Defaults to True.
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

    def simplify(
            self,
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
            *,
            node_budget: int = 48,
            mode: Mode | str = Mode.SOUND,
            form: str | None = None) -> str | list[str] | tuple[str, ...] | np.ndarray:
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
        node_budget : int, optional
            Bound on the outer rewrite iterations. The default is far above the typical
            fixpoint depth (2-4 iterations).
        mode : Mode or str, optional
            The soundness mode (see :class:`Mode`). ``Mode.LOSSY`` relaxes every
            certificate -- training-corpus canonicalisation only. The enum's names are
            accepted as strings, any case (``mode='lossy'``); an unknown string raises
            rather than silently running SOUND.
        form : str, optional
            Output projection of the canonical answer; the simplification itself is
            identical in every case.

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
        if node_budget < 0:
            # would otherwise surface as a raw pyo3 OverflowError at the usize
            # conversion (hardening H-006, 2026-08-03)
            raise ValueError(f"node_budget must be non-negative, got {node_budget}")
        # A STRING mode must coerce, never silently compare unequal to the enum:
        # `mode='lossy'` used to run SOUND because `'lossy' == Mode.LOSSY` is False
        # (audit Tier-2, 2026-08-03). Accept the enum, its names (any case), and its
        # integer values; everything else raises.
        if isinstance(mode, str):
            try:
                mode = Mode[mode.upper()]
            except KeyError:
                raise ValueError(
                    f"unknown mode {mode!r}: expected one of "
                    f"{[m.name for m in Mode]} (or a simplipy.Mode)") from None
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

        if form is None:
            form = 'infix' if isinstance(expression, str) else 'tagged'
        # Reject an unknown form HERE, where all three are visible. The core only ever sees
        # 'tagged'/'explicit' (infix is dispatched just below), so its own message named two
        # of the three and a caller who mistyped `form='INFIX'` was told 'infix' was not an
        # option (C1.13's sibling in the audit register, C1.16).
        if form not in ('tagged', 'explicit', 'infix'):
            raise ValueError(
                f"unknown form {form!r}: expected 'tagged', 'explicit' or 'infix'")
        if form == 'infix':
            return self._core.ac_simplify_infix(tokens, node_budget, mode == Mode.LOSSY)

        out = self._core.ac_simplify(tokens, node_budget, mode == Mode.LOSSY, form)

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

    def find_rules(self, *args: Any, **kwargs: Any):
        """Mine rewrite rules into this engine's rule set.

        Thin delegator to :meth:`simplipy.mining.RuleMiner.find_rules` (C23a,
        D27) — the signature, semantics, and single-flight lock are the
        miner's; the public API is unchanged."""
        return RuleMiner(self).find_rules(*args, **kwargs)

    # the delegators carry the miner's REAL signature for inspect/docs/tests
    find_rules.__wrapped__ = RuleMiner.find_rules  # type: ignore[attr-defined]

    def certify_rules(self, *args: Any, **kwargs: Any):
        """Certify proposed rules against this engine.

        Thin delegator to :meth:`simplipy.mining.RuleMiner.certify_rules`
        (C23a, D27); the public API is unchanged."""
        return RuleMiner(self).certify_rules(*args, **kwargs)

    certify_rules.__wrapped__ = RuleMiner.certify_rules  # type: ignore[attr-defined]

    _MEASURE_PROBES = (('1000',), ('1/2',), ('355/113',), ('0.2',), ('<constant>',), ('x0',))

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
        change to `L`, to the fraction/decimal codes, to `mu_free` or to `mu_sym` moves at
        least one entry.
        """
        probes: dict[str, int | None] = {}
        for tokens in self._MEASURE_PROBES:
            try:
                probes[tokens[0]] = int(self.complexity(list(tokens)))
            except Exception:  # a config that cannot express a probe records it as absent
                probes[tokens[0]] = None
        return {
            'unit': 'milli-bits',
            'mu_sym': probes.get('x0'),
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
