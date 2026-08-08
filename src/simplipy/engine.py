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
    get_used_modules,
    deduplicate_rules,
    enumerate_expressions, count_expressions, sample_expression,
    remap_expression,
    violates_wildcard_multiplicity)
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
_MINE_LOCK = threading.Lock()

# THE ARTIFACT-AFFECTING SWITCH REGISTRY (H-042, doctrine-propagation sweep D4).
# Every environment switch that can change a MINED ARTIFACT or a CERTIFICATION VERDICT,
# in ONE place: the mine's provenance writer records every SET entry verbatim
# (`soundness.env_overrides`), so a switch added here is recorded automatically.
# Adding an artifact-affecting switch ANYWHERE (a rust OnceLock or a python read)
# without listing it here violates the kill-switch recording doctrine: a mine run
# under a non-default instrument must say so in its own sidecar. Pure observability
# switches (the *_TRACE family, SIMPLIPY_MINE_PROGRESS_INTERVAL) do NOT belong here.
# Caveat: the rust side reads its switches ONCE per process (OnceLock at first use),
# so the record is faithful unless the caller mutates os.environ between first engine
# use and the mine -- do not do that.
ARTIFACT_ENV_SWITCHES = (
    'SIMPLIPY_IVL_GATE',          # interval domain gate layer (recorded as bool too)
    'SIMPLIPY_IVL_CLASS',         # interval value-class layer (bool too)
    'SIMPLIPY_IVL_REACH',         # interval reachability layer (bool too)
    'SIMPLIPY_SPECIAL_BATTERY',   # special-point battery layer (bool too)
    'SIMPLIPY_IVL_NODE_BUDGET',   # interval node-budget override (raw too)
    'SIMPLIPY_AC_ABSORB_FIRST',   # bag-match attempt order (serve outputs + mined rules)
    'SIMPLIPY_MU_SYM',            # mu symbol unit (the reduction ordering itself)
    'SIMPLIPY_MU_FREE',           # mu <constant> cost (the ordering)
    'SIMPLIPY_ZERO_SIGN',         # miner sign-combo grid (reference/repro)
    'SIMPLIPY_POLE_GRID',         # miner magnitude-grid ablation
    'SIMPLIPY_HIPREC_FRAC',       # hiprec near-miss escalation gate (calibration)
    'SIMPLIPY_TAGGED_FRACTION_MAX',  # tagged structural-fraction bound: changes MINED output
)

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
_CONSTLIKE_LEAVES = {'<constant>', 'np.e', 'np.pi',
                     'float("inf")', 'float("-inf")', 'float("nan")'}

# The bag delimiters and inverse-section markers of the TAGGED serialization --
# `simplify`'s default token output (`<add> ... <sub> ... </add>`). The explicit-dialect
# entry points (`is_valid`, `prefix_to_infix`) do not read this form; they use this set
# to NAME the dialect in their diagnostics instead of failing with a bare arity error (B1).
_TAGGED_DIALECT_TOKENS = frozenset({'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>'})


def _tokens_in_vocabulary(tokens: Any, vocabulary: set) -> bool:
    """Vocabulary test for proposal/hint token sequences: numeric and constant-like
    literals are valid leaves everywhere (the generation-2 spellings carry their
    numbers as literal TOKENS -- `pow x0 2` where the retired ring spelled
    `pow2 x0` -- so a bare vocabulary-set test silently skipped every such
    proposal; audit Tier-1 #3)."""
    return all(t in vocabulary or t in _CONSTLIKE_LEAVES or is_numeric_string(t)
               for t in tokens)


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


def _validate_ndarray_input(expression: 'np.ndarray', inplace: bool) -> None:
    """The ndarray input contract for ``simplify`` (engine.py): 1-D, string-like dtype, no ``inplace``.
    Validated BEFORE the core call so malformed inputs fail with a clean ValueError."""
    if expression.ndim != 1:
        raise ValueError('`simplify` expects a one-dimensional numpy array of tokens')
    if expression.dtype.kind not in {'U', 'S', 'O'}:
        raise ValueError('`simplify` expects a numpy array of string-like tokens')
    if inplace:
        raise ValueError('`inplace=True` is not supported when the expression is a numpy array')


def _load_proposals(
        proposals: 'str | list | dict') -> tuple[list[tuple[tuple[str, ...], tuple[str, ...] | None]], dict]:
    """Load and normalize a :meth:`SimpliPyEngine.find_rules` proposal batch.

    Accepts a path to a proposals JSON file, or the equivalent in-memory object.
    TWO schemas are accepted (both reduce to a list of ``{source, target?}`` objects):

    (a) the consolidated artifact format -- a dict with key ``"proposals"`` whose
        entries are objects with ``"source"`` (a prefix token list) and an optional
        ``"target"`` (a prefix token list, used as the certification HINT); every
        other key (``why``, ``family``, ``tier``, ...) is ignored;
    (b) a bare list of such ``{source, target?}`` objects.

    Returns ``(entries, record)``: entries as ``(source_tuple, hint_tuple_or_None)``
    in FILE ORDER (the batch's processing order is the file's order -- part of the
    determinism contract), and the provenance record pinning WHAT was proposed:
    ``file`` (absolute path, or None for in-memory input) and ``sha256`` (of the raw
    file bytes, or of the normalized entries when no file exists -- the sidecar must
    pin the batch either way) plus ``count``. Malformed batches raise ``ValueError``
    HERE, before any mining compute is spent.
    """
    if isinstance(proposals, str):
        with open(proposals, 'rb') as file:
            raw = file.read()
        data = json.loads(raw)
        record: dict = {'file': os.path.abspath(proposals),
                        'sha256': hashlib.sha256(raw).hexdigest()}
    else:
        data = proposals
        record = {'file': None, 'sha256': None}
    if isinstance(data, dict):
        if 'proposals' not in data:
            raise ValueError("consolidated proposals object must carry a 'proposals' key")
        items = data['proposals']
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError(f'proposals must be a list of {{source, target?}} objects, got {type(items).__name__}')
    entries: list[tuple[tuple[str, ...], tuple[str, ...] | None]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or 'source' not in item:
            raise ValueError(f'proposal {index}: expected an object with a "source" token list, got {item!r}')
        source = tuple(str(token) for token in item['source'])
        target = item.get('target')
        hint = tuple(str(token) for token in target) if target else None
        entries.append((source, hint))
    if record['sha256'] is None:
        normalized = json.dumps([[list(source), list(hint) if hint else None] for source, hint in entries])
        record['sha256'] = hashlib.sha256(normalized.encode()).hexdigest()
    record['count'] = len(entries)
    return entries, record


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

    Attributes
    ----------
    operator_tokens : list[str]
        A list of all defined operator names.
    operator_arity : dict[str, int]
        A mapping from operator names to their arity (number of arguments).
    simplification_rules : list[tuple]
        The list of simplification rules loaded into the engine (mirrored into the
        compiled core by :meth:`compile_rules`).
    """
    def __init__(self, operators: dict[str, dict[str, Any]], rules: list[tuple] | None = None) -> None:
        # Cache operator metadata for quick access during parsing and evaluation.
        self.operator_tokens = list(operators.keys())
        self.operator_aliases = {alias: operator for operator, properties in operators.items() for alias in properties['alias']}

        self.operator_realizations = {k: v["realization"] for k, v in operators.items()}

        self.operator_arity = {k: v["arity"] for k, v in operators.items()}
        self.operator_arity_compat = deepcopy(self.operator_arity)
        self.operator_arity_compat['**'] = 2
        self.operators = list(self.operator_arity.keys())

        self.modules = get_used_modules(''.join(f"{op}(" for op in self.operator_realizations.values()))
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
        """
        state = self.__dict__.copy()
        del state['_core']
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the engine from pickled state, exactly as ``__init__`` would.

        Mirrors the construction order: realization modules first (a spawn
        worker unpickles into a fresh interpreter), then the compiled core from
        the same in-memory config + rules.
        """
        self.__dict__.update(state)
        self.import_modules()
        self._core = self._build_core(self._operators_config, self.simplification_rules)

    def compile_rules(self) -> None:
        """Sync the compiled core's rule set from ``self.simplification_rules``.

        Public contract (unchanged since the pure-Python engine): after mutating
        ``self.simplification_rules``, call this so ``simplify`` sees the new rules.
        With the Rust-only engine this is a straight ``set_rules`` push to the core.
        """
        self._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in self.simplification_rules])

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

        self.simplification_rules = [rule for rule in full if rule in kept]
        self.compile_rules()
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
        for i, rule in enumerate(self.simplification_rules):
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
                self.simplification_rules[i] = (tuple(lhs), (result_token,))
                n_resolved += 1
        if n_resolved > 0:
            self.compile_rules()
        if verbose:
            print(f'Resolved {n_resolved} constant rules '
                  f'({len(self.simplification_rules)} rules total)')
        return n_resolved

    def import_modules(self) -> None:
        """Imports Python modules required by operator realizations.

        The engine inspects the 'realization' strings of all operators
        (e.g., 'np.sin') to identify necessary modules (e.g., 'numpy') and
        imports them into the global namespace to make them available for
        expression evaluation.
        """
        for module in self.modules:
            if module not in globals():
                globals()[module] = importlib.import_module(module)

    @classmethod
    def from_config(cls, config_path: str) -> "SimpliPyEngine":
        """Creates a SimpliPyEngine instance from a YAML configuration file.

        The configuration file should specify the `operators` and can
        optionally provide a path to a `rules` file.

        Parameters
        ----------
        config_path : str
            The absolute or relative path to the YAML configuration file.

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
        engine = cls(operators=config['operators'], rules=rules)
        # (The compiled core self-attaches in __init__ from the SAME in-memory state.)
        return engine

    @classmethod
    def load(cls, path: str, install: bool = False, local_dir: Path | str | None = None, repo_id: str | None = None, manifest_filename: str | None = None) -> "SimpliPyEngine":
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
        return cls.from_config(get_path(path, install=install, local_dir=local_dir, repo_id=repo_id, manifest_filename=manifest_filename))

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
            node_budget: int = 48,
            inplace: bool = False,
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
        inplace : bool, optional
            Mutate a list input in place (the returned list is the same object).
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
            mode = Mode(mode)  # ints Mode(1)/Mode(3); anything else raises ValueError

        if isinstance(expression, str):
            tokens = self._core.parse(expression, True, False)
        elif isinstance(expression, np.ndarray):
            _validate_ndarray_input(expression, inplace)
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
        return self._denormalize(out, expression, inplace)

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
            inplace: bool) -> str | list[str] | tuple[str, ...] | np.ndarray:
        """Map a core prefix-token result back to the input's type (shared by simplify/mask)."""
        if isinstance(expression, str):
            return self._core.prefix_to_infix(out, '**', False)
        if isinstance(expression, np.ndarray):
            # Re-infer the string WIDTH from the result, keeping only the input dtype KIND: a fold
            # can emit a token wider than any input token (e.g. `1/0 -> float("inf")`), and a fixed
            # `dtype=expression.dtype` (whose width numpy sized to the inputs) would silently truncate.
            return np.array(out).astype(expression.dtype.kind)
        if isinstance(expression, tuple):
            return tuple(out)
        if inplace:
            expression[:] = out
            return expression
        return out

    def _mining_sample_x(self, n_rows: int, n_vars: int, rng: np.random.Generator) -> np.ndarray:
        """Sample the mine's numerical evaluation matrix X.

        Heavy-tailed MIXTURE instead of the historical pure N(0, 5): a mined pattern's
        wildcards bind arbitrary SUBTREE values at application time, so equivalence
        certification must exercise magnitudes far outside a Gaussian's bulk (tanh/exp
        saturation plateaus, huge/tiny magnitudes) and the exact algebraic corner points
        (0, +-1, ...) where false identities like ``asin(cosh(_0)) -> nan`` are refuted.
        Mixture per ELEMENT (so a row can pair a huge x0 with a corner-point x1):

        - 40% N(0, 5) (the historical distribution; dense near typical values)
        - 25% U(-50, 50)
        - 25% signed log-uniform magnitudes 1e-4..1e3 (exposes saturation false-equalities)
        - 10% exact special values {+-0.0, +-0.1, +-0.5, +-1, +-2, +-e, +-pi, +-10}

        Under GENERIC-EQUIVALENCE semantics the corner points
        refute wrong-VALUE identities (``asin(cosh(_0)) -> nan`` is false AT 0, where the
        source is pi/2) while domain EXTENSION remains allowed: where the source is
        NaN/inf the replacement may complete it (``div(_0, _0) -> 1``, the 0/0 limit).

        The log-uniform tier's upper magnitude is 1e3, deliberately not wider: 1e3
        still exercises every saturation (tanh/exp plateau by |x|~40) and f64 overflow
        (exp overflows past |x|~710) that the tier is FOR, while a 1e6 design column
        pushes the constant-fit conditioning to cond(A) ~ 1e6 and the fit cannot
        recover an intercept to rtol at an exact-zero-crossing row -- silently
        rejecting the entire ``C0*f(x)+C1`` affine family.
        """
        shape = (n_rows, n_vars)
        choice = rng.choice(4, size=shape, p=(0.40, 0.25, 0.25, 0.10))
        normal = rng.normal(0.0, 5.0, size=shape)
        uniform = rng.uniform(-50.0, 50.0, size=shape)
        magnitudes = 10.0 ** rng.uniform(-4.0, 3.0, size=shape) * rng.choice((-1.0, 1.0), size=shape)
        specials = np.array([0.0, -0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0,
                             2.0, -2.0, np.e, -np.e, np.pi, -np.pi, 10.0, -10.0])
        special = rng.choice(specials, size=shape)
        return np.where(choice == 0, normal,
                        np.where(choice == 1, uniform,
                                 np.where(choice == 2, magnitudes, special)))

    def _confirm_mined_rules(
            self,
            found: list,
            dummy_variables: list[str],
            X_confirm: np.ndarray,
            constants_fit_challenges: int,
            constants_fit_retries: int,
            rtol: float,
            atol: float,
            min_informative: int,
            seed: int) -> list:
        """STAGE-2 CONFIRMATION: re-verify each mined (source -> target)
        pair on an INDEPENDENT, wider X before it may enter the rule set.

        Same checker, fresh data, fresh constant draws and seeds: this kills data-luck
        accepts (agreement within tolerance only on the mine X, saturation plateaus,
        lucky constant draws) without a second implementation to keep in sync. The
        caller scales ``min_informative`` to the confirm matrix's row count.

        ORDER-INDEPENDENT confirm seed: the seed is a pure function of (confirm seed,
        THIS rule's tokens), never of the rule's POSITION in ``found`` -- position-derived
        seeds reroll every later rule whenever an earlier rule appears or disappears,
        flipping the fit-flaky ones and drowning A/B rule-set diffs in noise. Same rule ->
        same seed, no matter what else was mined; the same policy as the candidate fit
        seed (``worker.rs``).
        """
        assert self._core is not None
        n_rows = X_confirm.shape[0]
        x_flat = X_confirm.flatten(order='C').tolist()
        confirmed = []
        for source, target in found:
            # blake2b, not hash(): PYTHONHASHSEED randomises str hashing per process, which
            # would make the confirm stage irreproducible across runs.
            key = f'{" ".join(source)}->{" ".join(target)}'.encode()
            rule_seed = seed + int.from_bytes(hashlib.blake2b(key, digest_size=7).digest(), 'little')
            result = self._core.find_rule(
                list(source), len(source), None, [list(target)], list(dummy_variables),
                x_flat, n_rows, constants_fit_challenges, constants_fit_retries,
                rule_seed, rtol, atol, min_informative)
            # The result must be THE TARGET WE ASKED ABOUT, not merely "something".
            # `find_rule` does not always answer the question it was asked: its all-constant
            # short-circuit classifies a variable-free `<constant>`-bearing source and returns
            # that CLASS LITERAL without ever reading the candidate list. Testing only
            # `result is not None` therefore made stage-2 confirmation VACUOUS for every such
            # source -- measured: `exp <constant> -> 0`, `exp <constant> -> float("nan")` and
            # `+ <constant> 1 -> float("-inf")` all "confirmed", while the control
            # `+ x0 x0 -> 0` (which does not hit the short-circuit) correctly failed. Since the
            # candidate list here holds exactly one entry, any other answer is the short-circuit
            # speaking, and the honest verdict is "not confirmed".
            if result is not None and list(result) == list(target):
                confirmed.append((source, target))
        return confirmed

    def _mine_tier_with_progress(self, length: int, n_sources: int, mine_call: Callable, verbose: bool) -> Any:
        """Run one length tier's ``mine_one_length`` (a single blocking, rayon-parallel call) while a
        daemon thread polls the core's within-tier ``mining_progress()`` and prints a rate / ETA /
        RSS line, so a long tier is not an opaque wait. Quick early reports (20 / 60 / 180 s), then
        every ``SIMPLIPY_MINE_PROGRESS_INTERVAL`` seconds (default 600). No effect unless ``verbose``."""
        if not verbose:
            return mine_call()
        import threading
        import time
        try:
            interval = float(os.environ.get('SIMPLIPY_MINE_PROGRESS_INTERVAL', '600'))
        except ValueError:
            interval = 600.0

        def _rss_gb() -> float:
            try:
                with open('/proc/self/status') as status:
                    for line in status:
                        if line.startswith('VmRSS:'):
                            return int(line.split()[1]) / 1048576
            except OSError:
                pass
            return float('nan')

        def _fmt(sec: float) -> str:
            if sec != sec or sec == float('inf'):
                return '?'
            isec = int(sec)
            hours, isec = divmod(isec, 3600)
            mins, isec = divmod(isec, 60)
            return f'{hours}h{mins:02d}m' if hours else (f'{mins}m{isec:02d}s' if mins else f'{isec}s')

        t0 = time.perf_counter()

        def _report() -> None:
            done, total = self._core.mining_progress()
            total = total or n_sources
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float('inf')
            pct = 100.0 * done / total if total else 0.0
            print(f'  Length {length}: {done:,}/{total:,} sources ({pct:.1f}%) | {rate:,.0f} src/s'
                  f' | ETA {_fmt(eta)} | elapsed {_fmt(elapsed)} | RSS {_rss_gb():.1f} GB', flush=True)

        stop = threading.Event()

        def _monitor() -> None:
            schedule = [20.0, 60.0, 180.0]   # quick early confirmations that it is alive + moving
            i, last = 0, 0.0
            while not stop.is_set():
                elapsed = time.perf_counter() - t0
                due = schedule[i] if i < len(schedule) else last + interval
                if elapsed >= due:
                    _report()
                    last = elapsed
                    if i < len(schedule):
                        i += 1
                stop.wait(5.0)

        monitor = threading.Thread(target=_monitor, daemon=True)
        monitor.start()
        try:
            return mine_call()
        finally:
            stop.set()
            monitor.join(timeout=2.0)
            _report()   # final within-tier line before the end-of-tier summary

    def _find_rules_native(
            self,
            expressions_of_length: dict[int, set[tuple[str, ...]]],
            max_source_pattern_length: int,
            max_target_pattern_length: int | None,
            dummy_variables: list[str],
            X_data: np.ndarray,
            constants_fit_challenges: int,
            constants_fit_retries: int,
            output_file: str | None,
            prune: bool | str,
            verbose: bool,
            interrupted: Callable[[], bool],
            rtol: float = 1e-9,
            atol: float = 1e-12,
            min_informative: int = 128,
            mine_seed: int = 42,
            confirm_seed: int = 43,
            X_confirm: np.ndarray | None = None,
            candidate_fold_filter: bool = True,
            relaxed_kruskal: bool = True,
            provenance: dict | None = None,
            proposal_entries: list[tuple[tuple[str, ...], tuple[str, ...] | None]] | None = None,
            leaf_nodes: list[str] | None = None,
            promote_sorts: bool = True,
            symbolic_gate: bool = True,
            snapshot_at: dict[int, str] | None = None) -> None:
        """Phase 2 of :meth:`find_rules` on the compiled Rust core (``simplipy._core``).

        Mirrors the pure-Python worker pool, but correctly against the core: per source
        length (shortest first), each source is Kruskal-pruned against the rules found so
        far and searched for an equivalent shorter candidate; the growing rule set is
        pushed into the core between lengths (``set_rules``), which the Python pool cannot
        do (the core is immutable from forked workers, so it would mine nothing).
        Parallelism is rayon over all cores; cap it with ``RAYON_NUM_THREADS``.

        ``proposal_entries`` (with its token universe ``leaf_nodes``) is the optional
        proposal channel: after the length loop completes and BEFORE the optional prune
        (a certified proposal is a rule like any other and must survive or fall in the
        same prune), each proposal is certified against the just-mined state via
        :meth:`_certify_proposals`, reusing this mine's candidate library, evaluation
        matrices and seeds.

        ``snapshot_at`` emits the SHORTER CELLS the climb passes through (see
        :meth:`find_rules`). The post-mine stages are factored into one closure precisely
        so that every cell -- snapshotted or final -- goes through the identical block.
        """
        assert self._core is not None

        if max_target_pattern_length is None:
            # Any strictly shorter candidate is a valid replacement.
            max_candidate_length = max_source_pattern_length - 1
        else:
            max_candidate_length = max_target_pattern_length

        # `sorted` inner iteration: set order is PYTHONHASHSEED-dependent, and the checker
        # accepts the FIRST passing candidate -- unsorted iteration makes the mined rule set
        # non-reproducible run-to-run.
        candidates = [
            list(expression)
            for length in sorted(expressions_of_length)
            if length <= max_candidate_length
            for expression in sorted(expressions_of_length[length])
        ]
        library = self._core.build_candidate_library(
            candidates, list(dummy_variables), X_data.flatten(order='C').tolist(), X_data.shape[0],
            fold_filter=candidate_fold_filter)
        if verbose:
            print(f'Candidate library: {library.n_candidates:,} candidates'
                  f' ({library.n_filtered:,} var-free filtered)')

        # Existing rules (empty after reset_rules=True) join the Kruskal pruning and the
        # final deduplicated set, like the pure-Python path.
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.simplification_rules]

        # --- The post-pass: what turns a raw mined rule set into a CELL ARTIFACT ---
        # A closure, because the ladder runs it MORE THAN ONCE. A (7,4) climb passes through
        # the (5,4) and (6,4) cells on its way up and each of those is a publishable artifact
        # in its own right (see `snapshot_at`). Running the IDENTICAL block for every cell is
        # what makes a snapshot equal to a one-shot mine of that cell; a second copy of these
        # stages would be free to drift from this one.
        def _finalize(mined: list, out_path: str | None, prov: dict | None) -> list:
            self.simplification_rules = list(mined)
            self.compile_rules()
            self._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in mined])
            # Proposal channel: certify externally proposed rules against the just-mined
            # state, BEFORE the optional prune. Skipped on interrupt: a partial mine is not
            # the rule state the proposals were aimed at, and a clean abort must stay cheap.
            # An explicitly-given EMPTY batch still runs (and records all-zero outcome
            # counts): the sidecar must show the channel ran, not silently omit it.
            if proposal_entries is not None and not interrupted():
                outcomes, trail = self._certify_proposals(
                    proposal_entries, library,
                    leaf_nodes if leaf_nodes is not None else list(dummy_variables),
                    dummy_variables, X_data, X_confirm, max_target_pattern_length,
                    constants_fit_challenges, constants_fit_retries, rtol, atol,
                    min_informative, mine_seed, confirm_seed, verbose)
                if prov is not None and 'proposals' in prov:
                    prov['proposals']['outcomes'] = outcomes
                    # The per-candidate verdict trail travels WITH the artifact: an aggregate
                    # tally cannot be audited (it never says which candidate died at which gate).
                    prov['proposals']['trail'] = trail
            # SYMBOLIC GATE (the third authority; owner-approved 2026-08-02): every
            # confirmed rule is re-judged by the independent symbolic verifier
            # (`simplipy.verify`) at exact symbolic trigger points with the
            # precision-stability discriminator -- a residual that COLLAPSES as
            # precision doubles is computational rounding (exact identity,
            # CERTIFIED); one that is STABLE is a real gap between two functions
            # (KILL), however small. This is the only instrument that can separate
            # `sin pi -> 0` (true) from `tanh(exp pi) -> 1` (false by 1.5e-20):
            # every numeric acceptance is tolerance-bounded, and the impostor
            # family lives below any usable tolerance. KILLs are dropped here,
            # BEFORE the prune and promotion see the set; TOLERATED (null-set
            # extension doctrine, matching the engine's own licences) passes.
            if symbolic_gate and not interrupted() and self.simplification_rules:
                from .verify import verify_ruleset
                _rep = verify_ruleset(
                    [[list(lhs), list(rhs)] for lhs, rhs in self.simplification_rules])
                _detail = _rep['detail'] if isinstance(_rep, dict) else _rep
                # detail entries carry lhs/rhs as SPACE-JOINED strings.
                _kills = {(tuple(d['lhs'].split()), tuple(d['rhs'].split()))
                          for d in _detail if d.get('verdict') == 'KILL'}
                if _kills:
                    self.simplification_rules = [
                        (lhs, rhs) for lhs, rhs in self.simplification_rules
                        if (tuple(lhs), tuple(rhs)) not in _kills]
                    self.compile_rules()
                    self._core.set_rules(
                        [(list(lhs), list(rhs)) for lhs, rhs in self.simplification_rules])
                if verbose:
                    print(f'Symbolic gate: {len(_kills)} killed, '
                          f'{len(self.simplification_rules)} remain')
                if prov is not None:
                    prov['symbolic_gate'] = {
                        'killed': sorted([list(kl), list(kr)] for kl, kr in _kills),
                        'kept': len(self.simplification_rules)}
            if prune == 'covered':
                self.prune_covered_rules(verbose=verbose)
            elif prune:
                raise ValueError(
                    "prune=True (the redundant-rule prune) died with the legacy kernel; "
                    "use prune='covered' (AC-judged behavioral coverage) or prune=False")
            # Sort promotion: re-certify every rule (mined + proposed) at the stronger `_`/`!`
            # sorts and ship it at the strongest sound one (see `simplipy.promotion`). Runs AFTER
            # the prune so promotion works on the already-reduced rule set (the promotion carries
            # its own subsumption/derivability refund for the instances a promoted rule newly
            # covers). Emits the deployment-strength ruleset directly.
            if promote_sorts and not interrupted():
                from .promotion import promote
                kept, promotion_report = promote(self.simplification_rules, self)
                self.simplification_rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in kept]
                self.compile_rules()
                self._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in self.simplification_rules])
                if verbose:
                    print(f'Sort promotion: {promotion_report.get("stage_counts")}')
                if prov is not None:
                    prov['sort_promotion'] = promotion_report.get('stage_counts')
                # POST-PROMOTION covered-prune (fold unification, 2026-08-02): the
                # ladder's seeded identities (`* 0 !0 -> 0`, the multiplicative
                # twins) and the promoted sorts only exist AFTER promotion, so the
                # pre-promotion prune cannot see that thousands of pure-literal
                # ground rows (`* 0 cos 1 -> 0`) are wildcard-derivable instances.
                # A second prune with the final ruleset live removes exactly the
                # redundant instances; genuinely novel ground rules (`cos 0 -> 1`)
                # have no wildcard cover and survive.
                if prune == 'covered':
                    self.prune_covered_rules(verbose=verbose)
            # SOUNDNESS PROVENANCE, second half (audit Tier-1 #4): the interval layer's
            # three fail-closed miss counters, read AFTER the mine as deltas over this
            # mine's baseline -- the exposure ("how often could the gate not decide?")
            # is a recorded number, never assumed zero. The counters mark UNDECIDED
            # verdicts whose callers rejected (lost recall, not lost soundness).
            if prov is not None:
                prov['soundness']['interval_undecided'] = {
                    'horizon': self._core.interval_horizon_misses() - _ivl_baseline[0],
                    'node_budget': self._core.interval_node_budget_misses() - _ivl_baseline[1],
                    'unanalyzable': self._core.interval_unanalyzable_misses() - _ivl_baseline[2],
                }
            if out_path is not None:
                with open(out_path, 'w') as file:
                    json.dump(self.simplification_rules, file, indent=4)
                self._write_provenance(out_path, prov, self.simplification_rules, final=True)
            return [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.simplification_rules]

        snapshot_at = dict(snapshot_at or {})

        # SOUNDNESS PROVENANCE, first half (audit Tier-1 #4): the four default-ON
        # soundness kill-switches (ablation/repro escape hatches; `<var>=0` disables a
        # LAYER) ship recorded in the artifact -- a mine run with a layer off must say
        # so -- plus the node-budget override if any. Recorded up front so interrupted
        # mines carry it; the switch semantics mirror the rust side exactly
        # (on unless the variable is literally "0").
        _ivl_baseline = (self._core.interval_horizon_misses(),
                         self._core.interval_node_budget_misses(),
                         self._core.interval_unanalyzable_misses())
        if provenance is not None:
            provenance['soundness'] = {
                'kill_switches': {
                    var: os.environ.get(var) != '0'
                    for var in ('SIMPLIPY_IVL_GATE', 'SIMPLIPY_IVL_CLASS',
                                'SIMPLIPY_IVL_REACH', 'SIMPLIPY_SPECIAL_BATTERY')},
                'node_budget_env': os.environ.get('SIMPLIPY_IVL_NODE_BUDGET'),
                # EVERY set entry of the artifact-affecting switch registry, raw (H-042):
                # a default run records {}, any override is visible verbatim.
                'env_overrides': {
                    var: os.environ[var]
                    for var in ARTIFACT_ENV_SWITCHES if var in os.environ},
                'interval_undecided': None,  # filled at finalize (deltas over this mine)
            }

        # Sources: lengths up to max_source_pattern_length only (`construct_expressions`
        # over-produces a tail of longer expressions beyond the documented contract).
        for length in sorted(k for k in list(expressions_of_length) if k <= max_source_pattern_length):
            if interrupted():
                break
            self._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in rules])
            sources = [list(expression) for expression in sorted(expressions_of_length[length])]
            n_sources = len(sources)
            # Release this length's source UNIVERSE as soon as it has been linearised: at the
            # top lengths that set of tuples is the largest object in the process (length 5
            # alone is ~7e6 expressions) and nothing reads it again -- the candidate library
            # was built up front, above, and holds its own copy on the Rust side.
            del expressions_of_length[length]
            # Per-length seed block: lengths are spaced by 2^40 (far above any per-length
            # source count) so the per-source streams (seed + index) never collide.
            found = self._mine_tier_with_progress(
                length, len(sources),
                lambda: self._core.mine_one_length(
                    sources, library, max_target_pattern_length,
                    constants_fit_challenges, constants_fit_retries,
                    mine_seed + (length << 40), rtol, atol, min_informative,
                    relaxed_kruskal),
                verbose)
            if found and X_confirm is not None:
                n_mined = len(found)
                found = self._confirm_mined_rules(
                    found, dummy_variables, X_confirm,
                    constants_fit_challenges, constants_fit_retries, rtol, atol,
                    max(1, round(min_informative * X_confirm.shape[0] / X_data.shape[0])),
                    confirm_seed + (length << 40))
                if verbose and len(found) < n_mined:
                    print(f'Length {length}: stage-2 confirmation rejected {n_mined - len(found):,} of {n_mined:,} mined rules')
            if found:
                rules = deduplicate_rules(
                    rules + [(tuple(lhs), tuple(rhs)) for lhs, rhs in found],
                    dummy_variables, verbose=verbose)
            if verbose:
                print(f'Length {length}: {n_sources:,} sources, {len(found):,} new rules, {len(rules):,} total')
            if output_file is not None:
                with open(output_file, 'w') as file:
                    json.dump(rules, file, indent=4)
                self._write_provenance(output_file, provenance, rules, completed_length=length)
            # LADDER SNAPSHOT: finishing this length also finishes the shorter cell
            # (length, max_target_pattern_length) of the same ladder. Emit it as a completed
            # artifact, then restore the RAW state and keep climbing -- the taller cell must
            # continue from the un-pruned, un-promoted mine, which is exactly what a one-shot
            # run of it does.
            if length in snapshot_at and length < max_source_pattern_length:
                if verbose:
                    print(f'--- ladder snapshot: cell ({length},{max_target_pattern_length})'
                          f' -> {snapshot_at[length]}')
                _finalize(list(rules), snapshot_at[length],
                          self._snapshot_provenance(provenance, length, max_source_pattern_length))
                # `_finalize` prunes/promotes the engine's own rule state; put the raw mine back.
                self.simplification_rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
                self.compile_rules()
                self._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in rules])

        _finalize(rules, output_file, provenance)

    @staticmethod
    def _snapshot_provenance(provenance: dict | None, length: int, climb_max: int) -> dict | None:
        """Provenance for a LADDER SNAPSHOT -- the sidecar of cell ``(length, j)`` as emitted
        mid-climb by a taller mine (see ``snapshot_at`` in :meth:`find_rules`).

        The rule set is identical to a one-shot mine of the cell, so ``params`` describes the
        CELL rather than the climb (and the universe census is trimmed to the lengths this
        cell actually covers -- it must not claim to have sampled a length above its own).
        But the sidecar also states plainly that it came from a climb: an artifact may never
        overstate how it was produced.
        """
        if provenance is None:
            return None
        prov = deepcopy(provenance)
        prov['params']['max_source_pattern_length'] = length
        prov['params']['source_sample_per_length'] = {
            k: v for k, v in prov['params'].get('source_sample_per_length', {}).items()
            if int(k) <= length}
        prov['universe'] = {k: v for k, v in prov.get('universe', {}).items() if int(k) <= length}
        prov['ladder_snapshot'] = {
            'emitted_at_source_length': length,
            'climb_max_source_pattern_length': climb_max,
            'equivalence': 'Lengths are mined shortest-first, the per-source seed is '
                           'mine_seed + (length << 40), and the candidate library is bounded by '
                           'max_target_pattern_length -- so this prefix does not depend on how far '
                           'the climb continues, and the rule set equals a one-shot mine of this cell.',
        }
        return prov

    #: Probe expressions whose complexities identify the MEASURE. Chosen to separate the
    #: changes the measure has actually undergone: an integer (the L-formula), a unit
    #: fraction and a non-unit one (the fraction code and the inversion bit), a decimal
    #: whose denominator carries a five (the print/argmin split), a `<constant>` (mu_free),
    #: and a bare symbol (mu_sym).
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

    @staticmethod
    def _write_provenance(output_file: str, provenance: dict | None, rules: list,
                          completed_length: int | None = None, final: bool = False) -> None:
        """Write/refresh the mined artifact's PROVENANCE sidecar (`<output>.provenance.json`).

        The rules.json format is a bare list (the engine loads it directly), so provenance
        lives beside it: parameters, seeds, X spec, universe coverage (filled by
        :meth:`find_rules`) plus rolling progress and the final rule census. A mine is
        reproducible from the sidecar alone unless X was passed as an explicit array
        (recorded by content hash in that case).
        """
        if provenance is None:
            return
        import time as _time
        by_len: dict[str, int] = {}
        for lhs, _ in rules:
            by_len[str(len(lhs))] = by_len.get(str(len(lhs)), 0) + 1
        provenance['progress'] = {
            'updated': _time.strftime('%Y-%m-%d %H:%M:%S %z'),
            'last_completed_source_length': completed_length if not final
            else provenance.get('progress', {}).get('last_completed_source_length'),
            'final': final,
            'rules_total': len(rules),
            'rules_by_lhs_length': dict(sorted(by_len.items(), key=lambda x: int(x[0]))),
        }
        with open(output_file + '.provenance.json', 'w') as file:
            json.dump(provenance, file, indent=2)

    def _build_source_universe(
            self,
            leaf_nodes: list[str],
            non_leaf_nodes: dict[str, int],
            max_source_pattern_length: int,
            max_target_pattern_length: int | None,
            source_sample_per_length: dict[int, int],
            rng: np.random.Generator,
            verbose: bool) -> tuple[dict[int, set[tuple[str, ...]]], dict[int, int]]:
        """Phase 1 of :meth:`find_rules`: build the per-length source/candidate universe.

        COMPLETE bottom-up DP enumeration per length: enumeration must SATURATE each
        length, not merely reach it (a pass-based closure that stops once
        ``max_source_pattern_length`` is reached silently misses whole expression
        families). Lengths whose complete universe is infeasible are drawn as a seeded
        uniform sample from the complete universe instead
        (``source_sample_per_length``).

        Returns ``(expressions_of_length, counts)``: the enumerated-or-sampled universe
        per length, and the complete-universe count DP.
        """
        counts = count_expressions(len(leaf_nodes), non_leaf_nodes, max_source_pattern_length)
        enumerate_max = max(
            (length for length in range(1, max_source_pattern_length + 1)
             if length not in source_sample_per_length),
            default=1)
        if verbose:
            print(f"Phase 1: enumerating all expressions up to length {enumerate_max}"
                  + (f", sampling lengths {sorted(source_sample_per_length)}" if source_sample_per_length else ""))

        expressions_of_length = enumerate_expressions(leaf_nodes, non_leaf_nodes, enumerate_max)
        # Two paths, one truth: the enumerated universe must match the count DP exactly.
        for length, expressions in expressions_of_length.items():
            if len(expressions) != counts[length]:
                raise AssertionError(
                    f'enumeration incomplete at length {length}: {len(expressions):,} != {counts[length]:,}')

        max_candidate_length = (max_source_pattern_length - 1 if max_target_pattern_length is None
                                else max_target_pattern_length)
        for length in sorted(source_sample_per_length):
            if not 1 < length <= max_source_pattern_length:
                raise ValueError(f'source_sample_per_length length {length} outside 2..{max_source_pattern_length}')
            if length <= max_candidate_length:
                warnings.warn(
                    f'length {length} is SAMPLED but lies inside the candidate/replacement range '
                    f'(<= {max_candidate_length}): the candidate library will be INCOMPLETE and '
                    f'shorter equivalents can be missed', UserWarning)
            target = min(source_sample_per_length[length], counts[length])
            draws: set[tuple[str, ...]] = set()
            attempts = 0
            while len(draws) < target and attempts < 20 * target:
                draws.add(sample_expression(length, leaf_nodes, non_leaf_nodes, counts, rng))
                attempts += 1
            # Per-run sampler cross-check: the count-DP assertion above
            # guards ENUMERATED lengths only, so validate every draw's membership in the
            # intended universe (exact length, known tokens, well-formed arity).
            vocabulary = set(leaf_nodes) | set(self.operator_arity)
            for expression in draws:
                if (len(expression) != length or not set(expression) <= vocabulary
                        or not self.is_valid(list(expression))):
                    raise AssertionError(f'sampler produced a non-member at length {length}: {expression}')
            expressions_of_length[length] = draws
            # NO silent caps: always state the achieved coverage.
            print(f'Phase 1: length {length} SAMPLED: {len(draws):,} of {counts[length]:,} '
                  f'({len(draws) / counts[length]:.3%} of the complete universe)')

        return expressions_of_length, counts

    def _build_mine_provenance(
            self, *,
            X: np.ndarray | int | None,
            X_data: np.ndarray,
            counts: dict[int, int],
            expressions_of_length: dict[int, set[tuple[str, ...]]],
            source_sample_per_length: dict[int, int],
            max_source_pattern_length: int,
            max_target_pattern_length: int | None,
            dummy_variables: list[str],
            extra_internal_terms: list[str],
            constants_fit_challenges: int,
            constants_fit_retries: int,
            rtol: float,
            atol: float,
            min_informative: int,
            seed: int | None,
            mine_seed: int,
            confirm_seed: int,
            confirm: bool,
            candidate_fold_filter: bool,
            relaxed_kruskal: bool,
            prune: bool | str,
            reset_rules: bool) -> dict:
        """Assemble the mined artifact's PROVENANCE record: the mine must
        be reproducible from its sidecar alone. Everything that determines the mine is
        recorded: parameters, seeds, the X specification (seed-derived, or content-hashed
        when an explicit array was passed -- the one case a seed cannot reproduce), universe
        counts and coverage. Written beside the rules by :meth:`_write_provenance`.
        """
        import hashlib
        import platform
        import time as _time
        from simplipy import __version__ as _simplipy_version
        try:
            from simplipy import _core as _core_mod
            _core_build = getattr(_core_mod, '__build__', None)
        except ImportError:
            _core_build = None
        if X is None or isinstance(X, int):
            x_spec: dict = {'rows': int(X_data.shape[0]), 'cols': int(X_data.shape[1]),
                            'source': 'seeded_mixture (_mining_sample_x from `seed`)'}
        else:
            x_spec = {'rows': int(X_data.shape[0]), 'cols': int(X_data.shape[1]),
                      'source': 'explicit_array (NOT reproducible from `seed`)',
                      'sha256': hashlib.sha256(np.ascontiguousarray(X_data).tobytes()).hexdigest()}
        return {
            'created': _time.strftime('%Y-%m-%d %H:%M:%S %z'),
            'host': platform.node(),
            'simplipy_version': _simplipy_version,
            'core_build': _core_build,
            'measure': self._measure_fingerprint(),
            'params': {
                'max_source_pattern_length': max_source_pattern_length,
                'max_target_pattern_length': max_target_pattern_length,
                'dummy_variables': list(dummy_variables),
                'extra_internal_terms': list(extra_internal_terms),
                'constants_fit_challenges': constants_fit_challenges,
                'constants_fit_retries': constants_fit_retries,
                'rtol': rtol, 'atol': atol, 'min_informative': min_informative,
                'seed': seed, 'mine_seed': mine_seed, 'confirm_seed': confirm_seed,
                'confirm': confirm, 'candidate_fold_filter': candidate_fold_filter,
                'relaxed_kruskal': relaxed_kruskal,
                'prune': prune, 'reset_rules': reset_rules,
                'source_sample_per_length': {str(k): int(v) for k, v in source_sample_per_length.items()},
            },
            'X': x_spec,
            'universe': {
                str(length): {
                    'complete_count': int(counts[length]),
                    'used': len(expressions_of_length[length]),
                    # An empty universe cell (e.g. even lengths under a binary-only
                    # alphabet) is vacuously covered: all zero of its expressions were used.
                    'coverage': (len(expressions_of_length[length]) / counts[length]
                                 if counts[length] else 1.0),
                    'sampled': length in source_sample_per_length,
                } for length in sorted(expressions_of_length)
            },
            'operators': sorted(self.operator_arity),
        }

    @staticmethod
    def _resolve_dummy_variables(
            dummy_variables: int | list[str] | None,
            *,
            default: Callable[[], list[str]]) -> list[str]:
        """Normalize a ``dummy_variables`` argument to a concrete variable-name list.

        ``None`` -> ``default()`` (each caller supplies its own policy: :meth:`find_rules`
        derives the count from ``max_source_pattern_length``, :meth:`certify_rules` collects
        the variables named in the proposed sources); an ``int`` -> ``['x0', ...]`` of that
        length; a list is passed through unchanged.
        """
        if dummy_variables is None:
            return default()
        if isinstance(dummy_variables, int):
            return [f'x{i}' for i in range(dummy_variables)]
        return dummy_variables

    def certify_rules(
            self,
            sources: list,
            targets: list | None = None,
            max_target_pattern_length: int = 4,
            dummy_variables: int | list[str] | None = None,
            extra_internal_terms: list[str] | None = None,
            X: int | None = None,
            constants_fit_challenges: int = 16,
            constants_fit_retries: int = 16,
            rtol: float = 1e-9,
            atol: float = 1e-12,
            min_informative: int | None = None,
            seed: int | None = 42,
            verbose: bool = False) -> list[tuple[tuple[str, ...], tuple[str, ...], str]]:
        """Certify externally proposed simplification rules with the mining gates.

        For each proposed ``source`` expression this runs the exact certification chain
        the miner uses: validity check, then a shortest-first scan of the complete
        candidate library up to ``max_target_pattern_length`` with the engine's OWN
        result as the mark to beat -- only targets strictly below that mark in the
        serve-time reduction ordering are selectable (yielding a certified MINIMAL
        target) -- and re-verification on an independent, twice-as-wide evaluation
        matrix. If no library target exists and a parallel ``targets`` hint is given,
        the specific (source, target) pair is verified instead (against the same
        ordering mark) -- sound, but without a minimality certificate. A source the
        engine already reduces is still SEARCHED: coverage is decided by the search,
        never by the reduction alone (a canon respell used to read as "covered" and
        silently discarded certifiable proposals, finding F2).

        Intended for LLM- or human-proposed identities: proposals only ever ADD source
        expressions, so certified output is exactly as sound as mined rules.

        Parameters mirror :meth:`find_rules`. The engine is not modified; returns a list
        of ``(source, target, certificate)`` tuples with certificate ``'minimal'`` or
        ``'verified'``. Merge accepted pairs explicitly, e.g.::

            pairs = engine.certify_rules(proposals, hints)
            engine.simplification_rules = deduplicate_rules(
                engine.simplification_rules + [(s, t) for s, t, _ in pairs], dummies)
            engine.compile_rules()
        """
        sources = [tuple(str(t) for t in s) for s in sources]
        hint_list: list = list(targets) if targets is not None else [None] * len(sources)
        if len(hint_list) != len(sources):
            raise ValueError('targets must parallel sources')
        dummy_variables = self._resolve_dummy_variables(
            dummy_variables,
            default=lambda: (sorted({t for s in sources for t in s
                                     if t.startswith('x') and t[1:].isdigit()},
                                    key=lambda v: int(v[1:])) or ['x0']))
        extra_internal_terms = extra_internal_terms or []

        _rng = np.random.default_rng(seed)
        X_data = self._mining_sample_x(X or 1024, len(dummy_variables), _rng)
        X_confirm = self._mining_sample_x(2 * X_data.shape[0], len(dummy_variables), _rng)
        mine_seed = int(_rng.integers(2 ** 62))
        confirm_seed = int(_rng.integers(2 ** 62))
        if min_informative is None:
            # Floor at 1 (mirrors the rust FFI default `(n_rows/8).max(1)`): a small X
            # (< 8 rows) must never yield 0, which would disable the evidence gate and
            # re-admit the vacuous all-NaN/inf accepts it exists to kill.
            min_informative = max(X_data.shape[0] // 8, 1)

        leaf_nodes = list(dummy_variables) + [t for t in extra_internal_terms
                                              if t not in dummy_variables]
        if '<constant>' not in leaf_nodes:
            leaf_nodes.append('<constant>')
        non_leaf_nodes = dict(sorted(self.operator_arity.items(), key=lambda x: x[1]))
        expressions = enumerate_expressions(leaf_nodes, non_leaf_nodes, max_target_pattern_length)
        candidates = [list(expression)
                      for length in sorted(expressions)
                      for expression in sorted(expressions[length])]
        library = self._core.build_candidate_library(
            candidates, list(dummy_variables), X_data.flatten(order='C').tolist(),
            X_data.shape[0], fold_filter=True)
        if verbose:
            print(f'Candidate library: {library.n_candidates:,} candidates '
                  f'({library.n_filtered:,} var-free filtered)')

        vocabulary = set(leaf_nodes) | set(self.operator_arity)
        certified: list = []
        confirm_min = max(1, round(min_informative * X_confirm.shape[0] / X_data.shape[0]))
        for index, (source, hint) in enumerate(zip(sources, hint_list)):
            if not _tokens_in_vocabulary(source, vocabulary) or not self.is_valid(list(source)):
                continue
            # The engine's own result is the MARK TO BEAT, never a reason to skip
            # (finding F2): a source canon merely RESPELLS shorter used to read as
            # "already covered" and was never searched, losing e.g.
            # exp(t)*exp(t) -> exp(t+t) forever (the respelled form speaks tokens
            # outside the mining alphabet, so no later tier picks it up). The
            # serve-time ordering acceptance rides the candidate scan via `mark`.
            _, _, ac_out = self._core.ac_judge(list(source), 48)
            target = self._core.find_rule_lib(
                list(source), len(source), max_target_pattern_length, library,
                challenges=constants_fit_challenges, retries=constants_fit_retries,
                seed=mine_seed + (len(source) << 40) + index, rtol=rtol, atol=atol,
                min_informative=min_informative, mark=ac_out)
            certificate = 'minimal'
            if target is None and hint is not None:
                hint = [str(t) for t in hint]
                if (_tokens_in_vocabulary(hint, vocabulary) and len(hint) < len(source)
                        and self.is_valid(list(hint))
                        and not violates_wildcard_multiplicity(list(source), list(hint))
                        # the Const-count invariant (one invariant, every layer: the
                        # scan's mint gate, the translate-time count gate, and this hint
                        # arm): a hint may never INCREASE the number of `<constant>`
                        # placeholders -- abstraction is masking's job downstream. A
                        # count-increasing hint is dead on arrival at translation; the
                        # library scan may still find (and literal-resolve) a target.
                        and hint.count('<constant>') <= list(source).count('<constant>')
                        # the hint too must sit strictly below the engine's own
                        # result, or the minted rule is dead on arrival (the serving
                        # pass would refuse it; translation would drop it)
                        and self._core.ac_ordered_below(list(hint), ac_out)
                        and self._confirm_mined_rules(
                            [(source, tuple(hint))], dummy_variables, X_data,
                            constants_fit_challenges, constants_fit_retries, rtol, atol,
                            min_informative, mine_seed + (len(source) << 40) + index)):
                    target = hint
                    certificate = 'verified'
            if target is None:
                continue
            if self._confirm_mined_rules(
                    [(source, tuple(target))], dummy_variables, X_confirm,
                    constants_fit_challenges, constants_fit_retries, rtol, atol,
                    confirm_min, confirm_seed + (len(source) << 40) + index):
                certified.append((source, tuple(target), certificate))
                if verbose:
                    print(f'certified ({certificate}): {list(source)} -> {list(target)}')
        return certified

    def _certify_proposals(
            self,
            proposal_entries: list[tuple[tuple[str, ...], tuple[str, ...] | None]],
            library: Any,
            leaf_nodes: list[str],
            dummy_variables: list[str],
            X_data: np.ndarray,
            X_confirm: np.ndarray | None,
            max_target_pattern_length: int | None,
            constants_fit_challenges: int,
            constants_fit_retries: int,
            rtol: float,
            atol: float,
            min_informative: int,
            mine_seed: int,
            confirm_seed: int,
            verbose: bool) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """The proposal channel of :meth:`find_rules`: certify externally proposed rules
        against the just-mined rule state and merge the survivors.

        PLUMBING ONLY -- no new certification semantics. Each proposal runs the exact
        chain of :meth:`certify_rules` (vocabulary/validity check; shortest-first scan
        of THIS mine's candidate library with the engine's own result as the ordering
        mark to beat; hint verification against the same mark as the fallback;
        independent stage-2 confirmation)
        with THIS mine's evaluation matrices, challenge counts and tolerances, so a
        certified proposal is precisely as sound as a mined rule. ``X_confirm`` is None
        exactly when the mine ran with ``confirm=False``; proposals then skip stage-2
        like every mined rule of the same run.

        DETERMINISM: proposals are processed in file order against the FIXED just-mined
        state (:meth:`certify_rules` likewise never mutates the engine mid-batch), and
        per-proposal seeds are CONTENT-derived (blake2b of the source tokens on top of
        the master-derived ``mine_seed``/``confirm_seed``, the stage-2-confirm policy)
        -- never position-derived, so editing the proposals file cannot reroll the
        certification of untouched proposals.

        The certified pairs then join the ruleset through the same
        :func:`~simplipy.utils.deduplicate_rules` path as mined rules (shortest target
        per canonical source). Mutates the engine in place (rules, compiled state, core
        rules) and returns ``(counts, trail)``.

        ``counts`` is the per-outcome tally for the provenance sidecar: ``certified`` /
        ``already_covered`` / ``rejected`` / ``duplicate`` (certified, but canonically
        identical to an earlier certified proposal and not shorter).

        ``trail`` is the PER-PROPOSAL verdict record, one entry per input proposal in file
        order: ``{source, target, verdict, stage, certificate}``. A tally alone cannot be
        audited -- "93 rejected" does not say WHICH candidates died or at WHICH gate, so a
        reviewer cannot tell a correctly-killed hallucination from a wrongly-killed identity.
        ``stage`` names the gate that decided: ``vocabulary`` (token outside the engine's
        alphabet, or malformed), ``covered`` (SEARCH-VERIFIED: the engine already reduces
        the source and neither the library nor the hint beats the engine's own result),
        ``search`` (no target in the candidate library and no verifiable hint), ``confirm``
        (failed independent stage-2 re-verification), ``merge`` (certified but folded away as
        a duplicate), or ``accepted``. ``certificate`` is ``minimal`` (found by library scan)
        or ``verified`` (the proposal's own hint, confirmed).
        """
        assert self._core is not None
        counts = {'certified': 0, 'already_covered': 0, 'rejected': 0, 'duplicate': 0}
        vocabulary = set(leaf_nodes) | set(self.operator_arity)
        confirm_min = (max(1, round(min_informative * X_confirm.shape[0] / X_data.shape[0]))
                       if X_confirm is not None else None)
        certified_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        trail: list[dict[str, Any]] = []
        # Index into `trail` for each certified pair, so the post-fold duplicate accounting
        # below can flip that proposal's verdict without re-deriving which entry it was.
        certified_trail_idx: list[int] = []

        def record(source: tuple[str, ...], target: Any, verdict: str, stage: str,
                   certificate: str | None = None) -> int:
            trail.append({'source': list(source),
                          'target': list(target) if target is not None else None,
                          'verdict': verdict, 'stage': stage, 'certificate': certificate})
            return len(trail) - 1

        for source, hint in proposal_entries:
            if not _tokens_in_vocabulary(source, vocabulary) or not self.is_valid(list(source)):
                counts['rejected'] += 1
                record(source, None, 'rejected', 'vocabulary')
                continue
            # The engine's own result is the MARK TO BEAT, never a reason to skip
            # (finding F2): the serve-time ordering acceptance rides the candidate
            # scan via `mark`, so 'already_covered' below is a SEARCH-VERIFIED
            # verdict ("the engine reduces it and nothing expressible beats its
            # result"), not a guess from a spelling shrink.
            _, _, ac_out = self._core.ac_judge(list(source), 48)
            # "The engine already reduces it" is judged in the serve ordering (ONE
            # coverage ordering, everywhere): the walk strictly descends, or the
            # result is atomic (nothing mintable sits strictly below a single-token
            # state -- canon-owned collapses like `+ x0 0` land here). A respell --
            # same state, shorter spelling -- is neither: if the search also comes
            # back empty-handed, the honest verdict below is 'rejected', never a
            # coverage claim nothing backs.
            engine_reduces = (len(ac_out) == 1
                              or self._core.ac_ordered_below(ac_out, list(source)))
            simplified_length = len(source)
            # Content-derived per-proposal seed offset (blake2b, not hash(): PYTHONHASHSEED
            # randomises str hashing per process; same policy as _confirm_mined_rules).
            offset = int.from_bytes(
                hashlib.blake2b(' '.join(source).encode(), digest_size=7).digest(), 'little')
            target = self._core.find_rule_lib(
                list(source), simplified_length, max_target_pattern_length, library,
                challenges=constants_fit_challenges, retries=constants_fit_retries,
                seed=mine_seed + offset, rtol=rtol, atol=atol,
                min_informative=min_informative, mark=ac_out)
            certificate = 'minimal'
            if target is None and hint is not None:
                hint_tokens = [str(token) for token in hint]
                if (_tokens_in_vocabulary(hint_tokens, vocabulary) and len(hint_tokens) < len(source)
                        and self.is_valid(list(hint_tokens))
                        and not violates_wildcard_multiplicity(list(source), list(hint_tokens))
                        # the Const-count invariant (same gate as certify_rules' hint
                        # arm): a count-increasing hint is dead on arrival at translation
                        and hint_tokens.count('<constant>') <= list(source).count('<constant>')
                        # the hint too must sit strictly below the engine's own
                        # result, or the minted rule is dead on arrival
                        and self._core.ac_ordered_below(list(hint_tokens), ac_out)
                        and self._confirm_mined_rules(
                            [(source, tuple(hint_tokens))], dummy_variables, X_data,
                            constants_fit_challenges, constants_fit_retries, rtol, atol,
                            min_informative, mine_seed)):
                    target = hint_tokens
                    certificate = 'verified'
            if target is None:
                if engine_reduces:
                    counts['already_covered'] += 1
                    record(source, None, 'already_covered', 'covered')
                else:
                    counts['rejected'] += 1
                    record(source, None, 'rejected', 'search')
                continue
            if X_confirm is not None and not self._confirm_mined_rules(
                    [(source, tuple(target))], dummy_variables, X_confirm,
                    constants_fit_challenges, constants_fit_retries, rtol, atol,
                    int(confirm_min if confirm_min is not None else max(1, X_confirm.shape[0] // 8)),
                    confirm_seed):
                counts['rejected'] += 1
                record(source, target, 'rejected', 'confirm', certificate)
                continue
            certified_trail_idx.append(record(source, target, 'certified', 'accepted', certificate))
            certified_pairs.append((tuple(source), tuple(target)))
            if verbose:
                print(f'Proposal certified ({certificate}): {list(source)} -> {list(target)}')

        # Per-proposal accounting that mirrors the deduplicate_rules fold EXACTLY (keep
        # first canonical source unless a strictly shorter target arrives): a certified
        # pair that would not change the merged set is a 'duplicate', not a 'certified'.
        before = [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.simplification_rules]
        seen: dict[tuple[str, ...], int] = {}
        for lhs, rhs in before:
            canon_source, mapping = remap_expression(list(lhs), dummy_variables, variable_prefix='?')
            canon_target, _ = remap_expression(list(rhs), dummy_variables, mapping, variable_prefix='?')
            key = tuple(canon_source)
            if key not in seen or len(canon_target) < seen[key]:
                seen[key] = len(canon_target)
        for idx, (source, target) in zip(certified_trail_idx, certified_pairs):
            canon_source, mapping = remap_expression(list(source), dummy_variables, variable_prefix='?')
            canon_target, _ = remap_expression(list(target), dummy_variables, mapping, variable_prefix='?')
            key = tuple(canon_source)
            if key in seen and len(canon_target) >= seen[key]:
                counts['duplicate'] += 1
                trail[idx].update(verdict='duplicate', stage='merge')
            else:
                counts['certified'] += 1
                seen[key] = len(canon_target)
        if certified_pairs:
            # The SAME merge path as mined rules: shortest target per canonical source.
            self.simplification_rules = deduplicate_rules(
                before + certified_pairs, dummy_variables, verbose=verbose)
            self.compile_rules()
            self._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in self.simplification_rules])
        if verbose:
            print(f"Proposals: {len(proposal_entries)} processed -> "
                  f"{counts['certified']} certified, {counts['already_covered']} already covered, "
                  f"{counts['rejected']} rejected, {counts['duplicate']} duplicate")
        return counts, trail

    def find_rules(
            self,
            max_source_pattern_length: int = 7,
            max_target_pattern_length: int | None = None,
            dummy_variables: int | list[str] | None = None,
            extra_internal_terms: list[str] | None = None,
            X: np.ndarray | int | None = None,
            constants_fit_challenges: int = 16,
            constants_fit_retries: int = 16,
            output_file: str | None = None,
            save_every: int = 100,
            reset_rules: bool = True,
            prune: bool | str = False,
            verbose: bool = False,
            rtol: float = 1e-9,
            atol: float = 1e-12,
            min_informative: int | None = None,
            seed: int | None = 42,
            confirm: bool = True,
            source_sample_per_length: dict[int, int] | None = None,
            candidate_fold_filter: bool = True,
            relaxed_kruskal: bool = True,
            proposals: str | list | dict | None = None,
            promote_sorts: bool = True,
            symbolic_gate: bool = True,
            snapshot_at: dict[int, str] | None = None) -> None:
        """Systematically discovers new simplification rules.

        This powerful method automates the discovery of simplification rules.
        It operates in two phases:
        1.  **Generation**: It combinatorially generates all possible valid
            expressions up to `max_source_pattern_length`.
        2.  **Verification**: It tests each generated expression for equivalence
            with any shorter expression, natively on the compiled Rust core
            (rayon-parallel across all cores; see Notes). Equivalences are found
            by evaluating both expressions on random numerical data.

        Discovered rules are deduplicated, compiled into the running engine, and
        can optionally be saved to disk.

        Parameters
        ----------
        max_source_pattern_length : int, optional
            The maximum length of expressions to generate and test.
        max_target_pattern_length : int or None, optional
            The maximum length of a valid simplified expression. If None, any
            shorter expression is considered a valid simplification.
        dummy_variables : int or list[str] or None, optional
            The variables to use when generating expressions.
        extra_internal_terms : list[str] or None, optional
            Additional leaf nodes (e.g., '<constant>') to include.
        X : np.ndarray or int or None, optional
            The numerical data for testing equivalence. If an int, specifies
            the number of samples to generate. If None, defaults to 1024 samples.
        constants_fit_challenges : int, optional
            Number of random constant sets to test for equivalence.
        constants_fit_retries : int, optional
            Number of retries for the curve fitting process.
        output_file : str or None, optional
            If provided, saves the discovered rules to this JSON file.
        save_every : int, optional
            How often to save the rules to the output file.
        reset_rules : bool, optional
            If True, clears existing rules before starting.
        prune : bool or str, optional
            If ``'covered'``, runs :meth:`prune_covered_rules` after the
            length loop, removing any rule the remaining rules cover
            behaviorally (compositional, can be expensive for large rule
            sets). ``True`` (the retired redundant-rule prune, which died
            with the legacy kernel) raises immediately. Defaults to False.
        verbose : bool, optional
            If True, shows progress bars and status updates.
        rtol : float, optional
            Relative tolerance of the numerical equivalence check. The default (1e-9)
            is deliberately strict: looser tolerances (e.g. 1e-5) accept saturation
            plateaus (tanh/exp towers within tolerance of 1 or 0 over the whole
            sample) as identities.
        atol : float, optional
            Absolute tolerance of the numerical equivalence check (default 1e-12).
        min_informative : int or None, optional
            Minimum number of SOURCE-FINITE evidence rows (accumulated across challenge
            instances) required to certify a rule. Defaults to ``X.shape[0] // 8``. This
            is the vacuous-acceptance gate: an almost-everywhere-NaN source (e.g.
            ``asin(cosh(_0))``) has no evidence and cannot be rewritten by its corner
            values alone.

            Equivalence is GENERIC (with domain extension): rows where the source is
            finite bind (the replacement must be finite and equal within tolerance);
            rows where the source is NaN/inf are extendable (``div(_0, _0) -> 1``
            certifies -- the 0/0 limit -- and ``log(exp(_0)) -> _0`` certifies under
            f64 overflow). The reverse stays rejected: a replacement that is NaN where
            the source is finite loses a defined value.
        seed : int or None, optional
            Seed for the evaluation matrix, constant challenges and per-source RNG
            streams. The default (42) makes the mine REPRODUCIBLE run-to-run; pass
            None for entropy-based seeding.
        confirm : bool, optional
            If True (default), every mined rule is re-verified on an independent,
            twice-as-wide X with fresh constant draws before it enters the rule set
            (stage-2 confirmation; kills data-luck accepts).
        source_sample_per_length : dict[int, int] or None, optional
            The UNIVERSE POLICY for lengths whose complete expression universe is too
            large to enumerate. A length mapped here is represented by that many
            expressions drawn UNIFORMLY from its complete universe (seeded top-down
            count-weighted sampler; see :func:`simplipy.utils.sample_expression`)
            instead of exhaustive enumeration. All other lengths are enumerated
            COMPLETELY (enumeration must saturate a length, not merely reach it).
            Coverage is always logged; sampling a length inside the candidate/
            replacement range additionally warns, because the candidate library then
            no longer certifies "no shorter equivalent exists". The dev operator set
            crosses enumeration feasibility between lengths 5 (6.8e6) and 6 (2.4e8).
        candidate_fold_filter : bool, optional
            Drop VARIABLE-FREE candidates of length >= 2 from the candidate library
            (default True). Sound: a var-free candidate evaluates to one scalar per
            constant-assignment, so any source it matches is constant-valued and the
            length-1 ``<constant>`` candidate (whose presence gates the filter)
            already matches at a strictly shorter length, preempting it in the
            shortest-first scan. This removes the bulk of the constant-bearing
            (LM-fit) candidate arm -- the dominant per-source cost for const-free
            sources -- without changing any mined rule. Set False only for a
            reference mine (e.g. the filtered-vs-unfiltered parity gate).
        relaxed_kruskal : bool, optional
            If True (the default), EVERY source is searched with the engine's own
            result as the mark to beat: only targets strictly below that mark in
            the serve-time reduction ordering are selectable (the acceptance rides
            the candidate scan itself). False restores strict Kruskal pruning:
            a source the engine already reduces is skipped entirely -- including
            sources canonicalization merely RESPELLS shorter, whose respelled form
            may never re-enter the ladder (a documented recall loss). The relaxed
            mine finds one-step shortcut rules the strict mine provably cannot
            (degenerate constant collapses like ``C * acos(np.e) -> <constant>``
            and diagonal-collection shortcuts like ``exp(t)*exp(t) -> exp(t+t)``).
        proposals : str or list or dict or None, optional
            The PROPOSAL CHANNEL (LLM- or human-proposed identities): a path to a
            proposals JSON file, or the equivalent in-memory object. Two schemas are
            accepted -- the consolidated artifact format (a dict with key
            ``"proposals"``) and a bare list, both holding ``{source, target?}``
            objects with prefix token lists (``target`` is used as the certification
            HINT; extra keys are ignored). After the mining length loop completes and
            BEFORE the optional prune, every proposal runs the exact certification
            chain of :meth:`certify_rules` against the just-mined rule state, with
            this mine's evaluation matrices, challenges, tolerances and
            master-seed-derived seeds: every proposal is SEARCHED with the engine's
            own result as the ordering mark to beat (``already_covered`` is a
            search-verified verdict, never a spelling-shrink guess), and a certified
            proposal joins the ruleset through the same
            :func:`~simplipy.utils.deduplicate_rules` path. Deterministic: file
            order, content-derived per-proposal seeds (the stage-2-confirm policy).
            The provenance sidecar records the proposals file, its sha256, and the
            per-outcome counts (``certified`` / ``already_covered`` / ``rejected``
            / ``duplicate``). Config key ``proposals:`` in the find-rules YAML.
        promote_sorts : bool, optional
            Run the sort-promotion ladder after the prune, re-certifying every rule at the
            stronger ``_``/``!``/``$`` sorts and shipping it at the strongest sound one.
            ON by default (owner ruling 2026-08-01: features do not ship disabled) -- a
            default mine emits the deployment-strength ruleset. Pass ``False`` for a raw,
            entirely ``?``-sorted mine (promotion is fail-safe, so ``False`` costs only
            composite-subtree recall, never soundness).
        snapshot_at : dict[int, str] or None, optional
            LADDER RE-USE: ``{source_length: output_path}``. The cells of one ``j`` form a
            PREFIX CHAIN -- mining ``(7, 4)`` does all the work of ``(5, 4)`` and ``(6, 4)``
            on its way up, because lengths are mined shortest-first, the per-source seed is
            ``mine_seed + (length << 40)`` (indexed by length alone, not by how far the climb
            goes), the master seeds are drawn BEFORE the universe is built, and enumeration
            consumes no randomness. So a climb can emit each shorter cell as it passes
            through it: at the end of a listed length the full post-pass (proposals, prune,
            promotion) runs on a COPY of the mine and writes that cell's artifact plus a
            sidecar recording its ladder origin, then the raw un-pruned state is restored and
            the climb continues. This makes one ``(7, 4)`` run produce ``(5, 4)``, ``(6, 4)``
            and ``(7, 4)`` for the price of the tallest -- worth roughly two exhaustions of
            lengths <= 5 (days of wall-clock). A snapshot at or above
            ``max_source_pattern_length`` is refused: that cell is the run's own output.

        Notes
        -----
        The mine runs NATIVELY on the compiled Rust core (``simplipy._core``),
        parallelized over all cores via rayon; cap it with the ``RAYON_NUM_THREADS``
        environment variable. The engine must therefore be constructed via
        :meth:`from_config` or :meth:`load`. The pure-Python mining mirror was removed
        in 0.5.0: it duplicated the Rust checker (``rust/worker.rs``/``fit.rs``) and
        repeatedly desynced from it.
        """
        if not isinstance(prune, bool) and prune != 'covered':
            raise ValueError(f"prune must be a bool or 'covered', got {prune!r}")
        if prune is True:
            # Fail HERE, not after the length loop: the old late raise fired at the END
            # of a potentially week-scale mine (audit Tier-2, 2026-08-03).
            raise ValueError(
                "prune=True (the redundant-rule prune) died with the legacy kernel; "
                "use prune='covered' (AC-judged behavioral coverage) or prune=False")

        # Proposal channel: load + normalize FIRST (a malformed proposals file must fail
        # here, before any mining compute is spent). None stays None: only an explicitly
        # given batch (even an empty one) activates the pass in _find_rules_native.
        proposal_entries: list[tuple[tuple[str, ...], tuple[str, ...] | None]] | None = None
        proposal_record: dict | None = None
        if proposals is not None:
            proposal_entries, proposal_record = _load_proposals(proposals)
            if verbose:
                print(f'Loaded {len(proposal_entries):,} proposals'
                      + (f' from {proposal_record["file"]}' if proposal_record['file'] else ''))

        # Signal handler for main process
        interrupted = False

        def signal_handler(signum: Any, frame: Any) -> None:
            nonlocal interrupted
            interrupted = True
            print("\nInterrupt received, cleaning up...")

        # All the initialization from the sequential version
        extra_internal_terms = extra_internal_terms or []

        def _default_dummy_variables() -> list[str]:
            max_leaf_nodes_if_operators_binary = int(max_source_pattern_length - (max_source_pattern_length - 1) / 2)
            dummies = [f"x{i}" for i in range(max_leaf_nodes_if_operators_binary)]
            if verbose:
                print(f"Using {len(dummies)} dummy variables: {dummies}")
            return dummies

        dummy_variables = self._resolve_dummy_variables(dummy_variables, default=_default_dummy_variables)

        # Ladder snapshots: fail closed on a length this climb never completes, rather than
        # silently writing nothing (a missing artifact would be discovered days later).
        snapshot_at = {int(k): str(v) for k, v in (snapshot_at or {}).items()}
        for _length in sorted(snapshot_at):
            if not 1 <= _length < max_source_pattern_length:
                raise ValueError(
                    f'snapshot_at length {_length} must satisfy 1 <= length < '
                    f'max_source_pattern_length ({max_source_pattern_length}); the top cell is '
                    f'written to output_file.')

        if reset_rules:
            self.simplification_rules = []
            self.compile_rules()

        # The mine's X is SEEDED (reproducibility) and defaults to a heavy-tailed MIXTURE
        # (N(0,5) + wide uniform + signed log-uniform magnitudes + special values, see
        # _mining_sample_x), so equivalence certification sees the value ranges that
        # rule APPLICATION will see (wildcards bind arbitrary subtree values).
        _rng = np.random.default_rng(seed)
        if X is None:
            X_data = self._mining_sample_x(1024, len(dummy_variables), _rng)
        elif isinstance(X, int):
            X_data = self._mining_sample_x(X, len(dummy_variables), _rng)
        else:
            X_data = np.asarray(X, dtype=np.float64)
        if min_informative is None:
            # Floor at 1 (mirrors the rust FFI default `(n_rows/8).max(1)`): a small X
            # (< 8 rows) must never yield 0, which would disable the evidence gate and
            # re-admit the vacuous all-NaN/inf accepts it exists to kill.
            min_informative = max(X_data.shape[0] // 8, 1)

        # Independent, wider confirm matrix + derived integer seeds (drawn from the same
        # master stream, so one `seed` reproduces the whole mine).
        X_confirm = self._mining_sample_x(2 * X_data.shape[0], len(dummy_variables), _rng) if confirm else None
        mine_seed = int(_rng.integers(2 ** 62))
        confirm_seed = int(_rng.integers(2 ** 62))

        leaf_nodes = dummy_variables + extra_internal_terms
        non_leaf_nodes = dict(sorted(self.operator_arity.items(), key=lambda x: x[1]))

        # --- Phase 1: build the source/candidate universe (see _build_source_universe) ---
        source_sample_per_length = dict(source_sample_per_length or {})
        expressions_of_length, counts = self._build_source_universe(
            leaf_nodes, non_leaf_nodes, max_source_pattern_length, max_target_pattern_length,
            source_sample_per_length, _rng, verbose)

        total_expressions = sum(len(v) for v in expressions_of_length.values())

        if verbose:
            print(f"Finished generating expressions up to size {max_source_pattern_length}. Total expressions: {total_expressions:,}")
            for length, expressions in sorted(expressions_of_length.items()):
                print(f"Size {length}: {len(expressions):,} expressions")

        # PROVENANCE (see _build_mine_provenance).
        provenance = self._build_mine_provenance(
            X=X, X_data=X_data, counts=counts,
            expressions_of_length=expressions_of_length,
            source_sample_per_length=source_sample_per_length,
            max_source_pattern_length=max_source_pattern_length,
            max_target_pattern_length=max_target_pattern_length,
            dummy_variables=dummy_variables,
            extra_internal_terms=extra_internal_terms,
            constants_fit_challenges=constants_fit_challenges,
            constants_fit_retries=constants_fit_retries,
            rtol=rtol, atol=atol, min_informative=min_informative,
            seed=seed, mine_seed=mine_seed, confirm_seed=confirm_seed,
            confirm=confirm, candidate_fold_filter=candidate_fold_filter,
            relaxed_kruskal=relaxed_kruskal, prune=prune, reset_rules=reset_rules)
        if proposal_record is not None:
            # The sidecar pins the proposal batch (file + sha256 + count); the per-outcome
            # counts are filled in by the proposal pass itself (_find_rules_native).
            provenance['proposals'] = proposal_record

        # --- Phase 2: mine natively on the Rust engine (the only mining path since 0.5.0) ---
        # SINGLE-FLIGHT: fail fast if another mine is active in this process (see
        # _MINE_LOCK). Acquired HERE, directly above the try whose finally releases it
        # -- the validation/prep above may raise (that is its job), and acquiring any
        # earlier would leak the lock on those paths (hardening H-002/H-009).
        if not _MINE_LOCK.acquire(blocking=False):
            raise RuntimeError(
                "another find_rules mine is active in this process; mines are "
                "single-flight (the interval soundness counters recorded in the "
                "provenance sidecar are process-global)")
        # Graceful-interrupt handler: MAIN THREAD ONLY -- `signal.signal` raises
        # ValueError anywhere else (hardening H-008: this used to make find_rules
        # unusable from worker threads, and installing it before the prep code could
        # leak it past an early raise, H-009). Off the main thread the mine runs
        # without interrupt cleanup; `interrupted` then simply never fires.
        handler_installed = False
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGINT, signal_handler)
            handler_installed = True
        try:
            self._find_rules_native(
                expressions_of_length,
                max_source_pattern_length,
                max_target_pattern_length,
                dummy_variables,
                X_data,
                constants_fit_challenges,
                constants_fit_retries,
                output_file,
                prune,
                verbose,
                lambda: interrupted,
                rtol=rtol,
                atol=atol,
                min_informative=min_informative,
                mine_seed=mine_seed,
                confirm_seed=confirm_seed,
                X_confirm=X_confirm,
                candidate_fold_filter=candidate_fold_filter,
                relaxed_kruskal=relaxed_kruskal,
                provenance=provenance,
                proposal_entries=proposal_entries,
                leaf_nodes=leaf_nodes,
                promote_sorts=promote_sorts,
                symbolic_gate=symbolic_gate,
                snapshot_at=snapshot_at,
            )
        finally:
            if handler_installed:
                signal.signal(signal.SIGINT, old_handler)
            _MINE_LOCK.release()

    def operators_to_realizations(self, prefix_expression: list[str] | tuple[str, ...]) -> list[str]:
        """Convert canonical operator names to their runtime realizations (e.g. ``'sin'`` -> ``'np.sin'``)."""
        return self._core.operators_to_realizations(list(prefix_expression))

    def realizations_to_operators(self, prefix_expression: list[str]) -> list[str]:
        """Convert realization tokens (e.g. ``'np.sin'``) back to canonical operator names."""
        return self._core.realizations_to_operators(list(prefix_expression))

    @staticmethod
    def code_to_lambda(code: CodeType) -> Callable[..., float]:
        """Converts a Python code object into an executable lambda function.

        Parameters
        ----------
        code : CodeType
            The compiled code object to convert.

        Returns
        -------
        Callable[..., float]
            An executable lambda function.
        """
        return FunctionType(code, globals())()
