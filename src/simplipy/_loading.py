"""The rule sets an engine serves and their life cycle: the core built from them, the modes built
eagerly or on first use, pickling, the mutation API, the realization modules, and the fingerprint of
the measure the sets were mined under."""

import builtins
import hashlib
import importlib
import json
import logging
import os
import threading
import warnings
from typing import Any, Callable

import numpy as np
import yaml

from simplipy._modes import (
    Mode, _MODE_RULES_ATTR, _RULE_MODE, _coerce_mode, _coerce_rule_mode)
from simplipy._schema import (
    _CORE_IMPORT_ERROR, _CORE_SERIALIZATION_ARITIES, _RustEngine, _coverage_variants,
    _reaches_promise)
from simplipy.trust import check_chain, check_root, package_for
from simplipy.utils import is_numeric_string

#: The engine's announcements, on its own logger (see :mod:`simplipy.engine`).
_log = logging.getLogger('simplipy.engine')


class _Loading:
    """The rule sets an engine serves and the core it serves them from."""

    # The engine's state, set by the constructor and rebuilt on unpickling (see
    # :class:`~simplipy.engine.SimpliPyEngine`).
    operator_tokens: list[str]
    operator_aliases: dict[str, str]
    operator_realizations: dict[str, Any]
    operator_arity: dict[str, Any]
    operator_arity_compat: dict[str, Any]
    operators: list[str]
    modules: list[str]
    simplification_rules: list[tuple[tuple[str, ...], tuple[str, ...]]]
    permissive_simplification_rules: list[tuple] | None
    _modes: str | tuple[str, ...]
    _rule_canons: dict[str, str]
    _realization_roots: dict[str, list[str]]
    _trusted_modules: frozenset[str]
    _operators_config: dict[str, dict[str, Any]]
    defined_functions: dict[str, dict[str, Any]]
    _core: Any
    _mode_rule_sources: dict[str, str]
    _mode_load_lock: threading.Lock
    _core_loaded_modes: set[str]

    @property
    def _effective_arity(self) -> dict[str, Any]:
        """THE EFFECTIVE OPERATOR SCHEMA, the one every helper that walks an expression reads:
        the arity of each operator of the core serialization language (``+``, ``-``, ``*``,
        ``/``, ``neg``, ``inv``, ``pow``, ``rootn``), which the core reads and writes whatever
        the config declares (an engine over ``+`` and ``*`` answers ``x0*x0`` with
        ``pow(x0, 2)``), and of each operator the config declares."""
        return {**_CORE_SERIALIZATION_ARITIES, **self.operator_arity}

    @property
    def _syntax_arity(self) -> dict[str, Any]:
        """The effective schema with the config's DEFINED FUNCTIONS beside it, each at its arity:
        what a walk over an expression AS WRITTEN reads (:meth:`expression_variables`,
        :meth:`evaluate_constants`). The canonical entries never meet a defined head: they
        read the expansion (:meth:`_expand_defined`)."""
        return {**self._effective_arity,
                **{name: spec['arity'] for name, spec in self.defined_functions.items()}}

    def _expand_defined(self, tokens: list[str]) -> list[str]:
        """``tokens`` with every DEFINED FUNCTION expanded into its definition (the core's
        ``expand_defined``, either dialect): what every entry into the canonical state and every
        evaluation reads, so a defined head answers, prices and evaluates as its definition. The
        same list when the engine defines nothing."""
        if not self.defined_functions:
            return tokens
        return self._core.expand_defined(list(tokens))

    @staticmethod
    def _normalized_modes(modes: 'str | Mode | tuple | list') -> 'str | tuple[str, ...]':
        """Validate + normalize the ``modes=`` selection: ``'all'``, or the tuple of
        PUBLIC mode names built eagerly (``'strict'`` always present -- the default set
        is the core's construction substrate and cannot be deferred).

        Loud on everything else: a bare mode-name string is refused with the tuple
        spelling suggested (a ``str`` is also a sequence of characters, and iterating
        it would silently select nothing), an unknown name or a non-mode element
        raises exactly as ``simplify(mode=...)`` would.
        """
        if isinstance(modes, str):
            if modes.strip().lower() == 'all':
                return 'all'
            raise TypeError(
                f"modes must be 'all' or a tuple/list of mode names or simplipy.Mode "
                f"members; a bare mode name is refused because a string is also a "
                f"sequence of characters -- did you mean modes=({modes!r},)?")
        if isinstance(modes, Mode):
            modes = (modes,)
        if not isinstance(modes, (tuple, list, set, frozenset)):
            raise TypeError(
                f"modes must be 'all', a simplipy.Mode, or a tuple/list of mode names "
                f"or Mode members, not {type(modes).__name__} ({modes!r})")
        names = {'strict'}
        for m in modes:
            if isinstance(m, str):
                m = _coerce_mode(m)
            elif not isinstance(m, Mode):
                raise TypeError(
                    f"modes entries must be mode names or simplipy.Mode members, "
                    f"not {type(m).__name__} ({m!r})")
            names.add(m.name)
        return tuple(n for n in ('strict', 'permissive') if n in names)

    @staticmethod
    def _normalized_rule_canons(canons: dict[str, str] | None) -> dict[str, str]:
        """``rule_canons`` as the core spells it: rule mode -> canon (``'legacy'`` or a rule
        mode). A mode not named, or named with its own canon, is left out: it reads its set
        in its own canon."""
        out: dict[str, str] = {}
        for mode, canon in (canons or {}).items():
            rule_mode = _RULE_MODE[_coerce_mode(mode)]
            read_in = canon if canon == 'legacy' else _RULE_MODE[_coerce_mode(canon)]
            if read_in != rule_mode:
                out[rule_mode] = read_in
        return out

    @staticmethod
    def _normalized_defined_functions(
            defined: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
        """The ``defined_functions`` section as the engine keeps it: name -> ``{'arity': int,
        'definition': str}``, a copy. Its SHAPE is checked here, loudly; what a definition may
        say (the engine's operators, numbers and its arguments ``_0``, ``_1``, ...) is checked by
        the core when it is built."""
        if defined is None:
            return {}
        if not isinstance(defined, dict):
            raise ValueError(f"defined_functions must be a mapping of name to "
                             f"{{arity, definition}}, not {type(defined).__name__}")
        out: dict[str, dict[str, Any]] = {}
        for name, spec in defined.items():
            if not isinstance(name, str) or not isinstance(spec, dict) \
                    or set(spec) != {'arity', 'definition'} \
                    or isinstance(spec['arity'], bool) or not isinstance(spec['arity'], int) \
                    or not isinstance(spec['definition'], str):
                raise ValueError(
                    f"defined_functions: {name!r} must give exactly an integer 'arity' and a "
                    f"'definition' string, as in sqrt: {{arity: 1, definition: 'rootn(_0, 2)'}}; "
                    f"got {spec!r}")
            out[name] = {'arity': spec['arity'], 'definition': spec['definition']}
        return out

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
                    rules_permissive: list[tuple] | None = None, *,
                    canons: dict[str, str] | None = None,
                    defined_functions: dict[str, dict[str, Any]] | None = None) -> Any:
        """Build the compiled core (``simplipy._core``) from in-memory config + rules.

        The core is REQUIRED: a missing extension or a load failure is a hard error --
        the pure-Python engine was removed.

        ``rules_permissive`` is that mode's OWN COMPLETE set, pushed after construction
        through the core's ``set_mode_rules``. ``None`` means the push does not happen at
        all, so a config naming no permissive set builds byte-for-byte the default-only
        core -- the no-op property holds because there is nothing extra to execute, not
        because something extra is a no-op. The push is a separate statement rather than
        another argument to ``from_strs`` because the mine driver needs exactly this
        surface too (it installs a freshly minted set on a live core), and one entry point
        is easier to keep honest than two.

        ``canons`` maps a rule mode to the canon its set was certified in
        (:meth:`_normalized_rule_canons`); a mode not named reads its set in its own.

        ``defined_functions`` is the config's ``defined_functions:`` section
        (:meth:`_normalized_defined_functions`), which the core reads and checks beside the
        operators; none, or an empty section, builds the core from the operators alone, as
        before.
        """
        canons = canons or {}
        if _RustEngine is None:
            raise ImportError(
                'simplipy._core is required (the pure-Python engine was removed): the compiled '
                f'extension failed to import ({_CORE_IMPORT_ERROR!r})')
        config = {'operators': operators}
        if defined_functions:
            config['defined_functions'] = defined_functions
        config_text = yaml.safe_dump(config, sort_keys=False)
        rules_text = json.dumps([[list(lhs), list(rhs)] for lhs, rhs in rules])
        core = _RustEngine.from_strs(config_text, rules_text, canons.get('default'))
        if rules_permissive is not None:
            core.set_mode_rules(
                'permissive', [(list(lhs), list(rhs)) for lhs, rhs in rules_permissive],
                canon=canons.get('permissive'))
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

        The recipe is REFUSED when it is ambiguous. The core echoes a digest of
        the rules it was actually given (:meth:`compile_rules` or construction), and a
        mismatch with ``simplification_rules`` here means the two have diverged -- the
        list was mutated without :meth:`compile_rules`, or the core's rules were set
        directly. A pickle rebuilds the worker from the LIST, so a diverged parent
        would silently spawn workers running different rules. Sync first
        (:meth:`compile_rules`), then pickle.
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
        # The lazy-mode runtime state: the lock is process-local (and has no pickle
        # surface), and `_core_loaded_modes` describes the CORE, which the unpickling
        # side rebuilds and re-derives. The RECIPE halves (`_modes`,
        # `_mode_rule_sources`) stay in the state: a worker unpickling a lean engine
        # inherits the same lean profile and lazily loads a deferred set from the same
        # file on first use -- note this makes such a pickle portable exactly as far
        # as the recorded artifact paths resolve (the spawn-worker case it serves).
        state.pop('_mode_load_lock', None)
        state.pop('_core_loaded_modes', None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the engine from pickled state, exactly as ``__init__`` would.

        Mirrors the construction order: realization modules first (a spawn
        worker unpickles into a fresh interpreter), then the compiled core from
        the same in-memory config + rules. The trusted set travels IN the state, so
        a worker imports exactly what the parent was allowed to import -- widening
        trust needs the caller or the environment, in the parent, as everywhere else.

        A state written by simplipy 0.14 or earlier is REFUSED (``IncompatibleArtifactError``):
        its rules were mined under another engine generation's canonical forms, the gate
        :meth:`from_config` applies to the same rule files ("serves only rule files mined by
        simplipy 0.15 or later"). Such a state carries the retired real mode's
        ``real_simplification_rules`` or no ``_rule_canons``, which every 0.15 engine records.
        """
        if 'real_simplification_rules' in state or '_rule_canons' not in state:
            from .compat import IncompatibleArtifactError
            raise IncompatibleArtifactError(
                'this pickled SimpliPyEngine was written by simplipy 0.14 or earlier: its rules '
                'were mined under another engine generation, and simplipy 0.15 serves only rule '
                'files mined by 0.15 or later (from_config refuses the same files). Rebuild the '
                'engine with SimpliPyEngine.load or from_config on a 0.15 asset, and pickle that')
        self.__dict__.update(state)
        self.import_modules()
        # The permissive set travels IN the state (a plain attribute), so the recipe a
        # worker unpickles is complete. `setdefault(..., None)` keeps pickles written
        # before per-mode sets existed loadable: they rebuild with no set of their own,
        # which is exactly what they were. The same treatment keeps pickles written
        # before the lazy-mode machinery loadable: they rebuild as
        # `modes='all'` engines with no file sources, which is exactly what they were.
        self.__dict__.setdefault('_modes', 'all')
        self.__dict__.setdefault('_mode_rule_sources', {})
        self._mode_load_lock = threading.Lock()
        # Pickles written before defined functions existed define none.
        self.__dict__.setdefault('defined_functions', {})
        self._core = self._build_core(
            self._operators_config, self.simplification_rules,
            self.__dict__.setdefault('permissive_simplification_rules', None),
            canons=self._rule_canons, defined_functions=self.defined_functions)
        # The fresh core carries exactly the non-None in-memory sets `_build_core`
        # just pushed; a set that was deferred (or unloaded) in the parent stays
        # deferred here and lazily loads from its recorded source on first use.
        self._core_loaded_modes = {
            m for m in _MODE_RULES_ATTR
            if getattr(self, _MODE_RULES_ATTR[m]) is not None}

    def compile_rules(self) -> None:
        """Sync the compiled core's rule set from ``self.simplification_rules``.

        Public contract (unchanged since the pure-Python engine): after mutating
        ``self.simplification_rules``, call this so ``simplify`` sees the new rules.

        COPY-ON-WRITE: a fresh core is built from the in-memory state and the reference
        is swapped atomically, so a concurrent reader is never mutated mid-``simplify``.
        A reader holding the old core finishes on a consistent (old) ruleset; the
        rebuild is mutation-API cost only, never on the hot path.

        The permissive set is re-pushed alongside: a fresh core starts with no set of its
        own for it, so NOT re-pushing it would silently retract it on every default-rule
        sync. Re-pushed means the IN-MEMORY set: a set deferred by a
        lean ``modes=`` selection (or dropped by :meth:`unload_mode`) stays deferred and
        lazily loads from its recorded source on first use, exactly as before the sync.
        The swap runs under the mode-load lock so it cannot race a first-use build.
        """
        with self._mode_load_lock:
            self._core = self._build_core(
                self._operators_config, self.simplification_rules,
                self.permissive_simplification_rules, canons=self._rule_canons,
                defined_functions=self.defined_functions)
            self._core_loaded_modes = {
                m for m in _MODE_RULES_ATTR
                if getattr(self, _MODE_RULES_ATTR[m]) is not None}

    def _push_mode_rules(self, rule_mode: str, rules: list[tuple] | None) -> None:
        """Push (or with ``None`` retract) one mode's set into the LIVE core.

        Safe while other threads are inside the core: the core's push waits for the
        calls in flight to return, and translates the set before it returns. Callers
        hold ``_mode_load_lock``, so two pushes cannot race each other.
        """
        self._core.set_mode_rules(
            rule_mode, None if rules is None else [(list(lhs), list(rhs)) for lhs, rhs in rules],
            canon=self._rule_canons.get(rule_mode))

    def _ensure_mode_loaded(self, rule_mode: str) -> None:
        """Build ``rule_mode``'s set NOW if this engine names one that the core does
        not currently hold -- the first-use half of the lazy ``modes=`` machinery.

        Called on every :meth:`simplify`/:meth:`complexity` call that addresses a
        non-default mode; the fast path is one set-membership test. The slow path
        runs ONCE per set (lock-guarded, so two threads' first calls cannot
        double-build), is ADDITIVE-ONLY (it installs into the live core and evicts
        nothing), and announces itself with one WARNING record on the ``simplipy``
        logger, like :meth:`load`'s default-engine announcement, because the build pauses the
        call for a few seconds on a large artifact.
        """
        if rule_mode not in _MODE_RULES_ATTR:
            return
        if rule_mode in self._core_loaded_modes:
            return
        with self._mode_load_lock:
            if rule_mode in self._core_loaded_modes:
                return
            attr = _MODE_RULES_ATTR[rule_mode]
            rules = getattr(self, attr)
            if rules is None:
                source = self._mode_rule_sources.get(rule_mode)
                if source is None:
                    # This engine names no set for the mode: nothing to build, and
                    # (sources only ever attach at load) nothing will ever arrive.
                    # Marking it loaded makes every later call take the fast path.
                    self._core_loaded_modes.add(rule_mode)
                    return
                _log.warning("simplipy: building the %r rule set on first use from '%s' "
                             "(deferred by modes=%r); expect a pause while it loads and "
                             "compiles", rule_mode, source, self._modes)
                with open(source, 'r') as f:
                    rules = self._normalized_mode_rules(json.load(f))
                setattr(self, attr, rules)
            else:
                _log.warning("simplipy: building the %r rule set on first use from its "
                             "in-memory recipe (deferred by modes=%r); expect a pause while "
                             "it compiles", rule_mode, self._modes)
            self._push_mode_rules(rule_mode, rules)
            self._core_loaded_modes.add(rule_mode)

    def _serve(self, mode: Mode, call: Callable[[Any], Any]) -> Any:
        """``call(core)`` for one public call in ``mode``, on the live core captured once,
        after a set deferred by a lean ``modes=`` selection has been built. A mode naming no
        set of its own is served the default set by the core."""
        if mode is not Mode.strict:
            self._ensure_mode_loaded(_RULE_MODE[mode])
        return call(self._core)

    def unload_mode(self, mode: 'Mode | str') -> None:
        """Drop ``mode``'s built rule structures; the next use lazily rebuilds them.

        The OPERATIONAL RAM KNOB paired with lean loading (``modes=`` at
        construction): a long-lived process that needed a mode only transiently --
        say a worker that canonicalised one corpus in ``permissive`` and then serves
        ``strict`` -- calls this to hand the memory back. Dropping is EXPLICIT and only
        ever happens here: no lazy build, sync, or other call evicts a set
        implicitly. The engine keeps the set's recipe (its config-named file, or the
        in-memory list it was constructed with), so the mode's next use rebuilds the
        set exactly as a lazy first use would -- announced by the same log line, with
        the same pause, serving byte-identical results.

        What is freed depends on the recipe. A CONFIG-NAMED set (built by
        ``from_config``/``load``) drops both the compiled core structures and the
        wrapper's parsed rule list -- the mode's full footprint, about 100 MB per set
        on the acj-5-4-llm artifact (how much of it the allocator returns to the OS
        immediately depends on heap layout). A set handed IN MEMORY to ``__init__``
        drops the compiled structures only: the wrapper's list is the sole recipe,
        so it stays.

        The DEFAULT (strict) set is REFUSED, loudly: it is the core's construction
        substrate -- the canon every mode's rewrite descends runs on it, the dedup
        key is a function of it, and every mode naming no set of its own serves it --
        so an engine without it cannot exist, and half-supporting the call would turn
        the refusal into a silent lie. Unloading a mode that names no set of its own
        is a no-op, and the call is idempotent.

        Calling it while other threads simplify is safe. The drop waits for in-flight
        core calls and cannot corrupt one. A ``permissive`` call that races it may run
        against the core's absence semantics for the mode, falling back to the default set
        for that one call. Call it at a quiet moment for the mode being unloaded, since
        every racing call that rebuilds the set undoes the drop.

        Parameters
        ----------
        mode : Mode or str
            The mode to unload, as a :class:`Mode` member or its name (``'permissive'``).
            ``Mode.strict`` (``'strict'``) raises ``ValueError``.
        """
        mode = _coerce_mode(mode)
        if mode is Mode.strict:
            raise ValueError(
                "the default (strict) rule set cannot be unloaded: it is the core's "
                "construction substrate (the canon every mode's rewrite descends "
                "runs on it, the dedup key is a function of it, and every mode "
                "naming no set of its own serves it), so an engine without it "
                "cannot exist. Unload 'permissive' instead.")
        rule_mode = _RULE_MODE[mode]
        with self._mode_load_lock:
            # Retract from the live core: drops the compiled (interned) set and its
            # translated index. `None` is the core's documented retraction spelling.
            self._push_mode_rules(rule_mode, None)
            if rule_mode in self._mode_rule_sources:
                # File-backed: the parsed list is re-readable, so it is dropped too --
                # this is where the bulk of the RAM comes back.
                setattr(self, _MODE_RULES_ATTR[rule_mode], None)
            self._core_loaded_modes.discard(rule_mode)

    def _replace_rules(self, rules: list) -> None:
        """Build-first-or-unchanged: compile a fresh core from the CANDIDATE
        ruleset, and only on success assign both the list and the core. The shim
        mutators used to rewrite ``simplification_rules`` before pushing, so a failed
        push left the wrapper and the core silently diverged; this turns that into
        loud-and-unchanged."""
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
        with self._mode_load_lock:
            new_core = self._build_core(
                self._operators_config, rules, self.permissive_simplification_rules,
                canons=self._rule_canons, defined_functions=self.defined_functions)
            self.simplification_rules = rules
            self._core = new_core
            self._core_loaded_modes = {
                m for m in _MODE_RULES_ATTR
                if getattr(self, _MODE_RULES_ATTR[m]) is not None}

    def prune_covered_rules(self, verbose: bool = False, *, rule_mode: 'Mode | str') -> int:
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
        up to 3 wide slots (``_``/``!``/``$`` sigils) probed with the composite
        ``x{i} + x{k}`` (``_coverage_variants``) -- leaf-only instantiation under-tests
        wide-sort claims.
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
        rule_mode : str or Mode
            The mode whose own set is pruned: ``'strict'``,
            ``'permissive'``, or a :class:`Mode` member. Coverage is asked of that mode's
            engine: its serve answer (the miner's mark) and its serve ordering, with the kept
            rules installed as its set.

        Returns
        -------
        int
            The number of rules that were pruned.
        """
        from .progress import Progress
        prog = Progress(verbose, prefix='  [prune] ')

        rule_mode = _coerce_rule_mode(rule_mode)
        attr = 'simplification_rules' if rule_mode == 'default' else _MODE_RULES_ATTR[rule_mode]
        full = [(tuple(lhs), tuple(rhs)) for lhs, rhs in (getattr(self, attr) or [])]
        if not full:
            return 0

        arity = self._effective_arity

        def covered(core: Any, lhs: tuple[str, ...], rhs: tuple[str, ...]) -> bool:
            # State-coverage in the serve-time reduction ordering (`_reaches_promise`). The
            # complexity-score judge this replaces was blind on the tie tier (equal-score
            # DIFFERENT states: `abs neg ?0 -> abs ?0` read 2 <= 2 "covered" while nothing
            # performed the rewrite), deleting rules mint acceptance had certified as strict
            # improvements. (mine_mark/ac_ordered_below_in: the AC parser is the arbiter, no
            # config-vocabulary gate; an undecidable ordering raises.)
            return all(_reaches_promise(core, rule_mode, variant_lhs, variant_rhs, arity)
                       for variant_lhs, variant_rhs in _coverage_variants(lhs, rhs, rule_mode))

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
                # Both of these run for hours on a large wave and neither reports anything
                # on its own: the rebuild is one call over the whole kept set, and the scan
                # is one AC judge per pending rule per coverage variant.
                prog.stage(f'wave {wave_length} round {rounds}: rebuilding probe engine',
                           kept=len(kept), pending=len(pending))
                probe_set = [rule for rule in full if rule in kept]
                if rule_mode == 'default':
                    core = self._build_core(self._operators_config, probe_set,
                                            self.permissive_simplification_rules,
                                            canons=self._rule_canons,
                                            defined_functions=self.defined_functions)
                else:
                    core = self._build_core(self._operators_config, self.simplification_rules,
                                            probe_set, canons=self._rule_canons,
                                            defined_functions=self.defined_functions)
                readd = [rule for rule in
                         prog.track(list(pending), f'  wave {wave_length} round {rounds}: coverage scan')
                         if not covered(core, *rule)]
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

        remaining = [rule for rule in full if rule in kept]
        if rule_mode == 'default':
            self._replace_rules(remaining)
        else:
            setattr(self, attr, remaining)
            self._replace_rules(self.simplification_rules)
        n_pruned = len(full) - len(remaining)
        if verbose:
            print(f'Pruned {n_pruned} covered rules ({len(remaining)} rules remaining)')
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
        candidate = list(self.simplification_rules)  # Mutate a copy, swap on success
        arity = self._effective_arity
        for i, rule in enumerate(candidate):
            lhs, rhs = rule
            if tuple(rhs) != ('<constant>',):
                continue
            leaves = [t for t in lhs if t not in arity]
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
        to come first), and the rest of its dotted path is then walked against the
        same set, because attribute traversal from a trusted root reaches whatever
        that root imported. The module is bound into this engine's own namespace --
        not into shared module globals, so engines cannot contaminate each other's
        expression namespaces. ``np`` is always bound, because ``np.pi`` and ``np.e``
        are token grammar and may appear in any expression regardless of config.

        Raises
        ------
        simplipy.trust.UntrustedModuleError
            If a realization names a module root that is neither trusted by default
            nor opted into via ``trusted_modules=`` or ``SIMPLIPY_TRUSTED_MODULES``,
            or if the rest of its dotted path reaches a module outside that set.
            See :mod:`simplipy.trust` for why the opt-in cannot live in the config.
        """
        # `__builtins__` is bound explicitly. Inline arithmetic (`x1 + x2`) runs in the expression's own frame, so a
        # numpy floating-point warning (`inf - inf`, an overflow in `*`) is raised there, and Python's warning
        # machinery reads that frame's globals for `__builtins__`: without it the evaluation raised
        # KeyError('__builtins__') instead of returning nan with a RuntimeWarning. Bare names resolved through the
        # interpreter's builtins before and still do, so what an expression can reach is unchanged.
        namespace: dict[str, Any] = {'__builtins__': builtins, 'np': np}
        for root, operators in sorted(self._realization_roots.items()):
            check_root(root, operators, self._trusted_modules)
            module = importlib.import_module(package_for(root))
            # THE ROOT IS ONLY THE FIRST HOP. The realization is resolved by
            # attribute traversal, and a module's attributes include every module it
            # imported, so a trusted root is a doorway to `os`, `shutil`, `importlib`.
            # `check_chain` walks the rest against the same allowlist. It runs AFTER the
            # import because the walk needs the resolved objects -- which costs nothing,
            # since the only module imported by then is the one `check_root` licensed.
            for operator in operators:
                check_chain(str(self.operator_realizations[operator]), str(operator),
                            module, self._trusted_modules)
            namespace[root] = module
        self._eval_namespace = namespace

    _MEASURE_PROBES = (
        # the LITERAL codebook: an integer (the L-formula), a unit fraction and a
        # non-unit one (the fraction code and the inversion bit), a decimal whose
        # denominator carries a five (the print/argmin split), a beyond-i128 leaf
        # (the numeric-string pricer -- its clamp once stayed
        # at two bits after the floor moved and NO probe noticed), and `<constant>`.
        ('1000',), ('1/2',), ('355/113',), ('0.2',), ('1e-40',), ('<constant>',),
        # a LONG LITERAL, which the strict reading writes as float64's double of it (cheaper than
        # its 64 digits): the probe moves with the reading of numbers
        ('0.0766541268471677307861497420516056347394916180597539647375669841',),
        # a value strict's number domain holds in pieces (here a literal whose denominator passes
        # 2^1022 times the exact reciprocal of a number), each piece priced as its number
        ('/', '1.2345678901234567e-305', '9.876543210987654e-10'),
        # the RECIPROCAL codeword: 1/c, c a terminating decimal, priced as the divisor `/c`
        ('/', '1', '0.225147640565301'),
        # the SYMBOL TABLE, one probe per entry. Without these, six of the nine entries
        # were invisible to the fingerprint: changing `Pow` from 4 bits to 3 moved no probe.
        ('x0',),                    # leaf
        ('+', 'x0', 'x1'),          # Add
        ('*', 'x0', 'x1'),          # Mul
        ('pow', 'x0', '2'),         # Pow
        ('np.pi',), ('np.e',),      # the named constants
        ('sin', 'x0'),              # an elementary head
        ('asin', 'x0'),             # a transcendental head
        ('atan2', 'x0', 'x1'),      # a 0.15 head (binary; absent where a config lacks it)
        ('float("inf")',),          # the infinities
        # the SIGN, one bit wherever it sits: a literal's sign, a negation standing alone (a
        # sign, not a product, so no `Mul` head), the coefficient -1 beside other factors
        ('-2',), ('neg', 'x0'), ('*', 'neg', 'x0', 'x1'),
        # THE READINGS: compound probes whose price depends on how a mode reads them, so
        # a change to a mode's canonical forms moves its fingerprint. Each mode prices
        # every probe in its own reading; the comments give where the readings differ.
        ('pow', 'pow', 'x0', '2', '0.5'),        # (x0^2)^(1/2): x0 only in permissive
        ('asin', '1e-08'),                       # ground constants are read as written in
        ('*', 'inv', 'exp', '5132.3', 'x0'),     # every mode (the float fold is a walk step)
        ('atanh', 'tanh', 'x0'),                 # x0 except in strict, where tanh saturates
        ('/', 'x0', '*', 'x0', 'x1'),            # a shared factor cancels in every mode
    )

    #: The sidecars published before per-mode fingerprints key each probe by its FIRST
    #: token and price it in strict's reading. These are the compound probes of that
    #: battery; every other legacy key is the probe's whole spelling.
    _LEGACY_PROBE_KEYS: dict[str, str] = {
        '+': '+ x0 x1', '*': '* x0 x1', 'pow': 'pow x0 2', 'sin': 'sin x0', 'asin': 'asin x0'}

    def _measure_fingerprint(self, mode: 'Mode | str' = Mode.strict) -> dict:
        """A fingerprint of the REDUCTION MEASURE in ``mode``, for the provenance sidecar.

        Without it an artifact mined under one measure is indistinguishable from one mined
        under another: the sidecar records the version, the build and the parameters, and
        every one of those can be identical across a measure change (worse, uncommitted
        work stamps the same `-dirty` build string). Artifacts mined before the fingerprint
        existed are identified retroactively by the same values.

        The fingerprint is BEHAVIOURAL, not a version string: it records what the measure
        actually charges on probes chosen to separate the changes it has undergone. Any
        change to `L`, to the fraction/decimal codes, to `mu_free`, or to ANY ENTRY OF THE
        SYMBOL TABLE moves at least one entry -- the last clause is why there is one probe
        per table entry rather than the single bare symbol this used to carry. The compound
        probes are priced in ``mode``'s own reading, so a change to that mode's canonical
        forms moves them too.

        Probes are keyed by their whole token spelling. The price is the certified one
        :meth:`complexity` returns, read from the core directly: the measure and the
        readings belong to the constructor, not to a rule set, so fingerprinting a mode
        never builds that mode's deferred rules.
        """
        mode = _coerce_mode(mode)
        probes: dict[str, int | None] = {}
        for tokens in self._MEASURE_PROBES:
            try:
                probes[' '.join(tokens)] = int(
                    self._core.ac_complexity_certified(list(tokens), 'mode', rule_mode=_RULE_MODE[mode]))
            except ValueError:  # a config that cannot express a probe records it as absent
                probes[' '.join(tokens)] = None
        return {
            'unit': 'milli-bits',
            'mode': mode.name,
            # `mu_sym` is the historical key and stays, for sidecars already published;
            # what it holds is the LEAF entry, which is what a bare symbol costs.
            'mu_sym': probes.get('x0'),
            'mu_leaf': probes.get('x0'),
            'mu_free': probes.get('<constant>'),
            'probes': probes,
            'digest': hashlib.sha256(
                repr(sorted(probes.items())).encode()).hexdigest()[:16],
        }

    def _measure_drift(self, recorded: dict) -> tuple[dict[str, tuple[int | None, int | None]], list[str]]:
        """Compare a sidecar's ``measure`` record with what this engine charges now.

        Returns ``(moved, unrecorded)``: ``moved`` maps each probe the record holds whose
        price differs to ``(recorded, now)``, and ``unrecorded`` lists the probes the
        record does not hold. A probe added to the battery after the record was written
        is unrecorded, not moved. The record is compared in the mode it names; a record
        without a mode is the earlier layout (strict's reading, probes keyed by their first
        token), compared through :attr:`_LEGACY_PROBE_KEYS`.
        """
        held = recorded.get('probes') or {}
        if 'mode' in recorded:
            mode = _coerce_mode(recorded['mode'])
        else:
            mode = Mode.strict
            held = {self._LEGACY_PROBE_KEYS.get(key, key): value for key, value in held.items()}
        current = self._measure_fingerprint(mode)['probes']
        moved = {key: (held[key], now) for key, now in current.items()
                 if key in held and held[key] != now}
        unrecorded = [key for key in current if key not in held]
        return moved, unrecorded

    def _warn_on_measure_drift(self, rules_path: str) -> None:
        """Warn when the sidecar beside ``rules_path`` records a probe the measure now prices
        differently. No sidecar, or one without a measure record, has nothing to compare."""
        sidecar_path = rules_path + '.provenance.json'
        if not os.path.exists(sidecar_path):
            return
        try:
            with open(sidecar_path, 'r') as f:
                recorded = json.load(f).get('measure')
        except (OSError, ValueError):
            return
        if not isinstance(recorded, dict):
            return
        moved, _ = self._measure_drift(recorded)
        if moved:
            mode = recorded.get('mode', Mode.strict.name)
            detail = '; '.join(f'{probe!r}: {old} -> {now}' for probe, (old, now) in moved.items())
            warnings.warn(
                f"measure fingerprint mismatch: the rules at '{rules_path}' were mined under "
                f"a measure that priced {len(moved)} probe(s) differently in mode {mode!r} "
                f"(recorded -> now, in milli-bits: {detail}). The rules stay sound, but their "
                f"minimality and ordering were decided under the other measure.",
                UserWarning)
