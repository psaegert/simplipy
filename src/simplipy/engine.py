"""Core simplification engine for symbolic expressions in prefix notation.

Defines :class:`SimpliPyEngine`, which parses, converts, evaluates, and simplifies
mathematical expressions given as prefix token lists using a configurable set of
operators and rewrite rules. The engine dispatches all simplification, conversion,
validation, and mining work to the compiled Rust extension (``simplipy._core``);
the compiled core is REQUIRED; there is no pure-Python fallback.
"""
import logging
import os
import threading
import warnings
from typing import Callable
from pathlib import Path
from typing import Any, NamedTuple
from copy import deepcopy

import numpy as np
import json

from simplipy.utils import deduplicate_rules
from simplipy.trust import check_realization, package_for, resolve_trusted
from simplipy.io import load_config
from simplipy.asset_manager import get_path
from simplipy._modes import (  # noqa: F401 (simplipy.engine keeps the names it held)
    Mode, _MODE_RULES_ATTR, _REAL_MODE_RETIRED, _RULE_MODE, _canon_of_record, _coerce_mode,
    _coerce_rule_mode)
from simplipy._schema import (  # noqa: F401 (simplipy.engine keeps the names it held)
    _CORE_SERIALIZATION_ARITIES, _TAGGED_DIALECT_TOKENS, _coverage_variants,
    _normalized_operator_specs, _reaches_promise)
from simplipy._evaluation import _Evaluation


#: The library's announcements (the implicit default engine, a deferred set's first-use
#: build) are WARNING records on the ``simplipy`` logger: visible without any logging
#: setup, silenced with ``logging.getLogger('simplipy').setLevel(logging.ERROR)``.
_log = logging.getLogger(__name__)


# The mine half lives in `simplipy.mining` as RuleMiner; the
# engine keeps `find_rules`/`certify_rules` as thin delegators so the public
# API does not move. These aliases keep the module surface stable (tests and
# downstream reach ARTIFACT_ENV_SWITCHES and _MINE_LOCK here, and one miner per
# process requires the lock to stay ONE object across both modules).
from .mining import (  # noqa: E402,F401
    ARTIFACT_ENV_SWITCHES, RuleMiner, _MINE_LOCK,
    _CONSTLIKE_LEAVES, _load_proposals, _tokens_in_vocabulary,
)

# ARTIFACT_ENV_SWITCHES is the
# documented machine-readable registry of artifact-affecting env switches.
__all__ = ['SimpliPyEngine', 'Mode', 'SimplifyStatus', 'ARTIFACT_ENV_SWITCHES', 'DEFAULT_EFFORT']


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
#: The measure-fingerprint check remains the safety net for an artifact loaded by
#: NAME across a measure change; this pin is what stops that happening by default.
DEFAULT_ENGINE = 'acj-5-4-llm'
DEFAULT_ENGINE_REVISION: str | None = None

#: The exploration budget ``simplify()`` runs under when ``effort`` is not given.
#: ``None``: the search runs until a round finds nothing, so a second call returns the
#: answer unchanged. The previous default, 4, stops searches between
#: two improvements: on srbf's 125,127 model predictions 1,632 answers change on a second
#: call in strict at cap 4 (none uncapped).
#: Callers on throughput-critical paths pin ``effort=0``
#: explicitly; ``effort=k`` caps the search at ``k`` candidate descents.
DEFAULT_EFFORT: int | None = None


class SimplifyStatus(NamedTuple):
    """How a :meth:`SimpliPyEngine.simplify` call ended, returned with its answer when it is
    called with ``return_status=True``.

    ``interrupted`` is true where a resource cap stopped the call. Its answer is then the state
    the walk reached by sound steps, never costlier than the input, but possibly not where the
    walk ends: a second call may simplify it further. ``caps`` names the caps that bound:
    ``'steps'`` (the step cap, 10^6 accepted rewrite steps per call), ``'max_passes'`` (the passes
    ran out before one confirmed the fixpoint) and ``'effort'`` (the search budget ran out with
    candidates left to try). Every call is finite (Theorem T6 in ``docs/formal.md``), but not
    short: a chain of n rules can take 2^n - 1 steps, so a binding cap is a resource limit.
    """

    interrupted: bool
    caps: tuple[str, ...]


#: The caps the core reports (`ac_call_status`'s bits), in order.
_CAPS = (('steps', 1), ('max_passes', 2), ('effort', 4))


def _status(bits: int) -> SimplifyStatus:
    caps = tuple(name for name, bit in _CAPS if bits & bit)
    return SimplifyStatus(bool(caps), caps)


class SimpliPyEngine(_Evaluation):
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
    rules_permissive : list[tuple] or None, optional
        The ``permissive`` mode's OWN COMPLETE rule set -- core rules AND the permissive
        rules, in one self-contained list, NOT a supplement to ``rules``: each mode has one
        distinct rule set, so what is loaded IS what is served and the engine never computes
        a union to decide what fires. ``None`` (the default) means this engine names no
        ``permissive`` set of its own and that mode serves ``rules``; ``[]`` is the
        different, sayable statement "the ``permissive`` mode serves nothing".
    rule_canons : dict[str, str] or None, optional
        The canon each set was certified in, which the engine reads it in, by mode name
        (``'strict'``, ``'permissive'``): a mode name for a set mined in that mode, or
        ``'legacy'`` for a set mined before per-mode mines (simplipy 0.14 and earlier mined
        all their files in one canon). A mode not named reads its set in its own canon.
        :meth:`from_config` fills this from each file's provenance sidecar.
    modes : str or Mode or tuple or list, optional
        Which mode rule sets are built EAGERLY, at construction. The default ``'all'``
        builds every set the engine names -- byte-identical to the historical
        behavior, so lean loading is opted into and never inherited. A tuple/list of
        mode names or :class:`Mode` members (e.g. ``modes=('strict', 'permissive')``)
        builds exactly those; a set NOT named builds LAZILY on its mode's first use
        (any ``simplify``/``complexity`` call addressing it) -- once,
        lock-guarded, announced with one WARNING record on the ``simplipy`` logger
        like the default-engine announcement in :meth:`load`, and with a pause of a few
        seconds for a large artifact while the set loads and compiles. The machinery
        is ADDITIVE-ONLY: a lazy build never evicts another mode's structures, and
        nothing is ever dropped implicitly -- dropping is :meth:`unload_mode`, an
        explicit call. The default (strict) set is always built eagerly regardless of
        ``modes``: it is the core's construction substrate (the canon context, the
        dedup key, and the set every absent mode falls back to). Naming ``'strict'`` in
        the tuple is therefore allowed but redundant.

        A built set is compiled at construction too, so the first call in a mode
        pays nothing extra. Measured on the acj-5-4-llm artifact (fresh process each,
        peak RSS directly after ``load``): about 0.7 GB with both sets and 0.45 GB for
        ``modes=('strict',)``, about 100 MB per set; a deferred set then costs its build on
        first use (about 3-4 s on this artifact, announced by the log line).
    defined_functions : dict[str, dict] or None, optional
        The DEFINED FUNCTIONS (a config's ``defined_functions:`` section): name ->
        ``{'arity': n, 'definition': infix}``, the definition written over this engine's
        operators with the arguments ``_0`` to ``_{n-1}``, e.g. ``sqrt: {arity: 1,
        definition: 'rootn(_0, 2)'}``. A defined function is supported without being mined
        (ruling 40: the config is a floor for what the engine supports, not a ceiling):
        :meth:`read_infix`, the conversions and :meth:`is_valid` read and keep its head like a
        declared function's; :meth:`simplify`, :meth:`complexity`, :meth:`prettify`,
        :meth:`mask` and every evaluation (:meth:`as_callable`, :meth:`evaluate_constants`)
        read its definition, so ``simplify('sqrt(x0)')`` answers what
        ``simplify('rootn(x0, 2)')`` answers and the value is the definition's, bit for bit.
        It is not an operator: it is in neither ``operator_arity`` nor the operator config the
        miner, the monitor and the rule files' fingerprints read, so mining is unchanged. A
        definition calls only declared (or core) operators, never another defined function;
        the core refuses any other definition, and a name that is already an operator or an
        alias, when the engine is built.
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
        ``Mode.strict`` serves; the permissive set is a separate attribute and never
        appears here, so a consumer reading this reads exactly what it is served.
    permissive_simplification_rules : list[tuple] or None
        The ``permissive`` mode's own complete rule set, or ``None`` when this engine names
        none and that mode serves ``simplification_rules``. Under a lean ``modes=``
        selection, a config-named set that has not been USED yet also reads ``None``
        here -- it materializes on the mode's first use (and returns to ``None``
        after :meth:`unload_mode`); the engine still knows the set's file and the
        mode still serves it, lazily.
    defined_functions : dict[str, dict]
        The defined functions (see the parameter), ``{}`` for none.
    modules : list[str]
        The importable package names this engine's realizations need (``numpy`` is
        always present: ``np.pi``/``np.e`` are token grammar). These are PACKAGE
        names, not the canonical config spellings -- ``np.sin`` appears here as
        ``numpy``.
    """
    def __init__(self, operators: dict[str, dict[str, Any]], rules: list[tuple] | None = None, *,
                 rules_permissive: list[tuple] | None = None,
                 rule_canons: dict[str, str] | None = None,
                 modes: 'str | Mode | tuple | list' = 'all',
                 trusted_modules: list[str] | None = None,
                 defined_functions: dict[str, dict[str, Any]] | None = None) -> None:
        # Loud validation FIRST, like the operator specs below: a mistyped `modes`
        # must never construct an engine that silently built the wrong sets.
        self._modes = self._normalized_modes(modes)
        self._rule_canons = self._normalized_rule_canons(rule_canons)
        # Loud spec validation + normalization BEFORE anything reads a key --
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

        # THE REALIZATION TRUST MODEL.
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
            # reference nothing, or REFUSES, so nothing passes unchecked to code_to_lambda.
            root = check_realization(str(realization), str(operator))
            if root is not None:
                self._realization_roots.setdefault(root, []).append(operator)
        self._trusted_modules = resolve_trusted(trusted_modules)
        self.modules = sorted({package_for(root) for root in self._realization_roots} | {'numpy'})
        self.import_modules()

        # The raw operator config, kept for core (re)construction.
        self._operators_config = deepcopy(operators)
        # The defined functions: part of the recipe, read by every core this engine builds,
        # and kept apart from the operator config the miner reads.
        self.defined_functions = self._normalized_defined_functions(defined_functions)

        # Normalize the incoming rule list and eliminate duplicate patterns.
        #
        # ORDER: dedup keys on the engine's INTERNAL FORM, so it
        # needs a core -- and the core is built FROM the deduplicated rules. The knot is
        # cut by the fact that the internal form is a function of the OPERATOR TABLE
        # alone: a rules-less core over this very config is a complete and correct
        # canonicaliser, and it is the cheap half of the build (nothing to intern beyond
        # the vocabulary). So: keying core, dedup, serving core.
        dummy_variables = [f'x{i}' for i in range(100)]
        if rules is None:
            self.simplification_rules = []
        else:
            self._core = self._build_core(self._operators_config, [],
                                          defined_functions=self.defined_functions)
            self.simplification_rules = deduplicate_rules(
                rules, dummy_variables=dummy_variables, engine=self, rule_mode='strict',
                canon=self._rule_canons.get('default'))

        # THE PERMISSIVE SET. Normalized to the same (tuple, tuple) shape as the default
        # list, and deliberately NOT deduplicated against it: each set is COMPLETE and
        # self-contained, so there is no cross-set relationship for a dedup to express.
        # Within a set, the core's own translation applies the arbiters that already exist
        # (global first-match-wins ordering, the `lhs == rhs` subsumption drop), exactly as
        # it does for the default set.
        #
        # `None` is preserved as `None` and never flattened to `[]`: `None` means "this
        # engine names no set for that mode, which therefore serves the default one",
        # `[]` means "that mode serves nothing". Collapsing them would make an empty set
        # unsayable.
        self.permissive_simplification_rules = self._normalized_mode_rules(rules_permissive)

        # Build the compiled core (REQUIRED; see the module docstring): every
        # construction path (from_config/load AND direct in-memory construction) attaches it
        # here, from the SAME in-memory state, so no path can exist without a core.
        #
        # LAZY MODES: only the EAGER sets are pushed. A mode outside the
        # `modes=` selection keeps its recipe (the attribute above, or a file source
        # `from_config` attaches after construction) and builds on first use through
        # `_ensure_mode_loaded` -- see the class docstring's `modes` entry.
        eager = self._modes if self._modes != 'all' else ('strict', 'permissive')
        self._core = self._build_core(
            self._operators_config, self.simplification_rules,
            self.permissive_simplification_rules if 'permissive' in eager else None,
            canons=self._rule_canons, defined_functions=self.defined_functions)
        # The lazy-mode bookkeeping. `_mode_rule_sources` maps a rule mode name to the
        # FILE its set loads from (attached by `from_config` for a config-named
        # set); `_core_loaded_modes` holds the non-default modes whose sets the CURRENT
        # core carries (or that were checked and name nothing to load); the lock
        # guards every first-use build and unload so two threads cannot double-build.
        self._mode_rule_sources: dict[str, str] = {}
        self._mode_load_lock = threading.Lock()
        self._core_loaded_modes = {
            m for m in _MODE_RULES_ATTR
            if m in eager and getattr(self, _MODE_RULES_ATTR[m]) is not None}

    @classmethod
    def from_config(cls, config_path: str, *, modes: 'str | Mode | tuple | list' = 'all',
                    trusted_modules: list[str] | None = None) -> "SimpliPyEngine":
        """Creates a SimpliPyEngine instance from a YAML configuration file.

        The configuration file should specify the `operators` and can
        optionally provide a path to a `rules` file.

        Parameters
        ----------
        config_path : str
            The absolute or relative path to the YAML configuration file.
        modes : str or Mode or tuple or list, optional
            Which mode rule sets are built eagerly at load; the default ``'all'``
            builds every set the config names. Under a lean selection (``modes=('strict',)``)
            a config-named set outside it is not even read here -- its resolved
            path is recorded and the file loads on the mode's first use. See the
            class docstring's ``modes`` entry for the full contract and the peak RSS
            measured on the acj-5-4-llm artifact, and :meth:`unload_mode` for the
            explicit drop.
        trusted_modules : list[str] or None, optional
            Extra module roots the config is allowed to import (see
            :mod:`simplipy.trust`). Deliberately an argument HERE and not a key in
            the config: a config that granted itself permission would grant nothing.

        Returns
        -------
        SimpliPyEngine
            A new instance of the engine configured as per the file.

        Raises
        ------
        IncompatibleArtifactError
            The config's artifact generation or ``requires_simplipy`` excludes this
            package, or a rule file it names was mined before simplipy 0.15 (its provenance
            sidecar says by which version; :func:`simplipy.compat.check_rule_file`).
        ValueError
            The config names a ``rules_real`` set: the real mode is retired.
        """
        return cls._from_config(config_path, modes=modes, trusted_modules=trusted_modules)

    @classmethod
    def _from_config(cls, config_path: str, *, modes: 'str | Mode | tuple | list' = 'all',
                     trusted_modules: list[str] | None = None,
                     legacy: bool = False) -> "SimpliPyEngine":
        """:meth:`from_config`, and with ``legacy`` the rule files mined before 0.15 too:
        the explicit opt-in the test suites read their pinned legacy artifacts through
        (``tests/fixtures``), which the public API does not offer."""
        # Loud validation BEFORE any file is read: a mistyped `modes` must fail here,
        # not after the multi-megabyte rule files are already parsed.
        eager = cls._normalized_modes(modes)
        config_path = os.path.abspath(config_path)
        config = load_config(config_path)
        # ARTIFACT-GENERATION GATE: generation-1 artifacts
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

        # THE PERMISSIVE SET (`rules_permissive:`), read by the same resolver as `rules:`
        # above. It names a COMPLETE, self-contained set for its mode -- not a supplement
        # to `rules.json` -- so what is loaded is what is served. The key is OPTIONAL.
        #
        # An ABSENT key leaves the set at `None`, i.e. that mode serves `rules.json`. A
        # configured-but-MISSING file is deliberately NOT an error: an engine without it is
        # fully functional and simply serves that mode the default set, and refusing would
        # make a partially-synced asset directory unloadable. It DOES warn, for the same
        # reason the missing `rules:` file does -- a silently-ignored artifact path is how a
        # broken layout survives to production.
        #
        # An EMPTY file is honoured as an empty set, not as an absent one: `[]` says
        # "this mode serves nothing" and the loader has no business overruling it.
        if 'rules_real' in config:
            raise ValueError(f"the config '{config_path}' names a 'rules_real' set: "
                             f"{_REAL_MODE_RETIRED}")
        mode_rules: dict[str, list | None] = {'permissive': None}
        # Resolved paths of every config-named mode file that EXISTS, attached to the
        # engine below: the recipe `unload_mode` re-reads from, and -- for a set
        # outside a lean `modes=` selection -- the source its lazy first-use build
        # loads, instead of being read here.
        mode_sources: dict[str, str] = {}
        for mode_name in mode_rules:
            declared = config.get(f'rules_{mode_name}')
            if not declared and mode_name == 'permissive':
                # Artifacts published before the 'permissive' rename declare this set
                # under the pre-release key `rules_corpus:` (acj-4 among them). The key
                # and its file keep their published spellings; the loader reads both.
                declared = config.get('rules_corpus')
            if not declared:
                continue
            mode_path = resolve_artifact_path(declared)
            if os.path.exists(mode_path):
                mode_sources[mode_name] = mode_path
                if eager == 'all' or mode_name in eager:
                    with open(mode_path, 'r') as f:
                        mode_rules[mode_name] = json.load(f)
                # else: deferred -- the existence check above keeps the
                # missing-file warning at load time even for a lazy set.
            else:
                warnings.warn(
                    f"the config '{config_path}' declares a {mode_name!r}-mode rule set "
                    f"{declared!r} that could not be resolved (looked for "
                    f"'{mode_path}'): the engine is built WITHOUT it, so Mode {mode_name!r} "
                    f"serves the default rule set instead. The strict mode is unaffected -- "
                    f"each mode's set is its own file.",
                    UserWarning)
        # THE RULE FILES OF THIS VERSION: a file mined before 0.15 is refused, by the version
        # its sidecar records (the legacy opt-in of the test suites excepted).
        if not legacy:
            from .compat import check_rule_file
            for checked_path in [rules_path, *mode_sources.values()]:
                if checked_path is not None:
                    check_rule_file(checked_path, source=config_path)
        # Each file is read in the canon it was certified in, which its sidecar records.
        canons = {mode_name: _canon_of_record(path) for mode_name, path in mode_sources.items()}
        if rules_path is not None and os.path.exists(rules_path):
            canons['strict'] = _canon_of_record(rules_path)
        engine = cls(operators=config['operators'], rules=rules,
                     rules_permissive=mode_rules['permissive'],
                     rule_canons=canons, modes=modes, trusted_modules=trusted_modules,
                     defined_functions=config.get('defined_functions'))
        # The file recipe, attached AFTER construction because paths are this
        # loader's knowledge, not `__init__`'s: a deferred set lazily loads from
        # here on first use, and `unload_mode` re-reads from here after a drop.
        engine._mode_rule_sources.update(mode_sources)
        # WARNING ON THE RESULTING STATE (an engine
        # without rules warns), broadened from the missing-file case it replaces: an engine
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
        # THE MEASURE CHECK. Each rule file's provenance sidecar records what the measure
        # charged on the fingerprint probes when the file was mined. A ruleset served
        # under a different measure stays sound, but its minimality and ordering were
        # decided under the other one, so a probe that moved warns (it never refuses).
        # Every sidecar present is read, one per rule file, each compared probe by probe.
        for checked_path in [rules_path, *mode_sources.values()]:
            if checked_path is not None:
                engine._warn_on_measure_drift(checked_path)
        return engine

    @classmethod
    def load(cls, engine: str | None = None, install: bool = False,
             local_dir: Path | str | None = None, repo_id: str | None = None,
             manifest_filename: str | None = None, *,
             modes: 'str | Mode | tuple | list' = 'all',
             trusted_modules: list[str] | None = None) -> "SimpliPyEngine":
        """Loads a pre-defined engine configuration from the asset manager.

        This provides a convenient way to load standard engine configurations
        distributed with the `simplipy` package.

        Parameters
        ----------
        engine : str or None, optional
            The NAME of an official engine artifact (e.g. ``'acj-4'``). Defaults to
            :data:`simplipy.DEFAULT_ENGINE`, the artifact this simplipy version was
            built and tested against; when it is used implicitly, the choice is
            announced (a WARNING record on the ``simplipy`` logger), because a silent
            default is one a user cannot reproduce.
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
        modes : str or Mode or tuple or list, optional
            Which mode rule sets are built eagerly at load; ``'all'`` (the default)
            builds both, a lean selection such as ``modes=('strict',)`` defers the rest to
            first use. See :meth:`from_config` and the class docstring's ``modes`` entry (with
            the peak RSS measured on the acj-5-4-llm artifact); :meth:`unload_mode` is the
            paired explicit drop.
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
            _log.warning(
                "simplipy: loading the default engine %r%s (simplipy %s); pass engine=... to "
                "choose another", engine,
                f" @ {DEFAULT_ENGINE_REVISION}" if DEFAULT_ENGINE_REVISION else "", __version__)
        # Known generation-1 names refuse BEFORE any download (the config-level gate
        # in from_config covers everything else, including local paths).
        if not os.path.exists(engine):
            from .compat import check_asset_name
            check_asset_name(engine)
        return cls.from_config(
            get_path(engine, install=install, local_dir=local_dir, repo_id=repo_id,
                     manifest_filename=manifest_filename),
            modes=modes, trusted_modules=trusted_modules)

    def mask(self, expression: str | list[str],
             policy: 'str | Callable[[str, Any], str | None]' = 'all',
             *, collect: bool = True) -> str | list[str]:
        """Mask literals: the front door over :mod:`simplipy.masking`.

        Pure delegation to the toolkit -- the mechanism, the role walk, and the
        collect stage that removes the redundant constants it recognises are all
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
            which removes the redundant ``<constant>``s it recognises -- ``2*x0/3`` is
            one free value, and a positional pass alone would abstract it into two.
            Not every redundancy is recognised: ``(a*x + b)/(c*y + d)`` keeps a common
            scale of its four constants. Because the collect stage IS a ``simplify`` call,
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
            tokens = self._core.parse(expression, strict=True)
        elif isinstance(expression, list):
            tokens = expression
        else:
            raise TypeError(
                f'mask expects an infix str or a token list, '
                f'not {type(expression).__name__}')
        # Checked as every reader checks an expression (`check_form`: the token grammar and
        # either dialect's well-formedness), so a malformed tagged sequence (`<add> </add>`) is
        # refused rather than passed through unchanged.
        if tokens:
            self._core.check_form(tokens)
        # A defined function is masked as its definition: the answer is canonical.
        masked = _masking.mask(self._expand_defined(tokens), self, policy_fn, collect=collect)
        return self.to_infix(masked) if isinstance(expression, str) else masked

    def simplify(
            self,
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
            *,
            max_passes: int | None = None,
            mode: Mode | str = Mode.strict,
            effort: int | None = None,
            return_status: bool = False) -> Any:
        """Simplify through the AC CORE: the n-ary associative-commutative engine.

        The AC core represents ``+`` and ``*`` as flat, sorted n-ary bags with EXACT rational
        coefficients composed explicitly (``* 7 x``, ``pow x 3``, ``rootn x 3``) instead of the
        retired hyper-operator vocabulary. Rules are widened to SUB-MULTISET
        matching with the unmatched remainder preserved, so a rule fires wherever the algebra
        permits, regardless of operand order or bracketing -- both axes of the binary engine's
        commutative-order invariance defect are removed at the representation level, and the
        result is invariant under permutation of commutative operands by construction.

        Like-term/like-factor collection (the AC form of term cancellation) runs inside the
        canonical constructors under the same soundness certificates as the rule matcher
        (finite-a.e. for sign-cancelling addition, finite-and-nonzero-a.e. for
        exponent-cancelling multiplication), and coefficient arithmetic is exact rational
        computation rather than mined rules. Every mode runs one walk: its chain of rules and
        constructors, then the search, in its own canonical form and rules. The output is
        idempotent (``simplify(simplify(x)) == simplify(x)``) when the search runs until a
        round finds nothing, the default in every mode; a search that an ``effort`` cap
        stops can continue on a second call.

        .. note::
            Pair this engine with sort-promoted rulesets, which every published engine
            (such as ``acj-5-4-llm``) is. Generation-1 artifacts (the retired hyper-operator
            vocabulary, such as ``dev_7-3``) are refused at load; they need
            ``simplipy<0.12``.

        Parameters
        ----------
        expression : str | list[str] | tuple[str, ...] | np.ndarray
            The expression to simplify: a ``str`` is parsed as infix; a list, tuple or
            1-D ``ndarray`` is a prefix token sequence (old grammar and tagged form are
            both accepted -- one shared parser).
        max_passes : int, optional
            Safety cap on the number of outer REWRITE PASSES -- not on nodes, and not on
            rule firings. One pass is one full rewrite sweep over the whole expression,
            and it reaches the fixpoint: it re-enters every node it changes, so the next
            pass only confirms that nothing moves. Defaults to 48; the cap is
            defense-in-depth against an ordering bug (the descent is well-founded, so the
            fixpoint is reached in finitely many passes: Theorem T6 in ``docs/formal.md``),
            not a tuning knob. Must be at least 1; ``0`` raises ``ValueError``.

        effort : int or None, optional
            The SEARCH BUDGET: after the chain reaches its fixpoint, an
            exploration phase proposes the moves of the distributive law the strict
            descent refuses (distributing a product over its sums, expanding an integer
            power of a sum, factoring what terms of a sum share), runs each candidate
            through the same certified constructors and the same descent loop, cheapest
            first, and replaces the result only when the candidate's endpoint lands
            STRICTLY below it in the serve-time reduction ordering.
            ``None`` (the default, ``DEFAULT_EFFORT``) searches until a round finds
            nothing, so a second call returns the answer unchanged (up to the residual
            classes listed in ``docs/formal.md``);
            an int caps the search at that many candidate descents, after which
            a second call can continue it; ``0`` never enters the phase and is byte-identical to the plain chain
            (rules and constructors), which does not factor.
            Every guarantee survives any budget: soundness (same certificates), never-worse
            (strictly-below acceptance: never above the input, and never above the answer at a
            smaller ``effort``), termination (well-founded ordering,
            independent of the budget) and deterministic output.
        return_status : bool, optional
            Return ``(answer, status)``, where ``status`` is a :class:`SimplifyStatus` that says
            whether a resource cap interrupted the call (the step cap, ``max_passes`` or the
            ``effort`` budget). Default ``False``: the answer alone.

        Returns
        -------
        str | list[str] | tuple[str, ...] | np.ndarray
            The simplified expression (with ``return_status=True``, the pair ``(answer,
            status)``), in the same container and dialect as the input: a
            ``str`` answers in infix, a token sequence carrying a bag delimiter in the tagged
            form, and any other token sequence in the explicit binary prefix.

            * tagged -- the STRICT prefix form, the AC engine's native serialization: n-ary bags are delimited (``<add> ... </add>``,
              ``<mul> ... </mul>``) and carry their group inverse as a SECTION -- terms
              after ``<sub>`` subtract, factors after ``<div>`` divide, so
              ``(2*x1)/(x2*x3)`` is ``<mul> 2 x1 <div> x2 x3 </mul>``. ``pow`` and the
              unary functions stay plain prefix; ``neg`` exists only as the standalone
              unary spelling (``tan neg x0``), and a reciprocal or negative power is a
              quotient like any other (``<mul> 1 <div> x0 </mul>``, ``<mul> 1 <div> pow x0 3
              </mul>``, as ``to_tagged`` writes ``inv x0`` and ``inv pow x0 3``) -- inside bags
              the sections own all inverses. Integers are one token (``7``), every fraction is a
              bag (``<mul> 1 <div> 3 </mul>``), and so is a small decimal (``0.2`` is
              ``<mul> 1 <div> 5 </mul>``); other decimals are one token. So a tagged answer
              that holds a fraction always carries a bag and reads back as tagged. Tagged
              output is accepted back as input (one shared, liberal parser).
            * infix -- the PRETTY human-readable rendering, a ``str``: ``x8 + 1.2*x3``, ``-x0/3``, ``(x0 + 1)^2``.
              Round-trips: feeding the rendering back as a ``str`` input reaches the same
              canonical state. The bare names ``pi``, ``e``, ``inf`` and ``nan`` are
              RESERVED constant spellings of the infix language (the renderer emits them
              for the canonical constants, so the parser reads them back as constants,
              never as variables).
            * explicit -- the binary-chain form (``* 6 x``, ``- a b``, literal
              coefficients only -- the hyper-operator vocabulary is deleted). This is
              the project's INTERNAL dialect, not a recommended consumer format:
              rule artifacts are stored in it, the miner's enumeration universe and
              the certificate tapes are defined on it, and it stays available here
              for debugging and compatibility. Tagged and infix are the public output
              forms. Sign placement follows the same doctrine as the
              tagged form: a sign lives in the numeric coefficient literal whenever
              one is emitted (``* -2 x``), additive term signs are structural
              (binary ``-``), and ``neg`` spells only the pure sign (``neg x0``,
              ``neg * np.pi x0``).

            Only the infix answer is SPELLED FOR A READER: outside exponents, a value with
            no finite decimal whose numerator has a factor 2 or 5 is written as an integer
            over a decimal where that is shorter (``1/6.283185307179586``, not
            ``500000000000000/3141592653589793``); ``1/3`` stays ``1/3``. It re-reads to
            the same state, and :meth:`complexity` does not change. The token answers keep
            the engine's own
            spelling, because tokens are what gets read back -- by the engine's
            certificates, served rules and miner, and by callers that mask or key on them
            -- and a spelling chosen for a reader must not move any of those. To read a
            token answer, convert it: ``simplify(to_infix(tokens))``.

        Raises
        ------
        ValueError
            On a malformed expression, including infix text that :meth:`read_infix` refuses
            (two operands separated only by whitespace, an ambiguous product inside the
            argument of a function without parentheses).
        """
        if max_passes is None:
            max_passes = 48
        if max_passes < 1:
            # 0 would run no pass at all (the core reads it literally), and a negative
            # value would surface as a raw pyo3 OverflowError at the usize conversion
            #
            raise ValueError(f"max_passes must be at least 1, got {max_passes}")
        if effort is None:
            effort = DEFAULT_EFFORT
        # None stays None: the core then searches until a round finds nothing.
        if effort is not None:
            if isinstance(effort, bool):
                # bool is an int subclass and would silently mean 0 or 1 candidate
                # descents -- a type error, not a budget.
                raise TypeError(f"effort must be None or an int >= 0, not bool ({effort!r})")
            try:
                # __index__: plain and numpy ints alike (max_passes takes numpy ints
                # through pyo3's __index__ extraction; the two int knobs agree).
                effort = effort.__index__()
            except AttributeError:
                raise TypeError(
                    f"effort must be None or an int >= 0, not {type(effort).__name__} ({effort!r})") from None
            if effort < 0:
                raise ValueError(f"effort must be non-negative, got {effort}")
            if effort > 2 ** 63 - 1:
                effort = None  # beyond the index range no cap binds: the uncapped search
        mode = _coerce_mode(mode)

        # DIALECT-PRESERVING: `simplify` only simplifies and
        # answers in the form it was handed -- a str is infix, a token sequence carrying
        # a bag delimiter is tagged, anything else is the explicit binary prefix, so a
        # prefix input never comes back tagged; to change the NOTATION, convert:
        # `simplify(to_tagged(x))`. A defined function enters as its definition.
        form, tokens = self._read_input(expression, expand=True)

        # ONE lookup, both branches: the rule mode is decided here and the core is asked
        # for it by name, so the two output paths cannot end up serving different sets.
        # LAZY MODES: a set deferred by a lean `modes=` selection builds in
        # `_serve`, on the mode's first use -- once, lock-guarded, announced; under the
        # default `modes='all'` that is one set-membership test.
        rule_mode = _RULE_MODE[mode]
        if form == 'infix':
            answer, bits = self._serve(mode, lambda core: (core.ac_simplify_infix_in_mode(
                tokens, max_passes, effort, rule_mode=rule_mode),
                core.ac_call_status() if return_status else 0))
        else:
            out, bits = self._serve(mode, lambda core: (core.ac_simplify_in_mode(
                tokens, max_passes, form, effort, rule_mode=rule_mode),
                core.ac_call_status() if return_status else 0))
            answer = self._denormalize(out, expression)
        if return_status:
            return answer, _status(bits)
        return answer

    def prettify(
            self,
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
            *,
            digits: int = 3,
            precision: int | None = None,
            form: str | None = None,
            keep_grouping: bool = False,
            mode: Mode | str = Mode.strict) -> str | list[str]:
        """Write an expression for a READER: each number written to a digit budget, nothing else decided.

        :meth:`simplify` returns the canonical form, whose numbers are exact
        (``2.14583142772210003669120562399025*x1`` stays as long as it was given). ``prettify``
        is the display of that form: it decides how each number is WRITTEN and leaves the
        structure as it is: it changes how a number is written, never the structure, and
        applies no rules. Nothing inside SimpliPy calls it or reads what it returns; call it
        where people read expressions, on the answer of :meth:`simplify`::

            engine.prettify(engine.simplify('2.14583142772210003669120562399025*x1'))
            # '2.1458314277221002*x1'
            engine.prettify(engine.simplify('x1*5/21'))        # '5*x1/21'
            engine.prettify(engine.simplify('1/(2*3.141592653589793)'))   # '1/(2*pi)'
            engine.prettify(engine.simplify('x1*exp(-381/2)'))
            # '1.8488479557720672e-83*x1'

        The rules, at the digit budget ``D = digits`` (the significant digits a spelling
        writes count, never an exponent's, a numerator of 1 included: ``1/12`` counts 3,
        ``2.5e307`` counts 2, ``0.05`` counts 1). Each number is written in the first form
        that fits ``D``:

        * Exactly, as one number, as :meth:`simplify` writes it, a decimal in the notation
          below (``3*x1/10`` is ``0.3*x1``, ``x2 + 0.5`` is ``x2 + 1/2``, ``5/21`` beside a
          factor is ``5*x1/21``, ``1000/333`` is ``x1/0.333``, ``22/7`` stays ``22*x1/7``,
          ``x1/2`` stays ``x1/2``).
        * At full precision, as a short exact sum ``s + r`` or ``s - r``: ``s`` the number's
          first significant digits, ``r`` the rest to its value, each written exactly as one
          number, within ``D`` in total; so it keeps the value (``x1 + 1.000000000000000000003`` is ``x1 + 1 + 3e-21``,
          ``2.5e30 + 1e15`` is ``2.5e30 + 1e15``, ``0.30000000000000004`` is ``0.3 + 4e-17``).
          A number float64 writes exactly in 15 digits (``1905``, ``1.0001``) is written as its
          float instead.
        * Otherwise the float64 nearest to it, in its shortest form, as a multiplier
          (``x1/1024`` is ``9.765625e-4*x1``, ``0.0009853704666076428*x1`` is
          ``9.853704666076429e-4*x1``). A number shown as a float is read as a float. With
          ``precision=k`` such a float shows ``k`` significant digits instead, with no rest
          (``2.1458314277221002*x1`` is ``2.15*x1`` at ``precision=3``), and that number is
          written by these rules as every number is, so the display reads back as itself
          (``0.175`` is 7/40, written ``7/40``) and no short sum is written; a number written
          exactly is not affected.
        * pi: a value that is an exact number within ``D`` times or over the literal
          ``3.141592653589793`` is written with pi factored out as the symbol ``pi``
          (``np.pi`` in the token forms): ``x0/(2*pi)``, ``0.15*x0^2/(pi*x1*x2)``. In ``strict``
          the symbol is that float. It holds at every precision: pi is never shortened, and a
          precision applies to the other numbers only. The literal is written instead where
          the symbol would read back as another expression (reading ``sin(x1 + pi)`` builds
          ``-sin(x1)``, so ``sin(x1 + 3.141592653589793)`` keeps the literal), and where the
          expression holds the symbol pi itself.
        * Where writing a number so would change the structure -- it would be 1 and a
          multiplier or an exponent would drop, an integer where a non-integer exponent stood,
          or equal to another number so that parts collect or cancel -- it is written as that
          spelling and the rest to its value, the rest by the same rules
          (``1.0000000000000000000031*x1`` is ``(1 + 3.1e-21)*x1``,
          ``sin(1.0001*x1) - sin(x1)`` at ``precision=3`` is ``sin((1 + 1e-4)*x1) - sin(x1)``).
          Of two numbers that would be equal, the one written first keeps its spelling.
        * A float whose reciprocal float64 holds as written stays a divisor where the
          product has another divisor; a bare ``x1/0.225147640565301`` is
          ``4.441529999999994*x1``.
        * A function of numbers that would all be written as floats is one float: its
          float64 as SimpliPy's evaluator computes it, shown at the ``precision``
          (``x1*exp(-381/2)`` is ``1.8488479557720672e-83*x1``, ``1.85e-83*x1`` at
          ``precision=3``). A function of a number written exactly stays (``sin(2)``), and so
          does one whose float would be 0, inf or subnormal, and one whose float would let
          reading the display build another sum, a number distributed over a sum or a term
          collected with a like term (``x1 + atan(1.2345678)*(x2 + x3)``,
          ``x1 + x2 + x1/log(0.7377233226141932)``). No numbers combine to inf, 0 or a
          subnormal.
        * Exponents follow the same rules; ``x^(1/n)`` and ``x^(-1/n)`` are written
          ``rootn(x, n)`` and ``1/rootn(x, n)`` for even ``n``.
        * One slash: every divisor of a product joins one denominator, functions and sums
          included (``x1/x2/log(x3)`` is ``x1/(x2*log(x3))``, ``x1/sin(x3)/(x2 + 1)`` is
          ``x1/(sin(x3)*(x2 + 1))``). The display is of the product reading it back builds:
          the same whether the canonical form holds a reciprocal per factor or one of their
          product, its factors in canonical order and a sign the denominator holds kept there
          (``x3/(-(x11 + x9))``). Tagged tokens write a section member per reciprocal, so a
          reciprocal of a product with a sum or a function stays one bag in the section.
        * Scientific notation from ``1e6`` and below ``1e-3``.

        At the default ``precision=None`` a number is written exactly, as a short exact sum or
        as the float64 nearest to it, so the display reads back to the same float64 values: it evaluates as the expression does, up to the order of its
        operations (a number the explicit tokens write as a quotient, ``x1/0.225147640565301``,
        is written as the float of that quotient, ``4.441529999999994*x1``, one rounding
        apart). A sum is written as one flat chain, ``x1 + x2 + exp(x3)``, and an infix
        product as one run of factors, which reading groups from the left; SimpliPy's explicit
        tokens group them from the right (``+ x1 + x2 exp x3``), and where terms nearly cancel,
        or a function amplifies its argument, the two orders give different float64 values.
        ``keep_grouping=True`` writes every sum and product with the grouping of the explicit
        tokens, ``x1 + (x2 + exp(x3))``, ``2*(x1*(x2*x3))``, and at full precision every number
        exactly, as the tokens write it (``sin(3.1415926535897e37*(x1*x2)/3313)``, where the
        default writes the quotient's float ``9.482621954692726e33``), so the display evaluates
        as the expression does, bit for bit. With a ``precision`` the display does not read back
        to the same values: a float shortened to ``k`` digits reads back as another number. At
        every precision the display reads back to the expression's structure, and displaying a
        display gives it back (``prettify(prettify(x)) == prettify(x)``).

        Parameters
        ----------
        expression : str | list[str] | tuple[str, ...] | np.ndarray
            Infix text, or explicit or tagged prefix tokens (as :meth:`simplify` reads them).
            Its canonical state is displayed; pass the answer of :meth:`simplify`.
        digits : int, optional
            The digit budget ``D`` (default 3).
        precision : int or None, optional
            The significant digits a number written as a float shows. ``None`` (the default)
            writes the shortest decimal that reads back as the same float64.
        form : str, optional
            ``'infix'``, ``'explicit'`` or ``'tagged'``; by default the input's dialect, as
            :meth:`simplify` answers.
        keep_grouping : bool, optional
            Write every sum and product with the grouping of SimpliPy's explicit tokens of the
            canonical form, the order in which it evaluates the expression: a sum's positive
            terms grouped from the right, then each subtracted term in turn
            (``x1 + (x2 + exp(x3)) - x4``), a number written with its rest as one term
            (``x1 + (1 + 3e-21)``), and a product's factors grouped from the right
            (``x1*(x2*x3)/(x4*(x5*x6))``), and at full precision every number exactly, as the
            tokens write it. Default ``False``: one flat chain per sum and product. The tagged form writes bags, which have no order, and is the same either
            way.
        mode : Mode or str, optional
            The mode whose canonical constructors read the expression, ``Mode.strict`` by
            default, as :meth:`simplify` answers by default. Pass the mode the answer was
            simplified in: a ``permissive`` answer read in ``permissive`` keeps the groups of its
            shared numbers (``x3 + 3.71649*(x1 + x2)``), which ``strict`` reads distributed
            (``3.71649*x1 + 3.71649*x2 + x3``). ``keep_grouping`` groups that state too.

        Returns
        -------
        str | list[str]
            Infix text when the output form is ``'infix'`` (an infix input, or
            ``form='infix'``), else a token list.
        """
        if isinstance(digits, bool) or not isinstance(digits, (int, np.integer)):
            raise TypeError(f"digits must be an int >= 1, not {type(digits).__name__} ({digits!r})")
        if digits < 1:
            raise ValueError(f"digits must be at least 1, got {digits}")
        if precision is not None:
            if isinstance(precision, bool) or not isinstance(precision, (int, np.integer)):
                raise TypeError(
                    f"precision must be None or an int >= 1, not {type(precision).__name__} ({precision!r})")
            if precision < 1:
                raise ValueError(f"precision must be at least 1, got {precision}")
            precision = int(precision)
        if not isinstance(keep_grouping, (bool, np.bool_)):
            raise TypeError(f"keep_grouping must be a bool, not {type(keep_grouping).__name__} ({keep_grouping!r})")
        in_form, tokens = self._read_input(expression, expand=True)
        target = form or in_form
        if target not in ('infix', 'explicit', 'tagged'):
            raise ValueError(f"unknown form {form!r}: expected 'infix', 'explicit' or 'tagged'")
        if not tokens:
            return '' if target == 'infix' else []
        return self._core.prettify(tokens, int(digits), target, precision,
                                   keep_grouping=bool(keep_grouping),
                                   rule_mode=_RULE_MODE[_coerce_mode(mode)])

    def complexity(
            self,
            expression: str | list[str] | tuple[str, ...] | np.ndarray,
            certified: bool = True,
            mode: Mode | str = Mode.strict) -> int:
        """The SEMANTIC COMPLEXITY of an expression in a mode: its price as that mode reads it.

        Each mode reads an expression into its own canonical form, because the modes accept
        different simplifications as true: ``strict`` follows the deployed float evaluator
        wherever no operation overflows or underflows (``atanh(tanh(x))`` is not ``x``: in
        floats ``tanh`` saturates to ``1``; past an overflow the true value is accepted as
        well, so ``exp(x)/exp(x)`` is ``1``), and ``permissive`` real numbers
        (``atanh(tanh(x))`` is ``x``) and also what holds only up to sign or domain
        (``(x^2)^(1/2)`` is ``x``, where ``strict`` gives ``abs(x)``). A ground
        constant is read as written in every mode: folding it to its float is a step of
        :meth:`simplify`'s walk in ``strict`` and ``permissive``, not part of the reading.
        This prices that form, on the
        CERTIFIED canonical state (the same certificate-carrying canonicalization the
        simplify chain runs on), with the one codebook (the unified measure mu). It is the
        measure :meth:`simplify` descends in that mode, so for every mode
        ``complexity(simplify(e, mode=m), mode=m) <= complexity(e, mode=m)``. Prices in
        different modes price different readings and are not comparable across modes;
        :func:`simplipy.mining.cross_mode_complexity` prices answers of every mode on one
        shared scale.

        With ``certified=False`` the expression is priced on the bare
        (certificate-less, fail-closed) canonicalization instead: still invariant
        to operand order, bracketing and serialization sugar (a product is built once from
        all of its factors). A certificate-licensed respelling the bare context
        cannot re-derive keeps its own measure, so a simplify output can price ABOVE its
        input (measured live: 0.48% of 64k corpus rows, in quanta of one symbol unit; the
        certified default closes exactly this).

        Invariant in every case: mu prices structure and information, not spelling --
        a sign costs one bit wherever it sits (a literal's sign, a coefficient -1, a
        negation standing alone, which is a sign and not a product: ``-x1`` costs one bit
        more than ``x1``, ``x1 - x2`` one bit more than ``x1 + x2``), a coefficient +1 is
        free and an exponent -1 is division, literals pay their description length on the
        exact value, symbols pay their entry of the symbol table.

        The code book prices every number as it is spelled: a number carrying more digits
        than its float needs costs those digits (:meth:`written_complexity` charges them).
        The reading, like :meth:`simplify`, writes the cheaper value its mode admits wherever
        a literal lands (the whole expression, a sum's constant term, a coefficient, a power's
        literal operand), so ``complexity('0.1000000000000000055511151231257827')`` is
        ``complexity('0.1')``: ``strict`` reads float64's double of the literal,
        ``permissive`` whichever of the two doubles around it prints shorter, which need not
        be the nearest (``0.3000000000000000444`` reads as ``0.3``). A ground constant is read
        as written (above), so ``sin(0.1000000000000000055511151231257827)`` pays every
        digit. ``strict`` and ``permissive`` admit a number only where its numerator and
        denominator are at most 2^1022, and hold any other value in pieces
        (``1.2345678901234567e-305`` as written, and its quotient by ``9.876543210987654e-10``
        as that literal times a number), which cost at most what the exact number is priced
        at. A literal beyond the 1,100-bit cap keeps its own price. The price is an integer
        in milli-bits.

        :meth:`written_complexity` is the other half of the pair: the same code book applied to
        the expression exactly as written, with no mode and nothing read into a canonical
        state. ``complexity`` measures what an expression is in a mode (srbf's symbolic
        recovery); ``written_complexity`` measures what was written (srbf's MDL).
        """
        _, tokens = self._read_input(expression, expand=True)
        rule_mode = _RULE_MODE[_coerce_mode(mode)]
        # LAZY MODES: same first-use hook as `simplify` -- the mode
        # argument reaches the core below, so a deferred set must be built first.
        self._ensure_mode_loaded(rule_mode)
        if certified:
            return self._core.ac_complexity_certified(tokens, 'mode', rule_mode=rule_mode)
        return self._core.ac_complexity(tokens, 'mode', rule_mode=rule_mode)

    def written_complexity(
            self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> int:
        """The WRITTEN COMPLEXITY of an expression: its price exactly as written, in milli-bits.

        The code book of :meth:`complexity` (the measure mu: its symbol table, the sign bit and
        the literal code) applied to the expression as written. There is no mode and no
        certificate: nothing is read into a canonical state and nothing is simplified, so every
        operator, function, variable, sign and number written is charged. The only
        identifications are the ones infix notation forces, and where the notation leaves a
        choice, the reading is the one the code book prices:

        * a run of ``+`` and ``-`` is ONE sum and a run of ``*`` and ``/`` ONE product, whatever
          the bracketing: a left operand and a ``+`` (``*``) operand continue the run, a
          bracketed subtrahend or divisor stays whole; operands are unordered. A sum pays its
          head (3 bits) once, a product its head (3 bits) once. A call of ``max`` or ``min``
          with more than two arguments, ``max(a, b, c)``, is ONE call and pays its head once
          (the reader nests it to the right, ``max(a, max(b, c))``, and a right operand of the
          same head continues the call; ``max(max(a, b), c)`` writes two calls);
        * ``-y`` and ``neg(y)`` are a SIGN, one bit on ``y``, as is the sign of a subtrahend; a
          written ``-inf``, and ``inf`` as a subtrahend, is the symbol ``float("-inf")`` (8 bits);
        * ``1/y`` is the RECIPROCAL of ``y`` (the 1 is the notation's) and ``inv(y)`` is the same:
          a divisor, which pays the power head (3 bits) over itself, as ``complexity`` prices
          ``y^-1``. ``-1/y`` is the sign of the reciprocal, ``-(1/y)``: a numerator ``-1`` is one
          sign bit (``-1/x0`` costs 10 bits). Dividing by a written power ``y^k`` is ``y^-k`` when
          ``k`` is one number (a literal, ``1/c`` or ``-1/c``), the division's sign merging into
          it (``x/y**2`` holds y^-2 and ``x/y**-2`` holds y^2), and the reciprocal of the power
          otherwise (``x/y**z``: the power head over ``y**z``). Dividing by a number ``c`` is the
          literal ``1/c`` (``x/3`` holds the literal 1/3, as ``complexity`` prices it);
        * a NUMBER pays the literal code on the EXACT value written, with every digit: a long
          literal is not rounded to its double (``6.674000000000001e-11`` costs its 16 digits),
          and a written minus is its sign bit. A FRACTION costs its division and a decimal its
          value (the owner's ruling, 2026-11-28): ``p/q`` is ``p`` and the literal ``1/q``, as
          any number divided by a number, so ``3/4`` costs 10.322 bits where ``0.75`` costs
          5.322, ``x0**(3/2)`` costs 18.585, and ``3/4*x0`` and ``4*x0/3`` cost the same, 16.322
          (``2*x0*3`` 14.585: numbers written apart pay apart). Only ``1/q`` and ``-1/q`` are one
          literal, the reciprocal's 1 being the notation's (``1/2`` 3.585 bits);
        * ``np.pi``, ``np.e``, the infinities, NaN, ``<constant>`` and named constants
          (``<constant_NAME>``) are priced exactly as ``complexity`` prices them; every other
          symbol is a variable (6 bits);
        * a function pays its head's entry of the symbol table and its arguments, ``pow`` its
          head, base and exponent; an alias is priced as its target (``ln`` as ``log``);
        * a DEFINED FUNCTION (the config's ``defined_functions:``) pays its own head at the tier
          of the function its definition applies: ``sqrt`` at ``rootn``'s (6 bits, without the
          2), ``cot``, ``sec`` and ``csc`` at ``tan``'s, ``cos``'s and ``sin``'s (6), and ``acot``,
          ``asec`` and ``acsc`` at ``atan``'s, ``acos``'s and ``asin``'s (8). Any other head the engine
          neither declares nor defines is refused with ``ValueError``, never guessed.

        For example ``x0 + 0`` costs 11 bits (``complexity``: 6), ``x0 + x0`` 15 (11.585),
        ``exp(1)`` 8 (4), ``-(x0 - x1)`` 17 (16), ``sqrt(x0)`` 12, ``x0**0.5`` 12.585 and
        ``x0 - x0/1 + 2`` 23.585 (2.585); ``6.674e-11*x0*x1/x2**2`` costs 45.197 bits either way,
        and ``x0*x1/(4*np.pi*x2*x3**2)`` 45.907 written where its canonical form, which
        distributes the reciprocal, costs 47.907: the written price is not bounded by the
        canonical one in either direction.

        It reads an infix ``str`` as :meth:`read_infix` does, strictly (an undeclared call is
        refused), or explicit prefix tokens as :meth:`read_infix` returns them; a tagged
        sequence is read through :meth:`to_prefix`, and realization notation as the conversions
        read it. The price is an integer in milli-bits, on the scale of :meth:`complexity`.

        On a canonical form the two prices agree where the form is written as its state is held,
        which is not promised: the canonical serializations write some states otherwise.
        Measured over the 0.15 census (31,993 canonical answers per mode: srbf's 6,531 ground
        truths and a tenth of flash-ansr T8.1's training inputs and of srbf's predictions), the
        written price of the answer equals ``complexity`` on 89.81% (``strict``) and 88.98%
        (``permissive``) of the infix answers, and on 87.66% and 87.09% of the prefix answers.
        Every difference is a serializer's: a fraction the state holds as one number, printed
        ``p/q``, which costs its division written and its value canonical (``3/4``, 10.322 bits
        against 5.322; ``x0^(3/2)``, 18.585 against 13.585), a negation distributed over a sum
        (``-x0 - x1`` for ``-(x0 + x1)``, 17 bits against 16), a coefficient p/q printed
        ``p*x/q``, its numbers apart (``4*x0/3``, 16.322 against 14.322), and in the prefix
        answers a product of reciprocals written as the reciprocal of a product (``inv * x0
        x1``, 18 against 21) and a sign on a product inside a product (``/ neg * x0 x1 x2``, 28
        against 25); a few answers print a factored form that ``complexity`` reads distributed.
        """
        if isinstance(expression, str):
            tokens = self._core.parse(expression, strict=True)
        else:
            form, tokens = self._read_input(expression)
            if form == 'tagged':
                tokens = self._core.to_prefix_syntactic(tokens)
        return self._core.written_complexity(list(tokens))

    def _denormalize(
            self,
            out: list[str],
            expression: list[str] | tuple[str, ...] | np.ndarray,
    ) -> list[str] | tuple[str, ...] | np.ndarray:
        """Map a core token result back to a token input's container (``simplify``'s tail;
        a ``str`` input answers in infix before reaching it)."""
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

        Thin delegator to :meth:`simplipy.mining.RuleMiner.find_rules`: the
        signature, the semantics and the one-mine-per-process lock are the miner's."""
        return RuleMiner(self).find_rules(*args, **kwargs)

    # the delegators carry the miner's REAL signature for inspect/docs/tests
    find_rules.__wrapped__ = RuleMiner.find_rules  # type: ignore[attr-defined]

    def certify_rules(self, *args: Any, **kwargs: Any) -> Any:
        """Certify proposed rules against this engine, in an explicit ``mode``.

        Thin delegator to :meth:`simplipy.mining.RuleMiner.certify_rules`: the
        signature and the semantics are the miner's; each result is a
        :class:`~simplipy.mining.CertifiedRule` with its mode and basis."""
        return RuleMiner(self).certify_rules(*args, **kwargs)

    certify_rules.__wrapped__ = RuleMiner.certify_rules  # type: ignore[attr-defined]
