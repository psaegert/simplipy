"""Rule mining: discovering, certifying and checking an engine's rule files.

:class:`RuleMiner` holds the mining and certification methods. It is an ordinary
object over an engine, ``RuleMiner(engine).find_rules(...)``, and owns no state of its
own beyond the engine reference; ``SimpliPyEngine.find_rules`` and ``certify_rules``
are thin delegators to it. One mine runs per process: a module-level lock serializes
every mine and certify entry point, however many miners exist.

:func:`verify_mine` checks a mined file against the re-mine gates in its own mode, and
:func:`cross_mode_complexity` prices answers of different modes on one shared scale.
"""

import hashlib
import os
import re
import signal
import threading
import time
from typing import Callable, TYPE_CHECKING
from typing import Any
from copy import deepcopy

import numpy as np
import copy
import functools
import json

from simplipy.utils import (
    is_numeric_string,
    deduplicate_rules,
    dedup_target_wins,
    _dedup_keyed_rules,
    enumerate_expressions, count_expressions,
    remap_expression)

if TYPE_CHECKING:
    from .engine import SimpliPyEngine

# D11 column R30 (owner-ratified 2026-08-16): the miner is this module's
# power-user surface; the module helpers (_MINE_LOCK, ...) are private.
__all__ = ['RuleMiner', 'verify_mine', 'cross_mode_complexity']


# THE ARTIFACT-AFFECTING SWITCH REGISTRY: every environment variable that can change a
# mined artifact, a certification verdict or a served answer. The core declares the list
# (`rust/env_switches.rs`, the constants its read sites use), so it cannot drift from what
# is read. The mine's provenance writer records every SET entry verbatim
# (`soundness.env_overrides`). The core reads its switches ONCE per process (at first
# use), so the record is faithful unless the caller mutates os.environ between first
# engine use and the mine.
try:
    from simplipy._core import ARTIFACT_ENV_SWITCHES as _CORE_ENV_SWITCHES  # type: ignore[attr-defined]
except ImportError:  # no extension: engine construction raises with the reason anyway
    _CORE_ENV_SWITCHES = []
ARTIFACT_ENV_SWITCHES: tuple[str, ...] = tuple(_CORE_ENV_SWITCHES)

_CONSTLIKE_LEAVES = {'<constant>', 'np.e', 'np.pi',
                     'float("inf")', 'float("-inf")', 'float("nan")'}

_MINE_LOCK = threading.Lock()

# A slot token is a SORT SIGIL plus an index (`?0` variable-leaf, `_0` subtree, `!0`
# certified subtree, `$0` certified-nonzero subtree). The promotion respells the sigil
# in place and never renumbers, so erasing it leaves an identity stable across sorts.
_SLOT_SIGIL = re.compile(r'^[_!?$](\d+)$')


def _rule_identity(lhs: Any, rhs: Any,
                   dummy_variables: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """One key for every spelling a rule wears between certification and the artifact.

    The proposal channel certifies CONCRETE variables (`x0`), :func:`deduplicate_rules`
    canonicalises them to the `?` sort, and the promotion pass then respells the sigil
    (`?0` -> `_0`/`!0`/`$0`). Tracking a certified pair through the post-mine stages
    needs an identity invariant under all three, so the variable canonicalisation runs
    first and the sort sigil is erased after it.
    """
    source, mapping = remap_expression(list(lhs), dummy_variables, variable_prefix='?')
    target, _ = remap_expression(list(rhs), dummy_variables, mapping, variable_prefix='?')
    return (tuple(_SLOT_SIGIL.sub(r'@\1', token) for token in source),
            tuple(_SLOT_SIGIL.sub(r'@\1', token) for token in target))


def _reconcile_proposal_census(
        outcomes: dict, trail: list, watch: dict, drops: dict,
        rules: Any, dummy_variables: list[str]) -> None:
    """Rewrite the proposal census so it describes the ARTIFACT, not the moment of
    certification.

    :meth:`RuleMiner._certify_proposals` records its verdicts BEFORE the symbolic gate,
    the covered-prune and the sort promotion have run, and every one of those stages may
    still drop a certified pair -- soundly. Left alone, the sidecar advertises rules the
    artifact does not contain: measured on the published ``acj-4-3-llm`` asset, 19 of the
    73 pairs it calls ``certified`` are absent from ``rules.json``.

    A dropped pair keeps its certification history (``stage``/``certificate``: it WAS
    certified) and gains the TERMINAL verdict ``certified_then_dropped`` naming the stage
    that removed it and that stage's own reason, so the drop is auditable rather than
    invisible. A certified pair that is neither present nor attributable is a hole in the
    provenance chain and raises: the sidecar must never be silently wrong.
    """
    present = {_rule_identity(lhs, rhs, dummy_variables) for lhs, rhs in rules}
    for identity, index in watch.items():
        if identity in present:
            continue
        record = drops.get(identity)
        if record is None:
            entry = trail[index]
            raise RuntimeError(
                f"certified proposal {entry['source']} -> {entry['target']} is absent from "
                f"the mined ruleset and no stage recorded dropping it -- the provenance "
                f"sidecar cannot account for one of its own certified pairs")
        stage, reason = record
        trail[index].update(verdict='certified_then_dropped',
                            dropped_at=stage, drop_reason=reason)
        outcomes['certified'] -= 1
        outcomes['certified_then_dropped'] += 1


def _tokens_in_vocabulary(tokens: Any, vocabulary: set) -> bool:
    """Vocabulary test for proposal/hint token sequences: numeric and constant-like
    literals are valid leaves everywhere (the generation-2 spellings carry their
    numbers as literal TOKENS -- `pow x0 2` where the retired ring spelled
    `pow2 x0` -- so a bare vocabulary-set test would silently skip every such
    proposal)."""
    return all(t in vocabulary or t in _CONSTLIKE_LEAVES or is_numeric_string(t)
               for t in tokens)


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


#: ONE MINE PER FILE: each file is mined in its own mode -- the mode's serve answer is the
#: mark, its canon reads the sources, its numerics accept -- in this order, so the
#: permissive mine reads the finished f64 set as serving does. (rule mode, file key)
_MINE_MODES = (('default', 'f64'), ('real', 'real'), ('permissive', 'permissive'))

#: The mode `certify_rules` judges a proposal in: the union of the three files' acceptance.
_CERTIFY_RULE_MODE = 'permissive'

#: The engine attribute holding each rule mode's own set.
_RULES_ATTR = {'default': 'simplification_rules', 'real': 'real_simplification_rules',
               'permissive': 'permissive_simplification_rules'}

#: The judge loop gives up on ONE SOURCE -- loudly -- when the judge has refused its pick this
#: many times. Each refusal strictly grows a finite exclusion set, so the loop terminates; a
#: cap that binds says the scan and the judge disagree about that source, and that must be
#: seen, not papered over. The cap is per source, so one stubborn source neither costs the
#: others their rounds nor hides behind them.
_JUDGE_ROUNDS_MAX = 8

#: Batches below this size are judged in-process: spawning workers costs more than it saves.
_JUDGE_POOL_MIN = 64


#: The mine's judge watchdog (`find_rules`' `judge_warn_s`/`judge_fail_s`): a judgment past
#: the first is logged, one past the second fails the mine.
JUDGE_WARN_S = 60.0
JUDGE_FAIL_S = 600.0


class _JudgmentOverrun(BaseException):
    """Raised in the judging thread by the watchdog (`_watched_judgments`). A BaseException,
    so the sweep's per-rule ``except Exception`` cannot file it as a verdict."""


def _watched_judgments(pairs: list, warn_s: float = JUDGE_WARN_S,
                       fail_s: float = JUDGE_FAIL_S) -> list[dict]:
    """The contract judge's verdict record of each ``(lhs, rhs)`` pair, in order, judged one at
    a time with NO wall-clock timeout: a timeout would file a verdict the machine's load
    decided, and the mine memoises verdicts into its checkpoint. A WATCHDOG instead logs a
    judgment still running after ``warn_s`` seconds and fails the mine (``RuntimeError``,
    naming the rule) when one runs past ``fail_s``: on the main thread the judgment is
    interrupted there, elsewhere the mine fails as soon as it returns."""
    import _thread
    import contextlib
    import io
    import logging
    from .verify import verify_ruleset
    log = logging.getLogger(__name__)
    on_main = threading.current_thread() is threading.main_thread()

    def _overrun(signum: int, frame: Any) -> None:
        raise _JudgmentOverrun

    previous = signal.signal(signal.SIGUSR1, _overrun) if on_main else None
    out: list[dict] = []
    try:
        for idx, (lhs, rhs) in enumerate(pairs):
            rule = f"{' '.join(lhs)} -> {' '.join(rhs)}"
            done = threading.Event()
            fired = threading.Event()

            def _watch(rule: str = rule, done: threading.Event = done,
                       fired: threading.Event = fired) -> None:
                if done.wait(warn_s):
                    return
                log.warning('simplipy mine: the judgment of %s is still running after %g s',
                            rule, warn_s)
                if done.wait(max(fail_s - warn_s, 0.0)):
                    return
                fired.set()
                if on_main:
                    _thread.interrupt_main(signal.SIGUSR1)

            watchdog = threading.Thread(target=_watch, daemon=True)
            started = time.monotonic()
            try:
                watchdog.start()
                try:
                    # The sweep prints its CLI report; the mine's record is the sidecar.
                    with contextlib.redirect_stdout(io.StringIO()):
                        report = verify_ruleset([[list(lhs), list(rhs)]], judge_timeout_s=0)
                finally:
                    done.set()
                    watchdog.join()
                if fired.is_set():
                    raise _JudgmentOverrun
            except _JudgmentOverrun:
                raise RuntimeError(
                    f'the judgment of {rule} ran past {fail_s:g} s (judge_fail_s); the mine '
                    f'fails rather than file a verdict the clock decided') from None
            elapsed = time.monotonic() - started
            if elapsed > warn_s:
                log.warning('simplipy mine: the judgment of %s took %.1f s', rule, elapsed)
            detail = dict(report['detail'][0], idx=idx)
            out.append(detail)
    finally:
        if on_main:
            signal.signal(signal.SIGUSR1, previous)
    return out


def _judge_chunk(pairs: list, warn_s: float = JUDGE_WARN_S,
                 fail_s: float = JUDGE_FAIL_S) -> list[str]:
    from .verify import effective_tier
    return [effective_tier(d) for d in _watched_judgments(pairs, warn_s, fail_s)]


def judge_tiers(pairs: list, processes: int = 4, warn_s: float = JUDGE_WARN_S,
                fail_s: float = JUDGE_FAIL_S) -> list[str]:
    """THE JUDGE of the mine's per-length loop: the contract judge's effective tier of each
    ``(lhs, rhs)`` pair, in order, under the mine's watchdog (``warn_s``, ``fail_s``; see
    :func:`_watched_judgments`). Rules are judged independently, so a pool of
    ``processes`` workers (spawned: the parent holds the core's threads) gives the same
    answers as one; a small batch is judged in-process."""
    pairs = [(list(lhs), list(rhs)) for lhs, rhs in pairs]
    if processes <= 1 or len(pairs) < _JUDGE_POOL_MIN:
        return _judge_chunk(pairs, warn_s, fail_s) if pairs else []
    import multiprocessing
    size = -(-len(pairs) // processes)
    chunks = [pairs[i:i + size] for i in range(0, len(pairs), size)]
    with multiprocessing.get_context('spawn').Pool(len(chunks)) as pool:
        parts = pool.map(functools.partial(_judge_chunk, warn_s=warn_s, fail_s=fail_s), chunks)
    return [tier for part in parts for tier in part]


def _file_mode(rule_mode: str) -> str:
    return dict(_MINE_MODES)[rule_mode]


def _mode_tiers(rule_mode: str) -> frozenset:
    """The judged tiers a rule mode's file may hold (`verify.MODE_TIERS`, the one table)."""
    from .verify import MODE_TIERS
    return MODE_TIERS[_file_mode(rule_mode)]


def _twin_shadow_owners(census: list) -> set[int]:
    """The rows TWIN-SHADOW REMOVAL drops, read off a shadow census: the owner of each
    ``(victim, owner, False, 'twin-shadow')`` entry whose victim stays -- the victim's own
    orientation twin carries the owner's canonical rewrite, so with the owner gone it mints
    in its place and the served set is unchanged. Lowest rows first, so of a mutual pair
    the earlier row stays."""
    gone: set[int] = set()
    for victim, owner, owner_was_twin, kind in sorted(census):
        if kind == 'twin-shadow' and not owner_was_twin and victim != owner and victim not in gone:
            gone.add(owner)
    return gone


#: `SIMPLIPY_*` variables that change no artifact: the diagnostics (`*_TRACE`), the
#: test-asset switch, the build stamp, the progress cadence and the trusted-module list.
_INERT_ENV = ('SIMPLIPY_TEST_REQUIRE_ASSETS', 'SIMPLIPY_BUILD', 'SIMPLIPY_MINE_PROGRESS_INTERVAL',
              'SIMPLIPY_TRUSTED_MODULES')


def env_overrides(environ: Any = None) -> dict[str, str]:
    """Every ``SIMPLIPY_*`` variable set in ``environ`` (default: this process) that is not
    inert: the declared switches (:data:`ARTIFACT_ENV_SWITCHES`, empty since 0.15) and any
    other, so a switch a later build reads, or a stale one, is recorded rather than assumed
    harmless. A mine records it in its sidecar; ``simplipy verify-mine`` requires it empty."""
    environ = os.environ if environ is None else environ
    return {key: value for key, value in sorted(environ.items())
            if key.startswith('SIMPLIPY_') and not key.endswith('_TRACE')
            and key not in _INERT_ENV}


def code_digest() -> dict[str, str]:
    """The code a mine runs: the sha256 of the compiled core and of the package's Python
    sources. A checkpoint resumes only under the digest it was written with."""
    import simplipy
    from simplipy import _core
    root = os.path.dirname(os.path.abspath(simplipy.__file__))
    python = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith('.py'):
                path = os.path.join(dirpath, name)
                python.update(os.path.relpath(path, root).encode())
                with open(path, 'rb') as file:
                    python.update(file.read())
    with open(_core.__file__, 'rb') as file:
        core = hashlib.sha256(file.read()).hexdigest()
    return {'core': core, 'python': python.hexdigest()}


def _near_miss_frac() -> float:
    """The high-precision re-judgement's near-miss gate, as the core declares it."""
    from simplipy import _core
    return float(_core.NEAR_MISS_FRAC)


def _judge_digest() -> str:
    """The sha256 of the contract judge's source (`verify/_contract.py`)."""
    from .verify import _contract
    with open(_contract.__file__, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


#: Slot sigils. A pattern carrying one matches many instances, and the constructor may
#: reach the endpoint on the PATTERN while refusing it on some of them.
_SLOT_SIGILS = ('_', '?', '!', '$', '<')


def _triple_paths(out_path: str) -> dict[str, str]:
    """The three files a mine writes, derived from the requested output path.

    The f64 set takes the path exactly as passed. The siblings replace the mode
    marker rather than append to it: a trailing ``_f64`` on the stem is stripped
    first, so ``rules_f64.json`` (the naming convention from 0.14.0 on) yields
    ``rules_real.json`` / ``rules_permissive.json``. Markerless names get the same
    siblings: reproducing a pre-rename artifact byte for byte pins the pre-rename
    simplipy anyway, so a historic-filename carve-out here would protect nothing.
    """
    base, ext = os.path.splitext(out_path)
    if base.endswith('_f64'):
        base = base[:-len('_f64')]
    return {'f64': out_path, 'real': f'{base}_real{ext}', 'permissive': f'{base}_permissive{ext}'}


def _is_ground(tokens: Any) -> bool:
    """No slots and no `<constant>`: the pattern IS its only instance."""
    return not any(str(t).startswith(_SLOT_SIGILS) for t in tokens)


def _preempted_by(engine: Any, lhs: Any, rhs: Any, mode: str) -> bool:
    """THE PREEMPTION GATE: does `mode`'s CONSTRUCTOR already reach this rule's endpoint
    without the rule? `_finalize` drops such a rule from the mode's file and lists it in the
    sidecar (``gates.preempted``); the expected count is 0, because the miner's mark is the
    mode's serve answer, which contains the constructor.

    Asked with the engine's own simplifier in that mode, against a RULE-LESS twin, so the
    answer is "the constructor does this" and never "some other rule does this" -- the
    second would make the prune order-dependent, dropping whichever of two mutually
    covering rules happened to be tested first.

    Compared on SIMPLIFIED sides, not on tokens. `1e-08` and `0.00000001` are the same
    number, and an earlier version of this comparison read them as different and reported
    17 preempted where the truth was 29.
    """
    # Cached ON THE ENGINE, not in a dict keyed by `id()`. CPython reuses an id once the
    # object behind it is collected, so an id-keyed cache can hand a fresh engine a twin
    # built from a DIFFERENT operator table -- which is exactly what happened, and the
    # prune then silently answered "not preempted" for everything.
    # GROUND RULES ONLY, and this is a correctness bound rather than caution. Asking the
    # constructor about a PATTERN answers a question about the pattern, not about the
    # instances the rule matches, and the two genuinely differ: in permissive the constructor
    # takes `/ $0 $0` to `1` -- the slot carries a nonzero-a.e. certificate -- while
    # taking the instance `inf / inf` to `nan`. Pruning on the pattern therefore deleted
    # `/ $0 $0 -> 1` and with it the mask-sentinel doctrine, which is exactly the rule
    # that makes `inf/inf * x0` collapse to `x0` in permissive.
    #
    # For a GROUND rule the pattern IS its only instance, so the question the constructor
    # answers is the question asked. That still retires what the prune was for: the
    # 18-rule `f(+-10^-k) -> +-10^-k` family is ground, and so is every literal
    # evaluation folding now performs.
    if not (_is_ground(lhs) and _is_ground(rhs)):
        return False
    bare = getattr(engine, '_ruleless_twin', None)
    if bare is None:
        from .engine import SimpliPyEngine
        bare = SimpliPyEngine(operators=engine._operators_config, rules=[])
        # `real` fails closed without a set of its own; an EMPTY one says "this mode
        # serves no rules", which is exactly the constructor-only engine we want.
        bare._core.set_mode_rules('real', [])
        engine._ruleless_twin = bare
    # No error is read as "not preempted": a ground rule side the engine cannot read is a
    # defect of the rule set, and the gate reports it by raising instead of keeping the rule.
    return tuple(bare.simplify(list(lhs), mode=mode)) == tuple(bare.simplify(list(rhs), mode=mode))


class RuleMiner:
    """The rule miner of one engine: :meth:`find_rules` and :meth:`certify_rules`.

    Stateless beyond the engine reference: mining reads the engine's compiled core and
    operator table, swaps rule sets only through the engine's copy-on-write rule
    replacement, and calls back into the engine's own ``prune_covered_rules``,
    ``is_valid`` and fingerprint methods. One mine runs per process: a module-level
    lock serializes every mine and certify entry point, however many miners are
    constructed.
    """

    def __init__(self, engine: 'SimpliPyEngine') -> None:
        self.engine = engine

    def _mode_rules(self, rule_mode: str) -> list:
        """The rules the engine holds for ``rule_mode`` (its own set)."""
        return [(tuple(lhs), tuple(rhs)) for lhs, rhs in
                (getattr(self.engine, _RULES_ATTR[rule_mode]) or [])]

    def _install(self, rule_mode: str, rules: list) -> None:
        """Install ``rules`` as ``rule_mode``'s own set, keeping the other modes' sets. The
        mine reads and judges the set in ``rule_mode``'s canon, so it is read in that one."""
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
        self.engine._rule_canons.pop(rule_mode, None)
        if rule_mode == 'default':
            self.engine._replace_rules(rules)
        else:
            setattr(self.engine, _RULES_ATTR[rule_mode], rules)
            self.engine._replace_rules(self.engine.simplification_rules)

    def _unlicensed(self, picks: dict, rule_mode: str, judge: Callable, judged: dict,
                    dummy_variables: list[str]) -> set:
        """The picks the mode may not ship: a pick whose pattern (the spelling the file will
        carry) or one of whose served entries (the rule as the mode loads it, orientation
        twins included) carries a judged tier the mode does not license, or that the mode's
        load does not serve at all. Verdicts are memoised by pair across rounds and lengths
        (``judged``)."""
        order = sorted(picks)
        patterns = []
        for idx in order:
            src, tgt = picks[idx]
            p_src, mapping = remap_expression(list(src), dummy_variables=dummy_variables,
                                              variable_prefix='?')
            p_tgt, _ = remap_expression(list(tgt), dummy_variables, mapping, variable_prefix='?')
            patterns.append((tuple(p_src), tuple(p_tgt)))
        served = self.engine._core.ac_translate_preview(
            [(list(lhs), list(rhs)) for lhs, rhs in patterns], rule_mode=rule_mode)
        todo = list(dict.fromkeys(
            [pair for pair in patterns if pair not in judged]
            + [(tuple(lhs), tuple(rhs)) for lhs, rhs, _ in served
               if (tuple(lhs), tuple(rhs)) not in judged]))
        if todo:
            for pair, tier in zip(todo, judge(todo)):
                judged[pair] = tier
        allowed = _mode_tiers(rule_mode)
        bad = {k for k, pair in enumerate(patterns) if judged[pair] not in allowed}
        bad |= set(range(len(order))) - {k for _, _, k in served}
        bad |= {k for lhs, rhs, k in served if judged[(tuple(lhs), tuple(rhs))] not in allowed}
        return {order[k] for k in bad}

    def _final_gate(self, rule_mode: str, judge: Callable, judged: dict) -> tuple[list, list]:
        """THE FINAL PER-FILE GATE over the mode's installed set: every file
        spelling AND every entry the mode serves from it is judged (``judged`` memoises by
        pair), and a row with any unlicensed spelling is dropped. The file spelling is what
        release verification reads; the served entries are what fires.
        -> (kept rules, [[lhs, rhs, tier] per dropped row])."""
        final = self._mode_rules(rule_mode)
        served = self.engine._core.ac_served_rules_in_mode(rule_mode=rule_mode)
        todo = list(dict.fromkeys(final + [(tuple(lhs), tuple(rhs)) for lhs, rhs, _ in served]))
        new = [pair for pair in todo if pair not in judged]
        for pair, tier in zip(new, judge(new) if new else []):
            judged[pair] = tier
        allowed = _mode_tiers(rule_mode)
        off = {i: judged[pair] for i, pair in enumerate(final) if judged[pair] not in allowed}
        for lhs, rhs, i in served:
            if judged[(tuple(lhs), tuple(rhs))] not in allowed:
                off.setdefault(i, judged[(tuple(lhs), tuple(rhs))])
        return ([rule for i, rule in enumerate(final) if i not in off],
                [[list(final[i][0]), list(final[i][1]), tier] for i, tier in sorted(off.items())])

    def _deshadow(self, rule_mode: str, rules: list) -> tuple[list, list]:
        """TWIN-SHADOW REMOVAL: drop every row the mode's load would mint
        anyway. A census entry ``(victim, owner, False, 'twin-shadow')`` says the victim's
        orientation twin was skipped because the owner's own row carries the identical
        canonical rewrite -- so with the owner removed the twin mints in its place and the
        served set is unchanged. The owner is dropped while its victim stays (lowest rows
        first, so of a mutual pair the earlier row is kept), and the census is re-read
        until no such entry remains. -> (kept rules, dropped rules)."""
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
        dropped: list = []
        while True:
            self.engine._core.set_mode_rules(rule_mode, [(list(lhs), list(rhs))
                                                         for lhs, rhs in rules])
            gone = _twin_shadow_owners(self.engine._core.ac_shadow_census(rule_mode=rule_mode))
            if not gone:
                return rules, dropped
            dropped += [rules[i] for i in sorted(gone)]
            rules = [rule for i, rule in enumerate(rules) if i not in gone]

    def _mine_length(self, sources: list, mine: Callable, rule_mode: str, judge: Callable,
                     judged: dict, dummy_variables: list[str]) -> tuple[list, dict]:
        """THE PER-LENGTH JUDGE LOOP: mine the length, judge the picks, and
        re-scan exactly the sources whose pick the mode does not license, with that pick
        excluded, until every pick is licensed. The scan is deterministic and exclusions only
        remove refused targets, so the fixed point is the scan that consults the judge per
        candidate: another tier's pick can never shadow a licensed alternative.

        ``mine(sources, indices, exclusions)`` runs `mine_one_length` with the global source
        indices (which seed the sources and key the exclusions).

        A source whose pick the judge refuses ``_JUDGE_ROUNDS_MAX`` times is given up on --
        it is not re-scanned again -- and listed in ``stats['stubborn']`` with the targets it
        excluded; the loop runs on for the others. The caller fails the mine on it."""
        picks = {idx: (list(src), list(tgt)) for idx, src, tgt in mine(sources, None, None)}
        exclusions: dict[int, list] = {}
        refusals: dict[int, int] = {}
        stubborn: list[int] = []
        rounds = 0
        while True:
            rounds += 1
            bad = self._unlicensed(picks, rule_mode, judge, judged, dummy_variables)
            again: list[int] = []
            for idx in sorted(bad):
                exclusions.setdefault(idx, []).append(picks.pop(idx)[1])
                refusals[idx] = refusals.get(idx, 0) + 1
                (stubborn if refusals[idx] >= _JUDGE_ROUNDS_MAX else again).append(idx)
            if not again:
                break
            for idx, src, tgt in mine([sources[i] for i in again], again, exclusions):
                picks[idx] = (list(src), list(tgt))
        stats = {'judge_rounds': rounds,
                 'excluded': sorted([list(sources[i]), tgt] for i, ts in exclusions.items()
                                    for tgt in ts),
                 'licensed': len(picks),
                 'stubborn': [[list(sources[i]), exclusions[i]] for i in sorted(stubborn)]}
        return [picks[i] for i in sorted(picks)], stats

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

    @staticmethod
    def _confirm_kwargs(X_confirm: np.ndarray | None, X_data: np.ndarray, min_informative: int,
                        seed: int) -> dict[str, Any]:
        """Stage 2's arguments for the core's scan entries: the independent, wider X every
        numeric pick is re-certified on before it may mint, with the evidence floor scaled
        to its row count. Empty when the mine runs with ``confirm=False``.

        Confirmation runs INSIDE the scan (``Acceptor::confirm``): an unconfirmed pick yields
        to the next candidate. Its seed is order-free -- the core adds a hash of both sides
        of the pair to ``seed`` -- so a pair's verdict never depends on what else was judged.
        """
        if X_confirm is None:
            return {}
        return {'confirm_x': X_confirm.flatten(order='C').tolist(),
                'confirm_rows': X_confirm.shape[0],
                'confirm_min_informative': max(1, round(min_informative * X_confirm.shape[0]
                                                        / X_data.shape[0])),
                'confirm_seed': seed}

    def _certify_one(self, source: tuple[str, ...], hint: Any, library: Any, vocabulary: set,
                     max_target_pattern_length: int | None, fit: dict[str, Any], seed: int,
                     confirm: dict[str, Any],
                     rule_mode: str,
                     excluded: Any = ()) -> tuple[list[str] | None, str | None, str | None]:
        """THE certification chain for one proposed source, shared by :meth:`certify_rules`
        and the proposal channel of :meth:`find_rules`.

        First the library scan, with the engine's own result as the mark to beat; only when
        it finds nothing, the hint, judged by the SAME acceptance (``judge_hint``: the
        licences against the mark, the pair judgment on the mine's X, then stage 2). Both
        run stage-2 confirmation inside (``confirm``). Returns ``(target, certificate,
        refusal)``: certificate ``'minimal'`` (scan) or ``'verified'`` (hint); with no
        target, ``refusal`` names why the hint was refused (None without a hint).

        NO TOKEN-LENGTH BAR on a hint: a proposed rule need not shorten the token count.
        Descent is decided by the serve ordering; the predicate G7 and the serve-time
        `oriented` check apply.

        ``excluded``: the targets the judge loop refused for this source; both arms skip
        their class (a hint of an excluded class is refused as ``'excluded'``).
        """
        exclusions = [list(t) for t in excluded]
        core = self.engine._core
        assert core is not None
        target = core.find_rule_lib(list(source), library, max_target_pattern_length, seed=seed,
                                    rule_mode=rule_mode, exclusions=exclusions, **fit, **confirm)
        if target is not None:
            return list(target), 'minimal', None
        if hint is None:
            return None, None, None
        hint = [str(t) for t in hint]
        if not _tokens_in_vocabulary(hint, vocabulary) or not self.engine.is_valid(hint):
            return None, None, 'vocabulary'
        ok, why = core.judge_hint(list(source), hint, library, seed=seed,
                                  rule_mode=rule_mode, exclusions=exclusions, **fit, **confirm)
        return (hint, 'verified', None) if ok else (None, None, why)

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
            done, total = self.engine._core.mining_progress()
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
            relaxed_kruskal: bool = True,
            provenance: dict | None = None,
            proposal_entries: list[tuple[tuple[str, ...], tuple[str, ...] | None]] | None = None,
            leaf_nodes: list[str] | None = None,
            promote_sorts: bool = True,
            symbolic_gate: bool = True,
            snapshot_at: dict[int, str] | None = None,
            rule_mode: str = 'default',
            initial_rules: list | None = None,
            finals: dict[str, list] | None = None,
            judge: Callable | None = None,
            judge_warn_s: float = JUDGE_WARN_S,
            judge_fail_s: float = JUDGE_FAIL_S) -> None:
        """Phase 2 of :meth:`find_rules` on the compiled Rust core (``simplipy._core``), for
        ONE file: ``rule_mode`` is the mode the file serves, and its mine runs in that mode
        throughout (marks, sources, library, acceptance, the rules installed so far). The
        file's rules are written and recorded in ``finals[rule_mode]``.

        Mirrors the pure-Python worker pool, but correctly against the core: per source
        length (shortest first), each source is Kruskal-pruned against the rules found so
        far and searched for an equivalent shorter candidate; the growing rule set is
        pushed into the core between lengths (``set_mode_rules``), which the Python pool cannot
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
        assert self.engine._core is not None
        if judge is None:
            judge = functools.partial(judge_tiers, warn_s=judge_warn_s, fail_s=judge_fail_s)

        # THE TARGET BOUND IS mu, NOT TOKENS (owner ruling, re-mine 2026-08-20).
        # `max_target_pattern_length=None` now admits EVERY enumerated expression to the
        # candidate library and lets the scan stop at the first mu tier that cannot descend
        # below the source's mark -- the same predicate acceptance already applied, so the
        # split between a token bound for search and mu for acceptance is gone.
        #
        # This is a WIDENING, and deliberately so: a longer target can be strictly cheaper
        # (`1/5` -> `0.2`, `pow(inf,x)` -> `exp(inf*x)`), and `max_source_pattern_length - 1`
        # refused exactly those without ever pricing them. An explicit integer still caps the
        # library, for callers that want a small one; it is no longer the criterion.
        if max_target_pattern_length is None:
            max_candidate_length = max_source_pattern_length
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
        library = self.engine._core.build_candidate_library(
            candidates, list(dummy_variables), X_data.flatten(order='C').tolist(), X_data.shape[0],
            rule_mode=rule_mode)
        file_mode = dict(_MINE_MODES)[rule_mode]
        out_file = None if output_file is None else _triple_paths(output_file)[file_mode]
        if verbose:
            print(f'=== mining the {file_mode} file (rule mode {rule_mode})')
        if verbose:
            print(f'Candidate library: {library.n_candidates:,} candidates'
                  f' ({library.n_filtered:,} var-free filtered)')

        # The mode's own existing rules (empty after reset_rules=True) join the Kruskal pruning
        # and the final deduplicated set; another mode's set never seeds this one.
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in (
            initial_rules if initial_rules is not None else self._mode_rules(rule_mode))]

        # --- The post-pass: what turns a raw mined rule set into a CELL ARTIFACT ---
        # A closure, because the ladder runs it MORE THAN ONCE. A (7,4) climb passes through
        # the (5,4) and (6,4) cells on its way up and each of those is a publishable artifact
        # in its own right (see `snapshot_at`). Running the IDENTICAL block for every cell is
        # what makes a snapshot equal to a one-shot mine of that cell; a second copy of these
        # stages would be free to drift from this one.
        def _finalize(mined: list, out_path: str | None, prov: dict | None) -> list:
            self._install(rule_mode, list(mined))
            # --- F94 census reconciliation state -------------------------------------
            # The proposal census is written at CERTIFICATION time, upstream of four
            # stages that may each still drop a certified pair (symbolic gate, both
            # covered-prunes, sort promotion). `_watch` follows every certified pair
            # through them and `_drops` records WHICH stage took it and WHY, so the
            # sidecar can carry a terminal outcome instead of a stale claim.
            _outcomes: dict = {}
            _trail: list = []
            _watch: dict = {}
            _drops: dict = {}

            def _note_drops(stage: str, reason_of: Callable[[Any], str]) -> None:
                """Attribute every watched pair that left the ruleset during `stage`."""
                if not _watch:
                    return
                alive = {_rule_identity(lhs, rhs, dummy_variables)
                         for lhs, rhs in self._mode_rules(rule_mode)}
                for identity in _watch:
                    if identity not in _drops and identity not in alive:
                        _drops[identity] = (stage, reason_of(identity))
            # Proposal channel: certify externally proposed rules against the just-mined
            # state, BEFORE the optional prune. Skipped on interrupt: a partial mine is not
            # the rule state the proposals were aimed at, and a clean abort must stay cheap.
            # An explicitly-given EMPTY batch still runs (and records all-zero outcome
            # counts): the sidecar must show the channel ran, not silently omit it.
            if proposal_entries is not None and not interrupted():
                outcomes, trail, judge_loop = self._certify_proposals(
                    proposal_entries, library,
                    leaf_nodes if leaf_nodes is not None else list(dummy_variables),
                    dummy_variables, X_data, X_confirm, max_target_pattern_length,
                    constants_fit_challenges, constants_fit_retries, rtol, atol,
                    min_informative, mine_seed, confirm_seed, verbose, rule_mode,
                    judge, judged)
                if prov is not None and 'proposals' in prov:
                    prov['proposals']['outcomes'] = outcomes
                    prov['proposals']['judge_loop'] = judge_loop
                    # The per-candidate verdict trail travels WITH the artifact: an aggregate
                    # tally cannot be audited (it never says which candidate died at which gate).
                    prov['proposals']['trail'] = trail
                # Arm the F94 watch on the pairs the census currently calls certified.
                # `deduplicate_rules` has already folded the batch in, so a pair missing
                # from the ruleset ALREADY lost the merge -- attributed here, not left to
                # the fail-closed hole check at the end.
                _outcomes, _trail = outcomes, trail
                _watch = {_rule_identity(e['source'], e['target'], dummy_variables): i
                          for i, e in enumerate(_trail) if e['verdict'] == 'certified'}
                _note_drops('merge', lambda _identity: 'folded-by-deduplicate')
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
            if symbolic_gate and not interrupted() and self._mode_rules(rule_mode):
                from .verify import CONST_CHANNEL_DETAIL, FATAL_BUCKETS
                # JUDGE WHAT SERVES, NOT WHAT THE ARTIFACT SAYS. The gate used to judge
                # `simplification_rules` -- the artifact -- but translation is not the
                # identity on it: rules drop, and ORIENTATION TWINS are MINTED at load
                # (`AcRules::translate`), from the LOADING engine's canon. Those twins fire
                # and had never been judged by any authority, because they do not exist
                # until after the artifact is read. `ac_served_rules` is the matcher's own
                # rule set, so coverage is 100% of what can fire -- and it stays 100% if a
                # later engine mints a different closure, which is exactly what an artifact
                # cannot promise. Judging the served set also judges the DEPLOYED spelling
                # (the canonical projection) rather than the mine's.
                _served = self.engine._core.ac_served_rules_in_mode(rule_mode=rule_mode)
                # Under the mine's watchdog, with no wall-clock timeout (`_watched_judgments`).
                _detail = _watched_judgments([(lhs, rhs) for lhs, rhs, _ in _served],
                                             judge_warn_s, judge_fail_s)
                _buckets: dict = {b: [] for b in ('CERTIFIED', 'TOLERATED', *FATAL_BUCKETS)}
                for d in _detail:
                    _buckets.setdefault(d['verdict'], []).append(d['idx'])
                # EVERY fatal bucket drops, not only KILL (audit B7): a rule the judge
                # cannot evaluate (NO-WITNESS / UNSUPPORTED-SHAPE / JUDGE-TIMEOUT) or
                # cannot reconcile (ENGINE-MISALIGN / UNRESOLVED-COVERAGE) never passed
                # the second authority, and "confirmed by two independent authorities"
                # is the claim a shipped rule makes. Fail closed: a vocabulary the
                # contract cannot judge yields no rules, visibly, rather than rules
                # with an unearned certificate. ONE exemption (D36): the judge's own
                # non-jurisdiction sentinel for the multi-<constant> family, whose
                # soundness authority is the Const-channel chain (fitting + count gate
                # + licence registry) -- exempted rules ship and are recorded under
                # their own sidecar key, so the carve-out is visible, never silent.
                # A verdict is attributed through the served entry's PROVENANCE, not
                # through its spelling: the served side is canonical and the artifact side
                # is the mine's, so `d['lhs'].split()` no longer names a droppable row
                # (`/ cosh <constant> 0` serves as `* float("inf") cosh <constant>`). A
                # twin has no row of its own at all and reports its SOURCE, so condemning
                # one drops the rule that mints it -- fail-closed, and the only action that
                # removes a twin from a later load.
                _fatal: dict = {b: set() for b in FATAL_BUCKETS}
                _const_channel: set = set()
                _routed: dict = {}
                for d in _detail:
                    if d.get('verdict') not in _fatal:
                        continue
                    _src_lhs, _src_rhs = self._mode_rules(rule_mode)[
                        _served[d['idx']][2]]
                    _pair = (tuple(_src_lhs), tuple(_src_rhs))
                    _tier_of = d.get('tier') or 'reject'
                    if d.get('detail') == CONST_CHANNEL_DETAIL:
                        _const_channel.add(_pair)
                    elif _tier_of in _mode_tiers(rule_mode):
                        # THE MODE LICENSES THE TIER, so the gate must not delete it:
                        # a fatal bucket is fatal only where its tier is not licensed --
                        # ENGINE-MISALIGN is `real` (true, the deployed engine diverges),
                        # an f64-realised KILL is `f64` (false, bit-exact in the deployed
                        # algebra) -- and this file keeps what its own mode licenses.
                        # Fail-closed is UNCHANGED for everything else: NO-WITNESS,
                        # UNSUPPORTED-SHAPE, JUDGE-TIMEOUT and UNRESOLVED-COVERAGE carry
                        # no tier at all, so they read `reject` and still drop.
                        _routed.setdefault(_tier_of, set()).add(_pair)
                    else:
                        _fatal[d['verdict']].add(_pair)
                _drop = set().union(*_fatal.values())
                if _drop:
                    self._install(rule_mode, [
                        (lhs, rhs) for lhs, rhs in self._mode_rules(rule_mode)
                        if (tuple(lhs), tuple(rhs)) not in _drop])
                    # F94: the gate's own bucket IS the drop reason -- a certified
                    # proposal killed here reads `certified_then_dropped(symbolic_gate,
                    # KILL)` in the sidecar rather than staying `certified`.
                    _gate_reason = {_rule_identity(bl, br, dummy_variables): bucket
                                    for bucket, pairs in _fatal.items() for bl, br in pairs}
                    _note_drops('symbolic_gate',
                                lambda identity: _gate_reason.get(identity, 'fatal-verdict'))
                if verbose:
                    print(f'Symbolic gate: {len(_fatal["KILL"])} killed, '
                          f'{len(_drop) - len(_fatal["KILL"])} unjudgeable, '
                          f'{sum(len(v) for v in _routed.values())} routed to a mode tier '
                          f'({", ".join(f"{k}={len(v)}" for k, v in sorted(_routed.items())) or "none"}), '
                          f'{len(self._mode_rules(rule_mode))} remain')
                if prov is not None:
                    # The full census travels with the artifact (B7): every fatal
                    # bucket has its own key, PRESENT EVEN WHEN EMPTY -- the sidecar
                    # must show what the gate saw, not only what it dropped.
                    prov['symbolic_gate'] = {
                        **{('killed' if b == 'KILL' else b.lower().replace('-', '_')):
                           sorted([list(bl), list(br)] for bl, br in _fatal[b])
                           for b in FATAL_BUCKETS},
                        'const_channel': sorted(
                            [list(cl), list(cr)] for cl, cr in _const_channel),
                        # Convicted by the contract but KEPT, because a mode tier owns
                        # them. Recorded so the carve-out is visible in the sidecar
                        # rather than looking like the gate simply missed them.
                        'routed_to_tier': {k: sorted([list(bl), list(br)] for bl, br in v)
                                           for k, v in sorted(_routed.items())},
                        'census': {b: len(ids) for b, ids in _buckets.items()},
                        'kept': len(self._mode_rules(rule_mode))}
            # THE FILE IS WHAT THIS MODE LICENSES (miner.md 2.11): the final gate judges the
            # file's own spellings AND the entries the mode serves from them, with the loop's
            # judge and memo, and drops what the mode does not license, counted. It judges
            # BEFORE EACH PRUNE, so a prune only ever leans on licensed rules (a rule pruned as
            # covered by one the gate then dropped would be lost), and once more at the end,
            # on what ships: the post-pass (promotion) respells and re-sorts, so the per-length
            # verdicts describe an earlier set.
            gate: dict = {'unlicensed': [], 'preempted': []}

            def _final_gate() -> None:
                if not (symbolic_gate and self._mode_rules(rule_mode) and not interrupted()):
                    return
                kept, off = self._final_gate(rule_mode, judge, judged)
                if off:
                    gate['unlicensed'] += off
                    self._install(rule_mode, kept)
                    _note_drops('final_gate', lambda _identity: 'unlicensed-or-preempted')
            if prune == 'covered':
                _final_gate()
                self.engine.prune_covered_rules(verbose=verbose, rule_mode=rule_mode)
                # F94: the prune removes only rules the REMAINING set already reaches,
                # so a certified proposal dropped here is still served compositionally
                # -- a harmless drop, but one the census must still name.
                _note_drops('prune_covered',
                            lambda _identity: 'behaviourally-covered-by-remaining-rules')
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
                kept, promotion_report = promote(self._mode_rules(rule_mode), self.engine,
                                                 verbose=verbose, rule_mode=rule_mode)
                self._install(rule_mode, kept)
                # F94: the ladder's own per-rule verdict lists name the reason a pair
                # left here (`killed_q` -- unsound at the `?` sort, `unsupported`, ...);
                # a pair the refund pruned as derivable appears in none of them.
                _promo_reason = {
                    _rule_identity(item[0], item[1], dummy_variables): bucket
                    for bucket, items in promotion_report.items()
                    if isinstance(items, list)
                    for item in items if len(item) >= 2}
                _note_drops('sort_promotion',
                            lambda identity: _promo_reason.get(identity, 'derivable-refund'))
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
                    _final_gate()
                    self.engine.prune_covered_rules(verbose=verbose, rule_mode=rule_mode)
                    _note_drops('prune_covered_post_promotion',
                                lambda _identity: 'behaviourally-covered-by-remaining-rules')
            # SOUNDNESS PROVENANCE, second half (audit Tier-1 #4): the interval layer's
            # three fail-closed miss counters, read AFTER the mine as deltas over this
            # mine's baseline -- the exposure ("how often could the gate not decide?")
            # is a recorded number, never assumed zero. The counters mark UNDECIDED
            # verdicts whose callers rejected (lost recall, not lost soundness).
            if prov is not None:
                prov['soundness']['interval_undecided'] = {
                    'horizon': self.engine._core.interval_horizon_misses() - _ivl_baseline[0],
                    'node_budget': self.engine._core.interval_node_budget_misses() - _ivl_baseline[1],
                    'unanalyzable': self.engine._core.interval_unanalyzable_misses() - _ivl_baseline[2],
                }
            # F94: the census must describe the ARTIFACT, not the moment of
            # certification. Runs before ANY of it is written (artifact and sidecar
            # alike) and fails closed on a certified pair that is neither present nor
            # attributable -- a census that can advertise an absent rule is exactly the
            # defect this reconciliation exists to make impossible.
            # The final gate on what ships (see `_final_gate` above), then the preemption
            # gate: a ground rule the mode's constructor already performs cannot fire;
            # expected 0, since the mark contains the constructor.
            _final_gate()
            final = self._mode_rules(rule_mode)
            _kept = []
            for lhs, rhs in final:
                if _preempted_by(self.engine, lhs, rhs, file_mode):
                    gate['preempted'].append([list(lhs), list(rhs)])
                else:
                    _kept.append((lhs, rhs))
            if gate['preempted']:
                final = _kept
                self._install(rule_mode, final)
                _note_drops('final_gate', lambda _identity: 'unlicensed-or-preempted')
            if _watch:
                _reconcile_proposal_census(
                    _outcomes, _trail, _watch, _drops, final, dummy_variables)
            if verbose:
                print(f'Final gate ({file_mode}): {len(gate["unlicensed"])} unlicensed, '
                      f'{len(gate["preempted"])} preempted, {len(final)} rules')
            if out_path is not None:
                _path = _triple_paths(out_path)[file_mode]
                with open(_path, 'w') as file:
                    json.dump([[list(lhs), list(rhs)] for lhs, rhs in final], file, indent=4)
                if prov is not None:
                    prov['mode'] = file_mode
                    # How this file's mine accepted a pick: its tolerances, whether a near
                    # miss is re-judged at high precision (every mode but f64) and the gate
                    # that admits it, the judged tiers the file may hold, and the judge.
                    _rescue = rule_mode != 'default'
                    prov['acceptance'] = {
                        'rtol': (prov.get('params') or {}).get('rtol'),
                        'atol': (prov.get('params') or {}).get('atol'),
                        'rescue': _rescue,
                        'near_miss_frac': _near_miss_frac() if _rescue else None,
                        'tiers': sorted(_mode_tiers(rule_mode)),
                        'judge_digest': _judge_digest(),
                    }
                    prov['gates'] = {'unlicensed': gate['unlicensed'],
                                     'preempted': gate['preempted'],
                                     'file': os.path.basename(_path)}
                # An interrupted mine still writes what it has, but it is NOT final: the
                # lengths after the interrupt were never mined and the proposal channel,
                # the gates and sort promotion above were skipped. The sidecar says so.
                _complete = not interrupted()
                self._write_provenance(
                    _path,
                    None if prov is None else dict(
                        prov, measure=self.engine._measure_fingerprint(file_mode)),
                    final, final=_complete, interrupted=not _complete)
                if _complete and out_path == output_file and os.path.exists(_path + '.checkpoint.json'):
                    os.remove(_path + '.checkpoint.json')
            # The run's own cell (never a snapshot), written to disk or not: the next mode
            # mines beside it and the engine ends holding it.
            if finals is not None and out_path == output_file:
                finals[rule_mode] = [(tuple(lhs), tuple(rhs)) for lhs, rhs in final]
            return [(tuple(lhs), tuple(rhs)) for lhs, rhs in final]

        snapshot_at = dict(snapshot_at or {})

        # SOUNDNESS PROVENANCE, first half: every set switch (`env_overrides`; the node
        # budget is a parameter). Recorded up front so interrupted mines carry it.
        _ivl_baseline = (self.engine._core.interval_horizon_misses(),
                         self.engine._core.interval_node_budget_misses(),
                         self.engine._core.interval_unanalyzable_misses())
        if provenance is not None:
            provenance['soundness'] = {
                # EVERY set non-inert SIMPLIPY_* variable, raw (H-042): a default run
                # records {}, any override is visible verbatim.
                'env_overrides': env_overrides(),
                'interval_undecided': None,  # filled at finalize (deltas over this mine)
                'ac_classes': None,  # filled per length as the climb proceeds
            }

        # Per-length AC-class census, recorded in the sidecar: how many enumerated
        # spellings collapsed to how many canonical classes. An artifact states what it
        # actually mined, and under representative-only certification that is CLASSES.
        ac_classes_per_length: dict[int, dict[str, int]] = {}

        # Sources: lengths up to max_source_pattern_length only (`construct_expressions`
        # over-produces a tail of longer expressions beyond the documented contract).
        # RESUME FROM DISK FIRST: a checkpoint of THIS mine (same parameters, same mode)
        # restores the licensed rules, the per-length record and the judge's memo, and the
        # climb continues after the last completed length.
        per_length: dict[int, dict] = {}
        judged: dict = {}
        # Everything the length loop's state is a function of: the parameters, the mode, the
        # code, the operator table, the evaluation data (its content, explicit or seeded),
        # the numeric environment that evaluates it, the universe and the starting sets.
        _environment = {k: v for k, v in ((provenance or {}).get('environment') or {}).items()
                        if k != 'platform'}
        checkpoint_fp = hashlib.sha256(json.dumps(
            {'params': (provenance or {}).get('params'), 'rule_mode': rule_mode,
             'code': code_digest(),
             'operators': self.engine._operators_config,
             'X': hashlib.sha256(np.ascontiguousarray(X_data).tobytes()).hexdigest(),
             'X_confirm': (None if X_confirm is None else hashlib.sha256(
                 np.ascontiguousarray(X_confirm).tobytes()).hexdigest()),
             'environment': _environment,
             'env_overrides': env_overrides(),
             'universe': {str(k): len(v) for k, v in sorted(expressions_of_length.items())},
             'initial': [[list(lhs), list(rhs)] for lhs, rhs in rules],
             'other_sets': [getattr(self.engine, _RULES_ATTR[m]) for m, _ in _MINE_MODES
                            if m != rule_mode]},
            sort_keys=True, default=str).encode()).hexdigest()
        resumed_at = 0
        if out_file is not None and os.path.exists(out_file + '.checkpoint.json'):
            with open(out_file + '.checkpoint.json') as file:
                ckpt = json.load(file)
            if ckpt.get('fingerprint') != checkpoint_fp:
                # A checkpoint of ANOTHER mine -- other parameters, mode, operators, data,
                # numeric environment, universe, starting sets, or code (the compiled core or
                # the package's sources) -- is never resumed and never silently overwritten.
                raise RuntimeError(
                    f"'{out_file}.checkpoint.json' was written by a different mine (its "
                    f"parameters, mode, operators, data, numeric environment, universe, "
                    f"starting rules or code differ); remove it to mine from the start")
            else:
                resumed_at = ckpt['completed_length']
                rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in ckpt['rules']]
                per_length = {int(k): v for k, v in ckpt['per_length'].items()}
                ac_classes_per_length.update({k: {'enumerated': v['enumerated'],
                                                  'classes': v['classes']}
                                              for k, v in per_length.items()})
                judged = {(tuple(lhs), tuple(rhs)): tier for lhs, rhs, tier in ckpt['judged']}
                if provenance is not None:
                    # The mine began where and when the checkpoint says (its `created` and
                    # `host`); this run is recorded as a resume of it, not as its origin.
                    import platform
                    provenance.update(ckpt['provenance'])
                    provenance['resumed'] = list(provenance.get('resumed') or []) + [
                        {'host': platform.node(), 'at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                         'after_length': resumed_at}]
                if verbose:
                    print(f'=== resuming the {file_mode} mine after length {resumed_at}')

        def _checkpoint(completed_length: int, completed_rules: list) -> None:
            """THE RESUME POINT, never under the final filename: the rules file and its sidecar
            are written once, by `_finalize`, and a crash leaves only this. The judge's memo
            is written whole, so a resume re-judges nothing."""
            if out_file is None:
                return
            with open(out_file + '.checkpoint.json', 'w') as file:
                json.dump({'fingerprint': checkpoint_fp, 'completed_length': completed_length,
                           'rules': [[list(lhs), list(rhs)] for lhs, rhs in completed_rules],
                           'per_length': {str(k): v for k, v in per_length.items()},
                           'judged': [[list(lhs), list(rhs), tier]
                                      for (lhs, rhs), tier in judged.items()],
                           'provenance': provenance},
                          file, default=str)

        completed = resumed_at
        for length in sorted(k for k in list(expressions_of_length) if k <= max_source_pattern_length):
            if interrupted():
                break
            if length <= resumed_at:
                continue
            # The mark is this mode's serve answer under ITS rules so far.
            self.engine._core.set_mode_rules(rule_mode, [(list(lhs), list(rhs)) for lhs, rhs in rules])
            # CERTIFY THE CANONICAL REPRESENTATIVE ONLY (owner ruling, re-mine 2026-08-20).
            #
            # Certification runs in f64 on whichever spelling was enumerated, and AC says all
            # regroupings of `a*b*c` are ONE expression while f64 says they are not --
            # `(1e200*1e200)/1e200` is inf, `1e200*(1e200/1e200)` is 1e200. So a class could
            # certify or not depending on association, which is what the 13 rules lost to
            # evaluation order were. Mining one representative per class makes certification a
            # function of the CLASS, so AC-dedup is sound by construction and every member
            # gets an identical budget -- skipping a duplicate cannot lose a rule the skipped
            # spelling would have reached.
            #
            # `ac_canonical_key` IS the loader's key: two sources share it iff the loader
            # builds the same internal pattern from them, so a skipped spelling is served by
            # the representative's rule rather than going unserved. An unparseable source has
            # no class and is mined on its own.
            # The mode's CANONICAL PROJECTION of each class is its source (one per class,
            # atomic ones dropped): what is certified is what is written.
            _raw_sources = [list(expression) for expression in sorted(expressions_of_length[length])]
            sources = self.engine._core.mine_sources(_raw_sources, rule_mode=rule_mode)
            n_ac_classed = len(_raw_sources) - len(sources)
            if verbose and n_ac_classed:
                print(f'Length {length}: {len(_raw_sources):,} sources -> {len(sources):,} AC classes '
                      f'({n_ac_classed:,} duplicate spellings skipped)')
            ac_classes_per_length[length] = {'enumerated': len(_raw_sources),
                                             'classes': len(sources)}
            if provenance is not None:
                provenance['ac_classes'] = {str(k): v for k, v in ac_classes_per_length.items()}
            del _raw_sources
            n_sources = len(sources)
            # Release this length's source UNIVERSE as soon as it has been linearised (the
            # caller passes each mode its own view of the universe).
            del expressions_of_length[length]
            # Per-length seed block: lengths are spaced by 2^40 (far above any per-length
            # source count) so the per-source streams (seed + index) never collide.
            # Stage 2 runs inside the scan: an unconfirmed pick yields to the next candidate.
            confirm_kwargs = self._confirm_kwargs(X_confirm, X_data, min_informative,
                                                  confirm_seed + (length << 40))

            def mine(batch: list, indices: Any, exclusions: Any, _length: int = length) -> list:
                return self.engine._core.mine_one_length(
                    batch, library, max_target_pattern_length,
                    constants_fit_challenges, constants_fit_retries,
                    mine_seed + (_length << 40), rtol, atol, min_informative,
                    relaxed_kruskal, rule_mode=rule_mode, indices=indices,
                    exclusions=exclusions, **confirm_kwargs)

            _t0 = time.monotonic()
            found, _loop = self._mine_tier_with_progress(
                length, len(sources),
                lambda: self._mine_length(sources, mine, rule_mode, judge, judged,
                                          dummy_variables),
                verbose)
            if _loop['stubborn']:
                # Every verdict this length reached is kept for the resume; the length itself
                # is not complete, so the checkpoint stays at the last one that is.
                _checkpoint(completed, rules)
                raise RuntimeError(
                    f'the judge loop gave up on {len(_loop["stubborn"])} source(s) of length '
                    f'{length} in mode {rule_mode}, each refused {_JUDGE_ROUNDS_MAX} times '
                    f'(source, excluded targets): {_loop["stubborn"]}; every other verdict is '
                    f'checkpointed')
            # GATED BEFORE INSTALLED (miner.md 2.7): only licensed picks join the set the next
            # length's marks are computed against, so a rule the judge refuses never prunes a
            # longer source.
            _shadowed: list = []
            if found:
                rules = deduplicate_rules(
                    rules + [(tuple(lhs), tuple(rhs)) for lhs, rhs in found],
                    dummy_variables, verbose=verbose, engine=self.engine, rule_mode=rule_mode)
                rules, _shadowed = self._deshadow(rule_mode, rules)
            per_length[length] = {**ac_classes_per_length[length], 'picks': len(found),
                                  'judge_rounds': _loop['judge_rounds'],
                                  'excluded': _loop['excluded'],
                                  'twin_shadowed': [[list(lhs), list(rhs)]
                                                    for lhs, rhs in _shadowed],
                                  'total': len(rules),
                                  'wall_s': round(time.monotonic() - _t0, 3)}
            if provenance is not None:
                provenance['per_length'] = {str(k): v for k, v in per_length.items()}
            if verbose:
                print(f'Length {length}: {n_sources:,} sources, {len(found):,} new rules '
                      f'({_loop["judge_rounds"]} judge rounds, {len(_loop["excluded"])} excluded, '
                      f'{len(_shadowed)} twin-shadowed), {len(rules):,} total')
            self._write_provenance(None, provenance, rules, completed_length=length)
            completed = length
            _checkpoint(completed, rules)
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
                self._install(rule_mode, rules)

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
        prov['universe'] = {k: v for k, v in prov.get('universe', {}).items() if int(k) <= length}
        prov['ladder_snapshot'] = {
            'emitted_at_source_length': length,
            'climb_max_source_pattern_length': climb_max,
            'equivalence': 'Lengths are mined shortest-first and the per-source seed is '
                           'mine_seed + (length << 40). Every bound the search applies is a '
                           'function of the SOURCE and the candidate alone: the candidate library '
                           'is bounded by max_target_pattern_length, mu is scored once from the '
                           'bare context at library-build time, acceptance requires mu(target) < '
                           'mu(source), and certification is per canonical source class. None of '
                           'them reads how far the climb continues, so this prefix equals a '
                           'one-shot mine of this cell.',
        }
        return prov

    @staticmethod
    def _write_provenance(output_file: str | None, provenance: dict | None, rules: list,
                          completed_length: int | None = None, final: bool = False,
                          interrupted: bool = False) -> None:
        """Write/refresh the mined artifact's PROVENANCE sidecar (`<output>.provenance.json`).

        The rules.json format is a bare list (the engine loads it directly), so provenance
        lives beside it: parameters, seeds, X spec, universe coverage (filled by
        :meth:`find_rules`) plus rolling progress and the final rule census. A mine is
        reproducible from the sidecar alone unless X was passed as an explicit array
        (recorded by content hash in that case). An interrupted mine's sidecar carries
        ``final: False`` and ``interrupted: True``. ``output_file=None`` records the progress
        in ``provenance`` without writing (the climb's checkpoint carries it).
        """
        if provenance is None:
            return
        import time as _time
        by_len: dict[str, int] = {}
        for lhs, _ in rules:
            by_len[str(len(lhs))] = by_len.get(str(len(lhs)), 0) + 1
        provenance['progress'] = {
            'updated': _time.strftime('%Y-%m-%d %H:%M:%S %z'),
            # A finalize write passes no length and keeps the one the climb recorded.
            'last_completed_source_length': completed_length if completed_length is not None
            else provenance.get('progress', {}).get('last_completed_source_length'),
            'final': final,
            'rules_total': len(rules),
            'rules_by_lhs_length': dict(sorted(by_len.items(), key=lambda x: int(x[0]))),
        }
        if interrupted:
            provenance['progress']['interrupted'] = True
        if output_file is None:
            return
        with open(output_file + '.provenance.json', 'w') as file:
            json.dump(provenance, file, indent=2)

    def _build_source_universe(
            self,
            leaf_nodes: list[str],
            non_leaf_nodes: dict[str, int],
            max_source_pattern_length: int,
            verbose: bool) -> tuple[dict[int, set[tuple[str, ...]]], dict[int, int]]:
        """Phase 1 of :meth:`find_rules`: build the per-length source/candidate universe.

        COMPLETE bottom-up DP enumeration per length: enumeration must SATURATE each
        length, not merely reach it (a pass-based closure that stops once
        ``max_source_pattern_length`` is reached silently misses whole expression
        families).

        Returns ``(expressions_of_length, counts)``: the enumerated universe per
        length, and the complete-universe count DP.
        """
        counts = count_expressions(len(leaf_nodes), non_leaf_nodes, max_source_pattern_length)
        if verbose:
            print(f"Phase 1: enumerating all expressions up to length {max_source_pattern_length}")

        expressions_of_length = enumerate_expressions(leaf_nodes, non_leaf_nodes, max_source_pattern_length)
        # Two paths, one truth: the enumerated universe must match the count DP exactly.
        for length, expressions in expressions_of_length.items():
            if len(expressions) != counts[length]:
                raise AssertionError(
                    f'enumeration incomplete at length {length}: {len(expressions):,} != {counts[length]:,}')

        return expressions_of_length, counts

    def _build_mine_provenance(
            self, *,
            X: np.ndarray | int | None,
            X_data: np.ndarray,
            counts: dict[int, int],
            expressions_of_length: dict[int, set[tuple[str, ...]]],
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
            relaxed_kruskal: bool,
            prune: bool | str,
            reset_rules: bool,
            interval_node_budget: int | None,
            judge_warn_s: float = JUDGE_WARN_S,
            judge_fail_s: float = JUDGE_FAIL_S) -> dict:
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
        from simplipy.compat import requires_simplipy_specifier
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

        # R5 + C23c-prov (the G2 gate): the numeric stack that minted the artifact is
        # part of its identity. scipy's PRESENCE alone changes mined output (audit N3:
        # a promotion reads PROMOTE with scipy, NO-WITNESS without -- now a hard error,
        # C23c-loud, but the VERSION still matters); the host libm decides real rules
        # (audit L9: >= 5 shipped rules exist only because glibc 2.43 is 1 ulp wrong,
        # and libm resolves at runtime, so no wheel pin can carry this). The
        # "reproducible from the sidecar alone" claim holds AT THE RECORDED
        # ENVIRONMENT; this block is what makes that qualification checkable.
        def _pkg_version(name: str) -> str | None:
            try:
                from importlib.metadata import version
                return version(name)
            except Exception:
                return None
        environment = {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'libc': '-'.join(v for v in platform.libc_ver() if v) or None,
            'numpy': _pkg_version('numpy'),
            'scipy': _pkg_version('scipy'),
            'mpmath': _pkg_version('mpmath'),
            'libm_fingerprint': self.engine.libm_fingerprint(),
        }
        return {
            'created': _time.strftime('%Y-%m-%d %H:%M:%S %z'),
            'host': platform.node(),
            'simplipy_version': _simplipy_version,
            # The versions that serve these rules (simplipy.compat): the published
            # config carries this value as its `requires_simplipy` key.
            'requires_simplipy': requires_simplipy_specifier(_simplipy_version),
            'core_build': _core_build,
            # The code that minted the file: the compiled core's and the package's digests.
            'code_digest': code_digest(),
            'environment': environment,
            'measure': self.engine._measure_fingerprint(),
            'params': {
                'max_source_pattern_length': max_source_pattern_length,
                'max_target_pattern_length': max_target_pattern_length,
                'dummy_variables': list(dummy_variables),
                'extra_internal_terms': list(extra_internal_terms),
                'constants_fit_challenges': constants_fit_challenges,
                'constants_fit_retries': constants_fit_retries,
                'rtol': rtol, 'atol': atol, 'min_informative': min_informative,
                'seed': seed, 'mine_seed': mine_seed, 'confirm_seed': confirm_seed,
                'confirm': confirm,
                'relaxed_kruskal': relaxed_kruskal,
                'prune': prune, 'reset_rules': reset_rules,
                'interval_node_budget': interval_node_budget,
                'judge_warn_s': judge_warn_s, 'judge_fail_s': judge_fail_s,
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
                } for length in sorted(expressions_of_length)
            },
            'operators': sorted(self.engine.operator_arity),
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
        never by the reduction alone (otherwise a canon respell would read as "covered" and
        silently discard a certifiable proposal).

        Intended for LLM- or human-proposed identities: proposals only ever ADD source
        expressions, and every accepted pair passes the same symbolic gate the mine
        runs (:func:`~simplipy.verify.verify_ruleset` with the precision-stability
        discriminator; KILLs dropped) -- so certified output is held to exactly the bar
        mined rules are.

        Parameters mirror :meth:`find_rules`. The engine is not modified; returns a list
        of ``(source, target, certificate)`` tuples with certificate ``'minimal'`` or
        ``'verified'``. Merge accepted pairs explicitly, e.g.::

            pairs = engine.certify_rules(proposals, hints)
            engine.simplification_rules = deduplicate_rules(
                engine.simplification_rules + [(s, t) for s, t, _ in pairs], dummies,
                engine=engine, rule_mode='f64')
            engine.compile_rules()
        """
        sources = [tuple(str(t) for t in s) for s in sources]
        hint_list: list = list(targets) if targets is not None else [None] * len(sources)
        if len(hint_list) != len(sources):
            raise ValueError('targets must parallel sources')
        # D28: one-miner-per-process is DOCTRINE, and this entry was found unguarded --
        # it runs the same certification machinery and mutates the same process-global
        # interval soundness counters the provenance sidecar records. Same lock, same
        # single-flight semantics, acquired after the cheap validation above (the
        # H-002/H-009 leak discipline) and released in the finally below.
        if not _MINE_LOCK.acquire(blocking=False):
            raise RuntimeError(
                "another find_rules mine is active in this process; certification is "
                "single-flight with mining (the interval soundness counters recorded "
                "in the provenance sidecar are process-global)")
        try:
            return self._certify_rules_locked(
                sources, hint_list, max_target_pattern_length, dummy_variables,
                extra_internal_terms, X, constants_fit_challenges,
                constants_fit_retries, rtol, atol, min_informative, seed, verbose)
        finally:
            _MINE_LOCK.release()

    def _certify_rules_locked(
            self, sources: list, hint_list: list, max_target_pattern_length: int,
            dummy_variables: int | list[str] | None, extra_internal_terms: list[str] | None,
            X: int | None, constants_fit_challenges: int, constants_fit_retries: int,
            rtol: float, atol: float, min_informative: int | None, seed: int | None,
            verbose: bool) -> list[tuple[tuple[str, ...], tuple[str, ...], str]]:
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
        non_leaf_nodes = dict(sorted(self.engine.operator_arity.items(), key=lambda x: x[1]))
        expressions = enumerate_expressions(leaf_nodes, non_leaf_nodes, max_target_pattern_length)
        candidates = [list(expression)
                      for length in sorted(expressions)
                      for expression in sorted(expressions[length])]
        library = self.engine._core.build_candidate_library(
            candidates, list(dummy_variables), X_data.flatten(order='C').tolist(),
            X_data.shape[0], rule_mode=_CERTIFY_RULE_MODE)
        if verbose:
            print(f'Candidate library: {library.n_candidates:,} candidates '
                  f'({library.n_filtered:,} var-free filtered)')

        vocabulary = set(leaf_nodes) | set(self.engine.operator_arity)
        certified: list = []
        fit = dict(challenges=constants_fit_challenges, retries=constants_fit_retries,
                   rtol=rtol, atol=atol, min_informative=min_informative)
        for index, (source, hint) in enumerate(zip(sources, hint_list)):
            if not _tokens_in_vocabulary(source, vocabulary) or not self.engine.is_valid(list(source)):
                continue
            # The engine's own result is the MARK TO BEAT, never a reason to skip
            # (finding F2): a source canon merely RESPELLS shorter used to read as
            # "already covered" and was never searched, losing e.g.
            # exp(t)*exp(t) -> exp(t+t) forever (the respelled form speaks tokens
            # outside the mining alphabet, so no later tier picks it up).
            target, certificate, _ = self._certify_one(
                source, hint, library, vocabulary, max_target_pattern_length, fit,
                mine_seed + (len(source) << 40) + index,
                self._confirm_kwargs(X_confirm, X_data, min_informative,
                                     confirm_seed + (len(source) << 40) + index),
                _CERTIFY_RULE_MODE)
            if target is None:
                continue
            certified.append((source, tuple(target), certificate))
            if verbose:
                print(f'certified ({certificate}): {list(source)} -> {list(target)}')
        # SYMBOLIC GATE, matching the mining path (`_finalize`; audit B3, fatal set B7).
        # Every numeric acceptance above is tolerance-bounded, and the impostor family
        # lives BELOW any usable tolerance: `tanh(exp(e)) -> 1` is false by a stable
        # 1.37e-13 and came back certified 'verified' before this gate existed -- while
        # the same pair, arriving through `find_rules(proposals=...)`, was killed. The
        # parity sentence above ("exactly as sound as mined rules") is earned here, not
        # assumed: every accepted pair is re-judged by the independent symbolic
        # verifier, and kept iff its tier is one the certifying mode licenses
        # (`MODE_TIERS`, the table every file's gate reads). Fail-closed is untouched for
        # the verdicts that carry no tier at all (NO-WITNESS / UNSUPPORTED-SHAPE /
        # JUDGE-TIMEOUT / UNRESOLVED-COVERAGE): they read `reject` and drop, because an
        # unjudged certificate is an unearned one. TOLERATED passes, exactly as at the
        # mining gate.
        if certified:
            from .verify import effective_tier, verify_ruleset
            _rep = verify_ruleset([[list(s), list(t)] for s, t, _ in certified])
            # detail entries carry lhs/rhs as SPACE-JOINED strings; the Const-channel
            # sentinel is exempt exactly as at the mining gate (D36).
            _drop = {(tuple(d['lhs'].split()), tuple(d['rhs'].split())): d['verdict']
                     for d in _rep['detail']
                     if effective_tier(d) not in _mode_tiers(_CERTIFY_RULE_MODE)}
            if _drop:
                if verbose:
                    for s, t, _c in certified:
                        v = _drop.get((tuple(s), tuple(t)))
                        if v is not None:
                            print(f'symbolic gate {v}: {list(s)} -> {list(t)}')
                certified = [(s, t, c) for s, t, c in certified
                             if (tuple(s), tuple(t)) not in _drop]
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
            verbose: bool,
            rule_mode: str,
            judge: Callable,
            judged: dict) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
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
        :func:`~simplipy.utils.deduplicate_rules` path as mined rules (per canonical
        source, the target lowest in the mode's serve ordering). Mutates the engine in place (rules, compiled state, core
        rules) and returns ``(counts, trail, judge_loop)``.

        THE JUDGE LOOP, AS ONE EXTRA LENGTH: the certified targets go to the
        mine's judge exactly as a length's picks do; a target the mode does not license is
        excluded and its proposal certified again without it, so the hint or a scan
        alternative of another class can still certify. A proposal left without a licensed
        target reads ``rejected`` at stage ``judge``. ``judge_loop`` records the rounds and
        the exclusions.

        ``counts`` is the per-outcome tally for the provenance sidecar: ``certified`` /
        ``already_covered`` / ``rejected`` / ``duplicate`` (certified, but canonically
        identical to an earlier certified proposal and not lower in the serve ordering) /
        ``certified_then_dropped``. The last one is filled in LATER, by
        :func:`_reconcile_proposal_census`: certification runs upstream of the symbolic
        gate, the covered-prunes and the sort promotion, so the tally this method returns
        is provisional and the reconciliation moves every certified pair the artifact
        does not contain out of ``certified``. The key is present even when zero.

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

        A trail entry whose pair does not survive to the artifact is rewritten by
        :func:`_reconcile_proposal_census` to the TERMINAL verdict
        ``certified_then_dropped``, keeping ``stage``/``certificate`` (it WAS certified)
        and gaining ``dropped_at`` (the stage that removed it) and ``drop_reason`` (that
        stage's own verdict for it).
        """
        assert self.engine._core is not None
        counts = {'certified': 0, 'already_covered': 0, 'rejected': 0, 'duplicate': 0,
                  'certified_then_dropped': 0}
        vocabulary = set(leaf_nodes) | set(self.engine.operator_arity)
        fit = dict(challenges=constants_fit_challenges, retries=constants_fit_retries,
                   rtol=rtol, atol=atol, min_informative=min_informative)
        confirm = self._confirm_kwargs(X_confirm, X_data, min_informative, confirm_seed)
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

        pending: dict[int, dict[str, Any]] = {}
        for index, (source, hint) in enumerate(proposal_entries):
            if not _tokens_in_vocabulary(source, vocabulary) or not self.engine.is_valid(list(source)):
                continue
            # The engine's own result is the MARK TO BEAT, never a reason to skip
            # (finding F2): the serve-time ordering acceptance rides the candidate
            # scan, so 'already_covered' below is a SEARCH-VERIFIED
            # verdict ("the engine reduces it and nothing expressible beats its
            # result"), not a guess from a spelling shrink.
            # "The engine already reduces it" is asked of the mode the file serves: its
            # serve answer (the mine's mark) is an atom, or a different state from that
            # mode's own reading of the source (the serve walk only descends). A respell
            # is the same state: if the search also comes back empty-handed, the honest
            # verdict below is 'rejected', never a coverage claim nothing backs.
            mark = self.engine._core.mine_mark(list(source), rule_mode=rule_mode)
            reading = self.engine._core.mine_sources([list(source)], rule_mode=rule_mode)
            engine_reduces = len(mark) == 1 or not reading or reading[0] != list(mark)
            # Content-derived per-proposal seed offset (blake2b, not hash(): PYTHONHASHSEED
            # randomises str hashing per process).
            offset = int.from_bytes(
                hashlib.blake2b(' '.join(source).encode(), digest_size=7).digest(), 'little')
            pending[index] = {'reduces': engine_reduces, 'seed': mine_seed + offset,
                              'result': self._certify_one(
                                  source, hint, library, vocabulary, max_target_pattern_length,
                                  fit, mine_seed + offset, confirm, rule_mode)}

        excluded: dict[int, list] = {}
        stubborn: list[int] = []
        rounds = 0
        while True:
            rounds += 1
            picks = {i: (list(proposal_entries[i][0]), list(p['result'][0]))
                     for i, p in pending.items() if p['result'][0] is not None}
            bad = self._unlicensed(picks, rule_mode, judge, judged, dummy_variables) if picks else set()
            again: list[int] = []
            for i in sorted(bad):
                excluded.setdefault(i, []).append(list(pending[i]['result'][0]))
                # the cap is per proposal, as per source in the length loop
                (stubborn if len(excluded[i]) >= _JUDGE_ROUNDS_MAX else again).append(i)
            if stubborn:
                raise RuntimeError(
                    f'the proposals\' judge loop gave up on {len(stubborn)} proposal(s) in mode '
                    f'{rule_mode}, each refused {_JUDGE_ROUNDS_MAX} times (source, excluded '
                    f'targets): {[[list(proposal_entries[i][0]), excluded[i]] for i in stubborn]}')
            if not again:
                break
            for i in again:
                source, hint = proposal_entries[i]
                pending[i]['result'] = self._certify_one(
                    source, hint, library, vocabulary, max_target_pattern_length, fit,
                    pending[i]['seed'], confirm, rule_mode, excluded=excluded[i])

        for index, (source, hint) in enumerate(proposal_entries):
            if index not in pending:
                counts['rejected'] += 1
                record(source, None, 'rejected', 'vocabulary')
                continue
            target, certificate, refusal = pending[index]['result']
            if target is None:
                if index in excluded:
                    counts['rejected'] += 1
                    record(source, None, 'rejected', 'judge')
                elif refusal == 'unconfirmed':
                    counts['rejected'] += 1
                    record(source, hint, 'rejected', 'confirm')
                elif pending[index]['reduces']:
                    counts['already_covered'] += 1
                    record(source, None, 'already_covered', 'covered')
                else:
                    counts['rejected'] += 1
                    record(source, None, 'rejected', 'search')
                continue
            certified_trail_idx.append(record(source, target, 'certified', 'accepted', certificate))
            certified_pairs.append((tuple(source), tuple(target)))
            if verbose:
                print(f'Proposal certified ({certificate}): {list(source)} -> {list(target)}')
        judge_loop = {'judge_rounds': rounds,
                      'excluded': [[list(proposal_entries[i][0]), target]
                                   for i, targets in sorted(excluded.items()) for target in targets]}

        # Per-proposal accounting that mirrors the deduplicate_rules fold EXACTLY (keep
        # first canonical source unless a target strictly below it in the mode's ordering
        # arrives): a certified pair that would not change the merged set is a
        # 'duplicate', not a 'certified'.
        before = self._mode_rules(rule_mode)
        seen: dict[tuple, Any] = {}
        for key, (_, canon_target) in _dedup_keyed_rules(before, dummy_variables, self.engine,
                                                         rule_mode=rule_mode):
            if key not in seen or dedup_target_wins(self.engine, canon_target, seen[key],
                                                    rule_mode):
                seen[key] = canon_target
        for idx, (key, (_, canon_target)) in zip(
                certified_trail_idx,
                _dedup_keyed_rules(certified_pairs, dummy_variables, self.engine,
                                   rule_mode=rule_mode)):
            if key in seen and not dedup_target_wins(self.engine, canon_target, seen[key],
                                                     rule_mode):
                counts['duplicate'] += 1
                trail[idx].update(verdict='duplicate', stage='merge')
            else:
                counts['certified'] += 1
                seen[key] = canon_target
        if certified_pairs:
            # The SAME merge path as mined rules: per canonical source, the lowest target.
            self._install(rule_mode, deduplicate_rules(
                before + certified_pairs, dummy_variables, verbose=verbose,
                engine=self.engine, rule_mode=rule_mode))
        if verbose:
            print(f"Proposals: {len(proposal_entries)} processed -> "
                  f"{counts['certified']} certified, {counts['already_covered']} already covered, "
                  f"{counts['rejected']} rejected, {counts['duplicate']} duplicate")
        return counts, trail, judge_loop

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
            relaxed_kruskal: bool = True,
            proposals: str | list | dict | None = None,
            promote_sorts: bool = True,
            symbolic_gate: bool = True,
            snapshot_at: dict[int, str] | None = None,
            judge: Callable | None = None,
            judge_processes: int = 4,
            interval_node_budget: int | None = None,
            judge_warn_s: float = JUDGE_WARN_S,
            judge_fail_s: float = JUDGE_FAIL_S) -> None:
        """Systematically discovers new simplification rules.

        This method automates the discovery of simplification rules, one file per mode
        (``rules_f64.json``, ``rules_real.json``, ``rules_permissive.json``), each mined in
        its own mode. It operates in two phases:
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
            Deprecated and ignored (kept for signature compatibility). A mine writes
            ``<output_file>.checkpoint.json`` after every completed length and the rule
            files once, when each mode's mine ends.
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
            stronger ``_``/``!``/``$`` sorts and shipping it at the strongest one it
            certifies. On by default, so a default mine emits the ruleset that is
            deployed. Pass ``False`` for a raw,
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
        judge : callable or None, optional
            The judge of the per-length loop: ``judge(pairs) -> tiers``, the effective
            judged tier of each served ``(lhs, rhs)`` pattern. A pick whose tier the file's
            mode does not license is excluded and its source re-scanned. ``None`` uses
            :func:`judge_tiers` (the contract judge) with ``judge_processes`` workers.
        judge_processes : int, optional
            Worker processes of the default judge. Verdicts do not depend on it.
        interval_node_budget : int or None, optional
            The interval layer's subdivision node budget for this mine (``None``: the
            default, 400,000). An exhausted search is undecided and refuses its candidate;
            raise it when the sidecar's ``interval_undecided`` counters flag systematic
            exhaustion. Set for the mine only, and recorded in the sidecar's parameters.
        judge_warn_s, judge_fail_s : float, optional
            The judge's watchdog. A mine judges with no wall-clock timeout, so no verdict
            depends on the machine's load; a judgment still running after ``judge_warn_s``
            seconds (default 60) is logged, and one past ``judge_fail_s`` (default 600) fails
            the mine with the rule's name. Recorded in the sidecar's parameters.

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

        # (missed-003: the reset used to happen HERE, before the single-flight lock
        # below -- a REJECTED call had already destroyed the caller's rules. It now
        # runs under the lock; nothing between here and the acquire reads the
        # ruleset, so the mine's semantics are unchanged.)

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
        non_leaf_nodes = dict(sorted(self.engine.operator_arity.items(), key=lambda x: x[1]))

        # --- Phase 1: build the source/candidate universe (see _build_source_universe) ---
        expressions_of_length, counts = self._build_source_universe(
            leaf_nodes, non_leaf_nodes, max_source_pattern_length, verbose)

        total_expressions = sum(len(v) for v in expressions_of_length.values())

        if verbose:
            print(f"Finished generating expressions up to size {max_source_pattern_length}. Total expressions: {total_expressions:,}")
            for length, expressions in sorted(expressions_of_length.items()):
                print(f"Size {length}: {len(expressions):,} expressions")

        # PROVENANCE (see _build_mine_provenance).
        provenance = self._build_mine_provenance(
            X=X, X_data=X_data, counts=counts,
            expressions_of_length=expressions_of_length,
            max_source_pattern_length=max_source_pattern_length,
            max_target_pattern_length=max_target_pattern_length,
            dummy_variables=dummy_variables,
            extra_internal_terms=extra_internal_terms,
            constants_fit_challenges=constants_fit_challenges,
            constants_fit_retries=constants_fit_retries,
            rtol=rtol, atol=atol, min_informative=min_informative,
            seed=seed, mine_seed=mine_seed, confirm_seed=confirm_seed,
            confirm=confirm,
            relaxed_kruskal=relaxed_kruskal, prune=prune, reset_rules=reset_rules,
            interval_node_budget=interval_node_budget,
            judge_warn_s=judge_warn_s, judge_fail_s=judge_fail_s)
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
        # The reset happens UNDER the single-flight lock (missed-003): a rejected
        # call must leave the engine exactly as it found it.
        if reset_rules:
            self.engine._replace_rules([])
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
        from simplipy import _core as _core_module
        previous_budget = _core_module.set_interval_node_budget(interval_node_budget)
        try:
            if reset_rules:
                # every mode's set starts empty: each file is mined from nothing
                self.engine.real_simplification_rules = None
                self.engine.permissive_simplification_rules = None
            else:
                for rule_mode, _ in _MINE_MODES[1:]:  # a set deferred by `modes=` loads now
                    self.engine._ensure_mode_loaded(rule_mode)
            # EACH MODE'S MINE STARTS FROM ITS OWN SET, never from another mode's: a mine reads
            # its set in its own canon, and another mode's facts read there are not facts. A
            # mode with no set of its own starts from nothing.
            initial = {rule_mode: self._mode_rules(rule_mode) for rule_mode, _ in _MINE_MODES}
            finals: dict[str, list] = {}
            for rule_mode, _file_mode in _MINE_MODES:
                if interrupted:
                    break
                # Each mode after f64 mines against the finished f64 set in the default slot, as
                # its file is served beside it.
                self.engine._replace_rules(finals.get('default', initial['default']))
                self._find_rules_native(
                    dict(expressions_of_length),
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
                    relaxed_kruskal=relaxed_kruskal,
                    provenance=copy.deepcopy(provenance),
                    proposal_entries=proposal_entries,
                    leaf_nodes=leaf_nodes,
                    promote_sorts=promote_sorts,
                    symbolic_gate=symbolic_gate,
                    snapshot_at=snapshot_at,
                    rule_mode=rule_mode,
                    initial_rules=initial[rule_mode],
                    finals=finals,
                    judge=judge if judge is not None else functools.partial(
                        judge_tiers, processes=judge_processes, warn_s=judge_warn_s,
                        fail_s=judge_fail_s),
                    judge_warn_s=judge_warn_s,
                    judge_fail_s=judge_fail_s,
                )
            # The engine ends holding the three files it wrote, as it would load them.
            if 'default' in finals:
                self.engine.real_simplification_rules = finals.get('real')
                self.engine.permissive_simplification_rules = finals.get('permissive')
                self.engine._rule_canons.clear()  # each set was mined in its own mode
                self.engine._replace_rules(finals['default'])
        finally:
            _core_module.set_interval_node_budget(previous_budget)
            if handler_installed:
                signal.signal(signal.SIGINT, old_handler)
            _MINE_LOCK.release()


def verify_mine(rules_path: str, mode: Any, operators: dict, *,
                judge: Callable | None = None, judge_processes: int = 4) -> dict:
    """The re-mine gates of one mined file, in its own mode.

    ``rules_path`` is a rule file a mine wrote, read beside its provenance sidecar;
    ``mode`` the mode it serves (``'f64'``, also ``'default'``, ``'real'``, ``'permissive'``
    or a :class:`~simplipy.Mode` member); ``operators`` the engine's operator table. Each gate holds or names what breaks it:

    - ``sidecar``: the sidecar exists, records a final, uninterrupted mine, and its mode is
      ``mode``;
    - ``rules``: the file holds rules (an empty file passes every other gate vacuously);
    - ``twin_shadow``: no row the mode's load would mint anyway (a twin-shadow owner);
    - ``engine_owned_law``: no rule that is a respelling within one canonical state, or an
      instance of the distributive law as the search applies it (``law_instance``). The sign
      orbit is not among them yet: a sign pull through an odd function
      (``neg sin _0 -> sin neg _0``) passes until the sign rule makes the orbit one canon;
    - ``unlicensed``: every file spelling and every entry the mode serves from it carries
      a judged tier the mode licenses (the mine's final gate, with ``judge``);
    - ``preempted``: no ground rule the mode's constructor already performs;
    - ``translation``: the mode's load subsumes and drops nothing;
    - ``fingerprint``: the sidecar's measure record is of ``mode``, holds every probe, and
      prices each as this engine does;
    - ``env_switches``: the mine ran with no ``SIMPLIPY_*`` switch set, and so does this
      check.

    Returns ``{'file', 'mode', 'ok', 'gates': {name: {'ok', 'count', 'examples'}}}``.
    """
    from .engine import SimpliPyEngine, _RULE_MODE, _coerce_mode
    mode = _coerce_mode(mode).name
    rule_mode = _RULE_MODE[_coerce_mode(mode)]
    with open(rules_path) as file:
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(file)]
    sidecar_path = rules_path + '.provenance.json'
    sidecar: dict = {}
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as file:
            sidecar = json.load(file)
    engine = SimpliPyEngine(operators=operators, rules=[])
    miner = RuleMiner(engine)
    miner._install(rule_mode, rules)
    core = engine._core
    gates: dict[str, list] = {}
    # NOTHING PASSES VACUOUSLY: the file is a finished mine of this mode, and holds rules.
    progress = sidecar.get('progress') or {}
    gates['sidecar'] = (
        [['no provenance sidecar', sidecar_path]] if not sidecar else
        ([['not final', progress.get('final')]] if progress.get('final') is not True else [])
        + ([['interrupted', True]] if progress.get('interrupted') else [])
        + ([['mode', sidecar.get('mode'), mode]] if sidecar.get('mode') != mode else []))
    gates['rules'] = [] if rules else [['the file holds no rules', rules_path]]
    gates['twin_shadow'] = [
        [list(rules[owner][0]), list(rules[owner][1])]
        for owner in sorted(_twin_shadow_owners(core.ac_shadow_census(rule_mode=rule_mode)))]
    gates['engine_owned_law'] = [
        [list(lhs), list(rhs), law] for lhs, rhs in rules
        if (law := core.law_instance(list(lhs), list(rhs), rule_mode=rule_mode)) is not None]
    gates['unlicensed'] = miner._final_gate(
        rule_mode, judge if judge is not None else functools.partial(
            judge_tiers, processes=judge_processes), {})[1]
    gates['preempted'] = [[list(lhs), list(rhs)] for lhs, rhs in rules
                          if _preempted_by(engine, lhs, rhs, mode)]
    _, subsumed, dropped, _ = core.ac_rules_info_in_mode(rule_mode=rule_mode)
    gates['translation'] = ([{'subsumed': subsumed, 'dropped': dropped}]
                            if subsumed or dropped else [])
    recorded = sidecar.get('measure')
    if isinstance(recorded, dict):
        # Every probe is recorded, in this file's mode, at this engine's price: a probe the
        # record lacks is one it cannot vouch for (an empty record vouches for nothing).
        moved, unrecorded = engine._measure_drift(recorded)
        gates['fingerprint'] = (
            [[probe, old, now] for probe, (old, now) in sorted(moved.items())]
            + [[probe, None, 'unrecorded'] for probe in unrecorded]
            + ([['the measure record is of mode', recorded.get('mode'), mode]]
               if recorded.get('mode') != mode else []))
    else:
        gates['fingerprint'] = [['no measure record in the sidecar', None, None]]
    recorded_env = (sidecar.get('soundness') or {}).get('env_overrides')
    gates['env_switches'] = (
        [['the sidecar records no env_overrides', None]] if recorded_env is None else
        [[key, value] for key, value in sorted(recorded_env.items())]
        + [[key, value] for key, value in env_overrides().items()])
    report = {name: {'ok': not found, 'count': len(found), 'examples': found[:10]}
              for name, found in gates.items()}
    return {'file': rules_path, 'mode': mode, 'ok': all(g['ok'] for g in report.values()),
            'gates': report}


def cross_mode_complexity(engine: Any, tokens: Any, mode: Any) -> int:
    """The price of an answer of ``mode`` on a scale shared by all three modes.

    Each mode's own :meth:`~simplipy.SimpliPyEngine.complexity` prices that mode's
    reading, so prices in different modes are not comparable. This compares answers of
    different modes in one declared reading instead: ``tokens`` (the answer as a prefix
    token list, explicit or tagged) is parsed as ``mode`` reads it and priced, certified,
    in f64's canon. ``mode`` is ``'f64'`` (also ``'default'``), ``'real'``,
    ``'permissive'`` or a :class:`~simplipy.Mode` member. Returns the price in
    milli-bits, the unit of :meth:`~simplipy.SimpliPyEngine.complexity`.
    """
    from .engine import _RULE_MODE, _coerce_mode
    return int(engine._core.ac_complexity_certified(list(tokens), 'default', rule_mode=_RULE_MODE[_coerce_mode(mode)]))


def assert_permissive_dominates(engine: Any, expressions: Any) -> list:
    """`permissive` must be at least as capable as `f64` and as `real`, on every expression.

    THE REPLACEMENT for `rules_permissive == rules_f64 UNION rules_real`, which was retired
    once folding made the three modes live in different canonical worlds. That set
    identity was a proxy for this, and a bad one: it measured spelling overlap, so it read
    `acos 1 -> 0` as "missing from permissive" when permissive performs it -- and performs
    all 15 such rules -- through its own constructor.

    Stated as behaviour it is checkable and it is what anyone actually wants: for every
    expression, permissive's answer costs no more than either other mode's, all three
    priced by :func:`cross_mode_complexity`. Returns the violations, empty when the
    property holds. A row on which a mode raises or an answer cannot be priced is a
    violation too, reported with its error: a crash never reads as dominance.
    """
    from .engine import Mode
    modes = (Mode.f64, Mode.real, Mode.permissive)
    bad: list[dict[str, Any]] = []
    for expr in expressions:
        try:
            mu = {m: cross_mode_complexity(engine, engine.simplify(list(expr), mode=m), m)
                  for m in modes}
        except Exception as error:  # reported as a violation below, never skipped
            bad.append({'expression': list(expr), 'error': f'{type(error).__name__}: {error}'})
            continue
        if mu[Mode.permissive] > min(mu[Mode.f64], mu[Mode.real]):
            bad.append({'expression': list(expr),
                        'mu': {m.name: v for m, v in mu.items()}})
    return bad
