"""The MINE half of the package (C23a, D27).

``RuleMiner`` holds the 13 mining/certification methods that lived on
``SimpliPyEngine`` until 0.13.0. It is an ordinary object over an engine:
``RuleMiner(engine).find_rules(...)``. Every access to engine state is
EXPLICIT (``self.engine._core``, ``self.engine.simplification_rules``, ...) —
the miner owns no state of its own beyond the engine reference (C23a step 2,
the explicit-engine-access rewrite; step 1 was the verbatim facade move).
``SimpliPyEngine.find_rules`` / ``certify_rules`` remain as thin delegators,
so the public API does not move. The single-flight ``_MINE_LOCK`` and the
mine-only module helpers live HERE; ``engine.py`` re-exports them so the lock
stays ONE object across both modules (D28).

Equality gate (D29): full-mine byte-identity on ``rules.json``, machine-pinned
— applied to step 1 and step 2 separately.
"""

import hashlib
import os
import re
import warnings
import signal
import threading
from typing import Callable, TYPE_CHECKING
from typing import Any
from copy import deepcopy

import numpy as np
import json

from simplipy.utils import (
    is_numeric_string,
    deduplicate_rules,
    _dedup_keyed_rules,
    enumerate_expressions, count_expressions, sample_expression,
    remap_expression,
    violates_wildcard_multiplicity)

if TYPE_CHECKING:
    from .engine import SimpliPyEngine

# D11 column R30 (owner-ratified 2026-08-16): the miner is this module's
# power-user surface; the module helpers (_MINE_LOCK, ...) are private.
__all__ = ['RuleMiner']


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
    certification (F94, 2026-08-18).

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
                f"sidecar cannot account for one of its own certified pairs (F94)")
        stage, reason = record
        trail[index].update(verdict='certified_then_dropped',
                            dropped_at=stage, drop_reason=reason)
        outcomes['certified'] -= 1
        outcomes['certified_then_dropped'] += 1


def _tokens_in_vocabulary(tokens: Any, vocabulary: set) -> bool:
    """Vocabulary test for proposal/hint token sequences: numeric and constant-like
    literals are valid leaves everywhere (the generation-2 spellings carry their
    numbers as literal TOKENS -- `pow x0 2` where the retired ring spelled
    `pow2 x0` -- so a bare vocabulary-set test silently skipped every such
    proposal; audit Tier-1 #3)."""
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


#: Which modes a judged tier licenses a rule to serve in. `corpus` is DERIVED (a rule
#: serves there iff it serves in either), never listed: the ledger's invariant
#: `rules_corpus == rules_f64 UNION rules_real` then holds by construction rather than by
#: assertion, and a fourth entry here could silently break it.
_TIER_MODES: dict[str, frozenset] = {
    'core': frozenset({'f64', 'real'}),
    'real': frozenset({'real'}),
    'f64': frozenset({'f64'}),
    'reject': frozenset(),
}


#: Slot sigils. A pattern carrying one matches many instances, and the constructor may
#: reach the endpoint on the PATTERN while refusing it on some of them.
_SLOT_SIGILS = ('_', '?', '!', '$', '<')


def _is_ground(tokens: Any) -> bool:
    """No slots and no `<constant>`: the pattern IS its only instance."""
    return not any(str(t).startswith(_SLOT_SIGILS) for t in tokens)


def _preempted_by(engine: Any, lhs: Any, rhs: Any, mode: str) -> bool:
    """Does `mode`'s CONSTRUCTOR already reach this rule's endpoint without the rule?

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
    # instances the rule matches, and the two genuinely differ: in corpus the constructor
    # takes `/ $0 $0` to `1` -- the slot carries a nonzero-a.e. certificate -- while
    # taking the instance `inf / inf` to `nan`. Pruning on the pattern therefore deleted
    # `/ $0 $0 -> 1` and with it the mask-sentinel doctrine, which is exactly the rule
    # that makes `inf/inf * x0` collapse to `x0` in corpus.
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
    try:
        return tuple(bare.simplify(list(lhs), mode=mode)) == tuple(bare.simplify(list(rhs), mode=mode))
    except Exception:
        return False


def _route_triple(engine: Any, verbose: bool = False) -> tuple[dict, list, dict]:
    """Judge the FINAL served set and route every source rule into the triple.

    -> ({'f64': [...], 'real': [...], 'corpus': [...]}, rejected, tiers)

    Judged HERE and not reused from the symbolic gate, even though that is a second full
    pass: the gate runs before the covered-prune and the sort promotion, both of which
    change which rules exist AND how they are spelled, so its verdicts describe a rule set
    that is not the one being written. What is routed must be what is served.

    A source rule can mint several SERVED entries (orientation twins, minted at load from
    the loading engine's canon). Its licence is the INTERSECTION over all of them, which is
    the fail-closed combination and the same doctrine the symbolic gate already applies
    when it condemns a twin by dropping the rule that mints it: a rule may serve in a mode
    only if EVERY way it can fire is licensed there.
    """
    from .verify import CONST_CHANNEL_DETAIL, verify_ruleset

    served = engine._core.ac_served_rules()
    report = verify_ruleset([[list(lhs), list(rhs)] for lhs, rhs, _ in served])
    detail = report['detail'] if isinstance(report, dict) else report

    allowed: dict[int, frozenset] = {}
    tiers: dict[int, set] = {}
    for d in detail:
        src = served[d['idx']][2]
        # D36: the multi-`<constant>` family's soundness authority is the Const-channel
        # chain, not the contract, so the judge's non-jurisdiction sentinel is not a
        # verdict about the rule. Treated as `core`, which is exactly where those rules
        # serve today -- the carve-out stays visible in the sidecar, never silent.
        tier = 'core' if d.get('detail') == CONST_CHANNEL_DETAIL else (d.get('tier') or 'reject')
        tiers.setdefault(src, set()).add(tier)
        allowed[src] = allowed.get(src, _TIER_MODES['core']) & _TIER_MODES.get(tier, frozenset())

    triple: dict[str, list] = {'f64': [], 'real': [], 'corpus': []}
    rejected: list = []
    preempted: dict[str, int] = {'f64': 0, 'real': 0, 'corpus': 0}
    for i, (lhs, rhs) in enumerate(engine.simplification_rules):
        # A rule with no served entry cannot fire at all, so it is licensed nowhere. This
        # is the load-minted-twins lesson in the other direction: never infer a licence
        # for something the matcher never saw.
        modes = allowed.get(i, frozenset())
        pair = [list(lhs), list(rhs)]
        if not modes:
            rejected.append(pair)
            continue
        for m in ('f64', 'real', 'corpus'):
            if m != 'corpus' and m not in modes:
                continue
            # PER-MODE PRUNE. A mode's file carries only what that mode can actually
            # serve. Folding is mode-dependent, so a rule can be load-bearing in `real`
            # and pure dead weight in `f64`, where the constructor reaches the same
            # endpoint without it -- `asin 1e-08 -> 1e-08` is the whole 18-rule
            # `f(+-10^-k)` family. Shipping it anyway would put rules in a file that can
            # never fire from it, which is exactly the computed-rather-than-stated shape
            # the load-minted twins already punished once.
            if _preempted_by(engine, lhs, rhs, m):
                preempted[m] += 1
                continue
            triple[m].append(pair)

    _assert_triple_invariants(triple, rejected, engine.simplification_rules, allowed,
                              preempted)
    if verbose:
        print(f'Triple: f64={len(triple["f64"])} real={len(triple["real"])} '
              f'corpus={len(triple["corpus"])} rejected={len(rejected)}')
    return triple, rejected, {'tiers': {i: sorted(t) for i, t in tiers.items()},
                              'preempted': preempted}


def _assert_triple_invariants(triple: dict, rejected: list, mined: list,
                              allowed: dict, preempted: dict | None = None) -> None:
    """The ledger's four relationship invariants, checked before anything is written.

    They are a statement about the RELATIONSHIP between the three files, which is a
    stronger provenance guarantee than three independently pinned counts: a routing bug
    that moved a rule between sets leaves every individual count plausible and breaks one
    of these. Fail LOUDLY -- a partial triple is not shippable.

    Each check is written so it CAN fail. An invariant that restates its own construction
    proves nothing, and two of these did on the first pass.
    """
    def keys(rs: Any) -> set:
        return {(tuple(lhs), tuple(rhs)) for lhs, rhs in rs}

    f64, real, corpus = keys(triple['f64']), keys(triple['real']), keys(triple['corpus'])
    rej = keys(rejected)
    mined_keys = keys(mined)
    # INVARIANT 1 and INVARIANT 2 are RETIRED, and this is why. They compared three rule
    # sets as if they lived in ONE canonical world. Once folding is mode-dependent they do
    # not: `acos 1 -> 0` is absent from the corpus set purely because corpus's own
    # constructor performs it, and corpus performs all 15 such rules -- measured 15/15.
    # Set overlap was therefore measuring SPELLING, not capability, and the property it
    # stood proxy for ("corpus can do everything the other two can") is behavioural.
    # It is checked as behaviour, by `assert_corpus_dominates`, on the corpus sweep the
    # invariance gate already runs. Owner ruling 2026-08-20.
    #
    # What every set here must still satisfy: a rule may only appear where its tier
    # licenses it, MINUS what that mode's constructor already performs.
    # UNCONDITIONAL. This was gated on `preempted is not None`, which is only supplied by
    # the production path, so every direct call skipped the one invariant that replaced
    # the two retired ones. The check needs `allowed`, which is always passed.
    if True:
        for m, ks in (('f64', f64), ('real', real)):
            licensed = {(tuple(lhs), tuple(rhs)) for i, (lhs, rhs) in enumerate(mined)
                        if m in allowed.get(i, frozenset())}
            if not ks <= licensed:
                raise AssertionError(
                    f'triple invariant broken: {len(ks - licensed)} rules in rules_{m} '
                    f'are not licensed for that mode')
    if corpus & rej:
        raise AssertionError(
            f'triple invariant 3 broken: {len(corpus & rej)} rules are BOTH served and '
            f'rejected')
    if (corpus | rej) - mined_keys:
        raise AssertionError(
            f'triple invariant 3 broken: {len((corpus | rej) - mined_keys)} rules are '
            f'served or rejected but were never mined')
    for i, (lhs, rhs) in enumerate(mined):
        k = (tuple(lhs), tuple(rhs))
        want = allowed.get(i, frozenset())
        got = frozenset({m for m, ks in (('f64', f64), ('real', real)) if k in ks})
        # SUBSET, not equality: the per-mode prune legitimately removes a rule from a
        # file whose tier licenses it, when that mode's constructor already reaches the
        # endpoint. What must never happen is the other direction -- a rule in a file its
        # tier does NOT license.
        if not got <= want:
            raise AssertionError(
                f'triple invariant 4 broken: {" ".join(lhs)} -> {" ".join(rhs)} is in '
                f'{sorted(got)} but its tier licenses only {sorted(want) or "none"}')


class RuleMiner:
    """The dedicated miner (C23a step 2): 13 methods, engine access explicit.

    Stateless beyond the engine reference: mining reads ``engine._core`` /
    ``engine.operator_arity``, swaps rule sets only through the engine's
    copy-on-write ``_replace_rules``, and calls back into the engine's own
    ``prune_covered_rules`` / ``is_valid`` / fingerprint methods. One miner
    per process (D28): the module-level ``_MINE_LOCK`` single-flights every
    mine/certify entry point regardless of how many miners are constructed.
    """

    def __init__(self, engine: 'SimpliPyEngine') -> None:
        self.engine = engine

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
        assert self.engine._core is not None
        n_rows = X_confirm.shape[0]
        x_flat = X_confirm.flatten(order='C').tolist()
        confirmed = []
        for source, target in found:
            # blake2b, not hash(): PYTHONHASHSEED randomises str hashing per process, which
            # would make the confirm stage irreproducible across runs.
            key = f'{" ".join(source)}->{" ".join(target)}'.encode()
            rule_seed = seed + int.from_bytes(hashlib.blake2b(key, digest_size=7).digest(), 'little')
            # THE BOUND MUST ADMIT THE PAIR IT IS ASKED ABOUT (2026-08-18). `find_rule`'s
            # second argument is the MINE's candidate-length bound: the scan runs
            # `1..min(simplified_length, max_target + 1)` (rust/worker.rs), so `len(source)`
            # silently skips every candidate as long as the source -- and the candidate
            # list here holds exactly ONE entry, the target being re-verified. That made
            # "not confirmed" indistinguishable from "never looked at": a proposal hint
            # the serve ordering accepts came back refused for a reason with no numerics
            # in it. `max(...)` never LOWERS the old bound -- for a strictly shorter target
            # `len(target) + 1 <= len(source)`, so every pair that confirms today keeps its
            # verdict, mined pairs included (the mine's own bound, untouched, makes every
            # mined target strictly shorter than its source).
            scan_len = max(len(source), len(target) + 1)
            result = self.engine._core.find_rule(
                list(source), scan_len, None, [list(target)], list(dummy_variables),
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
        assert self.engine._core is not None

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
            candidates, list(dummy_variables), X_data.flatten(order='C').tolist(), X_data.shape[0])
        if verbose:
            print(f'Candidate library: {library.n_candidates:,} candidates'
                  f' ({library.n_filtered:,} var-free filtered)')

        # Existing rules (empty after reset_rules=True) join the Kruskal pruning and the
        # final deduplicated set, like the pure-Python path.
        rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.engine.simplification_rules]

        # --- The post-pass: what turns a raw mined rule set into a CELL ARTIFACT ---
        # A closure, because the ladder runs it MORE THAN ONCE. A (7,4) climb passes through
        # the (5,4) and (6,4) cells on its way up and each of those is a publishable artifact
        # in its own right (see `snapshot_at`). Running the IDENTICAL block for every cell is
        # what makes a snapshot equal to a one-shot mine of that cell; a second copy of these
        # stages would be free to drift from this one.
        def _finalize(mined: list, out_path: str | None, prov: dict | None) -> list:
            self.engine._replace_rules(list(mined))
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
                         for lhs, rhs in self.engine.simplification_rules}
                for identity in _watch:
                    if identity not in _drops and identity not in alive:
                        _drops[identity] = (stage, reason_of(identity))
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
            if symbolic_gate and not interrupted() and self.engine.simplification_rules:
                from .verify import CONST_CHANNEL_DETAIL, FATAL_BUCKETS, verify_ruleset
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
                _served = self.engine._core.ac_served_rules()
                _rep = verify_ruleset([[list(lhs), list(rhs)] for lhs, rhs, _ in _served])
                _detail = _rep['detail'] if isinstance(_rep, dict) else _rep
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
                    _src_lhs, _src_rhs = self.engine.simplification_rules[
                        _served[d['idx']][2]]
                    _pair = (tuple(_src_lhs), tuple(_src_rhs))
                    _tier_of = d.get('tier') or 'reject'
                    if d.get('detail') == CONST_CHANNEL_DETAIL:
                        _const_channel.add(_pair)
                    elif _tier_of != 'reject':
                        # THE TIER HAS SOMEWHERE TO PUT IT, so the gate must not delete
                        # it. Every fatal bucket used to drop, which was right when
                        # `rules.json` was the only destination: a rule the contract
                        # convicts had no home. Under the triple two of those buckets
                        # DO have one -- ENGINE-MISALIGN is `real` (true, the deployed
                        # engine diverges) and an f64-realised KILL is `f64` (false,
                        # bit-exact in the deployed algebra) -- and deleting them here
                        # would leave `_route_triple` nothing to route, making
                        # rules_real.json and rules.json identical and silently voiding
                        # the whole split. This is the hazard the roadmap flagged as
                        # "arming the deployed check without the tier makes the next
                        # mine DELETE sound rules"; it is closed HERE, not downstream.
                        # Fail-closed is UNCHANGED for everything else: NO-WITNESS,
                        # UNSUPPORTED-SHAPE, JUDGE-TIMEOUT and UNRESOLVED-COVERAGE carry
                        # no tier at all, so they read `reject` and still drop.
                        _routed.setdefault(_tier_of, set()).add(_pair)
                    else:
                        _fatal[d['verdict']].add(_pair)
                _drop = set().union(*_fatal.values())
                if _drop:
                    self.engine._replace_rules([
                        (lhs, rhs) for lhs, rhs in self.engine.simplification_rules
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
                          f'{len(self.engine.simplification_rules)} remain')
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
                        'census': ({b: len(ids) for b, ids in _rep['buckets'].items()}
                                   if isinstance(_rep, dict) else None),
                        'kept': len(self.engine.simplification_rules)}
            if prune == 'covered':
                self.engine.prune_covered_rules(verbose=verbose)
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
                kept, promotion_report = promote(self.engine.simplification_rules, self.engine)
                self.engine._replace_rules(kept)
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
                    self.engine.prune_covered_rules(verbose=verbose)
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
            if _watch:
                _reconcile_proposal_census(
                    _outcomes, _trail, _watch, _drops,
                    self.engine.simplification_rules, dummy_variables)
            if out_path is not None:
                # THE ARTIFACT IS A TRIPLE (owner ruling 2026-08-19): a mine run is valid
                # only if all three sets fall out of it, so they are routed and their
                # invariants checked BEFORE a byte is written. `rules.json` keeps its name
                # and carries the f64 set -- it is the default mode, and every config and
                # loader that predates the ruling goes on reading it unchanged.
                _triple, _rejected, _route_meta = _route_triple(self.engine, verbose=verbose)
                _base, _ext = os.path.splitext(out_path)
                for _mode, _path in (('f64', out_path),
                                     ('real', f'{_base}_real{_ext}'),
                                     ('corpus', f'{_base}_corpus{_ext}')):
                    with open(_path, 'w') as file:
                        json.dump(_triple[_mode], file, indent=4)
                if prov is not None:
                    # The REJECTS are recorded, never silently absent (owner ruling): a
                    # rule the mine found and then licensed nowhere is evidence, and the
                    # drop census is where it lives.
                    prov['triple'] = {
                        'counts': {m: len(rs) for m, rs in _triple.items()},
                        'rejected': sorted(_rejected),
                        # How many rules each file OMITS because that mode's own
                        # constructor already performs them. Without this the sidecar
                        # cannot explain why the files are smaller than the gate's kept
                        # count, and "smaller than expected" is indistinguishable from
                        # "something was lost".
                        'preempted': _route_meta['preempted'],
                        'files': {'f64': os.path.basename(out_path),
                                  'real': os.path.basename(f'{_base}_real{_ext}'),
                                  'corpus': os.path.basename(f'{_base}_corpus{_ext}')}}
                # The sidecar covers the TRIPLE, not one third of it (owner ruling:
                # "the provenance sidecar covers the triple, the manifest entry lists
                # three files as ONE artifact"), so its totals describe everything the
                # mine serves. The per-mode split lives in `prov['triple']['counts']`.
                self._write_provenance(out_path, prov, _triple['corpus'], final=True)
            return [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.engine.simplification_rules]

        snapshot_at = dict(snapshot_at or {})

        # SOUNDNESS PROVENANCE, first half (audit Tier-1 #4): the four default-ON
        # soundness kill-switches (ablation/repro escape hatches; `<var>=0` disables a
        # LAYER) ship recorded in the artifact -- a mine run with a layer off must say
        # so -- plus the node-budget override if any. Recorded up front so interrupted
        # mines carry it; the switch semantics mirror the rust side exactly
        # (on unless the variable is literally "0").
        _ivl_baseline = (self.engine._core.interval_horizon_misses(),
                         self.engine._core.interval_node_budget_misses(),
                         self.engine._core.interval_unanalyzable_misses())
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
                'ac_classes': None,  # filled per length as the climb proceeds
            }

        # Per-length AC-class census, recorded in the sidecar: how many enumerated
        # spellings collapsed to how many canonical classes. An artifact states what it
        # actually mined, and under representative-only certification that is CLASSES.
        ac_classes_per_length: dict[int, dict[str, int]] = {}

        # Sources: lengths up to max_source_pattern_length only (`construct_expressions`
        # over-produces a tail of longer expressions beyond the documented contract).
        for length in sorted(k for k in list(expressions_of_length) if k <= max_source_pattern_length):
            if interrupted():
                break
            self.engine._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in rules])
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
            _raw_sources = [list(expression) for expression in sorted(expressions_of_length[length])]
            _keys = self.engine._core.ac_canonical_keys(_raw_sources)
            _seen: set[tuple[str, ...]] = set()
            sources = []
            for _src, _key in zip(_raw_sources, _keys):
                if _key is None:
                    sources.append(_src)
                    continue
                _k = tuple(_key)
                if _k in _seen:
                    continue
                _seen.add(_k)
                sources.append(_src)
            n_ac_classed = len(_raw_sources) - len(sources)
            if verbose and n_ac_classed:
                print(f'Length {length}: {len(_raw_sources):,} sources -> {len(sources):,} AC classes '
                      f'({n_ac_classed:,} duplicate spellings skipped)')
            ac_classes_per_length[length] = {'enumerated': len(_raw_sources),
                                             'classes': len(sources)}
            if provenance is not None:
                provenance['ac_classes'] = {str(k): v for k, v in ac_classes_per_length.items()}
            del _raw_sources, _keys, _seen
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
                lambda: self.engine._core.mine_one_length(
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
                    dummy_variables, verbose=verbose, engine=self.engine)
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
                self.engine._replace_rules(rules)

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

        # Same bound as the library build above -- under the mu ruling an unset target cap
        # admits every enumerated length, so SAMPLING any of them thins the candidate library.
        max_candidate_length = (max_source_pattern_length if max_target_pattern_length is None
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
            vocabulary = set(leaf_nodes) | set(self.engine.operator_arity)
            for expression in draws:
                if (len(expression) != length or not set(expression) <= vocabulary
                        or not self.engine.is_valid(list(expression))):
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
            'core_build': _core_build,
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
        never by the reduction alone (a canon respell used to read as "covered" and
        silently discarded certifiable proposals, finding F2).

        Intended for LLM- or human-proposed identities: proposals only ever ADD source
        expressions, and every accepted pair passes the same symbolic gate the mine
        runs (:func:`~simplipy.verify.verify_ruleset` with the precision-stability
        discriminator; KILLs dropped, audit B3) -- so certified output is exactly as
        sound as mined rules.

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
            X_data.shape[0])
        if verbose:
            print(f'Candidate library: {library.n_candidates:,} candidates '
                  f'({library.n_filtered:,} var-free filtered)')

        vocabulary = set(leaf_nodes) | set(self.engine.operator_arity)
        certified: list = []
        confirm_min = max(1, round(min_informative * X_confirm.shape[0] / X_data.shape[0]))
        for index, (source, hint) in enumerate(zip(sources, hint_list)):
            if not _tokens_in_vocabulary(source, vocabulary) or not self.engine.is_valid(list(source)):
                continue
            # The engine's own result is the MARK TO BEAT, never a reason to skip
            # (finding F2): a source canon merely RESPELLS shorter used to read as
            # "already covered" and was never searched, losing e.g.
            # exp(t)*exp(t) -> exp(t+t) forever (the respelled form speaks tokens
            # outside the mining alphabet, so no later tier picks it up). The
            # serve-time ordering acceptance rides the candidate scan via `mark`.
            _, _, ac_out = self.engine._core.ac_judge(list(source), 48)
            target = self.engine._core.find_rule_lib(
                list(source), len(source), max_target_pattern_length, library,
                challenges=constants_fit_challenges, retries=constants_fit_retries,
                seed=mine_seed + (len(source) << 40) + index, rtol=rtol, atol=atol,
                min_informative=min_informative, mark=ac_out)
            certificate = 'minimal'
            if target is None and hint is not None:
                hint = [str(t) for t in hint]
                # NO TOKEN-LENGTH BAR on a proposal hint (owner ruling 2026-08-18:
                # "Only 'remove' the length gate for the LLM proposed rules"). The
                # retired conjunct was `len(hint) < len(source)` -- a RAW TOKEN COUNT,
                # which is not the engine's ordering and disagrees with it:
                # `1 - cos(2*x0) -> 2*sin(x0)^2` is 6 tokens against 6 while the target
                # sits strictly BELOW the engine's endpoint (the `-` spelling costs two
                # extra canonical nodes). Descent is decided below by the ordering, the
                # same predicate G7 and the serve-time `oriented` check apply. The MINE's
                # candidate-length bound is a separate thing and stays: a proposal skips
                # the search entirely, so this costs no mining time, while an unbounded
                # target length in the SEARCH is what would make e.g. a 5-5 mine
                # unaffordable.
                if (_tokens_in_vocabulary(hint, vocabulary)
                        and self.engine.is_valid(list(hint))
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
                        and self.engine._core.ac_ordered_below(list(hint), ac_out)
                        # THE LICENCE REGISTRY (C1.20, 2026-08-08): a hint that passes
                        # every bar above is still a MINT, and every mint channel
                        # consults the registry -- an a.e.-equal pole mirror or a
                        # special-eating Const hint passes vocabulary, ordering AND
                        # the numeric confirmation (measure-zero differences are
                        # invisible to sampling), and was refused nowhere on this
                        # path before the registry.
                        and self.engine._core.registry_mint_refusal(
                            list(source), ac_out, list(hint), dummy_variables) is None
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
        # SYMBOLIC GATE, matching the mining path (`_finalize`; audit B3, fatal set B7).
        # Every numeric acceptance above is tolerance-bounded, and the impostor family
        # lives BELOW any usable tolerance: `tanh(exp(e)) -> 1` is false by a stable
        # 1.37e-13 and came back certified 'verified' before this gate existed -- while
        # the same pair, arriving through `find_rules(proposals=...)`, was killed. The
        # parity sentence above ("exactly as sound as mined rules") is earned here, not
        # assumed: every accepted pair is re-judged by the independent symbolic
        # verifier. What "refused" MEANS changed with the triple, and it has to change
        # here in lockstep with `_finalize` or the parity claim is false: a pair whose
        # tier owns it -- ENGINE-MISALIGN is `real`, an f64-realised KILL is `f64` --
        # is KEPT for routing, exactly as a mined pair is, and only a `reject` is
        # dropped. Fail-closed is untouched for the verdicts that carry no tier at all
        # (NO-WITNESS / UNSUPPORTED-SHAPE / JUDGE-TIMEOUT / UNRESOLVED-COVERAGE): they
        # read `reject` and still drop, because an unjudged certificate is an unearned
        # one. TOLERATED passes, exactly as at the mining gate.
        if certified:
            from .verify import CONST_CHANNEL_DETAIL, FATAL_BUCKETS, verify_ruleset
            _rep = verify_ruleset([[list(s), list(t)] for s, t, _ in certified])
            _detail = _rep['detail'] if isinstance(_rep, dict) else _rep
            # detail entries carry lhs/rhs as SPACE-JOINED strings; the Const-channel
            # sentinel is exempt exactly as at the mining gate (D36).
            _drop = {(tuple(d['lhs'].split()), tuple(d['rhs'].split())): d['verdict']
                     for d in _detail if d.get('verdict') in FATAL_BUCKETS
                     and d.get('detail') != CONST_CHANNEL_DETAIL
                     and (d.get('tier') or 'reject') == 'reject'}
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
        identical to an earlier certified proposal and not shorter) /
        ``certified_then_dropped``. The last one is filled in LATER, by
        :func:`_reconcile_proposal_census`: certification runs upstream of the symbolic
        gate, the covered-prunes and the sort promotion, so the tally this method returns
        is provisional and the reconciliation moves every certified pair the artifact
        does not contain out of ``certified`` (F94). The key is present even when zero.

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
            if not _tokens_in_vocabulary(source, vocabulary) or not self.engine.is_valid(list(source)):
                counts['rejected'] += 1
                record(source, None, 'rejected', 'vocabulary')
                continue
            # The engine's own result is the MARK TO BEAT, never a reason to skip
            # (finding F2): the serve-time ordering acceptance rides the candidate
            # scan via `mark`, so 'already_covered' below is a SEARCH-VERIFIED
            # verdict ("the engine reduces it and nothing expressible beats its
            # result"), not a guess from a spelling shrink.
            _, _, ac_out = self.engine._core.ac_judge(list(source), 48)
            # "The engine already reduces it" is judged in the serve ordering (ONE
            # coverage ordering, everywhere): the walk strictly descends, or the
            # result is atomic (nothing mintable sits strictly below a single-token
            # state -- canon-owned collapses like `+ x0 0` land here). A respell --
            # same state, shorter spelling -- is neither: if the search also comes
            # back empty-handed, the honest verdict below is 'rejected', never a
            # coverage claim nothing backs.
            engine_reduces = (len(ac_out) == 1
                              or self.engine._core.ac_ordered_below(ac_out, list(source)))
            simplified_length = len(source)
            # Content-derived per-proposal seed offset (blake2b, not hash(): PYTHONHASHSEED
            # randomises str hashing per process; same policy as _confirm_mined_rules).
            offset = int.from_bytes(
                hashlib.blake2b(' '.join(source).encode(), digest_size=7).digest(), 'little')
            target = self.engine._core.find_rule_lib(
                list(source), simplified_length, max_target_pattern_length, library,
                challenges=constants_fit_challenges, retries=constants_fit_retries,
                seed=mine_seed + offset, rtol=rtol, atol=atol,
                min_informative=min_informative, mark=ac_out)
            certificate = 'minimal'
            if target is None and hint is not None:
                hint_tokens = [str(token) for token in hint]
                # NO TOKEN-LENGTH BAR (same ruling, same reasoning as certify_rules'
                # hint arm above -- one certification chain, both entries).
                if (_tokens_in_vocabulary(hint_tokens, vocabulary)
                        and self.engine.is_valid(list(hint_tokens))
                        and not violates_wildcard_multiplicity(list(source), list(hint_tokens))
                        # the Const-count invariant (same gate as certify_rules' hint
                        # arm): a count-increasing hint is dead on arrival at translation
                        and hint_tokens.count('<constant>') <= list(source).count('<constant>')
                        # the hint too must sit strictly below the engine's own
                        # result, or the minted rule is dead on arrival
                        and self.engine._core.ac_ordered_below(list(hint_tokens), ac_out)
                        # THE LICENCE REGISTRY (C1.20, 2026-08-08): same gate as the
                        # certify_rules hint arm -- every mint channel consults it.
                        and self.engine._core.registry_mint_refusal(
                            list(source), ac_out, list(hint_tokens), dummy_variables) is None
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
        before = [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.engine.simplification_rules]
        seen: dict[tuple, int] = {}
        for key, (_, canon_target) in _dedup_keyed_rules(before, dummy_variables, self.engine):
            if key not in seen or len(canon_target) < seen[key]:
                seen[key] = len(canon_target)
        for idx, (key, (_, canon_target)) in zip(
                certified_trail_idx,
                _dedup_keyed_rules(certified_pairs, dummy_variables, self.engine)):
            if key in seen and len(canon_target) >= seen[key]:
                counts['duplicate'] += 1
                trail[idx].update(verdict='duplicate', stage='merge')
            else:
                counts['certified'] += 1
                seen[key] = len(canon_target)
        if certified_pairs:
            # The SAME merge path as mined rules: shortest target per canonical source.
            self.engine._replace_rules(deduplicate_rules(
                before + certified_pairs, dummy_variables, verbose=verbose,
                engine=self.engine))
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
            confirm=confirm,
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


def assert_corpus_dominates(engine: Any, expressions: Any) -> list:
    """`corpus` must be at least as capable as `f64` and as `real`, on every expression.

    THE REPLACEMENT for `rules_corpus == rules_f64 UNION rules_real`, which was retired
    once folding made the three modes live in different canonical worlds (owner ruling
    2026-08-20). That set identity was a proxy for this, and a bad one: it measured
    spelling overlap, so it read `acos 1 -> 0` as "missing from corpus" when corpus
    performs it -- and performs all 15 such rules -- through its own constructor.

    Stated as behaviour it is checkable and it is what anyone actually wants: for every
    expression, corpus's endpoint costs no more under the serve ordering than either
    other mode's. Returns the counterexamples, empty when the property holds.
    """
    from .engine import Mode
    bad = []
    for expr in expressions:
        try:
            mu = {m: engine._core.ac_complexity(engine.simplify(list(expr), mode=m))
                  for m in (Mode.f64, Mode.real, Mode.corpus)}
        except Exception:
            continue
        if None in mu.values():
            continue
        if mu[Mode.corpus] > min(mu[Mode.f64], mu[Mode.real]):
            bad.append({'expression': list(expr),
                        'mu': {m.name: v for m, v in mu.items()}})
    return bad
