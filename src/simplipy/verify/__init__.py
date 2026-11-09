"""Soundness verification for rule sets: the rule judge.

This subpackage owns the criterion a rule must meet (`holds` per mode) and what a mode's file
may hold (`licensed`); the miner's own certification is a pre-filter that may only refuse what
the judge refuses. Every value the judge compares is THE TABLE's, each mode's values defined
once in the compiled core; the judge's own part is its procedure. It judges a rule set two
ways:

- :func:`verify_ruleset` (the GATE) judges every rule at its own symbolic trigger points
  in THE TABLE's values (the core's one definition of each mode's values), classifying each
  into nine buckets (CERTIFIED / TOLERATED / KILL / ENGINE-MISALIGN / NO-WITNESS /
  UNRESOLVED-COVERAGE / UNDECIDED-MEASURE / UNSUPPORTED-SHAPE / JUDGE-TIMEOUT). Coverage is
  100% of the rule set by construction.

- :func:`licensed` is the criterion a mode's file holds: the rule holds in the mode, and no
  named policy (evidence, constant count, ground composites, ordering, ...) refuses it.

- :func:`monitor_ruleset` (the MONITOR) runs the *deployed* engine over an
  adversarial-plus-sampled corpus and judges every input->output rewrite as the judge judges
  a rule, attributing any violation to the rule set or to the engine. It complements the
  gate: what the rules do together on whole expressions rather than each rule alone.

Both carry poison self-tests that must pass before they are trusted. Neither has an
evaluator of its own: every value either compares is the core's.
"""
import json
import os
import warnings
from typing import Any

from . import _contract, _gate, _monitor

#: Every bucket that is NOT sound to ship: all verdicts but CERTIFIED / TOLERATED.
#: THE single source of truth for "fatal" -- consumed by this module's
#: ``is_clean``, by the mining gate in ``engine._finalize`` and by the
#: ``certify_rules`` gate, so the three sites cannot drift apart. A rule the judge
#: cannot evaluate (NO-WITNESS / UNSUPPORTED-SHAPE / JUDGE-TIMEOUT) is exactly as
#: unshippable as a KILL: the second authority never passed it.
FATAL_BUCKETS = ('KILL', 'ENGINE-MISALIGN', 'NO-WITNESS', 'UNRESOLVED-COVERAGE',
                 'UNDECIDED-MEASURE', 'UNSUPPORTED-SHAPE', 'JUDGE-TIMEOUT')


def _load(rules: list | str) -> list:
    """Accept a rule list ([lhs, rhs] pairs) or a path to a JSON rule file."""
    if isinstance(rules, str):
        return json.load(open(rules))
    return rules


def verify_rule(lhs: list[str], rhs: list[str], deployed_check: bool = True) -> dict:
    """Judge a single ``lhs -> rhs`` rule at its symbolic trigger points, each mode on its own
    values: ``f64`` on float64's, where its two sides agree (F6, the true value accepted at a
    range event) or the rule is true in the reals and they agree in kind at every point
    (``_contract.strict_reading``); ``permissive`` on the reals' values at float resolution, and
    wherever ``f64`` holds. Variables are read whatever they are called.

    Returns a dict with the verdict per mode -- ``realised`` (``f64``'s reading), ``permissive``
    (``permissive``'s, which includes ``f64``'s; ``permissive_own`` is its own criterion's alone),
    its ``tier`` (the modes it holds in) and its
    ``basis`` per mode: ``'proved'`` (an exact route decides the rule over its whole domain,
    ``exact``), ``'evidence'`` (it holds at every point the judge read) or ``'refused'`` -- and,
    as a diagnostic that decides no mode, the reals' outcome ``verdict`` (CERTIFIED / TOLERATED
    / KILL / ENGINE-MISALIGN / NO-WITNESS / UNRESOLVED-COVERAGE / UNDECIDED-MEASURE /
    UNSUPPORTED-SHAPE) with the supporting witness points and measure.
    """
    return _contract.judge_rule(list(lhs), list(rhs), deployed_check=deployed_check)


#: The tiers (`_contract.TIERS`, the one registry): each tier names the modes a judged rule
#: holds in. Permissive includes f64: `core` holds in f64, and so in permissive (float64's own
#: value included, `floor(tanh(40)) -> 1`); `permissive` in permissive alone, by its own
#: criterion judged at the `_` sort its matcher binds (its extensions, `exp(log x) -> x`, and the
#: rules within float resolution of the reals' values); `reject` in no mode.
TIERS: dict[str, tuple] = dict(_contract.TIERS)

#: Which judged tiers each mode's rule set is allowed to contain: the tiers that hold in the
#: mode, read off `TIERS`.
MODE_TIERS: dict[str, frozenset] = {
    mode: frozenset(tier for tier, modes in TIERS.items() if mode in modes)
    for mode in _contract.MODES}


def tier_table() -> str:
    """The tier table of the verify guide, rendered from the registry (`TIERS`): one row per
    tier, one column per mode saying whether the mode's file may hold it, and its meaning."""
    head = ['tier'] + [f'`{mode}`' for mode in _contract.MODES] + ['meaning']
    rows = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for tier, modes in TIERS.items():
        cells = [f'`{tier}`'] + ['yes' if mode in modes else 'no' for mode in _contract.MODES]
        rows.append('| ' + ' | '.join(cells + [_contract.TIER_MEANINGS[tier]]) + ' |')
    return '\n'.join(rows)


def effective_tier(detail: dict) -> str:
    """A judged rule's EFFECTIVE tier -- what ``MODE_TIERS`` licenses against. A verdict that
    carries no tier reads ``reject`` (fail-closed): a rule the judge could not decide ships in
    no mode."""
    return detail.get('tier') or 'reject'


#: The named policies of the specification's section 4 that :func:`licensed` applies: which
#: true rules the project wants. Each is defined in one place and read from there.
POLICIES = ('evidence', 'constant_count', 'ground_composite', 'cell_bound', 'ordering',
            'specials', 'duplication', 'engine_owned', 'well_formed')

#: The load's refusals (`Engine.load_refusal`) by the policy each applies. `pole-class-change`
#: is a pre-filter of truth, not a policy, and licenses nothing either way. `catch-all-lhs` is a
#: left side the canon leaves no composite: written as one, it is what the engine folds itself
#: (`1/0 -> nan`, `C - C -> 0`: `engine_owned`), and only a written leaf is ill-formed.
_LOAD_POLICY = {'const-count-increase': 'constant_count', 'ground-composite': 'ground_composite',
                'not-ordered-below': 'ordering', 'engine-owned-law': 'engine_owned',
                'subsumed': 'engine_owned', 'catch-all-lhs': 'well_formed',
                'unbound-rhs-wildcard': 'well_formed', 'unparseable-side': 'well_formed'}

#: A mode's name as the engine's rule sets are named.
_RULE_MODE = {'f64': 'default', 'permissive': 'permissive'}


def load_policy_refusal(lhs: list[str], rhs: list[str], mode: str, engine: Any,
                        canon: str | None = None) -> str | None:
    """The named policy (``POLICIES``) by which the LOAD refuses ``lhs -> rhs`` in ``mode``, or
    ``None``: the load's own predicate (`Engine.load_refusal`, the gates every rule file passes
    when it is loaded), which applies the specification's 4.11 -- the constant count (4.2), the
    ground composite (4.3), the ordering (4.5), what the engine does itself (4.8), and
    well-formedness. The load, :func:`licensed` and :func:`verify_ruleset` read it here, so the
    three agree. It checks policies, never truth: `sin(x0) -> x0` passes it. ``canon`` is the
    canon the rule is read in (``'legacy'``, ``'f64'`` or ``'permissive'``; ``None``, the
    mode's own), as the load reads a file in the canon its sidecar records."""
    read_in = None if canon is None else _RULE_MODE.get(canon, canon)
    load = engine._core.load_refusal(list(lhs), list(rhs), rule_mode=_RULE_MODE[mode],
                                     canon=read_in)
    if load == 'catch-all-lhs' and len(lhs) > 1:
        return 'engine_owned'
    return _LOAD_POLICY.get(load) if load is not None else None


def policy_refusals(lhs: list[str], rhs: list[str], mode: str, engine: Any,
                    max_source: int | None = None,
                    operator_max_length: dict | None = None) -> list[str]:
    """The named policies (``POLICIES``) that refuse ``lhs -> rhs`` in ``mode``, each read
    from its one place: evidence (4.1, the core's certified cells in the mode's values,
    `_core.table_mode_evidence`; permissive includes f64, so its evidence is float64's or the
    reals'); the load's gates (4.2 constant count, 4.3 ground composite, 4.5
    ordering, 4.8 the engine's own laws, well-formedness: :func:`load_policy_refusal`); the
    mine's cell (4.4, with ``max_source`` and the mine's per-length operator set
    ``operator_max_length``); the
    registry's special absorption and constant introduction (4.6, 4.2); the duplication
    predicate (4.7); and the ground facts the engine's fold performs (4.8, the preemption
    gate). ``engine`` is a :class:`~simplipy.SimpliPyEngine` holding the operators."""
    from .. import _core  # type: ignore[attr-defined]
    from ..mining import _in_cell, _preempted_by
    lhs, rhs = list(lhs), list(rhs)
    out: list[str] = []
    if _core.table_mode_evidence(_evidence_spelling(lhs, rhs), dict(_contract.ARITY),
                                 _RULE_MODE[mode]) is not True:
        out.append('evidence')
    load = load_policy_refusal(lhs, rhs, mode, engine)
    if load is not None:
        out.append(load)
    if not _in_cell({'completion': [lhs, rhs]}, max_source, operator_max_length):
        out.append('cell_bound')
    registry = engine._core.registry_mint_refusal(lhs, lhs, rhs, [])
    if registry == 'special_absorption':
        out.append('specials')
    elif registry == 'const_introduction':
        out.append('constant_count')
    if _core.policy_duplicates(lhs, rhs):
        out.append('duplication')
    if _preempted_by(engine, lhs, rhs, mode):
        out.append('engine_owned')
    return [p for p in POLICIES if p in out]


def _evidence_spelling(lhs: list[str], rhs: list[str]) -> list[str]:
    """The left side as the evidence function reads it: each slot (a variable read as a dummy
    `x<k>`, and that as `?<k>`, as the judge reads it) spelled as the interval layer's dummy,
    numbered over both sides."""
    lhs, rhs = _contract.variables_as_dummies(lhs, rhs)
    slots = sorted({key for t in list(lhs) + list(rhs) if (key := _contract.slot_key(t))})
    return _contract._spelled(list(lhs), {s: f'x{i}' for i, s in enumerate(slots)})


def _held_with_evidence(lhs: list[str], rhs: list[str], detail: dict, tier: str) -> bool:
    """Does permissive hold the rule in a reading that has evidence? Permissive includes f64, so
    a rule holds there where f64 holds it or by permissive's own criterion, and its evidence is
    float64's or the reals' (4.1): the evidence of a reading the rule holds in. A rule f64 holds
    needs float64's evidence, one held by permissive's own criterion the reals'.
    `tan(atan2(x0, 0)) -> nan` holds by permissive's own criterion, vacuously (the left side has
    no value in the reals off x0 = 0), and float64's evidence, where the rule is false, licenses
    nothing. Where only the tier is known (`core`), permissive's own reading is not, and is taken
    as not holding (fail closed)."""
    from .. import _core  # type: ignore[attr-defined]
    held_f64 = (detail['realised'] is True) if 'realised' in detail else tier == 'core'
    held_own = (detail['permissive_own'] is True) if 'permissive_own' in detail \
        else tier == 'permissive'
    spelled, arity = _evidence_spelling(lhs, rhs), dict(_contract.ARITY)
    return (held_f64 and _core.table_evidence(spelled, arity, True) is True) or \
        (held_own and _core.table_evidence(spelled, arity, False) is True)


def licensed(lhs: list[str], rhs: list[str], mode: str, engine: Any, *,
             max_source: int | None = None, operator_max_length: dict | None = None,
             detail: dict | None = None) -> dict:
    """``licensed(L, R, mode)`` (the specification's 3.4): the rule holds in the mode -- its
    judged tier is one ``MODE_TIERS`` gives the mode -- and no named policy refuses it
    (:func:`policy_refusals`). In permissive, which includes f64, the rule's evidence is that
    of a reading it holds in (float64's for a rule f64 holds, the reals' for one permissive's own
    criterion holds, `_held_with_evidence`). ``detail`` is the judge's
    verdict on the rule when the caller has it. Returns ``{'licensed', 'holds', 'tier',
    'refused_by'}``: ``asin(1) -> acos(0)`` holds in every mode and is refused by
    ``ground_composite``; ``exp(tan(asin(1))) -> inf`` holds in f64 and is refused by
    ``evidence``."""
    if mode not in MODE_TIERS:
        from ..engine import _coerce_mode
        mode = _coerce_mode(mode).name
    detail = detail if detail is not None else verify_rule(lhs, rhs)
    tier = effective_tier(detail)
    holds = tier in MODE_TIERS[mode]
    refused = policy_refusals(lhs, rhs, mode, engine, max_source, operator_max_length)
    if mode == 'permissive' and holds and 'evidence' not in refused \
            and not _held_with_evidence(lhs, rhs, detail, tier):
        refused = [p for p in POLICIES if p in refused or p == 'evidence']
    return {'licensed': holds and not refused, 'holds': holds, 'tier': tier,
            'refused_by': refused}


def _vocabulary_engine(names: tuple[str, ...] | None = None) -> Any:
    """A rule-less engine over the judge's vocabulary (`_contract.ARITY`, each operator realized
    by `simplipy.operators`), or over its operators among `names`: the operators
    :func:`verify_ruleset` reads the load's policies with when it is given no engine. Built once
    per operator set."""
    key = None if names is None else tuple(sorted(set(names)))
    if key not in _VOCABULARY_ENGINES:
        from .. import _core, operators
        from ..engine import SimpliPyEngine  # type: ignore[attr-defined]
        core = _core.core_serialization_ops()
        specs: dict[str, dict] = {}
        for name, arity in _contract.ARITY.items():
            if key is not None and name not in key:
                continue
            if name in core:
                specs[name] = {'realization': core[name]['realization'] or name, 'alias': [],
                               'arity': arity, 'precedence': core[name]['precedence'],
                               'commutative': name in ('+', '*')}
            elif name in operators.__all__:
                specs[name] = {'realization': f'simplipy.operators.{name}', 'alias': [],
                               'arity': arity, 'commutative': False}
        _VOCABULARY_ENGINES[key] = SimpliPyEngine(operators=specs, rules=[])
    return _VOCABULARY_ENGINES[key]


_VOCABULARY_ENGINES: dict = {}


def _operators_of_record(rules_path: str) -> tuple[str, ...] | None:
    """The operators the rule file at ``rules_path`` was mined over, as its provenance sidecar
    records them (``operators``), or None where it records none."""
    sidecar = rules_path + '.provenance.json'
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as fh:
        record = json.load(fh)
    names = record.get('operators') if isinstance(record, dict) else None
    return tuple(str(n) for n in names) if isinstance(names, list) else None


def verify_ruleset(rules: list | str, *, mode: str | None = None, engine: Any = None,
                   report_path: str | None = None,
                   build_path: str | None = None, judge_timeout_s: int = 30) -> dict:
    """Gate a whole rule set: judge every rule at its own trigger points.

    ``rules``: a list of ``[lhs, rhs]`` token-list pairs, or a path to such a JSON file.
    ``mode``: which mode's set this is -- ``'f64'`` (also ``'default'``), ``'permissive'``
    or a :class:`~simplipy.Mode` member. See below; ``None`` keeps the mode-less meaning.
    ``report_path``: optional path to dump the full per-rule report.
    ``build_path``: optional path to write the kept set (CERTIFIED + TOLERATED).
    ``judge_timeout_s``: per-rule wall-clock cap (a rule that exceeds it is bucketed
    JUDGE-TIMEOUT rather than blocking the sweep).

    CLEANLINESS IS PER MODE. With one rule set per mode, "no fatal bucket" is the wrong
    test in both directions. `atanh(tanh t) -> t` is ENGINE-MISALIGN and belongs in
    ``rules_permissive.json``, where it is perfectly clean -- calling that file dirty would
    condemn the rules the split exists to keep. And a rule that is f64-realised but
    false in the reals' values beyond float resolution is clean for ``rules_f64.json`` and NOT
    for ``rules_permissive.json``, which a bucket count cannot express at all.

    So with ``mode`` given, ``is_clean`` means *every rule's judged tier is one this mode
    licenses* (``MODE_TIERS``) *and the load's policies admit it* (:func:`load_policy_refusal`,
    the predicate the load applies) -- a rule can be clean for one file and dirty for another,
    which is the whole point, and `asin(1) -> acos(0)` holds in every mode and is clean in
    none (`ground_composite`). ``report['offenders']`` lists the ones that are not, with their
    tier and ``refused_by``. ``engine`` holds the operators the policies are read with (a
    :class:`~simplipy.SimpliPyEngine`); by default a rule-less engine over the operators the
    file's provenance sidecar records (``rules`` a path), else over the judge's vocabulary.

    With ``mode=None`` only the reals' CERTIFIED/TOLERATED are clean: a diagnostic that
    decides no mode. It is NOT the right gate for a shipped artifact; use
    :func:`verify_artifact`.

    ``report['buckets']`` maps each bucket name to the list of rule indices in it and
    ``report['exit_code']`` is 0 iff clean.
    """
    import tempfile
    import os
    if mode is not None:
        from ..engine import _coerce_mode
        mode = _coerce_mode(mode).name
    # Without a `report_path` the sweep's report is read back from a scratch directory
    # that is removed afterwards (it used to stay behind in the system temp dir).
    with tempfile.TemporaryDirectory() as scratch:
        tmp = report_path or os.path.join(scratch, 'gate_report.json')
        code = _gate.sweep(_load(rules), report_path=tmp, build_path=build_path,
                           judge_timeout_s=judge_timeout_s,
                           announce_report=report_path is not None)
        with open(tmp) as fh:
            report = json.load(fh)
    if mode is None:
        report['is_clean'] = not any(report['buckets'].get(k) for k in FATAL_BUCKETS)
        report['exit_code'] = code
        return report

    allowed = MODE_TIERS[mode]
    if engine is None:
        engine = _vocabulary_engine(_operators_of_record(rules) if isinstance(rules, str)
                                    else None)
    # a file is read in the canon it was certified in, as the load reads it
    from ..engine import _canon_of_record
    canon = _canon_of_record(rules) if isinstance(rules, str) else None
    offenders = []
    for d in report.get('detail', []):
        refused = load_policy_refusal(str(d.get('lhs')).split(), str(d.get('rhs')).split(),
                                      mode, engine, canon)
        if effective_tier(d) not in allowed or refused is not None:
            offenders.append({'idx': d.get('idx'), 'lhs': d.get('lhs'), 'rhs': d.get('rhs'),
                              'verdict': d.get('verdict'), 'tier': d.get('tier'),
                              'refused_by': [refused] if refused is not None else []})
    report['mode'] = mode
    report['offenders'] = offenders
    report['is_clean'] = not offenders
    report['exit_code'] = 0 if not offenders else 1
    return report


def verify_artifact(f64_rules: list | str, permissive_rules: list | str, *,
                    engine_config: str | None = None, corpus_rows: list | None = None,
                    judge_timeout_s: int = 30) -> dict:
    """Gate a whole ARTIFACT: each file against its own mode, plus permissive capability.

    A mine run is valid only if both sets fall out of it, so the artifact is verified as one
    thing. Two per-file sweeps answer "is every rule in this file licensed for this mode".

    WHAT IS *NOT* CHECKED, AND WHY: a set identity between the files. Folding is
    mode-dependent, so the modes reduce in different canonical worlds and each file omits what
    its OWN constructor already performs -- set overlap measures SPELLING, not capability.

    The property that matters is behavioural: **permissive is at least as capable as f64**.
    Pass ``engine_config`` (and optionally ``corpus_rows``) to check it -- the report then
    carries ``permissive_dominance``. The check runs on an engine built from the config's
    operators and the two rule sets given here, never the rule files the config names, and
    prices every answer by
    :func:`simplipy.mining.cross_mode_complexity`. A row on which a mode raises is a
    violation, reported with its error. Without an engine nothing can be said about the
    relationship between the files, and ``relationships`` stays empty rather than
    asserting something false.

    ``permissive_dominance`` CONTRACT. ``None`` when ``engine_config`` is not given
    (nothing was asked). The rows swept are ``corpus_rows``, or :func:`default_corpus`
    (shipped with the package) when it is ``None``. When rows were swept:
    ``{'checked': True, 'violations': [...]}``, dirty iff ``violations`` is non-empty.
    An explicitly empty ``corpus_rows`` sweeps nothing, and the report says so:
    ``{'checked': False, 'reason': '<why>'}`` plus a ``UserWarning``, never an empty
    violation list, which would read as a zero-violation sweep. ``is_clean`` is not
    affected by an unchecked dominance: unchecked means unknown -- not passing, and not
    failing either.

    Returns ``{'is_clean', 'modes': {mode: report}, 'relationships': [...],
    'permissive_dominance': ... }``.
    """
    reports = {}
    for mode, rules in (('f64', f64_rules), ('permissive', permissive_rules)):
        reports[mode] = verify_ruleset(rules, mode=mode, judge_timeout_s=judge_timeout_s)

    problems: list = []
    dominance: dict | None = None
    if engine_config is not None:
        rows = corpus_rows if corpus_rows is not None else default_corpus()
        if not rows:
            # UNCHECKED IS NOT A PASS: an empty violation list would read as "zero
            # violations".
            reason = 'corpus_rows was given but is empty'
            warnings.warn(
                f'verify_artifact: permissive dominance was NOT checked -- {reason}. '
                'is_clean does not cover it.', UserWarning)
            dominance = {'checked': False, 'reason': reason}
        else:
            from ..engine import SimpliPyEngine
            from ..io import load_config
            from ..mining import assert_permissive_dominates
            # The SUBJECT is the pair passed in, not the files the config names: the
            # config contributes its operators (the deployed semantics) and nothing else.
            engine = SimpliPyEngine(operators=load_config(engine_config)['operators'],
                                    rules=_load(f64_rules),
                                    rules_permissive=_load(permissive_rules))
            violations = assert_permissive_dominates(engine, rows)
            dominance = {'checked': True, 'violations': violations}
            if violations:
                problems.append(
                    f'permissive is less capable than f64 on {len(violations)} '
                    f'rows, e.g. {violations[0]}')

    return {'is_clean': all(x['is_clean'] for x in reports.values()) and not problems,
            'modes': reports, 'relationships': problems,
            'permissive_dominance': dominance}


def verify_triple(*args: Any, **kwargs: Any) -> dict:
    """Refused: the gate of the three-file artifact, whose middle file was the retired real
    mode's. An artifact is the f64 and the permissive file; :func:`verify_artifact` gates it."""
    from ..engine import _REAL_MODE_RETIRED
    raise ValueError(f'verify_triple gated the real set: {_REAL_MODE_RETIRED}; '
                     'verify_artifact(f64_rules, permissive_rules) gates an artifact')


def default_corpus() -> list[list[str]]:
    """The release gate's expression corpus: 400 prefix-token expressions.

    Shipped with the package (``simplipy/verify/data/raw_skeletons_nv.json``), so the
    gate sweeps the same rows in a wheel install as in a checkout. :func:`verify_artifact`
    sweeps it when no ``corpus_rows`` are given, and the test suites read it from here.
    """
    from importlib.resources import files
    rows: list[list[str]] = json.loads(
        files(__name__).joinpath('data', 'raw_skeletons_nv.json').read_text(encoding='utf-8'))
    return rows


def monitor_ruleset(rules: list | str, engine_config: str, *, corpus_n: int = 6000,
                    seed: int = 20260718, run_selftest: bool = False,
                    judge_timeout_s: int = 10, label: str = '') -> dict:
    """Sweep the deployed engine over a corpus and attribute any violation to a rule.

    ``rules``: a list of ``[lhs, rhs]`` pairs or a path to a JSON rule file.
    ``engine_config``: path to the engine config whose operator realizations define the
    deployed semantics the monitor judges against.
    ``run_selftest``: run the poison self-test first (raises if a known-unsound rule is
    not caught and attributed).

    Each rewrite is made in the engine's default mode (``f64``) and judged as the rule
    judge judges a rule in that mode. Returns a dict with ``violations``
    (artifact-attributed), ``native`` (pre-existing engine behavior, present even with an
    empty rule set), ``unscored`` (rewrites the judge cannot read or decide), ``changed`` and
    ``corpus`` counts.
    """
    return _monitor.monitor(_load(rules), engine_config, corpus_n=corpus_n, seed=seed,
                            run_selftest=run_selftest, judge_timeout_s=judge_timeout_s,
                            label=label)


def selftest() -> bool:
    """Run the gate + contract judge poison self-tests. Returns True iff both pass."""
    return _gate.selftest() and _contract.selftest(verbose=False)


__all__ = ['FATAL_BUCKETS', 'MODE_TIERS', 'POLICIES', 'TIERS', 'default_corpus', 'tier_table',
           'effective_tier', 'licensed', 'load_policy_refusal', 'policy_refusals', 'verify_rule',
           'verify_ruleset', 'verify_artifact', 'verify_triple', 'monitor_ruleset',
           'selftest']
