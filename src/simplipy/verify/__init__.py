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
    """Judge a single ``lhs -> rhs`` rule at its symbolic trigger points.

    Returns a dict with ``verdict`` (CERTIFIED / TOLERATED / KILL / ENGINE-MISALIGN /
    NO-WITNESS / UNSUPPORTED-SHAPE) and the supporting witness points / measure.
    """
    return _contract.judge_rule(list(lhs), list(rhs), deployed_check=deployed_check)


#: Which judged tiers each mode's rule set is allowed to contain: the tiers whose rules hold in
#: the mode (`_contract._tier`). `real` holds what holds in real (`real-only` among it: a `$`
#: slot real owes nothing at 0, where permissive's real-valued guarantee reaches); `f64` what
#: holds in float64 (`f64-only` among it: float64's own value, off real's beyond float
#: resolution); `permissive` what holds in permissive -- its own extensions (`permissive`:
#: `exp(log x) -> x`) and the rules within float resolution of real's values (P3, `f64`) among
#: it; no mode may carry `reject`.
MODE_TIERS: dict[str, frozenset] = {
    'f64': frozenset({'core', 'f64', 'f64-only'}),
    'real': frozenset({'core', 'real', 'real-only'}),
    'permissive': frozenset({'core', 'f64', 'real', 'permissive'}),
}


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
#: is a pre-filter of truth, not a policy, and licenses nothing either way.
_LOAD_POLICY = {'multi-constant-lhs': 'constant_count', 'const-count-increase': 'constant_count',
                'not-ordered-below': 'ordering', 'engine-owned-law': 'engine_owned',
                'subsumed': 'engine_owned', 'catch-all-lhs': 'well_formed',
                'unbound-rhs-wildcard': 'well_formed', 'unparseable-side': 'well_formed'}

#: A mode's name as the engine's rule sets are named.
_RULE_MODE = {'f64': 'default', 'real': 'real', 'permissive': 'permissive'}


def policy_refusals(lhs: list[str], rhs: list[str], mode: str, engine: Any,
                    max_source: int | None = None) -> list[str]:
    """The named policies (``POLICIES``) that refuse ``lhs -> rhs`` in ``mode``, each read
    from its one place: evidence (4.1, the core's certified cells in the mode's values,
    `_core.table_evidence`); the load's gates (4.2 constant count and the one-constant
    stopgap, 4.5 ordering, 4.8 the engine's own laws, well-formedness: `load_refusal`); the
    ground-composite predicate (4.3); the mine's cell (4.4, with ``max_source``); the
    registry's special absorption and constant introduction (4.6, 4.2); the duplication
    predicate (4.7); and the ground facts the engine's fold performs (4.8, the preemption
    gate). ``engine`` is a :class:`~simplipy.SimpliPyEngine` holding the operators."""
    from .. import _core  # type: ignore[attr-defined]
    from ..mining import _in_cell, _preempted_by
    lhs, rhs = list(lhs), list(rhs)
    rule_mode = _RULE_MODE[mode]
    out: list[str] = []
    slots = sorted({'?' + t[1:] if t in ('x0', 'x1', 'x2') else t
                    for t in lhs + rhs if _contract.SLOT_RE.match(t) or t in ('x0', 'x1', 'x2')})
    spelling = {s: f'x{i}' for i, s in enumerate(slots)}
    if _core.table_evidence(_contract._spelled(lhs, spelling), dict(_contract.ARITY),
                            mode == 'f64') is not True:
        out.append('evidence')
    load = engine._core.load_refusal(lhs, rhs, rule_mode=rule_mode)
    if load in _LOAD_POLICY:
        out.append(_LOAD_POLICY[load])
    if _core.policy_ground_composite(rhs):
        out.append('ground_composite')
    if not _in_cell({'completion': [lhs, rhs]}, max_source):
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


def licensed(lhs: list[str], rhs: list[str], mode: str, engine: Any, *,
             max_source: int | None = None, detail: dict | None = None) -> dict:
    """``licensed(L, R, mode)`` (the specification's 3.4): the rule holds in the mode -- its
    judged tier is one ``MODE_TIERS`` gives the mode, so permissive also takes what holds in
    f64 -- and no named policy refuses it (:func:`policy_refusals`). ``detail`` is the judge's
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
    refused = policy_refusals(lhs, rhs, mode, engine, max_source)
    return {'licensed': holds and not refused, 'holds': holds, 'tier': tier,
            'refused_by': refused}


def verify_ruleset(rules: list | str, *, mode: str | None = None,
                   report_path: str | None = None,
                   build_path: str | None = None, judge_timeout_s: int = 30) -> dict:
    """Gate a whole rule set: judge every rule at its own trigger points.

    ``rules``: a list of ``[lhs, rhs]`` token-list pairs, or a path to such a JSON file.
    ``mode``: which mode's set this is -- ``'f64'`` (also ``'default'``), ``'real'``,
    ``'permissive'`` or a :class:`~simplipy.Mode` member. See below; ``None`` keeps the
    mode-less meaning.
    ``report_path``: optional path to dump the full per-rule report.
    ``build_path``: optional path to write the kept set (CERTIFIED + TOLERATED).
    ``judge_timeout_s``: per-rule wall-clock cap (a rule that exceeds it is bucketed
    JUDGE-TIMEOUT rather than blocking the sweep).

    CLEANLINESS IS PER MODE. With one rule set per mode, "no fatal bucket" is the wrong
    test in both directions. `atanh(tanh t) -> t` is ENGINE-MISALIGN and belongs in
    ``rules_real.json``, where it is perfectly clean -- calling that file dirty would
    condemn the rules the split exists to keep. And a rule that is f64-realised but
    mathematically false is clean for ``rules_f64.json`` and NOT for ``rules_real.json``,
    which a bucket count cannot express at all.

    So with ``mode`` given, ``is_clean`` means *every rule's judged tier is one this mode
    licenses* (``MODE_TIERS``) -- a rule can be clean for one file and dirty for another,
    which is the whole point. ``report['offenders']`` lists the ones that are not, with
    their tier.

    With ``mode=None`` only CERTIFIED/TOLERATED are clean, the meaning before rule sets
    were split by mode. It is NOT the right gate for a shipped triple; use
    :func:`verify_triple`.

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
    offenders = [{'idx': d.get('idx'), 'lhs': d.get('lhs'), 'rhs': d.get('rhs'),
                  'verdict': d.get('verdict'), 'tier': d.get('tier')}
                 for d in report.get('detail', [])
                 if effective_tier(d) not in allowed]
    report['mode'] = mode
    report['offenders'] = offenders
    report['is_clean'] = not offenders
    report['exit_code'] = 0 if not offenders else 1
    return report


def verify_triple(f64_rules: list | str, real_rules: list | str,
                  permissive_rules: list | str, *, engine_config: str | None = None,
                  corpus_rows: list | None = None, judge_timeout_s: int = 30) -> dict:
    """Gate a whole TRIPLE: each file against its own mode, plus permissive capability.

    A mine run is valid only if all three sets fall out of it, so the artifact is verified
    as one thing. Three per-file sweeps answer "is every rule in this file licensed for
    this mode".

    WHAT IS *NOT* CHECKED, AND WHY. An earlier version required
    ``rules_permissive == rules_f64 UNION rules_real``. That is wrong once folding is
    mode-dependent, because the three modes then reduce in different canonical worlds and
    each file omits what its OWN constructor already performs -- so set overlap measures
    SPELLING, not capability. On the 0.14.0 artifact the union exceeds the permissive
    file by 362 rules and permissive performs every one of them; the identity would report a
    correct artifact as dirty.

    The property it stood proxy for is behavioural: **permissive is at least as capable as
    either other mode**. Pass ``engine_config`` (and optionally ``corpus_rows``) to check
    it -- the report then carries ``permissive_dominance``. The check runs on an engine
    built from the config's operators and the three rule sets given here, never the rule
    files the config names, and prices every answer by
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
    for mode, rules in (('f64', f64_rules), ('real', real_rules), ('permissive', permissive_rules)):
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
                f'verify_triple: permissive dominance was NOT checked -- {reason}. '
                'is_clean does not cover it.', UserWarning)
            dominance = {'checked': False, 'reason': reason}
        else:
            from ..engine import SimpliPyEngine
            from ..io import load_config
            from ..mining import assert_permissive_dominates
            # The SUBJECT is the triple passed in, not the files the config names: the
            # config contributes its operators (the deployed semantics) and nothing else.
            engine = SimpliPyEngine(operators=load_config(engine_config)['operators'],
                                    rules=_load(f64_rules), rules_real=_load(real_rules),
                                    rules_permissive=_load(permissive_rules))
            violations = assert_permissive_dominates(engine, rows)
            dominance = {'checked': True, 'violations': violations}
            if violations:
                problems.append(
                    f'permissive is not the most capable mode on {len(violations)} '
                    f'rows, e.g. {violations[0]}')

    return {'is_clean': all(x['is_clean'] for x in reports.values()) and not problems,
            'modes': reports, 'relationships': problems,
            'permissive_dominance': dominance}


def default_corpus() -> list[list[str]]:
    """The release gate's expression corpus: 400 prefix-token expressions.

    Shipped with the package (``simplipy/verify/data/raw_skeletons_nv.json``), so the
    gate sweeps the same rows in a wheel install as in a checkout. :func:`verify_triple`
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


__all__ = ['FATAL_BUCKETS', 'MODE_TIERS', 'POLICIES', 'default_corpus',
           'effective_tier', 'licensed', 'policy_refusals', 'verify_rule',
           'verify_ruleset', 'verify_triple', 'monitor_ruleset', 'selftest']
