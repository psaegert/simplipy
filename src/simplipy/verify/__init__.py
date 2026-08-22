"""Independent soundness verification for rule sets.

This subpackage is a SECOND, independent soundness authority alongside the miner's own
certification. Where the miner certifies each rule as it is discovered (via the compiled
core), this package re-judges a *finished* rule set two ways:

- :func:`verify_ruleset` (the GATE) judges every rule at its own symbolic trigger points
  under an arbitrary-precision contract evaluator, classifying each into eight buckets
  (CERTIFIED / TOLERATED / KILL / ENGINE-MISALIGN / NO-WITNESS / UNRESOLVED-COVERAGE /
  UNSUPPORTED-SHAPE / JUDGE-TIMEOUT). Coverage is 100% of the rule set by construction.

- :func:`monitor_ruleset` (the MONITOR) runs the *deployed* engine over an
  adversarial-plus-sampled corpus and re-judges every input->output rewrite under an
  independent high-precision evaluator, attributing any deployed-value violation to the
  responsible rule. It complements the gate: a corpus sweep rather than a per-rule scan.

Both carry poison self-tests that must pass before they are trusted. Both are deliberately
implemented independently of the compiled core so they cross-check it rather than echo it.
"""
import json
import warnings

from . import _contract, _gate, _monitor

#: Every bucket that is NOT sound to ship: all verdicts but CERTIFIED / TOLERATED.
#: THE single source of truth for "fatal" (audit B7) -- consumed by this module's
#: ``is_clean``, by the mining gate in ``engine._finalize`` and by the
#: ``certify_rules`` gate, so the three sites cannot drift apart. A rule the judge
#: cannot evaluate (NO-WITNESS / UNSUPPORTED-SHAPE / JUDGE-TIMEOUT) is exactly as
#: unshippable as a KILL: the second authority never passed it.
FATAL_BUCKETS = ('KILL', 'ENGINE-MISALIGN', 'NO-WITNESS', 'UNRESOLVED-COVERAGE',
                 'UNSUPPORTED-SHAPE', 'JUDGE-TIMEOUT')

#: The one exemption from the fatal drop (B7/D36): the judge's own non-jurisdiction
#: sentinel for the multi-`<constant>` family (see `_contract.CONST_CHANNEL_DETAIL`).
CONST_CHANNEL_DETAIL = _contract.CONST_CHANNEL_DETAIL


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


#: Which judged tiers each mode's rule set is allowed to contain. `corpus` carries both
#: off-diagonal tiers because it is their union; no mode may carry `reject`.
MODE_TIERS: dict[str, frozenset] = {
    'f64': frozenset({'core', 'f64'}),
    'real': frozenset({'core', 'real'}),
    'corpus': frozenset({'core', 'f64', 'real'}),
}


def verify_ruleset(rules: list | str, *, mode: str | None = None,
                   report_path: str | None = None,
                   build_path: str | None = None, judge_timeout_s: int = 30) -> dict:
    """Gate a whole rule set: judge every rule at its own trigger points.

    ``rules``: a list of ``[lhs, rhs]`` token-list pairs, or a path to such a JSON file.
    ``mode``: which mode's set this is -- ``'f64'``, ``'real'`` or ``'corpus'``. See
    below; ``None`` keeps the pre-triple meaning.
    ``report_path``: optional path to dump the full per-rule report.
    ``build_path``: optional path to write the kept set (CERTIFIED + TOLERATED).
    ``judge_timeout_s``: per-rule wall-clock cap (a rule that exceeds it is bucketed
    JUDGE-TIMEOUT rather than blocking the sweep).

    CLEANLINESS IS PER MODE (owner ruling, 2026-08-20: "0.14.0 is sound, full stop").
    Before the triple there was one rule set and one contract, so "clean" could mean "no
    fatal bucket". That is now wrong in BOTH directions. `atanh(tanh t) -> t` is
    ENGINE-MISALIGN and belongs in ``rules_real.json``, where it is perfectly clean --
    calling that file dirty would condemn the rules the split exists to keep. And a rule
    that is f64-realised but mathematically false is clean for ``rules.json`` and NOT for
    ``rules_real.json``, which a bucket count cannot express at all.

    So with ``mode`` given, ``is_clean`` means *every rule's judged tier is one this mode
    licenses* (``MODE_TIERS``) -- a rule can be clean for one file and dirty for another,
    which is the whole point. ``report['offenders']`` lists the ones that are not, with
    their tier.

    With ``mode=None`` the legacy meaning is kept -- only CERTIFIED/TOLERATED are clean --
    because that is what every pre-triple caller means. It is NOT the right gate for a
    shipped triple; use :func:`verify_triple`.

    ``report['buckets']`` maps each bucket name to the list of rule indices in it and
    ``report['exit_code']`` is 0 iff clean.
    """
    import tempfile
    import os
    if mode is not None and mode not in MODE_TIERS:
        raise ValueError(
            f"unknown mode {mode!r}: expected one of {sorted(MODE_TIERS)}")
    tmp = report_path or os.path.join(tempfile.mkdtemp(), 'gate_report.json')
    code = _gate.sweep(_load(rules), report_path=tmp, build_path=build_path,
                       judge_timeout_s=judge_timeout_s,
                       announce_report=report_path is not None)
    report = json.load(open(tmp))
    # D36 carve-out, mirrored from `mining._route_triple`: the multi-`<constant>`
    # family's soundness authority is the Const-channel chain, not the contract, so
    # the judge's non-jurisdiction sentinel (UNSUPPORTED-SHAPE + CONST_CHANNEL_DETAIL)
    # is not a verdict about the rule. Without this mirror, the moment a mine routes
    # such a rule into the triple as `core`, `verify_ruleset` on the mine's own
    # output reads tier None -> `reject` and reports dirty in every mode -- the gate
    # disagreeing with the router about a ratified decision (audit U3, 2026-08-22).
    def _eff_tier(d: dict) -> str:
        if d.get('detail') == CONST_CHANNEL_DETAIL:
            return 'core'
        return d.get('tier') or 'reject'

    if mode is None:
        d36 = {d.get('idx') for d in report.get('detail', [])
               if d.get('detail') == CONST_CHANNEL_DETAIL}
        dirty = any(set(report['buckets'].get(k) or []) - d36 for k in FATAL_BUCKETS)
        report['is_clean'] = not dirty
        report['exit_code'] = code
        return report

    allowed = MODE_TIERS[mode]
    offenders = [{'idx': d.get('idx'), 'lhs': d.get('lhs'), 'rhs': d.get('rhs'),
                  'verdict': d.get('verdict'), 'tier': d.get('tier')}
                 for d in report.get('detail', [])
                 if _eff_tier(d) not in allowed]
    report['mode'] = mode
    report['offenders'] = offenders
    report['is_clean'] = not offenders
    report['exit_code'] = 0 if not offenders else 1
    return report


def verify_triple(f64_rules: list | str, real_rules: list | str,
                  corpus_rules: list | str, *, engine_config: str | None = None,
                  corpus_rows: list | None = None, judge_timeout_s: int = 30) -> dict:
    """Gate a whole TRIPLE: each file against its own mode, plus corpus capability.

    A mine run is valid only if all three sets fall out of it, so the artifact is verified
    as one thing. Three per-file sweeps answer "is every rule in this file licensed for
    this mode".

    WHAT IS *NOT* CHECKED, AND WHY. An earlier version required
    ``rules_corpus == rules_f64 UNION rules_real``. That is wrong once folding is
    mode-dependent, because the three modes then reduce in different canonical worlds and
    each file omits what its OWN constructor already performs -- so set overlap measures
    SPELLING, not capability. On the shipped 0.14.0 artifact the union exceeds the corpus
    file by 362 rules and corpus performs every one of them; the identity would report a
    correct artifact as dirty.

    The property it stood proxy for is behavioural: **corpus is at least as capable as
    either other mode**. Pass ``engine_config`` (and optionally ``corpus_rows``) to check
    it -- the report then carries ``corpus_dominance``. Without an engine there is nothing
    sound to say about the relationship between the files, and ``relationships`` stays
    empty rather than asserting something false.

    Returns ``{'is_clean', 'modes': {mode: report}, 'relationships': [...],
    'corpus_dominance': ... }``.
    """
    reports = {}
    for mode, rules in (('f64', f64_rules), ('real', real_rules), ('corpus', corpus_rules)):
        reports[mode] = verify_ruleset(rules, mode=mode, judge_timeout_s=judge_timeout_s)

    problems: list = []
    dominance: list | None = None
    if engine_config is not None:
        from ..engine import SimpliPyEngine
        from ..mining import assert_corpus_dominates
        engine = SimpliPyEngine.from_config(engine_config)
        rows = corpus_rows if corpus_rows is not None else _default_corpus_rows()
        dominance = assert_corpus_dominates(engine, rows)
        if dominance:
            problems.append(
                f'corpus is not the most capable mode on {len(dominance)} rows, '
                f'e.g. {dominance[0]}')

    return {'is_clean': all(x['is_clean'] for x in reports.values()) and not problems,
            'modes': reports, 'relationships': problems,
            'corpus_dominance': dominance}


def _default_corpus_rows() -> list:
    """The repository's own corpus, when it is present; an empty list otherwise."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(os.path.dirname(here), 'benchmarks', 'corpus',
                        'raw_skeletons_nv.json')
    if not os.path.exists(path):
        return []
    return json.load(open(path))


def monitor_ruleset(rules: list | str, engine_config: str, *, corpus_n: int = 6000,
                    seed: int = 20260718, run_selftest: bool = False,
                    judge_timeout_s: int = 10, label: str = '') -> dict:
    """Sweep the deployed engine over a corpus and attribute any violation to a rule.

    ``rules``: a list of ``[lhs, rhs]`` pairs or a path to a JSON rule file.
    ``engine_config``: path to the engine config whose operator realizations define the
    deployed semantics the monitor judges against.
    ``run_selftest``: run the poison self-test first (raises if a known-unsound rule is
    not caught and attributed).

    Returns a dict with ``violations`` (artifact-attributed), ``native`` (pre-existing
    engine behavior, present even with an empty rule set), ``tolerated`` (null-event
    classes), ``changed`` and ``corpus`` counts.
    """
    return _monitor.monitor(_load(rules), engine_config, corpus_n=corpus_n, seed=seed,
                            run_selftest=run_selftest, judge_timeout_s=judge_timeout_s,
                            label=label)


def selftest() -> bool:
    """Run the gate + contract judge poison self-tests. Returns True iff both pass."""
    return _gate.selftest() and _contract.selftest(verbose=False)


__all__ = ['CONST_CHANNEL_DETAIL', 'FATAL_BUCKETS', 'MODE_TIERS', 'verify_rule',
           'verify_ruleset', 'verify_triple', 'monitor_ruleset', 'selftest']
