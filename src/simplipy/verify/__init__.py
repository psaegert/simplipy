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

from . import _contract, _gate, _monitor


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


def verify_ruleset(rules: list | str, *, report_path: str | None = None,
                   build_path: str | None = None, judge_timeout_s: int = 30) -> dict:
    """Gate a whole rule set: judge every rule at its own trigger points.

    ``rules``: a list of ``[lhs, rhs]`` token-list pairs, or a path to such a JSON file.
    ``report_path``: optional path to dump the full per-rule report.
    ``build_path``: optional path to write the kept set (CERTIFIED + TOLERATED).
    ``judge_timeout_s``: per-rule wall-clock cap (a rule that exceeds it is bucketed
    JUDGE-TIMEOUT rather than blocking the sweep).

    Returns the report dict: ``report['buckets']`` maps each bucket name to the list of
    rule indices in it, ``report['is_clean']`` is True iff only CERTIFIED / TOLERATED are
    non-empty, and ``report['exit_code']`` is 0 iff clean.
    """
    import tempfile
    import os
    tmp = report_path or os.path.join(tempfile.mkdtemp(), 'gate_report.json')
    code = _gate.sweep(_load(rules), report_path=tmp, build_path=build_path,
                       judge_timeout_s=judge_timeout_s,
                       announce_report=report_path is not None)
    report = json.load(open(tmp))
    dirty = any(report['buckets'].get(k) for k in
                ('KILL', 'ENGINE-MISALIGN', 'NO-WITNESS', 'UNRESOLVED-COVERAGE',
                 'UNSUPPORTED-SHAPE', 'JUDGE-TIMEOUT'))
    report['is_clean'] = not dirty
    report['exit_code'] = code
    return report


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


__all__ = ['verify_rule', 'verify_ruleset', 'monitor_ruleset', 'selftest']
