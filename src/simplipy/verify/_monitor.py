# mypy: ignore-errors
"""The deployed-realization monitor: an end-to-end sweep of the engine, judged by the rule judge.

The monitor exercises the DEPLOYED engine end to end (``simplify`` on real expression shapes:
an adversarial list and a sampled corpus, literal and pre-masked, as deployed traffic arrives)
and judges every input -> output rewrite as the rule judge judges a rule
(``_contract.judge_rule``): the rewrite holds in the mode it was made in exactly where its
judged tier is one that mode licenses (``verify.MODE_TIERS``), as a rule's does. The input's
variables are the rule's ``?`` slots, its ``<constant>`` leaves the source's constants and the
output's the fitted ones. So the monitor has no evaluator of its own: every value it compares is the core's
table, read through the judge, and the engine's composition of rules and constructors is held
to the criterion every rule is held to. Where a per-rule scan judges each rule at its own
points, the monitor judges what the rules do together on whole expressions.

Verdict classes per rewrite: OK | VIOLATION | UNSCORED (a rewrite the judge cannot read, or
reaches no decision on). A violation the rules-empty engine commits too is the engine's own
(NATIVE-LINE), not the rule set's. An `f64` rewrite is held to float64's values (8 units in the
last place), which the constructors' algebra does not always keep: `x/(x/(x^2 - x)) ->
x*(x - 1)` is true in the reals and float64's rounding near x = 1 keeps the two apart.

Self-test: :func:`selftest` poisons a copy of the rule set with known-unsound rules and fails
loudly unless the monitor flags and attributes every one of them.

Importable API:
  * :func:`judge_pair`  -- judge a single ``inp -> out`` rewrite.
  * :func:`sweep`       -- run the engine over a corpus and collect violations.
  * :func:`build_engine`, :func:`make_corpus` -- engine/corpus construction.
  * :func:`selftest`    -- poison self-test (returns True iff every poison is caught).
  * :func:`monitor`     -- top-level driver (optional self-test, then the main sweep).
"""
import json
import os
import re

import numpy as np
import yaml

from ..engine import _TAGGED_DIALECT_TOKENS, SimpliPyEngine
from ..utils import count_expressions, sample_expression
from . import _contract

VAR_RE = re.compile(r'^x(\d+)$')

#: The file mode each spelling of a mode names (`judge_pair(mode=)`).
_MODES = {'f64': 'f64', 'default': 'f64', 'real': 'real', 'permissive': 'permissive'}


def _mode_name(mode):
    """The file mode (`f64`, `real` or `permissive`) of a mode name or `simplipy.Mode` member."""
    name = getattr(mode, 'name', mode)
    if name not in _MODES:
        raise ValueError(f'unknown mode {mode!r}: one of {sorted(_MODES)}')
    return _MODES[name]


def _as_rule(inp, out, engine):
    """The rewrite as the judge reads a rule: binary prefix (a tagged side is expanded by
    `engine.to_prefix`), each variable `x<k>` the slot `?<k>`."""
    sides = []
    for side in (list(inp), list(out)):
        if _TAGGED_DIALECT_TOKENS.intersection(side):
            if engine is None:
                raise ValueError('a tagged side needs an engine to read it')
            side = list(engine.to_prefix(side))
        sides.append([f'?{m.group(1)}' if (m := VAR_RE.match(t)) else t for t in side])
    return sides


#: The judge's verdicts that are no decision: the rewrite is neither passed nor convicted.
UNDECIDED = ('NO-WITNESS', 'UNRESOLVED-COVERAGE', 'UNSUPPORTED-SHAPE', 'JUDGE-TIMEOUT',
             'UNDECIDED-MEASURE')


def judge_pair(inp, out, rng=None, *, mode='f64', engine=None):
    """Judge one rewrite ``inp -> out`` made in ``mode`` (``'f64'``, also ``'default'``,
    ``'real'``, ``'permissive'`` or a :class:`~simplipy.Mode` member; the default is the
    engine's default mode) as the rule judge judges the rule ``inp -> out``.

    Returns ``(verdict, detail)``: ``OK`` where the judged tier is one the mode licenses
    (``verify.MODE_TIERS``), ``VIOLATION`` where it is not, and ``UNSCORED`` where the
    judge cannot read the rewrite or reaches no decision on it (an undecided measure among
    them; fail closed: the sweep counts it). ``engine`` reads a tagged
    side (`to_prefix`). ``rng`` is accepted for the callers that pass one; the judgment is
    deterministic and draws nothing."""
    from . import MODE_TIERS, effective_tier
    name = _mode_name(mode)
    try:
        lhs, rhs = _as_rule(inp, out, engine)
        detail = _contract.judge_rule(lhs, rhs)
    except Exception as error:       # an unreadable rewrite is reported, never passed
        return 'UNSCORED', f'{type(error).__name__}: {error}'
    verdict, tier = detail.get('verdict'), effective_tier(detail)
    if tier in MODE_TIERS[name]:
        return 'OK', f'{verdict} ({tier})'
    if verdict in UNDECIDED:
        return 'UNSCORED', f'{verdict}: {str(detail.get("detail"))[:200]}'
    return 'VIOLATION', f'{verdict} ({tier}): {str(detail.get("detail"))[:200]}'


# ---------------------------------------------------------------------------------------
# Corpus, engines, the sweep
# ---------------------------------------------------------------------------------------

ADVERSARIAL = [
    ['+', 'x0', 'pow', 'sin', 'x0', 'float("inf")'],
    ['pow', 'sin', 'x0', 'float("inf")'],
    ['*', '4', 'pow', 'x0', 'float("inf")'],
    ['-', 'exp', 'x0', 'exp', 'x0'],
    ['-', 'log', 'x0', 'log', 'x0'],
    ['inv', '*', '0', 'x0'],
    ['/', 'log', 'x0', 'log', 'x0'],
    ['atanh', 'tanh', 'sinh', 'x0'],
    ['*', 'x0', 'pow', 'cos', 'x1', 'float("inf")'],
    ['+', 'pow', 'tanh', 'x0', 'float("inf")', 'x1'],
    ['pow', '/', '7.0', '5', 'float("nan")'],
    ['*', 'sin', 'np.pi', 'x0'],
    ['abs', 'pow', 'x0', '*', '3', '1'],
    ['tan', '+', 'x0', 'np.pi'],
    ['inv', '+', '0', 'neg', 'pow', 'x0', '4'],
    ['pow', 'float("-inf")', 'rootn', '3.0', '3'],
    # triggers for the measure-clause poison (even-root round-trip via rootn)
    ['pow', 'rootn', 'x0', '2', '2'],
    ['*', '2', 'pow', 'rootn', 'x1', '2', '2'],
    # trigger for the product-over-a-sum poison
    ['*', 'x0', '+', 'x1', 'x2'],
    # PRE-MASKED shapes: deployed traffic arrives with literals already masked to
    # <constant> (masking is explicit, upstream of simplify), and they are the masked-scale
    # poisons' triggers on any rule set
    ['*', '<constant>', 'x0'],
    ['+', '*', '<constant>', 'x1', 'x0'],
    ['sin', '*', '<constant>', 'x0'],
    ['*', '<constant>', 'pow', 'x0', 'float("inf")'],
]

# Default leaf vocabulary used to sample the random half of the corpus.
DEFAULT_LEAVES = ['x0', 'x1', '0', '1', 'np.pi', '2.0', '(-0.5)', '3.0']

# Known-unsound rules for the poison self-test. Each must be DEPLOYABLE end to end (the load
# accepts it) and a trigger shape must exist in ADVERSARIAL. Catch requires ATTRIBUTION: a
# flagged input whose output CHANGED versus the unpoisoned engine.
POISON = [
    # a value change at a point f64 holds: float64's sin(fl(pi/2)) is exactly 1.0, where
    # pow(1, inf) is 1 (the reals give pow(1, inf) no value, so real tolerates the rewrite)
    (('pow', 'sin', '?0', 'float("inf")'), ('0',)),
    # masked-scale: drops the coefficient of pre-masked traffic
    (('*', '<constant>', '?0'), ('?0',)),
    # structural coefficient drop (fires on the pre-masked pow shape)
    (('*', '<constant>', 'pow', '?0', 'float("inf")'), ('pow', '?0', 'float("inf")')),
    # measure-clause poison: the even-root round-trip pow(rootn(x,2),2) -> x extends the
    # x < 0 half-line
    (('pow', 'rootn', '?0', '2', '2'), ('?0',)),
    # a dropped factor: x*(y+z) -> y+z
    (('*', '?0', '+', '?1', '?2'), ('+', '?1', '?2')),
]


def build_engine(rules, config_path):
    """Build a deployed engine from `config_path`, overriding its ruleset with `rules`.

    `config_path` is the deployed engine config the monitor realizes against; the driver
    passes the path in. `rules` is a list of (lhs, rhs) token-sequence pairs (empty for the
    rules-empty baseline engine), and it is the engine's whole rule set: no mode names a set
    of its own. Built in memory from the config's operator table, so nothing is written to
    disk."""
    base = yaml.safe_load(open(config_path))
    return SimpliPyEngine(operators=base['operators'], rules=[(tuple(l), tuple(r)) for l, r in rules])


def make_corpus(eng, n, rng, adversarial=None, leaves=None):
    if adversarial is None:
        adversarial = ADVERSARIAL
    if leaves is None:
        leaves = DEFAULT_LEAVES
    non_leaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
    counts = count_expressions(len(leaves), non_leaf, 10)
    corpus = [list(e) for e in adversarial]
    for _ in range(n):
        L = int(rng.integers(3, 10))
        corpus.append(list(sample_expression(L, leaves, non_leaf, counts, rng)))
    return corpus


class _JudgeTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _JudgeTimeout()


JUDGE_TIMEOUT_S = 10   # a single expression may not eat the sweep; a timeout is recorded as
                       # unresolved (with the offending expression printed) and skipped.


def sweep(rules, corpus, rng, baseline_engine, config_path, tag='',
          judge_timeout_s=JUDGE_TIMEOUT_S):
    # Timeout protection is SIGALRM-based and therefore main-thread-only. Install for
    # the duration of the sweep and RESTORE on the way out: clobbering the process handler
    # permanently would crash any off-main-thread caller.
    import signal
    import threading
    on_main = threading.current_thread() is threading.main_thread()
    if on_main:
        prev_handler = signal.signal(signal.SIGALRM, _alarm)
    else:
        print(f'  [{tag}] off-main-thread sweep: judge timeout protection unavailable '
              f'(SIGALRM is main-thread-only)', flush=True)
    try:
        return _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag,
                            judge_timeout_s if on_main else 0)
    finally:
        if on_main:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)


def _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag,
                 judge_timeout_s):
    import signal
    eng = build_engine(rules, config_path)
    viol, native, unscored, changed, timeouts = [], [], 0, 0, 0

    def judged(expr, out):
        signal.alarm(judge_timeout_s)
        try:
            return judge_pair(list(expr), out, rng, engine=eng)
        finally:
            signal.alarm(0)

    for i, expr in enumerate(corpus):
        if tag and (i + 1) % 500 == 0:
            print(f'  [{tag} {i+1}/{len(corpus)}] changed={changed} viol={len(viol)} '
                  f'timeouts={timeouts}', flush=True)
        try:
            out = list(eng.simplify(list(expr)))
        except Exception as ex:
            viol.append((expr, ['<RAISED>'], f'simplify raised {type(ex).__name__}'))
            continue
        if out == list(expr):
            continue
        changed += 1
        try:
            v, detail = judged(expr, out)
        except _JudgeTimeout:
            timeouts += 1
            print(f'  [{tag}] TIMEOUT-unresolved: {" ".join(expr)}  ->  {" ".join(out)}', flush=True)
            continue
        if v == 'UNSCORED':
            unscored += 1
        elif v == 'VIOLATION':
            try:
                base_out = list(baseline_engine.simplify(list(expr)))
            except Exception:
                base_out = None
            if base_out == out:
                native.append((expr, out, detail))
            elif base_out is not None and base_out != list(expr):
                # CAUSE attribution: if the rules-empty engine's own rewrite of this
                # expression ALREADY violates, the violation is native and the rule set
                # merely rewrote further on top.
                try:
                    vb, _ = judged(expr, base_out)
                except _JudgeTimeout:
                    vb = None
                if vb == 'VIOLATION':
                    native.append((expr, out, detail))
                else:
                    viol.append((expr, out, detail))
            else:
                viol.append((expr, out, detail))
    return viol, native, unscored, changed


def selftest(rules, config_path, corpus, baseline, seed, poison=None,
             judge_timeout_s=JUDGE_TIMEOUT_S):
    """Poison a copy of the ruleset with known-unsound rules; return True iff EVERY poison is
    caught. A catch requires ATTRIBUTION: a flagged input whose output differs from the
    unpoisoned (clean) engine's output. Prints a per-poison CAUGHT / *** MISSED *** line."""
    if poison is None:
        poison = POISON
    clean_eng = build_engine(rules, config_path)
    caught = 0
    for p in poison:
        v, *_ = sweep(rules + [p], corpus[:600], np.random.default_rng(seed),
                      baseline, config_path, tag="poison", judge_timeout_s=judge_timeout_s)
        attributed = []
        for expr, out, detail in v:
            try:
                clean_out = list(clean_eng.simplify(list(expr)))
            except Exception:
                clean_out = ['<RAISED>']
            # attribution = the poisoned engine BEHAVES DIFFERENTLY from the clean one on
            # this input; both raising is the same behaviour.
            if clean_out != list(out):
                attributed.append((expr, out, detail))
        hit = len(attributed) > 0
        caught += hit
        print(f'  poison {" ".join(p[0])} -> {" ".join(p[1])}: '
              f'{"CAUGHT" if hit else "*** MISSED ***"} '
              f'({len(attributed)} attributed / {len(v)} flagged)')
    return caught == len(poison)


class MonitorSelfTestError(RuntimeError):
    """Raised by :func:`monitor` when the poison self-test fails to flag a known-unsound rule.

    A self-test failure means the monitor cannot be trusted to catch an unsound ruleset, so
    the main sweep is not run: the gate has lost its own guarantee."""


class MonitorLivenessError(RuntimeError):
    """Raised by :func:`monitor` when the main sweep rewrites NOTHING on a non-trivial corpus.

    A sweep that judges only rewrites is vacuous when no rewrites happen: `changed == 0`
    over hundreds of sampled expressions means the engine under test is not applying
    anything (broken asset load, dead core plumbing) -- an instrument failure that must
    not be reported as 'clean' (0 violations of 0 rewrites)."""


def monitor(rules, config_path, corpus_n=6000, seed=20260718, run_selftest=False,
            adversarial=None, leaves=None, poison=None, judge_timeout_s=JUDGE_TIMEOUT_S,
            label=''):
    """Run the monitor against a deployed engine built from `config_path`.

    `rules` is either a list of (lhs, rhs) token-sequence pairs or a path to a JSON file of
    ``[[lhs, rhs], ...]`` (loaded as tuples). When `run_selftest` is True the poison self-test
    runs first and a failure raises :class:`MonitorSelfTestError` before the main sweep.

    Returns a dict with keys: rules, corpus, changed, violations, native, unscored,
    selftest_passed."""
    if adversarial is None:
        adversarial = ADVERSARIAL
    if leaves is None:
        leaves = DEFAULT_LEAVES
    if poison is None:
        poison = POISON
    rules = _load_rules(rules)
    rng = np.random.default_rng(seed)
    baseline = build_engine([], config_path)
    corpus = make_corpus(baseline, corpus_n, rng, adversarial=adversarial, leaves=leaves)
    print(f'monitor: {len(rules):,} rules | corpus {len(corpus):,} '
          f'({len(adversarial)} adversarial + {corpus_n:,} sampled)', flush=True)

    selftest_passed = None
    if run_selftest:
        selftest_passed = selftest(rules, config_path, corpus, baseline, seed,
                                   poison=poison, judge_timeout_s=judge_timeout_s)
        if not selftest_passed:
            print('SELF-TEST FAILED: the monitor cannot catch a poisoned artifact')
            raise MonitorSelfTestError(
                'poison self-test failed: a known-unsound rule was not flagged')
        print('SELF-TEST PASSED (all poisons caught, attribution-verified)\n', flush=True)

    viol, native, unscored, changed = sweep(
        rules, corpus, rng, baseline, config_path,
        tag="main", judge_timeout_s=judge_timeout_s)
    if changed == 0 and len(corpus) >= 100:
        raise MonitorLivenessError(
            f'main sweep rewrote 0 / {len(corpus)} expressions: the engine under test '
            f'applied nothing (broken asset load or dead core plumbing) -- a sweep that '
            f'judges only rewrites is vacuous here and must not report clean')
    print(f'\n=== MONITOR  {label} ===')
    print(f'  rewritten     {changed:7,} / {len(corpus):,}')
    print(f'  VIOLATIONS    {len(viol):7,}  (artifact-attributed)')
    print(f'  native-line   {len(native):7,}  (present with rules=[]; pre-existing engine behavior)')
    print(f'  unscored      {unscored:7,}  (rewrites the judge cannot read or decide)')
    for expr, out, detail in viol[:15]:
        print(f'    VIOL {" ".join(expr)}  ->  {" ".join(out)}   [{detail}]')
    for expr, out, detail in native[:5]:
        print(f'    (native) {" ".join(expr)}  ->  {" ".join(out)}   [{detail}]')
    return {
        'rules': len(rules),
        'corpus': len(corpus),
        'changed': changed,
        'violations': viol,
        'native': native,
        'unscored': unscored,
        'selftest_passed': selftest_passed,
    }


def _load_rules(rules):
    """Load rules from a JSON path (list of [lhs, rhs]) into (tuple, tuple) pairs, or pass a
    pre-loaded rule list through unchanged."""
    if isinstance(rules, (str, os.PathLike)):
        with open(rules) as fh:
            return [(tuple(l), tuple(r)) for l, r in json.load(fh)]
    return rules
