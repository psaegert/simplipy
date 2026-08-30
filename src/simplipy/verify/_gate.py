# mypy: ignore-errors
"""Rule-completeness gate: judge every shipped rule at ITS OWN trigger points.

Coverage is 100% of the rule set by construction (unlike a corpus sweep, no rule can
escape by never firing). Judgment = ``judge_rule`` (the canonical per-rule judge):
per-sort symbolic batteries, witness-fitted constants, generic-offset measure grids,
two-rung precision confirmation, deployed-consistency check.

Poison self-test (``selftest``): the gate must CATCH a planted clause-(a) poison, a
planted clause-(b) positive-measure poison, and a planted infinity poison, and must
NOT kill the planted phantom (the tolerated sign-flip rule). A gate that cannot catch
a poisoned rule set, or that convicts the sound, refuses to bless.
"""
import json
import signal

from ._contract import judge_rule


class _JudgeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _JudgeTimeout()


JUDGE_TIMEOUT_S = 30   # per-rule budget. A gate that can HANG is not a gate: mpmath has
                       # uninterruptible code paths that can wedge; the judge caps the
                       # KNOWN wedge classes and this alarm covers the unknown ones.
                       # Timeout -> JUDGE-TIMEOUT bucket, held out fail-closed, never
                       # silently skipped.

POISONS = [
    # legacy-vocabulary spellings (the gate keeps judging pre-0.12 artifacts)
    ('pow1_2 pow2 ?0'.split(), '?0'.split(), 'KILL',
     'clause (a): sqrt(x^2)->x changes real values on x<0'),
    ('pow4 pow5 pow1_2 !0'.split(), 'pow !0 <constant>'.split(), 'KILL',
     'clause (b): even-root half-line domain extension'),
    ('pow1_2 pow float("-inf") ?0'.split(), 'pow float("inf") ?0'.split(), 'KILL',
     'clause (b): pow(-inf, data) undefined a.e.'),
    # live 0.12-vocabulary spellings (rootn + the $ mult-certified sort)
    ('pow rootn !0 2 2'.split(), '!0'.split(), 'KILL',
     'clause (b): even-root round-trip extends the x<0 half-line (rootn spelling)'),
    ('/ $0 $0'.split(), '0'.split(), 'KILL',
     'clause (a): x/x -> 0 is wrong on the whole nonzero line'),
]
PHANTOMS = [
    ('neg inv neg !0'.split(), 'inv !0'.split(), 'TOLERATED',
     'clause (c): must NOT be killed'),
    ('rootn exp _0 (-1)'.split(), 'exp neg _0'.split(), 'CERTIFIED',
     'sound rootn identity (from the shipping artifact): must NOT be killed'),
    ('/ $0 $0'.split(), '1'.split(), 'TOLERATED',
     '$-sort licence f/f -> 1: nan events only at null-excused atoms (0, +-inf)'),
]


def selftest():
    print('=== rule-complete gate poison self-test ===')
    ok = True
    for lhs, rhs, want, why in POISONS + PHANTOMS:
        r = judge_rule(lhs, rhs)
        got = r['verdict']
        good = (got == want)
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {' '.join(lhs)} -> {' '.join(rhs)}: "
              f"{got} (want {want}) -- {why}")
        if not good:
            print(f'        {r}')
    print('GATE SELF-TEST', 'PASSED' if ok else 'FAILED')
    return ok


def sweep(rules, report_path=None, build_path=None, judge_timeout_s=JUDGE_TIMEOUT_S,
          announce_report=True):
    # Install the timeout handler for the duration of the sweep ONLY: the process may
    # have its own SIGALRM use (a host harness, another timeout wrapper), and clobbering
    # it without restore leaks our handler past this call (audit Tier-2, 2026-08-03).
    # MAIN THREAD ONLY: `signal.signal` raises anywhere else, which made every
    # off-main-thread mine crash in its finalize gate (hardening H-010, 2026-08-03).
    # Off the main thread the sweep runs with timeouts DISARMED (judge_timeout_s=0
    # makes every per-rule `signal.alarm(0)` a harmless cancel; arming a real alarm
    # off-main would deliver SIGALRM to an unhandled main thread and kill the process).
    import threading
    on_main = threading.current_thread() is threading.main_thread()
    if not on_main:
        print('off-main-thread sweep: judge timeout protection unavailable '
              '(SIGALRM is main-thread-only)', flush=True)
        return _sweep_inner(rules, report_path, build_path, 0, announce_report)
    prev_handler = signal.signal(signal.SIGALRM, _alarm)
    try:
        return _sweep_inner(rules, report_path, build_path, judge_timeout_s,
                            announce_report)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def _sweep_inner(rules, report_path, build_path, judge_timeout_s, announce_report):
    buckets = {'CERTIFIED': [], 'TOLERATED': [], 'KILL': [],
               'ENGINE-MISALIGN': [], 'NO-WITNESS': [],
               'UNRESOLVED-COVERAGE': [], 'UNSUPPORTED-SHAPE': [], 'JUDGE-TIMEOUT': []}
    detail = []
    for idx, (lhs, rhs) in enumerate(rules):
        signal.alarm(judge_timeout_s)
        try:
            r = judge_rule(list(lhs), list(rhs))
        except _JudgeTimeout:
            r = {'verdict': 'JUDGE-TIMEOUT', 'detail': f'> {judge_timeout_s}s'}
        except Exception as ex:
            r = {'verdict': 'NO-WITNESS', 'detail': f'judge error {type(ex).__name__}: {ex}'}
        finally:
            signal.alarm(0)
        r['idx'] = idx
        r['lhs'] = ' '.join(lhs)
        r['rhs'] = ' '.join(rhs)
        buckets[r['verdict']].append(idx)
        detail.append(r)
        if idx % 200 == 0:
            print(f"[{idx}/{len(rules)}] " +
                  ' '.join(f'{k}={len(v)}' for k, v in buckets.items()), flush=True)
    # THE TIMEOUT MUST NOT DECIDE UNDER LOAD (audit U3, 2026-08-22). JUDGE-TIMEOUT is
    # fatal downstream, and a 30 s wall-clock alarm is a fact about the BOX, not the
    # rule: an ordinary CERTIFIED row measured 33-36 s on an idle 12-core host, so a
    # 32-worker mine drops rules its own gate would pass -- a non-reproducible
    # artifact, which D29 byte-identity cannot survive. Every timed-out rule is
    # therefore re-judged SERIALLY here, after the sweep, at 4x the budget -- the
    # retry environment is the quietest this process can offer, and the widened
    # budget is deterministic. A rule that times out twice stays JUDGE-TIMEOUT:
    # fail-closed, and recorded as retried.
    if judge_timeout_s and buckets['JUDGE-TIMEOUT']:
        retry = list(buckets['JUDGE-TIMEOUT'])
        buckets['JUDGE-TIMEOUT'] = []
        for idx in retry:
            lhs, rhs = rules[idx]
            signal.alarm(judge_timeout_s * 4)
            try:
                r = judge_rule(list(lhs), list(rhs))
            except _JudgeTimeout:
                r = {'verdict': 'JUDGE-TIMEOUT',
                     'detail': f'> {judge_timeout_s}s, retried > {judge_timeout_s * 4}s'}
            except Exception as ex:
                r = {'verdict': 'NO-WITNESS',
                     'detail': f'judge error {type(ex).__name__}: {ex}'}
            finally:
                signal.alarm(0)
            r['idx'] = idx
            r['lhs'] = ' '.join(lhs)
            r['rhs'] = ' '.join(rhs)
            r['retried'] = True
            buckets[r['verdict']].append(idx)
            detail[idx] = r
        print(f"retried {len(retry)} timed-out rule(s): "
              f"{len(retry) - len(buckets['JUDGE-TIMEOUT'])} resolved, "
              f"{len(buckets['JUDGE-TIMEOUT'])} still out", flush=True)
    print('\n================ GATE RESULT ================')
    for k, v in buckets.items():
        print(f'  {k:16s} {len(v)}')
    for k in ('KILL', 'ENGINE-MISALIGN', 'NO-WITNESS', 'UNRESOLVED-COVERAGE',
              'UNSUPPORTED-SHAPE', 'JUDGE-TIMEOUT'):
        for i in buckets[k]:
            d = detail[i]
            why = d.get('clause') or ','.join(d.get('kinds', [])) or d.get('detail', '')
            print(f"    {k} idx {i}: {d['lhs']}  ->  {d['rhs']}   [{why}]")
    if report_path:
        json.dump({'buckets': buckets, 'detail': detail,
                   'timeout_protection': bool(judge_timeout_s)}, open(report_path, 'w'),
                  default=str)
        # Announced only when the CALLER chose the path: verify_ruleset routes its
        # read-back through an anonymous tempdir, and printing that random path
        # makes otherwise byte-identical mine stdout differ across processes
        # (found by the determinism-across-PYTHONHASHSEED pin, 2026-08-02).
        if announce_report:
            print(f'report -> {report_path}')
    if build_path:
        keep = [rules[i] for i in sorted(buckets['CERTIFIED'] + buckets['TOLERATED'])]
        json.dump(keep, open(build_path, 'w'))
        print(f'built {len(keep)} kept rules -> {build_path}')
    return 0 if not (buckets['KILL'] or buckets['ENGINE-MISALIGN']
                     or buckets['NO-WITNESS'] or buckets['UNRESOLVED-COVERAGE']
                     or buckets['UNSUPPORTED-SHAPE'] or buckets['JUDGE-TIMEOUT']) else 1
