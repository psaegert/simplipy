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
    ('pow1_2 pow2 ?0'.split(), '?0'.split(), 'KILL',
     'clause (a): sqrt(x^2)->x changes real values on x<0'),
    ('pow4 pow5 pow1_2 !0'.split(), 'pow !0 <constant>'.split(), 'KILL',
     'clause (b): even-root half-line domain extension'),
    ('pow1_2 pow float("-inf") ?0'.split(), 'pow float("inf") ?0'.split(), 'KILL',
     'clause (b): pow(-inf, data) undefined a.e.'),
]
PHANTOM = ('neg inv neg !0'.split(), 'inv !0'.split(), 'TOLERATED',
           'clause (c): must NOT be killed')


def selftest():
    print('=== rule-complete gate poison self-test ===')
    ok = True
    for lhs, rhs, want, why in POISONS + [PHANTOM]:
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


def sweep(rules, report_path=None, build_path=None, judge_timeout_s=JUDGE_TIMEOUT_S):
    signal.signal(signal.SIGALRM, _alarm)
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
        json.dump({'buckets': buckets, 'detail': detail}, open(report_path, 'w'),
                  default=str)
        print(f'report -> {report_path}')
    if build_path:
        keep = [rules[i] for i in sorted(buckets['CERTIFIED'] + buckets['TOLERATED'])]
        json.dump(keep, open(build_path, 'w'))
        print(f'built {len(keep)} kept rules -> {build_path}')
    return 0 if not (buckets['KILL'] or buckets['ENGINE-MISALIGN']
                     or buckets['NO-WITNESS'] or buckets['UNRESOLVED-COVERAGE']
                     or buckets['UNSUPPORTED-SHAPE'] or buckets['JUDGE-TIMEOUT']) else 1
