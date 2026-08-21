"""Parallel gate sweep over a shipped rule file. One process per shard; each shard
arms its OWN SIGALRM (it is that process's main thread), exactly like _gate.sweep."""
import json, os, signal, sys, time
from multiprocessing import Pool

sys.path.insert(0, '/home/psaegert/Projects/simplipy/src')

PATH = sys.argv[1]
OUT = sys.argv[2]
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 10
TIMEOUT = 60


class _T(Exception):
    pass


def _alarm(_s, _f):
    raise _T()


def work(args):
    lo, hi = args
    from simplipy.verify._contract import judge_rule
    rules = json.load(open(PATH))
    signal.signal(signal.SIGALRM, _alarm)
    out = []
    for idx in range(lo, hi):
        lhs, rhs = rules[idx]
        signal.alarm(TIMEOUT)
        t0 = time.time()
        try:
            r = judge_rule(list(lhs), list(rhs))
        except _T:
            r = {'verdict': 'JUDGE-TIMEOUT', 'detail': f'> {TIMEOUT}s'}
        except Exception as ex:
            r = {'verdict': 'NO-WITNESS', 'detail': f'judge error {type(ex).__name__}: {ex}'}
        finally:
            signal.alarm(0)
        r['idx'] = idx
        r['lhs'] = ' '.join(lhs)
        r['rhs'] = ' '.join(rhs)
        r['secs'] = round(time.time() - t0, 3)
        out.append(r)
    return out


if __name__ == '__main__':
    n = len(json.load(open(PATH)))
    step = 25
    shards = [(i, min(i + step, n)) for i in range(0, n, step)]
    res = []
    done = 0
    with Pool(NPROC) as p:
        for chunk in p.imap_unordered(work, shards):
            res.extend(chunk)
            done += len(chunk)
            if done % 250 < step:
                print(f'{done}/{n}', flush=True)
    res.sort(key=lambda d: d['idx'])
    json.dump(res, open(OUT, 'w'), default=str)
    print('WROTE', OUT, len(res))
