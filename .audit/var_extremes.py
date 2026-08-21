"""Same probe on VARIABLE slots: bind ONE slot of each rules.json row to an f64-extreme
literal (leaving every other slot quantified) and re-judge. Rules whose RHS carries a
fitted `<constant>` are excluded -- there the witness bisection, not the rule, decides
realisation."""
import json, signal, sys, time, re
from multiprocessing import Pool
sys.path.insert(0, '/home/psaegert/Projects/simplipy/src')

PATH = '/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/rules.json'
OUT = '/home/psaegert/Projects/simplipy/.audit/var_extremes.json'
EXTREMES = ['1e300', '(-1e300)', '1e-300', '(-1e-300)', '1e17', '(-1e17)',
            '1e5', '(-1e5)', '800', '(-800)', '1e-17', '(-1e-17)']
SLOT = re.compile(r'^([?!_$])(\d+)$')
ALLOWED = {'core', 'f64'}


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
        if '<constant>' in rhs:
            continue
        slots = sorted({t for t in lhs if SLOT.match(t)})
        for s in slots:
            for c in EXTREMES:
                l2 = [c if t == s else t for t in lhs]
                r2 = [c if t == s else t for t in rhs]
                signal.alarm(60)
                try:
                    r = judge_rule(list(l2), list(r2))
                except _T:
                    r = {'verdict': 'JUDGE-TIMEOUT'}
                except Exception as ex:
                    r = {'verdict': 'JUDGE-ERROR', 'detail': f'{type(ex).__name__}: {ex}'}
                finally:
                    signal.alarm(0)
                if r.get('realised') is False:
                    out.append({'idx': idx, 'slot': s, 'const': c,
                                'lhs': ' '.join(lhs), 'rhs': ' '.join(rhs),
                                'inst_l': ' '.join(l2), 'inst_r': ' '.join(r2),
                                'verdict': r.get('verdict'), 'tier': r.get('tier')})
    return out


if __name__ == '__main__':
    n = len(json.load(open(PATH)))
    step = 25
    shards = [(i, min(i + step, n)) for i in range(0, n, step)]
    res = []
    t0 = time.time()
    with Pool(10) as p:
        for k, chunk in enumerate(p.imap_unordered(work, shards)):
            res.extend(chunk)
            if k % 25 == 0:
                print(f'{k}/{len(shards)} shards, {len(res)} hits, {time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT, 'w'), default=str)
    print('WROTE', OUT, len(res))
