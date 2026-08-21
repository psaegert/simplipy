"""Instantiate the LHS `<constant>` slot of every rules.json row at f64-EXTREME values
the judge's own cl battery (max |5|, plus pi/e) never reaches, and re-judge each
instance with judge_rule. Only the LHS `<constant>` is bound; an RHS `<constant>` is
left as the independent, fitted slot."""
import json, signal, sys, time
from multiprocessing import Pool
sys.path.insert(0, '/home/psaegert/Projects/simplipy/src')

PATH = '/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/rules.json'
OUT = '/home/psaegert/Projects/simplipy/.audit/const_extremes.json'
EXTREMES = ['800', '(-800)', '750', '(-750)', '710', '(-710)', '100', '(-100)',
            '20', '(-20)', '50', '(-50)',
            '1e300', '(-1e300)', '1e-300', '(-1e-300)',
            '1e17', '(-1e17)', '1e-17', '(-1e-17)']
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
        if lhs.count('<constant>') != 1:
            continue
        for c in EXTREMES:
            l2 = [c if t == '<constant>' else t for t in lhs]
            signal.alarm(60)
            try:
                r = judge_rule(list(l2), list(rhs))
            except _T:
                r = {'verdict': 'JUDGE-TIMEOUT'}
            except Exception as ex:
                r = {'verdict': 'JUDGE-ERROR', 'detail': f'{type(ex).__name__}: {ex}'}
            finally:
                signal.alarm(0)
            tier = r.get('tier') or 'reject'
            if tier not in ALLOWED:
                out.append({'idx': idx, 'const': c, 'lhs': ' '.join(lhs),
                            'inst': ' '.join(l2), 'rhs': ' '.join(rhs),
                            'verdict': r.get('verdict'), 'tier': r.get('tier'),
                            'realised': r.get('realised'), 'detail': r.get('detail'),
                            'kinds': r.get('kinds'), 'clause': r.get('clause')})
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
            if k % 20 == 0:
                print(f'{k}/{len(shards)} shards, {len(res)} hits, {time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT, 'w'), default=str)
    print('WROTE', OUT, len(res))
