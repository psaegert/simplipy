"""Corrected A/B for the division-tower certificate completion.

Fixes two instrument confounds in the first attempt:
  * change detection now compares the STORED tagged outputs verbatim against the new
    engine's native tagged outputs (no projection -- any engine-side projection
    re-canonicalizes old spellings under the new licences and hides the change);
  * spelling-level pricing uses complexity(certified=False): bare-Cx canon carries no
    cert_nzae, so a tower prices as spelled; certified mu re-canonicalizes both sides
    and prices old == new by construction (mu(simplify(e)) <= mu(e) theorem).
Certified (deployed-measure) numbers are reported alongside as secondary.
"""
import json
import os
import pickle
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(REPO, 'benchmarks')
OUTDIR = HERE
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'remine'))
from desugar_corpus import desugar  # noqa: E402
from simplipy import SimpliPyEngine  # noqa: E402

CONFIG = os.path.join(REPO, 'remine', 'acj-4-3', 'config.yaml')


def _judge_worker(args):
    idx, sk, out = args
    import numpy as _np
    from simplipy.verify._monitor import judge_pair
    try:
        v, d = judge_pair(list(sk), list(out), _np.random.default_rng(0))
    except Exception as ex:
        return (idx, f'JUDGE_ERROR:{type(ex).__name__}', str(ex)[:120])
    return (idx, v, str(d)[:120])


def main():
    corpus = pickle.load(open(os.path.join(REPO, 'remine', 'corpus_nv_64k.pkl'), 'rb'))
    r012 = pickle.load(open(os.path.join(HERE, 'ecdf_vs_sympy_results.pkl'), 'rb'))
    r011 = pickle.load(open(os.path.join(HERE, 'ecdf_011_results.pkl'), 'rb'))
    tags = json.load(open(os.path.join(HERE, 'deep_win_subclasses.json')))
    e = SimpliPyEngine.from_config(CONFIG)

    old_tagged = r012['simplipy']['acj-4-3']['outputs']
    out11 = [desugar(o)[0] for o in r011['outputs']]

    print('simplifying 64k (tagged + explicit) ...', flush=True)
    new_tagged = [e.simplify(r) for r in corpus]
    new_explicit = [e.simplify(r, form='explicit') for r in corpus]

    changed = [i for i in range(len(corpus)) if list(new_tagged[i]) != list(old_tagged[i])]
    const_free = np.array(['<constant>' not in r for r in corpus])
    print(f'rows changed (verbatim tagged): {len(changed)} '
          f'(const-free: {sum(1 for i in changed if const_free[i])})', flush=True)

    print('judging every changed row ...', flush=True)
    with Pool(24) as pool:
        verdicts = {i: (v, d) for i, v, d in pool.imap_unordered(
            _judge_worker, [(i, corpus[i], new_explicit[i]) for i in changed],
            chunksize=16)}
    vc_free = Counter(v for i, (v, _) in verdicts.items() if const_free[i])
    vc_const = Counter(v for i, (v, _) in verdicts.items() if not const_free[i])
    alarms_free = {i: vd for i, vd in verdicts.items()
                   if const_free[i] and vd[0] not in ('OK', 'F64_FOLD')}
    print(f'const-free verdicts: {dict(vc_free)}  (alarms: {len(alarms_free)})',
          flush=True)
    print(f'const-bearing verdicts (E3-unreliable instrument): {dict(vc_const)}',
          flush=True)
    for i, (v, d) in sorted(alarms_free.items())[:20]:
        print(f'  CONST-FREE ALARM row {i}: {v} {d}')
        print(f'    in : {" ".join(corpus[i])}')
        print(f'    out: {" ".join(new_explicit[i])}')

    def bare_mu(x):
        return e.complexity(x, certified=False)

    print('pricing spellings (bare mu) ...', flush=True)
    mu_o = np.array([bare_mu(r) for r in corpus], float)
    mu_new = np.array([bare_mu(o) for o in new_tagged], float)
    mu_old = np.array([bare_mu(o) for o in old_tagged], float)
    mu11 = np.array([bare_mu(o) for o in out11], float)
    r_new, r_old, r_11 = mu_new / mu_o, mu_old / mu_o, mu11 / mu_o

    unlock = {}
    for tag in ('B-division-tower', 'A-stuck-zero', 'C-inf-tower', 'D-other'):
        rows = [r['row'] for r in tags['records'] if r['tag'] == tag]
        reach = [i for i in rows if mu_new[i] <= mu11[i]]
        better = [i for i in rows if mu_new[i] < mu_old[i]]
        unlock[tag] = {'n': len(rows), 'reach_011': len(reach),
                       'improved_vs_old': len(better)}
        print(f'{tag}: n={len(rows)} reach-0.11={len(reach)} '
              f'improved-vs-old={len(better)}', flush=True)

    deep = const_free & ((r_11 < 0.9) | (r_new < 0.9))
    summary = {
        'rows_changed': len(changed),
        'rows_changed_const_free': int(sum(1 for i in changed if const_free[i])),
        'judge_verdicts_const_free': dict(vc_free),
        'judge_verdicts_const_bearing_E3_unreliable': dict(vc_const),
        'const_free_alarms': {str(i): list(vd) for i, vd in alarms_free.items()},
        'unlock_bare_mu': unlock,
        'bare_mu': {
            'ratio_mean_new': float(np.mean(r_new)),
            'ratio_mean_old': float(np.mean(r_old)),
            'ratio_mean_011': float(np.mean(r_11)),
            'frac_below_1_new': float(np.mean(r_new < 1)),
            'frac_below_1_old': float(np.mean(r_old < 1)),
            'frac_below_1_011': float(np.mean(r_11 < 1)),
            'frac_below_0.9_new': float(np.mean(r_new < 0.9)),
            'frac_below_0.9_old': float(np.mean(r_old < 0.9)),
            'frac_below_0.9_011': float(np.mean(r_11 < 0.9)),
            'corpus_total_new': float(mu_new.sum()),
            'corpus_total_old': float(mu_old.sum()),
            'corpus_total_011': float(mu11.sum()),
        },
        'paired_vs_011_bare': {
            'win_012_new': float(np.mean(mu_new < mu11)),
            'tie': float(np.mean(mu_new == mu11)),
            'win_011': float(np.mean(mu_new > mu11)),
            'win_012_old': float(np.mean(mu_old < mu11)),
            'win_011_old': float(np.mean(mu_old > mu11)),
        },
        'deep_region_bare': {
            'rows': int(deep.sum()),
            'win_012_new': float(np.mean((mu_new < mu11)[deep])),
            'win_011': float(np.mean((mu_new > mu11)[deep])),
        },
    }
    json.dump(summary, open(os.path.join(OUTDIR, 'ab_divtower_summary.json'), 'w'),
              indent=1)
    with open(os.path.join(OUTDIR, 'new12_tagged.pkl'), 'wb') as f:
        pickle.dump({'tagged': new_tagged, 'explicit': new_explicit}, f)
    print(json.dumps({k: v for k, v in summary.items() if k != 'const_free_alarms'},
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
