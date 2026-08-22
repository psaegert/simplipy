"""Sound-only scoring: every changed row is adjudicated; unsound rewrites score ratio 1.

Owner directive (2026-08-02): "we must compare the ratios on sound simplifications only,
and punish unsound ones by refusing the simplification and yielding ratio=1."

Instrument: ``simplipy.verify._monitor.judge_pair`` -- the same contract-precision judge
the 64k/1M scale gates use for flag adjudication (E1/E4 lineage) -- applied to every
CHANGED (input, output) pair of each system. A non-OK verdict (except F64_FOLD, the
named correctly-rounded-fold class) refuses the rewrite: the row scores ratio 1.

SCOPE: ``<constant>``-free rows only (43,672 of 65,536; the same subset the scale
gates screen). The judge's shared-witness binding on masked rows fabricates spurious
convictions (documented at E3), so Const-bearing rows are NOT adjudicable with the
current instrument; they are reported as a separate count, uncompared. The 0.11
Const-absorption class (``acos(<constant>) -> <constant>``, a family-WIDENING rewrite
our policy refuses) therefore does not enter the sound-only comparison either way.

RUNS IN THE 0.12 ENV. Consumes the three results pickles.
Artifacts: benchmarks/ecdf_sound_only_summary.json (tracked),
           benchmarks/out/simplipy_sound_only.{png,svg}.
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "remine"))
from desugar_corpus import desugar  # noqa: E402
from simplipy import SimpliPyEngine  # noqa: E402

CORPUS_PKL = os.path.join(REPO, "remine", "corpus_nv_64k.pkl")
OUT_JSON = os.path.join(HERE, "ecdf_sound_only_summary.json")

_ENGINE = None


def _judge_worker(args):
    idx, sk, out = args
    import numpy as _np
    from simplipy.verify._monitor import judge_pair
    try:
        v, d = judge_pair(list(sk), list(out), _np.random.default_rng(0))
    except Exception as ex:
        return (idx, 'JUDGE_ERROR', f'{type(ex).__name__}: {ex}'[:200])
    return (idx, v, str(d)[:200])


def adjudicate(corpus, outputs, changed_mask, label, workers=24):
    """Judge every changed Const-free row; returns {idx: verdict}."""
    tasks = [(i, corpus[i], outputs[i]) for i in np.where(changed_mask)[0]]
    print(f'{label}: judging {len(tasks)} changed rows ...', flush=True)
    verdicts = {}
    with Pool(workers) as pool:
        for k, (idx, v, d) in enumerate(pool.imap_unordered(_judge_worker, tasks,
                                                            chunksize=32)):
            verdicts[idx] = v
            if (k + 1) % 8192 == 0:
                print(f'  {label} {k + 1}/{len(tasks)}', flush=True)
    counts = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    print(f'  {label} verdicts: {counts}', flush=True)
    return verdicts, counts


def main():
    corpus = pickle.load(open(CORPUS_PKL, 'rb'))
    r012 = pickle.load(open(os.path.join(HERE, 'ecdf_vs_sympy_results.pkl'), 'rb'))
    r011 = pickle.load(open(os.path.join(HERE, 'ecdf_011_results.pkl'), 'rb'))
    e = SimpliPyEngine.from_config(os.path.join(REPO, 'remine', 'acj-4-3',
                                                'config.yaml'))
    import yaml
    ops = yaml.safe_load(open(os.path.join(REPO, 'remine', 'acj-4-3',
                                           'config.yaml')))['operators']
    bare = SimpliPyEngine(operators=ops, rules=[])

    const_free = np.array(['<constant>' not in r for r in corpus])
    out11 = [desugar(o)[0] for o in r011['outputs']]
    out12 = [bare.simplify(o) for o in
             r012['simplipy']['acj-4-3']['outputs']]
    sy = r012['sympy']
    out_sy = [toks if status == 'ok' else None for (_, toks, status) in sy]

    mu_o = np.array([e.complexity(r) for r in corpus], float)
    mu11 = np.array([e.complexity(o) for o in out11], float)
    mu12 = np.array([e.complexity(o) for o in out12], float)
    mu_sy = np.array([e.complexity(o) if o is not None else np.nan for o in out_sy],
                     float)

    # "changed" = the output is a different token sequence than the input's own
    # canonical projection (spelling-invariant via the bare projection of the input).
    canon_in = [bare.simplify(r) for r in corpus]
    ch11 = np.array([const_free[i] and out11[i] != canon_in[i]
                     for i in range(len(corpus))])
    ch12 = np.array([const_free[i] and out12[i] != canon_in[i]
                     for i in range(len(corpus))])
    chsy = np.array([const_free[i] and out_sy[i] is not None
                     and out_sy[i] != canon_in[i] for i in range(len(corpus))])

    results = {}
    all_counts = {}
    for label, outs, mask, mu in (('v0.11', out11, ch11, mu11),
                                  ('v0.12', out12, ch12, mu12),
                                  ('sympy', out_sy, chsy, mu_sy)):
        verdicts, counts = adjudicate(corpus, outs, mask, label)
        refused = np.zeros(len(corpus), bool)
        for i, v in verdicts.items():
            if v not in ('OK', 'F64_FOLD'):
                refused[i] = True
        r = mu.copy() / mu_o
        r[np.isnan(r)] = 1.0          # timeouts / no spelling: unsimplified
        r[refused] = 1.0              # UNSOUND rewrite: refused, ratio 1
        cf = const_free
        results[label] = {'ratio': r, 'refused': refused}
        all_counts[label] = counts
        print(f'{label}: const-free rows {cf.sum()}, changed {mask.sum()}, '
              f'refused-unsound {refused.sum()}; sound-only ratio_mean '
              f'{np.mean(r[cf]):.4f}, frac<1 {np.mean(r[cf] < 1):.4f}', flush=True)

    cf = const_free
    r11, r12, rsy = (results['v0.11']['ratio'], results['v0.12']['ratio'],
                     results['sympy']['ratio'])
    summary = {
        'scope': 'const-free rows only (judge-adjudicable subset)',
        'n_const_free': int(cf.sum()),
        'n_const_bearing_uncompared': int((~cf).sum()),
        'judge_verdicts': all_counts,
        'refused_unsound': {k: int(v['refused'].sum()) for k, v in results.items()},
        'sound_only': {
            k: {'ratio_mean': float(np.mean(results[k]['ratio'][cf])),
                'frac_below_1': float(np.mean(results[k]['ratio'][cf] < 1)),
                'frac_below_0.9': float(np.mean(results[k]['ratio'][cf] < 0.9))}
            for k in results},
        'paired_012_vs_011': {
            'win_012': float(np.mean(r12[cf] < r11[cf])),
            'tie': float(np.mean(r12[cf] == r11[cf])),
            'win_011': float(np.mean(r12[cf] > r11[cf]))},
        'deep_region_below_0.9': {
            'rows': int(((r11[cf] < 0.9) | (r12[cf] < 0.9)).sum()),
            'win_012': float(np.mean((r12[cf] < r11[cf])[(r11[cf] < 0.9)
                                                         | (r12[cf] < 0.9)])),
            'win_011': float(np.mean((r12[cf] > r11[cf])[(r11[cf] < 0.9)
                                                         | (r12[cf] < 0.9)])),
        },
    }
    json.dump(summary, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    n = int(cf.sum())
    for label, r, color in (('v0.11 (sound-only)', r11, 'tab:blue'),
                            ('v0.12 (sound-only)', r12, 'tab:green'),
                            ('sympy (sound-only)', rsy, 'tab:red')):
        v = np.sort(r[cf])
        ax.plot(v, np.arange(1, n + 1) / n, color=color, label=label, lw=1.6)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r'simplification ratio  $\mu(\mathrm{simp})/\mu(\mathrm{orig})$')
    ax.set_ylabel('ECDF')
    ax.set_title('Sound-only scoring, const-free 64k subset: unsound rewrites '
                 'refused (ratio = 1)', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    os.makedirs(os.path.join(REPO, 'benchmarks', 'out'), exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(REPO, 'benchmarks', 'out',
                                 f'simplipy_sound_only.{ext}'), dpi=160)
    print('figure -> benchmarks/out/simplipy_sound_only.{png,svg}')


if __name__ == '__main__':
    main()
