"""Cross-version head-to-head: 0.11.0 (legacy 4-3) vs 0.12 (acj-4-3) on common ground.

Consumes benchmarks/ecdf_011_results.pkl (run_011_leg.py inside the pinned 0.11 venv)
and benchmarks/ecdf_vs_sympy_results.pkl (the 0.12 run). The 0.11 outputs are spelled
in the legacy hyper-op language; they are desugared at the boundary (the SAME desugar
that produced the corpus, remine/desugar_corpus.py) and then BOTH versions' outputs are
scored two ways against the same denominators:

  * mu       -- SimpliPyEngine.complexity of the 0.12 engine (the MDL yardstick);
  * nodes    -- explicit-prefix node count (len of the binary prefix spelling; the
                0.11 outputs and the corpus are already explicit binary prefix).

Per-row win/tie/loss under each yardstick is reported alongside the ECDFs so "which
one is better" is answered by paired comparison, not just marginals.

RUNS IN THE 0.12 ENV.
Artifacts: benchmarks/ecdf_011_vs_012_summary.json (tracked),
           assets/images/simplipy_011_vs_012.{png,svg}.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "remine"))
from desugar_corpus import desugar  # noqa: E402
from simplipy import SimpliPyEngine  # noqa: E402

CORPUS_PKL = os.path.join(REPO, "remine", "corpus_nv_64k.pkl")
R012_PKL = os.path.join(HERE, "ecdf_vs_sympy_results.pkl")
R011_PKL = os.path.join(HERE, "ecdf_011_results.pkl")
OUT_JSON = os.path.join(HERE, "ecdf_011_vs_012_summary.json")


def main():
    corpus = pickle.load(open(CORPUS_PKL, 'rb'))
    r012 = pickle.load(open(R012_PKL, 'rb'))
    r011 = pickle.load(open(R011_PKL, 'rb'))
    e = SimpliPyEngine.from_config(os.path.join(REPO, 'remine', 'acj-4-3',
                                                'config.yaml'))
    import yaml
    ops = yaml.safe_load(open(os.path.join(REPO, 'remine', 'acj-4-3',
                                           'config.yaml')))['operators']
    bare = SimpliPyEngine(operators=ops, rules=[])

    print('desugaring 0.11 outputs ...', flush=True)
    out011 = [desugar(o)[0] if o is not None else None for o in r011['outputs']]
    out012 = r012['simplipy']['acj-4-3']['outputs']

    print('scoring ...', flush=True)
    mu_orig = np.array([e.complexity(r) for r in corpus], dtype=float)
    n_orig = np.array([len(r) for r in corpus], dtype=float)
    mu11 = np.array([e.complexity(o) if o is not None else np.nan for o in out011],
                    dtype=float)
    mu12 = np.array([e.complexity(o) for o in out012], dtype=float)
    n11 = np.array([len(o) if o is not None else np.nan for o in out011], dtype=float)
    n12 = np.array([len(bare.simplify(o, form='explicit')) for o in out012],
                   dtype=float)
    t11 = np.array([t if t is not None else np.nan for t in r011['seconds']],
                   dtype=float)
    t12 = np.array(r012['simplipy']['acj-4-3']['seconds'], dtype=float)

    def wtl(a, b):
        """(win, tie, loss) fractions of a vs b (smaller wins)."""
        ok = ~(np.isnan(a) | np.isnan(b))
        return {'win_012': float(np.mean(a[ok] < b[ok])),
                'tie': float(np.mean(a[ok] == b[ok])),
                'win_011': float(np.mean(a[ok] > b[ok]))}

    summary = {
        'n_rows': len(corpus),
        'errors_011': int(r011['errors']),
        'rules': {'0.11_4-3': r011['n_rules'], '0.12_acj-4-3': 1176},
        'time_us': {'0.11_p50': float(np.nanpercentile(t11, 50) * 1e6),
                    '0.11_mean': float(np.nanmean(t11) * 1e6),
                    '0.12_p50': float(np.nanpercentile(t12, 50) * 1e6),
                    '0.12_mean': float(np.nanmean(t12) * 1e6)},
        'mu': {'0.11_ratio_mean': float(np.nanmean(mu11 / mu_orig)),
               '0.11_frac_below_1': float(np.nanmean((mu11 / mu_orig) < 1)),
               '0.11_frac_above_1': float(np.nanmean((mu11 / mu_orig) > 1)),
               '0.12_ratio_mean': float(np.mean(mu12 / mu_orig)),
               '0.12_frac_below_1': float(np.mean((mu12 / mu_orig) < 1)),
               '0.12_frac_above_1': float(np.mean((mu12 / mu_orig) > 1)),
               'paired': wtl(mu12, mu11)},
        'nodes': {'0.11_ratio_mean': float(np.nanmean(n11 / n_orig)),
                  '0.11_frac_below_1': float(np.nanmean((n11 / n_orig) < 1)),
                  '0.11_frac_above_1': float(np.nanmean((n11 / n_orig) > 1)),
                  '0.12_ratio_mean': float(np.mean(n12 / n_orig)),
                  '0.12_frac_below_1': float(np.mean((n12 / n_orig) < 1)),
                  '0.12_frac_above_1': float(np.mean((n12 / n_orig) > 1)),
                  'paired': wtl(n12, n11)},
    }
    json.dump(summary, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    n = len(corpus)

    def ecdf(v):
        v = np.sort(np.asarray(v)[~np.isnan(v)])
        return v, np.arange(1, len(v) + 1) / n

    for ax, (a11, a12, xlabel, xlim) in zip(axes, [
            (t11, t12, 'simplification time [s]', None),
            (mu11 / mu_orig, mu12 / mu_orig,
             r'simplification ratio  $\mu(\mathrm{simp})/\mu(\mathrm{orig})$', (0, 1.6)),
            (n11 / n_orig, n12 / n_orig,
             'simplification ratio  nodes(simp)/nodes(orig)', (0, 1.6))]):
        x, y = ecdf(a11)
        ax.plot(x, y, color='tab:blue', label=f"0.11.0 4-3 ({r011['n_rules']} rules)",
                lw=1.6)
        x, y = ecdf(a12)
        ax.plot(x, y, color='tab:green', label='0.12 acj-4-3 (1176 rules)', lw=1.6)
        if xlim is None:
            ax.set_xscale('log')
        else:
            ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc='lower right')
    axes[0].set_ylabel('ECDF')
    fig.suptitle('simplipy 0.11.0 (legacy 4-3) vs 0.12 (acj-4-3) — same 64k corpus, '
                 'both scored by the 0.12 mu and by explicit node count', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(REPO, 'assets', 'images',
                                 f'simplipy_011_vs_012.{ext}'), dpi=160)
    print('figure -> assets/images/simplipy_011_vs_012.{png,svg}')


if __name__ == '__main__':
    main()
