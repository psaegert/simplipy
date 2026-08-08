"""Extensions of ecdf_vs_sympy: node-count rescoring + per-region example dump.

Consumes benchmarks/ecdf_vs_sympy_results.pkl (run ecdf_vs_sympy.py first).

1. NODE-COUNT scoring: the same outputs re-scored by the node count of the explicit
   prefix form (one token = one node of the binary tree): originals and the SymPy
   bridge output are already explicit prefix (len()); SimpliPy outputs are converted
   tagged -> explicit through the idempotent ``simplify(out, form='explicit')``
   (idempotence at scale is gate-verified, so this is a pure form conversion).
   This isolates the yardstick question: mu prices signs and small literals FREE, so
   token-shrinking but mu-neutral rewrites count as "improved" here and as "unchanged"
   under mu.

2. REGION EXAMPLES (acj-4-3 SOUND, mu ratios): samples from ratio < 0.9,
   0.9 <= ratio < 1, ratio == 1, ratio > 1, printed as infix with mu and node counts,
   plus forensics on the ratio > 1 class: is the output a fixpoint (re-simplify), and
   does the input re-simplify to the same state (spelling-dependent endpoint)?

Usage: ecdf_vs_sympy_ext.py [--regions-only | --nodes-only]
Artifacts: benchmarks/ecdf_vs_sympy_nodes_summary.json (tracked),
           assets/images/simplipy_vs_sympy_012_nodes.{png,svg},
           benchmarks/ecdf_region_examples.md (tracked).
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from simplipy import SimpliPyEngine  # noqa: E402

RESULTS_PKL = os.path.join(HERE, "ecdf_vs_sympy_results.pkl")
CORPUS_PKL = os.path.join(REPO, "remine", "corpus_nv_64k.pkl")
NODES_JSON = os.path.join(HERE, "ecdf_vs_sympy_nodes_summary.json")
EXAMPLES_MD = os.path.join(HERE, "ecdf_region_examples.md")


def node_scores(corpus, res, e):
    # Form projection must NOT re-simplify: a variant's outputs are fixpoints of THEIR
    # engine, not of acj-4-3, so converting through a ruled engine silently converges
    # every variant to that engine's answer (found live: 2-1/3-2/4-3 measured
    # identical). A RULES-FREE engine shares the constructors/canon of every cell, so
    # for already-canonical states its simplify is a pure form projection. The lossy
    # variant projects in LOSSY mode (its states need lossy canon to be fixpoints).
    from simplipy import Mode
    import yaml
    ops = yaml.safe_load(open(os.path.join(REPO, 'remine', 'acj-4-3',
                                           'config.yaml')))['operators']
    bare = SimpliPyEngine(operators=ops, rules=[])
    n_orig = np.array([len(r) for r in corpus], dtype=float)
    scored = {'n_orig': n_orig}
    for label, d in res['simplipy'].items():
        kw = {'mode': Mode.LOSSY} if label.endswith('lossy') else {}
        n = np.array([len(bare.simplify(o, form='explicit', **kw))
                      for o in d['outputs']], dtype=float)
        scored[label] = n
        print(f'  {label}: node ratio mean {np.mean(n / n_orig):.4f} '
              f'frac<1 {np.mean(n / n_orig < 1):.3f}', flush=True)
    sy_n = []
    for (dt, toks, status) in res['sympy']:
        sy_n.append(float(len(toks)) if status == 'ok' else np.nan)
    scored['sympy'] = np.array(sy_n)
    return scored


def nodes_figure(scored, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_orig = scored['n_orig']
    n = len(n_orig)

    def ecdf(v):
        v = np.sort(np.asarray(v))
        return v, np.arange(1, len(v) + 1) / n

    greens = plt.cm.Greens
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    BUDGETS = (1, 2, 4, 8, 16)
    cols = {
        'Mined Rulesets': [(f'acj-{c}', greens(0.45 + 0.2 * i))
                           for i, c in enumerate(('2-1', '3-2', '4-3'))],
        'Safe vs Aggressive': [('acj-4-3', greens(0.85)),
                               ('acj-4-3-lossy', 'tab:purple')],
        'Search Budget': [(f'acj-4-3-b{b}', greens(0.4 + 0.45 * i / (len(BUDGETS) - 1)))
                          for i, b in enumerate(BUDGETS)]
                         + [('acj-4-3', greens(0.85))],
    }
    sy_ratio_1 = np.where(np.isnan(scored['sympy']), 1.0, scored['sympy'] / n_orig)
    sy_ratio_none = (scored['sympy'] / n_orig)[~np.isnan(scored['sympy'])]
    for j, (title, variants) in enumerate(cols.items()):
        ax = axes[j]
        for label, color in variants:
            x, y = ecdf(scored[label] / n_orig)
            ax.plot(x, y, color=color, label=label, lw=1.6)
        x, y = ecdf(sy_ratio_none)
        ax.plot(x, y, color='tab:orange', label='sympy (ratio=None)', lw=1.6)
        x, y = ecdf(sy_ratio_1)
        ax.plot(x, y, color='tab:red', label='sympy (ratio=1)', lw=1.6)
        ax.set_title(title)
        ax.set_xlabel('simplification ratio  nodes(simp)/nodes(orig)')
        ax.set_xlim(0, 1.6)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('ECDF')
    fig.suptitle('Same outputs, node-count yardstick: explicit prefix nodes(simp)/nodes(orig) '
                 '— 64k nv corpus', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(REPO, 'assets', 'images',
                                 f'simplipy_vs_sympy_012_nodes.{ext}'), dpi=160)
    print('figure -> assets/images/simplipy_vs_sympy_012_nodes.{png,svg}')


def nodes_summary(scored):
    n_orig = scored['n_orig']
    s = {}
    for label, v in scored.items():
        if label in ('n_orig', 'sympy'):
            continue
        r = v / n_orig
        s[label] = {'ratio_p50': float(np.percentile(r, 50)),
                    'ratio_mean': float(np.mean(r)),
                    'frac_below_1': float(np.mean(r < 1)),
                    'frac_above_1': float(np.mean(r > 1))}
    r1 = np.where(np.isnan(scored['sympy']), 1.0, scored['sympy'] / n_orig)
    s['sympy'] = {'ratio1_p50': float(np.percentile(r1, 50)),
                  'ratio1_mean': float(np.mean(r1)),
                  'frac_below_1_ratio1': float(np.mean(r1 < 1)),
                  'frac_above_1_ratio1': float(np.mean(r1 > 1))}
    return s


def region_examples(corpus, res, e, n_per_region=12, seed=20260802):
    rng = np.random.default_rng(seed)
    mu_orig = np.array([e.complexity(r) for r in corpus], dtype=float)
    d = res['simplipy']['acj-4-3']
    mu_out = np.array([e.complexity(o) for o in d['outputs']], dtype=float)
    ratio = mu_out / mu_orig
    regions = {
        'ratio < 0.9': np.where(ratio < 0.9)[0],
        '0.9 <= ratio < 1': np.where((ratio >= 0.9) & (ratio < 1))[0],
        'ratio == 1': np.where(ratio == 1)[0],
        'ratio > 1': np.where(ratio > 1)[0],
    }
    lines = ['# Region examples — acj-4-3 SOUND, mu ratios (64k nv corpus)', '']
    for name, idx in regions.items():
        lines.append(f'## {name}  ({len(idx)} rows, {len(idx) / len(corpus):.2%})')
        lines.append('')
        take = rng.choice(idx, size=min(n_per_region, len(idx)), replace=False)
        for i in sorted(take.tolist()):
            # infix of the ORIGINAL without simplifying is not expressible;
            # show explicit prefix for the original and infix for the output
            # (infix form returns a STRING for token input).
            out_inf = e.simplify(d['outputs'][i], form='infix')
            lines.append(f'- **row {i}** | mu {mu_orig[i]:.0f} -> {mu_out[i]:.0f} '
                         f'(ratio {ratio[i]:.3f}) | nodes {len(corpus[i])} -> '
                         f'{len(e.simplify(d["outputs"][i], form="explicit"))}')
            lines.append(f'  - orig: `{" ".join(corpus[i])}`')
            lines.append(f'  - simp: `{out_inf}`')
            if name == 'ratio > 1':
                out = d['outputs'][i]
                refix = e.simplify(out)
                idem = (list(refix) == list(out))
                re_in = e.simplify(corpus[i])
                same = (list(re_in) == list(out))
                lines.append(f'  - forensics: output-is-fixpoint={idem}; '
                             f'fresh simplify(orig) == stored output: {same}; '
                             f'mu(fresh)={e.complexity(re_in):.0f}')
        lines.append('')
    open(EXAMPLES_MD, 'w').write('\n'.join(lines))
    print(f'examples -> {EXAMPLES_MD}')
    return regions


def main():
    args = sys.argv[1:]
    corpus = pickle.load(open(CORPUS_PKL, 'rb'))
    res = pickle.load(open(RESULTS_PKL, 'rb'))
    e = SimpliPyEngine.from_config(os.path.join(REPO, 'remine', 'acj-4-3', 'config.yaml'))
    if '--regions-only' not in args:
        scored = node_scores(corpus, res, e)
        summary = nodes_summary(scored)
        json.dump(summary, open(NODES_JSON, 'w'), indent=1)
        print(json.dumps(summary, indent=1))
        nodes_figure(scored, res)
    if '--nodes-only' not in args:
        region_examples(corpus, res, e)


if __name__ == '__main__':
    main()
