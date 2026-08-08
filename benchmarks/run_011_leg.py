"""Prior-version leg: simplipy==0.11.0 (legacy 4-3 artifact) over the 64k nv corpus.

RUNS INSIDE THE PINNED 0.11 VENV (never the 0.12 env): the corpus is resugared
rootn -> pow1_k at the boundary (0.11 predates rootn; every corpus rootn index is a
literal in {2,3,4,5} by construction of the nv desugar), timed per call single-threaded
(median of REPS), outputs saved in the legacy token language for the 0.12 side to
desugar and score (mu + explicit node count) on common ground.

Usage: <venv011>/python run_011_leg.py
Artifact: benchmarks/ecdf_011_results.pkl (untracked, regenerable).
"""
import gc
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CORPUS_PKL = os.path.join(REPO, "remine", "corpus_nv_64k.pkl")
OUT_PKL = os.path.join(HERE, "ecdf_011_results.pkl")
REPS = 3

from simplipy import SimpliPyEngine  # noqa: E402


def resugar(tokens):
    """rootn X k -> pow1_k X (postfix index literal to prefix unary)."""
    out = []
    i = 0
    n = len(tokens)

    def walk():
        nonlocal i
        t = tokens[i]
        i += 1
        if t == 'rootn':
            arg = walk()
            k = tokens[i]
            i += 1
            assert k in ('2', '3', '4', '5'), f'non-literal rootn index {k}'
            return [f'pow1_{k}'] + arg
        if t in ('+', '-', '*', '/', 'pow'):
            return [t] + walk() + walk()
        if t in ('abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos',
                 'cosh', 'exp', 'inv', 'log', 'neg', 'sin', 'sinh', 'tan', 'tanh'):
            return [t] + walk()
        return [t]

    out = walk()
    assert i == n, 'trailing tokens'
    return out


def main():
    import simplipy
    assert simplipy.__version__.startswith('0.11'), simplipy.__version__
    corpus = pickle.load(open(CORPUS_PKL, 'rb'))
    rows = [resugar(r) for r in corpus]
    e = SimpliPyEngine.load('4-3', install=True)
    print(f'0.11.0, legacy 4-3, {len(e.simplification_rules)} rules, '
          f'{len(rows)} rows', flush=True)
    for r in rows[:20]:
        e.simplify(r)
    times, outputs, errors = [], [], 0
    gc.disable()
    t_start = time.time()
    for j, r in enumerate(rows):
        try:
            ts = []
            res = None
            for _ in range(REPS):
                t0 = time.perf_counter_ns()
                res = e.simplify(r)
                ts.append(time.perf_counter_ns() - t0)
            times.append(float(np.median(ts)) / 1e9)
            outputs.append(list(res))
        except Exception:
            errors += 1
            times.append(None)
            outputs.append(None)
        if (j + 1) % 8192 == 0:
            print(f'  {j + 1}/{len(rows)}  ({time.time() - t_start:.0f}s)', flush=True)
    gc.enable()
    pickle.dump({'seconds': times, 'outputs': outputs, 'errors': errors,
                 'version': simplipy.__version__, 'artifact': '4-3',
                 'n_rules': len(e.simplification_rules)},
                open(OUT_PKL, 'wb'))
    ok = np.array([t for t in times if t is not None])
    print(f'done: errors={errors} p50={np.percentile(ok, 50)*1e6:.0f}us '
          f'mean={ok.mean()*1e6:.0f}us')


if __name__ == '__main__':
    main()
