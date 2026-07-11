"""Stratified 7-4 cost probe: source-length x source-class x candidate-arm.

For L in {5,6,7} and source class in {const-free, const-bearing}, sample N uniform
sources per stratum (exactly uniform via the count-weighted sampler; const-free =
the 12-leaf universe, const-bearing = 13-leaf rejection sampling), Kruskal-prune
with the calibration's complete L<=4 ruleset, and measure per-survivor CPU-s of
find_rule_lib against:
  (a) FULL      the production fold-filtered <=4 library (sound config), and
  (b) CF-ARM    only the const-free candidates of that library (the hypothetical
                cheap sweep; UNSOUND alone for minimality -- the cf-min lemma is
                false -- priced here for the stage-1-screen design).
Also records survivor fractions and rule hits per stratum (yield anchors).

Output: STRATA_RESULTS.json in --out. Usage: python -u stratified_cost_probe.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SIMPLIPY_SRC = os.environ.get('CALIB_SIMPLIPY_SRC', '/home/psaegert/Projects/simplipy/src')
if SIMPLIPY_SRC and os.path.isdir(SIMPLIPY_SRC):
    sys.path.insert(0, SIMPLIPY_SRC)

from simplipy import SimpliPyEngine  # noqa: E402
from simplipy.utils import count_expressions, enumerate_expressions, sample_expression  # noqa: E402

CONFIG = os.environ.get(
    'CALIB_CONFIG', '/home/psaegert/Projects/simplipy/simplipy-assets/engines/dev_7-3/config.yaml')
RULES_L4 = os.environ.get('CALIB_L4_RULES', 'calib_out/calib_l4_rules.json')
DUMMY = ['x0', 'x1', 'x2', 'x3']
EXTRA = ['<constant>', '0', '1', '(-1)', 'np.e', 'np.pi',
         'float("inf")', 'float("-inf")', 'float("nan")']
RTOL, ATOL = 1e-9, 1e-12
CH, RT = 16, 16
MAX_TARGET = 4
N_PER_STRATUM = int(os.environ.get('N_PER_STRATUM', '300'))
THREADS = int(os.environ.get('CALIB_PARITY_THREADS', '28'))


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eng = SimpliPyEngine.from_config(CONFIG)
    assert eng._core is not None
    with open(RULES_L4) as f:
        rules = json.load(f)
    eng.simplification_rules = [(tuple(l), tuple(r)) for l, r in rules]
    eng.compile_rules()
    eng._core.set_rules([(list(l), list(r)) for l, r in eng.simplification_rules])
    core = eng._core
    print(f'[setup] {len(rules):,} L<=4 rules loaded for Kruskal pruning', flush=True)

    non_leaf = dict(eng.operator_arity)
    leaves13 = DUMMY + EXTRA
    leaves12 = [t for t in leaves13 if t != '<constant>']
    counts13 = count_expressions(len(leaves13), non_leaf, 7)
    counts12 = count_expressions(len(leaves12), non_leaf, 7)

    X = eng._mining_sample_x(1024, len(DUMMY), np.random.default_rng(42))
    x_flat = X.flatten(order='C').tolist()
    exprs = enumerate_expressions(leaves13, non_leaf, MAX_TARGET)
    cands_all = [list(t) for length in sorted(exprs) for t in sorted(exprs[length])]
    cands_cf = [c for c in cands_all if '<constant>' not in c]
    lib_full = core.build_candidate_library(cands_all, DUMMY, x_flat, 1024, fold_filter=True)
    # CF-ARM: replicate the fold-filter manually (bare <constant> absent -> filter inert),
    # so the arm equals exactly the const-free candidates the FULL library scans.
    has_var = lambda c: any(t in DUMMY for t in c)  # noqa: E731
    cands_cf_kept = [c for c in cands_cf if len(c) == 1 or has_var(c)]
    lib_cf = core.build_candidate_library(cands_cf_kept, DUMMY, x_flat, 1024, fold_filter=False)
    print(f'[libs] full kept {lib_full.n_candidates:,} | cf-arm {lib_cf.n_candidates:,}', flush=True)

    def draw(length: int, klass: str, n: int, seed: int) -> list[list[str]]:
        rng = np.random.default_rng(seed)
        seen: set = set()
        if klass == 'cf':
            while len(seen) < n:
                seen.add(sample_expression(length, leaves12, non_leaf, counts12, rng))
        else:
            while len(seen) < n:  # rejection: uniform over the cb stratum
                t = sample_expression(length, leaves13, non_leaf, counts13, rng)
                if '<constant>' in t:
                    seen.add(t)
        return [list(t) for t in sorted(seen)]

    results: dict = {'meta': {'n_per_stratum': N_PER_STRATUM, 'threads': THREADS,
                              'library_full': lib_full.n_candidates,
                              'library_cf_arm': lib_cf.n_candidates,
                              'n_l4_rules': len(rules)}}
    for length in (5, 6, 7):
        for klass in ('cf', 'cb'):
            srcs = draw(length, klass, N_PER_STRATUM, 1000 * length + (0 if klass == 'cf' else 1))
            surv = []
            for s in srcs:
                slen = len(eng.simplify(list(s)))
                if slen >= len(s):
                    surv.append((s, slen))
            row: dict = {'n': len(srcs), 'survivors': len(surv),
                         'survivor_frac': round(len(surv) / len(srcs), 4)}
            for libname, lib in (('full', lib_full), ('cf_arm', lib_cf)):
                def one(item):
                    i, (s, slen) = item
                    t0 = time.perf_counter()
                    r = core.find_rule_lib(s, slen, MAX_TARGET, lib, challenges=CH, retries=RT,
                                           seed=300_000 + i, rtol=RTOL, atol=ATOL)
                    return time.perf_counter() - t0, r is not None
                c0 = cpu_seconds()
                with ThreadPoolExecutor(max_workers=THREADS) as ex:
                    outs = list(ex.map(one, enumerate(surv)))
                dcpu = cpu_seconds() - c0
                walls = sorted(o[0] for o in outs)
                row[libname] = {
                    'cpu_s_per_survivor': round(dcpu / max(1, len(surv)), 4),
                    'rules_found': sum(1 for o in outs if o[1]),
                    'wall_median': round(walls[len(walls) // 2], 4) if walls else None,
                    'wall_p90': round(walls[int(len(walls) * 0.9)], 4) if walls else None,
                }
            results[f'L{length}_{klass}'] = row
            print(f'[L{length} {klass}] surv {row["survivor_frac"]:.2f} '
                  f'full {row["full"]["cpu_s_per_survivor"]:.3f}s/src '
                  f'(hits {row["full"]["rules_found"]}) | cf-arm '
                  f'{row["cf_arm"]["cpu_s_per_survivor"]:.4f}s/src '
                  f'(hits {row["cf_arm"]["rules_found"]})', flush=True)
            with open(os.path.join(args.out, 'STRATA_RESULTS.json'), 'w') as f:
                json.dump(results, f, indent=2)
    print('[STRATA PROBE COMPLETE]', flush=True)


if __name__ == '__main__':
    main()
