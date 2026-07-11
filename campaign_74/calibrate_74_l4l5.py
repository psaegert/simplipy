"""7-4 go/no-go calibration (BLOCKER 3 of simplipy MINE_7_4_READINESS_2026-07-11.md).

Runs on the REAL 13-leaf dev_7-4 mining config (4 dummy variables + 9 constant leaves,
38 operators, X=1024 heavy-tailed mixture, challenges=retries=16, rtol=1e-9/atol=1e-12,
stage-2 confirmation on):

  Phase 1  complete L<=4 mine, candidate fold-filter ON (the production path).
           -> wall/CPU time, per-length rule counts, the mined ruleset (JSON).
  Phase 2  PARITY (Doc A protocol): N_L4 uniform length-4 sources + N_L5 uniform
           length-5 sources, Kruskal-pruned by the phase-1 ruleset; each survivor is
           decided via find_rule_lib against the FILTERED and the UNFILTERED <=4
           candidate library with the SAME per-source seed (fit seeds are
           order-independent, so decisions can differ ONLY through the dropped
           var-free candidates). Required outcome: zero divergence.
  Phase 3  COST ANCHOR: N_COST uniform length-5 sources -> Kruskal survivors ->
           per-source CPU-seconds against the filtered library. This is the number
           the readiness decision rule reads (<= ~2 CPU-s -> commit tiers A+C and
           offer complete-L6; > ~5 CPU-s -> tier A + sampled-6/7 only).

Results are appended to <out>/CALIB_RESULTS.json after every phase (crash-safe);
ruleset at <out>/calib_l4_rules.json; progress on stdout (line-buffered via -u).

Usage: python -u calibrate_74_l4l5.py --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Local checkout by default; on a remote box set CALIB_SIMPLIPY_SRC= (empty) to use the
# installed wheel and CALIB_CONFIG to the deployed dev_7-3 config.yaml.
SIMPLIPY_SRC = os.environ.get('CALIB_SIMPLIPY_SRC', '/home/psaegert/Projects/simplipy/src')
if SIMPLIPY_SRC and os.path.isdir(SIMPLIPY_SRC):
    sys.path.insert(0, SIMPLIPY_SRC)

from simplipy import SimpliPyEngine  # noqa: E402
from simplipy.utils import count_expressions, enumerate_expressions, sample_expression  # noqa: E402

CONFIG = os.environ.get(
    'CALIB_CONFIG', '/home/psaegert/Projects/simplipy/simplipy-assets/engines/dev_7-3/config.yaml')
DUMMY = ['x0', 'x1', 'x2', 'x3']
EXTRA = ['<constant>', '0', '1', '(-1)', 'np.e', 'np.pi',
         'float("inf")', 'float("-inf")', 'float("nan")']
RTOL, ATOL = 1e-9, 1e-12
CH, RT = 16, 16
MAX_TARGET = 4
MINE_SEED = 42
SAMPLE_SEED = 7
PARITY_THREADS = int(os.environ.get('CALIB_PARITY_THREADS', '24'))


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def fresh_engine() -> SimpliPyEngine:
    e = SimpliPyEngine.from_config(CONFIG)
    assert e._core is not None, 'compiled core missing'
    return e


def save(out: str, results: dict) -> None:
    with open(os.path.join(out, 'CALIB_RESULTS.json'), 'w') as f:
        json.dump(results, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true',
                    help='tiny run to validate the harness end-to-end (~minutes)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    max_source = 3 if args.smoke else 4
    x_rows = 256 if args.smoke else 1024
    # Parity weight sits on the L5 sources: they scan the length-4 candidate arm, which is
    # where the fold-filter actually bites (L4 sources only ever scan candidates <= 3).
    n_l4, n_l5, n_cost = (40, 15, 15) if args.smoke else (1000, 500, 2000)
    parity_len_hi = max_source + 1  # the "one above the mined range" parity length

    git_sha = os.environ.get('CALIB_SIMPLIPY_SHA', '')
    if not git_sha and SIMPLIPY_SRC and os.path.isdir(SIMPLIPY_SRC):
        git_sha = subprocess.run(
            ['git', '-C', os.path.dirname(SIMPLIPY_SRC), 'rev-parse', 'HEAD'],
            capture_output=True, text=True).stdout.strip()
    results: dict = {
        'meta': {
            'host': platform.node(), 'started': time.strftime('%Y-%m-%d %H:%M:%S %z'),
            'simplipy_sha': git_sha, 'smoke': args.smoke,
            'rayon_threads': os.environ.get('RAYON_NUM_THREADS'),
            'config': {'max_source': max_source, 'max_target': MAX_TARGET,
                       'dummies': DUMMY, 'extra': EXTRA, 'x_rows': x_rows,
                       'challenges': CH, 'retries': RT, 'rtol': RTOL, 'atol': ATOL,
                       'mine_seed': MINE_SEED, 'sample_seed': SAMPLE_SEED,
                       'n_parity_l4': n_l4, 'n_parity_l5': n_l5, 'n_cost': n_cost},
        },
    }
    save(args.out, results)

    # ---------------- Phase 1: complete L<=max_source mine, fold-filter ON --------------
    print(f'[P1] complete L<={max_source} mine (fold-filter ON) ...', flush=True)
    eng = fresh_engine()
    w0, c0 = time.perf_counter(), cpu_seconds()
    eng.find_rules(
        max_source_pattern_length=max_source,
        max_target_pattern_length=MAX_TARGET,
        dummy_variables=len(DUMMY),
        extra_internal_terms=EXTRA,
        X=x_rows,
        constants_fit_challenges=CH,
        constants_fit_retries=RT,
        output_file=os.path.join(args.out, 'calib_l4_rules.json'),
        seed=MINE_SEED,
        verbose=True,
        candidate_fold_filter=True,
    )
    p1_wall, p1_cpu = time.perf_counter() - w0, cpu_seconds() - c0
    rules = [(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules]
    by_len: dict[int, int] = {}
    for lhs, _ in rules:
        by_len[len(lhs)] = by_len.get(len(lhs), 0) + 1
    results['phase1'] = {'wall_s': round(p1_wall, 1), 'cpu_s': round(p1_cpu, 1),
                         'n_rules': len(rules), 'rules_by_lhs_len': by_len}
    save(args.out, results)
    print(f'[P1] DONE {len(rules)} rules, wall {p1_wall:.0f}s cpu {p1_cpu:.0f}s', flush=True)

    # Shared assets for phases 2+3: mine X (as the mine draws it), candidate libraries,
    # exact-count DP for uniform source sampling.
    core = eng._core
    rng_x = np.random.default_rng(MINE_SEED)
    X = eng._mining_sample_x(x_rows, len(DUMMY), rng_x)
    x_flat = X.flatten(order='C').tolist()
    non_leaf = dict(eng.operator_arity)
    leaves = DUMMY + EXTRA
    exprs = enumerate_expressions(leaves, non_leaf, MAX_TARGET)
    candidates = [list(t) for length in sorted(exprs) for t in sorted(exprs[length])]
    counts = count_expressions(len(leaves), non_leaf, parity_len_hi)
    print(f'[libs] {len(candidates):,} candidates <= {MAX_TARGET}', flush=True)
    lib_f = core.build_candidate_library(candidates, DUMMY, x_flat, x_rows, fold_filter=True)
    lib_u = core.build_candidate_library(candidates, DUMMY, x_flat, x_rows, fold_filter=False)
    results['library'] = {'total': lib_u.n_candidates, 'kept_filtered': lib_f.n_candidates,
                          'dropped_var_free': lib_f.n_filtered}
    save(args.out, results)
    print(f"[libs] filtered keeps {lib_f.n_candidates:,}, drops {lib_f.n_filtered:,}", flush=True)

    def draw_sources(length: int, n: int, seed: int) -> list[list[str]]:
        rng = np.random.default_rng(seed)
        seen: set[tuple[str, ...]] = set()
        while len(seen) < n:
            seen.add(sample_expression(length, leaves, non_leaf, counts, rng))
        return [list(t) for t in sorted(seen)]

    def survivors(srcs: list[list[str]]) -> list[tuple[list[str], int]]:
        out = []
        for s in srcs:
            slen = len(eng.simplify(s))
            if slen >= len(s):  # Kruskal: skip sources the current rules already shorten
                out.append((s, slen))
        return out

    # ---------------- Phase 2: sampled parity, filtered vs unfiltered -------------------
    print('[P2] parity sampling ...', flush=True)
    par_srcs = (survivors(draw_sources(max_source, n_l4, SAMPLE_SEED))
                + survivors(draw_sources(parity_len_hi, n_l5, SAMPLE_SEED + 1)))
    results['phase2'] = {'n_sampled': n_l4 + n_l5, 'n_survivors': len(par_srcs)}
    save(args.out, results)

    def decide(lib, src: list[str], slen: int, seed: int):
        return core.find_rule_lib(src, slen, MAX_TARGET, lib,
                                  challenges=CH, retries=RT, seed=seed,
                                  rtol=RTOL, atol=ATOL)

    def parity_one(item):
        i, (src, slen) = item
        seed = 100_000 + i
        rf = decide(lib_f, src, slen, seed)
        ru = decide(lib_u, src, slen, seed)
        return (src, rf, ru) if rf != ru else None

    w0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARITY_THREADS) as ex:
        mismatches = [m for m in ex.map(parity_one, enumerate(par_srcs)) if m is not None]
    results['phase2'].update({
        'wall_s': round(time.perf_counter() - w0, 1),
        'n_mismatches': len(mismatches),
        'mismatches': mismatches[:20],
        'parity': 'EXACT' if not mismatches else 'DIVERGED',
    })
    save(args.out, results)
    print(f"[P2] DONE parity={results['phase2']['parity']} "
          f"({len(par_srcs)} survivors, {len(mismatches)} mismatches)", flush=True)

    # ---------------- Phase 3: per-source cost anchor on the filtered library -----------
    print('[P3] cost anchor sampling ...', flush=True)
    cost_all = draw_sources(parity_len_hi, n_cost, SAMPLE_SEED + 2)
    cost_srcs = survivors(cost_all)
    walls: list[float] = []

    def cost_one(item):
        i, (src, slen) = item
        t0 = time.perf_counter()
        r = decide(lib_f, src, slen, 200_000 + i)
        return time.perf_counter() - t0, r is not None

    w0, c0 = time.perf_counter(), cpu_seconds()
    with ThreadPoolExecutor(max_workers=PARITY_THREADS) as ex:
        outcomes = list(ex.map(cost_one, enumerate(cost_srcs)))
    p3_wall, p3_cpu = time.perf_counter() - w0, cpu_seconds() - c0
    walls = sorted(o[0] for o in outcomes)
    n_hit = sum(1 for o in outcomes if o[1])
    per_source_cpu = p3_cpu / max(1, len(cost_srcs))
    results['phase3'] = {
        'n_sampled': n_cost, 'n_survivors': len(cost_srcs),
        'survivor_frac': round(len(cost_srcs) / max(1, n_cost), 4),
        'n_rules_found': n_hit,
        'wall_s': round(p3_wall, 1), 'cpu_s': round(p3_cpu, 1),
        'per_source_cpu_s_mean': round(per_source_cpu, 3),
        'per_call_wall_s': {
            'median': round(walls[len(walls) // 2], 3) if walls else None,
            'p90': round(walls[int(len(walls) * 0.9)], 3) if walls else None,
            'max': round(walls[-1], 3) if walls else None,
        },
        'note': ('per_source_cpu_s_mean is the readiness decision number '
                 '(<=~2 commit A+C offer B; >~5 tier A + sampled-6/7 only); '
                 'per_call_wall_s under thread contention is indicative only'),
    }
    results['meta']['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    save(args.out, results)
    print(f'[P3] DONE per-source CPU {per_source_cpu:.2f}s '
          f'(survivors {len(cost_srcs)}/{n_cost}, rules found {n_hit})', flush=True)
    print('[CALIBRATION COMPLETE]', flush=True)


if __name__ == '__main__':
    main()
