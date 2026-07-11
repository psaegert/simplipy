"""Stage-2 for LLM proposals rejected as 'no_rule_found': lift the target-length limit.

Two tiers, run in order per proposal:
  T5   minimal-certified target of length exactly 5: find_rule_lib against the <=5
       fold-filtered candidate library (shortest-first scan -> the <=5 MINIMAL target;
       expensive per source, fine for hundreds of proposals).
  HINT gate-certified proposal pair: if the LLM supplied a target hint (valid tokens,
       shorter than the source, wildcard-multiplicity-safe), certify (source, hint) via
       _confirm_mined_rules on BOTH the mine X and the independent 2x-wide X. Sound and
       length-reducing, but carries NO minimality certificate -> tagged 'hint_verified'.

Usage: python -u certify_long_targets.py --rejected llm_round2/rejected.json \
           --rules <snapshot rules.json> --out llm_round2_long
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SIMPLIPY_SRC = os.environ.get('CALIB_SIMPLIPY_SRC', '/home/psaegert/Projects/simplipy/src')
if SIMPLIPY_SRC and os.path.isdir(SIMPLIPY_SRC):
    sys.path.insert(0, SIMPLIPY_SRC)
os.environ.setdefault('OMP_NUM_THREADS', '1')

from simplipy import SimpliPyEngine  # noqa: E402
from simplipy.utils import enumerate_expressions  # noqa: E402

CONFIG = os.environ.get(
    'CALIB_CONFIG', '/home/psaegert/Projects/simplipy/simplipy-assets/engines/dev_7-3/config.yaml')
DUMMY = ['x0', 'x1', 'x2', 'x3']
EXTRA = ['<constant>', '0', '1', '(-1)', 'np.e', 'np.pi',
         'float("inf")', 'float("-inf")', 'float("nan")']
RTOL, ATOL = 1e-9, 1e-12
CH, RT = 16, 16
SEED = 42
X_ROWS = 1024
T5_MAX_TARGET = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rejected', required=True)
    ap.add_argument('--proposals', default='llm_proposals_round2.json',
                    help='original proposals (for target hints)')
    ap.add_argument('--rules', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    try:
        os.sched_setaffinity(0, set(range(24, 32)))
        os.nice(10)
    except OSError:
        pass

    eng = SimpliPyEngine.from_config(CONFIG)
    assert eng._core is not None
    rules = json.load(open(args.rules))
    eng.simplification_rules = [(tuple(l), tuple(r)) for l, r in rules]
    eng.compile_rules()
    eng._core.set_rules([(list(l), list(r)) for l, r in eng.simplification_rules])

    rng = np.random.default_rng(SEED)
    X = eng._mining_sample_x(X_ROWS, len(DUMMY), rng)
    X_confirm = eng._mining_sample_x(2 * X_ROWS, len(DUMMY), rng)
    mine_seed = int(rng.integers(2 ** 62))
    confirm_seed = int(rng.integers(2 ** 62))
    min_informative = X_ROWS // 8

    hints = {tuple(str(t) for t in p.get('source', [])): p.get('expected_target')
             for p in json.load(open(args.proposals))}
    todo = [r for r in json.load(open(args.rejected)) if r['verdict'] == 'no_rule_found']
    print(f'{len(todo)} no_rule_found proposals to lift', flush=True)

    non_leaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
    print('building <=5 candidate library (21M raw, minutes) ...', flush=True)
    t0 = time.perf_counter()
    exprs = enumerate_expressions(DUMMY + EXTRA, non_leaf, T5_MAX_TARGET)
    cands = [list(t) for length in sorted(exprs) for t in sorted(exprs[length])]
    lib5 = eng._core.build_candidate_library(
        cands, DUMMY, X.flatten(order='C').tolist(), X_ROWS, fold_filter=True)
    del exprs, cands
    print(f'lib <=5: {lib5.n_candidates:,} kept, {lib5.n_filtered:,} filtered '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)

    vocab = set(DUMMY + EXTRA) | set(eng.operator_arity)
    out_t5, out_hint, still = [], [], []
    t0 = time.time()
    for i, r in enumerate(todo):
        src = r['source']
        slen = len(eng.simplify(list(src)))
        target = eng._core.find_rule_lib(
            list(src), slen, T5_MAX_TARGET, lib5, challenges=CH, retries=RT,
            seed=mine_seed + (len(src) << 40) + 950_000_000 + i, rtol=RTOL, atol=ATOL)
        if target is not None and len(target) < len(src):
            kept = eng._confirm_mined_rules(
                [(tuple(src), tuple(target))], DUMMY, X_confirm, CH, RT, RTOL, ATOL,
                max(1, min_informative * 2), confirm_seed + (len(src) << 40) + i)
            if kept:
                out_t5.append({**r, 'target': target, 'tier': 'minimal_le5'})
                continue
        hint = hints.get(tuple(src))
        if hint:
            hint = [str(t) for t in hint]
            from simplipy.utils import violates_wildcard_multiplicity  # noqa: E402
            if (set(hint) <= vocab and len(hint) < len(src) and eng.is_valid(list(hint))
                    and not violates_wildcard_multiplicity(list(src), list(hint))):
                k1 = eng._confirm_mined_rules(
                    [(tuple(src), tuple(hint))], DUMMY, X, CH, RT, RTOL, ATOL,
                    min_informative, mine_seed + (len(src) << 40) + 960_000_000 + i)
                k2 = eng._confirm_mined_rules(
                    [(tuple(src), tuple(hint))], DUMMY, X_confirm, CH, RT, RTOL, ATOL,
                    max(1, min_informative * 2), confirm_seed + (len(src) << 40) + 500 + i)
                if k1 and k2:
                    out_hint.append({**r, 'target': hint, 'tier': 'hint_verified_nonminimal'})
                    continue
        still.append(r)
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(todo)} ({time.time() - t0:.0f}s) '
                  f't5={len(out_t5)} hint={len(out_hint)}', flush=True)

    json.dump(out_t5 + out_hint, open(os.path.join(args.out, 'certified_long.json'), 'w'), indent=1)
    json.dump(still, open(os.path.join(args.out, 'still_rejected.json'), 'w'), indent=1)
    print(json.dumps({'input': len(todo), 'minimal_le5': len(out_t5),
                      'hint_verified': len(out_hint), 'still_rejected': len(still)}, indent=1))
    print(f'DONE in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
