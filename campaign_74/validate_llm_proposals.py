"""Validate + certify LLM-proposed rule sources against the mine's own machinery.

The LLM only PROPOSES sources (its target hints are ignored for soundness): each proposal
is (1) syntax-checked (vocabulary, prefix arity, length 5-7), (2) deduplicated,
(3) Kruskal-checked against the CURRENT ruleset snapshot (already-reducible -> covered),
(4) certified by find_rule_lib against the complete fold-filtered <=4 candidate library
(the same shortest-first scan the mine runs -> certified-MINIMAL target or None),
(5) stage-2 confirmed on the independent 2x-wide X. Output pairs are exactly as sound as
mined ones; provenance marks them as an 'llm_proposals' stratum (proposal-source label,
not a soundness difference).

Usage: python -u validate_llm_proposals.py --proposals raw.json --rules <snapshot rules.json> --out <dir>
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
MAX_TARGET = 4
SEED = 42
X_ROWS = 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--proposals', required=True)
    ap.add_argument('--rules', required=True, help='current ruleset snapshot (Kruskal reference)')
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

    # find_rules-identical X / seeds (engine.py derivation order)
    rng = np.random.default_rng(SEED)
    X = eng._mining_sample_x(X_ROWS, len(DUMMY), rng)
    X_confirm = eng._mining_sample_x(2 * X_ROWS, len(DUMMY), rng)
    mine_seed = int(rng.integers(2 ** 62))
    confirm_seed = int(rng.integers(2 ** 62))
    min_informative = X_ROWS // 8

    non_leaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
    exprs = enumerate_expressions(DUMMY + EXTRA, non_leaf, MAX_TARGET)
    cands = [list(t) for length in sorted(exprs) for t in sorted(exprs[length])]
    lib = eng._core.build_candidate_library(
        cands, DUMMY, X.flatten(order='C').tolist(), X_ROWS, fold_filter=True)
    vocab = set(DUMMY + EXTRA) | set(eng.operator_arity)

    proposals = json.load(open(args.proposals))
    stats = {'total': len(proposals), 'invalid_syntax': 0, 'duplicate': 0,
             'already_reducible': 0, 'no_rule_found': 0, 'confirm_rejected': 0,
             'already_known_lhs': 0, 'certified': 0}
    known_lhs = {tuple(l) for l, _ in eng.simplification_rules}
    seen: set = set()
    certified, rejected = [], []
    t0 = time.time()
    for i, p in enumerate(proposals):
        src = [str(t) for t in p.get('source', [])]
        meta = {'source': src, 'why': p.get('why', ''), 'family': p.get('family')}
        if (not 5 <= len(src) <= 13 or not set(src) <= vocab
                or not eng.is_valid(list(src))):
            stats['invalid_syntax'] += 1
            rejected.append({**meta, 'verdict': 'invalid_syntax'})
            continue
        key = tuple(src)
        if key in seen:
            stats['duplicate'] += 1
            continue
        seen.add(key)
        if key in known_lhs:
            stats['already_known_lhs'] += 1
            rejected.append({**meta, 'verdict': 'already_known_lhs'})
            continue
        slen = len(eng.simplify(list(src)))
        if slen < len(src):
            stats['already_reducible'] += 1
            rejected.append({**meta, 'verdict': 'already_reducible',
                             'reduces_to_len': slen})
            continue
        target = eng._core.find_rule_lib(
            list(src), slen, MAX_TARGET, lib, challenges=CH, retries=RT,
            seed=mine_seed + (len(src) << 40) + 900_000_000 + i,
            rtol=RTOL, atol=ATOL)
        if target is None:
            stats['no_rule_found'] += 1
            rejected.append({**meta, 'verdict': 'no_rule_found'})
            continue
        kept = eng._confirm_mined_rules(
            [(tuple(src), tuple(target))], DUMMY, X_confirm, CH, RT, RTOL, ATOL,
            max(1, round(min_informative * 2)), confirm_seed + (len(src) << 40) + i)
        if not kept:
            stats['confirm_rejected'] += 1
            rejected.append({**meta, 'verdict': 'confirm_rejected', 'target': target})
            continue
        stats['certified'] += 1
        certified.append({**meta, 'target': target,
                          'hint_matched': target == p.get('expected_target')})
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(proposals)} ({time.time() - t0:.0f}s) '
                  f'certified so far: {stats["certified"]}', flush=True)

    json.dump(certified, open(os.path.join(args.out, 'certified.json'), 'w'), indent=1)
    json.dump(rejected, open(os.path.join(args.out, 'rejected.json'), 'w'), indent=1)
    json.dump(stats, open(os.path.join(args.out, 'stats.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    print(f'DONE in {time.time() - t0:.0f}s -> {args.out}/certified.json', flush=True)


if __name__ == '__main__':
    main()
