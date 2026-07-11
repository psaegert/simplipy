"""Periodic 7-4 mine telemetry: pull the growing ruleset from solomon, benchmark on valkyrie.

Each round (intended cadence ~5h, driven by the wrapper loop):
  1. scp solomon:~/simplipy-mine-74/mine_out/{rules_74.json,campaign_provenance.json}
     (temp + JSON-validate before accepting; skip the round if unchanged by hash).
  2. Build a snapshot engine (dev_7-3 operators + the pulled rules) and benchmark
     `simplify` over the PAPER's fixed 4,096-skeleton corpus with the l_max sweep 0..7 --
     PAIRED with the dev_7-3 baseline in the same process/core/conditions, because the
     per-skeleton RATIO is robust to machine load while absolute times are not
     (mirrors experimental/eval/_simplipy_speed_bench.py: gc disabled, blake2b hashes).
  3. Record capability too: simplified-length distribution vs baseline + how many
     skeletons simplify differently (the new rules' actual reach on decode outputs).
  4. Append a summary row to bench_timeline.jsonl; keep the full per-pull pickle.

Interference policy: pinned to ONE core (sched_setaffinity, CPU 31), nice 10 -- never
above the box's experiments. Load average + experiment flags recorded per round so
polluted absolute timings are identifiable; ratios stay interpretable throughout.

Usage: python -u pull_and_bench_74.py --workdir <dir> [--once]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

SIMPLIPY_SRC = '/home/psaegert/Projects/simplipy/src'
sys.path.insert(0, SIMPLIPY_SRC)
os.environ.setdefault('OMP_NUM_THREADS', '1')

from simplipy import SimpliPyEngine  # noqa: E402

REMOTE = 'solomon:~/simplipy-mine-74/mine_out'
SKELETONS = '/home/psaegert/Projects/flash-ansr/results/simplification/rustbench_skeletons_v23.0-3M_4096.pkl'
DEV_CONFIG = '/home/psaegert/Projects/simplipy/simplipy-assets/engines/dev_7-3/config.yaml'
DEV_RULES = '/home/psaegert/Projects/simplipy/simplipy-assets/engines/dev_7-3/rules.json'
L_MAX = [0, 1, 2, 3, 4, 5, 6, 7, None]  # None = unlimited (the new deployed default)
PIN_CPU = 31
BENCH_NICE = 10


def stable_hash(tokens: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(' '.join(tokens).encode(), digest_size=8).digest(), 'big')


def bench_paired(eng_new: SimpliPyEngine, eng_base: SimpliPyEngine, skeletons: list) -> tuple[dict, dict]:
    """STRICTLY per-skeleton interleaved timing of both engines, so each ratio's two legs
    run microseconds apart -- load drift between sequential whole-engine blocks would
    otherwise bias the paired ratios this telemetry leans on under co-load."""
    out_new: dict = {}
    out_base: dict = {}
    n = len(skeletons)
    for lm in L_MAX:
        rec = {k: {'times': np.empty(n), 'lengths': np.empty(n, dtype=np.int32),
                   'hashes': np.empty(n, dtype=np.uint64)} for k in ('new', 'base')}
        gc.disable()
        for i, s in enumerate(skeletons):
            for key, engine in (('new', eng_new), ('base', eng_base)):
                t0 = time.perf_counter()
                r = engine.simplify(list(s), max_pattern_length=lm)
                rec[key]['times'][i] = time.perf_counter() - t0
                rec[key]['lengths'][i] = len(r)
                rec[key]['hashes'][i] = stable_hash(r) % (2 ** 64)
        gc.enable()
        out_new[lm], out_base[lm] = rec['new'], rec['base']
    return out_new, out_base


def pull(workdir: str) -> tuple[str, dict] | None:
    tmp_rules = os.path.join(workdir, '_pull_rules.json')
    tmp_prov = os.path.join(workdir, '_pull_prov.json')
    r = subprocess.run(['scp', '-q', f'{REMOTE}/rules_74.json', tmp_rules],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        print(f'[pull] scp failed for rules_74.json: {r.stderr.decode()[:200]}', flush=True)
        return None
    # campaign provenance appears at the phase-A barrier; during phase A the per-length
    # find_rules sidecar is the live progress record. Either (or none, very early) is fine.
    prov: dict = {}
    for candidate in ('campaign_provenance.json', 'rules_74.json.provenance.json'):
        r = subprocess.run(['scp', '-q', f'{REMOTE}/{candidate}', tmp_prov],
                           capture_output=True, timeout=120)
        if r.returncode == 0:
            try:
                prov = json.load(open(tmp_prov))
                prov['_source'] = candidate
                break
            except json.JSONDecodeError:
                continue
    try:
        rules = json.load(open(tmp_rules))
    except json.JSONDecodeError:
        print('[pull] rules JSON truncated mid-write on solomon; skipping round', flush=True)
        return None
    return tmp_rules, {'rules': rules, 'prov': prov}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    try:
        os.sched_setaffinity(0, {PIN_CPU})
        os.nice(BENCH_NICE)
    except OSError:
        pass

    os.environ.setdefault('SSH_AUTH_SOCK', os.path.expanduser('~/.ssh/agent.sock'))
    pulled = pull(args.workdir)
    if pulled is None:
        return
    tmp_rules, data = pulled
    rules_hash = hashlib.blake2b(
        json.dumps(data['rules']).encode(), digest_size=16).hexdigest()
    timeline = os.path.join(args.workdir, 'bench_timeline.jsonl')
    if os.path.exists(timeline):
        last = [json.loads(line) for line in open(timeline)]
        if last and last[-1]['rules_hash'] == rules_hash:
            print(f'[bench] ruleset unchanged ({len(data["rules"]):,} rules); skipping round', flush=True)
            return

    stamp = time.strftime('%Y%m%d_%H%M%S')
    snap = os.path.join(args.workdir, f'snapshot_{stamp}')
    os.makedirs(snap, exist_ok=True)
    os.replace(tmp_rules, os.path.join(snap, 'rules.json'))
    cfg_text = open(DEV_CONFIG).read()
    # dev config points at its own rules file; retarget the snapshot copy
    with open(os.path.join(snap, 'config.yaml'), 'w') as f:
        f.write(cfg_text.replace('./rules.json', 'rules.json') if './rules.json' in cfg_text
                else cfg_text)
    import re
    cfg = open(os.path.join(snap, 'config.yaml')).read()
    cfg = re.sub(r'rules:.*', 'rules: ./rules.json', cfg)
    with open(os.path.join(snap, 'config.yaml'), 'w') as f:
        f.write(cfg)

    skeletons = pickle.load(open(SKELETONS, 'rb'))
    load1, load5, _ = os.getloadavg()
    hybrid = subprocess.run(['pgrep', '-f', 'hybrid_grid_ge[n].py'], capture_output=True).returncode == 0
    calib = subprocess.run(['pgrep', '-f', 'calibrate_74_l4l5.p[y]'], capture_output=True).returncode == 0

    print(f'[bench] {stamp}: {len(data["rules"]):,} rules; load {load1:.1f}; '
          f'hybrid={hybrid} calib={calib}', flush=True)
    t0 = time.time()
    eng_new = SimpliPyEngine.from_config(os.path.join(snap, 'config.yaml'))
    eng_base = SimpliPyEngine.from_config(DEV_CONFIG)
    res_new, res_base = bench_paired(eng_new, eng_base, skeletons)

    row: dict = {'stamp': stamp, 'rules_hash': rules_hash, 'n_rules': len(data['rules']),
                 'mine_done': data['prov'].get('finished') is not None,
                 'phases_done': data['prov'].get('phases', {}).get('final') is not None,
                 'state_done': list(data['prov'].get('phases', {}).keys()),
                 'load1': round(load1, 2), 'load5': round(load5, 2),
                 'hybrid_running': hybrid, 'calib_running': calib,
                 'bench_wall_s': None, 'per_lmax': {}}
    for lm in L_MAX:
        tn, tb = res_new[lm]['times'], res_base[lm]['times']
        ln, lb = res_new[lm]['lengths'], res_base[lm]['lengths']
        row['per_lmax'][str(lm)] = {
            'median_us_new': round(float(np.median(tn)) * 1e6, 2),
            'median_us_base': round(float(np.median(tb)) * 1e6, 2),
            'median_ratio': round(float(np.median(tn / np.maximum(tb, 1e-9))), 3),
            'mean_len_new': round(float(ln.mean()), 3),
            'mean_len_base': round(float(lb.mean()), 3),
            'n_diff_outputs': int((res_new[lm]['hashes'] != res_base[lm]['hashes']).sum()),
        }
    row['bench_wall_s'] = round(time.time() - t0, 1)
    with open(os.path.join(snap, 'bench_full.pkl'), 'wb') as f:
        pickle.dump({'new': res_new, 'base': res_base, 'row': row}, f)
    with open(timeline, 'a') as f:
        f.write(json.dumps(row) + '\n')
    lm4 = row['per_lmax']['4']
    lm7 = row['per_lmax']['7']
    print(f"[bench] DONE in {row['bench_wall_s']}s | L4: {lm4['median_us_new']}us vs "
          f"{lm4['median_us_base']}us (x{lm4['median_ratio']}), dlen "
          f"{lm4['mean_len_base'] - lm4['mean_len_new']:+.3f}, diff {lm4['n_diff_outputs']} | "
          f"L7: {lm7['median_us_new']}us, dlen {lm7['mean_len_base'] - lm7['mean_len_new']:+.3f}, "
          f"diff {lm7['n_diff_outputs']}", flush=True)


if __name__ == '__main__':
    main()
