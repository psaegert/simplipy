"""Full dev_i-j re-mine driver for a remote CPU box (quadxeon7). Mines FRESH with the current Rust
miner (challenges=16, retries=16, n_samples=1024, NO budget cap) and writes the new rules.json +
per-phase info. Also runs the independent soundness gate on the result.

Usage:
  RAYON_NUM_THREADS=<n> python run_full_mine.py --i 7 --j 3 --config assets/dev_7-3/config.yaml \
      --out out/dev_7-3_remined_rules.json
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))
sys.path.insert(0, os.path.dirname(__file__))
from simplipy import SimpliPyEngine
from mine_grid import native_mine, soundness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--challenges", type=int, default=16)
    ap.add_argument("--retries", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] full re-mine dev_{a.i}-{a.j} "
          f"(challenges={a.challenges} retries={a.retries} n_samples={a.n_samples} "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS','all')})", flush=True)
    t0 = time.perf_counter()
    eng = SimpliPyEngine.from_config(a.config)
    rules, info = native_mine(eng, a.i, a.j, n_samples=a.n_samples,
                              challenges=a.challenges, retries=a.retries, seed=0)
    dt = time.perf_counter() - t0
    # rules -> [[lhs],[rhs]] json (the engine's rules.json format)
    out_rules = [[list(l), list(r)] for l, r in rules]
    json.dump(out_rules, open(a.out, "w"))
    json.dump({"i": a.i, "j": a.j, "n_rules": len(rules), "minutes": round(dt/60, 1),
               "per_length": info["per_length"], "challenges": a.challenges, "retries": a.retries},
              open(a.out + ".info", "w"), indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] mined {len(rules)} rules in {dt/3600:.2f} h -> {a.out}", flush=True)
    print(f"  per_length: {info['per_length']}", flush=True)
    s, n = soundness(eng, rules)
    print(f"  SOUNDNESS: {s}/{n} verify as true equivalences ({100*s/max(n,1):.2f}%)", flush=True)

if __name__ == "__main__":
    main()
