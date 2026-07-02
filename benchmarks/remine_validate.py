"""Scoped re-mine validation: mine dev_5-3 FRESH with the current Rust miner and diff (bidirectionally,
per LHS length) against the shipped dev_7-3 rules restricted to LHS<=5. Confirms whether the shipped
ruleset's incompleteness (37% at len-4 on a sample) is real + compounds across lengths, and whether the
current miner LOSES any shipped rules (strictly-more-complete check).

Run (valkyrie, capped cores): RAYON_NUM_THREADS=16 python benchmarks/remine_validate.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
from simplipy import SimpliPyEngine
from mine_grid import native_mine

CONFIG = os.path.join(os.path.dirname(__file__), "..", "simplipy-assets", "engines", "dev_7-3", "config.yaml")
RULES = os.path.join(os.path.dirname(__file__), "..", "simplipy-assets", "engines", "dev_7-3", "rules.json")
OUT = os.environ.get("OUT", "/tmp/remine_dev5-3_validate.json")

def main():
    t0 = time.perf_counter()
    eng = SimpliPyEngine.from_config(CONFIG)
    print(f"[{time.strftime('%H:%M:%S')}] mining dev_5-3 FRESH with the current miner "
          f"(RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS','all')}) ...", flush=True)
    rules, info = native_mine(eng, 5, 3, n_samples=1024, challenges=16, retries=16, seed=0)
    dt = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] mined {len(rules)} canonical rules in {dt/60:.1f} min; "
          f"per_length={info['per_length']}", flush=True)

    cur = set((tuple(l), tuple(r)) for l, r in rules)
    shipped_all = json.load(open(RULES))
    shipped = set((tuple(l), tuple(r)) for l, r in shipped_all if len(l) <= 5)

    def by_len(s):
        c = Counter(len(l) for l, r in s)
        return {k: c[k] for k in sorted(c)}

    new = cur - shipped           # current produces, shipped lacks (the incompleteness a re-mine fixes)
    missing = shipped - cur       # shipped has, current lacks (would a re-mine LOSE these?)
    common = cur & shipped

    result = {
        "mine_minutes": round(dt / 60, 1),
        "n_current": len(cur), "n_shipped_le5": len(shipped),
        "new_count": len(new), "missing_count": len(missing), "common_count": len(common),
        "new_by_len": by_len(new), "missing_by_len": by_len(missing),
        "current_by_len": by_len(cur), "shipped_by_len": by_len(shipped),
        "per_length_mine": info["per_length"],
        "new_examples": [[list(l), list(r)] for l, r in list(new)[:40]],
        "missing_examples": [[list(l), list(r)] for l, r in list(missing)[:40]],
    }
    json.dump(result, open(OUT, "w"), indent=2)
    print("\n=== dev_5-3 re-mine vs shipped (LHS<=5) ===", flush=True)
    print(f"  current miner: {len(cur)} rules | shipped<=5: {len(shipped)}", flush=True)
    print(f"  NEW (current has, shipped lacks): {len(new)}  by_len={by_len(new)}", flush=True)
    print(f"  MISSING (shipped has, current lacks): {len(missing)}  by_len={by_len(missing)}", flush=True)
    print(f"  COMMON: {len(common)}", flush=True)
    print(f"  incompleteness of shipped per length: "
          f"{ {k: f'{by_len(new).get(k,0)}/{by_len(cur).get(k,0)}' for k in sorted(by_len(cur))} }", flush=True)
    print(f"  saved -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
