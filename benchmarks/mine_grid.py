"""Offline rule-mining GRID harness (Phase B, M4 driver).

Times the OFFLINE rule-finding phase for a grid of dev_i-j configs (i = max_source, j = max_target,
i>j), ALL CORES, to map the scaling surface and extrapolate to larger configs.

Architecture (hybrid): Python keeps the cheap outer loop (generation + `deduplicate_rules`
canonicalization + the length barrier); Rust does the hot inner loop in parallel
(`mine_one_length`: rayon over a length's sources -> Kruskal-prune + find_rule). The growing
Kruskal rule set is pushed back into the Rust engine with `set_rules` between lengths.

Usage:
    python benchmarks/mine_grid.py --smoke              # tiny config + faithfulness vs Python find_rules
    python benchmarks/mine_grid.py --grid I J [I J ...] # run named configs, record offline wall-clock
    python benchmarks/mine_grid.py --full               # the full i>j, i=2..7 grid (HEAVY -- needs a free box)
"""
import sys, os, time, argparse, json, warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simplipy import SimpliPyEngine
from simplipy.utils import construct_expressions, deduplicate_rules

warnings.filterwarnings("ignore")

CONFIG = os.path.join(os.path.dirname(__file__), "..", "simplipy-assets", "engines", "dev_7-3", "config.yaml")
EXTRA = ['<constant>', '0', '1', '(-1)', 'np.pi', 'np.e', 'float("inf")', 'float("-inf")', 'float("nan")']


def generate(eng, max_source, dummy):
    """Replicate find_rules Phase-1 generation -> expressions_of_length (incl. the over-produced tail)."""
    leaf = dummy + EXTRA
    nonleaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
    eol = defaultdict(set)
    nl = defaultdict(set)
    for l in leaf:
        eol[1].add((l,))
    new = set()
    while max(eol.keys()) < max_source:
        for e in construct_expressions(eol, nonleaf, must_have_sizes=new):
            nl[len(e)].add(e)
        new = set()
        before = {k: len(v) for k, v in eol.items()}
        for k, v in nl.items():
            eol[k].update(v)
        after = {k: len(v) for k, v in nl.items()}
        for k in after:
            if k not in before or after[k] > before[k]:
                new.add(k)
        for k, v in nl.items():
            eol[k].update(v)
        nl.clear()
    return eol


def native_mine(eng, max_source, max_target, n_samples=1024, challenges=16, retries=16, seed=0,
                rng=None, budget_s=None, chunk=50000):
    """The native all-cores mine for dev_(max_source)-(max_target): patterns (sources) up to length
    `max_source`, candidate replacements up to length `max_target`. Returns (rules, info) with per-phase
    wall-clock. `budget_s` caps the mine phase (checked between source chunks) -> info['incomplete']."""
    rng = rng or np.random.default_rng(seed)
    max_leaf = int(max_source - (max_source - 1) / 2)
    dummy = [f"x{i}" for i in range(max_leaf)]
    t = {}
    t0 = time.perf_counter()
    eol = generate(eng, max_source, dummy)
    t["generate"] = time.perf_counter() - t0

    candidates = [list(e) for L in sorted(eol) if L <= max_target for e in eol[L]]
    X = rng.normal(0, 5, size=(n_samples, len(dummy))).astype(np.float64)
    xf = X.flatten(order="C").tolist()
    t0 = time.perf_counter()
    lib = eng._core.build_candidate_library(candidates, dummy, xf, n_samples)
    t["build_library"] = time.perf_counter() - t0

    rules = []
    per_len = {}
    incomplete = False
    t0 = time.perf_counter()
    # i-j semantics: sources of length up to max_source ONLY (drop the generation's over-produced
    # tail -- `construct_expressions` reaches len 7 via binary(len3,len3) regardless of max_source).
    for L in sorted(k for k in eol if k <= max_source):
        eng._core.set_rules(rules)
        sources_L = [list(e) for e in eol[L]]
        tl = time.perf_counter()
        found_L = []
        for c0 in range(0, len(sources_L), chunk):
            found_L += eng._core.mine_one_length(
                sources_L[c0:c0 + chunk], lib, max_target, challenges, retries, seed + L, 1e-5, 1e-8)
            if budget_s and (time.perf_counter() - t0) > budget_s:
                incomplete = True
                break
        per_len[L] = (len(sources_L), len(found_L), round(time.perf_counter() - tl, 3))
        if found_L:
            rules = deduplicate_rules(rules + [tuple(map(tuple, r)) for r in found_L], dummy)
        print(f"    [len {L}] sources={len(sources_L)} found={len(found_L)} "
              f"pass={per_len[L][2]:.0f}s cum_rules={len(rules)} elapsed={time.perf_counter()-t0:.0f}s",
              flush=True)
        if incomplete:
            break
    t["mine"] = time.perf_counter() - t0
    t["total"] = t["generate"] + t["build_library"] + t["mine"]
    return rules, {"timing": t, "per_length": per_len, "n_rules": len(rules), "incomplete": incomplete,
                   "n_candidates": len(candidates), "dummy": dummy}


def soundness(eng, rules, n_samples=512, seed=7):
    """Independent gate: every mined rule must be a TRUE equivalence. Treat the canonical wildcard
    tokens `_j` as the variables; for a const-free target check the no-constant equivalence test, for a
    const-bearing target check `exist_constants_fit` on the source's output. Returns (n_sound, n_total).
    NOTE: the Python `find_rules` cannot serve as the reference here -- it returns 0 rules with the Rust
    core (the forked worker uses the immutable `_core`); soundness + the M4a per-source gate cover it."""
    import re as _re
    wc = _re.compile(r"^_\d+$")
    allv = sorted({t for lhs, rhs in rules for t in list(lhs) + list(rhs) if wc.match(t)},
                  key=lambda v: int(v[1:]))
    rng = np.random.default_rng(seed)
    xf = rng.normal(0, 5, size=(n_samples, max(len(allv), 1))).astype(np.float64).flatten(order="C").tolist()
    nc = lambda e: sum(1 for t in e if t == "<constant>")
    sound = checked = 0
    for lhs, rhs in rules:
        lhs, rhs = list(lhs), list(rhs)
        try:
            if nc(rhs) == 0:
                ok = eng._core.equivalent_no_const(lhs, rhs, allv, xf, n_samples, 16, 1e-5, 1e-8)
            else:
                ns = nc(lhs)
                p = rng.normal(0, 5, max(ns, 1)).tolist()
                y = eng._core.evaluate_batch(lhs, allv, xf, n_samples, p[:ns])
                ok = eng._core.exist_constants_fit(rhs, allv, xf, n_samples, list(y), 1e-5, 1e-8, 16, 0)
            checked += 1
            sound += 1 if ok else 0
        except Exception:
            pass
    return sound, checked


def smoke():
    eng = SimpliPyEngine.from_config(CONFIG)
    i, j = 3, 2  # fast (1 generation pass, len<=3); dev_4-2 is ~5 min (4.7M over-produced sources)
    print(f"=== SMOKE: native mine dev_{i}-{j} + soundness gate ===")
    nat_rules, info = native_mine(eng, i, j, n_samples=512, challenges=16, retries=16, seed=0)
    print(f"  native: {info['n_rules']} rules, {info['n_candidates']} candidates, timing {info['timing']}")
    print(f"  per-length (sources, found, s): {info['per_length']}")
    s, n = soundness(eng, nat_rules)
    print(f"  SOUNDNESS: {s}/{n} mined rules verify as true equivalences ({100*s/max(n,1):.1f}%)")
    print("  (Python find_rules cannot cross-check: it returns 0 rules with the Rust core -- a separate bug.)")


def run_grid(configs, n_samples=1024, out=None, budget_s=1800):
    # cheapest-first: smaller candidate-library (j) dominates per-source cost, then smaller i.
    configs = sorted(configs, key=lambda ij: (ij[1], ij[0]))
    eng = SimpliPyEngine.from_config(CONFIG)
    results = []
    for (i, j) in configs:
        print(f"--- dev_{i}-{j} (budget {budget_s}s) ---", flush=True)
        rules, info = native_mine(eng, i, j, n_samples=n_samples, seed=0, budget_s=budget_s)
        rec = {"i": i, "j": j, **info["timing"], "n_rules": info["n_rules"],
               "incomplete": info["incomplete"], "n_candidates": info["n_candidates"],
               "per_length": info["per_length"]}
        results.append(rec)
        print(f"    dev_{i}-{j}: total={rec['total']:.1f}s mine={rec['mine']:.1f}s rules={rec['n_rules']} "
              f"incomplete={rec['incomplete']}", flush=True)
        if out:
            json.dump(results, open(out, "w"), indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid", nargs="+", type=int, help="flat I J I J ... pairs")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--budget-s", type=int, default=1800, help="per-config mine wall-clock cap (s)")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.full:
        configs = [(i, j) for i in range(2, 8) for j in range(1, i)]
        run_grid(configs, n_samples=args.n_samples, out=args.out or "mine_grid_results.json", budget_s=args.budget_s)
    elif args.grid:
        flat = args.grid
        configs = list(zip(flat[0::2], flat[1::2]))
        run_grid(configs, n_samples=args.n_samples, out=args.out, budget_s=args.budget_s)
    else:
        ap.print_help()
