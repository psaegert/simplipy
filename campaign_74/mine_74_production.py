"""PRODUCTION 7-4 mine, stratified max-feasible schedule (2026-07-11 calibration verdict).

  Phase A  complete lengths 2-5 (both classes) via find_rules VERBATIM (fold-filter on,
           stage-2 confirm, provenance sidecar). ~4.4 solomon-days.
  Phase B  EXHAUSTIVE length-6 const-bearing stratum (87,925,893 sources, exact-count-
           validated slice generation; rules FIXED within the length). ~2.8 days.
  Phase C  stratified sampling with saturation stop: L6 const-free, L7 const-bearing,
           L7 const-free (budget-capped).

Phases B/C reuse the ENGINE'S OWN primitives (the same ones find_rules calls:
_core.mine_one_length / _confirm_mined_rules / deduplicate_rules), and reproduce
find_rules' internal seed derivation exactly, so the whole campaign behaves like one
find_rules run with a stratified length-6/7 universe policy.

Checkpointed: rules snapshot after every length barrier; per-chunk found-pairs appended
to JSONL; state.json records completed units -> safe to relaunch after interruption.
Campaign provenance in campaign_provenance.json.

Usage: python -u mine_74_production.py --out <dir> [--smoke]
Env: CALIB_CONFIG, CALIB_SIMPLIPY_SRC, CALIB_SIMPLIPY_SHA, RAYON_NUM_THREADS,
     BUDGET_L6CF_H / BUDGET_L7CB_H / BUDGET_L7CF_H (hours, defaults 36/24/30)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time

import numpy as np

SIMPLIPY_SRC = os.environ.get('CALIB_SIMPLIPY_SRC', '/home/psaegert/Projects/simplipy/src')
if SIMPLIPY_SRC and os.path.isdir(SIMPLIPY_SRC):
    sys.path.insert(0, SIMPLIPY_SRC)

from simplipy import SimpliPyEngine  # noqa: E402
from simplipy.utils import (count_expressions, deduplicate_rules, enumerate_expressions,  # noqa: E402
                            sample_expression)

CONFIG = os.environ.get(
    'CALIB_CONFIG', '/home/psaegert/Projects/simplipy/simplipy-assets/engines/dev_7-3/config.yaml')
DUMMY = ['x0', 'x1', 'x2', 'x3']
EXTRA = ['<constant>', '0', '1', '(-1)', 'np.e', 'np.pi',
         'float("inf")', 'float("-inf")', 'float("nan")']
RTOL, ATOL = 1e-9, 1e-12
CH, RT = 16, 16
SEED = 42
X_ROWS = 1024
MAX_TARGET = 4
CHUNK = 500_000            # sources per mine_one_length call (memory bound)
SAMPLE_BATCH = 250_000     # draws per saturation batch
SATURATION_MIN_NEW = 5     # stop a stratum after 2 consecutive batches below this many new rules
CF_OFFSET = 1 << 39        # seed sub-block for sampled strata (disjoint from exhaustive idx range)


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class Campaign:
    def __init__(self, out: str, smoke: bool) -> None:
        self.out = out
        self.smoke = smoke
        os.makedirs(out, exist_ok=True)
        self.rules_file = os.path.join(out, 'rules_74.json')
        self.state_file = os.path.join(out, 'state.json')
        self.prov_file = os.path.join(out, 'campaign_provenance.json')
        self.state = (json.load(open(self.state_file))
                      if os.path.exists(self.state_file) else {'done': []})
        self.prov = (json.load(open(self.prov_file))
                     if os.path.exists(self.prov_file) else {
                         'campaign': '7-4 stratified max-feasible',
                         'host': platform.node(),
                         'started': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                         'simplipy_sha': os.environ.get('CALIB_SIMPLIPY_SHA', ''),
                         'seed': SEED, 'phases': {}})
        # smoke: shrink everything (phase A <=3; B = L4-cb exhaustive; C samples L5 strata)
        # Phase A = find_rules through max_complete; the LAST complete length (a5_len) is
        # mined as a CHUNKED driver stratum instead (user directive 2026-07-12): find_rules
        # runs a whole length as ONE multi-day Rust call with no checkpoint and no progress
        # output -- a crash would lose all of it. Chunking is bit-identical to the monolithic
        # call (verified for phase B: per-source seed = length_seed + global offset).
        self.max_complete = 2 if smoke else 4
        self.a5_len = 3 if smoke else 5            # chunked complete length (both classes)
        self.exh_len = 4 if smoke else 6           # exhaustive const-bearing at this length
        self.samp_lens = [4, 5] if smoke else [6, 7]
        self.budgets_h = ({'L4_cf': 0.02, 'L5_cb': 0.02, 'L5_cf': 0.02} if smoke else {
            'L6_cf': float(os.environ.get('BUDGET_L6CF_H', '36')),
            'L7_cb': float(os.environ.get('BUDGET_L7CB_H', '24')),
            'L7_cf': float(os.environ.get('BUDGET_L7CF_H', '30'))})
        self.chunk = 20_000 if smoke else CHUNK
        self.sample_batch = 300 if smoke else SAMPLE_BATCH

        self.eng = SimpliPyEngine.from_config(CONFIG)
        assert self.eng._core is not None
        # Reproduce find_rules' internal derivations for the SAME seed (engine.py:2688-2702):
        # master rng -> X (X_ROWS) -> X_confirm (2x) -> mine_seed -> confirm_seed.
        rng = np.random.default_rng(SEED)
        self.X = self.eng._mining_sample_x(X_ROWS, len(DUMMY), rng)
        self.X_confirm = self.eng._mining_sample_x(2 * X_ROWS, len(DUMMY), rng)
        self.mine_seed = int(rng.integers(2 ** 62))
        self.confirm_seed = int(rng.integers(2 ** 62))
        self.min_informative = X_ROWS // 8
        self.non_leaf = dict(sorted(self.eng.operator_arity.items(), key=lambda x: x[1]))
        self.leaves = DUMMY + EXTRA
        self.leaves_cf = [t for t in self.leaves if t != '<constant>']
        self.counts13 = count_expressions(len(self.leaves), self.non_leaf, max(self.samp_lens))
        self.counts12 = count_expressions(len(self.leaves_cf), self.non_leaf, max(self.samp_lens))
        self.library = None

    # ---------------- shared helpers ----------------
    def save_state(self) -> None:
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
        self.prov['updated'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        with open(self.prov_file, 'w') as f:
            json.dump(self.prov, f, indent=2)

    def rules(self) -> list:
        return [(tuple(lhs), tuple(rhs)) for lhs, rhs in self.eng.simplification_rules]

    def set_rules(self, rules: list) -> None:
        self.eng.simplification_rules = list(rules)
        self.eng.compile_rules()
        self.eng._core.set_rules([(list(lhs), list(rhs)) for lhs, rhs in rules])

    def save_rules(self, tag: str) -> None:
        with open(self.rules_file, 'w') as f:
            json.dump(self.eng.simplification_rules, f, indent=4)
        by_len: dict[str, int] = {}
        for lhs, _ in self.eng.simplification_rules:
            by_len[str(len(lhs))] = by_len.get(str(len(lhs)), 0) + 1
        self.prov['phases'][tag] = self.prov['phases'].get(tag, {})
        self.prov['phases'][tag]['rules_total_after'] = len(self.eng.simplification_rules)
        self.prov['phases'][tag]['rules_by_lhs_length_after'] = by_len
        self.save_state()

    def build_library(self) -> None:
        if self.library is not None:
            return
        exprs = enumerate_expressions(self.leaves, self.non_leaf, MAX_TARGET)
        cands = [list(t) for length in sorted(exprs) for t in sorted(exprs[length])]
        self.library = self.eng._core.build_candidate_library(
            cands, DUMMY, self.X.flatten(order='C').tolist(), X_ROWS, fold_filter=True)
        log(f'library: {self.library.n_candidates:,} kept, {self.library.n_filtered:,} filtered')

    def mine_chunk(self, sources: list, length_seed: int, offset: int) -> list:
        """mine_one_length on a source chunk with the campaign's per-length seed layout."""
        return self.eng._core.mine_one_length(
            sources, self.library, MAX_TARGET, CH, RT,
            length_seed + offset, RTOL, ATOL, self.min_informative)

    def confirm(self, found: list, length: int) -> list:
        if not found:
            return []
        n0 = len(found)
        kept = self.eng._confirm_mined_rules(
            found, DUMMY, self.X_confirm, CH, RT, RTOL, ATOL,
            max(1, round(self.min_informative * self.X_confirm.shape[0] / X_ROWS)),
            self.confirm_seed + (length << 40))
        if len(kept) < n0:
            log(f'  confirm rejected {n0 - len(kept):,} of {n0:,}')
        return kept

    def integrate(self, found: list, tag: str) -> int:
        """Dedup `found` into the engine ruleset; returns the net new-rule count."""
        before = len(self.eng.simplification_rules)
        merged = deduplicate_rules(
            self.rules() + [(tuple(lhs), tuple(rhs)) for lhs, rhs in found], DUMMY, verbose=False)
        self.set_rules(merged)
        self.save_rules(tag)
        return len(merged) - before

    # ---------------- phases ----------------
    def phase_a(self) -> None:
        if 'A' in self.state['done']:
            log('phase A already done (resume)')
            self.set_rules([(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(open(self.rules_file))])
            return
        log(f'PHASE A: complete lengths 2-{self.max_complete} via find_rules')
        t0 = time.time()
        self.eng.find_rules(
            max_source_pattern_length=self.max_complete,
            max_target_pattern_length=MAX_TARGET,
            dummy_variables=len(DUMMY), extra_internal_terms=EXTRA,
            X=X_ROWS, constants_fit_challenges=CH, constants_fit_retries=RT,
            output_file=self.rules_file, seed=SEED, verbose=True,
            rtol=RTOL, atol=ATOL, candidate_fold_filter=True)
        # COMPLETENESS GUARD (driver verification, CRITICAL finding): find_rules swallows
        # SIGINT by design (graceful truncation, NORMAL return), so a mid-phase interrupt
        # would otherwise mark 'A' done with a truncated ruleset. The provenance sidecar's
        # last completed source length is the ground truth for whether the loop finished.
        side = json.load(open(self.rules_file + '.provenance.json'))
        last_done = side.get('progress', {}).get('last_completed_source_length')
        if last_done != self.max_complete:
            raise RuntimeError(
                f'phase A INCOMPLETE (last completed length {last_done} != {self.max_complete}; '
                f'interrupted?): refusing to mark done -- relaunch to redo phase A')
        self.prov['phases']['A'] = {'wall_s': round(time.time() - t0, 1),
                                    'provenance_sidecar': self.rules_file + '.provenance.json'}
        self.state['done'].append('A')
        self.save_rules('A')
        log(f'PHASE A DONE: {len(self.eng.simplification_rules):,} rules')

    def phase_a5(self) -> None:
        """Chunked COMPLETE mine of a5_len (both classes) with per-chunk checkpoints --
        replaces find_rules' single opaque multi-day call for the big complete length.
        Same barrier semantics: rules FIXED throughout, confirm + integrate at the end."""
        L = self.a5_len
        tag = f'A5_L{L}'
        if tag in self.state['done']:
            log(f'phase A5 ({tag}) already done (resume)')
            return
        log(f'PHASE A5: chunked complete L{L} ({self.counts13[L]:,} sources)')
        exprs = enumerate_expressions(self.leaves, self.non_leaf, L)
        universe = sorted(exprs[L])
        if self.smoke:
            universe = universe[:5_000]
        found_file = os.path.join(self.out, f'found_{tag}.jsonl')
        done_chunks = set(self.state.get(f'{tag}_chunks', []))
        length_seed = self.mine_seed + (L << 40)
        t0 = time.time()
        n_chunks = (len(universe) + self.chunk - 1) // self.chunk
        for ci in range(n_chunks):
            if ci in done_chunks:
                continue
            batch = [list(t) for t in universe[ci * self.chunk:(ci + 1) * self.chunk]]
            found = self.mine_chunk(batch, length_seed, ci * self.chunk)
            with open(found_file, 'a') as f:
                for pair in found:
                    f.write(json.dumps(pair) + '\n')
            self.state.setdefault(f'{tag}_chunks', []).append(ci)
            self.save_state()
            el = time.time() - t0
            eta_h = el / max(1, ci + 1 - len(done_chunks)) * (n_chunks - ci - 1) / 3600
            log(f'  chunk {ci + 1}/{n_chunks}: +{len(found)} pairs '
                f'({el / 3600:.1f}h elapsed, ~{eta_h:.1f}h left)')
        found_all = ([json.loads(line) for line in open(found_file)]
                     if os.path.exists(found_file) else [])
        found_all = [(tuple(l), tuple(r)) for l, r in found_all]
        kept = self.confirm(found_all, L)
        new = self.integrate(kept, tag)
        self.prov['phases'][tag] = {**self.prov['phases'].get(tag, {}),
                                    'sources': len(universe), 'pairs_mined': len(found_all),
                                    'pairs_confirmed': len(kept), 'net_new_rules': new,
                                    'wall_s': round(time.time() - t0, 1)}
        self.state['done'].append(tag)
        self.save_state()
        log(f'PHASE A5 DONE: +{new:,} rules from complete L{L}')

    def split_classes(self, upto: int) -> tuple[dict, dict]:
        exprs = enumerate_expressions(self.leaves, self.non_leaf, upto)
        cf, cb = {}, {}
        for length, s in exprs.items():
            cf[length] = sorted(t for t in s if '<constant>' not in t)
            cb[length] = sorted(t for t in s if '<constant>' in t)
            assert len(cf[length]) == self.counts12[length], f'cf split mismatch at L{length}'
            assert len(cf[length]) + len(cb[length]) == self.counts13[length]
        return cf, cb

    def gen_cb_slices(self, length: int, cf: dict, cb: dict):
        """Yield the EXACT const-bearing universe of `length` as (slice_name, iterator)."""
        for op, arity in self.non_leaf.items():
            if arity == 1:
                yield f'{op}(cb{length - 1})', ((op,) + t for t in cb[length - 1])
            else:  # arity 2: cb pairs = (cf x cb) + (cb x cf) + (cb x cb) per composition
                for a in range(1, length - 1):
                    b = length - 1 - a
                    if b < 1:
                        continue
                    yield (f'{op}({a},{b})',
                           itertools.chain(
                               ((op,) + l + r for l in cf[a] for r in cb[b]),
                               ((op,) + l + r for l in cb[a] for r in cf[b]),
                               ((op,) + l + r for l in cb[a] for r in cb[b])))

    def phase_b(self) -> None:
        L = self.exh_len
        tag = f'B_L{L}cb'
        if tag in self.state['done']:
            # Short-circuit (driver verification): never regenerate the universe -- and never
            # re-mine chunks -- once the length barrier has integrated (re-mining after the
            # barrier would run under a LATER rule state and violate barrier semantics).
            log(f'phase B ({tag}) already done (resume)')
            return
        expected = self.counts13[L] - self.counts12[L]
        log(f'PHASE B: exhaustive L{L} const-bearing ({expected:,} sources, chunks of {self.chunk:,})')
        cf, cb = self.split_classes(L - 1)
        found_file = os.path.join(self.out, f'found_{tag}.jsonl')
        done_chunks = set(self.state.get(f'{tag}_chunks', []))
        length_seed = self.mine_seed + (L << 40)
        n_generated, chunk_idx, offset = 0, 0, 0
        t0 = time.time()
        buf: list = []

        def flush(buf: list, chunk_idx: int, offset: int) -> None:
            if chunk_idx not in done_chunks:
                found = self.mine_chunk([list(t) for t in buf], length_seed, offset)
                with open(found_file, 'a') as f:
                    for pair in found:
                        f.write(json.dumps(pair) + '\n')
                self.state.setdefault(f'{tag}_chunks', []).append(chunk_idx)
                self.save_state()
                el = time.time() - t0
                log(f'  chunk {chunk_idx}: {len(buf):,} sources, +{len(found)} pairs '
                    f'({n_generated:,}/{expected:,} generated, {el / 3600:.1f}h elapsed)')

        for name, it in self.gen_cb_slices(L, cf, cb):
            for t in it:
                buf.append(t)
                n_generated += 1
                if len(buf) >= self.chunk:
                    flush(buf, chunk_idx, offset)
                    offset += len(buf)
                    chunk_idx += 1
                    buf = []
        if buf:
            flush(buf, chunk_idx, offset)
            n_generated_final = n_generated
        assert n_generated == expected, f'cb generator produced {n_generated:,} != {expected:,}'
        # length barrier: confirm + integrate ALL of L6-cb at once (rules were FIXED throughout)
        if tag not in self.state['done']:
            found_all = [json.loads(line) for line in open(found_file)] if os.path.exists(found_file) else []
            found_all = [(tuple(l), tuple(r)) for l, r in found_all]
            kept = self.confirm(found_all, L)
            new = self.integrate(kept, tag)
            self.prov['phases'][tag] = {**self.prov['phases'].get(tag, {}),
                                        'sources': expected, 'pairs_mined': len(found_all),
                                        'pairs_confirmed': len(kept), 'net_new_rules': new,
                                        'wall_s': round(time.time() - t0, 1)}
            self.state['done'].append(tag)
            self.save_state()
            log(f'PHASE B DONE: +{new:,} rules from L{L}-cb')

    def draw_stratum(self, length: int, klass: str, n: int, seed: int, seen: set) -> list:
        rng = np.random.default_rng(seed)
        leaves = self.leaves_cf if klass == 'cf' else self.leaves
        counts = self.counts12 if klass == 'cf' else self.counts13
        out: list = []
        attempts = 0
        while len(out) < n and attempts < 30 * n:
            t = sample_expression(length, leaves, self.non_leaf, counts, rng)
            attempts += 1
            if klass == 'cb' and '<constant>' not in t:
                continue
            if t not in seen:
                seen.add(t)
                out.append(list(t))
        return out

    def phase_c(self) -> None:
        for length in self.samp_lens:
            strata = ([f'L{length}_cb', f'L{length}_cf'] if length == max(self.samp_lens)
                      else [f'L{length}_cf'])  # exhaustive length already covered its cb stratum
            for stratum in strata:
                if stratum not in self.budgets_h:
                    continue
                if stratum in self.state['done']:
                    log(f'{stratum} already done (resume)')
                    continue
                klass = stratum.split('_')[1]
                budget_s = self.budgets_h[stratum] * 3600
                log(f'PHASE C stratum {stratum}: batches of {self.sample_batch:,}, '
                    f'budget {self.budgets_h[stratum]}h, saturation <{SATURATION_MIN_NEW} new x2')
                seen: set = set()
                length_seed = self.mine_seed + (length << 40) + CF_OFFSET + (0 if klass == 'cf' else 1 << 38)

                def batch_seed(idx: int) -> int:
                    return SEED + 7000 + length * 100 + idx + (50 if klass == 'cb' else 0)

                # PER-BATCH CHECKPOINT (driver verification, MAJOR finding): without it, a
                # mid-stratum crash resumed by REPLAYING already-integrated batches, whose
                # rules are all in rules_74.json already -> deterministic +0 new twice ->
                # FALSE 'saturated' verdict, silently skipping the remaining budget. Resume
                # now replays completed batches' DRAWS only (rebuilding `seen` identically,
                # since the draw sequence is a pure function of seed + seen-evolution),
                # restores the saturation window and consumed budget, and mines onward.
                completed = sorted(self.state.get(f'{stratum}_batches', []))
                curve = [e for e in (self.prov['phases'].get(stratum, {}).get('saturation_curve', []) or [])
                         if e['batch'] in set(completed)]  # drop entries from a crash mid-batch
                for idx in completed:
                    self.draw_stratum(length, klass, self.sample_batch, batch_seed(idx), seen)
                consumed = curve[-1]['elapsed_s'] if curve else 0.0
                low_batches = 0
                for entry in reversed(curve):
                    if entry['new_rules'] < SATURATION_MIN_NEW:
                        low_batches += 1
                    else:
                        break
                batch_idx = (max(completed) + 1) if completed else 0
                if completed:
                    log(f'  resume: {len(completed)} batches replayed (draws only), '
                        f'{consumed / 3600:.2f}h consumed, saturation window {low_batches}')
                t0 = time.time() - consumed
                while time.time() - t0 < budget_s and low_batches < 2:
                    srcs = self.draw_stratum(length, klass, self.sample_batch,
                                             batch_seed(batch_idx), seen)
                    if not srcs:
                        break
                    found = self.mine_chunk(srcs, length_seed, batch_idx * self.sample_batch)
                    kept = self.confirm(found, length)
                    curve.append({'batch': batch_idx, 'drawn': len(srcs), 'new_rules': -1,
                                  'elapsed_s': round(time.time() - t0, 1)})
                    self.prov['phases'].setdefault(stratum, {})['saturation_curve'] = curve
                    new = self.integrate(kept, stratum)  # saves rules + provenance
                    curve[-1]['new_rules'] = new
                    self.state.setdefault(f'{stratum}_batches', []).append(batch_idx)
                    self.save_state()
                    log(f'  {stratum} batch {batch_idx}: +{new} new rules '
                        f'({len(seen):,} sampled, {(time.time() - t0) / 3600:.2f}h)')
                    low_batches = low_batches + 1 if new < SATURATION_MIN_NEW else 0
                    batch_idx += 1
                if low_batches >= 2:
                    log(f'  {stratum}: SATURATED')
                self.prov['phases'][stratum] = {**self.prov['phases'].get(stratum, {}),
                                                'sampled': len(seen),
                                                'universe': int((self.counts12 if klass == 'cf'
                                                                 else self.counts13)[length]
                                                                - (self.counts12[length] if klass == 'cb' else 0)),
                                                'saturation_curve': curve,
                                                'stopped': 'saturated' if low_batches >= 2 else 'budget'}
                self.state['done'].append(stratum)
                self.save_state()
        log('PHASE C DONE')

    def run(self) -> None:
        self.build_library()
        self.phase_a()
        self.build_library()
        self.phase_a5()
        self.phase_b()
        self.phase_c()
        self.prov['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        self.save_rules('final')
        log(f'CAMPAIGN COMPLETE: {len(self.eng.simplification_rules):,} rules -> {self.rules_file}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    Campaign(a.out, a.smoke).run()
