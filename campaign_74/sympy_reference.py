"""One-time SymPy reference leg for the 7-4 progress dashboard.

Converts the paper's fixed 4,096-skeleton corpus to SymPy (object-faithful: a DISTINCT
symbol per `<constant>` occurrence, as in CAPABILITY_GAP_ANALYSIS.md), times
`sympy.simplify` per skeleton (10s alarm cap, capped entries flagged), and records a
complexity metric COMPARABLE to prefix token count: tree nodes with n-ary Add/Mul
expanded to (n-1) binary nodes (prefix token count == tree node count in our grammar).

Output: sympy_reference.pkl {times, node_counts_before, node_counts_after, capped, failed}.
Run once; the corpus never changes. Pinned to CPU 30 at nice 10 (same co-load policy).
"""
from __future__ import annotations

import os
import pickle
import signal
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
try:
    os.sched_setaffinity(0, {30})
    os.nice(10)
except OSError:
    pass

import sympy as sp  # noqa: E402

SKELETONS = '/home/psaegert/Projects/flash-ansr/results/simplification/rustbench_skeletons_v23.0-3M_4096.pkl'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sympy_reference.pkl')

X = sp.symbols('x0 x1 x2 x3')
UNARY = {
    'neg': lambda a: -a, 'inv': lambda a: 1 / a, 'abs': sp.Abs,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
    'log': sp.log, 'exp': sp.exp,
    'pow2': lambda a: a ** 2, 'pow3': lambda a: a ** 3, 'pow4': lambda a: a ** 4,
    'pow5': lambda a: a ** 5,
    'pow1_2': lambda a: a ** sp.Rational(1, 2), 'pow1_3': lambda a: a ** sp.Rational(1, 3),
    'pow1_4': lambda a: a ** sp.Rational(1, 4), 'pow1_5': lambda a: a ** sp.Rational(1, 5),
    'mult2': lambda a: 2 * a, 'mult3': lambda a: 3 * a, 'mult4': lambda a: 4 * a,
    'mult5': lambda a: 5 * a,
    'div2': lambda a: a / 2, 'div3': lambda a: a / 3, 'div4': lambda a: a / 4,
    'div5': lambda a: a / 5,
}
BINARY = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
          '/': lambda a, b: a / b, 'pow': lambda a, b: a ** b}
LEAF = {'x0': X[0], 'x1': X[1], 'x2': X[2], 'x3': X[3],
        '0': sp.Integer(0), '1': sp.Integer(1), '(-1)': sp.Integer(-1),
        'np.e': sp.E, 'np.pi': sp.pi,
        'float("inf")': sp.oo, 'float("-inf")': -sp.oo, 'float("nan")': sp.nan}


def to_sympy(tokens: list[str]) -> sp.Expr:
    it = iter(tokens)
    counter = [0]

    def build():
        t = next(it)
        if t in UNARY:
            return UNARY[t](build())
        if t in BINARY:
            a = build()
            return BINARY[t](a, build())
        if t == '<constant>':
            counter[0] += 1
            return sp.Symbol(f'C{counter[0]}')
        if t in LEAF:
            return LEAF[t]
        if t[0] == 'x' and t[1:].isdigit():  # flash-ansr variables x0..x17
            return sp.Symbol(t)
        return sp.Float(float(t))  # numeric literal

    e = build()
    try:
        next(it)
        raise ValueError('trailing tokens')
    except StopIteration:
        return e


def nodes(e: sp.Expr) -> int:
    """Tree nodes with n-ary Add/Mul expanded binary == prefix token count semantics."""
    if e.is_Atom:
        # non-integer rationals would print as div(p, q): 3 tokens
        if e.is_Rational and not e.is_Integer:
            return 3
        return 1
    k = len(e.args)
    op_cost = (k - 1) if e.func in (sp.Add, sp.Mul) else 1
    return op_cost + sum(nodes(a) for a in e.args)


class Timeout(Exception):
    pass


def alarm(_s, _f):
    raise Timeout


def main() -> None:
    skeletons = pickle.load(open(SKELETONS, 'rb'))
    signal.signal(signal.SIGALRM, alarm)
    times, before, after, capped, failed = [], [], [], [], []
    t_start = time.time()
    for i, s in enumerate(skeletons):
        try:
            e = to_sympy(list(s))
            nb = nodes(e)
        except Exception:
            times.append(float('nan')); before.append(-1); after.append(-1)
            capped.append(False); failed.append(True)
            continue
        signal.alarm(10)
        t0 = time.perf_counter()
        try:
            r = sp.simplify(e)
            dt = time.perf_counter() - t0
            signal.alarm(0)
            times.append(dt); before.append(nb); after.append(nodes(r))
            capped.append(False); failed.append(False)
        except Timeout:
            times.append(10.0); before.append(nb); after.append(nb)
            capped.append(True); failed.append(False)
        except Exception:
            signal.alarm(0)
            times.append(float('nan')); before.append(nb); after.append(-1)
            capped.append(False); failed.append(True)
        if (i + 1) % 256 == 0:
            print(f'{i + 1}/{len(skeletons)} ({time.time() - t_start:.0f}s)', flush=True)
    pickle.dump({'times': times, 'nodes_before': before, 'nodes_after': after,
                 'capped': capped, 'failed': failed}, open(OUT, 'wb'))
    import numpy as np
    ok = [i for i in range(len(times)) if not failed[i]]
    print(f'DONE: {len(ok)}/{len(skeletons)} ok, {sum(capped)} capped; '
          f'median time {np.median([times[i] for i in ok]) * 1e3:.1f} ms; '
          f'mean nodes {np.mean([before[i] for i in ok]):.2f} -> '
          f'{np.mean([after[i] for i in ok]):.2f}', flush=True)


if __name__ == '__main__':
    main()
