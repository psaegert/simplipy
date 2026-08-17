"""SimpliPy vs SymPy ECDFs on the 64k nv corpus, scored in the unified measure (mu).

Successor of the 0.11-era 2x3 figure (assets/images/simplipy_vs_sympy.svg, commit
7cd35ad), re-run against the 0.12 AC engine and the acj artifact family, with ONE
deliberate scoring change: the simplification-ratio row is no longer raw prefix-token
length. Both systems' outputs are parsed into SimpliPy token space and priced by the
engine's internal description-length measure (``SimpliPyEngine.complexity``, the mu of
docs/formal.md section 5) against the same denominator mu(original) -- SymPy's output is
converted back through a sympy->prefix bridge (below), so both systems are judged by the
same MDL yardstick instead of each by its own token count.

Protocol (matches the 0.11 figure where it still applies):
  * corpus: remine/corpus_nv_64k.pkl -- 65,536 Lample-Charton expressions from the
    Flash-ANSR v23.0 prior, desugared to the current 23-operator language;
  * SimpliPy timed per call, single-threaded, gc off, median of REPS runs;
  * SymPy: each ``<constant>`` occurrence becomes a DISTINCT real Symbol (mirrors the
    engine's Const-independence doctrine), variables are real Symbols; ``simplify()``
    inside a per-expression SIGALRM guard of 1 s, workers 24-wide; expressions SymPy
    does not finish are scored as what they are -- infinite time, ratio 1 -- and stay
    in the denominator (the ``ratio=None`` variant drops them instead; both are
    plotted, as in the original figure);
  * a SymPy output that has no spelling in the engine's real extended-value language
    (Piecewise, sign, complex/zoo, ...) counts as a conversion failure: ratio 1, and
    the count is reported (it is SymPy leaving the comparable language, not a win or
    a loss the measure can price).

Semantic caveat, carried from the original: ``rootn(x, n)`` (real odd root) maps to
``x**Rational(1, n)`` for SymPy -- principal-branch semantics on x < 0 differ; the
mismatch is confined to the negative branch of odd roots.

Usage: ecdf_vs_sympy.py [--sympy-only | --simplipy-only | --plot-only] [--limit N]
Artifacts: benchmarks/ecdf_vs_sympy_results.pkl (untracked, regenerable),
           benchmarks/ecdf_vs_sympy_summary.json (tracked),
           benchmarks/out/simplipy_vs_sympy_012_mu.{png,svg}.
"""
import gc
import json
import os
import pickle
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from simplipy import Mode, SimpliPyEngine  # noqa: E402

CORPUS_PKL = os.path.join(REPO, "remine", "corpus_nv_64k.pkl")
RESULTS_PKL = os.path.join(HERE, "ecdf_vs_sympy_results.pkl")
SUMMARY_JSON = os.path.join(HERE, "ecdf_vs_sympy_summary.json")
CELLS = {c: os.path.join(REPO, "remine", f"acj-{c}", "config.yaml")
         for c in ("2-1", "3-2", "4-3")}
BUDGETS = (1, 2, 4, 8, 16, 48)
REPS = 3
SYMPY_TIMEOUT_S = 1.0
SYMPY_WORKERS = 24

ARITY = {'+': 2, '-': 2, 'neg': 1, '*': 2, '/': 2, 'abs': 1, 'inv': 1, 'pow': 2,
         'rootn': 2, 'sin': 1, 'cos': 1, 'tan': 1, 'asin': 1, 'acos': 1, 'atan': 1,
         'sinh': 1, 'cosh': 1, 'tanh': 1, 'asinh': 1, 'acosh': 1, 'atanh': 1,
         'exp': 1, 'log': 1}


# ----------------------------------------------------------------------------- sympy leg
def _to_sympy(tokens):
    """Prefix tokens -> sympy expression. Each <constant> = a fresh real Symbol."""
    import sympy as sp
    it = iter(tokens)
    counter = [0]

    def walk():
        t = next(it)
        if t in ARITY:
            args = [walk() for _ in range(ARITY[t])]
            return {
                '+': lambda a, b: a + b, '-': lambda a, b: a - b,
                '*': lambda a, b: a * b, '/': lambda a, b: a / b,
                'neg': lambda a: -a, 'inv': lambda a: 1 / a, 'abs': sp.Abs,
                'pow': lambda a, b: a ** b,
                'rootn': lambda a, b: a ** (sp.Rational(1, int(b)) if b.is_Integer
                                            else 1 / b),
                'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
                'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
                'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
                'exp': sp.exp, 'log': sp.log,
            }[t](*args)
        if t == '<constant>':
            counter[0] += 1
            return sp.Symbol(f'C{counter[0]}', real=True)
        if t.startswith('x') and t[1:].isdigit():
            return sp.Symbol(t, real=True)
        if t == 'np.pi':
            return sp.pi
        if t == 'np.e':
            return sp.E
        return sp.Rational(t) if '/' in t or '.' not in t else sp.Float(t)

    e = walk()
    try:
        next(it)
        raise ValueError('trailing tokens')
    except StopIteration:
        return e


def _from_sympy(e):
    """sympy expression -> prefix tokens in the engine's language. None = no spelling."""
    import sympy as sp
    if e is sp.zoo or e.has(sp.zoo, sp.I, sp.Piecewise, sp.sign, sp.Heaviside,
                            sp.im, sp.re, sp.arg, sp.conjugate, sp.Min, sp.Max,
                            sp.ceiling, sp.floor, sp.gamma, sp.polygamma,
                            sp.LambertW, sp.erf, sp.Sum, sp.Integral, sp.Derivative):
        return None
    FUN = {sp.Abs: 'abs', sp.sin: 'sin', sp.cos: 'cos', sp.tan: 'tan',
           sp.asin: 'asin', sp.acos: 'acos', sp.atan: 'atan',
           sp.sinh: 'sinh', sp.cosh: 'cosh', sp.tanh: 'tanh',
           sp.asinh: 'asinh', sp.acosh: 'acosh', sp.atanh: 'atanh',
           sp.exp: 'exp', sp.log: 'log'}

    def num(r):
        """Rational/Integer -> tokens (neg-wrapped when negative; p/q one token)."""
        if r < 0:
            return ['neg'] + num(-r)
        if r.is_Integer:
            return [str(int(r))]
        return [f'{r.p}/{r.q}']

    def walk(t):
        if t is sp.pi:
            return ['np.pi']
        if t is sp.E:
            return ['np.e']
        if t is sp.oo:
            return ['float("inf")']
        if t is -sp.oo:
            return ['neg', 'float("inf")']
        if t is sp.nan:
            return ['float("nan")']
        if t.is_Integer or t.is_Rational:
            return num(t)
        if t.is_Float:
            v = float(t)
            return ['neg', repr(-v)] if v < 0 else [repr(v)]
        if t.is_Symbol:
            return ['<constant>'] if t.name.startswith('C') else [t.name]
        if isinstance(t, sp.Add):
            out = []
            for _ in range(len(t.args) - 1):
                out.append('+')
            for a in t.args:
                out.extend(walk(a))
            return out
        if isinstance(t, sp.Mul):
            out = []
            for _ in range(len(t.args) - 1):
                out.append('*')
            for a in t.args:
                out.extend(walk(a))
            return out
        if isinstance(t, sp.Pow):
            b, ex = t.args
            if ex is sp.S.NegativeOne:
                return ['inv'] + walk(b)
            return ['pow'] + walk(b) + walk(ex)
        if type(t) in FUN:
            return [FUN[type(t)]] + walk(t.args[0])
        raise KeyError(type(t).__name__)

    try:
        return walk(e)
    except (KeyError, RecursionError):
        return None


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _sympy_worker(tokens):
    """Returns (seconds_or_None, out_tokens_or_None, status)."""
    import sympy as sp  # noqa: F401
    signal.signal(signal.SIGALRM, _alarm)
    try:
        e = _to_sympy(tokens)
    except Exception:
        return (None, None, 'build_fail')
    signal.setitimer(signal.ITIMER_REAL, SYMPY_TIMEOUT_S)
    try:
        t0 = time.perf_counter()
        s = sp.simplify(e)
        dt = time.perf_counter() - t0
    except _Timeout:
        return (None, None, 'timeout')
    except Exception:
        return (None, None, 'simplify_fail')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out = _from_sympy(s)
    if out is None:
        return (dt, None, 'convert_fail')
    return (dt, out, 'ok')


def run_sympy(corpus):
    with Pool(SYMPY_WORKERS) as pool:
        results = []
        t_start = time.time()
        for i, r in enumerate(pool.imap(_sympy_worker, corpus, chunksize=16)):
            results.append(r)
            if (i + 1) % 4096 == 0:
                el = time.time() - t_start
                print(f'  sympy {i + 1}/{len(corpus)}  ({el:.0f}s elapsed)', flush=True)
    return results


# -------------------------------------------------------------------------- simplipy leg
def run_simplipy(corpus):
    """Per-variant: times (median of REPS) + outputs. Single-threaded, gc off."""
    out = {}
    for label, cfg, kwargs in (
            [(f'acj-{c}', CELLS[c], {}) for c in CELLS]
            + [('acj-4-3-lossy', CELLS['4-3'], {'mode': Mode.LOSSY})]
            + [(f'acj-4-3-b{b}', CELLS['4-3'], {'node_budget': b})
               for b in BUDGETS if b != 48]):
        e = SimpliPyEngine.from_config(cfg)
        for row in corpus[:20]:
            e.simplify(row, **kwargs)
        times = []
        outputs = []
        gc.disable()
        for row in corpus:
            ts = []
            res = None
            for _ in range(REPS):
                t0 = time.perf_counter_ns()
                res = e.simplify(row, **kwargs)
                ts.append(time.perf_counter_ns() - t0)
            times.append(float(np.median(ts)) / 1e9)
            outputs.append(list(res))
        gc.enable()
        out[label] = {'seconds': times, 'outputs': outputs}
        print(f'  {label}: p50={np.percentile(times, 50)*1e6:.0f}us '
              f'mean={np.mean(times)*1e6:.0f}us', flush=True)
    return out


# ------------------------------------------------------------------------------- scoring
def score_mu(corpus, simplipy_res, sympy_res):
    e = SimpliPyEngine.from_config(CELLS['4-3'])
    mu_orig = np.array([e.complexity(r) for r in corpus], dtype=float)
    scored = {'mu_orig': mu_orig.tolist()}
    for label, d in simplipy_res.items():
        mu = np.array([e.complexity(r) for r in d['outputs']], dtype=float)
        scored[label] = {'seconds': d['seconds'], 'mu': mu.tolist()}
    sy_sec, sy_mu, statuses = [], [], []
    for (dt, toks, status), mo in zip(sympy_res, mu_orig):
        statuses.append(status)
        sy_sec.append(dt if status in ('ok', 'convert_fail') else None)
        if status == 'ok':
            try:
                sy_mu.append(float(e.complexity(toks)))
            except Exception:
                statuses[-1] = 'convert_fail'
                sy_mu.append(None)
        else:
            sy_mu.append(None)
    scored['sympy'] = {'seconds': sy_sec, 'mu': sy_mu, 'status': statuses}
    return scored


# ------------------------------------------------------------------------------ plotting
def make_figure(scored):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mu_orig = np.array(scored['mu_orig'])
    n = len(mu_orig)

    def ecdf(v):
        v = np.sort(np.asarray(v))
        return v, np.arange(1, len(v) + 1) / n  # denominator = FULL corpus

    sy = scored['sympy']
    sy_sec = np.array([s if s is not None else np.inf for s in sy['seconds']],
                      dtype=float)
    sy_status = np.array(sy['status'])
    sy_mu = np.array([m if m is not None else np.nan for m in sy['mu']], dtype=float)
    # ratio=1 convention: unresolved rows (timeout / no spelling) count as unsimplified
    sy_ratio_1 = np.where(np.isnan(sy_mu), 1.0, sy_mu / mu_orig)
    # ratio=None convention: unresolved rows dropped (curve saturates below 1)
    sy_ratio_none = (sy_mu / mu_orig)[~np.isnan(sy_mu)]

    greens = plt.cm.Greens
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    cols = {
        'Mined Rulesets': [(f'acj-{c}', greens(0.45 + 0.2 * i))
                           for i, c in enumerate(('2-1', '3-2', '4-3'))],
        'Safe vs Aggressive': [('acj-4-3', greens(0.85)),
                               ('acj-4-3-lossy', 'tab:purple')],
        'Search Budget': [(f'acj-4-3-b{b}', greens(0.4 + 0.45 * i / (len(BUDGETS) - 1)))
                          for i, b in enumerate(BUDGETS) if b != 48]
                         + [('acj-4-3', greens(0.85))],
    }
    for j, (title, variants) in enumerate(cols.items()):
        ax_t, ax_r = axes[0, j], axes[1, j]
        for label, color in variants:
            d = scored[label]
            x, y = ecdf(d['seconds'])
            ax_t.plot(x, y, color=color, label=label, lw=1.6)
            x, y = ecdf(np.array(d['mu']) / mu_orig)
            ax_r.plot(x, y, color=color, label=label, lw=1.6)
        x, y = ecdf(sy_sec[np.isfinite(sy_sec)])
        ax_t.plot(x, y, color='tab:orange', label='sympy', lw=1.6)
        x, y = ecdf(sy_ratio_none)
        ax_r.plot(x, y, color='tab:orange', label='sympy (ratio=None)', lw=1.6)
        x, y = ecdf(sy_ratio_1)
        ax_r.plot(x, y, color='tab:red', label='sympy (ratio=1)', lw=1.6)
        ax_t.set_xscale('log')
        ax_t.set_title(title)
        ax_t.set_xlabel('simplification time [s]')
        ax_r.set_xlabel(r'simplification ratio  $\mu(\mathrm{simp})/\mu(\mathrm{orig})$')
        ax_t.set_ylim(0, 1.02)
        ax_r.set_ylim(0, 1.02)
        ax_r.set_xlim(0, 1.6)
        ax_t.grid(alpha=0.25)
        ax_r.grid(alpha=0.25)
        ax_t.legend(fontsize=8, loc='lower right')
        ax_r.legend(fontsize=8, loc='lower right')
    axes[0, 0].set_ylabel('ECDF')
    axes[1, 0].set_ylabel('ECDF')
    fig.suptitle('SimpliPy 0.12 (acj family, unified measure) vs SymPy — 64k nv corpus; '
                 'ratios scored by the internal description-length measure mu for BOTH systems',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.join(REPO, 'benchmarks', 'out'), exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(REPO, 'benchmarks', 'out',
                                 f'simplipy_vs_sympy_012_mu.{ext}'), dpi=160)
    print('figure -> benchmarks/out/simplipy_vs_sympy_012_mu.{png,svg}')


def summarize(scored):
    mu_orig = np.array(scored['mu_orig'])
    s = {}
    for label in [k for k in scored if k not in ('mu_orig', 'sympy')]:
        d = scored[label]
        r = np.array(d['mu']) / mu_orig
        t = np.array(d['seconds'])
        s[label] = {'t_p50_us': float(np.percentile(t, 50) * 1e6),
                    't_mean_us': float(np.mean(t) * 1e6),
                    'ratio_p50': float(np.percentile(r, 50)),
                    'ratio_mean': float(np.mean(r)),
                    'frac_below_1': float(np.mean(r < 1.0)),
                    'frac_above_1': float(np.mean(r > 1.0))}
    sy = scored['sympy']
    st = np.array(sy['status'])
    mu = np.array([m if m is not None else np.nan for m in sy['mu']], dtype=float)
    r1 = np.where(np.isnan(mu), 1.0, mu / mu_orig)
    sec = np.array([x if x is not None else np.nan for x in sy['seconds']], dtype=float)
    s['sympy'] = {'status_counts': {k: int((st == k).sum()) for k in set(st)},
                  't_p50_us_completed': float(np.nanpercentile(sec, 50) * 1e6),
                  'ratio1_p50': float(np.percentile(r1, 50)),
                  'ratio1_mean': float(np.mean(r1)),
                  'frac_below_1_ratio1': float(np.mean(r1 < 1.0))}
    return s


def main():
    args = sys.argv[1:]
    limit = None
    if '--limit' in args:
        limit = int(args[args.index('--limit') + 1])
    corpus = pickle.load(open(CORPUS_PKL, 'rb'))[:limit]
    print(f'corpus: {len(corpus)} rows')
    prev = pickle.load(open(RESULTS_PKL, 'rb')) if os.path.exists(RESULTS_PKL) else {}
    if '--plot-only' not in args:
        if '--sympy-only' not in args:
            prev['simplipy'] = run_simplipy(corpus)
            pickle.dump(prev, open(RESULTS_PKL, 'wb'))
        if '--simplipy-only' not in args:
            prev['sympy'] = run_sympy(corpus)
            pickle.dump(prev, open(RESULTS_PKL, 'wb'))
    scored = score_mu(corpus, prev['simplipy'], prev['sympy'])
    summary = summarize(scored)
    json.dump(summary, open(SUMMARY_JSON, 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    make_figure(scored)


if __name__ == '__main__':
    main()
