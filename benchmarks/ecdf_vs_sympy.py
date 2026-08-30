"""The published SimpliPy-vs-SymPy benchmark, end to end.

Protocol (pre-registered; docs/guides/simplify.md states it alongside the
results): every arm runs serially on one pinned core of an otherwise idle
machine (the published numbers: an AMD Ryzen 9 9950X, BLAS thread caps at 1),
paired per row against SymPy 1.14.0's ``simplify()`` under a 1 s cap, on three
declared corpora — the v25 SR training prior (seed 20260830, n = 65,536), the
same prior under the engine mask policy 'all' (n = 65,536), and an external
neutral problem set (SOOSE fc/nc/wc, n = 600; every row compiles in the engine
language). The engine is the pinned acj-5-4-llm artifact: ``f64`` is the
shipped default, ``real`` is ``Mode.real``, ``permissive`` is
``Mode.permissive``, every arm at the default ``effort=4``; the unmasked leg
adds the explore-budget sweep arms ``effort=0`` and ``effort=64``.

Scoring runs in the deployment space: ratio = complexity(output) /
complexity(input), priced by the engine's shipped ``complexity()`` instrument
in the default (f64) canonicalization; lower is better. SymPy is censored — a
1 s timeout, or an output with no spelling in the engine's language
(Piecewise, sign, complex, ...) — and censored rows score ratio 1.0 in every
mean and table stat, the charitable choice; in the ECDF panels the censored
curves simply end below 1.

Timing: the SimpliPy arms run in-process, single-threaded, gc off, wall clock
per call, median of REPS = 3 calls per row. The SymPy leg runs in a supervised
process pool (pebble) whose workers cap their address space at 4 GB; each row
is timed inside its worker around ``simplify()`` alone, under an in-worker 1 s
SIGALRM cap with a hard pool-side timeout as backstop.

SymPy bridge: each ``<constant>`` occurrence becomes a distinct real Symbol,
variables are real Symbols. ``rootn(x, n)`` maps to ``x**Rational(1, n)``;
principal-branch semantics differ on x < 0, the mismatch confined to the
negative branch of odd roots.

Corpora resolve through the symbolic-data artifact registry
(``resolve("bench-nv25")``); the engine artifact installs on first use.

Outputs: ``benchmarks/ecdf_vs_sympy_summary.json`` and the five figure panels
under ``docs/assets/benchmarks/`` — ecdf_readme.png, ecdf_unmasked.png,
ecdf_masked_raw.png, ecdf_external.png, ecdf_effort_sweep.png. A results
pickle (untracked, regenerable) checkpoints each finished leg.

Usage: ecdf_vs_sympy.py [--limit N] [--out DIR]
  --limit N   slice every corpus to its first N rows (smoke runs)
  --out DIR   write summary, figures and checkpoint to DIR (default: the
              repo paths above)
"""
import argparse
import gc
import json
import os
import pickle
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ENGINE_ASSET = 'acj-5-4-llm'
CORPUS_ARTIFACT = 'bench-nv25'
CORPUS_FILES = {
    'unmasked': 'corpus_nv25unmasked_64k.pkl',
    'masked': 'corpus_nv25masked_64k.pkl',
    'external': 'corpus_external_soose.pkl',
}
CORPORA = ('unmasked', 'masked', 'external')
SWEEP_EFFORTS = (0, 64)  # extra f64 arms on the unmasked leg
REPS = 3
SYMPY_TIMEOUT_S = 1.0
SYMPY_WORKERS = 24
SYMPY_RLIMIT_AS = 4 << 30  # per-worker address-space cap, bytes
RESULTS_PKL_NAME = 'ecdf_vs_sympy_results.pkl'

ARITY = {'+': 2, '-': 2, 'neg': 1, '*': 2, '/': 2, 'abs': 1, 'inv': 1, 'pow': 2,
         'rootn': 2, 'sin': 1, 'cos': 1, 'tan': 1, 'asin': 1, 'acos': 1, 'atan': 1,
         'sinh': 1, 'cosh': 1, 'tanh': 1, 'asinh': 1, 'acosh': 1, 'atanh': 1,
         'exp': 1, 'log': 1}


# ----------------------------------------------------------------------- sympy leg
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
    import sympy as sp
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


def _pool_init():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (SYMPY_RLIMIT_AS, SYMPY_RLIMIT_AS))


def run_sympy(corpus, tag):
    """Supervised pool: RLIMIT_AS-capped workers, hard per-row timeout backstop."""
    try:
        from pebble import ProcessPool
    except ImportError:
        sys.exit('the SymPy leg runs under pebble: pip install pebble')
    from concurrent.futures import TimeoutError as FutTimeout
    out = []
    t0 = time.time()
    with ProcessPool(max_workers=SYMPY_WORKERS, initializer=_pool_init) as pool:
        futs = [pool.schedule(_sympy_worker, args=(row,),
                              timeout=SYMPY_TIMEOUT_S + 1.5) for row in corpus]
        for i, fut in enumerate(futs):
            try:
                out.append(fut.result())
            except FutTimeout:
                out.append((None, None, 'timeout'))
            except Exception:
                out.append((None, None, 'worker_lost'))
            if (i + 1) % 8192 == 0:
                print(f'  sympy[{tag}] {i + 1}/{len(corpus)} '
                      f'({time.time() - t0:.0f}s)', flush=True)
    print(f'  sympy[{tag}] done in {time.time() - t0:.0f}s', flush=True)
    return out


# -------------------------------------------------------------------- simplipy legs
def run_simplipy(cfg, corpus, label, **kwargs):
    """One arm: fresh engine, 20-row warmup, per-row median of REPS calls, gc off."""
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.from_config(cfg)
    for row in corpus[:20]:
        engine.simplify(row, **kwargs)
    times, outputs = [], []
    gc.disable()
    for row in corpus:
        ts, out = [], None
        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            out = engine.simplify(row, **kwargs)
            ts.append(time.perf_counter_ns() - t0)
        times.append(float(np.median(ts)) / 1e9)
        outputs.append(list(out))
    gc.enable()
    print(f'  {label}: p50={np.percentile(times, 50) * 1e6:.0f}us '
          f'mean={np.mean(times) * 1e6:.0f}us', flush=True)
    return {'seconds': times, 'outputs': outputs}


# ------------------------------------------------------------------------- scoring
def score(cfg, corpora, results):
    """Per-row ratio/time arrays + the tracked summary, priced by complexity()."""
    from simplipy import SimpliPyEngine
    pricer = SimpliPyEngine.from_config(cfg)
    arrays, summary = {}, {}
    for tag in CORPORA:
        corpus = corpora[tag]
        mu0 = np.array([pricer.complexity(r) for r in corpus], float)
        arrays[f'{tag}/mu0'] = mu0
        for label, d in results[tag]['modes'].items():
            mu = np.array([pricer.complexity(r) for r in d['outputs']], float)
            ratio = mu / mu0
            arrays[f'{tag}/{label}/t'] = np.array(d['seconds'], float)
            arrays[f'{tag}/{label}/ratio'] = ratio
            summary[f'{tag}/{label}'] = {
                't_p50_us': float(np.percentile(d['seconds'], 50) * 1e6),
                'ratio_mean': float(np.mean(ratio)),
                'wins': float(np.mean(ratio < 1)),
                'inflated': int(np.sum(ratio > 1))}
        sy_t, sy_ratio = [], []
        for (dt, toks, status), m0 in zip(results[tag]['sympy'], mu0):
            sy_t.append(dt if (status in ('ok', 'convert_fail') and dt is not None)
                        else np.nan)
            val = np.nan
            if status == 'ok':
                try:
                    val = float(pricer.complexity(toks)) / m0
                except Exception:
                    pass
            sy_ratio.append(val)
        arrays[f'{tag}/sympy/t'] = np.array(sy_t, float)
        arrays[f'{tag}/sympy/ratio'] = np.array(sy_ratio, float)
        r = np.array(sy_ratio, float)
        fin = r[np.isfinite(r)]
        summary[f'{tag}/sympy'] = {
            't_p50_us': float(np.nanpercentile(np.array(sy_t, float), 50) * 1e6),
            'censored': int(np.sum(~np.isfinite(r))),
            'wins': float(np.sum(fin < 1) / len(r)),
            'inflated': int(np.sum(fin > 1))}
    return arrays, summary


# --------------------------------------------------------------------- statistics
def _boot_mean_ci(a, n_resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    means = np.empty(n_resamples)
    chunk = 200
    for i in range(0, n_resamples, chunk):
        k = min(chunk, n_resamples - i)
        idx = rng.integers(0, n, size=(k, n))
        means[i:i + k] = a[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def compute_stats(arrays):
    """Everything the figures annotate: per-arm stats, bootstrap CIs, the sweep."""
    out = {}
    for c in CORPORA:
        out[c] = {}
        sy_t = arrays[f'{c}/sympy/t']
        sy_fin = np.isfinite(sy_t)
        for arm in ARMS:
            r = arrays[f'{c}/{arm}/ratio']
            t = arrays[f'{c}/{arm}/t']
            cens_r = int(np.isnan(r).sum())
            cens_t = int(np.isnan(t).sum())
            r_imp = np.where(np.isnan(r), 1.0, r)  # censored -> 1.0 for stats
            t_fin = t[np.isfinite(t)]
            d = {
                'n': len(r),
                'censored_ratio': cens_r, 'censored_t': cens_t,
                'ratio_mean': float(r_imp.mean()),
                'ratio_median': float(np.median(r_imp)),
                'wins': float((r_imp < 1).mean()),
                'same': float((r_imp == 1).mean()),
                'bigger': float((r_imp > 1).mean()),
                'ratio_max': float(r_imp.max()),
                't_p50_us': float(np.median(t_fin) * 1e6),
                't_p95_us': float(np.percentile(t_fin, 95) * 1e6),
                't_max_ms': float(t_fin.max() * 1e3),
            }
            if arm != 'sympy':
                sp = sy_t[sy_fin] / t[sy_fin]
                d['paired_speedup'] = {
                    'median': float(np.median(sp)),
                    'q25': float(np.percentile(sp, 25)),
                    'q75': float(np.percentile(sp, 75)),
                    'max': float(sp.max()),
                    'n_pairs': int(sy_fin.sum())}
            out[c][arm] = d
    cis = {}
    for arm in ARMS:
        r = arrays[f'unmasked/{arm}/ratio']
        r = np.where(np.isnan(r), 1.0, r)
        lo, hi = _boot_mean_ci(r)
        cis[f'unmasked/{arm}'] = [float(lo), float(hi)]
    for eff in SWEEP_EFFORTS:
        lo, hi = _boot_mean_ci(arrays[f'unmasked/f64_e{eff}/ratio'])
        cis[f'unmasked/f64_e{eff}'] = [float(lo), float(hi)]
    out['boot_ci_mean_ratio'] = cis
    sweep = {}
    for arm in ('f64_e0', 'f64', 'f64_e64'):
        r = arrays[f'unmasked/{arm}/ratio']
        t = arrays[f'unmasked/{arm}/t']
        sweep[arm] = {'mean': float(r.mean()), 'wins': float((r < 1).mean()),
                      'same': float((r == 1).mean()),
                      'bigger': float((r > 1).mean()),
                      't_p50_us': float(np.median(t) * 1e6)}
    out['sweep'] = sweep
    return out


# ---------------------------------------------------------------------- figures
BG = '#fbfbf9'
INK = '#1a1a1a'
SUB = '#5a5a5a'
FOOT = '#666666'
GRID = '#e3e3df'
SPINE = '#c9c9c4'
ANNOT = '#444444'

C = {'f64': '#1f77b4', 'real': '#9467bd', 'permissive': '#2ca02c',
     'sympy': '#d62728'}
LBL = {'f64': 'SimpliPy f64 (default)', 'real': 'SimpliPy real',
       'permissive': 'SimpliPy permissive', 'sympy': 'SymPy simplify'}
ARMS = ['f64', 'real', 'permissive', 'sympy']
ENGINE_ARMS = ['f64', 'real', 'permissive']

DPI = 170

TIME_TICKS = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
              ['1 us', '10 us', '100 us', '1 ms', '10 ms', '100 ms',
               '1 s', '10 s'])
SPEED_TICKS = ([1, 10, 100, 1e3, 1e4, 1e5, 1e6],
               ['1x', '10x', '100x', '1,000x', '10,000x', '100,000x',
                '1,000,000x'])

CORPUS_META = {
    'unmasked': dict(
        head='Benchmark — SR prior, unmasked',
        meta='Benchmark — unmasked leg  ·  engine acj-5-4-llm, '
             'effort 4  ·  sympy 1.14.0 (1 s cap)  ·  corpus: SR prior '
             'v25, seed 20260830  ·  scored by the shipped complexity()',
        out='ecdf_unmasked.png'),
    'masked': dict(
        head='Benchmark — SR prior, masked',
        meta='Benchmark — masked leg  ·  engine acj-5-4-llm, '
             'effort 4  ·  sympy 1.14.0 (1 s cap)  ·  corpus: v25 prior, '
             "seed 20260830, mask policy 'all'  ·  scored by the shipped "
             'complexity()',
        out='ecdf_masked_raw.png'),
    'external': dict(
        head='Benchmark — external: SOOSE',
        meta='Benchmark — external leg  ·  engine acj-5-4-llm, '
             'effort 4  ·  sympy 1.14.0 (1 s cap)  ·  corpus: SOOSE '
             'fc/nc/wc, all rows compile  ·  scored by the '
             'shipped complexity()',
        out='ecdf_external.png'),
}

PANEL_X = [0.040, 0.373, 0.706]
PANEL_W = 0.270


def f_us(v):
    return f'{v:,.0f}us'


def f_ms(v):
    return f'{v:,.1f}ms'


def f_pct1(f):
    return f'{f * 100:.1f}%'


def f_big(frac):
    p = frac * 100
    if p == 0.0:
        return '0.0%'
    if p >= 0.1:
        return f'{p:.1f}%'
    if p >= 0.01:
        return f'{p:.2f}%'
    return f'{p:.3f}%'


def f_x(v):
    return f'{v:,.0f}x'


def f_max(v):
    return '1' if v == 1.0 else f'{v:.3f}'


def ecdf(vals, n):
    v = np.sort(np.asarray(vals)[np.isfinite(vals)])
    return v, np.arange(1, len(v) + 1) / n


def style_ax(ax, fs=10):
    ax.set_facecolor(BG)
    ax.grid(True, which='major', color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=INK, labelsize=fs, direction='out')
    ax.tick_params(which='minor', colors=SPINE)


def panel_title(fig, x, y, title, subtitle, fs=13, sfs=10.5, dy=0.017):
    fig.text(x, y, title, fontweight='bold', fontsize=fs, color=INK)
    fig.text(x, y - dy, subtitle, fontsize=sfs, color=SUB)


def table(fig, label_x, y0, dy, cols, rows, fs=9.5, sw=0.011, sh=0.014,
          label_pad=0.016):
    """cols: [(header, x_right)]; rows: [(label, color_or_None, [cells])]."""
    from matplotlib.patches import Rectangle
    for h, xr in cols:
        fig.text(xr, y0, h, ha='right', va='center', color=SUB,
                 fontsize=fs, family='monospace')
    for i, (label, color, cells) in enumerate(rows):
        y = y0 - dy * (i + 1)
        if color is not None:
            fig.patches.append(Rectangle(
                (label_x, y - sh / 2), sw, sh, transform=fig.transFigure,
                facecolor=color, edgecolor='none', clip_on=False))
        fig.text(label_x + label_pad, y, label, va='center',
                 fontsize=fs + 0.5, color=INK)
        for (h, xr), cell in zip(cols, cells):
            fig.text(xr, y, cell, ha='right', va='center', fontsize=fs,
                     color=INK, family='monospace')


def footnote(fig, x, y, s, fs=8.7):
    fig.text(x, y, s, fontsize=fs, color=FOOT, style='italic')


def ratio_panel(ax, A, S, corpus, arms, lw=2.4, arrow=True):
    n = S[corpus]['f64']['n']
    for a in arms:
        r = A[f'{corpus}/{a}/ratio']
        x, y = ecdf(r, n)
        ax.plot(x, y, color=C[a], lw=lw, solid_capstyle='butt', zorder=3)
        ax.scatter([1.0], [S[corpus][a]['wins']], s=26, color=C[a], zorder=6)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 2.01, 0.25))
    ax.set_xticklabels([f'{v:.2f}' for v in np.arange(0, 2.01, 0.25)])
    ax.axvline(1.0, color=ANNOT, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(1.035, 0.56, 'ratio 1.0 = unchanged', fontsize=9, color=ANNOT,
            bbox=dict(fc=BG, ec='none', pad=1))
    if arrow:
        ax.annotate('', xy=(0.08, 0.40), xytext=(0.60, 0.40),
                    arrowprops=dict(arrowstyle='->', color=SUB, lw=1.2))
        ax.text(0.34, 0.445, 'compresses more', fontsize=9.5, color=SUB,
                ha='center')
    ax.set_xlabel('compression ratio   (output MDL / input MDL)',
                  fontsize=11, color=INK)
    ax.set_ylabel('fraction of rows', fontsize=11, color=INK)


def time_panel(ax, A, S, corpus, arms, lw=2.4, cap_note=None):
    n = S[corpus]['f64']['n']
    for a in arms:
        t = A[f'{corpus}/{a}/t']
        x, y = ecdf(t, n)
        ax.plot(x, y, color=C[a], lw=lw, solid_capstyle='butt', zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(1e-6, 10)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(TIME_TICKS[0])
    ax.set_xticklabels(TIME_TICKS[1])
    ax.axvline(1.0, color=ANNOT, ls=(0, (4, 3)), lw=1.4, zorder=2)
    ax.text(0.78, 0.30, '1 s cap', fontsize=9, color=ANNOT, rotation=90,
            ha='right', va='bottom')
    if cap_note:
        ax.text(1.15, 0.45, cap_note, fontsize=9, color=ANNOT, va='center')
    ax.set_xlabel('wall clock per row  (log scale)', fontsize=11, color=INK)
    ax.set_ylabel('fraction of rows', fontsize=11, color=INK)


def speed_panel(ax, A, S, corpus, lw=2.4):
    sy_t = A[f'{corpus}/sympy/t']
    fin = np.isfinite(sy_t)
    n_pairs = int(fin.sum())
    for a in ENGINE_ARMS:
        sp = sy_t[fin] / A[f'{corpus}/{a}/t'][fin]
        x, y = ecdf(sp, n_pairs)
        ax.plot(x, y, color=C[a], lw=lw, solid_capstyle='butt', zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 1.2e6)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(SPEED_TICKS[0])
    ax.set_xticklabels(SPEED_TICKS[1])
    ax.axvline(1.0, color=ANNOT, ls=(0, (4, 3)), lw=1.4, zorder=2)
    ax.text(1.25, 0.05, 'parity with SymPy', fontsize=9, color=ANNOT)
    med = S[corpus]['f64']['paired_speedup']['median']
    ax.axvline(med, color=C['f64'], ls=(0, (1, 2)), lw=1.6, zorder=2)
    ax.text(med * 0.8, 0.95, f'median {med:,.0f}x', fontsize=9.5,
            color=C['f64'], ha='right', zorder=7)
    ax.set_xlabel('SymPy time / SimpliPy time, same row  (log scale)',
                  fontsize=11, color=INK)
    ax.set_ylabel('fraction of rows', fontsize=11, color=INK)


def make_readme(plt, A, S, figdir):
    n = S['unmasked']['f64']['n']
    fig = plt.figure(figsize=(12.4, 5.4), dpi=DPI, facecolor=BG)

    # left: ratio ECDF
    axl = fig.add_axes([0.055, 0.415, 0.42, 0.445])
    style_ax(axl, fs=10.5)
    ratio_panel(axl, A, S, 'unmasked', ARMS, lw=3.0)
    panel_title(fig, 0.055, 0.945, 'Expression size after simplification',
                'output MDL / input MDL, paired per row;  SR prior, '
                f'unmasked (n={n:,})', fs=14.5, sfs=11, dy=0.037)

    ci = S['boot_ci_mean_ratio']
    rows = []
    for a in ARMS:
        d = S['unmasked'][a]
        lo, hi = ci[f'unmasked/{a}']
        rows.append((LBL[a], C[a], [
            f"{d['ratio_mean']:.3f} [{lo:.3f}, {hi:.3f}]",
            f_pct1(d['wins']), f_pct1(d['same']), f_big(d['bigger'])]))
    table(fig, 0.055, 0.315, 0.052,
          [('mean [95% CI]', 0.345), ('smaller', 0.40), ('same', 0.443),
           ('bigger', 0.497)], rows, fs=9.2, sw=0.010, sh=0.022)
    cens_pct = S['unmasked']['sympy']['censored_ratio'] / n * 100
    footnote(fig, 0.055, 0.055,
             "'smaller'/'same'/'bigger' = ratio <1/=1/>1;  sympy censored "
             f'rows ({cens_pct:.1f}%: timeout/unpriceable)', fs=8.3)
    footnote(fig, 0.055, 0.022,
             'score ratio 1 and end the curve below 1;  engine acj-5-4-llm '
             '(effort 4), sympy 1.14.0, one pinned core', fs=8.3)

    # right: wall clock ECDF
    axr = fig.add_axes([0.575, 0.415, 0.42, 0.445])
    style_ax(axr, fs=10.5)
    time_panel(axr, A, S, 'unmasked', ARMS, lw=3.0)
    panel_title(fig, 0.575, 0.945, 'Wall clock: time to simplify one row',
                'all arms serial on one pinned core;  sympy 1.14.0 under a '
                '1 s cap', fs=14.5, sfs=11, dy=0.037)

    rows = []
    for a in ARMS:
        d = S['unmasked'][a]
        rows.append((LBL[a], C[a],
                     [f_us(d['t_p50_us']), f_us(d['t_p95_us'])]))
    table(fig, 0.575, 0.315, 0.052,
          [('median', 0.83), ('p95', 0.94)], rows, fs=9.5, sw=0.010,
          sh=0.022)
    med = S['unmasked']['f64']['paired_speedup']['median']
    n_pairs = S['unmasked']['f64']['paired_speedup']['n_pairs']
    cap_pct = S['unmasked']['sympy']['censored_t'] / n * 100
    footnote(fig, 0.575, 0.055,
             f'median paired speedup {med:,.0f}x (sympy/f64, rows where '
             f'sympy finished: {n_pairs:,})', fs=8.3)
    footnote(fig, 0.575, 0.022,
             f'{cap_pct:.1f}% of rows hit the cap (censored): curve ends '
             'below 1; sympy stats are lower bounds', fs=8.3)

    path = os.path.join(figdir, 'ecdf_readme.png')
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path


def make_corpus(plt, A, S, corpus, figdir):
    meta = CORPUS_META[corpus]
    d = S[corpus]
    n = d['f64']['n']
    cens_r, cens_t = d['sympy']['censored_ratio'], d['sympy']['censored_t']
    n_pairs = d['f64']['paired_speedup']['n_pairs']

    fig = plt.figure(figsize=(19.5, 6.5), dpi=DPI, facecolor=BG)
    fig.text(0.008, 0.952, f"{meta['head']} (n={n:,})", fontweight='bold',
             fontsize=15.5, color=INK)
    fig.text(0.008, 0.912, meta['meta'], fontsize=9, color=SUB)

    def col_ax(x0):
        ax = fig.add_axes([x0, 0.40, PANEL_W, 0.40])
        style_ax(ax)
        return ax

    def foot(x0, lines):
        for i, text in enumerate(lines):
            footnote(fig, x0, 0.104 - 0.027 * i, text, fs=8.4)

    # --- left: headline MDL ECDF
    x0 = PANEL_X[0]
    panel_title(fig, x0, 0.858, 'Headline: MDL in SimpliPy space',
                'scored in the deployment space', fs=12, sfs=10, dy=0.030)
    ratio_panel(col_ax(x0), A, S, corpus, ARMS)
    rows = []
    for a in ARMS:
        s = d[a]
        rows.append((LBL[a], C[a], [
            f"{s['ratio_median']:.3f}", f_pct1(s['wins']), f_pct1(s['same']),
            f_big(s['bigger']), f_max(s['ratio_max'])]))
    table(fig, x0, 0.300, 0.040,
          [('median', x0 + 0.125), ('wins', x0 + 0.158),
           ('same', x0 + 0.191), ('bigger', x0 + 0.228),
           ('max', x0 + 0.268)], rows, sw=0.006, sh=0.021, label_pad=0.011)
    foot(x0, [
        "'wins'/'same'/'bigger' = ratio <1/=1/>1, summing to 100%;",
        "'max' is the true max (axis clipped at 2.0)",
        f'sympy censored rows — {cens_r:,} of {n:,} '
        f'({cens_r / n * 100:.1f}%) — score ratio 1.0 in the stats',
        'and end the curve below 1'])

    # --- middle: wall clock
    x0 = PANEL_X[1]
    panel_title(fig, x0, 0.858, 'Wall clock: time to simplify one row',
                'all arms serial on one pinned core;  sympy 1.14.0 under a '
                '1 s cap', fs=12, sfs=10, dy=0.030)
    time_panel(col_ax(x0), A, S, corpus, ARMS,
               cap_note=f'{cens_t / n * 100:.1f}% of rows stop\nat the '
                        'cap; curve\nends below 1')
    rows = []
    for a in ARMS:
        s = d[a]
        cens = f'{cens_t / n * 100:.1f}%' if a == 'sympy' else '-'
        rows.append((LBL[a], C[a], [
            f_us(s['t_p50_us']), f_us(s['t_p95_us']), f_ms(s['t_max_ms']),
            cens]))
    table(fig, x0, 0.300, 0.040,
          [('median', x0 + 0.131), ('p95', x0 + 0.178),
           ('max', x0 + 0.225), ('censored', x0 + 0.269)], rows,
          sw=0.006, sh=0.021, label_pad=0.011)
    foot(x0, [
        'sympy stats cover finished rows; censored rows are ≥ 1 s:',
        'true median/p95/max are larger'])

    # --- right: paired speedup
    x0 = PANEL_X[2]
    panel_title(fig, x0, 0.858,
                'Paired speedup: how much faster on the SAME expression',
                f'over the {n_pairs:,} rows where sympy finished; censored '
                'rows could only raise it', fs=12, sfs=10, dy=0.030)
    speed_panel(col_ax(x0), A, S, corpus)
    rows = []
    for a in ENGINE_ARMS:
        p = d[a]['paired_speedup']
        rows.append((f'SymPy / {a}', C[a], [
            f_x(p['median']), f_x(p['q25']), f_x(p['q75']), f_x(p['max'])]))
    table(fig, x0, 0.300, 0.040,
          [('median', x0 + 0.125), ('q25', x0 + 0.161),
           ('q75', x0 + 0.201), ('max', x0 + 0.268)], rows,
          sw=0.006, sh=0.021, label_pad=0.011)
    foot(x0, [
        'medians over rows where sympy finished; censored rows are',
        "sympy's slowest, so every number here is an underestimate"])

    path = os.path.join(figdir, meta['out'])
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path


def make_sweep(plt, A, S, figdir):
    n = S['unmasked']['f64']['n']
    fig = plt.figure(figsize=(8.6, 5.6), dpi=DPI, facecolor=BG)
    panel_title(fig, 0.085, 0.945,
                'Search budget: expression size at effort 0 / 4 / 64',
                f'unmasked SR leg, {n:,} rows; effort 4 and 64 coincide '
                '(budget 4 captures every budget-64 win)',
                fs=13.5, sfs=10.5, dy=0.045)
    ax = fig.add_axes([0.085, 0.40, 0.885, 0.445])
    style_ax(ax, fs=10)
    CE = {'f64_e0': '#666666', 'f64': '#1f77b4', 'f64_e64': '#e0a010'}
    for a in ('f64_e0', 'f64_e64'):
        x, y = ecdf(A[f'unmasked/{a}/ratio'], n)
        ax.plot(x, y, color=CE[a], lw=2.6, zorder=3)
    x, y = ecdf(A['unmasked/f64/ratio'], n)
    ax.plot(x, y, color=CE['f64'], lw=2.6, ls=(0, (5, 3)), zorder=4)
    for a, z in (('f64_e0', 5), ('f64_e64', 6), ('f64', 7)):
        ax.scatter([1.0], [S['sweep'][a]['wins']], s=26 if z < 7 else 12,
                   color=CE[a], zorder=z)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 2.01, 0.25))
    ax.set_xticklabels([f'{v:.2f}' for v in np.arange(0, 2.01, 0.25)])
    ax.axvline(1.0, color=ANNOT, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(1.035, 0.50, 'ratio 1.0 = unchanged', fontsize=9, color=ANNOT,
            bbox=dict(fc=BG, ec='none', pad=1))
    ax.set_xlabel('compression ratio   (output MDL / input MDL)',
                  fontsize=11, color=INK)
    ax.set_ylabel('fraction of rows', fontsize=11, color=INK)

    ci = S['boot_ci_mean_ratio']
    rows = []
    for a, lbl in (('f64_e0', 'effort 0 (plain chain)'),
                   ('f64', 'effort 4 (the default)'),
                   ('f64_e64', 'effort 64')):
        s = S['sweep'][a]
        lo, hi = ci[f'unmasked/{a}']
        rows.append((lbl, CE[a], [
            f"{s['mean']:.4f} [{lo:.4f}, {hi:.4f}]",
            f_pct1(s['wins']), f_pct1(s['same']), f_big(s['bigger']),
            f_us(s['t_p50_us'])]))
    table(fig, 0.10, 0.30, 0.055,
          [('mean [95% CI]', 0.545), ('smaller', 0.635), ('same', 0.715),
           ('bigger', 0.80), ('med', 0.90)], rows, fs=9.5, sw=0.012,
          sh=0.022, label_pad=0.022)
    footnote(fig, 0.085, 0.035,
             'engine acj-5-4-llm, unmasked SR leg (v25 prior, seed '
             '20260830); identical protocol and rows as the published '
             'panels')
    path = os.path.join(figdir, 'ecdf_effort_sweep.png')
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path


def make_figures(arrays, stats, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    paths = [make_readme(plt, arrays, stats, figdir)]
    for corpus in CORPORA:
        paths.append(make_corpus(plt, arrays, stats, corpus, figdir))
    paths.append(make_sweep(plt, arrays, stats, figdir))
    return paths


# ------------------------------------------------------------------------ driver
def resolve_corpora(limit):
    try:
        from symbolic_data.resolver import resolve
    except ImportError:
        sys.exit('corpora resolve through symbolic-data: pip install symbolic-data')
    artifact = resolve(CORPUS_ARTIFACT)
    corpora = {}
    for tag, fname in CORPUS_FILES.items():
        with open(artifact.paths[fname], 'rb') as fh:
            rows = pickle.load(fh)
        corpora[tag] = rows[:limit] if limit else rows
        print(f'{tag}: {len(corpora[tag])} rows ({fname})', flush=True)
    return corpora


def main():
    parser = argparse.ArgumentParser(
        description='The published SimpliPy-vs-SymPy benchmark: four arms '
                    '(f64/real/permissive at effort 4, sympy under a 1 s cap), '
                    'three corpora, plus the effort 0/64 sweep on the '
                    'unmasked leg.')
    parser.add_argument('--limit', type=int, default=None,
                        help='slice every corpus to its first N rows (smoke runs)')
    parser.add_argument('--out', default=None,
                        help='directory for summary + figures + checkpoint '
                             '(default: the repo paths)')
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        summary_path = os.path.join(args.out, 'ecdf_vs_sympy_summary.json')
        figdir = args.out
        results_pkl = os.path.join(args.out, RESULTS_PKL_NAME)
    else:
        summary_path = os.path.join(HERE, 'ecdf_vs_sympy_summary.json')
        figdir = os.path.join(REPO, 'docs', 'assets', 'benchmarks')
        results_pkl = os.path.join(HERE, RESULTS_PKL_NAME)

    from simplipy import Mode
    from simplipy.asset_manager import get_path
    cfg = get_path(ENGINE_ASSET, install=True)
    corpora = resolve_corpora(args.limit)

    results = {}
    for tag in CORPORA:
        corpus = corpora[tag]
        print(f'== {tag}: {len(corpus)} rows', flush=True)
        sympy_res = run_sympy(corpus, tag)
        modes = {}
        for mode in (Mode.f64, Mode.real, Mode.permissive):
            modes[mode.name] = run_simplipy(cfg, corpus, f'{tag}/{mode.name}',
                                            mode=mode, effort=4)
        if tag == 'unmasked':
            for eff in SWEEP_EFFORTS:
                modes[f'f64_e{eff}'] = run_simplipy(
                    cfg, corpus, f'{tag}/f64_e{eff}', mode=Mode.f64, effort=eff)
        results[tag] = {'sympy': sympy_res, 'modes': modes}
        with open(results_pkl, 'wb') as fh:  # checkpoint after each leg
            pickle.dump(results, fh)

    arrays, summary = score(cfg, corpora, results)
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'summary -> {summary_path}', flush=True)

    stats = compute_stats(arrays)
    for path in make_figures(arrays, stats, figdir):
        print(f'figure -> {path}', flush=True)


if __name__ == '__main__':
    main()
