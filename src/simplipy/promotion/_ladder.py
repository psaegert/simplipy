# mypy: ignore-errors
"""The subtree-sort promotion ladder driver: `_` -> `!` -> `?`.

Certifies each mined rewrite rule against the strongest wildcard sort it can soundly
bind at, stepping DOWN a three-rung ladder of progressively weaker bindings when a
stronger rung's bar is not met. The atom lattice (see `._pointwise.ATOMS`: 0, +-1,
integers, half-integers, +-pi, +-e, +-inf, nan) is the finite set of exact probe points;
on it, exact value equality is required, tolerating ONLY undefined -> defined extensions
(e.g. `x/x -> 1` filling the hole at 0). A defined value may never change to a different
defined value, even on a null set. The continuum keeps measure tolerance (the sampled
jurisdiction of the miner, not this instrument).

THREE TIERS over a rules artifact (each with its own bar, by the pushforward theorem):

  `_` (subtree sort, Dirac): the full exact-pointwise bar on the enriched atom lattice.
      Failures are RESPELLED to `?` (they keep everything they were ever certified for)
      and then face the `?` bar.
  `?` (variable-leaf sort, identity pushforward): the atom bar -- full atom product, no
      random draws (the continuum is the miner's sampled jurisdiction, not this
      instrument's). Kill = defined->different-defined (value change) or defined->undefined
      (domain shrink); undefined->defined is the tolerated extension.
  ground (no wildcards; `<constant>`-bearing): SKELETON semantics, for-all c exists finite
      c_t. Probed over the finite constant lattice (fitted constants are finite; no inf/nan
      constants exist to bind):
        - literal RHS: exact value equality at EVERY probed c (mpmath, high precision);
        - bare `<constant>` RHS: the LHS value must be FINITE at every probed c (a finite
          fitted constant must exist; nan/inf values are not constant-representable --
          e.g. `pow(c/5, nan) -> c` would invent a value for an undefined expression).
      Any other RHS shape with `<constant>` is UNSUPPORTED -> conservative kill.

Positive controls are checked first (raise on any miss). The public entry point is
`promote_rules(rules, engine)`; it returns the promoted rules plus a per-bucket report.
"""
import numpy as np

from ._pointwise import (ATOMS, MAX_PRODUCT, judge, prefold,  # noqa: E402
                         substitute, eval_point, wildcards, valuations_for,
                         correlated_valuations)
from ._const_bearing import certify_rule  # noqa: E402
from ._hp_equiv import evaluate as hp_eval  # noqa: E402
from mpmath import mp, mpf, isnan, isinf  # noqa: E402

SEED = 20260718
GROUND_DRAWS = 16     # seeded finite draws on the constant axis, atop the finite atom lattice
DPS = 60              # ground expressions are special-value algebra; 60 digits is ample

FINITE_ATOMS = [a for a in ATOMS if not a.startswith('float(')]


def atom_valuations(ws, rng):
    """FULL atom product for the `?` bar (no random draws); sampled past MAX_PRODUCT (k>=4)."""
    k = len(ws)
    if k == 0:
        return [{}]                     # ground expression: one empty valuation
    na = len(ATOMS) ** k
    if na <= MAX_PRODUCT:
        idx = np.stack(np.meshgrid(*[np.arange(len(ATOMS))] * k, indexing='ij'), -1).reshape(-1, k)
    else:
        idx = rng.integers(0, len(ATOMS), size=(MAX_PRODUCT, k))
    return [{w: ATOMS[j] for w, j in zip(ws, row)} for row in idx]


def _symbolic_atom(t):
    """The mp value a probe token denotes under real semantics: transcendental atoms bind
    SYMBOLICALLY at working precision -- a variable or constant ranges over the REALS, so
    pi/2 (where sin = 1 exactly) is a legitimate probe point even though no f64 equals it.
    Everything else binds as its dyadic value. Build INSIDE the caller's workdps."""
    from mpmath import mp, mpf, inf as mp_inf, nan as mp_nan
    special = {'float("inf")': lambda: mp_inf, 'float("-inf")': lambda: -mp_inf,
               'float("nan")': lambda: mp_nan,
               '3.141592653589793': lambda: +mp.pi, '(-3.141592653589793)': lambda: -mp.pi,
               '2.718281828459045': lambda: +mp.e, '(-2.718281828459045)': lambda: -mp.e,
               '1.5707963267948966': lambda: mp.pi / 2, '(-1.5707963267948966)': lambda: -mp.pi / 2,
               '6.283185307179586': lambda: 2 * mp.pi, '(-6.283185307179586)': lambda: -2 * mp.pi,
               '0.7853981633974483': lambda: mp.pi / 4, '(-0.7853981633974483)': lambda: -mp.pi / 4}
    return special[t]() if t in special else mpf(t.strip('()'))


def _env_mp(wmap):
    """ATOM valuation -> mp env (symbolic transcendentals per `_symbolic_atom`)."""
    return {w: _symbolic_atom(t) for w, t in wmap.items()}


def mp_value_at(tokens, wmap):
    """Exact value classification of `tokens` at an atom valuation.

    f64 is unreliable at the atoms in two ways: zero coincidences (f64 sin(pi) = 1.2e-16, so
    `sin(np.pi) * inf` reads inf where exact arithmetic gives 0 * inf = nan) and POLE
    coincidences (`tan(atan(inf))`: atan(inf) is exactly pi/2 and tan(pi/2) is a pole, but
    every finite precision renders a huge finite number). Both are classified by a
    precision-stability gate: evaluate at dps 60, escalate suspicious values (|v| < 1e-40,
    |v| > 1e40, or non-finite) to dps 120 --
      residue SHRINKS with precision  -> exact zero;
      magnitude GROWS with precision  -> exact pole -> undefined (nan);
      stable                          -> the value itself.
    Returns an mp value (nan encodes undefined)."""
    from mpmath import mp, mpf, isnan as mnan, isinf as minf, nan as mp_nan
    with mp.workdps(60):
        a = hp_eval(list(tokens), _env_mp(wmap), [])
    suspicious = mnan(a) or minf(a) or (a != 0 and (abs(a) < mpf('1e-40') or abs(a) > mpf('1e40')))
    if not suspicious:
        return a
    with mp.workdps(120):
        b = hp_eval(list(tokens), _env_mp(wmap), [])
    if mnan(a) and mnan(b):
        return mp_nan
    if mnan(a) != mnan(b):
        return mp_nan                                    # nan-unstable: not a provable value
    if minf(a) or minf(b):
        return a if (minf(a) and minf(b) and (a > 0) == (b > 0)) else mp_nan
    if (a == 0 and b == 0) or (abs(a) < mpf('1e-40') and abs(b) <= abs(a) * mpf('1e-20')):
        return mpf(0)                                    # residue shrinks: exact zero
    if abs(a) > mpf('1e40') and abs(b) >= abs(a) * mpf('1e20'):
        return mp_nan                                    # magnitude grows with dps: exact pole
    return b


def contract_value_at(tokens, wmap):
    """The contract value of `tokens` at an atom valuation.

    Under the semantics this gate enforces (real-valued quantifiers, a single zero with
    1/0 = +inf, transcendental-literal denotation) the mp arm (`mp_value_at`: symbolic
    transcendental atoms, two-rung zero/pole snaps, single-zero mpmath arithmetic, the
    high-precision evaluation table) IS the value. f64 is NOT consulted for signed zeros:
    a single zero has no sign to disagree on, and letting a -0-mediated infinity sign in
    would wrongly convict sound rewrites (e.g. inv(0*x0) -> +inf). f64 remains only as an
    evaluation FALLBACK when the mp arm cannot evaluate at all. Returns a float (nan encodes
    undefined)."""
    from mpmath import isnan as mnan, isinf as minf
    try:
        vm = mp_value_at(tokens, wmap)
    except Exception:
        try:
            return eval_point(substitute(list(tokens), wmap))   # fallback: deployed f64
        except Exception:
            return float('nan')
    if mnan(vm):
        return float('nan')
    if minf(vm):
        return float('inf') if vm > 0 else float('-inf')
    return float(vm)


def judge_r2prime(lhs, rhs, valuations):
    """The `?` atom bar: extension tolerated, value change / domain shrink killed. Judged on
    `contract_value_at` (see its docstring). Prefolded first; rule-correlated atom valuations
    appended."""
    lhs, rhs = prefold(lhs), prefold(rhs)
    valuations = list(valuations) + correlated_valuations(lhs, rhs, wildcards(list(lhs) + list(rhs)))
    for wmap in valuations:
        try:
            a = contract_value_at(lhs, wmap)
            b = contract_value_at(rhs, wmap)
        except Exception as ex:
            return 'EVAL-ERR', f'{type(ex).__name__}: {ex}'
        an, bn = np.isnan(a), np.isnan(b)
        if an:
            continue                                     # undefined source: extension tolerated
        if bn:
            return 'KILL', (dict(wmap), a, b)            # defined -> undefined: domain shrink
        if np.isinf(a) or np.isinf(b):
            if a == b:
                continue
            return 'KILL', (dict(wmap), a, b)            # inf sign/finiteness change
        # Single zero (1/0 = +inf): an f64 zero's sign is measurement rendering, never a value
        # difference, so no signed-zero tie-break is applied here (float == treats -0 == +0).
        if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)):
            continue
        return 'KILL', (dict(wmap), a, b)                # defined-value change at an atom
    return 'PASS', None


# --- the pow integer-preimage probe --------------------------------------------------------
# A rule like `pow float("-inf") asinh _0 -> pow float("inf") _0` is violated exactly where
# the exponent chain crosses a small integer (asinh(x) = 1 at x = sinh(1) = 1.1752...) -- a
# point NO finite atom lattice contains. For single-wildcard rules whose pow has a literal
# +-inf / 0 / negative-numeric base, solve chain(x) = k for k in [-5, 5] by bisection and
# compare the two sides' EXACT pow classes at the root (integer-exponent algebra, so the
# transcendental root needs no precision at all on the solved side).

def _pow_class(base, k):
    """Exact value of pow(base, y) for INTEGER y = k, in the deployed algebra."""
    if np.isnan(base):
        return float('nan')
    if k == 0:
        return 1.0
    if np.isinf(base):
        if k > 0:
            return base if k % 2 else abs(base)
        return -0.0 if (base < 0 and k % 2) else 0.0
    if base == 0.0:
        neg = np.signbit(base)
        if k > 0:
            return -0.0 if (neg and k % 2) else 0.0
        return float('-inf') if (neg and k % 2) else float('inf')
    try:
        return float(base) ** k
    except OverflowError:
        return float('inf') if (base > 0 or k % 2 == 0) else float('-inf')


def _subtree_end(tokens, i, arity):
    j = i + 1
    for _ in range(arity.get(tokens[i], 0)):
        j = _subtree_end(tokens, j, arity)
    return j


def _val_token(v):
    """Spell a float value as an expression token both evaluators parse."""
    if np.isnan(v):
        return 'float("nan")'
    if np.isinf(v):
        return 'float("inf")' if v > 0 else 'float("-inf")'
    return repr(float(v)) if v >= 0 else f'({float(v)!r})'


def pow_preimage_witness(lhs, rhs):
    """Search for an integer-preimage witness killing the rule; None if none found.

    Scope: single-wildcard rules where some `pow` on either side has a LITERAL base
    (+-inf, 0, or a negative numeric) and the wildcard inside its exponent subtree.
    Mechanism: solve chain(x*) = k (k integer) by bisection; at x* the solved pow's value
    is EXACT integer-exponent algebra (`_pow_class`) -- substitute that value back into the
    expression as a token and evaluate the surroundings numerically. Any pow on either side
    whose chain is TOKEN-IDENTICAL to the solved chain lands on exactly k too and gets the
    same exact substitution, so identical-chain rules (`pow -inf f(_0) -> pow +inf f(_0)`)
    are compared exactly on both sides. Comparison follows the atom bar (extension toward the
    target tolerated; value change or shrink kills)."""
    from ._f64_eval import ARITY, LITERALS as LITS, _num
    toks_l, toks_r = list(lhs), list(rhs)
    ws = wildcards(toks_l + toks_r)
    if len(ws) != 1:
        return None
    w = ws[0]

    def pow_sites(tokens):
        out = []
        for i, t in enumerate(tokens):
            if t != 'pow':
                continue
            b0 = i + 1
            b1 = _subtree_end(tokens, b0, ARITY)
            e1 = _subtree_end(tokens, b1, ARITY)
            if b1 != b0 + 1:
                continue                      # base must be a single literal token
            bv = LITS.get(tokens[b0], _num(tokens[b0]))
            if bv is None or not (np.isinf(bv) or bv <= 0.0):
                continue
            if w in tokens[b1:e1]:
                out.append((i, e1, bv, tuple(tokens[b1:e1])))
        return out

    def f64_at(tokens, x):
        try:
            return eval_point(substitute(list(tokens), {w: _val_token(x)}))
        except Exception:
            return float('nan')

    def side_with_exact_pows(tokens, solved_chain, k, x):
        """Evaluate a side at x, with every literal-base pow whose chain equals the solved
        chain replaced by its exact integer-exponent class value."""
        toks = list(tokens)
        for i, e1, bv, chain in sorted(pow_sites(toks), reverse=True):
            if chain == solved_chain:
                toks[i:e1] = [_val_token(_pow_class(bv, k))]
        return f64_at(toks, x)

    all_sites = pow_sites(toks_l) + pow_sites(toks_r)
    if not all_sites:
        return None
    grid = np.linspace(-40.0, 40.0, 4001)
    seen_chains = set()
    for _, _, _, chain in all_sites:
        if chain in seen_chains:
            continue
        seen_chains.add(chain)
        hv = np.array([f64_at(list(chain), x) for x in grid])
        for k in range(-5, 6):
            d = hv - k
            sgn = np.sign(d)
            roots = [float(grid[i]) for i in np.where(d == 0.0)[0][:3]]   # exact grid hits
            hits = np.where((sgn[:-1] * sgn[1:] < 0) & np.isfinite(d[:-1]) & np.isfinite(d[1:]))[0]
            for i in hits[:3]:
                lo, hi = grid[i], grid[i + 1]
                slo = np.sign(f64_at(list(chain), lo) - k)
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if np.sign(f64_at(list(chain), mid) - k) == slo:
                        lo = mid
                    else:
                        hi = mid
                roots.append(0.5 * (lo + hi))
            for x_star in roots:
                a = side_with_exact_pows(toks_l, chain, k, x_star)
                b = side_with_exact_pows(toks_r, chain, k, x_star)
                an, bn = np.isnan(a), np.isnan(b)
                if an:
                    continue                              # extension toward target: tolerated
                kill = bn or ((np.isinf(a) or np.isinf(b)) and a != b) or \
                    (np.isfinite(a) and np.isfinite(b)
                     and abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b)))
                if kill:
                    return {'x*': float(x_star), 'k': k, 'lhs_val': a, 'rhs_val': b}
    return None


def mp_finite(v):
    return not (isnan(v) or isinf(v))


def mp_eq(a, b):
    if isnan(a) and isnan(b):
        return True
    if isnan(a) != isnan(b):
        return False
    if isinf(a) or isinf(b):
        return a == b
    from mpmath import fabs
    return fabs(a - b) <= mpf('1e-30') * max(1, fabs(a), fabs(b))


def judge_ground(lhs, rhs, rng):
    """SKELETON semantics for `<constant>`-bearing ground rules: for-all c exists finite c_t."""
    ns = lhs.count('<constant>')
    if ns != 1:
        return 'UNSUPPORTED', f'{ns} LHS <constant> slots (certifier handles exactly 1)'
    bare_slot = list(rhs) == ['<constant>']
    if '<constant>' in rhs and not bare_slot:
        return 'UNSUPPORTED', 'structured <constant> RHS (needs a witness map)'
    mp.dps = DPS
    # Constants range over the REALS, so transcendental probe atoms bind SYMBOLICALLY here
    # too. This keeps the ground tier consistent with the `?` bar: binding pi/2 as the f64
    # rational (sin < 1) would pass `pow sin <constant> inf -> 0` while the `?` bar binds it
    # symbolically (sin = 1) and kills it -- a tier inconsistency that ships an unsound rule.
    probes = ([_symbolic_atom(a) for a in FINITE_ATOMS]
              + [mpf(repr(float(v))) for v in np.round(rng.normal(0, 5, GROUND_DRAWS), 6)])
    for c in probes:
        try:
            v = hp_eval(list(lhs), {}, [c])
        except Exception as ex:
            return 'EVAL-ERR', f'{type(ex).__name__}: {ex}'
        if bare_slot:
            if not mp_finite(v):
                return 'KILL', (float(c), str(v), '<no finite c_t>')
        else:
            try:
                w = hp_eval(list(rhs), {}, [])
            except Exception as ex:
                return 'EVAL-ERR', f'{type(ex).__name__}: {ex}'
            if not mp_eq(v, w):
                return 'KILL', (float(c), str(v), str(w))
            # (No f64 zero-sign check here: a single zero makes `/ <constant> inf -> 0` sound.)
    return 'PASS', None


def _pow_inf_sites(toks, ARITY):
    """Base token-tuples of every pow(base, literal +-inf) subterm."""
    INF_T = ('float("inf")', 'float("-inf")')
    out = []
    for i, t in enumerate(toks):
        if t != 'pow':
            continue
        b0 = i + 1
        b1 = _subtree_end(toks, b0, ARITY)
        e1 = _subtree_end(toks, b1, ARITY)
        if e1 - b1 == 1 and toks[b1] in INF_T:
            out.append(tuple(toks[b0:b1]))
    return out


def _base_range_spiky(base, core):
    """Does this base's interval range REACH +-1 non-degenerately (so pow(base,+-inf) has a
    spike at a real point)? Wildcards/<constant> become independent wide axes (they range over
    the reals). Fail-closed: an unevaluable base counts as spiky. `core` is the compiled engine
    core exposing the `interval_value_set_box` FFI query."""
    toks, m = [], {}
    for t in base:
        if t.startswith(('_', '?', '!')) or t == '<constant>':
            m.setdefault(t, f'x{len(m)}')
            toks.append(m[t])
        else:
            toks.append(t)
    n = max(len(m), 1)
    try:
        has_fin, pinf, ninf, nan, lo, hi = core.interval_value_set_box(
            list(toks), [-1e6] * n, [1e6] * n)
    except Exception:
        return True
    lo = float('-inf') if ninf else lo
    hi = float('inf') if pinf else hi
    return lo < hi and (lo <= 1.0 <= hi or lo <= -1.0 <= hi)


def spike_flattener_witness(lhs, rhs, core):
    """Structural refusal of pow-inf SPIKE FLATTENERS (scope: flatteners only; wrappers pass).

    pow(t, +-inf) is a STEP in its base t: |t|<1 -> 0, |t|=1 -> 1, |t|>1 -> +inf. If the
    base's real range reaches +-1 over a non-degenerate interval, the term SPIKES at a real
    point (pow(sin(x),inf) = 1 exactly at x = pi/2). A rule whose RHS DROPS every such term
    replaces the spike with the a.e. value -- unsound at a real point under real semantics --
    and no fixed atom lattice can chase the spike when it moves with a fitted constant
    (pow(sin(c*x),inf) violates at x = pi/(2c), a point the fit chooses). Refusing the FORM
    is invariant to that motion. Rules whose RHS keeps a spiky pow-inf term (wrappers like
    abs(pow(_0,inf)) -> pow(_0,inf)) carry the spike honestly and are allowed through to the
    normal bars. Conservative by construction (interval over-approximation can only
    over-refuse); the lo==hi degenerate guard keeps exact ground facts like pow(-1,inf)->1.

    `core` is the compiled engine core supplying the interval query (see `_base_range_spiky`)."""
    from ._f64_eval import ARITY
    l_spiky = [b for b in _pow_inf_sites(list(lhs), ARITY) if _base_range_spiky(b, core)]
    if not l_spiky:
        return None
    r_spiky = [b for b in _pow_inf_sites(list(rhs), ARITY) if _base_range_spiky(b, core)]
    if r_spiky:
        return None                       # spike carried over in kind: wrapper, honest
    return {'spiky_base': ' '.join(l_spiky[0])}


NONFINITE_TOKENS = {'float("inf")', 'float("-inf")', 'float("nan")'}


def finite_only(valuations):
    return [v for v in valuations if not any(t in NONFINITE_TOKENS for t in v.values())]


def judge_bang(lhs, rhs, rng):
    """The `!`-sort promotion bar: exact pointwise over FINITE valuations only (atoms + draws
    + correlated), plus the pow-preimage probe. A `!`-bound subtree is certified finite a.e.,
    so its pushforward puts NO mass on {nan, +-inf}: behaviour at nonfinite valuations is
    a.e.-irrelevant and dropped from the bar. A constant subtree is a Dirac at a FINITE value,
    which is why the finite bar must stay EXACT.

    An UNDECIDED near-band (f64 rounding is denied the authority to convict) escalates to the
    exact arbiter (`._overturn.judge_exact`) on the same finite valuations -- this is what lets
    saturating true identities (`atanh tanh ?0 -> ?0`) clear the bar and regain subtree
    binding."""
    ws = wildcards(list(lhs) + list(rhs))
    vals = finite_only(valuations_for(ws, rng))
    v, info = judge(list(lhs), list(rhs), vals)
    if v in ('UNDECIDED', 'DEMOTE'):
        # DEMOTE escalates too: an f64 kill at a FINITE valuation can be pure saturation
        # (tanh(37) == 1.0 exactly, so atanh(tanh(37)) reads inf), and f64 is denied the
        # authority to convict; the exact arbiter confirms real kills and overturns
        # saturation artifacts (real semantics governs).
        from ._overturn import judge_exact
        v2, _kill = judge_exact(list(lhs), list(rhs), vals)
        if v2 == 'PROMOTE':
            v, info = 'PROMOTE', 'exact-arbiter'
    if v == 'PROMOTE' and pow_preimage_witness(lhs, rhs) is not None:
        return 'DEMOTE', 'POW-PREIMAGE'
    return v, info


def respell_bang(tokens):
    return tuple('!' + t[1:] if t.startswith(('_', '?')) else t for t in tokens)


def tier_of(lhs, rhs):
    """Five-way routing: {`_`, `?`} x {const-free, const-bearing}, plus ground. Const-bearing
    wildcard rules go through the forall-exists witness machinery (`._const_bearing.certify_rule`);
    routing them to the const-free judge would raise on the unbound `<constant>` slot."""
    toks = list(lhs) + list(rhs)
    cb = '<constant>' in toks
    if any(t.startswith('_') for t in toks):
        return '_cb' if cb else '_cf'
    if any(t.startswith('?') for t in toks):
        return '?cb' if cb else '?cf'
    return 'ground'


def respell(tokens):
    return tuple('?' + t[1:] if t.startswith('_') else t for t in tokens)


def run_controls(rng, core):
    """Positive-control self-test: run each labelled control through its judge and raise
    RuntimeError on any verdict that does not match the expected outcome. This guards against
    a miscalibrated judge silently mis-promoting rules."""
    ctrl = [
        # a null-set value change (4 vs 1 at x0 = +-1): killed on the ? bar
        ('?', (['*', '4.0', 'pow', '?0', 'float("inf")'], ['pow', '?0', 'float("inf")']), 'KILL'),
        # x/x -> 1: extension-only (nan at 0 and +-inf) -- must SURVIVE the ? bar
        ('?', (['/', '?0', '?0'], ['1']), 'PASS'),
        ('?', (['+', '?0', '0'], ['?0']), 'PASS'),
        # domain shrink at an atom: log(?0) undefined at ?0 = -1 where the source is defined
        ('?', (['+', '?0', '0'], ['log', '?0']), 'KILL'),
        # transcendental zero + single zero: sin(pi) IS 0 (prefold), 0*t = 0 for finite t
        # (zero-sign is measurement rendering, not a value), and 0*inf = nan -> 0 is a
        # tolerated null-set extension at the atoms. PASS.
        ('?', (['*', 'sin', 'np.pi', '?0'], ['0']), 'PASS'),
        # pole coincidence: atan(inf) = pi/2 exactly, tan(pi/2) undefined -> extension
        ('?', (['tan', 'atan', '?0'], ['?0']), 'PASS'),
        # signed zero in the deployed algebra: pow1_3(-0.0) = +0.0 (the where-branch), so
        # LHS(-inf) = inv(+0.0) = +inf vs RHS = -inf -- a genuine kill
        ('?', (['inv', 'pow1_3', 'inv', '?0'], ['pow1_3', '?0']), 'KILL'),
        # pow(0, exp(-inf)) = pow(0, 0) = 1 vs 0: a genuine value change at the atom
        ('?', (['pow', '0', 'exp', '?0'], ['0']), 'KILL'),
        # real semantics: sin attains 1 AT the real point pi/2, so pow(sin(?0),inf) -> 0
        # changes a defined value at a real point -- KILL. The symbolic pi/2 atom sees it.
        ('?', (['pow', 'sin', '?0', 'float("inf")'], ['0']), 'KILL'),
        # pow(c/5, nan) is nan for c != 5: cannot be simplified to a fitted <constant>
        ('ground', (['pow', 'div5', '<constant>', 'float("nan")'], ['<constant>']), 'KILL'),
        # the acos(pow(c, inf)) family: nan on ALL of c > 1 -- positive-measure, not just an atom
        ('ground', (['acos', 'pow', '<constant>', 'float("inf")'], ['<constant>']), 'KILL'),
        # sound ground rule: c + log(0) = -inf for every finite c
        ('ground', (['+', '<constant>', 'log', '0'], ['float("-inf")']), 'PASS'),
        # single zero: c/inf = 0 for every finite c -- sound
        ('ground', (['/', '<constant>', 'float("inf")'], ['0']), 'PASS'),
        # ground powsin: killed by the symbolic pi/2 probe on the CONSTANT axis
        ('ground', (['pow', 'sin', '<constant>', 'float("inf")'], ['0']), 'KILL'),
        # structural refusal (belt over the atoms' suspenders): flatteners refused, wrappers
        # and degenerate ground facts allowed
        ('spike', (['pow', 'sin', '?0', 'float("inf")'], ['0']), 'REFUSE'),
        ('spike', (['pow', 'sin', '<constant>', 'float("inf")'], ['0']), 'REFUSE'),
        ('spike', (['abs', 'pow', '_0', 'float("inf")'], ['pow', '_0', 'float("inf")']), 'ALLOW'),
        ('spike', (['inv', 'pow', '_0', 'float("inf")'], ['pow', '_0', 'float("-inf")']), 'ALLOW'),
        ('spike', (['pow', '(-1)', 'float("inf")'], ['1']), 'ALLOW'),
        # a sign-clean survivor: c * inf = +inf is spelled inf, no zero involved
        ('ground', (['+', '<constant>', 'atanh', '1'], ['float("inf")']), 'PASS'),
    ]
    for tier, (lhs, rhs), want in ctrl:
        if tier == '?':
            v, info = judge_r2prime(lhs, rhs, atom_valuations(wildcards(lhs + rhs), rng))
        elif tier == 'spike':
            w = spike_flattener_witness(lhs, rhs, core)
            v, info = ('REFUSE', w) if w is not None else ('ALLOW', None)
        else:
            v, info = judge_ground(lhs, rhs, rng)
        if v != want:
            raise RuntimeError(
                f'positive control failed: [{tier}] {" ".join(lhs)} -> {" ".join(rhs)} '
                f'got {v} want {want} ({str(info)[:120]})')


def promote_rules(rules, engine, *, seed=SEED, run_positive_controls=True):
    """Certify `rules` down the `_` -> `!` -> `?` ladder (plus the ground tier) and return the
    promoted rules alongside a per-bucket report.

    Parameters
    ----------
    rules : iterable of (lhs, rhs)
        Each side is a sequence of prefix-expression tokens (wildcards `_i`/`?i`, literals,
        `<constant>`), as produced by the miner. Matches the format the standalone pass loaded
        from JSON.
    engine : SimpliPyEngine
        Supplies the compiled interval core via `engine._core`; its `interval_value_set_box`
        FFI query drives the spike-flattener refusal. The interval query is over raw
        expressions, so a rules-empty engine is sufficient (and matches the original setup).
    seed : int
        Seed for the single shared numpy RNG (all sampling flows from it); defaults to the
        pass's fixed seed so behaviour is reproducible.
    run_positive_controls : bool
        Run the positive-control self-test first (raising on any miss) before promotion.

    Returns
    -------
    (kept, report) : (list of (lhs, rhs), dict)
        `kept` is the promoted rule set (with `_`/`!`/`?` sort respelling applied); `report`
        buckets every routing decision (`promoted_bang`, `demoted_to_q`, `refused_spike`,
        `killed_q`, `killed_ground`, `undecided`, `no_witness`, `eval_err`, `unsupported`).
    """
    core = engine._core
    rng = np.random.default_rng(seed)
    if run_positive_controls:
        run_controls(rng, core)
    rules = list(rules)
    tiers = {'_cf': [], '_cb': [], '?cf': [], '?cb': [], 'ground': []}
    for lhs, rhs in rules:
        tiers[tier_of(lhs, rhs)].append((lhs, rhs))

    report = {'demoted_to_q': [], 'promoted_bang': [], 'refused_spike': [],
              'killed_q': [], 'killed_ground': [],
              'undecided': [], 'no_witness': [], 'eval_err': [], 'unsupported': []}
    kept = []

    # `_` tiers: exact-pointwise bar on the ENRICHED lattice; failures step DOWN THE SORT
    # LADDER `_` -> `!` -> `?` (each strictly weaker binding, each with its own bar).
    q_cf = list(tiers['?cf'])
    q_cb = list(tiers['?cb'])

    def ladder_cf(lhs, rhs, verdict, info):
        """A `_cf` failure tries `!` (finite-exact bar) before falling to `?`."""
        bv, _ = judge_bang(lhs, rhs, rng)
        if bv == 'PROMOTE':
            report['promoted_bang'].append((lhs, rhs, verdict, str(info)[:120]))
            kept.append((respell_bang(lhs), respell_bang(rhs)))
        else:
            report['demoted_to_q'].append((lhs, rhs, verdict, str(info)[:120]))
            q_cf.append((respell(lhs), respell(rhs)))

    for lhs, rhs in tiers['_cf']:
        ws = wildcards(list(lhs) + list(rhs))
        v, info = judge(list(lhs), list(rhs), valuations_for(ws, rng))
        if v == 'PROMOTE':
            pw = pow_preimage_witness(lhs, rhs)
            if pw is not None:
                ladder_cf(lhs, rhs, 'POW-PREIMAGE', pw)
                continue
            # Contract-atom check: the f64 `_` judge speaks deployed-numpy semantics at the
            # atoms (signed zeros, numpy pow(-inf, non-int) = nan) while the enforced contract
            # is single-zero real (the mp arm). A `_` promote must ALSO hold at the mp-contract
            # atoms, else it takes the ladder -- otherwise `pow _0 div4 1 -> pow1_4 _0` would
            # ship while the certifier kills its own ?-tier div2 twin on the identical witness.
            av, ainfo = judge_r2prime(list(lhs), list(rhs), atom_valuations(ws, rng))
            if av != 'PASS':
                ladder_cf(lhs, rhs, 'CONTRACT-ATOM', ainfo)
                continue
            kept.append((lhs, rhs))
        elif v in ('DEMOTE', 'UNDECIDED'):
            # UNDECIDED at the `_` bar is fail-safe demotion (rounding may not convict, but
            # it may not certify either); the ladder decides where it lands.
            ladder_cf(lhs, rhs, v, info)
        else:
            report['eval_err'].append((lhs, rhs, v, str(info)[:120]))

    for lhs, rhs in tiers['_cb']:
        v, info = certify_rule(lhs, rhs, rng)
        if v == 'PROMOTE':
            ws = wildcards(list(lhs) + list(rhs))
            av, ainfo = certify_rule(lhs, rhs, rng, judge_fn=judge_r2prime,
                                     vals_override=atom_valuations(ws, rng))
            if av != 'PROMOTE':
                bv, _ = certify_rule(lhs, rhs, rng, vals_override=finite_only(
                    valuations_for(ws, rng)))
                if bv == 'PROMOTE' and pow_preimage_witness(lhs, rhs) is None:
                    report['promoted_bang'].append((lhs, rhs, 'CONTRACT-ATOM', str(ainfo)[:120]))
                    kept.append((respell_bang(lhs), respell_bang(rhs)))
                else:
                    report['demoted_to_q'].append((lhs, rhs, 'CONTRACT-ATOM', str(ainfo)[:120]))
                    q_cb.append((respell(lhs), respell(rhs)))
                continue
            kept.append((lhs, rhs))
        elif v in ('DEMOTE', 'NO-WITNESS'):
            bv, _ = certify_rule(lhs, rhs, rng, vals_override=finite_only(
                valuations_for(wildcards(list(lhs) + list(rhs)), rng)))
            if bv == 'PROMOTE' and pow_preimage_witness(lhs, rhs) is None:
                report['promoted_bang'].append((lhs, rhs, v, str(info)[:120]))
                kept.append((respell_bang(lhs), respell_bang(rhs)))
            else:
                report['demoted_to_q'].append((lhs, rhs, v, str(info)[:120]))
                q_cb.append((respell(lhs), respell(rhs)))
        else:
            report['eval_err'].append((lhs, rhs, v, str(info)[:120]))

    # `?` const-free (incl. fresh demotions): the atom bar
    seen = set()
    for lhs, rhs in q_cf:
        if (lhs, rhs) in seen:
            continue
        seen.add((lhs, rhs))
        v, info = judge_r2prime(list(lhs), list(rhs), atom_valuations(wildcards(list(lhs) + list(rhs)), rng))
        if v == 'PASS':
            pw = pow_preimage_witness(lhs, rhs)
            if pw is not None:
                # a preimage witness is a REAL variable-leaf input (x = sinh(1) etc.), so at
                # the ? sort this is a defined-value change on an actual point: killed.
                report['killed_q'].append((lhs, rhs, 'POW-PREIMAGE ' + str(pw)[:140]))
                continue
            # UPGRADE attempt: `!` strictly generalizes `?` (a variable leaf is trivially
            # finite a.e.), so a ? rule that also clears the finite-exact bar ships as `!`
            # and gains certified-subtree recall for free.
            bv, _ = judge_bang(lhs, rhs, rng)
            if bv == 'PROMOTE':
                report['promoted_bang'].append((lhs, rhs, '?-upgrade', ''))
                kept.append((respell_bang(lhs), respell_bang(rhs)))
            else:
                kept.append((lhs, rhs))
        elif v == 'KILL':
            report['killed_q'].append((lhs, rhs, str(info)[:160]))
        elif v == 'UNDECIDED':
            report['undecided'].append((lhs, rhs, str(info)[:120]))
        else:
            report['eval_err'].append((lhs, rhs, v, str(info)[:120]))

    # `?` const-bearing (incl. fresh demotions): the atom bar THROUGH the witness machinery
    # (forall c_s exists c_t; instantiated pairs held to judge_r2prime on atom-only valuations).
    # NO-WITNESS here is a solver limitation: held OUT of the artifact, conservative, reported
    # in its own bucket.
    for lhs, rhs in q_cb:
        if (lhs, rhs) in seen:
            continue
        seen.add((lhs, rhs))
        ws = wildcards(list(lhs) + list(rhs))
        v, info = certify_rule(lhs, rhs, rng, judge_fn=judge_r2prime,
                               vals_override=atom_valuations(ws, rng))
        if v == 'PROMOTE':
            bv, _ = certify_rule(lhs, rhs, rng, vals_override=finite_only(
                valuations_for(ws, rng)))
            if bv == 'PROMOTE' and pow_preimage_witness(lhs, rhs) is None:
                report['promoted_bang'].append((lhs, rhs, '?-upgrade', ''))
                kept.append((respell_bang(lhs), respell_bang(rhs)))
            else:
                kept.append((lhs, rhs))
        elif v == 'DEMOTE':
            report['killed_q'].append((lhs, rhs, str(info)[:160]))
        elif v == 'NO-WITNESS':
            report['no_witness'].append((lhs, rhs, str(info)[:120]))
        else:
            report['eval_err'].append((lhs, rhs, v, str(info)[:120]))

    # Seeded canonical identities: the additive-cancel family cannot be MINED -- its sources
    # (`- x0 x0`) are pre-canceled at enumeration, so no rule carrier ever exists. Seeds are
    # certified through the SAME judge_bang bar as everything else, never trusted by fiat.
    lhs_seen = {tuple(l) for l, _ in kept}
    for lhs, rhs in [(('-', '!0', '!0'), ('0',)),
                     (('+', '!0', 'neg', '!0'), ('0',)),
                     (('+', 'neg', '!0', '!0'), ('0',))]:
        if tuple(lhs) in lhs_seen:
            continue
        bv, _ = judge_bang(lhs, rhs, rng)
        if bv == 'PROMOTE':
            report['promoted_bang'].append((lhs, rhs, 'seed', ''))
            kept.append((lhs, rhs))

    # ground tier: skeleton semantics, mpmath
    for lhs, rhs in tiers['ground']:
        v, info = judge_ground(list(lhs), list(rhs), rng)
        if v == 'PASS':
            kept.append((lhs, rhs))
        elif v == 'KILL':
            report['killed_ground'].append((lhs, rhs, str(info)[:160]))
        elif v == 'UNSUPPORTED':
            report['unsupported'].append((lhs, rhs, str(info)[:120]))
        else:
            report['eval_err'].append((lhs, rhs, v, str(info)[:120]))

    # STRUCTURAL spike-flattener refusal over EVERYTHING kept (all tiers): the powsin class
    # dies here even when no atom happens to sit on its spike (moving-coincidence insurance).
    still = []
    for lhs, rhs in kept:
        w = spike_flattener_witness(lhs, rhs, core)
        if w is None:
            still.append((lhs, rhs))
        else:
            report['refused_spike'].append((lhs, rhs, str(w)[:120]))
    kept = still

    return kept, report
