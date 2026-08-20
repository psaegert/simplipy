"""The in-suite corpus gate: numeric equivalence of input vs simplified output.

Port of the scale harness (the scale gate, the 64k/1M campaign instrument) to the
tracked 400-row corpus, so the pytest suite itself can catch a numerically unsound
engine change; before this wall the suite had no corpus walk at all -- both the
structural and the numeric corpus gates lived in scripts that only ran when someone
launched a campaign. The
independent per-rule authority remains simplipy.verify (verify_ruleset); this wall is
the end-to-end realization check: the deployed engine, the deployed serialization, the
deployed numeric realizations.

Two-tier, like the campaign gates: a 64-point f64 screen over every <constant>-free row,
then every flagged row is adjudicated at contract precision by simplipy.verify's
judge_pair. A flag the judge cannot certify sound FAILS the suite -- fail-closed, no
orphaned flags.

Two instrument corrections dated 2026-07-29 (the campaign harness moved in lockstep,
see the scale gate):
  * agreement is sign/kind-aware: both-NaN or isclose (numpy isclose certifies
    same-sign infinity pairs and rejects +inf vs -inf and NaN vs inf). The previous
    both-nonfinite clause hid every top-level pole-sign flip and every
    indeterminate-vs-inf mismatch.
  * a row flags at >= 1 disagreeing point. The previous >= 1/5 bar left rows with
    1-12/64 disagreeing points entirely unadjudicated; at 64k/1M scale that band --
    together with the sign-blind clause -- concealed 18 rows that initially
    adjudicated VIOLATION at HEAD, none visible to the old instrument. Per-row deep
    adjudication decomposed all 18 into 14 judge artifacts (two _monitor defects,
    since fixed: correlated all-equal draws counted as extension measure;
    cancellation residues fed to domain-edged/root-amplifying ops), 3 F64_FOLD
    (the engine's f64 constant folds, now a named non-failing verdict) and 1
    contract-tolerated null-set extension -- no contract violations at either
    scale. The screen only triages; the judge decides.

The screen's sensitivity is PINNED as tests rather than silently assumed
(TestScreenSensitivity):
  * a wrong rewrite with finite disagreement must both flag and convict -- the
    anti-vacuity check that the screen and the judge actually see this corpus grammar;
  * a top-level signed-infinity disagreement must now FLAG (sign/kind-aware clause)
    and convict under the judge's measure clause -- before the correction this class
    passed the screen and only the constructor-licence wall covered it;
  * a bounded wrapper (tanh) around the same class stays flagged (finite +/-1
    disagreement), the form in which the 1M campaign originally saw it.

Complexity is asserted as corpus-TOTAL non-inflation plus a dated ratchet pin.
e.complexity measures under the BARE (certificate-less, fail-closed) canonicalization,
so two spellings of one state that differ by a certificate-licensed move measure
differently: 14/400 rows measure UP per-row because the licensed respelling the chain
emits (e.g. a distributed (a*b)^-1) is one the bare context cannot reproduce from the
input -- the odd-negative pow-distribution licence is refused without the zero-set
certificate -- so each spelling keeps its own bare measure. Per-row monotonicity holds
for the chain's internal certified metric, which that boundary hides from this
measurement.

Two known sound-flag families double as the walk's liveness canaries
(test_expected_flag_set pins the exact flag set, so wiring rot in the numeric tier --
a screen that stops seeing disagreements, an inverted flag predicate -- fails loudly
instead of passing vacuously):
  * row 20 (43/64 points): a structural zero (x13 - x13) in a denominator evaluates
    under numpy's SIGNED zeros, flipping the pole sign pointwise vs the contract's one
    unsigned zero; the judge works at contract semantics and certifies the engine's
    inf-spelling fold.
  * row 263 (7/64 points): an exp-argument respelling (exp(inv((1/3)y)) -> exp(3/y))
    with f64 wobble past rtol=1e-5 where the exponent is large; judge-certified sound.
    This row sat inside the old 20% dead band -- it is the in-suite witness that the
    band is now adjudicated rather than ignored.

The 135 masked (`<constant>`-bearing) rows are OUTSIDE the numeric tier -- and not only
for f64-evaluability parity with the harness: the contract judge binds every
`<constant>` occurrence to one shared witness and assumes verbatim pass-through, while
corpus simplification routinely TRANSFORMS constants (absorption, merges), so a direct
judge pass over the masked rows returns 80/135 spurious convictions (measured
2026-07-29). Masked-row numeric coverage needs the literal-provenance carve-out
tracked on the monitor-polish ledger; until then their coverage here is structural.
"""
import json
import os
import re

import numpy as np
import pytest

import simplipy
from simplipy.engine import SimpliPyEngine
from simplipy.verify._monitor import judge_pair
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()
CORPUS = os.path.join(REPO, 'benchmarks', 'corpus', 'raw_skeletons_nv.json')

# Commutative-operand swap for the permutation check (the campaign harnesses' swapper).
# The arity tables span the live 0.12 vocabulary AND the retired hyper-ops so the swap
# stays total on legacy-era spellings.
ARITY1 = {"abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh",
          "div2", "div3", "div4", "div5", "exp", "inv", "log", "mult2", "mult3",
          "mult4", "mult5", "neg", "pow1_2", "pow1_3", "pow1_4", "pow1_5", "pow2",
          "pow3", "pow4", "pow5", "sin", "sinh", "tan", "tanh"}
ARITY2 = {"*", "+", "-", "/", "pow", "rootn"}


def swap_commutative(tokens: list[str], i: int = 0) -> tuple[list[str], int]:
    t = tokens[i]
    arity = 2 if t in ARITY2 else (1 if t in ARITY1 else 0)
    parts = []
    j = i + 1
    for _ in range(arity):
        p, j = swap_commutative(tokens, j)
        parts.append(p)
    if t in {"+", "*"}:
        parts.reverse()
    out = [t]
    for p in parts:
        out.extend(p)
    return out, j


def evaluate(e: SimpliPyEngine, tokens: list[str], X: np.ndarray) -> np.ndarray:
    code = e.prefix_to_infix(tokens, realization=True)
    names = {f'x{i}': X[:, i] for i in range(X.shape[1])}
    with np.errstate(all='ignore'):
        return np.asarray(eval(code, {'simplipy': simplipy, 'np': np}, names),
                          dtype=float) + np.zeros(X.shape[0])


def screen_disagreements(e: SimpliPyEngine, src: list[str], dst: list[str], X: np.ndarray) -> int:
    a = evaluate(e, src, X)
    b = evaluate(e, dst, X)
    # sign/kind-aware agreement: both-NaN, or isclose (True for same-sign inf pairs,
    # False for +inf vs -inf and NaN vs inf)
    agree = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-8)
    return int((~agree).sum())


def flags(n_disagree: int) -> bool:
    # every disagreeing row goes to the judge -- the screen triages, the judge decides
    return n_disagree >= 1


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


@pytest.fixture(scope='module')
def X():
    return np.random.default_rng(0).uniform(-3, 3, size=(64, 32))


@pytest.fixture(scope='module')
def walk(eng, X):
    """One pass over the corpus; every test below asserts a different invariant of it."""
    from conftest import require_or_skip
    require_or_skip(CORPUS, 'tracked corpus not present')
    corpus = json.load(open(CORPUS))
    r = {'n': len(corpus), 'errors': [], 'idem_rows': [], 'perm_rows': [],
         'complexity_in': 0, 'complexity_out': 0, 'screened': 0, 'flagged': []}
    for i, sk in enumerate(corpus):
        sk = list(sk)
        try:
            out = eng.simplify(sk)
            if eng.simplify(list(out)) != out:
                r['idem_rows'].append(i)
            swapped, consumed = swap_commutative(sk)
            if consumed != len(sk):
                raise ValueError(f'swap arity tables stale: consumed {consumed}/{len(sk)}')
            if eng.simplify(swapped) != out:
                r['perm_rows'].append(i)
            explicit = eng.simplify(sk)
            r['complexity_in'] += eng.complexity(sk)
            r['complexity_out'] += eng.complexity(explicit)
            if '<constant>' not in sk and '<constant>' not in explicit:
                r['screened'] += 1
                num = screen_disagreements(eng, sk, explicit, X)
                if flags(num):
                    r['flagged'].append((i, num, sk, explicit))
        except Exception as ex:  # noqa: BLE001 -- every raise is a gate failure, collected
            r['errors'].append((i, f'{type(ex).__name__}: {ex}'))
    return r


class TestCorpusGate:
    def test_no_row_errors(self, walk):
        assert not walk['errors'], walk['errors'][:5]

    def test_idempotent(self, walk):
        assert not walk['idem_rows'], f'idempotence failures at rows {walk["idem_rows"]}'

    def test_permutation_invariant(self, walk):
        assert not walk['perm_rows'], f'permutation failures at rows {walk["perm_rows"]}'

    def test_screen_coverage_exact(self, walk):
        # anti-vacuity, pinned exactly: 400 rows minus the 135 whose INPUT carries
        # <constant> (measured 2026-07-29 on the tracked corpus). Engine-side
        # <constant> leakage into the explicit output of a constant-free input would
        # shrink this count and must fail by count, not hide inside a loose floor.
        assert walk['screened'] == 265, f'{walk["screened"]}/{walk["n"]} rows screened'

    def test_total_complexity_never_inflates(self, walk):
        # the self-describing bound: outputs never total above inputs
        assert walk['complexity_out'] <= walk['complexity_in'], (
            walk['complexity_in'], walk['complexity_out'])
        # EXACT pin, tied to the instrument (audit Tier-2, 2026-08-03): this is the SAME
        # number as the artifact gate REFS['acj-4-3']['complexity'] -- measured in
        # both the native and the explicit projection (complexity is computed on the
        # canonical state, form-independent). The old `<= 97911` ratchet had accumulated
        # two pin generations of slack (97911 mu-ship -> 97623 -> 96443), a silent
        # 1468-point regression window; drift in EITHER direction now fails, exactly as
        # in the gate. Re-pin BOTH constants together, with the evidence recorded in
        # the research harness 96443 -> 96683 at H-020 (2026-08-04): the sign-fold into
        # Const-bearing sums moved 48 corpus rows (family-equal re-spellings) and the
        # divisor-side sign amendment re-spelled 2 more; every moved row audited
        # against the pre-fold walk (see gate_acj REFS comment). 96,687,890 ->
        # 295,677,890 at H-054 (2026-08-06): `mu_free` derived from the worst case
        # (contract Sec 10.10(5)), which is exactly the 198 `<constant>` tokens in
        # the walk times the 1005-bit rise -- no expression re-spelled, idem/perm/
        # translation/twins all unmoved.
        # 295,646,776 -> 295,599,928 at F51 (2026-08-07): the parity arm. FIRST canon
        # change in this arc to move a corpus, and it moves DOWN (-46,848). Every moved row
        # in the 64k/1M corpora audited; see the gate_acj REFS comment.
        # 295,599,928 -> 295,148,608 at (b), 2026-08-08: the mu-argmin orientation
        # ('mirrors equal; bigger scores bigger'); -451,320 milli-bits over 400 rows.
        # 295,148,608 -> 294,961,873 at F63 (2026-08-08, owner-ruled FULL sign-owner
        # family + tie convention A): the product-level sign-placement owner (Add /
        # Pow-odd / rootn-odd / odd-Fun trade sites, materialize-and-price, ties to
        # the positive coefficient), canonical negative joins in term_join, the
        # odd-negative pow pre-fold (pole-faithful confluence), and even-carrier
        # argument orientation. -194,735 milli-bits over 400 rows total (final census
        # at the landed build): 79 rows moved (41 down, 33 equal-mu respells, 4 up --
        # rows 3/199/256/310, the pole-orientation
        # class, e.g. row 310's (1-x4) whose value class the one-zero contract
        # forbids conflating with the cheaper mirror, plus lone-neg odd-fun ties).
        # 294,961,873 -> 294,953,873 same day: the CARRIER-aware placement (the sign
        # toggle rides a bare infinity / the Const refit / an absorbing sum, per
        # H-014/H-020/H-030, instead of minting an illegal raw -1 -- found as 9/50
        # idem failures at 64k/1M, all closed) lets one more row rest cheaper.
        # 294,953,873 -> 294,945,873 at F68 (2026-08-09): the kept-zero bag joins the
        # sign owner (`Mul[0, ..]` assembled through sign_place with the zero as a
        # sign-eating carrier -- 0*X == 0*(-X) in every case -- instead of freezing
        # the arrival mirror, which had made the direct build and the parse of its
        # own rendering two states: fuzz rows 392777/647852, introduced by F63).
        # ONE row moved, row 209 (0 * rootn(tan(u) - x3, 3)): the orbit files the
        # mirror with the sign traded through the odd rootn/tan chain, -8,000
        # milli-bits; value-equal by the kept-zero licence, re-screened in-walk.
        # F80 E1 (2026-08-11): 294,945,873 -> 294,875,873 (-70,000, pure descent).
        # NINE rows moved, all Const-bearing (the E1 fingerprint; counts preserved):
        # 5 judge-OK; 4 shared-witness convictions adjudicated as sound Const-family
        # moves with EXHIBITED witness maps (79: -c; 244: -c; 274: 1/c; 301: 1/c,
        # maxrel 2e-15 on 311 deployed-evaluator draws); row 317 +16,000 is the
        # H-015 greedy-endpoint phenomenon, judge-OK. Audit F80 BUILD 1 close.
        # E2 (2026-08-11, audit F81): 294,875,873 -> 294,730,680 (-145,193). The
        # denominator-clearing certificate licenses odd-negative distributions on
        # analytic factors, so division towers restructure. NINETEEN rows moved
        # (5, 54, 63, 75, 128, 130, 149, 191, 201, 242, 257, 268, 288, 328, 330,
        # 337, 349, 372, 388); ALL adjudicated OK against the pre-E2 worktree
        # baseline on 4,800 deployed-evaluator draws each (6 param sets x 800):
        # zero NaN-vs-value, zero inf-sign, zero value mismatches, worst rel
        # 6.5e-13; NaN domains matched where present; the four Const rows (54,
        # 63, 337, 372) preserve occurrence counts. Row 288 +8,000 is the H-015
        # greedy-endpoint phenomenon again; five rows respell at +-0.
        # B2 mu' (2026-08-17): 294,730,680 -> 84,650,486. The MEASURE changed (D38:
        # two-codeword literal pricing + c_free' 1133 -> 67 bits), so the old pin is
        # incommensurable, not comparable row-by-row; re-earned by running this walk
        # under the mu' build. The drop is dominated by <constant> repricing (135
        # Const-bearing rows at -1,066,000 mB each); made-bigger rows under native
        # mu' descent: ZERO (the D38 3.79%-class is empty by construction, checked
        # per-row in tests/test_mu_prime.py::TestNativeDescentNeverWorse).
        # 84,650,486 -> 84,698,486 (+48,000) at the INVERSE-PAIR BAND (2026-08-19,
        # owner-ratified). The four unguarded inverse-pair arms now carry a magnitude
        # band in f64 mode, so one corpus row stops collapsing an `asinh(sinh(.))` the
        # deployed evaluator cannot reproduce (sinh overflows at 710.475860) and keeps
        # its larger form. LOSSY output is UNMOVED, 0 of 400 rows.
        # RE-PINNED for the 0.14.0 triple: 84,698,486 -> 84,714,486. TWO corpus rows,
        # +8,000 each, and TWO rules -- an earlier version of this comment named only
        # the first and attributed the whole +16,000 to it.
        #
        #   cos(asin(sin x)) -> |cos x|     true on R; f64 disagrees
        #   exp(x/0)         -> inf^x       true on R; at x = 0 f64 gives exp(nan) = nan
        #                                   against inf^0 = 1
        #
        # Both are true and NOT f64-realised, so both serve `real` and no longer the
        # default mode, and f64's own constructor reaches a form that costs 8,000 more
        # in each case. This is the measured price of the mode split, not a regression.
        assert walk['complexity_out'] == 84714486, walk['complexity_out']
        gate_src = os.path.join(REPO, 'remine', 'gate_acj.py')
        if os.path.exists(gate_src):  # absent in an sdist; present in every checkout
            m = re.search(r'"acj-4-3":\s*{[^}]*"complexity":\s*(\d+)', open(gate_src).read())
            assert m and int(m.group(1)) == 84714486, \
                'gate_acj REFS complexity pin drifted from the in-suite pin: re-pin BOTH'

    def test_expected_flag_set(self, walk):
        # the walk's liveness canaries: row 20 (structural-zero inf fold, 43/64 points)
        # and row 263 (exp-argument respell, 7/64 points -- the old dead band's in-suite
        # witness) flag deterministically under the pinned corpus/config/X. An empty or
        # shrunken flag set here means the numeric tier stopped seeing disagreements --
        # vacuity, not health.
        assert [(i, n) for (i, n, *_) in walk['flagged']] == [(20, 43), (263, 7)], \
            walk['flagged']

    def test_flagged_rows_adjudicate_sound(self, walk):
        # every f64 flag is adjudicated at contract precision; anything the judge cannot
        # certify sound (VIOLATION or UNSCORED alike) fails -- no orphaned flags
        bad = []
        for i, num, sk, explicit in walk['flagged']:
            v, d = judge_pair(list(sk), list(explicit), np.random.default_rng(0))
            if v != 'OK':
                bad.append((i, num, v, d))
        assert not bad, bad


class TestScreenSensitivity:
    def test_wrong_rewrite_flags_and_convicts(self, eng, X):
        # the anti-vacuity pin: a finitely-wrong rewrite in the live bag grammar must
        # trip BOTH tiers, or the wall is decorative
        src, dst = '* x0 + x1 x2'.split(), '+ x1 x2'.split()
        assert flags(screen_disagreements(eng, src, dst, X))
        v, _ = judge_pair(list(src), list(dst), np.random.default_rng(0))
        assert v == 'VIOLATION'

    def test_bounded_wrapper_keeps_signed_inf_screenable(self, eng, X):
        # the 1M campaign's 2-ppm class in the form the OLD screen first saw it: tanh
        # maps the +inf/-inf disagreement on the fat zero set to a finite +/-1
        # disagreement -- still flagged under the sharpened clause
        src = 'tanh / 1 - abs x0 x0'.split()
        dst = 'tanh neg / 1 - x0 abs x0'.split()
        assert flags(screen_disagreements(eng, src, dst, X))
        v, _ = judge_pair(list(src), list(dst), np.random.default_rng(0))
        assert v == 'VIOLATION'

    def test_top_level_signed_inf_flags_directly(self, eng, X):
        # the 2026-07-29 correction's own pin: at the TOP level a pole-sign flip used
        # to hide inside the both-nonfinite clause (this exact pair was pinned as
        # screen-BLIND before); the sign/kind-aware clause must now flag it, and the
        # judge convicts under the measure clause
        src = '/ 1 - abs x0 x0'.split()
        dst = 'neg / 1 - x0 abs x0'.split()
        assert flags(screen_disagreements(eng, src, dst, X))
        v, _ = judge_pair(list(src), list(dst), np.random.default_rng(0))
        assert v == 'VIOLATION'


class TestB4IntervalParityAtRuntime:
    """B4: the interval layer's truncating-remainder parity test, seen from `simplify()`.

    `interval.rs` dropped the `-INF` pole candidate for every odd NEGATIVE exponent
    (`k % 2 == 1` is false when Rust's `%` truncates: `(-1) % 2 == -1`), so the box for
    `pow(x, -1)` over a range straddling zero came back REFLECTED -- an under-approximation,
    the direction `rust/lib.rs:764` calls fatal. `finite_ae` then certified expressions that
    are nan on positive measure, and the cancellation consumers fired on them.

    The audit filed this as a mining-gate defect. It is not: it is a wrong `simplify()`
    answer in SOUND mode, reachable through the public API. The literal below is 40 digits
    so it exceeds i128 and stays an opaque leaf -- 39 digits folds to an integer exponent
    and never reaches the branch."""

    @staticmethod
    def _nan_bearing():
        # A = log(x0**(-1.0) + 1): nan across all of (-1, 0), finite elsewhere
        return ['log', '+', 'pow', 'x0', '-1.' + '0' * 40, '1']

    def test_a_minus_a_does_not_cancel_when_a_is_nan_on_positive_measure(self, eng):
        engine = eng
        a = self._nan_bearing()
        assert engine.simplify(['-'] + a + a) != ['0']

    def test_zero_times_a_does_not_absorb_when_a_is_nan_on_positive_measure(self, eng):
        engine = eng
        assert engine.simplify(['*', '0'] + self._nan_bearing()) != ['0']

    def test_the_finite_control_still_cancels(self, eng):
        """Recall control: the fix must refuse the nan-bearing case ONLY. `log(x0**2 + 1)`
        is finite everywhere, so its self-difference must still reach 0."""
        engine = eng
        b = ['log', '+', 'pow', 'x0', '2', '1']
        assert engine.simplify(['-'] + b + b) == ['0']


class TestB1NanFabrication:
    """B1 (numsem-1, ruling D8): SOUND `simplify` fabricated `nan` -- and in one family
    a definite `inf` -- for grounds whose contract value is finite, and the fabrication
    escaped into enclosing variable-bearing expressions. Two mechanisms, both fixed:

    * the interval magnitude-step arms (`pow(t, +-inf)`) merged the attained value 1
      only for an EXACT +-1 point base, so a BRACKETED base (H-019: `cos(np.pi)`
      encloses `[-1, -0.9999999999999991]`) asserted a definite infinity where the
      contract value is 1 -- the 46-row fabricated-Inf family (D8(iii)), an
      under-approximation in the direction the module calls fatal;
    * the ground-class fold trusted a `Class::Nan` verdict on composites containing a
      `pow` with a certified-negative base whose exponent enclosure cannot rule out an
      integer or an infinity -- the continuum convention asserts a.e.-nan there, but
      the exponent's true value can be exactly the integer/infinity where pow is
      defined (`(-1)^(3/sin(np.pi))`: judge value 1; 54 sweep rows).

    Refusal counts as the fix (D8(i)): the engine may leave the ground symbolic; it
    must never assert a value class the contract contradicts."""

    def test_nan_no_longer_escapes_into_a_sum(self, eng):
        # the plan's witness: contract value x0 + 1; shipped float("nan") pre-B1
        out = eng.simplify(['+', 'x0', 'pow', '(-1)', '/', '3', 'sin', 'np.pi'])
        assert list(out) != ['float("nan")']

    def test_mixed_exponent_family_refuses(self, eng):
        # exponent attains an infinity component: judge value 1; shipped nan pre-B1
        out = eng.simplify(['pow', '-1', '/', 'float("inf")', 'sin', 'np.pi'])
        assert list(out) != ['float("nan")']

    def test_bracketed_base_no_longer_fabricates_a_definite_infinity(self, eng):
        # D8(iii) falsifier: cos(np.pi) is exactly -1; pow(-1, -inf) = 1 (magnitude
        # step at |t| = 1, ratified). The bracketed enclosure dips below 1 and the
        # step arm asserted +inf -- and it ANNIHILATED x0 through the sum.
        out = eng.simplify(['pow', 'cos', 'np.pi', 'float("-inf")'])
        assert list(out) != ['float("inf")']
        escape = eng.simplify(['+', 'x0', 'pow', 'cos', 'np.pi', 'float("-inf")'])
        assert list(escape) != ['float("inf")'], "the fabricated infinity ate x0"

    def test_honest_nan_still_folds(self, eng):
        # recall control: a genuinely-nan ground keeps folding -- sqrt of a certified
        # negative is nan for EVERY value, no pow-exponent uncertainty involved
        assert list(eng.simplify(['pow', '-2', '0.5'])) == ['float("nan")']
