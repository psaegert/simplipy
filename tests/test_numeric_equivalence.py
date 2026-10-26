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

Two instrument corrections (the campaign harness moved in lockstep,
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

A known sound-flag family doubles as the walk's liveness canary
(test_expected_flag_set pins the exact flag set, so wiring rot in the numeric tier --
a screen that stops seeing disagreements, an inverted flag predicate -- fails loudly
instead of passing vacuously). Row 20, a structural zero (x13 - x13) in a denominator,
flagged until f64 kept the sign IEEE gives that zero: its answer now evaluates as numpy's
signed zeros do.
  * row 263 (7/64 points): an exp-argument respelling (exp(inv((1/3)y)) -> exp(3/y))
    with f64 wobble past rtol=1e-5 where the exponent is large; judge-certified sound.
    This row sat inside the old 20% dead band -- it is the in-suite witness that the
    band is now adjudicated rather than ignored.

The 135 masked (`<constant>`-bearing) rows are OUTSIDE the numeric tier -- and not only
for f64-evaluability parity with the harness: the contract judge binds every
`<constant>` occurrence to one shared witness and assumes verbatim pass-through, while
corpus simplification routinely TRANSFORMS constants (absorption, merges), so a direct
judge pass over the masked rows returns 80/135 spurious convictions (measured).
Masked-row numeric coverage needs the literal-provenance carve-out
not yet built; until then their coverage here is structural.
"""
import numpy as np
import pytest

import simplipy
from simplipy.verify import default_corpus
from simplipy.engine import SimpliPyEngine
from simplipy.verify._monitor import judge_pair
from conftest import acj_config_path

CONFIG = acj_config_path()

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
    corpus = default_corpus()
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

    def test_step_cap_never_binds(self, eng, walk):
        # The walk ran the deployed default effort on every row. A binding step cap would
        # leave a non-fixpoint answer with no other signal.
        assert eng._core.step_cap_binds() == 0

    def test_screen_coverage_exact(self, walk):
        # anti-vacuity, pinned exactly: 400 rows minus the 135 whose INPUT carries
        # <constant> (measured on the tracked corpus). Engine-side
        # <constant> leakage into the explicit output of a constant-free input would
        # shrink this count and must fail by count, not hide inside a loose floor.
        assert walk['screened'] == 265, f'{walk["screened"]}/{walk["n"]} rows screened'

    def test_total_complexity_never_inflates(self, walk):
        # the self-describing bound: outputs never total above inputs
        assert walk['complexity_out'] <= walk['complexity_in'], (
            walk['complexity_in'], walk['complexity_out'])
        # EXACT pin, tied to the instrument: this is the SAME
        # number as the artifact gate REFS['acj-4-3']['complexity'] -- measured in
        # both the native and the explicit projection (complexity is computed on the
        # canonical state, form-independent). The old `<= 97911` ratchet had accumulated
        # two pin generations of slack (97911 mu-ship -> 97623 -> 96443), a silent
        # 1468-point regression window; drift in EITHER direction now fails, exactly as
        # in the gate. Re-pin BOTH constants together, with the evidence recorded.
        # 96443 -> 96683: the sign-fold into
        # Const-bearing sums moved 48 corpus rows (family-equal re-spellings) and the
        # divisor-side sign amendment re-spelled 2 more; every moved row audited
        # against the pre-fold walk (see gate_acj REFS comment). 96,687,890 ->
        # 295,677,890: `mu_free` derived from the worst case
        # (the sup-construction), which is exactly the 198 `<constant>` tokens in
        # the walk times the 1005-bit rise -- no expression re-spelled, idem/perm/
        # translation/twins all unmoved.
        # 295,646,776 -> 295,599,928: the parity arm. FIRST canon
        # change in this arc to move a corpus, and it moves DOWN (-46,848). Every moved row
        # in the 64k/1M corpora audited; see the gate_acj REFS comment.
        # 295,599,928 -> 295,148,608: the mu-argmin orientation
        # ('mirrors equal; bigger scores bigger'); -451,320 milli-bits over 400 rows.
        # 295,148,608 -> 294,961,873 (FULL sign-owner
        # family + tie convention A): the product-level sign-placement owner (Add /
        # Pow-odd / rootn-odd / odd-Fun trade sites, materialize-and-price, ties to
        # the positive coefficient), canonical negative joins in term_join, the
        # odd-negative pow pre-fold (pole-faithful confluence), and even-carrier
        # argument orientation. -194,735 milli-bits over 400 rows total (final census
        # at the landed build): 79 rows moved (41 down, 33 equal-mu respells, 4 up --
        # rows 3/199/256/310, the pole-orientation
        # class, e.g. row 310's (1-x4) whose value class the one-zero contract
        # forbids conflating with the cheaper mirror, plus lone-neg odd-fun ties).
        # 294,961,873 -> 294,953,873: the CARRIER-aware placement (the sign
        # toggle rides a bare infinity / the Const refit / an absorbing sum, per
        # the bare-inf, Const and absorbing arms, instead of minting an illegal raw -1 -- found as 9/50
        # idem failures at 64k/1M, all closed) lets one more row rest cheaper.
        # 294,953,873 -> 294,945,873: the kept-zero bag joins the
        # sign owner (`Mul[0, ..]` assembled through sign_place with the zero as a
        # sign-eating carrier -- 0*X == 0*(-X) in every case -- instead of freezing
        # the arrival mirror, which had made the direct build and the parse of its
        # own rendering two states: fuzz rows 392777/647852).
        # ONE row moved, row 209 (0 * rootn(tan(u) - x3, 3)): the orbit files the
        # mirror with the sign traded through the odd rootn/tan chain, -8,000
        # milli-bits; value-equal by the kept-zero licence, re-screened in-walk.
        # Generic-parameter licence: 294,945,873 -> 294,875,873 (-70,000, pure descent).
        # NINE rows moved, all Const-bearing (counts preserved):
        # 5 judge-OK; 4 shared-witness convictions adjudicated as sound Const-family
        # moves with EXHIBITED witness maps (79: -c; 244: -c; 274: 1/c; 301: 1/c,
        # maxrel 2e-15 on 311 deployed-evaluator draws); row 317 +16,000 is the
        # lossy greedy-endpoint phenomenon, judge-OK.
        # 294,875,873 -> 294,730,680 (-145,193). The
        # denominator-clearing certificate licenses odd-negative distributions on
        # analytic factors, so division towers restructure. NINETEEN rows moved
        # (5, 54, 63, 75, 128, 130, 149, 191, 201, 242, 257, 268, 288, 328, 330,
        # 337, 349, 372, 388); ALL adjudicated OK against the previous
        # baseline on 4,800 deployed-evaluator draws each (6 param sets x 800):
        # zero NaN-vs-value, zero inf-sign, zero value mismatches, worst rel
        # 6.5e-13; NaN domains matched where present; the four Const rows (54,
        # 63, 337, 372) preserve occurrence counts. Row 288 +8,000 is the lossy
        # greedy-endpoint phenomenon again; five rows respell at +-0.
        # mu': 294,730,680 -> 84,650,486. The MEASURE changed (the
        # two-codeword literal pricing + c_free' 1133 -> 67 bits), so the old pin is
        # incommensurable, not comparable row-by-row; re-earned by running this walk
        # under the mu' build. The drop is dominated by <constant> repricing (135
        # Const-bearing rows at -1,066,000 mB each); made-bigger rows under native
        # mu' descent: ZERO (the made-bigger class is empty by construction, checked
        # per-row in tests/test_mu_prime.py::TestNativeDescentNeverWorse).
        # 84,650,486 -> 84,698,486 (+48,000) at the INVERSE-PAIR BAND.
        # The four unguarded inverse-pair arms now carry a magnitude
        # band in f64 mode, so one corpus row stops collapsing an `asinh(sinh(.))` the
        # deployed evaluator cannot reproduce (sinh overflows at 710.475860) and keeps
        # its larger form. LOSSY output is UNMOVED, 0 of 400 rows.
        # RE-PINNED for the 0.14.0 triple: 84,698,486 -> 84,714,486. TWO corpus rows,
        # +8,000 each, and TWO rules:
        #
        #   cos(asin(sin x)) -> |cos x|     true on R; f64 disagrees
        #   exp(x/0)         -> inf^x       true on R; at x = 0 f64 gives exp(nan) = nan
        #                                   against inf^0 = 1
        #
        # Both are true and NOT f64-realised, so both serve `real` and no longer the
        # default mode, and f64's own constructor reaches a form that costs 8,000 more
        # in each case. This is the measured price of the mode split, not a regression.
        # THE SYMBOL TABLE: 84,714,486 -> 56,665,486. The MEASURE changed
        # -- structure is no longer a flat 8 bits a node -- so the old pin is not
        # comparable and this is a re-pin, not a movement. What is comparable, and
        # measured beside it: sound (`f64`) and `real` outputs are BYTE-IDENTICAL on all
        # 400 rows across the change, and 14 corpus-mode rows re-spell through the
        # mu-driven reciprocal-rejoin projection, which is the one output decision the
        # prices own.
        # 56,665,486 -> 56,521,606 at the ONE-BIT LITERAL FLOOR. Also
        # a measure change and also a re-pin, and this one DOES move outputs: four f64
        # rows and six corpus rows re-spell, and the corpus rests cheaper by 3,000
        # milli-bits. 0 of 5,451 shipped rules change direction; 0 of 105 printed literal
        # spellings move.
        # THE acj-4 TRIPLE: same measure, new artifact -- a real movement,
        # not a re-pin: 56,521,606 -> 56,508,606 (the refreshed mine rests the corpus
        # 13,000 milli-bits cheaper; translation pristine 5347/0/0).
        # THE SOUNDNESS ROUND: 56,508,606 -> 56,310,982. Three
        # movements land together: the cleared-witness certificate gains its
        # denominator-finiteness side condition, the instruments price the
        # chain's own parse route (the re-pin half), and the ruled
        # DEFAULT_EFFORT=4 explores every walk row (a real movement: outputs rest
        # 197,624 milli-bits cheaper). The inflation bound held throughout.
        # THE UNCAPPED SEARCH (0.15.0, DEFAULT_EFFORT=None): the walk does not move.
        # DIVISOR PRODUCTS AND CONSTANT SUMS: 56,310,982 -> 56,312,982. A
        # divisor that is a product gives the enclosing product its members' inverses, each
        # keeping its pole as the product of the inverses does: rows 130 and 288 keep a
        # divisor sum's spelled orientation (+3,000 each). And a Const-bearing sum carries a
        # product's sign into its other terms instead of eating it (`(x2 - x1)(x1 - c)` shipped
        # as `(x1 + c)(x1 - x2)`): the five `<constant>` rows whose old answers let a sign
        # vanish so now keep it: 340 and 395 (-6,000 each), 111 (0), 172 (+6,000), 200
        # (+2,000).
        # INVERTED QUOTIENTS: 56,312,982 -> 56,307,982. A negative power is zero
        # only where its base is infinite, so the inverse of a quotient distributes as the
        # quotient's members do: rows 279 (-2,000) and 399 (-3,000).
        # THE FACTOR STAGE: 56,307,982 -> 56,106,168. After the expansion moves, the search
        # takes the shared factor out of a sum (`x1/(x1*x2 - x1**2)` is `1/(x2 - x1)`), with
        # the common-factor rules also serving bare terms; it starts from the expansion
        # answer and accepts only states below it, so no walk row rests dearer.
        # FACTORING UNDER AN INVERSE TAKES DISTRIBUTION'S LICENCE: 56,106,168 -> 56,107,168. A
        # sum under a negative power is factored only where the power would distribute over the
        # factored product; one walk row keeps its sum (+1,000).
        # READING A AT POLES: 56,107,168 -> 56,053,168. A negative odd power's pole is
        # two-signed, so its orientation is the product's choice in every mode (a one-signed
        # pole keeps its spelling); a repeated wildcard binds a class up to sign, and FACTOR
        # takes a factor the terms share up to sign. The walk rests 54,000 milli-bits cheaper.
        # THE SEARCH OWNS THE DISTRIBUTIVE LAW: 56,053,168 -> 55,815,451. FACTOR takes what
        # some terms of a sum share out of exactly those terms, DISTRIBUTE also works on one
        # factor or one term of a sum factor, and every candidate is tried cheapest rebuilt state
        # first in one search: 50 walk rows move, every one to a cheaper answer.
        # ONE CLASS READING: 55,815,451 -> 55,812,451. An odd function of a negated argument is a
        # class member like one of a sum, so an even power over it files the positive argument:
        # row 397's `sinh(sinh(-x15))^2` is `sinh(sinh(x15))^2` (-3,000).
        # THE SIGN KEY: 55,812,451 -> 55,795,451. Fewest signs, then outermost, on
        # every equal price, and an odd function of any literal multiple trades its sign: 10
        # walk rows move. Eight rest cheaper, the sign traded where it costs least: 32 and 86
        # (-3,000 each), 160 and 233 (-2,000), 180 and 336 (-1,000), 256 (-6,000), 319 (-5,000).
        # Two rest dearer by 3,000, 97 and 274, by the search's path.
        # ONE PRICE FOR A SIGN: 55,795,451 -> 55,127,866. A sign costs one bit
        # wherever it sits (a negation standing alone paid a `Mul` head, a -1 beside other factors
        # nothing), so the measure moved: the previous answers price 55,158,451 under it. 67 walk
        # rows move; 11 rest cheaper (7, 18, 46, 67, 83, 95, 133, 149, 262, 327, 373: -30,585 in
        # all), none dearer, and 56 respell at an equal price, the sign key placing their signs.
        # GROUND NaN FACTS: 55,127,866 -> 54,750,550. A family of `<constant>` values
        # that is NaN at every constant folds to NaN: rows 205 (`acosh(cos(c)/5)`, -136,229) and
        # 343 (`acos(cosh(c))`, finite only at c = 0; -241,087) are NaN. Both used to keep the
        # family read as a free constant, `acosh(c)` and `acos(c)`, which is defined.
        # AN OVERFLOW CUTS NO BAND: 54,750,550 -> 54,718,550. Where sinh overflows
        # the collapse of `asinh(sinh(t))` gives the true value, so f64 collapses it for every
        # t: row 380 rests 32,000 milli-bits cheaper.
        # POWERS OF e COLLECT: 54,718,550 -> 54,715,550. The product constructor merges
        # `exp(a) exp(b)` into `exp(a + b)`: row 7's `exp(x6) exp(-x3)` is `exp(x6 - x3)`
        # (-3,000).
        # A CONSTANT ABSORBS EVERY CLASS'S ORIENTATION SIGN: 54,715,550 -> 54,705,550. Beside a bare
        # `<constant>` a sum holding `<constant>`s of its own takes the orientation with the fewer
        # signs: rows 203 (-2,000), 296 (-1,000), 312 (-4,000), 321 (-2,000) and 364
        # (`<constant>*(<constant> - x7)/x3` is `<constant>*(x7 + <constant>)/x3`, -1,000) rest
        # cheaper, and row 274 takes its tied orientation by its own sum (0).
        # A ZERO KEEPS ITS SIGN WHERE IT IS INVERTED (f64): 54,705,550 -> 54,728,550. The
        # inverse of a product holding an exact zero distributes over its members, the zero
        # giving +inf, as IEEE's signed zero has it: rows 20 (`inf*x14*(x11 - x17)` is
        # `-inf*x14*(x17/x11 - 1)`, +8,000) and 283 (+15,000) keep the members' signs.
        # ONE ORIENTATION WHEREVER A CLASS STANDS: 54,728,550 -> 54,727,550. A site alone takes
        # the orientation it takes as a product's one factor: row 163's odd root of a sum under
        # `cos` takes its sign outside, where `cos` drops it (-1,000), and row 82 respells its
        # `tan(tan(.)^3)` at an equal price.
        # A ZERO SET IS READ ON THE DOMAIN: 54,727,550 -> 54,672,228. The zero-set certificate
        # applies the identity theorem on an expression's domain, so sums over log, roots, acosh,
        # asin and quotients are null there, and inverses split and signs move over them. 15
        # walk rows move: ten rest cheaper (50, 286 and 351 -6,000 each, 68 -1,000, 119
        # -16,737, 156 -4,000, 196 -6,585, 318 -2,000, 362 -7,000, 366 -3,000), four respell
        # at an equal price (65, 87, 160, 239) and row 55 rests dearer (+3,000): its inverse
        # now splits over `acosh(acos(<constant>))`, which takes the constant's inverse.
        # A SUM THAT VANISHES ON A REGION KEEPS ITS ZERO'S SIGN WHERE IT IS INVERTED:
        # 54,672,228 -> 54,657,536. Where a product holding a sum the certificate cannot show
        # null is inverted, f64 keeps the sign with the inverse and distributes it over the
        # members. 10 walk rows move: five rest cheaper (77 -3,000, 135 -1,000, 154 -1,000, 160
        # -8,000, 331 -2,692), four respell at an equal price (176, 325, 329, 336) and row 163
        # rests dearer (+1,000): its odd root under `cos` keeps the orientation it is written in.
        # A ZERO SET IS READ ON A SLICE: 54,657,536 -> 54,615,636. The zero-set certificate reads a
        # sum one variable at a time (the sum moves with it on every slice, by its slope or its
        # derivative's zero set), through a factor every term shares, and past the horizon for
        # an entire function, so more inverses split and more signs move. 11 walk rows move:
        # seven rest cheaper (82 -1,000, 87 -6,000, 117 -3,000, 125 -19,000, 163 -8,900, 290
        # -1,000, 341 -3,000) and four respell at an equal price (55, 113, 160, 297).
        assert walk['complexity_out'] == 54615636, walk['complexity_out']

    def test_expected_flag_set(self, walk):
        # the walk's liveness canary: row 263 (exp-argument respell, 7/64 points -- the old
        # dead band's in-suite witness) flags deterministically under the pinned
        # corpus/config/X. An empty flag set here means the numeric tier stopped seeing
        # disagreements -- vacuity, not health. (Row 20's structural zero no longer flags:
        # f64 keeps the sign IEEE gives it.)
        assert [(i, n) for (i, n, *_) in walk['flagged']] == [(263, 7)], \
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
        # the sign-aware correction's own pin: at the TOP level a pole-sign flip used
        # to hide inside the both-nonfinite clause (this exact pair was pinned as
        # screen-BLIND before); the sign/kind-aware clause must now flag it, and the
        # judge convicts under the measure clause
        src = '/ 1 - abs x0 x0'.split()
        dst = 'neg / 1 - x0 abs x0'.split()
        assert flags(screen_disagreements(eng, src, dst, X))
        v, _ = judge_pair(list(src), list(dst), np.random.default_rng(0))
        assert v == 'VIOLATION'


class TestB4IntervalParityAtRuntime:
    """The interval layer's truncating-remainder parity test, seen from `simplify()`.

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
    """SOUND `simplify` fabricated `nan` -- and in one family
    a definite `inf` -- for grounds whose contract value is finite, and the fabrication
    escaped into enclosing variable-bearing expressions. Two mechanisms, both fixed:

    * the interval magnitude-step arms (`pow(t, +-inf)`) merged the attained value 1
      only for an EXACT +-1 point base, so a BRACKETED base (outward rounding: `cos(np.pi)`
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
        # the witness: contract value x0 + 1; once answered float("nan")
        out = eng.simplify(['+', 'x0', 'pow', '(-1)', '/', '3', 'sin', 'np.pi'])
        assert list(out) != ['float("nan")']

    def test_mixed_exponent_family_refuses(self, eng):
        # exponent attains an infinity component: judge value 1; once answered nan
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
