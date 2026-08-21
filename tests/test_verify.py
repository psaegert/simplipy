"""Regression wall for the verify package (E1): the judges must SPEAK THE ENGINE'S LANGUAGE.

The 0.12 cutover changed the operator vocabulary (hyper-ops deleted, rootn added), the
boundary output format (n-ary bag tags), and the rule sorts ($-certified slots) -- and
because nothing in pytest imported simplipy.verify, both judges rotted silently: the
monitor judged every bag-tagged output 'OK' through a bare-except probe skip (fail-open),
and the gate crashed on rootn rules while its own poison self-test failed via silently
emptied DEP_OPS slots. Each test here pins one of those failure modes shut:

1. the monitor judge understands the full bag grammar (<add>/<sub>, <mul>/<div>,
   rational coefficients) and CONVICTS a wrong rewrite spelled in it,
2. unknown tokens and unevaluable pairs fail CLOSED (UNSCORED, never OK),
3. rootn is judged with the engine's five-surface semantics on both judges,
4. the gate + contract poison self-test passes (legacy AND live vocabulary,
   including the $-sort),
5. the shipping artifact's rootn/$ rules get real verdicts (no crashes, no
   NO-WITNESS-by-SyntaxError),
6. tiny exists-witnesses survive fitting (the absolute-snap / zero-freeze regression
   that fabricated clause-(a) kills of sound constant-space rules),
7. the monitor's poison self-test runs ROUTINELY (E5): every production poison is
   caught on a fast corpus and a sound rule is NOT -- the positive control that
   previously existed only for manual runs, now pinned into the wall.
"""
import json

from simplipy import SimpliPyEngine
import math
import os

import numpy as np
import pytest

import simplipy.verify as v
from simplipy.verify._contract import judge_rule
from simplipy.verify._monitor import evaluate, judge_pair
from conftest import acj_config_path, acj_rules_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACJ_RULES = acj_rules_path()
ACJ_CONFIG = acj_config_path()


def _staged_or_skip(path):
    """Asset gate, via the shared conftest helper (the E4-3 doctrine: a missing
    asset under SIMPLIPY_TEST_REQUIRE_ASSETS is a FAILURE, never a silent skip)."""
    from conftest import require_or_skip
    require_or_skip(path, f'{path} not staged')


def rng():
    return np.random.default_rng(0)


class TestMonitorSpeaksBagLanguage:
    def test_bag_evaluator_sections(self):
        from mpmath import mp, mpf
        from simplipy.verify._monitor import DPS
        mp.dps = DPS  # direct-internal evaluator call: the caller owns the precision (C35)
        env = {'x0': mpf(3), 'x1': mpf(5), 'x2': mpf(7)}
        cases = [
            (['<add>', 'x0', 'x1', '</add>'], mpf(8)),
            (['<add>', 'x0', '<sub>', 'x1', 'x2', '</add>'], mpf(-9)),
            (['<mul>', 'x0', '<add>', 'x1', 'x2', '</add>', '</mul>'], mpf(36)),
            (['<mul>', '2', 'x1', '<div>', 'x2', 'x0', '</mul>'], mpf(10) / 21),
            (['<mul>', '1/3', 'x0', '</mul>'], mpf(1)),
            (['<add>', '<sub>', 'x0', '</add>'], mpf(-3)),
            (['<mul>', '<div>', 'x0', 'x1', '</mul>'], mpf(1) / 15),
        ]
        for toks, want in cases:
            assert abs(evaluate(list(toks), env) - want) < mpf('1e-40'), toks

    def test_wrong_bag_rewrite_is_violation(self):
        # THE E1 regression: this exact pair was judged OK before the fix
        verdict, _ = judge_pair(['+', 'x0', 'x1'], ['<mul>', 'x0', 'x1', '</mul>'], rng())
        assert verdict == 'VIOLATION'

    def test_correct_bag_rewrite_is_ok(self):
        verdict, _ = judge_pair(['+', 'x0', 'x1'], ['<add>', 'x0', 'x1', '</add>'], rng())
        assert verdict == 'OK'

    def test_div_section_rewrite_is_ok(self):
        verdict, _ = judge_pair(
            ['/', 'rootn', 'x0', '2', 'rootn', 'x0', '2'],
            ['<mul>', 'pow', 'x0', '0.5', '<div>', 'pow', 'x0', '0.5', '</mul>'], rng())
        assert verdict == 'OK'


class TestMonitorFailsClosed:
    def test_unknown_token_is_unscored(self):
        verdict, detail = judge_pair(['+', 'x0', 'x1'], ['frobnicate', 'x0'], rng())
        assert verdict == 'UNSCORED'
        assert 'frobnicate' in detail

    def test_legacy_hyperop_spellings_remain_judgeable(self):
        # the monitor's mpmath table deliberately KEEPS the deleted 0.11 vocabulary so
        # legacy-engine rewrites stay judgeable -- and convicts this wrong one
        verdict, _ = judge_pair(['pow1_2', 'x0'], ['x0'], rng())
        assert verdict == 'VIOLATION'

    def test_nothing_evaluable_is_unscored_not_ok(self):
        # atanh at a written 1 is Unresolved at every probe under the precision-honesty
        # band: the honest verdict is UNSCORED -- the pre-fix code fell through to OK
        verdict, _ = judge_pair(['atanh', '1'], ['float("inf")'], rng())
        assert verdict == 'UNSCORED'


class TestRootnSemantics:
    def test_monitor_judges_rootn(self):
        assert judge_pair(['rootn', 'pow', 'x0', '3', '3'], ['x0'], rng())[0] == 'OK'
        assert judge_pair(['rootn', 'pow', 'x0', '2', '2'], ['x0'], rng())[0] == 'VIOLATION'

    def test_gate_judges_rootn(self):
        assert judge_rule(['rootn', 'exp', '_0', '(-1)'],
                          ['exp', 'neg', '_0'])['verdict'] == 'CERTIFIED'
        assert judge_rule(['pow', 'rootn', '!0', '2', '2'], ['!0'])['verdict'] == 'KILL'

    def test_vacuous_index_rules_are_killed(self):
        # the artifact-defect class E1's repaired gate caught: a non-integer rootn index
        # is an invalid operation, so the LHS is NaN a.e. and the rewrite is a
        # positive-measure extension (the miner's generic-equivalence policy accepted it)
        r = judge_rule(['rootn', '(-1)', 'tanh', '?0'], ['(-1)'])
        assert r['verdict'] == 'KILL'
        assert r['clause'] == 'bc-positive-measure'


class TestGateSelftest:
    def test_gate_and_contract_selftest(self):
        assert v.selftest() is True


class TestDollarSort:
    def test_dollar_slots_are_judged(self):
        assert judge_rule(['/', '$0', '$0'], ['1'])['verdict'] == 'TOLERATED'
        assert judge_rule(['/', '$0', '$0'], ['0'])['verdict'] == 'KILL'


class TestTinyWitnesses:
    def test_tiny_witness_survives_fitting(self):
        # pow(exp(c), pi) at c = -5 is 1.5e-7: the old absolute integer-snap flattened
        # the fitted witness to 0 and the zero-freeze in mp_polish kept it, killing the
        # (sound) rule with a fabricated clause-(a) real change
        r = judge_rule(['pow', 'exp', '<constant>', 'np.pi'], ['<constant>'])
        assert r['verdict'] == 'CERTIFIED', r

    def test_f64_saturated_witness_survives(self):
        # acos(tanh(cosh(4))) = 2.2e-12 reads exactly 0.0 in f64 (tanh saturation);
        # the mp side must re-derive the true witness, the deployed side keeps its own
        r = judge_rule(['acos', 'tanh', 'cosh', '<constant>'], ['<constant>'])
        assert r['verdict'] == 'CERTIFIED', r


class TestMonitorPoisonSelftest:
    """E5: the monitor's poison self-test is the proof its sweep CAN convict -- but
    nothing routine ever ran it (selftest defaults OFF; no pytest or CI step invoked
    it), so the positive control existed only for manual runs. This is the fast
    subset: the FULL production poison list against a rules-empty clean engine on a
    small corpus (~1 s; the catches are driven by the deterministic adversarial rows,
    measured identical across seeds and corpus sizes), plus a NEGATIVE control: a
    sound rule presented as poison must NOT be 'caught' -- a selftest that cannot
    fail would launder a broken sweep as verified."""

    @pytest.fixture(scope='class')
    def setup(self):
        _staged_or_skip(ACJ_CONFIG)
        from simplipy.verify._monitor import build_engine, make_corpus
        baseline = build_engine([], ACJ_CONFIG)
        corpus = make_corpus(baseline, 120, np.random.default_rng(3))
        return baseline, corpus

    def test_all_production_poisons_caught(self, setup):
        from simplipy.verify._monitor import selftest
        baseline, corpus = setup
        assert selftest([], ACJ_CONFIG, corpus, baseline, seed=3) is True

    def test_sound_rule_is_not_laundered_as_caught(self, setup):
        from simplipy.verify._monitor import selftest
        baseline, corpus = setup
        sound = [(('+', '?0', '0'), ('?0',))]
        assert selftest([], ACJ_CONFIG, corpus, baseline, seed=3, poison=sound) is False


@pytest.mark.integration
class TestShippingArtifact:
    def test_artifact_rootn_and_dollar_rules_judgeable(self):
        _staged_or_skip(ACJ_RULES)
        rules = json.load(open(ACJ_RULES))
        census = {'CERTIFIED': 0, 'TOLERATED': 0}
        tolerated = set()
        for lhs, rhs in rules:
            toks = list(lhs) + list(rhs)
            if 'rootn' not in toks and not any(t.startswith('$') for t in toks):
                continue
            r = v.verify_rule(lhs, rhs)
            # every such rule gets a REAL verdict -- never a crash and never the
            # NO-WITNESS-by-judge-error that unjudgeable vocabulary produced. KILL is
            # NOT admitted (audit Tier-2, 2026-08-03): a shipped rule the judge
            # convicts is an alarm, not a tolerated outcome -- the old three-outcome
            # form would have waved it through.
            assert r['verdict'] in census, (lhs, rhs, r)
            census[r['verdict']] += 1
            if r['verdict'] == 'TOLERATED':
                tolerated.add((tuple(lhs), tuple(rhs)))
        # measured census of the shipped artifact (2026-08-07, acj-4-3 @ 7,147 rules;
        # was 68/4 at 1008 rules on 2026-08-03, then 4330/4 at 7,153). An artifact
        # refresh re-pins this alongside the gate_acj REFS, with evidence. The 4330 ->
        # 4325 step is B1(c): `certainly_nonneg` gained its `rootn` arm, so the five
        # `abs rootn _0 {2,4,6,8,10}` rules moved into the constructors and the re-mine
        # no longer mints them. (A sixth rule dropped, `sin acos tanh _0`, carries
        # neither `rootn` nor `$` and so was never in this subset.)
        # 4325 -> 4322 is B1(d): `rootn(e,n) -> exp(1/n)` took the three
        # `log rootn np.e {2,4,8}` rules native as well (all three carry `rootn`).
        # 4322 -> 4301 is F49 (`compose_e_power` + `exp(1) -> E`, artifact @ 7,006).
        # Single cause and fully enumerated: of the 140 rules the re-mine dropped, exactly
        # 21 carry `rootn` or `$`, and all 21 are `rootn <k> exp 1 -> nan` for
        # k in -10..10. Their index was the COMPOSITE `exp(1)`, which the constructor left
        # symbolic; folded to the LEAF `e` it meets the provably-non-integer-index arm and
        # the NaN is derived natively (`rootn 2 exp 1 -> nan` on a rules-free engine).
        # 4301 -> 4258 at F53 (2026-08-07, the reciprocal-base arms, artifact @ 6,710).
        # Single cause and fully enumerated: of the 90 rules that re-mine dropped, exactly 43
        # carry `rootn` or `$`, and all 43 are the `acosh rootn k (-m) -> nan` family for
        # k in {2,10} and m in -10..-2 -- a NEGATIVE root index, which the constructor now
        # reciprocates into a literal base and folds.
        # 4258 -> 4237 at F55 (2026-08-08, a ground source may DENOTE a special by value;
        # artifact @ 6,669). Single cause and fully enumerated, and it is F49's own shape a
        # second time: of the 55 rules the re-mine dropped, exactly 21 carry `rootn` or `$`,
        # and all 21 are `rootn <k> acos (-1) -> nan` for k in -10..10. Their INDEX was the
        # composite `acos (-1)`, left symbolic; once the bare `acos (-1) -> np.pi` rule folds
        # it to the LEAF `np.pi` the index meets the provably-non-integer-index arm and the NaN
        # is derived (`rootn 2 np.pi -> nan` holds on a rules-free engine; `rootn 2 acos (-1)`
        # does NOT, because unlike F49's `exp(1) -> E` the fold here is a rule, not a
        # constructor arm). None of the 14 ADDED rules carries `rootn` or `$`, so the excused
        # set below is untouched.
        # RE-PINNED for the 0.14.0 triple: 4237/4 -> 3815/0, on the f64 file (5,319 rules
        # against the 0.13 line's 6,594). Two causes, both intended. The judge is
        # stricter -- the contract tolerance became purely RELATIVE, so rules that
        # survived on an absolute 1e-25 floor no longer do -- and the file is now the
        # f64 THIRD of a triple, so every rule that is true but not f64-realised has
        # moved to rules_real.json. TOLERATED reaching 0 follows from the same split:
        # a tolerated-but-unrealised rule is `real`-tier and is not in this file.
        assert census == {'CERTIFIED': 3815, 'TOLERATED': 0}, census
        # ...and WHICH rules are excused, not merely how many. The count alone would
        # wave through a swap (a $-rule turning CERTIFIED while some rootn rule turns
        # TOLERATED); this pins the excused set itself. All four are the one documented
        # R3 class -- clause (b), undefined->defined on the null set {0} only -- and are
        # the $-sort spellings of the judge's own '/ ?0 ?0 -> 1' and '/ 0 mult2 ?0 -> 0'
        # touchstones. A NEW member here is a contract question, never a re-pin.
        # ...and WHICH rules are excused. All four moved to rules_real.json with the
        # triple: they are TOLERATED (true off the null set {0}) and NOT f64-realised
        # (f64 answers nan at 0 and at both infinities), which is precisely the `real`
        # tier. So the f64 file excuses NOTHING, and the four are asserted where they
        # now live. A new member in either place is a contract question, never a re-pin.
        # (Implied by `census['TOLERATED'] == 0` above, so it guards nothing on its own;
        # kept as the statement of intent, with the real check being the membership
        # assertions below -- WHICH rules are excused, not how many.)
        assert tolerated == set(), sorted(tolerated)
        real_rules = json.load(open(ACJ_RULES.replace('rules.json', 'rules_real.json')))
        real_keys = {(tuple(a), tuple(b)) for a, b in real_rules}
        assert (('/', '$0', '$0'), ('1',)) in real_keys      # x/x -> 1
        assert (('/', '0', '$0'), ('0',)) in real_keys       # 0/x -> 0
        # The two MULTIPLICATIVE spellings are AC-duplicates of `x/x -> 1` and the 0.14
        # mine dedups them, so the SPELLINGS are gone and the BEHAVIOUR is not. Asserting
        # spellings here would pin the mine's dedup rather than the contract question,
        # which is whether the excused rewrite still happens.
        from simplipy import Mode
        engine = SimpliPyEngine.from_config(ACJ_RULES.replace('rules.json', 'config.yaml'))
        for tokens in (['*', 'x0', 'inv', 'x0'], ['*', 'inv', 'x0', 'x0'],
                       ['/', 'x0', 'x0']):
            assert engine.simplify(list(tokens), mode=Mode.real) == ['1'], tokens
        assert engine.simplify(['/', '0', 'x0'], mode=Mode.real) == ['0']


class TestJudgeParityExactness:
    """H-050 (2026-08-05, extreme-literal lane): the judge reads an exponent's
    integrality/parity off the mpf REPRESENTATION (odd-normalized mantissa: exp<0
    non-integer, exp==0 odd, exp>0 even) instead of refusing everything beyond the
    old PARITY_CAP=1e15 -- which convicted sound engine folds whose exponents are
    beyond-2^53 integer literals (extreme-lane smoke rows 32/301/796, all judged
    VIOLATION pre-fix, all OK post-fix)."""

    def test_pow_parity_exact_beyond_2_53(self):
        from mpmath import mp, mpf
        from simplipy.verify._monitor import DPS, c_pow
        mp.dps = DPS  # direct-internal call (C35)
        assert c_pow(mpf('-inf'), mpf('9007199254740993')) == mpf('-inf')  # odd
        assert c_pow(mpf('-inf'), mpf('9007199254740992')) == mpf('inf')   # even
        assert c_pow(mpf('-inf'), mpf('1e40')) == mpf('inf')               # even
        assert c_pow(mpf(-1), mpf('10000000000000000001')) == mpf(-1)      # odd

    def test_beyond_precision_magnitudes_stay_unresolvable(self):
        from mpmath import isnan, mpf
        from simplipy.verify._monitor import c_pow
        # BEYOND the precision budget the mpf is a rounding and representation-
        # integrality says nothing (exp(2.703^5) is provably irrational yet its
        # 166-bit mpf reads integral -- 1M row 294377): honest nan, both bases.
        assert isnan(c_pow(mpf('-inf'), mpf(10) ** 400))
        assert isnan(c_pow(mpf(-2), mpf(10) ** 400))
        from mpmath import exp as mpexp
        assert isnan(c_pow(mpf(-1), mpexp(mpf('2.703') ** 5)))
        # non-integer spellings keep the honest nan
        assert isnan(c_pow(mpf('-inf'), mpf('2.5')))

    def test_rootn_index_value_strict(self):
        from mpmath import isnan, mp, mpf
        from simplipy.verify._monitor import DPS, c_rootn
        mp.dps = DPS  # direct-internal call (C35)
        r = c_rootn(mpf(2), mpf('9007199254740993'))  # huge odd index resolves
        assert abs(r - 1) < mpf('1e-14') and r > 1
        assert isnan(c_rootn(mpf(2), mpf(10) ** 400))  # bignum-hazard index refuses

    def test_convicted_smoke_rows_judge_ok(self):
        from simplipy.verify._monitor import judge_pair
        cases = [
            ("* + 92233720368547758/7 x0 + x1 pow * x4 x2 9007199254740993",
             "* + x0 13176245766935394 + x1 * pow x2 9007199254740993 pow x4 9007199254740993"),
            ("- pow -1 10000000000000000001 exp x0", "neg + exp x0 1"),
            ("rootn pow pow * x4 x0 9007199254740993 1/3 2",
             "pow * pow x0 9007199254740993 pow x4 9007199254740993 / 1 6"),
        ]
        for inp, out in cases:
            verd, _ = judge_pair(inp.split(), out.split(), np.random.default_rng(0))
            assert verd == "OK", (inp, verd)


class TestRuleFilesAreDataNotCode:
    """R2/B21: `verify_ruleset` adjudicates rule sets it did NOT produce -- so a leaf
    token is untrusted DATA. It used to reach `eval(t, {'np': np, 'float': float})`,
    whose globals dict omits __builtins__, so CPython injects the real ones and a token
    becomes an arbitrary Python expression. Demonstrated end to end before the fix: a
    rules.json whose lhs token was an __import__ payload WROTE ITS MARKER FILE through
    the documented public entry point, and was then judged as if it were a rule."""

    def test_a_rules_file_cannot_execute_code(self, tmp_path):
        marker = tmp_path / 'PWNED'
        payload = f'__import__("pathlib").Path("{marker}").write_text("pwned")'
        path = tmp_path / 'hostile_rules.json'
        path.write_text(json.dumps([[[payload], ['0']]]))
        report = v.verify_ruleset(str(path))
        assert not marker.exists(), 'a rules.json token executed as Python'
        assert report['buckets']['UNSUPPORTED-SHAPE'] == [0], report['buckets']
        for bucket in ('CERTIFIED', 'TOLERATED', 'KILL'):
            assert report['buckets'][bucket] == [], f'payload reached the {bucket} path'

    def test_every_shipped_literal_spelling_is_accepted(self):
        """The acceptor must not refuse the corpus it exists to gate: `(-N)` alone is
        26.98% of literal occurrences in the shipped artifact (3,981 of 6,803 rules)."""
        from fractions import Fraction

        from simplipy.verify._contract import literal_value
        assert literal_value('(-10)') == Fraction(-10)
        assert literal_value('.5') == Fraction(1, 2) and literal_value('2.125') == Fraction(17, 8)
        assert literal_value('np.pi') == math.pi and literal_value('np.e') == math.e
        assert literal_value('float("inf")') == math.inf
        assert literal_value("float('-inf')") == -math.inf
        assert math.isnan(literal_value('float("nan")'))
        assert literal_value('1/3') == Fraction(1, 3)  # legal in a foreign ruleset

        # F97: a decimal literal denotes the EXACT rational it spells, not the nearest
        # double. Asserted where the two differ, because that difference is the defect:
        # a mis-rendered literal is wrong by the same 5.5e-17 at every precision rung, so
        # the gap does not decay and the judge reads it as an analytic difference. It
        # convicted the exact identity `inv(-10/t) -> -0.1*t` as a clause-(a) REAL-CHANGE
        # at measure 1.0. The deployed lane still reads the double -- that is
        # what deployment computes -- so only the CONTRACT lane changed.
        assert literal_value('-1e-09') == Fraction(-1, 10 ** 9) != Fraction(-1e-09)
        assert literal_value('-0.1') == Fraction(-1, 10) != Fraction(float('-0.1'))

    @pytest.mark.parametrize('token', [
        '__import__("os").system("true")', 'open("/etc/passwd")', 'np.pi + 1',
        '(-1)*2', 'float("1/3")', 'x0.__class__', '1/0', '()', 'lambda: 1',
        '[1,2]', 'np.__loader__', 'eval("1")',
    ])
    def test_anything_outside_the_grammar_is_refused(self, token):
        from simplipy.verify._contract import UnsupportedToken, literal_value
        with pytest.raises(UnsupportedToken):
            literal_value(token)


class TestD15DiagonalBinding:
    """D15 / X11's opposite-polarity hole: `judge_rule` sampled wildcard slots
    INDEPENDENTLY, so the diagonal binding {_0 = _1} was a null event to it -- yet
    for a rewrite rule the diagonal is a real, engine-reachable instance family:
    with _0 = _1 = log(x0) the collector refuses t - t -> 0 (nan-capable), the
    pattern (a-b)/(b-a) matches, and the engine rewrote a nan-EVERYWHERE expression
    to -1. judge_pair (instance-level) convicts the same pair; the two authorities
    must not be disjointly blind. The judge now scans a diagonal lane."""

    def test_diagonal_nan_rule_is_killed(self):
        from simplipy.verify import verify_rule
        v = verify_rule(['/', '-', '_0', '_1', '-', '_1', '_0'], ['(-1)'])
        assert v['verdict'] == 'KILL', v

    def test_sound_multislot_rules_stay_certified(self):
        from simplipy.verify import verify_rule
        # diagonal-safe multi-slot rules must not be collateral: x*y -> y*x is exact
        # everywhere including the diagonal; (a-b)+(b-a) -> 0 is nan == nan there.
        assert verify_rule(['*', '_0', '_1'], ['*', '_1', '_0'])['verdict'] in ('CERTIFIED', 'TOLERATED')
        assert verify_rule(['+', '-', '_0', '_1', '-', '_1', '_0'], ['0'])['verdict'] != 'KILL'


class TestTheRealisationAxis:
    """F100: the judge reports WHICH AUTHORITY a rule answers to, not just a verdict.

    The two notions of soundness are INCOMPARABLE -- a rule can be true and
    unrealised (`atanh(tanh t) -> t`: exactly t on R, `inf` in f64 past 18.99) or
    realised and untrue (`asin(1e-8) -> 1e-8`: bit-identical in f64, wrong by
    1.7e-17). So this is a second axis, and the tier is the cell it lands in.
    """

    def test_the_four_tiers_are_reachable_and_correctly_assigned(self):
        from simplipy.verify._contract import judge_rule
        cases = [
            # true AND realised
            (['*', '(-1)', 'asin', '_0'], ['asin', 'neg', '_0'], 'core'),
            # true, NOT realised: tanh attains exactly 1.0 from 18.990341103219276,
            # so the deployed engine answers inf where the rule answers the argument
            (['atanh', 'neg', 'tanh', '!0'], ['neg', '!0'], 'real'),
            # realised, NOT true: bit-identical in f64, wrong by 1.7e-17 relatively
            (['asin', 'pow', '10', '(-8)'], ['1e-08'], 'f64'),
            # NEITHER -- constructed, not drawn from the artifact. Which artifact rules
            # land in `reject` depends on the constructor (the inverse-pair band moved
            # two of them into `real`), so an artifact exemplar would pin this test to
            # one combination of branches. `sin t -> t` is false on R and not
            # reproduced by f64, unconditionally.
            (['sin', '_0'], ['_0'], 'reject'),
        ]
        for lhs, rhs, want in cases:
            got = judge_rule(list(lhs), list(rhs))
            assert got['tier'] == want, f'{" ".join(lhs)} -> {" ".join(rhs)}: {got}'

    def test_a_TOLERATED_rule_the_engine_contradicts_is_real_not_f64(self) -> None:
        """`_tier` filed contract-accepted-but-unrealised under `f64`, which is backwards
        in both directions at once: it put the rule in the one mode whose authority
        CONTRADICTS it, and removed it from the mode that honours it.

        `/ $0 $0 -> 1` is true off the null set {0}, which is what TOLERATED means; f64
        answers nan at 0 AND at both infinities, so it is not realised. True and not
        realised is `real`. Six acj-4-3 rules land here, so the arm was live, not
        theoretical -- at the re-mine they would have been written into rules_f64.json."""
        from simplipy.verify._contract import _tier, judge_rule
        got = judge_rule(['/', '$0', '$0'], ['1'])
        assert got['verdict'] == 'TOLERATED'
        assert got['realised'] is False
        assert got['tier'] == 'real'
        # the whole table, so a future edit has to move a documented cell on purpose
        # `None` files as `real` too since the 2026-08-20 ruling: absent deployed
        # evidence cannot support a claim of f64 soundness.
        assert [_tier(v, r) for v in ('CERTIFIED', 'TOLERATED') for r in (True, False, None)] \
            == ['core', 'real', 'real', 'core', 'real', 'real']
        assert (_tier('KILL', True), _tier('KILL', False)) == ('f64', 'reject')
        assert (_tier('ENGINE-MISALIGN', False), _tier('NO-WITNESS', None)) == ('real', 'reject')

    def test_realisation_is_undetermined_across_a_snap_not_asserted(self):
        """A snap means the f64 algebra evaluates a DIFFERENT point, so the deployed
        comparison is skipped rather than answered -- `None`, never a guess. Asserting
        realisation there would convict on a measurement artifact."""
        from simplipy.verify._contract import judge_rule
        got = judge_rule(['sin', 'np.pi'], ['0'])
        assert got['verdict'] == 'CERTIFIED'
        assert got['realised'] is None

    def test_the_axis_is_additive_engine_misalign_keeps_its_gate(self):
        """The realisation accumulator runs unconditionally; ENGINE-MISALIGN keeps its
        original gate (contract certifies AND deployment diverges AND all-real
        bindings). A rule the contract also rejects is a KILL, not a misalignment."""
        from simplipy.verify._contract import judge_rule
        got = judge_rule(['asin', 'pow', '10', '(-8)'], ['1e-08'])
        assert got['verdict'] == 'KILL' and got['realised'] is True


class TestTheRealisationBound:
    """`realised` decides the `f64` tier and therefore what ships in `rules.json`. The
    bar it uses is derived (see `compare_deployed_realised`), and these pin the three
    properties that derivation rests on."""

    def test_the_old_1e_9_floor_would_admit_a_100_percent_error(self) -> None:
        """The defect the audit found. `e**sinh(-5)` is 5.942307292381135e-33 and the
        rule says 0.0 -- two ordinary, different doubles. The retired bound was
        `1e-9 * max(1.0, |a|, |b|)`, an ABSOLUTE 1e-9 for anything below 1, so it read
        that as an f64 equality. Harmless while every KILL was deleted; an admission path
        into the default rule set once the gate began routing on the tier."""
        from simplipy.verify._contract import judge_rule
        got = judge_rule(['pow', 'np.e', 'sinh', '(-5)'], ['0'])
        assert got['verdict'] == 'KILL'
        assert got['realised'] is False
        assert got['tier'] == 'reject'

    def test_rounding_is_realised_but_divergence_is_not(self) -> None:
        """The bound has to separate two things bit-exactness cannot. `1/exp(x)` and
        `exp(-x)` differ by ONE ulp -- libm doing its job, and still what f64 computes.
        `acos(cos(cos x))` and `|cos x|` differ by 2377 ULP at x=300, where cos's
        argument reduction genuinely falls apart."""
        from simplipy.verify._contract import judge_rule
        rounding = judge_rule(['/', '1', 'exp', '_0'], ['exp', 'neg', '_0'])
        assert rounding['realised'] is True and rounding['tier'] == 'core'
        divergent = judge_rule(['acos', 'cos', 'cos', '_0'], ['abs', 'cos', '_0'])
        assert divergent['realised'] is False and divergent['tier'] == 'real'

    def test_the_bound_sits_in_an_empty_region_so_its_value_is_not_load_bearing(self) -> None:
        """WHY 8 AND NOT 4 OR 16. Over the rules bit-exactness moves, the gap distribution
        is 310 at 1 ULP, a thin tail through 7, then NOTHING until 57. Every bound in
        [8, 56] gives the identical partition; 4 splits the rounding cluster in half.

        The probes below are chosen to DISCRIMINATE, which is what makes this evidence
        rather than decoration: two shipped rules whose worst gaps are 6 and 7 ULP tier
        `real` at a bound of 4 and `core` at 8, 16 and 56. An earlier version used probes
        with gaps of 0, 1 and 2377, so its assertion was identical for every bound in
        [1, 2376] and gave no reason to prefer 8 over the 4 that was explicitly rejected.
        """
        from simplipy.verify import _contract as C
        straddling = [(['abs', 'sin', 'acos', '_0'], ['cos', 'asin', '_0']),   # 6 ULP
                      (['-', 'np.pi', 'acos', '_0'], ['acos', 'neg', '_0'])]   # 7 ULP
        anchors = [(['/', '1', 'exp', '_0'], ['exp', 'neg', '_0']),            # 1 ULP
                   (['acos', 'cos', 'cos', '_0'], ['abs', 'cos', '_0']),       # 2377 ULP
                   (['pow', 'np.e', 'sinh', '(-5)'], ['0'])]                   # gross
        original = C.REALISED_ULP
        try:
            # the REJECTED bound splits the rounding cluster: both straddlers lose f64
            C.REALISED_ULP = 4
            assert [C.judge_rule(list(a), list(b))['tier'] for a, b in straddling] \
                == ['real', 'real']
            # ... and every bound in the empty region agrees with 8, on all five probes
            baseline = None
            for bar in (8, 16, 32, 56):
                C.REALISED_ULP = bar
                tiers = [C.judge_rule(list(a), list(b))['tier']
                         for a, b in straddling + anchors]
                assert tiers[:2] == ['core', 'core'], (bar, tiers)
                if baseline is None:
                    baseline = tiers
                assert tiers == baseline, (bar, tiers, baseline)
        finally:
            C.REALISED_ULP = original
        assert C.REALISED_ULP == 8


class TestCleanlinessIsPerMode:
    """Before the triple there was one rule set and one contract, so "clean" could mean
    "no fatal bucket". That is now wrong in both directions, and R3 ("0.14.0 is sound,
    full stop") makes the per-mode gate load-bearing rather than optional."""

    CORE = [['+', 'x0', '0'], ['x0']]
    REAL = [['atanh', 'tanh', 'x0'], ['x0']]
    REJECT = [['sin', 'x0'], ['x0']]

    def test_the_same_rule_is_clean_for_one_file_and_dirty_for_another(self) -> None:
        """`atanh(tanh t) -> t` is true on R and not f64-realised. In rules_real.json it
        is exactly what belongs there; in rules.json it would be a rewrite the deployed
        evaluator contradicts. One rule, two answers -- which a bucket count cannot say."""
        from simplipy.verify import verify_ruleset
        assert verify_ruleset([self.REAL], mode='real')['is_clean'] is True
        assert verify_ruleset([self.REAL], mode='corpus')['is_clean'] is True
        assert verify_ruleset([self.REAL], mode='f64')['is_clean'] is False

    def test_no_mode_may_carry_a_reject(self) -> None:
        from simplipy.verify import verify_ruleset
        for mode in ('f64', 'real', 'corpus'):
            rep = verify_ruleset([self.CORE, self.REJECT], mode=mode)
            assert rep['is_clean'] is False, mode
            assert [o['tier'] for o in rep['offenders']] == ['reject'], mode

    def test_the_legacy_meaning_survives_for_pre_triple_callers(self) -> None:
        """`mode=None` keeps "only CERTIFIED/TOLERATED are clean", because that is what
        every caller written before the triple means by the word."""
        from simplipy.verify import verify_ruleset
        assert verify_ruleset([self.CORE])['is_clean'] is True
        assert verify_ruleset([self.REAL])['is_clean'] is False

    def test_an_unknown_mode_raises_rather_than_silently_gating_nothing(self) -> None:
        from simplipy.verify import verify_ruleset
        with pytest.raises(ValueError, match='unknown mode'):
            verify_ruleset([self.CORE], mode='banana')

    def test_a_triple_is_verified_as_ONE_artifact(self) -> None:
        """Three per-file sweeps, and NOT the retired set identity. `corpus == f64 UNION
        real` was dropped once folding became mode-dependent: each file omits what its own
        constructor performs, so set overlap measures spelling. On the shipped artifact the
        union exceeds the corpus file by 362 rules and corpus performs every one -- the
        identity would call a correct artifact dirty."""
        from simplipy.verify import verify_triple
        good = verify_triple([self.CORE], [self.CORE, self.REAL], [self.CORE, self.REAL])
        assert good['is_clean'] is True and good['relationships'] == []

        # a corpus file SMALLER than the union is now legitimate, not a defect
        pruned = verify_triple([self.CORE], [self.CORE, self.REAL], [self.CORE])
        assert pruned['is_clean'] is True, pruned['relationships']
        assert pruned['relationships'] == []
        # a genuine defect is still caught, per file
        dirty = verify_triple([self.CORE, self.REJECT], [self.CORE], [self.CORE])
        assert dirty['is_clean'] is False
        assert dirty['modes']['f64']['is_clean'] is False

    def test_the_shipped_triple_passes_its_own_public_gate(self) -> None:
        """The regression that made this necessary: `verify_triple` kept enforcing the
        retired union identity after the miner stopped, so the documented artifact gate
        reported the 0.14.0 artifact as dirty while every per-file sweep was clean."""
        from conftest import acj_config_path, require_or_skip
        from simplipy.verify import verify_triple
        require_or_skip(acj_config_path(), 'needs the shipped acj-4-3 triple')
        base = acj_config_path().replace('config.yaml', '')
        report = verify_triple(base + 'rules.json', base + 'rules_real.json',
                               base + 'rules_corpus.json',
                               engine_config=acj_config_path())
        assert report['is_clean'] is True, report['relationships']
        assert all(r['is_clean'] for r in report['modes'].values())
        assert report['corpus_dominance'] == []


class TestOperandScaledPrecision:
    """Rung 2 was a fixed dps 120, which confirms nothing when the intermediates are
    10^217: both rungs are swamped alike, agree on a manufactured verdict, and the class
    comparison reads that agreement as evidence."""

    def test_a_true_identity_swamped_by_cancellation_is_no_longer_killed(self) -> None:
        """`log(cosh(25t) + sinh(25t)) = log(e^{25t}) = 25t` for every real t -- exactly
        true, no domain holes. It was KILLed at bc-positive-measure (measure 0.1198)
        because the cancellation drives the sum to a computed ZERO, so the left side read
        `log(0) = -inf` at BOTH rungs. It is `real`: true on R, and not f64-realised
        because exp overflows at 709.782713."""
        from simplipy.verify._contract import judge_rule
        got = judge_rule('log + cosh mult5 mult5 ?0 sinh mult5 mult5 ?0'.split(),
                         'mult5 mult5 ?0'.split())
        assert got['verdict'] != 'KILL'
        assert got['tier'] == 'real'

    def test_the_shipped_identities_are_no_longer_near_the_bar(self) -> None:
        """Three EXACT identities on the shipped artifact sat at 6 of 167 grid points
        against a kill bar of 9 -- three points from being wrongly convicted at the next
        mine. `cos(asin(tanh t)) = sech t` and siblings."""
        from simplipy.verify._contract import MEASURE_KILL, judge_rule
        for lhs, rhs in ((['cos', 'asin', 'tanh', '_0'], ['inv', 'cosh', '_0']),
                         (['sin', 'acos', 'tanh', '!0'], ['inv', 'cosh', '!0'])):
            got = judge_rule(list(lhs), list(rhs))
            assert got['verdict'] != 'KILL', lhs
            assert float(got['measure']) < MEASURE_KILL / 3, (lhs, got['measure'])

    def test_the_precision_is_derived_from_the_operands_actually_seen(self) -> None:
        """Not a formula guessed from the expression: `c_eval` reports the largest finite
        intermediate it produced, and the requirement follows from it. Adding two numbers
        of magnitude 10^k whose true sum is 10^-k destroys about 2k significant digits."""
        from mpmath import mpf
        from simplipy.verify._contract import BASE_DPS, MAX_DPS, _required_dps
        assert _required_dps(mpf(1)) == BASE_DPS
        assert _required_dps(None) == BASE_DPS
        assert _required_dps(mpf('1e217')) == BASE_DPS + 2 * 218
        # and it is CAPPED: past the ceiling the point is Unresolved, never convicted
        assert _required_dps(mpf('1e100000')) == MAX_DPS
