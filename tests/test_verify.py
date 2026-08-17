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
        assert census == {'CERTIFIED': 4237, 'TOLERATED': 4}, census
        # ...and WHICH rules are excused, not merely how many. The count alone would
        # wave through a swap (a $-rule turning CERTIFIED while some rootn rule turns
        # TOLERATED); this pins the excused set itself. All four are the one documented
        # R3 class -- clause (b), undefined->defined on the null set {0} only -- and are
        # the $-sort spellings of the judge's own '/ ?0 ?0 -> 1' and '/ 0 mult2 ?0 -> 0'
        # touchstones. A NEW member here is a contract question, never a re-pin.
        assert tolerated == {
            (('/', '$0', '$0'), ('1',)),            # x/x -> 1
            (('*', '$0', 'inv', '$0'), ('1',)),     # x*(1/x) -> 1
            (('*', 'inv', '$0', '$0'), ('1',)),     # (1/x)*x -> 1, commuted
            (('/', '0', '$0'), ('0',)),             # 0/x -> 0
        }, sorted(tolerated)


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
        from simplipy.verify._contract import literal_value
        assert literal_value('(-10)') == -10.0
        assert literal_value('-1e-09') == -1e-09
        assert literal_value('.5') == 0.5 and literal_value('2.125') == 2.125
        assert literal_value('np.pi') == math.pi and literal_value('np.e') == math.e
        assert literal_value('float("inf")') == math.inf
        assert literal_value("float('-inf')") == -math.inf
        assert math.isnan(literal_value('float("nan")'))
        assert literal_value('1/3') == 1.0 / 3.0  # legal in a foreign ruleset

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
