"""Regression wall for the verify package: the judges must SPEAK THE ENGINE'S LANGUAGE.

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
7. the monitor's poison self-test runs ROUTINELY: every production poison is
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
from simplipy.verify._monitor import judge_pair
from conftest import PERMISSIVE_DOMINANCE_KNOWN, acj_config_path, acj_rules_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACJ_RULES = acj_rules_path()
ACJ_CONFIG = acj_config_path()


def _staged_or_skip(path):
    """Asset gate, via the shared conftest helper (a missing
    asset under SIMPLIPY_TEST_REQUIRE_ASSETS is a FAILURE, never a silent skip)."""
    from conftest import require_or_skip
    require_or_skip(path, f'{path} not staged')


def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope='module')
def bag_engine():
    """An engine that reads the tagged form (`to_prefix`), as the sweep's engine does."""
    import yaml
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[])


class TestMonitorSpeaksBagLanguage:
    """The monitor reads a tagged output through its engine (`to_prefix`) and judges the
    rewrite as the judge judges a rule."""

    def test_wrong_bag_rewrite_is_violation(self, bag_engine):
        # THE bag regression: this exact pair was once judged OK
        verdict, _ = judge_pair(['+', 'x0', 'x1'], ['<mul>', 'x0', 'x1', '</mul>'], rng(),
                                engine=bag_engine)
        assert verdict == 'VIOLATION'

    def test_correct_bag_rewrite_is_ok(self, bag_engine):
        verdict, _ = judge_pair(['+', 'x0', 'x1'], ['<add>', 'x0', 'x1', '</add>'], rng(),
                                engine=bag_engine)
        assert verdict == 'OK'

    def test_div_section_rewrite_is_ok(self, bag_engine):
        verdict, _ = judge_pair(
            ['/', 'rootn', 'x0', '2', 'rootn', 'x0', '2'],
            ['<mul>', 'pow', 'x0', '0.5', '<div>', 'pow', 'x0', '0.5', '</mul>'], rng(),
            mode='real', engine=bag_engine)
        assert verdict == 'OK'

    def test_a_tagged_side_without_an_engine_is_unscored(self):
        verdict, detail = judge_pair(['+', 'x0', 'x1'], ['<add>', 'x0', 'x1', '</add>'], rng())
        assert verdict == 'UNSCORED' and 'engine' in detail


class TestMonitorFailsClosed:
    def test_unknown_token_is_unscored(self):
        verdict, detail = judge_pair(['+', 'x0', 'x1'], ['frobnicate', 'x0'], rng())
        assert verdict == 'UNSCORED'
        assert 'frobnicate' in detail

    def test_legacy_hyperop_spellings_remain_judgeable(self):
        # the judge's alphabet KEEPS the deleted 0.11 vocabulary, so legacy-engine rewrites
        # stay judgeable -- and convicts this wrong one
        verdict, _ = judge_pair(['pow1_2', 'x0'], ['x0'], rng())
        assert verdict == 'VIOLATION'

    def test_a_ground_rewrite_is_judged_at_its_one_point(self):
        # atanh(1) is +inf in the table, exactly, and in float64: the rewrite holds; a wrong
        # infinity does not
        assert judge_pair(['atanh', '1'], ['float("inf")'], rng())[0] == 'OK'
        assert judge_pair(['atanh', '1'], ['float("-inf")'], rng())[0] == 'VIOLATION'


class TestRootnSemantics:
    def test_monitor_judges_rootn(self):
        assert judge_pair(['rootn', 'pow', 'x0', '3', '3'], ['x0'], rng())[0] == 'OK'
        assert judge_pair(['rootn', 'pow', 'x0', '2', '2'], ['x0'], rng())[0] == 'VIOLATION'

    def test_gate_judges_rootn(self):
        assert judge_rule(['rootn', 'exp', '?0', '(-1)'],
                          ['exp', 'neg', '?0'])['verdict'] == 'CERTIFIED'
        # a `_` slot is owed +-inf: at -inf the left side is rootn(0, -1) = 1/0, which has no
        # value (the limit rule), and the right side is +inf
        wild = judge_rule(['rootn', 'exp', '_0', '(-1)'], ['exp', 'neg', '_0'])
        assert (wild['verdict'], wild['points']) == ('KILL', [{'_0': -math.inf}]), wild
        assert judge_rule(['pow', 'rootn', '!0', '2', '2'], ['!0'])['verdict'] == 'KILL'

    def test_vacuous_index_rules_are_killed(self):
        # the artifact-defect class the repaired gate caught: a non-integer rootn index
        # is an invalid operation, so the LHS is NaN a.e. and the rewrite would invent a
        # function. The interval layer's real reading encloses the expression-position index:
        # tanh stays inside (-1, 1), which holds no nonzero integer, so the left side is
        # undefined everywhere and the extension is measured
        r = judge_rule(['rootn', '(-1)', 'tanh', '?0'], ['(-1)'])
        assert r['verdict'] == 'KILL' and r['measure'] > 0, r
        assert r['tier'] not in v.MODE_TIERS['real']


class TestZeroLanes:
    """The monitor reads a rewrite in its mode's values, through the judge: f64's IEEE 754
    signed zeros (`1/(x*0)` is +-inf by the sign of x), real's one zero, where an inverted
    exact zero has no value (the limit rule), and permissive's real values, with nothing owed
    where the input has none."""

    #: Inputs whose exact zero is inverted, with the IEEE answer and the one-zero `+inf`.
    POLES = [
        ('inv * x1 0', '/ float("inf") x1', 'float("inf")'),
        ('inv neg * 0 x1', '/ float("-inf") x1', 'float("inf")'),
    ]

    @pytest.mark.parametrize('src,ieee,one_zero', POLES)
    def test_each_mode_reads_its_own_values(self, src, ieee, one_zero):
        def verdict(out, mode):
            return judge_pair(src.split(), out.split(), rng(), mode=mode)[0]
        assert verdict(ieee, 'f64') == 'OK' and verdict(one_zero, 'f64') == 'VIOLATION'
        assert verdict('float("nan")', 'real') == 'OK'
        assert verdict(ieee, 'real') == verdict(one_zero, 'real') == 'VIOLATION'
        assert verdict(ieee, 'permissive') == verdict(one_zero, 'permissive') == 'OK'

    def test_an_undecided_judgment_is_unscored(self):
        # the interval layer does not see `x13 - x13` as an exact zero, so the measure of the
        # extension is undecided: neither passed nor convicted
        verdict, detail = judge_pair('/ 1 * x11 - x13 x13'.split(), '/ float("inf") x11'.split(),
                                     rng())
        assert verdict == 'UNSCORED' and 'UNDECIDED-MEASURE' in detail, detail

    def test_the_default_lane_is_the_engines_default_mode(self):
        src, ieee, one_zero = self.POLES[0]
        assert judge_pair(src.split(), ieee.split(), rng()) == \
            judge_pair(src.split(), ieee.split(), rng(), mode='f64')
        assert judge_pair(src.split(), one_zero.split(), rng())[0] == 'VIOLATION'

    def test_f64_reads_the_deployed_kernels_zero_signs(self):
        # the table's f64 values are the deployed kernels', IEEE 754's signed zeros among them
        # (`x**0.5` is a square root there, as in numpy: sqrt(-0) is -0)
        from simplipy.verify._contract import d_eval, parse
        env = {'?1': -2.0, '?2': 3.0}
        for tokens, ieee in [
                ('inv neg - x2 x2', -np.inf),          # -(+0) is -0
                ('inv * x1 - x2 x2', -np.inf),         # (-2)*(+0) is -0
                ('inv / 0 x1', -np.inf),               # 0/(-2) is -0
                ('inv / x2 float("-inf")', -np.inf),   # 3/(-inf) is -0
                ('inv + neg 0 neg 0', -np.inf),        # -0 + -0 is -0
                ('inv - neg 0 0', -np.inf),            # -0 - (+0) is -0
                ('inv - x2 x2', np.inf),               # x - x is +0
                ('inv abs neg 0', np.inf),             # |-0| is +0
                ('pow neg 0 -3', -np.inf), ('pow neg 0 -2', np.inf),
                ('rootn neg 0 -3', -np.inf), ('rootn neg 0 -2', np.inf),
                ('inv pow neg 0 3', -np.inf), ('inv pow neg 0 0.5', -np.inf),
                ('inv rootn float("-inf") -3', -np.inf),
                ('inv sin neg 0', -np.inf), ('inv tan neg 0', -np.inf),
                ('inv atan neg 0', -np.inf), ('inv tanh neg 0', -np.inf),
                ('inv asinh neg 0', -np.inf), ('inv sinh neg 0', -np.inf),
                ('inv asin neg 0', -np.inf), ('inv atanh neg 0', -np.inf),
                ('inv exp float("-inf")', np.inf),     # exp(-inf) is +0
                ('inv -0.0', -np.inf), ('inv -0', np.inf),   # a float spelling keeps its sign
                ('/ 0 0', np.nan)]:
            got = d_eval(parse(tokens.split()), env)
            assert (np.isnan(ieee) and got != got) or got == ieee, (tokens, got)

    def test_a_zero_valued_literal_span_keeps_its_sign(self):
        # `* -1 0` folds to an exact zero, whose sign IEEE reads as -0; real reads one zero,
        # whose inverse has no value
        assert judge_pair(['inv', '*', '-1', '0'], ['float("-inf")'], rng(), mode='f64')[0] == 'OK'
        assert judge_pair(['inv', '*', '-1', '0'], ['float("nan")'], rng(), mode='real')[0] == 'OK'
        assert judge_pair(['inv', '*', '-1', '0'], ['float("inf")'], rng(),
                          mode='real')[0] == 'VIOLATION'

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match='mode'):
            judge_pair(['x1'], ['x1'], rng(), mode='complex')


class TestGateSelftest:
    def test_gate_and_contract_selftest(self):
        assert v.selftest() is True


class TestDollarSort:
    def test_dollar_slots_are_judged(self):
        # undefined only at 0 and +-inf, the values the sort's certificate owes nothing
        assert judge_rule(['/', '$0', '$0'], ['1'])['verdict'] == 'CERTIFIED'
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
    """The monitor's poison self-test is the proof its sweep CAN convict -- but
    nothing routine ever ran it (selftest defaults OFF; no pytest or CI step invoked
    it), so the positive control existed only for manual runs. This is the fast
    subset: the FULL production poison list against a rules-empty clean engine on the
    adversarial rows alone (the catches are driven by them), plus a NEGATIVE control: a
    sound rule presented as poison must NOT be 'caught' -- a selftest that cannot
    fail would launder a broken sweep as verified."""

    @pytest.fixture(scope='class')
    def setup(self):
        _staged_or_skip(ACJ_CONFIG)
        from simplipy.verify._monitor import build_engine, make_corpus
        baseline = build_engine([], ACJ_CONFIG)
        corpus = make_corpus(baseline, 0, np.random.default_rng(3))
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
        census = {'CERTIFIED': 0, 'TOLERATED': 0, 'KILL': 0}
        for lhs, rhs in rules:
            toks = list(lhs) + list(rhs)
            if 'rootn' not in toks and not any(t.startswith('$') for t in toks):
                continue
            r = v.verify_rule(lhs, rhs)
            # every such rule gets a REAL verdict -- never a crash and never the
            # NO-WITNESS-by-judge-error that unjudgeable vocabulary produced. KILL is
            # NOT admitted: a shipped rule the judge
            # convicts is an alarm, not a tolerated outcome -- the old three-outcome
            # form would have waved it through.
            assert r['verdict'] in census, (lhs, rhs, r)
            # a KILL is admitted only as what this f64 file may hold: false over the reals and
            # bit-exact in float64 (below)
            assert r['verdict'] != 'KILL' or r['tier'] == 'f64', (lhs, rhs, r)
            census[r['verdict']] += 1
        # measured census of the shipped artifact. An artifact
        # refresh re-pins this alongside the gate_acj REFS, with evidence. The 4330 ->
        # 4325 step: `certainly_nonneg` gained its `rootn` arm, so the five
        # `abs rootn _0 {2,4,6,8,10}` rules moved into the constructors and the re-mine
        # no longer mints them. (A sixth rule dropped, `sin acos tanh _0`, carries
        # neither `rootn` nor `$` and so was never in this subset.)
        # 4325 -> 4322: `rootn(e,n) -> exp(1/n)` took the three
        # `log rootn np.e {2,4,8}` rules native as well (all three carry `rootn`).
        # 4322 -> 4301: `compose_e_power` + `exp(1) -> E`.
        # Single cause and fully enumerated: of the 140 rules the re-mine dropped, exactly
        # 21 carry `rootn` or `$`, and all 21 are `rootn <k> exp 1 -> nan` for
        # k in -10..10. Their index was the COMPOSITE `exp(1)`, which the constructor left
        # symbolic; folded to the LEAF `e` it meets the provably-non-integer-index arm and
        # the NaN is derived natively (`rootn 2 exp 1 -> nan` on a rules-free engine).
        # 4301 -> 4258: the reciprocal-base arms.
        # Single cause and fully enumerated: of the 90 rules that re-mine dropped, exactly 43
        # carry `rootn` or `$`, and all 43 are the `acosh rootn k (-m) -> nan` family for
        # k in {2,10} and m in -10..-2 -- a NEGATIVE root index, which the constructor now
        # reciprocates into a literal base and folds.
        # 4258 -> 4237: a ground source may DENOTE a special by value.
        # Single cause and fully enumerated, and it is the previous step's shape a
        # second time: of the 55 rules the re-mine dropped, exactly 21 carry `rootn` or `$`,
        # and all 21 are `rootn <k> acos (-1) -> nan` for k in -10..10. Their INDEX was the
        # composite `acos (-1)`, left symbolic; once the bare `acos (-1) -> np.pi` rule folds
        # it to the LEAF `np.pi` the index meets the provably-non-integer-index arm and the NaN
        # is derived (`rootn 2 np.pi -> nan` holds on a rules-free engine; `rootn 2 acos (-1)`
        # does NOT, because unlike `exp(1) -> E` the fold here is a rule, not a
        # constructor arm). None of the 14 ADDED rules carries `rootn` or `$`, so the excused
        # set below is untouched.
        # RE-PINNED for the 0.14.0 triple: 4237/4 -> 3815/0, on the f64 file (5,319 rules
        # against the 0.13 line's 6,594). Two causes, both intended. The judge is
        # stricter -- the contract tolerance became purely RELATIVE, so rules that
        # survived on an absolute 1e-25 floor no longer do -- and the file is now the
        # f64 THIRD of a triple, so every rule that is true but not f64-realised has
        # moved to rules_real.json. TOLERATED reaching 0 follows from the same split:
        # a tolerated-but-unrealised rule is `real`-tier and is not in this file.
        # 3815 -> 3820 at THE acj-4 TRIPLE (f64 5,319 -> 5,338). Fully
        # enumerated, two families: +21 structural rootn admissions (neg-through-odd-root
        # commutation, abs/rootn orderings, rootn->pow(+-0.2) respellings at |index| 5)
        # under the widened judge; -16 `rootn {cosh,exp} <constant> k -> <constant>`
        # constant-binding folds retired by the canonical-source-class certification.
        # Net +5 in this subset, every one CERTIFIED; TOLERATED stays 0.
        # 3820/0 -> 3817/3 when the judge read THE TABLE (0.15): a root with a negative odd
        # index at a zero radicand is 1/0, which has no value under the limit rule, where the
        # rewrite gives pow(0, -0.2) = +inf -- an extension at one point (`abs rootn _0 (-5)
        # -> pow abs _0 -0.2` at 0, the `acos`/`acosh` siblings at 1). Fully enumerated.
        # 3817/3 -> 3817/0/3 under the owner's guard ruling: that one point is a null set no
        # `_` slot guards (bound to a subexpression that is 0 on an interval the extension
        # has positive measure), so the three are KILLs -- `f64`-tier, since float64 reads
        # both sides as +inf there, and so still this file's to hold.
        assert census == {'CERTIFIED': 3817, 'TOLERATED': 0, 'KILL': 3}, census
        # The f64 file excuses NOTHING: the four excused rules of the one documented null-set
        # class (clause (b), undefined->defined on the null set {0} only) are TOLERATED
        # and not f64-realised, which is the `real` tier, so they live in rules_real.json
        # and are asserted there by the next test. A new excused rule in either place is
        # a contract question, never a re-pin.

    def test_the_excused_rules_live_in_the_real_file(self):
        """The other half of the split, kept as its own test so that shipping the f64
        set WITHOUT a triple reports honestly: the census above still runs and passes,
        and this one skips for want of a subject rather than dragging the pair down."""
        from conftest import require_triple
        require_triple()
        real_path = ACJ_RULES.replace('rules.json', 'rules_real.json')
        real_rules = json.load(open(real_path))
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
    """The table reads an exponent's integrality and parity exactly, beyond 2^53 (a written
    integer is the integer it spells), so sound engine folds whose exponents are beyond-2^53
    integer literals are not convicted (extreme-lane smoke rows 32/301/796); past its
    precision budget it refuses rather than guess."""

    @staticmethod
    def _value(tokens):
        from mpmath import mp
        from simplipy.verify._contract import c_eval, parse
        with mp.workdps(50):
            return c_eval(parse(tokens.split()), {})

    def test_pow_parity_exact_beyond_2_53(self):
        from mpmath import mpf
        assert self._value('pow float("-inf") 9007199254740993') == mpf('-inf')   # odd
        assert self._value('pow float("-inf") 9007199254740992') == mpf('inf')    # even
        assert self._value('pow float("-inf") 1e40') == mpf('inf')                # even
        assert self._value('pow -1 10000000000000000001') == -1                   # odd

    def test_beyond_precision_magnitudes_stay_unresolvable(self):
        from mpmath import isnan
        from simplipy.verify._contract import Unresolved
        for tokens in ('pow float("-inf") 1e400', 'pow -2 1e400', 'pow -1 exp 214.6'):
            with pytest.raises(Unresolved):
                self._value(tokens)
        # a non-integer exponent of a negative base has no value
        assert isnan(self._value('pow float("-inf") 2.5'))

    def test_rootn_index_value_strict(self):
        from mpmath import isnan, mpf
        r = self._value('rootn 2 9007199254740993')        # huge odd index resolves
        assert abs(r - 1) < mpf('1e-14') and r > 1
        assert isnan(self._value('rootn 2 1e400'))          # past the index cap: no value


class TestRuleFilesAreDataNotCode:
    """`verify_ruleset` adjudicates rule sets it did NOT produce -- so a leaf
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

        from simplipy.verify._contract import UnsupportedToken, literal_value
        assert literal_value('(-10)') == Fraction(-10)
        assert literal_value('.5') == Fraction(1, 2) and literal_value('2.125') == Fraction(17, 8)
        assert literal_value('np.pi') == math.pi and literal_value('np.e') == math.e
        assert literal_value('float("inf")') == math.inf
        assert literal_value('float("-inf")') == -math.inf
        assert math.isnan(literal_value('float("nan")'))
        # ONE GRAMMAR: the judge reads a literal exactly where the engine's evaluator does
        # (THE TABLE's reader), which spells the infinities with double quotes only
        with pytest.raises(UnsupportedToken):
            literal_value("float('-inf')")
        assert literal_value('1/3') == Fraction(1, 3)  # legal in a foreign ruleset

        # A decimal literal denotes the EXACT rational it spells, not the nearest
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
    """The opposite-polarity hole: `judge_rule` sampled wildcard slots
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
        # everywhere including the diagonal (nan*x and x*nan agree at the atom too).
        assert verify_rule(['*', '_0', '_1'], ['*', '_1', '_0'])['verdict'] == 'CERTIFIED'
        # (a-b)+(b-a) -> 0 at `_` dies under the nan-erasure rule: it erases nan whenever
        # EITHER subtree is nan-valued -- off the diagonal, not because of it. The
        # data-sort spelling keeps this test's question -- the DIAGONAL must not
        # convict -- alive without the nan question: at ?0 = ?1 the residual is 0.
        assert verify_rule(['+', '-', '_0', '_1', '-', '_1', '_0'], ['0'])['verdict'] == 'KILL'
        assert verify_rule(['+', '-', '?0', '?1', '-', '?1', '?0'], ['0'])['verdict'] == 'CERTIFIED'


class TestTheRealisationAxis:
    """The judge reports WHICH AUTHORITY a rule answers to, not just a verdict.

    The two notions of soundness are INCOMPARABLE -- a rule can be true and
    unrealised (`atanh(tanh t) -> t`: exactly t on R, `inf` in f64 past 19.06) or
    realised and untrue (`asin(1e-8) -> 1e-8`: bit-identical in f64, wrong by
    1.7e-17). So this is a second axis, and the tier is the cell it lands in.
    """

    def test_the_four_tiers_are_reachable_and_correctly_assigned(self):
        from simplipy.verify._contract import judge_rule
        cases = [
            # true AND realised
            (['*', '(-1)', 'asin', '_0'], ['asin', 'neg', '_0'], 'core'),
            # true, NOT realised: tanh attains exactly 1.0 from 19.061547465398498,
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

    def test_a_TOLERATED_rule_that_fills_f64_nan_holes_on_a_null_set_is_core(self) -> None:
        """One f64 standard: the f64 constructors fill a NaN hole on a null set (`x1/x1`
        is `1`), so the judge files a rule that does only that as realised in f64.

        `/ ?0 ?0 -> 1` is true off the null set {0}, which is what TOLERATED means; f64
        answers nan at 0, a hole the rewrite fills at exactly the event the contract
        tolerated. A disagreement of any other kind keeps a TOLERATED rule `real` (the
        table below)."""
        from simplipy.verify._contract import _tier, judge_rule
        got = judge_rule(['/', '?0', '?0'], ['1'])
        assert got['verdict'] == 'TOLERATED'
        assert got['realised'] is True
        assert got['tier'] == 'core'
        # the whole table, so a future edit has to move a documented cell on purpose
        # `None` files as `real` too: absent deployed
        # evidence cannot support a claim of f64 soundness.
        assert [_tier(v, r) for v in ('CERTIFIED', 'TOLERATED') for r in (True, False, None)] \
            == ['core', 'real', 'real', 'core', 'real', 'real']
        # held in f64 and not judged in permissive (read as real's): float64's own value
        assert (_tier('KILL', True), _tier('KILL', False)) == ('f64-only', 'reject')
        assert (_tier('ENGINE-MISALIGN', False), _tier('NO-WITNESS', None)) == ('real', 'reject')
        # permissive's criterion is the third axis: an extension holds there alone, and a rule
        # real holds only by a `$` slot's excused zero is real's alone
        assert (_tier('KILL', False, True), _tier('KILL', True, True)) == ('permissive', 'f64')
        assert (_tier('CERTIFIED', False, False), _tier('CERTIFIED', True, False)) \
            == ('real-only', 'core')

    def test_realisation_is_undetermined_across_a_snap_not_asserted(self):
        """A snap means the f64 algebra evaluates a DIFFERENT point, so the deployed
        comparison is skipped rather than answered -- `None`, never a guess. Asserting
        realisation there would convict on a measurement artifact. (`sin(np.pi)` itself no
        longer snaps: the table reads pi exactly, sin(pi) = 0, and float64's
        1.2246467991473532e-16 is simply not 0 -- `real`, not realised.)"""
        from simplipy.verify._contract import judge_rule
        got = judge_rule(['sin', 'exp', 'log', 'np.pi'], ['0'])
        assert got['verdict'] == 'CERTIFIED'
        assert got['realised'] is None
        exact = judge_rule(['sin', 'np.pi'], ['0'])
        assert (exact['verdict'], exact['realised'], exact['tier']) == ('CERTIFIED', False, 'real')

    def test_the_axis_is_additive_engine_misalign_keeps_its_gate(self):
        """The realisation accumulator runs unconditionally; ENGINE-MISALIGN keeps its
        original gate (contract certifies AND deployment diverges AND all-real
        bindings). A rule the contract also rejects is a KILL, not a misalignment."""
        from simplipy.verify._contract import judge_rule
        got = judge_rule(['asin', 'pow', '10', '(-8)'], ['1e-08'])
        assert got['verdict'] == 'KILL' and got['realised'] is True


class TestUlpGapZeroCanonicalization:
    """view(-0.0) is int64-min: the raw view subtraction against a positive double
    wrapped (RuntimeWarning at the F-flag site, seen live in the acj-5-4 sweep) and
    convicted 1-ULP agreements as real changes. The fix: both
    zeros canonicalize to +0.0 on entry; everything else is pinned unchanged."""

    def test_negative_zero_pairs_like_positive_zero(self) -> None:
        import warnings
        from simplipy.verify._contract import _ulp_gap
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert _ulp_gap(-0.0, 5e-324) == 1
            assert _ulp_gap(-0.0, 1.0) == _ulp_gap(0.0, 1.0) != float('inf')
            assert _ulp_gap(-0.0, 0.0) == 0

    def test_the_unfixed_semantics_are_pinned(self) -> None:
        from simplipy.verify._contract import _ulp_gap
        assert _ulp_gap(-0.0, -5e-324) == float('inf')    # zero vs negative half-line
        assert _ulp_gap(-5e-324, 5e-324) == float('inf')  # a genuine sign change
        assert _ulp_gap(-1.0, -1.0000000000000002) == 1   # ordinary negative pair


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
        rounding = judge_rule(['/', '1', 'exp', '?0'], ['exp', 'neg', '?0'])
        assert rounding['realised'] is True and rounding['tier'] == 'core'
        divergent = judge_rule(['acos', 'cos', 'cos', '_0'], ['abs', 'cos', '_0'])
        assert divergent['realised'] is False and divergent['tier'] == 'real'

    def test_the_bound_is_the_tables_and_sits_in_an_empty_region(self) -> None:
        """WHY 8 AND NOT 4 OR 16. Over the rules bit-exactness moves, the gap distribution
        is 310 at 1 ULP, a thin tail through 7, then NOTHING until 57: every bound in
        [8, 56] gives the identical partition; 4 splits the rounding cluster in half. The
        bound is THE TABLE's f64 agreement (F6), defined once in the core
        (`_core.TABLE_REALISED_ULP`), and the judge's mirror is the same number.

        The 1-ULP rounding of `1/exp(x)` is `core`, and the 2377-ULP divergence is not. (The
        6-ULP straddler of the old probe set, `abs(sin(acos x)) -> cos(asin x)`, is `real` now
        for another reason: at x = 1 the table reads cos(pi/2) as exactly 0, no longer
        snapping, so the f64 lane compares 0 against float64's cos(fl(pi/2)) = 6.1e-17. And
        `pi - acos(x) -> acos(-x)`, whose worst gap on the judge's grid was 7 ULP, is `real`
        since the f64 lane reads the mint's rows too: near x = -1 the subtraction cancels, 77
        ULP at x = -0.99971.)"""
        from simplipy import _core
        from simplipy.verify import _contract as C
        assert C.REALISED_ULP == _core.TABLE_REALISED_ULP == 8
        eight = 1.0
        for _ in range(8):
            eight = float(np.nextafter(eight, np.inf))
        assert _core.table_f64_agree(1.0, eight) == 'eq'
        assert _core.table_f64_agree(1.0, float(np.nextafter(eight, np.inf))) == 'REAL-CHANGE'
        probes = [(['-', 'np.pi', 'acos', '_0'], ['acos', 'neg', '_0'], 'real'),     # 77 ULP
                  (['/', '1', 'exp', '?0'], ['exp', 'neg', '?0'], 'core'),           # 1 ULP
                  (['acos', 'cos', 'cos', '_0'], ['abs', 'cos', '_0'], 'real'),      # 2377 ULP
                  (['abs', 'sin', 'acos', '_0'], ['cos', 'asin', '_0'], 'real'),
                  (['pow', 'np.e', 'sinh', '(-5)'], ['0'], 'reject')]                # gross
        assert [C.judge_rule(a, b)['tier'] for a, b, _ in probes] == [t for _, _, t in probes]


class TestTheF64RangeStandard:
    """f64 promises the evaluator's value wherever nothing overflows or underflows;
    overflows are a special case. The constructors give the true value where an
    intermediate leaves the range (`x1*x2/x2 -> x1`, `exp(x1)/exp(x1) -> 1`), so at a point
    where an operation of either side overflows or underflows the judge's deployed lane
    accepts the true value as well as the evaluator's, and it compares every other point
    with the same bar as before."""

    F64 = v.MODE_TIERS['f64']

    def test_a_rule_that_differs_only_out_of_range_is_f64(self) -> None:
        """`rootn(exp(x), 2) -> exp(x/2)` and `exp(x)/exp(x) -> 1` are true on R and
        computed by f64 at every point where exp stays in range; they differed only past
        exp's overflow (709.78) and underflow (-745.13), which made them ENGINE-MISALIGN,
        tier `real`, and kept them out of the f64 set. Their ground instances, which the
        constructors perform (`rootn(exp(800.5), 2) -> exp(1601/4)`), are f64 too."""
        for lhs, rhs in ((['rootn', 'exp', '?0', '2'], ['exp', '/', '?0', '2']),
                         (['rootn', 'exp', '?0', '2'], ['exp', '*', '0.5', '?0']),
                         (['/', 'exp', '?0', 'exp', '?0'], ['1']),
                         (['rootn', 'exp', '800.5', '2'], ['exp', '400.25']),
                         (['/', 'exp', '800', 'exp', '800'], ['1']),
                         (['log', 'exp', '1000'], ['1000'])):
            got = judge_rule(lhs, rhs)
            assert (got['verdict'], got['realised'], got['tier']) \
                == ('CERTIFIED', True, 'core'), (lhs, rhs, got)
            assert got['out_of_range'] > 0, got
        assert v.verify_ruleset([[['rootn', 'exp', 'x0', '2'], ['exp', '/', 'x0', '2']],
                                 [['/', '1', 'exp', 'x0'], ['exp', 'neg', 'x0']]],
                                mode='f64')['is_clean'] is True

    def test_at_a_range_event_only_the_two_values_are_accepted(self) -> None:
        """The special case admits the true value beside the evaluator's, nothing else.
        `exp(1000) -> 5` gives neither and stays `reject`; the evaluator's own value stays
        realised, whether true (`exp(1000) -> inf` is the extended reading) or not
        (`exp(sinh(-10))` is 1e-4783, and f64 computes 0.0)."""
        got = judge_rule(['exp', '1000'], ['5'])
        assert (got['tier'], got['out_of_range']) == ('reject', 0), got
        assert judge_rule(['exp', '1000'], ['float("inf")'])['tier'] in self.F64
        got = judge_rule(['exp', 'sinh', '(-10)'], ['0'])
        assert (got['verdict'], got['realised'], got['tier']) == ('KILL', True, 'f64'), got

    def test_saturation_inside_the_range_is_still_compared(self) -> None:
        """`tanh` attains exactly 1.0 from 19.06 without leaving the range, and
        `atanh(1.0)` is a pole, not an overflow: no point is a range event, and f64 still
        refuses `atanh(tanh(x)) -> x`."""
        got = judge_rule(['atanh', 'tanh', '?0'], ['?0'])
        assert (got['verdict'], got['tier'], got['out_of_range']) \
            == ('ENGINE-MISALIGN', 'real', 0), got
        assert got['tier'] not in self.F64

    def test_a_difference_at_an_ordinary_point_still_refuses(self) -> None:
        """Accepting the true value at a range event removes none of the bar elsewhere.
        `log(exp(x)) -> x` gives the true value past exp's overflow and underflow, and is
        still refused in f64: exp rounds a small argument into the neighbourhood of 1, so
        at the grid point 0.0137421 the deployed sides differ by 34 ULP, with nothing out
        of range."""
        from simplipy.verify import _contract as C
        got = judge_rule(['log', 'exp', '?0'], ['?0'])
        assert got['out_of_range'] > 0
        assert (got['verdict'], got['realised'], got['tier']) == ('CERTIFIED', False, 'real')
        x = C.F(0.0137421)
        in_range = [True]
        with np.errstate(all='ignore'):
            lhs = C.d_eval(C.parse(['log', 'exp', '?0']), {'?0': x}, in_range)
        assert in_range[0] and C._ulp_gap(float(lhs), float(x)) > C.REALISED_ULP
        assert judge_rule(['sin', '?0'], ['?0'])['tier'] == 'reject'

    def test_real_reads_the_contract_alone(self) -> None:
        """The contract lane is untouched: every rule above is true on R, so each stays
        licensed in `real` (and in `permissive`), and a false one stays out."""
        for lhs, rhs in ((['rootn', 'exp', '?0', '2'], ['exp', '/', '?0', '2']),
                         (['log', 'exp', '?0'], ['?0']), (['atanh', 'tanh', '?0'], ['?0'])):
            tier = judge_rule(lhs, rhs)['tier']
            assert tier in v.MODE_TIERS['real'] and tier in v.MODE_TIERS['permissive']
        assert judge_rule(['sin', '?0'], ['?0'])['tier'] not in v.MODE_TIERS['real']

    @pytest.mark.parametrize('op, args, out', [
        ('exp', (800.0,), True), ('exp', (-800.0,), True), ('exp', (-740.0,), True),
        ('*', (1e200, 1e200), True), ('*', (1e-200, 1e-200), True),
        ('+', (1e308, 1e308), True), ('pow', (10.0, 400.0), True),
        ('inv', (1e308,), True), ('cosh', (800.0,), True), ('rootn', (5e-324, -1.0), True),
        # poles: an exact infinity is what the operator computes
        ('atanh', (1.0,), False), ('log', (0.0,), False), ('/', (1.0, 0.0), False),
        ('pow', (0.0, -1.0), False), ('rootn', (0.0, -2.0), False), ('inv', (0.0,), False),
        # exact zeros are not underflows
        ('-', (0.5, 0.5), False), ('log', (1.0,), False), ('acos', (1.0,), False),
        ('*', (0.0, 1e-200), False),
        # saturation inside the range, and non-finite operands
        ('tanh', (20.0,), False), ('exp', (math.inf,), False), ('exp', (700.0,), False),
    ])
    def test_the_operation_classifier(self, op, args, out) -> None:
        """THE TABLE's f64 reading marks exactly the operations that overflow or underflow."""
        from simplipy.verify import _contract as C
        names = ['?0', '?1'][:len(args)]
        in_range = [True]
        value = C.d_eval(C.parse([op] + names), dict(zip(names, map(C.F, args))), in_range)
        assert (not in_range[0]) is out, (op, args, value)

    def test_the_classifier_covers_the_whole_vocabulary(self) -> None:
        """The event is the core's one definition, which the miner's scan reads too. An
        operator outside its table is never read as out of range, so every operator the
        judge speaks must be in it on purpose."""
        from simplipy import _core
        from simplipy.verify import _contract as C
        unknown = [op for op, arity in C.ARITY.items()
                   if _core.range_event(op, [1.0] * arity, 1.0) is None]
        assert unknown == []
        assert _core.range_event('no-such-operator', [1.0], math.inf) is None


class TestCleanlinessIsPerMode:
    """Before the triple there was one rule set and one contract, so "clean" could mean
    "no fatal bucket". That is now wrong in both directions, and the soundness claim ("0.14.0 is sound,
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
        assert verify_ruleset([self.REAL], mode='permissive')['is_clean'] is True
        assert verify_ruleset([self.REAL], mode='f64')['is_clean'] is False

    def test_no_mode_may_carry_a_reject(self) -> None:
        from simplipy.verify import verify_ruleset
        for mode in ('f64', 'real', 'permissive'):
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
        """Three per-file sweeps, and NOT the retired set identity. `permissive == f64
        UNION real` was dropped once folding became mode-dependent: each file omits what its
        own constructor performs, so set overlap measures spelling. On the shipped artifact
        the union exceeds the permissive file by 362 rules and permissive performs every one -- the
        identity would call a correct artifact dirty."""
        from simplipy.verify import verify_triple
        good = verify_triple([self.CORE], [self.CORE, self.REAL], [self.CORE, self.REAL])
        assert good['is_clean'] is True and good['relationships'] == []

        # a permissive file SMALLER than the union is now legitimate, not a defect
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
        from conftest import acj_config_path, acj_corpus_rules_path, require_triple
        from simplipy.verify import verify_triple
        base = acj_config_path().replace('config.yaml', '')
        # the TRIPLE is the subject, so guard on the triple and not merely on the config:
        # the f64 set ships on its own while the real/permissive sets are re-mined
        require_triple('needs the shipped acj-4-3 triple')
        # the third file keeps its published name on pre-rename artifacts; the
        # conftest helper resolves whichever spelling the cell actually ships
        report = verify_triple(base + 'rules.json', base + 'rules_real.json',
                               acj_corpus_rules_path(),
                               engine_config=acj_config_path())
        # THE ARTIFACT WAS MINED UNDER THE JUDGE BEFORE THE TABLE, and 0.15 mines it again.
        # Read in THE TABLE's values and by the one criterion, its files hold rules that do
        # not hold in their mode, each for a reason the specification names, and exactly
        # these counts: f64 6 (`sin acos _0 -> cos asin _0` and kin: at 1, float64 gives 0
        # on one side and cos(fl(pi/2)) = 6.1e-17 on the other, which the judge read past
        # while it snapped cos(pi/2)); real 56 (36 where the right side's 1/0 has no value at
        # the left side's pole, `abs(log(cos _0)) -> log(1/cos _0)` at pi/2; 14 where a `_`
        # slot at -inf meets 1/exp(-inf) = 1/0; 6 ground facts through pow(-1, inf) or
        # tan(pi/2)); permissive 24 (the first family at its real-valued points). Under the
        # owner's rulings on isolated infinities and guards: real 77 (+22 extensions at a null
        # set no `_` slot guards, `abs atan tan _0 -> abs asin sin _0` at pi/2; -1 now holding
        # by clause (c), `neg log log ?0 -> log inv log ?0`), permissive 23 (-1, the same).
        # Under P3 as ruled (permissive serves real's values up to float resolution, except at
        # isolated points, no longer float64's values): permissive 34 (+12 that hold in f64
        # only -- 7 `neg log acos _0 -> log inv acos _0` and kin, an infinity of the left side
        # at a point a `_` slot does not guard; 4 `pow 0 atan _0 -> pow 0 _0` and kin at
        # `_0 = -inf`; `tanh sinh atanh _0 -> tanh tan asin _0` at `_0 = +-1`, unguarded -- and
        # -1, `exp * float("-inf") ?0 -> pow 0 ?0`, now an extension within the criterion).
        # With the f64 lane reading the mint's rows too: f64 25 (+19, float64 losing the rule at
        # the rows' small and near-1 magnitudes: `pi - acos(_0) -> acos(-_0)` cancels near -1,
        # `exp(log(asin(_0))) -> asin(exp(log(_0)))` and kin, `pow(_0, log(4)) -> pow(4,
        # log(_0))` and kin, `sinh(atanh(atan(_0))) -> tan(asin(atan(_0)))` and kin), and in
        # every mode +2 false in the reals at the rows near 1, `sin(abs(atanh(_0))) ->
        # abs(sin(atanh(_0)))` and its twin, where atanh passes pi and sin turns negative.
        # the offenders by mode, shown on a failure (captured output is printed in full)
        for m, r in report['modes'].items():
            print(m, sorted((o['idx'], o['lhs'], o['rhs'], o['verdict']) for o in r['offenders']))
        assert {m: len(r['offenders']) for m, r in report['modes'].items()} == \
            {'f64': 25, 'real': 79, 'permissive': 36}
        # the package ships the corpus, so the sweep runs in any install. Exactly the known
        # violations (conftest: a row where real's one zero and permissive's signed zero give
        # different values).
        assert report['permissive_dominance'] == {
            'checked': True, 'violations': list(PERMISSIVE_DOMINANCE_KNOWN)}
        assert report['is_clean'] is False, report['relationships']
        assert len(report['relationships']) == (1 if PERMISSIVE_DOMINANCE_KNOWN else 0)

    def test_an_empty_corpus_reports_UNCHECKED_never_a_clean_pass(self) -> None:
        """An explicitly empty `corpus_rows` sweeps nothing. An empty violation list
        would read as a genuine zero-violation sweep, so unchecked must SAY unchecked,
        loudly, and must neither pass nor fail `is_clean`."""
        import simplipy.verify as V
        with pytest.warns(UserWarning, match='NOT checked'):
            report = V.verify_triple([self.CORE], [self.CORE], [self.CORE],
                                     engine_config='<never dereferenced on 0 rows>',
                                     corpus_rows=[])
        dom = report['permissive_dominance']
        assert not isinstance(dom, list), 'an empty list would read as a pass'
        assert dom == {'checked': False, 'reason': 'corpus_rows was given but is empty'}
        # unchecked is unknown: it must not dirty an otherwise clean triple
        assert report['is_clean'] is True and report['relationships'] == []

    def test_the_default_corpus_ships_with_the_package(self) -> None:
        """The gate's corpus is package data, so the default sweep runs wherever the
        package is installed: 400 prefix-token rows, read from inside the package."""
        import os
        import simplipy.verify as V
        rows = V.default_corpus()
        assert len(rows) == 400
        assert all(isinstance(r, list) and r and all(isinstance(t, str) for t in r) for r in rows)
        data = os.path.join(os.path.dirname(V.__file__), 'data', 'raw_skeletons_nv.json')
        assert os.path.isfile(data)

    def test_rows_actually_swept_report_checked_True(self) -> None:
        """The positive half of the contract: with rows given, the report says the sweep
        ran (`checked: True`) and carries the violation list -- empty on a dominating
        artifact, so a real pass is now spelled differently from a skipped check."""
        from conftest import acj_config_path, require_triple
        require_triple('the dominance sweep simplifies in all three modes')
        from simplipy.verify import verify_triple
        report = verify_triple([self.CORE], [self.CORE], [self.CORE],
                               engine_config=acj_config_path(),
                               corpus_rows=[['sin', 'x1'], ['+', 'x1', '1']])
        assert report['permissive_dominance'] == {'checked': True, 'violations': []}
        assert report['is_clean'] is True

    def test_dominance_is_checked_on_the_triple_given(self) -> None:
        """The subject is the three rule sets passed in, not the files the config names:
        the config contributes its operators and nothing else. A real set holding a rule
        the permissive set lacks (`sin x0 -> x0`, judged unsound, but dominance is about
        capability) out-simplifies permissive on `sin(x0)`, and the check must see it --
        the config's own files never carry that rule."""
        from conftest import acj_config_path, require_or_skip
        require_or_skip(acj_config_path(), 'needs the acj-4 config for its operators')
        from simplipy.verify import verify_triple
        report = verify_triple([self.CORE], [self.REJECT], [self.CORE],
                               engine_config=acj_config_path(),
                               corpus_rows=[['sin', 'x0'], ['+', 'x1', '1']])
        assert report['permissive_dominance'] == {'checked': True, 'violations': [
            {'expression': ['sin', 'x0'], 'mu': {'f64': 12000, 'real': 6000, 'permissive': 12000}}]}
        assert report['is_clean'] is False


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
        got = judge_rule('log + cosh * 25 ?0 sinh * 25 ?0'.split(),
                         '* 25 ?0'.split())
        assert got['verdict'] != 'KILL'
        assert got['tier'] == 'real'

    def test_the_shipped_identities_are_no_longer_near_the_bar(self) -> None:
        """Three EXACT identities on the shipped artifact sat at 6 of 167 grid points
        against a kill bar of 9 -- three points from being wrongly convicted at the next
        mine. `cos(asin(tanh t)) = sech t` and siblings. There is no grid bar any more: a
        grid point convicts alone (clause (a)), so every point must read 'eq', and the
        measure is a certified cell's, which these have none of."""
        from simplipy.verify._contract import judge_rule
        for lhs, rhs in ((['cos', 'asin', 'tanh', '_0'], ['inv', 'cosh', '_0']),
                         (['sin', 'acos', 'tanh', '!0'], ['inv', 'cosh', '!0'])):
            got = judge_rule(list(lhs), list(rhs))
            assert got['verdict'] != 'KILL', (lhs, got)
            assert got['measure'] == 0.0, (lhs, got['measure'])

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


class TestTheBatterySweepIsExhaustiveWhereItClaimsToBe:
    """The slot-product cap was the literal 500, under a comment asserting that "500
    covers every <=2-slot rule exhaustively (23^2 = 529 ~ capped edge)". 529 > 500, so
    the comment described the one thing the number could not do."""

    def test_the_cap_is_the_full_two_slot_product(self) -> None:
        from simplipy.verify._contract import BATTERY_CAP, battery_for
        widest = max(len(battery_for(s)) for s in '?_!$')
        assert BATTERY_CAP == widest ** 2, (BATTERY_CAP, widest)

    def test_a_two_slot_rule_is_never_sampled(self) -> None:
        """The real assertion is behavioural: the sampling branch must not fire. Under
        the old cap it fired for EVERY 2-slot rule and dropped 29 of the 529 pairs --
        and since the cap applies once per slot as the product is built, no combo
        extending one of those 29 could reach a 3-slot sample either."""
        import simplipy.verify._contract as C
        drawn = []
        real_rng = C.np.random.default_rng

        def spy(*a, **k):
            drawn.append(a)
            return real_rng(*a, **k)

        C.np.random.default_rng = spy
        try:
            got = C.judge_rule(['+', '_0', '_1'], ['+', '_1', '_0'])
            assert got['verdict'] == 'CERTIFIED', got
            assert drawn == [], 'the two-slot sweep was sampled, not exhaustive'
            # and the guard is load-bearing: at the old cap the same rule IS sampled
            C.BATTERY_CAP = 500
            C.judge_rule(['+', '_0', '_1'], ['+', '_1', '_0'])
            assert drawn, 'the spy never observed the branch it is guarding'
        finally:
            C.np.random.default_rng = real_rng
            C.BATTERY_CAP = max(len(C.battery_for(s)) for s in '?_!$') ** 2


class TestTheExistsWitnessIsActuallySearchedFor:
    """`forall c_s exists c_t` was decided by a search that stopped at 1e12, and the
    failures it produced were filed in `skipped_cl` -- a key every verdict returns and
    nothing reads."""

    def test_the_search_reaches_the_f64_ceiling(self) -> None:
        """`pow(exp(9), 5) = 3.5e19` is an unremarkable constant-fold whose witness is
        the value itself. The old grid could not BRACKET it, so 15 rules of the shipped
        `rules_real.json` were reported witness-less on a search artifact."""
        import math

        from simplipy.verify._contract import _XS, fit_witness, parse
        assert max(_XS) >= 1e300 and min(_XS) <= -1e300
        tl = parse(['pow', 'exp', '9', '<constant>'], '<C_L>')
        tr = parse(['<constant>'], '<C_R>')
        want = math.exp(9) ** 5
        got = fit_witness(tl, tr, set(), 5.0)
        assert got is not None, 'no witness for a plain constant-fold'
        assert got == want, (got, want)

    def test_a_defined_lhs_with_no_witness_is_convicted_not_swallowed(self) -> None:
        """The rule holds at every constant that fits, so the verdict used to be
        CERTIFIED. A cl where the LHS HAS a value and no c_t reproduces it falsifies the
        exists-claim outright, whether or not that cl is one of the core CONSTS."""
        import math

        import simplipy.verify._contract as C
        lhs, rhs = ['pow', 'exp', '9', '<constant>'], ['<constant>']
        assert C.judge_rule(list(lhs), list(rhs))['verdict'] == 'CERTIFIED'

        real_fit = C.fit_witness

        def blind_at_pi(tl, tr, shared, cl_val=None, env_out=None, rname='<C_R>'):
            # pi is NOT in CONSTS, so the old code appended it to `skipped_cl`
            if cl_val and abs(cl_val['<C_L0>'] - math.pi) < 1e-12:
                return None
            return real_fit(tl, tr, shared, cl_val, env_out, rname)

        C.fit_witness = blind_at_pi
        try:
            got = C.judge_rule(list(lhs), list(rhs))
        finally:
            C.fit_witness = real_fit
        assert got['verdict'] == 'NO-WITNESS', got
        assert 'cl=' in got['detail'], got
        # pi is the cl the fit was BLINDED at, so pi is what convicts.
        assert str(math.pi) in got['detail'], got
        # `skipped_cl` is no longer empty, and that is correct rather than a regression:
        # the widened battery reaches |c| = 1e4, and `exp(9)^c` overflows there, so those
        # constants are genuinely degenerate. Assert the PROPERTY instead of the old
        # empty list -- every skipped cl must be one where no witness can be DETERMINED
        # (`_fit_target_env` is the judge's own test for that), and the blinded pi must
        # not be among them, or the conviction would be coming from the wrong place.
        assert (math.pi,) not in got['skipped_cl'], got
        tl = C.parse(lhs, '<C_L>', number=True)
        tr = C.parse(rhs, '<C_R>', number=True)
        assert got['skipped_cl'], 'the widened battery must reach the degenerate range'
        for (cl,) in got['skipped_cl']:
            assert C._fit_target_env(tl, tr, set(), {'<C_L0>': cl}, '<C_R0>') is None, (cl, got)

    def test_an_undefined_lhs_is_still_a_legitimate_skip(self) -> None:
        """The other half of the split must not move: where the LHS has NO value the
        rule is vacuous at that constant, and skipping is correct. `log(cosh(tan(c)))`
        at c = +-pi/2 is infinite in f64, and stays a skip rather than a conviction."""
        import simplipy.verify._contract as C
        got = C.judge_rule(['log', 'cosh', 'tan', '<constant>'], ['<constant>'])
        assert got['verdict'] != 'NO-WITNESS', got
        assert got.get('skipped_cl'), 'the degenerate skip stopped being recorded'


class TestTheWitnessOutResolvesTheLadder:
    """`mp_polish` says it in its own docstring -- "THE WITNESS MUST OUT-RESOLVE THE
    LADDER" -- and then polished to a FIXED `GAP_RUNGS[-1] + 50`. The ladder's
    depth is operand-scaled, so that constant stopped naming its deepest rung."""

    def test_the_polish_depth_follows_the_operands(self) -> None:
        """`cosh(710) ~ 1.1e308` puts rung 2 at dps 668 and rung 3 at dps 1336. A witness
        frozen at 300 leaves the SAME ~1e-300 residue at both rungs: it does not decay,
        so the decay test reads it as an analytic gap."""
        from mpmath import cosh, log, mp, mpf

        from simplipy.verify._contract import mp_polish, parse
        tl = parse(['log', 'cosh', '<constant>'], '<C_L>')
        tr = parse(['<constant>'], '<C_R>')
        got = mp_polish(tl, tr, {}, lambda: mpf(710), 709.3068528194401)
        old = mp.dps
        try:
            mp.dps = 1500
            want = log(cosh(mpf(710)))
            digits = -mp.log10(abs(got - want) / abs(want))
        finally:
            mp.dps = old
        # rung 3 sits at dps 1336; the witness must be deeper than that, not at 300
        assert digits > 1336, f'witness good to only {digits} digits'

    def test_a_rule_true_by_construction_is_not_killed_at_the_overflow_edge(self) -> None:
        """`log(cosh(C)) -> c_t` holds for every C with c_t = log(cosh(C)). At C = 710 --
        the last constant where f64 `cosh` is still finite -- the shallow witness made it
        a clause-(a) KILL."""
        from mpmath import mpf

        import simplipy.verify._contract as C
        base = C.judge_cl_battery
        C.judge_cl_battery = lambda **kw: base(**kw) + [lambda: mpf(710)]
        try:
            got = C.judge_rule(['log', 'cosh', '<constant>'], ['<constant>'])
        finally:
            C.judge_cl_battery = base
        assert got['verdict'] != 'KILL', got

    def test_a_witness_under_the_old_absolute_floor_is_not_flattened_to_zero(self) -> None:
        """The witness ACCEPTANCE test kept the shape removed from the contract:
        `|residual| <= _WIT_RESID * max(1, |target|)` is a pure absolute floor for every
        target below 1. `asin(inv(cosh(710))) = 8.95e-309` is under 1e-270 absolutely, so
        0 was accepted as its witness though it is 100% wrong; the contract then judged by
        relative decay, saw a frozen gap and KILLed a rule true by construction."""
        from mpmath import asin, cosh, mp, mpf

        from simplipy.verify._contract import mp_polish, parse
        tl = parse(['asin', 'inv', 'cosh', '<constant>'], '<C_L>')
        tr = parse(['<constant>'], '<C_R>')
        got = mp_polish(tl, tr, {}, lambda: mpf(710), 0.0)   # 0.0 is what f64 fits
        assert got != 0, 'the subnormal witness was flattened to an exact zero'
        old = mp.dps
        try:
            mp.dps = 1500
            assert abs(got - asin(1 / cosh(mpf(710)))) < mpf(10) ** -1000
        finally:
            mp.dps = old

    def test_a_cancellation_zero_is_still_accepted_as_zero(self) -> None:
        """The other side of the same bar. A relative-only test would reject 0 for a
        target that is zero mathematically but reads as precision residue, and send the
        secant chasing that noise. The floor is the largest intermediate scaled by the
        working precision -- below it, a value is not distinguishable from zero."""
        from mpmath import mpf

        from simplipy.verify._contract import mp_polish, parse
        tl = parse(['sin', '*', '<constant>', 'np.pi'], '<C_L>')
        tr = parse(['<constant>'], '<C_R>')
        got = mp_polish(tl, tr, {}, lambda: mpf(2), 0.0)
        assert got == 0, f'a cancellation zero was chased to {got}'


class TestAnExactZeroIsNotAnUndecayableGap:
    """The decay test compares ONE reading at TWO rungs, so any normaliser that is
    the same at both cancels out of the ratio and cannot change a verdict. The old
    `max(|l|, |r|)` was not the same at both: against an exact zero the scale IS the
    residue, so the quotient was 1.0 at EVERY precision. A gap that cannot move by
    construction is read as an analytic one, and clause (a) convicted it."""

    #: True on the reals, with one side EXACTLY zero -- so the judge is comparing pure
    #: precision residue against an exact zero, the shape that could never decay.
    #: Spelled `?` (the nan-erasure rule): every row drops its slot into a constant
    #: RHS, which erases nan at a `_` binding and correctly dies `a-nonfinite-binding` there
    #: -- the exact-zero question this class pins is sort-independent.
    TRUE_ZERO_SIDE = [
        (['log', '*', 'exp', '?0', 'exp', 'neg', '?0'], ['0']),
        (['-', 'atanh', 'tanh', '?0', '?0'], ['0']),
        (['-', 'asinh', 'sinh', '?0', '?0'], ['0']),
        (['-', 'log', '+', 'cosh', '?0', 'sinh', '?0', '?0'], ['0']),
        (['-', '*', 'tanh', '?0', 'cosh', '?0', 'sinh', '?0'], ['0']),
        (['-', 'log', 'exp', '?0', '?0'], ['0']),
    ]

    def test_a_true_identity_against_an_exact_zero_is_not_killed(self) -> None:
        """Every row here is zero by construction on the reals. All six were clause-(a)
        KILLs at `tier=reject` -- convicted for a gap that was arithmetic residue and
        nothing else."""
        from simplipy.verify._contract import judge_rule
        bad = []
        for lhs, rhs in self.TRUE_ZERO_SIDE:
            got = judge_rule(lhs, rhs)
            if got.get('verdict') == 'KILL' or got.get('tier') == 'reject':
                bad.append((' '.join(lhs), got.get('verdict'), got.get('clause')))
        assert not bad, f'true zero-side identities convicted: {bad}'

    def test_the_wildcard_spelling_of_a_slot_dropper_dies_at_the_nan_atom(self) -> None:
        """The same identities at `_` claim soundness for nan-valued subtrees they
        cannot deliver -- `atanh(tanh(log x0)) - log x0` is nan on x0 < 0 and the
        rewrite defines it. The nan-erasure rule routes that to `a-nonfinite-binding`, and the honest sort
        for a slot-dropping rule is `?` (above), never `_`."""
        from simplipy.verify._contract import judge_rule
        got = judge_rule(['-', 'atanh', 'tanh', '_0', '_0'], ['0'])
        assert got['verdict'] == 'KILL', got
        assert got['clause'] == 'a-nonfinite-binding', got

    def test_the_saturation_family_is_still_convicted(self) -> None:
        """The other direction, and the reason the floor cannot simply be widened to
        cover the rows above. `exp(sinh(-10))` is exactly 1.03e-4783 at EVERY working
        precision -- tiny, but FIXED, and false. Flooring a nonzero separation at an
        absolute `10^-dps` swamps it and manufactures a decay it does not have, so
        nothing nonzero is floored. These must stay `f64`-tier kills, never `real`."""
        from simplipy.verify._contract import judge_rule
        for lhs, rhs in ((['exp', 'sinh', '(-10)'], ['0']),
                         (['exp', 'sinh', '(-8)'], ['0']),
                         (['tanh', '(-19)'], ['(-1)']),
                         (['tanh', '20'], ['1'])):
            got = judge_rule(lhs, rhs)
            assert got.get('verdict') == 'KILL', (lhs, got)
            assert got.get('clause') == 'a-real-change', (lhs, got)
            assert got.get('tier') == 'f64', (lhs, got)

    def test_the_reading_is_absolute_so_no_scale_can_normalise_it_away(self) -> None:
        """The unit underneath. A separation reported as a QUOTIENT of its own scale is
        1.0 whatever the residue is, which is what made the decay test blind."""
        from mpmath import mpf

        from simplipy.verify._contract import gap_reading
        for residue in ('1e-51', '1e-121', '1e-4783'):
            d, scale = gap_reading(('fin', mpf(0)), ('fin', mpf(residue)))
            assert d == mpf(residue), f'{residue}: separation normalised away -> {d}'
            assert scale == mpf(residue), f'{residue}: scale {scale}'
        # and it still reports exact agreement as an exact zero
        assert gap_reading(('fin', mpf(3)), ('fin', mpf(3)))[0] == 0


class TestABoundedFunctionNeverAttainsItsLimit:
    """The gap ladder cannot reach the saturation family and no affordable depth
    can: `tanh(cosh(10))` differs from 1 by 2e-9566 and `tanh(exp(10))` by 1e-19132, both
    inside the boundary band at every rung, so the honest numeric verdict is Unresolved
    forever. The question does not need a precision -- a bounded function never attains its
    limit at a finite argument -- and 174 rows of the shipped artifact were judged by
    nobody for want of asking it."""

    def test_deep_saturation_is_convicted_and_routed_to_f64(self) -> None:
        """These are real rows of the shipped sets. They are f64-REALISED (the deployed
        `tanh(cosh(-10))` is exactly 1.0), so conviction routes them to the f64 tier --
        exactly the file they ship in. The gate confirms them instead of abstaining."""
        from simplipy.verify._contract import judge_rule
        for lhs, rhs in ((['tanh', 'cosh', '(-10)'], ['1']),
                         (['tanh', 'cosh', 'cosh', '(-3)'], ['1']),
                         (['tanh', 'exp', '10'], ['1']),
                         (['tanh', '400'], ['1']),
                         (['exp', 'sinh', '(-50)'], ['0']),
                         (['inv', 'cosh', 'exp', '10'], ['0'])):
            got = judge_rule(lhs, rhs)
            assert got.get('verdict') == 'KILL', (lhs, got)
            assert got.get('tier') == 'f64', (lhs, got)

    def test_an_infinite_argument_does_attain_the_limit(self) -> None:
        """The condition that keeps this sound. `tanh(inf)` IS exactly 1 and `exp(-inf)`
        IS exactly 0, so the same shape is TRUE there and must stay certified."""
        from simplipy.verify._contract import judge_rule
        for lhs, rhs in ((['tanh', 'float("inf")'], ['1']),
                         (['tanh', 'float("-inf")'], ['(-1)']),
                         (['exp', 'float("-inf")'], ['0'])):
            got = judge_rule(lhs, rhs)
            assert got.get('verdict') == 'CERTIFIED', (lhs, got)
            assert got.get('tier') == 'core', (lhs, got)

    def test_a_computed_bound_is_not_a_written_one(self) -> None:
        """The other condition, and the subtle one. `1 - 2*exp(-2*cosh(10))` IS
        `tanh(cosh(10))` exactly, and it rounds to exactly 1.0 at every affordable
        precision -- so a test that accepted a COMPUTED side would convict an identity.
        Only a bound the rule WRITES counts, which is the literal-provenance doctrine
        `c_eval` already states for atanh/acosh."""
        from mpmath import mp, mpf

        from simplipy.verify._contract import parse, saturation_verdict
        tl = parse(['tanh', 'cosh', '10'])
        tr = parse(['-', '1', '*', '2', 'exp', '*', '(-2)', 'cosh', '10'])
        old = mp.dps
        try:
            mp.dps = 50
            assert mp.mpf(1) - mpf(2) * mp.exp(-2 * mp.cosh(mpf(10))) == 1, \
                'the premise of this test died: the computed side no longer rounds to 1'
            assert saturation_verdict(tl, tr, dict) is None, \
                'a computed bound was taken for a written one'
        finally:
            mp.dps = old

    def test_the_ladder_still_owns_every_verdict_it_can_reach(self) -> None:
        """This is consulted ONLY where the ladder refused, so it can add a verdict and
        never change one. Shallow saturation stays the ladder's call, and the true
        identities that live at the same asymptote stay untouched."""
        from simplipy.verify._contract import judge_rule
        assert judge_rule(['tanh', '30'], ['1'])['clause'] == 'a-real-change'
        # the slot-dropping identity is spelled `?`: its `_` spelling
        # correctly dies `a-nonfinite-binding`, which is sort doctrine, not the ladder's.
        for lhs, rhs in ((['cos', 'asin', 'tanh', '_0'], ['inv', 'cosh', '_0']),
                         (['log', '*', 'exp', '?0', 'exp', 'neg', '?0'], ['0'])):
            got = judge_rule(lhs, rhs)
            assert got.get('verdict') != 'KILL', (lhs, got)
            assert got.get('tier') == 'real', (lhs, got)


class TestTheQuantifierWidensWithoutConvictingTheTrue:
    """The judging battery for a source constant reaches |c| <= 5 (plus pi, e) for
    generic values, while the miner draws log-uniform over [1e-3, 1e3]; the gate cannot
    check what the miner explores. Widening it to +-1e4 convicted 97 of 343 shipped
    constant rows, and every one diagnosed was an artifact of the instrument at magnitudes
    it could not resolve. Six of them, each measurable on its own."""

    #: `(lhs, rhs, c)` -- the rule, and the source constant that convicted it.
    #: (lhs, rhs, source constant, (verdict, tier)) -- the verdict and tier measured for
    #: each row: certified in the tier its truth earns, or the documented abstention.
    WAS_CONVICTED = [
        # the noise floor swallowed an honestly tiny target and took 0 as the witness
        (['pow', 'acos', '0', '<constant>'], ['<constant>'], -3000.0, ('CERTIFIED', 'core')),
        (['pow', 'acosh', '2', '<constant>'], ['<constant>'], -3000.0, ('CERTIFIED', 'core')),
        (['asin', 'tanh', 'exp', '<constant>'], ['<constant>'], -1000.0, ('CERTIFIED', 'core')),
        # the floor was scaled by a magnitude that had already cancelled away
        (['asin', 'inv', 'cosh', '<constant>'], ['<constant>'], 1000.0, ('CERTIFIED', 'core')),
        (['log', 'atan', 'cosh', '<constant>'], ['<constant>'], 100.0, ('CERTIFIED', 'core')),
        # rung 1 lost a side entirely and the drop was measured from the rail
        (['acos', 'cos', 'exp', '<constant>'], ['<constant>'], -100.0, ('CERTIFIED', 'core')),
        (['log', 'cosh', 'exp', '<constant>'], ['<constant>'], -100.0, ('CERTIFIED', 'core')),
        # an absolute snap band inside the evaluator; realised since the fit keeps its own
        # witness (5.1e-131), which the band used to snap to 0
        (['asin', 'sin', 'exp', '<constant>'], ['<constant>'], -300.0, ('CERTIFIED', 'core')),
        (['atanh', 'sin', 'exp', '<constant>'], ['<constant>'], -300.0, ('CERTIFIED', 'core')),
        # past MAX_DPS, where the judge documents abstention and used to convict
        (['acos', 'cos', 'exp', '<constant>'], ['<constant>'], -3000.0, ('UNRESOLVED-COVERAGE', None)),
        # the witness search stopped eight decades short of the f64 ceiling
        (['pow', 'cosh', '(-3)', '<constant>'], ['<constant>'], 300.0, ('CERTIFIED', 'core')),
        (['pow', 'sinh', '3', '<constant>'], ['<constant>'], 300.0, ('CERTIFIED', 'core')),
    ]

    @staticmethod
    def _judge_at(lhs, rhs, c):
        """judge `lhs -> rhs` with the source-constant battery replaced by `c` alone."""
        from mpmath import mpf
        from simplipy.verify import _contract as C
        base = C.judge_cl_battery
        try:
            C.judge_cl_battery = lambda saturation=True: [lambda: mpf(c)]
            return C.judge_rule(list(lhs), list(rhs))
        finally:
            C.judge_cl_battery = base

    def test_no_true_rule_is_convicted_at_a_widened_constant(self) -> None:
        """Each row is true for EVERY source constant by construction -- the RHS constant
        is the fitted witness, so there is nothing for the quantifier to falsify. Each row's
        verdict and tier are pinned: certified, or -- past `MAX_DPS` -- the abstention the
        contract documents."""
        verdicts = [self._judge_at(lhs, rhs, c) for lhs, rhs, c, _ in self.WAS_CONVICTED]
        got = [(v.get('verdict'), v.get('tier')) for v in verdicts]
        assert got == [want for *_, want in self.WAS_CONVICTED]

    def test_the_noise_floor_still_refuses_a_cancellation_zero(self) -> None:
        """The other direction. The floor exists to stop `sin(pi)` -- which reads ~1e-300
        at dps 300 and ~1e-600 at dps 600 -- from being chased as a witness, and asking
        whether the target DECAYS is what tells that apart from an honestly tiny value."""
        from mpmath import mpf
        from simplipy.verify._contract import mp_polish, parse
        tl = parse(['sin', 'np.pi'], '<C_L>')
        tr = parse(['<constant>'], '<C_R>')
        assert mp_polish(tl, tr, {}, lambda: mpf(2), 0.0) == 0

    def test_an_honestly_tiny_target_gets_a_real_witness(self) -> None:
        """`pow(acos(0), -3000)` is 4.37e-589 out of one power and no subtraction at all.
        It is far below the noise floor and is not noise, and 0 is 100% wrong for it."""
        from mpmath import mp, mpf
        from simplipy.verify._contract import mp_polish, parse
        tl = parse(['pow', 'acos', '0', '<constant>'], '<C_L>')
        tr = parse(['<constant>'], '<C_R>')
        got = mp_polish(tl, tr, {}, lambda: mpf(-3000), 0.0)
        assert got != 0, 'the noise floor took 0 as the witness for 4.4e-589'
        old = mp.dps
        try:
            mp.dps = 800
            assert abs(got / (mp.pi / 2) ** mpf(-3000) - 1) < mpf('1e-100')
        finally:
            mp.dps = old

    def test_the_snap_band_is_relative_to_the_argument(self) -> None:
        """`sin(exp(-300)) = 5.1e-131` is exactly the size its own argument predicts --
        nothing cancelled. Under an absolute band it read 0 at dps 50, 0 at dps 120 and
        5.1e-131 at dps 250: a value that MOVES with the working precision."""
        from mpmath import mp, mpf
        from simplipy.verify._contract import c_eval, parse
        tree = parse(['sin', 'exp', '(-300)'], '<C_L>')
        old = mp.dps
        try:
            seen = []
            for dps in (50, 120, 250):
                mp.dps = dps
                seen.append(c_eval(tree, {}))
            assert all(v != 0 for v in seen), f'a well-resolved value snapped to 0: {seen}'
            mp.dps = 250
            # rung 1 read it at dps 50, so it is exp(-300) to about that many digits
            assert abs(seen[0] / mp.e ** mpf(-300) - 1) < mpf('1e-45')
        finally:
            mp.dps = old

    def test_a_cancellation_still_snaps(self) -> None:
        """The band still fires for every cancellation it was written for: there the
        argument is at or above 1, the predicted scale is 1, and the floor is unchanged."""
        from mpmath import mp
        from simplipy.verify._contract import c_eval, parse
        old = mp.dps
        try:
            mp.dps = 50
            assert c_eval(parse(['sin', 'np.pi'], '<C_L>'), {}) == 0
        finally:
            mp.dps = old

    def test_a_small_intermediate_sizes_the_rung_like_a_large_one(self) -> None:
        """`cos x = 1 - x^2/2 + ...` at x = exp(-300) puts the answer 261 digits below the
        1 it sits next to, so dps 50, 120 and 250 all read exactly 1.0. A magnitude of
        10^-k costs the same 2k digits as one of 10^k."""
        from mpmath import mpf
        from simplipy.verify._contract import _required_dps
        assert _required_dps(mpf(1), mpf('5.1482e-131')) == 312
        assert _required_dps(mpf('1.344e43'), None) == 138
        assert _required_dps(mpf(1), mpf(1)) == 50

    def test_a_sum_does_not_size_its_own_rung(self) -> None:
        """A `+`/`-` output may BE the cancellation residue, whose size is a fact about
        the precision and not about the expression. Sizing the next rung from it is
        circular -- the seam that rung exists to expose would confirm itself."""
        from mpmath import mp, mpf
        from simplipy.verify import _contract as C
        tree = C.parse(['-', 'cosh', '_0', 'sinh', '_0'], '<C_L>')
        old = mp.dps
        try:
            mp.dps = 50
            C._MAG_SINK.append([mpf(0), mpf(0)])
            try:
                C.c_eval(tree, {'_0': mpf(30)})
            finally:
                _mag, _small = C._MAG_SINK.pop()
            assert _small >= 1, f'a difference was taken for a small operand: {_small}'
        finally:
            mp.dps = old

    def test_past_the_ceiling_the_point_is_unresolved_not_convicted(self) -> None:
        """`MAX_DPS` documents "past the ceiling a point is UNRESOLVED -- never convicted".
        At c = -3000 `exp(c) = 1e-1303` needs dps 2656 before `cos` of it is anything but
        exactly 1.0, and the ladder read its clamped rung as the rung it asked for."""
        from mpmath import mpf
        from simplipy.verify._contract import _demanded_dps, _required_dps, MAX_DPS
        assert _demanded_dps(mpf(1), mpf('1e-1303')) > MAX_DPS
        assert _required_dps(mpf(1), mpf('1e-1303')) == MAX_DPS
        got = self._judge_at(['acos', 'cos', 'exp', '<constant>'], ['<constant>'], -3000.0)
        assert got.get('verdict') != 'KILL', got

    def test_a_written_side_is_still_convicted_past_the_ceiling(self) -> None:
        """The exemption, and the reason it is needed. A side the rule SPELLS is exact at
        every rung, so a gap against it is not one more precision could close:
        `10^sinh(-10)` is 5.8e-11014 and no closer to the written 0 than at dps 50.
        Withholding there would abstain on 26 shipped rows the judge convicts correctly."""
        from simplipy.verify._contract import judge_rule
        for lhs in (['pow', '10', 'sinh', '(-10)'], ['pow', '2', 'sinh', '(-9)'],
                    ['pow', '7', 'sinh', '(-8)']):
            got = judge_rule(lhs, ['0'])
            assert got.get('verdict') == 'KILL', (lhs, got)
            assert got.get('tier') == 'f64', (lhs, got)

    def test_the_witness_search_reaches_the_f64_ceiling(self) -> None:
        """A search that stops at a round 1e300 has no bracket for `cosh(-3)^300 =
        7.6e300`, and a rule whose witness cannot be bracketed is reported NO-WITNESS --
        a conviction, for a row true by construction."""
        import numpy as np
        from simplipy.verify._contract import _XS
        assert _XS.max() > 1e307 and _XS.min() < -1e307
        assert np.isfinite(_XS).all()


class TestTheWidenedQuantifierDoesNotConvictWhatItCannotResolve:
    """The constant battery reaches |c| = 1e4 because the miner draws from that span.
    Getting there needed two fixes in the exists-witness machinery, and both are the
    same shape as every other false conviction in this lane: the judge answered a
    question at a point where it could not read the answer.
    """

    LHS = ['pow', 'abs', '?0', '<constant>']

    def test_an_identity_survives_the_widened_battery(self) -> None:
        """The reproducer. At cl = 10000 the deployed LHS is inf at ?0 = 1.7 and at
        -1.3, and underflows to exactly 0.0 at -0.8 -- the one finite env, and the one
        the fit used to commit to. `0.8^c - 0.0` is then zero for EVERY c past ~3339, so
        the fit returned 3339.25 against a true 10000 and the judge convicted an
        IDENTITY on the difference."""
        import simplipy.verify._contract as C
        r = C.judge_rule(list(self.LHS), list(self.LHS))
        assert r['verdict'] == 'CERTIFIED', r

    def test_a_saturated_env_determines_nothing_and_is_skipped(self) -> None:
        """An equation that holds on an INTERVAL names no constant. The env is refused,
        every other env is tried, and a cl with no determinate env is a SKIP -- not a
        conviction. `skipped_cl` is exactly the tally that fact belongs in."""
        import simplipy.verify._contract as C
        tl = C.parse(self.LHS, '<C_L>')
        tr = C.parse(self.LHS, '<C_R>')
        assert C._fit_target_env(tl, tr, {'?0'}, 10000.0) is None
        # ...and the neighbouring magnitude, where one env is NOT saturated, still fits.
        assert C._fit_target_env(tl, tr, {'?0'}, -3000.0) is not None

    def test_the_search_reaches_the_constant_it_is_asked_about(self) -> None:
        """`_XS` is a fixed grid and its density collapses where the deployed algebra
        saturates: at ?0 = -0.8 the RHS overflows below c ~ -3178, so every finite grid
        point near the true witness -3000 sits on ONE side of it and the bracketing
        finds no sign change at all. The source constant is tried as a seed, which is
        the answer for every identity and costs one residual evaluation."""
        import simplipy.verify._contract as C
        tl = C.parse(self.LHS, '<C_L>')
        tr = C.parse(self.LHS, '<C_R>')
        w = C.fit_witness(tl, tr, {'?0'}, -3000.0)
        assert w == -3000.0, w

    def test_the_battery_spans_what_the_miner_draws(self) -> None:
        """The point of the widening: `POLE_GRID` runs to 500 and fitted constants are
        drawn log-uniform over [1e-3, 1e3]. A battery that stops at |c| <= 5 checks a
        quantifier the search does not obey."""
        import simplipy.verify._contract as C
        mags = sorted({abs(float(b())) for b in C.judge_cl_battery(saturation=False)}
                      - {0.0})
        assert min(mags) <= 1e-3 and max(mags) >= 1e4, mags
        assert {-1e4, -1e3, -0.001, 1e-3, 1e3, 1e4} <= {float(c) for c in C.WIDE_CONSTS}


class TestAtanGetsTheBoundaryBandTanhAlreadyHad:
    """The boundary band refuses at `tanh`/`atanh`/`acosh` where a value is inside the working
    precision's band of a boundary. `atan` never got the same treatment, and its pi/2 is
    the same kind of boundary: an ASYMPTOTE, not an ordinary point. (That distinction is
    why `asin`/`acos` deliberately do NOT get a band -- their +-1 has a finite value and
    banding it cost 67 rules.)
    """

    GUDERMANNIAN = (['cos', 'atan', 'sinh', '_0'], ['inv', 'cosh', '_0'])

    def test_the_band_refuses_where_the_information_is_gone(self) -> None:
        """At |x| = 1e4, `atan(sinh x)` rounds to pi/2 at any affordable precision -- the
        true gap is 2.2e-4343 -- and `cos` of that returns -5.05e-52: the residue of
        pi/2's own rounding, with the WRONG SIGN against a true `sech(1e4)` that is
        positive. Refuse where the information is lost, rather than propagate it."""
        from mpmath import mp, mpf
        import simplipy.verify._contract as C
        old = mp.dps
        try:
            mp.dps = 50
            with pytest.raises(C.Unresolved):
                C.c_eval(('atan', ('lit', mpf(10) ** 4400)), {})
        finally:
            mp.dps = old

    def test_the_identity_is_still_certified_and_nothing_is_convicted(self) -> None:
        """The Gudermannian `cos(atan(sinh x)) = sech(x)` is exact. The band must make
        the two extreme points ABSTAIN, never convict, and must not cost the rule its
        coverage."""
        import simplipy.verify._contract as C
        r = C.judge_rule(list(self.GUDERMANNIAN[0]), list(self.GUDERMANNIAN[1]))
        assert r['verdict'] == 'CERTIFIED', r
        assert r['measure'] == 0.0, r

    def test_the_band_narrows_with_precision_and_spares_the_shallow(self) -> None:
        """At dps 50 the band refuses only |a| > 1e40, so ordinary magnitudes are still
        judged. A band that swallowed those would trade one false conviction class for a
        coverage collapse."""
        from mpmath import mp, mpf
        import simplipy.verify._contract as C
        old = mp.dps
        try:
            mp.dps = 50
            for a in ('1', '30', '1e12', '1e30'):
                v = C.c_eval(('atan', ('lit', mpf(a))), {})
                assert abs(abs(v) - mp.pi / 2) >= mpf(10) ** (-mp.dps + 10), a
        finally:
            mp.dps = old


class TestNanIsInTheWidestSortsQuantifier:
    """`_` binds an ARBITRARY subtree, and a subtree can
    evaluate to nan on a positive-measure set of its own inputs -- `log x0` on the whole
    negative half-line. The `_` battery therefore carries a nan atom, and any
    disagreement at that binding is the fatal clause `a-nonfinite-binding`, not a tolerated
    extension: EXT there is nan-ERASURE, SHRINK is nan-introduction. The other sorts
    keep their own licences -- `?` is the data sort (nan outside the quantifier), and
    whether `!`/`$` owe nan the same treatment is a separate, unruled question.
    """

    def test_nan_erasure_at_a_wildcard_binding_is_fatal(self) -> None:
        """`- _0 _0 -> 0` erases nan for every nan-valued subtree; without the nan-erasure rule
        it judged TOLERATED/real and `verify_ruleset` called it clean."""
        r = judge_rule('- _0 _0'.split(), ['0'])
        assert r['verdict'] == 'KILL', r
        assert r['clause'] == 'a-nonfinite-binding', r
        assert r['kinds'] == ['EXT'], r

    def test_nan_arising_at_a_real_binding_is_judged_by_its_guard(self) -> None:
        """`atan tan asin ?0 -> asin ?0`: the nan lives on the singular manifold of a REAL
        binding (?0 = +-1), a null set a `?` slot guards -- TOLERATED, because the interception
        keys on the nan ATOM binding, never on nan appearing during evaluation. Spelled with a
        `_` slot the same null set is unguarded (bound to floor(x) + 1 the slot is 1 on all of
        [0, 1)): refused."""
        r = judge_rule('atan tan asin ?0'.split(), 'asin ?0'.split())
        assert r['verdict'] == 'TOLERATED', r
        assert r['classes'] == ['EXT'], r
        r = judge_rule('atan tan asin _0'.split(), 'asin _0'.split())
        assert (r['verdict'], r['clause']) == ('KILL', 'b-unguarded'), r

    def test_strict_wildcard_identities_are_unmoved(self) -> None:
        """nan propagates through strict identities: both sides read nan at the atom,
        compare 'eq', and the verdict cannot move."""
        for lhs, rhs in (('/ 1 abs _0', 'abs inv _0'),
                         ('rootn pow _0 3 3', '_0'),
                         ('inv abs abs _0', 'inv abs _0')):
            r = judge_rule(lhs.split(), rhs.split())
            assert r['verdict'] == 'CERTIFIED', (lhs, r)

    def test_the_data_sorts_keep_their_own_licences(self) -> None:
        """`- ?0 ?0 -> 0` stays CERTIFIED (data slots never carry nan) and so does
        `- !0 !0 -> 0`: its +-inf atoms are the values its certificate puts on a null set,
        owed nothing, and the nan-erasure rule deliberately does not extend to `!`."""
        r = judge_rule('- ?0 ?0'.split(), ['0'])
        assert r['verdict'] == 'CERTIFIED', r
        r = judge_rule('- !0 !0'.split(), ['0'])
        assert r['verdict'] == 'CERTIFIED', r


class TestTheGateAgreesWithTheRouterAndTheClock:
    """Two ways the gate's verdict could depend on something other than the rules. (a) A
    rule with more than one `<constant>` on its left side was one the judge could not
    decide; it is decided now, by `verify_ruleset` as by every other gate. (b) JUDGE-TIMEOUT
    is fatal and was decided by a single 30 s wall-clock
    alarm -- a fact about the box's load, not the rule -- so the artifact was
    non-reproducible under load; timed-out rules are now retried serially at 4x after the
    sweep."""

    def test_a_rule_with_two_left_constants_is_judged(self) -> None:
        """Each `<constant>` is its own symbol (the specification's 3.2), so a rule with two
        on its left side is decided like any other: `C1 + C2*x -> d1*x + d2` holds with
        d1 = C2, d2 = C1, and `pow(rootn(x, C1), C2) -> pow(x, C3)` fails at C1 = 3, C2 = 1
        on x < 0. (The mine and the load take such rules where `licensed` does.)"""
        from simplipy.verify import effective_tier, verify_rule, verify_ruleset
        true = [['+', '<constant>', '*', '<constant>', 'x0'],
                ['+', '*', '<constant>', 'x0', '<constant>']]
        false = [['pow', 'rootn', '?0', '<constant>', '<constant>'], ['pow', '?0', '<constant>']]
        r = verify_rule(*true)
        assert (r['verdict'], effective_tier(r)) == ('CERTIFIED', 'core'), r
        r = verify_rule(*false)
        assert (r['verdict'], effective_tier(r)) == ('KILL', 'reject'), r
        for mode in (None, 'f64', 'real', 'permissive'):
            assert verify_ruleset([true], mode=mode)['is_clean'] is True, mode
            assert verify_ruleset([false], mode=mode)['is_clean'] is False, mode
        # a false rule stays dirty everywhere too
        bad = [[['rootn', 'pow', '?0', '2', '2'], ['?0']]]
        for mode in (None, 'f64'):
            assert verify_ruleset(bad, mode=mode)['is_clean'] is False, mode

    def test_a_timed_out_rule_is_retried_serially(self, monkeypatch) -> None:
        from simplipy.verify import _gate
        calls = {'n': 0}
        real_judge = _gate.judge_rule

        def flaky(lhs, rhs):
            calls['n'] += 1
            if calls['n'] == 1:
                raise _gate._JudgeTimeout()   # the load spike, deterministically
            return real_judge(lhs, rhs)

        monkeypatch.setattr(_gate, 'judge_rule', flaky)
        code = _gate.sweep([(['neg', 'neg', '?0'], ['?0'])])
        assert calls['n'] == 2, "the timed-out rule must be re-judged"
        assert code == 0, "and the retry's CERTIFIED verdict must replace the timeout"

    def test_a_double_timeout_stays_fatal(self, monkeypatch) -> None:
        from simplipy.verify import _gate

        def hung(lhs, rhs):
            raise _gate._JudgeTimeout()

        monkeypatch.setattr(_gate, 'judge_rule', hung)
        code = _gate.sweep([(['neg', 'neg', '?0'], ['?0'])])
        assert code != 0, "fail-closed: a rule that times out twice is held out"


def test_the_self_tests_speak_the_live_vocabulary() -> None:
    # The touchstones and the gate's poisons exercise the spelling a 0.15 artifact uses
    # (`pow`, `rootn`, `*`), not the retired hyper-operators.
    from simplipy.verify._contract import TOUCHSTONES
    from simplipy.verify._gate import POISONS, PHANTOMS
    retired = {f'{f}{k}' for f in ('pow', 'mult', 'div', 'pow1_') for k in range(2, 6)}
    for lhs, rhs, *_ in [*TOUCHSTONES, *POISONS, *PHANTOMS]:
        tokens = [*(lhs.split() if isinstance(lhs, str) else lhs),
                  *(rhs.split() if isinstance(rhs, str) else rhs)]
        assert not retired & set(tokens), (lhs, rhs)


def test_the_legacy_odd_roots_judge_like_rootn() -> None:
    # The deployed lane's retired `pow1_3`/`pow1_5` are the odd roots `rootn(x, 3)`/`rootn(x, 5)`, the sign of zero
    # included (IEEE-754 rootn: the root of -0 is -0), so a legacy rule is judged on the semantics `rootn` has
    # (THE TABLE's f64 values: the engine's own kernels).
    from simplipy.verify import _contract as C

    def f64(tokens, x):
        return float(C.d_eval(C.parse(tokens), {'?0': x}))

    for legacy, index in (('pow1_3', '3'), ('pow1_5', '5')):
        for x in (-8.0, -0.0, 0.0, 32.0, -math.inf, math.inf):
            got, want = f64([legacy, '?0'], x), f64(['rootn', '?0', index], x)
            assert math.isclose(got, want, rel_tol=1e-15) or got == want, (legacy, x, got, want)
            assert math.copysign(1.0, got) == math.copysign(1.0, want), (legacy, x, got, want)
        assert math.isnan(f64([legacy, '?0'], math.nan))
        assert math.copysign(1.0, f64([legacy, '?0'], -0.0)) == -1.0, legacy


# ------------------------------------------------------------------------------------------
# A verdict does not depend on the machine: the multi-constant witness fit reads no BLAS.
# ------------------------------------------------------------------------------------------
_MULTI_FIT = '''
from simplipy.verify import _contract as C
tl = C.parse('pow + ?0 <constant> 2'.split(), '<C_L>', number=True)
tr = C.parse('+ + pow ?0 2 * <constant> ?0 <constant>'.split(), '<C_R>', number=True)
for c in (0.0, -1.5, 3.0, 4.0, 10.0, -100.0, 1000.0):
    print(repr(C.fit_witness_multi(tl, tr, {'?0'}, {'<C_L0>': c}, ['<C_R0>', '<C_R1>'])))
'''


@pytest.mark.parametrize('lhs, rhs', [
    ('pow + ?0 <constant> 2', '+ + pow ?0 2 * <constant> ?0 <constant>'),
    ('pow - ?0 <constant> 2', '+ - pow ?0 2 * <constant> ?0 <constant>'),
    ('/ + ?0 <constant> <constant>', '+ * <constant> ?0 <constant>'),
])
def test_a_multi_constant_identity_is_certified(lhs, rhs) -> None:
    # The fit at <C_L0> = 0 returned the noise -2.5e-17 for an exact 0 (a coordinate snap built
    # from the unsnapped fit undid the joint snap on a tie), and the polish took it to ~1e-557:
    # a nonzero right side at x = 0, where the left side is exactly 0, convicted the identity.
    got = v.verify_rule(lhs.split(), rhs.split())
    assert got['verdict'] == 'CERTIFIED' and v.effective_tier(got) == 'real', got


def test_the_multi_constant_fit_is_the_same_on_every_blas_kernel() -> None:
    # The CI's py3.12 and py3.13 jobs disagreed on `(x + c)^2 -> x^2 + c x + c` because the two
    # runners dispatched different OpenBLAS kernels: `@`, `np.linalg.norm` and `np.linalg.solve`
    # sum in the kernel's order, and the fit's last bits followed. Forcing the kernel reproduces
    # it on one machine; the fit's own arithmetic (exactly rounded sums, a plain-float solve)
    # gives the same bits under every kernel.
    import platform
    import subprocess
    import sys
    if platform.machine() not in ('x86_64', 'AMD64'):
        pytest.skip('the forced OpenBLAS kernels are x86 kernels')
    outs = set()
    for core in ('Haswell', 'Sandybridge', 'Prescott'):
        env = {**os.environ, 'OPENBLAS_CORETYPE': core}
        out = subprocess.run([sys.executable, '-c', _MULTI_FIT], env=env, capture_output=True,
                             text=True, check=True).stdout
        outs.add(out)
    assert len(outs) == 1, outs
    assert '[0.0, 0.0]' in outs.pop()


# ------------------------------------------------------------------------------------------
# The monitor reads F8: each rule step in the mode, the engine's own algebra exact in the reals.
# ------------------------------------------------------------------------------------------
def test_the_monitor_holds_each_rule_step_to_the_mode() -> None:
    import yaml
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    eng = SimpliPyEngine(operators=ops, rules=[(('sin', 'acos', '?0'), ('cos', 'asin', '?0'))])
    out, fires = eng._core.ac_simplify_fires(['sin', 'acos', 'x0'], rule_mode='default')
    assert fires == [(['sin', 'acos', 'x0'], ['cos', 'asin', 'x0'])], fires
    # true in the reals, and not in f64 (at 1: 0 against cos(fl(pi/2)) = 6.1e-17): the step
    verdict, detail = judge_pair(['sin', 'acos', 'x0'], out, engine=eng)
    assert verdict == 'VIOLATION' and 'the step' in detail, detail
    assert judge_pair(['sin', 'acos', 'x0'], out, engine=eng, mode='real')[0] == 'OK'


def test_the_monitor_passes_exact_algebra_float64_rounds_apart() -> None:
    # F8: no promise of 8 units in the last place for a whole answer, only the engine's algebra
    # exact in the reals; a zero's sign is float64's own event and is held (F7)
    import yaml
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    eng = SimpliPyEngine(operators=ops, rules=[])
    for src in ('/ x0 / x0 - pow x0 2 x0', 'tan + x0 np.pi'):
        out = eng.simplify(src.split())
        assert judge_pair(src.split(), out, engine=eng)[0] == 'OK', (src, out)
    src = 'inv + 0 neg pow x0 4'.split()
    verdict, detail = judge_pair(src, eng.simplify(src), engine=eng)
    assert verdict == 'VIOLATION' and 'INF-CHANGE' in detail, detail
