"""THE DIFFERENTIAL TEST: every step against the one semantics.

The specification (one table of values per mode, one criterion `holds(L, R, mode)`, both owned
by the judge) makes every other step of the miner and the engine one of three kinds: a
pre-filter that can only reject what the judge rejects, a named policy applied the same way
everywhere, or a reader of the table. This module checks each kind against the judge, on fixed
inputs:

1. pre-filters (the mint, promotion's sort bars): a rejection must be a judge rejection;
2. the mint: a rule it accepts must be one the judge admits in the mode;
3. policies (ground composites, evidence, the constant-collapse licence): every step that applies
   one gives the same answer;
4. the table: each mode's values at fixed probes, and every consumer that folds or encloses a
   value agrees with them;
5. the judge's verdict on the specification's examples (sections 2 and 3 of the specification);
6. the judge decides every pair: none is left unjudged.

A test marked `expected_failure` fails today on a known defect, named by its id (`DEFECTS`) and
the phase that fixes it. The marks are strict: when a fix lands, its tests pass, the mark makes
them fail, and the fix removes the mark.

"Admits" is the judge's word for "this mode's file may hold the rule by its truth": `real`
admits what holds in real, `f64` what holds in f64, and `permissive` what holds in permissive
or in f64 (the specification's P3). Policies (evidence, ground composites) are not part of it:
`simplipy.verify.licensed` adds them (the specification's 3.4).
"""
import functools
import os

import numpy as np
import pytest
import yaml
from mpmath import mp

from simplipy import SimpliPyEngine
from simplipy.verify import MODE_TIERS, effective_tier, licensed, verify_rule
from simplipy.verify import _contract

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ('real', 'f64', 'permissive')
RULE_MODE = {'f64': 'default', 'real': 'real', 'permissive': 'permissive'}

#: The known defects, each fixed by one phase of the plan.
DEFECTS = {
    'X1': "promotion's sort bars check +-inf for `?` data and ignore the mode",
    'X2': "a point the table decides is left undecided, or read off float64's value",
    'X3': "a policy (ground composite, evidence) applied in the mint only, not in the overlap gate",
    'X4': 'the judge tolerates extension on a ground rule, whose one point is its whole domain',
    'X5': 'rules with several constants on the left are not judged',
    'X6': 'constant collapse: two licences in the engine, the mint stricter than the judge',
    'N1': 'positive measure read as more than 5% of a sample grid',
    'N2': 'the f64 check of the 0.15 operators probes data variables at +-inf',
    'N3': '`_` excused at +-inf',
    'N6': 'real reads 1/0 and its kin as +inf, against the limit rule (V4)',
    'N7': 'real reads 1^inf, inf^0 and NaN^0 as 1, against the limit rule (V5) and V6',
    'N8': "the judge's f64 reading of a written -0.0 is +0 (a literal is its nearest double, F2)",
    'N9': 'the interval enclosure of a two-sided pole omits the undefined value',
    'N10': "the mint's f64 rows agree to a relative 1e-11, looser than f64's 8 units (F6)",
    'R1': "permissive's criterion is clause (a) alone (the owner's ruling 1), not real's",
}


def expected_failure(defect: str, phase: str):
    assert defect in DEFECTS, defect
    return pytest.mark.xfail(strict=True,
                             reason=f'{defect}: {DEFECTS[defect]} (fixed in phase {phase})')


def case(lhs: str, rhs: str, mode: str, expected, *marks):
    return pytest.param(lhs, rhs, mode, expected, marks=list(marks),
                        id=f'{lhs} -> {rhs} [{mode}]')


@functools.lru_cache(maxsize=None)
def judged(lhs: str, rhs: str) -> dict:
    return verify_rule(lhs.split(), rhs.split())


def admits(lhs: str, rhs: str, mode: str) -> bool:
    return effective_tier(judged(lhs, rhs)) in MODE_TIERS[mode]


@functools.lru_cache(maxsize=1)
def bare() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])


@functools.lru_cache(maxsize=1)
def mine_rows() -> np.ndarray:
    from simplipy.mining import RuleMiner
    return RuleMiner(bare())._mining_sample_x(512, 2, np.random.default_rng(0))


def mint_accepts(lhs: str, rhs: str, mode: str) -> bool:
    """The mint's pair judgment on the mine's own rows, in the mode."""
    X = mine_rows()
    ok, _ = bare()._core.judge_pair(lhs.split(), rhs.split(), ['x0', 'x1'], X.flatten().tolist(),
                                    X.shape[0], rtol=1e-11, atol=1e-12,
                                    rule_mode=RULE_MODE[mode])
    return ok


# ------------------------------------------------------------------------------------------
# Assertion 5: the judge gives every example the specification's verdict.
# ------------------------------------------------------------------------------------------
SPEC_EXAMPLES = [
    # section 3.1: rules with variables
    case('/ x0 x0', '1', 'real', True), case('/ x0 x0', '1', 'f64', True),
    case('/ x0 x0', '1', 'permissive', True),
    case('exp log x0', 'x0', 'real', False), case('exp log x0', 'x0', 'f64', False),
    case('exp log x0', 'x0', 'permissive', True),
    case('- log x0 log x0', '0', 'real', False), case('- log x0 log x0', '0', 'f64', False),
    case('- log x0 log x0', '0', 'permissive', True),
    case('atanh tanh x0', 'x0', 'real', True), case('atanh tanh x0', 'x0', 'f64', False),
    case('atanh tanh x0', 'x0', 'permissive', True),
    case('asin 1e-8', '1e-8', 'real', False), case('asin 1e-8', '1e-8', 'f64', True),
    case('asin 1e-8', '1e-8', 'permissive', True),
    case('rootn pow x0 2 2', 'abs x0', 'real', True),
    case('rootn pow x0 2 2', 'abs x0', 'f64', True),
    case('rootn pow x0 2 2', 'abs x0', 'permissive', True),
    case('rootn pow x0 2 2', 'x0', 'real', False), case('rootn pow x0 2 2', 'x0', 'f64', False),
    case('rootn pow x0 2 2', 'x0', 'permissive', False),
    case('tan asin ceil x0', 'atanh ceil x0', 'real', False),
    case('tan asin ceil x0', 'atanh ceil x0', 'f64', False),
    case('tan asin ceil x0', 'atanh ceil x0', 'permissive', True),
    case('ceil inv cosh x0', '1', 'real', True),
    case('ceil inv cosh x0', '1', 'f64', True),
    case('ceil inv cosh x0', '1', 'permissive', True),
    # the 5% grid bar: an extension on (-0.1, 0.1) is positive measure
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'real', False),
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'f64', False),
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'permissive', True),
    # the owner's rulings: isolated infinities (c), the guard, coincident slots, undecided
    case('abs log cos x0', 'log inv cos x0', 'real', True),
    case('abs log cos _0', 'log inv cos _0', 'real', False),
    case('atan tan asin _0', 'asin _0', 'real', False),
    case('atan tan asin x0', 'asin x0', 'real', True),
    case('/ - x0 x1 - x1 x0', '-1', 'real', False),
    case('/ - x0 x1 - x1 x0', '-1', 'f64', False),
    case('/ - x0 x1 - x1 x0', '-1', 'permissive', True),
    case('/ !0 !0', '1', 'real', False),
    # section 2: the sorts
    case('- ?0 ?0', '0', 'real', True), case('- ?0 ?0', '0', 'f64', True),
    case('- !0 !0', '0', 'real', True), case('- !0 !0', '0', 'f64', True),
    case('/ $0 $0', '1', 'real', True), case('/ $0 $0', '1', 'f64', True),
    case('- _0 _0', '0', 'real', False), case('- _0 _0', '0', 'f64', False),
    case('- _0 _0', '0', 'permissive', True),
    # `_` at +inf: 0*tanh(inf) = 0 has a value, 0*inf has none
    case('* 0 tanh _0', '* 0 _0', 'real', False),
    case('* 0 tanh _0', '* 0 _0', 'f64', False),
    # section 3.2: constants
    case('acos <constant>', '<constant>', 'real', True),
    case('acos <constant>', '<constant>', 'f64', True),
    case('exp + x0 <constant>', '* <constant> exp x0', 'real', True),
    case('* <constant> exp x0', 'exp + x0 <constant>', 'real', False),
    case('* <constant> exp x0', 'exp + x0 <constant>', 'f64', False),
    case('pow rootn x0 <constant> <constant>', 'pow x0 <constant>', 'real', False),
    case('pow rootn x0 <constant> <constant>', 'pow x0 <constant>', 'f64', False),
    case('pow rootn x0 <constant> <constant>', 'pow x0 <constant>', 'permissive', False),
    case('* exp + x0 <constant> <constant>', '* <constant> exp x0', 'real', True),
    case('pow 0 <constant>', '<constant>', 'real', True),
    case('pow 0 <constant>', '<constant>', 'f64', True),
    case('pow -1 <constant>', '<constant>', 'real', True),
    case('pow -1 <constant>', '1', 'real', False), case('pow -1 <constant>', '1', 'f64', False),
    # section 3.3: ground rules
    case('cos np.pi', '-1', 'real', True), case('cos np.pi', '-1', 'f64', True),
    case('sin np.pi', '0', 'real', True), case('sin np.pi', '0', 'f64', False),
    case('sin np.pi', '0', 'permissive', True),
    case('log -1', '0', 'real', False),
    case('log -1', '0', 'f64', False),
    case('log -1', '0', 'permissive', True),
    case('exp tan asin 1', 'float("inf")', 'real', False),
    case('exp tan asin 1', 'float("inf")', 'f64', True),
    case('exp tan asin 1', 'float("inf")', 'permissive', True),
    case('asin 1', 'acos 0', 'real', True), case('asin 1', 'acos 0', 'f64', True),
    # the limit rule (V4, V5, V6)
    case('/ 1 0', 'float("inf")', 'real', False),
    case('/ 1 0', 'float("inf")', 'f64', True),
    case('/ 1 0', 'float("inf")', 'permissive', True),
    case('pow 0 -1', 'float("inf")', 'real', False),
    case('pow 0 -2', 'float("inf")', 'real', True),
    case('pow 1 float("inf")', '1', 'real', False),
    case('pow 1 float("inf")', '1', 'f64', True),
    # P3 as ruled: permissive serves real's values up to float resolution, except at isolated
    # points -- a deep saturation (1 - 2e-9566 is the double 1.0), a disagreement at x = +-1
    # only; not float64's own values at a point a `_` slot does not guard
    case('tanh cosh (-10)', '1', 'permissive', True),
    case('tanh sinh atanh x0', 'tanh tan asin x0', 'permissive', True),
    case('tanh sinh atanh _0', 'tanh tan asin _0', 'permissive', False),
    case('pow 0 asinh _0', 'pow 0 _0', 'f64', True),
    case('pow 0 asinh _0', 'pow 0 _0', 'permissive', False),
]


@pytest.mark.parametrize('lhs, rhs, mode, expected', SPEC_EXAMPLES)
def test_the_judge_gives_the_specifications_verdict(lhs, rhs, mode, expected) -> None:
    assert admits(lhs, rhs, mode) is expected, judged(lhs, rhs)


#: Section 4: rules that hold and are refused by a named policy (`licensed` = holds and every
#: policy), each with the policy the specification names.
POLICY_EXAMPLES = [
    ('asin 1', 'acos 0', 'f64', 'ground_composite'),
    ('asin 1', 'acos 0', 'real', 'ground_composite'),
    ('exp tan asin 1', 'float("inf")', 'f64', 'evidence'),
    ('pow -1 <constant>', '<constant>', 'real', 'evidence'),
    ('abs x0', 'rootn pow x0 2 2', 'real', 'ordering'),
    ('exp * 2 _0', '* exp _0 exp _0', 'real', 'duplication'),
    ('+ * x0 x1 * x0 x2', '* x0 + x1 x2', 'real', 'engine_owned'),
    ('log -3', 'float("nan")', 'real', 'engine_owned'),
    # permissive cancels an inverse pair itself (4.9), so the rule is the engine's
    ('exp log x0', 'x0', 'permissive', 'engine_owned'),
    ('pow + x0 <constant> 2', '+ + pow x0 2 * <constant> x0 <constant>', 'real',
     'constant_count'),
]


@pytest.mark.parametrize('lhs, rhs, mode, policy', POLICY_EXAMPLES)
def test_a_named_policy_refuses_what_holds(lhs, rhs, mode, policy) -> None:
    got = licensed(lhs.split(), rhs.split(), mode, bare(), detail=judged(lhs, rhs))
    assert got['holds'] and not got['licensed'] and policy in got['refused_by'], got


@pytest.mark.parametrize('lhs, rhs, mode', [
    ('exp + log x0 log x1', '* x0 x1', 'permissive'), ('acos <constant>', '<constant>', 'real'),
    ('exp + x0 <constant>', '* <constant> exp x0', 'real'), ('ceil inv cosh x0', '1', 'f64'),
])
def test_licensed_is_holds_and_every_policy(lhs, rhs, mode) -> None:
    got = licensed(lhs.split(), rhs.split(), mode, bare(), detail=judged(lhs, rhs))
    assert got == {'licensed': True, 'holds': True, 'tier': got['tier'], 'refused_by': []}, got


# ------------------------------------------------------------------------------------------
# Assertion 6: the judge decides every pair.
# ------------------------------------------------------------------------------------------
UNDECIDED = {'UNSUPPORTED-SHAPE', 'UNRESOLVED-COVERAGE', 'UNDECIDED-MEASURE', 'JUDGE-TIMEOUT'}


@pytest.mark.parametrize('lhs, rhs', [
    ('pow rootn x0 <constant> <constant>', 'pow x0 <constant>'),
    ('* exp + x0 <constant> <constant>', '* <constant> exp x0'),
    ('+ <constant> * <constant> x0', '+ * <constant> x0 <constant>'),
    ('- <constant> <constant>', '0'), ('sin * x0 * 1e15 + 1 1e-20', 'sin * x0 1e15'),
    ('tan asin ceil x0', 'atanh ceil x0'), ('exp tan asin 1', 'float("inf")'),
    ('acos <constant>', '<constant>'), ('* 0 tanh _0', '* 0 _0'),
])
def test_the_judge_decides_every_pair(lhs, rhs) -> None:
    assert judged(lhs, rhs)['verdict'] not in UNDECIDED, judged(lhs, rhs)


# ------------------------------------------------------------------------------------------
# Assertion 4: the table, and its consumers.
# ------------------------------------------------------------------------------------------
def real_value(expr: str) -> str:
    """The judge's real value of a ground expression, as 'nan', '+inf', '-inf' or a float."""
    tree = _contract.parse(expr.split())
    old = mp.dps
    try:
        mp.dps = 50
        v = _contract.c_eval(tree, {})
    finally:
        mp.dps = old
    if v != v:
        return 'nan'
    if v in (mp.inf, -mp.inf):
        return '+inf' if v > 0 else '-inf'
    return repr(float(v))


def literal_reading(tokens: list[str]) -> str | None:
    """A folded answer's value as `real_value` spells it; None when it is not one literal."""
    if len(tokens) != 1:
        return None
    t = tokens[0]
    if t in ('float("nan")', 'nan'):
        return 'nan'
    if t in ('float("inf")', 'inf'):
        return '+inf'
    if t in ('float("-inf")', '-inf'):
        return '-inf'
    try:
        return repr(float(_contract.literal_value(t)))
    except Exception:
        return None


REAL_TABLE = [
    ('inv 0', 'nan', 'N6'), ('pow 0 -1', 'nan', 'N6'), ('rootn 0 -3', 'nan', 'N6'),
    ('pow 0 -2', '+inf', None), ('pow 0 -0.5', '+inf', None), ('rootn 0 -2', '+inf', None),
    ('log 0', '-inf', None), ('atanh 1', '+inf', None), ('atanh -1', '-inf', None),
    ('tan asin 1', 'nan', None), ('exp float("-inf")', '0.0', None),
    ('tanh float("inf")', '1.0', None), ('atan float("inf")', repr(np.pi / 2), None),
    ('sin float("inf")', 'nan', None), ('- float("inf") float("inf")', 'nan', None),
    ('* 0 float("inf")', 'nan', None), ('pow 1 float("inf")', 'nan', 'N7'),
    ('pow float("inf") 0', 'nan', 'N7'), ('pow float("nan") 0', 'nan', 'N7'),
    ('pow 1 float("nan")', 'nan', 'N7'), ('pow 0 0', '1.0', None),
    ('pow -8 / 1 3', 'nan', None), ('rootn -8 3', '-2.0', None), ('neg 0', '0.0', None),
    ('asin 2', 'nan', None), ('log -1', 'nan', None), ('acosh 1', '0.0', None),
    ('sin np.pi', '0.0', None), ('cos np.pi', '-1.0', None),
]


@pytest.mark.parametrize('expr, expected', [
    pytest.param(e, v, id=e) for e, v, _ in REAL_TABLE])
def test_the_table_reads_real_values(expr, expected) -> None:
    assert real_value(expr) == expected


@pytest.mark.parametrize('expr, expected', [
    pytest.param(e, v, id=e) for e, v, _ in REAL_TABLE])
def test_the_real_fold_agrees_with_the_table(expr, expected) -> None:
    """A ground expression the real engine folds to a literal folds to the table's value (it may
    also leave it as written)."""
    folded = literal_reading(list(bare().simplify(expr.split(), mode='real')))
    assert folded is None or folded == expected


F64_TABLE = [
    ('inv 0', np.inf), ('inv -0.0', -np.inf), ('pow 0 -1', np.inf), ('log 0', -np.inf),
    ('atanh 1', np.inf), ('tan asin 1', 1.633123935319537e16), ('tanh 20', 1.0),
    ('atanh tanh 20', np.inf), ('pow 1 float("inf")', 1.0), ('neg 0', -0.0),
    ('+ 0.1 0.2', 0.30000000000000004), ('sin np.pi', 1.2246467991473532e-16),
]


@pytest.mark.parametrize('expr, expected', [
    pytest.param(e, v, id=e) for e, v in F64_TABLE])
def test_f64_reads_the_evaluator(expr, expected) -> None:
    v = float(_contract.d_eval(_contract.parse(expr.split()), {}))
    assert v == expected and np.signbit(v) == np.signbit(expected)


@pytest.mark.parametrize('expr, expected', [
    ('inv 0', 'nan'), ('pow 0 -1', 'nan'), ('pow 0 -2', '+inf'), ('log 0', '-inf'),
    ('tan asin 1', 'nan'), ('pow 1 float("inf")', 'nan'), ('pow float("nan") 0', 'nan'),
    ('rootn 0 -3', 'nan'), ('tanh float("inf")', '1.0'), ('exp float("-inf")', '0.0')])
def test_interval_enclosures_contain_the_table_values(expr, expected) -> None:
    """The interval layer's real reading of a ground expression contains the table's value."""
    got = bare()._core.interval_value_set_box(expr.split(), [0.0], [0.0], real=True)
    assert got is not None
    has_fin, pinf, ninf, nan, lo, hi = got
    if expected in ('nan', '+inf', '-inf'):
        assert {'nan': nan, '+inf': pinf, '-inf': ninf}[expected], got
    else:
        assert has_fin and lo <= float(expected) <= hi, got


@pytest.mark.parametrize('expr', ['acosh tanh x0', 'acosh erf x0'])
def test_the_real_reading_has_no_float64_saturation(expr) -> None:
    """Past 19.06 float64's tanh is exactly 1 (erf past 5.92), the reals' never is: so
    `acosh(tanh(x0))` is finite (0) on a ray in float64's reading and undefined everywhere in
    the reals', where it has no evidence."""
    from simplipy import _core
    assert _core.table_evidence(expr.split(), dict(_contract.ARITY), True) is True
    assert _core.table_evidence(expr.split(), dict(_contract.ARITY), False) is False


@pytest.mark.parametrize('rule', ['rootn -1 tanh x0'])
def test_rootn_over_an_expression_index_is_read(rule) -> None:
    """An index that reaches no nonzero integer makes the root undefined at every radicand: in
    the reals `tanh(x0)` stays inside (-1, 1)."""
    got = bare()._core.interval_value_set_box(rule.split(), [-50.0], [50.0], real=True)
    assert got is not None and got[3] and not got[0], got


# ------------------------------------------------------------------------------------------
# Assertion 1: a pre-filter rejects only what the judge rejects.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lhs, rhs, mode', [
    ('tan asin ceil x0', 'atanh ceil x0', 'real'),
    ('tan asin ceil x0', 'atanh ceil x0', 'permissive'),
    ('pow 0 <constant>', '<constant>', 'real'),
    ('pow 0 <constant>', '<constant>', 'f64'),
    ('exp <constant>', '<constant>', 'f64'),
    # the mint's f64 bar reads data as finite (no probe at +-inf), and permissive's mint owes no
    # clause (b)
    ('ceil inv cosh x0', '1', 'f64'),
    ('exp log x0', 'x0', 'permissive'),
    ('rootn pow x0 2 2', 'x0', 'real'), ('exp log x0', 'x0', 'f64'),
    # rows the table cannot read are skipped, as the judge skips them (erf(sinh(-100)) is
    # -1 + e^-(1.7e86)); the special points check float64's divergence in f64 only
    ('ceil erf sinh x0', 'ceil erf x0', 'real'), ('floor erf sinh x0', 'floor erf x0', 'real'),
    # the judge's isolated infinity: +inf against tan's pole at x0 = 1
    ('sinh atanh x0', 'tan asin x0', 'real'),
    # a value float64 gives off a rounded zero is read by the table: tan(acos(-1)) is 0
    ('tan acos floor x0', 'inv atanh floor x0', 'real'),
    # permissive excuses a disagreement at an isolated point (P3): the battery's points, the
    # exceptional points and a sampled row on one
    ('abs ceil sin x0', 'ceil sin x0', 'permissive'), ('ceil abs sin x0', '1', 'permissive'),
    ('inv atanh ceil x0', '0', 'permissive'),
])
def test_a_mint_rejection_is_a_judge_rejection(lhs, rhs, mode) -> None:
    assert mint_accepts(lhs, rhs, mode) or not admits(lhs, rhs, mode), judged(lhs, rhs)


@pytest.mark.parametrize('lhs, rhs, mode', [
    ('ceil inv cosh ?0', '1', 'real'), ('ceil inv cosh ?0', '1', 'f64'),
    ('- ?0 ?0', '0', 'real'), ('/ ?0 ?0', '1', 'real'),
])
def test_a_promotion_rejection_is_a_judge_rejection(lhs, rhs, mode) -> None:
    """Promotion keeps a rule the judge admits at its own sort (it may ship it at a stronger
    one, or prune it where the engine derives it without the rule), and every sort it ships
    is one the judge admits."""
    from simplipy.promotion import promote
    unsort = lambda ts: ['?' + t[1:] if t[:1] in '_!$' and t[1:].isdigit() else t  # noqa: E731
                         for t in ts]
    kept, _ = promote([(lhs.split(), rhs.split())], bare(), rule_mode=RULE_MODE[mode],
                      run_positive_controls=False)
    shipped = [(lhs_k, rhs_k) for lhs_k, rhs_k in kept if unsort(lhs_k) == lhs.split()]
    derived = bare().simplify(lhs.replace('?0', 'x0').split(), mode=mode) == rhs.split()
    assert shipped or derived or not admits(lhs, rhs, mode), kept
    for lhs_k, rhs_k in shipped:
        assert admits(' '.join(lhs_k), ' '.join(rhs_k), mode), (lhs_k, rhs_k)


# ------------------------------------------------------------------------------------------
# Assertion 2: a rule the mint accepts is one the judge admits.
# ------------------------------------------------------------------------------------------
MINT_SOURCES = [
    'inv inv x0', 'neg neg x0', 'abs abs x0', 'abs neg x0', 'exp log x0', 'log exp x0',
    'atanh tanh x0', 'tanh atanh x0', 'asin sin x0', 'sin asin x0', 'acos cos x0',
    'cos acos x0', 'atan tan x0', 'tan atan x0', 'asinh sinh x0', 'sinh asinh x0',
    'acosh cosh x0', 'cosh acosh x0', 'rootn pow x0 2 2', 'pow rootn x0 2 2',
    'rootn pow x0 3 3', 'pow rootn x0 3 3', '/ x0 x0', '- x0 x0', '* x0 0', '+ x0 0',
    'ceil floor x0', 'floor ceil x0', 'floor floor x0', 'ceil inv cosh x0', 'exp neg log x0',
    'log * x0 x0', 'acos <constant>', 'exp <constant>', 'log abs <constant>',
    'pow 0 <constant>', 'tan asin ceil x0', 'sin * np.pi floor x0', 'cos * np.pi floor x0',
]
MINT_TARGETS = [
    '0', '1', '-1', 'x0', 'neg x0', 'abs x0', 'inv x0', 'pow x0 2', 'rootn x0 2', 'exp x0',
    'log x0', 'log abs x0', 'floor x0', 'ceil x0', 'atanh ceil x0', 'float("nan")',
    '<constant>', 'pow -1 floor x0', '* 2 log abs x0', '* 2 log x0',
]


@functools.lru_cache(maxsize=None)
def mint_picks(mode: str) -> tuple:
    """The mint's first certified target per source, over a fixed candidate list."""
    picks = []
    for src in MINT_SOURCES:
        for tgt in MINT_TARGETS:
            if tgt == src or len(tgt.split()) >= len(src.split()):
                continue
            if mint_accepts(src, tgt, mode):
                picks.append((src, tgt))
                break
    return tuple(picks)


@pytest.mark.parametrize('mode', [
    'real', 'f64', 'permissive'])
def test_a_mint_acceptance_is_a_judge_admission(mode) -> None:
    refused = [(s, t, judged(s, t)['verdict']) for s, t in mint_picks(mode)
               if not admits(s, t, mode)]
    assert not refused, refused


# ------------------------------------------------------------------------------------------
# Assertion 3: a policy gives the same answer from every step that applies it.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lhs, rhs, mode', [
    ('asin 1', 'acos 0', 'f64'), ('asin 1', 'acos 0', 'real'),
    ('exp tan asin 1', 'float("inf")', 'real'),
    ('rootn pow x0 2 2', 'x0', 'real'),
])
def test_the_overlap_gate_demands_only_what_the_mint_mints(monkeypatch, lhs, rhs, mode) -> None:
    """An overlap whose completion is `lhs -> rhs`: the gate lists it as missing only if the
    mint would mint it (ground composites and evidence are policies of both)."""
    import simplipy.mining as mining
    completion = [lhs.split(), rhs.split()]
    monkeypatch.setattr(mining, '_unjoined_overlaps', lambda *a, **k: [
        {'rule': completion, 'instance': completion[0], 'other_path': completion[0],
         'rhs_path': completion[1], 'completion': completion}])
    missing, _, _ = mining._overlap_gate(bare(), RULE_MODE[mode], [], mining.judge_tiers)
    assert bool(missing) == mint_accepts(lhs, rhs, mode), (missing, judged(lhs, rhs))


@pytest.mark.parametrize('expr, mode', [
    ('exp <constant>', 'f64'),
    ('exp <constant>', 'real'),
    ('acos <constant>', 'f64'), ('log <constant>', 'real'),
])
def test_the_fold_and_the_mint_read_one_collapse_licence(expr, mode) -> None:
    """The engine folds a constant expression to `<constant>` exactly where the mint would mint
    the collapse."""
    folded = list(bare().simplify(expr.split(), mode=mode)) == ['<constant>']
    assert folded == mint_accepts(expr, '<constant>', mode)


@pytest.mark.parametrize('expr, mode, folds', [
    ('pow 0 <constant>', 'f64', True), ('pow 0 <constant>', 'real', True),
    ('exp <constant>', 'f64', True), ('acos <constant>', 'real', True),
    ('pow -1 <constant>', 'f64', False), ('pow -1 <constant>', 'real', False),
    ('pow -1 <constant>', 'permissive', True), ('+ <constant> <constant>', 'real', True),
])
def test_the_fold_reads_the_collapse_licence(expr, mode, folds) -> None:
    """The engine collapses a constant expression to `<constant>` exactly where the collapse
    holds (3.2: the witness is the expression's own value) and has evidence (4.1) in the mode's
    values -- `pow(0, C)` collapses, `(-1)^C` (finite only at the integers) does not -- and in
    permissive without evidence (4.9)."""
    from simplipy import _core
    folded = list(bare().simplify(expr.split(), mode=mode)) == ['<constant>']
    holds = admits(expr, '<constant>', mode)
    evidence = _core.table_evidence(expr.split(), dict(_contract.ARITY), mode == 'f64')
    assert folded is folds
    assert folded == (holds and (evidence is True or mode == 'permissive'))


# ------------------------------------------------------------------------------------------
# Permissive's own simplifications (4.9): the named list, defined once in the core, and
# nothing else.
# ------------------------------------------------------------------------------------------
def test_the_core_names_permissives_list() -> None:
    from simplipy import _core
    assert _core.permissive_simplifications() == [
        'inverse pairs cancel', 'powers compose', 'like terms cancel',
        'constant collapse without evidence', 'literal to float', 'any slot matches',
        'float-resolution rules served', 'holds by agreement']


INVERSE_PAIRS = [('exp', 'log'), ('sin', 'asin'), ('cos', 'acos'), ('tan', 'atan'),
                 ('sinh', 'asinh'), ('cosh', 'acosh'), ('tanh', 'atanh')]


@pytest.mark.parametrize('outer, inner', [
    pytest.param(o, i, id=f'{o} {i}') for f, g in INVERSE_PAIRS for o, i in ((f, g), (g, f))])
def test_permissive_cancels_an_inverse_pair_in_both_orders(outer, inner) -> None:
    """Ignoring domain and branch: `asin(sin(x0))`, `exp(log(x0))` and `acosh(cosh(x0))` are
    `x0` in permissive."""
    assert list(bare().simplify([outer, inner, 'x0'], mode='permissive')) == ['x0']


@pytest.mark.parametrize('expr, want', [
    ('rootn pow x0 2 2', 'x0'), ('pow rootn x0 2 2', 'x0'), ('rootn pow x0 4 4', 'x0'),
    ('pow pow x0 2 0.5', 'x0'), ('pow pow x0 6 0.5', 'pow x0 3'),
    ('pow pow x0 3 / 1 3', 'x0'), ('pow pow x0 2 0.25', 'rootn x0 2'),
    ('rootn pow x0 2 4', 'rootn x0 2'), ('pow pow x0 2 1.5', 'pow x0 3')])
def test_permissive_composes_powers_ignoring_the_sign(expr, want) -> None:
    """`(x^a)^b` is `x^(a*b)` (the owner's ruling): `(x0^2)^(1/4)` is `x0^(1/2)`, though the
    input is `abs(x0)^(1/2)`, defined where x0 < 0."""
    assert ' '.join(bare().simplify(expr.split(), mode='permissive')) == want


@pytest.mark.parametrize('expr, want', [
    ('floor tanh 40', '0'), ('ceil exp -800', 'ceil exp -800'), ('tanh 20', '1'),
    ('asin 1e-8', '1e-8'), ('mod sin 1 2', 'sin 1')])
def test_permissive_writes_the_tables_ground_values_as_floats(expr, want) -> None:
    """P3 as ruled: permissive's ground values are the table's, written as floats --
    `floor(tanh(40))` is 0 (tanh(40) is below 1, though float64 rounds it to 1.0), `tanh(20)` is
    the float 1.0 nearest to its value. A nonzero value below the normal doubles has no
    faithful float and stays (`exp(-800)` is no 0)."""
    assert ' '.join(bare().simplify(expr.split(), mode='permissive')) == want


@pytest.mark.parametrize('expr', [
    'abs x0', 'max x0 0', 'floor x0', 'min x0 x1', 'abs - x0 x1', 'max abs x0 x1',
    'floor * 2 x0', 'ceil x0'])
def test_permissive_leaves_what_is_no_entry_as_real_does(expr) -> None:
    """`abs`, `max`, `floor` and `min` are no entry of the list: their shape is the point of
    the expression, and permissive gives real's answer."""
    e = bare()
    assert e.simplify(expr.split(), mode='permissive') == e.simplify(expr.split(), mode='real')


# ------------------------------------------------------------------------------------------
# The judge reads every literal spelling the engine emits.
# ------------------------------------------------------------------------------------------
EMITTED = ['inv neg 0', 'neg 0', 'inv 0', '/ 1 float("inf")', 'neg float("inf")',
           'float("nan")', '/ 1 3', '+ 0.1 0.2', 'np.pi', 'np.e', 'exp 1', '* 1e300 1e300',
           '/ 1 1e400', 'neg 1e-320', 'ceil -0.5', 'atan2 0 -1', 'pow 2 0.5', 'rootn -8 3',
           '/ 12345678901234567 1000000000000000000000', '* -1 0.30000000000000004',
           '* 2.827881176741212e-244756 x0', '+ x0 1.277294672630119e+53527']


@pytest.mark.parametrize('expr', EMITTED)
@pytest.mark.parametrize('mode', MODES)
def test_the_judge_reads_every_literal_the_engine_emits(expr, mode) -> None:
    """Every leaf of an engine answer that is no variable is a literal the judge reads -- a
    decimal far past float64's range included (`2.827881176741212e-244756`, which the engine
    keeps as written)."""
    from simplipy import _core
    out = bare().simplify(expr.split(), mode=mode)
    for t in out:
        if t in _contract.ARITY or t.startswith('x') or t == '<constant>':
            continue
        _contract.literal_value(t)
        assert _core.table_literal_real(t) is not None, t


def test_a_decimal_past_the_exact_readers_is_judged() -> None:
    """The table reads `m * 10^k` at any exponent up to 1e6: a rule over it is judged, not
    refused as unsupported."""
    v = verify_rule('* 2.827881176741212e-244756 x0'.split(), '* x0 2.827881176741212e-244756'.split())
    assert v['verdict'] == 'CERTIFIED', v
