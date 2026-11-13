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

"Admits" is the judge's word for "this mode's file may hold the rule by its truth": `f64`
admits what holds in f64 and `permissive` what holds in permissive (the specification's P3
among it, and every rule f64 holds: permissive includes f64). The reference permissive is
defined against, real's values, is the column `reals`:
the judge's verdict in them. Policies (evidence, ground composites) are not part of it:
`simplipy.verify.licensed` adds them (the specification's 3.4).
"""
import functools
import os

import numpy as np
import pytest
import yaml
from mpmath import mp, mpf

from simplipy import SimpliPyEngine
from simplipy.verify import MODE_TIERS, effective_tier, licensed, verify_rule
from simplipy.verify import _contract

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ('f64', 'permissive')
RULE_MODE = {'f64': 'default', 'permissive': 'permissive'}
#: The reference column: the judge's verdict in real's values, which permissive is defined against.
REALS = 'reals'

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
    'N6': 'the table reads 1/0 and its kin as +inf, against the limit rule (V4)',
    'N7': 'the table reads 1^inf, inf^0 and NaN^0 as 1, against the limit rule (V5) and V6',
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
    """Does the rule hold in `mode`: the reals' verdict, or the mode's own reading of the
    judgment (`_contract.judged_reading`), which for a rule with variables is the tier's (the
    tier holds the modes that hold it) and for a fully numeric rule each mode's reading of its
    own value."""
    if mode == REALS:
        return judged(lhs, rhs)['verdict'] in _contract.HOLDS_IN_THE_REALS
    d = judged(lhs, rhs)
    if not d.get('ground'):
        assert (effective_tier(d) in MODE_TIERS[mode]) == (_contract.judged_reading(d, mode) is True)
    return _contract.judged_reading(d, mode) is True


@functools.lru_cache(maxsize=1)
def bare() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


@functools.lru_cache(maxsize=1)
def mine_rows() -> np.ndarray:
    """The rows the mint reads here: the judge's (`_contract.judge_rows`, the shipped mines'
    matrix), as a mine of the shipped configuration reads them, so a counterexample the mint
    finds on them is one the judge reads too."""
    return np.array(_contract.judge_rows())


@pytest.mark.parametrize('lhs, rhs', [('log exp atanh x0', 'atanh x0'),
                                      ('log exp atanh _0', 'atanh _0')])
def test_strict_and_its_mint_read_a_rounding_alike(lhs, rhs):
    # float64 rounds `log(exp(y))` for small y (exp rounds 1 + y): the rule is true in the reals
    # and both sides stay finite, so strict holds it, and the mint's rows, a relative 1e-11 apart
    # at most, certify it
    assert admits(lhs, rhs, REALS), judged(lhs, rhs)
    assert admits(lhs, rhs, 'f64'), judged(lhs, rhs)
    assert mint_accepts(lhs.replace('_0', 'x0'), rhs.replace('_0', 'x0'), 'f64')


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
    case('/ x0 x0', '1', REALS, True), case('/ x0 x0', '1', 'f64', True),
    case('/ x0 x0', '1', 'permissive', True),
    case('exp log x0', 'x0', REALS, False), case('exp log x0', 'x0', 'f64', False),
    case('exp log x0', 'x0', 'permissive', True),
    case('- log x0 log x0', '0', REALS, False), case('- log x0 log x0', '0', 'f64', False),
    case('- log x0 log x0', '0', 'permissive', True),
    case('atanh tanh x0', 'x0', REALS, True), case('atanh tanh x0', 'x0', 'f64', False),
    case('atanh tanh x0', 'x0', 'permissive', True),
    case('asin 1e-8', '1e-8', REALS, False), case('asin 1e-8', '1e-8', 'f64', True),
    case('asin 1e-8', '1e-8', 'permissive', True),
    case('rootn pow x0 2 2', 'abs x0', REALS, True),
    case('rootn pow x0 2 2', 'abs x0', 'f64', True),
    case('rootn pow x0 2 2', 'abs x0', 'permissive', True),
    case('rootn pow x0 2 2', 'x0', REALS, False), case('rootn pow x0 2 2', 'x0', 'f64', False),
    case('rootn pow x0 2 2', 'x0', 'permissive', False),
    case('tan asin ceil x0', 'atanh ceil x0', REALS, False),
    case('tan asin ceil x0', 'atanh ceil x0', 'f64', False),
    case('tan asin ceil x0', 'atanh ceil x0', 'permissive', True),
    case('ceil inv cosh x0', '1', REALS, True),
    case('ceil inv cosh x0', '1', 'f64', True),
    case('ceil inv cosh x0', '1', 'permissive', True),
    # measure for f64 on the real cells: a set of measure zero lies on lower-dimensional cells.
    # No value on (-0.001, 0.001), narrower than any cell, read at doubles whose neighbours have
    # no value either: no null set in f64. A saturation plateau is a full-dimensional cell:
    # float64's 0/0 past 19.06 is no null set. Float64's 0/0 at the one double x = 0.1 is an
    # extension on a point: f64 holds, by its own values
    case('+ x0 * 0 rootn - pow x0 2 1e-6 2', 'x0', 'f64', False),
    case('+ x0 * 0 rootn - pow x0 2 1e-6 2', 'x0', 'permissive', True),
    case('/ - tanh x0 1 - tanh x0 1', '1', REALS, True),
    case('/ - tanh x0 1 - tanh x0 1', '1', 'f64', False),
    case('/ - x0 0.1 - x0 0.1', '1', 'f64', True),
    # the reals' measure is owed by no mode: both sides have no value in the reals on x0 >= 0,
    # where the cells read x0 - |x0| as a hull through 0 and cannot decide the reals' measure;
    # f64 decides it in its own values (the deployed evaluator's infinities agree there)
    case('inv * x1 - x0 abs x0', '/ inv - x0 abs x0 x1', 'f64', True),
    case('inv * x1 - x0 abs x0', '/ inv - x0 abs x0 x1', 'permissive', True),
    # the 5% grid bar: an extension on (-0.1, 0.1) is positive measure
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', REALS, False),
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'f64', False),
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'permissive', True),
    # the owner's rulings: an infinity is a value (no clause (c)), the guard, coincident slots,
    # undecided. At x = pi/2 the left side is +inf and the right side meets 1/0: an enclosing
    # atan shows the difference, so it is no excuse in any mode
    case('abs log cos x0', 'log inv cos x0', REALS, False),
    case('abs log cos x0', 'log inv cos x0', 'f64', False),
    case('atan abs log cos x0', 'atan log inv cos x0', REALS, False),
    case('- ceil x0 floor x0', '1', REALS, False),
    case('- ceil x0 floor x0', '1', 'f64', False),
    case('- ceil x0 floor x0', '1', 'permissive', False),
    case('abs log cos _0', 'log inv cos _0', REALS, False),
    case('atan tan asin _0', 'asin _0', REALS, False),
    case('atan tan asin x0', 'asin x0', REALS, True),
    case('/ - x0 x1 - x1 x0', '-1', REALS, False),
    case('/ - x0 x1 - x1 x0', '-1', 'f64', False),
    case('/ - x0 x1 - x1 x0', '-1', 'permissive', True),
    case('/ !0 !0', '1', REALS, False),
    # section 2: the sorts
    case('- ?0 ?0', '0', REALS, True), case('- ?0 ?0', '0', 'f64', True),
    case('- !0 !0', '0', REALS, True), case('- !0 !0', '0', 'f64', True),
    case('/ $0 $0', '1', REALS, True), case('/ $0 $0', '1', 'f64', True),
    case('- _0 _0', '0', REALS, False), case('- _0 _0', '0', 'f64', False),
    case('- _0 _0', '0', 'permissive', True),
    # `_` at +inf: 0*tanh(inf) = 0 has a value, 0*inf has none
    case('* 0 tanh _0', '* 0 _0', REALS, False),
    case('* 0 tanh _0', '* 0 _0', 'f64', False),
    # section 3.2: constants
    case('acos <constant>', '<constant>', REALS, True),
    case('acos <constant>', '<constant>', 'f64', True),
    case('exp + x0 <constant>', '* <constant> exp x0', REALS, True),
    case('* <constant> exp x0', 'exp + x0 <constant>', REALS, False),
    case('* <constant> exp x0', 'exp + x0 <constant>', 'f64', False),
    case('pow rootn x0 <constant> <constant>', 'pow x0 <constant>', REALS, False),
    case('pow rootn x0 <constant> <constant>', 'pow x0 <constant>', 'f64', False),
    case('pow rootn x0 <constant> <constant>', 'pow x0 <constant>', 'permissive', False),
    case('* exp + x0 <constant> <constant>', '* <constant> exp x0', REALS, True),
    case('pow 0 <constant>', '<constant>', REALS, True),
    case('pow 0 <constant>', '<constant>', 'f64', True),
    case('pow -1 <constant>', '<constant>', REALS, True),
    case('pow -1 <constant>', '1', REALS, False), case('pow -1 <constant>', '1', 'f64', False),
    # section 3.3: ground rules, one point, read by each mode's own value of it: f64 by float64's
    # double or, where float64's kind agrees, the reals' value; permissive by the table at float
    # resolution, a fully numeric expression without a value being NaN (permissive does not
    # include f64 here: a fully numeric part keeps each mode's value)
    case('cos np.pi', '-1', REALS, True), case('cos np.pi', '-1', 'f64', True),
    case('sin np.pi', '0', REALS, True), case('sin np.pi', '0', 'f64', True),
    case('sin np.pi', '0', 'permissive', True),
    case('log -1', '0', REALS, False),
    case('log -1', '0', 'f64', False),
    case('log -1', '0', 'permissive', False),
    case('exp tan asin 1', 'float("inf")', REALS, False),
    case('exp tan asin 1', 'float("inf")', 'f64', True),
    case('exp tan asin 1', 'float("inf")', 'permissive', False),
    case('asin 1', 'acos 0', REALS, True), case('asin 1', 'acos 0', 'f64', True),
    # the limit rule (V4, V5, V6)
    case('/ 1 0', 'float("inf")', REALS, False),
    case('/ 1 0', 'float("inf")', 'f64', True),
    case('/ 1 0', 'float("inf")', 'permissive', False),
    case('pow 0 -1', 'float("inf")', REALS, False),
    case('pow 0 -2', 'float("inf")', REALS, False),
    case('pow 1 float("inf")', '1', REALS, False),
    case('pow 1 float("inf")', '1', 'f64', True),
    # P3 as ruled: permissive serves real's values up to float resolution -- a deep saturation
    # (1 - 2e-9566 is the double 1.0). Judged at the `_` its matcher binds (the owner's ruling,
    # 2026-10-31), its own criterion excuses no isolated point: a disagreement at x = +-1 alone
    # is one on (0, 1] at `?0 = ceil(x0)`, and an infinity of the left side is a value it owes,
    # as every mode does. Permissive includes f64: a rule float64 holds holds there too, though
    # its own criterion refuses it (float64's asin(1.0) falls short of pi/2, and cos(asin(1.0))
    # is 6.1e-17)
    case('tanh cosh (-10)', '1', 'permissive', True),
    case('tanh sinh atanh x0', 'tanh tan asin x0', 'permissive', True),
    case('tanh sinh atanh _0', 'tanh tan asin _0', 'permissive', True),
    case('ceil abs cos x0', '1', 'permissive', True),
    case('abs log cos x0', 'log inv cos x0', 'permissive', False),
    case('pow 0 asinh _0', 'pow 0 _0', 'f64', True),
    # a negative power of zero has no value however its exponent is spelled, so `0^asinh(x)`
    # and `0^x` agree at every binding: 0 for x > 0, 1 at 0, no value for x < 0
    case('pow 0 asinh _0', 'pow 0 _0', 'permissive', True),
]


@pytest.mark.parametrize('lhs, rhs, mode, expected', SPEC_EXAMPLES)
def test_the_judge_gives_the_specifications_verdict(lhs, rhs, mode, expected) -> None:
    assert admits(lhs, rhs, mode) is expected, judged(lhs, rhs)


#: Section 4: rules that hold and are refused by a named policy (`licensed` = holds and every
#: policy), each with the policy the specification names.
POLICY_EXAMPLES = [
    ('asin 1', 'acos 0', 'f64', 'ground_composite'),
    ('asin 1', 'acos 0', 'permissive', 'ground_composite'),
    ('exp tan asin 1', 'float("inf")', 'f64', 'evidence'),
    ('pow -1 <constant>', '<constant>', 'f64', 'evidence'),
    ('abs x0', 'rootn pow x0 2 2', 'f64', 'ordering'),
    ('exp * 2 _0', '* exp _0 exp _0', 'permissive', 'duplication'),
    ('+ * x0 x1 * x0 x2', '* x0 + x1 x2', 'permissive', 'engine_owned'),
    ('log -3', 'float("nan")', 'permissive', 'engine_owned'),
    # permissive cancels an inverse pair itself (4.9), so the rule is the engine's
    ('exp log x0', 'x0', 'permissive', 'engine_owned'),
    ('pow + x0 <constant> 2', '+ + pow x0 2 * <constant> x0 <constant>', 'permissive',
     'constant_count'),
]


@pytest.mark.parametrize('lhs, rhs, mode, policy', POLICY_EXAMPLES)
def test_a_named_policy_refuses_what_holds(lhs, rhs, mode, policy) -> None:
    got = licensed(lhs.split(), rhs.split(), mode, bare(), detail=judged(lhs, rhs))
    assert got['holds'] and not got['licensed'] and policy in got['refused_by'], got


@pytest.mark.parametrize('lhs, rhs, mode', [
    ('exp + log x0 log x1', '* x0 x1', 'permissive'), ('acos <constant>', '<constant>', 'f64'),
    ('exp + x0 <constant>', '* <constant> exp x0', 'permissive'), ('ceil inv cosh x0', '1', 'f64'),
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
    ('pow 0 -2', 'nan', None), ('pow 0 -0.5', 'nan', None), ('rootn 0 -2', 'nan', None),
    ('inv pow 0 2', 'nan', None), ('mod 1 0', 'nan', None), ('mod 0 float("inf")', 'nan', None),
    ('mod 0 float("-inf")', 'nan', None), ('rootn 2 float("inf")', 'nan', None),
    ('pow float("-inf") 0', '1.0', None), ('mod 1 float("inf")', '1.0', None),
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
def test_the_permissive_fold_agrees_with_the_table(expr, expected) -> None:
    """A ground expression permissive folds to a literal folds to the table's value, written as
    the double nearest to it (it may also leave it as written)."""
    folded = literal_reading(list(bare().simplify(expr.split(), mode='permissive')))
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
    ('inv 0', 'nan'), ('pow 0 -1', 'nan'), ('pow 0 -2', 'nan'), ('log 0', '-inf'),
    ('rootn 0 -2', 'nan'), ('mod 0 float("inf")', 'nan'), ('pow float("-inf") 0', '1.0'),
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
    ('tan asin ceil x0', 'atanh ceil x0', 'permissive'),
    ('pow 0 <constant>', '<constant>', 'permissive'),
    ('pow 0 <constant>', '<constant>', 'f64'),
    ('exp <constant>', '<constant>', 'f64'),
    # the mint's f64 bar reads data as finite (no probe at +-inf), and permissive's mint owes no
    # clause (b)
    ('ceil inv cosh x0', '1', 'f64'),
    ('exp log x0', 'x0', 'permissive'),
    ('rootn pow x0 2 2', 'x0', 'permissive'), ('exp log x0', 'x0', 'f64'),
    # rows the table cannot read are skipped, as the judge skips them (erf(sinh(-100)) is
    # -1 + e^-(1.7e86)); the special points check float64's divergence in f64 only
    ('ceil erf sinh x0', 'ceil erf x0', 'permissive'),
    ('floor erf sinh x0', 'floor erf x0', 'permissive'),
    # a value float64 gives off a rounded zero is read by the table: tan(acos(-1)) is 0
    ('tan acos floor x0', 'inv atanh floor x0', 'permissive'),
    # permissive excuses a disagreement at an isolated point (P3): the battery's points, the
    # exceptional points and a sampled row on one
    ('abs ceil sin x0', 'ceil sin x0', 'permissive'), ('ceil abs sin x0', '1', 'permissive'),
    ('inv atanh ceil x0', '0', 'permissive'),
])
def test_a_mint_rejection_is_a_judge_rejection(lhs, rhs, mode) -> None:
    assert mint_accepts(lhs, rhs, mode) or not admits(lhs, rhs, mode), judged(lhs, rhs)


@pytest.mark.parametrize('lhs, rhs, mode', [
    ('ceil inv cosh ?0', '1', 'permissive'), ('ceil inv cosh ?0', '1', 'f64'),
    ('- ?0 ?0', '0', 'f64'), ('/ ?0 ?0', '1', 'f64'),
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


@pytest.mark.parametrize('lhs, rhs, mode', [
    # permissive excuses a disagreement that vanishes on both sides of its point; at x = 1,
    # the edge of the band (0.99961, 1] it holds on, this one does not
    ('ceil acosh asin x0', '1', 'permissive'), ('ceil acosh asin x0', 'max 1 x0', 'permissive'),
    ('ceil abs sin x0', '1', 'permissive'),
])
def test_a_mint_acceptance_of_a_pair_is_a_judge_admission(lhs, rhs, mode) -> None:
    assert not mint_accepts(lhs, rhs, mode) or admits(lhs, rhs, mode), judged(lhs, rhs)


@pytest.mark.parametrize('mode', MODES)
def test_a_mint_acceptance_is_a_judge_admission(mode) -> None:
    """The f64 mint reads the judge's magnitude ladder (`table::F64_MAGNITUDES`), in kind:
    `sinh(asinh(x)) -> x` (66 units in the last place at 1e100) and `rootn(pow(x, 3), 3) -> x`
    (101 at 1e-100) pass it, and strict holds them by its second reading."""
    from simplipy import _core
    assert tuple(_core.TABLE_F64_MAGNITUDES) == _contract.F64_MAGNITUDES
    refused = [(s, t, judged(s, t)['verdict']) for s, t in mint_picks(mode)
               if not admits(s, t, mode)]
    assert not refused, refused


@pytest.mark.parametrize('lhs, rhs, mode', [
    # a constant that owes nothing binds nothing (3.2): at C < 0, `log(C^x)` is finite at the even
    # integers alone, so the domain gate never reads it there
    ('log pow <constant> x0', '* <constant> x0', 'permissive'),
    # permissive reads the reals at the battery's points, at float resolution: at x = 1,
    # cos(asin(x)) is exactly 0 (float64 gives 6.1e-17), and at x = pi, asinh(cos(x)) rounds up
    # to 0 where cos(x) is -1; asin(1e-8) agrees with 1e-8 at float resolution
    ('ceil cos asin x0', '1', 'permissive'), ('ceil asinh cos x0', 'ceil cos x0', 'permissive'),
    ('floor abs sin x0', '0', 'permissive'), ('asin 1e-8', '1e-8', 'permissive'),
    # f64 reads float64 at the battery's doubles, by strict's two readings: pow(1.0, inf) is 1
    # at x = pi/2's double, of one kind with 0, and in the reals sin of that double is below 1
    ('pow sin x0 float("inf")', '0', 'f64'), ('pow sin x0 float("inf")', '0', 'permissive'),
    # a variable-free source owes its value only where it is finite (3.2): pow(asin(C/2), inf) is
    # 0 for |C| < 1.683 and +inf on the band up to 2, where `0` owes nothing
    ('pow asin / <constant> 2 float("inf")', '0', 'permissive'),
    ('pow cos <constant> float("inf")', '0', 'permissive'),
    # at C = 0 float64's pow(1, inf) is 1 against 0, where the reals give the left side no value:
    # F6 there, and both refuse
    ('pow cos <constant> float("inf")', '0', 'f64'),
    # the magnitude ladder
    ('sinh asinh x0', 'x0', 'f64'), ('tan atan asinh x0', 'asinh x0', 'f64'),
])
def test_the_mint_reads_as_the_judge_reads(lhs, rhs, mode) -> None:
    """One reading: the mint's pair judgment accepts exactly where the judge admits."""
    assert mint_accepts(lhs, rhs, mode) == admits(lhs, rhs, mode), judged(lhs, rhs)


@pytest.mark.parametrize('lhs, rhs', [
    ('log pow <constant> x0', '* <constant> x0'),
    # at the special constant C = 0, float64's `0*exp(x)` is NaN where exp overflows, and the
    # fit there reads no range standard
    ('exp + x0 log <constant>', '* <constant> exp x0'),
])
def test_the_f64_mints_tolerances_may_refuse_stricts_second_reading(lhs, rhs) -> None:
    """The f64 mint reads float64's values within the mine's tolerances (a relative 1e-11, an
    absolute 1e-12), and they stay: a rule strict holds by its second reading alone (true in the
    reals, float64's two sides of one kind, not within F6) may be refused by them, a loss of the
    scan's recall and not of its soundness."""
    d = judged(lhs, rhs)
    assert admits(lhs, rhs, 'f64') and d['realised_f6'] is False, d
    assert d['realised_kinds'] is True, d
    assert not mint_accepts(lhs, rhs, 'f64')


@pytest.mark.parametrize('src, tgt, rows', [
    # `tan(atan(x))` is up to 565 units from x on the mine's rows, 8e-14 relative: past F6, inside
    # the tolerance, and strict holds it by its second reading
    ('tan atan x0', 'x0', True), ('* tan atan x0 x1', '* x0 x1', True),
    ('log exp tanh x0', 'tanh x0', True),
    # a step float64 takes past 19.06 (1 against 0) and a change of kind (+inf against x)
    ('floor tanh abs x0', '0', False), ('atanh tanh x0', 'x0', False),
])
def test_the_mints_f64_rows_read_kinds_within_the_tolerances(src, tgt, rows) -> None:
    """The f64 scan reads its rows as strict's prefilter: the same kind and within F6 or the
    mine's tolerances (a relative 1e-11, an absolute 1e-12), before any check of the acceptance
    (`find_rule_lib_traced`)."""
    X = mine_rows()
    core = bare()._core
    lib = core.build_candidate_library([tgt.split()], ['x0', 'x1'], X.flatten().tolist(),
                                       X.shape[0], rule_mode='default')
    _, steps = core.find_rule_lib_traced(src.split(), lib, None, rtol=1e-11, atol=1e-12,
                                         rule_mode='default')
    verdicts = [v for cand, _, v in steps if cand == tgt.split()]
    assert (verdicts != ['rows']) == rows, steps


@pytest.mark.parametrize('lhs, rhs', [('log exp x0', 'x0'), ('acosh cosh x0', 'abs x0')])
def test_an_overflow_row_and_a_rounding_row_are_read_by_one_agreement(lhs, rhs) -> None:
    """The f64 mint has one row agreement (`battery::strict_row_agrees`): the range standard,
    which takes the true value where an operation overflowed, reads every other row by it too.
    On the judge's rows these pairs have both an overflow row (|x0| > 710) and a rounding row
    past F6 within the tolerances, and each set of rows alone was always accepted: the pair is
    accepted, as the judge holds it."""
    X = mine_rows()
    assert (np.abs(X[:, 0]) > 710).any()
    assert mint_accepts(lhs, rhs, 'f64') and admits(lhs, rhs, 'f64'), judged(lhs, rhs)


@pytest.mark.parametrize('lhs, rhs', [('besselj cos 1 <constant>', '<constant>')])
def test_a_fitted_constant_agrees_within_the_fits_band(lhs, rhs) -> None:
    """The mint's one row agreement reads a value by where it came from: an evaluated side by
    rounding (the mine's tolerances), a fitted constant by the fit's witness band. At the special
    constant C = 0.1 the fitted witness of `besselj(cos(1), C) -> C'` is 3.2e-12 from the
    source's value, inside the fit's band and past the tolerances of a rounding: the pair is
    minted, as the judge holds it."""
    assert mint_accepts(lhs, rhs, 'f64') and admits(lhs, rhs, 'f64'), judged(lhs, rhs)


@pytest.mark.parametrize('src', ['inv ceil sin x0', 'inv ceil tanh x0', 'inv floor erf x0',
                                 'inv max 0 x0', 'inv floor tanh x0', 'asin cosh x0',
                                 'tan atan2 x0 0', 'acosh tanh x0', 'atanh floor x0'])
@pytest.mark.parametrize('mode', MODES)
def test_the_mints_evidence_is_the_policys(src, mode) -> None:
    """The mint scans a source exactly where the evidence policy (4.1, `table_mode_evidence`,
    the function the judge's licence reads; permissive's is float64's or the reals') gives it
    evidence: a source pairs with itself where it has evidence and nowhere else, so a change to
    the interval layer moves the mint and the licence together."""
    from simplipy import _core
    evidence = _core.table_mode_evidence(src.split(), dict(_contract.ARITY),
                                         RULE_MODE[mode]) is True
    assert mint_accepts(src, src, mode) == evidence


# ------------------------------------------------------------------------------------------
# Assertion 3: a policy gives the same answer from every step that applies it.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lhs, rhs, mode', [
    ('asin 1', 'acos 0', 'f64'), ('asin 1', 'acos 0', 'permissive'),
    ('exp tan asin 1', 'float("inf")', 'permissive'),
    ('rootn pow x0 2 2', 'x0', 'f64'),
])
def test_the_overlap_gate_demands_only_what_the_mint_mints(monkeypatch, lhs, rhs, mode) -> None:
    """An overlap whose completion is `lhs -> rhs`: the gate lists it as missing only if the
    mint would mint it (ground composites and evidence are policies of both)."""
    import simplipy.mining as mining
    completion = [lhs.split(), rhs.split()]
    monkeypatch.setattr(mining, '_unjoined_overlaps', lambda *a, **k: [
        {'rule': completion, 'instance': completion[0], 'other_path': completion[0],
         'rhs_path': completion[1], 'completion': completion}])
    missing, _, _, _ = mining._overlap_gate(bare(), RULE_MODE[mode], [], mining.judge_tiers)
    assert bool(missing) == mint_accepts(lhs, rhs, mode), (missing, judged(lhs, rhs))


@pytest.mark.parametrize('completion, left', [
    # licensed core; the f64 mint's row agreement refuses it on the mine's rows: acos near 1
    # magnifies the rounding past the tolerances (3.80e-8 against 3.94e-8 at x0 = 17.8)
    (('abs atan sinh x0', 'acos inv cosh x0'), True),
    # licensed, and the agreement passes it: still demanded, and the gate fails without it
    (('log exp x0', 'x0'), False),
])
def test_the_overlap_gate_leaves_to_the_prefilter_only_what_it_refuses(monkeypatch, completion,
                                                                       left) -> None:
    """The f64 mine's prefilter stays, and strict has no high-precision rescue: an identity
    whose rounding a function magnifies past it stays permissive's. So the overlap gate does not
    demand in the f64 file a completion the mint's row agreement refuses (`_prefilter_reading`,
    the scan's own reading through `_core.prefilter_agrees`); it records it as left unjoined by
    the prefilter, with the instance and both paths. A missing completion that passes the
    agreement still fails the gate."""
    import simplipy.mining as mining
    lhs, rhs = (side.split() for side in completion)
    overlap = {'rule': [['acos', 'cos', 'atan', '_0'], ['abs', 'atan', '_0']],
               'instance': ['acos', 'cos', 'atan', 'sinh', 'x0'], 'other_path': rhs,
               'rhs_path': lhs, 'completion': [lhs, rhs]}
    monkeypatch.setattr(mining, '_unjoined_overlaps', lambda *a, **k: [overlap])
    X = mine_rows()
    prefilter = mining._prefilter_reading(bare(), X, ['x0', 'x1'], 1e-11, 1e-12)
    assert admits(' '.join(lhs), ' '.join(rhs), 'f64'), judged(' '.join(lhs), ' '.join(rhs))
    missing, _, _, prefiltered = mining._overlap_gate(bare(), 'default', [], mining.judge_tiers,
                                                      prefilter=prefilter)
    want = (1, 0) if left else (0, 1)
    assert (len(prefiltered), len(missing)) == want, (missing, prefiltered)
    if left:
        assert prefiltered[0]['instance'] == overlap['instance']
        assert {'other_path', 'rhs_path', 'completion'} <= set(prefiltered[0])
        # permissive's file joins it: no prefilter reads its gate
        missing, _, _, prefiltered = mining._overlap_gate(bare(), 'permissive', [],
                                                          mining.judge_tiers)
        assert missing and not prefiltered


@pytest.mark.parametrize('expr, mode', [
    ('exp <constant>', 'f64'),
    ('exp <constant>', 'permissive'),
    ('acos <constant>', 'f64'), ('log <constant>', 'f64'),
])
def test_the_fold_and_the_mint_read_one_collapse_licence(expr, mode) -> None:
    """The engine folds a constant expression to `<constant>` exactly where the mint would mint
    the collapse."""
    folded = list(bare().simplify(expr.split(), mode=mode)) == ['<constant>']
    assert folded == mint_accepts(expr, '<constant>', mode)


@pytest.mark.parametrize('expr, mode, folds', [
    ('pow 0 <constant>', 'f64', True), ('pow 0 <constant>', 'permissive', True),
    ('exp <constant>', 'f64', True), ('acos <constant>', 'f64', True),
    ('pow -1 <constant>', 'f64', False),
    ('pow -1 <constant>', 'permissive', True), ('+ <constant> <constant>', 'f64', True),
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


#: Why the judge refuses a constant collapse that holds: a fitted constant is finite, so the
#: constant part's value is the witness wherever it has one, and a value of the constant at which
#: it has none is no member of the family.
MEMBER = ("the judge reads float64's infinity of the constant part (log(0) at C = 0, 1/0, "
          "atanh(1), 0^C for C < 0) as a member of the family, owed a witness")
SATURATION = ("the judge's witness search misses the constant part's own value where tanh is +-1 "
              "on most of its grid (past C = 709.78 exp(C) overflows, a range event)")


class KnownRefusal(AssertionError):
    """The judge refuses a constant collapse that holds, for the cause the row names."""


def collapse_case(around, node, *causes):
    marks = [pytest.mark.xfail(strict=True, raises=KnownRefusal, reason='; '.join(causes))]
    return pytest.param(around, node, causes, marks=marks if causes else [],
                        id=around.format(node))


#: A collapse in context, `around[node] -> around[<constant>]`, with the cause of the judge's
#: refusal where it has one.
COLLAPSES_IN_CONTEXT = [
    collapse_case('+ x0 {}', 'log <constant>'),
    collapse_case('* x0 {}', 'tanh log <constant>'),
    collapse_case('+ x0 {}', 'inv <constant>'),
    collapse_case('* x0 {}', 'exp <constant>'),
    collapse_case('tanh + x0 {}', 'log <constant>', MEMBER),
    collapse_case('exp + x0 {}', 'log <constant>', MEMBER),
    collapse_case('atan + x0 {}', 'log <constant>', MEMBER),
    collapse_case('inv + x0 {}', 'log <constant>', MEMBER),
    collapse_case('exp * x0 {}', 'log <constant>', MEMBER),
    collapse_case('tanh * x0 {}', 'log <constant>', MEMBER),
    collapse_case('tanh + x0 {}', 'atanh <constant>', MEMBER),
    collapse_case('tanh + x0 {}', 'inv <constant>', MEMBER, SATURATION),
    collapse_case('tanh + x0 {}', 'exp <constant>', SATURATION),
    collapse_case('tanh + x0 {}', 'pow <constant> 2', SATURATION),
    collapse_case('tanh + x0 {}', 'cosh <constant>', SATURATION),
    collapse_case('tanh + x0 {}', '* 5 <constant>', SATURATION),
]


@pytest.mark.parametrize('around, node', [p.values[:2] for p in COLLAPSES_IN_CONTEXT])
@pytest.mark.parametrize('mode', ['f64', 'permissive'])
def test_a_constant_node_collapses_wherever_it_stands(around, node, mode) -> None:
    """A fitted constant is finite: the constant part's value is the witness wherever it has
    one, and a value of the constant at which it has none is no member, so a constant node
    collapses in any surroundings, in both modes."""
    engine = bare()
    lhs, rhs = around.format(node), around.format('<constant>')
    assert (list(engine.simplify(lhs.split(), mode=mode))
            == list(engine.simplify(rhs.split(), mode=mode))), (lhs, mode)


@pytest.mark.parametrize('around, node, causes', COLLAPSES_IN_CONTEXT)
def test_the_judge_holds_a_constant_collapse_in_context(around, node, causes) -> None:
    """The judge holds the collapse in strict, or refuses it for the cause the row names: at a
    constant it finds no witness for, the constant part is float64's infinity where the reals
    give it no value (`MEMBER`), or it has a value, its own witness, which the search misses
    (`SATURATION`)."""
    import re
    lhs, rhs = around.format(node), around.format('<constant>')
    d = judged(lhs, rhs)
    if d.get('realised') is True:
        return
    named = [float(c) for c in re.findall(r'\(([-\d.e+]+),\)', str(d.get('detail') or ''))]

    def member(c):
        at = node.replace('<constant>', repr(c)).split()
        with np.errstate(all='ignore'):
            f = float(bare().as_callable(at, [])())
        old = mp.dps
        try:
            mp.dps = 50
            v = _contract.c_eval(_contract.parse(at), {})
        finally:
            mp.dps = old
        return bool(np.isinf(f)) and (v != v or v in (mp.inf, -mp.inf))

    assert named, (lhs, d.get('verdict'), d.get('detail'))
    if MEMBER in causes:
        assert any(member(c) for c in named), (lhs, named)
    if SATURATION in causes:
        assert not all(member(c) for c in named), (lhs, named)
    if not causes:
        assert False, (lhs, d.get('verdict'), d.get('detail'))
    raise KnownRefusal(f'{lhs}: {d.get("verdict")} {d.get("detail")}')


@pytest.mark.parametrize('lhs, rhs, policy', [
    ('/ 1 0', 'float("nan")', 'engine_owned'),
    ('- <constant> <constant>', '0', 'engine_owned'),
    ('x0', '0', 'well_formed'),
])
def test_a_left_side_the_engine_folds_to_a_leaf_is_the_engines_own(lhs, rhs, policy) -> None:
    """A written composite the canon folds to a leaf is what the engine does itself
    (`engine_owned`); only a written leaf is an ill-formed left side (`well_formed`)."""
    from simplipy.verify import policy_refusals
    got = policy_refusals(lhs.split(), rhs.split(), 'f64', bare())
    assert policy in got and ({'engine_owned', 'well_formed'} - {policy}).isdisjoint(got), got


@pytest.mark.parametrize('lhs, rhs, verdict', [
    # true: d1 = e^c2 and d2 = c1, at c2 = 300 a witness of 1e130 and at c1 = 1e4 an exponent
    # float64 cannot fit across the generic values
    ('exp + * <constant> x0 <constant>', '* <constant> exp * <constant> x0', 'CERTIFIED'),
    # false: no d at c = (2.5, -1.5) (it asks 15.5 at (1.7, -1.3) and 5.93 at (-1.3, 0.6)), nor
    # at c = 2.5 (4 at 1.7, 2 at -1.3); a search that finds no witness refuses with
    # uncertainty, and sampling proves no impossibility
    ('+ * <constant> x0 * <constant> x1', '* <constant> + x0 x1', 'NO-WITNESS'),
    ('floor * <constant> x0', '* <constant> floor x0', 'NO-WITNESS'),
    # at C = 0 the left side is -1 at every x (log 0 is -inf), a value no tanh(x + D) gives:
    # owed (the left side is finite there), and refused, though float64's fit saturates
    ('tanh + x0 log <constant>', 'tanh + x0 <constant>', 'NO-WITNESS'),
    # an explicit witness exists for every (C0, C1) (D0 = C0, D1 = atan(C1^(1/101))/(12 pi)):
    # no counterexample, whatever the search finds
    ('+ * <constant> x0 <constant>', '+ * <constant> x0 pow tan * 12 * np.pi <constant> 101',
     'NO-WITNESS'),
])
def test_the_judge_decides_a_rule_with_constants(lhs, rhs, verdict) -> None:
    """Assertion 6: the witness fit stays in moderate environments (or the polish's arithmetic)
    and finds a true rule's witness. Eligibility is decided apart from the search: where the
    left side is finite at a constant a witness is owed, and a search that finds none refuses
    the rule with uncertainty (NO-WITNESS), never 'owes nothing' and never a proved
    impossibility (KILL)."""
    got = judged(lhs, rhs)
    assert got['verdict'] == verdict, got
    if verdict == 'NO-WITNESS':
        assert got['basis'] == {'f64': 'refused', 'permissive': 'refused'}, got


@pytest.mark.parametrize('mode', MODES)
def test_the_fold_writes_nan_only_where_the_family_is_nan_at_every_constant(mode) -> None:
    """3.2 owes a constant family its value wherever it is finite, so the fold writes nan only
    for a family with no value at any constant: `asin(cosh(C))` is pi/2 at C = 0 and
    `acosh(cos(C))` is 0 at C = 0, so neither is nan (the judge kills both as rules), while
    `acosh(acos(cosh(C)))` (acosh(0) has no value either) is."""
    for expr in ('asin cosh <constant>', 'acosh cos <constant>'):
        assert 'float("nan")' not in bare().simplify(expr.split(), mode=mode), expr
        assert not admits(expr, 'float("nan")', mode), judged(expr, 'float("nan")')
    assert 'float("nan")' not in bare().simplify('+ x1 acosh cos <constant>'.split(), mode=mode)
    assert list(bare().simplify('acosh acos cosh <constant>'.split(), mode=mode)) \
        == ['float("nan")']


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
    ('floor tanh 40', '0'), ('floor tanh 100', '0'), ('ceil exp -800', '1'),
    ('* x0 exp -800', '* x0 exp -800'), ('tanh 20', '1'), ('asin 1e-8', '1e-8'),
    ('mod sin 1 2', 'sin 1')])
def test_permissive_writes_the_tables_ground_values_as_floats(expr, want) -> None:
    """P3 as ruled: permissive's ground values are the table's, read at as many digits as they
    need and written as floats -- `floor(tanh(40))` and `floor(tanh(100))` are 0 (below 1, though
    float64 rounds both to 1.0), `ceil(exp(-800))` is 1, `tanh(20)` is the float 1.0 nearest to
    its value. A nonzero value below the normal doubles has no faithful float and stays
    (`exp(-800)` is no 0)."""
    assert ' '.join(bare().simplify(expr.split(), mode='permissive')) == want


@pytest.mark.parametrize('expr', [
    'abs x0', 'max x0 0', 'floor x0', 'min x0 x1', 'abs - x0 x1', 'max abs x0 x1',
    'floor * 2 x0', 'ceil x0'])
def test_permissive_leaves_what_is_no_entry_as_f64_does(expr) -> None:
    """`abs`, `max`, `floor` and `min` are no entry of the list: their shape is the point of
    the expression, and permissive gives f64's answer."""
    e = bare()
    assert e.simplify(expr.split(), mode='permissive') == e.simplify(expr.split(), mode='f64')


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
    refused as unsupported. Its readings demand more digits than the precision ceiling, so two
    equal readings there are no equality: the judge's answer is uncertain, a refusal."""
    v = verify_rule('* 2.827881176741212e-244756 x0'.split(), '* x0 2.827881176741212e-244756'.split())
    assert v['verdict'] == 'UNRESOLVED-COVERAGE', v


@pytest.mark.parametrize('slot', ['x0', '?0', '_0'])
def test_precision_run_out_is_no_equality(slot) -> None:
    """`x^2 + 1 + 1e-3000 -> x^2 + 1` is false by 1e-3000 at every x in the reals. Their
    readings demand 6,000 digits (a literal is an operand the precision is sized by), past the
    ceiling, so their agreement is no equality: the reals' verdict is undecided, never
    certified. Each mode is judged on its own values: float64 reads `1e-3000` as 0.0 and the two
    sides are bit-identical, so the rule holds in f64, whatever the reals resolved; permissive's
    float resolution needs no equality, and the enclosures of both sides decide at each atom but
    e that they round to the same doubles. At e, an approximation, `x^2 + 1` encloses wider than
    float resolution: permissive's own reading leaves that owed point unread and reaches no
    answer (`permissive_own`), and permissive holds the rule through f64, as it holds every rule
    strict holds. At 1e-100 the rungs reach the separation:
    false in the reals, while float64 adds the double 1e-100 to 1 exactly and permissive reads the
    reals' values at float resolution (P3), so it holds in both."""
    v = verify_rule(f'+ + pow {slot} 2 1 1e-3000'.split(), f'+ pow {slot} 2 1'.split())
    assert v['verdict'] == 'UNRESOLVED-COVERAGE', v
    assert (v['realised'], v['permissive_own'], v['permissive'], v['tier']) == \
        (True, None, True, 'core'), v
    name = '?0' if slot == 'x0' else slot
    assert v['unread_owed'] == {'permissive': [{name: 2.718281828459045}]}, v
    assert v['basis'] == {'f64': 'evidence', 'permissive': 'evidence'}, v
    w = verify_rule('+ + pow x0 2 1 1e-100'.split(), '+ pow x0 2 1'.split())
    assert (w['verdict'], w['tier']) == ('KILL', 'core'), w


# ------------------------------------------------------------------------------------------
# P4 in one place, the wide miss, one evidence.
# ------------------------------------------------------------------------------------------
def test_permissive_is_judged_at_the_sort_its_matcher_binds() -> None:
    """P4 as ruled on 2026-10-31 (option A): permissive's matcher binds every slot to any
    subexpression, so a permissive rule is judged at `_`, over real-valued bindings, and a
    measure-zero allowance needs a match-time guard, which permissive has none of. A
    disagreement at an isolated point of `x0` is one on an interval where `x0` is bound to
    `asin(ceil(x1))` or `ceil(x1)`: no isolated point is excused, and no move off one is
    defined any more."""
    from simplipy import _core
    assert not hasattr(_core, 'TABLE_ISOLATION_STEP')
    # differs at pi/2 + k*pi alone: at `x0 = asin(ceil(x1))`, 0 against 1 on all of (0, 1], so
    # permissive's own criterion refuses it; it holds in f64, and so in permissive
    assert judged('ceil abs cos x0', '1')['permissive_own'] is False
    assert admits('ceil abs cos x0', '1', 'f64')
    assert admits('ceil abs cos x0', '1', 'permissive')
    # differs at 1 alone, and so on (0, 1] at `x0 = ceil(x1)`; past 1 the left side has no
    # value, which still owes nothing in permissive (an extension is no disagreement)
    assert not admits('acos ceil acos x0', '0', 'permissive')
    assert admits('acos ceil ceil acos x0', 'acos ceil acos x0', 'permissive')
    # differs at the integers alone
    assert not admits('- ceil x0 floor x0', '1', 'permissive')
    # differs at 1 and on (0.99961, 1], and at pi and on (pi, 2*pi): refused as before
    assert not admits('ceil acosh asin x0', '1', 'permissive')
    assert not admits('ceil erf sinh x0', 'ceil sin x0', 'permissive')


def test_the_saturation_rule_reads_a_battery_atom() -> None:
    # erf(cosh(cosh(2))) is inside erf's boundary band at every rung, so the point goes to the
    # saturation rule, whose written bound is the slot's battery atom 2 (a `TV`): it read it as an
    # mpf and raised; it is false (erf never reaches 2, and at 0 the left side is erf(cosh(1)))
    got = verify_rule('erf cosh cosh ?1'.split(), ['?1'])
    assert effective_tier(got) == 'reject', got


def test_the_table_reads_a_harmonic_number_exactly() -> None:
    # H_n at a positive integer is the rational 1 + 1/2 + ... + 1/n: atanh(harmonic(ceil(x))) meets
    # atanh's pole exactly where ceil(x) = 1, where an approximation of 1 was unreadable at every
    # rung, so the mint passed every candidate on there and the judge refused it eight times over
    with mp.workdps(50):
        assert _contract.c_eval(_contract.parse('harmonic 3'.split()), {}) == mpf(11) / 6
        assert _contract.c_eval(_contract.parse('atanh harmonic 1'.split()), {}) == mpf('inf')
    assert not mint_accepts('atanh harmonic ceil x1', '0', 'permissive')


@pytest.mark.parametrize('pair', [('erf cosh cosh x0', 'log x0'), ('erf cosh cosh x1', '0'),
                                  ('erf pow x0 x1', 'cos erf x1')])
def test_an_unreadable_wide_miss_is_refused(pair) -> None:
    # float64 reads erf(cosh(cosh(x))) as 1.0 wherever x is not small, and the table cannot read
    # those rows either (erf's boundary band at every rung): the mint passes a wide miss on only
    # where it read some failing rows and none disagreed
    assert not mint_accepts(*pair, 'permissive')


@pytest.mark.parametrize('mode', ['permissive'])
def test_a_wide_miss_is_read_not_refused(mode) -> None:
    # float64 rounds erf(cosh(x)) to 1 wherever |x| > 2.46, most of the mint's rows; the reals keep
    # it below 1, and the mint reads a spread of the failing rows by the table
    assert admits('floor erf cosh x0', '0', mode), judged('floor erf cosh x0', '0')
    assert mint_accepts('floor erf cosh x0', '0', mode)


@pytest.mark.parametrize('mode', MODES)
def test_the_mint_and_the_policy_read_one_evidence(mode) -> None:
    # off x = 0, tan(atan2(x, 0)) is tan(pi/2): no value in the reals, 1.633e16 in float64. The
    # enclosure read the constant pi/2 as a varying argument's pole; the evidence function reads
    # its certified cells at points, for the mint's sources and the judge's policy alike.
    # Permissive includes f64, and its evidence is that of a reading the rule holds in: `-> nan`
    # holds by permissive's own criterion alone, vacuously (no value in the reals), and float64's
    # evidence, where it is false, licenses nothing
    from simplipy import _core
    tokens = 'tan atan2 x0 0'.split()
    assert _core.table_evidence(tokens, dict(_contract.ARITY), True) is True
    assert _core.table_evidence(tokens, dict(_contract.ARITY), False) is False
    assert _core.table_mode_evidence(tokens, dict(_contract.ARITY), RULE_MODE[mode]) is True
    got = licensed(tokens, ['float("nan")'], mode, bare(),
                   detail=judged('tan atan2 x0 0', 'float("nan")'))
    assert not got['licensed'], got
    assert ('evidence' in got['refused_by']) is (mode != 'f64'), got
    assert not mint_accepts('tan atan2 x0 0', 'float("nan")', mode)
