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
or in f64 (the specification's P3). Policies (evidence, ground composites) are not part of it.
"""
import functools
import os

import numpy as np
import pytest
import yaml
from mpmath import mp

from simplipy import SimpliPyEngine
from simplipy.verify import MODE_TIERS, effective_tier, verify_rule
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


def expected_failure(defect: str, phase: int):
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
    case('exp log x0', 'x0', 'permissive', True, expected_failure('R1', 2)),
    case('- log x0 log x0', '0', 'real', False), case('- log x0 log x0', '0', 'f64', False),
    case('- log x0 log x0', '0', 'permissive', True, expected_failure('R1', 2)),
    case('atanh tanh x0', 'x0', 'real', True), case('atanh tanh x0', 'x0', 'f64', False),
    case('atanh tanh x0', 'x0', 'permissive', True),
    case('asin 1e-8', '1e-8', 'real', False), case('asin 1e-8', '1e-8', 'f64', True),
    case('asin 1e-8', '1e-8', 'permissive', True),
    case('rootn pow x0 2 2', 'abs x0', 'real', True),
    case('rootn pow x0 2 2', 'abs x0', 'f64', True),
    case('rootn pow x0 2 2', 'abs x0', 'permissive', True),
    case('rootn pow x0 2 2', 'x0', 'real', False), case('rootn pow x0 2 2', 'x0', 'f64', False),
    case('rootn pow x0 2 2', 'x0', 'permissive', False),
    case('tan asin ceil x0', 'atanh ceil x0', 'real', False, expected_failure('X2', 2)),
    case('tan asin ceil x0', 'atanh ceil x0', 'f64', False),
    case('tan asin ceil x0', 'atanh ceil x0', 'permissive', True),
    case('ceil inv cosh x0', '1', 'real', True),
    case('ceil inv cosh x0', '1', 'f64', True, expected_failure('N2', 2)),
    case('ceil inv cosh x0', '1', 'permissive', True),
    # the 5% grid bar: an extension on (-0.1, 0.1) is positive measure
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'real', False, expected_failure('N1', 2)),
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'f64', False),
    case('+ x0 * 0 rootn - pow x0 2 0.01 2', 'x0', 'permissive', True),
    # section 2: the sorts
    case('- ?0 ?0', '0', 'real', True), case('- ?0 ?0', '0', 'f64', True),
    case('- !0 !0', '0', 'real', True), case('- !0 !0', '0', 'f64', True),
    case('/ $0 $0', '1', 'real', True), case('/ $0 $0', '1', 'f64', True),
    case('- _0 _0', '0', 'real', False), case('- _0 _0', '0', 'f64', False),
    case('- _0 _0', '0', 'permissive', True, expected_failure('R1', 2)),
    # `_` at +inf: 0*tanh(inf) = 0 has a value, 0*inf has none
    case('* 0 tanh _0', '* 0 _0', 'real', False, expected_failure('N3', 2)),
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
    case('* exp + x0 <constant> <constant>', '* <constant> exp x0', 'real', True,
         expected_failure('X5', 2)),
    case('pow 0 <constant>', '<constant>', 'real', True),
    case('pow 0 <constant>', '<constant>', 'f64', True),
    case('pow -1 <constant>', '<constant>', 'real', True),
    case('pow -1 <constant>', '1', 'real', False), case('pow -1 <constant>', '1', 'f64', False),
    # section 3.3: ground rules
    case('cos np.pi', '-1', 'real', True), case('cos np.pi', '-1', 'f64', True),
    case('sin np.pi', '0', 'real', True), case('sin np.pi', '0', 'f64', False),
    case('sin np.pi', '0', 'permissive', True),
    case('log -1', '0', 'real', False, expected_failure('X4', 2)),
    case('log -1', '0', 'f64', False, expected_failure('X4', 2)),
    case('log -1', '0', 'permissive', True),
    case('exp tan asin 1', 'float("inf")', 'real', False, expected_failure('X4', 2)),
    case('exp tan asin 1', 'float("inf")', 'f64', True, expected_failure('X2', 2)),
    case('exp tan asin 1', 'float("inf")', 'permissive', True),
    case('asin 1', 'acos 0', 'real', True), case('asin 1', 'acos 0', 'f64', True),
    # the limit rule (V4, V5, V6)
    case('/ 1 0', 'float("inf")', 'real', False, expected_failure('N6', 2)),
    case('/ 1 0', 'float("inf")', 'f64', True),
    case('/ 1 0', 'float("inf")', 'permissive', True),
    case('pow 0 -1', 'float("inf")', 'real', False, expected_failure('N6', 2)),
    case('pow 0 -2', 'float("inf")', 'real', True),
    case('pow 1 float("inf")', '1', 'real', False, expected_failure('N7', 2)),
    case('pow 1 float("inf")', '1', 'f64', True),
]


@pytest.mark.parametrize('lhs, rhs, mode, expected', SPEC_EXAMPLES)
def test_the_judge_gives_the_specifications_verdict(lhs, rhs, mode, expected) -> None:
    assert admits(lhs, rhs, mode) is expected, judged(lhs, rhs)


# ------------------------------------------------------------------------------------------
# Assertion 6: the judge decides every pair.
# ------------------------------------------------------------------------------------------
UNDECIDED = {'UNSUPPORTED-SHAPE', 'UNRESOLVED-COVERAGE', 'JUDGE-TIMEOUT'}


@pytest.mark.parametrize('lhs, rhs', [
    pytest.param('pow rootn x0 <constant> <constant>', 'pow x0 <constant>',
                 marks=expected_failure('X5', 2)),
    pytest.param('* exp + x0 <constant> <constant>', '* <constant> exp x0',
                 marks=expected_failure('X5', 2)),
    pytest.param('+ <constant> * <constant> x0', '+ * <constant> x0 <constant>',
                 marks=expected_failure('X5', 2)),
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
    ('asin 2', 'nan', None), ('log -1', 'nan', None),
]


@pytest.mark.parametrize('expr, expected', [
    pytest.param(e, v, marks=[expected_failure(d, 2)] if d else [], id=e)
    for e, v, d in REAL_TABLE])
def test_the_table_reads_real_values(expr, expected) -> None:
    assert real_value(expr) == expected


@pytest.mark.parametrize('expr, expected', [
    pytest.param(e, v, marks=[expected_failure(d, 3)] if d else [], id=e)
    for e, v, d in REAL_TABLE])
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
    pytest.param(e, v, marks=[expected_failure('N8', 2)] if e == 'inv -0.0' else [], id=e)
    for e, v in F64_TABLE])
def test_f64_reads_the_evaluator(expr, expected) -> None:
    v = float(_contract.d_eval(_contract.parse(expr.split()), {}))
    assert v == expected and np.signbit(v) == np.signbit(expected)


@pytest.mark.parametrize('expr, expected', [
    pytest.param('inv 0', 'nan', marks=expected_failure('N6', 3)),
    pytest.param('pow 0 -1', 'nan', marks=expected_failure('N6', 3)),
    ('pow 0 -2', '+inf'), ('log 0', '-inf'),
    pytest.param('tan asin 1', 'nan', marks=expected_failure('N9', 3))])
def test_interval_enclosures_contain_the_table_values(expr, expected) -> None:
    """The interval layer's value set of a ground expression contains the table's value."""
    got = bare()._core.interval_value_set_box(expr.split(), [0.0], [0.0])
    assert got is not None
    has_fin, pinf, ninf, nan, lo, hi = got
    assert {'nan': nan, '+inf': pinf, '-inf': ninf}.get(expected, has_fin), got


# ------------------------------------------------------------------------------------------
# Assertion 1: a pre-filter rejects only what the judge rejects.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lhs, rhs, mode', [
    pytest.param('tan asin ceil x0', 'atanh ceil x0', 'real', marks=expected_failure('X2', 2)),
    pytest.param('tan asin ceil x0', 'atanh ceil x0', 'permissive',
                 marks=expected_failure('X2', 3)),
    pytest.param('pow 0 <constant>', '<constant>', 'real', marks=expected_failure('X6', 3)),
    pytest.param('pow 0 <constant>', '<constant>', 'f64', marks=expected_failure('X6', 3)),
    pytest.param('exp <constant>', '<constant>', 'f64', marks=expected_failure('X6', 3)),
    ('rootn pow x0 2 2', 'x0', 'real'), ('exp log x0', 'x0', 'f64'),
])
def test_a_mint_rejection_is_a_judge_rejection(lhs, rhs, mode) -> None:
    assert mint_accepts(lhs, rhs, mode) or not admits(lhs, rhs, mode), judged(lhs, rhs)


@pytest.mark.parametrize('lhs, rhs, mode', [
    pytest.param('ceil inv cosh ?0', '1', 'real', marks=expected_failure('X1', 3)),
    ('- ?0 ?0', '0', 'real'), ('/ ?0 ?0', '1', 'real'),
])
def test_a_promotion_bar_rejection_is_a_judge_rejection(lhs, rhs, mode) -> None:
    """Promotion's `?` bar, on the rule's own sort."""
    from simplipy.promotion import _f64_eval
    from simplipy.promotion._ladder import atom_valuations, judge_r2prime
    from simplipy.promotion._pointwise import wildcards
    _f64_eval.configure(bare())
    L, R = lhs.split(), rhs.split()
    verdict, _ = judge_r2prime(tuple(L), tuple(R), atom_valuations(wildcards(L + R)))
    assert verdict == 'PASS' or not admits(lhs, rhs, mode)


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
    'real', pytest.param('f64', marks=expected_failure('N10', 3)), 'permissive'])
def test_a_mint_acceptance_is_a_judge_admission(mode) -> None:
    refused = [(s, t, judged(s, t)['verdict']) for s, t in mint_picks(mode)
               if not admits(s, t, mode)]
    assert not refused, refused


# ------------------------------------------------------------------------------------------
# Assertion 3: a policy gives the same answer from every step that applies it.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lhs, rhs, mode', [
    pytest.param('asin 1', 'acos 0', 'f64', marks=expected_failure('X3', 3)),
    pytest.param('asin 1', 'acos 0', 'real', marks=expected_failure('X3', 3)),
    pytest.param('exp tan asin 1', 'float("inf")', 'real', marks=expected_failure('X4', 2)),
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
    pytest.param('exp <constant>', 'f64', marks=expected_failure('X6', 3)),
    pytest.param('exp <constant>', 'real', marks=expected_failure('X6', 3)),
    ('acos <constant>', 'f64'), ('log <constant>', 'real'),
])
def test_the_fold_and_the_mint_read_one_collapse_licence(expr, mode) -> None:
    """The engine folds a constant expression to `<constant>` exactly where the mint would mint
    the collapse."""
    folded = list(bare().simplify(expr.split(), mode=mode)) == ['<constant>']
    assert folded == mint_accepts(expr, '<constant>', mode)
