"""The external audit of 2026-10-31 (findings F01-F45), replayed: one test per witness that had no
test of its own, named for its finding. Each asserts the outcome the finding's disposition gives
(fixed, or ruled out of contract by an owner ruling), against an independent oracle where one
applies: numpy for float64's values, mpmath for the reals'.

The findings already pinned elsewhere are listed in `ELSEWHERE`, so the map from finding to test is
complete in one place. A witness that still reproduces is a strict xfail naming what is wrong, so
the fix flips it and the xfail must go.
"""
import contextlib
import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import mpmath
import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine, _core
from simplipy.verify import _contract, licensed, verify_rule

REPO = Path(__file__).resolve().parent.parent
OPERATORS = yaml.safe_load(open(REPO / 'assets' / 'engines' / 'acj2-4' / 'config.yaml'))['operators']
MODES = ('strict', 'permissive')

#: Where each finding's reproduction lives when it is not in this file (file: test name).
ELSEWHERE = {
    'F01': 'test_exact_routes.py: test_a_narrow_tent_is_refuted_in_every_mode',
    'F02': 'test_differential.py: test_precision_run_out_is_no_equality',
    'F04': 'test_verify.py: TestTheF64MagnitudeLadder (ruling 23(b) admits it)',
    'F06': 'test_laws.py: pow.zero / pow.base_one rows (numpy\'s value in strict)',
    'F07': 'test_differential.py: test_the_fold_writes_nan_only_where_the_family_is_nan_at_every_constant',
    'F08': 'test_ground_values.py: test_a_number_is_written_as_the_cheapest_value_its_mode_admits (ruling 22)',
    'F09': 'test_ground_values.py: test_groundness_belongs_to_the_tree_not_its_spelling',
    'F10': 'test_verify.py: TestTheJudgeKeepsItsStateToItself',
    'F11': 'test_differential.py: fun.sinh_asinh rows of `sinh * min -1 cos 1 asinh x0`',
    'F12': 'test_artifact_compat.py: test_a_rule_file_mined_before_0_15_refuses',
    'F13': 'test_one_policy.py',
    'F15': 'test_loading_hardening.py: test_a_zero_arity_sin_is_refused_at_load',
    'F16': 'test_operator_kernels.py (the measured claim)',
    'F17': 'test_loading_hardening.py: test_the_judge_reads_a_fraction_on_every_mpmath',
    'F21': 'test_is_valid_forms.py: test_an_ndarray_of_byte_strings_is_read_as_the_tokens_it_spells',
    'F22': 'test_conversion_quirk_fixes.py: test_the_depth_of_an_expression_sets_no_limit',
    'F26': 'test_helpers.py: test_a_tensor_on_either_side_takes_torch (skips without torch)',
    'F27': 'test_ground_values.py: test_a_literal_binds_no_variable_slot',
    'F28': 'test_differential.py: test_the_judge_decides_a_rule_with_constants (owner, 2026-11-13: '
           'a constant is finite and a member whose constant part has no real value is not read)',
    'F29': 'test_exceptional_values.py',
    'F32': 'test_differential_workload.py (section 7 as narrowed, E2-10)',
    'F34': 'test_helpers.py: test_generated_names_avoid_the_supplied_ones',
    'F35': 'test_helpers.py: test_a_placeholder_is_a_whole_identifier',
    'F36': 'test_verify.py: TestTheCallersTimerIsKept',
    'F38': 'test_table_edges.py',
    'F39': 'test_normalization_forms.py: test_names_are_resolved_before_any_algebra',
    'F41': 'test_helpers.py: test_sampling_is_uniform_beyond_int64_counts',
    'F43': 'test_verify.py: test_the_monitor_attributes_to_the_baseline_engines_own_answer',
    'F45': 'test_differential.py: test_the_judge_decides_a_rule_with_constants (NO-WITNESS, never KILL)',
}


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine(operators=OPERATORS, rules=[], rules_permissive=[])


def simplified(engine, expr: str, mode: str) -> str:
    return ' '.join(engine.simplify(expr.split(), mode=mode))


def test_F03_the_mined_slot_rule_is_refused_in_both_modes(engine) -> None:
    """`sinh(atanh(?0)) -> tan(asin(?0))`, which the old length-3 mine wrote, is false where the
    permissive matcher binds `?0` to `ceil(x0)`: on (0, 1] the left side is sinh(+inf) and the
    right side has no real value (tan(pi/2)); float64 gives +inf against 1.633e16."""
    lhs, rhs = 'sinh atanh ?0'.split(), 'tan asin ?0'.split()
    detail = verify_rule(lhs, rhs)
    for mode in MODES:
        assert not licensed(lhs, rhs, mode, engine, detail=detail)['licensed'], (mode, detail)
    x = np.float64(1.0)
    with np.errstate(all='ignore'):
        assert np.isinf(np.sinh(np.arctanh(x))) and np.isfinite(np.tan(np.arcsin(x)))


@pytest.mark.parametrize('expr, reals', [
    ('ceil exp 1e-60', lambda m: m.ceil(m.exp(m.mpf('1e-60')))),
    ('floor exp -1e-60', lambda m: m.floor(m.exp(m.mpf('-1e-60')))),
    ('floor cos 1e-30', lambda m: m.floor(m.cos(m.mpf('1e-30')))),
    ('ceil pow 0.5 -1e-308', lambda m: m.ceil(m.power(m.mpf('0.5'), m.mpf('-1e-308')))),
    ('floor tanh 100', lambda m: m.floor(m.tanh(100))),
])
def test_F05_a_near_integer_is_floored_by_each_modes_own_value(engine, expr, reals) -> None:
    """Permissive takes the reals' value (P3, ruling 18): 2, 0, 0, 2 and 0, read by mpmath at
    1,000 digits. Strict writes the cheapest value ruling 24 admits: numpy's double of the part,
    or the reals' value where numpy's kind agrees; every candidate here is finite, so the cheaper
    integer is kept (the real value at a tie: `floor(tanh(100))` is 0, not numpy's 1)."""
    with mpmath.workdps(1000):
        want = int(reals(mpmath))
    assert simplified(engine, expr, 'permissive') == str(want)
    with np.errstate(all='ignore'):
        f64 = int(_numpy_value(expr))
    candidates = {str(want), str(f64)}
    cheapest = min(candidates, key=lambda s: (engine.complexity([s], mode='strict'), s != str(want)))
    assert simplified(engine, expr, 'strict') == cheapest


def _numpy_value(expr: str) -> float:
    """numpy's float64 value of a prefix expression over the heads F05 uses, in written order."""
    fns = {'ceil': np.ceil, 'floor': np.floor, 'exp': np.exp, 'cos': np.cos, 'tanh': np.tanh}
    tokens = expr.split()

    def walk(i):
        t = tokens[i]
        if t in fns:
            v, j = walk(i + 1)
            return fns[t](v), j
        if t == 'pow':
            a, j = walk(i + 1)
            b, k = walk(j)
            return np.power(a, b), k
        return np.float64(t), i + 1
    return walk(0)[0]


@pytest.mark.parametrize('entry, where', [
    ({'directory': 'safe', 'entrypoint': '../../outside.yaml', 'files': ['rules.json']}, 'entrypoint'),
    ({'directory': 'safe', 'entrypoint': '/ABSOLUTE/outside.yaml', 'files': ['rules.json']}, 'entrypoint'),
    ({'directory': 'safe', 'entrypoint': 'rules.json', 'files': ['rules.json'],
      'sha256': {'../../outside.yaml': '0' * 64}}, 'sha256 file name'),
])
def test_F14_a_manifest_entrypoint_or_digest_key_never_leaves_the_cache(tmp_path, entry,
                                                                        where) -> None:
    """The manifest is data: an entrypoint or a digest key that is absolute or climbs out of the
    asset's directory is refused before any path outside the cache is read or returned."""
    from simplipy import asset_manager as am
    cache = tmp_path / 'cache'
    (cache / 'safe').mkdir(parents=True)
    outside = tmp_path / 'outside.yaml'
    outside.write_text('operators: {}\n')
    (cache / 'safe' / 'rules.json').write_text('[]')
    entry = dict(entry)
    entry['entrypoint'] = entry['entrypoint'].replace('/ABSOLUTE/outside.yaml', str(outside))
    entry.setdefault('sha256', {})['rules.json'] = hashlib.sha256(b'[]').hexdigest()
    with patch.object(am, '_cached_manifest', return_value={'audit-fixture': entry}):
        with pytest.raises(ValueError, match=where):
            am.get_path('audit-fixture', local_dir=cache)


@pytest.mark.parametrize('lhs, rhs', [
    ('ceil / atan ?0 ?0', '1'),          # atan(x)/x has its one zero-source extension at 0
    ('erf * 0.1 ?0', 'erf / ?0 10'),     # one unit in the last place at x = 3: F6, not bit equality
    ('floor + floor ?0 0.5', 'floor ?0'),
])
def test_F18_the_new_heads_are_read_by_f6_not_by_bit_equality(lhs, rhs) -> None:
    d = verify_rule(lhs.split(), rhs.split())
    assert d['realised'] is True and d['tier'] == 'core', d


def test_F19_the_two_constant_exponential_family_is_certified() -> None:
    """exp(a*x + b) = exp(b)*exp(a*x): the witness (exp(b), a) exists for every finite (a, b)."""
    d = verify_rule('exp + * <constant> ?0 <constant>'.split(),
                    '* <constant> exp * <constant> ?0'.split())
    assert d['verdict'] == 'CERTIFIED' and d['tier'] == 'core', d


@pytest.mark.parametrize('n', [64, 65, 80])
def test_F20_a_sum_whose_every_term_shares_a_factor_is_factored_past_64_terms(engine, n) -> None:
    terms = [t for i in range(1, n + 1) for t in ('*', 'x0', f'x{i}')]
    out = engine.simplify(['+'] * (n - 1) + terms, mode='strict')
    assert out.count('x0') == 1, (n, out[:12])


def test_F31_a_binding_step_cap_is_reported_not_hidden() -> None:
    """Twenty equal-price rules implement a binary counter, so the terminating descent from
    b1^20(x0) to b0^20(x0) takes 2^20 - 1 steps, past the step cap of 10^6. The cap is a
    resource limit, not a bug: the call stops at a state reached by sound steps and says so."""
    ops = {s: dict(realization='simplipy.operators.sin', alias=[], arity=1, precedence=2,
                   commutative=False) for s in ('b0', 'b1')}
    n = 20
    rules = [(['b1'] + ['b0'] * k + ['x0'], ['b0'] + ['b1'] * k + ['x0']) for k in range(n)]
    e = SimpliPyEngine(ops, rules=rules, rules_permissive=[])
    before = e._core.step_cap_binds()
    answer, status = e.simplify(['b1'] * n + ['x0'], mode='strict', effort=0, return_status=True)
    assert status.interrupted and 'steps' in status.caps, status
    assert e._core.step_cap_binds() > before
    assert answer != ['b0'] * n + ['x0']


@pytest.mark.parametrize('mode', MODES)
def test_F33_the_prefilter_accepts_a_family_whose_negative_constants_owe_nothing(mode) -> None:
    """`log(C^x) -> D*x`: at C < 0 the left side is finite only at the integers, a null set, so
    those constants owe nothing; the native prefilter agrees with the judge's licence."""
    ops = {'*': dict(realization='simplipy.operators.mul', arity=2, precedence=2, commutative=True,
                     alias=[]),
           'pow': dict(realization='simplipy.operators.pow', arity=2, precedence=3, commutative=False,
                       alias=[]),
           'log': dict(realization='simplipy.operators.log', arity=1, precedence=2, commutative=False,
                       alias=[])}
    e = SimpliPyEngine(ops, rules=[], rules_permissive=[])
    left, right = 'log pow <constant> x0'.split(), '* <constant> x0'.split()
    assert licensed(left, right, mode, e)['licensed']
    rows = np.array([[x, 0.0] for x in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)] * 16)
    ok, why = e._core.judge_pair(left, right, ['x0', 'x1'], rows.ravel().tolist(), len(rows), 16,
                                 16, 20261101, 1e-11, 1e-12, min_informative=1, rule_mode=mode)
    assert ok, why


def test_F40_a_structural_exponent_stays_structural_through_a_function() -> None:
    from simplipy import masking
    e = SimpliPyEngine({'pow': dict(realization='simplipy.operators.pow', alias=[], arity=2,
                                    commutative=False),
                        'sin': dict(realization='np.sin', alias=[], arity=1, commutative=False)},
                       rules=[], rules_permissive=[])
    x = e.simplify(['pow', 'x0', 'sin', '1'])
    assert x == ['pow', 'x0', 'sin', '1']
    assert [(v, r) for _, v, r in masking.literal_sites(x, e)] == [('1', masking.Role.EXPONENT)]
    assert e.mask(x, 'fittable', collect=False) == x


@pytest.mark.parametrize('mode', MODES)
def test_F42_a_certificate_carries_its_mode_and_is_licensed_there(engine, mode) -> None:
    """`log((e^x)^2) -> 2x` is certified in the mode asked for, and each returned rule is
    licensed in that mode (the old fixed permissive certification installed it into f64)."""
    source = ['log', 'pow', 'exp', 'x0', '2']
    with contextlib.redirect_stdout(io.StringIO()):
        pairs = engine.certify_rules([source], [['*', '2', 'x0']], mode=mode,
                                     dummy_variables=['x0'], max_target_pattern_length=3,
                                     extra_internal_terms=['2'], X=64, seed=7,
                                     constants_fit_challenges=4, constants_fit_retries=4)
    assert pairs
    for p in pairs:
        assert str(getattr(p.mode, 'value', p.mode)) == mode, p
        assert licensed(list(p[0]), list(p[1]), mode, engine)['licensed'], p


# ------------------------------------------------------------------ witnesses that still reproduce

@pytest.mark.xfail(strict=True, reason=(
    'F08 (log witness), reproduces at 3a6ed3a: strict answers NaN for log(sin(1e-20 + np.pi)). '
    "numpy gives -36.6387 and the reals' value with np.pi as the double is -36.6388; with np.pi as "
    "pi it has no value, a kind numpy's does not share, so ruling 24 admits neither NaN nor the "
    "reals' reading. The ground evaluator refuses the answer; formal.md: a step inside a fully "
    'numeric part is admitted for the whole part (sin(x + pi) -> -sin(x) is taken first).'))
def test_F08_a_shift_below_pis_resolution_keeps_float64s_finite_log(engine) -> None:
    expr = 'log sin + 1e-20 np.pi'
    with np.errstate(all='ignore'):
        f64 = float(np.log(np.sin(np.float64(1e-20) + np.pi)))
    assert np.isfinite(f64)
    out = engine.simplify(expr.split(), mode='strict')
    assert out != ['float("nan")'], out
    arity = dict(_contract.ARITY)
    verdict = _core.ground_rule_verdict(expr.split(), out, arity)
    assert verdict is None or verdict[0] or verdict[1], (out, verdict)


@pytest.mark.xfail(strict=True, reason=(
    'F08 (ceil witness), reproduces at 3a6ed3a: strict answers 1 for ceil(1/0.9999999999999999). '
    'numpy gives ceil(1.0000000000000002) = 2 and the reals ceil(10^16/(10^16 - 1)) = 2: strict '
    'rounds the exact reciprocal to the double 1.0 before ceil reads it, a value its own ground '
    'evaluator refuses for `inv 0.9999999999999999` (neither the same double nor the same real '
    'value).'))
def test_F08_ceil_of_a_reciprocal_reads_the_reciprocals_own_value(engine) -> None:
    expr = 'ceil inv 0.9999999999999999'
    f64 = float(np.ceil(np.divide(1.0, np.float64('0.9999999999999999'))))
    with mpmath.workdps(60):
        reals = int(mpmath.ceil(1 / mpmath.mpf('0.9999999999999999')))
    assert f64 == reals == 2
    assert simplified(engine, expr, 'permissive') == '2'
    assert simplified(engine, expr, 'strict') == '2'


@pytest.mark.xfail(strict=True, reason=(
    "F37, partly reproduces at 3a6ed3a: an engine whose config declares `*` (or `+`, `*`, `sin`) "
    "answers x0*x0 with the core's `pow(x0, 2)`; is_valid and expression_variables read it (59dee9e), "
    "but as_callable raises NameError ('simplipy' is not in the engine's namespace) and the "
    'realization spelling does not read back.'))
def test_F37_the_callable_of_the_cores_own_answer_evaluates() -> None:
    e = SimpliPyEngine({'*': dict(realization='*', alias=[], arity=2, commutative=True)},
                       rules=[], rules_permissive=[])
    y = e.to_prefix(e.simplify(['*', 'x0', 'x0'], mode='strict'))
    assert y == ['pow', 'x0', '2'] and e.is_valid(y) and e.expression_variables(y) == ['x0']
    assert float(e.as_callable(y)(3.0)) == 9.0
    assert e.to_prefix(e.to_realization(y)) == y


@pytest.mark.xfail(strict=True, reason=(
    "F23, reproduces at 3a6ed3a: the wheel compiles mimalloc in (Cargo feature mimalloc-allocator), "
    "but the bundled notices (LICENSES_bundled.txt, the wheel's only third-party notice file) do not "
    "carry its MIT notice (release checklist item I: notices from the Cargo tree)."))
def test_F23_the_wheel_ships_the_notice_of_the_allocator_it_compiles_in() -> None:
    notices = REPO / 'LICENSES_bundled.txt'
    if not notices.exists():
        pytest.skip('outside the source tree')
    text = notices.read_text()
    assert 'mimalloc' in text and 'Microsoft Corporation' in text


@pytest.mark.xfail(strict=True, reason=(
    "Constructor rule renaming, found while replaying F34's class at 3a6ed3a: the strict set's "
    "dedup renames a rule's x-variables to ?0, ?1, ... from 0, over the rule's own slots, so "
    "`inv(x1*?0) -> 1` loads as `inv(?0*?0) -> 1` (another rule; the permissive set is not renamed)."))
def test_a_rules_variable_name_never_collides_with_its_slots() -> None:
    rule = (('inv', '*', 'x1', '?0'), ('1',))
    e = SimpliPyEngine(OPERATORS, rules=[rule], rules_permissive=[rule])
    assert [lhs.count('?0') for lhs, _ in e.simplification_rules] == [1], e.simplification_rules
