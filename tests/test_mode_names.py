"""One mode vocabulary at every entry: ``'strict'`` and ``'permissive'``.

``simplify(mode=)``, ``complexity(mode=)``, ``modes=``, ``rule_mode=``
(``prune_covered_rules``, ``utils.deduplicate_rules``, ``promotion.promote``),
``verify_ruleset(mode=)``, ``mining.verify_mine`` and ``mining.cross_mode_complexity`` all
read a mode through one coercion, and an unknown name raises ``ValueError`` naming the modes.
The strict mode's pre-0.15 names, ``'f64'`` and ``'default'`` (and ``Mode.f64``), select it with
a ``DeprecationWarning`` until 0.16.
"""
import json
import warnings

import pytest

from simplipy import Mode, SimpliPyEngine
from simplipy.engine import _canon_of_record, _coerce_mode, _coerce_rule_mode

_OPS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 3, "commutative": False},
}

_CHOICES = r"'strict' or 'permissive'"
_OLD = 'removed in simplipy 0.16'


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine(operators=_OPS, rules=[(("*", "1", "?0"), ("?0",))])


@pytest.mark.parametrize('name', ['strict', 'STRICT', ' Strict ', Mode.strict])
def test_strict_is_the_mode_without_a_warning(name) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _coerce_mode(name) is Mode.strict
        assert _coerce_rule_mode(name) == 'default'


@pytest.mark.parametrize('name', ['f64', 'default', 'F64', 'Default', ' default '])
def test_the_pre_015_names_select_strict_with_a_warning(name) -> None:
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert _coerce_mode(name) is Mode.strict
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert _coerce_rule_mode(name) == 'default'


def test_the_pre_015_member_selects_strict_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert Mode.f64 is Mode.strict
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert Mode['f64'] is Mode.strict
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert Mode('f64') is Mode.strict
    assert [m.name for m in Mode] == ['strict', 'permissive']
    with pytest.raises(AttributeError):
        Mode.nope  # noqa: B018
    with pytest.raises(ValueError):
        Mode('default')


@pytest.mark.parametrize('name, rule_mode', [('permissive', 'permissive'),
                                             (Mode.permissive, 'permissive')])
def test_the_other_mode(name, rule_mode) -> None:
    assert _coerce_rule_mode(name) == rule_mode


@pytest.mark.parametrize('name', ['real', 'REAL', ' real '])
def test_the_retired_mode_is_refused_by_the_one_coercion(name) -> None:
    with pytest.raises(ValueError, match='the real mode is retired'):
        _coerce_mode(name)
    with pytest.raises(ValueError, match='the real mode is retired'):
        Mode(name.strip())


def test_an_unknown_name_names_the_modes() -> None:
    with pytest.raises(ValueError, match=_CHOICES):
        _coerce_mode('nope')
    with pytest.raises(TypeError, match=_CHOICES):
        _coerce_mode(3)


def test_simplify_and_complexity_read_the_pre_015_names(engine) -> None:
    expr = ['+', 'x0', 'x0']
    for old in ('f64', 'default'):
        with pytest.warns(DeprecationWarning, match=_OLD):
            assert engine.simplify(expr, mode=old) == engine.simplify(expr, mode='strict')
        with pytest.warns(DeprecationWarning, match=_OLD):
            assert engine.complexity(expr, mode=old) == engine.complexity(expr, mode='strict')
    assert engine.simplify(expr) == engine.simplify(expr, mode=Mode.strict)
    with pytest.raises(ValueError, match=_CHOICES):
        engine.simplify(expr, mode='nope')


@pytest.mark.parametrize('name', ['strict', Mode.strict])
def test_prune_covered_rules_reads_strict(name) -> None:
    eng = SimpliPyEngine(operators=_OPS, rules=[(("*", "1", "?0"), ("?0",))])
    assert eng.prune_covered_rules(rule_mode=name) == 1
    assert eng.simplification_rules == []


def test_prune_covered_rules_refuses_an_unknown_mode(engine) -> None:
    with pytest.raises(ValueError, match=_CHOICES):
        engine.prune_covered_rules(rule_mode='nope')


def test_deduplicate_rules_reads_every_name(engine) -> None:
    from simplipy.utils import deduplicate_rules
    rules = [(('*', '(-1)', 'sin', 'x0'), ('neg', 'sin', 'x0')),
             (('*', 'sin', 'x0', '(-1)'), ('neg', 'sin', 'x0'))]
    out = {name: deduplicate_rules(rules, ['x0'], engine=engine, rule_mode=name)
           for name in ('strict', Mode.strict)}
    with pytest.warns(DeprecationWarning, match=_OLD):
        out['f64'] = deduplicate_rules(rules, ['x0'], engine=engine, rule_mode='f64')
    assert out['strict'] == out[Mode.strict] == out['f64']
    with pytest.raises(ValueError, match=_CHOICES):
        deduplicate_rules(rules, ['x0'], engine=engine, rule_mode='nope')


def test_promote_refuses_an_unknown_mode_before_any_work(engine) -> None:
    from simplipy.promotion import promote
    with pytest.raises(ValueError, match=_CHOICES):
        promote([], engine, rule_mode='nope')


def test_verify_mine_refuses_an_unknown_mode_before_reading(tmp_path) -> None:
    from simplipy.mining import verify_mine
    with pytest.raises(ValueError, match=_CHOICES):
        verify_mine(str(tmp_path / 'absent.json'), 'nope', _OPS)


def test_verify_mine_reads_the_pre_015_name_as_strict(tmp_path) -> None:
    from simplipy.mining import verify_mine
    path = tmp_path / 'rules_strict.json'
    path.write_text('[]')

    def judge(pairs):
        return ['core'] * len(pairs)
    reports = {'strict': verify_mine(str(path), 'strict', _OPS, judge=judge)}
    with pytest.warns(DeprecationWarning, match=_OLD):
        reports['f64'] = verify_mine(str(path), 'f64', _OPS, judge=judge)
    assert reports['f64']['mode'] == reports['strict']['mode'] == 'strict'
    assert reports['f64']['gates'] == reports['strict']['gates']


def test_a_sidecar_recording_the_pre_015_name_reads_as_strict(tmp_path) -> None:
    path = tmp_path / 'rules_strict.json'
    path.write_text('[]')
    (tmp_path / 'rules_strict.json.provenance.json').write_text(json.dumps({'mode': 'f64'}))
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert _canon_of_record(str(path)) == 'strict'
    (tmp_path / 'rules_strict.json.provenance.json').write_text(json.dumps({'mode': 'strict'}))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _canon_of_record(str(path)) == 'strict'


def test_a_mine_writes_its_strict_file_and_the_permissive_sibling() -> None:
    from simplipy.mining import _mode_paths
    assert _mode_paths('out/rules_strict.json') == {
        'strict': 'out/rules_strict.json', 'permissive': 'out/rules_permissive.json'}
    assert _mode_paths('out/rules.json')['permissive'] == 'out/rules_permissive.json'


def test_the_cli_mines_strict_and_permissive_and_reads_the_pre_015_name() -> None:
    from simplipy.__main__ import _mode_names
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _mode_names('strict,permissive') == ['strict', 'permissive']
    with pytest.warns(DeprecationWarning, match=_OLD):
        assert _mode_names('f64') == ['strict']


def test_verify_ruleset_refuses_an_unknown_mode() -> None:
    from simplipy.verify import verify_ruleset
    with pytest.raises(ValueError, match=_CHOICES):
        verify_ruleset([], mode='nope')


def test_cross_mode_complexity_reads_strict(engine) -> None:
    from simplipy.mining import cross_mode_complexity
    expr = ['+', 'x0', 'x0']
    with pytest.warns(DeprecationWarning, match=_OLD):
        old = cross_mode_complexity(engine, expr, 'default')
    assert old == cross_mode_complexity(engine, expr, 'strict')


def test_the_mining_surface_declares_its_entries() -> None:
    import simplipy.mining as mining
    assert {'RuleMiner', 'verify_mine', 'cross_mode_complexity'} <= set(mining.__all__)


_RETIRED = 'the real mode is retired'


def test_every_entry_refuses_the_retired_mode(engine, tmp_path) -> None:
    """`real` names no mode at any entry: each one raises the one refusal before any work."""
    import yaml
    from simplipy.mining import cross_mode_complexity, verify_mine
    from simplipy.promotion import promote
    from simplipy.utils import deduplicate_rules
    from simplipy.verify import licensed, verify_artifact, verify_ruleset, verify_triple
    from simplipy.verify._monitor import judge_pair
    expr = ['+', 'x0', 'x0']
    entries = {
        'simplify': lambda: engine.simplify(expr, mode='real'),
        'complexity': lambda: engine.complexity(expr, mode='real'),
        'unload_mode': lambda: engine.unload_mode('real'),
        'prune_covered_rules': lambda: engine.prune_covered_rules(rule_mode='real'),
        'deduplicate_rules': lambda: deduplicate_rules([], ['x0'], engine=engine, rule_mode='real'),
        'promote': lambda: promote([], engine, rule_mode='real'),
        'find_rules': lambda: engine.find_rules(modes=['real'], max_source_pattern_length=1,
                                                verbose=False),
        'verify_ruleset': lambda: verify_ruleset([], mode='real'),
        'verify_triple': lambda: verify_triple([], [], []),
        'licensed': lambda: licensed(['sin', 'x0'], ['sin', 'x0'], 'real', engine),
        'judge_pair': lambda: judge_pair(['x0'], ['x0'], mode='real'),
        'verify_mine': lambda: verify_mine(str(tmp_path / 'absent.json'), 'real', _OPS),
        'cross_mode_complexity': lambda: cross_mode_complexity(engine, expr, 'real'),
        'core': lambda: engine._core.set_mode_rules('real', []),
    }
    for name, call in entries.items():
        with pytest.raises(ValueError, match=_RETIRED if name != 'core' else 'real'):
            call()
    # a config naming a real set, at load
    (tmp_path / 'rules.json').write_text('[]')
    (tmp_path / 'rules_real.json').write_text('[]')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(
        {'engine_generation': 2, 'operators': _OPS, 'rules': 'rules.json',
         'rules_real': 'rules_real.json'}))
    for load in (SimpliPyEngine.from_config, lambda path: SimpliPyEngine.from_config(path, modes='all')):
        with pytest.raises(ValueError, match=_RETIRED):
            load(str(tmp_path / 'config.yaml'))
    with pytest.raises(ValueError, match=_RETIRED):
        SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'), modes=('strict', 'real'))
    assert verify_artifact([], [])['is_clean'] is True
    assert json.loads((tmp_path / 'rules_real.json').read_text()) == []
