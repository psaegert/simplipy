"""One mode vocabulary at every entry: ``'f64'`` and ``'default'`` name the same mode.

``simplify(mode=)``, ``complexity(mode=)``, ``modes=``, ``rule_mode=``
(``prune_covered_rules``, ``utils.deduplicate_rules``, ``promotion.promote``),
``verify_ruleset(mode=)``, ``mining.verify_mine`` and ``mining.cross_mode_complexity`` all
read a mode through one coercion, and an unknown name raises ``ValueError`` naming both
spellings of the f64 mode.
"""
import pytest

from simplipy import Mode, SimpliPyEngine
from simplipy.engine import _coerce_mode, _coerce_rule_mode

_OPS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 3, "commutative": False},
}

_BOTH = r"'f64' \(also 'default'\) or 'permissive'"


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine(operators=_OPS, rules=[(("*", "1", "?0"), ("?0",))])


@pytest.mark.parametrize('name', ['f64', 'default', 'F64', 'Default', ' default ', Mode.f64])
def test_f64_and_default_are_one_mode(name) -> None:
    assert _coerce_mode(name) is Mode.f64
    assert _coerce_rule_mode(name) == 'default'


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


def test_an_unknown_name_names_both_spellings() -> None:
    with pytest.raises(ValueError, match=_BOTH):
        _coerce_mode('nope')
    with pytest.raises(TypeError, match=_BOTH):
        _coerce_mode(3)


def test_simplify_and_complexity_read_default(engine) -> None:
    expr = ['+', 'x0', 'x0']
    assert engine.simplify(expr, mode='default') == engine.simplify(expr, mode='f64')
    assert engine.complexity(expr, mode='default') == engine.complexity(expr, mode='f64')
    with pytest.raises(ValueError, match=_BOTH):
        engine.simplify(expr, mode='nope')


@pytest.mark.parametrize('name', ['f64', 'default', Mode.f64])
def test_prune_covered_rules_reads_both(name) -> None:
    eng = SimpliPyEngine(operators=_OPS, rules=[(("*", "1", "?0"), ("?0",))])
    assert eng.prune_covered_rules(rule_mode=name) == 1
    assert eng.simplification_rules == []


def test_prune_covered_rules_refuses_an_unknown_mode(engine) -> None:
    with pytest.raises(ValueError, match=_BOTH):
        engine.prune_covered_rules(rule_mode='nope')


def test_deduplicate_rules_reads_both(engine) -> None:
    from simplipy.utils import deduplicate_rules
    rules = [(('*', '(-1)', 'sin', 'x0'), ('neg', 'sin', 'x0')),
             (('*', 'sin', 'x0', '(-1)'), ('neg', 'sin', 'x0'))]
    out = {name: deduplicate_rules(rules, ['x0'], engine=engine, rule_mode=name)
           for name in ('f64', 'default', Mode.f64)}
    assert out['f64'] == out['default'] == out[Mode.f64]
    with pytest.raises(ValueError, match=_BOTH):
        deduplicate_rules(rules, ['x0'], engine=engine, rule_mode='nope')


def test_promote_refuses_an_unknown_mode_before_any_work(engine) -> None:
    from simplipy.promotion import promote
    with pytest.raises(ValueError, match=_BOTH):
        promote([], engine, rule_mode='nope')


def test_verify_mine_refuses_an_unknown_mode_before_reading(tmp_path) -> None:
    from simplipy.mining import verify_mine
    with pytest.raises(ValueError, match=_BOTH):
        verify_mine(str(tmp_path / 'absent.json'), 'nope', _OPS)


def test_verify_mine_reads_default_as_f64(tmp_path) -> None:
    from simplipy.mining import verify_mine
    path = tmp_path / 'rules_f64.json'
    path.write_text('[]')
    reports = {name: verify_mine(str(path), name, _OPS, judge=lambda pairs: ['core'] * len(pairs))
               for name in ('f64', 'default')}
    assert reports['f64']['mode'] == reports['default']['mode'] == 'f64'
    assert reports['f64']['gates'] == reports['default']['gates']


def test_verify_ruleset_refuses_an_unknown_mode() -> None:
    from simplipy.verify import verify_ruleset
    with pytest.raises(ValueError, match=_BOTH):
        verify_ruleset([], mode='nope')


def test_cross_mode_complexity_reads_both(engine) -> None:
    from simplipy.mining import cross_mode_complexity
    expr = ['+', 'x0', 'x0']
    assert cross_mode_complexity(engine, expr, 'default') == cross_mode_complexity(engine, expr, 'f64')


def test_the_mining_surface_declares_its_entries() -> None:
    import simplipy.mining as mining
    assert {'RuleMiner', 'verify_mine', 'cross_mode_complexity'} <= set(mining.__all__)


_RETIRED = 'the real mode is retired'


def test_every_entry_refuses_the_retired_mode(engine, tmp_path) -> None:
    """`real` names no mode at any entry: each one raises the one refusal before any work."""
    import json
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
        SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'), modes=('f64', 'real'))
    assert verify_artifact([], [])['is_clean'] is True
    assert json.loads((tmp_path / 'rules_real.json').read_text()) == []
