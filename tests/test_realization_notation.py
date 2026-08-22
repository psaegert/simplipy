"""REALIZATION is the fourth notation, and the compile pipeline is NOT a notation.

`to_prefix`/`to_infix`/`to_tagged`/`to_realization` are spellings of one expression:
each converts to the others and back. `as_code`/`as_callable` are terminal -- a code
object has no syntax to recover -- so they are named `as_`, and this file pins that
distinction rather than leaving it to the docstrings.
"""
import pytest
import numpy as np

from simplipy import SimpliPyEngine
from conftest import acj_config_path, require_or_skip


@pytest.fixture(scope='module')
def eng() -> SimpliPyEngine:
    require_or_skip(acj_config_path(), 'needs the shipped acj engine')
    return SimpliPyEngine.from_config(acj_config_path())


class TestRealizationIsANotation:
    def test_it_spells_operators_as_their_callables(self, eng) -> None:
        assert eng.to_realization(['sin', 'x0']) == ['simplipy.operators.sin', 'x0']

    def test_operators_realized_as_themselves_are_unchanged(self, eng) -> None:
        # `+` realizes as `+`, so both spellings coincide -- nothing to detect, and
        # nothing that could be misdetected.
        assert eng.to_realization(['+', 'x0', 'x1']) == ['+', 'x0', 'x1']

    @pytest.mark.parametrize('expr', [
        ['+', 'x0', 'sin', 'x1'],
        ['*', '2', 'cos', 'x0'],
        ['/', 'x0', 'exp', 'x1'],
    ])
    def test_every_conversion_reads_realization_back(self, eng, expr) -> None:
        r = eng.to_realization(expr)
        assert eng.to_prefix(r) == eng.to_prefix(expr)
        assert eng.to_infix(r) == eng.to_infix(expr)
        assert eng.to_tagged(r) == eng.to_tagged(expr)

    def test_the_round_trip_is_a_fixpoint(self, eng) -> None:
        expr = ['+', 'x0', 'sin', 'x1']
        assert eng.to_realization(eng.to_prefix(eng.to_realization(expr))) == \
            eng.to_realization(expr)

    def test_it_accepts_every_input_dialect(self, eng) -> None:
        target = eng.to_realization(['+', 'x0', 'sin', 'x1'])
        assert eng.to_realization('x0 + sin(x1)') == target
        assert eng.to_realization(tuple(['+', 'x0', 'sin', 'x1'])) == target
        assert eng.to_realization(np.array(['+', 'x0', 'sin', 'x1'])) == target
        assert eng.to_realization(eng.to_tagged(['+', 'x0', 'sin', 'x1'])) == target


class TestReadingBackNeedsAnInjectiveMap:
    """Two operators may legally share a realization; then the spelling names both and
    no reader can recover which was meant. Refused loudly, not resolved by guessing."""

    def _colliding(self, tmp_path) -> SimpliPyEngine:
        import yaml
        (tmp_path / 'rules.json').write_text('[]')
        ops = {
            '+': {'realization': '+', 'alias': [], 'arity': 2, 'precedence': 1,
                  'commutative': True},
            'abs': {'realization': 'np.abs', 'alias': [], 'arity': 1, 'precedence': 3,
                    'commutative': False},
            'absolute': {'realization': 'np.abs', 'alias': [], 'arity': 1,
                         'precedence': 3, 'commutative': False},
        }
        cfg = tmp_path / 'config.yaml'
        cfg.write_text(yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return SimpliPyEngine.from_config(str(cfg))

    def test_a_colliding_config_refuses_with_both_names(self, tmp_path) -> None:
        eng = self._colliding(tmp_path)
        with pytest.raises(ValueError, match='not invertible'):
            eng._assert_realizations_are_invertible()

    def test_the_shipped_engine_is_invertible(self, eng) -> None:
        eng._assert_realizations_are_invertible()  # must not raise

    def test_writing_never_refuses(self, tmp_path) -> None:
        # Only READING needs injectivity; producing the notation is always well defined.
        eng = self._colliding(tmp_path)
        assert eng.to_realization(['abs', 'x0']) == ['np.abs', 'x0']


class TestTheCompilePipelineIsTerminal:
    def test_as_callable_evaluates(self, eng) -> None:
        f = eng.as_callable('x0 + sin(x1)')
        assert f(1.0, 0.0) == pytest.approx(1.0)
        assert f(0.0, 0.0) == pytest.approx(0.0)

    def test_the_signature_is_first_appearance_order(self, eng) -> None:
        assert eng.expression_variables(['-', 'x1', 'x0']) == ['x1', 'x0']
        f = eng.as_callable(['-', 'x1', 'x0'])
        assert f(5.0, 3.0) == pytest.approx(2.0)   # x1=5, x0=3

    def test_slots_and_specials_are_not_variables(self, eng) -> None:
        assert eng.expression_variables(['*', '<constant>', 'x0']) == ['x0']
        assert eng.expression_variables(['*', 'np.pi', 'x0']) == ['x0']

    def test_as_code_returns_a_code_object(self, eng) -> None:
        from types import CodeType
        assert isinstance(eng.as_code('x0 + 1'), CodeType)

    def test_there_is_no_to_compiled_or_to_lambda(self, eng) -> None:
        """The naming IS the contract: `to_*` round-trips, `as_*` does not. A
        `to_lambda` would advertise a conversion back that cannot exist."""
        assert not hasattr(eng, 'to_compiled')
        assert not hasattr(eng, 'to_lambda')

    def test_a_callable_is_not_accepted_as_conversion_input(self, eng) -> None:
        f = eng.as_callable('x0 + 1')
        with pytest.raises(TypeError, match='conversion expects'):
            eng.to_prefix(f)
