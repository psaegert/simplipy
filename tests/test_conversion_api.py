"""The conversion API's SURFACE: three forms, three entries, fixed return types
(the research harness; the original brief was CONVERSION_API_DESIGN.md).

D1 changed on 2026-08-18: the entries no longer route through the canonical AC state.
They are PURE, syntactic re-notations -- `tests/test_pure_conversions.py` owns that
contract and its property tests. What stays here is the surface: input detection by
type (D2), the target form fixing the return type (D3), `read_infix` keeping its
tolerant contract (D4), composition with `simplify` (D5), and the error taxonomy (D6).
"""
import numpy as np
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


# One value, spelled in each of the three external forms. The spellings now correspond
# EXACTLY (a conversion preserves them), so this is one expression written three ways.
INFIX = '2 * atan(x0) / 3'
PREFIX = ['/', '*', '2', 'atan', 'x0', '3']
TAGGED = ['<mul>', '2', 'atan', 'x0', '<div>', '3', '</mul>']

INPUT_FORMS = [
    pytest.param(INFIX, id='from-infix'),
    pytest.param(PREFIX, id='from-prefix'),
    pytest.param(TAGGED, id='from-tagged'),
]

# A battery of value-distinct expressions covering bags, pow, structural
# negation, free and named constants -- infix spellings the parser reads.
BATTERY = [
    'x0 + x0',
    '(x0 + 1)**2',
    'x1 - sin(x0)',
    '-x0/3',
    '2*atan(x0)/3',
    '<constant>*x0',
]


class TestNineDirections:
    """3 input forms x 3 targets: the three spellings of one expression convert into
    each other exactly, because a conversion re-notates and nothing else."""

    @pytest.mark.parametrize("source", INPUT_FORMS)
    def test_to_infix(self, engine: SimpliPyEngine, source) -> None:
        out = engine.to_infix(source)
        assert isinstance(out, str)
        assert out == INFIX

    @pytest.mark.parametrize("source", INPUT_FORMS)
    def test_to_prefix(self, engine: SimpliPyEngine, source) -> None:
        out = engine.to_prefix(source)
        assert isinstance(out, list)
        assert out == PREFIX

    @pytest.mark.parametrize("source", INPUT_FORMS)
    def test_to_tagged(self, engine: SimpliPyEngine, source) -> None:
        out = engine.to_tagged(source)
        assert isinstance(out, list)
        assert out == TAGGED


class TestInputContainers:
    """D2/D3: tuple and 1-D ndarray token inputs are read; the RETURN type is
    fixed by the target form, never mirrored from the input container."""

    def test_tuple_input(self, engine: SimpliPyEngine) -> None:
        assert engine.to_tagged(tuple(PREFIX)) == TAGGED
        assert engine.to_prefix(tuple(TAGGED)) == PREFIX

    def test_ndarray_input(self, engine: SimpliPyEngine) -> None:
        arr = np.array(PREFIX)
        assert engine.to_infix(arr) == INFIX
        out = engine.to_prefix(arr)
        assert isinstance(out, list)
        assert out == PREFIX


class TestRoundTrips:
    """D5: a conversion never moves the CANONICAL STATE. The per-direction spelling
    relations live in tests/test_pure_conversions.py."""

    @pytest.mark.parametrize("expr", BATTERY)
    def test_conversion_never_moves_the_state(self, engine: SimpliPyEngine,
                                              expr: str) -> None:
        # to_X(to_Y(e)) reaches the same canonical state as to_X(e), for all 9 pairs.
        converters = {
            'infix': engine.to_infix,
            'prefix': engine.to_prefix,
            'tagged': engine.to_tagged,
        }
        for yname, to_y in converters.items():
            mid = to_y(expr)
            for xname, to_x in converters.items():
                assert engine.simplify(to_x(mid)) == engine.simplify(to_x(expr)), \
                    (xname, yname, expr)

    @pytest.mark.parametrize("expr", BATTERY)
    def test_idempotence(self, engine: SimpliPyEngine, expr: str) -> None:
        for to_x in (engine.to_infix, engine.to_prefix, engine.to_tagged):
            once = to_x(expr)
            assert to_x(once) == once, expr

    @pytest.mark.parametrize("expr", BATTERY)
    def test_simplify_composes_byte_identically(self, engine: SimpliPyEngine,
                                                expr: str) -> None:
        # CONVERT FIRST, then simplify: the composition reaches the canonical state of
        # the target dialect, byte for byte -- the exact migration for the retired
        # `form=` parameter.
        assert engine.simplify(engine.to_infix(expr)) == engine.simplify(expr)
        for src in (expr, engine.to_prefix(expr), engine.to_tagged(expr)):
            assert engine.simplify(engine.to_prefix(src)) \
                == engine.simplify(engine.to_prefix(expr)), (expr, src)
            assert engine.simplify(engine.to_tagged(src)) \
                == engine.simplify(engine.to_tagged(expr)), (expr, src)

    @pytest.mark.parametrize("expr", BATTERY)
    def test_a_conversion_is_the_identity_on_its_own_form(self, engine: SimpliPyEngine,
                                                          expr: str) -> None:
        # `to_X` leaves an already-X answer alone, canonical answers included.
        simplified = engine.simplify(engine.read_infix(expr))
        assert engine.to_prefix(simplified) == list(simplified)
        assert engine.to_tagged(engine.simplify(engine.to_tagged(expr))) \
            == engine.simplify(engine.to_tagged(expr))
        assert engine.to_infix(engine.simplify(expr)) == engine.simplify(expr)

    def test_spelling_IS_preserved(self, engine: SimpliPyEngine) -> None:
        # The 2026-08-18 split: a conversion is notation, so nothing collects.
        assert engine.to_prefix(['+', 'x0', 'x0']) == ['+', 'x0', 'x0']
        assert engine.simplify(['+', 'x0', 'x0']) == ['*', '2', 'x0']
        # D4: read_infix keeps its own (tolerant) contract.
        assert engine.read_infix('x0 + x0') == ['+', 'x0', 'x0']


class TestUnchangedSurface:
    """D7 is RETIRED: the tagged leak is gone by construction (owner ruling
    2026-08-18), because `simplify` answers in the dialect it was handed."""

    def test_simplify_token_default_is_the_input_dialect(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(['*', '2', 'atan', 'x0']) == ['*', '2', 'atan', 'x0']
        assert engine.simplify(['<mul>', '2', 'atan', 'x0', '</mul>']) \
            == ['<mul>', '2', 'atan', 'x0', '</mul>']

    def test_to_prefix_still_reads_a_tagged_answer(self, engine: SimpliPyEngine) -> None:
        out = engine.simplify(['<mul>', '2', 'atan', 'x0', '</mul>'])
        assert engine.to_prefix(out) == ['*', '2', 'atan', 'x0']

    def test_prefix_to_infix_still_refuses_tagged(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match='tagged'):
            engine.prefix_to_infix(TAGGED)


class TestErrors:
    """D6: ValueError on malformed input in any form, TypeError on wrong type."""

    @pytest.mark.parametrize("bad", [
        pytest.param('2*+', id='malformed-infix'),
        pytest.param(['*', '2'], id='missing-operand'),
        pytest.param(['<mul>', '2', 'x0'], id='unclosed-tag'),
        pytest.param(['bogusfn', 'x0'], id='unknown-operator'),
    ])
    def test_malformed_raises_value_error(self, engine: SimpliPyEngine, bad) -> None:
        for to_x in (engine.to_infix, engine.to_prefix, engine.to_tagged):
            with pytest.raises(ValueError):
                to_x(bad)

    @pytest.mark.parametrize("bad", [
        pytest.param(42, id='int'),
        pytest.param({'x0': 1}, id='dict'),
        pytest.param(None, id='none'),
    ])
    def test_wrong_type_raises_type_error(self, engine: SimpliPyEngine, bad) -> None:
        for to_x in (engine.to_infix, engine.to_prefix, engine.to_tagged):
            with pytest.raises(TypeError):
                to_x(bad)

    def test_2d_ndarray_raises_value_error(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError):
            engine.to_infix(np.array([PREFIX, PREFIX]))
