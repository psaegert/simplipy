"""DEFINED FUNCTIONS (ruling 40, 2026-11-26): "We should always support at least that what is
written in the config. This can also be more, e.g. operators which we support but haven't mined
any rules yet."

A config's ``defined_functions:`` section gives ``sqrt(u) = rootn(u, 2)`` and the trigonometric
family completed (the owner, 2026-11-27): ``cot(u) = 1/tan(u)``, ``sec(u) = 1/cos(u)``,
``csc(u) = 1/sin(u)``, ``acot(u) = atan(1/u)``, ``asec(u) = acos(1/u)`` and
``acsc(u) = asin(1/u)``, SymPy's conventions (``acot(-1) = -pi/4``, ``asec(-1) = pi``). The
readers keep the head as written; every canonical entry and every evaluation reads the
definition, so the answer, the price and the value are the definition's; the miner never sees
them. Any other undeclared head is still refused."""

import math
import os
import pickle
import warnings

import numpy as np
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path, legacy_from_config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFINED = {
    'sqrt': {'arity': 1, 'definition': 'rootn(_0, 2)'},
    'cot': {'arity': 1, 'definition': 'inv(tan(_0))'},
    'acot': {'arity': 1, 'definition': 'atan(inv(_0))'},
    'sec': {'arity': 1, 'definition': 'inv(cos(_0))'},
    'csc': {'arity': 1, 'definition': 'inv(sin(_0))'},
    'asec': {'arity': 1, 'definition': 'acos(inv(_0))'},
    'acsc': {'arity': 1, 'definition': 'asin(inv(_0))'},
}
_NAMES = list(_DEFINED)
# (written, its definition written out: the expansion, token for token)
_PAIRS = [
    ('sqrt(x0)', 'rootn(x0, 2)'),
    ('cot(x0)', 'inv(tan(x0))'),
    ('acot(x0)', 'atan(inv(x0))'),
    ('sqrt(x0**2 + x1**2)', 'rootn(x0**2 + x1**2, 2)'),
    ('2*cot(3*x0) + acot(x1/x0)', '2*inv(tan(3*x0)) + atan(inv(x1/x0))'),
    ('sqrt(cot(acot(x0)))', 'rootn(inv(tan(atan(inv(x0)))), 2)'),
    ('x0*sqrt(x0)', 'x0*rootn(x0, 2)'),
    ('sec(x0)', 'inv(cos(x0))'),
    ('csc(x0)', 'inv(sin(x0))'),
    ('asec(x0)', 'acos(inv(x0))'),
    ('acsc(x0)', 'asin(inv(x0))'),
    ('sec(x0)**2 - x1*csc(2*x0)', 'inv(cos(x0))**2 - x1*inv(sin(2*x0))'),
    ('asec(x0 + 1) + acsc(x1/2)', 'acos(inv(x0 + 1)) + asin(inv(x1/2))'),
    ('csc(sec(asec(acsc(x0))))', 'inv(sin(inv(cos(acos(inv(asin(inv(x0))))))))'),
]


def _engines() -> dict[str, SimpliPyEngine]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {'acj-4': legacy_from_config(acj_config_path()),
                'acj2-4': SimpliPyEngine.from_config(
                    os.path.join(_ROOT, 'assets', 'engines', 'acj2-4', 'config.yaml'))}


_ENGINES: dict[str, SimpliPyEngine] = {}


@pytest.fixture(scope='module', params=['acj-4', 'acj2-4'])
def engine(request) -> SimpliPyEngine:
    if not _ENGINES:
        _ENGINES.update(_engines())
    return _ENGINES[request.param]


def test_the_shipped_configs_define_the_three(engine: SimpliPyEngine) -> None:
    assert engine.defined_functions == _DEFINED


class TestReading:
    @pytest.mark.parametrize('name', _NAMES)
    def test_the_head_is_kept_as_written(self, engine: SimpliPyEngine, name: str) -> None:
        assert engine.read_infix(f'{name}(x0)') == [name, 'x0']
        assert engine.read_infix(f'{name} x0^2') == [name, 'pow', 'x0', '2']
        assert engine.to_prefix(f'{name}(x0) + 1') == ['+', name, 'x0', '1']
        assert engine.to_infix([name, '+', 'x0', '1']) == f'{name}(x0 + 1)'
        assert engine.to_tagged(['*', name, 'x0', 'x1']) == ['<mul>', name, 'x0', 'x1', '</mul>']
        assert engine.is_valid([name, 'x0'])
        assert engine.is_valid(f'{name}(x0)')
        assert not engine.is_valid([name, 'x0', 'x1'])
        with pytest.raises(ValueError, match=f'{name} takes 1 argument, got 2'):
            engine.read_infix(f'{name}(x0, x1)')

    def test_a_defined_head_is_not_an_operator(self, engine: SimpliPyEngine) -> None:
        for name in _DEFINED:
            assert name not in engine.operator_arity
            assert name not in engine._operators_config
            assert name not in engine.operator_aliases

    @pytest.mark.parametrize('name', _NAMES)
    def test_the_expansion_is_the_definition_token_for_token(self, engine: SimpliPyEngine, name: str) -> None:
        definition = _DEFINED[name]['definition'].replace('_0', 'x0')
        assert engine._expand_defined([name, 'x0']) == engine.read_infix(definition)

    def test_variables_and_realizations(self, engine: SimpliPyEngine) -> None:
        assert engine.expression_variables('sqrt(x1) + cot(x0)') == ['x1', 'x0']
        assert engine.to_realization(['sqrt', 'x0']) == engine.to_realization(['rootn', 'x0', '2'])


class TestUnknownHeadsAreStillRefused:
    def test_the_tolerant_reader_passes_it_through(self, engine: SimpliPyEngine) -> None:
        assert engine.read_infix('zeta(x0)') == ['zeta', 'x0']

    @pytest.mark.parametrize('entry', ['to_prefix', 'to_infix', 'to_tagged', 'simplify', 'complexity',
                                       'prettify', 'mask', 'evaluate_constants'])
    def test_every_strict_entry_refuses_it(self, engine: SimpliPyEngine, entry: str) -> None:
        with pytest.raises(ValueError, match=r"unknown function 'zeta'"):
            getattr(engine, entry)('zeta(x0) + sqrt(x1)')

    def test_is_valid_says_no(self, engine: SimpliPyEngine) -> None:
        assert engine.is_valid('zeta(x0)') is False
        assert engine.is_valid(['zeta', 'x0']) is False


class TestTheCanonicalEntriesReadTheDefinition:
    @pytest.mark.parametrize('written, expanded', _PAIRS)
    @pytest.mark.parametrize('mode', ['strict', 'permissive'])
    def test_simplify_complexity_prettify_mask(self, engine: SimpliPyEngine, written: str,
                                               expanded: str, mode: str) -> None:
        assert engine.simplify(written, mode=mode) == engine.simplify(expanded, mode=mode)
        prefix, prefix_expanded = engine.to_prefix(written), engine.read_infix(expanded)
        assert engine.simplify(prefix, mode=mode) == engine.simplify(prefix_expanded, mode=mode)
        # a tagged sequence (one that carries a bag)
        tagged = engine.to_tagged(['*', 'x5'] + prefix)
        assert tagged[0] == '<mul>'
        assert engine.simplify(tagged, mode=mode) == engine.simplify(
            engine.to_tagged(['*', 'x5'] + prefix_expanded), mode=mode)
        for certified in (True, False):
            assert engine.complexity(written, certified=certified, mode=mode) == \
                engine.complexity(expanded, certified=certified, mode=mode)
        assert engine.prettify(written, mode=mode) == engine.prettify(expanded, mode=mode)
        assert engine.mask(written) == engine.mask(expanded)
        assert engine.mask(prefix) == engine.mask(prefix_expanded)

    @pytest.mark.parametrize('written', ['cot(x0)', 'x1*cot(x0)', 'x1*csc(2*x0) + sec(x0)', 'acot(x0)',
                                         'x1*asec(x0) - acsc(x1)'])
    def test_a_defined_function_adds_no_fitted_constant_to_a_mask(self, engine: SimpliPyEngine,
                                                                  written: str) -> None:
        # the definitions write `inv`, not `1/`: a written 1 is a literal masking may fit
        literals = [t for t in engine.read_infix(written) if t[0].isdigit()]
        for policy in ('all', 'fittable'):
            assert engine.mask(written, policy).count('<constant>') <= len(literals)

    def test_the_answers_the_brief_names(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify('cot(x0)') == '1/tan(x0)'
        assert engine.simplify('sqrt(x0)') == engine.simplify('rootn(x0, 2)')
        assert engine.simplify(['sqrt', 'x0']) == engine.simplify(['rootn', 'x0', '2'])
        assert engine.simplify('sec(x0)') == '1/cos(x0)'
        assert engine.simplify('csc(x0)') == '1/sin(x0)'
        assert engine.simplify('asec(x0)') == engine.simplify('acos(1/x0)')
        assert engine.simplify('acsc(x0)') == engine.simplify('asin(1/x0)')

    def test_the_mined_rules_apply_through_the_definition(self, engine: SimpliPyEngine) -> None:
        # whatever the rule set does to the definition, it does to the head
        for written, expanded in [('sqrt(x0)**2', 'rootn(x0, 2)**2'), ('cot(x0)*tan(x0)', 'tan(x0)/tan(x0)'),
                                  ('acot(1/x0)', 'atan(x0)'), ('sec(x0)*cos(x0)', 'cos(x0)/cos(x0)'),
                                  ('csc(-x0)', '1/sin(-x0)'), ('asec(1/x0)', 'acos(x0)'),
                                  ('acsc(1/x0)', 'asin(x0)')]:
            for mode in ('strict', 'permissive'):
                assert engine.simplify(written, mode=mode) == engine.simplify(expanded, mode=mode)


# x values over the regions each definition branches on: signed zeros, the poles of cot,
# negatives (rootn's NaN), large and tiny magnitudes, infinities and NaN
_GRID = np.array([0.0, -0.0, 1e-310, -1e-310, 1e-8, -1e-8, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0,
                  math.pi / 2, -math.pi / 2, math.pi, 3.0, -3.0, 10.0, 1e8, -1e8, 1e300, -1e300,
                  math.inf, -math.inf, math.nan])


class TestEvaluation:
    @pytest.mark.parametrize('written, expanded', _PAIRS)
    def test_bit_for_bit_the_definition_on_a_grid(self, engine: SimpliPyEngine, written: str,
                                                  expanded: str) -> None:
        f = engine.as_callable(written, variables=['x0', 'x1'])
        g = engine.as_callable(expanded, variables=['x0', 'x1'])
        x0, x1 = np.meshgrid(_GRID, _GRID)
        with np.errstate(all='ignore'):
            a = np.asarray(f(x0.ravel(), x1.ravel()), dtype=float)
            b = np.asarray(g(x0.ravel(), x1.ravel()), dtype=float)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64))

    @pytest.mark.parametrize('name, definition', [('sqrt', 'rootn({}, 2)'), ('cot', 'inv(tan({}))'),
                                                  ('acot', 'atan(inv({}))'), ('sec', 'inv(cos({}))'),
                                                  ('csc', 'inv(sin({}))'), ('asec', 'acos(inv({}))'),
                                                  ('acsc', 'asin(inv({}))')])
    @pytest.mark.parametrize('x', ['2', '-2', '0.5', '3', '1e-8', '100'])
    def test_constants_fold_to_the_definitions_value(self, engine: SimpliPyEngine, name: str,
                                                     definition: str, x: str) -> None:
        written, expanded = f'{name}({x}) + x0', definition.format(x) + ' + x0'
        value = engine.evaluate_constants(engine.read_infix(definition.format(x)))
        if len(value) != 1:
            # the definition's value is not finite, so it is never substituted: the head stays as
            # written (the expanded spelling may still fold a finite part inside, `1/0.5`)
            assert engine.evaluate_constants(written) == engine.to_infix(written)
        else:
            assert engine.evaluate_constants(written) == engine.evaluate_constants(expanded)
            assert engine.evaluate_constants([name, x]) == value

    def test_the_inverses_take_sympys_convention(self, engine: SimpliPyEngine) -> None:
        assert engine.evaluate_constants('acot(-1)') == repr(-math.pi / 4)
        assert engine.evaluate_constants('acot(1)') == repr(math.pi / 4)
        assert engine.evaluate_constants('asec(-1)') == repr(math.pi)
        assert engine.evaluate_constants('asec(2)') == repr(math.acos(0.5))
        assert engine.evaluate_constants('acsc(-1)') == repr(-math.pi / 2)
        assert engine.evaluate_constants('acsc(2)') == repr(math.asin(0.5))


class TestTheRecipe:
    def test_a_pickle_keeps_the_section(self, engine: SimpliPyEngine) -> None:
        clone = pickle.loads(pickle.dumps(engine))
        assert clone.defined_functions == _DEFINED
        assert clone.simplify('cot(x0)') == engine.simplify('cot(x0)')

    def test_direct_construction(self) -> None:
        operators = {'+': {'realization': '+', 'alias': [], 'arity': 2, 'precedence': 1, 'commutative': True},
                     'tan': {'realization': 'simplipy.operators.tan', 'alias': [], 'arity': 1, 'precedence': 2,
                             'commutative': False}}
        e = SimpliPyEngine(operators=operators, rules=[], defined_functions={'cot': _DEFINED['cot']})
        assert e.simplify('cot(x0) + x1') == e.simplify('1/tan(x0) + x1')
        bare = SimpliPyEngine(operators=operators, rules=[])
        assert bare.defined_functions == {}
        with pytest.raises(ValueError, match="unknown function 'cot'"):
            bare.simplify('cot(x0)')

    @pytest.mark.parametrize('section, message', [
        ({'tan': {'arity': 1, 'definition': '_0'}}, 'already an operator'),
        ({'f': {'arity': 1, 'definition': 'zeta(_0)'}}, "unknown function 'zeta'"),
        ({'f': {'arity': 1, 'definition': '_0 + x1'}}, "'x1'|\"x1\""),
        ({'f': {'arity': 1}}, "exactly an integer 'arity' and a 'definition'"),
        ({'f': {'arity': True, 'definition': '_0'}}, "exactly an integer 'arity'"),
        ('sqrt', 'must be a mapping'),
    ])
    def test_a_bad_section_is_refused_when_the_engine_is_built(self, section, message) -> None:
        operators = {'+': {'realization': '+', 'alias': [], 'arity': 2, 'precedence': 1, 'commutative': True},
                     'tan': {'realization': 'simplipy.operators.tan', 'alias': [], 'arity': 1, 'precedence': 2,
                             'commutative': False}}
        with pytest.raises(ValueError, match=message):
            SimpliPyEngine(operators=operators, rules=[], defined_functions=section)
