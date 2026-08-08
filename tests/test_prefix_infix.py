import pytest

import simplipy as sp
from simplipy import SimpliPyEngine


VARIABLES = ['x', 'y', 'z']
TEST_POINT = (0.7, -1.3, 2.5)
STRESS_POINTS = [
    (-0.7, 1.3, 2.5),
    (-0.2, 0.9, 1.1),
    (-1.5, -0.4, 3.2),
]


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    # Raw legacy TABLE from the in-repo fixture (conversion input-language tests
    # need no rules and no asset); the public artifact-load path refuses
    # generation 1 (see tests/conftest.py).
    from conftest import construct_legacy_table
    return construct_legacy_table()


@pytest.mark.parametrize(
    ("prefix", "kwargs", "expected"),
    [
        (['+', 'x', 'y'], {}, 'x + y'),
        (['*', '+', 'x', 'y', 'z'], {}, '(x + y) * z'),
        (['*', 'x', '+', 'y', 'z'], {}, 'x * (y + z)'),
        (['-', 'x', '*', 'y', 'z'], {}, 'x - y * z'),
        (['-', '+', 'x', 'y', 'z'], {}, 'x + y - z'),
        (['-', 'x', '-', 'y', 'z'], {}, 'x - (y - z)'),
        (['+', 'x', '*', 'y', 'z'], {}, 'x + y * z'),
        (['*', '-', 'x', 'y', '+', 'z', 'w'], {}, '(x - y) * (z + w)'),
        (['+', 'x3', '*', 'x1', '/', 'x2', '<constant>'], {}, 'x3 + x1 * (x2 / <constant>)'),
        (['*', 'x1', '/', 'x2', '<constant>'], {}, 'x1 * (x2 / <constant>)'),
        (['*', 'x1', '*', 'x2', 'x3'], {}, 'x1 * (x2 * x3)'),
        (['+', 'x1', '+', 'x2', 'x3'], {}, 'x1 + (x2 + x3)'),
        (['+', 'x1', '-', 'x2', 'x3'], {}, 'x1 + (x2 - x3)'),
        (['/', 'x', '*', 'y', 'z'], {}, 'x / (y * z)'),
        (['neg', '+', 'x', 'y'], {}, '-(x + y)'),
        (['inv', '+', 'x', 'y'], {}, '1/(x + y)'),
        (['**', 'x', '3'], {'power': '**'}, 'x ** 3'),
        (['pow', 'x', '3'], {'power': 'func'}, 'pow(x, 3)'),
        (['pow', 'x', '3'], {'power': '**'}, 'x ** 3'),
        (['pow', '+', 'x', 'y', '3'], {'power': '**'}, '(x + y) ** 3'),
        (['sin', '+', 'x', 'y'], {}, 'sin(x + y)'),
        (
            ['sin', '+', 'x', 'y'],
            {'realization': True},
            'simplipy.operators.sin(x + y)'
        ),
    ],
)
def test_prefix_to_infix_expected_output(
    engine: SimpliPyEngine,
    prefix: list[str],
    kwargs: dict,
    expected: str,
) -> None:
    result = engine.prefix_to_infix(prefix, **kwargs)
    assert result == expected


@pytest.mark.parametrize(
    "prefix",
    [
        ['+', 'x', 'y'],
        ['*', '+', 'x', 'y', 'z'],
        ['neg', '+', 'x', 'y'],
        ['/', 'x', '+', 'y', 'z'],
        ['/', '+', 'x', 'y', 'z'],
        ['**', 'x', '3'],
        ['sin', '+', 'x', 'y'],
    ],
)
def test_prefix_to_infix_roundtrip_preserves_structure(engine: SimpliPyEngine, prefix: list[str]) -> None:
    infix = engine.prefix_to_infix(prefix, power='**')
    reconstructed = engine.parse(infix, convert_expression=False)

    canonical_original = tuple(engine.convert_expression(prefix.copy()))
    canonical_roundtrip = tuple(engine.convert_expression(reconstructed.copy()))

    assert canonical_roundtrip == canonical_original


def test_prefix_to_infix_raises_on_extra_operands(engine: SimpliPyEngine) -> None:
    malformed_prefix = ['x', 'y', 'z']

    with pytest.raises(ValueError, match='Malformed prefix expression'):
        engine.prefix_to_infix(malformed_prefix)


def test_prefix_to_infix_names_the_tagged_dialect(engine: SimpliPyEngine) -> None:
    # simplify's default token output is the TAGGED serialization; this converter reads
    # the explicit binary-prefix dialect. The refusal must name the dialect and both
    # escapes, not fail with a bare arity error (B1).
    tagged = ['<mul>', '<constant>', '<div>', 'log', 'x3', '</mul>']
    with pytest.raises(ValueError, match='tagged-serialization tokens') as exc_info:
        engine.prefix_to_infix(tagged)
    message = str(exc_info.value)
    assert "form='infix'" in message
    assert "form='explicit'" in message


def test_is_valid_verbose_names_the_tagged_dialect(engine: SimpliPyEngine, capsys) -> None:
    # Same seam on the validity side: the verdict stays False (core-owned, explicit
    # dialect only), but the verbose diagnostic must name the tagged dialect instead of
    # the misleading 'Variable must be leaf node' (B1).
    tagged = ['<add>', '<mul>', '2', 'x0', '</mul>', '1', '</add>']
    assert engine.is_valid(tagged, verbose=True) is False
    out = capsys.readouterr().out
    assert 'tagged-serialization' in out
    assert "form='explicit'" in out
    assert 'Variable must be leaf node' not in out


@pytest.mark.parametrize(
    "infix",
    [
        'x + y * z',
        '(x + y) * z',
        'x * (y + z)',
        'x ** (y + z)',
        '1/(x + y)',
        '-(x + y) + z',
        'x**2 + y**2',
        'pow1_2(x + y)',
        'pow1_3(x * y)',
        'sin(x + y)',
        'sin(x) + cos(y)',
        'x / (y * z)',
    ],
)
def test_infix_to_prefix_roundtrip_preserves_semantics(engine: SimpliPyEngine, infix: str) -> None:
    prefix = engine.parse(infix, convert_expression=False)
    roundtrip_infix = engine.prefix_to_infix(prefix, power='**')
    roundtrip_prefix = engine.parse(roundtrip_infix, convert_expression=False)

    canonical_original = tuple(engine.convert_expression(prefix.copy()))
    canonical_roundtrip = tuple(engine.convert_expression(roundtrip_prefix.copy()))

    assert canonical_roundtrip == canonical_original


def test_parse_handles_scientific_notation(engine: SimpliPyEngine) -> None:
    tokens = engine.parse('1.234e-5 * sin(v1)', convert_expression=False)
    assert tokens == ['*', '1.234e-5', 'sin', 'v1']

    canonical_tokens = engine.parse('1.234e-5 * sin(v1)')
    assert canonical_tokens == ['*', '1.234e-5', 'sin', 'v1']

    rendered = engine.prefix_to_infix(canonical_tokens)
    assert rendered == '1.234e-5 * sin(v1)'


def test_parse_handles_caret_power(engine: SimpliPyEngine) -> None:
    # caret '^' should be accepted as power and be semantically equivalent to '**'
    # We don't assert the exact unconverted token layout (implementation details may vary
    # between engines), but the canonical converted form must represent a power
    # and the roundtrip infix must be a power expression.
    canonical = engine.parse('x1 ^ 3')
    assert isinstance(canonical, list) and len(canonical) >= 1

    rendered = engine.prefix_to_infix(canonical, power='**')
    # Accept either 'x1**3' or 'x1 ** 3' formatting
    assert rendered.replace(' ', '') == 'x1**3'


def evaluate_prefix(
        engine: SimpliPyEngine,
        prefix: list[str],
        variables: list[str],
        values: tuple[float, ...],
        kwargs: dict | None = None) -> float:
    kwargs = kwargs or {}
    eval_kwargs = {'power': kwargs.get('power', '**')}
    # Always use realization for executable Python code
    eval_kwargs['realization'] = True
    infix = engine.prefix_to_infix(prefix, **eval_kwargs)
    code = sp.codify(infix, variables)
    func = engine.code_to_lambda(code)
    return func(*values)


@pytest.mark.parametrize(
    ("prefix", "kwargs"),
    [
        (['inv', '*', 'x', 'y'], {}),
        (['pow', 'x', '3'], {'power': '**'}),
        (['pow', '+', 'x', 'y', '3'], {'power': '**'}),
        (['**', 'x', '3'], {'power': '**'}),
    ],
)
def test_prefix_to_infix_roundtrip_functionally_equivalent(
    engine: SimpliPyEngine,
    prefix: list[str],
    kwargs: dict,
) -> None:
    infix = engine.prefix_to_infix(prefix, **({'power': '**'} | kwargs))
    reconstructed = engine.parse(infix, convert_expression=False)

    original_value = evaluate_prefix(engine, prefix, VARIABLES, TEST_POINT)
    reconstructed_value = evaluate_prefix(engine, reconstructed, VARIABLES, TEST_POINT, kwargs)

    assert reconstructed_value == pytest.approx(original_value, rel=1e-9, abs=1e-9)
