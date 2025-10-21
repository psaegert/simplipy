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
    return SimpliPyEngine.load("dev_7-3", install=True)


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
        (['+', 'x3', '*', 'x1', '/', 'x2', '<constant>'], {}, 'x3 + x1 * x2 / <constant>'),
        (['*', 'x1', '/', 'x2', '<constant>'], {}, 'x1 * x2 / <constant>'),
        (['*', 'x1', '*', 'x2', 'x3'], {}, 'x1 * x2 * x3'),
        (['+', 'x1', '+', 'x2', 'x3'], {}, 'x1 + x2 + x3'),
        (['+', 'x1', '-', 'x2', 'x3'], {}, 'x1 + x2 - x3'),
        (['+', 'x', '*', 'y', '/', 'z', '-', 'pow1_2', '+', 'x', 'y', 'pow1_3', 'z'], {}, 'x + y * z / (pow1_2(x + y) - pow1_3(z))'),
        (['-', '*', 'pow1_2', '+', 'x', 'y', '+', '/', '1', '-', 'x', 'y', 'pow1_3', 'z', '*', 'x', '+', 'y', 'z'], {}, 'pow1_2(x + y) * (1 / (x - y) + pow1_3(z)) - x * (y + z)'),
        (['-', 'x', '+', 'y', '*', 'pow1_2', 'z', '-', 'x', '/', '1', '+', 'y', 'z'], {}, 'x - (y + pow1_2(z) * (x - 1 / (y + z)))'),
        (['+', '*', 'pow1_2', '+', 'x', 'y', 'pow1_3', 'z', '*', 'x', '-', 'y', '/', '1', '+', 'z', 'x'], {}, 'pow1_2(x + y) * pow1_3(z) + x * (y - 1 / (z + x))'),
        (['+', '/', 'pow', '+', 'x', 'y', '3', '*', 'pow1_2', 'z', '-', 'x', 'y', '/', '1', '+', 'x', 'y'], {'power': '**'}, '(x + y) ** 3 / (z**(1/2) * (x - y)) + 1 / (x + y)'),
        (['/', 'x', '*', 'y', 'z'], {}, 'x / (y * z)'),
        (['neg', '+', 'x', 'y'], {}, '-(x + y)'),
        (['inv', '+', 'x', 'y'], {}, '1/(x + y)'),
        (['pow1_2', 'x'], {}, 'pow1_2(x)'),
        (['pow1_2', '+', 'x', 'y'], {}, 'pow1_2(x + y)'),
        (['pow1_3', '+', 'x', 'y'], {'power': '**'}, '(x + y)**(1/3)'),
        (['pow2', '+', 'x', 'y'], {'power': '**'}, '(x + y)**2'),
        (['pow3', '*', 'x', 'y'], {'power': '**'}, '(x * y)**3'),
        (['**', 'x', '3'], {'power': '**'}, 'x ** 3'),
        (['pow', 'x', '3'], {'power': 'func'}, 'pow(x, 3)'),
        (['pow', 'x', '3'], {'power': '**'}, 'x ** 3'),
        (['pow', '+', 'x', 'y', '3'], {'power': '**'}, '(x + y) ** 3'),
        (['pow', 'pow1_2', 'x', '3'], {'power': '**'}, '(x**(1/2)) ** 3'),
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
        ['pow1_2', 'x'],
        ['pow1_3', 'x'],
        ['pow2', '+', 'x', 'y'],
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
        (['inv', 'pow2', 'x'], {}),
        (['pow', 'x', '3'], {'power': '**'}),
        (['pow1_2', '+', 'x', 'y'], {'power': '**'}),
        (['pow1_3', '+', 'x', 'y'], {'power': 'func'}),
        (['pow3', '*', 'x', 'y'], {'power': '**'}),
        (['pow', '+', 'x', 'y', '3'], {'power': '**'}),
        (['pow', 'pow1_2', 'x', '3'], {'power': '**'}),
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


COMPLEX_PREFIX_CASES = [
    (
        ['+', '*', '-', 'x', 'pow2', 'y', '/', '1', '-', 'sin', 'x', 'pow1_2', 'z', '*', 'cos', 'y', 'pow1_3', '+', 'x', 'z'],
        {'power': '**'},
    ),
    (
        ['-', '**', '+', 'x', 'y', '3', '/', 'pow1_3', '*', 'x', 'z', '+', 'y', 'pow2', 'neg', 'x'],
        {'power': '**'},
    ),
    (
        ['/', '1', '*', '+', 'x', 'pow1_2', 'y', '**', 'neg', 'x', '+', 'pow1_3', 'y', 'pow1_2', 'z'],
        {'power': '**'},
    ),
]


@pytest.mark.parametrize(("prefix", "kwargs"), COMPLEX_PREFIX_CASES)
def test_prefix_to_infix_complex_stress(engine: SimpliPyEngine, prefix: list[str], kwargs: dict) -> None:
    infix = engine.prefix_to_infix(prefix, **({'power': '**'} | kwargs))
    reconstructed = engine.parse(infix, convert_expression=False)

    canonical_original = tuple(engine.convert_expression(prefix.copy()))
    canonical_roundtrip = tuple(engine.convert_expression(reconstructed.copy()))
    assert canonical_roundtrip == canonical_original

    canonical_original_list = list(canonical_original)
    canonical_roundtrip_list = list(canonical_roundtrip)

    for point in STRESS_POINTS:
        original_value = evaluate_prefix(engine, canonical_original_list, VARIABLES, point, kwargs)
        reconstructed_value = evaluate_prefix(engine, canonical_roundtrip_list, VARIABLES, point, kwargs)
        assert reconstructed_value == pytest.approx(original_value, rel=1e-8, abs=1e-8)
