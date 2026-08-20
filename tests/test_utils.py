# test_utils.py

import math
import pytest
import numpy as np
from types import CodeType

# Import all functions from the utils module to be tested
from simplipy import utils
from simplipy import SimpliPyEngine


# ==============================================================================
# Test Cases for each function
# ==============================================================================

def test_apply_on_nested():
    """Tests that a function is applied to all non-list/dict elements."""
    data = {'a': 1, 'b': {'c': 2, 'd': [{'e': 3}, {'f': 4}, 5]}}
    expected = {'a': 10, 'b': {'c': 20, 'd': [{'e': 30}, {'f': 40}, 50]}}
    result = utils.apply_on_nested(data, lambda x: x * 10)
    assert result == expected

    data_list = [1, {'a': 2, 'b': [3, 4]}, 5]
    expected_list = [10, {'a': 20, 'b': [30, 40]}, 50]
    result_list = utils.apply_on_nested(data_list, lambda x: x * 10)
    assert result_list == expected_list


def test_traverse_dict():
    """Tests the recursive traversal of a nested dictionary."""
    data = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}
    expected = [('a', 1), ('c', 2), ('d', 3), ('g', 4)]
    result = list(utils.traverse_dict(data))
    assert sorted(result) == sorted(expected)  # Sort for order-insensitivity


def test_codify():
    """Tests the compilation of a string into a code object."""
    code_obj = utils.codify("x + y * 2", variables=['x', 'y'])
    assert isinstance(code_obj, CodeType)
    # The real test is evaluating it
    func = eval(code_obj)
    assert func(5, 3) == 11


def test_get_used_modules():
    """Tests the extraction of module names from an expression string."""
    expression = "numpy.sin(x) + math.exp(y) + custom.module.func(z)"
    expected = ['math', 'numpy', 'custom']
    result = utils.get_used_modules(expression)
    # Using sets to ignore order
    assert set(result) == set(expected)

    # Test that 'numpy' is always included
    expression_no_modules = "x + y"
    assert 'numpy' in utils.get_used_modules(expression_no_modules)




def test_apply_variable_mapping():
    """Tests renaming variables in a prefix expression."""
    expr = ['+', 'var1', '*', 'var2', 'var1']
    mapping = {'var1': 'x', 'var2': 'y'}
    expected = ['+', 'x', '*', 'y', 'x']
    result = utils.apply_variable_mapping(expr, mapping)
    assert result == expected


def test_explicit_constant_placeholders():
    """Tests converting placeholders and numbers to indexed constants like C_0."""
    expr = ['*', '<constant>', '+', 'x', '2.5']
    expected_expr = ['*', 'C_0', '+', 'x', '2.5']
    expected_constants = ['C_0']
    result_expr, result_constants = utils.explicit_constant_placeholders(expr, convert_numbers_to_constant=False)
    assert result_expr == expected_expr
    assert sorted(result_constants) == sorted(expected_constants)


def test_explicit_constant_placeholders_reuses_provided_constants():
    expr = ['+', 'C_3', '<constant>']
    result_expr, result_constants = utils.explicit_constant_placeholders(expr, constants=['K'], convert_numbers_to_constant=False)
    assert result_expr == ['+', 'K', 'C_0']
    assert result_constants == ['K', 'C_0']


def test_explicit_constant_placeholders_discards_unused_constants():
    expr = ['+', '<constant>']
    result_expr, result_constants = utils.explicit_constant_placeholders(expr, constants=['K', 'L'], convert_numbers_to_constant=False)
    assert result_expr == ['+', 'K']
    assert result_constants == ['K']


def test_flatten_nested_list():
    """Tests flattening of a nested list."""
    # Note: The implementation reverses the list, so the test reflects that.
    nested = [1, [2, [3, 4], 5], 6]
    expected = [6, 5, 4, 3, 2, 1]
    assert utils.flatten_nested_list(nested) == expected


@pytest.mark.parametrize("n, expected", [
    (2, True),
    (3, True),
    (29, True),
    (4, False),
    (30, False),
    (91, False),  # 7 * 13
    (1, False),   # D2: vacuous `all` over the empty divisor range said True
    (0, False),
    (-7, False),
])
def test_is_prime(n, expected):
    """Tests the primality test function."""
    assert utils.is_prime(n) == expected


def test_safe_f():
    """Tests the safe evaluation wrapper for functions."""
    X = np.array([[1, 2], [3, 4], [5, 6]])

    # Test with a function that returns a scalar
    def scalar_func(x, y):
        return x + y

    result = utils.safe_f(scalar_func, X)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, np.array([3, 7, 11]))

    # Test with constants
    def const_func(x, y, c1, c2):
        return x * c1 + y * c2

    constants = np.array([10, 2])
    result_const = utils.safe_f(const_func, X, constants)
    np.testing.assert_array_equal(result_const, np.array([14, 38, 62]))


def test_remap_expression():
    """Tests the standardization of variable names."""
    expr = ['+', 'y', '*', 'x', 'y']
    dummy_vars = ['x', 'y']
    remapped_expr, mapping = utils.remap_expression(expr, dummy_vars)

    # The specific mapping can vary, but the structure must be consistent
    expected_expr = ['+', '_0', '*', '_1', '_0']
    expected_mapping = {'y': '_0', 'x': '_1'}

    assert remapped_expr == expected_expr
    assert mapping == expected_mapping


def test_deduplicate_rules():
    """Tests the deduplication of simplification rules."""
    dummy_vars = ['x', 'y', 'z']
    rules = [
        # Duplicate rule with different variable names
        (('+', 'x', 'y'), ('+', 'y', 'x')),
        (('+', 'z', 'y'), ('+', 'y', 'z')),
    ]

    # `engine` is REQUIRED and keyword-only: the dedup key is the engine's internal
    # form, so there is no engine-free reading of "the same rule".
    ops = {'+': {'realization': '+', 'alias': [], 'inverse': '-', 'arity': 2,
                 'precedence': 1, 'commutative': True}}
    engine = SimpliPyEngine(operators=ops, rules=[])
    with pytest.raises(TypeError):
        utils.deduplicate_rules(rules, dummy_vars)   # type: ignore[call-arg]

    deduped = utils.deduplicate_rules(rules, dummy_vars, engine=engine)

    # Canonical forms of the expected rules (?-sorted: the miner's default --
    # dedup emits variable-leaf wildcards)
    expected_rules = [
        (('+', '?0', '?1'), ('+', '?1', '?0')),
    ]

    print(deduped)

    # We need to remap the output to check for canonical equivalence
    remapped_deduped = []
    for src, tgt in deduped:
        rem_src, _ = utils.remap_expression(list(src), dummy_vars)
        rem_tgt, _ = utils.remap_expression(list(tgt), dummy_vars)
        remapped_deduped.append((tuple(rem_src), tuple(rem_tgt)))

    # Use sets to compare as order doesn't matter
    assert set(remapped_deduped) == set(expected_rules)


@pytest.mark.parametrize("s, expected", [
    ("123", True),
    ("-1.5", True),
    ("1.5e-2", True),
    ("abc", False),
    ("1.2.3", False),
    ("", False),
])
def test_is_numeric_string(s, expected):
    """Tests the numeric string check."""
    assert utils.is_numeric_string(s) == expected


def test_factorize_to_at_most():
    """Tests integer factorization with a max factor limit."""
    factors_100 = utils.factorize_to_at_most(100, 10)
    assert math.prod(factors_100) == 100
    assert all(f <= 10 for f in factors_100)

    factors_90 = utils.factorize_to_at_most(90, 10)
    assert math.prod(factors_90) == 90
    assert all(f <= 10 for f in factors_90)

    with pytest.raises(ValueError):
        utils.factorize_to_at_most(99, 10)

    with pytest.raises(ValueError):
        utils.factorize_to_at_most(13, 10)

    # Test for potential infinite loop
    with pytest.raises(ValueError):
        utils.factorize_to_at_most(99, 10, max_iter=1)


def _assert_factorization(p: int, max_factor: int, factors: list[int]) -> None:
    assert math.prod(factors) == p
    assert all(f <= max_factor for f in factors)
    # Adjacent factors should already be maximally packed.
    for left, right in zip(factors, factors[1:]):
        assert left * right > max_factor


@pytest.mark.parametrize(
    "p,max_factor,expected",
    [
        (6, 6, [6]),
        (16, 4, [4, 4]),
        (64, 8, [8, 8]),
        (72, 6, [3, 4, 6]),
        (90, 9, [3, 5, 6]),
        (96, 8, [3, 4, 8]),
        (125, 5, [5, 5, 5]),
        (225, 15, [5, 5, 9]),
        (256, 10, [4, 8, 8]),
        (343, 7, [7, 7, 7]),
    ],
)
def test_factorize_to_at_most_success_profiles(p, max_factor, expected):
    factors = utils.factorize_to_at_most(p, max_factor)
    _assert_factorization(p, max_factor, factors)
    assert sorted(factors) == sorted(expected)


def test_factorize_to_at_most_returns_empty_for_one():
    assert utils.factorize_to_at_most(1, 10) == []


@pytest.mark.parametrize(
    "p,max_factor",
    [
        (97, 10),
        (49, 6),
        (2021, 20),
        (2 * 3 * 5 * 7, 6),
    ],
)
def test_factorize_to_at_most_rejects_large_prime_factors(p, max_factor):
    with pytest.raises(ValueError):
        utils.factorize_to_at_most(p, max_factor)


@pytest.mark.parametrize(
    "p,max_factor,error_message",
    [
        (0, 10, "p must be a positive integer"),
        (-4, 10, "p must be a positive integer"),
        (10, 1, "max_factor must be at least 2"),
    ],
)
def test_factorize_to_at_most_validates_inputs(p, max_factor, error_message):
    with pytest.raises(ValueError) as exc_info:
        utils.factorize_to_at_most(p, max_factor)
    assert error_message in str(exc_info.value)


def test_factorize_to_at_most_max_iter_guard_triggers():
    with pytest.raises(ValueError, match=r"exceeded 3 steps"):
        utils.factorize_to_at_most(16, 4, max_iter=3)


def test_factorize_to_at_most_respects_large_max_iter():
    factors = utils.factorize_to_at_most(3 ** 8, 27, max_iter=20)
    _assert_factorization(3 ** 8, 27, factors)


def test_construct_expressions():
    """Tests the generation of new expressions from smaller ones."""
    expressions_of_length = {
        1: {('x',), ('1',)},
        2: {('sin', 'x')}
    }
    non_leaf_nodes = {'+': 2, 'neg': 1}

    generator = utils.construct_expressions(expressions_of_length, non_leaf_nodes)
    result = set(generator)

    expected_expressions = {
        # Unary op
        ('neg', 'x'),
        ('neg', '1'),
        ('neg', 'sin', 'x'),
        # Binary op
        ('+', 'x', 'x'),
        ('+', 'x', '1'),
        ('+', '1', 'x'),
        ('+', '1', '1'),
        ('+', 'x', 'sin', 'x'),
        ('+', 'sin', 'x', 'x'),
        ('+', '1', 'sin', 'x'),
        ('+', 'sin', 'x', '1'),
        ('+', 'sin', 'x', 'sin', 'x'),
    }
    assert result == expected_expressions


def test_apply_mapping():
    """Tests applying a placeholder-to-subtree mapping."""
    tree = ['+', [['_0'], ['*', [['_1'], ['_0']]]]]
    mapping = {'_0': ['x'], '_1': ['y']}
    expected = ['+', [['x'], ['*', [['y'], ['x']]]]]
    result = utils.apply_mapping(tree, mapping)
    assert result == expected


def test_match_pattern():
    """Tests the structural pattern matching of expression trees."""
    # Simple successful match
    tree1 = ['+', [['x'], ['y']]]
    pattern1 = ['+', [['_0'], ['_1']]]
    does_match, mapping = utils.match_pattern(tree1, pattern1)
    assert does_match
    assert mapping == {'_0': ['x'], '_1': ['y']}

    # Successful match with repeated placeholder
    tree2 = ['+', [['x'], ['x']]]
    pattern2 = ['+', [['_0'], ['_0']]]
    does_match, mapping = utils.match_pattern(tree2, pattern2)
    assert does_match
    assert mapping == {'_0': ['x']}

    # Failed match with repeated placeholder
    tree3 = ['+', [['x'], ['y']]]
    pattern3 = ['+', [['_0'], ['_0']]]
    does_match, _ = utils.match_pattern(tree3, pattern3)
    assert not does_match

    # Failed match with different operator
    tree4 = ['-', [['x'], ['y']]]
    pattern4 = ['+', [['_0'], ['_1']]]
    does_match, _ = utils.match_pattern(tree4, pattern4)
    assert not does_match


def test_remove_pow1():
    """Tests the removal of `pow1` and replacement of `pow_1`."""
    expr = ['pow1', 'x', '+', 'y', 'pow_1', 'z']
    expected = ['x', '+', 'y', 'inv', 'z']
    result = utils.remove_pow1(expr)
    assert result == expected


def test_explicit_constant_placeholders_leaves_numerals_alone_by_default():
    # The mechanical contract since 0.13.0: only explicit placeholder slots are renamed.
    # Digit-only literals -- including pow exponents, whose integrality controls the
    # DOMAIN -- stay concrete; abstraction is simplipy.masking's decision, made upstream.
    expr = ['*', '2', 'pow', 'x1', '3']
    result_expr, result_constants = utils.explicit_constant_placeholders(expr, convert_numbers_to_constant=False)
    assert result_expr == expr
    assert result_constants == []


def test_explicit_constant_placeholders_renumbers_ci_unconditionally():
    # C_i re-numbering is codegen mechanics, NOT gated by the deprecated numeral flag
    # (before 0.13.0 the flag conflated the two).
    expr = ['+', 'C_3', '<constant>', '7']
    result_expr, result_constants = utils.explicit_constant_placeholders(expr, convert_numbers_to_constant=False)
    assert result_expr == ['+', 'C_0', 'C_1', '7']
    assert result_constants == ['C_0', 'C_1']


def test_explicit_constant_placeholders_numeral_conversion_is_deprecated():
    expr = ['*', '2', 'pow', 'x1', '3']
    with pytest.warns(DeprecationWarning, match="masking"):
        result_expr, result_constants = utils.explicit_constant_placeholders(
            expr, convert_numbers_to_constant=True)
    # legacy behavior preserved under the flag: digit-only spellings convert ('2', '3'),
    # while '-3' / '2.0' / '3.14' would not -- the incoherence that got it deprecated
    assert result_expr == ['*', 'C_0', 'pow', 'x1', 'C_1']
    assert result_constants == ['C_0', 'C_1']




def test_is_constant_placeholder_predicate():
    assert utils.is_constant_placeholder('<constant>')
    assert utils.is_constant_placeholder('C_0') and utils.is_constant_placeholder('C_17')
    assert not utils.is_constant_placeholder('3')
    assert not utils.is_constant_placeholder('x1')
    assert not utils.is_constant_placeholder('k1')
    assert utils.is_constant_placeholder('k1', ['k1'])


class TestD12ConvertNumbersRequired:
    """D12 / N1-B2 REVERSED (L6): removing `convert_numbers_to_constant` would break
    symbolic-data 0.15.0 at five sites -- the only downstream that migrated
    correctly -- while doing nothing for flash-ansr's two unflagged call sites. The
    substitute: the parameter STAYS, keyword-only and REQUIRED for one release, so
    an unflagged caller gets a loud TypeError instead of a silently flipped
    masking decision."""

    def test_flag_is_required_and_keyword_only(self):
        from simplipy.utils import explicit_constant_placeholders
        with pytest.raises(TypeError):
            explicit_constant_placeholders(['+', 'x0', '2.5'])          # unflagged
        with pytest.raises(TypeError):
            explicit_constant_placeholders(['+', 'x0', '2.5'], None, False, True)  # positional

    def test_both_explicit_choices_work(self):
        from simplipy.utils import explicit_constant_placeholders
        out, consts = explicit_constant_placeholders(
            ['+', '<constant>', '2.5'], convert_numbers_to_constant=False)
        assert '2.5' in out and len(consts) == 1
        out2, consts2 = explicit_constant_placeholders(
            ['+', '<constant>', '25'], convert_numbers_to_constant=True)
        assert '25' not in out2 and len(consts2) == 2  # digit-only tokens convert
