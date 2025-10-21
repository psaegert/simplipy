import re
import time
import math
import itertools
from types import CodeType
from typing import Any, Generator, Callable
from copy import deepcopy
from tqdm import tqdm
import numpy as np


def apply_on_nested(structure: list | dict, func: Callable) -> list | dict:
    """Recursively apply a function to all non-dict/list values in a nested structure.

    This function traverses a nested dictionary or list and applies the provided
    function `func` to every value that is not a dictionary or a list itself.
    The modification is done in-place.

    Parameters
    ----------
    structure : list or dict
        The nested list or dictionary to process.
    func : Callable
        The function to apply to each non-structural value.

    Returns
    -------
    list or dict
        The modified nested structure with the function applied to its values.

    Examples
    --------
    >>> data = {'a': 1, 'b': {'c': 2, 'd': [{'e': 3}, {'f': 4}, 3]}}
    >>> sp.utils.apply_on_nested(data, lambda x: x * 10)
    {'a': 10, 'b': {'c': 20, 'd': [{'e': 30}, {'f': 40}, 30]}}
    """
    if isinstance(structure, list):
        for i, value in enumerate(structure):
            if isinstance(value, (list, dict)):
                structure[i] = apply_on_nested(value, func)
            else:
                structure[i] = func(value)
        return structure

    if isinstance(structure, dict):
        for key, value in structure.items():
            if isinstance(value, (list, dict)):
                structure[key] = apply_on_nested(value, func)
            else:
                structure[key] = func(value)
        return structure

    return structure


def traverse_dict(dict_: dict[str, Any]) -> Generator[tuple[str, Any], None, None]:
    """Recursively traverse a nested dictionary and yield key-value pairs.

    This generator function walks through a dictionary, descending into any
    nested dictionaries it finds. It yields the key and value for any
    value that is not a dictionary.

    Parameters
    ----------
    dict_ : dict[str, Any]
        The nested dictionary to traverse.

    Yields
    ------
    tuple[str, Any]
        A tuple containing the key and its corresponding non-dictionary value.

    Examples
    --------
    >>> data = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> list(traverse_dict(data))
    [('a', 1), ('c', 2), ('d', 3)]
    """

    for key, value in dict_.items():
        if isinstance(value, dict):
            yield from traverse_dict(value)
        else:
            yield key, value


def codify(code_string: str, variables: list[str] | None = None) -> CodeType:
    """Compile a string expression into a Python code object.

    This function takes a string representing a mathematical expression and
    compiles it into a code object that can be executed later using `eval` or
    converted into a lambda function. It wraps the expression in a lambda
    function signature.

    Parameters
    ----------
    code_string : str
        The mathematical expression string to compile.
    variables : list[str] or None, optional
        A list of variable names to be used as arguments for the lambda
        function, by default None.

    Returns
    -------
    CodeType
        The compiled code object, ready for execution.

    Examples
    --------
    >>> code_obj = codify("x + y", variables=['x', 'y'])
    >>> compiled_func = eval(code_obj)
    >>> compiled_func(2, 3)
    5
    """
    if variables is None:
        variables = []
    func_string = f'lambda {", ".join(variables)}: {code_string}'
    filename = f'<lambdifygenerated-{time.time_ns()}'
    return compile(func_string, filename, 'eval')


def get_used_modules(infix_expression: str) -> list[str]:
    """Extract top-level Python modules used in an infix expression string.

    Parses a string to find all occurrences of module-like function calls
    (e.g., `numpy.sin(...)`, `math.cos(...)`) and returns a unique list of the
    top-level modules. The 'numpy' module is always included by default.

    Parameters
    ----------
    infix_expression : str
        The mathematical expression in infix notation.

    Returns
    -------
    list[str]
        A list of unique top-level module names found in the expression.

    Examples
    --------
    >>> get_used_modules("numpy.sin(x) + math.exp(y)")
    ['math', 'numpy']
    """
    # Match the expression against `module.submodule. ... .function(`
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\(')

    # Find all matches in the whole expression
    matches = pattern.findall(infix_expression)

    # Return the unique matches
    modules_set = set(m.split('.')[0] for m in matches)

    modules_set.update(['numpy'])

    return list(modules_set)


def substitude_constants(prefix_expression: list[str], values: list | np.ndarray, constants: list[str] | None = None, inplace: bool = False) -> list[str]:
    """Substitute placeholders in a prefix expression with numeric values.

    This function replaces constant placeholders like `<constant>` or `C_i`
    in a prefix-notated expression with the provided numerical values in order.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression containing constant placeholders.
    values : list or np.ndarray
        The numerical values to substitute into the expression.
    constants : list[str] or None, optional
        An explicit list of constant names to be replaced, by default None.
    inplace : bool, optional
        If True, modifies the list in-place; otherwise, returns a new list.
        Defaults to False.

    Returns
    -------
    list[str]
        The prefix expression with placeholders replaced by values.

    Examples
    --------
    With default constant placeholders:
    >>> expr = ['*', '<constant>', '+', 'x', '<constant>']
    >>> substitude_constants(expr, [3.14, 2.71], constants=None)
    ['*', '3.14', '+', 'x', '2.71']

    With default constant names:
    >>> expr = ['*', 'C_0', '+', 'x', 'C_1']
    >>> substitute_constants(expr, [3.14, 2.71], constants=['C_0', 'C_1'])
    ['*', '3.14', '+', 'x', '2.71']

    With custom constant names:
    >>> expr = ['*', 'k1', '+', 'x', 'k2']
    >>> substitute_constants(expr, [3.14, 2.71], constants=['k1', 'k2'])
    ['*', '3.14', '+', 'x', '2.71']
    """
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    constant_index = 0
    if constants is None:
        constants = []
    else:
        constants = list(constants)

    for i, token in enumerate(prefix_expression):
        if token == "<constant>" or re.match(r"C_\d+", token) or token in constants:
            modified_prefix_expression[i] = str(values[constant_index])
            constant_index += 1

    return modified_prefix_expression


def apply_variable_mapping(prefix_expression: list[str], variable_mapping: dict[str, str]) -> list[str]:
    """Rename variables in a prefix expression using a mapping.

    Applies a given mapping to rename variables within a prefix expression.
    Any token in the expression that is a key in the mapping will be
    replaced by its corresponding value.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to modify.
    variable_mapping : dict[str, str]
        A dictionary mapping original variable names to new names.

    Returns
    -------
    list[str]
        A new prefix expression with variables renamed.

    Examples
    --------
    >>> expr = ['+', 'var1', 'var2']
    >>> mapping = {'var1': 'x', 'var2': 'y'}
    >>> apply_variable_mapping(expr, mapping)
    ['+', 'x', 'y']
    """
    return list(map(lambda token: variable_mapping.get(token, token), prefix_expression))


def numbers_to_constant(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    """Replace all numeric literals in a prefix expression with '<constant>'.

    This function standardizes an expression by replacing all tokens that can be
    interpreted as numbers with a generic `<constant>` placeholder. This is
    useful for structural comparison and rule matching.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to process.
    inplace : bool, optional
        If True, modifies the list in-place; otherwise, returns a new list.
        Defaults to False.

    Returns
    -------
    list[str]
        The modified prefix expression.

    Examples
    --------
    >>> expr = ['+', 'x', '3.14', '*', 'y', '-2']
    >>> numbers_to_constant(expr)
    ['+', 'x', '<constant>', '*', 'y', '<constant>']
    """
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    for i, token in enumerate(prefix_expression):
        try:
            float(token)
            modified_prefix_expression[i] = '<constant>'
        except ValueError:
            modified_prefix_expression[i] = token

    return modified_prefix_expression


def explicit_constant_placeholders(prefix_expression: list[str], constants: list[str] | None = None, inplace: bool = False, convert_numbers_to_constant: bool = True) -> tuple[list[str], list[str]]:
    """Convert numeric placeholders to indexed constant names (e.g., C_0, C_1).

    Replaces `<constant>` tokens and optionally numeric strings with unique,
    indexed constant names. This prepares the expression for compilation into a
    function where constants are passed as named arguments.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to process.
    constants : list[str] or None, optional
        An initial list of constants to use for naming, by default None.
    inplace : bool, optional
        If True, modifies the list in-place; otherwise, returns a new list.
        Defaults to False.
    convert_numbers_to_constant : bool, optional
        If True, also convert numeric strings to indexed constants.
        Defaults to True.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - The modified prefix expression.
        - The list of constant names used.

    Examples
    --------
    >>> expr = ['*', '<constant>', '+', 'x', '2.5']
    >>> explicit_constant_placeholders(expr)
    (['*', 'C_0', '+', 'x', 'C_1'], ['C_0', 'C_1'])
    """
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    constant_index = 0
    if constants is None:
        constants = []
    else:
        constants = list(constants)

    for i, token in enumerate(prefix_expression):
        if token == "<constant>" or (convert_numbers_to_constant and (re.match(r"C_\d+", token) or token.isnumeric())):
            if constants is not None and len(constants) > constant_index:
                modified_prefix_expression[i] = constants[constant_index]
            else:
                modified_prefix_expression[i] = f"C_{constant_index}"
            constants.append(f"C_{constant_index}")
            constant_index += 1

    return modified_prefix_expression, constants


def flatten_nested_list(nested_list: list) -> list[str]:
    """Flatten an arbitrarily nested list into a single list.

    This function uses a non-recursive, stack-based approach to efficiently
    flatten a nested list structure into a single flat list of elements.

    Parameters
    ----------
    nested_list : list
        The nested list to flatten.

    Returns
    -------
    list[str]
        The flattened list.

    Examples
    --------
    >>> flatten_nested_list([1, [2, [3, 4], 5], 6])
    [6, 5, 4, 3, 2, 1]
    """
    flat_list: list[str] = []
    stack = [nested_list]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            flat_list.append(current)
    return flat_list


def is_prime(n: int) -> bool:
    """Check if an integer is a prime number.

    Determines if the input number `n` is prime. The implementation includes
    optimizations such as checking for even numbers and only testing divisors
    up to the square root of `n`.

    Parameters
    ----------
    n : int
        The integer to check.

    Returns
    -------
    bool
        True if `n` is a prime number, False otherwise.

    Examples
    --------
    >>> is_prime(29)
    True
    >>> is_prime(30)
    False
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    """Safely evaluate a compiled function on an array of inputs.

    This wrapper executes a function `f`, handling optional constants and
    ensuring the output is always a NumPy array of the correct shape, even if
    the function returns a scalar.

    Parameters
    ----------
    f : Callable
        The function to evaluate.
    X : np.ndarray
        The input data array, where rows are samples and columns are features.
    constants : np.ndarray or None, optional
        An array of constant values to pass to the function, by default None.

    Returns
    -------
    np.ndarray
        The result of the function evaluation as a NumPy array.
    """
    if constants is None:
        y = f(*X.T)
    else:
        y = f(*X.T, *constants)
    if not isinstance(y, np.ndarray) or y.shape[0] == 1:
        y = np.full(X.shape[0], y)
    return y


def remap_expression(source_expression: list[str], dummy_variables: list[str], variable_mapping: dict | None = None, variable_prefix: str = "_", enumeration_offset: int = 0) -> tuple[list[str], dict]:
    """Standardize variable names in a prefix expression for canonical representation.

    Remaps variables (identified from `dummy_variables`) to a generic,
    enumerated format (e.g., `_0`, `_1`). This is crucial for comparing the
    structure of two expressions regardless of their original variable names.

    Parameters
    ----------
    source_expression : list[str]
        The prefix expression to remap.
    dummy_variables : list[str]
        A list of tokens to be treated as variables.
    variable_mapping : dict or None, optional
        An existing mapping to apply. If None, a new one is created.
        Defaults to None.
    variable_prefix : str, optional
        The prefix for the new standardized variable names, by default "_".
    enumeration_offset : int, optional
        The starting number for enumeration, by default 0.

    Returns
    -------
    tuple[list[str], dict]
        A tuple containing:
        - The remapped prefix expression.
        - The variable mapping that was created or used.
    """
    source_expression = deepcopy(source_expression)
    if variable_mapping is None:
        variable_mapping = {}
        for i, token in enumerate(source_expression):
            if token in dummy_variables:
                if token not in variable_mapping:
                    variable_mapping[token] = f'{variable_prefix}{len(variable_mapping) + enumeration_offset}'

    for i, token in enumerate(source_expression):
        if token in dummy_variables:
            source_expression[i] = variable_mapping[token]

    return source_expression, variable_mapping


def deduplicate_rules(rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]], dummy_variables: list[str], verbose: bool = False) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Deduplicate a list of simplification rules by canonicalizing variables.

    This function processes a list of (source, target) simplification rules. It
    standardizes the variables in each rule to a canonical form and then

    removes duplicates. If multiple rules simplify to different targets from
    the same canonical source, it keeps the one with the shortest target.

    Parameters
    ----------
    rules_list : list[tuple[tuple[str, ...], tuple[str, ...]]]
        The list of simplification rules to deduplicate.
    dummy_variables : list[str]
        A list of tokens to be treated as variables for remapping.
    verbose : bool, optional
        If True, displays a progress bar. Defaults to False.

    Returns
    -------
    list[tuple[tuple[str, ...], tuple[str, ...]]]
        The deduplicated and optimized list of simplification rules.
    """
    deduplicated_rules: dict[tuple[str, ...], tuple[str, ...]] = {}
    for rule in tqdm(rules_list, desc='Deduplicating rules', disable=not verbose):
        # Rename variables in the source expression
        remapped_source, variable_mapping = remap_expression(list(rule[0]), dummy_variables=dummy_variables)
        remapped_target, _ = remap_expression(list(rule[1]), dummy_variables, variable_mapping)

        remapped_source_key = tuple(remapped_source)
        remapped_target_value = tuple(remapped_target)

        existing_replacement = deduplicated_rules.get(remapped_source_key)
        if existing_replacement is None or len(remapped_target_value) < len(existing_replacement):
            # Found a better (shorter) target expression for the same source
            deduplicated_rules[remapped_source_key] = remapped_target_value

    return list(deduplicated_rules.items())


def is_numeric_string(s: str) -> bool:
    """Check if a string represents a number (integer or float).

    This function determines if the given string can be interpreted as a
    numeric value. It handles integers, floats, and scientific notation.

    Original author: Cecil Curry
    Source: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int

    Parameters
    ----------
    s : str
        The string to check.

    Returns
    -------
    bool
        True if the string represents a number, False otherwise.

    Examples
    --------
    >>> is_numeric_string("123")
    True
    >>> is_numeric_string("-1.5e-2")
    True
    >>> is_numeric_string("abc")
    False
    """
    return isinstance(s, str) and s.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e', '', 1).isdigit()


def factorize_to_at_most(p: int, max_factor: int, max_iter: int = 1000) -> list[int]:
    """Factorize an integer into factors limited by ``max_factor``.

    This helper decomposes ``p`` into a list of factors whose product equals
    ``p`` such that every factor is less than or equal to ``max_factor``. If the
    decomposition is impossible (for example because ``p`` contains a prime
    factor larger than ``max_factor``) a :class:`ValueError` is raised instead of
    returning an invalid factorization.

    Parameters
    ----------
    p : int
        The integer to factorize. Must be greater than or equal to ``1``.
    max_factor : int
        The maximum allowable value for any single factor. Must be at least 2.
    max_iter : int, optional
        A soft cap on the number of prime factors processed. If the algorithm
        exceeds this limit, a :class:`ValueError` is raised to guard against
        accidental infinite loops.

    Returns
    -------
    list[int]
        The factors of ``p``. Their product is equal to ``p`` and each factor is
        less than or equal to ``max_factor``.

    Raises
    ------
    ValueError
        If ``p`` cannot be decomposed using the specified ``max_factor`` value
        or if ``max_iter`` is exceeded.

    Examples
    --------
    >>> factorize_to_at_most(100, 10)
    [10, 10]
    >>> factorize_to_at_most(18, 5)
    [3, 3, 2]
    """

    if p < 1:
        raise ValueError("p must be a positive integer")
    if max_factor < 2:
        raise ValueError("max_factor must be at least 2")

    if p == 1:
        return []

    remaining = p
    factors: list[int] = []
    current_factor = 1
    processed_factors = 0

    def flush_current() -> None:
        nonlocal current_factor
        if current_factor > 1:
            factors.append(current_factor)
            current_factor = 1

    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            processed_factors += 1
            if processed_factors > max_iter:
                raise ValueError(
                    f'Factorization of {p} into factors <= {max_factor} exceeded {max_iter} steps')

            if divisor > max_factor:
                raise ValueError(f'Cannot factorize {p} with factors <= {max_factor}')

            if current_factor * divisor <= max_factor:
                current_factor *= divisor
            else:
                flush_current()
                current_factor = divisor

            remaining //= divisor
        divisor = 3 if divisor == 2 else divisor + 2

    if remaining > 1:
        # remaining is prime at this point
        if remaining > max_factor:
            raise ValueError(f'Cannot factorize {p} with factors <= {max_factor}')

        if current_factor * remaining <= max_factor:
            current_factor *= remaining
        else:
            flush_current()
            current_factor = remaining

    flush_current()

    return factors


def mask_elementary_literals(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    """Replace all numeric string literals with the '<constant>' token.

    Scans a prefix expression and replaces any token that represents a number
    (e.g., "0", "1", "3.14") with the generic placeholder "<constant>". This is
    used to abstract away specific numbers for general simplification rules.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to modify.
    inplace : bool, optional
        If True, modifies the list in-place; otherwise, returns a new list.
        Defaults to False.

    Returns
    -------
    list[str]
        The expression with numeric literals masked.
    """
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    for i, token in enumerate(prefix_expression):
        if is_numeric_string(token):
            modified_prefix_expression[i] = '<constant>'

    return modified_prefix_expression


def construct_expressions(expressions_of_length: dict[int, set[tuple[str, ...]]], non_leaf_nodes: dict[str, int], must_have_sizes: list | set | None = None) -> Generator[tuple[str, ...], None, None]:
    """Generate new, larger expressions by combining existing smaller ones.

    This generator function builds complex mathematical expressions by taking a
    set of existing expressions (grouped by length) and combining them using
    a given set of operators (non-leaf nodes). It systematically creates all
    possible new valid expressions.

    Parameters
    ----------
    expressions_of_length : dict[int, set[tuple[str, ...]]]
        A dictionary mapping expression length to a set of expressions of that
        length. These are the building blocks.
    non_leaf_nodes : dict[str, int]
        A dictionary of operators, mapping the operator token to its arity.
    must_have_sizes : list or set or None, optional
        If provided, only generates combinations where at least one child
        expression has a length present in this set. This is an optimization
        to avoid redundant constructions. Defaults to None.

    Yields
    ------
    tuple[str, ...]
        A new, valid prefix expression constructed from the inputs.
    """
    expressions_of_length_with_lists = {k: list(v) for k, v in expressions_of_length.items()}

    filter_sizes = must_have_sizes is not None and not len(must_have_sizes) == 0
    if must_have_sizes is not None and filter_sizes:
        must_have_sizes_set = set(must_have_sizes)

    # Append existing trees to every operator
    for new_root_operator, arity in non_leaf_nodes.items():
        # Start with the smallest arity-tuples of trees
        for child_lengths in sorted(itertools.product(list(expressions_of_length_with_lists.keys()), repeat=arity), key=lambda x: sum(x)):
            # Check all possible combinations of child trees
            if filter_sizes and not any(length in must_have_sizes_set for length in child_lengths):
                # Skip combinations that do not have any of the required sizes (e.g. duplicates is used correctly)
                continue
            for child_combination in itertools.product(*[expressions_of_length_with_lists[child_length] for child_length in child_lengths]):
                yield (new_root_operator,) + tuple(itertools.chain.from_iterable(child_combination))


def apply_mapping(tree: list, mapping: dict[str, Any]) -> list:
    """Apply a variable mapping to a target expression tree.

    This function is used after a successful pattern match. It takes a target
    expression tree (which may contain placeholders like `_0`, `_1`) and a
    mapping from those placeholders to actual subtrees. It returns a new tree
    where all placeholders have been replaced by their corresponding subtrees.

    Parameters
    ----------
    tree : list
        The target expression tree containing placeholders.
    mapping : dict[str, Any]
        The dictionary mapping placeholders to subtrees.

    Returns
    -------
    list
        The new expression tree with placeholders substituted.
    """
    # If the tree is a leaf node, replace the placeholder with the actual subtree defined in the mapping
    if len(tree) == 1 and isinstance(tree[0], str):
        if tree[0].startswith('_'):
            return mapping[tree[0]]  # TODO: I put a bracket here. Find out why this is necessary
        return tree

    operator, operands = tree
    return [operator, [apply_mapping(operand, mapping) for operand in operands]]


def match_pattern(tree: list, pattern: list, mapping: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
    """Recursively match an expression tree against a pattern tree.

    This function performs structural pattern matching. It checks if `tree`
    conforms to the structure of `pattern`. The pattern can contain
    placeholders (e.g., `_0`, `_1`) which match any subtree. If a match is
    found, it returns True and a dictionary mapping the placeholders to the
    subtrees they matched.

    Parameters
    ----------
    tree : list
        The expression tree to be matched.
    pattern : list
        The pattern tree to match against.
    mapping : dict[str, Any] or None, optional
        An initial mapping dictionary. If None, an empty one is created.
        Defaults to None.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        A tuple containing:
        - A boolean indicating if the match was successful.
        - The dictionary mapping placeholders to the matched subtrees.
    """
    if mapping is None:
        mapping = {}

    pattern_length = len(pattern)

    # The leaf node is a variable but the pattern is not
    if len(tree) == 1 and isinstance(tree[0], str) and pattern_length != 1:
        return False, mapping

    # Elementary pattern
    pattern_key = pattern[0]
    if pattern_length == 1 and isinstance(pattern_key, str):
        # Check if the pattern is a placeholder to be filled with the tree
        if pattern_key.startswith('_'):
            # Try to match the tree with the placeholder pattern
            existing_value = mapping.get(pattern_key)
            if existing_value is None:
                # Placeholder is not yet filled, can be filled with the tree
                mapping[pattern_key] = tree
                return True, mapping
            else:
                # The placeholder has a mapped value already

                # If the existing value is a constant, it is not a match
                # We cannot map multiple (independent) constants to the same placeholder
                if "<constant>" in flatten_nested_list(existing_value):
                    return False, mapping

                # Placeholder is occupied by another tree, check if the existing value matches the tree
                return (existing_value == tree), mapping

        # The literal pattern must match the tree
        return (tree == pattern), mapping

    # The pattern is tree-structured
    tree_operator, tree_operands = tree
    pattern_operator, pattern_operands = pattern

    # If the operators do not match, the tree does not match the pattern
    if tree_operator != pattern_operator:
        return False, mapping

    # Try to recursively match the operands
    for tree_operand, pattern_operand in zip(tree_operands, pattern_operands):
        # If the pattern operand is a leaf node
        if isinstance(pattern_operand, str):
            # Check if the pattern operand is a placeholder to be filled with the tree operand
            existing_value = mapping.get(pattern_operand)
            if existing_value is None:
                # Placeholder is not yet filled, can be filled with the tree operand
                mapping[pattern_operand] = tree_operand
                return True, mapping
            elif existing_value != tree_operand:
                # Placeholder is occupied by another tree, the tree does not match the pattern
                return False, mapping
        else:
            # Recursively match the tree operand with the pattern operand
            does_match, mapping = match_pattern(tree_operand, pattern_operand, mapping)

            # If the tree operand does not match the pattern operand, the tree does not match the pattern
            if not does_match:
                return False, mapping

    # The tree matches the pattern
    return True, mapping


def remove_pow1(prefix_expression: list[str]) -> list[str]:
    """Remove identity power operations from a prefix expression.

    This utility cleans up an expression by removing `pow1` operators, which
    represent raising to the power of 1 (an identity operation), and replaces
    `pow_1` (power of -1) with its canonical equivalent, `inv`.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to clean.

    Returns
    -------
    list[str]
        The cleaned prefix expression without `pow1` or `pow_1` tokens.

    Examples
    --------
    >>> expr = ['pow1', 'x', '+', 'y', 'pow_1', 'z']
    >>> remove_pow1(expr)
    ['x', '+', 'y', 'inv', 'z']
    """
    filtered_expression = []
    for token in prefix_expression:
        if token == 'pow1':
            continue

        if token == 'pow_1':
            filtered_expression.append('inv')
            continue

        filtered_expression.append(token)

    return filtered_expression
