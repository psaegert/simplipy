import os
import re
import time
import math
import copy
import yaml
import itertools
from types import CodeType
from typing import Any, Generator, Callable
from copy import deepcopy
from tqdm import tqdm
import numpy as np


def get_path(*args: str, filename: str | None = None, create: bool = False) -> str:
    '''
    Get the path to a file or directory.

    Parameters
    ----------
    args : str
        The path to the file or directory, starting from the root of the project.
    filename : str, optional
        The filename to append to the path, by default None.
    create : bool, optional
        Whether to create the directory if it does not exist, by default False.

    Returns
    -------
    str
        The path to the file or directory.
    '''
    if any(not isinstance(arg, str) for arg in args):
        raise TypeError("All arguments must be strings.")

    path = os.path.join(os.path.dirname(__file__), '..', '..', *args, filename or '')

    if create:
        if filename is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    return os.path.abspath(path)


def substitute_root_path(path: str) -> str:
    '''
    Replace {{ROOT}} with the root path of the project given by get_path().

    Parameters
    ----------
    path : str
        The path to replace

    Returns
    -------
    new_path : str
        The new path with the root path replaced
    '''
    return path.replace(r"{{ROOT}}", get_path())


def load_config(config: dict[str, Any] | str, resolve_paths: bool = True) -> dict[str, Any]:
    '''
    Load a configuration file.

    Parameters
    ----------
    config : dict or str
        The configuration dictionary or path to the configuration file.
    resolve_paths : bool, optional
        Whether to resolve relative paths in the configuration file, by default True.

    Returns
    -------
    dict
        The configuration dictionary.
    '''

    if isinstance(config, str):
        config_path = substitute_root_path(config)
        config_base_path = os.path.dirname(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file {config_path} not found.')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as config_file:
                config_ = yaml.safe_load(config_file)
        else:
            raise ValueError(f'Config file {config_path} is not a valid file.')

        def resolve_path(value: Any) -> str:
            if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.json')) and value.startswith('.'):  # HACK: Find a way to check if a string is a path
                return os.path.join(config_base_path, value)
            return value

        if resolve_paths:
            config_ = apply_on_nested(config_, resolve_path)

    else:
        config_ = config

    return config_


def apply_on_nested(structure: list | dict, func: Callable) -> list | dict:
    '''
    Apply a function to all values in a nested dictionary.

    Parameters
    ----------
    d : list or dict
        The dictionary to apply the function to.
    func : Callable
        The function to apply to the dictionary values.

    Returns
    -------
    dict
        The dictionary with the function applied to all values.
    '''
    if isinstance(structure, list):
        for i, value in enumerate(structure):
            if isinstance(value, dict):
                structure[i] = apply_on_nested(value, func)
            else:
                structure[i] = func(value)
        return structure

    if isinstance(structure, dict):
        for key, value in structure.items():
            if isinstance(value, dict):
                structure[key] = apply_on_nested(value, func)
            else:
                structure[key] = func(value)
        return structure

    return structure


def save_config(config: dict[str, Any], directory: str, filename: str, reference: str = 'relative', recursive: bool = True, resolve_paths: bool = False) -> None:
    '''
    Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save.
    directory : str
        The directory to save the configuration file to.
    filename : str
        The name of the configuration file.
    reference : str, optional
        Determines the reference base path. One of
        - 'relative': relative to the specified directory
        - 'project': relative to the project root
        - 'absolute': absolute paths
    recursive : bool, optional
        Save any referenced configs too
    # '''
    config_ = copy.deepcopy(config)

    def save_config_relative_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = os.path.join('.', os.path.basename(value))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    def save_config_project_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = value.replace(get_path(), '{{ROOT}}')
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    def save_config_absolute_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = os.path.abspath(substitute_root_path(value))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    if recursive:
        match reference:
            case 'relative':
                apply_on_nested(config_, save_config_relative_func)
            case 'project':
                apply_on_nested(config_, save_config_project_func)
            case 'absolute':
                apply_on_nested(config_, save_config_absolute_func)
            case _:
                raise ValueError(f'Invalid reference type: {reference}')

    with open(get_path(directory, filename=filename, create=True), 'w') as config_file:
        yaml.dump(config_, config_file, sort_keys=False)


def traverse_dict(dict_: dict[str, Any]) -> Generator[tuple[str, Any], None, None]:
    '''
    Traverse a dictionary recursively.

    Parameters
    ----------
    d : dict
        The dictionary to traverse.

    Yields
    ------
    tuple
        A tuple containing the key and value of the current dictionary item.
    '''
    for key, value in dict_.items():
        if isinstance(value, dict):
            yield from traverse_dict(value)
        else:
            yield key, value


def codify(code_string: str, variables: list[str] | None = None) -> CodeType:
    '''
    Compile a string into a code object.

    Parameters
    ----------
    code_string : str
        The string to compile.
    variables : list[str] | None
        The variables to use in the code.

    Returns
    -------
    CodeType
        The compiled code object.
    '''
    if variables is None:
        variables = []
    func_string = f'lambda {", ".join(variables)}: {code_string}'
    filename = f'<lambdifygenerated-{time.time_ns()}'
    return compile(func_string, filename, 'eval')


def get_used_modules(infix_expression: str) -> list[str]:
    '''
    Get the python modules used in an infix expression.

    Parameters
    ----------
    infix_expression : str
        The infix expression to parse.

    Returns
    -------
    list[str]
        The python modules used in the expression.
    '''
    # Match the expression against `module.submodule. ... .function(`
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\(')

    # Find all matches in the whole expression
    matches = pattern.findall(infix_expression)

    # Return the unique matches
    modules_set = set(m.split('.')[0] for m in matches)

    modules_set.update(['numpy'])

    return list(modules_set)


def substitude_constants(prefix_expression: list[str], values: list | np.ndarray, constants: list[str] | None = None, inplace: bool = False) -> list[str]:
    '''
    Substitute the numeric placeholders or constants in a prefix expression with the given values.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to substitute the values in.
    values : list | np.ndarray
        The values to substitute in the expression, in order.
    constants : list[str] | None
        The constants to substitute in the expression.
    inplace : bool
        Whether to modify the expression in place.

    Returns
    -------
    list[str]
        The prefix expression with the values substituted.
    '''
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
    '''
    Apply a variable mapping to a prefix expression.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to apply the mapping to.
    variable_mapping : dict[str, str]
        The variable mapping to apply.

    Returns
    -------
    list[str]
        The prefix expression with the variable mapping applied.
    '''
    return list(map(lambda token: variable_mapping.get(token, token), prefix_expression))


def numbers_to_constant(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    '''
    Replace all numbers in a prefix expression with the string '<constant>'.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to replace the numbers in.
    inplace : bool
        Whether to modify the expression in place.

    Returns
    -------
    list[str]
        The prefix expression with the numbers replaced.
    '''
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


def num_to_constants(prefix_expression: list[str], constants: list[str] | None = None, inplace: bool = False, convert_numbers_to_constant: bool = True) -> tuple[list[str], list[str]]:
    '''
    Replace all '<constant>' tokens in a prefix expression with constants named 'C_i'.
    This allows the expression to be compiled into a function.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to replace the '<constant>' tokens in.
    constants : list[str] | None
        The constants to use in the expression.
    inplace : bool
        Whether to modify the expression in place.

    Returns
    -------
    tuple[list[str], list[str]]
        The prefix expression with the constants replaced and the list of constants used.
    '''
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
    '''
    Flatten a nested list.

    Parameters
    ----------
    nested_list : list
        The nested list to flatten.

    Returns
    -------
    list[str]
        The flattened list.
    '''
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
    '''
    Check if a number is prime.

    Parameters
    ----------
    n : int
        The number to check.

    Returns
    -------
    bool
        True if the number is prime, False otherwise.
    '''
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    if constants is None:
        y = f(*X.T)
    else:
        y = f(*X.T, *constants)
    if not isinstance(y, np.ndarray) or y.shape[0] == 1:
        y = np.full(X.shape[0], y)
    return y


def remap_expression(source_expression: list[str], dummy_variables: list[str], variable_mapping: dict | None = None, variable_prefix: str = "_", enumeration_offset: int = 0) -> tuple[list[str], dict]:
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
    """
    by Cecil Curry
    https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int
    """
    return isinstance(s, str) and s.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e', '', 1).isdigit()


def factorize_to_at_most(p: int, max_factor: int, max_iter: int = 1000) -> list[int]:
    '''
    Factorize an integer into factors at most max_factor

    Parameters
    ----------
    p : int
        The integer to factorize
    max_factor : int
        The maximum factor
    max_iter : int, optional
        The maximum number of iterations, by default 1000

    Returns
    -------
    list[int]
        The factors of the integer
    '''
    if is_prime(p):
        return [p]
    p_factors = []
    i = 0
    while p > 1:
        for j in range(max_factor, 0, -1):
            if j == 1:
                p_factors.append(p)
                p = 1
                break
            if p % j == 0:
                p_factors.append(j)
                p //= j
                break
        i += 1
        if i > max_iter:
            raise ValueError(f'Factorization of {p} into at most {max_factor} factors failed after {max_iter} iterations')

    return p_factors


def mask_elementary_literals(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    '''
    Mask elementary literals such as <0> and <1> with <constant>

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression
    inplace : bool, optional
        Whether to modify the expression in place, by default False

    Returns
    -------
    list[str]
        The expression with elementary literals masked
    '''
    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    for i, token in enumerate(prefix_expression):
        if is_numeric_string(token):
            modified_prefix_expression[i] = '<constant>'

    return modified_prefix_expression


def construct_expressions(expressions_of_length: dict[int, set[tuple[str, ...]]], non_leaf_nodes: dict[str, int], must_have_sizes: list | set | None = None) -> Generator[tuple[str, ...], None, None]:
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
    # If the tree is a leaf node, replace the placeholder with the actual subtree defined in the mapping
    if len(tree) == 1 and isinstance(tree[0], str):
        if tree[0].startswith('_'):
            return mapping[tree[0]]  # TODO: I put a bracket here. Find out why this is necessary
        return tree

    operator, operands = tree
    return [operator, [apply_mapping(operand, mapping) for operand in operands]]


def match_pattern(tree: list, pattern: list, mapping: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
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
    filtered_expression = []
    for token in prefix_expression:
        if token == 'pow1':
            continue

        if token == 'pow_1':
            filtered_expression.append('inv')
            continue

        filtered_expression.append(token)

    return filtered_expression


def deparenthesize(term: str) -> str:
    '''
    Removes outer parentheses from a term.

    Parameters
    ----------
    term : str
        The term.

    Returns
    -------
    str
        The term without parentheses.
    '''
    # HACK
    if term.startswith('(') and term.endswith(')'):
        return term[1:-1]
    return term
