import copy
import os
import time
import re
import math
import yaml
from types import CodeType
from typing import Any, Generator, Callable, Literal, Iterator, Mapping
from copy import deepcopy

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


class GenerationConfig(Mapping[str, Any]):
    '''
    A class to store generation configuration.

    Parameters
    ----------
    method : str, optional, one of 'beam_search' or 'softmax_sampling'
        The generation method to use, by default 'beam_search'.
    **kwargs : Any
        Additional configuration parameters.

    Attributes
    ----------
    method : str
        The generation method to use.
    config : dict
        The configuration dictionary.
    '''
    def __init__(self, method: Literal['beam_search', 'softmax_sampling'] = 'beam_search', **kwargs: Any) -> None:
        self.defaults = {
            'beam_search': {
                'beam_width': 32,
                'max_len': 32,
                'mini_batch_size': 128,
                'equivalence_pruning': True
            },
            'softmax_sampling': {
                'choices': 32,
                'top_k': 0,
                'top_p': 1,
                'max_len': 32,
                'mini_batch_size': 128,
                'temperature': 1,
                'valid_only': True,
                'simplify': True,
                'unique': True
            }
        }

        if method not in self.defaults:
            raise ValueError(f'Invalid generation method: {method}')

        self.method = method

        self.config = dict(**kwargs)

        # Set defaults if not provided
        if method in self.defaults:
            for key, value in self.defaults[method].items():
                if key not in self.config:
                    self.config[key] = value

        for key, value in self.config.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        del self.config[key]
        delattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    # When printed, show the config as a dictionary
    def __repr__(self) -> str:
        return str(self.config)

    def __str__(self) -> str:
        return str(self.config)


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
        if token == "<num>" or re.match(r"C_\d+", token) or token in constants:
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


def numbers_to_num(prefix_expression: list[str], inplace: bool = False) -> list[str]:
    '''
    Replace all numbers in a prefix expression with the string '<num>'.

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
            modified_prefix_expression[i] = '<num>'
        except ValueError:
            modified_prefix_expression[i] = token

    return modified_prefix_expression


def num_to_constants(prefix_expression: list[str], constants: list[str] | None = None, inplace: bool = False, convert_numbers_to_constant: bool = True) -> tuple[list[str], list[str]]:
    '''
    Replace all '<num>' tokens in a prefix expression with constants named 'C_i'.
    This allows the expression to be compiled into a function.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to replace the '<num>' tokens in.
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
        if token == "<num>" or (convert_numbers_to_constant and (re.match(r"C_\d+", token) or token.isnumeric())):
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


def generate_ubi_dist(max_n_operators: int, n_leaves: int, n_unary_operators: int, n_binary_operators: int) -> list[list[int]]:
    '''
    Precompute the number of possible trees for a given number of operators and leaves.

    Parameters
    ----------
    max_n_operators : int
        The maximum number of operators.
    n_leaves : int
        The number of leaves.
    n_unary_operators : int
        The number of unary operators.
    n_binary_operators : int
        The number of binary operators.

    Notes
    -----
    See https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/src/nesymres/dataset/generator.py
    '''
    # enumerate possible trees
    # first generate the tranposed version of D, then transpose it
    D: list[list[int]] = []
    D.append([0] + ([n_leaves ** i for i in range(1, 2 * max_n_operators + 1)]))
    for n in range(1, 2 * max_n_operators + 1):  # number of operators
        s = [0]
        for e in range(1, 2 * max_n_operators - n + 1):  # number of empty nodes
            s.append(n_leaves * s[e - 1] + n_unary_operators * D[n - 1][e] + n_binary_operators * D[n - 1][e + 1])
        D.append(s)
    assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
    D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
    return D


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


def remap_expression(source_expression: list[str], dummy_variables: list[str], variable_mapping: dict | None = None) -> tuple[list[str], dict]:
    source_expression = deepcopy(source_expression)
    if variable_mapping is None:
        variable_mapping = {}
        for i, token in enumerate(source_expression):
            if token in dummy_variables:
                if token not in variable_mapping:
                    variable_mapping[token] = f'_{len(variable_mapping)}'

    for i, token in enumerate(source_expression):
        if token in dummy_variables:
            source_expression[i] = variable_mapping[token]

    return source_expression, variable_mapping


def deduplicate_rules(rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]], dummy_variables: list[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    deduplicated_rules: dict[tuple[str, ...], tuple[str, ...]] = {}
    for rule in rules_list:
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


def is_string_numeric(s: str) -> bool:
    """
    by Cecil Curry
    https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int
    """
    return s.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e', '', 1).isdigit()
