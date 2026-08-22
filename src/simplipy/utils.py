"""Helper utilities for manipulating prefix-notation expressions and configs.

Collects the shared, engine-independent building blocks used across SimpliPy: prefix
token-tree operations (traversal, mapping, and remapping of variables and constants),
pattern matching and wildcard handling for rewrite rules, rule deduplication,
constant/placeholder handling, expression compilation and safe evaluation (``codify``,
``safe_f``), small number-theory helpers (``is_prime``, ``factorize_to_at_most``), and
generic nested-container utilities.
"""
import re
import time
import math
import warnings
import itertools
from collections import Counter
from types import CodeType
from typing import Any, Generator, Callable
from copy import deepcopy
from tqdm import tqdm
import numpy as np

# The declared public surface of this module (D11 column R18, owner-ratified
# 2026-08-16): the evidence-backed names. Everything else here is reachable
# but carries no stability promise (see the compatibility policy).
__all__ = [
    'codify', 'deduplicate_rules', 'explicit_constant_placeholders',
    'remap_expression', 'substitute_constants',
    'construct_expressions', 'is_numeric_string',
    'enumerate_expressions', 'count_expressions', 'sample_expression',
    'compositions',
]


def apply_on_nested(structure: list | dict, func: Callable) -> list | dict:
    """Recursively apply a function to all non-structural values in a nested container.

    This function traverses a nested dictionary or list and applies ``func`` to
    every value that is not itself a ``dict`` or ``list``. The original
    ``structure`` is mutated; the same instance is returned for convenience. If
    ``structure`` is neither a list nor a dictionary, it is returned unchanged.

    Parameters
    ----------
    structure : list or dict
        The nested list or dictionary to process.
    func : Callable
        The function to apply to each non-structural value.

    Returns
    -------
    list or dict
        The input ``structure`` with ``func`` applied to all terminal values.

    Examples
    --------
    >>> data = {'a': 1, 'b': {'c': 2, 'd': [{'e': 3}, {'f': 4}, 3]}}
    >>> result = apply_on_nested(data, lambda x: x * 10)
    >>> result
    {'a': 10, 'b': {'c': 20, 'd': [{'e': 30}, {'f': 40}, 30]}}
    >>> data is result
    True
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

    .. warning::
        This compiles arbitrary Python source: it is unsafe by construction
        on attacker-supplied input. Only pass strings you trust — the
        namespace scoping of :meth:`SimpliPyEngine.code_to_lambda` is not a
        sandbox (:mod:`simplipy.trust`).

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
    """Return the names of top-level Python modules referenced in an infix expression.

    The function scans for dotted attribute accesses that look like module
    usages (for example ``numpy.sin(...)`` or ``math.cos(...)``) and collects
    their leading module names. The module ``numpy`` is always included so that
    downstream evaluation logic can rely on it being available.

    .. note::
       This is a plain string scanner and carries no trust decision. Since 0.13 the
       engine derives its own realization roots and checks them against
       :mod:`simplipy.trust` before importing anything (register C1.12) -- consult
       that module, not this function, for what a config is allowed to import.

    Parameters
    ----------
    infix_expression : str
        The mathematical expression in infix notation.

    Returns
    -------
    list[str]
        Unique module names referenced in ``infix_expression``. The order is
        derived from the underlying ``set`` and should be treated as arbitrary.

    Examples
    --------
    >>> sorted(get_used_modules("numpy.sin(x) + math.exp(y)"))
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


_CONSTANT_ID_PATTERN = re.compile(r"C_\d+")


def is_constant_placeholder(token: str, extra: "Any" = ()) -> bool:
    """True iff ``token`` is an abstract-constant slot.

    A slot is the generic ``"<constant>"`` placeholder, an indexed ``C_<n>`` identifier,
    or one of the caller-declared names in ``extra``. This is the ONE definition of
    "placeholder" shared by the mechanical codegen helpers (`explicit_constant_placeholders`,
    `substitute_constants`); deciding which LITERALS become placeholders in the first
    place is a masking-policy question and lives in ``simplipy.masking``.
    """
    return token == "<constant>" or bool(_CONSTANT_ID_PATTERN.match(token)) or token in extra


def substitute_constants(prefix_expression: list[str], values: list | np.ndarray, constants: list[str] | None = None, inplace: bool = False) -> list[str]:
    """Substitute placeholders in a prefix expression with numeric values.

    This helper replaces constant placeholders such as ``"<constant>"`` or the
    tokens listed in ``constants`` with the values supplied in ``values``. Values
    are consumed from left to right as matching tokens are encountered.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression containing constant placeholders.
    values : list or np.ndarray
        The numeric values to substitute into the expression.
    constants : list[str] or None, optional
        An explicit list of placeholder names to be replaced. When ``None``,
        the function considers ``"<constant>"`` and ``C_i`` tokens. Defaults to
        ``None``.
    inplace : bool, optional
        If ``True``, modifies ``prefix_expression`` in-place; otherwise, works on
        a shallow copy. Defaults to ``False``.

    Returns
    -------
    list[str]
        The prefix expression with placeholders replaced by strings holding the
        given numeric values.

    Raises
    ------
    IndexError
        If there are more placeholders than supplied ``values``.

    Examples
    --------
    >>> expr = ['*', '<constant>', '+', 'x', '<constant>']
    >>> substitute_constants(expr, [3.14, 2.71])
    ['*', '3.14', '+', 'x', '2.71']

    >>> expr = ['*', 'C_0', '+', 'x', 'C_1']
    >>> substitute_constants(expr, [3.14, 2.71], constants=['C_0', 'C_1'])
    ['*', '3.14', '+', 'x', '2.71']

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
        if is_constant_placeholder(token, constants):
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


def explicit_constant_placeholders(prefix_expression: list[str], constants: list[str] | None = None, inplace: bool = False, *, convert_numbers_to_constant: bool) -> tuple[list[str], list[str]]:
    """Rename abstract-constant slots to explicit identifiers (``C_0``, ``C_1``, ...).

    A purely MECHANICAL code-generation step: every placeholder slot -- ``"<constant>"``
    or an existing ``C_i`` identifier (re-numbered) -- is renamed to a positional
    constant name, so ``codify`` can build a call signature where constants are passed
    as named arguments. This function decides NOTHING about which literals are
    abstract; that is a masking-policy question and lives in ``simplipy.masking``.

    Parameters
    ----------
    prefix_expression : list[str]
        The prefix expression to process.
    constants : list[str] or None, optional
        Initial constant names to reuse before generating new ones. The returned
        list includes the used values plus any newly generated identifiers.
    inplace : bool, optional
        If ``True``, modifies the input list; otherwise, works on a shallow copy.
        Defaults to ``False``.
    convert_numbers_to_constant : bool, optional
        Deprecated. When ``True``, digit-only numeric tokens are ALSO converted into
        fittable constants -- a masking decision smuggled into a mechanical helper, and
        an incoherent one (``3`` converts, ``-3``/``2.0``/``3.14`` do not; a ``pow``
        exponent's integrality controls the DOMAIN, so converting it silently changes
        semantics). Kept accepted through this release so pinned consumers keep
        working; decide abstraction upstream with ``simplipy.masking`` instead.
        Defaults to ``False`` (it defaulted to ``True`` before 0.13.0).

    Returns
    -------
    tuple[list[str], list[str]]
        Two items: the modified prefix expression and the list of constant
        names used in order of appearance.

    Examples
    --------
    >>> expr = ['*', '<constant>', '+', 'x', '2']
    >>> explicit_constant_placeholders(expr, convert_numbers_to_constant=False)
    (['*', 'C_0', '+', 'x', '2'], ['C_0'])

    >>> explicit_constant_placeholders(['+', 'C_3', '<constant>'], constants=['K'],
    ...                                convert_numbers_to_constant=False)
    (['+', 'K', 'C_0'], ['K', 'C_0'])
    """
    if convert_numbers_to_constant:
        warnings.warn(
            "explicit_constant_placeholders(convert_numbers_to_constant=True) is "
            "deprecated: which literals become fittable constants is a masking-policy "
            "decision -- apply simplipy.masking upstream and let this helper only "
            "rename explicit placeholder slots.",
            DeprecationWarning, stacklevel=2)

    if inplace:
        modified_prefix_expression = prefix_expression
    else:
        modified_prefix_expression = prefix_expression.copy()

    provided_constants = list(constants) if constants is not None else []
    used_constants: list[str] = []
    provided_index = 0
    generated_index = 0

    for i, token in enumerate(prefix_expression):
        # C_i re-numbering is part of the MECHANICAL contract and is unconditional; the
        # deprecated flag only gates the numeral conversion. (Before 0.13.0 the flag
        # gated BOTH, conflating codegen renaming with a masking decision.)
        if is_constant_placeholder(token) or (convert_numbers_to_constant and token.isnumeric()):
            if provided_index < len(provided_constants):
                constant_name = provided_constants[provided_index]
                provided_index += 1
            else:
                constant_name = f"C_{generated_index}"
                generated_index += 1

            modified_prefix_expression[i] = constant_name
            used_constants.append(constant_name)

    return modified_prefix_expression, used_constants


def flatten_nested_list(nested_list: list) -> list[str]:
    """Flatten an arbitrarily nested list into a single list of leaf values.

    A stack-based traversal is used to avoid recursion limits. Because a LIFO
    stack is employed, values appear in reverse depth-first order relative to
    the original nesting. ``list(reversed(...))`` can be used to restore a
    left-to-right ordering if required.

    Parameters
    ----------
    nested_list : list
        The nested list to flatten.

    Returns
    -------
    list[str]
        The flattened list of elements encountered during traversal.

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
    if n < 2:
        # Not prime. Without this guard, 0 and 1 hit an empty divisor range below and
        # `all` vacuously said True; a negative n raised at `math.sqrt` (D2 finding).
        return False
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    """Safely evaluate a compiled function on an array of inputs.

    The callable ``f`` is invoked with the columns of ``X`` unpacked as separate
    arguments, followed by any optional ``constants``. Scalar results are
    broadcast to all samples to guarantee a one-dimensional NumPy array of
    length ``X.shape[0]``.

    Parameters
    ----------
    f : Callable
        The function to evaluate.
    X : np.ndarray
        Two-dimensional array of input samples. Each column is passed as a
        positional argument to ``f``.
    constants : np.ndarray or None, optional
        Extra constant values appended when calling ``f``. Defaults to ``None``.

    Returns
    -------
    np.ndarray
        A one-dimensional array with the evaluation results for each row of
        ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x, y: x + y
    >>> safe_f(f, np.array([[1, 2], [3, 4]]))
    array([3, 7])

    >>> g = lambda x, y, c0: c0
    >>> safe_f(g, np.array([[1, 2], [3, 4]]), constants=np.array([5]))
    array([5, 5])
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


# The dedup key of a rule side: either the engine's INTERNAL FORM (the AC parser
# accepted the side) or, for a side the parser refuses, its raw spelling. The bool
# keeps the two APART -- a tagged canonical key can be spelled exactly like a raw
# token list (`sin _0` is both), and a malformed rule must never collide with a
# well-formed one.
_DedupKey = tuple[bool, tuple[str, ...]]


def _dedup_keyed_rules(
        rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]],
        dummy_variables: list[str],
        engine: Any,
        variable_prefix: str = '?',
        verbose: bool = False) -> list[tuple[_DedupKey, tuple[tuple[str, ...], tuple[str, ...]]]]:
    """The variable-remapped rules paired with their dedup keys, in input order.

    ONE key, every consumer: :func:`deduplicate_rules` folds this list, and the mine's
    per-proposal duplicate accounting folds it identically. A second copy of the keying
    would be free to drift from this one, and the accounting would then report a
    'certified' rule the merge silently discards.
    """
    remapped: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for rule in tqdm(rules_list, desc='Deduplicating rules', disable=not verbose):
        # Rename variables in the source expression.
        #
        # The prefix is the SORT: this remap canonicalises over VARIABLE NAMES, which
        # certifies exactly the `?`-sort (variable-leaf) quantifier -- so that is the sort a
        # freshly mined rule ships with. `_` (arbitrary subtree, pointwise-certified) is
        # EARNED by the post-mine promotion pass, never emitted here. Already-sorted rule
        # sets pass through untouched: `?N`/`_N` tokens are not dummy variables and are
        # never remapped.
        remapped_source, variable_mapping = remap_expression(list(rule[0]), dummy_variables=dummy_variables, variable_prefix=variable_prefix)
        remapped_target, _ = remap_expression(list(rule[1]), dummy_variables, variable_mapping, variable_prefix=variable_prefix)
        remapped.append((tuple(remapped_source), tuple(remapped_target)))

    # THE KEY IS THE INTERNAL FORM, not the spelling (owner ruling 2026-08-18). Batched
    # through the core in ONE call: a ruleset is keyed side-by-side and a per-rule call
    # would re-pay the token-overlay setup for every entry.
    keys = engine._core.ac_canonical_keys([list(source) for source, _ in remapped])
    out: list[tuple[_DedupKey, tuple[tuple[str, ...], tuple[str, ...]]]] = []
    for key, pair in zip(keys, remapped):
        # A side the AC parser refuses keeps its SPELLING as the key: the loader drops
        # such rules anyway (`unparseable-side`), and one malformed entry must not fail
        # a whole ruleset.
        out.append(((True, tuple(key)) if key is not None else (False, pair[0]), pair))
    return out


def deduplicate_rules(rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]], dummy_variables: list[str], verbose: bool = False, variable_prefix: str = '?', *, engine: Any) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Deduplicate a list of simplification rules by canonicalizing variables.

    This function processes a list of (source, target) simplification rules. It
    standardizes the variables in each rule to a canonical form and then

    removes duplicates. If multiple rules simplify to different targets from
    the same canonical source, it keeps the one with the shortest target.

    Two rules are the SAME rule when ``engine`` translates their sources into the same
    internal pattern -- not when they are spelled alike (owner ruling 2026-08-18,
    "instead of comparing spelling, compare internal form"). ``* (-1) asin _0`` and
    ``* asin _0 (-1)`` are one rule; keying on the spelling shipped both, and the second
    could never fire because the first already owns every subject it matches.

    The key is the BARE-context canonical form -- the one
    :func:`~simplipy._core.Engine.ac_canonical_keys` computes, which is literally what
    the rule loader builds for a rule side. It is deliberately NOT
    :meth:`~simplipy.engine.SimpliPyEngine.to_prefix`/``to_tagged``: those run the
    rewrite pass under the LOADED ruleset and the full certificate context, so on an
    engine holding acj-4-3 ``to_prefix(['*','(-1)','asin','_0'])`` returns
    ``asin neg _0`` -- that rule's own target -- while a rules-less engine over the same
    vocabulary returns ``neg asin _0``. A dedup key that depends on the ruleset being
    deduplicated is circular; this one is a pure function of the operator table and
    returns the same answer on both engines.

    Parameters
    ----------
    rules_list : list[tuple[tuple[str, ...], tuple[str, ...]]]
        The list of simplification rules to deduplicate.
    dummy_variables : list[str]
        A list of tokens to be treated as variables for remapping.
    verbose : bool, optional
        If True, displays a progress bar. Defaults to False.
    engine : SimpliPyEngine
        REQUIRED, keyword-only: the engine whose operator table defines the internal
        form. Keyword-only on purpose -- a positional third argument would have
        silently swallowed an existing caller's positional ``verbose``.

    Returns
    -------
    list[tuple[tuple[str, ...], tuple[str, ...]]]
        The deduplicated and optimized list of simplification rules.
    """
    deduplicated_rules: dict[_DedupKey, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for key, pair in _dedup_keyed_rules(rules_list, dummy_variables, engine, variable_prefix, verbose):
        existing = deduplicated_rules.get(key)
        # The WHOLE PAIR wins or loses: a strictly shorter target brings its own source
        # spelling with it, so every emitted rule is a (source, target) that was
        # certified together -- never a cross of one rule's source with another's target.
        if existing is None or len(pair[1]) < len(existing[1]):
            deduplicated_rules[key] = pair

    return list(deduplicated_rules.values())


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
    if not isinstance(s, str):
        return False
    # The ``e+`` arm closes the predicate under the engine's own emissions:
    # ``py_float_repr`` spells big magnitudes ``1e+16`` exactly as Python ``repr``
    # does (H-007). Kept in sync with the Rust ``is_numeric_string``.
    if s.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e+', '', 1).replace('e', '', 1).isdigit():
        return True
    # The AC core's exact-fraction literal ``p/q`` (``1/3``, ``-7/4``): integer '/'
    # integer, one slash -- kept in sync with the Rust ``is_numeric_string``.
    if s.count('/') == 1:
        p, q = s.split('/')
        return p.lstrip('-').isdigit() and q.isdigit()
    return False


_RADIX_DIGITS = {'0x': '0123456789abcdef', '0o': '01234567', '0b': '01'}


def reserved_numeric_spelling(t: str) -> bool:
    """H-007: a spelling a STANDARD numeric reader interprets but the engine's symbolic
    core does not -- refused at every semantic boundary (kept in sync with the Rust
    ``reserved_numeric_spelling``).

    Such a token has two contradictory readings: the AC canon would treat it as a free
    symbol and apply symbol algebra (``inf - inf -> 0``) while Python ``float()`` reads a
    value (``inf - inf = nan``). Three families: textual non-finites (``inf``/
    ``infinity``/``nan``, any case, optional sign), underscore digit groupings
    (``1_000``), and base-prefixed integer literals (``0x10``/``0o17``/``0b101``).
    Canonical numeric literals (``5``, ``+5``, ``1e-05``, ``1/3``, ``np.pi``,
    ``float("inf")``) and genuine free symbols (``x0``, ``_0``, ``x_0``) are NOT
    reserved.
    """
    core = t[1:] if t[:1] in '+-' else t
    # Family 1: textual non-finites.
    if core.lower() in ('inf', 'infinity', 'nan'):
        return True
    # Family 3: base-prefixed integer literals (loose underscore grouping: fail closed).
    lower = core.lower()
    for prefix, digits_ok in _RADIX_DIGITS.items():
        if lower.startswith(prefix):
            digits = lower[len(prefix):].replace('_', '')
            if digits and all(c in digits_ok for c in digits):
                return True
    # Family 2: underscore digit groupings -- every ``_`` strictly between two ASCII
    # digits (Python's float-literal placement rule), and the cleaned spelling must be
    # a float literal (non-finite words were handled above).
    if '_' in t:
        placed = all(
            0 < i < len(t) - 1 and t[i - 1].isdigit() and t[i + 1].isdigit()
            for i, c in enumerate(t) if c == '_'
        )
        if placed:
            try:
                float(t.replace('_', ''))
                return True
            except ValueError:
                pass
    return False


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
        less than or equal to ``max_factor``. The factors are yielded in the
        order they are discovered and are not sorted.

    Raises
    ------
    ValueError
        If ``p`` cannot be decomposed using the specified ``max_factor`` value
        or if ``max_iter`` is exceeded.

    Examples
    --------
    >>> factorize_to_at_most(100, 10)
    [4, 5, 5]
    >>> factorize_to_at_most(18, 5)
    [2, 3, 3]
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


def compositions(total: int, k: int) -> Generator[tuple[int, ...], None, None]:
    """Yield all ordered compositions of ``total`` into ``k`` parts, each part >= 1.

    Deterministic (lexicographic in the first part) -- the expression-universe DP,
    the count DP and the uniform sampler below all iterate compositions in this
    exact order, which is what makes the sampler's cumulative-weight walk valid.
    """
    if k == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - k + 2):
        for rest in compositions(total - first, k - 1):
            yield (first,) + rest


def enumerate_expressions(
        leaf_nodes: list[str],
        non_leaf_nodes: dict[str, int],
        max_length: int) -> dict[int, set[tuple[str, ...]]]:
    """COMPLETE bottom-up enumeration of prefix expressions by length.

    Every prefix expression of length L > 1 decomposes uniquely as a root operator
    plus child expressions of lengths >= 1 summing to L - 1, so filling lengths in
    ascending order is exhaustive by induction -- in contrast to a pass-based
    closure, which stops when the maximum length is REACHED rather than SATURATED
    and silently misses whole expression families.

    The work and memory are exactly the universe size; check
    :func:`count_expressions` FIRST -- complete enumeration is infeasible beyond
    ~1e7 expressions (the dev operator set crosses that between lengths 5 and 6).
    """
    out: dict[int, set[tuple[str, ...]]] = {1: {(leaf,) for leaf in leaf_nodes}}
    for target in range(2, max_length + 1):
        bucket: set[tuple[str, ...]] = set()
        for operator, arity in non_leaf_nodes.items():
            for parts in compositions(target - 1, arity):
                for combination in itertools.product(*(out[part] for part in parts)):
                    bucket.add((operator,) + tuple(itertools.chain.from_iterable(combination)))
        out[target] = bucket
    return out


def count_expressions(n_leaves: int, non_leaf_nodes: dict[str, int], max_length: int) -> dict[int, int]:
    """Count the complete expression universe per length (same DP as ``enumerate_expressions``).

    Cheap (polynomial in ``max_length``), so it can size a universe long before
    enumeration is attempted; ``enumerate_expressions`` results must match these
    counts exactly (two paths, one truth).
    """
    counts = {1: n_leaves}
    for target in range(2, max_length + 1):
        total = 0
        for _operator, arity in non_leaf_nodes.items():
            for parts in compositions(target - 1, arity):
                weight = 1
                for part in parts:
                    weight *= counts[part]
                total += weight
        counts[target] = total
    return counts


def sample_expression(
        length: int,
        leaf_nodes: list[str],
        non_leaf_nodes: dict[str, int],
        counts: dict[int, int],
        rng: np.random.Generator) -> tuple[str, ...]:
    """Draw ONE expression uniformly from the COMPLETE universe of the given length.

    Top-down count-weighted sampling: pick the root operator and the child-length
    composition with probability proportional to the number of expressions they
    root (``counts`` from :func:`count_expressions`), then recurse. Exactly uniform
    over the full universe WITHOUT materializing it -- this is the only sound way
    to represent lengths whose complete universe is too large to enumerate
    (dev operator set: 2.4e8 at length 6, 8.8e9 at length 7).
    """
    if length == 1:
        return (leaf_nodes[int(rng.integers(len(leaf_nodes)))],)
    remaining = int(rng.integers(counts[length]))
    for operator, arity in non_leaf_nodes.items():
        for parts in compositions(length - 1, arity):
            weight = 1
            for part in parts:
                weight *= counts[part]
            if remaining < weight:
                return (operator,) + tuple(itertools.chain.from_iterable(
                    sample_expression(part, leaf_nodes, non_leaf_nodes, counts, rng) for part in parts))
            remaining -= weight
    raise AssertionError(f'count DP inconsistent at length {length}')


def construct_expressions(expressions_of_length: dict[int, set[tuple[str, ...]]], non_leaf_nodes: dict[str, int], must_have_sizes: list | set | None = None) -> Generator[tuple[str, ...], None, None]:
    """Generate new prefix expressions by combining existing building blocks.

    Expressions are grouped by length in ``expressions_of_length``. For each
    operator in ``non_leaf_nodes`` the generator enumerates every compatible
    tuple of child expressions and yields the resulting prefix encoding. When
    ``must_have_sizes`` is provided, at least one operand must have a length
    contained in that collection before the expression is yielded.

    Parameters
    ----------
    expressions_of_length : dict[int, set[tuple[str, ...]]]
        Mapping from expression length to the set of expressions with that
        length.
    non_leaf_nodes : dict[str, int]
        Mapping from operator tokens to their arity.
    must_have_sizes : list or set or None, optional
        If provided, filters generated combinations so that at least one child
        expression has a length contained in this collection. Defaults to
        ``None``.

    Yields
    ------
    tuple[str, ...]
        Newly constructed prefix expressions.

    Examples
    --------
    >>> expressions = {1: {('x',), ('y',)}}
    >>> operators = {'+': 2}
    >>> sorted(construct_expressions(expressions, operators))
    [('+', 'x', 'x'), ('+', 'x', 'y'), ('+', 'y', 'x'), ('+', 'y', 'y')]
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
    """Apply a placeholder-to-subtree mapping to a target expression tree.

    Trees are represented as ``[operator, [operands...]]`` where each operand is
    itself a tree. Leaves are encoded as one-element lists, for example
    ``['x']``. Placeholders such as ``'_0'`` are replaced with the corresponding
    subtree provided in ``mapping``.

    Parameters
    ----------
    tree : list
        The target expression tree containing placeholders.
    mapping : dict[str, Any]
        Dictionary mapping placeholder names to the subtrees that should
        replace them.

    Returns
    -------
    list
        A new expression tree with placeholders substituted.

    Examples
    --------
    >>> template = ['mul', [['_0'], ['_1']]]
    >>> mapping = {'_0': ['x'], '_1': ['add', [['y'], ['z']]]}
    >>> apply_mapping(template, mapping)
    ['mul', [['x'], ['add', [['y'], ['z']]]]]
    """
    # If the tree is a leaf node, replace the placeholder with the actual subtree defined in the mapping
    if len(tree) == 1 and isinstance(tree[0], str):
        if tree[0].startswith(('_', '?', '!')):
            # NOTE: mapping values are complete subtree nodes (lists) as bound by
            # match_pattern, so the lookup is returned as-is: every node in this tree
            # representation is a list (leaves are one-element lists), and wrapping or
            # unwrapping here would break the tree invariant.
            return mapping[tree[0]]
        return tree

    operator, operands = tree
    return [operator, [apply_mapping(operand, mapping) for operand in operands]]


def match_pattern(tree: list, pattern: list, mapping: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
    """Recursively match an expression tree against a pattern tree.

    ``tree`` and ``pattern`` use the same representation as described in
    :func:`apply_mapping`. Placeholders in ``pattern`` (for example ``'_0'``)
    match any subtree. When a match succeeds the mapping is populated with the
    subtrees that correspond to each placeholder.

    Parameters
    ----------
    tree : list
        The expression tree to be matched.
    pattern : list
        The pattern tree to match against.
    mapping : dict[str, Any] or None, optional
        Initial mapping dictionary. If ``None``, an empty one is created.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        ``(True, mapping)`` when the structures align; otherwise ``(False, mapping)``.
        The returned mapping may contain partial assignments even when the match
        fails.

    Examples
    --------
    >>> tree = ['mul', [['x'], ['add', [['y'], ['z']]]]]
    >>> pattern = ['mul', [['_a'], ['_b']]]
    >>> match_pattern(tree, pattern)
    (True, {'_a': ['x'], '_b': ['add', [['y'], ['z']]]})
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
        # Check if the pattern is a placeholder to be filled with the tree.
        # `!` (third sort): this legacy Python path has no interval certifier, so `!` binds
        # single variable leaves ONLY -- fail-closed (the compiled core certifies subtrees).
        # (Known divergence, predating the sorts: this path does not enforce `?` leaf-only
        # either; the deployed simplify line always runs the Rust core.)
        if pattern_key.startswith('!') and not (
                isinstance(tree, str) or (isinstance(tree, list) and len(tree) == 1)):
            return False, mapping
        if pattern_key.startswith(('_', '?', '!')):
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


# four sorts: `_N` any subtree, `!N` a subtree the interval engine certifies
# defined-and-finite a.e., `$N` a subtree certified defined-finite-AND-nonzero a.e.
# (the multiplicative-cancellation sort), `?N` a variable leaf.
_WILDCARD_RE = re.compile(r'^[_?!$]\d+$')


def violates_wildcard_multiplicity(lhs: list[str] | tuple[str, ...], rhs: list[str] | tuple[str, ...]) -> bool:
    """Check whether a rule violates the non-increasing wildcard multiplicity condition.

    A rule ``lhs -> rhs`` violates the condition when any wildcard token
    (matching ``_\\d+``) appears *more* times on the right-hand side than on
    the left-hand side. Enforcing this property prevents duplication of
    wildcard-matched subtrees by ensuring that no wildcard occurs more often
    in the replacement than in the pattern.

    Parameters
    ----------
    lhs : list[str] or tuple[str, ...]
        The source (left-hand side) of the rule in prefix notation.
    rhs : list[str] or tuple[str, ...]
        The target (right-hand side) of the rule in prefix notation.

    Returns
    -------
    bool
        ``True`` if the rule violates the condition (i.e. some wildcard has
        higher multiplicity on the RHS), ``False`` otherwise.
    """
    lhs_wc = Counter(t for t in lhs if _WILDCARD_RE.match(t))
    rhs_wc = Counter(t for t in rhs if _WILDCARD_RE.match(t))
    return any(rhs_wc[w] > lhs_wc[w] for w in rhs_wc)
