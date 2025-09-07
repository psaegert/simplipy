import re
import importlib
import fractions
import os
import warnings
import multiprocessing as mp
import queue
import time
import signal
import pprint
from types import CodeType, FunctionType
from typing import Callable
from multiprocessing import Queue, Process
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Literal
from copy import deepcopy
from math import prod
from collections import defaultdict
from itertools import product

import numpy as np
import json
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm import tqdm

from simplipy.utils import (
    factorize_to_at_most, is_numeric_string, load_config,
    get_used_modules, numbers_to_constant, flatten_nested_list, is_prime, num_to_constants,
    codify, safe_f, deduplicate_rules, mask_elementary_literals as mask_elementary_literals_fn,
    construct_expressions, apply_mapping, match_pattern, remove_pow1)


class SimpliPyEngine:
    """
    Management and manipulation of expressions / equations with properties and methods for parsing, encoding, decoding, and transforming equations

    Parameters
    ----------
    operators : dict[str, dict[str, Any]]
        A dictionary of operators with their properties
    rules : list[tuple] | None
        A list of simplification rules. If None, the engine will have no rules.
    """
    def __init__(self, operators: dict[str, dict[str, Any]], rules: list[tuple] | None = None) -> None:
        # This part, which sets up all the operator properties, is unchanged.
        self.operator_tokens = list(operators.keys())
        self.operator_aliases = {alias: operator for operator, properties in operators.items() for alias in properties['alias']}
        self.operator_inverses = {k: v["inverse"] for k, v in operators.items() if v.get("inverse") is not None}

        self.inverse_base = {'*': ['inv', '/', '1'], '+': ['neg', '-', '0']}
        self.inverse_unary = {v[0]: [k, v[1], v[2]] for k, v in self.inverse_base.items()}
        self.inverse_binary = {v[1]: [k, v[0], v[2]] for k, v in self.inverse_base.items()}

        self.unary_mult_div_operators = {k: v["inverse"] for k, v in operators.items() if k.startswith('mult') or k.startswith('div')}
        self.commutative_operators = [k for k, v in operators.items() if v.get("commutative", False)]

        self.operator_realizations = {k: v["realization"] for k, v in operators.items()}
        self.realization_to_operator = {v: k for k, v in self.operator_realizations.items()}

        self.operator_precedence_compat = {k: v.get("precedence", i) for i, (k, v) in enumerate(operators.items())}
        self.operator_precedence_compat['**'] = 3
        self.operator_precedence_compat['sqrt'] = 3

        self.operator_arity = {k: v["arity"] for k, v in operators.items()}
        self.operator_arity_compat = deepcopy(self.operator_arity)
        self.operator_arity_compat['**'] = 2
        self.operators = list(self.operator_arity.keys())

        self.max_power = max([int(op[3:]) for op in self.operator_tokens if re.match(r'pow\d+(?!\_)', op)] + [0])
        self.max_fractional_power = max([int(op[5:]) for op in self.operator_tokens if re.match(r'pow1_\d+', op)] + [0])

        self.modules = get_used_modules(''.join(f"{op}(" for op in self.operator_realizations.values()))
        self.import_modules()

        self.connection_classes = {'add': (['+', '-'], "0"), 'mult': (['*', '/'], "1")}
        self.operator_to_class = {'+': 'add', '-': 'add', '*': 'mult', '/': 'mult'}
        self.connection_classes_inverse = {'add': "neg", 'mult': "inv"}
        self.connection_classes_hyper = {'add': "mult", 'mult': "pow"}
        self.binary_connectable_operators = {'+', '-', '*', '/'}

        # This is the simplified rules handling logic.
        # It no longer checks if `rules` is a string or performs any file I/O.
        # It only accepts a list of rules or None.
        dummy_variables = [f'x{i}' for i in range(100)]
        if rules is None:
            self.simplification_rules = []
        else:
            self.simplification_rules = deduplicate_rules(rules, dummy_variables=dummy_variables)

        # This part is also unchanged.
        self.compile_rules()
        self.rule_application_statistics: defaultdict[tuple, int] = defaultdict(int)

    def compile_rules(self) -> None:
        # ... (This method is unchanged)
        simplification_rules_patterns = []
        simplification_rules_no_patterns = []
        for r in self.simplification_rules:
            if any(re.match(r'_\d+', t) for t in r[0]):
                simplification_rules_patterns.append(r)
            else:
                simplification_rules_no_patterns.append(r)
        self.max_pattern_length = 0
        self.simplification_rules_patterns: dict[tuple, list[tuple[list, list]]] = self.construct_rule_patterns(simplification_rules_patterns)
        self.simplification_rules_no_patterns: dict[tuple, tuple] = {tuple(r[0]): tuple(r[1]) for r in simplification_rules_no_patterns}

    def import_modules(self) -> None:
        for module in self.modules:
            if module not in globals():
                globals()[module] = importlib.import_module(module)

    @classmethod
    def from_config(cls, config_path: str) -> "SimpliPyEngine":
        config_path = os.path.abspath(config_path)
        config = load_config(config_path)
        rules = []
        rules_file = config.get('rules')
        if rules_file:
            if not os.path.isabs(rules_file):
                config_dir = os.path.dirname(config_path)
                rules_path = os.path.join(config_dir, rules_file)
            else:
                rules_path = rules_file
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules = json.load(f)
            else:
                warnings.warn(f"Rules file '{rules_path}' specified in config not found.", UserWarning)
        return cls(operators=config['operators'], rules=rules)

    def is_valid(self, prefix_expression: list[str], verbose: bool = False) -> bool:
        '''
        Check if a prefix expression is valid.

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression.
        verbose : bool, optional
            Whether to print error messages, by default False.

        Returns
        -------
        bool
            Whether the expression is valid.
        '''
        stack: list[str] = []

        if len(prefix_expression) > 1 and prefix_expression[0] not in self.operator_arity:
            if verbose:
                print(f'Invalid expression {prefix_expression}: Variable must be leaf node')
            return False

        for token in reversed(prefix_expression):
            # Check if token is not a constant and numeric
            if token != '<constant>' and is_numeric_string(token):
                try:
                    float(token)
                except ValueError:
                    if verbose:
                        print(f'Invalid token {token} in expression {prefix_expression}')
                    return False

            if token in self.operator_arity:
                if len(stack) < self.operator_arity[token]:
                    if verbose:
                        print(f'Not enough operands for operator {token} in expression {prefix_expression}')
                    return False

                # Consume the operands based on the arity of the operator
                for _ in range(self.operator_arity[token]):
                    stack.pop()

            # Add the token to the stack
            stack.append(token)

        if len(stack) != 1:
            if verbose:
                print(f'Stack is not empty after parsing the expression {prefix_expression}')
            return False

        return True

    def prefix_to_infix(self, tokens: list[str], power: Literal['func', '**'] = 'func', realization: bool = False) -> str:
        '''
        Convert a list of tokens in prefix notation to infix notation

        Parameters
        ----------
        tokens : list[str]
            List of tokens in prefix notation
        power : Literal['func', '**'], optional
            Whether to use the 'func' or '**' notation for power operators, by default 'func'
        realization : bool, optional
            Whether to use the realization (python code) of the operators, by default False

        Returns
        -------
        str
            The infix notation of the expression
        '''
        # FIXME: Avoid unnecessary patentheses but keep necessary ones
        stack: list[str] = []

        for token in reversed(tokens):
            operator = self.realization_to_operator.get(token, token)
            operator_realization = self.operator_realizations.get(operator, operator)
            if operator in self.operator_tokens or operator in self.operator_aliases or operator in self.operator_arity_compat:
                write_operator = operator_realization if realization else operator
                write_operands = [stack.pop() for _ in range(self.operator_arity_compat[operator])]

                # If the operator is a power operator, format it as
                # "pow(operand1, operand2)" if power is 'func'
                # "operand1**operand2" if power is '**'
                # This regex must not match pow1_2 or pow1_3
                if re.match(r'pow\d+(?!\_)', operator) and power == '**':
                    exponent = int(operator[3:])
                    stack.append(f'(({write_operands[0]})**{exponent})')

                # If the operator is a fractional power operator such as pow1_2, format it as
                # "pow(operand1, 0.5)" if power is 'func'
                # "operand1**0.5" if power is '**'
                elif re.match(r'pow1_\d+', operator) and power == '**':
                    exponent = int(operator[5:])
                    stack.append(f'({write_operands[0]}**(1/{exponent}))')

                # If the operator is a function from a module, format it as
                # "module.function(operand1, operand2, ...)"
                elif '.' in write_operator or self.operator_arity_compat[operator] > 2:
                    # No need for parentheses here
                    stack.append(f'{write_operator}({", ".join([operand for operand in write_operands])})')

                # ** stays **
                elif self.operator_aliases.get(operator, operator) == '**':
                    stack.append(f'({write_operands[0]} {write_operator} {write_operands[1]})')

                # If the operator is a binary operator, format it as
                # "(operand1 operator operand2)"
                elif self.operator_arity_compat[operator] == 2:
                    stack.append(f'({write_operands[0]} {write_operator} {write_operands[1]})')

                elif operator == 'neg':
                    stack.append(f'-({write_operands[0]})')

                elif operator == 'inv':
                    stack.append(f'(1/{write_operands[0]})')

                else:
                    stack.append(f'{write_operator}({", ".join([operand for operand in write_operands])})')

            else:
                stack.append(token)

        infix_expression = stack.pop()

        return infix_expression  # FIXME: Sometimes result in "1 + x) / (2 * x" instead of "(1 + x) / (2 * x)"

    def infix_to_prefix(self, infix_expression: str) -> list[str]:
        '''
        Convert an infix expression to a prefix expression

        Parameters
        ----------
        infix_expression : str
            The infix expression

        Returns
        -------
        list[str]
            The prefix expression
        '''
        # Regex to tokenize expression properly (handles floating-point numbers)
        token_pattern = re.compile(r'<constant>|\d+\.\d+|\d+|[A-Za-z_][\w.]*|\*\*|[-+*/()]')

        # Tokenize the infix expression
        tokens = token_pattern.findall(infix_expression.replace(' ', ''))

        stack: list[str] = []
        prefix_expr: list[str] = []

        # Reverse the tokens for right-to-left parsing
        tokens = tokens[::-1]

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Handle numbers (integers or floats)
            if re.match(r'\d+\.\d+|\d+', token):  # Match positive or negative floats and integers
                prefix_expr.append(token)
            elif re.match(r'[A-Za-z_][\w.]*', token) or token == '<constant>':  # Match functions and variables
                prefix_expr.append(token)
            elif token == ')':
                stack.append(token)
            elif token == '(':
                while stack and stack[-1] != ')':
                    prefix_expr.append(stack.pop())
                if stack and stack[-1] == ')':
                    stack.pop()  # Pop the ')'
            else:
                # Handle binary and unary operators
                if token == '-' and (i == len(tokens) - 1 or tokens[i + 1] == '(' or (tokens[i + 1]) in self.operator_precedence_compat):
                    # Handle unary negation (not part of a number)
                    token = 'neg'

                if stack and stack[-1] != ')' and token != ')':
                    while stack and self.operator_precedence_compat.get(stack[-1], 0) >= self.operator_precedence_compat.get(token, 0):
                        prefix_expr.append(stack.pop())
                    stack.append(token)
                else:

                    if (token == 'neg' and not stack) or (stack and stack[-1] != ')'):
                        stack.insert(-1, token)
                    else:
                        stack.append(token)

            i += 1

        while stack:
            prefix_expr.append(stack.pop())

        return prefix_expr[::-1]

    def convert_expression(self, prefix_expr: list[str]) -> list[str]:
        '''
        Convert an expression to a supported form

        Parameters
        ----------
        prefix_expr : list[str]
            The prefix expression

        Returns
        -------
        list[str]
            The converted expression
        '''
        stack: list = []
        i = len(prefix_expr) - 1

        while i >= 0:
            token = prefix_expr[i]

            if token in self.operator_arity_compat or token in self.operator_aliases or re.match(r'pow\d+(?!\_)', token) or re.match(r'pow1_\d+', token):
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]

                if operator == 'neg':
                    # If the operand of neg is a number, combine them
                    if isinstance(stack[-1][0], str):
                        if is_numeric_string(stack[-1][0]):
                            stack[-1][0] = f'-{stack[-1][0]}'
                        elif is_numeric_string(stack[-1][0]):
                            stack[-1][0] = stack[-1][0][1:]
                        else:
                            # General case: assemble operator and its operands
                            operands = [stack.pop() for _ in range(arity)]
                            stack.append([operator, operands])
                    else:
                        # General case: assemble operator and its operands+
                        operands = [stack.pop() for _ in range(arity)]
                        stack.append([operator, operands])

                elif operator == '**':
                    # Check for floating-point exponent
                    base = stack.pop()
                    exponent = stack.pop()

                    if len(exponent) == 1:
                        if re.match(r'-?\d+$', exponent[0]):  # Integer exponent
                            exponent_value: int | float = int(exponent[0])
                            pow_operator = f'pow{abs(exponent_value)}'
                            if exponent_value < 0:
                                stack.append(['inv', [[pow_operator, [base]]]])
                            else:
                                stack.append([pow_operator, [base]])
                        elif is_numeric_string(exponent[0]):  # Floating-point exponent
                            exponent_value = float(exponent[0])

                            # Try to convert the exponent into a fraction
                            abs_exponent_fraction = fractions.Fraction(abs(float(exponent[0]))).limit_denominator()
                            if abs_exponent_fraction.numerator <= 5 and abs_exponent_fraction.denominator <= 5:
                                # Format the fraction as a combination of power operators, i.e. "x**(2/3)" -> "pow1_3(pow2(x))"
                                new_expression = [base]
                                if abs_exponent_fraction.numerator != 1:
                                    new_expression = [f'pow{abs_exponent_fraction.numerator}', new_expression]
                                if abs_exponent_fraction.denominator != 1:
                                    new_expression = [f'pow1_{abs_exponent_fraction.denominator}', new_expression]
                                if exponent_value < 0:
                                    new_expression = ['inv', new_expression]
                                stack.append(new_expression)
                            else:
                                stack.append(['pow', [base, exponent]])
                        else:
                            stack.append(['pow', [base, exponent]])

                    elif len(exponent) == 2 and exponent[0][0] == '/' and is_numeric_string(exponent[1][0][0]) and is_numeric_string(exponent[1][1][0]):
                        # Handle fractional exponent, e.g. "x**(2/3)"
                        if re.match(r'-?\d+$', exponent[1][0][0]) and re.match(r'-?\d+$', exponent[1][1][0]):
                            # Integer fraction exponent
                            numerator = int(exponent[1][0][0])
                            denominator = int(exponent[1][1][0])
                            numerator_power = f'pow{abs(numerator)}'
                            denominator_power = f'pow1_{abs(denominator)}'
                            if numerator * denominator < 0:
                                stack.append(['inv', [[denominator_power, [[numerator_power, [base]]]]]])
                            else:
                                stack.append([denominator_power, [[numerator_power, [base]]]])
                        else:
                            exponent_value = int(exponent[1][0][0]) / int(exponent[1][1][0])
                            abs_exponent_fraction = fractions.Fraction(abs(exponent_value)).limit_denominator()
                            if abs_exponent_fraction.numerator <= 5 and abs_exponent_fraction.denominator <= 5:
                                # Format the fraction as a combination of power operators, i.e. "x**(2/3)" -> "pow1_3(pow2(x))"
                                new_expression = [base]
                                if abs_exponent_fraction.numerator != 1:
                                    new_expression = [f'pow{abs_exponent_fraction.numerator}', new_expression]
                                if abs_exponent_fraction.denominator != 1:
                                    new_expression = [f'pow1_{abs_exponent_fraction.denominator}', new_expression]
                                if exponent_value < 0:
                                    new_expression = ['inv', new_expression]
                                stack.append(new_expression)
                            else:
                                stack.append(['pow', [base, exponent]])
                    else:
                        stack.append(['pow', [base, exponent]])

                else:
                    # General case: assemble operator and its operands
                    operands = [stack.pop() for _ in range(arity)]
                    stack.append([operator, operands])
            else:
                # Non-operator token (operand)
                stack.append([token])

            i -= 1

        need_to_convert_powers_expression = flatten_nested_list(stack)[::-1]

        stack = []
        i = len(need_to_convert_powers_expression) - 1

        while i >= 0:
            token = need_to_convert_powers_expression[i]

            if re.match(r'pow\d+(?!\_)', token) or re.match(r'pow1_\d+', token):
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat.get(operator, 1)
                operands = list(reversed(stack[-arity:]))

                # Identify chains of pow<i> xor pow1_<i> operators
                # Mixed chains are ignored
                operator_chain = [operator]
                current_operand = operands[0]

                operator_bases = ['pow1_', 'pow']
                operator_patterns = [r'pow1_\d+', r'pow\d+']
                operator_patterns_grouped = [r'pow1_(\d+)', r'pow(\d+)']
                max_powers = [self.max_fractional_power, self.max_power]
                for base, pattern, pattern_grouped, p in zip(operator_bases, operator_patterns, operator_patterns_grouped, max_powers):
                    if re.match(pattern, operator):
                        operator_base = base
                        operator_pattern = pattern
                        operator_pattern_grouped = pattern_grouped
                        max_power = p
                        break

                while len(current_operand) == 2 and re.match(operator_pattern, current_operand[0]):
                    operator_chain.append(current_operand[0])
                    current_operand = current_operand[1]

                if len(operator_chain) > 0:
                    p = prod(int(re.match(operator_pattern_grouped, op).group(1)) for op in operator_chain)  # type: ignore

                    # Factorize p into at most self.max_power or self.max_fractional_power
                    p_factors = factorize_to_at_most(p, max_power)

                    # Construct the new operators
                    new_operators = []
                    for p in p_factors:
                        new_operators.append(f'{operator_base}{p}')

                    if len(new_operators) == 0:
                        new_chain = current_operand
                    else:
                        new_chain = [new_operators[-1], [current_operand]]
                        for op in new_operators[-2::-1]:
                            new_chain = [op, [new_chain]]

                    _ = [stack.pop() for _ in range(arity)]
                    stack.append(new_chain)
                    i -= 1
                    continue

            elif token in self.operator_arity_compat or token in self.operator_aliases:
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))

                _ = [stack.pop() for _ in range(arity)]
                stack.append([operator, operands])
                i -= 1
                continue

            else:
                stack.append([token])
                i -= 1

        return flatten_nested_list(stack)[::-1]

    # PARSING
    def parse(
            self,
            infix_expression: str,
            convert_expression: bool = True,
            mask_numbers: bool = False) -> list[str]:
        '''
        Parse an infix expression into a prefix expression

        Parameters
        ----------
        infix_expression : str
            The infix expression
        substitute_special_constants : bool, optional
            Whether to substitute special constants, by default True
        convert_expression : bool, optional
            Whether to convert the expression, by default True
        convert_variable_names : bool, optional
            Whether to convert variable names, by default True
        mask_numbers : bool, optional
            Whether to mask numbers, by default False
        too_many_variables : Literal['ignore', 'raise'], optional
            Whether to ignore or raise an error if there are too many variables, by default 'ignore'

        Returns
        -------
        list[str]
            The prefix expression
        '''

        parsed_expression = self.infix_to_prefix(infix_expression)

        if convert_expression:
            parsed_expression = self.convert_expression(parsed_expression)
        if mask_numbers:
            parsed_expression = numbers_to_constant(parsed_expression, inplace=True)

        return remove_pow1(parsed_expression)  # HACK: Find a better place to put this

    def prefix_to_tree(self, expression: list[str]) -> list:
        def build_tree(index: int) -> tuple[list | None, int]:
            if index >= len(expression):
                return None, index

            token = expression[index]

            # If token is not an operator or is an operator with arity 0
            if isinstance(token, dict) or token not in self.operator_arity or self.operator_arity[token] == 0:
                return [token], index + 1

            # If token is an operator
            operands = []
            current_index = index + 1

            # Process operands based on the operator's arity
            for _ in range(self.operator_arity[token]):
                if current_index >= len(expression):
                    break

                subtree, current_index = build_tree(current_index)
                if subtree:
                    operands.append(subtree)

            return [token, operands], current_index

        result, _ = build_tree(0)

        if result is None:
            raise ValueError(f'Failed to build tree from expression {expression}')

        return result

    def construct_rule_patterns(self, rules_list: list[tuple[tuple[str, ...], tuple[str, ...]]], verbose: bool = False) -> dict[tuple, list[tuple[list, list]]]:
        # Group the rules by arity
        rules_list_of_operator: defaultdict[str, list] = defaultdict(list)
        for rule in rules_list:
            rules_list_of_operator[rule[0][0]].append(rule)
        rules_list_of_operator = dict(rules_list_of_operator)  # type: ignore

        # Sort the rules by length of the left-hand side to make matching more efficient
        for operator, rules_list_of_operator_list in rules_list_of_operator.items():
            rules_list_of_operator[operator] = sorted(rules_list_of_operator_list, key=lambda x: len(x[0]))

        # Construct the trees for pattern matching
        rules_trees = {operator: [
            (
                self.prefix_to_tree(list(rule[0])),
                self.prefix_to_tree(list(rule[1]))
            )
            for rule in rules_list_of_operator_a] for operator, rules_list_of_operator_a in tqdm(rules_list_of_operator.items(), desc='Constructing patterns', disable=not verbose)}

        rules_trees_organized: defaultdict[tuple, list] = defaultdict(list)
        for operator, rules in rules_trees.items():
            for (pattern, replacement) in rules:
                pattern_length = len(flatten_nested_list(pattern))
                rules_trees_organized[(pattern_length, operator,)].append((pattern, replacement))

                if pattern_length > self.max_pattern_length:
                    self.max_pattern_length = pattern_length

        return rules_trees_organized

    def parse_subtree(self, tokens: list[str] | tuple[str, ...], start_idx: int) -> tuple[list, int]:
        """Parse a subtree from tokens starting at start_idx, return (subtree, next_idx)"""
        if start_idx >= len(tokens):
            raise ValueError(f"Start index {start_idx} is out of bounds for tokens {tokens}")

        token = tokens[start_idx]

        if token in self.operator_arity_compat or token in self.operator_aliases:
            operator = self.operator_aliases.get(token, token)
            arity = self.operator_arity_compat[operator]
            operands = []
            idx = start_idx + 1

            for _ in range(arity):
                operand, idx = self.parse_subtree(tokens, idx)
                operands.append(operand)

            return [operator, operands], idx
        else:
            # It's a terminal (constant or variable)
            return [token], start_idx + 1

    def apply_rules_top_down(self, subtree: list, max_pattern_length: int | None = None, collect_rule_statistics: bool = False, verbose: bool = False) -> list:
        """Apply simplification rules to a subtree in a top-down manner"""
        if len(subtree) == 1:
            # Terminal node, no rules to apply
            return subtree

        operator = subtree[0]
        operands = subtree[1]

        # First, check if all operands are constants
        if all(len(operand) == 1 and operand[0] == '<constant>' for operand in operands):
            return ['<constant>']

        # Convert subtree to flat form for rule matching
        flat_subtree = tuple(flatten_nested_list(subtree)[::-1])
        subtree_length = len(flat_subtree)

        if verbose:
            print(f'Checking if explicit rule applies to subtree: {flat_subtree} with length {subtree_length}')

        # Check explicit rules first
        replacement = self.simplification_rules_no_patterns.get(flat_subtree, None)
        if verbose:
            print(f'Explicit rule found: {flat_subtree} -> {replacement}' if replacement else 'No explicit rule found')
        if replacement is not None:
            if collect_rule_statistics:
                self.rule_application_statistics[(flat_subtree, replacement)] += 1
            if verbose:
                print(f'Applied explicit rule\t{flat_subtree} ->\n\t\t{replacement}\nto subtree\t{subtree}\n')
            # Parse and recursively simplify the replacement
            parsed_replacement, _ = self.parse_subtree(list(replacement), 0)
            return self.apply_rules_top_down(parsed_replacement)

        # Check pattern rules, starting with the largest patterns
        if max_pattern_length is None:
            subtree_max_pattern_length = min(subtree_length, self.max_pattern_length)
        else:
            subtree_max_pattern_length = min(max_pattern_length, subtree_length, self.max_pattern_length)

        for pattern_length in reversed(range(1, subtree_max_pattern_length + 1)):
            if verbose:
                print(f'Checking pattern rules for operator {operator} with subtree length {pattern_length}')
            for rule in self.simplification_rules_patterns.get((pattern_length, operator,), []):
                does_match, mapping = match_pattern(subtree, rule[0], mapping=None)
                if does_match:
                    # Apply the mapping to get the replacement
                    replacement_tree = apply_mapping(deepcopy(rule[1]), mapping)
                    if collect_rule_statistics:
                        self.rule_application_statistics[(
                            tuple(flatten_nested_list(rule[0])[::-1]),
                            tuple(flatten_nested_list(rule[1])[::-1]))] += 1
                    if verbose:
                        print(f'Applied pattern rule\t{rule[0]} ->\n\t\t{rule[1]}\nto subtree\t{subtree}\nwith mapping\t{mapping}\n')
                    # Recursively simplify the replacement
                    return self.apply_rules_top_down(replacement_tree, max_pattern_length)

        # No rule applied at this level, recursively simplify operands
        simplified_operands = [self.apply_rules_top_down(operand, max_pattern_length) for operand in operands]
        simplified_subtree = [operator, simplified_operands]

        # After simplifying operands, check again if a rule now applies
        # (This handles cases where simplification of operands enables a rule)
        flat_simplified = tuple(flatten_nested_list(simplified_subtree)[::-1])

        # Check explicit rules again
        replacement = self.simplification_rules_no_patterns.get(flat_simplified, None)
        if replacement is not None:
            if collect_rule_statistics:
                self.rule_application_statistics[(flat_simplified, replacement)] += 1
            if verbose:
                print(f'Applied explicit rule (after operand simplification)\t{flat_simplified} ->\n\t\t{replacement}\nto subtree\t{simplified_subtree}\n')
            parsed_replacement, _ = self.parse_subtree(list(replacement), 0)
            return self.apply_rules_top_down(parsed_replacement, max_pattern_length)

        # Check pattern rules again
        for pattern_length in reversed(range(1, subtree_max_pattern_length + 1)):
            for rule in self.simplification_rules_patterns.get((pattern_length, operator,), []):
                does_match, mapping = match_pattern(simplified_subtree, rule[0], mapping=None)
                if does_match:
                    replacement_tree = apply_mapping(deepcopy(rule[1]), mapping)
                    if collect_rule_statistics:
                        self.rule_application_statistics[(
                            tuple(flatten_nested_list(rule[0])[::-1]),
                            tuple(flatten_nested_list(rule[1])[::-1]))] += 1
                    if verbose:
                        print(f'Applied pattern rule (after operand simplification)\t{rule[0]} ->\n\t\t{rule[1]}\nto subtree\t{simplified_subtree}\nwith mapping\t{mapping}\n')
                    return self.apply_rules_top_down(replacement_tree, max_pattern_length)

        return simplified_subtree

    def apply_simplifcation_rules(self, expression: list[str] | tuple[str, ...], max_pattern_length: int | None = None, collect_rule_statistics: bool = False, verbose: bool = False) -> list[str]:
        if all(t == '<constant>' or t in self.operator_arity for t in expression):
            return ['<constant>']

        # Parse the entire expression into a tree
        tree, _ = self.parse_subtree(expression, 0)
        if tree is None:
            return list(expression)

        # Apply rules top-down
        simplified_tree = self.apply_rules_top_down(tree, max_pattern_length, collect_rule_statistics, verbose)

        # Flatten back to prefix notation
        return flatten_nested_list(simplified_tree)[::-1]

    def collect_multiplicities(self, expression: list[str] | tuple[str, ...], verbose: bool = False) -> tuple[list, list, list]:
        stack: list = []
        stack_annotations: list = []
        stack_labels: list = []

        i = len(expression) - 1

        # Traverse the expression from right to left
        while i >= 0:
            token = expression[i]

            if token in self.binary_connectable_operators:
                operator = token
                arity = 2
                operands = list(reversed(stack[-arity:]))
                operands_annotations_dicts = list(reversed(stack_annotations[-arity:]))
                operands_labels = list(reversed(stack_labels[-arity:]))

                operator_annotation_dict: dict[str, dict[tuple[str, ...], list[int]]] = {cc: {} for cc in self.connection_classes}

                cc = self.operator_to_class[operator]

                # Carry over annotations from operand nodes
                if verbose:
                    print(f'---- {token} ----')

                for branch, operand_annotations_dict in enumerate(operands_annotations_dicts):  # One dict for left and right branch
                    if verbose:
                        print(branch)
                        pprint.pprint(operand_annotations_dict)
                    for subtree_hash in operand_annotations_dict[0][cc]:  # All subtrees appearing in either branch (0 gets root node of the branch)
                        # Add to operator dict if not already present
                        if subtree_hash not in operator_annotation_dict[cc]:
                            if verbose:
                                print(f'Initializing {subtree_hash} for {cc}')
                            operator_annotation_dict[cc][subtree_hash] = [0, 0]

                        if operator in {'-', '/'} and branch == 1:
                            for p in range(2):
                                if verbose:
                                    print(f'Adding {operand_annotations_dict[0][cc][subtree_hash][p]} to {operator_annotation_dict[cc][subtree_hash][1 - p]} at {1 - p} of {subtree_hash} (reversed)')
                                operator_annotation_dict[cc][subtree_hash][1 - p] += operand_annotations_dict[0][cc][subtree_hash][p]
                        else:
                            for p in range(2):
                                if verbose:
                                    print(f'Adding {operand_annotations_dict[0][cc][subtree_hash][p]} to {operator_annotation_dict[cc][subtree_hash][p]} at {p} of {subtree_hash}')
                                operator_annotation_dict[cc][subtree_hash][p] += operand_annotations_dict[0][cc][subtree_hash][p]

                if verbose:
                    print(f'/---- {token} ----')
                    print()

                # Label each subtree with its own hash to know which to cancel later
                _ = [stack.pop() for _ in range(arity)]
                _ = [stack_annotations.pop() for _ in range(arity)]
                _ = [stack_labels.pop() for _ in range(arity)]
                stack.append([operator, operands])
                stack_annotations.append([operator_annotation_dict, operands_annotations_dicts])
                new_label = tuple(flatten_nested_list([operator, operands])[::-1])
                stack_labels.append([new_label, operands_labels])
                i -= 1
                continue

            if token in self.operator_arity:
                operator = token
                arity = self.operator_arity[token]
                operands = list(reversed(stack[-arity:]))
                operands_annotations_dicts = list(reversed(stack_annotations[-arity:]))
                operands_labels = list(reversed(stack_labels[-arity:]))

                # Label each subtree with its own hash to know which to cancel later
                _ = [stack.pop() for _ in range(arity)]
                _ = [stack_annotations.pop() for _ in range(arity)]
                _ = [stack_labels.pop() for _ in range(arity)]
                stack.append([operator, operands])
                stack_annotations.append([{cc: {} for cc in self.connection_classes}, operands_annotations_dicts])
                new_label = tuple(flatten_nested_list([operator, operands])[::-1])
                stack_labels.append([new_label, operands_labels])
                i -= 1
                continue

            stack.append([token])
            stack_annotations.append([{cc: {tuple([token]): [1, 0]} for cc in self.connection_classes}])
            stack_labels.append([tuple([token])])
            i -= 1

        if verbose:
            pprint.pprint(stack_annotations)
            print()

        return stack, stack_annotations, stack_labels

    def cancel_terms(self, expression_tree: list, expression_annotations_tree: list, stack_labels: list, verbose: bool = False) -> list[str]:
        stack = expression_tree
        stack_annotations = expression_annotations_tree
        stack_parity = [{cc: 1 for cc in self.connection_classes} for _ in range(len(stack_labels))]
        stack_still_connected = [False]

        expression: list[str] = []

        cancellation_candidate = None
        n_replaced = 0
        still_connected = False

        while len(stack) > 0:
            subtree = stack.pop()
            subtree_annotation = stack_annotations.pop()
            subtree_labels = stack_labels.pop()
            subtree_parities = stack_parity.pop()
            still_connected = stack_still_connected.pop()

            if cancellation_candidate is not None:
                argmax_class, cancelled_subtree, cancelled_multiplicity_sum = cancellation_candidate
                still_connected = still_connected and (subtree[0] in self.connection_classes[argmax_class][0] or subtree[0] not in self.operator_arity)

                if still_connected:
                    if cancelled_subtree == subtree_labels[0]:
                        neutral_element = self.connection_classes[argmax_class][1]

                        if cancelled_subtree == ('<constant>',):
                            first_replacement = ('<constant>',)
                            other_replacements: str | tuple[str, ...] = neutral_element
                        else:
                            current_parity = subtree_parities[argmax_class]
                            inverse_operator = self.connection_classes_inverse[argmax_class]

                            if verbose:
                                print()
                                print(f'Processing subtree {subtree_labels[0]} with current parity {current_parity} and total multiplicity sum {cancelled_multiplicity_sum}')

                            # FIXME
                            if current_parity * cancelled_multiplicity_sum >= 0:  # Negative parity and negative multiplicity cancel out
                                inverse_operator_prefix: tuple[str, ...] = ()
                                double_inverse_operator_prefix: tuple[str, ...] = (inverse_operator,)
                            else:
                                inverse_operator_prefix = (inverse_operator,)
                                double_inverse_operator_prefix = ()

                            if verbose:
                                print(f'Inverse operator prefix: {inverse_operator_prefix}, double inverse operator prefix: {double_inverse_operator_prefix}')

                            if cancelled_multiplicity_sum == 0:
                                # Term is cancelled entirely. Replace all occurences with the neutral element
                                first_replacement = (neutral_element,)
                                other_replacements = neutral_element
                                if verbose:
                                    print(f'Cancelled term {cancelled_subtree} entirely: first replacement {first_replacement}, other replacements {other_replacements}')

                            if abs(cancelled_multiplicity_sum) == 1:
                                # Term occurs once. Replace every occurence after the first one with the neutral element
                                first_replacement = inverse_operator_prefix + cancelled_subtree
                                other_replacements = (neutral_element,)
                                if verbose:
                                    print(f'Cancelled term {cancelled_subtree} once: first replacement {first_replacement}, other replacements {other_replacements}')

                            if abs(cancelled_multiplicity_sum) > 1:
                                # Term occurs multiple times. Replace the first occurence with a multiplication or power of the term. Replace every occurence after the first one with the neutral element
                                hyper_operator = self.connection_classes_hyper[argmax_class]
                                operator = self.connection_classes[argmax_class][0][0]  # Positive multiplicity
                                if cancelled_multiplicity_sum > 5 and is_prime(abs(cancelled_multiplicity_sum)):
                                    powers = factorize_to_at_most(abs(cancelled_multiplicity_sum) - 1, self.max_power)
                                    first_replacement = inverse_operator_prefix + (operator,) + tuple(f'{hyper_operator}{p}' for p in powers) + cancelled_subtree + cancelled_subtree
                                else:
                                    powers = factorize_to_at_most(abs(cancelled_multiplicity_sum), self.max_power)
                                    first_replacement = inverse_operator_prefix + tuple(f'{hyper_operator}{p}' for p in powers) + cancelled_subtree

                                other_replacements = (neutral_element,)

                                if verbose:
                                    print(f'Cancelled term {cancelled_subtree} multiple times: first replacement {first_replacement}, other replacements {other_replacements}')

                                if verbose:
                                    print(f'Cancelled term {cancelled_subtree} multiple times inverted: first replacement {first_replacement}, other replacements {other_replacements}')

                        if n_replaced == 0:
                            expression.extend(first_replacement)
                            if verbose:
                                print(f'{n_replaced}: Added first replacement {first_replacement} to expression')
                        else:
                            expression.extend(other_replacements)
                            if verbose:
                                print(f'{n_replaced}: Added other replacements {other_replacements} to expression')
                        n_replaced += 1
                        continue

            # Leaf node
            if len(subtree) == 1:
                expression.append(subtree[0])
                continue

            # Non-leaf node
            operator, operands = subtree
            _, operands_annotations_sets = subtree_annotation
            _, operands_labels = subtree_labels
            operator_parity = subtree_parities  # No operand parity information yet

            # TODO: Propagate parities of unary inverse operators

            if verbose:
                print(f'Operator {operator} with operands {operands} is still connected: {still_connected}')
                print(f'Operator parities: {operator_parity}')

            if operator in self.binary_connectable_operators:
                propagated_operand_parities: list[dict[str, int]] = [{}, {}]
                if still_connected:
                    for cc, (operator_set, _) in self.connection_classes.items():
                        propagated_operand_parities[0][cc] = operator_parity[cc]
                        propagated_operand_parities[1][cc] = operator_parity[cc] * (-1 if operator == self.operator_inverses[operator_set[0]] else 1)
                    if verbose:
                        print(f'Propagated operand parities: {propagated_operand_parities}')
                else:
                    for cc, (operator_set, _) in self.connection_classes.items():
                        propagated_operand_parities[0][cc] = 1
                        propagated_operand_parities[1][cc] = (-1 if operator == self.operator_inverses[operator_set[0]] else 1)
                    if verbose:
                        print(f'Reset parities to {propagated_operand_parities}')

                # If no cancellation candidate has been identified yet, try to find one in the current subtree
                if cancellation_candidate is None:
                    for cc in self.connection_classes:
                        for subtree_hash, multiplicity in subtree_annotation[0][cc].items():
                            # Consider candidates where
                            # 1. there is something to cancel (i.e. the sum of the absolute multiplicities is greater than 1)
                            # 2. constants are allowed to be cancelled:
                            #   a. single constants <constant> can be cancelled
                            #   b. composite terms with constants cannot be cancelled with the current method (one <constant> needs to survive)
                            if sum(abs(m) for m in multiplicity) > 1 and ('<constant>' not in subtree_hash or len(subtree_hash) == 1):  # Cannot cancel terms with arbitrary constants
                                cancellation_candidate = (cc, subtree_hash, multiplicity[0] - multiplicity[1])
                                still_connected = True

                # Add the operator to the expression
                expression.append(operator)

                # Add the children to the stack
                for operand, operand_an, operand_label, propagated_operand_parity in zip(
                        reversed(operands),
                        reversed(operands_annotations_sets),
                        reversed(operands_labels),
                        reversed(propagated_operand_parities)):
                    stack.append(operand)
                    stack_annotations.append(operand_an)
                    stack_labels.append(operand_label)
                    stack_parity.append(propagated_operand_parity)
                    stack_still_connected.append(still_connected)

            else:
                # Add the operator to the expression
                expression.append(operator)

                # Add the children to the stack
                for operand, operand_an, operand_label in zip(reversed(operands), reversed(operands_annotations_sets), reversed(operands_labels)):
                    stack.append(operand)
                    stack_annotations.append(operand_an)
                    stack_labels.append(operand_label)
                    stack_parity.append({cc: 1 for cc in self.connection_classes})
                    stack_still_connected.append(still_connected)

        return expression

    def sort_operands(self, expression: list[str] | tuple[str, ...]) -> list[str]:
        stack: list = []
        i = len(expression) - 1

        while i >= 0:
            token = expression[i]

            if token in self.operator_arity_compat or token in self.operator_aliases:
                operator = self.operator_aliases.get(token, token)
                arity = self.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))

                if operator in self.commutative_operators:
                    # Check for the pattern [*, *, A, B, C] -> [*, A, *, B, C] or [+, +, A, B, C] -> [+, A, +, B, C]
                    if len(operands[0]) == 2 and operator == operands[0][0]:
                        _ = [stack.pop() for _ in range(arity)]
                        stack.append([operator, [operands[0][1][0], [operator, [operands[0][1][1], operands[1]]]]])
                        i -= 1
                        continue

                    subtree = [operator, operands]

                    # Traverse through the tree in breadth-first order
                    queue = [subtree]
                    commutative_paths: list[tuple] = [tuple()]
                    commutative_positions = []
                    while queue:
                        node = queue.pop(0)
                        current_path = commutative_paths.pop(0)
                        for child_index, child in enumerate(node[1]):  # I conclude that using `i` as a variable name here is not very clever
                            if len(child) > 1:
                                if child[0] == node[0]:
                                    # Continue: Same commutative perator
                                    queue.append(child)
                                    commutative_paths.append(current_path + (child_index,))
                                else:
                                    # Stop: Different operator
                                    commutative_positions.append(current_path + (child_index,))
                            else:
                                # Stop: Leaf
                                commutative_positions.append(current_path + (child_index,))

                    # Sort the positions
                    sorted_indices = sorted(range(len(commutative_positions)), key=lambda x: commutative_positions[x])

                    commutative_paths = [commutative_positions[i] for i in sorted_indices]
                    commutative_positions = [commutative_positions[i] for i in sorted_indices]

                    operands_to_sort = []
                    for position in commutative_positions:
                        node = subtree
                        for position_index in position:
                            node = node[1][position_index]
                        operands_to_sort.append(node)

                    sorted_operands = sorted(operands_to_sort, key=self.operand_key)

                    # Replace the operands in the tree
                    new_subtree: list = deepcopy(subtree)

                    for position, operand in zip(commutative_positions, sorted_operands):
                        node = new_subtree
                        for position_index in position:
                            node = node[1][position_index]
                        node[:] = operand

                    operands = new_subtree[1]

                    _ = [stack.pop() for _ in range(arity)]
                    stack.append([operator, operands])
                    i -= 1
                    continue

                _ = [stack.pop() for _ in range(arity)]
                stack.append([operator, operands])

            else:
                stack.append([token])

            i -= 1

        return flatten_nested_list(stack)[::-1]

    def simplify(
            self,
            expression: str | list[str] | tuple[str, ...],
            max_iter: int = 5,
            max_pattern_length: int | None = None,
            mask_elementary_literals: bool = True,
            apply_simplification_rules: bool = True,
            inplace: bool = False,
            collect_rule_statistics: bool = False,
            verbose: bool = False) -> str | list[str] | tuple[str, ...]:
        if isinstance(expression, str):
            return_type = 'str'
            original_expression: str | list[str] | tuple[str, ...] = "" + expression  # Create a copy
            expression = self.parse(expression, convert_expression=True, mask_numbers=False)
        elif isinstance(expression, tuple):
            return_type = 'tuple'
            original_expression = expression  # No need to copy immutable tuple
            expression = list(expression)
        else:
            return_type = 'list'
            original_expression = expression.copy()

        new_expression = expression.copy()

        length_before = len(expression)

        if verbose:
            print(f'Initial expression: {new_expression}')

        # # Apply simplification rules and sort operands to get started
        # if apply_simplification_rules:
        #     new_expression = self.apply_simplifcation_rules(new_expression, max_pattern_length, collect_rule_statistics=collect_rule_statistics, verbose=verbose)

        # if verbose:
        #     print(f'_apply_simplifcation_rules: {new_expression}')

        for i in range(max_iter):
            # Cancel any terms
            expression_tree, annotated_expression_tree, stack_labels = self.collect_multiplicities(new_expression, verbose=verbose)
            new_expression = self.cancel_terms(expression_tree, annotated_expression_tree, stack_labels, verbose=verbose)

            if verbose:
                print(f'{i}: cancel_terms: {new_expression}')

            # Apply simplification rules
            if apply_simplification_rules:
                new_expression = self.apply_simplifcation_rules(new_expression, max_pattern_length, collect_rule_statistics=collect_rule_statistics, verbose=verbose)

            if verbose:
                print(f'{i}: _apply_simplifcation_rules: {new_expression}')

            if new_expression == expression:
                break
            expression = new_expression

        # Sort operands
        new_expression = self.sort_operands(new_expression)

        if verbose:
            print(f'{i}: sort_operands: {new_expression}')

        if mask_elementary_literals:
            new_expression = mask_elementary_literals_fn(new_expression, inplace=inplace)

            if verbose:
                print(f'{i}: mask_elementary_literals: {new_expression}')

        if len(new_expression) > length_before:
            # The expression has grown, which is not a simplification
            match return_type:
                case 'str':
                    return original_expression
                case 'tuple':
                    return tuple(original_expression)
                case 'list':
                    if inplace:
                        expression[:] = original_expression
                    else:
                        expression = original_expression
            return expression

        match return_type:
            case 'str':
                return self.prefix_to_infix(new_expression, realization=False, power='**')
            case 'tuple':
                return tuple(new_expression)
            case 'list':
                if inplace:
                    expression[:] = new_expression
                else:
                    expression = new_expression

        return new_expression

    def exist_constants_that_fit(self, expression: list[str] | tuple[str, ...], variables: list[str], X: np.ndarray, y_target: np.ndarray) -> bool:
        if isinstance(expression, tuple):
            expression = list(expression)

        executable_prefix_expression = self.operators_to_realizations(expression)
        prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression, convert_numbers_to_constant=False)
        code_string = self.prefix_to_infix(prefix_expression_with_constants, realization=True)
        code = codify(code_string, variables + constants)
        f = self.code_to_lambda(code)

        def pred_function(X: np.ndarray, *constants: np.ndarray | None) -> float | np.ndarray:
            if len(constants) == 0:
                y = safe_f(f, X)
            y = safe_f(f, X, constants)

            # If the numbers are complex, return nan
            if np.iscomplexobj(y):
                return np.full(X.shape[0], np.nan)

            return y

        p0 = np.random.normal(loc=0, scale=5, size=len(constants))

        is_valid = np.isfinite(X).all(axis=1) & np.isfinite(y_target)

        if not np.any(is_valid) or len(constants) > is_valid.sum():  # https://github.com/scipy/scipy/issues/13969
            return False

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                popt, _ = curve_fit(pred_function, X[is_valid], y_target[is_valid].flatten(), p0=p0)
        except RuntimeError:
            return False

        y = f(*X.T, *popt)
        if not isinstance(y, np.ndarray):
            y = np.full(X.shape[0], y)  # type: ignore

        return np.allclose(y_target, y, equal_nan=True)

    def find_rule_worker(
            self,
            worker_id: int,
            work_queue: Queue,
            result_queue: Queue,
            X_shape: tuple,
            X_dtype: np.dtype,
            X_shm_name: str,
            expressions_of_length_and_variables: dict,
            dummy_variables: list[str],
            operator_arity: dict,
            constants_fit_challenges: int,
            constants_fit_retries: int) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            # Reconstruct arrays from shared memory
            X_shm = SharedMemory(name=X_shm_name)
            X: np.ndarray = np.ndarray(X_shape, dtype=X_dtype, buffer=X_shm.buf)

            # Main work loop
            while True:
                work_item = work_queue.get()

                # Check for sentinel
                if work_item is None:
                    break

                expression, simplified_length, allowed_candidate_lengths = work_item

                if len(allowed_candidate_lengths) == 0 or max(allowed_candidate_lengths) <= 0 or simplified_length <= min(allowed_candidate_lengths):  # Request unrealistic simplification or already have better simplification than requested
                    # No candidates allowed, skip this expression
                    result_queue.put(None)
                    continue

                # Check if purely numerical
                if all([t == '<constant>' or t in operator_arity for t in expression]) and len(expression) > 1:
                    result_queue.put((expression, ('<constant>',)))
                    continue

                expression_variables = list(set(expression) & set(dummy_variables))

                # Evaluate expression
                executable_prefix_expression = self.operators_to_realizations(expression)
                prefix_expression_with_constants, constants = num_to_constants(executable_prefix_expression, convert_numbers_to_constant=False)
                code_string = self.prefix_to_infix(prefix_expression_with_constants, realization=True)
                code = codify(code_string, dummy_variables + constants)

                f = self.code_to_lambda(code)

                # Suppress warnings in worker
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    found_simplifications = []

                    # Check against all smaller expressions
                    for candidate_length in allowed_candidate_lengths:
                        for candidate_variables, candidate_expressions in expressions_of_length_and_variables.get(candidate_length, {}).items():
                            if any(var not in expression_variables for var in candidate_variables):
                                # The candidate expression contains variables not in the original expression. It cannot be a simplification.
                                continue

                            for candidate_expression in candidate_expressions:
                                executable_candidate = self.operators_to_realizations(candidate_expression)
                                prefix_candidate_w_constants, candidate_constants = num_to_constants(executable_candidate, convert_numbers_to_constant=False)
                                candidate_code = self.prefix_to_infix(prefix_candidate_w_constants, realization=True)
                                candidate_compiled = codify(candidate_code, dummy_variables + candidate_constants)
                                f_candidate = self.code_to_lambda(candidate_compiled)

                                # Check if expressions are equivalent
                                if len(candidate_constants) == 0:
                                    y_candidate = safe_f(f_candidate, X)
                                    if not isinstance(y_candidate, np.ndarray):
                                        y_candidate = np.full(X.shape[0], y_candidate)

                                    # Resample constants to avoid false positives
                                    # The expression is considered a match unless one of the challenges fails
                                    expressions_match = True
                                    for challenge_id in range(constants_fit_challenges):
                                        random_constants = np.random.normal(loc=0, scale=5, size=len(constants))
                                        # Try all combinations of positive and negative constants
                                        for positive_negative_constant_combination in product((-1, 0, 1), repeat=len(constants)):
                                            y = safe_f(f, X, np.abs(random_constants) * positive_negative_constant_combination)  # abs may be redundant here
                                            if not np.allclose(y, y_candidate, equal_nan=True):
                                                expressions_match = False
                                                break

                                        if not expressions_match:
                                            # A combination produced a different result, abort this candidate
                                            break

                                else:
                                    # Resample constants to avoid false positives
                                    # The expression is considered a match unless one of the challenges fails
                                    expressions_match = True
                                    for challenge_id in range(constants_fit_challenges):
                                        # Need to check if constants can be fitted
                                        random_constants = np.random.normal(loc=0, scale=5, size=len(constants))
                                        # Try all combinations of positive and negative constants
                                        for positive_negative_constant_combination in product((-1, 0, 1), repeat=len(constants)):
                                            y = safe_f(f, X, np.abs(random_constants) * positive_negative_constant_combination)  # abs may be redundant here
                                            for _ in range(constants_fit_retries):
                                                if self.exist_constants_that_fit(candidate_expression, dummy_variables, X, y):
                                                    # Found a candidate that fits, next challenge please
                                                    break
                                            else:
                                                # No candidate found that fits, not all challenges could be solved, abort this candidate
                                                expressions_match = False
                                                break

                                        if not expressions_match:
                                            # A combination produced a different result, abort this candidate
                                            break

                                if expressions_match:
                                    found_simplifications.append(candidate_expression)
                                    # Still check for further candidates of the same length

                        if found_simplifications:
                            # Found at least one simplification for the current length
                            # Every further candidate will be longer, so we can stop checking
                            break

                if not found_simplifications:
                    # No simplification found
                    result_queue.put(None)
                else:
                    found_simplifications_without_num = [simplification for simplification in found_simplifications if '<constant>' not in simplification]
                    if found_simplifications_without_num:
                        # Prefer simplifications without <constant>
                        result_queue.put((expression, found_simplifications_without_num[0]))
                    else:
                        # No simplification without <constant> found, return the first found simplification
                        result_queue.put((expression, found_simplifications[0]))

        except Exception as e:
            # Log exceptions to result queue
            result_queue.put(('ERROR', e, (expression, simplified_length, allowed_candidate_lengths)))
        finally:
            X_shm.close()

    def find_rules(
            self,
            max_source_pattern_length: int = 7,
            max_target_pattern_length: int | None = None,
            dummy_variables: int | list[str] | None = None,
            extra_internal_terms: list[str] | None = None,
            X: np.ndarray | int | None = None,
            constants_fit_challenges: int = 5,
            constants_fit_retries: int = 5,
            output_file: str | None = None,
            save_every: int = 100,
            reset_rules: bool = True,
            verbose: bool = False,
            n_workers: int | None = None) -> None:

        # Signal handler for main process
        interrupted = False

        def signal_handler(signum: Any, frame: Any) -> None:
            nonlocal interrupted
            interrupted = True
            print("\nInterrupt received, cleaning up...")

        # Set up signal handler in main process
        old_handler = signal.signal(signal.SIGINT, signal_handler)

        # All the initialization from the sequential version
        extra_internal_terms = extra_internal_terms or []

        if dummy_variables is None:
            max_leaf_nodes_if_operators_binary = int(max_source_pattern_length - (max_source_pattern_length - 1) / 2)
            dummy_variables = [f"x{i}" for i in range(max_leaf_nodes_if_operators_binary)]
            if verbose:
                print(f"Using {len(dummy_variables)} dummy variables: {dummy_variables}")
        elif isinstance(dummy_variables, int):
            dummy_variables = [f"x{i}" for i in range(dummy_variables)]

        if reset_rules:
            self.simplification_rules = []
            self.compile_rules()

        if X is None:
            X_data = np.random.normal(loc=0, scale=5, size=(1024, len(dummy_variables)))
        elif isinstance(X, int):
            X_data = np.random.normal(loc=0, scale=5, size=(X, len(dummy_variables)))

        leaf_nodes = dummy_variables + extra_internal_terms
        non_leaf_nodes = dict(sorted(self.operator_arity.items(), key=lambda x: x[1]))

        # --- Phase 1: Generate expressions ---
        if verbose:
            print(f"Phase 1: Generating all expressions up to length {max_source_pattern_length}")

        expressions_of_length: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        new_expressions_of_length: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)

        # Initialize with leaf nodes
        for leaf in leaf_nodes:
            expressions_of_length[1].add((leaf,))

        # Generate expressions level by level
        new_sizes: set[int] = set()
        while max(expressions_of_length.keys()) < max_source_pattern_length:  # This means that every smaller size is already generated
            for expression in construct_expressions(expressions_of_length, non_leaf_nodes, must_have_sizes=new_sizes):
                new_expressions_of_length[len(expression)].add(expression)

            new_sizes = set()
            lengths_before = {k: len(v) for k, v in expressions_of_length.items()}
            for new_length, new_hashes in new_expressions_of_length.items():
                expressions_of_length[new_length].update(new_hashes)
            lengths_after = {k: len(v) for k, v in new_expressions_of_length.items()}

            for length in lengths_after.keys():
                if length not in lengths_before or lengths_after[length] > lengths_before[length]:
                    new_sizes.add(length)

            if verbose:
                print(f'Constructed expressions of sizes {sorted(new_sizes)}:')
                for length, count in sorted(lengths_after.items()):
                    print(f'  {length:2d}: {count:,} new expressions')

            # Move the new hashes to the main dictionary
            for length, new_hashes in new_expressions_of_length.items():
                expressions_of_length[length].update(new_hashes)

            new_expressions_of_length.clear()

        total_expressions = sum(len(v) for v in expressions_of_length.values())

        if verbose:
            print(f"Finished generating expressions up to size {max_source_pattern_length}. Total expressions: {total_expressions:,}")
            for length, expressions in sorted(expressions_of_length.items()):
                print(f"Size {length}: {len(expressions):,} expressions")

        expressions_of_length_and_variables: dict[int, dict[tuple[str, ...], set[tuple[str, ...]]]] = {}
        for length, expressions in expressions_of_length.items():
            expressions_of_length_and_variables[length] = defaultdict(set)
            for expression in expressions:
                expression_variables = list(set(expression) & set(dummy_variables))  # This gets the dummy variables used in the expression
                expressions_of_length_and_variables[length][tuple(sorted(expression_variables))].add(expression)

        # --- Phase 2: Parallel rule finding ---
        if n_workers is None:
            n_workers = mp.cpu_count()

        # Create shared memory for arrays
        X_shm = SharedMemory(create=True, size=X_data.nbytes)
        X_shared: np.ndarray = np.ndarray(X_data.shape, dtype=X_data.dtype, buffer=X_shm.buf)
        X_shared[:] = X_data[:]

        # Create queues
        work_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()

        # Start workers
        workers = []
        for i in range(n_workers):
            p = Process(
                target=self.find_rule_worker,
                args=(
                    i, work_queue, result_queue,
                    X_data.shape, X_data.dtype, X_shm.name,
                    dict(expressions_of_length_and_variables),  # Make a copy for each worker
                    dummy_variables,
                    self.operator_arity,
                    constants_fit_challenges,
                    constants_fit_retries,
                )
            )
            p.daemon = True  # Make workers daemon processes
            p.start()
            workers.append(p)

        # Main processing loop
        n_scanned = 0
        active_tasks = 0

        # Create iterator over all work items
        work_items = [
            expression_to_simplify
            for _, expressions in sorted(expressions_of_length.items())  # We don't care about the variables here
            for expression_to_simplify in expressions
        ]
        work_iter = iter(work_items)

        pbar = tqdm(total=len(work_items), desc="Finding rules", disable=not verbose)

        current_length = 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            try:
                # Initial work distribution
                for _ in range(min(n_workers * 2, len(work_items))):  # Queue 2x workers for efficiency
                    try:
                        expression_to_simplify = next(work_iter)
                        simplified_length = len(self.simplify(expression_to_simplify, max_iter=5))
                        if max_target_pattern_length is None:
                            allowed_candidate_lengths = tuple(range(simplified_length))
                        else:
                            allowed_candidate_lengths = tuple(range(min(simplified_length, max_target_pattern_length + 1)))
                        work_queue.put((expression_to_simplify, simplified_length, allowed_candidate_lengths))
                        active_tasks += 1
                    except StopIteration:
                        break

                current_length = len(expression_to_simplify)

                # Process results and distribute new work
                try:
                    while active_tasks > 0 and not interrupted:
                        # Get result with timeout to allow checking stop conditions
                        try:
                            result = result_queue.get(timeout=0.1)
                        except queue.Empty:
                            # Check if interrupted during wait
                            if interrupted:
                                break
                            continue

                        active_tasks -= 1

                        # Process result
                        if result is not None:
                            if result[0] == 'ERROR':
                                print(f"Error in worker {result[1]}: {result[2]}")
                                print(result[2])
                                raise result[1]
                            self.simplification_rules.append(result)

                        # Send new work if available (but not if interrupted)
                        if not interrupted:
                            try:
                                expression_to_simplify = next(work_iter)

                                if len(expression_to_simplify) > current_length:
                                    # This means that the collected rules can be applied to coming expressions
                                    # To avoid redundant rules, we incorporate the rules into the simplification to raise the requirements for rules
                                    if verbose:
                                        print(f'Increasing expression length from {current_length} to {len(expression_to_simplify)}')
                                    self.simplification_rules = deduplicate_rules(self.simplification_rules, dummy_variables, verbose=verbose)
                                    self.compile_rules()
                                    if output_file is not None:
                                        if verbose:
                                            print("Saving rules after increasing expression length...")
                                        with open(output_file, 'w') as file:
                                            json.dump(self.simplification_rules, file, indent=4)
                                    current_length = len(expression_to_simplify)

                                simplified_length = len(self.simplify(expression_to_simplify, max_iter=5))
                                if max_target_pattern_length is None:
                                    allowed_candidate_lengths = tuple(range(simplified_length))
                                else:
                                    allowed_candidate_lengths = tuple(range(min(simplified_length, max_target_pattern_length + 1)))
                                work_queue.put((expression_to_simplify, simplified_length, allowed_candidate_lengths))
                                active_tasks += 1
                            except StopIteration:
                                pass

                        n_scanned += 1
                        pbar.update(1)
                        # Calculate the display string for the last rule with truncation
                        last_rule = self.simplification_rules[-1] if self.simplification_rules else 'None'
                        last_rule_str = str(last_rule)[:64].ljust(64)  # Truncate and pad

                        # Format with fixed widths
                        pbar.set_postfix_str(
                            f"Rules: {len(self.simplification_rules):>6,}, "  # 6 chars, right-aligned
                            f"Active tasks: {active_tasks:>3}, "              # 3 chars, right-aligned
                            f"Last rule: {last_rule_str}"                     # Fixed 30 chars
                        )

                        # Periodic saving
                        if output_file is not None and n_scanned % save_every == 0:
                            if verbose:
                                print(f"Saving rules after processing {n_scanned} expressions...")
                            self.simplification_rules = deduplicate_rules(self.simplification_rules, dummy_variables, verbose=verbose)
                            self.compile_rules()
                            with open(output_file, 'w') as file:
                                json.dump(self.simplification_rules, file, indent=4)
                except Exception as e:
                    print(f"Error during processing: {e}")
                    interrupted = True

            finally:
                # Restore original signal handler
                signal.signal(signal.SIGINT, old_handler)

                pbar.close()

                # Clean shutdown or force termination
                if interrupted:
                    print("Force terminating workers...")
                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=0.5)
                            if p.is_alive():
                                p.kill()
                else:
                    # Normal shutdown
                    print("Shutting down workers...")
                    for _ in workers:
                        try:
                            work_queue.put(None, timeout=0.1)
                        except Exception as e:
                            print(e)
                            pass

                    for p in workers:
                        p.join(timeout=2)
                        if p.is_alive():
                            p.terminate()

                # Cleanup resources
                try:
                    X_shm.close()
                    X_shm.unlink()
                except Exception as e:
                    print(e)
                    pass

                # Close queues
                work_queue.close()
                result_queue.close()

                if output_file is not None:
                    if verbose:
                        print("Saving results...")
                    time.sleep(1)  # Give time for the user to interrupt the process
                    self.simplification_rules = deduplicate_rules(self.simplification_rules, dummy_variables, verbose=verbose)
                    self.compile_rules()
                    with open(output_file, 'w') as file:
                        json.dump(self.simplification_rules, file, indent=4)

    def operand_key(self, operands: list) -> tuple:
        '''
        Returns a key for sorting the operands of a commutative operator.

        Parameters
        ----------
        operands : list
            The operands to sort.

        Returns
        -------
        tuple
            The key for sorting the operands.
        '''
        if len(operands) > 1 and isinstance(operands[0], str):
            # if operands[0] in self.operator_arity_compat or operands[0] in self.operator_aliases:
            # Node
            operand_keys = tuple(self.operand_key(op) for op in operands[1])
            return (2, len(flatten_nested_list(operands)), operand_keys, operands[0])

        # Leaf
        if len(operands) == 1 and isinstance(operands[0], str):
            try:
                return (1, float(operands[0]))
            except ValueError:
                return (0, operands[0])

        if isinstance(operands, str):
            return (0, operands)

        raise ValueError(f'None of the criteria matched for operands {operands}:\n1. ({len(operands) > 1}, {isinstance(operands[0], str)}, {operands[0] in self.operator_arity_compat or operands[0] in self.operator_aliases})\n2. ({len(operands) == 1}, {isinstance(operands[0], str)})\n3. ({isinstance(operands, str)})')

    def operators_to_realizations(self, prefix_expression: list[str] | tuple[str, ...]) -> list[str] | tuple[str, ...]:
        '''
        Converts a prefix expression from operators to realizations.

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression to convert.

        Returns
        -------
        list[str]
            The converted prefix expression.
        '''
        return [self.operator_realizations.get(token, token) for token in prefix_expression]

    def realizations_to_operators(self, prefix_expression: list[str]) -> list[str]:
        '''
        Converts a prefix expression from realizations to operators.

        Parameters
        ----------
        prefix_expression : list[str]
            The prefix expression to convert.

        Returns
        -------
        list[str]
            The converted prefix expression.
        '''
        return [self.realization_to_operator.get(token, token) for token in prefix_expression]

    @staticmethod
    def code_to_lambda(code: CodeType) -> Callable[..., float]:
        '''
        Converts a code object to a lambda function.

        Parameters
        ----------
        code : CodeType
            The code object to convert.

        Returns
        -------
        Callable[..., float]
            The lambda function.
        '''
        return FunctionType(code, globals())()
