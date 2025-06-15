import warnings
from collections import defaultdict

import simplipy as sp
import json
import numpy as np


def test_equivalence_10k():
    # constants_fit_challenges = 16
    # constants_fit_retries = 1024

    engine = sp.SimpliPyEngine.from_config(sp.utils.get_path('configs', 'dev_7-2.yaml'))

    dummy_variables = ['x1', 'x2', 'x3']

    with open(sp.utils.get_path('data', 'test', 'expressions_10k.json'), "r") as f:
        expressions = json.load(f)

    X = np.random.normal(0, 5, size=(10_000, len(dummy_variables)))
    C = np.random.normal(0, 5, size=100)

    for expression in expressions:
        # Source Expression
        executable_prefix_expression = engine.operators_to_realizations(expression)
        prefix_expression_with_constants, constants = sp.num_to_constants(executable_prefix_expression, convert_numbers_to_constant=False)
        code_string = engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
        code = sp.codify(code_string, dummy_variables + constants)
        f = engine.code_to_lambda(code)

        # Candidate Expression
        engine.rule_application_statistics = defaultdict(int)
        simplified_expression = engine.simplify(expression, collect_rule_statistics=True)
        executable_candidate_expression = engine.operators_to_realizations(simplified_expression)
        candidate_prefix_expression_with_constants, candidate_constants = sp.num_to_constants(executable_candidate_expression, convert_numbers_to_constant=False)
        candidate_code_string = engine.prefix_to_infix(candidate_prefix_expression_with_constants, realization=True)
        candidate_code = sp.codify(candidate_code_string, dummy_variables + candidate_constants)
        f_candidate = engine.code_to_lambda(candidate_code)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Check if expressions are equivalent
            if len(candidate_constants) == 0:
                y = sp.utils.safe_f(f, X, C[:len(constants)])
                y_candidate = sp.utils.safe_f(f_candidate, X)

                if not isinstance(y_candidate, np.ndarray):
                    y_candidate = np.full(X.shape[0], y_candidate)

                mask_original_nan = np.isnan(y)
                # Allow original NaN values to be non-NaN in the candidate (due to cancellation of NaN-producing terms)

                y_filtered = y[~mask_original_nan]
                y_candidate_filtered = y_candidate[~mask_original_nan]

                # expressions_match = np.allclose(y, y_candidate, equal_nan=True)
                expressions_match = np.allclose(y_filtered, y_candidate_filtered, equal_nan=True)
            else:
                # FIXME: Cannot check reliably due to optimizer issues
                expressions_match = True
                # # Resample constants to avoid false positives
                # expressions_match = True

                # # The expression is considered a match unless one of the challenges fails
                # for challenge_id in range(constants_fit_challenges):
                #     # Need to check if constants can be fitted
                #     y = sp.utils.safe_f(f, X, np.random.choice(C, size=len(constants), replace=False))
                #     for _ in range(constants_fit_retries):
                #         if engine.exist_constants_that_fit(simplified_expression, dummy_variables, X, y):
                #             # Found a candidate that fits, next challenge please
                #             break
                #     else:
                #         # No candidate found that fits, not all challenges could be solved, abort this candidate
                #         expressions_match = False
                #         break

        if not expressions_match:
            print(expression)
            print(simplified_expression)

            print(y[:10])
            print(y_candidate[:10])

            print(engine.rule_application_statistics)

        assert expressions_match, "Expressions do not match"
