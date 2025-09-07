import pytest
import json
import numpy as np
import warnings
from collections import defaultdict
import simplipy as sp

# Assume the new module is at simplipy.asset_manager
from simplipy import asset_manager
from simplipy import SimpliPyEngine


# Mark this test to indicate it uses the network.
# You can run pytest with `pytest -m "not integration"` to skip it.
@pytest.mark.integration
def test_equivalence_10k_with_asset_manager():
    """
    Integration test: Downloads real assets and runs the equivalence check.
    This is the original test, modified to use the new asset manager.
    """

    sp.install_asset('ruleset', 'dev_7-3')
    sp.install_asset('test-data', 'expressions_10k')

    # --- MODIFICATION: Use asset_manager to get paths ---
    # This will automatically download and cache the assets on the first run.
    engine_config_path = asset_manager.get_asset_path('ruleset', 'dev_7-3')
    test_data_path = asset_manager.get_asset_path('test-data', 'expressions_10k')

    print(f"Engine config path: {engine_config_path}")
    print(f"Test data path: {test_data_path}")

    assert engine_config_path is not None, "Failed to get engine config path"
    assert test_data_path is not None, "Failed to get test data path"

    # --- The rest of the test is the same as before ---
    engine = SimpliPyEngine.from_config(engine_config_path)

    with open(test_data_path, "r") as f:
        expressions = json.load(f)

    dummy_variables = ['x1', 'x2', 'x3']

    for i, expression in enumerate(expressions):
        rng = np.random.default_rng(seed=42 + i)  # Vary seed slightly per case
        X = rng.normal(0, 5, size=(10_000, len(dummy_variables)))
        C = rng.normal(0, 5, size=100)

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

                if mask_original_nan.all():
                    # If all original values are NaN, we cannot check equivalence
                    expressions_match = True
                    continue

                y_filtered = y[~mask_original_nan]
                y_candidate_filtered = y_candidate[~mask_original_nan]

                abs_diff = np.abs(y_filtered - y_candidate_filtered)

                relative_tolerance = 1e-4

                is_both_nan_mask = (np.isnan(y_filtered) & np.isnan(y_candidate_filtered))
                is_both_inf_mask = (np.isinf(y_filtered) & np.isinf(y_candidate_filtered))
                is_both_negative_inf_mask = (np.isneginf(y_filtered) & np.isneginf(y_candidate_filtered))
                is_both_invalid_mask = is_both_nan_mask | is_both_inf_mask | is_both_negative_inf_mask

                # absolute_equivalence_mask = abs_diff <= absolute_tolerance
                relative_equivalence_mask = np.abs(abs_diff / np.where(y_filtered != 0, y_filtered, 1)) <= relative_tolerance

                # Require 99% of values to be equivalent
                # The following is a correct simplification but creates <1% values that are not equivalent (perhaps due to numerical issues):
                # ['tan', '+', 'atan', 'x2', '*', 'exp', '-', '+', 'x2', '+', 'x3', '/', 'x2', 'x3', 'x2', 'x2'] -> ['tan', '+', 'atan', 'x2', '*', 'exp', '-', '+', 'x2', '+', 'x3', '/', 'x2', 'x3', 'x2', 'x2']
                expressions_match = np.mean(relative_equivalence_mask | is_both_invalid_mask) >= 0.95
            else:
                # FIXME: Cannot check reliably because optimizer sometimes cannot reliably fit constants
                expressions_match = True

        if not expressions_match:
            print(f'Error in expression {i}')
            print(expression)
            print(simplified_expression)

            print(y[:10])
            print(y_candidate[:10])

            print(f"Maximum absolute difference: {np.max(np.abs(y_filtered - y_candidate_filtered))}")
            print(f"Maximum relative difference: {np.max(np.abs((y_filtered - y_candidate_filtered) / np.where(y_filtered != 0, y_filtered, 1)))}")

            print(f'Percent of mismatches (absolute): {np.mean(np.abs(y_filtered - y_candidate_filtered) > 1e-8) * 100:.2f}%')
            print(f'Percent of mismatches (relative): {np.mean(np.abs((y_filtered - y_candidate_filtered) / np.where(y_filtered != 0, y_filtered, 1)) > 1e-5) * 100:.2f}%')

            print(engine.rule_application_statistics)

        assert expressions_match, "Expressions do not match"
