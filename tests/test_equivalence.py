import pytest
import json
from pathlib import Path
import numpy as np
import warnings
from collections import defaultdict
import simplipy as sp
from unittest.mock import patch, MagicMock

# Assume the new module is at simplipy.asset_manager
from simplipy import asset_manager
from simplipy import SimpliPyEngine
from simplipy.utils import load_config # Assuming this is where it is

# --- Mock Data for Unit Tests ---

# This is the mock manifest.json that our tests will imagine is on Hugging Face.
MOCK_MANIFEST = {
    "rulesets": {
        "dev_7-2": {
            "description": "Mock engine 7-2 for testing.",
            "repo_id": "psaegert/simplipy-assets",
            "files": ["dev_7-2/config.yaml", "dev_7-2/rules.json"],
            "entrypoint": "dev_7-2/config.yaml"
        }
    },
    "test-data": {
        "expressions_10k": {
            "description": "Mock 10k expressions for testing.",
            "repo_id": "psaegert/simplipy-assets",
            "files": ["test/expressions_10k.json"],
            "entrypoint": "expressions_10k.json"
        }
    }
}

# --- Pytest Fixtures for Mocking ---

@pytest.fixture
def mock_cache_dir(tmp_path):
    """Fixture to mock the cache directory to a temporary directory."""
    with patch('simplipy.asset_manager.get_cache_dir', return_value=tmp_path):
        yield tmp_path

@pytest.fixture
def mock_hf_download(mock_cache_dir):
    """
    Fixture to mock hf_hub_download.
    It simulates downloading by creating mock files in the temp cache dir.
    """
    def _mock_download(repo_id, filename, **kwargs):
        if filename == "manifest.json":
            # Create and return the path to the mock manifest
            manifest_path = mock_cache_dir / "manifest.json"
            manifest_path.write_text(json.dumps(MOCK_MANIFEST))
            return str(manifest_path)

        # Simulate downloading an asset file
        local_dir = Path(kwargs.get("local_dir"))

        # hf_hub_download places files inside a structure, we just create the file directly
        # The filename from the manifest includes the subdirectory
        output_path = local_dir / Path(filename).name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write some dummy content to verify
        if output_path.name == "config.yaml":
            output_path.write_text("rules: rules.json\noperators: {}")
        else:
            output_path.write_text("[]") # Empty JSON for other files

        return str(output_path)

    with patch('simplipy.asset_manager.hf_hub_download', side_effect=_mock_download) as mock:
        yield mock

# --- Unit Tests for Asset Manager ---

def test_install_asset(mock_cache_dir, mock_hf_download):
    """Verify that installing an asset creates the correct files."""
    assert asset_manager.install_asset('ruleset', 'dev_7-2') is True

    asset_dir = mock_cache_dir / "rulesets" / "dev_7-2"
    assert asset_dir.exists()
    assert (asset_dir / "config.yaml").is_file()
    assert (asset_dir / "rules.json").is_file()

    # Verify the mock was called for the manifest and the two asset files
    assert mock_hf_download.call_count == 3

def test_install_already_exists(mock_cache_dir, mock_hf_download, capsys):
    """Verify that installing an existing asset does not re-download the asset files."""
    asset_manager.install_asset('ruleset', 'dev_7-2')
    mock_hf_download.reset_mock() # Reset call count after first install

    assert asset_manager.install_asset('ruleset', 'dev_7-2') is True

    # The mock should be called exactly once to fetch the manifest, but not for asset files.
    assert mock_hf_download.call_count == 1
    assert mock_hf_download.call_args.kwargs['filename'] == 'manifest.json'

    captured = capsys.readouterr()
    assert "is already installed" in captured.out

def test_force_reinstall(mock_cache_dir, mock_hf_download):
    """Verify that force=True re-downloads an existing asset."""
    asset_manager.install_asset('ruleset', 'dev_7-2')
    # The mock was called 3 times (manifest + 2 files)
    assert mock_hf_download.call_count == 3

    # Now force reinstall
    assert asset_manager.install_asset('ruleset', 'dev_7-2', force=True) is True
    # It should be called 3 more times
    assert mock_hf_download.call_count == 6

def test_remove_asset(mock_cache_dir, mock_hf_download):
    """Verify that removing an asset deletes its directory."""
    asset_manager.install_asset('ruleset', 'dev_7-2')
    asset_dir = mock_cache_dir / "rulesets" / "dev_7-2"
    assert asset_dir.exists()

    assert asset_manager.remove_asset('ruleset', 'dev_7-2') is True
    assert not asset_dir.exists()

def test_get_asset_path_auto_install(mock_cache_dir, mock_hf_download):
    """Verify get_asset_path triggers installation for a missing asset."""
    asset_path = asset_manager.get_asset_path('ruleset', 'dev_7-2')

    expected_path = mock_cache_dir / "rulesets" / "dev_7-2" / "dev_7-2" / "config.yaml"
    assert Path(asset_path).name == "config.yaml"
    assert mock_hf_download.call_count > 0 # Verify download was triggered

def test_get_asset_path_local_exists(mock_cache_dir, mock_hf_download):
    """Verify get_asset_path returns a local path without downloading."""
    local_file = mock_cache_dir / "my_local_config.yaml"
    local_file.touch()

    asset_path = asset_manager.get_asset_path('ruleset', str(local_file))

    assert asset_path == str(local_file)
    assert mock_hf_download.call_count == 0 # No download should occur

def test_list_assets(mock_cache_dir, mock_hf_download, capsys):
    """Verify listing of available and installed assets."""
    # 1. List available (nothing installed)
    asset_manager.list_assets('ruleset')
    captured = capsys.readouterr()
    assert "dev_7-2" in captured.out
    assert "[installed]" not in captured.out

    # 2. Install and list again
    asset_manager.install_asset('ruleset', 'dev_7-2')
    asset_manager.list_assets('ruleset')
    captured = capsys.readouterr()
    assert "dev_7-2" in captured.out
    assert "[installed]" in captured.out

    # 3. List only installed
    asset_manager.list_assets('ruleset', installed_only=True)
    captured = capsys.readouterr()
    assert "dev_7-2" in captured.out
    assert "[installed]" in captured.out

# --- Integration Test ---

# Mark this test to indicate it uses the network.
# You can run pytest with `pytest -m "not integration"` to skip it.
@pytest.mark.integration
def test_equivalence_10k_with_asset_manager(mock_cache_dir):
    """
    Integration test: Downloads real assets and runs the equivalence check.
    This is the original test, modified to use the new asset manager.
    """
    # --- MODIFICATION: Use asset_manager to get paths ---
    # This will automatically download and cache the assets on the first run.
    engine_config_path = asset_manager.get_asset_path('ruleset', 'dev_7-2')
    test_data_path = asset_manager.get_asset_path('test-data', 'expressions_10k')

    assert engine_config_path is not None, "Failed to get engine config path"
    assert test_data_path is not None, "Failed to get test data path"

    # --- The rest of the test is the same as before ---
    engine = SimpliPyEngine.from_config(engine_config_path)

    with open(test_data_path, "r") as f:
        expressions = json.load(f)

    dummy_variables = ['x1', 'x2', 'x3']

    X = np.random.normal(0, 5, size=(10_000, len(dummy_variables)))
    C = np.random.normal(0, 5, size=100)

    for i, expression in enumerate(expressions):
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

                relative_tolerance = 1e-5

                is_both_nan_mask = (np.isnan(y_filtered) & np.isnan(y_candidate_filtered))
                is_both_inf_mask = (np.isinf(y_filtered) & np.isinf(y_candidate_filtered))
                is_both_negative_inf_mask = (np.isneginf(y_filtered) & np.isneginf(y_candidate_filtered))
                is_both_invalid_mask = is_both_nan_mask | is_both_inf_mask | is_both_negative_inf_mask

                # absolute_equivalence_mask = abs_diff <= absolute_tolerance
                relative_equivalence_mask = np.abs(abs_diff / np.where(y_filtered != 0, y_filtered, 1)) <= relative_tolerance

                # Require 99% of values to be equivalent
                # The following is a correct simplification but creates <1% values that are not equivalent (perhaps due to numerical issues):
                # ['tan', '+', 'atan', 'x2', '*', 'exp', '-', '+', 'x2', '+', 'x3', '/', 'x2', 'x3', 'x2', 'x2'] -> ['tan', '+', 'atan', 'x2', '*', 'exp', '-', '+', 'x2', '+', 'x3', '/', 'x2', 'x3', 'x2', 'x2']
                expressions_match = np.mean(relative_equivalence_mask | is_both_invalid_mask) >= 0.99
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
