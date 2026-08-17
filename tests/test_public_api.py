"""The declared public surface (D11): the ratified __all__ column, landed.

The column is `remine/D11_API_COLUMN.md` (owner-ratified 2026-08-16); these
tests pin its landing plan. Red on the pre-landing tree by construction: the
root had no ``__all__``, so ``from simplipy import *`` injected eight
submodules into the caller's namespace — one of which shadowed stdlib ``io``
(N2 §2c, the measured probe).
"""
import importlib

import pytest


# R1-R13 + __version__ — the root column, verbatim.
ROOT_ALL = {
    'Mode', 'SimpliPyEngine', '__version__',
    'codify', 'deduplicate_rules', 'explicit_constant_placeholders',
    'get_path', 'install', 'list_assets', 'uninstall',
    'normalize_expression', 'normalize_skeleton', 'normalize_variable_token',
}

# R18, R21, R22, R27, R30 — the module columns, verbatim.
MODULE_ALLS = {
    'simplipy.utils': {
        'codify', 'deduplicate_rules', 'explicit_constant_placeholders',
        'remap_expression', 'substitute_constants', 'substitude_constants',
        'construct_expressions', 'numbers_to_constant', 'is_numeric_string',
        'enumerate_expressions', 'count_expressions', 'sample_expression',
        'compositions',
    },
    'simplipy.io': {'load_config'},
    'simplipy.asset_manager': {
        'get_path', 'install_asset', 'uninstall_asset', 'list_assets',
        'AssetType',
    },
    'simplipy.engine': {'SimpliPyEngine', 'Mode', 'ARTIFACT_ENV_SWITCHES'},
    'simplipy.mining': {'RuleMiner'},
}


class TestRootSurface:
    def test_root_declares_the_column(self) -> None:
        import simplipy
        assert hasattr(simplipy, '__all__'), 'the root declares no __all__'
        assert set(simplipy.__all__) == ROOT_ALL

    def test_star_import_binds_exactly_the_declared(self) -> None:
        ns: dict = {}
        exec('from simplipy import *', ns)
        bound = {n for n in ns if n != '__builtins__'}
        assert bound == ROOT_ALL

    def test_star_import_does_not_shadow_stdlib_io(self) -> None:
        """N2 §2c: `from simplipy import *` used to rebind `io` to
        simplipy.io. DP-A: submodule names stay out of __all__."""
        import io as stdlib_io
        ns: dict = {}
        exec('import io\nfrom simplipy import *', ns)
        assert ns['io'] is stdlib_io

    def test_every_declared_root_name_resolves(self) -> None:
        import simplipy
        for name in simplipy.__all__:
            assert getattr(simplipy, name, None) is not None, name


class TestModuleSurfaces:
    @pytest.mark.parametrize('module_name', sorted(MODULE_ALLS))
    def test_module_declares_the_column(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        assert hasattr(module, '__all__'), f'{module_name} declares no __all__'
        assert set(module.__all__) == MODULE_ALLS[module_name]

    @pytest.mark.parametrize('module_name', sorted(MODULE_ALLS))
    def test_every_declared_name_resolves(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            assert getattr(module, name, None) is not None, name


class TestPowerUserCaveatsInline:
    """R13/R31: the compile-to-source caveat rides in the docstring, not only
    in index.md prose."""

    def test_codify_carries_the_caveat(self) -> None:
        from simplipy.utils import codify
        assert 'unsafe' in (codify.__doc__ or '')

    def test_code_to_lambda_carries_the_caveat(self) -> None:
        from simplipy import SimpliPyEngine
        assert 'unsafe' in (SimpliPyEngine.code_to_lambda.__doc__ or '')


class TestRunningDeprecations:
    """The policy's §2 promises, executable."""

    def test_substitude_misspelling_warns_and_works(self) -> None:
        """R19: the misspelling a shipped downstream imports stays for 0.13.x,
        warns from 0.13.0, and remains substitute_constants by identity of
        behaviour."""
        from simplipy.utils import substitude_constants
        with pytest.warns(DeprecationWarning, match='substitute_constants'):
            out = substitude_constants(['+', '<constant>', 'x0'], [3.14])
        assert out == ['+', '3.14', 'x0']

    def test_correct_spelling_does_not_warn(self) -> None:
        import warnings
        from simplipy.utils import substitute_constants
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            out = substitute_constants(['+', '<constant>', 'x0'], [3.14])
        assert out == ['+', '3.14', 'x0']


class TestD30ImportIsLight:
    """D30 (ratified): the import-cost acceptance is structural —
    `huggingface_hub` must be absent from sys.modules after `import simplipy`.
    Measured motivation (N3): the hf import chain was 122 ms of a 282 ms
    import paid by users who only call `simplify`."""

    def test_huggingface_hub_absent_after_import(self) -> None:
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, '-c',
             'import simplipy, sys; print("huggingface_hub" in sys.modules)'],
            capture_output=True, text=True, check=True)
        assert out.stdout.strip() == 'False'
