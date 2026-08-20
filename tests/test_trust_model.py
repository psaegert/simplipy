"""The realization trust model (register C1.12, owner-ruled 2026-08-09).

Two defects, both reproduced on the pre-fix engine before this file existed:

1. **A config file was executable.** ``get_used_modules`` scanned realization
   strings, and every root it found was handed to ``importlib.import_module`` --
   which runs the module's top-level code. Loading a config was therefore enough to
   execute code, before any expression was evaluated, on every path including
   ``SimpliPyEngine.load`` (which fetches a config from Hugging Face) and unpickling.
2. **All engines shared one namespace.** Imports landed in ``simplipy.engine``'s
   module globals and ``code_to_lambda`` compiled expressions against that same
   dict, so one engine's imports were reachable from another engine's expressions,
   as were simplipy's own imports (``os``, ``importlib``, ``hashlib``, ...).

The fix: an allowlist of canonical spellings (``math``, ``np``, ``scipy``,
``simplipy``) checked BEFORE any import, opt-in from outside the config only, and a
per-engine evaluation namespace. ``np`` -- not ``numpy`` -- is the spelling, because
``np.pi``/``np.e`` are normative token grammar carried by 396 rules across the
shipped triple.

The load-bearing test in this file is
:meth:`TestConfigIsNotExecutable.test_refusal_happens_before_the_import`: refusing is
worth nothing if the module was already imported when the refusal fired.
"""
import json
import os
import pickle
import sys

import pytest
import yaml

from simplipy import SimpliPyEngine, codify, explicit_constant_placeholders
from simplipy.trust import UntrustedModuleError
from conftest import acj_config_path

_UNTRUSTED = 'simplipy_trust_probe_mod'

BINARY = {"alias": [], "inverse": None, "arity": 2, "precedence": 1, "commutative": True}
UNARY = {"alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False}


def _ops(realization: str) -> dict:
    """A minimal operator table whose single unary operator carries ``realization``."""
    return {'+': {"realization": "+", **BINARY},
            'probe': {"realization": realization, **UNARY}}


@pytest.fixture
def probe_module(tmp_path, monkeypatch):
    """An importable module whose TOP-LEVEL code writes a marker file when it runs.

    The marker is what distinguishes "refused" from "refused too late": an allowlist
    that fires after the import has already happened prevents nothing.
    """
    marker = tmp_path / 'IMPORT_RAN.txt'
    (tmp_path / f'{_UNTRUSTED}.py').write_text(
        'import pathlib\n'
        f'pathlib.Path(r"{marker}").write_text("module-level code executed")\n'
        'def f(x):\n'
        '    return x\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(_UNTRUSTED, None)
    yield marker
    sys.modules.pop(_UNTRUSTED, None)


def _config(tmp_path, realization: str, extra: dict | None = None) -> str:
    path = tmp_path / 'config.yaml'
    document: dict = {'operators': _ops(realization)}
    document.update(extra or {})
    path.write_text(yaml.safe_dump(document))
    return str(path)


class TestConfigIsNotExecutable:
    def test_refusal_happens_before_the_import(self, tmp_path, probe_module):
        """THE falsifier: the untrusted module's code must not have run."""
        with pytest.raises(UntrustedModuleError):
            SimpliPyEngine.from_config(_config(tmp_path, f'{_UNTRUSTED}.f'))
        assert not probe_module.exists(), 'the module was imported before it was refused'
        assert _UNTRUSTED not in sys.modules

    def test_refused_on_direct_construction_too(self, probe_module):
        # `from_config` is not the only door: raw in-memory construction deliberately
        # bypasses the artifact-generation gate, so the trust check cannot live there.
        with pytest.raises(UntrustedModuleError):
            SimpliPyEngine(operators=_ops(f'{_UNTRUSTED}.f'), rules=[])
        assert not probe_module.exists()

    def test_refused_on_unpickle(self, probe_module, monkeypatch):
        # A worker rebuilds the engine through __setstate__ -> import_modules. Trust
        # travels IN the state, so a pickle made under an opt-in keeps it and one made
        # without it stays refused even if the worker's own environment is permissive.
        trusted = SimpliPyEngine(operators=_ops(f'{_UNTRUSTED}.f'), rules=[],
                                 trusted_modules=[_UNTRUSTED])
        assert probe_module.exists(), 'the opted-in engine should have imported it'
        probe_module.unlink()
        sys.modules.pop(_UNTRUSTED, None)
        restored = pickle.loads(pickle.dumps(trusted))
        assert restored.modules == trusted.modules
        assert probe_module.exists(), 'the worker must rebuild its own namespace'

        untrusted_state = pickle.dumps(SimpliPyEngine(operators=_ops('np.sin'), rules=[]))
        engine = pickle.loads(untrusted_state)
        assert engine.simplify(['+', 'x0', '0']) == ['x0']

    def test_error_names_the_operator_and_both_opt_ins(self, tmp_path, probe_module):
        with pytest.raises(UntrustedModuleError) as excinfo:
            SimpliPyEngine.from_config(_config(tmp_path, f'{_UNTRUSTED}.f'))
        message = str(excinfo.value)
        assert "'probe'" in message, 'the refusal must point at the config line'
        assert _UNTRUSTED in message
        assert 'trusted_modules=' in message and 'SIMPLIPY_TRUSTED_MODULES' in message


class TestOptIn:
    def test_argument_grants_trust(self, tmp_path, probe_module):
        engine = SimpliPyEngine.from_config(
            _config(tmp_path, f'{_UNTRUSTED}.f'), trusted_modules=[_UNTRUSTED])
        assert probe_module.exists()
        assert _UNTRUSTED in engine.modules

    def test_environment_grants_trust(self, tmp_path, probe_module, monkeypatch):
        monkeypatch.setenv('SIMPLIPY_TRUSTED_MODULES', f'other_mod,{_UNTRUSTED}')
        engine = SimpliPyEngine.from_config(_config(tmp_path, f'{_UNTRUSTED}.f'))
        assert probe_module.exists()
        assert _UNTRUSTED in engine.modules

    def test_a_config_cannot_authorize_itself(self, tmp_path, probe_module):
        """The trap the design exists to avoid: the author of the dangerous line is
        the author of the permission line, so an in-config key must grant nothing."""
        path = _config(tmp_path, f'{_UNTRUSTED}.f', extra={'trusted_modules': [_UNTRUSTED]})
        with pytest.raises(UntrustedModuleError):
            SimpliPyEngine.from_config(path)
        assert not probe_module.exists()

    def test_opting_in_to_one_module_does_not_trust_another(self, tmp_path, probe_module):
        with pytest.raises(UntrustedModuleError):
            SimpliPyEngine.from_config(_config(tmp_path, f'{_UNTRUSTED}.f'),
                                       trusted_modules=['some_other_module'])
        assert not probe_module.exists()


class TestOneSpellingPerModule:
    def test_numpy_spelling_refused_with_a_hint(self):
        with pytest.raises(UntrustedModuleError) as excinfo:
            SimpliPyEngine(operators=_ops('numpy.sin'), rules=[])
        message = str(excinfo.value)
        assert "'np'" in message and 'token grammar' in message

    def test_np_spelling_is_the_canonical_one(self):
        engine = SimpliPyEngine(operators=_ops('np.sin'), rules=[])
        # `modules` keeps its historical meaning -- IMPORTABLE package names, which
        # downstream consumers iterate and import (flash-ansr's Refiner does).
        assert engine.modules == ['numpy']
        for module in engine.modules:
            __import__(module)

    def test_default_roots_load(self):
        for realization in ('math.sin', 'scipy.special.erf', 'simplipy.operators.sin', 'np.sin'):
            SimpliPyEngine(operators=_ops(realization), rules=[])


class TestPerEngineNamespace:
    def test_engines_do_not_share_a_namespace(self, tmp_path, probe_module):
        loaded = SimpliPyEngine.from_config(
            _config(tmp_path, f'{_UNTRUSTED}.f'), trusted_modules=[_UNTRUSTED])
        assert loaded.code_to_lambda(codify(f'{_UNTRUSTED}.f(41) + 1', []))() == 42

        innocent = SimpliPyEngine(operators=_ops('np.sin'), rules=[])
        with pytest.raises(NameError):
            innocent.code_to_lambda(codify(f'{_UNTRUSTED}.f(1)', []))()

    def test_simplipys_own_imports_are_not_reachable(self):
        engine = SimpliPyEngine(operators=_ops('np.sin'), rules=[])
        for expression in ("os.path.basename('a/b')", "importlib.import_module('base64')",
                           "hashlib.sha256(b'x')", "json.dumps({})", "yaml.safe_load('1')"):
            with pytest.raises(NameError):
                engine.code_to_lambda(codify(expression, []))()

    def test_token_grammar_constants_always_evaluate(self):
        # `np.pi` / `np.e` may appear in ANY expression regardless of what the config
        # names, so `np` is bound unconditionally.
        engine = SimpliPyEngine(operators={'+': {"realization": "+", **BINARY}}, rules=[])
        assert engine.modules == ['numpy']
        assert engine.code_to_lambda(codify('np.pi + np.e', []))() == pytest.approx(5.859874482048838)


class TestDeployedEvaluationPath:
    """The realization -> infix -> codify -> lambda path, which is what actually
    consumes the imports (the Rust core dispatches natively and needs none of it)."""

    @pytest.fixture(scope='class')
    def engine(self):
        return SimpliPyEngine.from_config(acj_config_path())

    def test_shipped_config_needs_only_default_roots(self, engine):
        assert engine.modules == ['numpy', 'simplipy']

    def test_corpus_expressions_still_evaluate(self, engine):
        """Namespace completeness against real expressions, in the shape the deployed
        refiner uses: simplify, render with realizations, turn ``<constant>`` into
        named lambda arguments, codify, call. Every name the rendered row mentions has
        to resolve in the engine's own namespace, or the downstream refiner breaks."""
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, 'benchmarks', 'corpus', 'raw_skeletons_nv.json')) as file:
            corpus = json.load(file)

        checked = 0
        with_constants = 0
        for row in corpus:
            variables = sorted({t for t in row if t.startswith('x') and t[1:].isdigit()})
            if not variables:
                continue
            simplified = engine.simplify(list(row))
            with_placeholders, constants = explicit_constant_placeholders(
                simplified, convert_numbers_to_constant=False)
            infix = engine.prefix_to_infix(with_placeholders, realization=True)
            function = engine.code_to_lambda(codify(infix, variables + constants))
            function(*[0.7] * (len(variables) + len(constants)))  # a NameError here is the regression
            checked += 1
            with_constants += bool(constants)
            if checked == 80:
                break
        assert checked == 80, 'expected enough corpus rows to be meaningful'
        assert with_constants, 'the constant-bearing shape (the refiner path) must be covered'


class TestRealizationShapeIsTotal:
    """R3/B14 (SEC-L1): the shape check must decide EVERY realization, not just dotted ones.

    ``realization_root`` returns ``None`` for anything that is not a dotted name, and the
    engine read that as "nothing to check" (``if root is not None:``, no else branch). So a
    realization that was neither a module reference nor a bare name was never checked and
    never refused -- it travelled to ``code_to_lambda``, which INTERPOLATES IT INTO
    GENERATED SOURCE. Reproduced before the fix: the config below loaded without complaint
    and wrote its marker at first evaluation."""

    def test_an_expression_realization_is_refused_at_load(self, tmp_path):
        marker = tmp_path / 'PWNED'
        payload = f'__import__("pathlib").Path(r"{marker}").write_text("pwned") or abs'
        with pytest.raises(UntrustedModuleError) as excinfo:
            SimpliPyEngine.from_config(_config(tmp_path, payload))
        assert 'probe' in str(excinfo.value)          # names the operator, not the engine
        assert not marker.exists(), 'the realization executed'

    def test_the_payload_never_reaches_the_evaluation_path(self, tmp_path):
        """Refusal at LOAD, not at first evaluation: the engine must not exist at all."""
        marker = tmp_path / 'PWNED_EVAL'
        payload = f'__import__("pathlib").Path(r"{marker}").write_text("pwned") or abs'
        with pytest.raises(UntrustedModuleError):
            engine = SimpliPyEngine.from_config(_config(tmp_path, payload))
            engine.code_to_lambda(codify(engine.prefix_to_infix(['probe', 'x0'], realization=True), ['x0']))(1.0)
        assert not marker.exists()

    @pytest.mark.parametrize('realization', [
        '__import__("os").system', 'lambda x: x', 'abs(x0) or __import__("os")',
        '__import__', '_private', 'a b', '', 'np.sin(1) if True else 0', 'np . sin',
    ])
    def test_non_name_shapes_are_refused(self, tmp_path, realization):
        with pytest.raises(UntrustedModuleError):
            SimpliPyEngine.from_config(_config(tmp_path, realization))

    @pytest.mark.parametrize('realization', ['+', '-', '*', '/', '**', 'abs'])
    def test_the_shapes_real_configs_use_still_load(self, tmp_path, realization):
        """Controls: the bare operator symbols every shipped config carries, and the bare
        builtin the module docstring promises, reference no module and stay allowed."""
        engine = SimpliPyEngine.from_config(_config(tmp_path, realization))
        assert engine.operator_realizations['probe'] == realization
