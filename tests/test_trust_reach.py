"""What a realization reaches: the callable it lands on, and submodules not imported yet.

The trust walk holds every module on a realization's path to the trusted set. Two further cases
are checked at construction:

* the terminal callable must be defined in a trusted package. A trusted module's attributes
  include callables it imported by name from other modules, and those are not modules;
* a component that names a submodule of a trusted package that is not imported yet is imported
  and walked, so the realization cannot resolve later to something the walk did not see.

No shipped config is affected: every dotted realization of the shipped configs is a
`simplipy.operators` function.
"""
import glob
import os
import subprocess
import sys

import pytest

from simplipy import SimpliPyEngine
from simplipy.io import load_config
from simplipy.trust import UntrustedModuleError

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UNARY = {'alias': [], 'arity': 1, 'precedence': 3, 'commutative': False}
_BINARY = {'alias': [], 'arity': 2, 'precedence': 1, 'commutative': True}


def _engine(realization: str, **kwargs) -> SimpliPyEngine:
    return SimpliPyEngine({'+': {'realization': '+', **_BINARY},
                           'probe': {'realization': realization, **_UNARY}}, [], **kwargs)


@pytest.mark.parametrize('realization', [
    'np.lib._datasource._open',
    'scipy._lib.deprecation.import_module',
    'np.testing._private.utils.mkdtemp',
    'simplipy.mining._MINE_LOCK.acquire',
])
def test_a_callable_defined_outside_the_trusted_packages_is_refused(realization) -> None:
    with pytest.raises(UntrustedModuleError) as excinfo:
        _engine(realization)
    message = str(excinfo.value)
    assert "'probe'" in message and realization in message
    assert 'trusted_modules=' in message


def test_a_submodule_not_imported_yet_is_walked_at_construction() -> None:
    """Run in a fresh interpreter, where nothing has imported `simplipy.verify` yet."""
    code = (
        'import sys\n'
        'from simplipy import SimpliPyEngine\n'
        'from simplipy.trust import UntrustedModuleError\n'
        'assert "simplipy.verify" not in sys.modules\n'
        "ops = {'+': {'realization': '+', 'alias': [], 'arity': 2, 'precedence': 1, 'commutative': True},\n"
        "       'probe': {'realization': 'simplipy.verify._monitor.os.getcwd', 'alias': [], 'arity': 1,\n"
        "                 'precedence': 3, 'commutative': False}}\n"
        'try:\n'
        '    SimpliPyEngine(ops, [])\n'
        "    print('loaded')\n"
        'except UntrustedModuleError as error:\n'
        "    print('refused:', error)\n")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.startswith('refused:') and "'os'" in out, out


def test_trusting_the_package_from_outside_still_grants_it() -> None:
    engine = _engine('np.lib._datasource._open', trusted_modules=['io'])
    assert engine.operator_realizations['probe'] == 'np.lib._datasource._open'
    engine = _engine('simplipy.engine.os.getcwd', trusted_modules=['os'])  # defined in `posix`
    assert engine.operator_realizations['probe'] == 'simplipy.engine.os.getcwd'


@pytest.mark.parametrize('realization', ['math.sin', 'scipy.special.erf', 'scipy.special.jv',
                                         'simplipy.operators.sin', 'np.sin', 'np.add', 'np.clip',
                                         'np.float64'])
def test_callables_of_the_trusted_packages_load(realization) -> None:
    _engine(realization)


def test_every_shipped_config_still_loads() -> None:
    configs = sorted(glob.glob(os.path.join(_ROOT, 'assets', 'engines', '*', 'config.yaml')))
    configs.append(os.path.join(_ROOT, 'tests', 'fixtures', 'legacy_vocab_config.yaml'))
    assert len(configs) >= 3
    for path in configs:
        config = load_config(path)
        SimpliPyEngine(config['operators'], [])


def test_an_unresolvable_component_still_loads() -> None:
    """A name that is neither an attribute nor a submodule reaches nothing (version skew in a
    trusted dependency must not kill an engine that never evaluates the operator)."""
    assert _engine('np.os.system').operator_realizations['probe'] == 'np.os.system'
    assert _engine('simplipy.operators.pow2').operator_realizations['probe'] == 'simplipy.operators.pow2'
