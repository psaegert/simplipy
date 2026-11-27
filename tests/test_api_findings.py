"""Open findings of the pre-audit review of the public API (angle "api", 2026-11-27, at 3a6ed3a).

Each test states the behaviour the contract asks for and is a STRICT xfail today: it fails now,
and the day a fix lands it passes, which fails the suite until the marker is removed. The fixes
are with the owner (several change what an input reads as, or what the miner's `is_valid` call
answers), so none is applied here; the reason of each marker names the finding of the review
report (API-n) and the behaviour measured at 3a6ed3a.
"""
import copy
import os
import warnings

import numpy as np
import pytest

from simplipy import SimpliPyEngine
from simplipy.io import load_config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BARE_CONFIG = os.path.join(_ROOT, 'assets', 'engines', 'acj2-4', 'config.yaml')


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SimpliPyEngine.from_config(_BARE_CONFIG)


def _with_operator(**changes) -> SimpliPyEngine:
    config = load_config(_BARE_CONFIG)
    operators = copy.deepcopy(config['operators'])
    for name, spec in changes.items():
        operators[name] = dict(operators[name], **spec)
    return SimpliPyEngine(operators, [], defined_functions=config.get('defined_functions'))


# API-3: `mask` returns some malformed tagged sequences unchanged instead of raising
# ("Raises ValueError ... on a malformed expression"); every other reader refuses them.
@pytest.mark.xfail(strict=True, reason='API-3: mask passes malformed tagged sequences through')
@pytest.mark.parametrize('tokens', [['<add>', '</add>'], ['<add>', '<sub>', 'x0', '</add>'],
                                    ['<mul>', 'x0', '<sub>', 'x1', '</mul>']])
def test_mask_refuses_a_malformed_tagged_sequence(engine, tokens) -> None:
    with pytest.raises(ValueError):
        engine.to_prefix(tokens)
    with pytest.raises(ValueError):
        engine.mask(tokens)


# API-8: a pre-0.15 engine pickle unpickles in 0.15 without the gate `from_config` applies to the
# same rule files (IncompatibleArtifactError: "serves only rule files mined by simplipy 0.15 or
# later"), and reads them in the strict canon, not the legacy canon the suites' explicit opt-in
# uses. Measured with a real simplipy 0.14.7 pickle of the acj-4 engine: 5,338 + 5,499 rules
# served, no warning. Simulated here with the 0.14.7 state's shape (no `_rule_canons`, a
# `real_simplification_rules` attribute).
@pytest.mark.xfail(strict=True, reason='API-8: a pre-0.15 engine pickle bypasses the 0.15 rule-file gate')
def test_a_pre_015_engine_pickle_is_refused_or_announced(engine) -> None:
    state = engine.__getstate__()
    state.pop('_rule_canons', None)
    state.pop('defined_functions', None)
    state['real_simplification_rules'] = None
    state['simplification_rules'] = [(('sin', '_0'), ('_0',))]  # rules the 0.15 gate never saw
    clone = SimpliPyEngine.__new__(SimpliPyEngine)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(Exception):
            clone.__setstate__(state)
