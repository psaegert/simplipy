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

