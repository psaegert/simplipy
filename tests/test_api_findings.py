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


# API-1: the infix reader drops every character it has no token for, silently, so text outside
# the grammar reads as another expression instead of being refused ("Raises ValueError on a
# malformed expression"). `|x0|` is read as `x0`, `x0!` as `x0`, `0x1F` as `0*x1F`, `?0 + x1` as
# `0 + x1`, `~x0` as `x0`, `<constant_a-b>` as `constant_a - b`, and a fullwidth `x0` as the empty
# expression, whose simplification is ''. The tokenizer documents the
# drop as "pinned legacy parity" (rust/convert.rs, tokenize_infix); the public docs disclose only
# the dropped unmatched opening parenthesis.
@pytest.mark.xfail(strict=True, reason='API-1: unknown characters are dropped silently (|x0| reads as x0)')
@pytest.mark.parametrize('text', ['|x0|', 'x0!', '0x1F', '?0 + x1', '~x0', '<constant_a-b>*x0',
                                  'ｘ０', '[x0]', 'x0\x00 + 1'])
def test_a_character_outside_the_grammar_is_refused(engine, text) -> None:
    with pytest.raises(ValueError):
        engine.simplify(text)


# API-2: `is_valid` answers True for token sequences that every reader refuses: a reserved
# numeric spelling (`nan`, `inf`, `1/0`, `1/-3`) or an empty or whitespace token. The explicit
# path asks only the core's arity oracle; the readers also run the token grammar (`check_form`)
# and the 4,096-token cap (`is_valid` answers True for a 4,097-token sequence every reader refuses).
# The miner calls `engine.is_valid` (mining.py, three sites), so aligning it is the owner's call.
@pytest.mark.xfail(strict=True, reason='API-2: is_valid is True for tokens every reader refuses')
@pytest.mark.parametrize('tokens', [['+', 'nan', 'x0'], ['+', 'inf', 'x0'], ['+', '1/0', 'x0'],
                                    ['+', '1/-3', 'x0'], ['+', '', 'x0'], ['+', 'x 0', 'x0'],
                                    ['sin'] * 4096 + ['x0']])
def test_is_valid_agrees_with_the_readers(engine, tokens) -> None:
    with pytest.raises(ValueError):
        engine.to_prefix(tokens)
    assert engine.is_valid(tokens) is False


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


# API-4: below the 4,096-token cap, `mask` (and `normalization.to_skeleton`) recurse in Python
# and raise RecursionError from depth 1,000 (the F22 class, fixed for `evaluate_constants`).
@pytest.mark.xfail(strict=True, reason='API-4: mask raises RecursionError at depth 1000 (< the token cap)')
def test_mask_reads_every_depth_the_core_reads(engine) -> None:
    deep = ['sin'] * 1500 + ['x0']
    assert engine.simplify(deep) == deep
    assert engine.mask(deep) == deep


# API-5: an alias that names another declared operator loads silently, and then the infix reader
# and the prefix reader disagree: with `sin: {alias: [cos]}`, `simplify('cos(x0)')` is `sin(x0)`
# while `simplify(['cos', 'x0'])` is `cos(x0)`. Defined functions refuse such a name; aliases do not.
@pytest.mark.xfail(strict=True, reason='API-5: an alias may shadow another operator (infix and prefix then disagree)')
def test_an_alias_naming_another_operator_is_refused() -> None:
    with pytest.raises(ValueError):
        _with_operator(sin={'alias': ['cos']})


# API-7: `as_code` interpolates every leaf token into Python source, and the token grammar admits
# any whitespace-free string as a variable, so a token that is Python source runs when the
# compiled function is called. The docs disclaim hostile expressions (docs/guides/trust.md), but
# `to_prefix` and `is_valid` accept the token, and nothing tells a caller that the expression is
# not one. A leaf that is neither a number, a constant, an operator nor a Python identifier could
# be refused before compiling, which changes no simplify or complexity answer.
@pytest.mark.xfail(strict=True, reason='API-7: as_code compiles a leaf token that is Python source')
def test_as_code_refuses_a_leaf_that_is_not_an_identifier(engine, tmp_path) -> None:
    mark = tmp_path / 'ran'
    payload = f"__import__('pathlib').Path({str(mark)!r}).touch()"
    with pytest.raises(ValueError):
        engine.as_callable(['+', 'x0', payload], variables=['x0'])(np.array([1.0]))
    assert not mark.exists()


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
