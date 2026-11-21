"""A rule file is read in the canon it was certified in.

A rule is judged on the pattern its mine wrote, read in its mine's canon. Read in another
canon, its LHS can become a pattern nothing judged: a canon that cancels `atanh(tanh(t))` reads
the f64 fact `rootn 2 atanh tanh 3 -> nan` (f64's atanh(tanh(3)) is not 3) as
`rootn 2 3 -> nan`, which is false. A per-mode mine's sidecar records its mode; a file mined
before per-mode mines records none and is read in the one canon those mines used."""
import json
import re

import numpy as np
import pytest
import yaml

from conftest import acj_config_path, require_or_skip
from simplipy import SimpliPyEngine
from conftest import legacy_from_config  # noqa: E402

_OPS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "rootn": {"realization": "simplipy.operators.rootn", "alias": [], "arity": 2, "precedence": 3,
              "commutative": False},
    "tanh": {"realization": "np.tanh", "alias": [], "inverse": "atanh", "arity": 1, "precedence": 3,
             "commutative": False},
    "atanh": {"realization": "np.arctanh", "alias": [], "inverse": "tanh", "arity": 1, "precedence": 3,
              "commutative": False},
}
#: An f64 fact: f64's atanh(tanh(3)) is not 3, so rootn reads a non-integer degree.
_F64_FACT = [[['rootn', '2', 'atanh', 'tanh', '3'], ['float("nan")']]]
MODES = ('f64', 'permissive')


@pytest.mark.parametrize('canon, core_name', [
    ('legacy', 'legacy'), ('permissive', 'permissive'), ('f64', 'default')])
def test_a_set_is_read_in_the_canon_it_was_certified_in(canon, core_name) -> None:
    """The canon a set was certified in is part of its recipe: the core reads the set in it, and
    a rebuilt core reads it the same way. The f64 fact never fires on `rootn(2, 3)`."""
    expected = SimpliPyEngine(operators=_OPS, rules=[]).simplify('rootn(2, 3)', mode='permissive')
    engine = SimpliPyEngine(operators=_OPS, rules=[], rules_permissive=_F64_FACT,
                            rule_canons={'permissive': canon})
    assert engine._core.mode_rules_canon(rule_mode='permissive') == core_name
    assert engine.simplify('rootn(2, 3)', mode='permissive') == expected
    engine.compile_rules()
    assert engine._core.mode_rules_canon(rule_mode='permissive') == core_name
    assert engine.simplify('rootn(2, 3)', mode='permissive') == expected


def test_the_retired_canon_is_refused() -> None:
    with pytest.raises(ValueError, match='the real mode is retired'):
        SimpliPyEngine(operators=_OPS, rules=[], rules_permissive=_F64_FACT,
                       rule_canons={'permissive': 'real'})


def test_an_in_memory_set_is_read_in_its_own_mode() -> None:
    engine = SimpliPyEngine(operators=_OPS, rules=[], rules_permissive=[])
    assert {m: engine._core.mode_rules_canon(rule_mode=m)
            for m in ('default', 'permissive')} == {
        'default': 'default', 'permissive': 'permissive'}


def test_from_config_reads_the_canon_off_each_sidecar(tmp_path) -> None:
    for name in ('rules.json', 'rules_permissive.json'):
        (tmp_path / name).write_text(json.dumps(_F64_FACT))
    # a per-mode mine of permissive; a pre-0.15 sidecar without a mode
    (tmp_path / 'rules_permissive.json.provenance.json').write_text(json.dumps({'mode': 'permissive'}))
    (tmp_path / 'rules.json.provenance.json').write_text(json.dumps({'simplipy_version': '0.14.0'}))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({
        'engine_generation': 2, 'operators': _OPS, 'rules': 'rules.json',
        'rules_permissive': 'rules_permissive.json'}))
    # (the pre-0.15 file is read through the legacy opt-in; the public loader refuses it)
    engine = legacy_from_config(str(tmp_path / 'config.yaml'))
    assert {m: engine._core.mode_rules_canon(rule_mode=m)
            for m in ('default', 'permissive')} == {
        'default': 'legacy', 'permissive': 'permissive'}
    (tmp_path / 'rules_permissive.json.provenance.json').write_text(json.dumps({'mode': 'f65'}))
    with pytest.raises(ValueError, match='f65'):
        legacy_from_config(str(tmp_path / 'config.yaml'))
    (tmp_path / 'rules_permissive.json.provenance.json').write_text(json.dumps({'mode': 'real'}))
    with pytest.raises(ValueError, match='the real mode is retired'):
        legacy_from_config(str(tmp_path / 'config.yaml'))


@pytest.fixture(scope='module')
def shipped():
    require_or_skip(acj_config_path(), 'needs the acj asset')
    return legacy_from_config(acj_config_path())


def test_the_shipped_asset_is_read_in_the_legacy_canon(shipped) -> None:
    assert {m: shipped._core.mode_rules_canon(rule_mode=m)
            for m in ('default', 'permissive')} == {
        'default': 'legacy', 'permissive': 'legacy'}


@pytest.mark.parametrize('expr', ['rootn(2, 3)', 'x1 + rootn(2, 3)', 'x1*rootn(10, 5)', 'rootn(-2, 3)'])
def test_rootn_of_a_literal_in_every_mode(shipped, expr) -> None:
    for mode in MODES:
        assert 'nan' not in shipped.simplify(expr, mode=mode), (expr, mode)


_WC = re.compile(r'^[_?!$]\d+$')


def _off_rows(engine, mode: str) -> list:
    """The served rows of `mode` whose sides the evaluator reads differently on more than a
    fifth of the points where the LHS is finite (wildcards as variables, `<constant>` rows
    left out): the family the misreading made, re-checked by the evaluator."""
    rng = np.random.default_rng(0)
    data = {f'w{i}': rng.uniform(-3, 3, 64) for i in range(8)}

    def value(tokens):
        tokens = ['w' + t[1:] if _WC.match(t) else t for t in tokens]
        names = sorted({t for t in tokens if re.fullmatch(r'w\d', t)}) or ['w0']
        with np.errstate(all='ignore'):
            y = engine.as_callable(tokens, variables=names)(*[data[v] for v in names])
        return np.broadcast_to(np.asarray(y, dtype=float), (64,))

    off = []
    for lhs, rhs, row in engine._core.ac_served_rules_in_mode(rule_mode=mode):
        if '<constant>' in lhs + rhs or not any(t == 'float("nan")' for t in rhs):
            continue
        a, b = value(lhs), value(rhs)
        fa = np.isfinite(a)
        if fa.any() and (fa & ~np.isfinite(b)).sum() > 0.2 * 64:
            off.append((row, lhs, rhs))
    return off


@pytest.fixture(scope='module')
def default_engine():
    """The default artifact (the one the family was found on), when it is staged: it is too
    large for the CI asset stage, so it is not required there."""
    import os
    import simplipy
    config = os.path.join(os.path.expanduser('~'), '.cache', 'simplipy', 'engines',
                          simplipy.DEFAULT_ENGINE, 'config.yaml')
    if not os.path.exists(config):
        pytest.skip(f'the default artifact {simplipy.DEFAULT_ENGINE} is not staged')
    # a pre-0.15 mine, read through the legacy opt-in
    return legacy_from_config(config, modes=('f64', 'permissive'))


def test_no_served_nan_rule_has_a_finite_lhs(shipped, default_engine) -> None:
    """Every served `-> nan` row's LHS is undefined where the evaluator reads it."""
    for engine in (shipped, default_engine):
        assert _off_rows(engine, 'permissive') == []
