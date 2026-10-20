"""Shared test helpers.

`require_or_skip` is the ONE asset gate for python suites: a missing staged asset
skips locally but FAILS under SIMPLIPY_TEST_REQUIRE_ASSETS (CI sets it right after
staging) -- a suite that silently skips in CI reports green while running nothing,
so the only honest failure mode is a loud one (the E4-3 doctrine, mirrored from the
cargo side).

`construct_legacy_table` builds an engine over the GENERATION-1 (retired
hyper-operator) vocabulary from the IN-REPO fixture (`tests/fixtures/
legacy_vocab_config.yaml`, the dev_7-3 operator table verbatim, no rules, no asset
dependence -- audit Tier-1 #3) through RAW construction, the sanctioned in-memory
path. The public artifact-loading boundary (`from_config`/`load`) refuses
generation-1 artifacts outright (owner ruling 2026-08-03, `simplipy.compat`); the
suites that legitimately need the legacy TABLE -- the conversion layer's
input-language tests and the chain-combining caps -- go through here instead, so
the refusal stays total on the public surface and no test downloads a legacy asset.
"""
import os
import tempfile

import pytest

from simplipy import SimpliPyEngine
from simplipy.io import load_config

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'legacy_vocab_config.yaml')


_ACJ_REPO_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj-4')
_ACJ_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'simplipy', 'engines', 'acj-4')


@pytest.fixture(autouse=True, scope='session')
def _temporary_files_stay_in_the_pytest_tree(tmp_path_factory):
    """Every temporary file a test makes -- itself or through library code and spawned
    workers (``tempfile``, ``TMPDIR``) -- lands in pytest's own tmp tree, never in the
    system temp dir, and is cleaned up with it."""
    scratch = str(tmp_path_factory.mktemp('tempfile'))
    old_env, old_tempdir = os.environ.get('TMPDIR'), tempfile.tempdir
    os.environ['TMPDIR'] = scratch
    tempfile.tempdir = scratch
    yield
    tempfile.tempdir = old_tempdir
    if old_env is None:
        os.environ.pop('TMPDIR', None)
    else:
        os.environ['TMPDIR'] = old_env


def acj_asset_dir() -> str:
    """Directory of the acj-4 asset the suites run on: the repo's own mine output
    when present (the dev tree), else the STAGED asset the package actually serves
    (public checkouts, CI, sdists) -- the same local-then-cache order as the rust
    test loader. Post-republish the two are byte-identical, so the suites test the
    same engine either way."""
    return _ACJ_REPO_DIR if os.path.isdir(_ACJ_REPO_DIR) else _ACJ_CACHE_DIR


def acj_config_path() -> str:
    return os.path.join(acj_asset_dir(), 'config.yaml')


def acj_rules_path() -> str:
    return os.path.join(acj_asset_dir(), 'rules.json')


def acj_real_rules_path() -> str:
    return os.path.join(acj_asset_dir(), 'rules_real.json')


def acj_corpus_rules_path() -> str:
    # the published acj-4 artifact predates the 'permissive' rename and ships the
    # file under its historic name; a re-mined cell ships the new one
    for name in ('rules_permissive.json', 'rules_corpus.json'):
        path = os.path.join(acj_asset_dir(), name)
        if os.path.exists(path):
            return path
    return os.path.join(acj_asset_dir(), 'rules_permissive.json')


def require_or_skip(path: str, why: str) -> None:
    if not os.path.exists(path):
        if os.environ.get('SIMPLIPY_TEST_REQUIRE_ASSETS'):
            pytest.fail(f'SIMPLIPY_TEST_REQUIRE_ASSETS is set but {why}')
        pytest.skip(why)


def require_triple(why: str = 'the acj-4 triple is not staged') -> None:
    """All three rule sets of the acj-4 cell, gated like every other staged asset: a
    missing file skips locally and fails under SIMPLIPY_TEST_REQUIRE_ASSETS."""
    for path in (acj_rules_path(), acj_real_rules_path(), acj_corpus_rules_path()):
        require_or_skip(path, f'{why} ({os.path.basename(path)} is missing)')


# INTERIM PINS (2026-10-18), until the acj-4 re-mine, which resets both lists to empty. The sign
# key (fewest signs, then outermost) and the one-bit sign moved the canon and the measure past the
# staged acj-4 asset, which was mined under the earlier ones, and the f64 load refuses exactly
# these rows of it:
#   * SUBSUMED (gate G2): a sign moved at an odd function, which the canon now places itself, and
#     a product of two powers of e, which the product constructor merges (`exp(2)/e` is `e`),
#     so both sides are one canonical state;
#   * UPHILL (gate G7): two rows whose sides now climb in price. A sign costs one bit wherever it
#     sits, so a negation standing alone pays no `Mul` head: `-exp(-x)` is cheaper than
#     `-1/exp(x)` and `-log(acos(x))` than `log(1/acos(x))`, where the two sides cost the same
#     before. G7 refuses only a rule whose sides climb; the rules whose two spellings cost the same
#     and which the sign key orders the other way load, and the runtime gate fires them only where
#     an application descends.
# The suites pin the counts and these rows, so any other drift still fails.
ACJ4_INTERIM_SUBSUMED = (
    ('neg asin _0', 'asin neg _0'), ('neg asinh _0', 'asinh neg _0'),
    ('neg atan _0', 'atan neg _0'), ('neg atanh _0', 'atanh neg _0'),
    ('neg sin _0', 'sin neg _0'), ('neg sinh _0', 'sinh neg _0'),
    ('neg tan _0', 'tan neg _0'), ('neg tanh _0', 'tanh neg _0'),
    ('inv neg asin _0', 'inv asin neg _0'), ('inv neg asinh _0', 'inv asinh neg _0'),
    ('inv neg atan _0', 'inv atan neg _0'), ('inv neg atanh _0', 'inv atanh neg _0'),
    ('neg rootn _0 3', 'rootn neg _0 3'), ('neg rootn _0 5', 'rootn neg _0 5'),
    ('neg rootn _0 7', 'rootn neg _0 7'), ('neg rootn _0 9', 'rootn neg _0 9'),
    ('/ exp 2 np.e', 'np.e'),
)
ACJ4_INTERIM_UPHILL = (
    ('neg exp neg !0', 'inv neg exp !0'), ('neg log acos _0', 'log inv acos _0'),
)


def acj4_refused_rows(engine: SimpliPyEngine) -> tuple[list, list]:
    """The rows of `engine`'s f64 set its load refuses, each as `(lhs, rhs)` strings: the rows no
    served rule answers for (`ac_served_rules_in_mode`'s provenance), split into the subsumed
    ones (one canonical state on both sides, in the canon the set was certified in) and the
    rest."""
    core = engine._core
    canon = engine._rule_canons.get('default')
    served = {src for _, _, src in core.ac_served_rules_in_mode(rule_mode='default')}
    subsumed, other = [], []
    for i, (lhs, rhs) in enumerate(engine.simplification_rules):
        if i in served:
            continue
        row = (' '.join(lhs), ' '.join(rhs))
        key_l, key_r = core.ac_canonical_keys([list(lhs), list(rhs)], rule_mode='default',
                                              canon=canon)
        (subsumed if key_l == key_r else other).append(row)
    return subsumed, other


# Known violations of the permissive-dominance sweep over the shipped corpus (rows where the
# permissive answer prices above f64's on f64's yardstick): none.
PERMISSIVE_DOMINANCE_KNOWN: tuple = ()


def construct_legacy_table() -> SimpliPyEngine:
    config = load_config(_FIXTURE)
    return SimpliPyEngine(operators=config['operators'], rules=[])
