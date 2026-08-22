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

import pytest

from simplipy import SimpliPyEngine
from simplipy.io import load_config

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'legacy_vocab_config.yaml')


_ACJ_REPO_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj-4-3')
_ACJ_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'simplipy', 'engines', 'acj-4-3')


def acj_asset_dir() -> str:
    """Directory of the acj-4-3 asset the suites run on: the repo's own mine output
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
    return os.path.join(acj_asset_dir(), 'rules_corpus.json')


def require_triple_or_skip(why: str = 'the shipped asset carries the f64 set only') -> None:
    """Skip when the acj-4-3 cell ships no `real`/`corpus` set of its own.

    The cell carries the f64 third alone while the other two are re-mined, so a test
    whose subject is a `real` or `corpus` RULE SET has no subject and says so -- locally
    AND in CI. `SIMPLIPY_TEST_REQUIRE_ASSETS` deliberately does NOT harden this gate:
    that switch guards the assets CI actually STAGES (the f64 four and the legacy
    refusal input) against a silently broken staging step, and hardening the triple on
    the same switch turned the documented interim into 11 CI failures (audit U1). The
    gate never goes quietly wrong instead, because both of its failure channels are
    loud on their own terms:

    * a PARTIAL triple always fails, in every environment -- no intended state ships
      one file of the pair, so half a triple is breakage wherever it is observed;
    * `SIMPLIPY_TEST_REQUIRE_TRIPLE` is the release-gate switch for the job that
      validates a build whose artifact IS the triple: there, absence must read as
      failure, never as a skip.

    No switch needs flipping for the tests themselves: the moment `rules_real.json`
    and `rules_corpus.json` land beside the f64 set, every gated test runs everywhere.
    """
    real, corpus = acj_real_rules_path(), acj_corpus_rules_path()
    present = [p for p in (real, corpus) if os.path.exists(p)]
    if len(present) == 2:
        return
    if present:
        pytest.fail(
            f'the acj-4-3 cell carries PART of a triple ({os.path.basename(present[0])} '
            f'without its sibling) -- a triple ships whole or not at all')
    if os.environ.get('SIMPLIPY_TEST_REQUIRE_TRIPLE'):
        pytest.fail(f'SIMPLIPY_TEST_REQUIRE_TRIPLE is set but {why}')
    pytest.skip(why)


def require_or_skip(path: str, why: str) -> None:
    if not os.path.exists(path):
        if os.environ.get('SIMPLIPY_TEST_REQUIRE_ASSETS'):
            pytest.fail(f'SIMPLIPY_TEST_REQUIRE_ASSETS is set but {why}')
        pytest.skip(why)


def construct_legacy_table() -> SimpliPyEngine:
    config = load_config(_FIXTURE)
    return SimpliPyEngine(operators=config['operators'], rules=[])
