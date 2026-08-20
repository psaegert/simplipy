"""A rules-less engine says so (ruling 2026-08-18, evening batch 2 item 2):
"Warning on engine without rules" -- broadened from the missing-FILE case to the
resulting STATE, and explicitly NON-FATAL.

The failure this catches is silent success. A config that names a ruleset it
cannot resolve used to warn about the file and then hand back a fully functional
engine with zero rules: `simplify` still ran, still returned canonical output,
and simply never rewrote anything. Every downstream number (a benchmark, a mine,
a corpus) then looks merely disappointing rather than broken.

So the warning is keyed on the STATE, not on the cause -- an engine that ends up
with no rules says so however it got there -- and when the cause WAS an
unresolvable configured path it names both the literal config value and the
absolute path that was actually looked for, because "rules.json" alone never told
anyone where the engine looked.

SCOPE (deliberate): this fires on the CONFIG-DRIVEN path, `from_config` and the
`load` that funnels into it. Direct construction with `rules=[]` is the sanctioned
bare-engine idiom -- the caller typed the empty list -- and stays silent; see
`test_explicit_bare_construction_stays_silent`.
"""
import json
import os
import warnings

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.io import load_config
from conftest import acj_config_path, require_or_skip


@pytest.fixture(scope="module")
def operators() -> dict:
    require_or_skip(acj_config_path(), 'the acj-4-3 asset is not staged')
    return load_config(acj_config_path())['operators']


def _write_config(directory, operators: dict, rules_value=None) -> str:
    """A minimal, VALID config -- optionally naming a rules file."""
    config: dict = {'operators': operators}
    if rules_value is not None:
        config['rules'] = rules_value
    path = os.path.join(str(directory), 'config.yaml')
    with open(path, 'w') as handle:
        yaml.safe_dump(config, handle)
    return path


class TestTheWarningFires:
    def test_unresolvable_rules_path_warns(self, tmp_path, operators: dict) -> None:
        config_path = _write_config(tmp_path, operators, rules_value='./missing.json')
        with pytest.warns(UserWarning) as record:
            SimpliPyEngine.from_config(config_path)
        assert len(record) >= 1

    def test_the_warning_names_the_state_not_only_the_file(
            self, tmp_path, operators: dict) -> None:
        config_path = _write_config(tmp_path, operators, rules_value='./missing.json')
        with pytest.warns(UserWarning) as record:
            SimpliPyEngine.from_config(config_path)
        message = ' '.join(str(w.message) for w in record)
        assert 'no simplification rules' in message.lower()

    def test_the_warning_names_both_the_literal_and_the_resolved_path(
            self, tmp_path, operators: dict) -> None:
        """'./missing.json' says what was configured; the absolute path says where
        the engine actually looked. A user needs both to fix it."""
        config_path = _write_config(tmp_path, operators, rules_value='./missing.json')
        with pytest.warns(UserWarning) as record:
            SimpliPyEngine.from_config(config_path)
        message = ' '.join(str(w.message) for w in record)
        assert './missing.json' in message, 'the literal config value is missing'
        resolved = os.path.join(str(tmp_path), 'missing.json')
        assert resolved in message, 'the resolved absolute path is missing'

    def test_a_config_with_no_rules_key_warns(self, tmp_path, operators: dict) -> None:
        """The broadening: no file was named, so the old warning could not fire --
        but the STATE is the same."""
        config_path = _write_config(tmp_path, operators)
        with pytest.warns(UserWarning, match='(?i)no simplification rules'):
            SimpliPyEngine.from_config(config_path)

    def test_an_empty_rules_file_warns(self, tmp_path, operators: dict) -> None:
        """The file resolved and loaded perfectly, and still there are no rules."""
        with open(os.path.join(str(tmp_path), 'rules.json'), 'w') as handle:
            json.dump([], handle)
        config_path = _write_config(tmp_path, operators, rules_value='./rules.json')
        with pytest.warns(UserWarning, match='(?i)no simplification rules'):
            SimpliPyEngine.from_config(config_path)


class TestTheWarningIsNonFatal:
    def test_the_engine_is_returned_and_usable(
            self, tmp_path, operators: dict) -> None:
        config_path = _write_config(tmp_path, operators, rules_value='./missing.json')
        with pytest.warns(UserWarning):
            engine = SimpliPyEngine.from_config(config_path)
        assert isinstance(engine, SimpliPyEngine)
        assert engine.simplification_rules == []
        # It still simplifies -- canonical construction is representation, not rules.
        assert engine.simplify(['+', 'x0', 'x0']) == ['*', '2', 'x0']


class TestTheWarningStaysQuietWhenItShould:
    def test_a_real_ruleset_does_not_warn(self) -> None:
        """The rules-less warning must stay silent for a real ruleset, and so must
        every other warning.

        The temporary measure-fingerprint allowance is GONE. It existed because the
        shipped acj-4-3 was mined under the previous measure and tripped D25/R6, and its
        own instruction was to delete it once the re-mine landed. The re-mine landed, so
        the filter now has nothing to excuse -- and while it stayed, it would have
        silently swallowed a genuine fingerprint mismatch on the new artifact.
        """
        require_or_skip(acj_config_path(), 'the acj-4-3 asset is not staged')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', UserWarning)
            engine = SimpliPyEngine.from_config(acj_config_path())
        assert len(engine.simplification_rules) > 0
        assert not [w for w in caught if 'NO simplification rules' in str(w.message)], \
            'a real ruleset must never raise the rules-less warning'
        unexpected = [str(w.message) for w in caught]
        assert unexpected == [], f'unexpected warnings on a real ruleset: {unexpected}'

    def test_explicit_bare_construction_stays_silent(self, operators: dict) -> None:
        """`rules=[]` is the sanctioned bare-engine idiom (the test suite, the
        miner, and pickling all rely on it). The caller asked for it in as many
        words, so warning would be crying wolf -- the ruling targets the engine
        that ends up rules-less by SURPRISE."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            SimpliPyEngine(operators=operators, rules=[])
