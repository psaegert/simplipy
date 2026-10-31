"""ONE DISTINCT RULE SET PER MODE -- complete files, not a base plus supplements.

One distinct rule set for each mode, so the artifact is two COMPLETE, self-contained files::

    rules.json              `Mode.f64`        -- every rule the deployed evaluator reproduces
    rules_permissive.json   `Mode.permissive` -- every rule that holds in permissive

Each file is COMPLETE and carries only what its mode can serve: the per-mode prune drops
what that mode's own constructor already performs, so no set identity holds between the
files -- the modes reduce in different canonical worlds, so overlap measures spelling rather
than capability -- and the property that matters is checked behaviourally, by
`assert_permissive_dominates`.

WHY COMPLETE FILES. The provenance hole closed earlier existed PRECISELY because the served
set was COMPUTED rather than STATED: `rules.json` said 5,545 while the engine served 5,553,
and eight load-minted twins went unjudged for exactly as long as that gap stood. An overlay
design (a base plus a tier) recreates that shape by construction. With distinct sets, WHAT
YOU LOAD IS WHAT YOU SERVE, and this suite pins that as behaviour rather than as a comment.

THE ABSOLUTE REQUIREMENT, pinned first below: an artifact whose config names no
`rules_permissive:` (or `rules_corpus:`) behaves exactly as a single-file artifact.
"""
import copy
import json
import os
import pickle

import pytest
import yaml

from conftest import acj_config_path, acj_rules_path, require_or_skip
from simplipy import Mode, SimpliPyEngine
from simplipy.verify import default_corpus
from conftest import legacy_from_config  # noqa: E402

MODES = ('default', 'permissive')

# Four PROBE RULES, each of which the canonical CONSTRUCTORS cannot reach unaided (a sign
# moved into or out of an odd function is the constructors' own decision, so no probe moves
# one). That property is what makes them
# witnesses: which FILE holds a rule is then observable in `simplify` itself, not merely
# in a count. A rule the constructors already realize (`abs asinh exp _0` is one -- `exp`
# is positive, so the `abs` folds) fires in every mode whatever the rule sets say, and
# would make every assertion here pass vacuously. `test_probes_need_a_rule_to_fire` is
# the control that keeps this suite honest.
P_ALL = [['sin', 'acos', '_0'], ['cos', 'asin', '_0']]
P_DEFAULT = [['sinh', 'abs', '_0'], ['abs', 'sinh', '_0']]
P_OTHER = [['tanh', 'asinh', '_0'], ['sin', 'atan', '_0']]
P_CORPUS = [['tanh', 'abs', '_0'], ['abs', 'tanh', '_0']]


def instantiate(rule: list) -> tuple[list[str], list[str]]:
    """The probe rule with its `_0` placeholder bound to `x0`."""
    return ([('x0' if t == '_0' else t) for t in rule[0]],
            [('x0' if t == '_0' else t) for t in rule[1]])


def fires(engine: SimpliPyEngine, rule: list, mode: str) -> bool:
    """Does this mode serve a set containing `rule`?"""
    lhs, rhs = instantiate(rule)
    return engine._core.ac_simplify_in_mode(lhs, 48, 'explicit', 0, rule_mode=mode) == rhs


#: The config keys that name the permissive mode's own rule set. `rules_corpus` is the
#: pre-rename spelling of `rules_permissive` that published artifacts still carry.
MODE_SET_KEYS = ('rules_permissive', 'rules_corpus')


def without_mode_sets(config: dict) -> dict:
    """The config with every mode-set key removed, under either spelling."""
    for key in MODE_SET_KEYS:
        config.pop(key, None)
    return config


def write_artifact(tmp_path, rules, *, corpus=None, declare=()) -> str:
    """A two-file artifact directory over the shipped operator table.

    `declare` names the keys the config declares; a key can be declared with no file
    written behind it, which is the partially-synced-asset-directory case.
    """
    require_or_skip(acj_config_path(), 'acj-4-3 config not staged')
    config = yaml.safe_load(open(acj_config_path()))
    # The shipped config names the permissive set, so copying it and only ever ADDING keys
    # made `declare=()` unbuildable. Strip it first, so `declare` means what the docstring
    # says.
    without_mode_sets(config)
    config['rules'] = 'rules.json'
    json.dump(rules, open(os.path.join(tmp_path, 'rules.json'), 'w'))
    if 'permissive' in declare or corpus is not None:
        config['rules_permissive'] = 'rules_permissive.json'
    if corpus is not None:
        json.dump(corpus, open(os.path.join(tmp_path, 'rules_permissive.json'), 'w'))
    path = os.path.join(tmp_path, 'config.yaml')
    yaml.safe_dump(config, open(path, 'w'), sort_keys=False)
    return path


@pytest.fixture
def pair(tmp_path):
    """The canonical two-file artifact this suite reasons about: one rule shared by both
    files, one exclusive to the default set and two to the permissive one. Every set is
    COMPLETE -- the shared rule is written into both files, not factored out."""
    return write_artifact(
        tmp_path,
        rules=[P_ALL, P_DEFAULT],
        corpus=[P_ALL, P_OTHER, P_CORPUS])


def _default_only_config():
    """A copy of the shipped artifact declaring ONLY `rules:`.

    The no-op path this class pins needs an artifact that names no permissive set, and the
    shipped one names it. Building the fixture here keeps the property under test rather
    than quietly retargeting it."""
    import tempfile
    d = tempfile.mkdtemp()
    cfg = without_mode_sets(yaml.safe_load(open(acj_config_path())))
    cfg['rules'] = 'rules.json'
    json.dump(json.load(open(acj_rules_path())), open(os.path.join(d, 'rules.json'), 'w'))
    yaml.safe_dump(cfg, open(os.path.join(d, 'config.yaml'), 'w'))
    return os.path.join(d, 'config.yaml')


class TestAbsentFilesAreANoOp:
    """THE ABSOLUTE REQUIREMENT. A config naming no permissive set must behave as the
    single-file engine -- so nothing about serving moves."""

    def test_the_shipped_artifact_names_both(self):
        """The shipped acj-4 cell names the f64 and the permissive file, and its directory
        carries exactly those."""
        require_or_skip(acj_config_path(), 'acj-4-3 config not staged')
        base = os.path.dirname(acj_config_path())
        config = yaml.safe_load(open(acj_config_path()))
        # The permissive set is named under exactly one of its two spellings.
        permissive_keys = [k for k in ('rules_permissive', 'rules_corpus') if config.get(k)]
        assert len(permissive_keys) == 1, permissive_keys
        named = {key: config[key] for key in ('rules', *permissive_keys)}
        # THE CONFIG AND THE DIRECTORY MUST AGREE, in both directions. Naming a set it
        # does not ship makes `Mode.permissive` fall back SILENTLY, and shipping a set it
        # does not name makes that set dead weight no loader will ever read.
        for key, declared in named.items():
            assert os.path.isfile(os.path.join(base, declared)), (
                f'{key} names {declared!r}, which the shipped directory does not carry')
        on_disk = {f for f in os.listdir(base) if f.startswith('rules') and f.endswith('.json')
                   and not f.endswith('.provenance.json')}
        assert on_disk == {os.path.basename(declared) for declared in named.values()}, (
            f'rule files on disk {sorted(on_disk)} vs named {named}')

    def test_every_mode_serves_the_default_set(self):
        """No set of its own means the DEFAULT set -- for every mode, counted and
        served. `mode_rules_len` returns None (not 0): "this mode names no set", which
        is a different statement from "this mode names an empty set"."""
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        e = SimpliPyEngine.from_config(_default_only_config())
        assert e.permissive_simplification_rules is None
        assert e._core.mode_rules_len(rule_mode='permissive') is None
        assert e._core.mode_rules_len(rule_mode='default') == len(e.simplification_rules)
        base = e._core.ac_rules_info_in_mode(rule_mode='default')
        for mode in MODES:
            assert e._core.ac_rules_info_in_mode(rule_mode=mode) == base
            assert e._core.ac_served_rules_in_mode(rule_mode=mode) == e._core.ac_served_rules_in_mode(rule_mode='default')

    def test_a_mode_with_no_set_selects_the_default_SET_not_the_default_ANSWER(self):
        """The no-op is about RULE SELECTION, and only that: the CONSTRUCTOR is mode-aware,
        so a mode naming no set of its own serves the default SET and still answers in its
        own reading. `atanh(tanh 30)` is the whole story: f64 gives inf (tanh saturates),
        permissive 30, with zero rules loaded either way."""
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        e = SimpliPyEngine.from_config(_default_only_config())
        assert e._core.mode_rules_len(rule_mode='permissive') is None
        # what IS a no-op: the set the mode selects
        assert e._core.ac_served_rules_in_mode(rule_mode='permissive') == \
            e._core.ac_served_rules_in_mode(rule_mode='default')
        # what is NOT: the answer, because the constructor is mode-aware
        bare = SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))['operators'],
                              rules=[])
        assert bare.simplify(['atanh', 'tanh', '30'], mode='f64') == ['float("inf")']
        assert bare.simplify(['atanh', 'tanh', '30'], mode='permissive') == ['30']

    def test_naming_a_copy_of_the_default_file_changes_no_output(self, tmp_path):
        """The no-op stated the other way round: an artifact that DECLARES the permissive
        key, with a file whose content equals the default set, produces byte-identical
        answers in both modes to the same artifact declaring none. Adding the key is
        therefore inert until the file's CONTENT differs -- there is no per-mode machinery
        cost hiding behind the declaration."""
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        rules = json.load(open(acj_rules_path()))[:600]
        rows = default_corpus()[:80]
        plain = SimpliPyEngine.from_config(write_artifact(tmp_path, rules=rules))
        declared_dir = tmp_path / 'declared'
        declared_dir.mkdir()
        declared = SimpliPyEngine.from_config(write_artifact(
            str(declared_dir), rules=rules, corpus=rules))
        for row in rows:
            for mode in (Mode.f64, Mode.permissive):
                assert declared.simplify(row, mode=mode) == plain.simplify(row, mode=mode)

    def test_a_declared_but_missing_file_is_not_an_error(self, tmp_path):
        """A configured-but-missing optional file WARNS and builds an engine that serves
        that mode the default set. Refusing would make a partially-synced asset
        directory unloadable, for a file whose absence costs nothing; being silent is
        how a broken layout survives to production."""
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], declare=('permissive',))
        with pytest.warns(UserWarning, match="could not be resolved"):
            e = SimpliPyEngine.from_config(cfg)
        assert e.permissive_simplification_rules is None
        for mode in MODES:
            assert fires(e, P_DEFAULT, mode)

    def test_the_warning_names_the_mode_and_both_spellings(self, tmp_path):
        """The declared value says what the config asked for, the absolute path says
        where the engine looked, and the mode says which one lost its set. Neither of
        the three alone is enough to fix a broken artifact layout."""
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], declare=('permissive',))
        with pytest.warns(UserWarning) as record:
            SimpliPyEngine.from_config(cfg)
        msg = '\n'.join(str(w.message) for w in record)
        assert "'permissive'" in msg
        assert 'rules_permissive.json' in msg
        assert str(tmp_path) in msg


class TestEachModeServesExactlyItsOwnFile:
    """The ruling's content, stated as behaviour."""

    def test_probes_need_a_rule_to_fire(self, tmp_path):
        """THE CONTROL for this whole suite: with an EMPTY default set none of the four
        probes fires anywhere, so any firing below is attributable to a rule set and not
        to the canonical constructors."""
        cfg = write_artifact(tmp_path, rules=[])
        with pytest.warns(UserWarning, match='NO simplification rules'):
            e = SimpliPyEngine.from_config(cfg)
        for probe in (P_ALL, P_DEFAULT, P_OTHER, P_CORPUS):
            for mode in MODES:
                assert not fires(e, probe, mode)

    @pytest.mark.parametrize('probe,expected', [
        (P_ALL, {'default', 'permissive'}),
        (P_DEFAULT, {'default'}),
        (P_OTHER, {'permissive'}),
        (P_CORPUS, {'permissive'}),
    ])
    def test_a_rule_fires_in_exactly_the_modes_whose_file_holds_it(
            self, pair, probe, expected):
        e = SimpliPyEngine.from_config(pair)
        assert {m for m in MODES if fires(e, probe, m)} == expected

    def test_a_modes_set_is_exactly_the_file_it_names(self, pair):
        """WHAT IS LOADED IS WHAT IS SERVED. The loaded count, the served count and the
        file's own length agree for every mode -- the gap that let eight twins go
        unjudged cannot open here, because there is nothing to compute in between."""
        e = SimpliPyEngine.from_config(pair)
        d = os.path.dirname(pair)
        on_disk = {
            'default': json.load(open(os.path.join(d, 'rules.json'))),
            'permissive': json.load(open(os.path.join(d, 'rules_permissive.json'))),
        }
        for mode in MODES:
            assert e._core.mode_rules_len(rule_mode=mode) == len(on_disk[mode])
            assert e._core.ac_rules_info_in_mode(rule_mode=mode)[0] == len(on_disk[mode])
            served = e._core.ac_served_rules_in_mode(rule_mode=mode)
            assert len(served) == len(on_disk[mode])
            # ...and provenance indexes THAT MODE'S OWN file, with no tier offsets.
            assert [src for _, _, src in served] == list(range(len(on_disk[mode])))

    def test_the_python_attributes_mirror_the_files(self, pair):
        e = SimpliPyEngine.from_config(pair)
        assert len(e.simplification_rules) == 2
        assert len(e.permissive_simplification_rules) == 3

    def test_public_modes_map_onto_both_sets(self, pair):
        """Each public mode selects its OWN file."""
        e = SimpliPyEngine.from_config(pair)
        d_lhs, d_rhs = instantiate(P_DEFAULT)
        c_lhs, c_rhs = instantiate(P_CORPUS)
        assert e.simplify(d_lhs, mode=Mode.f64) == d_rhs
        assert e.simplify(d_lhs, mode=Mode.permissive) != d_rhs
        assert e.simplify(c_lhs, mode=Mode.permissive) == c_rhs
        assert e.simplify(c_lhs, mode=Mode.f64) != c_rhs

    def test_the_infix_rendering_selects_the_same_set(self, pair):
        """An infix RENDERING is the same answer in another notation, so it must not
        reach a different rule set: the two output branches share one mode lookup.
        Pinned at the core entry and through the convert-then-simplify route."""
        e = SimpliPyEngine.from_config(pair)
        lhs, _ = instantiate(P_CORPUS)
        assert 'tan' in e._core.ac_simplify_infix_in_mode(lhs, 48, 0, rule_mode='permissive')
        assert e._core.ac_simplify_infix_in_mode(lhs, 48, 0, rule_mode='default') != \
            e._core.ac_simplify_infix_in_mode(lhs, 48, 0, rule_mode='permissive')
        # The old `form='infix'` shim is removed in 0.14.0; CONVERTING first is the
        # documented equivalent, and it must still select by mode identically.
        assert e.simplify(e.to_infix(lhs), mode=Mode.permissive) == \
            e._core.ac_simplify_infix_in_mode(lhs, 48, 0, rule_mode='permissive')
        assert e.simplify(e.to_infix(lhs), mode=Mode.f64) == \
            e._core.ac_simplify_infix_in_mode(lhs, 48, 0, rule_mode='default')


class TestAnEmptySetIsNotAnAbsentOne:
    """`None` (no set of its own) and `[]` (an empty set) are DIFFERENT statements, and
    the distinction survives every layer from the config file to the served ruleset.
    Collapsing them -- by keying the fallback on emptiness rather than on presence --
    would make an empty set unsayable."""

    def test_an_empty_file_makes_that_mode_serve_nothing(self, tmp_path):
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], corpus=[])
        e = SimpliPyEngine.from_config(cfg)
        assert e.permissive_simplification_rules == []
        assert e._core.mode_rules_len(rule_mode='permissive') == 0
        assert e._core.ac_rules_info_in_mode(rule_mode='permissive')[0] == 0
        assert not fires(e, P_DEFAULT, 'permissive')
        assert fires(e, P_DEFAULT, 'default')

    def test_retraction_restores_the_default_set(self):
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        e = legacy_from_config(acj_config_path())
        e._core.set_mode_rules('permissive', [])
        assert e._core.mode_rules_len(rule_mode='permissive') == 0
        e._core.set_mode_rules('permissive', None)
        assert e._core.mode_rules_len(rule_mode='permissive') is None
        assert e._core.ac_rules_info_in_mode(rule_mode='permissive') == \
            e._core.ac_rules_info_in_mode(rule_mode='default')


class TestTheSetsSurviveEveryRebuild:
    """A core is derived state, rebuilt on pickle, on `compile_rules` and on every shim
    mutator. A fresh core starts with no set of its own for the permissive mode, so any
    rebuild that forgets to re-push it silently retracts half of the artifact."""

    def test_pickle_carries_both_sets(self, pair):
        e = SimpliPyEngine.from_config(pair)
        back = pickle.loads(pickle.dumps(e))
        assert back.permissive_simplification_rules == e.permissive_simplification_rules
        for probe in (P_ALL, P_DEFAULT, P_OTHER, P_CORPUS):
            for mode in MODES:
                assert fires(back, probe, mode) == fires(e, probe, mode)

    def test_deepcopy_carries_both_sets(self, pair):
        e = SimpliPyEngine.from_config(pair)
        back = copy.deepcopy(e)
        for probe in (P_ALL, P_OTHER, P_CORPUS):
            for mode in MODES:
                assert fires(back, probe, mode) == fires(e, probe, mode)

    def test_compile_rules_does_not_retract_them(self, pair):
        e = SimpliPyEngine.from_config(pair)
        e.compile_rules()
        assert e._core.mode_rules_len(rule_mode='permissive') == 3
        assert fires(e, P_CORPUS, 'permissive')
        assert not fires(e, P_CORPUS, 'default')

    def test_a_pickle_without_the_permissive_set_still_loads(self, pair):
        """Backward compatibility of the RECIPE: state without the attribute rebuilds an
        engine that names no set of its own -- which is what it was."""
        e = SimpliPyEngine.from_config(pair)
        state = e.__getstate__()
        del state['permissive_simplification_rules']
        back = SimpliPyEngine.__new__(SimpliPyEngine)
        back.__setstate__(state)
        assert back.permissive_simplification_rules is None
        assert fires(back, P_DEFAULT, 'permissive')


class TestTheModeSurfaceRefusesNonsense:
    def test_an_unknown_rule_mode_raises(self):
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        e = legacy_from_config(acj_config_path())
        # retired spellings refuse: `lossy`, `sound` (0.14.0) and `real` (0.15)
        for bad in ('lossy', 'sound', 'real', 'Default', ''):
            with pytest.raises(ValueError, match='unknown rule_mode'):
                e._core.ac_simplify_in_mode(['x0'], 48, 'tagged', rule_mode=bad)
            with pytest.raises(ValueError, match='unknown rule_mode'):
                e._core.ac_rules_info_in_mode(rule_mode=bad)
            with pytest.raises(ValueError, match='unknown rule_mode'):
                e._core.mode_rules_len(rule_mode=bad)

    def test_mode_sets_face_the_same_load_gates(self, tmp_path):
        """A mode relaxes what a rule may CLAIM about the deployed engine, never what
        the loader will admit. `sin _0 -> log _0` changes a defined value at a
        constructed exceptional point (licence-registry refusal) and `sin _0 -> exp sin _0`
        climbs in price (G7) -- both are refused in a mode's set exactly as they are in the
        default one."""
        bad = [[['sin', '_0'], ['log', '_0']], [['sin', '_0'], ['exp', 'sin', '_0']]]
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], corpus=bad)
        e = SimpliPyEngine.from_config(cfg)
        assert e._core.mode_rules_len(rule_mode='permissive') == 2
        assert e._core.ac_rules_info_in_mode(rule_mode='permissive')[0] == 0
        assert e._core.ac_simplify_in_mode(['sin', 'x0'], 48, 'explicit', 0, rule_mode='permissive') \
            == ['sin', 'x0']
