"""ONE DISTINCT RULE SET PER MODE -- the triple, not a base plus supplements.

Owner ruling (2026-08-19): "I'd like one distinct rule set for each mode", and "for
provenance, the triple should be pinned. And we will only ever mine and distribute
triples." So the artifact is three COMPLETE, self-contained files::

    rules.json          `Mode.f64`    -- every rule the deployed evaluator reproduces
    rules_real.json     `Mode.real`   -- every rule that is true over R
    rules_corpus.json   `Mode.corpus` -- the permissive superset

Each file is COMPLETE and carries only what its mode can serve: the per-mode prune drops
what that mode's own constructor already performs, so `rules_corpus` is NOT the union of
the other two. That set identity was retired once folding became mode-dependent -- the
three modes reduce in different canonical worlds, so overlap measures spelling rather
than capability -- and the property it stood proxy for is checked behaviourally, by
`assert_corpus_dominates`.

WHY IT IS WORTH 2.9 MB AGAINST 1.0 MB. The provenance hole closed earlier existed
PRECISELY because the served set was COMPUTED rather than STATED: `rules.json` said
5,545 while the engine served 5,553, and eight load-minted twins went unjudged for
exactly as long as that gap stood. An overlay design (a base plus a tier) recreates
that shape by construction. With distinct sets, WHAT YOU LOAD IS WHAT YOU SERVE, and
this suite pins that as behaviour rather than as a comment.

THE ABSOLUTE REQUIREMENT, pinned first below: an artifact whose config names neither
`rules_real:` nor `rules_corpus:` behaves exactly as it always did. Every artifact
that shipped before the ruling is a single file and must keep loading unchanged.
"""
import copy
import json
import os
import pickle

import pytest
import yaml

from conftest import acj_config_path, acj_rules_path, require_or_skip
from simplipy import Mode, SimpliPyEngine

MODES = ('default', 'real', 'corpus')

# Four PROBE RULES taken verbatim from the shipped acj-4-3 artifact, each of which the
# canonical CONSTRUCTORS cannot reach unaided. That property is what makes them
# witnesses: which FILE holds a rule is then observable in `simplify` itself, not merely
# in a count. A rule the constructors already realize (`abs asinh exp _0` is one -- `exp`
# is positive, so the `abs` folds) fires in every mode whatever the rule sets say, and
# would make every assertion here pass vacuously. `test_probes_need_a_rule_to_fire` is
# the control that keeps this suite honest.
P_ALL = [['*', '(-1)', 'asin', '_0'], ['asin', 'neg', '_0']]
P_DEFAULT = [['*', '(-1)', 'asinh', '_0'], ['asinh', 'neg', '_0']]
P_REAL = [['*', '(-1)', 'atan', '_0'], ['atan', 'neg', '_0']]
P_CORPUS = [['*', '(-1)', 'tan', '_0'], ['tan', 'neg', '_0']]


def instantiate(rule: list) -> tuple[list[str], list[str]]:
    """The probe rule with its `_0` placeholder bound to `x0`."""
    return ([('x0' if t == '_0' else t) for t in rule[0]],
            [('x0' if t == '_0' else t) for t in rule[1]])


def fires(engine: SimpliPyEngine, rule: list, mode: str) -> bool:
    """Does this mode serve a set containing `rule`?"""
    lhs, rhs = instantiate(rule)
    return engine._core.ac_simplify_in_mode(lhs, 48, mode, 'explicit') == rhs


def write_artifact(tmp_path, rules, *, real=None, corpus=None, declare=()) -> str:
    """A three-file artifact directory over the shipped operator table.

    `declare` names the keys the config declares; a key can be declared with no file
    written behind it, which is the partially-synced-asset-directory case.
    """
    require_or_skip(acj_config_path(), 'acj-4-3 config not staged')
    config = yaml.safe_load(open(acj_config_path()))
    # The shipped config now NAMES all three, so copying it and only ever ADDING keys made
    # `declare=()` unbuildable -- the "declares neither" arm compared a declared artifact
    # with itself. Strip them first, so `declare` means what the docstring says.
    config.pop('rules_real', None)
    config.pop('rules_corpus', None)
    config['rules'] = 'rules.json'
    json.dump(rules, open(os.path.join(tmp_path, 'rules.json'), 'w'))
    for name, content in (('real', real), ('corpus', corpus)):
        if name in declare or content is not None:
            config[f'rules_{name}'] = f'rules_{name}.json'
        if content is not None:
            json.dump(content, open(os.path.join(tmp_path, f'rules_{name}.json'), 'w'))
    path = os.path.join(tmp_path, 'config.yaml')
    yaml.safe_dump(config, open(path, 'w'), sort_keys=False)
    return path


@pytest.fixture
def triple(tmp_path):
    """The canonical three-file artifact this suite reasons about: one rule shared by
    all three files, one exclusive to each of `default`, `real` and `corpus`. Every set
    is COMPLETE -- the shared rule is written into all three files, not factored out."""
    return write_artifact(
        tmp_path,
        rules=[P_ALL, P_DEFAULT],
        real=[P_ALL, P_REAL],
        corpus=[P_ALL, P_REAL, P_CORPUS])


def _default_only_config():
    """A copy of the shipped artifact declaring ONLY `rules:`.

    The no-op path this class pins needs an artifact that names neither optional key, and
    the shipped one now names both. Building the fixture here keeps the property under
    test rather than quietly retargeting it at a triple."""
    import tempfile
    d = tempfile.mkdtemp()
    cfg = yaml.safe_load(open(acj_config_path()))
    cfg.pop('rules_real', None)
    cfg.pop('rules_corpus', None)
    cfg['rules'] = 'rules.json'
    json.dump(json.load(open(acj_rules_path())), open(os.path.join(d, 'rules.json'), 'w'))
    yaml.safe_dump(cfg, open(os.path.join(d, 'config.yaml'), 'w'))
    return os.path.join(d, 'config.yaml')


class TestAbsentFilesAreANoOp:
    """THE ABSOLUTE REQUIREMENT. A config naming neither optional key must behave
    byte-identically to the engine that shipped -- so the shipped single-file artifacts
    keep loading unchanged and nothing about serving moves."""

    def test_the_shipped_artifact_now_names_all_three(self):
        """This class pins the ABSENT-file no-op, which was the live path while the
        shipped artifact predated the triple. From 0.14.0 it ships one, so the no-op is
        exercised by the fixtures below instead -- and the shipped config must name all
        three, or `mode='real'` fails closed for every user."""
        require_or_skip(acj_config_path(), 'acj-4-3 config not staged')
        config = yaml.safe_load(open(acj_config_path()))
        assert config['rules'] == './rules.json'
        assert config['rules_real'] == './rules_real.json'
        assert config['rules_corpus'] == './rules_corpus.json'

    def test_every_mode_serves_the_default_set(self):
        """No set of its own means the DEFAULT set -- for every mode, counted and
        served. `mode_rules_len` returns None (not 0): "this mode names no set", which
        is a different statement from "this mode names an empty set".

        Built from a config that names ONLY `rules:`, because the shipped artifact now
        names all three; the no-op path is what this class exists to pin, so it gets a
        fixture that still exercises it."""
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        import json as _json
        import tempfile as _tempfile
        d = _tempfile.mkdtemp()
        cfg = yaml.safe_load(open(acj_config_path()))
        cfg.pop('rules_real', None)
        cfg.pop('rules_corpus', None)
        cfg['rules'] = 'rules.json'
        _json.dump(_json.load(open(acj_rules_path())), open(os.path.join(d, 'rules.json'), 'w'))
        yaml.safe_dump(cfg, open(os.path.join(d, 'config.yaml'), 'w'))
        e = SimpliPyEngine.from_config(os.path.join(d, 'config.yaml'))
        assert e.real_simplification_rules is None
        assert e.corpus_simplification_rules is None
        assert e._core.mode_rules_len('real') is None
        assert e._core.mode_rules_len('corpus') is None
        assert e._core.mode_rules_len('default') == len(e.simplification_rules)
        base = e._core.ac_rules_info()
        for mode in MODES:
            assert e._core.ac_rules_info_in_mode(mode) == base
            assert e._core.ac_served_rules_in_mode(mode) == e._core.ac_served_rules()

    def test_a_mode_with_no_set_selects_the_default_SET_not_the_default_ANSWER(self):
        """The no-op is about RULE SELECTION, and only that.

        This test used to assert the stronger thing -- that a mode naming no set returns
        the default mode's answer token for token -- and that is now FALSE, because the
        CONSTRUCTOR is mode-aware: `real` skips the f64 saturation bands and never folds a
        transcendental. Measured on an artifact where `real` names no set at all, the
        served sets are identical and the ANSWERS still differ on 4 of 400 corpus rows.
        `atanh(tanh 30)` is the whole story: f64 gives inf, real gives 30, with zero rules
        loaded either way.

        (It stayed green only because the corpus was truncated to the first 150 rows, and
        because it was loading the shipped artifact, where `real` names its own 5,471-rule
        set and the comparison was meaningless in a second way.)"""
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        rows = json.load(open(os.path.join(
            os.path.dirname(__file__), '..', 'benchmarks', 'corpus',
            'raw_skeletons_nv.json')))
        # Built from a config with the optional keys POPPED, because the shipped artifact
        # now names all three and `real` there serves its own 5,471-rule set. Loading it
        # directly made this test assert `real == default` about an artifact where the two
        # legitimately differ -- it stayed green only because the corpus was truncated to
        # the first 150 rows, and rows 210, 370 and 380 diverge.
        e = SimpliPyEngine.from_config(_default_only_config())
        assert e._core.mode_rules_len('real') is None
        # what IS a no-op: the set the mode selects
        assert e._core.ac_served_rules_in_mode('real') == \
            e._core.ac_served_rules_in_mode('default')
        # what is NOT: the answer, because the constructor is mode-aware. Pinned as a
        # bounded, explained divergence rather than asserted away.
        differ = [row for row in rows
                  if e._core.ac_simplify_in_mode(row, 48, 'real', 'tagged')
                  != e._core.ac_simplify_in_mode(row, 48, 'default', 'tagged')]
        assert len(differ) == 4, len(differ)
        bare = SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))['operators'],
                              rules=[])
        bare._core.set_mode_rules('real', [])
        assert bare.simplify(['atanh', 'tanh', '30'], mode='f64') == ['float("inf")']
        assert bare.simplify(['atanh', 'tanh', '30'], mode='real') == ['30']

    def test_naming_a_copy_of_the_default_file_changes_no_output(self, tmp_path):
        """The no-op stated the other way round: an artifact that DECLARES both optional
        keys, with files whose content equals the default set, produces byte-identical
        answers in both public modes to the same artifact declaring neither. Adding the
        keys is therefore inert until the files' CONTENT differs -- there is no
        per-mode machinery cost hiding behind the declaration."""
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        rules = json.load(open(acj_rules_path()))[:600]
        rows = json.load(open(os.path.join(
            os.path.dirname(__file__), '..', 'benchmarks', 'corpus',
            'raw_skeletons_nv.json')))[:80]
        plain = SimpliPyEngine.from_config(write_artifact(tmp_path, rules=rules))
        declared_dir = tmp_path / 'declared'
        declared_dir.mkdir()
        declared = SimpliPyEngine.from_config(write_artifact(
            str(declared_dir), rules=rules, real=rules, corpus=rules))
        for row in rows:
            for mode in (Mode.f64, Mode.corpus):
                assert declared.simplify(row, mode=mode) == plain.simplify(row, mode=mode)

    def test_a_declared_but_missing_file_is_not_an_error(self, tmp_path):
        """A configured-but-missing optional file WARNS and builds an engine that serves
        that mode the default set. Refusing would make a partially-synced asset
        directory unloadable, for a file whose absence costs nothing; being silent is
        how a broken layout survives to production."""
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], declare=('real', 'corpus'))
        with pytest.warns(UserWarning, match="could not be resolved"):
            e = SimpliPyEngine.from_config(cfg)
        assert e.real_simplification_rules is None
        assert e.corpus_simplification_rules is None
        for mode in MODES:
            assert fires(e, P_DEFAULT, mode)

    def test_the_warning_names_the_mode_and_both_spellings(self, tmp_path):
        """The declared value says what the config asked for, the absolute path says
        where the engine looked, and the mode says which rung lost its set. Neither of
        the three alone is enough to fix a broken artifact layout."""
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], declare=('real',))
        with pytest.warns(UserWarning) as record:
            SimpliPyEngine.from_config(cfg)
        msg = '\n'.join(str(w.message) for w in record)
        assert "'real'" in msg
        assert 'rules_real.json' in msg
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
        for probe in (P_ALL, P_DEFAULT, P_REAL, P_CORPUS):
            for mode in MODES:
                assert not fires(e, probe, mode)

    @pytest.mark.parametrize('probe,expected', [
        (P_ALL, {'default', 'real', 'corpus'}),
        (P_DEFAULT, {'default'}),
        (P_REAL, {'real', 'corpus'}),
        (P_CORPUS, {'corpus'}),
    ])
    def test_a_rule_fires_in_exactly_the_modes_whose_file_holds_it(
            self, triple, probe, expected):
        e = SimpliPyEngine.from_config(triple)
        assert {m for m in MODES if fires(e, probe, m)} == expected

    def test_a_modes_set_is_exactly_the_file_it_names(self, triple):
        """WHAT IS LOADED IS WHAT IS SERVED. The loaded count, the served count and the
        file's own length agree for every mode -- the gap that let eight twins go
        unjudged cannot open here, because there is nothing to compute in between."""
        e = SimpliPyEngine.from_config(triple)
        d = os.path.dirname(triple)
        on_disk = {
            'default': json.load(open(os.path.join(d, 'rules.json'))),
            'real': json.load(open(os.path.join(d, 'rules_real.json'))),
            'corpus': json.load(open(os.path.join(d, 'rules_corpus.json'))),
        }
        for mode in MODES:
            assert e._core.mode_rules_len(mode) == len(on_disk[mode])
            assert e._core.ac_rules_info_in_mode(mode)[0] == len(on_disk[mode])
            served = e._core.ac_served_rules_in_mode(mode)
            assert len(served) == len(on_disk[mode])
            # ...and provenance indexes THAT MODE'S OWN file, with no tier offsets.
            assert [src for _, _, src in served] == list(range(len(on_disk[mode])))

    def test_the_python_attributes_mirror_the_files(self, triple):
        e = SimpliPyEngine.from_config(triple)
        assert len(e.simplification_rules) == 2
        assert len(e.real_simplification_rules) == 2
        assert len(e.corpus_simplification_rules) == 3

    def test_public_modes_map_onto_all_three_sets(self, triple):
        """Each public mode selects its OWN file. This pinned the retired two-valued
        model -- "`real` is deliberately unreachable" -- and so never touched
        `Mode.real`, leaving the mode slice unable to catch a `Mode.real` misrouting."""
        e = SimpliPyEngine.from_config(triple)
        d_lhs, d_rhs = instantiate(P_DEFAULT)
        r_lhs, r_rhs = instantiate(P_REAL)
        c_lhs, c_rhs = instantiate(P_CORPUS)
        assert e.simplify(d_lhs, mode=Mode.f64) == d_rhs
        assert e.simplify(d_lhs, mode=Mode.corpus) != d_rhs
        assert e.simplify(r_lhs, mode=Mode.real) == r_rhs
        assert e.simplify(r_lhs, mode=Mode.f64) != r_rhs
        assert e.simplify(c_lhs, mode=Mode.corpus) == c_rhs
        assert e.simplify(c_lhs, mode=Mode.f64) != c_rhs

    def test_the_infix_rendering_selects_the_same_set(self, triple):
        """An infix RENDERING is the same answer in another notation, so it must not
        reach a different rule set: the two output branches share one mode lookup.
        Pinned at the core entry and through the convert-then-simplify route."""
        e = SimpliPyEngine.from_config(triple)
        lhs, _ = instantiate(P_CORPUS)
        assert 'tan' in e._core.ac_simplify_infix_in_mode(lhs, 48, 'corpus')
        assert e._core.ac_simplify_infix_in_mode(lhs, 48, 'default') != \
            e._core.ac_simplify_infix_in_mode(lhs, 48, 'corpus')
        # The old `form='infix'` shim is removed in 0.14.0; CONVERTING first is the
        # documented equivalent, and it must still select by mode identically.
        assert e.simplify(e.to_infix(lhs), mode=Mode.corpus) == \
            e._core.ac_simplify_infix_in_mode(lhs, 48, 'corpus')
        assert e.simplify(e.to_infix(lhs), mode=Mode.f64) == \
            e._core.ac_simplify_infix_in_mode(lhs, 48, 'default')


class TestAnEmptySetIsNotAnAbsentOne:
    """`None` (no set of its own) and `[]` (an empty set) are DIFFERENT statements, and
    the distinction survives every layer from the config file to the served ruleset.
    Collapsing them -- by keying the fallback on emptiness rather than on presence --
    would make an empty set unsayable, which is the same computed-instead-of-stated
    mistake the triple exists to remove."""

    def test_an_empty_file_makes_that_mode_serve_nothing(self, tmp_path):
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], real=[])
        e = SimpliPyEngine.from_config(cfg)
        assert e.real_simplification_rules == []
        assert e._core.mode_rules_len('real') == 0
        assert e._core.ac_rules_info_in_mode('real')[0] == 0
        assert not fires(e, P_DEFAULT, 'real')
        # ...while the modes that name no set still serve the default one.
        assert fires(e, P_DEFAULT, 'default')
        assert fires(e, P_DEFAULT, 'corpus')

    def test_retraction_restores_the_default_set(self):
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        e = SimpliPyEngine.from_config(acj_config_path())
        e._core.set_mode_rules('real', [])
        assert e._core.mode_rules_len('real') == 0
        e._core.set_mode_rules('real', None)
        assert e._core.mode_rules_len('real') is None
        assert e._core.ac_rules_info_in_mode('real') == e._core.ac_rules_info()


class TestTheSetsSurviveEveryRebuild:
    """A core is derived state, rebuilt on pickle, on `compile_rules` and on every shim
    mutator. A fresh core starts with no set of its own for either mode, so any rebuild
    that forgets to re-push them silently retracts two thirds of the artifact."""

    def test_pickle_carries_the_triple(self, triple):
        e = SimpliPyEngine.from_config(triple)
        back = pickle.loads(pickle.dumps(e))
        assert back.real_simplification_rules == e.real_simplification_rules
        assert back.corpus_simplification_rules == e.corpus_simplification_rules
        for probe in (P_ALL, P_DEFAULT, P_REAL, P_CORPUS):
            for mode in MODES:
                assert fires(back, probe, mode) == fires(e, probe, mode)

    def test_deepcopy_carries_the_triple(self, triple):
        e = SimpliPyEngine.from_config(triple)
        back = copy.deepcopy(e)
        for probe in (P_ALL, P_REAL, P_CORPUS):
            for mode in MODES:
                assert fires(back, probe, mode) == fires(e, probe, mode)

    def test_compile_rules_does_not_retract_them(self, triple):
        e = SimpliPyEngine.from_config(triple)
        e.compile_rules()
        assert e._core.mode_rules_len('real') == 2
        assert e._core.mode_rules_len('corpus') == 3
        assert fires(e, P_CORPUS, 'corpus')
        assert not fires(e, P_CORPUS, 'default')

    def test_a_pickle_written_before_the_triple_still_loads(self, triple):
        """Backward compatibility of the RECIPE: state without the two attributes
        rebuilds an engine that names no set of its own -- which is what it was."""
        e = SimpliPyEngine.from_config(triple)
        state = e.__getstate__()
        del state['real_simplification_rules']
        del state['corpus_simplification_rules']
        back = SimpliPyEngine.__new__(SimpliPyEngine)
        back.__setstate__(state)
        assert back.real_simplification_rules is None
        assert back.corpus_simplification_rules is None
        assert fires(back, P_DEFAULT, 'real')


class TestTheModeSurfaceRefusesNonsense:
    def test_an_unknown_rule_mode_raises(self):
        require_or_skip(acj_rules_path(), 'acj-4-3 rules not staged')
        e = SimpliPyEngine.from_config(acj_config_path())
        for bad in ('lossy', 'sound', 'Default', ''):  # retired spellings refuse (0.14.0)
            with pytest.raises(ValueError, match='unknown rule_mode'):
                e._core.ac_simplify_in_mode(['x0'], 48, bad, 'tagged')
            with pytest.raises(ValueError, match='unknown rule_mode'):
                e._core.ac_rules_info_in_mode(bad)
            with pytest.raises(ValueError, match='unknown rule_mode'):
                e._core.mode_rules_len(bad)

    def test_mode_sets_face_the_same_load_gates(self, tmp_path):
        """A mode relaxes what a rule may CLAIM about the deployed engine, never what
        the loader will admit. `sin _0 -> log _0` changes a defined value at a
        constructed exceptional point (licence-registry refusal) and `sin _0 -> tan _0`
        does not descend the reduction ordering -- both are refused in a mode's set
        exactly as they are in the default one."""
        bad = [[['sin', '_0'], ['log', '_0']], [['sin', '_0'], ['tan', '_0']]]
        cfg = write_artifact(tmp_path, rules=[P_DEFAULT], corpus=bad)
        e = SimpliPyEngine.from_config(cfg)
        assert e._core.mode_rules_len('corpus') == 2
        assert e._core.ac_rules_info_in_mode('corpus')[0] == 0
        assert e._core.ac_simplify_in_mode(['sin', 'x0'], 48, 'corpus', 'explicit') \
            == ['sin', 'x0']
