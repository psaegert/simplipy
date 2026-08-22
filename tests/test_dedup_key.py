"""The rule dedup key is the engine's INTERNAL FORM, not the token spelling.

Owner ruling 2026-08-18: "dedup key accepted (instead of comparing spelling, compare
internal form)". Two rules that the loader translates into the SAME pattern are ONE
rule; shipping both costs an artifact row, a match attempt per fire, and -- upstream --
a full mining certification.

These observe from OUTSIDE: through `simplipy.utils.deduplicate_rules`, through
`SimpliPyEngine` construction, and through the served behaviour on the corpus.
"""
import json
import yaml
import os

import pytest

from simplipy import SimpliPyEngine
from simplipy.io import load_config
from simplipy.utils import deduplicate_rules

from conftest import acj_config_path, acj_rules_path, require_or_skip


@pytest.fixture(scope='module')
def vocab_engine() -> SimpliPyEngine:
    """The shipped acj vocabulary with NO rules: dedup needs the operator table (the
    internal form is defined by it), never the ruleset."""
    require_or_skip(acj_config_path(), 'acj-4-3 asset not staged')
    return SimpliPyEngine(operators=load_config(acj_config_path())['operators'], rules=[])


# The specimen the form audit measured: one pattern, two AC spellings, one RHS.
ASIN_A = (('*', '(-1)', 'asin', '_0'), ('asin', 'neg', '_0'))
ASIN_B = (('*', 'asin', '_0', '(-1)'), ('asin', 'neg', '_0'))


def test_ac_equivalent_spellings_collapse(vocab_engine):
    """`* (-1) asin _0` and `* asin _0 (-1)` are two spellings of one pattern."""
    out = deduplicate_rules([ASIN_A, ASIN_B], dummy_variables=[], engine=vocab_engine)
    assert len(out) == 1
    # The FIRST pair survives whole -- source and target are the pair that was mined
    # together, never a synthesized cross of two rules.
    assert out[0] == ASIN_A


def test_permuted_bags_and_chains_collapse(vocab_engine):
    """AC permutation, in bags and through the sugar operators, is one key."""
    for a, b in [
        ((('+', '_0', '_1'), ('_0',)), (('+', '_1', '_0'), ('_0',))),
        ((('*', '_0', 'sin', '_1'), ('_0',)), (('*', 'sin', '_1', '_0'), ('_0',))),
        # `- a b` and `+ a neg b` are the same Add bag with the same inverse section.
        ((('-', '_0', '_1'), ('_0',)), (('+', '_0', 'neg', '_1'), ('_0',))),
        # `/ a b` and `* a inv b` are the same Mul bag with the same inverse section.
        ((('/', '_0', '_1'), ('_0',)), (('*', '_0', 'inv', '_1'), ('_0',))),
    ]:
        assert len(deduplicate_rules([a, b], dummy_variables=[], engine=vocab_engine)) == 1, (a, b)


def test_the_key_is_not_the_conversion_api(vocab_engine):
    """THE WRINKLE the form audit flagged -- and why it no longer bites.

    The audit found `to_prefix`/`to_tagged` running the rewrite pass under the LOADED
    RULESET, so `to_prefix(['*','(-1)','asin','_0'])` returned that rule's own target on
    an acj-4-3 engine and the bare spelling on a rules-less one. A dedup key computed
    through the conversion API would then have keyed a rule on its own RHS while
    deduplicating the very artifact that caused the collapse.

    `feat/pure-conversions` removed the premise: conversions are NOTATION ONLY and fire
    no rules (owner-ruled 2026-08-19, "pure-conversion wins because we don't fire
    rules"). So the circularity cannot arise at all -- it is MOOT, not merely avoided.

    The bare-context key is therefore still correct, and for a stronger reason than the
    one it was introduced with: not "we route around a ruleset-dependent conversion" but
    "no conversion depends on the ruleset in the first place". This test now pins BOTH
    halves, so a regression in either -- conversions firing rules again, or the key
    reaching for them -- fails here.
    """
    require_or_skip(acj_rules_path(), 'acj-4-3 asset not staged')
    loaded = SimpliPyEngine.from_config(acj_config_path())

    # 1. Conversion is PURE: same answer with a full artifact loaded and with none,
    #    and that answer is the input's own spelling -- no rule fired.
    assert loaded.to_prefix(list(ASIN_A[0])) == list(ASIN_A[0])
    assert vocab_engine.to_prefix(list(ASIN_A[0])) == loaded.to_prefix(list(ASIN_A[0]))

    # 2. The rule itself is untouched -- it fires where rules are supposed to fire.
    assert loaded.simplify(list(ASIN_A[0])) == list(ASIN_A[1])

    # 3. The key is bare-context and artifact-independent.
    key = [['neg', 'asin', '_0']]
    assert loaded._core.ac_canonical_keys([list(ASIN_A[0])]) == key
    assert vocab_engine._core.ac_canonical_keys([list(ASIN_A[0])]) == key

    # 4. ... so the rule survives dedup with two DISTINCT sides, on either engine.
    for engine in (vocab_engine, loaded):
        out = deduplicate_rules([ASIN_A], dummy_variables=[], engine=engine)
        assert out == [ASIN_A]
        assert out[0][0] != out[0][1]


def test_wildcard_sorts_and_indices_stay_distinct(vocab_engine):
    """Wildcards are PATTERN VARIABLES, not values. The four sorts carry different
    certified strength and distinct indices are distinct slots: merging any of them
    would ship a rule stronger than what was certified."""
    sorts = [(('sin', w), ('0',)) for w in ('_0', '?0', '!0', '$0')]
    assert len(deduplicate_rules(sorts, dummy_variables=[], engine=vocab_engine)) == 4
    two_slots = [(('+', '_0', '_0'), ('_0',)), (('+', '_0', '_1'), ('_0',))]
    assert len(deduplicate_rules(two_slots, dummy_variables=[], engine=vocab_engine)) == 2


def test_shortest_target_still_wins_and_keeps_its_own_pair(vocab_engine):
    """The fold is unchanged apart from the key: the strictly shorter target wins, and
    the winning SOURCE is that target's own source spelling."""
    long_ = (('*', '(-1)', 'asin', '_0'), ('*', '(-1)', 'asin', '_0'))
    short = (('*', 'asin', '_0', '(-1)'), ('asin', 'neg', '_0'))
    assert deduplicate_rules([long_, short], dummy_variables=[], engine=vocab_engine) == [short]
    assert deduplicate_rules([short, long_], dummy_variables=[], engine=vocab_engine) == [short]


def test_unparseable_sides_fall_back_to_the_spelling(vocab_engine):
    """A side the AC parser refuses keeps its spelling as the key (the loader drops such
    rules under `unparseable-side`): one malformed entry must not fail a whole ruleset,
    and must not collide with a well-formed rule's key."""
    junk_a = (('sin',), ('0',))            # arity underflow
    junk_b = (('sin', 'sin'), ('0',))      # ditto, different spelling
    good = (('sin', '_0'), ('0',))
    out = deduplicate_rules([junk_a, junk_b, good], dummy_variables=[], engine=vocab_engine)
    assert len(out) == 3


def test_variable_remap_still_runs_before_the_key(vocab_engine):
    """Dummy variables are still canonicalised to the `?` sort first -- the key sits on
    top of that remap, it does not replace it."""
    out = deduplicate_rules(
        [(('+', 'x0', 'x1'), ('+', 'x1', 'x0')), (('+', 'x2', 'x3'), ('+', 'x3', 'x2'))],
        dummy_variables=['x0', 'x1', 'x2', 'x3'], engine=vocab_engine)
    assert out == [(('+', '?0', '?1'), ('+', '?1', '?0'))]


def test_engine_construction_drops_ac_duplicates_of_the_shipped_artifact():
    """The shipped artifacts carry AC-duplicate rows. Loading one must serve each
    pattern ONCE -- and the served behaviour over the corpus must not move."""
    require_or_skip(acj_config_path(), 'acj-4-3 asset not staged')
    raw = json.load(open(acj_rules_path()))
    engine = SimpliPyEngine.from_config(acj_config_path())
    served = engine.simplification_rules
    # THE LOADER, on a ruleset that actually carries duplicates. The shipped artifact
    # stopped carrying them when the 0.14.0 mine started deduping, so asserting against
    # it proved a property of the ARTIFACT and nothing about the collapse: `dropped` was
    # empty and every check below held vacuously.
    dup = raw + [raw[0], raw[1]]
    dup_engine = SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))['operators'],
                                rules=dup)
    assert len(dup_engine.simplification_rules) < len(dup), \
        'the loader must collapse AC duplicates'
    assert len(served) <= len(raw)
    # No two SERVED rules share an internal form.
    keys = engine._core.ac_canonical_keys([list(lhs) for lhs, _ in served])
    assert len(keys) == len(set(map(tuple, keys)))
    # Every dropped row would be genuinely dead -- but the 0.14.0 mine emits none, so
    # there is nothing to drop and the set is empty. Asserting it non-empty pinned a
    # property of the OLD artifact, not of the loader.
    dropped = {(tuple(a), tuple(b)) for a, b in raw}
    dropped -= {(tuple(lhs), tuple(rhs)) for lhs, rhs in served}
    assert not dropped or all(
        engine._core.ac_canonical_keys([list(lhs)])[0]
        in {tuple(k) for k in engine._core.ac_canonical_keys([list(x) for x, _ in served])}
        for lhs, _ in list(dropped)[:20])


def test_corpus_behaviour_is_unchanged_by_the_dedup():
    """Behavioural neutrality, observed from outside: the corpus output of the engine
    is byte-identical to the output of an engine whose rules were force-fed WITHOUT the
    AC collapse (the core keeps first-match-wins, so a duplicate can only ever lose)."""
    require_or_skip(acj_config_path(), 'acj-4-3 asset not staged')
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'benchmarks',
                               'corpus', 'raw_skeletons_nv.json')
    require_or_skip(corpus_path, 'nv corpus not present')
    corpus = json.load(open(corpus_path))
    engine = SimpliPyEngine.from_config(acj_config_path())
    raw = [(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(open(acj_rules_path()))]
    undeduped = SimpliPyEngine(
        operators=load_config(acj_config_path())['operators'], rules=[])
    undeduped._replace_rules(raw)
    assert len(undeduped.simplification_rules) == len(raw)
    for expr in corpus:
        assert engine.simplify(expr) == undeduped.simplify(expr)
