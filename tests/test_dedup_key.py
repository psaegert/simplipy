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

import pytest

from simplipy import SimpliPyEngine
from simplipy.verify import default_corpus
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
LOG_A = (('*', '5', 'log', '_0'), ('log', 'pow', '_0', '5'))


def test_ac_equivalent_spellings_collapse(vocab_engine):
    """`* (-1) asin _0` and `* asin _0 (-1)` are two spellings of one pattern."""
    out = deduplicate_rules([ASIN_A, ASIN_B], dummy_variables=[], engine=vocab_engine, rule_mode='default')
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
        assert len(deduplicate_rules([a, b], dummy_variables=[], engine=vocab_engine, rule_mode='default')) == 1, (a, b)


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
    # The specimen is a rule of the asset whose two sides are two states (the audit's own
    # specimen, `-asin(t) -> asin(-t)`, moves a sign the canonical form now places itself).
    loaded = SimpliPyEngine.from_config(acj_config_path())

    # 1. Conversion is PURE: same answer with a full artifact loaded and with none,
    #    and that answer is the input's own spelling -- no rule fired.
    assert loaded.to_prefix(list(LOG_A[0])) == list(LOG_A[0])
    assert vocab_engine.to_prefix(list(LOG_A[0])) == loaded.to_prefix(list(LOG_A[0]))

    # 2. The rule itself is untouched -- it fires where rules are supposed to fire.
    assert loaded.simplify(list(LOG_A[0])) == list(LOG_A[1])

    # 3. The key is bare-context and artifact-independent.
    key = [['<mul>', '5', 'log', '_0', '</mul>']]
    assert loaded._core.ac_canonical_keys([list(LOG_A[0])], rule_mode="default") == key
    assert vocab_engine._core.ac_canonical_keys([list(LOG_A[0])], rule_mode="default") == key

    # 4. ... so the rule survives dedup with two DISTINCT sides, on either engine.
    for engine in (vocab_engine, loaded):
        out = deduplicate_rules([LOG_A], dummy_variables=[], engine=engine, rule_mode='default')
        assert out == [LOG_A]
        assert out[0][0] != out[0][1]


def test_wildcard_sorts_and_indices_stay_distinct(vocab_engine):
    """Wildcards are PATTERN VARIABLES, not values. The four sorts carry different
    certified strength and distinct indices are distinct slots: merging any of them
    would ship a rule stronger than what was certified."""
    sorts = [(('sin', w), ('0',)) for w in ('_0', '?0', '!0', '$0')]
    assert len(deduplicate_rules(sorts, dummy_variables=[], engine=vocab_engine, rule_mode='default')) == 4
    two_slots = [(('+', '_0', '_0'), ('_0',)), (('+', '_0', '_1'), ('_0',))]
    assert len(deduplicate_rules(two_slots, dummy_variables=[], engine=vocab_engine, rule_mode='default')) == 2


def test_shortest_target_still_wins_and_keeps_its_own_pair(vocab_engine):
    """The fold is unchanged apart from the key: the strictly shorter target wins, and
    the winning SOURCE is that target's own source spelling."""
    # (A sign moved at an odd function is no longer a lower target: both spellings are one
    # canonical state.)
    long_ = (('*', '5', 'log', '_0'), ('*', '5', 'log', '_0'))
    short = (('*', 'log', '_0', '5'), ('log', 'pow', '_0', '5'))
    assert deduplicate_rules([long_, short], dummy_variables=[], engine=vocab_engine, rule_mode='default') == [short]
    assert deduplicate_rules([short, long_], dummy_variables=[], engine=vocab_engine, rule_mode='default') == [short]


def test_unparseable_sides_fall_back_to_the_spelling(vocab_engine):
    """A side the AC parser refuses keeps its spelling as the key (the loader drops such
    rules under `unparseable-side`): one malformed entry must not fail a whole ruleset,
    and must not collide with a well-formed rule's key."""
    junk_a = (('sin',), ('0',))            # arity underflow
    junk_b = (('sin', 'sin'), ('0',))      # ditto, different spelling
    good = (('sin', '_0'), ('0',))
    out = deduplicate_rules([junk_a, junk_b, good], dummy_variables=[], engine=vocab_engine, rule_mode='default')
    assert len(out) == 3


def test_variable_remap_still_runs_before_the_key(vocab_engine):
    """Dummy variables are still canonicalised to the `?` sort first -- the key sits on
    top of that remap, it does not replace it."""
    out = deduplicate_rules(
        [(('+', 'x0', 'x1'), ('+', 'x1', 'x0')), (('+', 'x2', 'x3'), ('+', 'x3', 'x2'))],
        dummy_variables=['x0', 'x1', 'x2', 'x3'], engine=vocab_engine, rule_mode='default')
    assert out == [(('+', '?0', '?1'), ('+', '?1', '?0'))]


def _ac_respellings(engine: SimpliPyEngine, rules: list, n: int) -> list:
    """`n` pairs `(rule, respelling)`: the respelling swaps the two operands of the rule's
    root `+`/`*` -- the same pattern under AC, a different token sequence."""
    arity = engine.operator_arity

    def span(tokens: list, i: int) -> int:
        j = i + 1
        for _ in range(arity.get(tokens[i], 0)):
            j = span(tokens, j)
        return j

    out = []
    for lhs, rhs in rules:
        if lhs[0] in ('+', '*'):
            mid = span(lhs, 1)
            swapped = [lhs[0]] + lhs[mid:] + lhs[1:mid]
            if swapped != lhs:
                out.append(([lhs, rhs], [swapped, rhs]))
        if len(out) == n:
            break
    assert len(out) == n
    return out


def test_engine_construction_drops_ac_duplicates_of_the_shipped_artifact():
    """Loading a ruleset that carries AC duplicates serves each pattern ONCE, keeps the
    first spelling, and drops exactly the respellings."""
    require_or_skip(acj_config_path(), 'acj-4-3 asset not staged')
    raw = json.load(open(acj_rules_path()))
    operators = yaml.safe_load(open(acj_config_path()))['operators']
    engine = SimpliPyEngine(operators=operators, rules=raw)
    pairs = _ac_respellings(engine, raw, 40)
    respelled = [new for _, new in pairs]
    # each respelling is an AC duplicate of its rule: one internal form, two spellings
    assert engine._core.ac_canonical_keys([new[0] for new in respelled], rule_mode="default") == \
        engine._core.ac_canonical_keys([old[0] for old, _ in pairs], rule_mode="default")
    dup_engine = SimpliPyEngine(operators=operators, rules=raw + respelled)
    served = {(tuple(lhs), tuple(rhs)) for lhs, rhs in dup_engine.simplification_rules}
    assert len(served) == len(engine.simplification_rules)
    dropped = {(tuple(a), tuple(b)) for a, b in raw + respelled} - served
    assert dropped == {(tuple(a), tuple(b)) for a, b in respelled}
    # No two SERVED rules share an internal form.
    keys = dup_engine._core.ac_canonical_keys([list(lhs) for lhs, _ in served], rule_mode="default")
    assert len(keys) == len(set(map(tuple, keys)))


def test_corpus_behaviour_is_unchanged_by_the_dedup():
    """Behavioural neutrality, observed from outside: on a ruleset carrying AC
    respellings, the corpus output of the deduplicating loader is byte-identical to the
    output of an engine whose rules were force-fed WITHOUT the AC collapse (the core
    keeps first-match-wins, so a duplicate can only ever lose)."""
    require_or_skip(acj_config_path(), 'acj-4-3 asset not staged')
    corpus = default_corpus()
    operators = load_config(acj_config_path())['operators']
    raw = json.load(open(acj_rules_path()))
    probe = SimpliPyEngine(operators=operators, rules=[])
    rules = raw + [new for _, new in _ac_respellings(probe, raw, 40)]
    deduped = SimpliPyEngine(operators=operators, rules=rules)
    undeduped = SimpliPyEngine(operators=operators, rules=[])
    undeduped._replace_rules([(tuple(lhs), tuple(rhs)) for lhs, rhs in rules])
    assert len(undeduped.simplification_rules) == len(rules)
    assert len(undeduped.simplification_rules) > len(deduped.simplification_rules)
    for expr in corpus:
        assert deduped.simplify(expr) == undeduped.simplify(expr)
