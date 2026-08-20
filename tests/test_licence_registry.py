"""The licence registry -- C1.20's dual (owner-approved design, 2026-08-08).

The registry (``rust/engine/refusals.rs``) is the single home of every reified
doctrinal refusal, consulted by EVERY mint channel and by the AC load path:

* NATIVE BATCH (``mine_one_length``): the two relocated syntactic licences run first,
  the pole entry runs last (after the ordering acceptance) -- a refused candidate
  yields to the next one.
* LIBRARY SCAN (``find_rule_lib``): same registry, same order; previously carried
  only the determined-source gate.
* HINT/PROPOSAL (``certify_rules`` / ``find_rules(proposals=...)``): the registry
  gates hint acceptance -- an a.e.-equal pole mirror passes vocabulary, ordering AND
  the numeric confirmation (measure-zero differences are invisible to sampling), so
  before the registry this channel was a full bypass.
* LOAD (``ac_rules``): every raw rule is adjudicated at its constructed exceptional
  points before translate -- a stale or foreign artifact cannot smuggle a
  defined-value change past serving.

The pole entry's spec is contract R2 at constructed exceptional points: a DEFINED
extended-real value (sign of infinity included) may never change at a definable
point; ``Ext`` (a nan hole filled at the point) is R3's licence and passes. The
worked hostile pair throughout is the verified mirror family:
``-(1/(1-w)) -> 1/(w-1)`` -- stable states, a.e.-equal, mu 42,000 -> 26,000 strict
descent, differing exactly at w=1 (-inf vs +inf under one-zero).
"""


import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path

NEG = {"realization": "simplipy.operators.neg", "alias": [], "inverse": None,
       "arity": 1, "precedence": 2.5, "commutative": False}
INV = {"realization": "simplipy.operators.inv", "alias": [], "inverse": None,
       "arity": 1, "precedence": 4, "commutative": False}
SUB = {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1,
       "commutative": False}

HOSTILE_SRC = ['neg', 'inv', '-', '1', 'x0']
HOSTILE_TGT = ['inv', '-', 'x0', '1']


@pytest.fixture(scope='module')
def eng():
    return SimpliPyEngine.from_config(acj_config_path())


def _scratch(tmp_path, rules):
    import json
    import yaml
    (tmp_path / 'rules.json').write_text(json.dumps(rules))
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(
        {'operators': {'neg': NEG, 'inv': INV, '-': SUB}, 'rules': 'rules.json'}))
    return SimpliPyEngine.from_config(str(cfg))


class TestLoadConsumer:
    def test_shipped_artifact_registry_clean(self, eng):
        # The audit pin: no shipped rule changes a defined value at a constructed
        # exceptional point, and the pass costs nothing the translation pins notice.
        # ARTIFACT REFRESH 2026-08-11 (owner-ruled ADOPT, audit F83-adoption): the
        # first post-F83 mine ships, 6,661 -> 6,671 rules (+10, 0 removed, ALL
        # F83-caused by control mine; each judge-OK with zero pointwise
        # exceptional-point disagreements). Translations 6671 -> 6681 (+10 =
        # exactly the adopted rules), subsumed/dropped stay 0, twins stay 10.
        # G2 REPUBLISH (2026-08-15): the full-lane re-mine ships (6,594 rules, x3
        # byte-identical, minted on the reference mint host per D20) -- pristine again: 6602/0/0 + 8
        # twins, the B5+B19 interim discharged.
        # RE-PINNED for the 0.14.0 triple: (5558, 0, 0, 8) -> (5327, 0, 0, 8).
        #
        # WHY, and it is not the dedup. The shipped `rules.json` is now the f64 THIRD of a
        # triple: 5,319 rows, against the 0.13 line's 6,594. Rules that are true over R
        # and NOT f64-realised moved to rules_real.json, and the ground literal
        # evaluations folding now performs were pruned. `kept` counts SERVED patterns, so
        # it follows the file. Translation stays pristine 0/0 and twins stay 8: nothing
        # about the surviving rules changed.
        #
        # The earlier comment here justified the pin with the 2026-08-18 dedup (6602 ->
        # 5553, an artifact of 6,594 rows, complexity 294,730,680). Every one of those
        # numbers is false for the artifact under test.
        kept, subsumed, dropped, twins = eng._core.ac_rules_info()
        assert eng._core.ac_registry_dropped() == 0
        # 5553 -> 5558 at the INVERSE-PAIR BAND (2026-08-19): five rules the
        # constructor used to fold away now survive translation, because the band
        # declines the fold on an unbounded argument. Nothing is subsumed or dropped;
        # the artifact is untouched at 6,594 rows.
        # RE-PINNED for the 0.14.0 triple (the 0.13 artifact served 5,558).
        assert (kept, subsumed, dropped, twins) == (5327, 0, 0, 8)

    def test_poisoned_artifact_dropped_and_inert(self, tmp_path):
        # END-TO-END Part-4 falsifier: an artifact carrying the mirror rule loads
        # with the rule REFUSED (counted), and serving is exactly the rules-free
        # engine's -- the pole classes stay apart.
        hostile = [[['neg', 'inv', '-', '1', '_0'], ['inv', '-', '_0', '1']]]
        poisoned = _scratch(tmp_path, hostile)
        out = poisoned.simplify(poisoned.to_tagged(list(HOSTILE_SRC)))
        assert poisoned._core.ac_registry_dropped() == 1, \
            'the load consumer must refuse the pole-mirror rule'
        # the source state survives as itself: the mirror never fires
        assert 'neg' in out or out[0] == 'neg' or '<mul>' in out, out
        rt = poisoned.simplify(list(out))
        assert rt == out, 'refused rule must leave a stable state'

    def test_clean_scratch_artifact_loads_without_drops(self, tmp_path):
        e = _scratch(tmp_path, [])
        e.simplify(HOSTILE_SRC)
        assert e._core.ac_registry_dropped() == 0


class TestRegistryApi:
    def test_pole_mirror_refused(self, eng):
        # Both spellings of the variable frame: mine dummies (x0) and artifact
        # wildcards (_0) -- the API unions wildcard-shaped tokens into var_names.
        assert eng._core.registry_mint_refusal(
            HOSTILE_SRC, HOSTILE_SRC, HOSTILE_TGT, ['x0']) == 'pole_class_change'
        assert eng._core.registry_mint_refusal(
            ['neg', 'inv', '-', '1', '_0'], ['neg', 'inv', '-', '1', '_0'],
            ['inv', '-', '_0', '1'], []) == 'pole_class_change'

    def test_incident_three_shape_refused(self, eng):
        # `pow acos (-1) <constant> -> <constant>` -- the shipped incident-3 family,
        # spelling-independent through the endpoint side of the count.
        assert eng._core.registry_mint_refusal(
            ['pow', 'acos', '(-1)', '<constant>'],
            ['pow', 'np.pi', '<constant>'],
            ['<constant>'], []) == 'special_absorption'

    def test_determined_const_refused(self, eng):
        assert eng._core.registry_mint_refusal(
            ['+', 'np.pi', '1'], ['+', 'np.pi', '1'],
            ['<constant>'], []) == 'const_introduction'

    def test_sound_and_ext_pass(self, eng):
        # Sound pole agreement and R3's point-level licence must NOT be refused.
        assert eng._core.registry_mint_refusal(
            ['inv', 'inv', 'x0'], ['x0'], ['x0'], ['x0']) is None
        assert eng._core.registry_mint_refusal(
            ['/', 'x0', 'x0'], ['/', 'x0', 'x0'], ['1'], ['x0']) is None


class TestMintChannels:
    def _mine(self, tmp_path, proposals=None):
        import yaml
        (tmp_path / 'rules.json').write_text('[]')
        cfg = tmp_path / 'config.yaml'
        cfg.write_text(yaml.safe_dump(
            {'operators': {'neg': NEG, 'inv': INV, '-': SUB}, 'rules': 'rules.json'}))
        engine = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / 'mined.json')
        engine.find_rules(
            max_source_pattern_length=5, max_target_pattern_length=4,
            dummy_variables=1, extra_internal_terms=['1'], X=256,
            constants_fit_challenges=2, constants_fit_retries=2,
            seed=7, verbose=False, output_file=out, proposals=proposals,
            promote_sorts=False)
        return engine

    def test_native_mine_at_raised_tiers_never_mints_the_mirror(self, tmp_path):
        # THE (5,4) test-mine of the build protocol: the tier where the mirror
        # family first fits (source 5 incl. the outer neg, target 4). The micro
        # universe contains the hostile source and the library enumerates the
        # mirror target; without the registry the pair is mu-descending,
        # a.e.-confirmed and otherwise gate-invisible.
        # The counter is process-global and other tests in this file increment it
        # (the registry API tests); assert the DELTA across THIS mine, or the
        # assertion passes vacuously off their residue.
        before = SimpliPyEngine.from_config(acj_config_path())._core.registry_pole_refusals()
        engine = self._mine(tmp_path)
        rules = [(list(lhs_), list(rhs_)) for lhs_, rhs_ in engine.simplification_rules]
        for lhs, rhs in rules:
            assert not eng_pair_is_mirror(lhs, rhs), (lhs, rhs)
        # The gate FIRED (the mirror candidate was enumerated, ordering-accepted
        # and refused) -- absence of the rule alone would also be consistent with
        # the source never being judged; the counter delta closes that gap.
        assert engine._core.registry_pole_refusals() > before, \
            'the pole entry must actually fire on the (5,4) micro-universe'

    def test_hint_channel_refuses_the_mirror(self, tmp_path):
        # The proposal/hint channel -- previously a full registry bypass.
        proposals = [{'source': ['neg', 'inv', '-', '1', 'x0'],
                      'target': ['inv', '-', 'x0', '1']}]
        engine = self._mine(tmp_path, proposals=proposals)
        rules = [(list(lhs_), list(rhs_)) for lhs_, rhs_ in engine.simplification_rules]
        for lhs, rhs in rules:
            assert not eng_pair_is_mirror(lhs, rhs), (lhs, rhs)


def eng_pair_is_mirror(lhs, rhs):
    """The mirror family in either variable spelling: -(1/(1-w)) -> 1/(w-1)."""
    def norm(toks):
        return [('W' if (t.startswith(('_', '!', '?', 'x')) and t[1:].isdigit() and len(t) > 1)
                 else t) for t in toks]
    return (norm(lhs) == ['neg', 'inv', '-', '1', 'W']
            and norm(rhs) == ['inv', '-', 'W', '1'])
