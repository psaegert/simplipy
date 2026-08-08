"""Orientation closure of the ruleset (rules fire modulo the sign orientation class).

A rule ``L -> T`` entails ``-L -> -T`` (negate both sides), but a sum and its negation
are two DIFFERENT canonical states: negating a sum redistributes the sign into the term
coefficients, so the original Add never survives as a subtree and the rule never meets
the flipped subject. ``1 - 2sin^2(x) -> cos(2x)`` used to fire while ``2sin^2(x) - 1``
(which equals ``-cos(2x)``) stayed unrewritten.

The fix is ORIENTATION CLOSURE OF THE RULESET at translate time: for every admitted rule
whose negated LHS canonically ABSORBS the sign (a distributed sum, or a product
coefficient other than -1), the negated twin is minted through the exact gates every
loaded rule passes (Knuth-Bendix orientation, wildcard subset, dedupe against the raw
set). With both orientations loaded, matching is complete modulo the orientation class
as a property of the DATA: any subject either matches L or its negation does. Twins
whose flipped LHS keeps an explicit -1 factor are NOT minted -- there the original LHS
survives as a subtree (Fun/Pow roots, kept-primitive sums) or the sub-multiset remainder
absorbs the -1 (coefficient-free products), so the source rule already fires.
"""
import json
import os
import tempfile

import yaml

import simplipy as sp
from simplipy import SimpliPyEngine

OPS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "-": {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "pow": {"realization": "simplipy.operators.pow", "alias": ["power"], "inverse": None, "arity": 2, "precedence": 3, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 3, "commutative": False},
    "cos": {"realization": "np.cos", "alias": [], "inverse": "acos", "arity": 1, "precedence": 3, "commutative": False},
}

DOUBLE_ANGLE = [[["-", "1", "*", "2", "pow", "sin", "?0", "2"], ["cos", "*", "2", "?0"]]]


def make_engine(rules: list) -> SimpliPyEngine:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "rules.json"), "w") as fh:
        json.dump(rules, fh)
    cfg = os.path.join(d, "config.yaml")
    with open(cfg, "w") as fh:
        fh.write(yaml.safe_dump({"operators": OPS, "rules": "rules.json"}))
    return SimpliPyEngine.from_config(cfg)


class TestSumOrientationTwins:
    def test_flipped_subject_fires(self) -> None:
        e = make_engine(DOUBLE_ANGLE)
        # the rule's own orientation (sanity)
        assert e.simplify(["-", "1", "*", "2", "pow", "sin", "x0", "2"], form="explicit") == \
            ["cos", "*", "2", "x0"]
        # the FLIPPED subject: 2sin^2(x) - 1 == -cos(2x)
        out = e.simplify(["-", "*", "2", "pow", "sin", "x0", "2", "1"], form="explicit")
        assert out == ["neg", "cos", "*", "2", "x0"], out
        # idempotent
        assert e.simplify(list(out), form="explicit") == out

    def test_flipped_subject_fires_with_remainder(self) -> None:
        e = make_engine(DOUBLE_ANGLE)
        got = e.simplify(["+", "x1", "-", "*", "2", "pow", "sin", "x0", "2", "1"], form="tagged")
        want = e.simplify(["+", "x1", "neg", "cos", "*", "2", "x0"], form="tagged")
        assert got == want, (got, want)

    def test_twin_accounting_and_dedupe(self) -> None:
        e = make_engine(DOUBLE_ANGLE)
        kept, subsumed, dropped, twins = e._core.ac_rules_info()
        assert (kept, twins) == (2, 1), (kept, subsumed, dropped, twins)
        # loading BOTH orientations explicitly dedupes to the same serving set
        both = DOUBLE_ANGLE + [
            [["-", "*", "2", "pow", "sin", "?0", "2", "1"], ["neg", "cos", "*", "2", "?0"]],
        ]
        e2 = make_engine(both)
        kept2, _, _, twins2 = e2._core.ac_rules_info()
        assert (kept2, twins2) == (2, 0), (kept2, twins2)
        # and both engines rewrite both orientations identically
        for s in (["-", "1", "*", "2", "pow", "sin", "x0", "2"],
                  ["-", "*", "2", "pow", "sin", "x0", "2", "1"]):
            assert e.simplify(list(s), form="explicit") == e2.simplify(list(s), form="explicit")

    def test_function_rooted_rules_mint_no_twin(self) -> None:
        # -f(x) keeps f(x) as an intact subtree, so the source rule fires on the child;
        # a twin would be redundant and none is minted. (The rule must genuinely orient:
        # sign-free complexity makes plain neg-shuffles like `sin neg ?0 -> neg sin ?0`
        # ties that drop at the Knuth-Bendix gate.)
        e = make_engine([[["sin", "+", "?0", "np.pi"], ["neg", "sin", "?0"]]])
        _, _, _, twins = e._core.ac_rules_info()
        assert twins == 0
        # the source rule fires on the intact child of the negated subject
        assert e.simplify(["neg", "sin", "+", "x0", "np.pi"], form="explicit") == ["sin", "x0"]


class TestShippedArtifactTwin:
    def test_pi_minus_acos_flip(self) -> None:
        """The one live orientation gap in acj-4-3: pi - acos(x) -> acos(-x) never had its
        twin mined (the flipped target exceeds the tier's token budget); the load-time
        closure supplies it: acos(x) - pi -> -acos(-x)."""
        e = SimpliPyEngine.from_config(sp.get_path("acj-4-3"))
        got = e.simplify(["-", "acos", "x0", "np.pi"], form="explicit")
        want = e.simplify(["neg", "acos", "neg", "x0"], form="explicit")
        assert got == want, (got, want)
