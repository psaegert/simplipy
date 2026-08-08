"""One coverage ordering, used by every gate (the F5 ruling).

Coverage is a redundancy claim about STATES, judged in the serve-time reduction
ordering (semantic complexity, literal size, canonical total order) -- the ordering
every fire strictly descends and mint acceptance already judges by (finding F2). The
score projection it replaces was blind exactly on the tie tier (equal-complexity
DIFFERENT states): ``abs neg ?0 -> abs ?0`` scored 2 <= 2 "covered" and was pruned
while nothing else performed the rewrite -- the prune deleted rules the acceptance
gate had certified as strict serve-ordering improvements. In the full ordering the
non-strict comparison is honest, because equality there is state identity: "covered"
means the probe engine literally reaches the promised state or better.

Canon's own folds are the quotient map, not reductions: ``+ x0 0`` IS ``x0`` as a
state. The "engine already handles it" predicates (the strict-Kruskal skip, the
proposal verdicts) therefore read: the walk strictly descends, OR the result is
atomic (nothing mintable sits strictly below a single-token state). A respell --
canon spelling the SAME state differently, ``* exp x0 exp x0`` -> ``pow exp x0 2``
-- is neither, and is searched.
"""
import json
import os

from simplipy import SimpliPyEngine

_PRUNE_OPS = {
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "abs": {"realization": "np.abs", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 3, "commutative": False},
}

_STRICT_OPS = {
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
}

_PROPOSAL_OPS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
}


class TestPruneStateCoverage:
    def test_tie_tier_rule_survives_prune(self) -> None:
        """The E5a witness: a strict serve-ordering descent whose complexity score is a
        TIE (signs are free). The score judge pruned it; the state judge must not --
        nothing else performs the rewrite, so pruning it lengthens serving output.

        WITNESS REPLACED 2026-08-07. It was `abs neg ?0 -> abs ?0` until the parity
        arm (`sign_blind_rep`) made that rewrite CONSTRUCTOR-NATIVE, at which point
        `prune_covered_rules` was right to prune it and the test was asserting a
        premise the engine had outgrown -- "nothing else performs the rewrite" is the
        load-bearing clause, and it stopped being true. The successor is the same KIND
        of fact (sign placement, free under mu) still owned by a rule: the canon pulls
        a sign INTO an odd function, `-sin(x) -> sin(-x)`, at 24,000 either way."""
        rule = (("*", "(-1)", "sin", "?0"), ("sin", "neg", "?0"))
        engine = SimpliPyEngine(operators=_PRUNE_OPS, rules=[rule])
        assert engine.simplify(["*", "(-1)", "sin", "x0"], form="explicit") == ["sin", "neg", "x0"]

        assert engine.prune_covered_rules() == 0
        assert engine.simplification_rules == [rule]
        # and the kept rule still fires on the serving engine after recompile
        assert engine.simplify(["abs", "neg", "x0"], form="explicit") == ["abs", "x0"]

    def test_arithmetic_covered_rule_still_pruned(self) -> None:
        """Positive control: a rule whose promise the probe engine reaches EXACTLY
        (canon owns `* 1 t` natively) is genuine coverage -- equality in the total
        order is state identity -- and must still be pruned."""
        engine = SimpliPyEngine(operators=_PRUNE_OPS,
                                rules=[(("*", "1", "?0"), ("?0",))])
        assert engine.prune_covered_rules() == 1
        assert engine.simplification_rules == []
        assert engine.simplify(["*", "1", "x0"], form="explicit") == ["x0"]


class TestConstAbstractionFulfillment:
    """``<constant>`` in a promise is EXISTENTIAL (the forall/exists Const doctrine):
    a probe that reaches an INSTANCE of the promise -- each ``<constant>`` slot bound
    to some ground subterm -- has fulfilled it, and better (it reached the specific
    constant the abstraction only names). The canonical total order cannot see this
    tier: Const and ground leaves are just different leaf kinds there, and the
    direction between them is semantically arbitrary (``<constant>`` sorts below
    ``0`` but not below ``np.pi``), so the pure ordering judge kept 1,098 abstraction
    rules on the acj-4-3 re-mine whose exact ground result the engine already
    computes natively."""

    def test_bare_constant_promise_fulfilled_by_exact_literal(self) -> None:
        """Probe reaches `0`; the promise was `<constant>`. Reaching the exact
        literal fulfills (exceeds) the abstract promise: covered, pruned."""
        engine = SimpliPyEngine(operators=_PRUNE_OPS,
                                rules=[(("*", "?0", "abs", "0"), ("<constant>",))])
        assert engine.prune_covered_rules() == 1
        assert engine.simplification_rules == []

    def test_compound_constant_promise_fulfilled_by_ground_instance(self) -> None:
        """Probe reaches `* 2.718... x0`; the promise was `* <constant> ?0`. The
        ground coefficient instantiates the Const slot: covered, pruned."""
        engine = SimpliPyEngine(
            operators=_PRUNE_OPS,
            rules=[(("*", "abs", "np.e", "?0"), ("*", "<constant>", "?0"))])
        assert engine.prune_covered_rules() == 1
        assert engine.simplification_rules == []

    def test_constant_collapse_rule_survives(self) -> None:
        """An instance of a `<constant>` slot is a constlike LEAF, never a compound
        variable-free subtree: `acos cos <constant> -> <constant>` genuinely collapses
        a 3-token constant expression to one leaf (a strict serve-ordering descent the
        engine cannot fold itself -- Const is opaque), so the probe's unreduced
        `acos cos <constant>` does NOT fulfill the promise and the rule is kept."""
        ops = dict(_PRUNE_OPS)
        for name, real, inverse in (("cos", "np.cos", "acos"), ("acos", "np.arccos", "cos")):
            ops[name] = {"realization": real, "alias": [], "inverse": inverse,
                         "arity": 1, "precedence": 3, "commutative": False}
        rule = (("acos", "cos", "<constant>"), ("<constant>",))
        engine = SimpliPyEngine(operators=ops, rules=[rule])
        assert engine.prune_covered_rules() == 0
        assert engine.simplification_rules == [rule]

    def test_const_promise_rule_is_dead_weight_under_mu(self) -> None:
        """STAGE 2 flip of the old control ("an unfulfilled Const promise survives").
        Under mu the rule `exp(e*?0) -> pow(<constant>, ?0)` cannot serve at all:
        `<constant>` is the priciest atom (128), so the target (144) sits far ABOVE
        the 32-unit source and translation refuses the non-descending orientation.
        An unservable rule is dead weight, and the coverage prune correctly removes
        it -- there is no "promise" left to keep. (Pre-mu the target was CHEAPER by
        node count and the probe's refusal to reach it was genuine rule work.)"""
        ops = dict(_PRUNE_OPS)
        ops["exp"] = {"realization": "np.exp", "alias": [], "inverse": None,
                      "arity": 1, "precedence": 3, "commutative": False}
        ops["pow"] = {"realization": "simplipy.operators.pow", "alias": [],
                      "inverse": None, "arity": 2, "precedence": 3, "commutative": False}
        rule = (("exp", "*", "np.e", "?0"), ("pow", "<constant>", "?0"))
        engine = SimpliPyEngine(operators=ops, rules=[rule])
        assert engine.prune_covered_rules() == 1
        assert engine.simplification_rules == []


class TestStrictKruskalSearchesRespells:
    def test_respelled_source_is_mined_in_strict_mode(self, tmp_path) -> None:
        """`/ 1 exp ?0` canon-respells to `inv exp ?0` (4 -> 3 tokens, SAME state).
        The old strict-mode skip read the token shrink as "already reduced" and never
        searched, losing `1/exp(t) -> exp(-t)` to the enumeration alphabet (the
        respelled spelling speaks `inv`, which the ladder never enumerates). Under
        the serve ordering a respell is not a reduction: strict mode must search it
        and mint the genuinely-below target `exp neg ?0` (complexity 3 -> 2)."""
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        import yaml
        cfg.write_text(yaml.safe_dump({"operators": _STRICT_OPS, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(cfg))

        engine.find_rules(
            max_source_pattern_length=4, max_target_pattern_length=3,
            dummy_variables=1, extra_internal_terms=["1"], X=256,
            constants_fit_challenges=2, constants_fit_retries=2,
            seed=7, relaxed_kruskal=False, verbose=False)

        assert engine.simplify(["/", "1", "exp", "x0"], form="explicit") == \
            ["exp", "neg", "x0"], engine.simplification_rules


class TestProposalVerdictHonesty:
    def _mine(self, directory, proposals):
        import yaml
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "rules.json"), "w") as fh:
            fh.write("[]")
        cfg = os.path.join(directory, "config.yaml")
        with open(cfg, "w") as fh:
            fh.write(yaml.safe_dump({"operators": _PROPOSAL_OPS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                       dummy_variables=1, extra_internal_terms=["0", "1"],
                       X=256, seed=7, verbose=False, output_file=out,
                       proposals=proposals)
        with open(out + ".provenance.json") as fh:
            return eng, json.load(fh)["proposals"]["outcomes"]

    def test_respell_only_proposal_is_rejected_not_covered(self, tmp_path) -> None:
        """`* exp x0 exp x0` collects to `pow exp x0 2`: same state, shorter spelling.
        The engine does NOT reduce it, and at target length 3 nothing expressible
        beats its state either -- the honest verdict is 'rejected' (search exhausted),
        not 'already_covered' (a coverage claim nothing backs: the wanted rule
        `exp(t)*exp(t) -> exp(t+t)` is genuinely lost at this target budget)."""
        _, outcomes = self._mine(str(tmp_path), [{"source": ["*", "exp", "x0", "exp", "x0"]}])
        assert outcomes == {
            "certified": 0, "already_covered": 0, "rejected": 1, "duplicate": 0}

    def test_canon_owned_proposal_stays_already_covered(self, tmp_path) -> None:
        """Control: `+ x0 0` IS `x0` as a state -- the class collapsed at parse, the
        result is atomic, and no rule could sit below it. That is genuine coverage."""
        _, outcomes = self._mine(str(tmp_path), [{"source": ["+", "x0", "0"]}])
        assert outcomes == {
            "certified": 0, "already_covered": 1, "rejected": 0, "duplicate": 0}
