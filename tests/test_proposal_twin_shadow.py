"""Rules that join a mine through the proposal channel or through sort promotion are decided where
they join, by the test a length's picks pass: the twin-shadow removal. `sin(x)/cos(-x) -> tan(x)`
and `sin(-x)/cos(x) -> -tan(x)` are each other's orientation twin, so of the two only one row
ships (the other's rewrite is the shipped row's twin), and the final gate, which only checks,
finds no twin-shadow owner in the file."""
import json
import os

import yaml

from simplipy import SimpliPyEngine
from simplipy.mining import _rule_gates

_OPS = {
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "inv": {"realization": "simplipy.operators.inv", "alias": [], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 3, "commutative": False},
    "cos": {"realization": "np.cos", "alias": [], "inverse": "acos", "arity": 1, "precedence": 3, "commutative": False},
    "tan": {"realization": "np.tan", "alias": [], "inverse": "atan", "arity": 1, "precedence": 3, "commutative": False},
}

#: The proposals the 36-operator kit's channel certified, which met as twins after promotion.
PROPOSALS = [{"source": ["*", "sin", "x0", "inv", "cos", "x0"], "target": ["tan", "x0"]},
             {"source": ["/", "sin", "neg", "x0", "cos", "x0"], "target": ["neg", "tan", "x0"]},
             {"source": ["/", "sin", "x0", "cos", "neg", "x0"], "target": ["tan", "x0"]}]


def test_twins_from_the_proposal_channel_and_promotion_ship_once(tmp_path) -> None:
    (tmp_path / "rules.json").write_text("[]")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"operators": _OPS, "rules": "rules.json"}))
    engine = SimpliPyEngine.from_config(str(cfg))
    out = str(tmp_path / "mined.json")
    # finalization raises on a twin-shadow owner in the file; it must not
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=3,
                      dummy_variables=1, X=256, seed=7, verbose=False, output_file=out,
                      proposals=PROPOSALS, modes="f64")
    rules = json.load(open(out))
    with open(out + ".provenance.json") as fh:
        provenance = json.load(fh)
    assert _rule_gates(engine, "default", rules)["twin_shadow"] == []
    # the twin pair met, and the row whose rewrite the other's twin serves was dropped where
    # it joined, under its stage
    dropped = [row for rows in provenance.get("twin_shadowed", {}).values() for row in rows]
    assert dropped, provenance.get("twin_shadowed")
    # THE DROP LEDGER: every rule that left the set records its stage and why, so a diff of two
    # mines is explained from their sidecars alone; each length records its folded picks
    ledger = provenance["dropped"]
    assert all(set(e) == {"rule", "stage", "reason"} for e in ledger), ledger
    shadowed = {(tuple(e["rule"][0]), tuple(e["rule"][1])) for e in ledger
                if e["reason"] == "twin-shadowed" and e["stage"].endswith("twin_shadow")}
    for lhs, rhs in dropped:
        assert (tuple(lhs), tuple(rhs)) in shadowed, (lhs, rhs, ledger)
    assert all("folded" in record for record in provenance["per_length"].values())
    tan = [r for r in rules if r[1] in (["tan", "_0"], ["neg", "tan", "_0"], ["tan", "?0"],
                                        ["neg", "tan", "?0"])]
    assert tan, rules
    assert os.path.exists(out)
