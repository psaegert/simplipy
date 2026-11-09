"""The judge's f64 lane reads one matrix of rows, the shipped artifacts' mine's (seed 42, 1,024 rows
over two variables, confirmed), whoever asks: a verdict is the rule's. A mine of other rows judges
its picks as `verify_rule` judges them; its own rows only prefilter."""
import numpy as np

from simplipy.verify import _contract, verify_rule


def test_the_judge_reads_one_matrix() -> None:
    rows = _contract.judge_rows()
    assert rows.shape == (3 * 1024, 2)
    assert np.array_equal(rows, _contract.mine_matrix(*_contract.DEFAULT_ROWS))
    assert not rows.flags.writeable
    assert not hasattr(_contract, 'set_rows')


def test_a_mine_of_other_rows_leaves_the_judges_rows_as_they_are(tmp_path) -> None:
    import yaml

    from simplipy import SimpliPyEngine
    ops = {
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
        "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
        "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    }
    (tmp_path / "rules.json").write_text("[]")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"operators": ops, "rules": "rules.json"}))
    engine = SimpliPyEngine.from_config(str(tmp_path / "config.yaml"))
    before = _contract.judge_rows().copy()
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=2, dummy_variables=1,
                      X=64, seed=3, verbose=False, output_file=str(tmp_path / "mined.json"),
                      modes="f64")
    assert np.array_equal(_contract.judge_rows(), before)
    # the two log-exp identities the reduced mine's own rows refused at one rounding, held alike
    for lhs, rhs in (('asin sinh asinh x0', 'asin x0'), ('log exp acos x0', 'acos x0')):
        assert verify_rule(lhs.split(), rhs.split())['tier'] == 'core'
