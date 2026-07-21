import hashlib
import json

import pytest
import yaml

from simplipy.__main__ import main

# Minimal operator set for the find-rules proposal-channel smoke test (kept tiny so the
# L<=3 mine completes in well under a second).
_SMOKE_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "pow2": {"realization": "simplipy.operators.pow2", "alias": [], "inverse": "sqrt", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
}


class TestCLI:
    """Tests for the CLI entry point."""

    def test_help_exits_zero(self) -> None:
        """--help prints usage and exits with code 0."""
        with pytest.raises(SystemExit, match="0"):
            main(["--help"])

    def test_list_runs_without_error(self, capsys) -> None:
        """'list' subcommand runs without crashing."""
        main(["list"])
        captured = capsys.readouterr()
        # Should print at least one section header
        assert len(captured.out) > 0

    def test_find_rules_missing_engine_exits(self) -> None:
        """find-rules with a nonexistent engine exits with code 1."""
        with pytest.raises(SystemExit, match="1"):
            main(["find-rules", "-e", "nonexistent_engine_xyz",
                  "-c", "dummy.yaml", "-o", "out.json"])

    def test_resolve_rules_missing_engine_exits(self) -> None:
        """resolve-rules with a nonexistent engine exits with code 1."""
        with pytest.raises(SystemExit, match="1"):
            main(["resolve-rules", "-e", "nonexistent_engine_xyz",
                  "-o", "out.json"])

    def test_remove_unknown_asset_exits_cleanly(self, capsys) -> None:
        """remove of an unknown asset exits 1 with a clean message (not a traceback), and
        passes the NAME (regression: it previously passed the --type default 'engine')."""
        with pytest.raises(SystemExit, match="1"):
            main(["remove", "__nonexistent_asset_xyz__"])
        captured = capsys.readouterr()
        # the error names the asset the user asked for, not the removed --type flag's 'engine'
        assert "__nonexistent_asset_xyz__" in (captured.out + captured.err)

    def test_install_has_no_type_flag(self) -> None:
        """install takes only a name (+ --force); the vestigial --type flag is gone."""
        with pytest.raises(SystemExit):
            main(["install", "some_asset", "--type", "engine"])  # --type is no longer accepted

    def test_find_rules_forwards_proposals_config_key(self, tmp_path) -> None:
        """END-TO-END smoke of the proposal channel's reproduction path: one config +
        one command. A find-rules YAML with a `proposals:` key (consolidated artifact
        schema, extra keys ignored) certifies the proposal against the mined state; the
        certified rule lands in the output artifact and the provenance sidecar records
        the proposals file, its sha256, and the per-outcome counts."""
        (tmp_path / "rules.json").write_text("[]")
        engine_cfg = tmp_path / "engine.yaml"
        engine_cfg.write_text(yaml.safe_dump({"operators": _SMOKE_OPERATORS, "rules": "rules.json"}))
        proposals_file = tmp_path / "proposals.json"
        proposals_file.write_text(json.dumps({
            "schema": "llm-proposals-v1",
            "proposals": [{"source": ["*", "exp", "x0", "exp", "x0"],
                           "target": ["pow2", "exp", "x0"], "why": "e^x * e^x = (e^x)^2"}],
        }))
        mine_cfg = tmp_path / "find_rules.yaml"
        mine_cfg.write_text(yaml.safe_dump({
            "max_source_pattern_length": 3,
            "max_target_pattern_length": 3,
            "dummy_variables": 1,
            "extra_internal_terms": ["0", "1", "<constant>"],
            "n_samples": 256,
            "constants_fit_retries": 16,
            "seed": 7,
            "proposals": str(proposals_file),
        }))
        out = tmp_path / "out" / "mined.json"
        main(["find-rules", "-e", str(engine_cfg), "-c", str(mine_cfg), "-o", str(out)])

        rules = {tuple(tuple(side) for side in rule) for rule in json.load(open(out))}
        assert (("*", "exp", "?0", "exp", "?0"), ("pow2", "exp", "?0")) in rules
        sidecar = json.load(open(str(out) + ".provenance.json"))
        assert sidecar["proposals"]["file"] == str(proposals_file)
        assert sidecar["proposals"]["sha256"] == hashlib.sha256(proposals_file.read_bytes()).hexdigest()
        assert sidecar["proposals"]["count"] == 1
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 1, "already_covered": 0, "rejected": 0, "duplicate": 0}
