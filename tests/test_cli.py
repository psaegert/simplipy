import pytest

from simplipy.__main__ import main


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
