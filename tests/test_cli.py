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
