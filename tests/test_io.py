import pytest
import yaml

from simplipy.io import load_config


class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_from_file(self, tmp_path) -> None:
        """Loads a YAML file and returns its contents as a dict."""
        cfg = {"operators": {"+": {"arity": 2}}}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)

        result = load_config(str(path))
        assert result == cfg

    def test_path_values_returned_verbatim(self, tmp_path) -> None:
        """Path-valued entries are NOT rewritten: resolution is the consumer's job
        (the 0.12 removal of the resolve_paths value-sniffing pass)."""
        cfg = {"rules": "./rules.json", "proposals": "proposals.json"}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)

        assert load_config(str(path)) == cfg

    def test_load_from_dict(self) -> None:
        """Passing a dict returns it unchanged."""
        cfg = {"key": "value"}
        assert load_config(cfg) is cfg

    def test_nonexistent_file_raises(self, tmp_path) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "missing.yaml"))

    def test_directory_path_raises(self, tmp_path) -> None:
        """A directory path raises ValueError."""
        with pytest.raises(ValueError):
            load_config(str(tmp_path))
