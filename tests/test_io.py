import os

import pytest
import yaml

from simplipy.io import load_config, save_config


class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_from_file(self, tmp_path) -> None:
        """Loads a YAML file and returns its contents as a dict."""
        cfg = {"operators": {"+": {"arity": 2}}}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)

        result = load_config(str(path), resolve_paths=False)
        assert result == cfg

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


class TestSaveConfig:
    """Tests for save_config()."""

    def test_roundtrip(self, tmp_path) -> None:
        """save_config then load_config returns the same dict."""
        cfg = {"a": 1, "b": [2, 3]}
        save_config(cfg, str(tmp_path), "out.yaml", recursive=False)
        loaded = load_config(str(tmp_path / "out.yaml"), resolve_paths=False)
        assert loaded == cfg

    def test_creates_directory(self, tmp_path) -> None:
        """save_config creates missing parent directories."""
        subdir = str(tmp_path / "sub" / "dir")
        save_config({"x": 1}, subdir, "cfg.yaml", recursive=False)
        assert os.path.isfile(os.path.join(subdir, "cfg.yaml"))
