from pathlib import Path

# This assumes your asset management code is saved in 'asset_manager.py'
import simplipy as sp
import pytest
import huggingface_hub as hf_hub  # the lazy-import patch target (D30)

# --- Test Constants ---
# These tests use real, known assets from the psaegert/simplipy-assets-test manifest.
# An active internet connection is required to run them. The asset MANAGER is
# generation-agnostic (mirroring/inspection is a sanctioned-open boundary), but the
# suite exercises it on the CURRENT generation-2 artifact (audit Tier-1 #3: no test
# depends on a legacy asset).
VALID_ENGINE = "acj-4-3"
VALID_TEST_DATA = "expressions_10k"
INVALID_ASSET = "this-asset-does-not-exist"

HF_MANIFEST_REPO = "psaegert/simplipy-assets-test"
HF_MANIFEST_FILENAME = "manifest.json"


def test_install_and_remove_asset(tmp_path: Path):
    """
    Tests the full lifecycle: installing a valid asset and then removing it.
    """
    # --- 1. Installation ---
    # Arrange: Use tmp_path as our isolated local directory.
    # Act: Install a known valid engine.
    install_success = sp.install(
        asset=VALID_ENGINE,
        local_dir=tmp_path,
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )

    # Assert: Check that the installation was successful and files exist.
    assert install_success is True
    expected_dir = tmp_path / "engines" / VALID_ENGINE
    assert expected_dir.is_dir()
    # Based on the manifest for 'acj-4-3', these files should exist.
    assert (expected_dir / "config.yaml").is_file()
    assert (expected_dir / "rules.json").is_file()

    # --- 2. Removal ---
    # Arrange: The asset is now installed.
    # Act: Remove the same asset.
    remove_success = sp.uninstall(
        asset=VALID_ENGINE,
        local_dir=tmp_path,
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )

    # Assert: The removal should be successful and the directory should be gone.
    assert remove_success is True
    assert not expected_dir.exists()


def test_install_non_existent_asset(tmp_path: Path):
    """
    Tests that attempting to install a non-existent asset fails gracefully.
    """
    # Arrange: An invalid asset name.
    # Act: Attempt to install the non-existent asset.
    success = sp.install(
        asset=INVALID_ASSET,
        local_dir=tmp_path,
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )

    # Assert: The function should return False and not create any directories.
    assert success is False
    assert not (tmp_path / "engines" / INVALID_ASSET).exists()


def test_get_asset_path_auto_install(tmp_path: Path):
    """
    Tests get_asset_path's ability to automatically download and return the path
    for an asset that is not yet installed.
    """
    # Arrange: The asset is not installed since tmp_path is empty.
    # Act: Get the path with auto_install=True (the default).
    path_str = sp.get_path(
        asset=VALID_ENGINE,
        local_dir=tmp_path,
        install=True,
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )

    # Assert: A valid path string is returned and the asset is now installed.
    assert path_str is not None
    # The manifest for 'acj-4-3' has entrypoint 'engines/acj-4-3/config.yaml'.
    expected_path = tmp_path / "engines" / VALID_ENGINE / "config.yaml"
    assert path_str == str(expected_path)
    assert expected_path.is_file()


def test_get_asset_path_no_install(tmp_path: Path):
    """
    Tests that get_asset_path raises FileNotFoundError if an asset is not installed
    and install is explicitly set to False.
    """
    # Arrange: The asset is not installed.
    # Act & Assert: Get the path with install=False should raise FileNotFoundError.

    with pytest.raises(FileNotFoundError):
        sp.get_path(
            asset=VALID_ENGINE,
            local_dir=tmp_path,
            install=False,
            repo_id=HF_MANIFEST_REPO,
            manifest_filename=HF_MANIFEST_FILENAME
        )

    # Assert: The asset should not be installed.
    assert not (tmp_path / "engines" / VALID_ENGINE).exists()


def test_get_asset_path_for_local_file(tmp_path: Path):
    """
    Tests that get_asset_path correctly identifies and returns a path
    that already exists on the local filesystem.
    """
    # Arrange: Create a dummy local file.
    local_file = tmp_path / "my_custom_rule.yaml"
    local_file.touch()

    # Act: Call get_asset_path with the full path to the local file.
    path_str = sp.get_path(
        asset=str(local_file),
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )

    # Assert: The function returns the original path string without modification.
    assert path_str == str(local_file)


def test_list_assets_installed_and_available(capsys, tmp_path: Path):
    """
    Tests the list_assets function for both available and installed-only views.
    `capsys` is a pytest fixture to capture stdout.
    """
    # --- 1. List all available assets when none are installed ---
    sp.list_assets('engine', installed_only=False, local_dir=tmp_path, repo_id=HF_MANIFEST_REPO, manifest_filename=HF_MANIFEST_FILENAME)
    captured = capsys.readouterr()
    output = captured.out

    assert "--- Available engine assets ---" in output
    assert VALID_ENGINE in output
    assert "[installed]" not in output  # Should not be marked as installed

    # --- 2. Install an asset and list only installed ---
    sp.install(VALID_ENGINE, local_dir=tmp_path, repo_id=HF_MANIFEST_REPO, manifest_filename=HF_MANIFEST_FILENAME)
    sp.list_assets('engine', installed_only=True, local_dir=tmp_path, repo_id=HF_MANIFEST_REPO, manifest_filename=HF_MANIFEST_FILENAME)
    captured = capsys.readouterr()
    output = captured.out

    assert "--- Installed engine assets ---" in output
    assert VALID_ENGINE in output
    assert "[installed]" in output
    # A known asset that wasn't installed should not be in the output.
    assert "cis-benchmark-v1" not in output

    # --- 3. The empty installed view names the type it is empty OF ---
    sp.list_assets('test-data', installed_only=True, local_dir=tmp_path, repo_id=HF_MANIFEST_REPO, manifest_filename=HF_MANIFEST_FILENAME)
    output = capsys.readouterr().out
    assert "--- Installed test-data assets ---" in output
    assert "No test-data assets installed." in output


def test_force_reinstall(tmp_path: Path):
    """
    Tests that the `force=True` flag correctly removes an existing
    asset before reinstalling it.
    """
    # Arrange: Install an asset and add a custom file to its directory.
    sp.install(VALID_ENGINE, local_dir=tmp_path, repo_id=HF_MANIFEST_REPO, manifest_filename=HF_MANIFEST_FILENAME)
    asset_dir = tmp_path / "engines" / VALID_ENGINE
    custom_file = asset_dir / "custom_file.txt"
    custom_file.touch()
    assert custom_file.exists()  # Verify setup

    # Act: Reinstall the same asset with force=True.
    success = sp.install(
        asset=VALID_ENGINE,
        force=True,
        local_dir=tmp_path
    )

    # Assert: The reinstall succeeded and the custom file is now gone.
    assert success is True
    assert not custom_file.exists()
    assert (asset_dir / "config.yaml").is_file()  # Check original files exist.


def test_install_different_asset_types(tmp_path: Path):
    """
    Tests that different asset types ('engine', 'test-data') are handled
    and stored in their respective subdirectories.
    """
    # Arrange & Act: Install one of each asset type.
    engine_success = sp.install(
        asset=VALID_ENGINE,
        local_dir=tmp_path,
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )
    test_data_success = sp.install(
        asset=VALID_TEST_DATA,
        local_dir=tmp_path,
        repo_id=HF_MANIFEST_REPO,
        manifest_filename=HF_MANIFEST_FILENAME
    )

    # Assert: Both installations succeeded and created the correct directories.
    assert engine_success is True
    assert test_data_success is True

    expected_engine_dir = tmp_path / "engines" / VALID_ENGINE
    expected_test_data_dir = tmp_path / "test-data" / "expressions_10k.json"

    print(expected_engine_dir)
    print(expected_test_data_dir)

    assert expected_engine_dir.is_dir()
    assert expected_test_data_dir.is_file()


class TestR6ArtifactIdentity:
    """R6 (G2's identity half): the manifest carries `revision` + per-file `sha256`,
    and both are ENFORCED -- the audit watched the shared cache change mid-pytest-run
    and an artifact republished under the same name mid-audit; that is the one defect
    class that changes behaviour with no commit. Corrupting a byte of a cached file
    must make resolution RAISE, never silently serve; entries without digests
    (pre-R6 manifests) stay permissive."""

    def _stage(self, tmp_path, with_digests=True):
        import hashlib
        d = tmp_path / 'engines' / 'toy'
        d.mkdir(parents=True)
        contents = {'config.yaml': 'operators: {}\n', 'rules.json': '[]'}
        digests = {}
        for name, content in contents.items():
            (d / name).write_text(content)
            digests[name] = hashlib.sha256(content.encode()).hexdigest()
        entry = {'type': 'engine', 'repo_id': 'x/y', 'directory': 'engines/toy',
                 'files': list(contents), 'entrypoint': 'config.yaml'}
        if with_digests:
            entry['revision'] = 'deadbeefcafe'
            entry['sha256'] = digests
        return {'toy': entry}

    def test_cache_corruption_raises_at_resolution(self, tmp_path, monkeypatch):
        from simplipy import asset_manager as am
        manifest = self._stage(tmp_path)
        monkeypatch.setattr(am, 'fetch_manifest', lambda **kw: manifest)
        assert am.get_path('toy', local_dir=tmp_path).endswith('config.yaml')
        (tmp_path / 'engines' / 'toy' / 'rules.json').write_text('[["evil"]]')
        with pytest.raises(RuntimeError, match='sha256'):
            am.get_path('toy', local_dir=tmp_path)

    def test_predigest_manifest_stays_permissive(self, tmp_path, monkeypatch):
        from simplipy import asset_manager as am
        manifest = self._stage(tmp_path, with_digests=False)
        monkeypatch.setattr(am, 'fetch_manifest', lambda **kw: manifest)
        (tmp_path / 'engines' / 'toy' / 'rules.json').write_text('[["evil"]]')
        assert am.get_path('toy', local_dir=tmp_path).endswith('config.yaml')

    def test_install_pins_revision_and_verifies(self, tmp_path, monkeypatch):
        from simplipy import asset_manager as am
        manifest = self._stage(tmp_path, with_digests=True)
        # a fresh cache dir: force the install path
        cache = tmp_path / 'fresh'
        seen = {}

        def fake_download(repo_id, filename, repo_type, local_dir, **kw):
            seen['revision'] = kw.get('revision')
            out = Path(local_dir) / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text('CORRUPTED')  # wrong content -> digests must convict
            return str(out)

        monkeypatch.setattr(am, 'fetch_manifest', lambda **kw: manifest)
        monkeypatch.setattr(hf_hub, 'hf_hub_download', fake_download)
        with pytest.raises(RuntimeError, match='sha256'):
            am.install_asset('toy', local_dir=cache)
        assert seen['revision'] == 'deadbeefcafe', 'download not pinned to the manifest revision'
        assert not (cache / 'engines' / 'toy').exists(), 'corrupt install must be cleaned up'


class TestRmtreeBoundary:
    """missed-004 + missed-006 + threatmodel-4 (+ dep-4-hf-exception-miss): one
    boundary, every rmtree/join site. The manifest's `directory` reached rmtree
    unvalidated -- and no `..` is needed: Path(cache) / '/etc/x' IS Path('/etc/x'),
    which defeats any fix that only rejects traversal segments. Separately,
    `uninstall_asset` returned True having deleted nothing whenever
    fetch_manifest() failed to {} -- a silent no-op sold as success."""

    def _hostile(self, directory):
        return {'evil': {'type': 'engine', 'repo_id': 'x/y', 'directory': directory,
                         'files': ['config.yaml'], 'entrypoint': 'config.yaml'}}

    @pytest.mark.parametrize('directory', [
        '/etc/simplipy-escape-target',      # absolute: the segment-filter defeat
        '../../simplipy-escape-target',     # classic traversal
        'engines/../../simplipy-escape',    # embedded traversal
    ])
    def test_escaping_directory_refuses_everywhere(self, tmp_path, monkeypatch, directory):
        from simplipy import asset_manager as am
        victim = tmp_path.parent / 'simplipy-escape-target'
        monkeypatch.setattr(am, 'fetch_manifest', lambda **kw: self._hostile(directory))
        cache = tmp_path / 'cache'
        cache.mkdir()
        for fn in (lambda: am.get_path('evil', local_dir=cache),
                   lambda: am.uninstall_asset('evil', local_dir=cache),
                   lambda: am.install_asset('evil', local_dir=cache)):
            with pytest.raises((ValueError, RuntimeError)):
                fn()
        assert not victim.exists(), 'a hostile manifest directory escaped the cache root'

    def test_uninstall_without_manifest_is_loud(self, tmp_path, monkeypatch):
        from simplipy import asset_manager as am
        monkeypatch.setattr(am, 'fetch_manifest', lambda **kw: {})
        with pytest.raises(RuntimeError, match='manifest'):
            am.uninstall_asset('acj-4-3', local_dir=tmp_path)


class TestC51OfflineResolution:
    """C51: named-asset load ALWAYS round-tripped to the Hub -- a user on a plane
    with the asset FULLY INSTALLED got 23.5 s of retries and a raw
    `LocalEntryNotFoundError` naming no file. The manifest fetch now falls back to
    the last CACHED manifest copy when the network fails, so an installed asset
    resolves offline; only a cold cache with no network stays a (loud) failure."""

    def test_installed_asset_resolves_when_the_network_is_down(self, tmp_path, monkeypatch):
        import json as _json
        from simplipy import asset_manager as am
        # a fully installed asset in a scratch cache
        d = tmp_path / 'cache' / 'engines' / 'toy'
        d.mkdir(parents=True)
        (d / 'config.yaml').write_text('operators: {}\n')
        manifest = {'toy': {'type': 'engine', 'repo_id': 'x/y', 'directory': 'engines/toy',
                            'files': ['config.yaml'], 'entrypoint': 'config.yaml'}}
        cached = tmp_path / 'cached_manifest.json'
        cached.write_text(_json.dumps(manifest))

        def fake_download(repo_id, filename, repo_type, **kw):
            if kw.get('local_files_only'):
                return str(cached)          # the hub cache still holds the last copy
            raise OSError('network is unreachable')  # the plane

        monkeypatch.setattr(hf_hub, 'hf_hub_download', fake_download)
        p = am.get_path('toy', local_dir=tmp_path / 'cache')
        assert p.endswith('config.yaml')

    def test_cold_cache_offline_stays_loud(self, tmp_path, monkeypatch):
        from simplipy import asset_manager as am

        def dead(*a, **kw):
            raise OSError('network is unreachable')
        monkeypatch.setattr(hf_hub, 'hf_hub_download', dead)
        with pytest.raises((RuntimeError, FileNotFoundError)):
            am.get_path('toy', local_dir=tmp_path)
