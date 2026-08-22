"""The DEFAULT engine: which artifact `load()` resolves when the caller names none.

Pinned IN THE PACKAGE (`simplipy.DEFAULT_ENGINE` / `DEFAULT_ENGINE_REVISION`) and not
in the hosted manifest, so that "what do I get by default" is answerable offline and
cannot change under a user without a simplipy release.

The revision is what disambiguates one NAME across simplipy versions: `acj-4` mined
under 0.14's judge and `acj-4` mined under 0.15's are both legitimately `acj-4`, and
only (name, revision) picks one. The measure-fingerprint check (D25/R6) stays the
safety net for an artifact loaded by name across a measure change; this pin is what
stops that happening on the path the user did not choose.

These tests never touch the network: the resolver is stubbed, because what is under
test is the ARGUMENT handling and the announcement, not the download.
"""

import pytest

import simplipy
from simplipy import SimpliPyEngine


@pytest.fixture
def resolved(monkeypatch):
    """Stub the resolver + config load; record what name reached the asset manager."""
    seen = {}

    def fake_get_path(asset, **kwargs):
        seen['asset'] = asset
        seen['kwargs'] = kwargs
        return '/nonexistent/config.yaml'

    monkeypatch.setattr('simplipy.engine.get_path', fake_get_path)
    monkeypatch.setattr(SimpliPyEngine, 'from_config',
                        classmethod(lambda cls, path, **kw: ('CONFIG', path)))
    return seen


class TestTheDefaultIsPinnedAndAnnounced:
    def test_no_argument_resolves_the_pinned_default(self, resolved):
        SimpliPyEngine.load()
        assert resolved['asset'] == simplipy.DEFAULT_ENGINE

    def test_the_implicit_default_is_announced(self, resolved, capsys):
        SimpliPyEngine.load()
        out = capsys.readouterr().out
        # The NAME must appear -- a default the user did not choose is one they cannot
        # reproduce unless they are told which it was.
        assert simplipy.DEFAULT_ENGINE in out
        assert 'engine=' in out, 'the announcement must say how to choose another'

    def test_an_explicit_name_is_NOT_announced(self, resolved, capsys):
        SimpliPyEngine.load('acj-4-3')
        assert capsys.readouterr().out == '', 'only the implicit path announces'

    def test_an_explicit_name_reaches_the_resolver_unchanged(self, resolved):
        SimpliPyEngine.load('acj-4-3')
        assert resolved['asset'] == 'acj-4-3'

    def test_the_pin_is_exported_from_the_package(self):
        assert 'DEFAULT_ENGINE' in simplipy.__all__
        assert 'DEFAULT_ENGINE_REVISION' in simplipy.__all__

    def test_the_default_names_a_generation2_artifact(self):
        # The default must not be a name this package refuses to load (compat gate).
        from simplipy.compat import check_asset_name
        check_asset_name(simplipy.DEFAULT_ENGINE)
