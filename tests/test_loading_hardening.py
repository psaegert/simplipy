"""Loading hardening: every path a manifest names stays inside the asset cache, a config's
operator declares the arity the core evaluates it with, and the judge reads a `Fraction` on
every mpmath the package supports."""
import hashlib
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy import asset_manager as am

DIGEST = hashlib.sha256(b'[]').hexdigest()


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    root = tmp_path / 'cache'
    (root / 'safe').mkdir(parents=True)
    (tmp_path / 'outside.yaml').write_text('operators: {}\n')
    (root / 'safe' / 'rules.json').write_text('[]')
    (root / 'safe' / 'config.yaml').write_text('operators: {}\n')
    return root


def _entry(**fields) -> dict:
    entry = {'directory': 'safe', 'entrypoint': 'config.yaml',
             'files': ['config.yaml', 'rules.json'], 'sha256': {'rules.json': DIGEST}}
    entry.update(fields)
    return entry


@pytest.mark.parametrize('field, value', [
    ('entrypoint', '../../outside.yaml'), ('entrypoint', 'ABSOLUTE'),
    ('sha256', {'rules.json': DIGEST, '../../outside.yaml': '0' * 64}),
    ('directory', '../outside'), ('directory', 'ABSOLUTE'),
    ('files', ['config.yaml', '../../outside.yaml'])])
def test_every_manifest_path_stays_inside_the_cache(cache, monkeypatch, field, value) -> None:
    """A manifest is downloaded data, never a path authority: an absolute path or one that
    leaves its directory is refused before the filesystem is touched, in every field."""
    if value == 'ABSOLUTE':
        value = str(cache.parent / 'outside.yaml') if field == 'entrypoint' else str(cache.parent)
    entry = _entry(**{field: value})
    monkeypatch.setattr(am, '_cached_manifest', lambda **kwargs: {'fixture': entry})
    monkeypatch.setattr(am, 'fetch_manifest', lambda **kwargs: {'fixture': entry})
    with pytest.raises(ValueError, match='manifest'):
        am.get_path('fixture', local_dir=cache)


def test_a_contained_manifest_resolves(cache, monkeypatch) -> None:
    entry = _entry()
    monkeypatch.setattr(am, '_cached_manifest', lambda **kwargs: {'fixture': entry})
    path = Path(am.get_path('fixture', local_dir=cache)).resolve()
    assert path == (cache / 'safe' / 'config.yaml').resolve()


@pytest.mark.parametrize('name, arity', [
    ('sin', 0), ('cos', 0), ('exp', 0), ('floor', 0), ('max', 0), ('besselj', 0),
    ('sin', 2), ('max', 1), ('pow', 1)])
def test_a_config_declares_the_arity_the_core_evaluates(name, arity) -> None:
    """The core dispatches its own operators by their kernel's arity, so a config declaring
    another one is refused at ingestion with an ordinary exception."""
    ops = {name: {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
                  'commutative': False}}
    with pytest.raises(ValueError, match='arity'):
        SimpliPyEngine(operators=ops, rules=[])


def test_a_zero_arity_sin_is_refused_at_load(tmp_path) -> None:
    (tmp_path / 'empty.json').write_text('[]')
    config = {'operators': {'sin': {'realization': 'simplipy.operators.sin', 'alias': [],
                                    'arity': 0, 'commutative': False}},
              'rules': 'empty.json', 'rules_permissive': 'empty.json'}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='arity'):
        SimpliPyEngine.load(str(tmp_path / 'config.yaml'))


def test_the_judge_reads_a_fraction_on_every_mpmath(monkeypatch) -> None:
    """mpmath before 1.4 builds no `mpf` from a `Fraction`; the judge reads one through its
    numerator and denominator, rounded once at the working precision, so it decides the same
    rule on either mpmath."""
    import mpmath
    from simplipy.verify import _contract, verify_rule
    old = mpmath.mp.dps
    try:
        mpmath.mp.dps = 50
        for q in (Fraction(1, 3), Fraction(10 ** 80 + 1, 3 ** 70), Fraction(-7, 2)):
            try:
                want = mpmath.mpf(q)  # mpmath 1.4 reads a Fraction, rounded once
            except TypeError:  # mpmath 1.3
                continue
            assert _contract.mpf(q) == want, q
        assert _contract.mpf(Fraction(1, 3)) == mpmath.mpf(1) / 3
    finally:
        mpmath.mp.dps = old
    real = _contract._mpmath_mpf

    def no_fraction(v=0, *args, **kwargs):
        if isinstance(v, Fraction):
            raise TypeError(f'cannot create mpf from {v!r}')
        return real(v, *args, **kwargs)

    monkeypatch.setattr(_contract, '_mpmath_mpf', no_fraction)
    got = verify_rule('tan asin ceil x0'.split(), 'atanh ceil x0'.split())
    assert got['verdict'] == 'KILL' and got['tier'] == 'permissive', got
