"""The artifact-generation allowlist (owner ruling 2026-08-03): generation-1 artifacts
(the retired hyper-operator vocabulary: 2-1/3-2/4-3, dev_7-*) load only on
simplipy <= 0.11, and this package refuses them at every ARTIFACT-LOADING path with
the actionable pin in the message. Generation-2 artifacts (acj-*) load exactly as
before, and an artifact declaring a NEWER generation refuses too (mutual pinning).

The boundary is deliberate: asset MANAGEMENT (install/get_path) and raw IN-MEMORY
construction stay open -- mirroring a legacy artifact is fine, and the retired
vocabulary remains a supported INPUT language for the conversion layer.
"""
import os

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.compat import (GENERATION_1_ASSET_NAMES, IncompatibleArtifactError,
                             infer_generation)
from simplipy.io import load_config
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACJ_CONFIG = acj_config_path()
LEGACY_CONFIG = os.path.expanduser('~/.cache/simplipy/engines/4-3/config.yaml')


def _require_or_skip(path: str, why: str) -> None:
    from conftest import require_or_skip
    require_or_skip(path, why)


class TestNameAllowlist:
    def test_known_generation1_names_refuse_before_download(self):
        for name in sorted(GENERATION_1_ASSET_NAMES):
            with pytest.raises(IncompatibleArtifactError, match='simplipy<0.12'):
                SimpliPyEngine.load(name)

    def test_generation2_name_is_not_blocked_by_the_name_gate(self):
        # acj-4-3 must never trip the NAME gate; load itself is exercised by the
        # asset-manager suite (network), the config path below (local) suffices here.
        from simplipy.compat import check_asset_name
        check_asset_name('acj-4-3')
        check_asset_name('base')


class TestConfigGate:
    def test_legacy_cached_config_refuses(self):
        _require_or_skip(LEGACY_CONFIG, 'legacy 4-3 asset not staged in ~/.cache')
        with pytest.raises(IncompatibleArtifactError, match='retired hyper-operator'):
            SimpliPyEngine.from_config(LEGACY_CONFIG)

    def test_acj_config_loads(self):
        _require_or_skip(ACJ_CONFIG, 'acj-4-3 config not staged')
        engine = SimpliPyEngine.from_config(ACJ_CONFIG)
        assert engine.simplify(['+', 'x0', 'x0']) == ['<mul>', '2', 'x0', '</mul>']

    def test_vocabulary_sniff_refuses_retired_tokens(self, tmp_path):
        # The sniff classifies UN-declared configs (the published legacy artifacts):
        # drop the declaration the pinned acj config now carries.
        _require_or_skip(ACJ_CONFIG, 'acj-4-3 config not staged')
        config = load_config(ACJ_CONFIG)
        config.pop('engine_generation', None)
        config['operators']['mult3'] = {
            # R3: a realization is compiled into generated source, so the shape is
            # validated at load -- a dotted name, not an inline lambda (audit B14).
            'realization': 'np.tan', 'alias': [], 'inverse': None,
            'arity': 1, 'precedence': None, 'commutative': False}
        config.pop('rules', None)
        p = tmp_path / 'config.yaml'
        p.write_text(yaml.safe_dump(config, sort_keys=False))
        with pytest.raises(IncompatibleArtifactError, match='mult3'):
            SimpliPyEngine.from_config(str(p))

    def test_newer_generation_refuses_with_upgrade_hint(self, tmp_path):
        _require_or_skip(ACJ_CONFIG, 'acj-4-3 config not staged')
        config = load_config(ACJ_CONFIG)
        config['engine_generation'] = 3
        config.pop('rules', None)
        p = tmp_path / 'config.yaml'
        p.write_text(yaml.safe_dump(config, sort_keys=False))
        with pytest.raises(IncompatibleArtifactError, match='upgrade simplipy'):
            SimpliPyEngine.from_config(str(p))

    def test_explicit_declaration_is_authoritative_for_custom_vocabularies(self, tmp_path):
        # A config DECLARING generation 2 may legitimately reuse retired names in a
        # CUSTOM vocabulary (mining universes, test tables): the declaration wins.
        # The sniff exists to classify the UN-declared, already-published legacy
        # artifacts; every new config opts in explicitly. (Hand-editing a legacy
        # artifact's declaration is deliberate tampering, outside the threat model.)
        _require_or_skip(ACJ_CONFIG, 'acj-4-3 config not staged')
        config = load_config(ACJ_CONFIG)
        config['engine_generation'] = 2
        config['operators']['pow2'] = {
            'realization': 'np.square', 'alias': [], 'inverse': None,
            'arity': 1, 'precedence': None, 'commutative': False}
        config.pop('rules', None)
        p = tmp_path / 'config.yaml'
        p.write_text(yaml.safe_dump(config, sort_keys=False))
        engine = SimpliPyEngine.from_config(str(p))
        assert engine.simplify(['+', 'x0', 'x0']) == ['<mul>', '2', 'x0', '</mul>']

    def test_malformed_declaration_refuses(self, tmp_path):
        _require_or_skip(ACJ_CONFIG, 'acj-4-3 config not staged')
        config = load_config(ACJ_CONFIG)
        config['engine_generation'] = 'two'
        config.pop('rules', None)
        p = tmp_path / 'config.yaml'
        p.write_text(yaml.safe_dump(config, sort_keys=False))
        with pytest.raises(IncompatibleArtifactError, match='malformed'):
            SimpliPyEngine.from_config(str(p))

    def test_pinned_generation_2_loads(self, tmp_path):
        _require_or_skip(ACJ_CONFIG, 'acj-4-3 config not staged')
        config = load_config(ACJ_CONFIG)
        config['engine_generation'] = 2
        config.pop('rules', None)
        p = tmp_path / 'config.yaml'
        p.write_text(yaml.safe_dump(config, sort_keys=False))
        engine = SimpliPyEngine.from_config(str(p))
        assert engine.simplify(['+', 'x0', 'x0']) == ['<mul>', '2', 'x0', '</mul>']


class TestDeliberateBoundaries:
    def test_raw_construction_stays_open_for_the_input_language(self):
        # The conversion layer hosts the retired vocabulary as an INPUT language;
        # building a TABLE for it from a raw dict is caller-owned and stays legal.
        _require_or_skip(LEGACY_CONFIG, 'legacy 4-3 asset not staged in ~/.cache')
        config = load_config(LEGACY_CONFIG)
        assert infer_generation(config['operators']) == 1
        engine = SimpliPyEngine(operators=config['operators'], rules=[])
        assert engine.convert_expression(['pow2', 'pow1_3', 'x1']) == ['pow2', 'pow1_3', 'x1']

    def test_asset_management_of_legacy_artifacts_stays_open(self):
        # get_path resolves a cached legacy asset without refusing: mirroring and
        # inspecting are legal, serving is not.
        _require_or_skip(LEGACY_CONFIG, 'legacy 4-3 asset not staged in ~/.cache')
        from simplipy.asset_manager import get_path
        assert get_path('4-3').endswith('config.yaml')


class TestD12CleanCut:
    """D12's clean-cut rulings, owner-ratified 2026-08-16: `pow1`/`pow_1` join
    RETIRED_OPERATOR_TOKENS (a config declaring them was accepted while the parser
    unconditionally rewrote their tokens away -- one engine, two answers for one
    expression); the `inverse` spec key is no longer REQUIRED (nothing reads it --
    parse-only ceremony that made every wild config fail construction)."""

    def test_pow1_config_refuses_at_load(self, tmp_path):
        # the refusal choke point is the ARTIFACT-LOADING path (from_config/load);
        # raw in-memory construction stays open by design (0.12.0 contract)
        import pytest as _pytest
        import yaml
        from simplipy import SimpliPyEngine
        ops = {'pow1': {'realization': 'simplipy.operators.pow1', 'alias': [],
                        'arity': 1, 'precedence': 3, 'commutative': False}}
        cfg = tmp_path / 'config.yaml'
        cfg.write_text(yaml.safe_dump({'operators': ops}))
        with _pytest.raises(Exception, match='pow1|[Gg]eneration|retired'):
            SimpliPyEngine.from_config(str(cfg))

    def test_inverse_key_is_optional(self):
        from simplipy import SimpliPyEngine
        ops = {'+': {'realization': '+', 'alias': [], 'arity': 2,
                     'precedence': 1, 'commutative': True}}
        eng = SimpliPyEngine(operators=ops)  # no 'inverse' anywhere: must load
        assert eng.simplify(['+', 'x0', '0']) is not None

    def test_shipped_configs_carry_no_inverse_key(self):
        import os
        import yaml
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'remine')
        for cell in ('acj-2-1', 'acj-3-2', 'acj-4-3'):
            path = os.path.join(base, cell, 'config.yaml')
            if not os.path.exists(path):
                continue
            cfg = yaml.safe_load(open(path))
            for name, spec in cfg['operators'].items():
                assert 'inverse' not in spec, f'{cell}:{name} still carries inverse:'
