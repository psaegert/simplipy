"""Artifact-generation compatibility: the load-time allowlist (owner ruling 2026-08-03).

An engine ARTIFACT (a config + mined ruleset) belongs to a GENERATION:

* **Generation 1** -- the retired hyper-operator vocabulary era (``mult2..5``,
  ``div2..5``, ``pow2..5``, ``pow1_2``/``pow1_4``): the ``2-1``/``3-2``/``4-3``
  artifacts and the ``dev_7-*`` development engines, mined by simplipy <= 0.11.
  Their pattern rules predate the certificate sorts and the AC core's unified
  representation makes spellings reachable on which those rules are unsound --
  serving them on this engine is exactly the configuration the compatibility
  matrix forbids.
* **Generation 2** -- the AC-engine era (the clean 23-operator vocabulary): the
  ``acj-*`` family and the ``base`` config, mined by simplipy >= 0.12.

The rule is MUTUAL: this package loads only the generations in
:data:`SUPPORTED_ENGINE_GENERATIONS`, and an artifact declaring a NEWER generation
than this package knows refuses too (a future generation-3 artifact must not serve
on generation-2 code). Artifacts declare their generation explicitly with an
``engine_generation`` key in ``config.yaml``; the DECLARATION IS AUTHORITATIVE.
Configs published before the key existed are classified by their operator
vocabulary instead (any retired token present = generation 1) -- which covers
every already-published legacy artifact without republishing, while custom
configs whose vocabularies legitimately reuse retired names (mining universes,
test tables) opt in by declaring ``engine_generation: 2``. The threat model is
ACCIDENTAL cross-generation use; hand-editing a legacy artifact's declaration is
deliberate tampering and out of scope.

Enforcement lives at the ARTIFACT-LOADING boundary (:meth:`SimpliPyEngine.from_config`,
:meth:`SimpliPyEngine.load`). In-memory construction from a raw operator dict is
deliberately NOT gated: custom vocabularies are a feature, and the conversion layer
legitimately hosts the retired vocabulary as an INPUT language (legacy spellings
parse as sugar; no projection emits them). Asset MANAGEMENT (``install``/``list``/
``get_path``) is not gated either -- mirroring or inspecting a legacy artifact is
fine; serving it is not.
"""

from typing import Any

__all__ = [
    "ENGINE_GENERATION",
    "SUPPORTED_ENGINE_GENERATIONS",
    "RETIRED_OPERATOR_TOKENS",
    "GENERATION_1_ASSET_NAMES",
    "IncompatibleArtifactError",
    "infer_generation",
    "check_config",
    "check_asset_name",
]

#: The generation of THIS package's engine.
ENGINE_GENERATION = 2

#: Generations this package agrees to load. A one-element set today; a future
#: engine generation extends it only if it genuinely serves older artifacts soundly.
SUPPORTED_ENGINE_GENERATIONS = frozenset({2})

#: The retired hyper-operator vocabulary (deleted from the engine in 0.12.0). The
#: real odd roots ``pow1_3``/``pow1_5`` are NOT retired -- they remain genuine
#: generation-2 operators (a different function from the rational power on
#: negative bases).
RETIRED_OPERATOR_TOKENS = frozenset({
    "mult2", "mult3", "mult4", "mult5",
    "div2", "div3", "div4", "div5",
    "pow2", "pow3", "pow4", "pow5",
    "pow1_2", "pow1_4",
})

#: Published asset names known to be generation 1 -- refused by name BEFORE any
#: download, with the actionable pin in the message. Names outside this table are
#: still classified after resolution via the config-level checks.
GENERATION_1_ASSET_NAMES = frozenset({
    "2-1", "3-2", "4-3",
    "dev_7-2", "dev_7-3", "dev_7-3_numeric",
})

_PIN_HINT = 'pin the legacy package to load it: pip install "simplipy<0.12"'


class IncompatibleArtifactError(ValueError):
    """An artifact and this simplipy generation refuse each other at load."""


def infer_generation(operators: dict[str, Any]) -> int:
    """Classify an operator vocabulary: any retired hyper-operator token present
    means generation 1; a clean vocabulary is generation 2."""
    return 1 if RETIRED_OPERATOR_TOKENS & set(operators) else 2


def check_config(config: dict[str, Any], source: str) -> None:
    """Refuse a config whose artifact generation this package does not serve.

    The explicit ``engine_generation`` declaration is authoritative when present;
    an un-declared config is classified by its operator vocabulary
    (:func:`infer_generation`). A generation outside
    :data:`SUPPORTED_ENGINE_GENERATIONS` refuses with the actionable direction.
    """
    operators = config.get("operators") or {}
    declared = config.get("engine_generation")
    # bool is excluded explicitly: isinstance(True, int) holds in Python, so a YAML
    # `engine_generation: true` would otherwise classify as generation 1 and produce a
    # nonsense "generation True" refusal message instead of naming the malformation (D3).
    if declared is not None and (isinstance(declared, bool) or not isinstance(declared, int)):
        raise IncompatibleArtifactError(
            f"{source}: malformed engine_generation {declared!r} (expected an integer)")

    if declared is not None:
        generation, evidence = declared, "declares"
    else:
        generation, evidence = infer_generation(operators), "carries"
    if generation in SUPPORTED_ENGINE_GENERATIONS:
        return
    if generation < ENGINE_GENERATION:
        retired = sorted(RETIRED_OPERATOR_TOKENS & set(operators))
        detail = f" (retired operators: {', '.join(retired)})" if retired else ""
        raise IncompatibleArtifactError(
            f"{source} {evidence} engine generation {generation}{detail}: this artifact "
            f"was mined for the retired hyper-operator vocabulary and is served only by "
            f"simplipy <= 0.11 -- {_PIN_HINT}. simplipy {_package_version()} serves "
            f"generation(s) {sorted(SUPPORTED_ENGINE_GENERATIONS)} "
            f"(the acj-* artifact family).")
    raise IncompatibleArtifactError(
        f"{source} {evidence} engine generation {generation}, newer than this package "
        f"(simplipy {_package_version()} serves generation(s) "
        f"{sorted(SUPPORTED_ENGINE_GENERATIONS)}): upgrade simplipy to load it.")


def check_asset_name(name: str) -> None:
    """Refuse a KNOWN generation-1 asset by name, before any download."""
    if name in GENERATION_1_ASSET_NAMES:
        raise IncompatibleArtifactError(
            f"asset '{name}' is a generation-1 artifact (retired hyper-operator "
            f"vocabulary), served only by simplipy <= 0.11 -- {_PIN_HINT}. On this "
            f"version use the acj-* family instead (e.g. 'acj-4-3').")


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("simplipy")
    except Exception:
        return "(unknown version)"
