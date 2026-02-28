## SimpliPy — Copilot instructions (concise)

This file orients AI coding agents to the SimpliPy repository: core architecture, common workflows, repo-specific conventions and concrete examples to be used when editing code.

- Big picture
  - Core components live under `src/simplipy/`:
    - `engine.py`: SimpliPyEngine — parses, represents (prefix tokens), applies compiled simplification rules, and renders infix. Primary entrypoint for simplification logic.
    - `asset_manager.py`: fetch/install/list/uninstall engine rule/config assets via a HuggingFace-style manifest. Use `get_path(...)` to resolve assets; assets can be installed with `install=True`.
    - `io.py`: configuration loading/saving (`load_config`, `save_config`). Engine configs are YAML/JSON files referenced by assets.
    - `operators.py`: numeric operator realizations (numpy-first, optional PyTorch handling). Operators provide `realization` strings referenced in configs.
    - `utils.py`: many small helpers (pattern matching, remapping, placeholder handling). Key tokens: `'<constant>'`, `C_0` style placeholders, and functions such as `explicit_constant_placeholders`, `numbers_to_constant`, `deduplicate_rules`.

- Data flow and conventions
  - Expressions: canonical internal form is prefix token lists (e.g. `['*','<constant}','x']`). The engine also supports infix strings via `parse`/`prefix_to_infix`.
  - Rules: loaded from `rules.json` in assets — lists of (pattern, replacement) in prefix token form. `engine.compile_rules()` separates wildcard-pattern rules (placeholders like `_0`) from explicit rules.
  - Operators: config `operators` map token -> properties including `realization` (e.g. `np.sin`). The engine introspects realizations and imports required modules via `utils.get_used_modules`.
  - Assets: asset manifest defaults to `psaegert/simplipy-assets`. Tests reference a test manifest `psaegert/simplipy-assets-test`.

- Developer workflows (concrete commands)
  - Setup: `pip install -e .[dev]` then `pre-commit install` (see `README.md`). Requires Python >= 3.11 per `pyproject.toml`.
  - Tests: run `pytest tests --cov src --cov-report html`. Integration tests are marked with `@pytest.mark.integration`; to skip them use `-m "not integration"`.
  - Asset-driven workflows: to load an engine in code/examples use:
    - `engine = SimpliPyEngine.load("dev_7-3", install=True)` — this will auto-download the engine asset via the manifest if missing.

- Project-specific patterns to follow
  - Prefer prefix-token manipulations and utility helpers in `utils.py` for transformations (do not reimplement placeholder handling; reuse `explicit_constant_placeholders`, `numbers_to_constant`, `apply_variable_mapping`, etc.).
  - Operator realizations expect `numpy` names; when adding a realization string include the module (e.g. `np.sin` or `math.cos`) so `SimpliPyEngine.import_modules()` can import them.
  - Rules should be canonicalized via `deduplicate_rules` when being generated or updated.
  - Typing: `pyproject.toml` enforces `disallow_untyped_defs = true` in `mypy` — add type hints on new functions and maintain existing style.

- Integration points & external deps
  - Hugging Face Hub is used for assets (`huggingface_hub` functions in `asset_manager.py`). Tests that exercise asset download use `psaegert/simplipy-assets-test` and require network access.
  - Optional `torch` extra exists in `pyproject.toml` — `operators.py` conditionally supports torch tensors when available.
  - CI: repository uses `pre-commit` and GitHub Actions for `pytest` and `pre-commit` checks (see badges in `README.md`).

- Quick examples to reference while coding
  - Converting expressions: `SimpliPyEngine.prefix_to_infix(tokens, power='**', realization=False)` (see `engine.py`).
  - Resolving assets programmatically: `from simplipy.asset_manager import get_path; get_path('dev_7-3', install=True)`.
  - Config loading: `from simplipy.io import load_config; cfg = load_config(path_to_yaml)` (relative paths in configs are resolved).

If anything here is unclear or you need more specific examples (e.g. how rules are formatted inside asset `rules.json`, or how `SimpliPyEngine.simplify()` is typically called by downstream code), tell me which area to expand and I will iterate.
