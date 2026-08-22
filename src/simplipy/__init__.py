"""SimpliPy: efficient simplification of mathematical expressions.

Exposes the public API for parsing, transforming, and simplifying symbolic
expressions in prefix notation: the :class:`SimpliPyEngine` (backed by the
required compiled Rust core, ``simplipy._core``), expression-normalization
helpers, token/rule utilities, and asset management for downloading and
resolving engine rulesets and test data.
"""
import warnings as _warnings
from typing import Any as _Any

from .engine import DEFAULT_ENGINE, DEFAULT_ENGINE_REVISION, SimpliPyEngine, Mode
from . import engine
from . import operators
from . import utils
from .utils import (
    codify, deduplicate_rules, explicit_constant_placeholders
)
from . import normalization
from .normalization import (
    normalize_variable_token, to_expression, to_skeleton
)
from . import masking
from .asset_manager import (
    get_path, install_asset as install, uninstall_asset as uninstall, list_assets
)

from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version("simplipy")
except _PackageNotFoundError:  # running from a source checkout without an installed dist
    __version__ = "0.0.0+unknown"

# The declared public surface (D11 column, owner-ratified 2026-08-16; see the
# compatibility policy). Submodule names are deliberately NOT declared (DP-A):
# they stay importable and documented, but `from simplipy import *` no longer
# injects modules into the caller's namespace (simplipy.io shadowed stdlib io).
__all__ = [
    'DEFAULT_ENGINE', 'DEFAULT_ENGINE_REVISION', 'Mode', 'SimpliPyEngine', '__version__',
    'codify', 'deduplicate_rules', 'explicit_constant_placeholders',
    'get_path', 'install', 'list_assets', 'uninstall',
    'normalize_variable_token', 'to_expression', 'to_skeleton',
]
