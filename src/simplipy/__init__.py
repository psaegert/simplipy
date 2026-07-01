"""SimpliPy: efficient simplification of mathematical expressions.

Exposes the public API for parsing, transforming, and simplifying symbolic
expressions in prefix notation: the :class:`SimpliPyEngine` and its
:class:`SimplificationStatistics` companion, expression-normalization helpers,
token/rule utilities, and asset management for downloading and resolving engine
rulesets and test data.
"""
from .engine import SimpliPyEngine, SimplificationStatistics
from . import engine
from . import operators
from . import utils
from .utils import (
    codify, deduplicate_rules, explicit_constant_placeholders
)
from . import normalization
from .normalization import (
    normalize_variable_token, normalize_skeleton, normalize_expression
)
from .asset_manager import (
    get_path, install_asset as install, uninstall_asset as uninstall, list_assets
)

from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version("simplipy")
except _PackageNotFoundError:  # running from a source checkout without an installed dist
    __version__ = "0.0.0+unknown"
