from .engine import SimpliPyEngine, SimplificationStatistics
from . import engine
from . import operators
from . import utils
from .utils import (
    codify, deduplicate_rules, explicit_constant_placeholders
)
from .asset_manager import (
    get_path, install_asset as install, uninstall_asset as uninstall, list_assets
)

from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version("simplipy")
except _PackageNotFoundError:  # running from a source checkout without an installed dist
    __version__ = "0.0.0+unknown"
