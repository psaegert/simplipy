import json
import shutil
from pathlib import Path
from typing import Literal

import platformdirs
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- Configuration ---
# The central manifest file defining all official assets.
HF_MANIFEST_REPO = "psaegert/simplipy-assets"
HF_MANIFEST_FILENAME = "manifest.json"

AssetType = Literal['ruleset', 'test-data']

ASSET_KEYS = {
    'ruleset': 'rulesets',
    'test-data': 'test-data'
}


# --- Core Functions ---


def get_default_cache_dir() -> Path:
    """
    Gets the OS-appropriate cache directory for SimpliPy assets.
    Follows XDG Base Directory Specification on Linux.
    """
    cache_dir = Path(platformdirs.user_cache_dir(appname="simplipy"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fetch_manifest() -> dict:
    """
    Downloads the latest asset manifest from Hugging Face.
    """
    try:
        manifest_path = hf_hub_download(
            repo_id=HF_MANIFEST_REPO,
            filename=HF_MANIFEST_FILENAME,
            repo_type="dataset",
        )
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except HfHubHTTPError as e:
        print(f"Error: Could not download the asset manifest from Hugging Face: {e}")
        print("Please check your internet connection and repository access.")
        return {}


def get_asset_path(asset_type: AssetType, name: str, install: bool = False, local_dir: Path | str | None = None) -> str | None:
    """
    Gets the local path to an asset's entrypoint file.

    Handles local paths, official asset names, and auto-installation.

    Returns the path to the asset's entrypoint file (e.g., config.yaml).
    """
    # 1. Check if 'name' is already a valid local path
    if Path(name).exists():
        return name

    # 2. Treat 'name' as an official asset name
    manifest = fetch_manifest()
    if not manifest:
        return None

    key = ASSET_KEYS[asset_type]
    asset_info = manifest.get(key, {}).get(name)

    if not asset_info:
        print(f"Error: Unknown {asset_type} '{name}'.")
        return None

    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)

    entrypoint_path = local_dir / asset_info['entrypoint']

    if entrypoint_path.exists():
        return str(entrypoint_path)

    if install:
        print(f"{asset_type.capitalize()} '{name}' not found locally. Attempting to install")
        if install_asset(asset_type, name, local_dir=local_dir):
            return str(entrypoint_path)
        else:
            print(f"Failed to install {asset_type} '{name}'.")
            return None

    return None


def install_asset(asset_type: AssetType, name: str, force: bool = False, local_dir: Path | str | None = None) -> bool:
    """
    Installs an asset (e.g., a ruleset directory) from Hugging Face.
    """
    manifest = fetch_manifest()
    if not manifest:
        return False

    key = ASSET_KEYS[asset_type]
    asset_info = manifest.get(key, {}).get(name)
    if not asset_info:
        print(f"Error: Unknown {asset_type}: '{name}'.")
        list_assets(asset_type)
        return False

    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / key / name

    if local_path.exists() and not force:
        print(f"{asset_type.capitalize()} '{name}' is already installed at {local_path}")
        print("Use --force to reinstall.")
        return True

    if local_path.exists() and force:
        print(f"Force option specified. Removing existing version of '{name}'...")
        remove_asset(asset_type, name, quiet=True, local_dir=local_dir)

    print(f"Downloading {asset_type} '{name}' from {asset_info['repo_id']} to {local_path}")
    try:
        for file in asset_info['files']:
            print(file)
            hf_hub_download(
                repo_id=asset_info['repo_id'],
                filename=file,
                repo_type="dataset",
                local_dir=local_dir,
            )
        print(f"Successfully installed '{name}' to {local_dir}")
        return True
    except HfHubHTTPError as e:
        print(f"Error downloading '{name}': {e}")
        # Clean up partial download
        if local_dir.exists():
            shutil.rmtree(local_dir)
        return False


def remove_asset(asset_type: AssetType, name: str, quiet: bool = False, local_dir: Path | str | None = None) -> bool:
    """
    Removes a locally installed asset.
    """
    key = ASSET_KEYS[asset_type]
    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)
    local_path = local_dir / key / name

    if not local_path.exists():
        if not quiet:
            print(f"{asset_type.capitalize()} '{name}' is not installed.")
        return True

    try:
        shutil.rmtree(local_path)
        if not quiet:
            print(f"Successfully removed '{name}'.")
        return True
    except OSError as e:
        if not quiet:
            print(f"Error removing '{name}': {e}")
        return False


def list_assets(asset_type: AssetType, installed_only: bool = False, local_dir: Path | str | None = None) -> None:
    """
    Lists available or installed assets.
    """
    manifest = fetch_manifest()
    if not manifest:
        return

    key = ASSET_KEYS[asset_type]

    print(f"--- {'Installed' if installed_only else 'Available'} {key.capitalize()} ---")

    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)
    asset_collection = manifest.get(key, {})

    found_any = False
    for name, info in asset_collection.items():
        local_path = local_dir / key / name
        is_installed = local_path.exists()

        if installed_only and not is_installed:
            continue

        status = "[installed]" if is_installed else ""
        print(f"- {name:<15} {status:<12} {info['description']}")
        found_any = True

    if not found_any:
        print(f"No {asset_type}s found.")
