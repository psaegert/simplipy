import json
import os
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

AssetType = Literal['ruleset', 'test-data', 'all']

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
        return {}


def get_path(asset: str, install: bool = False, local_dir: Path | str | None = None) -> str | None:
    """
    Gets the local path to an asset's entrypoint file.

    Handles local paths, official asset names, and auto-installation.

    Returns the path to the asset's entrypoint file (e.g., config.yaml).
    """
    # Check if 'asset' is a valid local path
    if Path(asset).exists():
        return asset

    # Otherwise, treat 'asset' as an official asset name
    manifest = fetch_manifest()
    if not manifest:
        raise RuntimeError("Could not fetch asset manifest.")

    asset_info = manifest.get(asset, {})
    if not asset_info:
        list_assets(asset_type='all')
        raise ValueError(f"Error: Unknown asset: '{asset}'. See above for available assets.")

    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)

    entrypoint_path = local_dir / asset_info['directory'] / asset_info['entrypoint']

    if entrypoint_path.exists():
        return str(entrypoint_path)

    if install:
        print(f"Asset '{asset}' is not installed. Installing.")
        if install_asset(asset, local_dir=local_dir):
            return str(entrypoint_path)
        else:
            raise RuntimeError(f"Failed to install asset '{asset}'.")

    raise FileNotFoundError(f"Asset '{asset}' is not installed. Use install=True to download it.")


def install_asset(asset: str, force: bool = False, local_dir: Path | str | None = None) -> bool:
    """
    Installs an asset (e.g., a ruleset directory) from Hugging Face.
    """
    manifest = fetch_manifest()
    if not manifest:
        return False

    asset_info = manifest.get(asset)
    if not asset_info:
        print(f"Error: Unknown asset: '{asset}'.")
        list_assets(asset_type='all')
        return False

    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / asset_info['directory']

    if local_path.exists() and not force:
        print(f"Asset '{asset}' is already installed at {local_path}.")
        print("Use force=True or --force to reinstall.")
        return True

    if local_path.exists() and force:
        print(f"Force option specified. Removing existing version of '{asset}'...")
        uninstall_asset(asset, quiet=True, local_dir=local_dir)

    print(f"Installing asset '{asset}' to {local_path}.")
    try:
        for file in asset_info['files']:

            hf_hub_download(
                repo_id=asset_info['repo_id'],
                filename=os.path.join(asset_info['directory'], file),
                repo_type="dataset",
                local_dir=local_dir,
            )
        print(f"Successfully installed '{asset}'.")
        return True
    except HfHubHTTPError as e:
        print(f"Error downloading asset '{asset}': {e}")
        # Clean up partial download
        if local_dir.exists():
            shutil.rmtree(local_dir)
        return False


def uninstall_asset(asset: str, quiet: bool = False, local_dir: Path | str | None = None) -> bool:
    """
    Removes a locally installed asset.
    """
    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)

    manifest = fetch_manifest()
    if not manifest:
        return False

    asset_info = manifest.get(asset)
    if not asset_info:
        list_assets(asset_type='all', installed_only=True)
        raise ValueError(f"Error: Unknown asset: '{asset}'. See above for installed assets.")

    local_path = local_dir / asset_info['directory']

    if not local_path.exists():
        if not quiet:
            print(f"Asset '{asset}' is not installed.")
        return True

    try:
        shutil.rmtree(local_path)
        if not quiet:
            print(f"Successfully removed '{asset}'.")
        return True
    except OSError as e:
        if not quiet:
            print(f"Error removing '{asset}': {e}")
        return False


def list_assets(asset_type: AssetType, installed_only: bool = False, local_dir: Path | str | None = None) -> None:
    """
    Lists available or installed assets.
    """
    manifest = fetch_manifest()
    if not manifest:
        return

    print(f"--- {'Installed' if installed_only else 'Available'} Assets ---")

    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)

    found_any = False
    for name, info in manifest.items():
        if asset_type != 'all' and info.get('type') != asset_type:
            continue
        local_path = local_dir / info['directory']
        is_installed = local_path.exists()

        if installed_only and not is_installed:
            continue

        status = "[installed]" if is_installed else ""
        print(f"- {name:<15} {status:<12} {info['description']}")
        found_any = True

    if not found_any:
        print(f"No {asset_type}s found.")
