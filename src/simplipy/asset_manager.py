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

AssetType = Literal['engine', 'test-data', 'all']

ASSET_KEYS = {
    'engine': 'engines',
    'test-data': 'test-data'
}


# --- Core Functions ---


def get_default_cache_dir() -> Path:
    """Get the default OS-appropriate cache directory for SimpliPy assets.

    This function determines the standard cache location based on the user's
    operating system, following the XDG Base Directory Specification on Linux.
    It ensures the directory exists, creating it if necessary.

    Returns
    -------
    pathlib.Path
        The path to the cache directory.

    """
    cache_dir = Path(platformdirs.user_cache_dir(appname="simplipy"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fetch_manifest(repo_id: str | None = None, manifest_filename: str | None = None) -> dict:
    """Download the latest asset manifest from Hugging Face Hub.

    The manifest is a JSON file that contains metadata for all official
    assets, including engines and test data. This function handles
    potential network errors gracefully.

    Parameters
    ----------
    repo_id : str, optional
        The Hugging Face repository ID where the manifest is stored. If None, the default repository ID is used.
    manifest_filename : str, optional
        The filename of the manifest file. If None, the default filename is used.

    Returns
    -------
    dict
        The parsed JSON manifest as a dictionary. Returns an empty dictionary
        if the download fails.

    """
    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id or HF_MANIFEST_REPO,
            filename=manifest_filename or HF_MANIFEST_FILENAME,
            repo_type="dataset",
        )
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except HfHubHTTPError as e:
        print(f"Error: Could not download the asset manifest from Hugging Face: {e}")
        return {}


def get_path(asset: str, install: bool = False, local_dir: Path | str | None = None, repo_id: str | None = None, manifest_filename: str | None = None) -> str | None:
    """Resolve the local filesystem path to an asset's entrypoint file.

    This function serves as a universal resolver for SimpliPy assets. It first
    checks if the `asset` string is a valid local path. If not, it treats it
    as an official asset name and looks it up in the manifest.

    Parameters
    ----------
    asset : str
        The identifier for the asset. This can be a direct path to a local
        file (e.g., './my_rules.yaml') or the name of an official asset
        (e.g., 'core-rules-v1').
    install : bool, optional
        If True, automatically downloads and installs the asset from
        Hugging Face Hub if it is not found locally. Defaults to False.
    local_dir : pathlib.Path | str | None, optional
        The directory to check for the asset or install it into. If None,
        the default cache directory is used. Defaults to None.
    repo_id : str, optional
        The Hugging Face repository ID where the manifest is stored. If None, the default repository ID is used.
    manifest_filename : str, optional
        The filename of the manifest file. If None, the default filename is used.

    Returns
    -------
    str
        The absolute path to the asset's entrypoint file.

    Raises
    ------
    RuntimeError
        If the asset manifest cannot be fetched from Hugging Face Hub or if
        the installation fails when `install=True`.
    ValueError
        If `asset` is not a local path and is not a known asset name in the
        manifest.
    FileNotFoundError
        If the asset is not found locally and `install` is False.

    """
    if not asset or not isinstance(asset, str):
        raise ValueError("Error: 'asset' must be a non-empty string.")

    # Check if 'asset' is a valid local path
    if Path(asset).exists():
        return asset

    # Otherwise, treat 'asset' as an official asset name
    manifest = fetch_manifest(repo_id=repo_id, manifest_filename=manifest_filename)
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
        if install_asset(asset, local_dir=local_dir, repo_id=repo_id, manifest_filename=manifest_filename):
            return str(entrypoint_path)
        else:
            raise RuntimeError(f"Failed to install asset '{asset}'.")

    raise FileNotFoundError(f"Asset '{asset}' is not installed. Use install=True to download it.")


def install_asset(asset: str, force: bool = False, local_dir: Path | str | None = None, repo_id: str | None = None, manifest_filename: str | None = None) -> bool:
    """Install a SimpliPy asset from Hugging Face Hub.

    Downloads all files associated with a given asset from its corresponding
    Hugging Face repository and places them in the specified local directory.

    Parameters
    ----------
    asset : str
        The name of the official asset to install.
    force : bool, optional
        If True, any existing local version of the asset will be removed
        before the new version is installed. Defaults to False.
    local_dir : pathlib.Path | str | None, optional
        The directory to install the asset into. If None, the default cache
        directory is used. Defaults to None.
    repo_id : str, optional
        The Hugging Face repository ID where the manifest is stored. If None, the default repository ID is used.
    manifest_filename : str, optional
        The filename of the manifest file. If None, the default filename is used.

    Returns
    -------
    bool
        True if the installation was successful or if the asset was already
        installed. False if the asset name is unknown or a download error
        occurs.

    """
    manifest = fetch_manifest(repo_id=repo_id, manifest_filename=manifest_filename)
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


def uninstall_asset(asset: str, quiet: bool = False, local_dir: Path | str | None = None, repo_id: str | None = None, manifest_filename: str | None = None) -> bool:
    """Remove a locally installed SimpliPy asset.

    This function deletes the entire directory associated with the specified
    asset from the local filesystem.

    Parameters
    ----------
    asset : str
        The name of the asset to uninstall.
    quiet : bool, optional
        If True, suppresses console output messages. Defaults to False.
    local_dir : pathlib.Path | str | None, optional
        The directory from which to uninstall the asset. If None, the
        default cache directory is used. Defaults to None.
    repo_id : str, optional
        The Hugging Face repository ID where the manifest is stored. If None, the default repository ID is used.
    manifest_filename : str, optional
        The filename of the manifest file. If None, the default filename is used.

    Returns
    -------
    bool
        True if the asset was successfully removed or was not installed to
        begin with. False if an OS error occurs during removal.

    Raises
    ------
    ValueError
        If `asset` is not a known asset name in the manifest.

    """
    if local_dir is None:
        local_dir = get_default_cache_dir()
    elif isinstance(local_dir, str):
        local_dir = Path(local_dir)

    manifest = fetch_manifest(repo_id=repo_id, manifest_filename=manifest_filename)
    if manifest:
        asset_info = manifest.get(asset)
        if not asset_info:
            list_assets(asset_type='all', installed_only=True)
            raise ValueError(f"Error: Unknown asset: '{asset}'. See above for installed assets.")

        local_path = local_dir / asset_info['directory']
    else:
        local_path = local_dir / asset

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


def list_assets(asset_type: AssetType, installed_only: bool = False, local_dir: Path | str | None = None, repo_id: str | None = None, manifest_filename: str | None = None) -> None:
    """List available or installed SimpliPy assets.

    Fetches the asset manifest and checks the local filesystem to print a
    formatted list of assets, their descriptions, and their installation
    status to standard output.

    Parameters
    ----------
    asset_type : {'engine', 'test-data', 'all'}
        The category of assets to list.
    installed_only : bool, optional
        If True, the list is filtered to show only assets that are currently
        installed locally. Defaults to False.
    local_dir : pathlib.Path | str | None, optional
        The directory to check for installed assets. If None, the default
        cache directory is used. Defaults to None.
    repo_id : str, optional
        The Hugging Face repository ID where the manifest is stored. If None, the default repository ID is used.
    manifest_filename : str, optional
        The filename of the manifest file. If None, the default filename is used.

    """
    manifest = fetch_manifest(repo_id=repo_id, manifest_filename=manifest_filename)
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
